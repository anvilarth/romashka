import random
import numpy as np
import inspect
from typing import List, Optional, Tuple, Any, Dict, Union

import wandb
import torch
import torch.nn as nn

import transformers
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import WandbLogger

import bitsandbytes as bnb
from  romashka.transactions_qa.lion_optim import Lion
from romashka.transactions_qa.utils import inspect_init_signature
from romashka.transactions_qa.model.generation_utils import AnsweredQACriteria
from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.transactions_qa.tasks.task_token_updater import (collect_task_specific_tokens,
                                                               create_task_specific_tokens_map)
from romashka.logging_handler import get_logger

from transformers import GenerationConfig
from copy import deepcopy


class TransactionQAModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 tasks: List[AbstractTask],
                 learning_rate: Optional[float] = 5e-5,
                 optimizer_type: Optional[Union[torch.optim.Optimizer, str]] = "AdamW",
                 scheduler_type: Optional[Union[transformers.SchedulerType, str]] = "linear",
                 use_8bit_optim: Optional[bool] = False,
                 adam_beta1: Optional[float] = 0.9,
                 adam_beta2: Optional[float] = 0.999,
                 adam_epsilon: Optional[float] = 1e-8,
                 warmup_steps: Optional[int] = 100,
                 training_steps: Optional[int] = 10_000,
                 verbose_for_debug: Optional[bool] = False,
                 return_logits: Optional[bool] = False,
                 **additional_kwargs
                ):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.model = model
        self.tasks = tasks
        self.task_specific_tokens = collect_task_specific_tokens(self.tasks)
        self.task_specific_tokens_map = create_task_specific_tokens_map(self.model.tokenizer)

        self.metrics = nn.ModuleDict({task.task_name: deepcopy(task.metrics) for task in self.tasks})

        self.warmup_steps: int = warmup_steps
        self.training_steps: int = training_steps
        self.base_learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.use_8bit_optim = use_8bit_optim
        self.adam_beta1: float = adam_beta1
        self.adam_beta2: float = adam_beta2
        self.adam_epsilon = adam_epsilon

        self._is_multitask: bool = False
        self._is_encoder_decoder: bool = False

        self._verbose_for_debug: bool = verbose_for_debug
        self._return_logits: bool = return_logits

        self.hparams['task_specific_tokens_map'] = self.task_specific_tokens_map
        self.save_hyperparameters(ignore=['tasks', '_logger', 'model'])

        # ✨ W&B: Create a Table to store predictions for each test step
        # self.columns = ["epoch", "step #", "task",
        #                 "question", "prediction", "truth",
        #                 "transactions_history_lengths"]
        # self.log_eval_predictions_table = wandb.Table(columns=self.columns)
        # self.log_eval_steps_counter = 0
        # self.num_eval_batches_to_log = -1

    def configure_optimizers(self):
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Returns:
             **Dictionary**, with an "optimizer" key, and (optionally) a "lr_scheduler"
              key whose value is a single LR scheduler or `lr_scheduler_config`.
        """
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))

        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        # Init optimizer
        optimizer_type = None
        if isinstance(self.optimizer_type, torch.optim.Optimizer):
            optimizer_type = self.optimizer_type
            rank_zero_info(
                f"The model will train with {optimizer_type} optimizer."
            )
        elif self.optimizer_type in ["Adam", "AdamW"] and self.use_8bit_optim:
            optimizer_type = 'Adam8bit'
            optimizer = bnb.optim.Adam8bit(self.parameters(),
                                           lr=self.base_learning_rate,
                                           betas=(self.adam_beta1, self.adam_beta2))
            rank_zero_info(
                f"The model will train with 8-bit AdamW optimizer."
            )
        elif hasattr(torch.optim, self.optimizer_type):
            optimizer_type = getattr(torch.optim, self.optimizer_type)
            optim_params = {
                "params": self.parameters(),
                "lr": self.base_learning_rate
            }
            # Add beta1, beta2 for Adam-like optimizers
            if inspect_init_signature('betas', optimizer_type):
                optim_params['betas'] = (self.adam_beta1, self.adam_beta2)

            # Instantiate optimizer
            optimizer = optimizer_type(**optim_params)
            rank_zero_info(
                f"Instantiating {self.optimizer_type} optimizer"
            )
        elif self.optimizer_type == 'Lion':
            optimizer = Lion(self.parameters(),
                             betas=(self.adam_beta1, self.adam_beta2),
                             lr=self.base_learning_rate)
            rank_zero_info(
                f"Instantiating {self.optimizer_type} optimizer"
            )
        else:
            rank_zero_info(f"Unknown {optimizer_type} optimizer, so create AdamW optimizer.")
            optimizer = torch.optim.AdamW(self.parameters(),
                                          betas=(self.adam_beta1, self.adam_beta2),
                                          lr=self.base_learning_rate)
        # Select scheduler
        scheduler = transformers.get_scheduler(name=self.scheduler_type,
                                               optimizer=optimizer,
                                               num_warmup_steps=self.warmup_steps,
                                               num_training_steps=self.training_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def _prepare_model(self):
        if not len(self.tasks):
            raise AttributeError(f"For training at least one task should be specified!")
        elif len(self.tasks) > 1:
            self._is_multitask = True
            self._logger.info(f"Running in `multi task` setting with {len(self.tasks)} tasks provided.")
        else:
            self._is_multitask = False
            self._logger.info(f"Running in `single task` setting"
                              f"with a single task: {self.tasks[0].task_name} provided.")

        # Figure out what the model type passed% encoder-decoder / decoder-only
        self._set_model_type()

    def _set_model_type(self):
        # For encoder-decoder models
        if hasattr(self.model.language_model, "encoder"):
            self._is_encoder_decoder = True
        # For decoder-only
        elif hasattr(self.model.language_model, "transformer"):
            self._is_encoder_decoder = False
        else:
            raise NotImplementedError(f"Unknown model type: {type(self.model.language_model)}")

        self._logger.info(f"Language model type: `{'encoder-decoder' if self._is_encoder_decoder else 'decoder'}`")

    def add_task(self, new_task: AbstractTask):
        """
        Add new task to existing in model.
        Args:
            new_task: a new task instance;
        """
        self.tasks.append(new_task)

    def model_step(self, batch,
                   task_idx: Optional[int] = None,
                   generate: Optional[bool] = False,
                   generation_options: Optional[Dict[str, Any]] = None,
                   output_attentions: Optional[bool] = False) -> Tuple[Any, torch.Tensor]:

        # Sample single task
        if task_idx is None:
            task_idx = 0
        task = self.tasks[task_idx]
        qa_batch = task.process_input_batch(batch)

        if len(qa_batch) == 0:
            return None, None

        # Pass through inner model
        if generate:
            generation_config = GenerationConfig(**generation_options)
            return self.model.generate(qa_batch, generation_config)
        else:
            # join two batches
            for key, val in batch.items():
                qa_batch[key] = val
            outputs = self.model(qa_batch, output_attentions=output_attentions)
            batch_answers = qa_batch['answer_tokens'] if "answer_tokens" not in outputs else outputs['answer_tokens']

            return outputs, batch_answers

    def training_step(self, batch, batch_idx: Optional[int] = None) -> Optional[Any]:
        r"""
        Compute and return the training loss and some additional metrics for e.g.
        the progress bar or logger.

        Args:
            batch: The output of torch.utils.data.DataLoader. A tensor, tuple or list.
            batch_idx (`int`): Integer displaying index of this batch

        Return:
            Any of.
        Note:
            The loss value shown in the progress bar is smoothed (averaged) over the last values,
            so it differs from the actual loss returned in train/validation step.
        """
        # Sample a random single task
        task_idx = random.sample(list(range(len(self.tasks))), k=1)[0]
        task_name = self.tasks[task_idx].task_name

        outputs, answer = self.model_step(batch, task_idx=task_idx)
        if outputs is None:
            return None

        loss = outputs['loss']
        logging_dict = {
            'train_loss': loss,
            f'{task_name}_train_loss': loss,
        }
        # Log additional loss values
        for k in outputs:
            if k.endswith("loss"):
                logging_dict[f"train_{k}"] = outputs[k]
        if self._verbose_for_debug:
            additional_logging_dict = self._collect_additional_info(outputs)
            if additional_logging_dict is not None and len(additional_logging_dict):
                logging_dict = dict(list(logging_dict.items()) + list(additional_logging_dict.items()))

        self.log_dict(logging_dict)
        return loss

    def _collect_additional_info(self, outputs: Any) -> Dict[str, Any]:
        """
        Collect additional information from model's outputs for debugging.
        Calling this function is optional - it can be removed without ane troubles.
        Args:
            outputs: model's step outputs;
        Returns:
            a collected dictionary with information about a step.
        """
        logging_dict = {}
        if 'question_start_input_size' in outputs:
            logging_dict['question_start_input_size'] = outputs['question_start_input_size']

        if 'question_end_input_size' in outputs:
            logging_dict['question_end_input_size'] = outputs['question_end_input_size']

        if 'transactions_input_size' in outputs:
            logging_dict['transactions_input_size'] = outputs['transactions_input_size']

        if 'total_input_size' in outputs:
            logging_dict['total_input_size'] = outputs['total_input_size']

        # if 'transactions_history_lengths' in outputs:
        #     logging_dict['transactions_history_lengths'] = outputs['transactions_history_lengths'].detach()

        if self._verbose_for_debug:
            print(f"Additional info from a step:")
            print(logging_dict)
        return logging_dict

    def validation_step(self, batch,
                        batch_idx: Optional[int],
                        dataloader_idx: Optional[int] = None, **kwargs) -> Optional[Any]:
        r"""
        Operates on a single batch of data from the validation set.
        Args:
            batch: The output of your torch.utils.data.DataLoader.
            batch_idx: The index of this batch.
            dataloader_idx: The index of the dataloader that produced this batch.
                (only if multiple val dataloaders used)

        Return:
            - Any object or value
        """
        # Sample a random single task
        task_idx = random.sample(list(range(len(self.tasks))), k=1)[0]
        task = self.tasks[task_idx]

        outputs, batch_answers = self.model_step(batch, task_idx=task_idx)

        if outputs is None:
            return None

        loss = outputs['loss']

        # Calc metrics
        metrics_scores = {}
        try:
            metrics_scores = task.calculate_metrics(outputs, batch_answers, self.metrics[task.task_name])
            metrics_scores = {metric_name + "_" + task.task_name: score for metric_name, score in metrics_scores.items()}
        except Exception as e:
            self._logger.error(f"error occurred during task metric calculation:\n{e}")

        logging_dict = {
            'val_loss': loss,
            f'{task.task_name}_val_loss': loss
        }
        # Log additional loss values
        for k in outputs:
            if k.endswith("loss"):
                logging_dict[f"val_{k}"] = outputs[k]

        if self._verbose_for_debug:
            additional_logging_dict = self._collect_additional_info(outputs)
            if additional_logging_dict is not None and len(additional_logging_dict):
                logging_dict = dict(list(logging_dict.items()) + list(additional_logging_dict.items()))

        logging_dict = dict(list(logging_dict.items()) + list(metrics_scores.items()))

        self.log_dict(
            logging_dict,
            batch_size=batch_answers.size(0),
            sync_dist=True,
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True
        )

        # Log predictions on validation set
        # if self.num_eval_batches_to_log == -1:  # log all validation data
        #     questions = outputs['question_encoded'].detach().cpu() if hasattr(outputs, "question_encoded") \
        #     or ("question_encoded" in outputs.keys()) else ""
        #     transactions_history_lengths = outputs['transactions_history_lengths'].detach().cpu() \
        #         if hasattr(outputs, "transactions_history_lengths") else [0]
        #     self.log_predictions(logits=outputs['logits'].detach().cpu(),
        #                          answers=batch_answers.detach().cpu(),
        #                          questions=questions,
        #                          transactions_history_lengths=transactions_history_lengths,
        #                          predictions_table=self.log_eval_predictions_table,
        #                          log_counter=self.log_eval_steps_counter,
        #                          task_name=task.task_name)
        #     self.log_eval_steps_counter += 1
        # elif self.log_eval_steps_counter < self.num_eval_batches_to_log:
        #     questions = outputs['question_encoded'].detach().cpu() if hasattr(outputs, "question_encoded") \
        #     or ("question_encoded" in outputs.keys()) else ""
        #     transactions_history_lengths = outputs['transactions_history_lengths'].detach().cpu() \
        #         if hasattr(outputs, "transactions_history_lengths") else [0]
        #     self.log_predictions(logits=outputs['logits'].detach().cpu(),
        #                          answers=batch_answers.detach().cpu(),
        #                          questions=questions,
        #                          transactions_history_lengths=transactions_history_lengths,
        #                          predictions_table=self.log_eval_predictions_table,
        #                          log_counter=self.log_eval_steps_counter,
        #                          task_name=task.task_name)
        #     self.log_eval_steps_counter += 1

        return loss

    def predict_step(self,
                     batch: Any, batch_idx: int,
                     dataloader_idx: int = 0,
                     verbose: Optional[bool] = False) -> Any:
        """
        Step function called during Trainer.predict().

        TODO: to use pytorch_lightning.callbacks.BasePredictionWriter callback
        to write the predictions to disk or database after each batch or on epoch end.

        Args:
            batch: Current batch.
            batch_idx: Index of current batch.
            dataloader_idx: Index of the current dataloader.
        Return:
            Predicted output
        """
        # Predict for each task !!!
        # Calc metrics
        tasks_predictions = {}
        for task_idx, task in enumerate(self.tasks):
            try:
                # For encoder-decoder models make a step with a model and get answers from outputs
                if self._is_encoder_decoder:
                    predictions = self._predict_step_task(batch,
                                                          batch_idx=batch_idx,
                                                          task_idx=task_idx,
                                                          verbose=verbose,
                                                          calculate_metrics=False)
                else:
                    # For decoder-only models run generate() on questions
                    predictions = self._predict_with_generate_step_task(batch,
                                                                        batch_idx=batch_idx,
                                                                        task_idx=task_idx,
                                                                        verbose=verbose,
                                                                        calculate_metrics=False,
                                                                        return_embeddings=False,
                                                                        return_logits=False)

                tasks_predictions[task.task_name] = predictions
            except Exception as e:
                self._logger.error(f"Error occurred during task `{task.task_name}` evaluation:\n{e}")

        return tasks_predictions

    def _predict_step_task(self, batch: Any, task_idx: int,
                           calculate_metrics: Optional[bool] = False,
                           verbose: Optional[bool] = False,
                           batch_idx: Optional[int] = 0, **kwargs) -> Dict[str, Any]:
        """
        Predict for single task.
        Args:
            batch: Current batch;
            task_idx: selected task index;
            batch_idx: Index of current batch.

        Returns:
            results: as dictionary, where:
                keys are - metrics / predictions / answers / questions.
        """
        task = self.tasks[task_idx]
        outputs, batch_answers = self.model_step(batch, task_idx=task_idx)

        if outputs is None:
            return dict()

        # as list of strings
        predictions_decoded = self.model.tokenizer.batch_decode(
            outputs['logits'].argmax(2) if isinstance(outputs, dict) else outputs.logits.argmax(2),
            skip_special_tokens=True)
        batch_answers_decoded = self.model.tokenizer.batch_decode(batch_answers,
                                                            skip_special_tokens=True)
        batch_questions_decoded = self.model.tokenizer.batch_decode(
             outputs['question_encoded'].detach().cpu() if isinstance(outputs, dict) else outputs.question_encoded.detach().cpu(),
            skip_special_tokens=True)

        if verbose:
            print("----- Prediction step -----")
            for i, (pred, answer, question) in enumerate(
                    zip(predictions_decoded, batch_answers_decoded, batch_questions_decoded)):
                print(f"\t#{i} {question}:\n\tpredicted: {pred},\n\tanswer: {answer}")

        pred_output = dict(
            predictions=predictions_decoded,
            answers=batch_answers_decoded,
            questions=batch_questions_decoded,
            batch_idx=batch_idx
        )

        # Calc metrics
        if calculate_metrics:
            metrics_scores = {}
            for metric_name, metric in task.metrics.items():
                try:
                    metrics_scores[metric_name] = metric(predictions_decoded,
                                                         batch_answers_decoded)
                except Exception as e:
                    self._logger.error(f"error occurred during task metric `{metric_name}` calculation:\n{e}")

            pred_output['metrics'] = metrics_scores

        if self._return_logits:
            pred_output['logits'] = outputs['logits']

        return pred_output

    def _predict_with_generate_step_task(self, batch: Any, task_idx: int,
                                         calculate_metrics: Optional[bool] = False,
                                         verbose: Optional[bool] = False,
                                         batch_idx: Optional[int] = 0,
                                         return_embeddings: Optional[bool] = False,
                                         return_logits: Optional[bool] = False) -> Dict[str, Any]:
        """
        Predict with generate() function for single task.
        Args:
            batch: Current batch;
            task_idx: selected task index;
            batch_idx: Index of current batch.

        Returns:
            results: as dictionary, where:
                keys are - metrics / predictions / answers / questions.
        """
        task = self.tasks[task_idx]
        qa_batch = task.process_input_batch(batch)
        batch_answers = qa_batch['answer_tokens'].long() if "answer_tokens" in qa_batch else None

        # Get parameters for generation from model
        generation_config = self.model.generation_config

        # Whether to restrict outputs of generation to small subspace of vocabulary
        if generation_config.get("create_allowed_token_ids", False) \
                and (generation_config.get("allowed_token_ids") is None):
            if 'binary' in task.task_name:
                answer_options = list(task.binary_answer_options.values())
            else:
                answer_options = list(task.answers_options)
            allowed_token_ids = [i[0] for i in
                                 self.model.tokenizer(answer_options, add_special_tokens=False).input_ids]
        elif generation_config.get("allowed_token_ids") is not None:
            allowed_token_ids = generation_config.get("allowed_token_ids")
        else:
            allowed_token_ids = None

        # Whether to create create_stopping_criteria
        stopping_criteria = None
        if generation_config.get("create_stopping_criteria", False) and (allowed_token_ids is not None):
            stopping_criteria = AnsweredQACriteria(prompt_length=0,
                                                   answer_tokens_ids=allowed_token_ids)
        elif generation_config.get("create_stopping_criteria", False):
            self._logger.warning(f"Unable to create stopping criteria, no `allowed_token_ids` provided!")

        predictions = self.model.generate(
            questions=qa_batch['question_end_tokens'],
            transactions_batch=batch,
            prefix_prompt=qa_batch['question_start_tokens'],
            answer_template=task.answer_template,
            max_new_tokens=generation_config.get("max_new_tokens", False),
            min_new_tokens=generation_config.get("min_new_tokens", False),
            top_p=generation_config.get("top_p", False),
            temperature=generation_config.get("temperature", False),
            filter_value=generation_config.get("filter_value", False),
            allowed_token_ids=allowed_token_ids,
            hidden_dims_indexes=generation_config.get("hidden_dims_indexes", False),
            stopping_criteria=stopping_criteria,
            seed=generation_config.get("seed", False)
        )

        # as list of strings
        predictions_decoded = self.model.tokenizer.batch_decode(predictions['generated_texts'],
                                                                skip_special_tokens=True)
        batch_answers_decoded = self.model.tokenizer.batch_decode(batch_answers,
                                                                  skip_special_tokens=True) \
            if batch_answers is not None else None
        batch_questions_decoded = self.model.tokenizer.batch_decode(qa_batch['question_end_tokens'].detach().cpu(),
                                                                    skip_special_tokens=True)

        if verbose:
            self._logger.info("----- Prediction step -----")
            self._logger.info(f"Generated for batch #{batch_idx}:\n{predictions_decoded}")

        # Calc metrics
        metrics_scores = {}
        if calculate_metrics:
            for metric_name, metric in task.metrics.items():
                try:
                    metrics_scores[metric_name] = metric(predictions_decoded,
                                                         batch_answers_decoded)
                except Exception as e:
                    self._logger.error(f"error occurred during task metric `{metric_name}` calculation:\n{e}")

        outputs = {
            "predictions": predictions_decoded,
            "questions": batch_questions_decoded,
            "metrics": metrics_scores,
            "batch_idx": batch_idx
        }
        if batch_answers is not None:
            outputs['answers'] = batch_answers_decoded
        if return_embeddings:
            outputs['embeddings'] = predictions['output_embeddings']
        if return_logits:
            outputs['logits'] = predictions['output_logits']
        return outputs

    def on_validation_epoch_start(self) -> None:
        print(f"\n----------- Validation start ----------\n")
    #     # Reset log counter
    #     self.log_eval_steps_counter = 0

    def on_validation_epoch_end(self) -> None:
        # ✨ W&B: Log predictions table to wandb
        print(f"\n----------- Validation end ----------\n")
        # print(f"Using logger: {self.logger.experiment}")
        # wandb_logger = [logger for logger in self.trainer.loggers if isinstance(logger, WandbLogger)][0]
        # self.logger.log_metrics({"val_predictions": self.log_eval_predictions_table})

        # was directly to W&B: wandb.log({"val_predictions": self.log_eval_predictions_table})
        # ✨ W&B: Mark the run as complete (useful for multi-cell notebook)
        # wandb.finish()

    def on_before_optimizer_step(self, optimizer, optimizer_idx = 0):
        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 25 == 0:  # don't make logging too much
            for param_name, param in self.model.named_parameters():
                if param_name.startswith("transactions_start_embedding") \
                        or param_name.startswith("transactions_end_embedding") \
                        or param_name.startswith("ret_embedding") \
                        or param_name.startswith("projection_layers")\
                        or param_name.startswith("connector.query_tokens_embeddings"):
                    if param.grad is not None:
                        grad_sum = np.sum(np.abs(param.grad.detach().cpu().numpy()))
                        self._logger.info(f"Parameter `{param_name}` with grad of size: {param.grad.size()}")
                        self._logger.info(f"Summed `{param_name}` grad = {grad_sum}")
                        self.log(
                            name=f"{param_name}_grad_sum", value=grad_sum, sync_dist=True
                        )

    def log_predictions(self,
                        logits: torch.Tensor,
                        answers: torch.Tensor,
                        questions: torch.Tensor,
                        predictions_table: wandb.Table,
                        log_counter: int,
                        transactions_history_lengths: Optional[torch.Tensor] = [],
                        task_name: Optional[str] = "default"):
        predictions_decoded = self.model.tokenizer.batch_decode(logits.argmax(2).long(),
                                                                skip_special_tokens=True)
        answers_decoded = self.model.tokenizer.batch_decode(answers.long(),
                                                            skip_special_tokens=True)
        questions_decoded = self.model.tokenizer.batch_decode(questions.long(),
                                                              skip_special_tokens=True)

        print(f"Validation predictions vs. answers, batch #{log_counter}:")

        # columns = ["epoch", "step #", "task", "question", "prediction", "truth", "transactions_history_lengths"]
        for i, (pred, answer, question) in enumerate(zip(predictions_decoded, answers_decoded, questions_decoded)):
            print(f"\t#{i}:\tpredicted: {pred},\n\tanswer: {answer}")
            predictions_table.add_data(self.current_epoch,
                                       "_".join([str(log_counter), str(i)]),
                                       task_name,
                                       question,
                                       pred,
                                       answer,
                                       transactions_history_lengths[i] if i < len(transactions_history_lengths) else 0)
