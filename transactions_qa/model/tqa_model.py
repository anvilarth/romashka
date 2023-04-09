import random
import numpy as np
from typing import List, Optional, Tuple, Any, Dict, Union

import wandb
import torch
import torch.nn as nn

import transformers
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from romashka.transactions_qa.tasks.task_abstract import AbstractTask
from romashka.logging_handler import get_logger

from transformers import GenerationConfig
from copy import deepcopy


class TransactionQAModel(pl.LightningModule):
    def __init__(self,
                 model: nn.Module,
                 tasks: List[AbstractTask],
                 learning_rate: Optional[float] = 5e-5,
                 adam_beta1: Optional[float] = 0.9,
                 adam_beta2: Optional[float] = 0.999,
                 adam_epsilon: Optional[float] = 1e-8,
                 warmup_steps: Optional[int] = 100,
                 training_steps: Optional[int] = 10_000,
                 verbose_for_debug: Optional[bool] = False):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.model = model
        self.tasks = tasks

        self.metrics = nn.ModuleDict({task.task_name: deepcopy(task.metrics) for task in self.tasks})

        self.warmup_steps: int = warmup_steps
        self.training_steps: int = training_steps
        self.base_learning_rate = learning_rate
        self.adam_beta1: float = adam_beta1
        self.adam_beta2: float = adam_beta2
        self.adam_epsilon = adam_epsilon

        self._is_multitask: bool = False
        self._is_encoder_decoder: bool = False

        self._verbose_for_debug: bool = verbose_for_debug

        self.save_hyperparameters(ignore=['tasks', '_logger', 'columns', 'model',
                                          'log_eval_predictions_table', 'log_eval_steps_counter'])

        # ✨ W&B: Create a Table to store predictions for each test step
        self.columns = ["epoch", "step #", "task",
                        "question", "prediction", "truth",
                        "transactions_history_lengths"]
        self.log_eval_predictions_table = wandb.Table(columns=self.columns)
        self.log_eval_steps_counter = 0
        self.num_eval_batches_to_log = 10

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
        optimizer = torch.optim.AdamW(self.parameters(),
                                      betas=(self.adam_beta1, self.adam_beta2),
                                      lr=self.base_learning_rate)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=self.warmup_steps,
                                                                 num_training_steps=self.training_steps
                                                                 )  # was: 10_000 * 20
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
        if hasattr(self.language_model, "encoder"):
            self._is_encoder_decoder = True
        # For decoder-only
        elif hasattr(self.language_model, "transformer"):
            self._is_encoder_decoder = False
        else:
            raise NotImplementedError(f"Unknown model type: {type(self.language_model)}")

        self._logger.info(f"Language model type: `{'encoder-decoder' if self._is_encoder_decoder else 'decoder'}`")

    def model_step(self, batch,
                   task_idx: Optional[int] = None,
                   generate: Optional[bool] = False,
                   generation_options: Optional[Dict[str, Any]] = None) -> Tuple[Any, torch.Tensor]:

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
            # contains: ???
            outputs = self.model(qa_batch)

            # Create answers + masks for LM's decoder inputs
            batch_answers = qa_batch['answer_tokens']

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

        loss = outputs.loss

        logging_dict = {
            'train_loss': loss,
            f'{task_name}_train_loss': loss,
        }
        if self._verbose_for_debug:
            additional_logging_dict = self._collect_additional_info(outputs)
            if additional_logging_dict is not None and len(additional_logging_dict):
                logging_dict = dict(list(logging_dict.items()) + list(additional_logging_dict.items()))

        self.log_dict(logging_dict)
        # self._logger.info(f"Train step results:\n{logging_dict}")
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

        loss = outputs.loss

        # Calc metrics
        try:
            metrics_scores = task.calculate_metrics(outputs, batch_answers, self.metrics[task.task_name])
        except Exception as e:
            self._logger.error(f"error occurred during task metric calculation:\n{e}")

        logging_dict = {
            'val_loss': loss,
            f'{task.task_name}_val_loss': loss
        }
        if self._verbose_for_debug:
            additional_logging_dict = self._collect_additional_info(outputs)
            if additional_logging_dict is not None and len(additional_logging_dict):
                logging_dict = dict(list(logging_dict.items()) + list(additional_logging_dict.items()))

        logging_dict = dict(list(logging_dict.items()) + list(metrics_scores.items()))

        self.log_dict(
            logging_dict,
            batch_size=batch_answers.size(0)
        )

        # Log predictions on validation set
        if self.log_eval_steps_counter < self.num_eval_batches_to_log:
            self.log_predictions(logits=outputs.logits.detach().cpu(),
                                 answers=batch_answers.detach().cpu(),
                                 questions=outputs.question_encoded.detach().cpu(),
                                 transactions_history_lengths=outputs['transactions_history_lengths'].detach().cpu(),
                                 predictions_table=self.log_eval_predictions_table,
                                 log_counter=self.log_eval_steps_counter)
            self.log_eval_steps_counter += 1
        return loss

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
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
                predictions = self._predict_step_task(batch,
                                                      batch_idx=batch_idx,
                                                      task_idx=task_idx,
                                                      verbose=True,
                                                      calculate_metrics=False)

                tasks_predictions[task.task_name] = predictions
            except Exception as e:
                self._logger.error(f"Error occurred during task `{task.task_name}` evaluation:\n{e}")

        return tasks_predictions

    def _predict_step_task(self, batch: Any, task_idx: int,
                           calculate_metrics: Optional[bool] = False,
                           verbose: Optional[bool] = False,
                           batch_idx: Optional[int] = 0) -> Dict[str, Any]:
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
        predictions_decoded = self.tokenizer.batch_decode(outputs.logits.argmax(2),
                                                          skip_special_tokens=True)
        batch_answers_decoded = self.tokenizer.batch_decode(batch_answers,
                                                            skip_special_tokens=True)
        batch_questions_decoded = self.tokenizer.batch_decode(outputs.question_encoded.detach().cpu(),
                                                              skip_special_tokens=True)

        if verbose:
            print("----- Prediction step -----")
            for i, (pred, answer, question) in enumerate(
                    zip(predictions_decoded, batch_answers_decoded, batch_questions_decoded)):
                print(f"\t#{i}{question}:\tpredicted: {pred}, answer: {answer}")

        # Calc metrics
        metrics_scores = {}
        if calculate_metrics:
            for metric_name, metric in task.metrics.items():
                try:
                    metrics_scores[metric_name] = metric(predictions_decoded,
                                                         batch_answers_decoded)
                except Exception as e:
                    self._logger.error(f"error occurred during task metric `{metric_name}` calculation:\n{e}")

        return dict(
            predictions=predictions_decoded,
            answers=batch_answers_decoded,
            questions=batch_questions_decoded,
            metrics=metrics_scores,
            batch_idx=batch_idx
        )

    def on_validation_epoch_start(self) -> None:
        print(f"\n----------- Validation start ----------\n")
        # Reset log counter
        self.log_eval_steps_counter = 0

    def on_fit_end(self) -> None:
        # ✨ W&B: Log predictions table to wandb
        wandb.log({"val_predictions": self.log_eval_predictions_table})
        # was directly to W&B: wandb.log({"val_predictions": self.log_eval_predictions_table})
        # ✨ W&B: Mark the run as complete (useful for multi-cell notebook)
        wandb.finish()

    def log_predictions(self,
                        logits: torch.Tensor,
                        answers: torch.Tensor,
                        questions: torch.Tensor,
                        predictions_table: wandb.Table,
                        log_counter: int,
                        transactions_history_lengths: Optional[torch.Tensor] = [],
                        task_name: Optional[str] = "default"):
        predictions_decoded = self.tokenizer.batch_decode(logits.argmax(2),
                                                          skip_special_tokens=True)
        answers_decoded = self.tokenizer.batch_decode(answers,
                                                      skip_special_tokens=True)
        questions_decoded = self.tokenizer.batch_decode(questions,
                                                        skip_special_tokens=True)

        print(f"Validation predictions vs. answers, batch #{log_counter}:")

        # columns = ["epoch", "step #", "task", "question", "prediction", "truth", "transactions_history_lengths"]
        for i, (pred, answer, question) in enumerate(zip(predictions_decoded, answers_decoded, questions_decoded)):
            print(f"\t#{i}:\tpredicted: {pred}, answer: {answer}")
            predictions_table.add_data(self.current_epoch,
                                       "_".join([str(log_counter), str(i)]),
                                       task_name,
                                       question,
                                       pred,
                                       answer,
                                       transactions_history_lengths[i] if len(transactions_history_lengths) else 0)