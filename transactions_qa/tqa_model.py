import random
import numpy as np
from typing import List, Optional, Tuple, Any, Dict, Union

import wandb
import torch
import torch.nn as nn

import transformers
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from tasks.task_abstract import AbstractTask
from ..logging_handler import get_logger


def make_linear_connector(output_size: Optional[int] = None,
                          input_size: Optional[int] = None,
                          embedding_model: Optional[nn.Module] = None,
                          autoregressive_model: Optional[nn.Module] = None,
                          device: Optional[Union[torch.device, str]] = 'cpu'):
    required_output_size = None
    if output_size is not None:
        required_output_size = output_size
    elif embedding_model is not None:
        try:
            # As it is custom model
            required_output_size = embedding_model.head.output_size
        except Exception as e0:
            try:
                # If it is HF model, then  take output dimensions from config
                required_output_size = embedding_model.config.d_model
            except Exception as e1:
                raise AttributeError(f"Cannot get `output_size` from embeddings model:\n{e0}\n{e1}")
    else:
        raise AttributeError(f"Unable to define `output_size` from embeddings model"
                             "as none of `output_size` or `embedding_model` is specified.")

    required_input_size = None
    if input_size is not None:
        required_input_size = input_size
    elif autoregressive_model is not None:
        try:
            # If it is HF model, then take inputs dimensions from config
            required_input_size = autoregressive_model.config.d_model
        except Exception as e:
            raise AttributeError(f"Cannot get `input_size` from autoregressive model:\n{e}")
    else:
        raise AttributeError(f"Unable to define `input_size` from autoregressive model"
                             "as none of `input_size` or `autoregressive_model` is specified.")

    print(f"Output dimension of embedding model: {required_output_size}")
    print(f"Input dimension of autoregressive model: {required_input_size}")
    print(f"Creating linear connector from {required_output_size} to {required_input_size}"
          f"and move to device: {device}.")

    return nn.Linear(required_output_size, required_input_size).to(device)


class TransactionQAModel(pl.LightningModule):
    def __init__(self,
                 language_model: nn.Module,
                 transaction_model: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 tasks: List[AbstractTask],
                 connector: Optional[nn.Module] = None,
                 connector_input_size: Optional[int] = None,
                 connector_output_size: Optional[int] = None,
                 do_freeze_tm: Optional[bool] = True,
                 do_freeze_lm: Optional[bool] = False,
                 do_freeze_connector: Optional[bool] = False,
                 num_days: Optional[int] = 7,
                 learning_rate: Optional[float] = 5e-5,
                 adam_beta1: Optional[float] = 0.9,
                 adam_beta2: Optional[float] = 0.999,
                 adam_epsilon: Optional[float] = 1e-8,
                 warmup_steps: Optional[int] = 100,
                 training_steps: Optional[int] = 10_000):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.transaction_model = transaction_model
        self.connector = make_linear_connector(
            input_size=connector_output_size,
            output_size=connector_input_size,
            embedding_model=self.transaction_model,
            autoregressive_model=self.language_model) \
            if connector is None else connector
        self.tasks = tasks
        self.warmup_steps: int = warmup_steps
        self.training_steps: int = training_steps
        self.base_learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon

        self.do_freeze_tm: bool = do_freeze_tm
        self.do_freeze_lm: bool = do_freeze_lm
        self.do_freeze_connector: bool = do_freeze_connector
        self._is_multitask: bool = False
        self._prepare_model()

        self.save_hyperparameters(ignore=['tasks', '_logger', 'columns',
                                          'transaction_model', 'language_model', 'connector',
                                          'log_eval_predictions_table', 'log_eval_steps_counter'])

        # ✨ W&B: Create a Table to store predictions for each test step
        self.columns = ["epoch", "step #", "task", "question", "prediction", "truth"]
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
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

        # In case any of tasks extends initial tokenizer vocab with additional tokens
        self._resize_text_embeddings()

        # Freezing some weights
        if self.do_freeze_tm:
            self._logger.info(f"Freezing transaction model's parameters...")
            for param in self.transaction_model.parameters():
                param.requires_grad = False

        if self.do_freeze_lm:
            self._logger.info(f"Freezing language model's parameters...")
            for param in self.language_model.parameters():
                param.requires_grad = False

        if self.do_freeze_connector:
            self._logger.info(f"Freezing connector layer's parameters...")
            for param in self.connector.parameters():
                param.requires_grad = False

    def _resize_text_embeddings(self):
        init_embeddings = self.language_model.encoder.get_input_embeddings()
        self._logger.info(f"LM initial `num_embeddings`: {init_embeddings.num_embeddings}, "
                          f"`embedding_dim`: {init_embeddings.embedding_dim}")
        self.language_model.resize_token_embeddings(len(self.tokenizer))
        resized_embedds = self.language_model.encoder.get_input_embeddings()
        self._logger.info(f"LM resized `num_embeddings`: {resized_embedds.num_embeddings}, "
                          f"`embedding_dim`: {resized_embedds.embedding_dim}")

    def model_step(self, batch, task_idx: Optional[int] = None) -> Tuple[Any, torch.Tensor]:
        # Sample single task
        if task_idx is None:
            task_idx = 0
        task = self.tasks[task_idx]

        qa_batch = task.process_input_batch(batch)

        batch_size = batch['mask'].size()[0]

        # Question template: to embedding of LM
        question_start_embeddings = self.language_model.encoder.embed_tokens(
            qa_batch['question_start_tokens'])  # call for (embed_tokens): Embedding(32128, 512)
        question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)

        question_end_embeddings_batch = self.language_model.encoder.embed_tokens(
            qa_batch['question_end_tokens'])  # call for (embed_tokens): Embedding(32128, 512)

        # Answer template: encode + strip </s> (EOS) token
        answer_template_encoded = self.tokenizer.encode(task.answer_template,
                                                        return_tensors='pt')[:-1].to(self.device)
        batch_answer_template_encoded = answer_template_encoded.repeat(batch_size, 1)
        batch_answer_mask = torch.ones(batch_size, answer_template_encoded.shape[1]).to(self.device)

        # Get transactions embeddings for initial batch
        # transactions model requires: ['mask', 'cat_features', 'num_features', 'meta_features']
        # + optionally: 'time' - ??? maybe 'event_time' ???
        # return: Tuple[
        # torch.Tensor, - embeddings -> we need this
        # torch.Tensor - mask
        # ]
        transactions_embeddings = self.transaction_model.get_embs(batch)[0]

        # next pass them to connector == linear mapping -> to LM inner dim
        transactions_embeddings = self.connector(transactions_embeddings)

        # Get general LM's encoder input as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        encoder_input = torch.cat([question_start_embeddings_batch,
                                   transactions_embeddings,
                                   question_end_embeddings_batch], dim=1)

        # Create answers + masks for LM's decoder inputs
        batch_answers = torch.cat([qa_batch['answer_template_tokens'], qa_batch['target_tokens']], dim=1)
        batch_answers_mask = torch.cat([qa_batch['answer_mask'], qa_batch['target_attention_mask']], dim=1)

        # Pass through LM
        # contains: ['loss', 'logits', 'past_key_values', 'encoder_last_hidden_state']
        # `logits` of size: [batch_size, max_pred_len, vocab_size]
        lm_outputs = self.language_model(inputs_embeds=encoder_input,
                                         labels=batch_answers,
                                         decoder_attention_mask=batch_answers_mask)

        # Return question as:
        # Q_start_tokens + TRNS_embeddings + Q_end_tokens
        lm_outputs['question_encoded'] = torch.cat([question_start_embeddings_batch,
                                                    question_end_embeddings_batch], dim=1)
        return lm_outputs, batch_answers

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
            f'{task_name}_train_loss': loss
        }
        self.log_dict(logging_dict)
        wandb.log(logging_dict)
        # self._logger.info(f"Train step results:\n{logging_dict}")
        return loss

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

        predictions_decoded = self.tokenizer.batch_decode(outputs.logits.argmax(2),
                                                          skip_special_tokens=True)
        batch_answers_decoded = self.tokenizer.batch_decode(batch_answers,
                                                            skip_special_tokens=True)
        # Calc metrics
        metrics_scores = {}
        for metric_name, metric in task.metrics.items():
            try:
                metrics_scores[metric_name] = metric(predictions_decoded,
                                                     batch_answers_decoded)
            except Exception as e:
                self._logger.error(f"error occurred during task metric `{metric_name}` calculation:\n{e}")

        logging_dict = {
            'val_loss': loss,
            f'{task.task_name}_val_loss': loss
        }
        logging_dict = dict(list(logging_dict.items()) + list(metrics_scores.items()))
        # self._logger.info(f"Validation step results:\n{logging_dict}")
        self.log_dict(
            logging_dict,
            batch_size=batch_answers.size(0)
        )
        wandb.log(logging_dict)

        # Log predictions on validation set
        if self.log_eval_steps_counter < self.num_eval_batches_to_log:
            self.log_predictions(logits=outputs.logits.detach().cpu(),
                                 answers=batch_answers.detach().cpu(),
                                 predictions_table=self.log_eval_predictions_table,
                                 log_counter=self.log_eval_steps_counter)
            self.log_eval_steps_counter += 1
        return loss

    def on_validation_epoch_start(self) -> None:
        # Reset log counter
        self.log_eval_steps_counter = 0

    def on_fit_end(self) -> None:
        # ✨ W&B: Log predictions table to wandb
        wandb.log({"val_predictions": self.log_eval_predictions_table})
        # ✨ W&B: Mark the run as complete (useful for multi-cell notebook)
        wandb.finish()

    def log_predictions(self,
                        logits: torch.Tensor,
                        answers: torch.Tensor,
                        questions: torch.Tensor,
                        predictions_table: wandb.Table, log_counter: int,
                        task_name: Optional[str] = "default",
                        epoch: Optional[int] = 0):
        predictions_decoded = self.tokenizer.batch_decode(logits.argmax(2),
                                                          skip_special_tokens=True)
        answers_decoded = self.tokenizer.batch_decode(answers,
                                                      skip_special_tokens=True)
        questions_decoded = self.tokenizer.batch_decode(questions,
                                                      skip_special_tokens=True)
        self._logger.info(f"Validation predictions vs. answers, batch #{log_counter}:")
        # columns = ["epoch", "step #", "task", "question", "prediction", "truth"]
        for i, (pred, answer, question) in enumerate(zip(predictions_decoded, answers_decoded, questions_decoded)):
            self._logger.info(f"\t#{i}{question}:\tpredicted: {pred}, answer: {answer}")
            predictions_table.add_data(epoch, "_".join([str(log_counter), str(i)]), task_name, question, pred, answer)
