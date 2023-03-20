import random
import numpy as np
from typing import List, Optional, Tuple, Any, Dict

import wandb
import torch
import torch.nn as nn

import transformers
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.classification.f_beta import F1Score

from tasks.task_abstract import AbstractTask
from ..logging_handler import get_logger


class TransactionQAModel(pl.LightningModule):
    def __init__(self,
                 language_model: nn.Module,
                 transaction_model: nn.Module,
                 connector: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 tasks: List[AbstractTask],
                 do_freeze_tm: Optional[bool] = True,
                 do_freeze_lm: Optional[bool] = False,
                 num_days: Optional[int] = 7,
                 warmup_steps: Optional[int] = 100,
                 training_steps: Optional[int] = 10_000):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="DEBUG" if self.verbose else "INFO"
        )
        self.tokenizer = tokenizer
        self.language_model = language_model
        self.transaction_model = transaction_model
        self.connector = connector
        self.tasks = tasks
        self.warmup_steps: int = warmup_steps
        self.training_steps: int = training_steps

        self.do_freeze_tm: bool = do_freeze_tm
        self.do_freeze_lm: bool = do_freeze_lm
        self._is_multitask: bool = False
        self._prepare_model()

        self.save_hyperparameters(ignore=['tasks', '_logger'])

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


    def _resize_text_embeddings(self):
        init_embeddings = self.lm_model.encoder.get_input_embeddings()
        self._logger.info(f"LM initial `num_embeddings`: {init_embeddings.num_embeddings}, "
                          f"`embedding_dim`: {init_embeddings.embedding_dim}")
        self.lm_model.resize_token_embeddings(len(self.tokenizer))
        resized_embedds = self.lm_model.encoder.get_input_embeddings()
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
        question_start_embeddings = self.lm_model.encoder.embed_tokens(
            qa_batch['question_start_tokens'])  # call for (embed_tokens): Embedding(32128, 512)
        question_start_embeddings_batch = question_start_embeddings.repeat(batch_size, 1, 1)

        question_end_embeddings_batch = self.lm_model.encoder.embed_tokens(
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
        lm_outputs = self.lm_model(inputs_embeds=encoder_input,
                                   labels=batch_answers,
                                   decoder_attention_mask=batch_answers_mask)
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

        self.log_dict(
            {
                'train_loss': loss,
                f'{task_name}_train_loss': loss
            }
        )
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

        Examples:
            def validation_step(self, batch, batch_idx):
                x, y = batch

                # implement your own
                out = self(x)
                loss = self.loss(out, y)

                # log 6 example images
                # or generated text... or whatever
                sample_imgs = x[:6]
                grid = torchvision.utils.make_grid(sample_imgs)
                self.logger.experiment.add_image('example_images', grid, 0)

                # calculate acc
                labels_hat = torch.argmax(out, dim=1)
                val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

                # log the outputs!
                self.log_dict({'val_loss': loss, 'val_acc': val_acc})
        """
        # Sample a random single task
        task_idx = random.sample(list(range(len(self.tasks))), k=1)[0]
        task = self.tasks[task_idx]

        outputs, batch_answers = self.model_step(batch, task_idx=task_idx)
        if outputs is None:
            return None

        loss = outputs.loss

        predictions_decoded = self.tokenizer.batch_decode(outputs.logits.argmax(2))
        batch_answers_decoded = self.tokenizer.batch_decode(batch_answers)

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
        self.log_dict(
            logging_dict,
            batch_size=batch_answers.size(0)
        )
        return loss
