import random
import numpy as np
from copy import deepcopy
from typing import List, Optional, Tuple, Any, Dict, Union

import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

import transformers
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info

from romashka.logging_handler import get_logger


DEFAULT_CONFIG = {
    "sequence_len": 384,
    "num_queries": 32,
    "shared_dim": 256,
    "hidden_size": 256,
    "num_attention_heads": 4,
    "num_hidden_layers": 4,
    "intermediate_size": 1024,
    "cross_attention_frequency": 2,
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "max_position_embeddings": 1024,
    "max_text_sequence_len": 512,
    "truncation_side": "right",
    "position_embedding_type": "absolute",
}


class PretrainQFormerModel(pl.LightningModule):
    def __init__(self,
                 language_model_name: str,
                 sequence_encoder_model: nn.Module,
                 qformer: Optional[nn.Module] = None,
                 learning_rate: Optional[float] = 5e-5,
                 scheduler_type: Optional[Union[transformers.SchedulerType, str]] = "linear",
                 adam_beta1: Optional[float] = 0.9,
                 adam_beta2: Optional[float] = 0.999,
                 adam_epsilon: Optional[float] = 1e-8,
                 warmup_steps: Optional[int] = 100,
                 training_steps: Optional[int] = 10_000,
                 verbose_for_debug: Optional[bool] = False,
                 qformer_kwargs: Optional[Dict[str, Any]] = None,
                 **additional_kwargs
                 ):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.language_model_name = language_model_name
        self.sequence_encoder_model = sequence_encoder_model
        self.qformer = qformer

        self.qformer_kwargs = qformer_kwargs if qformer_kwargs is not None else DEFAULT_CONFIG
        self.qformer_kwargs['text_model_name'] = self.language_model_name
        self.qformer_kwargs['sequence_encoder_model'] = self.sequence_encoder_model

        self.warmup_steps: int = warmup_steps
        self.training_steps: int = training_steps
        self.base_learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.adam_beta1: float = adam_beta1
        self.adam_beta2: float = adam_beta2
        self.adam_epsilon = adam_epsilon

        self._verbose_for_debug: bool = verbose_for_debug
        self.save_hyperparameters(ignore=['_logger', 'sequence_encoder_model', 'qformer'])

        self._prepare_model()

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
        """
        Prepare model for training.
        - Freeze Event Sequence encoder & LLM;
        - Set LM model type: encoder-decoder / decoder-only;
        """
        # Freeze Event Sequence encoder & LLM
        self._freeze_parameters()

    def _freeze_parameters(self):
        # Freezing models weights
        self.sequence_encoder_model.eval()
        self._logger.info(f"Freezing sequence encoder model's parameters...")
        for param in self.sequence_encoder_model.parameters():
            param.requires_grad = False

    def model_step(self, batch: Union[Dict[str, torch.Tensor], Any],
                   output_attentions: Optional[bool] = True) -> Any:
        """
        Passes input batch through inner parts of module:
        1) input sequences through Sequence embedder model (i.e. sequence_encoder_model)
            -> to get embeddings [bs, hist_seq_len];
        2) those sequences embeddings + text captions through Q-Former model
            to get output queries representation [bs, num_queries];
        3) Calculate Contrastive loss for embedding of
            text representations vs. output queries representations vs. sequences representations
            (mine negatives within batch).
        Args:
            batch: a prepared with chosen task batch of items;
            output_attentions: whether to output attention maps;
            output_hidden_states: whether to output LM hidden states;

        Returns:
            LM model's outputs with added attributes (based on values of `output_attentions`
            and `output_hidden_states`)
        """
        # 1) Sequences through Sequence embedder model (i.e. sequence_encoder_model)
        #     -> to get embeddings [bs, hist_seq_len];
        sequences_embeddings, sequences_embeddings_mask = self.sequence_encoder_model.get_embs(batch)

        # 2) those sequences embeddings + text captions through Q-Former model
        batch_captions = [caption[0] for caption in batch['captions']]
        outputs = self.qformer(
            text=batch_captions,
            sequence_embeds=sequences_embeddings,
            is_train=True
        )

        return outputs

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
        outputs = self.model_step(batch)
        if outputs is None:
            return None

        loss = outputs.get('loss', 0.0)
        self.log(
            "train_loss", loss,
            on_step=True, on_epoch=True,
            prog_bar=True, logger=True,
            sync_dist=True,
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
        """
        outputs = self.model_step(batch)
        if outputs is None:
            return None

        loss = outputs.get('loss', 0.0)
        self.log(
            "val_loss", loss,
            on_step=False, on_epoch=True,
            prog_bar=True, logger=True,
            sync_dist=True,
        )
        return loss
