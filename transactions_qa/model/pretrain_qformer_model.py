import random
import numpy as np
from copy import deepcopy
from typing import List, Optional, Tuple, Any, Dict, Union

import wandb
import torch
import torch.nn as nn

import transformers
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import WandbLogger

from romashka.logging_handler import get_logger
from romashka.transactions_qa.utils import seed_everything
from romashka.transactions_qa.losses.infonce_loss import InfoNCE
from romashka.transactions_qa.utils import (mask_padding, mask_lm_labels_padding)


class PretrainQFormerModel(pl.LightningModule):
    def __init__(self,
                 language_model: nn.Module,
                 transaction_model: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 qformer: Optional[nn.Module] = None,
                 learning_rate: Optional[float] = 5e-5,
                 scheduler_type: Optional[Union[transformers.SchedulerType, str]] = "linear",
                 adam_beta1: Optional[float] = 0.9,
                 adam_beta2: Optional[float] = 0.999,
                 adam_epsilon: Optional[float] = 1e-8,
                 warmup_steps: Optional[int] = 100,
                 training_steps: Optional[int] = 10_000,
                 verbose_for_debug: Optional[bool] = False,
                 **additional_kwargs
                ):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.language_model = language_model
        self.tokenizer = tokenizer
        self.transaction_model = transaction_model
        self.qformer = qformer

        self.warmup_steps: int = warmup_steps
        self.training_steps: int = training_steps
        self.base_learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.adam_beta1: float = adam_beta1
        self.adam_beta2: float = adam_beta2
        self.adam_epsilon = adam_epsilon

        self._verbose_for_debug: bool = verbose_for_debug

        self.save_hyperparameters(ignore=['_logger', 'language_model', 'tokenizer',
                                          'transaction_model', 'qformer'])

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

        # Set LM model type: encoder-decoder / decoder-only
        self._set_model_type()

        # Create losses for contrastive
        self._create_losses()

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

    def _freeze_parameters(self):
        # Freezing models weights
        self.transaction_model.eval()
        self._logger.info(f"Freezing transaction model's parameters...")
        for param in self.transaction_model.parameters():
            param.requires_grad = False

        self.language_model.eval()
        self._logger.info(f"Freezing language model's parameters...")
        for param in self.language_model.parameters():
            param.requires_grad = False

    def _create_losses(self):
        # Use contrastive loss for embeddings comparison
        self.loss_fn = InfoNCE()
