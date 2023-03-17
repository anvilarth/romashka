import random
import numpy as np
from typing import List, Optional, Tuple

import wandb
import torch
import torch.nn as nn

import transformers
import pytorch_lightning as pl

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
                 tasks: List[]
                 num_days: Optional[int] = 7,
                 warmup_steps: Optional[int] = 100):
        super().__init__()
        self.logger = get_logger(
            name = self.__class__.__name__,
            logging_level = "DEBUG" if self.verbose else "INFO"
        )
        self.language_model = language_model
        self.transaction_model = transaction_model