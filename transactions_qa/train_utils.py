import os
import math
from typing import Optional

import torch
import torch.nn as nn
from typing import Sequence, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter


def get_warmup_steps(num_training_steps: int,
                     num_warmup_steps: Optional[int] = 0,
                     warmup_ratio: Optional[float] = 0.0):
    """
    Get number of steps used for a linear warmup.
    """
    warmup_steps = (
        num_warmup_steps if num_warmup_steps > 0 else math.ceil(num_training_steps * warmup_ratio)
    )
    return warmup_steps


class CustomPredictionsWriter(BasePredictionWriter):

    def __init__(self, output_dir: str, write_interval: Optional[str] = "epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            prediction: Any,
            batch_indices: Optional[Sequence[int]],
            batch: Any,
            batch_idx: int,
            dataloader_idx: int):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(prediction, os.path.join(self.output_dir, f"predictions_batch#{batch_idx}.pt"))


    def write_on_epoch_end(self,
                           trainer: "pl.Trainer",
                           pl_module: "pl.LightningModule",
                           predictions: Sequence[Any],
                           batch_indices: Optional[Sequence[Any]], ):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_epoch#{trainer.current_epoch}.pt"))
