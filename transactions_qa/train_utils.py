import os
import math
from copy import deepcopy
from typing import Optional, Sequence, Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import BasePredictionWriter, Callback
from pytorch_lightning.utilities import rank_zero_only


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


class EMA(Callback):

    def __init__(self, decay=0.9999, update_period=1, rm_modules=[], ema_device=None, pin_memory=True):
        self.decay = decay
        self.update_period = update_period
        self.ema_device = ema_device
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False

        self.rm_modules = rm_modules
        self.ema_state_dict = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    def get_state_dict(self, pl_module):
        state_dict = pl_module.state_dict()
        ema_state_dict = deepcopy(state_dict)
        for key in state_dict.keys():
            for rm_module in self.rm_modules:
                if key.startswith(rm_module):
                    ema_state_dict.pop(key)
        return ema_state_dict

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        if not self._ema_state_dict_ready:
            self.ema_state_dict = self.get_state_dict(pl_module)
            if self.ema_device:
                self.ema_state_dict = {
                    k: tensor.to(self.ema_device) for k, tensor in self.ema_state_dict.items()
                }
            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {
                    k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()
                }
        elif self._ema_state_dict_ready:
            for key, value in self.get_state_dict(pl_module).items():
                self.ema_state_dict[key] = self.ema_state_dict[key].to(value.device)
        self._ema_state_dict_ready = True

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.update_period == 0:
            for key, value in self.get_state_dict(pl_module).items():
                ema_state_value = self.ema_state_dict[key]
                if self.ema_device is not None:
                    value = value.to(self.ema_device)
                if value.dtype == torch.float32:
                    ema_state_value.detach().mul_(self.decay).add_(value, alpha=1. - self.decay)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return {
            "ema_state_dict": self.ema_state_dict,
            "ema_state_dict_ready": self._ema_state_dict_ready
        }

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self._ema_state_dict_ready = callback_state["ema_state_dict_ready"]
        self.ema_state_dict = callback_state["ema_state_dict"]