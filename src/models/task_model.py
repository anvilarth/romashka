from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.models import TransactionsModel
from src.models.components.my_utils import get_projections_maps, cat_features_names, num_features_names, meta_features_names

from src.tasks import AutoTask

class TaskModule(LightningModule):
    """
    HERE

    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        task_name,
        encoder_type='whisper/tiny',
        head_type='linear',
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # TODO fix on correct processing
        folder = '/home/jovyan/romashka'

        projections_maps = get_projections_maps(relative_folder=folder)
        transactions_model_config = {
            "cat_features": cat_features_names,
            "cat_embedding_projections": projections_maps.get('cat_embedding_projections'),
            "num_features": num_features_names,
            "num_embedding_projections": projections_maps.get('num_embedding_projections'),
            "meta_features": meta_features_names,
            "meta_embedding_projections": projections_maps.get('meta_embedding_projections'),
            "encoder_type": encoder_type,
            "head_type": head_type,
            "embedding_dropout": 0.1
        }

        self.transactions_model = TransactionsModel(**transactions_model_config)
        self.task = AutoTask.get(task_name=task_name, task_type='non-text')

        self.metrics = self.task.metrics
        # loss function
        self.criterion = self.task.criterion

    def forward(self, x: dict):
        return self.transactions_model(x)

    def shared_step(self, batch: Any, batch_idx=None):
        y = self.task.generate_target(batch)
        
        if len(y) == 2:
            if y[0] is None:
                return  None,  None
            y = y[0]

        logits = self.forward(batch)
        return logits, y

    def training_step(self, batch: Any, batch_idx: int):
        outputs, answers = self.shared_step(batch)
        if outputs is None:
            return None

        loss = self.criterion(outputs, answers)

        # update and log metrics
        self.log("train/loss",loss, on_step=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return loss

    def validation_step(self, batch: Any, batch_idx=None, **kwargs: Any):
        batch_size=batch['mask'].shape[0]

        outputs, answers = self.shared_step(batch)
        if outputs is None:
            return None

        loss = self.criterion(outputs, answers)

        metrics_scores = self.task.calculate_metrics(outputs, answers, self.metrics)

        # update and log metrics
        self.log("val/loss",loss, on_step=True, prog_bar=True)
        self.log_dict(
            metrics_scores,
            batch_size=batch_size
        )


        return loss


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}