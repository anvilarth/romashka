from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from .models import TransactionsModel
from .my_utils import get_projections_maps, cat_features_names, num_features_names, meta_features_names


class DefaultModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
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

        # loss function
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(self, x: dict):
        return self.transactions_model(x)

    def model_step(self, batch: Any):
        y = batch['label'].float()
        logits = self.forward(batch).flatten()
        loss = self.criterion(logits, y)
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.log("train/loss",loss, on_step=True, prog_bar=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}


    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}