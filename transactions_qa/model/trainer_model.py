import sys
import copy
import random
import traceback
import numpy as np
import collections
from copy import deepcopy
from typing import List, Optional, Tuple, Any, Dict, Union

import wandb
import torch
import torch.nn as nn

import transformers
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.loggers import WandbLogger

from deepspeed.ops.adam import FusedAdam

import bitsandbytes as bnb
from romashka.logging_handler import get_logger
from romashka.transactions_qa.utils import inspect_init_signature
from romashka.transactions_qa.model.pooler import PoolerType
from romashka.transactions_qa.model.projection import ProjectionsType
from romashka.transactions_qa.layers import MixedPrecisionLayerNorm


class ContrastiveTransactionsModel(pl.LightningModule):
    def __init__(self,
                 text_model: nn.Module,
                 trns_encoder: nn.Module,
                 encoder_pooler_type: str = "CLS_POOLER",
                 encoder_projection_type: str = "LINEAR",
                 shared_dim: int = 768,
                 learning_rate: Optional[float] = 5e-5,
                 optimizer_type: Optional[Union[torch.optim.Optimizer, str]] = "AdamW",
                 scheduler_type: Optional[Union[transformers.SchedulerType, str]] = "linear",
                 use_8bit_optim: Optional[bool] = False,
                 adam_beta1: Optional[float] = 0.9,
                 adam_beta2: Optional[float] = 0.999,
                 adam_epsilon: Optional[float] = 1e-8,
                 warmup_steps: Optional[int] = 100,
                 training_steps: Optional[int] = 10_000,
                 is_debug: Optional[bool] = False,
                 use_deepspeed: Optional[bool] = False,
                 return_logits: Optional[bool] = False,
                 **additional_kwargs
                 ):
        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            log_filename="./DEBUG_logging_contrastive.txt"
        )
        self.text_model = text_model
        self.trns_encoder = trns_encoder
        self.encoder_pooler_type = encoder_pooler_type
        self.encoder_projection_type = encoder_projection_type
        self.initial_params = None

        self.shared_dim = shared_dim
        self.warmup_steps: int = warmup_steps
        self.training_steps: int = training_steps
        self.base_learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.use_8bit_optim: bool = use_8bit_optim
        self.adam_beta1: float = adam_beta1
        self.adam_beta2: float = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.use_deepspeed = use_deepspeed

        self.metric = torchmetrics.CosineSimilarity(reduction='mean')

        self._is_encoder_decoder: bool = text_model.is_encoder_decoder
        if not self.use_deepspeed:
            self._prepare_model()

        self._is_debug: bool = is_debug
        self._return_logits: bool = return_logits
        self.save_hyperparameters(ignore=['_logger', 'text_model', 'trns_encoder'])

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
        # Init optimizer
        optimizer_type = None
        if self.use_deepspeed:
            optimizer = FusedAdam(self.parameters())
            rank_zero_info(
                f"The model will train with DeepSpeed FusedAdam optimizer."
            )
        elif isinstance(self.optimizer_type, torch.optim.Optimizer):
            optimizer_type = self.optimizer_type
            rank_zero_info(
                f"The model will train with {optimizer_type} optimizer."
            )
        elif self.optimizer_type in ["Adam", "AdamW"] and self.use_8bit_optim:
            optimizer_type = 'Adam8bit'
            optimizer = bnb.optim.Adam8bit(self.parameters(),
                                           lr=self.base_learning_rate,
                                           betas=(self.adam_beta1, self.adam_beta2))
            rank_zero_info(
                f"The model will train with 8-bit AdamW optimizer."
            )
        elif hasattr(torch.optim, self.optimizer_type):
            optimizer_type = getattr(torch.optim, self.optimizer_type)
            optim_params = {
                "params": self.parameters(),
                "lr": self.base_learning_rate
            }
            # Add beta1, beta2 for Adam-like optimizers
            if inspect_init_signature('betas', optimizer_type):
                optim_params['betas'] = (self.adam_beta1, self.adam_beta2)

            # Instantiate optimizer
            optimizer = optimizer_type(**optim_params)
            rank_zero_info(
                f"Instantiating {self.optimizer_type} optimizer"
            )
        else:
            rank_zero_info(f"Unknown {optimizer_type} optimizer, so create AdamW optimizer.")
            optimizer = torch.optim.AdamW(self.parameters(),
                                          betas=(self.adam_beta1, self.adam_beta2),
                                          lr=self.base_learning_rate)
        # Select scheduler
        if self.scheduler_type == "linear_schedule_with_warmup":
            scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=self.warmup_steps,
                                                                     num_training_steps=self.training_steps)
        elif self.scheduler_type == "cosine_schedule_with_warmup":
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=self.warmup_steps,
                                                                     num_training_steps=self.training_steps)
        else:
            scheduler = transformers.get_scheduler(name=self.scheduler_type,
                                                   optimizer=optimizer,
                                                   num_warmup_steps=self.warmup_steps,
                                                   num_training_steps=self.training_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def configure_model(self):
        """
        Created within sharded model context, modules are instantly sharded across processes
        as soon as they are made.
        For DeepSpeed Zero-3!
        """
        # Create additional layers
        self._create_encoder_pooler()
        self._create_encoder_projection()

        # Create norms & scales
        self.text_l_norm = MixedPrecisionLayerNorm(normalized_shape=(self.shared_dim))
        self.encoder_l_norm = MixedPrecisionLayerNorm(normalized_shape=(self.shared_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Check total trainable parameters
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        self._logger.info(
            f"Transactions encoder - totally trainable parameters: {len(trainable_parameters)} from {len(parameters)}")

        # Figure out what the model type passed encoder-decoder / decoder-only
        self._set_model_type()


    def _prepare_model(self):

        # Create additional layers
        self._create_encoder_pooler()
        self._create_encoder_projection()

        # Create norms & scales
        self.text_l_norm = MixedPrecisionLayerNorm(normalized_shape=(self.shared_dim))
        self.encoder_l_norm = MixedPrecisionLayerNorm(normalized_shape=(self.shared_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Check total trainable parameters
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        self._logger.info(
            f"Transactions encoder - totally trainable parameters: {len(trainable_parameters)} from {len(parameters)}")

        # Figure out what the model type passed encoder-decoder / decoder-only
        self._set_model_type()

    def _create_encoder_pooler(self):
        """
        Creates a Pooler network.
        """
        self.encoder_pooler = PoolerType.get(pooler_type_name=self.encoder_pooler_type)
        self._logger.info(f"Created `{self.encoder_pooler_type}` pooler.")

    def _create_encoder_projection(self):
        """
        Creates a Projection network.
        """
        proj_kwargs = {'in_dim': self.trns_encoder.output_size, "out_dim": self.shared_dim}
        self.encoder_projection = ProjectionsType.get(projection_type_name=self.encoder_projection_type,
                                                      **proj_kwargs)
        self._logger.info(f"Created `{self.encoder_projection_type}` projection.")

    def _set_model_type(self):
        if self._is_encoder_decoder is None:
            # For encoder-decoder models
            if hasattr(self.model.language_model, "encoder"):
                self._is_encoder_decoder = True
            # For decoder-only
            elif hasattr(self.model.language_model, "transformer"):
                self._is_encoder_decoder = False
            else:
                raise NotImplementedError(f"Unknown model type: {type(self.model.language_model)}")

        self._logger.info(f"Language model type: `{'encoder-decoder' if self._is_encoder_decoder else 'decoder'}`")

    def encode_transactions(self, batch) -> torch.Tensor:
        """
        Receives a transactions history batch and encode it to a single vector representation.
        Args:
            batch: a batch of samples;
        Returns:
            embeddings of a transactions histories - size [bs, shared_dim];
        """
        # 1) Encoder transactions history
        transaction_mask = batch['mask']
        batch_size = transaction_mask.size(0)

        # embeddings of size: [bs, trns_hist_len, encoder_dim]
        transactions_embeddings, transactions_embeddings_mask = self.trns_encoder.encode(batch)

        # 2) Pass through pooler
        # embeddings of size: [bs, encoder_dim]
        pooled_transactions_embeddings = self.encoder_pooler(
            dict(decoder_hidden_states=transactions_embeddings.unsqueeze(0)),
            transactions_embeddings_mask)

        # 3) Pass through projection
        # from [bs, encoder_dim] -> [bs, shared_dim]
        projected_transactions_embeddings = self.encoder_projection(pooled_transactions_embeddings)

        # 4) Final Layer Norm
        norm_projected_transactions_embeddings = self.encoder_l_norm(projected_transactions_embeddings)

        return norm_projected_transactions_embeddings

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
        device = self.device
        if device.type != 'cpu':
            torch.cuda.empty_cache()

        # Encode transactions history to a single vector representation
        # trns_outputs size [bs, shared_dim]
        trns_outputs = self.encode_transactions(batch)
        batch_size = trns_outputs.size(0)

        # Encode transactions history captions to a single vector representation
        # with keys: 'loss', 'logits', 'labels', 'projected_outputs',
        # 'decoder_hidden_states'/'encoder_last_hidden_state'/'encoder_hidden_states'/'hidden_states'
        # from initial batch: 'input_ids', 'attention_mask'
        # projected_outputs size [bs, shared_dim]
        lm_outputs = self.text_model(batch)

        # Text final Layer Norm
        text_outputs = self.text_l_norm(lm_outputs['projected_outputs'])

        # Re-normalized features
        text_outputs = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        trns_outputs = trns_outputs / trns_outputs.norm(dim=-1, keepdim=True)

        # Loss
        # For n_gpu == 1
        logit_scale = self.logit_scale.exp()
        logits_per_trns_hist = logit_scale * trns_outputs @ text_outputs.t()
        logits_per_text_capt = logit_scale * text_outputs @ trns_outputs.t()

        contrastive_labels = torch.arange(len(logits_per_trns_hist)).to(logits_per_trns_hist.device)

        trns_loss = torch.nn.functional.cross_entropy(logits_per_trns_hist, contrastive_labels)
        text_loss = torch.nn.functional.cross_entropy(logits_per_text_capt, contrastive_labels)

        loss = (trns_loss + text_loss) / 2

        logging_dict = {
            'train_loss': loss,
            'train_text_contr_loss': text_loss,
            'train_trns_contr_loss': trns_loss,
            "train_lm_text_loss": lm_outputs['loss']
        }

        # self.log_dict(
        #     logging_dict,
        #     sync_dist=True,
        #     on_step=False, on_epoch=True,
        #     prog_bar=True, logger=True
        # )

        self.log('train_loss', loss, logger=True,
                 on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log('train_text_contr_loss', text_loss, logger=True,
                 on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log('train_trns_contr_loss', trns_loss, logger=True,
                 on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log('train_lm_text_loss', lm_outputs['loss'], logger=True,
                 on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)

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
        # Encode transactions history to a single vector representation
        # trns_outputs size [bs, shared_dim]
        trns_outputs = self.encode_transactions(batch)
        batch_size = trns_outputs.size(0)

        # Encode transactions history captions to a single vector representation
        # with keys: 'loss', 'logits', 'labels', 'projected_outputs',
        # 'decoder_hidden_states'/'encoder_last_hidden_state'/'encoder_hidden_states'/'hidden_states'
        # from initial batch: 'input_ids', 'attention_mask'
        # projected_outputs size [bs, shared_dim]
        lm_outputs = self.text_model(batch)

        # Text final Layer Norm
        text_outputs = self.text_l_norm(lm_outputs['projected_outputs'])

        # Re-normalized features
        text_outputs = text_outputs / text_outputs.norm(dim=-1, keepdim=True)
        trns_outputs = trns_outputs / trns_outputs.norm(dim=-1, keepdim=True)

        # Loss
        # For n_gpu == 1
        logit_scale = self.logit_scale.exp()
        logits_per_trns_hist = logit_scale * trns_outputs @ text_outputs.t()
        logits_per_text_capt = logit_scale * text_outputs @ trns_outputs.t()

        contrastive_labels = torch.arange(len(logits_per_trns_hist)).to(logits_per_trns_hist.device)

        trns_loss = torch.nn.functional.cross_entropy(logits_per_trns_hist, contrastive_labels)
        text_loss = torch.nn.functional.cross_entropy(logits_per_text_capt, contrastive_labels)

        loss = (trns_loss + text_loss) / 2

        logging_dict = {
            'val_loss': loss,
            'val_text_contr_loss': text_loss,
            'val_trns_contr_loss': trns_loss,
            "val_lm_text_loss": lm_outputs['loss']
        }

        # Calculate similarity
        try:
            self.metric(text_outputs, trns_outputs)
            logging_dict['val_cosine_sim'] = self.metric
            self.log('val_cosine_sim', self.metric, logger=True,
                     on_step=True, on_epoch=True,
                     sync_dist=True, batch_size=batch_size)
        except Exception as e:
            self._logger.error(f"Error occurred during metric calculation!")
            self._logger.error(f"{traceback.format_exc()}")

        # self.log_dict(
        #     logging_dict,
        #     sync_dist=True,
        #     on_step=True, on_epoch=True,
        #     prog_bar=True, logger=True
        # )

        self.log('val_loss', loss, logger=True,
                 on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log('val_text_contr_loss', text_loss, logger=True,
                 on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log('val_trns_contr_loss', trns_loss, logger=True,
                 on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)
        self.log('val_lm_text_loss', lm_outputs['loss'], logger=True,
                 on_step=True, on_epoch=True,
                 sync_dist=True, batch_size=batch_size)

        return loss

    def predict_step(self, batch: Any,
                     batch_idx: int,
                     dataloader_idx: int = 0) -> Any:
        """
        Step function called during Trainer.predict().
        to write the predictions to disk or database after each batch or on epoch end.

        Args:
            batch: Current batch.
            batch_idx: Index of current batch;
            multiple_choice_grade: whether to use multiple_choice_grade evaluation scheme;
            dataloader_idx: Index of the current dataloader.
        Return:
            Predicted output
        """
        # Encode transactions history captions to a single vector representation
        # with keys: 'loss', 'logits', 'labels', 'projected_outputs',
        # 'decoder_hidden_states'/'encoder_last_hidden_state'/'encoder_hidden_states'/'hidden_states'
        # from initial batch: 'input_ids', 'attention_mask'
        # projected_outputs size [bs, shared_dim]
        return self.text_model(batch)

    def on_validation_epoch_start(self) -> None:
        print(f"\n----------- Validation start ----------\n")

    def on_validation_epoch_end(self) -> None:
        print(f"\n----------- Validation end ----------\n")
        # log epoch metric
        self.log('val_epoch_cosine_sim', self.metric)

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):

        # example to inspect gradient information in tensorboard
        if self.trainer.global_step % 1 == 0:  # don't make logging too much
            # log gradients of model parameters
            for param_name, param in self.named_parameters():
                if param_name.startswith("logit_scale") \
                        or param_name.startswith("trns_encoder") \
                        or param_name.startswith("encoder_pooler"):
                    if param.grad is not None:
                        grad_sum = np.sum(np.abs(param.grad.detach().cpu().numpy()))
                        self._logger.info(f"Parameter `{param_name}` with grad of size: {param.grad.size()}")
                        self._logger.info(f"Summed `{param_name}` grad = {grad_sum}")
                        self.log(
                            name=f"{param_name}_grad_sum", value=grad_sum,
                            sync_dist=True, on_epoch=True, on_step=True
                        )
                    else:
                        self._logger.info(f"Parameter `{param_name}` has NO grad!")
                        self.log(
                            name=f"{param_name}_grad_sum", value=0.0,
                            sync_dist=True, on_epoch=True, on_step=True
                        )
            # log norms of model embedding matrices
            # for param_name, param in self.trns_encoder.named_parameters():
            #     if param_name.startswith("embedding"):
            #         vec_norm = torch.norm(param.detach().cpu())
            #         self.log(
            #             name=f"{param_name}_norm", value=vec_norm, sync_dist=True
            #         )

            if self._is_debug:
                # Make sure that parameters updates
                self._stash_params()

    def _stash_params(self, params: Optional[Dict[str, Any]] = None):
        """
        Stashes parameters before updating.
        Args:
            params (Optional[List[str, Any]]): a dict of parameters and their names.
        """
        if params is None:
            # get a list of params that are allowed to change
            params = [name_param for name_param in self.named_parameters() if name_param[1].requires_grad]
        if isinstance(params, dict):
            params = [name_param for name_param in params.items()]

        # take a copy
        self.initial_params = [(name, p.clone()) for (name, p) in params]

    def on_before_zero_grad(self, optimizer) -> None:
        """
        Called after ``training_step()`` and before ``optimizer.zero_grad()``.
        Called in the training loop after taking an optimizer step and before zeroing grads.
        """
        if self._is_debug  and (self.initial_params is not None):
            # get a list of params that are allowed to change
            params = [name_param for name_param in self.named_parameters() if name_param[1].requires_grad]
            self._var_change_helper(True,
                                    initial_params=self.initial_params,
                                    updated_params=params,
                                    device=self.device)

    def _var_change_helper(self, vars_change: bool, initial_params, updated_params, device):
        """
        Check if given variables (params) change or not during training

        Parameters
        ----------
        vars_change : bool
          a flag which controls the check for change or not change
        initial_params : list,
          list of parameters of form (name, variable)
        updated_params : list,
          list of parameters of form (name, variable)

        Raises
        ------
        logs errors / warnings
          if vars_change is True and params DO NOT change during training
          if vars_change is False and params DO change during training
        """

        # check if variables have changed
        for (_, p0), (name, p1) in zip(initial_params, updated_params):
            try:
                if vars_change:
                    assert not torch.equal(p0.to(device), p1.to(device))
                else:
                    assert torch.equal(p0.to(device), p1.to(device))
            except Exception as e:
                msg = 'did not change!' if vars_change else 'changed!'
                var_name = name
                self._logger.warning(  # error message
                    f"{var_name} {msg}"
                )
