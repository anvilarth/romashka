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

        # Set LM model type: encoder-decoder / decoder-only
        self._set_model_type()

        # Create losses for contrastive
        self._create_losses()

    def _set_model_type(self):
        # For encoder-decoder models
        if hasattr(self.language_model, "encoder"):
            self._is_encoder_decoder = True
        # For decoder-only
        elif hasattr(self.language_model, "transformer") \
                or ("gpt" in self.language_model.config._name_or_path.lower()) \
                or ("opt" in self.language_model.config._name_or_path.lower()):
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
        self.loss_fn = InfoNCE(negative_mode='paired')

    def model_step(self, batch: Union[Dict[str, torch.Tensor], Any],
                   output_attentions: Optional[bool] = True) -> Any:
        """
        Passes input batch through inner parts of module:
        1) a transaction sequences through Sequence embedder model (i.e. transactions model)
            -> to get embeddings [bs, hist_seq_len];
        2) those transactions embeddings through Q-Former model
            to get output queries representation [bs, num_queries];
        3) transaction captions through LM model -> to get a single embedding of CLS token
            (aggregate all token sequence info in it);
        4) Calculate Contrastive loss for embedding of CLS token vs. output queries representations
            (mine negatives within batch).
        Args:
            batch: a prepared with chosen task batch of items;
            output_attentions: whether to output attention maps;
            output_hidden_states: whether to output LM hidden states;

        Returns:
            LM model's outputs with added attributes (based on values of `output_attentions`
            and `output_hidden_states`)
        """
        # 1) a transaction sequences through Sequence embedder model (i.e. transactions model)
        #     -> to get embeddings [bs, hist_seq_len];
        # transactions model requires: ['mask', 'cat_features', 'num_features', 'meta_features']
        # return: Tuple[
        # torch.Tensor, - embeddings [bs, hist_seq_len]
        # torch.Tensor - mask [bs, hist_seq_len]
        # ]
        transaction_mask = batch['mask']
        batch_size = transaction_mask.size(0)
        device = transaction_mask.device

        transactions_embeddings, transactions_embeddings_mask = self.transaction_model.get_embs(batch)

        # 2) those transactions embeddings through Q-Former model
        # to get output queries representation [bs, num_queries];
        transactions_embeddings = self.qformer(transactions_embeddings)

        # 3.1) Tokenize + pad to max_length captions
        batch_captions = [caption[0] for caption in batch['captions']]
        batch_captions_encoded = self.tokenizer(batch_captions,
                                                padding=True, truncation=False,
                                                return_tensors='pt').to(device)

        # 3.2) Pass tokenized transaction captions through LM model -> last hisdden state of last token in sequence
        lm_outputs = self.language_model(batch_captions_encoded['input_ids'],
                                         attention_mask=batch_captions_encoded['attention_mask'],
                                         output_attentions=output_attentions,
                                         output_hidden_states=True,
                                         )
        last_hidden_state = lm_outputs['last_hidden_state']

        # Get indexes of last token (before paddings)
        sequence_lengths = (torch.ne(batch_captions_encoded['input_ids'],
                                     self.tokenizer.pad_token_id).sum(-1) - 1).to(device)

        # Get last token embedding
        pooled_logits = last_hidden_state[torch.arange(batch_size, device=device), sequence_lengths, :]

        # 3.3) Compare with each query to get the one with max similarity per batch sample (size: [bs, 1])
        # pooled_logits_ = pooled_logits.unsqueeze(1).repeat(1, transactions_embeddings.size(1), 1)

        # for each caption select positive and all neagtive pairs in batch (max per queries)
        positive_queries = []
        all_negative_queries = []

        for bi, logits in enumerate(pooled_logits):
            logits_ = logits[None, None].repeat(batch_size, transactions_embeddings.size(1), 1)
            sim_ = F.cosine_similarity(logits_, transactions_embeddings, dim=2, eps=1e-6)
            max_sim_queries_ids = sim_.argmax(1)  # max sim score per each query per each elem in batch -> size: [bs,]

            # Positive pair -> max scored query in the SAME sample in batch
            pos_query = transactions_embeddings[bi][max_sim_queries_ids[bi]]
            positive_queries.append(pos_query.unsqueeze(0))

            # Negative pairs: all max scored queries (with current caption) in OTHER samples in batch
            negative_queries = [transactions_embeddings[i][max_sim_queries_ids[i]]
                                for i in range(batch_size) if i != bi]
            negative_queries = torch.vstack(negative_queries)
            all_negative_queries.append(negative_queries.unsqueeze(0))

        positive_queries = torch.vstack(positive_queries)  # [N * D] -> 1 positive pair
        all_negative_queries = torch.vstack(all_negative_queries)  # [N * M * D] -> M negative pairs

        # Calculate loss
        loss = self.loss_fn(
            pooled_logits,  # [N * D] -> 1 positive pair
            positive_key=positive_queries,  # [N * D] -> 1 positive pair
            negative_keys=all_negative_queries  # [N * M * D] -> M negative pairs
            )

        return dict(
            loss=loss
        )

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
