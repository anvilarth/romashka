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
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig

from romashka.logging_handler import get_logger
from romashka.transactions_qa.layers.qformer_text import Blip2QFormerTextEncoder

DEFAULT_CONFIG = {
    "num_queries": 32,
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
                 language_model: nn.Module,
                 sequence_encoder_model: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 qformer: nn.Module,
                 shared_dim: Optional[int] = None,
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
        self.qformer_kwargs = qformer_kwargs if qformer_kwargs is not None else DEFAULT_CONFIG
        self.qformer_config = Blip2QFormerConfig(**self.qformer_kwargs)

        self.language_model = language_model
        self.sequence_encoder_model = sequence_encoder_model
        self.tokenizer = tokenizer
        self.q_qformer = qformer  # a part for vis + queries part
        self.t_qformer = Blip2QFormerTextEncoder(qformer_kwargs)  # a text part

        # A shared dim for contrastive loss computation
        self.shared_dim = shared_dim if shared_dim is not None else self._get_hidden_dim()

        self.warmup_steps: int = warmup_steps
        self.training_steps: int = training_steps
        self.base_learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.adam_beta1: float = adam_beta1
        self.adam_beta2: float = adam_beta2
        self.adam_epsilon = adam_epsilon

        self._verbose_for_debug: bool = verbose_for_debug
        self.save_hyperparameters(ignore=['_logger', 'sequence_encoder_model', 'qformer', 'language_model'])

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
        - Create queries;
        - Tie weights of two parts;
        """
        self._create_layers()
        self._tie_parameters()

    def _get_hidden_dim(self) -> int:
        """
        Retrieve model hidden size (from inner cofig).
        Returns: (int) hidden size.
        """
        hidden_dim = None
        if hasattr(self.language_model.config, "hidden_size"):
            hidden_dim = self.language_model.config.hidden_size
        elif hasattr(self.language_model.config, "d_model"):  # may occur in encoder-decoder models (like T5)
            hidden_dim = self.language_model.config.d_model
        else:
            raise AttributeError(f"The default setting, where parameters embeddings dimensionality "
                                 "equals to LLM hidden dimensionality can not be run because "
                                 "model do not have required attribute: `hidden_size` or `d_model` in config.")
        return hidden_dim

    def _create_layers(self):
        self.query_tokens_embeddings = torch.nn.Parameter(
                torch.zeros(1, self.qformer_config.num_queries, self.qformer_config.hidden_size))

        self.text_projection_layer = torch.nn.Linear(self._get_hidden_dim(),
                                                     self.shared_dim)
        self.queries_projection_layer = torch.nn.Linear(self.qformer_config.hidden_size,
                                                        self.shared_dim)

    def _tie_parameters(self):
        """
        Share self-attention layer's weights between two towers.
        """
        # Select weights for sharing -> self-att layers
        shared_params_names = []
        for t_param_name, t_param in self.t_qformer.layer.named_parameters():
            if (("attention" in t_param_name) or ("intermediate_query" in t_param_name)
                or ("output_query" in t_param_name)) and not ("LayerNorm" in t_param_name):
                shared_params_names.append(".".join(t_param_name.split(".")[:-1]))
        shared_params_names = list(set(shared_params_names))

        for param_name in shared_params_names:
            try:
                param1 = self.t_qformer.layer.get_submodule(param_name)
                param2 = self.q_qformer.encoder.layer.get_submodule(param_name)
                param2 = param1
            except Exception as e:
                self._logger.error(f"Error occurred during parameter: `{param_name}` weights tying:\n{e}")
        self._logger.info(f"{len(shared_params_names)} parameters tied:\n{shared_params_names}")

    def model_step(self, batch: Union[Dict[str, torch.Tensor], Any],
                   output_hidden_states: Optional[bool] = False,
                   output_attentions: Optional[bool] = True,
                   return_dict: Optional[bool] = True) -> Any:
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
        #     -> to get embeddings [bs, hist_seq_len, 384];
        sequences_embeddings, sequences_embeddings_mask = self.sequence_encoder_model.get_embs(batch)

        # 2) Pass the query tokens through queries part of full Q-Former model,
        #  using input embeddings for cross-attention
        query_tokens = self.query_tokens_embeddings.expand(sequences_embeddings.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=sequences_embeddings,
            encoder_attention_mask=sequences_embeddings_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        query_output = query_outputs[0]
        # 3) Project query tokens to shared dim for loss computation
        query_output = self.queries_projection_layer(query_output)
        query_output = F.normalize(query_output, dim=-1)

        # 4) Texts
        # 4.1) Collect text captions and pass them through LM
        batch_captions = [caption[0] for caption in batch['captions']]
        # 4.2) Tokenize captions
        captions_encoded = self.tokenizer.batch_encode_plus(batch_captions,
                                                       padding='longest',
                                                       return_tensors='pt')
        # 4.3) Pass through LM
        if self.qformer_config.use_decoder_only_language_model:
            text_outputs = self.language_model(
                input_ids=captions_encoded['input_ids'],
                attention_mask=captions_encoded['attention_mask'],
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            inputs_embeds = self.language_model.get_input_embeddings()(captions_encoded['input_ids'])
            text_outputs = self.language_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=captions_encoded['attention_mask'],
                output_hidden_states=True,
                return_dict=True,
            )

        # obtain the the last_hidden_state from the T5 encoder output
        last_hidden_state = text_outputs.last_hidden_state  # shape is [batch_size, seq_len, hidden_size]

        # 4.4) Pass through part of full Q-Former model
        # (in this case - directly to input, without queries)
        text_outputs = self.t_qformer(last_hidden_state)
        # obtain the full caption embedding from the T5 as the last_hidden_state from the T5 encoder output
        last_hidden_state = text_outputs.last_hidden_state  # shape is [batch_size, seq_len, hidden_size]

        # 4.5) pool == sum/average the last_hidden_state
        pooled_last_hidden_state = torch.mean(last_hidden_state, dim=1)

        # 5) Project text tokens to shared dim for loss computation
        pooled_last_hidden_state = self.text_projection_layer(pooled_last_hidden_state)
        pooled_last_hidden_state = F.normalize(pooled_last_hidden_state, dim=-1)

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

    def _compute_retrieval_loss(self,
                                outputs: Dict[str, torch.Tensor],
                                ret_start_i: int, ret_end_i: int,
                                ret_embeddings: torch.Tensor,
                                output_hidden_states: Optional[bool] = False) -> Dict[str, torch.Tensor]:
        """
        Calculate contrastive retrieval loss based on InfoNCE loss implementation.
        Args:
            outputs: a LLM outputs, containing: 'logits', 'hidden_states', etc.
            ret_start_i: a starting index of transactions embeddings injection;
            ret_end_i: an ending index of transactions embeddings injection (non-inclusive);
            ret_embeddings: a reference embeddings (i.e. target embeddings);
            output_hidden_states: whether to output hidden_states for retrieval tokens;
            output_logits: whether to output logits for retrieval tokens;

        Returns:
            a dict, containing 'loss' and,  optionally, 'last_hidden_state' and 'ret_tokens_logits'.
        """