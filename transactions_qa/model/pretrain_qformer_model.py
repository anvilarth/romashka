import random
import numpy as np
from copy import deepcopy
from typing import List, Optional, Tuple, Any, Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import transformers
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from transformers.models.blip_2.configuration_blip_2 import Blip2QFormerConfig

from romashka.logging_handler import get_logger
from romashka.transactions_qa.layers.qformer_text import Blip2QFormerTextEncoder
from romashka.transactions_qa.dist_utils import concat_all_gather

DEFAULT_CONFIG = {
    "num_queries": 32,
    "hidden_size": 768,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
    "intermediate_size": 3072,
    "cross_attention_frequency": 2,
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "max_position_embeddings": 1024,
    "max_text_sequence_len": 512,
    "truncation_side": "right",
    "position_embedding_type": "absolute",
    "use_decoder_only_language_model": False
}  # as BERT-base

# DEFAULT_CONFIG = {
#     "num_queries": 32,
#     "hidden_size": 512,
#     "num_attention_heads": 4,
#     "num_hidden_layers": 4,
#     "intermediate_size": 1024,
#     "cross_attention_frequency": 2,
#     "attention_probs_dropout_prob": 0.1,
#     "hidden_act": "gelu",
#     "hidden_dropout_prob": 0.1,
#     "initializer_range": 0.02,
#     "max_position_embeddings": 1024,
#     "max_text_sequence_len": 512,
#     "truncation_side": "right",
#     "position_embedding_type": "absolute",
#     "use_decoder_only_language_model": False
# }  # as BERT-mini


class PretrainQFormerModel(pl.LightningModule):
    def __init__(self,
                 language_model: nn.Module,
                 sequence_encoder_model: nn.Module,
                 tokenizer: transformers.PreTrainedTokenizerBase,
                 qformer: nn.Module,
                 do_freeze_seq_m: Optional[bool] = True,
                 do_freeze_lm: Optional[bool] = True,
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
        self.t_qformer = Blip2QFormerTextEncoder(self.qformer_config)  # a text part

        self.do_freeze_seq_m = do_freeze_seq_m
        self.do_freeze_lm = do_freeze_lm

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
        self.temp = nn.Parameter(0.07 * torch.ones([]))
        self._freeze()

        # Check total trainable parameters
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        self._logger.info(f"Totally trainable parameters: {len(trainable_parameters)} from {len(parameters)}")

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

    def _freeze(self):
        """
        Freeze sequence encoder and language model.
        """
        # Freezing some weights
        if self.do_freeze_seq_m:
            self.sequence_encoder_model.eval()
            self._logger.info(f"Freezing sequence encoder model's parameters...")
            for param in self.sequence_encoder_model.parameters():
                param.requires_grad = False

        if self.do_freeze_lm:
            self.language_model.eval()
            self._logger.info(f"Freezing language model's parameters...")
            for param in self.language_model.parameters():
                param.requires_grad = False

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

    def model_step(self, batch: Union[Dict[str, torch.Tensor], Any]) -> Any:
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
        query_outputs = self.q_qformer(
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
            inputs_embeds = self.language_model.get_input_embeddings()(
                captions_encoded['input_ids'].to(self.language_model.device)
            )
            text_outputs = self.language_model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=captions_encoded['attention_mask'].to(self.language_model.device),
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

        # 6) Calculate loss
        outputs = {
                    "query_output": query_output,
                    "pooled_last_hidden_state": pooled_last_hidden_state
                }
        loss_outputs = self._compute_loss(outputs)

        # join two output dicts
        for key, val in loss_outputs.items():
            outputs[key] = val

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

    def _compute_loss(self,
                      outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate contrastive retrieval loss based on InfoNCE loss implementation.
        Args:
            outputs: a LLM outputs, containing: 'logits', 'hidden_states', etc.
            output_hidden_states: whether to output hidden_states for retrieval tokens;

        Returns:
            a dict, containing 'loss' and,  optionally, 'last_hidden_state' and 'ret_tokens_logits'.
        """
        sequence_feats = outputs['query_output']
        text_feats = outputs['pooled_last_hidden_state']

        # Collect across GPUs
        sequence_feats = concat_all_gather(
            sequence_feats
        )  # [batch_size*num_gpu, num_query_tokens, embed_dim]

        text_feats = concat_all_gather(text_feats)  # [batch_size*num_gpu, embed_dim]

        # Queries - text similarity
        sim_q2t = torch.matmul(
            sequence_feats.unsqueeze(1), text_feats.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # Sequences - text similarity: aggregate across all query tokens
        sim_seq2t, _ = sim_q2t.max(-1)
        sim_seq2t = sim_seq2t / self.temp

        # Text - Queries similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_feats.unsqueeze(1).unsqueeze(1), sequence_feats.permute(0, 2, 1)
        ).squeeze()

        # Text - sequences similarity: aggregate across all query tokens
        sim_t2seq, _ = sim_t2q.max(-1)
        sim_t2seq = sim_t2seq / self.temp  # [batch_size, batch_size*num_gpu]

        try:
            rank = dist.get_rank()
        except:
            rank = 0
        bs = sequence_feats.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            sequence_feats.device
        )
        # Total loss
        loss_seq2text_contrastive = (F.cross_entropy(sim_seq2t, targets, label_smoothing=0.1)
                                     + F.cross_entropy(sim_t2seq, targets, label_smoothing=0.1)) / 2

        loss_outputs = dict(loss=loss_seq2text_contrastive)
        return loss_outputs
