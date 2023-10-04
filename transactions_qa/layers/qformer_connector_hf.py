import os
import torch
from torch import nn
from typing import Optional, Union, List, Dict, Any

from transformers import Blip2QFormerConfig, Blip2QFormerModel

from romashka.logging_handler import get_logger
from romashka.transactions_qa.layers.initialization import (init_xavier_uniform_layers,
                                                            init_linear)

DEFAULT_CONFIG = {
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
    "cross_attention_frequency": 2,
    "attention_probs_dropout_prob": 0.1,
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.1,
    "initializer_range": 0.02,
    "max_position_embeddings": 1024,
    "layer_norm_eps": 1e-12,
    "position_embedding_type": 'absolute',
    "cross_attention_frequency": 2,
}


class QFormerConnector(nn.Module):
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 vocab_size: int,
                 pad_token_id: int,
                 num_queries: Optional[int] = 32,
                 config: Optional[Union[Blip2QFormerConfig, Dict[str, Any]]] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu',
                 from_checkpoint: Optional[bool] = False):

        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )

        self.output_size = output_size  # output size of embedder model
        self.input_size = input_size  # input size of second model / -> output shape for last linear layer
        self.num_queries = num_queries

        self.config = dict()
        if config is None:
            self.config = DEFAULT_CONFIG
        else:
            self.config = config

        # Configure model dependent parameters
        # usually set to LLM hidden_size
        self.config['hidden_size'] = input_size if self.config.get('hidden_size') is None \
            else self.config.get('hidden_size')
        self.config['encoder_hidden_size'] = output_size  # equals to embedder output size
        self.config['vocab_size'] = vocab_size
        self.config['pad_token_id'] = pad_token_id

        if not isinstance(self.config, Blip2QFormerConfig):
            self.config = Blip2QFormerConfig(**self.config)

        self.device = device
        self._create_layers(from_checkpoint=from_checkpoint)

    def _create_layers(self, from_checkpoint: Optional[bool] = False ):
        try:
            self.qformer = Blip2QFormerModel(self.config)
            if from_checkpoint and hasattr(self.config, 'text_model_name'):
                self._init_from_text_checkpoint(getattr(self.config, 'text_model_name'))

            elif from_checkpoint and hasattr(self.config, 'connector_model_name_or_path'):
                self._init_from_pretrained_checkpoint(getattr(self.config, 'connector_model_name_or_path'))

            self.qformer.to(self.device)

            self.query_tokens_embeddings = torch.nn.Parameter(
                torch.zeros(1, self.num_queries, self.config.hidden_size)).to(self.device)
            self.lm_projection_layer = torch.nn.Linear(self.config.hidden_size,
                                                       self.input_size).to(self.device)
        except Exception as e:
            self._logger.error(f"Error occurred during Q-Former connector creation:\n{e}")
            raise AttributeError(f"Error occurred during Q-Former connector creation:\n{e}")

    def _init_from_text_checkpoint(self, ckpt_path: str):
        self._logger.info(f"Initializing connector Q-Former from checkpoint: {ckpt_path}")
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        else:
            raise RuntimeError(f"Checkpoint path is invalid or doesn't exist: {ckpt_path}")

        num_params_to_fill = len(
            [param for param_name, param in self.named_parameters()
             if ("crossattention" not in param_name)
             and (param_name not in ['query_tokens_embeddings', 'qformer.layernorm.weight', 'qformer.layernorm.bias',
                                     'lm_projection_layer.weight', 'lm_projection_layer.bias'])])
        num_params_to_fill_with = len(checkpoint)
        self._logger.info(f"Connector params to fill: {num_params_to_fill} vs. "
                          f"from checkpoint: {num_params_to_fill_with}")
        self.load_state_dict(checkpoint, strict=False)
        self._logger.info(f"Connector weights initialized from checkpoint: {ckpt_path}")

    def _init_from_pretrained_checkpoint(self, ckpt_path: str):
        self._logger.info(f"Initializing connector Q-Former from checkpoint: {ckpt_path}")
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        else:
            raise RuntimeError(f"Checkpoint path is invalid or doesn't exist: {ckpt_path}")

        self.load_state_dict(checkpoint, strict=False)
        self._logger.info(f"Connector weights initialized from checkpoint: {ckpt_path}")

    def forward(self, embeds: torch.Tensor, mask: torch.Tensor,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                return_dict: Optional[bool] = True,
                *args, **kwargs) -> torch.Tensor:

        # step 1: get embeddings -> done!
        # step 2: forward the query tokens through the QFormer, using input embeddings for cross-attention
        # embeds_attention_mask = torch.ones(embeds.size()[:-1], dtype=torch.long, device=embeds.device)
        

        query_tokens = self.query_tokens_embeddings.expand(embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=embeds,
            encoder_attention_mask=mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.lm_projection_layer(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        return language_model_inputs