import torch
from torch import nn
from typing import Optional, Union, List

from transformers import Blip2QFormerConfig, Blip2QFormerModel
from romashka.transactions_qa.layers.initialization import (init_xavier_uniform_layers,
                                                            init_linear)

DEFAULT_CONFIG = {
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "intermediate_size": 1024,
    "hidden_act": 'gelu',
    "hidden_dropout_prob": 0.1,
    "attention_probs_dropout_prob": 0.1,
    "max_position_embeddings": 1024,
    "initializer_range": 0.02,
    "layer_norm_eps": 1e-12,
    "position_embedding_type": 'absolute',
    "cross_attention_frequency": 2,
}


class QFromerConnector(nn.Module):
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 vocab_size: int,
                 pad_token_id: int,
                 num_queries: Optional[int] = 32,
                 config: Optional[Blip2QFormerConfig] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        super().__init__()
        self.output_size = output_size  # output size of embedder model
        self.input_size = input_size  # input size of second model / -> output shape for last linear layer
        self.num_queries = num_queries

        self.config = dict()
        if config is None:
            self.config = DEFAULT_CONFIG
        else:
            self.config = config

        # Configure model dependent parameters
        self.config['hidden_size'] = input_size  # usually set to LLM hidden_size
        self.config['encoder_hidden_size'] = output_size  # equals to embedder output size
        self.config['vocab_size'] = vocab_size
        self.config['pad_token_id'] = pad_token_id

        self.device = device
        self._create_layers()

    def _create_layers(self):
        try:
            self.qformer = Blip2QFormerModel(self.config).to(self.device)

            self.query_tokens_embeddings = torch.nn.Parameter(
                torch.zeros(1, self.num_queries, self.config.hidden_size)).to(self.device)
            self.lm_projection_layer = torch.nn.Linear(self.config.hidden_size,
                                                       self.config.hidden_size).to(self.device)
        except Exception as e:
            print(f"Error occurred during Q-Former connector creation:\n{e}")
            raise AttributeError(f"Error occurred during Q-Former connector creation:\n{e}")

    def forward(self, embeds: torch.Tensor,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                *args, **kwargs) -> torch.Tensor:

        # step 1: get embeddings -> done!
        # step 2: forward the query tokens through the QFormer, using input embeddings for cross-attention
        embeds_attention_mask = torch.ones(embeds.size()[:-1], dtype=torch.long, device=embeds.device)

        query_tokens = self.query_tokens.expand(embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=embeds,
            encoder_attention_mask=embeds_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        return language_model_inputs