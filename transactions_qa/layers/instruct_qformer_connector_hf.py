import os
import torch
from torch import nn
from typing import Optional, Union,  Dict, Any

from romashka.transactions_qa.layers.instruct_qformer import InstructBlipQFormerConfig, InstructBlipQFormerModel

from romashka.logging_handler import get_logger

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


class InstructQFormerConnector(nn.Module):
    """
    Instruct Querying Transformer (Q-Former), used in InstructBLIP.
    """
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 vocab_size: Optional[int] = 30522,
                 pad_token_id: Optional[int] = 0,
                 num_queries: Optional[int] = 32,
                 config: Optional[Union[InstructBlipQFormerConfig, Dict[str, Any]]] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu',
                 from_checkpoint: Optional[bool] = False):
        """
        Instruct Querying Transformer (Q-Former), used in InstructBLIP.
        Args:
            output_size (`int`): a modality encoder output size;
            input_size (`int`): LLM input size;
            vocab_size (`int`, *optional*, defaults to 30522):
                Vocabulary size of the Q-Former model. Defines the number of different tokens that can be represented by
                the `inputs_ids` passed when calling the model.
            pad_token_id (`int`, *optional*, defaults to 0): a padding token id;
            num_queries (`int`, *optional*, defaults to 32): a number of trainable Q-Former queries;
            config (`InstructBlipQFormerConfig` or 'dict`, *optional*, defaults to DEFAULT_CONFIG): a Q-Former config;
            device (`torch.device` or `str`, *optional*, defaults to 'cpu'): a device to store model on;
            from_checkpoint (`bool`, *optional*, defaults to False): whether to load from checkpoint or not.
        """

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

        if not isinstance(self.config, InstructBlipQFormerConfig):
            self.config = InstructBlipQFormerConfig(**self.config)

        self.device = device
        self._create_layers(from_checkpoint=from_checkpoint)

    def _create_layers(self, from_checkpoint: Optional[bool] = False ):
        try:
            self.qformer = InstructBlipQFormerModel(self.config)
            if from_checkpoint and hasattr(self.config, 'text_model_name'):
                self._init_from_checkpoint(getattr(self.config, 'text_model_name'))
            self.qformer.to(self.device)

            self.query_tokens_embeddings = torch.nn.Parameter(
                torch.zeros(1, self.num_queries, self.config.hidden_size)).to(self.device)
            self.lm_projection_layer = torch.nn.Linear(self.config.hidden_size,
                                                       self.input_size).to(self.device)
        except Exception as e:
            self._logger.error(f"Error occurred during Q-Former connector creation:\n{e}")
            raise AttributeError(f"Error occurred during Q-Former connector creation:\n{e}")

    def _init_from_checkpoint(self, ckpt_path: str):
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

    def forward(self,
                embeds: torch.Tensor,
                input_text_ids: torch.Tensor,
                input_text_attention_mask: Optional[torch.LongTensor] = None,
                output_attentions: Optional[bool] = False,
                output_hidden_states: Optional[bool] = False,
                return_dict: Optional[bool] = True,
                *args, **kwargs) -> torch.Tensor:

        # step 1: get embeddings -> done!
        # step 2: forward the query tokens through the QFormer, using input embeddings for cross-attention
        embeds_attention_mask = torch.ones(embeds.size()[:-1], dtype=torch.long, device=embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens_embeddings.expand(embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=embeds.device)
        if input_text_attention_mask is None:
            input_text_attention_mask = torch.ones_like(input_text_ids)

        # concatenate two masks as: [queries att mask, input_ids att mask]
        embeds_attention_mask = torch.cat([query_attention_mask, input_text_attention_mask], dim=1)

        query_outputs = self.qformer(
            input_ids=input_text_ids,
            attention_mask=embeds_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=embeds,
            encoder_attention_mask=embeds_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.lm_projection_layer(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        return language_model_inputs