from typing import Optional, Union, List, Dict, Any

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import BertTokenizer, PreTrainedTokenizerBase

from romashka.logging_handler import get_logger
from romashka.transactions_qa.layers.initialization import init_linear
from romashka.transactions_qa.layers.bert_layers import BertConfig, BertLMHeadModel
from romashka.transactions_qa.dist_utils import concat_all_gather, is_dist_avail_and_initialized


class QFormerModel(nn.Module):
    """
    Querying Transformer (Q-Former), used in BLIP-2.
    """

    def __init__(self,
                 text_model_name: str,
                 tokenizer: Optional[PreTrainedTokenizerBase] = None,
                 sequence_len: Optional[int] = 384,
                 num_queries: Optional[int] = 32,
                 shared_dim: Optional[int] = 256,
                 hidden_size: Optional[int] = 768,
                 num_hidden_layers: Optional[int] = 4,
                 num_attention_heads: Optional[int] = 4,
                 intermediate_size: Optional[int] = 512,
                 max_position_embeddings: Optional[int] = 1024,
                 cross_attention_frequency: Optional[int] = 2,
                 hidden_dropout_prob: Optional[float] = 0.1,
                 attention_probs_dropout_prob: Optional[float] = 0.1,
                 hidden_act: Optional[str] = 'gelu',
                 max_text_sequence_len: Optional[int] = 512,
                 initializer_range: Optional[float] = 0.02,
                 layer_norm_eps: Optional[float] = 1e-12,
                 truncation_side: Optional[str] = 'right',
                 position_embedding_type: Optional[str] = 'absolute',
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        super().__init__()

        self._logger = get_logger(
            name=self.__class__.__name__,
            logging_level="INFO"
        )
        self.text_model_name = text_model_name
        self.tokenizer = tokenizer
        self.sequence_len = sequence_len
        self.shared_dim = shared_dim  # a dimension for contrastive representation comparison
        self.num_queries = num_queries
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.cross_attention_frequency = cross_attention_frequency
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_text_sequence_len = max_text_sequence_len
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.truncation_side = truncation_side
        self.position_embedding_type = position_embedding_type
        self.device = device

        # Configure models
        # self.sequence_encoder_model = sequence_encoder_model
        self._prepare_model()

    def _prepare_model(self):
        """
        Creates submodules and initialize them.
        """
        # 1) Init tokenizer
        self._init_tokenizer()
        # 2) Create & init queries & projections
        self._create_queries()
        self._create_projections()
        # 3) Init main model part (as text backbone)
        self._init_model()

        assert (
                hasattr(self, 'qformer_model')
                and hasattr(self, 'tokenizer')
                and hasattr(self, 'query_tokens_embeddings')
                and hasattr(self, 'sequence_proj')
                and hasattr(self, 'text_proj')
        )

    def _init_model(self):
        """
        Initializes Q-Former encoder (as BertLMHeadModel) from pretrained weights.
        """
        encoder_config = BertConfig.from_pretrained(self.text_model_name)
        encoder_config.encoder_width = self.sequence_len

        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = self.cross_attention_frequency
        encoder_config.query_length = self.num_queries
        self.qformer_model = BertLMHeadModel.from_pretrained(self.text_model_name,
                                                             config=encoder_config)
        try:
            self.qformer_model.resize_token_embeddings(len(self.tokenizer))
        except Exception as e:
            print(f"Error occurred during Q-Former embeddings resize:\n{e}\n"
                  f"Continue without resize.")

    def _init_tokenizer(self):
        """
        Initializes tokenizer (as BertTokenizer) from pretrained model.
        """
        if self.tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(self.text_model_name,
                                                           truncation_side=self.truncation_side)
            self._logger.info(f"Tokenizer created and initialized from `{self.text_model_name}`")
        else:
            self._logger.info(f"Tokenizer is pre-initialized with vocabulary size `{len(self.tokenizer)}`")

    def _create_queries(self):
        """
        Creates selected number of queries.
        """
        self.query_tokens_embeddings = torch.nn.Parameter(
                torch.zeros((1, self.num_queries, self.hidden_size), device=self.device), requires_grad=True) #.to(self.device)
        self.query_tokens_embeddings.data.normal_(mean=0.0, std=self.initializer_range)
        self._logger.info(f"{self.num_queries} queries created.")

    def _create_projections(self):
        """
        Creates two projection layers:
         - from sequence encoder output -> shared dim;
         - from text model output -> shared dim;
        """
        self.sequence_proj = nn.Linear(self.hidden_size, self.shared_dim)
        init_linear(self.sequence_proj)
        self.text_proj = nn.Linear(self.hidden_size, self.shared_dim)
        init_linear(self.text_proj)
        # Seq2Text matching
        self.seq2text_matching_head = nn.Linear(self.hidden_size, 2)
        # Softmax temperature
        self.temperature = nn.Parameter(0.07 * torch.ones([]))

    def forward(self,
                sequence_embeds: torch.Tensor,
                text: Optional[Union[str, List[str]]] = None,
                output_attentions: Optional[bool] = False,
                is_train: Optional[bool] = False,
                *args, **kwargs) -> torch.Tensor:

        # step 1: get sequence embeddings -> done!
        # step 2: forward the query tokens through the QFormer, using input embeddings for cross-attention
        batch_size = sequence_embeds.size(0)
        device = sequence_embeds.device
        sequence_embeds_attention_mask = torch.ones(sequence_embeds.size()[:-1],
                                                    dtype=torch.long,
                                                    device=device)

        query_tokens = self.query_tokens_embeddings.expand(batch_size, -1, -1)

        # Pass through base model
        query_output = self.qformer_model.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=sequence_embeds,
            encoder_attention_mask=sequence_embeds_attention_mask,
            output_hidden_states=True,  # required for contrastive pretraining!
            output_attentions=output_attentions,  # this is optional
            use_cache=True,
            return_dict=True,
        )
        # Normalize & project
        sequence_features = F.normalize(
            self.sequence_proj(query_output.last_hidden_state), dim=-1
        )

        # Encode text
        if text is not None:
            text_tokens = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=self.max_text_sequence_len,
                return_tensors="pt",
            ).to(device)

            text_output = self.qformer_model.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                output_hidden_states=True,  # required for contrastive pretraining!
                output_attentions=output_attentions,  # this is optional
                return_dict=True
            )
            text_features = F.normalize(
                self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
            )
        outputs = dict(sequence_features=sequence_features)
        if text is not None:
            outputs['text_features'] = text_features

        if is_train and (text is not None):
            contrastive_loss = self.compute_contrastive_loss(sequence_features=sequence_features,
                                                             text_features=text_features,
                                                             device=device)
            outputs['loss'] = contrastive_loss

        return outputs

    def compute_contrastive_loss(self, sequence_features: torch.Tensor,
                                 text_features: torch.Tensor,
                                 device: torch.device) -> torch.Tensor:
        """
        Compute BLIP-2-style contrastive loss for sequence vs. text representations (in shared dimensionality).
        Args:
            sequence_features (torch.Tensor): a sequence features of size [bs, sequence_len, shared_dim];
            text_features (torch.Tensor): a text features of size [bs, max_text_seq_len, shared_dim];
            device (torch.device): a device for placing tensors on;
        Returns:
            a loss.
        """
        sequence_features_all = concat_all_gather(
            sequence_features
        )  # [batch_size * num_gpu, num_query_tokens, embed_dim]
        text_features_all = concat_all_gather(text_features)  # [batch_size * num_gpu, embed_dim]

        # Queries to text similarity
        sim_q2t = torch.matmul(
            sequence_features.unsqueeze(1), text_features_all.unsqueeze(-1)
        ).squeeze()
        # [batch_size, batch_size * num_gpu, num_query_tokens]

        # Sequence to text similarity: aggregate across all query tokens
        sim_s2t, _ = sim_q2t.max(-1)
        sim_s2t = sim_s2t / self.temperature

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_t2q = torch.matmul(
            text_features.unsqueeze(1).unsqueeze(1), sequence_features_all.permute(0, 2, 1)
        ).squeeze()

        # Text to sequence similarity: aggregate across all query tokens
        sim_t2s, _ = sim_t2q.max(-1)
        sim_t2s = sim_t2s / self.temperature  # [batch_size, batch_size * num_gpu]

        rank = dist.get_rank() if is_dist_avail_and_initialized() else 0
        batch_size = sequence_features.size(0)
        targets = torch.linspace(rank * batch_size, rank * batch_size + batch_size - 1, batch_size, dtype=int).to(
            device
        )

        # Total contrastive loss
        loss_contrastive = (F.cross_entropy(sim_s2t, targets)  # label_smoothing=0.1
                            + F.cross_entropy(sim_t2s, targets)  # label_smoothing=0.1
                            ) / 2

        return loss_contrastive
