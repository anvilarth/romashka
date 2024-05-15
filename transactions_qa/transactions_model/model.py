import sys
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any

from romashka.transactions_qa.layers.embedding import EmbeddingLayer
from romashka.transactions_qa.transactions_model.head import TransactionHead
from romashka.transactions_qa.transactions_model.encoder import TransactionEncoder
from romashka.transactions_qa.transactions_model.connectors import TransactionConnector

from romashka.logging_handler import get_logger


class WrapVisionModel(nn.Module):
    def __init__(self, process, encoder):
        super().__init__()
        self.process = process
        self.encoder = encoder

    def forward(self, x, attention_mask=None):
        x = self.process(x, mask=attention_mask)
        return self.encoder(x)


class TransactionsModel(nn.Module):
    def __init__(self,
                 cat_embedding_projections: Dict[str, Tuple[int, int]],
                 cat_features: List[str],
                 cat_bias: Optional[bool] = True,
                 num_embedding_projections: Optional[Dict[str, Tuple[int, int]]] = None,
                 num_features: Optional[List[str]] = None,
                 num_embeddings_type: Optional[str] = 'linear',
                 numeric_bins: Optional[Dict[str, Any]] = None,
                 num_embeddings_kwargs: Optional[Dict[str, Any]] = {},
                 meta_embedding_projections: Optional[Dict[str, Tuple[int, int]]] = None,
                 meta_features: Optional[List[str]] = None,
                 time_embedding: Optional[bool] = None,
                 head_type: Optional[str] = 'linear',
                 encoder_type: Optional[str] = 'gpt2/base',
                 num_layers: Optional[int] = 1,
                 embedding_dropout: Optional[float] = 0.0,
                 cutmix: Optional[bool] = False,
                 mixup: Optional[bool] = False,
                 pretrained: Optional[bool] = False,
                 hidden_size: Optional[bool] = None,
                 alpha: Optional[float] = 1.0,
                 *args,
                 **kwargs):

        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__
        )

        self.embedding = EmbeddingLayer(cat_embedding_projections=cat_embedding_projections,
                                        cat_features=cat_features,
                                        cat_bias=cat_bias,
                                        num_embedding_projections=num_embedding_projections,
                                        num_features=num_features,
                                        numeric_bins=numeric_bins,
                                        num_embeddings_type=num_embeddings_type,
                                        num_embeddings_kwargs=num_embeddings_kwargs,
                                        meta_embedding_projections=meta_embedding_projections,
                                        meta_features=meta_features,
                                        time_embedding=time_embedding,
                                        dropout=embedding_dropout)
        inp_size = self.embedding.get_embedding_size()

        self.encoder_model = TransactionEncoder(encoder_type,
                                                inp_size,
                                                hidden_size,
                                                pretrained,
                                                num_layers)

        self.output_size = self.encoder_model.get_output_size()
        self.connector_type = self.encoder_model.get_connector_type()
        self.numeric_bins = numeric_bins

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.output_size))
        self.connector = TransactionConnector(inp_size, self.output_size, self.connector_type)

        self.head_type = head_type
        self.head = TransactionHead(head_type,  self.output_size, cat_embedding_projections, num_embedding_projections)

        self.cutmix = cutmix
        self.mixup = mixup
        self.alpha = alpha

    def get_base_embeddings(self, batch):
        mask = batch['mask']
        embedding = self.embedding(batch)
        embedding = self.connector(embedding, attention_mask=mask)
        return embedding

    def get_embs(self, batch, embeds=None):
        mask = batch['mask']
        batch_size = mask.shape[0]
        if embeds is None:
            embedding = self.get_base_embeddings(batch)
        else:
            embedding = embeds
        x, mask = self.encoder_model(embedding, mask)
        return x, mask

    def forward(self, batch=None, embeds=None):
        x, mask = self.get_embs(batch, embeds)
        logit = self.head(x, mask)
        return logit


class LastPrediction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        pass