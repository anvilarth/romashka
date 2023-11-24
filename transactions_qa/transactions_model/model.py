import sys
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from romashka.transactions_qa.layers.embedding import EmbeddingLayer
from romashka.transactions_qa.transactions_model.head import TransactionHead
from romashka.transactions_qa.transactions_model.encoder import TransactionEncoder
from romashka.transactions_qa.transactions_model.connectors import TransactionConnector


class TransactionsModel(nn.Module):
    def __init__(self,
                 cat_embedding_projections: Dict[str, Any],
                 cat_features: List[str],
                 num_embedding_projections: Dict[str, Any] = None,
                 num_features: List[str] = None,
                 meta_embedding_projections: Dict[str, Any] = None,
                 meta_features: List[str] = None,
                 time_embedding: Optional[Any] = None,
                 head_type: Optional[str] = 'linear',
                 encoder_type: Optional[str] = 'gpt2/base',
                 num_layers: Optional[int] = 1,
                 embedding_dropout: Optional[float] = 0.0,
                 pretrained: Optional[bool] = False,
                 hidden_size: Optional[int] = None,
                 *args,
                 **kwargs):

        super().__init__()
        self.embedding = EmbeddingLayer(cat_embedding_projections,
                                        cat_features,
                                        num_embedding_projections,
                                        num_features,
                                        meta_embedding_projections,
                                        meta_features,
                                        time_embedding,
                                        dropout=embedding_dropout)
        self.input_size = self.embedding.get_embedding_size()

        self.encoder_model = TransactionEncoder(encoder_type,
                                                self.input_size,
                                                hidden_size,
                                                pretrained,
                                                num_layers,
        )
        self.output_size = self.encoder_model.get_output_size()
        self.connector_type = self.encoder_model.get_connector_type()
        # todo: add it to history?
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.output_size))
        self.connector = TransactionConnector(self.input_size, self.output_size, self.connector_type)

        self.head_type = head_type
        self.head = TransactionHead(head_type,  self.output_size,
                                    cat_embedding_projections,
                                    num_embedding_projections)

    def _embed(self, batch):
        mask = batch['mask']
        embedding = self.embedding(batch)
        embedding = self.connector(embedding, attention_mask=mask)
        return embedding

    def encode(self, batch, embeds: Optional[torch.TensorType] = None):
        mask = batch['mask']
        batch_size = mask.shape[0]
        if embeds is None:
            embedding = self._embed(batch)
        else:
            embedding = embeds
        x, mask = self.encoder_model(embedding, mask)
        return x, mask

    def forward(self, batch, embeds: Optional[torch.TensorType] = None):
        x, mask = self.encode(batch, embeds)
        logit = self.head(x, mask)
        return logit