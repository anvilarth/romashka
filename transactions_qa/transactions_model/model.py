import sys
import torch
import torch.nn as nn
from typing import Optional

from romashka.transactions_qa.layers.embedding import EmbeddingLayer
from romashka.transactions_qa.layers import MixedPrecisionLayerNorm, ProjectionsType
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
                 cat_embedding_projections,
                 cat_features,
                 num_embedding_projections=None,
                 num_features=None,
                 meta_embedding_projections=None,
                 meta_features=None,
                 time_embedding=None,
                 add_projection: Optional[bool] = False,
                 projection_type: Optional[str] = None,
                 shared_dim: Optional[int] = None,
                 add_l_norm: Optional[bool] = False,
                 head_type: Optional[str] = 'linear',
                 encoder_type: Optional[str] = 'gpt2/base',
                 num_layers: Optional[int] = 1,
                 embedding_dropout: Optional[float] = 0.0,
                 cutmix: Optional[bool] = False,
                 mixup: Optional[bool] = False,
                 emb_mult: Optional[int] = 1,
                 rel_pos_embs: Optional[bool] = False,
                 pretrained: Optional[bool] = False,
                 adapters: Optional[bool] = False,
                 hidden_size: Optional[bool] = None,
                 alpha: Optional[float] = 1.0,
                 *args,
                 **kwargs):

        super().__init__()
        self._logger = get_logger(
            name=self.__class__.__name__
        )

        self.embedding = EmbeddingLayer(cat_embedding_projections,
                                        cat_features,
                                        num_embedding_projections,
                                        num_features,
                                        meta_embedding_projections,
                                        meta_features,
                                        time_embedding,
                                        dropout=embedding_dropout)
        inp_size = self.embedding.get_embedding_size()

        self.encoder_model = TransactionEncoder(encoder_type,
                                                inp_size,
                                                hidden_size,
                                                pretrained,
                                                num_layers)

        self.output_size = self.encoder_model.get_output_size()
        self.connector_type = self.encoder_model.get_connector_type()

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.output_size))
        self.connector = TransactionConnector(inp_size, self.output_size, self.connector_type)

        self.head_type = head_type
        self.head = TransactionHead(head_type,  self.output_size, cat_embedding_projections, num_embedding_projections)

        self.cutmix = cutmix
        self.mixup = mixup
        self.alpha = alpha

        if add_l_norm:
            if shared_dim is None:
                shared_dim = self.output_size
                self._logger.error(f"`shared_dim` is obligatory parameter for Layer Norm creation."
                                   f"Specified default value = `{self.output_size}`")
            self.encoder_l_norm = MixedPrecisionLayerNorm(normalized_shape=(shared_dim))
        else:
            self.encoder_l_norm = None
        if add_projection:
            # Projection: trns_encoder_dim -> shared dim
            # 768 -> 768
            if shared_dim is None:
                shared_dim = self.output_size
                self._logger.error(f"`shared_dim` is obligatory parameter for projection layer creation."
                                   f"Specified default value = `{self.output_size}`")
            proj_kwargs = {'in_dim': self.output_size, "out_dim": shared_dim}
            if projection_type is None:
                projection_type = 'IDENTITY'
                self._logger.error(f"`projection_type` is obligatory parameter for projection layer creation."
                                   f"Specified default value = `{projection_type}`")
            self.encoder_projection = ProjectionsType.get(projection_type, **proj_kwargs)
        else:
            self.encoder_projection = None

        self.shared_dim = shared_dim
        self.add_projection = add_projection
        self.projection_type = projection_type

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

        if self.encoder_projection is not None:
            x = self.encoder_projection(x * mask.unsqueeze(-1))
        if self.encoder_l_norm is not None:
            x = self.encoder_l_norm(x * mask.unsqueeze(-1))

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