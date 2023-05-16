import torch.nn as nn

from abc import ABC, abstractmethod
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from src.models.components.my_utils import get_projections_maps, cat_features_names, num_features_names, meta_features_names
from src.models.components.models import TransactionsModel

class EmbeddingPlusConnector(nn.Module):
    def __init__(self, embedding_layer, connector):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.connector = connector
        self.output_size = connector.output_size

    def forward(self, batch):
        mask = batch['mask']
        
        embedding = self.embedding_layer(batch)
        embedding = self.connector(embedding, attention_mask=mask)
        return embedding

class MySeqEncoder(SeqEncoderContainer):
    def __init__(self,         
                encoder_type='whisper/tiny',
                head_type='pretraining_last_output'
        ):

        #TODO Fix this
        folder = '/home/jovyan/romashka'
        projections_maps = get_projections_maps(relative_folder=folder)
        transactions_model_config = {
            "cat_features": cat_features_names,
            "cat_embedding_projections": projections_maps.get('cat_embedding_projections'),
            "num_features": num_features_names,
            "num_embedding_projections": projections_maps.get('num_embedding_projections'),
            "meta_features": meta_features_names,
            "meta_embedding_projections": projections_maps.get('meta_embedding_projections'),
            "encoder_type": encoder_type,
            "head_type": head_type,
        }
        super().__init__(
            trx_encoder=None,
            seq_encoder_cls=TransactionsModel,
            input_size=False,
            seq_encoder_params=transactions_model_config,
            is_reduce_sequence=False,
        )
        self.trx_encoder = EmbeddingPlusConnector(self.seq_encoder.embedding, self.seq_encoder.connector)
                
        self.full_model = self.seq_encoder
        self.seq_encoder = self.seq_encoder.encoder_model
        self.seq_encoder.embedding_size = self.seq_encoder.output_size
    
    def forward(self, x, h_0=None):
        x = self.full_model(x)
        return x