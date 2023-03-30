import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import IterableDataset
from romashka.data_generators import batches_generator
from torch.nn.utils.rnn import pad_sequence
from datasets import IterableDataset as HFIterableDataset

import pytorch_lightning as pl

class TransactionQADataset():
    def __init__(self, dataset, min_seq_len=50, max_seq_len=150):
        super().__init__()
        self.dataset = dataset
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
    
    def create_generator(self, dataset):
        return batches_generator(dataset, min_seq_len=self.min_seq_len, max_seq_len=self.max_seq_len)

    def build_dataset(self, seed=42, buffer_size=10_000):
        # Somehow it is important to pass dataset using gen_kwargs, because sharding is done using it

        dataset = HFIterableDataset.from_generator(self.create_generator, gen_kwargs={'dataset': self.dataset})
        if buffer_size is not None:
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size) 
        return dataset.with_format('torch')
    
    @classmethod
    def collate_fn(cls, batch):
        output = {}

        output['num_features'] = pad_sequence([d['num_features'].transpose(0, -1) for d in batch], batch_first=True).squeeze(2).permute(-1, 0, 1)
        output['cat_features'] = pad_sequence([d['cat_features'].transpose(0, -1) for d in batch], batch_first=True).squeeze(2).permute(-1, 0, 1)
        output['meta_features'] = torch.cat([d['meta_features'] for d in batch], dim=1)

        output['mask'] = pad_sequence([d['mask'].transpose(0, -1) for d in batch], batch_first=True).squeeze(2)
        output['app_id'] = torch.cat([d['app_id'] for d in batch])
        output['label'] = torch.cat([d['label'] for d in batch])

        return output
