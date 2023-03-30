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
    def __init__(self, dataset, min_seq_len: int = 50, max_seq_len: int = 150, seed: int = 42, buffer_size: int = 10_000, is_train: bool = True):
        super().__init__()
        self.dataset = dataset
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        self.seed = seed
        self.buffer_size = buffer_size
    
    def create_generator(self, dataset):
        return batches_generator(dataset, min_seq_len=self.min_seq_len, max_seq_len=self.max_seq_len, is_train=self.is_train)

    def build_dataset(self):
        # Somehow it is important to pass dataset using gen_kwargs, because sharding is done using it

        dataset = HFIterableDataset.from_generator(self.create_generator, gen_kwargs={'dataset': self.dataset})
        if self.buffer_size > 0:
            dataset = dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size) 
        return dataset.with_format('torch')
    
    @classmethod
    def collate_fn(cls, batch):
        output = {}

        # cat_features shape 1 x cat_features x seq_len
        # num_features shape 1 x num_features x seq_len
        # meta_feature shape 1 x meta_features
        # mask shape 1 x seq_len
        # label shape 1  

        # checking batch_size correctness
        assert batch[0]['num_features'].shape[1] == 1,  "Incorrect output of dataloader"

        output['num_features'] = pad_sequence([d['num_features'].transpose(0, -1) for d in batch], batch_first=True).squeeze(2).permute(-1, 0, 1)
        output['cat_features'] = pad_sequence([d['cat_features'].transpose(0, -1) for d in batch], batch_first=True).squeeze(2).permute(-1, 0, 1)
        output['meta_features'] = torch.cat([d['meta_features'] for d in batch], dim=1)

        output['mask'] = pad_sequence([d['mask'].transpose(0, -1) for d in batch], batch_first=True).squeeze(2)
        output['app_id'] = torch.cat([d['app_id'] for d in batch])

        if 'label' in batch[0]:
            output['label'] = torch.cat([d['label'] for d in batch])

        return output
