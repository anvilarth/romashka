import os
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import IterableDataset
from romashka.data_generators import batches_generator

import pytorch_lightning as pl


class TransactionQADataset(IterableDataset):
    def __init__(self, dataset_train, shuffle=True, batch_size=64, device='cuda', min_seq_len=50, max_seq_len=150,
                 wrap_with_question=None):
        self.data = dataset_train
        self.batch_size = batch_size
        self.device = device
        self.foo = lambda: batches_generator(self.data, batch_size=self.batch_size, shuffle=shuffle,
                                             device=self.device,
                                             is_train=True,
                                             output_format='torch',
                                             min_seq_len=min_seq_len,
                                             max_seq_len=max_seq_len)

    def __iter__(self):
        return self.foo()
