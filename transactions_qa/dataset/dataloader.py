import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from typing import Optional


from torch.nn.utils.rnn import pad_sequence
from datasets import IterableDataset as HFIterableDataset
import pytorch_lightning as pl


from .data_generator import batches_generator, batches_balanced_generator
from romashka.logging_handler import get_logger


class TransactionQADataset:

    def __init__(self, dataset,
                 generator_batch_size: Optional[int] = 1,
                 min_seq_len: Optional[int] = 50, max_seq_len: Optional[int] = 150,
                 seed: Optional[int] = 42,
                 buffer_size: Optional[int] = 10_000,
                 to_balance: Optional[bool] = False,
                 is_train: Optional[bool] = True,
                 shuffle: Optional[bool] = False, *args, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        self.seed = seed
        self.to_balance = to_balance
        self.generator_batch_size = generator_batch_size
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.logger = get_logger(
            name="dataloader"
        )

    def create_generator(self, dataset):
        if self.to_balance:
            self.logger.info(f"Running with balanced generator...")
            print(f"Running with balanced generator...")
            return batches_balanced_generator(
                dataset,
                batch_size=self.generator_batch_size,
                min_seq_len=self.min_seq_len, max_seq_len=self.max_seq_len,
                is_train=self.is_train
            )
        self.logger.info(f"Running with sequential generator...")
        print(f"Running with sequential generator...")
        return batches_generator(dataset,
                                 batch_size=self.generator_batch_size,
                                 min_seq_len=self.min_seq_len, max_seq_len=self.max_seq_len,
                                 is_train=self.is_train)

    def build_dataset(self):
        # Somehow it is important to pass dataset using gen_kwargs, because sharding is done using it
        dataset = HFIterableDataset.from_generator(self.create_generator, gen_kwargs={'dataset': self.dataset})
        if self.buffer_size > 0 and self.shuffle:
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
        assert batch[0]['num_features'].shape[1] == 1, "Incorrect output of dataloader"

        output['num_features'] = pad_sequence([d['num_features'].transpose(0, -1) for d in batch],
                                              # num_features x batch_size x seq_len
                                              batch_first=True).squeeze(2).permute(-1, 0, 1)
        output['cat_features'] = pad_sequence([d['cat_features'].transpose(0, -1) for d in batch],
                                              # cat_features x batch_size x seq_len
                                              batch_first=True).squeeze(2).permute(-1, 0, 1)
        output['meta_features'] = torch.cat([d['meta_features'] for d in batch], dim=1)  # meta_features x batch_size

        output['mask'] = pad_sequence([d['mask'].transpose(0, -1) for d in batch], batch_first=True).squeeze(2)
        output['app_id'] = torch.cat([d['app_id'] for d in batch])
        if 'real_num_features' in batch[0]:
            output['real_num_features'] = pad_sequence([d['real_num_features'].transpose(0, -1) for d in batch],
                                              # num_features x batch_size x seq_len
                                              batch_first=True).squeeze(2).permute(-1, 0, 1)

            if 'label' in batch[0]:
                output['label'] = torch.cat([d['label'] for d in batch])

            return output


class TransactionQADataModule(pl.LightningDataModule):
    def __init__(self, train_dataset_config: dict = None, val_dataset_config: dict = None):
        super().__init__()
        self.train_dataset_config = train_dataset_config
        self.val_dataset_config = val_dataset_config

        if train_dataset_config is not None:
            self.train_ds = TransactionQADataset(**train_dataset_config).build_dataset()

        if val_dataset_config is not None:
            self.val_ds = TransactionQADataset(**val_dataset_config).build_dataset()

    def train_dataloader(self):
        if self.train_dataset_config is None:
            raise KeyError("train_dataset_config is None")
        else:
            self.train_ds.set_epoch(self.trainer.current_epoch)
            return DataLoader(self.train_ds,
                              batch_size=self.train_dataset_config['batch_size'],
                              num_workers=self.train_dataset_config['num_workers'],
                              collate_fn=TransactionQADataset.collate_fn)

    def val_dataloader(self):
        if self.val_dataset_config is None:
            raise KeyError("train_dataset_config is None")
        else:
            return DataLoader(self.val_ds,
                              batch_size=self.val_dataset_config['batch_size'],
                              num_workers=self.val_dataset_config['num_workers'],
                              collate_fn=TransactionQADataset.collate_fn)
