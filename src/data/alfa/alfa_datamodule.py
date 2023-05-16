import os
import numpy as np
import torch
import torch.nn as nn
import random

from torch.utils.data import IterableDataset, DataLoader
from typing import Optional


from src.data.alfa.components.data_generator import batches_generator
from torch.nn.utils.rnn import pad_sequence
from datasets import IterableDataset as HFIterableDataset
import pytorch_lightning as pl
from .data_utils import split_process

class AlfaDataModule(pl.LightningDataModule):
    def __init__(self,
                data_dir: str,
                min_seq_len: Optional[int] = 50, 
                max_seq_len: Optional[int] = 150,
                seed: Optional[int] = 42,
                batch_size: Optional[int] = 32,
                val_batch_size: Optional[int] = 32,
                num_workers: Optional[int] = 1,
                pin_memory: Optional[bool] = False,
                shuffle: Optional[bool] = False,
        ):
        super().__init__()

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

        # TODO add checking validity of train data path
        data_files = {}
        dir_with_datasets = os.listdir(os.path.join(data_dir, 'train_buckets'))
        data_files["train"] = sorted([os.path.join(data_dir, 'train_buckets', x)
                                for x in dir_with_datasets])
        
        # logger.info(f"Detected {len(dataset_files)} files for training.")
       
        # TODO add checking validity of val data path
        shuffling_generator = self.shuffle
        dir_with_datasets = os.listdir(os.path.join(data_dir, 'val_buckets'))
        data_files["validation"] = sorted([os.path.join(data_dir, 'val_buckets', x)
                                for x in dir_with_datasets])
        # logger.info(f"Detected {len(dataset_files)} files for validation.")

        self.train_ds = self.create_data(list_paths=data_files["train"], shuffling_generator=self.shuffle)
        self.val_ds = self.create_data(list_paths=data_files["validation"])

    def create_data(self, list_paths, shuffling_generator=None):
        dataset = HFIterableDataset.from_generator(batches_generator,
                                                    gen_kwargs={
                                                        'list_of_paths': list_paths,
                                                        'min_seq_len': self.min_seq_len,
                                                        'max_seq_len': self.max_seq_len,
                                                        'shuffling_generator': shuffling_generator
                                                    }
        )
        # if self.buffer_size > 0:
        #     dataset = dataset.shuffle(seed=self.seed, buffer_size=self.buffer_size)
        
        return dataset.with_format('torch')

    def train_dataloader(self, pl_training=True):
        # if pl_training:
        #     self.train_ds.set_epoch(self.trainer.current_epoch)
        
        return DataLoader(self.train_ds, 
                            batch_size=self.batch_size,
                            num_workers=self.num_workers, 
                            collate_fn=self.collate_fn,
                            worker_init_fn=self.worker_init_fn,
                            persistent_workers=True)
        

    def val_dataloader(self):
        return DataLoader(self.val_ds, 
                            batch_size=self.batch_size,
                            num_workers=self.num_workers, 
                            collate_fn=self.collate_fn,
                            worker_init_fn=self.worker_init_fn,
                            persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.val_ds, 
                            batch_size=self.batch_size,
                            num_workers=self.num_workers, 
                            collate_fn=self.collate_fn,
                            worker_init_fn=self.worker_init_fn,
                            persistent_workers=True)

    @staticmethod
    def worker_init_fn(worker_id):
        seed = torch.utils.data.get_worker_info().id
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)       
                                                                                                                                    
        
    def collate_fn(self, batch):
        output = {}

        # cat_features shape 1 x cat_features x seq_len
        # num_features shape 1 x num_features x seq_len
        # meta_feature shape 1 x meta_features
        # mask shape 1 x seq_len
        # label shape 1

        # checking batch_size correctness
        assert batch[0]['num_features'].shape[1] == 1, "Incorrect output of dataloader"

        output['num_features'] = pad_sequence([d['num_features'].transpose(0, -1) for d in batch], # num_features x batch_size x seq_len
                                              batch_first=True).squeeze(2).permute(-1, 0, 1)
        output['cat_features'] = pad_sequence([d['cat_features'].transpose(0, -1) for d in batch], # cat_features x batch_size x seq_len
                                              batch_first=True).squeeze(2).permute(-1, 0, 1)
        output['meta_features'] = torch.cat([d['meta_features'] for d in batch], dim=1) # meta_features x batch_size

        output['mask'] = pad_sequence([d['mask'].transpose(0, -1) for d in batch], batch_first=True).squeeze(2)
        output['app_id'] = torch.cat([d['app_id'] for d in batch])

        if 'label' in batch[0]:
            output['label'] = torch.cat([d['label'] for d in batch])

        return output


class AlfaPretrainingDataModule(AlfaDataModule):
    def  __init__(self, splitter, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.splitter = splitter
        self.rep = splitter.split_count
        self.mode = mode

    def collate_fn(self, batch):
        batch = super().collate_fn(batch)
        len_batch = batch['num_features'][0].shape[0]
        labels = torch.arange(len_batch, device=batch['mask'].device).repeat(self.rep)
        batch = split_process(batch, self.splitter)
        
        if self.mode == 'coles':
            return batch, labels