import os
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import numpy as np 
import random

import time
from copy import deepcopy

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def zero_function(x):
    return 0
    
def freeze_model(model):
    pass

def masked_mean(inp, mask, axis=1):
    down = mask.sum(axis)
    out = (inp * mask).sum(axis) / down
    return out

class LambdaLayer(nn.Module):
    def __init__(self, function):
        super().__init__()
        
        self.function = function
    
    def forward(self, x, *args, **kwargs):
        return self.function(x)
    
    
def calculate_embedding_size(model):
    size = 0
    for module in model.modules():
        if type(module) == nn.LayerNorm:
            size = module.weight.shape[0]
    
    if size == 0:
        raise KeyError
    
    return size
    
def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                     num_parts_to_read: int = 2, columns=None, verbose=False) -> pd.DataFrame:
    """
    читает num_parts_to_read партиций, преобразует их к pd.DataFrame и возвращает
    :param path_to_dataset: путь до директории с партициями
    :param start_from: номер партиции, с которой начать чтение
    :param num_parts_to_read: количество партиций, которые требуется прочитать
    :param columns: список колонок, которые нужно прочитать из партиции
    :return: pd.DataFrame
    """

    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset) 
                              if filename.startswith('part')])
    
    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in tqdm.tqdm_notebook(chunks, desc="Reading dataset with pandas"):
        chunk = pd.read_parquet(chunk_path,columns=columns)
        res.append(chunk)
    return pd.concat(res).reset_index(drop=True)


def generate_subsequences(batch, K, m = 0.1, M=0.9):
    new_batches = [batch]
    seq_len = batch['mask'].shape[1]
    
    length = torch.rand(size=(K,)) * (M - m) + m
    int_length = (length * seq_len).type(torch.int)

    indices = seq_len - int_length
    start_indices = (indices * torch.rand_like(length)).type(torch.int)

    for (start, end) in zip(start_indices, start_indices + int_length): 
        new_batch = {}
        for elem in batch:
            new_feature_list = []
            if elem == 'label':
                new_batch[elem] = batch[elem]

            elif type(batch[elem]) == torch.Tensor:
                new_batch[elem] = batch[elem][:, start:end]

            elif elem == 'meta_features':
                for feature in batch[elem]:
                    new_feature_list.append(feature)
                new_batch[elem] = new_feature_list

            elif elem == 'app_id':
                pass

            else:
                for feature in batch[elem]:

                    new_feature = feature[:, start: end]
                    new_feature_list.append(new_feature)

                new_batch[elem] = new_feature_list
        new_batches.append(new_batch)
        
    return new_batches

def make_time_batch(batch, number_days=30):
    device = batch['mask'].device
    time_tr = batch['num_features'][1] * 365
    
    pairwise_difference_mask = abs(time_tr.unsqueeze(1) - time_tr.unsqueeze(2)) <= number_days
    last_elements_mask = time_tr >= number_days
    
    last_elements_repeated = last_elements_mask.unsqueeze(2).repeat(1, 1, time_tr.shape[1])
    tmp_mask = pairwise_difference_mask * last_elements_repeated
    
    matrix = torch.ones(time_tr.shape[1], time_tr.shape[1])
    autoregressive_mask = torch.triu(matrix, 1).unsqueeze(0).to(device)
    final_mask = tmp_mask * autoregressive_mask
    
    num_repeated = batch['num_features'][0].unsqueeze(1).repeat(1, time_tr.shape[1], 1)
    
    all_amnt_transactions = (num_repeated * final_mask).sum(2)
    all_num_transactions = final_mask.sum(2).float()
    
    cat_repeated = batch['cat_features'][11].unsqueeze(1).repeat(1, time_tr.shape[1], 1) * final_mask
    
    res = []
    for i in range(28):
        res.append(torch.any(cat_repeated == i,  dim=2))
    all_code_transactions = torch.stack(res, dim=-1).float()
    
    next_time_mask = torch.any(tmp_mask, dim=2).long()
    
    return all_amnt_transactions,  all_num_transactions, all_code_transactions, last_elements_mask.long()


def next_time_batch(batch, number_days=30):
    
    num_features = batch['num_features']
    cat_features = batch['cat_features']
    mask = batch['mask']
    
    next_time_mask = torch.zeros_like(mask, dtype=torch.long)
    
    all_num_transactions = torch.zeros_like(mask, dtype=torch.float)
    all_amnt_transactions = torch.zeros_like(mask, dtype=torch.float)
    all_code_transactions = torch.zeros(mask.shape[0], mask.shape[1], 28).to(all_amnt_transactions.device)

    for i in range(len(mask)):
        j = 0
        time_tr = num_features[1][i] * 365

        while j < len(time_tr):
            prost = torch.zeros(28)
            if time_tr[j] <= number_days:
                break

            k = 1
            tmp_amnt = 0.0

            while True:
                if j + k >= len(time_tr) or (time_tr[j] - time_tr[j+k] > number_days) or (time_tr[j+k] == 0.0):
                    break
                tmp_amnt += num_features[0][i][j+k].item()
                all_code_transactions[i][j][cat_features[11][i][j+k]] = 1
                k += 1


            all_amnt_transactions[i][j] = tmp_amnt

            # while True:
            #     if tmp[i] - tmp[i+k] > 7:
            #         break
            #     k += 1

            all_num_transactions[i][j] = k-1
            next_time_mask[i][j] = 1
            j += 1
            
    return all_amnt_transactions,  all_num_transactions, all_code_transactions, next_time_mask


class EMA(pl.Callback):

    def __init__(self, decay=0.9999, update_period=1, rm_modules=[], ema_device=None, pin_memory=True):
        self.decay = decay
        self.update_period = update_period
        self.ema_device = ema_device
        self.ema_pin_memory = pin_memory if torch.cuda.is_available() else False

        self.rm_modules = rm_modules
        self.ema_state_dict = {}
        self.original_state_dict = {}
        self._ema_state_dict_ready = False

    def get_state_dict(self, pl_module):
        state_dict = pl_module.state_dict()
        ema_state_dict = deepcopy(state_dict)
        for key in state_dict.keys():
            for rm_module in self.rm_modules:
                if key.startswith(rm_module):
                    ema_state_dict.pop(key)
        return ema_state_dict

    @rank_zero_only
    def on_train_start(self, trainer, pl_module):
        if not self._ema_state_dict_ready:
            self.ema_state_dict = self.get_state_dict(pl_module)
            if self.ema_device:
                self.ema_state_dict = {
                    k: tensor.to(self.ema_device) for k, tensor in self.ema_state_dict.items()
                }
            if self.ema_device == "cpu" and self.ema_pin_memory:
                self.ema_state_dict = {
                    k: tensor.pin_memory() for k, tensor in self.ema_state_dict.items()
                }
        elif self._ema_state_dict_ready:
            for key, value in self.get_state_dict(pl_module).items():
                self.ema_state_dict[key] = self.ema_state_dict[key].to(value.device)
        self._ema_state_dict_ready = True

    @rank_zero_only
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.update_period == 0:
            for key, value in self.get_state_dict(pl_module).items():
                ema_state_value = self.ema_state_dict[key]
                if self.ema_device is not None:
                    value = value.to(self.ema_device)
                if value.dtype == torch.float32:
                    ema_state_value.detach().mul_(self.decay).add_(value, alpha=1. - self.decay)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        return {
            "ema_state_dict": self.ema_state_dict,
            "ema_state_dict_ready": self._ema_state_dict_ready
        }

    def on_load_checkpoint(self, trainer, pl_module, callback_state):
        self._ema_state_dict_ready = callback_state["ema_state_dict_ready"]
        self.ema_state_dict = callback_state["ema_state_dict"]