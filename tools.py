import os
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import numpy as np 
import random

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def freeze_model(model):
    pass

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
