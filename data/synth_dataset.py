import torch
import torch.nn as nn
from torch.utils.data import Dataset

perm = torch.load('/home/jovyan/data/demo/indices.pt')
length = round(len(perm) * 0.2)
train_indices = perm[:-length]
test_indices = perm[-length:]

class TransactionDataset(Dataset):
    def __init__(self, train=True):
        if train:
            self.num_features = torch.load('/home/jovyan/data/demo/num.pt')[train_indices]
            self.labels  = torch.load('/home/jovyan/data/demo/labels.pt')[train_indices]
            self.cat_features = torch.load('/home/jovyan/data/demo/cat.pt')[train_indices]
            self.mask = self.cat_features[..., -6] != 0
        else:
            self.num_features = torch.load('/home/jovyan/data/demo/num.pt')[test_indices]
            self.labels = torch.load('/home/jovyan/data/demo/labels.pt')[test_indices]
            self.cat_features = torch.load('/home/jovyan/data/demo/cat.pt')[test_indices]
            self.mask = self.cat_features[..., -6] != 0
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {'num_features': self.num_features[idx], 'cat_features': self.cat_features[idx], 'label': self.labels[idx], 'mask':self.mask[idx]}
    
    @staticmethod
    def collate_fn(batch):
        num_features = [elem['num_features'] for elem in batch]
        num_data = list(torch.stack(num_features).permute(2, 0, 1))

        cat_features = [elem['cat_features'] for elem in batch]
        cat_data = list(torch.stack(cat_features).permute(2, 0, 1))

        labels = torch.stack([elem['label'] for elem in batch])
        mask = torch.stack([elem['mask'] for elem in batch])

        return {'num_features': num_data, 'cat_features': cat_data, 'mask':mask, 'label': labels}