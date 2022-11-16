import torch 
import pandas as pd
import torch.nn as nn

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2Model
from head import ClassificationHead

class ClickstreamDataset(Dataset):
    def __init__(self, data_path, label_path):
        super().__init__()
        # self.data = pd.read_csv(path, header=None, index_col=0).values
        self.data = torch.load(data_path)
        self.labels = torch.load(label_path).float() / 682

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)


class ClickstreamModel(nn.Module):
    def __init__(self, model_type, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(10, 32)
        self.model_type = model_type 

        if model_type == 'rnn':
            self.model = nn.GRU(input_size=32, hidden_size=32, batch_first=True)
        else:
            configuration = GPT2Config(vocab_size=1, n_positions=184, n_embd=32, n_layer=num_layers, n_head=2)
            self.model = GPT2Model(configuration) 
        
        self.head = nn.Linear(32, 1)

    def forward(self, x, mask=None):
        embedding = self.embedding(x)
        if self.model_type == 'rnn':
            out, _ = self.model(embedding)    
        else:
            out = self.model(inputs_embeds=embedding, attention_mask=mask).last_hidden_state
        
        return self.head(out) 