import torch
import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


class RNNClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        self._gru = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True, bidirectional=True)
        self.linear = ClassificationHead(input_size * 2, hidden_size)
    
    def forward(self, x):
        batch_size, _ , d = x.shape
        _, last_hidden = self._gru(x)

        last_hidden = torch.reshape(last_hidden.permute(1, 2, 0), shape=(batch_size, d * 2))

        return self.linear(last_hidden)

class NSPHead(nn.Module):
    def __init__(self, embedding_dim, embedding_projections):
        super().__init__()

        heads = []
        for elem in embedding_projections:
            head = nn.Linear(embedding_dim, embedding_projections[elem][0] + 1)
            heads.append(head)

        self.heads = nn.ModuleList(heads)

    def forward(self, x):
        res = []
        for m in self.heads:
            tmp = m(x)
            res.append(tmp)

        return res 