import torch 
import torch.nn as nn

class NextTransactionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, logits, labels):
        res = []
        for (pred, true) in zip(logits, labels):
            res.append(self.criterion(pred.permute(0, 2, 1), true))
        return sum(res)