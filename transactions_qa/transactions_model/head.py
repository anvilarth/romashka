import torch
import torch.nn as nn


class TransactionHead(nn.Module):
    def __init__(self, head_type, input_size, cat_embedding_projections, num_embedding_projections):
        super().__init__()

        if head_type == 'linear':
            self.head = LinearHead(input_size)
        elif head_type == 'rnn':
            self.head = RNNClassificationHead(input_size)
        elif head_type == 'mlp':
            self.head = MLPHead(input_size)
        elif head_type == 'transformer':
            self.head = TransformerHead(input_size)
        elif head_type == 'id':
            self.head = IdentityHead(input_size)
        elif head_type == 'next':
            # TODO fix kwargs
            self.head = NSPHead(input_size, cat_embedding_projections, num_embedding_projections)
        elif head_type == 'next_time':
            self.head = NextActionsHead(input_size)
        elif head_type == 'last_output':
            self.head = LastOutputHead()
        else:
            raise NotImplementedError

    def forward(self, x, mask=None):
        return self.head(x, mask)


class LastOutputHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, mask):
        device = mask.device
        batch_size = mask.shape[0]
        trx_index = mask.sum(1) - 1
        output = x[torch.arange(batch_size, device=device), trx_index]
        return output


class LinearHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.linear1 = nn.Linear(input_size, 1)

    def forward(self, x, mask=None):
        x = x[:, -1]
        return self.linear1(x).squeeze(1)


class MLPHead(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.act = nn.GELU()

    def forward(self, x, mask=None):
        x = x[:, -1]
        x = self.act(self.linear1(x))
        x = self.act(self.linear2(x))
        x = self.linear3(x)

        return x


class IdentityHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.id = nn.Identity()

    def forward(self, x, mask=None):
        return self.id(x)


class TransformerHead(nn.Module):
    def __init__(self, input_size, num_layers=2):
        super().__init__()
        self.transformer_encoder = BERT(input_size, heads=2, num_layers=num_layers,
                                        dropout=0.0, layer_norm_eps=1e-7, rel_pos_embs=False)
        self.linear = LinearHead(input_size)

    def forward(self, x, mask=None):
        x = self.transformer_encoder(x, mask.unsqueeze(1).unsqueeze(2))
        return self.linear(x)


class RNNClassificationHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self._gru = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(input_size * 2, 1)

    def forward(self, x, mask=None):
        batch_size, _, d = x.shape
        _, last_hidden = self._gru(x)
        last_hidden = torch.reshape(last_hidden.permute(1, 2, 0), shape=(batch_size, d * 2))

        return self.linear(last_hidden)


class NextActionsHead(nn.Module):
    def __init__(self,
                 embedding_dim,
                 ):
        super().__init__()

        self.amnt_head = nn.Linear(embedding_dim, 1)
        self.num_head = nn.Linear(embedding_dim, 1)
        self.need_head = nn.Linear(embedding_dim, 28)

    def forward(self, x, mask=None):
        amnt_out = self.amnt_head(x)
        num_out = self.num_head(x)
        need_out = self.need_head(x)

        return [amnt_out, num_out, need_out]


class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, n_classes):
        super().__init__()
        self.head = nn.Linear(embedding_dim, n_classes)

    def forward(self, x, mask=None):
        x = x[:, -1]
        return self.head(x)


class NSPHead(nn.Module):
    def __init__(self,
                 embedding_dim,
                 cat_embedding_projections,
                 num_embedding_projections=None
                 ):

        super().__init__()

        cat_heads = []
        for elem in cat_embedding_projections:
            head = nn.Linear(embedding_dim, cat_embedding_projections[elem][0] + 1)
            cat_heads.append(head)

        self.cat_heads = nn.ModuleList(cat_heads)

        num_heads = []
        self.num_heads = None

        if num_embedding_projections is not None:
            for elem in num_embedding_projections:
                head = nn.Linear(embedding_dim, 1)
                num_heads.append(head)

            self.num_heads = nn.ModuleList(num_heads)

    def forward(self, x, mask=None):
        cat_res = []
        for m in self.cat_heads:
            tmp = m(x)
            cat_res.append(tmp[:, :-1])

        num_res = []
        if self.num_heads is not None:
            for m in self.num_heads:
                tmp = m(x)
                num_res.append(tmp[:, :-1])

        return {'cat_features': cat_res, 'num_features': num_res} 