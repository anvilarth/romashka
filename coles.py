import os
import torch 
import numpy as np
import pickle
from torch.utils.data import IterableDataset, DataLoader
from models import TransactionsModel
from data_generators import batches_generator, cat_features_names, num_features_names, meta_features_names

import argparse
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
    
from embedding import EmbeddingLayer
from ptls.frames import PtlsDataModule
from ptls.frames.bert import MLMPretrainModule
from ptls.frames.coles import CoLESModule, ColesIterableDataset
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder
from ptls.nn.seq_encoder.containers import SeqEncoderContainer

from ptls.nn import TransformerEncoder
from functools import partial
from collections import namedtuple

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset, ColesIterableDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule
from ptls.nn import RnnSeqEncoder

from transformers import GPT2Config, GPT2Model

parser = argparse.ArgumentParser()
parser.add_argument('--group', type=str, default='coles')
parser.add_argument('--cnt_min', type=int, default=200)
parser.add_argument('--seq_len', type=int, default=100)
parser.add_argument('--num_splits', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--encoder', type=str, default='rnn')
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--checkpoint_dir', type=str, default='/home/jovyan/checkpoints')

args = parser.parse_args()


with open('./assets/num_embedding_projections.pkl', 'rb') as f:
    num_embedding_projections = pickle.load(f)
    
with open('./assets/cat_embedding_projections.pkl', 'rb') as f:
    cat_embedding_projections = pickle.load(f)

with open('./assets/meta_embedding_projections.pkl', 'rb') as f:
    meta_embedding_projections = pickle.load(f)
    
    
class MyPaddedBatch:
    def __init__(self, data):
        self.payload = data
        self.seq_lens = [data.shape[1]] * data.shape[0]
        
        
class IterDataset(IterableDataset):
    def __init__(self, dataset_train, batch_size=64, device='cuda', min_seq_len=200):
        self.data = dataset_train
        self.batch_size = batch_size
        self.device = device
        self.map = lambda: batches_generator(self.data, batch_size=self.batch_size, shuffle=True, device=self.device, is_train=True, output_format='torch', min_seq_len=min_seq_len)

    def __iter__(self):
        return self.map()
    
class MySampleUniform:
    """
    Sub samples with equal length = `seq_len`
    Start pos has fixed uniform distribution from sequence start to end with equal step
    |---------------------|       main sequence
    |------|              |        sub seq 1
    |    |------|         |        sub seq 2
    |         |------|    |        sub seq 3
    |              |------|        sub seq 4
    There is no random factor in this splitter, so sub sequences are the same every time
    Can be used during inference as test time augmentation
    """
    def __init__(self, split_count, seq_len, **_):
        self.split_count = split_count
        self.seq_len = seq_len

    def split(self, dates):
        date_len = dates.shape[0]
        date_range = np.arange(date_len)

        if date_len <= self.seq_len + self.split_count:
            return [date_range for _ in range(self.split_count)]

        start_pos = np.linspace(0, date_len - self.seq_len, self.split_count).round().astype(int)
        return [date_range[s:s + self.seq_len] for s in start_pos]
        
def coles_process(batch, splitter):
    res = {}
    
    local_date = batch['event_time']
    indexes = splitter.split(local_date)
    
    for k, v in batch.items():
        if type(v) == list and len(v) > 1:
            new_v = []
            for elem in v:
                tmp = []
                for ix in indexes:
                    tmp.append(elem[:, ix])
                new_v.append(torch.cat(tmp, dim=0))
        else:
            new_v = v 
        res[k] = new_v
    return res 

def coles_collate_fn(batch):
    batch = batch[0]
    len_batch = batch['num_features'][0].shape[0]
    batch = coles_process(batch, splitter)
    # print(batch)
    labels = torch.arange(len_batch).repeat(splitter.split_count)
    return batch, labels

class GPTEncoder(AbsSeqEncoder):
    def __init__(self, 
                 input_size=None,
                 hidden_size=None,
                 is_reduce_sequence=False,
                 type='gru'
                ):
    
        super().__init__(is_reduce_sequence=is_reduce_sequence)
        
        configuration = GPT2Config(vocab_size=1, n_positions=2000, 
                           n_embd=input_size, n_layer=2, 
                           n_head=1, resid_pdrop=0,
                           embd_pdrop=0, attn_pdrop=0)
        
        self.encoder = GPT2Model(configuration)
        self.hidden_size = input_size
        self.input_size = input_size
        
    def forward(self, x, h_0=None):
        """
        :param x:
        :param h_0: None or [1, B, H] float tensor
                    0.0 values in all components of hidden state of specific client means no-previous state and
                    use starter for this client
                    h_0 = None means no-previous state for all clients in batch
        :return:
        """
        shape = x.payload.size()
        out = self.encoder(inputs_embeds=x.payload).last_hidden_state[:, -1]
        
        return out

    @property
    def embedding_size(self):
        return self.hidden_size
    
class MySeqEncoder(SeqEncoderContainer):
    def __init__(self,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=False,
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=GPTEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )

    def forward(self, x, h_0=None):
        x = self.trx_encoder(x)
        x = self.seq_encoder(x, h_0)
        return x
    
class PtlsEmbeddingLayer(EmbeddingLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        return MyPaddedBatch(x)
    
    @property    
    def output_size(self):
        return super().get_embedding_size()

run_name = f'coles-{args.encoder}-splits={args.num_splits}-seqlen={args.seq_len}-bs={args.batch_size}'
    
wandb_logger = WandbLogger(name=run_name, project="romashka", entity="serofade", group=args.group)

        
path_to_dataset = '/home/jovyan/data/alfa/train_buckets'


dir_with_datasets = os.listdir(path_to_dataset)
dataset_train = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataloader = batches_generator(dataset_train, batch_size=args.batch_size, shuffle=True,
                                            device=device, is_train=True, output_format='torch', min_seq_len=args.cnt_min)

dataset = IterDataset(dataset_train, args.batch_size, device, args.cnt_min)
splitter=MySampleUniform(
        split_count=args.num_splits,
        seq_len=args.seq_len,
    )

dataloader = torch.utils.data.DataLoader(
    dataset,
    collate_fn=coles_collate_fn,
    num_workers=0,
    batch_size=1
)
    
ptls_emb_layer = PtlsEmbeddingLayer(cat_embedding_projections,
                                    cat_features_names,
                                    num_embedding_projections,
                                    num_features_names)

if args.encoder == 'rnn':
    seq_encoder = RnnSeqEncoder(
        input_size=ptls_emb_layer.get_embedding_size(),
        trx_encoder=ptls_emb_layer,
        hidden_size=args.hidden_size,
        type='gru',
    )
    
else:
    seq_encoder = MySeqEncoder(
        trx_encoder=ptls_emb_layer,
        hidden_size=args.hidden_size,
        type='gru',
    )

checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir, save_top_k=2, monitor='loss')

model = CoLESModule(
    seq_encoder=seq_encoder,
    optimizer_partial=partial(torch.optim.Adam, lr=0.001),
    lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=5, gamma=0.5),
)

trainer = pl.Trainer(
    max_epochs=args.num_epochs,
    gpus=1,
    callbacks=[checkpoint_callback],
    logger=wandb_logger,
)

trainer.fit(model, dataloader)
torch.save(model.state_dict(), args.checkpoint_dir + '/coles/' + run_name + '-final.ckpt') 