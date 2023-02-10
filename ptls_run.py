import os
import torch
import wandb
import pickle
import argparse
import torchmetrics

from functools import partial
from torch.utils.data import DataLoader

from data_generators import cat_features_names, num_features_names, meta_features_names

from ptls_my import my_collate_fn
from ptls_my import GPTEncoder, MySeqEncoder
from ptls_my import MyPaddedBatch, IterDataset, PtlsEmbeddingLayer, MySampleUniform

from ptls.nn import RnnEncoder, RnnSeqEncoder, TransformerEncoder
from ptls.nn.seq_encoder.utils import AllStepsHead, FlattenHead
from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder

from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.utils import collate_feature_dict
from ptls.data_load.iterable_processing import SeqLenFilter

from ptls.frames import PtlsDataModule

from ptls.frames.cpc import CpcModule
from ptls.frames.coles import CoLESModule
from ptls.frames.bert import RtdModule, MLMPretrainModule

from ptls.frames.coles.split_strategy import SampleSlices, SampleUniformBySplitCount

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

pl.seed_everything(12, workers=True)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('True or False was expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--group', type=str, default='coles', choices=['coles', 'cpc', 'mlm', 'rtd'])
parser.add_argument('--entity', type=str, default='vasilev-va') # serofade

parser.add_argument('--splitter', type=str, default=None, choices=['uniform', 'slices', 'uniform_by_split_count', None])
parser.add_argument('--is_sorted', type=str2bool, default=False)
parser.add_argument('--num_splits', type=int, default=5)
parser.add_argument('--seq_len', type=int, default=200)
parser.add_argument('--cnt_min', type=int, default=100)
parser.add_argument('--cnt_max', type=int, default=200)

parser.add_argument('--encoder', type=str, default='rnn', choices=['rnn', 'bert', 'gpt', 't5'])
parser.add_argument('--hidden_size', type=int, default=256)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--step_size', type=int, default=5)
parser.add_argument('--gamma', type=float, default=0.5)
parser.add_argument('--num_epochs', type=int, default=30)
parser.add_argument('--total_steps', type=int, default=int(1e6))
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--path_to_dataset', type=str, default='/home/jovyan/afilatov/data/alfa/')
parser.add_argument('--checkpoint_dir', type=str, default='/home/jovyan/v_vasilev/checkpoints/ptls_pretrain/')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('./assets/num_embedding_projections.pkl', 'rb') as f:
    num_embedding_projections = pickle.load(f)
    
with open('./assets/cat_embedding_projections.pkl', 'rb') as f:
    cat_embedding_projections = pickle.load(f)

with open('./assets/meta_embedding_projections.pkl', 'rb') as f:
    meta_embedding_projections = pickle.load(f)


run_name = f'ptls-{args.group}-{args.encoder}-splitter={str(args.splitter)}-splits={args.num_splits}-seqlen={args.seq_len}-bs={args.batch_size}-lr={args.lr}'
wandb_logger = WandbLogger(name=run_name, project="romashka", entity=args.entity, group=args.group)


path_to_train_dataset = args.path_to_dataset + 'train_buckets'
path_to_val_dataset = args.path_to_dataset + 'val_buckets'

dir_with_train_datasets = os.listdir(path_to_train_dataset)
dir_with_val_datasets = os.listdir(path_to_val_dataset)

dataset_train = sorted([os.path.join(path_to_train_dataset, x) for x in dir_with_train_datasets])[0:1]
dataset_val = sorted([os.path.join(path_to_val_dataset, x) for x in dir_with_val_datasets])[0:1]

train_dataset = IterDataset(dataset_train, args.batch_size, device, args.seq_len)
val_dataset = IterDataset(dataset_val, args.batch_size, device, args.seq_len)


if args.splitter == 'uniform':
    splitter=MySampleUniform(
        split_count=args.num_splits,
        seq_len=args.seq_len,
    )
elif args.splitter == 'slices':
    is_sorted = args.is_sorted
    if args.group == 'cpc':
        print('The mode is CPC - splitter should preserve order in samples. Param is_sorted was assigned to True.')
        is_sorted = True
    splitter = SampleSlices(split_count=args.num_splits, cnt_min=args.cnt_min, cnt_max=args.cnt_max, is_sorted=is_sorted)
elif args.splitter == 'uniform_by_split_count':
    splitter = SampleUniformBySplitCount(split_count=args.num_splits)
else:
    print('Splitter is None.')
    splitter = None
    

train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    collate_fn=partial(my_collate_fn, splitter=splitter,
                       rep=args.num_splits if splitter is not None else 1,
                       mode=args.group),
    num_workers=0,
    batch_size=1
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset,
    collate_fn=partial(my_collate_fn, splitter=splitter,
                       rep=args.num_splits if splitter is not None else 1,
                       mode=args.group),
    num_workers=0,
    batch_size=1
)

trx_encoder = PtlsEmbeddingLayer(splitter,
                                 cat_embedding_projections,
                                 cat_features_names,
                                 num_embedding_projections,
                                 num_features_names)
    

if args.encoder == 'rnn':
    if args.group == 'mlm':
        print(f'The mode is MLM. Param hidden_size was assigned to output_size of trx_encoder: {trx_encoder.output_size}')
        seq_encoder = RnnEncoder(
            input_size=trx_encoder.get_embedding_size(),
            is_reduce_sequence=False,
            hidden_size=trx_encoder.output_size,
            type='gru',
        )
    else:
        seq_encoder = RnnSeqEncoder(
            input_size=trx_encoder.get_embedding_size(),
            trx_encoder=trx_encoder,
            hidden_size=args.hidden_size,
            type='gru',
        )
else:
    seq_encoder = MySeqEncoder(
        trx_encoder=trx_encoder,
        hidden_size=args.hidden_size,
        type='gru',
    )

    
if args.group == 'coles':
    model = CoLESModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=args.lr),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=args.step_size, gamma=args.gamma),
    ).to(device)
elif args.group == 'cpc':
    model = CpcModule(
        seq_encoder=seq_encoder,
        optimizer_partial=partial(torch.optim.Adam, lr=args.lr),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=args.step_size, gamma=args.gamma)
    ).to(device)
elif args.group == 'rtd':
    model = RtdModule(
        seq_encoder=seq_encoder,
        validation_metric=torchmetrics.AUROC(task='binary'),
        optimizer_partial=partial(torch.optim.Adam, lr=args.lr),
        lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=args.step_size, gamma=args.gamma),
        head = torch.nn.Sequential(
            AllStepsHead(
                torch.nn.Sequential(
                    torch.nn.Linear(args.hidden_size, 1),
                    torch.nn.Sigmoid(),
                    torch.nn.Flatten(),
                )
            ),
            FlattenHead(),
        )
    ).to(device)
elif args.group == 'mlm':
    model = MLMPretrainModule(
        trx_encoder=trx_encoder, 
        seq_encoder=seq_encoder,
        total_steps=args.total_steps
    ).to(device)


if args.group == 'coles':
    val_monitor = 'recall_top_k'
elif args.group == 'cpc':
    val_monitor = 'cpc_accuracy'
elif args.group == 'rtd':
    val_monitor = 'valid_auroc'
elif args.group == 'mlm':
    val_monitor = 'mlm/valid_mlm_loss' 

lr_monitor = LearningRateMonitor(logging_interval='step')
checkpoint_callback = ModelCheckpoint(dirpath=args.checkpoint_dir + args.group + '/',
                                      filename=run_name+'{epoch}-{step}-{'+val_monitor+':.5f}',
                                      save_top_k=2, monitor=val_monitor)

trainer = pl.Trainer(
    max_epochs=args.num_epochs,
    gpus=args.num_gpus,
    callbacks=[checkpoint_callback, lr_monitor],
    logger=wandb_logger,
)

trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
torch.save(model.state_dict(), args.checkpoint_dir + args.group + '/' + run_name + '-final.ckpt')
