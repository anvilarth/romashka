import pickle
import torch
import pytorch_lightning as pl
import argparse

from torch.utils.data import ConcatDataset
from pytorch_lightning.loggers import WandbLogger
from ptls.data_load.datasets import MemoryMapDataset
from ptls.data_load.iterable_processing import SeqLenFilter
from ptls.frames.coles import ColesDataset
from ptls.frames.coles.split_strategy import SampleSlices
from ptls.frames import PtlsDataModule

from functools import partial
from ptls.nn import TrxEncoder, RnnSeqEncoder
from ptls.frames.coles import CoLESModule


parser = argparse.ArgumentParser()
parser.add_argument('--group', type=str, default='coles')
parser.add_argument('--cnt_min', type=int, default=25)
parser.add_argument('--cnt_max', type=int, default=200)

args = parser.parse_args()


print("LOADING DATASET")
with open(f'/home/jovyan/data/alfa/train_buckets_new/chunk_small', 'rb') as f:
    data = pickle.load(f)
    
print("LOADED DATASET")

    
columns = list(data[0].keys())[2:]
em_process = {elem:{'in': 200, 'out': 12} for elem in columns}


trx_encoder_params = dict(
    embeddings_noise=0.003,
    numeric_values={'amnt': 'identity'},
    embeddings=em_process,
)

seq_encoder = RnnSeqEncoder(
    trx_encoder=TrxEncoder(**trx_encoder_params),
    hidden_size=256,
    type='gru',
)

model = CoLESModule(
    seq_encoder=seq_encoder,
    optimizer_partial=partial(torch.optim.Adam, lr=0.001),
    lr_scheduler_partial=partial(torch.optim.lr_scheduler.StepLR, step_size=30, gamma=0.9),
)

dataset = MemoryMapDataset(
            data=data,
            i_filters=[
                SeqLenFilter(min_seq_len=args.cnt_min),
            ],
        )

concat_dataset = ConcatDataset([dataset])

wandb_logger = WandbLogger(project="romashka", entity="serofade", group=args.group)


train_dl = PtlsDataModule(
    train_data=ColesDataset(
        concat_dataset,
        splitter=SampleSlices(
            split_count=5,
            cnt_min=args.cnt_min,
            cnt_max=args.cnt_max,
        ),
    ),
    train_num_workers=32,
    train_batch_size=128,
)

trainer = pl.Trainer(
    max_epochs=15,
    gpus=1 if torch.cuda.is_available() else 0,
    enable_progress_bar=True,
    logger=wandb_logger,
)

trainer.fit(model, train_dl)