import os
import numpy as np
import torch
import torch.nn as nn
import wandb

import tqdm
import pickle
import pytorch_lightning as pl

from torch.utils.data import IterableDataset, DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models import TransactionsModel
from data_generators import batches_generator, cat_features_names, num_features_names, meta_features_names

from functools import partial
from collections import namedtuple
from tools import make_time_batch, calculate_embedding_size

from pl_models import TransactionQAModel
from pl_dataloader import TransactionQADataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

with open('./assets/num_embedding_projections.pkl', 'rb') as f:
    num_embedding_projections = pickle.load(f)
    
with open('./assets/cat_embedding_projections.pkl', 'rb') as f:
    cat_embedding_projections = pickle.load(f)

with open('./assets/meta_embedding_projections.pkl', 'rb') as f:
    meta_embedding_projections = pickle.load(f)
    
path_to_dataset = '/home/jovyan/data/alfa/train_buckets'
valpath_to_dataset = '/home/jovyan/data/alfa/val_buckets'

dir_with_datasets = os.listdir(path_to_dataset)
dataset_train = sorted([os.path.join(path_to_dataset, x) for x in dir_with_datasets])

valdir_with_datasets = os.listdir(valpath_to_dataset)
dataset_val = sorted([os.path.join(valpath_to_dataset, x) for x in valdir_with_datasets])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_days = torch.load('train_days.pt')
val_days = torch.load('val_days.pt')


ckpt = torch.load('/home/jovyan/romashka/wandb/run-20230222_133923-dhkmskss/files/checkpoints/final_model.ckpt')

logger = WandbLogger(
        project='romashka',
        entity='serofade',
        group='tqa'
    )

checkpoint_callback = ModelCheckpoint(
    # monitor='accuracy3',
    dirpath='/home/jovyan/romashka/checkpoints/',
    filename='tqa-{epoch:02d}-{accuracy3:.2f}',
    save_weights_only=True,
    every_n_epochs=1,
    save_last=True,
    mode='max',
)

model_transaction = TransactionsModel(cat_embedding_projections,
                          cat_features_names,
                          num_embedding_projections,
                          num_features_names,
                          meta_embedding_projections,
                          meta_features_names,
                          encoder_type='whisper/tiny',
                          head_type='next',
                          embedding_dropout=0.1
                         )
model_transaction.load_state_dict(ckpt)
model_transaction.to(device)

linear_mapping = nn.Linear(384, 512).to(device)

tok = AutoTokenizer.from_pretrained('google/flan-t5-small')
model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-small').to(device)

tqa = TransactionQAModel(model, model_transaction, linear_mapping, tok)

number_of_days = np.random.choice(train_days.numpy())

train_dataloader = TransactionQADataset(dataset_train, batch_size=16)
val_dataloader = TransactionQADataset(dataset_val, batch_size=16, shuffle=False)

trainer = pl.Trainer(limit_train_batches=10000, max_epochs=20, gpus=1, logger=logger, callbacks=[checkpoint_callback])
trainer.fit(model=tqa, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


