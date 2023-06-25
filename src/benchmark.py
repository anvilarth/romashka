import os
import sys 
import tqdm
import numpy as np
import wandb
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F

from skorch import NeuralNetRegressor, NeuralNetClassifier

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from functools import partial
from collections import namedtuple
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig

import argparse
import torch

from src.transactions_qa.tqa_model import TransactionQAModel
from src.models.components.models import TransactionsModel
from src.utils.tools import (make_time_batch, 
                   calculate_embedding_size)

from src.data.alfa.components import ( 
                             cat_features_names, 
                             num_features_names, 
                             meta_features_names)

from src.data import AlfaDataModule 
from src.transactions_qa.tqa_model import TransactionQAModel
from src.transactions_qa.utils import get_projections_maps, get_exponent_number, get_mantissa_number
from src.tasks import AbstractTask, AutoTask
from src.transactions_qa.utils import get_split_indices,  prepare_splitted_batch, collate_batch_dict
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.base import BaseEstimator


from catboost import CatBoostClassifier, CatBoostRegressor, metrics, Pool, cv

class MyCatBoostRegressor(CatBoostRegressor, BaseEstimator):
    def __init__(self, random_state=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = random_state

class MyCatBoostClassifier(CatBoostClassifier, BaseEstimator):
    def __init__(self, random_state=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = random_state

class SimpleMLP(nn.Module):
    def __init__(
            self,
            inp_size,
            hidden_size,
            nonlin=F.relu,
    ):
        super().__init__()

        self.inp_size = inp_size
        self.nonlin = nonlin
        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(inp_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, X, **kwargs):
        X = F.relu(self.linear1(X))
        X = self.linear2(X)
        return X


def load_transaction_model(encoder_type='whisper/tiny', head_type='next'):
    projections_maps = get_projections_maps(relative_folder="..")
    # Loading Transactions model & weights
    print(f"Loading Transactions model...")

    transactions_model_encoder_type = encoder_type
    transactions_model_head_type = head_type


    transactions_model_config = {
        "cat_features": cat_features_names,
        "cat_embedding_projections": projections_maps.get('cat_embedding_projections'),
        "num_features": num_features_names,
        "num_embedding_projections": projections_maps.get('num_embedding_projections'),
        "meta_features": meta_features_names,
        "meta_embedding_projections": projections_maps.get('meta_embedding_projections'),
        "encoder_type": transactions_model_encoder_type,
        "head_type": transactions_model_head_type,
        "embedding_dropout": 0.1
    }
    transactions_model = TransactionsModel(**transactions_model_config)

    return transactions_model, projections_maps

def load_datamodule():
    DATA_PATH = '/home/jovyan/romashka/data' 
    dataset_config = {
                'data_dir': DATA_PATH,
                'batch_size': 32,
                'min_seq_len': 0,
                'max_seq_len': 250,
                'shuffle': False,
                'num_workers': 5,
                'pin_memory': True,
                'seed': 42
    }    

    dm = AlfaDataModule(**dataset_config)
    return dm


def load_tasks(task_names, tokenizer):
    # Create tasks
    tasks = []
    tasks_kwargs = [{"num_options": 6, "floating_threshold": False, 'answer2text': True, 'use_numerical_output': False}, 
    {"num_options": 6, "floating_threshold": False, 'use_numerical_output': False}] # ground truth + 5 additional options
    if isinstance(task_names, str):
        task_names = eval(task_names)
    if isinstance(tasks_kwargs, str):
        tasks_kwargs = eval(tasks_kwargs)
    print(f"Got task_names: {task_names} with task_kwargs: {tasks_kwargs}")

    for task_i, task_name in enumerate(task_names):
        task_kwargs = tasks_kwargs[task_i] if task_i < len(tasks_kwargs) else {}
        if "tokenizer" not in task_kwargs:
            task_kwargs['tokenizer'] = tokenizer
        task = AutoTask.get(task_name=task_name, **task_kwargs)
        tasks.append(task)
    print(f"Created {len(tasks)} tasks.")
    return tasks


def load_language_model(language_model_name_or_path="google/flan-t5-small"):
    use_fast_tokenizer = True

    print(f"Loading Language model: `{language_model_name_or_path}`...")
    config_kwargs = {
        "use_auth_token": None,
        "return_unused_kwargs": True
    }

    tokenizer_kwargs = {
        "use_fast": use_fast_tokenizer,
        "use_auth_token": None,
        "do_lowercase": False
    }

    config, unused_kwargs = AutoConfig.from_pretrained(
        language_model_name_or_path, **config_kwargs
    )
    # Download vocabulary from huggingface.co and define model-specific arguments
    tokenizer = AutoTokenizer.from_pretrained(language_model_name_or_path, **tokenizer_kwargs)

    # Download model from huggingface.co and cache.
    lm_model = AutoModelForSeq2SeqLM.from_pretrained(
        language_model_name_or_path,
        config=config
    )
    return lm_model, tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='catboost')
parser.add_argument('--task_name', type=str, default='next_amnt_open_ended')
parser.add_argument('--task_type', type=str, default='regression')
parser.add_argument('--cross_validation', action='store_true', default=False)
parser.add_argument('--embeddings_path', type=str, default='/home/jovyan/romashka/assets/boosting_embeds/')

args = parser.parse_args()

wandb.init(project="benchmark", name=f'{args.method}-{args.task_type}-{args.task_name}')

device = 'cuda:0'
task_names = [args.task_name]
LM_NAME = 'google/flan-t5-small'

transactions_model, projections_maps = load_transaction_model()
dm = load_datamodule()

lm_model, tokenizer = load_language_model(language_model_name_or_path=LM_NAME)

ckpt = torch.load("/home/jovyan/checkpoints/transactions_model/final_model_v2.ckpt", map_location='cpu')
transactions_model.load_state_dict(ckpt, strict=False)
transactions_model = transactions_model.to(device)


tasks = load_tasks(task_names, tokenizer)

train_data = []
train_labels = []

val_data = []
val_labels = []

with torch.no_grad():
    for batch in tqdm.tqdm(dm.train_dataloader()):
        batch_size = batch['mask'].shape[0]
        new_batch = tasks[0].prepare_task_batch(batch)
        train_labels.append(new_batch['label'])

        if args.embeddings_path == '':
            for key in batch:
                batch[key] = batch[key].to(device)

            transactions_model.eval()
            embs, mask = transactions_model.get_embs(new_batch)
            trx_index = mask.sum(1) - 1
            train_data.append(embs[torch.arange(batch_size, device=device), trx_index])

    
    for batch in tqdm.tqdm(dm.val_dataloader()):
        batch_size = batch['mask'].shape[0]
        new_batch = tasks[0].prepare_task_batch(batch)
        val_labels.append(new_batch['label'])

        if args.embeddings_path == '': 
            for key in batch:
                batch[key] = batch[key].to(device)
            
            transactions_model.eval()
            embs, mask = transactions_model.get_embs(new_batch)
            trx_index = mask.sum(1) - 1
            val_data.append(embs[torch.arange(batch_size, device=device), trx_index])

if args.embeddings_path == '':
    train_embeds = torch.vstack(train_data).cpu().numpy()
    val_embeds = torch.vstack(val_data).cpu().numpy()

else:
    train_embeds = np.load(args.embeddings_path + 'train_embeds.npy')
    val_embeds = np.load(args.embeddings_path + 'val_embeds.npy')

train_labels = torch.cat(train_labels).cpu().numpy()
val_labels = torch.cat(val_labels).cpu().numpy()

if args.method == 'catboost':
    if args.task_type == 'regression': 
        model = MyCatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
        )
    elif args.task_type == 'classification':
        model = MyCatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
        )
    else:
        raise NotImplementedError

    model_params = {
        "learning_rate": optuna.distributions.LogUniformDistribution(1e-3, 0.1),
        "depth": optuna.distributions.IntUniformDistribution(1, 10),
        "subsample": optuna.distributions.UniformDistribution(0.05, 1.0),
        "min_data_in_leaf": optuna.distributions.IntUniformDistribution(1, 100)
    }

elif args.method == 'nn':
    if args.task_type == 'classification':
        pass
    elif args.task_type == 'regression':
        net = NeuralNetRegressor( 
            SimpleMLP,
            max_epochs=20,
            module__hidden_size=10,
            module__inp_size=384,
            lr=0.01,
            device='cuda',
            # uncomment this to train with CUDA
        )
    else:
        raise NotImplementedError

    model_params = {'module__hidden_size': optuna.distributions.IntUniformDistribution(10, 150)}


elif args.method == 'linear':
    if args.task_type == 'regression':
        model = LinearRegression(n_jobs=-1)
    elif args.task_type == 'classification':
        model = LogisticRegression(n_jobs=-1)
    else:
        raise NotImplementedError
    
    model_params = {}

elif args.method == 'mlp':
    if args.task_type == 'regression':
        model = MLPRegressor(hidden_layer_sizes=(transaction_model.output_size, 
                                                transaction_model.output_size,
                                                transaction_model.output_size))
    elif args.task_type == 'classification':
        model = MLPClassifier(hidden_layer_sizes=(transaction_model.output_size, 
                                                transaction_model.output_size,
                                                transaction_model.output_size))
    else:
        raise NotImplementedError
    
elif args.method == 'random_forest':
    if args.task_type == 'regression':
        model = RandomForestRegressor(n_jobs=-1, verbose=1)
    elif args.task_type == 'classification':
        model = RandomForestClassifier(n_jobs=-1)
    else:
        raise NotImplementedError

    model_params = {
        'n_estimators': optuna.distributions.IntUniformDistribution(50, 1000),
        'max_depth':  optuna.distributions.IntUniformDistribution(4, 50),
        'min_samples_split': optuna.distributions.IntUniformDistribution(1, 150),
        'min_samples_leaf':  optuna.distributions.IntUniformDistribution(1, 60),
    }
else: 
    raise NotImplementedError

if args.cross_validation:
        optuna_search = optuna.integration.OptunaSearchCV(model, model_params, cv=5, n_jobs=-1, n_trials=10, verbose=1)
        optuna_search.fit(train_embeds[:100], train_labels[:100])
        pred_labels = optuna_search.predict(val_embeds)

else:
    model.fit(train_embeds, train_labels)
    pred_labels = model.predict(val_embeds)

if args.task_type == 'regression':
    wandb.log({'val_l1_loss': mean_absolute_error(pred_labels, val_labels)})
