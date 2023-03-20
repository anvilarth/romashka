import os
import re
import sys
import tqdm
import pickle
import argparse
import numpy as np
from typing import Dict, Optional

import wandb
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import transformers
from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          HfArgumentParser)

sys.path.insert(1, '/Users/abdullaeva/Documents/Projects/TransactionsQA')
# for MlSpace: /home/jovyan/transactionsQA/romashka
print(sys.path)

from romashka.logging_handler import get_logger
from romashka.data_generators import (batches_generator,
                                      cat_features_names,
                                      num_features_names,
                                      meta_features_names)
from romashka.transactions_qa.train_args import (ModelArguments,
                                                 DataTrainingArguments,
                                                 TrainingArguments)
from romashka.pl_dataloader import TransactionQADataset
from romashka.models import TransactionsModel
from romashka.transactions_qa.tasks import MostFrequentMCCCodeTask

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))

def get_projections_maps(num_embedding_projections_fn: str = './assets/num_embedding_projections.pkl',
                         cat_embedding_projections_fn: str = './assets/cat_embedding_projections.pkl',
                         meta_embedding_projections_fn: str = './assets/meta_embedding_projections.pkl',
                         relative_folder: Optional[str] = None) -> Dict[str, dict]:
    """
    Loading projections mappings.
    Args:
        relative_folder: a relative path for all mappings;
        num_embedding_projections_fn: a filename for mapping loading;
        cat_embedding_projections_fn:  a filename for mapping loading;
        meta_embedding_projections_fn: a filename for mapping loading;

    Returns: a Dict[str, Mapping],
        where key - is a mapping name, value - a mapping itself.

    """
    if relative_folder is not None:
        num_embedding_projections_fn = os.path.join(relative_folder, num_embedding_projections_fn)
        cat_embedding_projections_fn = os.path.join(relative_folder, cat_embedding_projections_fn)
        meta_embedding_projections_fn = os.path.join(relative_folder, meta_embedding_projections_fn)

    with open(num_embedding_projections_fn, 'rb') as f:
        num_embedding_projections = pickle.load(f)

    with open(cat_embedding_projections_fn, 'rb') as f:
        cat_embedding_projections = pickle.load(f)

    with open(meta_embedding_projections_fn, 'rb') as f:
        meta_embedding_projections = pickle.load(f)

    return {
        "num_embedding_projections": num_embedding_projections,
        "cat_embedding_projections": cat_embedding_projections,
        "meta_embedding_projections": meta_embedding_projections
    }

def main():
    pl.seed_everything(11)

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set up logging
    logger = get_logger(
        name=__file__.__name__,
        logging_level=training_args.log_level
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (os.path.isdir(training_args.save_checkpoints_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir):
        last_checkpoint = get_last_checkpoint(training_args.save_checkpoints_dir)
        if last_checkpoint is None and len(os.listdir(training_args.save_checkpoints_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.save_checkpoints_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--save_checkpoints_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    elif not os.path.exists(training_args.save_checkpoints_dir):
        os.makedirs(training_args.save_checkpoints_dir)
    else:
        logger.error(f"Output directory argument: ({training_args.save_checkpoints_dir}) is not a directory!")
        raise AttributeError(f"Output directory argument: ({training_args.save_checkpoints_dir}) is not a directory!")

    # Get the datasets
    data_files = {}
    if data_args.train_folder is not None:
        dir_with_datasets = os.listdir(os.path.join(data_args.data_path, data_args.train_folder))
        dataset_files = sorted([os.path.join(data_args.data_path, data_args.train_folder, x)
                                for x in dir_with_datasets])
        data_files["train"] = dataset_files
    if data_args.validation_file is not None:
        dir_with_datasets = os.listdir(os.path.join(data_args.data_path, data_args.validation_folder))
        dataset_files = sorted([os.path.join(data_args.data_path, data_args.validation_folder, x)
                                for x in dir_with_datasets])
        data_files["validation"] = dataset_files

    # Check weights existence by paths from args
    if (model_args.transactions_model_name_or_path is None) \
            or not os.path.exists(model_args.transactions_model_name_or_path):
        logger.error(f"Transactions model weights path do not exists: {model_args.transactions_model_name_or_path}")
        raise FileExistsError(
            f"Transactions model weights path do not exists: {model_args.transactions_model_name_or_path}"
        )

    # Configure models
    if training_args.no_cuda:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f'Using device: {device}')

    projections_maps = get_projections_maps(relative_folder=data_args.projections_mappings_path)
    transactions_model_config = {
        "cat_features_names": cat_features_names,
        "cat_embedding_projections": projections_maps.get('cat_embedding_projections'),
        "num_features_names": num_features_names,
        "num_embedding_projections": projections_maps.get('num_embedding_projections'),
        "meta_features_names": meta_features_names,
        "meta_embedding_projections": projections_maps.get('meta_embedding_projections'),
        "encoder_type": model_args.transactions_model_encoder_type,
        "head_type": model_args.transactions_model_head_type,
        "embedding_dropout": 0.1
    }
    model_transaction = TransactionsModel(**transactions_model_config)

    # Load weights
    ckpt = torch.load(model_args.transactions_model_name_or_path, map_location='cpu')
    model_transaction.load_state_dict(ckpt)
    model_transaction.to(device)


    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None
    }

    # Load pretrained model and tokenizer
    if model_args.use_fast_tokenizer:
        logger.warning(f'-- Using fast Tokenizer --')

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "do_lowercase": False
    }


if __name__ == '__main__':






