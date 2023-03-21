import os
import re
import sys
import shutil
import tqdm
import pickle
import argparse
import numpy as np
from typing import Dict, Optional

import wandb

os.environ["WANDB_API_KEY"] = "de71b243e187c02735ee3d741c05d2d906905d2b"
os.environ["WANDB_MODE"] = "online"

import torch
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import transformers
from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          AutoConfig,
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
                                                 TrainingArguments,
                                                 TasksArguments)
from romashka.transactions_qa.tasks import AutoTask, AUTO_TASKS
from romashka.pl_dataloader import TransactionQADataset
from romashka.models import TransactionsModel
from romashka.transactions_qa.tqa_model import TransactionQAModel
from romashka.transactions_qa.utils import (get_last_checkpoint, get_projections_maps)


def main():
    pl.seed_everything(11)

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, TasksArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, tasks_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, tasks_args = parser.parse_args_into_dataclasses()

    # Set up logging
    logger = get_logger(
        name="train",
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
    elif os.path.exists(training_args.save_checkpoints_dir) and training_args.overwrite_output_dir:
        shutil.rmtree(training_args.save_checkpoints_dir)
        os.makedirs(training_args.save_checkpoints_dir)
    else:
        logger.error(f"Output directory argument: ({training_args.save_checkpoints_dir}) is not a directory!")
        raise AttributeError(f"Output directory argument: ({training_args.save_checkpoints_dir}) is not a directory!")

    # Get the datasets
    data_files = {}
    if data_args.train_folder is not None and training_args.do_train:
        dir_with_datasets = os.listdir(os.path.join(data_args.data_path, data_args.train_folder))
        dataset_files = sorted([os.path.join(data_args.data_path, data_args.train_folder, x)
                                for x in dir_with_datasets])
        logger.info(f"Detected {len(dataset_files)} files for training.")
        data_files["train"] = dataset_files
    if data_args.validation_folder is not None and training_args.do_eval:
        dir_with_datasets = os.listdir(os.path.join(data_args.data_path, data_args.validation_folder))
        dataset_files = sorted([os.path.join(data_args.data_path, data_args.validation_folder, x)
                                for x in dir_with_datasets])
        logger.info(f"Detected {len(dataset_files)} files for validation.")
        data_files["validation"] = dataset_files

    # Check weights existence by paths from args
    if (model_args.transactions_model_name_or_path is None) \
            or not os.path.exists(model_args.transactions_model_name_or_path):
        logger.error(f"Transactions model weights path do not exists: {model_args.transactions_model_name_or_path}")
        raise FileExistsError(
            f"Transactions model weights path do not exists: {model_args.transactions_model_name_or_path}"
        )

    # Configure device
    available_gpus = []
    if training_args.no_cuda:
        device = torch.device('cpu')
    else:
        available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
        if len(available_gpus) > 1:
            logger.info(f"Detected multiple available GPU devices: {len(available_gpus)}. "
                        f"In this case by default select `cpu` for data and model loading, "
                        f"but pass full list of available devices to Trainer..")
            device = torch.device('cpu')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
            available_gpus = [0]
        else:
            device = torch.device('cpu')

    logger.info(f'Using device: {device}, with number of available GPUs: {len(available_gpus)}')

    # Loading Transactions model & weights
    logger.info(f"Loading Transactions model...")
    projections_maps = get_projections_maps(relative_folder=data_args.projections_mappings_path)
    transactions_model_config = {
        "cat_features": cat_features_names,
        "cat_embedding_projections": projections_maps.get('cat_embedding_projections'),
        "num_features": num_features_names,
        "num_embedding_projections": projections_maps.get('num_embedding_projections'),
        "meta_features": meta_features_names,
        "meta_embedding_projections": projections_maps.get('meta_embedding_projections'),
        "encoder_type": model_args.transactions_model_encoder_type,
        "head_type": model_args.transactions_model_head_type,
        "embedding_dropout": 0.1
    }
    transactions_model = TransactionsModel(**transactions_model_config)

    # Load weights
    ckpt = torch.load(model_args.transactions_model_name_or_path, map_location='cpu')
    transactions_model.load_state_dict(ckpt)
    transactions_model.to(device)

    # Configure and load from HF hub LM model
    logger.info(f"Loading Language model: `{model_args.language_model_name_or_path}`...")
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "return_unused_kwargs": True
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

    config, unused_kwargs = AutoConfig.from_pretrained(
        model_args.language_model_name_or_path, **config_kwargs
    )
    # Download vocabulary from huggingface.co and define model-specific arguments
    tokenizer = AutoTokenizer.from_pretrained(model_args.language_model_name_or_path, **tokenizer_kwargs)

    # Download model from huggingface.co and cache.
    lm_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.language_model_name_or_path,
        config=config
    )

    # Create tasks
    tasks = []
    task_names = tasks_args.task_names
    if isinstance(task_names, str):
        task_names = eval(task_names)
    task_kwargs = tasks_args.task_kwargs
    if isinstance(task_kwargs, str):
        task_kwargs = eval(task_kwargs)
    print(f"Got task_names: {task_names} with task_kwargs: {task_kwargs}")

    for task_i, task_name in enumerate(task_names):
        task_kwargs = task_kwargs[task_i] if task_i < len(task_kwargs) else {}
        if "tokenizer" not in task_kwargs:
            task_kwargs['tokenizer'] = tokenizer
        task = AutoTask.get(task_name=task_name, **task_kwargs)
        tasks.append(task)
    logger.info(f"Created {len(tasks)} tasks.")

    # Create general Tranactions QA model
    transactionsQA_model_config = {
        "warmup_steps": training_args.warmup_steps,
        "training_steps": training_args.max_steps,
        "do_freeze_tm": training_args.do_freeze_transactions_model,
        "do_freeze_lm": training_args.do_freeze_language_model,
        "do_freeze_connector": training_args.do_freeze_connector,
        "connector_input_size": 384,
    }
    model = TransactionQAModel(
        language_model=lm_model,
        transaction_model=transactions_model,
        tokenizer=tokenizer,
        tasks=tasks,
        **transactionsQA_model_config
    )

    # Datasets & Dataloader & Other utils
    if training_args.do_train and "train" in data_files:
        train_dataloader = TransactionQADataset(data_files.get('train'),
                                                device=device,
                                                batch_size=training_args.per_device_train_batch_size)
        logger.info(f"Created train dataloader.")
    if training_args.do_eval and "validation" in data_files:
        val_dataloader = TransactionQADataset(data_files.get('validation'),
                                              device=device,
                                              batch_size=training_args.per_device_eval_batch_size,
                                              shuffle=False)
        logger.info(f"Created validation dataloader.")
    if (not training_args.do_train) and (not training_args.do_eval) :
        logger.error("There is nothing to do. Please pass `do_train` and/or `do_eval`.")
        return 1

    # Training & Callbacks
    wb_logger = WandbLogger(
        project=training_args.project_name,
        group=training_args.group_name
    )
    tb_logger = TensorBoardLogger(name=training_args.run_name,
                                  save_dir="./tb_logs",
                                  default_hp_metric=False)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(
        monitor=training_args.save_strategy_metric,  # default: 'val_loss'
        dirpath=training_args.save_checkpoints_dir,  # default: './checkpoints/'
        filename=training_args.save_filename_format,  # default: 'checkpoint-{epoch:02d}-{loss3:.2f}'
        save_weights_only=training_args.save_only_weights,  # default: 'True'
        every_n_epochs=training_args.save_epochs,  # default: '1'
        save_last=training_args.save_last_checkpoint,  # default: 'True'
        mode=training_args.save_strategy_mode,  # default: 'max'
    )

    trainer = pl.Trainer(
        fast_dev_run=training_args.fast_dev_run,
        max_steps=training_args.max_steps,
        max_epochs=training_args.max_epochs,
        gpus=len(available_gpus),
        auto_select_gpus=True,
        log_every_n_steps=1,
        # log_gpu_memory
        # track_grad_norm
        val_check_interval=2,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        logger=wb_logger,  #[tb_logger, wb_logger],
        callbacks=[checkpoint_callback, lr_monitor_callback])
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    import os
    # os.environ['HF_DATASETS_OFFLINE'] = '1'  # offline mode for HF datasets
    # os.environ['TRANSFORMERS_OFFLINE'] = '1'  # offline mode for HF Transformers
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # disable DataParallel for test

    # Pretrained models are downloaded and locally cached at: ~/.cache/huggingface/transformers/.
    # This is the default directory given by the shell environment variable TRANSFORMERS_CACHE.
    os.environ['TRANSFORMERS_CACHE'] = "/Users/abdullaeva/Documents/Projects/TransactionsQA/checkpoints/cache"
    # or "/home/jovyan/.cache/huggingface/hub"
    os.environ['HF_DATASETS_CACHE'] = "/Users/abdullaeva/Documents/Projects/TransactionsQA/checkpoints/cache"
    # or "/home/jovyan/.cache/huggingface/datasets"
    main()