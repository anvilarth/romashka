import os
import sys
import shutil
import yaml
from pathlib import Path

import wandb
os.environ["WANDB_MODE"] = "online"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import transformers
from transformers import (AutoModelForSeq2SeqLM,
                          AutoTokenizer,
                          AutoConfig,
                          HfArgumentParser)

print(sys.path)

from src.utils.logging_handler import get_logger
from src.data.alfa.components import (cat_features_names,
                                      num_features_names,
                                      meta_features_names)
from src.transactions_qa.train_args import (ModelArguments,
                                                 DataTrainingArguments,
                                                 TrainingArguments,
                                                 TasksArguments)

from src.tasks import AutoTask
from src.data import AlfaDataModule
from src.models.components.models import TransactionsModel
from src.transactions_qa.tqa_model import TransactionQAModel
from src.transactions_qa.layers.connector import (make_linear_connector,
                                                       make_recurrent_connector)
from src.transactions_qa.utils import (get_last_checkpoint, get_projections_maps)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, TasksArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, tasks_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, tasks_args = parser.parse_args_into_dataclasses()

    pl.seed_everything(training_args.seed)
    # Set up logging
    logger = get_logger(
        name="train",
        logging_level=training_args.log_level
    )

    with open(data_args.local_config, 'r') as f:
        logger.info("READING LOCAL CONFIG")
        cfg = yaml.safe_load(f)

    os.environ["WANDB_API_KEY"] = cfg['wandb_key']
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Detecting last checkpoint.
    last_checkpoint = None
    if (os.path.isdir(training_args.save_checkpoints_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir):

        if training_args.resume_from_checkpoint is not None:
            last_checkpoint = get_last_checkpoint(training_args.save_checkpoints_dir)
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
    transactions_model.load_state_dict(ckpt, strict=False)

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
    tasks_kwargs = tasks_args.tasks_kwargs
    if isinstance(tasks_kwargs, str):
        tasks_kwargs = eval(tasks_kwargs)
    print(f"Got task_names: {task_names} with task_kwargs: {tasks_kwargs}")

    for task_i, task_name in enumerate(task_names):
        task_kwargs = tasks_kwargs[task_i] if task_i < len(tasks_kwargs) else {}
        if "tokenizer" not in task_kwargs:
            task_kwargs['tokenizer'] = tokenizer
        task_kwargs['use_numerical'] = model_args.use_numerical
        task = AutoTask.get(task_name=task_name, **task_kwargs)
        tasks.append(task)
    logger.info(f"Created {len(tasks)} tasks.")

    dataset_config = {
                'data_dir': data_args.data_path,
                'batch_size': training_args.per_device_train_batch_size,
                'min_seq_len': data_args.min_trx_seq_len,
                'max_seq_len': data_args.max_trx_seq_len,
                'shuffle': data_args.shuffle,
                'num_workers': data_args.preprocessing_num_workers,
                'pin_memory': True,
                'seed': training_args.seed
    }    
    
    datamodule = AlfaDataModule(**dataset_config)

    # Training & Callbacks
    wb_logger = WandbLogger(
        project=training_args.project_name,
        group=training_args.group_name,
        name=training_args.run_name
    )
    tb_logger = TensorBoardLogger(name=training_args.run_name,
                                  save_dir="./tb_logs",
                                  default_hp_metric=False)
    lr_monitor_callback = LearningRateMonitor(logging_interval='step')

    # Create separate checkpoints directory for each run
    save_checkpoints_dir = f"{training_args.save_checkpoints_dir}/{training_args.run_name}"

    if not Path(save_checkpoints_dir).resolve().exists():
        logger.info(f"Checkpoints path do not exists: {Path(save_checkpoints_dir).resolve()}")
        logger.info("Creating...")
        Path(save_checkpoints_dir).resolve().mkdir()
    else:
        k = 1
        while True:
            new_save_checkpoints_dir = f"{save_checkpoints_dir}_v{k}"
            if not Path(new_save_checkpoints_dir).resolve().exists():
                save_checkpoints_dir = new_save_checkpoints_dir
                Path(save_checkpoints_dir).resolve().mkdir()
                break
            k += 1

    logger.info(f"Checkpoints path created at: {Path(save_checkpoints_dir).resolve()}")


    checkpoint_callback = ModelCheckpoint(
        monitor=training_args.save_strategy_metric,  # default: 'val_loss'
        dirpath=save_checkpoints_dir,  # default: './checkpoints/'
        filename=training_args.save_filename_format,  # default: 'checkpoint-{epoch:02d}-{loss3:.2f}'
        save_weights_only=training_args.save_only_weights,  # default: 'True'
        every_n_epochs=training_args.save_epochs,  # default: '1'
        save_last=training_args.save_last_checkpoint,  # default: 'True'
        mode=training_args.save_strategy_mode,  # default: 'max'
        save_top_k=training_args.save_top_k,  # default: '3'
    )

    # Create general Tranactions QA model
    transactionsQA_model_config = {
        "warmup_steps": training_args.warmup_steps,
        "training_steps": training_args.max_steps,
        "learning_rate": training_args.learning_rate,
        "do_freeze_tm": training_args.do_freeze_transactions_model,
        "do_freeze_lm": training_args.do_freeze_language_model,
        "do_freeze_connector": training_args.do_freeze_connector,
        "connector_input_size": 384,
        "scale_parameter": training_args.scale_parameter,
        "optimizer_name": training_args.optimizer_name,
        "scheduler_name": training_args.scheduler_name,
        "use_numerical": model_args.use_numerical,
        "num_head": model_args.num_head,
        "numerical_context": model_args.numerical_context,
    }

    model = TransactionQAModel(
        language_model=lm_model,
        transaction_model=transactions_model,
        tokenizer=tokenizer,
        tasks=tasks,
        checkpoint_dir=Path(save_checkpoints_dir).resolve(),
        **transactionsQA_model_config
    )


    trainer = pl.Trainer(
        fast_dev_run=training_args.fast_dev_run,
        max_steps=training_args.max_steps,
        max_epochs=training_args.max_epochs,
        gpus=len(available_gpus),
        auto_select_gpus=True,
        log_every_n_steps=100,
        limit_train_batches=training_args.limit_train_batches,
        limit_val_batches=training_args.limit_val_batches,
        # val_check_interval=training_args.val_check_interval,
        check_val_every_n_epoch=training_args.check_val_every_n_epoch,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=training_args.gradient_clip_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        logger=wb_logger,  #[tb_logger, wb_logger],
        callbacks=[checkpoint_callback, lr_monitor_callback])
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)


if __name__ == '__main__':
    import os
    main()