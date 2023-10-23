import os
import sys
import shutil
import yaml
from pathlib import Path
from collections import OrderedDict

import wandb
os.environ["WANDB_MODE"] = "online"
os.environ["WANDB_API_KEY"] = "de71b243e187c02735ee3d741c05d2d906905d2b"

import warnings
warnings.filterwarnings("ignore")

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

# Use custom transformers version == 4.27.4 + modifications
# sys.path.insert(0, "/home/jovyan/abdullaeva/transactionsQA/transformers/src/")
import transformers
from transformers import (AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          AutoTokenizer,
                          AutoConfig,
                          HfArgumentParser)
import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

sys.path.insert(1, '/home/jovyan/abdullaeva/transactionsQA')
# for MlSpace: /home/jovyan/abdullaeva/transactionsQA
print(sys.path)

from romashka.logging_handler import get_logger
from romashka.transactions_qa.train_args import (ModelArguments, DataTrainingArguments,
                                                 TrainingArguments, TasksArguments)
from romashka.transactions_qa.dataset.data_generator import (
    transaction_features,
    cat_features_names,
    num_features_names,
    meta_features_names)

from romashka.transactions_qa.dataset.text.dataloader import (TransactionCaptioningDataset,
                                                              TransactionCaptioningDataModule)

from romashka.transactions_qa.transactions_model.model import TransactionsModel

from romashka.transactions_qa.model import ESQATextModel
from romashka.transactions_qa.train_utils import get_warmup_steps

from romashka.transactions_qa.tasks import AutoTask
from romashka.transactions_qa.utils import (get_last_checkpoint, get_projections_maps)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, TasksArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, tasks_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, tasks_args = parser.parse_args_into_dataclasses()

    # print(f"\n\nmodel_args:\n{model_args}")
    # print(f"\n\ndata_args:\n{data_args}")
    # print(f"\n\ntraining_args:\n{training_args}")
    # print(f"\n\ntasks_args:\n{tasks_args}")
    # return 0

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

    # Configure and load from HF hub LM model
    logger.info(f"Loading Language model: `{model_args.language_model_name_or_path}`...")
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
        "return_unused_kwargs": True,
        # "tie_word_embeddings": True
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
    # Make encoder-decoder model for LM
    model_loading_kwargs = {
        "pretrained_model_name_or_path": model_args.language_model_name_or_path,
        "config": config
    }

    if model_args.language_model_type == "encoder-decoder":
        lm_model = AutoModelForSeq2SeqLM.from_pretrained(**model_loading_kwargs)  # .half()
    else:
        # Otherwise try to create decoder-only model for CLM
        lm_model = AutoModelForCausalLM.from_pretrained(**model_loading_kwargs)  # .half()

    # Create tasks
    tasks = []
    task_token_type = tasks_args.special_task_token_type

    task_names = tasks_args.task_names
    if isinstance(task_names, str):
        task_names = eval(task_names)
    tasks_kwargs = tasks_args.task_kwargs
    if isinstance(tasks_kwargs, str):
        tasks_kwargs = eval(tasks_kwargs)
    logger.info(f"Got task_names: {task_names} with task_kwargs: {tasks_kwargs}")

    for task_i, task_name in enumerate(task_names):
        task_kwargs = tasks_kwargs[task_i] if task_i < len(tasks_kwargs) else {}
        if "tokenizer" not in task_kwargs:
            task_kwargs['tokenizer'] = tokenizer
        if 'task_special_token_type' not in task_kwargs:
            task_kwargs['task_special_token_type'] = task_token_type
        task = AutoTask.get(task_name=task_name, **task_kwargs)
        tasks.append(task)
    logger.info(f"Created {len(tasks)} tasks.")