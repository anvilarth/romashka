import os
import sys
import shutil
import yaml
from pathlib import Path
from collections import OrderedDict

import wandb
os.environ["WANDB_MODE"] = "online"

import warnings

warnings.filterwarnings("ignore")

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Use custom transformers version == 4.27.4 + modifications
# sys.path.insert(0, "/home/jovyan/abdullaeva/transactionsQA/transformers/src/")
import transformers
from transformers import HfArgumentParser

sys.path.insert(1, '/home/jovyan/abdullaeva/transactionsQA')
# for MlSpace: /home/jovyan/abdullaeva/transactionsQA
print(sys.path)

from romashka.logging_handler import get_logger
from romashka.transactions_qa.train_args import (ModelArguments, DataTrainingArguments,
                                                 TrainingArguments, TasksArguments)
from romashka.transactions_qa.dataset.data_generator import (
    cat_features_names,
    num_features_names,
    meta_features_names)

from romashka.transactions_qa.dataset.pretrain.dataloader import (TransactionCaptioningDataset,
                                                                  TransactionCaptioningDataModule)
from romashka.transactions_qa.dataset.serializers import TemplateSerializer
from romashka.transactions_qa.transactions_model.model import TransactionsModel
from romashka.transactions_qa.layers.qformer import QFormerModel
from romashka.transactions_qa.model.pretrain_qformer_model import PretrainQFormerModel
from romashka.transactions_qa.train_utils import get_warmup_steps
from romashka.transactions_qa.utils import (get_last_checkpoint, get_projections_maps)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

        # Datasets & Dataloader & Other utils
        template = f"Transaction with the amount: %s; hour: %s; the day of week: %s; week of year: %s;" \
                   f" city: %s; MCC: %s; operation type: %s;" \
                   f" card type: %s."
        selected_attributes = ['amnt', 'hour', 'day_of_week', 'weekofyear',
                               'city', 'mcc', 'operation_type', 'card_type']

        serializer = TemplateSerializer(template=template,
                                        selected_feature_names=selected_attributes)

        if training_args.do_train and "train" in data_files:
            train_dataset_config = {
                'dataset': data_files['train'],
                'sub_seq_len': data_args.max_trx_seq_len,
                'seed': training_args.seed,
                'generator_batch_size': 1,
                'buffer_size': data_args.shuffle_buffer_size,
                'batch_size': training_args.per_device_train_batch_size,
                'num_workers': data_args.preprocessing_num_workers,
                'serializer': serializer,
                'shuffle': True
            }
        else:
            train_dataset_config = None
        logger.info(f"Created train config.")
        if training_args.do_eval and "validation" in data_files:
            val_dataset_config = {
                'dataset': data_files['validation'],
                'sub_seq_len': data_args.max_trx_seq_len,
                'seed': training_args.seed,
                'buffer_size': 0,
                'generator_batch_size': 1,
                'batch_size': training_args.per_device_eval_batch_size,
                'num_workers': data_args.preprocessing_num_workers,
                'serializer': serializer,
                'shuffle': True
            }
        else:
            val_dataset_config = None
        logger.info(f"Created validation config.")
        if (not training_args.do_train) and (not training_args.do_eval):
            logger.error("There is nothing to do. Please pass `do_train` and/or `do_eval`.")
            return 1

    datamodule = TransactionCaptioningDataModule(train_dataset_config, val_dataset_config)
    logger.info(f"Created Captioning Datamodule.")

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
    renamed_state_dict = OrderedDict()
    for key, param in ckpt.items():
        key_ = key
        if key.startswith("head"):
            key_ = ".".join(["head", key])
        elif key.startswith("encoder"):
            key_ = ".".join(["encoder_model", key])
        elif key.startswith("mapping_embedding"):
            key_ = ".".join(['connector', 'connector'] + key.split(".")[1:])
        renamed_state_dict[key_] = param

    logger.info(f"Renaming & loading transactions model...")
    transactions_model.load_state_dict(renamed_state_dict)

    # Configure and load from HF hub LM for Q-Former model
    logger.info(f"Loading Language model: `{model_args.language_model_name_or_path}`...")

    # Create connector
    if model_args.shared_dim is not None:
        shared_dim = model_args.shared_dim
    else:
        shared_dim = 768
        logger.warning(f"Unable to estimate shared ( == Language model input embeddings) dimension, "
                       f"use default setting: {shared_dim}")

    trns_output_size = None
    if model_args.connector_input_size is not None:
        trns_output_size = model_args.connector_input_size
    else:
        trns_output_size = 384
        logger.warning(f"Unable to estimate Transactions model output embeddings dimension, "
                       f"use default setting: {trns_output_size}")

    # Connector args are hardcoded, should be changed here
    qformer_config = {
        "text_model_name": model_args.language_model_name_or_path,  # bert-mini
        "sequence_len": trns_output_size,
        "num_queries": model_args.num_queries,
        "shared_dim": shared_dim,
        "hidden_size": 256,
        "num_attention_heads": model_args.num_attention_heads,
        "num_hidden_layers": model_args.num_hidden_layers,
        "intermediate_size": 1024,
        "cross_attention_frequency": 2,
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "initializer_range": 0.02,
        "max_position_embeddings": 1024,
        "max_text_sequence_len": 512,
        "truncation_side": "right",
        "position_embedding_type": "absolute",
        "device": device
    }

    qformer = QFormerModel(
        **qformer_config
    )

    # Full model
    training_model_config = {
        "learning_rate": training_args.learning_rate,
        "scheduler_type": training_args.lr_scheduler_type,
        "adam_beta1": training_args.adam_beta1,
        "adam_beta2": training_args.adam_beta2,
        "adam_epsilon": training_args.adam_epsilon,
        "warmup_steps": get_warmup_steps(
            num_training_steps=training_args.max_steps,
            num_warmup_steps=training_args.warmup_steps,
            warmup_ratio=training_args.warmup_ratio),
        "training_steps": training_args.max_steps
    }
    full_model = PretrainQFormerModel(
        language_model_name=model_args.language_model_name_or_path,
        sequence_encoder_model=transactions_model,
        qformer=qformer,
        qformer_kwargs=qformer_config,
        **training_model_config
    )

    # Training & Callbacks
    wb_logger = WandbLogger(
        project=training_args.project_name,
        group=training_args.group_name,
        name=training_args.run_name
    )
    # log gradients, parameter histogram and model topology
    # wb_logger.watch(model, log="all")

    # log gradients and model topology
    wb_logger.watch(full_model, log_graph=False)

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
        logger.info(f"Checkpoints path created at: {Path(save_checkpoints_dir).resolve()}")

    checkpoint_callback = ModelCheckpoint(
        monitor=training_args.save_strategy_metric,  # default: 'val_loss'
        dirpath=save_checkpoints_dir,  # default: './checkpoints/'
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
        log_every_n_steps=100,
        # val_check_interval=training_args.val_check_interval,
        reload_dataloaders_every_n_epochs=1,
        gradient_clip_val=training_args.gradient_clip_val,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        logger=wb_logger,  # [tb_logger, wb_logger],
        callbacks=[checkpoint_callback, lr_monitor_callback])

    trainer.fit(model=full_model, datamodule=datamodule)

if __name__ == '__main__':
    import os

    # os.environ['HF_DATASETS_OFFLINE'] = '1'  # offline mode for HF datasets
    # os.environ['TRANSFORMERS_OFFLINE'] = '1'  # offline mode for HF Transformers
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # disable DataParallel for test

    # Pretrained models are downloaded and locally cached at: ~/.cache/huggingface/transformers/.
    # This is the default directory given by the shell environment variable TRANSFORMERS_CACHE.
    # os.environ['TRANSFORMERS_CACHE'] = "/Users/abdullaeva/Documents/Projects/TransactionsQA/checkpoints/cache"
    # or "/home/jovyan/.cache/huggingface/hub"
    # os.environ['HF_DATASETS_CACHE'] = "/Users/abdullaeva/Documents/Projects/TransactionsQA/checkpoints/cache"
    # or "/home/jovyan/.cache/huggingface/datasets"
    main()