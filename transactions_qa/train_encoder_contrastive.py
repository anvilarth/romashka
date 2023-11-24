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

import transformers
from transformers import (AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          AutoTokenizer,
                          AutoConfig,
                          HfArgumentParser)

sys.path.insert(1, '/home/jovyan/abdullaeva/transactionsQA')
# for MlSpace: /home/jovyan/abdullaeva/transactionsQA
print(sys.path)

from romashka.logging_handler import get_logger
from romashka.transactions_qa.train_args import (ModelArguments, DataTrainingArguments,
                                                 TrainingArguments)
from romashka.transactions_qa.dataset.data_generator import (
    transaction_features,
    cat_features_names,
    num_features_names,
    meta_features_names)
from romashka.transactions_qa.dataset.text.dataloader import (TransactionCaptioningDataset,
                                                              TransactionCaptioningDataModule)

from romashka.transactions_qa.transactions_model.model import TransactionsModel
from romashka.transactions_qa.model import TextEncoderModel, ProjectionsType, PoolerType
from romashka.transactions_qa.model.trainer_model import ContrastiveTransactionsModel
from romashka.transactions_qa.train_utils import get_warmup_steps
from romashka.transactions_qa.utils import (get_last_checkpoint,
                                            get_projections_maps)


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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
            "do_lowercase": False,
            "model_max_length": 4096
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

        if (model_args.language_model_type == "encoder-decoder") \
                or hasattr(config, "is_encoder_decoder"):
            lm_model = AutoModelForSeq2SeqLM.from_pretrained(**model_loading_kwargs)  # .half()
        else:
            # Otherwise try to create decoder-only model for CLM
            lm_model = AutoModelForCausalLM.from_pretrained(**model_loading_kwargs)  # .half()

        # Create text encoder
        text_encoder = TextEncoderModel(
            language_model=lm_model,
            tokenizer=tokenizer,
            max_input_sequence_len=4096,
            pooler_type=model_args.text_pooler_type,
            projection_type=model_args.text_projection_type,
            shared_dim=model_args.shared_dim,
            do_freeze_lm=training_args.do_freeze_language_model,
            do_freeze_lm_embeddings=training_args.do_freeze_language_model_embeddings
        )

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

        # Create general Contrastive model
        contrastive_model_config = {
            "encoder_pooler_type": model_args.encoder_pooler_type,
            "encoder_projection_type": model_args.encoder_projection_type,
            "shared_dim": model_args.shared_dim,
            "learning_rate": training_args.learning_rate,
            "scheduler_type": training_args.lr_scheduler_type,
            "optimizer_type": "AdamW",  # training_args.optimizer_type,
            "adam_beta1": training_args.adam_beta1,
            "adam_beta2": training_args.adam_beta2,
            "adam_epsilon": training_args.adam_epsilon,
            "warmup_steps": get_warmup_steps(
                num_training_steps=training_args.max_steps,
                num_warmup_steps=training_args.warmup_steps,
                warmup_ratio=training_args.warmup_ratio),
            "training_steps": training_args.max_steps
        }

        model = ContrastiveTransactionsModel(text_model=text_model,
                                             trns_encoder=transactions_model,
                                             **contrastive_model_config)

        # Datasets & Dataloader & Other utils
        if training_args.do_train and "train" in data_files:
            train_dataset_config = {
                'dataset': data_files['train'],
                'seed': training_args.seed,
                'generator_batch_size': 1,
                'buffer_size': data_args.shuffle_buffer_size,
                'batch_size': training_args.per_device_train_batch_size,
                'num_workers': data_args.preprocessing_num_workers
            }
        else:
            train_dataset_config = None

            logger.info(f"Created train dataloader.")
        if training_args.do_eval and "validation" in data_files:
            val_dataset_config = {
                'dataset': data_files['validation'],
                'seed': training_args.seed,
                'buffer_size': 0,
                'generator_batch_size': 1,
                'batch_size': training_args.per_device_eval_batch_size,
                'num_workers': data_args.preprocessing_num_workers
            }
        else:
            val_dataset_config = None

            logger.info(f"Created validation dataloader.")
        if (not training_args.do_train) and (not training_args.do_eval):
            logger.error("There is nothing to do. Please pass `do_train` and/or `do_eval`.")
            return 1

        datamodule = TransactionCaptioningDataModule(train_dataset_config, val_dataset_config)

        # Training & Callbacks
        wb_logger = WandbLogger(
            project=training_args.project_name,
            group=training_args.group_name,
            name=training_args.run_name
        )
        # log gradients, parameter histogram and model topology
        wb_logger.watch(transactions_model, log="all")

        # log gradients and model topology
        # wb_logger.watch(model, log_graph=False)
        lr_monitor_callback = LearningRateMonitor(logging_interval='step')

        # Create separate checkpoints directory for each run
        save_checkpoints_dir = f"{training_args.save_checkpoints_dir}/{training_args.run_name}"
        if not Path(save_checkpoints_dir).resolve().exists():
            logger.info(f"Checkpoints path do not exists: {Path(save_checkpoints_dir).resolve()}")
            logger.info("Creating...")
            Path(save_checkpoints_dir).resolve().mkdir()
            logger.info(f"Checkpoints path created at: {Path(save_checkpoints_dir).resolve()}")

        every_n_train_steps = 1000
        checkpoint_callback = ModelCheckpoint(
            monitor=training_args.save_strategy_metric,  # default: 'val_loss'
            dirpath=save_checkpoints_dir,  # default: './checkpoints/'
            filename=training_args.save_filename_format,  # default: 'checkpoint-{epoch:02d}-{loss3:.2f}'
            save_weights_only=training_args.save_only_weights,  # default: 'True'
            # every_n_epochs=training_args.save_epochs,  # default: '1'
            every_n_train_steps=every_n_train_steps,
            save_last=training_args.save_last_checkpoint,  # default: 'True'
            save_top_k=1,  # training_args.save_top_k,  # default: 1
            mode=training_args.save_strategy_mode,  # default: 'min' for monitor='val_loss'
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=3,  # training_args.early_stopping_patience,
            mode="min",
            strict=False,
            verbose=True
        )

        # Saving tokenizer
        if Path(save_checkpoints_dir).resolve().exists():
            logger.info(f"Checkpoints path: {Path(save_checkpoints_dir).resolve()}")
            ckpt_files = [fn.name for fn in Path(save_checkpoints_dir).glob("*.json") if not fn.is_dir()]
            if ("tokenizer_config.json" not in ckpt_files) and ("config.json" not in ckpt_files):
                save_fn = str(Path(save_checkpoints_dir).resolve())
                logger.info(f"Saving tokenizer with `save_pretrained()` to {save_fn}")
                saved_files = tokenizer.save_pretrained(save_fn)
                logger.info(f"Saved to:\n{[f for f in saved_files]}")
            else:
                tok_fn = [fn for fn in ckpt_files if ("tokenizer_config.json" in fn) or ("config.json" in fn)][0]
                logger.info(f"Pretrained tokenizer exists: {tok_fn}")

        callbacks = [checkpoint_callback, lr_monitor_callback]
        until_convergence = True
        if until_convergence:  # training_args.until_convergence:
            callbacks += [early_stopping_callback]

        trainer = pl.Trainer(
            fast_dev_run=training_args.fast_dev_run,
            # training_args.until_convergence
            max_steps=-1 if until_convergence else training_args.max_steps,
            max_epochs=-1 if until_convergence else training_args.max_epochs,
            gpus=len(available_gpus),
            auto_select_gpus=True,
            # strategy="ddp",
            log_every_n_steps=10,
            reload_dataloaders_every_n_epochs=1,
            precision=training_args.precision,  # bf16 - ?
            limit_train_batches=100_000,
            gradient_clip_val=training_args.gradient_clip_val,
            gradient_clip_algorithm=training_args.gradient_clip_algorithm,
            accumulate_grad_batches=training_args.gradient_accumulation_steps,
            logger=wb_logger,  # [tb_logger, wb_logger],
            callbacks=callbacks
        )

        trainer.fit(model=model, datamodule=datamodule)

    if __name__ == '__main__':
        import os

        # os.environ['HF_DATASETS_OFFLINE'] = '1'  # offline mode for HF datasets
        # os.environ['TRANSFORMERS_OFFLINE'] = '1'  # offline mode for HF Transformers
        # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # disable DataParallel for test

        # Pretrained models are downloaded and locally cached at: ~/.cache/huggingface/transformers/.
        # This is the default directory given by the shell environment variable TRANSFORMERS_CACHE.
        # os.environ['TRANSFORMERS_CACHE'] = "/Users/abdullaeva/Documents/Projects/TransactionsQA/checkpoints/cache"
        # or "/home/jovyan/.cache/huggingface/hub"
        # os.environ['HF_DATASETS_CACHE'] = "/Users/abdullaeva/Documents/Projects/TransactionsQA/checkpoints/cache"
        # or "/home/jovyan/.cache/huggingface/datasets"
        main()
