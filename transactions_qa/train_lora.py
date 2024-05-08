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
from pytorch_lightning.strategies import DeepSpeedStrategy

import peft
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

sys.path.insert(1, '/home/jovyan/shares/SR004.nfs2/abdullaeva/transactionsQA')
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

from romashka.transactions_qa.dataset.dataloader import (TransactionQADataset, TransactionQADataModule)

from romashka.transactions_qa.transactions_model.model import TransactionsModel

from romashka.transactions_qa.model import (EncoderRetrievalSpecTokensModel,
                                            DecoderRetrievalSpecTokensModel)

from romashka.transactions_qa.model.tqa_model import TransactionQAModel
from romashka.transactions_qa.layers.connector import (CONNECTOR_TYPES,
                                                       make_linear_connector,
                                                       make_recurrent_connector,
                                                       make_complex_linear_connector,
                                                       make_transformer_connector,
                                                       make_qformer_connector,
                                                       make_instruct_qformer_connector)
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
    # last_checkpoint = None
    # if (os.path.isdir(training_args.save_checkpoints_dir)
    #         and training_args.do_train
    #         and not training_args.overwrite_output_dir):
    #     last_checkpoint = get_last_checkpoint(training_args.save_checkpoints_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.save_checkpoints_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.save_checkpoints_dir}) already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
    #         logger.info(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
    #             "the `--save_checkpoints_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )
    # elif not os.path.exists(training_args.save_checkpoints_dir):
    #     os.makedirs(training_args.save_checkpoints_dir)
    # elif os.path.exists(training_args.save_checkpoints_dir) and training_args.overwrite_output_dir:
    #     save_checkpoints_dir = os.path.abspath(training_args.save_checkpoints_dir)
    #     shutil.rmtree(save_checkpoints_dir)
    #     os.makedirs(save_checkpoints_dir)
    # else:
    #     logger.error(f"Output directory argument: ({training_args.save_checkpoints_dir}) is not a directory!")
    #     raise AttributeError(f"Output directory argument: ({training_args.save_checkpoints_dir}) is not a directory!")

    # Get the datasets
    data_files = {}
    if data_args.train_folder is not None and training_args.do_train:
        dir_with_datasets = os.listdir(os.path.join(data_args.data_path, data_args.train_folder))
        logger.info(f"Loading train data from: {data_args.train_folder}")
        dataset_files = sorted([os.path.join(data_args.data_path, data_args.train_folder, x)
                                for x in dir_with_datasets])
        logger.info(f"Detected {len(dataset_files)} files for training.")
        data_files["train"] = dataset_files
    if data_args.validation_folder is not None and training_args.do_eval:
        dir_with_datasets = os.listdir(os.path.join(data_args.data_path, data_args.validation_folder))
        logger.info(f"Loading validation data from: {data_args.validation_folder}")
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

    # Loading Transactions model & weights
    logger.info(f"Loading Transactions model...")
    projections_maps = get_projections_maps(
        num_embedding_projections_fn='./assets/num_embedding_projections_v1.pkl',
        cat_embedding_projections_fn='./assets/cat_embedding_projections_v1.pkl',
        meta_embedding_projections_fn='./assets/meta_embedding_projections.pkl',
        relative_folder=data_args.projections_mappings_path)
    transactions_model_config = {
        "cat_features": cat_features_names,
        "cat_embedding_projections": projections_maps.get('cat_embedding_projections'),
        "num_features": num_features_names,
        "num_embedding_projections": projections_maps.get('num_embedding_projections'),
        "num_embeddings_type": model_args.num_embeddings_type,
        "meta_features": meta_features_names,
        "meta_embedding_projections": projections_maps.get('meta_embedding_projections'),
        "encoder_type": model_args.transactions_model_encoder_type,
        "head_type": model_args.transactions_model_head_type,
        "embedding_dropout": 0.1,
        "shared_dim": 768,
        "projection_type": "LINEAR",
        "add_projection": False,
        "add_l_norm": False
    }
    transactions_model = TransactionsModel(**transactions_model_config)

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
    # Make encoder-decoder model for LM
    model_loading_kwargs = {
        "pretrained_model_name_or_path": model_args.language_model_name_or_path,
        "config": config
    }
    # if training_args.do_8bit:
    #     model_loading_kwargs['load_in_8bit'] = training_args.do_8bit
    #     model_loading_kwargs['device_map'] = "auto"
    if model_args.language_model_type == "encoder-decoder":
        lm_model = AutoModelForSeq2SeqLM.from_pretrained(**model_loading_kwargs)  #.half()
    else:
        # Otherwise try to create decoder-only model for CLM
        lm_model = AutoModelForCausalLM.from_pretrained(**model_loading_kwargs)  #.half()

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

    buckets_info_path = "romashka/assets/dense_features_buckets_v1.pkl"
    logger.info(f"Running with buckets file path: {buckets_info_path}")

    for task_i, task_name in enumerate(task_names):
        task_kwargs = tasks_kwargs[task_i] if task_i < len(tasks_kwargs) else {}
        if "tokenizer" not in task_kwargs:
            task_kwargs['tokenizer'] = tokenizer
        if 'task_special_token_type' not in task_kwargs:
            task_kwargs['task_special_token_type'] = task_token_type
        if "buckets_info_path" not in task_kwargs:
            task_kwargs['buckets_info_path'] = buckets_info_path
        task = AutoTask.get(task_name=task_name, **task_kwargs)
        tasks.append(task)
    logger.info(f"Created {len(tasks)} tasks.")

    # Create connector
    lm_input_size = None
    if model_args.connector_output_size is not None:
        lm_input_size = model_args.connector_output_size
    elif hasattr(lm_model.config, "d_model"):
        lm_input_size = lm_model.config.d_model
    elif lm_model.config.hidden_size:
        lm_input_size = lm_model.config.hidden_size
    else:
        raise AttributeError(f"Unable to estimate Language model input embeddings dimension!")

    trns_output_size = None
    if model_args.connector_input_size is not None:
        trns_output_size = model_args.connector_input_size
    else:
        trns_output_size = 384
        logger.warning(f"Unable to estimate Transactions model output embeddings dimension, "
                       f"use default setting: {trns_output_size}")

    if model_args.connector_type == "linear":
        connector_args = {}
        connector = make_linear_connector(
            output_size=trns_output_size,
            input_size=lm_input_size
        )
    elif model_args.connector_type == "complex_linear":
        # Connector args are hardcoded, should be changed here
        connector_args = {
            "n_layers": 2,
            "hidden_dims": [1024],
            "add_normalizations": [False, True],
            'add_activations': [True, False]
        }
        connector = make_complex_linear_connector(
            output_size=trns_output_size,
            input_size=lm_input_size,
            **connector_args

        )
    elif model_args.connector_type == "recurrent":
        # Connector args are hardcoded, should be changed here
        connector_args = {
            'num_recurrent_layers': 2,
            'is_bidirectional': False
        }
        connector = make_recurrent_connector(
            output_size=trns_output_size,
            input_size=lm_input_size,
            **connector_args
        )
    elif model_args.connector_type == "transformer":
        # Connector args are hardcoded, should be changed here
        connector_args = {
            'n_layers': 1,
            'n_heads': [8],
            "ff_output_dims": [1024],
            'add_rel_pos_embeddings': [False],
            'dropouts_p': [0.1]
        }
        connector = make_transformer_connector(
            output_size=trns_output_size,
            input_size=lm_input_size,
            **connector_args
        )
    elif model_args.connector_type == "qformer":
        # Connector args are hardcoded, should be changed here
        qformer_config = {
            "hidden_size": model_args.connector_hidden_size,
            "num_attention_heads": model_args.num_attention_heads,
            "num_hidden_layers": model_args.num_hidden_layers,
            "intermediate_size": model_args.intermediate_size,
            "cross_attention_frequency": 2,
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "initializer_range": 0.02,
            "max_position_embeddings": 1024,
            "position_embedding_type": "absolute",
            "connector_model_name_or_path": '/home/jovyan/shares/SR004.nfs2/abdullaeva/transactionsQA/pretrained_weights/q-former-blip2-with-whisper-small-pretrained/state_dict.pt'
        }
        if ("connector_model_name_or_path" in qformer_config) and \
                not os.path.exists(qformer_config["connector_model_name_or_path"]):
            raise FileExistsError(f"Initialization file doesn't exist: {qformer_config['connector_model_name_or_path']}")

        # if (model_args.connector_model_name_or_path is not None) \
        #         and ("bert" in model_args.connector_model_name_or_path):
        #     qformer_config['text_model_name'] = model_args.connector_model_name_or_path,  # bert-mini/small/base
        # elif (model_args.connector_model_name_or_path is not None) \
        #         and ("blip2" in model_args.connector_model_name_or_path):  # init from BLIP2-FLAN-T5 QFormer model
        #     qformer_config['connector_model_name_or_path'] = model_args.connector_model_name_or_path,  # bert-mini/small/base

        connector_args = {
            'config': qformer_config,
            'from_hf': True,
            "from_checkpoint": model_args.connector_model_name_or_path is not None,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "num_queries": model_args.num_queries
        }
        connector = make_qformer_connector(
            output_size=trns_output_size,
            input_size=lm_input_size,
            **connector_args
        )
    elif model_args.connector_type == "instruct_qformer":

        # Connector args are hardcoded, should be changed here
        qformer_config = {
            "hidden_size": model_args.connector_hidden_size,
            "num_attention_heads": model_args.num_attention_heads,
            "num_hidden_layers": model_args.num_hidden_layers,
            "intermediate_size": model_args.intermediate_size,
            "cross_attention_frequency": 2,
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "initializer_range": 0.02,
            "max_position_embeddings": 1024,
            "position_embedding_type": "absolute",
        }

        connector_args = {
            'config': qformer_config,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "num_queries": model_args.num_queries
        }
        connector = make_instruct_qformer_connector(
            output_size=trns_output_size,
            input_size=lm_input_size,
            **connector_args
        )

    else:
        raise AttributeError(f"Unknown connector type: {model_args.connector_type}")

    # Create general LLM model
    if model_args.language_model_type == "encoder-decoder":
        lm_model_config = {
            "do_freeze_tm": training_args.do_freeze_transactions_model,
            "do_freeze_lm": training_args.do_freeze_language_model,
            "do_freeze_connector": training_args.do_freeze_connector,
            "do_freeze_lm_embeddings": training_args.do_freeze_language_model_embeddings,
            "min_ret_tokens": model_args.min_ret_tokens,
            "max_ret_tokens": model_args.max_ret_tokens,
            "n_retrieval_layers": [-1],
            "retrieval_loss_scale": training_args.retrieval_loss_scale,
            "text_loss_scale": training_args.text_loss_scale,
            "embeddings_dropout_p": 0.1,
            "add_temporal_embeddings": model_args.add_temporal_embeddings,
            "transactions_embeddings_start_token": r"[trx]",
            "transactions_embeddings_end_token": r"[/trx]",
        }
        # EncoderRetrievalModel
        model_ = EncoderRetrievalSpecTokensModel(
            language_model=lm_model,
            transaction_model=transactions_model,
            tasks=tasks,
            tokenizer=tokenizer,
            connector=connector,
            is_debug=True,
            **lm_model_config
        )
    else:
        lm_model_config = {
            "do_freeze_tm": training_args.do_freeze_transactions_model,
            "do_freeze_lm": training_args.do_freeze_language_model,
            "do_freeze_connector": training_args.do_freeze_connector,
            "do_freeze_lm_embeddings": training_args.do_freeze_language_model_embeddings,
            "min_ret_tokens": model_args.min_ret_tokens,
            "max_ret_tokens": model_args.max_ret_tokens,
            "n_retrieval_layers": [-1],
            "retrieval_loss_scale": training_args.retrieval_loss_scale,
            "text_loss_scale": training_args.text_loss_scale,
            "embeddings_dropout_p": 0.1,
            "transactions_embeddings_start_token": r"[trx]",
            "transactions_embeddings_end_token": r"[/trx]",
        }
        model_ = DecoderRetrievalSpecTokensModel(
            language_model=lm_model,
            transaction_model=transactions_model,
            tasks=tasks,
            tokenizer=tokenizer,
            connector=connector,
            is_debug=True,
            **lm_model_config
        )

    # Create general Tranactions QA model
    transactionsQA_model_config = {
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
        "training_steps": training_args.max_steps,
        "save_checkpoints_dir": f"{training_args.save_checkpoints_dir}/{training_args.run_name}"
    }

    model = TransactionQAModel(model=model_,
                               tasks=tasks,
                               **transactionsQA_model_config,
                               **lm_model_config,  # as additional kwargs -> to save hyperparameters to checkpoint
                               **connector_args)
    # PEFT
    # Create general LLM model
    if model_args.language_model_type == "encoder-decoder":
        # Define LoRA Config for Encoder-Decoder
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
    else:
        # Define LoRA Config for Decoder-only
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
    if training_args.do_8bit:
        # prepare int-8 model for training
        lm_model = prepare_model_for_int8_training(lm_model)

    # add LoRA adaptor
    lm_model = get_peft_model(lm_model, lora_config)
    lm_model.print_trainable_parameters()

    # Datasets & Dataloader & Other utils
    if training_args.do_train and "train" in data_files:
        train_dataset_config = {
            'dataset': data_files['train'],
            'min_seq_len': data_args.min_trx_seq_len,
            'max_seq_len': data_args.max_trx_seq_len,
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
            'min_seq_len': data_args.min_trx_seq_len,
            'max_seq_len': data_args.max_trx_seq_len,
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

    datamodule = TransactionQADataModule(train_dataset_config, val_dataset_config)

    # Training & Callbacks
    wb_logger = WandbLogger(
        project=training_args.project_name,
        group=training_args.group_name,
        name=training_args.run_name
    )
    # log gradients, parameter histogram and model topology
    # wb_logger.watch(model, log="all")

    # log gradients and model topology
    wb_logger.watch(model, log_graph=False)
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
        save_top_k=1,  #training_args.save_top_k,  # default: 1
        mode=training_args.save_strategy_mode,  # default: 'min' for monitor='val_loss'
    )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,  #training_args.early_stopping_patience,
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

    trainer_kwargs = dict(
        fast_dev_run=training_args.fast_dev_run,
        max_steps=-1 if until_convergence else training_args.max_steps,
        max_epochs=-1 if until_convergence else training_args.max_epochs,
        strategy=DeepSpeedStrategy(
            stage=2,
            offload_optimizer=False,
            offload_parameters=False,
        ),
        log_every_n_steps=10,
        reload_dataloaders_every_n_epochs=1,
        precision=training_args.precision,  # bf16 - ?
        gradient_clip_val=training_args.gradient_clip_val,
        gradient_clip_algorithm=training_args.gradient_clip_algorithm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        logger=wb_logger,  # [tb_logger, wb_logger],
        callbacks=callbacks
    )
    if int(pl.__version__[0]) == 1:
        trainer_kwargs['gpus'] = len(available_gpus)
        trainer_kwargs['auto_select_gpus'] = True
    else:
        trainer_kwargs['accelerator'] = "gpu"
        trainer_kwargs['devices'] = "auto"


    trainer = pl.Trainer(**trainer_kwargs)

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
