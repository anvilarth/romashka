from dataclasses import dataclass, field
from pathlib import Path
from typing import (Optional, Union, Dict, List, Any)
from transformers.trainer_utils import (
    EvaluationStrategy,
    SchedulerType,
    ShardedDDPOption
)
# from transformers import TrainingArguments
from src.utils.logging_handler import get_logger

logger = get_logger(
    name="Tasks",
    logging_level="INFO"
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    transactions_model_encoder_type: Optional[str] = field(
        default="whisper/tiny",
        metadata={
            "help": (
                "The encoder type for a transactions model configuration."
            )
        },
    )

    transactions_model_head_type: Optional[str] = field(
        default="next",
        metadata={
            "help": (
                "The head type for a transactions model configuration."
            )
        },
    )

    transactions_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The transactions model checkpoint for weights initialization. "
                "Don't set if you want to train a model from scratch."
            )
        },
    )

    language_model_name_or_path: Optional[str] = field(
        default="google/flan-t5-small",
        metadata={
            "help": (
                "The text model checkpoint for weights initialization. "
                "Don't set if you want to train a model from scratch."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    def __post_init__(self):
        pass


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    local_config: Optional[str] = field(
        default="./configs/local_config.yaml",
        metadata={"help": "A path to yaml file with unique configs for local setting"},
    )

    data_path: Optional[str] = field(default="romashka/data", metadata={"help": "The input data files base path."})
    projections_mappings_path: Optional[str] = field(default=None,
                                                     metadata={"help": "The mappings files base path."})
    train_folder: Optional[str] = field(
        default="train_buckets",
        metadata={"help": "The input training data folder with preprocessed samples (in .pickle files)."}
    )
    validation_folder: Optional[str] = field(
        default="val_buckets",
        metadata={"help": "The input validations data folder with preprocessed samples (in .pickle files)."}
    )
    train_file: Optional[str] = field(
        default="train.csv",
        metadata={"help": "The input training data file (a csv/text file name for train split separation)."}
    )
    validation_file: Optional[str] = field(
        default="val.csv",
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )

    preprocessing_num_workers: Optional[int] = field(
        default=4,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )

    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    min_trx_seq_len: Optional[int] = field(
        default=0,
        metadata={
            "help": (
            "Restrict samples to have length more than `min_seq_len`. Other samples are dropped"
            )
        },
    )

    max_trx_seq_len: Optional[int] = field(
        default=250,
        metadata={
            "help": (
            "Restrict samples to have length less than `max_seq_len`. Other samples are dropped"
            )
        },
    )
    shuffle_buffer_size: Optional[int] = field(
        default=10_000,
        metadata={
            "help": (
            "Size of buffer which is used for shuffling."
            )
        },
    )

    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None and self.data_path is None:
            raise ValueError("Need either a training or a validation file names, or data path.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`train_file` should be a csv, a json or a txt file.")
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                if extension not in ["csv", "json", "txt"]:
                    raise ValueError("`validation_file` should be a csv, a json or a txt file.")
            if self.data_path is not None:
                p = Path(self.data_path).resolve()
                if p.exists():
                    files_buckets = list(p.rglob("*.pkl"))
                    logger.info(
                        f"Provided data path contains {len(files_buckets)} buckets files.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class TasksArguments:
    """
    Arguments for tasks creation.
    """
    task_names: Optional[List[str]] = field(
        default_factory=list,
        metadata={"help": "A list of task names, that would be used during training."},
    )
    task_kwargs: Optional[List[Dict[str, Any]]] = field(
        default_factory=list,
        metadata={"help": "A list of dictionary-like arguments for tasks creation."}
    )

    def __post_init__(self):
        if (len(self.task_names) > 0) and (len(self.task_kwargs) > 0):
            if len(self.task_names) != len(self.task_kwargs):
                raise ValueError("Provided tasks list does not match length with given tasks kwargs."
                                 "Check consistency for both lists and try again.")


@dataclass
class TrainingArguments:
    """
    Arguments for training/fine-tuning procedure.
    """
    output_dir: Optional[str] = field(
        default="./outputs",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    no_cuda: Optional[bool] = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    do_train: Optional[bool] = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: Optional[bool] = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: Optional[bool] = field(default=False, metadata={"help": "Whether to run predictions on the test set."})

    device: Optional[str] = field(
        default="cpu", metadata={"help": "The device to train on: GPU/TPU/core/CPU."}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    # -----------------
    do_freeze_language_model: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze weights of Language model during training."}
    )
    do_freeze_transactions_model: Optional[bool] = field(
        default=True, metadata={"help": "Whether to freeze weights of Transactions model during training."}
    )
    do_freeze_connector: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze weights of a Connector layer during training."}
    )
    
    gradient_clip_norm: float = field(
        default=1.0,
        metadata={"help": "Clipping norm of gradients. If ||g|| < val, g = val * g / ||g||."},
    )
    # -----------------
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: Optional[float] = field(default=1e-4, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: Optional[float] = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    optimizer_name: Optional[str] = field(default='AdamW', metadata={"help": "The optimizer to use."})
    scale_parameter: Optional[bool] = field(default=True, metadata={"help": "Adafactor scaling parameter"})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "Max gradient norm."})
    scheduler_name: Optional[str] = field(default='linear_schedule_with_warmup', metadata={"help": "The scheduler"})

    fast_dev_run: Optional[int] = field(
        default=False,
        metadata={"help": "Used for debugging purposes."
                          "Defines a number of batches to check that everything runs successfully without exceptions."
                          "If set to `True` when it is a single batch to run on, "
                          "otherwise (an int number) - it specifies an exact number of batches to run."
                  },
    )
    max_steps: Optional[int] = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override max_epochs."},
    )
    max_epochs: Optional[int] = field(
        default=None,
        metadata={"help": "If > 0: set total number of training epochs to perform. Override max_steps."},
    )
    val_check_interval: Optional[int] = field(
        default=1,
        metadata={"help": "How often to check the validation set. Pass a ``float`` in the range [0.0, 1.0] to check "
                          "after a fraction of the training epoch. Pass an ``int`` to check after a fixed number of training "
                          "batches."},
    )
    check_val_every_n_epoch: Optional[int] = field(
        default=1,
        metadata={"help": "How often to check the validation set."},
    )
    limit_train_batches: Optional[int] = field(
        default=1.0,
        metadata={"help": "Limit the number of train batches to run at the same time."},
    )
    limit_val_batches: Optional[int] = field(
        default=1.0,
        metadata={"help": "Limit the number of validation batches to run at the same time."},
    )
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    # -----------------
    log_level: Optional[str] = field(
        default="INFO",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
            "choices": ['INFO', "DEBUG", "WARNING", "CRITICAL"],
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )

    save_checkpoints_dir: Optional[str] = field(
        default="./checkpoints/checkpoints/",
        metadata={"help": "Where do you want to store the checkpoints of a model."},
    )

    save_strategy_metric: Optional[str] = field(
        default="val_loss",
        metadata={"help": "The checkpoint save metric to use: val_loss or train_loss or any other."},
    )
    save_strategy_mode: Optional[str] = field(
        default="min",
        metadata={"help": "The checkpoint save strategy to use: max or min"},
    )
    save_filename_format: Optional[str] = field(
        default='checkpoint-{epoch:02d}-{loss3:.2f}',
        metadata={"help": "The checkpoint save filename template."},
    )
    save_top_k: Optional[int] = field(
        default=1,
        metadata={"help": "How many checkpoints to keep."},
    )
    save_epochs: Optional[int] = field(default=1, metadata={"help": "Save checkpoint every X epochs."})
    save_last_checkpoint: Optional[bool] = field(default=True,
                                                 metadata={
                                                     "help": "Whether to save checkpoint from a very last epoch."})
    save_only_weights: Optional[bool] = field(default=True,
                                              metadata={
                                                  "help": "Whether to save only weights, without optimizer state."})
    # -----------------
    project_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the project. Notably used for wandb logging."}
    )
    group_name: Optional[str] = field(
        default='tqa', metadata={"help": "An optional descriptor for the project. Notably used for wandb logging."}
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "An optional descriptor for the run. Notably used for wandb logging."}
    )

    seed: Optional[int] = field(
        default=11, metadata={"help": "Number for seed"}
    )

    # -----------------
