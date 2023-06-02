from dataclasses import dataclass, field
from pathlib import Path
from typing import (Optional, Union, Dict, List, Any)
from transformers.trainer_utils import (
    EvaluationStrategy,
    SchedulerType,
    ShardedDDPOption
)
# from transformers import TrainingArguments
from ..logging_handler import get_logger
from romashka.transactions_qa.layers.connector import CONNECTOR_TYPES

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

    language_model_type: Optional[str] = field(
        default="decoder",
        metadata={
            "help": (
                "The text model type for corresponding Hugging Face class creation: decoder or encoder-decoder."
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

    connector_type: Optional[str] = field(
        default="linear",
        metadata={
            "help": (
                "The Connector layer(-s) type. Can be one from: linear, recurrent, transformer."
            )
        },
    )

    connector_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The connector model checkpoint for weights initialization. "
                "Don't set if you want to train a model from scratch."
            )
        },
    )

    connector_input_size: Optional[int] = field(
        default=None,
        metadata={"help": "A connector layer input size (as an embedding size of Transactions model)."},
    )

    connector_output_size: Optional[int] = field(
        default=None,
        metadata={"help": "A connector layer output size (as an embedding size of Language model)."},
    )

    num_queries: Optional[int] = field(
        default=32,
        metadata={"help": "A number of trainable queries for Q-Former connector."},
    )

    shared_dim: Optional[int] = field(
        default=768,
        metadata={"help": "A shared dimension for contrastive loss calculation."},
    )

    intermediate_size: Optional[int] = field(
        default=1024,
        metadata={"help": "An intermediate dimension for Q-Former connector (usually 1024 or 2048)."},
    )

    num_attention_heads: Optional[int] = field(
        default=8,
        metadata={"help": "A number of attention heads for Q-Former connector."},
    )

    num_hidden_layers: Optional[int] = field(
        default=4,
        metadata={"help": "A number of hidden layers for Q-Former connector."},
    )

    connector_hidden_size: Optional[int] = field(
        default=512,
        metadata={"help": "A size of a hidden layers of Q-Former connector."},
    )

    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    def __post_init__(self):
        if self.connector_type not in CONNECTOR_TYPES:
            raise ValueError(f"`connector_type` should be one from:\n{CONNECTOR_TYPES}.")



@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    local_config: Optional[str] = field(
        default="romashka/configs/local_config.yaml",
        metadata={"help": "A path to yaml file with unique configs for local setting"},
    )

    data_path: Optional[str] = field(default="data", metadata={"help": "The input data files base path."})
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
        default=None,
        metadata={"help": "A list of comma-separated task names, that would be used during training."},
    )
    task_kwargs: Optional[List[Dict[str, Any]]] = field(
        default_factory=list,
        metadata={"help": "A list of dictionary-like arguments for tasks creation."}
    )
    special_task_token_type: Optional[Union[str, int]] = field(
        default='TASK_SPECIFIC',
        metadata={"help": "A special task token type naming scheme. "
                          "Can be one of: [TASK_SPECIFIC, ATTRIBUTE_SPECIFIC, ANSWER_SPECIFIC]"},
    )

    def __post_init__(self):
        if self.task_names is None:
            raise ValueError("No tasks provided for training!")
        elif isinstance(self.task_names, list):
            if (len(self.task_names) == 1) and isinstance(self.task_names[0], str):
                self.task_names = [task_name.strip() for task_name in self.task_names[0].split(",")]
            else:
                raise ValueError(f"Unable to handle task_names provided in this format!")

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
        default=True, metadata={"help": "Whether to freeze weights of Language model during training."}
    )
    do_freeze_transactions_model: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze weights of Transactions model during training."}
    )
    do_freeze_connector: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze weights of a Connector layer during training."}
    )
    do_freeze_language_model_embeddings: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze Embeddings of Language model during training."}
    )

    # -----------------

    text_loss_scale: Optional[float] = field(
        default=1.0,
        metadata={"help": "A scaling factor for general text-based loss (usually Cross-Entropy)."},
    )

    retrieval_loss_scale: Optional[float] = field(
        default=1.0,
        metadata={"help": "A scaling factor for retrieval from embeddings loss (usually kind of Contractive loss)."},
    )
    
    gradient_clip_val: float = field(
        default=5.0,
        metadata={"help": "Clipping norm of gradients. If ||g|| < val, g = val * g / ||g||. "
                          "The clip value is usually between 0.5 and 10, depending on how harsh you want "
                          "to clip large gradients."},
    )

    gradient_clip_algorithm: Optional[str] = field(
        default='norm',
        metadata={"help": "The gradient clipping algorithm to use. Pass gradient_clip_algorithm='value' "
                          "to clip by value, and gradient_clip_algorithm='norm' to clip by norm. "
                          "By default it will be set to 'norm'."},
    )

    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: Optional[float] = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: Optional[float] = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: Optional[float] = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: Optional[float] = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: Optional[float] = field(default=1.0, metadata={"help": "Max gradient norm."})

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
    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="linear",
        metadata={"help": "The scheduler type to use. Can be one from: `linear`, `cosine`, "
                  "`cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`"},
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
        default='checkpoint-{epoch:02d}-{loss:.2f}',
        metadata={"help": "The checkpoint save filename template."},
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
