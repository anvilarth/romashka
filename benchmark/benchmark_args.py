from dataclasses import dataclass, field
from typing import (Optional, Union, Dict, List, Any)

from romashka.logging_handler import get_logger

logger = get_logger(
    name="Tasks",
    logging_level="INFO"
)


@dataclass
class BenchmarkArguments:
    """
    Arguments with which model/config/tokenizer we are going to run on benchmark.
    """

    # Model settings

    model_name: Optional[str] = field(
        default="google/flan-t5-small",
        metadata={
            "help": (
                "The text model checkpoint for weights initialization. "
                "Don't set if you want to train a model from scratch."
            )
        },
    )
    tokenizer_name: Optional[str] = field(
        default="google/flan-t5-small",
        metadata={
            "help": (
                "The tokenizer name or path."
            )
        },
    )
    from_checkpoint: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to load model's weights from pretrained checkpoint file or not."},
    )
    checkpoint_model_path: Optional[str] = field(
        default="",
        metadata={
            "help": (
                "The model's checkpoint path to load from if `from_checkpoint` is set to True."
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

    # Generation settings

    max_len_input: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum length of input sequence for encoding."},
    )
    max_len_output: Optional[int] = field(
        default=96,
        metadata={"help": "Maximum length of output sequence for encoding."},
    )

    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search on decoding."},
    )
    num_return_sequences: Optional[int] = field(
        default=1,
        metadata={"help": "Number of returned sequences for beam search on decoding."},
    )

    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "Repetition penalty on decoding."},
    )
    length_penalty: Optional[float] = field(
        default=1,
        metadata={"help": "Length penalty on decoding."},
    )

    # Data settings

    dataset_name: str = field(
        default=None,
        metadata={"help": "A name of dataset to use in benchmark."},
    )
    task_names: Optional[List[str]] = field(
        default=None,
        metadata={"help": "A list of task names from the selected dataset to use in benchmark. "
                          "If None - then use all tasks from dataset."},
    )
    task_num_samples: Optional[int] = field(
        default=-1,
        metadata={"help": "A number of samples per task to use in benchmark. "
                  "If -1 - then use all samples from tasks."},
    )
    task_split: Optional[str] = field(
        default="validation",
        metadata={"help": "A name of task data split to use in benchmark."},
    )

    # Outputs settings

    seed: Optional[int] = field(
        default=11,
        metadata={"help": "A random seed for generation."},
    )

    verbose: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log all benchmark results or not."},
    )

    save_metrics_folder: Optional[str] = field(
        default="./metrics",
        metadata={"help": "The choice of a 1-st level folder for predictions/metrics saving. "
                          "Will be creates (if not exists)."},
    )
    save_metrics_subfolder: Optional[str] = field(
        default="",
        metadata={"help": "The choice of a 2-nd level folder for predictions/metrics saving. "
                          "It should be unique!"},
    )

    def __post_init__(self):
        pass