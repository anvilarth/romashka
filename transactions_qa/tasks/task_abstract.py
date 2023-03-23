import torch
import functools
import pandas as pd
from abc import ABC, abstractmethod

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional)

import transformers
from torchmetrics.text.rouge import ROUGEScore

from romashka.logging_handler import get_logger

logger = get_logger(
    name="Tasks",
    logging_level="INFO"
)

from romashka.data_generators import (transaction_features)


@dataclass
class AbstractTask(ABC):
    """
    Defines the abstract class for all the tasks.
    name: the name of the task.
    task_specific_config: specifies the special configuration needs
        to be passed to encoder when decoding each task. Since different
        tasks, have different output space, the maximum decoding length
        varies based on the tasks.
    metrics: specifies the metrics to evaluate the task based on them.
    ...
    """
    task_name: Optional[str] = None
    target_feature_name: Optional[str] = None
    target_feature_index: Optional[int] = None
    task_specific_config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    question_templates: Optional[List[Tuple[str, str]]] = None  # (starting, ending)
    answer_template: Optional[str] = None
    answers_options: Optional[List[str]] = None

    seed: Optional[int] = 11
    verbose: Optional[bool] = False
    transactions_embeddings_start_token: Optional[str] = r"[trx]"
    transactions_embeddings_end_token: Optional[str] = r"[/trx]"
    add_tokens_to_tokenizer: Optional[bool] = False
    is_binary_task: Optional[bool] = True  # whether the answer is binary: yes/no, true/false, or multichoice
    multichoice_separator: Optional[str] = " - %s;"
    num_options: Optional[int] = 6  # ground truth + 5 additional options
    is_few_shot: Optional[bool] = False  # whether to provide few examples before question
    n_shot: Optional[int] = 1

    def __post_init__(self):
        # Fill in empty parameters with deafults
        if self.target_feature_name not in transaction_features:
            raise AttributeError(f"Provided feature name not in available"
                                 f"transactions feature names:\n{transaction_features}")
        self.target_feature_index = transaction_features.index(self.target_feature_name) \
            if self.target_feature_index is None else self.target_feature_index

        self.task_specific_config = {
            "source_msx_seq_len": 512,
            "target_max_seq_len": 128
        } if self.task_specific_config is None else self.task_specific_config
        self.metrics = {"rouge": ROUGEScore()} if self.metrics is None else self.metrics
        self.question_templates = [
            ("This is the client's transaction history ",
             " Is the last MCC category code 1?")
        ] if self.question_templates is None else self.question_templates
        self.answer_template = [
            ""  # empty for default
        ] if self.answer_template is None else self.answer_template
        if self.answers_options is None:
            if self.is_binary_task:
                self.answers_options = [
                    "Yes", "No"
                ]
            else:
                raise ValueError(f"The task is marked as multichoice, but no choices for answers provided!")

    @abstractmethod
    def process_input_batch(self, batch: Dict[str, Any], **kwargs):
        """
        Apply task-specific processing for a batch of data.
        """
        raise NotImplementedError

    @abstractmethod
    def process_input_sample(self, sample: Any, **kwargs) -> Any:
        """
        Apply task-specific processing for a single data sample.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_target(self, sample: Any, **kwargs) -> Any:
        """
        Generated question/answer-specific target sequence.
        """
        raise NotImplementedError

    @staticmethod
    def extend_vocabulary(
            tokenizer: transformers.PreTrainedTokenizerBase,
            new_tokens: List[str],
            special: Optional[bool] = True,
            model: Optional[transformers.PreTrainedModel] = None):
        """
        Add a list of new tokens to the tokenizer class. If the new tokens are not in the vocabulary, they are added to
        it with indices starting from length of the current vocabulary.

        """
        num_added_toks = tokenizer.add_tokens(new_tokens, special_tokens=special)
        logger.info(f"Added to tokenizer: {num_added_toks} tokens.")
        if model is not None:
            # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary,
            # i.e., the length of the tokenizer.
            model.resize_token_embeddings(len(tokenizer))
        else:
            logger.info(f"Notice: resize_token_embeddings of a model to adapt to the size of the new vocabulary!")
