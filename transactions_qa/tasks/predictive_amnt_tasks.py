import torch
import random
import numpy as np

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

import transformers
from torchmetrics import Accuracy, MeanAbsoluteError, MeanSquaredError
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.text.rouge import ROUGEScore

from .task_abstract import AbstractTask
from .numeric_task_abstract import NumericTaskAbstract
from romashka.transactions_qa.utils import get_buckets_info

from romashka.transactions_qa.dataset.data_generator import (
    transaction_features,
    num_features_names
)

@dataclass
class PredExactAmountTaskBinary(NumericTaskAbstract):
    """
    A task for predictive exact Binary QA task: given a discrete or continuous numeric target - Amount,
    answer question with binary answer.
    """

    def __post_init__(self):
        self.task_name = "pred_exact_amount_binary"
        self.target_feature_name = 'amnt'  # [0, 1] range of values
        self.task_special_token = "[AMNT]"
        self.is_open_ended_task = False  # for a default for this task
        # Select to not to convert integer range to floating point number
        # (with required processing of output predictions & answers)
        self.numeric_outputs: Optional[bool] = False

        self.metrics = {
            "accuracy": BinaryAccuracy()
        }
        self.starting_prompts = [
            "This is the client's transaction history:",
            "You are given the client's transaction history:",
            "The client's transaction history is given as a context:"
        ]

        self.ending_prompts = []


        # all options for a target feature
        # self.answers_options: List[str] = [str(i) for i in get_buckets_info(self.target_feature_name,
        #                                                                     "../../assets/dense_features_buckets.pkl")]
        self.binary_answer_options: Dict[str, str] = {"positive": "Yes", "negative": "No"}
        self.answer_template: str = " "  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        # Required to specify available feature value range
        self.feature_min = 0.0
        self.feature_max = 1.0

        # If buckets are not provided externally
        if self.buckets is None:
            # Load default buckets from assets folder
            self.buckets = get_buckets_info(self.target_feature_name,
                                            "romashka/assets/dense_features_buckets.pkl")
        # Note: in this case are not str values!
        self.answers_options = self._get_buckets_ranges(self.buckets,
                                                        self.feature_min,
                                                        self.feature_max)
        self.buckets_means = self._get_buckets_means(self.buckets,
                                                     self.feature_min,
                                                     self.feature_max)

        super().__post_init__()

        if self.tokenizer is None:
            raise AttributeError("This task requires tokenizer to be set!")
        if self.add_tokens_to_tokenizer:
            self.extend_vocabulary(tokenizer=self.tokenizer,
                                   new_tokens=self.speciaxl_tokens,
