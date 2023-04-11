import torch
import torch.nn as nn
import random

# DTO
from dataclasses import dataclass
from typing import (Dict, Tuple, List,
                    Any, Optional)

import transformers
from torchmetrics.text.rouge import ROUGEScore
from torchmetrics import AUROC
from torchmetrics.classification import BinaryAccuracy

from .task_abstract import AbstractTask
from src.losses import NextTransactionLoss

from src.data.alfa.components.data_generator import (transaction_features, num_features_names, cat_features_names)


@dataclass
class NextFeatureStandardTask(AbstractTask):
    tokenizer: transformers.PreTrainedTokenizerBase = None
    task_feature: str = 'hour_diff'

    def __post_init__(self):
        self.task_name = 'next_transaction'
        self.target_feature_name = 'amnt'
        self.is_open_ended_task = False  # for a default for this task
        
        self.metrics = nn.ModuleDict({
            "auc": AUROC(task='binary'),
            "accuracy": BinaryAccuracy()
        })


        self.question_templates = [
            ("This is the client's transaction history ",
             "Will the client have a credit default? Yes or No?"),
        ]

        # all options, for a sample can be reduced to [true_mcc_code + 4 other codes]
        self.positive_answer_word = "Yes"
        self.negative_answer_word = "No"

        self.answers_options = ["Yes", "No"]
        self.answer_template = ""  # left empty for a first time
        self.add_tokens_to_tokenizer = True

        super().__post_init__()

        # TODO: fix this comments
        if self.task_type == 'text':
            if self.tokenizer is None:
                raise AttributeError("This task requires tokenizer to be set!")
            if self.add_tokens_to_tokenizer:
                new_tokens = [self.transactions_embeddings_start_token,
                            self.transactions_embeddings_end_token]
                if self.task_special_token is not None:
                    new_tokens += [self.task_special_token]
                self.extend_vocabulary(tokenizer=self.tokenizer,
                                    new_tokens=new_tokens,
                                    special=False)

            self.positive_token = self.tokenizer(self.positive_answer_word).input_ids[0]
            self.negative_token = self.tokenizer(self.negative_answer_word).input_ids[0]

        if self.task_feature in num_features_names:
            cat_feature_ids = []
            num_feature_ids = [num_features_names.index(self.task_feature)]

        elif self.task_feature in cat_features_names:
            cat_feature_ids = [cat_features_names.index(self.task_feature)]
            num_feature_ids = []

        else:
            raise AttributeError(f"Unknown task feature {self.task_feature}")


        self.criterion = NextTransactionLoss(cat_feature_ids=cat_feature_ids,
                                             num_feature_ids=num_feature_ids
        )

    def process_input_batch(self, batch: Dict[str, Any], **kwargs):
        pass

    def generate_target(self, batch: Any, **kwargs) -> Any:
        return batch

    def calculate_metrics(self, outputs, answers, task_metrics):
        metrics = {}

        targets, preds = self.process_outputs(outputs, answers)

        if 'auc' in task_metrics:
            task_metrics['auc'](preds, targets)
            metrics['auc'] = task_metrics['auc']
        
        if 'accuracy' in task_metrics:
            task_metrics['accuracy'](preds, targets)
            metrics['accuracy'] = task_metrics['accuracy']

        return metrics 

