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