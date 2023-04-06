from abc import ABC

# DTO
from dataclasses import dataclass
from typing import Optional

from romashka.transactions_qa.tasks import AbstractTask

@dataclass
class CategoricalTaskAbstract(AbstractTask, ABC):

    """
    Parent class for all tasks with categorical target feature.
    """

    num_classes: Optional[int] = None
    ignore_class_index: Optional[int] = None
    decision_threshold: Optional[float] = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.target_feature_type = 'cat_features'