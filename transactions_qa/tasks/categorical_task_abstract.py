import torch
import random
import numpy as np

from abc import ABC

# DTO
from dataclasses import dataclass
from typing import Optional, Union, List

from romashka.transactions_qa.tasks import AbstractTask


@dataclass
class CategoricalTaskAbstract(AbstractTask, ABC):

    """
    Parent class for all tasks with categorical target feature.
    """
    # Needed to switch model in to numeric mode processing
    numeric_inputs: Optional[bool] = False
    numeric_outputs: Optional[bool] = False

    num_classes: Optional[int] = None
    ignore_class_index: Optional[int] = None
    decision_threshold: Optional[float] = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.target_feature_type = 'cat_features'

    @classmethod
    def sample_random_negative(cls,
                               true_target: Union[torch.Tensor, str],
                               answers_options: List[str],
                               output_dtype: Optional[Union[str, torch.dtype]] = torch.int64) -> torch.Tensor:
        if isinstance(true_target, torch.Tensor):
            if len(true_target.size()) != 0:  # not a scalar
                raise AttributeError(f"True target is not a scalar: {true_target}")
            true_target = str(true_target.long().item())  # as an integer number string

        # negative
        rand_target = None
        while rand_target is None:
            opt = random.sample(answers_options, k=1)[0]
            if opt != true_target:
                rand_target = opt

        rand_target = torch.as_tensor([int(rand_target)], dtype=output_dtype)
        return rand_target