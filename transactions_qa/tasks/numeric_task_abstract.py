from abc import ABC
import numpy as np

# DTO
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union

from romashka.transactions_qa.tasks import AbstractTask


@dataclass
class NumericTaskAbstract(AbstractTask, ABC):
    """
    Parent class for all tasks with numeric/discrete target feature.
    """
    # Needed to switch model in to numeric mode processing
    numeric_inputs: Optional[bool] = True
    numeric_outputs: Optional[bool] = True

    # Required to specify available feature value range
    feature_min: Optional[float] = 0.0
    feature_max: Optional[float] = 1.0

    # Identifies whether the feature value passes in discretized form or in real-valued
    is_real: Optional[bool] = False

    def __post_init__(self):
        super().__post_init__()
        self.target_feature_type = 'num_features'

    @staticmethod
    def _get_buckets_ranges(buckets: List[float],
                            feature_min: Optional[float] = 0.0,
                            feature_max: Optional[float] = 1.0) -> List[Tuple[float, float]]:
        """
        Creates buckets ranges for real number feature value.
        Returns:
            a list of (min, max) tuples for each bucket.
        """
        if not len(buckets):
            raise AttributeError(f"Buckets for numeric value was not provided or empty.")

        bucket_min = feature_min
        buckets_ranges = []

        for bucket_max in buckets:
            buckets_ranges.append((round(bucket_min, 3), round(bucket_max, 3)))
            bucket_min = bucket_max

        # if feature_max > bucket_max:
        buckets_ranges.append((round(bucket_max, 3), feature_max))
        return buckets_ranges

    @staticmethod
    def _get_buckets_means(buckets: List[float],
                           feature_min: Optional[float] = 0.0,
                           feature_max: Optional[float] = 1.0) -> List[float]:
        """
        Creates buckets means for real number feature value.
        Returns:
            a list of mean values for each bucket.
        """
        if not len(buckets):
            raise AttributeError(f"Buckets for numeric value was not provided or empty.")

        bucket_min = feature_min
        buckets_means = []

        for bucket_max in buckets:
            buckets_means.append((bucket_min + bucket_max) / 2)
            bucket_min = bucket_max

        # if feature_max > bucket_max:
        buckets_means.append(feature_max)
        return buckets_means

    @staticmethod
    def _get_random_options(num_options: int,
                            feature_min: Optional[float] = 0.0,
                            feature_max: Optional[float] = 1.0,
                            as_strings: Optional[bool] = False) -> List[Union[str, float]]:
        """
        Creates random options list for real number feature value.
        Returns:
            a list of random floating point options.
        """
        options_step = (feature_max - feature_min) / num_options
        answers_options = np.arange(feature_min, feature_max, options_step, dtype=float)
        np.random.shuffle(answers_options)
        if as_strings:
            answers_options = [str(round(opt, 2)) for opt in answers_options]
        return answers_options