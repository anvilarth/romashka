# Basic
import numpy as np
from abc import ABC, abstractmethod
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

# Set up logging
from romashka.logging_handler import get_logger
logger = get_logger(
    name="Tasks",
    logging_level="INFO"
)

from romashka.transactions_qa.dataset.data_generator import (transaction_features,
                                                             num_features_names,
                                                             cat_features_names,
                                                             meta_features_names,
                                                             num_features_indices,
                                                             cat_features_indices)

class AbstractSerializer(ABC):

    def __init__(self,
                 feature2idx: Optional[Dict[str, int]] = None,
                 idx2feature: Optional[Dict[int, str]] = None,
                 verbose: Optional[bool] = False):
        """
        Abstract Serializer class.
        Args:
            feature2idx (Dict[str, int]): a mapping from feature name to feature index;
            idx2feature (Dict[int, str]): a mapping from feature index to feature name;
            verbose (bool): whether to print info during serialization.
        """
        self.feature2idx = feature2idx
        self.idx2feature = idx2feature
        self.verbose = verbose

    @abstractmethod
    def serialize_sample(self, features: np.ndarray, *args, **kwargs) -> str:
        """
        Serialize with predefined logic a single sample (event / table row / ...).
        Returns: a single string representation.
        """
        raise NotImplementedError

    def serialize_batch(self, features: np.ndarray, *args, **kwargs) -> List[str]:
        """
        Serialize with predefined logic multiple samples (event sequence / table rows / ...).
        Returns: a list of string representations, one per each input sample.
        """
        raise NotImplementedError

