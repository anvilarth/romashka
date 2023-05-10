# Basic
import numpy as np
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

# Set up logging
from romashka.logging_handler import get_logger
logger = get_logger(
    name="Tasks",
    logging_level="INFO"
)

from romashka.transactions_qa.dataset.serializers import AbstractSerializer
from romashka.transactions_qa.dataset.data_generator import (transaction_features,
                                                             num_features_names,
                                                             cat_features_names,
                                                             meta_features_names,
                                                             num_features_indices,
                                                             cat_features_indices)

class TemplateSerializer(AbstractSerializer):

    def __init__(self,
                 template: str,
                 selected_feature_names: List[str],
                 feature2idx: Optional[Dict[str, int]] = None,
                 idx2feature: Optional[Dict[int, str]] = None,
                 verbose: Optional[bool] = False):
        """

        Args:
            template (str): a serialization template in form of formatted (with %s) string;
            selected_feature_names (List[str]): a list of sequential feature names (as they occur in template);
            feature2idx (Dict[str, int]): a mapping from feature name to feature index;
            idx2feature (Dict[int, str]): a mapping from feature index to feature name;
            verbose (bool): whether to print info during serialization.
        """
        super(TemplateSerializer, self).__init__(feature2idx=feature2idx,
                                                 idx2feature=idx2feature,
                                                 verbose=verbose)
        self.template = template
        self.selected_feature_names = selected_feature_names
        self.num_template_features = template.count("%s")

    def _check_features(self, features: np.ndarray,
                        feature_names: Optional[List[str]] = None,):
        if len(features) != self.num_template_features:
            raise AttributeError(f"A number of passed features ({len(features)}) does not equal "
                                 f"to required by template ({self.num_template_features})!")
        if feature_names is not None:
            if set(feature_names) != set(self.selected_feature_names):
                raise AttributeError(f"A passed list feature names:\n{feature_names}\ndoes not equal "
                                     f"to required by template:\n{self.selected_feature_names}")

    def serialize_sample(self, features: np.ndarray,
                         feature_names: Optional[List[str]] = None,
                         *args, **kwargs) -> str:
        self._check_features(features)
        if feature_names is not None:
            features_ = [str(features[feature_names.index(selected_fn)]) for selected_fn in self.selected_feature_names]
        else:
            features_ = [str(f) for f in features]

        return self.template % tuple(features_)



