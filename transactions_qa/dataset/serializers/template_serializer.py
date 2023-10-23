# Basic
import numpy as np
from typing import (Dict, Tuple, List,
                    Any, Optional, Union)

# Set up logging
from romashka.logging_handler import get_logger

logger = get_logger(
    name="TemplateSerializer",
    logging_level="INFO"
)

from romashka.transactions_qa.dataset.serializers import AbstractSerializer
from romashka.transactions_qa.dataset.data_generator import transaction_features


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
        if (feature2idx is None) and (idx2feature is None):
            feature2idx = {feature_name: i for i, feature_name in enumerate(transaction_features)}
            idx2feature = {i: feature_name for i, feature_name in enumerate(transaction_features)}
        elif feature2idx is None:
            feature2idx = {feature_name: i for i, feature_name in idx2feature.items()}
        elif idx2feature is None:
            idx2feature = {i: feature_name for feature_name, i in feature2idx.items()}

        super(TemplateSerializer, self).__init__(feature2idx=feature2idx,
                                                 idx2feature=idx2feature,
                                                 verbose=verbose)
        self.template = template
        self.selected_feature_names = selected_feature_names
        self.num_template_features = template.count("%s")

    def _check_features(self, features: np.ndarray,
                        feature_names: Optional[List[str]] = None, ):
        # features can be provided in two ways:
        # - all data features (selected and others also)
        # - only selected data features (in any order if feature_names were passed)
        if (len(features) != len(self.selected_feature_names)) \
                and (len(features) != len(self.feature2idx)):
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
            features_ = [self.map_to_string(features[feature_names.index(selected_fn)])
                         for selected_fn in self.selected_feature_names]
        # Passed all data features (selected and others also)
        # Assumed, that they have the same order as was encoded in feature2idx and idx2feature mappings
        elif len(features) == len(self.feature2idx):
            features_ = [self.map_to_string(features[self.feature2idx.get(selected_fn)])
                         for selected_fn in self.selected_feature_names]
        else:
            features_ = [self.map_to_string(f) for f in features]

        return self.template % tuple(features_)

    def serialize_batch(self, features: np.ndarray,
                        feature_names: Optional[List[str]] = None,
                        *args, **kwargs) -> List[str]:
        if len(features.shape) != 2:
            raise AttributeError(f"Batch serialization requires a 2-dim array of features as input, "
                                 f"passed array of features with size: `{features.shape}`.")

        batch_serialization = []
        for i in range(features.shape[0]):
            caption_ = ""
            try:
                caption_ = self.serialize_sample(features[i], feature_names=feature_names, *args, **kwargs)
            except Exception as e:
                logger.error(f"Error occured during {i}'th sample serialization:\n{e}"
                             f"Skipping (empty string)...")
            batch_serialization.append(caption_)

        return "\n".join(batch_serialization)
