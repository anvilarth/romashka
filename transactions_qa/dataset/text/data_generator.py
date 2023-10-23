import gc
import torch
import pickle
import numpy as np
from typing import List, Optional

# Set up logging
from romashka.logging_handler import get_logger

logger = get_logger(
    name="text_data_generator"
)

from romashka.transactions_qa.dataset.data_generator import (num_features_indices,
                                                             cat_features_indices)


def text_batches_generator_proc(list_of_paths: List[str],
                                batch_size: Optional[int] = 1,
                                verbose: Optional[bool] = False):
    """
    Infinite generator for time-based data reading and collation in padded batches.
    Args:
        list_of_paths: List[str], a list of paths to data files;
        batch_size: int, a number of samples in a single batch;
        verbose: bool, indicates whether to print results.

    Returns:
        a dict of features (as tensors) and captions (as strings).
    """
    for path in list_of_paths:
        # Faster loading (probably)
        if verbose:
            logger.info(f'reading {path}')

        # 1) Load and unpack
        gc.disable()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        gc.enable()

        # padded_sequences as np.array of shape [n_histories, history_len (no padding), n_features (=18)]
        # products and app_ids - 1D np.arrays of shape [n_histories]
        padded_sequences, products, app_ids = data['padded_sequences'], data['products'], data['app_id']
        captions = data['captions']

        # 2) With given batch_size iterate over sub_histories
        for kdx in range(0, len(padded_sequences), batch_size):
            batch_sequences = padded_sequences[kdx: kdx + batch_size]
            batch_products = products[kdx: kdx + batch_size]
            batch_app_ids = app_ids[kdx: kdx + batch_size]
            batch_captions = captions[kdx: kdx + batch_size]
            batch_mask = np.ones((batch_size, batch_sequences.shape[-1]), dtype=int)  # as no padding added

            # As shapes: [n_features, batch_size, hist_seq_len]
            ret = dict(
                num_features=torch.FloatTensor(batch_sequences[:, num_features_indices]).transpose(0, 1),
                cat_features=torch.LongTensor(batch_sequences[:, cat_features_indices]).transpose(0, 1),
                mask=torch.BoolTensor(batch_mask),
                meta_features=torch.LongTensor(batch_products).unsqueeze(0),
                app_id=torch.LongTensor(batch_app_ids),
                captions=batch_captions
            )

            yield ret