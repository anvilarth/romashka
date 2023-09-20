import gc
import torch
import pickle
import numpy as np
from time import time
from typing import List, Optional

# Set up logging
from romashka.logging_handler import get_logger

logger = get_logger(
    name="pretrain_data_generator"
)

from romashka.transactions_qa.dataset.serializers import AbstractSerializer
from romashka.transactions_qa.dataset.data_generator import (num_features_indices,
                                                             cat_features_indices)


def segment(sequence: np.ndarray,
            max_seq_len: int,
            drop_last: Optional[bool] = True) -> np.ndarray:
    """
    Segment given sequence with windows of 'max_seq_len' (over first dim).
    Optionally drops last not full segment.
    """
    segmented_sequence = []
    for i in range(0, len(sequence), max_seq_len):
        sub_seq = sequence[i: i + max_seq_len]

        if drop_last and (sub_seq.shape[0] < max_seq_len):
            continue

        segmented_sequence.append(sub_seq)

    return segmented_sequence


def text_batches_generator_raw(list_of_paths: List[str],
                               batch_size: Optional[int] = 1,
                               sub_seq_len: Optional[int] = 10,
                               serializer: Optional[AbstractSerializer] = None,
                               verbose: Optional[bool] = False):
    """
    Infinite generator for time-based data reading and collation in padded batches.
    Args:
        list_of_paths: List[str], a list of paths to data files;
        batch_size: int, a number of samples in a single batch;
        sub_seq_len: int, a maximum event sequence length.
        serializer: a Serializer instance for text caption generation;
        verbose: bool, indicates whether to print results.

    Returns:
        a dict of features (as tensors) and captions (as strings).
    """
    rng = np.random.default_rng()  # a Generator

    for path in list_of_paths:
        # Faster loading (probably)
        if verbose:
            print(f'reading {path}')

        gc.disable()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        gc.enable()

        padded_sequences, products, app_ids = data['padded_sequences'], data['products'], data['app_id']

        # all splitted sequences as [num_splitted_seq, num_features, each_seq_len]
        # -> [num_splitted_seq, 18, ~10]
        splitted_sub_sequences = []
        splitted_app_ids = []
        splitted_product_ids = []

        # 1) Split all buckets in file to separate sub_histories (independed of app_id (== user_id))
        # but with respect to paddings (mask out pad values in transactions)
        start = time()
        for idx in range(len(products)):
            bucket, product, app_id = padded_sequences[idx], products[idx], app_ids[idx]

            bucket[:, num_features_indices[-2]] = bucket[:, num_features_indices[-2]] / 365
            bucket[:, num_features_indices[-1]] = bucket[:, num_features_indices[-1]] / 95
            mask = bucket[:, -6] != 0

            for jdx in range(0, len(bucket)):
                mask_ = mask[jdx]
                # from [num_features, hist_len] -> [hist_len, num_features]
                bucket_ = bucket[jdx].swapaxes(0, 1)
                bucket_ = segment(bucket_[mask_], max_seq_len=sub_seq_len, drop_last=True)
                if len(bucket_):
                    bucket_ = np.vstack([b.swapaxes(0, 1)[np.newaxis, ...] for b in bucket_])
                    splitted_sub_sequences.append(bucket_)
                    splitted_app_ids.append(np.full((bucket_.shape[0],), app_id[jdx]))
                    splitted_product_ids.append(np.full((bucket_.shape[0],), product[jdx]))
        end = time()
        print(f"Time for split file data to segments = {(end - start) * 1000} ms.")
        splitted_app_ids = np.concatenate(splitted_app_ids)
        splitted_product_ids = np.concatenate(splitted_product_ids)
        splitted_sub_sequences = np.vstack(splitted_sub_sequences)

        # 2) Shuffle inside one file
        indices = np.arange(len(splitted_sub_sequences))
        rng.shuffle(indices)

        splitted_app_ids = splitted_app_ids[indices]
        splitted_product_ids = splitted_product_ids[indices]
        splitted_sub_sequences = splitted_sub_sequences[indices]

        # 3) With given batch_size iterate over sub_histories
        for kdx in range(0, len(splitted_sub_sequences), batch_size):
            batch_sequences = splitted_sub_sequences[kdx: kdx + batch_size]
            batch_products = splitted_product_ids[kdx: kdx + batch_size]
            batch_app_ids = splitted_app_ids[kdx: kdx + batch_size]
            batch_mask = np.ones((batch_size, batch_sequences.shape[-1]), dtype=int)  # as no padding added

            # 4) Generate text captions here: 1 per sequence -> num_captions == batch_size
            if serializer is not None:
                start = time()
                batch_captions = [serializer.serialize_batch(features=hist_features.swapaxes(0, 1))
                                  for hist_features in batch_sequences]
                end = time()
                print(f"Time for single sample serialization = {(end - start) * 1000} ms.")
            else:
                batch_captions = [""] * len(batch_sequences)

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
            print(f'reading {path}')

        gc.disable()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        gc.enable()

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
