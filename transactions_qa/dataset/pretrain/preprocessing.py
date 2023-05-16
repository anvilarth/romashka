import gc
import tqdm
import pickle
import numpy as np
from time import time
from typing import List, Optional
from pathlib import Path

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


def preprocess(list_of_paths: List[str],
               serializer: AbstractSerializer,
               save_size: Optional[int] = 10_000,
               sub_seq_len: Optional[int] = 10,
               save_folder: Optional[str] = "./processed_segmented/"):
    """
    Infinite generator for time-based data reading and collation in padded batches.
    Args:
        list_of_paths: List[str], a list of paths to data files;
        save_size: int, a number of samples in a single file;
        sub_seq_len: int, a maximum event sequence length.
        serializer: a Serializer instance for text caption generation;
        save_folder: str, a folder where to save preprocessed data.

    Returns:
        a dict of features (as tensors) and captions (as strings).
    """
    rng = np.random.default_rng()  # a Generator

    for path in list_of_paths:
        # Faster loading (probably)
        logger.info(f'\n\nReading {path}')

        gc.disable()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        gc.enable()

        padded_sequences, products, app_ids = data['padded_sequences'], data['products'], data['app_id']
        logger.info(f"Initial app ids of size: {app_ids.shape}")
        logger.info(f"Initial product ids of size: {products.shape}")
        logger.info(f"Initial subsequences of size: {padded_sequences.shape}")

        # all splitted sequences as [num_splitted_seq, num_features, each_seq_len]
        # -> [num_splitted_seq, 18, ~10]
        splitted_sub_sequences = []
        splitted_app_ids = []
        splitted_product_ids = []
        splitted_captions = []

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
                    # Generate text captions here: 1 per sequence -> num_captions == batch_size
                    batch_captions = [serializer.serialize_batch(features=hist_features.swapaxes(0, 1))
                                      for hist_features in bucket_]
                    splitted_captions.extend(batch_captions)

        end = time()
        logger.info(f"Time for split file data to segments = {(end - start) * 1000} ms.")
        splitted_app_ids = np.concatenate(splitted_app_ids)
        splitted_product_ids = np.concatenate(splitted_product_ids)
        splitted_sub_sequences = np.vstack(splitted_sub_sequences)
        splitted_captions = np.asarray(splitted_captions)

        # 2) Shuffle inside one file
        indices = np.arange(len(splitted_captions))
        rng.shuffle(indices)

        splitted_app_ids = splitted_app_ids[indices]
        splitted_product_ids = splitted_product_ids[indices]
        splitted_sub_sequences = splitted_sub_sequences[indices]
        splitted_captions = splitted_captions[indices]

        logger.info(f"Splitted app ids of size: {splitted_app_ids.shape}")
        logger.info(f"Splitted product ids of size: {splitted_product_ids.shape}")
        logger.info(f"Splitted subsequences of size: {splitted_sub_sequences.shape}")
        logger.info(f"Splitted & serialized captions of size: {len(splitted_captions)}")

        for kdx in tqdm.tqdm(range(0, len(splitted_sub_sequences), save_size),
                             total=int(len(splitted_sub_sequences) // save_size)):
            dict_result = dict(app_id=splitted_app_ids[kdx: kdx + save_size],
                               products=splitted_product_ids[kdx: kdx + save_size],
                               padded_sequences=splitted_sub_sequences[kdx: kdx + save_size],
                               captions=splitted_captions[kdx: kdx + save_size])

            save_to_file_path = Path(save_folder).resolve() / (Path(path).stem + f"_{kdx}.pkl")
            with open(save_to_file_path, 'wb') as f:
                pickle.dump(dict_result, f)