import gc
import tqdm
import pickle
import numpy as np
from time import time
from typing import List, Optional, Tuple
from pathlib import Path

# Set up logging
from romashka.logging_handler import get_logger

logger = get_logger(
    name="text_preprocessing"
)

from romashka.transactions_qa.dataset.serializers import AbstractSerializer
from romashka.transactions_qa.dataset.data_generator import (num_features_indices,
                                                             cat_features_indices)


def get_last_segment(sequence: np.ndarray,
                     max_seq_len: int,
                     add_paddings: Optional[bool] = True,
                     pad_value: Optional[int] = 0) -> Tuple[np.ndarray, ...]:
    """
    Returns last full part of sequence of given 'max_seq_len' (over first dim).
    Add additional paddings at the beggining of sequences shorter then 'max_seq_len'.
    If sequence was padded, then return mask.
    """
    if (sequence.shape[0] < max_seq_len) and add_paddings:
        d = max_seq_len - sequence.shape[0]
        padding = np.full((d, *sequence.shape[1:]), pad_value)
        mask = np.concatenate([np.zeros((d, *sequence.shape[1:])), np.ones_like(sequence)])
        last_segment = np.concatenate([padding, sequence])
    else:
        last_segment = sequence[-max_seq_len:]
        mask = np.ones_like(last_segment)

    return last_segment, mask


def preprocess_val(list_of_paths: List[str],
                   serializer: AbstractSerializer,
                   selected_ids: Optional[List[int]] = None,
                   save_size: Optional[int] = 10_000,
                   sub_seq_len: Optional[int] = 10,
                   save_folder: Optional[str] = "./processed_segmented/"):
    """
    Infinite generator for time-based data reading and collation in padded batches.
    Args:
        list_of_paths: List[str], a list of paths to data files;
        save_size: int, a number of samples in a single file;
        sub_seq_len: int, a maximum event sequence length;
        selected_ids: List[int], a list of transaction histories ids for evaluation;
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

        # all sequences as [num_splitted_seq, num_features, each_seq_len]
        # -> [num_splitted_seq, 18, N]
        sub_sequences = []
        segmented_app_ids = []
        segmented_product_ids = []
        captions = []

        # 1) Split all buckets in file to separate sub_histories (independed of app_id (== user_id))
        # but with respect to paddings (mask out pad values in transactions)
        start = time()
        for idx in range(len(products)):  # len(products)
            bucket, product, app_id = padded_sequences[idx], products[idx], app_ids[idx]

            if selected_ids is not None:
                app_id_list = app_id.tolist()
                found = list(set(app_id_list) & set(selected_ids))
                if not len(found):
                    continue

            # bucket[:, num_features_indices[-2]] = bucket[:, num_features_indices[-2]] / 365
            # bucket[:, num_features_indices[-1]] = bucket[:, num_features_indices[-1]] / 95
            mask = bucket[:, -6] != 0

            for jdx in range(0, len(bucket)):
                if (selected_ids is not None) and (app_id[jdx] not in found):
                    continue

                mask_ = mask[jdx]
                # from [num_features, hist_len] -> [hist_len, num_features]
                sequence = bucket[jdx].swapaxes(0, 1)
                # get last max_seq_len window
                last_segment, last_segment_mask = get_last_segment(sequence[mask_],
                                                                   sub_seq_len,
                                                                   add_paddings=False)
                if len(last_segment):
                    sub_sequences.append(last_segment)
                    segmented_app_ids.append(app_id[jdx])
                    segmented_product_ids.append(product[jdx])
                    # Generate text captions here: 1 per sequence -> num_captions == batch_size
                    caption = serializer.serialize_batch(features=last_segment)
                    captions.append(caption)

        end = time()
        logger.info(f"Time for split file data to segments = {(end - start) * 1000} ms.")
        segmented_app_ids = np.asarray(segmented_app_ids)
        segmented_product_ids = np.asarray(segmented_product_ids)
        sub_sequences = np.asarray(sub_sequences)
        captions = np.asarray(captions)

        # 2) Shuffle inside one file
        indices = np.arange(len(captions))
        rng.shuffle(indices)

        segmented_app_ids = segmented_app_ids[indices]
        segmented_product_ids = segmented_product_ids[indices]
        sub_sequences = sub_sequences[indices]
        captions = captions[indices]

        logger.info(f"Splitted app ids of size: {segmented_app_ids.shape}")
        logger.info(f"Splitted product ids of size: {segmented_product_ids.shape}")
        logger.info(f"Splitted subsequences of size: {sub_sequences.shape}")
        logger.info(f"Splitted & serialized captions of size: {len(captions)}")

        for kdx in tqdm.tqdm(range(0, len(sub_sequences), save_size),
                             total=int(len(sub_sequences) // save_size)):
            dict_result = dict(app_id=segmented_app_ids[kdx: kdx + save_size],
                               products=segmented_product_ids[kdx: kdx + save_size],
                               padded_sequences=sub_sequences[kdx: kdx + save_size],
                               captions=captions[kdx: kdx + save_size])

            fn = (Path(path).stem + f"_{kdx}.pkl")
            save_to_file_path = Path(save_folder).resolve() / fn
            with open(save_to_file_path, 'wb') as f:
                pickle.dump(dict_result, f, protocol=4)
            logger.info(f"Saved to {fn}")