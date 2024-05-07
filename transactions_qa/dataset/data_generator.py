import gc
import torch
import pickle
import random
import numpy as np
from typing import List, Optional

transaction_features = ['currency', 'operation_kind', 'card_type', 'operation_type',
                        'operation_type_group', 'ecommerce_flag', 'payment_system',
                        'income_flag', 'mcc', 'country', 'city', 'mcc_category',
                        'day_of_week', 'hour', 'weekofyear', 'amnt', 'days_before', 'hour_diff']

num_features_names = ['amnt', 'days_before', 'hour_diff']
cat_features_names = [x for x in transaction_features if x not in num_features_names]
meta_features_names = ['product']

num_features_indices = [transaction_features.index(x) for x in num_features_names]
cat_features_indices = [transaction_features.index(x) for x in cat_features_names]


def batches_generator(list_of_paths: List[str],
                      batch_size: int = 1,
                      min_seq_len: int = 50, max_seq_len: int = 750,
                      is_train: bool = True, verbose: bool = True):
    """
    Infinite generator for time-based data reading and collation in padded batches.
    Args:
        list_of_paths: List[str], a list of paths to data files;
        batch_size: int, a number of samples in a single batch;
        min_seq_len: int, a minimum event sequence length.
                    Note: all sequences shorter than this argument's value will be skipped.
        max_seq_len: int, a maximum event sequence length.
                    Note: all sequences longer than this argument's value will be skipped.
        is_train: bool, indicates whether dataset contains target values.
        verbose: bool, indicates whether to print results.
    """
    for path in list_of_paths:
        # Faster loading (probably)
        if verbose:
            print(f'reading {path}')

        gc.disable()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        gc.enable()

        padded_sequences, products = data['padded_sequences'], data['products']
        app_ids = data['app_id']
        padded_sequences_realval = None

        if 'padded_sequences_realval' in data:
            padded_sequences_realval = data['padded_sequences_realval']

        if is_train:
            targets = data['targets']

        for idx in range(len(products)):
            bucket, product = padded_sequences[idx], products[idx]
            if padded_sequences_realval is not None:
                real_valued_bucket = padded_sequences_realval[idx]
            app_id = app_ids[idx]

            if is_train:
                target = targets[idx]

            mask = bucket[:, -6] != 0

            for jdx in range(0, len(bucket), batch_size):

                batch_sequences = bucket[jdx: jdx + batch_size]
                if padded_sequences_realval is not None:
                    batch_realvals = real_valued_bucket[jdx: jdx + batch_size]

                if is_train:
                    batch_targets = target[jdx: jdx + batch_size]

                batch_products = product[jdx: jdx + batch_size]
                batch_app_ids = app_id[jdx: jdx + batch_size]

                batch_mask = mask[jdx: jdx + batch_size]

                # TODO Maybe just clip max_len sequence for more data
                if min_seq_len is not None:
                    if mask.shape[1] < min_seq_len:
                        continue
                if max_seq_len is not None:
                    if mask.shape[1] > max_seq_len:
                        continue

                ret = dict(
                    num_features=torch.FloatTensor(batch_sequences[:, num_features_indices]).transpose(0, 1),
                    cat_features=torch.LongTensor(batch_sequences[:, cat_features_indices]).transpose(0, 1),
                    mask=torch.BoolTensor(batch_mask),
                    meta_features=torch.LongTensor(batch_products).unsqueeze(0),
                    app_id=torch.LongTensor(batch_app_ids)
                )
                if padded_sequences_realval is not None:
                    ret['real_num_features'] = torch.FloatTensor(batch_realvals).transpose(0, 1)
                if is_train:
                    ret['label'] = torch.LongTensor(batch_targets)
                yield ret


def batches_balanced_generator(list_of_paths: List[str],
                               batch_size: Optional[int] = 1,
                               min_seq_len: Optional[int] = None, max_seq_len: Optional[int] = None,
                               balance_max: Optional[int] = None,
                               is_train: Optional[bool] = True,
                               verbose: Optional[bool] = False):
    """
    Infinite generator for time-based data reading and collation in padded batches.
    Args:
        list_of_paths: List[str], a list of paths to data files;
        batch_size: int, a number of samples in a single batch;
        min_seq_len: int, a minimum event sequence length.
                    Note: all sequences shorter than this argument's value will be skipped.
        max_seq_len: int, a maximum event sequence length.
                    Note: all sequences longer than this argument's value will be skipped.
        balance_max: int, a maximum number of samples to oversample the minority class to;
        is_train: bool, indicates whether dataset contains target values.
        verbose: bool, indicates whether to print results.
    """
    # worker_total_num = torch.utils.data.get_worker_info().num_workers
    # worker_id = torch.utils.data.get_worker_info().id
    # print(f"\nWorker id #{worker_id} / {worker_total_num} workers.")

    for path in list_of_paths:
        # Faster loading (probably)
        if verbose:
            print(f'reading {path}')

        gc.disable()
        with open(path, 'rb') as f:
            data = pickle.load(f)
        gc.enable()

        padded_sequences, products = data['padded_sequences'], data['products']
        app_ids = data['app_id']

        if is_train and ("targets" in data.keys()):
            targets = data['targets']

        for idx in range(len(products)):
            bucket, product = padded_sequences[idx], products[idx]
            app_id = app_ids[idx]

            # Max target count per bucket
            bucket_balance_max = balance_max if balance_max is not None else 0

            if is_train and ("targets" in data.keys()):
                target = targets[idx]
                unique, counts = np.unique(target, return_counts=True)
                target_to_counts = dict(zip(unique, counts))

                # estimate max target count from given labels
                # in case when `balance_max` was not provided
                if bucket_balance_max == 0:
                    for target_name, cnt in target_to_counts.items():
                        bucket_balance_max = cnt if cnt > bucket_balance_max else bucket_balance_max

                bucket_balance_max = bucket_balance_max * 0.5   # as hot-fix to not to oversample tooo much

                # Collect samples per target class
                samples_by_targets = dict()
                for target_name in unique:
                    inds = np.where(target == target_name)
                    samples_by_targets[target_name] = [inds[0], bucket[inds], product[inds]]

                # Oversample the classes with fewer elements than the max
                for target_name in unique:

                    target_inds = []
                    target_buckets = []
                    target_products = []

                    while target_to_counts[target_name] < bucket_balance_max:
                        # sample random index from indexes
                        addit_index = random.choice(list(range(len(samples_by_targets[target_name][0]))))

                        target_inds.append(samples_by_targets[target_name][0][addit_index])
                        target_buckets.append(samples_by_targets[target_name][1][addit_index])
                        target_products.append(samples_by_targets[target_name][2][addit_index])
                        target_to_counts[target_name] += 1

                    if len(target_inds) > 0:
                        samples_by_targets[target_name][0] = np.concatenate([samples_by_targets[target_name][0],
                                                                             np.asarray(target_inds)])
                        samples_by_targets[target_name][1] = np.concatenate([samples_by_targets[target_name][1],
                                                                             np.asarray(target_buckets)])
                        samples_by_targets[target_name][2] = np.concatenate([samples_by_targets[target_name][2],
                                                                             np.asarray(target_products)])

                # Concatenate back into bucket
                bucket = np.concatenate([target_samples[1] for _, target_samples in samples_by_targets.items()])
                product = np.concatenate([target_samples[2] for _, target_samples in samples_by_targets.items()])
                target = np.concatenate([np.full((len(target_samples[0])), target_name)
                                         for target_name, target_samples in samples_by_targets.items()])

                # Shuffle
                shuffled_indexes = np.random.permutation(len(target))
                bucket = bucket[shuffled_indexes]
                product = product[shuffled_indexes]
                target = target[shuffled_indexes]

            # bucket[:, num_features_indices[-2]] = bucket[:, num_features_indices[-2]] / 365
            # bucket[:, num_features_indices[-1]] = bucket[:, num_features_indices[-1]] / 95
            mask = bucket[:, -6] != 0

            for jdx in range(0, len(bucket), batch_size):

                batch_sequences = bucket[jdx: jdx + batch_size]

                if is_train:
                    batch_targets = target[jdx: jdx + batch_size]

                batch_products = product[jdx: jdx + batch_size]
                batch_app_ids = app_id[jdx: jdx + batch_size]

                batch_mask = mask[jdx: jdx + batch_size]

                # TODO Maybe just clip max_len sequence for more data
                if min_seq_len is not None:
                    if mask.shape[1] < min_seq_len:
                        continue
                if max_seq_len is not None:
                    if mask.shape[1] > max_seq_len:
                        continue

                ret = dict(
                    num_features=torch.FloatTensor(batch_sequences[:, num_features_indices]).transpose(0, 1),
                    cat_features=torch.LongTensor(batch_sequences[:, cat_features_indices]).transpose(0, 1),
                    mask=torch.BoolTensor(batch_mask),
                    meta_features=torch.LongTensor(batch_products).unsqueeze(0),
                    app_id=torch.LongTensor(batch_app_ids)
                )
                if is_train:
                    ret['label'] = torch.LongTensor(batch_targets)
                yield ret