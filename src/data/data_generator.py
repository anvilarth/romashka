import gc
import torch
import pickle
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
                      batch_size: Optional[int] = 1,
                      min_seq_len: Optional[int] = None, max_seq_len: Optional[int] = None,
                      is_train: Optional[bool] = True, verbose: Optional[bool] = False):
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

    Returns:

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

        if is_train:
            targets = data['targets']

        for idx in range(len(products)):
            bucket, product = padded_sequences[idx], products[idx]
            app_id = app_ids[idx]

            if is_train:
                target = targets[idx]

            bucket[:, num_features_indices[-2]] = bucket[:, num_features_indices[-2]] / 365
            bucket[:, num_features_indices[-1]] = bucket[:, num_features_indices[-1]] / 95
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
