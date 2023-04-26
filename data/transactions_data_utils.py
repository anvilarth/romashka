import torch
import pickle
import numpy as np
import pandas as pd
from typing import Dict
from tqdm.notebook import tqdm

features = ['currency', 'operation_kind', 'card_type', 'operation_type', 'operation_type_group', 'ecommerce_flag',
            'payment_system', 'income_flag', 'mcc', 'country', 'city', 'mcc_category', 'day_of_week',
            'hour', 'weekofyear', 'amnt', 'days_before', 'hour_diff']

embedding_projection = {'currency': (11, 6),
                        'operation_kind': (7, 5),
                        'card_type': (175, 29),
                        'operation_type': (22, 9),
                        'operation_type_group': (4, 3),
                        'ecommerce_flag': (3, 3),
                        'payment_system': (7, 5),
                        'income_flag': (3, 3),
                        'mcc': (108, 22),
                        'country': (24, 9),
                        'city': (163, 28),
                        'mcc_category': (28, 10),
                        'day_of_week': (7, 5),
                        'hour': (24, 9),
                        'weekofyear': (53, 15),
                        'amnt': (10, 6),
                        'days_before': (23, 9),
                        'hour_diff': (10, 6)}


def pad_sequence(array, max_len) -> np.array:
    """
    принимает список списков (array) и делает padding каждого вложенного списка до max_len
    :param array: список списков
    :param max_len: максимальная длина до которой нужно сделать padding
    :return: np.array после padding каждого вложенного списка до одинаковой длины
    """
    add_zeros = max_len - len(array[0])
    return np.array([list(x) + [0] * add_zeros for x in array])


def truncate(x, num_last_transactions=750):
    return x.values.transpose()[:, -num_last_transactions:].tolist()


def make_time_batch(batch, number_days=30):
    device = batch['mask'].device
    time_tr = batch['num_features'][1] * 365

    pairwise_difference_mask = abs(time_tr.unsqueeze(1) - time_tr.unsqueeze(2)) <= number_days
    last_elements_mask = time_tr >= number_days

    last_elements_repeated = last_elements_mask.unsqueeze(2).repeat(1, 1, time_tr.shape[1])
    tmp_mask = pairwise_difference_mask * last_elements_repeated

    matrix = torch.ones(time_tr.shape[1], time_tr.shape[1])
    autoregressive_mask = torch.triu(matrix, 1).unsqueeze(0).to(device)
    final_mask = tmp_mask * autoregressive_mask

    num_repeated = batch['num_features'][0].unsqueeze(1).repeat(1, time_tr.shape[1], 1)

    all_amnt_transactions = (num_repeated * final_mask).sum(2)
    all_num_transactions = final_mask.sum(2).float()

    cat_repeated = batch['cat_features'][11].unsqueeze(1).repeat(1, time_tr.shape[1], 1) * final_mask

    res = []
    for i in range(28):
        res.append(torch.any(cat_repeated == i,  dim=2))
    all_code_transactions = torch.stack(res, dim=-1).float()

    next_time_mask = torch.any(tmp_mask, dim=2).long()

    return all_amnt_transactions, all_num_transactions, all_code_transactions, last_elements_mask.long()


def next_time_batch(batch, number_days=30):
    num_features = batch['num_features']
    cat_features = batch['cat_features']
    mask = batch['mask']

    next_time_mask = torch.zeros_like(mask, dtype=torch.long)

    all_num_transactions = torch.zeros_like(mask, dtype=torch.float)
    all_amnt_transactions = torch.zeros_like(mask, dtype=torch.float)
    all_code_transactions = torch.zeros(mask.shape[0], mask.shape[1], 28).to(all_amnt_transactions.device)

    for i in range(len(mask)):
        j = 0
        time_tr = num_features[1][i] * 365

        while j < len(time_tr):
            prost = torch.zeros(28)
            if time_tr[j] <= number_days:
                break

            k = 1
            tmp_amnt = 0.0

            while True:
                if j + k >= len(time_tr) or (time_tr[j] - time_tr[j + k] > number_days) or (time_tr[j + k] == 0.0):
                    break
                tmp_amnt += num_features[0][i][j + k].item()
                all_code_transactions[i][j][cat_features[11][i][j + k]] = 1
                k += 1

            all_amnt_transactions[i][j] = tmp_amnt

            all_num_transactions[i][j] = k - 1
            next_time_mask[i][j] = 1
            j += 1

    return all_amnt_transactions, all_num_transactions, all_code_transactions, next_time_mask


def transform_transactions_to_sequences(transactions_frame: pd.DataFrame,
                                        num_last_transactions=750) -> pd.DataFrame:
    """
    принимает frame с транзакциями клиентов, сортирует транзакции по клиентам
    (внутри клиента сортирует транзакции по возрастанию), берет num_last_transactions танзакций,
    возвращает новый pd.DataFrame с двумя колонками: app_id и sequences.
    каждое значение в колонке sequences - это список списков.
    каждый список - значение одного конкретного признака во всех клиентских транзакциях.
    Всего признаков len(features), поэтому будет len(features) списков.
    Данная функция крайне полезна для подготовки датасета для работы с нейронными сетями.
    :param transactions_frame: фрейм с транзакциями клиентов
    :param num_last_transactions: количество транзакций клиента, которые будут рассмотрены
    :return: pd.DataFrame из двух колонок (app_id, sequences)
    """
    return transactions_frame \
        .sort_values(['app_id', 'transaction_number']) \
        .groupby(['app_id'])[features] \
        .apply(lambda x: truncate(x, num_last_transactions=num_last_transactions)) \
        .reset_index().rename(columns={0: 'sequences'})


def create_padded_buckets(frame_of_sequences: pd.DataFrame, bucket_info: Dict[int, int],
                          save_to_file_path=None, has_target=True):
    """
    Функция реализует sequence_bucketing технику для обучения нейронных сетей.
    Принимает на вход frame_of_sequences (результат работы функции transform_transactions_to_sequences),
    словарь bucket_info, где для последовательности каждой длины указано, до какой максимальной длины нужно делать
    padding, далее группирует транзакции по бакетам (на основе длины), делает padding транзакций и сохраняет результат
    в pickle файл, если нужно
    :param frame_of_sequences: pd.DataFrame c транзакциями (результат применения transform_transactions_to_sequences)
    :param bucket_info: словарь, где для последовательности каждой длины указано, до какой максимальной длины нужно делать
    padding
    :param save_to_file_path: опциональный путь до файла, куда нужно сохранить результат
    :param has_target: флаг, есть ли в frame_of_sequences целевая переменная или нет. Если есть, то
    будет записано в результат
    :return: возвращает словарь с следюущими ключами (padded_sequences, targets, app_id, products)
    """
    frame_of_sequences['bucket_idx'] = frame_of_sequences.sequence_length.map(bucket_info)
    padded_seq = []
    targets = []
    app_ids = []
    products = []

    for size, bucket in tqdm(frame_of_sequences.groupby('bucket_idx'), desc='Extracting buckets'):
        padded_sequences = bucket.sequences.apply(lambda x: pad_sequence(x, size)).values
        padded_sequences = np.array([np.array(x) for x in padded_sequences])
        padded_seq.append(padded_sequences)

        if has_target:
            targets.append(bucket.flag.values)

        app_ids.append(bucket.app_id.values)
        products.append(bucket['product'].values)

    frame_of_sequences.drop(columns=['bucket_idx'], inplace=True)

    dict_result = {
        'padded_sequences': np.array(padded_seq),
        'targets': np.array(targets) if targets else [],
        'app_id': np.array(app_ids),
        'products': np.array(products),
    }

    if save_to_file_path:
        with open(save_to_file_path, 'wb') as f:
            pickle.dump(dict_result, f)
    return dict_result


def read_parquet_dataset_from_local(path_to_dataset: str, start_from: int = 0,
                                    num_parts_to_read: int = 2, columns=None, verbose=False) -> pd.DataFrame:
    """
    читает num_parts_to_read партиций, преобразует их к pd.DataFrame и возвращает
    :param path_to_dataset: путь до директории с партициями
    :param start_from: номер партиции, с которой начать чтение
    :param num_parts_to_read: количество партиций, которые требуется прочитать
    :param columns: список колонок, которые нужно прочитать из партиции
    :return: pd.DataFrame
    """

    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset)
                            if filename.startswith('part')])

    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in tqdm.tqdm_notebook(chunks, desc="Reading dataset with pandas"):
        chunk = pd.read_parquet(chunk_path, columns=columns)
        res.append(chunk)
    return pd.concat(res).reset_index(drop=True)


def generate_subsequences(batch, K, m = 0.1, M=0.9):
    new_batches = [batch]
    seq_len = batch['mask'].shape[1]

    length = torch.rand(size=(K,)) * (M - m) + m
    int_length = (length * seq_len).type(torch.int)

    indices = seq_len - int_length
    start_indices = (indices * torch.rand_like(length)).type(torch.int)

    for (start, end) in zip(start_indices, start_indices + int_length):
        new_batch = {}
        for elem in batch:
            new_feature_list = []
            if elem == 'label':
                new_batch[elem] = batch[elem]

            elif type(batch[elem]) == torch.Tensor:
                new_batch[elem] = batch[elem][:, start:end]

            elif elem == 'meta_features':
                for feature in batch[elem]:
                    new_feature_list.append(feature)
                new_batch[elem] = new_feature_list

            elif elem == 'app_id':
                pass

            else:
                for feature in batch[elem]:
                    new_feature = feature[:, start: end]
                    new_feature_list.append(new_feature)

                new_batch[elem] = new_feature_list
        new_batches.append(new_batch)

    return new_batches
