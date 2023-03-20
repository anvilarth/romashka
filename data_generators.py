import numpy as np
import pickle
import torch
import gc

transaction_features = ['currency', 'operation_kind', 'card_type', 'operation_type',
                        'operation_type_group', 'ecommerce_flag', 'payment_system',
                        'income_flag', 'mcc', 'country', 'city', 'mcc_category',
                        'day_of_week', 'hour', 'weekofyear', 'amnt', 'days_before', 'hour_diff']

num_features_names = ['amnt', 'days_before', 'hour_diff']
cat_features_names = [x for x in transaction_features if x not in num_features_names]
meta_features_names = ['product']

num_features_indices = [transaction_features.index(x) for x in num_features_names]
cat_features_indices = [transaction_features.index(x) for x in cat_features_names]


def batches_generator(list_of_paths, batch_size=32, shuffle=False, is_infinite=False, dry_run=False,
                      skip_number_days=None,
                      verbose=False, device=None, output_format='torch', is_train=True, min_seq_len=None,
                      max_seq_len=None, reduce_size=1.):
    """
    функция для создания батчей на вход для нейронной сети для моделей на keras и pytorch.
    так же может использоваться как функция на стадии инференса
    :param list_of_paths: путь до директории с предобработанными последовательностями
    :param batch_size: размер батча
    :param shuffle: флаг, если True, то перемешивает list_of_paths и так же
    перемешивает последовательности внутри файла
    :param is_infinite: флаг, если True,  то создает бесконечный генератор батчей
    :param verbose: флаг, если True, то печатает текущий обрабатываемый файл
    :param device: device на который положить данные, если работа на торче
    :param output_format: допустимые варианты ['tf', 'torch']. Если 'torch', то возвращает словарь,
    где ключи - батчи из признаков, таргетов и app_id. Если 'tf', то возвращает картеж: лист input-ов
    для модели, и список таргетов.
    :param is_train: флаг, Если True, то для кераса вернет (X, y), где X - input-ы в модель, а y - таргеты, 
    если False, то в y будут app_id; для torch вернет словарь с ключами на device.
    :return: бачт из последовательностей и таргетов (или app_id)
    """
    while True:
        if shuffle:
            np.random.shuffle(list_of_paths)

        for path in list_of_paths:
            if verbose:
                print(f'reading {path}')

            # Faster loading (probably)
            gc.disable()
            with open(path, 'rb') as f:
                data = pickle.load(f)

            gc.enable()

            ind_list = []
            for elem in data['targets']:
                size = elem.shape[0]
                inds = np.arange(int(size * reduce_size))
                ind_list.append(inds)

            for key in data:
                for ind in range(len(ind_list)):
                    data[key][ind] = data[key][ind][ind_list[ind]]

            padded_sequences, targets, products = data['padded_sequences'], data['targets'], data[
                'products']
            app_ids = data['app_id']
            indices = np.arange(len(products))

            if shuffle:
                np.random.shuffle(indices)
                padded_sequences = padded_sequences[indices]
                targets = targets[indices]
                products = products[indices]
                app_ids = app_ids[indices]

            for idx in range(len(products)):
                bucket, product = padded_sequences[idx], products[idx]
                app_id = app_ids[idx]

                if is_train:
                    target = targets[idx]

                for jdx in range(0, len(bucket), batch_size):
                    if dry_run:
                        yield None

                    batch_sequences = bucket[jdx: jdx + batch_size]
                    if is_train:
                        batch_targets = target[jdx: jdx + batch_size]

                    batch_products = product[jdx: jdx + batch_size]
                    batch_app_ids = app_id[jdx: jdx + batch_size]
                    mask = batch_sequences[:, -6] != 0
                    batch_sequences[:, num_features_indices[-2]] = batch_sequences[:, num_features_indices[-2]] / 365
                    batch_sequences[:, num_features_indices[-1]] = batch_sequences[:, num_features_indices[-1]] / 95

                    if min_seq_len is not None:
                        if mask.shape[1] < min_seq_len:
                            continue
                    if max_seq_len is not None:
                        if mask.shape[1] > max_seq_len:
                            continue

                    if is_train:
                        yield dict(num_features=[torch.FloatTensor(batch_sequences[:, i]).to(device) for i in
                                                 num_features_indices],
                                   cat_features=[torch.LongTensor(batch_sequences[:, i]).to(device) for i in
                                                 cat_features_indices],
                                   mask=torch.BoolTensor(mask).to(device),
                                   event_time=torch.arange(mask.shape[-1], device=device),
                                   meta_features=[torch.LongTensor(batch_products).to(device)],
                                   label=torch.LongTensor(batch_targets).to(device),
                                   app_id=batch_app_ids)

        if not is_infinite:
            break
