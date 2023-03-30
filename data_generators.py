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


def batches_generator(list_of_paths, batch_size=1, min_seq_len=None, max_seq_len=None, is_train=True, verbose=False):
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

                if min_seq_len is not None:
                    if mask.shape[1] < min_seq_len:
                        continue
                if max_seq_len is not None:
                    if mask.shape[1] > max_seq_len:
                        continue
                
                ret = dict(num_features=torch.FloatTensor(batch_sequences[:, num_features_indices]).transpose(0, 1),
                            cat_features=torch.LongTensor(batch_sequences[:, cat_features_indices]).transpose(0, 1),
                            mask=torch.BoolTensor(batch_mask),
                            meta_features=torch.LongTensor(batch_products).unsqueeze(0),
                            app_id=torch.LongTensor(batch_app_ids)
                )

                if is_train:
                    ret['label'] = torch.LongTensor(batch_targets)

                yield ret