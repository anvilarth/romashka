import pandas as pd
import torch
import torch.nn as nn
import pickle
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class TransactionClickStreamDataset(Dataset):
    def __init__(self, 
                 datapath='/home/jovyan/data/vtb/merged.pkl',
                 mapping_mcc='/home/jovyan/romashka/assets/vtb_mapping_mcc.pkl',
                 mapping_click='/home/jovyan/data/vtb/grouping_click.pkl',
                 mapping_curr='/home/jovyan/romashka/assets/vtb_mapping_curr.pkl',
                 device='cuda:0'):
        
        super().__init__()
        
        with open(datapath, 'rb') as f:
            dataframe = pickle.load(f)
            
        with open(mapping_click, 'rb') as f:
            self.mapping_click = pickle.load(f)
        
        click_unique = len(set(self.mapping_click.values()))
        
        with open(mapping_curr, 'rb') as f:
            self.mapping_curr = pickle.load(f)
            
        # with open(mapping_mcc, 'rb') as f:
        #     self.mapping_mcc = pickle.load(f)
            
            
        self.splits_mcc = np.load('/home/jovyan/data/vtb/splits_mcc.npy')
        
        self.dataframe = dataframe
        self.transaction_features = ['mcc_code', 'currency_rk', 'day', 'month', 'hour', 'weekday', 'transaction_dttm']
        self.clickstream_features = ['cat_id', 'timestamp', 'day', 'hour', 'month', 'weekday']
        self.features2use = ['cat_id', 'day', 'hour', 'month', 'weekday', 'mcc_code', 'currency_rk']
        self.num_features2use = ['minutes_before']
        self.cat_embedding_projections = {'cat_id':(click_unique, 32), 'day':(31, 10), 'hour':(25, 10), 'month':(13, 10), 'weekday':(10, 10), 
                                          'mcc_code':(len(self.splits_mcc), 32), 'currency_rk':(3, 2)}
        self.num_embedding_projections = {'minutes_before': (10, 6)}
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        une_transaction = pd.DataFrame(self.dataframe.transactions.iloc[idx], columns=self.transaction_features)
        condition = (une_transaction.mcc_code == -1) | (une_transaction.currency_rk == -1)
        une_transaction = une_transaction.drop(une_transaction[condition].index)
        
        une_transaction.mcc_code = np.digitize(une_transaction.mcc_code, self.splits_mcc)
        une_transaction.currency_rk = une_transaction.currency_rk.apply(lambda x: self.mapping_curr[x])
            
        une_clickstream = pd.DataFrame(self.dataframe.clickstream.iloc[idx], columns=self.clickstream_features)
        une_clickstream['weekday'] += 1
        une_transaction['weekday'] += 1
        
        une_clickstream.cat_id = une_clickstream.cat_id.apply(lambda x: self.mapping_click[x])
        un_click = une_clickstream.rename(columns={'timestamp':'time'})
        un_trans = une_transaction.rename(columns={'transaction_dttm':'time'})
        
        new_df = pd.concat([un_click, un_trans]).sort_values('time').fillna(0).tail(1000)
        
        p = new_df['time'][:-1].dt.to_pydatetime()
        t = new_df['time'][1:].dt.to_pydatetime()

        new_df['minutes_before'] = [1]+ list(map(lambda x: divmod(x.total_seconds(), 60)[0], (t-p)))
        new_df['minutes_before'] = new_df['minutes_before'].apply(lambda x: np.clip(np.log(x) / np.log(14 * 24 * 60), 0, 1.0))
        
        return {'cat_features': torch.stack([torch.LongTensor(new_df[feature].values).to(self.device) for feature in self.features2use], dim=-1), 
                'num_features': torch.stack([torch.FloatTensor(new_df[feature].values).to(self.device) for feature in self.num_features2use], dim=-1),
                'time': torch.stack(new_df['time'].apply(lambda x: torch.FloatTensor(x.to_pydatetime().timetuple()[:6])).values.tolist())}
    
    @staticmethod
    def collate_fn(batch):
        cat_features = [elem['cat_features'] for elem in batch]
        num_features = [elem['num_features'] for elem in batch]
        time_features = [elem['time'] for elem in batch]
        
        cat_res = pad_sequence(cat_features, batch_first=True)
        num_res = pad_sequence(num_features, batch_first=True)
        time_res = pad_sequence(time_features, batch_first=True)
        
        mask = cat_res[..., 4] != 0
        cat_data = cat_res
        num_data = num_res
        time_data = time_res
        
        return {'cat_features': list(cat_data.permute(2, 0, 1)), 
                'num_features': list(num_data.permute(2, 0, 1)), 
                'time': time_data,
                'meta_features': [], 
                'labels': [], 
                'mask': mask[:, 1:]}
    
    
class TransactionClickStreamDatasetTransactions(Dataset):
    def __init__(self, 
                 datapath='/home/jovyan/data/vtb/merged.pkl',
                 mapping_mcc='/home/jovyan/romashka/assets/vtb_mapping_mcc.pkl',
                 mapping_click='/home/jovyan/romashka/assets/vtb_mapping_click.pkl',
                 mapping_curr='/home/jovyan/romashka/assets/vtb_mapping_curr.pkl',
                 device='cuda:0'):
        
        super().__init__()
        
        with open(datapath, 'rb') as f:
            dataframe = pickle.load(f)
            
        with open(mapping_click, 'rb') as f:
            self.mapping_click = pickle.load(f)
            
        with open(mapping_curr, 'rb') as f:
            self.mapping_curr = pickle.load(f)
            
        with open(mapping_mcc, 'rb') as f:
            self.mapping_mcc = pickle.load(f)
            
        self.dataframe = dataframe
        self.transaction_features = ['mcc_code', 'currency_rk', 'day', 'month', 'hour', 'weekday', 'transaction_dttm']
        self.features2use = ['day', 'hour', 'month', 'weekday', 'mcc_code', 'currency_rk']
        self.num_features2use = ['minutes_before']
        self.cat_embedding_projections = {'day':(31, 10), 'hour':(25, 10), 'month':(13, 10), 'weekday':(10, 10), 
                                          'mcc_code':(len(self.mapping_mcc), 32), 'currency_rk':(3, 2)}
        self.num_embedding_projections = {'minutes_before': (10, 6)}
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        une_transaction = pd.DataFrame(self.dataframe.transactions.iloc[idx], columns=self.transaction_features)
        condition = (une_transaction.mcc_code == -1) | (une_transaction.currency_rk == -1)
        une_transaction = une_transaction.drop(une_transaction[condition].index)
        
        une_transaction.mcc_code = une_transaction.mcc_code.apply(lambda x: self.mapping_mcc[x])
        une_transaction.currency_rk = une_transaction.currency_rk.apply(lambda x: self.mapping_curr[x])
        une_transaction = une_transaction.rename(columns={'transaction_dttm':'time'})
            
        une_transaction['weekday'] += 1
        
        
        new_df = une_transaction.tail(1000)  
        p = new_df['time'][:-1].dt.to_pydatetime()
        t = new_df['time'][1:].dt.to_pydatetime()

        new_df['minutes_before'] = [1]+ list(map(lambda x: divmod(x.total_seconds(), 60)[0], (t-p)))
        new_df['minutes_before'] = new_df['minutes_before'].apply(lambda x: np.clip(np.log(x) / np.log(14 * 24 * 60), 0, 1.0))
        
        return {'cat_features': torch.stack([torch.LongTensor(new_df[feature].values).to(self.device) for feature in self.features2use], dim=-1), 
                'num_features': torch.stack([torch.FloatTensor(new_df[feature].values).to(self.device) for feature in self.num_features2use], dim=-1)}
    
    @staticmethod
    def collate_fn(batch):
        cat_features = [elem['cat_features'] for elem in batch]
        num_features = [elem['num_features'] for elem in batch]
        
        cat_res = pad_sequence(cat_features, batch_first=True)
        num_res = pad_sequence(num_features, batch_first=True)
        
        mask = cat_res[..., 4] != 0
        cat_data = cat_res[:, :-1]
        num_data = num_res[:, :-1]
        
        return {'cat_features': list(cat_data.permute(2, 0, 1)), 'num_features': list(num_data.permute(2, 0, 1)), 'meta_features': [], 'labels': [], 'mask': mask[:, 1:]}
    
    
    
class TransactionClickStreamDatasetClickstream(Dataset):
    def __init__(self, 
                 datapath='/home/jovyan/data/vtb/merged.pkl',
                 mapping_mcc='/home/jovyan/romashka/assets/vtb_mapping_mcc.pkl',
                 mapping_click='/home/jovyan/romashka/assets/vtb_mapping_click.pkl',
                 mapping_curr='/home/jovyan/romashka/assets/vtb_mapping_curr.pkl',
                 device='cuda:0'):
        
        super().__init__()
        
        with open(datapath, 'rb') as f:
            dataframe = pickle.load(f)
            
        with open(mapping_click, 'rb') as f:
            self.mapping_click = pickle.load(f)
            
        with open(mapping_curr, 'rb') as f:
            self.mapping_curr = pickle.load(f)
            
        with open(mapping_mcc, 'rb') as f:
            self.mapping_mcc = pickle.load(f)
            
        self.dataframe = dataframe
        self.clickstream_features = ['cat_id', 'timestamp', 'day', 'hour', 'month', 'weekday']
        self.features2use = ['cat_id', 'day', 'hour', 'month', 'weekday']
        self.num_features2use = ['minutes_before']
        self.cat_embedding_projections = {'cat_id':(len(self.mapping_click), 32), 'day':(31, 10), 'hour':(25, 10), 'month':(13, 10), 'weekday':(10, 10)}
        self.num_embedding_projections = {'minutes_before': (10, 6)}
        self.device = device

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        une_clickstream = pd.DataFrame(self.dataframe.clickstream.iloc[idx], columns=self.clickstream_features)
        une_clickstream['weekday'] += 1
        
        une_clickstream.cat_id = une_clickstream.cat_id.apply(lambda x: self.mapping_click[x])
        une_clickstream = une_clickstream.rename(columns={'timestamp':'time'})
        
        new_df = une_clickstream.tail(1000)
        p = new_df['time'][:-1].dt.to_pydatetime()
        t = new_df['time'][1:].dt.to_pydatetime()

        new_df['minutes_before'] = [1]+ list(map(lambda x: divmod(x.total_seconds(), 60)[0], (t-p)))
        new_df['minutes_before'] = new_df['minutes_before'].apply(lambda x: np.clip(np.log(x) / np.log(14 * 24 * 60), 0, 1.0))
        
        return {'cat_features': torch.stack([torch.LongTensor(new_df[feature].values).to(self.device) for feature in self.features2use], dim=-1), 
                'num_features': torch.stack([torch.FloatTensor(new_df[feature].values).to(self.device) for feature in self.num_features2use], dim=-1)}
    
    @staticmethod
    def collate_fn(batch):
        cat_features = [elem['cat_features'] for elem in batch]
        num_features = [elem['num_features'] for elem in batch]
        
        cat_res = pad_sequence(cat_features, batch_first=True)
        num_res = pad_sequence(num_features, batch_first=True)
        
        mask = cat_res[..., 4] != 0
        cat_data = cat_res[:, :-1]
        num_data = num_res[:, :-1]
        
        return {'cat_features': list(cat_data.permute(2, 0, 1)), 'num_features': list(num_data.permute(2, 0, 1)), 'meta_features': [], 'labels': [], 'mask': mask[:, 1:]}