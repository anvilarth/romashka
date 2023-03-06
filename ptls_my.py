import torch
import numpy as np

from copy import deepcopy
from torch.utils.data import IterableDataset
from transformers import GPT2Model, GPT2Config, BertConfig, BertModel, T5Config, T5Model
from transformers import AutoModel, AutoConfig

from embedding import EmbeddingLayer, LinearMapping
from data_generators import batches_generator
from tools import LambdaLayer, calculate_embedding_size

from ptls.nn.seq_encoder.containers import SeqEncoderContainer
from ptls.nn.seq_encoder.abs_seq_encoder import AbsSeqEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyPaddedBatch:
    def __init__(self, data, mask):
        self.payload = data
        self.seq_lens = torch.LongTensor([data.shape[1]] * data.shape[0]).to(device)
        self.seq_len_mask = mask
       
    
class IterDataset(IterableDataset):
    def __init__(self, dataset_train, batch_size=64, device='cuda', min_seq_len=200):
        self.data = dataset_train
        self.batch_size = batch_size
        self.device = device
        self.map = lambda: batches_generator(self.data, batch_size=self.batch_size, shuffle=True, device=self.device, is_train=True, output_format='torch', min_seq_len=min_seq_len)

    def __iter__(self):
        return self.map()


class PtlsEmbeddingLayer(EmbeddingLayer):
    def __init__(self, *args, **kwargs):
        #self.splitter = splitter
        super().__init__(*args, **kwargs)
        self.output_size = self.get_embedding_size()

    def forward(self, x):
        mask = x['mask']
        x = super().forward(x)
        return MyPaddedBatch(x, mask)    
    
    
class MySampleUniform:
    """
    Sub samples with equal length = `seq_len`
    Start pos has fixed uniform distribution from sequence start to end with equal step
    |---------------------|       main sequence
    |------|              |        sub seq 1
    |    |------|         |        sub seq 2
    |         |------|    |        sub seq 3
    |              |------|        sub seq 4
    There is no random factor in this splitter, so sub sequences are the same every time
    Can be used during inference as test time augmentation
    """
    def __init__(self, split_count, seq_len, **_):
        self.split_count = split_count
        self.seq_len = seq_len

    def split(self, dates):
        date_len = dates.shape[0]
        date_range = np.arange(date_len)

        if date_len <= self.seq_len + self.split_count:
            return [date_range for _ in range(self.split_count)]

        start_pos = np.linspace(0, date_len - self.seq_len, self.split_count).round().astype(int)
        return [date_range[s:s + self.seq_len] for s in start_pos]

    
def split_process(batch, splitter):
    res = {}
    
    local_date = batch['event_time']
    if splitter is not None:
        indexes = splitter.split(local_date)
        pad_size = max([len(ixs) for ixs in indexes])
    
    for k, v in batch.items():
        if type(v) == list and len(v) > 1 and splitter is not None:
            new_v = []
            for elem in v:
                tmp = []
                for i, ixs in enumerate(indexes):
                    to_tmp = elem[:, ixs]
                    if to_tmp.shape[1] < pad_size:
                        to_tmp = torch.cat([
                            to_tmp, torch.zeros(to_tmp.shape[0], pad_size - to_tmp.shape[1]).to(device)
                        ], axis=1)
                    tmp.append(to_tmp)
                new_v.append(torch.cat(tmp, dim=0))
        else:
            new_v = v 
        res[k] = new_v
    res['mask'] = res['cat_features'][0] != 0
    return res


def replace_token(batch, replace_prob=0.15, skip_first=1):
    mask = batch['mask']
    to_replace = torch.bernoulli(mask * replace_prob).bool()
    to_replace[:, :skip_first] = False

    sampled_trx_ids = torch.multinomial(
        mask.flatten().float(),
        num_samples=to_replace.sum().item(),
        replacement=True,
    )

    to_replace_flatten = to_replace.flatten()
    new_x = deepcopy(batch)
    for k, v in new_x.items():
        if type(v) == list and len(v) > 1:
            for elem in v:
                elem.flatten()[to_replace_flatten] = elem.flatten()[sampled_trx_ids]
    return new_x, to_replace.long().flatten()#[mask.flatten().bool()]


def my_collate_fn(batch, splitter, rep=5, mode='coles'):
    batch = batch[0]
    len_batch = batch['num_features'][0].shape[0]
    labels = torch.arange(len_batch).repeat(rep)
    batch = split_process(batch, splitter)
    
    if mode == 'coles':
        return batch, labels
    
    if mode == 'cpc':
        return batch, None
    
    if mode == 'rtd':
        batch, labels = replace_token(batch)
        return batch, labels
        
    if mode == 'mlm':
        return batch


class MyEncoder(AbsSeqEncoder):
    def __init__(self,
                 input_size=None,
                 is_reduce_sequence=False,
                 encoder_type='gpt',
                 num_layers=2,
                 num_heads=1,
                 pretrained=True
                ):
    
        super().__init__(is_reduce_sequence=is_reduce_sequence)
        self.encoder_type = encoder_type
        print(f'Sequential Encoder is {self.encoder_type}')
        #print(f'Num layers is {num_layers}')
        #print(f'Num heads is {num_heads}')
        
        if self.encoder_type == 'gpt':
            configuration = GPT2Config(n_positions=2048,
                                       n_embd=input_size, n_layer=num_layers,
                                       n_head=num_heads, resid_pdrop=0.1,
                                       embd_pdrop=0.1, attn_pdrop=0.1)
            
            self.encoder = GPT2Model(configuration)
        elif self.encoder_type == 'bert':
            configuration = BertConfig(hidden_size=input_size,
                                       num_hidden_layers=num_layers, num_attention_heads=num_heads,
                                       intermediate_size=512, hidden_dropout_prob=0.1,
                                       attention_probs_dropout_prob=0.1,
                                       max_position_embeddings=2048)
            
            self.encoder = BertModel(configuration)
        elif self.encoder_type == 't5':
            configuration = T5Config(d_model=input_size,
                                     d_kv=input_size // 1, d_ff=512,
                                     num_layers=num_layers, num_heads=num_heads)
            
            self.encoder = T5Model(configuration)
            
        elif self.encoder_type == 'whisper/small':
            config_name = 'openai/whisper-small'
            encoder_type, encoder_size = self.encoder_type.split('/')
            
            print('pretrained', pretrained)
            
            if pretrained:
                model = AutoModel.from_pretrained(config_name)
            else:
                config = AutoConfig.from_pretrained(config_name)
                model  = AutoModel.from_config(config)
                
            self.encoder = model.decoder
            self.encoder.embed_positions = LambdaLayer(lambda x: 0)
            
            hidden_size = calculate_embedding_size(model)
            self.mapping_embedding = LinearMapping(input_size, hidden_size)
        
        self.hidden_size = input_size
        self.input_size = input_size
        
    def forward(self, x, mask):
        """
        :param x:
        :param h_0: None or [1, B, H] float tensor
                    0.0 values in all components of hidden state of specific client means no-previous state and
                    use starter for this client
                    h_0 = None means no-previous state for all clients in batch
        :return:
        """
        #shape = x.payload.size()
        if self.encoder_type == 'whisper/small':
            embedding = self.mapping_embedding(x.payload, attention_mask=mask)
            out = self.encoder(inputs_embeds=embedding, attention_mask=mask).last_hidden_state[:, -1]
        
        elif self.encoder_type == 't5':
            out = self.encoder(inputs_embeds=x.payload,
                               decoder_inputs_embeds=x.payload,
                               attention_mask=mask).last_hidden_state#[:, -1]
        else:
            out = self.encoder(inputs_embeds=x.payload,
                               attention_mask=mask).last_hidden_state#[:, -1]
                
        return out

    @property
    def embedding_size(self):
        return self.hidden_size

    
class MySeqEncoder(SeqEncoderContainer):
    def __init__(self,
                 trx_encoder=None,
                 input_size=None,
                 is_reduce_sequence=False,
                 **seq_encoder_params,
                 ):
        super().__init__(
            trx_encoder=trx_encoder,
            seq_encoder_cls=MyEncoder,
            input_size=input_size,
            seq_encoder_params=seq_encoder_params,
            is_reduce_sequence=is_reduce_sequence,
        )

    def forward(self, x, h_0=None):
        mask = x['mask']
        x = self.trx_encoder(x)
        x = self.seq_encoder(x, mask)
        return x
