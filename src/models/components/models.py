import torch
import sys
import torch.nn as nn

import timm
import torch.nn.functional as F
from time import time
from transformers import GPT2Model, GPT2Config, BertConfig, BertModel, T5Config, T5Model
from transformers import DecisionTransformerModel, Wav2Vec2Model, Data2VecAudioModel, Data2VecTextModel
from transformers import HubertForSequenceClassification, AutoModel, ViTModel
from transformers import PerceiverModel, Data2VecVisionModel, AutoConfig
import copy

from .embedding import EmbeddingLayer, PerceiverMapping, LinearMapping, IdentityMapping

from .head import LinearHead, RNNClassificationHead, NSPHead, MLPHead, TransformerHead, IdentityHead, NextActionsHead, \
    ClassificationHead
from .tools import LambdaLayer, calculate_embedding_size, zero_function

# sys.path.append('/home/jovyan/romashka/adapter_transformers')
# from adapter_transformers import AutoAdapterModel

# sys.path.append('/home/jovyan/romashka/perceiver-pytorch')
# from perceiver_pytorch import Perceiver

# from adapter_transformers import GPT2Model as GPT2ModelAdapter, BertModel as BertModelAdapter

config_names = {'decision-transformer': 'edbeeching/decision-transformer-gym-hopper-expert',
                'wav2vec2/large': "facebook/wav2vec2-large-960h",
                'wav2vec2/base': "facebook/wav2vec2-base-960h",
                'data2vec-audio/base': 'facebook/data2vec-audio-base-960h',
                'data2vec-audio/large': 'facebook/data2vec-audio-large-960h',
                'data2vec-text/base': 'facebook/data2vec-text-base',
                'data2vec-text/large': 'facebook/data2vec-text-large',
                'hubert/base': 'facebook/hubert-base-ls960',
                'hubert/large': 'facebook/hubert-large-ls960-ft',
                'hubert/xlarge': 'facebook/hubert-xlarge-ls960-ft',
                'bert/base': 'bert-base-uncased',
                'bert/large': 'bert-large-uncased',
                't5/small': 't5-small',
                't5/base': 't5-base',
                't5/large': 't5-large',
                'gpt2/base': 'gpt2',
                'gpt2/medium': 'gpt2-medium',
                'gpt2/large': 'gpt2-large',
                'gpt2/xl': 'gpt2-xl',
                'vit-mae/base': 'facebook/vit-mae-base',
                'vit-mae/large': 'facebook/vit-mae-large',
                'vit-mae/huge': 'facebook/vit-mae-huge',
                'videomae/base': "MCG-NJU/videomae-base",
                'videomae/large': "MCG-NJU/videomae-large",
                'data2vec-vision/base': 'facebook/data2vec-vision-base',
                'data2vec-vision/large': 'facebook/data2vec-vision-large',
                'graphcodebert': 'microsoft/graphcodebert-base',
                'whisper/tiny': 'openai/whisper-tiny',
                'whisper/small': 'openai/whisper-small',
                'whisper/medium': 'openai/whisper-medium',
                'whisper/large': 'openai/whisper-large',
                's2t/small': 'facebook/s2t-small-librispeech-asr',
                's2t/medium': 'facebook/s2t-medium-librispeech-asr',
                's2t/large': 'facebook/s2t-large-librispeech-asr',
                }

static_embedding_maps = [
    'decision-transformer',
    'wav2vec2',
    'data2vec-audio',
    'data2vec-text',
    'hubert',
    'bert',
    't5',
    'gpt2',
    'whisper',
    's2t',
]

seq_embedding_maps = [
    'vit-mae',
    'videomae',
    'data2vec-vision',
    'graphcodebert'
]


class WrapVisionModel(nn.Module):
    def __init__(self, process, encoder):
        super().__init__()
        self.process = process
        self.encoder = encoder

    def forward(self, x, attention_mask=None):
        x = self.process(x, mask=attention_mask)
        return self.encoder(x)


class TransactionsModel(nn.Module):
    def __init__(self,
                 cat_embedding_projections,
                 cat_features,
                 num_embedding_projections=None,
                 num_features=None,
                 meta_embedding_projections=None,
                 meta_features=None,
                 time_embedding=None,
                 head_type='linear',
                 encoder_type='gpt2/base',
                 add_token='before',
                 num_layers=1,
                 embedding_dropout=0.0,
                 cutmix=False,
                 mixup=False,
                 emb_mult=1,
                 rel_pos_embs=False,
                 pretrained=False,
                 adapters=False,
                 hidden_size=None,
                 alpha=1.0):

        super().__init__()
        self.embedding = EmbeddingLayer(cat_embedding_projections,
                                        cat_features,
                                        num_embedding_projections,
                                        num_features,
                                        meta_embedding_projections,
                                        meta_features,
                                        time_embedding,
                                        dropout=embedding_dropout)
        inp_size = self.embedding.get_embedding_size()
        config_name = None
        self.output_size = None

        if encoder_type in config_names:
            config_name = config_names[encoder_type]
            encoder_type, encoder_size = encoder_type.split('/')

        self.add_token = add_token
        self.head_type = head_type
        self.encoder_type = encoder_type

        if pretrained and config_name is not None:
            # if adapters:
            #     model = AutoAdapterModel.from_pretrained(config_name)
            # else:
            #     model = AutoModel.from_pretrained(config_name)
            model = AutoModel.from_pretrained(config_name)
        elif config_name is not None:
            config = AutoConfig.from_pretrained(config_name)
            if config_name == 'bert-base-uncased' or config_name == 'bert-large-uncased':
                config.update_from_string('max_position_embeddings=1024')

            # if adapters:
            #     model = AutoAdapterModel.from_config(config)
            # else:
            #     model  = AutoModel.from_config(config)
            model = AutoModel.from_config(config)

        if encoder_type == 'mybert':
            self.encoder = BERT(hidden_size, heads=2 * emb_mult, num_layers=num_layers, dropout=dropout,
                                layer_norm_eps=1e-7, rel_pos_embs=rel_pos_embs)

        elif encoder_type == 'informer':
            configs = InformerConfigs(d_model=hidden_size * emb_mult,
                                      dropout=dropout,
                                      n_heads=2 * emb_mult,
                                      num_layers=num_layers)
            self.encoder = Informer(configs)
        elif encoder_type == 'gru':
            if hidden_size is None:
                hidden_size = inp_size

            self.encoder = nn.GRU(inp_size, hidden_size, num_layers=num_layers, batch_first=True)
            self.output_size = hidden_size
            hidden_size = None

        elif encoder_type == 'bert':
            self.encoder = model
            self.encoder.wpe = LambdaLayer(zero_function)

        elif encoder_type == 't5':
            self.encoder = model
            self.encoder.wpe = LambdaLayer(zero_function)

        elif encoder_type == 'gpt2':
            self.encoder = model
            self.encoder.wpe = LambdaLayer(zero_function)

        elif encoder_type == 'decision-transformer':
            self.encoder = model.encoder
            self.encoder.wpe = LambdaLayer(zero_function)

        elif encoder_type == 'wav2vec2':
            self.encoder = model.encoder
            self.encoder.pos_conv_embed = nn.Identity()

        elif encoder_type == 'data2vec-audio':
            self.encoder = model.encoder
            self.encoder.pos_conv_embed = nn.Identity()

        elif encoder_type == 'data2vec-text':
            self.encoder = model
            self.encoder.wpe = LambdaLayer(zero_function)

        elif encoder_type == 'hubert':
            self.encoder = model.hubert.encoder
            self.encoder.pos_conv_embed = nn.Identity()

        elif encoder_type in ['videomae', 'vit-base', 'data2vec-vision', 'graphcodebert', 'vit-mae']:
            self.encoder = model.encoder

        elif encoder_type == 'perceiver-vision':
            self.encoder = PerceiverModel.from_pretrained("deepmind/vision-perceiver-fourier")
        elif encoder_type in ['whisper', 's2t']:
            self.encoder = model.decoder
            self.encoder.embed_positions = LambdaLayer(zero_function)
        else:
            raise NotImplementedError("Incorrect model name")

        if hidden_size is not None:
            self.mapping_embedding = LinearMapping(inp_size, hidden_size)

        elif encoder_type in static_embedding_maps:
            hidden_size = calculate_embedding_size(model)
            self.mapping_embedding = LinearMapping(inp_size, hidden_size)

        elif encoder_type in seq_embedding_maps:
            hidden_size = calculate_embedding_size(model)
            num_latents = 16
            if encoder_type == 'data2vec-vision/large' or encoder_type == 'data2vec-vision/base':
                num_latents = 196

            elif encoder_type == 'graphcodebert':
                num_latents = 10

            self.mapping_embedding = PerceiverMapping(inp_size, hidden_size, num_latents)

        else:
            hidden_size = inp_size
            self.mapping_embedding = IdentityMapping()

        print("USING", encoder_type)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        if self.output_size is None:
            self.output_size = hidden_size

        if head_type == 'linear':
            self.head = LinearHead(self.output_size)
        elif head_type == 'rnn':
            self.head = RNNClassificationHead(self.output_size)
        elif head_type == 'mlp':
            self.head = MLPHead(self.output_size)
        elif head_type == 'transformer':
            self.head = TransformerHead(self.output_size)
        elif head_type == 'id':
            self.head = IdentityHead()
        elif head_type == 'next':
            self.head = NSPHead(self.output_size, cat_embedding_projections, num_embedding_projections)
        elif head_type == 'next_time':
            self.head = NextActionsHead(self.output_size)
        elif head_type == 'product':
            self.head = ClassificationHead(self.output_size, 5)
        else:
            raise NotImplementedError

        self.cutmix = cutmix
        self.mixup = mixup
        self.alpha = alpha
        self.encoder_type = encoder_type

    def get_embs(self, batch=None, embeds=None):
        mask = batch['mask']
        batch_size = mask.shape[0]

        if embeds is None:
            embedding = self.embedding(batch)
            embedding = self.mapping_embedding(embedding, attention_mask=mask)
        else:
            embedding = embeds

        if self.head_type != 'next' and self.head_type != 'next_time' and self.add_token == 'before':
            cls_token = self.cls_token.repeat(batch_size, 1, 1)
            embedding = torch.cat([embedding, cls_token], dim=1)

            cls_token_mask = torch.ones(batch_size, 1, dtype=bool, device=mask.device)
            mask = torch.cat([mask, cls_token_mask], dim=1)

        if self.encoder_type in ['gpt2', 'decision-transformer', 'bert', 's2t', 'whisper']:
            x = self.encoder(inputs_embeds=embedding, attention_mask=mask).last_hidden_state

        elif self.encoder_type == 't5':
            x = self.encoder(inputs_embeds=embedding, decoder_inputs_embeds=embedding,
                             attention_mask=mask).last_hidden_state

        elif 'gru' in self.encoder_type:
            x, _ = self.encoder(embedding)
        elif self.encoder_type == 'mybert':
            mask = mask.unsqueeze(1).unsqueeze(2)
            x = self.encoder(embedding, mask)

        elif self.encoder_type == 'data2vec-text':
            tmp_mask = self.encoder.get_extended_attention_mask(mask, mask.shape)
            x = self.encoder.encoder(embedding, attention_mask=tmp_mask).last_hidden_state

        elif self.encoder_type in ['wav2vec2', 'data2vec-audio', 'hubert']:
            x = self.encoder(embedding, attention_mask=mask).last_hidden_state

        elif self.encoder_type in ['vit-base', 'videomae', 'data2vec-vision', 'graphcodebert', 'vit-mae']:
            x = self.encoder(embedding).last_hidden_state
        else:
            x = self.encoder(embedding, mask)

        if self.head_type != 'next' and self.head_type != 'next_time' and self.add_token == 'after':
            cls_token = self.cls_token.repeat(batch_size, 1, 1)
            x = torch.cat([x, cls_token], dim=1)

            cls_token_mask = torch.ones(batch_size, 1, dtype=bool, device=mask.device)
            mask = torch.cat([mask, cls_token_mask], dim=1)

        return x, mask

    def forward(self, batch=None, embeds=None):
        x, mask = self.get_embs(batch, embeds)

        logit = self.head(x, mask)

        return logit


class LastPrediction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        pass

# class NextTransactionModel(nn.Module):
#     def __init__(self, 
#                  cat_embedding_projections,
#                  cat_features,
#                  num_embedding_projections,
#                  num_features,  
#                  num_layers=4,
#                  emb_mult=1,
#                  num_buckets=None,
#                  model_type='rnn',
#                 ):

#         super().__init__()
#         self.embedding = EmbeddingLayer(cat_embedding_projections, 
#                                         cat_features, 
#                                         num_embedding_projections, 
#                                         num_features)

#         inp_size = self.embedding.get_embedding_size()

#         self.model_type = model_type
#         if model_type == 'gpt':
#             configuration = GPT2Config(vocab_size=1, n_positions=750, n_embd=inp_size, n_layer=num_layers, n_head=2)
#             self.model = GPT2Model(configuration)
#         else:
#             self.model = nn.GRU(inp_size, inp_size)

#         self.head = NSPHead(inp_size, embedding_projections)

#     def forward(self, batch):

#         mask = batch['mask']
#         batch_size = mask.shape[0]

#         embedding = self.embedding(batch)

#         cls_token = self.cls_token.repeat(batch_size, 1, 1)
#         embedding = torch.cat([cls_token, embedding], dim=1)

#         first_token_mask = torch.ones(batch_size, 1, dtype=bool, device=mask.device)
#         mask = torch.cat([mask, first_token_mask], dim=1)
#         mask = mask.unsqueeze(1).unsqueeze(2)

#         if self.model_type == 'gpt':
#             out = self.model(inputs_embeds=embedding, attention_mask=mask).last_hidden_state
#         else:
#             out, _ = self.model(embedding)

#         logits = self.head(out)

#         return logits

#     def modify_numerical_head(self):
#         for i in range(3):
#             self.head.heads[-3 + i] = nn.Linear(self.head.heads[-3 + i].in_features, 1, device=self.head.heads[-3 + i].weight.device)


# class TransactionsRnn(nn.Module):
#     def __init__(self, transactions_cat_features, embedding_projections, product_col_name='product', rnn_units=128, top_classifier_units=32, mixup=False):
#         super().__init__()
#         self._transaction_cat_embeddings = nn.ModuleList([self._create_embedding_projection(*embedding_projections[feature]) 
#                                                           for feature in transactions_cat_features])

#         self._product_embedding = self._create_embedding_projection(*embedding_projections[product_col_name], padding_idx=None)

#         inp_size = sum([embedding_projections[x][1] for x in transactions_cat_features])
#         self._gru = nn.GRU(input_size=inp_size, hidden_size=rnn_units, batch_first=True, bidirectional=False)

#         self._hidden_size = rnn_units

#         self._top_classifier = nn.Linear(in_features=rnn_units+embedding_projections[product_col_name][1], 
#                                          out_features=top_classifier_units)
#         self._intermediate_activation = nn.ReLU()

#         self._head = nn.Linear(in_features=top_classifier_units, out_features=1)
#         self.mixup = mixup

#     def forward(self, batch):
#         transactions_cat_features, product_feature = batch['transactions_features'], batch['product']
#         batch_size = product_feature.shape[0]

#         if self.mixup:
#             transactions_cat_features = add_noise(transactions_cat_features)

#         embeddings = [embedding(transactions_cat_features[i]) for i, embedding in enumerate(self._transaction_cat_embeddings)]
#         concated_embeddings = torch.cat(embeddings, dim=-1)

#         _, last_hidden = self._gru(concated_embeddings)

#         last_hidden = torch.reshape(last_hidden.permute(1, 2, 0), shape=(batch_size, self._hidden_size))
#         product_embed = self._product_embedding(product_feature)

#         intermediate_concat = torch.cat([last_hidden, product_embed], dim=-1)

#         classification_hidden = self._top_classifier(intermediate_concat)
#         activation = self._intermediate_activation(classification_hidden)

#         logit = self._head(activation)

#         return logit

#     @classmethod
#     def _create_embedding_projection(cls, cardinality, embed_size, add_missing=True, padding_idx=0, emb_mult=1):
#         add_missing = 1 if add_missing else 0
#         return nn.Embedding(num_embeddings=cardinality+add_missing, embedding_dim=embed_size*emb_mult, padding_idx=padding_idx)


# 


# class MaskedModel(nn.Module):
#     def __init__(self, 
#                  transactions_cat_features, 
#                  embedding_projections, 
#                  product_col_name='product',
#                  num_layers=4,
#                  emb_mult=1,
#                 ):

#         super().__init__()
#         self.embedding = EmbeddingLayer(transactions_cat_features, embedding_projections, product_col_name, emb_mult)
#         inp_size = self.embedding.get_embedding_size()

#         configuration = BertConfig(vocab_size=1, n_positions=760, n_embd=inp_size, n_layer=num_layers, n_head=2)
#         self.model = BertModel(configuration)

#         self.head = NSPHead(inp_size, embedding_projections)

#     def forward(self, transactions_cat_features, product_feature):
#         batch_size = product_feature.shape[0]
#         mask1 = transactions_cat_features[-6] != 0

#         mask = mask1.unsqueeze(1).unsqueeze(2)
#         embedding = self.embedding(transactions_cat_features, product_feature)

#         out = self.model(inputs_embeds=embedding, attention_mask=mask)
#         logits = self.head(out.last_hidden_state)

#         return logits