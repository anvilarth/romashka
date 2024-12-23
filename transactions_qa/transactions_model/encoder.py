import sys
import torch
import torch.nn as nn

from transformers import AutoModel
from transformers import PerceiverModel, AutoConfig

from romashka.transactions_qa.layers.layers import LambdaLayer
from romashka.transactions_qa.utils import calculate_embedding_size, zero_function


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


class TransactionEncoder(nn.Module):
    def __init__(self, encoder_type, inp_size, hidden_size=None, pretrained=False, config_name=None, num_layers=1):
        super().__init__()

        if encoder_type in config_names:
            config_name = config_names[encoder_type]
            encoder_type, encoder_size = encoder_type.split('/')

        self.encoder_type = encoder_type

        if pretrained and config_name is not None:
                model = AutoModel.from_pretrained(config_name)
        elif config_name is not None:
            config = AutoConfig.from_pretrained(config_name)
            if config_name == 'bert-base-uncased' or config_name == 'bert-large-uncased':
                config.update_from_string('max_position_embeddings=1024')

            model = AutoModel.from_config(config)

        if encoder_type == 'gru':
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
            self.connector_type = 'linear'

        elif encoder_type in static_embedding_maps:
            hidden_size = calculate_embedding_size(model)
            self.connector_type = 'linear'

        elif encoder_type in seq_embedding_maps:
            hidden_size = calculate_embedding_size(model)
            self.connector_type = 'perceiver'
        else:
            hidden_size = inp_size
            self.connector_type = 'id'

        self.output_size = hidden_size

        print("USING", encoder_type)

    def forward(self, embedding, mask):
        batch_size = mask.shape[0]
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

        return x, mask

    def get_output_size(self):
        return self.output_size

    def get_connector_type(self):
        return self.connector_type