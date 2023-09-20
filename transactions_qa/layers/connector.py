import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Any

from transformers import PretrainedConfig
from romashka.transactions_qa.layers.initialization import (init_xavier_uniform_layers,
                                                            init_linear)
from romashka.transactions_qa.layers.layers import TransformerEncoderLayer
from romashka.transactions_qa.layers.qformer_connector_hf import QFormerConnector
from romashka.transactions_qa.layers.instruct_qformer_connector_hf import InstructQFormerConnector


CONNECTOR_TYPES = [
    "linear",
    "complex_linear",
    "recurrent",
    "transformer",
    "qformer",
    "instruct_qformer"
]


def make_linear_connector(output_size: Optional[int] = None,
                          input_size: Optional[int] = None,
                          embedding_model: Optional[nn.Module] = None,
                          autoregressive_model: Optional[nn.Module] = None,
                          device: Optional[Union[torch.device, str]] = 'cpu'):
    required_output_size = None
    if output_size is not None:
        required_output_size = output_size
    elif embedding_model is not None:
        try:
            # As it is custom model
            required_output_size = embedding_model.head.output_size
        except Exception as e0:
            try:
                # If it is HF model, then  take output dimensions from config
                required_output_size = embedding_model.config.d_model
            except Exception as e1:
                raise AttributeError(f"Cannot get `output_size` from embeddings model:\n{e0}\n{e1}")
    else:
        raise AttributeError(f"Unable to define `output_size` from embeddings model"
                             "as none of `output_size` or `embedding_model` is specified.")

    required_input_size = None
    if input_size is not None:
        required_input_size = input_size
    elif autoregressive_model is not None:
        try:
            # If it is HF model, then take inputs dimensions from config
            required_input_size = autoregressive_model.config.d_model
        except Exception as e:
            raise AttributeError(f"Cannot get `input_size` from autoregressive model:\n{e}")
    else:
        raise AttributeError(f"Unable to define `input_size` from autoregressive model"
                             "as none of `input_size` or `autoregressive_model` is specified.")

    print(f"Output dimension of embedding model: {required_output_size}")
    print(f"Input dimension of autoregressive model: {required_input_size}")
    print(f"Creating linear connector from {required_output_size} to {required_input_size} "
          f"and move to device: {device}.")

    return LinearConnector(
        output_size=required_output_size,
        input_size=required_input_size
    )


class LinearConnector(nn.Module):
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        super().__init__()
        self.output_size = output_size  # output size of embedder model
        self.input_size = input_size  # input size of LM model
        self.device = device
        self.layer = self._create_layer()

    def _create_layer(self) -> Optional[nn.Module]:
        try:
            layer = nn.Linear(self.output_size, self.input_size).to(self.device)
            init_linear(layer)
            return layer
        except Exception as e:
            print(f"Error occurred during connector creation:\n{e}")
            raise AttributeError(f"Error occurred during connector creation:\n{e}")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.layer(x)


def make_recurrent_connector(layer_type: str,
                             num_recurrent_layers: Optional[int] = 2,
                             is_bidirectional: Optional[bool] = False,
                             output_size: Optional[int] = None,
                             input_size: Optional[int] = None,
                             embedding_model: Optional[nn.Module] = None,
                             autoregressive_model: Optional[nn.Module] = None,
                             device: Optional[Union[torch.device, str]] = 'cpu'):
    required_output_size = None
    if output_size is not None:
        required_output_size = output_size
    elif embedding_model is not None:
        try:
            # As it is custom model
            required_output_size = embedding_model.head.output_size
        except Exception as e0:
            try:
                # If it is HF model, then  take output dimensions from config
                required_output_size = embedding_model.config.d_model
            except Exception as e1:
                raise AttributeError(f"Cannot get `output_size` from embeddings model:\n{e0}\n{e1}")
    else:
        raise AttributeError(f"Unable to define `output_size` from embeddings model"
                             "as none of `output_size` or `embedding_model` is specified.")

    required_input_size = None
    if input_size is not None:
        required_input_size = input_size
    elif autoregressive_model is not None:
        try:
            # If it is HF model, then take inputs dimensions from config
            required_input_size = autoregressive_model.config.d_model
        except Exception as e:
            raise AttributeError(f"Cannot get `input_size` from autoregressive model:\n{e}")
    else:
        raise AttributeError(f"Unable to define `input_size` from autoregressive model"
                             "as none of `input_size` or `autoregressive_model` is specified.")

    print(f"Output dimension of embedding model: {required_output_size}")
    print(f"Input dimension of autoregressive model: {required_input_size}")
    print(f"Creating linear connector from {required_output_size} to {required_input_size} "
          f"and move to device: {device}.")

    try:
        layer = RecurrentConnector(
            layer_type=layer_type,
            num_recurrent_layers=num_recurrent_layers,
            is_bidirectional=is_bidirectional,
            output_size=required_output_size,
            input_size=required_input_size,
            device=device
        )
        return layer
    except Exception as e:
        print(f"error occurred during connector creation:\n{e}")
        return None


class RecurrentConnector(nn.Module):
    def __init__(self,
                 layer_type: str,
                 output_size: int,
                 input_size: int,
                 num_recurrent_layers: Optional[int] = 2,
                 is_bidirectional: Optional[bool] = False,
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        super().__init__()

        self._hidden_state = None  # encoder hidden states
        self.layer_type = layer_type
        self.num_recurrent_layers = num_recurrent_layers
        self.is_bidirectional = is_bidirectional
        self.output_size = output_size  # output size of embedder model
        self.input_size = input_size  # input size of LM model
        self.device = device
        self.layer = self._create_layer()

    def _create_layer(self) -> Optional[nn.Module]:
        try:
            layer = getattr(torch.nn, self.layer_type)(
                input_size=self.output_size,
                hidden_size=self.input_size,
                num_layers=self.num_recurrent_layers,
                bidirectional=self.is_bidirectional,
                batch_first=True,
            ).to(self.device)
            init_xavier_uniform_layers(layer)
            return layer
        except Exception as e:
            print(f"Error occurred during connector creation:\n{e}")
            raise AttributeError(f"Error occurred during connector creation:\n{e}")

    def forward(self, x: torch.Tensor,
                x_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:

        if x_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, x_lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.layer(x)

        if x_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        # get only the last layer output for the last step
        return torch.unsqueeze(outputs[:, -1, :], 1)


def make_complex_linear_connector(output_size: Optional[int] = None,
                                  input_size: Optional[int] = None,
                                  n_layers: Optional[int] = 2,
                                  hidden_dims: Optional[List[int]] = None,
                                  add_normalizations: Optional[List[bool]] = None,
                                  add_activations: Optional[List[bool]] = None,
                                  device: Optional[Union[torch.device, str]] = 'cpu'):
    """
    Creates more complex, but still linear connector by stacking with non-linearity few linear layers.
    Args:
        output_size: an output size of an embeddings model (i.e. input size for the first connector layer);
        input_size: an input size of an autoregressive model (i.e. output size for the last connector layer);
        n_layers: a number of inner layers in connector;
        hidden_dims: a list of dimensions for each inner layer;
        add_normalizations: a list of bool flags indicated whether to add or not LayerNorm after each layer;
        add_activations: a list of bool flags indicated whether to add or not activation after each layer;
        device: a device to allocate model.

    Returns:
        a connector.
    """
    # Check parameters consistency
    if (hidden_dims is None) or (len(hidden_dims) != (n_layers - 1)):
        raise AttributeError(f"Number of hidden dims (= {len(hidden_dims) if hidden_dims is not None else 0}) "
                             f"does not equal to number of layers - 1 (= {n_layers - 1})!")

    if add_normalizations is None:
        add_normalizations = [False] * n_layers
    elif len(add_normalizations) != n_layers:
        raise AttributeError(f"Number of normalization flags (= {len(add_normalizations)}) "
                             f"does not equal to number of layers (= {n_layers})!")

    if add_activations is None:
        add_activations = [False] * n_layers
    elif len(add_activations) != n_layers:
        raise AttributeError(f"Number of adding activations flags (= {len(add_activations)}) "
                             f"does not equal to number of layers (= {n_layers})!")

    print(f"Output dimension of embedding model: {output_size}")
    print(f"Input dimension of autoregressive model: {input_size}")
    print(f"Creating connector from {output_size} to {input_size} "
          f"and move to device: {device}.")

    return ComplexLinearConnector(
        output_size=output_size,
        input_size=input_size,
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        add_normalizations=add_normalizations,
        add_activations=add_activations,
        device=device
    )


class ComplexLinearConnector(nn.Module):
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 n_layers: Optional[int] = 2,
                 hidden_dims: Optional[List[int]] = None,
                 add_normalizations: Optional[List[bool]] = None,
                 add_activations: Optional[List[bool]] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        super().__init__()
        self.output_size = output_size  # output size of embedder model
        self.input_size = input_size  # input size of second model / -> output shape for last linear layer
        self.n_layers = n_layers
        self.hidden_dims = hidden_dims if hidden_dims is not None else [input_size] * (self.n_layers - 1)
        self.add_normalizations = add_normalizations if add_normalizations is not None else [False] * self.n_layers
        self.add_activations = add_activations if add_activations is not None else [False] * self.n_layers
        self.device = device
        self._create_layers()

    def _create_layers(self):
        layers = []
        try:
            input_dim = self.output_size
            output_dim = None
            final_output_dim = self.input_size

            for layer_n in range(self.n_layers):
                if layer_n >= (self.n_layers - 1):
                    output_dim = final_output_dim
                else:
                    output_dim = self.hidden_dims[layer_n]
                linear = nn.Linear(input_dim, output_dim)
                init_linear(linear)
                layers.append(linear)
                if self.add_normalizations[layer_n]:
                    layers.append(nn.LayerNorm(output_dim))
                if self.add_activations[layer_n]:
                    layers.append(nn.ELU())
                input_dim = output_dim

            self.layers = nn.Sequential(*layers)
            self.layers.to(self.device)
        except Exception as e:
            print(f"Error occurred during complex connector creation:\n{e}")
            raise AttributeError(f"Error occurred during complex connector creation:\n{e}")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.layers(x)


def make_transformer_connector(output_size: int,
                               input_size: int,
                               n_layers: Optional[int] = 1,
                               n_heads: Optional[List[int]] = None,
                               ff_output_dims: Optional[List[int]] = None,
                               forward_expansions: Optional[List[int]] = None,
                               add_rel_pos_embeddings: Optional[List[bool]] = None,
                               dropouts_p: Optional[List[float]] = None,
                               device: Optional[Union[torch.device, str]] = 'cpu'):
    """
    Creates a connector based on simple Transformer encoder blocks.
    Args:
        output_size: an output size of an embeddings model (i.e. input size for the first connector layer);
        input_size: an input size of an autoregressive model (i.e. output size for the last connector layer);
        n_layers: a number of Transformer layers;
        n_heads:  a list contains numbers of heads for each Transformer layer;
        ff_output_dims: a list of fast-forward dims (exact sizes) for each Transformer layer;
        forward_expansions: a list of multipliers for fast-forward dims calculation for each Transformer layer.
            Can be overwritten by `ff_output_dims` argument;
        add_rel_pos_embeddings: a list of bool flags whether
            to add relative position embeddings for each Transformer layer or not;
        dropouts_p: a list of dropout probabilities for each Transformer layer;
        device: a device to allocate model.
    Returns:
        a Transformer connector.
    """
    # Check parameters consistency
    print(f"Output dimension of embedding model: {output_size}")
    print(f"Input dimension of autoregressive model: {input_size}")
    print(f"Creating connector from {output_size} to {input_size} "
          f"and move to device: {device}.")

    # Check parameters consistency
    if (n_heads is None) or (len(n_heads) != n_layers):
        raise AttributeError(f"Number of heads (= {len(n_heads) if n_heads is not None else 0}) "
                             f"does not equal to number of layers  (= {n_layers})!")

    return TransformerConnector(
        output_size=output_size,
        input_size=input_size,
        n_layers=n_layers,
        n_heads=n_heads,
        ff_output_dims=ff_output_dims,
        forward_expansions=forward_expansions,
        add_rel_pos_embeddings=add_rel_pos_embeddings,
        dropouts_p=dropouts_p,
        device=device
    )


class TransformerConnector(nn.Module):
    def __init__(self,
                 output_size: int,
                 input_size: int,
                 n_layers: Optional[int] = 1,
                 n_heads: Optional[List[int]] = None,
                 ff_output_dims: Optional[List[int]] = None,
                 forward_expansions: Optional[List[int]] = None,
                 add_rel_pos_embeddings: Optional[List[bool]] = None,
                 dropouts_p: Optional[List[float]] = None,
                 device: Optional[Union[torch.device, str]] = 'cpu'):

        super().__init__()
        self.output_size = output_size  # output size of embedder model
        self.input_size = input_size  # input size of second model / -> output shape for last linear layer
        self.n_layers = n_layers  # a number of Transformer layers

        self.n_heads = n_heads if n_heads is not None else [8] * self.n_layers
        self.ff_output_dims = ff_output_dims if ff_output_dims is not None else None
        self.forward_expansions = forward_expansions if forward_expansions is not None else [2] * self.n_layers

        self.add_rel_pos_embeddings = add_rel_pos_embeddings if add_rel_pos_embeddings is not None else [
                                                                                                            False] * self.n_layers
        self.dropouts_p = dropouts_p if dropouts_p is not None else [0.1] * self.n_layers
        self.device = device
        self._create_layers()

    def _create_layers(self):
        layers = []
        try:
            for layer_n in range(self.n_layers):
                ff_output_dim = self.ff_output_dims[layer_n] if self.ff_output_dims is not None else None
                n_heads = self.n_heads[layer_n]
                forward_expansion = self.forward_expansions[layer_n]
                dropout_p = self.dropouts_p[layer_n]
                layer = TransformerEncoderLayer(
                    embedding_dim=self.output_size,
                    heads=n_heads,
                    ff_output_dim=ff_output_dim,
                    forward_expansion=forward_expansion,
                    dropout=dropout_p
                )
                layers.append(layer)

            self.layers = nn.ModuleList(layers)
            self.layers.to(self.device)

            self.lm_projection_layer = torch.nn.Linear(self.output_size,
                                                       self.input_size).to(self.device)

        except Exception as e:
            print(f"Error occurred during complex connector creation:\n{e}")
            raise AttributeError(f"Error occurred during complex connector creation:\n{e}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_projection_layer(x)


def make_qformer_connector(output_size: int,
                           input_size: int,
                           vocab_size: Optional[int] = None,
                           pad_token_id: Optional[int] = None,
                           config: Optional[Union[PretrainedConfig, Dict[str, Any]]] = None,
                           num_queries: Optional[int] = 32,
                           from_hf: Optional[bool] = True,
                           from_checkpoint: Optional[bool] = False,
                           device: Optional[Union[torch.device, str]] = 'cpu'):
    """
    Creates a connector based on Querying Transformer (Q-Former), used in BLIP-2.
    Args:
        output_size: an output size of an embeddings model (i.e. input size for the first connector layer);
        input_size: an input size of an autoregressive model (i.e. output size for the last connector layer);
        vocab_size: a vocabulary size of LM (need to be passed AFTER extending it with special/additional tokens!);
        pad_token_id: a pad_token_id from LM tokenizer;
        config: a pretrained config for Q-Former model;
        num_queries: a number of queries for cross attention with embeddings
                    (equals to output sequence length of transactions history);
        device: a device to allocate model.

    Returns:
        a connector.
    """
    # Check parameters consistency
    print(f"Output dimension of embedding model: {output_size}")
    print(f"Input dimension of autoregressive model: {input_size}")
    print(f"Creating connector from {output_size} to {input_size} "
          f"and move to device: {device}.")
    if from_hf:
        return QFormerConnector(
            output_size=output_size,
            input_size=input_size,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            num_queries=num_queries,
            config=config,
            from_checkpoint=from_checkpoint,
            device=device
        )
    else:
        raise AttributeError(f"Currently others implementations of Q-Former are not supported!")


def make_instruct_qformer_connector(output_size: int,
                                   input_size: int,
                                   vocab_size: Optional[int],
                                   pad_token_id: Optional[int],
                                   config: Optional[Union[PretrainedConfig, Dict[str, Any]]] = None,
                                   num_queries: Optional[int] = 32,
                                   from_checkpoint: Optional[bool] = False,
                                   device: Optional[Union[torch.device, str]] = 'cpu'):
    """
    Creates a connector based on Querying Transformer (Q-Former) with instruction-aware cross-attention,
    used in InstructBLIP.
    Args:
        output_size: an output size of an embeddings model (i.e. input size for the first connector layer);
        input_size: an input size of an autoregressive model (i.e. output size for the last connector layer);
        vocab_size: a vocabulary size of LM (need to be passed AFTER extending it with special/additional tokens!);
        pad_token_id: a pad_token_id from LM tokenizer;
        config: a pretrained config for Q-Former model;
        num_queries: a number of queries for cross attention with embeddings
                    (equals to output sequence length of transactions history);
        device: a device to allocate model.

    Returns:
        a connector.
    """
    # Check parameters consistency
    print(f"Output dimension of embedding model: {output_size}")
    print(f"Input dimension of autoregressive model: {input_size}")
    print(f"Creating connector from {output_size} to {input_size} "
          f"and move to device: {device}.")
    return InstructQFormerConnector(
            output_size=output_size,
            input_size=input_size,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            num_queries=num_queries,
            config=config,
            from_checkpoint=from_checkpoint,
            device=device
        )

