import torch
import torch.nn as nn
from typing import Optional, Union, List
from romashka.transactions_qa.layers.initialization import (init_xavier_uniform_layers,
                                                            init_linear)


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
        layer = ReccurrentConnector(
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


class ReccurrentConnector(nn.Module):
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
                                  device: Optional[Union[torch.device, str]] = 'cpu'):
    """
    Creates more complex, but still linear connector by stacking with non-linearity few linear layers.

    Args:
        output_size:
        input_size:
        device:

    Returns:

    """
    print(f"Output dimension of embedding model: {required_output_size}")
    print(f"Input dimension of autoregressive model: {required_input_size}")
    print(f"Creating linear connector from {required_output_size} to {required_input_size} "
          f"and move to device: {device}.")

    return LinearConnector(
        output_size=required_output_size,
        input_size=required_input_size
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
        self.layers = nn.Sequential()
        self._create_layers()

    def _create_layers(self):
        try:
            input_dim = self.output_size
            output_dim = None
            final_output_dim = self.input_size

            for layer_n in range(self.n_layers):
                if layer_n >= (self.n_layers - 1):
                    output_dim = final_output_dim
                else:
                    output_dim = self.hidden_dims[layer_n]
                self.layers.append(init_layers(nn.Linear(input_dim, output_dim)))
                if self.add_normalizations[layer_n]:
                    self.layers.append(nn.LayerNorm(output_dim))
                if self.add_activations[layer_n]:
                    self.layers.append(nn.ELU())
                input_dim = output_dim

            self.layers.to(self.device)
        except Exception as e:
            print(f"Error occurred during complex connector creation:\n{e}")
            raise AttributeError(f"Error occurred during complex connector creation:\n{e}")

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.layers(x)