import torch
import torch.nn as nn
from typing import Optional, Union
from ..utils import init_layers


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
            init_layers(layer)
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
            init_layers(layer)
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