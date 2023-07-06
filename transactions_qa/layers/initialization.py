import torch
import numpy as np


def init_xavier_uniform_layers(module: torch.nn.Module):
    """
    Initialize weights for Linear and RNN layers.
    """
    if isinstance(module, torch.nn.Linear):
         torch.nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, torch.nn.RNN) or isinstance(module, torch.nn.GRU):
        for param in module._flat_weights_names:
            if "weight" in param:
                torch.nn.init.xavier_uniform_(module._parameters[param])
    elif isinstance(module, torch.nn.Parameter):
        torch.nn.init.xavier_uniform_(module)
    else:
        pass


def init_linear(module: torch.nn.Module):
    """
    Initialize linear transformation.
    Fills the input Tensor with values according to the method described in:
    Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010),
    using a uniform distribution.
    """
    bias = np.sqrt(6.0 / (module.weight.size(0) + module.weight.size(1)))
    torch.nn.init.uniform(module.weight, -bias, bias)
    if module.bias is not None:
        module.bias.data.zero_()


def init_embeddings(module: torch.nn.Module):
    """
    Initialize embeddings layer.
    Fills the input Tensor with values drawn from the normal distribution (mu = 0, std = 0).
    """
    if isinstance(module, torch.nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=1.0)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def init_embeddings_with_tensor(module: torch.nn.Module, embedding_tensor: torch.Tensor):
    """
    Initialize embeddings layer.
    Fills the input Tensor with values from input LLM embedding space.
    """
    if len(embedding_tensor.size()) > 1:
        raise AttributeError(f"A tensor for embeddings initialization is multidimensional: {embedding_tensor.size()}")
    if isinstance(module, torch.nn.Embedding):
        module.weight.data = embedding_tensor.unsqueeze(0).repeat(module.weight.size(0), 1)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()


def init_layernorm(module: torch.nn.Module):
    """
    Initialize Layer Normalization layer.
    Fills the input Tensor with values drawn from the normal distribution (mu = 0, std = 0).
    """
    if isinstance(module, torch.nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


def init_parameter_with_tensor(module: torch.nn.Parameter, init_tensor: torch.Tensor):
    """
    Initialize nn.Parameters.
    Fills the input Tensor with values from input LLM embedding space.
    """
    if (len(init_tensor.size()) > 1) and (init_tensor.size(0) != module.size(0)):
        raise AttributeError(f"A tensor for parameter initialization is multi-dimensional: {init_tensor.size()}"
                             f" and 0 dimension is not equals to parameter 0 dimension.")
    if isinstance(module, torch.nn.Parameter):
        if init_tensor.size(0) == module.size(0):
            module.data = init_tensor
        else:
            module.data = init_tensor.unsqueeze(0).repeat(module.size(0), 1)