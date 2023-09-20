import os
import re

import json
import pickle
import random
import inspect
import numpy as np
from typing import Dict, Any, Optional, List, Union

import torch
import torch.nn as nn

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def count_parameters(model, verbose: Optional[bool] = True):
    total_params = 0
    trainable_parameters = 0
    if isinstance(model, torch.nn.Parameter):
        total_params = model.numel()
        trainable_parameters = model.numel() if model.requires_grad else 0
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        s = f"Model contains: "
        # Total
        if total_params > 1_000_000:
            s += f"{round(total_params / 1_000_000, 3)} M. total parameters, "
        elif total_params > 1000:
            s += f"{round(total_params / 1000, 3)} K. total parameters, "
        else:
            s += f"{total_params} total parameters, "
        # Trainable
        if trainable_parameters > 1_000_000:
            s += f"{round(trainable_parameters / 1_000_000, 3)} M. trainable parameters."
        elif trainable_parameters > 1000:
            s += f"{round(trainable_parameters / 1000, 3)} K. trainable parameters."
        else:
            s += f"{trainable_parameters} trainable parameters."
        print(s)
    return trainable_parameters


def inspect_forward_signature(param_name: str, model: nn.Module) -> bool:
    """
    Get the list of parameter names of `forward` function of the model
    and checks whether requested parameter name is in list.
    Args:
        param_name: str, a requested parameter name;
        model: nn.Module, a model to get `forward` function from;
    Returns:
        a bool flag, whether requested parameter name is in parameter names list.
    """
    # Inspect model forward signature to keep only the arguments it accepts
    signature = inspect.signature(model.forward)
    if param_name in list(signature.parameters.keys()):
        return True
    return False


def inspect_init_signature(param_name: str, object: Any) -> bool:
    """
    Get the list of parameter names of `__init__` function of the object
    and checks whether requested parameter name is in list.
    Args:
        param_name: str, a requested parameter name;
        object: Any, an object to get `__init__` function from;
    Returns:
        a bool flag, whether requested parameter name is in parameter names list.
    """
    # Inspect model forward signature to keep only the arguments it accepts
    signature = inspect.signature(object.__init__)
    if param_name in list(signature.parameters.keys()):
        return True
    return False


def masked_mean(inp, mask, axis=1):
    down = mask.sum(axis)
    out = (inp * mask).sum(axis) / down
    return out


def zero_function(_) -> int:
    """
    Returns zero on any input.
    Args:
        _: any,
    Returns:
        int, 0.
    """
    return 0


def calculate_embedding_size(model) -> Optional[int]:
    """
    Calculates embedding size.
    Assumes, that the model ends with Layer Normalization!
    Args:
        model: nn.Module, the model to calculate output embeddings;

    Returns:

    """
    size = 0
    for module in model.modules():
        if type(module) == nn.LayerNorm:
            size = module.weight.shape[0]

    if size == 0:
        raise AttributeError(f"Provided model configuration is not supported by currect function!")

    return size

def preprocess_logits_for_metrics(logits: Any, labels: Any):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)


def postprocess_text(predictions, labels):
    import nltk

    predictions = [pred.strip() for pred in predictions]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    predictions = ["\n".join(nltk.sent_tokenize(pred)) for pred in predictions]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return predictions, labels


def compute_task_max_decoding_length(word_list, tokenizer):
    """Computes the max decoding length for the given list of words
    Args:
      tokenizer ():
      word_list: A list of stringss.
    Returns:
      maximum length after tokenization of the inputs.
    """
    max_len = 0
    for word in word_list:
        ids = tokenizer.encode(word)
        max_len = max(max_len, len(ids))
    return max_len


def transform_labels(label: str, default_value: Optional[int] = -100) -> int:
    """
    Checks whether it is possible to transform label to integer and return corresponding value (if it is possible).
    Otherwise, set it as default value (-100).
    Args:
        label: a string representation of a label;
        default_value: a default value to return (-100).
    Returns:
        integer label or default value (if label is not a digit or a number).
    """
    if label.isdigit():
        return int(label)
    return default_value


def mask_padding(input_ids: torch.Tensor,
                 pad_token_id: Optional[int] = 1,
                 mask_value: Optional[int] = -100) -> torch.Tensor:
    """
    Creates a mask for padded input tensor.
    """
    mask = torch.eq(input_ids, pad_token_id)
    return mask


def mask_lm_labels_padding(input_ids: torch.Tensor,
                           pad_token_id: Optional[int] = 1,
                           mask_value: Optional[int] = -100) -> torch.Tensor:
    """
    Creates a mask for padded input tensor.
    """
    labels = input_ids.clone().detach()
    mask = mask_padding(labels, pad_token_id)
    labels[mask.nonzero(as_tuple=True)] = mask_value
    return labels


def get_exponent_number(f: torch.Tensor) -> torch.Tensor:
    """
    Extract from an input floating point number a number, which is used to scale significand / mantissa
    by an integer exponent of a fixed base.
    Args:
        f (torch.Tensor): a floating point tensor of numbers
    Returns:
        (torch.Tensor): a tensor of exponent parts.
    """
    mask = (f != 0)
    return torch.floor(torch.log10(abs(f))).int() * mask


def get_mantissa_number(f: torch.Tensor) -> torch.Tensor:
    """
    Extract from a floating point input number an floating point significand / mantissa with a fixed precision.
    Args:
        f (torch.Tensor): a floating point tensor of numbers
    Returns:
        (torch.Tensor): a tensor of mantissa parts.
    """
    return f / 10**get_exponent_number(f).float()


def get_number_from_parts(mantissa: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """
    Construct floating point number from mantissa and exponent parts.
    Args:
        mantissa (torch.Tensor): a tensor of mantissa parts;
        exponent (torch.Tensor): a tensor of exponent parts;
    Returns:
        (torch.Tensor): a tensor of floating point tensor of numbers.
    """
    base = torch.FloatTensor([10]).to(mantissa.device)
    return base.pow(exponent) * mantissa.squeeze()


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if (_re_checkpoint.search(path) is not None) and path.endswith("ckpt")
    ]
    if len(checkpoints) == 0:
        return

    checkpoints = [os.path.join(folder, fn) for fn in checkpoints]
    checkpoints.sort(key=lambda x: os.path.getmtime(x))
    return checkpoints[0]


def get_projections_maps(num_embedding_projections_fn: str = './assets/num_embedding_projections.pkl',
                         cat_embedding_projections_fn: str = './assets/cat_embedding_projections.pkl',
                         meta_embedding_projections_fn: str = './assets/meta_embedding_projections.pkl',
                         relative_folder: Optional[str] = None) -> Dict[str, dict]:
    """
    Loading projections mappings.
    Args:
        relative_folder: a relative path for all mappings;
        num_embedding_projections_fn: a filename for mapping loading;
        cat_embedding_projections_fn:  a filename for mapping loading;
        meta_embedding_projections_fn: a filename for mapping loading;

    Returns: a Dict[str, Mapping],
        where key - is a mapping name, value - a mapping itself.

    """
    if relative_folder is not None:
        num_embedding_projections_fn = os.path.join(relative_folder, num_embedding_projections_fn)
        cat_embedding_projections_fn = os.path.join(relative_folder, cat_embedding_projections_fn)
        meta_embedding_projections_fn = os.path.join(relative_folder, meta_embedding_projections_fn)

    with open(num_embedding_projections_fn, 'rb') as f:
        num_embedding_projections = pickle.load(f)

    with open(cat_embedding_projections_fn, 'rb') as f:
        cat_embedding_projections = pickle.load(f)

    with open(meta_embedding_projections_fn, 'rb') as f:
        meta_embedding_projections = pickle.load(f)

    return {
        "num_embedding_projections": num_embedding_projections,
        "cat_embedding_projections": cat_embedding_projections,
        "meta_embedding_projections": meta_embedding_projections
    }


def get_buckets_info(feature_name: str,
                     path: Optional[str] = None) -> Optional[List[float]]:
    """
    Loads buckets info (ranges) for numeric features.
    Args:
        feature_name: a feature_name for which we need to get buckets;
        path: a filename to load, requires a pickle/json.
    Returns:
        a list of buckets ranges.
    """
    print(f"Running script in {os.path.abspath(os.getcwd())} path.")

    default_path = "../assets/dense_features_buckets.pkl"
    path = path if (path is not None) and os.path.exists(path) else default_path

    # Load buckets for numeric features
    buckets = []

    try:
        if path.endswith("pkl"):
            with open(path, 'rb') as f:
                buckets = pickle.load(f)
        elif path.endswith("json"):
            with open(path, 'rb') as f:
                buckets = json.load(f)
        else:
            print(f"Provided file name extensions: `{os.path.splitext(path)[1]}` is not supported (only json / pkl).")
            return buckets

        # Select feature
        buckets = buckets.get(feature_name) if feature_name in buckets else buckets
        if (buckets is not None) and not isinstance(buckets, dict):
            print(f"Successfully loaded buckets info for feature `{feature_name}`:\n{buckets}")
            return buckets
        else:
            print(f"Requested feature name: `{feature_name}` was not found in bucket's keys: {buckets.keys()}")
            return []
    except Exception as e:
        print(f"Error occurred during buckets loading:\n{e}")
        return buckets


def maybe_autocast(args: Union[List[torch.Tensor], torch.Tensor],
                   dtype: Optional[torch.dtype] = torch.float16):
    """
    Casts input arguments to specified dtype.
    Args:
        args: a single arg or a list of arguments to cast;
        dtype: a dtype for casting;

    Returns:
         - casted arguments, if it is possible (on GPY & floating type);
         - initial arguments otherwise.
    """
    # if on cpu, don't use autocast
    # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
    if isinstance(args, list):
        enable_autocast = all([torch.is_tensor(arg) and (arg.device != torch.device("cpu")) for arg in args])
    else:
        enable_autocast = torch.is_tensor(args) and (args.device != torch.device("cpu"))

    if enable_autocast:

        if isinstance(args, list):
            casted_args = []
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    casted_args.append(arg.to(dtype))
                else:
                    casted_args.append(arg)
        else:
            casted_args = args
            if torch.is_tensor(args) and torch.is_floating_point(args):
                casted_args = args.to(dtype)

        return casted_args

    else:
        return args


