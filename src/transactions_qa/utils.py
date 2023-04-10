import os
import re

import json
import pickle
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn

PREFIX_CHECKPOINT_DIR = "checkpoint"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"\-(\d+)$")


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


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])))


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
    # print(os.path.abspath(os.getcwd()))

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


def init_layers(module: nn.Module):
    """
    Initialize weights for Linear and RNN layers.
    """
    if isinstance(module, nn.Linear):
         nn.init.xavier_uniform_(module.weight)
    elif isinstance(module, nn.RNN) or isinstance(module, nn.GRU):
        for param in module._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(module._parameters[param])
    else:
        pass