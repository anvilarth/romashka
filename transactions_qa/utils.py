import os
import re
import pickle
from typing import Dict, Any, Optional

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