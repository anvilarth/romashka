import torch
import torch.nn as nn
import numpy as np
import transformers

import tqdm
from typing import (Dict, List, Optional, Any, Union, Tuple)


def ce_to_ppl(ce: torch.Tensor) -> torch.Tensor:
    indexes = torch.where(ce)
    ce[indexes] = torch.exp(ce[indexes])
    ppl = ce.sum(-1) / torch.unique(indexes[0], return_counts = True)[1]
    return ppl


def evaluate_ppl(logits: torch.Tensor,
                 targets: torch.Tensor,
                 ignore_index: Optional[int] = -100,
                 reduction: Optional[str] = "none") -> float:
    """
    Calculate perplexity with logits & targets.
    """
    logits = logits.contiguous().float()
    targets = targets.contiguous().long()

    # PPL with logits
    ce = torch.nn.functional.cross_entropy(logits, targets,
                                           ignore_index = ignore_index,
                                           reduction = reduction)
    return ce_to_ppl(ce).mean().item()  # averaged


def evaluate_ppl_variants(model_outputs: Dict[str, Any],
                          true_target_idx: int,
                          input_prompt_length: Optional[int] = 0,
                          ignore_index: Optional[int] = -100,
                          reduction: Optional[str] = "none") -> Tuple[int, torch.Tensor, List[float]]:
    """
    Select answer's variant index based on PPL score.
    """
    pred_logits = model_outputs.get("logits")   # of size [n_variants, target_seq_len, vocab_size]
    labels = model_outputs.get("labels")   # of size [n_variants, target_seq_len]
    labels[labels == 0] = ignore_index  # mask paddings

    true_target = labels[true_target_idx].clone()
    true_target[true_target == 0] = ignore_index  # mask paddings
    true_target[:input_prompt_length] = ignore_index  # exclude general prompt from PPL calculation

    # PPL with logits
    ppl_per_var = []
    for i, logits in enumerate(pred_logits):
        ppl_per_var.append(float(evaluate_ppl(logits.contiguous().float(),
                                              labels[i].contiguous().long(),
                                              ignore_index=ignore_index,
                                              reduction=reduction)))

    selected_var_idx = np.argmin(ppl_per_var)
    pred_label = labels[selected_var_idx]
    return selected_var_idx, pred_label, ppl_per_var