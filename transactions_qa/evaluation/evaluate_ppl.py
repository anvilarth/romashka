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


def evaluate_ppl_variants(input_ids: Union[torch.Tensor, List[torch.Tensor]],
                          model: nn.Module,
                          input_prompt_length: Optional[int] = -1,
                          ignore_index: Optional[int] = -100,
                          reduction: Optional[str] = "none") -> Tuple[int, List[float]]:
    """
    Select answer's variant index based on PPL score.
    """
    # PPL with logits
    ppl_per_var = []
    for i, inputs in enumerate(input_ids):
        inputs = inputs.to(model.device)
        targets = inputs.clone()
        targets[:input_prompt_length] = -100.
        targets = targets.contiguous().long()

        with torch.no_grad():
            outputs = model(inputs, labels = targets)
            logits = outputs.logits.contiguous().float()

        ppl_per_var.append(float(evaluate_ppl(logits, targets,
                                              ignore_index = ignore_index,
                                              reduction = reduction)))

    selected_var_idx = np.argmin(ppl_per_var)
    return selected_var_idx, ppl_per_var