"""Token-level log-probability utilities for DPO / GRPO."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


def sequence_logprob(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    length_norm: bool = True,
) -> torch.Tensor:
    """Log p(labels | prompt) summed over label tokens (labels=-100 masked)."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    targets = labels[:, 1:]
    mask = targets != -100
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    token_lp = token_lp * mask
    total = token_lp.sum(dim=-1)
    if length_norm:
        denom = mask.sum(dim=-1).clamp(min=1)
        total = total / denom
    return total
