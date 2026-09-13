from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_inference.generate import generate_greedy


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    new_logits = logits.float() / temperature

    if top_k is not None:
        top_k_values, top_k_indices = torch.topk(new_logits, top_k, dim=-1)
        top_k_mask = torch.full_like(new_logits, float('-inf'))
        top_k_mask.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
        new_logits = top_k_mask

    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(new_logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = (cum_probs - sorted_probs) <= top_p
        top_p_mask[..., 0] = True
        unsorted_top_p_mask = torch.zeros_like(new_logits, dtype=torch.bool)
        unsorted_top_p_mask.scatter_(dim=-1, index=sorted_indices, src=top_p_mask)

        new_logits = torch.where(unsorted_top_p_mask, new_logits, float('-inf'))

    probs = torch.softmax(new_logits, dim=-1)
    new_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return new_token
