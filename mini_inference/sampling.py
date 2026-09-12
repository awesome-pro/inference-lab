"""Day 2 — sampling: temperature, top-k, top-p.

Day 1's loop, with ONE step replaced: instead of always taking the highest
scoring token, draw one at random from the distribution.

    logits
      -> / temperature      reshape how peaky the distribution is
      -> top-k mask         keep only the k best candidates
      -> top-p mask         keep only the smallest set reaching probability p
      -> softmax            turn scores into probabilities
      -> multinomial        draw ONE token

The body of ``sample_next_token`` is yours to write.
See ``notes/day02_sampling.md`` for the intuition, the measured numbers, and the
design questions. The harness at the bottom checks your work.
"""

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
    """Choose one token per batch row by sampling.

    Parameters
    ----------
    logits
        Raw model output for the last position. Shape ``[B, V]``, dtype float16.
        NOT normalised -- these are scores, they can be negative and do not sum
        to 1.
    temperature
        Divides the logits BEFORE softmax. ``< 1`` sharpens (more confident),
        ``> 1`` flattens (more random). ``1.0`` is a no-op. Must be ``> 0``.
    top_k
        If given, keep only the ``k`` highest-scoring tokens per row.
    top_p
        If given, keep only the smallest set of tokens whose probabilities sum
        to at least ``p``, per row.

    Returns
    -------
    torch.Tensor
        Token ids, shape ``[B]``, dtype ``int64``.

    Shapes and dtypes to keep straight
    ----------------------------------
        logits                       [B, V]      float16  (input)
        logits.float()               [B, V]      float32  <- upcast before exp()
        probs                        [B, V]      float32  sums to 1 along dim=-1
        torch.multinomial(probs, 1)  [B, 1]      int64    <- note the extra dim!
        result                       [B]         int64    (output)

    Invariants
    ----------
    1. ``probs`` sums to 1 along the VOCAB dimension (``dim=-1``) for every row.
    2. No probability is ever negative, and no token is selected that was masked
       out.
    3. ``temperature -> 0`` must reproduce greedy decoding exactly.
    4. The returned tensor is ``[B]`` int64 -- the same contract as Day 1's
       ``argmax``, so the caller can append it unchanged.

    Measured facts (Qwen2.5-0.5B, "The capital of India is", last position)
    ---------------------------------------------------------------------
        logits range          -17.19 .. 15.32      sum = -440138.2
        after softmax          0.0 .. 0.125469     sum = 1.000000
        top-1 ' located' p=0.12547
        top-8 cumulative p=0.51072   (151,928 tokens share the other 0.48928)

        temperature 0.5 -> top1 p=0.3623, top3 sum=0.6464
        temperature 2.0 -> top1 p=0.0095, top3 sum=0.0245
        top_k=5         -> exactly 5 nonzeros, sum 1.000000
        top_p=0.5       -> keeps 8 tokens
        top_p=0.9       -> keeps 131 tokens
        top_p=0.95      -> keeps 378 tokens
    """
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



# ---------------------------------------------------------------------------
