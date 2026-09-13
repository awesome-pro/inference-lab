"""Day 3 — prefill/decode split with a KV cache.

The whole change is two model calls that differ in two arguments:

    # PREFILL -- once
    out    = model(input_ids, attention_mask=mask, use_cache=True)
    logits = out.logits[:, -1, :]          # [B, V]
    cache  = out.past_key_values           # holds T positions

    # DECODE -- each step
    out    = model(input_ids[:, -1:], attention_mask=mask,
                   past_key_values=cache, use_cache=True)
    logits = out.logits[:, -1, :]          # [B, V]
    cache  = out.past_key_values           # holds T+1, T+2, ...

    prefill: input_ids [B, T]   no past_key_values
    decode:  input_ids [B, 1]   past_key_values=cache

Write those into ``prefill`` and ``decode_step``. Then in the loop, swap
``decode_step_reference`` for ``decode_step`` and pass ``input_ids[:, -1:]``.

Everything else here is Day 1 plumbing: masks, the EOS flag, padding, the
append. Unchanged.

Test with::

    uv run mini_inference/cached_generate_harness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from mini_inference.sampling import sample_next_token


def prefill(
    input_ids: torch.Tensor,
    model,
    attention_mask: torch.Tensor | None = None,
):
    """Process the whole prompt once and return the cache it produced.

    Parameters
    ----------
    input_ids   [B, T] int64, on the model's device.
    attention_mask  [B, T], 1 = real, 0 = pad. May be None when unpadded.

    Returns
    -------
    logits          [B, V]  -- last position only.
    past_key_values         -- holds T positions.

    TODO(you): 3 lines.

        out = model(input_ids, attention_mask=attention_mask, use_cache=True)
        return out.logits[:, -1, :], out.past_key_values

    The fallback below returns no cache, so the harness still runs. Delete it.
    """
    out = model(input_ids, attention_mask=attention_mask, use_cache=True)
    cache = out.past_key_values
    return out.logits[:, -1, :], cache


def decode_step(
    input_ids: torch.Tensor,
    past_key_values,
    attention_mask: torch.Tensor,
    model,
):
    """Feed ONE token, read the cache, extend the cache.

    Parameters
    ----------
    input_ids       [B, 1] int64 -- only the new token.
    past_key_values the cache, holding n positions.
    attention_mask  [B, n+1] -- INCLUDING the new token.

    Note the lengths disagree on purpose: ``input_ids`` is "what is new", the
    mask is "what exists". The cache holds the rest. Passing a length-1 mask
    instead raises nothing and silently produces garbage.

    Returns
    -------
    logits          [B, V]
    past_key_values -- the same object, now n+1 positions.

    TODO(you): 3 lines.

        out = model(input_ids, attention_mask=attention_mask,
                    past_key_values=past_key_values, use_cache=True)
        return out.logits[:, -1, :], out.past_key_values

    Why this is the whole point: without a cache, scoring one new token needs
    every token before it, so you must feed ``[B, n+1]``. The cache lets you
    feed ``[B, 1]``. That is the entire saving.
    """
    out = model(input_ids[:, -1:], attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def decode_step_reference(
    input_ids: torch.Tensor,
    past_key_values,
    attention_mask: torch.Tensor,
    model,
):
    """Same decode step, done slowly: re-score the whole prefix.

    Not the cache. Exists so ``generate_cached`` runs before ``decode_step`` is
    written. Reproduces Day 1's cost exactly, so it is also what you measure
    the cache against.
    """
    out = model(input_ids, attention_mask=attention_mask, use_cache=False)
    return out.logits[:, -1, :], past_key_values


def generate_cached(
    input_ids: torch.Tensor,
    model,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    debug_argmax: bool = True,
) -> torch.Tensor:
    """Generate with a KV cache. Same output contract as ``generate_greedy``.

    Returns ``[B, T + n]`` int64. EOS is dropped and rows that finish early are
    padded with ``pad_token_id``, exactly as in Day 1 -- that equivalence is the
    correctness check.

    Invariant worth asserting while developing::

        cache.get_seq_length() == attention_mask.shape[1]

    Both mean "how many tokens exist". If they drift, one missed an update.

    ``debug_argmax`` upcasts before the argmax when comparing against
    ``generate_greedy``. It removes fp16 tie-flips, which can otherwise cascade
    into a different continuation. It is a test aid, not part of the algorithm.
    """
    device = input_ids.device
    batch_size, prompt_len = input_ids.shape

    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)

    # Only needed when padding is actually present.
    needs_mask = pad_token_id is not None and bool((input_ids == pad_token_id).any())
    if needs_mask:
        attention_mask = (input_ids != pad_token_id).long()
    else:
        attention_mask = torch.ones_like(input_ids)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        # ---- PREFILL: one call, all T prompt tokens ----
        logits, past_key_values = prefill(
            input_ids, model, attention_mask if needs_mask else None
        )
        cache_len = 0 if past_key_values is None else past_key_values.get_seq_length()
        print(f"  prefill: prompt T={prompt_len}, cache holds {cache_len}")

        for step in range(max_new_tokens):
            # ---- SELECT: same as Day 1/2 ----
            if temperature <= 0.0:
                pick = logits.float() if debug_argmax else logits
                next_token = pick.argmax(dim=-1)                 # [B] int64
            else:
                next_token = sample_next_token(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p
                )                                                # [B] int64

            # ---- EOS: per-row flag, never `.any()` ----
            stop_now = torch.zeros_like(finished)
            if eos_token_id is not None:
                stop_now = (next_token == eos_token_id) & ~finished
                finished = finished | stop_now

            if pad_token_id is not None:
                already_done = finished & ~stop_now
                next_token = torch.where(
                    already_done, torch.full_like(next_token, pad_token_id), next_token
                )

            # ---- APPEND: input_ids and mask grow together ----
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

            new_col = (~already_done).long().unsqueeze(-1) if pad_token_id is not None \
                else torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, new_col], dim=-1)

            # ---- DECODE ----
            # Skeleton state: decode_step_reference re-scores the whole prefix,
            # so this is correct but costs what Day 1 cost. Swapping in
            # decode_step AND slicing to input_ids[:, -1:] is the whole change.
            if step == max_new_tokens - 1 or bool(finished.all()):
                break

            logits, past_key_values = decode_step(
                input_ids, past_key_values, attention_mask, model
            )

            if past_key_values is not None:
                assert past_key_values.get_seq_length() == attention_mask.shape[1], (
                    f"cache/mask desync at step {step}: "
                    f"cache={past_key_values.get_seq_length()} "
                    f"mask={attention_mask.shape[1]}"
                )

            if bool(stop_now.all()):
                input_ids = input_ids[:, :-1]
                attention_mask = attention_mask[:, :-1]
                break

    return input_ids


__all__ = ["prefill", "decode_step", "decode_step_reference", "generate_cached"]
