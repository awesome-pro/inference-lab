"""Day 1 — naive greedy autoregressive decoding.

    prompt -> tokenize -> input_ids -> model -> logits
           -> logits at the final position -> argmax -> next_token_id
           -> append to input_ids -> repeat -> stop

No KV cache. Every iteration pushes the ENTIRE sequence built so far through
all 24 layers, then appends one token. Deliberately the slow, obviously-correct
reference implementation; Day 3's cached decoder is checked against it.

The body of ``generate_greedy`` is yours to write.
See ``notes/day01_naive_greedy.md``.
"""

from __future__ import annotations

import torch


def generate_greedy(
    input_ids: torch.Tensor,
    model,
    max_new_tokens: int,
    eos_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> torch.Tensor:
    """Extend ``input_ids`` greedily, one token per step.

    Parameters
    ----------
    input_ids       [B, T] int64, on the model's device.
    model           a causal LM in ``eval()`` mode.
    max_new_tokens  upper bound on appended tokens.
    eos_token_id    if given, stop a row once this token is appended.
    pad_token_id    filler for rows that already finished, to keep the batch
                    rectangular. Falls back to ``model.config.pad_token_id``.
                    Qwen2.5-0.5B sets that to None.

    Returns
    -------
    [B, T + n] int64, ``n <= max_new_tokens``. Columns ``[T:]`` are new. A row
    that finished early is padded up to the batch's final length.

    Shapes
    ------
        input_ids              [B, T]      int64
        model(...).logits      [B, T, V]   float16
        logits[:, -1, :]       [B, V]
        argmax(dim=-1)         [B]         int64
        unsqueeze(-1)          [B, 1]
        cat(dim=-1)            [B, T+1]

    Two things that fail silently
    ----------------------------
    1. The last REAL position depends on the padding side (T = column count)::

           LEFT-padded   [pad pad A B C]   last real index = T-1 = -1  (all rows)
           RIGHT-padded  [A B C pad pad]   last real index = mask.sum(1)-1
           no padding                      last real index = T-1 = -1  (all rows)

       ``mask.sum(1)-1`` on a left-padded batch returns an ORDINAL, not an
       index, so it reads an early token. ``-1`` on a right-padded batch reads a
       pad position. Neither raises.

    2. The mask must grow with ``input_ids``. A stale mask means the model
       attends to padding.
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]

    # Per-row "done" flag. With B>1 the loop cannot break early: row 0 may
    # finish while row 1 keeps going. The flag lets rows stop independently --
    # the bug it prevents is `(next == eos).any()`, which kills every row as
    # soon as one finishes.
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)

    needs_mask = pad_token_id is not None and bool((input_ids == pad_token_id).any())
    if needs_mask:
        attention_mask = (input_ids != pad_token_id).long()
        # Infer from the data rather than trusting a flag.
        padding_is_left = bool(attention_mask[0, 0].item() == 0)
    else:
        attention_mask = None
        padding_is_left = False

    with torch.no_grad():
        for step in range(max_new_tokens):
            if attention_mask is not None:
                logits = model(input_ids, attention_mask=attention_mask).logits
            else:
                logits = model(input_ids).logits

            # Recompute every step: input_ids gains a column each iteration, so
            # an index captured on step 0 goes stale. Symptom: the model reads a
            # distribution it already generated from, and output degenerates to
            # one repeated token.
            T_cur = input_ids.shape[1]
            if needs_mask and not padding_is_left:
                last_index = attention_mask.sum(dim=-1) - 1
            else:
                last_index = torch.full(
                    (batch_size,), T_cur - 1, device=device, dtype=torch.long
                )

            # Pairwise advanced indexing picks logits[0, last_index[0], :] etc.
            # Result [B, V]. Passing [B, 1] index tensors would give [B, 1, V]
            # and break the later cat.
            rows = torch.arange(batch_size, device=device)
            last = logits[rows, last_index]                    # [B, V]
            next_token = last.argmax(dim=-1)                   # [B] int64

            stop_now = torch.zeros_like(finished)
            if eos_token_id is not None:
                stop_now = (next_token == eos_token_id) & ~finished
                finished = finished | stop_now

            # A row that just stopped contributes no token (that is how its EOS
            # is dropped). A row that stopped earlier contributes pad, purely to
            # keep input_ids rectangular.
            already_done = finished & ~stop_now
            if pad_token_id is not None:
                next_token = torch.where(
                    already_done, torch.full_like(next_token, pad_token_id), next_token
                )

            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

            if attention_mask is not None:
                # Pad fillers get mask 0, real tokens get 1.
                new_col = (~already_done).long().unsqueeze(-1)
                attention_mask = torch.cat([attention_mask, new_col], dim=-1)

            # Every row stopped on this step -> the final column is all filler.
            if bool(stop_now.all()):
                input_ids = input_ids[:, :-1]
                break

            if bool(finished.all()):
                break

    return input_ids
