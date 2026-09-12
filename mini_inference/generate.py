"""naive greedy autoregressive decoding.

    prompt -> tokenize -> input_ids -> model -> logits
           -> logits at the final position -> argmax -> next_token_id
           -> append to input_ids -> repeat -> stop

No KV cache. No sampling. Every iteration pushes the ENTIRE sequence built so
far through all 24 layers, then appends exactly one token. This is deliberately
the slow, obviously-correct reference implementation; Day 3's cached decoder
gets checked against it.

The body of ``generate_greedy`` is intentionally left to you.
See ``notes/day01_naive_greedy.md`` for the intuition, invariants and the
step-by-step TODO ladder.
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
    input_ids
        Prompt token ids. Shape ``[B, T]``, dtype ``torch.int64``, already on the
        model's device.
    model
        A causal LM in ``eval()`` mode. Only ``model(input_ids).logits`` is used.
    max_new_tokens
        Upper bound on the number of tokens appended.
    eos_token_id
        If given, stop a batch row once this token has been appended.
    pad_token_id
        Token used to fill rows that have already finished, so that the batch
        stays rectangular ``[B, T]``. Falls back to ``model.config.pad_token_id``.
        Required for ``B > 1`` with early stopping (Qwen2.5-0.5B sets it to None).

    Returns
    -------
    torch.Tensor
        Shape ``[B, T + n]`` with ``n <= max_new_tokens``, dtype int64. Columns
        ``[T:]`` are the newly generated tokens. A row that finished early is
        padded with ``pad_token_id`` up to the batch's final length.

    Invariants to preserve while writing this
    -----------------------------------------
    1. The running sequence always holds prompt + everything generated so far,
       in order. Nothing is ever dropped or overwritten.
    2. Inference builds no autograd graph.
    3. The tensor that becomes the next token has shape ``[B]`` — a token id is
       one scalar per batch row, never a logit vector or a probability.
    4. The model call happens inside the loop, on the *current* sequence. If you
       find yourself calling the model once outside the loop, stop.
    5. With ``B > 1``, the row's last REAL position depends on the padding side
       (T = number of columns)::

           LEFT-padded   [pad pad A B C]   last real index = T-1 = -1  (all rows)
           RIGHT-padded  [A B C pad pad]   last real index = mask.sum(1)-1
           no padding                      last real index = T-1 = -1  (all rows)

       Using ``mask.sum(1)-1`` on a LEFT-padded batch returns the ORDINAL of the
       last real token, not its index — it silently reads an early token.
       Using ``-1`` on a RIGHT-padded batch reads a pad position. Both return
       plausible-looking garbage instead of raising.

    Measured facts you can rely on (Qwen2.5-0.5B, B=1, T=5)
    -------------------------------------------------------
        input_ids             [1, 5]           int64
        model(...).logits     [1, 5, 151936]   float16
        logits[:, -1, :]      [1, 151936]      <-- valid ONLY when there is no padding
        argmax(dim=-1)        [1]              int64   -> 12095 -> ' Paris'
    """
    # ------------------------------------------------------------------
    # `finished` is the whole trick behind batched EOS.
    #
    # The B=1 version used `break`. That is a property of the FUNCTION.
    # With B>1, "done" becomes a property of each ROW, so it has to be a
    # tensor of shape [B] that survives across loop iterations.
    #
    # The loop itself can never break early: row 0 may finish on step 3
    # while row 1 is still generating. We keep stepping until every row is
    # finished, and pad the finished rows instead of breaking.
    # ------------------------------------------------------------------
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    # Use an explicit pad id if given; otherwise fall back to the model's.
    # Qwen2.5-0.5B has config.pad_token_id = None, so this can legitimately
    # be None -> then B>1 with early stopping cannot be supported safely.
    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)

    with torch.no_grad():
        # ------------------------------------------------------------------
        # `attention_mask` is RUNNING STATE, exactly like input_ids.
        #
        # It is not a step-0-only thing: it must gain a column every time
        # input_ids does, or it stops describing the sequence. A stale mask is
        # how you get a model that attends to padding -- symptom is output
        # built from the pad token's neighbourhood, e.g. 'HumanHuman' after a
        # row of <|endoftext|> pads.
        #
        # Only needed when padding is present; otherwise all-ones is implied.
        # ------------------------------------------------------------------
        needs_mask = pad_token_id is not None and bool((input_ids == pad_token_id).any())
        if needs_mask:
            attention_mask = (input_ids != pad_token_id).long()
            # Infer the padding side from the data rather than trusting a flag:
            # if row 0's first column is a pad, the batch is left-padded.
            padding_is_left = bool(attention_mask[0, 0].item() == 0)
        else:
            attention_mask = None
            padding_is_left = False

        for step in range(max_new_tokens):
            if attention_mask is not None:
                logits = model(input_ids, attention_mask=attention_mask).logits
            else:
                logits = model(input_ids).logits

            # ------------------------------------------------------------------
            # Index of each row's LAST REAL token -> [B].
            #
            # MUST be recomputed every step: `input_ids` gains a column each
            # iteration, so an index captured on step 0 goes stale immediately.
            # Symptom of getting this wrong: the model reads the distribution
            # for a position it has ALREADY generated, so output degenerates
            # into repeating one token forever ('Delhi Delhi Delhi ...').
            #
            # THE FORMULA DEPENDS ON PADDING SIDE -- and conflating them is the
            # trap. With T columns:
            #
            #   LEFT-padded   [pad pad A B C]  real tokens end at tensor index
            #                 0   1   2 3 4   T-1, i.e. -1, for EVERY row.
            #                                  mask.sum(1)-1 = 2 here, which is
            #                                  the ordinal "2nd real token", NOT
            #                                  a tensor index. Wrong.
            #
            #   RIGHT-padded  [A B C pad pad]  real tokens end at
            #                 0 1 2 3   4     mask.sum(1)-1 = 2 == tensor index.
            #                                  Correct.
            #
            # So: left  -> T-1 for all rows;  right -> mask.sum(1)-1.
            # Measured consequence of using mask.sum(1)-1 with left padding:
            # batched logits gave '-' where the same prompt run alone gave
            # ' Delhi'.
            # ------------------------------------------------------------------
            T_cur = input_ids.shape[1]
            if needs_mask and not padding_is_left:
                # RIGHT-padded: the row's last real token is at its own offset.
                last_index = attention_mask.sum(dim=-1) - 1
            else:
                # No padding, or LEFT-padded: every row ends on a real token,
                # so the last real position is the final column for all rows.
                last_index = torch.full(
                    (input_ids.shape[0],), T_cur - 1,
                    device=input_ids.device, dtype=torch.long,
                )

            # ------------------------------------------------------------------
            # Gather each row's own last position.
            #
            # `logits[:, -1, :]` is WRONG here and fails SILENTLY: with a
            # right-padded batch, row 1's index -1 is a pad token, so you read
            # a distribution over "what follows padding" — measured on this
            # model it confidently returned 'Human'. No error, just garbage.
            #
            # Advanced indexing: both index tensors are flat [B], and they are
            # broadcast PAIRWISE (element i with element i), so this picks
            # logits[0, last_index[0], :] and logits[1, last_index[1], :].
            # Result is [B, V] — the vocab dim survives, the indexed dims do not.
            #
            # Gotcha: passing two [B, 1] tensors instead gives [B, 1, V] (the
            # trailing 1 is kept), which then breaks torch.cat with a
            # "got 2 and 3 dimensions" error.
            # ------------------------------------------------------------------
            rows = torch.arange(input_ids.shape[0], device=input_ids.device)
            last = logits[rows, last_index]                    # [B, V]
            next_token = last.argmax(dim=-1)                   # [B] int64

            # ------------------------------------------------------------------
            # Batched EOS. Note what is NOT here: `.any()`.
            #
            # The B=1 version did `(next == eos).any()`, which asks "did ANY row
            # finish?". With B>1 that kills every row as soon as one row
            # finishes — the classic batched-generation bug. We OR into
            # `finished` instead, so each row stops independently.
            #
            # CONVENTION NOTE — where the EOS token ends up:
            #   HuggingFace keeps it:   ... city . <eos>  (stops AFTER appending)
            #   this function drops it: ... city .         (stops BEFORE appending)
            # Both are defensible; "the user does not want the stop token in
            # their text" is why this one trims. The length differs by one per
            # row that actually hit EOS, so a mismatch with model.generate()
            # here is a convention difference, not a logits bug.
            # ------------------------------------------------------------------
            stop_now = torch.zeros_like(finished)
            if eos_token_id is not None:
                stop_now = (next_token == eos_token_id) & ~finished
                finished = finished | stop_now

            # A row that just stopped contributes NO token: that is how it gets
            # trimmed. A row that stopped on an EARLIER step still contributes
            # pad_id, purely to keep input_ids rectangular [B, T] for the model.
            # (Note pad_id == eos_id here, so appending pad to a row that is
            # already `finished` cannot re-trigger the check above — the
            # `& ~finished` guard handles it.)
            if pad_token_id is not None:
                already_done = finished & ~stop_now
                next_token = torch.where(
                    already_done, torch.full_like(next_token, pad_token_id), next_token
                )

            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

            # Keep the running mask in lockstep with input_ids. Every appended
            # token is real (either a generated token or, for an already-
            # finished row, a pad filler) -- so pad fillers get mask 0 and the
            # rest get 1. Getting this wrong reintroduces the stale-mask bug.
            if attention_mask is not None:
                new_col = (~already_done).long().unsqueeze(-1)
                attention_mask = torch.cat([attention_mask, new_col], dim=-1)

            # Every row stopped on THIS step contributed nothing, so the whole
            # final column is filler -> drop it.
            if bool(stop_now.all()):
                input_ids = input_ids[:, :-1]
                break

            # Stop when no row is still live. Checked after appending so that
            # rows which stopped earlier keep their pad columns.
            if bool(finished.all()):
                break

    # ------------------------------------------------------------------
    # Output convention: rectangular [B, T+n]. A row that hit EOS earlier
    # than the longest row carries pad_token_id in its trailing columns.
    # Callers that need per-row lengths should derive them with
    # (output != pad_token_id).sum(1) -- or pass return_lengths, which is
    # deliberately NOT implemented here: Day 1 should not grow a scheduler.
    # ------------------------------------------------------------------
    return input_ids
