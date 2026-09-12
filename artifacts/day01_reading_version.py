"""Day 1 generation loop — the READING version. Not used by anything.

This is the same logic as mini_inference/generate.py with the padding
machinery and the long bug-archaeology comments removed, so the actual
mechanism is visible. Read this first; the real file is this plus bookkeeping.

    B = batch size
    T = prompt length (number of columns)
    V = vocab size (151936)
"""

import torch


def generate_greedy(input_ids, model, max_new_tokens, eos_token_id=None, pad_token_id=None):
    B = input_ids.shape[0]                                        # batch size
    finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

    with torch.no_grad():
        for step in range(max_new_tokens):

            # ---- 1. run the model on the CURRENT sequence --------------
            # input_ids is [B, T_cur]. logits is [B, T_cur, V]:
            # one distribution per position, for every row.
            logits = model(input_ids).logits

            # ---- 2. pick each row's last real position -----------------
            # Without padding this is just the last column, T_cur - 1, for
            # every row. (Padding is what complicates it; see the real file.)
            last_index = torch.full((B,), input_ids.shape[1] - 1,
                                    device=input_ids.device, dtype=torch.long)

            # ---- 3. gather those rows -> [B, V], then argmax -> [B] ----
            rows = torch.arange(B, device=input_ids.device)
            last = logits[rows, last_index]        # [B, V]
            next_token = last.argmax(dim=-1)       # [B] int64

            # ---- 4. per-row stopping -----------------------------------
            # NOT `if (next_token == eos).any(): break` -- that would kill
            # every row when one row finishes. Track a flag per row instead.
            if eos_token_id is not None:
                stop_now = (next_token == eos_token_id) & ~finished
                finished = finished | stop_now

            # ---- 5. append one column ----------------------------------
            # [B] -> [B, 1], concatenated on the SEQUENCE dimension.
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

            # ---- 6. stop only when EVERY row is done -------------------
            if eos_token_id is not None and bool(finished.all()):
                break

    return input_ids
