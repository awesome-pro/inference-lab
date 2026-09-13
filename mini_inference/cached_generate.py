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
    out = model(input_ids, attention_mask=attention_mask, use_cache=True)
    cache = out.past_key_values
    return out.logits[:, -1, :], cache


def decode_step(
    input_ids: torch.Tensor,
    past_key_values,
    attention_mask: torch.Tensor,
    model,
):
    out = model(input_ids[:, -1:], attention_mask=attention_mask,
                past_key_values=past_key_values, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


def decode_step_reference(
    input_ids: torch.Tensor,
    past_key_values,
    attention_mask: torch.Tensor,
    model,
):
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
    device = input_ids.device
    batch_size, prompt_len = input_ids.shape

    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)

    needs_mask = pad_token_id is not None and bool((input_ids == pad_token_id).any())
    if needs_mask:
        attention_mask = (input_ids != pad_token_id).long()
    else:
        attention_mask = torch.ones_like(input_ids)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with torch.no_grad():
        logits, past_key_values = prefill(
            input_ids, model, attention_mask if needs_mask else None
        )
        cache_len = 0 if past_key_values is None else past_key_values.get_seq_length()
        print(f"  prefill: prompt T={prompt_len}, cache holds {cache_len}")

        for step in range(max_new_tokens):
            if temperature <= 0.0:
                pick = logits.float() if debug_argmax else logits
                next_token = pick.argmax(dim=-1)                 # [B] int64
            else:
                next_token = sample_next_token(
                    logits, temperature=temperature, top_k=top_k, top_p=top_p
                )                                                # [B] int64

            stop_now = torch.zeros_like(finished)
            if eos_token_id is not None:
                stop_now = (next_token == eos_token_id) & ~finished
                finished = finished | stop_now

            if pad_token_id is not None:
                already_done = finished & ~stop_now
                next_token = torch.where(
                    already_done, torch.full_like(next_token, pad_token_id), next_token
                )

            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)

            new_col = (~already_done).long().unsqueeze(-1) if pad_token_id is not None \
                else torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=device)
            attention_mask = torch.cat([attention_mask, new_col], dim=-1)

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
