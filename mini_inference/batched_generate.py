import torch


def generate_batched(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model,
    max_new_tokens: int,
    eos_token_id: int | None,
    pad_token_id: int,
) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids, attention_mask=attention_mask, use_cache=True)
        last_logits, cache = out.logits[:, -1, :], out.past_key_values

        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

        for step in range(max_new_tokens):
            next_token = torch.argmax(input=last_logits, dim=-1)

            stop_now = (
                (next_token == eos_token_id) & ~finished
                if eos_token_id is not None
                else torch.zeros_like(finished)
            )
            finished = finished | stop_now

            already_done = finished & ~stop_now
            next_token = torch.where(
                already_done,
                torch.full_like(next_token, pad_token_id),
                next_token
            )

            input_ids = torch.cat(
                [input_ids, next_token.unsqueeze(-1)],
                dim=-1
            )

            new_mask_col= (~already_done).long().unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, new_mask_col],
                dim=-1
            )

            if bool(finished.all()):
                break

            if (step == max_new_tokens -1):
                break

            out = model(
                next_token.unsqueeze(-1),
                past_key_values=cache,
                attention_mask=attention_mask,
                use_cache=True
            )
            last_logits = out.logits[:, -1, :]
            cache = out.past_key_values

    return input_ids