import torch


def generate_cached(
    input_ids: torch.Tensor,
    model,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        # Prefill
        out = model(input_ids, use_cache=True)
        last_logits, cache = out.logits[:, -1, :], out.past_key_values

        for step in range(max_new_tokens):

            # 1. Select the next token 
            next_token = torch.argmax(
                last_logits, 
                dim=-1
            ) #[B]

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token.unsqueeze(-1)
                ],
                dim=-1
            )

            if(
                eos_token_id is not None and next_token.item() == eos_token_id
            ):
                break

            if (step == max_new_tokens - 1):
                break

            # Decode
            out = model(
                next_token.unsqueeze(-1),  #[B, 1]
                past_key_values=cache,
                use_cache=True
            )
            last_logits = out.logits[:, -1, :]
            cache = out.past_key_values

        return input_ids