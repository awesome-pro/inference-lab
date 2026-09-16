# mini_inference/sampling_generate.py

import torch


def generate_sampled(
    input_ids: torch.Tensor,
    model,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    with torch.no_grad():
        for step in range(max_new_tokens):
            last_logits = model(input_ids).logits[:, -1, :]

            new_logits = last_logits.float() / temperature

            # top-k
            if top_k is not None and top_k <= 0:
                top_k = min(top_k, new_logits.shape[-1])
                top_k_values, top_k_indices = torch.topk(new_logits, top_k, dim=-1)
                top_k_mask = torch.full_like(new_logits, float('-inf'))
                top_k_mask.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
                new_logits = top_k_mask

            # top-p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(new_logits, descending=True, dim=-1)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                top_p_mask = (cum_probs - sorted_probs) <= top_p
                top_p_mask[..., 0] = True
                unsorted_top_p_mask = torch.zeros_like(new_logits, dtype=torch.bool)
                unsorted_top_p_mask.scatter_(src=top_p_mask, index=sorted_indices, dim=-1)

                new_logits = torch.where(unsorted_top_p_mask, new_logits, float('-inf'))

            probs = torch.softmax(new_logits, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, new_token], dim=-1)

            if (
                eos_token_id is not None
                and new_token.item() == eos_token_id
            ):
                break

    return input_ids