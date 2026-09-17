import torch
import time

def sync_device(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def generate_batched(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model,
    max_new_tokens: int,
    eos_token_id: int | None,
    pad_token_id: int,
    temperature=0.0,
    top_k=None,
    top_p=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    output_ids:
        [B, prompt_width + steps] padded tensor. Rows that stopped keep their
        EOS in the sequence, and every later slot of that row is PAD.

    generated_lengths:
        [B] int64. How many *valid* tokens each request produced. EOS and the
        post-stop PAD slots are never counted, so this is the number of tokens
        the caller is allowed to show the user.
    """
    with torch.no_grad():

        device = input_ids.device

        sync_device(device)
        generation_start = time.perf_counter()

        token_times = []


        # Prefill
        out = model(input_ids, attention_mask=attention_mask, use_cache=True)
        last_logits = out.logits[:, -1, :]
        cache = out.past_key_values

        batch_len = input_ids.shape[0]
        finished = torch.zeros(batch_len, dtype=torch.bool, device=input_ids.device)

        generated_lengths = torch.zeros(batch_len, dtype=torch.long, device=input_ids.device)

        for step in range(max_new_tokens):
            if temperature == 0.0:
                next_token = torch.argmax(last_logits, dim=-1)
            else:
                new_logits = last_logits.float() / temperature

                # top-k
                if top_k is not None and top_k > 0:
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

                # [B, 1] -> [B] so every tensor below keeps the same batch shape.
                next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

            sync_device(device=device)
            token_times.append(time.perf_counter())

            active_before = ~finished
            stop_now = (
                (next_token == eos_token_id) & active_before
                if eos_token_id is not None
                else torch.zeros_like(finished)
            )
            valid_generated = active_before & ~stop_now
            generated_lengths += valid_generated.long()
            
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

            new_mask_col = (~already_done).long().unsqueeze(-1)
            attention_mask = torch.cat(
                [attention_mask, new_mask_col],
                dim=-1
            )

            if bool(finished.all()):
                break

            if (step == max_new_tokens - 1):
                break

            # Decode
            out = model(
                next_token.unsqueeze(-1),
                past_key_values=cache,
                attention_mask=attention_mask,
                use_cache=True
            )

            last_logits = out.logits[:, -1, :]
            cache = out.past_key_values

        ttft =  (token_times[0] - generation_start)
        itls = [
            (token_times[i] - token_times[i-1]) * 1000
            for i in range(1, len(token_times))
        ]

        tpot = (
            (sum(itls) / len(itls)) if itls else None
        )

        print("TTFT: ", ttft * 1000)
        print("ITLs ", itls)
        print("TPOT: ", tpot * 1000 if tpot else None)

    return input_ids, generated_lengths