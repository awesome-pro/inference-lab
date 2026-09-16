import torch


def generate_greedy(
    input_ids: torch.Tensor,
    model,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    with torch.inference_mode():
        for step in range(max_new_tokens):
            logits = model(input_ids).logits
            last = logits[:, -1, :]
            next_token = torch.argmax(last, dim=-1)
    
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
            
            if next_token == eos_token_id:
                break
        
    return input_ids