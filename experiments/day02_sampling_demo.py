"""Day 2 experiment — how do temperature, top-k and top-p change the text?

    uv run python experiments/day02_sampling_demo.py
    uv run python experiments/day02_sampling_demo.py --prompt "Once upon a time"

Question
    Which knob does what, and what do the settings actually look like in text?

This is an OBSERVATION script, not a test. It prints text so you can see the
effect. The quantitative version is day02_sampling_stats.py.

Reading the output
------------------
  temperature < 1   confident, repetitive, may loop
  temperature = 1   the model's own distribution
  temperature > 1   diverse, often incoherent (word salad at 1.5+)

  top_k             hard cap on candidates; k=1 is greedy
  top_p             adaptive cap; small p ~ greedy when confident

Note how the SAME temperature gives different-feeling text depending on how
confident the model is at that point. Temperature is a relative dial, not an
absolute quality setting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.common import encode, load_model
from mini_inference.sampling import sample_next_token

SETTINGS = [
    ("greedy (T=0.01)      ", dict(temperature=0.01)),
    ("T=0.3                ", dict(temperature=0.3)),
    ("T=0.7                ", dict(temperature=0.7)),
    ("T=1.0                ", dict(temperature=1.0)),
    ("T=1.5                ", dict(temperature=1.5)),
    ("T=2.5                ", dict(temperature=2.5)),
    ("top_k=1  (greedy)    ", dict(top_k=1)),
    ("top_k=5              ", dict(top_k=5)),
    ("top_k=50             ", dict(top_k=50)),
    ("top_p=0.5            ", dict(top_p=0.5)),
    ("top_p=0.9            ", dict(top_p=0.9)),
    ("top_p=0.95           ", dict(top_p=0.95)),
    ("T=0.8, top_p=0.9     ", dict(temperature=0.8, top_p=0.9)),
    ("T=1.0, top_k=50, p=.9", dict(temperature=1.0, top_k=50, top_p=0.9)),
]


def generate(input_ids, model, max_new_tokens, *, eos_token_id=None, **kw):
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids).logits[:, -1, :]
            nxt = sample_next_token(logits, **kw)
            input_ids = torch.cat([input_ids, nxt.unsqueeze(-1)], dim=-1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
    return input_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of India is")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tokenizer, model = load_model()
    input_ids = encode(tokenizer, model, args.prompt)
    n_prompt = input_ids.shape[1]

    print(f"prompt : {args.prompt!r}  ({n_prompt} tokens)")
    print(f"new    : {args.max_new_tokens} tokens max | seed {args.seed}")
    print("=" * 78)
    for name, kw in SETTINGS:
        torch.manual_seed(args.seed)
        out = generate(input_ids, model, args.max_new_tokens,
                       eos_token_id=model.config.eos_token_id, **kw)
        text = tokenizer.decode(out[0, n_prompt:])
        print(f"{name} | {text!r}")
    print("=" * 78)
    print("same seed for every row: differences come from the SETTING, not the draw")


if __name__ == "__main__":
    main()
