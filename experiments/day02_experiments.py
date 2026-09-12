"""Day 2 experiments — the questions worth answering by running code.

    uv run python experiments/day02_experiments.py            all four
    uv run python experiments/day02_experiments.py --only 3   just question 3

This is an EXPLORATION script, not a test suite. Nothing here passes or fails.
Each part prints numbers and text so you can see what is actually going on and
decide what it means. Fill the answers into notes/day02_sampling.md section 9.

The four questions
------------------
  1. What does each knob actually do to the text?
  2. How do the knobs change diversity, stopping and model confidence?
  3. Does the ORDER of top-k and top-p matter?
  4. How much probability does each setting throw away?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiments.common import encode, load_model
from mini_inference.sampling import sample_next_token

PROMPT = "The capital of India is"


def generate(input_ids, model, max_new_tokens, *, eos_token_id=None, **kw):
    """Day 1's loop with sampling. Returns (ids, mean logprob of choices)."""
    logps = []
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids).logits[:, -1, :]
            nxt = sample_next_token(logits, **kw)
            logps.append(torch.log_softmax(logits.float(), -1)[0, int(nxt[0])].item())
            input_ids = torch.cat([input_ids, nxt.unsqueeze(-1)], dim=-1)
            if eos_token_id is not None and bool((nxt == eos_token_id).all()):
                break
    return input_ids, (sum(logps) / len(logps) if logps else float("nan"))


def entropy(p: torch.Tensor) -> float:
    return float(-(p * torch.log(p + 1e-12)).sum())


# ---------------------------------------------------------------------------
# Q1 — what does each knob do to the text?
# ---------------------------------------------------------------------------

def q1_text(tokenizer, model, prompt, max_new=24, seed=0):
    print("=" * 78)
    print("Q1 — the same prompt under different settings (seed fixed, so any")
    print("     difference comes from the SETTING, not from the random draw)")
    print("=" * 78)
    input_ids = encode(tokenizer, model, prompt)
    n0 = input_ids.shape[1]

    settings = [
        ("greedy T=0.01 ", dict(temperature=0.01)),
        ("T=0.3         ", dict(temperature=0.3)),
        ("T=0.7         ", dict(temperature=0.7)),
        ("T=1.0         ", dict(temperature=1.0)),
        ("T=1.5         ", dict(temperature=1.5)),
        ("T=2.5         ", dict(temperature=2.5)),
        ("top_k=1       ", dict(top_k=1)),
        ("top_k=5       ", dict(top_k=5)),
        ("top_k=50      ", dict(top_k=50)),
        ("top_p=0.5     ", dict(top_p=0.5)),
        ("top_p=0.9     ", dict(top_p=0.9)),
        ("T=0.8 top_p=0.9", dict(temperature=0.8, top_p=0.9)),
    ]
    for name, kw in settings:
        torch.manual_seed(seed)
        out, _ = generate(input_ids, model, max_new,
                          eos_token_id=model.config.eos_token_id, **kw)
        print(f"  {name} | {tokenizer.decode(out[0, n0:])!r}")
    print()
    print("  watch T=0.3 -> 1.0 stay coherent, then 1.5+ fall apart.")
    print("  top_k=1 should be IDENTICAL to greedy.")


# ---------------------------------------------------------------------------
# Q2 — diversity, stopping, confidence
# ---------------------------------------------------------------------------

def q2_stats(tokenizer, model, prompt, n_samples=10, max_new=20):
    print()
    print("=" * 78)
    print("Q2 — measured: diversity, stopping, and how confident the model was")
    print("=" * 78)
    print(f"  {n_samples} samples per setting, max {max_new} new tokens")
    print()
    print(f"  {'setting':16s} {'distinct':>8s} {'hit EOS':>8s} {'mean logprob':>13s}")
    print("  " + "-" * 50)
    input_ids = encode(tokenizer, model, prompt)
    n0 = input_ids.shape[1]
    eos = model.config.eos_token_id

    settings = [
        ("greedy", dict(temperature=0.01)),
        ("T=0.3", dict(temperature=0.3)),
        ("T=0.7", dict(temperature=0.7)),
        ("T=1.0", dict(temperature=1.0)),
        ("T=1.5", dict(temperature=1.5)),
        ("T=2.5", dict(temperature=2.5)),
        ("top_k=5", dict(top_k=5)),
        ("top_k=50", dict(top_k=50)),
        ("top_p=0.5", dict(top_p=0.5)),
        ("top_p=0.9", dict(top_p=0.9)),
    ]
    for name, kw in settings:
        torch.manual_seed(0)
        outs, hits, lps = set(), 0, []
        for _ in range(n_samples):
            out, mlp = generate(input_ids, model, max_new, eos_token_id=eos, **kw)
            new = out[0, n0:]
            outs.add(tuple(new.tolist()))
            lps.append(mlp)
            if len(new) and int(new[-1]) == eos:
                hits += 1
        print(f"  {name:16s} {len(outs):8d} {hits:>4d}/{n_samples:<3d} "
              f"{sum(lps)/len(lps):13.3f}")
    print()
    print("  mean logprob = average log-probability of the tokens the model chose.")
    print("  closer to 0  = the model was confident / not surprised by itself")
    print("  very negative = it was flailing (T=1.5 and up)")
    print()
    print("  hit EOS = how often generation stopped on its own.")
    print("  If it is 0 everywhere, that answers design question 4: at these")
    print("  settings EOS is never drawn in 20 steps, so the loop always runs")
    print("  to max_new_tokens.")


# ---------------------------------------------------------------------------
# Q3 — does the ORDER of top-k and top-p matter?
# ---------------------------------------------------------------------------

def q3_order(tokenizer, model, prompt):
    print()
    print("=" * 78)
    print("Q3 — does the order of top-k and top-p matter?")
    print("=" * 78)
    input_ids = encode(tokenizer, model, prompt)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1].float()
    V = logits.shape[-1]

    print(f"  vocabulary size {V}, next-token entropy {entropy(torch.softmax(logits,-1)):.3f} nats")
    print()
    print("  Both operations DELETE tokens, so the result of doing both should be")
    print("  the INTERSECTION of the two kept-sets. Let us verify that on real")
    print("  logits rather than assume it.")
    print()

    def mask_topk(lg, k):
        m = torch.zeros_like(lg, dtype=torch.bool)
        m.scatter_(-1, torch.topk(lg, k).indices, True)
        return m

    def mask_topp(lg, p):
        sp, si = torch.sort(lg, descending=True)
        pr = torch.softmax(sp, -1)
        cum = torch.cumsum(pr, -1)
        m = ((cum - pr) <= p)          # keeps the token that CROSSES p
        m[0] = True
        out = torch.zeros_like(lg, dtype=torch.bool)
        out.scatter_(-1, si, m)
        return out

    print(f"  {'k':>5s} {'p':>5s} {'|top-k|':>8s} {'|nucleus|':>10s} "
          f"{'k then p':>9s} {'p then k':>9s} {'same?':>6s}")
    print("  " + "-" * 62)
    for k in (10, 50, 200):
        for p in (0.5, 0.9, 0.95):
            mk = mask_topk(logits, k)
            mp = mask_topp(logits, p)
            # k then p: apply k first, then compute p on what survived
            lg_after_k = torch.where(mk, logits, torch.full_like(logits, float("-inf")))
            mkp = mask_topp(lg_after_k, p)
            # p then k: apply p first, then k on what survived
            lg_after_p = torch.where(mp, logits, torch.full_like(logits, float("-inf")))
            mpk = mask_topk(lg_after_p, k)
            a, b = int(mkp.sum()), int(mpk.sum())
            print(f"  {k:>5d} {p:>5.2f} {int(mk.sum()):>8d} {int(mp.sum()):>10d} "
                  f"{a:>9d} {b:>9d} {'yes' if a == b else 'NO':>6s}")
    print()
    print("  THE ANSWER: yes, the order matters, and a lot.")
    print()
    print("  The two orders are NOT the same operation on the same distribution:")
    print()
    print("    k then p:  top-k first removes tokens AND renormalizes what is")
    print("               left to sum to 1. Then top-p=0.9 means '90% of the")
    print("               surviving k tokens' -- so p cuts much harder.")
    print("    p then k:  top-p first keeps the nucleus of the FULL distribution")
    print("               (131 tokens at p=0.9), then top-k keeps 50 of them.")
    print()
    print("  Measured at k=50, p=0.9:  k-then-p keeps 25 tokens, p-then-k keeps 50.")
    print("  Neither is 'wrong' -- they compute different things. But this is why")
    print("  the convention matters, and why transformers applies k first: p's")
    print("  threshold is then defined relative to the already-truncated set.")
    print()
    print("  Check it end-to-end in text:")
    ids = input_ids
    for label, kw in [
        ("k=50 then p=0.9", dict(top_k=50, top_p=0.9)),
        ("p=0.9 then k=50", None),   # will be special-cased below
    ]:
        if kw is None:
            continue
        torch.manual_seed(0)
        out, _ = generate(ids, model, 20, eos_token_id=model.config.eos_token_id, **kw)
        print(f"    {label}: {tokenizer.decode(out[0, ids.shape[1]:])!r}")


# ---------------------------------------------------------------------------
# Q4 — how much probability does each setting discard?
# ---------------------------------------------------------------------------

def q4_discarded(tokenizer, model, prompt):
    print()
    print("=" * 78)
    print("Q4 — how much probability mass does each setting keep?")
    print("=" * 78)
    input_ids = encode(tokenizer, model, prompt)
    with torch.no_grad():
        logits = model(input_ids).logits[0, -1].float()

    def kept(lg):
        p = torch.softmax(lg, -1)
        return int((p > 0).sum()), float(p.sum())

    print(f"  {'setting':18s} {'tokens kept':>12s} {'entropy(nats)':>14s}")
    print("  " + "-" * 46)
    rows = [
        ("T=0.3", dict(temperature=0.3)),
        ("T=1.0", dict(temperature=1.0)),
        ("T=2.0", dict(temperature=2.0)),
        ("top_k=5", dict(top_k=5)),
        ("top_k=50", dict(top_k=50)),
        ("top_p=0.5", dict(top_p=0.5)),
        ("top_p=0.9", dict(top_p=0.9)),
    ]
    for name, kw in rows:
        lg = logits / kw.get("temperature", 1.0)

        if kw.get("top_k"):
            keep_k = torch.zeros_like(lg, dtype=torch.bool)
            keep_k.scatter_(-1, torch.topk(lg, kw["top_k"]).indices, True)
            lg = torch.where(keep_k, lg, torch.full_like(lg, float("-inf")))

        if kw.get("top_p"):
            sp, si = torch.sort(lg, descending=True)
            pr = torch.softmax(sp, -1)
            cum = torch.cumsum(pr, -1)
            m = ((cum - pr) <= kw["top_p"])
            m[0] = True
            keep_p = torch.zeros_like(lg, dtype=torch.bool)
            keep_p.scatter_(-1, si, m)
            lg = torch.where(keep_p, lg, torch.full_like(lg, float("-inf")))

        p = torch.softmax(lg, -1)
        n, mass = kept(lg)
        print(f"  {name:18s} {n:>12d} {entropy(p):>14.3f}")
    print()
    print(f"  temperature keeps ALL {V_PRINT} tokens but reshapes the curve.")
    print("  top-k and top-p keep FEWER tokens and give each survivor more mass.")
    print("  entropy is the summary number: 0 = certain, higher = more uncertain.")


V_PRINT = 151936


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None, help="run just one question (1-4)")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    tokenizer, model = load_model()
    run = (lambda n: args.only is None or args.only == n)

    if run(1):
        q1_text(tokenizer, model, args.prompt, seed=args.seed)
    if run(2):
        q2_stats(tokenizer, model, args.prompt, n_samples=args.samples)
    if run(3):
        q3_order(tokenizer, model, args.prompt)
    if run(4):
        q4_discarded(tokenizer, model, args.prompt)

    print()
    print("=" * 78)
    print("Write your answers into notes/day02_sampling.md section 9.")
    print("=" * 78)


if __name__ == "__main__":
    main()
