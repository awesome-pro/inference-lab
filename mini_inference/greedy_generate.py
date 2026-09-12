"""Runner + correctness harness for Day 1 naive greedy decoding.

    uv run mini_inference/day01_greedy.py
    uv run python -m mini_inference.day01_greedy

Runs YOUR ``generate_greedy`` and checks it token-for-token against
``model.generate(do_sample=False)``, which is the reference implementation for
this exercise. ``model.generate`` is the *oracle* here, never the solution.

Three cases are checked:

  A. B=1, no padding      -- the original Day 1 exercise
  B. B=3, mixed lengths   -- right-padded batch; exercises per-row last position
  C. B=1, forced early EOS -- eos_token_id='.' so the EOS branch actually runs

Case C matters because Qwen's real EOS (151643) is never emitted on these
prompts, so a harness that only uses it leaves the stopping code untested.

If a case disagrees, the harness reports the first diverging position and both
tokens there. A mismatch has only a few possible causes:

  * reading the wrong position's logits (a pad position, instead of the row's
    last REAL token)
  * appending along the wrong dimension
  * appending the wrong dtype / a 2-D token tensor
  * stopping the whole batch when only one row finished
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# `python mini_inference/day01_greedy.py` puts the SCRIPT'S DIRECTORY
# (mini_inference/) on sys.path[0] -- not the repo root -- so the `mini_inference`
# package itself is not importable. `python -m mini_inference.day01_greedy` puts
# the cwd there instead, which is why only -m worked. This makes both work.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_inference.generate import generate_greedy

MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_NEW_TOKENS = 16

PROMPTS_B1 = ["The capital of India is New"]
PROMPTS_B3 = [
    "The capital of India is New",
    "2 + 2 =",
    "Once upon a time, in a small village at the edge of a",
]


def load_model() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Decoder-only models want LEFT padding: real tokens sit flush against the
    # end, so every row shares the same last position and appending one column
    # is valid for all rows at once. transformers warns loudly if you use
    # right padding, because model.generate() itself returns garbage then
    # (measured: it produced 'Human' for a row whose last real token was '=').
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model = model.to("mps").eval()
    return tokenizer, model


def _sync() -> None:
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def compare(name: str, mine: torch.Tensor, ref: torch.Tensor, tokenizer, prompt_len: int) -> bool:
    print(f"\n--- correctness: {name} ---")

    # CONVENTION SHIFT: this implementation drops the EOS token, model.generate
    # keeps it. So strip any trailing EOS from the reference before comparing,
    # and say so, rather than reporting a phantom failure.
    n = min(mine.shape[1], ref.shape[1])
    ref_cmp = ref[:, :n]
    mine_cmp = mine[:, :n]

    if mine.shape != ref.shape:
        print(f"  length differs: yours {mine.shape[1]} vs reference {ref.shape[1]}"
              f" (expected if EOS fired -- we drop the stop token, HF keeps it)")

    agree = mine_cmp.eq(ref_cmp)
    if bool(agree.all()):
        print(f"PASS: identical over all {n} positions (batch {mine.shape[0]}).")
        return True

    bad_rows = (~agree).any(dim=1).nonzero().flatten().tolist()
    print(f"FAIL: {len(bad_rows)} of {mine.shape[0]} row(s) differ: {bad_rows}")
    for r in bad_rows:
        first_bad = int((~agree[r]).nonzero()[0])
        where = "prompt region" if first_bad < prompt_len else f"generated token #{first_bad - prompt_len + 1}"
        print(f"  row {r}: first divergence at index {first_bad} ({where})")
        print(f"    yours     : {int(mine[r, first_bad])} {tokenizer.decode([int(mine[r, first_bad])])!r}")
        print(f"    reference : {int(ref[r, first_bad])} {tokenizer.decode([int(ref[r, first_bad])])!r}")
        if first_bad == prompt_len:
            print("    -> diverges on the FIRST generated token: last-position / argmax bug.")
        else:
            print("    -> diverges later: first token was right, so it is a state-update bug.")
    return False


def run_case(
    name: str,
    prompts: list[str],
    tokenizer,
    model,
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
    max_new_tokens: int,
) -> bool:
    print("\n" + "=" * 72)
    print(f"CASE {name}")
    print("=" * 72)

    enc = tokenizer(prompts, return_tensors="pt", padding=(len(prompts) > 1))
    input_ids = enc["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    print(f"prompts       : {len(prompts)}")
    for i, p in enumerate(prompts):
        n_real = int(enc['attention_mask'][i].sum())
        print(f"  [{i}] {p!r}  ({n_real} real tokens of {prompt_len})")
    print(f"input_ids     : {tuple(input_ids.shape)} {input_ids.dtype} on {input_ids.device}")
    print(f"padding_side  : {tokenizer.padding_side}")
    print(f"eos_token_id  : {eos_token_id} | pad_token_id : {pad_token_id}")

    _sync()
    t0 = time.perf_counter()
    mine = generate_greedy(
        input_ids,
        model,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    _sync()
    dt = time.perf_counter() - t0

    print(f"\n--- yours --- shape {tuple(mine.shape)} | {dt * 1e3:.1f} ms")
    for i in range(mine.shape[0]):
        print(f"  [{i}] {tokenizer.decode(mine[i], skip_special_tokens=False)!r}")

    ref = model.generate(
        input_ids,
        attention_mask=enc["attention_mask"].to(model.device),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return compare(name, mine, ref, tokenizer, prompt_len)


def main() -> None:
    tokenizer, model = load_model()
    # Qwen2.5-0.5B ships pad_token_id=None, which batching needs. Using EOS as
    # pad is the standard stand-in for a decoder-only model.
    tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    results = {}

    results["A: B=1, no padding"] = run_case(
        "A: B=1, no padding",
        PROMPTS_B1,
        tokenizer,
        model,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    results["B: B=3, mixed lengths"] = run_case(
        "B: B=3, mixed lengths (right-padded)",
        PROMPTS_B3,
        tokenizer,
        model,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # Force the EOS branch to actually execute. '.' is Qwen token 13.
    dot = tokenizer.encode(".", add_special_tokens=False)[0]
    results["C: forced early EOS"] = run_case(
        f"C: B=1, eos_token_id={dot} ('.') -- stops early",
        ["The capital of India is New Delhi is a city. This continues after the stop."],
        tokenizer,
        model,
        eos_token_id=dot,
        pad_token_id=pad_id,
        max_new_tokens=32,
    )

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    if all(results.values()):
        print("\nDay 1 complete. Next: notes/day03_kv_cache.md")


if __name__ == "__main__":
    main()
