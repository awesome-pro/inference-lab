"""Runner + correctness harness for Day 3 cached generation.

    uv run mini_inference/cached_generate_harness.py
    uv run python -m mini_inference.cached_generate_harness

Checks YOUR ``generate_cached`` against ``generate_greedy`` (Day 1), which is
the reference implementation. Same idea as the Day 1 harness: the slow version
is the oracle, never the solution.

Two things are checked, and they answer different questions.

  PART 1 -- CORRECTNESS
      Does cached generation produce the same tokens as naive generation?

      This runs with the skeleton as shipped (which uses the slow uncached
      reference path), so it PASSES before you write anything. That is
      deliberate: it tells you the harness works, and it means any later FAIL
      is caused by your change.

  PART 2 -- SPEED
      Does it actually avoid work?

      Prints total time and per-call time for both paths. THIS is the number
      that should change when you implement the two functions. If correctness
      still passes but this number does not move, your decode step is not
      actually using the cache.

Run it after every edit. A correct-but-slow result means the cache is not being
threaded through; a fast-but-wrong result means the mask or the input slice is
inconsistent -- see notes/day03_kv_cache.md.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_inference.cached_generate import generate_cached
from mini_inference.generate import generate_greedy

MODEL_NAME = "Qwen/Qwen2.5-0.5B"

# Prompts used for the equivalence check. Mixing a factual prompt with an
# open-ended one is intentional: they exercise different continuations.
PROMPTS = [
    "The capital of India is",
    "2 + 2 =",
]

# Generation length. Longer makes the speed gap more visible; 32 is enough to
# be clearly uneven between the two paths without a slow test.
MAX_NEW_TOKENS = 32


def load_model() -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16)
    model = model.to(device).eval()
    return tokenizer, model


def _sync() -> None:
    """Make timing honest on a GPU.

    MPS/CUDA kernels are queued asynchronously, so perf_counter() around a call
    can measure almost nothing unless we wait for the queue to drain.
    """
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, *args, **kwargs):
    """Run ``fn`` and return (result, seconds)."""
    _sync()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    _sync()
    return result, time.perf_counter() - t0


def compare(mine: torch.Tensor, ref: torch.Tensor, tokenizer, prompt_len: int) -> bool:
    """Report the first token where the two disagree."""
    if mine.shape != ref.shape:
        print(f"  SHAPE MISMATCH  yours {tuple(mine.shape)}  reference {tuple(ref.shape)}")
        print("  -> different length usually means a stopping-condition difference,")
        print("     not necessarily a cache bug.")
        n = min(mine.shape[1], ref.shape[1])
    else:
        n = mine.shape[1]

    if torch.equal(mine[:, :n], ref[:, :n]):
        return True

    for r in range(mine.shape[0]):
        for i in range(n):
            if int(mine[r, i]) != int(ref[r, i]):
                where = "prompt" if i < prompt_len else f"generated token #{i - prompt_len + 1}"
                print(f"  row {r}: first divergence at index {i} ({where})")
                print(f"    yours     : {int(mine[r, i])} {tokenizer.decode([int(mine[r, i])])!r}")
                print(f"    reference : {int(ref[r, i])} {tokenizer.decode([int(ref[r, i])])!r}")
                if i == prompt_len:
                    print("    -> first generated token: check prefill's logits slice.")
                else:
                    print("    -> later token: check the decode input slice and the mask length.")
                break
        break
    return False


def correctness(tokenizer, model) -> dict[str, bool]:
    print("=" * 72)
    print("PART 1 -- CORRECTNESS: cached vs naive, token for token")
    print("=" * 72)

    results = {}
    for prompt in PROMPTS:
        enc = tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(model.device)

        print(f"\nprompt: {prompt!r}")
        print(f"  input_ids {tuple(input_ids.shape)} -> {input_ids[0].tolist()}")

        ref = generate_greedy(input_ids, model, max_new_tokens=MAX_NEW_TOKENS)
        mine = generate_cached(
            input_ids, model, max_new_tokens=MAX_NEW_TOKENS, temperature=0.0
        )

        print(f"  naive  ({ref.shape[1] - input_ids.shape[1]} new): "
              f"{tokenizer.decode(ref[0, input_ids.shape[1]:])!r}")

        ok = compare(mine, ref, tokenizer, input_ids.shape[1])
        if ok:
            print(f"  cached ({mine.shape[1] - input_ids.shape[1]} new): "
                  f"{tokenizer.decode(mine[0, input_ids.shape[1]:])!r}")
            print("  PASS")
        results[prompt] = ok
    return results


def speed(tokenizer, model) -> None:
    print("\n" + "=" * 72)
    print("PART 2 -- SPEED: is the cache actually avoiding work?")
    print("=" * 72)

    prompt = PROMPTS[0]
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
    n = MAX_NEW_TOKENS

    # Warm up, so the first (lazy compilation / cache-filling) call is not
    # counted. Without this the naive path can look slower than it is.
    generate_greedy(input_ids, model, max_new_tokens=2)
    generate_cached(input_ids, model, max_new_tokens=2, temperature=0.0)

    _, t_naive = timed(generate_greedy, input_ids, model, max_new_tokens=n)
    _, t_cached = timed(
        generate_cached, input_ids, model, max_new_tokens=n, temperature=0.0
    )

    prompt_len = input_ids.shape[1]
    naive_calls = n
    cached_calls = n  # 1 prefill + (n-1) decode; counted separately below

    print(f"\nprompt length      : {prompt_len} tokens")
    print(f"new tokens         : {n}")
    print(f"\nnaive  total       : {t_naive * 1e3:8.1f} ms")
    print(f"cached total       : {t_cached * 1e3:8.1f} ms")
    if t_cached > 0:
        print(f"speedup            : {t_naive / t_cached:8.2f}x")

    naive_positions = sum(prompt_len + i for i in range(n))
    cached_positions = prompt_len + n
    print(f"\npositions processed (theory)")
    print(f"  naive  : {naive_positions}")
    print(f"  cached : {cached_positions}")
    print(f"  ratio  : {naive_positions / cached_positions:.2f}x")

    print(f"\nforward calls      : naive {naive_calls}, cached {cached_calls} "
          f"(1 prefill + {n - 1} decode)")

    if t_cached >= t_naive * 0.95:
        print("\n  NOTE: cached is NOT faster yet. If PART 1 passed, your decode")
        print("  step is probably still calling decode_step_reference -- check that")
        print("  the loop calls decode_step with input_ids[:, -1:].")
    else:
        print("\n  Cache is doing real work. Compare against the theory ratio above;")
        print("  decode steps are memory-bound, so you will not reach it exactly.")


def prefill_check(tokenizer, model) -> bool:
    """Inspect the cache right after prefill -- the smallest possible check."""
    print("\n" + "=" * 72)
    print("PART 3 -- PREFILL: does the cache exist and hold the prompt?")
    print("=" * 72)

    from mini_inference.cached_generate import prefill

    input_ids = tokenizer(PROMPTS[0], return_tensors="pt")["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    logits, cache = prefill(input_ids, model, None)

    print(f"\nprompt length      : {prompt_len}")
    print(f"logits             : {tuple(logits.shape)}  (want [B, V] = [1, 151936])")
    if cache is None:
        print("past_key_values    : None")
        print("\n  -> prefill is not implemented yet (or returns the fallback).")
        print("     Write it, then this should say: cache holds 5 positions.")
        return False

    cache_len = cache.get_seq_length()
    print(f"past_key_values    : {type(cache).__name__}, holds {cache_len} positions")
    key_shape = tuple(cache.layers[0].keys.shape)
    print(f"layer 0 keys       : {key_shape}  (want [B, H_kv, {prompt_len}, D])")
    print(f"prompt tokens      : {input_ids[0].tolist()}")
    print(f"top-1 token        : {tokenizer.decode([int(logits.argmax(-1))])!r}")

    ok = cache_len == prompt_len
    print(f"\n  {'PASS' if ok else 'FAIL'}  cache length == prompt length")
    return ok


def main() -> None:
    tokenizer, model = load_model()
    print(f"model: {MODEL_NAME} on {model.device}\n")

    results = {}
    corr = correctness(tokenizer, model)
    results.update({f"correctness: {k!r}": v for k, v in corr.items()})

    results["prefill populates cache"] = prefill_check(tokenizer, model)

    speed(tokenizer, model)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for k, v in results.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")

    if all(results.values()):
        print("\nDay 3 complete. Fill in the experiment record in notes/day03_kv_cache.md.")
    else:
        print("\nSee notes/day03_kv_cache.md -> 'If it produces different tokens'.")


if __name__ == "__main__":
    main()
