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
PROMPTS = [
    "The capital of India is New",
    "The future of Artifical Intelligenece is",
]
MAX_NEW_TOKENS = 10


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


def batch_check(tokenizer, model) -> bool:
    enc = tokenizer(PROMPTS, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(model.device)
    dot = tokenizer.encode(".", add_special_tokens=False)[0]
    out = generate_cached(input_ids, model, max_new_tokens=MAX_NEW_TOKENS, eos_token_id=dot, pad_token_id=tokenizer.pad_token_id, temperature=0.0)
    print(f"\ninput_ids shape: {tuple(input_ids.shape)}")
    print(f"output shape   : {tuple(out.shape)}")
    for i, prompt in enumerate(PROMPTS):
        print(f"prompt {i}: {prompt!r}")
        print(f"  input_ids: {input_ids[i].tolist()}")
        print(f"  output   : {out[i].tolist()}")
        print(f"  decoded  : {tokenizer.decode(out[i])!r}")


def main() -> None:
    tokenizer, model = load_model()
    print(f"model: {MODEL_NAME} on {model.device}\n")

    results = {}
    corr = correctness(tokenizer, model)
    results.update({f"correctness: {k!r}": v for k, v in corr.items()})


if __name__ == "__main__":
    main()
