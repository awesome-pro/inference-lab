# Day 3 — KV cache

## Goal

Stop re-feeding the whole sequence to the model on every step.

```
Day 1/2:   model(input_ids=[everything so far])      every step
Day 3:     model(input_ids=[prompt])       once     <- PREFILL
           model(input_ids=[one token])    n times  <- DECODE
```

---

## What I need to understand

Day 1 re-sends the growing sequence every step. The model recomputes keys and
values for tokens it has already seen. Nothing about those tokens changed — so
the K/V can be kept instead of rebuilt.

Only K and V are kept. Queries are not: a query is used once, to produce that
position's output, and never again. Every *future* token, though, needs to
attend back to every previous K and V.

Two calls. Same function, **two arguments differ**:

```python
# PREFILL — once
out    = model(input_ids, attention_mask=mask, use_cache=True)
logits = out.logits[:, -1, :]        # [B, V]
cache  = out.past_key_values         # holds T positions

# DECODE — each step
out    = model(input_ids[:, -1:], attention_mask=mask, past_key_values=cache, use_cache=True)
logits = out.logits[:, -1, :]        # [B, V]
cache  = out.past_key_values         # holds T+1, T+2, ...
```

| | prefill | decode |
|---|---|---|
| `input_ids` | `[B, T]` — everything | `[B, 1]` — just the new token |
| `past_key_values` | omit it (`None`) | `cache` |

That is the whole change. Mask bookkeeping, EOS handling, padding, and the
selection step are all Day 1 code and stay as they are.

`input_ids` and `attention_mask` disagree in length during decode
(`[B, 1]` vs `[B, n+1]`). That is correct: `input_ids` is "what is new", the
mask is "what exists". The cache holds the difference.

---

## What I need to implement

Two functions in `mini_inference/cached_generate.py`:

- `prefill` — 3 lines
- `decode_step` — 3 lines

The exact call is in the TODO comment of each. Then in the loop, swap
`decode_step_reference` for `decode_step` and change the argument from
`input_ids` to `input_ids[:, -1:]`.

---

## How I know I am done

```python
generate_cached(ids, model, max_new_tokens=8, temperature=0.0) \
    == generate_greedy(ids, model, 8)
```

Token for token. Day 1 stays the reference implementation.

While working: after prefill, `cache.get_seq_length()` must equal `T`. The
skeleton prints this — it currently says `0`, because the reference path returns
`None`.

---

## If it produces different tokens

Check in this order:

1. Does `input_ids` agree with `attention_mask` about how many tokens exist?
2. Is `cache.get_seq_length() == attention_mask.shape[1]`?
3. Does it also differ on CPU? (If yes, it is not a device/precision thing.)
4. Only then think about dtype.

The mistake I made building this: an uncached call given `[B, 1]` input and a
`[B, n+1]` mask. Legal shapes, no error, wrong token — the model scored the new
token as if the prefix never existed.

---

## Experiment

**Question** — does a KV cache change the output, and how much work does it save?

**Measure** — token equality vs `generate_greedy`, and wall-clock time for
cached vs uncached at the same `max_new_tokens`.

**Result** — (your numbers, prompt "The capital of India is", 32 new tokens, MPS)

| | naive | cached |
|---|---|---|
| total time | 993.8 ms | 576.1 ms |
| speedup | — | **1.75x** |
| positions processed (theory) | 656 | 37 (17.73x) |

Correctness: identical tokens on both test prompts. Cache after prefill:
`DynamicCache, holds 5 positions`.

**Why 1.75x and not 17.73x** — decode is memory-bound. Cached decode processes
one token but still reads all 24 layers of weights plus the whole cache every
step. At batch size 1 there is nothing to amortise that read against. The
17.73x is a count of *positions*, not of work actually saved.

**Known gap** — the harness only covers `B=1`, unpadded. Batched decode with a
cache is untested; deferred to Day 5.

---
---

# Appendix

Not needed to implement Day 3. Read when a specific question sends you here.

## A1. Cache shape and memory

Per layer: `k_cache` and `v_cache`, both `[B, H_kv, n, D]`, growing along `n`.

```
bytes = 2 (K,V) x 24 layers x B x 2 (H_kv) x n x 64 (D) x 2 (fp16)
      = B x n x 12288 bytes  =  12 KiB per token per sequence
```

`H_kv = 2` while `H_q = 14` — Grouped Query Attention. The model expands the 2
KV heads to 14 internally. This is why the cache is small; it is not something
you implement.

## A2. What the cache does and does not change

Steps 1, 2 and 5 of a layer (Q/K/V projections, MLP) now process **one** token
instead of `n+1`. The attention matmul still touches the full cache width.

| | positions processed |
|---|---|
| naive | `T + (T+1) + ... + (T+n-1)` = `nT + n(n-1)/2` |
| cached | `T + n` |

For `T=5, n=20`: 230 → 25.

Caveat: decode is memory-bound, not compute-bound. One token per step, but the
whole model plus the whole cache is read each time.

## A3. How HF represents it

- `DynamicCache` — one `DynamicLayer` per layer, each holding `keys`/`values`.
- Appending uses `torch.cat`, so it reallocates every step. That `O(n)` copy per
  layer per step is what vLLM's PagedAttention removes by writing into
  preallocated blocks.
- RoPE is applied **before** the cache update, so stored keys are already
  rotated. That is why no `position_ids` are passed on decode steps — the model
  derives them as `arange(q_len) + past_seen_tokens`.

## A4. Production differences

| concern | here | production |
|---|---|---|
| allocation | `torch.cat` per step | fixed blocks, `O(1)` append |
| sharing | none | prefix caching across requests |
| eviction | finished rows keep wasting slots | slots freed on completion |
| batching | static | continuous batching |
| prefill/decode | same process | sometimes disaggregated |

## A5. Open questions

- (Your questions here.)
