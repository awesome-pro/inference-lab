# Day 2 — sampling: temperature, top-k, top-p

Day 1 always took the single highest-scoring token. Day 2 keeps the loop
identical and replaces **only the selection step**.

```
Day 1:   next_token = last.argmax(dim=-1)
Day 2:   probs      = softmax(last)
         next_token = multinomial(probs, 1)
```

That is the entire change. `input_ids`, `logits`, the append, the loop — all
unchanged.

---

## 1. Why softmax appears now

Day 1 needed no softmax, for a precise reason:

> **softmax is monotonic.** It maps larger inputs to larger outputs and cannot
> reorder anything, so it can never change an `argmax`.

`multinomial` is different: it doesn't want *order*, it wants *probabilities*.
It draws a random index where index `i` has chance `probs[i]`. So the
distribution must be normalised and non-negative first.

Measured on `"The capital of India is"` (Qwen2.5-0.5B), last position:

| | logits (raw) | after softmax |
|---|---|---|
| range | `-17.19 .. 15.32` | `0.000000 .. 0.125469` |
| sum | `-440138.2` | `1.000000` |
| meaning | "score" | "probability" |

```python
probs = torch.softmax(logits.float(), dim=-1)
```

Two details that are easy to get wrong:

- **`.float()`** — logits arrive as `float16`. `exp()` over 151,936 fp16 values
  loses precision badly, and `exp(15)` is already ~3.3M. Upcast to float32 first.
- **`dim=-1`** — normalise across the vocabulary, so each row sums to 1. Using
  the wrong dim would normalise across *positions*, which is meaningless.

### The sampling problem this exposes

The raw distribution is very peaky in the head and very fat in the tail:

```
#1   ' located'  p=0.12547   cumulative=0.12547
#2     ' known'  p=0.08038   cumulative=0.20585
#3        ' __'  p=0.07670   cumulative=0.28255
...
#8        '\n'   p=0.03484   cumulative=0.51072      <- top-8 = half the mass
remaining 151,928 tokens     0.48928                <- the other half
```

Sampling straight from this picks a bad token roughly half the time. The three
knobs below each restrict the distribution before sampling.

---

## 2. Temperature — reshape the distribution

```python
logits = logits / temperature        # BEFORE softmax
probs  = softmax(logits)
```

Divide the **logits**, not the probabilities. Because softmax exponentiates,
dividing logits by `T` becomes raising probabilities to the power `1/T` —
which is exactly what sharpens or flattens the shape.

Measured effect on the top-3 mass:

| temperature | top-1 p | top-3 sum | effect |
|---|---|---|---|
| `0.5` | 0.3623 | 0.6464 | sharper — more confident |
| `1.0` | 0.1255 | 0.2825 | unchanged |
| `2.0` | 0.0095 | 0.0245 | flatter — more random |

Invariants worth checking yourself:

- `T -> 0` makes the distribution a spike on the argmax ⇒ **sampling converges
  to greedy**. This is your correctness test (see §5).
- `T = 1` is a no-op.
- `T -> inf` approaches uniform ⇒ output becomes gibberish, and EOS becomes
  unlikely, so generation runs to the token limit.
- `T` is a **float > 0**. `T = 0` divides by zero; guard it or treat it as greedy.

---

## 3. Top-k — keep the k best

Keep the `k` highest logits, set everything else to `-inf`, then softmax. Since
`exp(-inf) = 0`, the discarded tokens get exactly zero probability, and softmax
renormalises the survivors to sum to 1 automatically.

```python
vals, idx = torch.topk(logits, k)
masked = torch.full_like(logits, float("-inf"))
masked.scatter_(-1, idx, vals)
probs = softmax(masked)
```

Measured with `k=5`: nonzero count exactly 5, sum exactly 1.000000.

Why this works: softmax is **shift-invariant** in a useful way — adding the same
constant to every logit changes nothing. Setting the tail to `-inf` is the limit
of "push the tail down by an enormous amount".

Note `-inf`, not `0`. Setting discarded logits to `0` would make them *high*
values (most logits here are negative), so they'd dominate.

---

## 4. Top-p (nucleus) — keep the smallest set reaching p

Top-k is a fixed budget. Top-p adapts to how confident the model is.

Sort descending, take the cumulative sum, and keep the shortest prefix whose
cumulative probability is `>= p`.

Measured on the same distribution:

| p | tokens kept | cumulative reached |
|---|---|---|
| `0.5` | 8 | 0.5107 |
| `0.9` | 131 | 0.9002 |
| `0.95` | 378 | 0.9500 |

That adaptivity is the point: when the model is certain, p=0.9 might keep 1
token; when it's unsure, it keeps hundreds.

**The off-by-one that everyone hits.** If `cum[t] >= p` first becomes true at
index `t`, then token `t` is the one that *crossed* the threshold — it must be
**included**. The standard trick is to shift the cumulative sum right by one
before comparing, so the mask selects tokens `0..t` rather than `0..t-1`:

```python
sorted_probs, sorted_idx = torch.sort(probs, descending=True)
cumulative = torch.cumsum(sorted_probs, dim=-1)
remove = cumulative - sorted_probs > p        # remove tokens strictly past the cut
sorted_probs[remove] = 0.0
probs = sorted_probs.scatter(-1, sorted_idx, sorted_probs)
probs = probs / probs.sum()                   # renormalise
```

Two other things to get right:

- **Renormalise.** After zeroing the tail, the survivors no longer sum to 1.
  `multinomial` requires a proper distribution.
- **`scatter` back.** You sorted the probabilities, so the indices no longer
  match the vocabulary. You must put them back before sampling, or you will
  sample an index that means a different token.
- **Always keep at least 1 token**, even if `p` is tiny.

---

## 5. How to test a randomised function

Greedy is reproducible; sampling is not. `torch.manual_seed(n)` makes it
repeatable, but that only proves determinism, not *correctness*.

The real test is the convergence property:

```
temperature -> 0   should reproduce Day 1 greedy EXACTLY, token for token
```

At `T -> 0` the distribution becomes a spike on the argmax, so any draw returns
the argmax. If your temperature handling is wrong (for instance dividing the
probabilities instead of the logits), this test fails. That is the same pattern
as `naive <-> KV-cached` on Day 3: keep a known-good reference and check the new
path against it.

Test it with a small-but-nonzero temperature: `T=0.01` is effectively greedy for
fp32, without the division-by-zero.

Second property worth testing: **`top_k=1` and `top_p -> 0` must also equal
greedy**, since both reduce the distribution to a single token.

---

## 6. Design questions — answer before coding

1. Temperature divides the logits. What would happen if you divided `probs`
   instead? (Try it — the effect is not the same, and in the wrong direction.)
2. `argmax` returns `[B]` directly usable as token ids. `multinomial(probs, 1)`
   returns `[B, 1]`. Does that still need `unsqueeze(-1)` before `cat`?
   Predict, then check.
3. `probs` is float32; `input_ids` is int64. Where exactly does the dtype
   conversion happen now, and why is it needed?
4. With `T` very large, why does generation tend to run to `max_new_tokens`
   instead of stopping at EOS?
5. Applying top-k **and** top-p together: does order matter? Reason about it,
   then test with `k=50, p=0.5` versus `p=0.5, k=50`.
6. In a batch, `multinomial` draws independently per row. What does that mean
   for two identical prompts in the same batch?

---

## 7. Order of operations

The conventional order, and why:

```
logits
  -> / temperature          (reshape the landscape)
  -> top-k mask             (restrict to k candidates)
  -> top-p mask             (restrict to the nucleus)
  -> softmax                (normalise what survived)
  -> multinomial            (draw one)
```

Top-k and top-p are often applied to **logits** with `-inf` masking, with a
single softmax afterwards — that is cheaper than softmaxing three times. Either
is fine as long as you are consistent and renormalise before sampling.

---

## 8. Where to look in the installed source

| what | where |
|---|---|
| greedy branch (`torch.argmax`) | `.venv/.../transformers/generation/utils.py:3073` |
| sampling branch (`multinomial`) | `.venv/.../transformers/generation/utils.py:3069-3072` |
| temperature / top-k / top-p implementations | `.venv/.../transformers/generation/logits_process.py` |

In `logits_process.py`, look for `TemperatureLogitsWarper`, `TopKLogitsWarper`
and `TopPLogitsWarper`. Read `TopPLogitsWarper` specifically — it is the
off-by-one from §4, written out. Note that in `transformers` 5.x these are
"processors/warpers" applied through a pipeline rather than the older
`generate(do_sample=True, top_k=..., top_p=...)` argument path.

---

## 9. Experiment record

**Question** — how do temperature, top-k and top-p change the diversity and
quality of generated text?

**Hypothesis** — (write before running)

**Variables** — fixed: model, prompt, `max_new_tokens`. Changed: `temperature`,
`top_k`, `top_p`.

**Measurements** — for each setting: the generated text, and something
quantitative. Useful candidates:

- number of distinct continuations from N samples of the same prompt
- how often generation reaches `max_new_tokens` instead of EOS
- mean log-probability of the chosen tokens (a proxy for "how surprised")

**Result** — (fill in)

**Explanation** — why the result occurred

---

## 10. Open questions

- (Your questions here.)
