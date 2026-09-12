# Day 1 — the naive greedy generation loop

Milestone: own the inference loop. No `model.generate()` for the implementation;
it is allowed only as a correctness oracle.

Your hand-drawn diagram is structurally correct. The backward arrow from
`next_token_id` into `input_ids` is the whole mechanism.

---

## 1. The mental model

A causal LM does not write sentences. It answers one question, once:

> Given tokens `0..i`, what is the distribution over token `i+1`?

It answers that for **every position simultaneously** in one forward pass, and
then throws all but one answer away. Generation is the loop that feeds each
answer back in as the next question.

### Measured: `"The capital of France is"` (T=5, Qwen2.5-0.5B)

| pos | context | model top-1 | p | actual next prompt token | its rank | its p |
|----:|---------|------------|------:|--------------------------|---------:|------:|
| 0 | `'The'` | `' following'` | 0.685 | `' capital'` | 526 | 0.000 |
| 1 | `'The capital'` | `' of'` | 0.540 | `' of'` | 1 | 0.540 |
| 2 | `'The capital of'` | `' Ber'` | 0.188 | `' France'` | 4 | 0.042 |
| 3 | `'The capital of France'` | `' is'` | 0.721 | `' is'` | 1 | 0.721 |

Two lessons sit in that table:

1. **`The` → `capital` is rank 526.** The prompt was *given*, not predicted. At
   position 0 the model had no idea the sentence was about France.
2. **Teacher forcing vs generation.** During prompt processing the true token is
   always fed regardless of the model's guess. In your loop, the model's *own*
   guess is fed back, so mistakes compound. The final row shows why this works
   anyway: once the context is rich enough, the model is confident and correct.
   Day 1's whole bet is that greedy picks the right token once context is rich.

### Verified causality

```
torch.allclose(model(ids[:, :3]).logits[0, -1, :], model(ids).logits[0, 2, :])
  -> True, max|diff| = 0.0
```

Position `i`'s output depends only on tokens `0..i`. This is the property that
makes KV caching legal on Day 3 — and it is worth proving to yourself rather
than assuming.

---

## 2. Shape invariants

Measured on this model, B=1, T=5:

```
input_ids                          [1, 5]              int64
model(input_ids).logits            [1, 5, 151936]      float16
logits[:, -1, :]                   [1, 151936]         <-- the only slice needed
torch.argmax(logits[:, -1, :], -1) [1]                 int64  -> 12095 -> ' Paris'

next_token[:, None]                [1, 1]              int64
torch.cat([input_ids, that], -1)   [1, T_cur + 1]      int64
```

Then decode with `tokenizer.decode(ids[0])`.

### Config facts that come due later

```
vocab_size 151936   hidden_size 896    num_hidden_layers 24
num_attention_heads 14   num_key_value_heads 2   max_position_embeddings 32768
eos_token_id 151643   pad_token_id None   tie_word_embeddings True
```

- `num_key_value_heads=2` vs `num_attention_heads=14` -> GQA. A real KV cache is
  **7x smaller** than the naive `n_layers x n_heads` calculation.
- `pad_token_id is None` -> padding for batching (Day 5) needs an explicit decision.
- `tie_word_embeddings=True` -> `lm_head` is the transpose of the embedding table.

---

## 3. Why `input_ids` is not the only state

State that persists across iterations, Day 1 vs later:

| state | Day 1 | Day 3 | Day 4/5 |
|---|---|---|---|
| running token sequence | yes | yes | yes |
| stop flag (per batch row) | yes | yes | yes |
| generated-only tokens (for decoding text) | optional | yes | yes |
| `attention_mask` | not needed at B=1, no padding | needed | needed |
| `past_key_values` | no | yes | yes |

With one unpadded sequence you may omit `attention_mask` entirely; Hugging Face
treats it as all-ones. Send it explicitly the moment padding appears.

---

## 4. The cost you are about to build

For T prompt tokens and n generated tokens, this loop performs **n forward passes**
over lengths T, T+1, ..., T+n-1. Position 0 goes through all 24 layers n+1 times.

Attention work is quadratic in sequence length, so:

```
naive total   ~ sum_{L=T}^{T+n} L^2   ~= n * T^2
cached total  ~ T^2 (prefill) + n * T (decode)
ratio         ~= T
```

For T=1024, n=100 that is roughly **100x** more attention work than necessary,
and it grows with prompt length. Do not fix this yet. Build the correct version,
feel the slowness, and let that motivate Day 3.

Also note the logits waste, which is pure memory bandwidth:

| sequence length | `logits` size (fp16) |
|---|---:|
| 5 | ~1.5 MiB |
| 8192 | ~2.3 GiB |
| actually used | 297 KiB |

`transformers` already solves this: `Qwen2ForCausalLM.forward` accepts
`logits_to_keep` and does `self.lm_head(hidden_states[:, slice_indices, :])` at
`models/qwen2/modeling_qwen2.py:465`. Read that function — including the amusing
`slice(-logits_to_keep, None)`, where the default `0` yields `slice(0, None)`,
i.e. *everything*.

---

## 5. Design questions to answer BEFORE writing the body

Write your answers down. These decide whether Day 3 is a refactor or a rewrite.

1. Does `generate_greedy` return the **full sequence** or **only new tokens**?
   Which is easier to verify? Which is easier to batch?
2. Should the loop know about the tokenizer at all? (Consider: what makes a
   function testable against `model.generate`?)
3. The loop calls `model(input_ids)`. On Day 3 you will want
   `model(input_ids, past_key_values=...)` returning a *different* logits shape
   (length 1 instead of length T). If you keep this signature, what changes?
   Is there a function boundary you could introduce *then* rather than now?
4. Where does EOS stopping belong: inline in the loop, or as an object the loop
   queries? (Day 4. Naming it now is enough.)
5. Is there any state shared between two independent prompts decoded by this
   function? (Batching preview.)

Do not abstract prematurely. Answer, pick the simplest thing, move on.

---

## 6. TODO ladder

**Step 0.** Answer the prediction questions in section 8 first.
**Step 0b.** Run the microscope first (section 12) — watch one step happen before
you write a loop that does it n times.

**Step 1.** Inference must build no autograd graph. Decide how you enforce that
(decorator on the function, or a `with` block inside it) and note the tradeoff:
a `with` block must cover *every* model call.

**Step 2.** Loop exactly `max_new_tokens` times. `max_new_tokens` is an upper
bound, not a promise — EOS may end a row earlier.

**Step 3.** Call the model on the **current** sequence. What goes in, what comes
out, and does the output shape depend on the current length? (It does.)

**Step 4.** Select the distribution for the position *after the last real token*.
Remember `logits[b, i, :]` is the distribution over token `i+1`. You want the
token at index `T_cur`, so which `i` do you want? Express it as a negative index
and satisfy yourself that it is right for every iteration, not just the first.

**Step 5.** Collapse `[B, V] -> [B]`, int64. Convince yourself that no softmax is
needed for greedy: softmax is monotonic, so it cannot change an argmax. (It will
be needed on Day 2, when you sample instead.)

**Step 6.** Append. `next_token` is `[B]`; the sequence is `[B, T_cur]`. You need
`[B, 1]` and the **sequence** dimension. Predict what happens if you use `dim=0`
instead — then try it and confirm your prediction.

**Step 7.** Stopping. Outer bound is the loop count; EOS ends a row. At B=1 the
simplest correct thing is to break. Note what you would need for B>1 (a per-row
boolean mask) — do not build it yet.

**Step 8.** Return the sequence. Decoding tokens to text is the caller's job.

---

## 7. Correctness check

```
.venv/bin/python -m mini_inference.day01_greedy
```

The harness runs your loop and `model.generate(do_sample=False)` on the same
prompt and compares them token-for-token, reporting the first divergence and
classifying it:

- diverges at the **first generated token** -> bug in the logits slice / argmax
- diverges **later** -> first token was right, so the bug is state update across
  iterations
- **length mismatch only** -> stopping/EOS difference, not a logits bug

This is the pattern to keep for the whole milestone: keep a slow reference and
check the fast version against it (`naive <-> cached`, later).

---

## 8. Prediction questions — answer before you run

1. What text will `generate_greedy("The capital of France is", max_new_tokens=16)`
   produce? Write your guess down. Then compare. (The model's last-position top-1
   is `' Paris'` at logit 17.86.)
2. How many forward passes for n new tokens? How many times is position 0 pushed
   through the network?
3. For T prompt tokens and n generated tokens, total attention work is
   proportional to what? Do this before reading section 4.
4. `logits` has shape `[1, T, 151936]` and you use 1/T of it. Build a `[1, 8192, :]`
   logits tensor and measure its memory. What would you change, and what does
   `logits_to_keep` do?
5. Two prompts decoded in sequence by this function: is any state shared? What
   would have to change to decode both in one loop?

---

## 9. Experiment record

Fill this in as you go.

**Question** — does a hand-written greedy loop reproduce `model.generate(do_sample=False)` exactly?

**Hypothesis** — (write before running)

**Variables** — fixed: model, prompt, `max_new_tokens`. Changed: nothing yet.

**Measurements** — tokens/sec, ms/token, first-token agreement, full agreement

**Result** — (fill in)

**Explanation** — why the result occurred

---

## 10. Open questions

- (Your questions here — the point is to record them, not to answer them all today.)

---

## 11. Where to look in the installed source

Read these *after* your loop passes, to compare design choices:

| what | where |
|---|---|
| greedy argmax + append | `.venv/.../transformers/generation/utils.py:3073-3076` |
| only compute last-position logits | `.venv/.../transformers/models/qwen2/modeling_qwen2.py:465` |
| the LM head | `.venv/.../transformers/models/qwen2/modeling_qwen2.py:416` |

Read `logits_to_keep` and the `torch.cat([input_ids, next_tokens[:, None]], dim=-1)`
line specifically. They are the two lines your diagram already contains.

---

## 12. The microscope — inspecting the data flow

Two tools, both support code that contains no inference logic:

### `experiments/day01_inspect_step.py` — one step, by hand, no loop

```
.venv/bin/python experiments/day01_inspect_step.py
```

Unrolls a single decode step and prints every tensor's shape, dtype, device and
memory footprint as it is produced, then does a second step to show the state
grow. Each printed tensor is the output of the block above it, so reading top to
bottom *is* the data flow.

It ends by measuring the thing that motivates Day 3: on the second pass,
`logits_2[:, :5, :]` is `torch.equal` to `logits` from the first pass
(`max|diff| = 0.0`). All five positions were recomputed bit-identically and then
discarded. That is the waste a KV cache removes — measured, not read.

### `mini_inference/trace.py` — the same view, inside *your* loop

```python
from mini_inference.trace import Trace

trace = Trace(tokenizer, top_k=5)
trace.header()

for step in range(max_new_tokens):
    trace.observe(step, input_ids=input_ids)

    logits = model(input_ids).logits
    trace.observe(step, logits=logits, last_logits=logits[:, -1, :])

    next_token = torch.argmax(logits[:, -1, :], dim=-1)
    trace.observe(step, next_token=next_token)

    input_ids = torch.cat([input_ids, next_token[:, None]], dim=-1)
    trace.observe(step, new_ids=input_ids)

trace.summary()
```

Every argument is optional, so call it with only the tensors that exist at that
point. `summary()` reports total logits bytes computed and total positions put
through the LM head — watch those grow super-linearly as T_cur grows.

`Trace(..., enabled=False)` turns it off without deleting the calls, so the same
file can be both inspected and timed.

### Worth doing while you're in there

- Print `input_ids.device` at the top of the loop. Tokenizer output is on CPU;
  the model is on `mps:0`. Mixing them in a `torch.cat` is the classic failure.
- Print `logits.dtype`. It is `float16`. `torch.argmax` is fine on fp16, but any
  softmax you write from Day 2 onward should upcast to `float32` first.
- Change `dim=-1` to `dim=0` in the append and predict the error before running it.
- Print `torch.isfinite(logits).all()` once. Knowing your logits are healthy
  separates "model is broken" from "my indexing is broken".
