import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_inference.cached_generate import generate_cached


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "The capital of United States is"
MAX_NEW_TOKENS = 32


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dtype = (
        torch.float16
        if device.type == "mps"
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model


# ---------------------------------------------------------
# Slow Day-1 reference
# ---------------------------------------------------------

@torch.no_grad()
def generate_naive(
    input_ids,
    model,
    max_new_tokens,
    eos_token_id=None,
):
    for _ in range(max_new_tokens):

        logits = model(
            input_ids,
            use_cache=False,
        ).logits

        next_token = logits[:, -1, :].argmax(
            dim=-1
        )

        input_ids = torch.cat(
            [
                input_ids,
                next_token.unsqueeze(-1),
            ],
            dim=-1,
        )

        # B=1 only
        if (
            eos_token_id is not None
            and next_token.item() == eos_token_id
        ):
            break

    return input_ids


def time_generation(fn, device):
    synchronize(device)

    start = time.perf_counter()

    output = fn()

    synchronize(device)

    elapsed = time.perf_counter() - start

    return output, elapsed


def main():
    device = get_device()

    print(f"Device: {device}")
    print(f"Model:  {MODEL_NAME}")

    tokenizer, model = load_model(device)

    encoded = tokenizer(
        PROMPT,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)

    prompt_len = input_ids.shape[1]

    print("\nPrompt:")
    print(PROMPT)

    print("\nPrompt tokens:")
    print(prompt_len)

    print("\nInput shape:")
    print(input_ids.shape)

    # -----------------------------------------------------
    # Run your KV implementation
    # -----------------------------------------------------

    print("\nRunning your KV-cached implementation...")

    try:

        cached, cached_time = time_generation(
            lambda: generate_cached(
                input_ids=input_ids.clone(),
                model=model,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
            ),
            device,
        )

    except NotImplementedError as e:

        print("\n❌ KV generation not implemented.")
        print(e)

        return

    # -----------------------------------------------------
    # Run slow reference
    # -----------------------------------------------------

    print("\nRunning naive reference...")

    naive, naive_time = time_generation(
        lambda: generate_naive(
            input_ids=input_ids.clone(),
            model=model,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
        ),
        device,
    )

    # -----------------------------------------------------
    # Decode
    # -----------------------------------------------------

    print("\nCached output:")
    print(
        tokenizer.decode(
            cached[0],
            skip_special_tokens=True,
        )
    )

    print("\nNaive output:")
    print(
        tokenizer.decode(
            naive[0],
            skip_special_tokens=True,
        )
    )

    # -----------------------------------------------------
    # Correctness tests
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("CORRECTNESS")
    print("=" * 70)

    prefix_correct = torch.equal(
        cached[:, :prompt_len],
        input_ids,
    )

    print(
        "Prompt preserved:",
        "✅" if prefix_correct else "❌",
    )

    exact_match = torch.equal(
        cached,
        naive,
    )

    print(
        "Cached == naive:",
        "✅" if exact_match else "❌",
    )

    # -----------------------------------------------------
    # Find first divergence
    # -----------------------------------------------------

    if not exact_match:

        print("\nFirst divergence:")

        min_len = min(
            cached.shape[1],
            naive.shape[1],
        )

        found = False

        for i in range(min_len):

            if cached[0, i] != naive[0, i]:

                found = True

                cached_id = cached[0, i].item()
                naive_id = naive[0, i].item()

                print(f"Position: {i}")

                print(
                    "Cached:",
                    cached_id,
                    repr(
                        tokenizer.decode(
                            [cached_id]
                        )
                    ),
                )

                print(
                    "Naive:",
                    naive_id,
                    repr(
                        tokenizer.decode(
                            [naive_id]
                        )
                    ),
                )

                break

        if not found and cached.shape != naive.shape:

            print(
                "No token divergence in common prefix."
            )

            print(
                f"Cached length={cached.shape[1]}"
            )

            print(
                f"Naive length={naive.shape[1]}"
            )

    # -----------------------------------------------------
    # Timing
    # -----------------------------------------------------

    generated = cached.shape[1] - prompt_len

    print("\n" + "=" * 70)
    print("TIMING")
    print("=" * 70)

    print(
        f"Generated tokens: {generated}"
    )

    print(
        f"Naive:  {naive_time:.4f}s"
    )

    print(
        f"Cached: {cached_time:.4f}s"
    )

    if cached_time > 0:
        print(
            f"Speed ratio: "
            f"{naive_time / cached_time:.2f}x"
        )

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------

    print("\n" + "=" * 70)

    if exact_match:

        print(
            "🎉 KV-cached generation matches "
            "the naive reference."
        )

    else:

        print(
            "❌ Cached generation differs from "
            "the naive reference."
        )

        print(
            "Debug the first divergent token."
        )


if __name__ == "__main__":
    main()