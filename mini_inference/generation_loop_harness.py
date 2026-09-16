import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_inference.generation import generate_greedy


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "The capital of United States is"
MAX_NEW_TOKENS = 16


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dtype = torch.float16 if device.type == "mps" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model


def main():
    device = get_device()

    print(f"Device: {device}")
    print(f"Model:  {MODEL_NAME}")
    print()

    tokenizer, model = load_model(device)

    encoded = tokenizer(
        PROMPT,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)

    print("Prompt:")
    print(PROMPT)

    print("\nInput shape:")
    print(input_ids.shape)

    print("\nRunning your generation loop...")

    try:
        with torch.no_grad():
            ours = generate_greedy(
                input_ids=input_ids.clone(),
                model=model,
                max_new_tokens=MAX_NEW_TOKENS,
                eos_token_id=tokenizer.eos_token_id,
            )

    except NotImplementedError as e:
        print("\n❌ Generation loop is not implemented yet.")
        print(e)
        return

    print("\nYour output:")
    print(tokenizer.decode(ours[0], skip_special_tokens=True))

    print("\nYour output shape:")
    print(ours.shape)

    # ---------------------------------------------------------
    # Reference implementation
    # ---------------------------------------------------------

    print("\nRunning Hugging Face reference...")

    with torch.no_grad():
        reference = model.generate(
            input_ids=input_ids.clone(),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("\nReference output:")
    print(tokenizer.decode(reference[0], skip_special_tokens=True))

    # ---------------------------------------------------------
    # Tests
    # ---------------------------------------------------------

    print("\n--- TESTS ---")

    # 1. Prompt must remain unchanged
    prompt_len = input_ids.shape[1]

    prefix_correct = torch.equal(
        ours[:, :prompt_len],
        input_ids,
    )

    print(
        "Prompt preserved:",
        "✅" if prefix_correct else "❌",
    )

    # 2. Output should have added tokens
    grew = ours.shape[1] > input_ids.shape[1]

    print(
        "Generated tokens:",
        "✅" if grew else "❌",
    )

    # 3. Compare with HF greedy reference
    exact_match = torch.equal(ours, reference)

    print(
        "Matches HF greedy:",
        "✅" if exact_match else "❌",
    )

    if not exact_match:
        print("\nFirst divergence:")

        min_len = min(
            ours.shape[1],
            reference.shape[1],
        )

        for i in range(min_len):
            if ours[0, i] != reference[0, i]:
                print(f"Position: {i}")

                print(
                    "Yours:",
                    ours[0, i].item(),
                    repr(tokenizer.decode([ours[0, i].item()])),
                )

                print(
                    "HF:",
                    reference[0, i].item(),
                    repr(tokenizer.decode([reference[0, i].item()])),
                )

                break

        if ours.shape[1] != reference.shape[1]:
            print(
                f"Length mismatch: "
                f"yours={ours.shape[1]}, "
                f"reference={reference.shape[1]}"
            )

    print("\n----------------")

    if exact_match:
        print("\n🎉 Your greedy generation loop is correct.")
    else:
        print(
            "\nYour loop runs, but differs from the reference. "
            "Debug the first divergence."
        )


if __name__ == "__main__":
    main()