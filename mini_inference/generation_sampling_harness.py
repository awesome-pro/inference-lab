# day07_sampling_harness.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_inference.sampling_generation import generate_sampled


MODEL_NAME = "Qwen/Qwen2.5-0.5B"
PROMPT = "The future of machine learning is"
MAX_NEW_TOKENS = 30

SEED = 42


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dtype = torch.float16 if device.type == "mps" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model


def run_ours(
    input_ids,
    model,
    tokenizer,
    temperature,
    top_k,
    top_p,
):
    torch.manual_seed(SEED)

    return generate_sampled(
        input_ids=input_ids.clone(),
        model=model,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )


def run_reference(
    input_ids,
    model,
    tokenizer,
    temperature,
    top_k,
    top_p,
):
    torch.manual_seed(SEED)

    kwargs = {
        "input_ids": input_ids.clone(),
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": True,
        "temperature": temperature,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        "use_cache": False,
    }

    if top_k is not None:
        kwargs["top_k"] = top_k

    if top_p is not None:
        kwargs["top_p"] = top_p

    return model.generate(**kwargs)


def compare_case(
    name,
    input_ids,
    model,
    tokenizer,
    temperature,
    top_k,
    top_p,
):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"temperature={temperature}, "
        f"top_k={top_k}, "
        f"top_p={top_p}"
    )

    try:
        ours = run_ours(
            input_ids,
            model,
            tokenizer,
            temperature,
            top_k,
            top_p,
        )

    except NotImplementedError as e:
        print("\n❌ Sampling implementation missing.")
        print(e)
        return False

    reference = run_reference(
        input_ids,
        model,
        tokenizer,
        temperature,
        top_k,
        top_p,
    )

    print("\nYour output:")
    print(
        tokenizer.decode(
            ours[0],
            skip_special_tokens=True,
        )
    )

    print("\nHF reference:")
    print(
        tokenizer.decode(
            reference[0],
            skip_special_tokens=True,
        )
    )

    prompt_len = input_ids.shape[1]

    prefix_correct = torch.equal(
        ours[:, :prompt_len],
        input_ids,
    )

    print(
        "\nPrompt preserved:",
        "✅" if prefix_correct else "❌",
    )

    grew = ours.shape[1] > input_ids.shape[1]

    print(
        "Generated tokens:",
        "✅" if grew else "❌",
    )

    exact_match = torch.equal(
        ours,
        reference,
    )

    print(
        "Exact token match:",
        "✅" if exact_match else "❌",
    )

    if not exact_match:
        print(
            "\nNote: sampled generation does not always need to "
            "match Hugging Face token-for-token, even with the same seed, "
            "if the implementation consumes RNG differently."
        )

    return True


def test_top_k_one_equals_greedy(
    input_ids,
    model,
    tokenizer,
):
    print("\n" + "=" * 70)
    print("PROPERTY TEST: top_k=1 should equal greedy")
    print("=" * 70)

    sampled = generate_sampled(
        input_ids=input_ids.clone(),
        model=model,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
        top_k=1,
        top_p=None,
        eos_token_id=tokenizer.eos_token_id,
    )

    greedy = model.generate(
        input_ids=input_ids.clone(),
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    match = torch.equal(sampled, greedy)

    print(
        "top_k=1 == greedy:",
        "✅" if match else "❌",
    )


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

    print("\nPrompt:")
    print(PROMPT)

    cases = [
        {
            "name": "Plain sampling",
            "temperature": 1.0,
            "top_k": None,
            "top_p": None,
        },
        {
            "name": "Low temperature",
            "temperature": 0.5,
            "top_k": None,
            "top_p": None,
        },
        {
            "name": "Top-k",
            "temperature": 1.0,
            "top_k": 20,
            "top_p": None,
        },
        {
            "name": "Top-p",
            "temperature": 1.0,
            "top_k": None,
            "top_p": 0.9,
        },
        {
            "name": "Temperature + top-k + top-p",
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
        },
    ]

    for case in cases:
        success = compare_case(
            name=case["name"],
            input_ids=input_ids,
            model=model,
            tokenizer=tokenizer,
            temperature=case["temperature"],
            top_k=case["top_k"],
            top_p=case["top_p"],
        )

        if not success:
            return

    test_top_k_one_equals_greedy(
        input_ids,
        model,
        tokenizer,
    )


if __name__ == "__main__":
    main()