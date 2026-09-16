import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_inference.batched_generate import generate_batched


MODEL_NAME = "Qwen/Qwen2.5-0.5B"

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Machine learning",
]

MAX_NEW_TOKENS = 20


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Decoder-only batched generation is easiest to reason
    # about using LEFT padding.
    tokenizer.padding_side = "left"

    # Qwen may not define a separate pad token.
    # Using EOS as the physical PAD value is okay because
    # attention_mask distinguishes padding from real tokens.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = (
        torch.float16
        if device.type == "mps"
        else torch.float32
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    )

    model = model.to(device)
    model.eval()

    return tokenizer, model


def print_batch(tokenizer, input_ids, attention_mask):
    print("\n" + "=" * 70)
    print("PADDED BATCH")
    print("=" * 70)

    for i in range(input_ids.shape[0]):

        ids = input_ids[i].tolist()
        mask = attention_mask[i].tolist()

        tokens = [
            tokenizer.decode([token_id])
            for token_id in ids
        ]

        print(f"\nRow {i}")
        print("ids:   ", ids)
        print("mask:  ", mask)
        print("tokens:", [repr(x) for x in tokens])


@torch.no_grad()
def generate_single_reference(
    prompt,
    tokenizer,
    model,
    device,
):
    """
    Reference only.

    Generate each prompt independently with HF so we can ask:

        batched result for row i
        ==
        independent result for request i ?

    That is the important batching correctness property.
    """

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    return model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0]


def remove_left_padding_and_generated_padding(
    row,
    original_mask,
    pad_token_id,
    eos_token_id,
):
    """
    Convert a rectangular batched result into the logical
    sequence for one request.

    This is HARNESS plumbing, not the generation algorithm.
    """

    prompt_width = original_mask.shape[0]

    # Real prompt tokens only.
    prompt = row[:prompt_width][original_mask.bool()]

    generated = row[prompt_width:]

    clean_generated = []

    for token in generated.tolist():

        # Finished rows may accumulate physical PAD slots.
        if token == pad_token_id:
            break

        clean_generated.append(token)

        if eos_token_id is not None and token == eos_token_id:
            break

    if clean_generated:
        generated_tensor = torch.tensor(
            clean_generated,
            dtype=row.dtype,
            device=row.device,
        )

        return torch.cat(
            [prompt, generated_tensor],
            dim=0,
        )

    return prompt


def main():
    device = get_device()

    print(f"Device: {device}")
    print(f"Model:  {MODEL_NAME}")

    tokenizer, model = load_model(device)

    # -----------------------------------------------------
    # TOKENIZE AS ONE BATCH
    # -----------------------------------------------------

    encoded = tokenizer(
        PROMPTS,
        return_tensors="pt",
        padding=True,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print("\nRaw shapes:")

    print(
        "input_ids:     ",
        tuple(input_ids.shape),
    )

    print(
        "attention_mask:",
        tuple(attention_mask.shape),
    )

    print_batch(
        tokenizer,
        input_ids,
        attention_mask,
    )

    # -----------------------------------------------------
    # BASIC PADDING INVARIANTS
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("PADDING TESTS")
    print("=" * 70)

    assert input_ids.ndim == 2
    assert attention_mask.shape == input_ids.shape

    print("Rectangular [B,T] input: ✅")
    print("Mask shape matches input: ✅")

    lengths = attention_mask.sum(dim=-1)

    print("\nActual prompt lengths:")
    print(lengths.tolist())

    # Because we're left padded, every prompt's final column
    # should be a REAL token.
    last_positions_are_real = bool(
        (attention_mask[:, -1] == 1).all()
    )

    print(
        "Last position real for every row:",
        "✅" if last_positions_are_real else "❌",
    )

    # -----------------------------------------------------
    # YOUR IMPLEMENTATION
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("RUNNING YOUR STATIC BATCH")
    print("=" * 70)

    try:

        output = generate_batched(
            input_ids=input_ids.clone(),
            attention_mask=attention_mask.clone(),
            model=model,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    except NotImplementedError as e:

        print("\n❌ Batched generation isn't implemented yet.")
        print(e)
        return

    print("\nOutput shape:")
    print(tuple(output.shape))

    # -----------------------------------------------------
    # SHOW EACH REQUEST
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("BATCHED OUTPUTS")
    print("=" * 70)

    logical_outputs = []

    for i, prompt in enumerate(PROMPTS):

        logical = remove_left_padding_and_generated_padding(
            row=output[i],
            original_mask=attention_mask[i],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        logical_outputs.append(logical)

        print(f"\nRequest {i}")
        print("Prompt:")
        print(prompt)

        print("\nGenerated:")
        print(
            tokenizer.decode(
                logical,
                skip_special_tokens=True,
            )
        )

    # -----------------------------------------------------
    # INDEPENDENT B=1 REFERENCES
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("BATCH vs INDEPENDENT REQUESTS")
    print("=" * 70)

    all_match = True

    for i, prompt in enumerate(PROMPTS):

        reference = generate_single_reference(
            prompt=prompt,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        ours = logical_outputs[i]

        match = torch.equal(
            ours,
            reference,
        )

        all_match = all_match and match

        print(
            f"Request {i}:",
            "✅" if match else "❌",
        )

        if not match:

            min_len = min(
                ours.shape[0],
                reference.shape[0],
            )

            for j in range(min_len):

                if ours[j] != reference[j]:

                    print(
                        f"  first divergence at token {j}"
                    )

                    print(
                        "  batched:",
                        ours[j].item(),
                        repr(
                            tokenizer.decode(
                                [ours[j].item()]
                            )
                        ),
                    )

                    print(
                        "  single:",
                        reference[j].item(),
                        repr(
                            tokenizer.decode(
                                [reference[j].item()]
                            )
                        ),
                    )

                    break

            print(
                f"  lengths: batched={len(ours)}, "
                f"single={len(reference)}"
            )

    print("\n" + "=" * 70)

    if all_match:

        print(
            "🎉 Every batched request matches "
            "its independent B=1 generation."
        )

    else:

        print(
            "❌ At least one batched request differs "
            "from independent generation."
        )


if __name__ == "__main__":
    main()