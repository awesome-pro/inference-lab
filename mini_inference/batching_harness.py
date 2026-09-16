import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mini_inference.batched_generate import generate_batched


MODEL_NAME = "Qwen/Qwen2.5-0.5B"

PROMPTS = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "Machine learning",
]

# Chat-formatted prompts: Qwen emits its EOS token (<|im_end|>) after only a
# few tokens here, so the early-stop / length-accounting path really runs
# instead of every row using all MAX_NEW_TOKENS slots.
STOP_PROMPTS = [
    "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\nName one color.<|im_end|>\n<|im_start|>assistant\n",
    "<|im_start|>user\n2 + 2 = ?<|im_end|>\n<|im_start|>assistant\n",
]

MAX_NEW_TOKENS = 20

SEED = 42

# temperature == 0.0 means greedy; anything else samples.
CASES = [
    {
        "name": "Greedy (temperature=0)",
        "temperature": 0.0,
        "top_k": None,
        "top_p": None,
    },
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
    {
        "name": "top_k=1 (must equal greedy)",
        "temperature": 1.0,
        "top_k": 1,
        "top_p": None,
    },
]


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


def prepare_batch(
    prompts: list[str],
    tokenizer,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokenized = [
        tokenizer.encode(prompt)
        for prompt in prompts
    ]

    max_len = max(len(ids) for ids in tokenized)
    batch_size = len(tokenized)

    input_ids = torch.full((batch_size, max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for row, ids in enumerate(tokenized):
        seq_len = len(ids)
        ids_tensor = torch.tensor(
            ids,
            dtype=torch.long,
            device=device
        )

        input_ids[row, -seq_len:] = ids_tensor
        attention_mask[row, -seq_len:] = 1

    return input_ids, attention_mask


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


def strip_special_tail(tokens, pad_token_id, eos_token_id):
    """
    HARNESS plumbing: HF's generate() hands back a flat sequence and no length
    information, so for the *reference* we have to infer where the useful
    tokens end. Our generator does not need to guess -- it returns
    generated_lengths directly.
    """

    clean = []

    for token in tokens:

        if token == pad_token_id:
            break

        if eos_token_id is not None and token == eos_token_id:
            break

        clean.append(token)

    return clean


def extract_valid_request(row, original_mask, generated_length):
    """
    Turn one rectangular row into exactly the tokens the user should see.

    generated_length counts only tokens the generator considered valid, so
    slicing that many tokens drops the EOS *and* the post-stop PAD slots
    without scanning for them.
    """

    prompt_width = original_mask.shape[0]

    prompt = row[:prompt_width][original_mask.bool()]
    valid_generated = row[prompt_width:prompt_width + int(generated_length)]

    return torch.cat([prompt, valid_generated], dim=0)


def structural_checks(
    row,
    prompt_row,
    original_mask,
    generated_length,
    generated_width,
    pad_token_id,
    eos_token_id,
):
    """
    Per-request invariants of the (output_ids, generated_lengths) contract.

    Returns a list of (label, ok) pairs.
    """

    prompt_width = original_mask.shape[0]
    length = int(generated_length)

    valid = row[prompt_width:prompt_width + length]

    checks = [
        (
            f"generated_length {length} in [0, {MAX_NEW_TOKENS}]",
            0 <= length <= MAX_NEW_TOKENS,
        ),
        (
            "prompt preserved",
            bool(torch.equal(row[:prompt_width], prompt_row[:prompt_width])),
        ),
        (
            "no PAD inside valid tokens",
            bool((valid != pad_token_id).all()),
        ),
    ]

    if eos_token_id is not None:

        checks.append(
            (
                "no EOS inside valid tokens",
                bool((valid != eos_token_id).all()),
            )
        )

        # A row that stopped early must still have its EOS stored in the
        # rectangular tensor, right after the valid region.
        if length < generated_width:

            checks.append(
                (
                    "EOS stored right after valid tokens",
                    int(row[prompt_width + length]) == eos_token_id,
                )
            )

    return checks


def run_batched_case(
    input_ids,
    attention_mask,
    model,
    tokenizer,
    case,
    seed=SEED,
):
    torch.manual_seed(seed)

    return generate_batched(
        input_ids=input_ids.clone(),
        attention_mask=attention_mask.clone(),
        model=model,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        temperature=case["temperature"],
        top_k=case["top_k"],
        top_p=case["top_p"],
    )


def run_sampling_cases(input_ids, attention_mask, model, tokenizer, results):
    """
    One batched run per sampling configuration, then check the same things the
    single-prompt sampling harness checks, plus the batched-only invariants:

      * prompt preserved
      * generated_lengths exactly describes the valid tokens
      * EOS and PAD never leak into what the user is shown
      * temperature=0.0 == independent HF greedy generation
      * top_k=1 == greedy
      * a sampled run is reproducible under the same seed
    """

    prompt_width = input_ids.shape[1]

    greedy_output = None
    greedy_lengths = None

    for case in CASES:

        print("\n" + "=" * 70)
        print(f"CASE: {case['name']}")
        print(
            f"temperature={case['temperature']}, "
            f"top_k={case['top_k']}, "
            f"top_p={case['top_p']}"
        )
        print("=" * 70)

        output, generated_lengths = run_batched_case(
            input_ids,
            attention_mask,
            model,
            tokenizer,
            case,
        )

        generated_width = output.shape[1] - prompt_width

        print("\nOutput shape:", tuple(output.shape))
        print("Generated columns:", generated_width)
        print("generated_lengths:", generated_lengths.tolist())

        # ---------------------------------------------
        # Validity of the length contract, row by row
        # ---------------------------------------------

        structural_ok = True

        for i, prompt in enumerate(PROMPTS):

            checks = structural_checks(
                row=output[i],
                prompt_row=input_ids[i],
                original_mask=attention_mask[i],
                generated_length=generated_lengths[i],
                generated_width=generated_width,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            for label, ok in checks:

                if not ok:
                    structural_ok = False
                    print(f"  ❌ Row {i}: {label}")

        results.append(
            (
                f"{case['name']}: generated_lengths contract",
                structural_ok,
            )
        )

        print(
            "Valid tokens are EOS/PAD free:",
            "✅" if structural_ok else "❌",
        )

        # ---------------------------------------------
        # What the user would actually see
        # ---------------------------------------------

        print("\nUser-visible output:")

        logical_outputs = []

        for i, prompt in enumerate(PROMPTS):

            logical = extract_valid_request(
                row=output[i],
                original_mask=attention_mask[i],
                generated_length=generated_lengths[i],
            )

            logical_outputs.append(logical)

            prompt_len = int(attention_mask[i].sum())
            new_tokens = logical[prompt_len:]

            print(f"\n  Request {i} (valid tokens: {new_tokens.shape[0]})")
            print(f"  prompt:    {prompt!r}")
            print(
                "  generated:",
                repr(
                    tokenizer.decode(
                        new_tokens,
                        skip_special_tokens=False,
                    )
                ),
            )

        # ---------------------------------------------
        # Greedy must match independent generation
        # ---------------------------------------------

        if case["temperature"] == 0.0:

            greedy_output = output
            greedy_lengths = generated_lengths

            print("\n  BATCH vs INDEPENDENT REQUESTS (greedy)")

            greedy_ok = True

            for i, prompt in enumerate(PROMPTS):

                reference = generate_single_reference(
                    prompt=prompt,
                    tokenizer=tokenizer,
                    model=model,
                    device=input_ids.device,
                )

                # The independent reference has no padding, so its prompt is
                # exactly the real prompt tokens of row i.
                prompt_ids = input_ids[i][attention_mask[i].bool()].tolist()
                ref_prompt_len = len(tokenizer.encode(prompt))

                if reference[:ref_prompt_len].tolist() != prompt_ids:
                    print(f"    Request {i}: ⚠️ tokenization differs from reference")

                ref_ids = prompt_ids + strip_special_tail(
                    reference[ref_prompt_len:].tolist(),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                ours_ids = logical_outputs[i].tolist()

                match = ours_ids == ref_ids

                greedy_ok = greedy_ok and match

                print(
                    f"    Request {i}:",
                    "✅" if match else "❌",
                )

                if not match:

                    min_len = min(len(ours_ids), len(ref_ids))

                    for j in range(min_len):
                        if ours_ids[j] != ref_ids[j]:
                            print(f"      first divergence at token {j}")
                            print(
                                "      batched:",
                                ours_ids[j],
                                repr(tokenizer.decode([ours_ids[j]])),
                            )
                            print(
                                "      single: ",
                                ref_ids[j],
                                repr(tokenizer.decode([ref_ids[j]])),
                            )
                            break

                    print(
                        f"      lengths: batched={len(ours_ids)}, "
                        f"single={len(ref_ids)}"
                    )

            results.append(
                (
                    "greedy batch == independent greedy",
                    greedy_ok,
                )
            )

        # ---------------------------------------------
        # top_k=1 is greedy under the hood
        # ---------------------------------------------

        if case["top_k"] == 1 and greedy_output is not None:

            top_k_one_ok = torch.equal(output, greedy_output) and torch.equal(
                generated_lengths,
                greedy_lengths,
            )

            results.append(
                (
                    "top_k=1 sampling == greedy",
                    top_k_one_ok,
                )
            )

            print(
                "\n  top_k=1 == greedy:",
                "✅" if top_k_one_ok else "❌",
            )

        # ---------------------------------------------
        # Same seed -> same tokens (does the RNG actually drive this?)
        # ---------------------------------------------

        if case["temperature"] > 0.0:

            repeat_output, repeat_lengths = run_batched_case(
                input_ids,
                attention_mask,
                model,
                tokenizer,
                case,
            )

            reproducible = torch.equal(output, repeat_output) and torch.equal(
                generated_lengths,
                repeat_lengths,
            )

            results.append(
                (
                    f"{case['name']}: reproducible with same seed",
                    reproducible,
                )
            )

            print(
                "\n  Same seed reproduces output:",
                "✅" if reproducible else "❌",
            )


def run_early_stop_case(input_ids, attention_mask, model, tokenizer, results):
    """
    Force the EOS path to actually run.

    With chat-formatted prompts Qwen stops on its own, so at least one row
    should report generated_length < generated columns -- which is exactly the
    case where dropping EOS/PAD matters.
    """

    print("\n" + "=" * 70)
    print("EARLY STOP: EOS / PAD DROPPING")
    print("=" * 70)

    prompt_width = input_ids.shape[1]

    output, generated_lengths = run_batched_case(
        input_ids,
        attention_mask,
        model,
        tokenizer,
        case=CASES[0],
    )

    generated_width = output.shape[1] - prompt_width

    print("Generated columns:", generated_width)
    print("generated_lengths:", generated_lengths.tolist())

    stopped_rows = 0
    contract_ok = True

    for i in range(input_ids.shape[0]):

        length = int(generated_lengths[i])

        checks = structural_checks(
            row=output[i],
            prompt_row=input_ids[i],
            original_mask=attention_mask[i],
            generated_length=generated_lengths[i],
            generated_width=generated_width,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        for label, ok in checks:
            if not ok:
                contract_ok = False
                print(f"  ❌ Row {i}: {label}")

        if length < generated_width:
            stopped_rows += 1

        logical = extract_valid_request(
            row=output[i],
            original_mask=attention_mask[i],
            generated_length=generated_lengths[i],
        )

        raw = output[i, prompt_width:].tolist()

        print(f"\n  Request {i}")
        print(f"    raw generated slots: {raw}")
        print(f"    valid length:        {length}")
        print(
            "    shown to user:      ",
            repr(
                tokenizer.decode(
                    logical[int(attention_mask[i].sum()):],
                    skip_special_tokens=False,
                )
            ),
        )

    results.append(
        (
            "early-stop rows: length contract",
            contract_ok,
        )
    )

    results.append(
        (
            "early-stop actually happened",
            stopped_rows > 0,
        )
    )

    print(
        f"\nRows that stopped before {MAX_NEW_TOKENS} tokens:",
        f"{stopped_rows}/{input_ids.shape[0]}",
        "✅" if stopped_rows > 0 else "❌",
    )


def print_summary(results):

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for label, ok in results:
        print(f"  {'✅' if ok else '❌'} {label}")

    print()

    if all(ok for _, ok in results):
        print("🎉 All batched generation checks passed.")
    else:
        print("❌ Some batched generation checks failed.")


def main():
    device = get_device()

    print(f"Device: {device}")
    print(f"Model:  {MODEL_NAME}")

    tokenizer, model = load_model(device)

    # -----------------------------------------------------
    # TOKENIZE AS ONE BATCH
    # -----------------------------------------------------

    input_ids, attention_mask = prepare_batch(prompts=PROMPTS, tokenizer=tokenizer, device=device)

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
    # GREEDY + SAMPLING
    # -----------------------------------------------------

    results = []

    try:

        print("\n" + "=" * 70)
        print("BATCHED GENERATION (GREEDY + SAMPLING)")
        print("=" * 70)
        print("\nPrompts:")
        for i, prompt in enumerate(PROMPTS):
            print(f"  {i}: {prompt!r}")

        run_sampling_cases(
            input_ids,
            attention_mask,
            model,
            tokenizer,
            results,
        )

        # -------------------------------------------------
        # EOS / early stop
        # -------------------------------------------------

        stop_input_ids, stop_attention_mask = prepare_batch(
            prompts=STOP_PROMPTS,
            tokenizer=tokenizer,
            device=device,
        )

        run_early_stop_case(
            stop_input_ids,
            stop_attention_mask,
            model,
            tokenizer,
            results,
        )

    except NotImplementedError as e:

        print("\n❌ Batched generation isn't implemented yet.")
        print(e)
        return

    print_summary(results)


if __name__ == "__main__":
    main()
