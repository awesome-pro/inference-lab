"""Shared model loading for experiments and tests.

Not learning-critical: this exists purely so every script and test loads
Qwen2.5-0.5B the same way, with the same padding settings, instead of repeating
five lines in every file.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

MODEL_NAME = "Qwen/Qwen2.5-0.5B"


def load_env() -> None:
    """Read ``.env`` at the repo root and set any keys not already in os.environ.

    ``.env`` is only a text file -- nothing reads it automatically. Without
    this, ``HF_TOKEN`` sits in the file while ``os.environ`` has no idea, and
    every HF Hub request goes out unauthenticated.

    Keys already present in the real environment win, so
    ``HF_TOKEN=... uv run ...`` still overrides the file.
    """
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and value:
            os.environ.setdefault(key, value)


load_env()


def load_model(device: str | None = None):
    """Return ``(tokenizer, model)`` ready for generation.

    The tokenizer is configured with LEFT padding and ``pad_token = eos_token``:

    * left padding -- decoder-only models want real tokens flush against the
      end, so every row's last real position is the final column.
    * ``pad_token = eos_token`` -- Qwen2.5-0.5B ships ``pad_token_id = None``,
      and batching needs a filler token. Reusing EOS is the standard stand-in.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
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


def encode(tokenizer, model, prompt: str) -> torch.Tensor:
    """Tokenize one prompt straight onto the model's device."""
    return tokenizer(prompt, return_tensors="pt")["input_ids"].to(model.device)
