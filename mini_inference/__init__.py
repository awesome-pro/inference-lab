"""mini_inference — hand-written inference loop, built from first principles."""

from __future__ import annotations

import os
from pathlib import Path


def _load_env() -> None:
    """Set any keys from the repo-root ``.env`` that are not already in os.environ.

    ``.env`` is only a text file -- nothing reads it automatically. Without
    this, ``HF_TOKEN`` sits in the file while ``os.environ`` has no idea, and
    every HF Hub request goes out unauthenticated.

    Real environment variables win, so ``HF_TOKEN=... uv run ...`` overrides.
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


_load_env()
