"""Shared model capability checks for multimodal input."""

import re


def model_supports_audio_input(model_name: str) -> bool:
    """Return whether the model supports the implemented WAV input transport."""
    normalized = re.sub(r"[^a-z0-9]+", "", str(model_name or "").casefold())
    return normalized.startswith("gemma4")
