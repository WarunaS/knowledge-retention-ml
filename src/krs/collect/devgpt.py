"""DevGPT loader (Xiao et al., 2024). Keep repository language/topics in metadata as a weak domain signal."""

from __future__ import annotations


def load(snapshot_dir: str):
    """Yield Conversation records from the DevGPT snapshot JSON files."""
    raise NotImplementedError
