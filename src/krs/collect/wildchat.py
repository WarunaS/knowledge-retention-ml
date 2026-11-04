"""WildChat-1M loader (Zhao et al., 2024). Stream from Hugging Face; keep language/toxic metadata."""

from __future__ import annotations


def load(limit: int | None = None, streaming: bool = True):
    """Yield Conversation records (modality=conversational). Do not filter here."""
    raise NotImplementedError
