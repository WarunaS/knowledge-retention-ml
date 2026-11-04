"""Consented own-log loader (Claude Code / Cursor / Copilot). TRANSFER TEST ONLY - never used for training."""

from __future__ import annotations


def load(log_dir: str, consent_ids: set[str]):
    """Yield Conversation records only for members with written consent, after redaction."""
    raise NotImplementedError
