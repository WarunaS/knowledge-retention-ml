"""Cut conversations into Segment records: user-assistant pair + one preceding pair as context."""

from __future__ import annotations


def segment(conv, max_tokens: int = 512):
    """Yield Segment records with stable ids '<conversation_id>:<turn_index>'."""
    raise NotImplementedError
