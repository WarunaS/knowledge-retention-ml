"""MinHash LSH near-duplicate removal at conversation level (Broder, 1997). Run BEFORE splitting."""

from __future__ import annotations


def deduplicate(convs, threshold: float = 0.8):
    """Yield conversations with near-duplicates removed."""
    raise NotImplementedError
