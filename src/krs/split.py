"""Stage 6 - Split by conversation_id. Test = gold set only; gold segments never enter training."""

from __future__ import annotations


def make_splits(segments_path: str, gold_ids: set[str], val_frac: float = 0.1, seed: int = 42):
    """Assign Split to every segment; write data/splits/*.jsonl."""
    raise NotImplementedError
