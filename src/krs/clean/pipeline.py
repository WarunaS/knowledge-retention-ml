"""Runs stage 3 end to end and writes data/filtered and data/segments."""

from __future__ import annotations


def run_cleaning(config: str = 'clean'):
    """Load raw conversations, filter, dedup, normalise, segment, write JSONL + counts per step."""
    raise NotImplementedError
