"""LLM weak labelling with schema-constrained output, temperature 0. Raw responses stored."""

from __future__ import annotations


def label_segments(segments_path: str, out_path: str, codebook_version: str = 'v1'):
    """Write Label(source=llm) records; keep raw_response for audit."""
    raise NotImplementedError
