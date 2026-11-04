"""LLM extraction of flagged segments into typed KnowledgeCards, with faithfulness check."""

from __future__ import annotations


def extract(predictions_path: str, out_path: str):
    """Write KnowledgeCard records; reject cards without grounded evidence."""
    raise NotImplementedError
