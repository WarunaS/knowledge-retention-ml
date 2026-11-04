"""Semantic retrieval filtered by domain and knowledge type; optional one-hop graph expansion."""

from __future__ import annotations


def search(query: str, domains=None, knowledge_types=None, k: int = 5):
    """Return ranked cards with scores and citations."""
    raise NotImplementedError
