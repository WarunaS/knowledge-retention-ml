"""Rule filters plus a technical-relevance filter."""

from __future__ import annotations


def is_english(conv):
    """Language check (use dataset metadata when present, langdetect otherwise)."""
    raise NotImplementedError


def is_non_toxic(conv):
    """Use the dataset toxicity flag."""
    raise NotImplementedError


def has_min_exchanges(conv, n: int = 2):
    """At least n user-assistant exchanges. Agentic runs are exempt."""
    raise NotImplementedError


def is_technical(conv):
    """Programming relevance: keyword seed + zero-shot classifier. Hand-check 200 to report precision."""
    raise NotImplementedError
