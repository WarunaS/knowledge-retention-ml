"""Stage 4 - Explore: distributions, BERTopic clusters, estimated knowledge-bearing share."""

from __future__ import annotations


def describe(segments_path: str):
    """Return a DataFrame of turn counts, token lengths, sources and modalities."""
    raise NotImplementedError


def topics(segments_path: str, sample: int = 20000):
    """Fit BERTopic on a sample and return topic info for the domain sanity check."""
    raise NotImplementedError
