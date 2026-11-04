"""Stage 5 - Feature engineering. Fit on TRAIN ONLY to avoid leakage."""

from __future__ import annotations


def tfidf(train_texts, **kwargs):
    """Fit TF-IDF vectoriser (B1)."""
    raise NotImplementedError


def embeddings(texts, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    """Sentence embeddings (B2)."""
    raise NotImplementedError
