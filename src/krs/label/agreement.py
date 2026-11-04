"""Human gold set agreement: Cohen's kappa per axis, adjudication log."""

from __future__ import annotations


def cohens_kappa(rater1_path: str, rater2_path: str):
    """Return kappa per axis and per label. Below 0.6 triggers one codebook revision."""
    raise NotImplementedError
