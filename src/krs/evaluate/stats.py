"""Bootstrap 95% confidence intervals and paired tests (M1 vs B2)."""

from __future__ import annotations


def bootstrap_ci(gold_path: str, pred_path: str, n: int = 1000, seed: int = 42):
    """Return CI for macro-F1."""
    raise NotImplementedError


def paired_test(gold_path: str, pred_a: str, pred_b: str):
    """Return p-value for A vs B."""
    raise NotImplementedError
