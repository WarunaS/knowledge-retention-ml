"""Uncertainty for reported scores: bootstrap confidence intervals and paired comparisons.

Resampling is by CONVERSATION (cluster bootstrap): segments of one conversation are correlated, so
resampling single segments would make the intervals too narrow. Each resample draws conversations
with replacement and keeps all their segments.

Macro-F1 inside the bootstrap uses a fixed label set (labels present in the full gold data). In a
resample, a label with no true and no predicted cases has no defined F1 and is left out of that
resample's average.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def macro_f1_fixed(y: np.ndarray, yhat: np.ndarray, labels: np.ndarray) -> float:
    """Macro-F1 over `labels`; labels with no true and no predicted cases are skipped."""
    y, yhat = y[:, labels], yhat[:, labels]
    tp = (y & yhat).sum(0)
    fp = ((1 - y) & yhat).sum(0)
    fn = (y & (1 - yhat)).sum(0)
    denom = 2 * tp + fp + fn
    ok = denom > 0
    return float((2 * tp[ok] / denom[ok]).mean()) if ok.any() else 0.0


def _resample_index(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.where(inverse == g)[0] for g in range(len(uniq))]
    picks = rng.integers(0, len(uniq), len(uniq))
    return np.concatenate([members[p] for p in picks])


def bootstrap_ci(y: np.ndarray, yhat: np.ndarray, groups: np.ndarray, n: int = 1000, seed: int = 42,
                 alpha: float = 0.05) -> dict[str, float]:
    """Point estimate and percentile CI of macro-F1, resampling groups (conversations)."""
    y, yhat = y.astype(np.int64), yhat.astype(np.int64)
    labels = np.where(y.sum(0) > 0)[0]
    rng = np.random.default_rng(seed)
    scores = np.array([macro_f1_fixed(y[i], yhat[i], labels) for i in (_resample_index(groups, rng) for _ in range(n))])
    lo, hi = np.percentile(scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"macro_f1": round(macro_f1_fixed(y, yhat, labels), 4), "ci_low": round(float(lo), 4),
            "ci_high": round(float(hi), 4)}


def paired_bootstrap(y: np.ndarray, yhat_a: np.ndarray, yhat_b: np.ndarray, groups: np.ndarray, n: int = 1000,
                     seed: int = 42, alpha: float = 0.05) -> dict[str, Any]:
    """Difference A - B in macro-F1 on the same resamples; CI and two-sided bootstrap p-value."""
    y, a, b = (m.astype(np.int64) for m in (y, yhat_a, yhat_b))
    labels = np.where(y.sum(0) > 0)[0]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i = _resample_index(groups, rng)
        diffs.append(macro_f1_fixed(y[i], a[i], labels) - macro_f1_fixed(y[i], b[i], labels))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {"diff": round(macro_f1_fixed(y, a, labels) - macro_f1_fixed(y, b, labels), 4),
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4), "p_value": round(float(min(p, 1.0)), 4)}


def unpaired_bootstrap_diff(y1: np.ndarray, yhat1: np.ndarray, g1: np.ndarray, y2: np.ndarray, yhat2: np.ndarray,
                            g2: np.ndarray, n: int = 1000, seed: int = 42, alpha: float = 0.05) -> dict[str, float]:
    """Difference set1 - set2 in macro-F1 for two independent test sets (e.g. gold vs transfer), each
    resampled by its own conversations. Labels: those present in either set."""
    y1, yhat1, y2, yhat2 = (m.astype(np.int64) for m in (y1, yhat1, y2, yhat2))
    labels = np.where((y1.sum(0) + y2.sum(0)) > 0)[0]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        i, j = _resample_index(g1, rng), _resample_index(g2, rng)
        diffs.append(macro_f1_fixed(y1[i], yhat1[i], labels) - macro_f1_fixed(y2[j], yhat2[j], labels))
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": round(macro_f1_fixed(y1, yhat1, labels) - macro_f1_fixed(y2, yhat2, labels), 4),
            "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}


def within_margin(paired: dict[str, Any], margin: float = 0.05) -> bool:
    """Non-inferiority for H2: with A = model, B = teacher, A is within `margin` of B if the lower CI
    bound of A - B is above -margin."""
    return paired["ci_low"] > -margin
