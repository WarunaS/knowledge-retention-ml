"""Stage 14 - Offline batch inference. No deployment; CPU, no cloud call."""

from __future__ import annotations


def classify(segments_path: str, model_version: str, out_path: str, threshold: float = 0.5):
    """Write Prediction records with per-label probabilities and knowledge_bearing flag."""
    raise NotImplementedError
