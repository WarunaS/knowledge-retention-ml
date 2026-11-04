"""Stage 12 - Register / version: models/<version>/ with weights, config, metrics.json, model_card.md."""

from __future__ import annotations


def register(run_id: str, version: str):
    """Copy a gold-evaluated run into the registry. Unevaluated runs are refused."""
    raise NotImplementedError


def load(version: str):
    """Load a registered model for inference."""
    raise NotImplementedError
