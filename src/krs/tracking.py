"""Stage 11 - Track experiments: one run dir per config with config copy, metrics.json, predictions."""

from __future__ import annotations


def start_run(run_config: dict):
    """Create results/<run_id>/ and return the run id."""
    raise NotImplementedError


def log_metrics(run_id: str, metrics: dict):
    """Write metrics.json for the run."""
    raise NotImplementedError
