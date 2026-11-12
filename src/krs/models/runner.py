"""Run one experiment by id: `run_experiment("B1")` reads configs/runs/B1.yaml and dispatches it.

    B0 majority · B1 tfidf_lr · B2 minilm_lr      -> baselines (laptop CPU is fine)
    M1, A1, A2 deberta (fine-tune)               -> finetune (Colab GPU)
    M2 setfit                                    -> setfit_run (Colab GPU, uses gold with cross-fitting)
    T  llm_teacher                               -> teacher (Claude API, costs money: pass confirm=True)
"""

from __future__ import annotations

from typing import Any

from krs.config import load_config


def run_experiment(run_id: str, **kwargs: Any) -> dict[str, Any]:
    cfg = load_config(f"runs/{run_id}")
    model = cfg["model"]
    if model in ("majority", "tfidf_lr", "minilm_lr"):
        from krs.models.baselines import train_baseline

        return train_baseline(cfg, **kwargs)
    if model == "deberta":
        from krs.models.finetune import train

        return train(cfg, **kwargs)
    if model == "setfit":
        from krs.models.setfit_run import train

        return train(cfg, **kwargs)
    if model == "llm_teacher":
        from krs.models.teacher import run

        return run(cfg, **kwargs)
    raise ValueError(f"run {run_id}: unknown model {model!r}")
