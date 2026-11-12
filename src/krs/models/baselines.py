"""Baselines (Stage 7): B0 majority, B1 TF-IDF + logistic regression, B2 MiniLM embeddings + LR.

Each label is its own yes/no classifier (one-vs-rest), with balanced class weights because decision
and contextual knowledge are rare. The regularisation strength C is chosen from a small grid on the
validation split, then per-label thresholds are tuned on validation (Stage 10). Predictions are
saved for every split; test scores are computed in Stage 9, never used for choices here.
Runs on a laptop CPU in seconds to minutes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from krs.features import load_features
from krs.models.common import (
    run_dir,
    save_predictions,
    save_run,
    tune_thresholds,
    validation_scores,
)

SPLITS = ("train", "validation", "test", "transfer")


def _constant(prevalence: np.ndarray, n: int) -> np.ndarray:
    return np.tile(prevalence, (n, 1))


def fit_lr(X_train, Y_train: np.ndarray, C: float = 1.0, seed: int = 42, max_iter: int = 2000) -> list:
    """One logistic regression per label column; a float (training prevalence) for labels with one class."""
    from sklearn.linear_model import LogisticRegression

    models: list = []
    for j in range(Y_train.shape[1]):
        y = Y_train[:, j]
        if y.min() == y.max():
            models.append(float(y.mean()))
            continue
        models.append(LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, random_state=seed).fit(X_train, y))
    return models


def predict_lr(models: list, X) -> np.ndarray:
    out = np.zeros((X.shape[0], len(models)))
    for j, m in enumerate(models):
        out[:, j] = m if isinstance(m, float) else m.predict_proba(X)[:, 1]
    return out


def fit_predict_lr(X_train, Y_train: np.ndarray, X_list: list, C: float = 1.0, seed: int = 42,
                   max_iter: int = 2000) -> list[np.ndarray]:
    models = fit_lr(X_train, Y_train, C, seed, max_iter)
    return [predict_lr(models, X) for X in X_list]


def _save_model(out: Path, model: str, kind: str, with_ctx: bool, models_kt: list, models_dom: list,
                features_dir: Path) -> None:
    """Everything needed to score new segments later: classifiers plus the TF-IDF vectoriser or the
    name of the embedding model."""
    import pickle
    import shutil

    from krs.config import load_config
    from krs.features import VIEWS

    folder = out / "model"
    folder.mkdir(exist_ok=True)
    meta = {"model": model, "features": kind, "with_context": with_ctx}
    if kind == "tfidf":
        shutil.copy(features_dir / f"tfidf_{VIEWS[with_ctx]}.pkl", folder / "tfidf.pkl")
    else:
        meta["embedding_model"] = (load_config("features").get("embeddings") or {}).get(
            "model", "sentence-transformers/all-MiniLM-L6-v2")
    with open(folder / "classifiers.pkl", "wb") as f:
        pickle.dump({"meta": meta, "knowledge_types": models_kt, "domains": models_dom}, f)


def train_baseline(run_cfg: dict[str, Any], features_dir: str | Path | None = None,
                   results_root: str | Path | None = None) -> dict[str, Any]:
    model = run_cfg["model"]
    with_ctx = run_cfg.get("with_context", True)
    kind = {"majority": "tfidf", "tfidf_lr": "tfidf", "minilm_lr": "emb"}[model]
    data = {}
    for s in SPLITS:
        try:
            data[s] = load_features(s, kind, with_ctx, features_dir)
        except FileNotFoundError:
            continue
    if "train" not in data or "validation" not in data:
        raise FileNotFoundError("train and validation features needed - run Stages 6 and 5 first")
    Xtr, Ytr_kt, Ytr_dom = data["train"]
    Xva, Yva_kt, Yva_dom = data["validation"]
    order = [s for s in SPLITS if s in data]

    if model == "majority":
        probs = {s: (_constant(Ytr_kt.mean(0), data[s][0].shape[0]), _constant(Ytr_dom.mean(0), data[s][0].shape[0]))
                 for s in order}
        chosen_C = None
        models_kt = [float(v) for v in Ytr_kt.mean(0)]
        models_dom = [float(v) for v in Ytr_dom.mean(0)]
    else:
        best = None
        for C in run_cfg.get("C_grid", [0.1, 1.0, 10.0]):  # Stage 10: choose C on validation
            p_kt = fit_predict_lr(Xtr, Ytr_kt, [Xva], C, run_cfg.get("seed", 42))[0]
            p_dom = fit_predict_lr(Xtr, Ytr_dom, [Xva], C, run_cfg.get("seed", 42))[0]
            score = validation_scores(p_kt, p_dom, Yva_kt, Yva_dom, tune_thresholds(p_kt, Yva_kt),
                                      tune_thresholds(p_dom, Yva_dom))["macro_f1_mean"]
            if best is None or score > best[0]:
                best = (score, C)
        chosen_C = best[1]
        models_kt = fit_lr(Xtr, Ytr_kt, chosen_C, run_cfg.get("seed", 42))
        models_dom = fit_lr(Xtr, Ytr_dom, chosen_C, run_cfg.get("seed", 42))
        probs = {s: (predict_lr(models_kt, data[s][0]), predict_lr(models_dom, data[s][0])) for s in order}

    th_kt = tune_thresholds(probs["validation"][0], Yva_kt)
    th_dom = tune_thresholds(probs["validation"][1], Yva_dom)
    out = run_dir(run_cfg["run_id"], results_root)
    folder = Path(features_dir) if features_dir else _features_dir()
    for s in order:
        ids = np.load(folder / f"{s}_labels.npz")["segment_ids"]
        save_predictions(out, s, ids, probs[s][0], probs[s][1], th_kt, th_dom)
    _save_model(out, model, kind, with_ctx, models_kt, models_dom, folder)
    metrics = {"run_id": run_cfg["run_id"], "model": model, "with_context": with_ctx, "chosen_C": chosen_C,
               **validation_scores(*probs["validation"], Yva_kt, Yva_dom, th_kt, th_dom)}
    save_run(out, run_cfg, th_kt, th_dom, metrics)
    return metrics


def _features_dir() -> Path:
    from krs.config import get_paths

    return get_paths().features
