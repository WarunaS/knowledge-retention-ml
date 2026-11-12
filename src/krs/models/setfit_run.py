"""M2 - SetFit few-shot: how little human labelling is enough? (Stage 8)

The plan says "SetFit trained on fifty gold examples per class", but the gold set is also the test
set. Training on it and testing on it would be leakage, so M2 uses CROSS-FITTING: the gold set is
split into k folds by conversation; for each fold, SetFit is trained on up to `n_per_class`
examples per label drawn from the other folds and predicts the held-out fold. Every gold
prediction therefore comes from a model that never saw that segment (or its conversation).

One SetFit model per axis (one-vs-rest heads). Thresholds stay at 0.5: there is no separate
validation data inside the gold set. Needs a GPU for the real sentence model: run on Colab.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from krs.features import label_matrices, load_rows
from krs.models.common import run_dir, save_predictions, save_run
from krs.schema import LabelledSegment


def conversation_folds(rows: list[LabelledSegment], k: int = 5, seed: int = 42) -> np.ndarray:
    """Fold number per row; all rows of a conversation share a fold."""
    convs = sorted({r.segment.conversation_id for r in rows})
    random.Random(seed).shuffle(convs)
    fold_of = {c: i % k for i, c in enumerate(convs)}
    return np.array([fold_of[r.segment.conversation_id] for r in rows])


def few_shot_indices(Y: np.ndarray, candidates: np.ndarray, n_per_class: int, seed: int = 42) -> np.ndarray:
    """Up to n positives per label from `candidates` (union over labels), seeded."""
    rng = random.Random(seed)
    chosen: set[int] = set()
    for j in range(Y.shape[1]):
        pos = [int(i) for i in candidates if Y[i, j] == 1]
        rng.shuffle(pos)
        chosen.update(pos[:n_per_class])
    return np.array(sorted(chosen), dtype=int)


def _fit_predict(texts_train, Y_train, texts_pred, cfg: dict[str, Any]) -> np.ndarray:
    from datasets import Dataset
    from setfit import SetFitModel, Trainer, TrainingArguments

    model = SetFitModel.from_pretrained(cfg.get("base_model", "sentence-transformers/all-MiniLM-L6-v2"),
                                        multi_target_strategy="one-vs-rest")
    ds = Dataset.from_dict({"text": list(texts_train), "label": Y_train.astype(int).tolist()})
    with tempfile.TemporaryDirectory() as tmp:  # SetFit would otherwise write ./checkpoints
        args = TrainingArguments(output_dir=tmp, save_strategy="no", report_to="none",
                                 batch_size=cfg.get("batch_size", 16), num_epochs=cfg.get("num_epochs", 1),
                                 num_iterations=cfg.get("num_iterations", 20), seed=cfg.get("seed", 42))
        Trainer(model=model, args=args, train_dataset=ds).train()
    probs = model.predict_proba(list(texts_pred))
    probs = probs.cpu().numpy() if hasattr(probs, "cpu") else probs
    if isinstance(probs, list):  # some heads return one (n, 2) array per label
        probs = np.stack([p[:, 1] if np.ndim(p) == 2 else p for p in probs], axis=1)
    probs = np.asarray(probs, dtype=float)
    if probs.shape[1] != Y_train.shape[1]:  # labels absent from the training sample
        raise ValueError(f"SetFit returned {probs.shape[1]} columns for {Y_train.shape[1]} labels")
    return probs


def train(run_cfg: dict[str, Any], splits_dir: str | Path | None = None,
          results_root: str | Path | None = None, fit_predict=_fit_predict) -> dict[str, Any]:
    """Out-of-fold SetFit predictions for the whole gold (test) split."""
    rows = load_rows("test", splits_dir)
    if not rows:
        raise FileNotFoundError("no test split (gold labels) - finish Stage 4b gold labelling and Stage 6")
    with_ctx = run_cfg.get("with_context", True)
    texts = [r.segment.model_input(with_context=with_ctx) for r in rows]
    Y_kt, Y_dom = label_matrices(rows)
    k = run_cfg.get("folds", 5)
    folds = conversation_folds(rows, k, run_cfg.get("seed", 42))
    p_kt, p_dom = np.zeros(Y_kt.shape), np.zeros(Y_dom.shape)
    sizes = []
    for f in range(k):
        held = np.where(folds == f)[0]
        rest = np.where(folds != f)[0]
        for Y, P in ((Y_kt, p_kt), (Y_dom, p_dom)):
            idx = few_shot_indices(Y, rest, run_cfg.get("n_per_class", 50), run_cfg.get("seed", 42) + f)
            sizes.append(len(idx))
            P[held] = fit_predict([texts[i] for i in idx], Y[idx], [texts[i] for i in held], run_cfg)
    th_kt, th_dom = np.full(Y_kt.shape[1], 0.5), np.full(Y_dom.shape[1], 0.5)
    out = run_dir(run_cfg["run_id"], results_root)
    save_predictions(out, "test", [r.segment.segment_id for r in rows], p_kt, p_dom, th_kt, th_dom)
    metrics = {"run_id": run_cfg["run_id"], "model": "setfit", "scheme": f"{k}-fold cross-fitting on gold",
               "n_per_class": run_cfg.get("n_per_class", 50), "train_examples_per_fit": sizes,
               "note": "no validation split; thresholds fixed at 0.5; scored in Stage 9 on out-of-fold predictions"}
    save_run(out, run_cfg, th_kt, th_dom, metrics)
    return metrics
