"""Shared pieces for every model run (Stages 7, 8 and 10).

- Probabilities -> labels with the codebook rules: 'none' never sits with another knowledge type,
  and every segment gets at least one knowledge type and one domain.
- Stage 10 Tune: one decision threshold per label, chosen on the VALIDATION split only.
- Macro-F1 for model selection (validation only). Reported test numbers come from Stage 9.
- A run folder per experiment: results/<run_id>/ with the config, thresholds, predictions per split
  and validation metrics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from krs.config import RESULTS_DIR
from krs.label.taxonomy import KNOWLEDGE_TYPES

NONE_IDX = KNOWLEDGE_TYPES.index("none")
AXES = ("knowledge_types", "domains")
THRESHOLD_GRID = tuple(np.round(np.arange(0.05, 0.96, 0.05), 2))


def decode(p_kt: np.ndarray, p_dom: np.ndarray, th_kt: np.ndarray | float = 0.5,
           th_dom: np.ndarray | float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Turn probabilities into 0/1 labels following the codebook rules."""
    kt = (p_kt >= th_kt).astype(np.int8)
    dom = (p_dom >= th_dom).astype(np.int8)
    others = np.delete(kt, NONE_IDX, axis=1).sum(axis=1) > 0
    kt[others, NONE_IDX] = 0  # 'none' cannot sit with another type
    for y, p in ((kt, p_kt), (dom, p_dom)):
        empty = y.sum(axis=1) == 0
        y[empty, p[empty].argmax(axis=1)] = 1  # at least one label per axis
    return kt, dom


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the labels that occur in y_true (labels absent from the data are not averaged)."""
    from sklearn.metrics import f1_score

    present = np.where(y_true.sum(axis=0) > 0)[0]
    if len(present) == 0:
        return 0.0
    return float(f1_score(y_true[:, present], y_pred[:, present], average="macro", zero_division=0))


def tune_thresholds(p: np.ndarray, y: np.ndarray, grid: tuple[float, ...] = THRESHOLD_GRID) -> np.ndarray:
    """Per-label threshold that maximises that label's F1 on validation; 0.5 where the label is absent."""
    from sklearn.metrics import f1_score

    th = np.full(p.shape[1], 0.5)
    for j in range(p.shape[1]):
        if y[:, j].sum() == 0:
            continue
        scores = [f1_score(y[:, j], (p[:, j] >= t).astype(int), zero_division=0) for t in grid]
        th[j] = grid[int(np.argmax(scores))]
    return th


def validation_scores(p_kt, p_dom, y_kt, y_dom, th_kt, th_dom) -> dict[str, float]:
    kt, dom = decode(p_kt, p_dom, th_kt, th_dom)
    s = {"macro_f1_knowledge_types": macro_f1(y_kt, kt), "macro_f1_domains": macro_f1(y_dom, dom)}
    s["macro_f1_mean"] = (s["macro_f1_knowledge_types"] + s["macro_f1_domains"]) / 2
    return {k: round(v, 4) for k, v in s.items()}


def run_dir(run_id: str, root: str | Path | None = None) -> Path:
    path = (Path(root) if root else RESULTS_DIR) / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_predictions(out: Path, split: str, segment_ids, p_kt: np.ndarray, p_dom: np.ndarray,
                     th_kt: np.ndarray, th_dom: np.ndarray) -> None:
    kt, dom = decode(p_kt, p_dom, th_kt, th_dom)
    np.savez_compressed(out / f"predictions_{split}.npz", segment_ids=np.asarray(segment_ids),
                        p_knowledge_types=p_kt.astype(np.float32), p_domains=p_dom.astype(np.float32),
                        knowledge_types=kt, domains=dom)


def save_run(out: Path, run_cfg: dict[str, Any], th_kt: np.ndarray, th_dom: np.ndarray,
             metrics: dict[str, Any]) -> None:
    (out / "config.yaml").write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
    (out / "thresholds.json").write_text(json.dumps(
        {"knowledge_types": th_kt.round(3).tolist(), "domains": th_dom.round(3).tolist()}, indent=1), encoding="utf-8")
    (out / "metrics_validation.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def load_predictions(run_id: str, split: str, root: str | Path | None = None) -> dict[str, np.ndarray]:
    data = np.load((Path(root) if root else RESULTS_DIR) / run_id / f"predictions_{split}.npz")
    return {k: data[k] for k in data.files}
