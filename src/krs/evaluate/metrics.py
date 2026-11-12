"""Stage 9 - Evaluate on the human gold set (the test split) only.

    from krs.evaluate.metrics import evaluate_runs
    summary = evaluate_runs("evaluate")       # writes results/eval/

For every run with test predictions: macro-F1 per axis with conversation-bootstrap 95% CIs, micro-F1,
exact match, knowledge-bearing F1 (is there any knowledge at all?), per-label precision/recall/F1,
and macro-F1 per source. Then the hypothesis comparisons from configs/evaluate.yaml (paired
bootstrap), confusion matrices, a results table for the report, and the A2 learning curve.
Runs that do not exist yet are listed as missing, not treated as errors.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from krs.config import RESULTS_DIR, load_config
from krs.evaluate.stats import bootstrap_ci, macro_f1_fixed, paired_bootstrap, within_margin
from krs.features import label_matrices, load_rows
from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES
from krs.models.common import NONE_IDX, load_predictions

AXES = {"knowledge_types": KNOWLEDGE_TYPES, "domains": DOMAINS}


def load_gold(split: str = "test", splits_dir: str | Path | None = None) -> dict[str, Any]:
    rows = load_rows(split, splits_dir)
    if not rows:
        raise FileNotFoundError(f"no {split} split - finish gold labelling (Stage 4b) and Stage 6")
    kt, dom = label_matrices(rows)
    return {"segment_ids": np.array([r.segment.segment_id for r in rows]),
            "groups": np.array([r.segment.conversation_id for r in rows]),
            "sources": np.array([r.segment.source_dataset.value for r in rows]),
            "knowledge_types": kt.astype(np.int64), "domains": dom.astype(np.int64)}


def aligned_predictions(run_id: str, gold: dict[str, Any], split: str = "test",
                        results_root: str | Path | None = None) -> dict[str, np.ndarray]:
    """Predicted 0/1 matrices in gold order. Raises if a gold segment has no prediction."""
    pred = load_predictions(run_id, split, results_root)
    pos = {s: i for i, s in enumerate(pred["segment_ids"])}
    missing = [s for s in gold["segment_ids"] if s not in pos]
    if missing:
        raise KeyError(f"{run_id}: {len(missing)} gold segments have no prediction, e.g. {missing[:3]}")
    idx = np.array([pos[s] for s in gold["segment_ids"]])
    return {axis: pred[axis][idx].astype(np.int64) for axis in AXES}


def per_label(y: np.ndarray, yhat: np.ndarray, names: Sequence[str]) -> list[dict[str, Any]]:
    out = []
    for j, name in enumerate(names):
        tp = int((y[:, j] & yhat[:, j]).sum())
        fp = int(((1 - y[:, j]) & yhat[:, j]).sum())
        fn = int((y[:, j] & (1 - yhat[:, j])).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        out.append({"label": name, "support": int(y[:, j].sum()), "predicted": int(yhat[:, j].sum()),
                    "precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4)})
    return out


def micro_f1(y: np.ndarray, yhat: np.ndarray) -> float:
    tp, fp, fn = (y & yhat).sum(), ((1 - y) & yhat).sum(), (y & (1 - yhat)).sum()
    return round(float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0, 4)


def report_run(run_id: str, gold: dict[str, Any], n_boot: int = 1000, seed: int = 42,
               results_root: str | Path | None = None) -> dict[str, Any]:
    pred = aligned_predictions(run_id, gold, results_root=results_root)
    rep: dict[str, Any] = {"run_id": run_id, "segments": len(gold["segment_ids"])}
    for axis, names in AXES.items():
        y, yhat = gold[axis], pred[axis]
        labels = np.where(y.sum(0) > 0)[0]
        by_source = {}
        for s in sorted(set(gold["sources"])):
            m = gold["sources"] == s
            by_source[s] = round(macro_f1_fixed(y[m], yhat[m], np.where(y[m].sum(0) > 0)[0]), 4)
        rep[axis] = {
            **bootstrap_ci(y, yhat, gold["groups"], n_boot, seed),
            "micro_f1": micro_f1(y, yhat),
            "exact_match": round(float((y == yhat).all(axis=1).mean()), 4),
            "labels_in_gold": [names[j] for j in labels],
            "per_label": per_label(y, yhat, names),
            "macro_f1_by_source": by_source,
        }
    # knowledge-bearing = any knowledge type other than 'none' (the flag that triggers extraction)
    yk = 1 - gold["knowledge_types"][:, [NONE_IDX]]
    pk = 1 - pred["knowledge_types"][:, [NONE_IDX]]
    rep["knowledge_bearing_f1"] = per_label(yk, pk, ["knowledge_bearing"])[0]["f1"]
    rep["macro_f1_mean"] = round((rep["knowledge_types"]["macro_f1"] + rep["domains"]["macro_f1"]) / 2, 4)
    return rep


def label_confusion(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """Counts of (true label i, predicted label j) pairs within the same segment: where labels get confused."""
    return y.T @ yhat


def plot_confusion(y: np.ndarray, yhat: np.ndarray, names: Sequence[str], title: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = label_confusion(y, yhat)
    row_sum = np.maximum(y.sum(0, keepdims=True).T, 1)
    share = cm / row_sum
    fig, ax = plt.subplots(figsize=(1 + 0.6 * len(names), 1 + 0.55 * len(names)))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if cm[i, j]:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7,
                        color="white" if share[i, j] > 0.6 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true (gold)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_learning_curve(curve: dict[int, dict[str, float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = sorted(curve)
    fig, ax = plt.subplots(figsize=(6, 4))
    for axis, label in (("knowledge_types", "knowledge type"), ("domains", "domain")):
        mid = [curve[s][axis]["macro_f1"] for s in sizes]
        lo = [curve[s][axis]["ci_low"] for s in sizes]
        hi = [curve[s][axis]["ci_high"] for s in sizes]
        ax.plot(sizes, mid, marker="o", label=label)
        ax.fill_between(sizes, lo, hi, alpha=0.2)
    ax.set_xscale("log")
    ax.set_xlabel("training segments (weak labels)")
    ax.set_ylabel("macro-F1 on gold (95% CI)")
    ax.set_title("Learning curve (A2)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def markdown_table(df) -> str:
    """Plain Markdown table (no extra dependency), for pasting into the report."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _available(run_id: str, results_root: str | Path | None) -> bool:
    return ((Path(results_root) if results_root else RESULTS_DIR) / run_id / "predictions_test.npz").exists()


def evaluate_runs(config: str | Path = "evaluate", splits_dir: str | Path | None = None,
                  results_root: str | Path | None = None, out_dir: str | Path | None = None) -> dict[str, Any]:
    import pandas as pd

    cfg = load_config(config)
    boot = cfg.get("bootstrap") or {}
    n_boot, seed = boot.get("n", 1000), boot.get("seed", 42)
    out = Path(out_dir) if out_dir else (Path(results_root) if results_root else RESULTS_DIR) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    gold = load_gold(cfg.get("split", "test"), splits_dir)

    runs = [r for r in cfg.get("runs", []) if _available(r, results_root)]
    missing = [r for r in cfg.get("runs", []) if r not in runs]
    reports = {r: report_run(r, gold, n_boot, seed, results_root) for r in runs}
    for r, rep in reports.items():
        (out / f"report_{r}.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")

    table = pd.DataFrame([{
        "run": r,
        "knowledge_type_macro_f1": rep["knowledge_types"]["macro_f1"],
        "kt_95ci": f"{rep['knowledge_types']['ci_low']:.3f}–{rep['knowledge_types']['ci_high']:.3f}",
        "domain_macro_f1": rep["domains"]["macro_f1"],
        "domain_95ci": f"{rep['domains']['ci_low']:.3f}–{rep['domains']['ci_high']:.3f}",
        "knowledge_bearing_f1": rep["knowledge_bearing_f1"],
        "mean_macro_f1": rep["macro_f1_mean"],
    } for r, rep in reports.items()])
    if len(table):
        table.to_csv(out / "results_table.csv", index=False)
        (out / "results_table.md").write_text(markdown_table(table), encoding="utf-8")

    comparisons = []
    for c in cfg.get("comparisons", []):
        a, b = c["a"], c["b"]
        if a not in reports or b not in reports:
            comparisons.append({**c, "status": "skipped - run missing"})
            continue
        pa = aligned_predictions(a, gold, results_root=results_root)
        pb = aligned_predictions(b, gold, results_root=results_root)
        res = {**c}
        for axis in AXES:
            pt = paired_bootstrap(gold[axis], pa[axis], pb[axis], gold["groups"], n_boot, seed)
            if c.get("test") == "within":
                pt["within_margin"] = within_margin(pt, c.get("margin", 0.05))
            else:
                pt["a_better"] = pt["ci_low"] > 0
            res[axis] = pt
        comparisons.append(res)
    (out / "comparisons.json").write_text(json.dumps(comparisons, indent=2), encoding="utf-8")

    for r in cfg.get("confusion_for", []):
        if r in reports:
            pr = aligned_predictions(r, gold, results_root=results_root)
            for axis, names in AXES.items():
                plot_confusion(gold[axis], pr[axis], names, f"{r} - {axis.replace('_', ' ')}", out / f"confusion_{r}_{axis}.png")

    curve = {}
    lc = cfg.get("learning_curve")
    if lc:
        for size in lc.get("sizes", []):
            rid = f"{lc['run']}_{size}"
            if _available(rid, results_root):
                rep = report_run(rid, gold, n_boot, seed, results_root)
                curve[size] = {axis: rep[axis] for axis in AXES}
        if len(curve) >= 2:
            plot_learning_curve(curve, out / "learning_curve.png")

    summary = {"split": cfg.get("split", "test"), "gold_segments": len(gold["segment_ids"]),
               "gold_conversations": len(set(gold["groups"])), "bootstrap": {"n": n_boot, "by": "conversation"},
               "runs_evaluated": runs, "runs_missing": missing, "comparisons": comparisons,
               "learning_curve": {s: {a: v[a]["macro_f1"] for a in AXES} for s, v in curve.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
