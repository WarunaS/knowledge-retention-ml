"""T - the LLM teacher on the gold set: the ceiling for H2, plus cost and latency per 1,000 segments.

Uses the same Claude labeller as the weak labels (sync mode, one request per segment, latency
recorded). COSTS MONEY: `run` only returns an estimate unless called with confirm=True.
Segments the teacher failed on keep all-zero probabilities and are listed in the metrics; Stage 9
counts them as errors rather than dropping them.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import numpy as np

from krs.config import load_config
from krs.features import label_matrices, load_rows
from krs.io import read_jsonl
from krs.label import llm_labeller as LL
from krs.models.common import run_dir, save_predictions, save_run
from krs.schema import Label, LabelledSegment, LabelSource


def estimate(rows: list[LabelledSegment], llm: dict[str, Any], output_tokens: int = 400) -> dict[str, Any]:
    cb_tok = (len(LL.load_codebook()) + len(LL.INSTRUCTIONS)) / 4
    seg_tok = statistics.mean(len(LL.segment_prompt(r.segment)) for r in rows) / 4 if rows else 0
    p = llm.get("prices", {})
    per = (seg_tok * p.get("input", 0) + cb_tok * p.get("cache_read", 0) + output_tokens * p.get("output", 0)) / 1e6
    return {"segments": len(rows), "model": llm.get("model"), "estimated_cost_usd": round(per * len(rows), 2),
            "assumed_output_tokens": output_tokens}


def predictions_from_labels(rows: list[LabelledSegment], labels: dict[str, Label]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """0/1 'probabilities' from teacher labels, in row order; missing segments stay all-zero."""
    fake = [LabelledSegment(segment=r.segment, label=labels[r.segment.segment_id]) for r in rows
            if r.segment.segment_id in labels]
    kt_known, dom_known = label_matrices(fake) if fake else (None, None)
    from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES

    kt = np.zeros((len(rows), len(KNOWLEDGE_TYPES)))
    dom = np.zeros((len(rows), len(DOMAINS)))
    missing, j = [], 0
    for i, r in enumerate(rows):
        if r.segment.segment_id in labels:
            kt[i], dom[i] = kt_known[j], dom_known[j]
            j += 1
        else:
            missing.append(r.segment.segment_id)
    return kt, dom, missing


def run(run_cfg: dict[str, Any], confirm: bool = False, client: Any = None, splits_dir: str | Path | None = None,
        results_root: str | Path | None = None) -> dict[str, Any]:
    rows = load_rows("test", splits_dir)
    if not rows:
        raise FileNotFoundError("no test split (gold labels) - finish Stage 4b gold labelling and Stage 6")
    label_cfg = load_config("label")
    llm = {**label_cfg["llm"], **(run_cfg.get("llm") or {})}
    if not confirm:
        return {"dry_run": True, **estimate(rows, llm)}
    out = run_dir(run_cfg["run_id"], results_root)
    labels_path = out / "teacher_labels.jsonl"
    report = LL.label_sync([r.segment for r in rows], labels_path, llm, client=client,
                           codebook_version=label_cfg.get("codebook_version", "v1"), source=LabelSource.LLM)
    labels = {lab.segment_id: lab for lab in read_jsonl(labels_path, Label)}
    p_kt, p_dom, missing = predictions_from_labels(rows, labels)
    th = (np.full(p_kt.shape[1], 0.5), np.full(p_dom.shape[1], 0.5))
    save_predictions(out, "test", [r.segment.segment_id for r in rows], p_kt, p_dom, *th)
    metrics = {"run_id": run_cfg["run_id"], "model": llm.get("model"), "missing_segments": missing, **report}
    save_run(out, run_cfg, *th, metrics)
    return metrics
