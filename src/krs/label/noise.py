"""LLM label noise: a person checks 100 random weak labels and marks each axis correct or not.

The resulting error rates are reported next to the results, which are computed only on the gold set.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from krs.clean.precision_check import wilson
from krs.io import read_jsonl
from krs.label.sampling import load_segments
from krs.schema import Label


def export_noise_sample(weak_path: str | Path, out_csv: str | Path, n: int = 100, seed: int = 42,
                        segments_dir: str | Path | None = None) -> int:
    labels = list(read_jsonl(weak_path, Label))
    sample = random.Random(seed).sample(labels, min(n, len(labels)))
    segs = load_segments([lab.segment_id for lab in sample], segments_dir)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "context", "segment", "llm_knowledge_types", "llm_domains",
                    "types_correct", "domains_correct", "notes"])
        for lab in sample:
            s = segs[lab.segment_id]
            w.writerow([lab.segment_id, s.context_text, s.text,
                        ", ".join(k.value for k in lab.knowledge_types),
                        ", ".join(d.value for d in lab.domains), "", "", ""])
    return len(sample)


def noise_rate(csv_path: str | Path) -> dict[str, Any]:
    """Share of weak labels judged wrong per axis (y = correct, n = wrong), with 95% Wilson intervals."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, Any] = {}
    for axis, col in (("knowledge_type", "types_correct"), ("domain", "domains_correct")):
        marks = [r[col].strip().lower() for r in rows if r[col].strip().lower() in ("y", "n")]
        wrong = sum(m == "n" for m in marks)
        lo, hi = wilson(wrong, len(marks))
        out[axis] = {"checked": len(marks), "wrong": wrong,
                     "noise_rate": round(wrong / len(marks), 3) if marks else None,
                     "ci95_low": round(lo, 3), "ci95_high": round(hi, 3)}
    return out
