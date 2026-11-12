"""Human gold labels: CSV sheets for raters, and reading them back.

Each rater gets the same sheet: one row per segment with its context and text, and one column per
label (`k:<knowledge type>`, `d:<domain>`). Raters put `x` (or 1 / y) in every column that applies.
Raters work independently and must not see each other's sheet.

Reading a sheet back validates every row: at least one knowledge type, `none` alone, at least one
domain. Problems are returned with the row, so the rater can fix them.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES, Domain, KnowledgeType
from krs.schema import Label, LabelSource, Segment

K_COLS = [f"k:{k}" for k in KNOWLEDGE_TYPES]
D_COLS = [f"d:{d}" for d in DOMAINS]
INFO_COLS = ["segment_id", "source", "context", "segment"]
MARKS = {"x", "1", "y", "yes", "true"}


def export_rater_sheet(segments: Iterable[Segment], out_csv: str | Path, prefill: dict[str, Label] | None = None,
                       extra_cols: dict[str, dict[str, str]] | None = None) -> int:
    """Write a labelling sheet. `prefill` marks labels already chosen (used for adjudication sheets)."""
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    extra_names = sorted({k for v in (extra_cols or {}).values() for k in v})
    n = 0
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(INFO_COLS + extra_names + K_COLS + D_COLS + ["notes"])
        for seg in segments:
            lab = (prefill or {}).get(seg.segment_id)
            ks = {k.value for k in lab.knowledge_types} if lab else set()
            ds = {d.value for d in lab.domains} if lab else set()
            extras = [(extra_cols or {}).get(seg.segment_id, {}).get(c, "") for c in extra_names]
            w.writerow(
                [seg.segment_id, seg.source_dataset.value, seg.context_text, seg.text]
                + extras
                + ["x" if k in ks else "" for k in KNOWLEDGE_TYPES]
                + ["x" if d in ds else "" for d in DOMAINS]
                + [""]
            )
            n += 1
    return n


def import_rater_sheet(csv_path: str | Path, source: LabelSource, rater_id: str,
                       codebook_version: str = "v1") -> tuple[list[Label], list[str]]:
    """Read a filled sheet. Returns (labels, problems); rows with problems are left out."""
    labels: list[Label] = []
    problems: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for line_no, row in enumerate(csv.DictReader(f), start=2):
            seg_id = row.get("segment_id", "").strip()
            ks = [KnowledgeType(c[2:]) for c in K_COLS if row.get(c, "").strip().lower() in MARKS]
            ds = [Domain(c[2:]) for c in D_COLS if row.get(c, "").strip().lower() in MARKS]
            if not ks:
                problems.append(f"row {line_no} ({seg_id}): no knowledge type marked")
                continue
            if KnowledgeType.NONE in ks and len(ks) > 1:
                problems.append(f"row {line_no} ({seg_id}): 'none' marked together with another type")
                continue
            if not ds:
                problems.append(f"row {line_no} ({seg_id}): no domain marked")
                continue
            labels.append(Label(segment_id=seg_id, knowledge_types=ks, domains=ds, source=source,
                                rater_id=rater_id, codebook_version=codebook_version))
    return labels, problems
