"""Inter-rater agreement on the gold set, disagreement sheet, and the final adjudicated gold labels.

Agreement for multi-label annotation: each label is a yes/no decision per segment, so Cohen's kappa
(Cohen, 1960) is computed per label; the axis score is the macro mean over labels that occur.
Exact-match agreement per axis is reported too. Kappa below the threshold (0.6, "substantial" on
the Landis and Koch scale) means the codebook is revised and the gold set relabelled once.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from krs.label.gold import export_rater_sheet, import_rater_sheet
from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES
from krs.schema import Label, LabelSource, Segment


def binary_kappa(a: list[bool], b: list[bool]) -> float | None:
    """Cohen's kappa for two raters' yes/no answers. None when undefined (both raters constant and equal)."""
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return None
    return (po - pe) / (1 - pe)


def _sets(lab: Label) -> tuple[set[str], set[str]]:
    return {k.value for k in lab.knowledge_types}, {d.value for d in lab.domains}


def agreement(rater1: Iterable[Label], rater2: Iterable[Label], threshold: float = 0.6) -> dict[str, Any]:
    """Kappa per label and per axis on the segments both raters labelled."""
    a = {lab.segment_id: lab for lab in rater1}
    b = {lab.segment_id: lab for lab in rater2}
    common = sorted(a.keys() & b.keys())
    result: dict[str, Any] = {"segments_compared": len(common), "only_rater1": len(a.keys() - b.keys()),
                              "only_rater2": len(b.keys() - a.keys())}
    for axis, names, idx in (("knowledge_type", KNOWLEDGE_TYPES, 0), ("domain", DOMAINS, 1)):
        per_label = {}
        for name in names:
            ya = [name in _sets(a[s])[idx] for s in common]
            yb = [name in _sets(b[s])[idx] for s in common]
            per_label[name] = binary_kappa(ya, yb)
        defined = [k for k in per_label.values() if k is not None]
        macro = sum(defined) / len(defined) if defined else None
        exact = sum(_sets(a[s])[idx] == _sets(b[s])[idx] for s in common) / len(common) if common else None
        result[axis] = {
            "kappa_macro": None if macro is None else round(macro, 3),
            "exact_match": None if exact is None else round(exact, 3),
            "kappa_per_label": {k: (None if v is None else round(v, 3)) for k, v in per_label.items()},
            "meets_threshold": macro is not None and macro >= threshold,
        }
    return result


def disagreements(rater1: Iterable[Label], rater2: Iterable[Label]) -> list[str]:
    """Segment ids where the raters differ on either axis."""
    a = {lab.segment_id: lab for lab in rater1}
    b = {lab.segment_id: lab for lab in rater2}
    return sorted(s for s in a.keys() & b.keys() if _sets(a[s]) != _sets(b[s]))


def _show(lab: Label) -> str:
    ks, ds = _sets(lab)
    return f"types: {', '.join(sorted(ks))} | domains: {', '.join(sorted(ds))}"


def export_adjudication_sheet(rater1: list[Label], rater2: list[Label], segments: dict[str, Segment],
                              out_csv: str | Path) -> int:
    """Sheet for the third member: both raters' answers shown, label columns empty to fill."""
    a = {lab.segment_id: lab for lab in rater1}
    b = {lab.segment_id: lab for lab in rater2}
    ids = disagreements(rater1, rater2)
    extra = {s: {"rater1": _show(a[s]), "rater2": _show(b[s])} for s in ids}
    return export_rater_sheet([segments[s] for s in ids], out_csv, extra_cols=extra)


def merge_gold(rater1: list[Label], rater2: list[Label], adjudication_csv: str | Path | None = None,
               adjudicator_id: str = "adjudicator") -> tuple[list[Label], list[str]]:
    """Final gold labels: agreed segments as labelled, disagreements from the adjudication sheet.

    Returns (labels, problems). Every final label has source=adjudicated.
    """
    a = {lab.segment_id: lab for lab in rater1}
    b = {lab.segment_id: lab for lab in rater2}
    differ = set(disagreements(rater1, rater2))
    adjudicated: dict[str, Label] = {}
    problems: list[str] = []
    if adjudication_csv:
        labs, problems = import_rater_sheet(adjudication_csv, LabelSource.ADJUDICATED, adjudicator_id)
        adjudicated = {lab.segment_id: lab for lab in labs}
    final: list[Label] = []
    for s in sorted(a.keys() & b.keys()):
        if s in differ:
            if s in adjudicated:
                final.append(adjudicated[s])
            else:
                problems.append(f"{s}: raters disagree and no adjudicated label")
        else:
            final.append(a[s].model_copy(update={"source": LabelSource.ADJUDICATED, "rater_id": "agreed"}))
    return final, problems
