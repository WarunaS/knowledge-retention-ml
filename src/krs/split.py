"""Stage 6 - Split. Runs BEFORE feature engineering, so no feature is fitted on test data.

    from krs.split import make_splits
    report = make_splits("split")      # writes data/splits/{train,validation,test,transfer}.jsonl

- train / validation: LLM weak labels, split by conversation within each source (seeded), so the
  segments of one conversation are never on both sides and every source appears in validation.
- test: the adjudicated human gold set only. Reported results come only from here.
- transfer: hand-labelled consented own logs, if present (scored once, run X1).

Every conversation that contains a gold segment is removed from train and validation (Stage 4b
already excludes them; this is the second check). After writing, `check_leakage` confirms that no
conversation appears in two splits and raises if one does.
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from krs.config import get_paths, load_config
from krs.io import read_jsonl, write_jsonl
from krs.label.sampling import load_segments
from krs.schema import Label, LabelledSegment, Split


class LeakageError(RuntimeError):
    pass


def _labels(path: Path | None) -> list[Label]:
    return list(read_jsonl(path, Label)) if path and path.exists() else []


def _join(labels: list[Label], split: Split, segments_dir: str | Path | None) -> list[LabelledSegment]:
    segs = load_segments([lab.segment_id for lab in labels], segments_dir) if labels else {}
    return [
        LabelledSegment(segment=segs[lab.segment_id].model_copy(update={"split": split}), label=lab)
        for lab in labels
    ]


def validation_conversations(rows: Iterable[LabelledSegment], val_frac: float, seed: int) -> set[str]:
    """Choose validation conversations per source: shuffle each source's conversations, take val_frac."""
    by_source: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        by_source[r.segment.source_dataset.value].add(r.segment.conversation_id)
    rng = random.Random(seed)
    chosen: set[str] = set()
    for source in sorted(by_source):
        convs = sorted(by_source[source])
        rng.shuffle(convs)
        k = round(len(convs) * val_frac)
        if val_frac > 0 and len(convs) > 1:
            k = max(k, 1)  # every source with 2+ conversations appears in validation
        chosen.update(convs[:k])
    return chosen


def check_leakage(splits: dict[str, list[LabelledSegment]]) -> None:
    """Raise LeakageError if a conversation or segment appears in more than one split."""
    seen_conv: dict[str, str] = {}
    seen_seg: dict[str, str] = {}
    problems = []
    for name, rows in splits.items():
        for r in rows:
            cid, sid = r.segment.conversation_id, r.segment.segment_id
            if seen_conv.get(cid, name) != name:
                problems.append(f"conversation {cid} in {seen_conv[cid]} and {name}")
            if seen_seg.get(sid, name) != name:
                problems.append(f"segment {sid} in {seen_seg[sid]} and {name}")
            seen_conv.setdefault(cid, name)
            seen_seg.setdefault(sid, name)
    if problems:
        raise LeakageError(f"{len(problems)} leaks, e.g. {problems[:3]}")


def _summary(rows: list[LabelledSegment]) -> dict[str, Any]:
    n = len(rows)
    share = lambda c: {k: round(v / n, 3) for k, v in sorted(c.items())} if n else {}
    return {
        "segments": n,
        "conversations": len({r.segment.conversation_id for r in rows}),
        "by_source": dict(sorted(Counter(r.segment.source_dataset.value for r in rows).items())),
        "knowledge_type_share": share(Counter(k.value for r in rows for k in r.label.knowledge_types)),
        "domain_share": share(Counter(d.value for r in rows for d in r.label.domains)),
    }


def make_splits(config: str | Path = "split", segments_dir: str | Path | None = None,
                out_dir: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config(config)
    paths = get_paths()
    labels_dir = paths.labels
    weak = _labels(labels_dir / cfg.get("weak_labels", "weak.jsonl"))
    gold = _labels(labels_dir / cfg.get("gold_labels", "gold.jsonl"))
    transfer = _labels(labels_dir / cfg.get("transfer_labels", "transfer.jsonl"))
    if not weak:
        raise FileNotFoundError(f"no weak labels in {labels_dir} - run Stage 4b first")

    test = _join(gold, Split.TEST, segments_dir)
    transfer_rows = _join(transfer, Split.TRANSFER, segments_dir)
    held_out = {r.segment.conversation_id for r in test + transfer_rows}

    pool = _join(weak, Split.TRAIN, segments_dir)
    removed = sum(r.segment.conversation_id in held_out for r in pool)
    pool = [r for r in pool if r.segment.conversation_id not in held_out]

    val_convs = validation_conversations(pool, cfg.get("val_frac", 0.1), cfg.get("seed", 42))
    train, validation = [], []
    for r in pool:
        if r.segment.conversation_id in val_convs:
            validation.append(r.model_copy(update={"segment": r.segment.model_copy(update={"split": Split.VALIDATION})}))
        else:
            train.append(r)

    splits = {"train": train, "validation": validation, "test": test, "transfer": transfer_rows}
    check_leakage(splits)

    out = Path(out_dir) if out_dir else paths.splits
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        if rows or name in ("train", "validation"):
            write_jsonl(out / f"{name}.jsonl", rows)
    report = {
        "seed": cfg.get("seed", 42),
        "val_frac": cfg.get("val_frac", 0.1),
        "weak_rows_removed_held_out_conversation": removed,
        "test_present": bool(test),
        **{name: _summary(rows) for name, rows in splits.items()},
    }
    (out / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_split(name: str, splits_dir: str | Path | None = None) -> list[LabelledSegment]:
    folder = Path(splits_dir) if splits_dir else get_paths().splits
    return list(read_jsonl(folder / f"{name}.jsonl", LabelledSegment))
