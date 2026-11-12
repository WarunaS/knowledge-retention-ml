"""Choose which segments get labelled.

- Gold set (~400): human-labelled test set, stratified by source with equal shares, so every source
  can be scored on its own. Written to labels/gold_ids.json.
- Training pool (~10,000): LLM weak labels. Every segment from a conversation that has a gold
  segment is excluded, so no test conversation leaks into training (splits are by conversation).

Only ids and a few fields are held in memory while sampling; text is read again on demand.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from krs.config import get_paths
from krs.io import read_jsonl
from krs.schema import Segment


@dataclass(frozen=True)
class SegmentRef:
    segment_id: str
    conversation_id: str
    source: str
    modality: str


def _files(segments_dir: str | Path | None) -> list[Path]:
    folder = Path(segments_dir) if segments_dir else get_paths().segments
    return sorted(folder.glob("*.jsonl"))


def segment_refs(segments_dir: str | Path | None = None) -> list[SegmentRef]:
    return [
        SegmentRef(s.segment_id, s.conversation_id, s.source_dataset.value, s.modality.value)
        for path in _files(segments_dir)
        for s in read_jsonl(path, Segment)
    ]


def load_segments(ids: Iterable[str], segments_dir: str | Path | None = None) -> dict[str, Segment]:
    """Read only the requested segments (by id) from the segment files."""
    wanted = set(ids)
    found: dict[str, Segment] = {}
    for path in _files(segments_dir):
        for s in read_jsonl(path, Segment):
            if s.segment_id in wanted:
                found[s.segment_id] = s
    missing = wanted - found.keys()
    if missing:
        raise KeyError(f"{len(missing)} segment ids not found, e.g. {sorted(missing)[:3]}")
    return found


def stratified_sample(refs: list[SegmentRef], n: int, seed: int = 42, allocation: str = "equal") -> list[SegmentRef]:
    """Sample n refs across sources. 'equal': same share per source (leftovers go to larger sources);
    'proportional': shares follow source sizes."""
    rng = random.Random(seed)
    by_source: dict[str, list[SegmentRef]] = defaultdict(list)
    for r in refs:
        by_source[r.source].append(r)
    for items in by_source.values():
        rng.shuffle(items)
    sources = sorted(by_source)
    n = min(n, len(refs))
    if allocation == "proportional":
        quota = {s: round(n * len(by_source[s]) / len(refs)) for s in sources}
    else:
        quota, left, open_ = {s: 0 for s in sources}, n, list(sources)
        while left > 0 and open_:
            share = max(left // len(open_), 1)
            for s in list(open_):
                take = min(share, len(by_source[s]) - quota[s], left)
                quota[s] += take
                left -= take
                if quota[s] >= len(by_source[s]):
                    open_.remove(s)
                if left == 0:
                    break
    chosen = [r for s in sources for r in by_source[s][: quota[s]]]
    rng.shuffle(chosen)
    return chosen


def select_gold(n: int = 400, seed: int = 42, segments_dir: str | Path | None = None,
                out: str | Path | None = None) -> list[str]:
    refs = stratified_sample(segment_refs(segments_dir), n, seed, "equal")
    ids = sorted(r.segment_id for r in refs)
    path = Path(out) if out else get_paths().labels / "gold_ids.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ids, indent=0), encoding="utf-8")
    return ids


def read_ids(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_pool(n: int = 10000, gold_ids: Iterable[str] = (), seed: int = 42, allocation: str = "equal",
                segments_dir: str | Path | None = None, out: str | Path | None = None) -> list[str]:
    refs = segment_refs(segments_dir)
    gold = set(gold_ids)
    gold_convs = {r.conversation_id for r in refs if r.segment_id in gold}
    eligible = [r for r in refs if r.conversation_id not in gold_convs]
    ids = sorted(r.segment_id for r in stratified_sample(eligible, n, seed, allocation))
    path = Path(out) if out else get_paths().labels / "pool_ids.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ids, indent=0), encoding="utf-8")
    return ids
