"""Stage 3 runner: raw conversations -> redact (own logs) -> normalise -> filters -> dedup -> segments.

    from krs.clean.pipeline import run_cleaning
    report = run_cleaning("clean")

Reads data/raw/<source>.jsonl for every source that exists, writes data/filtered/<source>.jsonl
(kept conversations) and data/segments/<source>.jsonl, and data/filtered/cleaning_report.json with
the count after every step per source. Those counts go in the report.

Everything streams one conversation at a time; only the dedup fingerprints stay in memory.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from krs.clean.dedup import NearDuplicateFilter
from krs.clean.filters import drop_reason
from krs.clean.normalise import normalise
from krs.clean.redact import load_extra_terms, redact_conversation
from krs.clean.segment import segment
from krs.config import REPO_ROOT, get_paths, load_config
from krs.io import read_jsonl
from krs.schema import Conversation

SOURCES = ("own_logs", "devgpt", "swe_agent", "openhands", "wildchat")  # dedup keeps the first seen


def run_cleaning(config: str | Path = "clean", sources: list[str] | None = None) -> dict[str, dict[str, int]]:
    cfg = load_config(config)
    paths = get_paths()
    paths.filtered.mkdir(parents=True, exist_ok=True)
    paths.segments.mkdir(parents=True, exist_ok=True)

    red_cfg: dict[str, Any] = cfg.get("redact") or {}
    redact_sources = set(red_cfg.get("sources", ["own_logs"]))
    terms_file = red_cfg.get("extra_terms_file")
    if terms_file and not Path(terms_file).is_absolute():
        terms_file = REPO_ROOT / terms_file
    extra_terms = load_extra_terms(terms_file)
    norm_cfg = cfg.get("normalise") or {}
    seg_cfg = cfg.get("segment") or {}
    dd = cfg.get("dedup") or {}
    dedup = NearDuplicateFilter(
        threshold=dd.get("threshold", 0.8),
        num_perm=dd.get("minhash_num_perm", 128),
        shingle_words=dd.get("shingle_words", 5),
        max_chars=dd.get("max_chars", 5000),
    )

    report: dict[str, dict[str, int]] = {}
    for name in sources or SOURCES:
        raw = paths.raw / f"{name}.jsonl"
        if not raw.exists():
            continue
        counts: Counter = Counter()
        with open(paths.filtered / f"{name}.jsonl", "w", encoding="utf-8") as f_conv, \
             open(paths.segments / f"{name}.jsonl", "w", encoding="utf-8") as f_seg:
            for conv in read_jsonl(raw, Conversation):
                counts["input"] += 1
                if name in redact_sources:
                    conv, red = redact_conversation(conv, extra_terms)
                    counts["redactions"] += sum(red.values())
                conv = normalise(conv, norm_cfg.get("tool_output_max_chars", 2000))
                reason = drop_reason(conv, cfg, name)
                if reason:
                    counts[f"dropped_{reason}"] += 1
                    continue
                if dedup.is_duplicate(conv):
                    counts["dropped_near_duplicate"] += 1
                    continue
                counts["kept"] += 1
                f_conv.write(conv.model_dump_json() + "\n")
                for seg in segment(
                    conv,
                    context_pairs=seg_cfg.get("context_pairs", 1),
                    max_tokens=seg_cfg.get("max_tokens", 512),
                    chars_per_token=seg_cfg.get("chars_per_token", 4),
                    max_segments=seg_cfg.get("max_segments_per_conversation", 10),
                ):
                    counts["segments"] += 1
                    f_seg.write(seg.model_dump_json() + "\n")
        report[name] = dict(counts)
        print(f"{name}: {dict(counts)}")

    with open(paths.filtered / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
