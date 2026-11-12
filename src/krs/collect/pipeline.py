"""Stage 2 runner: load each configured dataset, write data/raw/<name>.jsonl, record the manifest.

    from krs.collect.pipeline import run_collect
    run_collect("collect")                    # all enabled public datasets
    run_collect("collect", only=["swe_agent"])

Own logs are collected separately and locally with `run_own_logs`, never on Colab.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import islice
from pathlib import Path
from typing import Any

from krs.collect import devgpt, manifest, own_logs, trajectories, wildchat
from krs.config import get_paths, load_config
from krs.io import write_jsonl
from krs.schema import Conversation


def unique(convs: Iterable[Conversation], counter: dict[str, int]) -> Iterator[Conversation]:
    """Keep the first record for each conversation id. Counts repeats in counter['duplicate_ids']."""
    seen: set[str] = set()
    counter.setdefault("duplicate_ids", 0)
    for c in convs:
        if c.conversation_id in seen:
            counter["duplicate_ids"] += 1
            continue
        seen.add(c.conversation_id)
        yield c


def _load(name: str, spec: dict[str, Any], raw_dir: Path) -> tuple[Iterator[Conversation], str, str | None]:
    """Return (conversations, source, version) for one dataset entry of collect.yaml."""
    limit = spec.get("limit")
    if name == "wildchat":
        convs = wildchat.load(
            limit,
            revision=spec.get("revision"),
            random_row_groups=spec.get("random_row_groups", True),
            seed=spec.get("seed", 42),
        )
        return convs, spec["hf_id"], spec.get("revision")
    if name == "devgpt":
        snapshot_dir = devgpt.download(raw_dir / "devgpt", spec.get("snapshot", devgpt.DEFAULT_SNAPSHOT), spec["url"])
        return islice(devgpt.load(snapshot_dir), limit), spec["url"], spec.get("snapshot")
    if name in trajectories.HF_IDS:
        convs = trajectories.load(
            name,
            limit,
            revision=spec.get("revision"),
            random_row_groups=spec.get("random_row_groups", True),
            seed=spec.get("seed", 42),
            rows_per_group=spec.get("rows_per_group"),
        )
        return convs, spec["hf_id"], spec.get("revision")
    raise ValueError(f"no loader for dataset {name!r}")


def run_collect(config: str = "collect", only: list[str] | None = None) -> dict[str, int]:
    """Collect every enabled dataset in the config. Returns rows written per dataset."""
    cfg = load_config(config)
    paths = get_paths()
    written: dict[str, int] = {}
    for name, spec in (cfg.get("datasets") or {}).items():
        if (only and name not in only) or not spec.get("enabled", True):
            continue
        convs, source, version = _load(name, spec, paths.raw)
        counter: dict[str, int] = {}
        n = write_jsonl(paths.raw / f"{name}.jsonl", unique(convs, counter))
        manifest.record(
            name,
            source=source,
            version=version,
            licence=spec["licence"],
            n_rows=n,
            extra={"limit": spec.get("limit"), "duplicate_ids_skipped": counter["duplicate_ids"]},
        )
        written[name] = n
        print(f"{name}: {n} conversations ({counter['duplicate_ids']} repeated ids skipped)")
    return written


def run_own_logs(consent_path: str | Path, out: str | Path | None = None) -> int:
    """Collect consented own logs LOCALLY to data/raw/own_logs.jsonl (gitignored). Redact before use."""
    out_path = Path(out) if out else get_paths().raw / "own_logs.jsonl"
    counter: dict[str, int] = {}
    n = write_jsonl(out_path, unique(own_logs.load(consent_path), counter))
    manifest.record(
        "own_logs",
        source="consented team logs (claude_code)",
        licence=own_logs.LICENCE,
        n_rows=n,
        extra={"redacted": False},
    )
    return n
