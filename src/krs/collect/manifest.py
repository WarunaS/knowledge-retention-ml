"""Dataset manifest: what was collected, from where, which version, under which licence, how many rows.

Written to data/raw/manifest.json. One entry per dataset name; re-collecting replaces the entry.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from krs.config import get_paths


def read(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else get_paths().manifest
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def record(
    name: str,
    *,
    source: str,
    licence: str,
    n_rows: int,
    version: str | None = None,
    extra: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> dict[str, Any]:
    """Add or replace the manifest entry for `name` and return it."""
    p = Path(path) if path else get_paths().manifest
    data = read(p)
    entry = {
        "source": source,
        "version": version,
        "licence": licence,
        "n_rows": n_rows,
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **(extra or {}),
    }
    data[name] = entry
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(dict(sorted(data.items())), f, indent=2)
    return entry
