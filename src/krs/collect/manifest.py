"""Dataset manifest: name, version/revision, download date, licence, row counts."""

from __future__ import annotations


def record(name: str, version: str, licence: str, n_rows: int):
    """Append or update an entry in data/raw/manifest.json."""
    raise NotImplementedError
