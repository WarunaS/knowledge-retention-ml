"""Read rows from a Hugging Face dataset's parquet files at a pinned revision.

Why not `datasets` streaming: with datasets 5.x, stopping a stream early (as every `limit` does)
leaves pyarrow's IO thread pool waiting on an fsspec read, and the Python process hangs at exit
(seen on 11 Nov 2026 with pyarrow 21 and 26). Reading the parquet files through HfFileSystem
with plain pyarrow fetches only the row groups needed and exits cleanly.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

BATCH_SIZE = 256


def _fs():
    from huggingface_hub import HfFileSystem

    return HfFileSystem()


def parquet_files(repo_id: str, revision: str | None = None) -> list[str]:
    """Full HfFileSystem paths of every parquet file in the dataset, in name order."""
    root = f"datasets/{repo_id}@{revision or 'main'}"
    return sorted(_fs().glob(f"{root}/**/*.parquet"))


def iter_rows(
    repo_id: str,
    revision: str | None = None,
    limit: int | None = None,
    *,
    random_row_groups: bool = False,
    seed: int = 42,
    rows_per_group: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield rows as dicts.

    In file order by default. With `random_row_groups`, row groups from all files are visited in a
    seeded random order, giving a reproducible sample spread across the whole dataset.
    With `rows_per_group`, only that many seeded-random rows are kept from each row group, so a
    sample spreads over many groups (needed when files are sorted, e.g. by repository).
    """
    import pyarrow.parquet as pq

    fs = _fs()
    files = parquet_files(repo_id, revision)
    if not files:
        raise FileNotFoundError(f"no parquet files in {repo_id}@{revision or 'main'}")

    rng = random.Random(seed)
    if random_row_groups:
        units: list[tuple[str, int]] = []
        for path in files:
            with fs.open(path, "rb") as f:
                units += [(path, rg) for rg in range(pq.ParquetFile(f).num_row_groups)]
        rng.shuffle(units)
    else:
        units = [(path, -1) for path in files]  # -1 = all row groups of the file

    n = 0
    for path, rg in units:
        if limit is not None and n >= limit:
            return
        with fs.open(path, "rb", block_size=8 * 2**20) as f:
            pf = pq.ParquetFile(f)
            groups = list(range(pf.num_row_groups)) if rg < 0 else [rg]
            for g in groups:
                keep: set[int] | None = None
                if rows_per_group:
                    size = pf.metadata.row_group(g).num_rows
                    keep = set(rng.sample(range(size), min(rows_per_group, size)))
                last = max(keep) if keep else -1
                pos = 0
                for batch in pf.iter_batches(batch_size=BATCH_SIZE, row_groups=[g]):
                    if keep is not None:
                        local = sorted(i - pos for i in keep if pos <= i < pos + batch.num_rows)
                        pos += batch.num_rows
                        rows = batch.take(local).to_pylist() if local else []  # decode chosen rows only
                    else:
                        rows = batch.to_pylist()
                    for row in rows:
                        if limit is not None and n >= limit:
                            return
                        n += 1
                        yield row
                    if keep is not None and pos > last:
                        break  # the rest of this row group holds no chosen rows
