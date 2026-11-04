"""JSON Lines read/write for schema records. Every stage hands data to the next this way."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path, model: type[T]) -> Iterator[T]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield model.model_validate_json(line)
