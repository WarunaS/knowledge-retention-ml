"""WildChat-1M loader (Zhao et al., 2024).

Real ChatGPT conversations, 837,989 rows, licence ODC-BY, not gated on Hugging Face.
Loaders never filter: language, toxicity, turn count and relevance are Stage 3.

Privacy: each WildChat turn carries `hashed_ip`, `country`, `state` and the browser `header`.
These can point towards a person, so they are never copied. No author is ever assigned.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from krs.collect._util import build_turns, to_utc
from krs.schema import Conversation, Modality, Role, SourceDataset

HF_ID = "allenai/WildChat-1M"
LICENCE = "ODC-BY-1.0"

_ROLES = {"user": Role.USER, "assistant": Role.ASSISTANT}
# WildChat stores language names ("English"); Stage 3 filters on ISO codes.
_LANG = {"English": "en"}


def normalise_language(name: str | None) -> str | None:
    if not name:
        return None
    return _LANG.get(name, name.lower())


def to_conversation(row: dict[str, Any]) -> Conversation:
    """Map one WildChat row to a Conversation."""
    turns = build_turns(
        [
            (_ROLES[m["role"]], m.get("content") or "", None)
            for m in row.get("conversation") or []
            if m.get("role") in _ROLES
        ]
    )
    return Conversation(
        conversation_id=f"wildchat:{row['conversation_hash']}",
        source_dataset=SourceDataset.WILDCHAT,
        licence=LICENCE,
        modality=Modality.CONVERSATIONAL,
        turns=turns,
        language=normalise_language(row.get("language")),
        toxic=bool(row.get("toxic")) if row.get("toxic") is not None else None,
        occurred_at=to_utc(row.get("timestamp")),
        metadata={
            "model": row.get("model"),
            "redacted": bool(row.get("redacted")),
            "n_exchanges_reported": row.get("turn"),
        },
    )


def load(
    limit: int | None = None,
    *,
    revision: str | None = None,
    streaming: bool = True,
    shuffle_buffer: int = 0,
    seed: int = 42,
) -> Iterator[Conversation]:
    """Stream Conversations from Hugging Face.

    `revision` pins the dataset commit so the run is reproducible. `shuffle_buffer` > 0 takes an
    approximately random sample instead of the first `limit` rows (which are ordered by time).
    """
    from datasets import load_dataset

    ds = load_dataset(HF_ID, split="train", streaming=streaming, revision=revision)
    if shuffle_buffer:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)
    for i, row in enumerate(ds):
        if limit is not None and i >= limit:
            break
        yield to_conversation(row)
