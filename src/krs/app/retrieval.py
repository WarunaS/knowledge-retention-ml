"""Retrieval: semantic search over knowledge cards, filtered by label, person, role, project and date,
with optional one-hop graph expansion (cards sharing a tool, or next in the same conversation)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from krs.app.storage import KnowledgeStore
from krs.schema import KnowledgeCard

EXPANSION_WEIGHT = 0.5  # an expanded card scores half of the card that led to it


@dataclass
class Hit:
    card: KnowledgeCard
    score: float
    via: str  # "semantic" or "graph:<card_id>"

    def citation(self) -> dict[str, Any]:
        c = self.card
        return {"card_id": c.card_id, "conversation_id": c.conversation_id, "segments": c.source_segment_ids,
                "occurred_at": c.occurred_at.isoformat() if c.occurred_at else None}


def _matches(c: KnowledgeCard, f: dict[str, Any]) -> bool:
    if f.get("domains") and not {d.value for d in c.domains} & set(f["domains"]):
        return False
    if f.get("knowledge_types") and not {k.value for k in c.knowledge_types} & set(f["knowledge_types"]):
        return False
    for key in ("author_id", "project_id"):
        if f.get(key) and getattr(c, key) != f[key]:
            return False
    if f.get("role") and (c.role is None or c.role.value != f["role"]):
        return False
    for key, cmp in (("since", lambda a, b: a >= b), ("until", lambda a, b: a <= b)):
        if f.get(key):
            bound = datetime.fromisoformat(str(f[key]))
            if c.occurred_at is None or not cmp(c.occurred_at.replace(tzinfo=None), bound.replace(tzinfo=None)):
                return False
    return True


def search(store: KnowledgeStore, query: str, k: int = 5, filters: dict[str, Any] | None = None,
           expand: bool = False, max_expanded: int | None = None) -> list[Hit]:
    """Top-k cards for the query among cards passing the filters; expanded neighbours appended after."""
    filters = filters or {}
    cards = {c.card_id: c for c in store.all() if _matches(c, filters)}
    if not cards:
        return []
    sims = store.similarity(query)
    ranked = sorted((cid for cid in cards if cid in sims), key=lambda cid: -sims[cid])[:k]
    hits = [Hit(cards[cid], round(sims[cid], 4), "semantic") for cid in ranked]
    if expand:
        seen = set(ranked)
        extra: list[Hit] = []
        for h in list(hits):
            for n in sorted(store.neighbours(h.card.card_id), key=lambda cid: -sims.get(cid, 0.0)):
                if n in cards and n not in seen:
                    seen.add(n)
                    extra.append(Hit(cards[n], round(h.score * EXPANSION_WEIGHT, 4), f"graph:{h.card.card_id}"))
        hits += extra[: max_expanded if max_expanded is not None else k]  # at most k extra cards by default
    return hits
