"""Knowledge repository: SQLite (cards, labels, provenance) + vector index + lightweight graph.

- SQLite (`cards.db`): every card as JSON, plus filter columns (project, person, role, date, source)
- vectors (`vectors.npy` + `vector_ids.json`): normalised sentence embeddings of each card; exact
  cosine search with numpy - at thousands of cards this is fast and needs no vector database
- graph (`graph.json`): card nodes linked to project, tool, domain and person nodes, and to the next
  card in the same conversation (NetworkX, in-process)

`delete_person(author_id)` removes every card of one person from all three (GDPR erasure, no retraining).
"""

from __future__ import annotations

import itertools
import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from krs.app.common import Embedder, app_config, source_of
from krs.config import get_paths
from krs.schema import KnowledgeCard


def card_text(c: KnowledgeCard) -> str:
    """What gets embedded: the card's own words."""
    return " ".join(filter(None, [c.title, c.summary, c.decision or "", " ".join(c.tools_involved)]))


class KnowledgeStore:
    def __init__(self, path: str | Path | None = None, embedder: Embedder | None = None):
        self.path = Path(path) if path else get_paths().store
        self.path.mkdir(parents=True, exist_ok=True)
        cfg = app_config().get("retrieval") or {}
        self.embedder = embedder or Embedder(cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
        self.db = sqlite3.connect(self.path / "cards.db", check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS cards (
            card_id TEXT PRIMARY KEY, json TEXT NOT NULL, conversation_id TEXT, source TEXT,
            project_id TEXT, author_id TEXT, role TEXT, occurred_at TEXT)""")
        self.db.commit()
        self._load_vectors()
        self._graph = None

    # ------------------------------------------------------------------ cards

    def add(self, cards: Iterable[KnowledgeCard]) -> int:
        cards = list(cards)
        if not cards:
            return 0
        self.db.executemany(
            "INSERT OR REPLACE INTO cards VALUES (?,?,?,?,?,?,?,?)",
            [(c.card_id, c.model_dump_json(), c.conversation_id, source_of(c.conversation_id), c.project_id,
              c.author_id, c.role.value if c.role else None, c.occurred_at.isoformat() if c.occurred_at else None)
             for c in cards])
        self.db.commit()
        new = self.embedder.encode([card_text(c) for c in cards])
        keep = [i for i, cid in enumerate(self.ids) if cid not in {c.card_id for c in cards}]
        self.vectors = np.vstack([self.vectors[keep], new]) if len(keep) else new
        self.ids = [self.ids[i] for i in keep] + [c.card_id for c in cards]
        self._save_vectors()
        self._graph = None
        self.save_graph()
        return len(cards)

    def get(self, card_id: str) -> KnowledgeCard | None:
        row = self.db.execute("SELECT json FROM cards WHERE card_id=?", (card_id,)).fetchone()
        return KnowledgeCard.model_validate_json(row[0]) if row else None

    def all(self) -> list[KnowledgeCard]:
        return [KnowledgeCard.model_validate_json(r[0]) for r in self.db.execute("SELECT json FROM cards ORDER BY card_id")]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM cards").fetchone()[0]

    def delete_person(self, author_id: str) -> int:
        """Remove every card of one person from the database, the vectors and the graph."""
        gone = [r[0] for r in self.db.execute("SELECT card_id FROM cards WHERE author_id=?", (author_id,))]
        self.db.execute("DELETE FROM cards WHERE author_id=?", (author_id,))
        self.db.commit()
        keep = [i for i, cid in enumerate(self.ids) if cid not in set(gone)]
        self.vectors = self.vectors[keep] if len(self.ids) else self.vectors
        self.ids = [self.ids[i] for i in keep]
        self._save_vectors()
        self._graph = None
        self.save_graph()
        return len(gone)

    # ------------------------------------------------------------------ vectors

    def _load_vectors(self) -> None:
        vpath, ipath = self.path / "vectors.npy", self.path / "vector_ids.json"
        if vpath.exists() and ipath.exists():
            self.vectors = np.load(vpath)
            self.ids = json.loads(ipath.read_text(encoding="utf-8"))
        else:
            self.vectors, self.ids = np.zeros((0, 0), dtype=np.float32), []

    def _save_vectors(self) -> None:
        np.save(self.path / "vectors.npy", self.vectors)
        (self.path / "vector_ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    def similarity(self, query: str) -> dict[str, float]:
        if not self.ids:
            return {}
        q = self.embedder.encode([query])[0]
        return dict(zip(self.ids, (self.vectors @ q).tolist(), strict=True))

    # ------------------------------------------------------------------ graph

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.build_graph()
        return self._graph

    def build_graph(self):
        import networkx as nx

        g = nx.Graph()
        by_conv: dict[str, list[KnowledgeCard]] = {}
        for c in self.all():
            g.add_node(c.card_id, kind="card")
            for kind, values in (("project", [c.project_id]), ("tool", [t.lower() for t in c.tools_involved]),
                                 ("domain", [d.value for d in c.domains]), ("person", [c.author_id])):
                for v in values:
                    if v:
                        g.add_node(f"{kind}:{v}", kind=kind)
                        g.add_edge(c.card_id, f"{kind}:{v}", rel=kind)
            by_conv.setdefault(c.conversation_id, []).append(c)
        for cards in by_conv.values():
            cards.sort(key=lambda c: c.source_segment_ids[0])
            for a, b in itertools.pairwise(cards):
                g.add_edge(a.card_id, b.card_id, rel="follows")
        return g

    def save_graph(self) -> None:
        from networkx.readwrite import json_graph

        (self.path / "graph.json").write_text(json.dumps(json_graph.node_link_data(self.graph, edges="edges")),
                                              encoding="utf-8")

    def neighbours(self, card_id: str, via: tuple[str, ...] = ("tool", "follows"), hub_limit: int = 10) -> set[str]:
        """Cards one hop away: next/previous in the same conversation, or sharing a tool node. Tools linked to
        more than `hub_limit` cards (e.g. 'bash') say nothing specific and are skipped."""
        g = self.graph
        if card_id not in g:
            return set()
        out = set()
        for n in g.neighbors(card_id):
            rel = g.edges[card_id, n].get("rel")
            if rel == "follows" and "follows" in via:
                out.add(n)
            elif rel in via and g.nodes[n].get("kind") == rel and g.degree(n) <= hub_limit:
                out.update(m for m in g.neighbors(n) if m != card_id and g.nodes[m].get("kind") == "card")
        return out

    def stats(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for _, d in self.graph.nodes(data=True):
            kinds[d.get("kind", "?")] = kinds.get(d.get("kind", "?"), 0) + 1
        return {"cards": self.count(), "vectors": len(self.ids), "graph_nodes": kinds, "graph_edges": self.graph.number_of_edges()}
