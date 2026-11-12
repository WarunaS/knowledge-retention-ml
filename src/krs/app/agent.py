"""Ask-predecessor agent and its REST API.

    uvicorn krs.app.agent:app          # http://localhost:8000/docs

`ask()` retrieves cards for a question and answers ONLY from them, citing [card_id]s. Citations are
checked against the retrieved cards; unknown ones are reported. Two backends:
- claude: a grounded written answer (public-source cards only)
- local: no LLM - lists the most relevant cards with their summaries and citations
If any retrieved card comes from own logs, the claude backend is not used (privacy rule) and the
answer says so.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from krs.app.common import app_config, claude_client, claude_text, cloud_allowed
from krs.app.handover import build_pack
from krs.app.retrieval import Hit, search
from krs.app.storage import KnowledgeStore

SYSTEM = (
    "You help a successor understand work done by a colleague who has left. Answer the question using ONLY the "
    "knowledge cards provided. After every claim, cite the card it comes from as [card_id]. If the cards do not "
    "answer the question, say so plainly and do not guess. Keep the answer short."
)
_CITE = re.compile(r"\[(k[0-9a-f]{12})\]")


def render_cards(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        c = h.card
        blocks.append("\n".join(filter(None, [
            f"[{c.card_id}] {c.title}",
            (f"date: {c.occurred_at.date().isoformat() if c.occurred_at else 'unknown'}; project: {c.project_id or '-'}; "
            f"role: {c.role.value if c.role else '-'}; types: {', '.join(k.value for k in c.knowledge_types)}"),
            f"summary: {c.summary}",
            f"decision: {c.decision}" if c.decision else "",
            f"evidence: \"{c.evidence}\"",
        ])))
    return "\n\n".join(blocks)


def local_answer(hits: list[Hit]) -> str:
    if not hits:
        return "No stored knowledge matches this question."
    lines = ["Most relevant knowledge (no generated text; read the cards):"]
    for h in hits:
        c = h.card
        lines.append(f"- {c.title}: {c.summary}" + (f" Decision: {c.decision}" if c.decision else "") + f" [{c.card_id}]")
    return "\n".join(lines)


def ask(store: KnowledgeStore, question: str, filters: dict[str, Any] | None = None, k: int | None = None,
        cfg: dict[str, Any] | None = None, client: Any = None) -> dict[str, Any]:
    cfg = cfg or app_config()
    rcfg, acfg = cfg.get("retrieval") or {}, cfg.get("answer") or {}
    hits = search(store, question, k or rcfg.get("k", 5), filters, rcfg.get("expand", True))
    backend, note = acfg.get("backend", "claude"), None
    if backend == "claude" and any(not cloud_allowed(h.card.conversation_id, cfg) for h in hits):
        backend, note = "local", "Some cards come from own logs, which are never sent to a cloud model; answered locally."
    if backend == "claude" and hits:
        text, msg = claude_text(client or claude_client(), acfg, SYSTEM,
                                f"Knowledge cards:\n\n{render_cards(hits)}\n\nQuestion: {question}")
        model = getattr(msg, "model", acfg.get("model"))
    else:
        text, model = local_answer(hits), None
    retrieved = {h.card.card_id for h in hits}
    cited = list(dict.fromkeys(_CITE.findall(text)))
    return {
        "question": question,
        "answer": text,
        "backend": backend if hits else "none",
        "model": model,
        "note": note,
        "citations": [h.citation() for h in hits if h.card.card_id in cited],
        "unknown_citations": [c for c in cited if c not in retrieved],
        "cards": [{"card_id": h.card.card_id, "title": h.card.title, "score": h.score, "via": h.via} for h in hits],
    }


# --------------------------------------------------------------------------- REST API


# Request bodies live at module level: FastAPI cannot resolve postponed annotations of classes
# defined inside a function (it would treat the body as query parameters).
class Filters(BaseModel):
    domains: list[str] | None = None
    knowledge_types: list[str] | None = None
    author_id: str | None = None
    role: str | None = None
    project_id: str | None = None
    since: str | None = None
    until: str | None = None


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    filters: Filters = Field(default_factory=Filters)
    expand: bool = False


class AskRequest(BaseModel):
    question: str
    k: int | None = None
    filters: Filters = Field(default_factory=Filters)


@lru_cache(maxsize=1)
def default_store() -> KnowledgeStore:
    return KnowledgeStore()


def create_app(store_factory=default_store, client: Any = None):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import PlainTextResponse

    api = FastAPI(title="Knowledge Retention - ask your predecessor",
                  description="Answers from stored knowledge cards, with citations. Never ranks or scores people.")

    @api.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", **store_factory().stats()}

    @api.post("/search")
    def do_search(req: SearchRequest) -> list[dict[str, Any]]:
        hits = search(store_factory(), req.query, req.k, req.filters.model_dump(exclude_none=True), req.expand)
        return [{"card": h.card.model_dump(mode="json"), "score": h.score, "via": h.via} for h in hits]

    @api.post("/ask")
    def do_ask(req: AskRequest) -> dict[str, Any]:
        return ask(store_factory(), req.question, req.filters.model_dump(exclude_none=True), req.k, client=client)

    @api.get("/cards/{card_id}")
    def get_card(card_id: str) -> dict[str, Any]:
        card = store_factory().get(card_id)
        if card is None:
            raise HTTPException(status_code=404, detail="card not found")
        return card.model_dump(mode="json")

    @api.get("/handover", response_class=PlainTextResponse)
    def handover(project_id: str | None = None, author_id: str | None = None) -> str:
        try:
            return build_pack(store_factory(), author_id=author_id, project_id=project_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return api


app = create_app()
