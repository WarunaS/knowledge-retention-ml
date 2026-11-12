"""MCP server: the successor asks from inside their own AI client instead of searching.

    claude mcp add krs -- python -m krs.app.mcp_server

Three tools over the same store and agent as the REST API: ask_predecessor, search_knowledge,
handover_pack. Uses the MCP Python SDK 2.x (`MCPServer`, formerly FastMCP). Thin by design (plan).
"""

from __future__ import annotations

from typing import Any

from krs.app.agent import ask, default_store
from krs.app.handover import build_pack
from krs.app.retrieval import search

INSTRUCTIONS = (
    "Knowledge left behind by colleagues who worked on these projects, extracted from their AI conversations. "
    "Use ask_predecessor for 'why/how did we...' questions and cite the card ids it returns. "
    "It describes knowledge, never people: do not use it to evaluate anyone."
)


def build_server(store_factory=default_store, client: Any = None):
    from mcp.server.mcpserver import MCPServer

    server = MCPServer("knowledge-retention", instructions=INSTRUCTIONS)

    def _filters(project_id: str | None, role: str | None, author_id: str | None) -> dict[str, Any]:
        return {k: v for k, v in {"project_id": project_id, "role": role, "author_id": author_id}.items() if v}

    @server.tool()
    def ask_predecessor(question: str, project_id: str | None = None, role: str | None = None,
                        author_id: str | None = None) -> dict[str, Any]:
        """Answer a question about past work from stored knowledge cards, with [card_id] citations."""
        r = ask(store_factory(), question, _filters(project_id, role, author_id), client=client)
        return {k: r[k] for k in ("answer", "citations", "unknown_citations", "backend", "note")}

    @server.tool()
    def search_knowledge(query: str, k: int = 5, project_id: str | None = None, role: str | None = None) -> list[dict[str, Any]]:
        """Find knowledge cards by meaning; optionally only one project or role."""
        hits = search(store_factory(), query, k, _filters(project_id, role, None))
        return [{"card_id": h.card.card_id, "title": h.card.title, "summary": h.card.summary,
                 "decision": h.card.decision, "score": h.score} for h in hits]

    @server.tool()
    def handover_pack(project_id: str | None = None, author_id: str | None = None) -> str:
        """Markdown handover pack (decisions, procedures, domain notes, context) for a project and/or person."""
        return build_pack(store_factory(), author_id=author_id, project_id=project_id)

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
