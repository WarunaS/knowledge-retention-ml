"""Handover pack generator: everything a successor needs about one person or one project.

Sections, each entry linked to its card and source conversation:
    Decision log (decision cards, oldest first) · Procedures · Domain notes · Context and constraints
Describes knowledge, not people: no counts or scores per person.
"""

from __future__ import annotations

from krs.app.storage import KnowledgeStore
from krs.label.taxonomy import KnowledgeType
from krs.schema import KnowledgeCard

SECTIONS = (
    (KnowledgeType.DECISION, "Decision log"),
    (KnowledgeType.PROCEDURAL, "Procedures"),
    (KnowledgeType.DOMAIN, "Domain notes"),
    (KnowledgeType.CONTEXTUAL, "Context and constraints"),
)


def _when(c: KnowledgeCard) -> str:
    return c.occurred_at.date().isoformat() if c.occurred_at else "undated"


def _entry(c: KnowledgeCard) -> str:
    lines = [f"- **{c.title}** ({_when(c)}; {', '.join(d.value for d in c.domains) or 'no domain'})", f"  {c.summary}"]
    if c.decision:
        lines.append(f"  Decision: {c.decision}")
    if c.tools_involved:
        lines.append(f"  Tools: {', '.join(c.tools_involved)}")
    lines.append(f"  Source: card `{c.card_id}`, conversation `{c.conversation_id}`, segments {', '.join(c.source_segment_ids)}")
    return "\n".join(lines)


def build_pack(store: KnowledgeStore, author_id: str | None = None, project_id: str | None = None) -> str:
    """Markdown handover pack for one person and/or one project (at least one must be given)."""
    if not author_id and not project_id:
        raise ValueError("give an author_id, a project_id, or both")
    cards = [c for c in store.all()
             if (not author_id or c.author_id == author_id) and (not project_id or c.project_id == project_id)]
    who = " · ".join(filter(None, [f"project `{project_id}`" if project_id else "", f"person `{author_id}`" if author_id else ""]))
    out = [f"# Handover pack - {who}", "", f"{len(cards)} knowledge cards. Every entry links to its source.", ""]
    for kt, title in SECTIONS:
        items = sorted((c for c in cards if kt in c.knowledge_types), key=lambda c: (c.occurred_at is None, c.occurred_at, c.card_id))
        out += [f"## {title} ({len(items)})", ""]
        out += [_entry(c) for c in items] or ["_None recorded._"]
        out.append("")
    return "\n".join(out)
