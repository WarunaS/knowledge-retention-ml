"""Handover pack generator: everything a successor needs about one person or one project.

Sections, each entry linked to its source card and turns:
    decision log (decision cards, ordered by occurred_at) · procedures · domain notes ·
    context and constraints
"""

from __future__ import annotations


def build_pack(cards_path: str, *, author_id: str | None = None, project_id: str | None = None):
    """Select cards for one person or project and return the pack as Markdown."""
    raise NotImplementedError
