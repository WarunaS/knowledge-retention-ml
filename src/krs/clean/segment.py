"""Cut normalised conversations into Segments, the unit of labelling and classification.

An exchange is a prompting turn (USER, or TOOL output in agent runs) followed by the ASSISTANT
turns that answer it. A segment is one exchange plus the previous exchange as context. In a chat
this is the user-assistant pair; in an agent run it is one step (observation -> agent action),
so long runs give many segments instead of one.

Agent runs can have hundreds of steps; at most `max_segments` evenly spaced exchanges are kept
per conversation so they do not swamp the chat data. Lengths are capped in characters
(max_tokens * chars_per_token) to avoid needing a tokenizer here.
"""

from __future__ import annotations

from collections.abc import Iterator

from krs.clean.normalise import shorten
from krs.schema import Conversation, Role, Segment, Turn


def exchanges(turns: list[Turn]) -> list[tuple[int, int]]:
    """(start, end) turn indices, inclusive, of every prompt + assistant-answer exchange."""
    out, i, n = [], 0, len(turns)
    while i < n:
        if turns[i].role in (Role.USER, Role.TOOL):
            j = i + 1
            while j < n and turns[j].role == Role.ASSISTANT:
                j += 1
            if j > i + 1:
                out.append((i, j - 1))
            i = j
        else:
            i += 1
    return out


def render(turns: list[Turn]) -> str:
    lines = []
    for t in turns:
        label = f"TOOL ({t.tool_name})" if t.role == Role.TOOL and t.tool_name else t.role.value.upper()
        lines.append(f"{label}: {t.content}")
    return "\n".join(lines)


def _spread(n: int, k: int) -> list[int]:
    if not k or n <= k:
        return list(range(n))
    if k == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def segment(
    conv: Conversation,
    context_pairs: int = 1,
    max_tokens: int = 512,
    chars_per_token: int = 4,
    max_segments: int | None = 10,
) -> Iterator[Segment]:
    turns = conv.turns
    ex = exchanges(turns)
    max_chars = max_tokens * chars_per_token
    for k in _spread(len(ex), max_segments or 0):
        start, end = ex[k]
        ctx = ex[max(0, k - context_pairs) : k]
        context = "\n".join(render(turns[a : b + 1]) for a, b in ctx)
        if len(context) > max_chars // 2:
            context = "[...] " + context[-(max_chars // 2) :]
        yield Segment(
            segment_id=f"{conv.conversation_id}:{start}",
            conversation_id=conv.conversation_id,
            source_dataset=conv.source_dataset,
            modality=conv.modality,
            turn_span=(ctx[0][0] if ctx else start, end),
            context_text=context,
            text=shorten(render(turns[start : end + 1]), max_chars),
            author_id=conv.author_id,
            role=conv.role,
            project_id=conv.project_id,
            occurred_at=conv.occurred_at,
        )
