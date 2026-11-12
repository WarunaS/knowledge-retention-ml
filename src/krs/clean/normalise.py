"""Text normalisation.

- Unicode NFC (not NFKC, which would change characters inside code), line endings to \\n
- zero-width and control characters removed, trailing spaces trimmed, 3+ blank lines -> 1 blank line
- indentation and code blocks kept as they are
- system turns dropped; long tool outputs shortened to head + tail
- turns renumbered 0..n-1 (segments refer to these numbers)
"""

from __future__ import annotations

import re
import unicodedata

from krs.schema import Conversation, Role, Turn

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\u2060\ufeff]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_BLANK_LINES = re.compile(r"\n{3,}")


def normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL.sub("", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANK_LINES.sub("\n\n", text).strip()


def shorten(text: str, max_chars: int) -> str:
    """Keep the first two thirds and the last third of `max_chars`, marking what was cut."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    removed = len(text) - max_chars
    return f"{text[:head]}\n[... {removed} characters removed ...]\n{text[-tail:]}"


def normalise(conv: Conversation, tool_output_max_chars: int = 2000) -> Conversation:
    turns: list[Turn] = []
    for t in conv.turns:
        if t.role == Role.SYSTEM:
            continue
        content = normalise_text(t.content)
        if t.role == Role.TOOL:
            content = shorten(content, tool_output_max_chars)
        if not content:
            continue
        turns.append(Turn(index=len(turns), role=t.role, content=content, tool_name=t.tool_name))
    return conv.model_copy(update={"turns": turns})
