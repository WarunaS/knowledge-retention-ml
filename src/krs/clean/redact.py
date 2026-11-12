"""Redaction of secrets and personal data. Applied to consented own logs before anything else.

Rule-based and conservative: it replaces what it can recognise and counts each replacement.
- secrets: private keys, JWTs, known API-key formats, bearer tokens, password/key assignments,
  credentials inside URLs
- personal data: email addresses, IPv4 addresses, phone numbers written with a + prefix,
  user names inside home-directory paths
- extra terms: names of people, clients or internal systems listed by the team (one per line in a
  gitignored file), because rules cannot recognise names

Limitation: names not on the extra-terms list are not caught. A human skim of the redacted
transfer-test segments is still required before labelling.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from krs.schema import Conversation, Turn

_RULES: list[tuple[str, re.Pattern[str], str]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL), "[PRIVATE_KEY]"),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "[TOKEN]"),
    ("api_key", re.compile(
        r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
        r"|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{35}|xox[abprs]-[A-Za-z0-9-]{10,}|hf_[A-Za-z0-9]{30,})"), "[SECRET]"),
    ("bearer", re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/-]{16,}=*"), r"\1 [SECRET]"),
    ("assignment", re.compile(
        r"(?i)\b((?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)"
        r"\s*[:=]\s*)(['\"]?)[^\s'\",;]{4,}\2"), r"\1\2[SECRET]\2"),
    ("url_credentials", re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://)[^\s:/@]+:[^\s@/]+@"), r"\1[CREDENTIALS]@"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    ("ipv4", re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), "[IP]"),
    ("phone", re.compile(r"(?<!\w)\+\d[\d\s().-]{7,}\d"), "[PHONE]"),
    ("home_path", re.compile(r"(/Users/|/home/|C:\\Users\\)[^/\\\s]+"), r"\1[USER]"),
]


def load_extra_terms(path: str | Path | None) -> list[str]:
    if not path or not Path(path).exists():
        return []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [t.strip() for t in lines if t.strip() and not t.startswith("#")]


def redact_text(text: str, extra_terms: list[str] | None = None) -> tuple[str, Counter]:
    counts: Counter = Counter()
    for name, pattern, repl in _RULES:
        text, n = pattern.subn(repl, text)
        counts[name] += n
    for term in sorted(extra_terms or [], key=len, reverse=True):
        text, n = re.subn(rf"(?i)(?<!\w){re.escape(term)}(?!\w)", "[NAME]", text)
        counts["extra_term"] += n
    return text, +counts


def redact_conversation(conv: Conversation, extra_terms: list[str] | None = None) -> tuple[Conversation, Counter]:
    total: Counter = Counter()
    turns = []
    for t in conv.turns:
        content, c = redact_text(t.content, extra_terms)
        total += c
        turns.append(Turn(index=t.index, role=t.role, content=content, tool_name=t.tool_name))
    meta = {**conv.metadata, "redacted": True, "redactions": dict(total)}
    return conv.model_copy(update={"turns": turns, "metadata": meta}), total
