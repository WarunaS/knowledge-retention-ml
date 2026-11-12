"""Stage 3 filters: language, toxicity, minimum exchanges, relevance.

Relevance uses keyword rules (decided 11 Nov 2026): fast, free, runs on a laptop. A conversation is
relevant if it is about programming or about requirements (business-analyst work). Precision is
measured by hand on 200 kept conversations (see precision_check.py); a zero-shot classifier is added
only if that precision is too low.

Relevance and the minimum-exchanges rule are applied to WildChat only (technical plan, data sources
table). DevGPT shares, agent trajectories and own logs are developer work by construction, and a
single-question DevGPT share can still carry knowledge. Every conversation still gets metadata["relevance"] for later analysis.
"""

from __future__ import annotations

import re
from typing import Any

from krs.schema import Conversation, Modality, Role

_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_LETTERS = re.compile(r"[^\W\d_]")

PROGRAMMING_TERMS = (
    "python", "javascript", "typescript", "java", "c++", "c#", "golang", "rust", "php", "ruby", "kotlin",
    "swift", "sql", "html", "css", "react", "angular", "vue", "node.js", "nodejs", "django", "flask",
    "spring boot", "api", "rest api", "endpoint", "database", "function", "method", "class", "variable",
    "compile", "compiler", "runtime", "exception", "error", "stack trace", "traceback", "bug", "debug",
    "refactor", "unit test", "pytest", "docker", "kubernetes", "git", "github", "repository", "deploy",
    "ci/cd", "pipeline", "server", "backend", "frontend", "json", "yaml", "regex", "algorithm",
    "library", "framework", "npm", "pip install", "linux", "bash", "script", "code", "syntax", "array",
    "loop", "import", "async", "http", "sdk", "aws", "azure", "terraform", "dataframe", "pandas",
)
REQUIREMENTS_TERMS = (
    "requirement", "requirements", "user story", "user stories", "acceptance criteria", "stakeholder",
    "stakeholders", "business rule", "business rules", "use case", "use cases", "brd", "srs", "prd",
    "functional specification", "product backlog", "backlog", "epic", "sprint", "moscow", "process flow",
    "business process", "gap analysis", "as-is", "to-be", "kpi", "sign-off", "uat", "user acceptance",
    "persona", "product requirement",
)
STRONG_REQUIREMENTS = (
    "user story", "user stories", "acceptance criteria", "business requirement", "business requirements",
    "functional requirement", "non-functional requirement", "brd", "srs", "prd", "product requirement",
    "business rule", "business rules",
)


def _pattern(terms: tuple[str, ...]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(t) for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alternatives})(?![a-z0-9])")


_PROG = _pattern(PROGRAMMING_TERMS)
_REQ = _pattern(REQUIREMENTS_TERMS)
_REQ_STRONG = _pattern(STRONG_REQUIREMENTS)
RELEVANCE_TEXT_MAX_CHARS = 20000


def relevance(conv: Conversation, min_hits: int = 2) -> list[str]:
    """Return the matched categories: 'programming', 'requirements', both, or none."""
    raw = "\n".join(t.content for t in conv.turns if t.role in (Role.USER, Role.ASSISTANT))
    raw = raw[:RELEVANCE_TEXT_MAX_CHARS]
    text = raw.lower()
    cats = []
    if "```" in raw or len(set(_PROG.findall(text))) >= min_hits:
        cats.append("programming")
    if _REQ_STRONG.search(text) or len(set(_REQ.findall(text))) >= min_hits:
        cats.append("requirements")
    return cats


def detect_language(conv: Conversation) -> str | None:
    """ISO code from the user's prose (code blocks removed); None if too little prose to tell."""
    prose = " ".join(t.content for t in conv.turns if t.role == Role.USER)
    prose = _CODE_FENCE.sub(" ", prose)[:3000]
    if len(_LETTERS.findall(prose)) < 20:
        return None
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # deterministic
    try:
        return detect(prose)
    except LangDetectException:
        return None


def is_english(conv: Conversation, code: str = "en") -> bool:
    lang = conv.language or detect_language(conv)
    return lang is None or lang == code  # unknown (mostly code) is kept


def is_non_toxic(conv: Conversation) -> bool:
    return conv.toxic is not True


def has_min_exchanges(conv: Conversation, n: int = 2, exempt_agentic: bool = True) -> bool:
    if exempt_agentic and conv.modality == Modality.AGENTIC:
        return True
    return conv.n_exchanges() >= n


def drop_reason(conv: Conversation, cfg: dict[str, Any], source: str) -> str | None:
    """Return why the conversation is dropped, or None to keep it. Sets metadata['relevance']."""
    rel_cfg = cfg.get("relevance") or {}
    cats = relevance(conv, rel_cfg.get("min_hits", 2))
    conv.metadata["relevance"] = cats
    if not is_english(conv, cfg.get("language", "en")):
        return "language"
    if cfg.get("drop_toxic", True) and not is_non_toxic(conv):
        return "toxic"
    min_sources = cfg.get("min_exchanges_apply_to", ["wildchat"])
    if source in min_sources and not has_min_exchanges(
        conv, cfg.get("min_exchanges", 2), cfg.get("exempt_agentic_from_min_exchanges", True)
    ):
        return "min_exchanges"
    keep = set(rel_cfg.get("keep", ["programming", "requirements"]))
    if source in rel_cfg.get("apply_to", ["wildchat"]) and not keep.intersection(cats):
        return "relevance"
    return None
