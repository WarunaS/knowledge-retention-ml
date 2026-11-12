"""Shared helpers for the application plane: the privacy rule, the embedder, and Claude calls."""

from __future__ import annotations

import re
from typing import Any

import numpy as np

from krs.config import load_config

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def app_config() -> dict[str, Any]:
    return load_config("app")


def source_of(conversation_id: str) -> str:
    """Conversation ids start with their source ('own_logs:p01:...', 'wildchat:...')."""
    return conversation_id.split(":", 1)[0]


def cloud_allowed(conversation_id: str, cfg: dict[str, Any] | None = None) -> bool:
    """Privacy rule: only listed public sources may be sent to a cloud LLM. Own logs never are."""
    allowed = set((cfg or app_config()).get("cloud_sources", []))
    return source_of(conversation_id) in allowed and source_of(conversation_id) != "own_logs"


class Embedder:
    """Sentence embeddings, normalised, loaded on first use."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", model: Any = None):
        self.model_name = model_name
        self._model = model

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        v = np.asarray(self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                                          convert_to_numpy=True), dtype=np.float32)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.maximum(n, 1e-12)


def claude_client() -> Any:
    import anthropic

    return anthropic.Anthropic(max_retries=4)


def claude_text(client: Any, llm: dict[str, Any], system: str, user: str,
                schema: dict[str, Any] | None = None) -> tuple[str, Any]:
    """One Claude request; returns (text, message). Refusals and truncation raise."""
    params: dict[str, Any] = {
        "model": llm.get("model", "claude-opus-5-5"),
        "max_tokens": llm.get("max_tokens", 4096),
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"effort": llm.get("effort", "low")},
    }
    if schema:
        params["output_config"]["format"] = {"type": "json_schema", "schema": schema}
    if llm.get("fallbacks", True):
        params.update(betas=[FALLBACK_BETA], fallbacks="default")
    msg = client.beta.messages.create(**params)
    if msg.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"Claude stopped: {msg.stop_reason}")
    text = next((b.text for b in msg.content if getattr(b, "type", None) == "text"), "")
    return text, msg


_WS = re.compile(r"\s+")
STOPWORDS = frozenset(["a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to", "of", "in", "on", "at", "by", "with", "from", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "into", "than", "so", "such", "can", "could", "should", "would", "will", "may", "might", "must", "do", "does", "did", "not", "no", "yes", "you", "your", "we", "our", "they", "their", "he", "she", "his", "her", "i", "me", "my", "use", "using", "used"])


def norm_space(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def content_words(text: str) -> set[str]:
    """Words of 4+ characters, keeping file names like app.py but not sentence punctuation ('calls.' -> 'calls')."""
    words = (w.strip("./-") for w in re.findall(r"[a-z0-9_./-]{4,}", text.lower()))
    return {w for w in words if len(w) >= 4 and w not in STOPWORDS}
