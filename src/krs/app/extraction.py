"""Knowledge extraction: segments the classifier flags as knowledge-bearing -> typed KnowledgeCards.

Two backends:
- claude: Claude writes the card as schema-checked JSON (public sources only)
- local: extractive, no LLM - the title, summary, decision and evidence are sentences taken from the
  segment itself. Own-log segments ALWAYS use this backend (privacy rule: own-log content never goes
  to a cloud LLM).

Faithfulness check on every card (plan: "cards without a grounded source are rejected"):
1. the evidence quote must appear verbatim in the segment or its context (whitespace/case ignored)
2. the share of the summary's content words found in the source must reach the threshold
Rejected cards go to a separate file with the reason. A decision card must state the decision.
"""

from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from krs.app.common import (
    app_config,
    claude_client,
    claude_credentials_set,
    claude_text,
    cloud_allowed,
    content_words,
    norm_space,
)
from krs.io import read_jsonl
from krs.label.sampling import load_segments
from krs.label.taxonomy import KnowledgeType
from krs.schema import KnowledgeCard, Prediction, Segment

CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "decision": {"type": "string"},
        "tools_involved": {"type": "array", "items": {"type": "string"}},
        "evidence": {"type": "string"},
    },
    "required": ["title", "summary", "decision", "tools_involved", "evidence"],
    "additionalProperties": False,
}
SYSTEM = (
    "You turn one exchange from a developer's conversation with an AI assistant into a knowledge card for a "
    "successor who never saw the conversation. Write only what the exchange states; do not add facts.\n"
    "- title: under 12 words\n"
    "- summary: 1-3 sentences, the knowledge in explicit form (the fact, procedure, decision or context)\n"
    "- decision: if a choice was made, 'Chose X over Y because Z'; otherwise an empty string\n"
    "- tools_involved: languages, frameworks, services or agent tools named in the exchange\n"
    "- evidence: ONE sentence copied word for word from the SEGMENT that best supports the summary"
)
_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_DECISION_CUE = re.compile(r"\b(instead of|rather than|chose|choose|decided|go with|opted|prefer|because)\b", re.IGNORECASE)
_TOOL = re.compile(r"TOOL \(([^)]+)\)|`([A-Za-z][\w.\-/]{1,40})`")


def card_id(segment_id: str) -> str:
    return "k" + hashlib.sha1(segment_id.encode("utf-8")).hexdigest()[:12]


def _turns(text: str) -> list[tuple[str, str]]:
    """'USER: ...' / 'ASSISTANT: ...' / 'TOOL (x): ...' lines of a rendered segment -> (role, content)."""
    out: list[tuple[str, str]] = []
    for line in text.split("\n"):
        m = re.match(r"^(USER|ASSISTANT|TOOL(?: \([^)]*\))?): ?(.*)$", line)
        if m:
            out.append((m.group(1), m.group(2)))
        elif out:
            out[-1] = (out[-1][0], out[-1][1] + "\n" + line)
    return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT.split(text) if len(s.strip()) > 15]


def _is_prose(text: str) -> bool:
    """A sentence a person wrote, not a tag, path, command output or code line."""
    s = text.strip()
    return len(s.split()) >= 4 and not s.startswith(("<", "[", "{", "$", "/", "#", "=", "```")) and sum(ch.isalpha() for ch in s) > len(s) * 0.6


def local_card_fields(seg: Segment, knowledge_types: list[KnowledgeType]) -> dict[str, Any]:
    """Extractive card: every field is copied from the segment (no generation)."""
    turns = _turns(seg.text)
    prompt = next((c for r, c in turns if r == "USER"), "")  # tool output is not a usable title
    # sentences per assistant turn, so no "sentence" spans two turns (it must be quotable verbatim)
    sents = [s for r, c in turns if r == "ASSISTANT" for s in _sentences(c)]
    sents = sents or _sentences(seg.text) or [seg.text.strip()[:300]]
    first_prompt = (_sentences(prompt) or [prompt.strip()])[0]
    title = (first_prompt if _is_prose(first_prompt) else sents[0])[:90]
    decision = ""
    if KnowledgeType.DECISION in knowledge_types:
        decision = next((s for s in sents if _DECISION_CUE.search(s)), "")
    tools = sorted({a or b for a, b in _TOOL.findall(seg.text)})[:10]
    return {"title": title, "summary": " ".join(sents[:2]), "decision": decision,
            "tools_involved": tools, "evidence": sents[0]}


def faithfulness(fields: dict[str, Any], seg: Segment) -> tuple[bool, float, str]:
    """(passes, score, reason). Evidence must be verbatim in the source; score = supported summary words."""
    source = norm_space(seg.text + " " + seg.context_text)
    ev = norm_space(fields.get("evidence", ""))
    if not ev or ev not in source:
        return False, 0.0, "evidence not found verbatim in the source"
    words = content_words(fields.get("summary", ""))
    score = len(words & content_words(source)) / len(words) if words else 0.0
    return True, round(score, 3), ""


def build_card(seg: Segment, pred: Prediction, fields: dict[str, Any], score: float,
               backend: str) -> KnowledgeCard:
    kts = list(pred.knowledge_types)
    if KnowledgeType.DECISION in kts and not fields.get("decision"):
        kts.remove(KnowledgeType.DECISION)  # keep the card's other knowledge; only the decision claim is unsupported
        if not kts:
            raise ValueError("only a decision was predicted but no decision is stated")
    conf = max((p for k, p in pred.knowledge_type_probs.items() if k != KnowledgeType.NONE), default=0.0)
    return KnowledgeCard(
        card_id=card_id(seg.segment_id), source_segment_ids=[seg.segment_id], conversation_id=seg.conversation_id,
        domains=list(pred.domains), knowledge_types=kts, title=fields["title"].strip()[:200],
        summary=fields["summary"].strip(), decision=fields.get("decision") or None,
        tools_involved=[t for t in fields.get("tools_involved", []) if t][:15], evidence=fields["evidence"].strip(),
        confidence=round(float(conf), 4), faithfulness_score=score, model_version=f"{pred.model_version}+{backend}",
        author_id=seg.author_id, role=seg.role, project_id=seg.project_id, occurred_at=seg.occurred_at,
    )


def _claude_fields(client: Any, llm: dict[str, Any], seg: Segment, pred: Prediction) -> dict[str, Any]:
    user = (f"Predicted knowledge types: {', '.join(k.value for k in pred.knowledge_types)}\n"
            f"Predicted domains: {', '.join(d.value for d in pred.domains)}\n\n"
            + (f"<context>\n{seg.context_text}\n</context>\n\n" if seg.context_text else "")
            + f"<segment>\n{seg.text}\n</segment>")
    text, _ = claude_text(client, llm, SYSTEM, user, CARD_SCHEMA)
    return json.loads(text)


def extract_cards(predictions_path: str | Path, out_path: str | Path, cfg: dict[str, Any] | None = None,
                  client: Any = None, segments_dir: str | Path | None = None) -> dict[str, Any]:
    """Cards for every knowledge-bearing prediction. Resumable; rejected cards logged with the reason."""
    cfg = cfg or app_config()
    ex = cfg.get("extraction") or {}
    threshold = ex.get("faithfulness_threshold", 0.5)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = out.with_name(out.stem + "_rejected.jsonl")
    done = {c.source_segment_ids[0] for c in read_jsonl(out, KnowledgeCard)} if out.exists() else set()
    preds = [p for p in read_jsonl(predictions_path, Prediction) if p.knowledge_bearing and p.segment_id not in done]
    segs = load_segments([p.segment_id for p in preds], segments_dir) if preds else {}

    counts = {"cards": 0, "rejected": 0, "local": 0, "claude": 0, "skipped_already_done": len(done)}

    def one(p: Prediction) -> tuple[Prediction, dict[str, Any], str]:
        seg = segs[p.segment_id]
        use_claude = ex.get("backend", "claude") == "claude" and cloud_allowed(seg.conversation_id, cfg)
        if use_claude:
            return p, _claude_fields(client, ex, seg, p), "claude"
        return p, local_card_fields(seg, list(p.knowledge_types)), "local"

    if ex.get("backend", "claude") == "claude" and client is None and any(cloud_allowed(segs[p.segment_id].conversation_id, cfg) for p in preds):
        if not claude_credentials_set():
            raise RuntimeError("extraction backend is 'claude' but no ANTHROPIC_API_KEY is set: set the key, "
                               "or set extraction.backend: local in configs/app.yaml")
        client = claude_client()
    with open(out, "a", encoding="utf-8") as f_ok, open(rejected_path, "a", encoding="utf-8") as f_bad, \
            ThreadPoolExecutor(max_workers=ex.get("workers", 4)) as pool:
        futures = {pool.submit(one, p): p for p in preds}
        for fut in as_completed(futures):
            p = futures[fut]
            seg = segs[p.segment_id]
            try:
                _, fields, backend = fut.result()
                ok, score, reason = faithfulness(fields, seg)
                if ok and score < threshold:
                    ok, reason = False, f"summary support {score} below {threshold}"
                if not ok:
                    raise ValueError(reason)
                card = build_card(seg, p, fields, score, backend)
            except Exception as e:  # noqa: BLE001 - record and continue
                f_bad.write(json.dumps({"segment_id": p.segment_id, "reason": f"{type(e).__name__}: {e}"}) + "\n")
                counts["rejected"] += 1
                continue
            f_ok.write(card.model_dump_json() + "\n")
            counts["cards"] += 1
            counts[backend] += 1
    return counts
