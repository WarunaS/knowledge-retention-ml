"""LLM labelling with Claude: weak labels for the training pool, and the teacher run (T) on the gold set.

Every request sends the codebook as a cached system prompt and one segment, and asks for JSON that
matches a fixed schema (structured outputs), so every answer parses. Raw answers are stored.

Two modes:
- batch (`submit_batch` + `collect_batch`): Message Batches API at 50% of the price; for the
  ~10,000-segment training pool. Most batches finish within an hour.
- sync (`label_sync`): one request per segment with a few threads; records latency per request.
  Used for the teacher run on the gold set (cost and latency per 1,000 segments, plan run T).
  Sync requests opt into server-side refusal fallbacks; the model that answered is recorded.

Resumable: segments already in the output file are skipped. Failures go to an errors file and are
retried on the next run. Pass any object with the same methods as `anthropic.Anthropic` as `client`
(tests use a fake one; nothing is sent).
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from krs.io import read_jsonl
from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES, Domain, KnowledgeType
from krs.schema import Label, LabelSource, Segment

CODEBOOK_PATH = Path(__file__).with_name("codebook.md")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
INSTRUCTIONS = (
    "You label segments of developer–AI conversations using the codebook below. "
    "Read the SEGMENT (and the CONTEXT only to understand it) and return the knowledge types and "
    "technical domains that apply, following the codebook exactly. Keep the rationale to one sentence."
)


def load_codebook(path: str | Path = CODEBOOK_PATH) -> str:
    return Path(path).read_text(encoding="utf-8")


def label_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "knowledge_types": {"type": "array", "items": {"type": "string", "enum": KNOWLEDGE_TYPES}},
            "domains": {"type": "array", "items": {"type": "string", "enum": DOMAINS}},
            "rationale": {"type": "string"},
        },
        "required": ["knowledge_types", "domains", "rationale"],
        "additionalProperties": False,
    }


def segment_prompt(seg: Segment) -> str:
    parts = []
    if seg.context_text:
        parts.append(f"<context>\n{seg.context_text}\n</context>")
    parts.append(f"<segment>\n{seg.text}\n</segment>")
    return "\n\n".join(parts)


def request_params(seg: Segment, codebook: str, llm: dict[str, Any]) -> dict[str, Any]:
    """Messages API parameters for one segment (shared by batch and sync)."""
    return {
        "model": llm.get("model", "claude-opus-5-5"),
        "max_tokens": llm.get("max_tokens", 4096),
        "system": [
            {"type": "text", "text": INSTRUCTIONS},
            {"type": "text", "text": codebook, "cache_control": {"type": "ephemeral"}},
        ],
        "messages": [{"role": "user", "content": segment_prompt(seg)}],
        "output_config": {
            "effort": llm.get("effort", "low"),
            "format": {"type": "json_schema", "schema": label_schema()},
        },
    }


def custom_id(segment_id: str) -> str:
    """Batch custom_id must be 1-64 chars of [a-zA-Z0-9_-]; segment ids are not, so hash them."""
    return "s" + hashlib.sha1(segment_id.encode("utf-8")).hexdigest()


class LabelError(ValueError):
    pass


def parse_message(segment_id: str, message: Any, codebook_version: str, source: LabelSource) -> Label:
    """Turn an API message into a Label, or raise LabelError with the reason."""
    if message.stop_reason == "refusal":
        raise LabelError("refusal")
    if message.stop_reason == "max_tokens":
        raise LabelError("max_tokens")
    text = next((b.text for b in message.content if getattr(b, "type", None) == "text"), None)
    if not text:
        raise LabelError("no text block")
    try:
        data = json.loads(text)
        return Label(
            segment_id=segment_id,
            knowledge_types=list(dict.fromkeys(KnowledgeType(k) for k in data["knowledge_types"])),
            domains=list(dict.fromkeys(Domain(d) for d in data["domains"])),
            source=source,
            rater_id=getattr(message, "model", None),
            codebook_version=codebook_version,
            raw_response=text,
        )
    except (ValueError, KeyError, TypeError) as e:  # bad JSON, unknown label, 'none' combined, empty
        raise LabelError(f"invalid label: {e}") from e


def _usage(message: Any) -> dict[str, int]:
    u = message.usage
    return {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }


def cost_usd(usage: dict[str, int], prices: dict[str, float], batch: bool = False) -> float:
    """Cost from token counts and $ per million tokens (prices from configs/label.yaml)."""
    total = (
        usage.get("input_tokens", 0) * prices.get("input", 0)
        + usage.get("output_tokens", 0) * prices.get("output", 0)
        + usage.get("cache_read_input_tokens", 0) * prices.get("cache_read", 0)
        + usage.get("cache_creation_input_tokens", 0) * prices.get("cache_write", 0)
    ) / 1_000_000
    return total * (0.5 if batch else 1.0)


def done_ids(path: str | Path) -> set[str]:
    p = Path(path)
    return {lab.segment_id for lab in read_jsonl(p, Label)} if p.exists() else set()


def _append(path: Path, line: str, lock: threading.Lock) -> None:
    with lock, open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _add_usage(total: dict[str, int], u: dict[str, int]) -> None:
    for k, v in u.items():
        total[k] = total.get(k, 0) + v


# --------------------------------------------------------------------------- sync (teacher run)


def label_sync(
    segments: Iterable[Segment],
    out_path: str | Path,
    llm: dict[str, Any],
    client: Any = None,
    codebook: str | None = None,
    codebook_version: str = "v1",
    source: LabelSource = LabelSource.LLM,
) -> dict[str, Any]:
    """Label segments one request each. Returns a report with counts, tokens, cost and latency."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic(max_retries=llm.get("max_retries", 4))
    codebook = codebook or load_codebook()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    errors = out.with_name(out.stem + "_errors.jsonl")
    skip = done_ids(out)
    todo = [s for s in segments if s.segment_id not in skip]
    lock = threading.Lock()
    usage: dict[str, int] = {}
    latencies: list[float] = []
    counts = {"labelled": 0, "failed": 0, "skipped_already_done": len(skip)}

    def one(seg: Segment) -> tuple[Segment, Any, float]:
        params = request_params(seg, codebook, llm)
        if llm.get("fallbacks", True):
            params.update(betas=[FALLBACK_BETA], fallbacks="default")
        t = time.perf_counter()
        msg = client.beta.messages.create(**params)
        return seg, msg, time.perf_counter() - t

    with ThreadPoolExecutor(max_workers=llm.get("workers", 4)) as pool:
        futures = {pool.submit(one, s): s for s in todo}
        for fut in as_completed(futures):
            seg = futures[fut]
            try:
                _, msg, secs = fut.result()
                _add_usage(usage, _usage(msg))
                latencies.append(secs)
                label = parse_message(seg.segment_id, msg, codebook_version, source)
                _append(out, label.model_dump_json(), lock)
                counts["labelled"] += 1
            except Exception as e:  # noqa: BLE001 - record and keep going; rerun retries failures
                err = {"segment_id": seg.segment_id, "error": f"{type(e).__name__}: {e}"}
                _append(errors, json.dumps(err), lock)
                counts["failed"] += 1

    latencies.sort()
    report = {
        **counts,
        "usage": usage,
        "cost_usd": round(cost_usd(usage, llm.get("prices", {})), 4),
        "latency_s_median": round(latencies[len(latencies) // 2], 3) if latencies else None,
        "latency_s_p90": round(latencies[int(len(latencies) * 0.9)], 3) if latencies else None,
    }
    if counts["labelled"]:
        report["cost_usd_per_1k"] = round(report["cost_usd"] / counts["labelled"] * 1000, 2)
    return report


# --------------------------------------------------------------------------- batch (weak labels)


def submit_batch(
    segments: Iterable[Segment],
    state_path: str | Path,
    llm: dict[str, Any],
    client: Any = None,
    codebook: str | None = None,
    done: Iterable[str] = (),
) -> dict[str, Any]:
    """Create Message Batches for every segment not already labelled. Saves batch ids to state_path."""
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    codebook = codebook or load_codebook()
    skip = set(done)
    todo = [s for s in segments if s.segment_id not in skip]
    chunk = llm.get("batch_chunk", 5000)
    state = {"batch_ids": [], "id_map": {}, "model": llm.get("model"), "submitted_at": time.time()}
    for i in range(0, len(todo), chunk):
        part = todo[i : i + chunk]
        requests = []
        for seg in part:
            cid = custom_id(seg.segment_id)
            state["id_map"][cid] = seg.segment_id
            requests.append(Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**request_params(seg, codebook, llm))))
        batch = client.messages.batches.create(requests=requests)
        state["batch_ids"].append(batch.id)
    path = Path(state_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state), encoding="utf-8")
    return {"batches": len(state["batch_ids"]), "requests": len(todo)}


def batch_status(state_path: str | Path, client: Any = None) -> dict[str, str]:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    return {bid: client.messages.batches.retrieve(bid).processing_status for bid in state["batch_ids"]}


def collect_batch(
    state_path: str | Path,
    out_path: str | Path,
    llm: dict[str, Any],
    client: Any = None,
    codebook_version: str = "v1",
) -> dict[str, Any]:
    """Write labels from every ended batch. Re-running is safe: already written segments are skipped."""
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    errors = out.with_name(out.stem + "_errors.jsonl")
    have = done_ids(out)
    lock = threading.Lock()
    usage: dict[str, int] = {}
    counts = {"labelled": 0, "failed": 0, "not_ended": 0}
    for bid in state["batch_ids"]:
        if client.messages.batches.retrieve(bid).processing_status != "ended":
            counts["not_ended"] += 1
            continue
        for res in client.messages.batches.results(bid):
            seg_id = state["id_map"].get(res.custom_id, res.custom_id)
            if seg_id in have:
                continue
            if res.result.type != "succeeded":
                _append(errors, json.dumps({"segment_id": seg_id, "error": res.result.type}), lock)
                counts["failed"] += 1
                continue
            msg = res.result.message
            _add_usage(usage, _usage(msg))
            try:
                label = parse_message(seg_id, msg, codebook_version, LabelSource.LLM)
            except LabelError as e:
                _append(errors, json.dumps({"segment_id": seg_id, "error": str(e)}), lock)
                counts["failed"] += 1
                continue
            _append(out, label.model_dump_json(), lock)
            have.add(seg_id)
            counts["labelled"] += 1
    counts["usage"] = usage
    counts["cost_usd"] = round(cost_usd(usage, llm.get("prices", {}), batch=True), 4)
    return counts
