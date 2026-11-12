"""Stage 14 - Offline batch inference with the registered model. No deployment; CPU, no cloud call.

    from krs.inference import classify
    report = classify("inference")     # writes data/predictions/<version>.jsonl + _report.json

Loads the current registered model (Stage 12), scores segments in batches and writes one Prediction
per segment: per-label probabilities, the decided labels (per-label thresholds + codebook rules) and
the knowledge-bearing flag that sends a segment to knowledge-card extraction. Works for every model
type the registry can hold: majority, TF-IDF + LR, embeddings + LR, fine-tuned DeBERTa.

Resumable: segments already in the output file are skipped. The report gives speed on this machine
(segments per second, seconds per 1,000), to set against the teacher's latency and cost (H2).
"""

from __future__ import annotations

import json
import pickle
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from krs import registry
from krs.config import get_paths, load_config
from krs.io import read_jsonl
from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES, Domain, KnowledgeType
from krs.models.common import NONE_IDX, decode
from krs.schema import Prediction, Segment


class Predictor:
    """Wraps one registered model: segments -> (knowledge-type probs, domain probs)."""

    def __init__(self, entry: dict[str, Any], device: str = "cpu", embed_model: Any = None):
        self.version = entry["version"]
        self.config = entry["config"]
        self.run_id = self.config.get("run_id", "?")
        self.with_context = bool(self.config.get("with_context", True))
        th = entry["thresholds"]
        self.th_kt = np.array(th["knowledge_types"])
        self.th_dom = np.array(th["domains"])
        model_dir: Path = entry["model_dir"]
        self.kind = "deberta" if self.config.get("model") == "deberta" else "linear"
        if self.kind == "linear":
            with open(model_dir / "classifiers.pkl", "rb") as f:
                self.clf = pickle.load(f)
            self.features = self.clf["meta"]["features"]
            if self.features == "tfidf":
                with open(model_dir / "tfidf.pkl", "rb") as f:
                    self.vectoriser = pickle.load(f)
            else:
                if embed_model is None:
                    from sentence_transformers import SentenceTransformer

                    embed_model = SentenceTransformer(self.clf["meta"]["embedding_model"], device=device)
                self.embedder = embed_model
                self.normalize = (load_config("features").get("embeddings") or {}).get("normalize", True)
        else:
            import torch
            from transformers import AutoTokenizer

            from krs.models.finetune import _build_model

            self.device = device
            self.tok = AutoTokenizer.from_pretrained(model_dir)
            self.model = _build_model(self.config.get("base_model", "microsoft/deberta-v3-small"),
                                      len(KNOWLEDGE_TYPES), len(DOMAINS), 0.0, pretrained=False)
            self.model.load_state_dict(torch.load(model_dir / "weights.pt", map_location=device))
            self.model.to(device).eval()
            self.max_length = self.config.get("max_length", 512)

    def predict_proba(self, segments: list[Segment]) -> tuple[np.ndarray, np.ndarray]:
        if self.kind == "linear":
            from krs.models.baselines import predict_lr

            texts = [s.model_input(with_context=self.with_context) for s in segments]
            if self.features == "tfidf":
                X = self.vectoriser.transform(texts)
            else:
                X = np.asarray(self.embedder.encode(texts, normalize_embeddings=self.normalize,
                                                    show_progress_bar=False, convert_to_numpy=True))
            return predict_lr(self.clf["knowledge_types"], X), predict_lr(self.clf["domains"], X)
        from krs.models.finetune import _predict, encode

        enc = [encode(self.tok, s.text, s.context_text if self.with_context else "", self.max_length) for s in segments]
        return _predict(self.model, enc, 16, self.tok.pad_token_id or 0, self.device)

    def predict(self, segments: list[Segment]) -> list[Prediction]:
        p_kt, p_dom = self.predict_proba(segments)
        kt, dom = decode(p_kt, p_dom, self.th_kt, self.th_dom)
        out = []
        for i, s in enumerate(segments):
            out.append(Prediction(
                segment_id=s.segment_id,
                model_version=self.version,
                run_id=self.run_id,
                knowledge_type_probs={KnowledgeType(k): round(float(p_kt[i, j]), 4) for j, k in enumerate(KNOWLEDGE_TYPES)},
                domain_probs={Domain(d): round(float(p_dom[i, j]), 4) for j, d in enumerate(DOMAINS)},
                knowledge_bearing=bool(kt[i, NONE_IDX] == 0),
                knowledge_types=[KnowledgeType(k) for j, k in enumerate(KNOWLEDGE_TYPES) if kt[i, j]],
                domains=[Domain(d) for j, d in enumerate(DOMAINS) if dom[i, j]],
            ))
        return out


def load_predictor(version: str | None = None, models_dir: str | Path | None = None, device: str = "cpu",
                   embed_model: Any = None) -> Predictor:
    return Predictor(registry.load(version, models_dir), device=device, embed_model=embed_model)


def _segments(inputs: Iterable[Path]) -> Iterator[Segment]:
    for path in inputs:
        yield from read_jsonl(path, Segment)


def _batched(items: Iterator[Segment], size: int) -> Iterator[list[Segment]]:
    batch: list[Segment] = []
    for it in items:
        batch.append(it)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def classify(config: str | Path = "inference", inputs: Iterable[str | Path] | None = None,
             out_path: str | Path | None = None, models_dir: str | Path | None = None,
             predictor: Predictor | None = None) -> dict[str, Any]:
    """Score every segment in `inputs` (default: all files in data/segments/) with the registered model."""
    cfg = load_config(config)
    pred = predictor or load_predictor(cfg.get("version"), models_dir, cfg.get("device", "cpu"))
    paths = get_paths()
    files = [Path(p) for p in inputs] if inputs else sorted(paths.segments.glob("*.jsonl"))
    if cfg.get("sources"):
        files = [f for f in files if f.stem in set(cfg["sources"])]
    out = Path(out_path) if out_path else paths.predictions / f"{pred.version}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {p.segment_id for p in read_jsonl(out, Prediction)} if out.exists() else set()

    todo = (s for s in _segments(files) if s.segment_id not in done)
    n, flagged, kt_counts, dom_counts = 0, 0, Counter(), Counter()
    t0 = time.perf_counter()
    with open(out, "a", encoding="utf-8") as f:
        for batch in _batched(todo, cfg.get("batch_size", 64)):
            for p in pred.predict(batch):
                f.write(p.model_dump_json() + "\n")
                n += 1
                flagged += p.knowledge_bearing
                kt_counts.update(k.value for k in p.knowledge_types)
                dom_counts.update(d.value for d in p.domains)
    secs = time.perf_counter() - t0
    report = {
        "model_version": pred.version, "run_id": pred.run_id, "output": str(out),
        "scored": n, "skipped_already_done": len(done),
        "knowledge_bearing": flagged, "knowledge_bearing_share": round(flagged / n, 3) if n else None,
        "knowledge_types": dict(kt_counts.most_common()), "domains": dict(dom_counts.most_common()),
        "seconds": round(secs, 2),
        "segments_per_second": round(n / secs, 1) if secs > 0 and n else None,
        "seconds_per_1k": round(secs / n * 1000, 2) if n else None,
        "cost_usd": 0.0,  # local model: no API cost
    }
    out.with_name(out.stem + "_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def knowledge_bearing(predictions_path: str | Path) -> Iterator[Prediction]:
    """The flagged predictions: the input to knowledge-card extraction."""
    return (p for p in read_jsonl(predictions_path, Prediction) if p.knowledge_bearing)
