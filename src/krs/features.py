"""Stage 5 - Feature engineering. Runs after the split; anything that is fitted is fitted on TRAIN ONLY.

    from krs.features import build_features
    report = build_features("features")     # writes data/features/

Per split (train, validation, test, transfer) and per input view (with / without the context turn,
for ablation A1) it writes:
- TF-IDF matrix (baseline B1): vectoriser fitted on the training texts only, then applied to the rest
- sentence embeddings (baseline B2): a fixed pre-trained model, nothing fitted, cached per split
- label matrices: one 0/1 column per knowledge type and per domain, in the fixed taxonomy order
And a token-length report for the fine-tuned model (M1): how many inputs exceed its 512-token limit.

Embedding is the heavy step: use a GPU on Colab for the full data; on a laptop keep it to small samples.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from krs.config import get_paths, load_config
from krs.io import read_jsonl
from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES
from krs.schema import LabelledSegment

SPLITS = ("train", "validation", "test", "transfer")
VIEWS = {True: "ctx", False: "noctx"}


def load_rows(split: str, splits_dir: str | Path | None = None) -> list[LabelledSegment]:
    path = (Path(splits_dir) if splits_dir else get_paths().splits) / f"{split}.jsonl"
    return list(read_jsonl(path, LabelledSegment)) if path.exists() else []


def texts(rows: Sequence[LabelledSegment], with_context: bool = True) -> list[str]:
    """Model input text; without context for ablation A1."""
    return [r.segment.model_input(with_context=with_context) for r in rows]


def label_matrices(rows: Sequence[LabelledSegment]) -> tuple[np.ndarray, np.ndarray]:
    """(knowledge-type matrix, domain matrix): rows x labels, 0/1, columns in taxonomy order."""
    kt = np.zeros((len(rows), len(KNOWLEDGE_TYPES)), dtype=np.int8)
    dom = np.zeros((len(rows), len(DOMAINS)), dtype=np.int8)
    k_idx = {k: i for i, k in enumerate(KNOWLEDGE_TYPES)}
    d_idx = {d: i for i, d in enumerate(DOMAINS)}
    for r, row in enumerate(rows):
        for k in row.label.knowledge_types:
            kt[r, k_idx[k.value]] = 1
        for d in row.label.domains:
            dom[r, d_idx[d.value]] = 1
    return kt, dom


def fit_tfidf(train_texts: Sequence[str], **params: Any):
    """TF-IDF vectoriser fitted on training texts only."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    p = dict(params)
    if "ngram_range" in p:
        p["ngram_range"] = tuple(p["ngram_range"])
    return TfidfVectorizer(**p).fit(train_texts)


def embed(text_list: Sequence[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
          batch_size: int = 64, normalize: bool = True, model: Any = None) -> np.ndarray:
    """Sentence embeddings with a fixed pre-trained model (nothing is fitted on our data)."""
    if not text_list:
        return np.zeros((0, 0), dtype=np.float32)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
    vecs = model.encode(list(text_list), batch_size=batch_size, normalize_embeddings=normalize,
                        show_progress_bar=False, convert_to_numpy=True)
    return np.asarray(vecs, dtype=np.float32)


def token_length_report(text_list: Sequence[str], tokenizer: Any, max_length: int = 512) -> dict[str, Any]:
    """How long the inputs are for the fine-tuned model, and how many it would cut."""
    if not text_list:
        return {"n": 0}
    lengths = np.array([len(tokenizer(t, add_special_tokens=True)["input_ids"]) for t in text_list])
    return {
        "n": len(lengths),
        "median": int(np.median(lengths)),
        "p90": int(np.percentile(lengths, 90)),
        "max": int(lengths.max()),
        "share_over_limit": round(float((lengths > max_length).mean()), 3),
    }


def build_features(config: str | Path = "features", splits_dir: str | Path | None = None,
                   out_dir: str | Path | None = None, embed_model: Any = None, tokenizer: Any = None) -> dict[str, Any]:
    """Build every feature and label matrix. `embed_model` / `tokenizer` can be passed in (tests use fakes)."""
    from scipy import sparse

    cfg = load_config(config)
    out = Path(out_dir) if out_dir else get_paths().features
    out.mkdir(parents=True, exist_ok=True)
    rows = {s: load_rows(s, splits_dir) for s in SPLITS}
    if not rows["train"]:
        raise FileNotFoundError("no train split - run Stage 6 first")
    present = [s for s in SPLITS if rows[s]]
    report: dict[str, Any] = {"splits": {s: len(rows[s]) for s in present}, "labels": {
        "knowledge_types": KNOWLEDGE_TYPES, "domains": DOMAINS}}

    for s in present:  # labels and ids, same for every view
        kt, dom = label_matrices(rows[s])
        np.savez_compressed(out / f"{s}_labels.npz", knowledge_types=kt, domains=dom,
                            segment_ids=np.array([r.segment.segment_id for r in rows[s]]),
                            conversation_ids=np.array([r.segment.conversation_id for r in rows[s]]))

    tf_cfg = dict(cfg.get("tfidf") or {})
    emb_cfg = cfg.get("embeddings") or {}
    if emb_cfg.get("enabled", True) and embed_model is None:
        from sentence_transformers import SentenceTransformer

        embed_model = SentenceTransformer(emb_cfg.get("model", "sentence-transformers/all-MiniLM-L6-v2"))

    for with_ctx in cfg.get("with_context", [True, False]):
        view = VIEWS[bool(with_ctx)]
        view_texts = {s: texts(rows[s], bool(with_ctx)) for s in present}
        vec = fit_tfidf(view_texts["train"], **tf_cfg)  # train only
        with open(out / f"tfidf_{view}.pkl", "wb") as f:
            pickle.dump(vec, f)
        report[f"tfidf_{view}"] = {"vocabulary": len(vec.vocabulary_)}
        for s in present:
            sparse.save_npz(out / f"{s}_tfidf_{view}.npz", vec.transform(view_texts[s]).tocsr())
            if emb_cfg.get("enabled", True):
                vecs = embed(view_texts[s], batch_size=emb_cfg.get("batch_size", 64),
                             normalize=emb_cfg.get("normalize", True), model=embed_model)
                np.save(out / f"{s}_emb_{view}.npy", vecs)
                report[f"embeddings_{view}_dim"] = int(vecs.shape[1])

    tok_cfg = cfg.get("tokenizer") or {}
    if tok_cfg.get("enabled", True):
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tok_cfg.get("model", "microsoft/deberta-v3-small"))
        sample = rows["train"][: tok_cfg.get("sample", 2000)]
        report["token_lengths"] = {
            VIEWS[bool(c)]: token_length_report(texts(sample, bool(c)), tokenizer, tok_cfg.get("max_length", 512))
            for c in cfg.get("with_context", [True, False])
        }

    (out / "features_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def load_features(split: str, kind: str = "tfidf", with_context: bool = True,
                  features_dir: str | Path | None = None) -> tuple[Any, np.ndarray, np.ndarray]:
    """(X, Y_knowledge_types, Y_domains) for a split. kind: 'tfidf' (sparse) or 'emb' (dense)."""
    from scipy import sparse

    folder = Path(features_dir) if features_dir else get_paths().features
    view = VIEWS[with_context]
    X = sparse.load_npz(folder / f"{split}_tfidf_{view}.npz") if kind == "tfidf" else np.load(folder / f"{split}_emb_{view}.npy")
    labels = np.load(folder / f"{split}_labels.npz")
    return X, labels["knowledge_types"], labels["domains"]
