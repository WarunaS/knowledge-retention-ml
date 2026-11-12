"""App evaluation, part 1: card faithfulness spot check and retrieval quality.

Faithfulness spot check (plan: "human spot check")
    export_card_check(store, n=50) -> a person marks each card faithful y/n -> card_check_rate()

Retrieval on a hand-verified question set
    export_query_sheet(store, n=50) -> for each sampled card a person writes a question IN THEIR OWN
        WORDS (copying the title would make search too easy) and lists every relevant card id
    load_queries() -> evaluate_retrieval(): Recall@k, hit@k and MRR per mode
        semantic       plain semantic search, no filters
        filtered       with the question's filters (project, role, ...)
        filtered+graph filters plus one-hop graph expansion
    with 95% bootstrap CIs over questions, and a paired test of "filters help" (filtered vs semantic).

Graph expansion appends neighbouring cards AFTER the top-k results, so it can only change Recall@k
when the filters leave fewer than k cards; its purpose is extra connected context for the agent,
not re-ranking. The report says this next to the numbers.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from krs.app.retrieval import search
from krs.app.storage import KnowledgeStore
from krs.clean.precision_check import wilson
from krs.config import RESULTS_DIR

MODES = {"semantic": (False, False), "filtered": (True, False), "filtered+graph": (True, True)}
FILTER_KEYS = ("project_id", "role", "author_id", "domains", "knowledge_types", "since", "until")


def out_dir(root: str | Path | None = None) -> Path:
    path = (Path(root) if root else RESULTS_DIR) / "app_eval"
    path.mkdir(parents=True, exist_ok=True)
    return path


# --------------------------------------------------------------------------- faithfulness spot check


def export_card_check(store: KnowledgeStore, n: int = 50, seed: int = 42, out_csv: str | Path | None = None) -> int:
    """Random cards next to their evidence, for a person to mark faithful (y) or not (n)."""
    cards = store.all()
    sample = random.Random(seed).sample(cards, min(n, len(cards)))
    out = Path(out_csv) if out_csv else out_dir() / "card_check.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["card_id", "title", "summary", "decision", "evidence", "source_segments", "faithful", "notes"])
        for c in sample:
            w.writerow([c.card_id, c.title, c.summary, c.decision or "", c.evidence, " ".join(c.source_segment_ids), "", ""])
    return len(sample)


def card_check_rate(csv_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(csv_path) if csv_path else out_dir() / "card_check.csv"
    with open(path, newline="", encoding="utf-8") as f:
        marks = [r["faithful"].strip().lower() for r in csv.DictReader(f)]
    done = [m for m in marks if m in ("y", "n")]
    k = sum(m == "y" for m in done)
    lo, hi = wilson(k, len(done))
    return {"checked": len(done), "faithful": k, "rate": round(k / len(done), 3) if done else None,
            "ci95_low": round(lo, 3), "ci95_high": round(hi, 3), "unchecked": len(marks) - len(done)}


# --------------------------------------------------------------------------- question set


@dataclass
class Query:
    query_id: str
    question: str
    relevant: list[str]
    filters: dict[str, Any] = field(default_factory=dict)


def export_query_sheet(store: KnowledgeStore, n: int = 50, seed: int = 42, out_csv: str | Path | None = None) -> int:
    """One row per sampled card: write a question it answers (own words) and list all relevant card ids."""
    cards = [c for c in store.all() if c.knowledge_types]
    sample = random.Random(seed).sample(cards, min(n, len(cards)))
    out = Path(out_csv) if out_csv else out_dir() / "queries.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "card_title", "card_summary", "question", "relevant_card_ids", "filters_json", "notes"])
        for i, c in enumerate(sample):
            filters = {k: v for k, v in {"project_id": c.project_id, "role": c.role.value if c.role else None}.items() if v}
            w.writerow([f"q{i:03d}", c.title, c.summary, "", c.card_id, json.dumps(filters), ""])
    return len(sample)


def load_queries(csv_path: str | Path | None = None) -> list[Query]:
    """Rows with a written question; relevant ids separated by spaces or commas."""
    path = Path(csv_path) if csv_path else out_dir() / "queries.csv"
    queries = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = r.get("question", "").strip()
            rel = [x for x in r.get("relevant_card_ids", "").replace(",", " ").split() if x]
            if q and rel:
                filters = json.loads(r["filters_json"]) if r.get("filters_json", "").strip() else {}
                queries.append(Query(r["query_id"], q, rel, {k: v for k, v in filters.items() if k in FILTER_KEYS}))
    return queries


# --------------------------------------------------------------------------- metrics


def rank_of_first_relevant(ranked: list[str], relevant: set[str]) -> int | None:
    return next((i + 1 for i, cid in enumerate(ranked) if cid in relevant), None)


def recall_at(ranked: list[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def _ci(values: np.ndarray, n: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return round(float(lo), 4), round(float(hi), 4)


def evaluate_retrieval(store: KnowledgeStore, queries: list[Query], ks: tuple[int, ...] = (1, 3, 5, 10),
                       n_boot: int = 1000, seed: int = 42, out: str | Path | None = None) -> dict[str, Any]:
    if not queries:
        raise ValueError("no usable questions - fill the question and relevant_card_ids columns first")
    kmax = max(ks)
    per_query: dict[str, dict[str, np.ndarray]] = {}
    for mode, (use_filters, expand) in MODES.items():
        rr, recalls = [], {k: [] for k in ks}
        for q in queries:
            hits = search(store, q.question, kmax, q.filters if use_filters else None, expand)
            ranked = [h.card.card_id for h in hits][:kmax]
            first = rank_of_first_relevant(ranked, set(q.relevant))
            rr.append(1.0 / first if first else 0.0)
            for k in ks:
                recalls[k].append(recall_at(ranked, set(q.relevant), k))
        per_query[mode] = {"rr": np.array(rr), **{f"recall@{k}": np.array(v) for k, v in recalls.items()}}

    report: dict[str, Any] = {"questions": len(queries), "bootstrap_n": n_boot, "modes": {}}
    for mode, m in per_query.items():
        entry = {"mrr": round(float(m["rr"].mean()), 4), "mrr_ci95": _ci(m["rr"], n_boot, seed)}
        for k in ks:
            entry[f"recall@{k}"] = round(float(m[f"recall@{k}"].mean()), 4)
            entry[f"recall@{k}_ci95"] = _ci(m[f"recall@{k}"], n_boot, seed)
        report["modes"][mode] = entry
    diff = per_query["filtered"]["rr"] - per_query["semantic"]["rr"]
    lo, hi = _ci(diff, n_boot, seed)
    report["filters_help"] = {"mrr_diff": round(float(diff.mean()), 4), "ci95": [lo, hi], "supported": lo > 0}
    report["note"] = ("filtered+graph appends expanded cards after the top-k; it changes these metrics only when "
                      "filters leave fewer than k cards. Its role is extra context for the agent.")

    folder = Path(out) if out else out_dir()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
