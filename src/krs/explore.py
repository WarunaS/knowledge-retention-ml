"""Stage 4 - Explore: what is in the cleaned data, before any labelling.

    from krs.explore import run_explore
    summary = run_explore("explore")      # tables, plots and summary.json in results/eda/

1. Distributions (RQ2 background): conversations, segments, projects, turns, lengths and tool use
   per source and modality. Text is not kept in memory, only lengths, so this scales to the full run.
2. Topic clusters (BERTopic on a seeded sample of segments) with keyword domain hints per topic,
   to sanity-check the domain taxonomy: topics that map to no domain point to a missing label.
3. Knowledge-bearing share: needs labels, so `knowledge_share()` runs after Stage 4b.

Heavy step: BERTopic embeds the sample with a sentence model. Use a small sample on a laptop
(e.g. 1,000) and the full sample (20,000) on Colab.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from krs.config import RESULTS_DIR, get_paths, load_config
from krs.io import read_jsonl
from krs.label.taxonomy import Domain
from krs.schema import Conversation, Label, Role, Segment

CHARS_PER_TOKEN = 4

# Keyword hints per domain. Used only to describe topic clusters, never as labels.
DOMAIN_KEYWORDS: dict[str, tuple[str, ...]] = {
    Domain.MOBILE.value: ("android", "ios", "swiftui", "kotlin", "flutter", "react native", "xcode", "mobile app"),
    Domain.WEB_FRONTEND.value: ("html", "css", "react", "vue", "angular", "dom", "frontend", "tailwind", "browser",
                                "jsx", "next.js", "javascript"),
    Domain.BACKEND_APIS.value: ("api", "endpoint", "rest", "graphql", "django", "flask", "fastapi", "express",
                                "spring", "backend", "node.js", "server"),
    Domain.DATA_ANALYTICS.value: ("pandas", "numpy", "dataframe", "matplotlib", "machine learning", "dataset",
                                  "analytics", "etl", "spark", "jupyter", "sql", "regression"),
    Domain.CLOUD_INFRA.value: ("aws", "azure", "gcp", "s3", "ec2", "lambda", "terraform", "kubernetes", "cloud", "vpc"),
    Domain.DEVOPS_CICD.value: ("docker", "ci/cd", "github actions", "jenkins", "pipeline", "deploy", "helm",
                               "dockerfile", "build"),
    Domain.SECURITY.value: ("authentication", "oauth", "jwt", "encryption", "vulnerability", "xss", "csrf",
                            "password", "tls", "security"),
    Domain.QA_TESTING.value: ("unit test", "pytest", "jest", "test case", "mock", "assert", "selenium",
                              "coverage", "qa", "testing"),
    Domain.ARCHITECTURE_DESIGN.value: ("architecture", "design pattern", "microservice", "scalability", "trade-off",
                                       "monolith", "event-driven", "coupling"),
    Domain.BUSINESS_REQUIREMENTS.value: ("user story", "requirement", "stakeholder", "acceptance criteria",
                                         "business rule", "use case", "backlog"),
}
_DOMAIN_PATTERNS = {
    d: re.compile(r"(?<![a-z0-9])(?:" + "|".join(re.escape(k) for k in kws) + r")(?![a-z0-9])")
    for d, kws in DOMAIN_KEYWORDS.items()
}


def domain_hints(text: str) -> list[str]:
    """Domains whose keywords appear in the text, strongest first; ['other'] if none."""
    low = text.lower()
    scores = {d: len(set(p.findall(low))) for d, p in _DOMAIN_PATTERNS.items()}
    hits = sorted((d for d, s in scores.items() if s), key=lambda d: -scores[d])
    return hits or [Domain.OTHER.value]


# --------------------------------------------------------------------------- 1. distributions


def _jsonl_files(folder: Path, sources: Iterable[str] | None) -> list[Path]:
    files = sorted(folder.glob("*.jsonl"))
    return [f for f in files if sources is None or f.stem in set(sources)]


def conversation_table(filtered_dir: str | Path | None = None, sources: Iterable[str] | None = None) -> pd.DataFrame:
    """One row per kept conversation: source, modality, project, turn counts, length, tools, relevance."""
    folder = Path(filtered_dir) if filtered_dir else get_paths().filtered
    rows = []
    for path in _jsonl_files(folder, sources):
        for c in read_jsonl(path, Conversation):
            roles = Counter(t.role for t in c.turns)
            rows.append({
                "conversation_id": c.conversation_id,
                "source": c.source_dataset.value,
                "modality": c.modality.value,
                "project_id": c.project_id,
                "has_author": c.author_id is not None,
                "n_turns": len(c.turns),
                "n_user": roles[Role.USER],
                "n_assistant": roles[Role.ASSISTANT],
                "n_tool": roles[Role.TOOL],
                "approx_tokens": sum(len(t.content) for t in c.turns) // CHARS_PER_TOKEN,
                "tools": sorted({t.tool_name for t in c.turns if t.tool_name}),
                "relevance": "+".join(c.metadata.get("relevance") or ["none"]),
                "year": c.occurred_at.year if c.occurred_at else None,
            })
    return pd.DataFrame(rows)


def segment_table(segments_dir: str | Path | None = None, sources: Iterable[str] | None = None) -> pd.DataFrame:
    """One row per segment, lengths only (no text kept)."""
    folder = Path(segments_dir) if segments_dir else get_paths().segments
    rows = []
    for path in _jsonl_files(folder, sources):
        for s in read_jsonl(path, Segment):
            rows.append({
                "segment_id": s.segment_id,
                "conversation_id": s.conversation_id,
                "source": s.source_dataset.value,
                "modality": s.modality.value,
                "project_id": s.project_id,
                "has_context": bool(s.context_text),
                "text_tokens": len(s.text) // CHARS_PER_TOKEN,
                "context_tokens": len(s.context_text) // CHARS_PER_TOKEN,
                "truncated": "characters removed ..." in s.text,
            })
    return pd.DataFrame(rows)


def summary_tables(conv: pd.DataFrame, seg: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tables for the report: overview per source, relevance mix, top agent tools."""
    overview = conv.groupby("source").agg(
        modality=("modality", "first"),
        conversations=("conversation_id", "count"),
        projects=("project_id", "nunique"),
        median_turns=("n_turns", "median"),
        median_tokens=("approx_tokens", "median"),
    )
    seg_by = seg.groupby("source").agg(
        segments=("segment_id", "count"),
        median_segment_tokens=("text_tokens", "median"),
        share_with_context=("has_context", "mean"),
        share_truncated=("truncated", "mean"),
    )
    overview = overview.join(seg_by, how="outer").fillna(0)
    overview.loc["total"] = overview.sum(numeric_only=True)
    for col in ("modality",):
        overview.loc["total", col] = ""
    for col in ("median_turns", "median_tokens", "median_segment_tokens", "share_with_context", "share_truncated"):
        overview.loc["total", col] = None
    relevance = pd.crosstab(conv["source"], conv["relevance"])
    tools = conv[conv["modality"] == "agentic"].explode("tools").dropna(subset=["tools"])
    top_tools = tools.groupby(["source", "tools"]).size().rename("conversations").reset_index()
    top_tools = top_tools.sort_values(["source", "conversations"], ascending=[True, False]).groupby("source").head(10)
    return {"overview": overview, "relevance": relevance, "top_tools": top_tools}


def plot_distributions(conv: pd.DataFrame, seg: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    def save(fig, name: str) -> None:
        path = out / name
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        made.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    seg["source"].value_counts().sort_values().plot.barh(ax=ax, color="#4c72b0")
    ax.set_xlabel("segments")
    ax.set_title("Segments per source")
    save(fig, "segments_per_source.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for src, g in conv.groupby("source"):
        ax.hist(g["n_turns"].clip(upper=200), bins=40, alpha=0.5, label=src)
    ax.set_xlabel("turns per conversation (capped at 200)")
    ax.set_ylabel("conversations")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Conversation length")
    save(fig, "turns_per_conversation.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    for src, g in seg.groupby("source"):
        ax.hist(g["text_tokens"], bins=40, alpha=0.5, label=src)
    ax.axvline(512, color="black", linestyle="--", linewidth=1, label="model limit (512)")
    ax.set_xlabel("approx. tokens per segment")
    ax.set_ylabel("segments")
    ax.legend()
    ax.set_title("Segment length")
    save(fig, "segment_length.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    pd.crosstab(conv["source"], conv["relevance"]).plot.bar(stacked=True, ax=ax)
    ax.set_ylabel("conversations")
    ax.set_title("Relevance category per source")
    ax.tick_params(axis="x", rotation=0)
    save(fig, "relevance_per_source.png")
    return made


# --------------------------------------------------------------------------- 2. topics


def sample_segment_texts(
    segments_dir: str | Path | None = None, n: int = 20000, seed: int = 42, sources: Iterable[str] | None = None
) -> list[tuple[str, str, str]]:
    """Seeded reservoir sample of (segment_id, source, text) across all segment files."""
    folder = Path(segments_dir) if segments_dir else get_paths().segments
    rng = random.Random(seed)
    sample: list[tuple[str, str, str]] = []
    i = 0
    for path in _jsonl_files(folder, sources):
        for s in read_jsonl(path, Segment):
            item = (s.segment_id, s.source_dataset.value, s.text)
            if len(sample) < n:
                sample.append(item)
            else:
                j = rng.randint(0, i)
                if j < n:
                    sample[j] = item
            i += 1
    return sample


def topics(
    sample: list[tuple[str, str, str]],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit BERTopic. Returns (topic table with domain hints, per-segment assignments)."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    ids, sources, docs = zip(*sample) if sample else ((), (), ())
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=seed),
        hdbscan_model=HDBSCAN(min_cluster_size=min_topic_size, metric="euclidean", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2),
        calculate_probabilities=False,
    )
    assigned, _ = model.fit_transform(list(docs))
    assign = pd.DataFrame({"segment_id": ids, "source": sources, "topic": assigned})

    info = model.get_topic_info()
    rows = []
    for _, r in info.iterrows():
        t = int(r["Topic"])
        members = [d for d, a in zip(docs, assigned) if a == t]
        hint_counts = Counter(domain_hints(d)[0] for d in members)
        top_domain, top_n = hint_counts.most_common(1)[0] if hint_counts else (Domain.OTHER.value, 0)
        by_source = assign[assign["topic"] == t]["source"].value_counts()
        rows.append({
            "topic": t,
            "segments": int(r["Count"]),
            "top_words": ", ".join(w for w, _ in (model.get_topic(t) or [])[:8]),
            "domain_hint": top_domain,
            "domain_hint_share": round(top_n / max(len(members), 1), 2),
            "main_source": by_source.index[0] if len(by_source) else "",
        })
    return pd.DataFrame(rows), assign


def domain_coverage(topic_table: pd.DataFrame) -> pd.DataFrame:
    """Segments per hinted domain across non-outlier topics: a large 'other' share means a missing label."""
    t = topic_table[topic_table["topic"] != -1]
    cov = t.groupby("domain_hint")["segments"].sum().sort_values(ascending=False)
    return pd.DataFrame({"segments": cov, "share": (cov / max(cov.sum(), 1)).round(3)})


# --------------------------------------------------------------------------- 3. knowledge share


def knowledge_share(labels_path: str | Path) -> dict[str, Any]:
    """Share of knowledge-bearing segments among labelled ones, with a 95% Wilson interval (after Stage 4b)."""
    from krs.clean.precision_check import wilson

    labels = list(read_jsonl(labels_path, Label))
    k = sum(lab.knowledge_bearing for lab in labels)
    lo, hi = wilson(k, len(labels))
    types = Counter(t.value for lab in labels for t in lab.knowledge_types)
    return {"labelled": len(labels), "knowledge_bearing": k, "share": k / len(labels) if labels else 0.0,
            "ci95_low": lo, "ci95_high": hi, "knowledge_types": dict(types)}


# --------------------------------------------------------------------------- runner


def run_explore(config: str | Path = "explore", out_dir: str | Path | None = None) -> dict[str, Any]:
    cfg = load_config(config)
    out = Path(out_dir) if out_dir else RESULTS_DIR / "eda"
    out.mkdir(parents=True, exist_ok=True)

    conv = conversation_table()
    seg = segment_table()
    tables = summary_tables(conv, seg)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv")
    plots = plot_distributions(conv, seg, out)
    summary: dict[str, Any] = {
        "conversations": len(conv),
        "segments": len(seg),
        "projects": int(conv["project_id"].nunique()),
        "plots": [p.name for p in plots],
    }

    tcfg = cfg.get("topics") or {}
    if tcfg.get("enabled", True):
        sample = sample_segment_texts(n=tcfg.get("sample", 20000), seed=tcfg.get("seed", 42))
        topic_table, assign = topics(
            sample,
            embedding_model=tcfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            min_topic_size=tcfg.get("min_topic_size", 30),
            seed=tcfg.get("seed", 42),
        )
        topic_table.to_csv(out / "topics.csv", index=False)
        assign.to_csv(out / "topic_assignments.csv", index=False)
        cov = domain_coverage(topic_table)
        cov.to_csv(out / "topic_domain_coverage.csv")
        summary.update({
            "topic_sample": len(sample),
            "topics_found": int((topic_table["topic"] != -1).sum()),
            "outlier_share": round(float((assign["topic"] == -1).mean()), 3) if len(assign) else None,
            "other_domain_share": float(cov["share"].get(Domain.OTHER.value, 0.0)),
        })

    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
