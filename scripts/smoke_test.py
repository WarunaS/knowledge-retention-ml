"""End-to-end smoke test on tiny real data: no API cost, no GPU, a few minutes on a laptop.

    python scripts/smoke_test.py            # writes everything under ./smoke/ (gitignored)
    python scripts/smoke_test.py --topics   # also run BERTopic (adds ~20 s)

Collect (small) -> clean -> explore -> PLACEHOLDER labels -> split -> features -> baselines B0-B2 ->
evaluate -> track/register -> inference -> cards (local backend) -> store -> ask/handover -> retrieval eval.

The labels are RANDOM placeholders so the pipeline can run without paying for LLM labels or waiting for
human gold labels. Every score it prints is meaningless; it checks that the stages work together.
Not covered here (GPU, money or consent needed): M1/A1/A2/M2 fine-tuning, teacher T, X1, Claude backends.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def step(name: str) -> float:
    print(f"\n=== {name}", flush=True)
    return time.perf_counter()


def done(t0: float, **info) -> None:
    extra = " | ".join(f"{k}: {v}" for k, v in info.items())
    print(f"    ok in {time.perf_counter() - t0:.1f}s" + (f" | {extra}" if extra else ""), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "smoke"))
    ap.add_argument("--topics", action="store_true")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    os.environ["KRS_DATA_ROOT"] = str(out / "data")
    os.environ["KRS_RESULTS_DIR"] = str(out / "results")
    os.environ["KRS_MODELS_DIR"] = str(out / "models")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "src"))

    import yaml

    from krs.config import CONFIGS_DIR, get_paths

    cfg_dir = out / "configs"
    cfg_dir.mkdir(exist_ok=True)

    t = step("2 Collect (WildChat 300, SWE-agent 30, OpenHands 30; DevGPT skipped: 680 MB download)")
    from krs.collect.pipeline import run_collect

    collect = yaml.safe_load((CONFIGS_DIR / "collect.yaml").read_text())
    collect["datasets"]["wildchat"]["limit"] = 300
    for name in ("swe_agent", "openhands"):
        collect["datasets"][name]["limit"] = 30
        collect["datasets"][name]["rows_per_group"] = 5
    collect["datasets"]["devgpt"]["enabled"] = False
    (cfg_dir / "collect.yaml").write_text(yaml.safe_dump(collect))
    done(t, **run_collect(cfg_dir / "collect.yaml"))

    t = step("3 Clean")
    from krs.clean.pipeline import run_cleaning

    rep = run_cleaning("clean")
    done(t, **{k: v.get("segments", 0) for k, v in rep.items()})

    t = step("4 Explore" + (" (with topics)" if args.topics else " (topics off)"))
    from krs.explore import run_explore

    (cfg_dir / "explore.yaml").write_text(yaml.safe_dump({"topics": {"enabled": args.topics, "sample": 500,
                                                                     "min_topic_size": 10, "seed": 42}}))
    done(t, **{k: v for k, v in run_explore(cfg_dir / "explore.yaml").items() if k in ("segments", "topics_found")})

    t = step("4b Label - PLACEHOLDER random labels (real run: notebook 05)")
    from krs.io import write_jsonl
    from krs.label import sampling
    from krs.label.taxonomy import Domain, KnowledgeType
    from krs.schema import Label, LabelSource

    refs = sampling.segment_refs()
    gold_ids = set(sampling.select_gold(min(120, len(refs) // 3), seed=42))
    gold_convs = {r.conversation_id for r in refs if r.segment_id in gold_ids}
    rng = random.Random(0)

    def fake(sid: str, src: LabelSource) -> Label:
        return Label(segment_id=sid, knowledge_types=[rng.choice(list(KnowledgeType))],
                     domains=[rng.choice(list(Domain))], source=src)

    write_jsonl(get_paths().labels / "weak.jsonl", [fake(r.segment_id, LabelSource.LLM) for r in refs
                                                     if r.conversation_id not in gold_convs])
    write_jsonl(get_paths().labels / "gold.jsonl", [fake(i, LabelSource.ADJUDICATED) for i in sorted(gold_ids)])
    done(t, gold=len(gold_ids))

    t = step("6 Split")
    from krs.split import make_splits

    sp = make_splits("split")
    done(t, **{n: sp[n]["segments"] for n in ("train", "validation", "test")})

    t = step("5 Features (TF-IDF, MiniLM embeddings, DeBERTa token lengths)")
    from krs.features import build_features

    done(t, **build_features("features")["splits"])

    t = step("7 Baselines B0, B1, B2 (C and thresholds tuned on validation)")
    from krs.models.runner import run_experiment

    done(t, **{r: run_experiment(r)["macro_f1_mean"] for r in ("B0", "B1", "B2")})

    t = step("9 Evaluate on the gold split (1,000 conversation-bootstrap resamples)")
    from krs.evaluate.metrics import evaluate_runs

    ev = evaluate_runs("evaluate")
    done(t, evaluated=ev["runs_evaluated"], missing=ev["runs_missing"])

    t = step("11-12 Track and register")
    from krs import registry, tracking

    tracking.index_runs()
    version = registry.register(registry.best_run(["B2", "B1", "B0"]))
    done(t, registered=version)

    t = step("14 Inference (CPU)")
    from krs.inference import classify

    inf = classify("inference")
    done(t, scored=inf["scored"], seconds_per_1k=inf["seconds_per_1k"])

    t = step("App: cards (local backend), store, ask, handover")
    from itertools import islice

    from krs.app import extraction
    from krs.app.agent import ask
    from krs.app.common import app_config
    from krs.app.handover import build_pack
    from krs.app.storage import KnowledgeStore
    from krs.io import read_jsonl
    from krs.schema import KnowledgeCard, Prediction

    app_cfg = app_config()
    app_cfg["extraction"]["backend"] = "local"
    app_cfg["answer"]["backend"] = "local"
    preds = out / "data" / "preds_sample.jsonl"
    write_jsonl(preds, islice(read_jsonl(inf["output"], Prediction), 150))
    cards_file = out / "data" / "cards.jsonl"
    ex = extraction.extract_cards(preds, cards_file, app_cfg)
    store = KnowledgeStore()
    store.add(read_jsonl(cards_file, KnowledgeCard))
    answer = ask(store, "how do I fix a failing test", cfg=app_cfg)
    project = next((c.project_id for c in store.all() if c.project_id), None)
    pack = build_pack(store, project_id=project) if project else ""
    (out / "results" / "handover_example.md").write_text(pack)
    done(t, cards=ex["cards"], rejected=ex["rejected"], answer_backend=answer["backend"], handover_entries=pack.count("Source: card"))

    t = step("App evaluation (questions = card titles: inflated scores, code check only)")
    import csv

    from krs.evaluate import retrieval_eval as RE

    RE.export_query_sheet(store, 30)
    sheet = RE.out_dir() / "queries.csv"
    with open(sheet, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["question"] = r["card_title"]
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rr = RE.evaluate_retrieval(store, RE.load_queries(sheet))
    done(t, semantic_mrr=rr["modes"]["semantic"]["mrr"], filtered_mrr=rr["modes"]["filtered"]["mrr"])

    print(f"\nSMOKE TEST PASSED. Outputs under {out}")
    print("  results table: results/eval/results_table.md · model card: models/*/model_card.md")
    print("  handover example: results/handover_example.md · dashboard: KRS_DATA_ROOT=... streamlit run src/krs/app/dashboard.py")


if __name__ == "__main__":
    main()
