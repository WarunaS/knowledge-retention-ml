"""X1 - transfer test (the drift check): does a model trained on public logs work on real agentic logs? (RQ3)

Workflow (all local; own logs never leave the machine and are redacted in Stage 3):
1. `export_transfer_sheet()` - pick ~150 own-log segments and write a labelling sheet (same format as gold)
2. a person labels it; `import_transfer_sheet()` writes labels/transfer.jsonl
3. Stage 6 (`make_splits`) puts them in the transfer split
4. `run()` - score them ONCE with the current registered model and compare with the public gold set

Run once: the result is locked. A second run is refused unless `force=True` with a written reason;
every forced re-run is recorded, so the report can state honestly whether X1 was repeated.

Reported: macro-F1 per axis on transfer, on the whole gold set and on its agentic part (own logs are
agent sessions, so the agentic gold subset is the fairer reference), the drop with a 95% CI
(conversation bootstrap of both sets), per-label F1 on both sides, label-frequency shift and length
shift. The report contains ids and numbers only, never conversation text.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from krs.config import RESULTS_DIR, get_paths, load_config
from krs.evaluate.metrics import per_label
from krs.evaluate.stats import bootstrap_ci, unpaired_bootstrap_diff
from krs.features import label_matrices, load_rows
from krs.io import write_jsonl
from krs.label.gold import export_rater_sheet, import_rater_sheet
from krs.label.sampling import segment_refs, stratified_sample
from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES
from krs.models.common import NONE_IDX, decode
from krs.schema import LabelledSegment, LabelSource, Segment

AXES = {"knowledge_types": KNOWLEDGE_TYPES, "domains": DOMAINS}


class TransferLockError(RuntimeError):
    pass


# --------------------------------------------------------------------------- labelling own logs


def export_transfer_sheet(n: int = 150, seed: int = 42, source: str = "own_logs",
                          segments_dir: str | Path | None = None, out_csv: str | Path | None = None) -> int:
    """Sheet of up to n own-log segments, spread over conversations, for hand labelling."""
    from krs.label.sampling import load_segments

    refs = [r for r in segment_refs(segments_dir) if r.source == source]
    if not refs:
        raise FileNotFoundError(f"no '{source}' segments - collect own logs (Stage 2) and clean them (Stage 3)")
    chosen = stratified_sample(refs, n, seed)
    segs = load_segments([r.segment_id for r in chosen], segments_dir)
    out = Path(out_csv) if out_csv else get_paths().labels / "transfer_sheet.csv"
    return export_rater_sheet([segs[r.segment_id] for r in sorted(chosen, key=lambda r: r.segment_id)], out)


def import_transfer_sheet(csv_path: str | Path | None = None, rater_id: str = "rater1",
                          out: str | Path | None = None) -> tuple[int, list[str]]:
    """Read the labelled sheet into labels/transfer.jsonl. Returns (labels written, problems)."""
    labels_dir = get_paths().labels
    labels, problems = import_rater_sheet(csv_path or labels_dir / "transfer_sheet.csv", LabelSource.RATER_1, rater_id)
    write_jsonl(Path(out) if out else labels_dir / "transfer.jsonl", labels)
    return len(labels), problems


# --------------------------------------------------------------------------- the test


def _arrays(rows: Sequence[LabelledSegment], predictor: Any) -> dict[str, Any]:
    segs: list[Segment] = [r.segment for r in rows]
    p_kt, p_dom = predictor.predict_proba(segs)
    kt_hat, dom_hat = decode(p_kt, p_dom, predictor.th_kt, predictor.th_dom)
    kt, dom = label_matrices(rows)
    return {"knowledge_types": (kt.astype(np.int64), kt_hat.astype(np.int64)),
            "domains": (dom.astype(np.int64), dom_hat.astype(np.int64)),
            "groups": np.array([s.conversation_id for s in segs]),
            "segment_ids": [s.segment_id for s in segs],
            "chars": np.array([len(s.text) for s in segs]),
            "p": (p_kt, p_dom)}


def _subset(a: dict[str, Any], mask: np.ndarray) -> dict[str, Any]:
    return {**a, **{ax: (a[ax][0][mask], a[ax][1][mask]) for ax in AXES}, "groups": a["groups"][mask],
            "chars": a["chars"][mask]}


def _prevalence(y: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    return {n: round(float(v), 3) for n, v in zip(names, y.mean(0), strict=True)} if len(y) else {}


def _knowledge_bearing_f1(a: dict[str, Any]) -> float:
    y, yhat = a["knowledge_types"]
    return per_label(1 - y[:, [NONE_IDX]], 1 - yhat[:, [NONE_IDX]], ["kb"])[0]["f1"]


def run(config: str | Path = "runs/X1", force: bool = False, reason: str | None = None,
        models_dir: str | Path | None = None, results_root: str | Path | None = None,
        splits_dir: str | Path | None = None, predictor: Any = None) -> dict[str, Any]:
    cfg = load_config(config)
    out = (Path(results_root) if results_root else RESULTS_DIR) / cfg.get("run_id", "X1")
    lock_path = out / "x1_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    if lock and not force:
        raise TransferLockError(f"X1 already ran on {lock['runs'][0]['at']} with {lock['runs'][0]['model_version']}; "
                                "it runs once. Re-run only with force=True and a reason (it will be recorded).")
    if lock and force and not reason:
        raise TransferLockError("a forced re-run of X1 needs a written reason")

    transfer = load_rows("transfer", splits_dir)
    gold = load_rows("test", splits_dir)
    if not transfer:
        raise FileNotFoundError("no transfer split - label own logs (export/import_transfer_sheet) and re-run Stage 6")
    if not gold:
        raise FileNotFoundError("no test split (gold) to compare against")
    if predictor is None:
        from krs.inference import load_predictor

        predictor = load_predictor(cfg.get("model_version"), models_dir)

    n_boot, seed = cfg.get("bootstrap_n", 1000), cfg.get("seed", 42)
    T, G = _arrays(transfer, predictor), _arrays(gold, predictor)
    agentic = np.array([r.segment.modality.value == "agentic" for r in gold])
    G_ag = _subset(G, agentic) if agentic.any() else None

    report: dict[str, Any] = {
        "model_version": predictor.version, "run_id": predictor.run_id,
        "transfer_segments": len(transfer), "transfer_conversations": len(set(T["groups"])),
        "gold_segments": len(gold), "gold_agentic_segments": int(agentic.sum()),
        "bootstrap": {"n": n_boot, "by": "conversation"},
        "knowledge_bearing_f1": {"transfer": _knowledge_bearing_f1(T), "gold": _knowledge_bearing_f1(G)},
        "length_chars_median": {"transfer": float(np.median(T["chars"])), "gold": float(np.median(G["chars"]))},
    }
    for ax, names in AXES.items():
        yt, pt = T[ax]
        yg, pg = G[ax]
        entry: dict[str, Any] = {
            "transfer": bootstrap_ci(yt, pt, T["groups"], n_boot, seed),
            "gold": bootstrap_ci(yg, pg, G["groups"], n_boot, seed),
            "drop_gold_minus_transfer": unpaired_bootstrap_diff(yg, pg, G["groups"], yt, pt, T["groups"], n_boot, seed),
            "per_label_transfer": per_label(yt, pt, names),
            "per_label_gold": per_label(yg, pg, names),
            "label_share_transfer": _prevalence(yt, names),
            "label_share_gold": _prevalence(yg, names),
        }
        if G_ag is not None:
            ya, pa = G_ag[ax]
            entry["gold_agentic"] = bootstrap_ci(ya, pa, G_ag["groups"], n_boot, seed)
            entry["drop_gold_agentic_minus_transfer"] = unpaired_bootstrap_diff(
                ya, pa, G_ag["groups"], yt, pt, T["groups"], n_boot, seed)
        report[ax] = entry

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "predictions_transfer.npz", segment_ids=np.array(T["segment_ids"]),
                        p_knowledge_types=T["p"][0].astype(np.float32), p_domains=T["p"][1].astype(np.float32),
                        knowledge_types=T["knowledge_types"][1], domains=T["domains"][1])
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model_version": predictor.version,
             "forced": bool(lock), "reason": reason}
    lock = {"runs": (lock["runs"] if lock else []) + [entry]}
    lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    report["runs_so_far"] = len(lock["runs"])
    return report
