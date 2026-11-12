"""Knowledge map: what the organisation knows, and where that knowledge is fragile.

Describes knowledge, never people (ethics section of the plans):
- coverage: cards per domain and knowledge type
- single-person dependency: areas where most cards come from one contributor. Reported as a share
  and a number of contributors only - the contributor is never named, and no one is ranked or scored.
- staleness: areas with no new cards for a while
Also gathers the model results that exist so far (results table, comparisons, model card, X1, app eval).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from krs.config import MODELS_DIR, RESULTS_DIR
from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES
from krs.schema import KnowledgeCard


def coverage(cards: list[KnowledgeCard]) -> pd.DataFrame:
    """Cards per domain (rows) and knowledge type (columns); 'none' is not a knowledge type to cover."""
    types = [k for k in KNOWLEDGE_TYPES if k != "none"]
    table = pd.DataFrame(0, index=DOMAINS, columns=types)
    for c in cards:
        for d in c.domains:
            for k in c.knowledge_types:
                if k.value in types:
                    table.loc[d.value, k.value] += 1
    table["total"] = table.sum(axis=1)
    return table


def _areas(c: KnowledgeCard, by: str) -> list[str]:
    if by == "domain":
        return [d.value for d in c.domains]
    return [c.project_id] if c.project_id else []


def single_person_dependency(cards: list[KnowledgeCard], by: str = "domain", min_cards: int = 5,
                             threshold: float = 0.8) -> pd.DataFrame:
    """Areas where one contributor holds at least `threshold` of the cards. No contributor is named."""
    per_area: dict[str, Counter] = defaultdict(Counter)
    for c in cards:
        if c.author_id:
            for a in _areas(c, by):
                per_area[a][c.author_id] += 1
    rows = []
    for area, counts in per_area.items():
        total = sum(counts.values())
        top = counts.most_common(1)[0][1]
        rows.append({by: area, "cards": total, "contributors": len(counts), "largest_share": round(top / total, 2),
                     "at_risk": total >= min_cards and top / total >= threshold})
    cols = [by, "cards", "contributors", "largest_share", "at_risk"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows, columns=cols).sort_values(["at_risk", "largest_share"], ascending=False).reset_index(drop=True)


def staleness(cards: list[KnowledgeCard], by: str = "domain", months: int = 4,
              now: datetime | None = None) -> pd.DataFrame:
    """Latest card date per area; stale if nothing new for `months`."""
    now = now or datetime.now(timezone.utc)
    latest: dict[str, datetime] = {}
    for c in cards:
        if c.occurred_at:
            for a in _areas(c, by):
                latest[a] = max(latest.get(a, c.occurred_at), c.occurred_at)
    cutoff = now - timedelta(days=30 * months)
    rows = [{by: a, "latest_card": d.date().isoformat(), "stale": d < cutoff} for a, d in latest.items()]
    if not rows:
        return pd.DataFrame(columns=[by, "latest_card", "stale"])
    return pd.DataFrame(rows).sort_values("latest_card").reset_index(drop=True)


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def model_results(results_root: str | Path | None = None, models_dir: str | Path | None = None) -> dict[str, Any]:
    """Whatever results exist so far; missing pieces are simply absent."""
    res = Path(results_root) if results_root else RESULTS_DIR
    models = Path(models_dir) if models_dir else MODELS_DIR
    out: dict[str, Any] = {}
    if (res / "eval" / "results_table.csv").exists():
        out["results_table"] = pd.read_csv(res / "eval" / "results_table.csv")
    summary = _json(res / "eval" / "summary.json")
    if summary:
        out["comparisons"] = summary.get("comparisons", [])
    for name in ("learning_curve.png",):
        if (res / "eval" / name).exists():
            out["learning_curve"] = res / "eval" / name
    out["confusions"] = sorted((res / "eval").glob("confusion_*.png")) if (res / "eval").exists() else []
    registry = _json(models / "registry.json")
    if registry and registry.get("current") and (models / registry["current"] / "model_card.md").exists():
        out["model_version"] = registry["current"]
        out["model_card"] = (models / registry["current"] / "model_card.md").read_text(encoding="utf-8")
    x1 = _json(res / "X1" / "report.json")
    if x1:
        out["transfer"] = {ax: {"transfer": x1[ax]["transfer"]["macro_f1"], "gold": x1[ax]["gold"]["macro_f1"],
                                "drop": x1[ax]["drop_gold_minus_transfer"]} for ax in ("knowledge_types", "domains")}
    retrieval = _json(res / "app_eval" / "retrieval.json")
    if retrieval:
        out["retrieval"] = retrieval
    return out
