"""App evaluation, part 2: the successor task (RQ4) - do knowledge cards help someone answer questions
about work they did not do?

Design (within-subject, counterbalanced):
- Each participant answers every question once: half WITH the knowledge system (ask-predecessor,
  search, handover pack), half WITHOUT it (the raw conversation logs only). Which half gets which
  condition alternates between participants, so every question is answered in both conditions.
- Participants record their answer and the minutes taken.
- A rater scores every answer BLIND: answers are shuffled and anonymised, the condition is hidden.
  Rubric 1-5: 1 wrong or no answer · 3 partly right · 5 correct, complete and specific.
- Analysis per participant (mean rating and minutes with vs without), Wilcoxon signed-rank test.
  With a handful of participants this is indicative only, and the report says so.

Participants are pseudonymous (P1, P2, ...); no personal data is stored. Get consent before the session.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np

from krs.config import RESULTS_DIR

CONDITIONS = ("with_cards", "without_cards")


def _out(root: str | Path | None = None) -> Path:
    path = (Path(root) if root else RESULTS_DIR) / "app_eval" / "successor"
    path.mkdir(parents=True, exist_ok=True)
    return path


def assign(question_ids: list[str], participants: list[str], seed: int = 42) -> dict[str, dict[str, str]]:
    """participant -> {question_id: condition}. Halves alternate between participants (counterbalanced)."""
    qs = sorted(question_ids)
    random.Random(seed).shuffle(qs)
    half = set(qs[: len(qs) // 2])
    plan = {}
    for i, p in enumerate(participants):
        first, second = CONDITIONS if i % 2 == 0 else CONDITIONS[::-1]
        plan[p] = {q: (first if q in half else second) for q in qs}
    return plan


def export_participant_sheets(questions: dict[str, str], participants: list[str], seed: int = 42,
                              root: str | Path | None = None) -> dict[str, Path]:
    """One CSV per participant: question, condition, answer (blank), minutes (blank). Order shuffled per person."""
    plan = assign(list(questions), participants, seed)
    paths = {}
    for i, (p, conds) in enumerate(plan.items()):
        order = sorted(conds)
        random.Random(seed + i + 1).shuffle(order)
        path = _out(root) / f"participant_{p}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["participant", "question_id", "question", "condition", "answer", "minutes"])
            for q in order:
                w.writerow([p, q, questions[q], conds[q], "", ""])
        paths[p] = path
    return paths


def read_responses(paths: list[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                minutes = r.get("minutes", "").strip()
                rows.append({**r, "minutes": float(minutes) if minutes else None})
    return rows


def export_rating_sheet(responses: list[dict[str, Any]], seed: int = 42, root: str | Path | None = None) -> tuple[Path, Path]:
    """Blind sheet for the rater (no participant, no condition, shuffled) and a private key file."""
    rows = [r for r in responses if r.get("answer", "").strip()]
    random.Random(seed).shuffle(rows)
    sheet, key = _out(root) / "rating_sheet.csv", _out(root) / "rating_key.csv"
    with open(sheet, "w", newline="", encoding="utf-8") as fs, open(key, "w", newline="", encoding="utf-8") as fk:
        ws, wk = csv.writer(fs), csv.writer(fk)
        ws.writerow(["answer_id", "question", "answer", "rating_1_to_5", "notes"])
        wk.writerow(["answer_id", "participant", "question_id", "condition", "minutes"])
        for i, r in enumerate(rows):
            aid = f"a{i:04d}"
            ws.writerow([aid, r["question"], r["answer"], "", ""])
            wk.writerow([aid, r["participant"], r["question_id"], r["condition"], r["minutes"] if r["minutes"] is not None else ""])
    return sheet, key


def analyse(rating_sheet: str | Path | None = None, rating_key: str | Path | None = None,
            root: str | Path | None = None) -> dict[str, Any]:
    """Join blind ratings to the key; compare conditions within participants."""
    from scipy.stats import wilcoxon

    sheet = Path(rating_sheet) if rating_sheet else _out(root) / "rating_sheet.csv"
    key = Path(rating_key) if rating_key else _out(root) / "rating_key.csv"
    with open(sheet, newline="", encoding="utf-8") as f:
        ratings = {r["answer_id"]: float(r["rating_1_to_5"]) for r in csv.DictReader(f) if r["rating_1_to_5"].strip()}
    with open(key, newline="", encoding="utf-8") as f:
        keyed = [r for r in csv.DictReader(f) if r["answer_id"] in ratings]
    per_person: dict[str, dict[str, dict[str, list[float]]]] = {}
    for r in keyed:
        d = per_person.setdefault(r["participant"], {c: {"rating": [], "minutes": []} for c in CONDITIONS})
        d[r["condition"]]["rating"].append(ratings[r["answer_id"]])
        if r["minutes"]:
            d[r["condition"]]["minutes"].append(float(r["minutes"]))

    def summary(metric: str) -> dict[str, Any]:
        pairs = [(np.mean(v["with_cards"][metric]), np.mean(v["without_cards"][metric]))
                 for v in per_person.values() if v["with_cards"][metric] and v["without_cards"][metric]]
        out: dict[str, Any] = {"participants_paired": len(pairs)}
        if pairs:
            w, wo = np.array(pairs).T
            out.update({"with_cards_mean": round(float(w.mean()), 3), "without_cards_mean": round(float(wo.mean()), 3),
                        "mean_diff_with_minus_without": round(float((w - wo).mean()), 3)})
            if len(pairs) >= 2 and np.any(w != wo):
                out["wilcoxon_p"] = round(float(wilcoxon(w, wo).pvalue), 4)
        return out

    n = len(per_person)
    return {"participants": n, "ratings": len(ratings), "rating": summary("rating"), "minutes": summary("minutes"),
            "note": "indicative only: small sample" if n < 10 else ""}
