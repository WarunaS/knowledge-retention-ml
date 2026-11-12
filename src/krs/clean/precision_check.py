"""Hand check of the relevance filter: how many kept conversations are really relevant?

1. export_sample() writes a CSV of 200 randomly chosen kept WildChat conversations.
2. A person fills the `is_relevant` column with y or n.
3. precision() reads it back and returns the precision with a 95% Wilson interval.
The number goes in the report; if it is low, a zero-shot classifier is added to the filter.
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from krs.io import read_jsonl
from krs.schema import Conversation, Role

PREVIEW_CHARS = 800


def export_sample(filtered_path: str | Path, out_csv: str | Path, n: int = 200, seed: int = 42) -> int:
    """Reservoir-sample `n` conversations from a filtered JSONL file into a CSV for hand checking."""
    rng = random.Random(seed)
    sample: list[Conversation] = []
    for i, conv in enumerate(read_jsonl(filtered_path, Conversation)):
        if len(sample) < n:
            sample.append(conv)
        else:
            j = rng.randint(0, i)
            if j < n:
                sample[j] = conv
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conversation_id", "relevance", "first_user_turn", "is_relevant"])
        for c in sample:
            first = next((t.content for t in c.turns if t.role == Role.USER), "")
            preview = first[:PREVIEW_CHARS].replace("\n", " ⏎ ")
            w.writerow([c.conversation_id, "+".join(c.metadata.get("relevance", [])), preview, ""])
    return len(sample)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def precision(csv_path: str | Path) -> dict[str, float | int]:
    """Precision of the relevance filter from a hand-filled CSV (y/n in `is_relevant`)."""
    with open(csv_path, encoding="utf-8") as f:
        answers = [r["is_relevant"].strip().lower() for r in csv.DictReader(f)]
    checked = [a for a in answers if a in ("y", "n")]
    k = sum(a == "y" for a in checked)
    lo, hi = wilson(k, len(checked))
    return {
        "checked": len(checked),
        "relevant": k,
        "precision": k / len(checked) if checked else 0.0,
        "ci95_low": lo,
        "ci95_high": hi,
        "unchecked": len(answers) - len(checked),
    }
