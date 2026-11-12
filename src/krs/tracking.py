"""Stage 11 - Track experiments, so every reported number can be traced to what produced it.

Every run folder (results/<run_id>/) gets a run_info.json next to its config, thresholds, predictions
and validation metrics: time, git commit (and whether the code had uncommitted changes), library
versions, and fingerprints of the dataset manifest and split report.

`index_runs()` collects all run folders into one table: results/runs_index.csv.
Plain files, no tracking server; MLflow stays optional, as in the plan.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

from krs.config import REPO_ROOT, RESULTS_DIR, get_paths

PACKAGES = ("knowledge-retention-ml", "torch", "transformers", "scikit-learn", "sentence-transformers",
            "setfit", "numpy", "anthropic")


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10,
                              check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _fingerprint(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.exists() else None


def run_info() -> dict[str, Any]:
    versions = {}
    for pkg in PACKAGES:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue
    paths = get_paths()
    dirty = _git("status", "--porcelain")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_uncommitted_changes": bool(dirty) if dirty is not None else None,
        "python": platform.python_version(),
        "packages": versions,
        "data": {
            "manifest_sha256": _fingerprint(paths.manifest),
            "split_report_sha256": _fingerprint(paths.splits / "split_report.json"),
        },
    }


def write_run_info(out: Path) -> dict[str, Any]:
    info = run_info()
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def index_runs(results_root: str | Path | None = None, write: bool = True):
    """One row per run folder: model, validation scores, commit, time. Skips folders that are not runs."""
    import pandas as pd

    root = Path(results_root) if results_root else RESULTS_DIR
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []:
        mfile = folder / "metrics_validation.json"
        if not mfile.exists():
            continue
        m = json.loads(mfile.read_text(encoding="utf-8"))
        info = json.loads((folder / "run_info.json").read_text(encoding="utf-8")) if (folder / "run_info.json").exists() else {}
        rows.append({
            "run_id": folder.name,
            "model": m.get("model"),
            "with_context": m.get("with_context"),
            "val_macro_f1_knowledge_types": m.get("macro_f1_knowledge_types"),
            "val_macro_f1_domains": m.get("macro_f1_domains"),
            "val_macro_f1_mean": m.get("macro_f1_mean"),
            "has_model": (folder / "model").exists(),
            "has_test_predictions": (folder / "predictions_test.npz").exists(),
            "git_commit": (info.get("git_commit") or "")[:10],
            "uncommitted_changes": info.get("git_uncommitted_changes"),
            "created_at": info.get("created_at"),
        })
    df = pd.DataFrame(rows)
    if write and len(df):
        df.to_csv(root / "runs_index.csv", index=False)
    return df
