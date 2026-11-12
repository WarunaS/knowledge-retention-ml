"""Paths and config loading.

Locally, data lives in `<repo>/data`. On Colab, set KRS_DATA_ROOT to a Google Drive folder
so each stage's output survives a disconnect:

    import os; os.environ["KRS_DATA_ROOT"] = "/content/drive/MyDrive/krs/data"
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:  # stage 2 output, immutable
        return self.root / "raw"

    @property
    def filtered(self) -> Path:  # stage 3 output: Conversation records after filters + dedup
        return self.root / "filtered"

    @property
    def segments(self) -> Path:  # stage 3 output: Segment records
        return self.root / "segments"

    @property
    def labels(self) -> Path:  # stage 4b output
        return self.root / "labels"

    @property
    def splits(self) -> Path:  # stage 6 output
        return self.root / "splits"

    @property
    def features(self) -> Path:  # stage 5 output: feature matrices and label matrices per split
        return self.root / "features"

    @property
    def predictions(self) -> Path:  # stage 14 output: Prediction records per model version
        return self.root / "predictions"

    @property
    def store(self) -> Path:  # application plane: cards database, vectors, graph
        return self.root / "store"

    @property
    def manifest(self) -> Path:
        return self.raw / "manifest.json"


def get_paths() -> Paths:
    root = Path(os.environ.get("KRS_DATA_ROOT", REPO_ROOT / "data"))
    return Paths(root=root)


MODELS_DIR = Path(os.environ.get("KRS_MODELS_DIR", REPO_ROOT / "models"))
RESULTS_DIR = Path(os.environ.get("KRS_RESULTS_DIR", REPO_ROOT / "results"))
CONFIGS_DIR = REPO_ROOT / "configs"


def load_config(name_or_path: str | Path) -> dict[str, Any]:
    """Load a YAML config by path, or by name relative to `configs/` (e.g. 'clean' or 'runs/B1')."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIGS_DIR / f"{name_or_path}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
