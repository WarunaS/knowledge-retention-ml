"""Stage 12 - Register / version a trained model.

    from krs.registry import best_run, register
    run = best_run(["M1", "B2", "B1"])     # chosen on VALIDATION, as the plan requires
    version = register(run)                # refused unless the run was evaluated on gold (Stage 9)

A registered version is a folder models/<version>/ holding the model files, its config, thresholds,
validation and gold metrics, run info (commit, versions, data fingerprints) and a model card
(Mitchell et al., 2019). models/registry.json lists every version and marks the current one, which
batch inference (Stage 14) and the transfer test (X1) load.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from krs.config import MODELS_DIR, RESULTS_DIR, get_paths


class RegistrationError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def best_run(candidates: list[str], results_root: str | Path | None = None,
             metric: str = "macro_f1_mean") -> str:
    """The candidate with the highest VALIDATION score among runs that saved a model."""
    root = Path(results_root) if results_root else RESULTS_DIR
    scored = []
    for r in candidates:
        m = _read_json(root / r / "metrics_validation.json")
        if m and (root / r / "model").exists() and m.get(metric) is not None:
            scored.append((m[metric], r))
    if not scored:
        raise RegistrationError(f"none of {candidates} has a saved model and validation metrics")
    return max(scored)[1]


def _registry(models_dir: Path) -> dict[str, Any]:
    return _read_json(models_dir / "registry.json") or {"versions": [], "current": None}


def _next_version(run_id: str, reg: dict[str, Any]) -> str:
    n = sum(v["run_id"] == run_id for v in reg["versions"]) + 1
    return f"{run_id.lower()}-v{n}"


def _fmt(axis: dict[str, Any]) -> str:
    return f"{axis['macro_f1']:.3f} (95% CI {axis['ci_low']:.3f}–{axis['ci_high']:.3f})"


def model_card(version: str, run_id: str, cfg: dict[str, Any], val: dict[str, Any], gold: dict[str, Any],
               info: dict[str, Any], split_report: dict[str, Any], comparisons: list[dict[str, Any]]) -> str:
    kt, dom = gold["knowledge_types"], gold["domains"]
    weakest = sorted((x for x in kt["per_label"] + dom["per_label"] if x["support"] > 0), key=lambda x: x["f1"])[:3]
    lines = [
        f"# Model card - {version}",
        "",
        "## Model details",
        f"- Run: `{run_id}` ({cfg.get('purpose', '')})",
        f"- Model: {cfg.get('base_model', cfg.get('model'))}; input with context turn: {cfg.get('with_context', True)}",
        f"- Registered: {datetime.now(timezone.utc).date().isoformat()}; code commit `{(info.get('git_commit') or '?')[:10]}`"
        + (" (with uncommitted changes)" if info.get("git_uncommitted_changes") else ""),
        ("- Outputs: probabilities for 5 knowledge types and 12 technical domains; labels via per-label thresholds "
        "tuned on validation and the codebook rules (none alone, at least one label per axis)."),
        "",
        "## Intended use",
        ("- Flag knowledge-bearing segments of developer–AI conversations and label them, so knowledge cards can be "
        "extracted and retrieved by a successor (onboarding and knowledge retention)."),
        "- Runs locally on CPU: logs need not leave the organisation.",
        "- Not for monitoring, scoring or ranking individual employees.",
        "",
        "## Training data",
        ("- LLM weak labels (Claude, codebook v1) on public developer–LLM conversations: WildChat-1M, DevGPT, "
        "SWE-agent and OpenHands trajectories."),
        (f"- Train segments: {split_report.get('train', {}).get('segments', '?')}; validation: "
        f"{split_report.get('validation', {}).get('segments', '?')} (split by conversation)."),
        "",
        "## Evaluation (human gold set only)",
        f"- Gold segments: {gold.get('segments')}; confidence intervals by conversation bootstrap.",
        f"- Knowledge-type macro-F1: {_fmt(kt)}",
        f"- Domain macro-F1: {_fmt(dom)}",
        f"- Knowledge-bearing F1: {gold.get('knowledge_bearing_f1')}",
        f"- Validation macro-F1 (model selection only): {val.get('macro_f1_mean')}",
        "- Weakest labels: " + ", ".join(f"{x['label']} (F1 {x['f1']:.2f}, n={x['support']})" for x in weakest),
        "- Macro-F1 per source (knowledge type): " + ", ".join(f"{s} {v:.2f}" for s, v in kt["macro_f1_by_source"].items()),
    ]
    rel = [c for c in comparisons if c.get("a") == run_id and "knowledge_types" in c]
    if rel:
        lines += ["", "## Comparisons (paired bootstrap, knowledge-type axis)"]
        for c in rel:
            r = c["knowledge_types"]
            verdict = r.get("a_better", r.get("within_margin"))
            lines.append(f"- {c['hypothesis']}: {c['a']} − {c['b']} = {r['diff']:+.3f} (95% CI {r['ci_low']:+.3f} to "
                         f"{r['ci_high']:+.3f}); supported: {verdict}")
    lines += [
        "",
        "## Limitations",
        "- Trained on LLM weak labels, not human labels; label noise is estimated from a 100-item check.",
        "- Public chatbot and agent logs are a proxy for workplace logs; see the transfer test (X1) for the gap.",
        "- Inputs longer than 512 tokens are cut, context first; long segments lose their middle.",
        "- Rare labels (decision, contextual, business_requirements) have few gold examples and wide intervals.",
        "",
        "## Ethical considerations",
        "- Own logs are used only with written consent, redacted, and processed locally.",
        "- Outputs describe knowledge in conversations, never people; individuals are not ranked or scored.",
        "",
    ]
    return "\n".join(lines)


def register(run_id: str, version: str | None = None, results_root: str | Path | None = None,
             models_dir: str | Path | None = None, eval_dir: str | Path | None = None,
             make_current: bool = True) -> str:
    root = Path(results_root) if results_root else RESULTS_DIR
    run = root / run_id
    models = Path(models_dir) if models_dir else MODELS_DIR
    ev = Path(eval_dir) if eval_dir else root / "eval"
    if not (run / "model").exists():
        raise RegistrationError(f"{run_id}: no saved model (results/{run_id}/model)")
    gold = _read_json(ev / f"report_{run_id}.json")
    if not gold:
        raise RegistrationError(f"{run_id}: not evaluated on the gold set - run Stage 9 first")

    reg = _registry(models)
    version = version or _next_version(run_id, reg)
    if any(v["version"] == version for v in reg["versions"]):
        raise RegistrationError(f"version {version} already exists")
    dest = models / version
    shutil.copytree(run / "model", dest / "model")
    for name in ("config.yaml", "thresholds.json", "metrics_validation.json", "run_info.json"):
        if (run / name).exists():
            shutil.copy(run / name, dest / name)
    (dest / "metrics_gold.json").write_text(json.dumps(gold, indent=2), encoding="utf-8")

    cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) if (run / "config.yaml").exists() else {}
    card = model_card(version, run_id, cfg, _read_json(run / "metrics_validation.json"), gold,
                      _read_json(run / "run_info.json"), _read_json(get_paths().splits / "split_report.json"),
                      _read_json(ev / "summary.json").get("comparisons", []))
    (dest / "model_card.md").write_text(card, encoding="utf-8")

    reg["versions"].append({"version": version, "run_id": run_id, "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "gold_macro_f1_mean": gold.get("macro_f1_mean")})
    if make_current:
        reg["current"] = version
    (models / "registry.json").write_text(json.dumps(reg, indent=2), encoding="utf-8")
    return version


def list_versions(models_dir: str | Path | None = None) -> dict[str, Any]:
    return _registry(Path(models_dir) if models_dir else MODELS_DIR)


def load(version: str | None = None, models_dir: str | Path | None = None) -> dict[str, Any]:
    """Paths and metadata of a registered version (the current one by default), for Stage 14 and X1."""
    models = Path(models_dir) if models_dir else MODELS_DIR
    reg = _registry(models)
    version = version or reg.get("current")
    if not version or not (models / version).exists():
        raise RegistrationError("no registered model - register one first")
    folder = models / version
    return {
        "version": version,
        "path": folder,
        "model_dir": folder / "model",
        "config": yaml.safe_load((folder / "config.yaml").read_text(encoding="utf-8")) if (folder / "config.yaml").exists() else {},
        "thresholds": _read_json(folder / "thresholds.json"),
        "metrics_gold": _read_json(folder / "metrics_gold.json"),
    }
