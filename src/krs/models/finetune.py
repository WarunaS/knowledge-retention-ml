"""M1 - DeBERTa-v3-small fine-tuned on the weak labels, with two multi-label heads (Stage 8).
Also A1 (same model without the context turn) and A2 (learning curve over training-set sizes).

Input = [CLS] context [SEP] segment [SEP], cut CONTEXT FIRST: the segment keeps its tokens (up to the
limit) and only the most recent part of the context fills the space left. Stage 5 showed that about
half of the inputs with context are longer than 512 tokens.

Training: shared encoder, two linear heads, binary cross-entropy per head (positive weights for rare
labels optional), AdamW with linear warm-up, the best epoch kept by validation macro-F1, then
per-label thresholds tuned on validation (Stage 10). Needs a GPU for the real model: run on Colab.
The tests use a tiny random DeBERTa so the code path runs on a laptop in seconds.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np

from krs.features import label_matrices, load_rows
from krs.label.taxonomy import DOMAINS, KNOWLEDGE_TYPES
from krs.models.common import (
    run_dir,
    save_predictions,
    save_run,
    tune_thresholds,
    validation_scores,
)
from krs.schema import LabelledSegment

SPLITS = ("train", "validation", "test", "transfer")


def encode(tokenizer: Any, text: str, context: str, max_length: int = 512) -> list[int]:
    """Token ids for [CLS] context [SEP] text [SEP], cutting the context first (keeping its end)."""
    # Built by hand from cls/sep ids: transformers 5 removed build_inputs_with_special_tokens.
    cls, sep = tokenizer.cls_token_id, tokenizer.sep_token_id
    text_ids = tokenizer.encode(text, add_special_tokens=False)
    ctx_ids = tokenizer.encode(context, add_special_tokens=False) if context else []
    room = max_length - 3 - len(text_ids)  # space left for context in [CLS] ctx [SEP] text [SEP]
    if not ctx_ids or room <= 0:  # no context, or the segment fills the input: segment only
        return [cls, *text_ids[: max_length - 2], sep]
    ctx_ids = ctx_ids[-room:]  # keep the most recent part of the context
    return [cls, *ctx_ids, sep, *text_ids, sep]


def _build_model(name: str, n_kt: int, n_dom: int, dropout: float):
    import torch
    from transformers import AutoModel

    class TwoHeadClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(name)
            hidden = self.encoder.config.hidden_size
            self.drop = torch.nn.Dropout(dropout)
            self.head_kt = torch.nn.Linear(hidden, n_kt)
            self.head_dom = torch.nn.Linear(hidden, n_dom)

        def forward(self, input_ids, attention_mask):
            h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            h = self.drop(h)
            return self.head_kt(h), self.head_dom(h)

    return TwoHeadClassifier()


def _batches(encoded: list[list[int]], idx: list[int], batch_size: int, pad_id: int):
    import torch

    for i in range(0, len(idx), batch_size):
        chunk = [encoded[j] for j in idx[i : i + batch_size]]
        width = max(len(x) for x in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for r, x in enumerate(chunk):
            ids[r, : len(x)] = torch.tensor(x)
            mask[r, : len(x)] = 1
        yield idx[i : i + batch_size], ids, mask


def _predict(model, encoded, batch_size, pad_id, device) -> tuple[np.ndarray, np.ndarray]:
    import torch

    model.eval()
    p_kt = np.zeros((len(encoded), model.head_kt.out_features))
    p_dom = np.zeros((len(encoded), model.head_dom.out_features))
    with torch.no_grad():
        for rows, ids, mask in _batches(encoded, list(range(len(encoded))), batch_size, pad_id):
            lk, ld = model(ids.to(device), mask.to(device))
            p_kt[rows] = torch.sigmoid(lk).float().cpu().numpy()
            p_dom[rows] = torch.sigmoid(ld).float().cpu().numpy()
    return p_kt, p_dom


def _device(pref: str = "auto") -> str:
    import torch

    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_one(run_cfg: dict[str, Any], rows: dict[str, list[LabelledSegment]],
              results_root: str | Path | None = None) -> dict[str, Any]:
    """Train and save one fine-tuned model from already-loaded split rows."""
    import torch
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    seed = run_cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _device(run_cfg.get("device", "auto"))
    name = run_cfg.get("base_model", "microsoft/deberta-v3-small")
    max_len = run_cfg.get("max_length", 512)
    with_ctx = run_cfg.get("with_context", True)
    bs = run_cfg.get("batch_size", 16)

    tok = AutoTokenizer.from_pretrained(name)
    pad_id = tok.pad_token_id or 0
    enc = {s: [encode(tok, r.segment.text, r.segment.context_text if with_ctx else "", max_len) for r in rs]
           for s, rs in rows.items()}
    Y = {s: label_matrices(rs) for s, rs in rows.items()}
    Ytr_kt, Ytr_dom = (torch.tensor(y, dtype=torch.float32) for y in Y["train"])

    model = _build_model(name, len(KNOWLEDGE_TYPES), len(DOMAINS), run_cfg.get("dropout", 0.1)).to(device)
    if run_cfg.get("pos_weight", True):  # rare labels count more, capped to stay stable
        pw = lambda y: ((len(y) - y.sum(0)) / y.sum(0).clamp(min=1)).clamp(1, run_cfg.get("pos_weight_cap", 10.0))
        loss_kt = torch.nn.BCEWithLogitsLoss(pos_weight=pw(Ytr_kt).to(device))
        loss_dom = torch.nn.BCEWithLogitsLoss(pos_weight=pw(Ytr_dom).to(device))
    else:
        loss_kt = loss_dom = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=float(run_cfg.get("lr", 3e-5)),
                            weight_decay=run_cfg.get("weight_decay", 0.01))
    epochs = run_cfg.get("epochs", 3)
    steps = epochs * ((len(enc["train"]) + bs - 1) // bs)
    sched = get_linear_schedule_with_warmup(opt, int(steps * run_cfg.get("warmup", 0.06)), steps)
    use_amp = device == "cuda" and run_cfg.get("fp16", True)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    Yva_kt, Yva_dom = Y["validation"]

    best: tuple[float, dict, int] | None = None
    history = []
    for epoch in range(epochs):
        model.train()
        order = list(range(len(enc["train"])))
        random.shuffle(order)
        for rows_idx, ids, mask in _batches(enc["train"], order, bs, pad_id):
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                lk, ld = model(ids.to(device), mask.to(device))
                loss = loss_kt(lk.float(), Ytr_kt[rows_idx].to(device)) + loss_dom(ld.float(), Ytr_dom[rows_idx].to(device))
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()
        p_kt, p_dom = _predict(model, enc["validation"], bs * 2, pad_id, device)
        score = validation_scores(p_kt, p_dom, Yva_kt, Yva_dom, tune_thresholds(p_kt, Yva_kt),
                                  tune_thresholds(p_dom, Yva_dom))["macro_f1_mean"]
        history.append({"epoch": epoch + 1, "loss_last_batch": round(float(loss), 4), "val_macro_f1_mean": score})
        if best is None or score > best[0]:
            best = (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch + 1)

    model.load_state_dict(best[1])
    probs = {s: _predict(model, enc[s], bs * 2, pad_id, device) for s in rows}
    th_kt = tune_thresholds(probs["validation"][0], Yva_kt)
    th_dom = tune_thresholds(probs["validation"][1], Yva_dom)
    out = run_dir(run_cfg["run_id"], results_root)
    for s, rs in rows.items():
        save_predictions(out, s, [r.segment.segment_id for r in rs], *probs[s], th_kt, th_dom)
    if run_cfg.get("save_model", True):
        (out / "model").mkdir(exist_ok=True)
        torch.save(best[1], out / "model" / "weights.pt")
        tok.save_pretrained(out / "model")
    metrics = {"run_id": run_cfg["run_id"], "model": name, "with_context": with_ctx, "train_size": len(rows["train"]),
               "best_epoch": best[2], "device": device, "history": history,
               **validation_scores(*probs["validation"], Yva_kt, Yva_dom, th_kt, th_dom)}
    save_run(out, run_cfg, th_kt, th_dom, metrics)
    return metrics


def train(run_cfg: dict[str, Any], splits_dir: str | Path | None = None,
          results_root: str | Path | None = None) -> dict[str, Any]:
    """M1 / A1: one run. A2: one run per size in `train_sizes` (seeded subsamples of train)."""
    rows = {s: load_rows(s, splits_dir) for s in SPLITS}
    rows = {s: r for s, r in rows.items() if r}
    if "train" not in rows or "validation" not in rows:
        raise FileNotFoundError("train and validation splits needed - run Stage 6 first")
    sizes = run_cfg.get("train_sizes")
    if not sizes:
        return train_one(run_cfg, rows, results_root)
    results = {}
    for n in sizes:
        sub = random.Random(run_cfg.get("seed", 42)).sample(rows["train"], min(n, len(rows["train"])))
        cfg = {**run_cfg, "run_id": f"{run_cfg['run_id']}_{n}", "train_sizes": None}
        results[n] = train_one(cfg, {**rows, "train": sub}, results_root)
    return {"run_id": run_cfg["run_id"], "learning_curve": {n: r["macro_f1_mean"] for n, r in results.items()}}
