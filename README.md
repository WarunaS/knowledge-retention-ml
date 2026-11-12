# knowledge-retention-ml

Trains and evaluates a classifier that finds knowledge-bearing segments in developer-LLM
conversations and labels them by **technical domain** and **knowledge type**. Its output feeds
knowledge-card extraction and a retrieval agent.

Install, test, run and see results: [`docs/how_to_test.html`](docs/how_to_test.html). Smoke test: `python scripts/smoke_test.py`.

Structured along the ML lifecycle - see [`docs/lifecycle.md`](docs/lifecycle.md).
Problem statement: [`docs/01_problem.md`](docs/01_problem.md).

## Layout
```
src/krs/       package: schema, collect, clean, explore, label, split, features, models,
               evaluate, tracking, registry, inference, app/
configs/       stage configs + one YAML per experiment run (runs/B0 ... X1)
notebooks/     thin Colab runners, one per lifecycle stage
data/          local data (gitignored); on Colab use Google Drive via KRS_DATA_ROOT
models/        registered model versions (gitignored)
results/       per-run metrics and plots (gitignored)
tests/
```

## Local setup
```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,clean]"
pytest
```
Add extras as stages need them: `explore`, `label`, `ml`, `app`.

## Colab
Each notebook's first cell mounts Drive, clones this repo, installs it, and points
`KRS_DATA_ROOT` at `MyDrive/krs/data`. Set the repo URL in that cell once the GitHub repo exists.

## Using the app (after a model is registered)
```bash
uv pip install -e ".[app]"
uvicorn krs.app.agent:app                         # REST API, docs at http://localhost:8000/docs
claude mcp add krs -- python -m krs.app.mcp_server # ask from inside Claude Code
uv pip install -e ".[dashboard]" && streamlit run src/krs/app/dashboard.py   # optional demo screen
```
Tools/endpoints: ask a question (answers cite `[card_id]`s), search cards, build a handover pack per
project or person. Own-log cards are never sent to a cloud model.

## Data rules
- Public corpora only for training; own logs only for the transfer test, with written consent.
- Never commit data, model weights, own logs or API keys.
