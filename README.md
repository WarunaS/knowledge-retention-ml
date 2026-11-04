# knowledge-retention-ml

Trains and evaluates a classifier that finds knowledge-bearing segments in developer-LLM
conversations and labels them by **technical domain** and **knowledge type**. Its output feeds
knowledge-card extraction and a retrieval agent.

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

## Data rules
- Public corpora only for training; own logs only for the transfer test, with written consent.
- Never commit data, model weights, own logs or API keys.
