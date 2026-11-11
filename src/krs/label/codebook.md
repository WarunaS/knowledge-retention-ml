# Codebook v1

Single human-readable definition of both label axes. Code enums live in `taxonomy.py`.
Every definition needs: description, 2 positive examples, 1 near-miss, observable cues.

## Axis 1 - Technical domain (multi-label)
mobile · web_frontend · backend_apis · data_analytics · cloud_infra · devops_cicd ·
security · qa_testing · architecture_design · business_requirements · other

Person, role and project are NOT labels: they are recorded at ingestion (schema.WorkContext).

## Axis 2 - Knowledge type (multi-label; `none` is exclusive)
- **domain** - TODO
- **procedural** - TODO
- **decision** - TODO
- **contextual** - TODO
- **none** - TODO

## Revision history
- v1 - initial draft
