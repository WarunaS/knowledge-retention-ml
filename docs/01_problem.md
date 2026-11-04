# Stage 1 - Define problem

**RQ1.** Can a small, locally trained model identify and classify organisational knowledge in
developer-AI conversations as well as a large LLM, so logs can be processed without leaving the organisation?

- RQ2: Which domains and knowledge types occur, and how often?
- RQ3: Does a model trained on public chatbot logs transfer to real agentic development logs?
- RQ4: Do extracted knowledge cards help a successor answer questions about unfamiliar work?

**Hypotheses**
- H1: Fine-tuned small transformer beats lexical and embedding baselines on macro-F1.
- H2: It comes within 5 F1 points of the LLM teacher on the human gold set.
- H3: Adding the preceding turn as context improves classification.

**Success metric:** macro-F1 per axis on the human gold set, with bootstrap 95% CI.

**Out of scope:** deployment, live capture, monitoring/retraining (future work).
