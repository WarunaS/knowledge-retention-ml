# Codebook v1 - knowledge in developer–AI conversations

This codebook is the single definition of both label axes. It is given word for word to the LLM
labeller and to both human raters. Code enums live in `taxonomy.py`.

## What you label

A **segment** is one exchange from a developer's conversation with an AI assistant or agent:
a user message (or a tool output, in agent runs) and the assistant's reply. The CONTEXT block, when
present, is the previous exchange; use it only to understand the SEGMENT, and label the SEGMENT.

Ask one question: **if this person left the team tomorrow, would a successor gain anything from
this exchange?** If yes, say which kind of knowledge it holds and which technical area it is about.

Both axes are multi-label: choose every label that clearly applies, usually one or two.

---

## Axis 1 - Knowledge type

### domain
Facts and expertise about a technology, system or problem area that stay true beyond this task.
- Cues: explains how something works, why a behaviour happens, what a limit or default is.
- Example: "Postgres takes a row-level lock on UPDATE, so two workers updating the same order block each other."
- Example: "React re-renders a child when its parent's state changes unless the child is memoised."
- Near-miss: "Here is the corrected function." → no explanation of *why*; not domain.

### procedural
How to do something: steps, commands, workflows, configuration, a fix someone could repeat.
- Cues: ordered steps, commands, settings, "first … then …", a reusable recipe.
- Example: "To rotate the key: create a new one in the console, update the secret in CI, redeploy, then revoke the old one."
- Example: an agent step that runs `pytest -k parser` to reproduce a failure before editing.
- Near-miss: a one-off edit to a file with no reusable method behind it → none, unless the method is shown.

### decision
A choice between options, with the reason. The successor learns *what was chosen over what, and why*.
- Cues: "instead of", "rather than", "we'll go with", trade-offs, rejected alternatives, constraints that forced a choice.
- Example: "Use a queue rather than calling the email service inline, because the service times out under load."
- Example: the agent decides to patch the caller instead of the library because the library is vendored.
- Near-miss: "Use `dict.get`." with no alternative or reason → procedural at most.

### contextual
Project-, team- or organisation-specific facts: constraints, conventions, ownership, environment, history.
- Cues: "in our setup", "the client requires", "this repo uses", names of internal services, deadlines, policies.
- Example: "Our staging database is a nightly copy of production, so never test destructive migrations there."
- Example: "The client forbids storing customer data outside the EU."
- Near-miss: a general fact about EU data law with no link to this project → domain, not contextual.

### none
Nothing a successor would reuse: greetings, thanks, pure code dumps without explanation, failed or empty
tool output, chit-chat, the assistant only restating the question.
- `none` cannot be combined with any other knowledge type.
- Example: "Thanks, that works!" / "ok" / a stack trace with no response that explains it.

---

## Axis 2 - Technical domain

Pick the area(s) the knowledge is about, not the programming language alone.

| Label | Covers | Example cues |
|---|---|---|
| mobile | Android, iOS, cross-platform mobile apps | Android Studio, SwiftUI, Flutter, app permissions |
| web_frontend | Browser UI, HTML/CSS, front-end frameworks | React components, CSS layout, DOM, bundlers |
| backend_apis | Servers, APIs, business logic services, databases accessed by services | REST/GraphQL endpoints, Django/Flask/Express, ORM, auth flow in a service |
| data_analytics | Data processing, analysis, ML/AI models, notebooks | pandas, SQL analysis, model training, ETL, charts |
| cloud_infra | Cloud resources and infrastructure as code | AWS/Azure/GCP services, Terraform, Kubernetes clusters, networking |
| devops_cicd | Build, release, deployment and CI/CD | Docker builds, GitHub Actions, pipelines, packaging, release scripts |
| security | Protecting systems and data | authentication, secrets, encryption, vulnerabilities, permissions |
| qa_testing | Testing and quality | unit/integration tests, fixtures, mocks, test strategy, reproducing bugs with tests |
| architecture_design | Structure of a system and its trade-offs | module boundaries, design patterns, scalability, service split |
| business_requirements | Requirements and business rules | user stories, acceptance criteria, stakeholder needs, process rules |
| general_programming | Everyday code work that belongs to no specific stack | language features, algorithms, refactoring, library internals, debugging plain code, repo navigation |
| other | Technical content that fits none of the above | hardware, embedded, games engines, office tooling |

- A segment labelled `none` on the knowledge-type axis still gets a domain if the topic is clear; use
  `other` only when it is not.
- Prefer a specific domain over `general_programming` when one clearly applies
  (a pytest fixture question is qa_testing, not general_programming).

---

## Rules of thumb

1. Label what the SEGMENT shows, not what the conversation might contain elsewhere.
2. The user's question alone rarely carries knowledge; the knowledge is usually in the reply or in
   the user's stated constraints (often contextual).
3. In agent runs, a step that only reads a file or lists a directory is usually `none`, unless the
   agent states what it learned or decided.
4. When unsure between two knowledge types, choose both only if each is clearly present.

## Revision history
- v1 (11 Nov 2026) - first full definitions; general_programming domain added after Stage 4 topics.
