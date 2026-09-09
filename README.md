# artifact-impact-lab

Hybrid change-impact analysis across software artifacts — **Requirements (R),
Config (C), Services (S), Tests (T)** — combining semantic similarity with
dependency-graph **matrix propagation**.

**Paper:** [Toward Semantically-Seeded, Graph-Propagated Impact Analysis Across 
Software Artifacts: A Vision](https://arxiv.org/abs/2606.18855) 

## Setup

```bash
pip install -r requirements.txt   # only needed for the embedding backend
# the lexical backend and the extractor are stdlib-only (no install)
```

## Run a test scenario

```bash
# 1. Build the dependency graph from the source (run once)
python3 extract_artifacts.py            # -> artifacts.json

# 2. Evaluate all scenarios (semantic vs proposed vs structural)
python3 evaluate.py --semantic lexical  # offline, instant
```

`evaluate.py` reads every file in `scenarios/`, runs all three methods for each
scenario's changed artifact, and prints precision / recall / F1 per scenario
plus a macro average (also saved to `evaluation_results.md`).

## Scenario files

Each `scenarios/scenario_*.json` defines one hypothetical change:

```json
{
  "changed": "R1",
  "actual_impact": ["payment.timeout", "payment_service.processPayment", ...]
}
```

- `changed` — the artifact you pretend changed (the analysis seed).
- `actual_impact` — the hand-labelled ground truth used to score predictions.

No source edits are needed to run a scenario — only the seed changes. Add a new
`.json` file to create a new scenario.

## Single-artifact impact (optional)

```bash
python3 semantic_impact.py          --target R1   # similarity only
python3 graph_propagation_impact.py --target R1   # semantic + graph (proposed)
python3 graph_propagation_impact.py --target R1 --lam 0.0   # structural only
```
