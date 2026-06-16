# ArgDebate

ArgDebate is a structure-aware selector for multi-agent debate. It preserves support and attack relations from debate records, maps them into a quantitative bipolar argumentation graph, evaluates graph strength when the route warrants it, and records when the final answer comes from graph resolution, consensus, routed skip, or fallback.

This directory is a clean public-code release candidate. It contains the core implementation, baseline selectors, formal experiment harness utilities, lightweight tests, and example configuration files. It intentionally excludes private API keys, local endpoint files, cached experiment outputs, checkpoints, manuscript build products, and author-identifying metadata.

## Repository Layout

- `src/agents/`: LLM agent wrapper.
- `src/argumentation/`: argument extraction, quality scoring, QBAF construction, and DF-QuAD-style graph evaluation.
- `src/debate/`: ArgDebate manager, fallback logic, conflict/deadlock utilities, and ablation helpers.
- `src/baselines/`: MajorityVote, WeightedVote, VanillaDebate, Self-Consistency, and scalar transcript judge baselines.
- `experiments/`: reusable experiment harness components and TruthfulQA/StrategyQA runners.
- `experiments/config/model_endpoints.example.json`: endpoint template; copy to a local file before running live model calls.
- `tests/`: offline unit tests and one optional live-agent smoke test.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local testing, install `pytest` if it is not already available:

```bash
pip install pytest
```

## Configuration

Copy the example environment file and fill in local values:

```bash
cp .env.example .env
```

The simple OpenAI-compatible client reads:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=...
```

For multi-endpoint formal runs, copy:

```bash
cp experiments/config/model_endpoints.example.json experiments/config/model_endpoints.local.json
```

`model_endpoints.local.json` is ignored by git and should never be committed.

## Offline Checks

Run offline tests that do not require live model calls:

```bash
PYTHONPATH=. pytest \
  tests/test_main_baseline_set.py \
  tests/test_method_execution.py \
  tests/test_formal_run_spec.py \
  tests/test_formal_stage_harness.py \
  tests/test_formal_table_eligibility.py \
  tests/test_trace_semantics.py
```

The file `tests/test_2agent_resolve.py` is a live smoke test and requires a configured model endpoint.

## Example Live Run

After setting API credentials, run a small TruthfulQA experiment:

```bash
PYTHONPATH=. python experiments/factual_qa/run_truthfulqa_exp.py
```

For formal multi-method runs, use the configuration templates in `experiments/config/` and start with a small subset before launching larger evaluations.

## Release Notes

- This repository is released under the MIT License.
- Keep `.env`, `model_endpoints.local.json`, and experiment results out of git.
- Manuscript PDFs, reviewer packages, and historical experiment caches are not part of this code release candidate.
