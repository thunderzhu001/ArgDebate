# Experiment Config

`model_endpoints.example.json` documents the formal model registry shape.

Create `model_endpoints.local.json` for real endpoint choices. Do not commit API keys. Store secrets in environment variables referenced by the registry, for example:

```sh
export DEEPSEEK_API_KEY=...
export DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
export QWEN_API_KEY=...
export QWEN_BASE_URL=...
export GEMINI_API_KEY=...
export GEMINI_BASE_URL=...
export GPT54_API_KEY=...
export GPT54_BASE_URL=...
export OPENAI_API_KEY=...      # legacy single-run default
export OPENAI_BASE_URL=...     # legacy single-run default
```

If one OpenAI-compatible proxy serves all formal models, use the same `api_key_env` and `base_url_env` for every model entry.

## Formal Gates

Run these before any paper-eligible experiment:

```sh
python3 experiments/probe_model_endpoints.py --attempts 3 --timeout-s 45
python3 experiments/validate_formal_specs.py --require-secrets --save
python3 experiments/run_stage_a_readiness.py --save
```

`run_stage_a_readiness.py` is dry-run by default. Add `--execute` only for the tiny Stage A smoke gate.

Model-specific launches must use registry aliases, for example:

```sh
python3 experiments/validation_first_runner.py --model-alias deepseek_v4_flash
python3 experiments/run_truthfulqa_formal_stage.py --model-alias deepseek_v4_flash --subset-size 100
```

Legacy runners preserve parent-provided `OPENAI_*` values, so a registry launcher can safely isolate each model. Checkpoint names include the model alias to prevent cross-model resume contamination.
