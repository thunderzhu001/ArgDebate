#!/usr/bin/env python3
"""Formal Run Spec planning helpers for Stage B.

The Formal Run Spec owns the paper-facing experiment matrix.  This module turns
that machine-readable contract into runnable jobs so launchers and cloud
adapters do not also become policy owners.
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

from experiments.model_registry import safe_run_id


ROOT = Path(__file__).resolve().parents[1]

TQA_STAGE = ROOT / "experiments" / "run_truthfulqa_formal_stage.py"
SQA_STAGE = ROOT / "experiments" / "run_strategyqa_transfer_stage.py"
TQA_ABLATION_STAGE = ROOT / "experiments" / "factual_qa" / "run_targeted_ablation_stage.py"
SQA_ABLATION_STAGE = ROOT / "experiments" / "strategy_qa" / "run_targeted_ablation_stage.py"

DEFAULT_TQA_FULL = ROOT / "experiments" / "factual_qa" / "truthfulqa_full.csv"
DEFAULT_SQA_FULL = ROOT / "experiments" / "strategy_qa" / "strategyqa_full.json"

BASELINE_TO_EXTRA = {
    "SelfConsistency": "self_consistency",
    "ScalarTranscriptJudge": "scalar_transcript_judge",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_key(name: str) -> str:
    text = str(name).strip().lower()
    if text in {"truthfulqa", "truthful_qa", "truthful qa"}:
        return "truthfulqa"
    if text in {"strategyqa", "strategy_qa", "strategy qa"}:
        return "strategyqa"
    return text.replace(" ", "_")


def dataset_by_key(spec: dict[str, Any], key: str) -> dict[str, Any]:
    wanted = dataset_key(key)
    for item in spec.get("datasets", []):
        if dataset_key(str(item.get("name", ""))) == wanted:
            return item
    raise KeyError(f"dataset not found in formal run spec: {key}")


def resolve_spec_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else ROOT / path


def dataset_run_path(spec: dict[str, Any], key: str) -> Path:
    return resolve_spec_path(str(dataset_by_key(spec, key).get("path", "")))


def dataset_full_path(spec: dict[str, Any], key: str) -> Path:
    dataset = dataset_by_key(spec, key)
    full_path = dataset.get("full_path")
    if full_path:
        return resolve_spec_path(str(full_path))
    return DEFAULT_TQA_FULL if dataset_key(key) == "truthfulqa" else DEFAULT_SQA_FULL


def dataset_formal_n(spec: dict[str, Any], key: str, default: int = 100) -> int:
    return int(dataset_by_key(spec, key).get("formal_n", default))


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def count_json_items(path: Path) -> int:
    if not path.exists():
        return 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data) if isinstance(data, list) else 0


def count_dataset_items(path: Path) -> int:
    if path.suffix.lower() == ".csv":
        return count_csv_rows(path)
    if path.suffix.lower() == ".json":
        return count_json_items(path)
    return 0


def prepare_truthfulqa_subset(source_path: Path, target_path: Path, n: int | None) -> None:
    if count_csv_rows(source_path) < n:
        raise ValueError(f"TruthfulQA full data has fewer than {n} rows: {source_path}")
    with source_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    subset = random.Random(seed).sample(rows, n)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(subset)


def prepare_strategyqa_subset(source_path: Path, target_path: Path, n: int | None) -> None:
    if count_json_items(source_path) < n:
        raise ValueError(f"StrategyQA full data has fewer than {n} rows: {source_path}")
    data = json.loads(source_path.read_text(encoding="utf-8"))
    subset = random.Random(seed).sample(data, n)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8")


def _plan_config(spec: dict[str, Any]) -> dict[str, Any]:
    return spec.get("stage_b_plan", {}) if isinstance(spec.get("stage_b_plan"), dict) else {}


def _extra_baselines(spec: dict[str, Any]) -> list[str]:
    plan = _plan_config(spec)
    explicit = plan.get("extra_baselines")
    if isinstance(explicit, list):
        return [str(item).strip() for item in explicit if str(item).strip()]
    return [
        BASELINE_TO_EXTRA[name]
        for name in spec.get("baselines", [])
        if name in BASELINE_TO_EXTRA
    ]


def _main_models(spec: dict[str, Any], key: str) -> list[str]:
    plan = _plan_config(spec)
    configured = plan.get("main_models", {})
    if isinstance(configured, dict) and isinstance(configured.get(key), list):
        return [str(alias) for alias in configured[key]]
    return [str(alias) for alias in spec.get("models", [])]


def _main_skips(spec: dict[str, Any]) -> list[dict[str, Any]]:
    plan = _plan_config(spec)
    records = plan.get("main_skips", [])
    return [dict(record) for record in records if isinstance(record, dict)]


def _ablation_datasets(spec: dict[str, Any]) -> list[str]:
    plan = _plan_config(spec)
    datasets = plan.get("ablation_datasets", ["truthfulqa", "strategyqa"])
    return [dataset_key(str(item)) for item in datasets]


def _ablation_variants(spec: dict[str, Any]) -> list[str]:
    plan = _plan_config(spec)
    main_ablation = spec.get("main_ablation_run", {}) if isinstance(spec.get("main_ablation_run"), dict) else {}
    variants = [str(item) for item in main_ablation.get("ablations", [])]
    include_full = bool(plan.get("include_full_ablation_reference", True))
    return (["full"] if include_full else []) + variants


def _paper_sections(spec: dict[str, Any]) -> str:
    plan = _plan_config(spec)
    sections = plan.get("paper_sections", ["main", "appendix"])
    if isinstance(sections, str):
        return sections
    return ",".join(str(item) for item in sections)


def _timeout_mode(spec: dict[str, Any]) -> str:
    return str(_plan_config(spec).get("timeout_mode", "process"))


def _filter_record(record: dict[str, Any], filters: dict[str, Any]) -> bool:
    job_ids = set(filters.get("job_ids") or [])
    models = set(filters.get("models") or [])
    if job_ids and record.get("job_id") not in job_ids:
        return False
    if filters.get("phase", "all") != "all" and record.get("phase") != filters["phase"]:
        return False
    if filters.get("dataset", "all") != "all" and record.get("dataset") != filters["dataset"]:
        return False
    if models and record.get("model_alias") not in models:
        return False
    return True


def _dataset_env(key: str, path: Path) -> dict[str, str]:
    if key == "truthfulqa":
        return {"EXP_TRUTHFULQA_SUBSET_FILE": str(path)}
    if key == "strategyqa":
        return {"EXP_STRATEGYQA_SUBSET_FILE": str(path)}
    return {}


def _main_job(
    *,
    spec: dict[str, Any],
    key: str,
    alias: str,
    n: int,
    timeout_s: int,
    python_executable: str,
    extra_baselines: str,
) -> dict[str, Any]:
    controls = spec.get("run_controls", {})
    num_agents = int(controls.get("num_agents", 3))
    max_rounds = int(controls.get("max_rounds", 3))
    request_timeout = int(controls.get("request_timeout_s", 90))
    max_retries = int(controls.get("max_retries", 1))
    method_timeout = int(controls.get("method_timeout_s", 900))
    dataset_path = dataset_run_path(spec, key)
    stage = TQA_STAGE if key == "truthfulqa" else SQA_STAGE
    cmd = [
        python_executable,
        str(stage),
        "--subset-size",
        str(n),
        "--num-agents",
        str(num_agents),
        "--max-rounds",
        str(max_rounds),
        "--request-timeout",
        str(request_timeout),
        "--max-retries",
        str(max_retries),
        "--method-timeout-s",
        str(method_timeout),
        "--method-timeout-mode",
        _timeout_mode(spec),
        "--runner-timeout-s",
        str(timeout_s),
    ]
    if key == "truthfulqa":
        cmd.append("--use-llm-judge")
    cmd.extend(
        [
            "--extra-baselines",
            extra_baselines,
            "--model-alias",
            alias,
            "--tag",
            f"stage_b_{key}_n{n}",
            "--paper-eligible",
            "main_table_eligible",
        ]
    )
    if key == "truthfulqa":
        cmd.extend(
            [
                "--evidence-tier",
                "stage_b_pilot",
                "--paper-sections",
                _paper_sections(spec),
                "--matched-to-main-config",
            ]
        )
    return {
        "job_id": f"main_{key}_{safe_run_id(alias)}_n{n}",
        "phase": "main",
        "dataset": key,
        "dataset_path": str(dataset_path),
        "model_alias": alias,
        "n": n,
        "timeout_s": timeout_s,
        "cmd": cmd,
        "env": _dataset_env(key, dataset_path),
    }


def _ablation_job(
    *,
    spec: dict[str, Any],
    key: str,
    alias: str,
    n: int,
    timeout_s: int,
    python_executable: str,
    variants: str,
) -> dict[str, Any]:
    controls = spec.get("run_controls", {})
    num_agents = int(controls.get("num_agents", 3))
    max_rounds = int(controls.get("max_rounds", 3))
    request_timeout = int(controls.get("request_timeout_s", 90))
    max_retries = int(controls.get("max_retries", 1))
    method_timeout = int(controls.get("method_timeout_s", 900))
    dataset_path = dataset_run_path(spec, key)
    stage = TQA_ABLATION_STAGE if key == "truthfulqa" else SQA_ABLATION_STAGE
    return {
        "job_id": f"ablation_{key}_{safe_run_id(alias)}_n{n}",
        "phase": "ablation",
        "dataset": key,
        "dataset_path": str(dataset_path),
        "model_alias": alias,
        "n": n,
        "timeout_s": timeout_s,
        "cmd": [
            python_executable,
            str(stage),
            "--subset-size",
            str(n),
            "--variants",
            variants,
            "--num-agents",
            str(num_agents),
            "--max-rounds",
            str(max_rounds),
            "--item-timeout-s",
            str(method_timeout),
            "--model-alias",
            alias,
            "--request-timeout",
            str(request_timeout),
            "--max-retries",
            str(max_retries),
            "--tag",
            f"stage_b_{key}_ablation_n{n}",
            "--evidence-tier",
            "stage_b_mechanism",
            "--paper-eligible",
            "main_table_eligible",
        ],
        "env": _dataset_env(key, dataset_path),
    }


def build_stage_b_plan(
    spec: dict[str, Any],
    *,
    dataset: str = "all",
    phase: str = "all",
    models: list[str] | None = None,
    job_ids: list[str] | None = None,
    smoke_size: int = 0,
    job_timeout_s: int = 0,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    controls = spec.get("run_controls", {})
    method_timeout = int(controls.get("method_timeout_s", 900))
    extra_baselines = _extra_baselines(spec)
    extra_baselines_arg = ",".join(extra_baselines)
    variants = _ablation_variants(spec)
    variants_arg = ",".join(variants)
    filters = {
        "dataset": dataset,
        "phase": phase,
        "models": models or [],
        "job_ids": job_ids or [],
    }

    jobs: list[dict[str, Any]] = []
    for key in ("truthfulqa", "strategyqa"):
        n = int(smoke_size or dataset_formal_n(spec, key, 100))
        timeout_s = int(job_timeout_s or max(7200, n * method_timeout * 2))
        for alias in _main_models(spec, key):
            jobs.append(
                _main_job(
                    spec=spec,
                    key=key,
                    alias=alias,
                    n=n,
                    timeout_s=timeout_s,
                    python_executable=python_executable,
                    extra_baselines=extra_baselines_arg,
                )
            )

    main_ablation = spec.get("main_ablation_run", {}) if isinstance(spec.get("main_ablation_run"), dict) else {}
    ablation_model = str(main_ablation.get("model_alias", ""))
    ablation_n = int(smoke_size or main_ablation.get("n_per_dataset", 50))
    ablation_timeout = int(job_timeout_s or max(7200, ablation_n * max(len(variants), 1) * method_timeout + 1800))
    if ablation_model:
        for key in _ablation_datasets(spec):
            jobs.append(
                _ablation_job(
                    spec=spec,
                    key=key,
                    alias=ablation_model,
                    n=ablation_n,
                    timeout_s=ablation_timeout,
                    python_executable=python_executable,
                    variants=variants_arg,
                )
            )

    skip_records = []
    for skip in _main_skips(spec):
        key = dataset_key(str(skip.get("dataset", "")))
        alias = str(skip.get("model_alias", "")).strip()
        if not key or not alias:
            continue
        n = int(smoke_size or dataset_formal_n(spec, key, 100))
        skip_records.append(
            {
                "job_id": f"main_{key}_{safe_run_id(alias)}_n{n}",
                "phase": "main",
                "dataset": key,
                "dataset_path": str(dataset_run_path(spec, key)),
                "model_alias": alias,
                "n": n,
                "status": "skipped",
                "reason": str(skip.get("reason", "")),
            }
        )

    filtered_jobs = [job for job in jobs if _filter_record(job, filters)]
    filtered_skips = [skip for skip in skip_records if _filter_record(skip, filters)]
    plan = _plan_config(spec)
    cost_cap = spec.get("cost_cap", {}) if isinstance(spec.get("cost_cap"), dict) else {}
    return {
        "jobs": filtered_jobs,
        "skip_records": filtered_skips,
        "matrix_policy": {
            "truthfulqa_models": _main_models(spec, "truthfulqa"),
            "strategyqa_models": _main_models(spec, "strategyqa"),
            "main_skips": _main_skips(spec),
            "main_ablation_model": ablation_model,
            "main_ablation_variants": variants,
            "extra_baselines": extra_baselines,
            "cost_control": str(plan.get("cost_control", cost_cap.get("mode", "soft_record_only"))),
            "execution_policy": str(plan.get("execution_policy", "single_job_default_allow_batch_required")),
        },
    }


def required_dataset_capacity(jobs: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    required: dict[tuple[str, str], int] = {}
    for job in jobs:
        key = dataset_key(str(job.get("dataset", "")))
        path = str(job.get("dataset_path", ""))
        if not key or not path:
            continue
        item_key = (key, path)
        required[item_key] = max(required.get(item_key, 0), int(job.get("n", 0) or 0))
    return required


def ensure_stage_b_dataset_capacity(
    spec: dict[str, Any],
    jobs: list[dict[str, Any]],
    *,
    prepare: bool,
    seed: int | None,
) -> dict[str, Any]:
    required = required_dataset_capacity(jobs)
    actual: dict[str, int] = {}
    errors: list[str] = []
    prepared_paths: list[str] = []

    for (key, raw_path), needed in sorted(required.items()):
        path = Path(raw_path)
        if prepare and count_dataset_items(path) < needed:
            source = dataset_full_path(spec, key)
            if key == "truthfulqa":
                prepare_truthfulqa_subset(source, path, needed, seed)
            elif key == "strategyqa":
                prepare_strategyqa_subset(source, path, needed, seed)
            prepared_paths.append(str(path))
        count = count_dataset_items(path)
        actual[f"{key}:{path}"] = count
        if needed and count < needed:
            errors.append(f"{key} requires n={needed}, but {path} contains {count} items")

    return {
        "ok": not errors,
        "required": {f"{key}:{path}": needed for (key, path), needed in sorted(required.items())},
        "actual": actual,
        "errors": errors,
        "prepared": bool(prepare),
        "prepared_paths": prepared_paths,
        "seed": seed,
    }
