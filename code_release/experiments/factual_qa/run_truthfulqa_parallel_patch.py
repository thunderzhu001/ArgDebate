#!/usr/bin/env python3
"""Patch TruthfulQA checkpoints by running selected questions in parallel."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments" / "factual_qa" / "run_truthfulqa_exp.py"
DEFAULT_SUBSET = ROOT / "experiments" / "factual_qa" / "truthfulqa_subset.csv"
RESULTS_DIR = ROOT / "experiments" / "results"

sys.path.insert(0, str(ROOT))

from experiments.formal_stage_harness import (  # noqa: E402
    artifact_check,
    count_internal_errors,
    count_internal_timeouts,
    load_env_file,
    load_json,
    run_python_subprocess_with_timeout,
    write_json,
)
from experiments.formal_table_eligibility import (  # noqa: E402
    method_formal_table_eligibility,
    summarize_formal_table_eligibility,
)
from experiments.model_registry import (  # noqa: E402
    DEFAULT_REGISTRY,
    endpoint_summary_for_alias,
    env_overlay_for_alias,
    safe_run_id,
)
from experiments.main_baseline_set import (  # noqa: E402
    configured_methods as _configured_methods,
    estimate_tokens as _estimate_tokens,
    extract_method_prediction as _shared_extract_method_prediction,
    status_counts as _main_status_counts,
)
from experiments.factual_qa.run_truthfulqa_exp import (  # noqa: E402
    _build_checkpoint_payload,
    _summarize_argdebate_traces,
    _summarize_runtime_profiles,
)


METHOD_RESULT_KEYS = {
    "majority_vote": "majority_vote",
    "vanilla_debate": "vanilla_debate",
    "weighted_vote": "weighted_vote",
    "self_consistency": "self_consistency",
    "scalar_transcript_judge": "scalar_transcript_judge",
    "arg_debate": "arg_debate",
}


def _extract_method_prediction(method_result: Any, method_name: str) -> str:
    return _shared_extract_method_prediction(
        method_result,
        method_name,
        prefer_formal_normalized=False,
    )


def _deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def status_for_method(row: dict[str, Any], method: str) -> str:
    result = row.get(METHOD_RESULT_KEYS.get(method, method))
    if not isinstance(result, dict):
        return "missing"
    status = str(result.get("status", "") or "")
    error = str(result.get("error", "") or "")
    if result.get("timeout") is True or status == "timeout" or error.startswith("method_timeout_after_"):
        return "timeout"
    if status == "error":
        return "error"
    if error and status not in {"completed", "resolved", "fallback", "no_conflict", "routed_skip", "fallback_disabled"}:
        return "error"
    return status or ("fallback" if method in {"vanilla_debate", "arg_debate"} else "completed")


def _selected_indexes(seed_rows: list[dict[str, Any]], target_n: int, explicit: list[int]) -> list[int]:
    selected = set(explicit)
    methods = list(METHOD_RESULT_KEYS)
    for idx, row in enumerate(seed_rows[:target_n], 1):
        for method in methods:
            if status_for_method(row, method) in {"missing", "timeout", "error"}:
                selected.add(idx)
                break
    for idx in range(len(seed_rows) + 1, target_n + 1):
        selected.add(idx)
    return sorted(idx for idx in selected if 1 <= idx <= target_n)


def _write_single_item_subset(subset_df: pd.DataFrame, item_index: int, path: Path) -> None:
    if item_index < 1 or item_index > len(subset_df):
        raise IndexError(f"item index out of range for TruthfulQA subset: {item_index}")
    subset_df.iloc[[item_index - 1]].to_csv(path, index=False)


def _latest_checkpoint(worker_dir: Path) -> Path | None:
    done = sorted(worker_dir.glob("checkpoint.done_*.json"))
    if done:
        return done[-1]
    checkpoint = worker_dir / "checkpoint.json"
    return checkpoint if checkpoint.exists() else None


def _run_worker(
    *,
    item_index: int,
    subset_df: pd.DataFrame,
    work_dir: Path,
    env_base: dict[str, str],
    num_agents: int,
    max_rounds: int,
    request_timeout: int,
    max_retries: int,
    method_timeout_s: int,
    method_timeout_mode: str,
    item_timeout_s: int,
    extra_baselines: str,
    use_llm_judge: bool,
) -> dict[str, Any]:
    worker_dir = work_dir / "workers" / f"row_{item_index:03d}"
    worker_dir.mkdir(parents=True, exist_ok=True)
    subset_path = worker_dir / "subset.csv"
    _write_single_item_subset(subset_df, item_index, subset_path)

    env = dict(env_base)
    env.update(
        {
            "EXP_SUBSET_SIZE": "1",
            "EXP_TRUTHFULQA_SUBSET_FILE": str(subset_path),
            "EXP_NUM_AGENTS": str(num_agents),
            "EXP_MAX_ROUNDS": str(max_rounds),
            "EXP_USE_LLM_JUDGE": "1" if use_llm_judge else "0",
            "EXP_ENABLE_RESUME": "1",
            "EXP_RESUME_FILE": str(worker_dir / "checkpoint.json"),
            "EXP_EXTRA_BASELINES": extra_baselines,
            "EXP_METHOD_TIMEOUT_S": str(method_timeout_s),
            "EXP_METHOD_TIMEOUT_MODE": method_timeout_mode,
            "EXP_OUTPUT_DIR": str(worker_dir),
            "EXP_OUTPUT_STAMP": f"row_{item_index:03d}",
            "REQUEST_TIMEOUT": str(request_timeout),
            "MAX_RETRIES": str(max_retries),
            "PYTHONUNBUFFERED": "1",
        }
    )
    started = time.time()
    run = run_python_subprocess_with_timeout(
        script=RUNNER,
        cwd=ROOT,
        env=env,
        log_path=worker_dir / "runner.log",
        timeout_s=item_timeout_s,
    )
    results_path = worker_dir / f"results_row_{item_index:03d}.json"
    summary_path = worker_dir / f"summary_row_{item_index:03d}.json"
    checkpoint_path = _latest_checkpoint(worker_dir)
    rows = load_json(results_path) if results_path.exists() else []
    row = rows[0] if isinstance(rows, list) and rows else None
    checkpoint = load_json(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else {}
    row_timeout_count = count_internal_timeouts(row)
    row_error_count = count_internal_errors(row)
    ok = (
        int(run["returncode"]) == 0
        and not bool(run["timed_out"])
        and isinstance(row, dict)
        and row_timeout_count == 0
        and row_error_count == 0
    )
    return {
        "item_index": item_index,
        "ok": ok,
        "returncode": run["returncode"],
        "timed_out": run["timed_out"],
        "elapsed_s": round(time.time() - started, 3),
        "worker_dir": str(worker_dir),
        "results_file": str(results_path) if results_path.exists() else None,
        "summary_file": str(summary_path) if summary_path.exists() else None,
        "checkpoint_file": str(checkpoint_path) if checkpoint_path else None,
        "row": row,
        "worker_checkpoint": checkpoint,
        "row_timeout_count": row_timeout_count,
        "row_error_count": row_error_count,
    }


def _metric_from_worker(worker: dict[str, Any], field: str, method: str, default: Any) -> Any:
    repaired_methods = worker.get("repaired_methods")
    if isinstance(repaired_methods, list) and method not in repaired_methods:
        return default
    checkpoint = worker.get("worker_checkpoint")
    if not isinstance(checkpoint, dict):
        return default
    values = checkpoint.get(field, {})
    if not isinstance(values, dict):
        return default
    method_values = values.get(method)
    if isinstance(method_values, list) and method_values:
        return method_values[0]
    return default


def _method_needs_repair(row: dict[str, Any], method: str) -> bool:
    if status_for_method(row, method) in {"missing", "timeout", "error"}:
        return True
    eligibility = row.get("formal_table_eligibility", {}) if isinstance(row, dict) else {}
    method_eligibility = eligibility.get(method) if isinstance(eligibility, dict) else None
    if isinstance(method_eligibility, dict) and method_eligibility.get("formal_table_eligible") is False:
        return True
    return False


def _methods_needing_repair(row: dict[str, Any], methods: list[str]) -> list[str]:
    return [method for method in methods if _method_needs_repair(row, method)]


def _merge_repaired_methods(
    *,
    seed_row: dict[str, Any],
    worker_row: dict[str, Any],
    repaired_methods: list[str],
) -> dict[str, Any]:
    merged = _deepcopy_json(seed_row)
    for method in repaired_methods:
        result_key = METHOD_RESULT_KEYS.get(method, method)
        if result_key in worker_row:
            merged[result_key] = _deepcopy_json(worker_row[result_key])
        for nested_key in ("correctness", "correctness_judge", "formal_table_eligibility"):
            source_nested = worker_row.get(nested_key)
            if not isinstance(source_nested, dict) or method not in source_nested:
                continue
            target_nested = merged.setdefault(nested_key, {})
            if not isinstance(target_nested, dict):
                target_nested = {}
                merged[nested_key] = target_nested
            target_nested[method] = _deepcopy_json(source_nested[method])
    return merged


def _build_merged_checkpoint(
    *,
    seed: dict[str, Any],
    rows: list[dict[str, Any]],
    worker_by_index: dict[int, dict[str, Any]],
    target_n: int,
) -> dict[str, Any]:
    cfg = dict(seed.get("config", {}) if isinstance(seed.get("config"), dict) else {})
    cfg["subset_size"] = target_n
    methods = list(cfg.get("methods") or _configured_methods(cfg.get("extra_baselines", [])))
    max_rounds = int(cfg.get("max_rounds", 3) or 3)
    seed_timing = seed.get("timing", {}) if isinstance(seed.get("timing"), dict) else {}
    seed_status = seed.get("status_log", {}) if isinstance(seed.get("status_log"), dict) else {}
    seed_rounds = seed.get("rounds_log", {}) if isinstance(seed.get("rounds_log"), dict) else {}
    seed_tokens = seed.get("token_log", {}) if isinstance(seed.get("token_log"), dict) else {}
    seed_route = seed.get("route_log", []) if isinstance(seed.get("route_log"), list) else []

    timing: dict[str, list[Any]] = {method: [] for method in methods}
    status_log: dict[str, list[Any]] = {method: [] for method in methods}
    rounds_log: dict[str, list[Any]] = {method: [] for method in methods}
    token_log: dict[str, list[Any]] = {method: [] for method in methods}
    correctness: dict[str, int] = {method: 0 for method in methods}
    route_log: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, 1):
        worker = worker_by_index.get(row_index)
        for method in methods:
            result = row.get(METHOD_RESULT_KEYS.get(method, method), {}) if isinstance(row, dict) else {}
            pred = _extract_method_prediction(result if isinstance(result, dict) else {}, method)
            timing_default = (
                seed_timing.get(method, [])[row_index - 1]
                if isinstance(seed_timing.get(method), list) and len(seed_timing.get(method, [])) >= row_index
                else 0.0
            )
            status_default = (
                seed_status.get(method, [])[row_index - 1]
                if isinstance(seed_status.get(method), list) and len(seed_status.get(method, [])) >= row_index
                else status_for_method(row, method)
            )
            rounds_default = (
                seed_rounds.get(method, [])[row_index - 1]
                if isinstance(seed_rounds.get(method), list) and len(seed_rounds.get(method, [])) >= row_index
                else (1 if method in {"majority_vote", "weighted_vote", "self_consistency"} else max_rounds)
            )
            tokens_default = (
                seed_tokens.get(method, [])[row_index - 1]
                if isinstance(seed_tokens.get(method), list) and len(seed_tokens.get(method, [])) >= row_index
                else _estimate_tokens(pred)
            )
            timing[method].append(_metric_from_worker(worker or {}, "timing", method, timing_default))
            status_log[method].append(_metric_from_worker(worker or {}, "status_log", method, status_default))
            rounds_log[method].append(_metric_from_worker(worker or {}, "rounds_log", method, rounds_default))
            token_log[method].append(_metric_from_worker(worker or {}, "token_log", method, tokens_default))
            if bool(row.get("correctness", {}).get(method, False)) if isinstance(row, dict) else False:
                correctness[method] += 1

        route_default = (
            seed_route[row_index - 1]
            if len(seed_route) >= row_index and isinstance(seed_route[row_index - 1], dict)
            else _route_record_from_row(row)
        )
        worker_checkpoint = worker.get("worker_checkpoint") if worker else None
        worker_route = (
            worker_checkpoint.get("route_log", [])[0]
            if isinstance(worker_checkpoint, dict)
            and isinstance(worker_checkpoint.get("route_log"), list)
            and worker_checkpoint.get("route_log")
            else route_default
        )
        route_log.append(worker_route)

    return _build_checkpoint_payload(
        config=cfg,
        results=rows,
        timing=timing,
        correctness=correctness,
        status_log=status_log,
        rounds_log=rounds_log,
        token_log=token_log,
        route_log=route_log,
    )


def _route_record_from_row(row: dict[str, Any]) -> dict[str, Any]:
    ad = row.get("arg_debate", {}) if isinstance(row.get("arg_debate"), dict) else {}
    meta = ad.get("meta", {}) if isinstance(ad.get("meta"), dict) else {}
    return {
        "debate_routed": bool(meta.get("debate_routed", True)),
        "route_disagreement_score": float(meta.get("route_disagreement_score", 0.0) or 0.0),
        "route_info_density_score": float(meta.get("route_info_density_score", 0.0) or 0.0),
        "route_disagreement_hit": bool(meta.get("route_disagreement_hit", False)),
        "route_density_hit": bool(meta.get("route_density_hit", False)),
        "route_reason": list(meta.get("route_reason", [])) if isinstance(meta.get("route_reason"), list) else [],
        "status": ad.get("status", "fallback"),
    }


def _status_counts(status_log: dict[str, list[Any]]) -> dict[str, dict[str, int]]:
    return _main_status_counts(status_log)


def _build_summary(checkpoint: dict[str, Any], results_path: Path) -> dict[str, Any]:
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint.get("config"), dict) else {}
    rows = checkpoint.get("results", []) if isinstance(checkpoint.get("results"), list) else []
    methods = list(cfg.get("methods") or _configured_methods(cfg.get("extra_baselines", [])))
    refresh_formal_table_eligibility(rows, methods)
    status_log = checkpoint.get("status_log", {}) if isinstance(checkpoint.get("status_log"), dict) else {}
    rounds_log = checkpoint.get("rounds_log", {}) if isinstance(checkpoint.get("rounds_log"), dict) else {}
    token_log = checkpoint.get("token_log", {}) if isinstance(checkpoint.get("token_log"), dict) else {}
    timing = checkpoint.get("timing", {}) if isinstance(checkpoint.get("timing"), dict) else {}
    route_log = checkpoint.get("route_log", []) if isinstance(checkpoint.get("route_log"), list) else []
    correctness = {
        method: sum(1 for row in rows if bool(row.get("correctness", {}).get(method, False)))
        for method in methods
    }
    return {
        "dataset": "TruthfulQA",
        "subset_size": len(rows),
        "accuracy": {method: correctness[method] / max(len(rows), 1) for method in methods},
        "avg_time_s": {
            method: sum(timing.get(method, []) or [0.0]) / max(len(timing.get(method, []) or []), 1)
            for method in methods
        },
        "deadlock_rate": {
            method: sum(1 for status in status_log.get(method, []) if status == "fallback")
            / max(len(status_log.get(method, []) or []), 1)
            for method in methods
        },
        "status_counts": _status_counts(status_log),
        "method_timeout_count": sum(1 for statuses in status_log.values() for status in statuses if status == "timeout"),
        "avg_rounds": {
            method: sum(rounds_log.get(method, []) or [0]) / max(len(rounds_log.get(method, []) or []), 1)
            for method in methods
        },
        "token_efficiency_estimated": {
            method: sum(token_log.get(method, []) or [0]) / max(correctness[method], 1)
            for method in methods
        },
        "routing": {
            "enabled": bool(cfg.get("enable_debate_routing", True)),
            "disagreement_threshold": cfg.get("route_disagreement_threshold"),
            "info_density_gate_enabled": bool(cfg.get("enable_info_density_gate", True)),
            "info_density_threshold": cfg.get("route_info_density_threshold"),
            "logic": cfg.get("route_logic", "or"),
            "routed_skip_rate": sum(1 for item in route_log if not item.get("debate_routed", True)) / max(len(route_log), 1),
            "avg_route_disagreement": sum(float(item.get("route_disagreement_score", 0.0) or 0.0) for item in route_log)
            / max(len(route_log), 1),
            "avg_route_info_density": sum(float(item.get("route_info_density_score", 0.0) or 0.0) for item in route_log)
            / max(len(route_log), 1),
            "disagreement_hit_rate": sum(1 for item in route_log if bool(item.get("route_disagreement_hit", False)))
            / max(len(route_log), 1),
            "info_density_hit_rate": sum(1 for item in route_log if bool(item.get("route_density_hit", False)))
            / max(len(route_log), 1),
        },
        "formal_table_eligibility": summarize_formal_table_eligibility(rows, methods),
        "arg_debate_trace_summary": _summarize_argdebate_traces(rows),
        "arg_debate_runtime_profile_summary": _summarize_runtime_profiles(rows),
        "config": cfg,
        "source_file": str(results_path),
    }


def refresh_formal_table_eligibility(rows: list[dict[str, Any]], methods: list[str]) -> None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        eligibility = row.setdefault("formal_table_eligibility", {})
        if not isinstance(eligibility, dict):
            eligibility = {}
            row["formal_table_eligibility"] = eligibility
        judge_by_method = row.get("correctness_judge", {}) if isinstance(row.get("correctness_judge"), dict) else {}
        for method in methods:
            result = row.get(METHOD_RESULT_KEYS.get(method, method), {})
            judge_record = judge_by_method.get(method) if isinstance(judge_by_method, dict) else None
            eligibility[method] = method_formal_table_eligibility(
                dataset="TruthfulQA",
                method=method,
                method_result=result,
                judge_record=judge_record,
            )


def _render_md(report: dict[str, Any]) -> str:
    lines = [
        "# TruthfulQA Parallel Patch Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- work_dir: `{report['work_dir']}`",
        f"- artifact_ok: `{report['artifact_check'].get('ok')}`",
        f"- artifact_reason: `{report['artifact_check'].get('reason')}`",
        f"- selected_item_indexes: `{report['selected_item_indexes']}`",
        f"- max_workers: `{report['parallel_config'].get('max_workers')}`",
        "",
        "## Workers",
        "",
    ]
    for worker in report.get("workers", []):
        lines.append(
            f"- item_index=`{worker['item_index']}` ok=`{worker['ok']}` "
            f"returncode=`{worker['returncode']}` timed_out=`{worker['timed_out']}` "
            f"elapsed_s=`{worker['elapsed_s']}`"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected TruthfulQA rows in bounded parallel and merge a checkpoint")
    parser.add_argument("--seed-checkpoint-file", required=True)
    parser.add_argument("--target-subset-size", type=int, required=True)
    parser.add_argument("--item-index", type=int, action="append", default=[], help="1-based item index to rerun. Defaults to failed seed rows plus missing rows.")
    parser.add_argument("--subset-file", default=str(DEFAULT_SUBSET))
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--tag", default="truthfulqa_parallel_patch")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--item-timeout-s", type=int, default=7200)
    parser.add_argument("--num-agents", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--request-timeout", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--method-timeout-s", type=int, default=900)
    parser.add_argument("--method-timeout-mode", choices=["signal", "process"], default="process")
    parser.add_argument("--extra-baselines", default="self_consistency,scalar_transcript_judge")
    parser.add_argument("--use-llm-judge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plan-only", action="store_true", help="Print selected repair/continuation rows without making API calls.")
    args = parser.parse_args()

    if args.max_workers < 1:
        raise ValueError("--max-workers must be >= 1")
    if args.max_workers > 4:
        raise ValueError("--max-workers > 4 is intentionally blocked for API stability")

    load_env_file()
    os.environ.update(
        env_overlay_for_alias(
            args.model_alias,
            args.registry,
            request_timeout_s=args.request_timeout,
            max_retries=args.max_retries,
        )
    )
    env_base = dict(os.environ)
    seed_path = Path(args.seed_checkpoint_file).expanduser().resolve()
    seed = load_json(seed_path)
    seed_rows = seed.get("results", []) if isinstance(seed, dict) and isinstance(seed.get("results"), list) else []
    seed_cfg = seed.get("config", {}) if isinstance(seed.get("config"), dict) else {}
    methods = list(seed_cfg.get("methods") or _configured_methods(seed_cfg.get("extra_baselines", [])))
    subset_df = pd.read_csv(Path(args.subset_file).expanduser().resolve())

    selected = _selected_indexes(seed_rows, args.target_subset_size, args.item_index)
    if args.plan_only:
        print(
            json.dumps(
                {
                    "seed_checkpoint_file": str(seed_path),
                    "seed_rows": len(seed_rows),
                    "target_subset_size": args.target_subset_size,
                    "selected_item_indexes": selected,
                    "max_workers": args.max_workers,
                    "model_alias": args.model_alias,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    work_dir = RESULTS_DIR / f"{args.tag}_{safe_run_id(args.model_alias)}_n{args.target_subset_size}_{stamp}"
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed_path, work_dir / "source_checkpoint.json")

    workers: list[dict[str, Any]] = []
    started = time.time()
    if selected:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            future_map = {
                pool.submit(
                    _run_worker,
                    item_index=item_index,
                    subset_df=subset_df,
                    work_dir=work_dir,
                    env_base=env_base,
                    num_agents=args.num_agents,
                    max_rounds=args.max_rounds,
                    request_timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    method_timeout_s=args.method_timeout_s,
                    method_timeout_mode=args.method_timeout_mode,
                    item_timeout_s=args.item_timeout_s,
                    extra_baselines=args.extra_baselines,
                    use_llm_judge=args.use_llm_judge,
                ): item_index
                for item_index in selected
            }
            for future in concurrent.futures.as_completed(future_map):
                workers.append(future.result())

    workers = sorted(workers, key=lambda item: int(item["item_index"]))
    failed_workers: list[dict[str, Any]] = []
    worker_by_index = {int(worker["item_index"]): worker for worker in workers if isinstance(worker.get("row"), dict)}
    merged_rows = _deepcopy_json(seed_rows[: args.target_subset_size])
    for item_index in selected:
        worker = worker_by_index.get(item_index)
        if worker is None:
            continue
        while len(merged_rows) < item_index - 1:
            merged_rows.append({})
        if len(merged_rows) == item_index - 1:
            worker["repaired_methods"] = list(methods)
            merged_rows.append(worker["row"])
        else:
            repaired_methods = _methods_needing_repair(merged_rows[item_index - 1], methods)
            worker["repaired_methods"] = repaired_methods
            merged_rows[item_index - 1] = _merge_repaired_methods(
                seed_row=merged_rows[item_index - 1],
                worker_row=worker["row"],
                repaired_methods=repaired_methods,
            )
    for worker in workers:
        if int(worker.get("returncode", 1)) != 0 or bool(worker.get("timed_out")):
            failed_workers.append(worker)
            continue
        row = worker.get("row")
        if not isinstance(row, dict):
            failed_workers.append(worker)
            continue
        repaired_methods = worker.get("repaired_methods")
        if isinstance(repaired_methods, list) and repaired_methods:
            if any(status_for_method(row, method) in {"missing", "timeout", "error"} for method in repaired_methods):
                failed_workers.append(worker)
        elif not worker.get("ok"):
            failed_workers.append(worker)

    checkpoint = _build_merged_checkpoint(
        seed=seed,
        rows=merged_rows,
        worker_by_index=worker_by_index,
        target_n=args.target_subset_size,
    )
    cfg = checkpoint.get("config", {}) if isinstance(checkpoint.get("config"), dict) else {}
    methods = list(cfg.get("methods") or _configured_methods(cfg.get("extra_baselines", [])))
    checkpoint_rows = checkpoint.get("results", []) if isinstance(checkpoint.get("results"), list) else []
    refresh_formal_table_eligibility(checkpoint_rows, methods)
    checkpoint_path = work_dir / "checkpoint.json"
    results_path = work_dir / "results.json"
    summary_path = work_dir / "summary.json"
    write_json(checkpoint_path, checkpoint)
    write_json(results_path, checkpoint_rows)
    summary = _build_summary(checkpoint, results_path)
    write_json(summary_path, summary)
    check = artifact_check(summary, merged_rows, include_summary_status_errors=True)
    if failed_workers:
        check = {**check, "ok": False, "reason": "worker_failed"}

    report = {
        "generated_at": datetime.now().isoformat(),
        "work_dir": str(work_dir),
        "source_checkpoint_file": str(seed_path),
        "merged_checkpoint_file": str(checkpoint_path),
        "merged_results_file": str(results_path),
        "merged_summary_file": str(summary_path),
        "selected_item_indexes": selected,
        "elapsed_s": round(time.time() - started, 3),
        "model_endpoint": endpoint_summary_for_alias(args.model_alias, args.registry),
        "parallel_config": {
            "max_workers": args.max_workers,
            "item_timeout_s": args.item_timeout_s,
            "method_timeout_s": args.method_timeout_s,
            "method_timeout_mode": args.method_timeout_mode,
            "use_llm_judge": args.use_llm_judge,
        },
        "workers": [
            {key: value for key, value in worker.items() if key not in {"row", "worker_checkpoint"}}
            for worker in workers
        ],
        "artifact_check": check,
        "summary": summary,
    }
    report_path = work_dir / "parallel_patch_report.json"
    md_path = work_dir / "parallel_patch_report.md"
    write_json(report_path, report)
    md_path.write_text(_render_md(report), encoding="utf-8")

    print(f"saved json: {report_path}")
    print(f"saved md: {md_path}")
    print(f"work_dir: {work_dir}")
    print(f"artifact_ok: {check.get('ok')}")
    print(f"selected_item_indexes: {selected}")
    if not check.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
