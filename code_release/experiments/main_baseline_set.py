#!/usr/bin/env python3
"""Shared Main Baseline Set helpers for formal experiments.

The Main Baseline Set is a paper-facing interface, not a pair of
dataset-local conventions.  This module keeps method ordering, prediction
extraction, and lightweight accounting in one place so dataset runners can
focus on dataset-specific scoring.
"""

from __future__ import annotations

from typing import Any


CORE_BASELINE_METHODS = ("majority_vote", "vanilla_debate", "weighted_vote")
SUPPORTED_EXTRA_BASELINES = ("self_consistency", "scalar_transcript_judge")
ARGDEBATE_METHOD = "arg_debate"


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().replace("\n", " ").split())


def select_consensus_response(response_dict: dict[str, Any]) -> str:
    items = []
    for key in sorted(response_dict.keys()):
        value = str(response_dict.get(key, "")).strip()
        if value:
            items.append(value)
    if not items:
        return ""

    buckets: dict[str, list[str]] = {}
    for text in items:
        buckets.setdefault(normalize_text(text), []).append(text)

    best_bucket = sorted(
        buckets.values(),
        key=lambda values: (-len(values), len(min(values, key=len))),
    )[0]
    return min(best_bucket, key=len)


def estimate_tokens(text: Any) -> int:
    return max(1, len(str(text)) // 4)


def validate_extra_baselines(extra_baselines: list[str] | None) -> list[str]:
    requested = [str(name).strip() for name in extra_baselines or [] if str(name).strip()]
    invalid = [name for name in requested if name not in SUPPORTED_EXTRA_BASELINES]
    if invalid:
        raise ValueError(f"Unsupported EXP_EXTRA_BASELINES: {invalid}")
    return requested


def configured_methods(extra_baselines: list[str] | None = None) -> list[str]:
    methods = list(CORE_BASELINE_METHODS)
    for method in validate_extra_baselines(extra_baselines):
        if method not in methods:
            methods.append(method)
    methods.append(ARGDEBATE_METHOD)
    return methods


def empty_stats_dict(methods: list[str]) -> dict[str, list[Any]]:
    return {method: [] for method in methods}


def empty_correctness_dict(methods: list[str]) -> dict[str, int]:
    return {method: 0 for method in methods}


def status_counts(status_log: dict[str, list[Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for method, statuses in status_log.items():
        method_counts: dict[str, int] = {}
        for status in statuses:
            key = str(status or "unknown")
            method_counts[key] = method_counts.get(key, 0) + 1
        counts[method] = method_counts
    return counts


def extract_method_prediction(
    method_result: Any,
    method_name: str,
    *,
    prefer_formal_normalized: bool = True,
) -> str:
    result = method_result if isinstance(method_result, dict) else {}
    if prefer_formal_normalized and str(result.get("formal_normalized_final_answer", "")).strip():
        return str(result.get("formal_normalized_final_answer", ""))
    if str(result.get("final_answer", "")).strip():
        return str(result.get("final_answer", ""))
    if method_name in {"majority_vote", "weighted_vote"}:
        return str(result.get("final_answer", ""))
    if isinstance(result.get("result"), dict):
        return select_consensus_response(result["result"])
    return ""
