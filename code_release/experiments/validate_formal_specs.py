#!/usr/bin/env python3
"""Validate formal experiment specs before any paper-eligible run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.model_registry import DEFAULT_REGISTRY, endpoint_summary_for_alias, load_env_file, load_registry
from src.debate.ablation_suite import SUPPORTED_ABLATIONS


DEFAULT_SPECS = [
    ROOT / "experiments" / "config" / "formal_stage_a_readiness.json",
    ROOT / "experiments" / "config" / "formal_stage_b_pilot.json",
    ROOT / "experiments" / "config" / "formal_stage_c_scale.json",
]
RESULTS_DIR = ROOT / "experiments" / "results"

IMPLEMENTED_BASELINES = {
    "ArgDebate",
    "MajorityVote",
    "WeightedVote",
    "VanillaDebate",
    "SelfConsistency",
    "ScalarTranscriptJudge",
}
IMPLEMENTED_ABLATIONS = {
    *SUPPORTED_ABLATIONS,
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dataset_size(path: Path) -> int | None:
    if not path.exists():
        return None
    if path.suffix.lower() == ".csv":
        return max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
    if path.suffix.lower() == ".json":
        data = load_json(path)
        return len(data) if isinstance(data, list) else None
    return None


def _dataset_requested_n(dataset: dict[str, Any]) -> int | None:
    for key in ("formal_n", "pilot_n", "smoke_n"):
        if key in dataset:
            return int(dataset[key])
    return None


def validate_spec(
    spec_path: Path,
    registry: dict[str, Any],
    registry_path: Path,
    *,
    require_secrets: bool,
    allow_planned_missing: bool,
) -> dict[str, Any]:
    spec = load_json(spec_path)
    registry_aliases = {entry["alias"] for entry in registry.get("models", [])}
    errors: list[str] = []
    warnings: list[str] = []

    for alias in spec.get("models", []):
        if alias not in registry_aliases:
            errors.append(f"model alias missing from registry: {alias}")

    judge_policy = spec.get("judge_policy", {})
    for key in ("primary_judge_alias", "backup_judge_alias"):
        alias = judge_policy.get(key)
        if alias and alias not in registry_aliases:
            errors.append(f"judge alias missing from registry: {alias}")
    if judge_policy.get("formal_table_silent_fallback") is not False:
        errors.append("judge_policy.formal_table_silent_fallback must be false")

    endpoint_checks = {}
    for alias in sorted(set(spec.get("models", [])) | {v for k, v in judge_policy.items() if k.endswith("_alias") and v}):
        if alias not in registry_aliases:
            continue
        endpoint = endpoint_summary_for_alias(alias, registry_path)
        endpoint_checks[alias] = endpoint
        if require_secrets:
            if not endpoint.get("api_key_present"):
                errors.append(f"missing API key env for alias: {alias}")
            if not endpoint.get("base_url_present"):
                errors.append(f"missing base URL env for alias: {alias}")

    dataset_checks = []
    for dataset in spec.get("datasets", []):
        rel = Path(str(dataset.get("path", "")))
        path = rel if rel.is_absolute() else ROOT / rel
        planned_missing = dataset.get("status") == "not_yet_created"
        size = dataset_size(path)
        requested_n = _dataset_requested_n(dataset)
        ok = bool(path.exists())
        if not ok and not (allow_planned_missing and planned_missing):
            errors.append(f"dataset missing: {dataset.get('name')} path={dataset.get('path')}")
        if ok and requested_n is not None and size is not None and size < requested_n:
            errors.append(f"dataset too small: {dataset.get('name')} size={size} requested_n={requested_n}")
        if planned_missing:
            warnings.append(f"planned dataset not yet created: {dataset.get('name')}")
        dataset_checks.append(
            {
                "name": dataset.get("name"),
                "path": str(path),
                "exists": ok,
                "size": size,
                "requested_n": requested_n,
                "status": dataset.get("status", ""),
            }
        )

    baselines = set(spec.get("baselines", []))
    missing_baselines = sorted(baselines - IMPLEMENTED_BASELINES)
    if missing_baselines:
        errors.append(f"declared baselines not implemented: {missing_baselines}")

    ablations = set(spec.get("ablations", []))
    main_ablation_run = spec.get("main_ablation_run", {})
    ablations.update(main_ablation_run.get("ablations", []))
    ablations.discard("main_ablation_set_only_if_stage_b_supports_scaling")
    missing_ablations = sorted(ablations - IMPLEMENTED_ABLATIONS)
    if missing_ablations:
        errors.append(f"declared ablations not implemented: {missing_ablations}")

    controls = spec.get("run_controls", {})
    for key in ("num_agents", "max_rounds", "request_timeout_s", "max_retries"):
        if key in controls and int(controls[key]) < 0:
            errors.append(f"run_controls.{key} must be non-negative")
    if int(controls.get("num_agents", 1)) <= 0:
        errors.append("run_controls.num_agents must be > 0")

    artifacts = set(spec.get("artifact_requirements", []))
    if "artifact_manifest_with_hashes" not in artifacts:
        errors.append("artifact_manifest_with_hashes is required")

    return {
        "spec": str(spec_path),
        "name": spec.get("name"),
        "status": spec.get("status"),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "datasets": dataset_checks,
        "endpoint_checks": endpoint_checks,
        "implemented_baselines": sorted(IMPLEMENTED_BASELINES),
        "implemented_ablations": sorted(IMPLEMENTED_ABLATIONS),
    }


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# Formal Spec Validation",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- registry: `{report['registry']}`",
        f"- all_ok: `{report['all_ok']}`",
        "",
    ]
    for item in report["specs"]:
        lines.extend(
            [
                f"## {item['name']}",
                "",
                f"- ok: `{item['ok']}`",
                f"- spec: `{item['spec']}`",
            ]
        )
        if item["errors"]:
            lines.append("- errors:")
            for err in item["errors"]:
                lines.append(f"  - {err}")
        if item["warnings"]:
            lines.append("- warnings:")
            for warning in item["warnings"]:
                lines.append(f"  - {warning}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ArgDebate formal experiment specs")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--spec", action="append", default=[], help="Spec JSON path; repeatable")
    parser.add_argument("--require-secrets", action="store_true")
    parser.add_argument("--no-allow-planned-missing", action="store_true")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    load_env_file(override=False)
    registry_path = Path(args.registry).expanduser().resolve()
    registry = load_registry(registry_path)
    spec_paths = [Path(p).expanduser().resolve() for p in args.spec] or DEFAULT_SPECS
    reports = [
        validate_spec(
            path,
            registry,
            registry_path,
            require_secrets=args.require_secrets,
            allow_planned_missing=not args.no_allow_planned_missing,
        )
        for path in spec_paths
    ]
    payload = {
        "generated_at": datetime.now().isoformat(),
        "registry": str(registry_path),
        "all_ok": all(item["ok"] for item in reports),
        "specs": reports,
    }

    for item in reports:
        status = "OK" if item["ok"] else "FAIL"
        print(f"{status} {item['name']} errors={len(item['errors'])} warnings={len(item['warnings'])}")
        for err in item["errors"]:
            print(f"  error: {err}")
        for warning in item["warnings"]:
            print(f"  warning: {warning}")

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = RESULTS_DIR / f"formal_spec_validation_{stamp}.json"
        md_path = RESULTS_DIR / f"formal_spec_validation_{stamp}.md"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        md_path.write_text(render_md(payload), encoding="utf-8")
        print(f"saved json: {json_path}")
        print(f"saved md: {md_path}")

    if not payload["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
