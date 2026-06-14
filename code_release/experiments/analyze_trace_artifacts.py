#!/usr/bin/env python3
"""
Analyze ArgDebate result/checkpoint artifacts into paper-friendly trace summaries.

Usage:
  python3 experiments/analyze_trace_artifacts.py --input <json-path>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "results"


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_result_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        return [item for item in payload["results"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise RuntimeError("Unsupported artifact format: expected checkpoint dict with results or a result list")


def _count_values(items: Iterable[Any]) -> Dict[str, int]:
    counter = Counter()
    for item in items:
        key = str(item if item not in {None, ""} else "none")
        counter[key] += 1
    return dict(counter)


def _rate_dict(counts: Dict[str, int], total: int) -> Dict[str, float]:
    if total <= 0:
        return {k: 0.0 for k in counts}
    return {k: round(v / total, 6) for k, v in counts.items()}


def _route_mode_label(meta: Dict[str, Any], res: Dict[str, Any]) -> str:
    status = str(res.get("status", "unknown") or "unknown")
    debate_routed = meta.get("debate_routed")
    if debate_routed is False or status == "routed_skip":
        return "routing_shortcut"
    if debate_routed is True:
        return "full_debate"
    return "unknown"


def _primary_route_reason(meta: Dict[str, Any]) -> str:
    reasons = meta.get("route_reason", [])
    if isinstance(reasons, list) and reasons:
        return str(reasons[0])
    return "none"


def _collapsed_final_source(source: Any) -> str:
    source_text = str(source or "unknown")
    if source_text.startswith("agent_weighted:"):
        return "agent_weighted_selector"
    if source_text.startswith("routing_"):
        return "routing_shortcut"
    if source_text == "proposal_consensus":
        return "proposal_consensus"
    return source_text


def analyze(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(items)

    arg_results = [item.get("arg_debate", {}) if isinstance(item.get("arg_debate"), dict) else {} for item in items]
    statuses = _count_values(res.get("status", "unknown") for res in arg_results)
    reasons = _count_values(res.get("reason", "none") for res in arg_results)
    answer_sources = _count_values(res.get("final_answer_source", "unknown") for res in arg_results)

    route_reason_counter = Counter()
    termination_counter = Counter()
    routed_flags: List[int] = []
    round_counts: List[int] = []
    relation_counts: List[int] = []
    attack_counts: List[int] = []
    support_counts: List[int] = []
    argument_counts: List[int] = []
    flow_records: List[Dict[str, Any]] = []
    route_to_status = Counter()
    status_to_source = Counter()
    route_to_status_to_source = Counter()

    for idx, res in enumerate(arg_results, start=1):
        meta = res.get("meta", {}) if isinstance(res.get("meta"), dict) else {}
        if "debate_routed" in meta:
            routed_flags.append(1 if bool(meta.get("debate_routed")) else 0)

        termination = meta.get("termination", {}) if isinstance(meta.get("termination"), dict) else {}
        termination_counter[str(termination.get("type", "unknown"))] += 1

        for reason in meta.get("route_reason", []) if isinstance(meta.get("route_reason"), list) else []:
            route_reason_counter[str(reason)] += 1

        diagnostics = meta.get("round_diagnostics", [])
        route_mode = _route_mode_label(meta, res)
        status = str(res.get("status", "unknown") or "unknown")
        final_source = str(res.get("final_answer_source", "unknown") or "unknown")
        collapsed_source = _collapsed_final_source(final_source)
        primary_route_reason = _primary_route_reason(meta)

        route_to_status[(route_mode, status)] += 1
        status_to_source[(status, collapsed_source)] += 1
        route_to_status_to_source[(route_mode, status, collapsed_source)] += 1

        record = {
            "item_index": idx,
            "route_mode": route_mode,
            "route_reason_primary": primary_route_reason,
            "status": status,
            "termination_type": str(termination.get("type", "unknown") or "unknown"),
            "final_answer_source": final_source,
            "final_answer_source_group": collapsed_source,
            "rounds_recorded": len(diagnostics) if isinstance(diagnostics, list) else 0,
        }

        if isinstance(diagnostics, list) and diagnostics:
            round_counts.append(len(diagnostics))
            last_diag = diagnostics[-1] if isinstance(diagnostics[-1], dict) else {}
            qbaf_data = last_diag.get("qbaf_data", {}) if isinstance(last_diag.get("qbaf_data"), dict) else {}
            relations = qbaf_data.get("relations", []) if isinstance(qbaf_data.get("relations"), list) else []
            arguments = qbaf_data.get("arguments", []) if isinstance(qbaf_data.get("arguments"), list) else []
            relation_counts.append(len(relations))
            argument_counts.append(len(arguments))
            attack_counts.append(sum(1 for rel in relations if isinstance(rel, dict) and rel.get("type") == "attack"))
            support_counts.append(sum(1 for rel in relations if isinstance(rel, dict) and rel.get("type") == "support"))
            record["accepted_args_last_round"] = int(last_diag.get("accepted_args", 0) or 0)
            record["defeated_args_last_round"] = int(last_diag.get("defeated_args", 0) or 0)
            record["conflicts_last_round"] = int(last_diag.get("conflict_count", 0) or 0)
        else:
            record["accepted_args_last_round"] = 0
            record["defeated_args_last_round"] = 0
            record["conflicts_last_round"] = 0

        flow_records.append(record)

    return {
        "n_items": total,
        "status_counts": statuses,
        "status_rates": _rate_dict(statuses, total),
        "reason_counts": reasons,
        "reason_rates": _rate_dict(reasons, total),
        "final_answer_source_counts": answer_sources,
        "final_answer_source_rates": _rate_dict(answer_sources, total),
        "route_reason_counts": dict(route_reason_counter),
        "route_reason_rates": _rate_dict(dict(route_reason_counter), total),
        "termination_type_counts": dict(termination_counter),
        "termination_type_rates": _rate_dict(dict(termination_counter), total),
        "aggregate_trace_stats": {
            "debate_routed_rate": round(mean(routed_flags), 6) if routed_flags else None,
            "avg_recorded_rounds": round(mean(round_counts), 6) if round_counts else None,
            "avg_last_round_relations": round(mean(relation_counts), 6) if relation_counts else None,
            "avg_last_round_arguments": round(mean(argument_counts), 6) if argument_counts else None,
            "avg_last_round_attacks": round(mean(attack_counts), 6) if attack_counts else None,
            "avg_last_round_supports": round(mean(support_counts), 6) if support_counts else None,
        },
        "flow_summary": {
            "route_mode_counts": _count_values(record["route_mode"] for record in flow_records),
            "route_reason_primary_counts": _count_values(record["route_reason_primary"] for record in flow_records),
            "final_answer_source_group_counts": _count_values(record["final_answer_source_group"] for record in flow_records),
            "route_to_status_counts": [
                {"route_mode": route_mode, "status": status, "count": count}
                for (route_mode, status), count in sorted(route_to_status.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
            ],
            "status_to_source_counts": [
                {"status": status, "final_answer_source_group": source_group, "count": count}
                for (status, source_group), count in sorted(status_to_source.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
            ],
            "route_to_status_to_source_counts": [
                {
                    "route_mode": route_mode,
                    "status": status,
                    "final_answer_source_group": source_group,
                    "count": count,
                }
                for (route_mode, status, source_group), count in sorted(
                    route_to_status_to_source.items(),
                    key=lambda kv: (-kv[1], kv[0][0], kv[0][1], kv[0][2]),
                )
            ],
            "flow_records": flow_records,
        },
    }


def render_md(summary: Dict[str, Any], input_path: Path) -> str:
    lines = [
        "# ArgDebate Trace Audit",
        "",
        f"- source: `{input_path}`",
        f"- generated_at: `{datetime.now().isoformat()}`",
        f"- n_items: `{summary['n_items']}`",
        "",
        "## Status Distribution",
        "",
        "| status | count | rate |",
        "|---|---:|---:|",
    ]
    for key, count in sorted(summary["status_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {key} | {count} | {summary['status_rates'].get(key, 0.0):.4f} |")

    lines.extend([
        "",
        "## Fallback / Termination Reasons",
        "",
        "| reason | count | rate |",
        "|---|---:|---:|",
    ])
    for key, count in sorted(summary["reason_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {key} | {count} | {summary['reason_rates'].get(key, 0.0):.4f} |")

    lines.extend([
        "",
        "## Final Answer Sources",
        "",
        "| source | count | rate |",
        "|---|---:|---:|",
    ])
    for key, count in sorted(summary["final_answer_source_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {key} | {count} | {summary['final_answer_source_rates'].get(key, 0.0):.4f} |")

    lines.extend([
        "",
        "## Termination Types",
        "",
        "| termination_type | count | rate |",
        "|---|---:|---:|",
    ])
    for key, count in sorted(summary["termination_type_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {key} | {count} | {summary['termination_type_rates'].get(key, 0.0):.4f} |")

    lines.extend([
        "",
        "## Route Reasons",
        "",
        "| route_reason | count | rate |",
        "|---|---:|---:|",
    ])
    for key, count in sorted(summary["route_reason_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {key} | {count} | {summary['route_reason_rates'].get(key, 0.0):.4f} |")

    stats = summary["aggregate_trace_stats"]
    lines.extend([
        "",
        "## Aggregate Trace Stats",
        "",
        f"- debate_routed_rate: `{stats.get('debate_routed_rate')}`",
        f"- avg_recorded_rounds: `{stats.get('avg_recorded_rounds')}`",
        f"- avg_last_round_arguments: `{stats.get('avg_last_round_arguments')}`",
        f"- avg_last_round_relations: `{stats.get('avg_last_round_relations')}`",
        f"- avg_last_round_attacks: `{stats.get('avg_last_round_attacks')}`",
        f"- avg_last_round_supports: `{stats.get('avg_last_round_supports')}`",
    ])

    flow = summary.get("flow_summary", {}) if isinstance(summary.get("flow_summary"), dict) else {}
    route_counts = flow.get("route_mode_counts", {}) if isinstance(flow, dict) else {}
    source_group_counts = flow.get("final_answer_source_group_counts", {}) if isinstance(flow, dict) else {}
    lines.extend([
        "",
        "## Flow Summary",
        "",
        "| route_mode | count |",
        "|---|---:|",
    ])
    for key, count in sorted(route_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {key} | {count} |")
    lines.extend([
        "",
        "| final_answer_source_group | count |",
        "|---|---:|",
    ])
    for key, count in sorted(source_group_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {key} | {count} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ArgDebate trace artifacts")
    parser.add_argument("--input", required=True, help="Path to checkpoint or results JSON")
    parser.add_argument("--output-prefix", default="", help="Optional output filename prefix")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    payload = load_json(input_path)
    items = normalize_result_items(payload)
    summary = analyze(items)
    summary["source"] = str(input_path)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    json_path = OUT_DIR / f"{prefix}trace_audit_{stamp}.json"
    md_path = OUT_DIR / f"{prefix}trace_audit_{stamp}.md"

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_md(summary, input_path), encoding="utf-8")

    print(f"saved json: {json_path}")
    print(f"saved md: {md_path}")


if __name__ == "__main__":
    main()
