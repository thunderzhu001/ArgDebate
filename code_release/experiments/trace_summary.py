from __future__ import annotations


def summarize_argdebate_traces(results: list[dict]) -> dict:
    """Summarize route, reason, source, and termination fields from ArgDebate traces."""

    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    termination_counts: dict[str, int] = {}
    route_reason_counts: dict[str, int] = {}

    for item in results:
        ad = item.get("arg_debate", {}) if isinstance(item.get("arg_debate"), dict) else {}
        status = str(ad.get("status", "unknown") or "unknown")
        reason = str(ad.get("reason", "none") or "none")
        source = str(ad.get("final_answer_source", "unknown") or "unknown")
        meta = ad.get("meta", {}) if isinstance(ad.get("meta"), dict) else {}
        termination = meta.get("termination", {}) if isinstance(meta.get("termination"), dict) else {}
        termination_type = str(termination.get("type", "unknown") or "unknown")

        status_counts[status] = status_counts.get(status, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        termination_counts[termination_type] = termination_counts.get(termination_type, 0) + 1

        route_reasons = meta.get("route_reason", [])
        if isinstance(route_reasons, list):
            for route_reason in route_reasons:
                rr = str(route_reason or "unknown")
                route_reason_counts[rr] = route_reason_counts.get(rr, 0) + 1

    n = max(len(results), 1)
    return {
        "status_counts": status_counts,
        "status_rates": {k: v / n for k, v in status_counts.items()},
        "reason_counts": reason_counts,
        "reason_rates": {k: v / n for k, v in reason_counts.items()},
        "final_answer_source_counts": source_counts,
        "final_answer_source_rates": {k: v / n for k, v in source_counts.items()},
        "termination_type_counts": termination_counts,
        "termination_type_rates": {k: v / n for k, v in termination_counts.items()},
        "route_reason_counts": route_reason_counts,
        "route_reason_rates": {k: v / n for k, v in route_reason_counts.items()},
    }
