#!/usr/bin/env python3
"""Formal Table Eligibility and deterministic scoring rules.

This module keeps paper-table inclusion policy out of dataset runners.  The
interface is deliberately small: produce a dataset scoring record, produce a
method eligibility record, and summarize eligibility across result rows.
"""

from __future__ import annotations

import json
import re
from typing import Any


FORMAL_TABLE_ELIGIBILITY_VERSION = "formal_table_eligibility_v1"
STRATEGYQA_BOOLEAN_SCORING_VERSION = "strategyqa_boolean_v1"


TRUE_TOKENS = {"true", "yes"}
FALSE_TOKENS = {"false", "no"}
BOOLEAN_TOKENS = TRUE_TOKENS | FALSE_TOKENS


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().lower().replace("\n", " ").split())


def _boolean_from_token(token: str) -> bool | None:
    token = token.strip().lower().strip(" .,:;!?\"'`()[]{}")
    if token in TRUE_TOKENS:
        return True
    if token in FALSE_TOKENS:
        return False
    return None


def coerce_boolean_target(target: Any) -> bool | None:
    if isinstance(target, bool):
        return target
    text = normalize_text(target)
    return _boolean_from_token(text)


def _parse_json_boolean(text: str) -> bool | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if isinstance(payload, bool):
        return payload
    if isinstance(payload, dict):
        for key in ("final_answer", "answer", "prediction", "label"):
            if key in payload:
                value = payload[key]
                if isinstance(value, bool):
                    return value
                parsed = _boolean_from_token(normalize_text(value))
                if parsed is not None:
                    return parsed
    return None


def parse_strategyqa_prediction(prediction: Any) -> tuple[bool | None, str]:
    text = normalize_text(prediction)
    if not text:
        return None, "empty_prediction"

    parsed_json = _parse_json_boolean(str(prediction).strip())
    if parsed_json is not None:
        return parsed_json, "json_field"

    final_patterns = (
        r"\b(?:final\s+answer|concise\s+answer|answer|conclusion|previous\s+answer)\b[\W_]{0,40}(yes|no|true|false)\b",
        r"\b(?:therefore|so|thus)\s*[,:\-]?\s*(yes|no|true|false)\b",
        r"\b(?:factual\s+core|final\s+defense)\b[\W_]{0,80}(yes|no|true|false)\b",
    )
    for pattern in final_patterns:
        match = re.search(pattern, text)
        if match:
            return bool(_boolean_from_token(match.group(1))), "explicit_final_answer"

    first_token = re.match(r"^\W*(yes|no|true|false)\b", text)
    if first_token:
        return bool(_boolean_from_token(first_token.group(1))), "leading_boolean"

    tokens = set(re.findall(r"\b[a-z]+\b", text))
    has_true = bool(tokens & TRUE_TOKENS)
    has_false = bool(tokens & FALSE_TOKENS)
    if has_true and not has_false:
        return True, "single_boolean_token"
    if has_false and not has_true:
        return False, "single_boolean_token"
    if has_true and has_false:
        return None, "ambiguous_boolean_tokens"
    return None, "no_boolean_token"


def score_strategyqa_boolean(prediction: Any, target: Any, *, method: str = "") -> dict[str, Any]:
    gold = coerce_boolean_target(target)
    parsed, parse_status = parse_strategyqa_prediction(prediction)
    eligible = gold is not None and parsed is not None
    return {
        "dataset": "StrategyQA",
        "method": method,
        "correct": bool(eligible and parsed == gold),
        "scoring_policy": "deterministic_boolean",
        "scoring_policy_version": STRATEGYQA_BOOLEAN_SCORING_VERSION,
        "prediction": str(prediction or ""),
        "parsed_prediction": parsed,
        "gold_answer": gold,
        "parse_status": parse_status,
        "formal_table_eligible": bool(eligible),
        "eligibility_reason": "ok" if eligible else f"deterministic_scoring_{parse_status if parsed is None else 'invalid_gold'}",
    }


def method_formal_table_eligibility(
    *,
    dataset: str,
    method: str,
    method_result: Any,
    scoring_record: dict[str, Any] | None = None,
    judge_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    result = method_result if isinstance(method_result, dict) else {}
    status = str(result.get("status", "completed") or "completed")
    error = str(result.get("error", "") or "")

    if status in {"timeout", "error"}:
        reasons.append(f"method_status_{status}")
    if error.startswith("method_timeout_after_"):
        reasons.append("method_timeout")
    if result.get("timeout") is True:
        reasons.append("method_timeout")

    dataset_key = str(dataset).strip().lower()
    if dataset_key == "strategyqa":
        if not scoring_record:
            reasons.append("deterministic_scoring_missing")
        elif not bool(scoring_record.get("formal_table_eligible", False)):
            reasons.append(str(scoring_record.get("eligibility_reason", "deterministic_scoring_ineligible")))
    else:
        if not judge_record:
            reasons.append("formal_judge_record_missing")
        elif not bool(judge_record.get("formal_table_eligible", False)):
            reasons.append(str(judge_record.get("judge_error") or "formal_judge_ineligible"))

    # Stable ordering keeps tests and artifact diffs readable.
    deduped = sorted(set(reasons))
    return {
        "dataset": dataset,
        "method": method,
        "formal_table_eligible": not deduped,
        "eligibility_policy_version": FORMAL_TABLE_ELIGIBILITY_VERSION,
        "reasons": deduped,
    }


def summarize_formal_table_eligibility(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    method_counts: dict[str, dict[str, Any]] = {}
    ineligible_total = 0
    eligible_total = 0

    for method in methods:
        records = []
        for row in rows:
            by_method = row.get("formal_table_eligibility", {})
            if isinstance(by_method, dict) and isinstance(by_method.get(method), dict):
                records.append(by_method[method])
            else:
                records.append(
                    {
                        "formal_table_eligible": False,
                        "reasons": ["formal_table_eligibility_missing"],
                    }
                )
        eligible = sum(1 for record in records if record.get("formal_table_eligible") is True)
        ineligible = len(records) - eligible
        reason_counts: dict[str, int] = {}
        for record in records:
            if record.get("formal_table_eligible") is True:
                continue
            for reason in record.get("reasons", []) or ["unknown"]:
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
        method_counts[method] = {
            "total": len(records),
            "eligible": eligible,
            "ineligible": ineligible,
            "reason_counts": reason_counts,
        }
        eligible_total += eligible
        ineligible_total += ineligible

    return {
        "policy_version": FORMAL_TABLE_ELIGIBILITY_VERSION,
        "eligible_total": eligible_total,
        "ineligible_total": ineligible_total,
        "method_counts": method_counts,
    }


def formal_table_ineligible_count(summary: Any) -> int:
    """Return formal-table ineligible count.

    Missing eligibility metadata is itself ineligible for paper tables.  Older
    artifacts did not always store this block; treating absence as zero can
    accidentally promote non-auditable runs into manuscript tables.
    """
    if not isinstance(summary, dict):
        return 1
    if "formal_table_eligibility" not in summary:
        return 1
    eligibility = summary.get("formal_table_eligibility")
    if not isinstance(eligibility, dict):
        return 1
    if "ineligible_total" not in eligibility:
        return 1
    return int(eligibility.get("ineligible_total", 0) or 0)
