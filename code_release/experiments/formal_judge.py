#!/usr/bin/env python3
"""Formal Judge Policy module for paper-eligible experiment records."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from experiments.model_registry import DEFAULT_REGISTRY, get_model_entry, load_env_file, load_registry
from src.utils.openai_client import normalize_base_url


PRIMARY_JUDGE_ALIAS = "gpt54_mini"
BACKUP_JUDGE_ALIAS = "deepseek_v4_flash_judge_backup"
JUDGE_POLICY_VERSION = "formal_judge_policy_v1"


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("\n", " ").split())


def _tokenize_text(text: str) -> list[str]:
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in _normalize_text(text))
    return [tok for tok in cleaned.split() if tok]


def _truthfulqa_fallback(prediction: str, target: str) -> bool:
    pred = _normalize_text(prediction)
    gold = _normalize_text(target)
    if not pred:
        return False
    if gold in pred or pred in gold:
        return True
    pred_tokens = set(_tokenize_text(pred))
    gold_tokens = set(_tokenize_text(gold))
    if not gold_tokens:
        return False
    overlap = len(pred_tokens & gold_tokens) / max(1, len(gold_tokens))
    return overlap >= 0.6


def _strategyqa_fallback(prediction: str, target: Any) -> bool:
    pred = _normalize_text(prediction)
    gold = _normalize_text(str(target))
    if gold in {"true", "false"}:
        if "true" in pred or "yes" in pred:
            return gold == "true"
        if "false" in pred or "no" in pred:
            return gold == "false"
        return False
    return bool(pred) and (gold in pred or pred in gold)


class FormalJudge:
    """Deep module for the Formal Judge Policy.

    Interface: evaluate(question, prediction, target, method) -> auditable judge
    record. Provider endpoints and backup/fallback details stay inside.
    """

    def __init__(
        self,
        dataset: str,
        *,
        use_llm_judge: bool = True,
        primary_alias: str | None = None,
        backup_alias: str | None = None,
        registry_path: str | Path = DEFAULT_REGISTRY,
        timeout_s: int | None = None,
        max_retries: int | None = None,
    ):
        load_env_file(override=False)
        self.dataset = dataset
        self.use_llm_judge = bool(use_llm_judge)
        self.primary_alias = primary_alias or os.getenv("EXP_PRIMARY_JUDGE_ALIAS", PRIMARY_JUDGE_ALIAS)
        self.backup_alias = backup_alias or os.getenv("EXP_BACKUP_JUDGE_ALIAS", BACKUP_JUDGE_ALIAS)
        self.registry_path = Path(registry_path).expanduser().resolve()
        self.registry = load_registry(self.registry_path)
        self.timeout_s = int(timeout_s if timeout_s is not None else os.getenv("REQUEST_TIMEOUT", "90"))
        self.max_retries = int(max_retries if max_retries is not None else os.getenv("MAX_RETRIES", "1"))

    def policy_summary(self) -> dict[str, Any]:
        return {
            "judge_policy_version": JUDGE_POLICY_VERSION,
            "dataset": self.dataset,
            "use_llm_judge": self.use_llm_judge,
            "primary_judge_alias": self.primary_alias,
            "backup_judge_alias": self.backup_alias,
            "formal_table_silent_fallback": False,
        }

    def evaluate(self, question: str, prediction: str, target: Any, method: str = "") -> dict[str, Any]:
        prediction = str(prediction or "").strip()
        if not prediction:
            return self._fallback_record(
                question=question,
                prediction=prediction,
                target=target,
                method=method,
                error="empty_prediction",
                attempts=[],
            )

        attempts: list[dict[str, Any]] = []
        if self.use_llm_judge:
            for alias in [self.primary_alias, self.backup_alias]:
                attempt = self._attempt_llm_judge(alias, question, prediction, target)
                attempts.append(attempt)
                if attempt["ok"]:
                    data = attempt.get("data") or {}
                    return {
                        "correct": bool(data.get("correct", False)),
                        "dataset": self.dataset,
                        "method": method,
                        "judge_policy_version": JUDGE_POLICY_VERSION,
                        "judge_prompt_version": self._prompt_version(),
                        "judge_alias": alias,
                        "judge_model": attempt.get("model_id"),
                        "judge_raw_response": data,
                        "judge_raw_text": attempt.get("raw_text"),
                        "judge_fallback_used": False,
                        "judge_fallback_type": None,
                        "judge_error": None,
                        "judge_attempts": attempts,
                        "formal_table_eligible": True,
                    }

        return self._fallback_record(
            question=question,
            prediction=prediction,
            target=target,
            method=method,
            error="llm_judge_disabled" if not self.use_llm_judge else "all_judge_attempts_failed",
            attempts=attempts,
        )

    def _attempt_llm_judge(self, alias: str, question: str, prediction: str, target: Any) -> dict[str, Any]:
        started = time.time()
        try:
            client, model_id = self._client_for_alias(alias)
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": self._prompt(question, prediction, target)}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = response.choices[0].message.content or ""
            data = json.loads(raw)
            return {
                "alias": alias,
                "model_id": model_id,
                "ok": True,
                "latency_s": round(time.time() - started, 3),
                "raw_text": raw,
                "data": data,
                "error": None,
            }
        except Exception as exc:
            return {
                "alias": alias,
                "model_id": None,
                "ok": False,
                "latency_s": round(time.time() - started, 3),
                "raw_text": None,
                "data": None,
                "error": str(exc),
            }

    def _client_for_alias(self, alias: str) -> tuple[OpenAI, str]:
        entry = get_model_entry(alias, self.registry)
        api_key = os.getenv(str(entry["api_key_env"]), "")
        base_url = normalize_base_url(os.getenv(str(entry["base_url_env"]), ""))
        if not api_key:
            raise RuntimeError(f"missing judge API key env: {entry['api_key_env']}")
        if not base_url:
            raise RuntimeError(f"missing judge base URL env: {entry['base_url_env']}")
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout_s,
            max_retries=self.max_retries,
        )
        return client, str(entry["model_id"])

    def _fallback_record(
        self,
        *,
        question: str,
        prediction: str,
        target: Any,
        method: str,
        error: str,
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        correct = self._deterministic_fallback(prediction, target)
        return {
            "correct": bool(correct),
            "dataset": self.dataset,
            "method": method,
            "judge_policy_version": JUDGE_POLICY_VERSION,
            "judge_prompt_version": self._prompt_version(),
            "judge_alias": None,
            "judge_model": None,
            "judge_raw_response": None,
            "judge_raw_text": None,
            "judge_fallback_used": True,
            "judge_fallback_type": "deterministic_dataset_fallback",
            "judge_error": error,
            "judge_attempts": attempts,
            "formal_table_eligible": (not self.use_llm_judge and self.dataset.lower() == "strategyqa"),
        }

    def _deterministic_fallback(self, prediction: str, target: Any) -> bool:
        if self.dataset.lower() == "strategyqa":
            return _strategyqa_fallback(prediction, target)
        return _truthfulqa_fallback(prediction, str(target))

    def _prompt_version(self) -> str:
        if self.dataset.lower() == "strategyqa":
            return "strategyqa_correctness_v2_formal_judge"
        return "truthfulqa_correctness_v2_formal_judge"

    def _prompt(self, question: str, prediction: str, target: Any) -> str:
        if self.dataset.lower() == "strategyqa":
            return f"""
Evaluate whether the candidate answer to a yes/no question is correct.

Question: {question}
Gold answer: {target}
Candidate answer: {prediction}

Return JSON only:
{{"correct": true or false}}
"""
        return f"""
You are an evaluator for QA correctness.

Question: {question}
Reference answer: {target}
Model answer: {prediction}

Determine whether the model answer is factually correct given the reference.
Return JSON only:
{{"correct": true or false}}
"""
