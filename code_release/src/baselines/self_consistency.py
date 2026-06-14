from __future__ import annotations

from typing import Any, Dict
from collections import Counter

from ..utils.openai_client import create_openai_client, get_default_model


class SelfConsistency:
    """Matched-budget self-consistency baseline."""

    def __init__(self, model: str = "deepseek-v3.2", temperature: float = 0.7):
        self.client = create_openai_client()
        self.model = get_default_model(model)
        self.temperature = float(temperature)

    def _fallback_answer(self, samples: list[str]) -> str:
        if not samples:
            return ""

        normalized = []
        for sample in samples:
            text = str(sample).strip()
            lower = text.lower()
            if lower.startswith("yes") or "final answer: yes" in lower:
                normalized.append("Yes")
            elif lower.startswith("no") or "final answer: no" in lower:
                normalized.append("No")
            else:
                normalized.append(text)

        counts = Counter(normalized)
        return counts.most_common(1)[0][0]

    def run(self, task: str, num_samples: int = 9) -> Dict[str, Any]:
        samples = []
        sample_errors = []
        target_samples = max(1, int(num_samples))
        attempts = 0
        max_attempts = target_samples + 3
        while len(samples) < target_samples and attempts < max_attempts:
            attempts += 1
            for _ in range(2):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": (
                                    f"Task: {task}\n"
                                    "Reason briefly and provide a concise final answer. "
                                    "Use independent reasoning. "
                                    "If this is a yes/no question, start with exactly 'Yes' or 'No'."
                                ),
                            }
                        ],
                        temperature=self.temperature,
                    )
                    samples.append(response.choices[0].message.content.strip())
                    break
                except Exception as exc:
                    sample_errors.append(str(exc))
            if len(samples) >= target_samples:
                break

        if not samples:
            return {
                "status": "error",
                "requested_samples": target_samples,
                "num_samples": 0,
                "sample_error_count": len(sample_errors),
                "sample_errors": sample_errors,
                "samples": [],
                "final_answer": "",
                "error": "all_self_consistency_samples_failed",
            }

        aggregate_prompt = (
            f"Task: {task}\n"
            f"Independent sampled answers:\n{samples}\n\n"
            "Select the answer best supported by the samples. Output ONLY the final answer. "
            "If this is a yes/no question, start with exactly 'Yes' or 'No'."
        )
        aggregate_error = None
        try:
            final_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": aggregate_prompt}],
                temperature=0,
            )
            final_answer = final_response.choices[0].message.content.strip()
        except Exception as exc:
            aggregate_error = str(exc)
            final_answer = self._fallback_answer(samples)

        result = {
            "status": "completed",
            "requested_samples": target_samples,
            "num_samples": len(samples),
            "sample_error_count": len(sample_errors),
            "samples": samples,
            "final_answer": final_answer,
        }
        if sample_errors:
            result["sample_errors"] = sample_errors
        if aggregate_error:
            result["aggregate_error"] = aggregate_error
            result["aggregate_fallback"] = "sample_majority"
        return result
