from typing import Dict, Any
import ast
import json
import re
from ..utils.openai_client import create_openai_client, get_default_model

class WeightedVote:
    """Baseline: N agents answer independently with confidence, final answer by weighted vote."""
    
    def __init__(self, model="deepseek-v3.2"):
        self.client = create_openai_client()
        self.model = get_default_model(model)

    def _completion(self, messages, *, response_format=None, **kwargs):
        request = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        if response_format is not None:
            request["response_format"] = response_format
        try:
            return self.client.chat.completions.create(**request)
        except Exception:
            if response_format is None:
                raise
            request.pop("response_format", None)
            return self.client.chat.completions.create(**request)

    def _coerce_confidence(self, value: Any, default: float = 0.5) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))
        text = str(value or "").strip()
        if not text:
            return default
        try:
            number = float(text)
        except ValueError:
            matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", text)
            number = None
            for raw in matches:
                try:
                    candidate = float(raw)
                except ValueError:
                    continue
                if 0.0 <= candidate <= 1.0:
                    number = candidate
                    break
                if 1.0 < candidate <= 100.0:
                    number = candidate / 100.0
                    break
            if number is None:
                return default
        if number > 1.0 and number <= 100.0:
            number /= 100.0
        return max(0.0, min(1.0, float(number)))

    def _json_candidates(self, raw: str) -> list[str]:
        text = str(raw or "").strip()
        candidates = [text]
        candidates.extend(
            chunk.strip()
            for chunk in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
            if chunk.strip()
        )
        if "{" in text and "}" in text:
            candidates.append(text[text.find("{") : text.rfind("}") + 1])
        return [candidate for candidate in candidates if candidate]

    def _parse_agent_response(self, raw: str) -> Dict[str, Any]:
        text = str(raw or "").strip()
        for candidate in self._json_candidates(text):
            payload = None
            try:
                payload = json.loads(candidate)
            except Exception:
                try:
                    payload = ast.literal_eval(candidate)
                except Exception:
                    payload = None
            if isinstance(payload, dict):
                answer = str(payload.get("answer", "")).strip()
                if answer:
                    return {
                        "answer": answer,
                        "confidence": self._coerce_confidence(payload.get("confidence", 0.5)),
                        "parse_status": "structured",
                    }

        confidence_match = re.search(
            r"confidence\s*[:=]\s*([0-9]+(?:\.[0-9]+)?%?)",
            text,
            flags=re.IGNORECASE,
        )
        confidence = 0.5
        if confidence_match:
            confidence = self._coerce_confidence(confidence_match.group(1).rstrip("%"))
        return {
            "answer": text,
            "confidence": confidence,
            "parse_status": "raw_text_fallback",
        }

    def run(self, task: str, num_agents: int = 3) -> Dict[str, Any]:
        responses = []
        response_errors = []
        for i in range(num_agents):
            prompt = f"""
Task: {task}
Provide a concise answer and your confidence score (0.0 to 1.0).
If this is a yes/no question, the answer field must start with exactly "Yes" or "No".
Output ONLY as JSON:
{{
  "answer": "Your answer",
  "confidence": 0.95
}}
"""
            last_error = None
            for _attempt in range(2):
                try:
                    response = self._completion(
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0,
                    )
                    data = self._parse_agent_response(response.choices[0].message.content)
                    responses.append(data)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = str(exc)
            if last_error:
                response_errors.append({"agent": i, "error": last_error})

        if not responses:
            return {
                "status": "error",
                "requested_agents": int(num_agents),
                "responses": [],
                "response_error_count": len(response_errors),
                "response_errors": response_errors,
                "final_answer": "",
                "error": "all_weighted_vote_agents_failed",
            }
        
        # Use LLM to aggregate and pick the best answer based on confidence
        agg_prompt = (
            f"Task: {task}\n"
            f"Here are {num_agents} independent answers with confidence scores:\n{responses}\n"
            "Pick the best answer among them. Output ONLY the final answer. "
            "If this is a yes/no question, start with exactly 'Yes' or 'No'."
        )
        aggregate_error = None
        try:
            final_response = self._completion(
                messages=[{"role": "user", "content": agg_prompt}],
                temperature=0,
            )
            final_answer = final_response.choices[0].message.content.strip()
        except Exception as exc:
            aggregate_error = str(exc)
            final_answer = max(responses, key=lambda item: float(item.get("confidence", 0.0))).get("answer", "")
        
        result = {
            "status": "completed",
            "requested_agents": int(num_agents),
            "num_responses": len(responses),
            "response_error_count": len(response_errors),
            "responses": responses,
            "final_answer": str(final_answer).strip(),
        }
        if response_errors:
            result["response_errors"] = response_errors
        if aggregate_error:
            result["aggregate_error"] = aggregate_error
            result["aggregate_fallback"] = "highest_confidence_response"
        return result
