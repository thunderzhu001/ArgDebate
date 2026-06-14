from __future__ import annotations

import json
from typing import Any, Dict

from ..utils.openai_client import create_openai_client, get_default_model


class ScalarTranscriptJudge:
    """Transcript-level scalar judge baseline without QBAF relations."""

    def __init__(self, model: str = "deepseek-v3.2", max_rounds: int = 3):
        self.client = create_openai_client()
        self.model = get_default_model(model)
        self.max_rounds = int(max_rounds)

    def run(self, task: str, num_agents: int = 3) -> Dict[str, Any]:
        responses: Dict[str, str] = {}
        transcript = []

        for i in range(num_agents):
            agent_id = f"agent_{i}"
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Task: {task}\n"
                            "Provide a concise answer and brief reasoning. "
                            "If this is a yes/no question, start with exactly 'Yes' or 'No'."
                        ),
                    }
                ],
            )
            text = response.choices[0].message.content.strip()
            responses[agent_id] = text
            transcript.append({"round": 0, "agent": agent_id, "text": text})

        for round_num in range(self.max_rounds):
            new_responses: Dict[str, str] = {}
            for i in range(num_agents):
                agent_id = f"agent_{i}"
                others = {aid: text for aid, text in responses.items() if aid != agent_id}
                prompt = (
                    f"Task: {task}\n"
                    f"Your previous answer: {responses[agent_id]}\n"
                    f"Other agents' answers: {others}\n\n"
                    "Update or defend your answer. Keep it concise. "
                    "If this is a yes/no question, start with exactly 'Yes' or 'No'."
                )
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.choices[0].message.content.strip()
                new_responses[agent_id] = text
                transcript.append({"round": round_num + 1, "agent": agent_id, "text": text})
            responses = new_responses

        judge_prompt = (
            f"Task: {task}\n"
            f"Final candidate answers by agent:\n{responses}\n"
            f"Transcript:\n{transcript}\n\n"
            "Score each candidate answer for correctness and evidence quality from 0 to 1, "
            "then pick the best final answer. If this is a yes/no question, final_answer must start with exactly Yes or No. "
            "Return JSON only:\n"
            '{"scores": {"agent_0": 0.0}, "winner": "agent_0", "final_answer": "..."}'
        )
        try:
            judge_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": judge_prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            data = json.loads(judge_response.choices[0].message.content)
        except Exception as exc:
            data = {
                "scores": {},
                "winner": sorted(responses.keys())[0] if responses else "",
                "final_answer": next(iter(responses.values()), ""),
                "error": str(exc),
            }

        winner = str(data.get("winner", "")).strip()
        final_answer = str(data.get("final_answer", "")).strip()
        if not final_answer and winner in responses:
            final_answer = responses[winner]
        if not final_answer:
            final_answer = next(iter(responses.values()), "")

        return {
            "status": "completed",
            "round": self.max_rounds,
            "result": responses,
            "transcript": transcript,
            "judge": data,
            "final_answer": final_answer,
        }
