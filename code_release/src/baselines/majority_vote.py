from typing import List, Dict, Any
import json
from ..utils.openai_client import create_openai_client, get_default_model

class MajorityVote:
    """Baseline: N agents answer independently, final answer by majority vote."""
    
    def __init__(self, model="deepseek-v3.2"):
        self.client = create_openai_client()
        self.model = get_default_model(model)

    def run(self, task: str, num_agents: int = 3) -> Dict[str, Any]:
        responses = []
        for i in range(num_agents):
            prompt = (
                f"Task: {task}\n"
                "Provide a concise answer. If this is a yes/no question, start with exactly 'Yes' or 'No'."
            )
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            responses.append(response.choices[0].message.content.strip())
        
        # Use LLM to aggregate and pick the majority/best answer
        agg_prompt = (
            f"Task: {task}\n"
            f"Here are {num_agents} independent answers:\n{responses}\n"
            "Pick the most common or best answer among them. Output ONLY the final answer. "
            "If this is a yes/no question, start with exactly 'Yes' or 'No'."
        )
        final_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": agg_prompt}]
        )
        
        return {
            "status": "completed",
            "responses": responses,
            "final_answer": final_response.choices[0].message.content.strip()
        }
