from typing import List, Dict, Any, Optional
from ..argumentation.argument_extractor import Argument, ArgumentExtractor
from ..utils.openai_client import create_openai_client, get_default_model

class LLMAgent:
    """An LLM agent with persistent beliefs and reasoning capabilities."""

    def __init__(
        self,
        agent_id: str,
        model="deepseek-v3.2",
        proposal_temperature: float = 0.2,
        argument_temperature: float = 0.3,
        belief_update_threshold: float = 0.6,
    ):
        self.agent_id = agent_id
        self.model = get_default_model(model)
        self.client = create_openai_client()
        self.beliefs = []  # List of beliefs (strings or structured)
        self.belief_confidences: Dict[str, float] = {}  # belief text -> confidence
        self.belief_update_threshold = belief_update_threshold
        self.base_proposal_temperature = float(proposal_temperature)
        self.base_argument_temperature = float(argument_temperature)
        self.current_proposal_temperature = float(proposal_temperature)
        self.current_argument_temperature = float(argument_temperature)
        self.proposal_temperature = float(proposal_temperature)
        self.argument_temperature = float(argument_temperature)
        self.extractor = ArgumentExtractor(model=model, temperature=self.argument_temperature)

    def generate_proposal(self, task: str, context: str = "") -> str:
        """Generates an initial proposal for the task.

        Phase 2.4: Step-by-step reasoning prompt with evidence citation request.
        """
        belief_context = "\n".join(self.beliefs[-5:]) if self.beliefs else "None"
        prompt = (
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Your current beliefs:\n{belief_context}\n\n"
            "Think step by step:\n"
            "1. What is the factual core of this question?\n"
            "2. What evidence or data supports your answer?\n"
            "3. What is your final, concise answer?\n\n"
            "Prioritize factual accuracy. Cite specific evidence where possible. "
            "If this is a yes/no question, start the final answer with exactly 'Yes' or 'No'."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.current_proposal_temperature,
        )
        return response.choices[0].message.content

    def generate_arguments(self, task: str, conflicts: str) -> List[Argument]:
        """Generates structured arguments to defend its proposal against conflicts."""
        prompt = (
            f"Task: {task}\n"
            f"Conflicts detected: {conflicts}\n\n"
            "Defend your position with 1-3 structured arguments.\n"
            "Return JSON only in the following schema:\n"
            "{\n"
            '  "arguments": [\n'
            '    {"claim": "...", "evidence": "...", "confidence": 0.0}\n'
            "  ]\n"
            "}\n"
            "Keep each claim concise, each evidence field concrete, and each confidence between 0 and 1."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=self.current_argument_temperature,
        )
        raw_response = response.choices[0].message.content
        return self.extractor.extract(self.agent_id, task, raw_response)

    def update_beliefs(self, accepted_args: List[Argument], defeated_args: List[Argument], strengths: Dict[str, float]):
        """Updates internal beliefs based on the outcome of the debate.

        Phase 2.2: Confidence-tracked belief updates with threshold control.
        - Only adopt external beliefs with strength > belief_update_threshold
        - Track confidence for each belief
        - Replace low-confidence beliefs more readily
        """
        # Remove defeated beliefs (those from this agent that were defeated)
        defeated_claims = {arg.claim for arg in defeated_args if arg.agent_id == self.agent_id}
        if defeated_claims:
            self.beliefs = [b for b in self.beliefs if not any(dc in b for dc in defeated_claims)]
            for claim in defeated_claims:
                self.belief_confidences.pop(claim, None)

        # Adopt high-strength external arguments
        for arg in accepted_args:
            if arg.agent_id == self.agent_id:
                continue
            arg_strength = float(strengths.get(arg.id, arg.confidence))
            if arg_strength >= self.belief_update_threshold:
                belief_text = f"Accepted from {arg.agent_id}: {arg.claim}"
                # Replace existing low-confidence belief if present
                existing_conf = self.belief_confidences.get(belief_text, 0.0)
                if arg_strength > existing_conf:
                    if belief_text not in self.beliefs:
                        self.beliefs.append(belief_text)
                    self.belief_confidences[belief_text] = arg_strength

        print(f"Agent {self.agent_id} updated beliefs based on {len(accepted_args)} accepted and {len(defeated_args)} defeated arguments.")

    def escalate_temperature(self, increment: float = 0.2, max_temp: float = 0.9):
        """Escalate temperature to break deadlock by increasing diversity."""
        self.current_proposal_temperature = min(
            self.current_proposal_temperature + increment,
            max_temp
        )
        self.current_argument_temperature = min(
            self.current_argument_temperature + increment,
            max_temp
        )
        print(f"Agent {self.agent_id} escalated temperature to proposal={self.current_proposal_temperature:.2f}, argument={self.current_argument_temperature:.2f}")

    def reset_temperature(self):
        """Reset temperature to base values."""
        self.current_proposal_temperature = self.base_proposal_temperature
        self.current_argument_temperature = self.base_argument_temperature
