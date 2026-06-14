from typing import List, Dict, Any
import json
import re
from ..utils.openai_client import create_openai_client, get_default_model

class ConflictDetector:
    """Detects conflicts between multiple agent proposals."""
    
    def __init__(self, model="deepseek-v3.2", use_llm: bool = True):
        self.client = create_openai_client()
        self.model = get_default_model(model)
        self.use_llm = bool(use_llm)

    def _normalize(self, text: str) -> str:
        return " ".join(str(text).strip().lower().replace("\n", " ").split())

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+", self._normalize(text)))

    def _extract_numbers(self, text: str) -> set[str]:
        return set(re.findall(r"\b\d+(?:\.\d+)?\b", str(text)))

    def _has_negation(self, text: str) -> bool:
        markers = [" no ", " not ", " never ", " cannot ", " can't ", " false ", " incorrect ", " wrong "]
        padded = f" {self._normalize(text)} "
        return any(marker in padded for marker in markers)

    def _heuristic_conflicts(self, proposals: Dict[str, str]) -> List[str]:
        agent_ids = sorted(proposals.keys())
        conflicts: List[str] = []

        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                a_id = agent_ids[i]
                b_id = agent_ids[j]
                a_text = str(proposals.get(a_id, ""))
                b_text = str(proposals.get(b_id, ""))
                if not a_text.strip() or not b_text.strip():
                    continue

                ta = self._tokenize(a_text)
                tb = self._tokenize(b_text)
                if not ta or not tb:
                    continue

                overlap = len(ta & tb) / max(1, len(ta | tb))
                a_neg = self._has_negation(a_text)
                b_neg = self._has_negation(b_text)
                if overlap > 0.15 and (a_neg != b_neg):
                    conflicts.append(
                        f"Heuristic contradiction between {a_id} and {b_id}: similar topic but opposite polarity."
                    )
                    continue

                nums_a = self._extract_numbers(a_text)
                nums_b = self._extract_numbers(b_text)
                if overlap > 0.12 and nums_a and nums_b and nums_a != nums_b:
                    conflicts.append(
                        f"Heuristic contradiction between {a_id} and {b_id}: conflicting numerical claims ({sorted(nums_a)} vs {sorted(nums_b)})."
                    )

        return conflicts

    def detect(self, proposals: Dict[str, str]) -> List[str]:
        """
        Detects semantic conflicts between proposals.
        Returns a list of conflict descriptions.
        """
        if len(proposals) < 2:
            return []
        if not self.use_llm:
            return self._heuristic_conflicts(proposals)
            
        prompt = f"""
Analyze the following proposals from different agents for the same task.
Identify any direct contradictions, significant disagreements, or conflicting reasoning.

Proposals:
{json.dumps(proposals, ensure_ascii=False, indent=2)}

Output a list of specific conflicts found. If no conflicts exist, output "No conflicts".
JSON Output:
{{
  "conflicts": ["Conflict 1 description", "Conflict 2 description", ...]
}}
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(response.choices[0].message.content)
            conflicts = data.get("conflicts", [])
            if isinstance(conflicts, str) and "no conflict" in conflicts.lower():
                return []
            elif isinstance(conflicts, list):
                llm_conflicts = [str(item).strip() for item in conflicts if str(item).strip()]
            else:
                llm_conflicts = []

            if llm_conflicts:
                return llm_conflicts
            return []
        except Exception as e:
            print(f"Error detecting conflicts: {e}")
            return self._heuristic_conflicts(proposals)
