import json
import re
import time
from typing import List, Dict, Optional, Any
import requests
from ..utils.openai_client import create_openai_client, get_default_model

class Argument:
    def __init__(self, id: str, agent_id: str, claim: str, evidence: str, confidence: float, source: str = "LLM"):
        self.id = id
        self.agent_id = agent_id
        self.claim = claim
        self.evidence = evidence
        self.confidence = confidence
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "claim": self.claim,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "source": self.source
        }

class ArgumentExtractor:
    """
    Extracts structured arguments from LLM natural language output with RAG support and retry logic.
    
    Attributes:
        model (str): The LLM model to use for extraction.
        max_retries (int): Maximum number of retries for API calls.
    """
    
    def __init__(self, model: str = "gpt-4.1-mini", max_retries: int = 3, temperature: float = 0.3):
        self.client = create_openai_client()
        self.model = get_default_model(model)
        self.max_retries = max_retries
        self.temperature = float(temperature)

    def _coerce_argument_list(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            candidate = payload.get("arguments", [])
            if isinstance(candidate, list) and candidate:
                return [item for item in candidate if isinstance(item, dict)]
            for value in payload.values():
                if isinstance(value, list):
                    return [item for item in value if isinstance(item, dict)]
            return []
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        return []

    def _parse_direct_payload(self, raw_response: str) -> List[Dict[str, Any]]:
        text = str(raw_response or "").strip()
        if not text:
            return []

        candidates = [text]
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(chunk.strip() for chunk in fenced if chunk.strip())

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            parsed = self._coerce_argument_list(payload)
            if parsed:
                return parsed
        return []

    def _coerce_confidence(self, value: Any, default: float = 0.5) -> float:
        if isinstance(value, (int, float)):
            return max(0.0, min(1.0, float(value)))

        text = str(value or "").strip()
        if not text:
            return default

        try:
            return max(0.0, min(1.0, float(text)))
        except ValueError:
            pass

        matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)
        numbers = []
        for raw in matches:
            try:
                number = float(raw)
            except ValueError:
                continue
            if number > 1.0 and number <= 100.0:
                number /= 100.0
            if 0.0 <= number <= 1.0:
                numbers.append(number)
        for number in numbers:
            if number > 0.0:
                return number
        if numbers:
            return numbers[-1]
        return default

    def extract(self, agent_id: str, task: str, raw_response: str, use_rag: bool = False) -> List[Argument]:
        """
        Converts raw LLM response into a list of structured Argument objects with exponential backoff.
        
        Args:
            agent_id (str): ID of the agent providing the response.
            task (str): The task description.
            raw_response (str): The natural language response from the agent.
            use_rag (bool): Whether to use RAG for verification.
            
        Returns:
            List[Argument]: A list of extracted structured arguments.
        """
        prompt = f"""
Extract structured arguments from the following agent's response to a task.
Each argument must have a clear claim, supporting evidence from the text, and a confidence score (0.0 to 1.0).

Task: {task}
Agent Response: {raw_response}

Output the result as a JSON object with a key "arguments" containing a list of objects with:
- "claim": The core assertion.
- "evidence": The reasoning or facts supporting the claim.
- "confidence": A numerical value between 0.0 and 1.0.

JSON Output:
"""
        direct_arguments = self._parse_direct_payload(raw_response)
        if direct_arguments:
            structured_args = []
            for i, arg_data in enumerate(direct_arguments):
                claim = arg_data.get("claim", "No claim")
                evidence = arg_data.get("evidence", "No evidence")
                confidence = self._coerce_confidence(arg_data.get("confidence", 0.5))
                structured_args.append(
                    Argument(
                        id=f"arg_{agent_id}_{i}",
                        agent_id=agent_id,
                        claim=claim,
                        evidence=evidence,
                        confidence=confidence,
                        source="LLM-DirectJSON",
                    )
                )
            if structured_args:
                return structured_args

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=self.temperature,
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                
                # Robustly handle different JSON structures
                arguments_data = self._coerce_argument_list(data)
                
                structured_args = []
                for i, arg_data in enumerate(arguments_data):
                    if not isinstance(arg_data, dict): continue
                    
                    claim = arg_data.get("claim", "No claim")
                    evidence = arg_data.get("evidence", "No evidence")
                    confidence = self._coerce_confidence(arg_data.get("confidence", 0.5))
                    source = "LLM"
                    
                    if use_rag:
                        verification_prompt = f"Verify the following claim: {claim}\nProvide supporting or contradicting evidence."
                        v_res = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": verification_prompt}],
                            temperature=self.temperature,
                        )
                        evidence = f"{evidence} [RAG Verified: {v_res.choices[0].message.content[:100]}...]"
                        source = "RAG-Enhanced"
                    
                    arg = Argument(
                        id=f"arg_{agent_id}_{i}",
                        agent_id=agent_id,
                        claim=claim,
                        evidence=evidence,
                        confidence=confidence,
                        source=source
                    )
                    structured_args.append(arg)
                
                return structured_args
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                else:
                    print(f"Max retries reached for argument extraction.")
                    return []
        return []

from typing import Any
