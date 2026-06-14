from typing import List, Dict, Tuple, Set, Optional
from .argument_extractor import Argument
from .qbaf_framework import QBAFramework
from .argument_quality_scorer import ArgumentQualityScorer
import json
import re
import os
from ..utils.openai_client import create_openai_client, get_default_model

class QBAFBuilder:
    """Builds a QBAF from a set of structured arguments using LLM for relation judging."""
    
    def __init__(self, model="gpt-4.1-mini", use_llm_relation: bool = True, use_quality_scoring: bool = True):
        self.client = create_openai_client()
        self.model = get_default_model(model)
        env_switch = os.getenv("EXP_RELATION_USE_LLM", "1").strip().lower() in {"1", "true", "yes", "on"}
        self.use_llm_relation = bool(use_llm_relation and env_switch)
        self.use_quality_scoring = use_quality_scoring
        self.quality_scorer = ArgumentQualityScorer() if use_quality_scoring else None
        self._stopwords: Set[str] = {
            "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with", "is", "are", "was", "were",
            "be", "as", "by", "that", "this", "it", "from", "at", "about", "into", "than", "then"
        }

    def build_from_debate(self, agent_arguments: Dict[str, List[Argument]]) -> QBAFramework:
        """
        agent_arguments: {agent_id: [Argument, ...]}
        Returns: A QBAFramework object.
        """
        all_args = []
        for agent_id, args in agent_arguments.items():
            all_args.extend(args)

        arg_ids = [arg.id for arg in all_args]
        initial_strengths = [arg.confidence for arg in all_args]

        # Compute quality scores if enabled
        quality_scores = None
        if self.use_quality_scoring and self.quality_scorer:
            quality_scores = {}
            for arg in all_args:
                arg_dict = {
                    "claim": arg.claim,
                    "premise": arg.evidence
                }
                quality_scores[arg.id] = self.quality_scorer.score_argument(arg_dict)

        relation_bundle = self._infer_relations(all_args)
        attacks = relation_bundle["attacks"]
        supports = relation_bundle["supports"]
        attack_weights = relation_bundle["attack_weights"]
        support_weights = relation_bundle["support_weights"]

        # Create QBAFramework with quality scores
        f = QBAFramework(
            arg_ids,
            initial_strengths,
            attacks,
            supports,
            semantics="DFQuAD_model",
            attack_weights=attack_weights,
            support_weights=support_weights,
            quality_scores=quality_scores,
        )
        return f

    def build_from_debate_with_metadata(self, agent_arguments: Dict[str, List[Argument]]) -> Tuple[QBAFramework, Dict[str, object]]:
        """Build a QBAF and return paper/artifact-friendly metadata for auditing."""
        all_args: List[Argument] = []
        for _, args in agent_arguments.items():
            all_args.extend(args)

        arg_ids = [arg.id for arg in all_args]
        initial_strengths = [arg.confidence for arg in all_args]

        quality_scores = None
        if self.use_quality_scoring and self.quality_scorer:
            quality_scores = {}
            for arg in all_args:
                arg_dict = {
                    "claim": arg.claim,
                    "premise": arg.evidence
                }
                quality_scores[arg.id] = self.quality_scorer.score_argument(arg_dict)

        relation_bundle = self._infer_relations(all_args)

        framework = QBAFramework(
            arg_ids,
            initial_strengths,
            relation_bundle["attacks"],
            relation_bundle["supports"],
            semantics="DFQuAD_model",
            attack_weights=relation_bundle["attack_weights"],
            support_weights=relation_bundle["support_weights"],
            quality_scores=quality_scores,
        )

        metadata = {
            "arguments": [arg.to_dict() for arg in all_args],
            "quality_scores": quality_scores or {},
            "relations": relation_bundle["relation_records"],
            "summary": {
                "num_arguments": len(all_args),
                "num_attacks": len(relation_bundle["attacks"]),
                "num_supports": len(relation_bundle["supports"]),
                "num_neutral_pairs": relation_bundle["neutral_pairs"],
                "llm_relation_enabled": bool(self.use_llm_relation),
                "quality_scoring_enabled": bool(self.use_quality_scoring),
            },
        }
        return framework, metadata

    def _infer_relations(self, all_args: List[Argument]) -> Dict[str, object]:
        """Infer inter-argument relations once and reuse them for both execution and tracing."""
        attacks: List[Tuple[str, str]] = []
        supports: List[Tuple[str, str]] = []
        attack_weights: Dict[Tuple[str, str], float] = {}
        support_weights: Dict[Tuple[str, str], float] = {}
        relation_records: List[Dict[str, object]] = []
        neutral_pairs = 0

        for i, arg_a in enumerate(all_args):
            for j, arg_b in enumerate(all_args):
                if i == j:
                    continue
                if arg_a.agent_id == arg_b.agent_id:
                    continue

                relation, confidence = self._judge_relation(arg_a, arg_b)
                if relation == "attack":
                    attacks.append((arg_a.id, arg_b.id))
                    attack_weights[(arg_a.id, arg_b.id)] = confidence
                elif relation == "support":
                    supports.append((arg_a.id, arg_b.id))
                    support_weights[(arg_a.id, arg_b.id)] = confidence
                else:
                    neutral_pairs += 1

                relation_records.append(
                    {
                        "type": relation,
                        "source": arg_a.id,
                        "target": arg_b.id,
                        "source_agent": arg_a.agent_id,
                        "target_agent": arg_b.agent_id,
                        "confidence": float(confidence),
                    }
                )

        return {
            "attacks": attacks,
            "supports": supports,
            "attack_weights": attack_weights,
            "support_weights": support_weights,
            "relation_records": relation_records,
            "neutral_pairs": neutral_pairs,
        }

    def _tokenize(self, text: str) -> Set[str]:
        tokens = re.findall(r"[a-zA-Z0-9_]+", str(text).lower())
        return {tok for tok in tokens if tok not in self._stopwords and len(tok) > 1}

    def _heuristic_relation(self, arg_a: Argument, arg_b: Argument) -> Tuple[str, float, bool]:
        text_a = f"{arg_a.claim} {arg_a.evidence}"
        text_b = f"{arg_b.claim} {arg_b.evidence}"

        tok_a = self._tokenize(text_a)
        tok_b = self._tokenize(text_b)

        overlap = 0.0
        if tok_a and tok_b:
            overlap = len(tok_a & tok_b) / max(1, len(tok_a | tok_b))

        ta = text_a.lower()
        tb = text_b.lower()

        contradiction_markers = ["not", "no", "never", "false", "incorrect", "wrong", "cannot", "can't"]
        support_markers = ["therefore", "thus", "consistent", "supports", "evidence", "confirm", "agrees"]

        has_negation_conflict = any(m in ta for m in contradiction_markers) != any(m in tb for m in contradiction_markers)
        has_support_signal = any(m in ta for m in support_markers) and overlap > 0.08

        if overlap < 0.04 and not has_negation_conflict:
            return "neutral", 0.85, True
        if has_negation_conflict and overlap > 0.06:
            return "attack", min(0.9, 0.55 + overlap), True
        if has_support_signal:
            return "support", min(0.9, 0.55 + overlap), True

        return "neutral", max(0.35, min(0.65, 0.3 + overlap)), False

    def _coerce_confidence(self, value: object, default: float = 0.5) -> float:
        if isinstance(value, (int, float)):
            return max(0.05, min(1.0, float(value)))

        text = str(value or "").strip()
        if not text:
            return default

        try:
            return max(0.05, min(1.0, float(text)))
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
                numbers.append(max(0.05, min(1.0, number)))
        for number in numbers:
            if number > 0.05:
                return number
        if numbers:
            return numbers[-1]
        return default

    def _parse_relation_payload(self, content: str) -> Dict[str, object]:
        text = str(content or "").strip()
        candidates = [text]
        candidates.extend(
            chunk.strip()
            for chunk in re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
            if chunk.strip()
        )
        if "{" in text and "}" in text:
            candidates.append(text[text.find("{") : text.rfind("}") + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, list):
                payload = next((item for item in payload if isinstance(item, dict)), {})
            if isinstance(payload, dict):
                return payload
        return {}

    def _judge_relation_llm(self, arg_a: Argument, arg_b: Argument) -> Tuple[str, float]:
        prompt = f"""
Determine the logical relation between two arguments.

Argument A:
Claim: {arg_a.claim}
Evidence: {arg_a.evidence}

Argument B:
Claim: {arg_b.claim}
Evidence: {arg_b.evidence}

Return JSON only:
{{
  "relation": "attack" | "support" | "neutral",
  "confidence": 0.0_to_1.0
}}
"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = self._parse_relation_payload(content)

        relation = str(data.get("relation", "neutral")).strip().lower()
        confidence = self._coerce_confidence(data.get("confidence", 0.5))

        if relation not in {"attack", "support", "neutral"}:
            relation = "neutral"
        return relation, confidence

    def _judge_relation(self, arg_a: Argument, arg_b: Argument) -> Tuple[str, float]:
        """Hybrid relation inference: heuristic pre-filter + LLM confidence."""
        heuristic_relation, heuristic_conf, decisive = self._heuristic_relation(arg_a, arg_b)
        if decisive or not self.use_llm_relation:
            return heuristic_relation, heuristic_conf

        try:
            llm_relation, llm_conf = self._judge_relation_llm(arg_a, arg_b)
            blended_conf = max(0.05, min(1.0, 0.7 * llm_conf + 0.3 * heuristic_conf))
            return llm_relation, blended_conf
        except Exception as e:
            print(f"Error judging relation: {e}")
            return heuristic_relation, heuristic_conf

    def evaluate(self, framework: QBAFramework, max_iter=100, tol=1e-4) -> Dict[str, float]:
        """Delegates to framework.evaluate() — single implementation."""
        return framework.evaluate(max_iter=max_iter, tolerance=tol)
