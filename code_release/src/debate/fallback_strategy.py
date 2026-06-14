from typing import List, Dict, Any, Optional
from ..agents.llm_agent import LLMAgent
from ..argumentation.argument_extractor import Argument

# Evidence quality patterns (reuse from argument_quality_scorer logic)
import re
_EVIDENCE_PATTERNS = [
    r'\b\d{4}\b',           # years
    r'\d+\.?\d*\s*%',       # percentages
    r'\d+\.\d+',            # decimals
    r'\b(study|studies|research|data|according to|evidence|survey|report)\b',
    r'\b(for example|specifically|such as|including)\b',
]

def _evidence_score(text: str) -> float:
    """Quick heuristic evidence score for a proposal text."""
    text_lower = text.lower()
    hits = sum(1 for p in _EVIDENCE_PATTERNS if re.search(p, text_lower))
    return min(hits / len(_EVIDENCE_PATTERNS), 1.0)


class FallbackStrategy:
    """Intelligent fallback strategies when debate reaches deadlock."""

    def __init__(self, agent_reliability: Dict[str, float]):
        self.agent_reliability = agent_reliability

    def fallback_to_weighted_vote(
        self,
        agents: List[LLMAgent],
        proposals: Dict[str, str]
    ) -> str:
        """Select proposal based on agent reliability weighted voting."""
        if not proposals:
            return ""

        proposal_weights = {}
        for agent_id, proposal in proposals.items():
            normalized = self._normalize_text(proposal)
            reliability = self.agent_reliability.get(agent_id, 1.0)
            proposal_weights[normalized] = proposal_weights.get(normalized, 0) + reliability

        if proposal_weights:
            best_proposal = max(proposal_weights.items(), key=lambda x: x[1])[0]
            for proposal in proposals.values():
                if self._normalize_text(proposal) == best_proposal:
                    return proposal

        return list(proposals.values())[0] if proposals else ""

    def fallback_to_evidence_based(
        self,
        agents: List[LLMAgent],
        proposals: Dict[str, str],
        all_arguments: Optional[List[Argument]] = None
    ) -> str:
        """Phase 2.3: Select proposal with strongest evidence quality.

        Ranks proposals by:
        1. Evidence quality score of the proposal text itself
        2. Sum of argument confidence from that agent
        Falls back to weighted vote if evidence is unclear.
        """
        if not proposals:
            return ""

        # Score each proposal by evidence quality
        proposal_scores: Dict[str, float] = {}
        for agent_id, proposal in proposals.items():
            ev_score = _evidence_score(proposal)
            arg_score = 0.0
            if all_arguments:
                agent_args = [a for a in all_arguments if a.agent_id == agent_id]
                if agent_args:
                    arg_score = sum(a.confidence for a in agent_args) / len(agent_args)
            # Combined: 60% proposal evidence, 40% argument confidence
            proposal_scores[agent_id] = 0.6 * ev_score + 0.4 * arg_score

        max_score = max(proposal_scores.values()) if proposal_scores else 0.0
        # If no proposal has meaningful evidence, fall back to weighted vote
        if max_score < 0.1:
            return self.fallback_to_weighted_vote(agents, proposals)

        best_agent = max(proposal_scores, key=lambda aid: proposal_scores[aid])
        return proposals.get(best_agent, list(proposals.values())[0])

    def fallback_to_conservative(
        self,
        agents: List[LLMAgent],
        initial_proposals: Dict[str, str]
    ) -> str:
        """Return to initial majority opinion (conservative fallback)."""
        if not initial_proposals:
            return ""

        proposal_counts = {}
        for proposal in initial_proposals.values():
            normalized = self._normalize_text(proposal)
            proposal_counts[normalized] = proposal_counts.get(normalized, 0) + 1

        if proposal_counts:
            majority_proposal = max(proposal_counts.items(), key=lambda x: x[1])[0]
            for proposal in initial_proposals.values():
                if self._normalize_text(proposal) == majority_proposal:
                    return proposal

        return list(initial_proposals.values())[0]

    def select_fallback_strategy(
        self,
        deadlock_type: str,
        agents: List[LLMAgent],
        proposals: Dict[str, str],
        initial_proposals: Dict[str, str],
        all_arguments: Optional[List[Argument]] = None
    ) -> str:
        """Select appropriate fallback strategy based on deadlock type."""

        if deadlock_type == "cyclic_arguments":
            return self.fallback_to_evidence_based(agents, proposals, all_arguments)

        elif deadlock_type == "oscillating_proposals":
            return self.fallback_to_conservative(agents, initial_proposals)

        elif deadlock_type == "semantic_stagnation":
            return self.fallback_to_evidence_based(agents, proposals, all_arguments)

        else:
            return self.fallback_to_weighted_vote(agents, proposals)

    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text).strip().lower().replace("\n", " ").split())
