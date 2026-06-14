from typing import List, Dict, Any, Optional
import numpy as np
from collections import defaultdict

class SemanticDeadlockDetector:
    """Detects various types of semantic deadlocks in multi-agent debates."""

    def __init__(self, similarity_threshold: float = 0.95, cycle_lookback: int = 3):
        self.similarity_threshold = similarity_threshold
        self.cycle_lookback = cycle_lookback

    def detect(self, rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Main detection method that checks for all deadlock types."""
        if len(rounds) < 2:
            return {"deadlock_detected": False, "deadlock_type": None}

        cyclic = self._detect_cyclic_arguments(rounds)
        if cyclic["detected"]:
            return {
                "deadlock_detected": True,
                "deadlock_type": "cyclic_arguments",
                "confidence": cyclic["confidence"],
                "evidence": cyclic["evidence"]
            }

        oscillating = self._detect_oscillating_proposals(rounds)
        if oscillating["detected"]:
            return {
                "deadlock_detected": True,
                "deadlock_type": "oscillating_proposals",
                "confidence": oscillating["confidence"],
                "evidence": oscillating["evidence"]
            }

        stagnation = self._detect_semantic_stagnation(rounds)
        if stagnation["detected"]:
            return {
                "deadlock_detected": True,
                "deadlock_type": "semantic_stagnation",
                "confidence": stagnation["confidence"],
                "evidence": stagnation["evidence"]
            }

        return {"deadlock_detected": False, "deadlock_type": None}

    def _detect_cyclic_arguments(self, rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect if arguments form cycles (A attacks B, B attacks C, C attacks A)."""
        if len(rounds) < self.cycle_lookback:
            return {"detected": False}

        recent_rounds = rounds[-self.cycle_lookback:]
        attack_graph = defaultdict(set)

        for round_data in recent_rounds:
            qbaf_data = round_data.get("qbaf_data", {})
            relations = qbaf_data.get("relations", [])

            for rel in relations:
                if rel.get("type") == "attack":
                    source = rel.get("source")
                    target = rel.get("target")
                    if source and target:
                        attack_graph[source].add(target)

        cycles = self._find_cycles(attack_graph)

        if cycles:
            return {
                "detected": True,
                "confidence": min(0.9, 0.6 + 0.1 * len(cycles)),
                "evidence": {
                    "cycles_found": len(cycles),
                    "example_cycle": cycles[0] if cycles else None,
                    "rounds_analyzed": len(recent_rounds)
                }
            }

        return {"detected": False}

    def _find_cycles(self, graph: Dict[str, set]) -> List[List[str]]:
        """Find cycles in directed graph using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []

        def dfs(node):
            if node in rec_stack:
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:])
                return
            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                dfs(neighbor)

            path.pop()
            rec_stack.remove(node)

        for node in graph:
            if node not in visited:
                dfs(node)

        return cycles[:5]  # Return at most 5 cycles

    def _detect_oscillating_proposals(self, rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect if proposals oscillate between two options."""
        if len(rounds) < 3:
            return {"detected": False}

        recent_proposals = []
        for round_data in rounds[-4:]:
            proposals = round_data.get("proposals", {})
            # Track agent-proposal pairs to detect oscillation
            agent_proposal_pairs = frozenset((agent_id, self._normalize_text(proposal))
                                            for agent_id, proposal in proposals.items())
            recent_proposals.append(agent_proposal_pairs)

        if len(recent_proposals) < 3:
            return {"detected": False}

        # Check if proposals alternate between two states
        oscillation_count = 0
        for i in range(len(recent_proposals) - 2):
            if recent_proposals[i] == recent_proposals[i + 2] and recent_proposals[i] != recent_proposals[i + 1]:
                oscillation_count += 1

        if oscillation_count >= 2:
            return {
                "detected": True,
                "confidence": 0.85,
                "evidence": {
                    "oscillation_count": oscillation_count,
                    "rounds_analyzed": len(recent_proposals)
                }
            }

        return {"detected": False}

    def _detect_semantic_stagnation(self, rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect if argument semantics have stagnated (high similarity across rounds)."""
        if len(rounds) < 2:
            return {"detected": False}

        recent_rounds = rounds[-2:]

        # Extract argument claims from each round
        claims_by_round = []
        for round_data in recent_rounds:
            qbaf_data = round_data.get("qbaf_data", {})
            arguments = qbaf_data.get("arguments", [])
            claims = [arg.get("claim", "") for arg in arguments]
            claims_by_round.append(claims)

        if len(claims_by_round) < 2 or not all(claims_by_round):
            return {"detected": False}

        # Simple token-based similarity
        similarity = self._compute_text_similarity(claims_by_round[0], claims_by_round[1])

        if similarity > self.similarity_threshold:
            return {
                "detected": True,
                "confidence": min(0.95, similarity),
                "evidence": {
                    "similarity_score": similarity,
                    "threshold": self.similarity_threshold,
                    "rounds_compared": 2
                }
            }

        return {"detected": False}

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        return " ".join(str(text).strip().lower().split())

    def _compute_text_similarity(self, texts1: List[str], texts2: List[str]) -> float:
        """Compute similarity between two sets of texts using token overlap."""
        if not texts1 or not texts2:
            return 0.0

        tokens1 = set()
        for text in texts1:
            tokens1.update(self._normalize_text(text).split())

        tokens2 = set()
        for text in texts2:
            tokens2.update(self._normalize_text(text).split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0
