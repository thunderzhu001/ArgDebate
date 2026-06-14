"""
Argument Quality Scorer

Evaluates argument quality based on:
1. Evidence specificity (citations, data, examples)
2. Logical coherence (no contradictions)
3. Relevance to question/claim

Used to weight QBAF argument strengths for better evaluation.
"""

import re
from typing import Dict, List


class ArgumentQualityScorer:
    """Scores argument quality on 0-1 scale."""

    def __init__(self):
        # Evidence indicators
        self.evidence_patterns = [
            r'\d{4}',  # Years
            r'\d+%',  # Percentages
            r'\d+\.\d+',  # Decimals
            r'study|research|report|survey|data|statistics',
            r'according to|based on|evidence shows|research indicates',
            r'for example|for instance|specifically|namely',
        ]

        # Coherence red flags
        self.incoherence_patterns = [
            r'but.*however',  # Double contradiction
            r'always.*never',  # Absolute contradiction
            r'all.*none',  # Universal contradiction
        ]

    def score_argument(self, argument: Dict) -> float:
        """
        Score argument quality 0-1.

        Args:
            argument: Dict with 'claim' and 'premise' keys

        Returns:
            Quality score 0-1 (higher is better)
        """
        text = f"{argument.get('claim', '')} {argument.get('premise', '')}"

        evidence_score = self._score_evidence(text)
        coherence_score = self._score_coherence(text)
        length_score = self._score_length(text)

        # Weighted average (evidence most important)
        return (evidence_score * 0.5 + coherence_score * 0.3 + length_score * 0.2)

    def _score_evidence(self, text: str) -> float:
        """Score based on evidence specificity."""
        if not text:
            return 0.0

        text_lower = text.lower()
        matches = 0

        for pattern in self.evidence_patterns:
            if re.search(pattern, text_lower):
                matches += 1

        # Normalize to 0-1 (cap at 4 evidence indicators)
        return min(matches / 4.0, 1.0)

    def _score_coherence(self, text: str) -> float:
        """Score based on logical coherence (penalize contradictions)."""
        if not text:
            return 0.0

        text_lower = text.lower()

        # Check for incoherence patterns
        for pattern in self.incoherence_patterns:
            if re.search(pattern, text_lower):
                return 0.5  # Penalty for contradiction

        return 1.0  # No obvious contradictions

    def _score_length(self, text: str) -> float:
        """Score based on argument length (prefer substantive arguments)."""
        if not text:
            return 0.0

        words = len(text.split())

        # Optimal range: 20-100 words
        if words < 10:
            return 0.3  # Too short
        elif words < 20:
            return 0.6
        elif words <= 100:
            return 1.0  # Optimal
        elif words <= 150:
            return 0.8
        else:
            return 0.6  # Too verbose

    def score_arguments(self, arguments: List[Dict]) -> List[float]:
        """Score multiple arguments."""
        return [self.score_argument(arg) for arg in arguments]


def test_scorer():
    """Test argument quality scorer."""
    scorer = ArgumentQualityScorer()

    # High quality: specific evidence
    arg1 = {
        "claim": "Climate change is accelerating",
        "premise": "According to NASA data from 2023, global temperatures increased by 1.2°C since 1880. Research shows 97% of climate scientists agree on human-caused warming."
    }

    # Medium quality: some evidence
    arg2 = {
        "claim": "Exercise improves health",
        "premise": "Studies show that regular exercise helps. For example, people who exercise feel better."
    }

    # Low quality: no evidence
    arg3 = {
        "claim": "This is true",
        "premise": "Because it is."
    }

    # Incoherent: contradiction
    arg4 = {
        "claim": "All cats are mammals",
        "premise": "But however, none of them are animals."
    }

    print("Argument Quality Scores:")
    print(f"High quality (specific evidence): {scorer.score_argument(arg1):.2f}")
    print(f"Medium quality (some evidence): {scorer.score_argument(arg2):.2f}")
    print(f"Low quality (no evidence): {scorer.score_argument(arg3):.2f}")
    print(f"Incoherent (contradiction): {scorer.score_argument(arg4):.2f}")


if __name__ == "__main__":
    test_scorer()
