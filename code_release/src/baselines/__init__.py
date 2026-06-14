# Baselines module
from .majority_vote import MajorityVote
from .scalar_transcript_judge import ScalarTranscriptJudge
from .self_consistency import SelfConsistency
from .vanilla_debate import VanillaDebate
from .weighted_vote import WeightedVote

__all__ = [
    "MajorityVote",
    "VanillaDebate",
    "WeightedVote",
    "SelfConsistency",
    "ScalarTranscriptJudge",
]
