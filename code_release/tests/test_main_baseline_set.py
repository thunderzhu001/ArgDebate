import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.main_baseline_set import (
    configured_methods,
    empty_correctness_dict,
    estimate_tokens,
    extract_method_prediction,
    select_consensus_response,
    status_counts,
    validate_extra_baselines,
)


def test_configured_methods_keeps_main_baseline_order():
    assert configured_methods(["self_consistency", "scalar_transcript_judge"]) == [
        "majority_vote",
        "vanilla_debate",
        "weighted_vote",
        "self_consistency",
        "scalar_transcript_judge",
        "arg_debate",
    ]


def test_invalid_extra_baseline_is_rejected():
    try:
        validate_extra_baselines(["router_only"])
    except ValueError as exc:
        assert "router_only" in str(exc)
    else:
        raise AssertionError("invalid baseline was accepted")


def test_extract_method_prediction_prefers_formal_normalized_when_enabled():
    result = {
        "formal_normalized_final_answer": "Yes",
        "final_answer": "Conditional answer",
        "result": {"agent_0": "No"},
    }
    assert extract_method_prediction(result, "arg_debate") == "Yes"
    assert (
        extract_method_prediction(result, "arg_debate", prefer_formal_normalized=False)
        == "Conditional answer"
    )


def test_consensus_response_uses_largest_normalized_bucket():
    response = {"b": "  Same answer  ", "a": "same   answer", "c": "other"}
    assert select_consensus_response(response).strip().lower() == "same answer"


def test_lightweight_accounting_helpers():
    methods = ["majority_vote", "arg_debate"]
    assert empty_correctness_dict(methods) == {"majority_vote": 0, "arg_debate": 0}
    assert status_counts({"arg_debate": ["fallback", "fallback", "resolved"]}) == {
        "arg_debate": {"fallback": 2, "resolved": 1}
    }
    assert estimate_tokens("abcd") == 1


if __name__ == "__main__":
    test_configured_methods_keeps_main_baseline_order()
    test_invalid_extra_baseline_is_rejected()
    test_extract_method_prediction_prefers_formal_normalized_when_enabled()
    test_consensus_response_uses_largest_normalized_bucket()
    test_lightweight_accounting_helpers()
    print("main baseline set tests passed")
