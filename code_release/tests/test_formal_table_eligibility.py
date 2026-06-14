import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.formal_table_eligibility import (
    method_formal_table_eligibility,
    parse_strategyqa_prediction,
    score_strategyqa_boolean,
    summarize_formal_table_eligibility,
)


def test_strategyqa_boolean_scoring_is_deterministic():
    record = score_strategyqa_boolean("Final answer: yes.", True, method="arg_debate")
    assert record["correct"] is True
    assert record["formal_table_eligible"] is True
    assert record["scoring_policy"] == "deterministic_boolean"


def test_strategyqa_ambiguous_boolean_prediction_is_ineligible():
    parsed, reason = parse_strategyqa_prediction("It may be yes, but no is also plausible.")
    assert parsed is None
    assert reason == "ambiguous_boolean_tokens"
    record = score_strategyqa_boolean("It may be yes, but no is also plausible.", True, method="majority_vote")
    assert record["correct"] is False
    assert record["formal_table_eligible"] is False


def test_strategyqa_eligibility_ignores_llm_audit_correctness():
    scoring = score_strategyqa_boolean("No.", False, method="arg_debate")
    judge = {
        "correct": False,
        "judge_fallback_used": True,
        "judge_error": "all_judge_attempts_failed",
        "formal_table_eligible": False,
    }
    eligibility = method_formal_table_eligibility(
        dataset="StrategyQA",
        method="arg_debate",
        method_result={"status": "resolved"},
        scoring_record=scoring,
        judge_record=judge,
    )
    assert scoring["correct"] is True
    assert eligibility["formal_table_eligible"] is True


def test_truthfulqa_requires_formal_judge_eligibility():
    eligibility = method_formal_table_eligibility(
        dataset="TruthfulQA",
        method="arg_debate",
        method_result={"status": "resolved"},
        judge_record={
            "correct": True,
            "judge_fallback_used": True,
            "judge_error": "all_judge_attempts_failed",
            "formal_table_eligible": False,
        },
    )
    assert eligibility["formal_table_eligible"] is False
    assert "all_judge_attempts_failed" in eligibility["reasons"]


def test_summary_counts_ineligible_methods():
    rows = [
        {
            "formal_table_eligibility": {
                "arg_debate": {
                    "formal_table_eligible": True,
                    "reasons": [],
                },
                "majority_vote": {
                    "formal_table_eligible": False,
                    "reasons": ["deterministic_scoring_ambiguous_boolean_tokens"],
                },
            }
        }
    ]
    summary = summarize_formal_table_eligibility(rows, ["arg_debate", "majority_vote"])
    assert summary["eligible_total"] == 1
    assert summary["ineligible_total"] == 1
    assert summary["method_counts"]["majority_vote"]["reason_counts"][
        "deterministic_scoring_ambiguous_boolean_tokens"
    ] == 1


if __name__ == "__main__":
    test_strategyqa_boolean_scoring_is_deterministic()
    test_strategyqa_ambiguous_boolean_prediction_is_ineligible()
    test_strategyqa_eligibility_ignores_llm_audit_correctness()
    test_truthfulqa_requires_formal_judge_eligibility()
    test_summary_counts_ineligible_methods()
    print("formal table eligibility tests passed")
