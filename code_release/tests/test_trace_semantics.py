import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.analyze_trace_artifacts import analyze
from experiments.trace_summary import summarize_argdebate_traces


def test_routed_skip_semantics():
    payload = [
        {
            "arg_debate": {
                "status": "routed_skip",
                "reason": "routing_gate_skip",
                "final_answer_source": "routing_low_disagreement",
                "meta": {
                    "debate_routed": False,
                    "route_reason": ["low_disagreement_and_density"],
                    "termination": {
                        "type": "routed_skip",
                        "round": 0,
                        "strategy": "routing_gate_selector",
                    },
                },
            }
        }
    ]
    summary = summarize_argdebate_traces(payload)
    assert summary["status_counts"]["routed_skip"] == 1
    assert summary["reason_counts"]["routing_gate_skip"] == 1
    assert summary["termination_type_counts"]["routed_skip"] == 1


def test_trace_audit_keeps_routed_skip_distinct():
    payload = [
        {
            "arg_debate": {
                "status": "routed_skip",
                "reason": "routing_gate_skip",
                "final_answer_source": "routing_low_disagreement",
                "meta": {
                    "debate_routed": False,
                    "route_reason": ["low_disagreement_and_density"],
                    "termination": {"type": "routed_skip"},
                },
            }
        }
    ]
    summary = analyze(payload)
    assert summary["status_counts"]["routed_skip"] == 1
    assert summary["termination_type_counts"]["routed_skip"] == 1
    assert summary["route_reason_counts"]["low_disagreement_and_density"] == 1


def test_resolved_reason_is_explicit():
    payload = [
        {
            "arg_debate": {
                "status": "resolved",
                "reason": "resolved_after_debate",
                "final_answer_source": "agent_weighted:agent_0",
                "meta": {
                    "debate_routed": True,
                    "route_reason": ["routing_disabled"],
                    "termination": {"type": "resolved", "round": 1},
                },
            }
        }
    ]
    summary = summarize_argdebate_traces(payload)
    assert summary["status_counts"]["resolved"] == 1
    assert summary["reason_counts"]["resolved_after_debate"] == 1
    assert summary["termination_type_counts"]["resolved"] == 1


if __name__ == "__main__":
    test_routed_skip_semantics()
    test_trace_audit_keeps_routed_skip_distinct()
    test_resolved_reason_is_explicit()
    print("trace semantics tests passed")
