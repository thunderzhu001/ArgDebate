import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.llm_agent import LLMAgent
from src.debate.debate_manager import ArgDebateManager
from src.debate.semantic_deadlock_detector import SemanticDeadlockDetector
from src.debate.fallback_strategy import FallbackStrategy
from src.argumentation.argument_extractor import Argument
from src.argumentation.argument_extractor import ArgumentExtractor
from src.argumentation.qbaf_builder import QBAFBuilder
from src.debate.conflict_detector import ConflictDetector

def test_deadlock_detection():
    """Test semantic deadlock detector with mock data."""
    print("Testing Semantic Deadlock Detector...")

    detector = SemanticDeadlockDetector()

    # Test 1: Cyclic arguments
    rounds_cyclic = [
        {
            "round": 1,
            "qbaf_data": {
                "relations": [
                    {"type": "attack", "source": "arg1", "target": "arg2"},
                    {"type": "attack", "source": "arg2", "target": "arg3"},
                    {"type": "attack", "source": "arg3", "target": "arg1"}
                ]
            }
        },
        {
            "round": 2,
            "qbaf_data": {
                "relations": [
                    {"type": "attack", "source": "arg1", "target": "arg2"},
                    {"type": "attack", "source": "arg2", "target": "arg3"},
                    {"type": "attack", "source": "arg3", "target": "arg1"}
                ]
            }
        },
        {
            "round": 3,
            "qbaf_data": {
                "relations": [
                    {"type": "attack", "source": "arg1", "target": "arg2"},
                    {"type": "attack", "source": "arg2", "target": "arg3"},
                    {"type": "attack", "source": "arg3", "target": "arg1"}
                ]
            }
        }
    ]

    result = detector.detect(rounds_cyclic)
    assert result["deadlock_detected"], "Should detect cyclic arguments"
    assert result["deadlock_type"] == "cyclic_arguments"
    print("✓ Cyclic arguments detection passed")

    # Test 2: Oscillating proposals
    rounds_oscillating = [
        {"round": 1, "proposals": {"agent1": "Answer A", "agent2": "Answer B"}, "qbaf_data": {}},
        {"round": 2, "proposals": {"agent1": "Answer B", "agent2": "Answer A"}, "qbaf_data": {}},
        {"round": 3, "proposals": {"agent1": "Answer A", "agent2": "Answer B"}, "qbaf_data": {}},
        {"round": 4, "proposals": {"agent1": "Answer B", "agent2": "Answer A"}, "qbaf_data": {}}
    ]

    result = detector.detect(rounds_oscillating)
    assert result["deadlock_detected"], "Should detect oscillating proposals"
    assert result["deadlock_type"] == "oscillating_proposals"
    print("✓ Oscillating proposals detection passed")

    # Test 3: Semantic stagnation
    rounds_stagnation = [
        {
            "round": 1,
            "qbaf_data": {
                "arguments": [
                    {"claim": "The sky is blue because of Rayleigh scattering"},
                    {"claim": "Water appears blue due to light absorption"}
                ]
            }
        },
        {
            "round": 2,
            "qbaf_data": {
                "arguments": [
                    {"claim": "The sky is blue because of Rayleigh scattering"},
                    {"claim": "Water appears blue due to light absorption"}
                ]
            }
        }
    ]

    result = detector.detect(rounds_stagnation)
    assert result["deadlock_detected"], "Should detect semantic stagnation"
    assert result["deadlock_type"] == "semantic_stagnation"
    print("✓ Semantic stagnation detection passed")

    print("\nAll deadlock detection tests passed!\n")

def test_temperature_escalation():
    """Test temperature escalation mechanism."""
    print("Testing Temperature Escalation...")

    # Mock the client to avoid API key requirement
    import unittest.mock as mock
    with mock.patch('src.agents.llm_agent.create_openai_client'), \
         mock.patch('src.argumentation.argument_extractor.create_openai_client'):
        agent = LLMAgent(agent_id="test_agent", proposal_temperature=0.2, argument_temperature=0.3)

        assert agent.current_proposal_temperature == 0.2
        assert agent.current_argument_temperature == 0.3

        agent.escalate_temperature(increment=0.2)
        assert agent.current_proposal_temperature == 0.4
        assert agent.current_argument_temperature == 0.5
        print("✓ First escalation passed")

        agent.escalate_temperature(increment=0.2)
        assert abs(agent.current_proposal_temperature - 0.6) < 1e-9
        assert abs(agent.current_argument_temperature - 0.7) < 1e-9
        print("✓ Second escalation passed")

        agent.escalate_temperature(increment=0.5, max_temp=0.9)
        assert abs(agent.current_proposal_temperature - 0.9) < 1e-9
        assert abs(agent.current_argument_temperature - 0.9) < 1e-9
        print("✓ Max temperature cap passed")

        agent.reset_temperature()
        assert agent.current_proposal_temperature == 0.2
        assert agent.current_argument_temperature == 0.3
        print("✓ Temperature reset passed")

    print("\nAll temperature escalation tests passed!\n")

def test_fallback_strategies():
    """Test fallback strategy selection."""
    print("Testing Fallback Strategies...")

    agent_reliability = {"agent1": 0.8, "agent2": 0.6, "agent3": 0.9}
    fallback = FallbackStrategy(agent_reliability)

    # Test weighted vote (no need to create actual agents)
    import unittest.mock as mock
    with mock.patch('src.agents.llm_agent.create_openai_client'), \
         mock.patch('src.argumentation.argument_extractor.create_openai_client'):
        agents = [
            LLMAgent("agent1"),
            LLMAgent("agent2"),
            LLMAgent("agent3")
        ]
        proposals = {
            "agent1": "Answer A",
            "agent2": "Answer B",
            "agent3": "Answer A"
        }

        result = fallback.fallback_to_weighted_vote(agents, proposals)
        assert result == "Answer A", f"Expected 'Answer A', got '{result}'"
        print("✓ Weighted vote fallback passed")

        # Test conservative fallback
        initial_proposals = {
            "agent1": "Initial A",
            "agent2": "Initial A",
            "agent3": "Initial B"
        }

        result = fallback.fallback_to_conservative(agents, initial_proposals)
        assert result == "Initial A", f"Expected 'Initial A', got '{result}'"
        print("✓ Conservative fallback passed")

    print("\nAll fallback strategy tests passed!\n")

def test_integration():
    """Test integration of deadlock mitigation in debate manager."""
    print("Testing Integration with ArgDebateManager...")

    # Create agents with low temperature
    import unittest.mock as mock
    with mock.patch('src.agents.llm_agent.create_openai_client'), \
         mock.patch('src.argumentation.argument_extractor.create_openai_client'), \
         mock.patch('src.argumentation.qbaf_builder.create_openai_client'), \
         mock.patch('src.debate.conflict_detector.create_openai_client'):
        agents = [
            LLMAgent(f"agent{i}", proposal_temperature=0.2, argument_temperature=0.3)
            for i in range(3)
        ]

        config = {
            "max_rounds": 3,
            "enable_deadlock_mitigation": True,
            "deadlock_temperature_increment": 0.2,
            "enable_early_stop": True,
            "early_stop_patience": 2
        }

        manager = ArgDebateManager(agents, config=config)

        # Verify components are initialized
        assert manager.deadlock_detector is not None
        assert manager.fallback_strategy is not None
        assert manager.enable_deadlock_mitigation == True
        assert manager.deadlock_temperature_increment == 0.2

        print("✓ ArgDebateManager initialization passed")
        print("✓ Deadlock mitigation components integrated")

    print("\nIntegration test passed!\n")

def test_qbaf_trace_metadata():
    """Test that QBAF metadata now contains explicit relation records for auditing."""
    print("Testing QBAF Trace Metadata...")

    import unittest.mock as mock
    with mock.patch('src.argumentation.qbaf_builder.create_openai_client'):
        builder = QBAFBuilder(use_llm_relation=False, use_quality_scoring=False)

        agent_arguments = {
            "agent1": [
                Argument("arg_agent1_0", "agent1", "The claim is true", "Research data supports it", 0.8),
            ],
            "agent2": [
                Argument("arg_agent2_0", "agent2", "The claim is not true", "Evidence contradicts it", 0.7),
            ],
        }

        _, metadata = builder.build_from_debate_with_metadata(agent_arguments)
        assert metadata["summary"]["num_arguments"] == 2
        assert len(metadata["relations"]) >= 2, "Expected directional relation records between cross-agent arguments"
        assert metadata["summary"]["num_attacks"] >= 1, "Expected at least one attack relation from heuristic judging"
        print("✓ QBAF trace metadata passed")

    print("\nQBAF trace metadata test passed!\n")


def test_argument_extractor_direct_json():
    """Test that direct JSON outputs no longer trigger a second extraction-only LLM hop."""
    print("Testing Direct JSON Argument Extraction...")

    import unittest.mock as mock
    with mock.patch('src.argumentation.argument_extractor.create_openai_client'):
        extractor = ArgumentExtractor()
        raw_response = """
        {
          "arguments": [
            {
              "claim": "The Earth orbits the Sun.",
              "evidence": "Heliocentric astronomy and modern observations support this.",
              "confidence": 0.91
            }
          ]
        }
        """
        arguments = extractor.extract("agent_0", "Astronomy question", raw_response)
        assert len(arguments) == 1
        assert arguments[0].claim == "The Earth orbits the Sun."
        assert arguments[0].source == "LLM-DirectJSON"
        print("✓ Direct JSON argument extraction passed")

    print("\nDirect JSON argument extraction test passed!\n")


def test_conflict_detector_heuristic_mode():
    """Test that heuristic-only conflict detection can be used for lightweight validation."""
    print("Testing Heuristic Conflict Detection Mode...")

    import unittest.mock as mock
    with mock.patch('src.debate.conflict_detector.create_openai_client'):
        detector = ConflictDetector(use_llm=False)
        conflicts = detector.detect({
            "agent1": "The statement is correct and supported by evidence.",
            "agent2": "The statement is not correct and is contradicted by evidence.",
        })
        assert conflicts, "Expected heuristic mode to detect a polarity conflict"
        print("✓ Heuristic conflict detection passed")

    print("\nHeuristic conflict detection test passed!\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Running Deadlock Mitigation Tests")
    print("=" * 60 + "\n")

    try:
        test_deadlock_detection()
        test_temperature_escalation()
        test_fallback_strategies()
        test_integration()
        test_qbaf_trace_metadata()
        test_argument_extractor_direct_json()
        test_conflict_detector_heuristic_mode()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
