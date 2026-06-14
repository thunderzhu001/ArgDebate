import sys
import os
# Add src to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agents.llm_agent import LLMAgent
from src.debate.debate_manager import ArgDebateManager

def test_2agent_resolve():
    # 1. Initialize agents
    # Agent 1: Pro-nuclear energy
    # Agent 2: Anti-nuclear energy
    agent1 = LLMAgent("agent_1", model="gpt-4o-mini")
    agent2 = LLMAgent("agent_2", model="gpt-4o-mini")
    
    # Set initial beliefs to force a conflict
    agent1.beliefs = ["Nuclear energy is the most efficient carbon-free energy source."]
    agent2.beliefs = ["Nuclear energy is too dangerous and produces radioactive waste."]
    
    # 2. Define task
    task = "Should we build more nuclear power plants to combat climate change?"
    
    # 3. Initialize manager
    manager = ArgDebateManager([agent1, agent2], model="gpt-4o-mini")
    
    # 4. Run resolution
    print(f"Starting ArgDebate for task: {task}")
    result = manager.resolve(task)
    
    # 5. Output results
    print("\n--- Final Result ---")
    print(f"Status: {result['status']}")
    if 'round' in result:
        print(f"Resolved in round: {result['round']}")
    
    for agent_id, proposal in result['result'].items():
        print(f"\nAgent {agent_id} Final Proposal:")
        print(proposal)
        
    if 'audit_trail' in result:
        print("\nAudit Trail (Argument Strengths):")
        for arg_id, strength in result['audit_trail'].items():
            print(f"{arg_id}: {strength:.4f}")

if __name__ == "__main__":
    test_2agent_resolve()
