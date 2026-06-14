from typing import List, Dict, Any, Optional
from src.debate.debate_manager import ArgDebateManager
from src.agents.llm_agent import LLMAgent
from src.argumentation.qbaf_builder import QBAFBuilder
from src.argumentation.qbaf_framework import QBAFramework
from src.argumentation.argument_extractor import Argument


class SimpleQBAFBuilder(QBAFBuilder):
    """Backward-compatibility alias used by legacy strategy_qa scripts.

    Behaves identically to QBAFBuilder — DF-QuAD evaluation is handled by
    the framework object itself (called directly from ArgDebateManager).
    """
    pass


class NoDFQuADManager(ArgDebateManager):
    """Ablation: replaces DF-QuAD evaluation with simple initial-strength passthrough."""

    def resolve(self, task: str) -> Dict[str, Any]:
        original_build = self.qbaf_builder.build_from_debate
        original_build_with_metadata = self.qbaf_builder.build_from_debate_with_metadata

        def _build_no_dfquad(agent_arguments):
            framework = original_build(agent_arguments)
            framework.evaluate = lambda **kw: {
                aid: framework.initial_strength(aid) for aid in framework.arguments
            }
            return framework

        def _build_no_dfquad_with_metadata(agent_arguments):
            framework, metadata = original_build_with_metadata(agent_arguments)
            framework.evaluate = lambda **kw: {
                aid: framework.initial_strength(aid) for aid in framework.arguments
            }
            metadata.setdefault("summary", {})
            metadata["summary"]["dfquad_enabled"] = False
            metadata["summary"]["evaluation_semantics"] = "initial_strength_passthrough"
            return framework, metadata

        self.qbaf_builder.build_from_debate = _build_no_dfquad
        self.qbaf_builder.build_from_debate_with_metadata = _build_no_dfquad_with_metadata
        try:
            return super().resolve(task)
        finally:
            self.qbaf_builder.build_from_debate = original_build
            self.qbaf_builder.build_from_debate_with_metadata = original_build_with_metadata


class NoBeliefUpdateManager(ArgDebateManager):
    """Ablation: skips belief update step — agents never update beliefs."""

    def resolve(self, task: str) -> Dict[str, Any]:
        # Disable belief updates for all agents
        for agent in self.agents:
            agent.update_beliefs = lambda *a, **kw: None
        return super().resolve(task)


class NoQualityAttenuationManager(ArgDebateManager):
    """Ablation: keep DF-QuAD and relations, but remove argument-quality attenuation."""

    def __init__(self, agents, model: str = "deepseek-v3.2", config: Optional[Dict[str, Any]] = None):
        cfg = dict(config or {})
        cfg["use_quality_scoring"] = False
        super().__init__(agents, model=model, config=cfg)


class NoReliabilityManager(ArgDebateManager):
    """Ablation: disable adaptive agent reliability updates and weighting effects."""

    def __init__(self, agents, model: str = "deepseek-v3.2", config: Optional[Dict[str, Any]] = None):
        cfg = dict(config or {})
        cfg["adaptive_agent_weighting"] = False
        super().__init__(agents, model=model, config=cfg)


class NoFallbackManager(ArgDebateManager):
    """Ablation: do not produce a fallback final answer on unresolved debates."""

    def __init__(self, agents, model: str = "deepseek-v3.2", config: Optional[Dict[str, Any]] = None):
        cfg = dict(config or {})
        cfg["enable_fallback_answer"] = False
        super().__init__(agents, model=model, config=cfg)


def get_ablation_manager(
    agents: List[LLMAgent],
    ablation_type: str,
    model: str = "deepseek-v3.2",
    config: Optional[Dict[str, Any]] = None,
) -> ArgDebateManager:
    """Factory for ablation variants.

    Supported ablation_type values:
      - "no_dfquad": DF-QuAD replaced with initial-strength passthrough
      - "no_belief_update": agents never update beliefs
      - "no_relation_llm": heuristic-only relation judging
      - "no_quality_attenuation": remove argument-quality attenuation
      - "no_reliability": disable adaptive reliability weighting
      - "no_fallback": return no answer if unresolved
      - None / "full": standard ArgDebate
    """
    config = config or {}

    if ablation_type == "no_dfquad":
        return NoDFQuADManager(agents, model=model, config=config)
    elif ablation_type == "no_belief_update":
        return NoBeliefUpdateManager(agents, model=model, config=config)
    elif ablation_type == "no_relation_llm":
        cfg = dict(config)
        cfg["use_llm_relation"] = False
        return ArgDebateManager(agents, model=model, config=cfg)
    elif ablation_type == "no_quality_attenuation":
        return NoQualityAttenuationManager(agents, model=model, config=config)
    elif ablation_type == "no_reliability":
        return NoReliabilityManager(agents, model=model, config=config)
    elif ablation_type == "no_fallback":
        return NoFallbackManager(agents, model=model, config=config)
    else:
        return ArgDebateManager(agents, model=model, config=config)
