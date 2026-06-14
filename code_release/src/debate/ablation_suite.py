"""AblationSuite module for ArgDebate mechanism variants."""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.agents.llm_agent import LLMAgent
from src.debate.ablation_manager import get_ablation_manager
from src.debate.debate_manager import ArgDebateManager


MAIN_ABLATION_SET = (
    "no_relation_llm",
    "no_dfquad",
    "no_quality_attenuation",
    "no_fallback",
    "no_reliability",
)
SUPPLEMENT_ABLATION_SET = (
    "no_belief_update",
)
SUPPORTED_ABLATIONS = ("full",) + MAIN_ABLATION_SET + SUPPLEMENT_ABLATION_SET


def validate_ablation_variants(variants: Iterable[str]) -> list[str]:
    invalid = [variant for variant in variants if variant not in SUPPORTED_ABLATIONS]
    if invalid:
        raise ValueError(f"Unsupported ablation variants: {invalid}")
    return list(variants)


def variants_from_env(*, include_full_env: str = "EXP_ABLATION_INCLUDE_FULL", variants_env: str = "EXP_ABLATION_VARIANTS") -> list[str]:
    requested = [part.strip() for part in os.getenv(variants_env, "").split(",") if part.strip()]
    if requested:
        return validate_ablation_variants(requested)
    include_full = os.getenv(include_full_env, "0").strip().lower() in {"1", "true", "yes", "on"}
    variants = (["full"] if include_full else []) + list(MAIN_ABLATION_SET)
    return validate_ablation_variants(variants)


def config_for_variant(variant: str, base_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    validate_ablation_variants([variant])
    config = dict(base_config or {})
    if variant == "no_relation_llm":
        config["use_llm_relation"] = False
    return config


def build_ablation_manager(
    variant: str,
    agents: List[LLMAgent],
    *,
    model: str = "deepseek-v3.2",
    config: Optional[Dict[str, Any]] = None,
) -> ArgDebateManager:
    cfg = config_for_variant(variant, config)
    return get_ablation_manager(agents, variant, model=model, config=cfg)


def make_ablation_manager(
    variant: str,
    num_agents: int,
    *,
    model: str = "deepseek-v3.2",
    config: Optional[Dict[str, Any]] = None,
    agent_factory: Optional[Callable[[int], LLMAgent]] = None,
) -> ArgDebateManager:
    factory = agent_factory or (lambda idx: LLMAgent(f"agent_{idx}"))
    agents = [factory(i) for i in range(num_agents)]
    return build_ablation_manager(variant, agents, model=model, config=config)
