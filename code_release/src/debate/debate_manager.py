import concurrent.futures
import time
from typing import List, Dict, Any, Optional, Tuple
from ..agents.llm_agent import LLMAgent
from ..argumentation.qbaf_builder import QBAFBuilder
from .conflict_detector import ConflictDetector
from ..argumentation.argument_extractor import Argument
from .semantic_deadlock_detector import SemanticDeadlockDetector
from .fallback_strategy import FallbackStrategy

class ArgDebateManager:
    """
    ArgDebate core control loop with parallel execution and robust error handling.
    
    Attributes:
        agents (List[LLMAgent]): List of participating LLM agents.
        qbaf_builder (QBAFBuilder): Component to build and evaluate QBAFs.
        conflict_detector (ConflictDetector): Component to detect semantic conflicts.
        max_rounds (int): Maximum number of debate rounds.
        model (str): The LLM model to use for coordination.
    """
    
    def __init__(
        self,
        agents: List[LLMAgent],
        model: str = "gpt-4.1-mini",
        config: Optional[Dict[str, Any]] = None,
        use_parallelization: bool = True
    ):
        config = config or {}
        self.config = dict(config)
        self.agents = agents
        self.qbaf_builder = QBAFBuilder(
            model=model,
            use_llm_relation=bool(config.get("use_llm_relation", True)),
            use_quality_scoring=bool(config.get("use_quality_scoring", True)),
        )
        self.conflict_detector = ConflictDetector(
            model=model,
            use_llm=bool(config.get("use_llm_conflict", True))
        )
        self.max_rounds = config.get("max_rounds", 3)
        self.model = model
        self.use_parallelization = use_parallelization
        self.adaptive_agent_weighting = bool(config.get("adaptive_agent_weighting", True))
        self.reliability_ema_alpha = float(config.get("reliability_ema_alpha", 0.3))
        self.accept_threshold = float(config.get("accept_threshold", 0.5))
        self.defeat_threshold = float(config.get("defeat_threshold", 0.2))
        self.enable_early_stop = bool(config.get("enable_early_stop", True))
        self.early_stop_patience = int(config.get("early_stop_patience", 2))
        self.early_stop_min_rounds = int(config.get("early_stop_min_rounds", 2))
        self.agent_reliability = {agent.agent_id: 1.0 for agent in self.agents}
        self.deadlock_detector = SemanticDeadlockDetector()
        self.fallback_strategy = FallbackStrategy(self.agent_reliability)
        self.enable_deadlock_mitigation = bool(config.get("enable_deadlock_mitigation", True))
        self.enable_fallback_answer = bool(config.get("enable_fallback_answer", True))
        self.deadlock_temperature_increment = float(config.get("deadlock_temperature_increment", 0.2))
        # Phase 2.2: propagate belief_update_threshold to agents
        belief_update_threshold = float(config.get("belief_update_threshold", 0.6))
        for agent in self.agents:
            agent.belief_update_threshold = belief_update_threshold

    def _base_meta(self) -> Dict[str, Any]:
        return {
            "config": dict(self.config),
            "agent_reliability": dict(self.agent_reliability),
        }

    def _finalize_runtime_profile(self, runtime_profile: Dict[str, Any], started_at: float) -> Dict[str, Any]:
        runtime_profile["total_s"] = round(time.perf_counter() - started_at, 3)
        return runtime_profile

    def _normalize_text(self, text: str) -> str:
        return " ".join(str(text).strip().lower().replace("\n", " ").split())

    def _proposal_consensus(self, proposals: Dict[str, str]) -> str:
        items = []
        for key in sorted(proposals.keys()):
            value = str(proposals.get(key, "")).strip()
            if value:
                items.append(value)
        if not items:
            return ""

        buckets: Dict[str, List[str]] = {}
        for text in items:
            buckets.setdefault(self._normalize_text(text), []).append(text)

        best_bucket = sorted(
            buckets.values(),
            key=lambda lst: (-len(lst), len(min(lst, key=len)))
        )[0]
        return min(best_bucket, key=len)

    def _select_final_answer(
        self,
        proposals: Dict[str, str],
        all_args_list: Optional[List[Argument]] = None,
        weighted_strengths: Optional[Dict[str, float]] = None,
    ) -> Tuple[str, str, Dict[str, float]]:
        consensus_text = self._proposal_consensus(proposals)
        if not proposals:
            return "", "empty", {}

        if not all_args_list or not weighted_strengths:
            return consensus_text, "proposal_consensus", dict(self.agent_reliability)

        agent_score_list: Dict[str, List[float]] = {agent.agent_id: [] for agent in self.agents}
        for arg in all_args_list:
            if arg.agent_id in agent_score_list:
                agent_score_list[arg.agent_id].append(float(weighted_strengths.get(arg.id, 0.0)))

        combined_scores: Dict[str, float] = {}
        for agent in self.agents:
            agent_id = agent.agent_id
            quality_scores = agent_score_list.get(agent_id, [])
            quality = (sum(quality_scores) / len(quality_scores)) if quality_scores else 0.0
            reliability = float(self.agent_reliability.get(agent_id, 1.0))
            combined_scores[agent_id] = 0.7 * quality + 0.3 * reliability

        ranked_agents = sorted(
            proposals.keys(),
            key=lambda aid: (-combined_scores.get(aid, 0.0), -float(self.agent_reliability.get(aid, 1.0)), aid)
        )
        best_agent = ranked_agents[0] if ranked_agents else None
        best_text = str(proposals.get(best_agent, "")).strip() if best_agent else ""
        if not best_text:
            best_text = consensus_text

        return best_text, f"agent_weighted:{best_agent}", combined_scores

    def _fallback_disabled_result(
        self,
        reason: str,
        proposals: Dict[str, str],
        runtime_profile: Dict[str, Any],
        started_at: float,
        round_num: int,
        round_diagnostics: Optional[List[Dict[str, Any]]] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = {
            "round_diagnostics": round_diagnostics or [],
            **self._base_meta(),
            "runtime_profile": self._finalize_runtime_profile(runtime_profile, started_at),
            "termination": {
                "type": "fallback_disabled",
                "round": round_num,
                "strategy": "none",
            },
        }
        if extra_meta:
            meta.update(extra_meta)
        return {
            "status": "fallback_disabled",
            "reason": reason,
            "result": proposals,
            "final_answer": "",
            "final_answer_source": "fallback_disabled",
            "meta": meta,
        }

    def _get_agent_proposal(self, agent: LLMAgent, task: str) -> Tuple[str, str]:
        """Helper for parallel proposal generation."""
        return agent.agent_id, agent.generate_proposal(task)

    def _get_agent_arguments(self, agent: LLMAgent, task: str, conflicts: str) -> Tuple[str, List[Argument]]:
        """Helper for parallel argument generation."""
        return agent.agent_id, agent.generate_arguments(task, conflicts)

    def _collect_proposals(self, task: str) -> Dict[str, str]:
        proposals: Dict[str, str] = {}
        if self.use_parallelization:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_agent = {executor.submit(self._get_agent_proposal, agent, task): agent for agent in self.agents}
                for future in concurrent.futures.as_completed(future_to_agent):
                    agent_id, proposal = future.result()
                    proposals[agent_id] = proposal
        else:
            for agent in self.agents:
                agent_id, proposal = self._get_agent_proposal(agent, task)
                proposals[agent_id] = proposal
        return proposals

    def _collect_arguments(self, task: str, conflicts: str) -> Tuple[Dict[str, List[Argument]], List[Argument]]:
        agent_arguments: Dict[str, List[Argument]] = {}
        all_args_list: List[Argument] = []
        if self.use_parallelization:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_args = {executor.submit(self._get_agent_arguments, agent, task, conflicts): agent for agent in self.agents}
                for future in concurrent.futures.as_completed(future_to_args):
                    agent_id, args = future.result()
                    agent_arguments[agent_id] = args
                    all_args_list.extend(args)
        else:
            for agent in self.agents:
                agent_id, args = self._get_agent_arguments(agent, task, conflicts)
                agent_arguments[agent_id] = args
                all_args_list.extend(args)
        return agent_arguments, all_args_list

    def _update_agent_reliability(self, all_args_list: List[Argument], strengths: Dict[str, float]) -> None:
        if not self.adaptive_agent_weighting:
            return

        grouped_strengths: Dict[str, List[float]] = {agent.agent_id: [] for agent in self.agents}
        for arg in all_args_list:
            grouped_strengths.setdefault(arg.agent_id, []).append(float(strengths.get(arg.id, arg.confidence)))

        alpha = min(max(self.reliability_ema_alpha, 0.0), 1.0)
        for agent in self.agents:
            agent_id = agent.agent_id
            prev = float(self.agent_reliability.get(agent_id, 1.0))
            values = grouped_strengths.get(agent_id, [])
            if values:
                current = sum(values) / len(values)
                updated = (1 - alpha) * prev + alpha * current
                self.agent_reliability[agent_id] = max(0.2, min(1.2, updated))

    def resolve(self, task: str) -> Dict[str, Any]:
        """
        Full ArgDebate conflict resolution process with parallel agent calls.
        
        Args:
            task (str): The problem or question to resolve.
            
        Returns:
            Dict[str, Any]: Resolution status, final proposals, and audit trail.
        """
        started_at = time.perf_counter()
        runtime_profile: Dict[str, Any] = {
            "initial_proposals_s": 0.0,
            "initial_conflict_detection_s": 0.0,
            "rounds": [],
        }

        # 1. Initial proposals
        stage_started_at = time.perf_counter()
        proposals = self._collect_proposals(task)
        runtime_profile["initial_proposals_s"] = round(time.perf_counter() - stage_started_at, 3)
        
        # 2. Conflict detection
        stage_started_at = time.perf_counter()
        conflicts = self.conflict_detector.detect(proposals)
        runtime_profile["initial_conflict_detection_s"] = round(time.perf_counter() - stage_started_at, 3)
        
        if not conflicts:
            final_answer, final_answer_source, agent_scores = self._select_final_answer(proposals)
            return {
                "status": "no_conflict",
                "reason": "initial_consensus_or_no_detected_conflict",
                "result": proposals,
                "final_answer": final_answer,
                "final_answer_source": final_answer_source,
                "meta": {
                    **self._base_meta(),
                    "agent_scores": agent_scores,
                    "runtime_profile": self._finalize_runtime_profile(runtime_profile, started_at),
                    "termination": {
                        "type": "no_conflict",
                        "round": 0,
                    },
                }
            }

        round_diagnostics: List[Dict[str, Any]] = []
        previous_conflict_count = len(conflicts)
        non_improving_rounds = 0
        strengths: Dict[str, float] = {}
        initial_proposals = dict(proposals)  # Store initial proposals for fallback
        previous_max_strength = 0.0
        qbaf_stagnation_rounds = 0

        # 3. ArgDebate loop
        for round_num in range(self.max_rounds):
            print(f"--- ArgDebate Round {round_num + 1} ---")
            round_profile: Dict[str, Any] = {"round": round_num + 1}

            # 3.1 Generate structured arguments
            stage_started_at = time.perf_counter()
            agent_arguments, all_args_list = self._collect_arguments(task, str(conflicts))
            round_profile["argument_collection_s"] = round(time.perf_counter() - stage_started_at, 3)

            # 3.2 Build and Evaluate QBAF
            qbaf_metadata: Dict[str, Any] = {"arguments": [], "relations": [], "summary": {}}
            stage_started_at = time.perf_counter()
            try:
                qbaf_framework, qbaf_metadata = self.qbaf_builder.build_from_debate_with_metadata(agent_arguments)
                strengths = qbaf_framework.evaluate()
            except Exception as e:
                print(f"Error in QBAF evaluation: {e}. Falling back to initial strengths.")
                strengths = {arg.id: arg.confidence for arg in all_args_list}
                qbaf_metadata = {
                    "arguments": [arg.to_dict() for arg in all_args_list],
                    "relations": [],
                    "summary": {
                        "num_arguments": len(all_args_list),
                        "num_attacks": 0,
                        "num_supports": 0,
                        "num_neutral_pairs": 0,
                        "llm_relation_enabled": False,
                        "quality_scoring_enabled": False,
                        "evaluation_error": str(e),
                    },
                }
            round_profile["qbaf_eval_s"] = round(time.perf_counter() - stage_started_at, 3)

            stage_started_at = time.perf_counter()
            self._update_agent_reliability(all_args_list, strengths)
            round_profile["reliability_update_s"] = round(time.perf_counter() - stage_started_at, 3)

            weighted_strengths = {
                arg.id: float(strengths.get(arg.id, 0.0)) * float(self.agent_reliability.get(arg.agent_id, 1.0))
                for arg in all_args_list
            }

            # 3.3 Determine winners/losers and Update Beliefs
            accepted_args = [arg for arg in all_args_list if weighted_strengths.get(arg.id, 0.0) > self.accept_threshold]
            defeated_args = [arg for arg in all_args_list if weighted_strengths.get(arg.id, 0.0) < self.defeat_threshold]

            for agent in self.agents:
                agent.update_beliefs(accepted_args, defeated_args, strengths)

            # 3.4 Re-generate proposals
            stage_started_at = time.perf_counter()
            new_proposals = self._collect_proposals(task)
            round_profile["proposal_refresh_s"] = round(time.perf_counter() - stage_started_at, 3)

            # 3.5 Check if conflicts are resolved
            stage_started_at = time.perf_counter()
            new_conflicts = self.conflict_detector.detect(new_proposals)
            round_profile["conflict_refresh_s"] = round(time.perf_counter() - stage_started_at, 3)
            current_conflict_count = len(new_conflicts)

            # Track QBAF strength changes for stagnation detection
            current_max_strength = max(strengths.values()) if strengths else 0.0
            strength_change = abs(current_max_strength - previous_max_strength)
            if strength_change < 0.1:
                qbaf_stagnation_rounds += 1
            else:
                qbaf_stagnation_rounds = 0
            previous_max_strength = current_max_strength

            if current_conflict_count >= previous_conflict_count:
                non_improving_rounds += 1
            else:
                non_improving_rounds = 0
            previous_conflict_count = current_conflict_count

            round_diagnostics.append({
                "round": round_num + 1,
                "conflict_count": current_conflict_count,
                "accepted_args": len(accepted_args),
                "defeated_args": len(defeated_args),
                "agent_reliability": dict(self.agent_reliability),
                "max_strength": current_max_strength,
                "strength_change": strength_change,
                "proposals": dict(new_proposals),
                "accepted_argument_ids": [arg.id for arg in accepted_args],
                "defeated_argument_ids": [arg.id for arg in defeated_args],
                "qbaf_data": qbaf_metadata,
            })
            round_profile["round_total_s"] = round(
                round_profile["argument_collection_s"]
                + round_profile["qbaf_eval_s"]
                + round_profile["reliability_update_s"]
                + round_profile["proposal_refresh_s"]
                + round_profile["conflict_refresh_s"],
                3,
            )
            runtime_profile["rounds"].append(round_profile)

            # 3.6 Deadlock detection and mitigation
            if self.enable_deadlock_mitigation and round_num >= 1:
                deadlock_result = self.deadlock_detector.detect(round_diagnostics)

                if deadlock_result["deadlock_detected"]:
                    deadlock_type = deadlock_result["deadlock_type"]
                    print(f"Deadlock detected: {deadlock_type} (confidence: {deadlock_result.get('confidence', 0):.2f})")

                    # Try temperature escalation first
                    if round_num < self.max_rounds - 1:
                        print("Attempting to break deadlock with temperature escalation...")
                        for agent in self.agents:
                            agent.escalate_temperature(increment=self.deadlock_temperature_increment)
                        # Continue to next round with higher temperature
                        proposals = new_proposals
                        conflicts = new_conflicts
                        continue
                    else:
                        # Last round, use fallback strategy
                        if not self.enable_fallback_answer:
                            return self._fallback_disabled_result(
                                reason=f"deadlock_{deadlock_type}",
                                proposals=new_proposals,
                                runtime_profile=runtime_profile,
                                started_at=started_at,
                                round_num=round_num + 1,
                                round_diagnostics=round_diagnostics,
                                extra_meta={"deadlock_info": deadlock_result},
                            )
                        print(f"Using fallback strategy for {deadlock_type}")
                        fallback_answer = self.fallback_strategy.select_fallback_strategy(
                            deadlock_type=deadlock_type,
                            agents=self.agents,
                            proposals=new_proposals,
                            initial_proposals=initial_proposals,
                            all_arguments=all_args_list
                        )
                        return {
                            "status": "fallback",
                            "reason": f"deadlock_{deadlock_type}",
                            "result": new_proposals,
                            "final_answer": fallback_answer,
                            "final_answer_source": f"fallback_{deadlock_type}",
                            "audit_trail": strengths,
                            "meta": {
                                "round_diagnostics": round_diagnostics,
                                **self._base_meta(),
                                "runtime_profile": self._finalize_runtime_profile(runtime_profile, started_at),
                                "deadlock_info": deadlock_result,
                                "termination": {
                                    "type": "deadlock_fallback",
                                    "round": round_num + 1,
                                    "strategy": f"fallback_{deadlock_type}",
                                },
                            }
                        }

                # Check QBAF stagnation
                if qbaf_stagnation_rounds >= 2:
                    print(f"QBAF strength stagnation detected ({qbaf_stagnation_rounds} rounds)")
                    if round_num < self.max_rounds - 1:
                        for agent in self.agents:
                            agent.escalate_temperature(increment=self.deadlock_temperature_increment)

            if not new_conflicts:
                final_answer, final_answer_source, agent_scores = self._select_final_answer(
                    new_proposals,
                    all_args_list=all_args_list,
                    weighted_strengths=weighted_strengths,
                )
                return {
                    "status": "resolved",
                    "reason": "resolved_after_debate",
                    "round": round_num + 1,
                    "result": new_proposals,
                    "final_answer": final_answer,
                    "final_answer_source": final_answer_source,
                    "audit_trail": strengths,
                    "meta": {
                        "round_diagnostics": round_diagnostics,
                        **self._base_meta(),
                        "agent_scores": agent_scores,
                        "runtime_profile": self._finalize_runtime_profile(runtime_profile, started_at),
                        "termination": {
                            "type": "resolved",
                            "round": round_num + 1,
                        },
                    }
                }

            if (
                self.enable_early_stop
                and (round_num + 1) >= max(1, self.early_stop_min_rounds)
                and non_improving_rounds >= self.early_stop_patience
            ):
                if not self.enable_fallback_answer:
                    return self._fallback_disabled_result(
                        reason="early_stop_no_conflict_reduction",
                        proposals=new_proposals,
                        runtime_profile=runtime_profile,
                        started_at=started_at,
                        round_num=round_num + 1,
                        round_diagnostics=round_diagnostics,
                    )
                final_answer, final_answer_source, agent_scores = self._select_final_answer(
                    new_proposals,
                    all_args_list=all_args_list,
                    weighted_strengths=weighted_strengths,
                )
                return {
                    "status": "fallback",
                    "reason": "early_stop_no_conflict_reduction",
                    "result": new_proposals,
                    "final_answer": final_answer,
                    "final_answer_source": final_answer_source,
                    "audit_trail": strengths,
                    "meta": {
                        "round_diagnostics": round_diagnostics,
                        **self._base_meta(),
                        "agent_scores": agent_scores,
                        "runtime_profile": self._finalize_runtime_profile(runtime_profile, started_at),
                        "termination": {
                            "type": "early_stop",
                            "round": round_num + 1,
                            "strategy": "agent_weighted_selector",
                        },
                    }
                }
            
            proposals = new_proposals
            conflicts = new_conflicts
        
        final_answer, final_answer_source, agent_scores = self._select_final_answer(
            proposals,
            all_args_list=all_args_list,
            weighted_strengths=weighted_strengths,
        )
        if not self.enable_fallback_answer:
            return self._fallback_disabled_result(
                reason="max_rounds_exhausted",
                proposals=proposals,
                runtime_profile=runtime_profile,
                started_at=started_at,
                round_num=self.max_rounds,
                round_diagnostics=round_diagnostics,
            )
        return {
            "status": "fallback",
            "reason": "max_rounds_exhausted",
            "result": proposals,
            "final_answer": final_answer,
            "final_answer_source": final_answer_source,
            "audit_trail": strengths,
            "meta": {
                "round_diagnostics": round_diagnostics,
                **self._base_meta(),
                "agent_scores": agent_scores,
                "runtime_profile": self._finalize_runtime_profile(runtime_profile, started_at),
                "termination": {
                    "type": "max_rounds",
                    "round": self.max_rounds,
                    "strategy": "agent_weighted_selector",
                },
            }
        }

    def run_debate(self, task: str, max_rounds: Optional[int] = None) -> Dict[str, Any]:
        """Backward-compatible wrapper for benchmark scripts."""
        original_max_rounds = self.max_rounds
        if max_rounds is not None:
            self.max_rounds = max_rounds
        try:
            return self.resolve(task)
        finally:
            self.max_rounds = original_max_rounds
