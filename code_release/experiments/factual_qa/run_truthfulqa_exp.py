import pandas as pd
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.agents.llm_agent import LLMAgent
from src.debate.debate_manager import ArgDebateManager
from src.baselines.majority_vote import MajorityVote
from src.baselines.scalar_transcript_judge import ScalarTranscriptJudge
from src.baselines.self_consistency import SelfConsistency
from src.baselines.vanilla_debate import VanillaDebate
from src.baselines.weighted_vote import WeightedVote
from experiments.formal_judge import FormalJudge
from experiments.formal_table_eligibility import (
    method_formal_table_eligibility,
    summarize_formal_table_eligibility,
)
from experiments.main_baseline_set import (
    configured_methods as _main_configured_methods,
    empty_correctness_dict as _main_empty_correctness_dict,
    empty_stats_dict as _main_empty_stats_dict,
    estimate_tokens as _main_estimate_tokens,
    extract_method_prediction as _main_extract_method_prediction,
    select_consensus_response as _main_select_consensus_response,
    validate_extra_baselines,
)
from experiments.method_execution import run_method_safely as _main_run_method_safely


def _load_project_env() -> None:
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and key not in os.environ:
            os.environ[key] = value

    if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE", "")


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().lower().replace("\n", " ").split())


def _tokenize_text(text: str) -> list[str]:
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in _normalize_text(text))
    return [tok for tok in cleaned.split() if tok]


def _select_consensus_response(response_dict: dict) -> str:
    return _main_select_consensus_response(response_dict)


def _estimate_tokens(text: str) -> int:
    return _main_estimate_tokens(text)


def _text_jaccard_distance(a: str, b: str) -> float:
    ta = set(_tokenize_text(a))
    tb = set(_tokenize_text(b))
    if not ta and not tb:
        return 0.0
    overlap = len(ta & tb) / max(1, len(ta | tb))
    return 1.0 - overlap


def _proposal_disagreement_score(answers: list[str]) -> float:
    cleaned = [str(a).strip() for a in answers if str(a).strip()]
    if len(cleaned) < 2:
        return 0.0
    distances = []
    for i in range(len(cleaned)):
        for j in range(i + 1, len(cleaned)):
            distances.append(_text_jaccard_distance(cleaned[i], cleaned[j]))
    if not distances:
        return 0.0
    return sum(distances) / len(distances)


def _proposal_info_density_score(answers: list[str]) -> float:
    cleaned = [str(a).strip() for a in answers if str(a).strip()]
    if not cleaned:
        return 0.0

    densities = []
    for text in cleaned:
        tokens = _tokenize_text(text)
        if not tokens:
            densities.append(0.0)
            continue
        unique_ratio = len(set(tokens)) / max(1, len(tokens))
        density = unique_ratio * len(tokens)
        densities.append(density)
    return sum(densities) / len(densities)


def _extract_method_prediction(method_result, method_name: str) -> str:
    return _main_extract_method_prediction(
        method_result,
        method_name,
        prefer_formal_normalized=False,
    )


def _truthfulqa_method_eligibility(method: str, method_result: dict, judge_record: dict) -> dict:
    return method_formal_table_eligibility(
        dataset="TruthfulQA",
        method=method,
        method_result=method_result,
        judge_record=judge_record,
    )


def _is_correct_fallback(prediction: str, target: str) -> bool:
    pred = _normalize_text(prediction)
    gold = _normalize_text(target)
    if not pred:
        return False
    if gold in pred or pred in gold:
        return True

    pred_tokens = set(_tokenize_text(pred))
    gold_tokens = set(_tokenize_text(gold))
    if not gold_tokens:
        return False
    overlap = len(pred_tokens & gold_tokens) / max(1, len(gold_tokens))
    return overlap >= 0.6


def _judge_with_llm(client, model: str, question: str, prediction: str, target: str, use_llm_judge: bool = True) -> bool:
    return _judge_with_llm_record(client, model, question, prediction, target, use_llm_judge)["correct"]


def _judge_with_llm_record(client, model: str, question: str, prediction: str, target: str, use_llm_judge: bool = True) -> dict:
    if not prediction.strip():
        return {
            "correct": False,
            "judge_model": model,
            "judge_prompt_version": "truthfulqa_correctness_v1_original_runner",
            "judge_raw_response": None,
            "judge_fallback_used": True,
            "judge_error": "empty_prediction",
        }
    if not use_llm_judge or client is None:
        return {
            "correct": _is_correct_fallback(prediction, target),
            "judge_model": model,
            "judge_prompt_version": "truthfulqa_correctness_v1_original_runner",
            "judge_raw_response": None,
            "judge_fallback_used": True,
            "judge_error": None,
        }
    prompt = f"""
You are an evaluator for QA correctness.

Question: {question}
Reference answer: {target}
Model answer: {prediction}

Determine whether model answer is factually correct given the reference.
Return JSON only:
{{"correct": true or false}}
"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        return {
            "correct": bool(data.get("correct", False)),
            "judge_model": model,
            "judge_prompt_version": "truthfulqa_correctness_v1_original_runner",
            "judge_raw_response": data,
            "judge_fallback_used": False,
            "judge_error": None,
        }
    except Exception as exc:
        return {
            "correct": _is_correct_fallback(prediction, target),
            "judge_model": model,
            "judge_prompt_version": "truthfulqa_correctness_v1_original_runner",
            "judge_raw_response": None,
            "judge_fallback_used": True,
            "judge_error": str(exc),
        }


def _run_method_safely(method_name: str, runner, default_result):
    return _main_run_method_safely(method_name, runner, default_result)


def _write_json_atomic(path: str, payload: dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def _safe_run_id(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(raw).strip())
    return cleaned.strip("._-") or "unknown"


def _default_checkpoint_path(subset_size: int, num_agents: int, max_rounds: int, use_llm_judge: bool) -> str:
    suffix = "llmjudge" if use_llm_judge else "fallbackjudge"
    model_alias = _safe_run_id(os.getenv("EXP_MODEL_ALIAS") or os.getenv("OPENAI_MODEL") or "legacy")
    filename = f"checkpoint_truthfulqa_{model_alias}_n{subset_size}_a{num_agents}_r{max_rounds}_{suffix}.json"
    return os.path.join(os.path.dirname(__file__), filename)


def _resolve_subset_path() -> str:
    override = os.getenv("EXP_TRUTHFULQA_SUBSET_FILE", "").strip()
    if not override:
        return os.path.join(os.path.dirname(__file__), "truthfulqa_subset.csv")

    raw_path = Path(override).expanduser()
    if raw_path.is_absolute():
        return str(raw_path)

    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / raw_path,
        Path(__file__).resolve().parent / raw_path,
        Path.cwd() / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _configured_methods(extra_baselines: list[str] | None = None) -> list[str]:
    return _main_configured_methods(extra_baselines)


def _empty_stats_dict(methods: list[str]) -> dict:
    return _main_empty_stats_dict(methods)


def _empty_correctness_dict(methods: list[str]) -> dict:
    return _main_empty_correctness_dict(methods)


def _load_checkpoint(checkpoint_path: str, expected_config: dict) -> dict | None:
    if not os.path.exists(checkpoint_path):
        return None
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        print(f"Checkpoint load failed ({checkpoint_path}): {exc}")
        return None

    cfg = payload.get("config", {})
    for key, value in expected_config.items():
        if cfg.get(key) != value:
            print("Checkpoint config mismatch; starting a new run.")
            return None
    return payload


def _build_checkpoint_payload(
    config: dict,
    results: list,
    timing: dict,
    correctness: dict,
    status_log: dict,
    rounds_log: dict,
    token_log: dict,
    route_log: list,
) -> dict:
    return {
        "config": config,
        "updated_at": datetime.now().isoformat(),
        "results": results,
        "timing": timing,
        "correctness": correctness,
        "status_log": status_log,
        "rounds_log": rounds_log,
        "token_log": token_log,
        "route_log": route_log,
    }


def _summarize_argdebate_traces(results: list[dict]) -> dict:
    status_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    termination_counts: dict[str, int] = {}
    route_reason_counts: dict[str, int] = {}

    for item in results:
        ad = item.get("arg_debate", {}) if isinstance(item.get("arg_debate"), dict) else {}
        status = str(ad.get("status", "unknown") or "unknown")
        reason = str(ad.get("reason", "none") or "none")
        source = str(ad.get("final_answer_source", "unknown") or "unknown")
        meta = ad.get("meta", {}) if isinstance(ad.get("meta"), dict) else {}
        termination = meta.get("termination", {}) if isinstance(meta.get("termination"), dict) else {}
        termination_type = str(termination.get("type", "unknown") or "unknown")

        status_counts[status] = status_counts.get(status, 0) + 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        termination_counts[termination_type] = termination_counts.get(termination_type, 0) + 1

        for route_reason in meta.get("route_reason", []) if isinstance(meta.get("route_reason"), list) else []:
            rr = str(route_reason or "unknown")
            route_reason_counts[rr] = route_reason_counts.get(rr, 0) + 1

    n = max(len(results), 1)
    return {
        "status_counts": status_counts,
        "status_rates": {k: v / n for k, v in status_counts.items()},
        "reason_counts": reason_counts,
        "reason_rates": {k: v / n for k, v in reason_counts.items()},
        "final_answer_source_counts": source_counts,
        "final_answer_source_rates": {k: v / n for k, v in source_counts.items()},
        "termination_type_counts": termination_counts,
        "termination_type_rates": {k: v / n for k, v in termination_counts.items()},
        "route_reason_counts": route_reason_counts,
        "route_reason_rates": {k: v / n for k, v in route_reason_counts.items()},
    }


def _summarize_runtime_profiles(results: list[dict]) -> dict:
    totals = []
    initial_proposals = []
    initial_conflicts = []
    round_totals = []
    argument_collection = []
    qbaf_eval = []
    proposal_refresh = []
    conflict_refresh = []

    for item in results:
        ad = item.get("arg_debate", {}) if isinstance(item.get("arg_debate"), dict) else {}
        meta = ad.get("meta", {}) if isinstance(ad.get("meta"), dict) else {}
        profile = meta.get("runtime_profile", {}) if isinstance(meta.get("runtime_profile"), dict) else {}

        if "total_s" in profile:
            totals.append(float(profile["total_s"]))
        if "initial_proposals_s" in profile:
            initial_proposals.append(float(profile["initial_proposals_s"]))
        if "initial_conflict_detection_s" in profile:
            initial_conflicts.append(float(profile["initial_conflict_detection_s"]))

        round_entries = profile.get("rounds", []) if isinstance(profile.get("rounds"), list) else []
        for round_info in round_entries:
            if "round_total_s" in round_info:
                round_totals.append(float(round_info["round_total_s"]))
            if "argument_collection_s" in round_info:
                argument_collection.append(float(round_info["argument_collection_s"]))
            if "qbaf_eval_s" in round_info:
                qbaf_eval.append(float(round_info["qbaf_eval_s"]))
            if "proposal_refresh_s" in round_info:
                proposal_refresh.append(float(round_info["proposal_refresh_s"]))
            if "conflict_refresh_s" in round_info:
                conflict_refresh.append(float(round_info["conflict_refresh_s"]))

    def avg(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    return {
        "avg_total_s": avg(totals),
        "avg_initial_proposals_s": avg(initial_proposals),
        "avg_initial_conflict_detection_s": avg(initial_conflicts),
        "avg_round_total_s": avg(round_totals),
        "avg_argument_collection_s": avg(argument_collection),
        "avg_qbaf_eval_s": avg(qbaf_eval),
        "avg_proposal_refresh_s": avg(proposal_refresh),
        "avg_conflict_refresh_s": avg(conflict_refresh),
        "num_profiled_items": len(totals),
    }

def run_experiment(subset_size=5, num_agents=3, max_rounds=3, use_llm_judge=True):
    """Runs the experiment on a small subset of TruthfulQA."""
    _load_project_env()
    subset_path = _resolve_subset_path()
    
    if not os.path.exists(subset_path):
        print(f"Data file not found at {subset_path}. Please run prepare_truthfulqa.py first.")
        return
    
    df = pd.read_csv(subset_path).head(subset_size)
    judge = FormalJudge(dataset="TruthfulQA", use_llm_judge=use_llm_judge)

    accept_threshold = float(os.getenv("EXP_ACCEPT_THRESHOLD", "0.5"))
    defeat_threshold = float(os.getenv("EXP_DEFEAT_THRESHOLD", "0.2"))
    early_stop_patience = int(os.getenv("EXP_EARLY_STOP_PATIENCE", "2"))
    early_stop_min_rounds = int(os.getenv("EXP_EARLY_STOP_MIN_ROUNDS", "2"))
    enable_early_stop = os.getenv("EXP_ENABLE_EARLY_STOP", "1").strip().lower() in {"1", "true", "yes", "on"}
    enable_debate_routing = os.getenv("EXP_ENABLE_DEBATE_ROUTING", "1").strip().lower() in {"1", "true", "yes", "on"}
    route_disagreement_threshold = float(os.getenv("EXP_ROUTE_DISAGREEMENT_THRESHOLD", "0.65"))
    enable_info_density_gate = os.getenv("EXP_ENABLE_INFO_DENSITY_GATE", "1").strip().lower() in {"1", "true", "yes", "on"}
    route_info_density_threshold = float(os.getenv("EXP_ROUTE_INFO_DENSITY_THRESHOLD", "115.0"))
    route_logic = os.getenv("EXP_ROUTE_LOGIC", "or").strip().lower()
    if route_logic not in {"or", "and"}:
        route_logic = "or"
    proposal_temperature = float(os.getenv("EXP_PROPOSAL_TEMPERATURE", "0.7"))
    argument_temperature = float(os.getenv("EXP_ARGUMENT_TEMPERATURE", "0.3"))
    use_llm_conflict = os.getenv("EXP_USE_LLM_CONFLICT", "1").strip().lower() in {"1", "true", "yes", "on"}
    extra_baselines = [
        part.strip()
        for part in os.getenv("EXP_EXTRA_BASELINES", "").split(",")
        if part.strip()
    ]
    extra_baselines = validate_extra_baselines(extra_baselines)
    methods = _configured_methods(extra_baselines)
    
    run_config = {
        "dataset": "TruthfulQA",
        "subset_size": int(subset_size),
        "subset_file": str(Path(subset_path).resolve()),
        "model_alias": os.getenv("EXP_MODEL_ALIAS", ""),
        "model_id": os.getenv("OPENAI_MODEL", ""),
        "num_agents": int(num_agents),
        "max_rounds": int(max_rounds),
        "methods": methods,
        "extra_baselines": extra_baselines,
        "use_llm_judge": bool(use_llm_judge),
        "judge_policy": judge.policy_summary(),
        "use_llm_conflict": bool(use_llm_conflict),
        "enable_debate_routing": bool(enable_debate_routing),
        "route_logic": route_logic,
        "route_disagreement_threshold": float(route_disagreement_threshold),
        "route_info_density_threshold": float(route_info_density_threshold),
        "enable_info_density_gate": bool(enable_info_density_gate),
        "proposal_temperature": float(proposal_temperature),
        "argument_temperature": float(argument_temperature),
        "accept_threshold": float(accept_threshold),
        "defeat_threshold": float(defeat_threshold),
        "enable_early_stop": bool(enable_early_stop),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_rounds": int(early_stop_min_rounds),
    }

    checkpoint_enabled = os.getenv("EXP_ENABLE_RESUME", "1").strip().lower() in {"1", "true", "yes", "on"}
    checkpoint_path = os.getenv("EXP_RESUME_FILE", "").strip() or _default_checkpoint_path(
        subset_size=subset_size,
        num_agents=num_agents,
        max_rounds=max_rounds,
        use_llm_judge=use_llm_judge,
    )

    results = []
    timing = _empty_stats_dict(methods)
    correctness = _empty_correctness_dict(methods)
    status_log = _empty_stats_dict(methods)
    rounds_log = _empty_stats_dict(methods)
    token_log = _empty_stats_dict(methods)
    route_log: list[dict] = []

    if checkpoint_enabled:
        payload = _load_checkpoint(checkpoint_path, run_config)
        if payload:
            results = payload.get("results", [])
            timing = payload.get("timing", timing)
            correctness = payload.get("correctness", correctness)
            status_log = payload.get("status_log", status_log)
            rounds_log = payload.get("rounds_log", rounds_log)
            token_log = payload.get("token_log", token_log)
            route_log = payload.get("route_log", route_log)
            print(f"Loaded checkpoint: {checkpoint_path}")
            print(f"Resuming from {len(results)} completed questions.")
        else:
            print(f"No valid checkpoint found. Starting fresh run.")

    completed_questions = {str(item.get("question", "")).strip() for item in results if item.get("question")}
    
    for idx, row in df.iterrows():
        question = row['Question']
        correct_answer = row['Best Answer']
        if str(question).strip() in completed_questions:
            print(f"\n--- Question {idx+1}: already completed, skipping ---")
            continue
        print(f"\n--- Question {idx+1}: {question} ---")
        
        # 1. Majority Vote
        mv = MajorityVote()
        mv_res, elapsed = _run_method_safely(
            "Majority Vote",
            lambda: mv.run(question, num_agents=num_agents),
            {"status": "error", "final_answer": ""}
        )
        timing["majority_vote"].append(elapsed)
        mv_pred = _extract_method_prediction(mv_res, "majority_vote")
        mv_judge = judge.evaluate(question, mv_pred, correct_answer, method="majority_vote")
        mv_eligibility = _truthfulqa_method_eligibility("majority_vote", mv_res, mv_judge)
        if mv_judge["correct"]:
            correctness["majority_vote"] += 1
        status_log["majority_vote"].append(mv_res.get("status", "completed"))
        rounds_log["majority_vote"].append(1)
        token_log["majority_vote"].append(_estimate_tokens(mv_pred))
        print(f"Majority Vote: {mv_res.get('final_answer', '')[:50]}...")
        
        # 2. Vanilla Debate
        vd = VanillaDebate(max_rounds=max_rounds)
        vd_res, elapsed = _run_method_safely(
            "Vanilla Debate",
            lambda: vd.run(question, num_agents=num_agents),
            {"status": "fallback", "round": max_rounds, "result": {}}
        )
        timing["vanilla_debate"].append(elapsed)
        vanilla_pred = _extract_method_prediction(vd_res, "vanilla_debate")
        vanilla_judge = judge.evaluate(question, vanilla_pred, correct_answer, method="vanilla_debate")
        vanilla_eligibility = _truthfulqa_method_eligibility("vanilla_debate", vd_res, vanilla_judge)
        if vanilla_judge["correct"]:
            correctness["vanilla_debate"] += 1
        status_log["vanilla_debate"].append(vd_res.get("status", "fallback"))
        rounds_log["vanilla_debate"].append(int(vd_res.get("round", max_rounds)))
        token_log["vanilla_debate"].append(_estimate_tokens(vanilla_pred))
        print(f"Vanilla Debate Status: {vd_res.get('status', 'fallback')}")

        # 2.5 Weighted Vote
        wv = WeightedVote()
        wv_res, elapsed = _run_method_safely(
            "Weighted Vote",
            lambda: wv.run(question, num_agents=num_agents),
            {"status": "error", "final_answer": ""}
        )
        timing["weighted_vote"].append(elapsed)
        wv_pred = _extract_method_prediction(wv_res, "weighted_vote")
        wv_judge = judge.evaluate(question, wv_pred, correct_answer, method="weighted_vote")
        wv_eligibility = _truthfulqa_method_eligibility("weighted_vote", wv_res, wv_judge)
        if wv_judge["correct"]:
            correctness["weighted_vote"] += 1
        status_log["weighted_vote"].append(wv_res.get("status", "completed"))
        rounds_log["weighted_vote"].append(1)
        token_log["weighted_vote"].append(_estimate_tokens(wv_pred))
        print(f"Weighted Vote: {wv_res.get('final_answer', '')[:50]}...")

        sc_res = None
        sc_judge = None
        if "self_consistency" in extra_baselines:
            sc_samples = int(os.getenv("EXP_SELF_CONSISTENCY_SAMPLES", str(max(num_agents, num_agents * (max_rounds + 1)))))
            sc = SelfConsistency(temperature=proposal_temperature)
            sc_res, elapsed = _run_method_safely(
                "Self Consistency",
                lambda: sc.run(question, num_samples=sc_samples),
                {"status": "error", "final_answer": ""}
            )
            timing["self_consistency"].append(elapsed)
            sc_pred = _extract_method_prediction(sc_res, "self_consistency")
            sc_judge = judge.evaluate(question, sc_pred, correct_answer, method="self_consistency")
            sc_eligibility = _truthfulqa_method_eligibility("self_consistency", sc_res, sc_judge)
            if sc_judge["correct"]:
                correctness["self_consistency"] += 1
            status_log["self_consistency"].append(sc_res.get("status", "completed"))
            rounds_log["self_consistency"].append(1)
            token_log["self_consistency"].append(_estimate_tokens(sc_pred))
            print(f"Self Consistency: {sc_res.get('final_answer', '')[:50]}...")

        stj_res = None
        stj_judge = None
        if "scalar_transcript_judge" in extra_baselines:
            stj = ScalarTranscriptJudge(max_rounds=max_rounds)
            stj_res, elapsed = _run_method_safely(
                "Scalar Transcript Judge",
                lambda: stj.run(question, num_agents=num_agents),
                {"status": "error", "final_answer": ""}
            )
            timing["scalar_transcript_judge"].append(elapsed)
            stj_pred = _extract_method_prediction(stj_res, "scalar_transcript_judge")
            stj_judge = judge.evaluate(question, stj_pred, correct_answer, method="scalar_transcript_judge")
            stj_eligibility = _truthfulqa_method_eligibility("scalar_transcript_judge", stj_res, stj_judge)
            if stj_judge["correct"]:
                correctness["scalar_transcript_judge"] += 1
            status_log["scalar_transcript_judge"].append(stj_res.get("status", "completed"))
            rounds_log["scalar_transcript_judge"].append(int(stj_res.get("round", max_rounds)))
            token_log["scalar_transcript_judge"].append(_estimate_tokens(stj_pred))
            print(f"Scalar Transcript Judge: {stj_res.get('final_answer', '')[:50]}...")
        
        # 3. ArgDebate (Ours)
        route_answers = [mv_pred, wv_pred, vanilla_pred]
        disagreement_score = _proposal_disagreement_score(route_answers)
        info_density_score = _proposal_info_density_score(route_answers)
        disagreement_hit = disagreement_score >= route_disagreement_threshold
        density_hit = info_density_score >= route_info_density_threshold if enable_info_density_gate else False

        if route_logic == "and":
            gate_hit = disagreement_hit and (density_hit if enable_info_density_gate else True)
        else:
            gate_hit = disagreement_hit or density_hit

        # Bug fix (2026-04-12): if all upstream baselines failed (empty proposals),
        # disagreement_score=0 would falsely trigger a low-disagreement route-skip.
        # Force-route to full debate so ArgDebate is not silently nullified.
        all_proposals_empty = not any(str(a).strip() for a in route_answers)
        should_route_to_full_debate = (
            (not enable_debate_routing) or gate_hit or all_proposals_empty
        )
        route_reason = []
        if not enable_debate_routing:
            route_reason.append("routing_disabled")
        elif all_proposals_empty:
            route_reason.append("upstream_failed_force_route")
        else:
            if disagreement_hit:
                route_reason.append("high_disagreement")
            if density_hit:
                route_reason.append("high_info_density")
            if not route_reason:
                route_reason.append("low_disagreement_and_density")

        if should_route_to_full_debate:
            agents = [
                LLMAgent(
                    f"agent_{i}",
                    proposal_temperature=proposal_temperature,
                    argument_temperature=argument_temperature,
                )
                for i in range(num_agents)
            ]
            manager = ArgDebateManager(
                agents,
                config={
                    "max_rounds": max_rounds,
                    "accept_threshold": accept_threshold,
                    "defeat_threshold": defeat_threshold,
                    "enable_early_stop": enable_early_stop,
                    "early_stop_patience": early_stop_patience,
                    "early_stop_min_rounds": early_stop_min_rounds,
                    "use_llm_conflict": use_llm_conflict,
                },
            )
            ad_res, elapsed = _run_method_safely(
                "ArgDebate",
                lambda: manager.resolve(question),
                {"status": "fallback", "round": max_rounds, "result": {}}
            )
            ad_res.setdefault("meta", {})
            ad_res["meta"]["debate_routed"] = True
            ad_res["meta"]["route_disagreement_score"] = disagreement_score
            ad_res["meta"]["route_info_density_score"] = info_density_score
            ad_res["meta"]["route_disagreement_hit"] = disagreement_hit
            ad_res["meta"]["route_density_hit"] = density_hit
            ad_res["meta"]["route_logic"] = route_logic
            ad_res["meta"]["route_reason"] = route_reason
        else:
            t0 = time.time()
            ad_res = {
                "status": "routed_skip",
                "reason": "routing_gate_skip",
                "result": {
                    "agent_0": vanilla_pred,
                    "agent_1": mv_pred,
                    "agent_2": wv_pred,
                },
                "final_answer": vanilla_pred or mv_pred or wv_pred,
                "final_answer_source": "routing_low_disagreement",
                "meta": {
                    "debate_routed": False,
                    "route_disagreement_score": disagreement_score,
                    "route_info_density_score": info_density_score,
                    "route_disagreement_hit": disagreement_hit,
                    "route_density_hit": density_hit,
                    "route_logic": route_logic,
                    "route_reason": route_reason,
                    "route_threshold": route_disagreement_threshold,
                    "route_info_density_threshold": route_info_density_threshold,
                    "termination": {
                        "type": "routed_skip",
                        "round": 0,
                        "strategy": "routing_gate_selector",
                    },
                    "round_diagnostics": [],
                    "runtime_profile": {
                        "total_s": 0.0,
                        "initial_proposals_s": 0.0,
                        "initial_conflict_detection_s": 0.0,
                        "rounds": [],
                        "route_mode": "routed_skip",
                    },
                },
            }
            elapsed = time.time() - t0
            ad_res["meta"]["runtime_profile"]["total_s"] = round(elapsed, 6)
        timing["arg_debate"].append(elapsed)
        arg_pred = _extract_method_prediction(ad_res, "arg_debate")
        arg_judge = judge.evaluate(question, arg_pred, correct_answer, method="arg_debate")
        arg_eligibility = _truthfulqa_method_eligibility("arg_debate", ad_res, arg_judge)
        if arg_judge["correct"]:
            correctness["arg_debate"] += 1
        status_log["arg_debate"].append(ad_res.get("status", "fallback"))
        rounds_log["arg_debate"].append(int(ad_res.get("round", 1 if ad_res.get("status") == "routed_skip" else max_rounds)))
        token_log["arg_debate"].append(_estimate_tokens(arg_pred))
        route_log.append(
            {
                "debate_routed": bool(ad_res.get("meta", {}).get("debate_routed", True)),
                "route_disagreement_score": float(ad_res.get("meta", {}).get("route_disagreement_score", 0.0)),
                "route_info_density_score": float(ad_res.get("meta", {}).get("route_info_density_score", 0.0)),
                "route_disagreement_hit": bool(ad_res.get("meta", {}).get("route_disagreement_hit", False)),
                "route_density_hit": bool(ad_res.get("meta", {}).get("route_density_hit", False)),
                "route_reason": list(ad_res.get("meta", {}).get("route_reason", [])),
                "status": ad_res.get("status", "fallback"),
            }
        )
        print(f"ArgDebate Status: {ad_res.get('status', 'fallback')}")
        
        # Store results
        row_payload = {
            "question": question,
            "correct_answer": correct_answer,
            "majority_vote": mv_res,
            "vanilla_debate": vd_res,
            "weighted_vote": wv_res,
            "arg_debate": ad_res,
            "correctness": {
                "majority_vote": bool(mv_judge["correct"]),
                "vanilla_debate": bool(vanilla_judge["correct"]),
                "weighted_vote": bool(wv_judge["correct"]),
                "arg_debate": bool(arg_judge["correct"]),
            },
            "correctness_judge": {
                "policy": judge.policy_summary(),
                "majority_vote": mv_judge,
                "vanilla_debate": vanilla_judge,
                "weighted_vote": wv_judge,
                "arg_debate": arg_judge,
            },
            "formal_table_eligibility": {
                "majority_vote": mv_eligibility,
                "vanilla_debate": vanilla_eligibility,
                "weighted_vote": wv_eligibility,
                "arg_debate": arg_eligibility,
            },
        }
        if sc_res is not None and sc_judge is not None:
            row_payload["self_consistency"] = sc_res
            row_payload["correctness"]["self_consistency"] = bool(sc_judge["correct"])
            row_payload["correctness_judge"]["self_consistency"] = sc_judge
            row_payload["formal_table_eligibility"]["self_consistency"] = sc_eligibility
        if stj_res is not None and stj_judge is not None:
            row_payload["scalar_transcript_judge"] = stj_res
            row_payload["correctness"]["scalar_transcript_judge"] = bool(stj_judge["correct"])
            row_payload["correctness_judge"]["scalar_transcript_judge"] = stj_judge
            row_payload["formal_table_eligibility"]["scalar_transcript_judge"] = stj_eligibility
        results.append(row_payload)

        if checkpoint_enabled:
            checkpoint_payload = _build_checkpoint_payload(
                config=run_config,
                results=results,
                timing=timing,
                correctness=correctness,
                status_log=status_log,
                rounds_log=rounds_log,
                token_log=token_log,
                route_log=route_log,
            )
            _write_json_atomic(checkpoint_path, checkpoint_payload)
            print(f"Checkpoint updated: {len(results)}/{len(df)} questions completed")
        
    # Save results
    timestamp = os.getenv("EXP_OUTPUT_STAMP", "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.getenv("EXP_OUTPUT_DIR", "").strip() or os.path.dirname(__file__)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"results_{timestamp}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nExperiment results saved to {output_path}")

    summary = {
        "dataset": "TruthfulQA",
        "subset_size": len(results),
        "accuracy": {
            k: (v / max(len(results), 1)) for k, v in correctness.items()
        },
        "avg_time_s": {
            k: (sum(v) / max(len(v), 1)) for k, v in timing.items()
        },
        "deadlock_rate": {
            k: (sum(1 for s in status_log[k] if s == "fallback") / max(len(status_log[k]), 1)) for k in status_log
        },
        "avg_rounds": {
            k: (sum(v) / max(len(v), 1)) for k, v in rounds_log.items()
        },
        "token_efficiency_estimated": {
            k: (sum(token_log[k]) / max(correctness[k], 1)) for k in token_log
        },
        "routing": {
            "enabled": enable_debate_routing,
            "disagreement_threshold": route_disagreement_threshold,
            "info_density_gate_enabled": enable_info_density_gate,
            "info_density_threshold": route_info_density_threshold,
            "logic": route_logic,
            "routed_skip_rate": (
                sum(1 for item in route_log if not item.get("debate_routed", True)) / max(len(route_log), 1)
            ),
            "avg_route_disagreement": (
                sum(float(item.get("route_disagreement_score", 0.0)) for item in route_log) / max(len(route_log), 1)
            ),
            "avg_route_info_density": (
                sum(float(item.get("route_info_density_score", 0.0)) for item in route_log) / max(len(route_log), 1)
            ),
            "disagreement_hit_rate": (
                sum(1 for item in route_log if bool(item.get("route_disagreement_hit", False))) / max(len(route_log), 1)
            ),
            "info_density_hit_rate": (
                sum(1 for item in route_log if bool(item.get("route_density_hit", False))) / max(len(route_log), 1)
            ),
        },
        "formal_table_eligibility": summarize_formal_table_eligibility(results, methods),
        "arg_debate_trace_summary": _summarize_argdebate_traces(results),
        "arg_debate_runtime_profile_summary": _summarize_runtime_profiles(results),
        "config": run_config,
        "source_file": output_path
    }

    summary_path = str(output_dir / f"summary_{timestamp}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to {summary_path}")

    if checkpoint_enabled and os.path.exists(checkpoint_path):
        finished_checkpoint = checkpoint_path.replace(".json", f".done_{timestamp}.json")
        os.replace(checkpoint_path, finished_checkpoint)
        print(f"Checkpoint archived to {finished_checkpoint}")

    return summary

if __name__ == "__main__":
    subset_size = int(os.getenv("EXP_SUBSET_SIZE", "3"))
    num_agents = int(os.getenv("EXP_NUM_AGENTS", "3"))
    max_rounds = int(os.getenv("EXP_MAX_ROUNDS", "3"))
    use_llm_judge = os.getenv("EXP_USE_LLM_JUDGE", "1").strip().lower() in {"1", "true", "yes", "on"}
    run_experiment(
        subset_size=subset_size,
        num_agents=num_agents,
        max_rounds=max_rounds,
        use_llm_judge=use_llm_judge,
    )
