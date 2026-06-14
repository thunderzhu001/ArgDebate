import json
import sys
import os
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.agents.llm_agent import LLMAgent
from src.debate.debate_manager import ArgDebateManager
from src.debate.ablation_suite import make_ablation_manager
from src.baselines.majority_vote import MajorityVote
from src.baselines.scalar_transcript_judge import ScalarTranscriptJudge
from src.baselines.self_consistency import SelfConsistency
from src.baselines.weighted_vote import WeightedVote
from src.baselines.vanilla_debate import VanillaDebate
from experiments.formal_judge import FormalJudge
from experiments.formal_table_eligibility import (
    method_formal_table_eligibility,
    score_strategyqa_boolean,
    summarize_formal_table_eligibility,
)
from experiments.main_baseline_set import (
    configured_methods as _main_configured_methods,
    empty_correctness_dict as _main_empty_correctness_dict,
    empty_stats_dict as _main_empty_stats_dict,
    estimate_tokens as _main_estimate_tokens,
    extract_method_prediction as _main_extract_method_prediction,
    normalize_text as _main_normalize_text,
    select_consensus_response as _main_select_consensus_response,
    status_counts as _main_status_counts,
    validate_extra_baselines,
)
from experiments.method_execution import run_method_safely as _main_run_method_safely


ROOT = Path(__file__).resolve().parents[2]


def _load_project_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()
    if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE", "")


def _safe_run_id(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(raw).strip())
    return cleaned.strip("._-") or "unknown"


def _default_checkpoint_path(subset_size: int, num_agents: int, max_rounds: int, use_llm_judge: bool) -> str:
    suffix = "llmjudge" if use_llm_judge else "fallbackjudge"
    model_alias = _safe_run_id(os.getenv("EXP_MODEL_ALIAS") or os.getenv("OPENAI_MODEL") or "legacy")
    filename = f"checkpoint_strategyqa_{model_alias}_n{subset_size}_a{num_agents}_r{max_rounds}_{suffix}.json"
    return str(Path(__file__).resolve().parent / filename)


def _resolve_subset_path() -> str:
    override = os.getenv("EXP_STRATEGYQA_SUBSET_FILE", "").strip()
    if not override:
        return str(Path(__file__).resolve().parent / "strategyqa_subset.json")

    raw_path = Path(override).expanduser()
    if raw_path.is_absolute():
        return str(raw_path)

    candidates = [
        ROOT / raw_path,
        Path(__file__).resolve().parent / raw_path,
        Path.cwd() / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _write_json_atomic(path: str, payload: dict):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


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
    allow_prefix_checkpoint = os.getenv("EXP_ALLOW_PREFIX_CHECKPOINT", "0").strip().lower() in {"1", "true", "yes", "on"}
    for key, value in expected_config.items():
        if (
            allow_prefix_checkpoint
            and key == "subset_size"
            and int(cfg.get(key, 0) or 0) <= int(value or 0)
        ):
            continue
        if cfg.get(key) != value:
            print("Checkpoint config mismatch; starting a new run.")
            return None
    return payload


def _normalize_text(text: str) -> str:
    return _main_normalize_text(text)


def _select_consensus_response(response_dict: dict) -> str:
    return _main_select_consensus_response(response_dict)


def _estimate_tokens(text: str) -> int:
    return _main_estimate_tokens(text)


def _extract_method_prediction(method_result, method_name: str) -> str:
    return _main_extract_method_prediction(method_result, method_name)


def _evaluate_strategyqa_method(
    judge: FormalJudge,
    *,
    question: str,
    prediction: str,
    correct_answer,
    method: str,
    method_result: dict,
) -> tuple[dict, dict, dict]:
    judge_record = judge.evaluate(question, prediction, correct_answer, method=method)
    scoring_record = score_strategyqa_boolean(prediction, correct_answer, method=method)
    eligibility_record = method_formal_table_eligibility(
        dataset="StrategyQA",
        method=method,
        method_result=method_result,
        scoring_record=scoring_record,
        judge_record=judge_record,
    )
    return scoring_record, judge_record, eligibility_record


def _is_correct(prediction: str, target) -> bool:
    pred = _normalize_text(prediction)
    gold = _normalize_text(str(target))
    if gold in {"true", "false"}:
        if "true" in pred or "yes" in pred:
            return gold == "true"
        if "false" in pred or "no" in pred:
            return gold == "false"
        return False
    return gold in pred or pred in gold


def _judge_with_llm(client, model: str, question: str, prediction: str, target) -> bool:
    return _judge_with_llm_record(client, model, question, prediction, target)["correct"]


def _judge_with_llm_record(client, model: str, question: str, prediction: str, target) -> dict:
    prompt_version = "strategyqa_correctness_v1_original_runner"
    if client is None:
        return {
            "correct": _is_correct(prediction, target),
            "judge_model": model,
            "judge_prompt_version": prompt_version,
            "judge_raw_response": None,
            "judge_fallback_used": True,
            "judge_error": None,
        }
    prompt = f"""
Evaluate if the answer to a yes/no question is correct.

Question: {question}
Gold answer: {target}
Candidate answer: {prediction}

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
            "judge_prompt_version": prompt_version,
            "judge_raw_response": data,
            "judge_fallback_used": False,
            "judge_error": None,
        }
    except Exception as exc:
        return {
            "correct": _is_correct(prediction, target),
            "judge_model": model,
            "judge_prompt_version": prompt_version,
            "judge_raw_response": None,
            "judge_fallback_used": True,
            "judge_error": str(exc),
        }


def _run_method_safely(method_name: str, runner, default_result):
    return _main_run_method_safely(method_name, runner, default_result)


def _status_counts(status_log: dict) -> dict:
    return _main_status_counts(status_log)


def _configured_methods(extra_baselines: list[str] | None = None) -> list[str]:
    return _main_configured_methods(extra_baselines)


def _empty_stats_dict(methods: list[str]) -> dict:
    return _main_empty_stats_dict(methods)


def _empty_correctness_dict(methods: list[str]) -> dict:
    return _main_empty_correctness_dict(methods)

def run_experiment(subset_size=3, num_agents=3, max_rounds=3, use_llm_judge=True, run_ablation=True):
    """Runs the experiment on a small subset of StrategyQA."""
    _load_project_env()
    subset_path = _resolve_subset_path()
    
    if not os.path.exists(subset_path):
        print(f"Data file not found at {subset_path}. Please run prepare_strategyqa.py first.")
        return
    
    with open(subset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)[:subset_size]

    judge = FormalJudge(dataset="StrategyQA", use_llm_judge=use_llm_judge)
    use_llm_conflict = os.getenv("EXP_USE_LLM_CONFLICT", "1").strip().lower() in {"1", "true", "yes", "on"}
    enable_early_stop = os.getenv("EXP_ENABLE_EARLY_STOP", "1").strip().lower() in {"1", "true", "yes", "on"}
    early_stop_patience = int(os.getenv("EXP_EARLY_STOP_PATIENCE", "2"))
    early_stop_min_rounds = int(os.getenv("EXP_EARLY_STOP_MIN_ROUNDS", "2"))
    method_timeout_s = int(os.getenv("EXP_METHOD_TIMEOUT_S", "0") or "0")
    extra_baselines = [
        part.strip()
        for part in os.getenv("EXP_EXTRA_BASELINES", "").split(",")
        if part.strip()
    ]
    extra_baselines = validate_extra_baselines(extra_baselines)
    methods = _configured_methods(extra_baselines)

    run_config = {
        "dataset": "StrategyQA",
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
        "run_ablation": bool(run_ablation),
        "use_llm_conflict": bool(use_llm_conflict),
        "enable_early_stop": bool(enable_early_stop),
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_rounds": int(early_stop_min_rounds),
        "method_timeout_s": int(method_timeout_s),
    }
    
    results = []
    timing = _empty_stats_dict(methods)
    correctness = _empty_correctness_dict(methods)
    status_log = _empty_stats_dict(methods)
    rounds_log = _empty_stats_dict(methods)
    token_log = _empty_stats_dict(methods)

    checkpoint_enabled = os.getenv("EXP_ENABLE_RESUME", "1").strip().lower() in {"1", "true", "yes", "on"}
    checkpoint_path = os.getenv("EXP_RESUME_FILE", "").strip() or _default_checkpoint_path(
        subset_size=subset_size,
        num_agents=num_agents,
        max_rounds=max_rounds,
        use_llm_judge=use_llm_judge,
    )
    if checkpoint_enabled:
        payload = _load_checkpoint(checkpoint_path, run_config)
        if payload:
            results = payload.get("results", [])
            timing = payload.get("timing", timing)
            correctness = payload.get("correctness", correctness)
            status_log = payload.get("status_log", status_log)
            rounds_log = payload.get("rounds_log", rounds_log)
            token_log = payload.get("token_log", token_log)
            print(f"Loaded checkpoint: {checkpoint_path}")
            print(f"Resuming from {len(results)} completed questions.")
        else:
            print("No valid checkpoint found. Starting fresh run.")

    completed_questions = {str(item.get("question", "")).strip() for item in results if item.get("question")}
    
    for idx, item in enumerate(data):
        question = item.get('question', 'Unknown question')
        correct_answer = item.get('answer', 'Unknown')
        if str(question).strip() in completed_questions:
            print(f"\n--- Question {idx+1}: already completed, skipping ---")
            continue
        print(f"\n--- Question {idx+1}: {question} ---")
        
        # 1. Vanilla Debate (Baseline)
        mv = MajorityVote()
        mv_res, elapsed = _run_method_safely(
            "Majority Vote",
            lambda: mv.run(question, num_agents=num_agents),
            {"status": "error", "final_answer": ""}
        )
        timing["majority_vote"].append(elapsed)
        mv_pred = _extract_method_prediction(mv_res, "majority_vote")
        mv_scoring, mv_judge, mv_eligibility = _evaluate_strategyqa_method(
            judge,
            question=question,
            prediction=mv_pred,
            correct_answer=correct_answer,
            method="majority_vote",
            method_result=mv_res,
        )
        if mv_scoring["correct"]:
            correctness["majority_vote"] += 1
        status_log["majority_vote"].append(mv_res.get("status", "completed"))
        rounds_log["majority_vote"].append(1)
        token_log["majority_vote"].append(_estimate_tokens(mv_pred))

        wv = WeightedVote()
        wv_res, elapsed = _run_method_safely(
            "Weighted Vote",
            lambda: wv.run(question, num_agents=num_agents),
            {"status": "error", "final_answer": ""}
        )
        timing["weighted_vote"].append(elapsed)
        wv_pred = _extract_method_prediction(wv_res, "weighted_vote")
        wv_scoring, wv_judge, wv_eligibility = _evaluate_strategyqa_method(
            judge,
            question=question,
            prediction=wv_pred,
            correct_answer=correct_answer,
            method="weighted_vote",
            method_result=wv_res,
        )
        if wv_scoring["correct"]:
            correctness["weighted_vote"] += 1
        status_log["weighted_vote"].append(wv_res.get("status", "completed"))
        rounds_log["weighted_vote"].append(1)
        token_log["weighted_vote"].append(_estimate_tokens(wv_pred))

        vd = VanillaDebate(max_rounds=max_rounds)
        vd_res, elapsed = _run_method_safely(
            "Vanilla Debate",
            lambda: vd.run(question, num_agents=num_agents),
            {"status": "fallback", "round": max_rounds, "result": {}}
        )
        timing["vanilla_debate"].append(elapsed)
        vanilla_pred = _extract_method_prediction(vd_res, "vanilla_debate")
        vanilla_scoring, vanilla_judge, vanilla_eligibility = _evaluate_strategyqa_method(
            judge,
            question=question,
            prediction=vanilla_pred,
            correct_answer=correct_answer,
            method="vanilla_debate",
            method_result=vd_res,
        )
        if vanilla_scoring["correct"]:
            correctness["vanilla_debate"] += 1
        status_log["vanilla_debate"].append(vd_res.get("status", "fallback"))
        rounds_log["vanilla_debate"].append(int(vd_res.get("round", max_rounds)))
        token_log["vanilla_debate"].append(_estimate_tokens(vanilla_pred))
        print(f"Vanilla Debate Status: {vd_res.get('status', 'fallback')}")

        sc_res = None
        sc_judge = None
        if "self_consistency" in extra_baselines:
            sc_samples = int(os.getenv("EXP_SELF_CONSISTENCY_SAMPLES", str(max(num_agents, num_agents * (max_rounds + 1)))))
            sc = SelfConsistency()
            sc_res, elapsed = _run_method_safely(
                "Self Consistency",
                lambda: sc.run(question, num_samples=sc_samples),
                {"status": "error", "final_answer": ""}
            )
            timing["self_consistency"].append(elapsed)
            sc_pred = _extract_method_prediction(sc_res, "self_consistency")
            sc_scoring, sc_judge, sc_eligibility = _evaluate_strategyqa_method(
                judge,
                question=question,
                prediction=sc_pred,
                correct_answer=correct_answer,
                method="self_consistency",
                method_result=sc_res,
            )
            if sc_scoring["correct"]:
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
            stj_scoring, stj_judge, stj_eligibility = _evaluate_strategyqa_method(
                judge,
                question=question,
                prediction=stj_pred,
                correct_answer=correct_answer,
                method="scalar_transcript_judge",
                method_result=stj_res,
            )
            if stj_scoring["correct"]:
                correctness["scalar_transcript_judge"] += 1
            status_log["scalar_transcript_judge"].append(stj_res.get("status", "completed"))
            rounds_log["scalar_transcript_judge"].append(int(stj_res.get("round", max_rounds)))
            token_log["scalar_transcript_judge"].append(_estimate_tokens(stj_pred))
            print(f"Scalar Transcript Judge: {stj_res.get('final_answer', '')[:50]}...")
        
        # 2. ArgDebate (Full)
        agents_full = [LLMAgent(f"agent_{i}") for i in range(num_agents)]
        manager_full = ArgDebateManager(
            agents_full,
            config={
                "max_rounds": max_rounds,
                "use_llm_conflict": use_llm_conflict,
                "enable_early_stop": enable_early_stop,
                "early_stop_patience": early_stop_patience,
                "early_stop_min_rounds": early_stop_min_rounds,
            },
        )
        ad_full_res, elapsed = _run_method_safely(
            "ArgDebate (Full)",
            lambda: manager_full.resolve(question),
            {"status": "fallback", "round": max_rounds, "result": {}}
        )
        timing["arg_debate"].append(elapsed)
        arg_pred = _extract_method_prediction(ad_full_res, "arg_debate")
        arg_scoring, arg_judge, arg_eligibility = _evaluate_strategyqa_method(
            judge,
            question=question,
            prediction=arg_pred,
            correct_answer=correct_answer,
            method="arg_debate",
            method_result=ad_full_res,
        )
        if arg_scoring["correct"]:
            correctness["arg_debate"] += 1
        status_log["arg_debate"].append(ad_full_res.get("status", "fallback"))
        rounds_log["arg_debate"].append(int(ad_full_res.get("round", max_rounds)))
        token_log["arg_debate"].append(_estimate_tokens(arg_pred))
        print(f"ArgDebate (Full) Status: {ad_full_res.get('status', 'fallback')}")
        
        # 3. ArgDebate (Ablation: No DF-QuAD)
        ad_abl_res = None
        if run_ablation:
            manager_abl = make_ablation_manager(
                "no_dfquad",
                num_agents,
                config={
                    "max_rounds": max_rounds,
                    "use_llm_conflict": use_llm_conflict,
                    "enable_early_stop": enable_early_stop,
                    "early_stop_patience": early_stop_patience,
                    "early_stop_min_rounds": early_stop_min_rounds,
                },
            )
            ad_abl_res, _ = _run_method_safely(
                "ArgDebate (Ablation)",
                lambda: manager_abl.resolve(question),
                {"status": "fallback", "round": max_rounds, "result": {}}
            )
            print(f"ArgDebate (Ablation: No DF-QuAD) Status: {ad_abl_res.get('status', 'fallback')}")
        
        # Store results
        row_payload = {
            "question": question,
            "correct_answer": correct_answer,
            "majority_vote": mv_res,
            "weighted_vote": wv_res,
            "vanilla_debate": vd_res,
            "arg_debate_full": ad_full_res,
            "arg_debate_no_dfquad": ad_abl_res,
            "correctness": {
                "majority_vote": bool(mv_scoring["correct"]),
                "weighted_vote": bool(wv_scoring["correct"]),
                "vanilla_debate": bool(vanilla_scoring["correct"]),
                "arg_debate": bool(arg_scoring["correct"]),
            },
            "correctness_scoring": {
                "majority_vote": mv_scoring,
                "weighted_vote": wv_scoring,
                "vanilla_debate": vanilla_scoring,
                "arg_debate": arg_scoring,
            },
            "correctness_judge": {
                "policy": judge.policy_summary(),
                "majority_vote": mv_judge,
                "weighted_vote": wv_judge,
                "vanilla_debate": vanilla_judge,
                "arg_debate": arg_judge,
            },
            "formal_table_eligibility": {
                "majority_vote": mv_eligibility,
                "weighted_vote": wv_eligibility,
                "vanilla_debate": vanilla_eligibility,
                "arg_debate": arg_eligibility,
            },
        }
        if sc_res is not None and sc_judge is not None:
            row_payload["self_consistency"] = sc_res
            row_payload["correctness"]["self_consistency"] = bool(sc_scoring["correct"])
            row_payload["correctness_scoring"]["self_consistency"] = sc_scoring
            row_payload["correctness_judge"]["self_consistency"] = sc_judge
            row_payload["formal_table_eligibility"]["self_consistency"] = sc_eligibility
        if stj_res is not None and stj_judge is not None:
            row_payload["scalar_transcript_judge"] = stj_res
            row_payload["correctness"]["scalar_transcript_judge"] = bool(stj_scoring["correct"])
            row_payload["correctness_scoring"]["scalar_transcript_judge"] = stj_scoring
            row_payload["correctness_judge"]["scalar_transcript_judge"] = stj_judge
            row_payload["formal_table_eligibility"]["scalar_transcript_judge"] = stj_eligibility
        results.append(row_payload)

        if checkpoint_enabled:
            _write_json_atomic(
                checkpoint_path,
                {
                    "config": run_config,
                    "updated_at": datetime.now().isoformat(),
                    "results": results,
                    "timing": timing,
                    "correctness": correctness,
                    "status_log": status_log,
                    "rounds_log": rounds_log,
                    "token_log": token_log,
                },
            )
            print(f"Checkpoint updated: {len(results)}/{len(data)} questions completed")
        
    # Save results
    timestamp = os.getenv("EXP_OUTPUT_STAMP", "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(os.getenv("EXP_OUTPUT_DIR", "").strip() or os.path.dirname(__file__)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / f"results_ablation_{timestamp}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nExperiment results saved to {output_path}")

    summary = {
        "dataset": "StrategyQA",
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
        "status_counts": _status_counts(status_log),
        "method_timeout_count": sum(1 for statuses in status_log.values() for status in statuses if status == "timeout"),
        "avg_rounds": {
            k: (sum(v) / max(len(v), 1)) for k, v in rounds_log.items()
        },
        "token_efficiency_estimated": {
            k: (sum(token_log[k]) / max(correctness[k], 1)) for k in token_log
        },
        "formal_table_eligibility": summarize_formal_table_eligibility(results, methods),
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
    subset_size = int(os.getenv("EXP_SUBSET_SIZE", "2"))
    num_agents = int(os.getenv("EXP_NUM_AGENTS", "3"))
    max_rounds = int(os.getenv("EXP_MAX_ROUNDS", "3"))
    use_llm_judge = os.getenv("EXP_USE_LLM_JUDGE", "1").strip().lower() in {"1", "true", "yes", "on"}
    run_ablation = os.getenv("EXP_RUN_ABLATION", "1").strip().lower() in {"1", "true", "yes", "on"}
    run_experiment(
        subset_size=subset_size,
        num_agents=num_agents,
        max_rounds=max_rounds,
        use_llm_judge=use_llm_judge,
        run_ablation=run_ablation,
    )
