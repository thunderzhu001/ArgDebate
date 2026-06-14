import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.formal_run_spec import (
    ROOT,
    build_stage_b_plan,
    dataset_run_path,
    ensure_stage_b_dataset_capacity,
)


SPEC_PATH = ROOT / "experiments" / "config" / "formal_stage_b_pilot.json"


def _load_spec():
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_stage_b_plan_has_expected_runnable_and_skip_counts():
    spec = _load_spec()
    plan = build_stage_b_plan(spec, smoke_size=1, python_executable="/usr/bin/python3")
    assert len(plan["jobs"]) == 9
    assert len(plan["skip_records"]) == 1
    assert plan["skip_records"][0]["job_id"] == "main_strategyqa_qwen35_flash_n1"
    assert "timed out after 420s" in plan["skip_records"][0]["reason"]


def test_stage_b_plan_uses_spec_dataset_paths_as_execution_env():
    spec = _load_spec()
    plan = build_stage_b_plan(
        spec,
        phase="main",
        dataset="truthfulqa",
        models=["deepseek_v4_flash"],
        smoke_size=1,
        python_executable="/usr/bin/python3",
    )
    assert len(plan["jobs"]) == 1
    job = plan["jobs"][0]
    expected_path = str(dataset_run_path(spec, "truthfulqa"))
    assert job["dataset_path"] == expected_path
    assert job["env"]["EXP_TRUTHFULQA_SUBSET_FILE"] == expected_path


def test_stage_b_filter_selects_deepseek_main_modules():
    spec = _load_spec()
    plan = build_stage_b_plan(
        spec,
        phase="main",
        models=["deepseek_v4_flash"],
        smoke_size=1,
        python_executable="/usr/bin/python3",
    )
    job_ids = sorted(job["job_id"] for job in plan["jobs"])
    assert job_ids == [
        "main_strategyqa_deepseek_v4_flash_n1",
        "main_truthfulqa_deepseek_v4_flash_n1",
    ]
    assert plan["skip_records"] == []


def test_stage_b_dataset_capacity_uses_job_dataset_paths():
    spec = _load_spec()
    plan = build_stage_b_plan(
        spec,
        phase="main",
        dataset="strategyqa",
        models=["deepseek_v4_flash"],
        smoke_size=1,
        python_executable="/usr/bin/python3",
    )
    capacity = ensure_stage_b_dataset_capacity(spec, plan["jobs"], prepare=False, seed=None)
    expected_path = str(dataset_run_path(spec, "strategyqa"))
    assert capacity["required"][f"strategyqa:{expected_path}"] == 1
    assert f"strategyqa:{expected_path}" in capacity["actual"]
    assert isinstance(capacity["ok"], bool)


if __name__ == "__main__":
    test_stage_b_plan_has_expected_runnable_and_skip_counts()
    test_stage_b_plan_uses_spec_dataset_paths_as_execution_env()
    test_stage_b_filter_selects_deepseek_main_modules()
    test_stage_b_dataset_capacity_uses_job_dataset_paths()
    print("formal run spec tests passed")
