import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.formal_stage_harness import (  # noqa: E402
    ROOT,
    artifact_check,
    parse_saved_paths,
    run_python_subprocess_with_timeout,
)


def test_artifact_check_fails_on_internal_timeout():
    check = artifact_check({}, [{"arg_debate": {"status": "timeout"}}])
    assert check["ok"] is False
    assert check["method_timeout_count"] == 1
    assert check["reason"] == "internal_method_timeout"


def test_artifact_check_fails_on_formal_table_ineligible():
    summary = {"formal_table_eligibility": {"ineligible_total": 1}}
    check = artifact_check(summary, [])
    assert check["ok"] is False
    assert check["formal_table_ineligible_count"] == 1
    assert check["reason"] == "formal_table_ineligible"


def test_summary_status_errors_are_counted_when_requested():
    summary = {
        "status_counts": {"self_consistency": {"error": 1}},
        "formal_table_eligibility": {"ineligible_total": 0},
    }
    default_check = artifact_check(summary, [])
    strategyqa_check = artifact_check(summary, [], include_summary_status_errors=True)
    assert default_check["ok"] is True
    assert strategyqa_check["ok"] is False
    assert strategyqa_check["method_error_count"] == 1
    assert strategyqa_check["reason"] == "internal_method_error"


def test_parse_saved_paths():
    paths = parse_saved_paths("saved json: /tmp/a.json\nsaved md: /tmp/a.md\n")
    assert paths == {"json_path": "/tmp/a.json", "md_path": "/tmp/a.md"}


def test_run_python_subprocess_with_timeout_success():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        script = tmp_path / "ok.py"
        log_path = tmp_path / "runner.log"
        script.write_text("print('done')\n", encoding="utf-8")
        result = run_python_subprocess_with_timeout(
            script=script,
            cwd=ROOT,
            env=dict(os.environ),
            log_path=log_path,
            timeout_s=5,
        )
        assert result["returncode"] == 0
        assert result["timed_out"] is False
        assert "done" in log_path.read_text(encoding="utf-8")


if __name__ == "__main__":
    test_artifact_check_fails_on_internal_timeout()
    test_artifact_check_fails_on_formal_table_ineligible()
    test_summary_status_errors_are_counted_when_requested()
    test_parse_saved_paths()
    test_run_python_subprocess_with_timeout_success()
    print("formal stage harness tests passed")
