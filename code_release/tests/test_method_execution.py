import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.method_execution import run_method_safely


def test_run_method_safely_success():
    result, elapsed = run_method_safely(
        "ok",
        lambda: {"status": "completed", "final_answer": "A"},
        {"status": "error", "final_answer": ""},
        timeout_s=0,
    )
    assert result["status"] == "completed"
    assert result["final_answer"] == "A"
    assert elapsed >= 0


def test_run_method_safely_catches_exception():
    def fail():
        raise RuntimeError("boom")

    result, _ = run_method_safely(
        "fail",
        fail,
        {"status": "error", "final_answer": ""},
        timeout_s=0,
    )
    assert result["status"] == "error"
    assert result["error"] == "boom"


def test_run_method_safely_signal_timeout():
    result, elapsed = run_method_safely(
        "slow",
        lambda: time.sleep(2),
        {"status": "error", "final_answer": ""},
        timeout_s=1,
        timeout_mode="signal",
    )
    assert result["status"] == "timeout"
    assert result["error"] == "method_timeout_after_1s"
    assert elapsed < 2


if __name__ == "__main__":
    test_run_method_safely_success()
    test_run_method_safely_catches_exception()
    test_run_method_safely_signal_timeout()
    print("method execution tests passed")
