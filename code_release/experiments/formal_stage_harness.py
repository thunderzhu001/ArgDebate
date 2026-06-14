#!/usr/bin/env python3
"""Shared harness utilities for auditable formal experiment stages."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from experiments.formal_table_eligibility import formal_table_ineligible_count


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_env_file(root: Path = ROOT, *, override: bool = False) -> None:
    env_path = root / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and (override or key not in os.environ):
            os.environ[key] = value
    if os.getenv("OPENAI_API_BASE") and not os.getenv("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_API_BASE", "")


def parse_saved_paths(stdout: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {"json_path": None, "md_path": None}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("saved json:"):
            out["json_path"] = line.split("saved json:", 1)[1].strip()
        elif line.startswith("saved md:"):
            out["md_path"] = line.split("saved md:", 1)[1].strip()
    return out


def latest_file(directory: Path, pattern: str, before: set[str]) -> Path | None:
    after = {p.name for p in directory.glob(pattern)}
    delta = sorted(after - before)
    if not delta:
        return None
    return directory / delta[-1]


def find_archived_checkpoint(work_dir: Path) -> Path | None:
    candidates = sorted(work_dir.glob("checkpoint.done_*.json"))
    if candidates:
        return candidates[-1]
    checkpoint = work_dir / "checkpoint.json"
    if checkpoint.exists():
        return checkpoint
    return None


def count_internal_timeouts(value: Any) -> int:
    if isinstance(value, dict):
        count = 0
        if value.get("status") == "timeout":
            count += 1
        if str(value.get("error", "")).startswith("method_timeout_after_"):
            count += 1
        for child in value.values():
            count += count_internal_timeouts(child)
        return count
    if isinstance(value, list):
        return sum(count_internal_timeouts(item) for item in value)
    return 0


def count_internal_errors(value: Any) -> int:
    if isinstance(value, dict):
        count = 1 if value.get("status") == "error" else 0
        for child in value.values():
            count += count_internal_errors(child)
        return count
    if isinstance(value, list):
        return sum(count_internal_errors(item) for item in value)
    return 0


def _summary_status_error_count(summary: Any) -> int:
    if not isinstance(summary, dict):
        return 0
    status_counts = summary.get("status_counts", {})
    if not isinstance(status_counts, dict):
        return 0
    error_count = 0
    for counts in status_counts.values():
        if isinstance(counts, dict):
            error_count += int(counts.get("error", 0) or 0)
    return error_count


def artifact_check(
    summary: Any,
    results: Any,
    *,
    include_summary_status_errors: bool = False,
) -> dict[str, Any]:
    timeout_count = count_internal_timeouts(results)
    error_count = count_internal_errors(results)
    if isinstance(summary, dict):
        timeout_count += int(summary.get("method_timeout_count", 0) or 0)
        if include_summary_status_errors and error_count == 0:
            error_count = _summary_status_error_count(summary)
    ineligible_count = formal_table_ineligible_count(summary)
    ok = timeout_count == 0 and error_count == 0 and ineligible_count == 0
    return {
        "ok": ok,
        "method_timeout_count": timeout_count,
        "method_error_count": error_count,
        "formal_table_ineligible_count": ineligible_count,
        "reason": "ok"
        if ok
        else (
            "internal_method_timeout"
            if timeout_count
            else ("internal_method_error" if error_count else "formal_table_ineligible")
        ),
    }


def run_python_subprocess_with_timeout(
    *,
    script: Path,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout_s: int,
    log_mode: str = "w",
    resume_message: str | None = None,
) -> dict[str, Any]:
    started = time.time()
    timed_out = False
    returncode = -1
    with open(log_path, log_mode, encoding="utf-8") as log_f:
        if resume_message:
            log_f.write(f"\n[resume] {datetime.now().isoformat()} {resume_message}\n")
            log_f.flush()
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", str(script)],
                cwd=str(cwd),
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                text=True,
            )
            while True:
                polled = proc.poll()
                if polled is not None:
                    returncode = int(polled)
                    break
                if time.time() - started > timeout_s:
                    timed_out = True
                    proc.kill()
                    returncode = int(proc.wait(timeout=30))
                    break
                time.sleep(2)
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = -9
    return {
        "returncode": returncode,
        "elapsed_s": round(time.time() - started, 3),
        "timed_out": timed_out,
    }
