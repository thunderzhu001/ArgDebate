#!/usr/bin/env python3
"""Method-level execution harness for formal experiment runners."""

from __future__ import annotations

import multiprocessing
import os
import queue
import signal
import time
from collections.abc import Callable
from typing import Any


class MethodTimeout(Exception):
    pass


def method_timeout_result(method_name: str, timeout_s: int, default_result: dict[str, Any]) -> dict[str, Any]:
    print(f"{method_name} timed out after {timeout_s}s")
    return {**default_result, "error": f"method_timeout_after_{timeout_s}s", "status": "timeout"}


def _timeout_handler(signum: int, frame: Any) -> None:
    raise MethodTimeout()


def run_method_in_process(
    method_name: str,
    runner: Callable[[], Any],
    default_result: dict[str, Any],
    timeout_s: int,
) -> dict[str, Any] | None:
    if "fork" not in multiprocessing.get_all_start_methods():
        return None

    ctx = multiprocessing.get_context("fork")
    result_queue = ctx.Queue(maxsize=1)

    def _target(q: Any) -> None:
        try:
            q.put({"ok": True, "result": runner()})
        except Exception as exc:
            q.put({"ok": False, "error": str(exc)})

    proc = ctx.Process(target=_target, args=(result_queue,), daemon=True)
    proc.start()
    payload = None
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            payload = result_queue.get(timeout=min(0.2, max(0.01, deadline - time.time())))
            break
        except queue.Empty:
            if not proc.is_alive():
                break

    if payload is None and proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive() and hasattr(proc, "kill"):
            proc.kill()
            proc.join(5)
        return method_timeout_result(method_name, timeout_s, default_result)

    proc.join(10)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
    if payload is None:
        try:
            payload = result_queue.get_nowait()
        except Exception as exc:
            return {**default_result, "error": f"method_process_no_result: {exc}", "status": "error"}

    if payload.get("ok"):
        return payload.get("result", default_result)
    print(f"{method_name} failed: {payload.get('error')}")
    return {**default_result, "error": payload.get("error", "method_process_error"), "status": "error"}


def run_method_safely(
    method_name: str,
    runner: Callable[[], Any],
    default_result: dict[str, Any],
    *,
    timeout_s: int | None = None,
    timeout_mode: str | None = None,
) -> tuple[Any, float]:
    started = time.time()
    effective_timeout = int(timeout_s if timeout_s is not None else os.getenv("EXP_METHOD_TIMEOUT_S", "0") or "0")
    effective_mode = (timeout_mode or os.getenv("EXP_METHOD_TIMEOUT_MODE", "signal")).strip().lower()

    if effective_timeout > 0 and effective_mode in {"process", "subprocess"}:
        result = run_method_in_process(method_name, runner, default_result, effective_timeout)
        if result is not None:
            return result, time.time() - started

    old_handler = None
    try:
        if effective_timeout > 0 and hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            if hasattr(signal, "siginterrupt"):
                signal.siginterrupt(signal.SIGALRM, True)
            signal.alarm(effective_timeout)
        result = runner()
    except MethodTimeout:
        result = method_timeout_result(method_name, effective_timeout, default_result)
    except Exception as exc:
        print(f"{method_name} failed: {exc}")
        result = {**default_result, "error": str(exc), "status": default_result.get("status", "error")}
    finally:
        if effective_timeout > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
    return result, time.time() - started
