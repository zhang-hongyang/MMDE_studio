"""Subprocess runner: argv execution with merged line-streamed logging.

The process runs in its own session (``start_new_session=True``) so cancel
can kill the whole tree via ``os.killpg``. stdout/stderr are merged; every
line is appended to the log table (seq auto-increments) and broadcast to
WebSocket subscribers. Never uses a shell.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal

from . import catalog

log = logging.getLogger(__name__)


def kill_process_group(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


async def run(queue, task: dict) -> None:
    """Execute one queued task to completion, updating store + subscribers."""
    store = queue.store
    task_id = task["id"]
    proc: asyncio.subprocess.Process | None = None
    try:
        spec = catalog.get_spec(task["type"])
        if spec is None:
            raise catalog.ParamError(f"unknown task type: {task['type']}")
        argv = catalog.build_argv(spec, task["params"])
        if not os.path.isfile(argv[0]):
            raise catalog.ParamError(f"script not found: {argv[0]}")
        cmd = ([spec.interpreter] if spec.interpreter else []) + argv
        env = os.environ.copy()
        for key, val in spec.env:
            env[key] = val
        store.mark_running(task_id)
        queue.publish(task_id, {"type": "status", "status": "running"})
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=spec.cwd or None, env=env,
            start_new_session=True)
    except Exception as exc:  # build/start failure: fail without a process
        store.append_log(task_id, f"error: {exc}")
        store.finish(task_id, "failed", -1)
        queue.publish(task_id, {"type": "final", "status": "failed",
                                "exit_code": -1})
        queue.clear_cancel(task_id)
        return

    queue.set_pid(task_id, proc.pid)
    store.set_pid(task_id, proc.pid)
    assert proc.stdout is not None
    async for raw in proc.stdout:
        line = raw.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            continue
        seq = store.append_log(task_id, line)
        queue.publish(task_id, {"type": "log", "seq": seq, "line": line})
    exit_code = await proc.wait()

    if queue.cancel_requested(task_id) or exit_code == -signal.SIGKILL:
        status = "cancelled"
    else:
        status = "succeeded" if exit_code == 0 else "failed"
    store.finish(task_id, status, exit_code)
    queue.publish(task_id, {"type": "final", "status": status,
                            "exit_code": exit_code})
    queue.clear_cancel(task_id)
    queue.forget_pid(task_id)
