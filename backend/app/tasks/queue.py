"""In-process asyncio task queue over the SQLite store.

Concurrency is bounded by ``Settings.tasks_concurrency`` (default 1). The
scheduler is created lazily on first submit inside the running event loop.
Live log/final events are pushed to subscriber queues (one per WebSocket).
"""
from __future__ import annotations

import asyncio
import logging
import uuid

from ..config import get_settings
from . import runner
from .store import get_store

log = logging.getLogger(__name__)

TERMINAL = ("succeeded", "failed", "cancelled")


class TaskQueue:
    def __init__(self):
        self.store = get_store()
        self._sem: asyncio.Semaphore | None = None
        self._pending: asyncio.Queue | None = None
        self._worker: asyncio.Task | None = None
        self._subs: dict[str, set[asyncio.Queue]] = {}
        self._pids: dict[str, int] = {}
        self._cancel_req: set[str] = set()

    # -- scheduling --------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._sem is None:
            self._sem = asyncio.Semaphore(get_settings().tasks_concurrency)
            self._pending = asyncio.Queue()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def submit(self, type_: str, params: dict) -> dict:
        self._ensure_started()
        task_id = uuid.uuid4().hex[:12]
        self.store.create_task(task_id, type_, params)
        await self._pending.put(task_id)
        return self.store.get_task(task_id)

    async def _run(self) -> None:
        while True:
            task_id = await self._pending.get()
            asyncio.create_task(self._dispatch(task_id))

    async def _dispatch(self, task_id: str) -> None:
        async with self._sem:
            task = self.store.get_task(task_id)
            if task is None or task["status"] != "queued":
                return  # cancelled while waiting
            await runner.run(self, task)

    # -- cancel ------------------------------------------------------------

    def cancel(self, task_id: str) -> bool:
        task = self.store.get_task(task_id)
        if task is None:
            return False
        if task["status"] == "queued":
            self.store.finish(task_id, "cancelled", None)
            self.publish(task_id,
                         {"type": "final", "status": "cancelled",
                          "exit_code": None})
            return True
        if task["status"] == "running":
            self._cancel_req.add(task_id)
            pid = self._pids.get(task_id) or task.get("pid")
            if pid:
                runner.kill_process_group(int(pid))
            return True
        return False

    def cancel_requested(self, task_id: str) -> bool:
        return task_id in self._cancel_req

    def clear_cancel(self, task_id: str) -> None:
        self._cancel_req.discard(task_id)

    # -- running process bookkeeping (used by runner) -----------------------

    def set_pid(self, task_id: str, pid: int) -> None:
        self._pids[task_id] = pid

    def forget_pid(self, task_id: str) -> None:
        self._pids.pop(task_id, None)

    # -- pub/sub for WebSocket subscribers ----------------------------------

    def subscribe(self, task_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subs.setdefault(task_id, set()).add(q)
        return q

    def unsubscribe(self, task_id: str, q: asyncio.Queue) -> None:
        self._subs.get(task_id, set()).discard(q)

    def publish(self, task_id: str, msg: dict) -> None:
        for q in self._subs.get(task_id, ()):
            q.put_nowait(msg)


_queue: TaskQueue | None = None


def get_queue() -> TaskQueue:
    global _queue
    if _queue is None:
        _queue = TaskQueue()
    return _queue


def reset_queue() -> None:
    """Drop the singleton (test isolation)."""
    global _queue
    _queue = None
