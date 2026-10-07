"""Task center endpoints: catalog, submit, status, logs, live WS, cancel."""
from __future__ import annotations

import asyncio

from fastapi import (APIRouter, HTTPException, Query, WebSocket,
                     WebSocketDisconnect)
from pydantic import BaseModel

from ..tasks import catalog
from ..tasks.queue import TERMINAL, get_queue
from .common import check_name

router = APIRouter()


class SubmitBody(BaseModel):
    type: str
    params: dict = {}


def _queue():
    return get_queue()


def _task_or_404(task_id: str) -> dict:
    check_name(task_id, "task")
    task = _queue().store.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="unknown task")
    return task


@router.get("/api/tasks/catalog")
def task_catalog():
    """Task types + parameter schemas for the submission wizard."""
    return [catalog.spec_to_json(s) for s in catalog.all_specs()]


@router.get("/api/tasks")
def list_tasks(status: str | None = None,
               limit: int = Query(100, ge=1, le=1000)):
    if status is not None and status not in \
            ("queued", "running", "succeeded", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail="invalid status")
    return _queue().store.list_tasks(status=status, limit=limit)


@router.post("/api/tasks", status_code=201)
async def submit_task(body: SubmitBody):
    spec = catalog.get_spec(body.type)
    if spec is None:
        raise HTTPException(status_code=404, detail="unknown task type")
    try:
        params = catalog.validate_params(spec, body.params)
    except catalog.ParamError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    return await _queue().submit(spec.type, params)


@router.get("/api/tasks/{task_id}")
def task_detail(task_id: str):
    task = _task_or_404(task_id)
    task["logs"] = _queue().store.recent_logs(task_id, 50)
    return task


@router.get("/api/tasks/{task_id}/log")
def task_log(task_id: str, after: int = Query(0, ge=0)):
    _task_or_404(task_id)
    return {"lines": _queue().store.get_logs(task_id, after=after)}


@router.post("/api/tasks/{task_id}/cancel")
def cancel_task(task_id: str):
    _task_or_404(task_id)
    if not _queue().cancel(task_id):
        raise HTTPException(status_code=409,
                            detail="task not queued or running")
    return _queue().store.get_task(task_id)


@router.websocket("/ws/tasks/{task_id}")
async def task_ws(ws: WebSocket, task_id: str):
    await ws.accept()
    store = _queue().store
    task = store.get_task(task_id)
    if task is None:
        await ws.send_json({"type": "error", "detail": "unknown task"})
        await ws.close()
        return
    queue = _queue()
    sub = queue.subscribe(task_id)
    try:
        await ws.send_json({"type": "status", "task_id": task_id,
                            "status": task["status"]})
        for row in store.get_logs(task_id, after=0):
            await ws.send_json({"type": "log", "seq": row["seq"],
                                "line": row["line"]})
        if task["status"] in TERMINAL:  # late joiner: close immediately
            await ws.send_json({"type": "final", "status": task["status"],
                                "exit_code": task["exit_code"]})
            await ws.close()
            return
        while True:
            msg = await sub.get()
            await ws.send_json(msg)
            if msg.get("type") == "final":
                # let the client read the outcome, then close
                try:
                    await asyncio.sleep(5)
                    await ws.close()
                except (WebSocketDisconnect, RuntimeError):
                    pass
                return
    except WebSocketDisconnect:
        pass
    finally:
        queue.unsubscribe(task_id, sub)
