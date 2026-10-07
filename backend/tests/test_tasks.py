"""Task center tests (M3): an echo task registered by the test suite only.

Covers the full lifecycle against a scratch SQLite DB: submit -> poll ->
logs -> incremental log -> cancel -> catalog -> websocket push. Real mmde
scripts are discovered for the catalog but never executed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import create_app  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.tasks import catalog  # noqa: E402
from app.tasks import queue as task_queue  # noqa: E402
from app.tasks import store as task_store  # noqa: E402

ECHO_SCRIPT = """\
import argparse
import time
ap = argparse.ArgumentParser()
ap.add_argument("--lines", type=int, default=3)
ap.add_argument("--delay", type=float, default=0.0)
args = ap.parse_args()
for i in range(args.lines):
    print(f"echo line {i}", flush=True)
    if args.delay:
        time.sleep(args.delay)
print("done", flush=True)
"""


@pytest.fixture(scope="module")
def _echo_spec(tmp_path_factory):
    script = tmp_path_factory.mktemp("tasks") / "echo_task.py"
    script.write_text(ECHO_SCRIPT)
    catalog.register_extra(catalog.TaskSpec(
        type="echo",
        title="Echo (test only)",
        description="Fast echo task for the test suite; never runs real "
                    "mmde scripts.",
        params=(
            catalog.ParamSpec("lines", "Lines", "int", "--lines",
                              default=3),
            catalog.ParamSpec("delay", "Delay", "float", "--delay",
                              default=0.0),
        ),
        script=str(script),
        interpreter=sys.executable,
    ))
    yield
    catalog.clear_extras()


@pytest.fixture()
def client(tmp_path, monkeypatch, _echo_spec):
    app = create_app()  # rebuilds settings with the default config
    monkeypatch.setattr(get_settings(), "tasks_db", tmp_path / "tasks.db")
    task_store.reset_store()
    task_queue.reset_queue()
    with TestClient(app) as c:
        yield c
    task_queue.reset_queue()
    task_store.reset_store()


def wait_status(client, task_id, want, timeout=15.0, need_pid=False):
    deadline = time.time() + timeout
    task = {}
    while time.time() < deadline:
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] in want and (not need_pid or task["pid"]):
            return task
        time.sleep(0.05)
    raise AssertionError(
        f"task {task_id} stuck in {task.get('status')!r}, wanted {want}")


def test_catalog_lists_builtin_types(client):
    entries = client.get("/api/tasks/catalog").json()
    types = {e["type"] for e in entries}
    # each builtin type appears only when its script exists on this host
    # (the original deployment had /home/MMDE; this one may not)
    assert ("inference" in types) == catalog.METHODS_DIR.is_dir()
    assert ("eval" in types) == catalog.EVAL_SCRIPT.is_file()
    assert ("fuse_scene" in types) == catalog.FUSE_SCRIPT.is_file()
    for entry in entries:
        assert entry["title"] and entry["description"]
        for p in entry["params"]:
            assert {"name", "label", "type", "required",
                    "default", "help"} <= set(p)
    if "inference" in types:
        inference = next(e for e in entries if e["type"] == "inference")
        method = next(p for p in inference["params"] if p["name"] == "method")
        assert method["choices"]
    # the test-only echo spec never leaks into the production catalog
    assert all(s.type != "echo" for s in catalog.base_specs())


def test_method_wrappers_declare_contract_flags(client):
    """Every discovered run_<method>.py must declare the flags the catalog
    may pass (--dataset/--split at minimum, extras are filtered by what the
    script declares)."""
    methods = catalog._discover_methods()
    assert methods, "no methods discovered"
    for m in methods:
        script = catalog.METHODS_DIR / f"run_{m}.py"
        assert script.is_file(), f"missing wrapper {script}"
        flags = catalog._script_flags(str(script))
        assert "--dataset" in flags, f"method {m}: wrapper lacks --dataset"
        assert "--split" in flags, f"method {m}: wrapper lacks --split"
    assert "marigold_v2" in methods


def test_submit_and_poll_to_success(client):
    r = client.post("/api/tasks",
                    json={"type": "echo",
                          "params": {"lines": 3, "delay": 0.02}})
    assert r.status_code == 201
    task = r.json()
    assert task["status"] == "queued"
    tid = task["id"]

    t = wait_status(client, tid, {"succeeded", "failed"})
    assert t["status"] == "succeeded"
    assert t["exit_code"] == 0
    lines = [l["line"] for l in t["logs"]]
    assert lines == ["echo line 0", "echo line 1", "echo line 2", "done"]

    # incremental log: after=<seq> returns only newer lines
    full = client.get(f"/api/tasks/{tid}/log").json()["lines"]
    assert [l["seq"] for l in full] == [1, 2, 3, 4]
    inc = client.get(f"/api/tasks/{tid}/log",
                     params={"after": 2}).json()["lines"]
    assert [(l["seq"], l["line"]) for l in inc] \
        == [(3, "echo line 2"), (4, "done")]


def test_cancel_running_task(client):
    r = client.post("/api/tasks",
                    json={"type": "echo",
                          "params": {"lines": 100, "delay": 0.2}})
    tid = r.json()["id"]
    t = wait_status(client, tid, {"running", "succeeded", "failed"},
                    need_pid=True)
    assert t["status"] == "running"

    rc = client.post(f"/api/tasks/{tid}/cancel")
    assert rc.status_code == 200
    t = wait_status(client, tid, {"cancelled", "succeeded", "failed"})
    assert t["status"] == "cancelled"
    assert t["exit_code"] not in (0, None)
    logs = client.get(f"/api/tasks/{tid}/log").json()["lines"]
    assert len(logs) < 100  # killed well before printing all lines


def test_submit_validation(client):
    assert client.post("/api/tasks",
                       json={"type": "nope"}).status_code == 404
    assert client.post("/api/tasks",
                       json={"type": "echo",
                             "params": {"bogus": 1}}).status_code == 400
    assert client.post("/api/tasks",
                       json={"type": "echo",
                             "params": {"lines": "abc"}}).status_code == 400
    # name-injection attempts against a builtin type (when one is present)
    if catalog.FUSE_SCRIPT.is_file():
        r = client.post("/api/tasks",
                        json={"type": "fuse_scene",
                              "params": {"dataset": "..", "split": "x"}})
        assert r.status_code == 400
        r = client.post("/api/tasks",
                        json={"type": "fuse_scene",
                              "params": {"dataset": "not_in_registry",
                                         "split": "x"}})
        assert r.status_code == 400


def test_list_filter_and_unknown(client):
    tid = client.post("/api/tasks",
                      json={"type": "echo", "params": {"lines": 1}}) \
        .json()["id"]
    wait_status(client, tid, {"succeeded", "failed"})
    tasks = client.get("/api/tasks").json()
    assert tasks[0]["id"] == tid  # newest first
    assert client.get("/api/tasks",
                      params={"status": "succeeded"}).json()[0]["id"] == tid
    assert client.get("/api/tasks", params={"limit": 0}).status_code == 422
    assert client.get("/api/tasks/unknown").status_code == 404
    # cancelling a finished task is a conflict
    assert client.post(f"/api/tasks/{tid}/cancel").status_code == 409


def test_websocket_late_joiner_gets_history_and_final(client):
    tid = client.post("/api/tasks",
                      json={"type": "echo", "params": {"lines": 2}}) \
        .json()["id"]
    wait_status(client, tid, {"succeeded", "failed"})
    with client.websocket_connect(f"/ws/tasks/{tid}") as ws:
        msgs = [ws.receive_json() for _ in range(5)]
    assert [m["type"] for m in msgs] \
        == ["status", "log", "log", "log", "final"]
    assert [m["line"] for m in msgs[1:4]] \
        == ["echo line 0", "echo line 1", "done"]
    assert msgs[-1]["status"] == "succeeded"


def test_websocket_live_push(client):
    tid = client.post("/api/tasks",
                      json={"type": "echo",
                            "params": {"lines": 5, "delay": 0.1}}) \
        .json()["id"]
    with client.websocket_connect(f"/ws/tasks/{tid}") as ws:
        msgs = []
        while True:
            msg = ws.receive_json()
            msgs.append(msg)
            if msg["type"] == "final":
                break  # close client-side; skip the server's 5s hold
    assert msgs[0]["type"] == "status"
    logs = [m["line"] for m in msgs if m["type"] == "log"]
    assert logs == [f"echo line {i}" for i in range(5)] + ["done"]
    assert msgs[-1]["status"] == "succeeded"
    assert msgs[-1]["exit_code"] == 0
