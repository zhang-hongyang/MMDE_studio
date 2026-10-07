"""SQLite persistence for the task queue (tasks + per-task log lines).

The DB lives at ``Settings.tasks_db`` (default ``backend/.cache/tasks.db``).
Opening the store marks any task left ``running`` by a previous process as
``failed`` ("interrupted by restart") -- the queue itself is in-process and
does not survive restarts.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path

from ..config import get_settings

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks(
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  params TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at REAL,
  started_at REAL,
  finished_at REAL,
  exit_code INTEGER,
  pid INTEGER
);
CREATE TABLE IF NOT EXISTS task_logs(
  task_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  line TEXT NOT NULL,
  ts REAL,
  PRIMARY KEY(task_id, seq)
);
"""

STATUSES = ("queued", "running", "succeeded", "failed", "cancelled")


def _row_to_task(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "type": row["type"],
            "params": json.loads(row["params"]), "status": row["status"],
            "created_at": row["created_at"], "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "exit_code": row["exit_code"], "pid": row["pid"]}


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.executescript(SCHEMA)
            now = time.time()
            stale = self._conn.execute(
                "SELECT id FROM tasks WHERE status = 'running'").fetchall()
            for row in stale:
                log.warning("marking task %s failed: interrupted by restart",
                            row["id"])
                self._conn.execute(
                    "UPDATE tasks SET status='failed', finished_at=?, "
                    "exit_code=-1 WHERE id=?", (now, row["id"]))
                self._append_log(row["id"], "interrupted by restart", now)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- tasks -------------------------------------------------------------

    def create_task(self, task_id: str, type_: str, params: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO tasks(id, type, params, status, created_at) "
                "VALUES(?,?,?,?,?)",
                (task_id, type_, json.dumps(params), "queued", time.time()))

    def get_task(self, task_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        return _row_to_task(row) if row else None

    def list_tasks(self, status: str | None = None,
                   limit: int = 100) -> list[dict]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM tasks WHERE status=? "
                    "ORDER BY created_at DESC, id DESC LIMIT ?",
                    (status, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM tasks ORDER BY created_at DESC, "
                    "id DESC LIMIT ?", (limit,)).fetchall()
        return [_row_to_task(r) for r in rows]

    def mark_running(self, task_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE tasks SET status='running', started_at=? WHERE id=?",
                (time.time(), task_id))

    def set_pid(self, task_id: str, pid: int) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE tasks SET pid=? WHERE id=?",
                               (pid, task_id))

    def finish(self, task_id: str, status: str, exit_code: int | None) -> None:
        assert status in STATUSES
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE tasks SET status=?, finished_at=?, exit_code=? "
                "WHERE id=?", (status, time.time(), exit_code, task_id))

    # -- logs --------------------------------------------------------------

    def _append_log(self, task_id: str, line: str, ts: float) -> int:
        with self._conn:
            self._conn.execute(
                "INSERT INTO task_logs(task_id, seq, line, ts) "
                "SELECT ?, COALESCE(MAX(seq), 0) + 1, ?, ? "
                "FROM task_logs WHERE task_id=?",
                (task_id, line, ts, task_id))
            row = self._conn.execute(
                "SELECT MAX(seq) AS s FROM task_logs WHERE task_id=?",
                (task_id,)).fetchone()
        return int(row["s"])

    def append_log(self, task_id: str, line: str) -> int:
        with self._lock:
            return self._append_log(task_id, line, time.time())

    def get_logs(self, task_id: str, after: int = 0,
                 limit: int | None = None) -> list[dict]:
        sql = ("SELECT seq, line, ts FROM task_logs WHERE task_id=? "
               "AND seq > ? ORDER BY seq")
        args: list = [task_id, after]
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [{"seq": r["seq"], "line": r["line"], "ts": r["ts"]}
                for r in rows]

    def recent_logs(self, task_id: str, n: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, line, ts FROM task_logs WHERE task_id=? "
                "ORDER BY seq DESC LIMIT ?", (task_id, n)).fetchall()
        return [{"seq": r["seq"], "line": r["line"], "ts": r["ts"]}
                for r in reversed(rows)]


_store: Store | None = None
_store_lock = threading.Lock()


def get_store() -> Store:
    global _store
    with _store_lock:
        path = Path(get_settings().tasks_db)
        if _store is None or _store.path != path:
            if _store is not None:
                _store.close()
            _store = Store(path)
        return _store


def reset_store() -> None:
    """Drop the singleton (test isolation)."""
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None
