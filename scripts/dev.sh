#!/usr/bin/env bash
# MMDE-Studio development launcher: backend (uvicorn --reload) + frontend (vite dev)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${MMDE_PYTHON:-/home/zhy/miniconda3/envs/lunarecon/bin/python}"
export MMDE_PYTHON="$PY"   # task subprocesses must not fall back to /root
# 8000 is taken by another service on this host (returns 401); 8010 is MMDE-Studio's
BACKEND_PORT="${BACKEND_PORT:-8010}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

cleanup() {
  kill "$BACK_PID" "$FRONT_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[dev] backend  -> http://localhost:$BACKEND_PORT (uvicorn --reload)"
"$PY" -m uvicorn --factory app.main:create_app --app-dir backend --host 0.0.0.0 --port "$BACKEND_PORT" --reload &
BACK_PID=$!

cd frontend
echo "[dev] frontend -> http://localhost:$FRONTEND_PORT (vite, /api proxied to :$BACKEND_PORT)"
npm run dev -- --port "$FRONTEND_PORT" --host &
FRONT_PID=$!

wait
