#!/usr/bin/env bash
# MMDE-Studio production launcher: build frontend, serve everything from FastAPI (single port)
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${MMDE_PYTHON:-/home/zhy/miniconda3/envs/lunarecon/bin/python}"
export MMDE_PYTHON="$PY"   # task subprocesses must not fall back to /root
# 8000 is taken by another service on this host (returns 401); 8010 is MMDE-Studio's
PORT="${PORT:-8010}"

echo "[build] frontend"
(cd frontend && npm ci && npm run build)

echo "[start] http://localhost:$PORT"
exec "$PY" -m uvicorn --factory app.main:create_app --app-dir backend --host 0.0.0.0 --port "$PORT"
