"""FastAPI entry point for the MMDE-Studio backend.

Usage (from backend/):
    /root/miniconda3/envs/dggt/bin/python -m app.main --host 0.0.0.0 --port 8000
    /root/miniconda3/envs/dggt/bin/python -m app.main --config /path/to/datasets.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .config import BACKEND_DIR, configure
from . import data_access
from .registry import UnknownDatasetError, UnknownSplitError
from .routers import frames, metrics, points, registry, scenes, tasks

log = logging.getLogger(__name__)

FRONTEND_DIST = BACKEND_DIR.parent / "frontend" / "dist"


def create_app(config_path: str | Path | None = None,
               stride: int | None = None) -> FastAPI:
    configure(config_path, stride)
    data_access.configure_caches()

    app = FastAPI(title="MMDE-Studio API", docs_url=None, redoc_url=None)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_credentials=False,
        allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def format_version(request, call_next):
        resp = await call_next(request)
        resp.headers["X-Format-Version"] = "1"
        return resp

    @app.exception_handler(UnknownDatasetError)
    async def _unknown_dataset(request, exc):
        return JSONResponse(status_code=404, content={"detail": "unknown dataset"})

    @app.exception_handler(UnknownSplitError)
    async def _unknown_split(request, exc):
        return JSONResponse(status_code=404, content={"detail": "unknown split"})

    for r in (registry.router, frames.router, points.router, scenes.router,
              metrics.router, tasks.router):
        app.include_router(r)

    @app.get("/api/health")
    def health():
        return {"ok": True}

    # static frontend (if built) with SPA fallback -- tolerant of a missing dist
    if FRONTEND_DIST.is_dir():
        @app.get("/{full_path:path}", include_in_schema=False)
        def spa(full_path: str):
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="not found")
            f = FRONTEND_DIST / full_path
            if full_path and f.is_file():
                return FileResponse(f)
            index = FRONTEND_DIST / "index.html"
            if index.is_file():
                return FileResponse(index)
            raise HTTPException(status_code=404, detail="not found")
    else:
        @app.get("/", include_in_schema=False)
        def no_frontend():
            return JSONResponse({"detail": "frontend not built; API under /api"})

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser(description="MMDE-Studio FastAPI backend")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--config", default=None,
                    help="datasets yaml (default: bundled config, or "
                         "$MMDE_STUDIO_DATASETS)")
    ap.add_argument("--stride", type=int, default=None,
                    help="points subsample stride (default 2)")
    args = ap.parse_args()

    app = create_app(args.config, args.stride)
    import uvicorn
    print(f"MMDE-Studio backend on http://{args.host}:{args.port} "
          f"(config={get_settings().config_path}); Ctrl-C to stop", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


from .config import get_settings  # noqa: E402  (late import for main())

if __name__ == "__main__":
    main()
