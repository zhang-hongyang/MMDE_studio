#!/usr/bin/env python3
"""Download only the checkpoints required by the six MMDE method entries."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(os.environ.get(
    "MMDE_MODEL_ROOT",
    "/home/ZhangHongyang/ResearchHub-scratch/models/MMDE_studio"))
ROOT.mkdir(parents=True, exist_ok=True)


def one(repo: str, filename: str, subdir: str) -> Path:
    target = ROOT / subdir
    target.mkdir(parents=True, exist_ok=True)
    local = target / filename
    if local.is_file() and local.stat().st_size > 0:
        return local
    return Path(hf_hub_download(repo_id=repo, filename=filename,
                                local_dir=target))


def mapanything_snapshot() -> Path:
    target = ROOT / "map-anything"
    required = ("config.json", "model.safetensors", "README.md")
    if all((target / name).is_file() and (target / name).stat().st_size > 0
           for name in required):
        return target
    return Path(snapshot_download(
        repo_id="facebook/map-anything", local_dir=target,
        allow_patterns=list(required)))


def main() -> int:
    resolved = {
        "moge3": str(one("Ruicheng/moge-3-vitl", "model.pt", "moge3")),
        "mapanything": str(mapanything_snapshot()),
        "lingbot_map": str(one(
            "robbyant/lingbot-map", "lingbot-map.pt", "lingbot-map")),
        "dav2": str(one(
            "depth-anything/Depth-Anything-V2-Large",
            "depth_anything_v2_vitl.pth", "depth-anything-v2")),
    }
    manifest = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "files": resolved,
        "sizes": {key: (Path(path).stat().st_size if Path(path).is_file()
                         else sum(p.stat().st_size for p in Path(path).rglob("*")
                                  if p.is_file()))
                  for key, path in resolved.items()},
    }
    (ROOT / "required-models.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
