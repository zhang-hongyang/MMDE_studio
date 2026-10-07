"""Runtime settings for the MMDE-Studio backend.

A single process-wide Settings instance is built by ``configure()`` (from the
``--config`` CLI flag or the MMDE_STUDIO_DATASETS env var, falling back to the
bundled config/datasets.yaml) and consumed everywhere else via get_settings().
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = BACKEND_DIR / "config" / "datasets.yaml"
DEFAULT_MODELS_CONFIG = BACKEND_DIR / "config" / "models.yaml"


@dataclass
class Settings:
    config_path: Path
    stride: int = 2
    cache_dir: Path = BACKEND_DIR / ".cache"
    disk_cache_max_bytes: int = 2 * 1024**3
    rgb_cache_items: int = 32
    models_config_path: Path = DEFAULT_MODELS_CONFIG
    # task queue (M3)
    tasks_concurrency: int = 1
    tasks_db: Path = BACKEND_DIR / ".cache" / "tasks.db"


_settings: Settings | None = None


def configure(config_path: str | Path | None = None,
              stride: int | None = None) -> Settings:
    """(Re)build the global settings; resets the dataset registry cache."""
    global _settings
    if config_path is None:
        config_path = os.environ.get("MMDE_STUDIO_DATASETS") or DEFAULT_CONFIG
    _settings = Settings(config_path=Path(config_path),
                         stride=stride if stride else 2,
                         models_config_path=Path(
                             os.environ.get("MMDE_STUDIO_MODELS") or DEFAULT_MODELS_CONFIG))
    from . import registry
    registry.reset()
    return _settings


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        configure()
    return _settings
