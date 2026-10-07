"""Dataset registry endpoint: one call feeds every frontend dropdown."""
from __future__ import annotations

from fastapi import APIRouter

from ..registry import get_model_taxonomy, get_registry

router = APIRouter()


@router.get("/api/registry")
def registry_overview():
    reg = get_registry()
    datasets = []
    for name in reg.names():
        entry = reg.get_dataset(name)
        splits = [
            {
                "name": s.name,
                "test_type": s.test_type,
                "n_frames": s.n_frames,
                "evaluation_available": s.evaluation_available,
                "cameras": s.cameras,
                "control_sources": s.control_sources,
                "models": s.models,
                "model_availability": s.model_availability,
                "scene_models": s.scene_models,
            }
            for s in reg.splits(entry).values()
        ]
        datasets.append({
            "name": entry.name,
            "platform": entry.platform,
            "title": entry.title,
            "calib_protocols": list(entry.calib_protocols),
            "splits": splits,
        })
    return {"datasets": datasets}


@router.get("/api/model-taxonomy")
def model_taxonomy():
    """Model classification rules (config/models.yaml) for grouped dropdowns."""
    t = get_model_taxonomy()
    return {"excluded": t["excluded"], "categories": t["categories"]}
