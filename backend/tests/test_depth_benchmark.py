"""End-to-end viewer contract for the KITTI/nuScenes depth benchmark.

Run with MMDE_STUDIO_DATASETS pointing at the benchmark datasets.yaml.  This
test is intentionally separate from the bundled lunar-demo contract tests.
"""
from __future__ import annotations

import pytest


MODELS = {
    "unidepth_v2", "moge3", "ptc_dav2", "mtd_dav2", "manydepth2_vel",
}
SPLITS = {
    "kitti": {
        "eigen_test": 697,
        "official_test_anonymous": 1000,
        "sequence_seed20261006_01": 50,
        "sequence_seed20261006_02": 50,
        "sequence_seed20261006_03": 50,
    },
    "nuscenes": {
        "official_test": 6008,
        "val_official": 6019,
        "sequence_seed20261006_01": 40,
        "sequence_seed20261006_02": 40,
        "sequence_seed20261006_03": 40,
    },
}

EVALUABLE_SPLITS = {
    dataset: {split for split in splits if split != "official_test_anonymous"}
    for dataset, splits in SPLITS.items()
}


def test_benchmark_registry_and_frame_counts(client):
    response = client.get("/api/registry")
    assert response.status_code == 200
    datasets = {item["name"]: item for item in response.json()["datasets"]}
    assert set(SPLITS) <= set(datasets)
    for dataset, expected in SPLITS.items():
        found = {item["name"]: item for item in datasets[dataset]["splits"]}
        assert set(found) == set(expected)
        for split, count in expected.items():
            assert found[split]["n_frames"] == count
            assert found[split]["test_type"] == (
                "test_sequence" if split.startswith("sequence_") else "test_single")
            assert set(found[split]["models"]) == MODELS
            assert found[split]["evaluation_available"] is (
                split != "official_test_anonymous")


@pytest.mark.parametrize(
    "dataset,split,count",
    [(dataset, split, count) for dataset, splits in SPLITS.items()
     for split, count in splits.items()],
)
def test_all_frames_and_model_points_are_viewable(client, dataset, split, count):
    frames = client.get("/api/frames", params={"dataset": dataset,
                                               "split": split})
    assert frames.status_code == 200
    assert len(frames.json()) == count
    assert client.get(f"/api/rgb/{dataset}/{split}/0?w=320").status_code == 200
    gt = client.get(f"/api/gt/{dataset}/{split}/0")
    assert gt.status_code == 200
    if split == "official_test_anonymous":
        assert gt.content == b"\x00\x00\x00\x00"
    for model in sorted(MODELS):
        response = client.get(f"/api/points/{dataset}/{split}/0/{model}")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert len(response.content) > 16


def test_all_evaluations_reach_viewer_summary(client):
    response = client.get("/api/metrics/summary")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    summary = body["summary"]["datasets"]
    for dataset, splits in EVALUABLE_SPLITS.items():
        assert set(summary[dataset]["splits"]) == splits
        for split in splits:
            assert set(summary[dataset]["splits"][split]["models"]) == MODELS


def test_metrics_page_contract_for_depth_benchmark(client):
    response = client.get("/api/metrics/nuscenes/official_test")
    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 5
    assert {row["model"] for row in rows} == MODELS
    for row in rows:
        assert row["dataset"] == "nuscenes"
        assert row["split"] == "official_test"
        assert row["num_frames"] == 6008
        assert row["gt_range_m"] == [0.1, 80.0]
        assert "abs_rel" in row["aggregate"]
        assert "abs_mean_m" in row["aggregate"]
        assert isinstance(row["per_frame"], list)
        assert set(row["depth_bins"]) == {"0.1-10", "10-20", "20-40", "40-80"}


def test_anonymous_kitti_exposes_method_limitations(client):
    response = client.get("/api/registry")
    assert response.status_code == 200
    datasets = {item["name"]: item for item in response.json()["datasets"]}
    splits = {item["name"]: item for item in datasets["kitti"]["splits"]}
    availability = splits["official_test_anonymous"]["model_availability"]
    assert availability["ptc_dav2"]["available"] is False
    assert "temporal" in availability["ptc_dav2"]["reason"]
    assert availability["manydepth2_vel"]["mode"] == (
        "monocular_branch_no_temporal_context")


def test_continuous_scene_results_are_viewable(client):
    response = client.get("/api/scenes")
    assert response.status_code == 200
    found = {(row["dataset"], row["split"]): set(row["models"])
             for row in response.json()}
    expected_splits = {
        (dataset, f"sequence_seed20261006_{ordinal:02d}")
        for dataset in ("kitti", "nuscenes") for ordinal in range(1, 4)
    }
    assert set(found) == expected_splits
    for dataset, split in sorted(expected_splits):
        expected_models = MODELS
        if dataset == "nuscenes" and split.endswith("_01"):
            expected_models = MODELS - {"ptc_dav2"}
        assert found[(dataset, split)] == expected_models
        for model in sorted(expected_models):
            index_response = client.get(
                f"/api/scene/{dataset}/{split}/{model}/index.json")
            assert index_response.status_code == 200
            index = index_response.json()
            assert index["format"] == 2
            assert index["n_frames"] in (40, 50)
            assert index["n_points"] > 0
            assert len(index["cam_pos"]) == index["n_frames"]
            assert len(index["cam_quat"]) == index["n_frames"]
            assert len(index["cam_t"]) == index["n_frames"]
            assert index["groups"] and index["groups"][0]["chunks"]
        first_model = sorted(expected_models)[0]
        first_index = client.get(
            f"/api/scene/{dataset}/{split}/{first_model}/index.json").json()
        overview = first_index["groups"][0]["overview"]
        blob = client.get(
            f"/api/scene/{dataset}/{split}/{first_model}/blob/{overview}")
        assert blob.status_code == 200
        assert len(blob.content) > 16


def test_all_invalid_sequence_is_explained(client):
    response = client.get("/api/registry")
    datasets = {item["name"]: item for item in response.json()["datasets"]}
    splits = {item["name"]: item for item in datasets["nuscenes"]["splits"]}
    availability = splits["sequence_seed20261006_01"]["model_availability"]["ptc_dav2"]
    assert availability["available"] is False
    assert availability["mode"] == "all_frames_explicitly_invalid"
    assert "fused scene unavailable" in availability["reason"]
