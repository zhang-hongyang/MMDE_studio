def test_scenes_non_empty(client):
    r = client.get("/api/scenes")
    assert r.status_code == 200
    scenes = r.json()
    assert scenes, "no scenes discovered"
    combo = {(s["dataset"], s["split"]) for s in scenes}
    assert ("lunar_nac", "test_sequence") in combo
    for s in scenes:
        assert s["models"]


def test_scene_models(client):
    r = client.get("/api/scene_models/lunar_nac/test_sequence")
    assert r.status_code == 200
    models = r.json()
    assert "dav2_small" in models
    assert models == sorted(models)


def test_scene_index(client):
    r = client.get("/api/scene/lunar_nac/test_sequence/dav2_small/index.json")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    # no browser caching for the enriched index
    assert "no-cache" in r.headers.get("cache-control", "")
    idx = r.json()
    assert "cam_pos" in idx and "bbox" in idx
    assert idx["n_frames"] == 20
    assert len(idx["groups"]) == 20  # one group per region
    assert len(idx["cam_pos"]) == 20
    for g in idx["groups"]:
        assert g["overview"].startswith("overview_s")
        assert g["n_points"] > 0


def test_scene_blob_overview(client):
    r = client.get(
        "/api/scene/lunar_nac/test_sequence/dav2_small/blob/overview_s0.bin")
    assert r.status_code == 200
    assert len(r.content) > 4
    assert r.headers.get("cache-control") == "public, max-age=3600"


def test_scene_blob_rejects_bad_name(client):
    r = client.get(
        "/api/scene/lunar_nac/test_sequence/dav2_small/blob/..%2Findex.json")
    assert r.status_code in (404, 422)
    r = client.get(
        "/api/scene/lunar_nac/test_sequence/dav2_small/blob/evil.bin")
    assert r.status_code == 404


def test_unknown_scene_404(client):
    assert client.get(
        "/api/scene/lunar_nac/test_sequence/no_model/index.json"
    ).status_code == 404
