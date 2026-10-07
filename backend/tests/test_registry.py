"""Registry tests against the locally hosted lunar_nac demo dataset.

The original vehicle datasets (carizon/kitti/nuscenes under /home/data2) are
registered for portability but skipped where their roots are absent, so
contract tests target lunar_nac, which scripts/prepare_lunar_dataset.py
builds from the lunar_nac_eval outputs.
"""


def test_registry_lists_lunar_nac(client):
    r = client.get("/api/registry")
    assert r.status_code == 200
    assert r.headers["X-Format-Version"] == "1"
    names = [d["name"] for d in r.json()["datasets"]]
    assert "lunar_nac" in names


def test_lunar_nac_split_discovered(client):
    r = client.get("/api/registry")
    ds = next(d for d in r.json()["datasets"] if d["name"] == "lunar_nac")
    assert ds["platform"] == "satellite"
    split_names = [s["name"] for s in ds["splits"]]
    assert "test_sequence" in split_names
    ts = next(s for s in ds["splits"] if s["name"] == "test_sequence")
    assert ts["n_frames"] == 20
    assert ts["cameras"] == ["nac_2m"]
    assert ts["models"]  # predictions discovered
    assert {"dav2_small", "marigold_v1_1_input256"} <= set(ts["models"])
    assert ts["control_sources"] == ["dtm"]
    assert ts["scene_models"] == sorted(ts["scene_models"])
    assert ts["scene_models"]


def test_unknown_dataset_404(client):
    r = client.get("/api/frames?dataset=nope&split=test_sequence")
    assert r.status_code == 404
    assert r.json() == {"detail": "unknown dataset"}
    r = client.get("/api/meta/nope/test_sequence/0")
    assert r.status_code == 404


def test_bad_name_404(client):
    for url in ("/api/meta/bad%20name/test_sequence/0",
                "/api/points/lunar_nac/test_sequence/0/..",
                "/api/scene/lunar_nac/test_sequence/../index.json"):
        assert client.get(url).status_code == 404, url
