"""Model taxonomy contract: config/models.yaml drives grouped dropdowns."""
import fnmatch


def test_taxonomy_endpoint(client):
    r = client.get("/api/model-taxonomy")
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["excluded"], list) and body["excluded"]
    ids = [c["id"] for c in body["categories"]]
    for want in ("mtd", "ptc", "tta", "metric", "other"):
        assert want in ids
    labels = {c["id"]: c["label"] for c in body["categories"]}
    assert labels["metric"] == "度量模型"


def _matches(name, patterns):
    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def test_taxonomy_classifies_real_models(client):
    """Every discovered model is either excluded or hits exactly one category
    rule; relative-depth models must land in excluded."""
    tax = client.get("/api/model-taxonomy").json()
    reg = client.get("/api/registry").json()
    models = {m for d in reg["datasets"] for s in d["splits"] for m in s["models"]}
    assert models, "no models discovered"
    for m in models:
        if m.startswith(("dav2_rel", "zipdepth", "gemdepth")) and not m.startswith(
                ("mtd_", "ptc_", "tta_")):
            assert _matches(m, tax["excluded"]), f"{m} should be excluded"
            continue
        hits = [c["id"] for c in tax["categories"] if _matches(m, c["match"])]
        assert hits, f"{m} matches no category"


def test_taxonomy_file_missing_falls_back(client, monkeypatch, tmp_path):
    from app import registry
    from app.config import get_settings
    monkeypatch.setattr(get_settings(), "models_config_path", tmp_path / "nope.yaml")
    registry.reset_taxonomy_cache()
    try:
        body = client.get("/api/model-taxonomy").json()
        assert body["categories"][0]["match"] == ["*"]
        assert body["excluded"] == []
    finally:
        registry.reset_taxonomy_cache()
