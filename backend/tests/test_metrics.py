def test_metrics_lunar_nac(client):
    r = client.get("/api/metrics/lunar_nac/test_sequence")
    assert r.status_code == 200
    items = r.json()
    assert items, "lunar_nac/test_sequence metrics empty"
    for it in items:
        assert "aggregate" in it
        assert "abs_rel" in it["aggregate"]
        assert len(it["per_frame"]) == 20


def test_metrics_missing_split_404(client):
    # metrics_list resolves the split first: unknown split -> 404
    assert client.get(
        "/api/metrics/lunar_nac/no_such_split").status_code == 404


def test_metrics_summary(client):
    r = client.get("/api/metrics/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert "lunar_nac/test_sequence" in body["summary"]
    for model in body["summary"]["lunar_nac/test_sequence"].values():
        assert "abs_rel" in model
