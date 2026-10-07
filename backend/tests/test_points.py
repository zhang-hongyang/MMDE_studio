import struct

import numpy as np


def read_u32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def test_models_listed(client):
    r = client.get("/api/models/lunar_nac/test_sequence")
    assert r.status_code == 200
    models = r.json()
    assert models and models == sorted(models)
    assert "dav2_small" in models


def test_points_binary_layout(client):
    models = client.get("/api/models/lunar_nac/test_sequence").json()
    model = models[0]
    r = client.get(f"/api/points/lunar_nac/test_sequence/0/{model}")
    assert r.status_code == 200
    assert r.headers["X-Format-Version"] == "1"
    buf = r.content
    hp, wp, step = struct.unpack_from("<III", buf, 0)
    assert step == 2  # default stride
    n = hp * wp
    expect = 12 + n * 4 * 2 + n * 3
    assert len(buf) == expect, f"{len(buf)} != {expect}"
    # raster order: depth_raw finite somewhere, rgb bytes present
    d_raw = np.frombuffer(buf, np.float32, count=n, offset=12)
    assert np.isfinite(d_raw).any()
    d_al = np.frombuffer(buf, np.float32, count=n, offset=12 + n * 4)
    assert d_al.shape == (n,)
    # affine-aligned depth tracks the sparse DTM GT (same hm scale)
    finite = np.isfinite(d_al)
    assert finite.any()
    assert np.nanmin(d_al[finite]) > 0.0
    rgb = np.frombuffer(buf, np.uint8, count=n * 3, offset=12 + n * 8)
    assert rgb.size == n * 3


def test_points_unknown_model_404(client):
    r = client.get("/api/points/lunar_nac/test_sequence/0/no_such_model")
    assert r.status_code == 404


def test_points_calib_param_accepted(client):
    # lunar_nac frames have no calib_* fields; ?calib= falls through to
    # depth_gt_path and must still serve the frame
    models = client.get("/api/models/lunar_nac/test_sequence").json()
    r = client.get(
        f"/api/points/lunar_nac/test_sequence/0/{models[0]}?calib=seq")
    assert r.status_code == 200
    assert len(r.content) > 12


def test_gt_lunar_nac_has_points(client):
    r = client.get("/api/gt/lunar_nac/test_sequence/0")
    assert r.status_code == 200
    buf = r.content
    n = read_u32(buf, 0)
    assert n > 0
    assert len(buf) == 4 + n * 12
    d = np.frombuffer(buf, np.float32, count=n, offset=4 + 8 * n)
    # depths are elevations in hm, offset so they stay inside (MIN, MAX)
    assert np.isfinite(d).all()
    assert (d > 0.1).all() and (d < 200.0).all()


def test_control_sources_lunar_nac(client):
    r = client.get("/api/control_sources?dataset=lunar_nac&split=test_sequence")
    assert r.status_code == 200
    sources = r.json()
    assert sources == ["dtm"]


def test_controls_binary_layout(client):
    r = client.get("/api/controls/lunar_nac/test_sequence/0/dtm")
    assert r.status_code == 200
    buf = r.content
    n = read_u32(buf, 0)
    assert n > 0
    assert len(buf) == 4 + n * 4 * 6, f"{len(buf)} != {4 + n * 24}"


def test_controls_with_model_and_err(client):
    models = client.get("/api/models/lunar_nac/test_sequence").json()
    r = client.get(
        f"/api/controls/lunar_nac/test_sequence/0/dtm?model={models[0]}")
    assert r.status_code == 200
    buf = r.content
    n = read_u32(buf, 0)
    assert len(buf) == 4 + n * 24
    d_pred = np.frombuffer(buf, np.float32, count=n, offset=4 + n * 12)
    err_gt = np.frombuffer(buf, np.float32, count=n, offset=4 + n * 20)
    assert np.isfinite(d_pred).any()
    # controls sit exactly on GT samples -> per-point GT error is defined
    assert np.isfinite(err_gt).any()


def test_controls_unknown_source_404(client):
    r = client.get("/api/controls/lunar_nac/test_sequence/0/no_such_source")
    assert r.status_code == 404
