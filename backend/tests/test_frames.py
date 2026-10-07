import cv2
import numpy as np


def test_frames_non_empty_with_fields(client):
    r = client.get("/api/frames?dataset=lunar_nac&split=test_sequence")
    assert r.status_code == 200
    frames = r.json()
    assert len(frames) == 20
    first = frames[0]
    assert "idx" in first and "seq_id" in first and "frame_id" in first
    assert first["camera"] == "nac_2m"
    assert len({f["seq_id"] for f in frames}) == 20  # one frame per region


def test_frames_camera_filter(client):
    all_frames = client.get(
        "/api/frames?dataset=lunar_nac&split=test_sequence").json()
    cam = all_frames[0]["camera"]
    filtered = client.get(
        "/api/frames?dataset=lunar_nac&split=test_sequence&camera="
        + cam).json()
    assert len(filtered) == len(all_frames)
    # camera with no frames -> empty list, not an error
    none = client.get(
        "/api/frames?dataset=lunar_nac&split=test_sequence&camera=NOPE").json()
    assert none == []


def test_meta_lunar_nac(client):
    r = client.get("/api/meta/lunar_nac/test_sequence/0")
    assert r.status_code == 200
    m = r.json()
    assert m["w"] > 0 and m["h"] > 0
    assert m["w"] <= 1408 or m["h"] <= 1408  # downscaled orthophoto
    K = np.asarray(m["K"])
    assert K.shape == (3, 3)
    assert m["seq_id"] == "A17SIVB"  # first region, sorted
    assert m["frame_id"] is not None
    T = np.asarray(m["T"])
    assert T.shape == (4, 4)  # geo-registered pose present


def test_rgb_original_lunar_nac(client):
    r = client.get("/api/rgb/lunar_nac/test_sequence/0")
    assert r.status_code == 200
    head = r.content[:4]
    assert head[:2] == b"\xff\xd8" or head[:4] == b"\x89PNG", head
    img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None and img.shape[0] > 0


def test_rgb_thumbnail_width(client):
    r = client.get("/api/rgb/lunar_nac/test_sequence/0?w=320")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
