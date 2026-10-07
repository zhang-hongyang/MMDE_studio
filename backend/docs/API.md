# MMDE-Studio Backend API

FastAPI port of the legacy `pc_viewer_server.py` binary/data API, plus a
config-driven dataset registry. All responses carry the header:

```
X-Format-Version: 1
```

Base URL: `http://<host>:8000` (dev default). All endpoints are `GET`.
Binary payloads are **little-endian**. This document is the authoritative
spec for the binary buffer layouts.

Dataset paths come exclusively from `config/datasets.yaml` (or the file
pointed to by `MMDE_STUDIO_DATASETS` / `--config`); splits, cameras,
control sources, models and scenes are auto-discovered from the filesystem
with mtime-based cache invalidation.

Common errors:

- unknown dataset → `404 {"detail": "unknown dataset"}`
- unknown split → `404 {"detail": "unknown split"}`
- path param not matching `^[A-Za-z0-9_.-]+$` (or literally `.` / `..`) → `404`
- missing frame / model / blob → `404 {"detail": "..."}`

---

## Registry

### GET /api/registry

One call feeding every frontend dropdown.

```json
{
  "datasets": [
    {
      "name": "carizon",
      "platform": "vehicle",
      "title": "Carizon OEM",
      "calib_protocols": ["seq", "feye"],
      "splits": [
        {
          "name": "test_sequence",
          "n_frames": 885,
          "cameras": ["mono_undist"],
          "control_sources": [],
          "models": ["dav2_rel", "moge2", "..."],
          "scene_models": ["dav2_rel", "moge2", "..."]
        }
      ]
    }
  ]
}
```

Missing optional roots (e.g. `carizon_ccar` has no metrics/scene dirs) yield
empty lists, never an error.

---

## Frames & images

`frames.jsonl` records absolute paths from the machine that built the test set.
On a migrated deployment the backend resolves them transparently
(`data_access.resolve_frame_path`): the recorded path is used when it exists;
otherwise the tail after any `mmde_test/` prefix is retried under this
deployment's `test_root`, then `<split_dir>/<basename>` and
`<split_dir>/images/<basename>`.  Bundling images next to `frames.jsonl`
therefore needs no jsonl rewrite.

### GET /api/frames?dataset=&split=&camera=

- Defaults: `dataset=kitti`, `split=test_sequence` (legacy behavior).
- `camera` (optional) filters on the frame's `camera` field.

Returns `[{"idx": 0, "seq_id": "...", "frame_id": 69, "camera": "image_02"?}]`.
`idx` is the frame's `_index` field, falling back to the frames.jsonl line
number. `camera` is omitted when absent.

### GET /api/meta/{ds}/{split}/{idx}

```json
{"K": [[...3x3...]], "w": 1242, "h": 375,
 "seq_id": "...", "frame_id": 69, "T": [[...4x4...]]}
```

`w`/`h` are the actual image size. If the image cannot be read, they are
inferred from the first available depth prediction; if that fails too → 404.
`T` is the camera→world pose (`T_world_camera`), `null` when absent.

### GET /api/rgb/{ds}/{split}/{idx}?w=N

- No `w` (or `w=0`): original image bytes, `Content-Type: image/jpeg`,
  `Cache-Control: public, max-age=3600` (bytes are whatever the source file
  is — JPEG or PNG; magic bytes `FF D8` / `89 50 4E 47`).
- `w>0`: server-side downscaled thumbnail — the image is first resized with
  `INTER_AREA` to width `w` (aspect preserved, `round(h*w/src_w)`), then
  encoded as JPEG quality 85. Disk-cached. 4K images (carizon) are fully
  supported; the legacy cv2.remap-based pipeline crashed on them.

---

## Models, GT, points, controls

`calib` query param: `seq` | `feye` (anything else is treated as absent).

- `carizon` has no LiDAR GT; `?calib=` selects the sparse triangulation
  scheme (`seq` = sequence triangulation, `feye` = mono-feye triangulation).
- Without `calib` (or when the selected scheme is absent from the frame),
  the GT path falls back through
  `depth_gt_path -> calib_seq_path -> calib_mono_feye_path`.

### GET /api/models/{ds}/{split}

Sorted list of prediction model directory names.

### GET /api/gt/{ds}/{split}/{idx}?calib=

Sparse GT points; `application/octet-stream`, `Cache-Control: 1h`.
When the frame has no GT (carizon/test_sequence) `N=0` and the payload is
just the 4-byte header.

| offset | field  | type     | count |
|--------|--------|----------|-------|
| 0      | N      | u32      | 1     |
| 4      | u      | f32      | N     |
| 4+4N   | v      | f32      | N     |
| 4+8N   | d      | f32      | N     |

Total size: `4 + 12N` bytes.

### GET /api/points/{ds}/{split}/{idx}/{model}?calib=

Subsampled depth + color grid for the point-cloud viewer;
`application/octet-stream`, `Cache-Control: 1h`.

| offset        | field         | type | count      |
|---------------|---------------|------|------------|
| 0             | hp            | u32  | 1          |
| 4             | wp            | u32  | 1          |
| 8             | step (stride) | u32  | 1          |
| 12            | depth_raw     | f32  | hp*wp      |
| 12+4*hp*wp    | depth_aligned | f32  | hp*wp      |
| 12+8*hp*wp    | rgb           | u8   | 3*hp*wp    |

Total size: `12 + 11*hp*wp` bytes. Raster order, stride `step` (default 2,
configurable at server start).

- `depth_raw`: model prediction. Models in
  `DISP_MODELS = {dav2_rel, zipdepth, gemdepth}` store inverse depth; it is
  converted with `1/max(pred, 1e-6)`. `*_calib_*` model outputs are already
  metric depth and are **not** inverted.
- `depth_aligned`: per-frame affine fit `a*depth_raw + b` against the sparse
  GT points (least squares, in depth space; identity `a=1, b=0` when the
  frame has no GT or fewer than 10 valid correspondences).

### GET /api/control_sources?dataset=&split=

Sorted source names. Sources are subdirs of `sparse_controls/` plus the
carizon triangulation dirs (`calib_seq_triangulation`,
`calib_mono_feye_triangulation`) that live at split root; the latter resolve
per-frame npz paths from the frame's `calib_seq_path` /
`calib_mono_feye_path` field.

### GET /api/controls/{ds}/{split}/{idx}/{source}?model=&calib=

Sparse control points for diagnostics; `application/octet-stream`,
`Cache-Control: 1h`.

| offset      | field           | type | count |
|-------------|-----------------|------|-------|
| 0           | n               | u32  | 1     |
| 4           | u               | f32  | n     |
| 4+4n        | v               | f32  | n     |
| 4+8n        | d_ctrl          | f32  | n     |
| 4+12n       | w               | f32  | n     |
| 4+16n       | d_pred_aligned  | f32  | n     |
| 4+20n       | err_gt          | f32  | n     |

Total size: `4 + 24n` bytes.

- `d_pred_aligned`: same convention as `/api/points` (disparity inversion +
  per-frame affine fit) sampled bilinearly at the control `uv`;
  **NaN** where `model` is absent or has no prediction for this frame.
- `err_gt`: `|log(d_ctrl / gt)|` against the nearest GT point within 3 px
  (scipy cKDTree); **NaN** where no GT matches. Control npz without a
  `weight` key (carizon triangulation) get weight = 1.

Bilinear sampling returns **NaN** for out-of-bounds `uv`.

---

## Scenes (fused multi-frame reconstruction)

### GET /api/scenes

`[{"dataset": "...", "split": "...", "models": [...]}]` — every combo with
≥1 fused scene.

### GET /api/scene_models/{ds}/{split}

Sorted models having a `scene_root/{split}/{model}/index.json`.

### GET /api/scene/{ds}/{split}/{model}/index.json

The fused index JSON, enriched with `cam_quat` (wxyz) / `cam_t`
(seconds relative to first frame) from `frames.jsonl` when the index lacks
them. `Cache-Control: no-cache` — never browser-cached (it can gain
enrichment fields without re-fusing).

### GET /api/scene/{ds}/{split}/{model}/blob/{name}

Raw blob stream, `application/octet-stream`, `Cache-Control: public,
max-age=3600`. `name` must match:

```
^(?:overview|overview_s\d+|chunk_\d{3}|chunk_s\d+_\d{3})\.bin$
```

Blob layout (unchanged from the legacy viewer):
`u32 n | u8 rgb[3n] | pad | u16 qx[n] qy[n] qz[n]; pos = origin + q*step`.

---

## Metrics

### GET /api/metrics/summary

Reads the top-level `summary_path` from the yaml. Missing/unreadable →
`{"available": false}`. Otherwise:

```json
{"available": true, "summary": {"<dataset>/<split>": {"<model>": {"abs_rel": ..., "d1": ...}}}}
```

Non-finite floats (NaN is legitimately present in the source data) are
serialized as `null`.

### GET /api/metrics/{ds}/{split}

All `*.json` directly under `metrics_root/{split}` (subdirectories such as
`v1_backup` are excluded), parsed and returned as a list. Files that fail
to parse are skipped and logged. Missing metrics dir → `[]` (carizon_ccar).
Each element is the metrics file content, e.g.:

```json
{"model": "moge2", "dataset": "...", "split": "...",
 "protocol": "...", "aggregate": {"abs_rel": ..., "d1": ...},
 "per_frame": [...]}
```

---

## Tasks (M3 task center)

Async job system backed by a SQLite queue (`backend/.cache/tasks.db`,
configurable via `Settings.tasks_db`). Tasks execute the real mmde scripts
with `/root/miniconda3/envs/dggt/bin/python` as an argv list (never a
shell); concurrency is `Settings.tasks_concurrency` (default 1, in-process
asyncio). stdout/stderr are merged and captured line by line. Tasks left
`running` by a backend restart are marked `failed` ("interrupted by
restart") when the store is next opened.

Statuses: `queued | running | succeeded | failed | cancelled`.

### GET /api/tasks/catalog

Task types with parameter schemas for the submission wizard. A type is
listed only when its script exists on the host:

- `inference` — one of `/home/MMDE/mmde/methods/run_<method>.py`
  (methods discovered by globbing `run_*.py`); params: `method`
  (choices = discovered methods), `dataset` (choices = registry names),
  `split`, optional `max_frames` / `model_name` / `device` (a flag is only
  passed when the selected method's script declares it).
- `eval` — `/home/MMDE/mmde/eval/eval_mmde.py`;
  params: `dataset`, `split`, `models` (list, comma-separated),
  `protocol` (`mmde` | `standard`, default `mmde`).
- `fuse_scene` — `/home/MMDE/mmde/scripts/fuse_scene.py`; params:
  `dataset`, `split`, `model` (validated against the preds tree models of
  the dataset/split), optional `voxel`, `stride`, `min_depth`, `max_depth`,
  `max_frames`, `force`.

```json
[{"type": "inference", "title": "...", "description": "...",
  "params": [{"name": "method", "label": "Method", "type": "str",
              "required": true, "default": null,
              "choices": ["dav2_metric", "..."], "help": "..."}]}]
```

Param types: `str | int | float | bool | list`. Submission rejects unknown
keys, wrong types, values outside `choices`, and any name-like value
(dataset/split/model/...) that does not match `^[A-Za-z0-9_.-]+$`.

### GET /api/tasks

`?status=queued|running|succeeded|failed|cancelled&limit=N` (newest first,
default limit 100). Returns the task rows:

```json
[{"id": "...", "type": "eval", "params": {...}, "status": "succeeded",
  "created_at": ..., "started_at": ..., "finished_at": ...,
  "exit_code": 0, "pid": 12345}]
```

### POST /api/tasks

Body `{"type": "...", "params": {...}}`. Params are validated against the
catalog schema (`dataset` must be a registered dataset), then queued.
`201` with the task row; `400` on param errors, `404` on unknown type.

### GET /api/tasks/{id}

Task row plus the most recent 50 log lines under `logs`
(`[{"seq": 1, "line": "...", "ts": ...}]`).

### GET /api/tasks/{id}/log?after=<seq>

Incremental log for polling/resume: only lines with `seq > after`, in
order. `{"lines": [...]}`.

### POST /api/tasks/{id}/cancel

Kills the process group of a running task (SIGKILL via `killpg`) or marks
a queued one cancelled. `409` when the task is already finished.

### WS /ws/tasks/{id}

Live log push. On connect: a `{"type": "status"}` message, then the full
log history as `{"type": "log", "seq": n, "line": "..."}`, then live lines
as they are produced. When the task ends a
`{"type": "final", "status": "...", "exit_code": n}` message is sent and
the socket closes after 5 s. Late joiners on a finished task get status +
history + final immediately. If the task is unknown the socket is closed
with an `{"type": "error"}` message.

## Running

```bash
cd /home/MMDE/studio/backend
/root/miniconda3/envs/dggt/bin/python -m app.main --host 0.0.0.0 --port 8000
# or with an alternate dataset registry:
/root/miniconda3/envs/dggt/bin/python -m app.main --config /path/to/datasets.yaml
# or via env:  MMDE_STUDIO_DATASETS=/path/to/datasets.yaml python -m app.main
```

Tests: `cd /home/MMDE/studio && /root/miniconda3/envs/dggt/bin/python -m pytest backend/tests -q`
