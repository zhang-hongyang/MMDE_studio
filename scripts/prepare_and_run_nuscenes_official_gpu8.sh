#!/usr/bin/env bash
set -euo pipefail

ROOT=${MMDE_BENCHMARK_ROOT:?set MMDE_BENCHMARK_ROOT}
CONFIG=${MMDE_STUDIO_DATASETS:?set MMDE_STUDIO_DATASETS}
RELEASE=${MMDE_RELEASE:?set MMDE_RELEASE}
DATA_ROOT=${NUSCENES_GPU_ROOT:-/home/ZhangHongyang/ResearchHub-scratch/datasets/MMDE_studio/nuscenes-v1.0-test-official-20261007}
PY=/home/ZhangHongyang/miniconda3/envs/mmde/bin/python

while [[ ! -f "$DATA_ROOT/GPU_READY.json" ]]; do
  echo "$(date -Is) waiting for nuScenes GPU release" >&2
  sleep 60
done
while [[ ! -f "$ROOT/status/kitti-official_test_anonymous-all.exitcode" ]]; do
  echo "$(date -Is) waiting for KITTI official-test inference" >&2
  sleep 30
done
[[ $(cat "$ROOT/status/kitti-official_test_anonymous-all.exitcode") == 0 ]]

manifest="$ROOT/mmde_test/nuscenes/official_test/frames.jsonl"
if [[ ! -f "$manifest" ]] || [[ $(wc -l <"$manifest") != 6008 ]]; then
  "$PY" "$RELEASE/scripts/prepare_nuscenes_official_test.py" \
    --dataroot "$DATA_ROOT/release" \
    --output-root "$ROOT" --runtime-root "$ROOT" \
    >"$ROOT/logs/prepare-nuscenes-official-test.log" 2>&1
fi
[[ $(wc -l <"$manifest") == 6008 ]]

"$PY" "$RELEASE/scripts/verify_depth_benchmark.py" \
  --datasets-config "$CONFIG" \
  >"$ROOT/logs/verify-data-with-official-tests.json"

exec "$RELEASE/scripts/run_nuscenes_official_test_gpu8.sh"
