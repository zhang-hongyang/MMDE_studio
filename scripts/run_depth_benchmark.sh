#!/usr/bin/env bash
set -euo pipefail

ROOT=${MMDE_BENCHMARK_ROOT:?set MMDE_BENCHMARK_ROOT}
CONFIG=${MMDE_STUDIO_DATASETS:?set MMDE_STUDIO_DATASETS}
METHODS_DIR=${MMDE_METHODS_DIR:-"$(cd "$(dirname "$0")/../methods" && pwd)"}
LOG_ROOT="$ROOT/logs"
mkdir -p "$LOG_ROOT"

models=(unidepth_v2 moge3 ptc_dav2 mtd_dav2 manydepth2_vel)
datasets=(kitti nuscenes)
kitti_splits=(eigen_test sequence_seed20261006_01 sequence_seed20261006_02 sequence_seed20261006_03)
nuscenes_splits=(val_official sequence_seed20261006_01 sequence_seed20261006_02 sequence_seed20261006_03)

run_one() {
  local gpu=$1 dataset=$2 split=$3 method=$4
  local marker="$ROOT/status/${dataset}-${split}-${method}.exitcode"
  local log="$LOG_ROOT/${dataset}-${split}-${method}.log"
  local extra=()
  mkdir -p "$(dirname "$marker")"
  if [[ -f "$marker" ]] && [[ "$(cat "$marker")" == 0 ]]; then
    echo "skip completed $dataset $split $method"
    return 0
  fi
  if [[ "$dataset" == nuscenes && "$split" == val_official && \
        "$method" == mtd_dav2 && ${MMDE_MTD_SHARD_STRIDE:-1} -gt 1 ]]; then
    extra=(--frame-offset 0 --frame-stride "$MMDE_MTD_SHARD_STRIDE")
  fi
  set +e
  CUDA_VISIBLE_DEVICES="$gpu" MMDE_STUDIO_DATASETS="$CONFIG" \
    "${MMDE_RUNNER_PYTHON:-/home/ZhangHongyang/miniconda3/envs/mmde/bin/python}" \
    "$METHODS_DIR/run_${method}.py" --dataset "$dataset" --split "$split" \
    --device cuda "${extra[@]}" >"$log" 2>&1
  local code=$?
  set -e
  printf '%s\n' "$code" >"$marker"
  return "$code"
}

if [[ $# -eq 4 ]]; then
  run_one "$@"
  exit $?
fi

echo "Usage: $0 GPU DATASET SPLIT METHOD" >&2
exit 2
