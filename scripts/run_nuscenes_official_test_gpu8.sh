#!/usr/bin/env bash
set -euo pipefail

ROOT=${MMDE_BENCHMARK_ROOT:?set MMDE_BENCHMARK_ROOT}
CONFIG=${MMDE_STUDIO_DATASETS:?set MMDE_STUDIO_DATASETS}
RELEASE=${MMDE_RELEASE:?set MMDE_RELEASE}
PY=/home/ZhangHongyang/miniconda3/envs/mmde/bin/python
DATASET=nuscenes
SPLIT=official_test
EXPECTED=6008
SHARDS=4
GPU_COUNT=4
MODELS=(unidepth_v2 moge3 ptc_dav2 mtd_dav2 manydepth2_vel)

mkdir -p "$ROOT/logs" "$ROOT/status"

run_shards() {
  local method=$1
  local shard_kind=$2
  local shard_count=${3:-$SHARDS}
  local output="$ROOT/mmde_result/preds/$DATASET/$SPLIT/$method"
  local -a pids=()
  mkdir -p "$output"
  local existing
  existing=$(find "$output" -maxdepth 1 -type f -name '*.npy' | wc -l)
  if [[ "$existing" == "$EXPECTED" ]]; then
    echo "reusing $EXPECTED existing $method predictions" >&2
    printf '0\n' >"$ROOT/status/${DATASET}-${SPLIT}-${method}.exitcode"
    return 0
  fi
  for ((offset = 0; offset < shard_count; offset++)); do
    local -a shard_args
    if [[ "$shard_kind" == frame ]]; then
      shard_args=(--frame-offset "$offset" --frame-stride "$shard_count")
    else
      shard_args=(--group-offset "$offset" --group-stride "$shard_count")
    fi
    (
      export CUDA_VISIBLE_DEVICES=$((offset % GPU_COUNT))
      if [[ "$method" == ptc_dav2 ]]; then
        # PTC's CPU-side geometry can otherwise let every process spawn a
        # full BLAS/OpenMP pool. Bound per-shard threads on the 128-core host.
        export OMP_NUM_THREADS=2
        export MKL_NUM_THREADS=2
        export OPENBLAS_NUM_THREADS=2
        export NUMEXPR_NUM_THREADS=2
        export MMDE_OPENCV_THREADS=1
      fi
      "$PY" "$RELEASE/methods/run_${method}.py" \
        --dataset "$DATASET" --split "$SPLIT" --device cuda \
        "${shard_args[@]}"
    ) >"$ROOT/logs/${DATASET}-${SPLIT}-${method}-shard${offset}of${shard_count}.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  [[ "$failed" == 0 ]] || return 1
  local count
  count=$(find "$output" -maxdepth 1 -type f -name '*.npy' | wc -l)
  [[ "$count" == "$EXPECTED" ]] || {
    echo "$method produced $count/$EXPECTED predictions" >&2
    return 1
  }
  printf '0\n' >"$ROOT/status/${DATASET}-${SPLIT}-${method}.exitcode"
}

run_frame() { run_shards "$1" frame; }
run_group() { run_shards "$1" group; }

run_frame unidepth_v2
run_frame moge3
run_shards mtd_dav2 frame 12
run_shards ptc_dav2 group 24
run_group manydepth2_vel

"$PY" "$RELEASE/scripts/evaluate_mmde_depth.py" \
  --datasets-config "$CONFIG" --dataset "$DATASET" --split "$SPLIT" \
  --models "${MODELS[@]}" \
  >"$ROOT/logs/${DATASET}-${SPLIT}-evaluation.log" 2>&1
printf '0\n' >"$ROOT/status/${DATASET}-${SPLIT}-evaluation.exitcode"
printf '0\n' >"$ROOT/status/${DATASET}-${SPLIT}-all.exitcode"
echo "NUSCENES_OFFICIAL_TEST_COMPLETE $(date -Is)"
