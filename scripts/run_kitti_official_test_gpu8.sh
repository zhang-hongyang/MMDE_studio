#!/usr/bin/env bash
set -euo pipefail

ROOT=${MMDE_BENCHMARK_ROOT:?set MMDE_BENCHMARK_ROOT}
CONFIG=${MMDE_STUDIO_DATASETS:?set MMDE_STUDIO_DATASETS}
RELEASE=${MMDE_RELEASE:?set MMDE_RELEASE}
PY=/home/ZhangHongyang/miniconda3/envs/mmde/bin/python
DATASET=kitti
SPLIT=official_test_anonymous
EXPECTED=1000
SHARDS=4

mkdir -p "$ROOT/logs" "$ROOT/status"

run_frame_shards() {
  local method=$1
  local output="$ROOT/mmde_result/preds/$DATASET/$SPLIT/$method"
  local -a pids=()
  mkdir -p "$output"
  for offset in 0 1 2 3; do
    (
      export CUDA_VISIBLE_DEVICES=$offset
      "$PY" "$RELEASE/methods/run_${method}.py" \
        --dataset "$DATASET" --split "$SPLIT" --device cuda \
        --frame-offset "$offset" --frame-stride "$SHARDS"
    ) >"$ROOT/logs/${DATASET}-${SPLIT}-${method}-shard${offset}of${SHARDS}.log" 2>&1 &
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

run_group_shards() {
  local method=$1
  local output="$ROOT/mmde_result/preds/$DATASET/$SPLIT/$method"
  local -a pids=()
  mkdir -p "$output"
  for offset in 0 1 2 3; do
    (
      export CUDA_VISIBLE_DEVICES=$offset
      "$PY" "$RELEASE/methods/run_${method}.py" \
        --dataset "$DATASET" --split "$SPLIT" --device cuda \
        --group-offset "$offset" --group-stride "$SHARDS"
    ) >"$ROOT/logs/${DATASET}-${SPLIT}-${method}-shard${offset}of${SHARDS}.log" 2>&1 &
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

# PTC-Depth cannot recover metric scale without a temporal correspondence and
# metric pose on this anonymous selection. Preserve cardinality with explicit
# invalid sentinels instead of inventing predictions.
PTC_OUT="$ROOT/mmde_result/preds/$DATASET/$SPLIT/ptc_dav2"
"$PY" "$RELEASE/algorithms/mmde-adapters/write_unobservable.py" \
  --datasets-config "$CONFIG" --dataset "$DATASET" --split "$SPLIT" \
  --output-dir "$PTC_OUT" --device cpu \
  --reason "KITTI anonymous depth-selection frames provide no temporal correspondence or metric pose" \
  >"$ROOT/logs/${DATASET}-${SPLIT}-ptc_dav2.log" 2>&1
printf '0\n' >"$ROOT/status/${DATASET}-${SPLIT}-ptc_dav2.exitcode"

run_frame_shards unidepth_v2
run_frame_shards moge3
run_frame_shards mtd_dav2
run_group_shards manydepth2_vel

printf '0\n' >"$ROOT/status/${DATASET}-${SPLIT}-all.exitcode"
echo "KITTI_OFFICIAL_TEST_COMPLETE $(date -Is)"
