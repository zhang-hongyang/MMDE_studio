#!/usr/bin/env bash
set -euo pipefail

gpu=${1:?GPU index}
offset=${2:?frame offset}
stride=${3:?frame stride}
root=${MMDE_BENCHMARK_ROOT:?set MMDE_BENCHMARK_ROOT}
config=${MMDE_STUDIO_DATASETS:?set MMDE_STUDIO_DATASETS}
release=${MMDE_RELEASE:?set MMDE_RELEASE}
id="nuscenes-val_official-mtd_dav2-shard${offset}of${stride}"
marker="$root/status/$id.exitcode"
log="$root/logs/$id.log"

if [[ -f "$marker" && $(cat "$marker") == 0 ]]; then
  exit 0
fi

set +e
CUDA_VISIBLE_DEVICES="$gpu" MMDE_STUDIO_DATASETS="$config" \
  /home/ZhangHongyang/miniconda3/envs/mmde/bin/python \
  "$release/methods/run_mtd_dav2.py" --dataset nuscenes \
  --split val_official --device cuda --frame-offset "$offset" \
  --frame-stride "$stride" >"$log" 2>&1
code=$?
set -e
printf '%s\n' "$code" >"$marker"
exit "$code"
