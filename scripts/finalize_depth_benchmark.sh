#!/usr/bin/env bash
set -euo pipefail

ROOT=${MMDE_BENCHMARK_ROOT:?set MMDE_BENCHMARK_ROOT}
CONFIG=${MMDE_STUDIO_DATASETS:?set MMDE_STUDIO_DATASETS}
RELEASE=${MMDE_RELEASE:?set MMDE_RELEASE}
PY=/home/ZhangHongyang/miniconda3/envs/mmde/bin/python
MODELS=(unidepth_v2 moge3 ptc_dav2 mtd_dav2 manydepth2_vel)

echo "waiting for eight MTD shards" >&2
while true; do
  pending=0
  for offset in 0 1 2 3 4 5 6 7; do
    marker="$ROOT/status/nuscenes-val_official-mtd_dav2-shard${offset}of8.exitcode"
    if [[ ! -f "$marker" ]]; then
      pending=$((pending + 1))
    elif [[ $(cat "$marker") != 0 ]]; then
      echo "failed shard: $marker code=$(cat "$marker")" >&2
      exit 1
    fi
  done
  (( pending == 0 )) && break
  echo "$(date -Is) pending_mtd_shards=$pending" >&2
  sleep 45
done

pred="$ROOT/mmde_result/preds/nuscenes/val_official/mtd_dav2"
count=$(find "$pred" -maxdepth 1 -type f -name '*.npy' | wc -l)
[[ "$count" == 6019 ]] || { echo "MTD count $count != 6019" >&2; exit 1; }
printf '0\n' >"$ROOT/status/nuscenes-val_official-mtd_dav2.exitcode"

mtd_metrics="$ROOT/mmde_result/metrics/nuscenes/val_official/mtd_dav2.json"
if [[ -f "$mtd_metrics" ]]; then
  echo "reusing existing MTD metrics: $mtd_metrics" >&2
else
  "$PY" "$RELEASE/scripts/evaluate_mmde_depth.py" \
    --datasets-config "$CONFIG" --dataset nuscenes --split val_official \
    --models mtd_dav2
fi
"$PY" "$RELEASE/scripts/summarize_depth_benchmark.py" \
  --datasets-config "$CONFIG"
"$PY" "$RELEASE/scripts/verify_depth_benchmark.py" \
  --datasets-config "$CONFIG" --check-predictions --models "${MODELS[@]}" \
  >"$ROOT/logs/verify-predictions-final.json"

(cd "$RELEASE/backend" && MMDE_STUDIO_DATASETS="$CONFIG" \
  "$PY" -m pytest -q tests/test_depth_benchmark.py)

if [[ -f "$ROOT/status/viewer-8010.pid" ]]; then
  old=$(cat "$ROOT/status/viewer-8010.pid")
  kill "$old" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    kill -0 "$old" 2>/dev/null || break
    sleep 1
  done
fi
nohup env MMDE_STUDIO_DATASETS="$CONFIG" "$PY" -m uvicorn \
  --factory app.main:create_app --app-dir "$RELEASE/backend" \
  --host 0.0.0.0 --port 8010 >"$ROOT/logs/viewer-8010.log" 2>&1 </dev/null &
viewer_pid=$!
printf '%s\n' "$viewer_pid" >"$ROOT/status/viewer-8010.pid"
for _ in 1 2 3 4 5 6 7 8 9 10 11 12; do
  if curl -fsS http://127.0.0.1:8010/api/health >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8010/api/health >/dev/null

code_root=$(dirname "$(dirname "$RELEASE")")
current="$code_root/current"
next="$code_root/.current.next-$$"
ln -s "$RELEASE" "$next"
mv -Tf "$next" "$current"

"$PY" - "$ROOT/run_manifest.yaml" <<'PY'
import os, sys, time, yaml
from pathlib import Path
path = Path(sys.argv[1])
doc = yaml.safe_load(path.read_text())
doc["status"] = "COMPLETE"
doc["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
doc["viewer_url"] = "http://100.78.74.13:8010"
doc["prediction_integrity_gate"] = "PASSED"
doc["viewer_contract_gate"] = "PASSED"
temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
temporary.write_text(yaml.safe_dump(doc, sort_keys=False))
os.replace(temporary, path)
PY

echo "FINALIZE_COMPLETE $(date -Is) viewer_pid=$viewer_pid"
