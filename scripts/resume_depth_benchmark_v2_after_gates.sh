#!/usr/bin/env bash
# Resume finalization after prediction scan + viewer contracts already passed.
set -euo pipefail

ROOT=${MMDE_BENCHMARK_ROOT:?set MMDE_BENCHMARK_ROOT}
CONFIG=${MMDE_STUDIO_DATASETS:?set MMDE_STUDIO_DATASETS}
RELEASE=${MMDE_RELEASE:?set MMDE_RELEASE}
PY=/home/ZhangHongyang/miniconda3/envs/mmde/bin/python

"$PY" - "$ROOT/logs/verify-predictions-v2-final.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if not report.get("ok") or report.get("errors"):
    raise SystemExit("prediction verification is not clean")
PY
grep -Eq '13 passed' "$ROOT/logs/viewer-contract-v2.log"

"$PY" "$RELEASE/scripts/audit_depth_benchmark_local.py" \
  --datasets-config "$CONFIG" --run-root "$ROOT" --release "$RELEASE" \
  >"$ROOT/logs/local-experiment-audit.json"

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
  --host 0.0.0.0 --port 8010 \
  >"$ROOT/logs/viewer-8010.log" 2>&1 </dev/null &
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
import os, sys, yaml
from pathlib import Path
path = Path(sys.argv[1])
doc = yaml.safe_load(path.read_text())
doc["status"] = "TECHNICAL_GATES_PASSED_AUDIT_PENDING"
doc["prediction_integrity_gate"] = "PASSED_FULL_FILE_SCAN"
doc["viewer_contract_gate"] = "PASSED_13_TESTS"
doc["local_integrity_audit"] = "PASS_LOCAL_INDEPENDENT_REVIEW_UNAVAILABLE"
doc["viewer_url"] = "http://100.78.74.13:8010"
doc["viewer_local_url"] = "http://127.0.0.1:18010"
tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
tmp.write_text(yaml.safe_dump(doc, sort_keys=False))
os.replace(tmp, path)
PY

printf '0\n' >"$ROOT/status/finalize-v2-technical.exitcode"
echo "FINALIZE_V2_TECHNICAL_COMPLETE $(date -Is) viewer_pid=$viewer_pid"
