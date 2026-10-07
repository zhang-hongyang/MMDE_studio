#!/usr/bin/env bash
set -euo pipefail

ROOT=${MMDE_BENCHMARK_ROOT:?set MMDE_BENCHMARK_ROOT}
CONFIG=${MMDE_STUDIO_DATASETS:?set MMDE_STUDIO_DATASETS}
RELEASE=${MMDE_RELEASE:?set MMDE_RELEASE}
QUEUE=${MMDE_TASK_QUEUE:-"$ROOT/task_queue.tsv"}
MAX_JOBS=${MMDE_MAX_JOBS:-4}
STATUS="$ROOT/status"
LOGS="$ROOT/logs"
mkdir -p "$STATUS" "$LOGS"

task_id() { printf '%s-%s-%s' "$1" "$2" "$3"; }

active_jobs() {
  local count=0 pid_file pid marker
  for pid_file in "$STATUS"/*.active.pid; do
    [[ -e "$pid_file" ]] || continue
    marker=${pid_file%.active.pid}.exitcode
    [[ -f "$marker" ]] && continue
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      count=$((count + 1))
    fi
  done
  printf '%s' "$count"
}

gpu_claimed() {
  local wanted=$1 gpu_file pid_file pid marker
  for gpu_file in "$STATUS"/*.active.gpu; do
    [[ -e "$gpu_file" ]] || continue
    [[ "$(cat "$gpu_file")" == "$wanted" ]] || continue
    pid_file=${gpu_file%.gpu}.pid
    [[ -f "$pid_file" ]] || continue
    marker=${pid_file%.active.pid}.exitcode
    [[ -f "$marker" ]] && continue
    pid=$(cat "$pid_file")
    kill -0 "$pid" 2>/dev/null && return 0
  done
  return 1
}

gpu_free() {
  local gpu=$1 used
  gpu_claimed "$gpu" && return 1
  used=$(nvidia-smi -i "$gpu" --query-gpu=memory.used \
         --format=csv,noheader,nounits | tr -d ' ')
  [[ "$used" =~ ^[0-9]+$ ]] && (( used < 1000 ))
}

next_task() {
  local dataset split method id
  while IFS=$'\t' read -r dataset split method; do
    [[ -n "$dataset" ]] || continue
    id=$(task_id "$dataset" "$split" "$method")
    [[ -f "$STATUS/$id.exitcode" ]] && continue
    [[ -f "$STATUS/$id.active.pid" ]] && {
      local pid
      pid=$(cat "$STATUS/$id.active.pid")
      kill -0 "$pid" 2>/dev/null && continue
    }
    printf '%s\t%s\t%s\n' "$dataset" "$split" "$method"
    return 0
  done < "$QUEUE"
  return 1
}

launch_task() {
  local gpu=$1 dataset=$2 split=$3 method=$4 id pid
  id=$(task_id "$dataset" "$split" "$method")
  nohup env MMDE_BENCHMARK_ROOT="$ROOT" MMDE_STUDIO_DATASETS="$CONFIG" \
    MMDE_METHODS_DIR="$RELEASE/methods" \
    "$RELEASE/scripts/run_depth_benchmark.sh" \
    "$gpu" "$dataset" "$split" "$method" \
    >"$LOGS/launcher-$id.log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$STATUS/$id.active.pid"
  printf '%s\n' "$gpu" >"$STATUS/$id.active.gpu"
  printf '%s\n' "$(date -Is)" >"$STATUS/$id.started_at"
  echo "launched $id pid=$pid gpu=$gpu"
}

while true; do
  running=$(active_jobs)
  if (( running < MAX_JOBS )); then
    for gpu in 0 1 2 3 4 5 6 7; do
      (( running < MAX_JOBS )) || break
      gpu_free "$gpu" || continue
      sleep 5
      gpu_free "$gpu" || continue
      task=$(next_task) || break
      IFS=$'\t' read -r dataset split method <<<"$task"
      launch_task "$gpu" "$dataset" "$split" "$method"
      running=$((running + 1))
    done
  fi
  pending=0
  failures=0
  while IFS=$'\t' read -r dataset split method; do
    [[ -n "$dataset" ]] || continue
    id=$(task_id "$dataset" "$split" "$method")
    if [[ -f "$STATUS/$id.exitcode" ]]; then
      [[ "$(cat "$STATUS/$id.exitcode")" == 0 ]] || failures=$((failures + 1))
    else
      pending=$((pending + 1))
    fi
  done < "$QUEUE"
  if (( pending == 0 )) && (( $(active_jobs) == 0 )); then
    printf '%s\n' "$failures" >"$STATUS/scheduler.failures"
    (( failures == 0 ))
    exit $?
  fi
  sleep 15
done
