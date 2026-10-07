#!/usr/bin/env bash
set -euo pipefail

URL=${NUSCENES_TRANSFER_URL:-http://192.168.5.43:18081}
ROOT=${NUSCENES_GPU_ROOT:-/home/ZhangHongyang/ResearchHub-scratch/datasets/MMDE_studio/nuscenes-v1.0-test-official-20261007}
NAME=v1.0-test-camfront-lidartop.tar.zst
mkdir -p "$ROOT"

while ! curl -fsS --max-time 10 "$URL/$NAME.sha256" -o "$ROOT/$NAME.sha256"; do
  echo "$(date -Is) waiting for nuScenes transfer package" >&2
  sleep 60
done

curl --fail --location --retry 20 --retry-delay 10 --continue-at - \
  "$URL/$NAME" -o "$ROOT/$NAME"
expected=$(awk 'NR == 1 {print $1}' "$ROOT/$NAME.sha256")
actual=$(sha256sum "$ROOT/$NAME" | awk '{print $1}')
[[ -n "$expected" && "$actual" == "$expected" ]] || {
  echo "nuScenes transfer checksum mismatch" >&2
  exit 1
}

install="$ROOT/release.installing"
test ! -e "$install"
mkdir -p "$install"
tar --use-compress-program=unzstd -xf "$ROOT/$NAME" -C "$install"
for path in v1.0-test samples/CAM_FRONT samples/LIDAR_TOP; do
  test -e "$install/$path"
done
test ! -e "$ROOT/release"
mv "$install" "$ROOT/release"
printf '{"archive_sha256":"%s","installed_at":"%s"}\n' \
  "$actual" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$ROOT/GPU_READY.json"
echo "NUSCENES_GPU_READY $ROOT/release"
