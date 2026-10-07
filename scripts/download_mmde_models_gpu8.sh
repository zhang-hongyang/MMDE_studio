#!/usr/bin/env bash
set -Eeuo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/ZhangHongyang/miniconda3}"
STUDIO_ROOT="${MMDE_STUDIO_ROOT:-/home/ZhangHongyang/ResearchHub-workspaces/MMDE_studio/code/current}"
SCRATCH_ROOT="${MMDE_SCRATCH_ROOT:-/home/ZhangHongyang/ResearchHub-scratch}"
MODEL_ROOT="${SCRATCH_ROOT}/models/MMDE_studio"
INSTALL_MARKER="${SCRATCH_ROOT}/logs/MMDE_studio/setup/methods-install.complete"

until [[ -f "${INSTALL_MARKER}" ]]; do
  if systemctl --user is-failed --quiet mmde-methods-install-20260922.service; then
    echo "dependency installation failed; refusing to download models" >&2
    exit 1
  fi
  if ! systemctl --user is-active --quiet mmde-methods-install-20260922.service; then
    echo "dependency installation stopped without completion marker" >&2
    exit 1
  fi
  sleep 20
done

mkdir -p "${MODEL_ROOT}/huggingface"
export MMDE_MODEL_ROOT="${MODEL_ROOT}"
export HF_HOME="${MODEL_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

# The mirror redirects large LFS/Xet objects to cas-bridge.xethub.hf.co.
# systemd-resolved on this node intermittently returns SERVFAIL for that host,
# while public DNS resolves it consistently.  curl's explicit resolve entry
# plus durable target files gives us true cross-restart resume semantics;
# huggingface_hub removes its .incomplete file after a terminal timeout.
CAS_BRIDGE_IP="${MMDE_CAS_BRIDGE_IP:-13.33.183.70}"

fetch_model() {
  local url="$1"
  local target="$2"
  local expected="$3"
  local actual=0
  mkdir -p "$(dirname "${target}")"
  while true; do
    if [[ -f "${target}" ]]; then
      actual="$(stat -c %s "${target}")"
    else
      actual=0
    fi
    if [[ "${actual}" == "${expected}" ]]; then
      printf 'verified %s (%s bytes)\n' "${target}" "${actual}"
      return 0
    fi
    if (( actual > expected )); then
      printf 'oversized model file: %s (%s > %s)\n' \
        "${target}" "${actual}" "${expected}" >&2
      return 1
    fi
    printf 'resume %s: %s/%s bytes\n' "${target}" "${actual}" "${expected}"
    curl --fail --location --continue-at - --output "${target}" \
      --connect-timeout 20 --speed-time 120 --speed-limit 1024 \
      --retry 100 --retry-delay 5 --retry-all-errors \
      --resolve "cas-bridge.xethub.hf.co:443:${CAS_BRIDGE_IP}" \
      "${url}" || true
  done
}

fetch_model "${HF_ENDPOINT}/Ruicheng/moge-3-vitl/resolve/main/model.pt" \
  "${MODEL_ROOT}/moge3/model.pt" 1481333394 &
pid_moge=$!
fetch_model "${HF_ENDPOINT}/facebook/map-anything/resolve/main/model.safetensors" \
  "${MODEL_ROOT}/map-anything/model.safetensors" 4914062480 &
pid_map=$!
fetch_model "${HF_ENDPOINT}/robbyant/lingbot-map/resolve/main/lingbot-map.pt" \
  "${MODEL_ROOT}/lingbot-map/lingbot-map.pt" 4632303465 &
pid_lingbot=$!
fetch_model "${HF_ENDPOINT}/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth" \
  "${MODEL_ROOT}/depth-anything-v2/depth_anything_v2_vitl.pth" 1341395338 &
pid_dav2=$!

wait "${pid_moge}"
wait "${pid_map}"
wait "${pid_lingbot}"
wait "${pid_dav2}"
fetch_model "${HF_ENDPOINT}/facebook/map-anything/resolve/main/config.json" \
  "${MODEL_ROOT}/map-anything/config.json" 5776
fetch_model "${HF_ENDPOINT}/facebook/map-anything/resolve/main/README.md" \
  "${MODEL_ROOT}/map-anything/README.md" 1378

exec "${CONDA_ROOT}/envs/mmde-moge3/bin/python" \
  "${STUDIO_ROOT}/scripts/download_mmde_models.py"
