#!/usr/bin/env bash
# Download Depth-Anything-V2 pretrained weights into third_party/Depth-Anything-V2/checkpoints/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${ROOT}/third_party/Depth-Anything-V2/checkpoints"
mkdir -p "${CKPT_DIR}"

download_one() {
  local encoder="$1"
  local url="$2"
  local out="${CKPT_DIR}/depth_anything_v2_${encoder}.pth"
  if [[ -f "${out}" ]]; then
    echo "  [skip] ${out} already exists"
    return
  fi
  echo "  [get]  ${out}"
  wget -q --show-progress -O "${out}" "${url}"
}

ENCODERS="${*:-vitl}"

for enc in ${ENCODERS}; do
  case "${enc}" in
    vits)
      download_one vits "https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth"
      ;;
    vitb)
      download_one vitb "https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth"
      ;;
    vitl)
      download_one vitl "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
      ;;
    *)
      echo "Unknown encoder: ${enc}  (choose vits | vitb | vitl)"
      exit 1
      ;;
  esac
done

echo "Checkpoints saved under: ${CKPT_DIR}"
