#!/usr/bin/env bash
set -Eeuo pipefail

# Install the six MMDE method entries requested for the 8x RTX 5090 node.
# Third-party repositories are staged separately under algorithms/.  Each
# dependency family gets an isolated conda environment so upgrades cannot
# break the Studio backend or the already validated Marigold V2 environment.

CONDA_ROOT="${CONDA_ROOT:-/home/ZhangHongyang/miniconda3}"
STUDIO_ROOT="${MMDE_STUDIO_ROOT:-/home/ZhangHongyang/ResearchHub-workspaces/MMDE_studio/code/current}"
SCRATCH_ROOT="${MMDE_SCRATCH_ROOT:-/home/ZhangHongyang/ResearchHub-scratch}"
ALGORITHMS_ROOT="${STUDIO_ROOT}/algorithms"
LOG_ROOT="${SCRATCH_ROOT}/logs/MMDE_studio/setup"
MODEL_ROOT="${SCRATCH_ROOT}/models/MMDE_studio"
BASE_ENV="${CONDA_ROOT}/envs/mmde"

mkdir -p "${LOG_ROOT}" "${MODEL_ROOT}/huggingface" "${MODEL_ROOT}/torch"
export HF_HOME="${MODEL_ROOT}/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${MODEL_ROOT}/torch"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONUNBUFFERED=1

log() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*"
}

clone_env() {
  local name="$1"
  local prefix="${CONDA_ROOT}/envs/${name}"
  if [[ ! -x "${prefix}/bin/python" ]]; then
    log "creating ${name} from the validated mmde base environment"
    "${CONDA_ROOT}/bin/conda" create --yes --clone "${BASE_ENV}" --name "${name}"
  else
    log "reusing ${name}"
  fi
  # The source environment contains Marigold V2, which deliberately pins an
  # older huggingface_hub.  These method-specific clones do not run Marigold;
  # remove it before resolving their independent dependency sets.
  "${prefix}/bin/python" -m pip uninstall --yes marigold-v2 >/dev/null 2>&1 || true
}

pip_install() {
  local env_name="$1"
  shift
  "${CONDA_ROOT}/envs/${env_name}/bin/python" -m pip install "$@"
  "${CONDA_ROOT}/envs/${env_name}/bin/python" -m pip check
}

clone_env mmde-moge3
pip_install mmde-moge3 --upgrade --editable "${ALGORITHMS_ROOT}/moge"

clone_env mmde-mapanything
pip_install mmde-mapanything --upgrade --editable "${ALGORITHMS_ROOT}/map-anything"

clone_env mmde-lingbot
# The project documents PyTorch 2.8 for its optional Kaolin renderer, but the
# MMDE adapter uses the supported SDPA inference path only.  Keeping the
# validated CUDA 12.8 / PyTorch build avoids an unnecessary 5+ GB reinstall.
pip_install mmde-lingbot --upgrade --editable "${ALGORITHMS_ROOT}/lingbot-map"

clone_env mmde-temporal
# PTC-Depth needs Eigen headers for its pybind11 extension.  Install them in
# the conda prefix so no sudo/system package mutation is required.  Pin the
# CPython implementation: an unconstrained Eigen solve can otherwise select
# the experimental GraalPy build, whose JSSE ssl module is incompatible with
# pip/urllib3.  libopencv supplies the C++ CMake package required by PTC.
"${CONDA_ROOT}/bin/conda" install --yes --name mmde-temporal -c conda-forge \
  'python=3.10.*=*_cpython' 'openssl<4' 'eigen>=3.3,<4' \
  'libopencv>=4.5,<5'
export CMAKE_PREFIX_PATH="${CONDA_ROOT}/envs/mmde-temporal${CMAKE_PREFIX_PATH:+:${CMAKE_PREFIX_PATH}}"
export LD_LIBRARY_PATH="${CONDA_ROOT}/envs/mmde-temporal/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
pip_install mmde-temporal --upgrade scikit-build-core pybind11
pip_install mmde-temporal --upgrade --editable "${ALGORITHMS_ROOT}/ptc-depth"
# Keep the validated CUDA 12.8 PyTorch build inherited from the base env.
# MTD's requirements are unpinned; --upgrade would replace it with the newest
# generic PyPI torch wheel even though the installed version already satisfies
# the project.
pip_install mmde-temporal -r <(
  grep -vE '^[[:space:]]*(fast_slic|opencv-contrib-python)([[:space:]]|$)' \
    "${ALGORITHMS_ROOT}/mtd/requirements.txt"
)
# fast_slic/opencv-contrib are optional superpixel backends.  The deployed
# configuration uses scikit-image Felzenszwalb, so leave the unused native
# backends as an explicit opt-in instead of blocking reproducible setup on
# their legacy build toolchain.
log "optional fast_slic/opencv-contrib skipped (Felzenszwalb is configured)"
"${CONDA_ROOT}/envs/mmde-temporal/bin/python" -m pip check
"${CONDA_ROOT}/envs/mmde-temporal/bin/python" -c \
  'import cv2, ptc_depth; print("ptc_depth", ptc_depth.__file__, "cv2", cv2.__version__)'

log "all MMDE method environments installed"
for name in mmde-moge3 mmde-mapanything mmde-lingbot mmde-temporal; do
  "${CONDA_ROOT}/envs/${name}/bin/python" - <<'PY'
import sys
import torch
print(sys.executable)
print("torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
PY
done

touch "${LOG_ROOT}/methods-install.complete"
