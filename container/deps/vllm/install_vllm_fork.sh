#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [ -z "${VLLM_REF:-}" ]; then
  echo "VLLM_REF is empty; keeping vLLM from the upstream runtime image."
  exit 0
fi

: "${VLLM_REPO:?VLLM_REPO must be set when VLLM_REF is set}"

VLLM_PROTECTED_PACKAGES_FILE="${VLLM_PROTECTED_PACKAGES_FILE:-/tmp/vllm_omni_protected_packages.txt}"
PROTECTED_CONSTRAINTS="$(mktemp /tmp/vllm-fork-protected.XXXXXX.txt)"

cleanup() {
  rm -rf "${PROTECTED_CONSTRAINTS}" /tmp/vllm-fork-src
}

trap cleanup EXIT

# Must match the runtime image CUDA stack (v0.21.0-cu129-ubuntu2404 -> 12.9 / cu129).
# cu128 nightly precompiled wheels link libcudart.so.13 and break on this image.
VLLM_CUDA_VERSION="${VLLM_CUDA_VERSION:-12.9}"
VLLM_CUDA_TAG="cu$(echo "${VLLM_CUDA_VERSION}" | tr -d '.')"

python3 - "${VLLM_PROTECTED_PACKAGES_FILE}" <<'PY' > "${PROTECTED_CONSTRAINTS}"
import importlib.metadata as md
from pathlib import Path
import sys

SKIP = {"vllm", "torch", "torchvision", "torchaudio"}

for raw_line in Path(sys.argv[1]).read_text().splitlines():
    name = raw_line.strip()
    if not name or name.startswith("#"):
        continue
    if name.lower() in SKIP:
        continue
    try:
        dist = md.distribution(name)
    except Exception:
        continue
    project_name = dist.metadata.get("Name") or name
    print(f"{project_name}=={dist.version}")
PY

export MAX_JOBS="${MAX_JOBS:-10}"
export VLLM_MAIN_CUDA_VERSION="${VLLM_CUDA_VERSION}"
export VLLM_USE_PRECOMPILED="${VLLM_USE_PRECOMPILED:-1}"
export VLLM_PRECOMPILED_WHEEL_VARIANT="${VLLM_PRECOMPILED_WHEEL_VARIANT:-${VLLM_CUDA_TAG}}"
# Use "nightly" unless overridden; fork commits rarely have wheels on the index.
export VLLM_PRECOMPILED_WHEEL_COMMIT="${VLLM_PRECOMPILED_WHEEL_COMMIT:-nightly}"

# Docker passes ARG VLLM_PRECOMPILED_WHEEL_LOCATION="" by default; vLLM setup.py
# treats any set value (including empty) as a user URL and fails on urlretrieve('').
if [ -z "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]; then
  unset VLLM_PRECOMPILED_WHEEL_LOCATION
  BASE_WHEEL="$(ls /vllm-workspace/dist/vllm-*.whl 2>/dev/null | head -1 || true)"
  if [ -n "${BASE_WHEEL}" ]; then
    export VLLM_PRECOMPILED_WHEEL_LOCATION="file://${BASE_WHEEL}"
    echo "Using runtime image wheel: ${VLLM_PRECOMPILED_WHEEL_LOCATION}"
  fi
fi

if [ "${VLLM_TARGET_DEVICE:-cuda}" = "cuda" ]; then
  PIP_TARGET=(--system)
  TORCH_INDEX="https://download.pytorch.org/whl/${VLLM_CUDA_TAG}"
else
  PIP_TARGET=(--python /opt/venv/bin/python)
  TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git
rm -rf /var/lib/apt/lists/*

git clone --depth 1 --branch "${VLLM_REF}" "${VLLM_REPO}" /tmp/vllm-fork-src

VLLM_PATCHES_DIR="${VLLM_PATCHES_DIR:-/tmp/vllm-fork-patches}"
if [ -d "${VLLM_PATCHES_DIR}" ]; then
  shopt -s nullglob
  patches=( "${VLLM_PATCHES_DIR}"/*.patch )
  shopt -u nullglob
  if [ "${#patches[@]}" -gt 0 ]; then
    echo "Applying ${#patches[@]} vLLM fork patch(es) from ${VLLM_PATCHES_DIR}"
    git -C /tmp/vllm-fork-src apply --ignore-whitespace "${patches[@]}"
  fi
fi

# Torch / precompiled variant must match the runtime image (cu129 for cu129-openai).
uv pip install "${PIP_TARGET[@]}" \
  --index-url "${TORCH_INDEX}" \
  --extra-index-url https://pypi.org/simple \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

uv pip uninstall "${PIP_TARGET[@]}" vllm || true

echo "Installing vLLM fork with VLLM_USE_PRECOMPILED=${VLLM_USE_PRECOMPILED} variant=${VLLM_PRECOMPILED_WHEEL_VARIANT} commit=${VLLM_PRECOMPILED_WHEEL_COMMIT}"
if [ -n "${VLLM_PRECOMPILED_WHEEL_LOCATION:-}" ]; then
  echo "VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION}"
fi

# Python-only install: download precompiled .so from wheels.vllm.ai (no CMake build).
uv pip install "${PIP_TARGET[@]}" \
  --no-build-isolation \
  --no-deps \
  --force-reinstall \
  --constraints "${PROTECTED_CONSTRAINTS}" \
  /tmp/vllm-fork-src

python3 -c "import torch, vllm; print('torch:', torch.__version__, 'cuda:', torch.version.cuda); print('vLLM fork:', vllm.__version__, 'from', vllm.__file__)"
