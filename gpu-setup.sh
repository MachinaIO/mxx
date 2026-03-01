#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
: "${CUDA_ARCH:=89}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Set CUDA_HOME/CUDA_LIB_DIR and install CUDA toolkit."
  exit 1
fi

echo "Detected nvcc: $(command -v nvcc)"
nvcc --version | tail -n 1 || true
echo "CUDA_ARCH=${CUDA_ARCH}"

if [ "${GPU_SETUP_BUILD_LEGACY_FIDESLIB:-0}" = "1" ]; then
  echo "Legacy mode: building FIDESlib (GPU_SETUP_BUILD_LEGACY_FIDESLIB=1)."
  git submodule update --init --recursive
  cmake -B ./third_party/FIDESlib/build -S . --fresh \
    -DCMAKE_BUILD_TYPE="Release" \
    -DFIDESLIB_INSTALL_OPENFHE=OFF
  cmake --build ./third_party/FIDESlib/build -j
  if [ "${FIDESLIB_SKIP_INSTALL:-0}" = "1" ]; then
    echo "Skipping FIDESlib install step (FIDESLIB_SKIP_INSTALL=1)."
  else
    sudo cmake --build ./third_party/FIDESlib/build --target install -j
  fi
else
  echo "Skipping FIDESlib build; GPU runtime is now FIDESlib-independent."
  echo "Set GPU_SETUP_BUILD_LEGACY_FIDESLIB=1 to run the old FIDESlib build flow."
fi
