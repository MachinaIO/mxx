# !/bin/bash
set -euxo pipefail
cd "$(dirname "$0")"
: "${CUDA_ARCH:=89}"
export CUDA_ARCH
if [ "${FIDESLIB_SKIP_SUBMODULE_UPDATE:-0}" != "1" ]; then
  git submodule update --init --recursive
else
  echo "Skipping git submodule update (FIDESLIB_SKIP_SUBMODULE_UPDATE=1)."
fi
cmake -B ./third_party/FIDESlib/build -S . --fresh -DCMAKE_BUILD_TYPE="Release" -DFIDESLIB_INSTALL_OPENFHE=OFF -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
cmake --build ./third_party/FIDESlib/build -j
if [ "${FIDESLIB_SKIP_INSTALL:-0}" = "1" ]; then
  echo "Skipping FIDESlib install step (FIDESLIB_SKIP_INSTALL=1)."
else
  sudo cmake --build ./third_party/FIDESlib/build --target install -j
fi
