# !/bin/bash
set -euxo pipefail
cd "$(dirname "$0")"
: "${CUDA_ARCH:=120}"
git submodule update --init --recursive
cmake -B ./third_party/FIDESlib/build -S . --fresh -DCMAKE_BUILD_TYPE="Release" -DFIDESLIB_INSTALL_OPENFHE=OFF
cmake --build ./third_party/FIDESlib/build -j
if [ "${FIDESLIB_SKIP_INSTALL:-0}" = "1" ]; then
  echo "Skipping FIDESlib install step (FIDESLIB_SKIP_INSTALL=1)."
else
  sudo cmake --build ./third_party/FIDESlib/build --target install -j
fi
