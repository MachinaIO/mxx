# !/bin/bash
set -euxo pipefail
cd "$(dirname "$0")"
: "${CUDA_ARCH:=120}"
git submodule update --init --recursive
cmake -B ./third_party/FIDESlib/build -S . --fresh -DCMAKE_BUILD_TYPE="Release" -DFIDESLIB_INSTALL_OPENFHE=OFF
cmake --build ./third_party/FIDESlib/build -j
cmake --build ./third_party/FIDESlib/build  --target install -j
