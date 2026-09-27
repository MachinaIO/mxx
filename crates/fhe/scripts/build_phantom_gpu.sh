#!/usr/bin/env bash
set -euo pipefail
# Keep the GPLv3 dependency outside this repository. Pin before building.
checkout=${1:?Usage: build_phantom_gpu.sh EXTERNAL_CHECKOUT [CUDA_ARCH=89]}
architecture=${2:-89}
revision=1f4a198443b3af77118e51f53d5b8f332154b875
if [[ ! -d "$checkout/.git" ]]; then
  git clone https://github.com/encryptorion-lab/phantom-fhe.git "$checkout"
  git -C "$checkout" checkout --detach "$revision"
fi
[[ $(git -C "$checkout" rev-parse HEAD) == "$revision" ]] || { echo "Unexpected PhantomFHE revision" >&2; exit 1; }
cmake -S "$checkout" -B "$checkout/build" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$architecture" -DPHANTOM_ENABLE_EXAMPLE=OFF
cmake --build "$checkout/build" -j "${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir -p "$checkout/build/bin"
nvcc -std=c++17 -O3 --default-stream per-thread -arch="sm_$architecture" \
  "$script_dir/phantom_gpu_bgv.cu" -I "$checkout/include" -L "$checkout/build/lib" \
  -lPhantom -Xlinker -rpath -Xlinker "$checkout/build/lib" -o "$checkout/build/bin/phantom_gpu_bgv"
