#!/usr/bin/env bash
set -uo pipefail

source /workspace/env.sh
cd /workspace/mxx

BRANCH="feat/fix-we-test-logs"
COMMIT="b067d28d7e470bf04f3781db6bc75d813919c583"
POD_ID="gb7xqbsmaa0xge"
RUN_ID="bench_${POD_ID}_8rtxpro6000_b067d28_diamond_we_chunk1_repeat_w8b1_crt22_log15_iter5_20260621"
LOG_DIR="/workspace/logs/runpod/diamond_we"
SUMMARY="${LOG_DIR}/${RUN_ID}.summary.tsv"
STATUS="${LOG_DIR}/${RUN_ID}.status"

mkdir -p "${LOG_DIR}"

{
  echo "run_id=${RUN_ID}"
  echo "pod_id=${POD_ID}"
  echo "branch=${BRANCH}"
  echo "commit=${COMMIT}"
  echo "chunk=1"
  echo "repeat_count=3"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${STATUS}"

echo -e "timestamp_utc\ttrial\tchunk\tresult\texit_code\tpeak_memory_used_mib\tdecoded_line\tlog\tvram_status" > "${SUMMARY}"

git fetch origin
git checkout "${BRANCH}"
git reset --hard "${COMMIT}"
git submodule update --init --recursive

export FIDESLIB_SKIP_SUBMODULE_UPDATE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RUST_LOG=info
export AUX_SAMPLING_CHUNK_WIDTH=1

export DIAMOND_WE_GPU_BENCH_SECURITY_BITS=100
export DIAMOND_WE_GPU_BENCH_CIRCUIT_HEIGHT=7
export DIAMOND_WE_GPU_BENCH_WITNESS_SIZE=8
export DIAMOND_WE_GPU_BENCH_INJECTOR_BATCH_BITS=1
export DIAMOND_WE_GPU_BENCH_CRT_BITS=28
export DIAMOND_WE_GPU_BENCH_BASE_BITS=14
export DIAMOND_WE_GPU_BENCH_MIN_CRT_DEPTH=22
export DIAMOND_WE_GPU_BENCH_MAX_CRT_DEPTH=22
export DIAMOND_WE_GPU_BENCH_RING_DIM=32768
export DIAMOND_WE_GPU_BENCH_MIN_LOG_RING_DIM=15
export DIAMOND_WE_GPU_BENCH_MAX_LOG_RING_DIM=15
export DIAMOND_WE_GPU_BENCH_ITERATIONS=5
export DIAMOND_WE_GPU_BENCH_ERROR_SIGMA=4.0
export DIAMOND_WE_GPU_BENCH_TRAPDOOR_SIGMA=4.578
export DIAMOND_WE_GPU_BENCH_D_SECRET=1
export DIAMOND_WE_GPU_BENCH_SELECTED_CRT_DEPTH=22
export DIAMOND_WE_GPU_BENCH_SELECTED_RING_DIM=32768
export DIAMOND_WE_GPU_BENCH_SELECTED_LOG_RING_DIM=15
export DIAMOND_WE_GPU_BENCH_SELECTED_ACHIEVED_SECPAR_FOR_GAUSS=155
export DIAMOND_WE_GPU_BENCH_SELECTED_NOISY_PLAINTEXT_ERROR_BITS=592
export DIAMOND_WE_GPU_BENCH_SELECTED_INPUT_INJECTION_ERROR_BITS=354
export MXX_IO_SKIP_LATTICE_CHECK_FOR_EXPLICIT_LOG_RING_DIM=1

run_trial() {
  local trial="$1"
  local log="${LOG_DIR}/${RUN_ID}_trial${trial}.log"
  local vram="${LOG_DIR}/${RUN_ID}_trial${trial}.vram.csv"
  local vram_status="${LOG_DIR}/${RUN_ID}_trial${trial}.vram.status"
  local artifact_dir="${LOG_DIR}/${RUN_ID}_artifacts_trial${trial}"
  local start_utc
  start_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  rm -rf "${artifact_dir}"
  mkdir -p "${artifact_dir}"

  {
    echo "run_id=${RUN_ID}"
    echo "trial=${trial}"
    echo "chunk=1"
    echo "started_utc=${start_utc}"
    echo "branch=${BRANCH}"
    echo "commit=${COMMIT}"
    echo "artifact_dir=${artifact_dir}"
    echo "command=cargo test --release --features gpu --test test_gpu_diamond_we test_gpu_diamond_we_error_search_bench_estimate_and_round_trip -- --nocapture"
  } > "${vram_status}"

  echo "timestamp_utc,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_percent" > "${vram}"

  (
    set -o pipefail
    export DIAMOND_WE_GPU_BENCH_ARTIFACT_DIR="${artifact_dir}"
    cargo test --release --features gpu --test test_gpu_diamond_we test_gpu_diamond_we_error_search_bench_estimate_and_round_trip -- --nocapture
  ) > "${log}" 2>&1 &
  local pid=$!

  while kill -0 "${pid}" 2>/dev/null; do
    local ts
    ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits \
      | awk -v ts="${ts}" '{gsub(/ /, "", $0); print ts "," $0}' >> "${vram}" || true
    sleep 3
  done

  wait "${pid}"
  local exit_code=$?
  local end_utc
  end_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local peak
  peak="$(awk -F, 'NR > 1 && $3+0 > max { max=$3+0 } END { print max+0 }' "${vram}")"
  local decoded_line
  decoded_line="$(grep -E 'DiamondWE GPU round trip finished' "${log}" | tail -n 1 | tr '\t' ' ' || true)"
  local result="fail"
  if [ "${exit_code}" -eq 0 ]; then
    result="pass"
  fi

  {
    echo "finished_utc=${end_utc}"
    echo "exit_code=${exit_code}"
    echo "result=${result}"
    echo "peak_memory_used_mib=${peak}"
    echo "decoded_line=${decoded_line}"
  } >> "${vram_status}"

  printf "%s\t%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${end_utc}" "${trial}" "${result}" "${exit_code}" "${peak}" "${decoded_line}" "${log}" "${vram_status}" >> "${SUMMARY}"
}

overall=0
for trial in 1 2 3; do
  echo "trial ${trial} started $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${STATUS}"
  run_trial "${trial}" || overall=1
  echo "trial ${trial} finished $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${STATUS}"
done

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "overall_exit=${overall}"
  echo "summary=${SUMMARY}"
} >> "${STATUS}"

exit "${overall}"
