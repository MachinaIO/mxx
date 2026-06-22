#!/usr/bin/env bash
set -uo pipefail

POD_ID="gb7xqbsmaa0xge"
POD_NAME="mxx-rtxpro6000x8-diamond-we-chunk-scan-b067d28-3tb-20260621"
GPU_COUNT=8
BRANCH="feat/fix-we-test-logs"
EXPECTED_COMMIT="b067d28d7e470bf04f3781db6bc75d813919c583"
RUN_ID="${POD_ID}_8rtxpro6000_b067d28_diamond_we_chunk_scan_w8b1_crt22_log15_iter5_20260621"
LOG_DIR="/workspace/logs/runpod/diamond_we"
BASE="${LOG_DIR}/bench_${RUN_ID}"
SUMMARY="${BASE}.summary.tsv"
STATUS="${BASE}.status"
DONE="${BASE}.done"
SCAN_LOG="${BASE}.scanner.log"

mkdir -p "${LOG_DIR}"
rm -f "${SUMMARY}" "${STATUS}" "${DONE}" "${SCAN_LOG}"

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

run_candidate() {
  local chunk="$1"
  local candidate_id="${RUN_ID}_chunk${chunk}"
  local log="${BASE}_chunk${chunk}.log"
  local vram="${BASE}_chunk${chunk}.vram.csv"
  local vram_status="${BASE}_chunk${chunk}.vram.status"
  local artifact_dir="${LOG_DIR}/artifacts_${candidate_id}"
  local mon_pid=""
  local cargo_exit="not_started"

  rm -rf "${artifact_dir}"
  mkdir -p "${artifact_dir}"
  rm -f "${log}" "${vram}" "${vram_status}"

  {
    echo "candidate_started_utc=$(timestamp)"
    echo "chunk=${chunk}"
    echo "artifact_dir=${artifact_dir}"
  } >> "${SCAN_LOG}"

  monitor_vram() {
    echo "timestamp_utc,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_percent" > "${vram}"
    while true; do
      local ts
      ts="$(timestamp)"
      nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
        | awk -v ts="${ts}" -F, '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print ts","$1","$2","$3","$4}' >> "${vram}"
      sleep 3
    done
  }

  write_vram_status() {
    if [[ -s "${vram}" ]]; then
      awk -F, '
        NR>1 {
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3);
          used=$3+0;
          if (used > peak) {
            peak=used;
            peak_ts=$1;
            peak_gpu=$2;
            peak_total=$4;
            peak_util=$5;
          }
        }
        END {
          printf("peak_memory_used_mib=%s\npeak_timestamp_utc=%s\npeak_gpu_index=%s\npeak_memory_total_mib=%s\npeak_utilization_gpu_percent=%s\n", peak, peak_ts, peak_gpu, peak_total, peak_util)
        }
      ' "${vram}" > "${vram_status}"
    fi
  }

  monitor_vram &
  mon_pid=$!

  export RUST_LOG=info
  export RUST_BACKTRACE=1
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export FIDESLIB_SKIP_SUBMODULE_UPDATE=1
  export CARGO_TARGET_DIR="/workspace/mxx/target"
  export AUX_SAMPLING_CHUNK_WIDTH="${chunk}"
  export MXX_IO_SKIP_LATTICE_CHECK_FOR_EXPLICIT_LOG_RING_DIM=1
  export DIAMOND_WE_GPU_BENCH_ARTIFACT_DIR="${artifact_dir}"
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

  set +e
  cargo test --release --features gpu --test test_gpu_diamond_we test_gpu_diamond_we_error_search_bench_estimate_and_round_trip -- --nocapture > "${log}" 2>&1
  cargo_exit=$?
  set -u

  if [[ -n "${mon_pid}" ]]; then
    kill "${mon_pid}" 2>/dev/null || true
    wait "${mon_pid}" 2>/dev/null || true
  fi
  write_vram_status

  local result="fail"
  if [[ "${cargo_exit}" == "0" ]]; then
    result="pass"
  elif grep -qiE "out of memory|cuda.*memory|CUBLAS_STATUS_ALLOC_FAILED|memory allocation|OOM" "${log}" 2>/dev/null; then
    result="oom"
  fi

  local peak="unknown"
  if [[ -s "${vram_status}" ]]; then
    peak="$(awk -F= '$1=="peak_memory_used_mib"{print $2}' "${vram_status}")"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(timestamp)" "${chunk}" "${result}" "${cargo_exit}" "${peak}" "${log}" "${vram_status}" >> "${SUMMARY}"

  {
    echo "candidate_finished_utc=$(timestamp)"
    echo "chunk=${chunk}"
    echo "result=${result}"
    echo "cargo_exit=${cargo_exit}"
    echo "peak_memory_used_mib=${peak}"
    echo "log=${log}"
    echo "vram_status=${vram_status}"
  } >> "${SCAN_LOG}"

  [[ "${result}" == "pass" ]]
}

source /workspace/env.sh
cd /workspace/mxx

{
  echo "run_started_utc=$(timestamp)"
  echo "pod_id=${POD_ID}"
  echo "pod_name=${POD_NAME}"
  echo "gpu_count=${GPU_COUNT}"
  echo "branch=${BRANCH}"
  echo "expected_commit=${EXPECTED_COMMIT}"
  echo "actual_commit_before_sync=$(git rev-parse HEAD 2>/dev/null || true)"
  echo "fixed_injector_batch_bits=1"
  echo "fixed_witness_size=8"
} > "${STATUS}"

git fetch origin >> "${SCAN_LOG}" 2>&1
git checkout "${BRANCH}" >> "${SCAN_LOG}" 2>&1
git reset --hard "${EXPECTED_COMMIT}" >> "${SCAN_LOG}" 2>&1
git submodule update --init --recursive >> "${SCAN_LOG}" 2>&1

actual_commit="$(git rev-parse HEAD)"
echo "actual_commit=${actual_commit}" >> "${STATUS}"
if [[ "${actual_commit}" != "${EXPECTED_COMMIT}" ]]; then
  echo "commit_mismatch" >> "${STATUS}"
  touch "${DONE}"
  exit 2
fi

printf "timestamp_utc\tchunk\tresult\texit_code\tpeak_memory_used_mib\tlog\tvram_status\n" > "${SUMMARY}"

last_pass=1
first_fail=0
for chunk in 2 4 8 16 32 64 92; do
  if run_candidate "${chunk}"; then
    last_pass="${chunk}"
  else
    first_fail="${chunk}"
    break
  fi
done

if [[ "${first_fail}" != "0" ]]; then
  low=$((last_pass + 1))
  high=$((first_fail - 1))
  while [[ "${low}" -le "${high}" ]]; do
    mid=$(((low + high) / 2))
    if run_candidate "${mid}"; then
      last_pass="${mid}"
      low=$((mid + 1))
    else
      first_fail="${mid}"
      high=$((mid - 1))
    fi
  done
fi

{
  echo "run_finished_utc=$(timestamp)"
  echo "max_oom_free_chunk_cols=${last_pass}"
  echo "first_failing_chunk_cols=${first_fail}"
  echo "summary=${SUMMARY}"
  echo "scan_log=${SCAN_LOG}"
} >> "${STATUS}"

touch "${DONE}"
exit 0
