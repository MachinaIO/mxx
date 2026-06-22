#!/usr/bin/env bash
set -uo pipefail

RUN_ID="qgga47ub38w1py_8rtxpro6000_51b8079_diamond_we_full_w8b1_chunk1_selected_crt22_log15_iter5_20260620_2243JST"
LOG_DIR="/workspace/logs/runpod/diamond_we"
BASE="${LOG_DIR}/bench_${RUN_ID}"
LOG="${BASE}.log"
STATUS="${BASE}.status"
VRAM="${BASE}.vram.csv"
VRAM_STATUS="${BASE}.vram.status"
DONE="${BASE}.done"
ARTIFACT_DIR="${LOG_DIR}/artifacts_${RUN_ID}"
MON_PID=""
CARGO_EXIT="not_started"
STATUS_WRITTEN=0

mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}"
rm -f "${LOG}" "${STATUS}" "${VRAM}" "${VRAM_STATUS}" "${DONE}"

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

write_vram_status() {
  if [[ -s "${VRAM}" ]]; then
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
    ' "${VRAM}" > "${VRAM_STATUS}"
  fi
}

write_status() {
  local result="$1"
  local exit_code="$2"
  write_vram_status
  {
    echo "result=${result}"
    echo "exit_code=${exit_code}"
    echo "cargo_exit=${CARGO_EXIT}"
    echo "run_finished_utc=$(timestamp)"
    echo "pod_id=qgga47ub38w1py"
    echo "pod_name=mxx-rtxpro6000x8-diamond-we-full-w8b1-55179e3-3tb-20260620-jst"
    echo "gpu_count=8"
    echo "branch=feat/fix-we-test-logs"
    echo "expected_commit=51b80794d62075fa047b8d412da6c07649c15bb9"
    echo "actual_commit=$(cd /workspace/mxx && git rev-parse HEAD 2>/dev/null || true)"
    echo "aux_sampling_chunk_width=1"
    echo "log=${LOG}"
    echo "vram_csv=${VRAM}"
    echo "vram_status=${VRAM_STATUS}"
    echo "artifact_dir=${ARTIFACT_DIR}"
  } > "${STATUS}"
  STATUS_WRITTEN=1
}

cleanup_monitor() {
  if [[ -n "${MON_PID}" ]]; then
    kill "${MON_PID}" 2>/dev/null || true
    wait "${MON_PID}" 2>/dev/null || true
  fi
}

handle_signal() {
  local sig="$1"
  cleanup_monitor
  CARGO_EXIT="signal_${sig}"
  write_status "signal_${sig}" 128
  touch "${DONE}"
  exit 128
}

on_exit() {
  local rc=$?
  cleanup_monitor
  if [[ "${STATUS_WRITTEN}" != "1" ]]; then
    write_status "exit_trap" "${rc}"
  fi
  touch "${DONE}"
}

trap 'handle_signal HUP' HUP
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM
trap on_exit EXIT

monitor_vram() {
  echo "timestamp_utc,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_percent" > "${VRAM}"
  while true; do
    ts=$(timestamp)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null \
      | awk -v ts="${ts}" -F, '{gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $3); gsub(/^[[:space:]]+|[[:space:]]+$/, "", $4); print ts","$1","$2","$3","$4}' >> "${VRAM}"
    sleep 3
  done
}

monitor_vram &
MON_PID=$!

source /workspace/env.sh
cd /workspace/mxx

echo "run_started_utc=$(timestamp)" > "${STATUS}"
echo "pod_id=qgga47ub38w1py" >> "${STATUS}"
echo "branch=feat/fix-we-test-logs" >> "${STATUS}"
echo "expected_commit=51b80794d62075fa047b8d412da6c07649c15bb9" >> "${STATUS}"
echo "actual_commit=$(git rev-parse HEAD)" >> "${STATUS}"
echo "aux_sampling_chunk_width=1" >> "${STATUS}"
echo "artifact_dir=${ARTIFACT_DIR}" >> "${STATUS}"

export RUST_LOG=info
export RUST_BACKTRACE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export FIDESLIB_SKIP_SUBMODULE_UPDATE=1
export CARGO_TARGET_DIR="/workspace/mxx/target"
export AUX_SAMPLING_CHUNK_WIDTH=1
export MXX_IO_SKIP_LATTICE_CHECK_FOR_EXPLICIT_LOG_RING_DIM=1
export DIAMOND_WE_GPU_BENCH_ARTIFACT_DIR="${ARTIFACT_DIR}"
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
cargo test --release --features gpu --test test_gpu_diamond_we test_gpu_diamond_we_error_search_bench_estimate_and_round_trip -- --nocapture > "${LOG}" 2>&1
CARGO_EXIT=$?
set -u

cleanup_monitor
write_status "cargo_finished" "${CARGO_EXIT}"
touch "${DONE}"
exit "${CARGO_EXIT}"
