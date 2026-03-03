#!/usr/bin/env bash
set -euo pipefail

PLAN=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan)
      PLAN="${2:-}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: run_event.sh --plan <plan_md>"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PLAN" || ! -f "$PLAN" ]]; then
  echo "COMMANDS=none"
  echo "FAILURE_SUMMARY=plan file not found"
  echo "STATUS=fail"
  exit 1
fi

collect_changed_paths() {
  {
    git diff --name-only --relative HEAD --
    git ls-files --others --exclude-standard
  } | sed '/^$/d' | sort -u
}

derive_gpu_filter() {
  local p module
  declare -A modules=()
  while IFS= read -r p; do
    if [[ "$p" =~ ^src/([^/]+)/ ]]; then
      module="${BASH_REMATCH[1]}"
      modules["$module"]=1
    fi
  done < <(collect_changed_paths)

  if [[ ${#modules[@]} -eq 1 ]]; then
    for module in "${!modules[@]}"; do
      echo "$module"
      return 0
    done
  fi
  return 1
}

detect_cuda_change() {
  local p
  while IFS= read -r p; do
    case "$p" in
      cuda/*|*.cu|*.cuh|*.c|*.cc|*.cpp|*.cxx|*.ptx)
        return 0
        ;;
    esac
  done < <(collect_changed_paths)
  return 1
}

has_foundational_change() {
  local p
  while IFS= read -r p; do
    case "$p" in
      Cargo.toml|Cargo.lock|build.rs|src/lib.rs|src/mod.rs|src/*/mod.rs)
        return 0
        ;;
    esac
  done < <(collect_changed_paths)
  return 1
}

run_gpu_300() {
  local bin fails i log reason
  set -uo pipefail
  mkdir -p logs
  bin="$(find target/release/deps -maxdepth 1 -type f -perm -111 -name 'mxx-*' | head -n 1)"
  if [[ -z "$bin" ]]; then
    echo "No GPU-enabled lib test binary found in target/release/deps" >&2
    return 1
  fi

  fails=0
  : > logs/gpu_300_failures.txt
  for i in $(seq 1 300); do
    log="logs/gpu_300_iter_${i}.log"
    if ! "$bin" gpu --nocapture >"$log" 2>&1; then
      fails=$((fails + 1))
      reason="$(rg -m1 -n 'panicked at|FAILED|error:|CUDA|assertion' "$log" || true)"
      printf 'iter=%03d log=%s reason=%s\n' "$i" "$log" "${reason:-unknown}" | tee -a logs/gpu_300_failures.txt
    fi
  done
  printf 'total_runs=300 failed_runs=%d\n' "$fails" | tee logs/gpu_300_summary.txt
  if [[ "$fails" -gt 0 ]]; then
    return 1
  fi
  return 0
}

commands=()
commands+=("cargo +nightly fmt --all")

if command -v nvidia-smi >/dev/null 2>&1; then
  commands+=("nvidia-smi")
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=nvidia-smi exists but GPU runtime is unavailable"
    echo "STATUS=fail"
    exit 1
  fi
elif [[ ! -e /dev/nvidia0 ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=GPU runtime not detected (/dev/nvidia0 missing and nvidia-smi unavailable)"
  echo "STATUS=fail"
  exit 1
fi

if ! cargo +nightly fmt --all; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=cargo fmt failed"
  echo "STATUS=fail"
  exit 1
fi

cuda_changed="${EXECPLAN_CUDA_CHANGED:-auto}"
if [[ "$cuda_changed" == "auto" ]]; then
  if detect_cuda_change; then
    cuda_changed="1"
  else
    cuda_changed="0"
  fi
fi
if [[ "$cuda_changed" != "0" && "$cuda_changed" != "1" ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=invalid EXECPLAN_CUDA_CHANGED value: $cuda_changed (expected 0|1|auto)"
  echo "STATUS=fail"
  exit 1
fi

gpu_filter="${EXECPLAN_GPU_TEST_FILTER:-}"
if [[ -z "$gpu_filter" ]]; then
  gpu_filter="$(derive_gpu_filter || true)"
fi

ran_targeted=0
if [[ "$cuda_changed" == "1" && -n "$gpu_filter" ]]; then
  test_cmd="cargo test -r --lib --features gpu -- $gpu_filter"
  commands+=("$test_cmd")
  if ! cargo test -r --lib --features gpu -- "$gpu_filter"; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=filtered GPU test command failed"
    echo "STATUS=fail"
    exit 1
  fi
  ran_targeted=1
fi

run_full="${EXECPLAN_RUN_FULL_GPU_TESTS:-auto}"
if [[ "$run_full" != "0" && "$run_full" != "1" && "$run_full" != "auto" ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=invalid EXECPLAN_RUN_FULL_GPU_TESTS value: $run_full (expected 0|1|auto)"
  echo "STATUS=fail"
  exit 1
fi
if [[ "$run_full" == "auto" ]]; then
  run_full="0"
  if [[ "${EXECPLAN_SCOPE_COMPLETE:-0}" == "1" || "${EXECPLAN_FOUNDATIONAL_CHANGE:-0}" == "1" ]]; then
    run_full="1"
  elif has_foundational_change; then
    run_full="1"
  elif [[ "$cuda_changed" == "1" && $ran_targeted -eq 0 ]]; then
    run_full="1"
  fi
fi

if [[ "$run_full" == "1" ]]; then
  test_cmd="cargo test -r --lib --features gpu"
  commands+=("$test_cmd")
  if ! cargo test -r --lib --features gpu; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=full GPU lib test command failed"
    echo "STATUS=fail"
    exit 1
  fi
fi

work_complete="${EXECPLAN_WORK_COMPLETE:-auto}"
if [[ "$work_complete" == "auto" ]]; then
  if [[ "$cuda_changed" == "1" ]]; then
    work_complete="1"
  else
    work_complete="0"
  fi
fi
if [[ "$work_complete" != "0" && "$work_complete" != "1" ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=invalid EXECPLAN_WORK_COMPLETE value: $work_complete (expected 0|1|auto)"
  echo "STATUS=fail"
  exit 1
fi

if [[ "$cuda_changed" == "1" && "$work_complete" == "1" ]]; then
  build_cmd="cargo test gpu -r --lib --features gpu --no-run"
  commands+=("$build_cmd")
  if ! cargo test gpu -r --lib --features gpu --no-run; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=GPU no-run build failed before 300-run repetition"
    echo "STATUS=fail"
    exit 1
  fi

  repeat_cmd="300x GPU unit-test binary repetition with logs/gpu_300_* artifacts"
  commands+=("$repeat_cmd")
  if ! run_gpu_300; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=GPU 300-run repetition completed with failures; see logs/gpu_300_summary.txt and logs/gpu_300_failures.txt"
    echo "STATUS=fail"
    exit 1
  fi
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
