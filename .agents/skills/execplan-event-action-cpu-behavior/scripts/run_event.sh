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

derive_scope_filter() {
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

commands=()
commands+=("cargo +nightly fmt --all")

if ! cargo +nightly fmt --all; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=cargo fmt failed"
  echo "STATUS=fail"
  exit 1
fi

scope_filter="${EXECPLAN_TEST_FILTER:-}"
if [[ -z "$scope_filter" ]]; then
  scope_filter="$(derive_scope_filter || true)"
fi

ran_targeted=0
if [[ -n "$scope_filter" ]]; then
  test_cmd="cargo test -r --lib -- $scope_filter"
  commands+=("$test_cmd")
  if ! cargo test -r --lib -- "$scope_filter"; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=filtered CPU test command failed"
    echo "STATUS=fail"
    exit 1
  fi
  ran_targeted=1
fi

run_full="${EXECPLAN_RUN_FULL_LIB_TESTS:-auto}"
if [[ "$run_full" != "0" && "$run_full" != "1" && "$run_full" != "auto" ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=invalid EXECPLAN_RUN_FULL_LIB_TESTS value: $run_full (expected 0|1|auto)"
  echo "STATUS=fail"
  exit 1
fi

if [[ "$run_full" == "auto" ]]; then
  run_full="0"
  if [[ "${EXECPLAN_SCOPE_COMPLETE:-0}" == "1" || "${EXECPLAN_FOUNDATIONAL_CHANGE:-0}" == "1" ]]; then
    run_full="1"
  elif has_foundational_change; then
    run_full="1"
  elif [[ $ran_targeted -eq 0 ]]; then
    # If scope is unclear, fall back to full lib tests so behavior changes still get coverage.
    run_full="1"
  fi
fi

if [[ "$run_full" == "1" ]]; then
  test_cmd="cargo test -r --lib"
  commands+=("$test_cmd")
  if ! cargo test -r --lib; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=full CPU lib test command failed"
    echo "STATUS=fail"
    exit 1
  fi
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
