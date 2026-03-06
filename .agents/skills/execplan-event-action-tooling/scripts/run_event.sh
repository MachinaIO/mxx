#!/usr/bin/env bash
set -euo pipefail

# ETERNAL_CYCLER_ROOT and REPO_ROOT are exported by execplan_gate.sh before calling this script.
# Fall back to git-based resolution if invoked directly (e.g. during testing).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ETERNAL_CYCLER_ROOT="${ETERNAL_CYCLER_ROOT:-$(cd "${SCRIPT_DIR}/../../eternal-cycler" && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)}"

repo_rel_path() {
  local path="$1"
  if [[ "$path" == "${REPO_ROOT%/}/"* ]]; then
    printf '%s' "${path#${REPO_ROOT%/}/}"
  elif [[ "$path" == "${REPO_ROOT%/}" ]]; then
    printf '.'
  else
    printf '%s' "$path"
  fi
}

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

commands="bash -n $(repo_rel_path "$ETERNAL_CYCLER_ROOT")/scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh"

if ! bash -n "${ETERNAL_CYCLER_ROOT}"/scripts/*.sh "${REPO_ROOT}"/.agents/skills/execplan-event-*/scripts/*.sh; then
  echo "COMMANDS=$commands"
  echo "FAILURE_SUMMARY=tooling script syntax check failed"
  echo "STATUS=fail"
  exit 1
fi

echo "COMMANDS=$commands"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
