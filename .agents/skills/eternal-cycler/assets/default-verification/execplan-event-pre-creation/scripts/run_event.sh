#!/usr/bin/env bash
set -euo pipefail

# execplan.pre_creation — lightweight environment check run before the plan file exists.
# --plan is not accepted; branch management is the caller's responsibility.
# Records git state and returns pass. Follow with execplan.post_creation after writing the plan.

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      echo "Usage: run_event.sh"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

commands=()
commands+=("git branch --show-current")
commands+=("git status --short")
commands+=("git log --oneline --decorate --max-count=20")

git branch --show-current >/dev/null
git status --short >/dev/null
git log --oneline --decorate --max-count=20 >/dev/null

if command -v gh >/dev/null 2>&1; then
  commands+=("gh pr status")
  set +e
  gh pr status >/dev/null 2>&1
  set -e
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
