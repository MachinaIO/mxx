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

commands=()
commands+=("git diff --name-only --relative HEAD --")
commands+=("git ls-files --others --exclude-standard")
commands+=("rg -n <placeholder-pattern> <changed-doc-targets>")

mapfile -t paths < <({
  git diff --name-only --relative HEAD --
  git ls-files --others --exclude-standard
} | sed '/^$/d' | sort -u)

for path in "${paths[@]}"; do
  case "$path" in
    eternal-cycler-out/*|*.md|PLANS.md|REVIEW.md)
      ;;
    *)
      echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
      echo "FAILURE_SUMMARY=non-doc path found for docs-only event: $path"
      echo "STATUS=fail"
      exit 1
      ;;
  esac
done

placeholder_targets=()
for path in "${paths[@]}"; do
  if [[ -f "$path" ]]; then
    placeholder_targets+=("$path")
  fi
done

if [[ ${#placeholder_targets[@]} -eq 0 ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=none"
  echo "STATUS=pass"
  exit 0
fi

set +e
placeholder_hits="$(rg -n -e "TODO" -e "TBD" -e "FIXME" "${placeholder_targets[@]}" 2>/dev/null)"
placeholder_rc=$?
set -e

if [[ $placeholder_rc -eq 0 ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=stale policy/documentation placeholders found: $(echo "$placeholder_hits" | head -n 1)"
  echo "STATUS=fail"
  exit 1
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
