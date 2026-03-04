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

loop_script=".agents/skills/eternal-cycler/scripts/run_builder_reviewer_loop.sh"
branch_first_call_marker='PR_URL="$(resolve_or_create_pr_for_branch "$TARGET_BRANCH" "$pr_title" "$pr_body")"'
branch_first_title_default='local pr_title="${2:-}"'
branch_first_body_default='local pr_body="${3:-}"'

commands="bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh"
commands="$commands; rg -F '$branch_first_call_marker' $loop_script"
commands="$commands; rg -F '$branch_first_title_default' $loop_script"
commands="$commands; rg -F '$branch_first_body_default' $loop_script"

if ! bash -n scripts/*.sh .agents/skills/execplan-event-*/scripts/*.sh; then
  echo "COMMANDS=$commands"
  echo "FAILURE_SUMMARY=tooling script syntax check failed"
  echo "STATUS=fail"
  exit 1
fi

if [[ ! -f "$loop_script" ]]; then
  echo "COMMANDS=$commands"
  echo "FAILURE_SUMMARY=missing loop script for branch-first failure regression guard: $loop_script"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -F --quiet "$branch_first_call_marker" "$loop_script"; then
  echo "COMMANDS=$commands"
  echo "FAILURE_SUMMARY=missing branch-first failure PR resolve/create call with title/body forwarding"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -F --quiet "$branch_first_title_default" "$loop_script"; then
  echo "COMMANDS=$commands"
  echo "FAILURE_SUMMARY=missing optional-safe default for resolve_or_create_pr_for_branch pr_title argument"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -F --quiet "$branch_first_body_default" "$loop_script"; then
  echo "COMMANDS=$commands"
  echo "FAILURE_SUMMARY=missing optional-safe default for resolve_or_create_pr_for_branch pr_body argument"
  echo "STATUS=fail"
  exit 1
fi

echo "COMMANDS=$commands"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
