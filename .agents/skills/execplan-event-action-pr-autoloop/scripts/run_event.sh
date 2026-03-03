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

required_paths=(
  ".agents/skills/pr-autoloop/SKILL.md"
  ".agents/skills/pr-autoloop/agents/openai.yaml"
  ".agents/skills/pr-autoloop/references/comment_contract.md"
  ".agents/skills/pr-autoloop/references/state_schema.md"
  ".agents/skills/pr-autoloop/scripts/doctor.sh"
  ".agents/skills/pr-autoloop/scripts/run_loop.sh"
)

for path in "${required_paths[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "COMMANDS=validate required pr-autoloop files"
    echo "FAILURE_SUMMARY=required path missing: $path"
    echo "STATUS=fail"
    exit 1
  fi
done

commands=()
commands+=("bash -n .agents/skills/pr-autoloop/scripts/doctor.sh")
commands+=("bash -n .agents/skills/pr-autoloop/scripts/run_loop.sh")
commands+=(".agents/skills/pr-autoloop/scripts/run_loop.sh --self-test")
commands+=("rg -n AUTO_AGENT: BUILDER|AUTO_AGENT: REVIEWER|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT .agents/skills/pr-autoloop/references/comment_contract.md")
commands+=("rg -n -- --goal-file|--pr-url|--max-builder-failures|--max-iterations .agents/skills/pr-autoloop/scripts/run_loop.sh")
commands+=("rg -n run_id|pr_url|consecutive_builder_failures|last_reviewer_status .agents/skills/pr-autoloop/references/state_schema.md")

if ! bash -n .agents/skills/pr-autoloop/scripts/doctor.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=doctor.sh syntax check failed"
  echo "STATUS=fail"
  exit 1
fi

if ! bash -n .agents/skills/pr-autoloop/scripts/run_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=run_loop.sh syntax check failed"
  echo "STATUS=fail"
  exit 1
fi

if ! .agents/skills/pr-autoloop/scripts/run_loop.sh --self-test >/dev/null 2>&1; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=run_loop.sh self-test failed"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -q "AUTO_AGENT: BUILDER" .agents/skills/pr-autoloop/references/comment_contract.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=comment contract missing AUTO_AGENT: BUILDER"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "AUTO_AGENT: REVIEWER" .agents/skills/pr-autoloop/references/comment_contract.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=comment contract missing AUTO_AGENT: REVIEWER"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "AUTO_REVIEW_STATUS" .agents/skills/pr-autoloop/references/comment_contract.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=comment contract missing AUTO_REVIEW_STATUS"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "AUTO_TARGET_COMMIT" .agents/skills/pr-autoloop/references/comment_contract.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=comment contract missing AUTO_TARGET_COMMIT"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -q -- "--goal-file" .agents/skills/pr-autoloop/scripts/run_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=run_loop argument contract missing --goal-file"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--pr-url" .agents/skills/pr-autoloop/scripts/run_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=run_loop argument contract missing --pr-url"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--max-builder-failures" .agents/skills/pr-autoloop/scripts/run_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=run_loop argument contract missing --max-builder-failures"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--max-iterations" .agents/skills/pr-autoloop/scripts/run_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=run_loop argument contract missing --max-iterations"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -q "run_id" .agents/skills/pr-autoloop/references/state_schema.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=state schema missing run_id field"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "consecutive_builder_failures" .agents/skills/pr-autoloop/references/state_schema.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=state schema missing consecutive_builder_failures field"
  echo "STATUS=fail"
  exit 1
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
