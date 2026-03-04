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
  "scripts/run_builder_reviewer_doctor.sh"
  "scripts/run_builder_reviewer_loop.sh"
)

for path in "${required_paths[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "COMMANDS=validate required fixed-loop files"
    echo "FAILURE_SUMMARY=required path missing: $path"
    echo "STATUS=fail"
    exit 1
  fi
  if [[ ! -x "$path" ]]; then
    echo "COMMANDS=validate required fixed-loop file executability"
    echo "FAILURE_SUMMARY=required script is not executable: $path"
    echo "STATUS=fail"
    exit 1
  fi
done

legacy_paths=(
  ".agents/skills/pr-autoloop"
  ".agents/skills/pr-autoloop/scripts/reviewer_daemon.sh"
)

for path in "${legacy_paths[@]}"; do
  if [[ -e "$path" ]]; then
    echo "COMMANDS=validate daemon-era removal"
    echo "FAILURE_SUMMARY=legacy daemon-era path must be removed: $path"
    echo "STATUS=fail"
    exit 1
  fi
done

commands=()
commands+=("bash -n scripts/run_builder_reviewer_doctor.sh")
commands+=("bash -n scripts/run_builder_reviewer_loop.sh")
commands+=("rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer scripts/run_builder_reviewer_loop.sh")
commands+=("rg -n AUTO_AGENT: REVIEWER|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE scripts/run_builder_reviewer_loop.sh")
commands+=("rg -n prompt_for_task_text|prompt_for_resume_target_if_needed|auto_stage_commit_and_push scripts/run_builder_reviewer_loop.sh")
commands+=("rg -n gh\\ api\\ graphql scripts/run_builder_reviewer_loop.sh")
commands+=("rg -F -n comments(first:100 scripts/run_builder_reviewer_loop.sh")
commands+=("rg -F -n reviews(first:100 scripts/run_builder_reviewer_loop.sh")
commands+=("rg -n mergedAt|state|OPEN|headRefName scripts/run_builder_reviewer_loop.sh scripts/run_builder_reviewer_doctor.sh")
commands+=("rg -n gh\\ auth\\ status|codex\\ login\\ status scripts/run_builder_reviewer_doctor.sh")

if ! bash -n scripts/run_builder_reviewer_doctor.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=run_builder_reviewer_doctor.sh syntax check failed"
  echo "STATUS=fail"
  exit 1
fi

if ! bash -n scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=run_builder_reviewer_loop.sh syntax check failed"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -q -- "--task" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=fixed loop argument contract missing --task"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--task-file" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=fixed loop argument contract missing --task-file"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--pr-url" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=fixed loop argument contract missing --pr-url"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--max-iterations" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=fixed loop argument contract missing --max-iterations"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--max-builder-cleanup-retries" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=fixed loop argument contract missing --max-builder-cleanup-retries"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--max-reviewer-failures" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=fixed loop argument contract missing --max-reviewer-failures"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--model-builder" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=fixed loop argument contract missing --model-builder"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--model-reviewer" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=fixed loop argument contract missing --model-reviewer"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "prompt_for_task_text" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script missing interactive task prompt handler"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "prompt_for_resume_target_if_needed" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script missing interactive active-PR resume selection handler"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "auto_stage_commit_and_push" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script missing automatic stage/commit/push function"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -q "AUTO_AGENT: REVIEWER" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer prompt contract missing AUTO_AGENT: REVIEWER"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "AUTO_REVIEW_STATUS" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer prompt contract missing AUTO_REVIEW_STATUS"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "AUTO_TARGET_COMMIT" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer prompt contract missing AUTO_TARGET_COMMIT"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "APPROVE" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer prompt contract missing APPROVE token handling"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -q "gh api graphql" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script missing GraphQL comment/review retrieval"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -Fq "comments(first:100" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script missing issue comment collection query"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -Fq "reviews(first:100" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script missing review collection query"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -q "mergedAt" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script missing merged PR input guard"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "state" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script missing PR state input guard"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "OPEN" scripts/run_builder_reviewer_loop.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script missing OPEN-only PR guard"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "gh auth status" scripts/run_builder_reviewer_doctor.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=doctor script missing gh auth status check"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "codex login status" scripts/run_builder_reviewer_doctor.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=doctor script missing codex login status check"
  echo "STATUS=fail"
  exit 1
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
