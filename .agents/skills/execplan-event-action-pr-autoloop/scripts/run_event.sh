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
  ".agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh"
  ".agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh"
)

for path in "${required_paths[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "COMMANDS=validate required fixed-loop files"
    echo "FAILURE_SUMMARY=required path missing: $path"
    echo "STATUS=fail"
    exit 1
  fi
done

for path in ".agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh" ".agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh"; do
  if [[ ! -x "$path" ]]; then
    echo "COMMANDS=validate required fixed-loop file executability"
    echo "FAILURE_SUMMARY=required script is not executable: $path"
    echo "STATUS=fail"
    exit 1
  fi
done

legacy_paths=(
  "scripts/run_builder_reviewer_doctor.sh"
  "scripts/run_builder_reviewer_loop.sh"
  ".agents/skills/pr-autoloop/scripts/reviewer_daemon.sh"
)

for path in "${legacy_paths[@]}"; do
  if [[ -e "$path" ]]; then
    echo "COMMANDS=validate legacy path removal"
    echo "FAILURE_SUMMARY=legacy path must be removed: $path"
    echo "STATUS=fail"
    exit 1
  fi
done

LOOP=".agents/skills/pr-autoloop/scripts/run_builder_reviewer_loop.sh"
DOCTOR=".agents/skills/pr-autoloop/scripts/run_builder_reviewer_doctor.sh"

commands=()
commands+=("bash -n $DOCTOR")
commands+=("bash -n $LOOP")
commands+=("rg -n -- --task|--task-file|--pr-url|--max-iterations|--max-builder-cleanup-retries|--max-reviewer-failures|--model-builder|--model-reviewer $LOOP")
commands+=("rg -n AUTO_AGENT: REVIEWER|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE $LOOP")
commands+=("rg -n fromdateiso8601|%ct $LOOP")
commands+=("rg -n gh\\ api\\ graphql $LOOP")
commands+=("rg -F -n comments(first:100 $LOOP")
commands+=("rg -F -n reviews(first:100 $LOOP")
commands+=("rg -n resolve_current_branch|headRefName|must match current local branch $LOOP")
commands+=("rg -n pr_state|pr_merged_at|state,mergedAt|OPEN and unmerged $LOOP")
commands+=("rg -n gh\\ auth\\ status|codex\\ login\\ status $DOCTOR")
commands+=("rg -n prompt_for_task_text|prompt_for_resume_target_if_needed|is_interactive_session|log_path|LOG_DIR $LOOP")
commands+=("rg -F -n if ! printf '%s\\n' \"\$prompt_text\" | \"\${cmd[@]}\"; then $LOOP")

if ! bash -n "$DOCTOR"; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=run_builder_reviewer_doctor.sh syntax check failed"
  echo "STATUS=fail"
  exit 1
fi

if ! bash -n "$LOOP"; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=run_builder_reviewer_loop.sh syntax check failed"
  echo "STATUS=fail"
  exit 1
fi

for flag in --task --task-file --pr-url --max-iterations --max-builder-cleanup-retries --max-reviewer-failures --model-builder --model-reviewer; do
  if ! rg -q -- "$flag" "$LOOP"; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=fixed loop argument contract missing $flag"
    echo "STATUS=fail"
    exit 1
  fi
done

if ! rg -q "auto_stage_commit_and_push" "$LOOP"; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script missing automatic stage/commit/push function"
  echo "STATUS=fail"
  exit 1
fi

if rg -q "prompt_for_task_text|prompt_for_resume_target_if_needed|is_interactive_session" "$LOOP"; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script still contains interactive prompt handlers"
  echo "STATUS=fail"
  exit 1
fi

if rg -q "log_path|LOG_DIR|builder_initial\\.log|reviewer_iter_|builder_cleanup_attempt_|builder_followup_iter_" "$LOOP"; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script still contains codex log-file persistence markers"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -Fq "if ! printf '%s\\n' \"\$prompt_text\" | \"\${cmd[@]}\"; then" "$LOOP"; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script missing direct codex stdout/stderr forwarding marker"
  echo "STATUS=fail"
  exit 1
fi

for marker in "fromdateiso8601" "%ct" "AUTO_AGENT: REVIEWER" "AUTO_REVIEW_STATUS" "AUTO_TARGET_COMMIT" "APPROVE" "gh api graphql" "comments(first:100" "reviews(first:100"; do
  if ! rg -Fq "$marker" "$LOOP"; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=loop script missing required marker: $marker"
    echo "STATUS=fail"
    exit 1
  fi
done

for marker in "resolve_current_branch" "headRefName" "must match current local branch"; do
  if ! rg -Fq "$marker" "$LOOP"; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=loop script missing branch-match assertion marker: $marker"
    echo "STATUS=fail"
    exit 1
  fi
done

if rg -q "ensure_branch_checked_out" "$LOOP"; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script must not switch branches automatically"
  echo "STATUS=fail"
  exit 1
fi

if rg -q "pr_state|pr_merged_at|state,mergedAt|OPEN and unmerged" "$LOOP"; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=loop script must not gate PR URL by OPEN/merged state; only branch match is allowed"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -q "gh auth status" "$DOCTOR"; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=doctor script missing gh auth status check"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "codex login status" "$DOCTOR"; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=doctor script missing codex login status check"
  echo "STATUS=fail"
  exit 1
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
