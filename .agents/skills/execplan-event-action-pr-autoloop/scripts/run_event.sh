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
  ".agents/skills/pr-autoloop/scripts/reviewer_daemon.sh"
)

for path in "${required_paths[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "COMMANDS=validate required pr-autoloop files"
    echo "FAILURE_SUMMARY=required path missing: $path"
    echo "STATUS=fail"
    exit 1
  fi
done

legacy_paths=(
  ".agents/skills/pr-autoloop/scripts/run_loop.sh"
)

for path in "${legacy_paths[@]}"; do
  if [[ -e "$path" ]]; then
    echo "COMMANDS=validate daemon-only pr-autoloop interface"
    echo "FAILURE_SUMMARY=legacy loop-era file must be removed: $path"
    echo "STATUS=fail"
    exit 1
  fi
done

commands=()
commands+=("bash -n .agents/skills/pr-autoloop/scripts/doctor.sh")
commands+=("bash -n .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh")
commands+=("rg -n AUTO_AGENT: REVIEWER|AUTO_REQUEST_ID|AUTO_RUN_ID|AUTO_ITERATION|AUTO_REVIEW_STATUS|AUTO_TARGET_COMMIT|APPROVE .agents/skills/pr-autoloop/references/comment_contract.md")
commands+=("rg -n -- --start|--request|--status|--stop|--commit|--pr-url|--head-branch|--request-id|--run-id|--iteration|--runtime-dir|--wait-timeout-sec .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh")
commands+=("rg -n reviewer.pid|state.json|inbox/<request_id>.json|responses/<request_id>.json|WAITING|RUNNING|APPROVED .agents/skills/pr-autoloop/references/state_schema.md")
commands+=("rg -n CI|do not wait .agents/skills/pr-autoloop/references/comment_contract.md")

if ! bash -n .agents/skills/pr-autoloop/scripts/doctor.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=doctor.sh syntax check failed"
  echo "STATUS=fail"
  exit 1
fi

if ! bash -n .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer_daemon.sh syntax check failed"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -q "AUTO_AGENT: REVIEWER" .agents/skills/pr-autoloop/references/comment_contract.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=comment contract missing AUTO_AGENT: REVIEWER"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "AUTO_REQUEST_ID" .agents/skills/pr-autoloop/references/comment_contract.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=comment contract missing AUTO_REQUEST_ID"
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
if ! rg -q "AUTO_RUN_ID" .agents/skills/pr-autoloop/references/comment_contract.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=comment contract missing AUTO_RUN_ID"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "AUTO_ITERATION" .agents/skills/pr-autoloop/references/comment_contract.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=comment contract missing AUTO_ITERATION"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "APPROVE" .agents/skills/pr-autoloop/references/comment_contract.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=comment contract missing APPROVE token rule"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -qi "do not wait" .agents/skills/pr-autoloop/references/comment_contract.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=comment contract missing non-blocking CI timing rule"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -q -- "--start" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --start"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--request" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --request"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--status" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --status"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--stop" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --stop"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--commit" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --commit"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--pr-url" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --pr-url"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--head-branch" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --head-branch"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--request-id" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --request-id"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--run-id" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --run-id"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--iteration" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --iteration"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--runtime-dir" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --runtime-dir"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q -- "--wait-timeout-sec" .agents/skills/pr-autoloop/scripts/reviewer_daemon.sh; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=reviewer daemon argument contract missing --wait-timeout-sec"
  echo "STATUS=fail"
  exit 1
fi

if ! rg -q "state.json" .agents/skills/pr-autoloop/references/state_schema.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=state schema missing state.json contract"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "reviewer.pid" .agents/skills/pr-autoloop/references/state_schema.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=state schema missing reviewer.pid contract"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "inbox/<request_id>.json" .agents/skills/pr-autoloop/references/state_schema.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=state schema missing inbox request contract"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "responses/<request_id>.json" .agents/skills/pr-autoloop/references/state_schema.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=state schema missing response contract"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "WAITING" .agents/skills/pr-autoloop/references/state_schema.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=state schema missing WAITING state"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "RUNNING" .agents/skills/pr-autoloop/references/state_schema.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=state schema missing RUNNING state"
  echo "STATUS=fail"
  exit 1
fi
if ! rg -q "APPROVED" .agents/skills/pr-autoloop/references/state_schema.md; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=state schema missing APPROVED state"
  echo "STATUS=fail"
  exit 1
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
