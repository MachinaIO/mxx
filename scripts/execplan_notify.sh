#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  execplan_notify.sh --plan <plan_md> --event execplan.post_completion --status pass
USAGE
}

PLAN=""
EVENT=""
STATUS=""
POST_EVENT_ID="execplan.post_completion"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan)
      PLAN="${2:-}"
      shift 2
      ;;
    --event)
      EVENT="${2:-}"
      shift 2
      ;;
    --status)
      STATUS="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PLAN" || -z "$EVENT" || -z "$STATUS" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -f "$PLAN" ]]; then
  echo "Plan file not found: $PLAN" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI is required for PR notifications" >&2
  exit 1
fi

if [[ "$EVENT" != "$POST_EVENT_ID" ]]; then
  echo "Unsupported event for notify: $EVENT (only $POST_EVENT_ID is allowed)" >&2
  exit 2
fi

if [[ "$STATUS" != "pass" ]]; then
  echo "Unsupported status for notify: $STATUS (only pass is allowed)" >&2
  exit 2
fi

UPSTREAM_REF="$(git rev-parse --abbrev-ref --symbolic-full-name @{u} 2>/dev/null || true)"
if [[ -z "$UPSTREAM_REF" ]]; then
  echo "Current branch has no upstream; push branch before posting ExecPlan notifications" >&2
  exit 1
fi

LOCAL_HEAD="$(git rev-parse HEAD 2>/dev/null || true)"
UPSTREAM_HEAD="$(git rev-parse @{u} 2>/dev/null || true)"
if [[ -z "$LOCAL_HEAD" || -z "$UPSTREAM_HEAD" || "$LOCAL_HEAD" != "$UPSTREAM_HEAD" ]]; then
  echo "Current HEAD is not pushed to upstream; push before posting ExecPlan notifications" >&2
  exit 1
fi

PR_NUMBER="$(gh pr view --json number --jq '.number' 2>/dev/null || true)"
if [[ -z "$PR_NUMBER" ]]; then
  echo "No PR context found for current branch; cannot post notification" >&2
  exit 1
fi

if ! rg -q "<!-- verification-ledger:start -->" "$PLAN"; then
  echo "Verification Ledger block not found in plan: $PLAN" >&2
  exit 1
fi

LEDGER_LINE="$(
  sed -n '/<!-- verification-ledger:start -->/,/<!-- verification-ledger:end -->/p' "$PLAN" \
    | rg "event_id=${EVENT};" \
    | tail -n1 || true
)"
if [[ -z "$LEDGER_LINE" ]]; then
  echo "No ledger entry found for event $EVENT in Verification Ledger" >&2
  exit 1
fi

LEDGER_STATUS="$(echo "$LEDGER_LINE" | sed -nE 's/.*status=([^;]+);.*/\1/p' | xargs)"
if [[ "$LEDGER_STATUS" != "pass" ]]; then
  echo "Latest ledger entry for $EVENT is not pass; refusing to notify" >&2
  exit 1
fi

TIMESTAMP="$(date -u +"%Y-%m-%d %H:%MZ")"
REF="gh-pr-comment:pr-${PR_NUMBER}:event-${EVENT}:$TIMESTAMP"

BODY="$(printf "ExecPlan verification update\n\n- Plan: %s\n- Event: %s\n- Status: %s\n- Timestamp (UTC): %s\n- Ledger: %s\n" \
  "$PLAN" "$EVENT" "$STATUS" "$TIMESTAMP" "$LEDGER_LINE")"

gh pr comment "$PR_NUMBER" --body "$BODY" >/dev/null

echo "NOTIFY_REFERENCE=$REF"
