#!/usr/bin/env bash
set -euo pipefail

# execplan.resume — run when resuming an existing plan.
# Validates that the current branch matches the plan's start branch,
# refreshes the PR tracking doc, and appends a resume record to the plan.
# Branch management is the caller's responsibility.

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
  echo "FAILURE_SUMMARY=--plan is required and must point to an existing file"
  echo "STATUS=fail"
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"

commands=()
commands+=("git branch --show-current")

current_branch="$(git branch --show-current)"

# Validate branch matches plan's recorded start branch.
start_branch="$(grep -m1 'execplan_start_branch:' "$PLAN" | sed 's/.*execplan_start_branch:[[:space:]]*//' | tr -d '[:space:]' || true)"
if [[ -z "$start_branch" ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=execplan_start_branch not found in plan; was execplan.post_creation run?"
  echo "STATUS=fail"
  exit 1
fi

if [[ "$current_branch" != "$start_branch" ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=current branch '${current_branch}' does not match plan start branch '${start_branch}'; switch to '${start_branch}' before resuming"
  echo "STATUS=fail"
  exit 1
fi

# Refresh PR tracking doc.
gh_available=0
if command -v gh >/dev/null 2>&1; then
  gh_available=1
  commands+=("gh pr view --json number,title,body,state,headRefName,baseRefName,url")
  set +e
  gh pr view --json number,title,body,state,headRefName,baseRefName,url >/dev/null 2>&1
  set -e
fi

tracking_path="${EXECPLAN_PR_TRACKING_PATH:-${REPO_ROOT}/eternal-cycler-out/prs/active/pr_${current_branch//\//_}.md}"
resume_date="$(date -u +"%Y-%m-%d %H:%MZ")"
resume_commit="$(git rev-parse HEAD)"

pr_url="${EXECPLAN_MANUAL_PR_URL:-"(not available locally)"}"
pr_title="(not available locally)"
pr_state="unknown"
pr_head="$current_branch"
pr_base="(unknown)"

if [[ "$gh_available" -eq 1 ]]; then
  pr_url="$(gh pr view --json url --jq '.url' 2>/dev/null || echo "(not available locally)")"
  pr_title="$(gh pr view --json title --jq '.title' 2>/dev/null || echo "(not available locally)")"
  pr_state="$(gh pr view --json state --jq '.state' 2>/dev/null || echo "unknown")"
  pr_head="$(gh pr view --json headRefName --jq '.headRefName' 2>/dev/null || echo "$current_branch")"
  pr_base="$(gh pr view --json baseRefName --jq '.baseRefName' 2>/dev/null || echo "(unknown)")"
fi

if [[ -f "$tracking_path" ]]; then
  commands+=("update $tracking_path")
  cat > "$tracking_path" <<EOF
# PR Tracking: ${current_branch}

- PR link: ${pr_url}
- PR creation date: (see original)
- branch name: ${current_branch}
- commit hash at PR creation time: (see original)
- summary/content of the PR: ${pr_title}
- PR state: ${pr_state}
- PR head/base: ${pr_head} -> ${pr_base}
- last resumed: ${resume_date}
EOF
fi

# Append resume record to plan (idempotent: skip if this resume_commit already recorded).
if ! rg -q "resume_commit: ${resume_commit}" "$PLAN"; then
  commands+=("append resume record to plan")
  cat >> "$PLAN" <<EOF

## ExecPlan Resume Record

- resume_date: ${resume_date}
- resume_commit: ${resume_commit}
- operator_feedback: (none)
EOF
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
