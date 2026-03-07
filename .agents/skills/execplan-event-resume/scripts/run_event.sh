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

tracking_field_is_missing() {
  local value="${1:-}"
  [[ -z "$value" || "$value" == "(see original)" || "$value" == "(not available locally)" ]]
}

format_tracking_date_from_iso8601() {
  local value="$1"
  [[ -n "$value" && "$value" != "null" ]] || return 1
  date -u -d "$value" +"%Y-%m-%d %H:%MZ" 2>/dev/null
}

derive_creation_commit_from_pr_json() {
  local pr_json="$1"
  jq -r '
    def stamp: (.committedDate // .authoredDate // "");
    (.createdAt // "") as $created_at
    | (
        [(.commits // [])[]? | select(stamp != "" and stamp <= $created_at)]
        | sort_by(stamp)
        | last
        | .oid
      )
      // (
        [(.commits // [])[]?]
        | first
        | .oid
      )
      // (.headRefOid // empty)
      // empty
  ' <<< "$pr_json"
}

commands=()
commands+=("git branch --show-current")

current_branch="$(git branch --show-current)"

extract_tracking_field() {
  local file="$1"
  local key="$2"
  [[ -f "$file" ]] || return 1
  sed -n -E "s/^- ${key}:[[:space:]]+(.+)$/\\1/p" "$file" | head -n1 | sed -E 's/[[:space:]]+$//'
}

extract_plan_tracking_doc() {
  sed -n -E 's/^- pr_tracking_doc:[[:space:]]+(.+)$/\1/p' "$PLAN" | head -n1 | sed -E 's/[[:space:]]+$//'
}

resolve_pr_metadata() {
  local branch_name="$1"
  local pr_list_json selected_json selected_url

  pr_url="${EXECPLAN_MANUAL_PR_URL:-"(not available locally)"}"
  pr_title="(not available locally)"
  pr_state="unknown"
  pr_head="$branch_name"
  pr_base="(unknown)"
  pr_created_at_raw=""
  pr_creation_commit_from_api=""

  if [[ "$gh_available" -ne 1 ]]; then
    return 0
  fi

  if [[ -n "${EXECPLAN_MANUAL_PR_URL:-}" && "${EXECPLAN_MANUAL_PR_URL}" != "(not available locally)" ]]; then
    selected_url="${EXECPLAN_MANUAL_PR_URL}"
  else
    commands+=("gh pr list --state open --head ${branch_name} --json url,updatedAt --limit 20")
    pr_list_json="$(gh pr list --state open --head "$branch_name" --json url,updatedAt --limit 20 2>/dev/null || true)"
    if [[ -n "$pr_list_json" ]]; then
      selected_url="$(jq -r '[.[]] | sort_by(.updatedAt) | reverse | .[0].url // empty' <<< "$pr_list_json")"
    else
      selected_url=""
    fi
  fi

  if [[ -z "$selected_url" || "$selected_url" == "null" ]]; then
    return 0
  fi

  commands+=("gh pr view ${selected_url} --json url,title,state,headRefName,baseRefName,createdAt,headRefOid,commits")
  selected_json="$(gh pr view "$selected_url" --json url,title,state,headRefName,baseRefName,createdAt,headRefOid,commits 2>/dev/null || true)"
  if [[ -z "$selected_json" || "$selected_json" == "null" ]]; then
    return 0
  fi

  pr_url="$(jq -r '.url // "(not available locally)"' <<< "$selected_json")"
  pr_title="$(jq -r '.title // "(not available locally)"' <<< "$selected_json")"
  pr_state="$(jq -r '.state // "unknown"' <<< "$selected_json")"
  pr_head="$(jq -r --arg branch "$branch_name" '.headRefName // $branch' <<< "$selected_json")"
  pr_base="$(jq -r '.baseRefName // "(unknown)"' <<< "$selected_json")"
  pr_created_at_raw="$(jq -r '.createdAt // empty' <<< "$selected_json")"
  pr_creation_commit_from_api="$(derive_creation_commit_from_pr_json "$selected_json")"
}

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
fi

tracking_path="$(extract_plan_tracking_doc || true)"
if [[ -n "$tracking_path" && "$tracking_path" != /* ]]; then
  tracking_path="${REPO_ROOT}/${tracking_path}"
fi
if [[ -z "$tracking_path" ]]; then
  tracking_path="${EXECPLAN_PR_TRACKING_PATH:-${REPO_ROOT}/eternal-cycler-out/prs/active/pr_${current_branch//\//_}.md}"
fi
commands+=("mkdir -p $(repo_rel_path "$(dirname "$tracking_path")")")
mkdir -p "$(dirname "$tracking_path")"
resume_date="$(date -u +"%Y-%m-%d %H:%MZ")"
resume_commit="$(git rev-parse HEAD)"

existing_pr_url="$(extract_tracking_field "$tracking_path" "PR link" || true)"
creation_date="$(extract_tracking_field "$tracking_path" "PR creation date" || true)"
creation_commit="$(extract_tracking_field "$tracking_path" "commit hash at PR creation time" || true)"

resolve_pr_metadata "$current_branch"

if [[ -n "$pr_url" && "$pr_url" != "(not available locally)" && "$existing_pr_url" != "$pr_url" ]]; then
  creation_date="$(format_tracking_date_from_iso8601 "$pr_created_at_raw" || true)"
  creation_commit="$pr_creation_commit_from_api"
fi
if tracking_field_is_missing "$creation_date"; then
  creation_date="$(format_tracking_date_from_iso8601 "$pr_created_at_raw" || true)"
fi
if tracking_field_is_missing "$creation_commit"; then
  creation_commit="$pr_creation_commit_from_api"
fi
if tracking_field_is_missing "$creation_date"; then
  creation_date="$resume_date"
fi
if tracking_field_is_missing "$creation_commit"; then
  creation_commit="$resume_commit"
fi

commands+=("update $(repo_rel_path "$tracking_path")")
cat > "$tracking_path" <<EOF
# PR Tracking: ${current_branch}

- PR link: ${pr_url}
- PR creation date: ${creation_date}
- branch name: ${current_branch}
- commit hash at PR creation time: ${creation_commit}
- summary/content of the PR: ${pr_title}
- PR state: ${pr_state}
- PR head/base: ${pr_head} -> ${pr_base}
- last resumed: ${resume_date}
EOF

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
