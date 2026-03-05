#!/usr/bin/env bash
set -euo pipefail

# execplan.post_creation — run immediately after the new plan document is written.
# Records start snapshot, creates PR tracking doc, and writes plan linkage metadata.
# Requires --plan. Branch management is the caller's responsibility.

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

plan_rel="$PLAN"
if [[ "$plan_rel" == "$PWD/"* ]]; then
  plan_rel="${plan_rel#"$PWD/"}"
fi

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"

commands=()
commands+=("git branch --show-current")
commands+=("git status --short")

branch="$(git branch --show-current)"
git status --short >/dev/null

resolve_pr_metadata() {
  local branch_name="$1"
  local pr_list_json selected_json

  pr_url="${EXECPLAN_MANUAL_PR_URL:-"(not available locally)"}"
  pr_title="(not available locally)"
  pr_state="unknown"
  pr_head="$branch_name"
  pr_base="(unknown)"

  if [[ "$gh_available" -ne 1 ]]; then
    return 0
  fi

  if [[ -n "${EXECPLAN_MANUAL_PR_URL:-}" && "${EXECPLAN_MANUAL_PR_URL}" != "(not available locally)" ]]; then
    commands+=("gh pr view ${EXECPLAN_MANUAL_PR_URL} --json url,title,state,headRefName,baseRefName")
    selected_json="$(gh pr view "${EXECPLAN_MANUAL_PR_URL}" --json url,title,state,headRefName,baseRefName 2>/dev/null || true)"
  else
    commands+=("gh pr list --state open --head ${branch_name} --json url,title,state,headRefName,baseRefName,updatedAt --limit 20")
    pr_list_json="$(gh pr list --state open --head "$branch_name" --json url,title,state,headRefName,baseRefName,updatedAt --limit 20 2>/dev/null || true)"
    if [[ -n "$pr_list_json" ]]; then
      selected_json="$(jq -c '[.[]] | sort_by(.updatedAt) | reverse | .[0] // empty' <<< "$pr_list_json")"
    else
      selected_json=""
    fi
  fi

  if [[ -z "$selected_json" || "$selected_json" == "null" ]]; then
    return 0
  fi

  pr_url="$(jq -r '.url // "(not available locally)"' <<< "$selected_json")"
  pr_title="$(jq -r '.title // "(not available locally)"' <<< "$selected_json")"
  pr_state="$(jq -r '.state // "unknown"' <<< "$selected_json")"
  pr_head="$(jq -r --arg branch "$branch_name" '.headRefName // $branch' <<< "$selected_json")"
  pr_base="$(jq -r '.baseRefName // "(unknown)"' <<< "$selected_json")"
}

gh_available=0
if command -v gh >/dev/null 2>&1; then
  gh_available=1
  commands+=("gh pr status")
  set +e
  gh pr status >/dev/null 2>&1
  set -e
fi

tracking_path="${EXECPLAN_PR_TRACKING_PATH:-${REPO_ROOT}/eternal-cycler-out/prs/active/pr_${branch//\//_}.md}"
# Repo-relative version for writing into the plan document (policy: no absolute paths in docs).
tracking_path_rel="${tracking_path#"${REPO_ROOT}/"}"
commands+=("mkdir -p $(dirname "$tracking_path")")
mkdir -p "$(dirname "$tracking_path")"

creation_date="$(date -u +"%Y-%m-%d %H:%MZ")"
creation_commit="$(git rev-parse HEAD)"
resolve_pr_metadata "$branch"

commands+=("write $tracking_path")
cat > "$tracking_path" <<EOF
# PR Tracking: ${branch}

- PR link: ${pr_url}
- PR creation date: ${creation_date}
- branch name: ${branch}
- commit hash at PR creation time: ${creation_commit}
- summary/content of the PR: ${pr_title}
- PR state: ${pr_state}
- PR head/base: ${pr_head} -> ${pr_base}
EOF

if ! rg -q "^-[[:space:]]+pr_tracking_doc:" "$PLAN"; then
  commands+=("append PR Tracking Linkage to plan")
  cat >> "$PLAN" <<EOF

## PR Tracking Linkage

- pr_tracking_doc: ${tracking_path_rel}
EOF
fi

if ! rg -q "execplan_start_branch:" "$PLAN"; then
  cat >> "$PLAN" <<EOF

- execplan_start_branch: ${branch}
EOF
fi

if ! rg -q "execplan_start_commit:" "$PLAN"; then
  cat >> "$PLAN" <<EOF

- execplan_start_commit: ${creation_commit}
EOF
fi

if ! rg -q "<!-- execplan-start-tracked:start -->" "$PLAN"; then
  commands+=("capture start tracked snapshot")
  {
    echo
    echo "## ExecPlan Start Snapshot"
    echo
    echo "<!-- execplan-start-tracked:start -->"

    snapshot_count=0
    while IFS= read -r path; do
      [[ -z "$path" ]] && continue
      hash="(deleted)"
      if [[ -e "$path" ]]; then
        hash="$(git hash-object -- "$path" 2>/dev/null || echo "(missing)")"
      fi
      printf -- "- start_tracked_change: %s\t%s\n" "$hash" "$path"
      snapshot_count=$((snapshot_count + 1))
    done < <(git diff --name-only HEAD -- | sort)

    if [[ "$snapshot_count" -eq 0 ]]; then
      echo "- start_tracked_change: (none)	(none)"
    fi

    echo "<!-- execplan-start-tracked:end -->"
  } >> "$PLAN"
fi

if ! rg -q "<!-- execplan-start-untracked:start -->" "$PLAN"; then
  commands+=("capture start untracked snapshot")
  {
    echo
    echo "<!-- execplan-start-untracked:start -->"

    snapshot_count=0
    while IFS= read -r path; do
      [[ -z "$path" ]] && continue
      hash="$(git hash-object -- "$path" 2>/dev/null || echo "(missing)")"
      printf -- "- start_untracked_file: %s\t%s\n" "$hash" "$path"
      snapshot_count=$((snapshot_count + 1))
    done < <(git ls-files --others --exclude-standard | sort)

    if [[ "$snapshot_count" -eq 0 ]]; then
      echo "- start_untracked_file: (none)	(none)"
    fi

    echo "<!-- execplan-start-untracked:end -->"
  } >> "$PLAN"
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
