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
      echo "Usage: run_event.sh [--plan <plan_md>]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

has_plan_file=0
if [[ -n "$PLAN" && -f "$PLAN" ]]; then
  has_plan_file=1
fi

# REPO_ROOT: root of the consuming git repository (identical to ETERNAL_CYCLER_ROOT when
# eternal-cycler is the repo root; parent repo root when installed as a subtree/skill).
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"

commands=()
commands+=("git branch --show-current")
commands+=("git status --short")
commands+=("git log --oneline --decorate --max-count=20")

branch_before="$(git branch --show-current)"
status_before="$(git status --short || true)"
git log --oneline --decorate --max-count=20 >/dev/null

plan_rel=""
if [[ "$has_plan_file" -eq 1 ]]; then
  plan_rel="$PLAN"
  if [[ "$plan_rel" == "$PWD/"* ]]; then
    plan_rel="${plan_rel#"$PWD/"}"
  fi
fi

gh_available=0
if command -v gh >/dev/null 2>&1; then
  gh_available=1
  commands+=("gh pr status")
  commands+=("gh pr view --json number,title,body,state,headRefName,baseRefName,url")
  set +e
  gh pr status >/dev/null 2>&1
  gh pr view --json number,title,body,state,headRefName,baseRefName,url >/dev/null 2>&1
  set -e
fi

scope_alignment="${EXECPLAN_SCOPE_ALIGNMENT:-auto}"
case "$scope_alignment" in
  aligned|not_aligned|auto)
    ;;
  *)
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=invalid EXECPLAN_SCOPE_ALIGNMENT value: $scope_alignment (expected aligned|not_aligned|auto)"
    echo "STATUS=fail"
    exit 1
    ;;
esac

if [[ "$scope_alignment" == "auto" ]]; then
  if [[ "$branch_before" == "main" ]]; then
    scope_alignment="not_aligned"
  else
    scope_alignment="aligned"
  fi
fi

switch_required=0
if [[ "$branch_before" == "main" || "$scope_alignment" == "not_aligned" ]]; then
  switch_required=1
fi

new_branch_created=0
if [[ "$switch_required" -eq 1 ]]; then
  if [[ -n "$plan_rel" ]]; then
    status_for_switch="$(echo "$status_before" | awk -v plan="$plan_rel" '
      NF == 0 { next }
      $0 ~ (" " plan "$") { next }
      { print }
    ')"
  else
    status_for_switch="$status_before"
  fi
  if [[ -n "$status_for_switch" ]]; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=branch switch required but working tree is not clean; commit current work before switching"
    echo "STATUS=fail"
    exit 1
  fi

  new_branch="${EXECPLAN_NEW_BRANCH:-}"
  if [[ -z "$new_branch" ]]; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=branch switch required but EXECPLAN_NEW_BRANCH was not provided"
    echo "STATUS=fail"
    exit 1
  fi
  if [[ "$new_branch" != */* ]]; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=EXECPLAN_NEW_BRANCH must use <type>/<short-scope> format"
    echo "STATUS=fail"
    exit 1
  fi

  commands+=("git switch -c $new_branch")
  if ! git switch -c "$new_branch" >/dev/null 2>&1; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=failed to create/switch branch: $new_branch"
    echo "STATUS=fail"
    exit 1
  fi
  new_branch_created=1
fi

branch_after="$(git branch --show-current)"
status_after="$(git status --short || true)"

tracking_path="${EXECPLAN_PR_TRACKING_PATH:-${REPO_ROOT}/eternal-cycler-out/prs/active/pr_${branch_after//\//_}.md}"
commands+=("mkdir -p $(dirname "$tracking_path")")
mkdir -p "$(dirname "$tracking_path")"

creation_date="$(date -u +"%Y-%m-%d %H:%MZ")"
creation_commit="$(git rev-parse HEAD)"
pr_url="${EXECPLAN_MANUAL_PR_URL:-"(not available locally)"}"
pr_title="(not available locally)"
pr_state="unknown"
pr_head="$branch_after"
pr_base="(unknown)"

if [[ "$gh_available" -eq 1 ]]; then
  pr_url="$(gh pr view --json url --jq '.url' 2>/dev/null || echo "(not available locally)")"
  pr_title="$(gh pr view --json title --jq '.title' 2>/dev/null || echo "(not available locally)")"
  pr_state="$(gh pr view --json state --jq '.state' 2>/dev/null || echo "unknown")"
  pr_head="$(gh pr view --json headRefName --jq '.headRefName' 2>/dev/null || echo "$branch_after")"
  pr_base="$(gh pr view --json baseRefName --jq '.baseRefName' 2>/dev/null || echo "(unknown)")"
fi

cat > "$tracking_path" <<EOF
# PR Tracking: ${branch_after}

- PR link: ${pr_url}
- PR creation date: ${creation_date}
- branch name: ${branch_after}
- commit hash at PR creation time: ${creation_commit}
- summary/content of the PR: ${pr_title}
- PR state: ${pr_state}
- PR head/base: ${pr_head} -> ${pr_base}
EOF

if [[ "$has_plan_file" -eq 1 ]] && ! rg -q "$tracking_path" "$PLAN"; then
  cat >> "$PLAN" <<EOF

## PR Tracking Linkage

- pr_tracking_doc: ${tracking_path}
- execplan_start_branch: ${branch_after}
- execplan_start_commit: ${creation_commit}
EOF
fi

if [[ "$has_plan_file" -eq 1 ]] && ! rg -q "execplan_start_branch:" "$PLAN"; then
  cat >> "$PLAN" <<EOF
- execplan_start_branch: ${branch_after}
EOF
fi

if [[ "$has_plan_file" -eq 1 ]] && ! rg -q "execplan_start_commit:" "$PLAN"; then
  cat >> "$PLAN" <<EOF
- execplan_start_commit: ${creation_commit}
EOF
fi

if [[ "$has_plan_file" -eq 1 ]] && ! rg -q "<!-- execplan-start-tracked:start -->" "$PLAN"; then
  commands+=("capture execplan start tracked snapshot")
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

if [[ "$has_plan_file" -eq 1 ]] && ! rg -q "<!-- execplan-start-untracked:start -->" "$PLAN"; then
  commands+=("capture execplan start untracked snapshot")
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

commands+=("write/update $tracking_path")
if [[ "$has_plan_file" -eq 1 ]]; then
  commands+=("update plan linkage metadata")
fi

if [[ -n "$status_after" && "$new_branch_created" -eq 0 ]]; then
  # Reused branch is allowed to have in-progress edits.
  :
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
