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

extract_tracking_field() {
  local file="$1"
  local key="$2"
  [[ -f "$file" ]] || return 1
  sed -n -E "s/^- ${key}:[[:space:]]+(.+)$/\\1/p" "$file" | head -n1 | sed -E 's/[[:space:]]+$//'
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
commands+=("git status --short")

branch="$(git branch --show-current)"
git status --short >/dev/null

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
commands+=("mkdir -p $(repo_rel_path "$(dirname "$tracking_path")")")
mkdir -p "$(dirname "$tracking_path")"

creation_date="$(date -u +"%Y-%m-%d %H:%MZ")"
creation_commit="$(git rev-parse HEAD)"
existing_pr_url="$(extract_tracking_field "$tracking_path" "PR link" || true)"
existing_creation_date="$(extract_tracking_field "$tracking_path" "PR creation date" || true)"
existing_creation_commit="$(extract_tracking_field "$tracking_path" "commit hash at PR creation time" || true)"
resolve_pr_metadata "$branch"

if [[ -n "$pr_url" && "$pr_url" != "(not available locally)" ]]; then
  if [[ "$existing_pr_url" == "$pr_url" ]]; then
    creation_date="$existing_creation_date"
    creation_commit="$existing_creation_commit"
  else
    creation_date="$(format_tracking_date_from_iso8601 "$pr_created_at_raw" || true)"
    creation_commit="$pr_creation_commit_from_api"
  fi
else
  creation_date="$existing_creation_date"
  creation_commit="$existing_creation_commit"
fi

if tracking_field_is_missing "$creation_date"; then
  creation_date="$(format_tracking_date_from_iso8601 "$pr_created_at_raw" || true)"
fi
if tracking_field_is_missing "$creation_commit"; then
  creation_commit="$pr_creation_commit_from_api"
fi
if tracking_field_is_missing "$creation_date"; then
  creation_date="$(date -u +"%Y-%m-%d %H:%MZ")"
fi
if tracking_field_is_missing "$creation_commit"; then
  creation_commit="$(git rev-parse HEAD)"
fi

commands+=("write $(repo_rel_path "$tracking_path")")
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
