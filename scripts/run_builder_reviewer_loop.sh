#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_builder_reviewer_loop.sh (--task <text> | --task-file <path>) [options]

Options:
  --pr-url <url>                     Reuse an existing OPEN (unmerged) PR and its head branch.
  --max-iterations <n>               Max review iterations (default: 20).
  --max-builder-cleanup-retries <n>  Max builder cleanup retries per cycle (default: 5).
  --max-reviewer-failures <n>        Max consecutive reviewer-phase failures (default: 3).
  --model-builder <model>            Optional model for builder codex runs.
  --model-reviewer <model>           Optional model for reviewer codex runs.
  --help                             Show this help.
USAGE
}

log() {
  echo "[loop $(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*" >&2
}

die() {
  log "ERROR: $*"
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "missing required command: $cmd"
}

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

resolve_repo_root() {
  git rev-parse --show-toplevel 2>/dev/null || pwd
}

resolve_current_branch() {
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ -z "$branch" || "$branch" == "HEAD" ]]; then
    return 1
  fi
  printf '%s\n' "$branch"
}

contains_approve_token() {
  local body="$1"
  grep -Eq '(^|[[:space:]])APPROVE($|[[:space:]])' <<< "$body"
}

extract_tag_value() {
  local body="$1"
  local key="$2"
  local line prefix value

  while IFS= read -r line; do
    prefix="${key}:"
    if [[ "$line" == "$prefix"* ]]; then
      value="${line#"$prefix"}"
      value="$(echo "$value" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
      if [[ -n "$value" ]]; then
        printf '%s\n' "$value"
        return 0
      fi
    fi
  done <<< "$body"

  return 1
}

has_tracked_dirty() {
  if ! git diff --quiet --; then
    return 0
  fi
  if ! git diff --cached --quiet --; then
    return 0
  fi
  if [[ -n "$(git ls-files -u)" ]]; then
    return 0
  fi
  return 1
}

head_is_pushed() {
  local local_head remote_head
  local_head="$(git rev-parse HEAD)"
  remote_head="$(git ls-remote --heads origin "$TARGET_BRANCH" | awk '{print $1}' | head -n1)"
  [[ -n "$remote_head" && "$remote_head" == "$local_head" ]]
}

ensure_branch_checked_out() {
  local target_branch="$1"
  local current_branch

  current_branch="$(resolve_current_branch || true)"
  if [[ "$current_branch" == "$target_branch" ]]; then
    return 0
  fi

  if git show-ref --verify --quiet "refs/heads/$target_branch"; then
    git switch "$target_branch" >/dev/null 2>&1 || die "failed to switch to branch: $target_branch"
    return 0
  fi

  if git ls-remote --exit-code --heads origin "$target_branch" >/dev/null 2>&1; then
    git switch -c "$target_branch" --track "origin/$target_branch" >/dev/null 2>&1 || die "failed to create tracking branch from origin/$target_branch"
    return 0
  fi

  die "cannot find branch locally or on origin: $target_branch"
}

parse_pr_coordinates() {
  local pr_url="$1"
  if [[ "$pr_url" =~ ^https://github\.com/([^/]+)/([^/]+)/pull/([0-9]+)$ ]]; then
    PR_OWNER="${BASH_REMATCH[1]}"
    PR_REPO="${BASH_REMATCH[2]}"
    PR_NUMBER="${BASH_REMATCH[3]}"
    return 0
  fi
  return 1
}

run_codex_prompt() {
  local role="$1"
  local model="$2"
  local prompt_text="$3"
  local log_path="$4"

  local cmd=(codex exec --dangerously-bypass-approvals-and-sandbox --cd "$WORKDIR")
  if [[ -n "$model" ]]; then
    cmd+=(--model "$model")
  fi
  cmd+=(-)

  if ! printf '%s\n' "$prompt_text" | "${cmd[@]}" >"$log_path" 2>&1; then
    log "$role codex execution failed (log: $log_path)"
    return 1
  fi

  return 0
}

get_new_untracked_paths() {
  local path
  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    if [[ -z "${BASELINE_UNTRACKED[$path]+x}" ]]; then
      printf '%s\n' "$path"
    fi
  done < <(git ls-files --others --exclude-standard)
}

upsert_bullet_line() {
  local file="$1"
  local key="$2"
  local value="$3"
  local prefix tmp

  prefix="- ${key}:"
  tmp="$(mktemp)"

  awk -v prefix="$prefix" -v key="$key" -v value="$value" '
    BEGIN { found = 0 }
    {
      if (index($0, prefix) == 1) {
        print "- " key ": " value
        found = 1
        next
      }
      print
    }
    END {
      if (!found) {
        print "- " key ": " value
      }
    }
  ' "$file" > "$tmp"

  mv "$tmp" "$file"
}

find_pr_tracking_doc() {
  local pr_url="$1"
  local dir found_path

  for dir in docs/prs/active docs/prs/completed; do
    [[ -d "$dir" ]] || continue
    found_path="$(rg -l -F -- "- PR link: $pr_url" "$dir" 2>/dev/null | head -n1 || true)"
    if [[ -n "$found_path" ]]; then
      printf '%s\n' "$found_path"
      return 0
    fi
  done

  return 1
}

finalize_pr_tracking_doc() {
  local pr_url="$1"
  local target_branch="$2"
  local approved_commit="$3"
  local source_path default_active_path default_completed_path completed_path moved_from
  local now_utc

  default_active_path="docs/prs/active/pr_${target_branch//\//_}.md"
  default_completed_path="docs/prs/completed/pr_${target_branch//\//_}.md"

  source_path="$(find_pr_tracking_doc "$pr_url" || true)"
  if [[ -z "$source_path" ]]; then
    if [[ -f "$default_active_path" ]]; then
      source_path="$default_active_path"
    elif [[ -f "$default_completed_path" ]]; then
      source_path="$default_completed_path"
    else
      source_path="$default_completed_path"
      mkdir -p "$(dirname "$source_path")"
      now_utc="$(date -u +"%Y-%m-%d %H:%MZ")"
      cat > "$source_path" <<EOF
# PR Tracking: ${target_branch}

- PR link: ${pr_url}
- PR creation date: ${now_utc}
- branch name: ${target_branch}
- commit hash at PR creation time: ${approved_commit}
- summary/content of the PR: (set by run_builder_reviewer_loop.sh)
- PR state: OPEN
- PR head/base: ${target_branch} -> (unknown)
EOF
    fi
  fi

  if [[ "$source_path" == docs/prs/completed/* ]]; then
    completed_path="$source_path"
  elif [[ "$source_path" == docs/prs/active/* ]]; then
    completed_path="docs/prs/completed/$(basename "$source_path")"
  else
    completed_path="$default_completed_path"
  fi

  mkdir -p "$(dirname "$completed_path")"

  moved_from=""
  if [[ "$source_path" != "$completed_path" ]]; then
    mv "$source_path" "$completed_path"
    moved_from="$source_path"
  fi

  now_utc="$(date -u +"%Y-%m-%d %H:%MZ")"
  upsert_bullet_line "$completed_path" "PR link" "$pr_url"
  upsert_bullet_line "$completed_path" "branch name" "$target_branch"
  upsert_bullet_line "$completed_path" "PR state" "OPEN"
  upsert_bullet_line "$completed_path" "review state" "OPEN"
  upsert_bullet_line "$completed_path" "tracking state" "COMPLETED"
  upsert_bullet_line "$completed_path" "completion commit" "$approved_commit"
  upsert_bullet_line "$completed_path" "completed at" "$now_utc"

  if [[ -n "$moved_from" ]]; then
    git add -A -- "$moved_from" "$completed_path" >/dev/null 2>&1 || die "failed to stage PR tracking move/update"
  else
    git add -A -- "$completed_path" >/dev/null 2>&1 || die "failed to stage PR tracking update"
  fi

  if ! git diff --cached --quiet; then
    git commit -m "docs(pr): complete tracking on reviewer approval" >/dev/null 2>&1 || die "failed to commit PR tracking completion update"
    git push origin "$target_branch" >/dev/null 2>&1 || die "failed to push PR tracking completion update to origin/$target_branch"
    log "PR tracking document finalized and pushed: $completed_path"
  else
    log "PR tracking document already finalized: $completed_path"
  fi
}

resolve_or_create_pr_for_branch() {
  local branch="$1"
  local open_json open_url create_out

  open_json="$(gh pr list --state open --head "$branch" --json url,updatedAt --limit 20 2>/dev/null || true)"
  open_url="$(jq -r '[.[]] | sort_by(.updatedAt) | reverse | .[0].url // empty' <<< "$open_json")"
  if [[ -n "$open_url" ]]; then
    printf '%s\n' "$open_url"
    return 0
  fi

  set +e
  create_out="$(gh pr create --fill --head "$branch" 2>&1)"
  local rc=$?
  set -e
  if [[ "$rc" -ne 0 ]]; then
    log "gh pr create failed for branch $branch: $create_out"
  fi

  open_json="$(gh pr list --state open --head "$branch" --json url,updatedAt --limit 20 2>/dev/null || true)"
  open_url="$(jq -r '[.[]] | sort_by(.updatedAt) | reverse | .[0].url // empty' <<< "$open_json")"
  if [[ -z "$open_url" ]]; then
    die "failed to resolve an OPEN PR for branch '$branch'"
  fi

  printf '%s\n' "$open_url"
}

fetch_self_comments_since() {
  local pr_url="$1"
  local self_login="$2"
  local since_iso="$3"

  local issue_query review_query
  local issue_cursor review_cursor
  local issue_has_next review_has_next
  local response
  local tmp

  parse_pr_coordinates "$pr_url" || die "invalid PR URL format: $pr_url"

  issue_query='query($owner:String!, $name:String!, $number:Int!, $cursor:String) { repository(owner:$owner, name:$name) { pullRequest(number:$number) { comments(first:100, after:$cursor) { nodes { url body createdAt author { login } } pageInfo { hasNextPage endCursor } } } } }'
  review_query='query($owner:String!, $name:String!, $number:Int!, $cursor:String) { repository(owner:$owner, name:$name) { pullRequest(number:$number) { reviews(first:100, after:$cursor) { nodes { url body submittedAt author { login } } pageInfo { hasNextPage endCursor } } } } }'

  tmp="$(mktemp)"

  issue_cursor=""
  while true; do
    response="$(gh api graphql -f query="$issue_query" -F owner="$PR_OWNER" -F name="$PR_REPO" -F number="$PR_NUMBER" -F cursor="$issue_cursor" 2>/dev/null || true)"
    [[ -n "$response" ]] || { rm -f "$tmp"; return 1; }

    jq -c --arg login "$self_login" --arg since "$since_iso" '
      .data.repository.pullRequest.comments.nodes[]
      | select(.author.login == $login)
      | select(.createdAt > $since)
      | {url:.url, body:(.body // ""), time:.createdAt, kind:"issue_comment"}
    ' <<< "$response" >> "$tmp"

    issue_has_next="$(jq -r '.data.repository.pullRequest.comments.pageInfo.hasNextPage // false' <<< "$response")"
    issue_cursor="$(jq -r '.data.repository.pullRequest.comments.pageInfo.endCursor // ""' <<< "$response")"
    [[ "$issue_has_next" == "true" ]] || break
  done

  review_cursor=""
  while true; do
    response="$(gh api graphql -f query="$review_query" -F owner="$PR_OWNER" -F name="$PR_REPO" -F number="$PR_NUMBER" -F cursor="$review_cursor" 2>/dev/null || true)"
    [[ -n "$response" ]] || { rm -f "$tmp"; return 1; }

    jq -c --arg login "$self_login" --arg since "$since_iso" '
      .data.repository.pullRequest.reviews.nodes[]
      | select(.author.login == $login)
      | select(.submittedAt != null)
      | select(.submittedAt > $since)
      | {url:.url, body:(.body // ""), time:.submittedAt, kind:"review"}
    ' <<< "$response" >> "$tmp"

    review_has_next="$(jq -r '.data.repository.pullRequest.reviews.pageInfo.hasNextPage // false' <<< "$response")"
    review_cursor="$(jq -r '.data.repository.pullRequest.reviews.pageInfo.endCursor // ""' <<< "$response")"
    [[ "$review_has_next" == "true" ]] || break
  done

  if [[ -s "$tmp" ]]; then
    jq -s 'sort_by(.time)' "$tmp"
  else
    printf '[]\n'
  fi

  rm -f "$tmp"
}

run_builder_cleanup_until_stable() {
  local base_commit="$1"
  local attempt=0
  local current_commit tracked_dirty pushed
  local cleanup_prompt cleanup_log
  local untracked_msg

  while true; do
    current_commit="$(git rev-parse HEAD)"

    tracked_dirty=0
    if has_tracked_dirty; then
      tracked_dirty=1
    fi

    mapfile -t NEW_UNTRACKED < <(get_new_untracked_paths)

    pushed=0
    if head_is_pushed; then
      pushed=1
    fi

    if [[ "$current_commit" != "$base_commit" && "$tracked_dirty" -eq 0 && ${#NEW_UNTRACKED[@]} -eq 0 && "$pushed" -eq 1 ]]; then
      printf 'changed|%s\n' "$current_commit"
      return 0
    fi

    if [[ "$current_commit" == "$base_commit" && "$tracked_dirty" -eq 0 && ${#NEW_UNTRACKED[@]} -eq 0 && "$pushed" -eq 1 ]]; then
      printf 'unchanged|%s\n' "$current_commit"
      return 0
    fi

    if [[ "$attempt" -ge "$MAX_BUILDER_CLEANUP_RETRIES" ]]; then
      log "cleanup retries exhausted for base_commit=$base_commit (current=$current_commit, tracked_dirty=$tracked_dirty, new_untracked=${#NEW_UNTRACKED[@]}, pushed=$pushed)"
      return 1
    fi

    attempt=$((attempt + 1))

    if [[ ${#NEW_UNTRACKED[@]} -gt 0 ]]; then
      untracked_msg="$(printf '%s\n' "${NEW_UNTRACKED[@]}")"
    else
      untracked_msg="(none)"
    fi

    cleanup_prompt="You are the BUILDER agent in an autonomous loop.

Cleanup request:
- Base commit before your previous attempt: ${base_commit}
- Current commit: ${current_commit}
- Target branch: ${TARGET_BRANCH}
- tracked_dirty: ${tracked_dirty}
- new_untracked_outside_baseline:
${untracked_msg}
- pushed_to_origin: ${pushed}

Do the following now:
1) Resolve remaining tracked changes or newly-created untracked files produced by your previous work.
2) Commit required changes.
3) Push the current branch (${TARGET_BRANCH}) to origin.
4) Do not modify unrelated baseline untracked files that existed before this loop started.

After finishing, exit."

    cleanup_log="$LOG_DIR/builder_cleanup_attempt_${attempt}.log"
    if ! run_codex_prompt "builder_cleanup" "$MODEL_BUILDER" "$cleanup_prompt" "$cleanup_log"; then
      log "builder cleanup attempt ${attempt} failed"
      return 1
    fi
  done
}

TASK_TEXT=""
TASK_FILE=""
PR_URL=""
MAX_ITERATIONS=20
MAX_BUILDER_CLEANUP_RETRIES=5
MAX_REVIEWER_FAILURES=3
MODEL_BUILDER=""
MODEL_REVIEWER=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --task)
      TASK_TEXT="${2:-}"
      shift 2
      ;;
    --task-file)
      TASK_FILE="${2:-}"
      shift 2
      ;;
    --pr-url)
      PR_URL="${2:-}"
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="${2:-}"
      shift 2
      ;;
    --max-builder-cleanup-retries)
      MAX_BUILDER_CLEANUP_RETRIES="${2:-}"
      shift 2
      ;;
    --max-reviewer-failures)
      MAX_REVIEWER_FAILURES="${2:-}"
      shift 2
      ;;
    --model-builder)
      MODEL_BUILDER="${2:-}"
      shift 2
      ;;
    --model-reviewer)
      MODEL_REVIEWER="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

if [[ -n "$TASK_TEXT" && -n "$TASK_FILE" ]]; then
  die "--task and --task-file are mutually exclusive"
fi
if [[ -z "$TASK_TEXT" && -z "$TASK_FILE" ]]; then
  die "either --task or --task-file is required"
fi

if [[ -n "$TASK_FILE" ]]; then
  [[ -f "$TASK_FILE" ]] || die "task file not found: $TASK_FILE"
  TASK_TEXT="$(cat "$TASK_FILE")"
fi
[[ -n "$TASK_TEXT" ]] || die "task content is empty"

for n in "$MAX_ITERATIONS" "$MAX_BUILDER_CLEANUP_RETRIES" "$MAX_REVIEWER_FAILURES"; do
  is_positive_int "$n" || die "numeric options must be positive integers"
done

for cmd in git gh codex jq; do
  require_cmd "$cmd"
done

WORKDIR="$(resolve_repo_root)"
cd "$WORKDIR"

if [[ -n "$(git ls-files -u)" ]]; then
  die "unmerged paths detected; resolve conflicts first"
fi
if has_tracked_dirty; then
  die "tracked working tree is dirty; commit/stash before running the loop"
fi

TARGET_BRANCH=""
if [[ -n "$PR_URL" ]]; then
  scripts/run_builder_reviewer_doctor.sh --pr-url "$PR_URL" >/dev/null

  pr_info_json="$(gh pr view "$PR_URL" --json url,state,mergedAt,headRefName,number 2>/dev/null || true)"
  [[ -n "$pr_info_json" ]] || die "failed to read PR metadata: $PR_URL"

  pr_state="$(jq -r '.state // ""' <<< "$pr_info_json")"
  pr_merged_at="$(jq -r '.mergedAt // empty' <<< "$pr_info_json")"
  TARGET_BRANCH="$(jq -r '.headRefName // ""' <<< "$pr_info_json")"
  PR_URL="$(jq -r '.url // ""' <<< "$pr_info_json")"

  [[ -n "$TARGET_BRANCH" ]] || die "failed to resolve PR head branch from: $PR_URL"
  if [[ "$pr_state" != "OPEN" || -n "$pr_merged_at" ]]; then
    die "--pr-url must reference an OPEN and unmerged PR (state=$pr_state, mergedAt=${pr_merged_at:-null})"
  fi

  ensure_branch_checked_out "$TARGET_BRANCH"
else
  TARGET_BRANCH="$(resolve_current_branch || true)"
  [[ -n "$TARGET_BRANCH" ]] || die "unable to resolve current branch"
  scripts/run_builder_reviewer_doctor.sh --head-branch "$TARGET_BRANCH" >/dev/null
fi

if [[ -n "$(git ls-files -u)" ]]; then
  die "unmerged paths detected after branch selection"
fi
if has_tracked_dirty; then
  die "tracked working tree became dirty after branch selection"
fi

declare -A BASELINE_UNTRACKED=()
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  BASELINE_UNTRACKED["$path"]=1
done < <(git ls-files --others --exclude-standard)

START_COMMIT="$(git rev-parse HEAD)"
LATEST_COMMIT="$START_COMMIT"

SELF_LOGIN="$(gh api user --jq '.login' 2>/dev/null || true)"
[[ -n "$SELF_LOGIN" ]] || die "failed to resolve authenticated GitHub login"

RUN_ID="loop-$(date -u +%Y%m%dT%H%M%SZ)-$$"
LOG_DIR="/tmp/builder_reviewer_loop_${RUN_ID}"
mkdir -p "$LOG_DIR"

log "loop started (branch=$TARGET_BRANCH, start_commit=$START_COMMIT, run_id=$RUN_ID)"

initial_builder_prompt="You are the BUILDER agent in an autonomous loop.

Task:
${TASK_TEXT}

Requirements:
- Follow repository policies in AGENTS.md and PLANS.md.
- Work only on branch: ${TARGET_BRANCH}
- When done, ensure all required changes are committed and pushed to origin/${TARGET_BRANCH}.
- Keep unrelated baseline untracked files untouched."

if ! run_codex_prompt "builder_initial" "$MODEL_BUILDER" "$initial_builder_prompt" "$LOG_DIR/builder_initial.log"; then
  die "initial builder execution failed"
fi

cleanup_result="$(run_builder_cleanup_until_stable "$START_COMMIT" || true)"
[[ -n "$cleanup_result" ]] || die "builder cleanup failed after initial task"

cleanup_kind="${cleanup_result%%|*}"
cleanup_commit="${cleanup_result#*|}"

if [[ "$cleanup_kind" == "unchanged" ]]; then
  log "no new commit detected after initial builder run; exiting without review loop"
  exit 0
fi

LATEST_COMMIT="$cleanup_commit"

if [[ -z "$PR_URL" ]]; then
  PR_URL="$(resolve_or_create_pr_for_branch "$TARGET_BRANCH")"
  [[ -n "$PR_URL" ]] || die "failed to resolve/create PR for branch: $TARGET_BRANCH"
fi

log "target PR: $PR_URL"

REVIEWER_FAILURES=0

for ((ITERATION=1; ITERATION<=MAX_ITERATIONS; ITERATION++)); do
  log "review iteration $ITERATION started for commit $LATEST_COMMIT"

  reviewer_prompt="You are the REVIEWER agent in an autonomous loop.

Review target:
- PR URL: ${PR_URL}
- Target commit and newer commits on head branch: ${LATEST_COMMIT}

Policy requirements:
- Follow REVIEW.md strictly.
- Post your review result as PR comment(s).
- In autonomous mode, include contract tags:
  AUTO_AGENT: REVIEWER
  AUTO_REVIEW_STATUS: APPROVED or CHANGES_REQUIRED
  AUTO_TARGET_COMMIT: ${LATEST_COMMIT}
- If and only if approved, include a standalone token line: APPROVE
- If CI is running, do not wait for completion; post based on current evidence.
- If the latest plan in this PR appears unresolved after three failures, explicitly include a remediation request in the review comment."

  if ! run_codex_prompt "reviewer" "$MODEL_REVIEWER" "$reviewer_prompt" "$LOG_DIR/reviewer_iter_${ITERATION}.log"; then
    REVIEWER_FAILURES=$((REVIEWER_FAILURES + 1))
    if [[ "$REVIEWER_FAILURES" -ge "$MAX_REVIEWER_FAILURES" ]]; then
      die "reviewer execution failed $REVIEWER_FAILURES times consecutively"
    fi
    continue
  fi

  commit_time="$(git show -s --format=%cI "$LATEST_COMMIT")"
  comments_json="$(fetch_self_comments_since "$PR_URL" "$SELF_LOGIN" "$commit_time" || true)"
  if [[ -z "$comments_json" ]]; then
    REVIEWER_FAILURES=$((REVIEWER_FAILURES + 1))
    if [[ "$REVIEWER_FAILURES" -ge "$MAX_REVIEWER_FAILURES" ]]; then
      die "failed to fetch reviewer comments $REVIEWER_FAILURES times consecutively"
    fi
    continue
  fi

  comment_count="$(jq -r 'length' <<< "$comments_json")"
  if [[ "$comment_count" -eq 0 ]]; then
    REVIEWER_FAILURES=$((REVIEWER_FAILURES + 1))
    if [[ "$REVIEWER_FAILURES" -ge "$MAX_REVIEWER_FAILURES" ]]; then
      die "no new self-authored review comments found after commit $LATEST_COMMIT for $REVIEWER_FAILURES consecutive attempts"
    fi
    continue
  fi

  REVIEWER_FAILURES=0

  APPROVED=0
  COMMENT_URLS=()
  for ((i=0; i<comment_count; i++)); do
    url="$(jq -r ".[$i].url // empty" <<< "$comments_json")"
    body="$(jq -r ".[$i].body // \"\"" <<< "$comments_json")"

    if [[ -n "$url" ]]; then
      COMMENT_URLS+=("$url")
    fi

    target_commit="$(extract_tag_value "$body" "AUTO_TARGET_COMMIT" || true)"
    if contains_approve_token "$body" && [[ "$target_commit" == "$LATEST_COMMIT" ]]; then
      APPROVED=1
    fi
  done

  if [[ "$APPROVED" -eq 1 ]]; then
    finalize_pr_tracking_doc "$PR_URL" "$TARGET_BRANCH" "$LATEST_COMMIT"
    log "approval detected for commit $LATEST_COMMIT; loop finished"
    exit 0
  fi

  if [[ ${#COMMENT_URLS[@]} -eq 0 ]]; then
    REVIEWER_FAILURES=$((REVIEWER_FAILURES + 1))
    if [[ "$REVIEWER_FAILURES" -ge "$MAX_REVIEWER_FAILURES" ]]; then
      die "reviewer comments had no URLs for $REVIEWER_FAILURES consecutive attempts"
    fi
    continue
  fi

  comment_links_block="$(printf '%s\n' "${COMMENT_URLS[@]}")"
  builder_followup_prompt="You are the BUILDER agent in an autonomous loop.

Address all review comments referenced by these links:
${comment_links_block}

Requirements:
- Implement required fixes on branch ${TARGET_BRANCH}.
- Commit all required changes.
- Push branch ${TARGET_BRANCH} to origin.
- Keep unrelated baseline untracked files untouched."

  if ! run_codex_prompt "builder_followup" "$MODEL_BUILDER" "$builder_followup_prompt" "$LOG_DIR/builder_followup_iter_${ITERATION}.log"; then
    die "builder follow-up execution failed at iteration $ITERATION"
  fi

  cleanup_result="$(run_builder_cleanup_until_stable "$LATEST_COMMIT" || true)"
  [[ -n "$cleanup_result" ]] || die "builder cleanup failed at iteration $ITERATION"

  cleanup_kind="${cleanup_result%%|*}"
  cleanup_commit="${cleanup_result#*|}"

  if [[ "$cleanup_kind" == "changed" ]]; then
    LATEST_COMMIT="$cleanup_commit"
    log "new commit detected after builder follow-up: $LATEST_COMMIT"
  else
    log "no new commit after builder follow-up; re-checking review comments"
  fi

done

die "max iterations reached without APPROVE for commit $LATEST_COMMIT"
