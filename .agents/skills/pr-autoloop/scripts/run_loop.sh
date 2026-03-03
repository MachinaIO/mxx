#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_loop.sh --goal-file <path> [--pr-url <url> | --head-branch <branch>] [options]
  run_loop.sh --self-test

Required arguments:
  --goal-file <path>              Goal/spec input for the builder agent.
  --pr-url <url>                  Target pull request URL (or use --head-branch bootstrap mode).

Options:
  --head-branch <name>            Head branch used when PR URL is not provided.
  --base-branch <name>            Base branch used when builder must create a PR (default: origin/HEAD or main).
  --max-builder-failures <n>      Consecutive builder failure threshold (default: 3).
  --max-iterations <n>            Maximum loop iterations (default: 20).
  --runtime-dir <path>            Runtime directory (default: .agents/skills/pr-autoloop/runtime).
  --builder-worktree <path>       Builder worktree path (default: <runtime>/worktrees/builder-<lock_key>).
  --reviewer-worktree <path>      Reviewer worktree path (default: <runtime>/worktrees/reviewer-<lock_key>).
  --cleanup-lock                  Remove an existing lock file before acquiring lock.
  --model <name>                  Optional model value passed to codex exec.
  --self-test                     Run parser/contract self-tests only.
USAGE
}

ts_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(ts_utc)] $*"
}

die() {
  echo "[$(ts_utc)] ERROR: $*" >&2
  exit 1
}

require_cmd() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "missing required command: $cmd"
}

is_positive_int() {
  [[ "$1" =~ ^[0-9]+$ ]] && [[ "$1" -gt 0 ]]
}

extract_pr_number() {
  local url="$1"
  if [[ "$url" =~ /pull/([0-9]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

sanitize_for_key() {
  local value="$1"
  echo "$value" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

resolve_current_branch() {
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [[ -z "$branch" || "$branch" == "HEAD" ]]; then
    return 1
  fi
  printf '%s\n' "$branch"
}

default_base_branch() {
  local remote_head
  remote_head="$(git symbolic-ref refs/remotes/origin/HEAD 2>/dev/null || true)"
  if [[ "$remote_head" == refs/remotes/origin/* ]]; then
    printf '%s\n' "${remote_head#refs/remotes/origin/}"
    return 0
  fi
  printf '%s\n' "main"
}

local_branch_exists() {
  local branch="$1"
  git show-ref --verify --quiet "refs/heads/$branch"
}

remote_branch_exists() {
  local branch="$1"
  if git show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    return 0
  fi
  git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1
}

discover_pr_url_by_head_branch() {
  local branch="$1"
  local list_json
  list_json="$(gh pr list --state open --head "$branch" --json url,headRefName 2>/dev/null || true)"
  [[ -n "$list_json" ]] || return 1

  jq -r \
    --arg branch "$branch" \
    '[.[] | select(.headRefName == $branch)][0].url // empty' <<< "$list_json"
}

load_pr_context_from_url() {
  local input_url="$1"
  local pr_json
  local resolved_url
  local resolved_number
  local resolved_branch
  local resolved_base

  pr_json="$(gh pr view "$input_url" --json url,number,headRefName,baseRefName 2>/dev/null || true)"
  [[ -n "$pr_json" ]] || return 1

  resolved_url="$(jq -r '.url // empty' <<< "$pr_json")"
  resolved_number="$(jq -r '.number // empty' <<< "$pr_json")"
  resolved_branch="$(jq -r '.headRefName // empty' <<< "$pr_json")"
  resolved_base="$(jq -r '.baseRefName // empty' <<< "$pr_json")"

  [[ -n "$resolved_url" && -n "$resolved_number" && -n "$resolved_branch" ]] || return 1

  PR_URL="$resolved_url"
  PR_NUMBER="$resolved_number"
  PR_BRANCH="$resolved_branch"
  if [[ -n "$resolved_base" ]]; then
    BASE_BRANCH="$resolved_base"
  fi
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

self_test() {
  local sample status target bad

  sample="[AUTO_LOOP]
AUTO_RUN_ID: run-x
AUTO_ITERATION: 3
AUTO_AGENT: REVIEWER
AUTO_REVIEW_STATUS: CHANGES_REQUIRED
AUTO_TARGET_COMMIT: 0123abcd"

  status="$(extract_tag_value "$sample" "AUTO_REVIEW_STATUS")"
  [[ "$status" == "CHANGES_REQUIRED" ]] || die "self-test failed: status parse"

  target="$(extract_tag_value "$sample" "AUTO_TARGET_COMMIT")"
  [[ "$target" == "0123abcd" ]] || die "self-test failed: target parse"

  bad="[AUTO_LOOP]
AUTO_AGENT: REVIEWER
AUTO_REVIEW_STATUS: MAYBE"
  status="$(extract_tag_value "$bad" "AUTO_REVIEW_STATUS")"
  if [[ "$status" == "APPROVED" || "$status" == "CHANGES_REQUIRED" ]]; then
    die "self-test failed: invalid status accepted"
  fi

  log "self-test: PASS"
}

GOAL_FILE=""
PR_URL=""
HEAD_BRANCH=""
BASE_BRANCH=""
PR_NUMBER=""
PR_BRANCH=""
MAX_BUILDER_FAILURES=3
MAX_ITERATIONS=20
RUNTIME_DIR=".agents/skills/pr-autoloop/runtime"
BUILDER_WORKTREE=""
REVIEWER_WORKTREE=""
CLEANUP_LOCK=0
SELF_TEST=0
MODEL=""
LOCK_KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --goal-file)
      GOAL_FILE="${2:-}"
      shift 2
      ;;
    --pr-url)
      PR_URL="${2:-}"
      shift 2
      ;;
    --head-branch)
      HEAD_BRANCH="${2:-}"
      shift 2
      ;;
    --base-branch)
      BASE_BRANCH="${2:-}"
      shift 2
      ;;
    --max-builder-failures)
      MAX_BUILDER_FAILURES="${2:-}"
      shift 2
      ;;
    --max-iterations)
      MAX_ITERATIONS="${2:-}"
      shift 2
      ;;
    --runtime-dir)
      RUNTIME_DIR="${2:-}"
      shift 2
      ;;
    --builder-worktree)
      BUILDER_WORKTREE="${2:-}"
      shift 2
      ;;
    --reviewer-worktree)
      REVIEWER_WORKTREE="${2:-}"
      shift 2
      ;;
    --cleanup-lock)
      CLEANUP_LOCK=1
      shift
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --self-test)
      SELF_TEST=1
      shift
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

if [[ "$SELF_TEST" -eq 1 ]]; then
  self_test
  exit 0
fi

[[ -n "$GOAL_FILE" ]] || die "--goal-file is required"
[[ -f "$GOAL_FILE" ]] || die "goal file not found: $GOAL_FILE"
is_positive_int "$MAX_BUILDER_FAILURES" || die "--max-builder-failures must be a positive integer"
is_positive_int "$MAX_ITERATIONS" || die "--max-iterations must be a positive integer"

for cmd in git gh codex jq; do
  require_cmd "$cmd"
done

if [[ -z "$PR_URL" && -z "$HEAD_BRANCH" ]]; then
  HEAD_BRANCH="$(resolve_current_branch || true)"
fi

if [[ -z "$PR_URL" && -z "$HEAD_BRANCH" ]]; then
  die "either --pr-url or --head-branch is required (or run from a named branch)"
fi

if [[ -n "$PR_URL" ]]; then
  load_pr_context_from_url "$PR_URL" || die "failed to resolve PR metadata from --pr-url"
  if [[ -n "$HEAD_BRANCH" && "$HEAD_BRANCH" != "$PR_BRANCH" ]]; then
    die "--head-branch ($HEAD_BRANCH) does not match PR head branch ($PR_BRANCH)"
  fi
else
  PR_BRANCH="$HEAD_BRANCH"
fi

[[ -n "$PR_BRANCH" ]] || die "failed to resolve head branch"
if ! local_branch_exists "$PR_BRANCH" && ! remote_branch_exists "$PR_BRANCH"; then
  die "head branch not found locally or on origin: $PR_BRANCH"
fi

if [[ -z "$PR_URL" ]]; then
  CANDIDATE_PR_URL="$(discover_pr_url_by_head_branch "$PR_BRANCH" || true)"
  if [[ -n "$CANDIDATE_PR_URL" ]]; then
    load_pr_context_from_url "$CANDIDATE_PR_URL" || die "failed to resolve discovered PR metadata"
    log "found existing PR for branch $PR_BRANCH: $PR_URL"
  fi
fi

if [[ -z "$BASE_BRANCH" ]]; then
  BASE_BRANCH="$(default_base_branch)"
fi

if [[ -n "$PR_NUMBER" ]]; then
  LOCK_KEY="pr-$PR_NUMBER"
else
  LOCK_KEY="branch-$(sanitize_for_key "$PR_BRANCH")"
fi

if [[ -z "$BUILDER_WORKTREE" ]]; then
  BUILDER_WORKTREE="$RUNTIME_DIR/worktrees/builder-$LOCK_KEY"
fi
if [[ -z "$REVIEWER_WORKTREE" ]]; then
  REVIEWER_WORKTREE="$RUNTIME_DIR/worktrees/reviewer-$LOCK_KEY"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$RUNTIME_DIR/runs/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
FEEDBACK_DIR="$RUN_DIR/feedback"
STATE_FILE="$RUN_DIR/state.json"
LOCK_DIR="$RUNTIME_DIR/locks"
LOCK_FILE="$LOCK_DIR/$LOCK_KEY.lock"

mkdir -p "$LOG_DIR" "$FEEDBACK_DIR" "$LOCK_DIR"

LOCK_HELD=0
cleanup() {
  if [[ "$LOCK_HELD" -eq 1 ]] && [[ -f "$LOCK_FILE" ]]; then
    rm -f "$LOCK_FILE"
  fi
}
trap cleanup EXIT INT TERM

if [[ "$CLEANUP_LOCK" -eq 1 ]] && [[ -f "$LOCK_FILE" ]]; then
  log "cleanup-lock enabled; removing existing lock: $LOCK_FILE"
  rm -f "$LOCK_FILE"
fi

if ( set -o noclobber; echo "$$" > "$LOCK_FILE" ) 2>/dev/null; then
  LOCK_HELD=1
else
  holder="$(cat "$LOCK_FILE" 2>/dev/null || echo unknown)"
  die "lock already exists for $LOCK_KEY at $LOCK_FILE (holder=$holder)"
fi

write_state() {
  local iteration="$1"
  local consecutive_failures="$2"
  local last_builder_commit="$3"
  local last_reviewer_status="$4"
  local status="$5"

  jq -n \
    --arg run_id "$RUN_ID" \
    --arg pr_url "$PR_URL" \
    --arg pr_number "$PR_NUMBER" \
    --arg pr_branch "$PR_BRANCH" \
    --arg base_branch "$BASE_BRANCH" \
    --arg lock_key "$LOCK_KEY" \
    --argjson iteration "$iteration" \
    --argjson max_iterations "$MAX_ITERATIONS" \
    --argjson max_builder_failures "$MAX_BUILDER_FAILURES" \
    --argjson consecutive_builder_failures "$consecutive_failures" \
    --arg last_builder_commit "$last_builder_commit" \
    --arg last_reviewer_status "$last_reviewer_status" \
    --arg status "$status" \
    --arg updated_at "$(ts_utc)" \
    '{
      run_id: $run_id,
      pr_url: $pr_url,
      pr_number: $pr_number,
      pr_branch: $pr_branch,
      base_branch: $base_branch,
      lock_key: $lock_key,
      iteration: $iteration,
      max_iterations: $max_iterations,
      max_builder_failures: $max_builder_failures,
      consecutive_builder_failures: $consecutive_builder_failures,
      last_builder_commit: $last_builder_commit,
      last_reviewer_status: $last_reviewer_status,
      status: $status,
      updated_at: $updated_at
    }' > "$STATE_FILE"
}

resolve_worktree_seed_ref() {
  if remote_branch_exists "$PR_BRANCH"; then
    printf '%s\n' "origin/$PR_BRANCH"
    return 0
  fi
  if local_branch_exists "$PR_BRANCH"; then
    printf '%s\n' "$PR_BRANCH"
    return 0
  fi
  return 1
}

sync_worktree() {
  local role="$1"
  local worktree_path="$2"
  local local_branch="pr-autoloop-${role}-${LOCK_KEY}"
  local seed_ref

  git fetch origin "$PR_BRANCH" >/dev/null 2>&1 || true
  seed_ref="$(resolve_worktree_seed_ref || true)"
  [[ -n "$seed_ref" ]] || die "unable to resolve seed ref for branch $PR_BRANCH"

  if [[ -e "$worktree_path/.git" || -f "$worktree_path/.git" ]]; then
    git -C "$worktree_path" fetch origin "$PR_BRANCH" >/dev/null 2>&1 || true

    if ! git -C "$worktree_path" diff --quiet -- || ! git -C "$worktree_path" diff --cached --quiet --; then
      die "$role worktree has uncommitted changes: $worktree_path"
    fi

    if ! git -C "$worktree_path" checkout "$local_branch" >/dev/null 2>&1; then
      git -C "$worktree_path" checkout -B "$local_branch" "$seed_ref" >/dev/null 2>&1 || die "failed to create local branch $local_branch"
    fi
    if remote_branch_exists "$PR_BRANCH"; then
      git -C "$worktree_path" merge --ff-only "origin/$PR_BRANCH" >/dev/null 2>&1 || die "failed to fast-forward $role worktree"
    fi
    return 0
  fi

  mkdir -p "$(dirname "$worktree_path")"
  if [[ -e "$worktree_path" ]] && [[ -n "$(ls -A "$worktree_path" 2>/dev/null || true)" ]]; then
    die "$role worktree path exists and is non-empty: $worktree_path"
  fi

  if git show-ref --verify --quiet "refs/heads/$local_branch"; then
    git worktree add "$worktree_path" "$local_branch" >/dev/null 2>&1 || die "failed to add existing branch worktree for $role"
    if remote_branch_exists "$PR_BRANCH"; then
      git -C "$worktree_path" merge --ff-only "origin/$PR_BRANCH" >/dev/null 2>&1 || die "failed to fast-forward existing $role branch"
    fi
  else
    git worktree add -b "$local_branch" "$worktree_path" "$seed_ref" >/dev/null 2>&1 || die "failed to add new branch worktree for $role"
    if remote_branch_exists "$PR_BRANCH"; then
      git -C "$worktree_path" merge --ff-only "origin/$PR_BRANCH" >/dev/null 2>&1 || die "failed to fast-forward new $role branch"
    fi
  fi
}

build_codex_cmd() {
  local worktree="$1"
  local prompt_file="$2"
  local cmd=(codex exec --dangerously-bypass-approvals-and-sandbox --cd "$worktree")
  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi
  cmd+=(-)
  printf '%q ' "${cmd[@]}"
  echo
  "${cmd[@]}" < "$prompt_file"
}

post_builder_comment() {
  local iteration="$1"
  local commit="$2"
  local body

  body="[AUTO_LOOP]
AUTO_RUN_ID: $RUN_ID
AUTO_ITERATION: $iteration
AUTO_AGENT: BUILDER
AUTO_TARGET_COMMIT: $commit
AUTO_RESULT: PUSHED"

  gh pr comment "$PR_URL" --body "$body" >/dev/null 2>&1 || die "failed to post builder comment"
}

get_latest_head_commit() {
  if [[ -n "$PR_URL" ]]; then
    gh pr view "$PR_URL" --json headRefOid --jq '.headRefOid // empty' 2>/dev/null || true
    return 0
  fi

  git ls-remote --heads origin "$PR_BRANCH" 2>/dev/null | awk 'NR==1 {print $1}'
}

ensure_pr_context_for_review() {
  local discovered_url

  if [[ -n "$PR_URL" ]]; then
    return 0
  fi

  discovered_url="$(discover_pr_url_by_head_branch "$PR_BRANCH" || true)"
  if [[ -z "$discovered_url" ]]; then
    return 1
  fi

  load_pr_context_from_url "$discovered_url" || return 1
  if [[ -z "$BASE_BRANCH" ]]; then
    BASE_BRANCH="$(default_base_branch)"
  fi
  log "detected PR created by builder: $PR_URL"
  return 0
}

fetch_reviewer_comment_body() {
  local iteration="$1"
  local comments_json

  comments_json="$(gh pr view "$PR_URL" --json comments 2>/dev/null || true)"
  [[ -n "$comments_json" ]] || return 1

  jq -r \
    --arg run "$RUN_ID" \
    --arg iter "$iteration" \
    '[.comments[]
      | select(.body | contains("AUTO_AGENT: REVIEWER"))
      | select(.body | contains("AUTO_RUN_ID: " + $run))
      | select(.body | contains("AUTO_ITERATION: " + $iter))][-1].body // empty' <<< "$comments_json"
}

LAST_BUILDER_COMMIT=""
LAST_REVIEW_STATUS=""
LAST_FEEDBACK_FILE=""
CONSECUTIVE_BUILDER_FAILURES=0

write_state 0 "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "RUNNING"

for ((ITER=1; ITER<=MAX_ITERATIONS; ITER++)); do
  log "iteration $ITER/$MAX_ITERATIONS started"

  sync_worktree "builder" "$BUILDER_WORKTREE"

  PRE_HEAD="$(get_latest_head_commit)"
  if [[ -n "$PR_URL" && -z "$PRE_HEAD" ]]; then
    die "failed to fetch PR head commit before builder run"
  fi

  BUILDER_PROMPT="$RUN_DIR/builder_prompt_iter_${ITER}.md"
  BUILDER_LOG="$LOG_DIR/builder_iter_${ITER}.log"

  {
    echo "You are the BUILDER agent in an autonomous PR loop."
    echo
    echo "Role contract:"
    echo "- Implement requested changes on the PR head branch."
    echo "- Follow repository policies (PLANS.md / AGENTS.md / DESIGN.md / ARCHITECTURE.md)."
    echo "- Do not ask human questions; resolve autonomously."
    echo "- Commit and push to the current PR branch before exiting."
    echo
    echo "Context:"
    echo "- PR head branch: $PR_BRANCH"
    if [[ -n "$PR_URL" ]]; then
      echo "- PR URL: $PR_URL"
    else
      echo "- PR URL: (none yet)"
      echo "- Base branch for new PR: $BASE_BRANCH"
      echo "- A PR for this branch does not exist yet. After pushing commits, create a PR from $PR_BRANCH to $BASE_BRANCH using gh CLI."
      echo "- If a PR already exists by the time you check, reuse it and do not create duplicates."
    fi
    echo "- Run ID: $RUN_ID"
    echo "- Iteration: $ITER"
    echo
    echo "Goal:"
    cat "$GOAL_FILE"
    if [[ -n "$LAST_FEEDBACK_FILE" && -f "$LAST_FEEDBACK_FILE" ]]; then
      echo
      echo "Reviewer feedback from previous iteration:"
      cat "$LAST_FEEDBACK_FILE"
    fi
  } > "$BUILDER_PROMPT"

  log "running builder codex exec (log: $BUILDER_LOG)"
  if ! build_codex_cmd "$BUILDER_WORKTREE" "$BUILDER_PROMPT" > "$BUILDER_LOG" 2>&1; then
    CONSECUTIVE_BUILDER_FAILURES=$((CONSECUTIVE_BUILDER_FAILURES + 1))
    write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED"
    log "builder run failed (consecutive=$CONSECUTIVE_BUILDER_FAILURES)"

    if [[ "$CONSECUTIVE_BUILDER_FAILURES" -ge "$MAX_BUILDER_FAILURES" ]]; then
      write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED_LIMIT"
      die "builder failed $CONSECUTIVE_BUILDER_FAILURES times consecutively"
    fi
    continue
  fi

  POST_HEAD="$(get_latest_head_commit)"
  if [[ -z "$POST_HEAD" ]]; then
    CONSECUTIVE_BUILDER_FAILURES=$((CONSECUTIVE_BUILDER_FAILURES + 1))
    write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED"
    log "builder run finished but no remote head was found for branch $PR_BRANCH (consecutive=$CONSECUTIVE_BUILDER_FAILURES)"

    if [[ "$CONSECUTIVE_BUILDER_FAILURES" -ge "$MAX_BUILDER_FAILURES" ]]; then
      write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED_LIMIT"
      die "builder did not produce a reachable remote branch head $CONSECUTIVE_BUILDER_FAILURES times consecutively"
    fi
    continue
  fi

  if [[ "$POST_HEAD" == "$PRE_HEAD" ]]; then
    CONSECUTIVE_BUILDER_FAILURES=$((CONSECUTIVE_BUILDER_FAILURES + 1))
    write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED"
    log "builder finished without pushing a new commit (consecutive=$CONSECUTIVE_BUILDER_FAILURES)"

    if [[ "$CONSECUTIVE_BUILDER_FAILURES" -ge "$MAX_BUILDER_FAILURES" ]]; then
      write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED_LIMIT"
      die "builder did not produce a new commit $CONSECUTIVE_BUILDER_FAILURES times consecutively"
    fi
    continue
  fi

  LAST_BUILDER_COMMIT="$POST_HEAD"
  CONSECUTIVE_BUILDER_FAILURES=0

  if ! ensure_pr_context_for_review; then
    CONSECUTIVE_BUILDER_FAILURES=$((CONSECUTIVE_BUILDER_FAILURES + 1))
    write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED"
    log "builder pushed commit but PR could not be discovered for branch $PR_BRANCH (consecutive=$CONSECUTIVE_BUILDER_FAILURES)"

    if [[ "$CONSECUTIVE_BUILDER_FAILURES" -ge "$MAX_BUILDER_FAILURES" ]]; then
      write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED_LIMIT"
      die "builder failed to produce/discover a PR $CONSECUTIVE_BUILDER_FAILURES times consecutively"
    fi
    continue
  fi

  post_builder_comment "$ITER" "$LAST_BUILDER_COMMIT"

  sync_worktree "reviewer" "$REVIEWER_WORKTREE"

  REVIEWER_PROMPT="$RUN_DIR/reviewer_prompt_iter_${ITER}.md"
  REVIEWER_LOG="$LOG_DIR/reviewer_iter_${ITER}.log"

  {
    echo "You are the REVIEWER agent in an autonomous PR loop."
    echo
    echo "Review requirements:"
    echo "- Follow REVIEW.md strictly for reviewer checks."
    echo "- Review the current PR head commit and post one PR comment in English."
    echo "- If CI checks are still running, do not wait; post the review comment for this iteration immediately."
    echo "- Include mandatory machine tags exactly once:"
    echo "  AUTO_AGENT: REVIEWER"
    echo "  AUTO_RUN_ID: $RUN_ID"
    echo "  AUTO_ITERATION: $ITER"
    echo "  AUTO_REVIEW_STATUS: APPROVED or CHANGES_REQUIRED"
    echo "  AUTO_TARGET_COMMIT: $LAST_BUILDER_COMMIT"
    echo "- Use CHANGES_REQUIRED when fixes are needed; otherwise APPROVED."
    echo
    echo "Context:"
    echo "- PR URL: $PR_URL"
    echo "- Target commit: $LAST_BUILDER_COMMIT"
  } > "$REVIEWER_PROMPT"

  log "running reviewer codex exec (log: $REVIEWER_LOG)"
  if ! build_codex_cmd "$REVIEWER_WORKTREE" "$REVIEWER_PROMPT" > "$REVIEWER_LOG" 2>&1; then
    write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED"
    die "reviewer run failed"
  fi

  REVIEW_BODY="$(fetch_reviewer_comment_body "$ITER" || true)"
  if [[ -z "$REVIEW_BODY" ]]; then
    write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED_CONTRACT"
    die "reviewer comment with required run/iteration tags not found"
  fi

  REVIEW_STATUS="$(extract_tag_value "$REVIEW_BODY" "AUTO_REVIEW_STATUS" || true)"
  TARGET_COMMIT="$(extract_tag_value "$REVIEW_BODY" "AUTO_TARGET_COMMIT" || true)"

  if [[ -z "$REVIEW_STATUS" || -z "$TARGET_COMMIT" ]]; then
    write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED_CONTRACT"
    die "reviewer comment missing required contract tags"
  fi

  if [[ "$REVIEW_STATUS" != "APPROVED" && "$REVIEW_STATUS" != "CHANGES_REQUIRED" ]]; then
    write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED_CONTRACT"
    die "reviewer status is invalid: $REVIEW_STATUS"
  fi

  if [[ "$TARGET_COMMIT" != "$LAST_BUILDER_COMMIT" ]]; then
    write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED_CONTRACT"
    die "reviewer target commit mismatch (expected $LAST_BUILDER_COMMIT, got $TARGET_COMMIT)"
  fi

  LAST_REVIEW_STATUS="$REVIEW_STATUS"

  if [[ "$REVIEW_STATUS" == "APPROVED" ]]; then
    write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "APPROVED"
    log "loop finished successfully at iteration $ITER"
    exit 0
  fi

  LAST_FEEDBACK_FILE="$FEEDBACK_DIR/reviewer_iter_${ITER}.md"
  printf '%s\n' "$REVIEW_BODY" > "$LAST_FEEDBACK_FILE"
  write_state "$ITER" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "RUNNING"

  log "reviewer requested changes; continuing to next iteration"
done

write_state "$MAX_ITERATIONS" "$CONSECUTIVE_BUILDER_FAILURES" "$LAST_BUILDER_COMMIT" "$LAST_REVIEW_STATUS" "FAILED_LIMIT"
die "max iterations reached without approval"
