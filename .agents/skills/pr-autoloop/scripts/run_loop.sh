#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_loop.sh --goal-file <path> --pr-url <url> [options]
  run_loop.sh --self-test

Required arguments:
  --goal-file <path>              Goal/spec input for the builder agent.
  --pr-url <url>                  Target pull request URL.

Options:
  --max-builder-failures <n>      Consecutive builder failure threshold (default: 3).
  --max-iterations <n>            Maximum loop iterations (default: 20).
  --runtime-dir <path>            Runtime directory (default: .agents/skills/pr-autoloop/runtime).
  --builder-worktree <path>       Builder worktree path (default: <runtime>/worktrees/builder-pr-<pr_number>).
  --reviewer-worktree <path>      Reviewer worktree path (default: <runtime>/worktrees/reviewer-pr-<pr_number>).
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
MAX_BUILDER_FAILURES=3
MAX_ITERATIONS=20
RUNTIME_DIR=".agents/skills/pr-autoloop/runtime"
BUILDER_WORKTREE=""
REVIEWER_WORKTREE=""
CLEANUP_LOCK=0
SELF_TEST=0
MODEL=""

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
[[ -n "$PR_URL" ]] || die "--pr-url is required"
[[ -f "$GOAL_FILE" ]] || die "goal file not found: $GOAL_FILE"
is_positive_int "$MAX_BUILDER_FAILURES" || die "--max-builder-failures must be a positive integer"
is_positive_int "$MAX_ITERATIONS" || die "--max-iterations must be a positive integer"

for cmd in git gh codex jq; do
  require_cmd "$cmd"
done

PR_NUMBER="$(extract_pr_number "$PR_URL" || true)"
if [[ -z "$PR_NUMBER" ]]; then
  PR_NUMBER="$(gh pr view "$PR_URL" --json number --jq '.number' 2>/dev/null || true)"
fi
[[ -n "$PR_NUMBER" ]] || die "failed to determine PR number from URL or gh"

PR_BRANCH="$(gh pr view "$PR_URL" --json headRefName --jq '.headRefName' 2>/dev/null || true)"
[[ -n "$PR_BRANCH" ]] || die "failed to fetch PR head branch via gh"

if [[ -z "$BUILDER_WORKTREE" ]]; then
  BUILDER_WORKTREE="$RUNTIME_DIR/worktrees/builder-pr-$PR_NUMBER"
fi
if [[ -z "$REVIEWER_WORKTREE" ]]; then
  REVIEWER_WORKTREE="$RUNTIME_DIR/worktrees/reviewer-pr-$PR_NUMBER"
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$RUNTIME_DIR/runs/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
FEEDBACK_DIR="$RUN_DIR/feedback"
STATE_FILE="$RUN_DIR/state.json"
LOCK_DIR="$RUNTIME_DIR/locks"
LOCK_FILE="$LOCK_DIR/pr-$PR_NUMBER.lock"

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
  die "lock already exists for PR #$PR_NUMBER at $LOCK_FILE (holder=$holder)"
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

sync_worktree() {
  local role="$1"
  local worktree_path="$2"
  local local_branch="pr-autoloop-${role}-${PR_NUMBER}"

  git fetch origin "$PR_BRANCH" >/dev/null 2>&1 || die "git fetch failed for branch $PR_BRANCH"

  if [[ -e "$worktree_path/.git" || -f "$worktree_path/.git" ]]; then
    git -C "$worktree_path" fetch origin "$PR_BRANCH" >/dev/null 2>&1 || die "failed to fetch in $role worktree"

    if ! git -C "$worktree_path" diff --quiet -- || ! git -C "$worktree_path" diff --cached --quiet --; then
      die "$role worktree has uncommitted changes: $worktree_path"
    fi

    if ! git -C "$worktree_path" checkout "$local_branch" >/dev/null 2>&1; then
      git -C "$worktree_path" checkout -B "$local_branch" "origin/$PR_BRANCH" >/dev/null 2>&1 || die "failed to create local branch $local_branch"
    fi
    git -C "$worktree_path" merge --ff-only "origin/$PR_BRANCH" >/dev/null 2>&1 || die "failed to fast-forward $role worktree"
    return 0
  fi

  mkdir -p "$(dirname "$worktree_path")"
  if [[ -e "$worktree_path" ]] && [[ -n "$(ls -A "$worktree_path" 2>/dev/null || true)" ]]; then
    die "$role worktree path exists and is non-empty: $worktree_path"
  fi

  if git show-ref --verify --quiet "refs/heads/$local_branch"; then
    git worktree add "$worktree_path" "$local_branch" >/dev/null 2>&1 || die "failed to add existing branch worktree for $role"
    git -C "$worktree_path" merge --ff-only "origin/$PR_BRANCH" >/dev/null 2>&1 || die "failed to fast-forward existing $role branch"
  else
    git worktree add -b "$local_branch" "$worktree_path" "origin/$PR_BRANCH" >/dev/null 2>&1 || die "failed to add new branch worktree for $role"
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
  gh pr view "$PR_URL" --json headRefOid --jq '.headRefOid' 2>/dev/null || true
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
  [[ -n "$PRE_HEAD" ]] || die "failed to fetch PR head commit before builder run"

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
    echo "- PR URL: $PR_URL"
    echo "- PR head branch: $PR_BRANCH"
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
  [[ -n "$POST_HEAD" ]] || die "failed to fetch PR head commit after builder run"

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
