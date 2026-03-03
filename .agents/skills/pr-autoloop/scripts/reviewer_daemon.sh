#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  reviewer_daemon.sh --start [--pr-url <url>] [--head-branch <branch>] [options]
  reviewer_daemon.sh --request --commit <sha> [--pr-url <url>] [--head-branch <branch>] [options]
  reviewer_daemon.sh --status [options]
  reviewer_daemon.sh --stop [options]

Modes:
  --start                         Start reviewer daemon when not running.
  --request                       Send one review request and wait for response.
  --status                        Print reviewer daemon status.
  --stop                          Stop reviewer daemon process.

Required with --request:
  --commit <sha>                  Commit hash to review.

Optional:
  --pr-url <url>                  Target PR URL.
  --head-branch <branch>          Target head branch when PR URL is unknown.
  --request-id <id>               Explicit request id.
  --run-id <id>                   Run id tag in reviewer comment (default: execplan-post-completion).
  --iteration <n>                 Iteration tag in reviewer comment (default: 0).
  --runtime-dir <path>            Runtime root (default: .agents/skills/pr-autoloop/runtime).
  --workdir <path>                Working directory for codex exec (default: repository root).
  --model <name>                  Optional model passed to codex exec.
  --wait-timeout-sec <n>          Wait timeout for --request (0 = wait forever, default: 0).
  --help                          Show this help.
USAGE
}

ts_utc() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  echo "[$(ts_utc)] $*" >&2
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

discover_pr_url_by_head_branch() {
  local branch="$1"
  local list_json
  list_json="$(gh pr list --state open --head "$branch" --json url,headRefName 2>/dev/null || true)"
  [[ -n "$list_json" ]] || return 1

  jq -r \
    --arg branch "$branch" \
    '[.[] | select(.headRefName == $branch)][0].url // empty' <<< "$list_json"
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

contains_approve_token() {
  local body="$1"
  grep -Eq '(^|[[:space:]])APPROVE($|[[:space:]])' <<< "$body"
}

MODE=""
PR_URL=""
HEAD_BRANCH=""
TARGET_COMMIT=""
REQUEST_ID=""
RUN_ID="execplan-post-completion"
ITERATION="0"
RUNTIME_DIR=".agents/skills/pr-autoloop/runtime"
WORKDIR=""
MODEL=""
WAIT_TIMEOUT_SEC="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --start)
      MODE="start"
      shift
      ;;
    --serve)
      MODE="serve"
      shift
      ;;
    --request)
      MODE="request"
      shift
      ;;
    --status)
      MODE="status"
      shift
      ;;
    --stop)
      MODE="stop"
      shift
      ;;
    --pr-url)
      PR_URL="${2:-}"
      shift 2
      ;;
    --head-branch)
      HEAD_BRANCH="${2:-}"
      shift 2
      ;;
    --commit)
      TARGET_COMMIT="${2:-}"
      shift 2
      ;;
    --request-id)
      REQUEST_ID="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --iteration)
      ITERATION="${2:-}"
      shift 2
      ;;
    --runtime-dir)
      RUNTIME_DIR="${2:-}"
      shift 2
      ;;
    --workdir)
      WORKDIR="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --wait-timeout-sec)
      WAIT_TIMEOUT_SEC="${2:-}"
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

[[ -n "$MODE" ]] || die "mode is required (--start|--request|--status|--stop)"
[[ -n "$RUN_ID" ]] || die "--run-id cannot be empty"
[[ -n "$ITERATION" ]] || die "--iteration cannot be empty"

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(resolve_repo_root)"
fi

DAEMON_DIR="$RUNTIME_DIR/reviewer-daemon"
INBOX_DIR="$DAEMON_DIR/inbox"
RESPONSES_DIR="$DAEMON_DIR/responses"
REQUESTS_DIR="$DAEMON_DIR/requests"
LOGS_DIR="$DAEMON_DIR/logs"
PID_FILE="$DAEMON_DIR/reviewer.pid"
STATE_FILE="$DAEMON_DIR/state.json"
DAEMON_LOG="$LOGS_DIR/daemon.log"
POLL_SECONDS=2

mkdir -p "$INBOX_DIR" "$RESPONSES_DIR" "$REQUESTS_DIR" "$LOGS_DIR"

is_daemon_running() {
  local pid
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" >/dev/null 2>&1
}

load_state_field() {
  local field="$1"
  if [[ ! -f "$STATE_FILE" ]]; then
    return 0
  fi
  jq -r --arg field "$field" '.[$field] // empty' "$STATE_FILE" 2>/dev/null || true
}

write_state() {
  local pr_url="$1"
  local head_branch="$2"
  local status="$3"

  jq -n \
    --arg pr_url "$pr_url" \
    --arg head_branch "$head_branch" \
    --arg status "$status" \
    --arg pid "$$" \
    --arg updated_at "$(ts_utc)" \
    --arg workdir "$WORKDIR" \
    '{
      pr_url: $pr_url,
      head_branch: $head_branch,
      status: $status,
      pid: $pid,
      workdir: $workdir,
      updated_at: $updated_at
    }' > "$STATE_FILE"
}

write_response() {
  local request_id="$1"
  local success="$2"
  local error_message="$3"
  local comment_url="$4"
  local review_status="$5"
  local approved_token_found="$6"
  local target_commit="$7"

  local response_file="$RESPONSES_DIR/$request_id.json"
  local tmp_file="$response_file.tmp.$$"

  jq -n \
    --arg request_id "$request_id" \
    --argjson success "$success" \
    --arg error "$error_message" \
    --arg comment_url "$comment_url" \
    --arg review_status "$review_status" \
    --argjson approved_token_found "$approved_token_found" \
    --arg target_commit "$target_commit" \
    --arg responded_at "$(ts_utc)" \
    '{
      request_id: $request_id,
      success: $success,
      error: $error,
      comment_url: $comment_url,
      review_status: $review_status,
      approved_token_found: $approved_token_found,
      target_commit: $target_commit,
      responded_at: $responded_at
    }' > "$tmp_file"

  mv "$tmp_file" "$response_file"
}

fetch_comment_for_request() {
  local pr_url="$1"
  local request_id="$2"
  local comments_json

  comments_json="$(gh pr view "$pr_url" --json comments 2>/dev/null || true)"
  [[ -n "$comments_json" ]] || return 1

  jq -c \
    --arg request_id "$request_id" \
    '[.comments[]
      | select(.body | contains("AUTO_AGENT: REVIEWER"))
      | select(.body | contains("AUTO_REQUEST_ID: " + $request_id))][-1] // empty' <<< "$comments_json"
}

build_codex_cmd() {
  local cmd=(codex exec --dangerously-bypass-approvals-and-sandbox --cd "$WORKDIR")
  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi
  cmd+=(-)
  printf '%q ' "${cmd[@]}" >&2
  echo >&2
  "${cmd[@]}"
}

process_request() {
  local request_path="$1"
  local request_json
  local request_id req_pr_url req_head_branch req_commit req_run_id req_iteration
  local effective_pr_url effective_head_branch
  local prompt_file reviewer_log
  local comment_json comment_url comment_body review_status target_commit
  local approved_token_found

  request_json="$(cat "$request_path")"
  request_id="$(jq -r '.request_id // empty' <<< "$request_json")"
  req_pr_url="$(jq -r '.pr_url // empty' <<< "$request_json")"
  req_head_branch="$(jq -r '.head_branch // empty' <<< "$request_json")"
  req_commit="$(jq -r '.target_commit // empty' <<< "$request_json")"
  req_run_id="$(jq -r '.run_id // empty' <<< "$request_json")"
  req_iteration="$(jq -r '.iteration // "0"' <<< "$request_json")"

  if [[ -z "$request_id" || -z "$req_commit" ]]; then
    [[ -n "$request_id" ]] || request_id="invalid-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    write_response "$request_id" false "invalid request payload (request_id or target_commit missing)" "" "" false "$req_commit"
    rm -f "$request_path"
    return 0
  fi

  effective_pr_url="$req_pr_url"
  effective_head_branch="$req_head_branch"

  if [[ -z "$effective_pr_url" ]]; then
    effective_pr_url="$(load_state_field pr_url)"
  fi
  if [[ -z "$effective_head_branch" ]]; then
    effective_head_branch="$(load_state_field head_branch)"
  fi

  if [[ -z "$effective_pr_url" && -n "$effective_head_branch" ]]; then
    effective_pr_url="$(discover_pr_url_by_head_branch "$effective_head_branch" || true)"
  fi

  if [[ -z "$effective_pr_url" ]]; then
    write_response "$request_id" false "unable to resolve PR URL for review request" "" "" false "$req_commit"
    rm -f "$request_path"
    return 0
  fi

  write_state "$effective_pr_url" "$effective_head_branch" "RUNNING"

  prompt_file="$REQUESTS_DIR/reviewer_prompt_${request_id}.md"
  reviewer_log="$LOGS_DIR/reviewer_request_${request_id}.log"

  {
    echo "You are the REVIEWER agent process in an autonomous ExecPlan workflow."
    echo
    echo "Review policy:"
    echo "- Follow REVIEW.md strictly."
    echo "- Review target commit and post exactly one PR comment in English."
    echo "- If CI checks are running, do not wait; post immediately."
    echo
    echo "Mandatory comment tags (include exactly once):"
    echo "AUTO_AGENT: REVIEWER"
    echo "AUTO_REQUEST_ID: $request_id"
    echo "AUTO_RUN_ID: $req_run_id"
    echo "AUTO_ITERATION: $req_iteration"
    echo "AUTO_REVIEW_STATUS: APPROVED or CHANGES_REQUIRED"
    echo "AUTO_TARGET_COMMIT: $req_commit"
    echo
    echo "Approval token rule:"
    echo "- If and only if review is approved, include one separate line with exactly: APPROVE"
    echo "- If changes are required, do not include APPROVE."
    echo
    echo "Context:"
    echo "- PR URL: $effective_pr_url"
    echo "- Target commit: $req_commit"
  } > "$prompt_file"

  if ! build_codex_cmd < "$prompt_file" > "$reviewer_log" 2>&1; then
    write_response "$request_id" false "reviewer codex execution failed" "" "" false "$req_commit"
    rm -f "$request_path"
    return 0
  fi

  comment_json="$(fetch_comment_for_request "$effective_pr_url" "$request_id" || true)"
  if [[ -z "$comment_json" ]]; then
    write_response "$request_id" false "reviewer comment not found for AUTO_REQUEST_ID" "" "" false "$req_commit"
    rm -f "$request_path"
    return 0
  fi

  comment_url="$(jq -r '.url // empty' <<< "$comment_json")"
  comment_body="$(jq -r '.body // empty' <<< "$comment_json")"
  review_status="$(extract_tag_value "$comment_body" "AUTO_REVIEW_STATUS" || true)"
  target_commit="$(extract_tag_value "$comment_body" "AUTO_TARGET_COMMIT" || true)"

  if [[ "$target_commit" != "$req_commit" ]]; then
    write_response "$request_id" false "reviewer comment target commit mismatch" "$comment_url" "$review_status" false "$target_commit"
    rm -f "$request_path"
    return 0
  fi

  approved_token_found=false
  if contains_approve_token "$comment_body"; then
    approved_token_found=true
  fi

  write_response "$request_id" true "" "$comment_url" "$review_status" "$approved_token_found" "$target_commit"
  rm -f "$request_path"

  if [[ "$approved_token_found" == "true" ]]; then
    write_state "$effective_pr_url" "$effective_head_branch" "APPROVED"
    return 10
  fi

  write_state "$effective_pr_url" "$effective_head_branch" "WAITING"
  return 0
}

start_mode() {
  for cmd in git gh codex jq; do
    require_cmd "$cmd"
  done

  if [[ -z "$HEAD_BRANCH" && -z "$PR_URL" ]]; then
    HEAD_BRANCH="$(resolve_current_branch || true)"
  fi

  if is_daemon_running; then
    local pid
    pid="$(cat "$PID_FILE")"
    write_state "${PR_URL:-$(load_state_field pr_url)}" "${HEAD_BRANCH:-$(load_state_field head_branch)}" "WAITING"
    echo "STATUS=already_running"
    echo "PID=$pid"
    return 0
  fi

  rm -f "$PID_FILE"

  local cmd=("$0" --serve --runtime-dir "$RUNTIME_DIR" --workdir "$WORKDIR")
  if [[ -n "$PR_URL" ]]; then
    cmd+=(--pr-url "$PR_URL")
  fi
  if [[ -n "$HEAD_BRANCH" ]]; then
    cmd+=(--head-branch "$HEAD_BRANCH")
  fi
  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi

  nohup "${cmd[@]}" >> "$DAEMON_LOG" 2>&1 &
  sleep 1

  if ! is_daemon_running; then
    die "failed to start reviewer daemon"
  fi

  echo "STATUS=started"
  echo "PID=$(cat "$PID_FILE")"
}

serve_mode() {
  for cmd in git gh codex jq; do
    require_cmd "$cmd"
  done

  if [[ -z "$HEAD_BRANCH" && -z "$PR_URL" ]]; then
    HEAD_BRANCH="$(resolve_current_branch || true)"
  fi

  echo "$$" > "$PID_FILE"
  trap 'rm -f "$PID_FILE"' EXIT INT TERM

  write_state "$PR_URL" "$HEAD_BRANCH" "WAITING"
  log "reviewer daemon started (pid=$$, pr_url=${PR_URL:-none}, head_branch=${HEAD_BRANCH:-none})"

  while true; do
    local found=0
    while IFS= read -r request_path; do
      found=1
      set +e
      process_request "$request_path"
      rc=$?
      set -e

      if [[ "$rc" -eq 10 ]]; then
        log "approval token detected; reviewer daemon exiting"
        exit 0
      fi
      if [[ "$rc" -ne 0 ]]; then
        log "request processing failed unexpectedly (rc=$rc)"
      fi
    done < <(find "$INBOX_DIR" -maxdepth 1 -type f -name '*.json' | sort)

    if [[ "$found" -eq 0 ]]; then
      sleep "$POLL_SECONDS"
    fi
  done
}

request_mode() {
  for cmd in jq; do
    require_cmd "$cmd"
  done

  [[ -n "$TARGET_COMMIT" ]] || die "--commit is required with --request"

  if [[ -n "$WAIT_TIMEOUT_SEC" && "$WAIT_TIMEOUT_SEC" != "0" ]] && ! is_positive_int "$WAIT_TIMEOUT_SEC"; then
    die "--wait-timeout-sec must be 0 or a positive integer"
  fi

  if [[ -z "$REQUEST_ID" ]]; then
    REQUEST_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  fi

  if [[ -z "$PR_URL" ]]; then
    PR_URL="$(load_state_field pr_url)"
  fi
  if [[ -z "$HEAD_BRANCH" ]]; then
    HEAD_BRANCH="$(load_state_field head_branch)"
  fi

  is_daemon_running || die "reviewer daemon is not running"

  local request_file="$INBOX_DIR/$REQUEST_ID.json"
  local tmp_request="$request_file.tmp.$$"
  local response_file="$RESPONSES_DIR/$REQUEST_ID.json"
  local started_at now

  rm -f "$response_file"

  jq -n \
    --arg request_id "$REQUEST_ID" \
    --arg pr_url "$PR_URL" \
    --arg head_branch "$HEAD_BRANCH" \
    --arg target_commit "$TARGET_COMMIT" \
    --arg run_id "$RUN_ID" \
    --arg iteration "$ITERATION" \
    --arg requested_at "$(ts_utc)" \
    '{
      request_id: $request_id,
      pr_url: $pr_url,
      head_branch: $head_branch,
      target_commit: $target_commit,
      run_id: $run_id,
      iteration: $iteration,
      requested_at: $requested_at
    }' > "$tmp_request"
  mv "$tmp_request" "$request_file"

  started_at="$(date +%s)"

  while true; do
    if [[ -f "$response_file" ]]; then
      cat "$response_file"
      return 0
    fi

    if ! is_daemon_running; then
      die "reviewer daemon stopped while waiting for request response"
    fi

    if [[ "$WAIT_TIMEOUT_SEC" != "0" ]]; then
      now="$(date +%s)"
      if (( now - started_at >= WAIT_TIMEOUT_SEC )); then
        die "timed out waiting for reviewer response (request_id=$REQUEST_ID)"
      fi
    fi

    sleep "$POLL_SECONDS"
  done
}

status_mode() {
  if is_daemon_running; then
    echo "STATUS=running"
    echo "PID=$(cat "$PID_FILE")"
    if [[ -f "$STATE_FILE" ]]; then
      echo "STATE_FILE=$STATE_FILE"
    fi
  else
    echo "STATUS=stopped"
  fi
}

stop_mode() {
  if ! is_daemon_running; then
    echo "STATUS=stopped"
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid" >/dev/null 2>&1 || true
  sleep 1
  if is_daemon_running; then
    die "failed to stop reviewer daemon (pid=$pid)"
  fi
  rm -f "$PID_FILE"
  echo "STATUS=stopped"
}

case "$MODE" in
  start)
    start_mode
    ;;
  serve)
    serve_mode
    ;;
  request)
    request_mode
    ;;
  status)
    status_mode
    ;;
  stop)
    stop_mode
    ;;
  *)
    die "unsupported mode: $MODE"
    ;;
esac
