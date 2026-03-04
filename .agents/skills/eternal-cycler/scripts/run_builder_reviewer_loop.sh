#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  run_builder_reviewer_loop.sh [--task <text> | --task-file <path>] [options]

Options:
  --task <text>                      Builder initial task text.
  --task-file <path>                 Builder initial task file path.
  --pr-url <url>                     Optional target PR URL.
                                    If provided, its head branch must match the current local branch.
                                    If omitted, the current branch is used.
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

normalize_pr_url() {
  local pr_url="$1"
  printf '%s\n' "$pr_url" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s:/+$::'
}

parse_reviewer_payload_json() {
  local raw="$1"
  jq -e -c '
    if type != "object" then
      error("reviewer output must be a JSON object")
    else
      .
    end
    | if (.pr_url | type) != "string" then error("pr_url must be string") else . end
    | if (.comment_body | type) != "string" then error("comment_body must be string") else . end
    | .pr_url |= (sub("^[[:space:]]+"; "") | sub("[[:space:]]+$"; ""))
    | if (.pr_url | length) == 0 then error("pr_url must be non-empty") else . end
    | if (.comment_body | length) == 0 then error("comment_body must be non-empty") else . end
    | if (.approve_merge | type) != "boolean" then error("approve_merge must be boolean") else . end
    | {pr_url, comment_body, approve_merge}
  ' <<< "$raw" 2>/dev/null
}

parse_builder_payload_json() {
  local raw="$1"
  jq -e -c '
    if type != "object" then
      error("builder output must be a JSON object")
    else
      .
    end
    | if (.plan_doc_filename | type) != "string" then error("plan_doc_filename must be string") else . end
    | if (.result | type) != "string" then error("result must be string") else . end
    | if (.failure_reason | type) != "string" then error("failure_reason must be string") else . end
    | if (.pr_title | type) != "string" then error("pr_title must be string") else . end
    | if (.pr_body | type) != "string" then error("pr_body must be string") else . end
    | .plan_doc_filename |= (sub("^[[:space:]]+"; "") | sub("[[:space:]]+$"; ""))
    | .failure_reason |= (sub("^[[:space:]]+"; "") | sub("[[:space:]]+$"; ""))
    | .pr_title |= (sub("^[[:space:]]+"; "") | sub("[[:space:]]+$"; ""))
    | if (.plan_doc_filename | length) == 0 then error("plan_doc_filename must be non-empty") else . end
    | if (.pr_title | length) == 0 then error("pr_title must be non-empty") else . end
    | if (.result != "success" and .result != "failed_after_3_retries") then
        error("result must be success or failed_after_3_retries")
      else
        .
      end
    | if (.result == "success" and (.failure_reason | length) != 0) then
        error("failure_reason must be empty string when result=success")
      elif (.result == "failed_after_3_retries" and (.failure_reason | length) == 0) then
        error("failure_reason must be non-empty when result=failed_after_3_retries")
      else
        .
      end
    | {plan_doc_filename, result, failure_reason, pr_title, pr_body}
  ' <<< "$raw" 2>/dev/null
}

LAST_CODEX_OUTPUT=""

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
  local local_head remote_head remote_output rc
  local_head="$(git rev-parse HEAD)"

  set +e
  remote_output="$(git ls-remote --heads origin "$TARGET_BRANCH" 2>/dev/null)"
  rc=$?
  set -e

  if [[ "$rc" -ne 0 ]]; then
    log "failed to query origin/$TARGET_BRANCH via git ls-remote; treating branch as not pushed yet"
    return 1
  fi

  remote_head="$(awk '{print $1}' <<< "$remote_output" | head -n1)"
  [[ -n "$remote_head" && "$remote_head" == "$local_head" ]]
}

run_codex_prompt() {
  local role="$1"
  local model="$2"
  local prompt_text="$3"

  local cmd=(codex exec --dangerously-bypass-approvals-and-sandbox --cd "$WORKDIR")
  if [[ -n "$model" ]]; then
    cmd+=(--model "$model")
  fi
  cmd+=(-)

  if ! printf '%s\n' "$prompt_text" | "${cmd[@]}"; then
    log "$role codex execution failed"
    return 1
  fi

  return 0
}

write_reviewer_output_schema() {
  local schema_file
  schema_file="$(mktemp)"
  cat > "$schema_file" <<'EOF'
{
  "type": "object",
  "additionalProperties": false,
  "required": ["pr_url", "comment_body", "approve_merge"],
  "properties": {
    "pr_url": {
      "type": "string",
      "minLength": 1
    },
    "comment_body": {
      "type": "string",
      "minLength": 1
    },
    "approve_merge": {
      "type": "boolean"
    }
  }
}
EOF
  printf '%s\n' "$schema_file"
}

write_builder_output_schema() {
  local schema_file
  schema_file="$(mktemp)"
  cat > "$schema_file" <<'EOF'
{
  "type": "object",
  "additionalProperties": false,
  "required": ["plan_doc_filename", "result", "failure_reason", "pr_title", "pr_body"],
  "properties": {
    "plan_doc_filename": {
      "type": "string",
      "minLength": 1
    },
    "result": {
      "type": "string",
      "enum": ["success", "failed_after_3_retries"]
    },
    "failure_reason": {
      "type": "string"
    },
    "pr_title": {
      "type": "string",
      "minLength": 1
    },
    "pr_body": {
      "type": "string"
    }
  }
}
EOF
  printf '%s\n' "$schema_file"
}

run_codex_prompt_capture() {
  local role="$1"
  local model="$2"
  local prompt_text="$3"
  local schema_file="${4:-}"
  local rc last_message_file

  local cmd=(codex exec --dangerously-bypass-approvals-and-sandbox --cd "$WORKDIR")
  if [[ -n "$model" ]]; then
    cmd+=(--model "$model")
  fi
  if [[ -n "$schema_file" ]]; then
    last_message_file="$(mktemp)"
    cmd+=(--output-schema "$schema_file" --output-last-message "$last_message_file")
  fi
  cmd+=(-)

  # Stream codex output directly to the terminal so the operator can follow
  # builder/reviewer progress in real time.  The structured JSON payload is
  # captured separately via --output-last-message into $last_message_file.
  set +e
  printf '%s\n' "$prompt_text" | "${cmd[@]}"
  rc=$?
  set -e

  if [[ -n "$schema_file" ]]; then
    LAST_CODEX_OUTPUT="$(cat "$last_message_file" 2>/dev/null || true)"
    rm -f "$last_message_file"
  fi

  if [[ "$rc" -ne 0 ]]; then
    log "$role codex execution failed"
    return 1
  fi

  if [[ -n "$schema_file" && -z "$(printf '%s' "$LAST_CODEX_OUTPUT" | tr -d '[:space:]')" ]]; then
    log "$role codex execution produced empty structured output"
    return 1
  fi

  return 0
}

post_pr_comment() {
  local pr_url="$1"
  local comment_body="$2"
  local output rc

  set +e
  output="$(gh pr comment "$pr_url" --body "$comment_body" 2>&1)"
  rc=$?
  set -e

  if [[ "$rc" -ne 0 ]]; then
    log "failed to post PR comment for $pr_url: $output"
    return 1
  fi

  printf '%s\n' "$output"
  return 0
}

post_builder_failure_comment_after_push() {
  local stage="$1"
  local plan_doc_filename="$2"
  local failure_reason="$3"
  local normalized_target_pr_url comment_body

  auto_stage_commit_and_push "loop: checkpoint builder failure report"
  push_target_branch

  if [[ -z "$PR_URL" ]]; then
    PR_URL="$(resolve_or_create_pr_for_branch "$TARGET_BRANCH")"
    [[ -n "$PR_URL" ]] || die "failed to resolve/create PR for branch: $TARGET_BRANCH"
  fi

  normalized_target_pr_url="$(normalize_pr_url "$PR_URL")"
  comment_body="AUTO_AGENT: BUILDER
Builder agent failure report.
stage: ${stage}
plan_doc_filename: ${plan_doc_filename}
result: failed_after_3_retries
failure_reason: ${failure_reason}"

  post_pr_comment "$normalized_target_pr_url" "$comment_body" >/dev/null || die "failed to post builder failure comment to $normalized_target_pr_url"
}

handle_builder_payload_result() {
  local stage="$1"
  local payload_json="$2"
  local result plan_doc_filename failure_reason

  result="$(jq -r '.result' <<< "$payload_json")"
  plan_doc_filename="$(jq -r '.plan_doc_filename' <<< "$payload_json")"
  failure_reason="$(jq -r '.failure_reason' <<< "$payload_json")"

  if [[ "$result" == "success" ]]; then
    return 0
  fi

  post_builder_failure_comment_after_push "$stage" "$plan_doc_filename" "$failure_reason"
  die "builder reported failed_after_3_retries at stage=${stage}; reason=${failure_reason}; plan_doc_filename=${plan_doc_filename}"
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

push_target_branch() {
  git push origin "$TARGET_BRANCH" >/dev/null 2>&1 || die "failed to push branch to origin/$TARGET_BRANCH"
}

auto_stage_commit_and_push() {
  local commit_message="$1"

  git add -u -- . >/dev/null 2>&1 || die "failed to stage tracked changes"

  mapfile -t NEW_UNTRACKED < <(get_new_untracked_paths)
  if [[ ${#NEW_UNTRACKED[@]} -gt 0 ]]; then
    for path in "${NEW_UNTRACKED[@]}"; do
      git add -- "$path" >/dev/null 2>&1 || die "failed to stage new untracked path: $path"
    done
  fi

  if ! git diff --cached --quiet; then
    git commit -m "$commit_message" >/dev/null 2>&1 || die "failed to create loop commit: $commit_message"
  fi

  if ! head_is_pushed; then
    push_target_branch
  fi
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

  for dir in eternal-cycler-out/prs/active eternal-cycler-out/prs/completed; do
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

  default_active_path="eternal-cycler-out/prs/active/pr_${target_branch//\//_}.md"
  default_completed_path="eternal-cycler-out/prs/completed/pr_${target_branch//\//_}.md"

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

  if [[ "$source_path" == eternal-cycler-out/prs/completed/* ]]; then
    completed_path="$source_path"
  elif [[ "$source_path" == eternal-cycler-out/prs/active/* ]]; then
    completed_path="eternal-cycler-out/prs/completed/$(basename "$source_path")"
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
  local pr_title="${2:-}"
  local pr_body="${3:-}"
  local open_json open_url create_out

  [[ -n "$pr_title" ]] || pr_title="Auto: ${branch}"

  open_json="$(gh pr list --state open --head "$branch" --json url,updatedAt --limit 20 2>/dev/null || true)"
  open_url="$(jq -r '[.[]] | sort_by(.updatedAt) | reverse | .[0].url // empty' <<< "$open_json")"
  if [[ -n "$open_url" ]]; then
    printf '%s\n' "$open_url"
    return 0
  fi

  set +e
  create_out="$(gh pr create --title "$pr_title" --body "$pr_body" --head "$branch" 2>&1)"
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

run_builder_cleanup_until_stable() {
  local base_commit="$1"
  local attempt=0
  local current_commit tracked_dirty pushed
  local cleanup_prompt
  local untracked_msg

  while true; do
    auto_stage_commit_and_push "loop: checkpoint builder output"

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

    cleanup_prompt="You are called by a parent agent as a builder agent.
You are the BUILDER agent in an autonomous loop.

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
2) Do not modify unrelated baseline untracked files that existed before this loop started.
3) Do not run git commit or git push. The loop script performs commit/push automatically.

After finishing, exit."

    if ! run_codex_prompt "builder_cleanup" "$MODEL_BUILDER" "$cleanup_prompt"; then
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

if [[ -n "$TASK_FILE" ]]; then
  [[ -f "$TASK_FILE" ]] || die "task file not found: $TASK_FILE"
  TASK_TEXT="$(cat "$TASK_FILE")"
fi

if [[ -z "$TASK_TEXT" ]]; then
  echo "either --task or --task-file is required" >&2
  usage >&2
  exit 2
fi

for n in "$MAX_ITERATIONS" "$MAX_BUILDER_CLEANUP_RETRIES" "$MAX_REVIEWER_FAILURES"; do
  is_positive_int "$n" || die "numeric options must be positive integers"
done

for cmd in git gh codex jq rg; do
  require_cmd "$cmd"
done

WORKDIR="$(resolve_repo_root)"
cd "$WORKDIR"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SUBMODULE_ROOT: root of the eternal-cycler installation.
# Script lives at <submodule>/scripts/, so go up 1 level.
SUBMODULE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Path to eternal-cycler relative to the git repo root, for use in agent prompts.
SUBMODULE_REL="$(realpath --relative-to="$WORKDIR" "$SUBMODULE_ROOT")"

# Explicit path context injected into every agent prompt.
PATH_CONTEXT="Path context (all paths are from the repository root):
- Policy docs:           ${SUBMODULE_REL}/PLANS.md, ${SUBMODULE_REL}/REVIEW.md
- ExecPlan gate:         ${SUBMODULE_REL}/scripts/execplan_gate.sh
- Verification skills:   .agents/skills/execplan-event-*/  (copied from ${SUBMODULE_REL}/assets/default-verification/ by setup.sh)
- Event index map:       .agents/skills/execplan-event-index/references/event_skill_map.tsv
- Sandbox policy:        .agents/skills/execplan-sandbox-escalation/SKILL.md
- Plans dir:             eternal-cycler-out/plans/
- PR tracking dir:       eternal-cycler-out/prs/
Paths to policy docs and gate script are relative to ${SUBMODULE_REL}/. Paths to verification skills, plans, and PR tracking are relative to the repository root."

if [[ -n "$(git ls-files -u)" ]]; then
  die "unmerged paths detected; resolve conflicts first"
fi
if has_tracked_dirty; then
  die "tracked working tree is dirty; commit/stash before running the loop"
fi

TARGET_BRANCH=""
TARGET_BRANCH="$(resolve_current_branch || true)"
[[ -n "$TARGET_BRANCH" ]] || die "unable to resolve current branch"
"$SCRIPT_DIR/run_builder_reviewer_doctor.sh" --head-branch "$TARGET_BRANCH" >/dev/null

if [[ -n "$PR_URL" ]]; then
  "$SCRIPT_DIR/run_builder_reviewer_doctor.sh" --pr-url "$PR_URL" >/dev/null

  pr_info_json="$(gh pr view "$PR_URL" --json url,headRefName,number 2>/dev/null || true)"
  [[ -n "$pr_info_json" ]] || die "failed to read PR metadata: $PR_URL"

  pr_head_branch="$(jq -r '.headRefName // ""' <<< "$pr_info_json")"
  PR_URL="$(jq -r '.url // ""' <<< "$pr_info_json")"

  [[ -n "$pr_head_branch" ]] || die "failed to resolve PR head branch from: $PR_URL"
  if [[ "$pr_head_branch" != "$TARGET_BRANCH" ]]; then
    die "--pr-url head branch ($pr_head_branch) must match current local branch ($TARGET_BRANCH)"
  fi
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

RUN_ID="loop-$(date -u +%Y%m%dT%H%M%SZ)-$$"

log "loop started (branch=$TARGET_BRANCH, start_commit=$START_COMMIT, run_id=$RUN_ID)"

initial_builder_prompt="You are the BUILDER agent in an autonomous loop.

${PATH_CONTEXT}

Start by reading ${SUBMODULE_REL}/PLANS.md in full.
Follow the ExecPlan lifecycle defined in that document to complete the task below.

Task:
${TASK_TEXT}

Requirements:
- Work only on branch: ${TARGET_BRANCH}
- Leave your code edits in the worktree/index; the loop script will stage, commit, and push automatically.
- Keep unrelated baseline untracked files untouched.
- Try up to 3 implementation attempts before declaring failure.
- Return exactly one JSON object and nothing else:
  {\"plan_doc_filename\":\"<relative-plan-path>\",\"result\":\"success|failed_after_3_retries\",\"failure_reason\":\"<empty-if-success>\",\"pr_title\":\"<concise PR title>\",\"pr_body\":\"<PR description in markdown>\"}
- Use result=success only when implementation is complete for this request.
- Use result=failed_after_3_retries only after 3 attempts fail and include concrete failure reason in failure_reason.
- pr_title must be a concise, action-oriented title describing the change (e.g. \"feat: add input validation to login form\").
- pr_body must be a markdown description of what was changed and why, suitable for a pull request body."

builder_schema_file="$(write_builder_output_schema)"
if ! run_codex_prompt_capture "builder_initial" "$MODEL_BUILDER" "$initial_builder_prompt" "$builder_schema_file"; then
  rm -f "$builder_schema_file"
  die "initial builder execution failed"
fi
rm -f "$builder_schema_file"

builder_payload_json="$(parse_builder_payload_json "$LAST_CODEX_OUTPUT" || true)"
[[ -n "$builder_payload_json" ]] || die "initial builder output was not valid JSON payload"
handle_builder_payload_result "initial" "$builder_payload_json"

cleanup_result="$(run_builder_cleanup_until_stable "$START_COMMIT" || true)"
[[ -n "$cleanup_result" ]] || die "builder cleanup failed after initial task"

cleanup_kind="${cleanup_result%%|*}"
cleanup_commit="${cleanup_result#*|}"

if [[ "$cleanup_kind" == "unchanged" ]]; then
  log "no new commit detected after initial builder run; continuing to initial reviewer cycle"
else
  log "new commit detected after initial builder run: $cleanup_commit"
fi

LATEST_COMMIT="$cleanup_commit"

if [[ -z "$PR_URL" ]]; then
  _pr_title="$(jq -r '.pr_title' <<< "$builder_payload_json")"
  _pr_body="$(jq -r '.pr_body' <<< "$builder_payload_json")"
  PR_URL="$(resolve_or_create_pr_for_branch "$TARGET_BRANCH" "$_pr_title" "$_pr_body")"
  [[ -n "$PR_URL" ]] || die "failed to resolve/create PR for branch: $TARGET_BRANCH"
fi

log "target PR: $PR_URL"

REVIEWER_FAILURES=0

for ((ITERATION=1; ITERATION<=MAX_ITERATIONS; ITERATION++)); do
  log "review iteration $ITERATION started for commit $LATEST_COMMIT"

  reviewer_prompt="You are the REVIEWER agent in an autonomous loop.

${PATH_CONTEXT}

Start by reading ${SUBMODULE_REL}/REVIEW.md in full.
Follow the review policy defined in that document.

Review target:
- PR URL: ${PR_URL}
- Target commit and newer commits on head branch: ${LATEST_COMMIT}

- Do not post any GitHub comment directly in autonomous loop mode.
- Return exactly one JSON object and nothing else:
  {\"pr_url\":\"<target-pr-url>\",\"comment_body\":\"<comment body in English>\",\"approve_merge\":true|false}
- Set \"pr_url\" to the same PR URL shown above.
- If CI is running, do not wait for completion; decide from current evidence.
- If the latest plan in this PR appears unresolved after three failures, include explicit remediation request text in \"comment_body\".
- Use approve_merge=true only when merge should be approved now."

  reviewer_schema_file="$(write_reviewer_output_schema)"
  if ! run_codex_prompt_capture "reviewer" "$MODEL_REVIEWER" "$reviewer_prompt" "$reviewer_schema_file"; then
    rm -f "$reviewer_schema_file"
    REVIEWER_FAILURES=$((REVIEWER_FAILURES + 1))
    if [[ "$REVIEWER_FAILURES" -ge "$MAX_REVIEWER_FAILURES" ]]; then
      die "reviewer execution failed $REVIEWER_FAILURES times consecutively"
    fi
    continue
  fi
  rm -f "$reviewer_schema_file"

  reviewer_payload_json="$(parse_reviewer_payload_json "$LAST_CODEX_OUTPUT" || true)"
  if [[ -z "$reviewer_payload_json" ]]; then
    REVIEWER_FAILURES=$((REVIEWER_FAILURES + 1))
    if [[ "$REVIEWER_FAILURES" -ge "$MAX_REVIEWER_FAILURES" ]]; then
      die "reviewer output was not valid JSON payload for $REVIEWER_FAILURES consecutive attempts"
    fi
    log "reviewer output was not valid JSON payload; retrying review iteration"
    continue
  fi

  reviewer_pr_url="$(jq -r '.pr_url' <<< "$reviewer_payload_json")"
  reviewer_comment_body="$(jq -r '.comment_body' <<< "$reviewer_payload_json")"
  reviewer_approve_merge="$(jq -r '.approve_merge' <<< "$reviewer_payload_json")"

  normalized_reviewer_pr_url="$(normalize_pr_url "$reviewer_pr_url")"
  normalized_target_pr_url="$(normalize_pr_url "$PR_URL")"

  if [[ "$normalized_reviewer_pr_url" != "$normalized_target_pr_url" ]]; then
    REVIEWER_FAILURES=$((REVIEWER_FAILURES + 1))
    if [[ "$REVIEWER_FAILURES" -ge "$MAX_REVIEWER_FAILURES" ]]; then
      die "reviewer payload PR URL mismatch for $REVIEWER_FAILURES consecutive attempts"
    fi
    log "reviewer payload PR URL mismatch (expected=$normalized_target_pr_url got=$normalized_reviewer_pr_url)"
    continue
  fi

  if ! post_output="$(post_pr_comment "$normalized_target_pr_url" "$reviewer_comment_body")"; then
    REVIEWER_FAILURES=$((REVIEWER_FAILURES + 1))
    if [[ "$REVIEWER_FAILURES" -ge "$MAX_REVIEWER_FAILURES" ]]; then
      die "failed to post reviewer comment for $REVIEWER_FAILURES consecutive attempts"
    fi
    continue
  fi

  comment_url="$(printf '%s\n' "$post_output" | rg -o 'https://github\\.com/[^[:space:]]+' | head -n1 || true)"
  REVIEWER_FAILURES=0

  if [[ "$reviewer_approve_merge" == "true" ]]; then
    finalize_pr_tracking_doc "$PR_URL" "$TARGET_BRANCH" "$LATEST_COMMIT"
    log "reviewer approve_merge=true for commit $LATEST_COMMIT; loop finished"
    exit 0
  fi

  if [[ -n "$comment_url" ]]; then
    builder_followup_prompt="You are the BUILDER agent in an autonomous loop.

${PATH_CONTEXT}

Address the reviewer feedback in this PR comment:
${comment_url}

Requirements:
- Implement required fixes on branch ${TARGET_BRANCH}.
- Follow the ExecPlan lifecycle in ${SUBMODULE_REL}/PLANS.md.
- Leave your code edits in the worktree/index; the loop script will stage, commit, and push automatically.
- Keep unrelated baseline untracked files untouched.
- Try up to 3 implementation attempts before declaring failure.
- Return exactly one JSON object and nothing else:
  {\"plan_doc_filename\":\"<relative-plan-path>\",\"result\":\"success|failed_after_3_retries\",\"failure_reason\":\"<empty-if-success>\",\"pr_title\":\"<concise PR title>\",\"pr_body\":\"<PR description in markdown>\"}
- Use result=success only when implementation is complete for this request.
- Use result=failed_after_3_retries only after 3 attempts fail and include concrete failure reason in failure_reason.
- pr_title must be a concise, action-oriented title describing the change (e.g. \"feat: add input validation to login form\").
- pr_body must be a markdown description of what was changed and why, suitable for a pull request body."
  else
    builder_followup_prompt="You are the BUILDER agent in an autonomous loop.

${PATH_CONTEXT}

Address the reviewer feedback text below:
${reviewer_comment_body}

Requirements:
- Implement required fixes on branch ${TARGET_BRANCH}.
- Follow the ExecPlan lifecycle in ${SUBMODULE_REL}/PLANS.md.
- Leave your code edits in the worktree/index; the loop script will stage, commit, and push automatically.
- Keep unrelated baseline untracked files untouched.
- Try up to 3 implementation attempts before declaring failure.
- Return exactly one JSON object and nothing else:
  {\"plan_doc_filename\":\"<relative-plan-path>\",\"result\":\"success|failed_after_3_retries\",\"failure_reason\":\"<empty-if-success>\",\"pr_title\":\"<concise PR title>\",\"pr_body\":\"<PR description in markdown>\"}
- Use result=success only when implementation is complete for this request.
- Use result=failed_after_3_retries only after 3 attempts fail and include concrete failure reason in failure_reason.
- pr_title must be a concise, action-oriented title describing the change (e.g. \"feat: add input validation to login form\").
- pr_body must be a markdown description of what was changed and why, suitable for a pull request body."
  fi

  builder_schema_file="$(write_builder_output_schema)"
  if ! run_codex_prompt_capture "builder_followup" "$MODEL_BUILDER" "$builder_followup_prompt" "$builder_schema_file"; then
    rm -f "$builder_schema_file"
    die "builder follow-up execution failed at iteration $ITERATION"
  fi
  rm -f "$builder_schema_file"

  builder_payload_json="$(parse_builder_payload_json "$LAST_CODEX_OUTPUT" || true)"
  [[ -n "$builder_payload_json" ]] || die "builder follow-up output was not valid JSON payload at iteration $ITERATION"
  handle_builder_payload_result "followup_iter_${ITERATION}" "$builder_payload_json"

  cleanup_result="$(run_builder_cleanup_until_stable "$LATEST_COMMIT" || true)"
  [[ -n "$cleanup_result" ]] || die "builder cleanup failed at iteration $ITERATION"

  cleanup_kind="${cleanup_result%%|*}"
  cleanup_commit="${cleanup_result#*|}"

  if [[ "$cleanup_kind" == "changed" ]]; then
    LATEST_COMMIT="$cleanup_commit"
    log "new commit detected after builder follow-up: $LATEST_COMMIT"
  else
    log "no new commit after builder follow-up; running next reviewer cycle"
  fi

done

die "max iterations reached without reviewer approve_merge=true for commit $LATEST_COMMIT"
