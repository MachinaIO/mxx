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
  echo "FAILURE_SUMMARY=plan file not found"
  echo "STATUS=fail"
  exit 1
fi

plan_is_abs=0
if [[ "$PLAN" == /* ]]; then
  plan_is_abs=1
fi

commands=()
commands+=("rg -n docs/prs/active/|docs/prs/completed/ <plan>")

pr_doc_path=""
moved_pr_doc_from=""
moved_pr_doc_to=""
rollback_plan_path=""

emit_fail() {
  local summary="$1"
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=$summary"
  echo "STATUS=fail"
  if [[ -n "$rollback_plan_path" ]]; then
    echo "PLAN_PATH=$rollback_plan_path"
  fi
  exit 1
}

to_plan_style_path() {
  local rel="$1"
  if [[ "$plan_is_abs" -eq 1 ]]; then
    printf "%s/%s" "$PWD" "$rel"
  else
    printf "%s" "$rel"
  fi
}

rollback_to_active() {
  local target_rel target_path

  if [[ "$PLAN" == docs/plans/completed/* || "$PLAN" == */docs/plans/completed/* ]]; then
    target_rel="docs/plans/active/$(basename "$PLAN")"
    target_path="$(to_plan_style_path "$target_rel")"
    if [[ "$PLAN" != "$target_path" && -f "$PLAN" ]]; then
      mkdir -p "$(dirname "$target_path")"
      commands+=("rollback plan $PLAN -> $target_path")
      mv "$PLAN" "$target_path"
      PLAN="$target_path"
      rollback_plan_path="$target_path"
    fi
  fi

  if [[ -n "$moved_pr_doc_from" && -n "$moved_pr_doc_to" && -f "$moved_pr_doc_to" ]]; then
    mkdir -p "$(dirname "$moved_pr_doc_from")"
    commands+=("rollback pr doc $moved_pr_doc_to -> $moved_pr_doc_from")
    mv "$moved_pr_doc_to" "$moved_pr_doc_from"
    pr_doc_path="$moved_pr_doc_from"
    return
  fi

  if [[ -n "$pr_doc_path" && "$pr_doc_path" == docs/prs/completed/* && -f "$pr_doc_path" ]]; then
    target_path="docs/prs/active/$(basename "$pr_doc_path")"
    mkdir -p "$(dirname "$target_path")"
    commands+=("rollback pr doc $pr_doc_path -> $target_path")
    mv "$pr_doc_path" "$target_path"
    pr_doc_path="$target_path"
  fi
}

fail_validation() {
  local summary="$1"
  rollback_to_active
  emit_fail "$summary"
}

fail_after_git_add() {
  local summary="$1"
  emit_fail "$summary"
}

has_unresolved_latest_nonpass_event() {
  sed -n '/<!-- verification-ledger:start -->/,/<!-- verification-ledger:end -->/p' "$PLAN" | awk '
    /event_id=/ && /status=/ {
      event=""
      status=""
      n=split($0, parts, ";")
      for (i=1; i<=n; i++) {
        if (parts[i] ~ /event_id=/) {
          tmp=parts[i]
          gsub(/^.*event_id=/, "", tmp)
          gsub(/^ +| +$/, "", tmp)
          event=tmp
        }
        if (parts[i] ~ /status=/) {
          tmp=parts[i]
          gsub(/^.*status=/, "", tmp)
          gsub(/^ +| +$/, "", tmp)
          status=tmp
        }
      }
      if (event != "") {
        latest[event]=status
      }
    }
    END {
      for (e in latest) {
        if (e == "execplan.post_completion") {
          continue
        }
        if (latest[e] == "fail" || latest[e] == "escalated") {
          print e ":" latest[e]
          exit 0
        }
      }
      exit 1
    }
  '
}

if ! rg -q "event_id=execplan.pre_creation;.*status=pass" "$PLAN"; then
  fail_validation "missing pass entry for execplan.pre_creation"
fi

if ! awk '
  /event_id=/ && /status=pass/ {
    event=""
    n=split($0, parts, ";")
    for (i=1; i<=n; i++) {
      if (parts[i] ~ /event_id=/) {
        gsub(/^.*event_id=/, "", parts[i])
        gsub(/^ +| +$/, "", parts[i])
        event=parts[i]
      }
    }
    if (event != "" && event != "execplan.pre_creation" && event != "execplan.post_completion") {
      found=1
    }
  }
  END { exit(found?0:1) }
' "$PLAN"; then
  fail_validation "missing pass entry for non-lifecycle event"
fi

pr_doc_path="$(rg -o "docs/prs/(active|completed)/[^ )\t]+\\.md" "$PLAN" | head -n1 || true)"
if [[ -z "$pr_doc_path" ]]; then
  fail_validation "missing PR tracking document linkage in plan"
fi
commands+=("open $pr_doc_path")

if [[ ! -f "$pr_doc_path" ]]; then
  fail_validation "referenced PR tracking document not found: $pr_doc_path"
fi

missing_fields=()
for field in "PR link" "branch" "commit" "summary/content"; do
  if ! rg -qi "$field" "$pr_doc_path"; then
    missing_fields+=("$field")
  fi
done
if [[ ${#missing_fields[@]} -gt 0 ]]; then
  fail_validation "PR tracking metadata incomplete; missing fields: $(IFS=','; echo "${missing_fields[*]}")"
fi

pr_ready="${EXECPLAN_PR_READY:-auto}"
case "$pr_ready" in
  ready|not_ready|auto)
    ;;
  *)
    fail_validation "invalid EXECPLAN_PR_READY value: $pr_ready (expected ready|not_ready|auto)"
    ;;
esac

if [[ "$pr_ready" == "auto" ]]; then
  if rg -q "^- \[ \]" "$PLAN" || unresolved_latest="$(has_unresolved_latest_nonpass_event)"; then
    if [[ -n "${unresolved_latest:-}" ]]; then
      commands+=("auto-ready blocked by unresolved latest non-pass event: $unresolved_latest")
    fi
    pr_ready="not_ready"
  else
    pr_ready="ready"
  fi
fi

if [[ "$pr_ready" == "ready" ]]; then
  if [[ "$pr_doc_path" == docs/prs/active/* ]]; then
    mkdir -p docs/prs/completed
    target="docs/prs/completed/$(basename "$pr_doc_path")"
    commands+=("mv $pr_doc_path $target")
    mv "$pr_doc_path" "$target"
    moved_pr_doc_from="$pr_doc_path"
    moved_pr_doc_to="$target"
    pr_doc_path="$target"
  fi
else
  blockers="${EXECPLAN_BLOCKERS:-remaining blockers not provided}"
  {
    echo
    echo "## Post-Completion Blockers"
    echo
    echo "- ${blockers}"
  } >> "$PLAN"
  {
    echo
    echo "## Readiness Blockers"
    echo
    echo "- ${blockers}"
  } >> "$pr_doc_path"
fi

commands+=("git status --short")
git status --short >/dev/null

if ! rg -q "<!-- execplan-start-untracked:start -->" "$PLAN"; then
  fail_validation "missing execplan start untracked snapshot in plan; run pre-creation with --plan and retry"
fi
if ! rg -q "<!-- execplan-start-tracked:start -->" "$PLAN"; then
  fail_validation "missing execplan start tracked snapshot in plan; run pre-creation with --plan and retry"
fi

declare -A baseline_tracked_hash=()
while IFS= read -r line; do
  payload="${line#- start_tracked_change: }"
  hash="${payload%%$'\t'*}"
  path="${payload#*$'\t'}"
  hash="$(echo "$hash" | xargs)"
  path="$(echo "$path" | sed -E 's/^ +| +$//g')"
  if [[ -n "$path" && "$path" != "(none)" ]]; then
    baseline_tracked_hash["$path"]="$hash"
  fi
done < <(
  sed -n '/<!-- execplan-start-tracked:start -->/,/<!-- execplan-start-tracked:end -->/p' "$PLAN" \
    | rg "^- start_tracked_change:"
)

declare -A baseline_untracked_hash=()
while IFS= read -r line; do
  payload="${line#- start_untracked_file: }"
  hash="${payload%%$'\t'*}"
  path="${payload#*$'\t'}"
  hash="$(echo "$hash" | xargs)"
  path="$(echo "$path" | sed -E 's/^ +| +$//g')"
  if [[ -n "$path" && "$path" != "(none)" ]]; then
    baseline_untracked_hash["$path"]="$hash"
  fi
done < <(
  sed -n '/<!-- execplan-start-untracked:start -->/,/<!-- execplan-start-untracked:end -->/p' "$PLAN" \
    | rg "^- start_untracked_file:"
)

mapfile -d '' -t tracked_changes < <(git diff --name-only -z HEAD --)
mapfile -d '' -t untracked_changes < <(git ls-files --others --exclude-standard -z)

plan_rel="$PLAN"
if [[ "$plan_rel" == "$PWD/"* ]]; then
  plan_rel="${plan_rel#"$PWD/"}"
fi

declare -A seen_paths=()
changed_paths=()
for path in "${tracked_changes[@]}"; do
  [[ -z "$path" ]] && continue
  include_path=0
  if [[ -n "${baseline_tracked_hash[$path]:-}" ]]; then
    current_hash="(deleted)"
    if [[ -e "$path" ]]; then
      current_hash="$(git hash-object -- "$path" 2>/dev/null || echo "(missing)")"
    fi
    if [[ "$current_hash" != "${baseline_tracked_hash[$path]}" ]]; then
      include_path=1
    fi
  else
    include_path=1
  fi

  if [[ "$include_path" -eq 1 ]] && [[ -z "${seen_paths[$path]:-}" ]]; then
    seen_paths["$path"]=1
    changed_paths+=("$path")
  fi
done

skipped_untracked=()
for path in "${untracked_changes[@]}"; do
  [[ -z "$path" ]] && continue

  # Definition:
  # - Pre-existing untracked files that are unchanged since pre-creation remain
  #   unstaged.
  # - Files newly created during this lifecycle are staged (including the
  #   target plan document).
  # - Pre-existing untracked files modified during this lifecycle are staged.
  include_path=0

  if [[ "$path" == "$plan_rel" ]]; then
    include_path=1
  elif [[ -n "${baseline_untracked_hash[$path]:-}" ]]; then
    current_hash="$(git hash-object -- "$path" 2>/dev/null || echo "(missing)")"
    if [[ "$current_hash" != "${baseline_untracked_hash[$path]}" ]]; then
      include_path=1
    fi
  else
    include_path=1
  fi

  if [[ "$include_path" -eq 1 ]]; then
    if [[ -z "${seen_paths[$path]:-}" ]]; then
      seen_paths["$path"]=1
      changed_paths+=("$path")
    fi
  else
    skipped_untracked+=("$path")
  fi
done

if [[ ${#changed_paths[@]} -gt 0 ]]; then
  commands+=("git add changed plan files")
  git add -- "${changed_paths[@]}"
else
  commands+=("no changed files to stage")
fi

if [[ ${#skipped_untracked[@]} -gt 0 ]]; then
  commands+=("skip unrelated untracked files")
fi

if [[ ${#changed_paths[@]} -gt 0 ]] && ! git diff --quiet -- "${changed_paths[@]}"; then
  fail_after_git_add "unstaged changes remain in staged target files after git add; do not edit files after staging"
fi

if ! git diff --cached --quiet; then
  final_message="${EXECPLAN_FINAL_COMMIT_MESSAGE:-docs: finalize execplan post-completion state}"
  commands+=("git commit -m <finalize-message>")
  if ! git commit -m "$final_message" >/dev/null 2>&1; then
    fail_after_git_add "failed to create final post-completion commit"
  fi
else
  commands+=("no staged changes to commit")
fi

if [[ ${#changed_paths[@]} -gt 0 ]] && ! git diff --quiet -- "${changed_paths[@]}"; then
  fail_after_git_add "staged target files changed after git add/commit; stop and restage before push"
fi

branch="$(git branch --show-current)"
commands+=("git push origin $branch")
if ! git push origin "$branch" >/dev/null 2>&1; then
  fail_after_git_add "failed to push final post-completion state to origin/$branch"
fi

if [[ "$pr_ready" == "ready" ]]; then
  ready_confirmed="${EXECPLAN_PR_READY_CONFIRMED:-0}"
  if command -v gh >/dev/null 2>&1; then
    commands+=("gh pr view --json isDraft")
    is_draft="$(gh pr view --json isDraft --jq '.isDraft' 2>/dev/null || echo "unknown")"
    if [[ "$is_draft" == "true" ]]; then
      commands+=("gh pr ready")
      if ! gh pr ready >/dev/null 2>&1; then
        if [[ "$ready_confirmed" != "1" ]]; then
          fail_after_git_add "failed to transition PR to ready-for-review via gh (set EXECPLAN_PR_READY_CONFIRMED=1 only after manual UI fallback)"
        fi
      fi
    elif [[ "$is_draft" == "false" ]]; then
      commands+=("pr already ready")
    else
      if [[ "$ready_confirmed" != "1" ]]; then
        fail_after_git_add "failed to detect PR draft state via gh (set EXECPLAN_PR_READY_CONFIRMED=1 only after manual UI confirmation)"
      fi
    fi
  else
    if [[ "$ready_confirmed" != "1" ]]; then
      fail_after_git_add "gh CLI unavailable; set PR ready in web UI and re-run with EXECPLAN_PR_READY_CONFIRMED=1"
    fi
  fi
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
