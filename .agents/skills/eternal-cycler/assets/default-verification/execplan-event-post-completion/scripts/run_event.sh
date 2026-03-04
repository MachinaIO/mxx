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

# REPO_ROOT: root of the consuming git repository.
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"

commands=()
commands+=("rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan>")

pr_doc_path=""
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
    printf "%s/%s" "$REPO_ROOT" "$rel"
  else
    printf "%s" "$rel"
  fi
}

rollback_to_active() {
  local target_rel target_path

  if [[ "$PLAN" == eternal-cycler-out/plans/completed/* || "$PLAN" == */eternal-cycler-out/plans/completed/* ]]; then
    target_rel="eternal-cycler-out/plans/active/$(basename "$PLAN")"
    target_path="$(to_plan_style_path "$target_rel")"
    if [[ "$PLAN" != "$target_path" && -f "$PLAN" ]]; then
      mkdir -p "$(dirname "$target_path")"
      commands+=("rollback plan $PLAN -> $target_path")
      mv "$PLAN" "$target_path"
      PLAN="$target_path"
      rollback_plan_path="$target_path"
    fi
  fi

}

fail_validation() {
  local summary="$1"
  rollback_to_active
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
        if (e == "execplan.pre_creation" || e == "execplan.post_creation" || \
            e == "execplan.resume" || e == "execplan.post_completion") {
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

extract_pr_link_from_tracking_doc() {
  local tracking_doc="$1"
  sed -n -E 's/^- PR link:[[:space:]]+(.+)$/\1/p' "$tracking_doc" | head -n1 | sed -E 's/[[:space:]]+$//'
}

if ! rg -q "event_id=execplan.post_creation;.*status=pass" "$PLAN" && \
   ! rg -q "event_id=execplan.resume;.*status=pass" "$PLAN"; then
  fail_validation "missing pass entry for execplan.post_creation or execplan.resume"
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
    if (event != "" && event != "execplan.pre_creation" && event != "execplan.post_creation" \
        && event != "execplan.resume" && event != "execplan.post_completion") {
      found=1
    }
  }
  END { exit(found?0:1) }
' "$PLAN"; then
  fail_validation "missing pass entry for non-lifecycle event"
fi

pr_doc_path="$(rg -o "eternal-cycler-out/prs/(active|completed)/[^ )\t]+\\.md" "$PLAN" | head -n1 || true)"
if [[ -z "$pr_doc_path" ]]; then
  fail_validation "missing PR tracking document linkage in plan"
fi

# Resolve to absolute path for file tests (plan may record repo-relative paths).
[[ "$pr_doc_path" == /* ]] || pr_doc_path="${REPO_ROOT}/${pr_doc_path}"

if [[ ! -f "$pr_doc_path" && "$pr_doc_path" == */eternal-cycler-out/prs/active/* ]]; then
  fallback_path="${REPO_ROOT}/eternal-cycler-out/prs/completed/$(basename "$pr_doc_path")"
  if [[ -f "$fallback_path" ]]; then
    commands+=("fallback pr doc $pr_doc_path -> $fallback_path")
    pr_doc_path="$fallback_path"
  fi
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

pr_url="$(extract_pr_link_from_tracking_doc "$pr_doc_path")"
if [[ -z "$pr_url" || "$pr_url" == "(not available locally)" ]]; then
  fail_validation "PR tracking metadata missing resolvable PR link"
fi

if rg -q "^- \[ \]" "$PLAN"; then
  fail_validation "plan still contains incomplete Progress actions"
fi

if unresolved_latest="$(has_unresolved_latest_nonpass_event)"; then
  fail_validation "latest verification event is unresolved: $unresolved_latest"
fi

commands+=("git status --short")
git status --short >/dev/null

if ! rg -q "<!-- execplan-start-untracked:start -->" "$PLAN"; then
  fail_validation "missing execplan start untracked snapshot in plan; run execplan.post_creation and retry"
fi
if ! rg -q "<!-- execplan-start-tracked:start -->" "$PLAN"; then
  fail_validation "missing execplan start tracked snapshot in plan; run execplan.post_creation and retry"
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
