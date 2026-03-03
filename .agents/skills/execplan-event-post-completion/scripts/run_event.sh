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

commands=()
commands+=("rg -n docs/prs/active/|docs/prs/completed/ <plan>")

if ! rg -q "event_id=execplan.pre_creation;.*status=pass" "$PLAN"; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=missing pass entry for execplan.pre_creation"
  echo "STATUS=fail"
  exit 1
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
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=missing pass entry for non-lifecycle event"
  echo "STATUS=fail"
  exit 1
fi

pr_doc_path="$(rg -o "docs/prs/(active|completed)/[^ )\t]+\\.md" "$PLAN" | head -n1 || true)"
if [[ -z "$pr_doc_path" ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=missing PR tracking document linkage in plan"
  echo "STATUS=fail"
  exit 1
fi
commands+=("open $pr_doc_path")

if [[ ! -f "$pr_doc_path" ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=referenced PR tracking document not found: $pr_doc_path"
  echo "STATUS=fail"
  exit 1
fi

missing_fields=()
for field in "PR link" "branch" "commit" "summary/content"; do
  if ! rg -qi "$field" "$pr_doc_path"; then
    missing_fields+=("$field")
  fi
done
if [[ ${#missing_fields[@]} -gt 0 ]]; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=PR tracking metadata incomplete; missing fields: $(IFS=','; echo "${missing_fields[*]}")"
  echo "STATUS=fail"
  exit 1
fi

pr_ready="${EXECPLAN_PR_READY:-auto}"
case "$pr_ready" in
  ready|not_ready|auto)
    ;;
  *)
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=invalid EXECPLAN_PR_READY value: $pr_ready (expected ready|not_ready|auto)"
    echo "STATUS=fail"
    exit 1
    ;;
esac

if [[ "$pr_ready" == "auto" ]]; then
  if rg -q "^- \\[ \\]" "$PLAN" || rg -q "status=(fail|escalated)" "$PLAN"; then
    pr_ready="not_ready"
  else
    pr_ready="ready"
  fi
fi

if [[ "$pr_ready" == "ready" ]]; then
  ready_confirmed="${EXECPLAN_PR_READY_CONFIRMED:-0}"
  if command -v gh >/dev/null 2>&1; then
    commands+=("gh pr ready")
    if ! gh pr ready >/dev/null 2>&1; then
      if [[ "$ready_confirmed" != "1" ]]; then
        echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
        echo "FAILURE_SUMMARY=failed to transition PR to ready-for-review via gh (set EXECPLAN_PR_READY_CONFIRMED=1 only after manual UI fallback)"
        echo "STATUS=fail"
        exit 1
      fi
    fi
  else
    if [[ "$ready_confirmed" != "1" ]]; then
      echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
      echo "FAILURE_SUMMARY=gh CLI unavailable; set PR ready in web UI and re-run with EXECPLAN_PR_READY_CONFIRMED=1"
      echo "STATUS=fail"
      exit 1
    fi
  fi

  if [[ "$pr_doc_path" == docs/prs/active/* ]]; then
    mkdir -p docs/prs/completed
    target="docs/prs/completed/$(basename "$pr_doc_path")"
    commands+=("mv $pr_doc_path $target")
    mv "$pr_doc_path" "$target"
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
commands+=("git add -A")
commands+=("git commit -m <finalize-message> [if there are staged changes]")
commands+=("git push origin <current-branch>")

git status --short >/dev/null
git add -A
if ! git diff --cached --quiet; then
  final_message="${EXECPLAN_FINAL_COMMIT_MESSAGE:-docs: finalize execplan completion and post-validation results}"
  if ! git commit -m "$final_message" >/dev/null 2>&1; then
    echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
    echo "FAILURE_SUMMARY=failed to create final post-completion commit"
    echo "STATUS=fail"
    exit 1
  fi
fi

branch="$(git branch --show-current)"
if ! git push origin "$branch" >/dev/null 2>&1; then
  echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
  echo "FAILURE_SUMMARY=failed to push final post-completion state to origin/$branch"
  echo "STATUS=fail"
  exit 1
fi

echo "COMMANDS=$(IFS=' | '; echo "${commands[*]}")"
echo "FAILURE_SUMMARY=none"
echo "STATUS=pass"
