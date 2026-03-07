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

commands=()
commands+=("rg -n eternal-cycler-out/prs/active/|eternal-cycler-out/prs/completed/ <plan>")

pr_doc_path=""
rollback_plan_path=""
result_plan_path=""
ROLLBACK_BLOCK_START="<!-- execplan-post-completion-rollback:start -->"
ROLLBACK_BLOCK_END="<!-- execplan-post-completion-rollback:end -->"

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

to_abs_repo_path() {
  local path="$1"
  if [[ "$path" == /* ]]; then
    printf '%s' "$path"
  else
    printf '%s/%s' "$REPO_ROOT" "$path"
  fi
}

rollback_receipt_rel_for_plan() {
  local plan_path="$1"
  printf 'eternal-cycler-out/plans/active/.post-completion-rollbacks/%s.receipt' "$(basename "$plan_path")"
}

generate_rollback_token() {
  printf '%s\n' "$(date -u +%s%N)-$$-$RANDOM-$(basename "$PLAN")" | git hash-object --stdin | cut -c1-40
}

extract_rollback_field_from_plan() {
  local key="$1"
  sed -n "/${ROLLBACK_BLOCK_START}/,/${ROLLBACK_BLOCK_END}/p" "$PLAN" | \
    sed -n -E "s/^- ${key}:[[:space:]]+(.+)$/\\1/p" | head -n1 | sed -E 's/[[:space:]]+$//'
}

extract_rollback_field_from_receipt() {
  local receipt_path="$1"
  local key="$2"
  sed -n -E "s/^${key}=(.+)$/\\1/p" "$receipt_path" | head -n1 | sed -E 's/[[:space:]]+$//'
}

set_rollback_provenance_block() {
  local source_rel="$1"
  local retry_rel="$2"
  local receipt_rel="$3"
  local token="$4"
  local recorded_at="$5"
  local tmp

  tmp="$(mktemp)"
  awk \
    -v start="$ROLLBACK_BLOCK_START" \
    -v end="$ROLLBACK_BLOCK_END" \
    -v source_rel="$source_rel" \
    -v retry_rel="$retry_rel" \
    -v receipt_rel="$receipt_rel" \
    -v token="$token" \
    -v recorded_at="$recorded_at" '
    function print_block() {
      print start
      print "- rollback_source_plan: " source_rel
      print "- rollback_retry_plan: " retry_rel
      print "- rollback_receipt_doc: " receipt_rel
      print "- rollback_token: " token
      print "- rollback_recorded_at: " recorded_at
      print end
    }
    index($0, start) {
      if (!replaced) {
        print_block()
        replaced=1
      }
      in_block=1
      next
    }
    index($0, end) {
      in_block=0
      next
    }
    !in_block { print }
    END {
      if (!replaced) {
        print ""
        print "## Post-completion Rollback Provenance"
        print ""
        print_block()
      }
    }
  ' "$PLAN" > "$tmp"

  mv "$tmp" "$PLAN"
}

write_rollback_provenance() {
  local source_rel="$1"
  local retry_rel="$2"
  local receipt_rel receipt_path token recorded_at

  receipt_rel="$(rollback_receipt_rel_for_plan "$retry_rel")"
  receipt_path="$(to_abs_repo_path "$receipt_rel")"
  token="$(generate_rollback_token)"
  recorded_at="$(date -u +"%Y-%m-%d %H:%MZ")"

  mkdir -p "$(dirname "$receipt_path")"
  cat > "$receipt_path" <<EOF
event_id=execplan.post_completion
rollback_source_plan=${source_rel}
rollback_retry_plan=${retry_rel}
rollback_receipt_doc=${receipt_rel}
rollback_token=${token}
rollback_recorded_at=${recorded_at}
EOF
  commands+=("write $(repo_rel_path "$receipt_path")")
  commands+=("record rollback provenance in $(repo_rel_path "$PLAN")")

  set_rollback_provenance_block "$source_rel" "$retry_rel" "$receipt_rel" "$token" "$recorded_at"
}

validate_active_retry_provenance() {
  local latest_status expected_source_rel expected_retry_rel expected_receipt_rel
  local source_rel retry_rel receipt_rel token receipt_path
  local receipt_event receipt_source receipt_retry receipt_doc receipt_token

  latest_status="$(latest_post_completion_status || true)"
  if [[ "$latest_status" == "escalated" ]]; then
    fail_validation "execplan.post_completion escalation is terminal; the failed plan must stay under eternal-cycler-out/plans/completed/"
  fi

  if [[ "$latest_status" != "fail" ]]; then
    fail_validation "execplan.post_completion requires a completed plan path; active plans are allowed only after rollback from a failed post_completion attempt"
  fi

  expected_source_rel="eternal-cycler-out/plans/completed/$(basename "$PLAN")"
  expected_retry_rel="eternal-cycler-out/plans/active/$(basename "$PLAN")"
  expected_receipt_rel="$(rollback_receipt_rel_for_plan "$PLAN")"

  source_rel="$(extract_rollback_field_from_plan "rollback_source_plan")"
  retry_rel="$(extract_rollback_field_from_plan "rollback_retry_plan")"
  receipt_rel="$(extract_rollback_field_from_plan "rollback_receipt_doc")"
  token="$(extract_rollback_field_from_plan "rollback_token")"

  if [[ -z "$source_rel" || -z "$retry_rel" || -z "$receipt_rel" || -z "$token" ]]; then
    fail_validation "missing rollback provenance for active post_completion retry; rerun from the completed plan path to create a real rollback"
  fi

  if [[ "$source_rel" != "$expected_source_rel" || "$retry_rel" != "$expected_retry_rel" || "$receipt_rel" != "$expected_receipt_rel" ]]; then
    fail_validation "rollback provenance does not match the expected completed/active retry paths for $(basename "$PLAN")"
  fi

  receipt_path="$(to_abs_repo_path "$receipt_rel")"
  commands+=("open $(repo_rel_path "$receipt_path")")
  if [[ ! -f "$receipt_path" ]]; then
    fail_validation "missing rollback provenance receipt for active post_completion retry: $(repo_rel_path "$receipt_path")"
  fi

  receipt_event="$(extract_rollback_field_from_receipt "$receipt_path" "event_id")"
  receipt_source="$(extract_rollback_field_from_receipt "$receipt_path" "rollback_source_plan")"
  receipt_retry="$(extract_rollback_field_from_receipt "$receipt_path" "rollback_retry_plan")"
  receipt_doc="$(extract_rollback_field_from_receipt "$receipt_path" "rollback_receipt_doc")"
  receipt_token="$(extract_rollback_field_from_receipt "$receipt_path" "rollback_token")"

  if [[ "$receipt_event" != "execplan.post_completion" || "$receipt_source" != "$source_rel" || \
        "$receipt_retry" != "$retry_rel" || "$receipt_doc" != "$receipt_rel" || "$receipt_token" != "$token" ]]; then
    fail_validation "rollback provenance receipt does not match the plan's recorded retry provenance"
  fi
}

rollback_to_active() {
  local source_rel target_rel target_path

  if [[ "$PLAN" == eternal-cycler-out/plans/completed/* || "$PLAN" == */eternal-cycler-out/plans/completed/* ]]; then
    source_rel="$(repo_rel_path "$PLAN")"
    target_rel="eternal-cycler-out/plans/active/$(basename "$PLAN")"
    target_path="$(to_plan_style_path "$target_rel")"
    if [[ "$PLAN" != "$target_path" && -f "$PLAN" ]]; then
      mkdir -p "$(dirname "$target_path")"
      commands+=("rollback plan $(repo_rel_path "$PLAN") -> $(repo_rel_path "$target_path")")
      mv "$PLAN" "$target_path"
      PLAN="$target_path"
      rollback_plan_path="$target_path"
      write_rollback_provenance "$source_rel" "$target_rel"
    fi
  fi

}

fail_validation() {
  local summary="$1"
  rollback_to_active
  emit_fail "$summary"
}

latest_post_completion_status() {
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
      if (event == "execplan.post_completion") {
        latest=status
      }
    }
    END {
      if (latest != "") {
        print latest
        exit 0
      }
      exit 1
    }
  '
}

promote_retry_plan_to_completed() {
  local target_rel target_path receipt_rel receipt_path

  case "$PLAN" in
    eternal-cycler-out/plans/completed/*|*/eternal-cycler-out/plans/completed/*)
      return 0
      ;;
    eternal-cycler-out/plans/active/*|*/eternal-cycler-out/plans/active/*)
      validate_active_retry_provenance

      target_rel="eternal-cycler-out/plans/completed/$(basename "$PLAN")"
      target_path="$(to_plan_style_path "$target_rel")"
      if [[ "$PLAN" != "$target_path" ]]; then
        mkdir -p "$(dirname "$target_path")"
        commands+=("promote retry plan $(repo_rel_path "$PLAN") -> $(repo_rel_path "$target_path")")
        mv "$PLAN" "$target_path"
        PLAN="$target_path"
        result_plan_path="$target_path"
      fi
      receipt_rel="$(rollback_receipt_rel_for_plan "$PLAN")"
      receipt_path="$(to_abs_repo_path "$receipt_rel")"
      if [[ -f "$receipt_path" ]]; then
        commands+=("remove $(repo_rel_path "$receipt_path")")
        rm -f "$receipt_path"
      fi
      return 0
      ;;
    *)
      fail_validation "execplan.post_completion requires a plan under eternal-cycler-out/plans/completed/ (or an active rollback retry)"
      ;;
  esac
}

has_unresolved_latest_nonpass_event() {
  sed -n '/<!-- verification-ledger:start -->/,/<!-- verification-ledger:end -->/p' "$PLAN" | awk '
    /event_id=/ && /status=/ {
      event=""
      status=""
      failure=""
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
        if (parts[i] ~ /failure_summary=/) {
          tmp=parts[i]
          gsub(/^.*failure_summary=/, "", tmp)
          gsub(/^ +| +$/, "", tmp)
          failure=tmp
        }
      }
      if (event != "") {
        latest[event]=status
        latest_failure[event]=failure
      }
    }
    END {
      for (e in latest) {
        if (e == "execplan.pre_creation" || e == "execplan.post_creation" || \
            e == "execplan.resume" || e == "execplan.post_completion") {
          continue
        }
        if (latest[e] == "fail" && latest_failure[e] ~ /^unresolved verification status remains for execplan\./) {
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

extract_tracking_field() {
  local tracking_doc="$1"
  local key="$2"
  sed -n -E "s/^- ${key}:[[:space:]]+(.+)$/\\1/p" "$tracking_doc" | head -n1 | sed -E 's/[[:space:]]+$//'
}

extract_plan_tracking_doc() {
  sed -n -E 's/^- pr_tracking_doc:[[:space:]]+(.+)$/\1/p' "$PLAN" | head -n1 | sed -E 's/[[:space:]]+$//'
}

resolve_open_pr_for_branch() {
  local branch_name="$1"
  local pr_list_json

  [[ -n "$branch_name" ]] || return 1
  command -v gh >/dev/null 2>&1 || return 1

  commands+=("gh pr list --state open --head ${branch_name} --json url,updatedAt --limit 20")
  pr_list_json="$(gh pr list --state open --head "$branch_name" --json url,updatedAt --limit 20 2>/dev/null || true)"
  if [[ -z "$pr_list_json" ]]; then
    return 1
  fi
  jq -r '[.[]] | sort_by(.updatedAt) | reverse | .[0].url // empty' <<< "$pr_list_json"
}

promote_retry_plan_to_completed

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

pr_doc_path="$(extract_plan_tracking_doc || true)"
if [[ -z "$pr_doc_path" ]]; then
  pr_doc_path="$(rg -o "eternal-cycler-out/prs/(active|completed)/[^ ,)\t]+\\.md" "$PLAN" | head -n1 || true)"
fi
if [[ -z "$pr_doc_path" ]]; then
  fail_validation "missing PR tracking document linkage in plan"
fi

# Resolve to absolute path for file tests (plan may record repo-relative paths).
[[ "$pr_doc_path" == /* ]] || pr_doc_path="${REPO_ROOT}/${pr_doc_path}"

if [[ ! -f "$pr_doc_path" && "$pr_doc_path" == */eternal-cycler-out/prs/active/* ]]; then
  fallback_path="${REPO_ROOT}/eternal-cycler-out/prs/completed/$(basename "$pr_doc_path")"
  if [[ -f "$fallback_path" ]]; then
    commands+=("fallback pr doc $(repo_rel_path "$pr_doc_path") -> $(repo_rel_path "$fallback_path")")
    pr_doc_path="$fallback_path"
  fi
fi

commands+=("open $(repo_rel_path "$pr_doc_path")")

if [[ ! -f "$pr_doc_path" ]]; then
  fail_validation "referenced PR tracking document not found: $(repo_rel_path "$pr_doc_path")"
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
branch_name="$(extract_tracking_field "$pr_doc_path" "branch name")"
open_pr_url="$(resolve_open_pr_for_branch "$branch_name" || true)"

if [[ -n "$open_pr_url" ]]; then
  if [[ -z "$pr_url" || "$pr_url" == "(not available locally)" ]]; then
    fail_validation "PR tracking metadata missing live PR link for branch '$branch_name'; current open PR is $open_pr_url"
  fi
  if [[ "$pr_url" != "$open_pr_url" ]]; then
    fail_validation "PR tracking metadata is stale for branch '$branch_name'; recorded $pr_url but current open PR is $open_pr_url"
  fi
fi

if [[ -n "$pr_url" && "$pr_url" != "(not available locally)" ]]; then
  if command -v gh >/dev/null 2>&1; then
    commands+=("gh pr view ${pr_url} --json url,state")
    if ! gh pr view "$pr_url" --json url,state >/dev/null 2>&1; then
      fail_validation "PR tracking metadata points to an unreadable PR link: $pr_url"
    fi
  fi
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
if [[ -n "$result_plan_path" ]]; then
  echo "PLAN_PATH=$result_plan_path"
fi
echo "STATUS=pass"
