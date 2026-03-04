#!/usr/bin/env bash
set -euo pipefail

MAX_ATTEMPTS=3
PRE_CREATION_EVENT_ID="execplan.pre_creation"
POST_CREATION_EVENT_ID="execplan.post_creation"
RESUME_EVENT_ID="execplan.resume"
POST_EVENT_ID="execplan.post_completion"

usage() {
  cat <<'USAGE'
Usage:
  execplan_gate.sh --event <event_id> [--plan <plan_md>] [--attempt <n>]

Notes:
  --plan is not accepted by execplan.pre_creation (no plan file exists yet).
  All other events require --plan.
USAGE
}

is_lifecycle_event() {
  local e="$1"
  [[ "$e" == "$PRE_CREATION_EVENT_ID" || "$e" == "$POST_CREATION_EVENT_ID" \
    || "$e" == "$RESUME_EVENT_ID" || "$e" == "$POST_EVENT_ID" ]]
}

PLAN=""
EVENT=""
ATTEMPT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --plan)
      PLAN="${2:-}"
      shift 2
      ;;
    --event)
      EVENT="${2:-}"
      shift 2
      ;;
    --attempt)
      ATTEMPT_OVERRIDE="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$EVENT" ]]; then
  usage >&2
  exit 2
fi

if [[ "$EVENT" != "$PRE_CREATION_EVENT_ID" && -z "$PLAN" ]]; then
  echo "--plan is required for event: $EVENT" >&2
  usage >&2
  exit 2
fi

if [[ "$EVENT" == "$PRE_CREATION_EVENT_ID" && -n "$PLAN" ]]; then
  echo "--plan is not accepted for event: $EVENT (plan file does not exist yet)" >&2
  usage >&2
  exit 2
fi

if [[ -n "$PLAN" && ! -f "$PLAN" ]]; then
  echo "Plan file not found: $PLAN" >&2
  exit 1
fi

HAS_PLAN_FILE=0
if [[ -n "$PLAN" && -f "$PLAN" ]]; then
  HAS_PLAN_FILE=1
fi

# SUBMODULE_ROOT: root of this eternal-cycler installation (where scripts/, assets/ etc. live).
# Resolved from the script's own location so it works correctly whether eternal-cycler is used
# as a git subtree, a plain copy, or called directly from its own checkout.
SUBMODULE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# REPO_ROOT: root of the consuming git repository (may equal SUBMODULE_ROOT when eternal-cycler
# is checked out directly; differs when installed as a subtree under a consuming repo).
REPO_ROOT="$(git -C "$SUBMODULE_ROOT" rev-parse --show-toplevel)"

# Export so event scripts can locate both roots without recomputing them.
export ETERNAL_CYCLER_ROOT="$SUBMODULE_ROOT"
export REPO_ROOT

# Event map: read from the consuming repo's .agents/skills/ directory so that new events
# added by the consuming repo are automatically picked up.
# Default verification skills are copied to .agents/skills/ by setup.sh.
# To add new action events, add entries to .agents/skills/execplan-event-index/references/event_skill_map.tsv
# and create matching scripts under .agents/skills/<skill-dir>/.
EVENT_MAP="$REPO_ROOT/.agents/skills/execplan-event-index/references/event_skill_map.tsv"

if [[ ! -f "$EVENT_MAP" ]]; then
  echo "Event map not found: $EVENT_MAP" >&2
  exit 1
fi

# lookup_event_row <event_id>
# Prints "skill_dir<TAB>script_rel" for the event. Prints nothing and returns 1 if not found.
lookup_event_row() {
  local event_id="$1"
  local row
  row="$(awk -F '\t' -v e="$event_id" '$0 !~ /^#/ && $1 == e { print $2 "\t" $3; exit }' "$EVENT_MAP")"
  if [[ -z "$row" ]]; then
    return 1
  fi
  printf '%s\n' "$row"
}

is_registered_event() {
  local event_id="$1"
  lookup_event_row "$event_id" >/dev/null 2>&1
}

require_mandatory_lifecycle_events() {
  local required
  for required in "$PRE_CREATION_EVENT_ID" "$POST_CREATION_EVENT_ID" "$RESUME_EVENT_ID" "$POST_EVENT_ID"; do
    if ! is_registered_event "$required"; then
      echo "Mandatory lifecycle event missing from event map: $required" >&2
      exit 1
    fi
  done
}

resolve_event_script() {
  local event_id="$1"
  local row skill_dir script_rel script_abs

  row="$(awk -F '\t' -v e="$event_id" '$0 !~ /^#/ && $1 == e { print $2 "\t" $3; exit }' "$EVENT_MAP")"
  if [[ -z "$row" ]]; then
    echo ""
    return 1
  fi

  skill_dir="${row%%$'\t'*}"
  script_rel="${row#*$'\t'}"
  script_abs="$REPO_ROOT/.agents/skills/$skill_dir/$script_rel"

  if [[ ! -x "$script_abs" ]]; then
    echo "Event script is missing or not executable: $script_abs" >&2
    return 1
  fi

  echo "$script_abs"
}

sanitize() {
  echo "$1" | tr '\n' ' ' | sed -E 's/[;]+/,/g; s/[[:space:]]+/ /g; s/^ //; s/ $//'
}

ensure_ledger_block() {
  if [[ "$HAS_PLAN_FILE" -ne 1 ]]; then
    return 0
  fi
  if ! rg -q "<!-- verification-ledger:start -->" "$PLAN"; then
    {
      echo
      echo "## Verification Ledger"
      echo
      echo "<!-- verification-ledger:start -->"
      echo "<!-- verification-ledger:end -->"
    } >> "$PLAN"
  fi
}

ledger_lines() {
  if [[ "$HAS_PLAN_FILE" -ne 1 ]]; then
    return 0
  fi
  sed -n '/<!-- verification-ledger:start -->/,/<!-- verification-ledger:end -->/p' "$PLAN"
}

count_attempts() {
  local event_id="$1"
  ledger_lines | awk -v target="$event_id" '
    /event_id=/ {
      n=split($0, parts, ";")
      for (i=1; i<=n; i++) {
        if (parts[i] ~ /event_id=/) {
          tmp=parts[i]
          gsub(/^.*event_id=/, "", tmp)
          gsub(/^ +| +$/, "", tmp)
          if (tmp == target) { count++ }
        }
      }
    }
    END { print count+0 }
  '
}

has_pass() {
  local event_id="$1"
  ledger_lines | awk -v target="$event_id" '
    /event_id=/ && /status=/ {
      event=""; status=""
      n=split($0, parts, ";")
      for (i=1; i<=n; i++) {
        if (parts[i] ~ /event_id=/) {
          tmp=parts[i]; gsub(/^.*event_id=/, "", tmp); gsub(/^ +| +$/, "", tmp); event=tmp
        }
        if (parts[i] ~ /status=/) {
          tmp=parts[i]; gsub(/^.*status=/, "", tmp); gsub(/^ +| +$/, "", tmp); status=tmp
        }
      }
      if (event == target && status == "pass") { found=1 }
    }
    END { exit(found ? 0 : 1) }
  '
}

has_non_lifecycle_pass() {
  ledger_lines | awk '
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
      if (event != "" \
          && event != "'$PRE_CREATION_EVENT_ID'" \
          && event != "'$POST_CREATION_EVENT_ID'" \
          && event != "'$RESUME_EVENT_ID'" \
          && event != "'$POST_EVENT_ID'") {
        found=1
      }
    }
    END { exit(found ? 0 : 1) }
  '
}

find_unresolved_nonpass_event() {
  local current_event="$1"
  ledger_lines | awk -v current="$current_event" '
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
        latest_status[event]=status
      }
    }
    END {
      for (e in latest_status) {
        if (e == current) {
          continue
        }
        if (latest_status[e] == "fail" || latest_status[e] == "escalated") {
          print e ":" latest_status[e]
          exit 0
        }
      }
      exit 1
    }
  '
}

missing_required_verify_event_passes() {
  if [[ "$HAS_PLAN_FILE" -ne 1 ]]; then
    return 1
  fi
  local required_events event
  required_events="$(
    awk '
      /verify_events=/ {
        n=split($0, parts, ";")
        for (i=1; i<=n; i++) {
          if (parts[i] ~ /verify_events=/) {
            field=parts[i]
            gsub(/^.*verify_events=/, "", field)
            gsub(/^ +| +$/, "", field)
            split(field, events, ",")
            for (j in events) {
              ev=events[j]
              gsub(/^ +| +$/, "", ev)
              if (ev != "" && ev != "none" && ev != "-") {
                required[ev]=1
              }
            }
          }
        }
      }
      END {
        for (ev in required) {
          print ev
        }
      }
    ' "$PLAN" | sort
  )"

  if [[ -z "$required_events" ]]; then
    return 1
  fi

  local missing=()
  while IFS= read -r event; do
    [[ -z "$event" ]] && continue
    if ! has_pass "$event"; then
      missing+=("$event")
    fi
  done <<< "$required_events"

  if [[ ${#missing[@]} -gt 0 ]]; then
    printf '%s\n' "${missing[@]}" | sort | paste -sd ',' -
    return 0
  fi

  return 1
}

find_parallel_file_lock_conflict() {
  if [[ "$HAS_PLAN_FILE" -ne 1 ]]; then
    return 1
  fi
  local conflict
  conflict="$(
    awk '
      function trim(s) {
        gsub(/^[[:space:]]+/, "", s)
        gsub(/[[:space:]]+$/, "", s)
        return s
      }
      function extract_field(line, key,    n, parts, i, value) {
        n=split(line, parts, ";")
        for (i=1; i<=n; i++) {
          if (parts[i] ~ (key "=")) {
            sub("^.*" key "=", "", parts[i])
            value=trim(parts[i])
            return value
          }
        }
        return ""
      }
      function has_shared_lock(a, b,    n1, n2, i, j, lock1, lock2) {
        n1=split(locks[a], arr1, ",")
        n2=split(locks[b], arr2, ",")
        for (i=1; i<=n1; i++) {
          lock1=trim(arr1[i])
          if (lock1 == "" || lock1 == "none" || lock1 == "-") {
            continue
          }
          for (j=1; j<=n2; j++) {
            lock2=trim(arr2[j])
            if (lock2 == "" || lock2 == "none" || lock2 == "-") {
              continue
            }
            if (lock1 == lock2) {
              conflict_lock=lock1
              return 1
            }
          }
        }
        return 0
      }
      function depends_rec(start, target,    n, dep_list, i, dep) {
        if (start == "" || seen[start]) {
          return 0
        }
        seen[start]=1
        n=split(depends[start], dep_list, ",")
        for (i=1; i<=n; i++) {
          dep=trim(dep_list[i])
          if (dep == "" || dep == "none" || dep == "-") {
            continue
          }
          if (dep == target) {
            return 1
          }
          if (depends_rec(dep, target)) {
            return 1
          }
        }
        return 0
      }
      function depends_on(start, target) {
        delete seen
        return depends_rec(start, target)
      }
      /action_id=/ {
        action_id=extract_field($0, "action_id")
        mode[action_id]=extract_field($0, "mode")
        depends[action_id]=extract_field($0, "depends_on")
        locks[action_id]=extract_field($0, "file_locks")
        if (!(action_id in action_seen)) {
          action_seen[action_id]=1
          action_order[++action_count]=action_id
        }
      }
      END {
        for (i=1; i<=action_count; i++) {
          a=action_order[i]
          if (mode[a] != "parallel") {
            continue
          }
          for (j=i+1; j<=action_count; j++) {
            b=action_order[j]
            if (mode[b] != "parallel") {
              continue
            }
            if (!has_shared_lock(a, b)) {
              continue
            }
            if (depends_on(a, b) || depends_on(b, a)) {
              continue
            }
            printf "file_locks conflict: '\''%s'\'' shared by unordered parallel actions '\''%s'\'' and '\''%s'\''", conflict_lock, a, b
            exit 0
          }
        }
        exit 1
      }
    ' "$PLAN"
  )" || true

  if [[ -n "$conflict" ]]; then
    echo "$conflict"
    return 0
  fi
  return 1
}

append_ledger_entry() {
  if [[ "$HAS_PLAN_FILE" -ne 1 ]]; then
    return 0
  fi
  local entry="$1"
  local tmp
  tmp="$(mktemp)"

  awk -v entry="$entry" '
    /<!-- verification-ledger:end -->/ && !inserted {
      print entry
      inserted=1
    }
    { print }
    END {
      if (!inserted) {
        print "## Verification Ledger"
        print ""
        print "<!-- verification-ledger:start -->"
        print entry
        print "<!-- verification-ledger:end -->"
      }
    }
  ' "$PLAN" > "$tmp"

  mv "$tmp" "$PLAN"
}

require_mandatory_lifecycle_events
if ! is_registered_event "$EVENT"; then
  echo "Unsupported or unmapped event: $EVENT" >&2
  exit 2
fi

EVENT_SCRIPT="$(resolve_event_script "$EVENT")"

ensure_ledger_block

existing_attempts="$(count_attempts "$EVENT")"
if [[ -n "$ATTEMPT_OVERRIDE" ]]; then
  ATTEMPT="$ATTEMPT_OVERRIDE"
else
  ATTEMPT="$((existing_attempts + 1))"
fi

if ! [[ "$ATTEMPT" =~ ^[0-9]+$ ]] || [[ "$ATTEMPT" -lt 1 ]]; then
  echo "Invalid attempt value: $ATTEMPT" >&2
  exit 2
fi

STARTED_AT="$(date -u +"%Y-%m-%d %H:%MZ")"
FINAL_STATUS=""
COMMANDS=""
FAILURE_SUMMARY=""
NOTIFY_REFERENCE="not_requested"

if [[ -z "$FINAL_STATUS" ]] && ! is_lifecycle_event "$EVENT"; then
  if unresolved="$(find_unresolved_nonpass_event "$EVENT")"; then
    FINAL_STATUS="fail"
    COMMANDS="gate prerequisite: unresolved non-pass event scan"
    FAILURE_SUMMARY="unresolved verification status remains for ${unresolved}; resolve and re-run before advancing"
  fi
fi

if [[ -z "$FINAL_STATUS" ]] && ! is_lifecycle_event "$EVENT"; then
  if ! has_pass "$POST_CREATION_EVENT_ID" && ! has_pass "$RESUME_EVENT_ID"; then
    FINAL_STATUS="fail"
    COMMANDS="gate prerequisite: require ${POST_CREATION_EVENT_ID} or ${RESUME_EVENT_ID} pass"
    FAILURE_SUMMARY="missing pass evidence for ${POST_CREATION_EVENT_ID} or ${RESUME_EVENT_ID}"
  fi
fi

if [[ -z "$FINAL_STATUS" && "$EVENT" == "$POST_EVENT_ID" ]]; then
  if ! has_non_lifecycle_pass; then
    FINAL_STATUS="fail"
    COMMANDS="gate prerequisite: require non-lifecycle event pass"
    FAILURE_SUMMARY="missing pass entry for non-lifecycle verification events"
  fi
fi

if [[ -z "$FINAL_STATUS" && "$EVENT" == "$POST_EVENT_ID" ]]; then
  if missing_events="$(missing_required_verify_event_passes)"; then
    FINAL_STATUS="fail"
    COMMANDS="gate prerequisite: required verify_events pass coverage scan"
    FAILURE_SUMMARY="missing pass entries for required verify_events: ${missing_events}"
  fi
fi

if [[ -z "$FINAL_STATUS" ]] && ! is_lifecycle_event "$EVENT"; then
  if conflict="$(find_parallel_file_lock_conflict)"; then
    FINAL_STATUS="fail"
    COMMANDS="gate prerequisite: parallel file lock conflict scan"
    FAILURE_SUMMARY="$conflict"
  fi
fi

if [[ -z "$FINAL_STATUS" ]]; then
  if [[ "$ATTEMPT" -gt "$MAX_ATTEMPTS" ]]; then
    FINAL_STATUS="escalated"
    COMMANDS="gate retry bound pre-check"
    FAILURE_SUMMARY="retry bound exceeded before execution (attempt=${ATTEMPT})"
  fi
fi

if [[ -z "$FINAL_STATUS" ]]; then
  set +e
  if [[ -n "$PLAN" ]]; then
    EVENT_OUTPUT="$($EVENT_SCRIPT --plan "$PLAN" 2>&1)"
  else
    EVENT_OUTPUT="$($EVENT_SCRIPT 2>&1)"
  fi
  EVENT_RC=$?
  set -e

  COMMANDS="$(echo "$EVENT_OUTPUT" | sed -n 's/^COMMANDS=//p' | tail -n1)"
  FAILURE_SUMMARY="$(echo "$EVENT_OUTPUT" | sed -n 's/^FAILURE_SUMMARY=//p' | tail -n1)"
  RUN_STATUS="$(echo "$EVENT_OUTPUT" | sed -n 's/^STATUS=//p' | tail -n1)"
  NEW_PLAN_PATH="$(echo "$EVENT_OUTPUT" | sed -n 's/^PLAN_PATH=//p' | tail -n1)"

  if [[ -n "$NEW_PLAN_PATH" && -f "$NEW_PLAN_PATH" ]]; then
    PLAN="$NEW_PLAN_PATH"
    HAS_PLAN_FILE=1
    ensure_ledger_block
  fi

  if [[ -z "$COMMANDS" ]]; then
    COMMANDS="skill event runner ${EVENT}"
  fi

  if [[ $EVENT_RC -eq 0 && "$RUN_STATUS" == "pass" ]]; then
    FINAL_STATUS="pass"
    FAILURE_SUMMARY="none"
  else
    FINAL_STATUS="fail"
    if [[ -z "$FAILURE_SUMMARY" ]]; then
      FAILURE_SUMMARY="event runner failed"
    fi
    echo "$EVENT_OUTPUT" >&2
  fi
fi

if [[ "$FINAL_STATUS" == "fail" ]]; then
  if [[ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]]; then
    FINAL_STATUS="escalated"
    FAILURE_SUMMARY="$(sanitize "$FAILURE_SUMMARY ; retry bound exceeded (attempt=${ATTEMPT})")"
  fi
fi

if [[ -z "$FAILURE_SUMMARY" ]]; then
  FAILURE_SUMMARY="none"
fi

FINISHED_AT="$(date -u +"%Y-%m-%d %H:%MZ")"
ENTRY="- attempt_record: event_id=${EVENT}; attempt=${ATTEMPT}; status=${FINAL_STATUS}; started_at=${STARTED_AT}; finished_at=${FINISHED_AT}; commands=$(sanitize "$COMMANDS"); failure_summary=$(sanitize "$FAILURE_SUMMARY"); notify_reference=$(sanitize "$NOTIFY_REFERENCE");"
append_ledger_entry "$ENTRY"

echo "EVENT=$EVENT"
echo "ATTEMPT=$ATTEMPT"
echo "STATUS=$FINAL_STATUS"

if [[ "$FINAL_STATUS" == "pass" ]]; then
  exit 0
fi

exit 1
