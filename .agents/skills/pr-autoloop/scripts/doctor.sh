#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  doctor.sh --pr-url <url> [--offline-ok]
  doctor.sh --help

Checks:
  - required CLIs (`git`, `gh`, `codex`, `jq`)
  - `gh auth status`
  - `codex login status`
  - `gh pr view` for the target PR URL
USAGE
}

PR_URL=""
OFFLINE_OK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pr-url)
      PR_URL="${2:-}"
      shift 2
      ;;
    --offline-ok)
      OFFLINE_OK=1
      shift
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

if [[ -z "$PR_URL" ]]; then
  echo "--pr-url is required" >&2
  usage >&2
  exit 2
fi

missing=0
for bin in git gh codex jq; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "[FAIL] missing command: $bin" >&2
    missing=1
  else
    echo "[OK] found command: $bin"
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

hard_fail=0
warn_count=0

run_check() {
  local label="$1"
  local cmd="$2"
  local out_file
  out_file="$(mktemp)"

  set +e
  bash -lc "$cmd" >"$out_file" 2>&1
  rc=$?
  set -e

  if [[ "$rc" -eq 0 ]]; then
    echo "[OK] $label"
    rm -f "$out_file"
    return 0
  fi

  summary="$(head -n 1 "$out_file" | sed -E 's/[[:space:]]+/ /g')"
  if [[ "$OFFLINE_OK" -eq 1 ]] && grep -Eqi 'error connecting|network|timed out|connection refused|could not resolve host|api.github.com' "$out_file"; then
    echo "[WARN] $label (offline tolerated): ${summary:-unknown error}"
    warn_count=$((warn_count + 1))
    rm -f "$out_file"
    return 0
  fi

  echo "[FAIL] $label: ${summary:-unknown error}" >&2
  cat "$out_file" >&2
  hard_fail=1
  rm -f "$out_file"
  return 1
}

run_check "GitHub authentication" "gh auth status"
run_check "Codex authentication" "codex login status"
run_check "PR metadata access" "gh pr view '$PR_URL' --json number,title,headRefName,baseRefName,url,state,isDraft"

if [[ "$hard_fail" -ne 0 ]]; then
  echo "doctor result: FAIL" >&2
  exit 1
fi

echo "doctor result: PASS"
echo "warnings: $warn_count"
