#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/scripts/lib${PYTHONPATH:+:$PYTHONPATH}"

run_python=0
run_rust=0

if [[ $# -eq 0 ]]; then
  run_python=1
  run_rust=1
else
  for arg in "$@"; do
    case "$arg" in
      --python)
        run_python=1
        ;;
      --rust)
        run_rust=1
        ;;
      *)
        echo "Usage: $0 [--python] [--rust]" >&2
        exit 2
        ;;
    esac
  done
fi

if [[ $run_python -eq 1 ]]; then
  python3 -m unittest discover -s scripts/lib/codex_workflow/tests -p 'test_*.py'
fi

if [[ $run_rust -eq 1 ]]; then
  RUST_TEST_NOCAPTURE="${RUST_TEST_NOCAPTURE:-1}" \
    cargo test -r --lib

  python3 -m repo_validation maybe-run-gpu-repeat
fi
