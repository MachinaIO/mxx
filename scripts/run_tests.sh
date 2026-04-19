#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT/scripts/lib${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v cargo >/dev/null 2>&1 && [[ -x "${HOME:-}/.cargo/bin/cargo" ]]; then
  export PATH="${HOME}/.cargo/bin:${PATH:-}"
fi

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
  cargo +nightly fmt --all
  cargo test -r --features gpu --no-run

  rust_test_log="$(mktemp)"
  if RUST_TEST_NOCAPTURE="${RUST_TEST_NOCAPTURE:-1}" \
    cargo test -r --lib >"$rust_test_log" 2>&1; then
    cat "$rust_test_log"
  else
    status=$?
    cat "$rust_test_log"
    if grep -q "signal: 11, SIGSEGV: invalid memory reference" "$rust_test_log"; then
      echo "[run_tests] Retrying release lib tests with RUST_TEST_THREADS=1 after SIGSEGV"
      RUST_TEST_NOCAPTURE="${RUST_TEST_NOCAPTURE:-1}" \
        RUST_TEST_THREADS=1 \
        cargo test -r --lib
    else
      rm -f "$rust_test_log"
      exit "$status"
    fi
  fi
  rm -f "$rust_test_log"

  python3 -m repo_validation maybe-run-gpu-repeat
fi
