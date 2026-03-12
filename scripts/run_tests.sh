#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHONPATH="$REPO_ROOT/scripts/lib${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m unittest discover -s scripts/lib/codex_workflow/tests -p 'test_*.py'

RUST_TEST_NOCAPTURE="${RUST_TEST_NOCAPTURE:-1}" \
  cargo test -r --lib
