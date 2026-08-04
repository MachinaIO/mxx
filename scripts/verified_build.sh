#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT

cd "${workspace_root}"

MXX_CORRECTNESS_OUT_DIR="${scratch}/correctness" \
    cargo run -p mxx-correctness --example emit_correctness

MXX_CORRECTNESS_OUT_DIR="${scratch}/we" \
    cargo run -p mxx-we --example emit_correctness

diff -ru \
    crates/correctness/lean/MxxCorrectness/Generated \
    "${scratch}/correctness"

diff -ru \
    crates/we/lean/MxxWe/Generated \
    "${scratch}/we"

(
    cd lean
    lake build Mxx MxxCorrectness MxxWe mxx_diamond_checker
)

cargo run -p mxx-correctness --example verify_correctness
cargo build --workspace "$@"
