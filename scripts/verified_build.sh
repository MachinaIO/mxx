#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT

cd "${workspace_root}"

MXX_CORRECTNESS_OUT_DIR="${scratch}/correctness" \
    cargo run -p mxx-correctness --example emit_correctness
MXX_CORRECTNESS_OUT_DIR="${scratch}/gadgets" \
    cargo run -p mxx-gadgets --example emit_correctness

diff -ru \
    crates/correctness/lean/MxxCorrectness/Generated \
    "${scratch}/correctness"
diff -ru \
    crates/gadgets/lean/MxxGadgets/Generated \
    "${scratch}/gadgets"

(
    cd lean
    lake build Mxx MxxCorrectness MxxGadgets
)

cargo run -p mxx-correctness --example verify_correctness
cargo run -p mxx-gadgets --example verify_correctness
cargo build --workspace "$@"
