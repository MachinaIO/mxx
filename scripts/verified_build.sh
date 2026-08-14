#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "${scratch}"' EXIT

cd "${workspace_root}"

MXX_REGENERATE_CORRECTNESS=1 \
    MXX_CORRECTNESS_OUT_DIR="${scratch}/correctness" \
    cargo run -p mxx-correctness --example emit_correctness

MXX_REGENERATE_CORRECTNESS=1 \
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
    lake build Mxx MxxCorrectness MxxWe mxx_diamond_derivation_checker mxx_analysis_facts
)

lean/.lake/build/bin/mxx_analysis_facts target/correctness/m0-analysis-facts.json
python3 scripts/audit_correctness_ir.py --check

cargo build --workspace "$@"
