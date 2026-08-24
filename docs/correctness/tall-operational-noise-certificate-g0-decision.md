# Tall Operational-Noise Certificate G0 Decision

## Decision and scope

**PASS for G0.** The reviewed revisions provide deterministic Tall source-construction and CPU
feasibility evidence, together with fixed-size Lean kernel compilation evidence.

This decision does not claim a Tall backend execution, a generated Lean certificate, G1 or G2
semantic completeness, G3 generation or acceptance, or end-to-end certificate acceptance. The
committed JSON remains pre-gate review evidence only.

## Reviewed revisions

- Tall G0 evidence revision:
  `3f083892af77cebaca5f9d0bf705b7e168208cb5`
- Rust-only Lean kernel spike revision:
  `ae3e1031bc47d61a641ba99c1ef96f0b751881bd`

## Evidence identity and authority

- Artifact: `docs/correctness/tall-operational-noise-certificate-g0.json`
- Artifact SHA-256:
  `6a061702e4a4b609d606d26418a506c7c497d5a665558d4a71151af83497a95a`
- Wrapper: `mxx.operational-noise.tall-g0-review-evidence`, schema version 1
- Embedded CPU evidence: `mxx.operational-noise.g0-cpu-evidence`, schema version 5
- Ordered profiles: `Security0`, then `Security128`
- Status authority, at both wrapper and embedded-observation levels:
  `CpuObservationOnlyNotG0HardGateOrTallEvidence`

The existing Rust implementation remains the semantic and performance authority. The JSON is
authoritative only for the deterministic pre-gate observations reproduced by the commands below.
The Lean build and axiom scan are authoritative only for successful compilation and the reported
axiom dependencies; they do not turn the JSON into an acceptance artifact.

## Reproduction results

The three fixed-profile evidence gates passed with one test each:

```text
cargo test -p mxx-gadgets --features gpu \
  --test test_gpu_tall_bgg_nested_rns_modq_arith \
  fixed_tall_g0_security0_evidence_is_deterministic \
  -- --ignored --exact --nocapture
```

Result: PASS, 1 passed.

```text
cargo test -p mxx-gadgets --features gpu \
  --test test_gpu_tall_bgg_nested_rns_modq_arith \
  fixed_tall_g0_security128_evidence_is_deterministic \
  -- --ignored --exact --nocapture
```

Result: PASS, 1 passed.

```text
cargo test -p mxx-gadgets --features gpu \
  --test test_gpu_tall_bgg_nested_rns_modq_arith \
  fixed_tall_g0_combined_evidence_matches_committed_golden \
  -- --ignored --exact --nocapture
```

Result: PASS, 1 passed.

The Rust-only fixed-size Lean kernel spike also passed:

```text
cargo test -p mxx-correctness --lib \
  operational_noise::g0_kernel_spikes::g0_kernel_spikes_compile_exact_sizes \
  -- --ignored --exact --nocapture
```

Result: PASS with `balanced_rows=5000` and `fuel_haves=1000`; 1 passed.

The repository-authoritative Lean target passed:

```text
cd lean
lake build Mxx.Certificate.OperationalNoise.Fixtures
```

Result: PASS. The reported axiom dependencies were limited to `propext` and `Quot.sound`.

## Independent review

Independent design and code review concluded **PASS** with no findings. The reviewed revisions and
artifact hash above were clean, and the worktree was clean at the decision point.
