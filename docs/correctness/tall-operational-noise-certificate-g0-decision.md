# Tall Operational-Noise Certificate G0 Decision

## Decision and scope

**PASS for G0.** The reviewed revisions provide deterministic Tall source-construction and CPU
feasibility evidence, together with fixed-size Lean kernel compilation evidence.

This decision does not claim a Tall backend execution, a generated Lean certificate, G1 or G2
semantic completeness, G3 generation or acceptance, or end-to-end certificate acceptance. The
committed JSON remains pre-gate review evidence only.

## Reviewed revisions

- Tall G0 evidence revision:
  `dee3207d63715a29452591c43759bafdc9d50646`
- Canonical statement-row implementation revision:
  `069a36f7d1008826a7e29f1aa9fbec08a21ab865`
- Rust-only Lean kernel spike revision:
  `ae3e1031bc47d61a641ba99c1ef96f0b751881bd`

## Evidence identity and authority

- Artifact: `docs/correctness/tall-operational-noise-certificate-g0.json`
- Artifact SHA-256:
  `bae5456a4ae432284ff33f568d2106100c99493e2c379ef93b90ff0a441283ee`
- Artifact byte length: `8568724`
- Wrapper: `mxx.operational-noise.tall-g0-review-evidence`, schema version 1
- Embedded CPU evidence: `mxx.operational-noise.g0-cpu-evidence`, schema version 6
- Ordered profiles: `Security0`, then `Security128`
- Status authority, at both wrapper and embedded-observation levels:
  `CpuObservationOnlyNotG0HardGateOrTallEvidence`

The existing Rust implementation remains the semantic and performance authority. The JSON is
authoritative only for the deterministic pre-gate observations reproduced by the commands below.
The Lean build and axiom scan are authoritative only for successful compilation and the reported
axiom dependencies; they do not turn the JSON into an acceptance artifact.

Schema version 6 uses the same canonical statement-row authority for the exact `N`, inventory
encoding, and typed certificate projection. The corrected event count includes reached
`GadgetDecompose` operations in their exact closed or program scope. The fixed observations are:

| Profile | Expression rows | Program rows | Source rows | Event rows | Total `N` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Security0` | 30330 | 269 | 3850 | 1526 | 35975 |
| `Security128` | 71560 | 282 | 3883 | 3775 | 79500 |

The retained-size observations affected by the corrected statement rows are:

| Profile | Inventory logical items | Inventory canonical bytes | Proof canonical bytes | Proof projection peak | Aggregate generator peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| `Security0` | 2815524 | 9696690 | 7400723 | 16960908 | 16960908 |
| `Security128` | 5948790 | 29937424 | 21699933 | 57275330 | 318062027 |

The proof payload logical-item counts are 1736721 and 5075381, respectively. The exact LUT row
counts are 211263 and 63217087.

## Reproduction results

The three fixed-profile evidence gates passed with one test each:

```text
cargo test -p mxx-gadgets --features gpu \
  --test test_gpu_tall_bgg_nested_rns_modq_arith \
  fixed_tall_g0_security0_evidence_is_deterministic \
  -- --ignored --exact --nocapture
```

Result: PASS, 1 passed in 214.81 seconds.

```text
cargo test -p mxx-gadgets --features gpu \
  --test test_gpu_tall_bgg_nested_rns_modq_arith \
  fixed_tall_g0_security128_evidence_is_deterministic \
  -- --ignored --exact --nocapture
```

Result: PASS, 1 passed in 991.76 seconds.

```text
cargo test -p mxx-gadgets --features gpu \
  --test test_gpu_tall_bgg_nested_rns_modq_arith \
  fixed_tall_g0_combined_evidence_matches_committed_golden \
  -- --ignored --exact --nocapture
```

Result: PASS, 1 passed in 612.53 seconds. The regeneration run also passed, 1 passed in 596.28
seconds, before this non-regenerating byte-equality replay.

## Schema-version invariant comparison

A same-run typed comparison against the previous schema-version-5 evidence passed. Runtime
acceptance, plaintext and ciphertext moduli, margins, deterministic core counters, exact LUT row
counts and frontier products, proof payload logical-item counts and canonical bytes, and recorder
peak retained logical items were unchanged. The CPU evidence schema does not export recorder
current retained items.

The exact LUT retained logical-item counts were unchanged. Its canonical encoding changed from the
corrected typed row references: 7386914 to 7386403 bytes for `Security0`, and 2490317131 to
2490316442 bytes for `Security128`. The recorder peaks remained 1070792 and 3033359. The aggregate
generator peak changed for `Security0`, where proof projection dominates, and remained 318062027
for `Security128`, where the LUT dominates. Proof canonical payload bytes remained unchanged; the
expression/program namespace correction did not alter the chronological proof payload in these
two fixed profiles.

The repository-authoritative Lean target passed:

```text
cd lean
lake build Mxx.Certificate.OperationalNoise.Fixtures
```

Result: PASS. The reported axiom dependencies were limited to `propext` and `Quot.sound`.

## Independent review

Independent design and code review concluded **PASS** with no findings. The reviewed revisions and
artifact hash above were clean, and the worktree was clean at the decision point.
