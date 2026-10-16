# Date-row proof and export format

This package proves `PeProof.certificate : GeneratedClaim.CorrectnessClaim` for
the date predicate. The proposition is the current `generated/Claim.lean`, not
a copy of the previous export. Proof generation does not rewrite `generated/`.

## Scope names

`scope_names.json` was derived from corresponding declarations in the previous
and current exports with:

```sh
python3 export_format.py --previous /path/to/previous-export
```

The derivation compares declarations in export order within each stage, checks
equal declaration counts, and rejects mismatches of names retained by both
exports. It records 108 online and 33 preprocessing scope renamings, together
with the keygen inner loop's change from suffix `14_1286` to `14_5`.
This correspondence is specific to these two exports; changes in declaration
order or graph structure require reviewing the mapping again.

The proof generators retain symbolic loop coordinates while specializing the
W128 templates, then `export_format.translate` resolves the instantiated names.
Parsers use the inverse mapping when inspecting generated scope bodies.
The generated facts' `section` variables and `generatedRootCtx` notation need
no compatibility declarations: the proofs use their exported `Ctx`, wire, and
`facts` APIs directly, and the parsers read fact fields rather than explicit
implicit-binder lists.

## Unit-row helper accumulation

`PeKeygenScatterBase.lean.in` defines the contribution as a `1 × 256` row.
`gen_keygen.py` obtains the destination table from the current export; Lean's
`scatterList_eq` proves that table agrees with the mathematical helper index.
`PeKeygenScatter.lean.in` proves each update by `unitRow_window` and
`sliceMatrix_column`, sums the 45 lanes, and gathers all 256 rows into the
existing helper sum. `PeKeygenRoot.lean.in` reads the target residual through a
column slice. This replaces the old one-hot-column concatenation proofs.

The exporter also eliminates the identity helper-copy witnesses in encryption.
`PeEncrypt.lean.in` uses the resulting definitional equalities directly.

## Regeneration and validation

Regenerate the definitions from the protocol DSL first, from the repository root:

```sh
MXX_GPU_BACKEND=cuda \
MXX_PE_ROUND_TRIP_MIN_LOG_RING_DIMENSION=15 \
MXX_PE_ROUND_TRIP_MAX_LOG_RING_DIMENSION=15 \
MXX_PE_ROUND_TRIP_MAX_CRT_DEPTH=15 \
MXX_PE_ROUND_TRIP_CRT_BITS=24 \
MXX_PE_ROUND_TRIP_BASE_BITS=12 \
MXX_PE_ROUND_TRIP_SECURITY_BITS=128 \
MXX_PE_ROUND_TRIP_NOISE_MODEL=average \
MXX_PE_ROUND_TRIP_FAILURE_EXPONENT=128 \
MXX_PE_ROUND_TRIP_COMPACT_ENCODINGS=1 \
MXX_PE_ROUND_TRIP_LUT_RADIX=16 \
MXX_PE_ROUND_TRIP_LEAN_EXPORT=average-w256-n32768-crt15x24-compact \
MXX_PE_ROUND_TRIP_EXPORT_ONLY=1 \
cargo test -r -p mxx-pe --features gpu --test gpu_round_trip \
  test_gpu_pe_date -- --exact --nocapture
```

This requires the CUDA build toolchain and `lattice-estimator-cli` on `PATH`,
but the export-only workflow does not execute GPU operations. It runs the
parameter search, including the noise and lattice-security checks, before
recording setup, encryption, key generation, and decryption. Confirm that the
accepted parameters are `n = 32768`, CRT `15 x 24`, base `2^12`, `m_B = 295`,
`W = 256`, and `T_max = 2`.

The 2026-10-17 export from Rust merge commit `e2a2a8fd9` reproduced all 26
generated Lean files byte for byte from the preserved proof migration.
Regenerating the proof sources also preserved the exported definitions byte
for byte. The scope-name mapping remains specific to this export structure.

Run from this package directory:

```sh
python3 gen_all.py
ulimit -v 60000000
lake build PeGenerated
lake build PeProof
```

For a trust check, import `PeCertificate` and run
`#print axioms PeProof.certificate`. The permitted result is
`[propext, Classical.choice, Quot.sound]`; the proof must not depend on
`sorryAx` or additional axioms. No native decision oracle is used.

Validation on 2026-10-17 succeeded for both `lake build PeGenerated` and
`lake build PeProof`. The latter reported five existing tactic-linter warnings
and no proof errors. Lean reported the following axiom dependencies:

```text
PeProof.certificate: [propext, Classical.choice, Quot.sound]
PeProof.run_correct: [propext, Classical.choice, Quot.sound]
PeProof.margin: [propext]
PeCertificate.certificate: [propext, Classical.choice, Quot.sound]
```

The certificate concerns the generated DSL relations and their mathematical
primitive and sampling semantics. It does not certify CUDA kernels, GPU memory
lifetimes, checkpoint serialization, or a correspondence between those runtime
implementations and the relations. Runtime round trips are validated separately.
