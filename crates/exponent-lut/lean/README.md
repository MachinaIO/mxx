# Exponent-LUT Lean certificates

Generic integer matrix action bounds live in
`crates/primitives/lean/PrimitivesBounds.lean`, and balanced radix bounds in
`crates/primitives/lean/PrimitivesCrtRadix.lean`. Registered gadget witnesses and
execution bridges live in `crates/runtime/lean/RuntimeDecomposition.lean`.
Retained witnesses select digits of the corrected target; approximate execution
retains its invocation-local residual premise. Exact reconstruction requires a
registered layout with zero dropped moduli.

Application mathematics in `crates/exponent-lut/lean/*.lean` covers native program
evaluation, readout, modulus switching, strict rounding, and CRT recomposition.
The exact cache specializations carry explicit exact-layout premises; the
residual-aware cache equation keeps the source-secret multiplication of the
decomposition residual. Static package compilation alone does not certify an
approximate application execution. Build with the pinned Lean toolchain:

```sh
cd crates/exponent-lut/lean
lake build
```

Candidate-specific modules belong in ignored `test_data` directories. The
exporters in `crates/exponent-lut/src/lean.rs` and
`crates/exponent-lut/src/lean/` connect the reusable lemmas to the actual frozen
preprocessing and verification graphs, backend layouts, public inputs, and noise
report. Their proof chain follows the native program, readout, routed source
bounds, paired switching, strict decoding, and final CRT residual. Graph-derived
relations and shared execution witnesses keep internal samples, caches, and
outputs tied to the same executions.

The IR layer generates `Claim.lean`. The application proof contract is
`RefreshCorrectness.lean`, with theorem
`ExponentGeneratedRefreshCorrectness.correctness : GeneratedClaim.CorrectnessClaim`.
The IR layer then assembles `Certificate.lean` and its
`GeneratedCertificate.correctness` declaration against that claim. Application
code supplies the proof, not a replacement claim.

## Acceptance and checking

Worst-case numerical acceptance is a prerequisite for certification. The gate in
`crates/exponent-lut/src/lean/certify.rs` checks the selected candidate's complete
source closure and core certificate before retaining worst-case acceptance.
Average-case acceptance continues to use the Rust simulation; it does not produce
a worst-case Lean certificate.

The checker builds the static dependency closure first, records its source and
object fingerprints, and compiles candidate modules into a fresh check directory
in dependency order, with `Certificate` last. It compares candidate bytes with the
exporter's authoritative in-memory source map before and after checking. Missing
proofs or dependencies, compiler failures, timeouts, source changes, and failed
axiom audits reject certification. The certificate's allowed axioms are
`propext`, `Classical.choice`, and `Quot.sound`.

Only successful checks write `checked-record.json`, containing source and dependency
hashes, module logs, and elapsed time. Export-only tests do not certify a candidate.

The route exporter proves the actual per-slot/per-digit scalar table and its
offset/index lookup. Readout helper lemmas are emitted from the graph producers
required by their consumers; the complete execution graphs remain unchanged.

`ExponentNativeFirstStage` and `ExponentNativeProgramProof` are generated for
the current candidate, rather than retained as obsolete concrete snapshots in
this directory. Coverage includes the first-stage derivation and its `Approx`
soundness corollary, every native program stage, and the actual final program
output bound. The positive certificate test requires both modules' kernel
objects and logs in addition to `Certificate.olean` and `checked-record.json`.

`MXX_EXPONENT_LUT_LEAN_TIMEOUT_SECONDS` sets a positive integer wall-clock budget
for the entire checker invocation, including static builds, generated modules,
and final identity checks. Its default is 21600 seconds (six hours), an operational
allowance rather than a completion-time guarantee. Invalid values are errors.
Library callers can supply an explicit `Duration` to `certify`. This wall-clock
budget does not change the Lean limits: stack 131072, 2000000 heartbeats, and
recursion depth 16384.

## Actual-certificate tests

Run these ignored library tests from the repository root with the pinned Lean
package and complete candidate proof exporter available:

```sh
MXX_EXPONENT_LUT_LEAN_TIMEOUT_SECONDS=21600 cargo test -r -p mxx-exponent-lut --lib lean::certify::tests::test_actual_full_certificate -- --ignored --exact --nocapture
MXX_EXPONENT_LUT_LEAN_TIMEOUT_SECONDS=21600 cargo test -r -p mxx-exponent-lut --lib lean::certify::tests::test_actual_certificate_rejects_wrong_theorem_type -- --ignored --exact --nocapture
```

The positive test searches for a numerically accepted worst-case candidate,
materializes its actual graphs, and requires a checked core certificate. The
optional negative test preserves the actual claim but replaces the application
theorem with a proof of `True`. It requires a type mismatch specifically in
`Certificate`; an earlier failure or timeout does not satisfy the test.

Both tests share the accepted-worst-case fixture with the export-only test.
Fixture overrides are `MXX_PRIMITIVE_TEST_RING_DIMENSION` (default 32),
`MXX_PRIMITIVE_TEST_CRT_DEPTH` (maximum 8), `MXX_PRIMITIVE_TEST_CRT_BITS` (40), and
`MXX_PRIMITIVE_TEST_BASE_BITS` (4). Positive artifacts go under
`test_data/exponent_lut_actual_full_certificate`; negative artifacts go under
`test_data/exponent_lut_actual_wrong_theorem`. Expected negative rejection writes
`rejection-record.json` with the error kind, certificate log path, and source
hashes. The negative test checks theorem-to-claim binding; ordinary tamper tests
separately check source-integrity rejection.
