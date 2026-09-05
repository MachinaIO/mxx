# Witness-encryption proofs

Handwritten Lean sources live directly in this directory. There is no facade import module.
`MxxWe` is the Lake package/library name and the namespace of shared definitions, not a directory.

## Sources and build boundaries

- `Decoder.lean` proves the Boolean decoder's threshold guarantees.
- `Bounds.lean` supplies bound arithmetic and its proofs.
- `Certificate.lean` is the fixed final theorem and the starting point for a correctness audit.
- `Diamond*.lean` proves properties of the actual Diamond IR, from local identities and loop
  invariants through `DiamondClaimCorrectnessProof.lean`.
- `tests/SemanticCounterexamples.lean` contains separate semantic regression checks.
- `lakefile.toml`, `lake-manifest.json`, and `lean-toolchain` pin the build configuration.

The default `lake build` includes `Certificate` and all handwritten proofs. It requires a
selected candidate, because the proofs import generated modules such as `Stage_encrypt`,
`Stage_decrypt`, `Claim`, and `DiamondProofParameters`. On a fresh checkout, build the reusable
modules with `lake build Decoder Bounds` before generating the first candidate; this bootstrap
does not prove the final theorem. The runtime, IR, and BGG Lean dependencies must also be built
before running the production checker.

## Editing and auditing in VS Code

After parameter search reports a retained candidate directory, run from the repository root:

```sh
python3 scripts/select_we_lean_candidate.py <candidate-directory>
```

Then run `lake build` in `crates/we/lean`. Open `Certificate.lean` there and restart the Lean
server if VS Code still reports imports from the previous configuration. No custom `LEAN_PATH`
is needed. `lake env lean Certificate.lean` checks the same source using the editor's Lake
environment, and prints the final theorem's axioms.

The selector requires Python 3.11+ and copies only the generated sources declared by the
`DiamondCandidate` library into a local snapshot under `.lake/editor-candidates`. The ignored
`generated` symlink switches to that snapshot only after all copies succeed; previous snapshots
are retained. No compiled artifacts or handwritten proofs are copied. Thus imports use the
current proof sources here, not the proof copies in the original certificate directory.
Audit `generated/Claim.lean`, including `Runs`, alongside `Certificate.lean`.

Selection is explicit, not performed for every search candidate. It is not a correctness or
security check: `lake build` must succeed afterwards. The selected IR is a snapshot, not a live
view of the Rust DSL. After changing the protocol or parameters, regenerate through parameter
search and select the new artifact. The original artifact path is recorded in
`generated/selection.json`. Generated snapshots and the selection link are not committed.

## Candidate verification

The parameter search in `crates/we/src/diamond/parameter_search.rs` invokes the certificate
exporter in `crates/we/src/lean/diamond.rs`. It generates the candidate's IR modules and numeric
certificate, copies the handwritten Diamond sources unchanged into the artifact directory, and
checks their dependency graph. Generated files and compiler outputs are not source files here.
The exporter requires an empty output directory so an earlier candidate cannot leave stale proofs.

The handwritten theorem `DiamondGeneratedProof.generated_claim_correctness_of_capped_gate`
in `DiamondClaimCorrectnessProof.lean` proves the IR-derived claim given the strict numeric gate.
The generated `NumericCertificate.lean` proves that gate for a passing candidate (or its negation
for a rejected candidate). For passing candidates the exporter copies `Certificate.lean`
unchanged into the artifact directory. It exports the final theorem
`DiamondCertificate.correctness : GeneratedClaim.CorrectnessClaim` with no numeric premise.
Audit the generated `Claim.lean` alongside it: that file defines the actual conclusion and the
linked execution assumptions directly from the IR. Rejected candidates have no final certificate.

A CPU-only unit test exercises that complete production path from the repository root:

```sh
cargo test -p mxx-we --lib selected_candidate_retains_a_kernel_checked_certificate -- --ignored --nocapture
```

It requires the dependent Lean packages to be built. Its security estimator is a test double;
the Lean verification is real, but the test is not a GPU run or a security-parameter assessment.
