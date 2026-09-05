# Compositional Lean progress handoff

Status snapshot: 2026-09-05

This note records a resumable implementation checkpoint. It is not an acceptance
of the full Stage A or Diamond proof, and it does not replace the correctness
specification or the compositional implementation plan.

## Verified bounded work

- The WE decoder arithmetic package in `crates/we/lean` builds with:
  `XDG_CACHE_HOME=/tmp/mxx-we-cache lake build`.
  It proves the inclusive interval decoder for both Boolean messages, arbitrary
  admissible `q >= 4`, negative and positive integer residuals, and the stated
  `B < decoderRadius q` bound. It also includes the negative ceil-half modular
  identity. The package contains no `sorry`, custom axiom, `admit`, or
  `native_decide`.
- The BGG `multiplication_core` audit accepted the independent compatible-secret
  relation. Its Lean build and direct compilation passed, and `#print axioms`
  reported only the standard Lean foundations (`propext`, `Classical.choice`,
  and `Quot.sound`). This is a scoped algebra result, not a Diamond theorem.
- The IR-core Lean package built, and the focused IR-core Lean tests passed
  (14/14). The reviewed exporter subset preserves scoped bindings and typed
  iteration; unsupported operations remain explicit failures rather than
  default or unknown semantics.
- The Rust layout adapter's focused unit tests passed (5/5). It exports the
  ordered CRT basis returned by the concrete `DCRTPolyParams` supplied to
  backend setup, validates product/order/base/digit metadata, rejects conflicting
  duplicate ring keys, rejects insufficient numeric digit capacity, and derives
  separate regular and compact digit counts.
- Constant zero/identity/polynomial lowering and the named-output boundary
  projections have independent scoped acceptance. The regenerated constant
  proof fixture compiles, including aliases and retained internal outputs.

The interrupted CrtRadix edits were repaired in the current checkpoint. The
following dependent package builds now pass with the shared cache:

- `crates/primitives/lean`: 1809 jobs.
- `crates/gadgets/lean`: 1811 jobs.
- `crates/runtime/lean`: 1813 jobs.

The new capacity declarations report only the standard Lean foundations
(`propext`, `Classical.choice`, and `Quot.sound`) under `#print axioms`. The
capacity and downstream foundation linter/deprecation warnings have been repaired
without adding semantic placeholders. Final coherent package validation is tracked below.

## Runtime public-gadget API checkpoint

`crates/runtime/lean/RuntimePrimitives.lean` now separates the two runtime
paths:

- `preimageRuns` remains the sampled-secret relation used by existing fixtures.
  It takes the explicit ordered arguments `(publicMatrix, trapdoor, target,
  cutoff, preimage)` and retains the coupled public-matrix/trapdoor identity and
  successful-sample cutoff relation.
- `publicGadgetPreimageRuns` requires a public-gadget trapdoor, an exact public
  matrix match, regular mode, and an existential concrete `RegularLayout`. It
  computes the deterministic balanced CRT decomposition from the ordered CRT
  towers and requires the public matrix to be the canonical CRT-weighted gadget
  and the returned matrix to equal that decomposition. Reconstruction for an
  arbitrary target is proved, not a relation conjunct. It deliberately has no node-cutoff conjunct: public
  execution derives its supported digit bound from concrete layout facts rather
  than asserting a successful sampled bound.
- `preimageRunsDispatched` takes a fixed `BackendContext`, checks the trapdoor
  metadata against its ring-key lookup, and dispatches between those two relations,
  with `preimageRunsDispatched_public` and
  `preimageRunsDispatched_equation` eliminating either branch. The compact
  (`small=true`) interpretation is not treated as full-q reconstruction and is
  not part of this regular public-gadget gate.
- `RegularLayout` now stores the numeric tower capacity
  `modulus ≤ base ^ digitsPerTower`; its `ordered` conversion invokes
  `OrderedCrtLayout.ofCapacity`. No runtime caller supplies a free
  `residual_zero` function.

The exporter now emits `MxxRuntime.preimageRunsDispatched` in
`crates/ir-core/src/lean/mod.rs`; focused source assertions and the sampled
payload/nonzero-error fixtures use `preimageRunsDispatched_equation`. This
name switch is an exporter-owned dependency; the runtime relation remains the
single source of the sampled/public dispatch semantics.

## Actual graph and linked claim checkpoint

The fixed backend context, lexical child-layout checks, full plain-hash tag/key
semantics, canonical coefficient extraction, scalar broadcasting, and both-loop
binding guards have independent scoped acceptance. No loop lanes are enumerated.
The runtime renderer returns a private-field `LeanBackendArtifact`; the WE
assembler checks every exported root's layout metadata against that artifact.
The concrete generator rejects an all-missing context before claim assembly.

The production certificate exporter, invoked by Diamond parameter search, exports all actual frozen roots:
encryption (295 stored nodes), decryption (454), three requirements (79, 371,
199), and ideal (1). Backend and all six graph modules independently compile.
The generated `Claim.lean` also compiles and its construction is independently
accepted. It binds shared externals, all eight artifact values through exact
output tuple slots, all actual requirement runs with true outputs, the actual
ideal run, one backend/hash interpretation, and the observed coefficient residual.
The structural fixture alone does not satisfy `CorrectnessClaim`'s numeric gate;
the preferred candidate now has an independently accepted proof, described below.
The WE assembler now lowers the bundle's declared input contracts into
`ValidExternals`, conjoined with `Runs`. Integer ranges are inclusive, family
contracts use symbolic quantification, and byte lengths are exact. The mapping
uses the same external input IDs and rejects missing, duplicate, unsupported,
or type-inconsistent declarations. The three focused Rust tests, regenerated
claim, and actual raw-word/packing proofs have independent scoped acceptance.
The earlier omission gave a stronger domain than specified; it did not establish
the raw Boolean-word facts needed by the packing proof. No counterexample to the
small fixture was claimed, and no final noise premise is added by this correction.

The example uses a small structural configuration (`q = 1009`, ring dimension 8,
gadget base 16). No feasible final noise bound has been established for it.
The application proof sources in `crates/we/lean` import these actual
generated graph modules, rather than defining a second protocol:

- `DiamondGateProof.lean`: independently accepted active product-gate extraction,
  canonical decomposition equality, digit bound 8, and reconstruction.
- `DiamondEncryptedGateProof.lean`: direct Lean compilation, standard-axiom
  check, and independent scoped review pass. It composes actual decryption
  scopes with `multiplication_core`, retaining both BGG error terms. Its local
  encoding equations are induction premises, not assumptions of the final claim.
- The injector witness, selector, index, initial-state, transition, and layer
  templates have independent scoped acceptance. They preserve the actual root
  sample/pool families, exact source/target addresses, and common secret through
  the generated sequential body without enumerating lanes.
- `DiamondIntegerInvariant.lean` has independent scoped acceptance. It derives
  the sampled preimage cutoff from the same source-pool trapdoor, obtains the
  integer Gaussian error from the same generated target, and exploits selector
  sparsity for the `n * P` row bound. The bounded integer transition retains both
  error terms, `2 * n * P * E + m * n * N * K`.
- `DiamondBoundedLayerProof.lean` and `DiamondBoundedLoopProof.lean` directly
  compile with standard axioms and have independent scoped acceptance. They lift the
  actual initial row/error and compose the integer invariant through all lanes
  and symbolic `IterRuns`, choosing one integer sample family for the loop.
  The same bounded integer row now carries the message/bit second-coordinate
  invariant, using the actual selector scan and packed-family index. The loop
  theorem takes numeric recurrence equations and packing consistency. These are
  now discharged by `DiamondDecryptInjectorProof.lean` and the independently
  accepted `DiamondClaimInjectorProof.lean`, whose only premise is the actual
  linked `Runs`; raw-word contracts and terminal-state coverage are included.
- The joint `generated_injector_root` has independent scoped acceptance. One
  root destructure constructs the injector and final-public witnesses with an
  exact shared terminal pool address. It also links the actual exported witness
  preimages (slot 7) through scopes 74--78 to the same per-state terminal pool,
  matching public-input rows, negative gadget rows, and sampled cutoff.
- Boolean gate, circuit layer/iteration, requirement correspondence, initial
  encoding, and public-loop equality have independent scoped acceptance. Their
  actual Claim-level initial-array, accepting-output, and public-key composition
  also have independent scoped acceptance.
- Final public/preimage encoding, integer cancellation, exact decoder, and
  binary/capped affine recurrence bridges have independent scoped acceptance.
  `DiamondClaimStateProof.lean`, `DiamondClaimCircuitProof.lean`, and
  `DiamondClaimFinalProof.lean` have independent acceptance. The whole-polynomial
  theorem takes only the actual linked `Runs`, deriving all state/circuit/error
  witnesses and shared keys. `DiamondClaimCorrectnessProof.lean` connects its
  bound to the capped numeric gate and actual ideal/decoder endpoints.

The preferred two-digits-per-CRT-tower numeric probe has independent feasibility
acceptance: ring dimension 8, four 48-bit towers, base bits 24, total gadget
digits 8, inner dimension 20, error cutoff 29, preimage cutoff 85158441689,
one injector layer and two circuit layers. Its bound
3332445012031301517286688280911696390047037013499038720 is below radius
1569275433823053701793832124557973045606030056736041006492.
The same proof sources now compile at both backend geometries. The candidate's
complete 44-module chain compiles, and the official emitter produces
`DiamondCertificate.correctness : GeneratedClaim.CorrectnessClaim` with its
closed numeric gate fully discharged. Independent regeneration reproduced all
44 source files byte-for-byte and independently checked this theorem with only
`propext`, `Classical.choice`, and `Quot.sound`. This is the first accepted
concrete candidate theorem, not a runtime test, security estimate, general
topology theorem, or proof of successful-execution existence. The current
first accepted topology has one injector layer and two width-three circuit layers.

The compact numeric renderer and its caller now have independent acceptance.
The 45-module artifact includes kernel-checked numeral equations for each binary
exponent level, including zero, saturation, and large-count tests; a mutated
numeral is rejected. Equation count scales logarithmically in the loop counts
(numeral length also contributes to source size).

Parameter search now retains a private-field verified certificate with the exact
selected parameters and compiler. Numeric rejection is distinct from unsupported
topology, export, compiler, and timeout errors; there is no operational-checker
fallback. The fresh selected-candidate unit test passed in 159.92 seconds and its
45-module artifact and final theorem received independent acceptance. Its security
estimator is explicitly a test double: this validates correctness integration,
not a security estimate. The runner also has independent stale-source, timeout,
dependency, and fresh-compilation validation.

The same application proof now also has independent acceptance at depth four
and width five, with eight 48-bit CRT towers and two digits per tower. All 45
modules compile and the closed correctness theorem uses only the three standard
axioms. The circuit proof sources are byte-identical across these two circuit
sizes; all six generated graphs retain their declaration and line counts.

Actual two-input, one-bit-batch execution also has independent whole-theorem
acceptance: the same templates now use three states, nine bases, twelve
transitions, and four shared samples. Terminal coverage follows from the actual
state-family bound, not a fixed lane list. The two-bit-batch case has independent
acceptance as well. Independently checked 45-module artifacts also pass for two public input bits
and for a larger permitted radix (eight instead of four). The latter uses the
actual runtime condition `2^batch_bits <= digit_base` and preserves the same
packed-bit relation. The empty-public-input, width-one boundary also has independent
whole-theorem acceptance.

Essential negative fixtures now certify counterexamples to omitted target noise,
omitted right-hand BGG noise, and crossed same-shaped preimage/target relations.
The genuine sampler and nested/zero/large loop fixtures use the current exporter
and compile without warnings; their scoped independent review is complete.

## Final validation checkpoint

- All six Lean packages build without warnings.
- The current exporter and proof sources freshly generate and compile all 45
  candidate modules without warnings. The closed `DiamondCertificate.correctness`
  declaration uses only `propext`, `Classical.choice`, and `Quot.sound`.
- Both `cargo test -r --workspace --lib --no-run` and its `--features gpu` variant
  pass without warnings. These are compile-only checks, not GPU execution.
- Correctness production and test builds are warning-free; all 352 correctness
  unit tests pass. Focused WE tests pass 21 with four opt-in tests ignored;
  IR-core tests pass 52 and runtime layout tests pass five. The final generic
  unused-binder correction additionally passes all 16 focused exporter tests.
- Final independent semantic/design review explicitly accepts the completed
  checkpoint, including a fresh independent compile of the current certificate
  with only the three standard axioms. The theorem, scaling, search,
  negative-fixture, and warning-free validation scopes are accepted.

No integration tests or GPU execution were run. The selected-candidate test uses
a security-estimator test double, so these results do not establish a security
estimate or successful bounded-execution existence. No further arbitrary
parameter grid is planned.

Expanded durable provenance receipts and exhaustive mutation grids are deferred
by the user's priority decision. All warnings remain a required final gate. Existing fresh
kernel checking and the matching retained candidate artifact remain mandatory.

## Preservation and validation rules

The worktree contains shared, untracked Lean sources and `.lake` artifacts from
the ongoing implementation. They are intentional WIP and must be preserved;
do not clean them with broad deletion or reset commands. Staffing is parallel
GPT6 medium editors with disjoint proof-file ownership and one independent GPT6
medium reviewer. The injector editor alone owns shared emission/package changes.
Use narrow validation.
Run these reproduction commands from the repository root:

```sh
for package in primitives ir-core gadgets bgg runtime we; do
  (cd "crates/$package/lean" && lake build) || exit
done

cargo test -p mxx-we --lib selected_candidate_retains_a_kernel_checked_certificate -- --ignored --nocapture

cargo test -r --workspace --lib --no-run
cargo test -r --workspace --lib --features gpu --no-run
```

The certificate-retention unit test replaces the former example command; it uses a security
estimator test double and exercises real Lean generation/checking, not GPU execution or a real
security estimate. The historical checkpoint used the former example command. The last two commands compile
unit tests without running them; they are not integration or GPU execution tests.
