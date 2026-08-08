# Universal IR noise analysis and parameter checker plan

## 1. Status and objective

This plan deliberately separates two goals that were previously developed together:

1. derive conservative hard noise bounds and parameter checks mechanically from every executable
   IR graph; and
2. prove in Lean that the derivation and checker imply end-to-end protocol correctness.

This work completes the first goal. The second goal is a later task. Existing proof-oriented Lean
work is retained because it already provides useful typed identities, affine normalization,
family summaries, and sequential-recurrence machinery, but completion of the Diamond WE theorem
is not a prerequisite for this plan.

No separate noise-simulator crate is introduced. The analyzer and checker developed here are
intermediate, executable components of the final `mxx-correctness` design. The later correctness
task will prove soundness of these same definitions; it must not replace them with a parallel
implementation.

The result must be useful to parameter-search loops. Given a frozen graph, its input-bound
contracts, and a concrete parameter environment, the tooling must:

- analyze every current `mxx_ir_core::NodeKind` without an `UnsupportedNode` escape;
- derive a conservative hard bound or explicitly mark a value as not known to be small;
- derive all parameter-only side conditions from the graph rather than accepting user-supplied
  noise formulas;
- evaluate those conditions through one generic checker interface; and
- return both an acceptance result and an inspectable per-stage/per-wire report.

“All IR operations” means every variant and subvariant currently declared in
`crates/ir-core/src/node.rs`, including every constant-matrix form, sampler variant, loop input
mode, decode output form, and scalar operation. Future variants must fail an exhaustiveness test
until their transfer rule is added.

## 2. Non-goals

- Do not complete the generic analyzer soundness theorem, workflow erasure theorem, Diamond
  endpoint theorem, or application correctness theorem in this task.
- Do not create `mxx-noise-simulator` or any other standalone noise-analysis crate.
- Do not introduce a second Rust implementation of bound propagation. Lean remains the
  authoritative analyzer and arithmetic checker.
- Do not require protocol-specific certificates or handwritten per-node rule selections.
- Do not recover probabilistic or CLT-based estimates. Only deterministic hard bounds are used.
- Do not restore removed `ModUp`, `ModDown`, fold, unfold, disk, or compatibility APIs.
- Do not re-enable application modules that are intentionally disabled.

## 3. Checkpoint the existing work before changing direction

The current uncommitted Lean work must be preserved as a reviewable checkpoint before universal
node support begins.

### 3.1 Cleanup

1. Record the current branch, base commit, changed files, and untracked files.
2. Retain the implemented recurrence, family, basis-alignment, and carried-substitution work.
3. Remove temporary debugging declarations, abandoned experiments, stale comments, and unused
   imports only when their removal does not discard implemented behavior.
4. Do not delete a partially connected implementation merely because its final correctness
   theorem is incomplete.
5. If a partially implemented proof module prevents the stable checkpoint from compiling and a
   narrow repair is not practical, comment out only its import in place and add an English comment
   at that import site explaining that the source is retained for the later correctness task. Do
   not delete either the import line or the source, and do not replace it with an axiom, `sorry`,
   permissive fallback, or fabricated theorem.
6. Run `git diff --check`. Rust formatting, if any Rust source changed, must use
   `cargo +nightly fmt --all`.

### 3.2 Checkpoint validation

Run the narrowest stable checks that describe what the checkpoint actually claims:

```text
cd lean
lake build Mxx
```

Then run the relevant Rust compile-only checks for touched crates. Do not run integration tests in
this checkpoint stage. Scan active Lean modules for `sorry`, `admit`, and newly declared axioms.

If the full `Mxx` target cannot compile because an unfinished proof-only import is disabled under
section 3.1, build and record the exact smaller target that remains active. The checkpoint commit
message and `docs/diamond-we-correctness-status.md` must state that end-to-end correctness remains
incomplete.

### 3.3 Checkpoint commit

Create one English commit whose subject makes the boundary explicit, for example:

```text
Checkpoint incomplete generic correctness analysis work
```

The commit body must summarize retained recurrence/family work, list the validation commands, and
state that it does not establish the Diamond WE correctness theorem. Universal node support starts
only after this commit.

## 4. Authoritative architecture

### 4.1 One derivation implementation

The authoritative pipeline remains:

```text
Rust frozen Graph
  -> mechanical Rust-to-Lean IR emission
  -> Lean Phase A graph analysis
  -> derived bound expressions and obligations
  -> Lean Phase B exact parameter evaluation
  -> accepted/rejected report
```

Rust owns graph construction, freezing, hashing, process invocation, and report deserialization.
Rust must not duplicate matrix-bound, sampler-bound, recurrence, or decode-threshold formulas.
All Lean analysis/checking definitions and their Rust invocation support remain owned by
`mxx-correctness` and shared `lean/Mxx`.

### 4.2 Operational analysis precedes, but is not separate from, proof completion

The following definitions must remain executable without importing unfinished endpoint or
end-to-end correctness proofs:

- emitted IR syntax and graph validation;
- bound and scalar expression syntax;
- node transfer functions;
- subgraph/family/loop analysis;
- obligation generation;
- concrete parameter evaluation; and
- report serialization.

Existing theorem modules may prove facts about these definitions, but the operational checker must
not depend on completion of those proofs in order to compile or run.

This is a staging decision only. The final soundness theorem must quantify over the exact
`AnalysisResult` produced by this analyzer and the exact acceptance result produced by this
checker. No conversion into a second “proved” representation is planned.

### 4.3 Generic public interface

Replace the application-specific invocation boundary with a generic generated checker interface
owned by `mxx-correctness`. A generated protocol module supplies a closed graph and parameter
environment; the common executable entry point performs Phase A and Phase B and emits a stable
report. This is an interface to the in-progress correctness machinery, not a new simulator
subsystem.

The minimum result is:

```text
NoiseCheckReport
  schema identifier
  workflow hash
  accepted: Bool
  first failure, if any
  derived static obligations
  derived input-contract obligations
  resolved sequential recurrence summaries
  per-stage/per-scope/per-wire facts
  named-output bounds and classifications
```

Application parameter searches may wrap this generic interface, but must not own a separate
Diamond-specific bound evaluator. Infrastructure failure and a valid checker rejection remain
distinct results.

`accepted` is never conditional. It is `true` exactly when graph validation and Phase A succeed,
Phase B discharges every parameter-only obligation, and every input-dependent runtime-safety
obligation is entailed by a declared input contract or a mechanically analyzed protocol
precondition/requirement graph. Otherwise the checker returns `accepted = false`; an input
condition that cannot be established produces `UnprovenInputObligation`. The unresolved obligation
remains visible in the report so the protocol declaration can be strengthened and checked again.

## 5. Total conservative analysis domain

The analyzer currently rejects operations when it cannot preserve a precise affine
representation. Universal support instead uses an explicit conservative lattice.

### 5.1 Hard-bound domain

Every matrix-like value has one of:

- `Finite(BoundExpr)`: a deterministic maximum centered-coefficient bound;
- `NotSmall`: the value is valid but no useful small bound is justified; or
- `Invalid`: the graph or concrete parameters violate an executable precondition.

`NotSmall` is not an analysis error. It propagates monotonically and makes any requirement that
needs a small error fail. It may be displayed using the canonical centered residue cap when useful
for reporting, but it must never be reclassified as noise small enough for decoding.

This distinction handles, for example, a uniformly random full-modulus matrix multiplied by a
bounded error: the output is analyzable, but it is `NotSmall`, not a rejected node and not a
spuriously small product.

### 5.2 Matrix classification

Retain precise typed symbolic forms where supported:

- exact deterministic expression;
- affine signal terms plus a separate noise bound;
- noise-only bounded value; and
- whole-value `NotSmall`.

When an operation cannot soundly preserve the internal affine split, it may conservatively
materialize the whole output. It must not hide an existing noise term inside a newly declared exact
or signal term. A materialized result is `Finite` only when a hard whole-value bound follows from
the inputs and the actual runtime operation; otherwise it is `NotSmall`.

### 5.3 Other value kinds

- integers carry an exact runtime expression when available and a closed interval otherwise;
- reals carry an IEEE-754 binary64 runtime expression and domain obligations, but do not
  themselves have a cryptographic noise classification;
- booleans carry exact expression provenance when available;
- trapdoors carry their public-matrix identity and sampler parameters;
- indexed families carry an element summary and count; and
- bytes carry length and provenance only; and
- typed blobs carry their exact type name, schema hash, and input/artifact provenance, without
  inventing internal value semantics.

An operation that does not produce a matrix still must be analyzed because its value may control a
later matrix operation, family access, sampler, or decode check.

### 5.4 Inputs

External matrix and family inputs require an explicit input contract. A missing contract yields a
clear `MissingInputBound`/`MissingInputClassification` diagnostic, not a guessed norm. An input may
be intentionally declared `NotSmall`. Sampler outputs never obtain their bounds from an external
certificate; their serialized node cutoffs remain authoritative.

## 6. Required transfer rules for every current IR operation

The implementation must use the runtime evaluator and `ir-core` validation as the semantic source
of truth. The following table is an implementation checklist, not permission to omit variants.

### 6.1 Scalar and control operations

| IR operation | Required analysis behavior |
|---|---|
| `Input` | Load the exact typed input contract for every `WireType`, including bytes, trapdoors, families, and `TypedBlob { type_name, schema_hash }`. |
| `ConstantInt`, `EvaluateInt` | Exact integer expression and singleton interval. |
| `ConstantReal` | Exact binary64 bit-pattern result of the runtime `RealExpr::evaluate_f64` conversion, or the same conversion failure. Do not interpret it as an exact mathematical real. |
| `ConstantBool` | Exact Boolean expression. |
| `IntBinary::{Add,Subtract}` | Standard signed interval transfer. |
| `IntBinary::Multiply` | General signed interval using the minimum and maximum of all four endpoint products. |
| `IntBinary::{Divide,Remainder}` | Match runtime integer semantics; derive nonzero-divisor obligations and conservative signed intervals rather than rejecting signed inputs. |
| `IntCompare::{Equal,Less,LessEqual}` | Exact Boolean runtime expression for all three variants. |
| `BitExtract` | Exact Boolean expression plus a nonnegative-position and range-validity obligation when not structurally known. |
| `IntToReal`, `BoolToInt` | `IntToReal` records the runtime binary64 conversion and representability/finite-result condition; `BoolToInt` records the exact integer conversion. |
| `RealBinary::{Add,Subtract,Multiply,Divide}` | Record an IEEE-754 binary64 operation, including runtime rounding. Require a nonzero divisor for division and a finite output for every operation. |
| `RealSqrt` | Record binary64 `sqrt` and require only the runtime's `value < 0.0` rejection condition. In particular, runtime accepts `sqrt(+infinity) = +infinity`; do not add a finite-output condition unless runtime is changed separately. |
| `Select` | Require a valid index; use the selected exact expression where statically known and branch-wise interval/max-bound summaries otherwise. Empty selections are invalid. |

Real facts must not use mathematical-real equality: `mxx-runtime` evaluates these nodes as IEEE
`f64`. Add a `RuntimeRealExpr` whose values are binary64 bit patterns and whose evaluator matches
the runtime conversion, round-to-nearest operations, signed-zero comparison used by division,
the operation-specific finiteness checks, and square root. Do not impose one global finiteness
rule: `ConstantReal`, `IntToReal`, and every `RealBinary` output are finite-checked by runtime,
whereas `RealSqrt` rejects only a negative input. Cross-language fixtures compare the Lean result
bits and failure cases with Rust. Real arithmetic need not derive sampler hard bounds: Gaussian and preimage
nodes already serialize authoritative integer cutoffs. Consequently an irrational square root is
never converted into a floating-point noise formula; the real layer is needed only to analyze the
operation and its actual runtime domain conditions.

### 6.2 Constants, samplers, and relations

| IR operation | Required analysis behavior |
|---|---|
| Every `ConstantMatrix` form | Emit and analyze `Zero`, `Identity`, `UnitRow`, `UnitColumn`, both gadget forms, `PowerOfBase`, `Rotation`, and `Polynomial`. Derive exact identity and a hard centered bound from the actual constructor. |
| `GadgetTrapdoor` | Produce a typed trapdoor fact, its exact public gadget identity, and provenance recording that preimage evaluation is deterministic gadget decomposition rather than sampled preimage generation. |
| `TrapdoorPublic` | Recover the public matrix identity from the trapdoor input. |
| `UniformResidueSample` | A distinct executable node for matrices sampled uniformly from the full coefficient residue ring `R_q`. Its source role is statically `NotSmall`, independently of the concrete modulus; its centered hard cap is `q/2`. |
| `UniformIntervalSample` | A distinct executable node for small interval samples. The current backend accepts only `[-1,1]` and `[0,1]`; both are statically bounded by one. Every other interval is a validation error. Full-residue sampling must use `UniformResidueSample`, never a numeric `[0,q-1]` interval. |
| `GaussianSample` | Use exactly `max_coefficient_bound` serialized in the node. |
| every `HashSample` variant | Plain hash output is exact/public and normally `NotSmall`. Make decomposed variants self-contained: runtime and validation must enforce one layout from the node's resolved `base`, `digit_count`, variant, and output shape. Runtime must use that enforced layout rather than silently deriving a different backend-default layout. The analyzer records exactly that relation and digit bound. Missing or inconsistent decomposed metadata is invalid; Lean must not infer a second layout. |
| `TrapdoorSample` | Produce both public and private facts and use the serialized preimage cutoff. |
| `PreimageSample` | First align scalar and batched runtime paths: both must evaluate `max_coefficient_bound` and enforce its nonnegative constraint before branching on trapdoor provenance. The current batched gadget branch skips evaluation and must be corrected. For a sampled trapdoor, use the evaluated serialized cutoff as the output bound. For `GadgetTrapdoor`, evaluate the expression for identical runtime-domain behavior but ignore its value for the output, return `gadget_decompose(target, small)`, and derive the canonical digit bound from trapdoor base/layout/`small` provenance. In both cases retain only the correct modular relation `B * K = P` in `R_q`; never strengthen it to an integer equality. |
| `GadgetDecompose` | Support explicit and implicit digit counts after resolving the latter from validated types/parameters. Preserve deterministic canonical-residue decomposition identity and the hard digit bound for both `small` values. |

### 6.3 Matrix arithmetic and structural transforms

| IR operation | Required analysis behavior |
|---|---|
| `MatrixBinary::Add/Subtract` | Combine exact/affine forms when possible; hard bounds add. |
| `MatrixBinary::Multiply` | Support every typed operand classification. Preserve affine signal/noise terms when at most one side is signal-bearing; otherwise conservatively materialize the whole product. The hard product bound uses the actual ring-dimension and inner-dimension factors. `NotSmall` propagates. Apply preimage/decomposition rewrites before the fallback. |
| `MatrixNegate` | Negate exact/affine expressions and preserve hard bounds. |
| `MatrixScale` | Support every scalar expression with an absolute-value bound; retain relations only when the transformation of that relation is explicitly valid. |
| `Transpose` | Preserve coefficient bound and construct a typed transpose expression. |
| `Slice` | Preserve the maximum coefficient bound and build a typed slice expression; drop only relations that slicing cannot transport. |
| `Tensor` | Follow the runtime tensor definition exactly. Derive the corresponding hard product bound, including all required dimension factors, and conservatively materialize signal structure when necessary. |
| `Concat::{Rows,Columns,Diagonal}` | Support all axes. The hard maximum coefficient bound is the maximum of input bounds. Preserve a typed concat expression instead of hiding signal/noise; if precise affine distribution is unavailable, retain the whole-value classification conservatively. |
| `Reshape` | Preserve coefficient order and hard maximum bound; retain a typed reshape expression. Relations that are not reshape-invariant are dropped. |

### 6.4 Coefficient, decode, and packing operations

| IR operation | Required analysis behavior |
|---|---|
| `ExtractCoefficient` | Match canonical modular coefficient extraction and derive its exact integer interval. |
| `ConstantCoefficient` | Support the constant-polynomial projection used by runtime and derive the output matrix/scalar bound from the selected coefficient. |
| `ThresholdDecode` | Support Boolean and non-Boolean outputs. Derive decode obligations from the input noise classification and the actual modulus/plaintext/length expressions. If the input is `NotSmall`, return a normal checker rejection rather than an unsupported-node error. |
| `CrtRecompose` | Match the runtime coefficient-wise operation exactly. For canonical input coefficient `v` and full modulus `q`, first compute `r_i = ((p_i * v + floor(q/2)) / q) mod p_i` using runtime nonnegative integer division, then multiply `r_i` by the reconstruction coefficient modulo `q`, sum the levels, and reduce modulo `q`. Preserve a finer bound only when this nonlinear rounded expression proves one; otherwise return a valid whole-value `NotSmall` result bounded by the canonical centered residue cap. |
| `PackPolynomialCoefficients` | Model canonical little-endian bit packing and validate the exact family length. For every coefficient chunk, derive an input-contract obligation that the reconstructed unsigned value is `< modulus`, matching runtime rejection. It must be entailed by a declared contract or analyzed protocol precondition; otherwise the checker deterministically rejects with `UnprovenInputObligation`. Under a discharged obligation the output is exact with its actual centered bound. |

### 6.5 Structural graph operations

| IR operation | Required analysis behavior |
|---|---|
| `SubgraphCall` | Analyze the frozen child definition using facts transported from the actual arguments and transport every output fact back to the caller. Memoize by frozen definition plus input-fact schema; do not introduce protocol-specific node searches. |
| `ParallelLoop` | Analyze the body once as a typed family template. Support `Broadcast`, `Zip`, and `ZipOffset`, with exact range obligations for offsets. |
| `SequentialLoop` | Use the existing typed simultaneous recurrence transition. Phase A keeps a compact symbolic recurrence; Phase B iterates only numeric bound state for the concrete count. Zero count and multiple carried outputs are required. |
| `FamilyPack` | Construct a typed family summary from every input, validating homogeneous element types and declared count. |
| `FamilyGetStatic`, `FamilyGetDynamic` | Instantiate the family element summary and derive static/dynamic index range obligations. Arbitrary valid dynamic expressions must not be rejected merely because they are not a loop offset. |

## 7. Emitter completeness

Extend `crates/correctness/src/emit_lean.rs` and `lean/Mxx/Ir.lean` together.

1. Every Rust `NodeKind` and every nested enum variant must have a Lean representation.
2. Remove the wildcard `UnsupportedNode` fallback from normal emission. Use exhaustive Rust matches
   so adding a Rust variant causes a compile error.
3. Preserve all operation parameters needed by analysis. Do not erase `small`, implicit digit-count
   state, real expressions, decode output mode, CRT coefficients, or packing widths.
4. Add a mechanical inventory test that constructs one minimally valid node for every variant,
   emits it, and verifies that the Lean parser/build accepts the generated program.
5. Keep source/workflow hashes sensitive to the expanded emitter and generated IR.

## 8. Analyzer and checker implementation stages

### Stage A: inventory and total result types

- Generate a checked-in inventory mapping every Rust node subvariant to its Lean constructor and
  transfer-rule function.
- Add `Finite`/`NotSmall` propagation to matrix facts and bound expressions.
- Add stable diagnostics that distinguish invalid graph, missing input contract, domain failure,
  unproved input obligation, checker rejection, and infrastructure failure.
- Make analyzer pattern matches exhaustive before implementing individual rules.

### Stage B: leaf and scalar completeness

- Complete emitter support for real/scalar nodes and every constant-matrix form.
- Implement signed integer interval arithmetic, all comparisons, conversions, binary64 real
  provenance, runtime-equivalent real evaluation, and domain obligations.
- Complete all sampler/hash/trapdoor/decomposition leaves.
- Make `ir-core` and runtime hash/uniform validation agree with the exact layouts and distributions
  consumed by analysis; do not preserve graph-valid/runtime-invalid legacy cases.

### Stage C: matrix-operation completeness

- Complete add, subtract, multiply fallback, negate, arbitrary scale, transpose, slice, tensor,
  concat, and reshape.
- Compare every formula against the corresponding runtime executor branch.
- Add tests demonstrating that unsupported affine combinations become conservative results rather
  than errors.

### Stage D: coefficient/decode/packing completeness

- Implement both threshold-decode output modes, coefficient operations, CRT recomposition, and
  polynomial bit packing.
- Add accepted and rejected decode fixtures with hand-checkable hard bounds.

### Stage E: graph structure completeness

- Implement direct subgraph-call transfer.
- Complete family pack/get and all parallel-loop input modes.
- Reuse and finish the current sequential symbolic-bound recurrence path for arbitrary supported
  carried facts.
- Ensure recursive analysis terminates using frozen graph structure and explicit analysis fuel,
  not node-name heuristics.

### Stage F: generic checker and reports

- Expose one generated generic checker entry point from `mxx-correctness`; do not add a crate.
- Serialize the full `NoiseCheckReport` with stable schema and workflow hash.
- Change Diamond parameter search to a thin consumer of the generic result.
- Add a reusable Rust helper for other application crates to invoke the generated checker without
  reimplementing argument ordering or output parsing.

### Stage G: documentation and cleanup

- Update `docs/architecture.md` to describe the generic graph-derived noise checker rather than a
  Diamond-specific checker.
- Update `docs/diamond-we-correctness-status.md` to separate operational bound checking from the
  deferred theorem.
- Regenerate the M0 audit from the new report and remove obsolete unsupported-node allowlists.
- Delete no retained proof work unless it is demonstrably superseded; inactive proof modules remain
  clearly marked for the later correctness task.

## 9. Testing strategy

### 9.1 Exhaustive node coverage

Maintain a table-driven test with one case for every `NodeKind` and nested operation variant. The
test fails if the Rust enum grows without a corresponding emitter and analyzer case. Each case must
verify:

- Rust graph validation;
- Lean emission;
- Phase-A analysis success for valid contracts;
- Phase-B deterministic result; and
- expected finite/`NotSmall` classification.

### 9.2 Formula tests

For every operation, use small integer parameters and write the expected hard-bound calculation in
an English test comment. Test exact boundary acceptance and one-unit-over rejection where the
operation produces a decode obligation.

Required regression cases include:

- byte and typed-blob inputs preserving length/type/schema provenance;
- signed integer multiplication and division;
- binary64 signed zero, overflow/non-finite rejection, negative square root rejection, and accepted
  positive-infinity square root;
- both small supported uniform ranges, the supported full-modulus `NotSmall` range, and rejection of
  an unimplemented range;
- arbitrary matrix scale;
- signal-bearing multiply on either side and signal-bearing on both sides;
- transpose, tensor, all concat axes, and reshape;
- all constant-matrix forms;
- plain/decomposed/small-decomposed hashes;
- rejection of decomposed-hash metadata that disagrees with its runtime layout;
- explicit and implicit gadget digit counts;
- Boolean and non-Boolean threshold decode;
- CRT recomposition and polynomial bit packing;
- CRT rounding cases near half-modulus boundaries and negative/positive reconstruction residues;
- packed coefficients equal to `modulus - 1` and the rejected value `modulus`;
- subgraph calls with multiple outputs;
- `Broadcast`, `Zip`, and `ZipOffset` parallel inputs;
- zero-count and multi-carried sequential loops;
- static and general dynamic family access; and
- missing input metadata producing a specific diagnostic.

### 9.3 Runtime correspondence

For representative small graphs, execute the same graph through `mxx-runtime` and confirm that
every observed matrix coefficient is within the checker-derived hard bound. These are targeted unit
tests, not protocol integration tests. Expected runtime values must come from existing trusted
runtime primitives, not a hand-written duplicate evaluator.

### 9.4 Existing protocol gates

- Toy checker remains accepted.
- The generated Diamond graph completes Phase A without unsupported nodes.
- Valid existing Diamond search fixtures receive the same or more conservative result.
- Previously invalid fixtures remain rejected.
- No test may loosen a cutoff or replace `NotSmall` with a guessed finite noise bound merely to
  obtain acceptance.

## 10. Completion criteria

This task is complete only when all of the following hold:

1. The checkpoint commit described in section 3 exists before the universal-support commits.
2. Every current Rust IR operation and nested variant is emitted to Lean.
3. Every valid emitted node produces an analysis fact, family summary, trapdoor fact, or explicit
   `NotSmall` result; no valid operation returns `UnsupportedNode`.
4. Invalid arity, type, range, domain, or missing-contract cases still fail closed with specific
   diagnostics.
5. Phase B evaluates every parameter-only bound and static obligation produced by Phase A; every
   input-dependent obligation is discharged from declared contracts/analyzed requirements or causes
   deterministic rejection.
6. A generic checker report is callable from Rust parameter-search code.
7. Diamond uses the generic checker rather than a separate noise formula implementation.
8. Exhaustive node coverage and focused runtime-correspondence tests pass.
9. Active Lean code contains no `sorry`, `admit`, or new axiom.
10. Documentation states clearly that checker availability is achieved but the proof that checker
    acceptance implies protocol correctness is deferred.

Completion of the Diamond WE correctness theorem is explicitly not a criterion for this task.
The completed checker nevertheless remains part of the theorem's eventual trusted derivation path,
not an independent simulator that will later be discarded.
