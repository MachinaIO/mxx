# Indexed Lean Operational Checker: Specification and Implementation Handoff

## Status

This document is the implementation contract for replacing the current operation-specific family
and selection machinery in the Lean operational correctness checker with one indexed semantics.
It is written for a new contributor who has no prior knowledge of the migration.

The implementation described here is **not complete**. The current worktree contains a compiling,
auditable intermediate compact-selection engine. It must be preserved as a checkpoint and test
oracle only while the indexed replacement is built. It is not the target architecture.

The target deliberately has no backward-compatibility requirement. Do not retain deprecated Graph
IR nodes, old binary readers, aliases, dual operational checkers, or protocol-specific fallbacks.

## The Problem in One Example

Suppose an executable graph computes

```text
c = s * B + e
```

and a preimage sampler records the exact relation

```text
B * K = S' * P + E  in R_q.
```

The checker must derive

```text
c * K
  = s * S' * P + s * E + e * K.
```

The term `s * S' * P` is the expected signal. The terms `s * E` and `e * K` are
bounded noise. The checker must derive this from the executable Graph IR and the sampler relation,
not from a handwritten protocol recurrence.

The same rule must work when `K` is a family element, is dynamically selected, is gathered through
an index map, or appears in a compatible block product such as

```text
[B, I] * [K; 0] = B * K.
```

## Repository Ownership

There is one executable graph: the Graph IR owned by `mxx-ir-core`.

- `mxx-ir-core` owns executable nodes, wire and matrix types, parameter expressions, validation,
  artifact metadata, and Graph IR versions.
- `mxx-dsl` constructs immutable Graph IR nodes. It does not own a second symbolic graph.
- `mxx-runtime` executes validated Graph IR on CPU or GPU.
- `mxx-correctness` transports closed workflows and derivations to Lean, prepares operational
  modules, runs requests, and validates reports.
- `lean/Mxx/Certificate/OperationalBounds.lean` owns graph-derived operational expressions,
  relations, deterministic bounds, residual checks, and decoder acceptance.

Do not recreate `mxx-ir-symbolic`, `mxx-noise-simulator`, or a Rust-side graph-composed noise
checker. Runtime-enforced primitive sampler cutoffs, security estimation, performance estimation,
measured residual diagnostics, and non-accepting telemetry may remain in Rust.

## Terms Used by the Target Design

### Operational factor roles

A `Bounded` factor has a finite deterministic bound justified by a runtime contract. A `Large`
factor cannot be assumed small. A product containing any `Large` factor is a signal term. A product
whose factors are all `Bounded` is a noise term.

The residual is

```text
actual output - expected signal.
```

Acceptance first proves that no Large term remains in the residual. Only then does it evaluate the
noise bound.

### Families and selections

A family is a finite function from an index to a value:

```text
K : {0, ..., count - 1} -> Value.
```

A selection is function application at a runtime index:

```text
K(alpha).
```

It is not expanded into a one-hot indicator sum. In particular, the checker must not replace it by

```text
sum_i indicator(alpha = i) * K(i).
```

Such an expansion creates branch products before exclusivity is recovered and leads to quadratic
or Cartesian growth.

### Indexed facts

Every operational value is an `IndexedFact` with:

```text
IndexedFact {
    context,
    payload,
    storage,
}
```

The context contains the free family, loop, and runtime-selection indices. The storage tag is only
a performance representation:

```text
shared template
mapped template
explicit table
```

Storage must not change operation semantics.

A single value has an empty context. A family adds a lane index. A dynamic selection substitutes a
runtime selection variable for the family binder. Values selected by the same selector share the
same index variable; different selectors retain different variables.

### Reindexing

Reindexing substitutes index expressions throughout one indexed fact. The same substitution must
reach value identities, public identities, relation owners, relation targets, and provenance.

```text
static get:     K(i)[i := 3]
dynamic get:    K(i)[i := alpha]
zip offset:     K(i)[i := lane + offset]
gather:         K(i)[i := source_index(lane)]
broadcast A:    lane -> A
```

Nested reindexing composes maps instead of materializing intermediate families.

### Pointwise operations

All primitive operations are pointwise over the merged index context:

```text
op(E1, ..., En)(assignment)
  = op(E1(assignment), ..., En(assignment)).
```

Single matrices, families, and selections do not get separate matrix-addition, multiplication,
concat, transform, or relation rules.

### Exact identity and provenance

Cancellation and relation rewriting compare complete identities, not names or possible numerical
equality. Identities include the workflow stage, scope, wire, artifact producer origin, family or
loop index, runtime selection, deterministic hash query, matrix type, and relevant parameters.

Concrete runtime `ProductionId` contains an execution nonce and is not Lean cancellation identity.
Artifact bindings preserve their producer origin across stages.

## Matrix Expression Semantics

For one fixed index assignment, a matrix is represented as a flat integer-coefficient sum of
ordered products:

```text
Term       = integer coefficient * ordered factor list
Polynomial = finite sum of Term.
```

Factor order is never changed. Exact duplicate products are merged; exact opposites cancel.

Normalize locally after every operation:

1. flatten nested explicit sums;
2. flatten products without reordering factors;
3. canonicalize signed integer coefficients;
4. merge exactly identical products;
5. remove zero terms;
6. apply exact registered relations;
7. merge newly equal products;
8. compress consecutive unprotected bounded factors;
9. summarize bounded-only terms;
10. retain signal terms with their ordered factors.

Explicit `Add` nodes may distribute over products. Families and selections must not be expanded as
indicator sums.

Exact identity multiplication may normalize

```text
I * A = A
A * I = A
```

only when shape, parameter set, and exact identity are validated. This normalization occurs before
hard-bound evaluation so expression normalization and bound evaluation use the same result.

## Relations

The primary generic relation is

```text
B(index) * K(index) = T(index)  in R_q.
```

Rewrite only adjacent factors that match the recorded public identity, preimage owner identity,
index expression, matrix type, modulus, ring dimension, and layout.

The same implementation handles:

```text
B * K(i) = T(i)
B(i) * K(i) = T(i)
B(source(i)) * K(i) = T(target(i))
B(alpha) * K(alpha) = T(alpha).
```

It must not rewrite `B(alpha) * K(beta)` when `alpha` and `beta` are different selection
identities.

The target may be a polynomial such as `S' * P + E`. Splice that relation-free target snapshot
into the surrounding product and normalize. Do not recursively attach a relation inventory to the
snapshot.

Do not infer protocol motifs such as `B = A - xG`. The executable graph or sampler relation must
state every equality used by the checker.

## Block Matrix Semantics

The executable node remains `Concat { axis }`. Do not add executable `BlockRows` or
`BlockColumns` nodes merely for analysis.

Pairwise contraction applies only to a horizontal block row on the left and a vertical block
column on the right with matching inner partitions:

```text
BlockColumns([L1, ..., Ln]) * BlockRows([R1, ..., Rn])
  = sum_i Li * Ri.
```

The counts and every inner block boundary must match. This generic contraction is implemented in
one place and runs after the index assignment is fixed.

The reverse orientation is not a sum:

```text
BlockRows([L1, ..., Ln]) * BlockColumns([R1, ..., Rm])
```

is the block matrix with `(i, j)` block `Li * Rj`. Preserve it compactly or fail closed if the
active protocols do not require it. Never turn it into a pairwise sum.

Diagonal concat retains both row and column partitions. It is not implicitly treated as the
horizontal/vertical contraction above.

After complementary contraction, ordinary normalization and relation rewriting handle

```text
[B, I] * [K; 0]
  -> B * K + I * 0
  -> B * K
  -> T.
```

## Coefficient and Shape Operations

Remove the general executable `Reshape` node. Producers must emit the intended shape directly by
typed operations such as slice, concat, transpose, tensor, decomposition, or the original matrix
producer.

Remove the combined `ConstantCoefficient` node. Use:

```text
ExtractCoefficient(matrix, position) -> Int
LiftIntegerToConstantPolynomial(integer, scalar_matrix_type) -> Mat.
```

Extraction returns the canonical nonnegative residue in `[0, q)`. Lifting writes `integer mod q`
to the constant coefficient and zero to every other coefficient. The lift output is marked
constant-polynomial and retains the integer's bound and canonical-range information.

The runtime may fuse the two operations physically, but Graph IR and Lean semantics keep them
separate.

## Deterministic Hard Bounds

Operational acceptance uses Lean exact integers and runtime-enforced sampler cutoffs. It does not
use CLT, independence heuristics, square-root concentration, union bounds, or floating-point tail
estimates.

For bounded matrix multiplication:

```text
effective_inner
  = left.columns - right.known_zero_rows.getD(0)

ring_factor
  = 1                 if either input is constant-polynomial
  = ring_dimension    otherwise

product_bound
  = effective_inner * ring_factor * left_bound * right_bound.
```

Sampler rules:

- Gaussian output uses the nonnegative integer cutoff stored on the node and enforced by the CPU
  and GPU runtime.
- A trapdoor's public matrix is Large. Trapdoor metadata does not make the public matrix bounded.
- Preimage output uses the `PreimageSample` cutoff. If trapdoor metadata carries a corresponding
  cutoff, Lean requires both expressions to evaluate to the same integer.
- Gadget decomposition uses the existing deterministic digit bound and validated layout.

For selections, fully normalize, consume relations, cancel exact terms, classify signal and noise,
and evaluate each complete branch bound before taking a maximum. Do not combine maxima from partial
terms belonging to different branches.

Independent irregular selections require a compact transfer preserving polynomial structure,
factor roles, identity, metadata, and relation inventory. If no such transfer is available, fail
closed without Cartesian enumeration.

## Decoder Target

The closed protocol, not the parameter-search caller, owns the operational decoder target:

```text
OperationalDecoderTarget {
    target_id,
    residual_stage,
    residual_output,
    decoder_stage,
    decoder_node,
    plaintext_modulus : IntExpr,
}
```

The target's plaintext modulus expression must exactly match the executable threshold decoder.
The request supplies `target_id`, a parameter environment, and gadget layouts. Lean evaluates `p`
from the target expression. It derives and cross-checks `q` from the decoder input and residual
matrix types.

Acceptance requires

```text
p > 1
q > 0
N >= 0
2 * p * N < q.
```

Use exact multiplication, not integer-division rearrangements. Equality at the boundary is a
failure. A residual containing any Large term is a failure.

## Current Auditable Checkpoint

At the checkpoint described by this document:

- the branch is `codex/tall-bgg-encodings`;
- the checkpoint implementation is centered in `lean/Mxx/Certificate/OperationalBounds.lean`;
- generated Toy and Diamond modules contain refreshed source/toolkit hashes;
- `lake build Mxx.Certificate.OperationalBounds` succeeds;
- `lake build Mxx` and `lake build MxxWe.DiamondChecker` succeed;
- `cargo build -p mxx-correctness` and `cargo build -p mxx-we` succeed;
- the build intentionally prints generic fail-closed fixture diagnostics through `dbg_trace`;
- checker-evaluation fixtures use explicitly visible `native_decide`, as permitted by repository
  policy; this enlarges the trusted base and must remain reported;
- compact `Exact` and `Shared` selections, schema interning, lazy bound/relation/schema memos, and
  zero-Cartesian counters are implemented;
- the implementation still has `matrix`, `matrixExpr`, `familyUniform`, `familyPacked`, and many
  selected-specific mechanisms;
- `Reshape` and `ConstantCoefficient` still exist throughout Graph IR and runtime;
- the operational request still accepts residual names and concrete `p` and `q`;
- the new indexed semantics and protocol-owned decoder target are not implemented;
- no Tall GPU end-to-end claim follows from the local Lean build.

This checkpoint is useful evidence and a source of fixtures. Do not continue adding special cases
to its selection machinery. Replace it in the ordered stages below.

## Ordered Implementation Stages

Do not reorder these stages. Restore a compiling, reviewable state at the end of each stage.

### Stage 0: preserve baseline fixtures

Record current successful and fail-closed fixtures for single matrices, compact selections,
relations, memo hits, branch visits, and Cartesian visits. New constructors that do not exist yet
must be tested in the stage that introduces them; do not add non-compiling placeholders, `sorry`,
`admit`, or permanently skipped tests.

### Stage 1: remove `Reshape`

Replace active call sites first:

```text
crates/we/src/diamond/graph.rs
crates/bgg/src/noise_refresh.rs
```

If the existing producer already has the requested shape, remove the call. Otherwise make the
producer emit the intended shape directly. Then delete `Reshape` from:

```text
crates/ir-core/src/node.rs
crates/ir-core/src/constraints.rs
crates/ir-core/src/validate.rs
crates/dsl/src/lib.rs
crates/runtime/src/backend.rs
crates/runtime/src/backend/poly.rs
crates/runtime/src/executor.rs
crates/bench-estimator/src/gpu.rs
crates/we/src/diamond/estimate_gpu.rs
crates/correctness/src/ir_binary.rs
crates/correctness/src/emit_lean.rs
lean/Mxx/Ir.lean
lean/Mxx/Ir/BinaryFormat.lean
lean/Mxx/Certificate/OperationalBounds.lean
```

No alias or old reader remains.

### Stage 2: split coefficient extraction and lift

Delete `ConstantCoefficient`. Add one executable node with one integer argument and one scalar
matrix output:

```text
LiftIntegerToConstantPolynomial { matrix_type : MatrixType }
```

Validate one integer argument, a `1 x 1` output, positive modulus and ring dimension, and exact
agreement between the declared matrix type and output wire type.

Replace the active calls in `crates/bgg/src/slot_operation.rs` with explicit extraction followed by
lift. Runtime execution may construct an identity matrix and scale it by the integer. Update both
GPU estimators to measure the same production dataflow.

Update Rust and Lean node syntax, validation, execution, encoding, decoding, operational transfer,
and round-trip tests atomically.

### Stage 3: synchronize versions

The node-surface change requires coordinated updates to:

```text
crates/ir-core/src/encoding.rs                    IR_VERSION
crates/correctness/src/ir_binary.rs              IR_BINARY_FORMAT_VERSION
crates/correctness/src/freshness.rs              GENERATOR_VERSION
crates/correctness/src/operational_runner.rs     prepared-cache version
```

Bump the report schema when its fields change. Old programs, derivations, manifests, artifacts,
unknown tags, truncated inputs, and stale prepared modules must be rejected.

### Stage 4: add protocol-owned decoder targets

Add the Rust declaration to `crates/correctness/src/bundle.rs`, validate it with the closed bundle,
and transport it through:

```text
crates/correctness/src/emit_bundle_lean.rs
lean/Mxx/Certificate/ProtocolSyntax.lean
lean/Mxx/Certificate/Workflow.lean
```

Update Toy and Diamond declarations. Replace request fields `residual_stage`, `residual_output`,
`plaintext_modulus`, and `ciphertext_modulus` with `target_id`. Lean evaluates `p` and derives `q`.
Rust validates the target ID, request digest, freshness hashes, and modulus echo in the report.

### Stage 5: add the minimal indexed core

Introduce index variables, index expressions, contexts, maps, and `IndexedFact`. Implement only:

```text
emptyContext
extendContext
validateContext
reindex
composeIndexMap
sameIndexExpression
indexExpressionInBounds
```

Reindex every identity, relation, target, and provenance field. Test static, dynamic, offset,
gather, same-selector, different-selector, artifact-origin, and out-of-range cases.

### Stage 6: unify families and selections

Implement one `liftPointwise` helper that merges compatible index contexts and applies an existing
primitive transfer at a fixed assignment. Route matrix add, subtract, multiply, negate, scale,
transpose, slice, tensor, concat, and coefficient lift through it.

Construct Graph IR family operations as follows:

```text
FamilyPack           add a lane binder and store the ordered table once
ParallelLoop output  add a loop binder and store one shared or mapped body
FamilyGetStatic      reindex binder to a constant
FamilyGetDynamic     reindex binder to the selector variable
Select               build a family, then use the same dynamic reindex
Broadcast            use an empty-context value as a constant function
```

As each primitive moves, delete its old selected/family branches. At stage completion there is no
semantic `familyUniform`/`familyPacked` split. Arena IDs and storage tags may remain for
memoization, but cannot select different semantics.

### Stage 7: centralize complementary block contraction

Keep executable `Concat`. Preserve block boundaries in the operational polynomial and implement
one `contractComplementaryBlocks` function. Apply it only for matching
`BlockColumns * BlockRows`. Test the reverse orientation, mismatched counts, mismatched boundaries,
single values, families, and selections.

Run normal relation rewriting after contraction so `[B,I] * [K;0]` exposes and consumes `B*K`.

### Stage 8: unify indexed relation rewriting

Implement one adjacent-factor rewrite over exact indexed identities. Use it for common public
matrices, lane-dependent public matrices, gather mappings, and selections. Delete protocol-specific
or selection-specific preimage paths. Preserve relation-free target snapshots and normalize after
splicing the target.

### Stage 9: connect deterministic bounds

Run signal/noise classification after normalization and relation consumption. Preserve or
recompute `isConstantPolynomial`, `knownZeroRows`, canonical ranges, and hard bounds. Fail closed
when an operation invalidates required metadata. Take selection maxima only over complete branch
bounds.

### Stage 10: connect loops

Treat `Broadcast`, `Zip`, and `ZipOffset` as index maps. Analyze a parallel body once with a free
index. Represent a sequential body as a fixed-size numeric transition with simultaneous carried
slot updates. Reject unresolved relation-bearing carried state unless a generic substitution rule
exists.

### Stage 11: update diagnostics, report, and cache

Report the target ID, derived moduli, noise bound, freshness hashes, request digest, phase timings,
expression and memo statistics, logical/stored branch counts, relation rewrites, Cartesian visits,
and maximum polynomial terms. Diagnostics identify stage, scope, node, wire, context, operation,
relation owner, expected and actual identity, and the failure reason.

Generic debug logging may remain only if it is filterable and does not hard-code a protocol or node
number. Remove temporary unfiltered `dbg_trace` output before declaring the migration complete.

The current operational fixture suite uses `native_decide` for checker-evaluation facts. This is
permitted by the repository policy but enlarges the trusted base. Keep every use explicit, report
it during theorem validation, and do not use it to replace a mathematical proof obligation.

### Stage 12: regenerate checked-in Lean modules

Never hand-edit files below `Generated/`. Regenerate with:

```text
MXX_REGENERATE_CORRECTNESS=1 cargo run -p mxx-correctness --example emit_correctness
MXX_REGENERATE_CORRECTNESS=1 cargo run -p mxx-we --example emit_correctness
```

Confirm refreshed protocol, workflow, derivation, toolkit, version, and decoder-target data.

### Stage 13: remove obsolete paths

Search active code for and remove:

```text
Reshape
ConstantCoefficient
familyUniform
familyPacked
operation-specific selected transfer
joinDynamicFacts
legacy or deprecated aliases
old binary readers
protocol-specific operational fallbacks
temporary old-vs-new test oracles
```

Request-local arenas, interning, and memoization may remain only as analysis storage.

## Validation Order

Run the narrowest checks first. Do not start with integration tests.

```text
cargo test -p mxx-ir-core --lib
cargo test -p mxx-dsl --lib
cargo test -p mxx-runtime --lib
cargo test -p mxx-correctness --lib

cd lean
lake build Mxx.Ir.BinaryFormat
lake build Mxx.Certificate.OperationalBounds

cargo build -p mxx-correctness
cargo test -p mxx-bgg -p mxx-gadgets --lib
cargo test -p mxx-we --lib

cd lean
lake build MxxWe.DiamondChecker

cargo test --workspace --lib --no-run
cargo test --workspace --lib --features gpu --no-run
```

Check actual Lake target names before invoking them. Use `cargo +nightly fmt --all` for Rust.
Do not run GPU integration tests, long Tall tests, or remote jobs unless the user explicitly asks
for them in that task.

## Required Complexity Properties

- Shared template storage is independent of logical family size.
- Exact-table processing is linear in stored branch count.
- Same-domain Shared/Shared processing is independent of logical branch count.
- Exact/Shared processing visits only stored Exact branches.
- Index-domain comparison is constant time after collision-safe interning.
- A repeated memoized query is constant time after its first evaluation.
- No independent-domain Cartesian array is created or visited.
- Hashes select interning buckets only; a full canonical key comparison resolves collisions.

Test counts 2, 1,024, and 30,720 for shared templates. Test representative small explicit tables
and compare compact results with explicitly unrolled reference graphs.

## Error Investigation Procedure

When a new unsupported error appears, do not add a special case immediately. Check in this order:

1. Graph IR input and output types;
2. the index context;
3. complete reindexing of identity and relation data;
4. whether an old selected/family path bypassed `liftPointwise`;
5. block orientation and boundary compatibility;
6. exact public and preimage identity agreement;
7. normalization after relation rewriting;
8. metadata preservation, recomputation, or invalidation;
9. whether the case is truly outside the specification or is a missing generic rule.

If it is a missing generic rule, fix the one common rule and add single, family, and selection
fixtures together. If it is unsupported, diagnostics must explain what mathematical information is
missing, what would be underestimated without it, and why active protocols do not require it.

## Completion Checklist

Do not report completion until every item is checked.

```text
[ ] There is one executable Graph IR.
[ ] Reshape and ConstantCoefficient are absent from active code.
[ ] Extraction and constant-polynomial lift are explicit operations.
[ ] Graph IR, binary, artifact, generator, report, and cache versions agree.
[ ] Old formats and manifests are rejected.
[ ] The protocol owns decoder targets.
[ ] Lean evaluates p and derives q from validated graph data.
[ ] Single values, families, and selections share IndexedFact semantics.
[ ] Selection is reindexing, not an indicator sum.
[ ] Matrix primitives use one pointwise lifting path.
[ ] Same-selector correlation is preserved.
[ ] Independent selections are not Cartesian-expanded.
[ ] Only BlockColumns * BlockRows receives pairwise contraction.
[ ] [B,I] * [K;0] exposes and consumes B*K.
[ ] One generic indexed preimage rewrite handles all supported forms.
[ ] Residual Large terms are rejected.
[ ] Operational bounds use exact integers and runtime-enforced cutoffs.
[ ] Lean checks 2*p*N < q exactly.
[ ] Shared storage does not scale with logical branch count.
[ ] Cartesian pair visits remain zero.
[ ] Generated Lean files were regenerated rather than edited.
[ ] Toy, BGG, gadgets, Diamond, and workspace gates pass in the approved scope.
[ ] Old semantic paths, aliases, readers, and temporary oracles are gone.
[ ] Every unsupported active case has a mathematical explanation.
```
