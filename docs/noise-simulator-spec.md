# Noise Simulator Specification

## 1. Status and authority

This document is the implementation specification for `crates/noise-simulator`, whose Cargo
package name is `mxx-noise-simulator`. It describes the implemented replacement for the deleted
operational-noise system.

The migration is intentionally breaking:

- the predecessor correctness crate is deleted rather than wrapped.
- Lean generation, certificates, replay, and generated Lean artifacts are deleted.
- application code is migrated to the new graph and family APIs.
- no compatibility parser, adapter, legacy relation attachment, or fallback checker is retained.

This document is self-contained. The obsolete correctness documents were removed with their
implementation and are not part of this contract.

## 2. Problem statement

The simulator must compute a conservative coefficient-noise bound directly from frozen executable
IR. Adding a new application or reusable module must not require a second, module-specific noise
contract that restates the module's graph.

The deleted checker tracked general exact-signal expressions and searched or normalized them until
registered equations could be applied. The implemented simulator uses a deliberately smaller
abstraction:

1. every value has a numeric noise bound;
2. every matrix has a numeric bound on its represented coefficients;
3. a value may carry one distinguished preimage source on its right;
4. preimage and gadget-decomposition nodes create exact directed relations;
5. only an explicit preimage application consumes such a relation, and only when its exact
   identities match; and
6. no other large-term identity, cancellation, or symbolic matrix expression is tracked.

The central convention is that every matrix value is viewed as:

```text
actual value = implicit nominal value + error
```

The nominal value is defined by primitive execution semantics but is not stored as an expression.
For a preimage relation:

```text
B * K = T
T = nominal(T) + eT
X = L * B + other nominal terms + eX
```

the special multiplication rule derives:

```text
X * K
= L * nominal(T) + other nominal terms * K
  + L * eT + eX * K
```

It therefore needs bounds on `L`, `eT`, `eX`, and the magnitude of `K`. The target error is read
from its ordinary matrix state. The relation does not duplicate it and does not store `L`, `B`,
`T`, or the other nominal terms as algebraic expressions.

In this document, a **stage** is one frozen executable `Graph`; an **artifact edge** is the alias
resolved from an artifact input's existing producer metadata to a producer-stage output; a
**family** is a row-major runtime array with logical axes; **group axes** identify independent
relations; and the final **branch axis** selects one target/preimage while keeping its source fixed.

## 3. Goals

The implementation must:

1. analyze frozen IR without knowing an application or module name;
2. derive sampler bounds and preimage relations from first-class IR nodes;
3. support DiamondWE input injection, projection preimages, BGG+ operations, gadget decomposition,
   selectors, families, subgraphs, parallel grids, sequential loops, and linked artifacts;
4. compute bounds under one fully concrete parameter environment;
5. use exact graph identity and fail closed when a relation cannot be justified;
6. keep one uniform numeric summary for a family instead of one summary per element;
7. preserve selector correlation needed by preimage relations without enumerating branches;
8. produce actionable diagnostics containing stage, scope occurrence, node, port, and operation;
9. allow an application to request bounds for arbitrary graph outputs; and
10. make a new protocol composed from supported IR primitives require no simulator changes.

## 4. Non-goals

The crate does not:

- prove functional correctness;
- identify the ideal or approximate value represented by an output;
- prove that a decoder returns the intended plaintext;
- create or check Lean terms;
- implement an e-graph, rewrite saturation, polynomial normal form, or general CAS;
- search for a relation by compatible shape or similar expression;
- prove cancellation of arbitrary public matrices or other large terms;
- derive probabilistic tail bounds from a sampler sigma;
- enumerate selector assignments or family elements;
- infer logical family axes from application-specific flat-index arithmetic; or
- accept opaque operations without an explicit primitive transfer rule.

Functional tests remain responsible for checking that executable outputs have the intended
meaning. The simulator is responsible only for the stated numeric noise bound.

## 5. Workspace architecture

### 5.1 Dependencies

`mxx-noise-simulator` depends on:

- `mxx-ir-core`;
- `num-bigint` and `num-traits`;
- `thiserror`;
- `serde` only for public request/report data if persistence is required; and
- `tracing` for progress reporting.

It must not depend on:

- `mxx-dsl`;
- `mxx-runtime`;
- `mxx-gadgets`;
- `mxx-bgg`;
- `mxx-we`; or
- any application crate.

The dependency direction is therefore:

```text
mxx-ir-core
    ^
    |
mxx-noise-simulator
    ^
    |
application crates such as mxx-we
```

`mxx-we` constructs the simulation program from its frozen stage graphs and invokes the
simulator. Reusable gadget and BGG crates only build ordinary IR.

### 5.2 Concepts retained from the predecessor

Only the following concepts survive, under new minimal types owned by `mxx-noise-simulator`:

- stage identity;
- stage graph;
- artifact resolution from existing IR producer metadata;
- external input assumptions;
- selected output roots; and
- simulation request and report types.

The following concepts do not move:

- ideal programs;
- pure requirement programs;
- comparator specifications;
- endpoint semantic anchors;
- derivation attachments;
- registered module rules;
- proof/certificate data; and
- Lean rendering or replay data.

## 6. Concrete-only evaluation

Every simulation request contains a complete `ParamEnv`. Before analysis begins, the simulator:

1. validates every reached graph with that environment;
2. resolves every matrix dimension, modulus, loop count, family extent, sampler cutoff, and
   selector range needed by the reached roots; and
3. rejects any unresolved or invalid expression.

All bounds are then `BigUint` values. There is no symbolic bound-expression arena.

Parameter search reruns the simulator for each candidate environment. This is intentional: the
small cost of rerunning a numeric abstract interpreter is preferred to maintaining and simplifying
a large symbolic formula language.

Loop indices and selectors may remain symbolic only while analyzing one structural loop body.
Their analysis is limited to typed integer ranges and structural substitution; they never appear
inside matrix-noise bounds.

## 7. Public API

The public API is conceptually the following. Exact Rust field ownership may use references or
`Arc`, but the information content must not change.

```rust
pub struct StageId(pub String);

pub struct SimulationStage {
    pub id: StageId,
    pub production_id: ProductionId,
    pub graph: Graph,
}

pub struct SimulationProgram {
    pub stages: Vec<SimulationStage>,
}

pub struct SimulationRoot {
    pub stage: StageId,
    pub output: String,
}

pub struct ExternalInputFact {
    pub stage: StageId,
    pub input: String,
    pub value: ExternalInputValue,
}

pub struct SimulationLimits {
    /// `None` means no request-level cap beyond address-space and integer limits.
    pub maximum_planned_wires: Option<usize>,
    pub maximum_transfer_steps: Option<u64>,
}

pub struct SimulationRequest {
    pub program: SimulationProgram,
    pub environment: ParamEnv,
    pub roots: Vec<SimulationRoot>,
    pub external_inputs: Vec<ExternalInputFact>,
    pub limits: SimulationLimits,
}

pub fn simulate(request: &SimulationRequest) -> Result<SimulationReport, SimulationError>;
```

Root names must resolve to exact graph outputs. There is no target registry and no decoder kind in
the simulator. An application that wants to compare a returned bound with a decoder threshold does
so after simulation.

A root must be a matrix or a family whose element type is a matrix. A family root reports the one
uniform element-error bound, which is sound for every coordinate. Integer, Boolean, bytes, and
trapdoor roots are rejected because they do not have a coefficient-noise bound.

The report contains at least:

```rust
pub struct RootNoiseReport {
    pub root: SimulationRoot,
    pub maximum_absolute_coefficient_error: BigUint,
}

pub struct SimulationReport {
    pub roots: Vec<RootNoiseReport>,
    pub diagnostics: SimulationDiagnostics,
}

pub struct SimulationDiagnostics {
    pub planned_wires: usize,
    pub transfer_steps: u64,
    pub dropped_carriers: Vec<DroppedCarrierDiagnostic>,
}
```

The simulator returns a bound, not an `accepted` Boolean. This prevents decoder policy from being
mixed into graph analysis.

Stage IDs, production IDs, roots, and external facts are validated for uniqueness before planning.
Each `production_id.spec_hash` must equal the canonical specification hash of that stage's frozen
graph. This prevents an artifact input from aliasing a different graph under a borrowed production
ID.
Resource limits are operational safeguards only: exceeding either limit returns an error and no
partial report. Limits never alter a transfer rule or truncate a sequential loop.

## 8. External input facts

Artifact inputs linked to reached producer outputs inherit the producer state and must not have an
external fact.

A true external input requires exactly one fact keyed by exact `(stage, input name)`. The supported
facts are:

```rust
pub enum ExternalInputValue {
    Matrix {
        maximum_absolute_coefficient_error: BigUint,
        /// `None` means the full centered-residue bound for this matrix type.
        maximum_absolute_coefficient_value: Option<BigUint>,
        is_constant_polynomial: bool,
    },
    IntegerRange {
        minimum: BigInt,
        maximum_inclusive: BigInt,
    },
    Boolean,
    Bytes,
    Trapdoor {
        public_matrix_input: String,
    },
    Family {
        shape: Vec<usize>,
        element: Box<ExternalInputValue>,
    },
}
```

These are boundary assumptions for values that have no producer graph, not per-module contracts.
They cannot state an algebraic relation, override a node transfer, or attach an approximation to an
output. Reusable subgraphs and linked stages receive no such facts for their internal values.

An external matrix's represented coefficient magnitude defaults to the full centered residue
bound derived from its matrix type. A supplied smaller bound is accepted only when it does not
exceed that universal bound. It must never be inferred from observed runtime data.

An external matrix cannot introduce a right carrier. A carrier needed by a checked preimage
multiplication must come from a reached producer graph or a first-class relation-source input whose
pairing is explicitly represented by a trapdoor/public input fact.

`Trapdoor.public_matrix_input` names another exact input in the same stage. Its matrix type and
family shape must match the trapdoor's public type, and its declared error must be zero before it
can become a relation source. This restriction applies only to that direct public source matrix,
not to a noisy value containing the source and not to the relation target. An external `Family`
must have a nonempty shape, a non-family element fact, concrete extents whose checked product fits
`usize`, and the exact input wire shape. Nested families and implicit singleton-to-scalar
conversion are rejected.

## 9. Identities

### 9.1 Planned wire identity

The simulator builds an occurrence-aware plan from every requested root. A planned wire identity
contains:

```text
stage
frozen scope definition
structural call/loop occurrence
node
port
```

Node and port alone are never an identity. Two calls of the same subgraph have different
occurrences.

### 9.2 Canonical value identity

The job interns reached planned wires into a private ID:

```rust
struct ValueId(u32);
```

Subgraph input aliases, subgraph output mappings, and resolved artifact inputs are canonicalized before IDs
are assigned. Only explicit graph/plan links create equality. Equal type, shape, name, bound, or
operation does not.

`ValueId` has no algebraic meaning. It is used only for state tables, exact relation participants,
and diagnostics.

Family-aware identity uses one other private interned ID:

```rust
struct FamilyViewId(u32);
```

A `FamilyViewId` is a canonical structural mapping from declared coordinates to exact planned
value wires. A scalar is represented internally as the same kind of view with no coordinates. The
view is built from occurrence-aware value IDs, scoped grid binders, canonical deterministic index
maps, and alias-exact runtime selectors. It is never derived from a display name, node number
alone, equal bounds, or a hash treated as proof of equality.

Deterministic coordinate mapping uses a separate typed program:

```rust
pub struct IndexMap {
    pub input_indices: Vec<IndexExpr>,
}

pub enum IndexExpr {
    Axis(usize),
    Parameter(String),
    LoopIndex(u32),
    Constant(BigInt),
    Add(Box<IndexExpr>, Box<IndexExpr>),
    Subtract(Box<IndexExpr>, Box<IndexExpr>),
    Multiply(Box<IndexExpr>, Box<IndexExpr>),
    Divide(Box<IndexExpr>, Box<IndexExpr>),
    Remainder(Box<IndexExpr>, Box<IndexExpr>),
    Equal(Box<IndexExpr>, Box<IndexExpr>),
    Less(Box<IndexExpr>, Box<IndexExpr>),
    LessEqual(Box<IndexExpr>, Box<IndexExpr>),
    Select {
        selector: Box<IndexExpr>,
        branches: Vec<IndexExpr>,
    },
}
```

`IndexMap` and `IndexExpr` are serialized `mxx-ir-core` types. The DSL constructs them, runtime
executes them, and the simulator interns their normalized form. `Parameter(name)` has the same
lookup semantics as `IntExpr::Var(name)`. `LoopIndex(slot)` has the same scoped lookup semantics as
`IntExpr::LoopIndex(slot)` and may refer only to an enclosing sequential loop. Coordinates of the
map's own output grid use `Axis(position)` instead; this makes them alpha-normalizable across
occurrences and stages.

Validation first resolves every `Parameter` from `ParamEnv`. During structural execution, each
`LoopIndex` is then replaced by the concrete iteration of its occurrence-aware sequential loop.
Only after these substitutions are constants folded and axes alpha-normalized by position. The
remaining typed tree is compared structurally. Two independently constructed maps with the same
normalized tree therefore have the same identity across stages. Unbound parameters, out-of-scope
loop slots, or a loop-index leaf referring to the map's own parallel grid are errors. The simulator
performs no algebraic-equivalence search beyond this fixed normalization.

The same identity policy gives every scalar or group-indexed integer selector program a private
`SelectorId`. It denotes a canonical function from its group coordinates to an integer selector;
a scalar selector is the empty-coordinate case and may be broadcast. Artifact and subgraph aliases
preserve it. Unlike an `IndexMap`, it may depend on runtime inputs. Equal integer ranges, equal
display names, or independently rebuilt runtime arithmetic do not make two selector IDs equal.

### 9.3 Source identity

A preimage source is interned as:

```rust
struct SourceId(u32);
```

`SourceId` identifies one canonical function from zero or more **group coordinates** to a source
matrix. Its key is conceptually:

```rust
struct SourceKey {
    group_shape: Vec<usize>,
    producer: SourceProducer,
}

enum SourceProducer {
    /// A canonical structural view mapping each group coordinate to one matrix value.
    Matrix(FamilyViewId),

    /// The same canonical gadget matrix at every group coordinate.
    Gadget(GadgetDescriptor),
}
```

An empty `group_shape` is a scalar source. A matrix family source has one exact structural
`FamilyViewId`; two families with equal elements, types, or bounds are not the same source. A
gadget descriptor contains ring, matrix shape, base, digit count, and small/regular mode.

The group-coordinate function is part of source identity, but a relation's branch coordinate is
never part of it. Thus `B[g]` and the broadcast view `B[g]` over `(g, d)` have the same
`SourceId`, while a genuinely branch-dependent `B[g, d]` cannot be registered as a shared-source
relation. `RightCarrier` and relation records store only this interned `SourceId`; they do not
store an unknown left expression.

## 10. Numeric domains

### 10.1 Coefficient convention

All matrix bounds are maximum absolute centered coefficients of polynomial entries before the
final correctness comparison. Noise bounds are not reduced or capped modulo `q`.

Every represented ring value has the universal magnitude bound:

```text
centered_residue_bound(q) = floor(q / 2)
```

Tighter coefficient-magnitude bounds are retained when known. They may be capped by the centered
residue bound because they describe represented runtime ring values. Noise bounds must never be
capped this way because crossing the decoding interval is exactly what the simulator must detect.

### 10.2 Matrix state

```rust
pub struct MatrixState {
    /// Bound on actual - implicit nominal.
    pub error_bound: BigUint,

    /// Bound on both the represented actual value and its implicit nominal value.
    pub coefficient_magnitude_bound: BigUint,

    /// True only when every polynomial entry of both actual and nominal values is constant.
    pub is_constant_polynomial: bool,

    /// The only retained fragment of a preimage-relevant nominal expression.
    pub right_carrier: Option<RightCarrier>,
}

pub struct RightCarrier {
    pub source: SourceId,

    /// Coefficient-error amplification of the unknown left prefix.
    pub left_gain: BigUint,
}
```

If `right_carrier = Some({ source: B, left_gain: g })`, the implicit nominal contains a term whose
rightmost distinguished factor is `B`. Left-multiplying any compatible matrix error by that
unknown prefix contributes at most `g * e` when the error's coefficients are bounded by `e`.
“Compatible” requires the source and replacement to have the same row count; their column counts
may differ, as they normally do between `B` and the target of `B * K = T`.

The nominal may contain other terms. They are intentionally not represented. The simulator does
not claim that the entire nominal equals `L * B`.

For the source value itself:

```text
source = B
left_gain = 1
```

No `ValueId` for the unknown left prefix is stored.

The coefficient-magnitude bound is not a second symbolic value domain. It is the one scalar needed
to bound multiplication: zero-noise matrices such as secrets, selectors, and preimages can still
amplify another operand's error. In particular, `error * K` cannot be bounded from `K.error_bound`
because that bound is zero; it requires `K.coefficient_magnitude_bound`. This scalar never records
`B`, `G`, a matrix sum, or any other large-term structure.

### 10.3 Integer state

Selectors use only:

```rust
pub struct IntegerState {
    pub minimum: BigInt,
    pub maximum_inclusive: BigInt,
}
```

Ranges are computed by ordinary interval arithmetic. They are not matrix bounds and must be kept
in a separate table and type.

### 10.4 Family state

```rust
pub struct FamilyState {
    pub shape: Vec<usize>,
    pub element: AbstractValue,
}
```

`element` is one sound summary for every element. Bounds do not depend on family coordinates or
selector values. Joining alternatives uses maximum numeric bounds.

Family identity and index correlation are structural metadata; they do not duplicate numeric
states per element.

## 11. Bound arithmetic

For compatible matrix operands `A` and `B`, validation constructs:

```text
ProductGeometry {
    inner_dimension = A.columns = B.rows,
    ring_dimension,
}

convolution_factor(left_is_constant, right_is_constant, geometry) =
    1                         if either constant flag is true
    geometry.ring_dimension   otherwise

product_bound(a, b, geometry, left_is_constant, right_is_constant) =
    a * b * geometry.inner_dimension
    * convolution_factor(left_is_constant, right_is_constant, geometry)
```

Scalar matrix multiplication uses inner dimension one. Shape validation happens before this
function is called.

For ordinary multiplication, let:

- `EA`, `EB` be input error bounds;
- `VA`, `VB` be input coefficient-magnitude bounds; and
- `constA`, `constB` be constant-polynomial facts.

The output error is:

```text
product_bound(VA, EB, geometry, constA, false)
+ product_bound(EA, VB, geometry, false, constB)
```

This bounds:

```text
actual(A) * eB + eA * nominal(B)
```

without storing either matrix. This two-term identity is exact because
`actual(A) * actual(B) - nominal(A) * nominal(B)` has the displayed expansion. Adding a separate
`eA * eB` term would double-count it because `VA` already bounds `actual(A)`.

The raw output coefficient-magnitude bound is
`product_bound(VA, VB, geometry, constA, constB)`, capped by the centered residue bound. Addition
uses sum for error and coefficient-magnitude bounds, with the latter again capped by the centered
residue bound.

The two one-sided action gains used by carriers are:

```text
left_action_gain(A, B_geometry) =
    A.coefficient_magnitude_bound * inner_dimension(A, B_geometry)
    * (1 if A is constant-polynomial else ring_dimension)

right_action_gain(K, A_geometry) =
    K.coefficient_magnitude_bound * inner_dimension(A_geometry, K)
    * (1 if K is constant-polynomial else ring_dimension)
```

The replaced source error and the propagated left error are conservatively treated as
non-constant polynomials. These helpers therefore use only the known constantness of the bounded
multiplier. Dimensions come from the concrete multiplication being analyzed; they are not stored
in a carrier or relation.

The implementation may add proven support or zero-row optimizations later, but the first version
must use the conservative formula above. Such optimizations must be isolated pure bound helpers
and must not change identity or relation behavior.

## 12. Implicit nominal semantics

The simulator never stores the nominal expression, but each primitive has a defined nominal
interpretation:

- deterministic constants and ordinary exact random values are their own nominal;
- external matrix inputs have the caller-declared error around an otherwise opaque nominal;
- Gaussian error samples have nominal zero;
- addition, subtraction, scaling, and ordinary multiplication apply the same operation to
  nominals;
- a preimage sample is an exact bounded multiplier, while its registered relation transports the
  target's error when consumed; and
- selection chooses the corresponding nominal branch.

This is sufficient to propagate error without naming the final approximation. Functional
correctness of the implicit nominal is outside the crate.

## 13. Exact preimage relations

### 13.1 Unified relation record

Scalar and family relations use one record and one map:

```rust
struct RightPreimage {
    source: SourceId,
    target: FamilyViewId,
    view: Option<FamilyViewId>,
    selector: Option<SelectorId>,
}

preimages: Map<FamilyViewId, RightPreimage>
```

A scalar `ValueId` is embedded as the zero-coordinate `FamilyViewId` defined in Section 9.2. A
scalar `PreimageSample(public, trapdoor, target) -> preimage` therefore registers zero-coordinate
preimage and target views in this same map. The preimage is the map key, so it is not duplicated in
the record. The exact semantic equation is:

```text
public * preimage = target in R_q
```

In the formulas below, `state(view)` means the scalar `MatrixState` for an empty-coordinate view
and the uniform `FamilyState.element` matrix state otherwise.

Registration requires:

- exact public/trapdoor pairing;
- compatible concrete matrix types;
- zero error on the direct public source matrix;
- a nonnegative hard preimage cutoff; and
- one unambiguous relation for the preimage value.

The zero-error requirement applies only to the `B` that appears directly in `B * K = T`. It does
not require a carrier value such as `s * B + e_b` to have zero error, and it does not require the
target `T = P + E` to have zero error. The target error is read from the ordinary state of the
target view, and the preimage magnitude is read from the ordinary state of the preimage view.
Neither bound is duplicated in the relation.

For every registered relation, preimage and target views have the same shape. The source group
shape is either:

- that same shape, for a pointwise relation; or
- that shape with its final nonempty branch axis removed, for a shared-source branch relation.

This distinction is derived from the shapes and is not stored as a mode flag. Scalar relations are
the empty-shape pointwise case. Selecting the final branch of a shared-source relation turns it
into a pointwise relation without changing the record type.

### 13.2 Gadget relation

`GadgetDecompose(target) -> decomposition` registers the same relation shape:

```text
G(base, digits, mode) * decomposition = target
```

The gadget descriptor is interned as a `SourceId`. Public-trapdoor preimages and gadget
decompositions then use the same consumer logic. Only relation production differs.

Conceptually, the gadget descriptor is a preimage source whose sampling authority is public and
universal: anyone can decompose a target. It therefore needs no secret trapdoor pairing, but after
registration its `RightPreimage` data and consumption rule are identical to ordinary preimages.

### 13.3 Relation consumption

For `value * preimage`, let:

```text
value.right_carrier = (source, left_gain)
preimages[preimage_view] = (relation_source, target_view)
```

The relation may be consumed only if:

```text
source == relation_source
```

The output error is:

```text
left_gain * state(target_view).error_bound
+ right_action_gain(preimage_view) * value.error_bound
```

where:

```text
right_action_gain(K, value_geometry) =
    K.coefficient_magnitude_bound * inner_dimension(value, K)
    * (1 if K is constant-polynomial else ring_dimension)
```

using the concrete multiplication geometry and the ordinary state of the direct right-hand view.
Preimages are normally non-constant-polynomial, so
the convolution factor is the ring dimension.

Concretely, for:

```text
B * K = T = P + E
X = s * B + e_b
```

the implicit output nominal is `s * P`, and the simulator bounds exactly the two noise terms in:

```text
X * K = s * P + s * E + e_b * K
```

No symbolic expansion of `X` or `T` is needed at consumption time. The carrier created while
evaluating `s * B + e_b` retains `left_gain(s)` and its ordinary state retains `error(e_b)`. The
target state retains `error(E)`, while the preimage state retains the magnitude bound of `K`.
Consequently the two generic terms are respectively the bounds for `s * E` and `e_b * K`.

If the target has a carrier `(next_source, target_gain)`, the output carrier is:

```text
(next_source, left_gain * target_gain)
```

Otherwise the output has no carrier.

The output coefficient-magnitude bound is still the ordinary represented-value product bound
capped by `q/2`.

### 13.4 Explicit use only

The new IR has an `ApplyPreimage` matrix operation whose runtime semantics are ordinary matrix
multiplication. DSL methods `apply_preimage` and `mul_decomposed` lower to this operation.

`ApplyPreimage` must consume a unique matching relation or return a typed error.

Ordinary `MatrixBinary::Multiply` always uses ordinary numeric multiplication. It never looks up,
discovers, or consumes a relation. The distinct `WireType::Preimage` and DSL `Preimage` type enforce
this boundary: relation-bearing values cannot be passed as ordinary matrix operands. The DSL also
has a distinct `Decomposition` wrapper, and there is no unrestricted conversion to `Mat`.

`MaterializePreimageExact` is a guarded identity operation from `Preimage` to ordinary matrix. The
simulator accepts it only when the registered target has zero error. `DecompositionEntry` has the
same exact-target guard for one scalar digit.

## 14. Right-carrier transfer

### 14.1 Source discovery

Analysis is performed in two passes over the reached plan:

1. discover and validate relation producers and mark their exact sources; and
2. evaluate values in dependency order.

This ensures that a source used in `s * B + e` receives its source carrier even when the
corresponding preimage node appears later in topological order.

### 14.2 Addition and subtraction

- no carriers: output has no carrier;
- exactly one carrier: preserve it;
- two carriers with the same `SourceId`: preserve the source and add their gains;
- two carriers with different sources: drop the carrier and record a diagnostic reason.

Dropping a carrier does not change the numeric error bound. A later explicit preimage application
then fails because the required fact is absent.

### 14.3 Negation and integer scaling

Negation preserves the source and gain. Integer scaling multiplies the gain by the maximum absolute
scalar value.

### 14.4 Ordinary multiplication

If the right operand has a carrier, the result keeps that rightmost carrier. Its gain is
the right carrier's gain multiplied by `left_action_gain(left, right_geometry)` from Section 11.
If the right operand has no carrier, the output has no carrier.

The left operand's carrier is not preserved through an unmatched right multiplication because its
distinguished source is no longer the rightmost factor.

### 14.5 Views and layout-changing operations

Transpose, slice, tensor, and ordinary concat do not transport relation identity.

- the typed IR prevents applying ordinary matrix layout operations directly to a `Preimage`;
- applying a layout operation to an ordinary matrix carrying a source computes a conservative
  numeric bound, drops the carrier, and records a precision-loss diagnostic; and
- ordinary target construction may freely use views before the target is supplied to a relation
  producer.

This avoids view-specific relation exceptions while allowing Diamond target construction.
Here “view” means a matrix layout operation. Axis-aware `FamilyReindex` is governed separately by
Section 17.5 and may preserve relation identity when its mapping requirements are met.

## 15. Primitive initial states

The following rules apply after all parameters and matrix types are concrete.

| IR value | Error bound | Coefficient-magnitude bound | Constant polynomial |
|---|---:|---:|---|
| zero matrix | `0` | `0` | yes |
| identity/unit/constant polynomial matrix | `0` | exact maximum | yes |
| explicit or rotation polynomial matrix | `0` | exact represented maximum | yes only if all nonconstant coefficients are zero |
| regular gadget matrix | `0` | exact represented gadget maximum | yes |
| small gadget matrix | `0` | exact represented gadget maximum | yes |
| uniform interval sample | `0` | `min(max(abs(min), abs(max)), floor(q/2))` | no, unless ring dimension is one |
| Gaussian sample | hard cutoff | `min(hard cutoff, floor(q/2))` | no, unless ring dimension is one |
| uniform residue sample | `0` | `floor(q/2)` | no |
| plain hash sample | `0` | `floor(q/2)` | no |
| trapdoor public matrix | `0` | `floor(q/2)` | no |
| preimage sample | `0` | `min(hard cutoff, floor(q/2))` | no |
| regular decomposition with base `2^b` | `0` | `min(2^(b-1), floor(q/2))` | no |
| small unsigned decomposition with base `2^b` | `0` | `min(2^b - 1, floor(q/2))` | no |

Sampler sigma is validated but is not converted into a tail estimate. The serialized hard cutoff is
the authoritative correctness bound.

The regular-decomposition bound follows from balanced base-`2^b` digits, whose absolute value is
at most `2^(b-1)`, including the tie case. The small decomposition uses unsigned digits strictly
below `2^b`. The simulator validates `b > 0`, digit count, output shape, and mode against the
concrete matrix parameters instead of trusting duplicate DSL metadata.

The gadget-matrix bound is not a digit bound. For both modes it is the maximum centered lift of the
concrete gadget entries (powers of the base at the positions present in the validated matrix), or
`0` for an empty matrix.

A uniform binary, ternary, or other short random matrix is therefore an exact nominal value with a
finite coefficient-magnitude bound and zero error. A Gaussian error matrix is a zero-nominal value
whose error is the uncapped cutoff and whose coefficient-magnitude bound is capped by
`floor(q/2)`.

## 16. Remaining matrix operations

The first implementation supports every reached matrix operation explicitly:

- add/subtract: Section 11 and Section 14.2;
- multiply: Section 11, Section 13, and Section 14.4;
- fused multiply-accumulate: exactly the result of its declared ordinary scales, products, and
  additions;
- negate and integer scale: numeric scaling plus Section 14.3;
- transpose/slice/concat/tensor: conservative numeric transfer plus Section 14.5;
- coefficient extraction: requires an authoritative canonical range, separate from noise;
- integer-to-constant-polynomial lift: exact, zero error, constant-polynomial;
- CRT recomposition: declared weighted sum using ordinary scale/add rules; and
- packed polynomial reconstruction: exact value with zero error and a bound derived from bit
  ranges.

The layout transfers are fixed as follows:

- transpose and slice preserve per-coefficient error and coefficient-magnitude bounds;
- row, column, and diagonal concat take the maximum input error and maximum input
  coefficient-magnitude bound,
  because concat does not add coefficients;
- concat is constant-polynomial only when every input is constant-polynomial;
- tensor uses the ordinary two-term product error expansion with matrix factor `1` and the usual
  polynomial-convolution factor; and
- every resulting coefficient-magnitude bound is capped by `floor(q/2)`.

As stated in Section 14.5, these numeric rules do not imply carrier or relation transport.

Real nodes (`ConstantReal`, `IntToReal`, real arithmetic, and square root) are evaluated only as
exact concrete parameter values needed to validate reached samplers. Division by zero, a negative
square-root input, or an unresolved real expression is an error. Bytes and typed blobs are opaque
typed values with no numeric noise state. Trapdoor values retain only the exact public pairing and
validated sampler metadata. Boolean and integer `Select` uses exact branch selection when the
selector is a singleton and otherwise joins all reachable branches by the rules of Sections 17
and 20.

The old decomposed `HashSample` variants are deleted. `HashSample` produces only the plain bounded
matrix. DSL helpers for decomposed hashes emit `HashSample` followed by `GadgetDecompose`; an
execution optimizer may fuse them after semantic IR validation. This avoids a hidden plain target
that would otherwise exist only inside a fused hash node.

Threshold decode does not create a matrix-noise state. Applications normally request the residual
matrix input to the decoder as a simulation root and compare its returned bound with their decoder
interval.

Any newly added `NodeKind` must cause a compile error in the simulator's exhaustive dispatch until
its transfer or normal rejection is specified.

## 17. Rank-N family model

### 17.1 Logical shape

The one-dimensional `IndexedFamily { count }` model is replaced by one rank-N family abstraction:

```rust
WireType::Family {
    element: WireType,
    shape: Vec<IntExpr>,
}

ConcreteWireType::Family {
    element: Box<ConcreteWireType>,
    shape: Vec<usize>,
}
```

There are no nested family element types. Multiple logical axes are represented by one shape.
Runtime storage remains a flat row-major sequence, so this change does not require nested runtime
allocations.

Existing one-dimensional families use shape `[count]`.

The DSL exposes structural indexing rather than application-written flatten/unflatten arithmetic:

```rust
let table = Parallel::grid([levels, states, digits]).map(...)?;
let value = table.get([level, state, digit]);
let selected = table.select_axis(2, digit)?;
```

Axis positions, not debug names, are semantic.

The removed one-dimensional family, indexing, and parallel-loop input variants have no
compatibility surface. The implemented IR operations have the following information content:

```rust
FamilyPack { shape: Vec<IntExpr> }
FamilyGetStatic { indices: Vec<IntExpr> }
FamilyGetDynamic { rank: usize }
FamilySelectAxis { axis: usize }
FamilyReindex {
    output_shape: Vec<IntExpr>,
    map: IndexMap,
}
FamilyGather {
    output_shape: Vec<IntExpr>,
    input_rank: usize,
}
ParallelGrid {
    shape: Vec<IntExpr>,
    index_slots: Vec<u32>,
    bindings: Vec<(String, IntExpr)>,
    input_modes: Vec<GridInputMode>,
}

enum GridInputMode {
    Broadcast,
    Reindex { map: IndexMap },
}
```

`FamilyPack` takes exactly the product of the concrete extents in row-major order.
`FamilyGetStatic` takes one family input. `FamilyGetDynamic` takes the family followed by exactly
`rank` integer inputs. Both return one element. `FamilySelectAxis` takes a family and one integer
selector or integer-selector family and removes the named axis, returning an element when the input
rank was one.

For an input shape `prefix + [selected_extent] + suffix`, a selector family must have exactly
`prefix + suffix` shape. A scalar selector is broadcast to that shape. Any lower-rank selector must
first be expanded with `FamilyReindex`; there is no implicit application-specific broadcasting.
Every selector element must be in `[0, selected_extent)`.

`FamilyReindex` maps each output coordinate to a full input coordinate using a deterministic
`IndexMap`; it handles axis permutation, repetition, structural slicing, and deterministic gather.
`FamilyGather` handles runtime-dependent gather. Its inputs are the source family followed by
exactly `input_rank` integer-selector families, each with `output_shape`; those selectors provide
the full source coordinate for every output coordinate.

`ParallelGrid` executes one body over the Cartesian product and returns one family of the declared
shape for each body output. A `Reindex` input mode maps each body coordinate through a deterministic
`IndexMap`; `Broadcast` supplies the same input value to every body occurrence. Zip and zip-offset
are expressed as reindex maps and do not remain separate variants.

The ordinary `Select { count }` node remains distinct: its inputs are one integer selector followed
by `count` same-typed branch values, and it returns one branch value without adding or removing a
family axis.

### 17.2 Uniform bounds

One `FamilyState.element` summarizes all coordinates. `FamilyPack`, branch selection, and any merge
use the maximum error and coefficient-magnitude bounds. No bound contains an axis or selector
expression.

When every packed element has a carrier, `FamilyPack` interns the ordered coordinate-to-source
mapping as one group-indexed `SourceId` and takes the maximum gain. Missing carriers cause the
packed family carrier to be dropped unless the missing branch is a known numeric zero; a known zero
does not erase the other branch's source. The simulator does not invent a source for an unannotated
zero. `ParallelGrid` constructs the same mapping from its scoped body program without enumerating
coordinates.

### 17.3 Static and dynamic indexing

Static indexing validates each coordinate and returns the element summary.

Dynamic indexing requires each selector range to be contained in its axis domain. It returns the
same uniform summary. In both cases, source, preimage, and target views are specialized by the
same canonical coordinate program. Index identity is retained only for exact relation alignment.

`FamilyReindex` uses canonical `IndexMap` identity. `FamilyGather`, `FamilyGetDynamic`, and
`FamilySelectAxis` use alias-exact `SelectorId` identity for every runtime-dependent coordinate.

### 17.4 Selection

Selecting one of `k` scalar or family branches requires a selector range contained in `[0, k)` and
matching branch types/shapes.

The selected error and coefficient-magnitude bounds are maxima, not sums. A carrier is retained
when all nonzero branches have the same `SourceId`; its gain is the maximum branch gain. A known
numeric-zero branch does not remove that source, while distinct sources or a nonzero carrierless
branch drop it and produce a diagnostic. A preimage relation is retained only when every branch has
the same source and the targets are selected with the identical selector and branch order.

`FamilySelectAxis` on one uniformly summarized family does not take a numeric maximum again: it
returns the same element bounds. `Select { count }` across distinct branch values takes the maximum
of their bounds. Both operations retain the exact `SelectorId` used for correlation.

No branch assignments are enumerated.

### 17.5 Gather and reindex

Reindex and gather are structural composition of family indices. They preserve the uniform numeric
summary.

A relation is preserved only when the same canonical `IndexMap`, or the same ordered runtime
`SelectorId` vector, is applied to its preimage and target views. Dependency checking is structural:
each normalized `IndexExpr` and selector program records the set of output-axis binders on which it
depends. Equal numeric ranges are irrelevant.

For a pointwise relation, the complete mapping is also applied to the source view. The output is
another pointwise relation.

For a shared-source branch relation with input shape `G + [D]`, a preserving reindex or gather must
produce shape `G_out + [D_out]` and satisfy all of the following:

- the complete input-coordinate mapping is applied identically to preimage and target;
- every mapping for an input group axis is structurally independent of the output's final branch
  axis;
- those group-axis mappings alone are applied to the source view; and
- only the mapping for the input branch axis may depend on the output branch axis.

The branch mapping may also depend on output group axes. If any input group coordinate depends on
the output branch binder, the operation would require a forbidden source `B[g,d]` and is rejected
with `BranchDependentSource`. No relation fact is silently dropped for a relation-bearing preimage.

`FamilySelectAxis` on a relation-bearing value is supported only for the final branch axis. It
applies the exact selector to preimage and target, leaves the source branch-independent, and yields
a pointwise relation. Selecting a non-final group axis must instead be expressed as
`FamilyReindex` or `FamilyGather` satisfying the rule above; direct use is rejected. A full
`FamilyGetStatic` or `FamilyGetDynamic` is allowed because it removes all output axes: it applies
the same complete coordinate tuple to preimage and target and the group prefix to the source.

Two runtime selectors with the same numeric range are not identical. They must resolve to the same
`SelectorId` through an explicit graph/artifact alias. This restriction does not apply to
deterministic `IndexMap`s, whose normalized typed trees define identity as specified in Section 9.2.

## 18. Family preimage relations

### 18.1 Grouped shared-source semantics

A family preimage relation has zero or more group axes and exactly one branch axis:

```text
for every group coordinate g and branch d:
    B[g] * K[g, d] = T[g, d]
```

The source is independent of `d`. Preimage and target may differ for every `d`. With no group axes,
the source is one scalar matrix `B` shared by the whole family.

Here the relation-family index is `d`, not the whole storage coordinate. `g` packages independent
fixed-source relation families into one table. For each fixed `g`, there is exactly one `B[g]` and
no `B[g, d]`. The group-indexed `SourceId` exists only so that these independent families can cross
grid and artifact boundaries without enumerating `g`.

The family producer registers the unified `RightPreimage` record from Section 13.1. The branch axis
is always the final target/preimage axis. The source's group shape must equal the target shape with
that final axis removed. Therefore group shape, branch count, and branch position are derived from
the map key and interned views and are not duplicated in the record. The canonical source view
cannot contain the final branch-axis binder. This is structural identity only; it carries no
duplicate numeric bound.

### 18.2 DSL construction

The DSL provides a shared-source family sampler. Conceptually:

```rust
let preimages = trapdoors.sample_preimage_branches(
    targets,
    preimage_matrix_shape,
)?;
```

It lowers to a first-class `FamilyPreimageSample` node rather than a collection of unrelated
scalar nodes:

```rust
FamilyPreimageSample {
    matrix_type: MatrixType,
    max_coefficient_bound: IntExpr,
}
```

Its inputs are, in order, the public-source family, matching trapdoor family, and target family.
Its output is the preimage family. The public and trapdoor shapes equal the target shape with the
final branch axis removed; the output shape equals the target shape. A rank-one target therefore
uses a scalar public source and scalar trapdoor rather than singleton families. More generally, an
empty group shape means scalar public/trapdoor inputs; a nonempty group shape means family inputs.

Requirements:

- trapdoor/public inputs have exactly the group shape;
- targets have `group_shape` followed by one nonempty branch axis;
- output preimages have the target shape;
- the source is broadcast over the branch axis; and
- the hard preimage cutoff and matrix types are uniform.

Runtime lowering may execute the node as an ordinary parallel grid of scalar samplers. The frozen
IR retains the node's logical axis and broadcast semantics, so the simulator never has to infer
relation grouping from division, remainder, or a flat slot number.

Gadget family decomposition uses the same relation representation with a source descriptor shared
over all axes.

### 18.3 Branch selection

Selecting branch `d` produces:

```text
B[g] * K[g, d] = T[g, d]
```

The source is not selected along the branch axis. Only preimage and target receive `d`.

Selecting or gathering group axes applies the same group-axis mapping to the state carrier, source,
preimage, and target identities.

### 18.4 Family pointwise application

For:

```text
X[g] = L[g] * B[g] + eX[g]
T[g,d] = P[g,d] + eT[g,d]
```

and uniform bounds:

```text
gain(L[g]) <= GL
error(eX[g]) <= EX
error(eT[g,d]) <= ET
right_action(K[g,d]) <= GK
```

the selected pointwise product has:

```text
error <= GL * ET + GK * EX
```

This is one numeric calculation for the whole output family.

## 19. Structural execution

### 19.1 Subgraphs

A subgraph body is analyzed through its exact call occurrence. Inputs are aliases of caller values;
outputs are aliases of body outputs. There is no subgraph summary contract.

Caching by `(definition, concrete input abstract states)` is optional and must not merge distinct
identities.

### 19.2 Parallel grids

A parallel body is analyzed once with scoped logical axis binders. Numeric output bounds are reused
for every lane. Each binder has a distinct scoped identity, including in nested grids.

Input modes explicitly state each reindex map or broadcast. Relation registration uses these modes
to prove that a source is broadcast over its branch axis.

When a body output carrier depends structurally on grid binders, freezing the grid output interns
the corresponding group-indexed `SourceId`. This lifts the canonical binder program, not one
sampled lane or an enumerated source table. The carrier gain remains the single uniform numeric
bound computed for the body.

### 19.3 Sequential loops

Sequential loops use exact concrete iteration counts. The body transfer is applied repeatedly to
the carried abstract states. There is no widening, recurrence expression, exponentiation shortcut,
or fixed semantic iteration ceiling.

An optional request resource limit may stop work with `ResourceLimitExceeded`; it must not return a
partial bound as successful.

All carried outputs update simultaneously per iteration.

### 19.4 Artifacts

An artifact input's existing `(ProductionId, artifact_name, confidentiality)` metadata resolves it
to the unique stage with that production ID and the exact named producer output. No second
simulator-specific artifact-link list exists. The resolved input aliases that producer output,
including family shape, source identity, carrier, and relation facts. The simulator rejects
missing, duplicate, cyclic, type-incompatible, or confidentiality-incompatible resolution.

The rank-N migration replaces manifest `family_count: Option<usize>` with
`family_shape: Option<Vec<usize>>`. `None` means a scalar artifact and `Some(nonempty_shape)` means
a family artifact. Producer output type, consumer input type, and manifest shape must agree
exactly. The free-form manifest `layout` string may remain for display or runtime tooling, but it is
never used for simulator identity or axis semantics.

`ProductionId` is only the checked lookup key that resolves an artifact edge; after resolution, the
producer's occurrence-aware value and view identities are used. File names, artifact paths, and a
production ID by itself are never substituted for `ValueId`, `FamilyViewId`, or `SourceId`.

## 20. Selector rules

The integer range analyzer supports:

- integer and Boolean constants;
- concrete compile expressions;
- declared external ranges;
- loop/grid indices;
- addition, subtraction, multiplication;
- division and remainder when the divisor excludes zero;
- comparisons and Boolean-to-integer conversion;
- bit extraction; and
- selection by joining branch ranges.

Interval arithmetic uses unbounded `BigInt` endpoints. Addition and subtraction use the usual
endpoint formulas. Multiplication takes the minimum and maximum of the four endpoint products.

Runtime integer values and deterministic coordinate maps have distinct division contracts:

- `IntBinaryOp::Divide` and `IntBinaryOp::Remainder` use the positive absolute value of the
  divisor. For each concrete pair, `q = floor(n / abs(d))` and `r = n - q * abs(d)`, so
  `0 <= r < abs(d)`. Thus `-7 / 3 = -3` with remainder `2`, and divisor `-3` gives the same result.
  Division uses the four endpoint quotients, while remainder is bounded by
  `[0, max(abs(divisor)) - 1]`.
- `IndexExpr::Divide` and `IndexExpr::Remainder` match `BigInt` coordinate evaluation: the quotient
  truncates toward zero and the remainder has the dividend's sign. Thus `-7 / 3 = -2` with
  remainder `-1`. Division again uses the four endpoint quotients. If
  `M = max(abs(divisor)) - 1`, remainder is bounded by `[0, M]` for a nonnegative dividend,
  `[-M, 0]` for a nonpositive dividend, and `[-M, M]` when the dividend range crosses zero.

In both cases, the divisor interval must exclude zero. Comparisons and bit extraction return
singleton ranges when proven and `[0, 1]` otherwise. A branch selection joins only branches
reachable from the selector range.

Every family access and branch selection validates its range. A wholly or partially out-of-domain
range is an error; the simulator does not assume that runtime execution happens to choose a valid
value.

Selector correctness relative to a plaintext is outside the simulator. Because branch bounds are
joined by maximum, the noise result is sound for every in-range selected branch.

Range analysis answers only “which indices may execute.” Runtime relation correlation uses
`SelectorId`, not interval equality. Applying `FamilySelectAxis` to a registered preimage view
automatically applies that exact selector ID to its target view while leaving the source's
branch-independent view unchanged. Rebuilding numerically equal runtime selector arithmetic
creates a different ID unless the IR makes it an explicit alias.

Deterministic `IndexMap` comparison is different: it proves equality of two coordinate functions by
the fixed structural normalization in Section 9.2. It cannot contain an external integer wire or
runtime selector. If a mapping depends on such a value, the DSL must use `FamilyGather` or
`FamilySelectAxis`, and alias-exact selector identity applies.

## 21. Diamond input injector derivation

Let:

- `M` be the state count;
- `D` be the digit branch count;
- `l` be the level group coordinate;
- `t` be the target-state group coordinate; and
- `d` be the selected digit branch.

The migrated DSL represents transitions with logical shape:

```text
[level, state, digit]
```

rather than exposing only the old flat index `l * D * M + d * M + t`.

For fixed `(l, t)`, define the source-state mapping `src(l, t)`. The preprocessing relation is:

```text
for every d:
    B[l, src(l,t)] * K[l,t,d] = T[l,t,d]
```

where:

```text
T[l,t,d] = S[l,t,d] * B[l+1,t] + E[l,t,d]
```

The source depends on group coordinates but not on the branch `d`. The DSL's shared-source family
sampler makes this broadcast structural and checkable.

Preprocessing is therefore shaped approximately as:

```rust
let source_map = IndexMap::new([
    IndexExpr::Axis(0),
    source_state(IndexExpr::Axis(0), IndexExpr::Axis(1)),
]);
let group_trapdoors = bases.reindex([levels, states], source_map.clone())?;
let targets = Parallel::grid([levels, states, digits]).map(/* T[l,t,d] */)?;
let transitions = group_trapdoors.sample_preimage_branches(
    targets,
    (state_columns, state_columns),
)?;
```

The source/public and trapdoor families have shape `[level, state]`; target and transition families
have shape `[level, state, digit]`. The method emits one `FamilyPreimageSample` whose final axis is
the branch axis. `source_state` is an `IndexExpr` program over logical axes, including its
comparison and select; it is not an integer wire or an application-specific simulator rule.

At evaluation, the input digit selects only the digit axis:

```text
selected_K[l,t] = K[l,t,input_digit[l]]
selected_T[l,t] = T[l,t,input_digit[l]]
```

Concretely, the `[level]` digit family is first reindexed to `[level, state]` by the map
`(l, t) -> l`. That selector view is then passed to `select_axis(2, selectors)`. The selected
preimage and its internally retained target relation receive the same `SelectorId`; the source view
does not receive the digit selector.

The gathered source state has:

```text
X[l,src(l,t)] = L[l,t] * B[l,src(l,t)] + eX[l,t]
```

At concrete sequential iteration `l`, evaluation gathers the state family with the deterministic
map `[t] -> [src(l,t)]`. The simulator substitutes that concrete `l` into both this map and the
preprocessing `source_map`, specializes the transition relation's level axis, composes both views
with their underlying producer source, and compares the normalized results. Both become the same
view `t -> B[l,src(l,t)]`. This match does not use selector ranges or application names.

Therefore:

```text
X * selected_K
= L * selected_T + eX * selected_K
= L * S * B_next + L * E + eX * selected_K
```

```text
next_error <= left_gain * target_error
            + preimage_right_gain * current_error
```

Diamond's `target_error` is the bound on `E`, and `current_error` is the bound on `eX`. Neither is
required to be zero.

Because `T[l,t,d]` carries the rightmost source `B[l+1,t]`, Section 13.3 also gives every updated
state the carrier for the next level. This establishes the loop invariant structurally from one
iteration to the next. No transition contract or InputInjector-specific rule is involved.

Preprocessing also emits the initial state as a `[state]` family rather than emitting only a scalar
`p`. It is built from ordinary nodes as:

```text
initial[t] = L0[t] * B[0,t] + E0[t]

L0[0] = [secret, message]
E0[0] = GaussianError
L0[t] = 0 and E0[t] = 0 for t != 0
```

Thus `initial[0]` is the old `p`, while every other lane remains exactly zero. The interval secret
and message are exact nominal bounded values, each `B[0,t]` is a discovered source, and the Gaussian
has zero nominal. Ordinary transfer creates the group-indexed carrier `t -> B[0,t]` with one uniform
gain/error bound. This explicit construction is required: packing a scalar `p` together with raw
zero matrices would not identify which source each zero lane must use at the first preimage
application.

## 22. BGG and gadget behavior

A BGG operation receives no simulator contract. Its executable graph is analyzed directly.

Canonical BGG+ values are constructed as `L * G + e`, with the actual gadget matrix `G` as the
rightmost factor. This form is retained even when `L = 0`: ordinary multiplication keeps
`Some(source = G, gain = 0)` rather than replacing it with an untagged zero. Arbitrary projections
or matrices with compatible dimensions are not falsely tagged as gadget carriers.

`mul_decomposed` lowers to `ApplyPreimage`. A decomposition produced by `GadgetDecompose(A)` owns:

```text
G * D(A) = A
```

When the left operand carries `G`, the common relation consumer computes the resulting noise.

BGG selection must select correlated vector, public-key, plaintext, and decomposition branches with
the same selector. Gadget decomposition structures are reused by the vector and public-key
operations that need the same decomposition. Correlation needed for a relation is represented by
the structural family and selector operations, not by a BGG-specific derivation attachment.

## 23. Errors and diagnostics

All unsupported or ambiguous cases are typed errors. Required categories include:

- invalid or incomplete parameter environment;
- invalid graph/type/shape;
- missing or conflicting external input fact;
- missing, duplicate, cyclic, or incompatible artifact input resolution;
- unknown stage/output/root;
- unsupported IR node;
- selector range outside its domain;
- invalid deterministic index map, including an unbound parameter or out-of-scope loop index;
- family shape or axis mismatch;
- preimage source/trapdoor mismatch;
- duplicate or ambiguous preimage relation;
- explicit relation use without a relation;
- relation source mismatch;
- `BranchDependentSource` when a shared-source relation maps a source group coordinate through its
  output branch axis;
- unsupported structural operation on a typed relation-bearing preimage;
- carrier precision loss in concat, slice/transpose, tensor, fused accumulation, family packing,
  branch selection, or distinct-source addition/subtraction; and
- resource limit exceeded.

Diagnostics include the exact stage, occurrence, node, port, operation, expected source, actual
source, family axes, and selector range when applicable. A dropped-carrier diagnostic records loss
of relation precision, not an unsound numeric bound. If that carrier is later required by
`ApplyPreimage`, explicit consumption fails closed. Ordinary multiplication consumes its left
factor structurally and therefore does not report the left carrier as dropped. Protocol names and
display paths are never used to make a semantic decision.

## 24. Internal module layout

The initial crate should use the following responsibility split:

```text
crates/noise-simulator/
  Cargo.toml
  src/
    lib.rs          public API and exports
    request.rs      stages, roots, external facts, validation
    plan.rs         reachability, occurrences, aliases, artifact resolution
    identity.rs     ValueId, SourceId, family views, interning
    bound.rs        pure BigUint coefficient/gain arithmetic
    state.rs        matrix, integer, Boolean, family abstract states
    relation.rs     scalar/family relation registration and consumption
    family.rs       axes, selection, gather, uniform joins
    eval.rs         exhaustive IR transfer and structural execution
    error.rs        typed errors and diagnostic sites
    report.rs       root extraction, counters, progress events
```

Files should be split further before they approach the repository's normal size limit. Bound
arithmetic and relation consumption must remain independently unit-testable pure functions.

## 25. Implementation order

Implementation proceeds in these gates:

1. Add rank-N family shape, rank-N artifact manifests, deterministic `IndexMap`, runtime
   `FamilyGather`, and axis-aware DSL/IR/runtime validation with no simulator.
2. Add `ApplyPreimage` and the shared-source family preimage DSL/IR structure; replace decomposed
   hash variants with plain hash plus `GadgetDecompose`.
3. Create the crate, request types, concrete validation, occurrence-aware plan, and identity tables.
4. Implement pure numeric bound arithmetic and scalar matrix states.
5. Implement relation discovery, carriers, scalar preimage/gadget consumption, and explicit use.
6. Implement uniform families, axes, selectors, gather, and family relations.
7. Implement subgraphs, parallel grids, sequential loops, and artifact aliasing.
8. Migrate DiamondWE and BGG DSL code to the new axes and explicit relation-use APIs.
9. Migrate parameter search to consume `SimulationReport` and perform its own threshold comparison.
10. Delete the predecessor crate, proof-generation/replay code, old derivation attachments, old
    semantic anchors used only by correctness, and obsolete correctness documents/tests.
11. Update `docs/architecture.md` and workspace dependencies.

No compatibility layer is added at any gate.

## 26. Test requirements

### 26.1 Pure bound tests

Test zero, constant-polynomial, full-ring convolution, rectangular matrix inner dimensions,
modulus coefficient-magnitude capping, the exact two-term product identity, and uncapped error
growth.

### 26.2 Scalar relation tests

Test:

```text
x = L * B + eX
B * K = P + eT
x * K
```

and verify exactly:

```text
error = left_gain(L) * error(eT)
      + right_gain(K) * error(eX)
```

Also test source mismatch, duplicate relation, direct ordinary multiplication, `ApplyPreimage`, and
`mul_decomposed`. Verify that ordinary multiplication never consumes or searches for a relation.
Separately test that a noisy carrier value and noisy target are accepted, while a nonzero error on
the direct public relation source is rejected.

### 26.3 Primitive-source tests

Test ternary/binary samples as exact bounded nominal values, Gaussian samples as zero-nominal
errors, trapdoor public/preimage pairing, and regular/small gadget decompositions.

### 26.4 Family tests

Test shared scalar source, grouped sources, per-branch preimages and targets, branch selection,
group-axis gather, branch-axis selection, source mismatch, different selectors with equal ranges,
structurally identical cross-stage `IndexMap`s, parameter and sequential-loop substitution,
comparison/select index maps, and uniform maximum joins. Verify that equal runtime selector ranges
remain insufficient while equal normalized deterministic maps match. Verify that a branch mapping
may depend on the output branch, while a group/source mapping that does so is rejected with
`BranchDependentSource`. Also test pointwise relation reindex, final-branch selection, and full
static and dynamic specialization.

### 26.5 Structural tests

Test repeated subgraph occurrences, nested grid binders, broadcast versus zip modes, exact artifact
aliasing, concrete sequential-index substitution into `IndexMap`, and sequential carried-state
updates.

### 26.6 Diamond acceptance test

Construct a small Diamond graph using logical `[level, state, digit]` transitions. Verify that:

- the initial family has source view `t -> B[0,t]`, including its exact-zero lanes;
- source identity is independent of the digit axis;
- digit selection specializes only preimage and target;
- the state carrier source matches the selected relation source;
- every sequential level returns a finite bound; and
- changing the digit selector does not change the uniform numeric bound.

### 26.7 Differential tests

For very small rings and shapes, sample concrete values within declared cutoffs and assert that
measured centered errors never exceed the simulated bound. These tests validate transfer rules; they
are not a replacement for algebraic unit tests.

## 27. Completion criteria

The replacement is complete only when:

1. `mxx-noise-simulator` has no dependency on DSL, runtime, gadget, BGG, WE, proof, or application
   code;
2. every reachable IR node has an explicit transfer or typed rejection;
3. DiamondWE parameter simulation runs through the new crate with no module contract or derivation
   attachment;
4. scalar and family preimage relations use the same consumer;
5. family bounds are uniform and selectors are not enumerated;
6. the digit-branch source invariant is structurally checked;
7. application code performs decoder-threshold comparison outside the crate;
8. predecessor correctness and proof-generation paths are removed from the workspace; and
9. all focused unit tests, workspace compile gates, formatting, and documentation checks pass.

## 28. Consistency decisions

The following decisions resolve otherwise ambiguous parts of the design:

1. **No stored approximation versus target noise.** The approximation is implicit, but every
   target has a normal `MatrixState`; relations read its error directly.
2. **No large-term structure versus preimage matching.** General large terms are absent. Only one
   rightmost `SourceId` and its amplification gain are retained.
3. **No module contracts versus relation semantics.** Modules have no contracts. `PreimageSample`
   and `GadgetDecompose` are primitive IR semantics implemented once.
4. **Uniform family bounds versus indexed equality.** Numeric bounds ignore indices, while exact
   family views and selectors retain only the identity needed to align relations.
5. **Shared source versus Diamond's old flat layout.** The relation is grouped by `(level, state)`
   and quantified over the digit branch. Rank-N family axes expose this distinction directly.
6. **Random small values versus errors.** Short uniform values are exact nominal multipliers;
   Gaussian error nodes have nominal zero.
7. **Preimage bound versus target/carrier error.** The preimage cutoff bounds right multiplication;
   target and carrier-value errors come from their ordinary states. They may be nonzero and are not
   stored in the relation. Only the direct public source `B` in `B * K = T` must be exact.
8. **Explicit relation use versus search.** Only `ApplyPreimage` and DSL `mul_decomposed` consume a
   direct right-hand typed preimage. Ordinary multiplication performs no relation lookup, candidate
   scan, or algebraic equivalence search.
9. **Matrix views versus family indexing.** Matrix layout operations may build targets, but do not
   transport source/preimage identity in the first implementation. The explicitly specified
   family reindex, gather, and selection rules do transport relation views.
10. **Symbolic protocols versus simple bounds.** Graph parameters are symbolic at construction,
    but every simulator invocation is concrete and all numeric bounds are computed directly.
11. **One family summary versus group-dependent source.** Error, magnitude, and gain are uniform
    numbers; `SourceId` alone is a structural function of group coordinates. It never stores a
    numeric bound per coordinate.
12. **Source independence versus packed Diamond tables.** Source independence is with respect to
    the final digit branch. Different `(level, state)` groups may name different sources, and the
    group view records that mapping without creating `B[level,state,digit]`.
13. **Exact-zero lanes versus required source identity.** The simulator does not guess a source for
    a packed zero. Diamond constructs its initial family as `L0[t] * B[0,t] + E0[t]`, so even zero
    lanes obtain their source from ordinary IR.
14. **Artifact composition versus duplicate wiring.** Cross-stage aliases are resolved from the
    existing artifact input metadata. The request does not restate an artifact-link graph.
