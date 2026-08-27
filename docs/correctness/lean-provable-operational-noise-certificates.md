# Lean-Provable Operational Noise Certificates

## 1. Status and authority

This is the implementation specification for the current Tall vertical slice. Implementation is
authorized by the current task, subject to the gates below and review of every change. The
Security0 and Security128 implementation and final evidence gates are complete.

The operational-noise Rust implementation at the pinned source revision is the semantic and
performance authority. In particular, the behavior of `arena.rs`, `program.rs`, `lower.rs`,
`normal_form.rs`, `bound.rs`, `relation.rs`, `report.rs`, and `simulation.rs` is not changed by
this design. Certificate support may add opt-in observation hooks, a recorder, serialization,
proof rendering, and fixed Lean definitions. Any proposed change to the existing Rust core
semantics requires separate review and explicit approval; the existing Rust core remains the
semantic and performance authority.

If this document and the pinned Rust core disagree, certificate emission fails until the
certificate or Lean replay is corrected. It must not silently reinterpret the Rust result.

The first target is one accepted Tall-BGG nested-RNS operational-noise parameter set. The pinned
checker must resolve the request to exactly one `ResolvedAcceptanceTarget` whose kind is
`ResolvedDecoderKind::Threshold`. The canonical source recipe records the fixed Tall constructor,
the exact `OperationalCheckRequest`, and the pinned source/evaluator identities. The projection
recomputes the target, takes `p` from the threshold kind, `q` from
`ResolvedAcceptanceTarget.ciphertext_modulus`, and the residual root from
`ProductionRoots.residual`. A direct Rust run and a run reconstructed from `Source.json` must have
the same report and deterministic core counters. The certificate proves the strict operational
noise condition; it does not specify or prove a runtime decoder output. Parameter-family theorems
are out of scope.

## 2. Required guarantees

The design must provide all of the following:

1. The generated theorem has one fixed, non-generated type.
2. Lean checks the residual semantics, exact cancellations, surviving-noise bound, and strict
   operational inequality `2 * p * noise < q` with the kernel.
3. `native_decide`, `sorry`, generated axioms, protocol-specific trusted code, node-number
   shortcuts, debug-string identity, and fixture-value shortcuts are forbidden.
4. Statement data is deterministically projected from the fixed source recipe, exact request, and
   pinned audited Tall constructor. Proof schedules and proof terms cannot change the statement.
5. Within the residual proof closure, the certificate preserves the existing Rust checker's typed
   identities, scoped event occurrences, exact relation applicability, and bound-transfer rules.
6. Certificate generation adds no work to normal checking when emission is disabled.
7. Certificate work is linear in the residual proof closure and total emitted proof/context
   payload, plus the complete explicitly measured Cartesian rows of index-use LUTs in that closure.
   It never enumerates matrix-valued family elements or parallel-loop lanes, and never jointly
   enumerates unrelated index uses or selectors outside one registered use's dependency frontier.

## 3. Simplicity criterion and selected architecture

Simplicity is evaluated by the number of semantic data structures, independent concepts,
exception paths, and stages in the normal processing path. The selected design uses:

- the existing frozen protocol bundle plus `OperationalCheckRequest` as the only source format;
- one generic balanced `RowTable` representation for all ID-addressed tables;
- one expression table and one program table, mirroring the existing two Rust arenas;
- one tagged source table and one tagged sampler-event table;
- one exhaustive index-use LUT representation and one synchronized slice-group predicate;
- one validation schema exposed as compositional `Cert.Valid` for production and reflected
  `Cert.wellFormed` for small compatibility tests; and
- separate generated statement and proof files, plus one fixed acceptance module.

These structures are justified by measured lower bounds. Balanced row-local validity passed a
5,000-row stress test after dense whole-table validation with repeated lookup failed at 1,000
nodes. Fuel-stable one-step haves passed a 1,000-have stress test after recursive `rfl` over the
fuel interpreter failed at six nodes. The exhaustive LUT lets Lean check each required finite
function without duplicating Rust discovery. Generated declarations may be chunked for kernel
performance. Chunking and the sequence of ordinary `have` terms are proof-rendering techniques,
not additional semantic data structures.

## 4. Trust boundary

### 4.1 Fixed trusted code

The trusted code is deliberately small and protocol-independent:

- Lean matrix, scalar, and value semantics;
- the fixed Tall expression/program semantics for the reached residual closure;
- the tagged input and sampler contracts;
- `RowTable`, `Cert.Valid`, `Cert.wellFormed`, and their shared reflection lemmas;
- the fixed `TallSemantics.Security0Accepted` endpoint;
- the additive Rust source-identity sidecar and canonical source projection.

The existing Rust normalizer, relation search, bound search, proof schedule, and proof renderer are
untrusted proof producers. A mistake in them must fail Lean proof checking.

Three terms used below describe the semantic boundary. A `ValueClaim` is an event-local statement
about one recorded result: it says that the result is congruent to recorded terms and has the
recorded bound. It is not a global value assigned to every event carrying the same owner. A
`Witness` supplies the concrete environment, sampler contracts, relation congruences, and honest
terminal bridge needed to instantiate those event-local claims. `ExactClaimAt` ties a claim to one
specific event, owner invocation, terms, summary, and history row. Thus Lean can check each theorem
application without assuming that all rows with one owner have the same payload.

Statement-bearing certificate rows can change the theorem subject. The source-selection and
source-to-certificate projection boundary is therefore audited trusted code; kernel checking alone
does not prove that the selected source is the deployment intended by an operator.

The certificate path is an honest-run proof artifact. Its validation scope is intentionally limited
to four obligations: the correctness of output produced by the normal generator; the semantic
correspondence, for every fact used by the Lean theorem, of the recorded LHS, RHS, owner invocation,
and event with the actual Rust computation; complete cache and frame lifecycle for an honest run; and
the structural checks needed to reject malformed or dangling references before they can cause a
panic. This path is not a general adversarial certificate parser or an exhaustive anti-forgery
mechanism. Dedicated mutation tests and exhaustive witness-swap or witness-forgery rejection are
outside this scope. With emission disabled, the ordinary Rust checker path and its performance are
unchanged. Benchmark estimates are not a deliverable of this specification.

### 4.2 Protocol-dependent audited data

`Source.json` is a canonical fixed-profile recipe, not a complete frozen-bundle serialization. It
contains the exact `OperationalCheckRequest`, the fixed audited Tall constructor parameters, and
the pinned source/evaluator identities needed to reconstruct that profile. It does not duplicate
the bundle's execution semantics, generated certificate rows, proof terms, normal forms, LUT
outputs, or bound ledger. The constructor is audited as the source of the profile; it is not a
generic protocol decoder.

Canonical projection reconstructs the pinned checker run from this recipe and requires exactly one
successfully resolved `ResolvedAcceptanceTarget`. Its kind must be
`ResolvedDecoderKind::Threshold { plaintext_modulus }`; this value is `p`, its
`ciphertext_modulus` is `q`, and `ProductionRoots.residual` is the residual root. A
`BooleanInterval` target, target-resolution failure, missing or multiple resolved targets, a
request/target identity mismatch, a residual modulus mismatch, or a threshold-report `p`/`q`
mismatch rejects emission. The direct Rust run and the Source-reconstructed run must have identical
semantic report fields and deterministic core counters. This direct-vs-Source parity is the
statement-source audit; it is not a claim that `Source.json` is a portable serialization of every
bundle object. A digest may be included as audit evidence but is never semantic identity.

### 4.3 External applicability obligations

The Lean theorem is conditional. Applying it to one execution requires external evidence that:

- the deployed frozen bundle, request, source revision, and evaluator revision are exactly the
  selected canonical source artifact; the request itself contains the target ID and parameter
  environment;
- for every `SourceAccess` in the residual proof closure, the value supplied by the execution is
  the value assigned to that same typed source, owner invocation, scoped substitution, and optional
  `Nat` family selector, and it satisfies the raw facts in `InputContract`; and
- for every scoped event occurrence in the residual proof closure, the value produced by the
  execution is the value assigned to that exact event and owner invocation, and it satisfies the
  typing, cutoff, support, and relation clauses in `SamplerContract`.

The first item is an audited deployment-equality proposition outside the Lean claim. The second
and third items provide the concrete `InputAssignment` and `SamplerAssignment` to which the Lean
claim is applied. Equality of a digest is audit evidence but is not a replacement for the typed
equalities above.

The certificate does not claim unconditional satisfiability of `InputContract` or
`SamplerContract`. In particular, if a sampler cutoff is probabilistic rather than enforced by
the runtime, a separate probability theorem must bound the chance of leaving the contract. No
`SamplerEventsRealizable` alias or synthetic witness is part of the acceptance claim.

### 4.4 Exact premises remaining after Lean acceptance

Once the fixed acceptance module has checked the generated proof of
`TallSemantics.Security0Accepted`, the operational inequality, family-selector domains,
modulus/ring agreement, and certificate validity are proved facts, not remaining assumptions. There
is also no residual-root argument premise.

Applying the accepted operational theorem to a concrete run leaves the following contract and
modeling premises:

```text
InputContract document inputs
SamplerContract document inputs samplers
HonestTerminalCongruence document run
RecordedCoefficientCoverage document run
```

For a real execution, the operator must additionally establish the following bridge outside the
Lean claim:

```text
DeploymentMatches(run, document)
  := canonicalSource(run.bundle, run.request, run.sourceRevision, run.evaluatorRevision)
     = the accepted Source.json paired with document and history

InputsInstantiate(run, inputs)
  := for every SourceAccess a in the residual proof closure,
       inputs(a) = the value actually supplied by run at a

SamplersInstantiate(run, samplers)
  := for every event occurrence e and owner invocation o in the residual proof closure,
       samplers(e, o) = the value actually produced by run at (e, o)

HonestTerminalValuesInstantiate(run, witness)
  := for every reached terminal Result event e,
       witness.honestTerminalActual(e) = the value actually produced by run at e
```

`InputContract` then proves the recorded type, range, coefficient, and support facts about those
equal input values. `SamplerContract` proves the recorded type, cutoff, support, and exact
relations about those equal event values. If no concrete assignments satisfying both contracts
and the terminal bridge can be exhibited, the implication may be logically true but certifies no
real execution.

A terminal Result is an exact Result introduced immediately after one of the five reached base
transfers: fact-store authority, program-family-fact authority, operator authority, identity, or
scale. For each such event, `Witness.honestTerminalCongruence` states that the event-indexed
`honestTerminalActual` agrees modulo `q` with evaluation of the exact polynomial recorded in that
same history row. The Lean kernel proves the final theorem conditionally on this field. The
generated proof producer can prove which row is terminal, but it cannot prove or manufacture the
field: the caller must instantiate `honestTerminalActual` from the honest Rust execution and
establish the congruence. This is a modeling assumption connecting the theorem statement to the
execution, not a fact obtained merely by compiling the generated proof.

For a fixed selector and honest run, every exact `Result` used through a coefficient reference
also has an event-local modeling premise: its actual centered coefficient norm is at most the
authoritative post-normalization coefficient bound recorded in that same `Result` event.
`Witness.recordedCoefficientCovers` carries this conditional premise, indexed by the exact event,
frame, owner, normalized terms, coefficient producer, summary, and recorded bound. The opt-in Rust
projection asserts that all of those fields are identical to the immutable `ProofPayload`; this
prevents the generated statement from selecting a different row or bound. That Rust assertion
does not prove the norm inequality in Lean. The kernel theorem remains conditional on the caller
supplying `Witness.recordedCoefficientCovers` for the honest run.

Accordingly, the fixed Security0 statement takes the residual value as a dependent function of
both the selector and its honest-run `Witness`, rather than as a function of `Env` alone. The
Source/Cert-side function is part of the fixed statement, and the generated Proof supplies the
kernel-checked derivation for it. Instantiating that function with the honest witness is the
execution-correspondence obligation; no existential residual value is introduced by the proof.

Lean compilation also remains relative to two audited trusted-code propositions: the fixed Lean
value/matrix/interpreter semantics match the pinned Rust operational-noise semantics, and the
trusted canonical projection reruns the pinned checker and maps the accepted `Source.json` to the
statement-bearing certificate rows, including the uniquely resolved `p`, `q`, and residual root,
without changing their meaning. These are trust-boundary obligations, not hypotheses that a
generated proof may introduce.

## 5. Fixed Tall theorem types

The fixed `TallSemantics` module defines the ABI used by this vertical slice. Generated code cannot
redefine it or replace it with a generic certificate interpreter. The fixed acceptance endpoint is
`TallSemantics.Security0Accepted`. It binds the audited Tall document and immutable event history
to the final Result, PreFold, InvocationEnd, residual function, and direct strict inequality
`2 * p * noise < q`. Security0 and Security128 use this same fixed endpoint shape; the profile
parameters and generated rows differ, not the theorem's meaning.

The generated files are deliberately sharded: `Cert/` contains statement rows and local validity
proofs, `Proof/` contains the ordinary theorem applications and replay history, and `Semantic/`
contains the reached semantic theorem applications and their final composition. A small fixed
acceptance module imports these shards and checks the fully qualified `Security0Accepted` type;
the Tall acceptance path has one fixed endpoint and only the reached residual semantics.

`ResidualRoot` mirrors the Rust root classification exactly: it is either a closed matrix
expression or a one-argument matrix family. Rust's `FamilyDomain` is an exact half-open interval
with `u64` endpoints; Lean stores the same endpoints as `Nat`. A family selector is only a `Nat`
and must satisfy the exact lower and upper bounds. A closed root has no caller-supplied arguments,
and a family root is bounded for every selector in this domain. Root arguments are derived from
the checked root structure rather than supplied as an untyped external list.

`Cert` stores `plaintextModulus` and `ciphertextModulus` directly. Its validity requires positive
moduli and a closed matrix or matrix-family residual over exactly `R_q` and the certificate ring
dimension. The trusted projection takes `p` from `ResolvedDecoderKind::Threshold`, `q` from
`ResolvedAcceptanceTarget.ciphertext_modulus`, and the root from `ProductionRoots.residual`;
generated proof data cannot choose them. Runtime decoding is outside the theorem.

The fixed theorem `Cert.Valid.wellFormed` derives `Cert.wellFormed cert = true` without reducing
the entire production table. `Cert.Valid` and `Cert.wellFormed` are two proof interfaces to the
same fixed row predicates; they must not be maintained as independent validation rule sets.

A fixed acceptance module checks the generated proof at the fully qualified type
`TallSemantics.Security0Accepted` and runs `#print axioms` on its acceptance theorem. It does not
derive or check a runtime decode result.

For G1, the reviewed Lean-standard axiom allowlist is exactly `propext` and `Quot.sound`, which
are kernel-library foundations rather than certificate-specific trust assumptions. `#print axioms`
must report no `sorryAx`, generated or custom axiom, `native_decide`, or any axiom outside that
two-item allowlist. Any additional axiom requires a separate design review before acceptance.

## 6. Minimal certificate model

IDs are certificate-local natural numbers assigned deterministically. They are references, not
semantic identity. All semantic identity is stored in typed rows. ID-addressed rows use the same
generic balanced `RowTable`; exact lookup follows one branch, while its ordered view preserves
audit order.

```text
Cert = {
  plaintextModulus,
  ciphertextModulus,
  ringDimension,
  expressions : RowTable ExprRow,
  programs    : RowTable ProgramRow,
  sources     : RowTable SourceRow,
  events      : RowTable EventRow,
  indexUses   : RowTable IndexUseLut,
  sliceGroups : RowTable IndexedSliceLutGroup,
  residualRoot
}
```

`residualRoot` is the tagged `ResidualRoot` described in §5. Together with the two direct modulus
fields and `ringDimension`, it is the complete top-level target data.

The **residual proof closure** is the transitive dependency closure starting only at
`ProductionRoots.residual`. It includes expression and program dependencies and only the sources,
events, relations, bound facts, index uses, and synchronized slice groups actually required to
evaluate that root and prove its bound. `ProductionRoots.decoder` is not a second root of this
closure. Decoder-only expressions, events, traces, and `ThresholdDecode` semantics or lemmas are
not serialized. Every dependency in the residual proof closure is serialized, and every
serialized row must belong to that closure. This is only a certificate-projection boundary: the
existing Rust core may continue to analyze `ProductionRoots.decoder` exactly as it does now.

`Value` has exactly the value variants required by the current Rust `ResolvedValueType`:
`Bool`, `Int`, `Real`, `Bytes`, `Matrix`, and `Trapdoor`. Programs and families are references in
expression rows, not runtime `Value` variants.

`SourceRow` is one tagged sum for constants, declared protocol inputs, unbound occurrence-local
inputs, and producer artifacts. It stores the complete authoritative Rust identity and resolved
type. Family ownership is represented by the source's owner scope and signature rather than by a
second source-ID namespace. Direct and family rows also own the optional raw value contract
projected from the corresponding Rust facts: a signed half-open range, coefficient class,
canonical-coefficient exclusive upper, and polynomial-support upper. A constant row does not
repeat facts that follow from its literal value.

`EventRow` is one tagged sum for uniform, Gaussian, sampled hash, trapdoor-public, preimage, and
gadget-decomposition events. Every row stores its owner scope, signature, output type, and complete
kind-specific descriptor. Its optional raw contract contains only independently recorded range,
canonical-coefficient, or support facts. It does not repeat a cutoff or decomposition bound already
present in the event descriptor. A deterministic hash is an ordinary typed expression operator,
not a sampler event.

Each descriptor has exactly one canonical certificate location. Constant and external-input
payloads live only in `SourceRow`; sampler and transform descriptors live only in `EventRow`.
Expression rows that read them contain only a typed source or event reference plus the scoped
access information. Expression-local operator parameters that are not source or event identity
live only in the corresponding `ExprRow`. Program rows contain signatures and bodies, not copies
of referenced source, event, or expression descriptors. `Cert.Valid` rejects a row that attempts
to inline or shadow table-owned identity.

A reached `GadgetDecompose` transform has no source `SampleEventId`. The projection therefore
derives one canonical event reference from its typed scoped transform expression and exact
parameters without changing the Rust arena. `PackPolynomialCoefficients` remains unsupported
until the coverage gate supplies both its exact semantics and proof lemmas; reaching it before
that gate rejects emission.

The additive recorder-side identity map is captured in `lower.rs` while structured
`PlannedWire`, `ProtocolInputId`, `ProgramOccurrence`, artifact identity, and sampler information
are still available. It is associated with the resulting expression/event ID. Missing,
conflicting, or many-to-one mappings reject emission. Existing arena identity and interning are
not changed. Existing string fields may be preserved as named fields of their typed descriptor;
formatted debug text or a digest must not replace the descriptor.

Deterministic hash identity includes definition, version, key length and bytes, output type, tag
prefix, the distinct binary/decimal/little-endian-u64/dynamic tag groups with their ordered
boundaries, and decomposition parameters. Equality is equality of the fully evaluated typed query
in its owner scope.

## 7. Total interpreter and structural validity

The fixed Lean interpreter mirrors the current reachable `ExprArena` and `ProgramArena`
semantics. It uses structural fuel and total fallbacks. Fuel-stable one-step lemmas expose ordinary
nodes, `ProgramCall`, and `IndexUse` without recursively unfolding descendants. Rust memoizes and
emits adjacent-fuel child haves. `Cert.Valid` proves the fallbacks unreachable for
contract-satisfying assignments.

### 7.1 Semantic owner claims and the CP0/CP1 trust split

Within one replay, an `Owner` is the typed pair of scope and expression row. An owner used as a
central or ordered monomial factor must denote one value in that replay and, for a program scope,
for every selector in the program domain. The opt-in Rust generator enforces CP0 before emitting
Lean: coefficient-normalized exact-zero claims for a factor owner must agree, and an owner with
multiple distinct normalized result payloads must not occur as a factor. For a nonfactor owner,
CP0 checks only the multi-payload case where exact-zero claims coexist with alternate finite
claims: every alternate finite claim must be an empty exact result obtained by the recognized
direct-survivor or sum-after-survivor fold chain. Other nonfactor multiplicity is not
theorem-load-bearing because proof references identify event-level claims. In particular,
singleton coefficient-finite and nonempty exact-finite claims remain obligations for the CP2
semantic `Result` and `Transfer` proof. The deterministic semantic-owner statistics file records
the event-level claims, frame starts, summaries, and frame-root predecessor bindings used by this
check.

CP1 keeps these roles separate. `Env` is queried only for factor owners, and `ValueClaim` remains
an event-level statement. `ExactClaimAt` pairs such a claim with the exact owner, raw terms,
summary, and `Result` history index that it interprets, so an arithmetic proof cannot silently
substitute another event's claim. `Witness` bounds factor atoms and also carries the reached-only
modeling bridge described in Section 4.4: `honestTerminalActual` is indexed by the terminal Result
event, and `honestTerminalCongruence` relates that value to the polynomial in the same history row.
The five admitted terminal transfer forms are fact-store authority, program-family-fact authority,
operator authority, identity, and scale. The generated producer proves the row lookup and then
uses ordinary arithmetic theorems; it cannot construct the honest-execution congruence.

The generated Lean theorem is kernel-checked for every witness satisfying these explicit
premises; Lean compilation does not prove that an honest Rust execution supplies such a witness.
CP0 is separately a trusted-generator correspondence assertion that the owner keys used by that
document model the values in the honest Rust replay. This split does not add work to the ordinary
checker path.

Production constructs `Cert.Valid` from row-local `RowTable.AllFrom` witnesses. The Boolean
`Cert.wellFormed` remains for small fixtures and is connected by fixed reflection theorems. The
shared validation obligations include at least:

- dense topological expression references and one acyclic combined dependency graph covering
  expression children, program calls, producer-artifact edges, event operands, hash-query
  expressions, and relation links;
- exact operator arity and complete value types, including `Bytes`;
- matrix coefficient count, modulus, ring dimension, rows, columns, and all logical shape rules;
- residual-root classification, exact family domain where present, and absence of free arguments
  for a closed root;
- positive plaintext and ciphertext moduli, and exact residual modulus/ring agreement;
- program signatures, argument ownership, family domains, and call substitutions;
- unique ownership of every source, event, relation link, and index-use row, plus exact membership
  of every serialized row in the residual proof closure;
- complete typed hash descriptors and scope substitutions;
- in-range slices, coefficients, table references, and index consumers; and
- sufficient evaluation fuel derived from the checked combined DAG.

For the Tall target, the canonical projection selects `ProductionRoots.residual` and its exact
`FamilyDomain` (with `u64`/`Nat` endpoints) from the pinned checker run reconstructed from the frozen
Rust source. `Cert.Valid`
checks that structure and the residual-only closure; neither is an execution-supplied hypothesis.

Physical Rust `MatrixLayout` metadata remains in the canonical source and continues to be checked
by the existing Rust arena. Lean checks every logical routing and shape property its interpreter
uses, but does not duplicate unused physical stride checks.

## 8. Input and sampler contracts

`InputAssignment` is one total function from `SourceAccess` to `Value`. A `SourceAccess` contains
the source reference, normalized owner invocation, the exact checked scoped substitution, and an
optional `Nat` family selector. The selector is present exactly for a family access; two accesses
to the same family at different selectors are therefore distinct even within one owner invocation.
`Cert.Valid` checks the substitution against the owning program signature and checks the selector
with `domain.Contains selector`. `InputContract` requires the recorded resolved type
for every valid access and only the raw facts consumed by the fixed Rust analysis. Centered
coefficient bounds, canonical coefficient exclusive uppers, and polynomial-support uppers are
separate fields and predicates. Only a canonical exclusive upper can justify an
extracted-coefficient index domain. A support upper must be owner/source/family-access-selected
exactly, must not exceed the ring dimension, and proves that all later polynomial positions reduce
to zero. Current integer facts are declared signed half-open ranges. Family domains and
family-access selectors use `Nat`; this does not change the signed representation of other integer
facts. Derived range,
sparsity, and constant facts are proved, not silently assumed.

`SamplerAssignment` is one total function from an event reference and owner arguments to `Value`.
`SamplerContract` matches on the tagged event row and requires the current Rust event's exact
typing, support, cutoff, and relation clauses.

Decomposition is intentionally split into two notions:

- every regular or small decomposition event receives its authoritative variant-specific bound;
  regular uses `max(base / 2, 1)` and small uses `base - 1`; and
- only an event for which the fixed Rust core registered a gadget recomposition relation receives
  a `G * D ≡ input` link. In particular, the current small-decomposition link is recorded only
  under the fixed core's `ExactZero` applicability condition.

The certificate never strengthens a sampler relation beyond the relation actually registered by
the pinned Rust core. Preimage and hash/decomposition links likewise retain their exact event,
scope, argument-substitution, type, and descriptor conditions.

For Tall's universal preimage relation, the public matrix is selector-independent:
`B * K(i) = T(i)`. The public program remains represented through the generic family-lifting API,
but validity recursively proves that its root reaches no program argument and that its evaluation
is identical for any two matching selector arguments. `K(i)` and `T(i)` use the same `Nat` family
selector. Signed integer arithmetic facts elsewhere remain signed.
The certificate must not restate this relation as `B(i) * K(i) = T(i)`.

## 9. Exhaustive index-use semantics without a second analyzer

Index consumers are discovered only by typed registration at the generic APIs that consume an
index: family lookup, `ExplicitElement` operand 0, and the four dynamic `IndexedSlice`
coordinates. Integer-looking nodes are never found by a heuristic scan. Hash tags, scale factors,
comparisons and fixed descriptors are not consumers merely because they are integers.

For each registered use, `IndexUsePlan.index : ExprId` is the sole computation root projected to
Lean. The projection follows the existing typed expression table and uses exactly the same
`evaluate_typed_index` semantics as Rust for `Add`, `Sub`, `Mul`, `Div`, `Rem`, and `Negate`.
The projection uses the typed expression table as the sole expression representation. An
unsupported expression or operation fails closed.

The required domains are the exact nonnegative `FamilyDomain` for family lookup, the exact branch domain for
`ExplicitElement`, `[0, input.rows + 1)` for both dynamic row endpoints, and
`[0, input.columns + 1)` for both dynamic column endpoints.

Every finite integer computation that reaches a registered consumer is projected to one
`IndexUseLut`. Its identity contains owner, canonical consumer, operand, use kind, the exact
`ExprId` computation root, required output domain, fixed parameters, optional group, and ordered
frontier identities and domains. Cross-scope frontiers use a typed `ScopedExpressionRef` and an
explicit composed `ProgramCall` substitution; flattening them into caller ownership is forbidden.

Rust enumerates the complete finite Cartesian frontier in the recorded frontier order and domains
with the pinned evaluator semantics. A closed computation has zero axes, whose Cartesian product
has cardinality one and exactly one row. Every raw tuple and output is checked against the same
Rust typed-index evaluator before emission; a mismatch rejects generation. Prior LUT outputs form
a topologically emitted dependency DAG. Missing or conflicting domains, cycles, conversion
overflow, division by zero, evaluator panic, or evaluation failure reject emission.

Lean checks exact cardinality, reconstructs mixed-radix input tuples, proves the actual tuple is
in-domain, and composes bounded row/subtree proofs that every output satisfies the consumer range.
A production table is never reduced by one monolithic `rfl` or `decide`.

The four dynamic slice LUTs share one frontier in `IndexedSliceLutGroup`. A synchronized predicate
proves order and exact output extents for every row; four independent in-range scans are
insufficient.

This costs `Theta(product(frontier-domain cardinalities))` per distinct index use. There is no row,
byte, time, or domain cutoff, no partial table, and no trusted output range. Lean reconstructs each
row from the typed `ExprId` expression and checks the raw tuple/output correspondence. A
registered use may enumerate every selector in its own dependency frontier; unrelated
uses and selectors are not cross-multiplied. Family values and parallel-loop bodies remain
symbolic and are never duplicated per lane.
Lossless streaming, exact deduplication, or compression may change storage but not the evaluated
tuples.

## 10. Exact semantic and bound replay

The generated sequence of ordinary Lean `have` declarations is the proof plan. Rust-only replay
records are skipped by serialization and have no Lean semantics; there is no `StepId`, predecessor
graph, plan validator, or plan interpreter.

1. Replay local semantic equalities from `evalClosedResidual` or `evalFamilyResidual` to the
   current ordered-product `PolynomialNF` result.
2. Apply only the exact scoped relations registered by the fixed Rust core.
3. Cancel exact signed multiplicities.
4. Bound only surviving bounded terms and the subexpressions needed to justify them.
5. Transfer the bound across `MatrixModEq` using matrix well-formedness.
6. Check the direct strict inequality `2 * p * noise < q`; this closes
   `TallSemantics.Security0Accepted`.

The recorder must retain enough context to render every local have: exact predecessor
polynomials, additive and multiplicative prefix/suffix context, frozen rule identity and
parameters, cancellation coefficients and survivors, coefficient merges, survivor folds, and the
pre-fold final polynomial. Exact terms use a nonempty factor fold; a nonzero term with no factors
rejects. No zero matrix or invented generic identity matrix may serve as a multiplicative unit.

The honest frame lifecycle is replay-complete: frame creation, owner-local cache state, event
history, and frame finalization are recorded in order and are checked against the corresponding
Rust run. The current Tall profiles have zero specialization-cache hits, so no generic
specialization-cache replay or cache-transplant mechanism is part of this design. Any future
reached cache reuse must still emit the consuming frame's own relation, fold, and merge evidence;
renderer-side transplantation across owners and Lean-side normalization search remain forbidden.
Missing owner-local lifecycle events make certificate generation fail closed.

The Lean bound lemmas must reproduce, not approximate, the pinned Rust rules used by the accepted
run. This includes `ExactZero` annihilation, `Large`/missing rejection, scalar broadcast,
polynomial-support factors, constant-polynomial factors, tensor rules, exact matrix inner
dimension, any proved zero-row reduction actually used, CRT reconstruction coefficients, and all
operator-specific transfers in `bound.rs` and `normal_form.rs`. A looser replacement is not
accepted merely because it is conservative; it can change parameter acceptance and is unnecessary
noise overestimation.

The certificate theorem API for `MonomialProduct` is a nonempty fold whose per-step product factor
is one. A surviving nonzero coefficient magnitude is applied by a separate following `Scale`
event. The unrestricted factor of the internal Rust product helper is not part of the certificate
API; every accepted G0 `MonomialProduct` call site uses factor one.

Semantic replay is reached-only. The generator emits theorem applications for the rows actually
present in the residual proof closure, and a Lean compile failure identifies the next missing
semantic case. G2 therefore covers only the encountered Add/Sub/Product/Tensor merges, relation
prefix/source/suffix reconstruction, bound transfer, survivor folds, and
`PreFoldPolynomial → InvocationEnd` chain needed by the selected Tall run. It does not maintain a
whole-workload coverage matrix, a completeness ledger, or lemmas for unreached cases. The
generator remains a proof producer: Lean checks every ordinary theorem application and rejects an
unsupported reached case.

`ThresholdDecode`, `BoundAuthority::Unavailable`, and raw `EventKind::Trapdoor` are rejected before
matrix generation or canonical event projection. Decoder-only expressions are not part of the
residual closure. A reject-tagged row therefore cannot occur in emitted schema-version-6 CPU
evidence.

An unsupported reachable operator, missing relation lemma, surviving `Large`, missing bound, or
unproved side condition rejects emission.

## 11. Recorder, artifacts, and CI

The recorder is opt-in and additive. Observation calls may be added at existing construction,
relation, and bound decision points, but they must not change ordering, interning, facts,
normalization, relation selection, bounds, acceptance, or diagnostics. A differential test runs
the same request with recording off and on and requires identical Rust report bytes and core
counters, excluding recorder-only metrics.

The compared values are semantic report fields and deterministic core counters produced by the
ordinary checker. Elapsed-time diagnostics, RSS/GPU observations, and recorder-only metrics are
excluded because they are operational observations rather than checker semantics.

`docs/correctness/tall-operational-noise-certificate-g0.json` is deterministic G0 review evidence
for source construction and feasibility only. It is not a G3 certificate artifact and cannot be
used as `Source.json` or as an acceptance artifact.

The evidence uses two exhaustive, statically constructed Tall profiles in this order. The
security-0 profile fixes one multiplication, CRT depth 7, `log2(n) = 5`, 28-bit CRT moduli, an
automatically selected 6-bit nested-RNS p basis, 14-bit gadget base, two unreduced
multiplications, scale 64, error sigma 4, and trapdoor sigma 4.578. The security-128 profile fixes
the same values except for CRT depth 20 and `log2(n) = 15`; its reviewed static security lower
bound is 177 bits. These profiles do not read environment variables, run a parameter search, or
invoke an estimator. The file contains pre-gate CPU observations, not a Tall execution result.

Schema version 6 derives the exact `N` rows and descriptor inventory from the same canonical
statement-row projection used by the typed certificate schema. In particular, its event rows
include each reached gadget decomposition in its exact closed or program scope; the expression row
stores only the corresponding event reference. This keeps the expression, program, source, and
event counts under one authority without changing runtime acceptance or the ordinary checker path.

The generator emits the following sharded artifacts into an explicit output directory:

- `Source.json`: the canonical fixed-profile recipe, exact request identity, fixed audited Tall
  constructor parameters, and pinned source/evaluator identities. It is not a complete frozen-bundle
  serialization and contains no normal form, LUT output, bound ledger, or proof;
- `Cert/`: statement rows and local compositional validity witnesses;
- `Proof/`: immutable history and ordinary theorem applications; and
- `Semantic/`: reached semantic theorem applications and the final fixed acceptance composition.

The generated Lean output is a build artifact. Its size (approximately 720 MB for the current
profiles) does not require committing the generated files. The fixed `TallSemantics.Security0Accepted`
definition and the generator are the durable ABI; generated shards are recreated from the source
recipe.

Clean-room regeneration performs two independent complete generations from the same committed
`Source.json` recipe and compares every path-relative generated file byte-for-byte. Both runs must
reconstruct the same request, report, typed residual closure, and fixed acceptance input. Direct
Rust-versus-Source report/core-counter parity is checked before this equality comparison. A digest
is optional audit evidence and never replaces the byte comparison or typed identity checks.

The pre-serialization/render gate must re-derive constant-polynomial and polynomial-support facts
from the authoritative owner/source/family-selected facts and reject any mismatch; recorded facts
are not trusted merely because the recorder emitted them.

CI compiles the fixed acceptance module and rejects `sorryAx`, generated axioms, `native_decide`,
or non-standard axioms reported by `#print axioms`.

## 12. Complexity and implementation gates

Let `N` be the exact logical item count of expression/program/source/event rows in the reached
residual proof closure, `T` the exact emitted proof/context payload logical-item count, and `L` the
exact size of the exhaustive index-use tables in that closure. `T` includes predecessor
polynomials, prefix/suffix context, rule parameters, coefficient merges, survivor folds, and the
final polynomial, not merely the number of `have` declarations. Recording and rendering are
`O(N + T + L)` apart from unchanged Rust checker work. Matrix-family elements and parallel-loop
lanes are not expanded into `N`; each LUT has exactly the product of its recorded frontier-domain
cardinalities as its row count.

Completion evidence uses deterministic exact counts and bytes: Rust report/core-counter parity,
reached closure row counts, relation/bound/frame evidence counts, exact LUT row counts, generated
file counts, canonical encoded bytes, and actual generated-artifact bytes where artifacts exist.
Elapsed time, RSS, `size_of`, benchmark estimates, and runtime/GPU estimates are not deliverables.
The implementation must not add a runtime cutoff, truncate a table, or silently change semantics
to meet a metric.

The gates are:

1. **G0, existing structural boundary:** keep the existing Rust path, SchemaV1, and G0 index-use
   enumeration unchanged. The opt-in projection preserves typed `ExprId`, frontier order, domains,
   and SliceGroup rows, with exact Rust raw-row validation.
2. **G1, fixed Lean core:** build `Core.lean` and `Fixtures.lean` under
   `lean/Mxx/Certificate/OperationalNoise/`, including the toy fixture and the axiom scan.
3. **G2, reached semantic replay:** apply only the semantic theorem families reached by the selected
   run: Add/Sub/Product/Tensor merge, relation prefix/source/suffix reconstruction,
   `BoundTransfer`, `SurvivorFold`, and `PreFoldPolynomial → InvocationEnd`. A missing reached
   theorem is fixed after the Lean compile identifies it; unreached variants and a whole coverage
   ledger are not required.
4. **Security0 fixed acceptance:** generate sharded `Cert/`, `Proof/`, and `Semantic/` outputs from
   the canonical recipe; verify direct-vs-Source report/core-counter parity, reached residual
   closure identity/relation/bound/frame evidence, and compile the fixed
   `TallSemantics.Security0Accepted` endpoint with the allowed axioms.
5. **Security128:** apply the same generator and fixed ABI, then require full generation, fixed
   acceptance compilation, exact deterministic metrics, and two-run clean-room byte equality.

The Security0/Security128 vertical slice and its final evidence gates are complete. Fresh runs have
direct-vs-Source report/core-counter parity; deterministic row/file/count/byte metrics and exact
Rust-versus-generated final-bound equality are recorded; two independent complete generations from
the same committed source recipe are byte-identical; and both Security0 and Security128 fixed
acceptance modules pass Lean validation with the allowed axioms. Passing the conditional Lean
theorem still does not establish the external execution-instantiation obligations in §4.3.
