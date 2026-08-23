# Lean-Provable Operational Noise Certificates

## 1. Status and authority

This is a design specification. It does not authorize implementation yet.

The operational-noise Rust implementation at the pinned source revision is the semantic and
performance authority. In particular, the behavior of `arena.rs`, `program.rs`, `lower.rs`,
`normal_form.rs`, `bound.rs`, `relation.rs`, `report.rs`, and `simulation.rs` is not changed by
this design. Certificate support may add opt-in observation hooks, a recorder, serialization,
proof rendering, and fixed Lean definitions. Any proposed change to the existing Rust core
semantics requires separate review and explicit approval.

If this document and the pinned Rust core disagree, certificate emission fails until the
certificate or Lean replay is corrected. It must not silently reinterpret the Rust result.

The first target is one accepted tall-BGG nested-RNS operational-noise parameter set. Rerunning the
pinned checker must resolve the request to exactly one `ResolvedAcceptanceTarget` whose kind is
`ResolvedDecoderKind::Threshold`. Canonical projection takes the plaintext modulus `p` from that
kind, the ciphertext modulus `q` from `ResolvedAcceptanceTarget.ciphertext_modulus`, and the
residual root from `ProductionRoots.residual`. The certificate proves the strict operational-noise
condition; it does not specify or prove any runtime decoder output. Parameter-family theorems are
out of scope.

## 2. Required guarantees

The design must provide all of the following:

1. The generated theorem has one fixed, non-generated type.
2. Lean checks the residual semantics, exact cancellations, surviving-noise bound, and strict
   operational inequality `2 * p * noise < q` with the kernel.
3. `native_decide`, `sorry`, generated axioms, protocol-specific trusted code, node-number
   shortcuts, debug-string identity, and fixture-specific cases are forbidden.
4. Statement data is deterministically projected from a canonical frozen protocol bundle and
   request. Proof schedules and proof terms cannot change the statement.
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
- the Lean expression/program interpreter;
- the tagged input and sampler contracts;
- `RowTable`, `Cert.Valid`, `Cert.wellFormed`, and their shared reflection lemmas;
- the fixed operational claim;
- the additive Rust source-identity sidecar, canonical projection, and serializer.

The existing Rust normalizer, relation search, bound search, proof schedule, and proof renderer are
untrusted proof producers. A mistake in them must fail Lean proof checking.

Statement-bearing certificate rows can change the theorem subject. The source-selection and
source-to-certificate projection boundary is therefore audited trusted code; kernel checking alone
does not prove that the selected source is the deployment intended by an operator.

### 4.2 Protocol-dependent audited data

The canonical source artifact contains exactly the complete output of the single canonical
frozen-bundle serializer introduced for this feature, the exact `OperationalCheckRequest`, and the
pinned source/evaluator versions. This serializer is a wire-format boundary, not a second execution
semantics for `ProtocolDecl` or `ClosedProtocolBundle`; it may represent their data without
reimplementing their behavior. The request already
contains `target_id` and the parameter environment, so neither is duplicated as another source
field. Canonical projection reruns the pinned checker and requires exactly one successfully
resolved `ResolvedAcceptanceTarget`. Its kind must be
`ResolvedDecoderKind::Threshold { plaintext_modulus }`; this value is `p`, its
`ciphertext_modulus` is `q`, and `ProductionRoots.residual` is the residual root. A
`BooleanInterval` target, target-resolution failure, missing or multiple resolved targets, or any
of the following mismatches rejects certificate emission: the resolved `target_id` differs from
the request's `target_id`; the projected `q` differs from the residual ring modulus; or the
projected `p` or `q` differs from the values used by the Rust threshold acceptance report. The
clean-room claim below is not available until this serializer is complete at G3. A digest may be
included as an audit aid but is never semantic identity.

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
`OperationalClaim checkedCert`, the operational inequality, family-selector domains, modulus/ring
agreement, and certificate validity are proved facts, not remaining assumptions. There is also no
residual-root argument premise.

Applying the accepted operational theorem to concrete assignments leaves exactly these Lean
premises:

```text
InputContract checkedCert inputs
SamplerContract checkedCert inputs samplers
```

For a real execution, the operator must additionally establish the following bridge outside the
Lean claim:

```text
DeploymentMatches(run, checkedCert)
  := canonicalSource(run.bundle, run.request, run.sourceRevision, run.evaluatorRevision)
     = the accepted Source.json paired with checkedCert

InputsInstantiate(run, inputs)
  := for every SourceAccess a in the residual proof closure,
       inputs(a) = the value actually supplied by run at a

SamplersInstantiate(run, samplers)
  := for every event occurrence e and owner invocation o in the residual proof closure,
       samplers(e, o) = the value actually produced by run at (e, o)
```

`InputContract` then proves the recorded type, range, coefficient, and support facts about those
equal input values. `SamplerContract` proves the recorded type, cutoff, support, and exact
relations about those equal event values. If no concrete assignments satisfying both contracts
can be exhibited, the implication may be logically true but certifies no real execution.

Lean compilation also remains relative to two audited trusted-code propositions: the fixed Lean
value/matrix/interpreter semantics match the pinned Rust operational-noise semantics, and the
trusted canonical projection reruns the pinned checker and maps the accepted `Source.json` to the
statement-bearing certificate rows, including the uniquely resolved `p`, `q`, and residual root,
without changing their meaning. These are trust-boundary obligations, not hypotheses that a
generated proof may introduce.

## 5. Fixed theorem types

The fixed Lean module defines the complete types below. Generated code cannot redefine them.

```lean
structure CheckedCert where
  val : Cert
  valid : val.Valid

def OperationalClaim (cert : CheckedCert) : Prop :=
  ∀ (samplers : SamplerAssignment) (inputs : InputAssignment),
    InputContract cert inputs →
    SamplerContract cert inputs samplers →
    match cert.val.residualRoot with
    | .closed root =>
        2 * cert.val.plaintextModulus *
            maxCenteredCoefficientNorm
              (evalClosedResidual samplers inputs cert root) <
          cert.val.ciphertextModulus
    | .family family domain =>
        ∀ (selector : Nat), domain.Contains selector →
          2 * cert.val.plaintextModulus *
              maxCenteredCoefficientNorm
                (evalFamilyResidual samplers inputs cert family selector) <
            cert.val.ciphertextModulus
```

`ResidualRoot` mirrors the Rust root classification exactly: it is either a closed matrix
expression or a one-argument matrix family. Rust's `FamilyDomain` is an exact half-open interval
with `u64` endpoints; Lean stores the same endpoints as `Nat`. A family selector is only a `Nat`
and must satisfy `domain.Contains selector`. A closed root has no caller-supplied arguments, and a
family root is bounded for every selector in this exact domain. Root arguments are derived from
this checked root structure rather than supplied as an untyped external list.

`Cert` stores `plaintextModulus` and `ciphertextModulus` directly. `Cert.wellFormed` requires
`plaintextModulus > 0`, `ciphertextModulus > 0`, and a closed matrix or matrix-family residual over
exactly `R_q` and the certificate ring dimension. The trusted canonical projection takes `p` from
`ResolvedDecoderKind::Threshold`, `q` from `ResolvedAcceptanceTarget.ciphertext_modulus`, and the
root from `ProductionRoots.residual`; generated proof data cannot choose them. Runtime decoding is
outside the theorem.

`OperationalClaim` is the only certificate-specific mathematical endpoint. For a closed root it
proves one strict inequality. For a family root it proves the same inequality symbolically for
every selector contained in the exact half-open domain. Equality is rejected, and the direct product
inequality must not be replaced with a condition involving truncated integer division.

The fixed theorem `Cert.Valid.wellFormed` derives `Cert.wellFormed cert = true` without reducing
the entire production table. `Cert.Valid` and `Cert.wellFormed` are two proof interfaces to the
same fixed row predicates; they must not be maintained as independent validation rule sets.

A fixed acceptance module checks the generated proof at the fully qualified type
`OperationalClaim checkedCert` and runs `#print axioms` on its wrapper theorem. It does not derive
or check a runtime decode result.

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
second source-ID namespace.

`EventRow` is one tagged sum for uniform, Gaussian, deterministic hash, trapdoor-public,
preimage, and gadget-decomposition events. Every row stores its owner scope, signature, output
type, and complete kind-specific descriptor.

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

The required domains are the exact nonnegative `FamilyDomain` for family lookup, the exact branch domain for
`ExplicitElement`, `[0, input.rows + 1)` for both dynamic row endpoints, and
`[0, input.columns + 1)` for both dynamic column endpoints.

Every finite integer computation that reaches a registered consumer is projected to one
`IndexUseLut`. Its identity contains owner, canonical consumer, operand, use kind, computation
root, required output domain, fixed parameters, optional group, and ordered frontier identities
and domains. Cross-scope frontiers use a typed `ScopedExpressionRef` and an explicit composed
`ProgramCall` substitution; flattening them into caller ownership is forbidden.

Rust enumerates the complete finite Cartesian frontier with the pinned evaluator semantics. A
closed computation has zero axes, whose Cartesian product has cardinality one and exactly one
row. Prior LUT outputs form a topologically emitted dependency DAG. Missing or conflicting
domains, cycles, conversion overflow, division by zero, evaluator panic, or evaluation failure
reject emission.

Lean checks exact cardinality, reconstructs mixed-radix input tuples, proves the actual tuple is
in-domain, and composes bounded row/subtree proofs that every output satisfies the consumer range.
A production table is never reduced by one monolithic `rfl` or `decide`.

The four dynamic slice LUTs share one frontier in `IndexedSliceLutGroup`. A synchronized predicate
proves order and exact output extents for every row; four independent in-range scans are
insufficient.

This costs `Theta(product(frontier-domain cardinalities))` per distinct index use. There is no row,
byte, time, or domain cutoff, no partial table, no trusted output range, and no AST fallback.
Lossless streaming, exact deduplication, or compression may change storage but not the evaluated
tuples. A registered use may enumerate every selector in its own dependency frontier; unrelated
uses and selectors are not cross-multiplied. Family values and parallel-loop bodies remain
symbolic and are never duplicated per lane.

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
6. Check the direct strict inequality `2 * p * noise < q`; this closes `OperationalClaim`.

The recorder must retain enough context to render every local have: exact predecessor
polynomials, additive and multiplicative prefix/suffix context, frozen rule identity and
parameters, cancellation coefficients and survivors, coefficient merges, survivor folds, and the
pre-fold final polynomial. Exact terms use a nonempty factor fold; a nonzero term with no factors
rejects. No zero matrix or invented generic identity matrix may serve as a multiplicative unit.

Cache reuse must remain replay-complete. On every normalization or specialization cache hit, the
recorder re-emits the relation/gadget applications, finite survivor folds, and implicit coefficient
merges for the exact consuming owner and expression, in chronological order. Renderer-side rule
transplantation across owners and Lean-side normalization search are forbidden because they are
sign- and scope-unsafe. Missing owner-local events make certificate generation fail closed.

The Lean bound lemmas must reproduce, not approximate, the pinned Rust rules used by the accepted
run. This includes `ExactZero` annihilation, `Large`/missing rejection, scalar broadcast,
polynomial-support factors, constant-polynomial factors, tensor rules, exact matrix inner
dimension, any proved zero-row reduction actually used, CRT reconstruction coefficients, and all
operator-specific transfers in `bound.rs` and `normal_form.rs`. A looser replacement is not
accepted merely because it is conservative; it can change parameter acceptance and is unnecessary
noise overestimation.

At G0, an exhaustive coverage matrix is produced for every operator, transform, sampler kind,
relation kind, and bound rule in the exact Tall residual proof closure. Each row names the Rust
source location, fixed Lean semantics lemma, fixed Lean bound/relation lemma, or deliberate
fail-closed result. Decoder-only rows and `ThresholdDecode` lemmas are outside this matrix.
Differential fixtures cover signed divide/remainder, scalar broadcast, tensor, CRT recomposition,
tag encoding, regular/small decomposition, slice/view routing, and polynomial packing.

An unsupported reachable operator, missing relation lemma, surviving `Large`, missing bound, or
unproved side condition rejects emission.

## 11. Recorder, artifacts, and CI

The recorder is opt-in and additive. Observation calls may be added at existing construction,
relation, and bound decision points, but they must not change ordering, interning, facts,
normalization, relation selection, bounds, acceptance, or diagnostics. A differential test runs
the same request with recording off and on and requires identical Rust report bytes and core
counters, excluding recorder-only metrics.

The committed artifacts are:

- `Source.json`: the complete output of the single canonical frozen-bundle serializer, exact
  request, and pinned version identities; no duplicated target/environment fields, normal form,
  LUT output, bound ledger, or proof. This serializer is added at G3 and is not a second execution
  semantics for the existing bundle types;
- `Cert.lean`: statement data and local compositional validity witnesses;
- `Proof.lean`: ordinary `have` declarations ending in the proof term at the fixed
  `OperationalClaim`; and
- the fixed acceptance module introduced at G3.

After G3, clean-room regeneration receives only the complete `Source.json` emitted by the canonical
frozen-bundle serializer. It reconstructs the current checker run,
recomputes every row and proof in the residual proof closure, omits decoder-only data, and
byte-compares fresh `Cert.lean` and `Proof.lean` with the committed files. Unknown derived fields in
`Source.json` reject regeneration. First-run publication requires an explicit output directory and
explicit source revision, uses a synchronized staging directory and atomic publish, and rejects
symlinks, nonempty targets, and name mismatches. It never infers a dirty worktree's revision from
`HEAD`.

CI compiles the fixed acceptance module and rejects `sorryAx`, generated axioms, `native_decide`,
or non-standard axioms reported by `#print axioms`.

## 12. Complexity and implementation gates

Let `N` be the total size of expression/program/source/event rows in the residual proof closure,
`T` the total emitted proof/context payload size, and `L` the total size of exhaustive index-use
tables in that closure. `T` includes every serialized predecessor polynomial, prefix/suffix
context, rule parameter, coefficient merge, survivor fold, and final polynomial, not merely the
number of `have` declarations. Certificate recording and rendering are `O(N + T + L)` time and
space, apart from the unchanged Rust checker work. No matrix-family or parallel-loop cardinality
is included in `N`. For each LUT, its row count is exactly the product of its finite
frontier-domain cardinalities, and `L` includes the serialized tuple and proof payload per row.

Before implementation, G0 computes or measures, without full Lean certificate generation, the
exact `N`, every frontier product, and exact `L` payload for both the security-0 and exact
security-128 Tall sources. It estimates `T` payload, artifact bytes, and peak memory, and compares
them with the current checker. G3 measures exact `T`, artifact bytes, and peak memory at security 0;
G4 measures them exactly at security 128. If the
production estimate is infeasible, the design returns to review; the implementation must not add
a runtime cutoff, truncate a table, or silently select another semantics.

The phases are:

1. **G0, design feasibility:** complete residual-closure coverage matrix, exact `N`, `L`, and
   frontier products, estimated `T`/artifact bytes/peak memory, zero-axis and synchronized-slice
   LUT tests, and kernel spikes for fuel-stable haves and balanced row-local validity. Failure
   returns the design to review before trusted code is built.
2. **G1, fixed Lean core:** place the new project under `lean/lean-toolchain`,
   `lean/lakefile.toml`, `lean/Mxx/Certificate/OperationalNoise/Core.lean`, and
   `lean/Mxx/Certificate/OperationalNoise/Fixtures.lean`. Hand-prove toy closed and family-root
   certificates with different sampler values at different indices, exact strict inequalities,
   clean axioms, malformed-data rejection, and independent design review. The gate is
   `cd lean && lake build Mxx.Certificate.OperationalNoise.Fixtures` followed by the axiom scan.
   `AcceptedCertificates` is not required before G3.
3. **G2, replay library:** every G0 coverage row has a checked lemma or deliberate rejection;
   exact current Rust bounds are reproduced without overestimation; unsupported cases fail closed.
4. **G3, security 0:** opt-in recorder and deterministic generator complete; recording on/off is
   Rust-semantics-identical; cache-hit replay is owner-local and complete; clean-room regeneration
   is byte-identical; fixed Lean acceptance compiles.
5. **G4, security 128:** full generation and
   kernel checking complete; elapsed time, peak memory, artifact size, exact LUT counts,
   and Rust-versus-Lean differential results are recorded; the trust inventory receives
   independent review.

Passing a Rust compile or unit-test gate is not certificate acceptance. Passing the conditional
Lean theorem is not by itself evidence that one runtime execution satisfied the external
applicability obligations in §4.3.
