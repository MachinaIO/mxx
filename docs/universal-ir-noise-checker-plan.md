# Universal IR Operational Noise Checker

## Status

The operational checker is implemented and is the parameter-search path used by Diamond witness
encryption. It derives deterministic hard bounds from the emitted IR, evaluates generic decoding
obligations in Lean, and returns an inspectable report to Rust.

This is not yet the final end-to-end correctness theorem. The later proof milestone must prove
soundness of these exact executable definitions; it must not replace them with another analyzer or
a Rust copy of the bound arithmetic.

The normative primitive-by-primitive behavior is listed in
`docs/correctness/operational-protocol-inventory.md`. Historical formula reuse is recorded in
`docs/correctness/operational-affine-reuse-audit.md`. Final implementation and validation evidence
is recorded in `docs/correctness/operational-noise-completion-audit.md`.

## Architecture

The active path is:

```text
Rust DSL graph
  -> frozen IR and mechanically generated derivation
  -> generated Array-backed Lean IR
  -> one-time derivation validation and prepared node-order metadata
  -> flat operational facts and generic obligations
  -> exact Lean evaluation for one or more parameter requests
  -> report consumed by Rust parameter search
```

Rust owns graph construction, freezing, hashing, process invocation, candidate parallelism, and
report parsing. Lean owns all matrix-bound, sampler-bound, recurrence, and decoder-threshold
arithmetic. Rust does not maintain a second noise formula.

The Rust runner compiles each emitted workflow and derivation into a content-addressed prepared
module. Its cache key includes the generator, protocol source, workflow, derivation, toolkit, and
Lean-version hashes. Within one checker process, `prepareWorkflowOperational` validates the
derivation and resolves input indices, definition indices, and attachment buckets exactly once;
all parameter requests then call `evaluatePreparedWorkflowOperational` against that prepared
value. Reusing the module avoids re-elaborating the generated IR, while batching requests avoids
repeating preparation inside one process. Different candidate graphs still require distinct
prepared modules.

The active public module is `Mxx.Certificate`, which imports
`Mxx.Certificate.OperationalBounds`. It does not import the retired whole-graph symbolic analyzer,
an expression arena, or symbolic loop unrolling. Proof-oriented source retained for the later
correctness milestone is not part of the operational checker executable.

## Flat matrix facts

Each matrix value is represented as a sum of ordered products:

```text
term       = signed integer coefficient * ordered factor list
polynomial = sum of terms
```

The signed integer coefficient records only additive multiplicity. Matrices, bounded secrets,
dynamic scalar values, selectors, public matrices, and other multiplicative values are factors.
Factor order is semantic and is never sorted.

For example:

```text
s * A - s * A
  = +1 * [s, A] - 1 * [s, A]
  = 0
```

Cancellation requires complete structural equality of the ordered products, including origins,
types, transforms, product modes, and protected relation metadata. Equal names, equal numerical
bounds, or an expectation that two independently created values should be equal never establishes
cancellation.

Every factor has a role derived from its checked source:

- `bounded` means the factor has a deterministic support bound;
- `large` means the factor is an exact signal/public carrier whose smallness is not assumed.

A term with no Large factor is noise. A term with one or more Large factors is signal. Multiple
Large factors remain an ordinary signal product; their count is not an error and does not cause a
fallback classification.

## Deterministic hard bounds

Only worst-case arithmetic is used. For ordinary polynomial-matrix multiplication:

```text
effective_inner = left.columns - right.known_zero_rows.getD(0)
ring_factor = 1              if either input is constant-polynomial
              ring_dimension otherwise
bound = effective_inner * ring_factor * left_bound * right_bound
```

The checker retains the deterministic `is_constant_polynomial` and `known_zero_rows` behavior from
the previous simulator. It does not use CLT, square-root concentration, dependency-disjointness,
or probabilistic independence. Addition uses the triangle inequality after exact cancellation.

Sampler facts use explicit node cutoffs. A Gaussian or sampled preimage is bounded only by the
cutoff carried by its node and justified by the corresponding bounded-sampler contract. Randomness
alone never supplies a hard bound. Full uniform residues and public hash values are Large.

## Normalization and compression

Every arithmetic operation follows this order:

1. canonicalize signed integer coefficients;
2. merge completely identical ordered products;
3. remove exactly zero coefficients;
4. exhaustively apply checked preimage and decomposition relations;
5. compress consecutive unprotected bounded factors;
6. combine remaining bounded-only terms into one bounded noise summary;
7. place terms in a deterministic outer order without reordering product factors;
8. enforce explicit analysis-size limits.

Compression is internal analysis normalization. It is not an IR node, DSL operation, serialized
construct, or user-visible reinterpretation. Relation owners, exact-one indicators, endpoint
identities, and origin-preserving artifacts remain protected until their consuming rule has run.

At a sequential-loop boundary, the residual bounded contribution is one fixed-size summary. The
number of terms and factors therefore does not grow with the iteration count.

## Preimage and decomposition relations

Relations are attached to the exact operand that owns them and are checked only modulo the
polynomial residue ring. A preimage relation is:

```text
B * K = P  (mod R_q)
```

or, when the target itself has an operational expansion:

```text
B * K = S' * P + E  (mod R_q)
```

At a fixed level and source state, `B` is shared across the Boolean branches. The branch chooses a
different preimage and target:

```text
B * K_0 = P_0  (mod R_q)
B * K_1 = P_1  (mod R_q)
```

The checker therefore records one identical source-public identity for both relations. It does not
invent `B_0` and `B_1`. The two relation owners and target identities remain distinct.

Rewriting requires the bare source immediately followed by its bare relation owner, with matching
source, target, type, modulus, ring dimension, and backend layout. For example:

```text
[s, B, K]
  -> [s, S', P] + [s, E]
```

The all-bounded `[s, E]` contribution is then compressed into the noise summary. The relation is
never strengthened to an equality over the integers.

Gadget decomposition follows `docs/correctness/gadget-decomposition-semantics.md`. Backend-owned
layout descriptors are part of the request identity, and the checker validates them instead of
reconstructing a second layout.

## Structural operations and selection

Transpose, slice, concat, reshape, tensor, coefficient extraction, CRT recomposition, and packing
have explicit transfer rules. A transform that cannot preserve signal/noise separation returns a
specific error instead of hiding the value in a new opaque factor.

Dynamic selection uses a production- and scope-namespaced exact-one indicator domain. Bounded
contributions are summed within each branch and combined with a maximum across mutually exclusive
branches. Unrelated selection domains are added conservatively.

For the direct protocol-message encoding chain
`Boolean input -> BoolToInt -> Select(zero, nonzero constant carrier)`, an owning-crate attachment
names the four frozen wires. Lean validates the complete local chain before grouping the selected
value as one exact Large carrier. The attachment cannot declare a bound or role. Toy uses this rule
to construct the executable residual `ciphertext - encoded_message`, and a corrupted-carrier
fixture proves that a false grouping hint is rejected.

For `FamilyGetDynamic`, successful executable access is part of the operational premise. An
interval that overlaps the valid family range is analyzed by conservatively joining every possible
valid element. An interval wholly outside the family range is rejected. This avoids reconstructing
branch-guard correlations in the checker while preserving a sound bound conditional on successful
runtime evaluation.

## Nested scopes and loops

Subgraph calls and parallel loops analyze the checked child scope in node order. A parallel body is
analyzed once and its result is reused for all lanes; frozen binders and lane indices preserve the
required identities.

Sequential loops also analyze the body once. They produce a fixed-size numeric transition whose
inputs are the previous carried numeric slots and whose outputs are committed simultaneously. Lean
then repeats only that numeric transition for the evaluated iteration count. It does not unroll
matrix expressions or construct iteration-proportional symbolic state.

For the usual BGG recurrence:

```text
c_i       = a_i * B_i + N_i
B_i * K_i = S_i * P_(i+1) + E_i  (mod R_q)
c_(i+1)   = (a_i * S_i) * P_(i+1) + (a_i * E_i + N_i * K_i)
```

the parenthesized bounded contribution is reduced to one noise summary at every iteration. The
numeric transition updates the signal-coefficient bound, noise bound, and total bound from the old
state simultaneously.

## Primitive coverage

Coverage is enforced at three boundaries:

1. Rust-to-Lean emission matches every Rust `NodeKind` and nested variant exhaustively.
2. Lean `operationalTransferClass` matches every Lean `NodeKind` and nested variant exhaustively.
3. `docs/correctness/operational-protocol-inventory.md` is compile-time checked for a row matching
   every current emitted variant.

Adding a new constructor therefore breaks compilation or the inventory test until an explicit
transfer or normal rejection is added. There is no implicit successful fallback. Ordinary nodes,
constants, samplers, relations, transforms, scalar/control operations, families, subgraph calls,
parallel loops, and sequential loops all have explicit handling.

## Generic checker report

The operational traversal derives generic obligations from the graph. The decoder obligation is:

```text
2 * plaintext_modulus * noise_bound < ciphertext_modulus
```

It is evaluated using exact integers; equality rejects. The pure Lean report contains the output
facts, obligations, acceptance status, and stable rejection reason. The IO caller measures and logs
wall-clock time around the report evaluation. Application code may select which decoder output is
the protocol endpoint, but it does not supply a replacement noise formula.

Diamond WE parameter search invokes this report for every candidate. Independent candidate checks
may run in parallel, while each Lean checker invocation remains deterministic. The selected
candidate is then used by the GPU estimator and by the real runtime round-trip path.

## Validation boundary

The operational implementation is validated by:

- Lean fixtures for flat cancellation, multi-Large products, relation rewriting, shared preimage
  source identity, branch-max selection, transforms, samplers, scalar/control rejection, nested
  scopes, families, and fixed-size recurrence;
- Rust tests for exhaustive emission/inventory coverage and generated-source freshness;
- owner-crate unit tests for `mxx-correctness`, `mxx-bgg`, `mxx-gadgets`, and `mxx-we`;
- the ignored GPU Diamond integration test, which performs parameter search, GPU estimation,
  encryption, decryption, and message round trip in one execution.

Passing the operational checker proves that the implemented deterministic bound calculation accepts
the candidate. It does not yet constitute the final Lean theorem connecting every operational fact
to runtime semantics and the application-level Witness Encryption correctness statement.
