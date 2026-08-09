# Operational Correctness IR Audit

## Purpose

This report records the executable IR surface currently consumed by the Lean operational
hard-bound checker. It is an implementation audit, not an end-to-end correctness proof.

The active path consists of:

1. generated frozen IR and a canonical derivation program;
2. structural derivation validation in `lean/Mxx/Certificate/Derivation.lean`;
3. one forward operational pass in `lean/Mxx/Certificate/OperationalBounds.lean`; and
4. the owning-crate checker in `crates/we/lean/MxxWe/DiamondChecker.lean`.

The active path does not import the proof-oriented graph analyzer, an expression arena, a graph
search layer, or a reconstructed execution trace. It evaluates each frozen scope in node order,
looks up only exact prior `WireRef` operands, and descends through the exact definition named by a
call or loop node.

## Reproduction

Regenerate the checked-in modules and run the focused gates with:

```sh
MXX_REGENERATE_CORRECTNESS=1 cargo run -p mxx-correctness --example emit_correctness
MXX_REGENERATE_CORRECTNESS=1 cargo run -p mxx-we --example emit_correctness
(cd lean && lake build Mxx.Certificate.OperationalBounds MxxWe.DiamondChecker)
cargo test -p mxx-we exact_cutoffs_and_lean_checker_match_rust --lib -- --nocapture
```

Both generated protocol modules currently identify `mxx-correctness-emitter-v6`.

| Generated protocol | Workflow hash | Derivation hash |
|---|---|---|
| Toy example | `eec6cc84a07b935c537fee71c5f133e7e371b21f39e3757def3b287cbf269635` | `1eb7ee1d85bf85dc59fea9e4e198e1cfa94df8fd0bf8168d408a754514520933` |
| Diamond WE family | `b0cb1761132a3683375bc77da27bdac62b157142952ecaa1ee7b00c9aaf70c38` | `9121b6753e53963269c95afdea191a014643ddad27e056fbb57ad63cda19f719` |

The build scripts recompute source and toolkit hashes. A stale generated module rejects the Rust
build and prints the exact regeneration command. The parameter-checker request separately hashes
the complete candidate and ordered gadget-layout metadata; the checker echoes that hash with its
answer.

## Frozen node surface

The generated Toy and Diamond programs currently exercise these executable node families:

- scalar and control: constants, `EvaluateInt`, `IntBinary`, `IntCompare`, `BoolToInt`,
  `BitExtract`, `ExtractCoefficient`, `Dimension`, `Select`, and threshold decode;
- matrices: zero, identity, constant, add, subtract, negate, scale, multiply, concat, slice,
  reshape, Gaussian sampling, uniform-interval sampling, and hash sampling;
- lattice relations: gadget matrices, gadget decomposition, trapdoor sampling, and preimage
  sampling;
- structure: inputs, static and dynamic family access, parallel loops, and sequential loops.

The operational evaluator has a closed explicit transfer or normal rejection for every emitted
operation. A matrix-producing node never acquires a decomposition or preimage relation by fallback;
the centered residue cap is used only by the explicit leaf or transform rules that require it.

The derivation checker requires exactly one step per frozen node in canonical order. It rejects a
missing, duplicated, reordered, forward, or operand-mismatched instruction before operational
evaluation. For multiplication, Rust may select either the conservative product rule or the
relation-consuming rule. Lean validates the selected rule against the actual right operand and
the relation attached to that exact operand fact.

## Identity and relation transport

Operational identities are opaque provenance values, not symbolic matrix expressions.

- local values are namespaced by root program, nested scope, and wire;
- a protocol input retains one `ProtocolInputId` across workflow stages;
- a static external-family element additionally retains its selected index;
- a dynamic family access receives the actual get node as a conservative fresh origin;
- deterministic hashes retain their key origin, exact tags, matrix type, root parameter
  environment, and the complete nested parameter-binding domain chain; and
- static loop extraction substitutes the selected loop index through both bounds and hash
  identity metadata.

Artifact inputs receive the producer's actual output fact and only rebind the consumer subject.
Thus a relation produced in one workflow stage crosses an explicit artifact edge without graph
search and without changing its semantic origin.

The active relation payload is intentionally small and nonrecursive. It records the exact public
identity, target origin, resolved matrix parameters, hard-bound expression, canonical range, and
mode-specific gadget metadata. Available relations mean equality only in `R_q`:

```text
public * decomposition = input  (mod R_q)
public * preimage      = target (mod R_q)
```

No rule strengthens either relation to integer equality.

## Loop behavior

Parallel loops are analyzed once as a fact template. Loop-dependent numeric parameters retain a
closed domain description; extrema are evaluated over only the referenced loop coordinates.
Static extraction substitutes the exact selected index. Dynamic extraction conservatively joins
all possible bounds and drops matrix relations.

Sequential loops also analyze the body once. Relation-free carried matrix maxima become typed
numeric state references. Phase B iterates only the fixed-size numeric transition, with all carried
slots updated simultaneously. Nested sequential loops use a depth-indexed state stack, so an inner
carried slot cannot shadow an enclosing carried slot. The expression size depends on the body,
not on the concrete iteration count.

A decomposition or preimage relation may be created from a carried target and consumed within the
same body execution. A relation itself may not escape as a carried value. Zero-count, simultaneous
multi-slot update, nested recurrence depth, escaped state reference, and relation-carry rejection
are checked by Lean fixtures in `OperationalBounds.lean`.

Scalar carried values that do not contribute a matrix magnitude retain only their fixed structural
schema in the operational noise pass. Matrix selection remains conservative because dynamic
selection analyzes every possible branch bound rather than trusting the scalar's inferred value.

## Sampler and layout boundary

Every request supplies a complete `GadgetLayoutDescriptor`. Lean validates its CRT moduli, base,
regular and small digit counts, smallest CRT modulus, ring dimension, and resolved matrix tuple.
Missing, ambiguous, invalid, and mismatched descriptors reject distinctly.

The runtime hash backend receives the validated explicit base and digit count. Decomposed hash
sampling is checked against direct decomposition of the corresponding plain hash query; it does
not silently derive a backend-default layout. Small decomposition obtains its unsigned canonical
digit range from the sampler contract, independently of its centered hard bound.

The local Lean soundness layer currently proves the Gaussian and preimage hard-support rules,
the sampled trapdoor public/private pairing, ordinary matrix add/subtract/negate rules, the gadget
decomposition modular relation and hard bound, and the independent small-digit canonical range.
It also proves that a decomposed-hash query matching a plain query receives exactly the direct
decomposition relation and digit bound. The executable hash inversion lemma retains every trailing
integer operand in its actual argument order; it no longer substitutes an empty trailing-tag list.
No Lean axiom is introduced; backend sampler properties remain explicit fields of
`MxxBoundedSamplerContract`.

## Current acceptance boundary

The focused Diamond parameter-search test confirms that:

- generated derivations validate;
- encrypt and decrypt stages are evaluated through their exact artifact wiring;
- nested loops and relation-consuming operations complete in the operational pass;
- an accepted small candidate reaches the Lean checker; and
- candidates with a deliberately invalid hard cutoff are rejected.

This does **not** yet establish the final protocol theorem. The Diamond endpoint currently applies
an explicitly named operational estimate after graph-derived hard-bound evaluation. The later
correctness task must prove that every accepted report implies the runtime decoder returns the
ideal message under the reviewed sampler contracts. Until that bundle theorem is complete,
checker acceptance must be described as an operational parameter filter, not as proved protocol
correctness.
