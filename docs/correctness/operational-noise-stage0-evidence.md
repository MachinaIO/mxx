# Stage 0 evidence: Rust operational-noise checker replacement

## Status and purpose

This document freezes the evidence collected before replacing the Lean operational-noise
checker. It is a planning and migration record, not an approval, implementation claim, or
end-to-end correctness result. In particular, **Approval B has not been granted**.

The baseline was recorded on branch `codex/new-noise-simulate-0814` at commit
`ed7f1d1cab80572eda90e7aaae07d9893a7728b5`.

## Baseline size and ownership

The replacement target is deliberately smaller than the current checker. The Stage 0 source
inventory count for the current Lean operational implementation and its Rust runner is 13,557
nonblank source lines. The replacement has a hard implementation cap of 8,134 non-test lines:
eleven Rust modules with a combined 7,900-line working budget, leaving 234 lines of headroom.
This is a cap, not evidence that a replacement has already reached that size.

| Ownership | Current material | Migration action |
| --- | --- | --- |
| Keep | Graph IR, DSL, exact integer types, circuit construction, sampler contracts, generated-protocol emission, and the ordinary Lean proof pipeline | Retain as the execution and proof substrate. |
| Move | Operational request construction, closed target selection, report validation, and diagnostics now coupled to the Lean process | Re-home in the Rust checker at the same public call boundaries where that remains useful. |
| Rewrite | Operational expression analysis, e-class identity, integer-domain transfer, family selection, relation-guided contraction, target bound evaluation, and progress reporting | Implement as the compact fail-closed Rust/egg checker. |
| Delete | `crates/correctness/src/operational_runner.rs`, its generated Lean invocation/cache path, and the active Lean `Mxx.Certificate.OperationalBounds` checker/support after callers have migrated | Delete rather than retain a compatibility fallback. |

The current active ownership is visible in `crates/correctness/src/lib.rs`,
`crates/correctness/src/operational_runner.rs`, `lean/Mxx/Certificate/OperationalBounds.lean`,
and its imported `lean/Mxx/Certificate/OperationalBounds/` modules. The generated toy module
and Diamond checker are migration consumers, not independent operational implementations:
`crates/correctness/lean/MxxCorrectness/OperationalToy.lean` and
`crates/we/lean/MxxWe/DiamondChecker.lean`.

## Current validation boundary

The repository contains focused Lean fixtures and historical completion reports, but these are
evidence about the retired architecture only. They do not approve the replacement.

The current Tall integration source is
`crates/gadgets/tests/test_gpu_tall_bgg_nested_rns_modq_arith.rs`. Its Lean-only parameter
simulation is ignored at the test declaration, and the full GPU round trip is separately ignored.
Neither ignored test was executed in this worktree during Stage 0 because no authorization to run
the long Tall simulation was available. Consequently this document records no passing current
Tall runtime result.

The existing runner still exposes the central failure mode motivating the replacement: it can
return `unsupported_operational_expression` or `unsupported_node` from the Lean checker instead
of completing a valid Tall-shaped graph. Its progress schema also contains counters initialized
to zero in failure/report construction paths. Such placeholder values must not be interpreted as
measured complexity, coverage, or evidence that a branch was visited. The Rust replacement must
report only counters it actually maintains and must not add a field merely to mask that caveat.

## Tall source inventory

The relevant executable construction is not a synthetic checker fixture. The following source
areas build or consume the shapes the new checker must accept generically:

| Area | Evidence | What it establishes |
| --- | --- | --- |
| Nested-RNS lookup programs | `crates/gadgets/src/circuit_gadgets/arith/nested_rns/context.rs` | Each `p_i` registers a modulus, CRT-conversion, and scaled lookup program; lookup size is a concrete per-program contract. |
| Nested-RNS planner metadata | `crates/gadgets/src/circuit_gadgets/arith/nested_rns/poly.rs` | A `NestedRnsPoly` carries `max_plaintexts` and one `p_max_traces` value per active q-level. |
| Tall graph and closed residual | `crates/gadgets/tests/test_gpu_tall_bgg_nested_rns_modq_arith.rs` | The operational target is the residual family; the executable decoder witness is selected from that same family. |
| Protocol/target serialization | `crates/correctness/src/operational_protocol.rs` and `crates/correctness/src/emit_lean.rs` | Current operational target metadata is produced alongside Graph/DSL artifacts. |
| Current checker boundary | `crates/correctness/src/operational_runner.rs` and `lean/Mxx/Certificate/OperationalBounds/` | Unsupported transfer errors arise in the analyzer, not as a Tall-specific protocol node. |

`PublicLutProgram` is registered as a subcircuit program and is expanded through ordinary graph
construction; it is not an atomic noise primitive on the Tall path. A replacement must therefore
recognize the generic lookup subgraph and its declared input domain. It must not hard-code a Tall
node number, a particular program identifier, or a fixture table length. The Stage 0 inventory
requires a zero count of reachable root/direct PubLut operations in the Tall operational scope;
only leaf subcircuit lookup instances are in scope for the proposed lowering.

## Unresolved per-modulus lookup-range blocker

The source currently provides insufficient provenance for one required proof obligation. In
`nested_rns/context.rs`, the per-modulus lookup length is

```text
max(p_i * p_max, p_i * 2 * modulus_count)
```

whereas `NestedRnsPoly` stores `p_max_traces` once per q-level, rather than once per `p_i` lane.
The current global trace-capacity calculation uses the `p_max`-sized case. A global trace bound
does not by itself prove that the input of every smaller-`p_i` lookup is below that lookup's own
length.

This is an integration blocker, not permission for the checker to invent a smaller value or to
use a measured runtime value. Before Approval B and before the lookup lowering is implemented,
the producer must provide an authoritative per-lookup exclusive upper bound, or a reviewed
transfer rule must prove it from existing authoritative metadata. The Stage 0 ledger must record
the producer, every transfer, the final selector upper bound, and the lookup count for each
dynamic lookup. A checker that silently substitutes the full matrix modulus, the global trace
capacity, or a test value would either reject valid Tall graphs or accept an unproved lookup.

## Complexity and precision guardrails for implementation

The replacement is permitted to add generic logging where it is not protocol- or node-specific,
but the logging must be bounded and quantitative. The implementation must preserve the present
loop-asymptotic class: no independent-selector Cartesian expansion, no traversal of every logical
family member merely to obtain a maximum, and no annotation-triggered e-graph rebuild. It must
use canonical e-class identity and fail closed when provenance, relation ownership, range, or
concrete structure is unknown.

Noise remains a hard operational bound, not a label for every large quantity. Selector-only
integer provenance must survive the permitted integer transformations and be rejected at matrix
scales, dimensions, sampler cutoffs, and noise-bound arithmetic. Range metadata must be carried
as an existing-subgraph-call attribute aligned with its complete argument list; this adds neither
a Graph IR primitive nor an occurrence-index side map.

The old checker is retained only until all callers migrate. Once the Rust checker has the focused
fixtures, target integration, and independent review required by the replacement specification,
the old runner and Lean operational checker must be removed in the same migration rather than
left as a second implementation.
