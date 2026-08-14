# Diamond WE correctness checkpoint

The Diamond WE correctness proof is incomplete. This checkpoint keeps the implemented pieces
reviewable without presenting the unfinished end-to-end theorem as part of the normal build.

## Included in the normal build

- The generated parameterized `ClosedProtocolDecl`.
- The typed certificate identities, expressions, affine facts, recurrence paths, and closed rule
  universe under `lean/Mxx/Certificate`.
- Proof-producing symbolic normalization for the retained theorem-development path.
- DSL semantic anchors resolved directly at freeze time, with no graph-search fallback.
- The checked-in M0 node, transform, loop-bound, and origin audit.
- The generic Rust operational-noise checker over the frozen workflow, exact input contracts,
  registered sampler relations, compact family representations, and decoder acceptance targets.
- Diamond parameter search calls that Rust checker directly and retains its structured report for
  each accepted or rejected candidate.

Building `MxxWe` establishes only that this stable subset type-checks. Rust checker acceptance
establishes the implemented operational bound and decoder inequality for the supplied frozen
workflow and concrete parameter environment; it is not an end-to-end correctness theorem.

## Work in progress

The generic analyzer connects facts to exact denotational execution through typed
workflow traces, artifact origins, family templates, and recurrence paths. The remaining work is to
complete the registered node-rule soundness proofs, workflow and bundle erasure theorems, and the
Diamond endpoint composition. There is no public `MxxWe.Proofs.DiamondWe.correct` theorem yet.

`crates/we/examples/verify_correctness.rs` remains the final theorem gate. It is intentionally not
called by `scripts/verified_build.sh` at this checkpoint and must fail until the public theorem is
completed. No compatibility alias, axiom, `sorry`, or permissive fallback is used to bypass that
gap.

## Review commands

Run the stable checkpoint build with:

```text
cd lean
lake build MxxWe mxx_diamond_derivation_checker
```

The repository-wide generated-source and Rust build gate is:

```text
scripts/verified_build.sh
```

That gate does not claim the unfinished Diamond WE theorem.
