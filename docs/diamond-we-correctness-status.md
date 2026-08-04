# Diamond WE correctness checkpoint

The Diamond WE correctness proof is incomplete. This checkpoint keeps the implemented pieces
reviewable without presenting the unfinished end-to-end theorem as part of the normal build.

## Included in the normal build

- The generated parameterized workflow, statement, and structural certificate.
- The executable structural verifier and its soundness theorem.
- The protocol-independent hard-bound recurrence and Boolean algebra lemmas.
- The executable Diamond family parameter checker in `MxxWe.DiamondFamilyChecker`.
- The standalone `mxx_diamond_checker` executable used by Rust parameter search.

Building `MxxWe` establishes only that this stable subset type-checks. Checker acceptance proves
the checked parameter inequalities; it is not an end-to-end correctness theorem.

## Work in progress

The following source modules are intentionally excluded from the `MxxWe` root until their proof
obligations are closed:

- `MxxWe.Certificate.InputInjectionExecutionBridge`
- `MxxWe.Certificate.InputInjectionWorkflowBridge`
- `MxxWe.Certificate.BooleanExecutionBridge`
- `MxxWe.Certificate.EncryptionExecutionBridge`
- `MxxWe.Certificate.TransitionSelectorExecutionBridge`
- `MxxWe.Certificate.ProducerLoopExecutionBridge`
- `MxxWe.Proofs.DiamondWeFamily`

These modules connect certificate-verified graph references to the exact denotational execution of
the linked encryption and decryption stages. The remaining blocker is construction of the closed
execution evidence required by the final family theorem. The proof source deliberately exposes
only a private conditional theorem; there is no public `diamondWeFamily_correct` theorem yet.

`crates/we/examples/verify_correctness.rs` remains the final theorem gate. It is intentionally not
called by `scripts/verified_build.sh` at this checkpoint and must fail until the public theorem is
completed. No compatibility alias, axiom, `sorry`, or permissive fallback is used to bypass that
gap.

## Review commands

Run the stable checkpoint build with:

```text
cd lean
lake build MxxWe mxx_diamond_checker
```

Run individual work-in-progress modules to inspect their remaining goals without changing the
normal build boundary. For example:

```text
cd lean
lake build MxxWe.Certificate.TransitionSelectorExecutionBridge
```

The repository-wide generated-source and Rust build gate remains:

```text
scripts/verified_build.sh
```

That gate does not claim the unfinished Diamond WE theorem.
