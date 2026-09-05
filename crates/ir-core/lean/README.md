# Frozen-IR Lean extraction fixtures

Build the generic IR support package from this directory:

```text
lake build
```

The generated fixtures import both `MxxIR` and the runtime package.  Generate one from the
repository root, then check it from `crates/runtime/lean` with the IR package on `LEAN_PATH`:

```text
cargo run -p mxx-ir-core --example emit_sampler_fixture -- /tmp/mxx_generated_sampler.lean
LEAN_PATH=<repository>/crates/ir-core/lean/.lake/build/lib/lean lake env lean /tmp/mxx_generated_sampler.lean
```

The same commands apply to `emit_gadget_fixture` and to `emit_preimage_fixture`, whose second
argument selects `small` or `wide`.  The fixtures prove generated relation consequences only;
they do not establish CRT backend fidelity or sampler termination/distribution.

The structural loop fixture accepts explicit parallel and sequential counts.  Its generated
source keeps nested families under `Fin N` and sequential execution under `MxxIR.IterRuns`:

```text
cargo run -p mxx-ir-core --example emit_loop_fixture -- /tmp/mxx_generated_loops.lean 16 1024
LEAN_PATH=<repository>/crates/ir-core/lean/.lake/build/lib/lean lake env lean /tmp/mxx_generated_loops.lean
```

Changing `N` or `L` changes only closed count expressions; it does not enumerate lanes or steps.
`MxxIR.IterRuns.invariant` provides the reusable initial/step invariant elimination rule used by
the fixture proof.

The matrix-operation fixture exercises row, column, and block-diagonal concatenation together
with a nonzero loop-dependent slice offset:

```text
cargo run -p mxx-ir-core --example emit_matrix_ops_fixture -- /tmp/mxx_generated_matrix_ops.lean
LEAN_PATH=<repository>/crates/ir-core/lean/.lake/build/lib/lean lake env lean /tmp/mxx_generated_matrix_ops.lean
```
