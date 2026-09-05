# Frozen-IR Lean extraction fixtures

Build this package with `lake build` from `crates/ir-core/lean`.

Fixture generators are ordinary unit tests, not example executables. From the repository root:

```sh
cargo test -p mxx-ir-core --lib lean::fixtures
cargo test -p mxx-runtime --lib lean::fixtures
```

IR fixtures are written to `test_data/lean_ir_fixtures/<fixture>/Generated.lean`. The runtime
layout fixture is written to `test_data/lean_runtime_fixture/Generated.lean`. Generation tests
validate and export real frozen graphs; they do not themselves invoke the Lean kernel.

The IR fixtures cover constants, hashes, samplers, gadgets, small/wide preimages, matrix
operations, lexical loop bindings, and empty/nonempty structural loops. Their proof text can
be checked separately from `crates/runtime/lean`, after building the IR and runtime packages:

```sh
LEAN_PATH=../../ir-core/lean/.lake/build/lib/lean lake env lean ../../../test_data/lean_ir_fixtures/sampler/Generated.lean
```

The fixtures prove consequences of generated execution relations, not sampler termination or
distribution. The runtime fixture additionally supplies the concrete CRT gadget layout.

Families remain functions on `Fin N`; sequential loops use `MxxIR.IterRuns` with a single shared
state tuple. Changing closed counts does not enumerate lanes or steps. `MxxIR.IterRuns.invariant`
provides the reusable initial/step invariant elimination rule.

Production Diamond artifacts are generated and checked inside parameter search, not through
these fixtures. Application proofs remain in their owning crate; the shared exporter and
linked-claim renderer live in `crates/ir-core/src/lean`.
