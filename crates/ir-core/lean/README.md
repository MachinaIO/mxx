# Frozen-IR Lean extraction fixtures

Build this package with `lake build` from `crates/ir-core/lean`.

The handwritten modules are flat: `IRExpr.lean`, `IRIterRuns.lean`, `IRRel.lean`,
`IRScopeSpec.lean`, and `IRRegression.lean`. `MxxIR.lean` is the common import entry point;
the library's explicit roots also build the regression module. The mathematical namespace
remains `MxxIR`, regardless of filenames.

Fixture generators are ordinary unit tests, not example executables. From the repository root:

```sh
cargo test -p mxx-ir-core --lib lean::fixtures
cargo test -p mxx-runtime --lib lean::fixtures
```

IR fixtures are written to `test_data/lean_ir_fixtures/<fixture>/Generated.lean`. The runtime
layout fixture is written to `test_data/lean_runtime_fixture/Generated.lean`. Generation tests
validate and export real frozen graphs; they do not themselves invoke the Lean kernel.

The IR fixtures cover constants, hashes, samplers, gadgets, small/wide preimages, matrix
operations, quoted keyword identifiers, lexical loop bindings, and empty/nonempty structural
loops. Their proof text can be checked separately from the repository root, after building the
IR and runtime packages:

```sh
LEAN_PATH=crates/ir-core/lean/.lake/build/lib/lean \
  lake +leanprover/lean4:v4.28.0 -d crates/runtime/lean env lean \
  test_data/lean_ir_fixtures/sampler/Generated.lean
```

The explicit toolchain matches `crates/runtime/lean/lean-toolchain`: Lake's `-d` selects the
package without changing the repository-root working directory or Elan's initial toolchain choice.

The fixtures prove consequences of generated execution relations, not sampler termination or
distribution. The runtime fixture additionally supplies the concrete CRT gadget layout.

Families remain functions on `Fin N`; sequential loops use `MxxIR.IterRuns` with a single shared
state tuple. Changing closed counts does not enumerate lanes or steps. `MxxIR.IterRuns.invariant`
provides the reusable initial/step invariant elimination rule.

Production Diamond artifacts are generated and checked inside parameter search, not through
these fixtures. Application proofs remain in their owning crate; the shared exporter and
linked-claim renderer live in `crates/ir-core/src/lean`.
