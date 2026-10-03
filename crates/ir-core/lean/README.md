# Frozen-IR Lean extraction fixtures

Build this package with `lake build` from `crates/ir-core/lean`.

The handwritten modules are flat: `IRExpr.lean`, `IRIterRuns.lean`, `IRRel.lean`,
`IRScopeSpec.lean`, and `IRRegression.lean`. `MxxIR.lean` is the common import entry point;
the library's explicit roots also build the regression module. The mathematical namespace
remains `MxxIR`, regardless of filenames.

Fixture generators are ordinary unit tests, not example executables. From the repository root:

```sh
cargo test -p mxx-ir-core --lib lean::fixtures
cargo test -p mxx-backends --lib lean::fixtures
```

IR fixtures are written to `test_data/lean_ir_fixtures/<fixture>/Generated.lean`. The runtime
layout fixture is written to `test_data/lean_runtime_fixture/Generated.lean`. Generation tests
validate and export real frozen graphs; they do not themselves invoke the Lean kernel.

The IR fixtures cover constants, hashes, integer hash families, samplers, gadgets, small/wide
preimages, matrix operations, integer matrix-vector products, runtime monomial multiplication,
quoted keyword identifiers, lexical loop bindings, and empty/nonempty structural loops. Their proof text can be checked separately from the repository root, after building the
IR and runtime packages:

```sh
LEAN_PATH=crates/ir-core/lean/.lake/build/lib/lean \
  lake +leanprover/lean4:v4.28.0 -d crates/backends/lean env lean \
  test_data/lean_ir_fixtures/sampler/Generated.lean
```

Linked-claim fixtures hold several modules. `cargo test -p mxx-dsl --lib test_lwe_protocol`
writes `test_data/lean_ir_fixtures/lwe_protocol/`: a two-stage centered-residual integer-LWE claim
whose ciphertext crosses stages as an integer-family artifact, its proof, and the certificate. Check the modules in import order, writing each `.olean` into a
directory that is also on `LEAN_PATH`:

```sh
out=$(mktemp -d)
for module in LweFixture Stage_encrypt Stage_decrypt Ideal Claim LweProof Certificate; do
  LEAN_PATH=crates/ir-core/lean/.lake/build/lib/lean:$out \
    lake +leanprover/lean4:v4.28.0 -d crates/backends/lean env lean -o "$out/$module.olean" \
    "test_data/lean_ir_fixtures/lwe_protocol/$module.lean"
done
```

The explicit toolchain matches `crates/backends/lean/lean-toolchain`: Lake's `-d` selects the
package without changing the repository-root working directory or Elan's initial toolchain choice.

The fixtures prove consequences of generated execution relations, not sampler termination or
distribution. The runtime fixture additionally supplies the concrete CRT gadget layout.

A claim with a failure probability (`ClaimSemantics::failure_probability_log2`) instead reads each
sampled coefficient from a sampling tape. Stage `i` reads the site prefix `[i]`; a subgraph call
extends the site with its node, a loop body with its node and iteration, and a sampler reads
`path ++ [node]`. `MxxRuntime.tapeMeasure` in `crates/backends/lean/RuntimeSampling.lean` makes
every key independent with its sampler's law: the truncated discrete Gaussian, or a uniform
interval or residue. This is the ideal-sampler assumption, and `randomOracle` is the matching
law of hash models. The claim bounds the tape measure of the tapes that have a failing run, for
every hash model and external input. `RuntimeProbability.lean` and `RuntimeGaussian.lean` supply
protocol-independent tools: exponential moments bounded one block of fresh coordinates at a time,
Chernoff and Hoeffding tail bounds, and the sub-Gaussian moment of the truncated discrete
Gaussian.

Families remain functions on `Fin N`; sequential loops use `MxxIR.IterRuns` with a single shared
state tuple. Changing closed counts does not enumerate lanes or steps. `MxxIR.IterRuns.invariant`
provides the reusable initial/step invariant elimination rule.

Production Diamond artifacts are generated and checked inside parameter search, not through
these fixtures. Application proofs remain in their owning crate; the shared exporter and
linked-claim renderer live in `crates/ir-core/src/lean`.
