# Lean package for generated relations

Build this package with `lake build` from `crates/ir-core/lean`. Every generated module and every
proof package imports it. It has four libraries:

- `MxxIR` (`IRExpr.lean`, `IRIterRuns.lean`, `IRRel.lean`, `IRScopeSpec.lean`,
  `IRRegression.lean`, and the entry point `MxxIR.lean`): expressions, scope relations, and loop
  semantics. The mathematical namespace is `MxxIR`, regardless of filenames.
- `MxxPrimitives` (`Primitives*.lean`): bounds, CRT decomposition, radix digits, negacyclic
  arithmetic, and preimage and sampling facts.
- `MxxRuntime` (`Runtime*.lean`): the relations generated modules use, the sampling tape and its
  measure, and lemmas about them.
- `RuntimeExamples` (`RuntimeLayoutExamples.lean`): concrete regular gadget layouts, one of them
  with a dropped tower, checking the runtime facts generated relations rely on.

Fixture generators are ordinary unit tests, not example executables. From the repository root:

```sh
cargo test -p mxx-ir-core --lib lean::fixtures
```

IR fixtures are written to `test_data/lean_ir_fixtures/<fixture>/Generated.lean`. Generation
tests validate and export real frozen graphs; they do not themselves invoke the Lean kernel.

The IR fixtures cover constants, hashes, integer hash families, samplers, gadgets, small/wide
preimages, matrix operations, integer matrix-vector products, runtime monomial multiplication,
quoted keyword identifiers, lexical loop bindings, and empty/nonempty structural loops. Check
one from the repository root after building this package:

```sh
LEAN_PATH=crates/ir-core/lean/.lake/build/lib/lean \
  lake +leanprover/lean4:v4.28.0 -d crates/ir-core/lean env lean \
  test_data/lean_ir_fixtures/sampler/Generated.lean
```

Linked-claim fixtures hold several modules. `cargo test -p mxx-dsl --lib test_lwe_protocol`
writes `test_data/lean_ir_fixtures/lwe_protocol/`: a two-stage integer-LWE claim whose ciphertext
crosses stages as an integer-family artifact, its proof, and the certificate. Check the modules in
import order, writing each `.olean` into a directory that is also on `LEAN_PATH`:

```sh
out=$(mktemp -d)
for module in Backend Stage_encrypt Stage_decrypt Ideal Claim LweProof Certificate; do
  LEAN_PATH=crates/ir-core/lean/.lake/build/lib/lean:$out \
    lake +leanprover/lean4:v4.28.0 -d crates/ir-core/lean env lean -o "$out/$module.olean" \
    "test_data/lean_ir_fixtures/lwe_protocol/$module.lean"
done
```

The explicit toolchain matches `crates/ir-core/lean/lean-toolchain`: Lake's `-d` selects the
package without changing the repository-root working directory or Elan's initial toolchain choice.

The fixtures prove consequences of generated execution relations, not sampler termination or
distribution.

`mxx_ir_core::lean::protocol::export` writes a protocol's claim: one module per stage,
requirement, and ideal graph, the `Backend` module of the gadget layouts they use, and
`Claim.lean`. The claim states only that the workflow endpoint equals the ideal one, for every
execution, or except with probability `2^-k` when the declaration sets
`failure_probability_log2 = Some(k)`. A proof package proves `GeneratedClaim.CorrectnessClaim`
and checks it with a three-line certificate, `theorem certificate :
GeneratedClaim.CorrectnessClaim := <proof>` followed by `#print axioms certificate`.

A claim with a failure probability reads each sampled coefficient from a sampling tape. Stage
`i` reads the site prefix `[i]`; a subgraph call extends the site with its node, a loop body
with its node and iteration, and a sampler reads `path ++ [node]`. `MxxRuntime.tapeMeasure` in
`RuntimeSampling.lean` makes every key independent with its sampler's law: the truncated discrete
Gaussian, or a uniform interval or residue. This is the ideal-sampler assumption, and
`randomOracle` is the matching law of hash models. The claim bounds the tape measure of the tapes
that have a failing run, for every hash model and external input. `RuntimeProbability.lean` and
`RuntimeGaussian.lean` supply protocol-independent tools: exponential moments bounded one block
of fresh coordinates at a time, Chernoff and Hoeffding tail bounds, and the sub-Gaussian moment of
the truncated discrete Gaussian.

Families remain functions on `Fin N`; sequential loops use `MxxIR.IterRuns` with a single shared
state tuple. Changing closed counts does not enumerate lanes or steps. `MxxIR.IterRuns.invariant`
provides the reusable initial/step invariant elimination rule.

Production Diamond artifacts are generated and checked inside parameter search, not through
these fixtures. Application proofs remain in their owning crate; the shared exporter and
linked-claim renderer live in `crates/ir-core/src/lean`.

## Reproducible proof CI

CI installs Lean `leanprover/lean4:v4.28.0`, fetches mathlib's compiled cache in this package,
and runs `lake build` here, then in `crates/fhe/lean/tfhe`, `crates/fhe/lean/bgv`, and
`crates/dsl/examples/rlwe`. All four builds are required by the `ci success` job.
The certificate targets are included in each package's default targets.

After installing the normal Rust/OpenFHE dependencies, check that the committed claims
still describe the current Rust graphs and protocol declarations from the repository root:

```sh
python3 scripts/check_lean_claims.py
```

This builds the exporters without GPU features, regenerates TFHE, BGV and RLWE into a temporary
directory, and compares the full file sets and raw bytes with their committed `generated/`
directories. It never overwrites the checkout. Changed bytes, newly generated modules, and
committed modules no longer emitted all fail CI. Unset `FHE_TEST_*` overrides for this check;
the committed proofs use the default fixtures.

To intentionally update the claims after a graph or protocol change:

```sh
output=$(mktemp -d)
cargo run --locked -p mxx-fhe --example export_claims -- "$output"
# Review $output/tfhe and $output/bgv before replacing their generated directories.
cargo run --locked -p mxx-dsl --example rlwe_encrypt -- --export-lean crates/dsl/examples/rlwe/generated
```

The FHE exporters and GPU tests use `mxx_fhe::utils::protocol` for the same graph templates,
metadata and linked declarations. RLWE's export-only mode uses the same program and declaration
as its GPU mode. Exporting constructs graphs and validates declarations; it executes no
cryptographic computation and needs neither CUDA nor a GPU.

`crates/we/lean` is excluded: its proofs need parameter-search-selected modules absent from a
fresh checkout. Its parameter-search proof checks remain separate.
