# Lean package for generated relations

Build this package with `lake build` from `crates/ir-core/lean`. Every generated module and every
proof package imports it. It has four libraries:

- `MxxIR` (`IRExpr.lean`, `IRIterRuns.lean`, `IRCost.lean`, `IRRel.lean`, `IRScopeSpec.lean`,
  `IRRegression.lean`, and the entry point `MxxIR.lean`): expressions, scope relations, and loop
  semantics. The mathematical namespace is `MxxIR`, regardless of filenames.
- `MxxPrimitives` (`Primitives*.lean`): bounds, CRT decomposition, radix digits, negacyclic
  arithmetic, and preimage and sampling facts.
- `MxxRuntime` (`Runtime*.lean`): the relations generated modules use, the sampling tape and its
  measure, and lemmas about them. `RuntimeAllCoefficients.lean` (the all-coefficients constant) is
  not imported by `MxxRuntime.lean`; a generated module imports it only when it uses the constant.
- `RuntimeExamples` (`RuntimeLayoutExamples.lean`): concrete regular gadget layouts, one of them
  with a dropped tower, checking the runtime facts generated relations rely on.

`IRCost` provides application-independent additive execution work. Erasing costs from
`CostedIterRuns` recovers `IterRuns`; its iteration sum admits nonuniform bodies, and a
constant-cost premise gives multiplication without unrolling. Composition charges separate
invocations separately, while reuse of a computed value preserves the producer's cost.
`Costed condition` carries both counted derivations and an exact erasure certificate.
Its constructors preserve primitive conditions, conjunctions, existential intermediate
values, finite parallel loops and `IterRuns` state transitions. `HasCost` proofs compose
along the same structure; `runs_iff` then gives the exact cost for precisely the original runs.
`mxx_ir_core::lean::cost::export` builds ordinary and counted relations from one validated
frozen graph traversal with application-supplied opaque `WorkSite` labels. The ordinary
module is unchanged. The companion supplies `<scope>.costed`, `<scope>.work` and a checked
`<scope>.costed_cost` proof. `LeanArtifact::write_modules` writes them in dependency order,
partitioning large modules at declaration boundaries.
Scopes with no observed descendants are atomic base cases, charged the sum of their local
producer weights once per invocation. Observed calls and loops retain their compositional
derivations. This avoids elaborating every zero-work primitive condition again. Large counted
scopes use the native context projections and the same fragment association, including the
balanced conjunctions used by facts exports. Sampling-tape exports use the same rules.
Counted modules, like facts modules, keep `negacyclicModulus` locally irreducible during
structural type checking. They use a larger elaboration budget for production derivations;
this does not alter the native relation or its propositions.
These rules count work, not parallel latency. Applications still own annotation coverage,
the meaning of units, and the hardware throughput assumption; none is inferred by the IR.

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

`cargo test -p mxx-ir-core --lib lean::tests::facts` writes `test_data/lean_ir_fixtures/facts/`:
`GeneratedFacts.lean` and its facts module `GeneratedFacts_Facts.lean` (`ExportOptions::facts`:
wire constants, named atoms and their extraction theorems). Compile the first to an `.olean` as
for the linked-claim fixtures below, then check the second against it.

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

`blockDiagonalRuns` represents `I_k ⊗ X` with one relation. `sha256BitColumns` specifies
deterministic bit-column selector families by the concrete SHA-256 implementation in
`RuntimeHash.lean`, independently of `HashModel`. `RuntimeHashColumnsExamples.lean` checks its
digest stream against reference vectors, including multi-byte little-endian counters.
Long scope boundary types are named once with parameterized abbreviations, so facts do not
repeat an entire large output tuple in every theorem.
Fact sections also share environment binders and a local context notation, while deep child
scope names are bounded in length. Pure broadcast/zip repacking loops are definitional family
projections, retaining count guards and checked prefix indexing rather than extra witnesses.
Large protocol graph/facts sources use declaration-boundary `_Part_<index>` modules, each at
most 3,000,000 source bytes. The original module imports the final part and keeps its public
namespace unchanged. All declarations remain present; partitioning does not weaken the claim.
Record scopes with more than 512 local bindings move their `Ctx`, wire definitions and `outs`
from the facts module into the relation module. Their body uses one context instead of a long
let telescope; the facts module reuses those definitions with the same names and signatures.
Unused pure bindings remain named definitions. Prefix zips use `Fin.castLE` and share one
closed `by decide` inequality per pair of family lengths, without running `omega` in the body.
Offset/quotient zips likewise share a single small-context bound lemma. Large non-root scopes
use the same record representation; lifted definitions retain noncomputability across parts.
Lifted `noncomputable def` declarations skip unnecessary executable-code generation while
retaining their kernel-checked values for definitional equality and proofs.
Large independent bodies use balanced fragment conjunctions and matching logarithmic-depth
facts projections, rather than a long telescope of intermediate proof bindings.

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
