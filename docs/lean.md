# Lean correctness pipeline

The `lean/` package contains the Graph IR semantics, hard-bounded sampler contract, and
protocol-independent proof lemmas under `lean/Mxx`. Each crate owns its generated protocol
statements and human-written proofs. For example, `mxx-correctness` uses
`crates/correctness/lean/MxxCorrectness`, while `mxx-gadgets` uses
`crates/gadgets/lean/MxxGadgets`.

`ProtocolDecl` is the single protocol source. The owning crate exposes an argument-free generation
example. The current checked-in protocols are regenerated with

```text
cargo run -p mxx-correctness --example emit_correctness
cargo run -p mxx-gadgets --example emit_correctness
```

Each command mechanically translates its linked workflow, stage graph, artifact binding,
sampler cutoff, ideal graph, precondition, and comparator into a Lean constructor tree and a
generated statement. There is no per-protocol hand transcription and no JSON parser boundary in
Lean. Emission is atomic: if a registered graph contains a Core IR node without a common Lean
denotation, generation fails instead of inserting an opaque or permissive interpretation. Protocol
checkers and proofs under the owning crate's `lean` directory are the hand-written portion. An
emitter creates a proof scaffold only when the proof is absent and never overwrites an existing
proof.

The generated statement has a closed assumption interface: the bounded-sampler contract, an
accepted checker result, generated parameter validity, generated input well-formedness, and the
explicitly declared pure preconditions. Producer-artifact correctness is not assumed; all linked
stages execute inside the generated workflow denotation. The theorem proves exact zero failure
probability for fixed parameters. It contains no negligible-probability, CLT, or union-bound model.

The committed toy example workflow is a two-stage executable example: encryption maps a Boolean to
the `q/2` representative, adds a hard-bounded Gaussian sample, and exports a ciphertext artifact.
The decryption entrypoint forwards that artifact; the workflow comparator applies a separately
declared sampler-free threshold decoder through `EqualAfterMap`. Its Lean proof derives zero
failure probability from the sampler support contract and the checked decoding-radius inequality.

Correctness uses truncated CPU samplers and deterministic worst-case bounds. GPU sampling is not
yet covered by the runtime-correspondence claim. Lattice-security estimation intentionally keeps
using the corresponding ordinary untruncated distributions; that is a separate modeling
assumption, not part of the correctness theorem.

To update dependencies, first pin and validate a VCVio revision, then align the mathlib and Lean
toolchain revisions, run `lake update`, repair proofs, and commit `lake-manifest.json` with the
source changes. This pipeline translates serialized Graph IR data into Lean terms; it does not
transpile Rust. If Rust implementation verification is added later, the selected tool is Aeneas
with Charon, recorded here as an out-of-scope architectural decision.

Useful local gates are:

```text
scripts/verified_build.sh
cargo run -p mxx-correctness --example verify_correctness
cargo run -p mxx-gadgets --example verify_correctness
```

`scripts/verified_build.sh` is the repository's verified build gate. It regenerates both owning
crates into a temporary directory, checks that the committed generated files are current, builds
the common and crate-owned Lean libraries, verifies theorem hashes and axiom dependencies, and
finally builds the Rust workspace. Ordinary downstream Cargo builds do not run Lean recursively
from `build.rs`; doing so would create a same-crate generator cycle and mutate source files during
dependency builds.
