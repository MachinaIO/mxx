# Lean correctness pipeline

The `lean/` package contains the Graph IR semantics, hard-bounded sampler contract, and
protocol-independent proof lemmas under `lean/Mxx`. Each crate owns its generated protocol
statements and human-written proofs. `mxx-correctness` uses
`crates/correctness/lean/MxxCorrectness`; application-specific checker sources live with their
owning crate, such as `crates/we/lean/MxxWe`.

`ProtocolDecl` is the single protocol source. Each owning crate exposes an argument-free generation
example. The common toy protocol and Diamond WE family are regenerated with

```text
cargo run -p mxx-correctness --example emit_correctness
cargo run -p mxx-we --example emit_correctness
```

Each command mechanically translates its linked workflow, stage graph, artifact binding,
sampler cutoff, ideal graph, precondition, and comparator into a Lean constructor tree and a
generated statement. There is no per-protocol hand transcription and no JSON parser boundary in
Lean. Emission is atomic: if a registered graph contains a Core IR node without a common Lean
denotation, generation fails instead of inserting an opaque or permissive interpretation. Protocol
checkers and proofs under the owning crate's `lean` directory are the hand-written portion. An
emitter creates a proof scaffold only when the proof is absent and never overwrites an existing
proof.

Application certificates retain typed operation handles while the DSL graph is being constructed.
Freezing resolves each retained handle to exactly one stage, structural scope, node, and output
port. A missing or multiply-owned handle is rejected; the emitter never scans the frozen graph to
rediscover a semantic role from a node kind, source location, name, or position. Nested scope
references retain their parent scope and owning loop node, and Lean verifies that ownership against
the executable node. Numeric node identifiers occur only in regenerated certificate data, never as
hand-written proof constants.

Protocol artifact identity is stage-relative. An `ArtifactBinding` commits to the producer stage
and output, and validation requires that output name to equal the artifact name read by the
consumer graph. Concrete `ProductionId` values embedded in executable consumer graphs are runtime
session data and are excluded from the protocol hash. At runtime, `spec_hash` still commits to the
complete `ParamEnv`, and the execution nonce is added to form the `ProductionId`; manifest lookup
then checks that exact production, artifact name, type, family cardinality, and confidentiality.
Consequently parameter search can reuse one generic protocol hash without allowing artifacts from
different concrete parameter bindings to be interchanged.

The Diamond certificate enumerates every artifact edge with the exact producer output wire and
consumer input node, not only the public-key edge. Lean checks these references against the workflow
and follows the compared output's existing executable dependencies across loop boundaries and
artifact edges. The certificate is untrusted data: it does not add a second graph or supply an
equation that is absent from the executable IR.

The generated Diamond statement has a closed intended assumption interface: the bounded-sampler
contract, an accepted checker result, generated parameter validity, generated input
well-formedness, and the explicitly declared pure preconditions. The final bridge from the
certificate to execution of all linked stages is still under construction, so no end-to-end
Diamond theorem is currently exported. See `docs/diamond-we-correctness-status.md` for the exact
review boundary. The target theorem uses exact zero failure probability and contains no
negligible-probability, CLT, or union-bound model.

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
```

`scripts/verified_build.sh` regenerates checked-in declarations into a temporary directory,
verifies that they are current, builds the common and crate-owned stable Lean libraries, verifies
completed theorem packages, and finally builds the Rust workspace. It does not verify the
unfinished Diamond WE end-to-end theorem.

`mxx-we` builds its crate-owned hard-bound checker from `build.rs` and exposes the resulting private
executable to parameter search. Diamond `ProtocolDecl` emission remains a separate operation, but
emits one parameterized protocol family rather than a selected shape or cryptographic
configuration. The build script checks the generated family, stable structural verifier, and
executable parameter checker. It does not import the unfinished execution bridges or final family
proof. Parameter search evaluates the compiled checker for a concrete binding.
