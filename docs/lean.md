# Lean correctness pipeline

The `lean/` package contains the Graph IR semantics, bounded-sampler contract, and
protocol-independent proof lemmas under `lean/Mxx`. Each crate owns its generated protocol
statements and human-written proofs. `mxx-correctness` uses
`crates/correctness/lean/MxxCorrectness`; application-specific checker sources live with their
owning crate, such as `crates/we/lean/MxxWe`.

Reusable gadget proofs are owned by `crates/gadgets/lean/MxxGadgets`. The Diamond input injector
exports a fail-closed projection from analyzer facts and workflow execution to its typed matrix,
family, recurrence, and artifact interface. Its generic affine and sequential-loop theorems are
built by `mxx-gadgets` itself; the shared `Mxx` library does not import owner-specific proofs.

`ProtocolDecl` is the single protocol source. Each owning crate exposes an argument-free generation
example. The common toy protocol and Diamond WE family are regenerated with

```text
MXX_REGENERATE_CORRECTNESS=1 cargo run -p mxx-correctness --example emit_correctness
MXX_REGENERATE_CORRECTNESS=1 cargo run -p mxx-we --example emit_correctness
```

The explicit regeneration flag lets the emitter compile when the checked-in generated source is
stale. Ordinary Cargo builds do not invoke Lean or reject generated-source freshness: Rust
operational checking is self-contained. The explicit Lean gate regenerates both owners into a
scratch directory, compares the result with the checked-in sources, verifies metadata hashes, and
builds the Lean libraries.

Each command mechanically translates its linked workflow, stage graph, artifact binding,
sampler cutoff, ideal graph, requirements, comparator, input contract, and endpoint anchors into
one Lean `ClosedProtocolDecl`. There is no per-protocol hand transcription and no JSON parser
boundary in Lean. The emitter produces frozen protocol data only: it does not emit a theorem
statement, proof scaffold, rule choice, affine fact, or bound derivation. Protocol theorems remain
ordinary checked-in Lean modules owned by their crate.

The DSL can attach a semantic label to a wire or typed wire set without creating an executable
node. Labels travel through closure sealing, are retained while the graph is frozen, and resolve
once through `FreezeMap` to exact scoped wires. Protocol stages preserve that frozen label map.
Lean's `inferRules` normally selects registered rules automatically. Only an actually ambiguous
site may use a sparse override written in the owning Lean module. Such an override can refer to
these labels, but cannot state an output expression or bound. Rust neither selects nor emits rule
applications. Missing, ambiguous, duplicate, or wrong-arity labels are rejected; there is no
node-kind or numeric-ID search fallback.

The protocol-independent certificate syntax lives under `lean/Mxx/Certificate`. It distinguishes
concrete wires, body templates, instantiated loop paths, joint family elements, and recurrence
results. Matrix and bound recurrence paths are different types. Signal terms are affine pairs of a
proved-bounded coefficient and an ordered public basis; all remaining uncertainty is represented
by a worst-case noise bound. The retained Lean development proves symbolic facts about executable
semantics. Production operational parameter selection is separate: the generic Rust checker
lowers the frozen workflow, applies registered sampler relations, computes worst-case bounds, and
checks the declared decoder target. Rust protocol emission still supplies frozen IR, input
contracts, and resolved semantic labels to the theorem-development path.

Protocol artifact identity is stage-relative. An `ArtifactBinding` commits to the producer stage
and output, and validation requires that output name to equal the artifact name read by the
consumer graph. Concrete `ProductionId` values embedded in executable consumer graphs are runtime
session data and are excluded from the protocol hash. At runtime, `spec_hash` still commits to the
complete `ParamEnv`, and the execution nonce is added to form the `ProductionId`; manifest lookup
then checks that exact production, artifact name, type, family cardinality, and confidentiality.
Consequently parameter search can reuse one generic protocol hash without allowing artifacts from
different concrete parameter bindings to be interchanged.

Artifact bindings preserve producer origins across stages. Family and recurrence identities add
the lane or iteration path rather than replacing the producer identity. The verifier permits
cancellation only for normalized expressions with the same complete origin; separately recomputed
but numerically equal values are not treated as identical. The certificate remains untrusted data:
it neither adds a second executable graph nor supplies an equation absent from executable IR or a
registered semantic theorem.

The Diamond theorem has a closed intended assumption interface: the bounded-sampler contract, an
accepted static checker result, `ParamsWF`, `InputsWF`, and the executable pure requirements in
`ProtocolPreconditions`. The final bridge from analyzer facts to execution of all linked stages is
still under construction, so no end-to-end Diamond theorem is currently exported. See
`docs/diamond-we-correctness-status.md` for the exact review boundary. The target theorem uses
perfect correctness and contains no negligible-probability, CLT, or union-bound model.

The committed toy example workflow is a two-stage executable example: encryption maps a Boolean to
the `q/2` representative, adds a hard-bounded Gaussian sample, and exports a ciphertext artifact.
The decryption entrypoint applies the threshold decoder and labels its Boolean output as the
registered endpoint. The direct equality comparator compares that endpoint with the sampler-free
ideal Boolean. The Rust operational checker derives and evaluates the decoding-radius obligation
from the frozen Gaussian and decoder nodes.

Correctness uses truncated CPU and GPU samplers and deterministic worst-case bounds. GPU Gaussian
sampling rejects individual coefficients in CUDA; batched GPU preimage sampling rejects whole
candidates after full-CRT centered-norm checking so the preimage equation is preserved.
Lattice-security estimation intentionally keeps using the corresponding ordinary untruncated
distributions; that is a separate modeling assumption, not part of the correctness theorem.

To update dependencies, first pin and validate a VCVio revision, then align the mathlib and Lean
toolchain revisions, run `lake update`, repair proofs, and commit `lake-manifest.json` with the
source changes. This pipeline translates serialized Graph IR data into Lean terms; it does not
transpile Rust. If Rust implementation verification is added later, the selected tool is Aeneas
with Charon, recorded here as an out-of-scope architectural decision.

Useful local gates are:

```text
MXX_REGENERATE_CORRECTNESS=1 cargo run -p mxx-correctness --example emit_correctness
scripts/verified_build.sh
cargo build -p mxx-correctness
```

`mxx-we` uses the generic Rust operational checker directly during parameter search. Diamond
protocol emission produces one parameterized protocol family rather than a selected shape or
cryptographic configuration. The crate-owned Lean build remains the theorem-development and
derivation-validation path, invoked explicitly by `scripts/verified_build.sh` and CI; it is not
part of ordinary Cargo builds or the production operational parameter checker. The final workflow
soundness theorem is still under migration, so the current build must not be presented as an
end-to-end Diamond correctness result.
