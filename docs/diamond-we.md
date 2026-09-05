# Diamond witness encryption

`mxx-we` implements Diamond witness encryption as two linked typed-DSL stages. The encryption
stage exports public preprocessing artifacts; the decryption stage imports those exact artifacts.
Both stages declare `instance_width`, `witness_width`, `depth`, and `max_layer_width` as compile
parameters. A runtime compiler binds those parameters from its configured shape, while gate data
and the selected output remain ordinary runtime inputs.

Input-injector transitions, BGG+ public keys, and witness projection preimages are each exported as
one flat family artifact. Transition slots use a rectangular `(level, digit, state)` layout; inactive
state padding is never selected by the protocol. Online injection carries one fixed-width state
family through a `SequentialLoop`, while independent transition generation and application use
`ParallelLoop` nodes.

## Circuit interface

`BooleanCircuitData` is encoded as a rectangular `depth * max_layer_width` family. The runtime
inputs contain one active-gate count per layer, flattened opcode and predecessor families, and a
one-element output-source family. Indices refer only to the immediately preceding layer. Inactive
slots use canonical all-zero records. The canonical validator rejects unknown opcodes, out-of-range
indices, invalid active counts, invalid output sources, and nonzero padding before sampling or
execution.

The logical input order is instance bits followed by witness bits. They are represented by separate
zero-padded families so the encryption stage receives the instance but never the witness. The pure
satisfaction predicate combines those families only while interpreting the ideal Boolean circuit.
Witness bits are packed into input-injector digits inside the DSL graph; host code does not
specialize the graph to their values.

Layers are evaluated by one carried-state `SequentialLoop`. Each layer gathers predecessor families
with dynamic `FamilyGetDynamic` operations and evaluates independent gate slots with structural
`ParallelLoop` nodes. The six supported operations are
constant false, constant true, copy, not, and, and xor. BGG+ selection keeps each encoding vector,
public key, and revealed plaintext from the same selected candidate.

The encryption stage exports the base BGG+ public-key family, and the decryption stage imports that
exact family. Both stages derive intermediate gate public keys from the same circuit wiring. An
AND/XOR right-key gadget decomposition is then recomputed locally from that derived public key and
the protocol's gadget base and digit count. Gadget decomposition is deterministic, so it is not a
ciphertext or artifact field. The exported `diamond_r_decomposed` value is unrelated: it is the
separate projection used by the final decoder.

## Correctness and parameters

The active acceptance path is a generated Lean correctness theorem for the actual frozen protocol
graphs. It links encryption, decryption, all requirement graphs, and the ideal message through the
same external inputs and exported artifacts. For bounded successful executions with valid,
accepting inputs, it proves the actual residual is below the exact decoder threshold and the
operational decoder returns the ideal message, for both Boolean messages.

`crates/we/src/lean.rs::export_claim` supplies WE decoder semantics and runtime backend bindings.
`crates/ir-core/src/lean/protocol.rs::export_claim` exports the declaration's graphs and converts
their protocol connections into a linked claim.
`crates/ir-core/src/lean/claim.rs::assemble_claim` owns application-independent graph linking,
input predicates, and final Boolean threshold claim rendering. The WE adapter supplies the
decoder's Lean helper names; endpoint identities come from the core protocol declaration.
The generic renderer contains no WE
protocol or noise-inference rules. Application-specific bounds, proof sources, and certificate
verification remain under `crates/we/src/lean` and `crates/we/lean`.

Handwritten Lean sources live together in `crates/we/lean`. The `Diamond*.lean` proofs import
candidate-generated IR modules; the exporter copies them unchanged into the candidate directory
and checks them there. For local editing, select that directory with
`python3 scripts/select_we_lean_candidate.py <candidate-directory>`, then run `lake build` in
`crates/we/lean`. Lake builds the current handwritten sources through `Certificate` against the
selected generated snapshot. See `crates/we/lean/README.md` for bootstrap and VS Code setup.

The fixed audit entry point is `crates/we/lean/Certificate.lean`, containing
`DiamondCertificate.correctness : GeneratedClaim.CorrectnessClaim`. The candidate's generated
`Claim.lean` supplies the IR-derived statement and execution assumptions; generated
`NumericCertificate.lean` proves the numeric gate. The final theorem has no numeric premise.

The parameter search called by `test_gpu_diamond_we.rs` already generates and checks these
artifacts; no example executable or separate emission command is needed. Extractor fixtures are
ordinary unit tests. `mxx-ir-core::protocol` supplies declarations and structural validation;
the separate correctness crate and former generic symbolic noise simulator have been removed.

`DiamondWeProtocolFamily::protocol_decl` returns a validated `WitnessEncryptionProtocolDecl`. The
declaration is built directly from symbolic circuit and cryptographic parameters; it does not own a
concrete circuit shape or a parameter-search candidate. Its sole fixed value is the BGG domain tag.
Consequently, changing a runtime `ParamEnv` does not change the protocol hash. The declaration links
the encryption and decryption graphs, checks every artifact and protocol-input mapping, includes a
pure circuit-data validity predicate, includes a pure Boolean satisfiability predicate, and compares
the decrypted Boolean with the unchanged ideal message. `DiamondWeCompiler::protocol_decl` delegates
to the same family declaration, while its concrete shape remains confined to runtime validation and
parameter binding.

Correctness parameters are intended to use deterministic coefficient bounds. The default helper functions compute
`floor(6.5 * sigma)` exactly; the preimage helper first computes a rational upper bound for the
existing preimage sigma formula. CPU Gaussian sampling rejection-resamples coefficients outside the
declared cutoff. CPU preimage sampling rejects a whole candidate, preserving the preimage equation.

`DiamondParameterSearch` fixes the circuit shape and searches ring dimension and modulus depth. It
uses the ordinary untruncated Gaussian only for lattice-security estimation. Correctness uses a
worst-case recurrence with full ring and inner-dimension factors, then invokes Lean on freshly
generated source modules before accepting a candidate. A compact capped binary computation is
checked in the Lean kernel and discharges the strict numeric gate of the full theorem. All protocol
parameters, including circuit dimensions and exact sampler sigmas, come from the same compiler and
backend setup. The selected result retains the verified artifact directory. Conservative numeric
rejection is separate from export, unsupported-semantics, compiler, and timeout errors; there is no
fallback to the older operational checker. Search is a candidate-finding heuristic, not a claim of
CRT-depth minimality or exhaustive failure.

Certified GPU Diamond execution is not exposed. The GPU Gaussian and preimage paths must enforce
the same bounded support, including whole-candidate preimage rejection, before a GPU integration
path can claim the bounded-sampler theorem.
