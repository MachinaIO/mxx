# Diamond witness encryption

`mxx-we` implements Diamond witness encryption as two linked typed-DSL stages. The encryption
stage exports public preprocessing artifacts; the decryption stage imports those exact artifacts.
Both stages declare `instance_width`, `witness_width`, `depth`, and `max_layer_width` as compile
parameters. A runtime compiler binds those parameters from its configured shape, while gate data
and the selected output remain ordinary runtime inputs.

Input-injector transitions use a logical `[level, state, digit]` family. Their source/trapdoor
family has shape `[level, state]`, so the source is independent of the final branch axis while the
preimage and target vary with the digit. BGG+ public keys and witness projection preimages are also
typed artifacts. Online injection carries one fixed-width state family through a `SequentialLoop`;
independent work uses rank-N `ParallelGrid` nodes.

## Circuit interface

`BooleanCircuitData` is encoded as a rectangular `depth * max_layer_width` family. The runtime
inputs contain one active-gate count per layer, flattened opcode and predecessor families, and a
one-element output-source family. Indices refer only to the immediately preceding layer. Inactive
slots use canonical all-zero records. The canonical validator rejects unknown opcodes, out-of-range
indices, invalid active counts, invalid output sources, and nonzero padding before sampling or
execution.

The logical input order is instance bits followed by witness bits. They are represented by separate
zero-padded families so the encryption stage receives the instance but never the witness. Witness
bits are packed into input-injector digits inside the DSL graph; host code does not specialize the
graph to their values.

Layers are evaluated by one carried-state `SequentialLoop`. Each layer gathers predecessor families
with dynamic `FamilyGetDynamic` operations and evaluates independent gate slots with structural
`ParallelGrid` nodes. The six supported operations are
constant false, constant true, copy, not, and, and xor. BGG+ selection keeps each encoding vector,
public key, and revealed plaintext from the same selected candidate.

The encryption stage exports the base BGG+ public-key family, and the decryption stage imports that
exact family. Both stages derive intermediate gate public keys from the same circuit wiring. An
AND/XOR right-key gadget decomposition is then recomputed locally from that derived public key and
the protocol's gadget base and digit count. Gadget decomposition is deterministic, so it is not a
ciphertext or artifact field. The exported `diamond_r_decomposed` value is unrelated: it is the
separate projection used by the final decoder.

## Noise simulation and parameters

`DiamondParameterSearch` freezes linked encryption and decryption graphs, supplies concrete
external input facts, and requests the noisy plaintext matrix from `mxx-noise-simulator`. The
simulator returns a coefficient-error bound. Diamond applies its Boolean decoder interval outside
the simulator; the simulator has no Diamond-specific rule, decoder kind, ideal program, or
predicate.

Noise parameters use deterministic coefficient bounds. The default helper functions compute
`floor(6.5 * sigma)` exactly; the preimage helper first computes a rational upper bound for the
existing preimage sigma formula. CPU Gaussian sampling rejection-resamples coefficients outside the
declared cutoff. CPU preimage sampling rejects a whole candidate, preserving the preimage equation.

`DiamondParameterSearch` fixes the circuit shape and searches ring dimension and modulus depth. It
uses the ordinary untruncated Gaussian only for lattice-security estimation. Candidate evaluation
runs direct executable-IR noise simulation with full ring and inner-dimension factors, then applies
the decoder condition before accepting a candidate.

One verified integration-test run used `witness_size=10` and requested 128 security bits. It
selected CRT depth 4, ring dimension 16384, 60-bit CRT moduli, 240 total modulus bits, and a 30-bit
gadget base, giving two digits per CRT modulus. The estimator reported 215 achieved security bits.
The post-fix noise bound was
`251688648476311380159013360416031481340377932611078873344980140687360`, with 960 planned wires
and 640 transfer steps. The report contained 68 meaningful carrier-precision-loss diagnostics from
concat/slice, branch selection, and distinct-source addition/subtraction on intermediate paths. It
contained no ordinary-multiplication false positives. These diagnostics do not weaken the numeric
noise bound, and every `ApplyPreimage` remains fail-closed when its required carrier is unavailable.

These values are evidence from that concrete test environment and search range, not universal
defaults. After parameter search and simulation, the run stopped at the integration test's explicit
no-GPU assertion; it did not execute GPU cost measurement or the encryption/decryption round trip.

The GPU integration path requires detected hardware after CPU-only parameter search and noise
simulation. A no-GPU stop therefore establishes the search/simulation result only, not GPU runtime
acceptance.
