# mxx

`mxx` is a Rust and CUDA workspace for lattice-cryptography research. It contains
polynomial and matrix primitives, bounded samplers, an executable graph IR and DSL, reusable BGG+
and circuit gadgets, runtime backends, and application-specific Lean correctness proofs.

## Workspace layout

| Crate | Responsibility |
| --- | --- |
| `mxx-primitives` | Polynomial/matrix operations, CPU/GPU kernels, and concrete samplers. |
| `mxx-ir-core` | Executable DAG, protocol declarations, structural validation, artifact manifests, and Lean claim generation. |
| `mxx-dsl` | Typed graph construction and sampler-free ideal/predicate builders. |
| `mxx-runtime` | CPU/GPU graph execution, transcripts, sessions, and in-memory artifacts. |
| `mxx-bench-estimator` | Validated-graph cost and memory composition. |
| `mxx-gadgets` | BGG-independent circuits and reusable circuit gadgets. |
| `mxx-bgg` | BGG+ keys, encodings, sampling, evaluation, decoding, lookup, slot transfer, and refresh. |
| `mxx-fhe` | DSL-based Ring Regev/Ring-GSW and leveled BGV, SIMD, rotations, and ciphertext noise tracking. |
| `mxx-we` | Witness-encryption interfaces and parameterized dynamic-circuit Diamond WE. |
| `mxx-func-enc`, `mxx-io` | Functional-encryption and iO interfaces; protocol implementations have been removed. |

The retired symbolic IR and probabilistic noise simulator are not part of the workspace.
Correctness uses enforced integer coefficient cutoffs and deterministic worst-case bounds. CPU
samplers implement the current runtime-correspondence contract; GPU cutoff enforcement is tracked
as a follow-up. Lattice-security estimation intentionally continues to model the corresponding
ordinary untruncated distributions separately.

See `docs/architecture.md`, `docs/dsl.md`, `docs/ir-core.md`, `docs/runtime.md`, and
`docs/correctness/operational-protocol-inventory.md`.

## FHE graphs

`mxx-fhe` constructs cryptographic graphs; its methods do not encrypt eagerly.
Key generation, sampling, polynomial arithmetic, and decryption execute when
`mxx-runtime` runs the validated graph. CPU and GPU backends use the same FHE DSL.
Bootstrapping is not implemented.

| API | Representation and behavior |
| --- | --- |
| `FheCommonParams` | Existing `DCRTPolyParams`, binary/ternary secret interval, Gaussian sigma, and coefficient cutoff. Level zero keeps the first CRT prime; higher levels keep longer prefixes. |
| `FheScheme` | Shared `keygen`, `encrypt`, `decrypt`, `add`, and `mul` graph builders, with scheme-specific plaintext, multiplication operand, and evaluation-key types. |
| `RingGswParams` | `new(common, scale, plaintext_bound)`; scalar polynomial `Mat` plaintexts in the ciphertext ring R_q. Regev ciphertexts have separate a/b parts with phase `b - s*a = scale*m + e`. |
| `RingCiphertext` | Shared storage for Regev and GSW aliases, with a/b parts and public noise/plaintext bounds. GSW encrypts an unscaled gadget diagonal; its external product multiplies a Regev plaintext by the GSW polynomial. |
| `BgvParams` | `new(common, plaintext_modulus)`; messages are `Family<Int>` with 1 to N SIMD slots modulo t. Supports addition, multiplication with relinearization, CRT modulus switching, SIMD, and rotations. |
| `BgvCiphertext` | Components are descending coefficients in `-s`: `(a,b)` for ordinary ciphertexts or three rows before relinearization. `correction_factor` tracks the plaintext multiplier modulo t, while `noise_bound` tracks coefficient noise. |

Ring Regev's plaintext ring and ciphertext ring have the same modulus q.
The scale and declared centered coefficient bound restrict which plaintexts can
be recovered under noise; negative decoded coefficients are returned as canonical
residues modulo q. BGV instead decrypts modulo its separate plaintext modulus t.
For BGV multiplication, `mul` takes a relinearization key; alternatively,
`mul_unrelinearized` and `relinearize` expose the two steps explicitly.

BGV uses SIMD by default and requires a prime t with `t = 1 mod 2N`.
Call `encrypt(&key, &slots)` directly; `decrypt(&secret, &ciphertext)` returns
all N slots. Inputs with 1 to N integers fill successive slots, and unused slots
are zero. A single integer occupies slot zero without broadcasting. Returning
all slots preserves values moved into initially unused positions by rotations,
without storing an input length in the ciphertext.

Slots are interpreted as evaluation values in the plaintext ring R_t. Internal
encoding uses the native inverse NTT modulo t, centers and lifts the resulting
coefficients into R_Q, and uses the native evaluation representation for
ciphertext arithmetic. The t-to-Q coefficient lift is necessary; copying
R_t evaluation values directly into R_Q would change the message polynomial.
Encoding and decoding are internal details, so callers need no separate steps.
Slots occupy two rows of N/2 entries.
`rotate_rows` rotates both rows (positive offsets move entries left), and
`swap_rows` exchanges them. Nontrivial rotations and row swaps need their
respective evaluation keys at the ciphertext's level. CRT modulus switching
operates on residues without reconstructing whole ciphertext coefficients.

To execute a graph:

1. Construct a `DslContext`, declare inputs, and use the FHE methods to build
   encryption, evaluation, and decryption nodes. Mark secrets and decoded values
   as private outputs.
2. Build and validate the graph with a `ParamEnv`. Register the exact ordered
   ciphertext CRT bases with the runtime backend, including single-prime rings
   used by modulus switching and `batching_parameters()` for BGV.
3. Call runtime `execute` with inputs, a backend, a `MemoryArtifactStore`, and a
   sampling mode. Materialize lazy family outputs before inspecting their values.

Noise bounds propagate with each ciphertext through evaluation. `can_decrypt`
checks a conservative sufficient correctness condition; it does not measure
secret runtime values. Declared input bounds and compatible keys remain caller
obligations. DSL schemas retain public metadata, but a matrix artifact alone does
not contain correction factors or bounds: carry those alongside components when
connecting separate protocol stages. Artifacts remain in memory or are passed as
direct runtime inputs.

Start with the runtime unit tests in `crates/fhe/src/ring_gsw.rs` and
`crates/fhe/src/bgv.rs` for slot arithmetic, measured noise, and
staged evaluation. `crates/fhe/src/tests_gpu.rs` executes the production graphs
on GPU, including a public evaluator that receives no secret key. The design
and formulas are documented in `docs/plans/fhe.md`.

```sh
cargo test -r -p mxx-fhe --lib
cargo test -r -p mxx-fhe --lib --features gpu test_gpu_fhe
```

Run GPU tests outside the sandbox on a CUDA-capable machine. The unit-test toy
parameters are configurable through `FHE_TEST_RING_DIMENSION`,
`FHE_TEST_CRT_DEPTH`, `FHE_TEST_CRT_BITS`, `FHE_TEST_BASE_BITS`, `FHE_TEST_SIGMA`,
and `FHE_TEST_ERROR_CUTOFF`; parameter changes must preserve decoding margins
and each test's batching requirements. Defaults are correctness fixtures, not
security parameter recommendations.

## Diamond iO and AKY24 iO implementations

Diamond iO and AKY24 iO were removed from this branch as part of the migration to a
DSL-based design. The disabled AKY24 functional-encryption implementation was also removed.
The latest implementations of Diamond iO and AKY24 iO remain on the
[`main` branch](https://github.com/MachinaIO/mxx/tree/main):
[Diamond iO](https://github.com/MachinaIO/mxx/blob/main/src/io/diamond_io.rs) and
[AKY24 iO](https://github.com/MachinaIO/mxx/blob/main/src/io/aky24_io.rs).
For a fixed reference, `main` pointed to
[`d5d6fba26f1d20f11d4648a3fd1c9b35241ff4a9`](https://github.com/MachinaIO/mxx/tree/d5d6fba26f1d20f11d4648a3fd1c9b35241ff4a9)
when this removal was made. Diamond WE remains available in this branch.

## Requirements

- Rust with edition 2024 support.
- OpenFHE and OpenMP.
- CUDA toolkit for the optional `gpu` feature.

Rust formatting uses `cargo +nightly fmt --all`.
