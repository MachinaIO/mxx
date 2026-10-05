# mxx-fhe

`mxx-fhe` builds fully homomorphic encryption with the mxx DSL: TFHE with NAND bootstrapping over
integer LWE, and leveled BGV with SIMD slots, rotations, relinearization, and hybrid RNS key
switching. Its methods build graphs rather than encrypting eagerly, so the same key generation,
encryption, evaluation, and decryption can run on the CPU or a supported GPU backend. It depends on `mxx-dsl`,
`mxx-ir-core`, and `mxx-backends`.

## Contents

| Item | What it provides |
| --- | --- |
| `FheCommonParams` | Ring, secret, and Gaussian parameters shared by both schemes. |
| `TfheParams`, `LweCiphertext`, `TfheKeys` | TFHE parameters, ciphertexts, keys, and the bootstrapping stages. |
| `BgvParams`, `BgvCiphertext` | BGV parameters, SIMD encoding, arithmetic, rotations, and key switching. |
| `FheScheme` | The shared matrix-plaintext interface, implemented by BGV. |
| `utils` | Parameter helpers, including the standard TFHE Boolean profile used in tests. |
| `lean/` | Lean packages that prove the generated correctness claims of both protocols. |
| `cuda/` | The CUDA-only native TFHE blind-rotation kernel. |
| `tests/` | GPU round trips for TFHE and BGV, which also declare the executed graphs as a closed protocol and export its Lean claim. |
| `scripts/` | A GPU BGV comparison with PhantomFHE: building a pinned external checkout, the comparison driver, and validating and summarizing its measurements. |

## Design

- **Schemes as graph builders.** Each operation adds nodes to a `DslContext`. A protocol stage,
  such as key generation or one bootstrapped gate, is one graph that is planned once and run
  many times, with keys kept on the GPU between runs.
- **Public noise tracking.** Ciphertexts carry public noise bounds that evaluation propagates,
  and `can_decrypt` checks a conservative condition without inspecting secrets.
- **A native kernel where it pays.** TFHE blind rotation is a named subgraph whose GPU execution
  is one cooperative CUDA kernel, while the CPU runs its DSL body; tests check that both agree
  bit for bit.

## Lean correctness proofs

The TFHE NAND gate and the BGV round trip have machine-checked correctness proofs at the
parameters of the GPU integration tests. The statements are not written by hand: each
integration test declares the graphs it executes as a protocol and calls
`mxx_ir_core::lean::protocol::export`, which deterministically writes the statement
`GeneratedClaim.CorrectnessClaim` to `generated/` of `lean/tfhe` or `lean/bgv`.
[`crates/ir-core/LEAN-SPEC.md`](../ir-core/LEAN-SPEC.md) specifies the generated modules and statement.

### What is proved

| | TFHE (`utils::tfhe_params`) | BGV (`utils::bgv_params`) |
| --- | --- | --- |
| Stages | keygen, two encryptions, one bootstrapped NAND gate, decryption | keygen, two encryptions, multiply, relinearize, modulus switch, decryption |
| Ideal output | `1 - m1 m2` for message bits `m1`, `m2` | `x[i] * y[i] mod t` for every slot `i` |
| Input contracts | bits in `[0, 1]`, 32-byte hash keys | slots in `[0, t - 1]` |
| Statement | the decrypted bit differs from the ideal one with probability at most `2^-128` | every execution decrypts the ideal slots |

In Lean, the two statements read:

```lean
-- TFHE: the sampling tapes with a failing execution have measure at most 2^-128.
∀ hashModel external,
  MxxRuntime.tapeMeasure {tape | ∃ execution, Runs hashModel external tape execution ∧
    ¬ (execution.«stage_4» = execution.«ideal»)} ≤ (2 : ENNReal)⁻¹ ^ 128

-- BGV: every execution decrypts correctly.
∀ hashModel external execution, Runs hashModel external execution →
  execution.«stage_6» = execution.«ideal»
```

`Runs` links the generated stage relations exactly as the test passes outputs from one execution
to the next, and requires the input contracts.

### Package layout

| Path | Contents |
| --- | --- |
| `lean/{tfhe,bgv}/generated/` | The modules `export` writes: one per stage, `Ideal`, `Backend`, and `Claim`. Never edited by hand. |
| `lean/{tfhe,bgv}/*.lean` | The handwritten proof of `GeneratedClaim.CorrectnessClaim`. |
| `lean/{tfhe,bgv}/Certificate.lean` | `theorem certificate : GeneratedClaim.CorrectnessClaim := <proof>` and `#print axioms certificate`. |

### Regenerating and checking

Regenerate the statements with the GPU integration tests, then check both proofs inside each
package directory (integration tests require explicit authorization):

```bash
MXX_GPU_BACKEND=cuda cargo test -r -p mxx-fhe --features gpu --test gpu_tfhe --test gpu_bgv
```

```bash
cd crates/fhe/lean/tfhe && lake build
```

```bash
cd crates/fhe/lean/bgv && lake build
```

Regenerating at unchanged parameters writes byte-identical modules. Each package's
`Certificate.lean` prints the axioms of the checked theorem, which are only `propext`,
`Classical.choice`, and `Quot.sound`.

## GPU backend scope

`MXX_GPU_BACKEND=cuda|hip` is selected by `mxx-backends` and propagated through Cargo metadata;
FHE does not choose a separate native compiler. BGV uses the common backend on either build.
The TFHE blind-rotation kernel remains CUDA-only: HIP builds skip its native compilation and
`TfheParams::gpu_blind_rotation_kernel` returns `None`. AMD TFHE execution is deferred and is
not a supported or validated path in this change; CPU TFHE APIs remain available.

HIP workspace/FHE compilation passed for gfx1100 and gfx942 with ROCm 7.0.0; AMD BGV device
execution remains unverified. Five CUDA FHE device cases passed in each of three smoke repetitions; the whole device
gate still has a pending long KHE probe after the compiled planner/lowering correction. See [`AMD_GPU_VALIDATION.md`](../../AMD_GPU_VALIDATION.md). The external PhantomFHE comparison tools
under `crates/fhe/scripts/` target CUDA; they do not validate HIP or establish cross-vendor
performance equivalence. See the [backend support evidence](../backends/README.md#gpu-backend-selection).
