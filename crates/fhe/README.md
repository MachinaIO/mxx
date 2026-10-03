# mxx-fhe

`mxx-fhe` builds fully homomorphic encryption with the mxx DSL: TFHE with NAND bootstrapping over
integer LWE, and leveled BGV with SIMD slots, rotations, relinearization, and hybrid RNS key
switching. Its methods build graphs rather than encrypting eagerly, so the same key generation,
encryption, evaluation, and decryption run on the CPU or the GPU. It depends on `mxx-dsl`,
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
| `cuda/` | The native TFHE blind-rotation kernel used on the GPU. |
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
[`crates/ir-core/SPEC.md`](../ir-core/SPEC.md) specifies the generated modules and statement.

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
to the next, and requires the input contracts. The statements say only that decryption returns
the ideal output; noise bounds are steps of the proofs, not part of the statements.

### Assumptions

- **Ideal samplers (TFHE).** The TFHE parameters are correct only with high probability. Every
  sampled coefficient is read from a sampling tape whose entries are independent draws from
  their sampler's law: the truncated discrete Gaussian, or a uniform interval or residue. The
  proof needs none of the usual noise heuristics, such as independent noise terms, uniform
  digits, or Gaussian tails.
- **Truncated samplers (BGV).** Every sample lies within its cutoff, so the BGV statement holds
  for every execution, with no probability.
- **No assumption on hashes.** Both statements hold for every hash model, that is, for every
  function from hash inputs to outputs.
- **Trusted base.** The statement compiler in `mxx-ir-core`, the relations of the `MxxRuntime`
  Lean library, which describe what each runtime primitive computes, and the Lean kernel.

### How the proofs work

The TFHE proof (`TfheStages`, `TfheRotation`, `TfheNand`, `TfheDependence`, `TfheBound`,
`TfheProof`) restates each stage against the sampling tape and models the blind rotation as
explicit functions of it. A run fails only if the LWE secret has more than 506 ones, which
Hoeffding's bound makes rarer than `2^-168`, or if a linear form of the sampled key errors
exceeds `Δ` less a deterministic rounding allowance. That linear form is sub-Gaussian: each
blind-rotation step multiplies fresh key errors by digits of at most 32 that read only earlier
keys, and key switching multiplies its fresh errors by digits of at most 3, giving a tail below
`2^-129`.

The BGV proof (`BgvStages`, `BgvProof`) follows each stage through integer witnesses, including
hybrid key switching and modulus switching, and bounds the final phase so that decoding is exact.
It reads the slot permutations from the packed tables the generated stages define.

### Package layout

| Path | Contents |
| --- | --- |
| `lean/{tfhe,bgv}/generated/` | The modules `export` writes: one per stage, `Ideal`, `Backend`, and `Claim`. Never edited by hand. |
| `lean/{tfhe,bgv}/*.lean` | The handwritten proof of `GeneratedClaim.CorrectnessClaim`. |
| `lean/{tfhe,bgv}/Certificate.lean` | `theorem certificate : GeneratedClaim.CorrectnessClaim := <proof>` and `#print axioms certificate`. |

### Regenerating and checking

Regenerate the statements with the GPU integration tests, then check both proofs inside each
package directory:

```bash
cargo test -r -p mxx-fhe --features gpu --test gpu_tfhe --test gpu_bgv
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
