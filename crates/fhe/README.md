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

Each GPU integration test declares the graphs it executes as a closed protocol and calls
`mxx_ir_core::lean::protocol::export`, which writes the statement modules to `generated/` of
`lean/tfhe` or `lean/bgv`. Each package holds handwritten proofs of the generated
`GeneratedClaim.CorrectnessClaim` and a three-line `Certificate.lean` that checks the proof
against it. The claim states only that decryption returns the ideal output; noise bounds are
steps of the proofs, not part of the statement.

- **TFHE** (`utils::tfhe_params`): keygen, two encryptions, one bootstrapped NAND gate, and
  decryption should decode `1 - m1 m2`. These parameters are correct only with high probability,
  so the claim states that, for every hash model and external input, the sampled values with a
  failing run have probability at most `2^-128`. The probability assumes ideal samplers: every sampled coefficient is an independent
  draw from its sampler's law, the truncated discrete Gaussian or a uniform distribution. It
  needs no assumption on hash outputs and none of the usual noise heuristics (independence of
  noise terms, uniform digits, Gaussian tails).
- **BGV** (`utils::bgv_params`): the multiply, relinearize, and modulus-switch round trip
  decrypts the slotwise product for every execution.

The TFHE proof (`TfheStages`, `TfheRotation`, `TfheNand`, `TfheDependence`, `TfheBound`,
`TfheProof`) restates each stage against the sampling tape and models the blind rotation as
explicit functions of it. A run fails only if the LWE secret has more than 506 ones, which
Hoeffding's bound makes rarer than `2^-168`, or if a linear form of the sampled key errors
exceeds `Δ` less a deterministic rounding allowance. That linear form is sub-Gaussian: each
blind-rotation step multiplies fresh key errors by digits of at most 32 that read only earlier
keys, and key switching multiplies its fresh errors by digits of at most 3, giving a tail below
`2^-129`. The BGV proof follows each stage through integer witnesses, including hybrid key
switching and modulus switching, and bounds the final phase so that decoding is exact. It reads
the slot permutations from the packed tables the generated stages define.

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

Each package's `Certificate.lean` prints the axioms of the checked theorem, which are only
`propext`, `Classical.choice`, and `Quot.sound`.
