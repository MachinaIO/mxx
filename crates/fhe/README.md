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
| `cuda/` | The native TFHE blind-rotation kernel used on the GPU. |
| `tests/` | GPU round trips for TFHE and BGV. |
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
