# mxx-pe

`mxx-pe` implements predicate encryption for branching programs over key-homomorphic encodings,
without FHE. A ciphertext encrypts a binary message under an attribute `x`. A key for a branching
program `F` decrypts it exactly when `F(x) = 0`. Encryption needs no predicate, so one ciphertext
serves keys for every program of the setup's function class. The scheme runs on the Dynamic
RNS-BGG evaluator of [`mxx-exponent-lut`](../exponent-lut/README.md), and every stage is a graph
that runs on the CPU or the GPU. It depends on `mxx-exponent-lut`, `mxx-dsl`, `mxx-ir-core`, and
`mxx-backends`.

## Contents

| Item | What it provides |
| --- | --- |
| `branching_program` | `PeClass` (attribute length, alphabet, state width, maximal length) and `BranchingProgram`, checked against its class, evaluated in the clear, and compiled into an Exponent-LUT Program bound to the setup coordinates. |
| `params` | `PeParameters`: the class, the equal-basis chain of `2 T_max + 2` levels, the error, flooding, and programming contracts, and the key width and cutoff the existing sampler realizes; `PeParameters::select` picks the CRT depth from the simulated noise. |
| `setup` | `setup`: the anchor `B` from the existing TrapGen, the class's Dynamic setup, and the public message row. `PePublicKey` and `PeMasterSecretKey` keep public and secret material apart. |
| `encryption` | `encrypt(mpk, x, m)`: the anchor encoding, the attribute gadget ciphertexts `C_t = P Z_t + X^(Bz x_t) G`, the message component, and the Dynamic packages, under one fresh secret and session. |
| `keygen` | `keygen(msk, F)`: the predicate's public evaluation, including its output key `A_F`, and a short `k` with `M_F k = d_1 delta_1` for the implicit family matrix `M_F`. |
| `decryption` | `decrypt(sk_F, ct)`: the predicate evaluated on the ciphertext's packages and the family row contracted with the key, rounded to the message bits. |
| `noise` | `simulate`: the evaluation bound `E_F` from the exponent-lut simulator, the decryption bound `E_PE`, and the condition `4 E_PE + 2 < q`. |

## Design

- **Programs as exponents.** The state `z` of a branching program is the exponent of a monomial.
  Each step multiplies it by the package of the attribute it reads, giving `X^(z + Bz x)`, and
  applies the step's table as an Exponent-LUT lookup. The last table returns the output bit as a
  coefficient, and its full-gadget carrier `c_F = s (A_F - F(x) G) + e_F` is never directly
  decoded.
- **One ciphertext for every predicate.** Packages are bound to setup coordinates, not to a
  Program, level, or table, so a ciphertext's size depends on the attribute length and the class's
  lookup width but not on the program length or the predicate.
- **SampleLeft without the family matrix.** Key generation samples the one extension vector shared
  by every row of `M_F`, then calls the existing preimage sampler once per anchor block. `M_F` is
  never formed; its blocks are contracted where they are used.
- **Public actions computed once.** Key generation runs the exponent-lut preprocessing, which
  publishes the output key of every Dynamic product and lookup and the product keys of every
  lookup branch. Decryption imports them, so the online evaluation computes only the encoding rows.

## Conventions

- An unauthorized decryption returns the rounded phase, which still contains the term
  `-F(x) (s G)_2 v_F`. It is not a reliable rejection, and toy parameters may reproduce the message
  by chance.
- The anchor row is the public matrix of the existing TrapGen, with its identity block and its
  `D + 2` columns. The manuscript's uniform-anchor and anchor-width requirements are not enforced.
- The setup's other public matrices are expanded from a public hash key and are assumed
  indistinguishable from the manuscript's uniform matrices.
- Gaussian widths are the sampler's standard deviation `sigma`; the manuscript's width is
  `s = sqrt(2 pi) sigma`.

## Tests

**Unit tests** use small, insecure parameters with zero noise: the errors of `chi` and `chi_fl`
are zero, so every evaluation is exact and the simulated bounds `E_F` and `E_PE` are zero. Each
round trip asserts that the simulated bound guarantees correctness (`4 E_PE + 2 < q`). The tests
check exact relations: keys for several predicates decrypt one ciphertext with zero phase error,
every block of the family values equals the secret times `M_F` (with the predicate term for
`F(x) = 1`), and keys solve `M_F k = d_1 delta_1`. The parameters are overridable with
`MXX_PE_TEST_{RING_DIMENSION,CRT_DEPTH,CRT_BITS,BASE_BITS,TRAPDOOR_SIGMA,ERROR_SIGMA,FLOODING_SIGMA,PROGRAMMING_SIGMA}`;
nonzero error sigmas need parameters whose bound still guarantees correctness.

```bash
cargo test -r -p mxx-pe --lib
```

`src/gpu.rs` runs the same kind of round trip with every stage on the GPU; it needs a CUDA device:

```bash
cargo test -r -p mxx-pe --lib --features gpu -- --ignored gpu::
```

**The integration test** `tests/gpu_round_trip.rs` runs with noise. It searches ring dimensions
and CRT depths for parameters whose worst-case simulated noise guarantees correctness and whose two
Ring-LWE instances, the encryption secret and TrapGen, reach the security target (128 bits by
default) according to `lattice-estimator-cli`. It then runs setup, encryption, and authorized and
unauthorized key generation and decryption on the GPU with file-backed artifacts. Its rustdoc
lists the `MXX_PE_ROUND_TRIP_*` variables. The estimator must be on `PATH` with its Sage
environment:

```bash
cargo test -r -p mxx-pe --features gpu --test gpu_round_trip
```

The integration test checks only correctness and these two estimates. It does not enforce the
manuscript's leftover-hash, noise-flooding, and key-width conditions. The anchor's identity column
from the existing TrapGen gives one sample with coefficient `1`, which the estimates do not model.

## Lean proofs

[`lean/`](lean/README.md) proves, in Lean, the correctness claim that the integration test exports
for a recorded parameter set of `lean/parameters.csv`. For `average-w2-n8192-crt5x30`, the tapes
on which a run meeting the input contract and the requirement decrypts a wrong message have
measure at most `2^-100`. The proof reuses the exponent-LUT Lean library and depends only on the
standard axioms.
