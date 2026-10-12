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
| `branching_program` | `PeClass` (attribute length, alphabet, state width, maximal length) and `BranchingProgram`, checked against its class and evaluated in the clear. `PeClass::compile` gives the class's Exponent-LUT Program, bound to the setup coordinates, and `BranchingProgram::data` its runtime data: the attribute index and the table of every step. `PeClass::with_steps` fixes the kind of every step (`StepKind`: lookup or map). |
| `params` | `PeParameters`: the class, the equal-basis chain of `2 T_max + 2` levels, the statistical trapdoor, the error contract `chi` of every fresh error, and the key width and cutoff; `PeParameters::select` picks the CRT depth from the budget's conditions and the simulated noise. |
| `budget` | The statistical trapdoor's conditions: the split of the statistical-distance target `ε = 2^-30` between the regularity (leftover hash lemma) of TrapGen, which gives `m̄` and `m_B = m̄ + D`, and the truncation multiplier of `chi`. |
| `setup` | `setup`: the anchor `B` from the statistical TrapGen `[ā \| g - ā T]`, the class's Dynamic setup, and the public message row. `PePublicKey` and `PeMasterSecretKey` keep public and secret material apart. |
| `encryption` | `encrypt(mpk, x, m)`: the anchor encoding, the attribute gadget ciphertexts `C_t = [a_t; -s_bar a_t + e_t] + X^(Bz x_t) G` (with a fresh uniform `a_t` and `e_t` from `chi`), the message component, and the Dynamic packages, under one fresh secret and session. |
| `keygen` | `keygen(msk, F)`: the predicate's public evaluation on its program data, including its output key `A_F`, and a short `k` with `M_F k = d_1 delta_1` for the implicit family matrix `M_F`. |
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
- **One graph for every program of a class.** The class compiles into one Program of `T_max` steps
  whose reads and tables are runtime data families; a shorter program is padded with leading
  identity steps. The Dynamic product selects the attribute package by the step's read, and the
  LUT scalars are computed from the step's table, so key generation and decryption run the same
  graph for every program, and the key's commitment binds its data.
- **Map steps.** A class may fix each step's kind (`PeClass::with_steps`): a map step applies a
  table to the state without reading an attribute, so it needs no product. The kinds are the
  class's because one graph serves every program.
- **Radix-`B` lookups.** `PeParameterRequest::lut_radix` decomposes the cyclic lookup group of
  `W = B^r` automorphisms in radix `B` (a power of two; `W <= n / 2` for `B < W`): a ciphertext
  then carries switching packages for the `(B - 1) log_B W` automorphisms `K_B = {g^(a B^j)}`
  instead of `W - 1`, and a key carries `W |J_B|` helper blocks for `J_B = K_B u K_B^-1`. A lookup
  branch is switched once per nonzero radix-`B` digit of its exponent, at most `log_B W` times,
  and the noise follows those switches. `B = W` (`lut_radix: None`) is the direct
  switching. Each switch multiplies the noise by a decomposition gain, so a small `B` trades
  packages for noise (at `W = 4096`, `T = 4`: `log2 E_F` 557 direct, 705 at `B = 64`, 853 at
  `B = 16`, 2147 at `B = 2`). `PeClass::default_lut_radix` takes 3 or 2 switches per branch when
  they divide `log2 W` (`W = 4096`: `B = 16`; `W = 2`: direct), and the integration and noise
  tests use it unless `LUT_RADIX` is set (0 selects the direct switching). `B < W` is outside the
  exported Lean correctness claim.
- **Public top rows of the attribute ciphertexts.** The top row `a_t` of every attribute gadget
  ciphertext `C_t = [a_t; X^v s G - s_bar a_t + e_t]` is hashed by the setup
  (`RnsBggDynamicSetup::rhs_public_top`), so a ciphertext carries only the bottom row and the
  companions of its `D` private digit rows: `m² / 2` per package instead of `m²`, as for the
  switching packages. Every ciphertext reuses `a_t` under a fresh `s_bar`.
- **Compact fresh encodings.** `PeParameterRequest::compact_encodings` publishes the digit
  companions, automorphism helpers, and the private input `c_0` as two-column rows
  `s (Ã - p X I_2) + e mod pq`, and decryption expands them to the full gadget width at `q`
  (see the exponent-lut crate). A package then holds `m²` polynomials over `pq` instead of
  `m³ / 2` over `q`. The ring dimension is selected for the lattice estimate at `pq`, and the noise
  simulation uses the expanded bounds. The Lean claim covers it for the average-case row
  `average-w2-n32768-crt6x30-compact`.
- **SampleLeft without the family matrix.** Key generation samples the one extension vector shared
  by every row of `M_F`, then calls the existing preimage sampler once per anchor block. `M_F` is
  never formed; its blocks are contracted where they are used.
- **Public actions computed once.** Key generation runs the exponent-lut preprocessing, which
  publishes the output key of every Dynamic product and lookup and the product keys of every
  lookup branch. Decryption imports them, so the online evaluation computes only the encoding rows.

## Trapdoor and key widths

TrapGen is the manuscript's statistical instantiation `[ā \| g - ā T]` with `m_B = m̄ + D`
columns. The parameters take the relaxations of the manuscript's implementation section, so they
are outside its theorem:

- Every fresh error is a `chi` sample: there is no noise flooding, and the reduction's hybrid
  `H_1` does not apply.
- The uniformity conditions on the matrices `R` and `Z_t`, which appear only in the hybrids, are
  dropped, so `m̄` follows from the regularity of TrapGen alone.
- Keys are sampled at the trapdoor's minimum preimage width `s_min`, which relaxes the width at
  which SampleRight simulates keys, as in the CP-ABE 1 of Datta et al. (ACM CCS 2026).

`budget` splits `ε = 2^-30` between the regularity of TrapGen and the truncation of `chi` at `c`
standard deviations. Selection checks these conditions and the numeric range of the key cutoff
(below `2^52` and `q / 2`) and returns `PeError::Parameters` otherwise. The Ring-LWE estimate
covers the encryption secret; TrapGen is uniform by regularity.

The budget bounds statistical distances assuming that every sampler returns its ideal discrete
Gaussian. The floating-point error of the sampling primitives (as in OpenFHE), the preimage widths,
and the truncation and rejection of preimages and key samples (cutoff `floor(6.5 σ_key)`) are
heuristic and not budgeted. The statistical key width `√(c_G² (1 + ŝ_1²) + r²)` uses the estimate
`ŝ_1 = σ_T √n (√m̄ + √D + √(ln(n / (2δ))))`, `δ = 2^-10`, of `s_1(T)`; Setup checks it against the
sampled `T` by factoring the perturbation covariance at every evaluation point, redraws `T` on a
failure, and fails after 8 draws, so no key is sampled from a trapdoor the estimate does not
cover. The factors (`(n / 2)(m̄ D + m̄ (m̄ + 1) / 2)` complex values) belong to the master secret key
and stay in GPU memory.

## Conventions

- An unauthorized decryption returns the rounded phase, which still contains the term
  `-F(x) (s G)_2 v_F`. It is not a reliable rejection, and toy parameters may reproduce the message
  by chance.
- The anchor row is the public matrix of TrapGen, statistically close to uniform by the
  regularity conditions.
- The setup's other public matrices are expanded from a public hash key and are assumed
  indistinguishable from the manuscript's uniform matrices.
- Gaussian widths are the sampler's standard deviation `sigma`; the manuscript's width is
  `s = sqrt(2 pi) sigma`.

## Tests

**Unit tests** use small, insecure parameters with zero noise: the errors of `chi` are zero, so every evaluation is exact and the simulated bounds `E_F` and `E_PE` are zero. Each
round trip asserts that the simulated bound guarantees correctness (`4 E_PE + 2 < q`). The tests
check exact relations: keys for several predicates decrypt one ciphertext with zero phase error,
every block of the family values equals the secret times `M_F` (with
the predicate term for `F(x) = 1`), and keys solve `M_F k = d_1 delta_1`. The statistical
selection rejects zero widths, so the tests use a fixture with an explicit `m̄`
(`MXX_PE_TEST_M_BAR`, default `2 D`) and the minimum key width; separate tests check that the
selection follows its budget and rejects narrow widths. The parameters are overridable with
`MXX_PE_TEST_{RING_DIMENSION,CRT_DEPTH,CRT_BITS,BASE_BITS,TRAPDOOR_SIGMA,ERROR_SIGMA}`; nonzero
error sigmas need parameters whose bound still guarantees correctness.

```bash
cargo test -r -p mxx-pe --lib
```

`src/gpu.rs` runs the same kind of round trip with every stage on the GPU; it needs a CUDA device:

```bash
cargo test -r -p mxx-pe --lib --features gpu -- --ignored gpu::
```

**The integration test** `tests/gpu_round_trip.rs` runs with noise. It searches ring dimensions
and CRT depths for parameters that the instantiation accepts (by default the statistical one with
`Practical` keys), whose simulated noise guarantees correctness, and whose Ring-LWE instances
reach the security target (128 bits by default) according to `lattice-estimator-cli`. It then runs
on the GPU setup, the encryption of a random attribute, and key generation and decryption for two
random programs of the class adjusted to output `0` (authorized) and `1` on it. By
default its matrix artifacts stay in GPU memory (`MXX_PE_ROUND_TRIP_STORAGE=device`); the
ciphertext's digit companions grow as `m² D n` polynomial coefficients, so large parameters, such
as `n = 2^14`, need `host` or `file` storage on a 16 GiB GPU. Its
rustdoc lists the `MXX_PE_ROUND_TRIP_*` variables. The estimator must be on `PATH` with its Sage
environment:

```bash
cargo test -r -p mxx-pe --features gpu --test gpu_round_trip
```

## Lean proofs

[`lean/`](lean/README.md) proves, in Lean, the correctness claim that the integration test exports
(`MXX_PE_ROUND_TRIP_LEAN_EXPORT`) for a parameter set of `lean/parameters.csv`. The claim takes the
attribute and the branching program as inputs, so it covers every attribute and every program of
the class with `F(x) = 0`. For the average-case rows
`average-w2-n8192-crt5x30`, `average-w2-n8192-crt4x42`, and the compact row
`average-w2-n32768-crt6x30-compact`, the tapes on which a run meeting the
input contract and the requirement decrypts a wrong message have measure at most `2^-128`.
`lean/generator/gen.py` writes the proof of such a row from its exported claim; the proofs reuse the
exponent-LUT Lean library and depend only on the standard axioms.
