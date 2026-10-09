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
| `params` | `PeParameters`: the class, the equal-basis chain of `2 T_max + 2` levels, the trapdoor instantiation, the error, flooding, and programming contracts, and the key width and cutoff; `PeParameters::select` picks the CRT depth from the conditions of the instantiation and the simulated noise. |
| `budget` | The statistical instantiation's conditions: the split of the statistical-distance target `ε_total` (default `2^-30`), the regularity (leftover hash lemma) column counts `m̄` and `m_B`, the truncation and flooding multipliers, the flooding width, and the SampleRight key width of the `Strict` policy. |
| `setup` | `setup`: the anchor `B` from TrapGen (computational `[ā \| I \| G - (ā R + E)]` or statistical `[ā \| g - ā T]`), the class's Dynamic setup, and the public message row. `PePublicKey` and `PeMasterSecretKey` keep public and secret material apart. |
| `encryption` | `encrypt(mpk, x, m)`: the anchor encoding, the attribute gadget ciphertexts `C_t = P Z_t + X^(Bz x_t) G` (with `Z_t` sampled one column at a time), the message component, and the Dynamic packages, under one fresh secret and session. |
| `keygen` | `keygen(msk, F)`: the predicate's public evaluation, including its output key `A_F`, and a short `k` with `M_F k = d_1 delta_1` for the implicit family matrix `M_F`. |
| `decryption` | `decrypt(sk_F, ct)`: the predicate evaluated on the ciphertext's packages and the family row contracted with the key, rounded to the message bits. |
| `noise` | `simulate`: the evaluation bound `E_F` from the exponent-lut simulator, the decryption bound `E_PE`, and the condition `4 E_PE + 2 < q`; `witness_bound`: the bound `B_R̂` of the proof's ledger witness, from the same simulator with witness contracts. |

## Design

- **Programs as exponents.** The state `z` of a branching program is the exponent of a monomial.
  Each step multiplies it by the package of the attribute it reads, giving `X^(z + Bz x)`, and
  applies the step's table as an Exponent-LUT lookup. The last table returns the output bit as a
  coefficient, and its full-gadget carrier `c_F = s (A_F - F(x) G) + e_F` is never directly
  decoded.
- **One ciphertext for every predicate.** Packages are bound to setup coordinates, not to a
  Program, level, or table, so a ciphertext's size depends on the attribute length and the class's
  lookup width but not on the program length or the predicate.
- **Radix-`B` lookups.** `PeParameterRequest::lut_radix` decomposes the cyclic lookup group of
  `W = B^r` automorphisms in radix `B` (a power of two; `W <= n / 2` for `B < W`): a ciphertext
  then carries switching packages for the `(B - 1) log_B W` automorphisms `K_B = {g^(a B^j)}`
  instead of `W - 1`, and a key carries `W |J_B|` helper blocks for `J_B = K_B u K_B^-1`. A lookup
  branch is switched once per nonzero radix-`B` digit of its exponent, at most `log_B W` times,
  and the noise, the ledger witness `B_R̂`, and the `Strict` width `s_SR` (one decomposition gain
  per step of a helper path) follow those switches. `B = W` (the default) is the direct
  switching; `B < W` is outside the exported Lean correctness claim.
- **Compact fresh encodings.** `PeParameterRequest::compact_encodings` publishes the digit
  companions, automorphism helpers, and the private input `c_0` as two-column rows
  `s (Ã - p X I_2) + e mod pq`, and decryption expands them to the full gadget width at `q`
  (see the exponent-lut crate). A package then holds `2 m²` polynomials over `pq` instead of
  `m³` over `q`. The ring dimension is selected for the lattice estimate at `pq`, and the noise
  simulation uses the expanded bounds. This option is outside the exported Lean correctness claim.
- **SampleLeft without the family matrix.** Key generation samples the one extension vector shared
  by every row of `M_F`, then calls the existing preimage sampler once per anchor block. `M_F` is
  never formed; its blocks are contracted where they are used.
- **Public actions computed once.** Key generation runs the exponent-lut preprocessing, which
  publishes the output key of every Dynamic product and lookup and the product keys of every
  lookup branch. Decryption imports them, so the online evaluation computes only the encoding rows.

## Instantiations and key widths

| | `Statistical` (the manuscript) | `Computational` (legacy) |
| --- | --- | --- |
| TrapGen | `[ā \| g - ā T]`, `m_B = m̄ + D` columns | `[ā \| I \| G - (ā R + E)]`, `m_B = D + 2` |
| `m_B` | the regularity conditions of `budget` | `D + 2` |
| `chi_fl` | `σ_fl = 2^s_flood B_coef / √(2π)`, from the flooding condition; the requested sigma when `enforce_flooding` is off (outside the reduction) | the requested sigma |
| Cutoffs of `chi`, `chi_fl`, `Z_t` | `c` standard deviations from the budget | `6.5` standard deviations |
| Checked conditions | regularity, flooding (unless `enforce_flooding` is off), budget, key width, numeric range (`PeError::Parameters` otherwise) | none |
| Ring-LWE estimates | encryption | encryption and TrapGen |
| Key width | `Strict`: `max(s_min, s_SR)`; `Practical`: `s_min` | the trapdoor's minimum preimage width |

- `Statistical` with `Strict` implements every construction and parameter condition of the
  manuscript and corresponds to its security theorem, under the heuristics stated below.
- `Statistical` with `Practical` relaxes one condition of the reduction, the width at which
  SampleRight simulates keys, as in the CP-ABE 1 of Datta et al. (ACM CCS 2026). Benchmarks use
  this setting.
- `Computational` keeps the legacy parameters for compatibility. It is heuristic and outside the
  manuscript's theorem.

The budget bounds statistical distances assuming that every sampler returns its ideal discrete
Gaussian. The floating-point error of the sampling primitives (as in OpenFHE), the preimage widths
(the same heuristic singular-value estimate as the existing sampler), and the truncation and
rejection of preimages and key samples (cutoff `floor(6.5 σ_key)`) are heuristic and not budgeted.
Flooding is evaluated per coefficient at `2^-s_flood` (default `s_flood = 30`), separately from
`ε_total`, as in Noah's Ark. The `Strict` key width grows with the witness bound `B_R̂` and is
rejected when its key cutoff is not below `2^52` and `q / 2`.

## Conventions

- An unauthorized decryption returns the rounded phase, which still contains the term
  `-F(x) (s G)_2 v_F`. It is not a reliable rejection, and toy parameters may reproduce the message
  by chance.
- The anchor row is the public matrix of TrapGen. In the statistical instantiation it is
  statistically close to uniform by the regularity conditions. In the computational one it keeps
  the identity block and its `D + 2` columns, and the manuscript's uniform-anchor and anchor-width
  requirements are not enforced.
- The setup's other public matrices are expanded from a public hash key and are assumed
  indistinguishable from the manuscript's uniform matrices.
- Gaussian widths are the sampler's standard deviation `sigma`; the manuscript's width is
  `s = sqrt(2 pi) sigma`.

## Tests

**Unit tests** use small, insecure parameters with zero noise: the errors of `chi` and `chi_fl`
are zero, so every evaluation is exact and the simulated bounds `E_F` and `E_PE` are zero. Each
round trip asserts that the simulated bound guarantees correctness (`4 E_PE + 2 < q`). The tests
check exact relations: keys for several predicates decrypt one ciphertext with zero phase error
(in both instantiations), every block of the family values equals the secret times `M_F` (with
the predicate term for `F(x) = 1`), and keys solve `M_F k = d_1 delta_1`. The statistical
selection rejects zero widths, so the statistical round trip uses a test fixture with an explicit
`m̄` and the minimum key width; separate tests check that the selection follows its budget,
rejects narrow widths and wide anchors, and that `Strict` keys add the SampleRight width or are
rejected as unrepresentable. The parameters are overridable with
`MXX_PE_TEST_{RING_DIMENSION,CRT_DEPTH,CRT_BITS,BASE_BITS,TRAPDOOR_SIGMA,ERROR_SIGMA,FLOODING_SIGMA,PROGRAMMING_SIGMA}`,
the instantiation with `MXX_PE_TEST_TRAPDOOR_INSTANTIATION` (`computational` or `statistical`,
with `MXX_PE_TEST_M_BAR`), and the selection request with
`MXX_PE_TEST_{KEY_WIDTH_POLICY,RANDOMNESS_SIGMA}`; nonzero error sigmas need parameters whose
bound still guarantees correctness.

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
setup, encryption, and authorized and unauthorized key generation and decryption on the GPU. By
default its matrix artifacts stay in GPU memory (`MXX_PE_ROUND_TRIP_STORAGE=device`); the
ciphertext's digit companions grow as `m² D n` polynomial coefficients, so large parameters, such
as the statistical instantiation at `n = 2^14`, need `host` or `file` storage on a 16 GiB GPU. Its
rustdoc lists the `MXX_PE_ROUND_TRIP_*` variables. The estimator must be on `PATH` with its Sage
environment:

```bash
cargo test -r -p mxx-pe --features gpu --test gpu_round_trip
```

For the computational instantiation, the integration test checks only correctness and the two
estimates; the anchor's identity column gives one sample with coefficient `1`, which the estimates
do not model.

## Lean proofs

[`lean/`](lean/README.md) proves, in Lean, the correctness claim that the integration test exports
for a recorded parameter set of `lean/parameters.csv`. For the average-case rows
`average-w2-n8192-crt5x30` and `average-w2-n8192-crt4x42`, the tapes on which a run meeting the
input contract and the requirement decrypts a wrong message have measure at most `2^-128`.
`lean/generator/gen.py` writes the proof of such a row from its exported claim; the proofs reuse the
exponent-LUT Lean library and depend only on the standard axioms.
