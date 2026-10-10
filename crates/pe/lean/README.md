# PE Lean correctness proofs

This directory proves, in Lean, the correctness of the predicate encryption graphs that the PE
integration test executes. With `MXX_PE_ROUND_TRIP_LEAN_EXPORT=<parameter id>`,
`tests/gpu_round_trip.rs` exports the executed stages as a closed protocol with
`mxx_ir_core::lean::protocol::export`. Its claim is `GeneratedClaim.CorrectnessClaim`, and the
package of that parameter set proves it. The claim does not fix a predicate: key generation and
online evaluation take the branching program as an input, so one claim covers every attribute and
every program of the class.

## Layout

| Location | Contents |
| --- | --- |
| `lean/*.lean` | Lemmas independent of the parameters (package `MxxPe`) |
| `lean/parameters.csv` | The recorded parameter sets |
| `lean/<parameter id>/` | The proof of one parameter set (a Lake package that requires `MxxPe`) |
| `lean/<parameter id>/generated/` | The claim the integration test exported deterministically; never edited by hand |
| `lean/generator/` | `gen.py`, which writes the proof package of a parameter set from its row and its exported claim, and the templates it fills |

`MxxPe` requires the exponent-LUT package `crates/exponent-lut/lean` (`MxxExponentLut`), whose
encoding relation `Rel`, Dynamic products, digit companions, gadget lemmas, and runtime equations
the proofs reuse. Mathlib and the ir-core Lean library come through it.

### Parameter-independent lemmas

| File | Contents |
| --- | --- |
| `PeDynamic.lean` | The Dynamic LUT branch of a key-homomorphic automorphism: the switching ciphertext `C_k` opened by the helper of `k⁻¹` (`switching_rhs`), the switched product under the routed mask (`aut_switch`), the mask switch by the helper of `k` (`mask_switch`), the public scalar action (`Rel.blockAction`), and the digits of a gadget row (`decompose_row`) |
| `PeCompact.lean` | The expansion of a compact encoding: a compact encoding of `P x` over `PQ` times the centered gadget `G̃`, switched to `Q`, is an encoding of `x g` under the expanded key with the named error `expandErr` of at most `(B ⌊Q/2⌋ + (N + 2) ⌊P/2⌋) / P` (`RelE.expandCompact`) |
| `PeTail.lean` | The tail of a linear form in fresh Gaussian tape keys whose coefficients read only other keys and have bounded energy: some of `rows` forms reaches `m` with probability at most `rows · 2 exp(-m² / (2 σ² E))` (`tapeMeasure_linear_tail`) |

## Parameter IDs

`parameters.csv` records the parameter sets of the proofs under unique `parameter_id`s; `gen.py`
reads a row to write its proof package and checks that the exported layout matches it. The
integration test does not read the file: run with the environment variables that select the row's
parameters (see [Reproduction](#reproduction)) and `MXX_PE_ROUND_TRIP_LEAN_EXPORT` set to the ID,
it searches the parameters as usual, runs the round trip on a random attribute with random
programs adjusted to it, and exports the claim to `<parameter id>/generated`.

| Column | Meaning |
| --- | --- |
| `noise_model` | `worst` (`4 E_PE + 2 < q`) or `average` (failure probability at most `2^-failure_exponent`) |
| `failure_exponent` | For `average`, the exponent `k` of the failure bound `2^-k`; empty for `worst` |
| `attribute_length`, `alphabet_size`, `state_width`, `max_steps` | The class of branching programs |
| `ring_dimension`, `crt_bits`, `base_bits`, `crt_depth` | The equal-basis RNS chain and the gadget base |
| `error_sigma`, `flooding_sigma`, `programming_sigma`, `trapdoor_sigma` | Widths of `chi`, `chi_fl`, `Z_t`, and TrapGen |
| `encryption_security_bits`, `trapdoor_security_bits` | Lattice security of the encryption secret and of TrapGen, estimated by `lattice-estimator-cli` (`trapdoor_security_bits` is empty for the statistical instantiation, whose TrapGen has no estimate) |
| `trapdoor_instantiation` | `computational` (`[ā \| I \| G - (ā R + E)]`, `m_B = D + 2`) or `statistical` (`[ā \| g - ā T]`, `m_B = m̄ + D`) |
| `key_width_policy`, `m_bar`, `randomness_sigma` | For a statistical row: `practical` or `strict`, the selected `m̄`, and the sigma of the proof's `R` |
| `cutoff_multiplier` | The truncation multiplier of `chi`, `chi_fl`, and `Z_t`: `6.5` (computational) or the budget's `c` to four decimals (statistical) |
| `compact_encodings` | `1` if the fresh encodings are compact two-column rows over `PQ` (`PeParameterRequest::compact_encodings`), `0` otherwise |

| ID | Parameters | Proof |
| --- | --- | --- |
| `worst-w2-n16384-crt4x50` | One attribute bit and one step of width 1 (class `(1, 2, 1, 1)`), `n = 2^14`, four 50-bit primes, 276/283-bit security | None; its claim is not checked in |
| `average-w2-n8192-crt5x30` | The same class, `n = 2^13`, five 30-bit primes (`Q ≈ 2^150`), 15-bit digits, 160/166-bit security, failure `2^-128` | `average-w2-n8192-crt5x30/PeCertificate.lean` |
| `average-w2-n8192-crt4x42` | The same class, `n = 2^13`, four 42-bit primes (`Q ≈ 2^168`), 21-bit digits, 137/141-bit security, failure `2^-128` | `average-w2-n8192-crt4x42/PeCertificate.lean` |
| `average-w2-n32768-crt6x30-compact` | The same class with compact fresh encodings, `n = 2^15`, six 30-bit primes (`Q ≈ 2^180`, `P ≈ 2^180`), 15-bit digits, 318-bit encryption security at `PQ` (the trapdoor estimate saturates at `u32::MAX`), failure `2^-128` | `average-w2-n32768-crt6x30-compact/PeCertificate.lean` |

## The claim

`GeneratedClaim.CorrectnessClaim` states that, for every hash model and external inputs, the tapes
with a run that meets the input contract and the requirement and whose decrypted message differs
from the ideal one (the encrypted message) have measure at most `2^-failure_exponent`. The external
inputs are the hash key, the attribute exponents, the message bits, the coefficient and exponent of
the initial state, and the branching program as the data of the class program: the attribute index
`reads` read at every step and the LUT `tables` of every step. The input contract bounds the
attribute in `[0, 1]`, the reads in `[0, L - 1]`, and the table entries in `[0, W - 1]`; the
requirement accepts exactly the attributes and programs with `F(x) = 0`. For the class
`(1, 2, 1, 1)`, the program is a table `(t₀, t₁)`, and the requirement gives `t_{x₀} = 0`
(`requirement_table`).

## The proof of an average-case parameter set

The certificate `PeCertificate.certificate : GeneratedClaim.CorrectnessClaim` is checked by
`#guard_msgs` to depend only on `propext`, `Classical.choice`, and `Quot.sound`. The argument has
three parts.

**Every run decrypts up to a linear form in the key extension.** The stage lemmas give every value
of a run as an explicit function of the hash model, the hash key, and the tape (`PeModel`), with
every error named by its tape site. The phase of decryption is
`⌊Q/2⌋ m + e_m - (e_b k̃₀ + σ_K(e_b k̃₁)) - (e_h v₀ + σ_K(e_h) v₁ + e_F v_F)` (`PePhase`): the bias
`e_m - e_b k̃` is bounded on every tape by the cutoffs of `e_m`, `e_b`, and the preimages, and the
remainder is a linear form in the key extension `v = (v₀, v₁, v_F)`, whose coefficients are the
helper error `e_h` and the error `e_F` of the online output. The product encodes `X^{x₀}`, and the
LUT of the table `(t₀, t₁)` maps it to `X^{x₀} D_1 + σ_K(X^{x₀}) D_K = t_{x₀}`, which is `0` by the
requirement (`lutScalar_sum`). `e_F` is bounded by `onlineBound` (`PeOnlineRel`; `2^101.7` for
`average-w2-n8192-crt5x30`, `2^118.4` for `average-w2-n8192-crt4x42` and
`average-w2-n32768-crt6x30-compact`), from the Dynamic product, the identity branch, and the branch
of `K`. Each coefficient whose linear form stays below `mT ≈ Q/4` decodes to its message bit
(`run_decode`).

**The coefficients read no key of the extension.** `e_h` and `e_F` are functions of `prims`, the
samples of the secret, TrapGen, the prelude, and encryption, which read only the tape keys of
`preKeys` (`PeIndep`). The extension is sampled at the sites `[6, 10, 0, 0]`, `[6, 10, 1, 0]`, and
`[6, 7]`, none of which is in `preKeys`.

**The tail.** The coefficients are clamped to their bounds, so their energy is at most
`E = m N (c² + c² + onlineBound²)` on every tape, for the encoding width `m` and the cutoff `c` of
`e_h`. `tapeMeasure_linear_tail` bounds the probability that some coefficient of the linear form
reaches `mT` by `N · 2 exp(-mT² / (2 σ² E))` for the key width `σ`, and `mT² / (2 σ² E) ≥ k + 20`
for the row's failure exponent `k` (`numeric_margin`, decided on the parameters), so the bound is
at most `2^-(k+6) ≤ 2^-k` (`tail_numeric`); the recorded rows use `k = 128`.

| Module | Contents |
| --- | --- |
| `PeSetup` | The ring, the modulus, `K = N + 1`, the hash tags, and the keys hashed from the hash key |
| `PeOnline` | The output rows of online evaluation: the column loop of the product at the member that the program reads, its companion loop, the companion loop of the automorphism, and the branch of `K` with the scalars of the program's table (`online_rows`) |
| `PeOnlineRel` | The online output is an encoding of `t_{x₀}` under the output key `A_F = M V_1 + (H Dec(T)) V_K`, so of `0` when `t_{x₀} = 0` (`online_rel`, `lutScalar_sum`) |
| `PePreprocess` | Preprocessing outputs the companion keys `M` of the member that the program reads, the switched keys `T`, and `A_F` for the program's table (`preprocess_keys`) |
| `PeEncrypt` | Encryption: the initial state, the helper of `K`, the switching ciphertext, and the digit companions of the attribute and switching ciphertexts, with named errors (`RelE`) and their cutoffs |
| `PePrelude` | The anchor encoding, the attribute ciphertext `C = P Z + X^{x₀} G` and its opening `-(e_b Z)`, and the message component |
| `PeKeygen` | The key extension and the targets of the preimages under the anchor |
| `PeDecode` | The phase that the decode stage rounds |
| `PePhase` | The cancellation of every secret-dependent term of the phase |
| `PeModel` | The model of a run as a function of `prims`, the tape, the attribute, and the table |
| `PeLink` | Every run follows the model; the online error `e_F` of the model is within `onlineBound` |
| `PeIndep` | `preKeys` and `prims_congr`: tapes that agree on `preKeys` have the same samples |
| `PeInteger` | The phase as the reduction of an integer polynomial and the coefficients of the linear form |
| `PeCertificate` | The requirement (`requirement_table`), the decryption of every coefficient, the coefficient energy, the numeric tail, and the certificate |

The proof covers correctness only. The security estimates of the parameter set come from
`lattice-estimator-cli` in the integration test and are not part of the Lean claim.

## Generation

The proof of every average-case row is written by `generator/gen.py` from the row and its exported
claim. The templates are written for the class `(1, 2, 1, 1)` (every attribute and every program of
the class), ring dimension `2^13`, two digits per CRT prime (`base_bits = crt_bits / 2`), and the average noise
model, for the computational instantiation with the default cutoffs; the script rejects other
rows, including statistical ones, whose TrapGen the templates do not model. It reads the modulus,
the CRT basis, the gadget base, the Gaussian laws (each `(width, cutoff)` pair, of any size), the
widths, and the hash tags from the exported claim, derives the digit count `D`, the encoding width
`m = 2D`, the companion lanes `m²`, and the anchor width `D + 2`, and checks the numeric conditions
the proof decides before writing the package.

Encryption samples `Z_t` one column `z_j` at a time in a sequential loop and adds `P z_j` at
column `j` through a window of a constant unit row; `PePrelude` reads the rows `P Z_t` back with
`IterRuns.accumulate_entries`, and `PeIndep` lists the column sites `[3, 11, 0, 3, j, 2]`.

A row with `compact_encodings = 1` is filled from `generator/templates_compact` instead, for any
power-of-two ring dimension. Encryption there publishes the input `c_0`, the helper of `K`, and the
digit companions as compact rows over `PQ` with `1 × 1` errors (`PeEncrypt`), and every consumer
reads their expansions `expand` (`PeSetup`, by `RelE.expandCompact`). The expanded input, helper,
and companions carry errors within `(c ⌊Q/2⌋ + (N + 2) ⌊P/2⌋) / P` for the cutoff `c` (16395 for
`average-w2-n32768-crt6x30-compact`), which replace the cutoffs in `onlineBound` and in the
energy of the helper error.

## Reproduction

Export the claim of a recorded parameter set (`lattice-estimator-cli` must be on the `PATH` with its
Sage environment). The variables select the row's ring dimension, CRT and digit widths,
instantiation, noise model, and encodings; the search then accepts the row's CRT depth. The claim
does not depend on the sampled attribute and programs, so rerunning an ID writes identical
`generated` files:

```sh
MXX_PE_ROUND_TRIP_MIN_LOG_RING_DIMENSION=13 MXX_PE_ROUND_TRIP_MAX_LOG_RING_DIMENSION=13 \
  MXX_PE_ROUND_TRIP_CRT_BITS=30 MXX_PE_ROUND_TRIP_BASE_BITS=15 \
  MXX_PE_ROUND_TRIP_TRAPDOOR_INSTANTIATION=computational MXX_PE_ROUND_TRIP_NOISE_MODEL=average \
  MXX_PE_ROUND_TRIP_COMPACT_ENCODINGS=0 MXX_PE_ROUND_TRIP_LEAN_EXPORT=average-w2-n8192-crt5x30 \
  cargo test -r -p mxx-pe --features gpu --test gpu_round_trip -- --nocapture
```

| ID | Variables that differ from the command above |
| --- | --- |
| `average-w2-n8192-crt4x42` | `CRT_BITS=42`, `BASE_BITS=21` |
| `average-w2-n32768-crt6x30-compact` | `MIN_LOG_RING_DIMENSION=15`, `MAX_LOG_RING_DIMENSION=15`, `COMPACT_ENCODINGS=1` |

Write and check the proof of the exported claim:

```sh
python3 crates/pe/lean/generator/gen.py average-w2-n8192-crt4x42
cd crates/pe/lean/average-w2-n8192-crt4x42
lake build
```

`lake build` builds the exported claim (`PeGenerated`) and its proof (`PeProof`).
