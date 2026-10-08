# PE Lean correctness proofs

This directory proves, in Lean, the correctness of the predicate encryption graphs that the PE
integration test executes. Run with a recorded parameter set, `tests/gpu_round_trip.rs` exports
the executed stages as a closed protocol with `mxx_ir_core::lean::protocol::export`. Its claim
is `GeneratedClaim.CorrectnessClaim`, and the package of that parameter set proves it.

## Layout

| Location | Contents |
| --- | --- |
| `lean/*.lean` | Lemmas independent of the parameters (package `MxxPe`) |
| `lean/parameters.csv` | The recorded parameter sets |
| `lean/<parameter id>/` | The proof of one parameter set (a Lake package that requires `MxxPe`) |
| `lean/<parameter id>/generated/` | The claim the integration test exported deterministically; never edited by hand |

`MxxPe` requires the exponent-LUT package `crates/exponent-lut/lean` (`MxxExponentLut`), whose
encoding relation `Rel`, Dynamic products, digit companions, gadget lemmas, and runtime equations
the proofs reuse. Mathlib and the ir-core Lean library come through it.

### Parameter-independent lemmas

| File | Contents |
| --- | --- |
| `PeDynamic.lean` | The Dynamic LUT branch of a key-homomorphic automorphism: the switching ciphertext `C_k` opened by the helper of `k⁻¹` (`switching_rhs`), the switched product under the routed mask (`aut_switch`), the mask switch by the helper of `k` (`mask_switch`), the public scalar action (`Rel.blockAction`), and the digits of a gadget row (`decompose_row`) |
| `PeTail.lean` | The tail of a linear form in fresh Gaussian tape keys whose coefficients read only other keys and have bounded energy: some of `rows` forms reaches `m` with probability at most `rows · 2 exp(-m² / (2 σ² E))` (`tapeMeasure_linear_tail`) |

## Parameter IDs

`parameters.csv` records reproducible parameter sets under unique `parameter_id`s. With
`MXX_PE_ROUND_TRIP_PARAMETER_ID` set to an ID, the integration test runs that row without
searching, checks that the row's noise model accepts the parameters and that the recorded security
estimates are reproduced, runs the round trip on an attribute the row's predicate authorizes, and
exports the claim to `<parameter id>/generated`.

| Column | Meaning |
| --- | --- |
| `noise_model` | `worst` (`4 E_PE + 2 < q`) or `average` (failure probability at most `2^-failure_exponent`) |
| `failure_exponent` | For `average`, the exponent `k` of the failure bound `2^-k`; empty for `worst` |
| `attribute_length`, `alphabet_size`, `state_width`, `max_steps` | The class of branching programs |
| `reads`, `tables` | The predicate of the round trip: the attribute index read at every step and the LUT table of every step (`:`-separated entries) |
| `ring_dimension`, `crt_bits`, `base_bits`, `crt_depth` | The equal-basis RNS chain and the gadget base |
| `error_sigma`, `flooding_sigma`, `programming_sigma`, `trapdoor_sigma` | Widths of `chi`, `chi_fl`, `Z_t`, and TrapGen |
| `statistical_bits` | Tail budget of every cutoff |
| `encryption_security_bits`, `trapdoor_security_bits` | Lattice security of the encryption secret and of TrapGen, estimated by `lattice-estimator-cli` |

| ID | Parameters | Proof |
| --- | --- | --- |
| `worst-w2-n16384-crt4x50` | One attribute bit, predicate `F(x) = x₀`, `n = 2^14`, four 50-bit primes, 276/283-bit security | None; its claim is not checked in |
| `average-w2-n8192-crt5x30` | The same class and predicate, `n = 2^13`, five 30-bit primes (`Q ≈ 2^150`), 15-bit digits, 160/166-bit security, failure `2^-100` | `average-w2-n8192-crt5x30/PeCertificate.lean` |

## The claim

`GeneratedClaim.CorrectnessClaim` states that, for every hash model and external inputs, the tapes
with a run that meets the input contract and the requirement and whose decrypted message differs
from the ideal one (the encrypted message) have measure at most `2^-failure_exponent`. The external
inputs are the hash key, the attribute exponents, the message bits, and the coefficient and
exponent of the initial state. The requirement accepts exactly the attributes that the predicate
authorizes (`F(x) = 0`).

## The proof of `average-w2-n8192-crt5x30`

The certificate `PeCertificate.certificate : GeneratedClaim.CorrectnessClaim` is checked by
`#guard_msgs` to depend only on `propext`, `Classical.choice`, and `Quot.sound`. The argument has
three parts.

**Every run decrypts up to a linear form in the key extension.** The stage lemmas give every value
of a run as an explicit function of the hash model, the hash key, and the tape (`PeModel`), with
every error named by its tape site. The phase of decryption is
`⌊Q/2⌋ m + e_m - (e_b k̃₀ + σ_K(e_b k̃₁)) - (e_h v₀ + σ_K(e_h) v₁ + e_F v_F)` (`PePhase`): the bias
`e_m - e_b k̃` is bounded on every tape by the cutoffs of `e_m`, `e_b`, and the preimages, and the
remainder is a linear form in the key extension `v = (v₀, v₁, v_F)`, whose coefficients are the
helper error `e_h` and the error `e_F` of the online output. `e_F` is bounded by
`onlineBound ≈ 2^102.3` (`PeOnlineRel`), from the Dynamic product, the identity branch, and the
branch of `K` of the LUT of the table `(0, 1)`. Each coefficient whose linear form stays below
`mT ≈ 2^148` decodes to its message bit (`run_decode`).

**The coefficients read no key of the extension.** `e_h` and `e_F` are functions of `prims`, the
samples of the secret, TrapGen, the prelude, and encryption, which read only the tape keys of
`preKeys` (`PeIndep`). The extension is sampled at the sites `[6, 10, 0, 0]`, `[6, 10, 1, 0]`, and
`[6, 7]`, none of which is in `preKeys`.

**The tail.** The coefficients are clamped to their bounds, so their energy is at most
`E = 20 N (30² + 30² + onlineBound²)` on every tape. `tapeMeasure_linear_tail` bounds the
probability that some coefficient of the linear form reaches `mT` by
`N · 2 exp(-mT² / (2 σ² E))`, and `mT² / (2 σ² E) ≥ 200` for the key width `σ ≈ 5.18 · 10^8`, so
the bound is at most `2^-186 ≤ 2^-100` (`tail_numeric`).

| Module | Contents |
| --- | --- |
| `PeSetup` | The ring, the modulus, `K = 8193`, the hash tags, and the keys hashed from the hash key |
| `PeOnline` | The output rows of online evaluation: the column loop of the product, its companion loop, the companion loop of the automorphism, and the branch of `K` (`online_rows`) |
| `PeOnlineRel` | The online output is an encoding of `0` under the output key `A_F = M V_1 + (H Dec(T)) V_K` (`online_rel`); `D_1 + D_K = 0` for the table `(0, 1)` |
| `PePreprocess` | Preprocessing outputs the companion keys `M`, the switched keys `T`, and `A_F` (`preprocess_keys`) |
| `PeEncrypt` | Encryption: the initial state, the helper of `K`, the switching ciphertext, and the digit companions of the attribute and switching ciphertexts, with named errors (`RelE`) and their cutoffs |
| `PePrelude` | The anchor encoding, the attribute ciphertext `C = P Z + G` and its opening `-(e_b Z)`, and the message component |
| `PeKeygen` | The key extension and the targets of the preimages under the anchor |
| `PeDecode` | The phase that the decode stage rounds |
| `PePhase` | The cancellation of every secret-dependent term of the phase |
| `PeModel` | The model of a run as a function of `prims` and the tape |
| `PeLink` | Every run follows the model; the online error `e_F` of the model is within `onlineBound` |
| `PeIndep` | `preKeys` and `prims_congr`: tapes that agree on `preKeys` have the same samples |
| `PeInteger` | The phase as the reduction of an integer polynomial and the coefficients of the linear form |
| `PeCertificate` | The decryption of every coefficient, the coefficient energy, the numeric tail, and the certificate |

The proof covers correctness only. The security estimates of the parameter set come from
`lattice-estimator-cli` in the integration test and are not part of the Lean claim.

## Reproduction

Export the claim of a recorded parameter set (`lattice-estimator-cli` must be on the `PATH` with its
Sage environment); the export is deterministic, so rerunning an ID writes identical `generated`
files:

```sh
MXX_PE_ROUND_TRIP_PARAMETER_ID=average-w2-n8192-crt5x30 \
  cargo test -r -p mxx-pe --features gpu --test gpu_round_trip -- --nocapture
```

Check the proof in its parameter directory:

```sh
cd crates/pe/lean/average-w2-n8192-crt5x30
lake build
```

`lake build` builds the exported claim (`PeGenerated`) and its proof (`PeProof`).
