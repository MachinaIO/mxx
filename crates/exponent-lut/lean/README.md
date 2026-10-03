# Exponent-LUT Lean correctness proofs

This directory proves, in Lean, the worst-case and average-case correctness of the graphs that the
exponent-LUT integration tests actually execute. Run with a recorded parameter set, an integration
test exports the executed stages as a closed protocol with `mxx_ir_core::lean::protocol::export`;
its claim is `GeneratedClaim.CorrectnessClaim`, and the package of that parameter set proves it.

## Layout

| Location | Contents |
| --- | --- |
| `lean/*.lean` | Lemmas independent of the program and the parameters (package `MxxExponentLut`) |
| `lean/<test>/` | Lemmas about the program of integration test `<test>` that hold at every parameter set, and `parameters.csv` |
| `lean/<test>/<parameter id>/` | The proof of one parameter set (a Lake package) |
| `lean/<test>/<parameter id>/generated/` | The claim the integration test exported deterministically; never edited by hand |

`<test>` is `nand_circuit` (`tests/nand_circuit.rs`) or `lwr_prf` (`tests/lwr_prf.rs`). Each
package requires the package one directory up, and Mathlib and the ir-core Lean library come from
`crates/ir-core/lean`.

### Program- and parameter-independent lemmas

| File | Contents |
| --- | --- |
| `ExponentAutomorphism.lean` | The odd substitution `X ↦ X^σ` and its norm bound |
| `ExponentScale.lean` | `scaleUp`, the runtime ModDown (`blockModSwitchRuns`) and its remainder |
| `ExponentEncoding.lean` | The encoding relation `Rel` (`c_j = s₀ A_{0j} + A_{1j} - t_j + e_j`): sums, actions, substitution, ModDown |
| `ExponentGadget.lean` | Gadget reconstruction from decompositions, uniqueness of gadgets and rebases |
| `ExponentStacking.lean` | Row and column concatenation and slicing |
| `ExponentWalsh.lean` | Walsh (Fourier) interpolation of a LUT |
| `ExponentLut.lean` | Private multiplication with and without ModDown, one LUT branch, and the LUT ModDown |
| `ExponentPublic.lean` | Public multiplication and the public alignment of a public encoding to a lower modulus |
| `ExponentDecode.lean` | Terminal threshold decryption of binary coefficients and of the parity of coefficients in `[-2, 2]` |
| `ExponentSampling.lean` | Tape samplers (`…SampleAt`) as deterministic samplers |
| `ExponentRuntime.lean` | Runtime relations as equations: hashes, family reads, ternary and Gaussian samples, plaintext monomials |

### Program-dependent lemmas

- `nand_circuit/NandProgram.lean`: the four LUT tables of the depth-one NAND program and the
  exponent of every round; `rounds_nand` shows that the output exponent is `1 - b₀ b₁`.
- `lwr_prf/LwrProgram.lean`: the sparse-LWR readout table `τ(k) = ⌊p' k / w⌋ + 1`.

## Parameter IDs

`<test>/parameters.csv` records reproducible parameter sets under unique `parameter_id`s. With
`MXX_EXPONENT_LUT_NAND_PARAMETER_ID` or `MXX_EXPONENT_LUT_LWR_PRF_PARAMETER_ID` set to an ID, the
integration test runs that row without searching, checks that the recorded lattice security
(`security_bits`) is reproduced, and exports the claim to `<parameter id>/generated`.

Common columns:

| Column | Meaning |
| --- | --- |
| `noise_model` | `worst` or `average` |
| `failure_exponent` | For `average`, the exponent `k` of the failure bound `2^-k`; empty for `worst` |
| `ring_dimension`, `crt_bits`, `base_bits`, `crt_depth`, `primes_per_drop`, `operations_per_drop` | RNS-BGG parameters |
| `error_sigma` | Width of the encoding error |
| `security_bits` | Lattice security estimated by `lattice-estimator-cli` |

`nand_circuit` adds `nand_depth`; `lwr_prf` adds `input_modulus` (`Q`), `output_modulus` (`P`),
`lwr_dimension` (`ν`), `support_weight` (`h`), `outputs`, and `key_seed`, the seed of the PBC layout
and the key.

| Test | IDs | Parameters |
| --- | --- | --- |
| `nand_circuit` | `worst-n32-crt7x40`, `average-n32-crt7x40` | Depth-one NAND, `n = 32`, seven 40-bit primes, 11-bit security |
| `lwr_prf` | `worst-n32-crt7x40`, `average-n32-crt7x40` | `Q = 32`, `P = 4`, `ν = 16`, `h = 2`, `n = 32`, seven 40-bit primes, 11-bit security |

## Claims and proofs

`GeneratedClaim.CorrectnessClaim` in `generated/Claim.lean` states that every execution whose
external inputs meet the input contract and the requirement decrypts to the ideal (cleartext)
output.

- Worst case: every execution, under every hash model, matches.
- Average case: samplers read a sample tape (`MxxRuntime.SampleTape`), and the tapes with a
  mismatching execution have measure at most `2^-failure_exponent`. Tape samplers record their
  cutoffs, so the worst-case analysis holds on every tape and the failure set is empty.

The `…Certificate.lean` module of a parameter set proves `certificate :
GeneratedClaim.CorrectnessClaim`, and `#guard_msgs` checks that `#print axioms certificate`
lists only `propext`, `Classical.choice`, and `Quot.sound`.

### NAND (`nand_circuit`)

`NandSetup` (moduli, layouts, LUT automorphisms, hash tags), `NandEncrypt` (the state input and
the LUT helpers), `NandPreprocess` (the RHS and LUT switching ciphertexts), `NandOnline0` to
`NandOnline3` (the private multiplication, the fifteen branches of the generated loop, the last
branch, and the ModDown of each round), and `NandCertificate` (the requirement, the ideal, and
decryption). The state exponent follows `roundArgument` of `rounds_nand`, and coefficient one of
the final plaintext `X^(outputExponent b)` is the NAND of the inputs.

### Sparse-LWR PRF (`lwr_prf`)

`LwrSetup`, `LwrEncrypt` (the state `X^0`, every member of the public vector `a`, and the readout
LUT helpers), `LwrPreprocess` (the selector RHS of every block and the readout switching
ciphertexts), `LwrBlock0` to `LwrBlock4` (the public alignment, public multiplication, and private
multiplication with ModDown of every cell, and the sum of the fourteen cells), `LwrLut` (the
coefficient LUT from `Q2` to `Q1`), and `LwrCertificate`. The requirement makes the plaintext of
member `k` of `a` the monomial `X^(2 a_k)` and every block's selectors one-hot, so after the five
blocks the state is `X^(2z)` for the sum `z` of the selected coordinates; the readout LUT gives
`(-1)^⌊z/16⌋ τ(z mod 16)`, and decryption returns its parity, the ideal output.

## Reproduction

Export a claim and run the integration test on a GPU (`lattice-estimator-cli` must be on the
`PATH`):

```sh
MXX_EXPONENT_LUT_NAND_PARAMETER_ID=worst-n32-crt7x40 \
  cargo test -p mxx-exponent-lut --features gpu --test nand_circuit -- --nocapture
MXX_EXPONENT_LUT_LWR_PRF_PARAMETER_ID=worst-n32-crt7x40 \
  cargo test -p mxx-exponent-lut --features gpu --test lwr_prf -- --nocapture
```

The export is deterministic: rerunning an ID writes identical `generated` files.

Check a proof in its parameter directory:

```sh
cd crates/exponent-lut/lean/nand_circuit/worst-n32-crt7x40
lake build
```

`lake build` builds both the exported claim (`NandGenerated` or `LwrGenerated`) and its proof
(`NandProof` or `LwrProof`).
