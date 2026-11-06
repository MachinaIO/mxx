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
| `lean/public_ip/generator/` | The destructuring patterns of generated stages, and the script that writes the decode-stage proof of a `public_ip` parameter set |

`<test>` is `nand_circuit` (`tests/nand_circuit.rs`) or `public_ip` (`tests/public_ip.rs`). Each
package requires the package one directory up, and Mathlib and the ir-core Lean library come from
`crates/ir-core/lean`.

### Program- and parameter-independent lemmas

| File | Contents |
| --- | --- |
| `ExponentAutomorphism.lean` | The odd substitution `X ↦ X^σ` and its norm bound |
| `ExponentScale.lean` | `scaleUp`, the runtime ModDown (`blockModSwitchRuns`) and its remainder |
| `ExponentEncoding.lean` | The encoding relation `Rel` (`c_j = s₀ A_{0j} + A_{1j} - t_j + e_j`): sums, actions, substitution, ModDown; multiplication gains (`SmulGain`), which are `1` for monomials |
| `ExponentGadget.lean` | Gadget reconstruction from decompositions, uniqueness of gadgets and rebases |
| `ExponentStacking.lean` | Row and column concatenation and slicing |
| `ExponentWalsh.lean` | Walsh (Fourier) interpolation of a LUT |
| `ExponentLut.lean` | Private multiplication with and without ModDown (for any payload gain), one LUT branch, and the LUT ModDown |
| `ExponentPublic.lean` | Public multiplication (for any plaintext gain) and the public alignment of a public encoding to a lower modulus |
| `ExponentDecode.lean` | Terminal threshold decryption of binary coefficients, of the parity of coefficients in `[-T, T]`, and of finalized coefficient outputs modulo `t` (`decode_terminal_scaled`) |
| `ExponentSampling.lean` | Tape samplers (`…SampleAt`) as deterministic samplers |
| `ExponentRuntime.lean` | Runtime relations as equations: hashes, family reads, ternary and Gaussian samples, plaintext monomials |
| `ExponentIterate.lean` | Sums and conjunctions computed by generated `iterate` loops, and constant tables read by `select` |
| `ExponentDynamic.lean` | Dynamic RHS products: column-wise digits and the digits of one decomposed entry, digit companions, the companion row and keys, the Dynamic private multiplication and LUT branch, and the rows accumulated by flat and nested loops |
| `ExponentErrorForm.lean` | The encoding relation with a named error (`RelE`) and the operations with the error they produce: ModDown residues as functions of the input (`downResidue`), public alignment, public and private multiplication, LUT branch and finish, and the coefficient finalization |
| `ExponentMaskForm.lean` | Negacyclic product coefficients (`coeff_mul_eq`) and mask-weighted terms `s₀ w` as linear forms in the mask coordinates, sub-Gaussian with the proxy of their coefficient energy when `w` reads no mask coordinate (`maskForm_csubG`) |
| `ExponentFunctional.lean` | Uniqueness of the outputs of the runtime relations (ModDown, reductions, rebases, automorphisms, slices, concatenations, family reads, tape samples) |

### Program-dependent lemmas

- `nand_circuit/NandProgram.lean`: the four LUT tables of the depth-one NAND program and the
  exponent of every round; `rounds_nand` shows that the output exponent is `1 - b₀ b₁`.
- `public_ip/PublicIpProgram.lean`: an accepted plaintext `c X^e` (`c² = 1`, `0 ≤ e < n`,
  `2e + n (1 - c) = 2x`) is the monomial `X^x`, products of monomials, and `X^(2n) = 1`.

## Parameter IDs

`<test>/parameters.csv` records reproducible parameter sets under unique `parameter_id`s. With
`MXX_EXPONENT_LUT_NAND_PARAMETER_ID` or `MXX_EXPONENT_LUT_PUBLIC_IP_PARAMETER_ID` set to an ID, the
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

`nand_circuit` adds `nand_depth` and `rhs_mode` (`static` or `dynamic`, the RHS lifecycle the
round trip runs); `public_ip` adds `input_modulus` (`Q`), `dimension` (`m`), and `seed`, the seed of `a`
and of the hash seed of `t`.

| Test | IDs | Parameters |
| --- | --- | --- |
| `nand_circuit` | `worst-n32-crt7x40`, `average-n32-crt7x40` | Depth-one NAND, `n = 32`, seven 40-bit primes, 11-bit security |
| `nand_circuit` | `dynamic-worst-n32-crt7x40` | The same NAND parameters with the Dynamic RHS lifecycle |
| `public_ip` | `average-n32-crt2x40` | `Q = 32`, `m = 4`, `t = (1, 1, 0, 1)`, `n = 32`, two 40-bit primes (one level), 11-bit security |

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
the LUT helpers), `NandPreprocess` (the RHS and LUT switching ciphertexts), `NandOnlineEval0` to
`NandOnlineEval3` (the private multiplication, the fifteen branches of the generated loop, the last
branch, and the ModDown of each round), and `NandCertificate` (the requirement, the ideal, and
decryption). The state exponent follows `roundArgument` of `rounds_nand`, and coefficient one of
the final plaintext `X^(outputExponent b)` is the NAND of the inputs.

The Dynamic set `dynamic-worst-n32-crt7x40` has no preprocess or mask stage: the payload stage
samples the single secret, and encryption also produces the RHS ciphertexts, the switching
ciphertexts, and the digit companions. `NandEncrypt` proves the encodings and ciphertexts,
`NandDigits` the RHS and LUT digit companions (each lane decomposes only the ciphertext entry of its
digit row), and `NandRound0` to `NandRound3` each round: the RHS companion loop (one key column per
iteration, with a nested loop over the output columns that builds the companion keys and row), the
column loop (`c Dec(C)`), the Dynamic private multiplication, which keeps its modulus, the LUT
companion loop (nested the same way per branch and key column), the loop over the sixteen branches
with its two nested column loops, and the LUT ModDown. `NandRound1` to `NandRound3` repeat the
lemmas of `NandRound0` at their moduli and widths; the last LUT outputs one column.

### Public inner product (`public_ip`)

The Program of `PublicInnerProductProgram` has public inputs only, so its output is a public
encoding, which has no coefficient scaled by `⌊q/2⌋` to decrypt. Its workflow instead checks the
output against a claimed plaintext `c X^e` with `decode_public_output`: the decode stage subtracts the
mask part `s₀ A₀ⱼ + A₁ⱼ` from each row component, adds back `c X^e` times `sⱼ G`, threshold-decodes
every coefficient of every column of both residuals, and outputs their sum. The requirement makes the
plaintext of member `k` the monomial `X^(stride a_k)`, the last member (the encoding of zero) zero,
and the claimed plaintext `X^(stride (<a, t> mod Q))`; the ideal residual count is zero. The claim
therefore states that the output is a public encoding of `X^(stride <a, t>)` with error below
`q / 4`. With public inputs only, online evaluation derives every public key from the hash key, so
the workflow has no preprocessing or payload stage.

`PublicIpSetup`, `PublicIpInputs` (the requirement, which checks the members in a loop, and the
ideal), `PublicIpEncrypt` (every member of the input family), `PublicIpOnlineEval` (the first-level
pair inside its one-member parallel family and the root, which multiplies it by the carried odd
member; both are public multiplications at the native gadget), `PublicIpDecode` (both residuals are
the output error, and every decoded coefficient of a column is zero), and `PublicIpCertificate`.
`generator/gen_decode.py` writes `PublicIpDecode` from the exported decode stage, one decoded
coefficient per line; regenerating `average-n32-crt2x40` reproduces it byte for byte.

## Reproduction

Export a claim and run the integration test on a GPU (`lattice-estimator-cli` must be on the
`PATH`):

```sh
MXX_EXPONENT_LUT_NAND_PARAMETER_ID=worst-n32-crt7x40 \
  cargo test -p mxx-exponent-lut --features gpu --test nand_circuit -- --nocapture
MXX_EXPONENT_LUT_PUBLIC_IP_PARAMETER_ID=average-n32-crt2x40 \
  cargo test -r -p mxx-exponent-lut --features gpu --test public_ip -- --nocapture
```

The export is deterministic: rerunning an ID writes identical `generated` files. The decode-stage
proof of a `public_ip` parameter set is written by its generator:

```sh
cd crates/exponent-lut/lean/public_ip
python3 generator/gen_decode.py average-n32-crt2x40
```

Check a proof in its parameter directory:

```sh
cd crates/exponent-lut/lean/nand_circuit/worst-n32-crt7x40
lake build
```

`lake build` builds both the exported claim (`NandGenerated` or `PublicIpGenerated`) and its proof
(`NandProof` or `PublicIpProof`).
