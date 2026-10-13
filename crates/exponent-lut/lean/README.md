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
| `lean/lwr_prf/generator/` | The script that writes the proof modules and the Lake package of an average-case `lwr_prf` parameter set from its exported claim |
| `lean/public_ip/generator/` | The destructuring patterns of generated stages, and the script that writes the decode-stage proof of a `public_ip` parameter set |

`<test>` is `nand_circuit` (`tests/nand_circuit.rs`), `lwr_prf` (`tests/lwr_prf.rs`), or
`public_ip` (`tests/public_ip.rs`). Each
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
| `ExponentErrorForm.lean` | The encoding relation with a named error (`RelE`) and the operations of `lwr_prf` with the error they produce: ModDown residues as functions of the input (`downResidue`), public alignment, public and private multiplication, LUT branch and finish, and the coefficient finalization |
| `ExponentMaskForm.lean` | Negacyclic product coefficients (`coeff_mul_eq`) and mask-weighted terms `s₀ w` as linear forms in the mask coordinates, sub-Gaussian with the proxy of their coefficient energy when `w` reads no mask coordinate (`maskForm_csubG`) |
| `ExponentFunctional.lean` | Uniqueness of the outputs of the runtime relations (ModDown, reductions, rebases, automorphisms, slices, concatenations, family reads, tape samples) |
| `ExponentPairing.lean` | Real weights on the coefficients of an error (`pairing`) and their pullbacks through right multiplications, polynomial multiplications and substitutions, with the `l1` norm and energy of the pulled-back weight |
| `ExponentDecomp.lean` | The pairings with the cell, alignment and block numerators of `lwr_prf` as sums over their sources |
| `ExponentCellStep.lean` | The pairing with the error leaving a block of cells and with the finalized LUT error, in site form: the next weight, the residue sites with their coefficients, and the remainder with its bound |
| `ExponentH2.lean` | The tail of a noise form under H2 (`h2_tail`): Hoeffding on the clamped mask for the key sites, energies with tape-dependent bounds, the block bounds in terms of the `l1` norm of the weight, the one-tower gadget, and the numeric tails |
| `ExponentRefresh.lean` | The refresh to the top level (`crate::refresh::top`), see below |

### The refresh to the top level (`ExponentRefresh.lean`)

The refresh re-encrypts a level-`L` encoding as the level-0 encoding
`y = σ A' - Δ₀ r x G^st₀ + e'` under a new mask, decoding `y mod q` on one chain per prime `q` of
`Q₀` and recombining the residues. The module proves its average-case correctness for every
parameter set, reusing `ExponentEncoding` (the masked-row form), `RuntimeModulusSwitch` (the
decoder's rounding), `PrimitivesCrtRadix` (centered lifts), `RuntimeSubGaussian` (sub-Gaussian
combinations without independence and their tails) and `ExponentH2` (numeric tails):

- `target_weight_congr`: the target action `Δ_L G^st_L G⁻¹(R_q) ≡ Q_q T_q (mod Q)` of the weight
  `R_q`, for every carry of the decomposition;
- `mask_projection`: the mask child's payload `(M + 1) σ₀` (`q ∣ M`) acts as `σ₀` after `Q_q`;
- `decoded_value`: the refresh combination and the unmasking give `ζ = Q_q y + M + N` with the
  smudge `M` and the explicit noise `N = e_b K_pre - E - e_sm`;
- `modulusSwitch_residue`: the runtime's modulus switch returns `ŷ mod q` when `2 |ν| < Q_q`;
- `crt_recombine`, `crt_idempotent`, `top_refresh_coefficient`: the centered residues of every
  chain, weighted by the CRT idempotents, recombine to the reference value;
- `refresh_noise_tail`, `top_refresh_average`: when the source errors form a coordinatewise
  sub-Gaussian vector and every decoder's noise is a combination of them with coefficients
  independent of their block (the digit actions and the preimages), the output is wrong with
  probability at most `Σ_c Σ_j 2 exp(-(Q_q/2 - S)² / (2 w²))`;
- `refresh_top_test_tail`: at the parameters of `tests/refresh_top.rs`, with every noise proxy
  `w ≤ 2^34`, that probability is below `2^-100`.

The relations of the inputs are hypotheses: the child carriers (mask, CBD and smudge Programs) and
the target satisfy their row relations with sub-Gaussian errors, the gadget decompositions
reconstruct their weights, and the preimages satisfy `B Kpre = K`. They are discharged by the
runtime relations of an exported run and by the noise model's proxies, not by a certificate of
the exported graphs; that certificate, for one exported parameter set, is in `refresh_top/`.

### The exported claim of `tests/refresh_top.rs` (`refresh_top/`)

`refresh_top/average-n8-crt4x30/proof/RefreshTopCertificate.lean` proves the generated
`GeneratedClaim.CorrectnessClaim` of the average-case export of the refresh of the first CRT slot
(`SLOTS = 1` in `parameters.csv`): the bridge (`bridge/`, written by `generator/gen_bridge.py`)
maps every average-case run to a run of the renamed worst-case relation (`worst/`), and
`RefreshTopProof.correct`, ported from `worst-n8-crt4x30/` by `generator/gen_port.py`, shows that
every such run decodes to the ideal output, so the failure set is empty. The per-part proof files
are written by `generator/gen_all.sh <directory> <part>`; `gen_decode.py`, `gen_requirement.py`
and `gen_correct.py` write the decode, requirement and final theorems; the `Top*.lean` lemmas are
hand-written.

### Program-dependent lemmas

- `nand_circuit/NandProgram.lean`: the four LUT tables of the depth-one NAND program and the
  exponent of every round; `rounds_nand` shows that the output exponent is `1 - b₀ b₁`.
- `lwr_prf/LwrProgram.lean`: the sparse-LWR readout `τ(k) = ⌊p' k / w⌋ + 1` and the PRF table
  `2 τ(k) - 1`, which is odd.
- `public_ip/PublicIpProgram.lean`: an accepted plaintext `c X^e` (`c² = 1`, `0 ≤ e < n`,
  `2e + n (1 - c) = 2x`) is the monomial `X^x`, products of monomials, and `X^(2n) = 1`.

## Parameter IDs

`<test>/parameters.csv` records reproducible parameter sets under unique `parameter_id`s. With
`MXX_EXPONENT_LUT_NAND_PARAMETER_ID`, `MXX_EXPONENT_LUT_LWR_PRF_PARAMETER_ID`, or
`MXX_EXPONENT_LUT_PUBLIC_IP_PARAMETER_ID` set to an ID, the
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
round trip runs); `lwr_prf` adds `input_modulus` (`Q`), `output_modulus` (`P`),
`lwr_dimension` (`ν`), `support_weight` (`h`), `outputs`, and `key_seed`, the seed of the PBC layout
and the key; `public_ip` adds `input_modulus` (`Q`), `dimension` (`m`), and `seed`, the seed of `a`
and of the hash seed of `t`.

| Test | IDs | Parameters |
| --- | --- | --- |
| `nand_circuit` | `worst-n32-crt7x40`, `average-n32-crt7x40` | Depth-one NAND, `n = 32`, seven 40-bit primes, 11-bit security |
| `nand_circuit` | `dynamic-worst-n32-crt7x40` | The same NAND parameters with the Dynamic RHS lifecycle |
| `lwr_prf` | `worst-n32-crt7x40`, `average-n32-crt7x40` | `Q = 32`, `P = 4`, `ν = 16`, `h = 2`, `n = 32`, seven 40-bit primes, 11-bit security |
| `public_ip` | `average-n32-crt2x40` | `Q = 32`, `m = 4`, `t = (1, 1, 0, 1)`, `n = 32`, two 40-bit primes (one level), 11-bit security |
| `lwr_prf` | `average-n65536-crt36x40` | The recommended row of `tests/lwr_prf_parameters.csv`: `Q = 256`, `P = 32`, `ν = 1024`, `h = 31`, `n = 2^16`, thirty-six 40-bit primes with 10-bit digits, 124-bit security |

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
lists only `propext`, `Classical.choice`, and `Quot.sound`. The exception is
`lwr_prf/average-n65536-crt36x40`, whose `LwrH2Cert` proves the claim from the heuristic `LwrH2`
(`certificateH2 : LwrH2 → GeneratedClaim.CorrectnessClaim`, with the same axioms).

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

### Sparse-LWR PRF (`lwr_prf`)

`LwrSetup`, `LwrInputs` (the requirement and the ideal output, which check the members of `a` and
sum every block's selectors in loops), `LwrEncrypt` (the state `X^0`, every member of the public
vector `a`, and the readout LUT helpers), `LwrPreprocess` (the selector RHS of every block and the
readout switching ciphertexts), `LwrBlock0` to `LwrBlock<B-1>` (the public alignment, public
multiplication, and private multiplication with ModDown of every cell, and the sum of the cells),
`LwrLut` (the coefficient LUT from `Q2` to `Q1` and the finalization `G⁻¹(k)`), and
`LwrCertificate`. The requirement makes the
plaintext of member `k` of `a` the monomial `X^(stride a_k)` and every block's selectors one-hot, so
after the blocks the state is `X^(stride z)` for the sum `z` of the selected coordinates; the readout
LUT gives `(-1)^⌊z/w⌋ (2 τ(z mod w) - 1)` for `w = Q/2` against the full gadget of `Q1`, the
finalization `G⁻¹(k)` turns it into an encoding with scale `k` against the gadget `1`, and decryption
modulo `t = 2P` returns `m + b (2P - 2m)` for `m = 2 τ - 1` and `b = ⌊z/w⌋ mod 2`, the ideal output.

At `average-n65536-crt36x40` (34 blocks of 113 cells, a readout LUT of 128 branches) the
deterministic argument of the small sets does not fit: with the worst-case cutoffs the terminal
error bound is about `2^39.9`, against the decoding budget `(Q1 - 2 r W) / (2 t) ≈ 2^33`. Its
certificate goes through the residue-replacement heuristic H2 (`ResidueReplacement` of
`RuntimeHeuristics`): `LwrH2Cert` proves `certificateH2 : LwrH2 → GeneratedClaim.CorrectnessClaim`,
where `LwrH2` states that replacing the H2 sites by independent uniforms does not decrease the tail
of the decoded coefficient at the threshold. The sites are the ModDown residues of the rows of the
private multiplications, of the alignments of the members of `a`, and of the LUT finish, and the
key residues of the alignments, which the mask multiplies. The remainder (the encryption, RHS and
switching errors and the other key residues) is bounded on every tape; the row sites have a
deterministic energy; the key sites are linear forms in the mask, below `τ` times their norm except
on the union of their Hoeffding events. The modules of the library `LwrH2`:

- `LwrEncryptE`, `LwrPreprocessE`, `LwrErrorCommon`, `LwrBlockE0` to `LwrBlockE<B-1>`, `LwrLutE`:
  every operation with its error named (`RelE`);
- `LwrModelBase`, `LwrModel0` to `LwrModel<B-1>`, `LwrModelLut`: the keys, rows, numerators and
  errors of the model of a run as functions of the tape, with the samples clamped to their ranges;
- `LwrChainAll`, `LwrChainFinal`: every run is its model, and every ModDown numerator of the model
  is divisible by its dropped factor;
- `LwrH2Model`: the weights pulled back from the decoded coefficient, the sites, and the
  decomposition of the decoded coefficient into the remainder and the sum over the sites;
- `LwrH2Bounds`: the `l1` norms of the weights, the remainder, and the energies of the sites, on
  every tape; the finalization digits enter by their column `l1` norm (`FinalDigits`), which is
  `20` here: the digits of `k` are `(0, 0, -4, 16)`;
- `LwrH2Agree`: the site coefficients read only the payload, preprocessing and mask samples;
- `LwrH2Cert`: the certificate under `LwrH2`.

`generator/gen_h2.py` writes them after `generator/gen.py`. Its cells live in `LwrCells`, and its
proof libraries pass `--tstack` to Lean, since the kernel checks of the chained blocks need a larger
thread stack than the default.

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
MXX_EXPONENT_LUT_LWR_PRF_PARAMETER_ID=worst-n32-crt7x40 \
  cargo test -p mxx-exponent-lut --features gpu --test lwr_prf -- --nocapture
MXX_EXPONENT_LUT_PUBLIC_IP_PARAMETER_ID=average-n32-crt2x40 \
  cargo test -r -p mxx-exponent-lut --features gpu --test public_ip -- --nocapture
```

The export is deterministic: rerunning an ID writes identical `generated` files. The round trip of
`average-n65536-crt36x40` does not fit a local GPU; `MXX_EXPONENT_LUT_LWR_PRF_EXPORT_ONLY=1` exports
its claim from the same stage graphs without executing them:

```sh
MXX_EXPONENT_LUT_LWR_PRF_EXPORT_ONLY=1 MXX_EXPONENT_LUT_LWR_PRF_PARAMETER_ID=average-n65536-crt36x40 \
  cargo test -r -p mxx-exponent-lut --features gpu --test lwr_prf -- --nocapture
```

The proof of an average-case `lwr_prf` parameter set is written by the generator from its row and its
exported claim; it writes every proof module and the Lake package files. The cells, the block sums, and the
readout LUT of `average-n32-crt7x40` are its templates.

```sh
cd crates/exponent-lut/lean/lwr_prf
python3 generator/gen.py average-n65536-crt36x40
python3 generator/gen_h2.py average-n65536-crt36x40
cd ../public_ip
python3 generator/gen_decode.py average-n32-crt2x40
```

Check a proof in its parameter directory:

```sh
cd crates/exponent-lut/lean/nand_circuit/worst-n32-crt7x40
lake build
```

`lake build` builds both the exported claim (`NandGenerated`, `LwrGenerated`, or
`PublicIpGenerated`) and its proof (`NandProof`, `LwrProof`, or `PublicIpProof`).
