# Diamond iO 2 (dio2): specification

Status: implemented in `crates/io/src/dio2` (see §14 for how the implementation realizes this
specification). It fixes what every party computes, which `mxx-exponent-lut` operations compute
it, and how they were extended. Each statement about existing code cites the file that defines
the behavior. Paths are relative to the repository root.

## 1. Overview

dio2 obfuscates a *streaming function*. The function is public. It is evaluated over at most
`stream_length` steps. At step `i`:

- the obfuscator's secret chunk `y_i ∈ {0,1}^obf_chunk_bits` and the evaluator's chunk
  `x_i ∈ {0,1}^eval_chunk_bits` enter;
- the step's Exponent-LUT program maps `(x_i, y_i, z_i)` to the next state
  `z_{i+1} ∈ [0, 2^state_bits)`.

The obfuscator cannot predict `z_i`, and the evaluator never sees `z_i` or `y_i`.

Each step runs on RNS-BGG encodings in Static RHS mode under a fresh mask secret `s_i`. It ends
with a noise refresh (`crates/exponent-lut/src/refresh/mod.rs`), which re-encrypts every object
the next step needs under `s_{i+1}`. The evaluator derives `s_{i+1}` inside the encodings with
the sparse-LWR PRF: the key is chosen by `x_i`, and the input is the first column of a public
label matrix `A_i`, which evolves as `A_{i+1} = A_i[:, 1..] D_{x_i}`. Because the refresh output
noise does not depend on `i` (§9), the noise does not accumulate across steps.

## 2. Notation and encoding conventions

- `R = Z[X]/(X^n + 1)` and `R_M = R / M R`. All rings share `n` and the gadget base `B =
  2^base_bits`.
- **Moduli.** `Q_0 = ∏ P` for a set `P` of NTT primes (canonical order). `G_M` is the full RNS
  gadget row of a CRT sub-basis `M` of `Q_0` (`RnsBggParameters::public_gadget`,
  `crates/exponent-lut/src/rns_bgg/mod.rs`), with `m_M` columns. `G_M^{-1}` is its signed digit
  decomposition, so `G_M G_M^{-1}(w) = w mod M`.
- **Rows.** A *row* at modulus `M` is `c = (c_0, c_1)` with each `c_j ∈ R_M^{1×w}`, paired with a
  key `K = (K_{kj})_{k,j∈{0,1}}`, `K_{kj} ∈ R_M^{1×w}`. Under a mask `s = (s̄, 1)` it encodes a
  target `T = (T_0, T_1) ∈ R^2` in form `F ∈ R_M^{1×w}` with error `e` when

  `c_j = s̄ K_{0j} + K_{1j} − T_j F + e_j    (j = 0, 1)`                                    (Rel)

  This is `Rel` in `crates/exponent-lut/lean/ExponentEncoding.lean`, with `t_j = T_j F`.
- **Forms at level `l` of a chain `Q_0 ⊇ Q_1 ⊇ …`.**
  - The *state form* is `F^st_l = Δ_l G^st_l`, with `Δ_l = Q_l / Q_{l+1}` and `G^st_l = G_{Q_{l+1}}`
    (`message_scale`, `state_gadget`).
  - The *public form* is `F^pub_l = G_{Q_l}`.
- **Encoding kinds** (`crates/exponent-lut/src/rns_bgg/encoding/mod.rs`,
  `crates/exponent-lut/lean/ExponentPublic.lean`):
  - A *private encoding* of `μ` under payload secret `r = (r̄, 1)` has `T = μ r` in state form.
  - A *public encoding* of `α` has `T = α s` (payload = mask) in public form.
  - A *Static LUT helper* for automorphism `σ_k` at level `l` has key `P_{l,k}` and `T = σ_k(s)` in
    state form (`encrypt_lut_helper`, `crates/exponent-lut/src/rns_bgg/encoding/encrypt.rs`).
  - A *gadget carrier* has `T` in public form at the level after its LUT. It is the normalized
    coefficient output used with `RnsOutputUse::Gadget`.

  Only components matter for per-component operations (§2.1). A row whose `T` is not of the form
  `μ r` for the Static payload secret is still a valid row, but no private product or LUT may read
  it (§4.3).
- **Monomials.** `mono(e) = X^{e · 2^stride}` in the exponent layout `(stride_log2, width_log2)`,
  with exponent modulus `Q_exp = 2n / 2^stride` (`ExpLayout`,
  `crates/exponent-lut/src/program/mod.rs`).

### 2.1 Operations used on rows

| Operation | Effect on `(c, K, T, F)` | Reference |
| --- | --- | --- |
| Ring scalar `ρ ∈ R` | `(ρ c, ρ K, ρ T, F)`; error `ρ e` (unchanged norm for `ρ = ±X^a`) | `mul_small_constant` |
| Right action `D` (small) | `(c D, K D, T, F D)` | `pairMul` |
| ModDown `M → M'` | divides an exact multiple `(M/M') F'` form by `M/M'`, adds rounding | `ExponentScale.lean` |
| `MulPub(P, c)` (public `P` of `α`, any row `c`) | per component `j`: `P_0 G^{-1}(K_{0j}) + P_1 G^{-1}(K_{1j}) + α c_j`; target `α T_j` | `ExponentPublic.lean` (`mul_public`) |
| Private product / LUT | needs `T = μ r` with the Static payload `r` | `ExponentLut.lean` |

**Re-targeting `Γ`.** Let `c` have form `F = (M/N) G_N` at modulus `M`, with `N | M`. Let `M' | N`
be a target modulus and `F'` any form over `R_{M'}`. Then

`Γ_{M→M', F'}(c) = ModDown_{M→M'}( c · G_N^{-1}((N/M') F') )`

encodes the same `T` in form `F'` at `M'`. Here `D := G_N^{-1}((N/M') F')` is the signed digit
matrix of the centered lift of `F'`. Both the row and every key block are acted on by `D`
(`K' = ModDown(K D)`), and all three are rounded by the same runtime ModDown.

*Check.* `F · D = (M/N)(N/M') F' = (M/M') F' (mod M)`, so the ModDown is exact on the target. With
`p = M/M'`, rounding remainders `r_c` (row) and `r_0`, `r_1` (key blocks), each bounded by `p/2`,
the new error is `e'_j = (e_j D − r_{c,j} + s̄ r_{0j} + r_{1j}) / p`
(`Rel.modDown`, `crates/exponent-lut/lean/ExponentEncoding.lean`). For a binary or ternary `s̄`
this gives `‖e'‖∞ ≤ (‖e D‖∞ + (n + 2)⌊p/2⌋) / p`, the same bound as `lut_mod_down_bound`
(`crates/exponent-lut/src/rns_bgg/noise/rules.rs`), where
`‖e D‖∞ ≤ n · m_N · ⌊B/2⌋ · ‖e‖∞`. For `p = 1` there is no rounding: `e' = e D`.

`Γ` covers public alignment (`F' = G_{M'}`), private ModDown, and conversion between public,
state, and gadget forms. It requires `M' | N`: a state form at level `l` (`N = Q_{l+1}`) can only
move to a modulus that divides `Q_{l+1}`.

## 3. Parameters and the public function

- Streaming shape: `state_bits`, `obf_chunk_bits`, `eval_chunk_bits`, `stream_length`.
- **Step programs.** For each `i < stream_length`, a Static `ExponentLutProgram` `Π_i` with this
  fixed interface:
  - input `state`: a private scalar of exponent layout `L_z` with `2^state_bits ≤ 2^{width_log2(L_z)}`;
  - input `xbits`: a public family of `eval_chunk_bits` members;
  - an RHS family `yrhs`, with plaintexts `Enc_i(y_i)` given by a public encoder of the function
    (for example bit `b ↦ mono(b)`);
  - output `new_state`: the output of an exponent LUT of layout `L_z` (a signed monomial), plus the
    optional public output wires of §8.

  Distinct steps may share one `Π`. `Π_i` may read only these inputs, so its keys and
  preprocessing are independent of the evaluation path.
- **Initial state.** `z_0` is chosen by the obfuscator and kept secret.
- **Lattice parameters.**
  - Ring dimension `n`, base `B`, and the prime set `P` with `Q_0 = ∏ P`.
  - Encoding error: CBD(1), the distribution of the refresh output (§9).
  - Trapdoor width.
- **PRF parameters.**
  - The sparse-LWR profile (`SparseLwrPrfProfile`): layout, so `Q := Q_exp` (a power of two),
    output modulus `P_out`, LWR dimension `v := ν`, support weight `h`, and the bucket width.
  - `m_v = v log2 Q`.
- **Label transition.** `D_x = HashBits(hk, "dio2/D", x) ∈ {0,1}^{m_v × (m_v + 1)}`, one matrix per
  `x ∈ {0,1}^eval_chunk_bits`, shared by every step.
- **Refresh parameters.**
  - The `TopReference` with `top = next = P`: `Δ_0 = 1` and `G^st_0 = G_{Q_0}` (decision §4.1).
  - The chain depths `depths` of every step chain, with `depths[0] = depths[1] = |P|`.
  - CBD η = 1 for the fresh noise, and the decoder's smudge digits `d`.
  - The refreshed slots: all of `P`.
  - Extension primes: none (decision §4.2).
- **Hash key.** `hk`, from which every public matrix below is expanded (`HashSample`).

## 4. Design decisions forced by the existing primitives

### 4.1 `TopReference.next = TopReference.top` (`Δ_0 = 1`)

The refresh outputs `y_j = σ A'_j − Δ_0 T_j G^st_0 + e'_j` in the *reference* format, as
documented in `crates/exponent-lut/src/refresh/mod.rs`. Chain `q` evaluates the next step
starting from level 0 with primes `Q_0` and level 1 with primes
`order_q[..depths[1]]`, where `order_q = [q, P∖{q}]`. Its level-0 state form is therefore
`(Q_0 / Q^{(q)}_1) G_{Q^{(q)}_1}`.

A level-0 row in reference format can be re-targeted to chain `q`'s format only by `Γ` with
`Q^{(q)}_1 | Q_1` (§2.1). Every `q ∈ P` must survive to the decoding level, so this fails for
some `q` whenever `next ⊊ top`.

With `next = top`:
- every chain has `Q^{(q)}_1 = Q_0`, and level 0→1 is a non-drop logical step (drop 1, accepted by
  `RnsBggParameters::new`);
- `F^st_0 = F^pub_0 = G_{Q_0}`, `w_0 = m_{Q_0}`;
- the refresh output, a fresh Static private input, a level-0 public input, and a level-0 Static
  helper all have the same form `G_{Q_0}`.

`RefreshDeclaration::new` accepts `next = top`: it requires only `next ⊆ top` and
`depths[1] = |next|`.

### 4.2 No extension primes; every child chain is a CRT sub-chain of `Q_0`

The existing refresh evaluates its children on child chains that may reach extension primes:
- the CBD fresh-noise Program and the decoder smudge Program
  (`child_chain`, `crates/exponent-lut/src/rns_bgg/mod.rs`);
- the caller's σ Program.

Those children are encrypted freshly by a party that knows the mask. In dio2, from step 1 on, no
one knows `s_i`, so every child input is derived from the step inputs `S_i` by `Γ` (§6). Every
modulus of every child chain must therefore divide `Q_0`.

`P` must be large enough for the deepest of the following, each followed by the decoding level
`L` and `L + 1 ∋ q` on every chain `q`:
- the step program;
- the label products;
- the σ sampler;
- the CBD children;
- the smudge children.

In the recorded production refresh row (`crates/exponent-lut/lean/refresh_top/parameters.csv`:
8 slot primes and 48 extension primes), this means `|P| = 56`, with the ring dimension raised to
meet the lattice estimate at `Q_0`.

### 4.3 The new-mask encoding is built from component rows

Step 4 of the proposal asks for an encoding "of `s_{i+1} G`". The refresh needs exactly the
two-component row `Σ` with `T = σ = (σ̄, 1)` in public form at the decoding modulus `Q_L`. The
refresh's own combination reads `[g | c_one]` with the second component of a public encoding of
one (`refresh_combination`, `crates/exponent-lut/src/rns_bgg/refresh/mod.rs`).

The Static PRF output is a private encoding with payload `t`, so its target is `v (t̄, 1)`, not
`(σ̄, 1)`. Its *component 1*, however, has target `v · 1`, independent of `t`. Hence:

- `g := component 1 of the σ sampler's gadget carrier`, target `v G_{Q_L}` with
  `v ≡ σ̄ (mod q)` on chain `q`;
- `ω := component 1 of Γ_{Q_0 → Q_L, G_{Q_L}}(ONEPUB)`, target `1 · G_{Q_L}`;
- `Σ := (g, ω)`, key `(K_g | K_ω)` per component, target `(σ̄, 1)` mod `q` in form `G_{Q_L}`.

`Σ` is used only per component: by `MulPub` and by the refresh, which are linear in each component
(§2.1; `MulPub` reads component `j` of the private row and column `j` of its key). It is never the
input of a private product or a LUT, which would require payload `t`.

### 4.4 The automorphism primitive `Aut_k`

A bare substitution `X ↦ X^k` of an encoding also substitutes its secrets: it gives mask
`σ_k(s_i)` and payload `σ_k(t)`, so the result could not be refreshed with `Σ`, which is under
`s_i`. The LUT evaluation already switches both secrets back. One Static LUT branch (`lut_branch`,
`crates/exponent-lut/lean/ExponentLut.lean`; `lut_branch_action`,
`crates/exponent-lut/src/rns_bgg/encoding/lowering.rs`) performs these steps on a private state-form
input `c` of `μ` under `(s, r)` at level `l`:

1. **Routes:** applies `σ_k` to the row and to every key block, giving mask `σ_k(s)` and payload
   `σ_k(r)`.
2. **Switches the payload:** multiplies the stacked routed row by the decomposition `D_j` of a
   switching ciphertext `B_k` at level `l + 1`. `B_k` is preprocessed under `t` and satisfies
   `σ_k(r̄) B_{k,0j} + B_{k,1j} = T^sw_j + e_B`.
3. **ModDowns** the switched key.
4. **Switches the mask:** adds the helper `h_k` (target `σ_k(s)`), acted on by the decomposition
   `E_j` of the switched key.

The output is under the *original* mask `s` and has target `σ_k(μ) T^sw_j` at level `l + 1`.

dio2 adds this branch as a Program primitive:

`Aut_k(c)`: the branch with the switching plaintext `1`, i.e. `T^sw_j = Δ_{l+1} r_j G^st_{l+1}`,
and with no table coefficient and no sum.

- **Output:** the private encoding of `σ_k(μ)` under the same `(s, r)` at level `l + 1`, in state
  form, with the input's representation. Neither the mask nor the payload changes.
- **Plaintext:** any `μ`. The relation of `lut_branch` holds for every `μ`, because the
  substitution is a ring automorphism, so the LUT's monomial input contract is not needed.
- **Noise.** Let `p = Q_l / Q_{l+1}` and let `B_sw` bound the switching-ciphertext error `e_B`.
  The routed error times `D_j`, the helper error times `E_j`, the rounding of the intermediate
  ModDown of the switched key, and the final rounding are divided by the final ModDown. The
  switching term `p σ_k(μ) e_B` is not divided (`ExponentLut.lean`, the branch and finishing
  theorems; `crates/exponent-lut/src/rns_bgg/noise/rules.rs`). Hence

  `‖e_out‖∞ ≤ ⌈(routed + helper + switched-key rounding + final rounding) / p⌉ + ‖μ‖_1 B_sw`.

  For a monomial LUT input, `‖μ‖_1 = 1`. For the σ sampler's plaintext `v = (q + 1) σ̄` (the
  binary sampler represents a 1 as `2h = q + 1`, `prf_sampler.rs`), the gain is
  `‖v‖_1 ≤ (q + 1) n`. `Γ` preserves the plaintext and does not reduce it modulo `q`, so this
  gain is real and must be propagated through the later `Γ` steps and the refresh. It costs about
  `log2 q + log2 n` bits of noise on the `S^{σ,k}` path. Reducing it would need a separately
  specified normalization of `v`, which this spec does not use.
- **Material:** the switching ciphertext `B_k` (Static preprocessing under `t`, by the
  obfuscator), and the level-`l` helper `HLP_l[k]` of the consuming chain (§6.9). Because
  `Aut_k` is used only for `k ∈ 𝒦`, the helpers it needs are in `S_i`.

In dio2, `Aut_k` is applied to the σ sampler's private output, which is under `(s_i, t)`, before
its component 1 is taken (§6.3). This gives `g^{(k)}` with `v^{(k)} ≡ σ_k(σ̄) (mod q)` and
`Σ^{(k)} := (g^{(k)}, ω)`. The constant `1` is fixed by `σ_k`, so `ω` serves every `k`.

### 4.5 Every refresh PRF key depends on `x`

`A'` does not depend on `x` (proposal, step 7). Suppose two inputs `x ≠ x'` at the same prefix
(same `A_i`, same label) refreshed target `o` with the same CBD keys. Then they would give
`y_x − y_{x'} = (σ_x − σ_{x'}) A' − (T_x − T_{x'}) G + 0`, a noise-free linear system in
`σ_x − σ_{x'}`. The same holds for repeated smudges.

Therefore:
- the σ key `k^σ_x`, the CBD keys `k^cbd_{x,o,j}`, and the smudge keys `k^sm_{x,q,o,j}` are all
  indexed by `x`;
- they are shared across steps `i`, because the label `A_i[:, 0]` varies with the path;
- the CBD keys of one target are shared by the chains, which is required so that every chain
  decodes the same `e'_{o,j}`;
- distinct targets use distinct `A'` and CBD keys, so no two refreshes share `(s, A', e')` (the
  refresh's precondition in `crates/exponent-lut/src/refresh/mod.rs`).

### 4.6 Path independence (invariants of the design)

- **Labels separate inputs.** The label `A_i` depends on every past chunk: `A_i = A_0[:, 1..]
  D_{x_0}[:, 1..] ⋯ D_{x_{i−2}}[:, 1..] D_{x_{i−1}}`, so it takes exponentially many values over
  the paths. Two input sequences that differ in some `x_j` with `j < i` reach step `i` with
  different labels, so the PRFs give different outputs and the paths get independent binary
  secrets `s_{i+1}`. Inputs that agree on `x_0 … x_{i−1}` share `A_i` and are separated only by
  the key choice of the current chunk `x_i` (§4.5).
- **Keys scale with the chunk alphabet, not with steps or paths.** Every PRF key family (σ, CBD,
  smudge) is indexed by the current chunk `x ∈ {0,1}^eval_chunk_bits` and by the output it
  produces, never by `i` or by past chunks. Their number is proportional to `2^eval_chunk_bits`
  times the PRF outputs of one step.
- **Preprocessing is path-independent.** Refreshed keys `A'_{i,o}` and helper keys `P'_{i,k}`
  depend only on `(i, o)` or `(i, k)`, and BGG+ evaluation keys do not depend on attribute
  values. So every public key of step `i`, every decoder key `K`, and every preimage `K_pre`
  depend only on `(i, x_i, q, o, j)`, never on `x_0 … x_{i−1}`.

### 4.7 Anchor refreshes and the anchor width

The unmask decoder needs `b = s B + e_b` (`crates/exponent-lut/src/decoder/unmask.rs`). For
`i ≥ 1` this hint comes from the previous step: a refresh of the zero row (`c = 0`, `K = 0`,
`T = 0`) with `A' := ` a column block of `B_{i+1}` yields `σ B + e'`.

One refresh has output width `2 w_0`, and `B_{i+1} ∈ R_{Q_0}^{2 × M_B}`. The anchor therefore uses
`N_B = ⌈M_B / (2 w_0)⌉` zero-target refreshes. The last block is padded with columns from
`HashSample(hk, "dio2/B-pad", i + 1)`, which are discarded.

A single `B_i`, sampled with a trapdoor over `Q_0`, serves every part of step `i`. A part decoding
at `Q_L ⊂ Q_0` uses `B_i mod Q_L` and `b_i mod Q_L`. Its preimage is sampled over `Q_0` for a
CRT lift `K̃` of its key `K` (any lift, for example zero on `P∖Q_L`), so that
`B_i K_pre ≡ K̃ (mod Q_0)`, and hence `≡ K (mod Q_L)`.

## 5. Secrets and objects of step `i`

| Object | Value |
| --- | --- |
| `s_i = (s̄_i, 1)` | mask; `s̄_0` uniform binary, sampled by the obfuscator; `s̄_{i+1}` = σ PRF output (§6.3) |
| `t = (t̄, 1)` | Static payload secret, `t̄` ternary, shared by every step |
| `z_i` | state |
| `A_i ∈ Z_Q^{v × (m_v + 1)}` | label, **public in the clear**: `A_0` uniform, sampled and published by the obfuscator; the evaluator computes `A_{i+1} = A_i[:, 1..] D_{x_i}` |
| `B_i ∈ R_{Q_0}^{2 × M_B}` | anchor, sampled with trapdoor `T_i` by the obfuscator |
| `A'_{i,o} = HashSample(hk, "dio2/key", i, o) ∈ R_{Q_0}^{2 × 2 w_0}` | key of input `o` of step `i`; no `x` |
| `P'_{i,k} = HashSample(hk, "dio2/helper", i, k)` | level-0 helper key of `σ_k`, step `i` |

The step input `S_i` is the following set of level-0 rows (modulus `Q_0`, form `G_{Q_0}`, error
CBD(1)), all under mask `s_i`. This is exactly the proposal's input list:

| Name | Key | Target `T` | Proposal name |
| --- | --- | --- | --- |
| `ONEPRIV` | `A'_{i,priv}` | `(t̄, 1)` | `one_private_encoding` |
| `ONEPUB` | `A'_{i,pub}` | `(s̄_i, 1)` | `one_public_encoding` |
| `STATE` | `A'_{i,state}` | `mono(z_i) (t̄, 1)` | `state_encoding` |
| `LBL[r, c]`, `r < v`, `c ≤ m_v` | `A'_{i,(r,c)}` | `mono(A_i[r, c]) (s̄_i, 1)` | `label_encodings` |
| `HLP[k]`, `k ∈ 𝒦` | `P'_{i,k}` | `(σ_k(s̄_i), 1)` | `helper data` |
| `b_i` | `B_i` | 0 | `anchor_sample` |

`𝒦` is the union of `automorphism_indices` of every LUT input layout of every program evaluated in
a step:
- `Π_i` for all `i`;
- the PRF readout LUT and the sampling LUT;
- the CBD and smudge children.

`ONEPRIV` and `STATE` are Static private inputs at level 0. `ONEPUB` and `LBL` are public inputs.
As every existing public encoding does (`RnsBggEncoding::public_payload`), they carry their clear
plaintexts `1` and `mono(A_i[r, c])`, which the evaluator attaches from its clear `A_i`.
`MulPub` and the public product tree read them.
`HLP[k]` is the level-0 Static helper `s P − σ_k(s) G_{Q_0}`, matching the proposal's
`s A_k − σ_k(s) G + e`. Helpers at level `l ≥ 1` of chain `q` are derived publicly by the
evaluator (§6.9).

## 6. Evaluation of step `i` on input `x = x_i`

For a non-final step (`i < i_f`, §8), steps 6.1–6.8 run on every chain `q ∈ P`. Chain `q` has levels
`Q^{(q)}_l = ∏ order_q[..depths[l]]`, with `depths[0] = depths[1] = |P|`, decoding level `L`, and
`q | Q^{(q)}_{L+1}`. Every intermediate is under `s_i`. The keys of every intermediate are public
functions of `(i, x, q)` alone, because BGG+ keys do not depend on attribute values, so
preprocessing never depends on the path `x_0 … x_{i−1}`.

### 6.1 Evaluator bits (proposal step 2)

`XB[b] := mono(x_b) · ONEPUB` for `b < eval_chunk_bits`: a public exponent encoding of `x_b` with
key `mono(x_b) A'_{i,pub}` and the clear plaintext `mono(x_b)`, which the evaluator attaches. The
exponent form is the default because `MulPublic` accepts only exponent operands (Program
validation, `crates/exponent-lut/src/program/mod.rs`). A coefficient bit `x_b · ONEPUB`, with key
`x_b A'_{i,pub}` (the zero row when `x_b = 0`), is allowed only where `Π_i` reads it outside
`MulPublic`. `P_{i,x}` denotes `Π_i` with these bits fixed. The keys depend on `x`, which is
allowed because preprocessing is per `(i, x)`.

### 6.2 State transition (proposal step 3)

Evaluate `Π_i` (Static) on `state = STATE`, `xbits = XB`, and `yrhs = RHS(y_i)`. The Static RHS
ciphertexts encrypt `Enc_i(y_i)` under `t` at the levels where `Π_i` reads them.

The result is `NS`: a private encoding of `mono(z_{i+1})` under `t` at the level of the last LUT.
Then `NS_L := Γ(NS → Q_L, F^st_L)`.

### 6.3 Next mask (proposal step 4)

Run `PrfSampler::binary_polynomial` (`crates/exponent-lut/src/program_gadgets/prf_sampler.rs`)
with:
- `count = n` members and keys `k^σ_x`;
- `q_min = q`;
- PRF input family `[Γ(LBL[r, 0]) for r < v] ++ [Γ(ONEPUB)]`, where `ONEPUB` is the zero member
  `mono(0)`;
- accumulator start `Γ(ONEPRIV)` = `mono(0)` under `t`;
- offset RHS bound to one.

Let `S^σ` be the sampler's packed polynomial output, a private encoding of `v` under
`(s_i, t)`. Re-target `S^σ` to state form (`Γ`), so that it is a valid input of `Aut_k`. For each
`k ∈ 𝒦`, let `S^{σ,k} := Aut_k(S^σ)` (§4.4), one level lower. The level schedule must keep
`Q_L | Q_{l+2}` for the `Aut_k` input level `l`, so that the state-form output at `l + 1` can
reach the public-gadget form at `Q_L` by `Γ`. Each of `S^σ` and `S^{σ,k}` is then
finished with `RnsOutputUse::Gadget` at modulus `Q_L` (via `Γ`). The evaluation of the PRF
members is shared.

The output is `g` (component 1 of `S^σ`), plus `g^{(k)}` (component 1 of `S^{σ,k}`) for
`k ∈ 𝒦`. Here `b_o ∈ {0, 1}` is `ρ(⟨A_i[:, 0], k^σ_{x,o}⟩)`
mapped to a bit, the same on every chain. On chain `q`, `v = (q + 1) Σ_o b_o X^o ≡ σ̄ (mod q)`,
and `s̄_{i+1} := σ̄ = Σ_o b_o X^o`.

Form `Σ = (g, ω)`, `Σ^{(k)} = (g^{(k)}, ω)`, and the state-form row
`E^σ := Γ(Σ → Q_L, F^st_L) = Σ · G_{Q_L}^{-1}(F^st_L)`, with target `(σ̄, 1)`.

### 6.4 Label transition (proposal step 5)

For every `r < v` and `c ≤ m_v`:

`LBL'[r, c] := ∏_{k : D_x[k, c] = 1} LBL[r, 1 + k]`

This is a balanced public-by-public product tree (`public_ip`,
`crates/exponent-lut/src/program_gadgets/public_ip.rs`), with an explicit selection `D_x[:, c]`
instead of a hash-sampled one, and with `ONEPUB` as the empty product. Its target is
`mono(A_{i+1}[r, c]) s_i`, because `A_{i+1}[r, c] = Σ_k A_i[r, 1 + k] D_x[k, c] mod Q`.

The production builder represents this forest as a DSL family, not one scalar Program output
per label. One `Sha256BitColumns` node computes the column families from the fixed 32-byte
transition seed, packing each column's ascending selected indices in binary fields; structural
bit extraction reads a leaf without allocating a graph node per table entry. The runtime uses
the original SHA-256 expansion and the Lean relation specifies its concrete 64-round algorithm,
not an additional random-oracle or witness assumption. Within
each label member, successive parallel levels multiply adjacent pairs and select the unchanged
left member when the right member is absent. Zero/one-leaf columns alone are padded with
ONEPUB to two leaves, as in the scalar noise model. Thus no contributing leaf gains an action,
and both the multiplication depth and term-by-term noise are unchanged. The family has
logarithmically many graph levels; the graph/export stores only a seed and dimensions rather than
`O(m_v^2 log m_v)` literal bits. Runtime selector storage still has that original size.
Refresh parts, initial/zero rows,
and claim predicates use family ranges rather than scalar label enumerations. Noise-only
Programs retain scalar trees (or their existing maximal-leaf representative).

`tests/dio2_claim.rs` accepts `MXX_IO_DIO2_` overrides for `RING_DIMENSION`, `PRIMES`,
`CRT_BITS`, `BASE_BITS`, `LEVELS`, `SMUDGE_DIGITS`, `PRF_NU`, `PRF_H`, `PRF_Q`, `PRF_P`,
`PRF_BUCKET_WIDTH`, `ERROR_SIGMA`, and `TRAPDOOR_SIGMA`, retaining toy defaults. The test uses
`Dio2::with_inputs_by_shape` to declare keys, plaintexts and tables instead of generating host-side
PRF key material or allocating a runtime wrapper for every clear-label entry. Imported public
payloads likewise use symbolic shape bindings instead of polynomial vectors; unused host-side
clear-label sampling and transitions are omitted. This mode is for graph construction only:
execution requires the missing external bindings. Full PRF member counts still apply. Key
counts are addressable `usize` values; selector and coordinate-table cardinalities use their
128-bit product with the row width, and lower to arbitrary-precision `IntExpr` arithmetic.
Concrete execution still requires addressable, fully bound tables and selectors.
Validated uniform families, parallel-loop cardinalities, artifact manifests and Lean contracts
preserve these 128-bit counts. Canonical JSON retains exact integer numbers above `u64::MAX`;
runtime materialization converts counts with checked `usize` conversion and rejects overflow.

For the chosen `n = 2^16`, 27-prime, base-`2^6`, `d = 306`, `nu = 1024` row, the full smudge
PRF has `2,453,063,738,106,839,040` members and selector row width `130`. Its flattened selector
table needs `318,898,285,953,889,075,200` entries, exceeding 64-bit `usize`. This cardinality
is symbolic, without allocating a table or converting cell addresses to `usize`. Family
bounds, row-wise one-hot contracts and graph-only claim requirements retain the exact count;
representative noise counts must not replace the production multiplicities. The ignored
`symbolic_production_label_forest` unit test also isolates and validates the label forest.

The full smudge Program places all weight-one digits into a single row, in digit-major
column order: PRF member `(column n + coefficient) d + r` is placed at coefficient
`coefficient` of column `r * smudge_columns + column`. Both placement and column sums
use indexed bodies; adding digits changes counts, not the number of Program operations.
Refresh and the output decoder read that row in one sequential digit loop, accumulating
the re-targeted row and both key components. The scalar family of weights `P^r 2^-1 mod Q`
is generated by a single indexed power-of-base constant and multiplication by `2^-1`;
there is no digit-sized literal table. Both the loop body and operation count are independent
of `d`.
Re-targeting to `k` repeated targets uses one `BlockDiagonal` operation (`I_k ⊗ target`),
and the centering row is one all-coefficients constant of the full width. Neither creates
concat intermediates proportional to the gadget columns. Prime bodies remain unrolled.
Noise-representative Programs still store one digit and apply every weight to its bound;
they do not replace the full production member counts or the full noise event budget.
The noise model restores the packed sum node's logical per-digit reduction multiplicity,
preserving the pre-packing scalar-digit event budget and its tail multiplier.
The label selector is a fixed-seed SHA-256 recipe, not a quadratic literal. Label-height and
family-member multiplicities stay in loop counts. Binary index decoding and the balanced
public-product schedule retain logarithmic dependence on the label width; fixed PBC bucket
schedules and the prime bodies remain explicit. The Lean exporter shares large types and
environment binders and partitions remaining fixed circuit declarations into modules of at
most 3,000,000 bytes, with unchanged public namespaces and import entry points.

### 6.5 Payload switch of the labels (proposal step 6)

`N[r, c] := MulPub(Γ(LBL'[r, c] → Q_L), E^σ)` has target `mono(A_{i+1}[r, c]) (σ̄, 1)` in state
form at `Q_L`.

The proposal names `new_state_encoding` as the second operand. The operand that replaces the
payload by `s_{i+1}` is the step-4 encoding `E^σ`, so this spec uses `E^σ`.

`E^σ` is not a Program wire: it is not a Static private exponent wire, and Program validation
rejects coefficient operands of `MulPublic`. `MulPub` here is therefore the *raw-row*
multiplication of extension 10.9. It acts on a public row (with its clear `α`) and a general row
(modulus, form, key blocks, error bound), and its relation is the general component theorem
`mul_public` (`crates/exponent-lut/lean/ExponentPublic.lean`).

### 6.6 Refresh targets (proposal step 7)

All targets are rows at `Q_L` in form `F^st_L`:

| Target `o` | Row | `T` (mod `q`) | `A'` | Becomes in `S_{i+1}` |
| --- | --- | --- | --- | --- |
| `priv` | `Γ(ONEPRIV)` | `(t̄, 1)` | `A'_{i+1,priv}` | `ONEPRIV` |
| `pub` | `E^σ` | `(σ̄, 1)` | `A'_{i+1,pub}` | `ONEPUB` |
| `state` | `NS_L` | `mono(z_{i+1}) (t̄, 1)` | `A'_{i+1,state}` | `STATE` |
| `(r, c)` | `N[r, c]` | `mono(A_{i+1}[r, c]) (σ̄, 1)` | `A'_{i+1,(r,c)}` | `LBL[r, c]` |
| `(hlp, k)` | `Γ(Σ^{(k)} → F^st_L)` | `(σ_k(σ̄), 1)` | `P'_{i+1,k}` | `HLP[k]` |
| `(anc, β)`, `β < N_B` | zero row | `0` | block `β` of `B_{i+1}` | part of `b_{i+1}` |

### 6.7 Refresh of one part `(q, o, j)`

This is the existing per-part combination (`refresh_combination`,
`crates/exponent-lut/src/rns_bgg/refresh/mod.rs`), with the inputs replaced as follows:

- `u = [g | ω] (I_2 ⊗ G_{Q_L})^{-1}(Q_q A'_{o,(j)})`, where `Q_q = Q_L / q`.
- `v = c_{o,j} · G^st_L{}^{-1}(R_q)`, with `R_q ≡ Δ_L^{-1} Q_q (G_{Q_0} mod q) (mod q)` and
  `R_q ≡ 0` modulo the other primes of `Q_{L+1}`.
- `w = [f_{o,j,k} G_{Q_L}^{-1}(Q_q)]_k`. Here `f_{o,j}` is the CBD fresh-noise carrier with:
  - keys `k^cbd_{x,o,j}`;
  - PRF input `Γ(LBL[:, 0])` and accumulator `Γ(ONEPRIV)`;
  - target `e'_{o,j} G_{Q_L}`.
- `d = u − v + w = s_i K − Q_q y_{o,j} + E`.
- Decode with the smudged unmask decoder:
  - hint `b_i mod Q_L` and preimage `K_pre`, with `B_i K_pre ≡ K̃ + K^sm (mod Q_0)`;
  - smudge child with keys `k^sm_{x,q,o,j}`, the same PRF input, and the same accumulator.

  This gives `y^{(q)}_{o,j} = y_{o,j} mod q`.

`y_{o,j} = σ A'_{o,(j)} − T_j G_{Q_0} + e'_{o,j}`, which is a level-0 row of `S_{i+1}` (§4.1).

### 6.8 Recombination

`y_{o,j} := CRT_{q ∈ P}(y^{(q)}_{o,j}) mod Q_0`. Then `S_{i+1}` is:
- `ONEPRIV`, `ONEPUB`, `STATE`, `LBL`, and `HLP` with rows `(y_{o,0}, y_{o,1})` and keys
  `A'_{i+1,o}` / `P'_{i+1,k}`;
- `b_{i+1} = [y_{(anc,β),0} | y_{(anc,β),1}]_β`, truncated to `M_B` columns.

### 6.9 Derived helpers

The Static evaluator reads a helper at every non-terminal level where a LUT of layout `σ_k` runs
(`lut_helper_families`), and the helper's target depends on that chain's scale and state gadget
(`encrypt_lut_helper`). Helpers are needed on every *consuming chain* `C`:
- the main chain of every `q`;
- the σ-sampler chain, the CBD-child chain, and the smudge-child chain of every part.

All of these are CRT sub-chains of `Q_0` (§4.2). For each consuming chain `C` and every local
level `l` of `C`, **including local level 0**:

`HLP^{C}_l[k] := Γ_{Q_0 → Q^C_l, F^{st,C}_l}(HLP[k])`,

where `F^{st,C}_l = (Q^C_l / Q^C_{l+1}) G_{Q^C_{l+1}}`. This is valid because `Q^C_l | Q_0` and the
source form is `G_{Q_0}`. For the main chain at level 0 it is the identity.

This is a public computation on the clear rows. Its key `Γ(P'_{i,k})` replaces
`lut_public_matrix(l, σ_k)` in that chain's preprocessing, and its error is the `Γ` bound of §2.1
applied to `‖e'‖∞ ≤ 1`.

## 7. Obfuscator

**Setup.**
- Sample `hk`, `t̄`, `s̄_0`, `A_0`, and `(B_i, T_i)` for `i ≤ stream_length − 1` (§8).
- Sample the PRF key layouts `k^σ_x`, `k^cbd_{x,o,j}`, and `k^sm_{x,q,o,j}` for every
  `x ∈ {0,1}^eval_chunk_bits`.

**Step-0 encryption.** Encrypt `S_0` directly in the format of §5: `y = s_0 A'_{0,o} − T G_{Q_0}
+ e` and `b_0 = s_0 B_0 + e`, with `e ← CBD(1)` drawn by a CBD sampling contract (extension
10.10).

This contract applies only to the rows of `S_0` and to `b_0`. The Static preprocessing
ciphertexts keep their Gaussian contract (`encoding_contract`): the RHS ciphertexts and the LUT
switching ciphertexts under `t`.

**Per `(i, x, q)` preprocessing** of the chain-`q` evaluation of §6. This is the proposal's
`noise_refresh_preprocess` together with the program preprocessing:

- **Public keys.** Every public key of the Static evaluation is derived from the input keys
  `A'_{i,o}` and the helper keys `Γ(P'_{i,k})`. These replace the hash-sampled input and helper
  keys.
- **Static RHS ciphertexts under `t`** on chain `q`, for:
  - `Enc_i(y_i)` (the proposal's `RHS(y_i)`);
  - the selectors of `k^σ_x` (the proposal's `RHS(new_secret_prf_key)`; per `x` and `q`, not
    per `i`, when the chain levels coincide);
  - the selectors of `k^cbd_{x,o,j}` and `k^sm_{x,q,o,j}`;
  - the sampler offsets.
- The switching ciphertexts `B_k` of the `Aut_k` steps (§4.4), under `t`, for `k ∈ 𝒦`.
- The LUT switching ciphertexts of every LUT.
- **Per part `(o, j)`:** the decoder key `K = K_u − K_v + K_f Z + K^sm` and its preimage `K_pre`
  under `(B_i, T_i)` (§4.7).

The obfuscation is `hk`, `A_0`, `S_0`, `{B_i}`, and the per-`(i, x, q)` preprocessing. The RHS
ciphertexts that do not depend on `i` are stored once.

**Evaluator.** For `0 ≤ i < i_f`: run §6 with `x_i` to get `S_{i+1}`. At `i_f`, run §6.1–§6.2
with `x_{i_f}` and decode the outputs (§8).

## 8. Outputs

**Decision (user).** The plaintext output is released only at the last step
`i_f = stream_length − 1`. Only `Π_{i_f}` declares output wires `out_u`.

The last step evaluates `Π_{i_f}`, decodes its outputs, and performs no refresh, because
`S_{stream_length}` is never used. So the obfuscator samples `B_i` only for
`i ≤ stream_length − 1`, and §6.3–§6.8 run only for `i < i_f`.

Every earlier step still decodes internally: the refresh of §6.7 is a decode, by the smudged
unmask decoder, of `y_{o,j} mod q` for every part. Those decoded values are the rows of
`S_{i+1}`, which are encodings and not plaintext outputs of the function.

Each output wire `out_u` of `Π_{i_f}` is a *private exponent* LUT
output (a signed monomial `mono(w_u)` under `t`) at a level `L_u` with a strict modulus drop
after it. This is the source the existing smudged decoder accepts: a private, scalar,
non-coefficient source, followed by a strict drop (`crates/exponent-lut/src/decoder/decoding.rs`,
`decoder_source_level` in `crates/exponent-lut/src/rns_bgg/mod.rs`).

The output is decoded with that decoder on one fixed chain `q_out`, using:
- the unmask hint `b_{i_f} mod Q_{L_u}`;
- a preimage under `B_{i_f}` (CRT-lifted as in §4.7);
- a smudge child whose inputs are derived by `Γ` from `S_{i_f}` and whose keys `k^out_{x,u}` are
  indexed by the current chunk `x` and the output `u` only (§4.6).

The decoded monomial `w_u` is mapped to the released value by a public table of the function.
The preimage and the smudge preprocessing are per `(i_f, x)` and independent of past inputs.
The construction is thus an iO of the whole input `x_0 … x_{i_f}`.

**Output admission.** Refresh admission does not cover outputs, because an output decoder has
spacing `Δ_u = Q_{L_u} / Q_{L_u+1}`, not `Q_L / q`
(`crates/exponent-lut/src/rns_bgg/mod.rs`). Every output decoder `(i_f, x, u)` must satisfy

`2(P_s^{d_u}/2 + F_u) < Δ_u`.

Here `F_u` bounds the following, all derived from input bound 1 as in §9:
- the output-selection action applied to the source error;
- the children's and helpers' errors;
- `e_b K_pre` for the shared `Q_0` anchor's actual preimage bound.

The strict drop after `L_u` must therefore be wide enough for the smudge range. A single 40-bit
prime cannot hold a `2^100` smudge. Each decoder's failure probability enters the global
correctness budget, and its smudged values enter the aggregate hiding budget of §9.

Releasing coefficient outputs directly would instead need a new coefficient-output decoder
(normalization, selection action, spacing, and noise contract), which is not specified here.

## 9. Correctness and noise

**Steady state (unconditional part).** Every row of `S_{i+1}` has error exactly `e'_{o,j}`, the
fresh-noise PRF output, provided every part decodes correctly; the noise inside step `i` does
not enter. The sampler's support gives `‖e'‖∞ ≤ 1` for any label, and `S_0` meets the same bound
by construction. Correctness therefore reduces to one step whose input rows have `‖e‖∞ ≤ 1`.

The *distribution* CBD(1) of `e'`, and its independence across targets, steps, and paths, hold
only under the PRF and label assumptions of §12. The sampler's exact distribution requires a
uniform PRF readout (`crates/exponent-lut/src/program_gadgets/prf_sampler.rs`). Worst-case
admission uses only the support bound; average-case admission additionally uses these
assumptions.

**Admission of one step.** Every part `(q, o, j)`, for every non-final step program `Π_i` (`i < i_f`), every `x`, and
every target `o`, must satisfy the existing acceptance conditions (`fits`:
`2(P_s^d/2 + F) < Q_q`; `hides`: `P_s^d > F 2^bits`;
`crates/exponent-lut/src/rns_bgg/noise/refresh.rs`). Under the average-case model, the union of
their failure probabilities over all parts, steps, and inputs of one evaluation must fit an
overall failure budget.

`F` must bound all of the following error sources, each with a contract that dio2 supplies
instead of the simulator's fresh-sample assumption (extension 10.8):
- `e_g` (σ-sampler carrier) and `e_ω` (`Γ` of `ONEPUB`), through `(I_2 ⊗ G)^{-1}`. The simulator
  currently treats the one-encoding as fresh.
- The target error through `Y`. This is the simulated noise of `Π_i`, of the label tree followed
  by the raw-row `MulPub`, or of the `Γ` steps, all starting from input bound 1.
- The fresh carrier `f` through `Z`, with child inputs derived by `Γ` from rows of bound 1
  instead of fresh encryptions.
- The smudge carrier's encoding error.
- `e_b K_pre`, with `‖e_b‖∞ ≤ 1` and the preimage width and bound of the trapdoor of `B_i` over
  `Q_0` with `M_B` columns, not of a per-part trapdoor at the decoding modulus.
- The `Aut_k` branches of §6.3: the undivided switching term with payload gain
  `‖v‖_1 ≤ (q + 1) n` (§4.4), plus derived helper errors and the ModDown roundings.
- Inside every child and the step program: LUT helper errors from §6.9 (the `Γ` bound of input 1)
  instead of fresh helper errors.

Rows derived from one source (for example `ω`, every child accumulator, and every helper) share
provenance. The average-case model must either track it or use the worst-case bound for those
terms.

**Aggregate hiding budget.** The `hides` condition is per smudged value, and the decoder's
statistical hiding must be accounted over every observed value
(`crates/exponent-lut/src/decoder/decoding.rs`). The parameters must satisfy

`Σ_a F_a / P_s^{d_a} ≤ 2^{−λ}`

over all smudged coefficients `a` an adversary observes: columns, polynomial coefficients, both
components, all parts `(q, o, j)`, all steps, all evaluated paths, and all output decoders of
§8. With a uniform budget, every value needs at least `λ + ⌈log2 N_obs⌉` hiding bits. This
budget is separate from the correctness-failure budget.

**Plaintext condition.** The decode reads only `T_j mod q` on chain `q`, so the recombined target
is the integer polynomial `T_j` exactly when every chain carries the same `T_j` modulo its `q`.
This holds for every target in §6.6:
- the `(q + 1) σ̄` representation of the binary sampler is `≡ σ̄ (mod q)`;
- every other target is a signed monomial times a ternary or binary polynomial computed
  identically on each chain.

Admission therefore generalizes the current rule, which requires an exponent LUT output
(`target_admission`), to *row targets with chain-consistent `T_j`*.

**Algebraic check.** A scalar (`n = 1`) model with exact arithmetic over four primes
(`depths = [4, 4, 3, 2]`) confirmed the following. With `Δ_0 = 1` and §4.3 / §6.5 / §6.7, the
refreshed label rows recombine exactly to `y_j = σ A'_j − α σ_j G_{Q_0}`, that is
`y_j = σ̄ A'_{0j} + A'_{1j} − α (σ̄, 1)_j G_{Q_0}` with `e' = 0`, a level-0 public encoding of `α`
under the new mask. That check had zero noise and a scalar mask. The zero-noise runs (§13, item 1)
extend it to polynomial masks, componentwise. The correctness and admission with nonzero noise
are the Lean obligations of §13, item 2.

## 10. Required extensions of `mxx-exponent-lut`

These extend existing structures; no compatibility paths are kept.

1. **Refresh targets.** `RefreshDeclaration` and the refresh builders take a row target `(K, c)`
   in state form at level `L` with a chain-consistency contract, including the zero row, in
   place of a Program output wire with `target_admission`. Each component `j` may come from a
   different source row.
2. **External anchor.** The unmask decoder of each part takes a caller anchor `(B mod Q_L, b mod
   Q_L)` and a caller-supplied preimage, instead of sampling `B` per part and encrypting `b`.
3. **External one-encoding and σ row.** `c_one := ω` and `g` are caller rows, generalizing the
   existing `RefreshSigma` callback to rows produced by an online evaluation that has no
   encryption stage.
4. **Child inputs from rows.** The CBD and smudge children, and the σ sampler, read their PRF
   inputs and accumulator as given encodings (`Γ` of `S_i`). There is no child encryption
   stage, and extension primes are rejected.
5. **Static preprocessing with given keys.** Input keys and level-`l` helper keys are supplied
   (from `A'_{i,o}` and `Γ(P'_{i,k})`) instead of being hash-sampled per wire and level. Online
   evaluation takes the input rows and helper rows directly.
6. **Automorphism primitive.** `ExponentLutProgramBuilder::automorphism(input, k)` adds a
   `Step` whose output is `Aut_k(input)` (§4.4). It is lowered as one Static LUT branch with
   switching plaintext 1 and no table or sum:
   - **preprocessing:** the switching ciphertext `B_k` under `t` and the branch's public key;
   - **online evaluation:** `lut_branch_action`, the ModDown of the switched key, the helper
     action, then the final ModDown of both the row and the output key to level `l + 1`
     (`lut_branch_action` and the helper action alone leave the branch at the source modulus);
   - **noise:** the bound of §4.4, with the input's payload gain on the undivided switching
     term.

   The output keeps the input's knowledge and representation, at the next level. Program
   validation does not apply the LUT monomial input contract to it. The helper keys of
   `canonical_lut_helper_keys` include `k` at every non-terminal level of the Program.
7. **Explicit-selection public products.** `PublicInnerProductProgram` accepts a given binary
   selection (a column of `D_x`).
8. **Noise simulation.** The refresh and Static simulators take contracts for every source of §9:
   - row targets with given error bounds;
   - the `Σ` components;
   - child inputs and helpers derived by `Γ`;
   - the shared anchor's width and preimage bound over `Q_0`;
   - the smudge carrier error.

   They also take a provenance rule for derived rows, and an admission run over every step
   program, `x`, and target, with a global failure budget and the aggregate hiding budget of §9.
   The same contracts extend the ordinary output-decoder simulation of §8: derived child inputs
   and helpers, and the shared anchor over `Q_0`, in place of fresh child inputs and a
   local-modulus anchor (`crates/exponent-lut/src/rns_bgg/noise/decoder.rs`). Every output
   decoder is checked against `2(P_s^{d_u}/2 + F_u) < Δ_u`.
9. **Raw-row public multiplication.** `MulPub(P, c)` for a public row `P` (with clear `α`) and a
   general row `c` (modulus, form, key blocks, error bound), outside Program validation. For
   each component `j`, let `D_j = [G^{-1}(K^c_{0j}); G^{-1}(K^c_{1j})]` (stacked). Then the
   output row is `c^out_j = [P_0 | P_1] D_j + α c_j` and the output key is
   `K^out_{kj} = [K^P_{k0} | K^P_{k1}] D_j` for `k = 0, 1` (`pairMul (Aa k) (D j)` in the
   general `mul_public` theorem, `crates/exponent-lut/lean/ExponentPublic.lean`).
10. **CBD encryption contract.** A sampling contract `CBD(η)` for the rows of `S_0` and `b_0`,
    separate from the Gaussian contract of preprocessing ciphertexts.

## 11. Cost (per step and per evaluator input `x`)

- **Refresh targets:** `N_tgt = 3 + v (m_v + 1) + |𝒦| + N_B`.
- **Parts:** `2 N_tgt |P|`. Each part has one smudge child of `w_0 n d` PRF members and one
  decoder preimage.
- **CBD members:** `2 N_tgt w_0 n` per chain.
- **Label products:** `≈ v (m_v + 1) m_v / 2` public products per chain.
- **Obfuscation size:** about `stream_length · 2^eval_chunk_bits · |P|` times the per-chain
  preprocessing.

**Example.** At the recorded PRF row (`ν = 1024`, `Q = 256`), `m_v = 8192`. Then
`v (m_v + 1) ≈ 8.4 · 10^6` label targets and about `3.4 · 10^10` public products per chain and
step. Tests need small `v`.

## 12. Security assumptions outside this specification

- The sparse-LWR PRF is a weak PRF on labels `A_i[:, 0]`.
- **Label freshness.** Keys are indexed by `x` only (§4.6), so the same pair
  `(x_i, A_i[:, 0])` reached twice repeats `σ`, every `e'`, and every smudge. Within one path, a
  repeat at steps `i ≠ i'` would make the refreshed `ONEPRIV` rows (same payload target, keys
  `A'_{i+1,priv} ≠ A'_{i'+1,priv}`, same `σ`, same `e'`) satisfy a noise-free relation.

  The design therefore assumes that the labels `A_i[:, 0]` of all (step, path) pairs that an
  adversary can evaluate are pairwise distinct, except with probability below the collision
  budget `N_eval^2 / Q^v`. This holds when the labels are pseudorandom in `Z_Q^v`. Same-label
  repeats with equal `(i, x_{<i})` are the same computation and are harmless.
- `A_{i+1}` is pseudorandom. With `m_v = v log2 Q` exactly, the leftover-hash argument for
  `A_i[:, 1..] D_x` has no slack.
- One trapdoor anchor `B_i` is shared by all parts of a step (GPV preimages for many keys).
- The smudge heuristic of the existing decoder.
- Circular use of `t` across all steps.
- Hash-sampled public matrices model uniform ones.

## 13. Validation plan (for the later implementation)

**Decision (user).**

1. **First: correctness without noise.** Run at the smallest parameters (small `n`, small `v`,
   few primes, short `stream_length`) with every error contract zero: encoding errors, the CBD
   fresh noise, and the smudges. There the implementation must show that:
   - `S_{i+1}` satisfies (Rel) for every row with the expected `T` and zero error;
   - the refreshed rows recombine exactly;
   - the final decoded output equals the clear streaming function for every input of a small
     domain.

   The scalar model of §9 serves as a regression of the formats.
2. **Then: Lean proofs instead of large-`v` runs.** The noise admission of §9 (refresh parts,
   `Aut_k`, and output decoders) and the correctness of the step are established by Lean proofs
   on exported claims, as for `crates/exponent-lut/lean/refresh_top`. There are no hardware runs
   at production `v`.

## 14. Implementation

The extensions of §10 are implemented as follows:

| §10 | Implementation |
| --- | --- |
| 1, 2, 3, 4 | The refresh of row targets is composed in `crates/io/src/dio2/refresh.rs` from the exposed refresh pieces (`RnsBggParameters::refresh_actions`, `refresh_combination`, `residue_term`), the unmask of `crates/exponent-lut/src/decoder/unmask.rs`, and the anchor's preimages under one trapdoor per step. |
| 4, 5 | Rooted chains (`RnsBggRoot`, `RnsBggParameters::with_root`, `crates/exponent-lut/src/rns_bgg/mod.rs`): input keys and LUT helper keys are `Gamma` of root keys hashed at `Q_0`, and an imported encryption (`RnsEncryptBuild::import`) publishes `Gamma` of given root rows in place of encodings. |
| 6 | The automorphism table `LutTable::automorphism` (`crates/exponent-lut/src/program/mod.rs`): a one-branch Static LUT with coefficient `1`, which also reads a raw coefficient wire. |
| 7 | The label products are balanced DSL families described in `crates/io/src/dio2/step.rs` and lowered with `mul_public_rows` in `crates/io/src/dio2/refresh.rs`; noise-only Programs retain scalar trees. |
| 9 | `RnsBggParameters::retarget_encoding` and `RnsBggParameters::mul_public_rows` (`crates/exponent-lut/src/rns_bgg/rows.rs`). |

Choices the implementation makes within the specification:

- **One Program per step and chunk (§15).** The fresh-noise and smudge children of every part,
  and the σ sampler with its automorphisms, are PRFs inside the step Program (no child chains).
  Their inputs read the root objects of `S_i` under the same root tags as the step's inputs. Each
  refresh part has its own CBD sampler, and each part and prime its own smudge PRF (their own
  keys, §4.5). The refresh is composed into the step's online evaluation (`api::online_eval`'s
  compose hook), so rows and keys of every part and prime come from one evaluation.
- **Decoded keys.** The obfuscator obtains every part's decoded key by running the same composed
  evaluation on zero rows (keys do not depend on rows), then samples the preimages under the
  step's anchor. Hence keys and rows can never diverge.
- **Smudge weights.** The step Program packs the weight-one rows for all smudge digits `r`
  into one output; a sequential digit loop reads each corresponding slice.
  Both refresh and output decoders re-target each digit row and its key to the BigInt scalar
  form `P^r 2^-1 mod Q` at the decoding level, then sum the digits. The noise simulation uses
  the same re-targetings. Thus the smudge range `P^d` is not limited by small-constant payload
  bounds. Packing changes graph structure, not the per-digit arithmetic.
- **Output.** The last step decodes the state monomial modulo the first prime (§15);
  `Dio2::evaluate` returns the final state.
- **Noise.** `crates/io/src/dio2/noise.rs` simulates every decoder of an evaluation (worst case
  and the average-case model of §9 with one tail multiplier for all decoded coefficients) and
  checks `fits` and `hides`. The production parameter search explicitly selects noise-only
  representative targets: every non-label target, and a maximal-leaf label product per chunk.
  The public-product rule `E_out = A E_left + ||mu_left||_1 E_right`, with `A >= 1` and monomial
  `||mu_left||_1 = 1`, makes the balanced label product's noise monotone in its (at least two)
  leaves. Thus its maximum decoder error is unchanged. This does not replace any real label
  output: obfuscation, evaluation, and exported claim graphs still use the full plan.
  `StepPlan::full_target_count`, full sampler member counts, and `decoded_coefficients` retain
  the complete matrix's counts. The internal evaluation audit also restores the exact omitted
  scalar-product event count (`2 n w_0` per product), so neither average-case tail multiplier
  benefits from the reduction. In representative mode the identically bounded weight-one
  smudge digits also share one modeled row, re-targeted with every actual digit weight and
  summed with unknown covariance as before. Builder-owned wire intervals identify the PRF
  member operations and placed-row events that scale with the exact digit count; shared
  bindings and the audit's fixed-width inner PRF reductions remain unchanged. This
  avoids `members * bucket_width` overflow in large PRF shapes without shrinking any bound.
- **Execution.** `Dio2<E, S>` runs on any stage executor and store. The end-to-end test
  (`crates/io/tests/gpu_round_trip.rs`) runs on the GPU runtime.


## 15. Revision: one chain without modulus drops, uniform-sign σ

This revision supersedes the per-prime chains of §3, §4.1, §4.2, §6 and §7 and the binary σ of
§4.3 and §6.3. Every other section holds with the changes below.

**One chain.** Every level of the step chain has the modulus `Q_0`: the evaluation never drops a
prime (each logical level is a drop of one, accepted by `RnsBggParameters::new`), and `Δ_l = 1`
on every level but the last. The parameter `depths` becomes the level count `levels`, with the
decoding level `L = levels − 2`. All the chains `q` of §6 are then the same chain, so one step
Program per `(i, x)` serves every prime: its PRFs (σ, CBD, smudge) run once on one set of keys and
labels. A LUT no longer scales its noise down by a drop, so the noise grows by the LUT's gain at
every level; `Q_0` must hold the total, which costs about as many bits as the dropped primes did.

**Per-prime decoders.** A decoder still rounds modulo one prime (spacing `Q_0 / q`). Part
`(t, j)` therefore has one decoder per prime `q ∈ P` on the same Program outputs, each with its
own refresh actions (§6.7) and its own smudge (keys `k^sm_{x,q,o,j}` as in §4.5); the CBD noise
`f` of the part is shared by its primes, so that the residues are those of one row. The evaluator
sums the CRT terms of the residues into the part's row of `S_{i+1}` (§6.8) inside the step's
evaluation.

**Uniform-sign σ.** The σ sampler draws `σ̄ ∈ {−1, 1}^n` uniformly (`PrfDistribution::Sign`,
`PrfSampler::sign_polynomial`): the sampling LUT returns `±1` and adds no offset, so `σ̄` is the
same small polynomial modulo every prime, and its payload bound is `n` instead of `(q + 1) n`
(§4.4). The mask `s_{i+1} = (σ̄, 1)` is uniform over `{−1, 1}^n`; the initial mask `s̄_0` stays
binary. The plaintext condition of §9 holds with `σ̄` itself in place of `(q + 1) σ̄`.

**Output decoder.** The last step re-targets the state to the scalar form `Q_0 / q` of the first
prime `q` and rounds the unmasked row modulo `q` to the state monomial (§8 with spacing
`Q_0 / q`); the plaintext needs no level below `L`.

**Toy parameters.** With `n = 8`, five 50-bit primes, gadget base `2^12`, six levels, the PRF of
§13 and one smudge digit, the simulated decoder error is at most `2^159` (worst case) and
`2^158` (average case, failure probability `2^-40`) against the spacing `2^200`.
