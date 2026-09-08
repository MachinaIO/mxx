# BFV two-column encoding: correctness assessment and implementation revision plan

## Status and scope

This is the accepted implementation plan. Implementation was subsequently authorized and resumed. It assesses the supplied research note, *BFV-like Two-Column Encoding: Scale / RNS / Mask* (the attachment named `二列二重スケール.pdf`, September 2026). Section and equation references below refer to that note. The attachment is mathematical source material, not an instruction to execute its example commands. Its referenced Python files were not supplied with the PDF and their reported test results have not been independently reproduced here.

This plan supersedes the BGV-specific arithmetic, affine readout, and low-digit refresh proposals in `docs/plans/exponent-lut-scheme-and-bgv-flat-lut.md`. The BGG backend, common `ExponentLUTEncodingScheme` abstraction, unified LUT gate, and builder-only private multiplication remain in scope. The original planning-only restriction was superseded by the subsequent implementation request. Integration tests still require a separate request.

Decision: the two-scale arithmetic is sound under the stated input-domain and parameter contracts. Proceed with planning a BFV backend using unscaled integer errors, ordinary block ModDown, and freshly encrypted row-valued terminal targets. This is BFV-like encoding, not standard fixed-plaintext-modulus BFV multiplication. Its arithmetic correctness does not establish the auxiliary-information security claims expressly left open in section 15.

Revision after inspecting the existing digit readout: preserve the current signed native PRF domain and base-p digit output. The earlier proposal to require nonnegative-domain recompilation was unnecessarily broad. Move modular half factors into freshly encrypted targets, retaining the required public affine offsets; never halve an evaluated noisy encoding. The binary/CBD examples in the note are not the application's required output distribution.

## 1. Required contracts

- A state is two polynomials, a public matrix is four polynomial blocks, and each branch ciphertext is a 2-by-2 polynomial matrix. The BFV backend introduces no gadget matrices or trapdoor anchors, including in its decoder construction.
- Write the preprocess payload secret as `r = (r_bar, 1)` and the online mask secret as `s = (s_bar, 1)`. These distinguish the user's preprocess secret `t` from the obsolete BGV plaintext-modulus parameter. Set `S = ||s_bar||_1 + 1`.
- Levels satisfy `q_i = p_i q_(i+1)`, with disjoint dropped and retained CRT primes. Ordinary state scale is exactly `p_i`, not a fixed plaintext-modulus ratio chosen independently of the next transition.
- An ordinary state satisfies `c_i = s A_i - p_i mu_i r + e_i (mod q_i)`. Sample integer small errors once and reduce the same errors across towers.
- Add and public multiplication operate at a common level and scale. A monomial is the usual payload, but intermediate polynomial sums retain their declared domain and norm bounds.
- All terminal contributors merged before rounding share the same input modulus `Q = P q`, output modulus `q`, drop factor `P`, secrets, and coefficient layout. Their target scales need not match.
- The online preparation phase can access `s`; public evaluation cannot. RHS material is fixed before its public path is prepared. Public-label-dependent RHS aggregation is followed by preparation of its corresponding public path.
- Validate public structure and parameter inequalities while compiling/preparing. Do not inspect encrypted one-hot values, reconstruct secret coefficients, or repeat setup checks at every evaluation gate.

## 2. Independent algebraic assessment

### Ordinary ModDown

For odd `P`, define

```text
rho_P(z) = centered(z mod P)
MD_(P,q)(z) = (z - rho_P(z)) / P mod q.
```

This is coefficientwise nearest division, with no half-integer ties for integer inputs. If `z = s K - P T + U (mod P q)`, then

```text
c' = MD(z), A' = MD(K)
c' = s A' - T + e' mod q
e' = (U + s rho_P(K) - rho_P(z)) / P mod q.
```

The numerator is divisible by `P`. With `gamma = (P-1)/(2P)`, an integer error representative obeys `||e'||_inf <= ||U||_inf/P + (S+1) gamma`. This verifies the public-mask rounding term in section 3; rounding only the encoding would be incorrect.

### Arbitrary row target and intermediate scale

For a branch, generate fresh material at `q_(i+1)` satisfying

```text
sigma_k(r) C_k = T_k + F_k mod q_(i+1),
d_k = s A_d,k - p_i sigma_k(s) + e_d,k mod q_i.
```

Both equations have unscaled fresh error. With tildes denoting exact centered lifts to `q_i`, compute

```text
M_k = sigma_k(A_i) C_tilde_k
B_k = MD_i(M_k)
K = sum_k A_d,k B_tilde_k
z = sum_k [d_k B_tilde_k + sigma_k(c_i) C_tilde_k]
A_next = MD_i(K)
c_next = MD_i(z).
```

Substitution gives

```text
z = s K - p_i sum_k sigma_k(mu_i) T_k + U mod q_i
U = sum_k [sigma_k(s) rho_(p_i)(M_k)
           + e_d,k B_tilde_k + sigma_k(e_i) C_tilde_k
           - p_i sigma_k(mu_i) F_k].
```

The low-base ciphertext relation can be used here because its wrap is multiplied by `p_i`, and `p_i q_(i+1) = q_i`. Applying the preceding ModDown lemma proves

```text
c_next = s A_next - sum_k sigma_k(mu_i) T_k + e_next mod q_(i+1).
```

No divisibility or invertibility condition on the target scale appears in this proof. In particular, a target containing `H_t = q/pi_t`, which is not invertible modulo `q`, is valid. An arbitrary scale means a chosen residue in the output ring; it does not promise recovery of an arbitrary integer message from a noninjective scaled target.

For a row-valued table on `0 <= x < W`, with power-of-two `W | N`, the note's compiler uses

```text
D_k,f = W^-1 sum_(v=0)^(W-1) f[v] X^(-kv) mod q
sum_k sigma_k(X^x) D_k,f = f[x].
```

For two inputs, use `T_k = sigma_k(X^(B y)) D_k,f`, with a domain covering `x + B y`. Ordinary state targets are `f[z] = p_next X^L(z) r`. Terminal targets may instead be any two-component row. RHS automorphism applies to its monomial as well as to the state.

### Noise and target weights

For monomial inputs, with `J` branches, the section 6 bound is

```text
E_next <= [E_i sum_k K(C_tilde_k) + E_d sum_k K(B_tilde_k)]/p_i
          + J E_F + (J S + S + 1) gamma_i,
K(V) = max_column sum_row ||V_entry||_1.
```

Centered 2-by-2 matrices at `q_(i+1)` satisfy `K(V) <= N(q_(i+1)-1)`. Therefore a uniform bound independent of target weight exists, even for targets weighted by `2^j`, `H_t`, or `H_t Delta_*`. Actual noise and actual matrix norms may change with the weight: they are not claimed to be identical.

For general `mu`, replace each fresh-ciphertext contribution by `||mu||_1 ||F_k||_inf`; account separately for any RHS aggregation or non-monomial multiplier. Track payload bounds through Add and public multiplication. A one-hot contract can tighten the selected payload to a monomial, but does not eliminate errors from unselected ciphertexts. Routing one digit into one coefficient also does not restrict its encryption error to that coefficient.

Freshly generate each differently weighted branch target before adding error. Multiplying an existing ciphertext or state by a large weight would scale its error and is not an implementation of this theorem. This also allows a modular inverse of two inside a fresh target, despite its large centered representative. Monomial routing of a complete raw pair is allowed because it preserves norms and commutes with the secret.

## 3. Refresh construction without gadget matrices

Choose a common terminal modulus `q = product_t pi_t`, with at least two primes for a nontrivial low-order masking interval, and let `H_t = q/pi_t`. Choose output scale `Delta_*`; continuation as an ordinary state requires `Delta_* = q/q_next` for the next scheduled transition.

At the final transition from `Q = P q`, compile the following fresh row targets. The output relation is `c = s A - f[z] + e`. Write the PRF digit base as `p` (distinct from the level drop `p_i` and terminal drop `P`). In the existing native profile, `p = 2*p_prime`, and integer digit `d(z)` ranges over `[1-p/2, p/2]`. It satisfies `d(z+n') = 1-d(z)`, where `Q_LWR = 2*n'`. Its centered affine part `g(z) = d(z)-1/2` negates under the signed native wrap. Here `1/2` means the modular inverse of two at the terminal odd modulus.

| Contribution | Terminal target `f[z]` | Desired output contribution |
| --- | --- | --- |
| Data | `H_t Delta_* X^L(z) r` | `-H_t Delta_* X^L(z) r` |
| Mask digit at coefficient `v`, component `ell` | `-p^j g(z) X^v u_ell` | `+p^j g(z) X^v u_ell` |
| Fresh-error digit | `-H_t p^j g(z) X^v u_ell` | `+H_t p^j g(z) X^v u_ell` |

Here `u_0=(1,0)` and `u_1=(0,1)`. Sum raw data, mask, and fresh-error pairs before the outer ModDown. After finalization, add the exact public affine/centering offsets described below; these change neither the public matrix nor the error bound. The resulting pair satisfies

```text
v_t = s A_sum,t - H_t Delta_* mu r + m_t + H_t e_* + e_eval,t mod q.
```

Preprocess a new public 2-by-2 matrix `A_*` and `D_t = A_sum,t - H_t A_*`. Online preparation samples a small row `e_dec,t` and computes the two-polynomial decoder

```text
d_t = s D_t + e_dec,t mod q.
```

No generic preimage sampler is required because this preparation is explicitly allowed to know `s`. Evaluation subtracts this decoder only from the matching combined target:

```text
y_t = v_t - d_t
    = H_t (s A_* - Delta_* mu r + e_*) + m_t + f_t mod q
f_t = e_eval,t - e_dec,t.
```

If `||m_t + f_t||_inf < H_t/2`, exact `MD_(H_t,pi_t)` returns the desired encoding residue modulo `pi_t`. Assemble these tower residues into a two-polynomial RNS encoding:

```text
c_* = s A_* - Delta_* mu r + e_* mod q.
```

Fresh-error digits are shared across CRT channels so that `e_*` is one small integer row. Mask labels include the channel; fresh-error labels omit it and use a distinct purpose tag. Labels also bind the refresh site, instance/input identity, component, coefficient, and digit index. Preserve the existing distribution and label semantics rather than replacing them with the note's example CBD construction.

For `d` digits let `S_d = sum_(j=0)^(d-1) p^j` and `a = 1-p/2`. The native integer sum `sum_j p^j d_j` occupies every integer from `a S_d` to `(p/2) S_d`. The raw terminal outputs encode `sum_j p^j (d_j-1/2)`. For a mask with `d_m,t` digits, add the public offset

```text
O_m = (1/2 - a) S_(d_m,t) - p^(d_m,t)/2
```

to obtain exactly the centered interval `[-p^(d_m,t)/2, p^(d_m,t)/2 - 1]`. For fresh error with `d_f` digits, preserve the existing native integer sum by adding `H_t S_(d_f)/2` to the channel output. Its exact support is `[a S_(d_f), (p/2) S_(d_f)]`; account for its mean and endpoints instead of calling it a zero-mean CBD. All offsets are computed modulo `q` and are added to rows, not multiplied into encryption errors.

Choose mask counts `d_m,t` and shared fresh count `d_f` separately. For a required support cardinality `L`, the initial count is the smallest `d` with `p^d >= L`, computed with integer products. For masks iterate this count with the resulting evaluation/decoder noise bound and verify `p^(d_m,t) + 2 B_exp,t < H_t`; the conditional hiding-width requirement from section 15 supplies the lower support requirement. For fresh error derive its required support from the application's fresh-error parameters, then check the actual native-support endpoints against the next-state error budget. Do not choose a larger count without accounting for its larger fresh error or claim that every arbitrary support/distribution is realized by full independent digits.

For one refresh the distinct scalar PRF digit outputs number

```text
2 N (sum_t d_m,t + d_f).
```

Fresh-error terminal emissions still occur in every channel, even though their PRF digits are shared. Budget evaluation, terminal emissions, helper storage, and output dimension separately. The per-channel contribution count is `1 + 2 N (d_m,t + d_f)`, including data. Different large weights `p^j` and `H_t p^j` are encrypted as fresh targets; they do not multiply previously evaluated errors. The existing supported base-p profile constraints remain in force; this does not silently add arbitrary odd digit bases.

This construction refreshes error at the terminal modulus `q`. It does not recover dropped prime towers. Compile all input paths backwards from a common final `(P,q)` and use valid identity transitions where needed. Do not lift an already depleted state to restore its secret relation. PRF evaluation must finish with the available chain without depending on the mask it has yet to produce.

Security remains subject to section 15: a large mask interval and small error prove rounding correctness, but not masking security when PRF digits, evaluation errors, helpers, and decoders are correlated. The direct decoder gives a concrete authorized preparation algorithm, not a proof that forbidden decoders cannot be synthesized from the public transcript.

## 4. Implementation changes, in dependency order

### A. Rename and fix the semantic contract

Rename the backend module `crates/exponent-lut/src/bgv/` to `bfv/`, `Bgv*` types to `Bfv*`, `ExponentBgv.lean` to `ExponentBfv.lean`, and corresponding exports, tests, artifact tags, manifest fields, architecture documentation, and plan references. Keep `ExponentLUTEncodingScheme` unchanged. Preserve historical document titles when citing source PDFs; do not rename unrelated mathematical BGV references.

Remove the BFV backend's `plaintext_modulus` parameter and the distinction between normalized error and its old plaintext-modulus multiple. Sample and encode `e`, `F`, and `e_d` directly. Regenerate affected artifacts after implementation; do not add compatibility loaders for the obsolete backend format or reuse its checkpoints.

### B. Implement ordinary exact RNS ModDown using the existing conversion machinery

The current primitive code already has centered mixed-radix conversion and retained/dropped tower plans (`crates/primitives/native/ExactBasis.cc`, the DCRT polynomial wrappers, and their GPU counterparts). It currently rejects multiplier one and exposes BGV-style plaintext-preserving semantics. Reuse its exact conversion/NTT machinery, including the plumbing credited there to PR #157, while changing the planned BFV operation to ordinary ModDown throughout primitives, IR, DSL, and runtime.

For each retained prime `pi`, compute

```text
rho = centered(z mod P), from the dropped residues
out_pi = (z_pi - rho_pi) * inverse(P mod pi) mod pi.
```

Determine the centered representative exactly with word-sized mixed-radix digits; do not reconstruct coefficient values as big integers. Public modulus products and planning arithmetic may use big integers. Drop a product of primes with one rounding, not independently rounded successive prime drops. Perform centering in coefficient representation, not NTT slot representation. Reuse this primitive for the final `H_t` extraction by retaining only `pi_t`. Recomposition assembles residues; it need not reconstruct full-width integers either.

Keep GPU arithmetic on device, batch coefficients/towers, reuse prepared constants, and avoid host synchronization or CPU fallbacks. Any operation-name/parameter change must update its actual execution and Lean semantics together.

### C. Preserve the gate architecture and polynomial blocks

Retain `Add`, `MulPublic`, and `Lut { rhs: Option<_>, lut_id: Option<_> }` semantics. Builder `private_mul` continues to lower to an identity two-input LUT, never a primitive private-multiplication gate. Same-level Add and public multiplication use direct block arithmetic and their exact noise/norm transfers.

No public Program builder signature change is required by row-valued terminal compilation. Keep `lut(input, table, rhs)` and `private_mul(lhs, rhs)`. Application users do not provide row matrices, terminal scales, or inverse-two constants through those methods. Readout/refresh preparation lowers the declared digit function and requested weights into scheme-specific terminal material. Ordinary Program outputs remain exponent encodings.

Adapt the existing fresh branch sampler, online mask helpers, public preparation, and encoding evaluator to the section 2 equations. Keep scalar and structural-family traversal shared through the trait. Keep raw terminal accumulators separate from ordinary state values so mixed-scale rows cannot silently become next-step states.

### D. Move affine normalization inside fresh terminal targets

`crates/exponent-lut/src/bgv/readout.rs` currently requires even plaintext modulus, compiles `2*f-1`, and applies modular division by two to both output and public matrix. With an unscaled error, an odd error multiplied by `2^-1 mod q` can become large. Renaming the code or deleting its parity check would be incorrect.

Replace that transport by compiling a fresh target for `w*(2*d-1)/2`, with the sign convention required by `c=sA-f+e`, then adding `w/2` as a public row offset when an integer digit output is requested. The resulting encoding is `sA+w*d+e`, with no division of `A`, `c`, or their errors after evaluation. For aggregate mask/fresh generation, combine the affine and centering offsets into the formulas in section 3. Preserve existing terminal preparation, indexed material, family batching, and aggregate-before-rounding infrastructure. Remove post-evaluation half factors and `t/2` error metadata, but retain the mathematically necessary public affine offsets.

This works on the existing signed native domain. A linear row-valued LUT must send `-X^x` to the negative of its output on `X^x`. The current digit function is affine-compatible: `d(x+n')=1-d(x)`, so `g=d-1/2` is negating. The two halves return respectively `w*g+w/2=w*d` and `-w*g+w/2=w*(1-d)`. Thus the existing PRF function, exponent modulus, and signed accumulator layout can remain unchanged. The fresh target may contain a large modular inverse of two; section 2's weight-independent bound applies because error is sampled after target formation.

For arbitrary user-declared terminal functions that lack this affine compatibility, the note's bounded nonnegative domain is a separate valid compilation strategy; it is not a required migration of the existing LWR PRF. Do not generalize a linear LUT to an arbitrary full signed-domain function. One-hot selection must produce the declared signed monomial before the affine offset is used, since one affine offset cannot normalize an arbitrary sum of differently signed basis elements.

### E. Connect generic PRF setup and terminal routing

Keep the logical sparse-LWR accumulation, selector semantics, labels, output scheduling, and refresh traversal dependent only on scheme traits. Move backend-specific material generation and dimensions behind those traits. BGG continues to use its own encoding dimensions and constructions; BFV reports two components.

Preserve the builder identity-LUT path for private multiplication. For the optimized section 8 RHS preparation, expose a scheme preparation operation for one-hot weighted RHS families. The application declares selectors and public shifts, while BFV implements the branch aggregation. This must not introduce concrete BFV matrix operations into the generic PRF interpreter.

For a slot-composed RHS, sum shifted ciphertexts in the low modulus first, then center/lift the sum and prepare its public path. Account for `s_b E_F` even when the plaintext selection is one-hot. Compile the existing LWR digit readout into the terminal target using the signed-domain construction above. Preserve the separate `mask_digit_counts` and `fresh_digit_count` topology already present in `noise_refresh/simulation.rs` and the base-p label indexing in `noise_refresh/refresh.rs`. Reuse a terminal material across coefficient routing only for norm-preserving monomial shifts; different large weights require fresh target encryption.

Preprocess everything depending on the preprocess secret and fixed public data. Generate `s`-dependent helpers and direct decoders during online preparation. If a public label becomes known only in online preparation, its dependent public aggregation/preparation necessarily occurs then; cache it once rather than claiming it can be precomputed without the label. A fixed preprocessing schedule may instead declare all supported labels ahead of time.

### F. Remove BGG-specific assumptions from the shared refresh interface

`crates/exponent-lut/src/scheme/refresh.rs` currently requires an anchor, witness, and shared decoder base, while `noise_refresh/evaluation.rs` switches before applying the decoder. Generalize preparation artifacts so BFV can supply its direct two-column decoder without fake anchors or preimages.

Make the arithmetic stages explicit: raw terminal aggregation/finalization; any backend-specific pre-decoder conversion; combined-target decoder subtraction; residue extraction; recomposition. BFV performs `MD_(P,q)`, subtracts its decoder at `q`, then performs `MD_(H_t,pi_t)`. Do not extract the residue before decoder subtraction. BGG adapters preserve their existing pre-decoder conversion and use an identity post-decoder extraction where appropriate. Identity adapters should create no runtime arithmetic nodes.

Keep the common evaluator responsible for connecting these trait operations and for structural batching. Do not hide the whole refresh protocol in a backend-specific opaque callback. Prepared contracts bind the output key, channel, target weights, public aggregate, and helper material once.

### G. Rebuild noise, parameter, and resource planning

Replace the old `t*E` accounting with section 6's absolute-error recurrence. For heterogeneous terminal contributors, sum their actual symbolic raw bounds and add the outer `(S+1) gamma` only once. Include decoder error, one-hot RHS aggregation errors, and all coefficient/component emissions. Use public norm bounds, not expensive measurements of encrypted matrices at runtime.

Plan backwards from the terminal modulus using actual prime products. Ensure intermediate `E_i < p_i/2` where high-scale decoding is required, and ensure the refresh inequality `p^(d_m,t) + 2 B_exp,t < H_t`. Compute fresh and mask digit counts separately and iterate mask counts and their resulting terminal error budgets to a fixed point. Include `d_f` in the exposed evaluation-noise budget even though fresh PRF inputs are shared between channels. Distinguish this correctness condition from the note's conditional statistical-width bound and its still-open joint-distribution security requirement.

Include the actual sparse-LWR stage count, ring/domain requirements, per-weight helper multiplicity, public caches, indexed families, and streaming batch limits. Two columns do not imply small modulus depth: the note's conservative recurrence can produce approximately quadratic modulus growth per backward step. Its example profile is not a security-approved or measured production configuration.

## 5. Planned validation and completion criteria

Implementation validation uses focused unit tests and the existing trusted polynomial arithmetic to check:

1. Ordinary block ModDown and exact centered extension, including multi-prime drops and wrap boundaries, agree across CPU and GPU. Big integers are allowed in a test oracle, not the runtime coefficient path.
2. Arbitrary row targets, the fused two-input LUT, and a multistep chain satisfy the stated encoding and exact error equations. Include negative and noninvertible terminal weights and actual supported input domains.
3. Add, public multiplication, and one-hot selection preserve payload semantics and conservative but nonduplicated noise bounds. Include a legitimate non-monomial intermediate.
4. Mask/fresh-error digit readout works without post-evaluation modular halving, for both components, both halves of the signed native domain, supported nonbinary bases, separate mask/fresh digit counts, and all coefficient positions. Include odd evaluation errors so an accidental divide-by-two cannot pass unnoticed. Structural indexed/shared paths and reopened artifacts agree with the scalar mathematical semantics and exact native/centered offsets.
5. Raw mixed-target aggregation followed by one ModDown satisfies its equation. Do not require bitwise equality with separately rounded sums.
6. Combined-target direct decoders followed by residue extraction recover `s A_* - Delta_* mu r + e_*` exactly; the PRF-derived fresh error is shared across channels and old errors disappear under the stated rounding condition.
7. A refresh with `Delta_* = q/q_next` feeds a subsequent ordinary LUT at the remaining level. This verifies continuation, not modulus restoration.
8. The same generic PRF/refresh orchestration supports BGG and BFV with their respective dimensions, preparation, and operation order. Preserve BGG behavior.

Update the Lean arithmetic statements to unscaled error, ordinary ModDown, arbitrary terminal target correctness, and conditional refresh reconstruction. Do not label masking security as proved by arithmetic tests or these identities. Follow the repository's narrow validation, formatting, CPU/GPU build, and GPU repetition requirements when execution is authorized. Integration tests remain excluded unless explicitly requested.

Completion requires the new BFV path to use production preprocessing and online preparation through the traits, not only standalone terminal tests. Report remaining security assumptions and actual parameter/resource limits separately from functional correctness. Do not promise unlimited-depth evaluation or security/performance from the PDF's toy results.

## Implementation checkpoint

The BFV backend now implements unscaled state/RHS/helper errors, ordinary RNS
ModDown, fresh weighted affine readout, and direct-decoder refresh.
`noise_refresh/prf_batch.rs` and `noise_refresh/direct.rs` provide trait-based
structural accumulation and complete refresh orchestration. BFV prepared PRF
and refresh artifacts can be exported and reopened, with the mask secret first
introduced during online preparation. The arithmetic unit fixture exercises
actual native base-4 PRF digits, separate mask/fresh counts, both polynomial
components, CRT-shared fresh digits, exact error replacement, and continuation.

`BfvParameters::plan_refresh` solves the digit/error fixed point with the actual
terminal branch count and decoder error. It distinguishes distinct PRF outputs
from per-channel weighted emissions and reports terminal artifact storage, not
measured peak memory. The ordinary chain planner audits intermediate scale
separation. General Program DAGs still require their own traced input bounds.

The BGG witness-based sampler/artifact driver remains a backend adapter; the
common Program and refresh arithmetic are shared. BFV's direct preparation does
not construct BGG witnesses or anchors. The native CUDA conversion retains its
64-limb limit. CRT recomposition currently uses exact single-tower extension
and idempotents, without full-width coefficient reconstruction; its work grows
quadratically in output tower count. Refresh restores error at the remaining
level, not consumed moduli. Section 15's joint-distribution security condition
remains an assumption, separate from these arithmetic tests and Lean identities.

### Implemented entry points and phase boundaries

1. Construct the chain with `BfvParameters::plan` (or audit a full DAG with
   `search_program`). Account for the actual selection/RHS gains and feed its
   resulting data/PRF noise values to `BfvParameters::plan_refresh`; they may
   come from separate Programs. Its workload request contains no detached
   level or input-noise bounds. The returned digit
   plan determines the canonical `RefreshPrfLabelIndex`; its fresh range is
   shared across channels and its mask ranges are channel-specific.
2. With the payload secret only, sample input contributions, selector RHS
   families, and public mask banks. `PreparedSparseLwrPrf::new` consumes fixed
   public label monomials in label-major active-cell order. Each BFV selection
   composes its low-base RHS before preparing one direct private product. Public labels
   must belong to the declared native profile and frozen refresh instance.
3. Compile data via `sample_refresh_data_material` at `H_t * Delta_*`. Call
   `PreparedDirectRefresh::new` with the native PRF readout, canonical digit
   family, planned counts, and channel declarations. The output key supplied
   for decoder targets must be the recomposition key; setup owns this contract.
   This prepares fresh weighted mask/error targets, raw sums, public ModDown,
   decoder targets, and public affine offsets.
4. Export the prepared PRF and refresh bundles, plus private input material,
   through the BFV artifact adapters. Keep their descriptor identities in the
   trusted setup contract and bind runtime imports to the producer manifest.
5. During online preparation, sample `BfvMaskSecret`, encode the restored input
   contributions, prepare online mask helpers from the cached public banks,
   and call `prepare_decoders`. The evaluator then calls the prepared PRF and
   refresh `evaluate` methods, receiving helpers/decoders rather than secrets.
   Encoding ModDown remains necessary online. CRT extraction happens after
   decoder subtraction, and the resulting encoding can enter a remaining LUT.

The BFV Lean module additionally proves exact centered-error removal and
full-modulus wrap invariance under the strict refresh rounding inequality.
These coefficient lemmas complement the gate/decoder algebra; they are not a
proof of the full exported graph or of the masking security assumption.

### Validation recorded on 2026-09-08

- CPU BFV unit suite: 29 passed, including persisted native-PRF refresh and
  continuation, signed readout, structural families, and noise planning.
- Scheme/BGG regression filter: 8 passed. Native PRF filter: 8 passed.
  Refresh routing/decoder regression filter: 9 passed. Digit planning: 2 passed.
- CPU exact-RNS tests: 2 passed. GPU exact-RNS test: 2 consecutive passes.
- GPU persisted native-PRF/refresh/continuation test: 2 consecutive passes from
  the same binary (845.11 s and 841.45 s). Repetition was limited to two by the
  user's instruction. The fixture includes file artifact persistence and uses
  small arithmetic parameters; these timings are not throughput benchmarks.
- `cargo test -r --workspace --lib --no-run` and its `--features gpu` variant
  completed without warnings. `cargo +nightly fmt --all -- --check` passed.
- `lake build ExponentBfv` and `lake build ExponentLut` passed, including the
  new strict-error ModDown and wrap-invariance coefficient lemmas.
- Integration tests were not run. The security assumptions in section 15
  remain outside these arithmetic checks.
