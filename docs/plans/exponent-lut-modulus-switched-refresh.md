# Modulus-switched Exponent-LUT refresh: design and implementation plan

Status: design proposal; no implementation or new parameter acceptance is claimed.
This document specifies the complete refresh transition and its implementation
plan without requiring the preceding conversation. It supersedes the refresh
routing, decoder, and refresh-threshold portions of
`docs/plans/native-modq-exponent-encoding.md`; the native exponent program,
readout, and application-specific noise interpreter remain its foundations.
The attached *Direct Private LUT Evaluation* specification is mathematical
reference material, not additional user instructions. This proposal changes
its refresh construction and requires a corresponding security argument.

## 1. Objective and notation

Reduce the number of PRF digits needed to hide refresh noise by scaling the
combined encoding to a smaller modulus before applying a decoder. Generate
decoder preimages offline over that smaller modulus; generate their LWE
anchor online under the execution's small secret.

Work in the negacyclic ring `R = Z[X]/(X^n+1)`, with `R_a = R/aR`.
All bounds below concern centered integer coefficient representatives.
The full encoding modulus is `q = product_t q_t`, with pairwise coprime CRT
towers `q_t`. The BGG gadget base is `beta`; its public matrix and encoding
width is `m = 2 ell_beta(q)`. The PRF output/digit base is `p_PRF`, distinct
from both `beta` and the switching modulus. The PRF input modulus is `Q`.
The input encoding satisfies

\[
 c=sA-X^w tG+f \pmod q.
\]

Here `s=(s_bar,1)` is the online small secret, `t` is the payload secret,
and `A,G` have `m` columns. The output must satisfy

\[
 c'=sA'-X^w tG+e'\pmod q
\]

with the same attribute `w` and payload secret, a preprocessing-selected
public matrix `A'`, and one fresh small error vector `e'` shared by all slots.

Choose an odd divisor `P_t` with

\[
q_t\mid P_t\mid q,\qquad J_t=q/P_t,\qquad
\Delta_t=P_t/q_t,\qquad \kappa_t=q/q_t=J_t\Delta_t.
\]

Initially select `P_t` as a product of existing CRT towers. Slots may share
one `P` if it contains all their towers. A common divisor containing every
tower is `q`, so one genuinely smaller common `P` cannot serve every slot.
The degenerate `P_t=q` is a useful correctness comparison. `P_t=q_t` usually
cannot satisfy the mask threshold since `Delta_t=1`.

## 2. Exact switching and decoding

For an integer coefficient representative `x`, define

\[
 \operatorname{MS}_t(x)=\left[\left\lfloor x/J_t\right\rceil\right]_{P_t}.
\]

Apply this coefficientwise to polynomials and matrices. This is nearest
integer rounding after integer scaling, followed by reduction into the
destination ring; it is neither a modular inverse nor truncation. Odd `J_t`
precludes half-integer ties. For any integer `k`, switching `x+kq` changes
the rounded integer by `kP_t`, so the output is representative-independent.
Also `MS_t(x+J_t y)=MS_t(x)+y mod P_t`. This exact translation property is
needed for nominal mask and fresh-error separation.

Preprocessing prepares the scaled state public matrix and PRF routes. Their
online values obey

\[
\begin{aligned}
u_t&=sA_t-\kappa_t X^w tG+f_{u,t},\\
c_{m,t}&=sA_{m,t}+J_t m_t+e_{m,t},\\
c_{e,t}&=sA_{e,t}+\kappa_t e'+e_{e,t}.
\end{aligned}
\]

All three equations are over `R_q`. State scaling uses
`G^{-1}(kappa_t G)`, with its actual decomposition gain. Define

\[
v_t=u_t+c_{m,t}+c_{e,t},\quad
A_{\mathrm{sum},t}=A_t+A_{m,t}+A_{e,t},\quad
h_t=f_{u,t}+e_{m,t}+e_{e,t}.
\]

Switch the sum once: `v_hat_t=MS_t(v_t)`. Preprocessing likewise computes
`A_hat_t=MS_t(A_sum,t)` once after summing public matrices. Switching each
summand and then summing is a different algorithm with different rounding
error and must not replace this order.

For each distinct `P`, preprocessing generates `(B_P,T_P)` with the existing
trapdoor sampler over `R_P`. Let `B_P` have two rows and `m_B(P)` columns.
It samples a short preimage `K_t`, of shape `m_B(P_t) x m`, satisfying

\[
B_{P_t}K_t=\widehat A_t-\Delta_t[A']_{P_t}\pmod{P_t}.
\]

This target is entirely public and independent of `s`. Reduction `[A']_P`
is ordinary ring reduction, not scaled rounding. The gadget width used by
the low-modulus trapdoor can differ from `m`, but `K_t` retains exactly `m`
target columns.

During each online execution, the party holding `s` samples `e_P` and creates

\[
b_P=sB_P+e_P\pmod P.
\]

One anchor is shared among all decoders using the same `B_P,P,s` in that
execution. A new execution creates fresh anchor errors and uses its actual
execution secret; no secret-dependent decoder is generated offline. The
secret in this equation is `s_P=[s]_P`, preserving the same small integer
coefficients. Never switch `s` by `MS`, which would erase small coefficients,
and never independently resample its value for different moduli. Sample the
integer secret once and reduce it into each ring; anchor errors are fresh
per distinct anchor. The
evaluator computes `d_t=b_P K_t` and obtains

\[
\boxed{y_t=\widehat v_t-d_t
=\Delta_t(sA'-X^wtG+e')+m_t+\eta_t\pmod{P_t}.}
\]

Finally calculate `z_t=[nearest(y_t/Delta_t)]_(q_t)` coefficientwise and
CRT-recombine `z_t` to `R_q`. Under the strict bound below this recovers the
claimed `c'`. The final division is integer nearest rounding on coefficient
representatives; reduction modulo `q_t` makes it independent of a change of
representative by `P_t`. Correctness excludes ties.

The evaluator must not receive the old high-`q` decoder for this combined
target. Such a decoder would reveal a value of the form
`kappa_t z + J_t m_t + f_t`; reducing it modulo `J_t` reveals `f_t mod J_t`.
An instruction to switch before subtraction does not prevent this alternate
computation by a holder of both artifacts.

## 3. Mask and fresh-error construction

For each slot and coefficient use `d_m,t` independent ideal base-`p_PRF`
digits, domain-separated by refresh, slot, component, coefficient, and digit.
The PRF hybrid must justify replacing actual outputs by these ideal digits.
Let `M_t=p_PRF^(d_m,t)` and specify the final mask distribution exactly as

\[
m_t\ \leftarrow\ \{-\lfloor M_t/2\rfloor,\ldots,
M_t-1-\lfloor M_t/2\rfloor\},\quad B_{m,t}=\lfloor M_t/2\rfloor.
\]

This is a contiguous interval of exactly `M_t` integers. For even `M_t` it
is `[-M_t/2,M_t/2-1]`, not a symmetric integer interval.

If the existing readout returns `g=d-1/2` for consecutive integer digits
`d in [a,a+p_PRF-1]`, set `S_t=sum_(c<d_m,t) p_PRF^c`.
After weighted routing, add the public coefficient offset

\[
J_t\bigl(S_t/2-aS_t-\lfloor M_t/2\rfloor\bigr).
\]

The resulting payload is `J_t(sum p_PRF^c(d-a)-floor(M_t/2))`.
Half offsets use the inverse of two in the odd source modulus, and cancel
exactly in the complete payload. Apply this offset in every output
coefficient and component. It contributes zero operational noise. This
explicit normalization is required: independently centered even-base digits
alone do not establish the declared mask interval.

Use mask decomposition targets

\[
G^{-1}(-J_t p_{\rm PRF}^{c}X^j u_2\delta_k^T),
\]

where `u_2` selects the second payload row and `delta_k` the target component.
Do not multiply an existing noisy mask encoding by raw `J_t`. The new noise
gain comes from the actual balanced decomposition of the new target.

Fresh-error routes retain targets
`G^{-1}(-kappa_t p_PRF^c X^j u_2 delta_k^T)` and their matching public
half-offsets. Use the same fresh digit outputs, support, and centering in
every slot; fresh labels have no slot coordinate. Let their exact final
support yield `B_e=max |e'_coefficient|`. Preserve the original hard condition
`B_e < r` for every prime factor `r` of every original `q_t`. A conservative
bound is permitted; it must describe the actual fresh distribution.
Reducing decoder modulus does not relax this condition or shrink the output
encoding width.

## 4. Error bound and acceptance conditions

Let `H_t` bound `||h_t||_infinity`. Derive it from the actual state-scaling,
mask-route, and fresh-route decompositions, without decoder error or nominal
mask/fresh payload magnitude. In particular, recompute mask-route gains when
either `P_t` or `d_m,t` changes. Write rounding residuals as `rho_v,rho_A`.
For a right-action matrix `D`, define
`gamma_1(D)=max_j sum_i ||D_ij||_1`, where the inner norm sums polynomial
coefficient magnitudes. With input error bound `E_state` and individual
readout error bounds `E_read,label`, one concrete sufficient bound is

\[
H_t=\gamma_1(D_{\kappa,t})E_{state}
+\sum_{\text{mask labels }a}\gamma_1(D_{m,t,a})E_{read,a}
+\sum_{\text{fresh labels }a}\gamma_1(D_{e,t,a})E_{read,a}.
\]

Then, using consistent integer lifts modulo `P_t`,

\[
\eta_t=h_t/J_t+\rho_v-s\rho_A-e_{P_t}K_t.
\]

For odd `J_t`, each rounding residual has absolute value at most
`r_J=(J_t-1)/(2J_t)`. Negacyclic coefficient convolution gives

\[
\|\rho_v-s\rho_A\|_\infty
\le r_J(1+\textstyle\sum_i\|s_i\|_1)
=r_J(\|\bar s\|_1+2).
\]

The current ternary `s_bar` satisfies `||s_bar||_1 <= n`. The public planning
bound uses `n`, not an undisclosed execution-dependent secret norm. At
`n=65536` this rounding bound is below `32769` for `J>1`.

Let `D_t^(P)` bound `||e_P K_t||_infinity`, based on the low-modulus sampler,
anchor error law, preimage shape, and cutoff. It is not multiplied by `P/q`.
For example `D_t^(P)=E_P gamma_1(K_t)` is sufficient when
`||e_P||_infinity<=E_P`; a sampler-certified coefficient cutoff `C_K`
gives `gamma_1(K_t)<=n m_B(P_t) C_K`. Resolve the official sampler default
cutoff for the actual `P_t` and preimage shape; reject a requested trapdoor
cutoff below that default. Deterministic gadget-decomposition preimage bounds
are a separate contract and do not authorize tightening the sampler cutoff.
A sufficient integer error bound is

\[
\boxed{F_t=\left\lceil H_t/J_t+
\frac{J_t-1}{2J_t}(n+2)+D_t^{(P)}\right\rceil.}
\]

For total exposed mask-coordinate count `D_transcript`, sufficient uniform
mask shift-hiding and correctness conditions are

\[
M_t\ge 2^{\lambda_{mask}+1}D_{transcript}F_t,\qquad
\boxed{2(B_{m,t}+F_t)<\Delta_t.}
\]

Also retain `M_t>=Q` where the original mask-domain coverage condition is
required. The non-strict coverage/smudging inequalities accept equality;
the rounding inequality rejects equality. For one refresh with `L` slots,
`D_transcript=L n m`; add all exposed refresh instances for the declared
execution/query budget. More precisely, unequal widths can use
`sum_t D_t F_t/M_t` and its two-shift factor; the displayed global bound is
a simpler sufficient allocation. This statistical comparison is conditional
on a valid ideal-mask hybrid, not a proof of the full protocol.

Worst-case and optional average-case authorities remain explicit. Keep the
program-agnostic interpreter in `crates/exponent-lut/src/noise.rs` and put
refresh rounding and tail handling in `noise_refresh/simulation.rs`.
Retain conservative `Add` handling for correlated inputs. If an accepted
TFHE-style heuristic supplies probabilistic `H_t` or `D_t^(P)`, cover all
uses, shared-anchor decoder outputs, and inspected coordinates by one joint
tail budget. The deterministic rounding term remains additive. Do not assume
independence merely because two routes have different names, silently set
unknown variance to zero, or promote a heuristic tail result to a proven
worst-case bound. Fresh/mask payload support is separate from operational
error variance; count each term once.

The ratio `(H_t/J_t)/Delta_t=H_t/kappa_t` explains the optimization's limit:
shrinking the modulus reduces absolute mask width, not the dominant
noise-to-message-spacing ratio. Rounding and decoder errors create a floor.

## 5. Security obligations and artifact lifetime

Preprocessing exports ordinary public `Mat` artifacts for `B_P` and the
needed public matrices, and relation-bearing `Preimage` artifacts for `K_t`.
Trapdoors stay private to their producer. Reuse existing artifact storage;
do not cache secret online encodings as preprocessing outputs. The online
anchor is produced once per distinct public anchor and execution.
Export public matrices only when required by the protocol, such as `B_P`;
intermediate `A_out` or `A_hat` matrices must not become acceleration-cache
artifacts. Reuse the existing generic relation-bearing `Preimage` cache for
deterministic `G^{-1}` actions. Account for transmitted protocol decoder
preimages separately from deterministic decomposition-cache bytes, even
though both reuse the existing preimage representation.

The remaining security obligation is the joint view: high-`q` program/PRF
encodings, all low-`P` anchors under the same small secret, public switched
targets, and their trapdoor preimages. Individual RLWE estimates, a short
preimage equation, and the smudging inequality do not by themselves prove
that joint view secure. In particular, ideal-mask independence conditional
on public/auxiliary data and cross-modulus shared-secret samples must be
justified. Audit other available helpers for reconstruction of the prohibited
high-modulus combined decoder. Report the lattice security of each low-`P`
anchor distribution, including the fixed augmented secret coordinate, and
the original sparse-LWR/BGG requirements. Existing high-`q` security reports
are not acceptance evidence for new `P_t` values.
The visible high-modulus sum also has the complementary projection
`v_t mod J_t = s(A_sum,t mod J_t)+h_t mod J_t`, since both nominal payloads
are multiples of `J_t`. This is an additional LWE-type view, not bare error;
include it and correlations with low-`P` views in the security analysis.
Multiplying the new decoder by `J_t` does not reconstruct the prohibited old
decoder: the difference retains the secret-weighted public rounding term.

Use the existing ordinary parameter and artifact associations for modulus,
shape, and label consistency. No new commitment, digest-policing subsystem,
provenance ledger, or adversarial simulator protocol is required.

## 6. Current implementation and concrete changes

The following symbols exist in the current checkout; described changes are
proposed, not already implemented.

| Location / existing symbol | Required change |
| --- | --- |
| `crates/exponent-lut/src/noise_refresh/setup.rs`: `RefreshSetupParameters`, `refresh_noise_parameters`, `RefreshParameterSimulationRequest` | Add slot/group switching moduli and mask digit counts; derive exact `J,Delta`; expose per-modulus decoder sampler parameters and security inputs. |
| `crates/exponent-lut/src/noise_refresh/simulation.rs`: `RefreshNoiseParameters`, `RefreshRouteNoiseGains`, `RefreshSlotNoiseReport`, `AverageRefreshSlotNoiseReport`, `ExponentLutNoiseSnapshot` | Model `H/J`, deterministic paired rounding, unscaled low-modulus decoder noise, exact support, strict rounding and smudging; retain detailed stage reports. |
| `crates/exponent-lut/src/noise.rs`: `NoiseDecompositionPlan`, `simulate_program` | Reuse actual decomposition gains and the common program interpreter; no refresh-specific gate rules or generic-simulator fallback. |
| `crates/exponent-lut/src/noise_refresh/refresh.rs`: `RefreshSlot`, `RefreshPrfLabelIndex`, `RefreshCompiler::refresh` | Represent `P_t`; generalize uniform mask-digit indexing to per-slot counts with checked prefix offsets; combine then switch, subtract only low-`P` decoder, round and CRT-recombine. |
| `crates/exponent-lut/src/noise_refresh/setup.rs`: `RefreshPreprocessingProducer::build`, `RefreshPreprocessingArtifactNames`, `RefreshPreprocessingDeclaration` | Build one low-modulus trapdoor anchor per unique group, switch combined public targets, sample and export low-`P` decoder preimages. |
| Same file: `ImportedRefreshSetup::import`, `RefreshOnlineEvaluationRequest::evaluate`, `bind_online_outputs` | Import typed low-modulus artifacts, sample online anchors using the execution secret and fresh errors, route new masks, preserve shared fresh outputs. |
| Same file: `build_refresh_verification`, `RefreshSimulationDecoderTarget`, `mask_route_gain_for_digit_count` | Verify low-`P` equations and original-`q` final relation; derive target/gain metadata for the exact selected moduli and digit counts. |
| `crates/exponent-lut/src/noise_refresh/artifacts.rs` | Reuse typed matrix/preimage artifact mechanisms and lazy family loading; extend metadata only where needed. |
| `crates/exponent-lut/tests/refresh_parameter_search_support.rs`, `parameter_search_unit.rs`, `benchmark_estimate_gpu_tests.rs` | Extend candidate derivation/search and production-equivalent cost accounting; retain existing report fields. |

A minimal proposed field layout extends the existing `RefreshSlot` with
`switch_modulus` and `mask_digit_count`. Derive `J_t` and `Delta_t` from these
fields and `q,q_t` rather than storing redundant independently configurable
values. Existing preprocessing declarations group slots by their exact
`P_t` and associate each group with its `B_P` matrix artifact and producer
trapdoor handle; each slot retains its typed decoder `Preimage`. Existing
online evaluation state holds one `Mat` anchor per group for that execution.
These are ordinary fields/maps in the current declarations, not new encoding
or artifact type hierarchies. `RefreshPrfLabelIndex` uses checked prefix sums
of `n*m*d_m,t` to map ragged per-slot mask groups to existing canonical labels;
the one shared fresh-error group follows them.

`RefreshSetupManifest::bind` in
`crates/exponent-lut/src/noise_refresh/refresh.rs` currently assumes a single
decoder base handle across all slots. Replace that assumption with exact
per-group `P_t,B_P` binding: slots in the same group must share that anchor,
and each decoder must satisfy its own low-modulus target equation. Preserve
the other target, slot-order, shared-fresh, and graph-equation validations.

Do not treat the existing primitive `modulus_switch` as this operation without
repairing its contract: `FinRingElem::modulus_switch` in
`crates/primitives/src/element/finite_ring.rs` currently uses floor scaling
and retains the source modulus field. Matrix wrappers delegate to it and
do not establish a native destination-DCRT nearest-rounding operation.
Implement an exact destination-typed coefficient switching operation through
the existing primitive/DSL/IR/runtime layers after confirming the reusable
conversion path. A proposed name is `round_scale_to_modulus`; it is not an
existing API. Reuse existing `Mat` and ring types rather than introducing a
second encoding representation. Validate source/destination moduli, ring
dimension, columns, and coefficient/evaluation format at that boundary.

The GPU implementation must keep rounding on device, batch coefficient and
tower work, use exact integer/CRT arithmetic, and restore the expected
evaluation format. Dropping CRT limbs alone is reduction, not scaled
rounding. Floating-point approximate switching is insufficient for this
contract. Include both the public-matrix preprocessing switch and online
vector switch in planning and costs; avoid host coefficient reconstruction
in the production GPU path.

For an exact GPU implementation, let `r` be the centered reconstruction of
`x mod J_t` from the discarded CRT towers. Then

\[
\operatorname{MS}_t(x)=(x-r)/J_t\pmod{P_t},\qquad
\operatorname{MS}_t(x)\bmod r_i
=(x_i-[r]_{r_i})J_t^{-1}\bmod r_i
\]

for each retained tower `r_i`. This requires exact centered CRT reconstruction
or basis extension for the discarded residues; treat `J_t=1` as `r=0`.
Existing reconstruction/limb helpers in
`crates/primitives/cuda/src/matrix/MatrixSerde.cu` and
`crates/primitives/cuda/src/matrix/MatrixCrt.cu` are reuse
candidates after checking their range, format, and asynchronous contracts.

The runtime parameter registry in `crates/runtime/src/backend/poly.rs` already
keys rings by modulus and dimension. Reuse it for the multiple `P_t` rings.
The GPU parameter constructor in `crates/primitives/src/poly/dcrt/gpu.rs`
accepts an explicit prime list, whereas the CPU/OpenFHE construction path
generates a basis from depth and bit size. Preserve the selected tower list
and mapping explicitly; do not substitute another same-size modulus or
assume every slot fits a proper nested prefix. Extend the CPU parameter
construction only as necessary to represent that identical selected basis.
The low-modulus sampler and its auxiliary bases must use that actual ring.
Provide ordinary destination-typed reduction for `[s]_P` and `[A']_P`
separately from nearest scaled rounding. For a CRT subset this can select
retained limbs using the explicit prime mapping and compatible NTT format;
reuse a native copy/selection path where available. A generic unvalidated
cast between rings must not stand in for either operation.

Multiple GPU modulus parameter sets can create separate contexts, while the
shared-pool checks in `crates/runtime/src/backend/poly_gpu/fleet.rs` expect
one live context per device. Resolve ownership/context reuse for several
rings on a device, or use properly released phase-local contexts with
explicit lifetime accounting where data dependencies permit. Keep the
pool guard and production-equivalent memory accounting; simply disabling
the guard or arbitrarily serializing all work is not a solution.

## 7. Ordered implementation and validation plan

1. **Complete simulation first.** Extend parameters, exact support/offset
   definitions, route gains, low-`P` decoder bounds, security inputs, reports,
   and candidate search. Use integer/rational arithmetic with exact ceiling
   and strict comparisons. Resolve the average-case authority explicitly.
   Exit with a complete reproducible candidate report or a concrete rejection;
   do not begin evaluator changes on a partially modeled candidate.
2. **Implement the shared switching operation.** Establish its mathematical
   and destination-ring contract in primitives, then wire DSL/IR/runtime and
   GPU support consistently. Preserve architecture layering documented in
   `docs/architecture.md`.
3. **Modify preprocessing and artifacts.** Generate grouped `B_P`, build
   switched combined targets, sample low-modulus `K_t`, and export the
   necessary public artifacts. Remove the replaced high-`q` combined decoder
   from this refresh path rather than maintaining a compatibility branch.
4. **Modify online evaluation.** Sample per-execution anchors, route the
   centered masks, combine once, switch once, apply low-modulus decoders,
   round, and CRT-recombine. Keep all `m` columns and shared fresh labels.
5. **Validate algebra and implementation.** Add targeted unit checks for
   representative invariance, negative coefficients, rounding boundaries,
   `P=q`, invalid divisibility, and destination type. Test mask endpoint and
   offset identities for even bases; check low-`P` preimage equations,
   multi-slot shared-fresh reconstruction, fresh anchors across executions,
   and rejection at exact rounding equality. Compare observed operational
   errors term-by-term with the selected simulation bounds. Check the
   generated artifact graph has no old combined decoder. Use trusted
   primitives/round trips for expected values, not a duplicate evaluator.
6. **Validate production performance.** After correctness acceptance and
   explicit authorization for integration tests, run complete refresh tests
   and GPU benchmark waves through production functions and store/load paths.
   Follow `GPU.md` and `BUILDER.md`; no integration or GPU test was run while
   writing this design.

Search starts from the existing comparison profile B:
`n=65536`, `crt_depth=27`, `crt_bits=50`, `log2(beta)=25`,
`Q=4096`, `p_PRF=32`, sparse-LWR `v=8192,h=18`. This is a comparison input,
not a newly accepted switched-refresh candidate. First hold the program,
dimension, and base fixed and enumerate feasible CRT subsets/groups and
`d_m,t`; derive fresh support and all decoder parameters for every candidate.
Re-evaluate `H_t` because routing changes with the candidate. Subsequent
search may vary original parameters only with an explicit report of the
changed semantic problem and selected security target. Do not reuse old
report identifiers as new acceptance evidence.

Report `P_t,J_t,Delta_t,d_m,t,M_t,B_m,t,H_t`, the rounding term, decoder
term, final `F_t`, strict margin, smudging margin, fresh support checks,
transcript/tail counts, security estimates, and the selected authority.
Candidate feasibility is independent of how many GPUs execute it.

Benchmark accounting includes PRF label count
`n*m*(sum_t d_m,t+d_e)`, each unique anchor's generation/public storage,
decoder sampling/application dimensions, public and online switching,
format transforms, artifact bytes, transfers, final CRT, and temporary
workspace. Reduced PRF digits are a predicted saving, not measured speedup.
Use the existing production/estimator multi-GPU planning and 80% VRAM budget
policy, with actual device capacity and backend scratch space. Distribute
whole matrices across devices and keep all their limbs together. Bound
resident data by wave parallelism rather than total slot count; lazy-load
preimages and release a wave before loading the next. Preserve the full
candidate shape when selecting waves; never shrink ring dimension, columns,
or CRT depth to manufacture a benchmark fit. Report measured wave latency,
wave count, total stage time, maximum parallelism, peak VRAM, and disk bytes
separately for preprocessing and online evaluation.

Completion requires simulation acceptance under a named authority, exact
switching and decoder equations, final original-modulus correctness,
review of the remaining joint-view security obligations, and honest
validation/performance reporting. Until those steps are completed this
document establishes an implementable algebraic proposal, not a validated
secure deployment or a benchmark result.
