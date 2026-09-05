# Modulus-switched Exponent-LUT refresh: design and implementation plan

Status: implemented with unit validation and Profile B noise acceptance under
the explicitly selected average-case authority. This is not joint-view
security acceptance or a production benchmark; see the checkpoint in §8.
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
`G^{-1}(kappa_t G)`, with its actual decomposition gain. Apply this scaling
exactly once to the normal input state `c` to obtain `u_t`; `u_t` already
contains the message scale `kappa_t=q/q_t`. Switching scales by `P_t/q`,
so `kappa_t` becomes `Delta_t=P_t/q_t` without any additional multiplier.
Explicitly, `(P_t/q)(q/q_t) z=(P_t/q_t) z`; here `P_t` is the switching
modulus, distinct from the PRF output modulus `p_PRF`.
For a generic message value `z` (distinct from the nominal mask `m_t`), the
nominal payload mapping is:

| Payload | Before switching in `R_q` | After switching in `R_(P_t)` |
| --- | --- | --- |
| Message (`z=-X^w tG` here) | `kappa_t z` | `Delta_t z` |
| Shared fresh error | `kappa_t e'` | `Delta_t e'` |
| Nominal mask | `J_t m_t` | `m_t` |

These are exact translation terms in the combined switch, not separate
switches of the summands. `J_t=q/P_t` only lifts the nominal mask from its
desired post-switch scale. Do not additionally multiply either the already
scaled message or the fresh-error payload by `J_t`. Define

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

The following table records the baseline-to-implementation migration. The
baseline limitations described in this section motivated the changes;
current implementation and validation evidence is summarized in §8.

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

The GPU implementation must execute the entire coefficientwise `MS_t` on
device for both the preprocessing public matrix `A_sum,t` and the online
vector `v_t`: GPU INTT from the source evaluation format, exact coefficient
integer/CRT nearest switching, then GPU NTT into the target evaluation
format. No coefficient round trip through the host, host reconstruction,
or CPU fallback is permitted in either production GPU switch. Rounding in
the NTT domain is not coefficientwise switching. Dropping CRT limbs alone
is reduction, not scaled rounding; floating-point approximations must not
decide rounding boundaries.
The final nearest decoding from `R_(P_t)` to `R_(q_t)` must likewise stay
on the GPU, with coefficientwise exact integer rounding and no host
coefficient round trip.

Batch and parallelize coefficient, column, and tower work on the GPU; avoid
per-coefficient or per-column host launch loops. Use VRAM-calibrated waves
through the existing symmetric multi-GPU production/estimator planner,
including transform and switching scratch space. Preserve its capacity-based
distribution and whole-matrix limb placement; add no fixed wave caps or
unnecessary synchronization. Include both switches and their source INTT
and target NTT transforms in planning and measured costs.

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
   Compare selected GPU output coefficients against an exact CPU reference
   obeying the same nearest-switch contract, including negative values and
   both sides of rounding boundaries. Cover both public-matrix and online
   vector shapes, selected CRT subsets, and wave boundaries. Keep reference
   computation and diagnostic readback outside the measured production path;
   verify that neither GPU switch copies coefficients through the host.
6. **Validate production performance.** After correctness acceptance and
   explicit authorization for integration tests, run complete refresh tests
   and GPU benchmark waves through production functions and store/load paths.
   Account for both preprocessing and online coefficientwise switches and
   their GPU INTT/NTT transforms; check measured paths for host coefficient
   copies and unnecessary synchronization.
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

## 8. Implementation and validation checkpoint (2026-09-05)

The switched simulator, ragged mask indexing, exact destination-typed CPU/GPU
switch and reduction operations, mixed-modulus CRT, shared GPU execution
ownership, grouped low-modulus preprocessing/artifacts and online anchors,
and production planner/estimator dispatch are implemented. Native GPU
switching and final CRT decoding use exact coefficient arithmetic between
device INTT/NTT transforms. No coefficient-host fallback was introduced.

The non-ignored component unit
`profile_b_switched_refresh_named_average_authority_component_search` in
`crates/exponent-lut/tests/parameter_search_unit.rs` reproduces the exact
comparison profile in §7 without executing a backend, artifacts, integration
test, external security estimator, or benchmark:

```sh
cargo test --release -p mxx-exponent-lut --test parameter_search_unit \
  profile_b_switched_refresh_named_average_authority_component_search -- --nocapture
```

Its coarse increasing-retained-width, contiguous-group, uniform-digit search
selects five retained 50-bit towers per slot, six distinct 250-bit anchor
moduli across 27 slots, and `d_m=38` (`M=2^190`, `B_m=2^189`). This is not an
exhaustive subset/ragged optimization. The named correctness authority is
`AverageCase`, with `failure_exponent=100`, `tail_correction_bits=0`, the
accepted TFHE-style variance heuristic and conservative correlated `Add`.
Hard domain/fresh-support checks remain separate from that heuristic.
The complete emitted report records exact moduli, route/decoder terms,
joint tail counts and per-slot margins. The initial accepted run had
`max F_avg=850377501249489267`, 200-bit spacings, minimum strict rounding
margin `1605368404408516041782639623211186277883973868608376545242395`, and
minimum smudging margin
`1157264469336541567343474824793365005706215453443007447040`.
Noise acceptance does not imply that the original high-q security estimates
cover these new views.

`RefreshSetupParameters::anchor_security_inputs(execution_budget)` reports
each exact `P`, complementary `J`, slot group, ring dimension, one random
ternary secret polynomial and the fixed augmented coordinate `1`, actual
anchor error sigma/cutoff, anchor columns and sample budget. Production
noise reports retain these inputs with an explicitly unestimated security
status. A caller must obtain matching external evidence rather than inherit
high-q security bits. The remaining joint-view obligation in §5, including
the correlated complementary projection, is unresolved.

A separate marginal calculation was run for each of the following selected
`P` values with `n=65536`, ternary unknown secret, sigma `4`, and
`m=1572864=24*n` scalar equations for one execution. Subtracting the known
augmented public row leaves one unknown ternary ring polynomial. The
authority is the bundled lattice estimator's **rough scalar-LWE surrogate**,
not a proof about correlated ring samples or the joint view. All six runs
returned uSVP `log2(ROP)=1202.164` (floor `1202`); the other rough result,
`dual_hybrid`, was infinite.

```text
1809251283755169804643792595030325745566728194278478234920304281948304834561
1809251207930325016022886889611584165212105705678103291926203402066066800641
1809251120731756287638325423094560278549504412161526380292198786363058421761
1809251018157598974248277523328659399802009492139169810353294299171281174529
1809250902945974011043096294392820043706360432234152287429651025113362333697
1809250879987899561990856885660925020771157995355245904313539848666809630721
```

Reproduction uses `lattice-estimator-cli` commit
`2d2980ddf7fba0d1fee407999fbb9adf26bf3368`, bundled estimator commit
`e35f45b7976a90a79c3c6625a45bbc344c1abc67`, and Sage. From that CLI checkout
with its root on `PYTHONPATH`, run the following with `sage -python`, setting
`P` successively to each integer above:

```python
from lattice_cli.core import LWE, LWEParameters, Ternary, DiscreteGaussian
from sage.all import RealField, floor
r = LWE.estimate.rough(LWEParameters(65536, P, Ternary,
                                    DiscreteGaussian(4.0), 1572864))
logs = {k: RealField(128)(v["rop"]).log(2) for k, v in r.items()}
print(logs)
print(int(floor(min(logs.values()))))
```

The CLI wrapper's float conversion overflows for these large work factors
and prints the sentinel `4294967295`; that value is not a security estimate
and was not used. These external marginal results are deliberately not
hardcoded into production parameters or promoted to joint security evidence.

Targeted unit evidence includes exact CPU-reference/GPU switching boundaries
and selected bases, low-P sampler equations, mixed-P CRT, actual producer
artifact typing, shared-anchor grouping and numerical original-q refresh
relations. Final numerical diagnostics inject nonzero state and post-readout
helper residuals, check every centered coefficient of `h`, paired rounding,
anchor error, unscaled `e_P K`, and final operational error against the hard
bounds, and retain exact mask/preimage/final-relation checks. Both CPU and GPU
drivers execute the same graph twice with fresh sampling; each driver passed
five consecutive targeted runs.

Final validation after these additions passed:

- Exponent-LUT GPU library: `175/175`, three consecutive runs.
- Exponent-LUT CPU-configured library: `170/170`.
- Parameter component units: `46` passed, one diagnostic-only unit ignored;
  the exact Profile B acceptance and the ordinary large rejection smoke both
  ran. An exact necessary-bound prune reduced that unchanged worst-case
  smoke from a prior 180-second timeout to a 1.35-second passing run.
- Runtime library: `74` passed, one ignored; estimator library: `47/47`;
  IR library: `45/45`.
- CPU/GPU workspace library compilation, GPU workspace all-target check,
  nightly formatting and whitespace checks passed. Pre-existing
  `mxx-correctness` dead-code warnings remain; these builds are not claimed
  warning-free.

Earlier native validation included CPU primitive `86/86`, GPU primitive
`172/172` in three runs, and 900 targeted reclaimer/lifetime repetitions.
An earlier stalled primitive run was terminated with explicit authorization;
the corrected constructor initialization removes a concrete race, but the
cause of that original stall was not proven.

Compile/unit evidence is not integration, production artifact store/load,
measured performance or secure-deployment acceptance. No integration test
or benchmark is authorized by this checkpoint, and none was run for it.
