# Power-LUT Noise Models: Worst-Case Authority and Average-Case Estimates

## Status and scope

This specification defines a future extension of the Power-LUT-specific noise
simulator. It does not change the existing `WorstCase` protocol, its formulas,
or its acceptance authority. It is deliberately separate from the generic
`mxx-noise-simulator` graph evaluator. The proposed `AverageCase` channel is an
estimate unless an explicitly versioned authority policy permits acceptance.

The central invariant is compositional within the established TFHE,
OpenFHE, and TFHE-rs-style independence/variance heuristic envelope: every
wire carries a declared noise state, every primitive derives its exact
structural transfer from setup/action inputs, and every addition is classified
before propagation.
This remains heuristic acceptance evidence, not a formal proof.

## Modes and authority

```rust
enum NoiseModelKind { WorstCase, AverageCase }
enum NoiseMagnitude { Worst { hard_bound: BigUint }, Average { variance: BigRational } }
```

The simulator is single-channel: `WorstCase` computes only its hard bound and
`AverageCase` computes only variance. They must not silently substitute one for
the other. Lattice security remains harness-owned. Setup/identity/CRT validity,
mask-domain coverage, and strict fresh-error bounds remain mandatory production
gates; Average smudging and rounding are AverageCase authorities. Average acceptance requires
`allow_average_acceptance = true` and binds the complete failure budget, model
version, centering status, and heuristic ledger. The report exposes
`security_authority = harness`, `correctness_authority = AverageCase`,
`hard_authority_accepted`, `correctness_accepted`, and the paired `accepted`
result.

| Consumer | WorstCase | AverageCase by default | Explicit opt-in |
| --- | --- | --- | --- |
| lattice security | harness authority | not evaluated here | never probabilistic |
| setup/domain/fresh gates | authority | mandatory hard gates | never probabilistic |
| mask smudging and refresh correctness | authority | diagnostic only | allowed only with named heuristics and ε budget |
| refresh correctness | authority | diagnostic only | allowed only with named heuristics and ε budget |
| parameter search | authority | comparison/estimate | recorded estimate |
| report/snapshot | required | may coexist as non-authoritative | authority field required |

## Compositional state and structural transfers

An average proof requires H1: each state coefficient is zero mean, pairwise
uncorrelated, and has doubled-coordinate variance at most `V2=4V`. For a fixed right action `M`, the
exact variance transfer is

\[
 V2((eM)_{j,c})\le V2\,\gamma_2^2(M),\qquad
 \gamma_2^2(M)=\max_j\sum_i |M_{i,j}|^2.
\]

All variance arithmetic is exact in the squared domain; no square root,
floating-point CLT approximation, or saturating arithmetic is allowed.
Negacyclic wrapping only changes signs and therefore preserves the square sum.

```rust
struct VarianceTransfer { gain_sq: BigUint, additive_var: BigRational }
struct DualTransfer {
    worst: AffineNoiseTransfer,
    avg: VarianceTransfer,
    heuristics: Vec<HeuristicId>,
}
```

The simulator derives both gains from the public action shape and structural
model at the point of use; callers cannot inject an L2 gain or a digest.
Missing matrices may use the existing worst structural fallback
`F_beta = ell_beta * n * beta`; the average fallback
`F_beta^(2) = 2 * ell_beta * n * (beta^2 - 1) / 12` is explicitly heuristic
(H2), never an authority proof.

## Primitive transfers and addition invariant

Monomial actions have both gains one and no additive term. Fixed-RHS Fuse uses
`gamma`/`gamma_2^2` of the decomposition action plus one fresh helper term.
For a flat LUT of width `W_L`, independent branches add variance linearly:

\[
 V2' = W_L\Gamma_{C,L}^{(2)}V2 +
 4W_L(1+\Gamma_{A,L}^{(2)})\sigma_\chi^2.
\]

This is H3 and applies only when branch setup randomness is genuinely
independent. Sequential ClearCoeff branches share state/helper material and
must not receive a square-root reduction:

\[
 V2_{out}=W^2\max_k V2_{branch,k}.
\]

Accordingly the original average-estimate reductions are retained: independent
ring contributions use `sqrt(n)`, flat independent LUT branches use
`sqrt(W_L)`, and independent OneHot slots use `sqrt(r_j)` in standard-deviation
presentations. Distinct routed labels use square-root factors only after the
explicit mask/fresh topology counts are formed; there is no generic
`sqrt(N_coeff*d_p)` shortcut. These factors are represented as `n`, `W_L`,
`r_j`, and route-count multipliers in
the squared variance domain; they are not applied to shared-state branches.

The normative factor table is:

| contribution | standard-deviation estimate | authority rule |
| --- | --- | --- |
| ring accumulation | `sqrt(n)` | H1 |
| digit sum | `sqrt(2*ell_beta)` | H1 |
| centered digit amplitude | `beta/sqrt(12)` | H2 |
| independent flat branches | `sqrt(W)` | H3 |
| active OneHot slots | `sqrt(r)` | H4 |
| distinct PRF labels | topology-specific mask/fresh multipliers | H5 |
| shared sequential state branches | `W` | never weaken |
| grouped/round depth | original exponent | never weaken |

Any implementation that weakens a factor must log both the old and new factor
and the reason; it must be reviewed as a model change and cannot silently
change the WorstCase channel.

OneHot with `r_j` active slots uses `r_j` variance contributions (H4), while
the worst channel remains linear in `r_j`. The PRF plan is selection-only and
sequential: each bucket applies its selection transfer `S_j`, then each
intermediate reduction group applies exactly one `L_W`; the terminal range
applies exactly one terminal LUT. There is no independent per-bucket LUT
recurrence. For group `g` and terminal `T`:

\[
 V2_{g,out}=\gamma_2^2(L_W)V2_{g,in}+V2_{g,helper},\qquad
 V2_{T,out}=\gamma_2^2(L_T)V2_{T,in}+V2_{T,helper}.
\]

Selection and reduction ranges, `W`, inherited terms, additive components, and
growth are all logged. The depth exponent is unchanged in either mode. Every
IR addition is one of:

* `IndependentSum` (fresh independent randomness; variances add),
* `CoherentScale(k)` (same wire; squared gain `k^2`), or
* `SharedStateBranchSum(W)` (shared state; squared gain `W^2`).

An unclassified addition, including raw `z + phi(z)` without gadget mixing,
fails closed in `AverageCase`. There is no `V := E^2` escape hatch.

## Refresh, centering, and CRT safety

Average refresh requires setup inputs that directly establish centered
coordinates. Digits in `[0,p)` have
nonzero mean. For `p=2`, use doubled noise coordinates: represent every
coefficient as `y=2x`, subtract the integer offset `p-1=1` before routing, and
divide only in the final public decoding identity. Thus the centered digit has
integer support `{-1,+1}`, doubled variance `1`, and doubled hard amplitude `1`.
The scale and decoder factor are derived directly from the actual p=2 setup
inputs; no caller-supplied centering or decoder proof is accepted. Use one
doubled-coordinate convention throughout: `D2=p^d_m-1`, `V2=4V`, and
`s=q/q_t`. The strict deterministic precondition is `2s > 2D2`; the stochastic
test is `(2s-2D2)^2 > 4*z^2*4^tail*V2`. This introduces no one-bit loss. An uncentered artifact fails
closed. WorstCase remains unchanged; any centered worst-case improvement is a
separately reviewed protocol change.

Use separate mask and fresh-error digits `d_m` and `d_e`. For every CRT slot,
the fresh bound is strict per modulus: `B_e = p^d_e - 1 < q_t`; no minimum or
aggregate CRT modulus may replace this check. The exact refresh has two ell
columns (mask and fresh), not one merged count.

The hard authority is split into setup/identity/CRT validity, mask-domain
coverage, and the strict per-slot fresh predicate. Its composite result does
not include AverageCase smudging or rounding.

Split deterministic mask noise from stochastic noise:

\[
 D_{det}=B_m,
\quad V2_{stoch,t}=g_{k,t}^{(2)}V2_{state}
 +g_{m,t}^{(2)}V2_{mask}+g_{e,t}^{(2)}V2_{fresh}
 +4\sigma_\chi^2\gamma_2^2(K_t).
\]

The executable targets are literal and target-aware: state uses `kappa_t
beta^a`; mask uses `p^a`; fresh uses `kappa_t p^a`; decoder uses `K_t`.
Their l2-squared gains are `g_k2`, `g_m2`, `g_e2`, and `gamma_2^2(K_t)`;
there is no extra kappa factor on the mask target. The report binds each gain, label
count, mask/fresh digit count, and whether a factor is an event multiplicity
or a variance multiplier. No route count is inferred from a scalar total.

The routed PRF topology is explicit. Global event counts are
`slot_count*(2*ell_beta)*N_coeff*d_m` for mask and
`(2*ell_beta)*N_coeff*d_e` for shared fresh labels. For one output, however,
the variance sum contains only `N_coeff*d_m` mask labels and `N_coeff*d_e`
fresh labels, weighted by their per-digit target gains; the `2ell` columns and
slot count belong to event accounting and do not multiply one output's
variance. H5 may add distinct-label variances only when the artifact declares
the shared-helper correlation model.
The decoder fresh Gaussian term is exact under its stated Gaussian model.

WorstCase uses the unchanged strict test `2 E_pre,t < q/q_t` and requires
`q_t | q`. AverageCase uses the doubled-unit strict precondition `2s > 2D2`,
then compares exactly:

\[
 (2q/q_t-2D_{det})^2 >
 4z^2 4^{tail\_correction\_bits}V2_{stoch,t}.
\]

Equality fails. In AverageCase's single channel, reset stores only
`V2=4*(p^(2*d_e)-1)/12` plus doubled-coordinate scale metadata. A doubled
amplitude `p^d_e-1` may be retained as optional diagnostic data, but no
half-integral hard/variance pair is stored.

## Probability budget and heuristic ledger

`log2(N_events)` is the named sum of input-domain, coefficient/slot, and
inspection-event bounds. With failure exponent `lambda_fail`, use exact
`BigRational` upper arithmetic:

\[
 z^2=2(693148/1000000)(lambda_{fail}+log_2N_{events}+1).
\]

The report records `z_sq`, event factors, tail correction, mode, acceptance
authority, and a deduplicated ledger:

```rust
enum HeuristicId {
 H1StateUncorrelated, H2DigitUniformFallback,
 H3BranchSetupIndependence, H4SlotRhsIndependence,
 H5PrfLabelIndependence, H6GaussianTailClosure, ExactUnderGaussian,
}
```

H1--H5 are established industry heuristics, not proofs. The Gaussian tail
conversion is separately named H6, `GaussianTailClosure`, corresponding to the
subgaussian-closure/normal-approximation convention used by these libraries.
AverageCase opt-in may accept under H1--H6 with an explicit event union bound,
but the report must say that acceptance is heuristic and must include the
heuristic ledger and ε budget. `tail_correction_bits` is a reserved calibration
input, not evidence that repairs a violated model.

Outside that envelope the simulator fails closed: coherent same-wire sums,
shared-helper sequential branches, unknown addition provenance, uncentered
`p=2`, and identity mismatches cannot be accepted by AverageCase. This is a
conservative boundary on the heuristic model, not a requirement for a formal
covariance or subgaussian model.

## Reports, snapshots, and identity

Every gate report records its input/output magnitude, transfer gain, addition
class, model kind, and heuristic IDs. Refresh slot reports add
`deterministic_term`, `stochastic_variance`, `z_sq`, squared margin/deficit, and
`accepted_under`. Snapshots use a new schema version and retain the public
setup/model identity, `d_m/d_e`, event budget, model inputs, and the
setup-derived nonzero initial doubled variance; old schemas fail closed. No
secret support, schedule, or key material is serialized.

The grouped PRF report records plan `(q,p,B,k,W)`, intermediate group ranges,
terminal start/length, per-bucket selection inherited/additive terms, and each
group/terminal inherited, helper, gamma-A, total additive, output, and growth
values. This is diagnostic evidence, not a replacement for the compositional
structural model.

For the current profile, `B=h+3=34`; the canonical delayed-reduction plan is
`k=33`, `W=512`, since `nextPow2((k+1)(Q-1)+1)=nextPow2(34*15+1)=512`, with
one intermediate reduction group and a nonempty terminal range. This is within
`W <= 4096`; the intermediate group applies one `L_W`, followed by one terminal LUT. Any plan change must be reflected in the
immutable snapshot and rejected on identity mismatch.

For each slot, let `A_t = z_joint^2 * 4^tail * V2_t = N_t/R_t` and define
`Favg_t = ceil_sqrt(ceil_div(N_t, 4R_t))`; zero variance maps to zero. The
single checked `z_joint^2` is derived from one event count covering all
correctness, smudging, inspection, input-domain, and extra events. If
`D = mask_slot_count * (2*ell_beta) * N_coeff`, Average smudging accepts iff
`M_m >= 2^(lambda+1) * D * max_t(Favg_t)`; equality is accepted. The report
retains topology counts as evidence, the joint event count and epsilon, each
`Favg_t`, the maximum, and the signed smudging margin. The corresponding
masking-distance bound is `2^-lambda + epsilon_joint`.

Acceptance uses one immutable snapshot: setup/identity/CRT, mask-domain, and
strict fresh-error gates must pass, and explicitly enabled AverageCase
smudging and rounding must both pass. The average result cannot be paired with
a different parameter set or snapshot.

## Search evidence and ordering

For the current WorstCase search, the complete Phase-1 evidence is the single
measured tuple `(Q_L,p,nu,h)=(16,2,451,31)`, with all eight attacks finite,
estimator-source commit `53da5982597709ba0fdf94ea37a84d822310fd84`,
MATZOV/GSA, `m=infinity`, and measured minimum `100` bits (the conservative
floor of the measured real value). Earlier Q8/Q16/Q32 129-bit claims are
rejected diagnostics and are not qualifying profiles.

Search reports bind the complete finite tuple/grid, estimator commit/model,
induced error interval, raw entropy, tier policy, CRT/base grid, and ordering
metadata. WorstCase is always searched/validated first. Average results can be
reported for comparison but cannot change a default selection. If opt-in is
enabled, the report must include the exact probability space, event budget,
derived p=2 centering/decoder scale, heuristic ledger, and accepted authority.

The exact Phase-2 priority is: selected profile; largest `base_bits`; largest
`crt_bits`; smallest valid depth; smallest ring dimension. It enforces
`qbits < 2000`, `N <= 2^17`, and `base_bits <= floor(crt_bits/2)`. Each concrete
CRT/base pair is checked independently; an invalid pair does not invalidate
the whole mixed grid.

AverageCase candidate selection is setup-owned. The selector supplies the
canonical public PRF program, PBC layout, fixed `d_e`, candidate `d_m`, and
explicit opt-in configuration to `evaluate_average_candidate`. That operation
constructs the complete setup snapshot, derives its identity and setup-derived
initial variance, and returns the paired hard-authority/correctness report
without constructing a graph or accepting caller-provided variance. Final graph
assembly invokes the same snapshot builder and rejects any identity mismatch,
so a selected candidate is evaluated under exactly the immutable inputs that
will be reported and consumed.

## Tests and migration

Before enabling AverageCase, retain bit-exact WorstCase regression tests and
add exact toy tests for both gains, wraparound, fallback variance, addition
classification/coherent guards, z-squared monotonicity, strict CRT/fresh
boundaries, derived centering, two-column refresh, deterministic serialization,
setup-derived nonzero initial variance, and authority isolation. Property tests may compare a probabilistic estimate to a
worst bound for representative inputs, but such comparisons are never runtime
acceptance assertions.

Migration order: extend the structural model and schema; wire WorstCase through
the new single-channel enum with zero behavior change; implement compositional average
transfers and classifier; add centering/refresh formulas; add reports and
ledger; then enable opt-in tests. Do not alter generic noise simulation, reduce
WorstCase depth exponents, or silently weaken any existing reduction effect.
