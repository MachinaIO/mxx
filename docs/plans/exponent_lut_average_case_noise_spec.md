# Exponent-LUT Noise Models: Worst-Case Authority and Average-Case Estimates

## Status and scope

This specification documents the implemented Exponent-LUT-specific noise
simulator. It does not change the existing `WorstCase` protocol, its formulas,
or its acceptance authority. It is deliberately separate from the generic
`mxx-noise-simulator` graph evaluator. The implemented `AverageCase` channel is
an estimate by default; an explicitly enabled authority policy is required for
acceptance.

The native online-input boundary is normative for this model: every `a_j[i]`
is determined and generated anew by each `online_eval`. Preprocessing and
cache artifacts neither generate, fix, store, nor derive `a_j[i]` or
`X^{a_j[i]}`. The online `BggEncodingWire` carries the resulting vector and
`Some(X^{a_j[i]})` plaintext relation, and `MulPublic` consumes that online
public input. Public labels use `(s,s)`; the private accumulator uses `(s,t)`.

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
`sqrt(W_L)` under H3. The selector family does not receive a square-root
reduction: each active member executes `MulPrivate` followed by `MulPublic`,
then the members are combined by the conservative `FamilyAggregation::Add`.
That transfer is derived by the interpreter from the primitive gate rows and
the active range, not from a selector-specific formula. Distinct routed labels
use topology-specific multipliers only after the explicit mask/fresh counts are
formed; there is no generic `sqrt(N_coeff*d_p)` shortcut.

The normative factor table is:

| contribution | standard-deviation estimate | authority rule |
| --- | --- | --- |
| ring accumulation | `sqrt(n)` | H1 |
| digit sum | `sqrt(2*ell_beta)` | H1 |
| centered digit amplitude | `beta/sqrt(12)` | H2 |
| independent flat branches | `sqrt(W)` | H3 |
| distinct PRF labels | topology-specific mask/fresh multipliers | H5 |
| selector family `FamilyAggregation::Add` | conservative active-member sum | structural transfer |

Any implementation that weakens a factor must log both the old and new factor
and the reason; it must be reviewed as a model change and cannot silently
change the WorstCase channel.

The native PRF selector is the structural family
`MulPrivate -> MulPublic -> FamilyAggregation::Add`; the family reduction
remains conservative and is accounted for exactly as an ordinary `Add` transfer
over the active range. A flat `Lut` is independent only under H3 and retains its
`sqrt(W_L)` estimate. An unclassified addition, including raw `z + phi(z)`
without gadget mixing, fails closed in `AverageCase`. There is no `V := E^2`
escape hatch.

## Refresh, centering, and CRT safety

Average refresh requires setup inputs that directly establish centered
coordinates. The reviewed native profiles use the complete even bases `P=16`
and `P=32`; digits are centered before routing rather than treated as an
uncentered `[0,P)` variable. For a complete centered base-`P` digit, the
doubled-coordinate variance is exactly
`V2 = (P^2 - 1) / 3`, with `V2=4V`. The scale and decoder factors are derived
from the concrete setup inputs; no caller-supplied centering or decoder proof is
accepted. Use `D2=P^d_m-1` and `s=q/q_t` throughout. The strict deterministic
precondition is `2s > 2D2`; the stochastic test is
`(2s-2D2)^2 > 4*z^2*4^tail*V2`. An uncentered artifact fails closed.
WorstCase remains unchanged; any centered worst-case improvement is a
separately reviewed protocol change.

Use separate mask and fresh-error digits `d_m` and `d_e`. For every CRT slot,
the fresh bound is strict per prime factor: for every prime `r | q_t`,
`B_e = P^d_e - 1 < r`; no composite or aggregate CRT modulus may replace this
check. The exact refresh has two ell
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
beta^a`; mask uses `P^a`; fresh uses `kappa_t P^a`; decoder uses `K_t`.
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

WorstCase uses the unchanged strict per-slot test
`B_m + 2 F_t < kappa_t`, where `B_m = P^d_m - 1`, `F_t` is the route bound
for slot `t`, and `kappa_t = q/q_t`; it also requires `q_t | q`. AverageCase
uses the doubled-unit strict precondition `2s > 2D2`, then compares exactly:

\[
 (2q/q_t-2D_{det})^2 >
 4z^2 4^{tail\_correction\_bits}V2_{stoch,t}.
\]

Equality fails. In AverageCase's single channel, reset stores only
`V2=4*(P^(2*d_e)-1)/12` plus doubled-coordinate scale metadata. A doubled
amplitude `P^d_e-1` may be retained as optional diagnostic data, but no
half-integral hard/variance pair is stored.

## Probability budget and heuristic ledger

`log2(N_events)` is derived from the setup-owned transcript topology: it counts
the inspected mask and shared fresh coordinates, including their coefficient,
slot, and P-adic digit counts. No caller-provided event multiplier is included.
With failure exponent `lambda_fail`, use exact
`BigRational` upper arithmetic:

\[
 z^2=2(693148/1000000)(lambda_{fail}+log_2N_{events}+1).
\]

The report records `z_sq`, event factors, tail correction, mode, and acceptance
authority. It may also carry a deduplicated diagnostic ledger:

```rust
enum HeuristicId {
 H1StateUncorrelated, H2DigitUniformFallback,
 H3BranchSetupIndependence, H4RhsIndependence,
 H5PrfLabelIndependence, H6GaussianTailClosure, ExactUnderGaussian,
}
```

H1--H5 are established industry heuristics, not proofs. The Gaussian tail
conversion is separately named H6, `GaussianTailClosure`, corresponding to the
subgaussian-closure/normal-approximation convention used by these libraries.
AverageCase opt-in may accept under the named H1--H6 assumptions with an
explicit event union bound; the report identifies the selected authority and
event budget and states that acceptance is heuristic. A heuristic ledger is
diagnostic and optional in the current report. `tail_correction_bits` is a
reserved calibration input, not evidence that repairs a violated model.

Outside that envelope the simulator fails closed: coherent same-wire sums,
unknown addition provenance, uncentered digits, and identity mismatches cannot
be accepted by AverageCase. This is a conservative boundary on the heuristic
model, not a requirement for a formal covariance or subgaussian model.

## Reports, snapshots, and identity

Reports retain the useful fields emitted by the current implementation:
primitive program steps carry inherited/fresh/output bounds and variances,
constituent values, and family body/member counts; refresh slot reports carry
variance, `Favg`, CRT spacing, deterministic mask bound, rounding margin and
deficit, fresh-error status, and the variance breakdown. The average report
binds snapshot identity, selected security/correctness authorities, event
count, `z_sq`, acceptance flags, and per-slot results. Optional heuristic
labels are diagnostic and may be empty. Snapshots retain the public
setup/model identity, `d_m/d_e`, event budget, model inputs, and the
setup-derived nonzero initial doubled variance; old schemas fail closed. No
secret support, schedule, or key material is serialized.

The PRF report records the primitive `MulPrivate`, `MulPublic`, and active-range
`FamilyAggregation::Add` rows emitted by the shared interpreter. Flat LUT branch
counts and their H3 independence assumption remain separate report data. Any
program-shape change must be reflected in the immutable snapshot and rejected
on identity mismatch.

For each slot, let `A_t = z_joint^2 * 4^tail * V2_t = N_t/R_t` and define
`Favg_t = ceil_sqrt(ceil_div(N_t, 4R_t))`; zero variance maps to zero. The
single checked `z_joint^2` is derived from the setup-owned event count for
the inspected mask and shared fresh coordinates. If
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

For the current native WorstCase search, the reviewed Phase-1 evidence is the
ordered tuple grid
`(Q_L,P,nu,h) = (2048,16,16384,17), (4096,32,16384,17),
(4096,32,8192,18)`. Their pinned conservative security floors are respectively
100, 100, and 103 bits. The evidence uses estimator-source commit
`53da5982597709ba0fdf94ea37a84d822310fd84`, MATZOV/GSA, and `m=infinity`.
The grid is native and non-binary; no earlier binary-profile estimate is a
qualifying replacement.

Search reports bind the complete finite tuple/grid, estimator commit/model,
induced error interval, raw entropy, tier policy, CRT/base grid, and ordering
metadata. The selected `NoiseSearchModel` (`WorstCase` or `AverageCase`) is
evaluated, and its authority and acceptance are recorded. AverageCase opt-in
may select a candidate when enabled; it is not restricted to comparison-only
output. If opt-in is enabled, the report includes the exact probability space,
event budget, derived base-`P` centering/decoder scale, and accepted authority;
any heuristic ledger remains optional diagnostic data.

The exact Phase-2 objective exhausts every qualified Phase-1 profile and selects
the lexicographic minimum key
`(crt_depth, log_ring_dimension, -base_bits, -crt_bits, Q_L, P, nu, h)`.
The reviewed search bounds the encoding ring dimension by `N <= 2^16` and uses
the default CRT/base pair `(crt_bits, base_bits) = (50, 25)`. Explicit CRT/base
pairs are singleton search inputs governed only by backend capability; there is
no additional base-bit ratio policy or hard aggregate modulus-bit cap here.
An invalid pair does not invalidate the whole mixed grid, and a first
successful profile does not terminate global profile selection.

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
