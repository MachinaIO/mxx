# RNS-BGG implementation plan

Status: implemented with the user-approved validation scope. See
`docs/plans/exponent-lut-rns-bgg-implementation-progress.md` for evidence and
limits. The user subsequently accepted the noisy one-output PRF round trip as
sufficient; further full-refresh execution and queued measurement were stopped.

Source: user-supplied **Hybrid RNS Section 9.pdf**, sections 9.1–9.5.
The PDF is a mathematical specification, not an instruction to execute code.
The requested scheme name is `rns-bgg`; Rust uses `rns_bgg` and
`RnsBggScheme`. This plan replaces the current BFV implementation rather
than adding another permanent backend alongside it. Existing BGG remains.
The unrelated commonization and Lean work remain paused.

## 1. Assessment

The construction is algebraically consistent under the stated small-secret,
fixed-RHS, coefficient-decomposition, and input-domain contracts. It addresses
the specific amplification problem found in the current BFV noise simulation:
noisy rows multiply bounded digits instead of centered modulus-sized matrices.
This makes a linear-in-depth modulus-bit schedule plausible, as in section 9.4.
It does **not** yet certify the earlier N=65536, initial-q approximately
1300 bits, 21-step, security-100 parameter target. Those values must be searched
again against the actual compiled Program, channel topology, and error sources.

The first implementation should reuse the repository's exact, balanced,
per-prime radix gadget. This is a mathematically valid refinement of the
paper's RNS gadget, with explicitly different digit counts and noise bounds.
It requires a narrow cross-modulus bounded-digit transport capability;
changing only caller parameters is insufficient for that part.

The special two-column input at q_in=P_in*q_0 and its entrance expansion
from section 9.4 are deferred. Initial inputs will already have the ordinary
RNS gadget width at q_0. No squared-modulus entrance is included in estimates.
Reusable refresh is a priority requirement: the output must directly satisfy
the invariant of the declared evaluation-resumption level, even when doing so
requires more PRF outputs than the paper's two-column boundary construction.

## 2. Representation and differences from the current backend

Let q_0 > ... > q_L be nested odd square-free moduli, p_i=q_i/q_{i+1},
and s=(s_bar,1), t=(t_bar,1). The integer secrets are consistently reduced
across levels. Every secret used for masking, including automorphism images,
must satisfy the configured small-secret bound S.

For the first implementation, let the next modulus q_{i+1} contain h_i
CRT primes. Let b_i=2^a_i be the decomposition radix and

    kappa_i = ceil(max_prime_bits(q_{i+1}) / a_i),
    D_i = h_i * kappa_i,
    G_{i+1} = I_2 tensor (g_j * b_i^k)_(j,k).

Here g_j is the CRT idempotent of prime j in **q_{i+1}**, not in q_i.
Construct this gadget at q_{i+1}, then lift its constants to q_i.
The state invariant is

    A_i : 2 x 2D_i,
    c_i : 1 x 2D_i,
    c_i = s A_i - p_i mu_i t G_{i+1} + e_i  (mod q_i).

The row target of the final operation can instead have any configured width k.
For a two-column terminal output, k=2.

| Concern | Current BFV | Planned RNS-BGG |
| --- | --- | --- |
| Internal state | Fixed two-polynomial row | 2D_i columns, decreasing with remaining modulus |
| Internal public key | 2 x 2 blocks | 2 x 2D_i matrix |
| RHS | Centered 2 x 2 ciphertext lift | 2 x k ciphertext, then 2D_i x k bounded decomposition |
| Inherited-noise action | Modulus-sized RHS coefficients | Digit bound and actual number of digits |
| LUT alignment | Centered lifts of C and B | Decompose both C and B |
| ModDown | Exact coefficient scale-and-round | Same operation and rounding convention |
| Ordinary input | Two columns | Directly sampled gadget-width input at q_0 |
| Reusable refresh output | Currently two columns | Gadget-width output in the deferred-entrance phase |
| Final readout | Two-column row support | Retain arbitrary row targets, including two columns |

Use existing `Mat`, `Preimage`, and their schemas for variable-width values.
Do not emulate dynamic matrices with a growing collection of scalar blocks.
Retain the existing polynomial-block implementation for genuinely fixed 2 x 2
or 1 x 2 terminal arithmetic where useful. Do not add parallel scalar/family
type hierarchies or another encoding associated type.

## 3. Why the operations work

### Fixed private RHS

At q_{i+1}, freshly sample a ciphertext C with tC=F+eta. Prepare

    U = Dec_i(C),             G_{i+1} U = C  (mod q_{i+1}),
    A' = MS_i(A_i U),         c' = MS_i(c_i U).

The decomposition uses centered **coefficient** residues, split into balanced
radix digits for the reused implementation. Its output is lifted as bounded
integers to the input modulus before multiplication. Then

    c_i U = s A_i U - p_i mu_i F + e_i U - p_i mu_i eta  (mod q_i).

The decomposition discrepancy is a multiple of q_{i+1}; multiplication by
p_i makes it vanish modulo q_i. Thus

    c' = s A' - mu_i F + e'  (mod q_{i+1}).

For an intermediate output choose F=p_{i+1} nu r G_{i+2}. For a terminal
output choose F=Delta nu r, or any compatible row-valued table target.
The scale and public weights are included **before** sampling eta. Neither
the target scale nor the magnitude of the target gadget multiplies eta.

### LUT and mask alignment

For an automorphism branch sigma, use a helper

    h_sigma = s A_h - p_i sigma(s) G_{i+1} + e_h.

With an RHS under sigma(t), prepare

    U_sigma = Dec_i(C_sigma),
    B_sigma = MS_i(sigma(A_i) U_sigma),
    V_sigma = Dec_i(B_sigma),
    T_sigma = A_h V_sigma.

The online raw row is

    Z_sigma = h_sigma V_sigma + sigma(c_i) U_sigma.

Output MS_i(sum T_sigma), MS_i(sum Z_sigma). The un-cancelled mask term is
sigma(s)(sigma(A_i)U_sigma-p_i B_sigma); its divided rounding remainder
contributes at most S/2 per branch. Both inherited and helper errors multiply
bounded digits. The final two rounding errors occur once after aggregation.

Use the paper's subgroup trace table for 0<=z<W, W dividing N. Normalization,
payload-secret switching, and an optional RHS monomial sigma(X^(B y)) belong
inside the freshly encrypted row targets. Do not restore a multiply-then-LUT
sequence or scale existing RHS errors with target weights.

For two-input concatenation x+B y, check the declared domain against W=B^2;
do not infer that every modular PRF accumulator automatically has this domain.
The current native sparse-LWR profile instead uses signed monomial wrap with
Q=2*(N/stride). Its existing identity-product accumulation can remain if the
terminal table's anti-periodic extension is preserved. Verify that contract
against the existing native PRF clear oracle; do not silently replace it with
the PDF's illustrative nonnegative-domain reduction circuit. If explicit
reduction/known-shift operations are introduced, count their actual level drops.

### Addition and public multiplication

Keep existing same-level Add and public-polynomial multiplication: combine
both public matrices and encoding rows, and propagate the corresponding
plaintext norms and shared error sources. Operand level, gadget layout,
payload secret, and scale agree by preparation contract.

Section 9.2's known-value multiplication is a different operation: with
supplied c_x=sA_x-p_i x tG+e_x and c_y=tA_y-F_y+e_y, use

    U=Dec_i(A_y),
    A_xy=MS_i(A_x U),
    c_xy=MS_i(c_x U)+x c_y.

The error x e_y is **not** divided by p_i. Knowing x does not supply c_x.
Only use this path where those prepared operands already exist; retain the
existing ordinary public gate for public weighting at the same level.
No new Program primitive is required just to expose this formula.

## 4. Reusing the existing gadget implementation

Relevant existing code:

| Location | Reuse / limitation |
| --- | --- |
| `crates/primitives/native/ExactBasis.cc::exact_basis_matrix` | Exact CRT idempotent-times-radix gadget |
| `crates/primitives/src/matrix/dcrt_poly.rs::dcrt_decompose_poly` | Balanced per-prime digits; CPU implementation currently reconstructs big integers |
| `crates/primitives/src/matrix/gpu_dcrt_poly.rs::gadget_decompose` | Compact coefficient-domain digit buffer and existing GPU kernels |
| `crates/primitives/src/poly/dcrt/params.rs::validate_capability` | base_bits<=31 and base_bits<=crt_bits/2 |
| `crates/dsl/src/lib.rs::Ring::gadget`, `Mat::decompose` | Existing gadget/decomposition graph nodes and coefficient-bound metadata |
| `crates/dsl/src/lib.rs::Preimage::mul_small_rhs` | Existing bounded-RHS matrix multiplication |
| `crates/ir-core/src/validate.rs`, MatrixMulSmallRhs case | Requires identical input moduli |
| `crates/primitives/src/matrix/gpu_dcrt_poly.rs::multiply_small_rhs` | Also requires matching parameter/context ownership |

For each level, register the actual prime basis and radix in backend parameters.
Passing a different symbolic base to `Mat::decompose` alone does not change
the primitive's configured radix: runtime validates the request and dispatches
using the selected parameters. Reuse the existing ring/context lookup and
bind each actual modulus to its planned radix consistently.

Always use the complete exact gadget: `dropped_moduli=0` and the full digit
count for the selected low modulus. Existing approximate-gadget truncation is
not a substitute for changing levels; it introduces a residual absent from
the equations above. Likewise `small_decompose` is not the paper's Dec_i.
Do not decompose a lift of C under the full high-modulus gadget: that changes
the gadget identity, row count, and noise contract.

### The necessary transport extension

Current `Preimage` has no coefficient-preserving cross-modulus conversion,
and ordinary small-RHS multiplication rejects its low-modulus type against
a high-modulus matrix. Extend the existing bounded-matrix conversion support
so a decomposition at q_{i+1} can be used at q_i with its signed coefficients
and coefficient bound unchanged. Prefer extending the existing centered-
extension operation and `Preimage` API, not a new application-specific node,
matrix type, or decomposition algorithm.

On GPU, transport/rebind the compact signed digit storage into the destination
parameter/context ownership, reusing the existing expansion inside small-RHS
multiplication. Reuse storage only where device/context lifetime rules permit;
otherwise use the necessary device-side copy. A type cast or relabelled artifact
is not sufficient. Do not expand every digit permanently into every RNS limb,
copy coefficients to the CPU, or synchronize the device globally.

On CPU, preserve correctness using the existing matrix/basis conversion path
first. Its use of coefficient big integers is a real implementation overhead,
not a requirement of the paper. Keep ModDown on the existing exact RNS path;
do not introduce big-integer reconstruction there. A direct per-limb CPU
decomposition improvement is separable from implementing the scheme.

### Efficiency relative to the paper

The paper allows one centered residue per prime (or prime group). Existing
radix decomposition splits each prime into kappa_i digits. This is not merely
a naming difference:

| Quantity | Paper, one prime per digit | Reused radix gadget |
| --- | --- | --- |
| State columns | 2h_i | 2h_i*kappa_i |
| Internal row-product ring multiplications | 4h_i*h_{i+1} | 4D_i*D_{i+1} |
| Two-column terminal row-product multiplications | 4h_i | 4D_i |
| Digit coefficient cap | max prime / 2 | radix / 2 |
| Uniform column-action bound | N*h_i*max_prime | N*D_i*radix |

For even 50-bit primes and base_bits=25, kappa=2: state/RHS ciphertext widths
double, internal row-product terms grow by 4, and terminal row-product terms
double, relative to the paper with the same retained prime count. Compact
digit storage must be calculated from its actual word width, not assumed to
grow by 4 as well. Smaller digit bounds can allow smaller drops; therefore
these fixed-chain ratios are not end-to-end timing or total-work ratios.

At base_bits=floor(crt_bits/2), odd widths can require three digits; e.g.
51/25 gives kappa=3 and a ninefold internal product-term ratio. Prefer even
widths initially. Mixed-width bases use the implementation's common maximum
digit stride and can contain padding.

Multi-prime groups and a single 50-bit centered digit are not available just
by changing today's parameters. The base-width and arithmetic/storage limits
must not be bypassed. If the measured reuse penalty is unacceptable, extend
the existing primitive with a native centered-residue/grouped-digit mode in
a later phase, after comparing the changed noise/drop schedule. That phase
is distinct from the deferred two-column entrance optimization.

## 5. Noise evaluation and parameter planning

Replace the current q_{i+1}-sized multiplier in scheme noise functions with
the actual bounded-digit action. For the reused gadget, a uniform bound is
K_i=N*D_i*b_i. Use tighter public per-column gains when already available;
do not scan runtime ciphertexts to discover norms.

For a monomial fixed product:

    E_next <= K_i*E/p_i + E_eta + (S+1)/2.

For J aligned LUT branches aggregated before ModDown:

    E_next <= J*K_i*(E+E_h)/p_i + J*E_eta + J*S/2 + (S+1)/2.

Use exact rational or exact integer bounds for the true rounding remainder,
not a floating-point acceptance test. One-hot selection retains all selector
ciphertext errors. Public weighting of an existing selector ciphertext and
sampling a fresh ciphertext with a weighted plaintext are different error
contracts even when the selected plaintext is a monomial.

The existing average-case heuristics remain explicit. Derive their second-
moment transfers from the new bounded actions, including helper reuse,
nonzero means, internal/public ModDown remainders, and the single outer
rounding. Decomposition digits are functions of the prepared RHS/public B;
do not claim independent errors merely from their separate digit indices.
Run the actual execution Program through the common noise interpreter.
The old BFV simulation result and its modulus-sized-action recurrence do not
apply to this changed invariant.

Plan backward from terminal channel moduli, choosing actual prime groups,
radices and drop products together. `p_i >= 2 J_i K_i` is a simple sufficient
contraction condition, not an instruction to drop exactly one prime. Use the
simulator to choose the smallest valid depth near candidate boundaries.
Record initial q_0 before any LUT, per-level q_i, p_i, D_i, digit bounds,
and peak/transmitted storage. Check security at the actual largest modulus
and actual secret/error distributions, separately from failure and mask bits.

## 6. Refresh, terminal rows, and channel initialization

Reuse common PRF digit scheduling, raw readout accumulation, decoder subtraction,
per-channel rounding, and CRT recomposition. Apply the variable-width invariant
to targets and schemas; keep scheme arithmetic in `rns_bgg`, not in PRF or
readout orchestration.

For each channel M_t=H_t*pi_t, construct the combined target so that decoder
subtraction yields

    z_t = H_t*(s A_star - target(mu) + e_fresh) + mask_t + residual_t.

Require ||mask_t+residual_t||_infinity < H_t/2 before rounding to pi_t.
Remove independent evaluation errors before CRT. The same fresh-error labels
and sampler offsets are reused across channels; mask labels are distinct.
Insert beta^d and H_t*beta^d in fresh row targets. Determine mask and fresh
digit counts separately, including the changed decoder and aggregate noise.

The current direct secret-s decoder can be retained in online preparation
when s is available there. The paper's anchor/preimage decoder is an alternative,
not a mandatory new primitive: if used, bind its trapdoor to M_t, reuse existing
sampling, and include e_B*K_t in the residual. Never import a trapdoor or decoder
from an incompatible modulus by changing metadata.

### Consequence of deferring two-column input expansion

A two-column refreshed row is not yet a reusable RNS-gadget state. Multiplying
it by a large gadget after refresh would multiply its fresh noise and negate
the intended budget. In the first phase, reusable refresh therefore targets

    target(mu) = p_ref * mu * t * G_ref,
    output width k_ref = 2D_ref,

with a common k_ref-column A_star and a common k_ref-column fresh error.
Section 9.5's componentwise decoder/CRT algebra generalizes to this row width.
Standalone two-column final output remains supported, but its re-entry into
evaluation belongs to the deferred entrance-conversion phase.

For native r-bit digits, the first-phase reusable-refresh count is consequently

    N_PRF = N*k_ref*(sum_t ceil(ell_mask,t/r) + ceil(ell_fresh/r)).

For a genuine two-column terminal output k_ref=2, this specializes to the PDF's
formula. Internal D_i does not multiply that two-column formula, but D_ref
does multiply the reusable wide-output formula. Report this deliberate loss
of the boundary optimization, rather than advertising the paper's two-column
refresh count for a wide refreshed state.

The refresh output key, Q_ref, p_ref and G_ref must match the declared next
Program input. Returning a merely decodable row, requiring another unbudgeted
conversion, or returning an encoding of the BGG backend does not satisfy this
requirement. Allocate fresh-error labels for every one of the N*2D_ref output
coordinates, sharing each coordinate's fresh labels across channels only.
Mask labels remain distinct by channel and output coordinate. Solve digit
counts and aggregate noise together with this actual output width; multiplying
an old two-column count by D_ref without rerunning the mask/spacing fixed point
is insufficient. Count additional channel routing and stored terminal material
separately from the number of distinct PRF outputs.

### Channels must exist before their primes are lost

Treat reconstructed Q_ref and working M_t as different parameters. The common
setup owns the full channel list and makes all required data/PRF paths available
before dropping their needed prime factors. A single exhausted chain cannot
recreate missing channels by basis extension.

A concrete initialization option is a shared transition whose output Q contains
all channel starting bases, with a separate fresh RHS target for each channel:
F_t=p_(t,0)*mu_target*t*G_(t,1). After that transition, exact reduction from Q
to q_(t,0) preserves its deliberately chosen payload scale and the same small
error. Each channel then follows its own declared chain. This costs an actual
transition and per-channel target work, both included in planning. Alternatively
the caller supplies compatible per-channel initial encodings explicitly.
Do not silently reinterpret one gadget-width state as another channel's input.

## 7. Code organization and preparation ownership

Keep the requested five backend files:

| Planned file | Responsibility |
| --- | --- |
| `crates/exponent-lut/src/rns_bgg/mod.rs` | Parameters, invariant, level widths, trait binding, existing relation/schema types |
| `crates/exponent-lut/src/rns_bgg/public_key.rs` | Fresh RHS targets, U/V decompositions, public products, helper/public caches, artifact metadata |
| `crates/exponent-lut/src/rns_bgg/encoding.rs` | Input sampling and online row arithmetic, same-level linear gates, terminal/decoder arithmetic |
| `crates/exponent-lut/src/rns_bgg/noise.rs` | Gate moments/bounds, level planning and refresh budgets |
| `crates/exponent-lut/src/rns_bgg/gpu.rs` | GPU tests and genuinely scheme-specific GPU glue |

Secret-t and public-matrix-dependent preparation belongs in preprocess;
secret-s rows/helpers/decoders belong in online-eval preparation. Cache U and V
at the earliest phase where their actual RHS, table, label, and public inputs
are bound. Do not move online-dependent data into preprocess by assumption.
Online evaluation consumes prepared bounded digits and does not redo public
decomposition or public ModDown. It still performs encoding ModDown.

Maintain `ExponentLUTEncodingScheme` with the unified `Encoding`/`PublicKey`
associated types. The Program, PRF and readout compiler keep their existing
structural lowering, captures and source identities. Their needed changes are
shape/target binding through existing scheme contracts, not `if rns_bgg` branches.
Reuse existing level metadata and schemas rather than inventing a second
layout/family/cache framework. Preparation binds modulus, base, width, scale
and artifact identity once; avoid repeated runtime cryptographic validation.

Rename current BFV exports, test selectors and active documentation to
`rns-bgg` / `rns_bgg` / `RnsBgg*`. Do not retain compatibility aliases for the
old scheme or reinterpret old artifacts as the new relation. Historical BFV
measurement reports remain historical and must not be relabelled as results
for the new construction.

## 8. Implementation order and completion criteria

1. Bind the new parameter/invariant types and rename the backend. Keep existing
   polynomial blocks for fixed-size operations, but use matrices for wide states.
2. Connect existing low-ring exact gadget decomposition and bounded-digit
   transport to high-ring multiplication. Prove/test G*Dec(M)=M in the low ring,
   including reordered/mixed prime bases and signed digits.
3. Implement fixed products, aligned LUTs, and same-level linear operations.
   Add target normalization/weights before fresh RHS error sampling. Prepare
   all U/V caches once and preserve aggregation-before-ModDown.
4. Replace scheme gate noise transfers and the backward planner. Exercise the
   actual compiled sparse-LWR Program and trace its full selector error gains.
5. Bind channel initialization, variable-width refresh targets, native digit
   counts and decoder artifacts through common PRF/readout/refresh logic.
6. Validate GPU bounded-digit transport and variable matrix widths using existing
   primitives; preserve batched expansion, stream ordering, bounded memory and
   existing parallelism. Measure actual extra digit work rather than asserting
   a speedup from column counts alone.
7. Run the revised parameter simulation and security checks, then report accepted
   counts, initial modulus, per-level widths and measured/estimated work.

During implementation, run only targeted CPU unit checks once after a meaningful
change. Include tiny noiseless relation tests and a small nonzero-error case for
the new noise terms. At the end, run relevant GPU unit checks and the existing
`tests/lwr_prf.rs` and `tests/noise_refresh.rs` flows when authorized: actual
Program simulation, benchmark estimation, and PRF/refresh round trips with
artifact reopen and next-LUT continuation. Do not create a parallel test-support
framework or add ignored integration tests. No tests are run for this design task.
The refresh continuation test must consume the returned wide encoding directly
with its declared key and level, and check the result after at least one next
LUT. This is required, not an optional two-column readout test.

Acceptance requires the new invariant at every level, exact decomposition at
the intended low modulus, preserved coefficient bounds across transport, correct
terminal/channel scales, common fresh error after CRT, faithful shared-source
noise evaluation, and a reported performance comparison against the one-prime
digit construction. Algebraic correctness, modeled noise acceptance and
100-bit computational/joint-mask security remain separate claims.
