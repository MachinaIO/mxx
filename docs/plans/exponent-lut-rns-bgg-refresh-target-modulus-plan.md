# RNS-BGG refresh to a reusable target modulus

Status: proposal only. This document does not authorize implementation or test
execution. It corrects the target-modulus interpretation in the preceding
RNS-BGG migration. Source: the supplied *Hybrid RNS Section 9.pdf*, especially
sections 9.2–9.5. The PDF is a specification, not an operational instruction.

## 1. Required behavior and the initialization boundary

Refresh must return the canonical encoding for the declared resumption point,
including its modulus, public key, scale, gadget, width, and secret identities.
A row at a smaller modulus that permits one more gate is insufficient.
For cyclic use, restore the original resumption modulus Q_star and its complete
encoding invariant after each round of computation.

The present implementation binds the refresh working modulus to the level
immediately after the PRF. All channels use that same working modulus, extract
one of its primes, and reconstruct it. Consequently it reduces error but does
not restore consumed evaluation depth.

Separating a PRF auxiliary chain from Q_star is necessary, but is not sufficient.
The data carrying the unknown message must also reach each required channel.
An arbitrary already-spent scalar encoding cannot be converted to a higher
modulus simply by extending its coefficients. Existing exact RNS gate formulas
do not provide that conversion. This plan therefore maintains compatible data
channels from before their required residues are dropped, as required by
section 9.5. The caller must adopt this scheduling boundary. Supporting an
arbitrary exhausted scalar without those channels would require a separate
bootstrapping/decryption-circuit construction, outside this plan.

The public builder's scalar gate syntax remains unchanged. The computation
interval between refreshes is supplied once as a Program, but its refresh-aware
execution driver runs the necessary data channels from the boundary. This is
an explicit change to preparation/execution scheduling, not a transparent
replacement for `refresh(spent_scalar)`. Refresh still returns the canonical
scalar Encoding; the next interval repeats the prepared fan-out.

The initial and refreshed boundary encodings remain ordinary wide RNS-BGG
encodings. The special squared-modulus two-column entrance stays deferred.

## 2. Notation and invariant

All rings use the same polynomial dimension N. Moduli are products of distinct
odd NTT primes. The bounded integer secrets s=(s_bar,1) and t=(t_bar,1) are
represented consistently in every required ring.

The target is fixed before selecting any refresh working modulus:

    Q_star = product_j r_j,
    Q_next divides Q_star,
    p_star = Q_star / Q_next,
    G_star = the ordinary gadget constructed at Q_next and lifted to Q_star,
    k_star = 2 * D_star,
    T_star(mu) = p_star * mu * t * G_star,
    c_star = s*A_star - T_star(mu) + e_star  (mod Q_star).

A_star has shape 2 by k_star. The refreshed error e_star is small in integer
coefficient coordinates, with one shared value across all CRT channels.
The output uses this exact key and resumption contract, not a key inferred from
the final PRF level. Any supported public payload norm bounds remain explicit.

For channel j define independently:

    M_j = H_j * r_j,    gcd(H_j, r_j) = 1,
    q_(j,0) > ... > q_(j,L_j) = M_j,
    p_(j,i) = q_(j,i) / q_(j,i+1).

M_j is a working modulus, not Q_star. In particular H_j is M_j/r_j, and is NOT
assumed to equal Q_star/r_j. Each chain is nested, and every level retains r_j.
The final rounded channel output alone has modulus r_j. Working moduli for the
data and PRF paths may differ before they meet at M_j.

## 3. Channel-specific drops

For each output prime r_j:

1. Pin r_j in every modulus on its channel. Construct M_j from that prime and
   enough other primes to satisfy the channel's joint masking/rounding budget.
2. Add the primes required by preceding gate drops to form the chain's initial
   basis. Respect the actual data availability constraints in section 4;
   arbitrary new primes cannot be added to a spent data encoding.
3. Choose each dropped set from the current basis excluding r_j. Require the
   drop product to satisfy the actual gate error recurrence and output scale.
   Do not assume that one prime per gate is sufficient.
4. Construct the gadget and decomposition at the exact next modulus, using its
   radix and digit count. A different retained-prime set changes the gadget,
   even if the level number, bit length and number of columns are equal.
5. At terminal evaluation, drop the scheduled product to M_j. After decoder
   subtraction, drop H_j and retain only r_j. CRT recomposition then targets
   Q_star, which is a separate operation from either drop.

Use explicit ordered prime lists. Local level indices identify positions within
one channel; they are not global modulus identities. Runtime operations reuse
existing exact ModDown, bounded decomposition/transport and matrix products.
Changing a prime ordering must not silently change the intended next basis.

A symbolic example illustrates the different drop sets (it is not a numerical
security/correctness parameter set):

    Q_star = a*b*c*d*e*f*g, p_star = a, Q_next = b*c*d*e*f*g.

After the common boundary fan-out described below:

| Channel | Initial data modulus | First drop | Second drop | Terminal drop | M_j | Final drop |
| --- | --- | --- | --- | --- | --- | --- |
| b | b*c*d*e*f*g | c | d | e | b*f*g | f*g |
| c | b*c*d*e*f*g | b | d | e | c*f*g | f*g |

Every other message-bearing channel receives its own protected-prime schedule.
Here channel a has a zero data target because p_star is divisible by a; it
still needs the correct mask-secret term and the shared fresh error.
Actual schedules may drop several primes at each step and use unequal H_j.

For concrete search, build each chain backwards from M_j. At each operation,
choose a nonempty product P of currently unused allowed primes, then set
q_in = P*q_out. On data paths the primes must come from the available boundary
basis; on separately prepared PRF paths the planner may allocate auxiliary
primes. Replay forwards to verify all intermediate contracts. For a monomial
input and J aligned branches, section 9.3 supplies the conservative recurrence

    E_out <= J*K/p * (E_in + E_helper)
             + J*E_rhs + J*S/2 + (S+1)/2,
    K = N*D*b,    p = q_in/q_out.

Here D and radix b come from q_out's actual decomposition parameters, and S
bounds the masking-secret row action. For example p >= 2*J*K contracts the
inherited-plus-helper term by at least one half; it does not by itself bound
the fresh or rounding terms. Use existing gate-specific formulas for direct
products, composed selectors and non-monomial payloads. Average-case search
uses their existing moment transfers, not an invented independence shortcut.
Store the actual dropped prime lists and resulting error/scale margins.

## 4. Data availability and a repeatable boundary fan-out

The boundary state contains p_star*mu*t*G_star, so its residue at primes dividing
p_star has no message term. This matters: preserving every output prime in a
message-carrying data chain is unnecessarily strong, but ignoring primes where
the target message term is nonzero is incorrect.

### 4.1 Message-bearing output primes

For each r_j dividing Q_next, branch the canonical boundary encoding BEFORE
consuming the primes needed by that channel. Use a real prepared identity LUT
(or the existing aligned payload-switch operation), with the source gadget
G_star and drop p_star. For a desired channel start q_(j,0) dividing Q_next,
prepare a fresh target at Q_next that reduces to

    F_j = p_(j,0) * mu * t_j * G_(j,1)  (mod q_(j,0)).

The identity LUT's table contains this target as a function of its input
exponent. The public scale and target gadget belong in fresh RHS plaintexts.
Use the existing native signed-exponent identity-table contract and its
normalization; do not silently substitute the paper's illustrative nonnegative
LUT domain. After the real Q_star -> Q_next operation, exact reduction to
q_(j,0) gives that channel's correctly scaled state. If q_(j,0)=Q_next, no extra reduction
is needed. Reduction preserves the relation and existing error; its source
error must already fit the destination budget.

Merely reducing the original encoding or changing its level metadata does not
perform this conversion. Its scale and gadget usually differ from the desired
channel invariant. Count the fan-out operation, its errors, caches and work.
Each channel subsequently evaluates the same required data Program through the
common interpreter, retaining its protected r_j until terminal data routing.
Do not regenerate an unknown data plaintext using a sampler.

For a fixed Q_star, sufficiently long programs may leave too few primes for
M_j. Reject that schedule, shorten the interval between refreshes, or choose a
larger boundary modulus and recheck security. An auxiliary PRF modulus alone
cannot repair this data-path shortage.

### 4.2 Output primes with zero data target

For r_j dividing p_star, T_star(mu) is zero modulo r_j for every mu. The channel
can omit the data term and generate only the mask-secret term and shared fresh
error. Omitting data is justified by this exact public identity, not by a claim
that a lost unknown residue can be restored. Keep all k_star output coordinates.
Other identically-zero target components may use the same public-zero rule.

### 4.3 PRF channel inputs

Prepare PRF inputs natively in each required auxiliary chain using the existing
input/selector preparation workflow. Use the same logical PRF key and labels
across channels; no evaluated PRF encoding is moved into an unrelated modulus
by metadata or arbitrary coefficient lifting. Fresh-error labels are identical
across channels so their plaintext outputs agree; evaluation errors need not
agree and will be removed by channel rounding.

PRF channels may use auxiliary primes outside Q_star. Their largest modulus,
secrets, errors and sample counts participate in security estimation. Share a
PRF prefix only when its complete encoding contract matches, and use a real
branch transition before divergence. Initial implementation may evaluate
channel paths separately using the same common evaluator; report that cost.

### 4.4 Next cycle

CRT returns the canonical scalar Encoding at Q_star. The next computation cycle
uses the same prepared boundary fan-out and channel Programs. Fresh noise is
budgeted against that fan-out. The runtime does not claim to rewind a chain
merely by assigning the returned row an earlier level number.

## 5. Terminal and CRT correctness

Reduce target residues from Q_star to r_j before representing them in M_j.
Q_star need not divide M_j. For any polynomial/matrix entry x modulo Q_star,
choose its r_j residue x_j and embed it as H_j*x_j modulo M_j. This embedding
is independent of the integer representative because H_j*r_j=M_j. Do not
apply containing-basis CenteredExtend to unrelated working/target bases.

At M_j, terminal data/PRF rows and public affine offsets must produce

    v_j = s*A_sum,j - H_j*T_star(mu)_j + m_j + H_j*e_star + f_j^(0).

The target has width k_star, independently of local internal gadget widths.
Sample weighted/normalized RHS targets freshly. Mask digit d has weight beta^d;
fresh-error digit d has weight H_j*beta^d. Do not multiply an already evaluated
noisy digit row by these weights. Aggregate raw data, mask and fresh terms
before each applicable outer ModDown.

Preprocess the combined public decoder target, and prepare its s-dependent
row online at M_j:

    d_j = s*(A_sum,j - H_j*A_star,j) + e_dec,j,
    z_j = v_j - d_j
        = H_j*(s*A_star,j - T_star(mu)_j + e_star) + m_j + f_j,
    f_j = f_j^(0) - e_dec,j.

With centered coefficient representatives, require

    ||m_j + f_j||_infinity < H_j/2.

Then round/divide by H_j modulo r_j. CRT of these exact residues returns
s*A_star - T_star(mu) + e_star modulo Q_star. Evaluation residuals are removed
BEFORE CRT; independently small channel errors do not become one small error.
The same integer s is used at every modulus; no uniform residue is reinterpreted
as a small secret in another ring.

## 6. Digit counts, noise, parameter search and security

### Independent acceptance parameters

| Parameter | Meaning | Requirement in this plan |
| --- | --- | --- |
| lambda_comp | Estimated classical attack work is at least 2^lambda_comp | Target 100 bits, as requested; record estimator/distribution assumptions |
| lambda_fail | Modeled probability of any correctness failure in the declared execution scope is at most 2^(-lambda_fail) | Separate configurable target; do not infer 100 from lambda_comp |
| lambda_mask | Statistical distance assigned to the joint masking argument is at most 2^(-lambda_mask), under its stated assumptions | Separate configurable target and joint accounting |

The phrase "100-bit failure budget" must not be used as a synonym for 100-bit
computational security. If lambda_fail=100 is selected explicitly, the modeled
failure probability target is about 7.9e-31. For K checked events, an equal
union-bound allocation assigns at most 2^(-lambda_fail)/K per event. The union
bound itself does not require independence; the average-case moment/tail
estimates may rely on the documented heuristics.

Define whether the correctness scope is one refresh interval or a fixed number
R of repeated intervals. For an R-interval lifetime target, budget over those
R intervals as well. A one-interval probability is not an unlimited-lifetime
guarantee. Keep the earlier failure/mask settings explicit in search reports;
comparative runs at 100 for either parameter must be labeled as separate
assumptions, not consequences of the user's computational-security request.

### Counts and search

Set beta=2^r. For k_star output polynomials, N coefficients, fresh digit count
d_e, and channel mask counts d_m,j:

    distinct PRF outputs = N*k_star*(sum_j d_m,j + d_e),
    channel digit uses  = N*k_star*(sum_j d_m,j + h*d_e).

Without reusable evaluated prefixes, the latter also counts the per-channel
PRF evaluations needed for these digits. Report logical labels, executions,
terminal work and persistent storage separately. Do not multiply fresh label
count by h, and do not omit the repeated execution cost.

Solve the mask-support/noise fixed point with the existing digit distributions
and public offsets. Choose the smallest H_j supported by available primes that
passes both masking and rounding requirements. This is a per-channel choice;
changing H_j or digit counts changes target work and exposed-noise bounds.
Track every selector ciphertext error, input/fan-out error, helper error,
terminal error and decoder error through the SAME data/PRF Programs used by
execution. Preserve the documented average-case heuristics and shared-source
identities, including fresh labels shared across channels. Use joint event
counts covering actual channel widths, active members and all observed outputs.

Search procedure:

1. Fix the target resumption contract and the requested data Program interval.
2. Derive active/zero target channels and legal fan-out/data starting bases.
3. Search channel drop sets and H_j with real gadget digit counts. Replay each
   candidate through the common noise evaluator, including intermediate scales.
4. Build compatible auxiliary PRF paths; iterate counts and budgets to a fixed
   point. Check refreshed e_star against the next cycle's fan-out and Program.
5. Check LWE/RLWE security at every distinct required ring/distribution, including
   auxiliary starts and helpers, and check the chosen LWR tuple separately.
   The earlier Gaussian estimate is not evidence for CBD eta=2.
6. Report a candidate only after all conditions pass. Separate computational
   security, correctness failure exponent and mask statistical security.
   Related-secret/decoder/joint-mask assumptions remain explicitly identified;
   the CRT algebra is not their security proof.

The user's earlier baseline is N=65536 and LWR (nu,h,Q,p)=(8192,18,4096,32),
with 21 PRF products. It is a search starting point, not a certified result for
this revised channel construction.

## 7. Concrete code changes and ownership

| Location | Change |
| --- | --- |
| `crates/exponent-lut/src/rns_bgg/mod.rs` | Separate target key/resumption metadata from channel working modulus. Extend the existing refresh level/descriptor with explicit working-basis identity; stop resolving it from target level+1. Retain existing Blocks, Encoding and PublicKey carriers. |
| `crates/exponent-lut/src/rns_bgg/public_key.rs` | Prepare identity fan-out, channel U/V, public aggregate/decoder targets, residue embeddings and CRT idempotents at their declared rings. Persist prime order, radix, target key identity and channel binding. |
| `crates/exponent-lut/src/rns_bgg/encoding.rs` | Sample fresh targets from the declared output resumption contract, not bank.level+1's gadget/scale. Evaluate ordinary scalar operations inside channel/Family loops. Create s-dependent helpers and decoders at each working modulus online. |
| `crates/exponent-lut/src/rns_bgg/noise.rs` | Remove the assumption that data and PRF share a source level and refresh targets the next level. Trace their separate paths to each M_j, include fan-out, solve per-channel support/drop budgets, and account for k_star. |
| `crates/exponent-lut/src/noise_refresh/refresh.rs` | Change the existing channel records/orchestration to receive channel-local data, digits, bank and evaluation context. Retain common label routing, raw reduction, decoder order and CRT scheduling. Remove the single shared data/digits/bank restriction. |
| `crates/exponent-lut/src/noise_refresh/prf.rs` and `program.rs` | Reuse the existing Program and prepared-evaluation traversal for each supplied context. No RNS-BGG formulas or second scalar/Family evaluator. |
| `crates/exponent-lut/src/rns_bgg/gpu.rs` | Register exact channel/target rings and use existing kernels. Preserve per-level radix and same-device bounded transport. No new GPU arithmetic is planned. |
| `crates/exponent-lut/tests/noise_refresh.rs` | Replace the smaller-modulus acceptance criterion with explicit target restoration and channel-aware simulation/reporting. Keep the existing test file and setup style. |

Heterogeneous channels can use a host-level loop over existing scheme/context
instances and existing ProgramValue/Family carriers. Digit/member parallelism
stays in the shared DSL. Do not add a second cryptographic associated-type
hierarchy or a new global scheme registry merely for dispatch. Extend existing
preparation entrypoints instead of retaining legacy fallback APIs.

Validate basis containment/divisibility, scales, shapes and identities once in
setup/import binding. Operators rely on that contract; no ciphertext scans,
BigInteger reconstruction, or repeated per-gate defensive checks are added.
Reuse exact coefficient ModDown and the existing radix gadget with actual next
modulus/base parameters. Account for the radix refinement's additional columns
and products relative to one centered digit per prime.

## 8. Implementation order and validation boundary

1. Introduce the explicit target/channel binding in existing metadata and change
   parameter simulation first, so the old next-level result cannot be accepted.
2. Implement and validate the real boundary identity fan-out and zero-data
   channels using existing scalar target preparation. Bind channel Programs.
3. Connect per-channel PRF inputs, terminal targets, decoder preparation and CRT
   to the common orchestration. Update artifact schemas without legacy aliases.
4. Update noise tracing, output counts and search using the same Programs and
   channel bindings. Record prime/drop lists and source availability.
5. Add narrowly scoped assertions to existing tests: two different protected
   primes/drop orders; exact target key/modulus/scale/gadget after CRT; identical
   fresh-error residues; all output columns; artifact reopen; and refreshed
   output entering the SAME boundary fan-out again. A small noiseless two-cycle
   test distinguishes depth restoration from one-more-gate continuation.

This document schedules no execution. Preserve the user's validation limit:
the existing noisy one-output PRF round trip is sufficient; do not automatically
launch another large full-refresh round trip or benchmark. During any separately
authorized implementation, use minimal CPU checks and final targeted GPU checks.
A result must state which of the proposed structural/two-cycle checks actually
ran, rather than treating the existing smaller-modulus test as proof.

Completion requires a self-consistent repeated-cycle construction, legal data
availability for every nonzero target residue, per-output protected-prime drops,
and restoration of the declared target invariant. If source data channels were
not supplied before their residues were lost, reject the unsupported request
explicitly instead of producing a lower-modulus substitute.
