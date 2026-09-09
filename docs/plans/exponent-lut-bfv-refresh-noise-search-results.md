# BFV refresh noise simulation, 2026-09-09

No accepted parameter set was obtained for the current 21-product PRF and
an initial encoding modulus of at most 1400 bits. This is a rejection by
the current conditional average-case noise model, not a measurement of
actual ciphertext error or an impossibility theorem for every two-column scheme.

## Execution and scope

The numerical integration test is
`crates/exponent-lut/tests/noise_refresh.rs::test_bfv_average_refresh_noise_simulation`.
It uses the production `SparseLwrPrfProgram::accumulation`, the common
Program noise interpreter, and `BfvParameters::plan_refresh`. It constructs
no runtime graph, ciphertext, PRF-label array, or GPU benchmark. The commonization
refactor remains paused. The baseline commit is
`3eab362d0579cb24a104a647116071dccf9ce4b0`, plus the integration-test changes.

Parameters are N=65536, native LWR Q=4096, digit modulus p=32, dimension
8192, sparse weight 18, and the paper PBC profile with 21 buckets. The layout
is sampled once per run; all active selector errors in each bucket contribute
to its composed RHS. CBD eta=2 supplies physical input variance 1. The
failure exponent is 100, with average acceptance enabled and no tail correction.
The exponent layout is stride 5, width 11.

For each prime pool, dynamic programming retains the least modeled variance
at each (product step, remaining CRT depth). The production transfer is affine
and increasing in input variance, so larger values at the same state cannot
improve the final variance. Initial prefixes exceeding 1400 bits are excluded.
All remaining prefixes are considered, and the retained paths are replayed
through the actual Program. Their exact rational output variances must match.

This search deliberately omits intermediate acceptance constraints and uses
only one tail event in its final PRF necessary-condition check. Both relaxations
favor acceptance. Consequently rejection excludes every prefix-chain allocation
within the declared pool. A successful necessary-condition check would still
require the full Program audit and refresh-wide event budget before acceptance.

The refresh planner is also invoked with two output CRT primes, one native
fresh digit of base 32, fresh bound 16, and optimistically fresh data noise at
the terminal PRF level. These are diagnostic assumptions, not an accepted
refresh configuration. It rejects every candidate with `refresh mask spacing`.

## Results

The companion CSV was generated from the seven test JSON reports and checked
against their successful test-execution logs. Test success means the numerical
search and trace-consistency assertions ran successfully; it does not mean
that a cryptographic parameter set passed.

| CRT bits | Base bits | Pool depth | Best path initial modulus bits | PRF output log2(q) | Final formal log2(stddev) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 25 | 26 | 1300 | 150.000 | 12657.500 |
| 50 | 25 | 28 | 1400 | 150.000 | 12557.500 |
| 40 | 20 | 35 | 1400 | 120.000 | 9797.497 |
| 32 | 16 | 45 | 1376 | 95.997 | 7612.875 |
| 28 | 14 | 50 | 1393 | 83.961 | 6535.859 |
| 26 | 13 | 53 | 1312 | 77.848 | 6014.247 |
| 25 | 12 | 38 | 902 | 74.861 | 6063.232 |

Actual prime products determine modulus sizes; nominal CRT width times depth
is not substituted. Odd CRT width uses integer division for `base_bits`.

The enormous final numbers are **unreduced modeled error envelopes propagated
past earlier failures**. They must not be interpreted as observed errors or
physical standard deviations of reduced ciphertext coefficients. The JSON
also reports the first optimistic acceptance failure and its local modulus,
scale, divisor, standard-deviation envelope, and tail bound.

For example, the best 40-bit path begins with depths 35 -> 23 -> 22. The first
transition has input log2(q) about 1400, divisor log2(p) about 480, output
log2(q) about 920, and next message-scale log2(p) about 40. Its modeled
log2(stddev) is already 447.5 and its single-event tail-bound logarithm is
451.065. The output-modulus check passes at this point; the message-scale
check fails. This is more informative than the final formal value 9797.497.

## Why changing CRT width alone does not fix this model

In `crates/exponent-lut/src/bfv/noise.rs::private_mul_noise_variance`, let
v_i denote the propagated variance envelope, q_i the input modulus, and
p_i=q_i/q_{i+1}. The inherited term is

    v_i * N * (q_{i+1}-1)^2 / (2*p_i^2).

This corresponds to the implemented row product followed by ModDown in
`crates/exponent-lut/src/bfv/encoding.rs::evaluate_mul_private`. The supplied
"二列二重スケール.pdf", equations (6.6)-(6.7), likewise contains inherited
error times the centered RHS size divided by p_i. Its sufficient chain
allocation rule (13.2) increases p_i with q_{i+1}.

A deliberately weak necessary bound shows the issue is not an unlucky CRT
allocation. Since N>=8 and q_{i+1}>=2, the existing recurrence gives

    sqrt(v_{i+1}) >= sqrt(v_i) * q_{i+1}^2 / q_i.

Here the inequality compares values of the **propagated envelope**, not a
lower bound on actual physical variance. Fresh CBD eta=2 and at least one
selector give v_i>=1 at every stage, including the initial input. Write
x_i=log2(q_i). Acceptance at the final PRF state requires sqrt(v_t)<q_t;
the actual q_t/8 and tail-factor condition is stricter. Propagating from
each intermediate state j therefore requires

    x_j > sum(x_k, k=j+1..t).

Backward substitution implies x_0 > 2^(t-1) * x_t. The terminal PRF state
must retain at least three CRT primes: two for the refresh output and at
least one dropped by terminal ModDown. Each NTT prime is at least 2N+1,
so x_t>3*log2(2N)=51. For t=21 this yields

    initial log2(q_0) > 51 * 2^20 = 53,477,376.

This is a necessary bound for acceptance under the existing envelope, already
using much weaker conditions than the implemented checker. It rules out the
1300-1400-bit target for this serial Program under that model, independently
of CRT width or prefix ordering. It is not a recommended modulus size.
Reducing prime width or tuning digit counts cannot resolve this gap. A change
to the evaluated construction or a substantially different justified noise
model would be a separate task; neither was implemented here.

## Security status

The existing pinned sparse-LWR evidence for (nu,h,Q,p)=(8192,18,4096,32)
is 103.18615790798486 classical bits, rounded down to 103. The local evidence
consistency test
`parameter_search_tests::sparse_lwr_tests::reviewed_primary_tuple_derives_reviewed_values`
passed. This rechecks the stored tuple and rounding convention; it is not a
new Sage estimator run. The evidence names estimator commit
`53da5982597709ba0fdf94ea37a84d822310fd84`, MATZOV/GSA, unlimited samples.

No complete BFV parameter set has been certified at 100 security bits. In
particular, the previous BGG Gaussian-error security envelope is not reused
for BFV CBD eta=2. Correctness already fails before a refresh digit schedule
can be accepted. Consequently the reference PRF-output count is not adopted,
and no replacement accepted count or benchmark estimate is reported.
The PDF also explicitly leaves the correlated PRF/error-mask joint-security
condition open in section 15. The reports keep `security_verified=false`.

## Reproduction

Build the requested integration-test file:

```sh
cargo test -r -p mxx-exponent-lut --features gpu --test noise_refresh --no-run
```

Run its numerical entry (no GPU execution):

```sh
MXX_EXPONENT_LUT_TEST_SCHEME=bfv \
MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL=average-case \
MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS=100 \
MXX_EXPONENT_LUT_REFRESH_CRT_BITS=40 \
MXX_EXPONENT_LUT_REFRESH_MAX_CRT_DEPTH=35 \
cargo test -r -p mxx-exponent-lut --features gpu --test noise_refresh \
  test_bfv_average_refresh_noise_simulation -- --exact --nocapture
```

Change CRT bits and maximum depth using the table. Optional
`MXX_EXPONENT_LUT_REFRESH_SEARCH_OUTPUT` writes the JSON report. The explicit
average-case entry always uses the average model; the existing benchmark
entry continues to select its model through the existing configuration.
