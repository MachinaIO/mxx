# Exponent-LUT integration tests

Each GPU integration entry runs its simulation and subsequent stages in one
invocation, following `crates/we/tests/test_gpu_diamond_we.rs`.

| File | Sequential stages |
| --- | --- |
| `noise_refresh.rs` | Noise simulation / parameter search, then benchmark estimation of preprocessing and online refresh |
| `lwr_prf.rs` | Noise simulation/report validation, exact three-stage estimates, one three-stage GPU round trip, and strict digit readout |

`noise_refresh.rs` supports `MXX_EXPONENT_LUT_TEST_SCHEME=bgg` (default) or
`rns-bgg`. `lwr_prf.rs` is the RNS-BGG facade workflow. Both integration files
use `MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL`:
`average-case` (default) or `worst-case`. This selects the acceptance bound,
not just the printed report. `MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS` sets the
average-case failure exponent (small PRF default: 1; refresh default: 100).
The selector belongs to the test harness. Both implementations use
`PreparedSparseLwrPrf` and its executed Program; RNS-BGG refresh uses the common
`PreparedDirectRefresh` orchestration.

```sh
MXX_EXPONENT_LUT_TEST_SCHEME=rns-bgg cargo test -r -p mxx-exponent-lut --features gpu --test noise_refresh test_gpu_exponent_lut_benchmark_estimate -- --nocapture
cargo test -r -p mxx-exponent-lut --features gpu --test lwr_prf test_gpu_lwr_prf_one_output_round_trip -- --nocapture
```

Run these commands in a GPU-visible environment. Runtime execution requires the
`gpu` feature. Simulation constructs noise bounds on the host before GPU work;
the default fixtures do not require an intermediate plan file.

To measure an already accepted RNS-BGG parameter search, pass its report:

```sh
MXX_EXPONENT_LUT_TEST_SCHEME=rns-bgg \
MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL=average-case \
MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT="$(pwd)/docs/plans/exponent-lut-rns-bgg-refresh-security100.json" \
cargo test -r -p mxx-exponent-lut --features gpu --test noise_refresh \
  test_gpu_exponent_lut_benchmark_estimate -- --exact --nocapture
```

This uses the report's exact CRT primes, PRF block sizes, and mask/fresh digit
counts. Public monomial labels and trusted one-hot bits remain structural setup
inputs, as in the BGG estimator; hash-to-LWR-vector generation is outside this
GPU estimate. The `lwr_prf.rs` workflow always performs noise/report
validation, exact preprocess/encrypt/online graph estimates, and then the full
three-stage round trip. No stage-selection environment variable is supported,
so a partial estimate cannot be mistaken for a complete run. No PRF output
family is materialized on the host for this input binding.

Without an accepted report, RNS-BGG uses a small correctness fixture (one two-member selector block, base-4
digits, and a variable-width gadget encoding). `MXX_PRIMITIVE_TEST_RING_DIMENSION` controls
its ring dimension; `MXX_EXPONENT_LUT_REFRESH_MASK_STATISTICAL_SECURITY_BITS`
controls mask coverage. Parameter selection checks the PRF, the full refresh
mask fixed point, and the continuation level before constructing graph values.
The separate mask/fresh digit counts cover every output polynomial coordinate;
fresh digits are reused across CRT channels. RNS-BGG benchmark graphs exclude correctness
oracles and the post-refresh continuation gate.

BGG PRF parameters use the existing `MXX_EXPONENT_LUT_REFRESH_*` overrides and
small defaults. The PRF test checks arithmetic and noise bounds; it does not
require an accepted refresh-search report. These small PRF/RNS-BGG fixtures do not
certify lattice security.

The BGG refresh benchmark runs parameter search before constructing the graphs
and consumes the resulting report directly. Noise-only search is separate from
explicit Lean certificate verification. Average reports remain conditional:
RNS-BGG uses bounded low-ring gadget digits and the documented conditional
coefficient, fresh-target/branch decorrelation,
rounding-residue and Gaussian-tail heuristics. Its terminal moment transfer sums
all internal terms and includes outer ModDown once. Decoder CBD variance is
added under the residual-independence model; digit counts are solved again using
the selected bound and the current digit schedule's event budget. Worst-case
mode continues to use deterministic support and spacing inequalities.
The search's existing checkpoint/output overrides remain available; external
estimator inputs are needed only for tuples without reviewed estimator evidence.
Lean certification remains separate from this average-case benchmark flow.

The two integration files contain their own setup and execution code; there is
no `support/` directory or `mod gpu` wrapper. Search-harness regressions live
beside the harness in `noise_refresh.rs`. Both integration files enable the
GPU feature once at file scope; primitive
regressions remain under `src`. The refresh artifact/continuation checks also
live in `noise_refresh.rs`; the full RNS-BGG check can be selected by name:

```sh
MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL=average-case \
MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS=5 \
cargo test -r -p mxx-exponent-lut --features gpu --test noise_refresh test_gpu_rns_bgg_refresh_artifacts_with_native_prf_digits -- --nocapture
```

This full-refresh regression honors the same noise-model and failure-exponent
settings as the simulation and benchmark entry. It checks exact fresh-error
reconstruction followed by another LUT. Runtime checks
use GPU; no duplicate CPU runtime fixture is retained. Small search regressions
can be selected individually from the GPU-enabled integration target.

Integration tests have no `#[ignore]` attributes. Select a test name to run
only one workflow or parameter-search regression.

The RNS-BGG `lwr_prf.rs` round trip accepts
`MXX_EXPONENT_LUT_BENCH_ACCEPTED_REPORT`. It reuses the archived CRT chain and PBC
layout for one PRF output with the reviewed Gaussian encoding-error contract and
scaled terminal digit, then estimates and executes preprocess, encrypt, and
online evaluation in that order. The fixture samples a valid weight-h key on
that layout; this timing test does not certify the key distribution. There is
no partial-stage mode.

The NAND GPU test runs the same simulation, exact three-stage estimate, and
three-stage round trip. By default the NAND test runs the lattice estimator and
checks its result against `MXX_EXPONENT_LUT_NAND_SECURITY_BITS`. An explicit
`MXX_EXPONENT_LUT_NAND_SKIP_SAGE=1` skips both the estimator and that target
comparison, and logs `lattice_security_check=skipped`; no prevalidated evidence
is consumed by this mode.

For benchmark calibration, `MXX_EXPONENT_LUT_BENCH_WARMUPS` and
`MXX_EXPONENT_LUT_BENCH_ITERATIONS` control the estimator's unmeasured warmups
and measured repetitions (both default to one). They do not repeat the facade
round trip: the integration executes exactly one `preprocess`, one `encrypt`,
and one `online_eval`, timing each phase separately. The round-trip timings
include the phase's normal artifact and backend work; graph planning and
estimator calibration occur before those phase timers.
`MXX_EXPONENT_LUT_MAX_PARALLEL_INSTANCES` bounds sibling loop instances (default
64), independently of matrix column distribution across all detected GPUs.
