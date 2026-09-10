# RNS-BGG refresh H200 benchmark estimate, 2026-09-09

Both requested stages completed successfully on H200 GPUs with the accepted
parameters unchanged. Each integration invocation reported one passed test and
exit status 0. This is a GPU-calibrated benchmark estimate, not execution of all
163,577,856 PRF outputs or a full-refresh round trip.

## Results

| Metric | Preprocessing | Online evaluation |
| --- | ---: | ---: |
| Total GPU work (seconds) | 930715407341.8291 | 302504139281.21857 |
| Cumulative wave time (seconds) | 937148664266.75 | 302771340131.0382 |
| Unlimited-resource critical path (seconds) | 9.258870452999982 | 11.155515338000004 |
| Unlimited-resource maximum parallelism | 16709103205765 | 2748779069440 |
| Transmitted bytes | 33772524598438395936 | 61632352149588410368 |
| Persistent storage bytes | 165106358894969487392 | 63652764399806447616 |
| Observed peak VRAM (MiB, sampled every 3 seconds) | 77881 | 49371 |

The critical paths of approximately 9.26 and 11.16 seconds assume unlimited
resources. They are **not** completion times on one H200. Total work and
cumulative wave time include all logical invocations; neither is a measured
end-to-end refresh duration. Storage is the estimator's accumulated storage
accounting, not bytes physically written during calibration. The parameter set
still entails very large computation and storage despite compact host graphs.

The unlimited-resource memory fields use saturating `u64` arithmetic. Both
stages reach `18446744073709551615` for expanded workspace, workspace high water,
and peak memory; preprocessing also reaches it for resident VRAM. These values
indicate saturation and must not be presented as exact memory estimates or
physical GPU usage. The observed VRAM row above comes from `nvidia-smi` instead.
This interpretation follows `CostReport` and parallel-loop resource propagation
in `crates/bench-estimator/src/lib.rs`.

Exact metrics are in
`docs/plans/exponent-lut-rns-bgg-refresh-h200-metrics.csv`.
`docs/plans/exponent-lut-rns-bgg-refresh-h200.csv` retains attempt outcomes,
including earlier failures. Successful CSV fields were checked programmatically
against the raw reports and sampled VRAM logs, preserving integer byte counts.

## Configuration and provenance

- Parameter source: `docs/plans/exponent-lut-rns-bgg-refresh-security100.json`.
  Exact archived CRT primes and selector counts were retained.
- Ring dimension 65,536; 40-bit CRT primes; gadget base 2^20; initial depth 27.
- Sparse-LWR `(nu, h, Q, p) = (8192, 18, 4096, 32)`; 21 selector blocks.
- Average-case model and the existing 100-bit acceptance report; no new
  security claim or parameter search was introduced by benchmarking.
- 163,577,856 distinct PRF outputs and 167,772,160 terminal digit emissions.
- Two separate one-GPU H200 SXM pods, each with 251 GB allocated host RAM:
  `5tss91mxmyprlf` for preprocessing and `03a199xbjrk25f` for online evaluation.
  These results do not represent a two-GPU execution of either stage.
- Release build, `CUDA_ARCH=90`, `RUST_LOG=debug`, `RAYON_NUM_THREADS=16`,
  `MXX_GPU_VRAM_PERCENT=80`, one warmup and two measurement iterations.
- Branch `codex/exponent-lut-bfv`; base commit
  `919fa899116a09471b9686c510049805d9bdd307` plus uncommitted changes.
- Sources were copied directly by SCP and checked against per-file manifests.
  No commit or GitHub push was made.

| Stage | Source revision | Archive SHA-256 |
| --- | --- | --- |
| Online | `family-r2` | `5c3d08c63f8259fe680d05f0ee1d3c8ca3e4626c6fd631cd321fefbcb14fb5cc` |
| Preprocessing | `family-r3` | `51152334722b8194642210a35bd034cf99d5b413399a980d1e0afa07dbf250cd` |

The revision difference fixes measurement of compact centered extension in the
estimator by calling the existing runtime `centered_extend_small` operation.
It does not change the cryptographic graph, parameters, or GPU kernels. Online
measurement did not encounter the unsupported operand and completed on `family-r2`.

Command on each pod, with its respective `MXX_EXPONENT_LUT_BENCH_STAGE`:

```sh
cargo test -r -p mxx-exponent-lut --features gpu --test noise_refresh \
  test_gpu_exponent_lut_benchmark_estimate -- --exact --nocapture
```

`MXX_EXPONENT_LUT_TEST_SCHEME=rns-bgg`,
`MXX_EXPONENT_LUT_REFRESH_NOISE_MODEL=average-case`, and
`MXX_EXPONENT_LUT_REFRESH_SECURITY_BITS=100` were set. The accepted-report
variable selected the parameter JSON listed above. The harness calls production
PRF/refresh preparation, artifact export/import, and online graph construction.
Public monomial labels and trusted one-hot selectors are structural inputs;
hash-to-LWR-vector generation remains outside this GPU estimate.

## Graph changes and validation

Branch arithmetic is built once inside DSL `parallel`, retaining its original
multiplicity. Masks, terminal material, preparation, and online raw evaluation
use this structure. `Family::pack` reuses an existing Family for a full ordered
repack; reversed and partial member lists retain their original handling.
Existing artifact representations are preserved because the current IR rejects
nested indexed-Family exports.

The exact identity sieve and scalar digit-row compilation remain in use.
Every fresh branch ciphertext/error is retained, along with sampler tags,
RHS-before-fresh-encryption semantics, and raw accumulation before final ModDown.
No cryptographic noise formula or parameter was changed.

Targeted CPU tests passed for noiseless refresh followed by a LUT, terminal
weighted rows/raw accumulation, and Family projections/repacking. Full production
graph validation passed locally in 404.70 seconds with peak RSS 72,070,716 KiB
(approximately 68.7 GiB). The validated graphs contain 3,328,893 preprocessing
nodes in 974 scopes and 3,407,397 online nodes in 499 scopes. Both successful
H200 integration runs use these same graphs. Final GPU-feature compilation of
the integration target and estimator unit-test target was warning-free.
The compact-operand regression was added to the existing estimator unit test;
the full H200 preprocessing benchmark exercised the corrected operation.

An optional preliminary GPU unit test was stopped at the user's request; it is
not counted as a success. Benchmark estimation was then run directly. The first
preprocessing attempt reached GPU measurement but exposed the compact-operand
error; the corrected retry completed. Earlier `r4` attempts failed with SIGKILL
before measurement and remain recorded as failures, not timings.

## Logs and pod lifecycle

The durable log collection is `logs/rns-bgg-h200/20260909/`, including source
manifests, CPU validation logs, raw GPU logs, sampled VRAM, and exit statuses.
The successful remote log basenames are:

- `5tss91mxmyprlf-2pods-919fa899-20260909-preprocess-family-r3.log`
- `03a199xbjrk25f-2pods-919fa899-20260909-online_eval-family-r2.log`

Both successful pods were stopped after log retrieval: online at approximately
09:55 UTC and preprocessing at 10:10 UTC, with status `EXITED`. Their 300 GB
workspace volumes remain attached. The older failed pods `7me7t0wjowftjs` and
`dxl7dmqn0tkd0e` were deleted after restart was unavailable and logs were saved.

## Standalone one-output PRF estimate, 2026-09-10 JST

The exact one-output PRF producer and production-online graphs were subsequently
measured on one pod with two H200 SXM GPUs. This charges the entire selector
setup to one output, rather than dividing a full-refresh report by its output
count. The accepted parameters above, including the CRT primes and all 24,597
selectors across 21 blocks, were retained. The output is one modulus-32 digit
(five information bits), including terminal readout. The online estimate excludes
the round-trip test's secret-key correctness oracle.

| Metric | Preprocessing | Online evaluation |
| --- | ---: | ---: |
| Total GPU work (seconds/output) | 30226.4827334247 | 2159.9477590389984 |
| GPU work (seconds/information bit) | 6045.29654668494 | 431.98955180779967 |
| Cumulative two-GPU wave time (seconds) | 17609.74980554959 | 1391.4084882069997 |
| Unlimited-resource critical path (seconds) | 339.0799245650007 | 4.987623586999997 |

The cumulative wave sum is 19,001.15829375659 seconds, approximately 5.28 hours.
It is **not an end-to-end runtime prediction**: the measurement harness prepares
and completes input matrices before timing each operation. Runtime Family D2H,
RAM retention, subsequent H2D, artifact I/O and graph dispatch are not reproduced
as a complete execution in these timings. Nor may the unlimited-resource
critical path be interpreted as two-GPU runtime. The previous full-refresh
amortized GPU work divided by two was therefore not an adequate latency estimate
for this standalone test.

The selector-generation subgraphs account for 17,669.617892354 GPU seconds
(approximately 58.46% of preprocessing work), with exactly 24,597 invocations.
This is a component of the total, not an additional charge. The producer's
uncompressed artifact-payload estimate is 405,512,650,752 bytes. The estimator's
accumulated storage fields count logical intermediate outputs and must not be
substituted for actual disk usage; this run did not write those artifacts.

All 412 measured operation shapes completed with one primitive warmup and one
measurement iteration. The integration test passed once, exit status 0, in
398.19 seconds, including graph construction and calibration. This success is an
estimate test, not a completed production round trip. The small-parameter
estimate path also passed before deployment. Sampled peak VRAM was 63,459 MiB
on GPU 0 and 33,789 MiB on GPU 1, over the combined calibration of both stages;
the CSV records the maximum of these, not separate per-stage peaks.

Reproduction uses the same integration test with the existing accepted-report
path, average-case model, security setting 100, VRAM percentage 80, preprocessing
batch size 32 and `RAYON_NUM_THREADS=24`:

```sh
MXX_EXPONENT_LUT_TEST_SCHEME=rns-bgg \
MXX_EXPONENT_LUT_BENCH_STAGE=estimate \
MXX_EXPONENT_LUT_BENCH_WARMUPS=1 \
MXX_EXPONENT_LUT_BENCH_ITERATIONS=1 \
cargo test -r -p mxx-exponent-lut --features gpu --test lwr_prf \
  test_gpu_lwr_prf_one_output_round_trip -- --exact --nocapture
```

Pod `4wj6v4z7i03e0w`, revision `prf-estimate-r1`, used a direct SCP patch;
no commit or GitHub push was made. The source manifest SHA-256 is
`08bcdcaf718429204f77f68a6442850bd2bfd5cf16a6854658a8acfd451c5ae9`.
The primary log is
`4wj6v4z7i03e0w-2h200-919fa899-20260910-prf-estimate-r1.log`.
Logs, sampled VRAM, exit status, full reports and source manifest were retrieved
under `logs/rns-prf-h200/20260910/`. The two existing CSVs include explicitly
named `prf_one_output_*` rows, distinct from the original full-refresh results.
After log retrieval the pod was stopped and its status was confirmed as `EXITED`.

## Four-H200 round trip with a ten-hour limit, 2026-09-10 JST

The user requested a larger GPU allocation, one actual round trip, and an early
stop if measured progress clearly projected beyond ten hours. Eight-H200 pods
were unavailable at the attempted storage sizes. With the user's approval, a
four-H200 SXM pod (`imfeu375ctrf6g`) was provisioned with 500 GB container-local
storage, no network volume, 1,006 GB advertised RAM and 48 advertised vCPUs.
The runner selected 41 Rayon threads from the actual cgroup CPU quota. All four
detected GPUs participated; Family execution remained sequential and primitive
columns retained the common 80% VRAM budget.

The same cryptographic parameters and source implementation were used. The
source-manifest SHA-256 was
`61fb0d2377a1d8e547e4c75cbf98218b2d8f6a0d6e80bf15eae13137e4eb76cb`.
Sources and the OpenFHE dependency were copied directly. The release build
passed, followed by one 23.197265199-second primitive warmup. No GitHub push
was made. The run started at 03:44:01 JST, with a fixed deadline of 13:44:01 JST;
setup and compilation preceded that window.

**The round trip was interrupted during preprocessing, not completed.** After
about 21 minutes, progress had reached selector index 283 of 1,169 in the first
level. Excluding initial setup and the first eight selectors, progress advanced
275 selector indices over 959.673379108 seconds: 3.4897213785745453 seconds per
selector. The recent longer interval averaged 3.635877397683824 seconds. A short
32-selector window had reached 2.6376045146578924 seconds, but this improvement
did not persist over the longer observation.

The stop decision used the following operational projection:

1. Scale the measured complete selector-loop cost, including generation,
   weighting and accumulation, by the later levels' raw RNS data size. For
   level `i`, the relative size is `(26-i)*(50-2*i)/(26*50)`. Summing this over
   the actual selector counts gives 10,476.513846153846 first-level-equivalent
   selectors. This favors later levels by assuming no fixed per-selector cost.
2. Remove all 42 top-level selector-loop scopes from the prior one-output GPU
   estimate, avoiding double-counting the work already timed in the loop.
   Remaining preprocessing and online GPU work totals 6,626.276108013695 seconds.
   Assume ideal four-GPU speedup for that work: 1,656.5690270034238 seconds.
3. Do not add unmeasured host staging, artifact I/O or dispatch outside those
   selector loops. Even with these favorable assumptions, the sustained mean
   projects 10.615745380238538 hours; the recent rate projects
   11.041080257432172 hours.

These are projections from partial execution, not measured completion times or
rigorous lower bounds. The run was stopped under the requested projection rule;
it was not stopped by a cryptographic correctness assertion. Preprocessing had
not completed and online evaluation/correctness checks had not started. SIGTERM
produced cargo exit status 101, recorded as an intentional interruption in the
attempt CSV rather than a passing round trip. Sampled peak VRAM by device was
56,809, 24,017, 15,313 and 13,603 MiB.

The exact rates, assumptions and result are in
`docs/plans/exponent-lut-rns-bgg-prf-h200-10h.json`. Full bootstrap/run logs,
progress samples, VRAM, exit status and source manifest were retrieved under
`logs/rns-prf-h200/20260910/`, including
`imfeu375ctrf6g-4h200-919fa899-20260910-prf-10h-r1.log` and
`imfeu375ctrf6g-prf-10h-r1-progress.json`.
The pod was stopped after retrieval and confirmed `EXITED` at 04:10:55 JST.


## Protocol-independent dataflow estimate, 2026-09-10 JST

The common benchmark estimator now derives transfer counts from the validated IR,
including Family boundaries, captures, sequential carries, shared inputs and
artifact exports/imports. It uses the production backend, raw-RNS staging path,
compact codecs and local artifact store. No selector-specific timing or PRF-loop
sampling was added. GPU-resident edges do not receive an extra transfer charge.
CPU dispatch is calibrated with a small scalar IR graph and reported as a proxy.

One warmup and one measured iteration per shape were run on four H200 SXM GPUs
in pod `g4fkmavzy4o6sp`, with the previously accepted 100-bit average-case
parameters, one modulus-32 output (five information bits), batch size 32,
41 Rayon threads, and an 80% VRAM budget. All selector setup is charged to that
one output; the correctness oracle is excluded from the online estimate.
The run measured 412 primitive shapes and 137 transfer shapes and passed in
928.92 seconds. This is estimate-harness time, not PRF execution time.

| Component (seconds) | Preprocess | Online evaluation |
| --- | ---: | ---: |
| Primitive fleet-wave time | 27,792.63903445411 | 2,785.922365427001 |
| RAM to GPU | 28,474.240765832 | 412.425451922 |
| GPU to RAM | 8,488.857711752002 | 57.81189056600002 |
| Compact export and local write | 1,248.5036780179998 | 0 |
| Local read and compact import | 519.1008599800001 | 615.9350173620002 |
| CPU dispatch proxy | 10.969824826552529 | 1.197788311856031 |
| **Total estimate** | **66,534.31187486267** | **3,873.292513588857** |

The estimated totals are **18.4817533 hours preprocessing** and
**1.0759146 hours online**, or **19.5576679 hours combined**.
Transfer/store/load contributes 39,816.87537543201 seconds (11.0602432 hours).
The report's primitive GPU-work counters are 95,406.96235806022 and
7,486.275300306999 GPU-seconds; these sum device elapsed times and are not
pure CUDA-kernel timings. The separate fleet-wave rows above are wall-time costs.

The previous two-H200 estimate was 5.2781 hours and omitted these boundary
costs. The entire difference must not be attributed to the new transfer model:
this run's primitive-wave subtotal is also higher (8.4940448 hours).
For example, the same largest `MatrixMulSmallRhs` shape's maximum reported
representative-device latency increased from 0.395416 seconds to 0.713866 seconds.
The old pod used two GPUs with 24 Rayon threads and CPU affinity 24-47 on one
NUMA node; the new allocation uses four GPUs across NUMA nodes and 41 threads.
These logs establish a measured primitive-cost change, but do not isolate its
cause. No claim of a cryptographic workload increase or a specific hardware
bottleneck is made from this comparison.

The estimate adds separately measured costs and does not predict overlap,
cold-cache filesystem performance, or contention throughout a full execution.
The CPU dispatch term is a proxy. Import calibration includes public hash
verification, conservatively also charged to private imports. Large transfer
shapes round a final partial wave upward. Therefore 19.56 hours is a model
estimate, not a measured round-trip duration or a rigorous lower bound.

CPU estimator/runtime tests and the small GPU estimate passed. Earlier small
calibration attempts exposed missing/invalid fixture manifests and a scalar-IR
type mismatch; these were fixed. The first full attempt (`dataflow-r4`) was
intentionally stopped because a zero-valued matrix fixture undercounted adaptive
compact serialization. The successful `dataflow-r5` uses uniform full-modulus
matrix fixtures; no production cryptographic parameters were changed.

Sources were copied directly without a GitHub push. The base commit is
`919fa899116a09471b9686c510049805d9bdd307`; the source-manifest SHA-256 is
`8cfabb92166385bda35ed14bae8d6661aa8ee419d896f321d1cc4eecc5fdc7b2`.
The main log is `g4fkmavzy4o6sp-4h200-919fa899-20260910-dataflow-r5.log`.
Raw reports, previous-attempt logs, runner scripts, manifest and VRAM samples
are retained under `logs/rns-prf-h200/20260910/dataflow-r5/`.
Sampled peak VRAM was 63,939 / 18,819 / 17,317 / 17,061 MiB by device.
The existing CSVs were appended and programmatically checked against the raw
reports. A compact result is in
`docs/plans/exponent-lut-rns-bgg-prf-h200-dataflow.json`.
The pod was stopped after log retrieval and confirmed as `EXITED`.


## Pinned-buffer reuse investigation (in progress), 2026-09-10 JST

The `pool-r1` candidate reuses completed pinned transfer allocations, retaining
at most two buffers per detected device. Existing completion-event reclamation
controls eligibility for reuse. This is an experiment toward the three-hour
round-trip objective, **not an accepted performance improvement**.

A CUDA allocation/copy diagnostic transferred 184,025,088 bytes per GPU on four
H200s. Excluding the first iteration, the mean of the slowest-device elapsed
times was 0.35751108545809984 seconds with fresh allocation/free and
0.021604795241728425 seconds with the common pool. This is a microbenchmark;
it is not a PRF throughput claim. Event-protected transfer contents matched
on all four devices, three transfers per device. The existing pinned-copy and
RNS-column staging GPU unit tests each passed once; GPU-feature compilation passed.

The complete same-parameter estimate (`pool-r1`, 951.40-second harness run,
exit 0) nevertheless regressed: preprocessing was 83,024.76642144757 seconds
and online evaluation was 5,394.418075615581 seconds. Preprocessing raw H2D
fell from 28,474.240765832 to 985.998913484 seconds, but its primitive-wave
subtotal grew from 27,792.63903445411 to 68,612.4289298234 seconds.
The candidate must not be presented as an end-to-end speedup.

A short same-pod control with the pre-pool `Runtime.cu` restored representative
matrix-multiplication latencies to roughly 0.6-0.8 seconds. That control was
intentionally stopped after three representative shapes; it is neither a full
estimate nor a round-trip pass. Holding approximately 77 GB of pinned buffers
in a separate diagnostic increased small allocation time only by a few
milliseconds, insufficient by itself to explain the regression. CUDA API
instrumentation is being used to isolate the remaining cause. Instrumentation
lives outside repository sources.

The candidate manifest SHA-256 is
`7813a72c123f2463769203f945c3726d02e1af6a37483871e163c10251f70bc0`.
The complete candidate reports, unit-test output and diagnostic scripts/logs
are preserved under `logs/rns-prf-h200/20260910/pool-r1/`.
The three-hour production PRF round trip remains unverified.


### Compact-operand fixture correction (`pool-probe-r2`)

The compact operand fixture in `crates/bench-estimator/src/gpu.rs` allocated
an entire dense RNS zero matrix solely to obtain ring parameters. CUDA API
tracing found a 35,441,868,800-byte pinned allocation on this path. The fixture
now obtains the same parameters from a 1-by-1 matrix; the measured operand's
actual shape and coefficient bound are unchanged. Its GPU shape test passed.

With this correction and the common completed-buffer pool retained, the same
four-H200 estimate completed successfully (exit 0, both reports present):

| Stage | Primitive waves (s) | Transfers (s) | Dispatch (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: |
| Preprocess | 16722.154581492316 | 11422.570394163999 | 10.949652384093385 | 28155.674628040408 |
| Online | 1249.0544871160007 | 705.8506465319997 | 1.1956052719610897 | 1956.1007389199615 |

The combined estimate is 30111.77536696037 seconds (8.364382 hours), not an
actual round-trip duration. Preprocess D2H staging still contributes
8612.112280292 seconds. Sampled peak VRAM by device was 13295 / 13633 / 12833 /
12833 MiB. The source manifest SHA-256 is
`bd84db0a04e6d02ee6f1f4a3ceb2f529433e3ff4ac5f488a5f69e5311ba7c248`.
Raw reports and logs are retained under
`logs/rns-prf-h200/20260910/pool-probe-r2/`; the existing CSVs were appended
and checked against both raw reports. No cryptographic parameters changed,
and no source was pushed to GitHub.

Subsequent primitive changes remove the host upload for zero initialization
and parallelize canonical small-coefficient validation with precomputed byte
bounds. The zero round-trip and CPU-oracle coefficient-boundary GPU tests each
passed once. The full `host-r1` estimate completed, but regressed to
43769.809902032 seconds combined (preprocess 40624.427070789614, online
3145.382831242387). Small-coefficient export/import improved, while primitive
wave times and raw input loads increased. Its source manifest is
`211a08537301442fe6eb3968bdfeb78ea65c3766ee5f45978adbba390f3306c8`.

`rns-store-r1` batches the existing RNS unpack operation across a matrix's CRT
limbs when its allocation belongs to one device. The existing streaming
D2H/H2D test passed with both 17-bit and 40-bit CRT primes. The full estimate
completed at 43013.40545099258 seconds combined (preprocess 39767.521160888326,
online 3245.8842901042485), so this did not resolve the regression. The compact
transfer fixture now includes signed nonzero coefficients; its declared width
is unchanged. Its source manifest is
`62dff706d57bd033d2ef1ceca2b18fe27136bebbcae6ec1bf1ff8bb979bb0698`.
Both runs' raw reports, commands, manifests and VRAM logs are preserved under
`logs/rns-prf-h200/20260910/`, and both CSVs were appended and checked.

The next candidate removes byte-by-byte 128-bit remainders when a compact
coefficient fits in a 64-bit word. It retains reduction for values above the
CRT prime and the general path for wider coefficients. OpenFHE-profile and
tile-tail GPU tests passed. Its production-shape measurement is in progress.
No production round trip has completed. The eight-H200 Secure and Community
creation attempts with 500 GB local storage both reported unavailable stock;
no additional pod was created.


### Current execution candidate (`rns-load-r1`)

The `compact-word-r1` estimate completed at 43145.31138257129 seconds
(preprocess 40008.42363519131; online 3136.88774737998). It did not demonstrate
a material overall improvement. Its manifest is
`b27a48bab44f56f3320a52f85eaa19c74686e7f27bd296190445af7f1cf37856`;
its raw reports and CSV-checked results are retained with the other attempts.

The current candidate also batches whole-matrix H2D packing. When one transfer
completion event suffices, that event is handed directly to the existing pinned
reclaimer rather than joined behind unrelated release-stream operations. It
still retains the source host allocation until GPU completion. The pipeline
round-trip test passed for 17-bit and 40-bit CRT primes. Primitive measurement
inputs now use uniform full-modulus matrices and signed nonzero compact
coefficients, avoiding a zero-only arithmetic fixture. The estimator shape test
passed with these fixtures. This does not retrospectively convert earlier
zero-operand measurements into production runtime evidence.

The source manifest is
`82068ccfb06d62b39afa2c4fd1c15728f0639d78d5d60935dc9d30756f333c61`.
A production-parameter PRF round trip has been started on the existing four-H200
pod with primitive warmup and one measured run. Its log is
`g4fkmavzy4o6sp-4h200-919fa899-20260910-roundtrip-load-r1.log`. The run must be
judged by its actual completion/verification record; no three-hour success has
been established. Source transfer remains direct SCP, without a GitHub push.


### Production screening run and external interruption

`roundtrip-load-r1` completed primitive warmup in 17.451919499 seconds and
entered production preprocessing. SSH observations recorded selector index 1
at 124.310187994 seconds and index 183 at 471.801451829 seconds (45 progress
samples), giving 1.909292658434066 seconds per selector over that interval.
Progress denotes entry to the next selector after the previous selector's
retained RAM outputs; it is not a completed PRF. The run was intentionally
stopped for the fused-product change and its wrapper reported exit 143.

Before the complete log could be retrieved, SSH became unavailable and Runpod
returned HTTP 404 for pod `g4fkmavzy4o6sp`; the account's pod list was empty.
No pod-delete command was issued by this task. Subsequent eight-H200 allocation
failed for stock availability and four-H200 allocation failed with insufficient
account balance. Full remote logs from this screening run are unavailable;
the observed values, source manifest and runner are preserved under
`logs/rns-prf-h200/20260910/roundtrip-load-r1/`. Earlier complete estimate logs
had already been copied locally.

The next source candidate expresses selector generation and weighted
accumulation with the existing matrix-product-accumulation primitive. The
polynomial backend's singleton entry point now delegates to its existing batch
implementation so a fleet shard actually reaches the fused GPU operation.
Fresh error samples and shared selector errors are unchanged. The selector
sharing CPU test passed after both changes. The GPU-feature test binary builds
successfully. Its source manifest is
`7f99d365f78c40a399d22d3fb7e04c274be7f6f81ef8282af1f8c23c9a38bb2d`.
Production GPU validation and the requested complete three-hour round trip
remain outstanding; Runpod account funding is required before remote work
can resume. No GitHub push was performed.


Local RTX 4080 SUPER validation of the fused selector candidate also passed:
`test_gpu_rns_bgg_composed_rhs_preserves_relations` ran once, with one pass
and no failures. This is a small unit-test result, not H200 production timing.
The full source archive and bootstrap/round-trip runners are prepared for
resumption; their manifest matches all current source files.
