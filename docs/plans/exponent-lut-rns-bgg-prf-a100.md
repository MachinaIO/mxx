# One RNS-BGG PRF output on Runpod A100

Status: 2026-09-11 JST. Checkpoint preservation and the latest-source A100
estimate are complete. All 24,934 preprocessing files are backed up and verified
locally. The `policy-r1` estimate passed, predicting 8,899.102997905768 seconds
(2.47 hours) for one output. The user explicitly deferred round-trip resumption;
no replay was started. The original eight-A100 production attempt completed
preprocessing in 24,953.029870474 seconds, then failed with CUDA out-of-memory
before correctness verification. That failed attempt is not a completed
round-trip measurement.

## Objective and frozen workload

Complete the benchmark estimate and one noisy RNS-BGG LWR PRF output on a single
Runpod machine using at most eight A100 GPUs. The supplied H200 instruction's
original three-hour target is now a historical comparison only. The latest
instruction supersedes both a three-hour stopping rule and the 11,400-second
wrapper timeout: continue the new run through correctness verification without
a time limit. An estimate cannot establish actual round-trip completion.

The authoritative parameter input is
`docs/plans/exponent-lut-rns-bgg-refresh-security100.json`, SHA256
`7bf78a66a6cd1b0672050b073d0418fded774466f6667408258a29bacbdb1991`.
Use its exact CRT primes and PBC layout without weakening the parameters.

| Parameter | Frozen value |
| --- | ---: |
| Scheme | RNS-BGG |
| Ring dimension | 65,536 |
| CRT prime size / maximum depth | 40 bits / 27 |
| Initial modulus bit length | 1,080 |
| Gadget base | 2^20 |
| LWR dimension / sparse weight | 8,192 / 18 |
| LWR input / output modulus | 4,096 / 32 |
| PBC blocks / total selectors | 21 / 24,597 |
| Fresh error | Centered binomial, eta = 2 |
| Noise model | Average case |
| Requested security/failure exponent | 100 bits |

The block selector counts are
`[1169, 1183, 1156, 1149, 1149, 1201, 1240, 1157, 1149, 1181, 1178, 1181, 1136, 1186, 1185, 1131, 1191, 1200, 1174, 1159, 1142]`.
Fresh errors remain randomly sampled. One PRF output is one digit modulo 32,
equivalent to five information bits. The full refresh workload of 163,577,856
PRF outputs is outside this execution. Dividing work by five is a normalization,
not a separate one-bit experiment.

The accepted report permits continuation using simulated correctness and the
recorded heuristic noise model. It explicitly records
`joint_security_proved=false` and `numerical_run.security_verified=false`;
this run does not change those security-evidence boundaries.

## Current implementation and source identity

The local base is `1b75e4e9a` on branch
`codex/exponent-lut-a100-benchmark`. No source is pushed to GitHub. The remote
checkout receives local source archives with SHA256 verification.

The starting source restored six files to match all 501 entries of the supplied
`replica-owner-r1` manifest. This retained wide-CRT fused multiply-accumulate
with integer Barrett reduction, GPU loop-carried state and matching transfer
estimation, owned placement, immutable shard borrowing, fixed-replica reuse,
and deferred duplicate cache-hit memory queries. The wide-CRT restoration is
required for the accepted 40-bit workload.

The retained `advisory-r2` implementation follows the user's explicit
requirement that `MXX_GPU_VRAM_PERCENT=80` guide parallelization without causing
an error solely because residency reaches 80%:

- Shared runtime and estimator width calculation retains at least one column
  when available memory under the configured target would otherwise imply zero.
- Compact-RHS multiplication uses cached physical device capacity for admission
  and CUDA allocation-report validation. Context creation caches the minimum
  physical total across its devices; multiplication adds no driver memory query.
- Actual CUDA allocation errors, invalid memory/context metadata, and impossible
  compact-RHS footprints remain errors. Fresh errors and crypto parameters are
  unchanged.

The `checkpoint-r1` source added durable test-harness checkpointing
to these crypto/runtime/kernel semantics. Its source archive is
`logs/rns-prf-a100/20260910/checkpoint-r1/source.tar.gz`, with 506 source entries.
The manifest is `logs/rns-prf-a100/20260910/checkpoint-r1/source-manifest.json`,
SHA256 `94f2eed2567e5e0dc353904d6e718f6325cef2d762b6f1b5b2ef15764e789a9f`.
The archive identifies the executed source independently of later local
report updates. No GitHub push is performed.

The historical `advisory-r2` source archive and 506-entry manifest remain under
`logs/rns-prf-a100/20260910/advisory-r2/`. That manifest's SHA256 is
`cfa4afcb6eccaefb626ddcfc96f1c8280185a1fb8986097ea242cbdcf4479afa`.

## Artifact recovery and new checkpoint

The stopped run left 24,932 artifact files and its runtime manifest. Those
preserve computed preprocessing values, but do not preserve the original
independent correctness oracle. The manifest records artifact types, names,
confidentiality, hashes, and production identity; application descriptors and
the randomly selected expected PRF digit remained in the terminated process's
online closure. They were not serialized to disk or printed in the retained
log. The existing artifacts therefore cannot be reused through the existing
import/check path while preserving that original oracle. No output-derived
expected value is substituted for the missing oracle. At the user's explicit
request, the obsolete `.tmpH6umFn` directory was deleted on 2026-09-10 at
10:15 UTC after checking its production identity and confirming that the live
process used the separate `.tmpPUexnO` directory. This freed approximately
278.8 GB (259.7 GiB). Historical logs and source archives remain available;
the current run's artifacts were preserved. The deletion record is
`logs/rns-prf-a100/20260910/serialization-r1/old-preprocessing-deleted.json`.

For the new run, explicit `MXX_EXPONENT_LUT_BENCH_STAGE=roundtrip` mode retains the temporary
artifact directory and prints its path immediately. After real preprocessing,
store reopening, and online graph construction, it atomically writes
`online-checkpoint.json` in that directory. The file contains the serialized
online graph and parameter bindings, exact CRT parameter levels and eta,
production identity, oracle bounds, noise report, and completed phase timings.
The graph contains the original expected-value constants.

Before online execution, the harness reloads that file, checks its metadata,
reconstructs the parameters, and validates the graph against the original
manifest. It executes the reloaded graph with the existing correctness checks.
Checkpoint writing, syncing, reloading, and validation are included explicitly
as `checkpoint_seconds` in measured round-trip time. This adds no full-PRF
warmup or extra full execution. Default small tests without explicit roundtrip
mode retain their original temporary-directory cleanup behavior.

An ignored diagnostic integration test now restores and executes an existing
GPU checkpoint: `rns_bgg::test_gpu_rns_bgg_saved_online_checkpoint`, with
`MXX_EXPONENT_LUT_BENCH_CHECKPOINT` naming the saved JSON file. Its parent
directory supplies the artifact store. The test validates the saved graph
against the original manifest, uses detected GPUs (including CUDA visibility
selection), and checks the original saved oracle bounds. It does not regenerate
preprocessing. This is a test entry point, not a general resume CLI, and it
cannot recover metadata omitted by the stopped historical run.

The replacement pod's completed checkpoint is
`crates/exponent-lut/test_data/rns_bgg_round_trip/.tmp5sBfBL/online-checkpoint.json`.
After the failed online attempt, filesystem inspection confirmed a 32,062,905-byte
JSON and 24,934 files totaling 278,738,540,875 bytes in its artifact directory.
This inventory is presence/size evidence, not a complete content-integrity check.
The original run had already reloaded this checkpoint and validated its graph
before entering online execution. A separate-process production replay has not
yet completed. It can use the existing compiled checkpoint test without
regenerating preprocessing or deploying the deferred policy changes.

## Original hardware and configuration

Eight-device A100 SXM and PCIe allocations were unavailable. Pod
`a083o9u6yunqf3` was allocated with four A100 SXM4 80GB GPUs at a quoted
$6.36/hour. The execution uses all four detected devices on this one pod.

| Resource | Observed or requested value |
| --- | --- |
| GPU model/count | A100 SXM4 80GB × 4 |
| Host memory cgroup `memory.max` | 1,000,999,997,440 bytes |
| CPU cgroup `cpu.max` | `5440000 100000` (54.4 CPU equivalents) |
| Requested container disk | 750 GB |
| CUDA architecture | `CUDA_ARCH=80` |
| Rayon threads | `RAYON_NUM_THREADS=55`, ceiling of CPU quota |
| VRAM parallelization target | `MXX_GPU_VRAM_PERCENT=80` |
| Preprocess batch / maximum parallel instances | 32 / 32 |

Requested disk size is not measured free space or peak usage. The historical
production graph's online artifact payload was 405,512,650,752 bytes; that is
not a measured filesystem peak. Family execution remains bounded to one member
at a time on the GPU fleet, with primitive columns distributed in memory-bounded
waves. Active loop accumulators remain on GPU; completed root-loop values return
to RAM. The percentage controls wave size, with a one-column minimum. Online
artifacts persist on disk for reopening.

## Execution and acceptance contract

Use `test_gpu_lwr_prf_one_output_round_trip` in
`crates/exponent-lut/tests/lwr_prf.rs`, with release mode, GPU features,
`RUST_LOG=debug`, the accepted report, RNS-BGG, average-case noise, 100 requested
security bits, one primitive warmup, and one measured iteration.

The two stages are `MXX_EXPONENT_LUT_BENCH_STAGE=estimate` and
`MXX_EXPONENT_LUT_BENCH_STAGE=roundtrip`. Both final results must use the same
source revision. The estimate calibrates production
primitive shapes, generic transfers, artifact I/O, and dispatch. Its
production-online graph excludes the verification oracle. The actual round
trip includes correctness verification. Its GPU oracle arithmetic computes the
error relation inside the online graph and is included in `online_eval_seconds`.
The later host-side `check_matrix` checks are timed in `verification_seconds`.
Consequently, actual online execution and the production-online estimate have
different oracle boundaries.

The production test must export preprocessing artifacts, finish GPU outputs,
drop and reopen the store and manifest, construct and execute online evaluation,
finish GPU results, and pass the existing matrix-relation and noise checks.
Primitive warmup occurs once; a full-PRF warmup is not permitted.

Acceptance requires successful test exit and a record of this form:

```text
prf_round_trip_timing iteration=0 measured=true preprocess_seconds=... reopen_seconds=... online_graph_seconds=... checkpoint_seconds=... online_eval_seconds=... verification_seconds=... round_trip_seconds=... correctness=true
```

The round-trip timer includes preprocessing, reopening, online graph
construction, checkpoint persistence and validation, online execution, and
verification. It excludes compilation, initial fixture construction, and
primitive warmup. There is no elapsed-time stopping threshold for the new run.

Record commands, environment, source identity, topology, exit status, and GPU
memory/utilization approximately every three seconds. Retrieve raw evidence
under `logs/rns-prf-a100/20260910/` before pod cleanup. Report elapsed estimates
separately from measured elapsed time and aggregate GPU work seconds/hours.
An idealized critical path or partial selector interval cannot establish full
latency. Progress with `gpu_completion="not_observed"` is host dispatch evidence.

## Historical advisory-r2 validation

`advisory-r2` completed both workspace library release builds
(`cargo test -r --workspace --lib --no-run`, with and without `--features gpu`)
and the GPU `lwr_prf` binary build without warnings. Compilation is outside the
PRF timing contract.

Seven GPU unit tests each passed three consecutive invocations:

- `test_gpu_multiply_accumulate_full_modulus_matches_cpu`
- `gpu_fleet_waves_preserve_decomposition_and_canonical_artifacts`
- `test_gpu_bidirectional_peer_fleet_small_rhs_waves_match_single_device_canonical_output`
- `test_gpu_rns_bgg_composed_rhs_preserves_relations`
- `gpu_runtime_miss_records_a_profile_and_cache_hit_defers_width_derivation`
- `test_gpu_context_caches_configured_vram_percentage`
- `test_gpu_small_rhs_cuda_report_matches_planner_at_exact_budget`

Three CPU tests each passed once:

- `sequential_carried_family_uses_production_staging_rule`
- `iteration_returns_completed_matrix_state_to_ram`
- `pilot_cost_rounds_up_and_vram_target_limits_parallelism`

Total: **24 invocations passed, zero failed**. Evidence is
`logs/rns-prf-a100/20260910/advisory-r2/validation-summary.json` and its referenced
individual logs. These checks cover narrow arithmetic, ownership, staging,
capacity semantics, and context behavior; they do not establish production
round-trip correctness.

## Historical replica-owner-r1 evidence

The initial `replica-owner-r1` A100 x4 estimate exited successfully:

| Quantity | Preprocessing | Online |
| --- | ---: | ---: |
| Estimated elapsed seconds | 7,913.241 | 832.471 |
| Aggregate GPU work seconds | 22,246.089 | 1,254.158 |

Combined estimated elapsed time was 8,745.712 seconds (2.429 hours). Aggregate
GPU work was 23,500.247 GPU-seconds (6.528 GPU-hours across four A100s), not
elapsed time. This estimate predates the advisory-target change and cannot
stand in for the later advisory-r2 estimate below. It is also not a controlled
comparison with historical H200 estimates using different source revisions.
Comparing this initial estimate against `advisory-r2` actual preprocessing also
mixes source revisions and cannot establish a controlled timing discrepancy.
Raw evidence is in `logs/rns-prf-a100/20260910/estimate-reports.json`,
`estimate-summary.json`, and the corresponding `estimate-replica-owner-r1.log`
under the same directory.

The initial production attempt failed with exit 101 after 394.05 seconds,
including fixture construction and warmup, during the first selector block.
It reported `Gpu0 cannot fit one calibrated column`: the configured budget was
68,075,021,926 bytes, calculated residency 68,044,284,464 bytes, and calibrated
per-column demand 34,090,578 bytes. No preprocessing completion, online
execution, or correctness result was produced. This percentage-based rejection
prompted the final advisory-target semantics. Its raw log is
`logs/rns-prf-a100/20260910/a083o9u6yunqf3-4a100-1b75e4e9-20260910-roundtrip-replica-owner-r1.log`.

Initial estimate and failed-attempt rows remain in the existing
`docs/plans/exponent-lut-rns-bgg-refresh-h200-metrics.csv` and
`docs/plans/exponent-lut-rns-bgg-refresh-h200.csv`; their hardware fields
explicitly identify A100 despite the historical filenames.

## Storage accounting

The graph reports an export artifact payload of 405,512,650,752 bytes. The
initial estimator's `persistent_storage_bytes` value of 2,008,658,214,944 bytes
sums logical output sizes of all IR nodes, including intermediates. Neither
quantity is measured peak disk usage. Similarly, pool-reserved memory visible
in `nvidia-smi`, calculated resident bytes, and hypothetical estimator memory
peaks are distinct quantities.

## Historical advisory-r2 estimate and actual outcome

The matching-source `advisory-r2` estimate completed with exit status **0** on
four A100 SXM4 80GB GPUs:

| Component | Preprocessing seconds | Online seconds |
| --- | ---: | ---: |
| Primitive waves | 6,543.650518 | 342.601646 |
| Transfer and artifact I/O | 1,520.347452 | 585.664913 |
| Executor dispatch proxy | 8.564111 | 1.052829 |
| **Estimated elapsed total** | **8,072.562081** | **929.319388** |
| Aggregate GPU work seconds | 23,240.667367 | 1,140.699177 |

Combined estimated elapsed time is **9,001.881469277347 seconds (2.5005 hours)**.
Aggregate GPU work is **24,381.366544 GPU-seconds (6.772602 GPU-hours)** across
four A100s, not wall-clock elapsed time. Calibration VRAM peaks were
12,767 / 13,439 / 12,703 / 12,703 MiB; these do not bound production residency.

Advisory-r2 estimate evidence:

- `logs/rns-prf-a100/20260910/advisory-r2/estimate-summary.json`
- `logs/rns-prf-a100/20260910/advisory-r2/estimate-reports.json`
- `logs/rns-prf-a100/20260910/advisory-r2/a083o9u6yunqf3-4a100-1b75e4e9-20260910-estimate-advisory-r2.log`

Two advisory-r2 metric rows and an estimate-attempt row were appended to the
existing CSVs named above and checked against the raw reports.

The `advisory-r2` production attempt completed preprocessing in
**10,283.556546401 seconds**, approximately **27.4% above its matching-source
preprocessing estimate**, and reopened its artifact store before beginning
online evaluation. Both preprocessing paths use `fixture.producer`; the
measured difference is evidence of estimator error for this run, not a claimed
kernel regression. The last recorded online progress was
362.242183638 seconds, at branch 513 of 2,048. That progress describes host
dispatch, not completion of all corresponding GPU work, and is not a full
online measurement comparable with the 929.319388-second estimate.

The attempt was stopped because its observed progress indicated that completion
within the 10,800-second target was no longer feasible. An extension request
had received no answer. SIGTERM was sent to the test command managed by the
timeout wrapper, producing exit status **143**. This is an operator-initiated
termination, not a successful test exit or a correctness failure reported by
the test.

No final `prf_round_trip_timing` record or `correctness=true` result was emitted.
Online evaluation and verification did not complete, and there is no measured
full round-trip latency. The one-output round-trip objective therefore remains
incomplete; the three-hour target was not demonstrated. The observed partial
online interval must not be added to preprocessing and presented as a completed
round-trip measurement.

These figures describe the historical stopped attempt only. The latest user
instruction authorizes the subsequent unlimited run and supersedes the earlier
stopping decision.

## Checkpoint diagnostics and serialization correction

Three checkpoint-enabled small full-PRF GPU tests failed after online execution
with errors around 10^9 against the unchanged bound 7. Three controls bypassing
checkpoint persistence failed similarly, so the failures were not attributable
to graph serialization alone. The independent CPU full-PRF reference passed.
Single-GPU controls passed twice; the third exceeded the average-case,
security-1 bound only slightly (8 versus 7). That small statistical excursion is
separate from the large multi-GPU errors and does not justify weakening bounds.
Production parameters remain the frozen average-case, security-100 workload.

An attempted CPU replay of GPU checkpoint artifacts failed during byte decoding,
before arithmetic: CPU and GPU matrix artifact codecs differ. This did not
establish corrupt preprocessing. The saved-checkpoint diagnostic was therefore
changed to GPU execution so that one GPU can consume the original multi-GPU
artifacts with the correct codec. The independent CPU full-PRF test remains.

Source inspection identified a concrete signed-magnitude packing bug in
`crates/runtime/src/backend/poly_gpu/fleet.rs`, in `matrix_to_bytes`. When
shards have different packed coefficient widths, the old merge copied a narrow
shard's sign bit into the wider coefficient's magnitude field instead of moving
it to the new top bit. For example, a two-bit encoding of -1 becomes +3 if
copied unchanged into four bits. The CUDA encoder places the sign at
`bit_width - 1` in
`crates/primitives/cuda/src/matrix/MatrixSerdeBatch.cu`.

`serialization-r1` corrects this merge by copying the magnitude bits separately
and relocating the sign bit, while handling zero-width all-zero shards. This
is an artifact representation correction; it does not alter cryptographic
parameters, expected values, noise bounds, or GPU arithmetic. Matrix import
uses one uniform encoded width; the separate SmallMatrix codec already uses a
common schema-derived width and a separate sign byte.

The source archive and manifest are under
`logs/rns-prf-a100/20260910/serialization-r1/`. The manifest SHA256 is
`13a79cdd488e6c7b626f13e50f4e92412f8f165050e5411b26e931065e950156`.
Both workspace library release builds (CPU and GPU features) and the GPU
`lwr_prf` integration binary build passed without warnings. The signed-width
regression passed three consecutive invocations. Three full-PRF runs on four
GPUs also passed using the same small diagnostic configuration that previously
failed: ring dimension 4, average-case noise, and security exponent 1. No noise
bound or cryptographic parameter was weakened for these passes.

A fresh checkpoint produced by the corrected four-GPU run was replayed in a
new single-GPU process and passed in 0.87 seconds. This demonstrates that the
saved graph, oracle metadata, and GPU artifact codec can support online-only
restoration on another visible GPU count for that small fixture. It is not a
production-size recovery or correctness result.

The unlimited production round trip launched with the exact frozen accepted
parameters on four A100s, with its estimate queued only after successful
completion. That pod subsequently became unavailable during preprocessing, so
neither final result was obtained. There is no three-hour stopping rule or
190-minute wrapper timeout. Runtime source identity remains the archived
`serialization-r1` source; later local documentation edits do not change it.

The task is complete only after preprocessing, artifact reopening, online
evaluation, and existing correctness verification all finish with a successful
test exit, followed by the matching-source estimate. Preserve the current reusable artifacts
and all diagnostic evidence until validation finishes. Retrieve final source identities, checkpoints,
raw logs, resource samples, and exit statuses before eventual pod cleanup.

## Pod loss and current recovery

Original pod `a083o9u6yunqf3` became unavailable during the
`serialization-r1` production preprocessing run. REST returned 404, GraphQL
returned `pod: null`, and a start attempt reported that the pod could not be
found. Evidence is retained in
`logs/rns-prf-a100/20260910/serialization-r1/pod-unavailable.json`. No completed
preprocessing checkpoint or production correctness result from that run is
claimed.

The user replenished the account balance to approximately $149.65 and preferred
reuse of the old pod. Reuse was attempted and failed because the pod was no
longer found. Earlier, the user had also requested deletion of old preprocessing
artifacts; this request belongs to the historical storage-management record,
not a claim that those artifacts provide a usable current checkpoint.

Replacement pod `rxgmt08k3j3vab` is allocated as `recovery-r1`:

| Resource | Observed or requested value |
| --- | --- |
| GPU model/count | A100 SXM4 80GB × 8 |
| Quoted running price | $12.72/hour |
| Host memory cgroup `memory.max` | 942,999,998,464 bytes |
| CPU cgroup `cpu.max` | `16320000 100000` (163.2 CPU equivalents) |
| Rayon threads | `RAYON_NUM_THREADS=164`, ceiling of CPU quota |
| Requested container disk | 750 GB |
| Image | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` |
| CUDA architecture | `CUDA_ARCH=80` |
| VRAM parallelization target | `MXX_GPU_VRAM_PERCENT=80` |

Setup completed successfully. Both CPU and GPU workspace library release
builds and the GPU `lwr_prf` integration binary build completed with exit 0 and
without warnings. The signed-width serialization regression passed three
invocations, and the small full-PRF test passed three invocations using all
eight GPUs. Metadata and validation logs are retained under
`logs/rns-prf-a100/20260910/recovery-r1/`.

This is an active recovery, not a funding-blocked state. The replacement uses
the exact tested source archive
from `serialization-r1`, whose manifest SHA256 is
`13a79cdd488e6c7b626f13e50f4e92412f8f165050e5411b26e931065e950156`.
No source changes are attributed to the new run revision.

The full accepted production round trip ran on all eight A100s
with `RAYON_NUM_THREADS=164` and no timeout. At launch, the test PID was 18419,
the wrapper PID 14568, and the success-dependent estimate queue PID 14569.
These PIDs are launch metadata, not a completion signal.

Production logs and VRAM samples are mirrored locally during execution under
`logs/rns-prf-a100/20260910/recovery-r1/`, with filenames beginning
`rxgmt08k3j3vab-8a100-`. The mirror was started in managed session 50849.
The source manifest was verified against the unchanged archived identity above.

Preprocessing completed in 24,953.029870474 seconds (6 h 55 min 53 s).
The reusable `online-checkpoint.json` was persisted in 1.09944795 seconds under
`crates/exponent-lut/test_data/rns_bgg_round_trip/.tmp5sBfBL/` on the pod.
The artifact directory measured 278,738,540,875 bytes, including its 32,062,905-byte
checkpoint. Online evaluation failed at UTC 2026-09-10 19:29:59 with exit 101.
The reported CUDA allocation failure occurred in `gpu_matrix_create` for a
1-by-10 matrix at CRT depth 6, during terminal branch 766. The last dispatch
elapsed time was 4,195.633825852 seconds; total test process time was 29,219.55
seconds. The error was not triggered by the advisory 80% target. These are
completed preprocessing and failed-online measurements, not a full correctness
result. The panic does not identify which internal CUDA allocation/event call
failed, so it does not establish a specific allocator root cause.

Full production correctness and final round-trip timing remain unestablished.
The matching original-source estimate completed; checkpoint backup and the
latest-source estimate are the current authorized work. Round-trip resumption
is explicitly deferred. Retrieve final evidence before pod cleanup.

### Online progress interpretation

The saved online graph has multiple loops; the first 2,048-element loop is not
all online evaluation. Its root contains 28,876 nodes. In this frozen graph,
owner 8203 constructs online masks, owners `16436 + 5k` for `k = 0..20` apply
the 21 cached PRF blocks, and owner 28833 evaluates 2,048 terminal readout
branches. Owner 28834 indexes the reduction inputs without matrix arithmetic;
owners 28835–28845 perform the balanced reduction, followed by extraction,
terminal modulus switching, output offset, and correctness-oracle arithmetic.
Host dispatch progress for one loop must not be reported as completion of the
entire online phase.

During a checkpoint inspection, sourcing the remote environment script
unintentionally triggered maintenance, including vim package upgrades and
OpenFHE symlink refreshes. The original production process and estimate queue
continued, and all 506 frozen source hashes were verified unchanged. The action
inventory is retained in
`logs/rns-prf-a100/20260910/recovery-r1/environment-inspection-incident.md`.
This incidental maintenance occurred during online evaluation; no timing
attribution or fully isolated measurement is claimed for that interval.
Subsequent inspections use direct Python SSH without sourcing that script.

The latest instruction authorizes checkpoint preservation and an estimate of
the current GPU-policy implementation, then explicitly defers round-trip
resumption. No checkpoint replay was started. The latest implementation is
built separately under `mxx-policy` while the original artifact store and source
remain unchanged. Its local validation is recorded in
`exponent-lut-gpu-policy-remediation.md`.

### Matching original-source estimate

The `recovery-r1` estimate passed with exit 0 in 403.29 seconds. Its model reports
15,701.718836460821 seconds preprocessing and 1,879.5868064099268 seconds online,
for 17,581.30564287075 seconds total (4.88 hours). Aggregate measured GPU work is
83,601.8691394952 and 6,722.086439572995 GPU-seconds, respectively. Those sums are
not wall-clock durations. Peak sampled calibration VRAM was 10,559 MiB on GPU 1;
it does not bound full-production residency. The actual preprocessing duration
was 24,953.029870474 seconds, and actual online execution failed before
verification. The estimate excludes the online correctness oracle. Parsed JSON
and both CSV rows were checked programmatically against the raw reports.

Checkpoint JSON has been copied locally and its SHA256 matches the remote file:
`1a8a55220067ea0f0dcf81c919d24b3044af913215cd42cbe79c73eba150f7e3`.
The complete artifact store is saved locally as eight compressed archive parts,
totaling 262,843,816,813 bytes. All eight transfer SHA256 values match the pod.
Archive verification confirms 24,934 regular files and 278,738,540,875 uncompressed
bytes, with exact per-part membership, no duplicate files, and successful
Zstandard decompression. The archives, checkpoint, and metadata were fsynced
before starting the latest-source estimate. Transfer took 2,455.0821429968346
seconds; archive verification took 70.25349171203561 seconds. Evidence is in
`backup-artifacts-transfer.json`, `backup-verification.json`, and
`checkpoint-preservation.json` under the `recovery-r1` log directory.

### Preserved checkpoint restore contract

Keep all eight `preprocessing-part-0.tar.zst` through
`preprocessing-part-7.tar.zst` files together with `backup-manifest.json`,
`backup-artifacts-transfer.json`, `backup-verification.json`, the source archive,
and the run logs under `logs/rns-prf-a100/20260910/recovery-r1/` (the source
archive is indexed separately above). Restoration needs space for approximately
279 GB of uncompressed data. Extract every part into the same destination;
each part contains the `.tmp5sBfBL` root and a disjoint subset of its files.
The JSON alone is insufficient: the entire artifact store must accompany it.

The restored checkpoint test reads `online-checkpoint.json` from that root and
uses the sibling `productions` tree. It reconstructs the original graph,
parameters, production identity, and bounds. Fresh online randomness is sampled;
preprocessing is reused. Transfer and archive verification establish faithful
preservation of the enumerated files, not successful cryptographic evaluation.
The user has explicitly deferred replay, so these restoration instructions do
not authorize starting it automatically.

### Latest-source estimate

The `policy-r1` estimate started at 2026-09-11 05:30:39 JST on eight A100s after
checkpoint backup verification. The source manifest is
`129a7a1f2cfaf7cbac21ae55dab04727e418389233f65cd2c31758aa6a85be0e`;
the A100 executable SHA256 is
`d6e472fff515689a3950000dd9a725d48645490d29fcb6f1e18dad1489c35d71`.
The isolated estimate-binary build passed without warnings in 2 min 41 s.
The accepted cryptographic report, 80% advisory parallelism target, 164 Rayon
threads, one warmup, and one measured calibration iteration are unchanged.
The existing allocator retention default is retained for this estimate.
No saved-checkpoint replay or full round trip has been started.

The latest estimate passed with exit 0 in 390.14 seconds. Its model reports:

| Component | Preprocessing seconds | Online seconds |
| --- | ---: | ---: |
| Primitive waves | 6316.780530552677 | 432.8775669080006 |
| Transfers and artifact I/O | 1619.9040403219997 | 522.9551666100002 |
| Dispatch proxy | 5.86473174362646 | 0.720961769463035 |
| **Estimated elapsed total** | **7942.549302618303** | **956.5536952874638** |
| Aggregate GPU work seconds | 32390.32002260518 | 1737.3159843180006 |

The total is 8,899.102997905768 seconds (2.47 hours), compared with
17,581.30564287075 seconds (4.88 hours) for the original-source estimate on the
same eight-A100 pod and configuration. The ratio is 0.5061684938919409. Estimator
source files are unchanged between these manifests. Each estimate uses one
measured calibration iteration, so this comparison does not isolate the effects
of individual policy fixes, prove a corresponding production speedup, or
attribute the observed GPU utilization. GPU work sums are not wall-clock time.
The peak sampled calibration residency was 10,559 MiB; production peak residency
and successful full-workload correctness remain unestablished for `policy-r1`.

The two latest metric rows and attempt row were appended to the existing CSVs
and checked programmatically against parsed raw reports. The complete raw log
and VRAM file were copied locally and matched their remote SHA256 values.
Evidence is under `logs/rns-prf-a100/20260910/policy-r1/`, including
`estimate-summary.json`, `estimate-reports.json`, `completed-log-integrity.json`,
`a100-estimate-comparison.json`, and `run-metadata.json`. The latest estimate did
not run the correctness oracle or regenerate production preprocessing.

### Final pod state and hold

At 2026-09-11 05:38:41 JST, after both estimates and complete artifact/log
preservation, pod `rxgmt08k3j3vab` was stopped. The API confirmed
`desiredStatus=EXITED`; the pod ID was retained. The container had no persistent
volume, so future restoration must rely on the verified local archives rather
than assuming container-disk retention. The local original and latest A100 test
binaries are retained with their source manifests and hashes. No checkpoint
replay or production round trip was restarted. Wait for the user's explicit
resumption instruction before starting either.
