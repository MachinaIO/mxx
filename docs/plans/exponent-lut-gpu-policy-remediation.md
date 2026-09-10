# Branch-path GPU policy remediation

This follow-up addresses the four GPU.md findings reported during the A100
`recovery-r1` production run. The original Runpod production attempt and its matching estimate
used the immutable `serialization-r1` source manifest
`13a79cdd488e6c7b626f13e50f4e92412f8f165050e5411b26e931065e950156`.
The changes described here are a separate revision; results from the original
run do not validate these later changes.

## Scope and implementation status

- Arithmetic batch metadata: replace per-call matrix-by-limb host pointer tables
  with persistent allocation descriptors and device-side address calculation.
  The implementation uses one bounded submission packet naming matrix
  descriptors, with allocation/event ownership retained through kernel use.
  Local GPU validation passed; A100 round-trip execution remains deferred; estimate-only execution is authorized.
- Fleet export: group borrowed shards by device, invoke the existing batch
  serialization API within each device, and run device groups concurrently.
  Bound temporary export storage by `MXX_CUDA_STREAM_POOL_SIZE`. Preserve the
  original shard order, signed-magnitude widening, and source lifetimes through
  all transfer completion waits. Local GPU validation passed; A100 round-trip execution remains deferred; estimate-only execution is authorized.
- Calibration misses: move input completion, pool-baseline fencing, and pilot
  completion into explicit preparation boundaries outside arithmetic wrappers.
  Retain shared-context rejection, context-generation checks, peak accounting,
  contamination retries, and capacity checks. The executor stages the exact
  node inputs, completes preparation explicitly, and dispatches production once.
  A regression checks that preparation preserves input values and the sampling
  transcript. Local GPU validation passed; A100 round-trip execution remains deferred; estimate-only execution is authorized.
- GPU Rust filename: rename `backend/poly_gpu/fleet.rs` to
  `backend/poly_gpu/fleet_gpu.rs` within `crates/runtime/src`, and update the
  module declaration. Local implementation is present.

## Validation contract

Build CPU and GPU workspace library tests without warnings. Run focused GPU
checks repeatedly on the local GPU without disturbing the live A100 job. The
user has now authorized an estimate of the latest implementation on this A100
pod, after checkpoint preservation. Production-checkpoint replay remains
explicitly deferred. Use trusted arithmetic and
serialization oracles, including mixed signed coefficient widths and multiple
shards per device. Compare matched before/after timings without claiming that
these policy conflicts explain the observed utilization.

## Local validation results

Both CPU and GPU workspace library test builds completed without warnings.
The PRF integration-test binary also compiled without warnings. On one local
RTX 4080 SUPER, five focused checks each passed 300 consecutive executions
(1,500 total, zero failures): batched arithmetic against scalar operations,
multi-product multiply-accumulate against trusted CPU primitives, mixed signed
width export/reload, executor input/transcript preservation during preparation,
and runtime calibration miss followed by cache-hit execution. The wide-CRT
multiply-accumulate test passed three runs, and all 22 tests in the fleet module
passed in each of three runs. The local machine has one GPU; peer-device branches
of those tests are not multi-device evidence.

Three small RNS-BGG PRF round trips (ring dimension 4, requested security exponent
1, average-case noise) passed preprocessing, artifact reopen, online execution,
and correctness checks. These smoke parameters do not validate the frozen
production cryptographic workload. The saved small checkpoint also passed three
separate-process online replays with cold calibration profiles.

The matched batch timing probe used ring dimension 512, four 40-bit CRT primes,
batch size 8, inner dimension 16, eight output columns, and two products. Each
run warmed up once and timed one production batch through GPU output completion;
CPU-oracle validation was outside the timer. Five independently randomized runs
had median times 0.002643629 seconds before and 0.002318326 seconds after. Ranges
were 0.002281886–0.005992949 and 0.001707453–0.003896546 seconds, respectively.
This small, variable local measurement does not establish a production speedup
or explain the previously observed GPU utilization.

Local evidence is under `logs/rns-prf-a100/20260910/policy-r1/`. The tested source
manifest SHA256 is
`129a7a1f2cfaf7cbac21ae55dab04727e418389233f65cd2c31758aa6a85be0e`.
The user has subsequently authorized the latest-source estimate only and
instructed that round-trip execution must not resume after checkpoint saving
and estimation. The isolated A100 estimate-binary build passed without warnings in 2 min 41 s.
The estimate passed after complete checkpoint backup verification, with exit 0
in 390.14 seconds. Its modeled one-output total is 8,899.102997905768 seconds
(2.47 hours), compared with 4.88 hours for the original source on the same pod.
This is calibration/estimate evidence, not full production runtime or a
multi-device correctness regression suite. The frozen estimator source is
unchanged; the comparison does not attribute changes to individual fixes. No latest-source A100 round trip or checkpoint replay
has been started. The original-source estimate passed in 403.29 seconds;
its modeled workload timing is recorded separately in the A100 report.
