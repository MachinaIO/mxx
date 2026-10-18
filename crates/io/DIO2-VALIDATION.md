# DIO2 implementation verification

Implementation branch: `redesign/full`, original base `1830090b9`. Integration
branch: `codex/diamond-io2-roundtrip`; no push. Raw evidence is retained in the
task scratchpad `redesign/e3/`. Production capacity remains under investigation.

## Current merge validation (2026-10-18)

The results below predate the merge of local `feat/exponent-lut` (`300d9124a`)
into `redesign/full` as `560ad24e2`. They are historical acceptance evidence,
not acceptance of the current merged implementation. The new full four-input
GPU round trip and final synchronization repetition gate are still pending.

Two backend regressions were diagnosed after the merge:

- Packed scalar family replay replaced a whole-family binding with a single
  member's allocation, causing `physical view exceeds allocation`. Commit
  `336deb43f` preserves contiguous family storage on replay.
- Nested parallel waves dropped their enclosing host-loop identity. Selected
  streamed-carry imports belonged to that host loop, but the executor checked
  root imports instead, so it could launch consumers before the requested data
  arrived. The first observed divergence in the captured evaluation was the
  first refresh's two-row concatenation (operation 2746, region 116), reading
  `dio2/initial.4` and `.5`. Commit `224034ad9` propagates the enclosing loop
  identity and avoids joining requests issued outside the current range twice.

The selected-import regression failed before the fix (exit 101, 0.23 seconds)
and passed after it. This test and three related nested-loop/carry tests each
passed three GPU executions: 12 passes, zero failures. CPU and CUDA workspace
library compile gates and the production toy compile were warning-free.
The saved failing evaluation passed with ordinary production pooling, operation
ordering and I/O trials: `dio2/state = 1`, one test passed, exit 0, 837.95 seconds
(838.90 seconds process wall time). This is not a replacement for a fresh full
round trip. That four-input run is now in progress, using the existing backend
`MXX_GPU_MAX_PARALLEL_INSTANCES=1` setting to avoid a known local W=2 planning
OOM; C remains automatically selected. Evidence uses the
`codex-selected-carry-*` task-log prefix.

## Historical accepted toy GPU round trip

The RTX 4080 SUPER test passed all four clear-function equality assertions,
`1 passed; 0 failed`, with normal exit **0**. Framework duration was
**1,647.73 seconds (27m28s)**; process wall time was 1,648.94 seconds. The
45-minute timeout did not fire. This acceptance includes the backend residency
changes and evaluator target/hint lane placement described below.

The toy retains two steps and all four inputs, n=4, Q=P=4, six 60-bit primes,
base 2^30 and twelve gadget digits. Its modeled worst-case error is 285 bits
against 300-bit spacing. It defaults to one backend I/O trial wave, honoring
an explicit environment override. W/C selection remains automatic.

| Stage | Seconds, including planning and trials |
| --- | ---: |
| Obfuscation | 404.169708261 |
| Evaluate false, false | 310.665406875 |
| Evaluate true, false | 310.291123144 |
| Evaluate false, true | 313.693374122 |
| Evaluate true, true | 308.2596357 |

Planned device payload peak: obfuscation **3,765,884 bytes**, evaluation maximum
**5,853,044 bytes**. The 550 NVML samples, at three-second intervals, recorded
**15,382,609,920 bytes (14,670 MiB) device-wide** across planning and execution.
During execution after preparation, obfuscation peaked at **3,082,813,440 bytes**
and evaluation at **6,197,084,160 bytes**. These device-wide readings include
context/driver/cache memory outside the payload ledger; sampling can miss shorter
peaks. Host peak RSS was 15,125,716 KiB.

The prior accepted version took 1,689.13 seconds and planned up to 16,775,008
evaluation payload bytes. The new payload is smaller, but the overall NVML
planning peak increased from 13,798 to 14,670 MiB; payload reduction must not be
reported as the same reduction in total VRAM. No assertions or inputs were
removed, and the unchanged toy remains below 30 minutes.

Evidence: `codex-hint-lanes-toy.log`, `codex-hint-lanes-toy-vram.csv`,
`codex-hint-lanes-toy-phase-vram.json`, `codex-hint-lanes-source.patch`,
`codex-hint-lanes-hashes.txt`, and `codex-hint-lanes-toy.sh`.
Frozen toy SHA-256:
`7646e2e3323a7e7ecf95d819504a7fa92675c8e35be7163880e5d97fd4b790e2`.

## Validation gates

- Workspace CPU units: **1,022 passed, zero failed, 21 ignored, 43 filtered**.
  Round-trip names were excluded; no CPU DIO2 round trip was run.
- IO tests include noise bounds, graph-size invariance and resident-family
  assertions. Initial private/state/hint outputs publish from parallel loops.
- After the CUDA zero fix, zero-integer, wide-integer and U64Le hash comparisons
  each passed five times in identity, `0,0` and `0,0,0` modes: **45 executions,
  zero failures**, including replay with two fresh hash keys per execution.
- The preceding backend memory/capture changes passed **5,475 GPU repetitions**
  (including 900 two-concurrent-lane sampler repetitions), with zero failures.
  These completed gates are separate from the latest 45 hash executions.
- CPU/GPU workspace library no-run and the final GPU toy build were warning-free.
  Rust was formatted with `cargo +nightly fmt --all`.
- Both full production graphs validate at n=65,536, 27x60-bit primes, base 2^6,
  64 levels and production label/part counts. This was graph-only validation.

Evidence: `codex-wide-member-cpu-units.log`, `codex-wide-member-cpu-build.log`,
`codex-zero-tag-fixed-gpu-build.log`, `codex-zero-tag-fixed-toy-build.log`,
`codex-zero-tag-fixed-results.txt`, `codex-shared-hash-final-results.txt`,
`codex-concurrent-workspace-results.txt`, and
`codex-wide-member-production-liveness.log`. Logical-device aliases do not
constitute validation on multiple physical GPUs.

## Implementation and diagnosed regressions

DIO2 exposes independent work through parallel loops. The generic backend chooses
concurrency and tile widths; no application concurrency controller was added.
PRF/CBD/smudge members are consumed where needed, retargeting processes individual
blocks, and initial and carried material is streamed through artifact families.
Key-only and row composition retain their shared helpers and noise recurrence.

The computational sampler reuses coefficient/evaluation tails after their final
readers, including an in-place NTT for the correction. Ordered equal-width tiles
share native P1/GQ/cutoff resources; covariance refresh explicitly depends on the
previous retry loop. R/E views share their stacked allocation. Allocation accounting
unions overlapping ranges rather than counting aliases repeatedly.

Fixed-shape public hashes are rematerialized inside parallel bodies. A failed
intermediate implementation memoized dead temporary node pointers and conflated
hash domains; a 32-domain regression reproduced it. Sealing original arguments
and caching only original node identities fixed it. Packed symbolic-width hashes
remain captures. The failed intermediate toy is not acceptance evidence.

Full production evaluation originally rejected member tag component 5 with value
`318893230573092864000` (69 bits). All three member-tag producers/readers now use
the existing Integer/Operand encoding through `HashTag::push_integer`, ensuring
constant, loop and runtime indices agree. Instance/domain U64Le tags are unchanged.
This changes member-tag bytes, so old hash-derived artifacts must be regenerated.
No new cryptographic hash encoding or truncation was introduced.

The first toy after that migration failed on its first evaluation
(`codex-wide-member-toy.log`, 747.10 seconds, exit 101). A focused GPU test reproduced
the cause: dynamic integer zero used an empty magnitude, while static/CPU BigInt
tags use one zero byte. CUDA now matches the host representation. The latest
45 comparisons and full four-input toy above validate the correction.

## Production capacity: unresolved whole-program peak

Nine native sampler/ownership smoke checks and two physical capacity diagnostics
passed at n=128, 27x60-bit primes, base 2^6, two anchors and W=C=1:

| Diagnostic | Planned bytes | n=65,536 projection with compact-width correction |
| --- | ---: | ---: |
| Sampler | 301,778,112 | 164,171,972,608 |
| Sampler plus eight retained caller rows | 361,508,216 | 194,753,785,856 |

The projection multiplies by 512 and adds 9,661,579,264 bytes for the production
preimage/cutoff width. Initial-output streaming removes 22,988,980,224 production
bytes from root return storage. These figures are payload estimates, not a
whole-program 200 GB certificate: context/driver graph metadata, other stages
and additional caller state are not covered by the synthetic sampler envelope.

The liveness diagnostic now inspects evaluation preimage artifact/family reads,
not only sampling nodes. Seven evaluation read sites appear in
`codex-evaluation-preimage-reads.log`. One enclosing step scope has
191,230,377,984 logical full-word bytes, but views/captures can overlap, bounded
values can be compact, and native scratch is additional. The logical sum does
not prove a physical peak in either direction.

A full-graph physical planning experiment used synthetic ordinary inputs with
explicit ranges: full label modulus for labels, [0,1] for this example's one-bit
state/chunks and Boolean step data. The n=64 attempt was rejected because the
production PRF does not fit that ring. The n=128 experiment exposed backend
metadata enumeration, expanded bit-position candidates, and scratch lifetime
issues. Family-wide artifact payload bounds, lazy captures, repeated wave
templates, direct device bit extraction, and scratch reuse now address those
issues. Import payload bounds are checked again when the payload is loaded.

This experiment is not a production execution or a capacity certificate. Its
latest exploratory binary included a subsequently removed native-body scratch
coalescing prototype. At W=1, C=135 its region pool occupied 4,188,727,296 bytes,
but CUDA graph compilation still failed. The search was explicitly stopped
(exit 143, 583.95 seconds). Do not extrapolate that failed candidate into an
accepted production peak. Allocation-specific extrapolation and small component
measurements take priority over further full-shape OOM searches.

Sampler evidence: `codex-native-alias-capacity.log` and
`codex-native-alias-capacity_retained_rows.log`. Full-plan diagnostic source,
binary hash, settings and logs use the `codex-full-plan-*` prefix. The least-change
choices were to reuse existing integer tag encoding and backend planning options.
No production end-to-end or whole-program 200 GB acceptance is claimed.

## Backend memory changes validated by the latest toy

The backend retains automatic W/C selection. DIO2 introduces no application-specific
concurrency controller. The current work also includes:

- Family-wide Int/TypedBlob artifact payload bounds, enforced at import, so a plan
  need not enumerate metadata for every member.
- Lazy integer artifact captures and repeated streamed wave templates, including
  nested Zip imports, with actual member indices preserved on replay.
- Device-side dynamic bit extraction with BigInt-compatible sign extension,
  without a candidate array proportional to the bit-position range.
- Deferred sequential carries, allocation-by-allocation rebinding of composite
  views, and explicit liveness for indexed pointer-table reads.
- Scratch reuse within host loops while preserving the whole lifetime of wave
  staging that can overlap replays.

Four new GPU tests compare short and long chains, check results against ordinary
matrix arithmetic, and assert bounded memory growth. They do not execute a CPU
DIO2 round trip. Broader testing found a regression when arena lifetimes were
shortened inside waves: streamed carry and read-ahead tests failed 21 of 330
repetitions. Restoring the whole-wave lifetime while retaining host-loop-local
reuse fixed the focused seven tests and all 330 subsequent repetitions across
identity, two-logical-device, and three-logical-device configurations.
Evidence: task logs `codex-wave-arena-smoke.log`,
`codex-wave-arena-{identity,double,triple}.log`, and
`codex-wave-arena-results.txt`.

The warning-free CPU unit run passed 1,022 tests (21 ignored, 43 round-trip-name
filters excluded) in nine suites. The warning-free CPU/GPU workspace compile gates
also passed. Both n=128 sampler diagnostics were repeated on the updated backend
and retained the allocation totals in the table above. Evidence: task logs
`codex-wave-arena-cpu-units.log`, `codex-wave-arena-workspace-build.log`,
`codex-wave-arena-capacity.log`, and `codex-wave-arena-capacity_retained_rows.log`.

The three affected replay/read-ahead tests also passed 300 repetitions each in
each of the identity, two-logical-device, and three-logical-device modes:
2,700 executions, zero failures. Evidence: `codex-wave-arena-sync-progress.log`,
`codex-wave-arena-sync-{identity,double,triple}.log`, and
`codex-wave-arena-sync-results.txt`.

Whole-program production capacity accounting remains unresolved; the latest
successful toy above validates the backend and application changes, not a
production end-to-end execution.

The evaluator constructs temporary targets inside refreshed-component lanes,
matching the obfuscator, and computes hint anchor parts as a parallel family with
the existing disjoint-column sum. All 22 DIO2 unit tests passed after these
changes, including graph-size invariance, residency, and noise checks. Both
full-production graphs validated. The largest enclosing evaluation parent
scope in the preimage-read diagnostic decreased from 191,230,377,984 to
91,913,453,568 logical full-word bytes. This is a liveness comparison, not a
physical allocation peak: captures and views can alias, and native workspaces
must be added separately.

Final application gates: `codex-hint-lanes-units.log`,
`codex-hint-lanes-{cpu,gpu,toy}-build.log`, and
`codex-hint-lanes-production-liveness.log`. Backend code was unchanged after
the 330 + 2,700 successful GPU checks. The final application graph was then
tested by the complete four-input GPU round trip.
