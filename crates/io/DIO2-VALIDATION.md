# DIO2 implementation verification

Source branch: `redesign/full`, original base `1830090b9`. The previous accepted baseline is `7c2eefaf3`. Logs are retained in the task scratchpad `redesign/e3/`. The implementation is integrated into `codex/diamond-io2-roundtrip` in the chat worktree without a push.

## Accepted toy GPU round trip

The final RTX 4080 SUPER run passed all four clear-function equality assertions: `1 passed; 0 failed`, normal exit **0**. Framework duration was **1,679.23 seconds (27m59s)**; measured process wall time was 1,680.32 seconds. The 45-minute timeout did not fire.

The toy retains two steps and all four evaluator inputs, with n=4, Q=P=4, six 60-bit primes and base 2^30 (12 gadget digits). Its worst-case modeled error is 285 bits against 300-bit spacing. The backend automatically chooses W/C; the test defaults to one I/O trial wave and honors an explicit environment override. No production application concurrency controller was added.

| Stage | Seconds, including planning/trials |
| --- | ---: |
| Obfuscation | 404.175555581 |
| Evaluate false, false | 317.829052693 |
| Evaluate true, false | 319.691199763 |
| Evaluate false, true | 317.317860902 |
| Evaluate true, true | 319.577104527 |

Planned device payload peak was 4,642,348 bytes for obfuscation and 16,762,624 bytes for evaluation. The 561 NVML samples, taken every three seconds, recorded a **device-wide peak of 13,798 MiB**. This includes context/runtime allocations absent from the planned payload ledger and is not a precise instantaneous peak. Host peak RSS was 13,608,884 KiB.

Evidence: `codex-initial-stream-final-toy.log`, `codex-initial-stream-final-toy-vram.csv`, `codex-initial-stream-final-source.patch`, `codex-initial-stream-final-binaries.sha256`, and `codex-initial-stream-final-validation.sh`. Frozen toy SHA-256: `f67d88c7ff0edadd7e45e6e1bb241151defd4c4761804e30f6ec3ab4be05f9c8`.

The baseline took 2,066.01 seconds (34m26s); its separate logs remain in `codex-vram-final-*`. The latest full run is below 30 minutes without removing inputs or assertions. Earlier interrupted and failed runs are not acceptance evidence.

## Validation gates

- CPU workspace library tests after the generic hash fix: **1,020 passed, zero failed, 21 ignored, 43 filtered**. Round-trip names were excluded; no CPU DIO2 round trip was run.
- Subsequent IO changes: **22 passed, zero failed, one ignored**, including noise, residency and graph-size invariance checks.
- GPU regression repetitions: **5,475 passed, zero failures** across identity, `0,0`, and `0,0,0` logical-device modes. This includes 4,575 regression/smoke repetitions and 900 two-concurrent-lane sampler repetitions. Logical-device aliases are not multiple physical GPUs.
- Nine native sampler/ownership smoke checks and both capacity diagnostics passed.
- CPU/GPU workspace library no-run builds and the final toy harness build passed without warnings. Rust was formatted with `cargo +nightly fmt --all`.

Evidence: `codex-hash-identity-fixed-cpu-units.log`, `codex-initial-stream-final-io-units.log`, `codex-shared-hash-final-results.txt`, `codex-concurrent-workspace-results.txt`, and the corresponding scripts/logs. The 2,790 baseline GPU repetitions are separate from the 5,475 follow-up executions.

## Implementation and production-capacity limits

DIO2 exposes instance, target, component, initial-material and preimage work through parallel loops. The generic backend selects concurrency and tile widths. Deferred families and artifact streaming avoid keeping all members resident; reductions and state-carrying steps remain sequential. The following changes reduce sampler memory and root output lifetimes.


The computational sampler samples `p2` directly into the perturbation coefficient tail and uses the evaluation tail for its temporary transform. After the public product, `z` uses the coefficient tail and an in-place NTT alias; correction consumes it before the final inverse overwrites the owner. Equal-width ordered tiles share P1/GQ/cutoff resources, with an explicit previous-tile dependency on covariance refresh.

Fixed-shape public hashes are rematerialized inside parallel bodies, and DIO2 constructs targets inside decoded-key lanes. Parameter-sized packed hashes remain captures. A regression in the first rematerialization implementation memoized dead temporary pointer identities, conflating distinct hash domains. A 32-domain graph test reproduced that failure; sealing arguments directly fixes it. The failed follow-up toy (`codex-p2-final-toy.log`, 1,261.21 seconds) is not acceptance evidence. A nested GPU hash oracle now compares two distinct domains, dynamic outer tags and two fresh input keys against root hashes.

Initial private/state/hint outputs now use singleton parallel artifact families. Evaluation selects member zero into the unchanged carried state. The residency audit asserts that every initial output publishes from a parallel loop. This removes four full rows and the full-width hint from root return storage during refresh sampling. Both n=128 and the full n=65,536 obfuscation liveness diagnostics now retain only one scalar polynomial at the root, in addition to streamed families; the refresh caller retains eight full rows. The eliminated S_0 root-return payload is 22,988,980,224 production bytes. Logical aliases are not added as independent allocations.

Native sampler smoke checks passed nine times across identity/double/triple logical-device modes. Two physical capacity diagnostics passed with n=128, 27x60-bit primes, base 2^6, two anchors and W=C=1:

| Diagnostic | Planned bytes | n=65,536 projection, including compact-width correction |
| --- | ---: | ---: |
| Sampler | 301,778,112 | 164,171,972,608 |
| Sampler plus eight retained caller rows | 361,508,216 | 194,753,785,856 |

The projection multiplies by 512 and adds 9,661,579,264 bytes for the production preimage/cutoff width. The second diagnostic explicitly retains eight rows through sampling; it is a synthetic envelope, not a complete production plan. It excludes CUDA context/graph metadata and additional caller state. Whole-program 200 GB acceptance remains unverified.

CPU workspace tests after the hash identity repair passed (1,020 passed, zero failed, 21 ignored, 43 filtered); the subsequent IO-only initial-material change passed all 22 IO unit tests (one ignored), including noise, graph-size invariance and residency assertions. The 4,575 updated GPU repetitions passed with zero failures across identity/double/triple logical-device modes. A two-concurrent-lane smoke and 900 repetitions (300 per mode) also passed, for 5,475 successful GPU repetitions in this follow-up. The final all-input toy passed; see the acceptance results above. Binary hashes and exact commands are retained in the task scratchpad.

The toy now defaults to one I/O trial wave through the existing backend option,
while honoring `MXX_GPU_IO_TRIAL_WAVES` overrides. The frozen acceptance binary
used that same setting via the environment; the subsequent test-only default
change does not change the exercised configuration or any production code. W/C
selection, actual loop counts and the four equality assertions remain unchanged.

## Remaining production limitation and design decision

The **194,753,785,856-byte projection is a sampler-plus-eight-rows payload envelope, not a whole-program 200 GB certificate**. It excludes CUDA context/graph metadata and additional live state. The full n=65,536 obfuscation graph was constructed and its initial-output liveness inspected, but the full evaluation graph fails at the existing `little-endian hash tag must fit in u64` limit (nested scope node 445). The n=128 graph-only diagnostic constructs both graphs. Neither establishes practical production artifact volume or production end-to-end execution.

Evidence: `codex-initial-stream-production-liveness.log`, `codex-initial-stream-liveness.log`, `codex-native-alias-capacity.log`, and `codex-native-alias-capacity_retained_rows.log`.

The least-change decision is to preserve the existing hash transcript format instead of silently introducing an unvalidated wider cryptographic format. Whole-program production capacity remains unverified. No production end-to-end or multiple-physical-GPU acceptance is claimed.

### Production hash-tag diagnostic

A follow-up graph-only run at the full parameters localized the failure to tag
component 5, evaluated as `318893230573092864000` (69 bits), exceeding `u64`.
The validator now reports the component and value, while preserving the same
acceptance rule and hash bytes. Boundary checks cover -1, 2^64-1 and 2^64;
the existing runtime-operand type check also passes. Evidence:
`codex-production-hash-tag-value.log` and `codex-hash-tag-diagnostic-unit.log`.
The rejected value must not be truncated. The backend already has an integer
hash-tag encoding distinct from U64Le; determining whether all member-tag
producers/readers can migrate together remains separate from capacity acceptance.
