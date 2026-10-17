# DIO2 implementation verification

Source branch: `redesign/full`, base `1830090b9`. Logs are retained in the task scratchpad `redesign/e3/`.
Toy correctness and the listed gates passed. Production capacity is not certified.

## Current implementation

- DIO2 exposes independent instance, target, component and preimage work as parallel loops. The generic backend chooses W/C; no application concurrency controller was added.
- Deferred PRF/CBD/smudge members and block retargeting avoid parameter-sized resident families. Reductions and streaming steps remain sequential graph loops.
- Backend fixes cover scalar families, streamed carries, nested parallel outputs, replica storage identity, singleton selections, covariance-only trapdoor conversion, shared trapdoor leaves and correct wave rebinding.
- Preimage sampling publishes directly to deferred compact outputs, initialized at their first graph use. Computational perturbation temporaries are reused after their last reads. Ordered tiles share cutoff staging.
- Wave placeholders retain overlapping views of each original allocation. Rebinding keeps the widest replacement range. The memory ledger counts the union of device ranges.

## Completed checks

- CPU workspace libraries: 1,017 tests passed, 21 ignored, 43 filtered. Round-trip names were excluded; no CPU DIO2 round trip was run.
- Latest IO default unit tests: 22 passed, one ignored.
- CPU/GPU workspace no-run and nightly formatting gates have passed on successive revisions; the final CPU compilation also passed after the latest GPU-only changes.
- Before the latest memory fixes, 1,875 targeted GPU repetitions passed with no failures.
- Latest computational, tiled and statistical sampler smoke checks: nine passed across identity, `0,0`, and `0,0,0` logical-device modes. The trapdoor regression now checks four columns, two replays, and placeholder R/E aliasing.
- Allocation-range union unit regression passed, including overlap, nesting, adjacency and overflow.
- Final required GPU repetition set: **2,790 executions passed, zero failures**, across identity, `0,0`, and `0,0,0`. Three regression filters each ran 300 times per mode; six smoke filters each ran five times per mode. Binary identities and results are recorded in `codex-vram-final-*`.

## Round trip

The superseded four-prime/base-15 toy completed obfuscation in 616.145268758 seconds and its first evaluator input in 425.14830615 seconds. That input matched the clear function. The run was stopped during the second input after 1,110.29 seconds because it lacked the latest memory fixes and projected total duration exceeded 30 minutes. This is not a complete round-trip pass.

The latest toy retains two steps and all four evaluator inputs, with n=4, Q=P=4, six 60-bit primes and base 2^30 (12 gadget digits). Its worst-case modeled error is 285 bits against 300-bit spacing. The final GPU run passed all four clear-function equality assertions on an RTX 4080 SUPER. The framework reports `1 passed; 0 failed`, 2,066.01 seconds (34m26s). Obfuscation took 558.953170327 seconds; evaluations took 379.614344808, 370.832055963, 385.516062951 and 370.512698793 seconds, respectively. These times include planning/trials. Planned peak: obfuscation 4,801,284 bytes, evaluation maximum 16,625,728 bytes. NVML device-wide sampled peak was 13,768 MiB; it includes runtime/graph/context allocations excluded from the planned payload ledger.

The test process exited 0, confirmed by its kernel wait status in `codex-vram-final-test-exit.json`. Its original 30-minute GNU timeout was suspended and replaced with a cancellation-aware 45-minute watchdog to preserve all four inputs. The old wrapper subsequently reported 124 from its queued timer; this is distinct from the successful test process. The replacement watchdog exited 0. Raw logs, binary hashes and 3-second NVML samples are in `codex-vram-final-*`.

The smaller parameters reduced the gadget width from 16 to 12, but the complete run still exceeds 30 minutes. No inputs or assertions were removed. The resident-family audit and graph-size invariance assertions both pass.

## Production capacity: measured subtotal, not whole-program certification

The production-shape sampler diagnostic uses 27 60-bit primes, base 2^6, 270 gadget digits, two anchors and two refresh chunks selecting anchor `instance / 2`, with W=C=1. Only ring dimension is reduced from 65,536 to 128. Latest planned allocation ledger: **350,368,960 bytes** (`codex-deferred-preimage-production-capacity.log`).

Scaling by 512 gives 179,388,907,520 bytes. Conservatively adding one byte per coefficient for the production compact preimage (bound grows from 22 to 27 bits) and cutoff staging gives **189,050,486,784 bytes**. Two additional full rows would add 7,644,119,040 bytes, giving 196,694,605,824 bytes, but this allowance is not yet a proof of all DIO2 live values. CUDA context/driver graph metadata and artifact storage are outside the plan ledger. The whole-program 200 GB peak remains unverified.

The full production evaluation graph also reaches an existing u64 hash-tag index limit. The n=128 graph-only diagnostic constructs both graphs, but does not establish executable production parameters or practical host/artifact volume.

## Remaining

Whole-program production peak remains unverified: the sampler-only estimate leaves about 11 GB, without a proved allowance for all additional live values and runtime overhead. Do not use this report as a production 200 GB acceptance certificate. Full production graph validation is also blocked by the existing u64 hash-tag index limit. The least-change decision is to retain the hash format rather than introduce an unvalidated cryptographic format redesign. No production end-to-end run or multiple-physical-GPU validation is claimed.

The implementation is based on `redesign/full`. After the successful toy, it is committed there and merged into a new `codex/diamond-io2-roundtrip` branch in the chat worktree; no push is performed.
