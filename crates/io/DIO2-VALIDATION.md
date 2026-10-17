# DIO2 implementation verification

Implementation branch: `redesign/full`, original base `1830090b9`. Integration
branch: `codex/diamond-io2-roundtrip`; no push. Raw evidence is retained in the
task scratchpad `redesign/e3/`. Production capacity remains under investigation.

## Latest accepted toy GPU round trip

The RTX 4080 SUPER test passed all four clear-function equality assertions,
`1 passed; 0 failed`, with normal exit **0**. Framework duration was
**1,689.13 seconds (28m09s)**; process wall time was 1,690.24 seconds. The
45-minute timeout did not fire. This acceptance includes wide member tags and
the CUDA integer-zero encoding fix.

The toy retains two steps and all four inputs, n=4, Q=P=4, six 60-bit primes,
base 2^30 and twelve gadget digits. Its modeled worst-case error is 285 bits
against 300-bit spacing. It defaults to one backend I/O trial wave, honoring
an explicit environment override. W/C selection remains automatic.

| Stage | Seconds, including planning and trials |
| --- | ---: |
| Obfuscation | 407.892482285 |
| Evaluate false, false | 328.696769251 |
| Evaluate true, false | 317.56340612 |
| Evaluate false, true | 318.055239685 |
| Evaluate true, true | 316.271857182 |

Planned device payload peak: obfuscation **4,645,036 bytes**, evaluation maximum
**16,775,008 bytes**. The 564 NVML samples, at three-second intervals, recorded
**13,798 MiB device-wide**. This includes runtime/context memory missing from the
payload ledger, and is not an instantaneous peak measurement. Host peak RSS was
13,611,352 KiB. No assertions or inputs were removed to stay below 30 minutes.

Evidence: `codex-zero-tag-fixed-toy.log`, `codex-zero-tag-fixed-toy-vram.csv`,
`codex-zero-tag-fixed-source.patch`, `codex-zero-tag-fixed-binaries.sha256`, and
`codex-zero-tag-fixed-validation.sh`. Frozen toy SHA-256:
`f1acb1e8f797d3aa817fec5bc4754c72aa4e2a0fddab1c5e4503b8267087e845`.

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
production PRF does not fit that ring (`codex-full-plan-n64.log`). At n=128,
planning was rejected because Int artifacts require payload-size metadata. The
experiment was stopped after this structural failure (exit 143), rather than
repeating it for every column width. No physical full-graph capacity was obtained.
The existing `plan_with_store` path enumerates every member's size, so using it
unchanged would not be practical for the production family counts. The diagnostic
experiment remains in the task worktree/scratchpad and is not acceptance evidence.

Sampler evidence: `codex-native-alias-capacity.log` and
`codex-native-alias-capacity_retained_rows.log`. Full-plan diagnostic source,
binary hash, settings and logs use the `codex-full-plan-*` prefix. The least-change
choices were to reuse existing integer tag encoding and backend planning options.
No production end-to-end or whole-program 200 GB acceptance is claimed.
