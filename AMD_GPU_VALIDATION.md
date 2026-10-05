# AMD GPU implementation validation

Current checkpoint: source `03c5c0719` (see "Cache-release synchronization and final-source
gates" below). `gpu_device_release_cached_memory` now synchronizes the context streams before
its whole-device allocation probe. With that change, the native library reproduction, the root
addition, the parallel-wave, and the named/borrowed rebind tests report zero use-after-free.
The nested control-region test still reports use-after-free, matching a Compute Sanitizer
behavior that standalone CUDA programs reproduce without mxx. The final source passed
warning-free CUDA and HIP gfx1100 workspace GPU compile gates. It also passed the 300-iteration
GPU unit gate in identity, `0,0`, and `0,0,0` modes with zero failures. AMD runtime remains
unverified because no AMD GPU is available.

Previous checkpoint: diagnostic owner-trace scaffolding has been removed from seven code files,
with ownership/event protections retained. Root applied `owner-trace-cleanup.patch`, ran nightly
formatting and whitespace checks, and verified that owner-trace and temporary diagnostic
references are absent from code. `source-manifest.traceclean-v17.json` identifies the cleaned
source. All four warning-free full-workspace lib compile gates passed: CPU, CUDA,
HIP gfx1100, and HIP gfx942 (`traceclean-v17-*-result.json`). The cleaned CUDA
source passed 27 nonempty targeted checks (three cases, three logical modes,
three repetitions each; `traceclean-v17-cuda-checks-results.json`). The cleaned
root-add test passes numerically but still reports 16 memcheck errors, so memory
acceptance is not established. V16 compile, functional, and owner-trace results
below are historical evidence. Memory acceptance remains unresolved despite bounded
standalone reproduction, and AMD runtime is unverified because no AMD GPU or CI is available.
TFHE on AMD remains deferred. Formal implementation reviews have not started.

## Evidence location and environment

The external runtime artifact directory is `/tmp/mxx-amd-validation`. It is outside the
repository and must be retained separately if these results are to remain reproducible.
Filenames below are relative to that directory, not repository links. `source-manifest.json`
records per-file SHA-256 hashes for the tested snapshot; `source-manifest.postplanner.json`
records the subsequent isolated matrix planner loop-site correction; `native-source-diff.patch` records the
native changes. Subsequent edits require affected checks and an updated source record.

The HIP compile environment used ROCm 7.0.0 extracted under the artifact directory's `sdk/`,
with separate `target-hip-gfx1100` and `target-hip-gfx942` directories. These are compilation
targets, not validated AMD hardware. NVIDIA device checks used an RTX 4080 SUPER, SM 89, with 16,376 MiB VRAM and driver
580.178.04. `validation-environment.json` records nvcc 13.1.80, HIP 7.0.51831/Clang 20,
Rust 1.97.1, device identity, and validation environment.
The host has Ryzen 7950X3D Raphael graphics, but access to `/dev/kfd` was denied outside the
sandbox; this device is not the selected officially supported ROCm validation target.
No AMD device test result is inferred from its presence.

## Results recorded so far

| Gate | Evidence | Result and limits |
| --- | --- | --- |
| CPU release workspace lib compilation | `cpu-compile.log`, `cpu-final-compile.log` | Passed |
| CPU release workspace lib tests | `cpu-unit.log` | 629 passed, zero failures; ignored tests excluded |
| HIP release workspace GPU lib compilation, `gfx1100` | `hip-gfx1100-compile.log` | Passed without warnings; seven workspace lib binaries built |
| HIP release workspace GPU lib compilation, `gfx942` | `hip-gfx942-compile.log` | Passed without warnings; seven workspace lib binaries built |
| CPU-only tests in HIP-built binaries | `hip-host-results.json`, `hip-host-*.log` | Seven binaries, 652 passed, zero failures; GPU tests remained ignored |
| HIP dynamic linkage | `dynamic-linkage-results.json`, `hip-*-dynamic-linkage.log` | `amdhip64` dependency present; CUDA dependencies absent; this is linkage evidence only |
| Invalid backend, invalid HIP arch, missing HIP SDK | `negative-build-results.json` and corresponding logs | All three failed as expected, exit 101, expected diagnostic markers present |
| CUDA release workspace GPU lib compilation | postplanner CUDA/HIP compile logs | Postplanner code passed for CUDA, HIP gfx1100, and HIP gfx942 without warnings/errors |
| Final CUDA backend host tests | `cuda-postplanner-results.json`, `postplanner-*-host.log` | 229 passed, zero failures |
| Final HIP backend host tests | `hip-host-backends-final.log` | Latest HIP rebuild: 229 passed, zero failures; HIP CPU-only total remains 652 after modifying an existing pure binding test |
| Final CUDA backend device smoke | `cuda-postplanner-results.json`, `postplanner-*.log` | Three repetitions each: identity 70 cases, `0,0` 71 cases, `0,0,0` 71 cases; zero failures |
| Final CUDA FHE and KHE smoke | `cuda-postplanner-results.json`, `postplanner-*.log` | Three repetitions each: FHE five cases, KHE three cases; zero failures; long ring-GSW test excluded |
| CUDA nested named-subgraph/control regression | `cuda-final-repeat-results.json`, `final-cuda-nested-300-identity.log` | Identity, `0,0`, and `0,0,0`: 300 completed each, 900 total, zero failures; before the isolated matrix planner fix, unaffected integer fixture |
| Long KHE regression probe | `baseline-khe-long.log`, `final-cuda-khe-long-probe.log` (prior failures); `postplanner-cuda-khe-long-probe.log` (latest terminal failure) | Previous probe failed at child node 50; loop-site correction is compiled and matrix named-call regression passes; latest long probe failed at measured resource admission |
| CUDA dynamic linkage | `dynamic-linkage-results.json`, `cuda-dynamic-linkage.log` | Runtime is statically linked; absent `cudart` NEEDED entry is expected |
| Before-change CUDA BGV smoke baseline | `baseline-bgv-results.json`, `baseline-bgv-*.log` | Five executions passed; no speed or regression conclusion from pass counts |
| After-change CUDA BGV smoke timing | `cuda-postplanner-results.json` | Five passed; total whole-unit wall time 1.783960 s, mean 0.356792 s; before-change mean 0.367226 s |
| Same-target HIP → CUDA → HIP rebuild | `backend-switch-results.json`, `backend-switch-{0-hip,1-cuda,2-hip}.log` and recorded metadata | All three offline release workspace GPU lib compile stages exited zero; backend/FHE cfg, revisions, and linkage followed selection |
| AMD device execution | None | Incomplete |

The compile commands were release `cargo test --workspace --lib --no-run`, with
`--features gpu` and `MXX_GPU_BACKEND=cuda` or `MXX_GPU_BACKEND=hip` for GPU builds; HIP
additionally selected `HIP_ARCH=gfx1100` or `gfx942`, its extracted SDK, and an isolated target
directory. GPU warning enforcement preserves the repository OpenFHE rpath. Device suite logs
and JSON results contain the executed binaries and statuses. The baseline JSON contains its
exact test filter, environment, five exit statuses, and wall times. These compilation and
unit checks do not include integration tests.

## Subsequent alias repair: current gate

The earlier compile and smoke passes above belong to their recorded snapshots. They do not
validate the subsequent alias repair. `source-manifest.aliasfix-v4.json` records the v4
443-path snapshot; three production files changed from v3. Subsequent v5/v6 snapshots are
recorded in `source-manifest.aliasfix-v5.json` and `source-manifest.aliasfix-v6.json`.

| Stage | Evidence | Result |
| --- | --- | --- |
| Alias fixture v1 | `aliasfix-compile-results.json`, per-backend logs | CUDA and HIP gfx1100/gfx942 compile failed, exit 101: four E0308 tuple-family destructuring errors in the new fixture; no device execution |
| Alias fixture v2 | `aliasfix-v2-compile-results.json`, per-backend logs | All three compile gates failed, exit 101: two E0271 usize/i32 inference errors in fixture input generation; no device execution |
| Alias fixture v3 | `aliasfix-v3-cuda-compile-result.json`, `aliasfix-v3-cuda-alias-probe.json` | CUDA compilation passed; one exact alias probe failed admission for missing body producer at output 1 of NodeId(2) |
| Alias production repair v4 | `aliasfix-v4-cuda-compile-result.json`, `aliasfix-v4-cuda-compile.log` | CUDA release workspace GPU lib compilation passed; HIP has not yet compiled this repair |
| Alias production repair v4 probe | `aliasfix-v4-cuda-alias-probe.json`, `aliasfix-v4-cuda-alias-probe.log` | One exact probe, exit 101: `GPU loop member has multiple storage owners`; no successful execution or 300-run set |
| Alias production repair v5 | `aliasfix-v5-cuda-compile-result.json/log`, `aliasfix-v5-cuda-alias-probe.json/log` | CUDA compilation passed; one exact probe failed graph compilation: child wave body not strictly inside parent |
| Alias hierarchy repair v6 | `aliasfix-v6-cuda-compile-result.json/log`, `aliasfix-v6-cuda-alias-probe.json/log` | CUDA compilation passed; exact probe plan succeeded, then execution returned `GPU rebound view exceeds its replacement allocation` |
| Alias rebound repair v7 | `aliasfix-v7-cuda-backends-compile-result.json/log`, `aliasfix-v7-cuda-alias-probe.json/log` | CUDA mxx-backends lib only compiled; one exact probe executed but failed synchronization with illegal memory access |
| Alias v7 memcheck | `aliasfix-v7-cuda-alias-memcheck.json/log` | Completed within 60-second bound in 1.319 s, exit 99; first invalid 8-byte write in `raw_matrix_add_sub_kernel`; 189 reported errors include cascading API failures |
| Alias repair v8 | `aliasfix-v8-cuda-backends-compile-result.json/log`, `aliasfix-v8-cuda-alias-probe.json/log` | Backend-only CUDA compile passed; exact probe planned and executed twice on GPU, then failed at CPU oracle `CPU family output` |
| Alias v8 memcheck | `aliasfix-v8-cuda-alias-memcheck.json/log` | Exit 99; no invalid global-access reports, but 104 errors including approximately 96 printed use-after-free reports and four allocation API errors; not clean |
| Alias v9 symbol build | `aliasfix-v9-cuda-symbols-compile-result.json/log` | CUDA backend-only compile passed with `debug=line-tables-only`, `strip=none` |
| Alias v9 exact memcheck/probe | `aliasfix-v9-cuda-symbols-memcheck.json/log` | 60-second bounded run exited 99 after 1.869 s; 312 errors and genuine rebound-result mismatch, GPU 2 versus CPU 22 |
| Alias v10 symbol build | `aliasfix-v10-cuda-symbols-compile-result.json/log` | CUDA backend-only retained-symbol compile passed |
| Alias v10 exact memcheck/test | `aliasfix-v10-cuda-symbols-memcheck.json/log` | Six-output CPU/rebind/held-output alias test passed once; sanitizer harness exited 99 with 309 errors, not clean |
| Alias v11 symbol build and exact test | `aliasfix-v11-cuda-symbols-compile-result.json/log`, `aliasfix-v11-cuda-symbols-memcheck.json/log` | Retained-symbol backend-only compile passed; six-output CPU/rebind/held-output exact test passed once; sanitizer exit 99, 237 errors |
| Alias v12 budget accounting | `aliasfix-v12-cuda-symbols-compile-result.json/log`, `aliasfix-v12-backends-host.json/log`, `aliasfix-v12-cuda-alias-probe.json/log` | Backend-only symbol compile passed; 229 host tests passed, 72 ignored; one exact functional alias test passed; no clean memory/repetition gate |
| Alias v13 owner-trace diagnostic | `aliasfix-v13-cuda-symbols-compile-result.json/log`, `aliasfix-v13-cuda-symbols-memcheck.json/log` | Backend-only symbol compile passed; exact alias test passed once; memcheck still exit 99 with 237 errors; tracing is diagnostic, not an ownership repair |
| Alias v14 producer/retirement protocol | `aliasfix-v14-cuda-symbols-compile-result.json/log`, `aliasfix-v14-cuda-symbols-memcheck.json/log` | Retained-symbol backend-only CUDA compile and one six-output exact functional test passed; memcheck unchanged, exit 99 with 237 errors |

The exact probe is
`gpu_physical_control::tests::test_gpu_parallel_named_borrowed_matrix_outputs_rebind_and_hold`,
run with `--exact --ignored --nocapture --test-threads=1` in identity mode. V4 connected the
production provenance/binding repair but exposes the next ownership-contract rejection.
V5 then reached graph compilation and rejected the child hierarchy. V6 repaired equal-range
explicit hierarchy: planning now succeeds, but its single probe fails during execute at
`crates/backends/src/gpu_physical_control.rs:9635` with the rebound-view allocation error above.
The v7 rebound repair snapshot is `source-manifest.aliasfix-v7.json`. Only the CUDA backend
lib compiled, not the workspace. Its exact alias probe reached execution but
`gpu_device_synchronize` reported illegal memory access. Bounded memcheck located its first
invalid 8-byte write in `raw_matrix_add_sub_kernel`, `MatrixNTT.cu:697` → `:84`, at address
`0x4000190000000000`. The 189 errors include cascading API errors; an earlier malloc/OOM
message is not an established root cause. `addr2line` did not resolve symbols. No correction
is validated. Diagnosis is active with no GPU job live. Revised HIP source has not compiled; no
repetition set or implementation review has run. Neither successful alias execution nor a
successful repaired long KHE workload or AMD runtime result is established.

V8 (`source-manifest.aliasfix-v8.json`) compiled the CUDA backend lib only. Its plan and two
GPU execute calls succeeded, but CPU-reference extraction then panicked with `CPU family
output`, so the exact test still failed. V8 bounded memcheck completed in 0.715 s, exit 99.
It reported no invalid global accesses, but its 104 errors included approximately 96 printed
use-after-free reports (first at line 88: HtoD versus FreeAsync) and four allocation API errors.
This is not a clean memory check or a successful oracle comparison. The exact stripped binary
is preserved under `preserved-binaries/aliasfix-v8-backends`, identified by
`aliasfix-v8-preserved-binary.json`. V9 now compiles the CUDA backend only with retained line symbols, and its CPU oracle
materializes families. The exact bounded symbol-bearing memcheck run failed: 312 errors,
exit 99 after 1.869 s. The oracle reports `actual_nested_computed[0]` as GPU 2 versus CPU 22
after input rebinding; this is a result mismatch, not the earlier oracle-format panic.
The first use-after-free covers 5168 bytes in measurement-trial graph launch, with launch
frames at `gpu_runtime_direct.rs:496`, `:2556`, and `gpu_execution_plan.rs:2847`; the free
originates at `MatrixData.cu:332` through physical-frame drop at the same plan line.
`source-manifest.aliasfix-v9.json` and `aliasfix-v9-preserved-binary.json` identify the retained
symbol binary `preserved-binaries/aliasfix-v9-backends-symbols`. Native upload dependency and
Rust trial/rebinding/lifetime repairs are active. No successful exact test or repetition set
is established. Production control and native-runtime lifetime diagnosis is in progress; revised
workspace/HIP compilation, repetition sets, and implementation reviews have not started.

V10 (`source-manifest.aliasfix-v10.json`) connects the native NTT upload event dependency
and Rust wave/nested-snapshot input rebasing without adding native global synchronization.
Its retained-symbol CUDA backend-only compile passed. The bounded exact six-output alias test
passed once, including CPU comparison, rebinding, and held outputs. However, the memcheck
harness exited 99 after 1.870 s with 309 errors, so memory validation still fails. Reported UAF
counts are 91 matrix/5168-byte, 73 buffer/1024-byte, 132 buffer/512-byte, and nine
buffer/768-byte reports; four allocation API errors are included. All seven NTT HtoD UAF reports
seen in v9 were eliminated: four 512-byte twiddle uploads and three 16-byte uploads. The
passing fixture reaches four additional 512-byte buffer D2H reports (32 in v9, 36 in v10),
so the net total decreases from 312 to 309. The 64 NTT/512-byte graph-teardown reports remain
a separate unresolved group; no reports were suppressed. Remaining lifetime
errors are being diagnosed in native/runtime and control code. Full workspace/HIP compilation,
repetition sets, and implementation reviews have not run on this latest source.

V11 (`source-manifest.aliasfix-v11.json`) again passes retained-symbol backend-only compilation
and the exact six-output CPU/rebind/held-output test once. Its bounded memcheck still fails:
exit 99 after 1.869 s, 237 errors. Parsing attributes all 233 UAF reports to graph accesses,
with zero HtoD or D2H reports; four deliberate allocation API reports make up the remainder.
The 72 D2H reports present in v10 are removed, reducing 309 to 237, but this is not clean
production memory validation. Full workspace/HIP builds, repetitions, and reviews remain
pending on this source.

V12 (`source-manifest.aliasfix-v12.json`) changes nested-owner budget accounting. Its
retained-symbol backend-only compile passed, followed by 229 passing host tests with 72
ignored and one passing CUDA exact alias functional probe. Memcheck was not rerun because
budget accounting does not address the diagnosed lifetime cause; the latest memory evidence
remains v11's failing 237-error result. Production allocation/graph identity tracing is in
progress; no additional lifetime repair is established yet.

V13 (`source-manifest.aliasfix-v13.json`) adds opt-in native/Rust owner tracing under
`MXX_GPU_TRACE_OWNERS=1`, without GPU semantic changes. Backend-only symbol compilation and
one exact alias functional test pass, but bounded memcheck exits 99 with the same 237 errors,
including 233 graph UAF reports. The run retained 2719 shared-sequence records below the
4096-record cap. `aliasfix-v13-preserved-binary.json` identifies
`preserved-binaries/aliasfix-v13-backends-symbols`. The first reported address was reused
across 512-byte buffer and 5168-byte matrix generations; saved launch/free identities are
being correlated. No stale-generation or false-positive conclusion is established, and no
repair of the remaining graph lifetime reports is validated. Broader gates remain incomplete.

V14 (`source-manifest.aliasfix-v14.json`) connects the producer/retirement protocol across
native Runtime/MatrixData/SmallRhs headers and sources and Rust runtime/GPU/real wrappers.
Typed regions use precomputed data without a whole-plan scan, global fence, or extra copies.
The narrow retained-symbol CUDA backend compile and exact six-output CPU/rebind/held-output
test pass. With `MXX_GPU_TRACE_OWNERS=1`, 3539 records remain below the 4096 cap: 150 buffer
producer, 268 matrix producer, 150 buffer retirement, 101 matrix retirement, 41 context
retirement, 110 writer-tail, and 41 launch/completed-wait events. This source/trace evidence
shows that the new protocol is called. It does not establish that the remaining graph UAF
root cause is corrected: memcheck remains exit 99, 237 errors (233 graph UAF and four
allocation API reports), after 1.919 s. Diagnosis continues; broader gates remain pending.

Latest broader v14 artifacts are `aliasfix-v14-hip-{gfx1100,gfx942}-compile-result.json/log`
and `aliasfix-v14-hip-host-results.json` with all seven terminal per-binary logs.
`aliasfix-v14-cuda-workspace-compile-result.json/log` records the normal-release CUDA
workspace compile pass. `aliasfix-v14-cuda-functional-results.json` is terminal: alias and nested exact filters
passed 300 repetitions each in identity, `0,0`, and `0,0,0` (1800 total). Identity broad
smoke passed three repetitions; duplicate broad smoke failed three of three in each mode,
only at `test_gpu_direct_parallel_lanes_spread_over_devices` with a
`gpu_matrix_wait_compiled_inputs` device mismatch. FHE five-case and KHE three-case smoke
sets each passed three repetitions. This functional evidence does not clear memory errors. These demonstrate
HIP compilation and CPU-only host checks, not AMD execution. The scalar integer-control
production test passed functionally under bounded memcheck
(`aliasfix-v14-scalar-control-memcheck.json/log`), but the sanitizer exited 99 with two deliberate
allocation-probe API reports at `Runtime.cu:1644/1651`; it reported zero device/UAF errors.
This scalar result does not clear the unresolved 233 matrix graph UAF reports.

The ordinary matrix-wave diagnostic still reports 54 UAF, whereas the scalar control case
reports zero UAF; intentional cache-flush allocation API errors are tracked separately.
The external actual-matrix-add completion-only probe completed 24 launches, memcheck exit
zero/zero errors (`native-matrix-add-diagnostic-memcheck.json`, build record
`native-matrix-add-diagnostic-build.json`, and logs). It makes no numerical assertion and
does not clear Rust production UAF. Rust/native per-storage ownership correction is pending v15.

V15 is an unvalidated per-storage repair across `Matrix.h`, `MatrixData.cu`, Rust GPU and
matrix wrappers, and `gpu_runtime_direct.rs`. The ignored diagnostic
`test_gpu_diagnostic_root_matrix_addition_pool_owners` adds ordinary root matrix addition
without wave execution, with a numerical CPU oracle. CUDA full workspace and HIP gfx1100/gfx942
compile gates passed, as recorded in `aliasfix-v15-cuda-workspace-compile-result.json` and
`aliasfix-v15-hip-{gfx1100,gfx942}-compile-result.json` with per-gate logs.
`aliasfix-v15-hip-host-backends-result.json/log` records 229 host passes, 73 ignored.
`aliasfix-v15-memory-diagnostics-results.json` records functional passes but memcheck exit 99
for root-add (13 reports: nine graph UAF plus four intentional cache-flush API reports),
matrix-wave (30 reports), and alias (97 reports). These normal stripped-profile results
are not evidence of a direct reduction from v14's symbol-bearing run. `aliasfix-v15-cuda-functional-results.json` records terminal 1800 alias/nested passes and
three broad-smoke passes each in identity/`0,0`/`0,0,0`, plus FHE/KHE three smoke passes each.
The external actual-native cache-trim probe completed 24 launches with zero graph UAF,
but memcheck exit 99 from 16 intentional OOM API reports
(`native-matrix-cache-trim-diagnostic-memcheck.json/log`). It is completion-only with no
numerical assertion and is not a clean full memory gate. Production root-add's nine UAF
remain unresolved. An external actual-emitter replacement-owner diagnostic is being prepared;
no production change or validation pass is established by that preparation. The source is
frozen at `source-manifest.aliasfix-v15.json`; no clean memory gate is established.

## Independent lifetime diagnostics

The bounded standalone CUDA diagnostic is evidence about an ordering mechanism, not a
production validation pass. `async-lifetime-diagnostic-results.json` identifies source hash
prefix `681bc754`: both `host-event` and `release-event` completed eight launches with 64
verified outputs per launch, but each memcheck exited 99 with eight D2H UAF reports.
`async-lifetime-diagnostic-v2-results.json` identifies source prefix `0f7c9f15`: recording a
default-stream event after synchronous D2H and making release wait on that event
(`download-event`) exited zero with zero errors. Destroying the completion event after host
synchronization (`early-event-destroy`) exited 99 with eight D2H errors. Neither diagnostic
reported graph errors. Exact commands, 60-second bounds, full source hashes, and per-mode
logs are retained in those JSON records under the external artifact directory.

The native D2H completion dependency is being connected to production. Trial graph UAF
remains unexplained. The latest production v14 memory gate still fails with 237 errors;
these standalone results do not validate that repair or replace workspace/HIP builds,
repetitions, reviews, or the missing AMD hardware gates.

The v3 standalone diagnostic (`async-lifetime-diagnostic-v3-results.json`, source prefix
`f8de472f`, per-mode logs) tested `graph-regions` and `graph-replays`: each completed 16
launches with eight sets of 64 verified outputs, exit zero and zero memcheck errors. These
narrow cases do not reproduce or clear the remaining production graph UAF. A v4 late-kernel
binding diagnostic is prepared; no further production edit is established by that preparation.

Standalone v4 late-binding modes `exec-rebind` and `source-exec-rebind` each completed 16
launches and 1024 verified outputs with zero memcheck errors (`async-lifetime-diagnostic-v4-results.json`,
source prefix `d3f4d73a`, per-mode logs). The separate no-D2H graph-completion diagnostic
(`graph-completion-diagnostic-results.json`, source prefix `87375bd2`) passed both
`host-completion` and `release-completion` modes with zero memcheck errors and eight launches
each. It performs no readback or numerical assertion. Neither diagnostic clears production
UAF or establishes production correctness. Latest-source workspace/HIP builds, repetition
sets, reviews, and AMD hardware gates remain incomplete.

The argument ABI/event-reuse diagnostic tested typed pointers and integer limb arrays with
fresh and recycled events, four conditions with eight launches each, zero memcheck errors
(`graph-limb-argument-diagnostic-results.json` and per-condition logs). It performs no readback
or numerical correctness assertion and is separate from the production memory gate.

Additional v14 native upload and generic-buffer output diagnostics each completed 24 launches
with zero memcheck errors (`native-matrix-add-upload-diagnostic-memcheck.json` and
`native-matrix-buffer-output-diagnostic-memcheck.json`, with logs). Both are completion-only
and make no numerical correctness claim. The production v14 matrix-wave trace reports 60
errors: 54 UAF plus six intentional allocation API reports. Clean standalone diagnostics
do not clear that production memory failure.

## Further v15 diagnostics: memory gate still failing

The actual-emitter replacement-owner standalone diagnostic passed retain and release modes,
24 launches each with zero errors (`native-matrix-replacement-owner-diagnostic-results.json`
and logs). It is completion-only with no numerical assertion and does not clear production UAF.

An official NVIDIA sanitizer package 13.1.118-1, reporting version 2025.4.1, was SHA-256
verified and extracted only under the external `/tmp` artifact directory; no system install
occurred (`sanitizer-patch-provision.json`). Running the same v15 binary with it still fails:
root-add nine UAF plus four intentional API reports, matrix-wave 18 UAF plus six intentional
API reports, alias 93 UAF plus four intentional API reports
(`aliasfix-v15-patched-sanitizer-memory-diagnostics-results.json/logs`). Matrix-wave counts
vary with planner choices; that variation does not establish a repair.
`CUDA_LAUNCH_BLOCKING=1` leaves root-add's nine UAF plus four API reports unchanged
(`aliasfix-v15-root-add-launch-blocking-diagnostic.json/log`). This is diagnostic serialization,
not a production fix or asynchronous correctness proof. No suppression or false-positive
conclusion is established.

At this earlier v15 checkpoint, runtime tracing enhancements awaited v16 compilation without
semantic ordering changes; the later v16 compile passes are recorded below.
A concrete constant-return parallel-admission case is being repaired; the earlier long KHE
NodeId(1762) case has not been identified as that case. No successful new repair is claimed.

## Latest v16 compile, functional, and provenance evidence

`aliasfix-v16-cuda-workspace-compile-result.json` and
`aliasfix-v16-hip-{gfx1100,gfx942}-compile-result.json` record successful workspace compilation.
`aliasfix-v16-cuda-functional-results.json` records zero failures: alias, nested-computed,
and constant-return exact tests each ran 300 repetitions in identity, `0,0`, and `0,0,0`,
2700 total. Broad smoke in each mode and FHE/KHE smoke each passed three repetitions;
host checks passed. This does not clear the terminal failing memory records in
`aliasfix-v16-memory-diagnostics-results.json`: root-add 16 reports (12 graph UAF + four
expected cache-trim OOM API), matrix-wave 60 (54 + six), alias 237 (233 + four), all exit 99.
These retained-symbol counts must not be treated as direct reductions from v15 stripped
runs. The official sanitizer 2025.4.1 isolated under `/tmp` still reproduces the issue;
no suppression or false-positive conclusion is established.

The original Ring-GSW CPU provenance v2 follows NodeId(1762) through named call,
`FamilyGetStatic`, inner parallel, named calls, and `MatrixBinary`: its matrix is produced,
not a constant return. `aliasfix-v16-ring-gsw-cpu-provenance-v2.json/log` supports the
nested-computed regression choice, not successful original GPU admission or decryption.
`ring-gsw-cpu-provenance-cleanup.json` confirms restoration of the exact pre-diagnostic
fixture SHA after removing the temporary CPU test. Production code is unchanged by that
cleanup, but the compiled manifest includes the temporary test, so final fixture rebuild
validation remains required. Earlier native replacement-owner diagnostics remain narrow
completion-only evidence: 24 launches each, zero errors, no numerical proof. Missing AMD
hardware/runtime and all unresolved memory/review gates remain explicit.

## Same-target backend switch

The build-switch gate reused `target-backend-switch` under the external artifact directory
for `cargo test --offline -r --workspace --lib --features gpu --no-run`: HIP gfx1100, CUDA
SM 89, then HIP gfx1100. All three stages exited zero. Recorded backend metadata and FHE cfg
matched each selection. HIP native revision `9137f1a6b895d989` was restored after CUDA revision
`3a3cca7dab568bed`. HIP linked `amdhip64`, emitted no CUDA runtime or native TFHE link directives,
and the final HIP FHE binary had no matching TFHE native symbols. CUDA linked the TFHE native
archive and `cudart_static`. The validator corrected its assumption that CUDA must have a
`libcudart` dynamic NEEDED entry by inspecting the recorded static link directives; it did not
rerun the builds. This gate proves build selection and metadata/linkage propagation, not AMD
device execution. The latest long KHE probe failed; its production diagnosis/fix is in progress, and AMD
hardware gates remain incomplete.

## Validation corrections

The initial CUDA device suite reported 68/69 backend cases and 5/5 FHE cases. The new nested
fixture used an unsigned remainder operation not supported by resource admission. Its original
900 repetition attempts failed before GPU execution, as preserved in `cuda-repeat-results.json`;
these are fixture/admission failures, not intermittent device failures. The corrected fixture
passes the final smoke sets and all three 300-run sets above.

The original long KHE failure reproduced on the before-change source (`baseline-khe-long.log`)
and exposed an existing lexical loop-slot binding issue. A narrow correction is applied and a
stronger nested named-callee test validates that binding behavior. Its final long workload probe
failed after about 40 seconds with `GPU child node 50 no unique frozen physical choice`.
The planner inherited the caller loop site for a named canonical child while lowering expected
no loop site. The narrow matrix planner correction is compiled for CUDA and both HIP targets, and the new
matrix named-call regression passes in the three CUDA logical modes. The latest long probe is terminal: one completed run, one failure, wall time 5730.1545556 s.
`cuda-postplanner-results.json` and `postplanner-cuda-khe-long-probe.log` record the resource
admission error: no feasible measured `(W,C)` at `W=1,C=2` or `W=1,C=1`, because
`GPU parallel child output must be produced inside its body: output 0 of NodeId(1762)` for
`Matrix[crt1021 degree2 rows2 cols2]`. Production diagnosis and a correction are in progress;
no successful corrected workload or GPU arithmetic result is established by this failure.
The earlier 900 integer/control repetitions precede this isolated matrix correction; they
were not rerun because that fixture is unaffected. The final smoke
commands intentionally omit that long ring-GSW workload; the
identity backend smoke also omits the partition-specific allocation-query case, which executes
in the duplicate-device modes. Exact commands and skips are retained in the final result JSON.

The BGV wall times above include the complete unit-test process and are a small smoke
comparison on one NVIDIA device. They are not isolated kernel measurements, a statistical
performance conclusion, or evidence of AMD/NVIDIA performance equivalence.

## Required remaining evidence

- Repair the remaining alias lifetime errors and obtain a clean memory check,
  then compile the workspace for CUDA and both HIP targets
  on the repaired snapshot, and run affected exact probes and required repetition sets.
  V8 has a backend-only CUDA compile, two completed GPU execute calls, a CPU-oracle failure,
  and a failing use-after-free memcheck. V9 compiles backend-only but fails symbol-bearing
  memcheck and the corrected oracle comparison. V11
  passes the exact CPU/rebind/held-output test once but still fails memcheck with 237 errors.
  The new-source workspace, HIP, and repetition gates are incomplete.
- Diagnose and repair the latest long KHE measured resource-admission failure, then rebuild
  and run affected checks and the workload probe. The earlier compile and matrix named-call
  smoke checks pass, but this workload gate is currently failing.
  Preserve each set's completed count, failures, exact command, and source identity. Further
  affected synchronization/ownership filters still require 300 repetitions, and round-trip
  smoke filters require 3–5, per `GPU.md`.
- Execute AMD arithmetic, sampling/retry, nested IF/WHILE, rebinding, status/error, ownership,
  artifact visibility and round-trip, and BGV production graph checks on supported hardware.
  Validate both wave32 and wave64 targets; successful compilation does not prove lane behavior.
- Validate identity, `0,0`, and `0,0,0` logical mappings, plus multiple physical AMD GPUs,
  directional peer access, and forced host staging. Logical duplicates do not replace physical
  devices or validate aggregate memory capacity.
- Measure asynchronous behavior, transfers, launches, control waits, memory peaks, and the
  documented latency/total-time/parallelism contract on the production path. Compare CUDA
  before and after on the same NVIDIA device; establish the first correct AMD run as its own
  baseline. Do not claim equal performance across different devices.

Record SDK/compiler/driver versions, source hashes, architecture, command, environment, exit
status, completed repetitions, and failures for each new run. A failing or unavailable runtime
gate remains a failing or unavailable gate; it is not replaced by compilation or CPU checks.

## Reversed trial-retention diagnostic

`trial-retention-compile-result.json/log` records a successful backend-only CUDA compile of a
temporary retention diagnostic. Its corrected exact root-add run passed the numerical test
but failed memcheck, exit 99 with 13 reports: nine trial graph UAF plus four expected OOM API
reports (`trial-retention-memcheck-result.json/log`). Two candidate graphs/frames and six joined
completion events remained retained through final execute/download, followed by explicit
teardown. An initial runner argument-index error selected zero tests; that run was discarded,
and only the corrected one-test run supplies functional evidence.

A paired read-only lifetime audit finds baseline v16's nine trial plus three execute UAF
reports become nine trial plus zero execute reports under retention. Candidate two's output
has one live allocation with no intervening free. All relevant launches completed on the host,
and retirement dependencies joined before deferred free. All possible matching input trial
and production launches also completed before free. This excludes a simplistic early trial
frame drop explanation for the remaining nine reports; it does not establish a false positive,
tool defect, clean memory validation, or acceptance. The temporary patch was reversed:
`trial-retention-cleanup.json` confirms both Rust files exactly match pre-diagnostic v16.

## Bounded standalone CUDA comparison

The official CUDA sanitizer package 13.4.92-1 was SHA-256 verified and extracted only
under the external artifact directory, without changing the system driver or toolkit.
It reports Compute Sanitizer 2026.3.0. The preserved v16 binary still reports errors:
root-add 16, matrix-wave 60, and alias 213 (candidate choices can change counts).
The retained-trial binary reports 16; no clean memory gate follows from this comparison.
See `sanitizer-13-4-provision.json` and `sanitizer-2026-3-comparison-results.json`.

An external production-native two-column graph reports 64 allocation-before-use
accesses followed by launch failure with both sanitizer versions. A standalone CUDA
program, with no project code, reproduces that result for two independent kernel
nodes sharing allocations; one node passes. A read-only diagnostic audit found no
bounds, argument-lifetime, or successful-path ordering defect. Ordinary execution
of the two-node program passes its zero-input numerical check. A diagnostic variant
that orders the two nodes passes memcheck; a synchronous-allocation variant passes
numerically but reports nine lifetime errors. These variants are diagnostic only:
no serialization, allocator substitution, suppression, or production change was made.
The zero-input oracle is weak against omitted work; this is evidence of an external
tool/runtime interaction, not proof that every production report is a false positive.
See `native-matrix-two-operation-results.json`, `cuda-two-column-results.json`,
and `cuda-two-column-variant-results.json`. This bounded comparison concludes this
investigation checkpoint; memory acceptance and AMD hardware validation remain open.

## Current remaining gates after trace cleanup

- Obtain terminal cleaned-source CUDA and HIP gfx1100/gfx942 workspace compile results and
  affected validation. The v17 compile gates and 27 targeted CUDA checks have now passed;
  the memory gate still fails.
- Resolve memory acceptance. Standalone tool/runtime reproduction does not prove every
  production report is a false positive, and no blanket waiver or suppression is established.
  At `03c5c0719`, only the nested control-region test still reports use-after-free. Accepting
  that remaining case on the standalone evidence below is a pending review decision.
- Establish successful original unchanged long Ring-GSW admission/execution/decryption;
  CPU provenance and smaller regressions do not supply that result.
- Complete required 3–5 round-trip cases beyond demonstrated smoke scope, canonical artifact
  compatibility, and production measurement of transfers, launches, control waits, asynchronous
  behavior, latency/total-time/parallelism, and VRAM/pinned-memory peaks. Existing whole-unit
  BGV timings are not the complete performance contract.
- AMD hardware is unavailable: wave32/wave64 runtime, required 300-run ownership/synchronization
  sets, multiple physical AMD GPUs, directional peer access, forced staging, and per-device
  memory-budget evidence remain missing. Logical duplicates do not replace these gates.
- Begin formal Sol reviews once validation gates permit, followed by the required Astra review.
  Neither review stage has started, and implementation completion remains unproven.

## Astra memory-cause analysis and targeted follow-up

Astra medium inspected the current root-add successful path and found no missing
completion or retirement join in that path. Its leading hypothesis is a Compute
Sanitizer allocation-lifetime tracking defect across parallel CUDA graph branches.
An independent diamond-graph report on driver 610 / CUDA 13.3 has a similar
allocation-before-use failure; NVIDIA acknowledged a tracking ticket:
https://forums.developer.nvidia.com/t/compute-sanitizer-and-cuda-graph-false-positives/373484.
That report is supporting evidence, not an upstream confirmation of all mxx reports.

The external `cuda-graph-ready-nonzero.cu` preserves asynchronous allocation/free
and independent kernels, waits for allocation/input readiness, fills outputs with
a sentinel, and verifies distinct nonzero per-index/per-replay inputs after each
of three launches. One kernel has zero memcheck errors. Two kernels pass every
numerical check but report nine UAF errors. Removing the empty graph endpoints
does not change the nine reports. `cuda-graph-ready-nonzero-results.json` records
commands, exit statuses, counts, source and binary hashes.

Astra's external `cuda-graph-branch-events.cu` compares the same parallel graph
with no graph event, one event joining both branches, or one event per branch.
The first two cases each report nine UAF errors; separate branch events followed
by producer-stream waits report zero errors. All three cases pass the nonzero
per-replay oracle. `cuda-graph-branch-events-results.json` records the comparison.
The current production prototype exports each nonempty top-level operation's
completion through a private graph event and joins those events on the launch
stream before recording the submission-specific public event. Original graph
dependencies and parallel branches are preserved. No events are inserted in
conditional bodies. Successful submission does not add a host wait; partial
construction cleanup and uncertain-launch owner retention remain in place.

The v19 targeted CUDA backend compilation passed with warnings denied. Full
workspace CUDA and HIP compilation passed for the preceding v18 prototype;
those builds do not validate the subsequent v19 change. Actual v19 memcheck
results with Compute Sanitizer 2026.3 are:

| Case | UAF reports | Other API reports | Numerical test |
| --- | ---: | ---: | --- |
| Root matrix addition | 0 | 4 | Passed |
| Parallel matrix waves | 6 | 6 | Passed |
| Named borrowed matrix outputs | 69 | 4 | Passed |
| Nested control regions | 407 | 2 | Passed |

The other API reports are the existing cache-release allocation probe's expected
out-of-memory errors. They are not suppressed and are distinct from UAF reports.
No allocation-before-use or invalid-global-access reports occurred in these four
runs. The parallel-wave UAF reports are all from planning trials and do not
involve conditional nodes. Numerical success alone does not establish memory
correctness. Logs and commands are recorded in the external runtime artifact
`branchfix-v19-memory-diagnostics-results.json`.

Astra medium next inspected operation terminals and found no missing native
chain terminal. Its independent `cuda-conditional-terminal-diagnostic.cu` was
compiled and GPU-run with three distinct nonzero replays per case. Each launch
waits for allocation/input readiness, waits for its completion event, and
finishes D2H readback before any asynchronous free. Results in
`conditional-terminal-diagnostic-results.json` are:

| Graph | Completion records | UAF reports | Numerical checks |
| --- | --- | ---: | --- |
| Two sequential root kernels | Final kernel only | 0 | Passed |
| Two sequential root kernels | Each kernel | 0 | Passed |
| IF with one kernel | None | 6 | Passed |
| IF with one kernel | Outer conditional | 0 | Passed |
| IF with two independent kernels | None | 12 | Passed |
| IF with two independent kernels | Outer conditional | 9 | Passed |
| IF with two sequential kernels | Outer conditional | 0 | Passed |

These isolated results support a sanitizer tracking limitation for parallel
conditional bodies, rather than an identified lifetime defect in this
reproducer. They do not establish that every remaining mxx report is spurious.
CUDA conditional bodies cannot contain event-record nodes, so the root-branch
event mechanism cannot be applied directly inside them. Recording every native
root kernel is not justified by the sequential comparison, which already passes
with a final record. A valid complete correction must preserve parallelism,
submission-specific completion, event lifetime, conditional/loop restrictions,
and uncertain-launch owner retention, and clear actual mxx cases before a memory
acceptance claim is made. Current memory acceptance remains incomplete.

An additional external diagnostic containing an explicit device-wide wait before
free was initially not executed because automatic approval review was at model
capacity. A subsequent submission through the same approval mechanism succeeded.
The parallel IF case still produced nine UAF reports after that wait, with all
three nonzero replays correct. No device-wide wait was added to production code.

A second Astra diagnostic compared nine launches across three candidate graphs,
with either per-launch readback or only a fresh completion-event host wait before
the next replay and final-per-candidate readback. Both modes passed numerical
checks with zero memcheck reports. Thus omitting intermediate D2H readback does
not reproduce the remaining plain mxx trial reports in this model. Commands and
results are in `trial-and-global-sync-diagnostics-results.json`. Candidate
allocation, rebinding, and cache-trim differences remain to be isolated.

The production runtime instantiates with `cudaGraphInstantiateFlagAutoFreeOnLaunch`,
whereas the preceding external trial model used zero flags. An otherwise
identical `cuda-trial-autofree-diagnostic.cu` was prepared and compiled to isolate
that difference. After transient approval-review capacity failures, execution
through the same approval mechanism succeeded: nine launches passed numerical
checks with zero memcheck reports. This comparison does not justify changing
production instantiation flags.

Astra then modeled the actual matrix-input and buffer-output retirement event
protocol in `cuda-owner-retirement-diagnostic.cu`: a third release stream,
temporary retirement events destroyed after enqueuing their waits, persistent
producer events re-recorded after the public launch-completion record, and
owner destruction waiting before asynchronous free. Four modes independently
enable that protocol and executable kernel-argument rebinding after upload.
The weighted nonzero oracle detects stale swapped-operand bindings. All modes
passed nine launches across three candidate graphs with zero memcheck reports;
results are in `owner-retirement-results.json`. These source-derived differences
alone therefore do not reproduce the mxx reports. They establish no production
correction and do not prove the remaining reports spurious.

The plain parallel-wave numerical fixture has subsequently been strengthened:
distinct nonzero inputs replace the previous zero matrices, and every returned
member is compared with the existing CPU executor's materialized family output.
The preceding v19 runtime results cover the old fixture, not this stronger one.
Its source identity is recorded in `source-manifest.nonzero-wave.json`. Targeted
CUDA backend compilation passed with warnings denied. Actual GPU execution
passed all numerical comparisons but reported 15 UAF errors plus six expected
cache-probe API errors. The changed fixture therefore strengthens numerical
evidence without clearing the memory gate. Commands, logs, and results are in
`nonzero-wave-runtime-results.json`. Completion remains unproven.

An actual-graph DOT diagnostic recorded the plain parallel-wave test's four
executable graphs: one, two, four, and four independent root
`raw_matrix_add_sub_kernel` nodes, respectively. Each kernel has its own
event-record side node; there are no conditional nodes, empty joins, or
unrecorded kernel terminals. Kernels use grid `(1,1,2)` and block size 256.
The numerical test passed with 20 UAF reports and six expected cache-probe API
reports in this diagnostic run. Varying counts do not establish a new lifetime
defect or acceptance. Artifacts are `actual-wave-dot-result.json` and
`actual-wave-graph-*.dot`.

Temporary DOT instrumentation was removed; the resulting Runtime source was
checked byte-for-byte against its pre-diagnostic copy. A subsequent targeted
CUDA backend rebuild passed with warnings denied, restoring the test executable
to that source (`restored-nonzero-wave-compile-result.json`). The DOT diagnostic
runtime result belongs to the earlier instrumented executable, not that rebuild.

External follow-ups reproduced one/two/four independent root kernels, separate
branch records, executable rebinding, and the source-derived retirement protocol
with shared or disjoint input slices. Both variants passed nine launches with
zero reports. Matching the actual two-block, 256-thread launch shape also passed
both variants. Results are in `four-branch-retirement-results.json` and
`multiblock-branch-results.json`. Branch count, shared reads, and this launch
shape alone do not explain the mxx trial reports. Exact native operand layout
and candidate resource lifecycle remain unresolved; these comparisons justify
no additional speculative production change.

Astra's `cuda-raw-limbset-diagnostic.cu` then used the production 48-byte
`MxxRawMatrixLimb` declaration, three 768-byte by-value `RawLimbSet` arguments,
and the actual raw matrix kernel, load/store, and modular arithmetic bodies.
It reproduces one/two/four/four independent kernels with direct branch events,
two limbs, and 256-thread blocks. Both the simple mode and the combined
retirement-protocol/executable-rebinding mode passed 12 launches and nonzero
final-per-wave numerical checks with zero memcheck reports
(`raw-limbset-results.json`). This rules out the argument ABI and these kernel
bodies alone in that model. Its strides and limb offsets are valid contiguous
limb-major values, not captured actual mxx argument values. Capturing and
checking those seven bound arguments and their allocation-relative ranges is
the next discrimination; no native lifetime or overlap defect has yet been
identified from the recorded root graphs.

An actual bound-argument capture recorded the seven arithmetic arguments after
patching. Input strides are row 2048 / column 1024 / coefficient 8 bytes;
output strides are row 1024 / column 512 / coefficient 8 bytes. The two limbs
start 256 bytes apart. The moduli are 1125899906842177 and 1125899906840897.
The capture passed the CPU numerical oracle with 20 UAF reports and six expected
cache-probe API reports. `actual-wave-bound-args.json` stores descriptors, and
`actual-wave-range-analysis.json` computes the intervals of the observed
one/two/four/four/four/four-kernel binding batches: no simultaneous output
intervals overlap. Input reads remain within their 10288-byte allocation bounds;
each output wave stays within its 1024-byte buffer. This covers the observed
plain matrix-wave case, not every application graph.

The argument-capture instrumentation was removed byte-for-byte, followed by a
warning-free targeted CUDA rebuild (`restored-after-args-compile-result.json`).
An additional Astra follow-up was stopped by an automatic content filter before
delivering analysis; the primary agent completed the already specified external
stride comparison using captured descriptors.

The exact-stride model passed 12 launches with zero UAF reports both without
cache trimming and with graph/default-pool trimming. Adding the production-style
whole-device allocation probe still produced zero UAF reports, but reported
eight expected OOM API errors across four candidates; its total error count is
not zero. `exact-stride-cache-results.json` records all three modes. Varying the
candidate graph and allocation streams across a four-stream pool also passed
all three variants with zero reports (`candidate-stream-results.json`).

The diagnostic Rust line-table setting was found to set Cargo build-script
`DEBUG=true`; `cc` consequently adds CUDA `-G`, despite release `OPT_LEVEL=3`.
The same external exact-stride model compiled with `-G -O3` passed with zero
reports, as did a model separating kernel and graph construction into two CUDA
translation units. Results are in `exact-stride-cache-debug-result.json` and
`cross-module-diagnostic-result.json`. A direct mxx comparison with release
debug information disabled is being compiled; no result from that comparison
has yet established a cause or a complete memory fix.

## Cache-release synchronization and final-source gates

Evidence for source `03c5c0719` is in `final-03c5c0719/`. Its `source-manifest.json` records
the clean commit, nvcc 13.1.80, driver 580.178.04, the RTX 4080 SUPER, Compute Sanitizer
2026.3.0, and the build command. `binaries.sha256` and `libgpupoly.sha256` identify the tested
binaries and the static native library. The bisection reproducers and their logs are in
`cache-release-fix-2026-10-06/`.

Cause. Removing the `gpu_device_release_cached_memory` call from the native reproduction
removed all 12 reports and all eight API errors. Bisection inside it isolated the trigger:
the failing whole-device `cudaMalloc` probe alone reproduced the 12 reports, while graph-memory
trims, default-pool trims, and synchronization produced none. Before the probe, a
`cudaStreamSynchronize` of the stream removed the reports. An event record and synchronize on
the same stream, which the function used, did not. Both waits cover the same work. The
sanitizer, however, treats only the stream synchronization as completing the queued
stream-ordered frees. The function was already host-blocking and is planning-only, called
between candidate trials. The change therefore keeps one wait per context stream, does not
synchronize the device, and leaves production launch, retirement, and free paths unchanged.

| Gate | Evidence | Result |
| --- | --- | --- |
| Native library reproduction (12 launches, nonzero inputs) | `memcheck-native.txt/log` | Numerical pass; 0 use-after-free (12 before); 8 expected probe OOM API errors |
| `test_gpu_diagnostic_root_matrix_addition_pool_owners` | `memcheck-*.log` | Pass; 0 use-after-free; 4 expected probe API errors |
| `test_gpu_direct_parallel_waves_return_all_resident_matrix_members` | `memcheck-*.log` | Pass; 0 use-after-free (3 before); 6 expected probe API errors |
| `test_gpu_parallel_named_borrowed_matrix_outputs_rebind_and_hold` | `memcheck-*.log` | Pass; 0 use-after-free (93 before); 4 expected probe API errors |
| `test_gpu_nested_control_regions_match_cpu_and_hold_replayed_outputs` | `memcheck-*.log` | Pass; 475 use-after-free (420–482 before); 2 expected probe API errors; no invalid global accesses |
| CUDA sm_89 and HIP gfx1100 workspace GPU lib compilation | build logs of this session | Passed without warnings |
| 300-iteration GPU unit gate | `repeat-*/summary.txt`, `repeat-*/command.txt` | Identity, `0,0`, and `0,0,0`: 300 iterations each, 900 total, zero failed iterations |

Each gate iteration ran the `mxx-fhe` (5), `mxx-backends` (74), and `mxx-khe` (3) GPU test
binaries. The command was `<binary> gpu --ignored --skip
test_gpu_ring_gsw_arithmetic_executes_through_dsl_ir_runtime_and_decrypts`. Identity mode also
skipped `test_gpu_matrix_allocation_query_uses_partition_decomposition_metadata`, which asserts
two logical devices and fails identically without this change.

Remaining nested control-region reports. Standalone CUDA programs in
`cache-release-fix-2026-10-06/` reproduce the same report class without mxx (`pw4.cu`,
`cg.cu`). A Graph with two or more conditional nodes at one level reports use-after-free when:

- all work runs on one stream,
- the host synchronizes that stream after every launch,
- and the executable is never destroyed.

Serializing the conditionals, a final join kernel, event-record nodes, and separate executables
did not remove the reports. Compute Sanitizer 2025.4 behaves the same. Inside mxx, a
launch-stream synchronization after each launch left 450 reports. Host waits on every
per-branch completion event left 482. Only a device-wide synchronization after each launch or
before each free removed them, and `GPU.md` forbids that in production. No ordering defect has
been identified for this case, and no report was suppressed. HIP device execution, including
the review fixes in `2efe36f75` and `03c5c0719`, is unverified without AMD hardware.
