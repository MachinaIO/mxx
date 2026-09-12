# Fleet-wide GPU column sharding — acceptance summary (2026-09-12)

This document closes Step 7 of `fleet-wide-gpu-handoff-2026-09-12.md`. It
records the allocation/range/metric audit, the matched performance evidence,
the validation scope actually run, the independent review outcome, and the
items that remain explicitly deferred. Chronological detail lives in
`fleet-wide-gpu-implementation-status.md`.

## 1. Allocation, range and metric audit

Every GPU IR kind accepted by `GpuDcrtBackend::prepare_graph_admission`
(`gpu_inventory.rs`) is admitted through a compiled `PreparedMatrixOperation`
whose native claims are either computed from the same native layout queries the
production codec uses or recorded by `trace_native_claims` on an open domain.
Under a sealed domain every allocation must match the next reserved claim
exactly and in order, so a missing or mis-sized claim fails the invocation
instead of degrading it; there is no CPU fallback, no K-axis or limb tiling and
no budget knob.

| IR kind(s) | Admitted operation | Native allocation classes claimed | Range behaviour | Fixture |
|---|---|---|---|---|
| `MatrixBinary` add/sub, `MatrixNegate`, `MatrixScale`, `RingAutomorphism`, `Transpose`, `Slice`, `Tensor`, `Concat` | `Add`, `Subtract`, `Scale`, `Automorphism`, `Transpose`, `Slice`, `Tensor`, `ConcatRows/Columns`, `AddRowBlocks` | `Matrix` retained outputs; `BatchWorkspace`/`TransformWorkspace` per scratch slot | column ranges over any admitted width | `test_gpu_admitted_plans_execute_through_ordinary_fleet_matrix_calls`, `test_gpu_graph_execution_derives_complete_prepared_inventory` |
| `MatrixBinary` multiply, `MatrixMulAccumulate` | `Multiply{scales_left}`, `Accumulate` | fixed operand replica (`Matrix`), output `Matrix`, `BatchWorkspace` | scaled operand sharded; fixed operand replicated once per device | graph inventory fixture |
| `ConstantMatrix`, `Uniform*Sample`, `GaussianSample`, `HashSample` (matrix), `PolynomialValues`, `CenteredRebase`, `ModulusSwitch/Reduce`, `BlockModSwitch`, `RnsModUp/Down`, `CrtRecompose` | `Constant`, `Sample`, `Hash`, `Polynomial`, `CenteredRebase`, … | output `Matrix`, `SamplerWorkspace`/`TransformWorkspace` per range | seeded per-range sources | existing prepared fixtures (`gpu_view.rs`, `gpu_admission.rs`) |
| `GadgetDecompose`, `HashSample` decomposed/small-decomposed | `Decompose{hash}` | `Matrix` output, `CompactPayload` + `CompactWorkspace` for compact outputs, seeded COEFF source | per-range compact rectangles | `test_gpu_admitted_compact_decomposition_preserves_inputs_and_reuses_outputs`, `test_gpu_admitted_compact_hash_matches_column_runner_and_reuses_outputs` |
| `CenteredExtend` (small input) | `CenteredExtendCompact` | `CompactPayload`, `CompactWorkspace` (width-scaled) | `gpu_small_matrix_copy_range` per range | `test_gpu_admitted_compact_centered_extension_preserves_payload_and_shards` |
| `MatrixMulSmallRhs`, row blocks | `MultiplyCompact` | fixed EVAL LHS replica (`Matrix`), compact RHS (`CompactPayload`), width-scaled expansion `CompactWorkspace`, `CompletionEvent` | partial waves; row blocks admitted per block | `test_gpu_admitted_compact_multiplication_matches_cpu_over_partial_waves` |
| `Input` (canonical bytes, compact bytes, CPU staging) | `ImportMatrix`, `ImportCompact`, `ImportStaging` | `SubmissionStream` + `TransferWorkspace(Load)`; `PinnedHost` + `TransferWorkspace` + `CompletionEvent`; `CompactPayload` + `PinnedHost` + `CompletionEvent` (all width-scaled) | pinned staging retired per range | `test_gpu_admitted_imports_and_exports_preserve_canonical_bytes` |
| outputs (`matrix_to_bytes`, `small_matrix_to_bytes`) | explicit readback boundary via `PreparedClaimBroker::hold` | clone `Matrix` in the shard's format, `SubmissionStream`, store `TransferWorkspace` or `CompletionEvent` | whole shard | imports/exports fixture, graph inventory fixture |
| `TrapdoorSample`, `TrapdoorPublic`, `PreimageSample` | traced trapdoor boundary, `Preimage{plan}` | traced sequence of `Matrix`, `SamplerWorkspace`, `CompactWorkspace`, `CompactPayload`, `CompletionEvent`, `SubmissionStream` (covariance cache) | widths {1, target columns}; one dispatch extension per attempt | `test_gpu_graph_execution_admits_trapdoor_and_preimage_sampling`, `test_gpu_preimage_attempt_claims_are_traceable_and_reproduce_bounded_preimage` |

First use of a worker: pilots run through `measure_prepared_with_steps` on the
same device and inventory the production path uses, so the first admitted
invocation on a device exercises the same claim sequence the pilot measured;
`GpuDeviceWorkspace::acquire` and `GpuCudaResource::acquire` record every
first-time resource open in the claim trace. Metric: all admitted widths are
selected against `GpuCalibrationMetric::DefaultPoolIncrementalBytes` with the
exact native demand (`gpu_prepared_storage_demand`), and the estimator consumes
the resulting admitted plans (`AdmittedInvocationPlan` scenario) rather than a
nominal division of columns.

Kinds not in the table (`FamilyPack/Get*`, `Select`, scalar kinds) hold no
native owners; any other kind is rejected by `prepare_graph_admission` before
any node executes or any domain is sealed.

## 2. Validation scope run

Ordinary tests were run once; lifetime/synchronization-sensitive fixtures were
run three consecutive times on the same binary. Integration tests and physical
multi-GPU runs were not authorized and were not run. Evidence:
`test_data/fleet-sharding-implementation/final-2026-09-12/`.

Third pass, after every review-driven correction (`gpu-workspace.log`,
`cpu-workspace.log`, `gpu-executable-hashes.txt`, `source-manifest.txt`,
`diff-sha256.txt`; the two earlier passes are archived under `first-pass/` and
`second-pass/` with their outcomes):

| Suite | Result |
|---|---|
| `cargo test -r --workspace --lib --features gpu` | 1011 passed, 0 failed, 27 ignored; warning-free build |
| `cargo test -r --workspace --lib` | 610 passed, 0 failed, 25 ignored; warning-free build |
| Preimage graph fixture, 3 consecutive runs + ring dimension 512 | all pass |
| Derive-inventory graph fixture, 3 consecutive runs + ring dimension 512 | all pass |
| Primitive preimage claim-trace fixture, 3 consecutive runs | all pass |
| Estimator admitted-plan fixture, 3 consecutive runs | all pass |

The first pass exposed one stale fixture premise (the derive-inventory fixture
used trapdoor sampling as its unsupported example after Step 3 admitted it);
no assertion was weakened and no parallelism reduced to reach green.

## 3. Matched performance evidence

The estimator fixture measures the same `Add` (2x3) node twice with the same
parameters, harness (one warm-up, one measured iteration), device, transfer
lifecycle and timing boundary (execution-owner CUDA events plus coordinated
host wall): once under admitted plans from a prepared execution, once under
nominal placement (`estimator-matched-run{1,2,3}.log`).

| Run | Admitted work (s) | Nominal work (s) |
|---|---|---|
| 1 | 2.79e-5 | 2.77e-5 |
| 2 | 3.28e-5 | 2.97e-5 |
| 3 | 2.83e-5 | 2.66e-5 |

Caveat: this is a single tiny node on one device, so it establishes only that
the admitted path carries no measurable overhead at this scale and that the
estimator distinguishes the two scenarios. It is not a fleet throughput claim;
matched large-shape and multi-device performance stays within the deferred
physical multi-GPU scope.

## 4. Independent review

An independent reviewer (a read-only subagent, permitted by the plan for the
final review) examined the full dirty change set against `GPU.md` in four
focused passes: native lifetimes and ownership, admitted-operation arithmetic
versus the ordinary paths, admission and reservation semantics, and estimator
semantics plus test parallelism. No blockers were found. Every should-fix
finding was corrected and re-validated in the third pass (details in
`fleet-wide-gpu-implementation-status.md`, "Review-driven corrections"):

- admitted-plan log keyed by a stale operation identity;
- `PolynomialValues` admitted without a prepared runner;
- `hold_inner` masking a failed step's error;
- `trace_native_claims` not panic-safe;
- `gpu_matrix_dispatch_end` able to overflow its output while an extension
  was live;
- consumed-claim diagnostics able to throw after a committed claim;
- refused resource and workspace acquires leaving the handle unusable;
- `GpuTracedStepGuard` being `Send`;
- multi-device fleets accepted for trapdoor/preimage admission and failing
  only after sealing.

Verified clean by the review: reservation rollback and `next` advancement,
recycle/destroy re-entrancy, owner transfer, P1 covariance cache stream and
event ordering, atomic memory orders, no new synchronous `cudaMalloc` or
`cudaFree`, arithmetic equivalence of every admitted operation with its
ordinary path (compact extension, compact product, all three imports, hash
and gadget decomposition), exact wave classes and shared operation identity in
the estimator, Rayon parallelism preserved, no assertions weakened, no test
parallelism reduced, no CPU fallback, no K-axis or limb tiling, no new budget
knobs.

Recorded and intentionally unchanged: preimage seeding per tile start (admitted
and ordinary preimages coincide bit-for-bit only when tiles coincide; both are
valid under the same bound); root-scope-only inventory (graphs with loops or
subgraph calls are rejected in prepared mode before sealing); the hard-cutoff
decision synchronization at small-matrix destruction; `~GpuCudaResource` able
to overwrite a pending error message when it releases with unretired work.

## 5. Explicitly deferred

Physical two/three-GPU checks, peer transfer/lifetime coverage, role-1 pressure,
generation coherence across devices, and multi-device execution with
`RAYON_NUM_THREADS=1` remain deferred until the user reopens that scope. Resident
preimage targets that reach `preimage_target` (host download) and
transcript-replayed trapdoor imports remain outside prepared admission and fail
explicitly under a sealed domain rather than falling back. Nothing has been
pushed; the publication boundary is unchanged.
