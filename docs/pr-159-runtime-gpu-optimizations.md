# PR #159: runtime and GPU changes relative to main

This guide explains [PR #159](https://github.com/MachinaIO/mxx/pull/159) for a reader
who has not followed its implementation. It describes the complete PR, including
changes already committed before the final GPU review.

Comparison snapshot: `main` at `f3fe71efa301778f5ba178c985af919cfefa42c7`, PR branch
`codex/public-runtime-gpu-optimizations` at
`0a3d65b545551880866f6ec5a5b491f314fbe4d9`. Both refs were checked against the
remote on September 13, 2026. The scope is `git diff f3fe71efa301778f5ba178c985af919cfefa42c7...HEAD`,
not merely the last commit. This document is a subsequent documentation addition.

## 1. What the system does

The workspace implements lattice-cryptography computations. A polynomial matrix
contains polynomials rather than ordinary scalar entries. A polynomial can be
represented by residues modulo several small primes (CRT limbs), and in either
coefficient form or evaluation/NTT form.

Applications describe a protocol using the Rust DSL. The DSL produces an IR
(graph of typed operations). The runtime executes that graph using CPU or GPU
backends. An artifact store holds durable results that can be consumed by a
later execution. The benchmark estimator analyzes the graph and combines small
representative measurements to predict the cost of the larger protocol.

This PR changes how those layers share data, reserve GPU memory, schedule work,
and account for data movement. It also adds two exact modulus-conversion
operations and moves reusable Lean-checking support into the runtime layer.

## 2. Main changes at a glance

| Area | Change relative to main | Practical effect |
| --- | --- | --- |
| GPU scheduling | Explicit column intervals, bounded waves and persistent enqueue workers | A matrix can use several GPUs while retaining its existing ownership layout. |
| Memory admission | Native storage receipts, reusable inventory, retained-output and temporary-scratch reservations | The runtime checks whether work fits before submitting it and can reuse storage after the last owner releases it. |
| GPU primitives | Range operations, compact operands, direct views and explicit workspace sizes | Operations can process a part of a matrix without reconstructing or copying the complete matrix for each part. |
| Lifetime and transport | Producer/reader/release ordering and bounded pinned staging | Asynchronous copies and computations preserve their inputs until the final reader finishes. |
| DSL families | Shared captured fields, structural gather/zip and schema-based artifact imports | Shared data stays shared instead of becoming one copy or getter per loop member. |
| Runtime artifacts | RAM staging for intermediates and direct streaming of exports | Large protocols need not retain every intermediate on the GPU or write temporary families to disk. |
| Estimation | Representative measurement caching plus explicit transfer/dataflow costs | Repeating graph nodes changes the estimate, without executing the full protocol on the GPU. |
| Arithmetic and Lean | Centered extension, block modulus switching and reusable checking/linking support | New exact operations have CPU/GPU implementations and corresponding validation/semantic support. |

## 3. Column ownership and waves

A **shard** owns a contiguous range of matrix columns on one GPU. All CRT limbs
of that shard stay on that GPU. A **wave** is the set of column jobs submitted
at one scheduling step. A shard may require several waves when its temporary
workspace is larger than the currently available reusable storage.

For example, an eight-column result may have columns `[0, 4)` on GPU 0 and
`[4, 8)` on GPU 1. If each GPU can process two columns per wave, the first wave
processes `[0, 2)` and `[4, 6)`, and the second processes `[2, 4)` and `[6, 8)`.
The output owners retain all eight columns throughout; only temporary workspaces
are reused between waves. A short final wave uses its actual remaining width.
This example illustrates the mechanism, not a fixed partitioning policy.

For operations that inherit a matrix's layout, the runtime respects its source
owners and boundaries. Fresh outputs use capacity-aware distribution. Fixed
operands are prepared for their consumers, rather than reloaded for every job.
Unsupported direct peer transport is reported; it does not silently fall back
to reconstructing the matrix on the CPU.

The main source entry points are:

- `crates/runtime/src/gpu_schedule.rs`: intervals, jobs and lazy wave generation.
- `crates/runtime/src/gpu_enqueue.rs`: persistent per-device submission workers.
- `crates/runtime/src/backend/poly_gpu/gpu_compiled.rs`: compiled operation and range execution.
- `crates/runtime/src/backend/poly_gpu/fleet.rs`: GPU fleet backend and operation dispatch.

## 4. Admission and memory lifetime

Admission means deciding that a particular operation can use the available
resources before launching it. It is more precise than comparing the size of
one result with the GPU's nominal VRAM capacity.

At the initial graph preparation boundary, the runtime derives inventory demand
from types, operation workspaces and value liveness. It includes complete
retained outputs, normalization, nested scopes, carried loop values and aliases.
Reusable slots are sized by the peak live demand rather than by multiplying
scratch by the total number of loop iterations. Input partition alternatives
carry conservative capacity summaries; those summaries do not change execution
intervals.

Production preflight reserves the retained owners and fits temporary widths
against actual native storage slots. It does **not** launch a synthetic GPU
pilot at a newly encountered production node. Allocation-claim discovery for
polynomial readback, trapdoor sampling and preimage sampling requires an explicit
`prepare_graph_admission(..., warm_up = true)` call on the production backend.
Child scopes are included in that preparation. Ordinary `execute` uses false
and reports a missing plan rather than executing synthetic GPU work. Preparing
a new backend or only a different representative graph is not sufficient.

The separate low-level allocating path makes this distinction explicit through
`select_operation(operation, warm_up)`. Only `warm_up=true` permits a missing
calibration profile to trigger a pilot. Normal graph dispatch uses `false`;
prepared execution derives widths from native bounds without such a profile.

Native resource owners record allocation, production, reader completion and
release dependencies. Dropping a Rust handle does not imply that a GPU reader
has finished. Slots become reusable according to those native dependencies.
Pinned upload buffers are retired by the existing completion reclaimer; the
scheduler waits for its own staging slot to become writable before reusing it.
A final upload is not followed by an unconditional execution-wide drain.

The configured VRAM percentage is a planning budget, not a physical-memory
certificate. A later observation above that budget does not itself stop valid
production work. Real allocation errors or unsupported resource requests still
remain errors.

Relevant files: `gpu_inventory.rs`, `gpu_admit.rs`, and `gpu_prepare.rs` under
`crates/runtime/src/backend/poly_gpu/`; `crates/runtime/src/gpu_memory.rs`;
`crates/primitives/src/matrix/gpu_admission.rs`; and the native
`crates/primitives/cuda/src/gpu_admission.cu` and `Runtime.cu` implementations.

## 5. GPU primitive and transfer changes

The primitive layer adds matrix-range views and operations for arithmetic,
transforms, decomposition, compact multiplication, sampling and serialization.
A range identifies rows/columns in an existing owner. Compact small-coefficient
operands can remain compact instead of being expanded to full CRT matrices just
to participate in a multiplication.

CUDA batch operations use bounded metadata/workspace layouts and propagate
completion dependencies across the input readers and output writers. The NTT
and automorphism paths preserve evaluation-layout conventions, including the
bit-reversed ordering used by the GPU implementation.

Intermediate host snapshots use raw-RNS staging, preserving the matrix's
representation for reload. Durable artifacts use compact serialization instead.
Fleet snapshots pipeline at most two shard transfers with two pinned buffers,
so transfer scratch is not proportional to the total shard count.

See `crates/primitives/src/matrix/gpu_view.rs`, `gpu_staging.rs`,
`gpu_transform.rs`, `gpu_rns.rs`, and their implementations under
`crates/primitives/cuda/src/matrix/`. These are functional and resource-management
changes; this document does not claim a measured speedup for every kernel.

## 6. DSL families and runtime storage

A family is an indexed collection of graph values. A composite family can have
both shared and indexed fields. For example, each member of `(public_key,
member_result)` can refer to the same captured public key while keeping its
own result. The DSL records which fields are shared. A loop returning only
captured values can avoid creating a redundant parallel-loop node.

`Family::gather` expresses indexed selection structurally, and `zip_map` lowers
an aligned family operation once. Packing the ordered members of an existing
family preserves the original producer. `GraphValueSchema::artifact_input`
reopens exported fields using their scalar/family schemas.

The runtime releases values at their final live reference, including captures,
family-member aliases and loop-carried values. Intermediate GPU families are
staged in host RAM so live GPU work is bounded by the wave size. Retained host
data still requires RAM proportional to its live payload; there is no fixed
host-RAM cap and temporary families do not spill to disk. Root family
exports stream directly to their final artifact identities, and their manifest
is published after successful execution. Completed scalar exports can be
represented by artifact handles for subsequent access.

These changes affect `crates/dsl/src/control.rs`, `family.rs`, `value.rs`, and
`lib.rs`, plus `crates/runtime/src/executor.rs` and `artifact.rs`. Downstream
BGG, FHE, gadget and WE call sites are updated for the runtime/DSL API changes,
including mutable backend access when materializing outputs.

## 7. What the estimator measures

The estimator does not run the complete protocol to discover its cost. CPU
collection deduplicates requests by operation semantics and concrete parameter/
shape class before GPU measurement. If 1,024 nodes require the same measured
class, they reuse one measurement. Different shapes or parameter sets can
legitimately require separate measurements; the bound is not solely the number
of distinct operation names.

For column-separable work, the estimator measures representative wave classes
and applies their multiplicity to the full operation. Loop multiplicity is
handled by graph analysis. CUDA-event work is aggregate device-seconds; it is
not the wall-clock duration divided by the GPU count.

The PR also models runtime data boundaries separately: raw-RNS staging/reload,
compact artifact encoding/writing/reading/decoding, and a CPU dispatch proxy.
GPU-resident edges have no additional host-transfer charge. These boundary costs
contribute to the reported time without being mislabeled as primitive GPU work.

The reported GPU scenario is `SyntheticFreshPlacement`. Representative inputs
and capacities are not proof that a full application's exact layout, storage
contention or physical VRAM usage will match the prediction. An optional actual-
runtime observer exists for explicit diagnostics, but the estimator does not
accept a full-graph execution callback or collect its measurements that way.

See `crates/bench-estimator/src/gpu.rs`, `dataflow.rs`, `dataflow_gpu.rs`, and
`docs/benchmark-estimator.md` for formulas and measurement boundaries.

## 8. Exact arithmetic and Lean support

The IR gains `CenteredExtend` and `BlockModSwitch`. Centered extension lifts the
centered coefficient value into a containing CRT basis. Block modulus switching
performs exact block division with its plaintext-modulus/error-multiplier
contract. These are distinct operations, not interchangeable names for dropping
CRT limbs. Validation, DSL construction and CPU/GPU execution are updated
alongside the IR node definitions.

The CPU native implementation adds exact centered-basis conversion planning;
GPU implementations provide matching conversion operations. Relevant files are
`crates/ir-core/src/node.rs` and `validate.rs`,
`crates/primitives/native/ExactBasis.cc`, and the polynomial/matrix conversion
wrappers and CUDA sources.

Reusable generated-Lean checking moves from the WE application into
`crates/runtime/src/lean/check.rs`. It checks local modules in dependency order,
handles missing dependencies and timeouts, and checks the final linked
certificate under the configured closed axiom policy. The IR linker and runtime
Lean modules are extended for decomposition and matrix operations. Primitive
bounds distinguish an L1 gain (sum of coefficient magnitudes) from a maximum
single coefficient. These source changes do not, by themselves, establish a
new end-to-end cryptographic security claim.

## 9. Repository changes and validation status

`docs/plans/` is removed from Git tracking and added to `.gitignore`; local files
are retained. Documentation should be read against the current source, since
local historical plans may describe superseded designs.

Before the physical run, local validation included warning-free CPU/GPU workspace
library builds, 61 estimator tests, and 260 applicable runtime tests (259 in the
suite plus the corrected obsolete calibration-count test on targeted rerun).
Earlier wider workspace validation is recorded locally, but is not a substitute
for physical multi-GPU execution of this commit.

### Physical four-GPU validation (September 13, 2026)

RTX 4080 and RTX A6000 were unavailable in a four-GPU pod configuration. The
approved replacement was one Secure Cloud pod with four A40 GPUs in EU-SE-1,
30 GB container storage and 100 GB workspace storage. The GPU total was
USD 1.96/hour, excluding storage.

The initial request was blocked by account credit. After funding, the network-
volume service reported that EU-SE-1 does not support separate network volumes.
The run therefore used a 100 GB pod-attached workspace volume. No separate
network volume was created; deleting the pod also deletes its attached storage.

Pod `is4giiemek3mrp` (`mxx-pr159-a40x4`) used image
`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. All four A40 devices reported
46,068 MiB and driver 580.159.04. CUDA peer-access queries succeeded in both
directions for all six physical GPU pairs, including cross-NUMA links.

The initial code revision was `d7ab3e004d8e83daa11202ef54d96bbea5347c74`.
The release workspace library build with `--features gpu` passed without
compiler warnings. The runtime suite passed 257 tests and failed four, with
four ignored; primitives passed 299 and failed two; the estimator passed 59
and failed two. The explicitly enabled fleet peer regression selected one bidirectional GPU
pair and passed three runs. Three direct peer-copy/reader/lifetime tests passed three runs each on
all six physical device pairs (54 successful runs). These successes did not
make the full test suites green.

The failures exposed the following issues, corrected in `05b5e9fe4`:

- A candidate column width could make retained output and scratch requests
  compete for the same prepared slot. Planning now rejects that candidate as
  infeasible and searches a smaller width. Concrete reservation validation still
  rejects invalid requests.
- Batched RNS readback used additional consumer events when CRT limbs had
  different producer streams. It now records one completion per transfer batch
  and shares it with the snapshot and all limb readers. The prepared-resource
  count query follows the same rule. This retains reader dependencies without
  extra per-limb event claims or a global synchronization.
- Two tests configured only the GPU-0 width but accidentally constructed a
  fleet backend. They now explicitly select their intended single device;
  fleet and peer tests continue to exercise all four visible devices.
- A measurement-cache test used a fixed prime incompatible with some allowed
  ring dimensions. It now obtains valid primes from the existing CPU parameter
  constructor. Its 1,024-node/one-measurement assertions remain intact.
- A multi-partition allocation test compared workspace bytes against the total
  auxiliary allocation, which also includes descriptors. It now checks the
  dedicated workspace-byte field against its original exact expectation.

The test environment uses `CUDA_ARCH=86`, `CUDA_VISIBLE_DEVICES=0,1,2,3`,
`RUST_LOG=debug`, `MXX_PRIMITIVE_TEST_RING_DIMENSION=32` and
`MXX_PRIMITIVE_TEST_MATRIX_SIZE=5`. After sourcing the pod workspace environment, the build command is:

```sh
cargo test -r --workspace --lib --features gpu --no-run --message-format=json
```

The produced `mxx_runtime`, `mxx_primitives` and `mxx_bench_estimator` binaries
are run with `--test-threads=1 --nocapture`. Pair checks override the visible
GPU list with each of `0,1`, `0,2`, `0,3`, `1,2`, `1,3` and `2,3`.

The runtime's `test_gpu_column_plan_executes_retained_outputs_and_reusable_scratch`
and `test_gpu_prepared_fleet_transaction_transfers_to_workers_and_retains_readers`
enumerate all detected devices, allocate device-local owners and execute work
through per-device workers. With this environment they exercise four devices.
The ignored peer fleet regression instead deliberately selects a two-device
peer-capable backend; its success is reported as two-device fleet coverage.

At `05b5e9fe4`, the original eight failing tests passed. The estimator suite
passed all 61 tests, the three affected readback/owner tests passed three runs
each, and fleet plus six-pair peer checks passed all 57 runs. Other fixtures
still reserved the old per-stream event count; their exact-consumption checks
correctly rejected the unused reservations. Revision `d9c45185f` updates those
single-partition readback fixtures to one shared event while retaining output,
occupancy, lifetime and exact-consumption assertions.

At `d9c45185f`, primitives and the estimator passed their full suites. One
runtime transaction fixture still intermittently reentered a thread-bound
permit: its CPU-to-GPU conversion invoked nested Rayon work, which could steal
another dispatch onto the same host thread. Revision `0a3d65b54` constructs the
same rectangular identity using GPU zero/fill operations inside the permit.
CPU reference construction remains outside it; device/job parallelism,
transpose, early input release and canonical output checks remain unchanged.

The final runtime check at `0a3d65b545551880866f6ec5a5b491f314fbe4d9`
passed all four commands with a warning-free release build:

| Coverage | Validated revision | Result |
| --- | --- | --- |
| Runtime library suite, four GPUs visible | `0a3d65b54` | 261 passed, 0 failed, 4 ignored; 18.62 s |
| Four-device transaction/reader lifetime | `0a3d65b54` | 3/3 targeted runs passed |
| Primitive library suite | `d9c45185f` | 301 passed, 0 failed; 97.95 s |
| Estimator library suite | `d9c45185f` | 61 passed, 0 failed; 1.12 s |
| P1 completion-slot reuse | `d9c45185f` | 3/3 targeted runs passed |
| Explicitly enabled two-device peer fleet regression | `d9c45185f` | 3/3 runs passed |
| Three copy/reader/lifetime tests on each of six GPU pairs | `05b5e9fe4` | 54/54 runs passed |
| Prepared pilot, snapshot/reclaimer, and resident-row input lifetime | `05b5e9fe4` | 3/3 runs per test passed |

The revisions after `05b5e9fe4` only adjust test fixtures. Primitive and estimator
results were reused after the last runtime-test-only change; there was no
additional production-code change requiring those suites or all six pairs to
run again. The runtime's ordinary suite still reports four ignored tests;
the peer fleet regression is separately enabled as shown above.

The final targeted build command was
`cargo test -r -p mxx-runtime --lib --features gpu --no-run --message-format=json`.
Earlier workspace GPU library builds also passed without compiler warnings.
All successful command logs were checked for a nonzero passed-test count.

The complete log archive was retrieved and its SHA-256 checked against the
remote archive:
`1537d74b9217a5fb743bb1eaf17cfacedb166124612b1171a286b1a98945073e`.
After verification, Runpod confirmed deletion of pod `is4giiemek3mrp` and its
attached 100 GB workspace. No separate network volume was created or retained.

Run artifacts include exact commands, environment variables, source and binary
hashes, topology, individual exit codes, and approximately three-second VRAM
samples. Logs are retrieved to `logs/pr159-a40x4-20260913/` in the primary local
checkout. No application integration test or production-size throughput
benchmark is implied by these unit-test results.

## 10. Follow-up on the PR review

The review against `51bfdbd1eb` was checked against the current implementation.
The useful outstanding corrections are:

- Public import timing starts with a fresh store verification cache on every
  iteration, so warmup cannot remove content-hash verification from timed loads.
- Family export accounting distinguishes unique exported indices from export
  declarations. Aliases count as additional writes without making unrelated
  members artifact-backed; those members retain their staging/reload costs.
- If ordinary execution fails after an eager scalar export, abort cleanup removes
  the unreachable artifact. Session execution retains it for resume.

The ring-dimension lookup key and CenteredExtend column-separability expectation
were already corrected. Dynamic family access instead has a documented input
contract: all possible members must have the same transfer state. No new
validation pass was added. The trapdoor-shrinking suggestion targets an older
measurement contract; current transfer calibration deliberately measures full
canonical artifacts, as specified in `docs/benchmark-estimator.md`.

The new duplicate-export and failed-execution regressions passed, including the
session-retention case. Existing dataflow and canonical GPU transfer tests also
passed on a local RTX 4080 SUPER. This follow-up does not change GPU lifetime or
synchronization code and did not provision another remote pod.

## 11. Explicit resource-discovery warmup

Normal execution now requests cached resource plans only. A missing polynomial
readback, trapdoor or preimage plan is a warmup error, including in nested
scopes; it never starts a replacement GPU discovery trial. Call
`prepare_graph_admission(&graph, capture_trace, &inputs, true)` explicitly,
drop its guard, and retain that backend for production. See
`docs/benchmark-estimator.md` for the calling contract.

CPU and GPU workspace library builds passed without warnings. On the local
RTX 4080 SUPER, the final runtime library suite passed 262 tests with four
ignored. The three affected graph/readback/preimage fixtures also passed three
consecutive standalone runs each, covering warmup guard disposal, cold-cache
rejection, successful warmed execution and rejection after a cached plan is lost.
Independent review found no actionable issue. No integration or multi-GPU test
was run for this change.

The first suite run exposed an over-wide warmup in the forced-width tail fixture;
its explicit warmup now uses the fixture's requested width. Two unchanged manual
inventory fixtures (canonical import/export and polynomial constants) also failed
that first suite's admission checks, then passed both standalone and in the final
suite. Their initial failures were not reproduced or diagnosed by this change;
the passing final run does not establish the absence of intermittent failures.

## 12. Lazy checkpoint input admission

An artifact descriptor does not own a GPU matrix. Admission now tracks lazy
provenance through scalar inputs, family packs, selections and child scopes.
It reserves imported owners when a consumer materializes them, and retains
cached owners through their last use and live aliases. Unselected descriptors
do not become native payload reservations merely because their family is live.
This makes a lazy family and an equivalent pack of scalar artifact inputs use
the same inventory for the same selected members.
`Select` reserves the chosen owner's worst-case lifetime, including later uses
of its cached candidate, rather than importing every candidate in the planner.

Real imports still reserve matrix or compact staging, transfer workspaces,
streams, events and pinned buffers. Broadcasting an in-memory packed family
still reserves every member that placement actually imports. Lazy root outputs
retain import resources for later materialization. Unlike an input family used
only for selection, a returned family must reserve all members that the existing
`materialize_output` API loads and retains together; transfer scratch is reused.
The added output-family regression caught and corrected a one-member reservation
in the reference implementation. Budget rejection reports
requested and available bytes per device; it does not disable admission or
permit new discovery trials during production.

The regression coverage compares ordinary and compact artifact representations
at different family sizes, preserves mixed-family broadcast reservations, and
executes repeated uses of a selected/cached imported matrix across temporary-buffer
reuse, and materializes a complete lazy output family after execution.
The broadcast fixture uses a runtime remainder index because direct and offset
loop-index selections lower to Zip/ZipOffset and import only the selected member.

Local RTX 4080 SUPER validation passed all three lazy-inventory regressions on
three consecutive runs each. The final runtime library suite passed 264 tests
with four ignored. These runs include the additional `Select` lifetime and lazy
output-family corrections, rather than merely compiling the reference patch.
CPU and GPU release workspace library builds completed without warnings.
An independent review accepted the final selection-lifetime and output-family
corrections; execution evidence comes from the local GPU runs above.

This is a runtime-only change. No application-specific checkpoint driver or
private application crate is included. A successful full production checkpoint
replay and production-size utilization measurements remain separate validation
requirements; library tests do not establish either result.
