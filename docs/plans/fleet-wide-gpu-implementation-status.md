# Fleet-wide GPU implementation status

> **Stopped by the user on 2026-09-12.** The current handoff is
> [Fleet-wide GPU implementation handoff](fleet-wide-gpu-handoff-2026-09-12.md).
> Latest evidence: CPU 610 passed / 0 failed / 25 ignored; GPU-enabled units
> 999 passed / 2 failed / 27 ignored. Earlier all-green checkpoints below are
> historical and do not describe the latest binaries. Implementation must not
> resume without a new user instruction; physical multi-GPU checks remain deferred.

This is an implementation checkpoint, not an acceptance report. The authoritative
contract remains [the implementation plan](fleet-wide-gpu-column-sharding.md).
The complete requested implementation and all non-ignored unit tests were the
objective before the stop instruction above. The user permits subagents only for the
final independent review after the current edits. The latest validation instruction
allows one successful run for changes unlikely to cause GPU synchronization
failures. Synchronization-sensitive changes retain applicable repetition gates.

## Earlier checkpoint (2026-09-12; superseded by the stop-time handoff)

The latest CPU and GPU workspace unit builds are warning-free. Evidence covers
999 passing GPU-enabled unit tests with 27 ignored and 610 passing CPU unit tests
with 25 ignored. Seven changed GPU executables ran once; four unchanged GPU
executables and all eleven CPU executables reuse passing evidence after exact
SHA-256 comparison. The compact decomposition lifetime fixture passed three
targeted runs and one N=512 run; the full primitive suite passed 299 tests once.
Sources remained frozen during each build/test sequence. The hardware remains
one RTX 4080 SUPER. Physical multi-GPU checks are on hold at the user's request.
These results do not establish complete production admission.

Device scratch allocations route through typed workspaces, including RNS and
compact transfers and matrix equality. Runtime preflight and production share
sampler, arithmetic, transform and compact block range runners. Prepared CUDA
event and private submission-stream slots now have independent unit counters
and exact dispatch claims. The managed allocation audit now closes all four native admission domains at
an explicit complete-inventory setup boundary. Actual initial native residency
receipts can install the runtime ledger through `prepare_memory`. The approved
policy does not require a prospective bound on opaque CUDA queued-launch
consumption. Automatic provisioning and complete invocation coverage remain
unfinished.

The user approved production physical-budget excess on 2026-09-11 and explicitly
requested continued processing while resources remain available. The
[accepted decision](gpu-driver-memory-contract-decision.md) replaces the earlier
proposal to stop new submissions after detecting excess. The authoritative plan
and accounting amendment now reflect this decision. No further approval is needed.

The ledger update separates accepted initial residency and live managed charges
from later observations. Observed excess does not stop valid managed admission;
completed native frees return managed capacity even if pool pages remain resident.
Setup still checks both initial residency dimensions. The default-pool diagnostic
scheduler proposes at least the supported class minimum when target headroom is
exhausted; this does not grant managed admission. Estimator transfer fixtures log
their approximate allowance/target mismatch and observed physical excess, then
attempt the full declared shape. Actual CUDA failures still propagate.

Final validation is recorded under
`test_data/fleet-sharding-implementation/soft-physical-budget/`. Both workspace
unit builds are warning-free. Final evidence covers 962 GPU-enabled passes
(27 ignored) and 608 CPU passes (25 ignored). Five changed GPU executables ran
once; the other six GPU executables and all eleven CPU executables reuse passing
results only after exact SHA-256 comparison. All 68 source-manifest entries and
all 22 executable hashes were checked after the run. Runtime's 217 tests passed
three consecutive executions of the final binary, including real prepared
matrix execution after a synthetic over-budget observation. Synthetic observations
exercise admission behavior and do not claim actual physical-budget stress.

The initial full run had one failure in the existing
`gpu_runtime_pilots_reslice_wide_matrix_inputs` test: a context-generation change
invalidated its calibration interval. That initial failure, source identity and
logs are preserved under `initial-`. The same failing fixture then passed three
isolated runs; the initial candidate and the preceding baseline each passed
three full runtime runs. The final candidate also passed three full runtime runs.
No numerical assertion or contamination rejection was weakened. The cause of
the initial transient failure remains unestablished; successful repetitions do
not prove its absence.

Five alternating before/after runs of the unchanged wide-input calibration fixture
had median whole-process times of 0.2458 seconds before and 0.2199 seconds after
(ranges 0.1701-0.3226 and 0.1408-0.2717 seconds). The variability is substantial;
these timings do not establish a kernel speedup. Raw commands and times are in
`final-reference-timing.json` and its corresponding logs.

Default production admission, automatic import/setup provisioning, remaining
invocation families and estimator plan consumption remain required. Physical multi-GPU validation and final synchronization/performance
acceptance are outstanding. Unit success is not completion of these contracts.

A final independent read-only review of this soft physical-budget stage and its
estimator extension found no actionable code or documentation issues. This scoped
review does not accept the unfinished fleet-wide implementation. No implementation
subagents were used for this stage. An earlier full-plan reviewer attempt failed
with `agent thread limit reached`; no full-plan review acceptance is claimed.

### Admitted compact decomposition

The existing compiled column runner now retains either an ordinary matrix or a
bounded compact matrix, without constructing a full-DCRT output proxy. Decompose
and row-block decompose requests validate their actual parameter owners, levels,
block shapes, digit layout and output coefficient bound before publication.
Complete outputs reserve `CompactPayload` slots; normalization uses the existing
fixed-input planner and leaves original source formats intact. Compact zeros are
initialized on their existing CUDA stream before becoming safe public values,
using the payload's existing completion event and no host upload.

Regular approximate decomposition reserves one range-sized coefficient copy per
row block in dispatch order. Matrix scratch now preserves each block's row count
when column widths change. Additional correction metadata is reserved in source
order as fixed `TransformWorkspace` spans. The native correction query decides
whether auxiliary storage suffices at one column; if it does not, every wider
range uses the same separate span. This keeps allocation order and size stable
when a short tail follows a wide range. The owned ordinary/compact paths use the
same query; no per-wave allocation escapes its reservation.

The profile identity includes the decomposition mode, requested digits, input row
layout, resolved dropped moduli and bound. The ordinary and compact public result
types remain separate; the compiler checks the requested result kind before
consuming the queued invocation. Repeated batches reserve their own output
owners and reuse the admitted backing after release.

Evidence is under
`test_data/fleet-sharding-implementation/admitted-compact-decomposition/`.
The mixed-format, differently sharded row-block fixture passed three identical
runs and one N=512 run against existing CPU compact decomposition. It covers both
small/regular and exact/approximate layouts, malformed request rejection, queued
operand identity, canonical output bounds and repeated output reuse. The separate
64-limb fixture passed three runs with actual correction workspace allocation,
width-two jobs and a width-one tail, including a second invocation on the same
backing. CPU references are existing decomposition/canonical serialization.

Initial build diagnostics remain under `initial-build-` and `fixture-build-`.
The first runtime fixture mistakenly passed dropped-moduli count as the context's
partition count; complete-inventory setup correctly rejected its foreign owner.
Those four failed runs remain under `initial-runtime-`. The wide-basis fixture
initially provisioned too little free prepared capacity for a width-two slope
hint; its width assertion failed three times before computation. The fixture now
provides six ordinary slots of width two so the measured candidate reaches the
native width-two limit. These results remain under `capacity-fixture-`; admission
arithmetic and existing tests were not weakened.

The unchanged fleet row-block reference passed on saved baseline/final candidate
in 0.2441/0.2339 whole-process seconds. This checks the existing execution path;
it does not measure the newly admitted runner or establish final performance
acceptance. Runtime compact multiplication, hash decomposition, small extension,
preimage and automatic provisioning remain separate required work.

### Retained compact decomposition ranges

Compact decomposition now accepts borrowed ordinary row/column rectangles and
writes directly into a retained compact destination rectangle. Public layout
metadata (`rows_per_input_row`, dropped moduli and the inclusive coefficient
bound) is shared by owned and borrowed Rust entry points. Native validation
independently checks the destination shape/bound and actual CRT parameters.

Coefficient inputs that need no approximate correction are read directly.
Evaluation inputs and approximate inputs use only range-sized GPU copies;
independent inverse transforms are grouped by row count and submitted with Rayon.
Copy allocations and any correction workspace claims remain in dispatch-thread
order. The existing owned-block path retains its in-place batched normalization
and correction, with no new payload copy. The compact kernel maps input pitches
and a retained output offset/pitch directly. Linear output-coefficient indexing
keeps one kernel launch without a grid-Y limit on the number of input polynomials.
Bounded block metadata and numerical digit extraction retain their previous
contracts. Prior readers and source release reuse output completion events.

Evidence is recorded under
`test_data/fleet-sharding-implementation/retained-compact-decomposition/`.
The new fixture passed three identical-command runs and one N=512 run. It checks ordinary/small decomposition, exact/approximate layouts,
mixed narrow/wide CRT limbs, coefficient/mixed input formats, differently pitched
row blocks, short final ranges, pending readers and untouched output borders.
Its oracle is existing CPU compact decomposition and block copying. It also checks
fresh output through the same borrowed API and drops original owners before
reading snapshots. The mixed-width cases borrow correction metadata from their
copy's existing auxiliary storage; they do not exercise additional correction
workspace allocation.

Initial fixture runs attempted to rearm an empty reservation although no claims
had been consumed. The initial logs/build/source identities are preserved under
`initial-`; empty permits now continue unchanged. The next fixture runs reached
export of coefficient-format source snapshots, whose ordinary CPU export would
allocate an unreserved evaluation copy. These logs are preserved under
`readback-fixture-`. The snapshot now transforms its already-owned storage in
place before export, keeping the source and comparison assertions unchanged.
The next runs reached the CPU oracle's `BoundExceeded` rejection for small-mode
full-ring random input. Compact small decomposition requires common bounded
unsigned digits across CRT towers; arbitrary tower-wise digits cannot be encoded
as shared bounded integers. The small-mode fixture now samples common unsigned
coefficients below every prime (random bits scaled by the minimum prime minus
one), while regular decomposition retains full-ring random input. The failed
oracle runs are preserved under `small-input-fixture-`. All numerical, border and
lifetime assertions remain. No production reservation check or existing regression
test was weakened.

The unchanged owned-row-block fixture passed for the saved baseline and final
candidate in 0.2898/0.2788 whole-process seconds. Those times include setup and
assertions; they are not kernel timings or final matched-production acceptance.

Both CPU/GPU workspace unit builds are warning-free. Final evidence covers
999 GPU-enabled and 610 CPU unit passes (27/25 ignored), with seven changed GPU
executables run once and four unchanged GPU plus eleven unchanged CPU executables
reused only after exact SHA-256 comparison. All 73 modified/untracked crate-source
hashes and 22 executable hashes match their manifests; the six target/timing
records pass, and `git diff --check` is clean. No integration or physical
multi-GPU tests were run. Runtime compact admission, retained hash decomposition,
`CenteredExtendSmall`, trapdoor/preimage and the other remaining work below are
still required.

### Retained compact RHS multiplication ranges

The compact RHS row-block primitive can now write directly into retained ordinary
matrix rectangles. Each fixed LHS may itself be a row/column view with a different
physical pitch. Bounded block metadata maps those logical coordinates without
copying the operands. All row blocks share the same compact RHS view and its one
logical expanded workspace over all inner rows and active CRT limbs. Mixed
32/64-bit limb partitions keep their existing typed native spans and DIF/NTT
kernels. No K-tiling, full compact-payload expansion outside the current columns,
new pool, mutex, or host completion wait was introduced.

`small_rhs_workspaces` exposes exact nonzero native spans in allocation order.
The public query and native production allocation share the same checked size
calculation. The ordinary multiplication API now calls the range primitive with
whole-matrix views; its mathematical and allocation-report checks remain intact.
Full owner byte queries and the existing budget check run before output allocation.
Independent CPU size queries use Rayon. Output allocation stays on the dispatch
owner's thread to consume its ordered thread-local native claims.

Evidence is recorded under
`test_data/fleet-sharding-implementation/retained-small-rhs-ranges/`.
The new strict prepared-storage fixture passed three identical-command runs and
one ring-dimension-512 run. It covers all-narrow, all-wide and mixed-width bases,
different input/output offsets and pitches, multiple row blocks, a short final
column range, exact expansion-slot rearming, pending old readers, untouched
borders and immediate source/RHS/output release after enqueuing downstream work.
Expected products use the existing CPU compact multiplication primitive.

The initial four executions failed during the fixture's downstream ordinary
transpose, which requested unreserved completion resources after setup closure.
Those failures and exact source/build identities are preserved under `initial-`.
The fixture now uses the existing prepared range transpose while retaining the
same full numerical, border and lifetime assertions. No production reservation
or existing test was weakened. The unchanged compact-view fixture took
0.2487/0.2464 whole-process seconds for the saved baseline/final candidate. These include
setup and do not establish a kernel speedup or final production acceptance.

The subsequent full GPU unit suite exposed a setup-closure race: the tracked
pinned-host reclaimer could still be retiring initial uploads when closure
sampled the active-allocation counter. The failed run (297 primitive passes and
one failure) and exact source/build identities are preserved under `setup-race-`.
The reclaimer now tracks the complete retirement scope through the existing
device activity counters. Foreground allocation calls still use both owner and
device counters, and setup closure still rejects foreground activity, live
reservations, leased slots, timing and uncertain releases. The following initial
allocation epoch continues to fence setup streams, drain reclamation and require
idle coherent counters before issuing a residency receipt. No wait, mutex, retry
loop or weakened residency receipt was added. An intermediate broader closure
change was replaced before final validation; its passing evidence is retained
under `draft-closure-`.

Final evidence covers all 11 GPU-enabled and 11 CPU workspace unit executables:
998/610 passes, 27/25 ignored, no failures. The final GPU primitive executable
passed all 298 tests three times under the identical full-suite command. Its
three targeted lifetime runs, N=512 run and baseline/candidate fixture runs are
retained unchanged after the later artifact-only repair because the executable
SHA-256 is identical. Both builds are warning-free; all 73 source hashes and 22
executable hashes were checked, and `git diff --check` passed. The original
failures remain recorded separately. These are unit and lifetime results, not
full production admission, matched production performance or multi-GPU acceptance.

This completes the retained multiplication primitive and its exact workspace
query. Runtime compact invocation admission, retained compact decomposition/hash,
trapdoor/preimage, automatic graph-wide provisioning, import/export and scalar
boundaries, estimator plan consumption and final independent review remain
required. Physical multi-GPU validation stays deferred.

### Zero-row gadget validation

CPU/device constants, fleet constant runners and compiled invocation validation
now reject zero-row gadget declarations with `InvalidInteger` before dividing
columns by rows. Both regular and small gadgets reject `(0, 0)` and `(0, 1)`;
zero-row gadgets have no inferred digit count. Unit coverage checks each entry
point. Existing nonempty gadget and ordinary empty-zero constant fixtures retain
their mathematical assertions. This is input validation, with no GPU lifetime or
synchronization change, so its unit coverage runs once in each relevant suite.

### Explicit artifact session lock release

The CPU suite exposed one existing failure in
`file_store_persists_session_alias_transcript_and_finalization`: reopening after
`drop(store)` returned `SessionBusy`. The failed log and exact executable/source
identities are preserved under `artifact-race-` in the compact-range evidence
directory. The zero-row gadget test itself passed.

The former implementation released a session lock only by closing its file.
A duplicated or fork-inherited descriptor keeps that open-file-description lock
alive; a concurrent process launch can therefore extend ownership past store
destruction. This mechanism follows the [Linux flock contract](https://man7.org/linux/man-pages/man2/flock.2.html);
the original failed run did not capture which child inherited the descriptor.
`FileArtifactStore` now explicitly unlocks on session release, store destruction
and failed session persistence after acquisition. Explicit release propagates
unlock errors. No lock was added and test parallelism was not reduced.
A deterministic unit test keeps a duplicate descriptor alive across both release
paths, then checks that a new session can open and retains exclusive ownership
even after the old duplicate closes. The existing persistence test is unchanged.

### Prepared plaintext CRT recomposition

Plaintext CRT recomposition now executes through retained column-range admission.
Its common native arithmetic still reconstructs each source integer, applies the
existing exact nearest plaintext scaling, and accumulates signed reconstruction
coefficients in the registered destination basis. Logical input/output row and
column offsets are independent of physical owner pitches. Evaluation inputs
share explicitly reserved coefficient normalization; coefficient inputs remain
borrowed and unchanged. Only the written destination row range is transformed
back to evaluation form.

The metadata now references immutable setup-owned Garner tables rather than
copying a complete inverse table for every input level. One or two levels fit a
bounded by-value launch. Larger retained invocations use one exact typed GPU
workspace and setup launches carrying at most two records each; no per-wave
pinned metadata can remain in flight. Ordinary full-matrix calls share the
arithmetic and retain their staged-workspace path. Source reader tracking reuses
the output completion event, and submission failures retire output work before
owners can be released. CPU normalization submission and public residue
preparation use Rayon.

Admission prepares each secondary input against its own registered CRT basis.
Compiled binding checks its shared execution owner and coefficient format while
preserving its active level. Calibration identity now records actual bases and
levels for all secondary operands, as well as the destination, plaintext moduli
and reconstruction coefficients. A request with substituted inputs fails before
consuming its queued invocation.

Evidence is in
`test_data/fleet-sharding-implementation/prepared-crt-recompose/`.
Both CPU/GPU workspace unit builds are warning-free. Evidence covers 994
GPU-enabled passes (27 ignored) and 608 CPU passes (25 ignored). Seven changed
GPU executables ran once; four unchanged GPU executables and all eleven CPU
executables reuse passing evidence after exact SHA-256 comparison. Runtime
passed 234 tests and primitives passed 297. All 71 source hashes and 22 executable
hashes were verified after execution; crate sources remained frozen during tests.
The two new lifetime/reuse fixtures passed three identical-command executions
and one ring-dimension-512 execution each. Primitive checks cover one/two/five
levels, mixed-width and reordered bases, offset rectangles, both output formats,
empty rectangles, preserved borders and prior readers, and immediate owner
release. The runtime fixture verifies mixed bases against the existing CPU
recomposition oracle, different source column boundaries, both input formats,
small/large metadata, cached cycles, four profiles and strict complete-inventory
admission. Existing whole-matrix CRT rounding coverage also passed in the
baseline/candidate comparison.

No runtime test failed. The first warning-free build snapshot is preserved under
`draft-fixture-`; a test assertion was corrected before execution so its rejected
call actually substitutes a different input. Existing tests and numerical
assertions were unchanged. The unchanged whole-matrix CRT fixture took
0.2506/0.2558 seconds for the saved baseline/candidate. These are whole-process
measurements including setup, not a kernel speedup or final production acceptance.

Automatic graph-wide provisioning, compact hash/decomposition/trapdoor/preimage,
import/export and scalar boundaries, zero-row gadget handling, estimator plan
consumption and final independent review remain required. Physical multi-GPU
validation remains deferred at the user's request.

### Prepared RNS conversion ranges

RNS ModUp and ModDown now execute through retained column-range admission.
The shared primitive preserves the existing approximate RNS equations and
normalization, validates complete ordered bases and destination dimensions,
and maps each ModUp digit group to the correct logical row block regardless of
source/destination offsets or pitches. Prepared coefficient normalization is
shared without modifying original input formats. Range NTT touches only the
written output rectangle; callers may also retain coefficient-form outputs.

Small RNS plans retain their bounded by-value launch metadata. Larger plans
claim one exact GPU `TransformWorkspace` span, generated by the existing GPU
setup kernel. The retained-range query explicitly requires this span regardless
of the output owner's auxiliary capacity, so pilot and production widths share
one contract. There is no pinned metadata buffer to overwrite between waves.
The output completion event protects source readers; errors retire submitted
work. CPU preparation of public digit products now uses Rayon alongside the
existing parallel inverse/weight preparation. No new pool, mutex, host wait,
or CPU data fallback was introduced.

Runtime identity distinguishes ordered source bases, target types, digit size,
normalization and plaintext modulus. Invalid shapes/bases and calls that differ
from the admitted request fail before consuming the queued invocation. The
ordinary whole-matrix API shares the range implementation and retains its
auxiliary-workspace query behavior.

Evidence is recorded under
`test_data/fleet-sharding-implementation/prepared-rns-ranges/`.
Both CPU/GPU workspace unit builds are warning-free. Evidence covers 992
GPU-enabled passes (27 ignored) and 608 CPU passes (25 ignored). Seven changed
GPU executables ran once; four unchanged GPU executables and all eleven CPU
executables reuse passing evidence only after exact SHA-256 comparison. Runtime
passed 233 tests and primitives passed 296. All 71 source hashes and 22 executable
hashes were verified after execution, with crate sources frozen during testing.
The new primitive and prepared-runtime lifetime fixtures each passed three
identical-command runs plus one ring-dimension-512 run. They cover compact and
large metadata, mixed-width and reordered bases, both input/output formats,
ModUp row expansion, normalization, ModDown, preserved borders/prior readers,
immediate owner drops, empty rectangles, cached execution cycles and strict
complete-inventory resource admission. Numerical references come from existing
CPU primitives. Runtime coverage explicitly exercises large-workspace reuse with
five distinct conversion profiles.

Initial compiler diagnostics from two unqualified fleet invocation references
are preserved under `initial-`. Qualifying those references resolved the build
failure. No runtime test failed, and no existing numerical assertion or resource
bound was weakened. The unchanged RNS compact-boundary fixture took
0.2742/0.2701 whole-process seconds on the saved baseline/candidate. These include
setup and do not establish a kernel speedup or final production acceptance.

Automatic graph-wide provisioning, CRT recomposition, compact hash/trapdoor/
preimage, import/export and scalar boundaries, zero-row gadget handling,
estimator plan consumption and final independent review remain required.
Physical multi-GPU validation remains deferred at the user's request.

### Prepared modulus conversion ranges and resident CRT constants

Exact reduction, nearest modulus switching, centered extension and exact BGV
block switching now execute through retained column-range admission. The shared
primitive validates basis containment/subsets, execution ownership, active levels
and plaintext invertibility before submitting work. Public inverse preparation
uses Rayon. Coefficient inputs remain unchanged; shared prepared normalization
supplies coefficient form where required. Reduction preserves the input format;
other admitted conversions produce evaluation-form output in the registered
destination basis. Range NTT touches only the newly written rectangle.

Retained conversions use bounded by-value launch metadata and an immutable GPU
Garner table uploaded during ring setup. Each table requests `8 * L * L` bytes
per registered ring and device, with `L` the full source CRT depth. Its physical
residency is included in the existing initial native receipt. The ring-constant
owner now contains both NTT and CRT constants and releases them through the
existing owner release stream. The new kernel is included in native prewarming.
The common arithmetic keeps the existing mixed-radix comparison and rounding
formulas. Source consumer tracking reuses the destination completion event.

The retained-range workspace query reports zero per-wave device or pinned
metadata. The ordinary whole-matrix query retains its staged-metadata contract.
No host completion wait, mutex, extra pool, or CPU fallback was introduced.
Calibration keys distinguish the conversion mode, plaintext modulus, source
basis/level and destination type; wrong queued operations are rejected before
consuming a prepared invocation.

Evidence: `test_data/fleet-sharding-implementation/prepared-modulus-ranges/`.
Both workspace unit builds are warning-free. Evidence covers 990 GPU-enabled
passes (27 ignored) and 608 CPU passes (25 ignored). Seven changed GPU executables
ran once; four unchanged GPU executables and all eleven CPU executables reused
passing results only after exact SHA-256 comparison. Runtime passed 232 tests and
primitives passed 295. All 71 source hashes and 22 executable hashes were verified
after the run, with crate sources frozen throughout validation.

The two new reuse/lifetime fixtures passed three identical-command executions
and one ring-dimension-512 execution each. Coverage includes mixed-width/reordered
CRT bases, all four conversions, both formats, offset rectangles, untouched
borders, prior readers, immediate source/output release, empty rectangles,
shared normalization, two cached execution cycles and different plaintext moduli.
Existing CPU primitives supply references. A separate existing modulus-switch
fixture passed once with `MXX_PRIMITIVE_TEST_CRT_DEPTH=64`, covering boundary
rounding and non-prefix reductions through the common conversion arithmetic.
This is single-GPU evidence, not physical multi-GPU acceptance.

The initial retained-workspace design correctly rejected immediate pinned-buffer
reuse while prior DMA still owned that buffer. All three failing runtime runs
and the dimension-512 failure, together with passing primitive results, are
preserved under `pinned-reuse-`. Moving the immutable CRT table to setup and using
by-value launch metadata removed that per-wave resource entirely. The subsequent
constant-owner rename diagnostics are preserved under `constant-rename-`; all
native users now reference the same ring-constant owner. No existing numerical
assertion or admission/lifetime check was weakened.

The unchanged modulus-switch fixture took 0.2545/0.2583 seconds for the saved
baseline/candidate with the same command. These whole-process timings include
setup and do not establish a kernel speedup or final production performance
acceptance. Exact commands, environments and source/executable identities remain
in the evidence directory. The managed allocation audit documents the added
setup table and its release contract.

Automatic graph-wide provisioning, RNS up/down and CRT recomposition, compact
hash/trapdoor/preimage, import/export and scalar boundaries, zero-row gadget
handling, estimator plan consumption and final independent review remain required.
Physical multi-GPU validation remains deferred at the user's request.

### Prepared centered rebase and coefficient input normalization

Centered rebase now writes directly into a retained rectangle of a separately
registered output CRT basis. Native address calculation preserves input/output
row offsets and pitches. Coefficient inputs stay immutable; evaluation inputs
are normalized once into explicitly reserved fragments shared by admitted calls.
The existing full-matrix API uses the same range kernel. Evaluation outputs run
NTT only on the written rectangle, preserving the remaining entries and format.
The output completion event protects borrowed input lifetime without another
per-call event or metadata allocation. Error paths retire submitted output work.

Prepared normalization keys now explicitly distinguish coefficient and evaluation
fragments instead of overloading an original shard key. This prevents one
operation's normalized input from changing the format selected for another
operation. Admission and compilation derive output parameters independently of
input parameters. Calibration identity includes the actual input basis/level as
well as the destination type; different source bases cannot reuse a profile just
because their destination and shapes match. Equivalent registered rings in a
related native context remain supported when execution ownership matches.

Evidence: `test_data/fleet-sharding-implementation/prepared-centered-rebase/`.
Both workspace unit builds are warning-free. GPU-enabled evidence covers 988
passes and 27 ignored; CPU evidence covers 608 passes and 25 ignored. Seven
changed GPU executables ran once; four unchanged GPU executables and eleven CPU
executables reused evidence only after exact executable SHA-256 comparison.
Runtime passed 231 tests and primitives passed 294. All 70 source hashes and
22 executable hashes were checked after the run; no source changed during tests.

The two new lifetime/reuse fixtures passed three identical-command executions
and one ring-dimension-512 execution each. Primitive evidence was reused in the
last runtime-only revision after exact binary identity verification. Coverage
includes both narrow/wide target-limb orders, both input/output formats, offset
rectangles, preserved borders and prior readers, immediate source/output release,
empty columns, shared normalization, repeated cached execution, separate source
bases and related parameter contexts. CPU primitives provide numerical references.
Invalid shape/order calls leave the admitted queue available for the correct call.

Initial compiler diagnostics are preserved under `initial-`. New fixture failures
are preserved under `fixture-` and `readback-clone-`: the narrow CPU source declared
the wide bit count; short-column readback declared the full-width transfer span;
and coefficient-format readback omitted its existing GPU evaluation-copy matrix.
The fixtures now declare the actual parameters and all required readback resources.
No numerical assertion, production bound or existing test was weakened.

The unchanged prepared-hash fixture took 0.2545/0.2740 seconds for baseline/candidate
whole processes. This is an orchestration smoke comparison, not a centered-rebase
kernel measurement or final performance acceptance. Exact commands and hashes are
recorded with the results.

Automatic graph-wide provisioning, the other CRT/RNS conversions, compact
hash/trapdoor/preimage, import/export and scalar boundaries, zero-row gadget
handling, estimator plan consumption and final independent review remain required.
Physical multi-GPU validation remains deferred at the user's request. The next
CRT work must account for metadata and DMA resources against the actual logical
output owner. Prepared matrix claims already resize visible auxiliary capacity;
see `test_gpu_prepared_smaller_output_uses_its_queried_auxiliary_layout`. A wave
rectangle is not the full retained output owner used by that workspace query.

### Prepared polynomial constants and fixed transfer resources

PowerOfBase, Rotation and Polynomial singleton constants now use the prepared
fresh-output runner. Signed coefficient residues are computed with Rayon, copied
through the existing pinned RNS loader into the retained output, and transformed
on the GPU when evaluation format is required. The primitive preserves a
coefficient-format destination as well. No temporary polynomial/matrix allocation,
CPU NTT or extra synchronization is introduced. PowerOfBase uses modular
exponentiation to obtain the same ring value without constructing an unbounded
intermediate integer. Shape, exponent and coefficient-count checks precede dispatch.

Admission selects exact typed fixed workspaces alongside row-reduction scratch.
For these atomic constants the required order is pinned host staging, device
transfer staging, then one completion event. Complete-batch selection excludes
previously chosen slots, calibration uses the same claims and production preserves
them during job rearming. Compilation checks context, kind, size, alignment, count
and order. Fixed native resource claims are distinct from width-dependent row-tree
matrix claims. These constants execute as complete singleton operations.

Evidence is in
`test_data/fleet-sharding-implementation/prepared-polynomial-constants/`. Two new
fixtures passed three identical-command runs each and one dimension-512 run each.
Coverage includes mixed packed limb widths/order, signed and large coefficients,
empty/sparse/full polynomials, both formats, pending readers, early output release,
PowerOfBase, Rotation, reordered-call rejection and two cached cycles with strict
transfer/resource claims. References use existing CPU polynomial primitives.
Existing tests are unchanged.

The initial compile diagnostics for an unavailable primitive-crate import and a
new test's shift type are retained. The first runtime candidate's three failures
and dimension-512 failure are preserved under `claim-order-`: the planned device
staging preceded pinned staging, while the loader allocates pinned staging first.
The order was corrected, and compilation now verifies ordered workspace claims.
No bound, ownership check or numerical assertion was relaxed.

Both workspace unit builds are warning-free. Final evidence covers 986
GPU-enabled passes with 27 ignored and 608 CPU passes with 25 ignored. Seven
changed GPU executables ran once; four unchanged GPU executables and all eleven
CPU executables reused passing evidence only after exact SHA-256 comparison.
Runtime passed 230 tests and primitives passed 293. All 70 source-manifest entries
and all 22 executable hashes matched after the run.

The unchanged prepared-hash fixture passed on the saved baseline and candidate
with the same command, taking 0.2615 and 0.2680 whole-process seconds. This includes
startup/calibration and does not establish a kernel timing change. Exact commands,
source identities and results are retained in the stage evidence.

Automatic graph-wide provisioning, zero-row gadget edge handling, compact
hash/trapdoor/preimage, remaining CRT/import/export families, estimator plan
consumption and final independent review remain required. Physical multi-GPU
validation remains deferred at the user's request.

### Prepared plain hash sampling

Plain hash sampling now uses the retained sampler and fresh-output admission
path. The existing hash-to-ChaCha seed derivation is exposed for reuse without
changing its domain, digest inputs or generated stream. Production derives one
seed from its actual key and tag, then shares it across all scheduled ranges.
No tag or key bytes enter prepared invocation metadata or calibration identity.
The concrete output type and tag length remain part of that identity.

Pilots derive an independent hash seed using a private key and a tag of the
requested length. Production requires an explicit seed payload and rejects a
missing payload before consuming the queued invocation; it cannot accidentally
use pilot randomness. Ordinary operations reject an unexpected hash payload.
The output ownership, managed reservations, CUDA kernels and reader/release
ordering are the existing validated paths. Compact/decomposed hash remains an
explicitly unsupported prepared class pending its compact-output implementation.

Evidence is in `test_data/fleet-sharding-implementation/prepared-hash/`. The new
runtime fixture compares results against the existing full-matrix GPU hash
sampler. It supplies distinct random keys and tags, including different payloads
with equal tag lengths, across two cached cycles. Empty tags and empty-column
outputs are included. Wrong tag lengths and missing production seed payloads are
rejected without consuming the queued operation. Readback uses separately
claimed native resources. No existing tests were changed.

Both workspace unit builds are warning-free. Final evidence covers 984
GPU-enabled passes (27 ignored) and 608 CPU passes (25 ignored). Seven changed
GPU executables ran once; four unchanged GPU executables and all eleven CPU
executables reused passing evidence after exact SHA-256 comparison. The new
fixture ran once as part of runtime's 229 passing tests. Sources remained frozen;
all 70 source-manifest entries and 22 executable hashes were verified afterward.

The unchanged prepared-random-sampling fixture passed on the saved baseline and
candidate with the same command, taking 0.2730 and 0.2852 whole-process seconds.
This includes startup and calibration; the single pair does not establish a
kernel timing change. The native execution kernels and ownership ordering did
not change in this stage. No build or unit failure occurred.

Automatic graph-wide provisioning, compact hash/trapdoor/preimage, atomic
constants, remaining CRT/import/export families, estimator plan consumption and
final independent review remain required. Physical multi-GPU validation remains
deferred at the user's request.

### Prepared random sampling and retained rectangles

The existing sampler now writes logical column ranges directly into retained
output rectangles. One descriptor kernel handles all active CRT limbs, including
packed coefficient widths from one through eight bytes. The ChaCha domains,
logical polynomial and coefficient coordinates, four samples per thread, uniform
rejection and bounded Karney Gaussian sampling are preserved. The output's
physical offsets and pitch are independent of logical random coordinates.

The existing optimized NTT kernels now map an optional rectangle. Only newly
sampled coefficients are transformed; pre-existing evaluation entries outside the
rectangle remain intact. Whole-matrix and batch transforms retain their original
index mapping. Sampling needs no intermediate matrix or device metadata workspace,
and uses the retained owner's writer, reader and release ordering.

Uniform ring, bit, ternary and Gaussian backend calls now join the prepared
fresh-output capacity trials, calibration registry, reservations and enqueue
workers used by constants. One random seed is generated after matching each
production invocation and shared across all ranges. Pilots use independent seeds;
random values are excluded from profile identities. The exact output type,
distribution, sigma bits and coefficient bound enter invocation and profile
identities. Zero-sigma Gaussian uses the retained zero writer. Unsupported uniform
ranges, negative bounds and invalid sigma are rejected before production dispatch.

Evidence is in `test_data/fleet-sharding-implementation/prepared-sampling-ranges/`.
Three new fixtures each passed three identical-command executions. They cover
mixed narrow/wide limb order, both formats, nonzero destination offsets, logical
column splitting, preserved borders, old readers, early output release, strict
output-only dispatch, cached reuse, sampling bounds, empty columns and invocation
order. The rectangle fixture also passed once each at ring dimensions 2,048 and
32,768, covering both fused-top and global NTT stages. Existing tests are unchanged.

Initial build diagnostics for imports and fleet parameter lookup are preserved.
The first runtime candidate rejected valid packed widths by assuming four/eight
bytes; `packed-width-` records retain all three failures for each new fixture,
the size checks and reference comparison. The native validation now accepts the
existing one-to-eight-byte storage format. No numerical assertion, admission
bound or synchronization requirement was relaxed to obtain the passing results.

Final CPU and GPU workspace unit builds are warning-free. Evidence covers 983
GPU-enabled passes with 27 ignored and 608 CPU passes with 25 ignored. Seven
changed GPU executables ran once; four GPU executables and all eleven CPU
executables reused prior passing evidence only after exact SHA-256 comparison.
Runtime passed 228 tests and primitives passed 292. All 69 source-manifest entries
and all 22 executable hashes matched after testing.

The first unchanged hash-range comparison took 0.2215 seconds before and 0.2921
after, including process/context startup. Five further alternating comparisons
had medians of 0.2354 before and 0.2265 after (ranges 0.2160-0.2808 and
0.2175-0.2323 seconds). This did not reproduce the initial slowdown; process-level
variability prevents a kernel speedup claim. Exact commands, hashes and timings
are retained in `target-summary.json` and `timing-summary.json`.

Graph-wide automatic provisioning, hash/trapdoor and compact sampling, atomic
constants, remaining CRT/preimage/import/export families, estimator plan
consumption and final independent review remain required. Physical multi-GPU
validation is deferred at the user's request.

### Prepared fresh constants

Evidence is in `test_data/fleet-sharding-implementation/prepared-fresh-constants/`.
Zero, Identity, UnitRow, UnitColumn and ordinary/small Gadget constants now use the
same prepared preflight, calibration registry, column reservations and enqueue
workers as input-owned operations. The compiler represents absent primary inputs
explicitly. Every selected output carries its exact parameters, active level and
format independently of whether the operation has an input. No placeholder matrix
or extra GPU input allocation is introduced.

Fresh output selection uses the existing per-device capacity trials and capped
waterfill assignment. Inherited input ownership remains unchanged for existing
operations. The complete batch is selected before calibration or publication;
empty-column constants require no output intervals. Resolved constant values,
indices, digit counts, concrete output type and layout enter canonical profile and
invocation identities. Pilots and production call the same retained range writer,
using global column positions. Unsupported atomic polynomial constants remain
explicit errors under prepared admission.

The new runtime fixture mixes all five supported constant kinds, both gadget forms,
an alternate gadget digit count, an empty-column constant and an existing negate
operation in one batch. It rejects reordered execution and checks two complete
cached cycles against trusted CPU primitives using strict prepared output/readback
claims. It passed three identical-command runs and one dimension-512 run. The
unchanged prepared-accumulate fixture passed on the saved baseline and candidate,
with the same command and environment, in 0.2675 and 0.2583 whole-process seconds.
The pair checks the common execution path but does not establish a kernel speedup.
The first narrow build caught one optional-input borrowing mismatch; its source
manifest, patch and diagnostic are preserved. Both final workspace unit builds are
warning-free. No existing test was changed.

Final evidence covers 980 GPU-enabled unit passes (27 ignored) and 608 CPU unit
passes (25 ignored). Six changed GPU executables ran once; five unchanged GPU
executables and all eleven CPU executables reused passing evidence after exact
SHA-256 comparison. Runtime passed 227 tests. All 68 source manifest entries and
all 22 executable hashes were checked after the run.

Automatic graph-wide provisioning, atomic constant forms, remaining sampling,
compact/preimage/CRT/import/export invocation families, estimator plan consumption
and final independent review remain required. Physical multi-GPU validation stays
deferred at the user's request.

### Retained constant rectangles

The native Identity, UnitRow and Gadget column generators now share one constant
range kernel. The same primitive API also fills Zero and UnitColumn rectangles.
It resolves all active limb descriptors on the GPU and launches once across limbs;
it does not allocate a matrix, scalar polynomial or device metadata workspace.
Logical global columns are independent of the output owner's physical row/column
offsets. Writes join previous readers and writers and publish the retained output
completion. Zero preserves coefficient/evaluation format; nonzero constants require
evaluation format. The logical gadget layout uses the full parameter basis even
when a retained output projects it onto fewer active limbs.

Existing public column constructors call the same implementation. Gadget bounds
and digit counts are validated before their output allocation; the mutable range
API validates both the logical and destination ranges before submission. Native
kernel preparation now lists the consolidated constant kernel and the previously
added accumulate range kernel explicitly. The first build exposed stale entries
for the removed constant kernels; initial diagnostics and source identities are
retained under `test_data/fleet-sharding-implementation/prepared-constant-ranges/`.

Two new primitive fixtures passed three identical-command runs each and one
ring-dimension-512 run each. Coverage includes mixed narrow/wide CRT limb order,
Identity, UnitRow, UnitColumn, Zero, small and ordinary gadgets, alternate digit
counts, global block intersections, empty ranges, independent destination offsets,
preserved borders, old readers and early release. Zero-in-coefficient-format is
also checked. A complete native setup followed by two strict dispatch cycles
permits only the retained matrix output and separately claimed RNS readback
resources; constant generation itself needs no extra workspace or event slots.
Existing tests are unchanged. The unchanged full-layout range fixture passed on
the saved baseline and candidate with the same command/environment, taking 0.2134
and 0.1410 whole-process seconds. A single pair does not establish a kernel speedup.
The initial compile records include stale kernel-preparation names and an
Option-to-Result conversion in digit-count validation; final builds are warning-free.

Final evidence covers 979 GPU-enabled unit passes (27 ignored) and 608 CPU unit
passes (25 ignored). Seven changed GPU executables ran once; four unchanged GPU
executables and all eleven CPU executables reuse passing evidence after exact
SHA-256 checks. Runtime passed 226 tests and primitives passed 290 tests. All 68
source manifest entries and all 22 executable hashes match the final build.

This stage provided the reusable destination-writing primitive. The subsequent
fresh-constant stage above connects these generators to prepared runtime admission.
Atomic polynomial constants, other invocation families and graph-wide setup remain
required. Physical multi-GPU checks remain deferred.

### Prepared multiply-accumulate

Evidence is in `test_data/fleet-sharding-implementation/prepared-accumulate/`.
Multiply-accumulate now binds all original product operands and an optional bias
to prepared column intervals. Fixed factors are shared through the existing
preparation registry; all scalable factors and the bias contribute their column
boundaries. Coefficients, scalar positions, output geometry and bias presence enter
the profile identity and invocation matching. Pilots and production use the same
range runner on the existing device enqueue workers. CPU coefficient reduction
uses Rayon.

The native range path extends the existing batch entry point and writes directly
into one retained destination rectangle. It supports multiple rows, mixed narrow
and wide CRT limbs, scalar multiplication on either side, arbitrary signed integer
coefficients and zero-length dot products. Batches of four terms fit within the
portable 4096-byte CUDA argument limit. Larger sums continue on the same output
stream without allocating product matrices, coefficient matrices or device metadata
workspaces. Output ownership events protect input readers and native release.
The dynamic batch-workspace query still describes its existing pointer-array paths;
the new range runner uses fixed launch arguments and does not invoke that query.

A separate primitive regression covers seven terms, both scalar positions,
non-unit/negative/zero coefficients, both CRT limb orders, empty dot products,
independent row/column offsets, preserved output borders, an earlier output reader
and early source release. A runtime regression covers shared fixed operands, mixed
coefficient/evaluation inputs, distinct column boundaries, optional bias, rejected
reordered invocations and two cached execution cycles with strict prepared claims.
Both fixtures passed three identical-command executions and one dimension-512
execution. Runtime's three passes were reused after the final test-only edit only
because its executable SHA-256 was unchanged.

Initial compiler diagnostics are retained under `initial-` and `imports-`.
The first primitive test attempt failed three times in the CPU reference's
zero-inner-dimension multiplication, before the GPU comparison. Its records are
retained under `empty-reference-`. That reference now uses the existing trusted
zero-matrix constructor for the empty sum; the empty GPU case remains tested.
No existing test was changed. The unchanged ordinary owned-accumulate fixture
passed with the saved baseline and candidate, taking 0.2652 and 0.2600 whole-process
seconds under the same command and environment. This single pair does not establish
a speedup for the prepared kernel.

Final workspace evidence covers 977 GPU-enabled passes (27 ignored) and 608 CPU
passes (25 ignored). Both workspace unit builds are warning-free. Seven changed
GPU executables ran once; four unchanged GPU executables and all eleven CPU
executables reused passing evidence after exact SHA-256 comparison. All 68 source
manifest entries and all 22 executable hashes were checked after the run. Runtime
passed 226 tests and primitives passed 288 tests in their complete unit runs.

Automatic graph provisioning, remaining invocation families, estimator plan
consumption and final independent review remain required. Physical multi-GPU checks
remain deferred at the user's request.

### Prepared column and diagonal concatenation

Evidence is in `test_data/fleet-sharding-implementation/prepared-column-concat/`.
All three concatenation axes now have prepared matrix runners. Column and diagonal
concatenation bind each output interval to the corresponding original operand and
its exact native context. Checked prefix maps retain source column boundaries and
skip empty column intervals while preserving diagonal row offsets. Different source
rings or levels are rejected. Only actual range inputs enter preparation; unrelated
operands are not replicated. Diagonal destinations are zeroed once through the
existing asynchronous native initializer before their source rectangles are copied.
Pilots and production share both the initializer and range runner. Prefix maps and
result geometry enter the canonical profile identity.

The separate regression includes empty leading/interior column inputs, mixed
coefficient/evaluation inputs, different fragment widths, related native contexts,
original-owner assertions, CPU concatenation references and two cached batches.
It passed three identical-command runs and one ring-dimension-512 run. Runtime
passed 225 tests once. The unchanged diagnostic concat/CRT fixture passed on the
saved baseline and candidate with the same command/environment, taking 0.2807 and
0.2672 whole-process seconds. This pair does not measure a prepared-path speedup.

Initial readback checks exposed a resource-bound distinction: a smaller matrix
request inherits its prepared backing's producer streams. Reserving the maximum
shape's event count left unused slots on some owners; querying only the smaller
logical shape under-reserved other owners. Both initial three-run failures and
source manifests are preserved under `initial-` and `shape-query-`. The primitive
`rns_store_completion_events` query now reads the actual immutable producer stream
layout and counts cross-stream consumer events plus returned transfer completions.
It performs no device selection, GPU allocation or completion wait. The regression
uses this exact count while retaining strict dispatch-consumption checks and all
mathematical assertions. RNS workspace and any format-conversion owner remain
separate claims. This query is groundwork for the remaining import/export admission;
those invocation families and automatic setup/estimator integration remain required.

Final workspace evidence for column/diagonal concatenation covers 975 GPU-enabled
passes and 608 CPU passes, using the exact-binary reuse at the top. Both workspace
unit builds are warning-free. All 68 source-manifest entries and all 22 executable
hashes were rechecked after completion. This completes the local check for these
concatenation paths; full-plan completion still requires the remaining implementation
and final independent review. Physical multi-GPU checking remains deferred.

### Prepared row concatenation and fused row-block addition

Evidence is in `test_data/fleet-sharding-implementation/prepared-row-blocks/`.
The prepared compiler now binds an ordered collection of matrix operands, with
individual input identities, native contexts and column ranges. Row concatenation
and fused row-block addition use this same preflight, shared input preparation,
profile registry and retained-output admission path. They intersect every input's
column boundaries and preserve inherited ownership. Changing the order or identity
of any operand after admission is rejected. Row concatenation aligns input formats
and copies directly to the retained destination, without a stacked temporary.

The native row-block addition kernel accepts independent source/RHS/destination
rectangles and owner strides. Up to 16 blocks still share one all-limb launch;
larger block lists use bounded batches writing successive output row ranges.
No complete concatenated input or per-block arithmetic intermediate is allocated.
The retained output event protects source reader/release dependencies. Existing
whole-matrix callers use the same kernel with complete ranges. CPU input metadata
preparation uses Rayon where independent; slot selection and writes through a
shared destination retain their required ownership order.

The new primitive fixture covers 18 blocks including empty row views, unequal
source pitches, displaced source and output rectangles, both mixed limb orders,
preexisting output readers, early source release and untouched output borders.
The new runtime fixture covers 18 operands, mixed input formats, differing column
boundaries, ordered-operand rejection, exact CPU references and two cached batches.
Both fixtures passed three identical-binary runs. The primitive fixture also passed
once at ring dimension 512. The existing whole-matrix row-add fixture passed on
the saved baseline and candidate binaries with the same command and environment:
whole-process times were 0.4455 and 0.4249 seconds. One pair does not establish a
kernel speedup. Initial compile diagnostics are preserved separately. Column and
diagonal concatenation, other invocation families, automatic setup and estimator
plan consumption remain required; this stage is not full-plan acceptance.

Final workspace evidence for the row-block stage covers 974 GPU-enabled passes
and 608 CPU passes, with the exact-binary reuse stated at the top. Runtime passed
224 tests and primitives passed 287 tests. All 68 source-manifest entries and all
22 executable hashes were rechecked after completion. Both workspace builds were
warning-free. Physical multi-GPU validation remains deferred by the user.

### Mixed-format prepared input replicas

Evidence is in `test_data/fleet-sharding-implementation/prepared-mixed-inputs/`.
Prepared replicas now accept fragmented inputs containing both coefficient and
evaluation columns. Preflight reserves each mismatching fragment on the destination
parameter context, sharing normalization owners by value, fragment, context and
requested format. It reserves the complete batch before submission. Normalization
commands precede replica commands on the existing workers; native producer/reader
and release edges retain asynchronous dependencies without a GPU completion barrier.
Each fragment crosses to its destination context once for a requested format.
Homogeneous inputs keep the existing whole-replica transform path. Original input
owners remain unchanged; normalized temporary owners are released after compilation
retains the complete replicas actually used by each invocation.

A separate unit fixture covers alternating formats across related native contexts,
both NTT directions, transpose, tensor, fused tensor row sums, and two cached batch
cycles. Expected values use CPU primitives. It passed three identical-binary runs
because the change affects temporary-owner reuse. The full runtime suite passed
223 tests once. Complete workspace evidence covers 972 GPU-enabled and 608 CPU
passes, with the exact-binary reuse described above; all source and executable
hashes were verified afterward. Initial compile diagnostics are preserved separately.
Runtime's whole-process time was 1.5868 seconds versus the preceding shared-registry
run's 1.5634 seconds; the new fixture changes the workload, so this is not a matched
performance comparison or a speedup claim. Default provisioning, remaining invocation
families and estimator plan consumption still require implementation. Physical
multi-GPU checking remains deferred; no final independent acceptance is claimed.

### Shared prepared calibration registry

Evidence is in `test_data/fleet-sharding-implementation/prepared-shared-registry/`.
Prepared matrix calibration now reads and publishes the existing frozen registry,
using exact hardware/policy, allocation-class, prepared-configuration and logical
occupancy metric keys. A setup registry can be created from a frozen snapshot
without changing existing readers or copying executable/native owners. Prepared
profiles no longer use the backend's separate diagnostic operation cache. Every
hit still goes through the ledger's current output/scratch admission.

Fleet construction now rejects duplicate device IDs and mismatched GPU models,
compute capabilities or physical VRAM before creating enqueue workers. The
shared-registry regression installs a saved snapshot and holds an unrelated
native reservation across the second preflight: a measurement reset would fail,
so success verifies an actual profile hit and fresh native capacity checking.

Both workspace unit builds are warning-free, and all 222 runtime unit tests
passed once (four ignored) on the final source manifest. The last complete
workspace evidence remains the preceding idle-representative stage's 971 GPU
and 608 CPU passes; downstream suites were not repeated for this scoped cache
change. This provides the shared profile path, not complete estimator plan
consumption or automatic production provisioning. Multi-GPU validation remains
on hold at the user's request.

### Idle representative planning and test ownership

Evidence is in `test_data/fleet-sharding-implementation/prepared-idle-representative/`.
When inherited production ownership skips configured device 1 but uses later
devices, preflight now separately selects its actual input preparation, one-column
pilot destination and reduction scratch. These enter the profile identity and
measurement only; production ownership is preserved. The source class comes from
an active nonzero device. Physical multi-GPU execution of this branch remains
unverified and was explicitly deferred by the user on 2026-09-11.

The initial runtime suite exposed an unclassified GPU fixture:
`transcript_replay_preserves_preimage_small_owner_and_relation` uses the CPU
backend, but its trapdoor representation creates GPU contexts when `gpu` is
enabled. It ran outside the existing GPU test isolation boundary and overlapped
an exclusive native setup test, which correctly returned `NonexclusiveOwner`.
The fixture now has the same conditional GPU test annotation as the existing
trapdoor-family test. Production code, ownership rejection and arithmetic
assertions were not weakened. This explains the observed setup failure; older
context-generation failures lack enough timing evidence for exact attribution.

After the fixture correction, runtime passed 222 tests three consecutive times.
The warning-free workspace builds and final unit evidence cover 971 GPU-enabled
passes (27 ignored) and 608 CPU passes (25 ignored). Six changed GPU executables
ran once, with the runtime repetitions noted above; the other GPU binaries reused
exact-SHA evidence. CPU runtime ran once; the other ten CPU binaries were unchanged.
All source and all 22 binary hashes were checked afterward. Initial failure logs
and source identities are preserved. Multi-GPU checking is on hold at the user's
request; implementation, local validation and final independent review continue.

### Prepared row-reduction trees

Evidence is in `test_data/fleet-sharding-implementation/prepared-reduction-trees/`.
The prepared row-sum and fused tensor row-sum runners now handle arbitrary valid
row groups. Small groups are batched within the existing 16-group/32-term native
class. Larger groups use bounded leaves and parallel binary addition levels;
no complete tensor product is materialized. The shared primitive query reports
the exact one-row intermediate count. Runtime selects that finite scratch
inventory before output admission and checks its native fit at each candidate
column width. The existing dispatcher specializes and rearms those claims for
actual jobs and tails. Intermediates retain native source/read/release joins.

Matrix allocation claims are consumed on the dispatch owner before allocation-free
parent kernels run through Rayon. Coefficient-domain ordinary row sums retain
their format; tensor reductions use the explicitly prepared evaluation inputs.
One device's reduction jobs currently require the same exact scratch parameter
context, level and format; incompatible inherited jobs fail before preparation.

Both workspace unit builds are warning-free. GPU-enabled tests passed 971 with
27 ignored; CPU evidence covers 608 passes with 25 ignored. Changed GPU executables
ran once, and unchanged executables reused evidence only after exact SHA-256
comparison. All source and executable hashes were checked after completion.
The new real-residency tree fixture passed three identical-command runs because
it exercises asynchronous intermediate reuse. It covers 18 groups, a 65-term
reduction, coefficient/evaluation domains, tensor scratch narrower than the full
output, retained output admission and two cached batch cycles.

The initial fixture failed three times because coefficient readback creates an
evaluation copy that lacked its own reserved matrix slot. The fixture now reserves
that real copy separately from compute scratch. Those failures, the initial
source hashes and a diagnostic backtrace remain under `initial-`. Existing
mathematical assertions and tests are unchanged. These results close this local
row-tree implementation check; they do not complete default provisioning,
remaining invocation families, estimator integration, physical multi-GPU checks
or final independent review. No performance speedup is claimed for this stage.

### Prepared input redistribution and fresh ownership

Evidence is in `test_data/fleet-sharding-implementation/prepared-redistribution/`.
Native peer copying accepts rectangular source/destination ranges and preserves
existing producer, consumer and release dependencies. Same-device related
contexts use a two-dimensional asynchronous copy; different devices use a
three-dimensional peer copy. There is no host reconstruction path.

Prepared preflight selects a shared complete fixed operand per value, device,
exact native parameter context and format. Transpose and tensor operations
calculate owner capacities after hypothetical input preparation, then assign
fresh contiguous output intervals by capped water filling. Aggregate capacity
must cover the entire logical output before any preparation is submitted.
Independent device capacity calculations use Rayon. Inherited operations retain
their source ownership. Compiled execution binds each input to the exact retained
output context and keeps only the preparations used by that invocation.

The latest warning-free CPU/GPU workspace builds cover 970 GPU-enabled passes
(27 ignored) and 608 CPU passes (25 ignored). Changed GPU executables ran once;
unchanged executables reused passing evidence only after exact SHA-256 checks.
All source hashes and all 22 executable hashes were verified afterward. The
peer-rectangle and shared fixed-replica lifetime tests additionally passed three
identical runs on the earlier fixed-replica revision. The native primitive binary
is identical to that tested revision. The new fragmented-input transpose/tensor
case passed in the final runtime suite, including two cached executions.

The local hardware remains one RTX 4080 SUPER: cross-device peer copying and
multi-device capacity assignment have not been executed here. Initial compiler
failures and the earlier fixed-only revision are preserved. Automatic setup,
remaining invocation families, estimator admission and final acceptance remain
required.

### Prepared row transforms and exact tensor intervals

Evidence is under `test_data/fleet-sharding-implementation/prepared-row-transforms/`.
Transpose, bounded row sums, slices, tensor products and bounded fused tensor row
sums now have direct retained-destination runners. Existing kernels consume owner
strides and rectangular offsets. Tensor ranges use exact global product columns,
including intervals crossing factor boundaries. No smaller substitute factors,
extra GPU staging outputs, new kernels or worker pools are introduced.

Runtime preflight reserves every complete output and shared coefficient-input
normalization before pilots. The internal compiler borrows the original typed
requests, freezes their semantic arguments and rejects a reordered production
call. Cached execution preserves native slot reuse without a host release fence.
The range regression checks untouched output borders, mixed-width CRT limbs and
input destruction before readback against existing CPU primitives. Tensor tests
cover both packed-polynomial and separate-polynomial kernel classes (ring
dimensions 32 and 512). A real-residency runtime test covers all five operations,
coefficient RHS normalization and cached reuse.

The initial runtime test exposed a missing lazy consumer-event claim in the
transpose/row-sum view paths. They now record the retained output completion and
use it for source reuse/release joins, following the existing matrix batch view
contract. Both affected lifetime regressions then passed three consecutive runs.
Subsequent slice/tensor cases and the final changed workspace executables ran once,
following the user's latest repetition instruction. Initial failures, intermediate
row-only evidence, source manifests, commands and exact executable hashes are
preserved. No numeric assertion or native admission guard was weakened.

The latest full evidence covers 967 GPU-enabled passes and 608 CPU passes, with
27 and 25 ignored respectively. Single before/after whole-process reference times
for transpose were 0.5010/0.5067 seconds and for row sums 0.3702/0.3518 seconds.
These noisy process times do not establish kernel speedups. Native row-group
classes remain bounded to 16 groups and 32 terms; larger admitted reduction trees
remain required. Transpose/tensor prepared dispatch currently requires complete
resident operands on its selected owner. Automatic setup, planned redistribution,
remaining invocation families, estimator integration and multi-GPU acceptance
remain unfinished.

### Managed setup from actual native residency

Evidence is under `test_data/fleet-sharding-implementation/managed-setup/`.
The complete storage inventory closes ordinary, workspace, pinned, transfer and
typed CUDA resource allocation paths together. It rejects missing/duplicate
storage, active reservations and uncertain native activity. The runtime obtains
actual initial residency receipts for the configured fleet and validates
both initial budget dimensions before installing its dispatcher ledger. The
[audit](gpu-managed-allocation-audit.md) records each direct managed allocation
family and the private-stream lifetime coverage.

The primitive regression passed three consecutive runs; the real-residency
runtime regression also passed three consecutive runs. The existing synthetic
physical-budget-excess fixture passed once. Both workspace unit builds are
warning-free. Initial fixture errors and an FFI declaration warning are preserved
under `initial-` and `fixture-warning-`, respectively. The corrected fixture uses
the existing typed native compact API and the original context decomposition
parameter. No numeric assertion or native admission check was relaxed.

This evidence covers explicit prepared setup and admitted negate/add/subtract
calls. It does not claim default graph-wide provisioning, completion of the
remaining operations, or physical multi-GPU acceptance. Earlier full-suite counts
above identify the preceding checkpoint, not a full run of this newer executable.

### Prepared matrix and scalar multiplication

Evidence is under `test_data/fleet-sharding-implementation/prepared-multiply/`.
The existing tiled and thin-row batch matrix kernels now accept rectangular
input/output geometry. Owner strides and starting offsets are computed outside
their coefficient/contraction loops. The batch workspace query includes the
geometry, and the existing output auxiliary arena holds it when it fits. Existing
whole-matrix callers pass no geometry. No kernel, synchronization primitive,
stream, or worker pool is added.

The primitive column-view API multiplies evaluation ranges directly into a
retained destination. Complete scalar matrix owners use the existing scalar
batch kernel without extracting a separate GPU polynomial. Native view validation
checks contraction dimensions, owner shapes, destination overlap, context,
level and format; existing reader retirement covers both operands.

Prepared preflight binds actual operand order, normalizes coefficient inputs
once, and assigns complete output slots with the product's output row count.
The scalable operand supplies column ownership; the other operand must be fully
resident on that owner. Both scalar positions and ordinary matrix multiplication
share the same range runner between isolated pilots and production. The native
workspace query must prove that the retained output's auxiliary arena suffices
before a pilot; a product requiring separate metadata scratch remains unsupported.
Fixed-operand replication/redistribution, empty contractions, default certified
setup, remaining invocation families and estimator integration are not completed
by this change.

Initial focused tests exposed two fixture setup mistakes: the all-narrow CRT
basis declared a wide prime width, and the matrix-result download claimed the
larger scalar-result RNS workspace rather than its exact demand. Both are repaired
without changing mathematical assertions or production allocation enforcement.
Initial source/build manifests, failure logs and results are retained under the
`initial-` prefix. The existing unsupported-invocation regression now uses
transpose because multiplication is intentionally supported.

The revised primitive view regression passed three consecutive runs of the same
binary, covering thin/tiled kernels, mixed limb widths, displaced source and
destination rectangles, untouched output entries and early input destruction.
It also passed once at ring dimension 2048 with 19 columns. The prepared-runtime
regression passed with coefficient/evaluation operands, full-batch capacity
rejection, cached slot reuse, actual argument-order enforcement, both scalar
positions and existing CPU references. Existing prepared arithmetic regressions
also passed. The unchanged thin-row batch reference fixture took 0.5205 seconds
with the preceding binary and 0.4943 seconds with the final candidate; these
are whole-process times, not a measured speedup for prepared range execution.
Both CPU and GPU workspace unit builds are warning-free.

The first full workspace run had 960 passing tests, one failure and 27 ignored:
an existing metadata-query assertion still expected multiplication views to be
unsupported. Multiplication now participates in the existing checks for complete
geometry bytes at batch counts 1 and 1024; the unsupported-view check retains the
still unsupported fused-accumulate class. This changes a superseded capability
expectation, not any numerical assertion. The `before-query-fix-` artifacts retain
that complete run, its exact source and executable identities, and the successful
runtime regression evidence.

The final evidence covers 961 passing GPU-enabled tests (27 ignored), including
282 primitive and 216 runtime tests, and 608 passing CPU tests (25 ignored).
After the metadata-test update, only the primitive executable changed: its full
unit suite and focused reader-lifetime regression were rerun, while the other
ten GPU binaries and all eleven CPU binaries retained exact SHA-256 equality
with their successful evidence. All 68 source-manifest entries and all 22 final
binary hashes were verified. These results do not establish physical-resource
certification, remaining invocation coverage, estimator admission, multi-GPU
acceptance or final independent review.

### Prepared input normalization and binary format alignment

Evidence is under `test_data/fleet-sharding-implementation/prepared-unary-normalization/`.
Prepared preflight now accepts coefficient inputs for integer scaling and
automorphisms, and aligns coefficient/evaluation inputs for add/subtract. Before
any preparation or pilot, it checks the complete batch, selects distinct native
slots for every required input conversion and retained output, and rejects
missing configured role representatives. Each coefficient source shard receives
one full conversion shared by all invocations using that logical source; matching
evaluation inputs retain their original owners. Existing column boundaries and
device ownership are preserved. Cross-device redistribution remains separate.

`gpu_prepare.rs` reserves all conversion claims through the ledger, moves their
owned tokens with device states to the existing enqueue workers, copies each
source into its selected coefficient-format slot and performs an in-place NTT.
The primitive's existing in-place NTT/iNTT methods are now public; their bodies
and native synchronization are unchanged. Copies use the existing all-limb copy
runner and its reader tracking. No source is mutated, no CUDA pool or worker is
added, and no production host completion wait is introduced.

Converted native owners remain charged in the fixed logical baseline while
pilots claim their separate retained outputs. A profile miss waits for the actual
normalized sources only at the existing calibration boundary; cached production
uses normal GPU dependencies. Profile layout keys retain both original and
execution formats, including the RHS shape/format. The compiler binds original
argument IDs while executing through its private normalized owners. It retains
only each invocation's required inputs, releasing them after their last user
rather than extending them through unrelated later calls. Public manual plan
binding cannot substitute arbitrary normalized values.

Two new regressions check one shared conversion for two unary outputs, two fixed
conversions plus both binary outputs, mixed coefficient/evaluation operands,
capacity failure before preparation, frozen-call order, cached native-slot reuse,
unchanged source formats and existing CPU mathematical references. The initial
normalization fixture found an omitted claim in its input-reference download:
`to_cpu_matrix` itself creates an evaluation temporary for a coefficient source.
The fixture now admits that existing GPU conversion and its RNS/event resources;
the complete original-input equality assertion is unchanged. The initial failure
logs and source manifest are preserved, along with the intermediate unary-only
validation evidence. Both new regressions and the existing prepared matrix tests
pass on the final source revision.

Final workspace evidence covers 959 GPU-enabled tests (27 ignored), including
215 runtime and 281 primitive tests, and 608 CPU tests (25 ignored), with no
failures. Seven changed GPU binaries ran once; four unchanged GPU binaries and
all eleven unchanged CPU binaries reuse results only after exact SHA-256
comparison. All 68 source-manifest entries and all 22 executable hashes were
verified after completion. `git diff --check` passed. These are local unit-test
results; integration, physical multi-GPU and final independent review acceptance
are not claimed.

The matched existing unary reference fixture took 0.3546 seconds with the preceding
binary and 0.3149 seconds with the candidate. These are whole-process timings;
they do not establish prepared normalization throughput. Both workspace unit
builds are warning-free. Physical-resource certification, default prepared setup,
remaining invocation families, estimator integration and multi-GPU acceptance
remain incomplete; the pending memory-guarantee decision is unchanged.

### Direct integer scaling and evaluation automorphisms

Evidence is under `test_data/fleet-sharding-implementation/prepared-unary-kernels/`.
The shared matrix batch scalar kernel accepts either scalar-polynomial pointers
or canonical integer residues, requiring exactly one representation. Integer
residues use the existing scalar metadata region in the output auxiliary backing.
`GpuDCRTPolyMatrixColumnView::scale_integer` reduces signed arbitrary-precision
integers independently across CRT limbs with Rayon and avoids a GPU scalar
polynomial, upload workspace and scalar NTT. Explicit destinations preserve the
source format; standalone results retain the ordinary evaluation-format contract.

The existing automorphism kernel additionally gathers evaluation values directly.
For output frequency `k`, it reads frequency
`((a * (2*k + 1) mod (2*N)) - 1) / 2`, converting both frequencies to the
bit-reversed storage order of the native DIF NTT. The coefficient signed scatter
remains unchanged. Evaluation-to-evaluation views need no inverse NTT, temporary
coefficient matrix or final NTT. Explicit coefficient destinations still use the
existing conversion path. The former rejection test for an evaluation destination
now checks complete CPU output equality because that format is intentionally
supported; all other existing assertions remain intact.

Normal fleet unary calls and isolated preflight share these primitive paths.
Prepared matrix preflight and invocation binding now accept integer scaling and
automorphisms for evaluation inputs, reserve their complete outputs and use the
same range callback for pilots and production. Frozen invocation matching includes
the exact scalar/index; the allocation profile key excludes those values because
their allocation layouts are identical. Invalid indices and unplanned coefficient
conversion fail before a pilot. Prepared coefficient-input conversion, other
invocation families and automatic certified setup remain outstanding.

The new primitive regression checks mixed-width CRT limbs, signed 137-bit integer
scalars, zero, coefficient/evaluation sources, destination rectangles and untouched
entries against existing CPU operations. It passed at ring dimensions 32 and 2048,
covering local and multi-stage NTT layouts. The runtime regression supplies native
prepared outputs and a synthetic fully charged ledger, then verifies ordinary
preflight/calls, frozen-argument rejection, profile reuse across scalar/index values
and CPU output equality. Both new tests passed on the first run. The unchanged
view reference test took 0.3586 seconds with the preceding binary and 0.3088 seconds
with the initial new binary (0.2959 seconds after the empty-format fix); these
are whole-process timings, not kernel throughput.
The CPU and GPU unit builds are warning-free. No native synchronization, worker
count, physical seal or accepted memory-guarantee contract was changed.

The first full GPU-enabled unit run found one empty-matrix format regression in
the existing view metadata test (956 passed, one failed, 27 ignored across the
workspace). The direct evaluation path had changed the format of an empty
standalone automorphism. The implementation now preserves the ordinary
coefficient-format result for that empty case; its existing assertion is
unchanged. Initial logs, source manifest and aggregate results are retained with
the `initial-` prefix in this stage's evidence directory.

After that fix, every non-ignored workspace unit test passed: 957 GPU-enabled
tests with 27 ignored and 608 CPU tests with 25 ignored. The GPU run includes
281 primitive and 213 runtime tests. Changed GPU binaries ran once; unchanged
GPU binaries and all unchanged CPU binaries reuse the preceding results only
after exact SHA-256 comparison. Source manifests remained unchanged during the
final build/test stage. These local single-GPU results do not close the pending
physical-resource guarantee, other invocation classes or multi-GPU acceptance.

### Automatic prepared matrix preflight

Evidence is under `test_data/fleet-sharding-implementation/prepared-matrix-preflight/`.
`GpuDcrtBackend::set_memory_ledger` installs an accepted ledger once and checks
its execution identities and fixed per-device budgets against the fleet.
Ordinary GPU preflight then compiles negate/add/subtract requests: it intersects
stored input boundaries, preserves compatible device ownership, selects distinct
fitting native matrix slots across the complete batch and rejects missing
capacity before starting any pilot. Slots are matched to the exact native
parameter context, level and format without reservation side effects.

On a profile miss, the existing enqueue workers measure configured role 0 and
role 1 as needed. Fixed source preparation completes at the explicit calibration
boundary. A pilot claims the same full retained destination layout as production
and runs one column through the shared `PreparedMatrixOperation::run` callback;
unwritten private pilot entries are never read or published. Native logical demand
bounds the actual joint occupancy independently. Pilot outputs complete and drop
before production reservations are acquired. Release fences occur only at these
explicit calibration boundaries. A later nonzero owner without a configured
device-1 representative currently returns an explicit error; representative
redistribution still requires its own prepared plan.

Profiles are cached in the backend using operation, source/range layout and
prepared-inventory metadata, excluding sample contents, value IDs, pointers and
native reservation identities. Cache hits recheck current slot fit. After all
pilots finish, the ledger acquires every complete retained output and the compiled
batch is published atomically. Repeated preflight of a pending batch only validates
its frozen arguments. The ledger remains attached across recoverable errors.
Rayon handles independent boundary/key metadata and native demand queries;
ordered slot selection and reservation publication preserve exclusive ownership.

Ordered calibration groups may revisit a prepared store; joint occupancy now
deduplicates the inventory while preserving allocation order and separate native
claims. Duplicate/overlapping claims still fail native reservation before any
pilot submission. This supports interleaved related stores without summing their
independent high-water marks or reordering production allocations.

The new unit regression supplies synthetic fully charged residency with actual
owner identities and fixed budgets, then calls ordinary preflight and arithmetic.
It checks capacity failure before pilot claims, repeated preflight, real isolated
calibration, cached reuse, full output reservation, unsupported-request recovery,
and CPU reference equality for negate/add/subtract. No profile or executable plan
is hand-built by that test. Its synthetic residency is not a physical receipt or
seal acceptance. The new test, the preceding admitted-call regression and the
existing unary reference fixture all passed. Runtime passed 212 tests with four
ignored. The full GPU-enabled workspace evidence totals 955 passed, zero failed
and 27 ignored; CPU evidence totals 608 passed, zero failed and 25 ignored.
Changed GPU binaries each ran once. Unchanged GPU binaries and all unchanged
CPU binaries reuse preceding passing results only after SHA-256 equality checks.
Source/binary manifests identify the tested revision, and the runner verified
that crate sources remained unchanged throughout validation.

Both workspace builds are warning-free. The existing unary fixture took 0.3154
seconds in the preceding stage and 0.3057 now in whole-process timing; this is an
existing-path comparison, not a prepared-path throughput claim. Automatic finite
storage provisioning, full native driver/launch sealing, the other invocation
classes, explicit redistribution/alignment and complete estimator integration
remain outstanding. The default constructor does not yet install a certified
prepared ledger, and `allocation_tracking_complete` remains false.

### Admitted matrix calls through the ordinary backend

Evidence is under `test_data/fleet-sharding-implementation/prepared-fleet-matrix/`.
`GpuDcrtBackend::set_admitted_matrix_invocations` binds ordered typed arguments
and ledger-issued plans for negate, add and subtract. Compilation checks every
input owner/range, native parameter context, CRT level, format and exact output
claim before publishing the batch. Failed later entries cancel earlier claims.
The plan's fields are now crate-private, so external callers cannot construct a
plan from diagnostic widths or replace its native lease set.

Normal preflight and `Backend::negate/add/sub` consume the admitted batch. They
reject reordered/substituted inputs and missing or consumed admission. The
existing enqueue workers claim complete retained destinations once, then write
their disjoint rectangles through the existing column views. No intermediate
wave outputs, full-output clearing kernel, new pool, mutex or GPU completion
wait is introduced. Output owners remain private until every planned range has
been submitted; incomplete results are dropped on error.

The existing primitive constructor now exposes explicit destination allocation
at the native level and format. Native reservation/context matching checks exact
related-ring context identity before submission. Pending reservation ownership
keeps the backend non-Sync; the estimator's parallel result collection now uses
exclusive Rayon worker references without reducing concurrency.

Prepared width selection no longer rejects a native-fitting job solely because
its post-output slope hint is zero. Output-only range writers can have no free
scratch capacity and still fill the admitted destinations. An empty, zero-byte
temporary class checks the full supported range; native layout fitting and
physical bounds remain authoritative. Allocating observations are not relabeled
as allocation-free profiles.

The new unit test reserves three complete outputs with a synthetic fully charged
physical ledger, runs multiple waves through ordinary negate/add/sub calls and
compares each result with the existing CPU primitives. It verifies late batch
rollback, operand/order rejection, one retained output owner across waves,
consumed-plan rejection, and event-ordered reuse without a fence between calls.
The complete runtime binary passed 211 tests with four ignored. The physical
snapshot and profile in this fixture are synthetic: this is executable dispatch
evidence, not physical-seal acceptance.

Automatic storage provisioning, native requirement compilation for the other
invocation classes, default preflight/estimator plan construction, and physical
resource certification still remain. The ordinary methods consume supplied
admitted plans; the default diagnostic setup does not yet produce those plans.
`allocation_tracking_complete` remains false. No final review is claimed.

Both final workspace builds are warning-free. The unchanged unary reference
fixture took 0.3539 seconds before and 0.3154 after in whole-process timing; this
checks existing-path overhead, not prepared-path throughput. Initial private
metadata and exclusive-worker borrowing compile errors, plus the superseded
zero-filled destination test, are preserved with the source/binary evidence.
All changed GPU binaries passed once. Including unchanged binaries verified by
SHA256, current workspace evidence covers 954 GPU-enabled passes (27 ignored)
and 608 CPU passes (25 ignored). No integration or physical multi-GPU tests ran.

### Joint logical occupancy across related stores

Evidence is under `test_data/fleet-sharding-implementation/prepared-joint-occupancy/`.
The native execution owner now tracks actual claimed device-span bytes and their
joint high-water across all of its prepared stores. Claims increase this counter;
completed reuse-event observations retire charges. Event-ordered reuse counts a
backing span once. Pinned bytes and opaque event/stream units remain separate.
Storage destruction removes its remaining logical charge and inventory entry.

`GpuPreparedStorage::joint_occupancy` requires the complete distinct inventory on
one physical execution owner. It rejects empty, duplicate, incomplete and foreign
owner lists. Reset excludes reservations throughout the owner and rejects pending
release events without waiting. The caller still retains fixed baseline owners
and excludes unrelated submissions throughout calibration. This is logical
measurement, not physical certification or a concurrent admission receipt.

`GpuDeviceCalibration::measure_prepared` now accepts ordered groups of native
requests across related parameter stores. Native demand queries use Rayon;
reservation publication remains ordered and rolls back partial acquisition.
The pilot consumes all claims in the same composed dispatch and subtracts its
retained baseline from the actual joint peak. Its independent native demand bound
does not derive from that measured peak. The existing single-store test received
only a mechanical call-site migration, preserving its GPU/reference checks.

The new native regression distinguishes sequential storage peaks from overlapping
claims on Rayon workers and checks baseline retention, invalid inventories and
active-reservation reset rejection. The new runtime regression covers two CRT
rings on one execution owner and failure after actual submission followed by a
successful pilot. Both new tests and the existing reference pilot passed.

The GPU workspace build is warning-free (80 seconds); the CPU build is also
warning-free. All CPU binaries are unchanged from their 608-pass evidence.
Every changed GPU test binary passed once, including 280 primitive tests and
210 runtime tests. Together with identical-binary evidence, current workspace
coverage is 953 GPU-enabled passes with 27 ignored and 608 CPU passes with 25
ignored. No integration tests or physical multi-GPU acceptance ran in this stage.
The matched existing pilot took 0.5643 seconds before and 0.2404 after in
whole-process timing; this includes CUDA/process setup and establishes no kernel
speedup. Source manifests and the stage patch identify the tested revision.
Complete production invocation compilation, fleet/estimator consumption and the
finite driver/launch resource argument still remain. Certification stays disabled.

### Current workspace unit evidence and context-count isolation

The full workspace runs under
`test_data/fleet-sharding-implementation/workspace-after-column-dispatch/`
passed 608 CPU tests and 950 GPU-enabled tests, with one GPU failure in
`test_gpu_prepared_pinned_pending_upload_all_owners_drop`. Its process-global
context count changed from two to three while unrelated test groups ran.
The original failed run is preserved.

The test now uses the same child-process isolation pattern as the existing
prepared-occupancy regressions. Its original GPU operations and lifetime
assertion are unchanged. The parent requires both a successful child exit and
confirmation that exactly one test passed. This does not serialize production
work or change the suite's test concurrency.

Evidence under `test_data/fleet-sharding-implementation/context-count-isolation/`
records warning-free CPU/GPU workspace rebuilds, source hashes and the patch.
The changed primitive GPU binary passed all 279 tests once. All other GPU
binaries and every CPU binary have identical SHA256 hashes to the preceding
successful runs, so their results were reused without redundant execution.
The resulting exact-binary coverage is 951 GPU-enabled passes (27 ignored) and
608 CPU passes (25 ignored). Seven Python unit tests also passed with
`PYTHONPATH=scripts/lib`; the initial missing-path import failure is preserved.
No integration tests or physical multi-GPU tests ran in this checkpoint.

### Prepared CUDA resource evidence

Evidence is under
`test_data/fleet-sharding-implementation/prepared-cuda-resources/`.
The scoped GPU build is warning-free. Four new event/stream tests plus two
existing ownership/error-retirement tests each passed three identical runs.
The runtime suite passed 207 tests with four ignored. The first primitive suite
passed 275 and failed one existing test comparing process-global context counts
(the count decreased from two to one). That test passed three isolated runs, and
the unchanged full primitive suite passed all 276 tests at its original
concurrency. The original failure and rerun logs are both preserved; this is not
proof that the process-global counter assertion is isolated from other tests.

Completion events are retained by event sets and DMA reclaimers until their last
host observer releases them. Submission-stream slots include their bridge event.
Context release/epoch fences are created at setup. Scalar serde, release receipts,
compact coefficient readback and managed consumer links consume resource claims.
The copy path retires source readers against the output's existing completion.
Resource occupancy is counted in slots, separately from prepared device/pinned
bytes; no opaque event size is fabricated. Source hashes and `stage.patch`
identify the tested state. The subsequent checkpoint below covers timing
objects; remaining worker classes and complete production launch admission
still require coverage.

### P1 resource and reader-retirement evidence

Evidence is under `test_data/fleet-sharding-implementation/prepared-p1-events/`.
The scoped GPU build is warning-free. The new reserved-event P1 test and existing
cached/uncached P1 reuse test each passed three runs. The complete primitive
binary passed 277 tests; runtime passed 207 with four ignored. These are unit
results on one GPU, not physical multi-GPU coverage.

Cached and uncached P1 now retain prepared completion-resource leases with
sampled buffers. Existing sampled-ready events retire input/cache readers, and
output-owned scatter events retire scratch readers after the scatter launch.
The uncached path formerly linked the reader before the scatter, which did not
cover that read. Peer copies now wait for the reference sample and retain it
through copy completion. Cache destruction reuses its setup-created fence;
construction creates that fence before submitting work. No new stream/host
synchronization or concurrency reduction was introduced.

The first new test omitted permits for its final CPU readback. Its three failures
are preserved under `initial-readback-without-permit/`; the fixture now reserves
the readback events using the native execution class, while preserving exact
sample equality against the existing unprepared sampler. All existing tests
remain unchanged. The matched P1 fixture took 0.2844 seconds before and 0.2962
after in whole-process timing. This short diagnostic includes process/CUDA setup
and is not a kernel or production throughput measurement. Source manifests and
`stage.patch` preserve the exact tested native revision.

### Parallel calibration metadata

Per-instance concrete type resolution and operation identity calculation now use
Rayon. Group publication remains in first-occurrence order, preserving instance
and transcript ordering. No GPU submission or memory admission was moved to a
new worker. Evidence is under
`test_data/fleet-sharding-implementation/parallel-calibration-metadata/`: the
runtime GPU unit build is warning-free and all 207 tests passed (four ignored),
including the existing loop-dependent preimage-bound grouping regression.
`source-manifest.json` and `stage.patch` identify this subsequent Rust-only
change; the P1 native sources remain identical to their tested manifest.

### Setup-owned timing and related-context boundary

Evidence is under `test_data/fleet-sharding-implementation/prepared-timing/`.
The scoped GPU build is warning-free. The new prepared-context test and both
existing timing lifetime/abandonment tests each passed three identical runs.
The full primitive binary passed 278 tests; runtime passed 207 with four ignored.
Source hashes were verified unchanged after all runs. These are scoped unit
results on one RTX 4080 SUPER, not a final workspace or multi-GPU acceptance run.

Every execution owner now provisions one timing stream, start/stop events and
before/after participant events per device during context creation. Measurement
handles retain the existing exclusive timing lease and reuse those exact handles,
including across related rings and after an abandoned span. Host completion is
still awaited only by explicit timing collection. Timing resources are destroyed
with their execution owner, including partially constructed groups. Production
operand reader/release dependencies remain separate.

Related parameter contexts are rejected before CUDA construction once any
prepared reservation activates admission. Previously that path could allocate
additional NTT tables after preparation. The regression verifies precreated
related rings can still time work, new rings are rejected, overlapping spans
are rejected, and timing consumes no dynamic prepared event claim.

A source audit places every `Runtime.cu` stream/event creation in context setup.
Matrix backing, prepared-storage construction and covariance-cache construction
retain their setup guards; managed runtime events use prepared claims. This
closes these explicit handle-allocation sites, but does not prove a bound for
lazy driver function state, host-worker state or queued launches. Certification
remains disabled until those and complete production admission are implemented.
Matched whole-process timing for the existing abandoned-span fixture was
0.2447 seconds before and 0.2456 after; this is a short diagnostic, not a kernel
speedup or a production throughput result.

### Native demand and prepared pilot measurement

Evidence is under
`test_data/fleet-sharding-implementation/prepared-pilot-measurement/`.
`GpuPreparedStorage::demand` uses the same native claim planner as reservation
and reports separate device-span bytes, pinned bytes and CUDA resource units.
`GpuMatrixReservation::require_all_resources` closes all managed allocation
domains on the execution owner, including domains unused by earlier reservations.
The requirement persists across cancellation and applies to related workers.
It does not certify driver resources or physical residency.

`GpuDeviceCalibration::measure_prepared` now reserves the complete supplied
pilot request before running its range callback, enforces those managed domains,
and requires every claim to be consumed. It measures occupied high-water above
retained fixed storage and uses the independently queried native demand as the
logical bound. The caller must retain baseline owners throughout the boundary.
The callback and result preserve the actual production primitive path; the
regression runs the same column-view negate callback for its pilot and separately
reserved output, comparing both with the existing CPU primitive reference.
At this checkpoint the collector handled one storage/context. The joint
occupancy checkpoint below extends it across related-ring stores.
The fleet/estimator still need their frozen invocation plans connected to this
collector and to production ledger admission; this API is not that integration.

Enforcing the complete resource domains exposed per-input event creation in
`finish_matrix_batch`. It now records the existing output completion first and
uses that record for all input reader waits. This preserves the batch's single
stream and avoids additional lazy events, without a host wait or reduced
concurrency. The three initial missing-event failures are retained under
`initial-batch-reader-event-claims/`; the new fixture was not weakened. Initial
fixture compile errors are retained separately as `initial-test-api-errors.log`.

The final scoped GPU build is warning-free. The new pilot test and two existing
view/reader-reuse tests each passed three identical runs. Complete primitive
execution passed 278 tests; runtime passed 208 with four ignored. Source hashes
were unchanged throughout validation. The existing batch workspace fixture took
0.2730 seconds before and 0.2680 after in whole-process timing; this short
diagnostic is not a kernel speedup or production throughput claim. Full workspace,
physical multi-GPU, final synchronization and independent-review gates remain.

### Reservation handoff and initial specialization

Evidence is under `test_data/fleet-sharding-implementation/prepared-plan-handoff/`.
Reservations now retain their exact current requests through partition, worker
transfer, activation and completion. An unsubmitted request can be strictly
narrowed while preserving its original admitted envelope and exclusive capacity;
completed waves may grow back within that envelope. Unchanged/wider unsubmitted
requests remain errors. Native bound checks precede all mutations, and failed
rearming preserves the Rust request metadata.

Column-ledger publication now requires every managed resource domain on each
prepared execution owner. This connects publication to native enforcement; the
complete production invocation compiler and physical seal are still required.
The first full suite exposed a regression in the unchanged-request state guard.
Its failure and source manifest are preserved under
`initial-unsubmitted-rearm-regression/`; existing tests were unchanged.

The corrected scoped GPU build is warning-free. Four reservation tests each
passed three runs, including the previously failing reader/worker regression.
The primitive suite passed 279 tests; runtime passed 208 with four ignored.
The runtime binary finished linking before its suite began; crate source hashes
remained frozen. Whole-process batch-workspace timing was 0.2710 seconds before
and 0.2681 after, a diagnostic rather than production throughput evidence.

### Setup-owned kernel and driver function loading

Evidence is under `test_data/fleet-sharding-implementation/prepared-kernel-loading/`.
Initial execution-owner construction now loads the explicit CUDA kernel inventory
on every configured device before returning a context. Related parameter contexts
reuse that owner's prepared function handles. The inventory covers ordinary and
batch NTT layouts, both transform directions, all fused top-stage widths, both
compact word widths, and both tensor row-sum driver variants. The native object
contains 98 CUDA entry points; demangled `cuobjdump --dump-elf-symbols` output
matches all 98 explicit references with no missing or extra specialization.
The raw symbols and comparison are preserved beside the source manifest.

`cudaFuncGetAttributes` loads each kernel without executing a sample, as described
in the [CUDA lazy-loading documentation](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/lazy-loading.html).
Tensor row-sum launch entry points and CUDA function handles now belong to the
execution owner's device partition, replacing first-use thread-local driver
lookups. Dispatch consumes immutable setup handles, and missing provisioning is
an error. It introduces no host completion wait or production serialization.
This addresses kernel/function loading only: runtime launch-local memory, queued
launch resources and other permitted worker state still need their finite bound.
`allocation_tracking_complete` remains false; loading code is not certification.

The scoped build is warning-free. Existing tensor row-sum reference tests passed
at ring dimensions 32 and 256, covering both driver variants; related-context
lifetime checks passed. Runtime passed 208 tests with four ignored. The first
primitive suite passed 278 and failed the existing process-global context-count
assertion (two observed versus one expected). Its original failure is retained.
The assertion passed three isolated runs, and the unchanged full primitive suite
then passed all 279 tests at its original concurrency. This does not resolve the
known concurrent-test interference risk in that global-count assertion.

All crate source hashes remained frozen throughout validation. The matched
existing tensor row-sum fixture took 0.6272 seconds before and 0.6270 after in
whole-process timing; this includes setup and is not a kernel throughput claim.
The initial misspelled native error accessor compile failure is also preserved.
No new implementation subagents were used. Complete invocation compilation,
production/estimator consumption, final full-workspace and physical multi-GPU
validation, and independent review remain outstanding.

### Executable column reservations on existing workers

Evidence is under `test_data/fleet-sharding-implementation/prepared-column-dispatch/`.
`GpuColumnMemoryPlan::execute` now consumes a ledger-issued schedule and its
per-device leases using the existing `GpuEnqueuePool`. It validates active owner
coverage before moving backend states. Initialization consumes fixed and complete
output claims once. Subsequent lazy jobs retain those native owners, specialize
initial scratch requests when necessary, and rearm only scratch within its
original admitted envelope. Native dispatch completion requires every prepared
claim to be consumed. There is no release/re-reserve gap between waves.

The callbacks receive the corresponding physical leases for native owner binding.
Unsubmitted commands cancel normally; already submitted native owners retain their
ordinary retirement edges. Backend states return to the caller after callback
failure, while partial outputs and scratch leases follow their existing native
release lifecycle. No new worker pool, host completion wait, mutex or device copy
is introduced by this dispatch layer.

The new unit regression consumes real fixed/output/scratch reservations, writes a
retained destination through column views across a full wave and short tail, and
compares the result to the existing CPU identity primitive. An intentional failure
after GPU submission is followed by another invocation on the same worker states;
released logical slots become reservable while returned outputs remain exclusive.
Its physical ledger snapshot and profile slope are synthetic test inputs: this
proves the reservation execution protocol, not physical setup certification.

The runtime GPU build is warning-free. The new execution/error-recovery test
passed three runs, and all 209 runtime unit tests passed with four ignored. The
initial fixture omitted per-limb readback events; all three failures and source
hashes are preserved under `initial-readback-claims/`. The fixture now queries the
native output execution class to reserve the exact readback event count. Existing
reference tests are unchanged. Initial compile errors are preserved separately.
All crate source hashes remained frozen during final validation. The existing
unary fleet fixture took 0.3039 seconds before and 0.2918 after in whole-process
timing; that diagnostic does not measure the new dispatch layer's throughput.

The dispatch API is now executable and tested, but the typed fleet invocation
compiler still needs to supply these callbacks/plans at normal runtime and
estimator entry points. Native physical resource sealing and full final validation
remain required; this checkpoint is not an acceptance report.

## Implemented foundation

- Runtime and estimator role-width derivation inspect every configured device.
  Nonzero devices reuse the measured profile but contribute their own headroom
  to the shared nonzero capacity. The former shared-context one-column fallback
  is removed; runtime rejects nonexclusive execution owners.
- Capped water filling accepts individual device capacities, including idle
  zero-capacity devices, rather than requiring identical nonzero output caps.
- `crates/runtime/src/gpu_memory.rs` implements atomic fleet reservations,
  allocation IDs, reserved/resident/release-pending transitions, and explicit
  quiescent baseline refresh. Reservations survive residency transitions and
  queued frees. Explicit native release receipts now connect consumed ordinary
  and compact owners to nonblocking completion polling. Manual release-complete
  transitions are private to the ledger; queued frees cannot be retired by a
  production caller without polling. Bound native owners now notify the ledger
  automatically at their final drop; fleet aliases share those native owners.
  Automatic allocation admission through every production entry point is still
  pending: this module alone does not establish complete runtime memory safety.
- `crates/runtime/src/gpu_schedule.rs` provides validated ownership intervals,
  lazy local jobs and actual logical wave counts. Compact-RHS multiplication
  uses it without assuming each consecutive group of stored chunks contains
  at most one chunk per device. Ordinary unary operations preserve their stored
  ownership and interval boundaries when consumer widths change. Binary add/sub
  intersects both stored layouts and preserves the left ownership, retaining
  peer-only transport for incompatible right ownership. Fused compact row-block
  products now use the same lazy schedule, including repeated chunks per device.
- Ordinary and compact fleet clones share immutable native storage through an
  `Arc`; they no longer allocate another GPU payload while preserving the same
  logical identity. Native primitive clones remain explicit deep copies.
- Device-local fixed-operand preparation for compact products runs through Rayon.
  Independent block-output ordering is parallel; ordered boundary intersection
  and atomic reservation publication remain sequential.
- Native release receipts record events on release streams after producer/reader
  dependencies and queued frees. Failed frees leave a sticky execution-owner
  failure rather than permitting capacity reclamation. Compact hard-cutoff
  pinned buffers use the existing deferred reclaimer.
- The primitive allocation query is public and reports the execution class from
  the same native plan used by matrix creation: empty, shared stream, or per-limb
  streams. Widths 1/4/5 and the row boundary are covered. These classes are not
  yet sufficient bounds for all primitive temporary allocations.

## Initial foundation evidence

The checkpoint is an uncommitted working tree based on `51bfdbd1e`.
Source hashes and logs are under
`test_data/fleet-sharding-implementation/initial-foundation/`:

- `gpu-unit-build.log`: warning-free
  `cargo test -r --workspace --lib --features gpu --no-run`.
- `runtime-gpu-repeat.log`: the same built runtime binary with filter `gpu_`,
  normal multithreaded execution, five completed runs; each run passed 56 tests,
  failed zero and ignored one. The remaining 95 tests were filtered out.
- `allocation-query-repeat.log`: the same primitive binary with filter
  `test_gpu_matrix_allocation_`, five completed runs; each run passed three tests.
  The multi-partition query test returns early with fewer than two devices, so
  this is **not** physical multi-device coverage.
- `source-manifest.json`: exact tested source hashes and device metadata.

GPU execution was outside the sandbox on one RTX 4080 SUPER. Formatting and
`git diff --check` passed. These are narrow foundation checks, not the full
workspace CPU/GPU unit runs, 300-run synchronization gate, performance acceptance,
or final independent review.

## Release and shared-owner validation checkpoint

Evidence for this next checkpoint is kept separately under
`test_data/fleet-sharding-implementation/release-and-owner-lifetimes/`.
The initial 300-run runtime check had five failures because two runtime tests
used no guard or a different serial-test key while other tests required exclusive
calibration access to the default CUDA pool. Both now use the established
`gpu_context` test key. Production concurrency and fail-closed context checks
remain unchanged. The corrected runtime command completed all 300 runs with
zero failures (59 passed and one ignored per run).

The four focused native reader/release/compact-hard-cutoff tests passed all 300
runs of their identical command. Their source files did not change during the
subsequent test-harness repairs.

The initial complete CPU unit run passed ten crate binaries and failed 29 gadget
tests. Their shared helper and the public-output merge test read staged or
artifact outputs as resident matrices. Both now use
`ExecutionResult::materialize_output`; the helper cleans up staged storage;
mathematical expected values and individual arithmetic assertions are preserved.
CPU and GPU workspace builds after these repairs are warning-free. Complete
CPU execution passed all 11 crate binaries: 606 tests passed and 25 were
ignored. GPU-feature execution passed all 11 crate binaries: 825 tests passed
and 27 were ignored. GPU output helpers now materialize through the GPU backend;
no GPU computation moved to CPU. The estimator regression now covers an atomic
first request plus the column-separable ordinary/compact extension cases.
These repaired GPU paths also passed five identical-command smoke repetitions
per command (`harness-smoke.log`).

The current checkpoint files are `cpu-build.log`, `gpu-build.log`, `cpu-units.log`,
`gpu-units.log`, `runtime-repeat.log`, and `source-manifest.json`. Failed initial
runs remain alongside them for audit. Tests ran on one RTX 4080 SUPER with
16376 MiB; multi-device coverage is still uncompleted. Passing these unit suites
is not final implementation acceptance.

## Bound native owners

`GpuMemoryLedger::new` now accepts optional ordered device/execution identities.
Omitting them permits synthetic accounting tests but rejects native binding.
At this checkpoint, `bind_matrix` and `bind_small_matrix` validated the reserved allocation ID,
execution owner, physical device and minimum native footprint before attaching
an observer to the actual native owner. Binding twice is an error. The primitive
owner queues its normal producer/reader-ordered destruction before delivering a
release receipt to the dispatcher. Native deep clones have separate storage and
no inherited observer; logical fleet clones retain the same observed owner.

The dispatcher drains notifications at admission and refresh boundaries. Failed
release notifications retain charges and reject further admission/refresh.
Successful notifications still require a nonblocking CUDA event query before
capacity is returned. Dropping the dispatcher does not make native destruction
depend on a live receiver.

This is a lifecycle connection, not complete invocation admission. Callers must
still reserve the full planned output and simultaneous scratch before native
construction. The binding check covers a known minimum owner footprint; compact
preimage scratch, opaque CUDA resources and unrelated outputs are not proved by
that check. The normal backend dispatchers have not yet bound every allocation.
The submission-lease checkpoint below supersedes this binding API and closes the
failed partial-submission accounting gap. Initial quiescent epoch establishment,
complete primitive bounds and production reservation transfer remain required.

Evidence for this stage is kept under
`test_data/fleet-sharding-implementation/bound-native-owners/`. The targeted ledger
suite passed nine tests, including automatic ordinary/compact release after the
last fleet alias, source readers, wrong execution identities, undersized and
duplicate binding, and conservative failed-release handling. The runtime `gpu_` command and the
four native reader/release tests each completed 300 identical-command runs with
zero failures. The source snapshot for these runs is `source-manifest.json`.

Native context and shape metadata are now read-only outside the primitive crate;
callers use existing matrix accessors, preventing safe callers from changing the
owner identity after binding. `final-source-manifest.json` captures that API
encapsulation change. The final GPU workspace unit run passed all 11 binaries:
828 tests passed, 27 ignored, no failures. Both CPU/GPU unit builds were
warning-free. This stage did not change CPU-feature production code; the preceding
606-test CPU unit result remains the current CPU execution evidence. All GPU
checks were local single-device checks; physical multi-device, complete admission,
estimator migration, performance and independent-review gates remain open.

## Remaining work, in dependency order

These are the remaining implementation obligations in the authoritative plan;
completed primitive stages above do not establish full production admission.

1. Connect retained hash decomposition and complete `CenteredExtendSmall`, then
   trapdoor/preimage staging, caches and native resource claims. Retained compact
   decomposition, compact RHS multiplication and its exact narrow/wide expansion
   query are implemented at the primitive layer.
2. Extend runtime invocation admission to the remaining compact operand classes
   and preimage. Decomposition now has typed retained compact output admission.
   Plan width-dependent multiplication workspace claims, fixed LHS
   replicas and compact RHS ownership. Preserve compact payloads; do not model
   them by allocating full-DCRT proxy matrices. Validate and reserve complete
   batches before publishing production results.
3. Finish admitted import/export, scalar and atomic boundaries and remaining
   mixed-context reductions. Include artifact transfers, caches, source staging,
   finite event/stream/pinned resources and reader/release lifetimes in the
   canonical inventory. Zero-row gadget rejection is now covered.
4. Add default graph-wide automatic setup, inventory provisioning and admission.
   The existing explicit `prepare_memory` API and compiled ordinary/compact paths
   remain insufficient. Initial setup must retain both residency checks; later
   opaque CUDA budget excess must not stop otherwise valid admitted work.
5. Make estimator measurements consume actual admitted runtime plans and wave
   classes, preserve CUDA/device versus fleet-wall timing meanings and count
   transfers once. Remove the remaining unchecked/manual diagnostic production
   paths after their admitted replacements cover the required operations.
6. Audit the complete allocation/range/metric inventory and compare matched
   baseline/candidate production timings. Run all non-ignored workspace unit
   tests, repeating only necessary lifetime/synchronization checks. Integration
   tests remain outside the authorized validation scope.
7. Obtain the final independent full review. Physical two-/three-device checks
   are deferred at the user's request; report that hardware coverage separately
   without blocking local completion. Do not use implementation subagents.

Do not mark the goal complete until every non-deferred requirement in the
authoritative plan has current implementation and validation evidence.

## Persistent enqueue workers

`crates/runtime/src/gpu_enqueue.rs` owns one persistent host thread per configured
device for multi-device batches and preserves caller-thread execution for one
device. Each bounded batch transfers owned device states and drains every reply
before returning. An operation error cancels callbacks that have not started;
callbacks already running retain their native ownership and release protocol.
Caught callback panics make the pool unavailable for further submission. No
borrowed lifetimes are erased or extended. If the caller is a Rayon worker, it
services available Rayon work while collecting replies so nested CPU preparation
can progress even with one Rayon worker. This creates no replacement Rayon pool.

The estimator uses these workers for role pilots and measured fleet iterations,
with prepared inputs shared through one `Arc`. Its former blocking Rayon barriers
are removed. Independent input preparation and request ordering use Rayon. Fleet
wall timing still includes submission and completion; the existing per-device
host measurements have not yet migrated to CUDA-event spans.

The runtime common column dispatcher uses the same worker implementation for
ordinary unary operations, add/subtract, row-block addition, transpose, slices,
tensor products/row sums, and compact decomposition. Matrix operands retain
shared fleet storage when captured by a command; capturing them does not clone
native payloads. The worker layer does not add a device-completion wait. Existing
in-operation calibration boundaries and specialized dispatch paths remain to be
replaced as listed above, so this is not the complete worker/admission migration.

The tests include persistent thread identity, nested Rayon work, callback error
recovery, panic reporting, caller-thread execution, and an actual fleet launch
that fails after enqueuing native work and then successfully reuses the backend.
Watchdogs bound liveness failures. The fleet test uses every detected device;
on the local one-device host it does not establish physical multi-device progress.
Evidence for this checkpoint is under
`test_data/fleet-sharding-implementation/enqueue-workers/`.

The initial worker checkpoint passed 300 normal runtime runs, 300 worker-filter
runs with `RAYON_NUM_THREADS=1`, 300 targeted estimator runs and five targeted
estimator runs with one Rayon worker. Its full GPU suite found one initialization
regression: the metadata-only percentage test constructs an empty worker set.
The worker utility now accepts this empty set without creating a device or a
thread; physical fleet constructors still reject empty placement. The existing
test and its percentage assertions are unchanged. Initial failure logs are
retained in the checkpoint directory.

The next batch extends worker dispatch to constants, ordinary and compact
multiplication (including fused blocks), all hash/uniform/Gaussian samplers,
concat, CRT recomposition, centered compact extension, preimage sampling and
resident redistribution. `GpuMatrixOperand` retains the original shared shard
owner for resident inputs and owns only materialized ranges. Fixed ordinary
operands, compact inputs and trapdoor replicas can therefore cross the worker
boundary without deep native clones. Compact views are created within the
command while retaining their source owner. Each preimage wave prepares only its
target ranges with Rayon before handing their owned results to GPU commands;
global seed offsets and coupled sampler rows remain unchanged. Independent
trapdoor preparation and compact import row copies also use Rayon.

The batched checkpoint evidence is under
`test_data/fleet-sharding-implementation/enqueue-owned-inputs/`. It does not yet
establish complete memory admission, isolated preflight, allocation-free ordinary
range views, physical multi-device validation or estimator timing migration.

The final batched source passed both warning-free workspace unit builds.
All 11 CPU unit binaries passed: 606 tests passed, 25 ignored, zero failures.
All 11 GPU-feature unit binaries passed: 833 tests passed, 27 ignored, zero
failures. The normal runtime `gpu_` filter, the `gpu_enqueue` filter with
`RAYON_NUM_THREADS=1`, and the complete estimator unit binary each passed 300
identical-command repetitions. The targeted estimator extension test also passed
five runs with one Rayon worker. The respective per-run counts were 67 passed
and one ignored, five passed, 54 passed, and one passed. All repetitions completed;
none failed. GPU execution was outside the sandbox on one RTX 4080 SUPER.

The exact crate source hashes in `source-manifest.json` remained unchanged
throughout the repetitions and both full unit runs. `validation-summary.json`
records totals; `repeat.log`, `gpu-units.log`, `cpu-units.log`,
`gpu-build-final.log` and `cpu-build.log` retain the commands and raw results.
These close this batch's local unit checks, not the remaining plan requirements
or its physical multi-device, performance and independent-review gates.


## Submission leases and failed commands

The ledger now transfers complete reservation transactions into unique
`GpuAllocationLease` values before dispatch. The lifecycle distinguishes reserved,
leased, submitted, resident, release-pending and abandoned allocations. A command
owns its unstarted leases; dropping them cancels their charges through the owning
dispatcher. Native constructors run through the lease, which validates the
execution identity before calling the constructor and attaches the final-owner
release observer on success. Native initialization failure or panic before an
owner can be attached leaves its reservation charged and makes subsequent
admission fail. A callback failure after binding an owner uses the usual native
release protocol and does not poison an otherwise healthy ledger.

No caller can cancel a submitted allocation by presenting an old reservation ID.
The ledger processes lifecycle notifications before cancellation and admission.
Binding and becoming resident never subtract the reservation. Ordinary and
compact owners retain the same release-event requirement as earlier checkpoints.
There are no new GPU completion waits in the command or lease implementation.

The enqueue pool now also returns device state when a worker channel has stopped.
The failed-send command runs only its cancellation/reply path on the caller;
it never submits the callback there. This preserves device order and unstarted
leases and marks the worker pool unavailable instead of returning a shortened
state vector that could be mistaken for a different fleet.

Regression cases cover atomic lease transfer, all future output chunks plus
concurrently live temporary reservations, unstarted cancellation, partially
submitted failure/panic, failed worker channels, and real native output/reader
lifetimes through enqueue commands. The native test uses every detected GPU and
compares the result with the trusted CPU matrix primitive. Its locally available
one-device execution is not physical multi-device evidence.

This checkpoint makes the reservation lifecycle executable through the worker
boundary. The regular backend still needs the complete primitive requirements and
invocation admission that supply these leases. Known owner-footprint queries are
not complete kernel bounds. At this checkpoint, `gpu_matrix_negate_batch` still
allocated device pointer/stride/modulus arrays beyond its result owner; the batch
workspace checkpoint below replaces those allocations.
Other remaining resources include RNS conversion metadata in `MatrixCrt.cu`,
compact expansion buffers and preimage packing controls in `MatrixSmallRhs.cu`,
and sampler/covariance buffers in `MatrixTrapdoor.cu`. All these paths need their
native size planners and reusable workspace/destination contracts connected to
the invocation requirements before production admission is complete.

Validation for this checkpoint is under
`test_data/fleet-sharding-implementation/submission-leases/`. Both workspace
unit builds were warning-free. All 11 CPU binaries passed (606 passed, 25
ignored), and all 11 GPU-feature binaries passed (838 passed, 27 ignored).
There were no failures. The runtime `gpu_` command, the `gpu_enqueue` command
with `RAYON_NUM_THREADS=1`, and the `gpu_memory` command with
`RAYON_NUM_THREADS=1` each passed all 300 identical-command repetitions. The
complete estimator binary also passed five runs with one Rayon worker. Their
per-run counts were 72 passed/one ignored, six passed, 13 passed, and 54 passed,
respectively. All 905 repetitions completed without failure. Crate source hashes
were unchanged throughout validation; `source-manifest.json`,
`validation-summary.json`, and the raw logs retain the evidence.

These results establish the local lease and command-lifetime checkpoint. They do
not establish complete invocation memory bounds, backend-wide admission,
physical multi-device concurrency, or the final performance/review gates.


## Native batch metadata workspaces

`crates/primitives/cuda/src/matrix/MatrixArithBatch.cu` now has one checked,
aligned region planner for binary arithmetic, negation, automorphisms, scalar
multiplication, multiplication and fused multiply-accumulate. All six native
entry points allocate through that planner. Their arithmetic kernels, argument
ordering, coefficient widths, format transitions and input reader tracking are
preserved.

A batch first waits by stream events for its first output's exclusive auxiliary
partition. When its metadata fits there, it borrows that already allocated region;
it does not allocate additional device pointer/stride/modulus arrays. Independent
batches borrow different output owners. When the metadata exceeds that region,
one additional contiguous arena holds all metadata. Its queued free precedes a
new completion recorded on the first output, so that output's final release
receipt also dominates the arena release. Failed releases set the execution
owner's sticky failure flag. Error paths fence only the affected stream if normal
reader/write completion installation cannot be established; successful commands
remain asynchronous.

The ordinary native allocation query now exposes `aux_workspace_bytes`, a subset
of the existing `aux_bytes`, excluding descriptor storage and padding. Rust's
`GpuDCRTPolyParams::matrix_batch_workspace_bytes` calls the same native planner
as production and returns aligned metadata bytes plus the additional allocator
request. The two workspace classes are determined by whether that metadata fits
the first output's auxiliary region. The query checks integer overflow and
unsupported shapes without allocation. It describes batch metadata, not a full
invocation bound: fixed operands, output storage, conversion/sampler work,
allocator granularity and opaque CUDA resources still require their own planning.

New unit cases query every batch operation across the borrowed/separate boundary,
reject overflow and invalid counts, and execute both workspace classes while
releasing source and first-output owners with outstanding native readers.
Existing arithmetic, mixed-width/maximum-limb and NTT composition tests remain
unchanged. Evidence is in
`test_data/fleet-sharding-implementation/batch-workspace/`.

The final GPU workspace unit build was warning-free. All 11 GPU-feature unit
binaries passed: 840 passed, 27 ignored, zero failures. Each of the four native
commands (workspace classes/lifetimes, arithmetic operation batches, maximum and
mixed limb widths, and fused NTT composition) passed 300 identical-command runs.
The runtime `gpu_` command also passed all 300 runs; the complete estimator unit
binary passed five runs with `RAYON_NUM_THREADS=1`. All 1,505 repetitions finished
without failure. Source hashes were unchanged throughout execution.

The CPU unit build was warning-free and reused the same 11 cached binaries from
the preceding 606-passed/25-ignored full CPU run. This GPU-only native change did
not require another execution of unchanged CPU binaries. The evidence summary
links to that execution log explicitly.

A matched process-wall regression smoke alternated the saved baseline and current
primitive unit binaries for 12 pairs of the unchanged arithmetic-batch test. All
24 executions passed; median process times were 0.232399 and 0.229038 seconds.
These times include context setup, CPU reference checks, transfers and teardown;
they do not establish a kernel speedup or satisfy the plan's production-wave
performance gate. Binary hashes, every timing and the measurement scope are in
`paired-regression.json` and its raw log.

Complete invocation bounds, production ledger coverage, isolated preflight,
ordinary range/destination views, estimator metric migration and physical
multi-device/final-review gates remain open as listed above.

## Ordinary products and intersected column schedules

`GpuDcrtBackend::launch_owned_column_operation` now uses the shared lazy
`GpuColumnSchedule` for unary and binary operations, ordinary/scalar products,
fused multiply-accumulate, row-block addition, compact decomposition, and slices.
It retains the selected operand's device ownership, intersects every scalable
input's stored boundaries, clips slices in global input coordinates, and rebases
only their output column indices. Source boundaries and output ordering are
prepared with Rayon; their host storage scales with stored chunks, not future
compute waves. The existing calibration outputs are still discarded before the
retained production schedule is constructed.

Ordinary products retain RHS ownership; scalar products retain the matrix-valued
operand's ownership on either side. Complete resident fixed and scalable operands
are borrowed with their fleet owners retained through native enqueue. Fixed
operand preparation uses Rayon across devices. Multiply-accumulate intersects
all scalable operands and its bias and keeps the first product's scalable owner.
Row-block addition keeps the right-hand matrix's owner; decomposition keeps its
first block's owner. Slices reject invalid ranges before subtraction or dispatch.

New unit tests check width increases and decreases, both scalar positions,
three independently chunked accumulation inputs, row-block boundaries,
decomposition reconstruction, shifted and empty slices, and dropping input
owners before inspecting outputs. Arithmetic expectations use trusted CPU
matrix operations or gadget reconstruction. Existing boundary and lifetime tests
are unchanged.

This is a scheduling checkpoint, not complete invocation admission. Partial
ordinary ranges and accumulate operands still materialize copies. Incompatible
placements still use the existing peer-only transport; its destinations and
lifetimes must enter explicit redistribution admission. Fixed replicas are not
yet restricted to admitted participants, and in-operation calibration remains
until the separate preflight migration. Transpose, tensor/concat, preimage,
imports, CRT recomposition, estimator schedule identity, and the other remaining
requirements above are still open.

Validation evidence is in
`test_data/fleet-sharding-implementation/ordinary-schedules/`. Both workspace
unit builds were warning-free. All 11 GPU-feature unit binaries passed, with
843 passed, 27 ignored and zero failures. The runtime `gpu_` command passed all
300 identical-command repetitions (75 passed and one ignored per run). The full
estimator binary passed five runs with `RAYON_NUM_THREADS=1` (54 passed each).
All 305 repetitions completed with zero failures, and crate source hashes were
unchanged throughout validation. The CPU build reused the same eleven cached
binaries from the preceding 606-passed/25-ignored CPU execution; the summary
links that execution evidence rather than claiming a new CPU run.

A regression smoke alternated saved baseline and current runtime binaries for
12 pairs of the unchanged resident-operand lifetime test. All 24 executions
passed. Median process wall times were 0.224494 and 0.189332 seconds, including
setup, arithmetic verification, transfers and teardown. This test covers the
resident path and is not an isolated general-product or production-wave timing;
it does not establish a kernel speedup or close the performance gate. The
measurement contract, binary hashes and individual timings are retained in
`paired-regression.json`. All GPU execution used one local RTX 4080 SUPER;
physical multi-device and final independent-review gates remain open.

## Asynchronous retirement after native enqueue failures

The blocking failure cleanup introduced with batch metadata arenas has been
replaced by event retirement. `gpu_context_retire_stream` records submitted work
and makes the execution owner's producer and release streams observe that event.
This protects final destruction and subsequent in-place reuse even when normal
per-input/per-output completion installation was interrupted. Successful commands
retain their existing dependencies; this conservative fan-out runs on failed
enqueues only and does not wait on the host.

`MatrixBatchWorkspace` queues its separate arena free on the consuming stream
before retiring the first output's producer/writer streams. The three blocking
failure fences in `MatrixUtils.cu` have also been removed; failures in reader and
writer completion updates use the same retirement mechanism. This includes
previously uncovered CUDA error returns after event recording or stream waits.
The caller's current device is restored by retirement.

If CUDA cannot establish retirement, the execution owner records both an
unretired-work flag and the sticky memory-release failure. Ordinary and compact
allocation constructors and related-context creation reject that epoch. Release
receipts and explicit release fencing report failure. Last-owner destruction
retains its ordinary/compact allocations and the context itself, including NTT
tables, streams and live-context registration. It does not turn an uncertain
reader lifetime into reusable capacity. These quarantined resources last until
process exit; this is an error outcome, not a production fallback or replan.

The added lifetime test exercises the production retirement entrypoint from
concurrent Rayon jobs, shared/per-limb matrix allocation classes, borrowed and
separate batch metadata, pending downstream readers and release receipts. A
second unit test runs its deliberately failed retirement in an isolated child
process. A missing retirement stream reaches the shared failure branch without
inducing device loss. It checks allocation rejection, release/fence errors,
retained native pool allocations and retained live-context registration after
last-owner drop. The internal child-process marker is test-only and is not a
production configuration or calibration input.

This checkpoint does not complete production admission or isolated preflight;
the remaining plan requirements above continue to apply.

Validation evidence is in `test_data/fleet-sharding-implementation/async-error-retirement/`. Both final workspace unit builds were warning-free. All eleven CPU unit binaries passed (606 passed, 25 ignored); all eleven GPU-feature binaries passed (845 passed, 27 ignored). The retirement and batch-arithmetic filters each completed 300 repetitions without failure. The mixed-width metadata filter completed 58 successful children before the user changed intermediate GPU validation to three consecutive successful runs; 57 have driver END records and the last child exit is recorded in `repetition-policy-change.json`. No active test was killed. The resumed runtime GPU filter passed all three runs, 75 passed and one ignored per run. Crate source hashes stayed unchanged during validation.

The unchanged batch-arithmetic process benchmark passed all twelve alternating baseline/current pairs. Median whole-process times were 0.244721 and 0.235013 seconds; setup, CPU reference checks, transfers and teardown are included. These measurements do not establish a production-wave or kernel speedup. All GPU execution used one RTX 4080 SUPER. Subsequent intermediate checks use three consecutive GPU successes; final acceptance retains the stronger required gates.

## Parallel schedule, transform workspace and timing implementation

Independent edits were coordinated by file ownership and integrated before a
shared build. The latest user instruction permits parallel subagents and reduces
intermediate GPU checks to three consecutive successful runs. The final gates in
the approved plan remain unchanged.

The fleet now freezes contiguous fresh ownership independently of compute width
and routes sampling, constants, tensor/transpose, scatter/import range planning
and preimage output jobs through `GpuColumnSchedule`. Row concatenation and CRT
recomposition inherit intersected source intervals. Column/diagonal concatenation
preserves each source owner at its exact output offset. Empty source intervals
are omitted; empty concatenations preserve their declared shape. Transposing a
shardless empty matrix that lacks required ring metadata returns an explicit
error before dispatch rather than poisoning enqueue workers. Native empty
owners retaining their parameters remain supported. Fresh ownership currently
uses equal hypothetical caps, not checked output admission caps.

Decomposition and identity kernels use existing device descriptors and modulus
tables directly, removing their six/four metadata allocations and uploads. CRT
and gadget-correction metadata reuse exclusive output/copy auxiliary storage or
one checked contiguous arena. Output completion follows separate device-arena
free; error retirement precedes coefficient-copy destruction. CRT recomposition
uses one combined pinned upload. Native queries expose coefficient-copy requests,
correction requests, output rows, device metadata and pinned metadata through
`crates/primitives/src/matrix/gpu_transform.rs`. Allocator granularity and opaque
CUDA resources remain separate obligations. Pinned-reclaimer completion and a
coefficient copy's release cannot be inferred from output completion alone.

`GpuDCRTPolyParams::begin_device_timing` records CUDA-event spans covering all
compute/release streams of its execution owner, including related ring contexts.
Only the explicit `finish` measurement boundary waits on device completion.
`GpuDCRTPolyMatrix::is_ready` queries writer events without a host wait and is used
to verify timing completion before host materialization. Compact import/export
private streams are connected to the relevant producer boundaries; covariance
cache destruction joins its private-stream frees to the owner release stream.
The cache retains the execution owner independently of the raw context pointer.
These success-path joins preserve unrelated owners' overlap.

The estimator separately records device-event work and coordinated fleet wall
time, measures partial wave classes, and uses actual offsets where range position
changes computation. Outputs remain live until the fleet wall clock stops.
Structural cache identities include the nominal schedule and timing contract.
Shape-only reports explicitly identify `SyntheticFreshPlacement`; they do not
claim agreement with an admitted runtime invocation. Dataflow transfer charging
and documentation were updated with these metric meanings. Exact retained-input
plan identity and the existing transfer calibration extrapolation remain open.

Additional error cleanup retains contexts when matrix release ordering fails;
descriptor-product failures use asynchronous retirement instead of host stream
waits. Failed release-event queries poison the ledger persistently. Independent
GPU placement preparation and CPU CRT-residue conversion now use Rayon.

This checkpoint does not close complete invocation admission, isolated preflight,
ordinary zero-copy destination/range views, explicit redistribution, physical
multi-device validation or the final independent review. No push or commit is
part of this checkpoint.

The new decomposition regression exposed a pre-existing unsigned small-decomposition
bug: native code reused the first CRT residue's digits for every destination
tower. The CPU contract decomposes each tower's own unsigned residue. The native
all-tower kernel now reads each output tower's source descriptor, validates and
waits on all source towers, and preserves their reader lifetimes. Its balanced
decomposition branch retains the original arithmetic. Mixed 17-/54-bit towers in
both orders, negative residues, random residues, coefficient/evaluation inputs,
and pending readers are covered without changing existing expected values.

Evidence for this checkpoint is in
`test_data/fleet-sharding-implementation/parallel-schedules-timing-workspaces/`.
The CPU and GPU workspace unit builds were warning-free. CPU execution passed
606 tests with 25 ignored. All eleven GPU-feature unit binaries completed three
identical-command runs: 856 passed and 27 ignored per workspace run, with zero
failures. Crate source hashes remained unchanged. The initial unsigned
decomposition failures and their source manifest are preserved separately in
`before-small-decomposition-fix/`.

Twelve alternating baseline/current process-wall pairs passed for both unchanged
balanced decomposition and centered rebase tests. Median baseline/current times
were 0.208842/0.204095 seconds and 0.255540/0.254689 seconds respectively. These
include test setup, CPU references, transfers and teardown; they are not isolated
kernel or production-wave measurements. GPU execution used one RTX 4080 SUPER.
The next ordinary-view and transfer-boundary edits start from this validated
snapshot and require their own validation evidence.

## Ordinary input views and complete transfer boundaries

`GpuDCRTPolyMatrix::column_view` borrows the original owner and records a checked
column range without allocating another GPU payload. Native negate, scalar
multiplication and coefficient automorphism batches accept independent source
column ranges and retain each source's physical row pitch. Range metadata is
included in the shared native workspace query. The existing full-matrix batch
calls retain their previous metadata layout. Fleet unary dispatch selects a
containing resident shard and keeps its shared allocation alive through enqueue.

Evaluation-format scalar multiplication and negate consume those views directly.
Format conversion uses only the selected range: automorphism retains its existing
coefficient-domain arithmetic and a range-sized scratch matrix followed by NTT.
Other ordinary operations still materialize partial inputs; destination views and
complete invocation admission are not established by this input-view API.

Independent source review corrected empty automorphism format preservation,
rejected cross-input/duplicate output aliases in native view batches, and joined
every batch output's asynchronous allocation/descriptor initialization before the
common kernel stream can write it. These joins are stream-event dependencies,
not host waits, and the output initialization repair also applies to existing
full-matrix batches.

Transfer measurement now executes each declared full artifact/staging boundary
once per measured iteration. It neither shrinks the logical type nor multiplies
a small artifact's complete codec/store latency. Compact fixtures use the existing
canonical host encoder and configured fleet import. A conservative capacity
diagnostic rejects unsupported sizes but does not establish production admission.
Zero-shape artifacts report the existing unsupported runtime boundary explicitly.
Executor dispatch calibration runs even when no transfer was collected.

Validation evidence is in
`test_data/fleet-sharding-implementation/ordinary-views-transfer-boundaries/`.
CPU/GPU workspace unit builds were warning-free. A final test-only isolation
change was compiled with the targeted GPU primitives unit build, also without
warnings. The allocation-counter assertion runs in its own process because the
default pool's usage is shared by otherwise unrelated tests in the same process.

All eleven CPU unit binaries passed: 606 passed, 25 ignored. The changed GPU
primitive, runtime and estimator binaries each passed three identical-command
runs: 231, 170 and 59 passed respectively, with four runtime tests ignored and
zero failures. Source hashes were unchanged during validation. This is a narrow
three-crate GPU check; the preceding all-eleven-crate GPU run belongs to the
previous source snapshot and is recorded separately above.

Twelve alternating baseline/current pairs of the unchanged unary ownership test
passed. Median process wall times were 0.240571 and 0.246160 seconds; the candidate
was slower in seven of twelve pairs, with a median paired difference of 0.002117
seconds. The test includes setup, CPU verification, transfers and teardown and
uses complete stored input intervals. It does not measure isolated partial-view
speedup or satisfy production-wave performance acceptance. Native device-event
timing, physical multi-device checks and the remaining complete-plan gates still
require their own acceptance evidence. No commit or push was performed.

## Admission and explicit-preflight foundations

The next parallel implementation stage adds the native, ledger, and executor
interfaces needed to replace heuristic production admission. This remains an
incomplete integration checkpoint: the production fleet does not yet consume
these complete reservation plans.

`GpuPreparedMatrixStorage` consumes already allocated ordinary matrix owners and
prepares their completion resources at an explicit setup boundary. A thread-bound
dispatch claims its ordered slots before constructors run. Shape mismatches,
extra constructors, missing permits, duplicate slots and still-live outputs are
rejected. Returned owners retain the storage; recycling a slot joins its readers
and records reuse through device events. This path never grows its backing.
Compact allocations, arbitrary scratch, pinned buffers and fresh reader events
still require equivalent enforcement. Slot recycling must not retire the charge
for the underlying physical backing.

The native allocation-epoch API distinguishes explicit initial setup from
nonblocking refresh. Its receipts retain the execution owner and record activity,
context generation and allocator observations. External default-pool exclusivity
is an explicit operating assumption, not proof of complete native instrumentation.
Context construction is tracked before table allocation; release and reclaimer
activity participate in the revisions. Incomplete tracking returns an unverified
result. The coverage gate remains disabled until all relevant native allocation,
private-stream and event paths are instrumented. No production baseline can be
certified merely because several counters agree.

`GpuMemoryLedger::new` and `refresh` now require native receipts. Raw snapshots
are private accounting helpers used by lifecycle fixtures. Refresh rejects
pending leases before accepting evidence and rechecks the receipt immediately
before publishing its new baseline. A worker cannot turn a pending reservation
into a resident allocation and absorb it into an older snapshot. Failed or
unverified observations retain the existing conservative accounting.

`reserve_columns` accounts for hypothetical preparation and complete retained
outputs before class-valid scratch width selection. It supports capped fresh
assignment and inherited intervals, validates class tails and concrete output
bounds, and publishes all device reservations together. Every independently
released fixed/output/scratch owner receives its own lease. Inactive inherited
owners require no hypothetical preparation query. Native providers for the full
production families and the connection to prepared storage remain outstanding;
synthetic provider tests are not a physical-memory admission proof.

The existing calibration registry now identifies allocation classes and bound
identities, supports absent roles and declared allocation-free observations, and
checks nonlinear temporary requirements against the post-output budget. Unknown
bounds and raw pilot observations are explicitly diagnostic. The existing fleet
and estimator measurement paths temporarily use `candidate_widths`; those values
are not admitted capacities. Complete native bounds, production slot consumption,
and removal of the remaining manual-width path are still required.

The executor now issues ordered typed `GpuInvocation` requests after actual
materialization, including ordinary, fused and batch operations. Fresh sampling
requests exclude production keys, tags, seeds, loaders and transcript state;
replay imports are described independently. A shared primitive RNS staging parser
validates layout metadata without loading the payload, and the fleet uses it for
preimage source descriptions. The fleet preflight execution/plan-consumption hook
still needs its implementation; the current default hook is a no-op. External
immutable materialization APIs also require migration. Existing in-operation
pilots have not yet been replaced by isolated preflight pilots. Failed runtime
calibration attempts no longer permanently blacklist an operation signature.

Independent source review corrected per-interval class validation, baseline
refresh ordering, inactive preparation and aggregate lease lifetime issues before
validation. Additional GPU fixtures cover prepared slots across host threads,
pending readers, backing ownership after storage-handle drop, and failed
mixed-context setup cleanup. This stage adds no private CUDA pool, CPU fallback,
integration-test execution, commit or push.

Validation for this checkpoint is in
`test_data/fleet-sharding-implementation/admission-preflight-foundations/`.
Both workspace unit builds were warning-free. All eleven CPU unit binaries
passed once (606 passed, 25 ignored), and all eleven GPU unit binaries passed
three identical-command runs (885 passed and 27 ignored per complete pass).
There were zero failures, and the recorded source hashes were unchanged during
GPU validation. Subsequent activity-coverage and import-boundary edits require
new validation and are not covered by this snapshot.

Twelve alternating baseline/current pairs of the unchanged unary ownership test
passed. Median process wall times were 0.248873 and 0.249302 seconds; the candidate
was slower in six of twelve pairs, with a median paired difference of 0.000335
seconds. These timings include context setup, CPU references, transfers and
teardown. They do not establish isolated kernel performance or production-wave
acceptance. Physical multi-device and final synchronization gates remain open.

## Native activity coverage and mutable import boundaries

This follow-up checkpoint adds outer allocation-activity scopes to mutating
ordinary, batched, transform, CRT, sampling, serialization, compact and trapdoor
entrypoints. Delegated wrappers inherit their callee's complete scope. Prepared
dispatch tracking now spans reservation through completion or cancellation and
backing teardown. Cache and event-set destruction retain execution ownership
until the activity guard is destroyed. Thread-local event replacement and pinned
reclamation track the resource's recorded device, including cross-thread release.

Cleanup failures invalidate allocation evidence. Compact device-only error paths
join submitted readers to execution-owner streams instead of waiting on the host;
D2H observation boundaries retain their required completion waits. Raw activity
revisions may be inspected even when observation is unverified, but these fields
cannot construct a ledger receipt. The certification gate remains disabled: this
source-coverage work does not enforce every prepared resource request or approve
the proposed accounting amendment.

The four importing Backend methods and public artifact decode/materialization
boundaries now receive mutable backend access. Public decode performs Import
preflight once before allocation; internal decode delegates to that boundary.
Host-matrix materialization and hashing use the same preflight order. Workspace
callers were migrated and fleet replication retains Rayon parallelism. Probe
regressions cover ordering, rejection, repeated materialization and sampler replay.
The fleet's actual admission runner remains outstanding; the default hook is still
a no-op until that integration is implemented.

Independent source review found and corrected a guard/owner destruction-order bug,
pinned-device attribution and missed cleanup invalidation. A diagnostic-only
readiness assertion drains the test owner's reclaimer before checking unchanged
revisions, so asynchronous descriptor cleanup cannot race that assertion.

Validation evidence is in
`test_data/fleet-sharding-implementation/activity-import-boundaries/`.
Both workspace unit builds were warning-free. All eleven CPU unit binaries passed
once (606 passed, 25 ignored). All eleven GPU unit binaries passed three
identical-command runs (889 passed, 27 ignored per complete pass), with zero
failures. Source hashes remained unchanged during GPU validation. Formatting and
`git diff --check` passed. The initial GPU build's test-only type mismatch is
preserved in `gpu-build-initial.log`; the final build and runtime validation use
the corrected source.

Twelve alternating baseline/current pairs of the unchanged unary ownership test
passed. Median whole-process wall times were 0.244596 and 0.240352 seconds. The
candidate was slower in six of twelve pairs; the median paired difference was
0.000189 seconds. This includes context setup, CPU references, transfers and
teardown, so it establishes neither kernel speedup nor production-wave acceptance.
Physical multi-device checks, final synchronization repetitions and the remaining
complete-plan acceptance gates have not been run for this source snapshot.

The prepared-storage accounting amendment was subsequently accepted and is
integrated into the authoritative plan. It distinguishes physical residency from
allocator-adjusted live residency and requires a complete finite resource
configuration, including worker and kernel first-use state, before sealing.
Its implementation is in progress and is not covered by the checkpoint above.
Source instrumentation, passing units and sampled counter stability do not by
themselves grant that guarantee. No commit or push was performed.

## Prepared reservations and worker-local preimage targets

The accepted accounting amendment now has separate calibration metric identities
for default-pool increments, prepared occupied spans and allocation-free classes.
Prepared observations require an independent bound and a stable storage
configuration identity. Their slope uses typed available logical capacity;
the existing scalar physical admission path rejects them until native span and
resource fitting is connected. The estimator's current pool observations remain
explicitly diagnostic.

The memory ledger now checks both physical residency `P` and allocator-adjusted
residency `E`, using the same native epoch receipt. Reservations fit both budgets.
Cancellation before submission returns both charges, while completed asynchronous
frees retain physical growth until a coherent refresh: freed pool pages may still
be physically reserved. This is the existing complete-bound allocation path,
not prepared logical admission. Native epoch certification remains disabled.

Prepared ordinary matrix storage exposes stable native storage, backing-group and
slot identities. Reservation is transferable across host threads; activation is
thread-bound. Failed partial reservations release only their own slots, and the
first successful reservation permanently enables ordinary allocation checks.
Existing output-reader and backing-owner lifetimes survive cancellation and
cross-thread handoff. Other resource families remain to be integrated.

Ordinary unary, scalar and binary views now accept explicit destination rectangles
and retain each owner's real row pitch. Disjoint rectangles may share a batch
destination; overlapping writes and input aliases are rejected. Automorphism
rectangles use coefficient destinations, allowing one transform after assembly.
These primitive destinations are not yet wired into fleet output admission.

Preimage sources now describe retained resident ranges or host staging bytes.
Staging records the exact mathematical CRT identity without a GPU context; each
sampler worker materializes only its actual inner tile with its own parameters.
Absolute storage offsets remain separate from global randomness offsets. Resident
transport and placement shortcuts now distinguish execution owners even on the
same physical GPU. CPU preparation retains Rayon parallelism.

Validation is recorded in
`test_data/fleet-sharding-implementation/prepared-reservations-and-worker-targets/`.
Both workspace unit builds were warning-free. The CPU suite passed all eleven
unit binaries once (608 passed, 25 ignored). GPU evidence covers all eleven
binaries three times (906 passed, 27 ignored per pass). The final runtime suite
passed three identical-command repetitions. Other passing evidence was reused
after verifying that the final source delta changes only one runtime test body;
`test-only-delta.patch` and archived source hashes establish that scope. Earlier
reuse required exact binary hash equality. The manifest records both kinds of
provenance. Source hashes stayed unchanged during the final runtime validation.

The initial GPU build failures were a test-only owner-ID type mismatch and a
mutable-borrow error in replica preparation; both logs are retained. Initial
runtime tests exposed an obsolete physical-capacity expectation and an invalid
global-quiescence requirement in the worker-liveness fixture. The latter now
holds an independent execution owner throughout its manual-width operation. A
later staging-pool measurement was contaminated by concurrent GPU-backed CPU
trapdoor tests. Its complete original assertions now run in a separate CUDA
process, preserving concurrent tests and the same scratch bound.

Twelve alternating baseline/current pairs of the unchanged unary ownership test
passed. Median whole-process wall times were 0.240293 and 0.243251 seconds; the
candidate was slower in seven of twelve pairs, with a median paired difference
of 0.000428 seconds. Setup, CPU references, transfers and teardown are included;
this is neither kernel timing nor production-wave acceptance. Tests covered the
single detected RTX 4080 SUPER. Physical multi-device checks, final 300-repeat
synchronization checks and complete-plan acceptance remain outstanding.

## Native prepared matrix occupancy

Prepared ordinary storage now separates unconsumed reservation bytes from actual
occupied backing-group bytes and their high-water mark. Claims charge before
device submission. Event-ordered reuse of the same backing does not charge it
twice; cancelling unused slots does not release a live output's occupancy.
Nonblocking event queries retire completed logical releases without changing
physical backing ownership. Query fields are diagnostic during concurrent work,
not a substitute for native slot fitting.

An explicit high-water reset rejects outstanding reservation/dispatch tokens and
pending releases. Fixed baseline owners must remain retained throughout a pilot.
There is no new production host wait or mutex. This covers ordinary matrix groups
only; compact storage, additional scratch, pinned resources, worker/launch state,
complete native fitting and production pilot integration remain outstanding.
Allocation epoch certification remains disabled.

Evidence is in `test_data/fleet-sharding-implementation/prepared-occupancy/`.
Both workspace unit builds were warning-free. All nine prepared-storage GPU tests
passed three identical-command runs with zero failures (238 primitive tests were
filtered out per run). This is targeted validation, not a new full-suite result.
The pool-measurement test runs in an isolated process and verifies nonzero logical
pilot pressure with no increase in default-pool usage or reservation. Cross-worker
reader reuse, reservation cancellation, retained outputs and reset boundaries
also passed. Source hashes remained unchanged during validation.

Twelve alternating baseline/current pairs of the unchanged prepared reader-reuse
test passed. Median whole-process times were 0.244098 and 0.247063 seconds; the
candidate was slower in seven pairs, with a median paired difference of 0.005259
seconds. These include setup, transfers, reference checks and teardown; they do
not resolve kernel/dispatch overhead or establish production-wave performance.
Final matched performance and physical multi-device gates remain open.

## Typed device workspaces and Rayon child reservations

`GpuPreparedStorage` now owns ordinary matrices followed by typed device workspace
slots. `GpuPreparedSlotIdentity` records kind, alignment and backing capacity;
matrix levels are absent for workspace slots. Batch arithmetic metadata and
separate CRT/decomposition metadata now use `GpuDeviceWorkspace`. A prepared
dispatch checks kind, capacity, alignment and execution owner before obtaining
the next slot. Standalone operations retain the default asynchronous allocator.
Borrowed output auxiliary storage remains part of its matrix backing charge.
Two unused auxiliary allocation helpers were removed.

Workspace reuse records the preceding consumer and joins the execution owner's
release stream, preserving backing lifetime after the setup handle is dropped.
Actual workspace claims participate in logical occupancy; reservations alone do
not. Pending same-span reuse counts its backing once. Reuse failures retain the
backing and invalidate allocation evidence. No production host wait or new mutex
was added.

An unactivated `GpuMatrixReservation` can transfer its complete ordered claim list
into disjoint child reservations for existing Rayon jobs. All fallible host
preparation precedes ownership transfer, with no release/re-reserve window.
Cancelling one child returns only its unused slots; other children and returned
GPU owners retain their own resources. Native and Rust checks reject incomplete
or overflowing partitions. This does not certify finite worker/launch state.

Evidence is in `test_data/fleet-sharding-implementation/prepared-workspaces/`.
Workspace integration passed all 249 GPU primitive unit tests three times with
zero failures. After adding child partitioning, all 12 prepared-storage tests
passed three identical-command runs (238 other primitive tests filtered out).
Both source snapshots have warning-free CPU/GPU workspace unit builds and stable
source hashes during validation. The `workspace-` logs and manifest identify the
earlier full-primitive snapshot; the unprefixed files identify the child tests.
Existing assertions were preserved apart from API type/signature migration.

Twelve alternating baseline/current pairs of the unchanged batch workspace test
passed before child partitioning. Median process times were 0.270189 and 0.269310
seconds; the candidate was slower in five pairs, with a median paired difference
of -0.000469 seconds. These include setup, transfers, references and teardown,
not isolated kernel or production-wave timing.

Pinned buffers, compact/sampler scratch and caches, event/worker/launch resources,
complete native layout fitting, physical sealing, production preflight/pilots and
estimator integration remain incomplete. Allocation epoch certification remains
disabled; these tests do not establish a complete invocation memory guarantee.

## Variable prepared shapes and exact native requests

Prepared requests now bind native storage/slot identity, matrix dimensions and
format, or workspace kind, exact bytes and alignment. A nonblocking native fit
query checks layouts and current slot availability; reservation repeats those
checks and atomically claims the complete list. Rayon child reservations retain
the same frozen requests. Wrong actual allocation shapes, formats and workspace
sizes fail without consuming an unrelated slot.

A matrix slot can expose smaller rows/columns within its prepared dimensions.
The payload transfer length and borrowed auxiliary capacity follow the queried
logical shape, while backing pointers, per-polynomial descriptors and original
streams remain fixed. Pending readers keep their reuse dependencies. Logical
occupancy charges used payload/auxiliary prefixes and requested workspace spans;
same-backing reuse retains the largest unretired use without double counting.
Physical backing remains fully retained and is not inferred from these counters.

Evidence is in `test_data/fleet-sharding-implementation/prepared-shapes-and-fit/`.
Both workspace unit builds passed without warnings. All 253 GPU primitive unit
tests passed three identical-command runs with zero failures or ignored tests.
Source hashes remained unchanged. New coverage checks mixed-width CRT layouts,
changing source dimensions with retained readers, logical-length peer copies and
staging round trips, atomic rollback, stale identities and exact workspace use
when a smaller output can no longer lend its oversized auxiliary capacity.
The first three runs each had 252 passes and one test-setup failure: a second
execution context was created before a single-context pool reset. Its creation
now occurs after the measurement interval; all original assertions remain.
Those failed runs and their source manifest are retained under `initial/`.

Twelve alternating baseline/current pairs of the unchanged batch-workspace test
passed. Median whole-process times were 0.266522 and 0.270795 seconds, including
setup, references, transfers and teardown. This is not kernel or production-wave
acceptance. Full invocation resource coverage, physical sealing, native fitting
in fleet admission, isolated production pilots, estimator integration and the
final multi-device/synchronization/performance/review gates remain incomplete.

## Width selection with native prepared-resource fitting

The existing calibration width search now accepts typed candidate capacity and
typed temporary requirements. Default-pool classes retain independently bounded
physical-byte checks. Prepared classes use logical available capacity for their
slope hint and query exact native requests on every participating storage owner.
They cannot be admitted through a scalar zero-byte callback. Metric/configuration
mismatches, duplicate storage groups, changed native owners and unavailable
minimum requests are rejected. Every active owner is queried independently, and
the final shared nonzero-role width is rechecked on all its owners. Context fit
queries retain Rayon parallelism and introduce no GPU allocation or host wait.

Post-output ledger charges must still fit the physical and adjusted budgets;
logical reuse adds no second backing charge. Width selection remains advisory:
its caller must retain all fixed/output reservations, atomically acquire the
selected complete requests and supply the accepted finite-resource seal before
production dispatch. The existing scalar ledger provider still supplies bounded
byte requirements only. Native prepared transactions, complete retained-output
planning and production invocation integration remain to be connected.

Evidence is in `test_data/fleet-sharding-implementation/prepared-width-fit/`.
Both workspace unit builds passed without warnings, and the complete GPU runtime
unit binary passed three identical-command runs: 200 passed, 4 ignored, zero
failures each time. Source hashes stayed unchanged. The initial build failed on
a test-only import path; its logs and manifest are retained under `initial/`.
Existing scalar width-search assertions were preserved through the typed API
migration. `source-delta.patch` records the two-file change from the preceding
validated snapshot.

New tests combine real native slot fitting with synthetic ledger/width models.
They cover fully charged physical backing with reusable logical slots, logical
exhaustion despite physical headroom, full-output reservations retained during
width search, workspace-limited width reduction, metric mismatch, competing
reservations, multiple contexts per owner, and separate owner capacity under a
shared nonzero-role profile. A private one-column matrix pilot is also measured
against its independently queried native logical layout. Three execution owners
on one physical GPU are not physical three-GPU evidence or a resource seal.
Final physical multi-device tests, synchronization repetitions, production-wave
performance, estimator correspondence and independent review remain outstanding.

## Atomic native groups in the physical ledger transaction

The existing ledger reservation method now accepts exact native allocation
groups alongside separately bounded physical requirements. Setup binds and
retains the accepted storage inventory under each device/execution identity,
revalidating native epoch receipts after registration. Later backing on the same
execution owner is not implicitly accepted. Unknown storage, foreign execution
owners and invalid device assignments fail before logical acquisition.

Independent native groups are acquired with Rayon before committing physical
IDs or charges. A failure drops every successful native token, including when
two groups conflict on one storage. Prepared tokens retain independent lifetimes
and can move to existing GPU enqueue workers and Rayon child jobs. Reusing
registered backing does not add a second physical charge. Physical IDs retain
the existing explicit submit/cancel lifecycle.

Evidence is in `test_data/fleet-sharding-implementation/prepared-fleet-reservations/`.
CPU/GPU workspace unit builds passed without warnings. The complete GPU runtime
unit binary passed three identical-command runs: 202 passed, 4 ignored and zero
failures each time. Source hashes stayed unchanged. Tests include guaranteed
duplicate-group conflict/rollback, an independently held competing reservation,
unregistered backing, wrong placement, physical capacity rejection, and actual
worker dispatch followed by retained-reader verification after dispatcher and
inventory handles are dropped. Numeric ledger fixtures are explicitly synthetic;
they do not fabricate public epoch evidence or establish a resource seal.

Twelve alternating baseline/current pairs of the unchanged native lease-command
test passed. Median whole-process times were 0.235261 and 0.231387 seconds; the
candidate was slower in four pairs, with median paired difference -0.004990
seconds. Setup, transfers, references and teardown are included; production-wave
and kernel performance remain unverified. The detected hardware was one GPU.

Native groups remain one-shot dispatch reservations. Complete output planning,
prepared scratch-width selection/publication, and preservation of the scratch
footprint across all future waves remain to be integrated. Pinned/compact/cache/
worker/launch coverage and physical sealing are still incomplete, and native
allocation epoch certification remains disabled.

## Complete prepared output planning before scratch width selection

The column ledger now accepts concrete native requests and separately bounded
physical demand from the same requirement provider. Capacity search includes
fixed operands, minimum scratch and a monotone output layout envelope. Concrete
output layouts must fit that envelope. The planner retains every active owner's
fixed operands and complete outputs before selecting scratch widths from the
remaining native capacity and both physical budgets. Final scratch groups are
acquired atomically; late range or acquisition errors release prior reservations.
Native availability includes pending event-ordered device reuse and remains an
advisory hint until the exact reservation succeeds. It does not establish pinned
host-buffer readiness or complete physical sealing.

Evidence is in `test_data/fleet-sharding-implementation/prepared-column-plans/`.
Both workspace unit builds passed without warnings. All 253 GPU primitive tests
and all 204 non-ignored GPU runtime tests passed three identical-command runs
per binary, with four runtime tests ignored and no failures. Source hashes stayed
unchanged. Tests cover complete-output pressure reducing scratch width, fully
charged physical backing, separately bounded remaining demand, oversized output
rejection, concrete-envelope mismatch, and cancellation after late failures.
The numeric ledger fixtures remain synthetic and do not certify production.

Twelve alternating unchanged lease-command process comparisons passed. Median
baseline/candidate times were 0.269124/0.268087 seconds, including setup, transfers,
references and teardown; these are not production-wave or kernel acceptance.
Physical multi-GPU coverage, full resource sealing, production preflight/pilots,
estimator integration and final acceptance remain outstanding.

## Retained native reservations across successive waves

Native slots now track invocation ownership separately from their current matrix
or workspace lease. Completing a dispatch returns its existing reservation.
Retaining that token keeps idle slots unavailable to competing invocations;
dropping it releases idle capacity while live outputs keep their existing reader
and release dependencies. No release/re-reserve window is introduced.

`GpuMatrixReservation::rearm` prepares another wave in the same ordered slots,
within the original admitted shape/format/workspace envelope. Live prior owners,
changed slots and larger requests fail without surrendering the footprint.
Partial reacquisition rolls back while preserving invocation ownership. Pending
GPU readers remain ordered by the existing reuse events; no event wait or record
was removed or added. Child reservations transfer exclusive ownership directly
to existing Rayon workers and retain their independent admitted bounds.
Reserved-byte counters still represent unconsumed claims, not idle retained
footprints; native availability excludes both. Logical occupancy remains tied to
actual claims and pending readers.

Evidence is in `test_data/fleet-sharding-implementation/prepared-wave-reservations/`.
Both workspace unit builds passed without warnings. Under the latest instruction
for changes unlikely to cause GPU synchronization failures, each GPU unit binary
ran once: primitives passed 256 tests; runtime passed 204 tests with four ignored.
There were no failures, and source hashes stayed unchanged. New tests cover
Rayon child reuse with all prior wave readers retained, workspace-only reuse,
original-envelope enforcement, partial rollback and cancellation with live
outputs. The initial new reader fixture requested more identity columns than its
full matrix size; its dimensions were corrected without removing assertions.
That failed run (255 primitive passes, one failure, and 204 runtime passes) and
earlier test-import/API-warning build logs remain in separate `initial*` folders.

One matched unchanged reader-reuse process comparison took 0.195150 seconds for
the baseline and 0.218500 seconds for the candidate. It includes setup, transfers,
references and teardown; one pair cannot establish a performance regression or
speedup. Matched production performance remains an open final gate.

This supplies native reusable reservations, not a complete production range
runner. Production must still retain and rearm the admitted scratch footprint,
publish each complete output destination and use the same frozen recipe for
isolated calibration and execution. Pinned/compact/sampler/cache resources,
finite worker/launch coverage, physical sealing, estimator integration and final
physical multi-device/synchronization/unit/review acceptance remain incomplete.
Allocation epoch certification is still disabled.

## Composed dispatch over separately reserved stores

`GpuMatrixReservation::enter` now accepts following reservations in exact native
allocation order. One thread-bound dispatch can consume separate source, final
output and workspace stores, including related parameter contexts on the same
device/execution owner. Every reservation keeps its identity, request envelope
and exclusive footprint. Host preparation and full validation precede ownership
transfer; nested activation, duplicate native tokens, foreign execution owners
and consumed requests fail before publication. Actual allocation still checks
the next context and shape rather than searching for a convenient slot.

Finishing returns the original ordered reservation list. Callers can retain and
rearm scratch while final output owners keep their normal independent lifetime.
Incomplete dispatch cancels unused claims across every supplied reservation and
retains already published owners. No GPU event or synchronization edge changed.

Evidence is in `test_data/fleet-sharding-implementation/prepared-composed-dispatch/`.
CPU/GPU workspace unit builds were warning-free. One complete GPU primitive run
passed 258 tests; the GPU runtime run passed 204 with four ignored. There were no
failures and source hashes remained unchanged. New tests compose source upload,
modulus conversion, separate transform workspace and retained transpose outputs
over successive waves, while testing wrong-next-context rejection, foreign owners,
duplicate raw tokens and partial-command cleanup. All outputs are checked against
trusted CPU primitives after setup handles and scratch tokens are dropped.

One unchanged reader-reuse process pair took 0.198256/0.260259 seconds for the
baseline/candidate. This includes setup and teardown and is insufficient to
attribute a regression or establish production performance. Final matched
production measurements remain required. Full resource sealing, production
preflight/range-runner integration, estimator correspondence and the final
multi-device/synchronization/unit/review gates remain open.

## Reusing owned peer-copy completion events

`gpu_matrix_copy_peer` no longer creates and destroys a private completion event
for every transfer. It records the destination's owned writer completion and
uses that captured record for source release and producer-stream dependencies.
Prepared matrices already create those writer events during setup. Standalone
matrices retain the normal lazy writer-event initialization path.

Before a whole-buffer copy, all distinct destination producer queues are joined
on the copying stream, including their previously queued read-only consumers.
After enqueueing the copy, source producer queues as well as source release join
its completion. Consequently an immediate in-place source write cannot overtake
the pending transfer. Source writer metadata stays unchanged so independent
read-only consumers still depend only on its actual writer. Event waits remain
asynchronous; failures after a submitted copy quarantine both owners as before.

Evidence is in `test_data/fleet-sharding-implementation/prepared-peer-completions/`.
Both CPU/GPU workspace unit builds passed without warnings. Complete GPU primitive
and runtime runs passed 259 and 204 tests respectively, with four runtime tests
ignored. The two peer-filter tests passed three identical-command repetitions
because this change affects synchronization. No command failed; tested source
hashes stayed unchanged. New coverage retains old destination readers, copies
through independent contexts, immediately overwrites each source on its own
producer streams, and checks chained results against trusted CPU matrices.
The hardware was one RTX 4080 SUPER; these checks do not prove cross-device P2P.

One unchanged reused-writer/peer-destination process comparison took
0.239929/0.247693 seconds for baseline/candidate, including setup, transfers,
reference checks and teardown. This is diagnostic, not kernel or production-wave
performance acceptance. Peer-route setup, other event/worker/launch resources,
pinned/compact/sampler/cache coverage, production planning and estimator integration
remain incomplete. Allocation epoch certification remains disabled. Final physical
multi-device, synchronization, all-unit and independent-review gates remain open.


## Prepared managed pinned transfer storage

Native prepared storage now includes typed pinned host slots, with independent
capacity, reservation, occupancy, high-water and CPU-ready counters. They do not
contribute host bytes to device logical occupancy or VRAM capacity hints. Once an
execution owner reserves a pinned claim, managed pinned allocation requires the
next exact context, size and alignment in its active dispatch. Unplanned claims
fail before backing growth. Ordinary-only inventories do not yet certify this
resource domain.

CPU-held buffers keep their slot exclusive. Uploads transfer ownership to the
existing completion reclaimer, which returns a slot only after DMA completes;
a queued device wait alone cannot authorize host writes. Download snapshots keep
their pinned owner after transfer completion until the snapshot is dropped.
Reusable invocation tokens retain their original envelope across host reuse.
Pinned lease state is independent of the matrix/context inventory, avoiding a
last-owner cycle that could destroy the reclaimer on its own worker. Rust pinned
buffers retain their allocation parameters for later resize/clone requests;
transferring a pointer releases that Rust reference normally.

Evidence is in `test_data/fleet-sharding-implementation/prepared-pinned-storage/`.
CPU/GPU workspace unit builds were warning-free. Complete GPU primitive and runtime
runs passed 262 and 204 tests respectively, with four runtime tests ignored.
Three new pinned tests passed three identical-command runs, with zero failures:
CPU-held exclusion and exact claims, H2D/D2H round-trip/reuse, and immediate final
owner teardown. Source hashes remained unchanged. The hardware remains one
RTX 4080 SUPER. One unchanged reader process pair took 0.253234/0.246223 seconds
for baseline/candidate; this includes setup and teardown and is not production
performance acceptance.

Raw CUDA pinned allocation sites, arbitrary host transfer pointers, private
completion events and the remaining compact/sampler/cache resources are not
covered by this checkpoint. Full allocation certification remains disabled.
Production planning, physical sealing, estimator integration and final acceptance
remain incomplete.


## Native pinned allocation coverage

All explicit CUDA pinned allocation/free sites now pass through the existing
runtime pool and prepared pinned claims. This covers transform uploads, compact
payload loading, hard-cutoff metadata and acceptance buffers, and batched compact
serialization. Failed upload or download completion retains host ownership in
the existing reclaimer instead of immediately freeing a possibly active DMA
buffer. Successful synchronous stores return CPU-ready pinned slots directly.
Pinned free errors are reported to Rust and native cleanup callers.

Evidence is in `test_data/fleet-sharding-implementation/native-pinned-storage/`.
CPU/GPU workspace unit builds were warning-free. GPU primitives passed 264 tests;
runtime passed 204 with four ignored. The pinned filter (six tests) and hard-cutoff
filter (five tests) each passed three identical-command runs without failures.
New tests exercise prepared native transform uploads with retained readers and
compact upload/serialization against trusted CPU coefficients/scalar serialization.
Source hashes stayed unchanged. One unchanged compact batch process pair took
0.169644/0.137951 seconds for baseline/candidate; it is not production performance
acceptance. These are still single-device checks.

This centralizes explicit pinned backing allocation. Arbitrary borrowed host DMA
pointers, private completion resources and production preflight remain separate
coverage obligations; allocation certification is still disabled. The following
resource-arena edits are not covered by this evidence snapshot.


## Compact and Gaussian gadget device spans

Compact payloads, hard-cutoff metadata/decision storage, private acceptance
staging and narrow/wide RHS expansion now use typed prepared device spans.
Persistent compact owners release their spans on the final stream that has
joined their existing writer/readers. Column views retain their borrowed payload
semantics. Gaussian gadget sampling consumes five typed sampler spans; its public
native layout query supplies the same exact request sizes used by dispatch.
Unplanned/type/size-mismatched device claims fail before allocating backing.

Evidence is in `test_data/fleet-sharding-implementation/remaining-resource-arenas/`.
To reduce iteration latency, this stage built only `mxx-primitives` GPU unit tests
(warning-free) and ran that binary once: 266 passed, none failed/ignored.
Prepared sampler, prepared compact expansion, hard-cutoff and native compact
transfer filters each passed three identical-command runs (eight tests per pass
across the four filters). New prepared reuse tests retain earlier results without
an intervening host fence and check gadget identities against trusted CPU
primitives. The first build failed solely on missing `<array>` includes; that log
and source manifest are retained. Full workspace builds/execution after these
edits remain pending for final validation.

Whole-process baseline/candidate checks took 0.304396/0.275109 seconds for the
unchanged Gaussian gadget relation and 0.246982/0.251266 seconds for compact view
multiplication. These include setup/teardown and use workspace versus isolated
crate builds, so they are diagnostic rather than matched production acceptance.
The P1 sampling/cache/serialization scratch, completion/worker/launch resources,
production planning and final acceptance gates remain incomplete. Allocation
certification remains disabled.


## P1 sample and scratch reservations

Cached and uncached P1 sampling now consume typed sampled-integer and optional
large-dimension scratch spans. The public native query and actual dispatch share
the exact column-range requirements. Device copies retain their own workspace
owners through the existing scatter-consumer dependencies. Allocation failure
no longer halves the in-operation sample tile: the admitted column range fixes
scratch demand, and an insufficient claim fails without changing that range.
Covariance-cache and private completion-resource preparation remain separate.

Evidence is in `test_data/fleet-sharding-implementation/prepared-p1-storage/`.
The isolated primitive GPU unit build was warning-free; all 267 tests passed once.
The prepared P1 reuse test and the 15-test GPU-preimage filter each passed three
identical-command runs with no failures. The new test covers cached/uncached
sampling, both small and large kernel paths, and scratch reuse while retaining
previous outputs. Expected samples replay freshly random seeds through the
existing standalone path. Hashes stayed unchanged. Full workspace, actual
multi-device, final synchronization/performance and independent-review acceptance
remain pending; complete production admission is not enabled.


## Prepared compact completion reuse and cache setup boundary

Prepared compact payloads and hard-cutoff decision buffers reuse their slot's
setup-created event for writer/decision completion. Their final release records
the same owned event after joining consumers; borrowed events are never destroyed
by a compact view or transient owner. Standalone owners retain their own events.
Device workspace retirement now holds the execution owner directly rather than
dereferencing a potentially expired context. New P1 covariance caches are rejected
after prepared admission starts, keeping their private stream/backing creation at
setup; existing prepared caches remain usable by P1 dispatch.

Evidence is in `test_data/fleet-sharding-implementation/prepared-completion-reuse/`.
The isolated GPU primitive build was warning-free. All 268 unit tests passed once.
Prepared compact (two tests), prepared P1 (one test) and hard-cutoff (five tests)
filters each passed three identical-command runs with no failures. The new compact
case clones into a prepared payload, enqueues a product, drops/reuses that payload
without a host fence, retains all products, and checks them against CPU primitives.
Source hashes remained unchanged. Full workspace builds/runs, physical multi-GPU,
final synchronization/performance and independent review remain pending. Other
serialization/transfer/worker/launch resources and production planning are still
incomplete; allocation epoch certification remains disabled.

## Typed preflight for samplers and unary view kernels

The fleet backend now implements the executor's typed preflight hook for uniform,
Gaussian and hash sampling, preimage sampling, negation, integer scaling and ring
automorphisms. Preflight and production share the same column runner functions.
The preflight hash request contains only tag length, and its runner uses an
independent key. The preimage pilot constructs a one-column zero staging payload
with the validated production level, format and polynomial width. Its loader and
seed are independent of the production target and randomness. Each representative
loads that payload on its assigned execution owner. Pilot outputs are dropped
before production begins.

Evidence: `test_data/fleet-sharding-implementation/runtime-preflight/`. The targeted
runtime/primitives GPU build was warning-free after fixing two Rust type errors
(the initial log is preserved). Runtime passed 206 unit tests with four ignored;
primitives passed 269 with none ignored. Additional targeted tests passed once,
including exact preimage seed replay against the primitive sampler and the
public-matrix product relation. Source hashes stayed unchanged. This stage changes
Rust orchestration without changing native synchronization; repetitions follow the
user's one-pass instruction. Whole-workspace final gates remain pending.

This does not complete admission: profiles still use the existing default-pool
observation, several other invocation classes still use their existing pilot
paths, and native allocation certification remains disabled. The next arithmetic
runner edits are outside this evidence snapshot.

## Arithmetic preflight and shared compact block pilots

Typed preflight now also invokes the shared range kernels for range constants,
add/subtract/multiply/accumulate, transpose, slice, tensor and fused tensor row
sums, concatenation, ordinary modulus/CRT transforms, row-block addition,
gadget decomposition, CRT recomposition and compact RHS multiplication. Compact
row-block calibration uses the actual block-descriptor primitive; the former
concatenate/multiply/slice pilot path has been removed. Fixed replica preparation
and scoped pilot cleanup remain shared with production.

Evidence: `test_data/fleet-sharding-implementation/runtime-preflight-arithmetic/`.
The scoped GPU runtime build was warning-free after repairing closure type
inference and two test-expression warnings. Three preflight tests passed once,
then all runtime unit tests passed: 207 passed, four ignored, no failures. The new
arithmetic test verifies outputs through existing primitive operations and
confirms calibration finishes before production. Its compact block check compares
the individual outputs with slices of the trusted full product. The source
manifest stayed unchanged. One unchanged whole-process test pair took
0.258301/0.224513 seconds for baseline/candidate; builds have different feature
unification, so this is diagnostic and not matched production acceptance.

Import/setup and allocation-free classification, frozen admitted production
plans, complete prepared storage/resource coverage and estimator consumption
remain required. These pilot profiles still do not certify physical memory.

## Prepared RNS transfer spans

Batched RNS H2D packing and D2H unpacking now use a typed transfer workspace
instead of raw asynchronous allocations. A checked native layout query exposes
the exact single-device span. Transfer admission is an explicit resource domain,
like pinned admission: its first successful reservation makes subsequent RNS
transfers require exact claims. Arithmetic-only inventories do not certify that
domain; full allocation certification remains disabled.

Evidence: `test_data/fleet-sharding-implementation/prepared-rns-transfer/`.
The scoped primitives/runtime GPU build was warning-free. The new workspace
reuse test passed three identical-command runs. It retains both upload results,
queues all downloads before waiting for any, and compares their complete native
staging encodings. All primitives/runtime unit tests then passed once: 270/207
passed respectively, with zero/four ignored. Source hashes stayed unchanged.
Compact serialization scratch, private streams/events, arbitrary borrowed DMA
buffers, complete production admission and final acceptance gates remain open.

## Prepared compact transfer arena

Scalar compact store/load and homogeneous batch store now use one checked device
arena per operation. All raw asynchronous allocations in the serialization
sources have been removed. Store payload capacity is bounded by modulus widths;
load capacity uses the validated artifact width. Native queries and execution
use the same arena layout. Private streams, events and host DMA resources remain
separate outstanding coverage; this stage does not enable full certification.

Evidence: `test_data/fleet-sharding-implementation/prepared-compact-transfer/`.
The scoped GPU build was warning-free. The new bounded-span test passed three
identical-command runs, and primitives passed all 271 unit tests. Runtime passed
207 with four ignored on the same native revision before a test-only fixture
repair. Initial failures are preserved under `initial-missing-clone-claim/`:
the new fixture omitted the ordinary matrix clone performed by scalar storage;
its reservation now includes that real allocation. Existing tests were unchanged.
Source hashes stayed stable during each validation run.

An unchanged serialization test initially took 0.198849/0.275554 seconds for
baseline/candidate. Three further alternating process pairs all passed and had
medians 0.202765/0.142861 seconds. The initial slowdown did not persist; process
startup timings do not establish a production or kernel speedup.

## Prepared equality result and workspace unit validation

Matrix equality now derives polynomial addresses from native bases and strides,
removing host pointer tables and their two device allocations. Each active
physical partition consumes one four-byte transfer span for all of its CRT
limbs. Its existing boolean readback completes before reuse on the next limb
stream; retirement follows the last submitted stream, including early inequality.
No new host wait or device-wide synchronization was added. The native layout
query supplies the same result span used by execution.

Evidence: `test_data/fleet-sharding-implementation/prepared-equality/`. Both CPU
and GPU workspace unit builds were warning-free. All GPU-enabled workspace unit
tests passed once: 942 passed, 27 ignored. CPU-enabled workspace unit totals were
608 passed, 25 ignored, with no failures. The exact prepared equality test
passed three consecutive runs. The first abbreviated test path matched zero
tests; those logs remain labeled `equality-0/1/2`, and the actual three executions
are `equality-actual-0/1/2` with verified one-test summaries. Source hashes remained
unchanged through every run; no integration tests were run.

An unchanged equality/compact-zero round-trip took 0.261709/0.230479 seconds for
baseline/candidate whole processes. This supports no observed slowdown in that
smoke check, not a kernel or production speedup. Explicit device scratch
allocations are now routed through the workspace allocator; remaining raw
allocations are setup constants/cache, ordinary backing and allocator entrypoints.

Complete native resource sealing, production invocation admission, import/setup
planning, prepared logical calibration and estimator plan consumption are still
incomplete. The unit results do not establish those properties or physical
multi-GPU and final synchronization acceptance.

The final local audit retained the previous equality API's explicit propagation
of workspace-release errors on both equal and unequal success paths. RAII still
retires work during earlier failures. After this correction, the scoped GPU build
was warning-free, the exact equality reuse test passed three more runs, primitives
passed all 272 tests, and runtime passed 207 with four ignored. Those results and
unchanged hashes are in `release-status-*` under the same evidence directory.
No other workspace suite was repeated for this localized native error-path fix.

## Resumed work on 2026-09-12: context-count isolation, compact hash, compact inputs

The two recorded primitive failures were reproduced and explained before any
repair. Both tests pass in isolation (three runs each on the identical stop-time
binary) and both fail deterministically when run alongside the
`serial(gpu_context)` group (three of three attempts), because the unnamed
`#[sequential]` guard and the named guard do not exclude each other while both
groups create and destroy process-global contexts. The two tests now re-execute
in a child process through a shared helper and a new marker in `env.rs`, with
their assertions and the native one-context reset precondition unchanged.
Evidence: `test_data/fleet-sharding-implementation/context-count-isolation-2/`
(1001 GPU-enabled passes, 610 CPU passes, no failures, warning-free builds).

Compact hash sampling (`HashVariant::Decomposed` and `SmallDecomposed`) is now
a compiled admitted operation: `PreparedMatrixOperation::Decompose` carries an
optional hashed source, each admitted range samples the seeded COEFF gadget
source through the new public `sample_seeded_gadget_source_columns` and
decomposes it in place, and the production seed is supplied by the fleet call.
Fresh-output intervals derive their evaluation format from the operation
instead of assuming evaluation format, so the COEFF source scratch is reserved
and validated consistently. Fixture:
`test_gpu_admitted_compact_hash_matches_column_runner_and_reuses_outputs`
(width-two waves over five columns against the existing whole-matrix hash
runner, both variants, dropped-limb cases, invalid preflights, wrong-variant
rejection, output reuse).

Compact inputs are now first-class operands of compiled invocations:
`CompiledMatrixInvocation::arguments` returns the compact operand, admission
inherits compact shard ownership, prepares fixed ordinary inputs in the compact
input's context, and the pilot/production runners receive the resident compact
shard. `GpuFleetSmallMatrix` has a logical identity for invocation matching.
Two operations use it: `CenteredExtendCompact` copies compact columns into a
retained owner of a containing basis through the new native
`gpu_small_matrix_copy_range`, and `MultiplyCompact` multiplies a fixed
evaluation-format LHS replica by the compact RHS columns with width-scaled
expansion workspace claimed per range through `small_rhs_workspaces`. Row-block
compact products are admitted as one invocation per block; the fleet preflight
expands them. Fixtures:
`test_gpu_admitted_compact_centered_extension_preserves_payload_and_shards` and
`test_gpu_admitted_compact_multiplication_matches_cpu_over_partial_waves`.
Each new fixture passed three consecutive runs and once with ring dimension 512.
Evidence: `test_data/fleet-sharding-implementation/compact-hash-admission/`
(1004 GPU-enabled passes with 27 ignored, 610 CPU passes with 25 ignored, no
failures, warning-free builds, source manifest and executable hashes).
Trapdoor/preimage admission, imports/exports, automatic graph-wide inventory,
estimator plan consumption, performance acceptance, independent review and
physical multi-GPU checks remain open.

### Estimator consumption of admitted plans (Step 6, partial) and findings for Steps 3 and 4

`GpuColumnSchedule` now derives exact wave classes analytically
(`wave_classes`, `wave_jobs`), so owners with 90 and 10 columns and widths 10
and 90 report nine waves as one two-device class plus eight single-device
classes without enumerating waves. `GpuColumnMemoryPlan::summary` produces a
serializable `GpuAdmittedPlanSummary` (owner intervals, admitted widths, local
job counts, wave count, wave classes, and every claimed native resource per
device by kind, shape, format, bytes and alignment), and
`GpuDcrtBackend::admitted_invocation_summaries` exposes the admitted unconsumed
batch. The estimator accepts a plan per node through
`GpuNodeMeasurementBackend::admit_invocation_plan`, measures one representative
per admitted class, multiplies by exact multiplicities, keys observations by
scenario, and reports `MeasurementScenario::AdmittedInvocationPlan` only when
every fleet-wave node had a plan. `docs/benchmark-estimator.md` documents the
contract. Tests: `test_wave_classes_and_wave_jobs_match_enumerated_waves`
(runtime), `admitted_plan_classes_use_actual_owner_waves_not_nominal_division`
(estimator, CPU-only), and plan-summary assertions inside the admitted compact
multiplication fixture. Still open for Step 6: feeding plans automatically from
an admitted graph (depends on Step 5) and migrating transfer/dispatch keys.

Step 3 (preimage) was analyzed but not implemented. Under closed native
domains every allocation must match the next reserved claim in order
(`gpu_prepared_matrix_claim`, `GpuDeviceWorkspace::acquire`,
`GpuCudaResource::acquire`). One preimage candidate attempt allocates, in
order: the seeded p2 sample (dk x C, EVAL), the vertical-pair product tp2
(2d x C), the p1 sample (2d x C) with two `gpu_matrix_query_p1_workspaces`
layouts and an in-place INTT of tp2, the residual (d x C), the assembled
candidate (k x C), the gadget Gaussian output (d*log_base_q x C) with five
`gpu_matrix_query_gaussian_gadget_workspaces` layouts, an in-place INTT of the
candidate, and the compact pack tile staging (`CompactWorkspace`); the
destination compact payload and its hard-cutoff plan (two `CompactWorkspace`
owners and one event) are claimed once, the p1 covariance cache (stream, event,
three device buffers) must be created before sealing, and read-only consumer
tracking acquires `CompletionEvent` resources per cross-stream read. Retries
repeat the per-attempt list, so an admitted preimage needs either a native
ordered claim-plan query mirrored by the runner, or a per-attempt dispatch
boundary that rearms scratch. Neither exists yet; this is the first task of a
future Step 3.

Step 4 (imports/exports) was analyzed but not implemented. Imports decode the
canonical global row-major payload on the host and call
`gpu_matrix_load_compact_bytes` on a whole new matrix; there is no retained
rectangle loader with explicit pinned/transfer/event claims, so an admitted
`ImportMatrix` operation needs a native range loader plus width-scaled
`PinnedHost`/`TransferWorkspace`/`CompletionEvent` claims (the
`width_workspaces` mechanism added for compact products can carry them) and an
execution payload channel alongside the hash seed. Exports
(`matrix_to_bytes`) under prepared mode need an explicit export boundary that
reserves `rns_transfer_workspace` and store completion events from the ledger
inventory, as the new fixtures do by hand. Both remain open.

### Imports/exports, graph-derived inventory, trapdoor/preimage admission (Steps 3, 4, 5)

Imports are compiled admitted operations. `ImportMatrix` (canonical compact
bytes) and `ImportStaging` (native RNS staging bytes) load each admitted range
into a coefficient- or same-format staging owner through the production codec
(`load_compact_payload`, `from_cpu_staging_columns`) and copy it into the
retained output; `ImportCompact` stages compact payloads and uses
`copy_columns_from`. Their width-scaled claims (`SubmissionStream` +
`TransferWorkspace(Load)`, or `PinnedHost` + `TransferWorkspace` +
`CompletionEvent`, or `CompactPayload` + `PinnedHost` + `CompletionEvent`) come
from the same native queries the codec uses; the artifact payload travels
through an `ExecutionPayload` channel that is not part of admission identity,
and pilots use zero payloads of the production layout. Compact and staging
imports retire their pinned staging at the end of each range so the next range
rearms the same slots. Exports (`matrix_to_bytes`, `small_matrix_to_bytes`) are
explicit boundaries that hold the codec's clone, stream and store-workspace (or
completion-event) claims from the accepted inventory through a
`PreparedClaimBroker`; `Backend::matrix_to_bytes` now returns a result.
Fixture: `test_gpu_admitted_imports_and_exports_preserve_canonical_bytes`.

`GpuDcrtBackend::prepare_graph_admission` derives the complete inventory of a
validated root-scope graph before any node executes: retained outputs, fixed
input preparation owners, per-kind native scratch at full output width, import
staging and export readback resources, and (see below) traced claim plans for
trapdoor and preimage sampling. Kinds without a compiled runner fail before any
GPU work or sealing. `ExecutionConfig::prepared_gpu_admission` turns it on; the
`Backend::prepare_graph_admission` hook is a no-op elsewhere. Fixtures:
`test_gpu_graph_execution_derives_complete_prepared_inventory` (resident and
host-staged inputs, addition, decomposition, compact product, prepared exports,
unsupported-kind rejection before sealing).

Trapdoor and preimage sampling are admitted through exact native claim plans
recorded before sealing. New native facilities: an open-domain claim trace
(`gpu_claim_trace_begin/end`, `trace_native_claims`) that records every claim an
operation would make on a closed domain in claim order; ordered dispatch
extensions (`gpu_matrix_dispatch_extend/retract`,
`GpuMatrixReservation::enter_or_extend`) so a host-driven step inside an
admitted job holds its exact extra claims without a nested permit; the P1
covariance cache is built from claimable owners (one submission stream slot and
sampler workspaces) so it can be prepared inside the trapdoor's explicit
boundary; and, only inside traced steps (`GpuTracedStepGuard`, entered by
`trace_native_claims` and by `PreparedClaimBroker::hold_traced`), the consumer
tracker claims its event on the same-stream fast path too, so a recorded claim
sequence does not depend on which streams owners received. Ordinary prepared
execution keeps the untraced fast path; scoping the deterministic acquisition
to traced steps is what restored the six existing prepared fixtures that had
begun demanding unplanned `CompletionEvent` claims. The primitive trace fixture
warms the covariance cache (`prepare_preimage_cache`) before tracing, as the
runtime does, so the traced attempt records only per-attempt claims.
`GpuDCRTPolyTrapdoorSampler::preimage_destination`/`preimage_attempt` expose
the destination preparation and one bounded attempt so the runtime drives the
retry loop, each attempt a separate claim boundary; the hard-cutoff host decision
word is returned synchronously at destruction after its decision event.
`PreparedMatrixOperation::Preimage` carries a `PreimageClaimPlan` (destination
plan, per-width tile and attempt claims, synthetic trapdoor claims for pilots);
fleet `sample_trapdoor` and `sample_preimage` route through traced plans keyed by
shape, sigma, base, digits and bound. Pilots measure their broker-held steps as
declared demand (`measure_prepared_with_steps`). Fixture:
`test_gpu_graph_execution_admits_trapdoor_and_preimage_sampling` (trapdoor
sample, staged target, preimage, admitted compact product `A·x` equal to the
target through the prepared export). Admitted preimage widths are the traced
set {1, target columns}; other widths are rejected explicitly. Resident targets
reaching `preimage_target` (host download) and transcript-replayed trapdoor
imports remain outside prepared admission.

Step 6 completion: the fleet records every admitted invocation in
`admitted_plan_log`, keyed by the operation identity the executor selected;
`GpuNodeMeasurementBackend::admit_plans_from_log` maps a prepared execution's
log onto the graph's nodes by the same identity so `estimate` runs with
`AdmittedInvocationPlan` classes. Fixture:
`test_gpu_estimator_consumes_admitted_plans_from_a_prepared_execution` (a
prepared execution's log admitted into a measurement backend that provisions
its own context, because a sealed execution owner never reopens its native
domains; collect, `measure_collected`, then estimate).
Evidence: `test_data/fleet-sharding-implementation/{imports-exports-admission,
graph-admission,preimage-admission,final-2026-09-12}/`; the final directory
holds the complete workspace GPU-enabled and CPU library suites after the
traced-step scoping, plus three consecutive runs of the preimage graph fixture
and of the primitive claim-trace fixture on the same binaries.

### Review-driven corrections (Step 7)

The independent review of the full dirty change set produced these fixes,
each verified by the affected fixtures (three consecutive runs where the
fixture is lifetime-sensitive):

- **Admitted-plan log identity.** The log was keyed by the sticky
  `active_operation`, so an admission without a fresh executor selection was
  logged under the previous node's identity (or zero). `select_operation` now
  also records an `unlogged_operation` that the admission consumes once; an
  admission without a fresh selection is not logged and the estimator measures
  that node nominally. `admit_plans_from_log` reports unmatched log entries at
  info level so a partial admission is visible.
- **`PolynomialValues` had no prepared runner.** The inventory accepted the
  kind with a hand-listed pinned/transfer/event demand, but the fleet ran the
  ordinary readback and would have failed after sealing. The inventory now
  traces the readback on a 1x1 probe of the same parameters for both input
  formats (`trace_polynomial_values`), and the fleet holds the traced claims
  through `PreparedClaimBroker::hold_traced` around the ordinary readback. Int
  families are accepted as owner-free outputs. The derive-inventory fixture
  gained a scalar `coefficients()` output checked against the CPU polynomial.
- **`hold_inner` error masking.** A failed step used to `finish` its permit,
  replacing the step's own error with the unconsumed-claims diagnostic; a failed
  step now drops (retracts) the dispatch and returns its own error.
- **`trace_native_claims` panic safety.** The deterministic-events flag is now
  held by `GpuTracedStepGuard` and the thread-local trace is ended by a drop
  guard, so a panicking traced operation leaves the thread clean.
- **Stale fixture premise.** The derive-inventory fixture used trapdoor
  sampling as its "kind without a runner" example; Step 3 admitted that kind, so
  the fixture now uses a runtime polynomial import (`PolynomialFromValues`),
  which still fails before any node runs or any domain is sealed.

Findings recorded but deliberately not changed: admitted preimage output is
seeded per tile start, so admitted and ordinary paths produce bit-identical
preimages only when their tiles coincide (both are valid bounded preimages
under the same bound; no fixture asserts cross-path equality); the derived
inventory walks the root scope only, so graphs with loops or subgraph calls
are rejected in prepared mode before sealing; prepared trapdoor sampling is
single-device and a multi-device fleet fails explicitly at that node.

Native lifetime corrections from the same review:

- `gpu_matrix_dispatch_end` refuses to end a permit whose extensions are still
  live (`entered` count), so a parent dispatch can no longer write past its
  reservation buffer while an `enter_or_extend` child exists.
- The consumed-claim diagnostics ring is a fixed thread-local array recorded
  `noexcept` after each commit point, and the mismatch message falls back to
  its head if the summary cannot be built; nothing after a committed claim can
  throw across the `extern "C"` boundary.
- A refused `GpuCudaResource::acquire` or `GpuDeviceWorkspace::acquire` (no
  permit, wrong type, CUDA failure on the open path) resets the handle so a
  retry such as the hard-cutoff plan's re-preparation reports the real cause
  instead of "invalid request"; only the quarantined reuse edge retains the
  owner deliberately.
- `GpuTracedStepGuard` is thread-bound (`!Send`), matching the thread-local
  flag it holds.
- Prepared trapdoor and preimage sampling is single-device; the derived
  inventory refuses a multi-device fleet at those nodes before any domain is
  sealed.
