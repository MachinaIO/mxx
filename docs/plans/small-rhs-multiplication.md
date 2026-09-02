# Small-RHS multiplication plan

## Objective and non-negotiable boundaries

Replace the conceptual `mul_decomposed`/`apply_preimage` data flow for
small-coefficient right-hand sides with one explicit `mul_small_rhs` operation.
The operation receives a full-evaluation left-hand-side matrix and a compact
small-RHS value, and returns a full-evaluation matrix.  It must include every
RHS column in one Rust/FFI operation while internally using bounded tiles and
limb waves so peak VRAM does not grow by materialising a full DCRT copy of the
RHS.

The priority order is: mathematical correctness, a simple final ownership and
type/data flow, then performance within the VRAM and asynchronous-GPU
constraints.  The migration is allowed to delete APIs, formats, wrappers, and
old artifacts.  It must not retain compatibility shims, dual paths, hidden
decomposition, flags on full matrices, or fixture-specific fallbacks.

The existing DSL/IR distinction is authoritative.  A `Preimage` remains a
preimage wire from construction through validation, lowering, execution,
families, and artifacts.  `Preimage::as_mat` and its scale-by-one erasure are
deleted.  A preimage is consumed by an explicit `MatrixMulSmallRhs` node/API;
that node means ordinary matrix multiplication with a bounded RHS, not a
sampler and not an implicit decomposition.

## Types and contracts

### IR and DSL

Add a generic bounded `SmallMatrix` wire alongside the relation-carrying
`Preimage` wire.  Change `WireType::Preimage` and
`ConcreteWireType::Preimage` from an unbounded matrix wrapper to a structured
bounded type, and give `SmallMatrix` the same shape-plus-bound fields:

```text
SmallMatrix { matrix: MatrixType, max_coefficient_bound: IntExpr }
Preimage { matrix: MatrixType, max_coefficient_bound: IntExpr }
ConcreteSmallMatrix { matrix: ConcreteMatrixType,
                      max_coefficient_bound: BigInt }
ConcretePreimage { matrix: ConcreteMatrixType,
                   max_coefficient_bound: BigInt }
```

The concrete bound is non-negative and is part of equality, hashing, schema
validation, artifact metadata, and execution diagnostics.  The bound is
inclusive: a centered coefficient is accepted exactly when its absolute value
is `<= B`, matching the existing `matrix_within_coefficient_bound` contract.
Do not infer a bound from a serialized payload.  `SmallMatrix` means only
"bounded coefficients"; it does not assert a cryptographic preimage relation.
`Preimage` is the semantic relation-bearing wire for a bounded value; its
schema contains only the matrix type and inclusive bound.  Relation identity
is the exact producing node and its operand edges, as preserved by the graph
and correctness `StaticLhsKey`/`Job` state; no producer attachment strings or
owner/plan field is added to the Preimage schema.

There are two explicit decomposition producers.  Regular `decompose` uses the
current balanced digit step with `base = 2^base_bits`: each digit is in the
inclusive interval `[-ceil(base/2), ceil(base/2)]` (the tie rule permits either endpoint),
and produces `D = ceil(crt_bits/base_bits) * crt_depth` rows per input row.
`small_decompose` uses the unsigned OpenFHE digits in `[0, base-1]` and
produces `d = ceil(crt_bits/base_bits)` rows per input row.  Both return a
bounded compact value and both `GadgetDecompose` modes always lower to a
relation-typed `Preimage` output with the corresponding fixed bound and row
count.  Generic hash decomposition has no relation: regular balanced and
unsigned hash-decomposition producers always lower to generic `SmallMatrix`
outputs, with bounds `ceil(base/2)` and `base-1` respectively.  These are fixed IR
validation/output rules, not context-dependent runtime choices.

For the balanced tie case, when the remainder is exactly `base/2`, choose
`+base/2` when the Euclidean quotient is even and `-base/2` when it is odd;
this is the current `balanced_digit_step` rule and fixes the endpoint without
leaving the bound as an estimate.

Families use one generic `FamilyType<S> { element: S, count: IntExpr }`
schema and one one-wire `FamilyElement` contract for `Mat`, `SmallMatrix`,
`Preimage`, `Int`, and `Bool`; there are no parallel concrete matrix-family
schema types.  Same-count branch selection is likewise generic over a
one-wire `FamilyElement` and requires identical complete element wire types,
so selecting `Family<Preimage>` preserves its exact matrix schema and bound.
`TrapdoorFamily` remains the explicit multi-wire exception.

Add `NodeKind::MatrixMulSmallRhs` with exactly two arguments: a `Matrix` lhs
and either a `Preimage` or generic `SmallMatrix` rhs, and one `Matrix` output.
Validation requires matching modulus/ring dimension, `lhs.columns == rhs.rows`,
and a non-negative checked RHS bound.  The node has ordinary multiplication semantics;
operational-noise lowering must not add a decomposition or sampling term.
Add `Mat::mul_small_rhs(self, rhs: SmallMatrix) -> Mat` and
`Preimage::mul_small_rhs(self, lhs: Mat) -> Mat`.  Both construct the same node; the
first accepts only generic `SmallMatrix`, while the second accepts only a
relation-typed `Preimage`.  There is no `Mat` conversion, relabeling, or bound
discarding adapter.  Thus `mul_small_rhs` accepts exactly these two typed RHS
forms and no ordinary matrix, while the runtime backend has one compact owner
for both.

`small_decompose`, regular `decompose`, generic hash-decomposition, and
preimage sampling are the only producers used by these RHS paths.  They
produce their declared bounded type directly; `mul_small_rhs` never calls a
decomposition routine.  Any remaining regular decomposition consumer must be
made explicit and must not be routed through an ordinary untyped matrix.

### Runtime/backend types

At the primitive boundary, make `PolyTrapdoorSampler::M` implement
`PolyMatrixSmallRhs`; do not add a duplicate sampler-level associated owner.
Its bounded `preimage` takes the inclusive `max_coefficient_bound` and returns
`Result<<Self::M as PolyMatrixSmallRhs>::SmallMatrix, SmallMatrixError>`.
Delete the public expanded preimage path, `GpuPreimageRequest`, and
`preimage_batched_sharded`; the GPU compact sampler is the single trait
implementation, not an inherent side API hidden behind a runtime adapter.
Keep `preimage_extend(...) -> M` because the distinct `[B,C]D=U` construction
still has full-matrix primitive semantics; its implementation may use a private
expanded-candidate helper but must never route through compact `preimage`.
Consequently runtime has no private `CompactPreimageSampler` compatibility
trait.  Remove the old expanded decomposed methods from `PolyHashSampler` once
their call sites use the explicit compact backend producers.

Extend `Backend` with one associated compact owner type, `SmallMatrix`, and
separate semantic-producing operations:

```text
gadget_decompose(..., small=false|true) -> SmallMatrix
sample_hash_decomposed / sample_hash_small_decomposed -> SmallMatrix
sample_preimage(...)             -> SmallMatrix
multiply_small_rhs(lhs: Matrix, rhs: SmallMatrix) -> Matrix
small_matrix_to_active_placement / small_matrix_to_placements
small_matrix_to_bytes(value, expected_schema, semantic_kind) /
small_matrix_from_bytes(expected_schema, bytes, expected_semantic_kind)
```

The CPU backend may use its existing matrix reference representation as
`SmallMatrix`; its API contract still says that every coefficient is bounded
and canonical.  The GPU backend must use a distinct `GpuSmallMatrix` compact
owner.  Do not add a `small`/`compact`/`is_preimage` flag to
`GpuDCRTPolyMatrix`, and do not make `RuntimeValue::Matrix` carry one.

Define one concrete bounded schema passed by these calls:
`ConcreteBoundedMatrixSchema { matrix: ConcreteMatrixType,
max_coefficient_bound: BigInt }`.  The IR/runtime and artifact
descriptor/manifest validate this complete schema once.  `SmallMatrix` is
semantic-kind-free: the
backend owner stores only bounded coefficients and shape.  The artifact layer
supplies the expected schema and `SmallMatrix` or `Preimage` semantic kind to
the codec call, and the codec performs only O(1) header equality checks against
that already-validated schema before constructing an owner.  No semantic-kind
field is duplicated in `GpuSmallMatrix` or `RuntimeValue::SmallMatrix`.

Add `RuntimeValue::SmallMatrix(Arc<B::SmallMatrix>)`, including clone/drop,
placement, family, staging, materialisation, output, and value-kind error
handling.  Both `ConcreteWireType::SmallMatrix` and
`ConcreteWireType::Preimage` materialise to `RuntimeValue::SmallMatrix`; a
`Matrix` wire materialises only to `RuntimeValue::Matrix`.  The runtime keeps
the wire kind in validated metadata even though the backend owner is shared.
A mismatch is an error, never a conversion attempt.  Placement code must
replicate a small matrix to each device once, keep all DCRT limbs of a full
matrix on one placement, and retain the preimage type through indexed
families.

### Canonical coefficients and serialization

Define one canonical signed coefficient representation for all CPU/GPU
boundaries.  The exact linear coefficient index is
`((row * columns + column) * N + coefficient)`, where `N` is the ring
dimension.  Each coefficient is a signed integer in `[-B, B]` and has no
alternate residue representative.  Encode one complete sign byte (`0` for
zero, `1` for non-negative, `2` for negative), followed by exactly
`magnitude_bytes = max(1, ceil(bit_length(B) / 8))` little-endian magnitude
bytes.  Thus `B=255` uses one magnitude byte; zero has sign `0` and an all-zero
magnitude, while a nonzero value must have sign `1` or `2` and a nonzero
magnitude.  The encoded coefficient count is `rows * columns * N`, and the
payload length is exactly `count * (1 + magnitude_bytes)`.  Reject negative
zero, non-canonical width, values outside `[-B,B]`, trailing bytes, dimension
overflow, and a payload whose declared bound differs from the IR bound.

Introduce one current compact coefficient payload/codec for both semantic
kinds, rather than mapping preimages to `ArtifactType::Matrix`; delete old
formats without compatibility decoders.  Artifact descriptors remain distinct:
use `ArtifactType::SmallMatrix` for generic bounded values and
`ArtifactType::Preimage` for relation-bearing values.  A generic descriptor
must never import as a relation-bearing `Preimage`; the descriptor is validated
against the producing wire type before decoding.  There is one codec and one
payload layout, not duplicate storage formats.  Define
`SmallMatrixSemanticKind` in the IR artifact layer with exactly `Generic` and
`Preimage` values; it is an encode/decode argument and header field, never
backend-owner state.  The fixed header is, in order:
four ASCII magic bytes `SMR1`,
`semantic_kind:u8` (`0` = generic `SmallMatrix`, `1` = `Preimage`),
`rows:u64`, `columns:u64`, `ring_dimension:u64`, `bound_len:u32`, the
non-negative canonical BigInt bound as exactly `bound_len` little-endian
magnitude bytes, `magnitude_bytes:u32`, and `coefficient_count:u64`; all
integer fields are little-endian.  For the bound, `bound_len` is exactly
`max(1, ceil(bit_length(B)/8))`; zero is one zero byte and a nonzero bound has
no leading zero byte.  The payload follows with exactly
`coefficient_count * (1 + magnitude_bytes)` bytes under the linear index above.
The payload need not carry modulus data: the decoder uses the complete expected
concrete schema and does not recompute modulus products, NTT parameters,
relation facts, content hashes, or bounds.
The artifact layer hashes exactly this header plus payload (including
dimensions and bound) when a public content hash is required; any content-hash
verification remains the existing artifact-store check and is not repeated by
the codec.  Public hashes remain allowed and private hashes remain forbidden.
CPU store/load and
GPU store/load must produce identical canonical bytes and reject malformed
input before allocating a large destination.  `encode_artifact` passes the
descriptor's complete `ConcreteBoundedMatrixSchema` and semantic kind to
`small_matrix_to_bytes`; `decode_artifact` passes the complete expected schema
and kind to `small_matrix_from_bytes`.  Encode validates owner shape, modulus,
ring dimension, and inclusive bound against that schema using contract
assertions and validated owner metadata only; it performs no expensive
recomputation.  Decode performs O(1) checks of header rows, columns, ring
dimension, bound, coefficient width/count, and semantic kind exactly against
the expected schema before a large allocation, then constructs the owner with
the expected backend parameters, device, context, and placement; it never
derives them from the payload.  The mandatory decode
copy pass simultaneously decodes each coefficient and checks its canonical
sign/magnitude and `[-B,B]` range; there is no second payload scan or repeated
norm check.  The manifest layout records the same current format; it is not a
version-negotiation mechanism.

## GPU representation and operation

`GpuSmallMatrix` owns exactly: opaque compact-device handle, rows, columns,
ring dimension, coefficient bit/byte width, bound metadata, device/context
identity, and the completion/release event ownership required by the existing
stream pool.  It contains no CRT-limb pointer array, `level`, `is_ntt`, NTT
state, or host matrix clone.  The opaque CUDA object stores packed signed
coefficients in a single coefficient-major buffer.  Host-to-device load and
device-to-host store use async copies and per-stream event sets; destruction
queues release after the event, and no wrapper that only launches device work
waits for the device.

Add one Rust wrapper and one FFI entry point,
`gpu_matrix_mul_small_rhs(out, lhs_eval, rhs_small, tile_columns,
k_tile, limb_wave)`.  The wrapper validates contexts, device, dimensions,
bound, and that `lhs_eval` is full DCRT evaluation format.  It allocates a
full-evaluation `m x C` output and launches the operation asynchronously.  The
single C++/CUDA call may contain a host enqueue loop over
`(limb_wave,column_tile,K_tile)`, but it may not wait, synchronize, or return
between iterations.  Each staged kernel handles `ell = min(L, limb_wave)`
limbs and `C_t` columns together; there are no per-limb or per-column kernel
launches.  Staged NTT kernels are permitted inside a wave.

Use one simple schedule with `K_t = min(K, configured_k_tile)`,
`C_t = min(C, configured_column_tile)`, `ell = min(L, configured_limb_wave)`,
and `buffer_count = 1`.  A single in-place `rhs_eval` workspace holds
`ell*K_t*C_t*N` evaluation words: unpack signed coefficients into it, embed
the same integer independently modulo each of the `ell` CRT primes, and run
the per-prime NTT in place.  The full output's current `m*C_t` tile is itself
the accumulator: the first K wave zeroes its tile, later K waves add modulo
each prime, and the next column tile reuses the same workspace only after its
stream-order dependency.  No separate output tile or second buffer is live.
The full `lhs_eval[m x K]` remains resident and is read by every K wave.

The coefficient array can be shared across primes, but NTT evaluations cannot
generally be shared across CRT primes because each prime has a distinct
modulus and transform.  Reduce products/accumulators modulo each prime at
each safe step; use checked `u128`/CUDA modular multiplication and reject
unsupported modulus or bit-width instead of allowing integer overflow.

The one stream enqueues unpack, NTT, and accumulation in dependency order.
The returned completion event covers all launches and registers/retains the
`lhs_eval`, compact RHS, `rhs_eval` workspace, and full output until every
queued kernel finishes; the release event then permits asynchronous
destruction.  Each later consumer waits on that producer event through the
existing backend event model before reading or reusing any of those values.
There is no
`cudaDeviceSynchronize`, device-wide synchronization, host polling, or
host-side blocking loop.  Existing release-stream fencing remains the only
permitted deferred-resource fence.  Expose positive-value controls for
`MXX_MUL_SMALL_RHS_TILE_COLUMNS`, `MXX_MUL_SMALL_RHS_K_TILE`, and
`MXX_MUL_SMALL_RHS_LIMB_WAVE`; when unset, use column-chunk-one and maximize
the K tile and then limb wave from the exact residency report. Explicit values
remain available for debugging and benchmark comparisons and are still
rejected when they exceed the residency budget.
The enqueue loop uses scalar launch arguments and one final completion event,
not C-sized device pointer/event arrays, so its metadata remains bounded.

In `crates/primitives/src/env.rs`, add exactly one sampler control,
`MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS`, parsed as a positive integer by
`gpu_preimage_max_tile_attempts()`.  An unset variable defaults to `64`,
deliberately replacing the current unbounded retry loop; a zero, negative, or
malformed value is an error, not a fallback and not an alias.  This setting is
primitive-owned and applies per target-column tile.

Let `L` be active CRT limbs, `N` the ring dimension, `m` the lhs rows, `K` the
RHS rows, `C` the total RHS columns, `C_t` the configured column tile,
`K_t` the configured K tile, `ell=min(L,limb_wave)`, `b_s = 8*(1 +
magnitude_bytes)` compact coefficient bits, `w_q` the bytes per evaluation
word, and `E_bits` the fixed event/launch metadata bits.  With `buffer_count=1`,
the measured peak bound is:

```text
peak_bits <= 8*L*m*K*N*w_q                  // resident lhs evaluation
           + K*C*N*b_s                      // all-column compact RHS
           + 8*L*m*C*N*w_q                   // one full output/accumulator
           + buffer_count*8*ell*K_t*C_t*N*w_q // one in-place RHS eval wave
           + E_bits.
```

The implementation must report each term and overhead from the allocator
probe.  Total `C` affects only the compact RHS and required full output; it
does not create a full-DCRT RHS or a `K*C` evaluation temporary.  Select
`C_t`, `K_t`, and `ell` from an explicit byte budget, with defaults whose
measured workspace is no larger than the old column-chunk-one path.  Delete
`MXX_MUL_DECOMPOSE_COLUMN_CHUNK_WIDTH` and both
`mul_decompose` implementations and timing scaffolding; no old environment
variable may silently control the new operation.

### Residency bound (mandatory asymptotic check)

Let `D = ceil(crt_bits / base_bits) * crt_depth`, `N` be the ring dimension,
and use bits rather than bytes for this asymptotic statement.  The allowed
resident order is `O((crt_bits * crt_depth) * N * D)` bits.  The following
table is a required design check; the implementation must not rely on a
favourable constant to excuse a forbidden order.

| Resident object | Order | Status | Required handling |
| --- | --- | --- | --- |
| Full-DCRT lhs evaluation (`m x K`) | `O(crt_bits * crt_depth * N * mK)` bits | Allowed when `mK = O(D)` | Keep the production evaluation lhs resident. |
| Full-DCRT output (`m x C`) | `O(crt_bits * crt_depth * N * mC)` bits | Allowed when `mC = O(D)` | Allocate the one full output required by the API; do not make a second full output. |
| Compact all-column RHS (`K x C`) | `O(b_s * N * KC)` bits | Allowed when it fits the budget; for gadget digits `b_s=O(base_bits)` and `K=C=D`, this is `O(base_bits * N * D^2)`, exactly the allowed order | Keep all columns only in this compact coefficient form. Generic sampled preimages use their stored sign-plus-magnitude width `b_s`; do not substitute `base_bits` for them. |
| Expanded DCRT/NTT RHS (`K x C`) | `O(crt_bits * crt_depth * N * KC)` bits | Forbidden for `K=C=D` (`O(...D^2)` rather than the allowed `O(...D)`) | Expand only a bounded `K`/column/limb tile, then release it by event. |
| Decomposed full-DCRT RHS plus compact RHS | Expanded term plus compact term | Forbidden | No producer or serializer may materialise both representations. |
| NTT workspace for one wave (`K_t x C_t`, `ell` limbs, `buffer_count=1`) | `8*buffer_count*ell*K_t*C_t*N*w_q` bits | Allowed exactly when `8*buffer_count*ell*K_t*C_t*N*w_q <= remaining_budget_bits` | Bound all three tile axes from the remaining budget; never set `K_t=K` or `C_t=C` blindly. |

The shape-general peak formula must therefore report the compact all-column
term and only `K_t*C_t` expanded workspace, with every resident shape checked
against the configured residency budget, including `ell`, `K_t`, `C_t`, and
`buffer_count` in the workspace term.  Define
`remaining_budget_bits = configured_budget_bits - (lhs_bits + compact_rhs_bits
+ output_bits + E_bits)` and reject a configuration unless it is non-negative
and satisfies the exact workspace inequality
`8*buffer_count*ell*K_t*C_t*N*w_q <= remaining_budget_bits`.  For the representative gadget
case, a `1 x D` target/decomposition produces a `D x D` small RHS: its
all-column compact storage is `base_bits * N * D^2` bits and is permitted,
whereas its expanded DCRT/NTT form is `crt_bits * crt_depth * N * D^2` bits
and is forbidden.  With `C_t=O(1)` and `K=D`, each expanded wave is only
`O(crt_bits * N * ell*K_t*C_t)` bits and is
`O(crt_bits * crt_depth * N * D)` for the old chunk-one choice
`ell=L`, `K_t=D`, `C_t=1` (using `8*w_q=O(crt_bits)`).  Thus old
chunk-one residency is allowed by the user budget; larger `K_t`, `C_t`, or
`ell` values trade against one another only while satisfying the exact
remaining-budget inequality.  If a requested shape would make the
full lhs or output exceed the budget, fail with a resource/shape error or use
the existing bounded outer family wave; never silently allocate an
asymptotically larger RHS.

## Producers, artifacts, and semantics

Gadget decomposition must write directly into the CPU `SmallMatrix` or GPU
compact buffer.  The GPU sampler currently batches full-DCRT candidates; replace
that with bounded draws over all rows and target-column tiles.  Set the sampler
row tile `K_s=K` always: the preimage equation `A*x=t` couples all rows, so
rows cannot be sampled independently.  Fix `draw_batch=1`; do not add a draw
batch control in this migration.  Tile only target columns `C_s` (and retry
one draw at a time).  If one full-row, one-column draw cannot fit the budget,
fail with a resource/shape error rather than tiling rows or silently falling
back to a full expanded matrix.

Use the actual `GpuContext` allocation plan for every full `GpuMatrix`; do not
estimate it from a symbolic limb count.  Add one side-effect-free
`gpu_matrix_query_allocation_bytes(ctx, level, rows, columns, format)` API and
make it and `gpu_matrix_create` consume the same checked planner in
`MatrixData.cu`.  The planner sums each nonempty partition once, using its
active local limbs, two allocated coefficient/evaluation regions, its actual
`ctx->decomp_counts_by_partition[partition]`, and `ctx->max_aux_limbs` pointer
slab.  Its result separates `data_bytes`, `aux_bytes`, deterministic persistent
and allocation-ready event-handle bytes, and their checked total.  The query
must not change devices, advance streams, create events, or allocate memory.
Opaque CUDA event storage and allocator fragmentation are not predictable by
this query: keep a fixed allocator/headroom reserve `E_bytes` separate and
prove the physical peak with the allocator high-water measurement.

Let `T_candidate`, `T_perturb`, and `T_check` name the actual candidate,
perturbation, and centered-norm-check owners.  Count each named owner exactly
once from either the shared full-matrix allocation query or its actual raw
allocation; a distributed candidate is one owner, not one global allocation
per GPU, and it must not be included again inside `check_scratch`.  Include
residual, `z_hat`, perturbation blocks, target tile, any packed staging buffer,
the persistent public/trapdoor/target values, the persistent compact
all-column destination, the acceptance flag, deterministic event handles, and
the separate allocator/headroom reserve in `sampler_peak_bits`.  Define
`sampler_remaining_budget_bits = configured_budget_bits -
(persistent_public_bits + persistent_trapdoor_bits + persistent_target_bits +
compact_destination_bits + acceptance_flag_bits + sampler_event_bits + E_bits)` and
require the checked inequality
`candidate_bits + perturbation_bits + check_scratch_bits +
packed_staging_bits <= sampler_remaining_budget_bits` before choosing `C_s`;
report `sampler_peak_bits` as that sum plus all persistent terms.  The compact
destination term is exactly `K*C*N*b_s` bits for all accepted columns.
No full `K*C` expanded candidate is permitted.  For each candidate tile,
perform the authoritative full-CRT centered-norm check, then immediately pack
an accepted tile into the compact all-column destination and release the
expanded candidate, perturbation, and check scratch by stream/event order.
Copy only the compact acceptance flag D2H.  Count each acceptance-flag D2H
decision as one attempt.  At the adaptive decision boundary, the sampler
control stream may wait for that flag copy, choose a retry, and enqueue the
next draw; this is the sole sampler exception to the no-host-wait GPU rule.
Rejected draws must wait for stream-ordered release/reuse before their buffers
are reused.  After exactly `gpu_preimage_max_tile_attempts()` attempts for a
column tile, return a fail-closed sampler error naming the tile and exact
attempt count, but only after event-safe cleanup.  CPU
sampling may keep its existing reference representation but must apply the
same inclusive bound before returning its compact owner.  Matrix/device-only
wrappers and `mul_small_rhs` remain asynchronous.  A host wait is permitted
outside adaptive control only when a caller explicitly requests canonical D2H
artifact bytes, at the artifact store return boundary described above.

Accepted preimages, gadget outputs, and artifact loads all use the same bound
validation and canonical encoding.  Device-only artifact store/load wrappers
remain asynchronous; a D2H artifact serialization necessarily waits at the
host-return boundary for the returned bytes (and only there).  Artifact stores
use the compact serializer, and artifact loads decode directly to the
backend's `SmallMatrix`.

Update only the minimal exhaustive matches in correctness lowering and its
normal-form/arena/job plumbing so `MatrixMulSmallRhs` is represented as the
existing ordinary multiplication operator with a typed bounded RHS.  Preserve
existing semantics and preimage relation metadata where compilation requires
it; do not redesign the noise simulator, add error terms, or change its model.
Do not replace the node with a scale-by-one or infer a relation from a matrix
number.  This plan explicitly excludes noise-simulator accuracy work and
simulator validation as acceptance gates.

## File-by-file implementation slices

The following slices are serialized Luna ownership units.  Sol decides only
design questions that this plan leaves open; Luna implements one slice, runs
its gate, and records the exact result for the next daily review.

1. **IR/DSL contract:** `crates/ir-core/src/types.rs`, `node.rs`,
   `validate.rs`, `artifact.rs`, `constraints.rs`, `crates/dsl/src/lib.rs`.
   Add distinct bounded `SmallMatrix` and relation-typed `Preimage` schemas
   plus `MatrixMulSmallRhs`; delete `Preimage::as_mat`; update graph schemas,
   shape checks, serde, and the DSL's `SmallMatrix`/`Preimage` family,
   artifact-input, public-output, and import APIs.
2. **CPU small-owner foundation:** `crates/primitives/src/matrix/mod.rs`,
   `crates/primitives/src/matrix/dcrt_poly.rs`, and their focused tests.
   Add companion `SmallPolyMatrix` and `PolyMatrixSmallRhs` traits, the thin
   semantic-kind-free `CpuSmallMatrix<M>` owner, canonical signed coefficient
   payload encode/decode, CPU gadget/hash/preimage producers where their
   primitive APIs naturally belong, and trusted ordinary multiplication for
   `multiply_small_rhs`.  Keep `SmallPolyMatrix` metadata/placement/codec-only:
   it must not require a full-matrix associated type or materialization API;
   those remain inherent to the CPU owner.  Validate the complete shape/parameter/bound contract
   once with checked arithmetic; do not add an SMR1 header, semantic-kind
   field, duplicate matrix clone, or compatibility serializer.  GPU storage
   and runtime orchestration are deliberately deferred to slices 3 and 4.
3. **Backend/runtime ownership:** `crates/runtime/src/backend.rs`,
   `backend/poly.rs`, `backend/poly_gpu.rs`, `backend.rs`'s runtime-value
   implementations, `executor.rs`, `artifact.rs`, `session.rs`, and
   `transcript.rs` where value kind/hash handling is shared.  Add the
   associated `SmallMatrix` API, `RuntimeValue::SmallMatrix`, placement and
   family/staging paths, direct compact artifact codecs, and explicit node
   execution.  Remove matrix-only matches that would erase Preimage.
4. **GPU compact owner/FFI:** `crates/primitives/src/matrix/gpu_dcrt_poly.rs`,
   `poly/dcrt/gpu.rs`, `env.rs`, CUDA declarations in
   `cuda/include/matrix/Matrix*.cuh`, and implementations in
   `cuda/src/matrix/Matrix*.cu` (keep CUDA bodies in `src`).  Add the compact
   owner, explicit regular/unsigned decomposition mode in the FFI, async serde,
   producer paths, one FFI multiply operation, event rules,
   checked modular accumulation, and allocator accounting.  Remove
   `mul_decompose`, `mul_decompose_small`, per-column copies, and old timing
   logs/env parsing.
5. **Correctness compile plumbing (scope-limited):**
   `crates/correctness/src/operational_noise/lower.rs`,
   `normal_form.rs`, `arena.rs`, and `job.rs` only where exhaustive matches
   require the new node/type.  Lower the new node as ordinary multiply and
   preserve existing relation/bound semantics.  Do not edit simulator
   formulas, error-term modeling, or noise accuracy behavior; no simulator
   redesign is part of this task.
6. **Application call sites:** migrate every RHS use in
   `crates/bgg/src/{attribute_encoding,boolean,encoding,masked_decoder,
   lwe_lookup,naive_vec,noise_refresh,public_key,slot_operation,tall_encoding,
   wee25_commitment,wee25_opening,wee25_public_parameters}.rs`,
   `crates/gadgets/src/input_injector.rs`,
   `crates/we/src/diamond/{graph,estimate,estimate_gpu}.rs`, and
   `crates/io/src/{diamond/graph.rs,aky24/prfe.rs}`.  Replace `.as_mat()` at
   each actual small-RHS consumer with `mul_small_rhs`; preserve ordinary
   matrix operations where the RHS is not bounded.  Update application
   artifact declarations and family imports rather than adding adapters.
7. **Estimator/benchmarks:** `crates/bench-estimator/src/{lib.rs,gpu.rs,
   harness.rs}`, primitive benches, and owning WE estimators.  Measure the same
   one-call production path, compact RHS bytes, tile/limb-wave workspace, and
   full-output persistence; remove estimates that count Preimage as full DCRT.

## Validation matrix and review gates

No integration tests are part of this plan.  Each implementation slice must
first pass warning-free `cargo +nightly fmt --all`, focused unit tests for its
crate, and CPU/GPU `--no-run` compilation as applicable.  Then run all
workspace unit tests with and without GPU features using `--lib` unit-only
commands.  The only permitted exclusions are individual, explicitly named
`mxx-gadgets` unit tests; each excluded test must have its own recorded
`>20`-minute elapsed run as evidence.  Do not exclude a whole crate or command
without naming the tests.  GPU tests run outside the sandbox and are repeated
with identical commands (3--5 repetitions for smoke/correctness, and the
repository-required repeated run for synchronization-sensitive tests).

Required correctness tests include:

- IR round-trip/serde/hash and rejection tests for bounds, shape, wrong wire
  kind, generic-vs-preimage artifact semantic mismatch, negative zero,
  malformed width, and stale artifact format.  Codec/runtime tests must also
  reject an expected-schema modulus mismatch, an owner/backend-context or
  placement mismatch, schema shape or ring-dimension mismatch, and bound
  mismatch before large allocation; these are descriptor/schema contract
  failures, not coefficient-payload data.
  Header-mismatch cases must fail through the O(1) contract checks without a
  second payload scan, norm recomputation, or other heavy validation.
- DSL graph assertions proving no `MatrixScale(1)` or hidden decomposition is
  inserted and that `MatrixMulSmallRhs` has Matrix/(Preimage or
  SmallMatrix)/Matrix types.
- Slice 5 operational-lowering proof: two same-schema `Preimage` producers,
  `Family<Preimage>::get` followed by `mul_small_rhs` preserves the exact
  selected expression; the relation/recomposition rule uses that exact source,
  and `MatrixMulSmallRhs` adds neither a sampler nor a decomposition.
- CPU decomposition and preimage relation tests, signed boundary tests, all
  zero, negative, maximum-bound, multi-column, family, store/load, and
  ordinary-multiply equivalence against the trusted primitive.
- GPU compact load/store/hash tests, producer relation tests, wrong-device and
  format failures, multi-column/full-output tests, and repeated equivalence of
  `mul_small_rhs` against CPU/trusted full-DCRT multiplication.  Include
  distinct-prime checks demonstrating that per-prime NTT embedding is done and
  no limb is accidentally reused.
- Focused sampler tests for first-attempt success, retry-then-success, exact
  exhaustion at the configured maximum, invalid zero
  `MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS`, and event-safe buffer reuse after a
  rejected draw.  Retry tests must use genuine relation-valid GPU candidates
  and the production centered check/pack path: choose a bound between two
  observed distinct centered maxima for reject-then-success, and below every
  observed nonzero maximum for exhaustion.  Candidate search is test-bounded
  and diagnostic; it must not inject acceptance Booleans, fixed candidate
  bytes, or fixture-specific coefficients.  The exhaustion assertion checks
  tile identity, exact attempt count, no extra draw, and event-safe cleanup.
- Allocation-query tests cover single- and multi-GPU contexts, a decomposition
  count different from CRT depth, checked-overflow parity with creation,
  repeated side-effect-free queries with unchanged device memory, and creation
  through the same shared plan.
- Runtime placement, lazy/staged artifact, indexed-family, liveness/drop, and
  ordinary-lowering compile/unit tests; estimator tests for compact bytes, tile
  workspace, and column-independent temporary memory.  Existing correctness
  tests may be run as ordinary workspace unit tests, but noise-simulator
  accuracy or model changes are not an acceptance criterion here.

Before/after measurement is mandatory for acceptance, but is not itself a
correctness claim: record the old column-chunk-one latency/peak and the new
one-call latency/peak on the same parameters, device, seed policy, and
artifact load/store flow.  Report raw per-stage timings, tile/limb settings,
compact RHS size, full output size, allocator high-water mark, and command plus
commit.  Do not invent benchmark success from estimates or no-run builds.

Final gates are warning-free CPU/GPU no-run builds, nightly formatting, all
allowed unit tests, repeated GPU tests, estimator consistency, the measured
before/after report, clean diff review, and an independent new Sol review of
the implementation and evidence.  Roll back a slice if its gate fails; retain
the diagnostic artifact and exact failure rather than restoring a compatibility
path.  Completion requires explicit confirmation that no old API, env var,
artifact format, type-erasing conversion, per-column operation, hidden sync,
or unbounded DCRT RHS temporary remains.  Add a repository-wide zero-old-use
gate (`rg` over source, benches, tests, and docs outside this plan) proving
there are no `Preimage::as_mat`, `mul_decompose`,
`mul_decompose_small`, `apply_preimage`, or
`MXX_MUL_DECOMPOSE_COLUMN_CHUNK_WIDTH` uses.
