# Bounded Small-RHS Multiplication with Tiled DCRT Expansion

## Status

Replace the conceptual `mul_decomposed`/`apply_preimage` data flow for
small-coefficient right-hand sides with one explicit `mul_small_rhs` operation.
The operation receives a full-evaluation left-hand-side matrix and a compact
small-RHS value, and returns a full-evaluation matrix. Each primitive call
receives exactly one runtime shard of at most W columns and processes all of its
columns, K rows, and CRT limbs concurrently. Larger logical matrices are split
only by the runtime fleet.

The reviewed upstream revisions are:

- base: `9c61347bf54595e35bb900257155626076314d35`;
- initial PR revision: `7c497376f19c65943532e6a8949d1ba558eb7647`;
- GPU optimization revision: `5ad6819a37a846b8f71d81909f8a26a353bd0eda`;
- current reviewed PR head: `174ae6e53a7de9d09932339ac56f8fd4f30a0c0d`.

The updated revision adds persistent GPU matrix descriptors, owner-local
completion waits, coefficient-domain hash sources, batched domain conversion,
and compact-multiplication kernel improvements. It does not change the PR's
original plan document or its row-major compact payload. This plan therefore
specifies the final combined design rather than prescribing a commit-level
cherry-pick.

The current head additionally restores per-instance evaluation of preimage
coefficient bounds. A bounded schema may depend on a parallel-loop environment;
it must never be frozen using the loop-index-zero validation environment.

Implementation uses the PR as the primary code source. Copy compatible files,
functions, tests, and CUDA kernels directly from the reviewed PR revision, then
apply only the deltas required by this document and by the current branch. Do
not independently rebuild behavior already implemented correctly in the PR.
The principal intentional divergences are complete owner-schema validation,
per-instance bound preservation, stable-coordinate sampling, and
current-branch integration.

This document authorizes no implementation by itself. Implementation begins
only after this plan is approved.

## Objective

Introduce one explicit operation for ordinary matrix multiplication with a
bounded small-coefficient right-hand side:

\[
  Y = A S,
  \qquad
  A \in R_q^{m\times K},
  \quad
  S \in R^{K\times C},
  \quad
  Y \in R_q^{m\times C}.
\]

The operation must:

1. preserve the full mathematical dimensions of every input and output;
2. avoid materializing `S` as a full DCRT/NTT matrix;
3. represent generic bounded matrices and relation-bearing preimages without
   erasing their semantic distinction;
4. keep the complete `O((log q)^2)` small matrix resident in compact signed
   coefficient form, while streaming only its DCRT/NTT expansion by columns;
5. retain `O(log q)` matrix objects on the GPU when they fit the configured
   residency budget;
6. use the same production path in runtime execution and benchmark estimates;
7. keep GPU work asynchronous except at an explicit host-observation boundary;
8. make output values and serialized artifacts independent of tile, wave, and
   device scheduling choices; and
9. require only the smallest exhaustive-match update in
   `mxx-noise-simulator`, because that simulator is expected to be retired.

Correctness and exact value semantics take precedence over performance. Peak
memory is controlled by compact coefficient storage plus bounded DCRT
workspace streaming, never by shrinking a matrix, changing dimensions,
omitting columns, or replacing the production operation with a smaller
mathematical problem.

## Non-goals

This migration does not:

- preserve backward compatibility with old APIs, artifacts, environment
  variables, or type-erasing conversions;
- redesign an application's noise authority;
- improve or otherwise expand the generic noise simulator;
- add a CPU fallback for failed GPU execution;
- cache values that depend on an unavailable secret;
- change the algebraic preimage relation;
- make column tiles or wave widths part of artifact identity; or
- run integration tests unless separately requested.

## Architectural boundaries

The implementation follows the workspace dependency direction:

- `mxx-ir-core` owns wire types, node semantics, validation, and artifact
  schemas;
- `mxx-dsl` exposes typed construction APIs;
- `mxx-primitives` owns concrete compact matrices, preimage sampling, DCRT
  matrices, GPU wrappers, and CUDA;
- `mxx-runtime` owns lazy artifact access, placement, execution, and value
  lifetime;
- `mxx-bgg` and reusable gadgets consume the typed operation;
- application crates own their program-specific use and simulation;
- `mxx-noise-simulator` depends only on lower-level IR and must never depend on
  an application crate; and
- application crates do not depend on one another.

GPU-specific Rust code must live in a file whose name contains `gpu`. CUDA
headers declare only cross-file or Rust-facing interfaces; implementations
belong in CUDA source files.

## Terminology and size classes

Let:

- `N` be the ring dimension;
- `L` be the active CRT-limb count;
- `m` be the left-hand-side row count;
- `K` be the shared matrix dimension and small-RHS row count;
- `C` be the total small-RHS/output column count;
- `B` be the inclusive centered coefficient bound of the small RHS;
- `beta = 2^base_bits` be the power-of-two gadget base;
- `ell_beta = sum_i ceil(log2(q_i) / base_bits)` be the total active
  per-CRT gadget digit count (equal to
  `L * ceil(crt_bits / base_bits)` when all active limbs use `crt_bits`);
- `s_B` be the canonical bytes per small coefficient;
- `w_i` be the evaluation-word byte width for CRT limb `i`;
- `C_t` be the resident target-column count;
- `K_t` be the multiplication reduction tile; and
- `ell` be the active limb-wave width.

An `O(log q)` matrix has only one dimension proportional to the gadget digit
count and may remain resident when the exact allocation fits. A matrix with
both row and column counts proportional to the gadget digit count has
`O((log q)^2)` elements and must be column-streamed. This classification uses
actual checked dimensions, not type names or paper-specific symbols.
The symbol `ell_beta` is used only for gadget decomposition digits. It is not a
mask-digit count, fresh-error-digit count, plaintext-base digit count, or
secret dimension.

## Semantic model

### Bounded matrix types

Add the following logical wire types:

```text
SmallMatrix {
    matrix: MatrixType,
    max_coefficient_bound: IntExpr,
}

Preimage {
    matrix: MatrixType,
    max_coefficient_bound: IntExpr,
}
```

and their concrete counterparts:

```text
ConcreteSmallMatrix {
    matrix: ConcreteMatrixType,
    max_coefficient_bound: BigInt,
}

ConcretePreimage {
    matrix: ConcreteMatrixType,
    max_coefficient_bound: BigInt,
}
```

The bound is non-negative, inclusive, and part of type equality, hashing,
validation, artifact metadata, and diagnostics. Every centered coefficient
`x` must satisfy `abs(x) <= B`.

The symbolic bound remains an `IntExpr` until a concrete execution instance is
known. For every scalar or parallel instance, evaluate it in that instance's
`ParamEnv`. Batch construction, sampler requests, runtime owner metadata, and
artifact descriptors use this evaluated value. A batching key includes the
complete concrete bounded schema, including the evaluated bound; instances
with different bounds may not share a request that assumes one common cutoff.
This evaluation occurs before every branch-specific early return, including a
`gadget_small` decomposition shortcut. Such a shortcut may select a different
producer, but it may not bypass the instance's bound evaluation, owner-schema
construction, or output validation.

`SmallMatrix` asserts only coefficient boundedness. `Preimage` additionally
retains the exact producer relation already represented by the graph. A
generic bounded matrix cannot be relabeled as a preimage. Neither type can be
converted to an ordinary `Mat` merely to access multiplication.

### Matrix multiplication node

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
`gpu_matrix_mul_small_rhs(out, lhs_eval, rhs_small, residency_budget_bytes,
allocation_report)`. The runtime passes one already-sharded compact RHS with
`C <= W` columns. The wrapper validates contexts, device, dimensions, the exact
residency requirement, and that `lhs_eval` is full DCRT evaluation format. It
allocates one full-evaluation `m x C` output and launches asynchronously.

The primitive has no independent column, K, or limb tiling. A single in-place
workspace holds exactly `L*K*C*N` evaluation words for the runtime shard. One
enqueue sequence unpacks every coefficient, embeds it independently modulo all
CRT primes, transforms every local RHS polynomial in place, and accumulates the
full K dot product. There is no host loop over RHS columns, K ranges, or limb
ranges, no second expanded buffer, and no multiply-time compact D2D copy.
Larger logical matrices are divided at producer/load boundaries or exposed by
lifetime-bound zero-copy views of existing compact shards.

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
host-side blocking loop. Existing release-stream fencing remains the only
permitted deferred-resource fence. Runtime calibration derives the only widths,
`W_gpu0` and `W_nonzero`; the primitive does not derive another tile. At context
setup, cache a per-device budget using `MXX_GPU_VRAM_PERCENT`, which
accepts an integer from 1 through 100 and defaults to 80. Do not maintain
separate byte-budget, allocator-headroom, column-width, K-tile, or limb-wave
environment controls. Fleet-wide calibration and scheduling follow
`docs/plans/fleet-wide-gpu-column-sharding.md`.

The normal success path also contains no `cudaStreamSynchronize`. Error
unwinding uses a completion-event fence; only failure to create or record that
safety event may fall back to a stream fence so queued work cannot outlive
temporary storage returned to the allocator.
The enqueue sequence uses scalar launch arguments and one final completion event,
not C-sized device pointer/event arrays, so its metadata remains bounded.

In `crates/primitives/src/env.rs`, add exactly one sampler control,
`MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS`, parsed as a positive integer by
`gpu_preimage_max_tile_attempts()`.  An unset variable defaults to `64`,
deliberately replacing the current unbounded retry loop; a zero, negative, or
malformed value is an error, not a fallback and not an alias.  This setting is
primitive-owned and applies per target-column tile.

Let `L` be active CRT limbs, `N` the ring dimension, `m` the lhs rows, `K` the
RHS rows, `C <= W` the current runtime-shard columns, `b_s = 8*(1 +
magnitude_bytes)` compact coefficient bits, `w_q` the bytes per evaluation
word, and `event_metadata_bits` the measured event/launch metadata. The measured
per-device primitive peak bound is:

```text
peak_bits <= 8*L*m*K*N*w_q                  // resident lhs evaluation
           + K*C*N*b_s                       // compact runtime shard
           + 8*L*m*C*N*w_q                   // one full shard output
           + 8*L*K*C*N*w_q                   // one in-place shard RHS eval
           + event_metadata_bits.
```

The implementation reports each term and overhead from the allocator probe and
rejects the shard before launch unless the complete formula fits the device
budget. The full logical column count never enters this allocation: only the
runtime-selected `C <= W` shard does. Delete
`MXX_MUL_DECOMPOSE_COLUMN_CHUNK_WIDTH` and both
`mul_decompose` implementations and timing scaffolding; no old environment
variable may silently control the new operation.

- equal ring dimension and modulus parameters;
- `lhs.columns == rhs.rows`;
- a non-negative checked RHS bound;
- an output shape of `lhs.rows x rhs.columns`; and
- no implicit decomposition, sampling, scaling, or relation conversion.

Let `D = ceil(crt_bits / base_bits) * crt_depth`, `N` be the ring dimension,
and `C <= W` be one runtime shard. The full logical matrix may have many such
shards, but only one shard per active device is expanded at a time.

| Resident object | Order | Status | Required handling |
| --- | --- | --- | --- |
| Full-DCRT lhs evaluation (`m x K`) | `O(crt_bits * crt_depth * N * mK)` bits | Allowed when `mK = O(D)` | Keep the production evaluation lhs resident. |
| Full-DCRT output (`m x C`) | `O(crt_bits * crt_depth * N * mC)` bits | Allowed when `mC = O(D)` | Allocate the one full output required by the API; do not make a second full output. |
| Compact logical RHS (`K x total_C`) | `O(b_s * N * K*total_C)` bits | Allowed only as separately owned compact runtime shards | Producers and artifact loaders create `C <= W` shards; multiplication never joins them. |
| Expanded DCRT/NTT shard (`K x C`) | `O(crt_bits * crt_depth * N * KC)` bits | Allowed exactly when the complete shard peak fits the role budget | Expand all `L`, `K`, and `C` in one primitive call, then release by event. |
| Decomposed full-DCRT RHS plus compact RHS | Expanded term plus compact term | Forbidden | No producer or serializer may materialise both representations. |
| NTT workspace for one shard | `8*L*K*C*N*w_q` bits | Allowed exactly when `8*L*K*C*N*w_q <= remaining_budget_bits` | Runtime calibration bounds only `C=W`; the primitive always uses full K and L. |

The shape-general peak formula reports the resident compact source allocation
and exact `L*K*C*N` expanded workspace. Define
`remaining_budget_bits = configured_budget_bits - (lhs_bits + compact_rhs_bits
+ output_bits + event_metadata_bits)` and reject a configuration unless it is
non-negative and satisfies `8*L*K*C*N*w_q <= remaining_budget_bits`. For a
`D x D` logical small RHS, the runtime chooses W from the measured role budget
and executes `ceil(D / W)` shard waves. It never creates an expanded allocation
covering columns outside the current shard. If a single-column shard cannot
fit, fail with a resource error; do not introduce hidden K/limb tiling.

## Kernel-level optimization contract

These optimizations apply to `mul_small_rhs`, direct compact gadget
decomposition, and bounded preimage packing. The historical name
`mul_decomposed` describes only the operation being replaced; neither
`mul_decomposed` nor `apply_preimage` is restored as an API.

### Compact expansion and NTT

For an NTT stage of size `N`, exactly `N / 2` butterflies are launched. The
grid is therefore:

```text
ceil((N / 2) / threads_per_block)
```

The compact expansion kernel combines signed-magnitude decoding, independent
embedding into the active CRT limb, and the forward-NTT twist. It writes the
twisted evaluation workspace once; a separate unpack output and twist pass are
not materialized. The memory-oriented compact kernels may process two through
four coefficients per thread when alignment and a tail-safe loop permit it.
The staged NTT butterfly kernel stays simple and does not acquire an unrelated
grid-stride packing loop.

The implementation keeps the existing `uint64_t` evaluation workspace. It does
not introduce a second 32-bit or packed-width NTT implementation. The two bit
reversals may be replaced with a DIF/DIT pairing only after profiling shows they
remain material after the low-risk changes. A shared-memory hierarchical NTT is
also profile-gated and is not part of the baseline implementation.

### Exact lazy dot accumulation

The matrix dot-product kernel may defer modular reduction across the full K only
when the host proves the unsigned 128-bit accumulator cannot overflow. For a
limb modulus `q` and `t` accumulated products, including a previously reduced
accumulator, the required bound is:

```text
t * (q - 1)^2 + (q - 1) <= 2^128 - 1
```

This predicate is evaluated with checked host arithmetic for every limb in the
wave. If every limb is safe, the lazy kernel accumulates all `t` products in an
unsigned 128-bit value and reduces once. Otherwise the operation uses the
general per-term modular kernel. No supported modulus, shape, tail tile, or K
value may rely on the commonly expected `q < 2^52` and `t <= 180` without
checking it. Those values are merely a sufficient common case whose sum is
below `2^112`.

### Preimage hard-cutoff check and pack

The checked conversion used by bounded preimage sampling is named for its
semantics, for example `try_pack_preimage_hard_cutoff_tile`. It accepts a
coefficient-domain candidate and performs the exact inclusive test
`abs(coefficient) <= B` while writing the canonical compact sign-and-magnitude
payload. Rejected attempt storage is private and fully overwritten before the
next attempt; no rejected payload becomes a runtime value or artifact. Only the
acceptance decision crosses the device-to-host boundary during rejection
sampling.

The conversion selects one of two exact CRT plans from the active moduli and
bound. The plan and its constants are prepared once for the context/bound and
reused across attempts.

The single-anchor fast path selects a limb `s` satisfying:

```text
q_s > 2B
```

For each coefficient it takes the centered lift `y` of the anchor residue,
checks `abs(y) <= B`, verifies every other active residue equals `y mod q_i`,
and packs `y` in the same kernel. The verification of all limbs is mandatory:
an anchor residue alone does not prove that the full CRT value is small.

If no single modulus is large enough, choose the smallest deterministic subset
`S` whose product `P_S` satisfies `P_S > 2B`. Reconstruct the centered `y`
modulo `P_S`, check the bound, verify it against every remaining active limb,
and pack it. The subset reconstruction is quadratic in `|S|`; reducing its
multiword result into the remaining limbs makes the general bound
`O(|S|^2 + |S|L)`, which is effectively linear in `L` when the selected subset
is small. Failure to find a representable plan is an explicit configuration or
width error, never acceptance based on an incomplete CRT check.

Direct gadget decomposition already produces bounded compact digits without
first creating a full DCRT matrix. That path is the trusted-by-construction
producer and remains direct. Do not add a generic `pack_trusted_bounded` API
until a real production caller owns the required centered-coefficient contract.
Similarly, do not add an ambiguous full-DCRT `compact_from_matrix` conversion.
Artifact decoding checks the expected schema and canonical compact payload; it
does not rerun CRT reconstruction or the preimage norm computation.

### Required correctness and performance evidence

Focused GPU tests cover 32-, 40-, and 51-bit CRT primes; zero, positive,
negative, `B`, `-B`, and `B + 1`; single-anchor acceptance and cross-limb false
positive rejection; partial-CRT reconstruction; lazy and fallback dot kernels;
nondivisible packing work; full-K/full-limb shard multiplication and tail shards; accepted
preimage relation and exact cutoff; and the rule that rejected candidates are
never published. Fleet tests additionally cover the same cases across unequal
GPU-0 and nonzero-device shard widths.

Performance comparison uses the same parameters and records candidate
generation, cutoff-and-pack, acceptance wait, compact expansion/twist, bit
reversal, NTT stages, dot accumulation, launch overhead, peer copies,
per-device peak VRAM, and total wall latency. Timed GPU boundaries wait on the
specific completion event or result owner. Host enqueue duration and an
unrelated generic matrix benchmark are not accepted as small-RHS execution
latency. DIF/DIT or shared-memory NTT work starts only if this breakdown shows
that NTT or bit reversal remains a dominant cost.

### Final compact-path comparison

The predecessor tiled implementation and the final single-W implementation were
measured on one NVIDIA GeForce RTX 4080 SUPER (16,376 MiB, driver 580.173.02,
CUDA toolkit 13.1). The baseline is commit
`8306aeac39c200068dbe710d6d52d24f47208071`. Both trees used the same public
`gadget_decompose(true)` followed by `multiply_small_rhs` sequence, the same
completion waits, and five samples. Parameters were `N=1024`, CRT depth 4, 28
CRT bits, 8 base bits, two LHS rows, four target rows, and 32 target columns.
The final path therefore used `C=W=32`, `K=16`, and `L=4` in one primitive
call. Median results were:

| Metric | Baseline | Current implementation | Change |
|---|---:|---:|---:|
| Compact decomposition | 1.009475 ms | 1.086690 ms | 7.65% slower |
| Small-RHS multiplication | 1.896549 ms | 0.810891 ms | 2.339x faster |
| End to end | 2.940860 ms | 1.972733 ms | 1.491x faster |
| Incremental default-mempool high-water | 1,048,576 B | 15,466,496 B | 14.75x larger |

The final allocation report recorded a 1,048,576-byte compact RHS, a
2,359,432-byte full output owner, a 16,777,216-byte expanded RHS workspace, and
a 21,365,048-byte modeled complete high-water. The allocator reported
21,364,928 bytes at its absolute high-water after a 5,898,432-byte baseline.
The incremental values in the table are CUDA default asynchronous-pool
counters, not total physical-device usage, so they compare allocator-visible
incremental residency only. Runtime capacity calibration separately combines
this incremental cost with allocator-aware physical residency as specified in
`docs/plans/fleet-wide-gpu-column-sharding.md`.

Raw baseline `(decomposition, multiplication, end-to-end)` samples in seconds
were:

```text
(0.001009475, 0.001931164, 0.002940860)
(0.001113571, 0.001874608, 0.002988449)
(0.001056183, 0.001900767, 0.002957110)
(0.001001851, 0.001876210, 0.002878221)
(0.000991682, 0.001896549, 0.002888471)
```

Raw final samples were:

```text
(0.000952998, 0.000803637, 0.001757206)
(0.000972625, 0.000810891, 0.001784107)
(0.001347283, 0.000722413, 0.002070477)
(0.001216085, 0.000859271, 0.002075897)
(0.001086690, 0.000885501, 0.001972733)
```

The exact build command in both trees was:

```text
cargo bench -p mxx-primitives --bench bench_small_rhs_gpu --features gpu --no-run -j8
```

The resulting binary was executed five times without changing its environment.
The benchmark entry point is `crates/primitives/benches/bench_small_rhs_gpu.rs`.

```rust
lhs.mul_small_rhs(rhs)
```

Use a sealed `BoundedRhs` trait, or an equivalent closed typed interface, so
only `SmallMatrix` and `Preimage` are accepted. Do not accept an ordinary
`Mat`, a raw matrix identifier, or an unvalidated coefficient buffer.

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
this query.  Validate the complete percentage-derived budget against allocator
high-water measurements; do not introduce a second fixed headroom budget.

Let `T_candidate`, `T_perturb`, and `T_check` name the actual candidate,
perturbation, and centered-norm-check owners.  Count each named owner exactly
once from either the shared full-matrix allocation query or its actual raw
allocation; a distributed candidate is one owner, not one global allocation
per GPU, and it must not be included again inside `check_scratch`.  Include
residual, `z_hat`, perturbation blocks, target tile, any packed staging buffer,
the persistent public/trapdoor/target values, the persistent compact
all-column destination, the acceptance flag, and deterministic event handles
in `sampler_peak_bits`.  Define
`sampler_remaining_budget_bits = configured_budget_bits -
(persistent_public_bits + persistent_trapdoor_bits + persistent_target_bits +
compact_destination_bits + acceptance_flag_bits + sampler_event_bits)` and
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

Every producer declares its output type and exact inclusive bound:

| Producer | Output semantic type | Bound |
| --- | --- | --- |
| balanced gadget decomposition | `Preimage` when relation-bearing, otherwise `SmallMatrix` | the exact balanced-digit endpoint |
| unsigned gadget decomposition | `Preimage` when relation-bearing, otherwise `SmallMatrix` | `beta - 1` |
| hash gadget source/decomposition | `SmallMatrix` | decomposition-mode bound |
| trapdoor preimage sampling | `Preimage` | caller-validated sampler cutoff |
| artifact import | descriptor-selected type | descriptor bound, checked during decode |

The sampler cutoff is inclusive. A caller-selected tight cutoff below the
authoritative default preimage cutoff is rejected before sampling or any retry
attempt begins; it is never silently treated as the default.

For power-of-two gadget base `beta = 2^base_bits`, keep the existing balanced
tie convention. The exact inclusive balanced-digit bound is `beta/2`; the
unsigned bound is `beta-1`. Do not infer a bound
by scanning a payload after import, and do not substitute a gadget-digit bound
for a sampled-preimage cutoff.

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
   one-call production path, compact RHS bytes, exact `L*K*W*N` workspace, and
   full-output persistence; remove estimates that count Preimage as full DCRT.

### Full logical artifact and all-column compact owner

No integration tests are part of this plan.  Each implementation slice must
first pass `cargo +nightly fmt --all`, focused unit tests for its
crate, and CPU/GPU `--no-run` compilation as applicable.  Then run all
workspace unit tests with and without GPU features using `--lib` unit-only
commands.  The only permitted exclusions are individual, explicitly named
`mxx-gadgets` unit tests; each excluded test must have its own recorded
`>20`-minute elapsed run as evidence.  Do not exclude a whole crate or command
without naming the tests.  GPU tests run outside the sandbox and are repeated
with identical commands (3--5 repetitions for smoke/correctness, and the
repository-required repeated run for synchronization-sensitive tests).

```text
SmallMatrixArtifact {
    schema,
    semantic_kind,
    total_rows,
    total_columns,
    canonical_payload_locator,
}

GpuSmallMatrix {
    schema,
    rows,
    columns,
    device,
    compact_device_owner,
    ready_event,
}
```

Both objects represent all `K*C*N` coefficients. `GpuSmallMatrix` stores each
coefficient in the canonical sign-plus-magnitude width `s_B`, not once per CRT
limb. This all-column compact allocation is intentional and is the mechanism
that makes the quadratic matrix resident. It contains no full-DCRT limb array,
NTT representation, or host matrix clone.

Final gates are CPU/GPU no-run builds with no new warnings attributable to this
change, nightly formatting, all allowed unit tests, repeated GPU tests,
estimator consistency, the measured
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
