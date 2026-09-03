# Bounded Small-RHS Multiplication with Tiled DCRT Expansion

## Status

This document is the implementation plan for integrating MachinaIO/mxx pull
request 150 into the current branch. Its all-column compact small-matrix owner
is retained. Only representations expanded into DCRT/NTT form, and temporary
full-DCRT preimage target/candidate matrices, are column-tiled.

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
- redesign Exponent-LUT's application-specific noise authority;
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
- `mxx-exponent-lut` owns its program-specific use and application-specific
  simulation;
- `mxx-noise-simulator` depends only on lower-level IR and must never depend on
  Exponent-LUT; and
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

Add one node:

```text
MatrixMulSmallRhs(lhs: Matrix, rhs: SmallMatrix | Preimage) -> Matrix
```

Validation requires:

- equal ring dimension and modulus parameters;
- `lhs.columns == rhs.rows`;
- a non-negative checked RHS bound;
- an output shape of `lhs.rows x rhs.columns`; and
- no implicit decomposition, sampling, scaling, or relation conversion.

This node means ordinary ring-matrix multiplication. Its bounded RHS changes
representation and implementation strategy, not its algebra.

### DSL API

Expose one conceptual API:

```rust
lhs.mul_small_rhs(rhs)
```

Use a sealed `BoundedRhs` trait, or an equivalent closed typed interface, so
only `SmallMatrix` and `Preimage` are accepted. Do not accept an ordinary
`Mat`, a raw matrix identifier, or an unvalidated coefficient buffer.

Delete old `Preimage::as_mat`, scale-by-one erasure, `apply_preimage`, expanded
preimage multiplication, and `mul_decomposed` paths after all call sites have
migrated. No compatibility wrapper or legacy decoder remains.

### Producers and exact bounds

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

## Runtime data model

### Full logical artifact and all-column compact owner

An `O((log q)^2)` bounded matrix is represented by one full compact artifact
and one full compact CPU/GPU owner:

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

Column tiling applies only when a bounded subset is expanded to DCRT/NTT form
for multiplication, or when a full-DCRT target/candidate is loaded during
preimage sampling. It does not split the persistent compact small matrix.

The runtime value keeps validated wire semantics separately from the shared
backend owner:

```text
RuntimeValue::SmallMatrix(...)
```

Both `ConcreteSmallMatrix` and `ConcretePreimage` use this storage variant,
but runtime metadata retains which wire kind was validated. A kind mismatch is
an error, not a conversion attempt.

At every externally supplied input, lazy/staged artifact materialization, and
family-element boundary, compare the compact owner's complete metadata against
the validated concrete wire schema: modulus parameters, ring dimension, rows,
columns, inclusive coefficient bound, context/placement identity where
applicable, and semantic kind. Checking only the runtime enum variant is
insufficient. Reject a mismatch before executing `MatrixMulSmallRhs`; noise
lowering must never use a smaller declared bound than the value owner carries.

### Artifact identity

Artifact identity includes:

- semantic kind (`SmallMatrix` or `Preimage`);
- complete matrix parameters and dimensions;
- the inclusive coefficient bound;
- canonical encoding version implied by the current schema; and
- canonical bytes for all columns.

Artifact identity excludes:

- `C_t`, `K_t`, and `ell`;
- GPU count, placement, and stream assignment;
- cache location; and
- the order in which DCRT workspace tiles were evaluated.

No old artifact version is accepted. The migration replaces rather than
negotiates with old payloads.

## Canonical coefficient encoding

### Signed coefficient representation

Each coefficient uses:

```text
sign: u8
magnitude: exactly magnitude_bytes little-endian bytes
```

where:

- `sign = 0` means zero and requires an all-zero magnitude;
- `sign = 1` means a positive nonzero value;
- `sign = 2` means a negative nonzero value; and
- `magnitude_bytes = max(1, ceil(bit_length(B) / 8))`.

Reject negative zero, noncanonical sign values, noncanonical width, values
outside `[-B, B]`, overflow, trailing bytes, and schema mismatches before a
large allocation.

### Canonical row-major payload

The canonical coefficient index is:

\[
  ((row \cdot columns + column) \cdot N + coefficient).
\]

This is the updated PR's canonical compact payload. CPU and GPU codecs use the
same full-matrix layout. Preimage sampling kernels may scatter an accepted
target-column tile into its global columns of this all-column destination; the
persistent artifact is never defined by the temporary tile order.

The canonical header is, in order:

```text
magic:               [u8; 4] = "SMR1"
semantic_kind:       u8      // 0 = generic SmallMatrix, 1 = Preimage
rows:                u64 little-endian
columns:             u64 little-endian
ring_dimension:      u64 little-endian
bound_length:        u32 little-endian
bound:               bound_length canonical little-endian magnitude bytes
magnitude_bytes:     u32 little-endian
coefficient_count:   u64 little-endian
payload:             coefficient_count canonical signed coefficients
```

The non-negative bound uses exactly
`max(1, ceil(bit_length(B)/8))` bytes; zero is one zero byte and a nonzero
bound has no high zero byte. `coefficient_count` must equal
`rows*columns*N` under checked arithmetic. This is the only accepted format;
`SMR1` is a format discriminator, not a version-negotiation mechanism.

The artifact layer validates the complete expected schema before decoding
payload coefficients. Existing artifact-store hashing remains authoritative;
the compact codec does not add a second commitment or require recomputation by
a trusted local caller.

## Primitive/backend API

Introduce one backend-owned all-column compact matrix abstraction:

```text
small_matrix_from_bytes(schema, semantic_kind, bytes) -> SmallMatrix
small_matrix_to_bytes(value, schema, semantic_kind) -> bytes
multiply_small_rhs(lhs_eval, rhs_small) -> Matrix
sample_preimage_small(public_input, target, bound) -> SmallMatrix
```

The CPU backend may use a compact host representation. The GPU backend uses a
distinct `GpuSmallMatrix` owner; it must not put compact-state flags or a host
clone inside `GpuDCRTPolyMatrix`.

`PolyTrapdoorSampler::M` implements the common bounded-RHS primitive contract.
Its preimage operation returns the compact associated owner directly. Delete
the public expanded-preimage path and duplicate sampler-side compatibility
traits. Keep a separately meaningful full-matrix primitive such as
`preimage_extend` only if its algebra genuinely requires full-matrix output;
it must not route through or masquerade as the compact preimage API.

## Persistent GPU matrix descriptors

Adopt the updated PR's persistent descriptor design for ordinary DCRT
matrices. Each local limb has a device-resident descriptor containing the
base pointer, stride/offset, and coefficient width needed by kernels. Allocate
and initialize descriptors with the matrix owner, account for their exact
bytes, and release them with the owner's event lifetime.

Small-RHS multiplication must consume these persistent descriptors directly.
It must not rebuild and copy pointer, stride, width, or global-limb arrays for
every operation. Kernels derive local addresses from the descriptor and
checked partition-local indices.

Because this changes generic `GpuMatrix` allocation and accounting, implement
it as an independently validated primitive checkpoint. Verify active-level
mapping and nonuniform multi-GPU limb partitions before depending on it from
the compact operation.

## Small-RHS multiplication with tiled DCRT expansion

### Preconditions

The left-hand side is a full DCRT matrix in evaluation form. The output is one
full `m x C` DCRT evaluation matrix when that size is within the configured
budget. The RHS remains compact until a bounded workspace tile is embedded
for a specific CRT-limb wave.

If the full output itself cannot fit, use the existing bounded outer
family/output wave or return a resource error. Do not silently change output
dimensions or spill through an undocumented path.

### Execution schedule

For each target-column range `J` with `|J| = C_t`:

1. select `S[:, J]` from the already resident all-column compact owner;
2. retain `A` and the complete compact `S` through an owner event;
3. for each limb wave of width `ell` and reduction tile of width `K_t`:
   1. unpack centered signed coefficients;
   2. embed each coefficient independently modulo each active CRT prime;
   3. apply an in-place batched NTT for that prime;
   4. multiply the matching `A` tile and accumulate into `Y[:, J]` modulo the
      prime;
4. record one completion event covering every launch for `J`;
5. make later consumers wait on that event; and
6. release or reuse the expanded workspace only after the event.

Use one in-place RHS evaluation workspace of shape
`ell x K_t x C_t x N`. The output tile is the accumulator; do not allocate a
second full output or a second RHS wave. The first reduction tile initializes
the output tile, and later tiles add into it.

The Rust/FFI boundary is one operation for the complete all-column compact RHS,
not one call per column, limb, or polynomial. A single CUDA call may enqueue
bounded internal column/reduction/limb tiles, but it must not wait for device
completion before returning. Kernels process all columns and limbs in the
current tile rather than launching once per element.

### Arithmetic correctness

Compact coefficients are signed integers, not residues under a single shared
prime. Embed the same integer separately under every CRT modulus. NTT results
cannot be reused across distinct CRT primes. Products and accumulators use the
existing checked modular arithmetic for each prime; unsupported widths or
overflow risks return errors.

The compact kernel uses the canonical row-major index:

\[
  ((rhs\_row \cdot rhs\_columns + rhs\_column)N + coefficient).
\]

## Column-streamed preimage sampling

Let the target have `C` columns. Sampling executes:

```text
for J in target.column_ranges(C_t):
    load target[:, J]
    sample a relation-valid preimage for all rows and columns in J
    check the full-CRT centered coefficient bound
    pack the accepted tile into columns J of one all-column compact destination
    release target, candidate, and scratch through events
```

Rows are not independently tiled when the preimage equation couples them. A
column tile contains every row required by that relation. If one full-row,
one-column candidate cannot fit, return a resource error rather than changing
the relation.

Only the all-column compact destination, one candidate tile, perturbation
state, norm-check scratch, and one target tile may coexist. A full expanded
`K x C` candidate is forbidden.

The authoritative cutoff is inclusive and is checked in centered full-CRT
coordinates before packing. A rejected draw is released safely before buffer
reuse. The sampler may copy only the small acceptance result to the host at
the adaptive retry boundary. After the configured maximum attempts for a
column tile, return an error naming its global range and exact attempt count.

The conservative `default_preimage_cutoff` is also the minimum accepted API
bound. If the requested inclusive bound is smaller, reject the request before
sampling with `PreimageBoundTooSmall { requested, minimum }`. The bounded retry
loop is used only for requests satisfying
`requested >= default_preimage_cutoff`; those requests either produce a
relation-valid sample within the requested cutoff or return the tile-specific
exhaustion error after exactly the configured number of attempts.

### Determinism independent of streaming

Sampling randomness must be indexed by stable logical coordinates, for
example:

```text
(seed, sampler/domain tag, target identity,
 global column, row, coefficient, attempt)
```

It must not be consumed from a stream whose position depends on `C_t`, GPU
count, scheduling, or retry activity in a different column range. The same
inputs therefore produce the same full canonical output for every legal wave
configuration.

## Decomposition and hash-source domain

Add an explicit producer hook equivalent to
`sample_hash_gadget_source`. The GPU implementation may create the source in
coefficient form when decomposition consumes coefficients immediately. This
avoids sample-to-NTT followed by NTT-to-coefficient conversion without changing
the hash seed, domain-separation tag, or logical sampled matrix.

Gadget decomposition writes directly into the compact row-major owner. It
must not first produce a full DCRT decomposition and then serialize it.

Tests must establish that coefficient-source and evaluation-source routes
produce the same logical source and decomposition. This optimization remains
a representation change, not a new hash construction.

## DCRT domain rules

Adopt the updated PR's domain discipline as a generic primitive checkpoint:

- expose batched in-place NTT for complete matrix batches;
- make coefficient versus evaluation form explicit in every constructor and
  sampling result;
- normalize add/sub operands before combining them;
- preserve the correct result format in low-level CUDA arithmetic;
- load all-evaluation inputs without a transform;
- combine all-coefficient inputs before one batched transform;
- separate and then combine mixed-domain inputs; and
- generate constant gadget evaluation values directly when valid.

Matrices remain in evaluation form by default. No comparison, concatenation,
addition, or multiplication consumes mixed formats silently. These changes
have a wider blast radius than small-RHS multiplication and must not be
imported by replacing whole files without focused equivalence tests.

## Asynchrony, waiting, and lifetime

Adopt owner-local completion waits from the updated PR:

- a GPU wrapper that only launches device work returns without blocking;
- every output owner records the event covering its final write;
- consumers wait through stream events, not host/device-wide synchronization;
- benchmarks and host serialization call `wait_until_ready()` only on the
  owner whose result they observe;
- normal execution uses neither `cudaDeviceSynchronize` nor
  `cudaStreamSynchronize`; and
- a stream wait is permitted only on an allocation/error path where unfinished
  asynchronous initialization must be fenced before safe cleanup.

Allocation streams initialize persistent descriptors once. Other streams wait
on the owner's allocation event. Failure paths release partially created
owners without racing descriptor initialization.

## Memory contract

### Multiplication peak

Let `E` include checked event, descriptor, allocator, and fixed headroom bytes.
The multiplication peak is bounded by:

\[
\begin{aligned}
  Peak_{mul} \le{}& Bytes(A_{eval})
   + K C N s_B \\
   + Bytes(Y_{eval}) \\
   &+ \sum_{i\in limb\ wave} K_t C_t N w_i
   + E.
\end{aligned}
\]

The `K*C*N*s_B` term is the intended resident all-column compact owner. There
is no `L*K*C*N*w` all-column expanded DCRT/NTT RHS term: only the selected
`K_t*C_t` expanded workspace is materialized for each limb wave.

The runtime selects positive `C_t`, `K_t`, and `ell` satisfying the exact byte
budget after subtracting the measured allocations for the left-hand side, the
all-column compact RHS, output, descriptors, and reserve. Defaults begin with
one expanded target column, then increase parallelism only while the exact
allocation remains within the budget.

### Sampling peak

The sampling peak includes exactly once:

- persistent public/trapdoor inputs that truly remain resident;
- one expanded target column tile;
- one expanded relation-valid candidate chunk;
- perturbation and norm-check scratch;
- one all-column compact accepted-result destination;
- acceptance/event metadata; and
- allocator/headroom reserve.

It excludes a full expanded target and full expanded preimage. The compact
accepted preimage is intentionally resident as one all-column owner and is
counted exactly once as `K*C*N*s_B`.
Each owner is counted once even if its CRT limbs are distributed internally.

### Allocation queries

The side-effect-free allocation query and actual allocation must share one
checked planner. Reports separate data, auxiliary descriptor, deterministic
event, workspace, and reserve bytes. The query may not change devices, advance
streams, allocate memory, or create events. Physical acceptance additionally
uses allocator high-water measurements because fragmentation and opaque CUDA
state cannot be predicted exactly.

## Configuration

Reuse existing environment variables when their meaning is identical. Add a
new variable only when no existing control has the same production meaning.
Primitive-owned controls live in `crates/primitives/src/env.rs`.

The required controls are:

```text
MXX_MUL_SMALL_RHS_TILE_COLUMNS
MXX_MUL_SMALL_RHS_K_TILE
MXX_MUL_SMALL_RHS_LIMB_WAVE
MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS
```

Delete `MXX_MUL_DECOMPOSE_COLUMN_CHUNK_WIDTH` with the legacy decomposition
path. Although both old and new controls mention columns, the old variable
governs a per-column public operation, while the new control governs only the
internal DCRT/NTT expansion or sampler tile of one all-column compact
operation. Reusing it would silently preserve the obsolete execution
contract. The new column-tile default is `1`.

`MXX_GPU_PREIMAGE_MAX_TILE_ATTEMPTS` defaults to `64` when unset. All explicit
values must be positive checked integers. Invalid values are errors, not
fallbacks. Tile and wave settings affect peak memory and performance only;
they never affect artifact bytes or mathematical results.

## Benchmark and estimator contract

The benchmark must call the production functions and reproduce production
load, compute, event wait, and store boundaries.

It must not reduce rows, columns, CRT depth, ring dimension, or gadget digit
count. For an operation with multiple target columns:

1. construct/load the complete production compact owner, then benchmark one
   internal expanded column tile with the exact full row and reduction
   dimensions;
2. record raw chunk latency, chunk width, and maximum parallelism;
3. compute the full operation time as the sum over the actual number of column
   chunks; and
4. when `C_t = 1`, report the explicit one-column latency and
   `one_column_latency * C` total.

This follows the repository measurement contract: latency is one wave, while
total time includes every required wave, chunk, and slot. Setup, preprocess,
and online evaluation must charge the operation to the phase where the
corresponding public-key or encoding computation actually runs.

Reports distinguish:

- transmitted data such as hash keys and required RHS ciphertexts;
- deterministic cache data derivable from transmitted/setup inputs;
- persistent storage bytes for the full compact artifact;
- resident VRAM for the all-column compact owner;
- expanded workspace bytes;
- output bytes;
- raw stage latencies;
- chunk/wave counts;
- total time; and
- allocator high-water memory.

No estimator may count a preimage as both compact storage and expanded DCRT
storage. No enqueue-only timing is accepted: wait on the measured result owner
at the observation boundary.

## Noise-simulator scope

Make only the changes required for exhaustive matching and existing semantics:

- recognize `MatrixMulSmallRhs`;
- apply the existing ordinary bounded-RHS multiplication transfer using the
  declared inclusive coefficient bound and exact dimensions;
- preserve preimage relation metadata where current lowering requires it; and
- add no decomposition or sampling error term to the multiplication node.

Do not redesign the generic simulator, add a second graph evaluator, expand
its reporting, or make it an Exponent-LUT acceptance authority. Exponent-LUT's
application-specific simulator remains responsible for its program operations
and refresh threshold.

For an input error matrix with per-coefficient bound `E_A`, the exact
right-action transfer is expressed through

\[
  \gamma(S) = \max_j \sum_{k=0}^{K-1} \lVert S_{k,j}\rVert_1,
  \qquad
  E_Y \le \gamma(S)E_A.
\]

When only the declared entrywise bound is available, use

\[
  \gamma(S) \le KNB,
  \qquad
  E_Y \le KNB E_A.
\]

Do not multiply an error bound by `q`, `q/q_t`, or a raw encoded scalar when
the actual action is bounded by a gadget-decomposition or monomial norm.

## Cache semantics

Cacheability is determined by value dependence, not by representation.

- Data depending only on public setup, public keys, deterministic hash inputs,
  or the separately allowed secret key `t` may be cached under the current
  working assumption.
- Data depending on a not-yet-selected encoding secret `s`, or on a value with
  `s` already multiplied from the left, is not preprocess-cacheable.
- A decomposed helper public key may be cached when its complete inputs are
  cacheable.
- A runtime cache entry records complete logical identity and schema, never a
  wave-local node number or tile width.

Storage accounting labels cache bytes separately from bytes that must be sent.
Online evaluation consumes reusable RHS public-key decompositions from cache
and must not recompute the corresponding public-key program for each encoding.

## File ownership and implementation slices

Implementation is divided into independently reviewable checkpoints. Start
each checkpoint from the corresponding PR implementation whenever it applies.
An entire PR file may be adopted when its current-branch ownership and behavior
match this plan; otherwise transplant the largest compatible hunks and keep the
local changes limited to explicit semantic or integration differences.

### 1. IR and DSL contract

Primary ownership:

- `crates/ir-core/src/` type, node, validation, family, and artifact modules;
- `crates/dsl/src/` typed construction APIs.

Add bounded wire schemas, `MatrixMulSmallRhs`, sealed typed APIs, full shape
validation, family preservation, and row-major artifact metadata. Delete
type-erasing and legacy APIs only after repository-wide call-site migration is
prepared.

### 2. CPU compact foundation

Primary ownership:

- `crates/primitives/src/matrix/` CPU matrix modules;
- sampler and decomposition traits where already owned.

Add the all-column compact host owner, canonical row-major codec, exact bound checks,
trusted ordinary multiplication, and direct compact producer paths. Use
existing matrix arithmetic as the correctness reference.

### 3. Runtime and artifacts

Primary ownership:

- `crates/runtime/src/backend.rs`;
- `crates/runtime/src/backend/poly.rs`;
- `crates/runtime/src/backend/poly_gpu.rs`;
- executor, artifact, session, placement, and runtime-value modules.

Add the shared all-column bounded owner, whole-artifact loading and storing,
`RuntimeValue::SmallMatrix`, placement rules, and node execution. GPU-only Rust
stays in `poly_gpu.rs`. Preserve full logical wire identity through indexed
families and artifacts.

### 4. Generic GPU matrix foundation from PR 150

Primary ownership:

- GPU DCRT Rust modules;
- `crates/primitives/cuda/include/matrix/` declarations;
- `crates/primitives/cuda/src/matrix/` implementations.

Port persistent descriptors, owner-local waits, allocation accounting, batched
NTT, explicit domain handling, and event-safe allocation cleanup. Validate this
slice before compact multiplication depends on it.

### 5. GPU compact owner and multiplication

Add the all-column compact device owner, canonical row-major codec, one
whole-owner FFI operation whose implementation internally tiles expansion,
checked per-prime embedding, bounded NTT workspace, accumulation, and event
lifetime. Remove per-operation descriptor uploads and full-DCRT RHS
materialization.

### 6. Column-streamed preimage and decomposition producers

Add target-tile loading, stable-coordinate randomness, full-row candidate
tiles, bounded retry, full-CRT centered checks, immediate packing into the
proper columns of one all-column compact destination, and one final whole-owner
store. Add coefficient-domain hash sources and direct compact gadget
decomposition.

### 7. Lowering and application migration

Update only required correctness/noise exhaustive matches. Then migrate BGG,
gadgets, WE, IO, and Exponent-LUT call sites according to their actual RHS
semantics. Ordinary matrix multiplication remains ordinary; only declared
bounded RHS values use `mul_small_rhs`.

The currently known call-site audit includes:

- `crates/noise-simulator/src/eval.rs`;
- `crates/bgg/src/attribute_encoding.rs`;
- `crates/bgg/src/boolean.rs`;
- `crates/bgg/src/encoding.rs`;
- `crates/bgg/src/lwe_lookup.rs`;
- `crates/bgg/src/masked_decoder.rs`;
- `crates/bgg/src/naive_vec.rs`;
- `crates/bgg/src/noise_refresh.rs`;
- `crates/bgg/src/public_key.rs`;
- `crates/bgg/src/slot_operation.rs`;
- `crates/bgg/src/tall_encoding.rs`;
- `crates/bgg/src/wee25_opening.rs`;
- `crates/gadgets/src/input_injector.rs`;
- `crates/we/src/diamond/graph.rs`;
- `crates/io/src/aky24/prfe.rs`;
- `crates/io/src/diamond/graph.rs`;
- `crates/func-enc/src/aky24.rs`;
- `crates/exponent-lut/src/encoding.rs`;
- `crates/exponent-lut/src/public_key.rs`;
- `crates/exponent-lut/src/refresh.rs`; and
- `crates/exponent-lut/src/refresh_setup.rs`.

Re-run the repository-wide search immediately before migration because this
list is evidence from the current branch, not a permanently closed allowlist.

### 8. Estimator and benchmark migration

Use the production whole-compact-owner functions with internally tiled DCRT
expansion, exact parameter set, real storage flow, owner-local observation
waits, and full column count. Separate
transmission, cache, persistent storage, resident VRAM, and expanded workspace.

### 9. Legacy deletion

After all call sites and artifacts migrate, delete old APIs, obsolete
environment variables, serializers, expanded preimage routes, timing
scaffolding, and compatibility code. Retain only the new internal tile controls
listed in the Configuration section; do not retain or alias the obsolete
public-operation column-chunk control. A repository-wide search must find no
live old path.

## Validation plan

### Static and CPU gates

- `cargo +nightly fmt --all`;
- warning-free CPU workspace library no-run build;
- focused IR validation, serde, artifact, runtime, and CPU primitive tests;
- full workspace unit tests where their runtime is practical; and
- no integration tests unless explicitly requested.

Required CPU tests cover:

- exact bound endpoints and rejection above the bound;
- negative, zero, and maximum coefficients;
- malformed sign, width, count, shape, modulus, ring dimension, and semantic
  kind;
- row-major offsets and complete canonical artifact round trips;
- internal expansion tile widths producing identical output and artifact bytes;
- generic bounded matrix versus preimage type rejection;
- multi-column ordinary-multiplication equivalence; and
- exact preimage relation preservation;
- loop-dependent bounds evaluated separately for each parallel instance;
- batches with distinct evaluated bounds remaining distinct; and
- external bounded owners rejected when any parameter, shape, bound, or
  semantic-kind metadata differs from the validated wire schema.

### GPU compile and repeated tests

- warning-free workspace library no-run build with GPU features;
- GPU tests outside the sandbox;
- identical-command repeated runs for asynchronous/event-sensitive tests;
- no CPU fallback; and
- no weakening of existing tests.

Required GPU tests cover:

- persistent descriptor initialization, accounting, and cleanup;
- nonuniform multi-GPU partitions and active-level limb mapping;
- owner-local wait fencing only the requested result;
- all-coefficient, all-evaluation, and mixed-domain construction;
- direct gadget evaluation and coefficient-domain hash-source equivalence;
- row-major all-column compact CPU/GPU round trip;
- `mul_small_rhs` equivalence with trusted full-DCRT multiplication across
  distinct CRT primes;
- `C_t = 1` and larger legal tiles producing identical results;
- preimage output independent of tile width and GPU scheduling;
- cutoffs below the conservative default returning `PreimageBoundTooSmall`
  without launching a sampling attempt;
- retry, exact retry exhaustion, and event-safe candidate reuse; and
- allocator query versus actual high-water memory.

### Performance evidence

On the same device, parameters, build, and artifact flow, record:

- the prior implementation's latency and peak;
- one-column and selected multi-column chunk latency;
- total full-column time;
- persistent descriptor bytes;
- all-column compact owner bytes;
- expanded workspace bytes;
- output bytes;
- maximum parallelism; and
- physical allocator high-water memory.

Increase column, reduction, and limb parallelism up to the VRAM budget only
after the baseline result is captured. A performance estimate is not GPU
runtime validation.

## Acceptance criteria

The migration is complete only when all of the following are true:

1. every bounded RHS has an explicit inclusive bound and typed semantic kind;
2. no ordinary matrix can enter `MatrixMulSmallRhs` without validation;
3. an `O((log q)^2)` logical RHS or preimage may be resident only in compact
   low-bit form; its all-column DCRT/NTT expansion is never resident;
4. no full DCRT/NTT RHS or full expanded preimage is materialized;
5. all input and output mathematical dimensions are unchanged;
6. persisted compact artifacts are canonical row-major and independent of
   wave configuration;
7. sampling results are deterministic under legal chunking changes;
8. multiplication embeds and transforms coefficients independently for each
   CRT prime;
9. GPU matrix descriptors are persistent and included in allocation reports;
10. normal GPU execution has no device-wide or stream-wide synchronization;
11. benchmark estimates use the production path and report full total time;
12. transmission, deterministic cache, storage, VRAM, and workspace are
    reported separately;
13. the generic noise simulator received only the minimal semantic update;
14. legacy APIs, formats, controls, and compatibility shims are absent;
15. formatting, compile, unit, repeated GPU, and performance gates pass; and
16. an independent implementation review confirms that the integrated change
    preserves the specified algebra, bounds, memory order, and event lifetime.

## Explicit zero-old-use gate

After migration, repository-wide searches over source, tests, benches, and
documentation outside this plan must show no live use of:

```text
Preimage::as_mat
apply_preimage
mul_decomposed
mul_decompose
mul_decompose_small
mul_decompose_column_chunk_width
MXX_MUL_DECOMPOSE_COLUMN_CHUNK_WIDTH
```

Any conceptually equivalent expanded or type-erasing replacement also fails
this gate, even if it has a different name.
