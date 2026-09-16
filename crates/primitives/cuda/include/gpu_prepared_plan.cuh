#pragma once

#include <stddef.h>
#include <stdint.h>

#include "matrix/MatrixArith.cuh"
#include "matrix/Matrix.cuh"
#include "gpu_admission.cuh"

struct GpuPreparedPlanRef;

// Metadata-only planning for prepared primitive plans.
//
// Planning answers two questions before any native owner or plan exists: which
// exact allocations a prepared stage would request, and which physical streams
// it would submit on. Both answers come from the same native selection code the
// real preparation uses, so a plan can never drift from what preparation
// consumes. Planning itself performs no CUDA allocation, no stream/event
// creation and no kernel launch; every function here only reads structural
// metadata (device partition, limb structure, rows, columns, level, format,
// basis ranges) and an explicit owner stream cursor. Planning never observes
// or advances the execution owner's mutable stream counter.

// Native stage roles. A footprint key names the stage that consumes the
// resource, so the same physical stream/limb used by two stages stays
// distinguishable.
typedef enum GpuPreparedStageRole
{
    GPU_PREPARED_STAGE_UPLOAD = 0,
    GPU_PREPARED_STAGE_READBACK = 1,
    GPU_PREPARED_STAGE_RECONSTRUCTION = 2,
    GPU_PREPARED_STAGE_NTT = 3,
    GPU_PREPARED_STAGE_ARITHMETIC = 4,
    GPU_PREPARED_STAGE_SCALAR = 5,
    GPU_PREPARED_STAGE_TRANSFORM = 6,
    GPU_PREPARED_STAGE_SMALL_RHS = 7,
    GPU_PREPARED_STAGE_SCHEDULE = 8,
    GPU_PREPARED_STAGE_SAMPLING = 9,
    // Scalar resources are separate native stages.  Keeping these roles
    // distinct is important: scalar buffers own pinned staging and two
    // completion events, while scalar consumers own only their operation
    // workspace (and scalar pack owns one completion event).
    GPU_PREPARED_STAGE_SCALAR_BUFFER = 10,
    GPU_PREPARED_STAGE_SCALAR_OP = 11,
    GPU_PREPARED_STAGE_SCALAR_MATRIX_SELECT = 12,
    GPU_PREPARED_STAGE_THRESHOLD = 13,
    GPU_PREPARED_STAGE_SCALAR_PACK = 14,
} GpuPreparedStageRole;

// Where a prepared plan's submission stream comes from.
typedef enum GpuPreparedStreamOrigin
{
    // A compute stream owned by the execution context and shared with the
    // owners of the same partition. The plan creates nothing for it.
    GPU_PREPARED_STREAM_CONTEXT_REUSED = 0,
    // A submission stream the plan owns, together with its reusable bridge
    // event. The plan adds this stream to the context's fixed pool.
    GPU_PREPARED_STREAM_ADDED_SUBMISSION = 1,
} GpuPreparedStreamOrigin;

// Host-only prepared metadata that is never claimed from a native slot domain.
#define GPU_PREPARED_PLAN_HOST_ONLY 100

// One physical prepared-plan resource identity. `partition`/`device` are the
// physical execution placement, `limb_*` is the base owner limb that owns the
// storage, and `role` is the native stage that consumes it. Interior views are
// never keyed directly: gpu_prepared_base_owner normalizes them to the owner
// that actually owns the storage, so two plans reading the same physical
// resource agree.
typedef struct GpuPreparedResourceKey
{
    uint64_t execution_owner_identity;
    int partition;
    int device;
    uint32_t limb_x;
    uint32_t limb_y;
    int role;
} GpuPreparedResourceKey;

// One exact native allocation request of a prepared stage, in the order the
// preparation issues it. The fields mirror GpuClaimTraceEntry so the same
// ordered list can be handed to the provisioning/admission domain unchanged.
struct GpuPreparedAllocationLayout
{
    GpuPreparedResourceKey key;
    int kind;
    size_t rows;
    size_t columns;
    size_t bytes;
    size_t alignment;
    int level;
    int format;
};

// One submission stream of a prepared stage.
struct GpuPreparedStreamFootprint
{
    GpuPreparedResourceKey key;
    int origin;
    // Index in the partition's compute stream pool for a context-reused
    // stream; unused (0) for a plan-owned added submission stream.
    size_t pool_slot;
};

struct GpuPreparedLaunchLayout
{
    int phase;
    dim3 grid;
    dim3 block;
    uint32_t len;
    size_t limb_offset;
    size_t limb_count;
    int narrow;
};

// Bounded descriptor. The maxima cover one entry per active limb plus the
// stage's own typed resources, and the complete merged schedule.
#define GPU_PREPARED_PLAN_MAX_ALLOCATIONS 256
#define GPU_PREPARED_PLAN_MAX_STREAMS 512
#define GPU_PREPARED_OWNER_MAX_PARTITIONS 64

// Value-only stream assignment for one prepared matrix owner. It contains no
// CUDA handles and is computed before the native owner exists.
struct GpuPreparedOwnerPartitionLayout
{
    int device;
    size_t pool_size;
    size_t local_limb_count;
    size_t shared_stream_slot;
    size_t limb_stream_slots[GPU_RUNTIME_MAX_LIMBS];
};

struct GpuPreparedOwnerLayout
{
    uint64_t execution_owner_identity;
    size_t stream_ordinal_base;
    int execution_class;
    size_t partition_count;
    GpuPreparedOwnerPartitionLayout partitions[GPU_PREPARED_OWNER_MAX_PARTITIONS];
};

struct GpuPreparedPlanDescriptor
{
    size_t allocation_count;
    GpuPreparedAllocationLayout allocations[GPU_PREPARED_PLAN_MAX_ALLOCATIONS];
    size_t stream_count;
    GpuPreparedStreamFootprint streams[GPU_PREPARED_PLAN_MAX_STREAMS];
    size_t launch_count;
    GpuPreparedLaunchLayout launches[GPU_PREPARED_PLAN_MAX_STREAMS];
};

// Validate descriptor bounds and immutable key/stream fields before any
// prepared native entry acquires a resource. Operation-specific validators
// perform the remaining shape checks.
int gpu_prepared_validate_descriptor(const GpuPreparedPlanDescriptor *descriptor);

// NTT launch geometry constants. Planning and preparation share them so the
// launch table cannot drift from the kernels it describes.
#ifdef __cplusplus
constexpr uint32_t GPU_PREPARED_NTT_THREADS = 256;
constexpr uint32_t GPU_PREPARED_NTT_FUSED_COEFFICIENTS = 1024;
constexpr size_t GPU_PREPARED_NTT_MAX_GRID_Y = 65535;
#endif

// NTT launch geometry. The geometry is a pure function of the ring dimension,
// limb count and transformed polynomial count, so planning and preparation
// share this table instead of duplicating the selection rules.
typedef enum GpuPreparedNttLaunchKind
{
    GPU_PREPARED_NTT_FUSED_TOP = 0,
    GPU_PREPARED_NTT_TWIST = 1,
    GPU_PREPARED_NTT_STAGE = 2,
    GPU_PREPARED_NTT_SCALE = 3,
    GPU_PREPARED_NTT_FUSED_LOCAL = 4,
} GpuPreparedNttLaunchKind;

struct GpuPreparedNttLaunchLayout
{
    int kind;
    dim3 grid;
    dim3 block;
    uint32_t n;
    uint32_t len;
    uint32_t width;
    size_t limb_count;
    size_t poly_offset;
    size_t shared_bytes;
    int forward;
};

extern "C" {

// The base owner of a matrix. A prepared view is normalized to the owner that
// owns its storage, and an ordinary owner is returned unchanged.
const GpuMatrix *gpu_prepared_base_owner(const GpuMatrix *matrix);

// Pure selection of the compute stream pool slot an owner limb receives. The
// matrix allocator and every planner call this one rule.
size_t gpu_prepared_stream_slot(size_t pool_size, size_t counter, size_t ordinal);

// Pure owner stream assignment. `stream_ordinal_base` is a caller-owned
// cursor position and is never read from GpuExecutionOwner::next_compute_stream.
int gpu_prepared_owner_layout(
    const GpuContext *ctx, int level, size_t rows, size_t columns, int format,
    size_t stream_ordinal_base, GpuPreparedOwnerLayout *out);
int gpu_prepared_owner_layout_matches(
    const GpuContext *ctx, const GpuPreparedOwnerLayout *layout);

// Pool slot of an already selected stream, or an error when the stream does not
// belong to the partition's pool.
int gpu_prepared_stream_slot_of(
    const GpuContext *ctx, size_t partition, cudaStream_t stream, size_t *out_slot);

// Consumption check for a saved plan: the owner's actual stream must be the one
// the saved descriptor selected. A mismatch is a failed warmup, never a
// fallback allocation.
int gpu_prepared_require_stream_slot(
    const GpuContext *ctx, size_t partition, cudaStream_t stream, size_t pool_slot);

// Consumption check for the descriptor's allocation order: the exact request
// at `index` must match what the preparation is about to allocate.
int gpu_prepared_require_allocation(
    const GpuPreparedPlanDescriptor *descriptor, size_t index, int kind,
    const GpuPreparedResourceKey *key, size_t bytes, size_t alignment);

// Device placement of one active limb, shared by planning and preparation.
int gpu_prepared_limb_key(
    const GpuContext *ctx, int level, size_t index, int role, dim3 *out_limb,
    GpuPreparedResourceKey *out_key);

// Pure launch geometry of one NTT plan over `poly_count` polynomials. A null
// output only reports the required entry count; a too-small output reports an
// error instead of truncating the plan.
int gpu_prepared_ntt_launch_table(
    uint32_t n, size_t limb_count, size_t poly_count, int forward,
    GpuPreparedNttLaunchLayout *out, size_t capacity, size_t *count);

// Exact allocation layout and stream footprint of a prepared stage. These
// entry points never allocate, never create a stream or event, and never
// launch a kernel.
int gpu_prepared_plan_const_coeff_readback(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int format,
    size_t words_per_poly, size_t coefficient_index, size_t coefficient_count,
    GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_const_coeff_readback_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int format,
    size_t words_per_poly, size_t coefficient_index, size_t coefficient_count,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_rns_upload(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int target_format,
    int transform_to_eval, size_t bytes_per_poly, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_rns_upload_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int target_format,
    int transform_to_eval, size_t bytes_per_poly, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out);
// Replay upload plans. Matrix uploads decode a bounded compact artifact into
// an ordinary owner and may append a fixed forward NTT geometry; small uploads
// copy the canonical sign/magnitude payload into an already admitted compact
// owner. Both descriptors include the pinned host staging and terminal event.
int gpu_prepared_plan_compact_upload_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int target_format,
    size_t payload_capacity, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_small_upload_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level, size_t payload_bytes,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_rns_reconstruction(
    const GpuContext *ctx, size_t rows, size_t columns, int level,
    size_t words_per_poly, size_t coefficient_index, size_t coefficient_count,
    GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_rns_reconstruction_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level,
    size_t words_per_poly, size_t coefficient_index, size_t coefficient_count,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_ntt(
    const GpuContext *ctx, size_t rows, size_t columns, int level,
    const GpuMatrixRange *range, int forward, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_ntt_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level,
    const GpuMatrixRange *range, int forward, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_sampling_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, size_t full_ncol,
    size_t col_offset, int level, int format, int dist_type,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_arithmetic_with_owner(
    const GpuContext *ctx, size_t ring_dimension, size_t limb_count,
    size_t left_rows, size_t left_columns, size_t right_rows, size_t right_columns,
    size_t output_rows, size_t output_columns, size_t column_start,
    size_t group_count, size_t term_count, int kind, int device,
    int evaluation_format, int thin, int lazy_reduction,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_schedule(
    const GpuPreparedPlanDescriptor *const *plans, size_t count,
    GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_scalar_buffer(
    const GpuContext *ctx, size_t count, size_t words, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_scalar_op(
    const GpuContext *ctx, size_t left_words, size_t right_words, size_t output_words,
    size_t candidate_count, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_scalar_matrix_select(
    const GpuContext *ctx, size_t rows, size_t columns, size_t n, int level,
    size_t count, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_threshold(
    const GpuContext *ctx, size_t count, size_t plaintext_words, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_threshold_with_owner(
    const GpuContext *ctx, size_t count, size_t plaintext_words,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_scalar_pack(
    const GpuContext *ctx, size_t count, size_t coefficient_bits, int level, int output_format,
    GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_small_rhs(
    const GpuContext *ctx, int level, size_t inner, size_t columns,
    size_t residency_budget_bytes, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_small_rhs_with_owner(
    const GpuContext *ctx, int level, size_t inner, size_t columns,
    size_t residency_budget_bytes, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out);
// Composite matrix stages have separate entry points so their structural
// contracts cannot be hidden behind a generic rectangular placeholder.  The
// entries still share the native owner/stream and completion accounting.
int gpu_prepared_plan_input_copy_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int format,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_transpose_with_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t output_rows, size_t output_columns, int level, int format,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_centered_rebase_with_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t target_rows, size_t target_columns, int source_level, int target_level,
    int source_format, int target_format, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_gadget_decompose_with_source_format_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t output_rows, int level, int source_format, int format, uint32_t base_bits,
    int small, size_t dropped_moduli, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_modulus_conversion_with_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t target_rows, size_t target_columns, int source_level, int target_level,
    int source_format, int target_format, int mode, size_t digit_size,
    uint64_t plaintext_modulus, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_rns_conversion_with_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t target_rows, size_t target_columns, int source_level, int target_level,
    size_t digit_size, int normalize, uint64_t plaintext_modulus,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out);
int gpu_prepared_plan_crt_recompose_with_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t level_count, size_t output_rows, size_t output_columns, int target_level,
    int target_format, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out);
}
