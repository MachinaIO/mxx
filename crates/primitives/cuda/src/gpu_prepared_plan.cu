// Metadata-only prepared-planning implementation.
//
// Every rule below is also used by the preparation it describes: the stream
// slot rule is shared with the matrix allocator, the NTT launch geometry is
// shared with the prepared NTT plan, and the allocation requests are the exact
// requests the matching preparation issues. Nothing in this file allocates,
// creates a stream or event, or launches a kernel.

#include "gpu_prepared_plan.cuh"
#include "matrix/MatrixCrt.cuh"
#include "matrix/MatrixDecompose.cuh"
#include "matrix/MatrixArith.cuh"
#include "matrix/MatrixSampling.cuh"
#include "matrix/MatrixSerde.cuh"

#include <algorithm>
#include <cstring>
#include <limits>

static int validate_owner_layout_for_shape(
    const GpuContext *ctx, int level, size_t rows, size_t columns, int format,
    const GpuPreparedOwnerLayout *layout);

namespace
{

    bool plan_checked_mul_size(size_t left, size_t right, size_t *out)
    {
        if (!out)
        {
            return false;
        }
        if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
        {
            return false;
        }
        *out = left * right;
        return true;
    }

    // Ordered writer over one bounded descriptor. Capacity is structural, so a
    // full descriptor is a programming error rather than a runtime condition.
    struct PlanDescriptorWriter
    {
        GpuPreparedPlanDescriptor *descriptor;

        int allocation(const GpuPreparedResourceKey &key, int kind, size_t rows, size_t columns,
            size_t bytes, size_t alignment, int level, int format)
        {
            if (descriptor->allocation_count >= GPU_PREPARED_PLAN_MAX_ALLOCATIONS)
            {
                return set_error("prepared plan allocation layout is full");
            }
            auto &entry = descriptor->allocations[descriptor->allocation_count++];
            entry.key = key;
            entry.kind = kind;
            entry.rows = rows;
            entry.columns = columns;
            entry.bytes = bytes;
            entry.alignment = alignment;
            entry.level = level;
            entry.format = format;
            return 0;
        }

        int stream(const GpuPreparedResourceKey &key, int origin, size_t pool_slot)
        {
            if (descriptor->stream_count >= GPU_PREPARED_PLAN_MAX_STREAMS)
            {
                return set_error("prepared stream footprint is full");
            }
            auto &entry = descriptor->streams[descriptor->stream_count++];
            entry.key = key;
            entry.origin = origin;
            entry.pool_slot = pool_slot;
            return 0;
        }

        int launch(int phase, dim3 grid, uint32_t len, size_t limb_offset,
            size_t limb_count, bool narrow)
        {
            if (descriptor->launch_count >= GPU_PREPARED_PLAN_MAX_STREAMS)
                return set_error("prepared launch layout is full");
            auto &entry = descriptor->launches[descriptor->launch_count++];
            entry.phase = phase;
            entry.grid = grid;
            entry.block = dim3(256);
            entry.len = len;
            entry.limb_offset = limb_offset;
            entry.limb_count = limb_count;
            entry.narrow = narrow ? 1 : 0;
            return 0;
        }
    };

    int append_plan_descriptor(
        GpuPreparedPlanDescriptor *out, const GpuPreparedPlanDescriptor *part)
    {
        if (!out || !part || out->allocation_count > GPU_PREPARED_PLAN_MAX_ALLOCATIONS -
                part->allocation_count || out->stream_count > GPU_PREPARED_PLAN_MAX_STREAMS -
                part->stream_count || out->launch_count > GPU_PREPARED_PLAN_MAX_STREAMS -
                part->launch_count)
            return set_error("prepared composite descriptor capacity exceeded");
        std::memcpy(out->allocations + out->allocation_count, part->allocations,
            part->allocation_count * sizeof(part->allocations[0]));
        out->allocation_count += part->allocation_count;
        std::memcpy(out->streams + out->stream_count, part->streams,
            part->stream_count * sizeof(part->streams[0]));
        out->stream_count += part->stream_count;
        std::memcpy(out->launches + out->launch_count, part->launches,
            part->launch_count * sizeof(part->launches[0]));
        out->launch_count += part->launch_count;
        if (part->scratch_owner_layout.execution_owner_identity != 0)
        {
            if (out->scratch_owner_conflict != 0)
            {
                // Keep the explicit conflict marker sticky.
            }
            else if (out->scratch_owner_layout.execution_owner_identity != 0 &&
                std::memcmp(&out->scratch_owner_layout, &part->scratch_owner_layout,
                    sizeof(out->scratch_owner_layout)) != 0)
            {
                out->scratch_owner_layout = GpuPreparedOwnerLayout{};
                out->scratch_owner_conflict = 1;
            }
            else
                out->scratch_owner_layout = part->scratch_owner_layout;
        }
        return 0;
    }

    int append_plan_descriptor_without_completion_events(
        GpuPreparedPlanDescriptor *out, const GpuPreparedPlanDescriptor *part)
    {
        if (!out || !part || out->allocation_count > GPU_PREPARED_PLAN_MAX_ALLOCATIONS -
                part->allocation_count || out->launch_count > GPU_PREPARED_PLAN_MAX_STREAMS -
                part->launch_count)
            return set_error("prepared composite descriptor capacity exceeded");
        for (size_t index = 0; index < part->allocation_count; ++index)
        {
            const auto &allocation = part->allocations[index];
            if (allocation.kind == GPU_PREPARED_COMPLETION_EVENT)
                continue;
            if (out->allocation_count >= GPU_PREPARED_PLAN_MAX_ALLOCATIONS)
                return set_error("prepared plan allocation layout is full");
            out->allocations[out->allocation_count++] = allocation;
        }
        std::memcpy(out->launches + out->launch_count, part->launches,
            part->launch_count * sizeof(part->launches[0]));
        out->launch_count += part->launch_count;
        if (part->scratch_owner_layout.execution_owner_identity != 0)
        {
            if (out->scratch_owner_conflict != 0)
            {
                // Keep the explicit conflict marker sticky.
            }
            else if (out->scratch_owner_layout.execution_owner_identity != 0 &&
                std::memcmp(&out->scratch_owner_layout, &part->scratch_owner_layout,
                    sizeof(out->scratch_owner_layout)) != 0)
            {
                out->scratch_owner_layout = GpuPreparedOwnerLayout{};
                out->scratch_owner_conflict = 1;
            }
            else
                out->scratch_owner_layout = part->scratch_owner_layout;
        }
        return 0;
    }

    GpuPreparedResourceKey host_key(int role)
    {
        // Host-side allocations have no physical device placement.
        GpuPreparedResourceKey key{};
        key.execution_owner_identity = 0;
        key.context_identity = 0;
        key.instance = 0;
        key.partition = -1;
        key.device = -1;
        key.limb_x = 0;
        key.limb_y = 0;
        key.role = role;
        return key;
    }

    int device_key(const GpuContext *ctx, size_t partition, uint32_t limb_x, uint32_t limb_y,
        int role, GpuPreparedResourceKey *out)
    {
        if (!ctx || !out || partition >= ctx->gpu_ids.size())
        {
            return set_error("prepared plan partition is outside the execution context");
        }
        out->execution_owner_identity = ctx->execution->identity;
        out->context_identity = reinterpret_cast<uint64_t>(ctx);
        out->instance = 0;
        out->partition = static_cast<int>(partition);
        out->device = ctx->gpu_ids[partition];
        out->limb_x = limb_x;
        out->limb_y = limb_y;
        out->role = role;
        return 0;
    }

    size_t plan_partition_pool_size(const GpuContext *ctx, size_t partition)
    {
        if (!ctx || !ctx->execution ||
            partition >= ctx->execution->compute_streams_by_partition.size())
        {
            return 0;
        }
        return ctx->execution->compute_streams_by_partition[partition].size();
    }

    int plan_begin(const GpuContext *ctx, GpuPreparedPlanDescriptor *out)
    {
        if (!out)
        {
            return set_error("missing prepared plan descriptor output");
        }
        *out = GpuPreparedPlanDescriptor{};
        if (!ctx || !ctx->execution || ctx->N <= 0)
        {
            return set_error("invalid prepared plan context");
        }
        return 0;
    }

    // Active limb placement shared by every matrix-backed prepared stage.
    int plan_limb(const GpuContext *ctx, int level, size_t index, dim3 *out_limb)
    {
        if (!out_limb || level < 0 || index > static_cast<size_t>(level) ||
            index >= GPU_RUNTIME_MAX_LIMBS || ctx->limb_gpu_ids.size() < index + 1)
        {
            return set_error("incomplete prepared plan limb metadata");
        }
        *out_limb = ctx->limb_gpu_ids[index];
        if (out_limb->x >= ctx->gpu_ids.size())
        {
            return set_error("prepared plan limb partition is outside the context");
        }
        return 0;
    }

    // A context-reused stream is identified by the partition pool slot it
    // occupies, which is what two plans sharing one physical stream agree on.
    int plan_reused_stream(PlanDescriptorWriter &writer, const GpuContext *ctx,
        const dim3 &limb_id, int role, size_t ordinal,
        const GpuPreparedOwnerLayout *owner_layout = nullptr)
    {
        GpuPreparedResourceKey key{};
        const int key_status =
            device_key(ctx, limb_id.x, limb_id.x, limb_id.y, role, &key);
        if (key_status != 0)
        {
            return key_status;
        }
        const size_t pool_size = plan_partition_pool_size(ctx, limb_id.x);
        if (pool_size == 0)
        {
            return set_error("prepared plan partition has no compute stream pool");
        }
        size_t slot = gpu_prepared_stream_slot(pool_size, 0, ordinal);
        if (owner_layout)
        {
            if (gpu_prepared_owner_layout_matches(ctx, owner_layout) != 0 ||
                limb_id.x >= owner_layout->partition_count)
                return set_error("prepared stage owner layout does not match context");
            const auto &owner = owner_layout->partitions[limb_id.x];
            if (owner_layout->execution_class == 1)
                slot = owner.shared_stream_slot;
            else if (limb_id.y >= owner.local_limb_count)
                return set_error("prepared stage owner layout limb is outside owner");
            else
                slot = owner.limb_stream_slots[limb_id.y];
            if (slot >= pool_size)
                return set_error("prepared stage owner stream slot is outside pool");
        }
        return writer.stream(key, GPU_PREPARED_STREAM_CONTEXT_REUSED, slot);
    }

    int plan_matrix_launch(
        GpuPreparedPlanDescriptor *out, const GpuContext *ctx, size_t rows,
        size_t columns, size_t limb_count, size_t multiplier, uint32_t block,
        size_t *grid_x)
    {
        if (!out || !ctx || !ctx->N || !rows || !columns || !limb_count || !multiplier ||
            rows > SIZE_MAX / columns || rows * columns > SIZE_MAX / static_cast<size_t>(ctx->N) ||
            rows * columns * static_cast<size_t>(ctx->N) > SIZE_MAX / multiplier)
            return set_error("prepared matrix launch dimensions overflow");
        const size_t work = rows * columns * static_cast<size_t>(ctx->N) * multiplier;
        if (work > SIZE_MAX - (block - 1)) return set_error("prepared matrix launch overflow");
        const size_t blocks = (work + block - 1) / block;
        if (blocks > std::numeric_limits<unsigned int>::max() ||
            limb_count > std::numeric_limits<unsigned int>::max())
            return set_error("prepared matrix launch exceeds CUDA grid capacity");
        PlanDescriptorWriter writer{out};
        if (writer.launch(0, dim3(static_cast<unsigned int>(blocks), 1, 1),
                static_cast<uint32_t>(ctx->N), 0, limb_count, false) != 0)
            return -1;
        if (out->launch_count != 1) return set_error("prepared matrix launch is duplicated");
        out->launches[0].block = dim3(block, 1, 1);
        if (grid_x) *grid_x = blocks;
        return 0;
    }

    // Prepared readback: one pinned destination plus one completion event per
    // active limb, submitted on the limb streams of the owner being read.
    int plan_readback(const GpuContext *ctx, size_t rows, size_t columns, int level, int format,
        size_t words_per_poly, size_t coefficient_index, size_t coefficient_count, int role,
        GpuPreparedPlanDescriptor *out, const GpuPreparedOwnerLayout *owner_layout = nullptr)
    {
        const int begin_status = plan_begin(ctx, out);
        if (begin_status != 0)
        {
            return begin_status;
        }
        if (owner_layout)
        {
            if (validate_owner_layout_for_shape(
                    ctx, level, rows, columns, format, owner_layout) != 0)
                return set_error("prepared readback owner layout does not match matrix");
        }
        if (format != GPU_POLY_FORMAT_COEFF)
        {
            return set_error("prepared readback planning requires a coefficient matrix");
        }
        if (level < 0 || coefficient_count == 0 ||
            coefficient_index >= static_cast<size_t>(ctx->N) ||
            coefficient_count > static_cast<size_t>(ctx->N) - coefficient_index)
        {
            return set_error("invalid prepared readback plan dimensions");
        }
        size_t polynomial_count = 0;
        if (!plan_checked_mul_size(rows, columns, &polynomial_count) || polynomial_count == 0)
        {
            return set_error("invalid prepared readback plan matrix size");
        }
        const size_t limb_count = static_cast<size_t>(level) + 1;
        if (limb_count > GPU_RUNTIME_MAX_LIMBS || ctx->limb_gpu_ids.size() < limb_count)
        {
            return set_error("incomplete prepared readback plan limb metadata");
        }
        size_t required_words = 0;
        size_t total_words = 0;
        size_t total_bytes = 0;
        if (!plan_checked_mul_size(limb_count, coefficient_count, &required_words) ||
            words_per_poly < required_words ||
            !plan_checked_mul_size(polynomial_count, words_per_poly, &total_words) ||
            !plan_checked_mul_size(total_words, sizeof(uint64_t), &total_bytes))
        {
            return set_error("prepared readback plan output size overflow");
        }

        PlanDescriptorWriter writer{out};
        int status = writer.allocation(
            host_key(role), GPU_PREPARED_PINNED_HOST, 0, 0, total_bytes, alignof(uint64_t), -1, -1);
        if (status != 0)
        {
            return status;
        }
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            dim3 limb_id{};
            status = plan_limb(ctx, level, limb, &limb_id);
            if (status != 0)
            {
                return status;
            }
            GpuPreparedResourceKey key{};
            status = device_key(ctx, limb_id.x, limb_id.x, limb_id.y, role, &key);
            if (status != 0)
            {
                return status;
            }
            // A prepared CUDA resource slot never carries bytes or alignment.
            status = writer.allocation(key, GPU_PREPARED_COMPLETION_EVENT, 0, 0, 0, 1, -1, -1);
            if (status != 0)
            {
                return status;
            }
            status = plan_reused_stream(writer, ctx, limb_id, role, limb, owner_layout);
            if (status != 0)
            {
                return status;
            }
        }
        return 0;
    }
}

const GpuMatrix *gpu_prepared_base_owner(const GpuMatrix *matrix)
{
    const GpuMatrix *owner = matrix;
    // A prepared view keeps a non-owning link to the owner of its storage;
    // following it keys the base owner instead of the transient view.
    while (owner && owner->prepared_view_owner)
    {
        owner = owner->prepared_view_owner;
    }
    return owner;
}

int gpu_matrix_prepared_owner_layout(
    const GpuMatrix *matrix, GpuPreparedOwnerLayout *out)
{
    if (!matrix || !out || !matrix->ctx || !matrix->ctx->execution ||
        matrix->prepared_owner_partition_count > GPU_PREPARED_OWNER_MAX_PARTITIONS)
        return set_error("missing prepared matrix owner layout");
    *out = GpuPreparedOwnerLayout{};
    out->execution_owner_identity = matrix->prepared_owner_execution_identity;
    out->execution_class = matrix->prepared_owner_execution_class;
    out->partition_count = matrix->prepared_owner_partition_count;
    if (out->partition_count != matrix->ctx->gpu_ids.size())
        return set_error("prepared matrix owner layout partition count is invalid");
    for (size_t partition = 0; partition < out->partition_count; ++partition)
    {
        const auto &source = matrix->prepared_owner_partitions[partition];
        auto &destination = out->partitions[partition];
        destination.device = source.device;
        destination.pool_size = source.pool_size;
        destination.local_limb_count = source.local_limb_count;
        destination.shared_stream_slot = source.shared_stream_slot;
        std::memcpy(destination.limb_stream_slots, source.limb_stream_slots,
            sizeof(destination.limb_stream_slots));
        if (source.pool_size == 0 || source.device != matrix->ctx->gpu_ids[partition] ||
            source.local_limb_count > GPU_RUNTIME_MAX_LIMBS ||
            source.shared_stream_slot >= source.pool_size)
            return set_error("prepared matrix owner layout metadata is invalid");
        for (size_t limb = 0; limb < source.local_limb_count; ++limb)
            if (source.limb_stream_slots[limb] >= source.pool_size)
                return set_error("prepared matrix owner limb slot is invalid");
    }
    return 0;
}

size_t gpu_prepared_stream_slot(size_t pool_size, size_t counter, size_t ordinal)
{
    if (pool_size == 0)
    {
        return 0;
    }
    return (counter + ordinal) % pool_size;
}

int gpu_prepared_owner_layout(
    const GpuContext *ctx, int level, size_t rows, size_t columns, int format,
    GpuPreparedOwnerLayout *out)
{
    if (!ctx || !ctx->execution || !out || level < -1 || level > ctx->level ||
        (format != GPU_POLY_FORMAT_COEFF && format != GPU_POLY_FORMAT_EVAL) ||
        ctx->gpu_ids.size() > GPU_PREPARED_OWNER_MAX_PARTITIONS)
    {
        return set_error("invalid prepared owner layout arguments");
    }
    size_t count = 0;
    if (!plan_checked_mul_size(rows, columns, &count))
        return set_error("prepared owner layout matrix size overflow");
    *out = GpuPreparedOwnerLayout{};
    out->execution_owner_identity = ctx->execution->identity;
    out->partition_count = ctx->gpu_ids.size();
    // Keep this classification identical to MatrixData's allocation rule;
    // these numeric values are the public MatrixData enum values, while this
    // translation unit intentionally does not include MatrixData.cuh.
    out->execution_class = count == 0 ? 0
        : (rows <= 4 && columns <= 4 ? 1 : 2);

    const size_t active_limbs = level < 0 ? 0 : static_cast<size_t>(level + 1);
    if (active_limbs > GPU_RUNTIME_MAX_LIMBS || ctx->limb_gpu_ids.size() < active_limbs)
        return set_error("incomplete prepared owner layout limb metadata");
    size_t cursor = 0;
    for (size_t partition = 0; partition < out->partition_count; ++partition)
    {
        auto &entry = out->partitions[partition];
        entry.device = ctx->gpu_ids[partition];
        entry.pool_size = plan_partition_pool_size(ctx, partition);
        if (entry.pool_size == 0)
            return set_error("prepared owner layout partition has no stream pool");
        for (size_t limb = 0; limb < active_limbs; ++limb)
        {
            const dim3 limb_id = ctx->limb_gpu_ids[limb];
            if (limb_id.x >= ctx->gpu_ids.size())
                return set_error("prepared owner layout limb partition is invalid");
            if (limb_id.x == partition)
                entry.local_limb_count = std::max(entry.local_limb_count,
                    static_cast<size_t>(limb_id.y) + 1);
        }
        if (entry.local_limb_count == 0 || count == 0)
            continue;
        if (out->execution_class == GPU_MATRIX_SHARED_STREAM)
        {
            entry.shared_stream_slot = gpu_prepared_stream_slot(entry.pool_size, cursor++, 0);
            for (size_t limb = 0; limb < entry.local_limb_count; ++limb)
                entry.limb_stream_slots[limb] = entry.shared_stream_slot;
        }
        else
        {
            for (size_t limb = 0; limb < entry.local_limb_count; ++limb)
                entry.limb_stream_slots[limb] = gpu_prepared_stream_slot(entry.pool_size, cursor++, 0);
        }
    }
    return 0;
}

int gpu_prepared_owner_layout_matches(
    const GpuContext *ctx, const GpuPreparedOwnerLayout *layout)
{
    if (!ctx || !ctx->execution || !layout ||
        layout->execution_owner_identity != ctx->execution->identity ||
        layout->partition_count != ctx->gpu_ids.size() ||
        layout->partition_count > GPU_PREPARED_OWNER_MAX_PARTITIONS)
        return set_error("prepared owner layout does not belong to this execution owner");
    if (layout->execution_class != GPU_MATRIX_EMPTY &&
        layout->execution_class != GPU_MATRIX_SHARED_STREAM &&
        layout->execution_class != GPU_MATRIX_PER_LIMB_STREAMS)
        return set_error("prepared owner layout has an invalid execution class");
    if (ctx->limb_gpu_ids.size() > GPU_RUNTIME_MAX_LIMBS)
        return set_error("prepared owner layout context has too many limbs");
    for (size_t partition = 0; partition < layout->partition_count; ++partition)
    {
        const auto &entry = layout->partitions[partition];
        if (entry.device != ctx->gpu_ids[partition] ||
            partition >= ctx->execution->compute_streams_by_partition.size() ||
            entry.pool_size != ctx->execution->compute_streams_by_partition[partition].size() ||
            entry.pool_size == 0 ||
            entry.local_limb_count > GPU_RUNTIME_MAX_LIMBS)
            return set_error("prepared owner layout stream pool mismatch");
        size_t available_limbs = 0;
        for (const dim3 limb_id : ctx->limb_gpu_ids)
            if (limb_id.x == partition)
                available_limbs = std::max(available_limbs, static_cast<size_t>(limb_id.y) + 1);
        if (entry.local_limb_count > available_limbs)
            return set_error("prepared owner layout local limb count exceeds context limbs");
        if (layout->execution_class == GPU_MATRIX_EMPTY && entry.local_limb_count != 0)
            return set_error("empty prepared owner layout has active limbs");
        for (size_t limb = 0; limb < entry.local_limb_count; ++limb)
            if (entry.limb_stream_slots[limb] >= entry.pool_size)
                return set_error("prepared owner layout stream slot is outside its pool");
        if (layout->execution_class == GPU_MATRIX_SHARED_STREAM &&
            entry.local_limb_count != 0 && entry.shared_stream_slot >= entry.pool_size)
            return set_error("prepared owner layout shared stream slot is outside its pool");
        if (layout->execution_class == GPU_MATRIX_SHARED_STREAM)
        {
            for (size_t limb = 0; limb < entry.local_limb_count; ++limb)
                if (entry.limb_stream_slots[limb] != entry.shared_stream_slot)
                    return set_error("shared prepared owner layout has split limb streams");
        }
        else if (layout->execution_class == GPU_MATRIX_PER_LIMB_STREAMS &&
                 entry.shared_stream_slot != 0)
            return set_error("per-limb prepared owner layout has a shared stream slot");
        else if (layout->execution_class == GPU_MATRIX_EMPTY &&
                 entry.shared_stream_slot != 0)
            return set_error("empty prepared owner layout has a stream slot");
    }
    for (size_t partition = layout->partition_count;
         partition < GPU_PREPARED_OWNER_MAX_PARTITIONS; ++partition)
    {
        const auto &entry = layout->partitions[partition];
        // Native value-initialization leaves unused device fields at zero.
        if (entry.device != 0 || entry.pool_size != 0 || entry.local_limb_count != 0 ||
            entry.shared_stream_slot != 0)
            return set_error("prepared owner layout has nonzero trailing partition metadata");
        for (size_t limb = 0; limb < GPU_RUNTIME_MAX_LIMBS; ++limb)
            if (entry.limb_stream_slots[limb] != 0)
                return set_error("prepared owner layout has trailing limb metadata");
    }
    return 0;
}

static int validate_owner_layout_for_shape(
    const GpuContext *ctx, int level, size_t rows, size_t columns, int format,
    const GpuPreparedOwnerLayout *layout)
{
    if (!ctx || !ctx->execution || !layout || level < -1 || level > ctx->level ||
        (format != GPU_POLY_FORMAT_COEFF && format != GPU_POLY_FORMAT_EVAL) ||
        layout->execution_owner_identity != ctx->execution->identity ||
        layout->partition_count != ctx->gpu_ids.size() ||
        layout->partition_count > GPU_PREPARED_OWNER_MAX_PARTITIONS)
        return set_error("prepared owner layout context or shape mismatch");
    if (gpu_prepared_owner_layout_matches(ctx, layout) != 0)
        return -1;
    size_t count = 0;
    if (!plan_checked_mul_size(rows, columns, &count))
        return set_error("prepared owner layout matrix size overflow");
    const int expected_class = count == 0 ? 0 :
        (rows <= 4 && columns <= 4 ? GPU_MATRIX_SHARED_STREAM : GPU_MATRIX_PER_LIMB_STREAMS);
    if (layout->execution_class != expected_class)
        return set_error("prepared owner layout execution class mismatch");
    const size_t active_limbs = level < 0 ? 0 : static_cast<size_t>(level + 1);
    if (active_limbs > GPU_RUNTIME_MAX_LIMBS || ctx->limb_gpu_ids.size() < active_limbs)
        return set_error("prepared owner layout active limb count mismatch");
    for (size_t partition = 0; partition < layout->partition_count; ++partition)
    {
        const auto &entry = layout->partitions[partition];
        if (entry.device != ctx->gpu_ids[partition] ||
            partition >= ctx->execution->compute_streams_by_partition.size() ||
            entry.pool_size != ctx->execution->compute_streams_by_partition[partition].size() ||
            entry.pool_size == 0)
            return set_error("prepared owner layout partition metadata mismatch");
        size_t local_limbs = 0;
        for (size_t limb = 0; limb < active_limbs; ++limb)
            if (ctx->limb_gpu_ids[limb].x == partition)
                local_limbs = std::max(local_limbs,
                    static_cast<size_t>(ctx->limb_gpu_ids[limb].y) + 1);
        if (entry.local_limb_count != local_limbs)
            return set_error("prepared owner layout local limb count mismatch");
        if (local_limbs != 0 && entry.shared_stream_slot >= entry.pool_size)
            return set_error("prepared owner layout shared stream slot is invalid");
        for (size_t limb = 0; limb < local_limbs; ++limb)
            if (entry.limb_stream_slots[limb] >= entry.pool_size)
                return set_error("prepared owner layout limb stream slot is invalid");
        if (layout->execution_class == GPU_MATRIX_SHARED_STREAM)
        {
            for (size_t limb = 0; limb < local_limbs; ++limb)
                if (entry.limb_stream_slots[limb] != entry.shared_stream_slot)
                    return set_error("shared prepared owner layout has split limb streams");
        }
        else if (layout->execution_class == GPU_MATRIX_PER_LIMB_STREAMS &&
                 entry.shared_stream_slot != 0)
            return set_error("per-limb prepared owner layout has a shared stream slot");
        else if (layout->execution_class == GPU_MATRIX_EMPTY &&
                 (entry.local_limb_count != 0 || entry.shared_stream_slot != 0))
            return set_error("empty prepared owner layout has active stream metadata");
    }
    return 0;
}

int gpu_prepared_stream_slot_of(
    const GpuContext *ctx, size_t partition, cudaStream_t stream, size_t *out_slot)
{
    if (!out_slot || !ctx || !ctx->execution ||
        partition >= ctx->execution->compute_streams_by_partition.size() || !stream)
    {
        return set_error("prepared plan stream query has no partition pool");
    }
    const auto &pool = ctx->execution->compute_streams_by_partition[partition];
    for (size_t slot = 0; slot < pool.size(); ++slot)
    {
        if (pool[slot] == stream)
        {
            *out_slot = slot;
            return 0;
        }
    }
    return set_error("prepared plan stream is outside its partition pool");
}

int gpu_prepared_require_stream_slot(
    const GpuContext *ctx, size_t partition, cudaStream_t stream, size_t pool_slot)
{
    size_t actual = 0;
    const int status = gpu_prepared_stream_slot_of(ctx, partition, stream, &actual);
    if (status != 0)
    {
        return status;
    }
    if (actual != pool_slot)
    {
        return set_error("prepared plan stream differs from its saved descriptor");
    }
    return 0;
}

int gpu_prepared_require_allocation(
    const GpuPreparedPlanDescriptor *descriptor, size_t index, int kind,
    const GpuPreparedResourceKey *key, size_t bytes, size_t alignment)
{
    if (!descriptor || !key)
    {
        return set_error("missing prepared plan allocation descriptor");
    }
    if (index >= descriptor->allocation_count)
    {
        return set_error("prepared plan allocation descriptor is exhausted");
    }
    if ((key->partition < 0) != (key->device < 0))
    {
        return set_error("prepared allocation has a mixed host sentinel resource key");
    }
    const auto &entry = descriptor->allocations[index];
    if (entry.kind != kind || entry.bytes != bytes || entry.alignment != alignment ||
        entry.key.execution_owner_identity != key->execution_owner_identity ||
        entry.key.context_identity != key->context_identity ||
        entry.key.instance != key->instance ||
        entry.key.partition != key->partition || entry.key.device != key->device ||
        entry.key.limb_x != key->limb_x || entry.key.limb_y != key->limb_y ||
        entry.key.role != key->role)
    {
        return set_error("prepared allocation differs from its saved descriptor");
    }
    return 0;
}

int gpu_prepared_validate_descriptor(const GpuPreparedPlanDescriptor *descriptor)
{
    if (!descriptor || descriptor->allocation_count > GPU_PREPARED_PLAN_MAX_ALLOCATIONS ||
        descriptor->stream_count > GPU_PREPARED_PLAN_MAX_STREAMS ||
        descriptor->launch_count > GPU_PREPARED_PLAN_MAX_STREAMS)
        return set_error("prepared plan descriptor bounds are invalid");
    for (size_t index = 0; index < descriptor->allocation_count; ++index)
    {
        const auto &allocation = descriptor->allocations[index];
        if (allocation.alignment == 0 || allocation.key.partition < -1 ||
            allocation.key.device < -1)
            return set_error("prepared plan allocation key is invalid");
    }
    for (size_t index = 0; index < descriptor->stream_count; ++index)
    {
        const auto &stream = descriptor->streams[index];
        if (stream.origin != GPU_PREPARED_STREAM_CONTEXT_REUSED &&
            stream.origin != GPU_PREPARED_STREAM_ADDED_SUBMISSION)
            return set_error("prepared plan stream origin is invalid");
    }
    return 0;
}

// Device placement of one active limb, used by both planning and preparation.
int gpu_prepared_limb_key(
    const GpuContext *ctx, int level, size_t index, int role, dim3 *out_limb,
    GpuPreparedResourceKey *out_key)
{
    if (!ctx || !out_limb || !out_key)
    {
        return set_error("missing prepared plan limb key output");
    }
    const int limb_status = plan_limb(ctx, level, index, out_limb);
    if (limb_status != 0)
    {
        return limb_status;
    }
    return device_key(ctx, out_limb->x, out_limb->x, out_limb->y, role, out_key);
}

int gpu_prepared_ntt_launch_table(
    uint32_t n, size_t limb_count, size_t poly_count, int forward,
    GpuPreparedNttLaunchLayout *out, size_t capacity, size_t *count)
{
    if (!count || n < 2 || (n & (n - 1)) != 0 || limb_count == 0 ||
        limb_count > GPU_RUNTIME_MAX_LIMBS)
    {
        return set_error("invalid prepared NTT launch table request");
    }
    *count = 0;
    const uint32_t threads = GPU_PREPARED_NTT_THREADS;
    const size_t max_grid_y = GPU_PREPARED_NTT_MAX_GRID_Y;
    size_t required = 0;

    auto append_chunked = [&](int kind, dim3 grid, uint32_t len, uint32_t width,
                              size_t shared_bytes, bool launch_forward) {
        for (size_t offset = 0; offset < poly_count; offset += max_grid_y)
        {
            const size_t chunk = std::min(max_grid_y, poly_count - offset);
            if (out && required < capacity)
            {
                grid.y = static_cast<uint32_t>(chunk);
                auto &entry = out[required];
                entry.kind = kind;
                entry.grid = grid;
                entry.block = dim3{threads, 1, 1};
                entry.n = n;
                entry.len = len;
                entry.width = width;
                entry.limb_count = limb_count;
                entry.poly_offset = offset;
                entry.shared_bytes = shared_bytes;
                entry.forward = launch_forward ? 1 : 0;
            }
            ++required;
        }
    };

    const uint32_t tile_size =
        std::min(n, GPU_PREPARED_NTT_FUSED_COEFFICIENTS);
    const uint32_t stage_blocks = ((n >> 1) + threads - 1) / threads;
    const uint32_t wide_blocks = (n + threads - 1) / threads;
    if (!forward)
    {
        append_chunked(GPU_PREPARED_NTT_FUSED_LOCAL,
            dim3{n / tile_size, 0, static_cast<uint32_t>(limb_count)}, 0, 0,
            static_cast<size_t>(tile_size) * sizeof(uint64_t), false);
        if (n > GPU_PREPARED_NTT_FUSED_COEFFICIENTS &&
            n <= 16 * GPU_PREPARED_NTT_FUSED_COEFFICIENTS)
        {
            append_chunked(GPU_PREPARED_NTT_FUSED_TOP,
                dim3{n / threads, 0, static_cast<uint32_t>(limb_count)}, 0,
                n / GPU_PREPARED_NTT_FUSED_COEFFICIENTS, 0, false);
        }
        else if (n > GPU_PREPARED_NTT_FUSED_COEFFICIENTS)
        {
            for (uint32_t len = GPU_PREPARED_NTT_FUSED_COEFFICIENTS * 2; len <= n; len <<= 1)
            {
                append_chunked(GPU_PREPARED_NTT_STAGE,
                    dim3{stage_blocks, 0, static_cast<uint32_t>(limb_count)}, len, 0, 0, false);
            }
            append_chunked(GPU_PREPARED_NTT_SCALE,
                dim3{wide_blocks, 0, static_cast<uint32_t>(limb_count)}, 0, 0, 0, false);
            append_chunked(GPU_PREPARED_NTT_TWIST,
                dim3{wide_blocks, 0, static_cast<uint32_t>(limb_count)}, 0, 0, 0, false);
        }
    }
    else
    {
        if (n > GPU_PREPARED_NTT_FUSED_COEFFICIENTS &&
            n <= 16 * GPU_PREPARED_NTT_FUSED_COEFFICIENTS)
        {
            append_chunked(GPU_PREPARED_NTT_FUSED_TOP,
                dim3{n / threads, 0, static_cast<uint32_t>(limb_count)}, 0,
                n / GPU_PREPARED_NTT_FUSED_COEFFICIENTS, 0, true);
        }
        else if (n > GPU_PREPARED_NTT_FUSED_COEFFICIENTS)
        {
            append_chunked(GPU_PREPARED_NTT_TWIST,
                dim3{wide_blocks, 0, static_cast<uint32_t>(limb_count)}, 0, 0, 0, true);
            for (uint32_t len = n; len > GPU_PREPARED_NTT_FUSED_COEFFICIENTS; len >>= 1)
            {
                append_chunked(GPU_PREPARED_NTT_STAGE,
                    dim3{stage_blocks, 0, static_cast<uint32_t>(limb_count)}, len, 0, 0, true);
            }
        }
        append_chunked(GPU_PREPARED_NTT_FUSED_LOCAL,
            dim3{n / tile_size, 0, static_cast<uint32_t>(limb_count)}, 0, 0,
            static_cast<size_t>(tile_size) * sizeof(uint64_t), true);
    }
    if (out && required > capacity)
    {
        *count = required;
        return set_error("prepared NTT launch table capacity is too small");
    }
    *count = required;
    return 0;
}

int gpu_prepared_plan_const_coeff_readback_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int format,
    size_t words_per_poly, size_t coefficient_index, size_t coefficient_count,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    return plan_readback(ctx, rows, columns, level, format, words_per_poly, coefficient_index,
        coefficient_count, GPU_PREPARED_STAGE_READBACK, out, owner_layout);
}

int gpu_prepared_plan_rns_reconstruction_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level,
    size_t words_per_poly, size_t coefficient_index, size_t coefficient_count,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    // Reconstruction is a host-side composition over the coefficient
    // readback primitive. The descriptor must therefore carry READBACK keys;
    // the CRT reconstruction itself has no independent CUDA claim.
    return plan_readback(ctx, rows, columns, level, GPU_POLY_FORMAT_COEFF, words_per_poly,
        coefficient_index, coefficient_count, GPU_PREPARED_STAGE_READBACK, out,
        owner_layout);
}

int gpu_prepared_plan_borrowed_compact_store_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int format,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    const int status = plan_begin(ctx, out);
    if (status != 0) return status;
    if (!owner_layout || validate_owner_layout_for_shape(
            ctx, level, rows, columns, format, owner_layout) != 0)
        return set_error("prepared borrowed compact store owner layout is missing or invalid");
    if (level < 0 || (format != GPU_POLY_FORMAT_COEFF && format != GPU_POLY_FORMAT_EVAL))
        return set_error("prepared borrowed compact store format is invalid");
    // The native store is a no-op for an empty owner.  Keeping the descriptor
    // empty is important: there is no stream or transfer workspace to reserve.
    if (rows == 0 || columns == 0) return 0;
    if (ctx->limb_gpu_ids.empty())
        return set_error("prepared borrowed compact store has no active limb");
    dim3 first_limb{};
    if (plan_limb(ctx, level, 0, &first_limb) != 0) return -1;
    GpuPreparedResourceKey key{};
    if (device_key(ctx, first_limb.x, first_limb.x, first_limb.y,
            GPU_PREPARED_STAGE_STORE, &key) != 0)
        return -1;
    GpuPreparedWorkspaceLayout transfer{};
    if (gpu_matrix_query_compact_workspace(
            const_cast<GpuContext *>(ctx), level, rows, columns, 1, 0, 0, &transfer) != 0)
        return -1;
    PlanDescriptorWriter writer{out};
    if (writer.allocation(key, GPU_PREPARED_SUBMISSION_STREAM, 0, 0, 0, 1, -1, -1) != 0 ||
        writer.allocation(key, transfer.kind, 0, 0, transfer.bytes,
                transfer.alignment, -1, -1) != 0 ||
        writer.stream(key, GPU_PREPARED_STREAM_ADDED_SUBMISSION, 0) != 0)
        return -1;
    return 0;
}

int gpu_prepared_plan_rns_upload_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int target_format,
    int transform_to_eval, size_t bytes_per_poly, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out)
{
    // Keep one implementation of upload allocation order. The owner-aware
    // entry point is wired through the same body below; its stream selection
    // is applied by the saved owner descriptor.
    const int begin_status = plan_begin(ctx, out);
    if (begin_status != 0) return begin_status;
    if (!owner_layout || validate_owner_layout_for_shape(
            ctx, level, rows, columns, target_format, owner_layout) != 0)
        return set_error("prepared upload owner layout does not match matrix");
    if (level < 0 || (target_format != GPU_POLY_FORMAT_COEFF &&
            target_format != GPU_POLY_FORMAT_EVAL))
        return set_error("invalid prepared RNS upload plan format");
    size_t polynomial_count = 0;
    if (!plan_checked_mul_size(rows, columns, &polynomial_count) || polynomial_count == 0)
        return set_error("invalid prepared RNS upload plan matrix size");
    const size_t limb_count = static_cast<size_t>(level) + 1;
    if (limb_count > GPU_RUNTIME_MAX_LIMBS || ctx->limb_gpu_ids.size() < limb_count)
        return set_error("incomplete prepared RNS upload plan limb metadata");
    size_t expected_words = 0, expected_bytes = 0, host_bytes = 0;
    if (!plan_checked_mul_size(limb_count, static_cast<size_t>(ctx->N), &expected_words) ||
        !plan_checked_mul_size(expected_words, sizeof(uint64_t), &expected_bytes) ||
        bytes_per_poly < expected_bytes ||
        !plan_checked_mul_size(polynomial_count, bytes_per_poly, &host_bytes))
        return set_error("prepared RNS upload plan byte span is too small");
    size_t staging_bytes = 0;
    if (!plan_checked_mul_size(polynomial_count, static_cast<size_t>(ctx->N), &staging_bytes) ||
        !plan_checked_mul_size(staging_bytes, sizeof(uint64_t), &staging_bytes))
        return set_error("prepared RNS upload plan staging overflow");
    PlanDescriptorWriter writer{out};
    int status = writer.allocation(host_key(GPU_PREPARED_STAGE_UPLOAD), GPU_PREPARED_PINNED_HOST,
        0, 0, host_bytes, alignof(uint8_t), -1, -1);
    if (status != 0) return status;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        dim3 limb_id{};
        status = plan_limb(ctx, level, limb, &limb_id);
        if (status != 0) return status;
        GpuPreparedResourceKey key{};
        status = device_key(ctx, limb_id.x, limb_id.x, limb_id.y,
            GPU_PREPARED_STAGE_UPLOAD, &key);
        if (status != 0) return status;
        status = writer.allocation(key, GPU_PREPARED_TRANSFER_WORKSPACE, 0, 0,
            staging_bytes, alignof(uint64_t), -1, -1);
        if (status != 0) return status;
        status = writer.allocation(key, GPU_PREPARED_COMPLETION_EVENT, 0, 0, 0, 1, -1, -1);
        if (status != 0) return status;
        status = plan_reused_stream(writer, ctx, limb_id, GPU_PREPARED_STAGE_UPLOAD,
            limb, owner_layout);
        if (status != 0) return status;
    }
    if (transform_to_eval)
    {
        dim3 limb_id{};
        status = plan_limb(ctx, level, 0, &limb_id);
        if (status != 0) return status;
        GpuPreparedResourceKey key{};
        status = device_key(ctx, limb_id.x, limb_id.x, limb_id.y,
            GPU_PREPARED_STAGE_NTT, &key);
        if (status != 0) return status;
        size_t launch_count = 0;
        status = gpu_prepared_ntt_launch_table(static_cast<uint32_t>(ctx->N), limb_count,
            polynomial_count, 1, nullptr, 0, &launch_count);
        if (status != 0) return status;
        size_t geometry_bytes = 0;
        if (!plan_checked_mul_size(launch_count, sizeof(GpuPreparedNttLaunchLayout), &geometry_bytes))
            return set_error("prepared RNS upload transform geometry overflow");
        status = writer.allocation(key, GPU_PREPARED_PLAN_HOST_ONLY, 0, 0,
            geometry_bytes, alignof(void *), -1, -1);
        if (status != 0) return status;
        std::vector<GpuPreparedNttLaunchLayout> table(launch_count);
        size_t table_count = 0;
        status = gpu_prepared_ntt_launch_table(static_cast<uint32_t>(ctx->N), limb_count,
            polynomial_count, 1, table.data(), table.size(), &table_count);
        if (status != 0 || table_count != launch_count)
            return status != 0 ? status : set_error("prepared RNS upload NTT table drift");
        for (const auto &launch : table)
        {
            status = writer.launch(launch.kind, launch.grid, launch.len, launch.poly_offset,
                launch.limb_count, launch.forward != 0);
            if (status != 0) return status;
        }
        status = plan_reused_stream(writer, ctx, limb_id, GPU_PREPARED_STAGE_NTT, 0, owner_layout);
        if (status != 0) return status;
    }
    return 0;
}

int gpu_prepared_plan_compact_upload_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int target_format,
    size_t payload_capacity, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out)
{
    const int begin_status = plan_begin(ctx, out);
    if (begin_status != 0) return begin_status;
    if (!owner_layout ||
        (target_format != GPU_POLY_FORMAT_COEFF && target_format != GPU_POLY_FORMAT_EVAL) ||
        level < 0 || rows == 0 || columns == 0 || payload_capacity == 0)
        return set_error("invalid prepared compact replay upload layout");
    if (validate_owner_layout_for_shape(ctx, level, rows, columns, target_format, owner_layout) != 0)
        return set_error("invalid prepared compact replay upload owner layout");
    const size_t limbs = static_cast<size_t>(level) + 1;
    if (limbs > GPU_RUNTIME_MAX_LIMBS || ctx->limb_gpu_ids.size() < limbs)
        return set_error("incomplete prepared compact replay upload limbs");
    size_t polynomial_count = 0;
    size_t coefficients = 0;
    if (!plan_checked_mul_size(rows, columns, &polynomial_count) ||
        !plan_checked_mul_size(polynomial_count, static_cast<size_t>(ctx->N), &coefficients))
        return set_error("prepared compact replay upload shape overflow");
    size_t bits = 0;
    for (size_t limb = 0; limb < limbs; ++limb)
    {
        const uint64_t modulus = ctx->moduli[limb];
        if (modulus == 0) return set_error("prepared compact replay upload modulus is zero");
        bits += static_cast<size_t>(64 - __builtin_clzll(modulus));
    }
    size_t payload_bits = 0;
    size_t expected_payload = 0;
    if (!plan_checked_mul_size(coefficients, bits, &payload_bits) ||
        payload_bits > SIZE_MAX - 7 ||
        !plan_checked_mul_size((payload_bits + 7) / 8, 1, &expected_payload) ||
        payload_capacity < expected_payload || bits > UINT16_MAX)
        return set_error("prepared compact replay upload payload capacity is invalid");
    GpuPreparedWorkspaceLayout workspace{};
    if (gpu_matrix_query_compact_workspace(
            const_cast<GpuContext *>(ctx), level, rows, columns, 1, 2,
            static_cast<uint16_t>(bits), &workspace) != 0)
        return set_error("prepared compact replay upload workspace query failed");
    dim3 limb_id{};
    GpuPreparedResourceKey key{};
    int status = gpu_prepared_limb_key(ctx, level, 0, GPU_PREPARED_STAGE_UPLOAD, &limb_id, &key);
    if (status != 0) return status;
    PlanDescriptorWriter writer{out};
    status = writer.allocation(host_key(GPU_PREPARED_STAGE_UPLOAD), GPU_PREPARED_PINNED_HOST,
        0, 0, payload_capacity, 1, -1, -1);
    if (status != 0) return status;
    status = writer.allocation(key, workspace.kind, 0, 0, workspace.bytes,
        workspace.alignment, -1, -1);
    if (status != 0) return status;
    status = writer.allocation(key, GPU_PREPARED_COMPLETION_EVENT, 0, 0, 0, 1, -1, -1);
    if (status != 0) return status;
    status = plan_reused_stream(writer, ctx, limb_id, GPU_PREPARED_STAGE_UPLOAD, 0, owner_layout);
    if (status != 0) return status;
    if (target_format == GPU_POLY_FORMAT_EVAL)
    {
        size_t launches = 0;
        status = gpu_prepared_ntt_launch_table(static_cast<uint32_t>(ctx->N), limbs,
            polynomial_count, 1, nullptr, 0, &launches);
        if (status != 0) return status;
        size_t geometry_bytes = 0;
        if (!plan_checked_mul_size(launches, sizeof(GpuPreparedNttLaunchLayout), &geometry_bytes))
            return set_error("prepared compact replay upload NTT geometry overflow");
        status = gpu_prepared_limb_key(ctx, level, 0, GPU_PREPARED_STAGE_NTT, &limb_id, &key);
        if (status != 0) return status;
        status = writer.allocation(key, GPU_PREPARED_PLAN_HOST_ONLY, 0, 0,
            geometry_bytes, alignof(void *), -1, -1);
        if (status != 0) return status;
        status = plan_reused_stream(writer, ctx, limb_id, GPU_PREPARED_STAGE_NTT, 0,
            owner_layout);
        if (status != 0) return status;
        std::vector<GpuPreparedNttLaunchLayout> table(launches);
        size_t table_count = 0;
        status = gpu_prepared_ntt_launch_table(static_cast<uint32_t>(ctx->N), limbs,
            polynomial_count, 1, table.data(), table.size(), &table_count);
        if (status != 0 || table_count != launches)
            return status != 0 ? status : set_error("prepared compact replay NTT table drift");
        for (const auto &launch : table)
        {
            status = writer.launch(launch.kind, launch.grid, launch.len, launch.poly_offset,
                launch.limb_count, launch.forward != 0);
            if (status != 0) return status;
        }
    }
    return 0;
}

int gpu_prepared_plan_small_upload_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level, size_t payload_bytes,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    const int begin_status = plan_begin(ctx, out);
    if (begin_status != 0) return begin_status;
    if (!owner_layout || payload_bytes == 0 || ctx->gpu_ids.empty() || ctx->limb_gpu_ids.empty())
        return set_error("invalid prepared compact canonical replay upload layout");
    if (validate_owner_layout_for_shape(ctx, level, rows, columns,
            GPU_POLY_FORMAT_COEFF, owner_layout) != 0)
        return set_error("invalid prepared compact canonical replay upload owner layout");
    dim3 limb_id{};
    GpuPreparedResourceKey key{};
    int status = gpu_prepared_limb_key(ctx, 0, 0, GPU_PREPARED_STAGE_UPLOAD, &limb_id, &key);
    if (status != 0) return status;
    PlanDescriptorWriter writer{out};
    status = writer.allocation(host_key(GPU_PREPARED_STAGE_UPLOAD), GPU_PREPARED_PINNED_HOST,
        0, 0, payload_bytes, 1, -1, -1);
    if (status != 0) return status;
    status = writer.allocation(key, GPU_PREPARED_COMPLETION_EVENT, 0, 0, 0, 1, -1, -1);
    if (status != 0) return status;
    return plan_reused_stream(writer, ctx, limb_id, GPU_PREPARED_STAGE_UPLOAD, 0);
}

int gpu_prepared_plan_ntt_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level,
    const GpuMatrixRange *range, int forward, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out)
{
    // The NTT descriptor is structural; reproduce its small plan body only to
    // substitute the explicit owner stream assignment.
    const int begin_status = plan_begin(ctx, out);
    if (begin_status != 0) return begin_status;
    if (!owner_layout || validate_owner_layout_for_shape(
            ctx, level, rows, columns, GPU_POLY_FORMAT_EVAL, owner_layout) != 0)
        return set_error("prepared NTT owner layout does not match matrix");
    if (ctx->N < 2 || (static_cast<uint32_t>(ctx->N) & (static_cast<uint32_t>(ctx->N) - 1)) != 0)
        return set_error("invalid prepared NTT plan ring dimension");
    const size_t limb_count = static_cast<size_t>(level) + 1;
    if (level < 0 || limb_count > GPU_RUNTIME_MAX_LIMBS || ctx->limb_gpu_ids.size() < limb_count)
        return set_error("incomplete prepared NTT plan limb metadata");
    const GpuMatrixRange selected = range ? *range : GpuMatrixRange{0, rows, 0, columns};
    if (selected.row_start > selected.row_end || selected.row_end > rows ||
        selected.column_start > selected.column_end || selected.column_end > columns ||
        selected.row_end == selected.row_start || selected.column_end == selected.column_start)
        return set_error("invalid prepared NTT plan range");
    const size_t poly_count = (selected.row_end - selected.row_start) *
        (selected.column_end - selected.column_start);
    size_t launch_count = 0;
    int status = gpu_prepared_ntt_launch_table(static_cast<uint32_t>(ctx->N), limb_count,
        poly_count, forward, nullptr, 0, &launch_count);
    if (status != 0) return status;
    size_t geometry_bytes = 0;
    if (!plan_checked_mul_size(launch_count, sizeof(GpuPreparedNttLaunchLayout), &geometry_bytes))
        return set_error("prepared NTT plan geometry overflow");
    dim3 limb_id{};
    status = plan_limb(ctx, level, 0, &limb_id);
    if (status != 0) return status;
    GpuPreparedResourceKey key{};
    status = device_key(ctx, limb_id.x, limb_id.x, limb_id.y, GPU_PREPARED_STAGE_NTT, &key);
    if (status != 0) return status;
    PlanDescriptorWriter writer{out};
    status = writer.allocation(key, GPU_PREPARED_PLAN_HOST_ONLY, 0, 0,
        geometry_bytes, alignof(void *), -1, -1);
    if (status != 0) return status;
    std::vector<GpuPreparedNttLaunchLayout> launches(launch_count);
    size_t launch_table_count = 0;
    status = gpu_prepared_ntt_launch_table(
        static_cast<uint32_t>(ctx->N), limb_count, poly_count, forward,
        launches.data(), launches.size(), &launch_table_count);
    if (status != 0 || launch_table_count != launch_count)
        return status != 0 ? status : set_error("prepared NTT launch table count drift");
    for (const auto &launch : launches)
    {
        status = writer.launch(launch.kind, launch.grid, launch.len, launch.poly_offset,
            launch.limb_count, launch.forward != 0);
        if (status != 0) return status;
    }
    status = plan_reused_stream(writer, ctx, limb_id, GPU_PREPARED_STAGE_NTT, 0, owner_layout);
    if (status != 0) return status;
    out->scratch_owner_layout = *owner_layout;
    return 0;
}

int gpu_prepared_plan_sampling_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, size_t full_ncol,
    size_t col_offset, int level, int format, int dist_type,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    const int begin_status = plan_begin(ctx, out);
    if (begin_status != 0) return begin_status;
    if (format != GPU_POLY_FORMAT_COEFF && format != GPU_POLY_FORMAT_EVAL)
        return set_error("prepared sampling format is invalid");
    if (dist_type < GPU_MATRIX_DIST_UNIFORM || dist_type > GPU_MATRIX_DIST_TERNARY)
        return set_error("prepared sampling distribution is invalid");
    if (level < 0 || static_cast<size_t>(level) >= GPU_RUNTIME_MAX_LIMBS)
        return set_error("prepared sampling level is invalid");
    if (!owner_layout || validate_owner_layout_for_shape(
            ctx, level, rows, columns, format, owner_layout) != 0)
        return set_error("prepared sampling owner layout is missing or invalid");
    GpuPreparedSamplingLayout structural{};
    const int query_status = gpu_matrix_query_sampling_layout(
        static_cast<size_t>(ctx->N), static_cast<size_t>(level) + 1, rows, columns,
        full_ncol, col_offset, format, dist_type,
        ctx->gpu_ids.empty() ? -1 : ctx->gpu_ids.front(), &structural);
    if (query_status != 0) return query_status;
    if (structural.rows != rows || structural.columns != columns ||
        structural.format != format || structural.dist_type != dist_type)
        return set_error("prepared sampling query changed its structural contract");

    dim3 first_limb{};
    int status = plan_limb(ctx, level, 0, &first_limb);
    if (status != 0) return status;
    if (first_limb.x >= ctx->gpu_ids.size())
        return set_error("prepared sampling first limb is outside the context");
    GpuPreparedResourceKey key{};
    status = device_key(ctx, first_limb.x, first_limb.x, first_limb.y,
        GPU_PREPARED_STAGE_SAMPLING, &key);
    if (status != 0) return status;
    PlanDescriptorWriter writer{out};
    // Sampling writes the matrix on the first limb stream and records one
    // completion marker for every active limb. No sampler workspace is
    // currently allocated by the native implementation, so zero remains an
    // exact value rather than a guessed reservation.
    for (size_t limb = 0; limb <= static_cast<size_t>(level); ++limb)
    {
        dim3 limb_id{};
        status = plan_limb(ctx, level, limb, &limb_id);
        if (status != 0) return status;
        GpuPreparedResourceKey limb_key{};
        status = device_key(ctx, limb_id.x, limb_id.x, limb_id.y,
            GPU_PREPARED_STAGE_SAMPLING, &limb_key);
        if (status != 0) return status;
        status = writer.allocation(limb_key, GPU_PREPARED_COMPLETION_EVENT,
            0, 0, 0, 1, -1, -1);
        if (status != 0) return status;
    }
    status = plan_reused_stream(writer, ctx, first_limb,
        GPU_PREPARED_STAGE_SAMPLING, 0, owner_layout);
    if (status != 0) return status;

    // Evaluation sampling appends the host geometry owned by its fixed NTT
    // sub-plan. The submission stream is the same first-limb stream, so it is
    // deliberately not duplicated in the footprint.
    if (format == GPU_POLY_FORMAT_EVAL && structural.polynomial_count != 0)
    {
        GpuPreparedNttLayout ntt{};
        status = gpu_matrix_query_ntt_layout(
            static_cast<size_t>(ctx->N), static_cast<size_t>(level) + 1,
            structural.polynomial_count, first_limb.x, true, &ntt);
        if (status != 0) return status;
        size_t geometry_bytes = 0;
        if (!plan_checked_mul_size(ntt.launch_count,
                sizeof(GpuPreparedNttLaunchLayout), &geometry_bytes))
            return set_error("prepared sampling transform geometry overflow");
        GpuPreparedResourceKey ntt_key{};
        status = device_key(ctx, first_limb.x, first_limb.x, first_limb.y,
            GPU_PREPARED_STAGE_NTT, &ntt_key);
        if (status != 0) return status;
        status = writer.allocation(ntt_key, GPU_PREPARED_PLAN_HOST_ONLY,
            0, 0, geometry_bytes, alignof(void *), -1, -1);
        if (status != 0) return status;
        std::vector<GpuPreparedNttLaunchLayout> launches(ntt.launch_count);
        size_t launch_table_count = 0;
        status = gpu_prepared_ntt_launch_table(
            static_cast<uint32_t>(ctx->N), static_cast<size_t>(level) + 1,
            structural.polynomial_count, 1, launches.data(), launches.size(),
            &launch_table_count);
        if (status != 0 || launch_table_count != ntt.launch_count)
            return status != 0 ? status : set_error("prepared sampling NTT launch table count drift");
        for (const auto &launch : launches)
        {
            status = writer.launch(launch.kind, launch.grid, launch.len, launch.poly_offset,
                launch.limb_count, launch.forward != 0);
            if (status != 0) return status;
        }
    }
    out->scratch_owner_layout = *owner_layout;
    return 0;
}

int gpu_prepared_plan_arithmetic_with_owner(
    const GpuContext *ctx, size_t ring_dimension, size_t limb_count,
    size_t left_rows, size_t left_columns, size_t right_rows, size_t right_columns,
    size_t output_rows, size_t output_columns, size_t column_start,
    size_t group_count, size_t term_count, int kind, int device,
    int evaluation_format, int thin, int lazy_reduction,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    const int begin_status = plan_begin(ctx, out);
    if (begin_status != 0) return begin_status;
    if (!owner_layout || gpu_prepared_owner_layout_matches(ctx, owner_layout) != 0)
        return set_error("prepared arithmetic owner layout is missing or invalid");
    if (device < 0 || std::find(ctx->gpu_ids.begin(), ctx->gpu_ids.end(), device) == ctx->gpu_ids.end())
        return set_error("prepared arithmetic device is outside the context");
    GpuPreparedArithmeticLayout structural{};
    const int query_status = gpu_matrix_query_arithmetic_layout(
        ring_dimension, limb_count, left_rows, left_columns, right_rows, right_columns,
        output_rows, output_columns, column_start, group_count, term_count, kind, device,
        evaluation_format, thin, lazy_reduction, &structural);
    if (query_status != 0) return query_status;
    if (structural.limb_count != limb_count || structural.device != device)
        return set_error("prepared arithmetic query changed its structural contract");
    PlanDescriptorWriter writer{out};
    dim3 first_limb{};
    const int first_status = plan_limb(ctx, static_cast<int>(limb_count - 1), 0, &first_limb);
    if (first_status != 0) return first_status;
    GpuPreparedResourceKey first_key{};
    const int key_status = device_key(
        ctx, first_limb.x, first_limb.x, first_limb.y, GPU_PREPARED_STAGE_ARITHMETIC, &first_key);
    if (key_status != 0) return key_status;
    if (structural.workspace_bytes != 0 && writer.allocation(
            first_key, GPU_PREPARED_BATCH_WORKSPACE, structural.output_rows,
            structural.output_columns, structural.workspace_bytes, structural.alignment,
            static_cast<int>(limb_count - 1), evaluation_format ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF) != 0)
        return -1;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        dim3 limb_id{};
        const int limb_status = plan_limb(ctx, static_cast<int>(limb_count - 1), limb, &limb_id);
        if (limb_status != 0) return limb_status;
        GpuPreparedResourceKey key{};
        const int status = device_key(
            ctx, limb_id.x, limb_id.x, limb_id.y, GPU_PREPARED_STAGE_ARITHMETIC, &key);
        if (status != 0) return status;
        if (writer.allocation(key, GPU_PREPARED_COMPLETION_EVENT, 0, 0, 0, 1, -1, -1) != 0)
            return -1;
        const int stream_status = plan_reused_stream(
            writer, ctx, limb_id, GPU_PREPARED_STAGE_ARITHMETIC, limb, owner_layout);
        if (stream_status != 0) return stream_status;
    }
    out->scratch_owner_layout = *owner_layout;
    return 0;
}

int gpu_prepared_plan_schedule(
    const GpuPreparedPlanDescriptor *const *plans, size_t count, GpuPreparedPlanDescriptor *out)
{
    if (!out)
    {
        return set_error("missing prepared schedule plan output");
    }
    *out = GpuPreparedPlanDescriptor{};
    if (count != 0 && !plans)
    {
        return set_error("invalid prepared schedule plan list");
    }
    PlanDescriptorWriter writer{out};
    // Preserve each member's ordered non-event resources and launch
    // substages. Completion events belong to the merged schedule, not to the
    // member descriptors: retaining every member event would create several
    // events for one shared physical stream.
    for (size_t plan_index = 0; plan_index < count; ++plan_index)
    {
        if (!plans[plan_index] || append_plan_descriptor_without_completion_events(
                out, plans[plan_index]) != 0)
            return set_error("invalid prepared schedule member descriptor");
    }
    // Save one canonical schedule stream per distinct physical stream. Role is
    // deliberately normalized to SCHEDULE; owner/context/partition/device/
    // limbs plus origin/pool slot define the physical stream identity.
    for (size_t plan_index = 0; plan_index < count; ++plan_index)
    {
        const auto *plan = plans[plan_index];
        if (!plan)
        {
            return set_error("missing prepared schedule member plan");
        }
        for (size_t index = 0; index < plan->stream_count; ++index)
        {
            const auto &entry = plan->streams[index];
            GpuPreparedResourceKey key = entry.key;
            key.role = GPU_PREPARED_STAGE_SCHEDULE;
            bool known = false;
            for (size_t previous = 0; previous < writer.descriptor->stream_count; ++previous)
            {
                const auto &existing = writer.descriptor->streams[previous];
                const bool same_stream = existing.key.execution_owner_identity ==
                        key.execution_owner_identity &&
                    existing.key.context_identity == key.context_identity &&
                    existing.key.instance == key.instance &&
                    existing.key.partition == key.partition &&
                    existing.key.device == key.device &&
                    existing.key.limb_x == key.limb_x &&
                    existing.key.limb_y == key.limb_y &&
                    existing.key.role == key.role &&
                    existing.origin == entry.origin && existing.pool_slot == entry.pool_slot;
                if (same_stream)
                {
                    known = true;
                    break;
                }
            }
            if (known)
            {
                continue;
            }
            const int status = writer.stream(key, entry.origin, entry.pool_slot);
            if (status != 0)
            {
                return status;
            }
        }
    }
    for (size_t index = 0; index < writer.descriptor->stream_count; ++index)
    {
        const auto &entry = writer.descriptor->streams[index];
        const int status = writer.allocation(
            entry.key, GPU_PREPARED_COMPLETION_EVENT, 0, 0, 0, 1, -1, -1);
        if (status != 0)
        {
            return status;
        }
    }
    return 0;
}

namespace
{
int plan_scalar_stage(const GpuContext *ctx, int role, size_t pinned_bytes, size_t bytes,
    int level, int format, int events, GpuPreparedPlanDescriptor *out,
    const GpuPreparedOwnerLayout *owner_layout = nullptr)
{
    const int status = plan_begin(ctx, out);
    if (status != 0) return status;
    dim3 limb{};
    if (plan_limb(ctx, std::max(level, 0), 0, &limb) != 0) return -1;
    GpuPreparedResourceKey key{};
    if (device_key(ctx, limb.x, limb.x, limb.y, role, &key) != 0) return -1;
    PlanDescriptorWriter writer{out};
    if (pinned_bytes != 0 &&
        writer.allocation(key, GPU_PREPARED_PINNED_HOST, 0, 0, pinned_bytes, 8, -1, -1) != 0)
        return -1;
    if (bytes != 0 &&
        writer.allocation(key, GPU_PREPARED_BATCH_WORKSPACE, 0, 0, bytes, 8, level, format) != 0)
        return -1;
    for (int i = 0; i < events; ++i)
        if (writer.allocation(key, GPU_PREPARED_COMPLETION_EVENT, 0, 0, 0, 1, -1, -1) != 0)
            return -1;
    return plan_reused_stream(writer, ctx, limb, role, 0, owner_layout);
}
}

int gpu_prepared_plan_scalar_buffer(
    const GpuContext *ctx, size_t count, size_t words, GpuPreparedPlanDescriptor *out)
{
    if (count == 0 || words == 0 || words == SIZE_MAX || count > SIZE_MAX / (words + 1) / 8)
        return set_error("invalid scalar buffer plan");
    const size_t bytes = count * (words + 1) * 8;
    return plan_scalar_stage(ctx, GPU_PREPARED_STAGE_SCALAR_BUFFER, bytes, bytes, 0,
        GPU_POLY_FORMAT_COEFF, 2, out);
}

int gpu_prepared_plan_scalar_op(
    const GpuContext *ctx, size_t left_words, size_t right_words, size_t output_words,
    size_t candidate_count, GpuPreparedPlanDescriptor *out)
{
    const size_t largest = std::max({left_words, right_words, output_words});
    if (largest == SIZE_MAX || candidate_count > (SIZE_MAX - (largest + 1) * 3 * 8) / sizeof(PreparedScalarView))
        return set_error("scalar operation plan workspace overflow");
    return plan_scalar_stage(ctx, GPU_PREPARED_STAGE_SCALAR_OP, 0,
        (largest + 1) * 3 * 8 + candidate_count * sizeof(PreparedScalarView),
        0, GPU_POLY_FORMAT_COEFF, 1, out);
}

int gpu_prepared_plan_scalar_matrix_select(
    const GpuContext *ctx, size_t rows, size_t columns, size_t n, int level,
    size_t count, GpuPreparedPlanDescriptor *out)
{
    size_t elements = 0;
    if (count == 0 || !plan_checked_mul_size(rows, columns, &elements) ||
        !plan_checked_mul_size(elements, n, &elements) || elements > SIZE_MAX - 255)
        return set_error("scalar matrix selection plan dimensions overflow");
    const int status = plan_scalar_stage(ctx, GPU_PREPARED_STAGE_SCALAR_MATRIX_SELECT, 0,
        count * sizeof(PreparedScalarMatrixCopy), level, GPU_POLY_FORMAT_COEFF, 0, out);
    if (status == 0) {
        out->allocations[0].rows = (elements + 255) / 256;
        out->allocations[0].columns = static_cast<size_t>(level + 1);
    }
    return status;
}

int plan_threshold_impl(
    const GpuContext *ctx, size_t count, size_t plaintext_words,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    if (!ctx || count == 0 || plaintext_words == 0 || ctx->moduli.empty())
        return set_error("invalid threshold plan");
    std::vector<uint64_t> modulus;
    const std::vector<uint64_t> active(ctx->moduli.begin(), ctx->moduli.end());
    if (!serde_compute_modulus_words_le(active, &modulus)) return set_error("threshold modulus layout");
    const size_t scratch = plaintext_words == 1 ? 0 : modulus.size() + 2 * plaintext_words + 1;
    size_t words = 0, bytes = 0;
    if (!plan_checked_mul_size(count, scratch, &words) || words > SIZE_MAX - plaintext_words ||
        !plan_checked_mul_size(words + plaintext_words, sizeof(uint64_t), &bytes))
        return set_error("threshold plan workspace overflow");
    return plan_scalar_stage(ctx, GPU_PREPARED_STAGE_THRESHOLD, 0, bytes,
        static_cast<int>(ctx->moduli.size() - 1), GPU_POLY_FORMAT_COEFF, 0, out, owner_layout);
}

int gpu_prepared_plan_threshold(
    const GpuContext *ctx, size_t count, size_t plaintext_words, GpuPreparedPlanDescriptor *out)
{
    return plan_threshold_impl(ctx, count, plaintext_words, nullptr, out);
}

int gpu_prepared_plan_threshold_with_owner(
    const GpuContext *ctx, size_t count, size_t plaintext_words,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    if (!owner_layout) return set_error("missing threshold owner layout");
    return plan_threshold_impl(ctx, count, plaintext_words, owner_layout, out);
}

static int gpu_prepared_plan_scalar_pack_impl(
    const GpuContext *ctx, size_t count, size_t coefficient_bits, int level, int output_format,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    size_t view_bytes = 0, expected_count = 0;
    if (!ctx || count == 0 || level < 0 || static_cast<size_t>(level) >= ctx->moduli.size() ||
        (output_format != GPU_POLY_FORMAT_COEFF && output_format != GPU_POLY_FORMAT_EVAL) ||
        !plan_checked_mul_size(count, sizeof(PreparedScalarView), &view_bytes) ||
        !plan_checked_mul_size(static_cast<size_t>(ctx->N), coefficient_bits ? coefficient_bits : 1,
            &expected_count) || count != expected_count)
        return set_error("invalid scalar pack plan count");
    const int status = plan_scalar_stage(ctx, GPU_PREPARED_STAGE_SCALAR_PACK, 0,
        view_bytes, level, output_format, 1, out, owner_layout);
    if (status != 0 || output_format != GPU_POLY_FORMAT_EVAL) return status;
    dim3 limb{};
    if (plan_limb(ctx, level, 0, &limb) != 0) return -1;
    GpuPreparedResourceKey key{};
    if (device_key(ctx, limb.x, limb.x, limb.y, GPU_PREPARED_STAGE_NTT, &key) != 0)
        return -1;
    GpuPreparedNttLayout ntt{};
    const int query_status = gpu_matrix_query_ntt_layout(
        static_cast<size_t>(ctx->N), static_cast<size_t>(level) + 1, 1,
        static_cast<int>(limb.x < ctx->gpu_ids.size() ? ctx->gpu_ids[limb.x] : -1), true, &ntt);
    if (query_status != 0) return query_status;
    size_t geometry_bytes = 0;
    if (!plan_checked_mul_size(ntt.launch_count, sizeof(GpuPreparedNttLaunchLayout), &geometry_bytes))
        return set_error("scalar pack NTT geometry overflow");
    PlanDescriptorWriter writer{out};
    if (writer.allocation(key, GPU_PREPARED_PLAN_HOST_ONLY, 0, 0, geometry_bytes,
            alignof(void *), -1, -1) != 0)
        return -1;
    std::vector<GpuPreparedNttLaunchLayout> launches(ntt.launch_count);
    size_t launch_count = 0;
    const int table_status = gpu_prepared_ntt_launch_table(
        static_cast<uint32_t>(ctx->N), static_cast<size_t>(level) + 1, 1, 1,
        launches.data(), launches.size(), &launch_count);
    if (table_status != 0 || launch_count != ntt.launch_count)
        return table_status != 0 ? table_status : set_error("scalar pack NTT launch table drift");
    for (const auto &launch : launches)
        if (writer.launch(launch.kind, launch.grid, launch.len, launch.poly_offset,
                launch.limb_count, launch.forward != 0) != 0)
            return -1;
    return plan_reused_stream(writer, ctx, limb, GPU_PREPARED_STAGE_NTT, 0, owner_layout);
}

int gpu_prepared_plan_scalar_pack_with_owner(
    const GpuContext *ctx, size_t count, size_t coefficient_bits, int level, int output_format,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    if (!owner_layout) return set_error("missing scalar pack owner layout");
    return gpu_prepared_plan_scalar_pack_impl(ctx, count, coefficient_bits, level, output_format,
        owner_layout, out);
}

static int gpu_prepared_plan_small_rhs_impl(
    const GpuContext *ctx, int level, size_t inner, size_t columns,
    size_t residency_budget_bytes, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out)
{
    const int status = plan_begin(ctx, out);
    if (status != 0) return status;
    size_t narrow = 0, wide = 0;
    if (level < 0 || inner == 0 || columns == 0 ||
        gpu_matrix_query_small_rhs_workspace_bytes(ctx, level, inner, columns, &narrow, &wide) != 0 ||
        wide > SIZE_MAX - narrow || narrow + wide > residency_budget_bytes)
        return set_error("compact RHS plan workspace exceeds residency budget");
    dim3 limb{};
    if (plan_limb(ctx, level, 0, &limb) != 0) return -1;
    GpuPreparedResourceKey key{};
    if (device_key(ctx, limb.x, limb.x, limb.y, GPU_PREPARED_STAGE_SMALL_RHS, &key) != 0) return -1;
    PlanDescriptorWriter writer{out};
    if (writer.allocation(key, GPU_PREPARED_COMPACT_WORKSPACE, 0, 0, narrow, alignof(uint32_t), level, GPU_POLY_FORMAT_EVAL) != 0 ||
        writer.allocation(key, GPU_PREPARED_COMPACT_WORKSPACE, 0, 0, wide, alignof(uint64_t), level, GPU_POLY_FORMAT_EVAL) != 0)
        return -1;
    if (plan_reused_stream(writer, ctx, limb, GPU_PREPARED_STAGE_SMALL_RHS, 0, owner_layout) != 0)
        return -1;
    size_t poly_count = 0;
    if (!plan_checked_mul_size(inner, columns, &poly_count) || poly_count > UINT32_MAX)
        return set_error("compact RHS launch geometry overflow");
    constexpr uint32_t kCompactNttSuffixSize = 4096;
    constexpr size_t kSmallThreads = 256;
    size_t typed_limb_offset[2] = {0, 0};
    bool previous_narrow = false;
    bool have_group = false;
    size_t group_limb_offset = 0, group_limb_count = 0;
    auto append_group = [&](size_t limb_offset, size_t limb_count, bool narrow_group) -> int {
        size_t butterflies = 0, first_blocks = 0, suffix_blocks = 0;
        if (!plan_checked_mul_size(limb_count, poly_count, &butterflies) ||
            !plan_checked_mul_size(butterflies, static_cast<size_t>(ctx->N) / 2, &butterflies) ||
            butterflies > SIZE_MAX - 255 ||
            !plan_checked_mul_size(poly_count, static_cast<size_t>(ctx->N) / kCompactNttSuffixSize, &suffix_blocks) ||
            suffix_blocks > UINT32_MAX)
            return set_error("compact RHS launch geometry overflow");
        first_blocks = (butterflies + kSmallThreads - 1) / kSmallThreads;
        if (first_blocks > UINT32_MAX) return set_error("compact RHS launch geometry overflow");
        if (static_cast<uint32_t>(ctx->N) <= kCompactNttSuffixSize) {
            return writer.launch(0, dim3(static_cast<uint32_t>(poly_count), static_cast<uint32_t>(limb_count)), 0,
                limb_offset, limb_count, narrow_group);
        }
        if (writer.launch(1, dim3(static_cast<uint32_t>(first_blocks)), 0,
                limb_offset, limb_count, narrow_group) != 0)
            return -1;
        for (uint32_t len = static_cast<uint32_t>(ctx->N) >> 1; len > kCompactNttSuffixSize; len >>= 1)
            if (writer.launch(2, dim3(static_cast<uint32_t>(first_blocks)), len,
                    limb_offset, limb_count, narrow_group) != 0)
                return -1;
        return writer.launch(3, dim3(static_cast<uint32_t>(suffix_blocks), static_cast<uint32_t>(limb_count)), 0,
            limb_offset, limb_count, narrow_group);
    };
    for (size_t limb_index = 0; limb_index <= static_cast<size_t>(level); ++limb_index) {
        if (limb_index >= ctx->limb_prime_ids.size())
            return set_error("missing compact RHS prime metadata");
        const int prime = ctx->limb_prime_ids[limb_index];
        if (prime < 0 || static_cast<size_t>(prime) >= ctx->moduli.size())
            return set_error("invalid compact RHS prime metadata");
        const bool narrow_modulus = ctx->moduli[static_cast<size_t>(prime)] <= UINT32_MAX;
        if (!have_group || narrow_modulus != previous_narrow) {
            if (have_group && append_group(group_limb_offset, group_limb_count, previous_narrow) != 0)
                return -1;
            have_group = true;
            previous_narrow = narrow_modulus;
            group_limb_offset = limb_index;
            group_limb_count = 1;
            ++typed_limb_offset[narrow_modulus ? 0 : 1];
        } else {
            ++group_limb_count;
        }
    }
    if (have_group && append_group(group_limb_offset, group_limb_count, previous_narrow) != 0)
        return -1;
    return 0;
}

int gpu_prepared_plan_small_rhs_with_owner(
    const GpuContext *ctx, int level, size_t inner, size_t columns,
    size_t residency_budget_bytes, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out)
{
    return gpu_prepared_plan_small_rhs_impl(
        ctx, level, inner, columns, residency_budget_bytes, owner_layout, out);
}

static int plan_matrix_completion_stage(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int format,
    int stage_role, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out, size_t workspace_bytes = 0,
    size_t workspace_alignment = alignof(uint64_t), int workspace_kind = GPU_PREPARED_BATCH_WORKSPACE,
    bool rectangular_launch = false, bool include_completion_events = true)
{
    const int begin_status = plan_begin(ctx, out);
    if (begin_status != 0) return begin_status;
    if (!owner_layout || validate_owner_layout_for_shape(
            ctx, level, rows, columns, format, owner_layout) != 0)
        return set_error("prepared matrix stage owner layout is missing or invalid");
    if (stage_role < GPU_PREPARED_STAGE_UPLOAD || stage_role > GPU_PREPARED_STAGE_SAMPLING)
        return set_error("prepared matrix stage role is invalid");
    if (format != GPU_POLY_FORMAT_COEFF && format != GPU_POLY_FORMAT_EVAL)
        return set_error("prepared matrix stage format is invalid");
    if (level < 0 || static_cast<size_t>(level) >= GPU_RUNTIME_MAX_LIMBS ||
        rows == 0 || columns == 0 || ctx->limb_gpu_ids.size() <= static_cast<size_t>(level))
        return set_error("prepared matrix stage dimensions are invalid");
    dim3 limb{};
    int status = plan_limb(ctx, level, 0, &limb);
    if (status != 0) return status;
    GpuPreparedResourceKey key{};
    status = device_key(ctx, limb.x, limb.x, limb.y, stage_role, &key);
    if (status != 0) return status;
    PlanDescriptorWriter writer{out};
    if (workspace_bytes != 0)
    {
        status = writer.allocation(key, workspace_kind, rows, columns, workspace_bytes,
            workspace_alignment, level, format);
        if (status != 0) return status;
    }
    for (size_t index = 0; include_completion_events && index <= static_cast<size_t>(level); ++index)
    {
        dim3 limb_id{};
        status = plan_limb(ctx, level, index, &limb_id);
        if (status != 0) return status;
        GpuPreparedResourceKey limb_key{};
        status = device_key(ctx, limb_id.x, limb_id.x, limb_id.y, stage_role, &limb_key);
        if (status != 0 || writer.allocation(limb_key, GPU_PREPARED_COMPLETION_EVENT,
                0, 0, 0, 1, -1, -1) != 0)
            return status != 0 ? status : -1;
    }
    // Rectangular transforms have one fixed launch.  Persist the complete
    // geometry so a prepared bind can consume it without rediscovering the
    // launch shape (or calling the legacy planner).
    if (rectangular_launch)
    {
        GpuPreparedRectLayout rect{};
        const int rect_status = gpu_matrix_query_input_copy_layout(
            static_cast<size_t>(ctx->N), static_cast<size_t>(level + 1),
            rows, columns, ctx->gpu_ids[limb.x], &rect);
        if (rect_status != 0) return rect_status;
        if (writer.launch(0, dim3(rect.grid_x, rect.grid_y, rect.grid_z),
                static_cast<uint32_t>(ctx->N), 0, static_cast<size_t>(level + 1), false) != 0)
            return -1;
    }
    status = plan_reused_stream(writer, ctx, limb, stage_role, 0, owner_layout);
    if (status != 0) return status;
    out->scratch_owner_layout = *owner_layout;
    return 0;
}

extern "C" int gpu_prepared_plan_input_copy_with_owner(
    const GpuContext *ctx, size_t rows, size_t columns, int level, int format,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    return plan_matrix_completion_stage(ctx, rows, columns, level, format,
        GPU_PREPARED_STAGE_TRANSFORM, owner_layout, out, 0, alignof(uint64_t),
        GPU_PREPARED_BATCH_WORKSPACE, true);
}

extern "C" int gpu_prepared_plan_transpose_with_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t output_rows, size_t output_columns, int level, int format,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    if (source_rows != output_columns || source_columns != output_rows)
        return set_error("prepared transpose dimensions are not transposed");
    const int status = plan_matrix_completion_stage(ctx, source_rows, source_columns, level, format,
        GPU_PREPARED_STAGE_TRANSFORM, owner_layout, out, 0, alignof(uint64_t),
        GPU_PREPARED_BATCH_WORKSPACE, true);
    if (status != 0) return status;
    // Replace the generic rectangular launch with transpose's capped grid-X
    // rule.  The event and stream footprint remains identical.
    if (!out || out->launch_count != 1) return set_error("prepared transpose launch is missing");
    GpuPreparedRectLayout rect{};
    if (gpu_matrix_query_transpose_layout(
            static_cast<size_t>(ctx->N), static_cast<size_t>(level + 1),
            source_rows, source_columns, ctx->gpu_ids[ctx->limb_gpu_ids[0].x], &rect) != 0)
        return -1;
    out->launches[0].grid = dim3(rect.grid_x, rect.grid_y, rect.grid_z);
    return 0;
}

extern "C" int gpu_prepared_plan_centered_rebase_with_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t target_rows, size_t target_columns, int source_level, int target_level,
    int source_format, int target_format, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out)
{
    if (source_rows != target_rows || source_columns != target_columns || source_level != 0 ||
        source_format != GPU_POLY_FORMAT_COEFF ||
        (target_format != GPU_POLY_FORMAT_COEFF && target_format != GPU_POLY_FORMAT_EVAL))
        return set_error("prepared centered rebase structural contract is invalid");
    GpuPreparedPlanDescriptor primary{};
    const int plan_status = plan_matrix_completion_stage(ctx, target_rows, target_columns, target_level,
        target_format, GPU_PREPARED_STAGE_TRANSFORM, owner_layout, out);
    if (plan_status != 0) return plan_status;
    const int launch_status = plan_matrix_launch(out, ctx, target_rows, target_columns,
        static_cast<size_t>(target_level) + 1, static_cast<size_t>(target_level) + 1, 128, nullptr);
    if (launch_status != 0) return launch_status;
    if (target_format != GPU_POLY_FORMAT_EVAL)
        return 0;
    primary = *out;
    GpuPreparedPlanDescriptor forward{};
    const int ntt_status = gpu_prepared_plan_ntt_with_owner(ctx, target_rows, target_columns,
        target_level, nullptr, 1, owner_layout, &forward);
    if (ntt_status != 0) return ntt_status;
    *out = GpuPreparedPlanDescriptor{};
    if (append_plan_descriptor(out, &primary) != 0 || append_plan_descriptor(out, &forward) != 0)
        return -1;
    return 0;
}

extern "C" int gpu_prepared_plan_gadget_decompose_with_source_format_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t output_rows, int level, int source_format, int format, uint32_t base_bits, int small,
    size_t dropped_moduli, const GpuPreparedOwnerLayout *output_owner_layout,
    const GpuPreparedOwnerLayout *source_owner_layout,
    GpuPreparedPlanDescriptor *out)
{
    if (!ctx || level < 0 || (source_format != GPU_POLY_FORMAT_COEFF &&
            source_format != GPU_POLY_FORMAT_EVAL) ||
        (format != GPU_POLY_FORMAT_COEFF && format != GPU_POLY_FORMAT_EVAL) ||
        !output_owner_layout || !source_owner_layout ||
        base_bits == 0 || base_bits >= (small ? 64U : 63U) ||
        (small && dropped_moduli != 0) || dropped_moduli > static_cast<size_t>(level))
        return set_error("prepared gadget decomposition structural contract is invalid");
    if (validate_owner_layout_for_shape(ctx, level, output_rows, source_columns,
            format, output_owner_layout) != 0 ||
        validate_owner_layout_for_shape(ctx, level, source_rows, source_columns,
            source_format, source_owner_layout) != 0)
        return set_error("prepared gadget decomposition owner layout does not match its shape");
    GpuMatrixDecomposeWorkspaceBytes requirements{};
    int status = gpu_matrix_query_decompose_workspace_bytes(
        ctx, level, source_rows, source_columns, format, base_bits, small,
        dropped_moduli, &requirements);
    if (status != 0) return status;
    if (requirements.output_rows != output_rows)
        return set_error("prepared gadget decomposition output shape differs");
    GpuPreparedPlanDescriptor primary{};
    status = plan_matrix_completion_stage(ctx, output_rows, source_columns, level, format,
        GPU_PREPARED_STAGE_TRANSFORM, output_owner_layout, &primary,
        requirements.correction.workspace_bytes, requirements.correction.alignment,
        GPU_PREPARED_BATCH_WORKSPACE, false, false);
    if (status != 0) return status;
    if (source_format == GPU_POLY_FORMAT_EVAL || format == GPU_POLY_FORMAT_EVAL)
    {
        // Evaluation-domain decomposition has two immutable transform
        // substages: inverse source normalization, then forward output
        // normalization. Keep them in the descriptor's ordered records.
        GpuPreparedPlanDescriptor inverse{};
        if (source_format == GPU_POLY_FORMAT_EVAL) {
            status = gpu_prepared_plan_ntt_with_owner(ctx, source_rows, source_columns, level,
                nullptr, 0, source_owner_layout, &inverse);
            if (status != 0) return status;
        }
        GpuPreparedPlanDescriptor forward{};
        if (format == GPU_POLY_FORMAT_EVAL) {
            status = gpu_prepared_plan_ntt_with_owner(ctx, output_rows, source_columns, level,
                nullptr, 1, output_owner_layout, &forward);
            if (status != 0) return status;
        }
        *out = GpuPreparedPlanDescriptor{};
        GpuPreparedPlanDescriptor scratch{};
        GpuPreparedResourceKey scratch_key{};
        dim3 scratch_limb{};
        if (gpu_prepared_limb_key(ctx, level, 0, GPU_PREPARED_STAGE_TRANSFORM,
                &scratch_limb, &scratch_key) != 0)
            return -1;
        scratch.allocation_count = 1;
        scratch.allocations[0] = {
            scratch_key, GPU_PREPARED_MATRIX, source_rows, source_columns, 0, 1,
            level, GPU_POLY_FORMAT_COEFF};
        if ((source_format == GPU_POLY_FORMAT_EVAL || dropped_moduli != 0) &&
            append_plan_descriptor(out, &scratch) != 0)
            return -1;
        if ((source_format == GPU_POLY_FORMAT_EVAL && append_plan_descriptor(out, &inverse) != 0) ||
            append_plan_descriptor(out, &primary) != 0 ||
            (format == GPU_POLY_FORMAT_EVAL && append_plan_descriptor(out, &forward) != 0))
            return -1;
        // The descriptor now has an explicit owner for its one physical
        // coefficient scratch phase.  The inverse and forward substages may
        // legitimately belong to different matrix stores; their differing
        // stage owners are not a scratch-owner conflict.
        out->scratch_owner_layout = *source_owner_layout;
        out->scratch_owner_conflict = 0;
        return 0;
    }
    if (format == GPU_POLY_FORMAT_COEFF && dropped_moduli == 0)
    {
        *out = primary;
        return 0;
    }
    *out = GpuPreparedPlanDescriptor{};
    GpuPreparedPlanDescriptor scratch{};
    GpuPreparedResourceKey scratch_key{};
    dim3 scratch_limb{};
    if (gpu_prepared_limb_key(ctx, level, 0, GPU_PREPARED_STAGE_TRANSFORM,
            &scratch_limb, &scratch_key) != 0)
        return -1;
    scratch.allocation_count = 1;
    scratch.allocations[0] = {
        scratch_key, GPU_PREPARED_MATRIX, source_rows, source_columns, 0, 1,
        level, GPU_POLY_FORMAT_COEFF};
    if (append_plan_descriptor(out, &scratch) != 0 ||
        append_plan_descriptor(out, &primary) != 0)
        return -1;
    // The descriptor now has an explicit owner for its one physical
    // coefficient scratch phase.  The output correction stage can have a
    // different owner without changing that scratch provenance.
    out->scratch_owner_layout = *source_owner_layout;
    out->scratch_owner_conflict = 0;
    return 0;
}

extern "C" int gpu_prepared_plan_modulus_conversion_with_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t target_rows, size_t target_columns, int source_level, int target_level,
    int source_format, int target_format, int mode, size_t digit_size,
    uint64_t plaintext_modulus, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out)
{
    if (source_rows != target_rows || source_columns != target_columns || source_level < 0 ||
        target_level < 0 || source_format < GPU_POLY_FORMAT_COEFF ||
        source_format > GPU_POLY_FORMAT_EVAL || target_format < GPU_POLY_FORMAT_COEFF ||
        target_format > GPU_POLY_FORMAT_EVAL || mode < GPU_MATRIX_CRT_CONVERT_MODULUS ||
        mode > GPU_MATRIX_CRT_RNS_CONVERSION)
        return set_error("prepared modulus conversion structural contract is invalid");
    const auto operation = static_cast<GpuMatrixCrtOperation>(mode);
    GpuMatrixTransformWorkspaceBytes requirements{};
    const int status = gpu_matrix_query_crt_workspace_bytes(
        ctx, target_level, target_rows, target_columns, operation,
        static_cast<size_t>(source_level) + 1, 1,
        false, &requirements);
    if (status != 0) return status;
    (void)digit_size;
    (void)plaintext_modulus;
    const int plan_status = plan_matrix_completion_stage(ctx, target_rows, target_columns, target_level,
        target_format, GPU_PREPARED_STAGE_TRANSFORM, owner_layout, out,
        requirements.workspace_bytes, requirements.alignment);
    if (plan_status != 0) return plan_status;
    return plan_matrix_launch(out, ctx, target_rows, target_columns,
        static_cast<size_t>(target_level) + 1, 1, 128, nullptr);
}

extern "C" int gpu_prepared_plan_rns_conversion_with_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t target_rows, size_t target_columns, int source_level, int target_level,
    size_t digit_size, int normalize, uint64_t plaintext_modulus,
    const GpuPreparedOwnerLayout *owner_layout, GpuPreparedPlanDescriptor *out)
{
    if (source_rows == 0 || source_columns == 0 || target_rows == 0 || target_columns == 0 ||
        source_rows > SIZE_MAX / source_columns || target_rows > SIZE_MAX / target_columns ||
        source_level < 0 || target_level < 0 || digit_size == 0 ||
        (normalize != 0 && normalize != 1))
        return set_error("prepared RNS conversion structural contract is invalid");
    GpuMatrixTransformWorkspaceBytes requirements{};
    const int status = gpu_matrix_query_crt_workspace_bytes(
        ctx, target_level, target_rows, target_columns, GPU_MATRIX_CRT_RNS_CONVERSION,
        static_cast<size_t>(source_level) + 1, 1,
        false, &requirements);
    if (status != 0) return status;
    (void)plaintext_modulus;
    const int plan_status = plan_matrix_completion_stage(ctx, target_rows, target_columns, target_level,
        GPU_POLY_FORMAT_COEFF, GPU_PREPARED_STAGE_TRANSFORM, owner_layout, out,
        requirements.workspace_bytes, requirements.alignment);
    if (plan_status != 0) return plan_status;
    const size_t groups = target_rows / source_rows;
    if (!groups || target_rows % source_rows != 0)
        return set_error("prepared RNS conversion launch shape is invalid");
    return plan_matrix_launch(out, ctx, source_rows, source_columns,
        static_cast<size_t>(target_level) + 1, groups * (static_cast<size_t>(target_level) + 1),
        128, nullptr);
}

extern "C" int gpu_prepared_plan_crt_recompose_with_owner(
    const GpuContext *ctx, size_t source_rows, size_t source_columns,
    size_t level_count, size_t output_rows, size_t output_columns, int target_level,
    int target_format, const GpuPreparedOwnerLayout *owner_layout,
    GpuPreparedPlanDescriptor *out)
{
    if (level_count == 0 || source_rows != output_rows || source_columns != output_columns ||
        target_level < 0 || target_format != GPU_POLY_FORMAT_COEFF)
        return set_error("prepared CRT recomposition structural contract is invalid");
    GpuMatrixTransformWorkspaceBytes requirements{};
    const int status = gpu_matrix_query_crt_workspace_bytes(
        ctx, target_level, output_rows, output_columns, GPU_MATRIX_CRT_RECOMPOSE,
        level_count, level_count, false, &requirements);
    if (status != 0) return status;
    const int plan_status = plan_matrix_completion_stage(ctx, output_rows, output_columns, target_level,
        target_format, GPU_PREPARED_STAGE_RECONSTRUCTION, owner_layout, out,
        requirements.workspace_bytes, requirements.alignment);
    if (plan_status != 0) return plan_status;
    return plan_matrix_launch(out, ctx, output_rows, output_columns,
        static_cast<size_t>(target_level) + 1, 1, 128, nullptr);
}
