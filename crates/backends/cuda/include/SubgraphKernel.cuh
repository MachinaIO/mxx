#pragma once

// The native interface of subgraph kernels: a crate that registers a
// `GpuSubgraphKernel` implements its entry point against this header only.
// The GPU runtime calls the entry once while it builds a CUDA graph; the
// entry adds its kernels with `mxx_gpu_launch_kernel` (or the cooperative
// variant) on `launch->stream`, declaring every resident address a kernel
// argument holds as a patch of that operand's binding so graph replays
// rebind it.

#include <stddef.h>
#include <stdint.h>

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuContext GpuContext;

// A borrowed, plan-validated physical matrix rectangle. Each limb address
// points at coefficient zero of the first row/column in this view.
struct MxxRawMatrixLimb
{
    uint64_t address;
    uint64_t row_stride_bytes;
    uint64_t column_stride_bytes;
    uint64_t coefficient_stride_bytes;
    uint32_t word_bytes;
    uint32_t crt_limb_index;
    uint64_t modulus;
};
struct MxxRawMatrixView
{
    int32_t physical_device;
    uint32_t degree;
    uint64_t row_origin;
    uint64_t column_origin;
    uint64_t rows;
    uint64_t columns;
    const MxxRawMatrixLimb *limbs;
    size_t limb_count;
};

// Explicit graph node patches bind addresses at replay.
enum MxxGraphPatchTarget
{
    MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD = 0,
    MXX_GRAPH_PATCH_MEMCPY_1D_SRC = 1,
    MXX_GRAPH_PATCH_MEMCPY_1D_DST = 2,
    MXX_GRAPH_PATCH_MEMSET_1D_DST = 3,
    MXX_GRAPH_PATCH_INTEGER_ENCODING = 4,
};

struct MxxGraphPatch
{
    void *node;
    uint32_t target;
    uint32_t argument_index;
    uint32_t byte_offset;
    uint32_t byte_count;
    uint32_t binding_index;
    uint64_t address_addend;
};

// Add one kernel node to the graph being built on `stream` (or launch it
// when no graph is being built). `cooperative` launches it cooperatively,
// so its blocks may synchronize grid-wide; its grid must fit on the device.
int mxx_gpu_graph_dispatch_kernel(GpuContext *ctx, void *stream,
    const void *function, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
    uint32_t block_x, uint32_t block_y, uint32_t block_z, size_t shared_bytes,
    void **arguments, const size_t *argument_sizes, size_t argument_count,
    const MxxGraphPatch *patches, size_t patch_count, int cooperative);

// The negacyclic NTT tables of one CRT limb, as device pointers: psi^k and
// psi^-k for k < degree with their 64-bit Shoup constants, and 1/degree.
// The forward transform twists by psi^i and leaves slot s holding the
// evaluation at psi^(2 bitrev(s) + 1).
struct MxxNttTables
{
    const uint64_t *forward;
    const uint64_t *forward_shoup;
    const uint64_t *inverse;
    const uint64_t *inverse_shoup;
    const uint64_t *degree_inverse;
    const uint64_t *degree_inverse_shoup;
};

enum MxxSubgraphOperandKind
{
    // An evaluation-domain matrix; `matrix` holds its view, and the address
    // of limb t is patched by binding `binding + t`.
    MXX_SUBGRAPH_MATRIX = 0,
    // A family of evaluation-domain matrices: `family_table` is a device
    // table of `family_count * matrix.limb_count` limb descriptors, member
    // major, refreshed before every launch (not patched). `matrix` holds
    // the member shape, without addresses.
    MXX_SUBGRAPH_MATRIX_FAMILY = 1,
    // One integer, or a family of `integer_count` integers, at `integers`
    // in `integer_encoding` (0 signed i64, 1 canonical u64, 2 + w a sign
    // word then w little-endian magnitude words); patched by `binding`.
    MXX_SUBGRAPH_INTEGER = 2,
    MXX_SUBGRAPH_INTEGER_FAMILY = 3,
};

struct MxxSubgraphOperand
{
    uint32_t kind;
    uint32_t binding;
    MxxRawMatrixView matrix;
    const MxxRawMatrixLimb *family_table;
    uint64_t family_count;
    const void *integers;
    int32_t integer_encoding;
    uint32_t reserved;
    uint64_t integer_count;
};

// One subgraph call: its inputs, in call-argument order (captures last),
// then its outputs, all on one device and CRT basis. `scratch` is a device
// buffer of the registered size for the call alone, and `status` a word the
// kernels may set to a nonzero error code; both are patched by their
// bindings. `parameters` are the registered constants.
struct MxxSubgraphLaunch
{
    GpuContext *context;
    void *stream;
    int32_t physical_device;
    uint32_t degree;
    uint32_t limb_count;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t reserved;
    const MxxNttTables *ntt;
    const MxxSubgraphOperand *operands;
    void *scratch;
    uint64_t scratch_bytes;
    uint32_t scratch_binding;
    uint32_t status_binding;
    uint32_t *status;
    const uint64_t *parameters;
    uint64_t parameter_count;
};

// A subgraph kernel entry point returns 0, or a nonzero status after
// recording a message with `gpu_set_last_error`.
typedef int (*MxxSubgraphKernelEntry)(const MxxSubgraphLaunch *launch);
int gpu_set_last_error(const char *msg);

#ifdef __cplusplus
}

template <typename Kernel, typename... Args>
int mxx_gpu_launch_kernel(GpuContext *ctx, cudaStream_t stream, Kernel kernel,
    dim3 grid, dim3 block, size_t shared_bytes,
    const MxxGraphPatch *patches, size_t patch_count, Args... values)
{
    void *arguments[] = {static_cast<void *>(&values)...};
    const size_t sizes[] = {sizeof(Args)...};
    return mxx_gpu_graph_dispatch_kernel(ctx, reinterpret_cast<void *>(stream),
        reinterpret_cast<const void *>(kernel), grid.x, grid.y, grid.z,
        block.x, block.y, block.z, shared_bytes, arguments, sizes,
        sizeof...(Args), patches, patch_count, 0);
}

// As `mxx_gpu_launch_kernel`, launched cooperatively.
template <typename Kernel, typename... Args>
int mxx_gpu_launch_cooperative_kernel(GpuContext *ctx, cudaStream_t stream, Kernel kernel,
    dim3 grid, dim3 block, size_t shared_bytes,
    const MxxGraphPatch *patches, size_t patch_count, Args... values)
{
    void *arguments[] = {static_cast<void *>(&values)...};
    const size_t sizes[] = {sizeof(Args)...};
    return mxx_gpu_graph_dispatch_kernel(ctx, reinterpret_cast<void *>(stream),
        reinterpret_cast<const void *>(kernel), grid.x, grid.y, grid.z,
        block.x, block.y, block.z, shared_bytes, arguments, sizes,
        sizeof...(Args), patches, patch_count, 1);
}
#endif
