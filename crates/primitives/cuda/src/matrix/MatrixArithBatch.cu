#include <climits>
#include <unordered_map>
#include <unordered_set>

namespace
{
    struct MatrixBatchGeometry
    {
        size_t owner_columns[3];
        size_t start[3];
    };

    __device__ __forceinline__ size_t matrix_batch_polynomial(
        size_t polynomial, size_t matrix, size_t columns,
        const MatrixBatchGeometry *geometry, size_t operand)
    {
        if (!geometry) return polynomial;
        const auto &view = geometry[matrix];
        return (polynomial / columns) * view.owner_columns[operand] +
               view.start[operand] + polynomial % columns;
    }

    __global__ void matrix_binary_batch_kernel(
        const uint8_t *const *left,
        const uint8_t *const *right,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
        const uint64_t *moduli,
        size_t limb_count,
        size_t coefficients_per_limb,
        size_t total_coefficients,
        size_t n,
        int operation,
        const MatrixBatchGeometry *geometry,
        size_t output_columns)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total_coefficients)
        {
            return;
        }
        const size_t matrix_limb = idx / coefficients_per_limb;
        const size_t local = idx % coefficients_per_limb;
        const size_t limb = matrix_limb % limb_count;
        const size_t poly_idx = local / n;
        const size_t coefficient_idx = local % n;
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
        const size_t matrix = matrix_limb / limb_count;
        const size_t left_poly = matrix_batch_polynomial(poly_idx, matrix, output_columns, geometry, 0);
        const size_t right_poly = matrix_batch_polynomial(poly_idx, matrix, output_columns, geometry, 1);
        const size_t output_poly = matrix_batch_polynomial(poly_idx, matrix, output_columns, geometry, 2);
        const uint64_t lhs = matrix_load_limb_u64(
            left[matrix_limb], left_poly, coefficient_idx, stride, bytes);
        const uint64_t rhs = matrix_load_limb_u64(
            right[matrix_limb], right_poly, coefficient_idx, stride, bytes);
        const uint64_t value = operation == 0
                                   ? add_mod_u64(lhs, rhs, moduli[limb])
                                   : sub_mod_u64(lhs, rhs, moduli[limb]);
        matrix_store_limb_u64(
            outputs[matrix_limb], output_poly, coefficient_idx, stride, bytes, value);
    }

    __global__ void matrix_negate_batch_kernel(
        const uint8_t *const *inputs,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
        const uint64_t *moduli,
        size_t limb_count,
        size_t coefficients_per_limb,
        size_t total_coefficients,
        size_t n,
        const MatrixBatchGeometry *geometry,
        size_t output_columns)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total_coefficients)
        {
            return;
        }
        const size_t matrix_limb = idx / coefficients_per_limb;
        const size_t local = idx % coefficients_per_limb;
        const size_t limb = matrix_limb % limb_count;
        const size_t poly_idx = local / n;
        const size_t coefficient_idx = local % n;
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
        const size_t input_poly = matrix_batch_polynomial(
            poly_idx, matrix_limb / limb_count, output_columns, geometry, 0);
        const uint64_t value = matrix_load_limb_u64(
            inputs[matrix_limb], input_poly, coefficient_idx, stride, bytes);
        const uint64_t negated = value == 0 ? 0 : moduli[limb] - value;
        const size_t output_poly = matrix_batch_polynomial(
            poly_idx, matrix_limb / limb_count, output_columns, geometry, 2);
        matrix_store_limb_u64(
            outputs[matrix_limb], output_poly, coefficient_idx, stride, bytes, negated);
    }

    __global__ void matrix_ring_automorphism_batch_kernel(
        const uint8_t *const *inputs,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
        const uint64_t *moduli,
        const size_t *indices,
        size_t limb_count,
        size_t coefficients_per_limb,
        size_t total_coefficients,
        size_t n,
        const MatrixBatchGeometry *geometry,
        size_t output_columns,
        bool evaluation,
        unsigned log_n)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total_coefficients) return;
        const size_t matrix_limb = idx / coefficients_per_limb;
        const size_t matrix_idx = matrix_limb / limb_count;
        const size_t limb = matrix_limb % limb_count;
        const size_t local = idx % coefficients_per_limb;
        const size_t polynomial = local / n;
        const size_t position = local % n;
        // MatrixNTT.cu stores DIF evaluation values in bit-reversed order.
        // g(psi^(2k+1)) = f(psi^(a*(2k+1))) for g(X) = f(X^a).
        // Gather that permutation in evaluation format; coefficient format
        // retains the original signed scatter modulo X^n + 1.
        if (evaluation)
        {
            const size_t k = log_n == 0 ? 0 : __brev(static_cast<unsigned>(position)) >> (32 - log_n);
            const size_t exponent = ((2 * k + 1) * indices[matrix_idx]) % (2 * n);
            const unsigned natural = static_cast<unsigned>((exponent - 1) / 2);
            const size_t source = log_n == 0 ? 0 : __brev(natural) >> (32 - log_n);
            const size_t input_poly = matrix_batch_polynomial(
                polynomial, matrix_idx, output_columns, geometry, 0);
            const size_t output_poly = matrix_batch_polynomial(
                polynomial, matrix_idx, output_columns, geometry, 2);
            const uint64_t value = matrix_load_limb_u64(
                inputs[matrix_limb], input_poly, source, strides[limb], coefficient_bytes[limb]);
            matrix_store_limb_u64(
                outputs[matrix_limb], output_poly, position, strides[limb], coefficient_bytes[limb], value);
            return;
        }
        const size_t source = position;
        const size_t exponent = (source * indices[matrix_idx]) % (2 * n);
        const size_t target = exponent < n ? exponent : exponent - n;
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
        const uint64_t modulus = moduli[limb];
        const size_t input_poly = matrix_batch_polynomial(
            polynomial, matrix_idx, output_columns, geometry, 0);
        uint64_t value = matrix_load_limb_u64(inputs[matrix_limb], input_poly, source, stride, bytes);
        if (exponent >= n && value != 0) value = modulus - value;
        const size_t output_poly = matrix_batch_polynomial(
            polynomial, matrix_idx, output_columns, geometry, 2);
        matrix_store_limb_u64(outputs[matrix_limb], output_poly, target, stride, bytes, value);
    }

    __global__ void matrix_scalar_mul_batch_kernel(
        const uint8_t *const *matrices,
        const uint8_t *const *scalars,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
        const uint64_t *moduli,
        size_t limb_count,
        size_t coefficients_per_limb,
        size_t total_coefficients,
        size_t n,
        const MatrixBatchGeometry *geometry,
        size_t output_columns,
        const uint64_t *integer_residues)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total_coefficients) return;
        const size_t matrix_limb = idx / coefficients_per_limb;
        const size_t local = idx % coefficients_per_limb;
        const size_t limb = matrix_limb % limb_count;
        const size_t poly_idx = local / n;
        const size_t coefficient_idx = local % n;
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
        const size_t input_poly = matrix_batch_polynomial(
            poly_idx, matrix_limb / limb_count, output_columns, geometry, 0);
        const uint64_t value = matrix_load_limb_u64(
            matrices[matrix_limb], input_poly, coefficient_idx, stride, bytes);
        const uint64_t scalar = integer_residues ? integer_residues[matrix_limb] :
            matrix_load_limb_u64(scalars[matrix_limb], 0, coefficient_idx, stride, bytes);
        const size_t output_poly = matrix_batch_polynomial(
            poly_idx, matrix_limb / limb_count, output_columns, geometry, 2);
        matrix_store_limb_u64(
            outputs[matrix_limb],
            output_poly,
            coefficient_idx,
            stride,
            bytes,
            mul_mod_u64(value, scalar, moduli[limb]));
    }

    __global__ void matrix_matmul_batch_kernel(
        const uint8_t *const *left,
        const uint8_t *const *right,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
        const uint64_t *moduli,
        size_t limb_count,
        size_t rows,
        size_t inner,
        size_t columns,
        size_t n,
        size_t coefficient_groups,
        const MatrixBatchGeometry *geometry)
    {
        __shared__ uint64_t left_tile[kMatmulTileM][kMatmulTileK];
        __shared__ uint64_t right_tile[kMatmulTileK][kMatmulTileN];
        const size_t matrix_limb = static_cast<size_t>(blockIdx.z) / coefficient_groups;
        const size_t first_coefficient = static_cast<size_t>(blockIdx.z) % coefficient_groups;
        const size_t limb = matrix_limb % limb_count;
        const uint8_t *left_base = left[matrix_limb];
        const uint8_t *right_base = right[matrix_limb];
        uint8_t *output_base = outputs[matrix_limb];
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
        const uint64_t modulus = moduli[limb];
        const size_t matrix = matrix_limb / limb_count;
        const size_t left_columns = geometry ? geometry[matrix].owner_columns[0] : inner;
        const size_t right_columns = geometry ? geometry[matrix].owner_columns[1] : columns;
        const size_t output_columns = geometry ? geometry[matrix].owner_columns[2] : columns;
        const size_t left_start = geometry ? geometry[matrix].start[0] : 0;
        const size_t right_start = geometry ? geometry[matrix].start[1] : 0;
        const size_t output_start = geometry ? geometry[matrix].start[2] : 0;
        const size_t row_base = static_cast<size_t>(blockIdx.y) * kMatmulTileM;
        const size_t column_base = static_cast<size_t>(blockIdx.x) * kMatmulTileN;
        const size_t row = row_base + threadIdx.y;
        const size_t column = column_base + threadIdx.x;
        const int thread = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
        const int thread_count = blockDim.x * blockDim.y;
        for (size_t coefficient_idx = first_coefficient;
             coefficient_idx < n;
             coefficient_idx += coefficient_groups)
        {
            uint64_t accumulator = 0;
            for (size_t inner_base = 0; inner_base < inner; inner_base += kMatmulTileK)
            {
                for (int index = thread; index < kMatmulTileM * kMatmulTileK; index += thread_count)
                {
                    const int tile_row = index / kMatmulTileK;
                    const int tile_inner = index % kMatmulTileK;
                    const size_t source_row = row_base + static_cast<size_t>(tile_row);
                    const size_t source_inner = inner_base + static_cast<size_t>(tile_inner);
                    left_tile[tile_row][tile_inner] =
                        source_row < rows && source_inner < inner
                            ? matrix_load_limb_u64(
                                  left_base,
                                  left_start + source_row * left_columns + source_inner,
                                  coefficient_idx,
                                  stride,
                                  bytes)
                            : 0;
                }
                for (int index = thread; index < kMatmulTileK * kMatmulTileN; index += thread_count)
                {
                    const int tile_inner = index / kMatmulTileN;
                    const int tile_column = index % kMatmulTileN;
                    const size_t source_inner = inner_base + static_cast<size_t>(tile_inner);
                    const size_t source_column = column_base + static_cast<size_t>(tile_column);
                    right_tile[tile_inner][tile_column] =
                        source_inner < inner && source_column < columns
                            ? matrix_load_limb_u64(
                                  right_base,
                                  right_start + source_inner * right_columns + source_column,
                                  coefficient_idx,
                                  stride,
                                  bytes)
                            : 0;
                }
                __syncthreads();
                if (row < rows && column < columns)
                {
                    for (int k = 0; k < kMatmulTileK; ++k)
                    {
                        accumulator = add_mod_u64(
                            accumulator,
                            mul_mod_u64(
                                left_tile[threadIdx.y][k],
                                right_tile[k][threadIdx.x],
                                modulus),
                            modulus);
                    }
                }
                __syncthreads();
            }
            if (row < rows && column < columns)
            {
                matrix_store_limb_u64(
                    output_base,
                    output_start + row * output_columns + column,
                    coefficient_idx,
                    stride,
                    bytes,
                    accumulator);
            }
        }
    }

    __global__ void matrix_thin_row_matmul_batch_kernel(
        const uint8_t *const *left,
        const uint8_t *const *right,
        uint8_t *const *outputs,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
        const uint64_t *moduli,
        const uint64_t *reciprocals,
        size_t limb_count,
        size_t inner,
        size_t columns,
        size_t n,
        size_t coefficient_groups,
        bool lazy_reduction,
        const MatrixBatchGeometry *geometry)
    {
        const size_t matrix_limb = static_cast<size_t>(blockIdx.z) / coefficient_groups;
        const size_t first_coefficient = static_cast<size_t>(blockIdx.z) % coefficient_groups;
        const size_t limb = matrix_limb % limb_count;
        const size_t column =
            static_cast<size_t>(blockIdx.x) * kThinMatmulColumnsPerBlock + threadIdx.y;
        if (column >= columns)
        {
            return;
        }

        const uint8_t *left_base = left[matrix_limb];
        const uint8_t *right_base = right[matrix_limb];
        uint8_t *output_base = outputs[matrix_limb];
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
        const uint64_t modulus = moduli[limb];
        const size_t matrix = matrix_limb / limb_count;
        const size_t right_columns = geometry ? geometry[matrix].owner_columns[1] : columns;
        const size_t left_start = geometry ? geometry[matrix].start[0] : 0;
        const size_t right_start = geometry ? geometry[matrix].start[1] : 0;
        const size_t output_start = geometry ? geometry[matrix].start[2] : 0;
        const uint64_t reciprocal = reciprocals[limb];
        for (size_t coefficient_idx = first_coefficient * kThinMatmulWarpSize + threadIdx.x;
             coefficient_idx < n;
             coefficient_idx += coefficient_groups * kThinMatmulWarpSize)
        {
            uint64_t accumulator = 0;
            for (size_t k = 0; k < inner; ++k)
            {
                const uint64_t lhs = matrix_load_limb_u64(
                    left_base,
                    left_start + k,
                    coefficient_idx,
                    stride,
                    bytes);
                const uint64_t rhs = matrix_load_limb_u64(
                    right_base,
                    right_start + k * right_columns + column,
                    coefficient_idx,
                    stride,
                    bytes);
                if (lazy_reduction)
                {
                    accumulator += lhs * rhs;
                }
                else
                {
                    accumulator = add_mod_u64(
                        accumulator,
                        mul_mod_barrett_u32(lhs, rhs, modulus, reciprocal),
                        modulus);
                }
            }
            accumulator =
                lazy_reduction ? reduce_barrett_u32(accumulator, modulus, reciprocal) : accumulator;
            matrix_store_limb_u64(
                output_base,
                output_start + column,
                coefficient_idx,
                stride,
                bytes,
                accumulator);
        }
    }

    __global__ void matrix_thin_row_mul_accumulate_batch_kernel(
        const uint8_t *const *left,
        const uint8_t *const *right,
        const uint8_t *const *coefficients,
        const uint8_t *const *biases,
        uint8_t *const *outputs,
        const size_t *inner_dimensions,
        const size_t *strides,
        const uint8_t *coefficient_bytes,
        const uint64_t *moduli,
        const uint64_t *reciprocals,
        size_t limb_count,
        size_t product_count,
        size_t columns,
        size_t n,
        size_t coefficient_groups,
        bool lazy_reduction)
    {
        const size_t matrix_limb = static_cast<size_t>(blockIdx.z) / coefficient_groups;
        const size_t first_coefficient = static_cast<size_t>(blockIdx.z) % coefficient_groups;
        const size_t matrix = matrix_limb / limb_count;
        const size_t limb = matrix_limb % limb_count;
        const size_t column =
            static_cast<size_t>(blockIdx.x) * kThinMatmulColumnsPerBlock + threadIdx.y;
        if (column >= columns) return;
        const size_t stride = strides[limb];
        const uint8_t bytes = coefficient_bytes[limb];
        const uint64_t modulus = moduli[limb];
        const uint64_t reciprocal = reciprocals[limb];
        for (size_t coefficient_idx = first_coefficient * kThinMatmulWarpSize + threadIdx.x;
             coefficient_idx < n;
             coefficient_idx += coefficient_groups * kThinMatmulWarpSize)
        {
            uint64_t total = 0;
            for (size_t product = 0; product < product_count; ++product)
            {
                const size_t product_index = matrix * product_count + product;
                const size_t pointer_index = product_index * limb_count + limb;
                const size_t inner = inner_dimensions[product_index];
                uint64_t dot = 0;
                for (size_t k = 0; k < inner; ++k)
                {
                    const uint64_t lhs = matrix_load_limb_u64(
                        left[pointer_index], k, coefficient_idx, stride, bytes);
                    const uint64_t rhs = matrix_load_limb_u64(
                        right[pointer_index], k * columns + column, coefficient_idx, stride, bytes);
                    if (lazy_reduction)
                    {
                        dot += lhs * rhs;
                    }
                    else
                    {
                        dot = add_mod_u64(
                            dot, mul_mod_barrett_u32(lhs, rhs, modulus, reciprocal), modulus);
                    }
                }
                if (lazy_reduction) dot = reduce_barrett_u32(dot, modulus, reciprocal);
                if (coefficients[pointer_index])
                {
                    const uint64_t scalar = matrix_load_limb_u64(
                        coefficients[pointer_index], 0, coefficient_idx, stride, bytes);
                    dot = mul_mod_u64(dot, scalar, modulus);
                }
                total = add_mod_u64(total, dot, modulus);
            }
            const uint8_t *bias = biases[matrix_limb];
            if (bias)
            {
                total = add_mod_u64(
                    total,
                    matrix_load_limb_u64(bias, column, coefficient_idx, stride, bytes),
                    modulus);
            }
            matrix_store_limb_u64(
                outputs[matrix_limb], column, coefficient_idx, stride, bytes, total);
        }
    }

    // Fixed launch metadata keeps arbitrarily long sums bounded without
    // allocating coefficient matrices, product matrices or device pointer lists.
    constexpr size_t kAccumulateTerms = 4;
    struct AccumulateTerm {
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *left, *right;
        size_t left_offset, right_offset, left_pitch, right_pitch, inner;
        unsigned scalar;
        uint64_t residues[GPU_RUNTIME_MAX_LIMBS];
    };
    struct AccumulateRangeMetadata {
        AccumulateTerm terms[kAccumulateTerms];
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *output, *bias;
        size_t output_offset, bias_offset, output_pitch, bias_pitch;
        unsigned indices[GPU_RUNTIME_MAX_LIMBS];
        uint64_t moduli[GPU_RUNTIME_MAX_LIMBS], reciprocals[GPU_RUNTIME_MAX_LIMBS];
        size_t term_count;
        bool accumulate;
    };
    static_assert(sizeof(AccumulateRangeMetadata) + 3 * sizeof(size_t) <= 4096,
                  "accumulate arguments must fit the portable CUDA parameter limit");

    __global__ void matrix_accumulate_range_kernel(
        const AccumulateRangeMetadata metadata, size_t columns, size_t count, size_t n)
    {
        const size_t limb = blockIdx.z;
        const size_t index = metadata.indices[limb];
        const auto output = metadata.output[index];
        const uint64_t modulus = metadata.moduli[limb];
        const uint64_t reciprocal = metadata.reciprocals[limb];
        for (size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             local < count; local += static_cast<size_t>(gridDim.x) * blockDim.x) {
            const size_t coefficient = local % n;
            const size_t row = local / n / columns, column = local / n % columns;
            const size_t out_poly = metadata.output_offset + row * metadata.output_pitch + column;
            uint64_t total = metadata.accumulate
                ? matrix_load_limb_u64(output.base, out_poly, coefficient, output.stride, output.width) : 0;
            if (metadata.bias) {
                const auto bias = metadata.bias[index];
                total = add_mod_u64(total, matrix_load_limb_u64(bias.base,
                    metadata.bias_offset + row * metadata.bias_pitch + column,
                    coefficient, bias.stride, bias.width), modulus);
            }
            for (size_t term = 0; term < metadata.term_count; ++term) {
                const auto &product = metadata.terms[term];
                if (product.inner == 0 || product.residues[limb] == 0) continue;
                const auto left = product.left[index], right = product.right[index];
                uint64_t dot = 0;
                const bool narrow = modulus <= UINT32_MAX;
                const bool lazy = narrow && product.inner <= UINT64_MAX / ((modulus - 1) * (modulus - 1));
                for (size_t k = 0; k < product.inner; ++k) {
                    const size_t lp = product.left_offset + (product.scalar == 1 ? 0 :
                        row * product.left_pitch + (product.scalar == 2 ? column : k));
                    const size_t rp = product.right_offset + (product.scalar == 2 ? 0 :
                        (product.scalar == 1 ? row : k) * product.right_pitch + column);
                    const uint64_t lhs = matrix_load_limb_u64(left.base, lp, coefficient, left.stride, left.width);
                    const uint64_t rhs = matrix_load_limb_u64(right.base, rp, coefficient, right.stride, right.width);
                    if (lazy) dot += lhs * rhs;
                    else dot = add_mod_u64(dot, narrow ? mul_mod_barrett_u32(lhs, rhs, modulus, reciprocal)
                                                     : mul_mod_u64(lhs, rhs, modulus), modulus);
                }
                if (lazy) dot = reduce_barrett_u32(dot, modulus, reciprocal);
                total = add_mod_u64(total, mul_mod_u64(dot, product.residues[limb], modulus), modulus);
            }
            matrix_store_limb_u64(output.base, out_poly, coefficient, output.stride, output.width, total);
        }
    }

    struct MatrixBatchMetadata
    {
        GpuContext *context = nullptr;
        size_t matrix_count = 0;
        size_t limb_count = 0;
        size_t rows = 0;
        size_t inner = 0;
        size_t columns = 0;
        size_t n = 0;
        int level = -1;
        int device = -1;
        cudaStream_t stream = nullptr;
        std::vector<dim3> limb_ids;
        std::vector<size_t> strides;
        std::vector<uint8_t> coefficient_bytes;
        std::vector<uint64_t> moduli;
        std::vector<const uint8_t *> left;
        std::vector<const uint8_t *> right;
        std::vector<uint8_t *> outputs;
        std::vector<MatrixBatchGeometry> views;
    };


    struct MatrixBatchAllocationPlan
    {
        size_t offsets[10]{};
        size_t count = 0;
        size_t bytes = 0;
    };

    int plan_matrix_batch_allocation(
        size_t matrices, size_t limbs, size_t products,
        GpuMatrixBatchOperation operation, bool matrix_views,
        MatrixBatchAllocationPlan *out)
    {
        if (!out || matrices == 0 || limbs == 0 || products == 0 ||
            (operation != GPU_MATRIX_BATCH_ACCUMULATE && products != 1) ||
            matrices > SIZE_MAX / limbs || matrices > SIZE_MAX / products ||
            matrices * products > SIZE_MAX / limbs)
            return set_error("invalid or overflowing matrix batch workspace shape");
        if (matrix_views && operation != GPU_MATRIX_BATCH_BINARY && operation != GPU_MATRIX_BATCH_NEGATE &&
            operation != GPU_MATRIX_BATCH_AUTOMORPHISM && operation != GPU_MATRIX_BATCH_SCALAR &&
            operation != GPU_MATRIX_BATCH_MULTIPLY)
            return set_error("matrix batch operation does not support rectangular views");
        const size_t pointers = matrices * limbs;
        const size_t product_pointers = matrices * products * limbs;
        MatrixBatchAllocationPlan plan{};
        auto add = [&](size_t count, size_t element_bytes) -> bool {
            constexpr size_t alignment = alignof(void *);
            const size_t padding = (alignment - plan.bytes % alignment) % alignment;
            if (plan.count == 10 || count > SIZE_MAX / element_bytes ||
                padding > SIZE_MAX - plan.bytes) return false;
            const size_t offset = plan.bytes + padding;
            const size_t bytes = count * element_bytes;
            if (bytes > SIZE_MAX - offset) return false;
            plan.offsets[plan.count++] = offset;
            plan.bytes = offset + bytes;
            return true;
        };
        bool valid = false;
        switch (operation)
        {
        case GPU_MATRIX_BATCH_BINARY:
        case GPU_MATRIX_BATCH_SCALAR:
        case GPU_MATRIX_BATCH_MULTIPLY:
            valid = add(pointers, sizeof(void *)) && add(pointers, sizeof(void *)) &&
                    add(pointers, sizeof(void *)) && add(limbs, sizeof(size_t)) &&
                    add(limbs, sizeof(uint8_t)) && add(limbs, sizeof(uint64_t));
            if (operation == GPU_MATRIX_BATCH_MULTIPLY)
                valid = valid && add(limbs, sizeof(uint64_t));
            break;
        case GPU_MATRIX_BATCH_NEGATE:
        case GPU_MATRIX_BATCH_AUTOMORPHISM:
            valid = add(pointers, sizeof(void *)) && add(pointers, sizeof(void *)) &&
                    add(limbs, sizeof(size_t)) && add(limbs, sizeof(uint8_t)) &&
                    add(limbs, sizeof(uint64_t));
            if (operation == GPU_MATRIX_BATCH_AUTOMORPHISM)
                valid = valid && add(matrices, sizeof(size_t));
            break;
        case GPU_MATRIX_BATCH_ACCUMULATE:
            valid = add(product_pointers, sizeof(void *)) &&
                    add(product_pointers, sizeof(void *)) &&
                    add(product_pointers, sizeof(void *)) &&
                    add(pointers, sizeof(void *)) && add(pointers, sizeof(void *)) &&
                    add(matrices * products, sizeof(size_t)) && add(limbs, sizeof(size_t)) &&
                    add(limbs, sizeof(uint8_t)) && add(limbs, sizeof(uint64_t)) &&
                    add(limbs, sizeof(uint64_t));
            break;
        default:
            return set_error("unsupported matrix batch workspace operation");
        }
        if (matrix_views)
            valid = valid && add(matrices, sizeof(MatrixBatchGeometry));
        if (!valid) return set_error("matrix batch workspace byte size overflow");
        *out = plan;
        return 0;
    }

    // One bounded metadata arena per batch. An exclusively owned result already
    // has auxiliary storage; borrow its whole partition while this all-limb
    // operation owns it. Large batches use one separate, explicitly sized arena.
    struct MatrixBatchWorkspace
    {
        MatrixBatchWorkspace() = default;
        MatrixBatchWorkspace(const MatrixBatchWorkspace &) = delete;
        MatrixBatchWorkspace &operator=(const MatrixBatchWorkspace &) = delete;
        MatrixBatchAllocationPlan plan{};
        GpuMatrix *owner = nullptr;
        cudaStream_t stream = nullptr;
        uint8_t *base = nullptr;
        bool separate = false;
        bool completed = false;
        GpuDeviceWorkspace device_workspace;

        int acquire(GpuMatrix *output, const MatrixBatchMetadata &metadata,
                    GpuMatrixBatchOperation operation, size_t products = 1)
        {
            int status = plan_matrix_batch_allocation(
                metadata.matrix_count, metadata.limb_count, products, operation,
                !metadata.views.empty(), &plan);
            if (status != 0) return status;
            const size_t partition = metadata.limb_ids[0].x;
            if (partition >= output->shared_aux_buffers.size())
                return set_error("missing matrix batch output workspace");
            const auto &aux = output->shared_aux_buffers[partition];
            if (aux.slots_total > SIZE_MAX / sizeof(void *))
                return set_error("matrix batch output workspace overflow");
            owner = output;
            stream = metadata.stream;
            // prepare_matrix_batch has already joined every output's
            // allocation and initialization on this stream.
            if (aux.ptr && plan.bytes <= aux.slots_total * sizeof(void *))
                base = reinterpret_cast<uint8_t *>(aux.ptr);
            else
            {
                separate = true;
                const int allocated = device_workspace.acquire(
                    output->ctx, metadata.device, GPU_PREPARED_BATCH_WORKSPACE,
                    plan.bytes, alignof(void *), stream);
                if (allocated != 0) return allocated;
                base = device_workspace.data;
            }
            return 0;
        }

        void *region(size_t index) const { return base + plan.offsets[index]; }

        int complete()
        {
            if (separate && base)
            {
                const int released = device_workspace.release();
                base = nullptr;
                if (released != 0) return released;
                // The output's final release receipt must dominate the arena
                // free as well as the arithmetic kernel which consumed it.
                const int status = matrix_record_all_limb_writes(owner, stream, true);
                if (status != 0) return status;
            }
            completed = true;
            return 0;
        }

        ~MatrixBatchWorkspace()
        {
            if (completed || !stream) return;
            // Only an error path reaches here. Queue the arena free behind its
            // uploads/kernels, then make every producer/release stream observe
            // that completion. No normal completion installation is assumed.
            if (separate && base) device_workspace.release();
            if (gpu_matrix_retire_submitted_work(owner) != 0)
                gpu_execution_mark_allocation_unknown(owner->ctx->execution.get());
        }
    };

    int prepare_matrix_batch(
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right,
        size_t matrix_count,
        int operation,
        MatrixBatchMetadata *metadata,
        const GpuMatrixBatchView *views = nullptr)
    {
        if (!outputs || !left || matrix_count == 0 || !metadata)
            return set_error("invalid matrix batch arguments");
        const GpuMatrix *first_left = left[0];
        const GpuMatrix *first_right = right ? right[0] : nullptr;
        GpuMatrix *first_output = outputs[0];
        if (!first_left || !first_output || !first_left->ctx ||
            (right && !first_right) || first_left->level < 0 || first_left->ctx->N <= 0)
            return set_error("invalid first matrix in batch");
        const bool multiplication = operation == 1;
        const bool scalar_multiplication = operation == 2;
        if ((multiplication || scalar_multiplication) && !first_right)
            return set_error("missing right matrix batch input");
        const auto valid_range = [](const GpuMatrix *matrix, const GpuMatrixRange &range) {
            return matrix && range.row_start < range.row_end && range.row_end <= matrix->rows &&
                range.column_start < range.column_end && range.column_end <= matrix->cols &&
                matrix->cols != 0 && matrix->rows <= SIZE_MAX / matrix->cols;
        };
        if (views && !valid_range(first_left, views[0].left))
            return set_error("invalid matrix batch source range");
        const size_t input_rows = views ? views[0].left.row_end - views[0].left.row_start : first_left->rows;
        const size_t input_columns = views
            ? views[0].left.column_end - views[0].left.column_start : first_left->cols;
        if (!multiplication && input_columns != 0 && input_rows > SIZE_MAX / input_columns)
            return set_error("matrix batch view shape overflow");
        metadata->context = first_left->ctx;
        metadata->matrix_count = matrix_count;
        metadata->limb_count = static_cast<size_t>(first_left->level) + 1;
        metadata->rows = input_rows;
        if (views && multiplication && !valid_range(first_right, views[0].right))
            return set_error("invalid matrix batch right range");
        metadata->inner = multiplication ? input_columns : input_rows * input_columns;
        metadata->columns = multiplication
            ? (views ? views[0].right.column_end - views[0].right.column_start : first_right->cols)
            : input_columns;
        metadata->n = static_cast<size_t>(first_left->ctx->N);
        metadata->level = first_left->level;
        if ((multiplication || scalar_multiplication) &&
            (first_left->format != GPU_POLY_FORMAT_EVAL || first_right->format != GPU_POLY_FORMAT_EVAL))
            return set_error("matrix multiplication batch requires Eval format");
        if (metadata->limb_count > first_left->ctx->limb_gpu_ids.size() ||
            metadata->limb_count > first_left->ctx->moduli.size() ||
            matrix_count > SIZE_MAX / metadata->limb_count)
            return set_error("invalid matrix batch level");
        if (views)
        {
            const size_t pointers = matrix_count * metadata->limb_count;
            if (!multiplication && (metadata->inner > SIZE_MAX / metadata->n ||
                metadata->inner * metadata->n > (SIZE_MAX - 255) / pointers ||
                (pointers * metadata->inner * metadata->n + 255) / 256 > static_cast<size_t>(INT_MAX)))
                return set_error("matrix batch view exceeds CUDA grid size");
            std::unordered_set<const GpuMatrix *> readers;
            readers.reserve(matrix_count * (right ? 2 : 1));
            for (size_t matrix = 0; matrix < matrix_count; ++matrix)
            {
                readers.insert(left[matrix]);
                if (right) readers.insert(right[matrix]);
            }
            std::unordered_map<const GpuMatrix *, std::vector<GpuMatrixRange>> destinations;
            for (size_t matrix = 0; matrix < matrix_count; ++matrix)
            {
                const auto &range = views[matrix].output;
                if (!valid_range(outputs[matrix], range) || readers.count(outputs[matrix]))
                    return set_error("matrix batch destination is invalid or aliases an input owner");
                auto &prior = destinations[outputs[matrix]];
                for (const auto &other : prior)
                    if (range.row_start < other.row_end && other.row_start < range.row_end &&
                        range.column_start < other.column_end && other.column_start < range.column_end)
                        return set_error("matrix batch destination rectangles overlap");
                prior.push_back(range);
            }
            metadata->views.resize(matrix_count);
        }
        metadata->limb_ids.assign(first_left->ctx->limb_gpu_ids.begin(),
                                  first_left->ctx->limb_gpu_ids.begin() + metadata->limb_count);
        metadata->strides.resize(metadata->limb_count);
        metadata->coefficient_bytes.resize(metadata->limb_count);
        metadata->moduli.assign(first_left->ctx->moduli.begin(),
                                first_left->ctx->moduli.begin() + metadata->limb_count);
        metadata->left.resize(matrix_count * metadata->limb_count);
        if (right) metadata->right.resize(matrix_count * metadata->limb_count);
        metadata->outputs.resize(matrix_count * metadata->limb_count);
        for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
        {
            const GpuMatrix *lhs = left[matrix_idx];
            const GpuMatrix *rhs = right ? right[matrix_idx] : nullptr;
            GpuMatrix *out = outputs[matrix_idx];
            if (!lhs || !out || lhs->ctx != metadata->context || out->ctx != metadata->context ||
                lhs->level != metadata->level || out->level != metadata->level ||
                lhs->format != first_left->format || (right && (!rhs || rhs->ctx != metadata->context ||
                    rhs->level != metadata->level || rhs->format != first_right->format)) ||
                (scalar_multiplication && (rhs->rows != 1 || rhs->cols != 1)))
                return set_error("matrix batch is not homogeneous");
            if (views)
            {
                const auto &view = views[matrix_idx];
                const auto matching_range = [&](const GpuMatrix *matrix, const GpuMatrixRange &range,
                                                size_t rows, size_t columns) {
                    return valid_range(matrix, range) && range.row_end - range.row_start == rows &&
                        range.column_end - range.column_start == columns;
                };
                if (!matching_range(lhs, view.left, input_rows, input_columns) ||
                    !matching_range(out, view.output, metadata->rows, metadata->columns) ||
                    out->format != lhs->format ||
                    (lhs->format != GPU_POLY_FORMAT_COEFF && lhs->format != GPU_POLY_FORMAT_EVAL) ||
                    (rhs && !scalar_multiplication &&
                        (!matching_range(rhs, view.right, multiplication ? input_columns : input_rows,
                            metadata->columns) || rhs->format != lhs->format)))
                    return set_error("incompatible matrix batch rectangular views");
                metadata->views[matrix_idx] = {
                    {lhs->cols, rhs ? rhs->cols : 1, out->cols},
                    {view.left.row_start * lhs->cols + view.left.column_start,
                     rhs && !scalar_multiplication
                         ? view.right.row_start * rhs->cols + view.right.column_start : 0,
                     view.output.row_start * out->cols + view.output.column_start}};
            }
            else if (lhs->rows != metadata->rows ||
                (!multiplication && (lhs->cols != first_left->cols ||
                    out->rows != lhs->rows || out->cols != input_columns ||
                    (rhs && !scalar_multiplication && (rhs->rows != lhs->rows || rhs->cols != lhs->cols)))) ||
                (multiplication && (lhs->cols != rhs->rows || rhs->cols != metadata->columns ||
                    out->rows != metadata->rows || out->cols != metadata->columns)))
                return set_error("matrix batch shapes differ");
            for (size_t limb = 0; limb < metadata->limb_count; ++limb)
            {
                const dim3 limb_id = metadata->limb_ids[limb];
                int lhs_device = -1;
                int out_device = -1;
                cudaStream_t out_stream = nullptr;
                size_t lhs_stride = 0;
                size_t out_stride = 0;
                uint8_t lhs_bytes = 0;
                uint8_t out_bytes = 0;
                const uint8_t *lhs_pointer = matrix_limb_ptr_by_id(lhs, 0, limb_id);
                uint8_t *out_pointer = matrix_limb_ptr_by_id(out, 0, limb_id);
                if (!lhs_pointer || !out_pointer ||
                    matrix_limb_device(lhs, limb_id, &lhs_device) != 0 ||
                    matrix_limb_device(out, limb_id, &out_device) != 0 ||
                    matrix_limb_stream(out, limb_id, &out_stream) != 0 ||
                    !matrix_limb_metadata_by_id(lhs, limb_id, &lhs_stride, &lhs_bytes) ||
                    !matrix_limb_metadata_by_id(out, limb_id, &out_stride, &out_bytes) ||
                    lhs_device != out_device || lhs_stride != out_stride || lhs_bytes != out_bytes)
                {
                    return set_error("invalid matrix batch limb");
                }
                if (metadata->device < 0)
                {
                    metadata->device = out_device;
                    metadata->stream = out_stream;
                }
                else if (metadata->device != out_device)
                {
                    return set_error("matrix batch requires one placement");
                }
                if (matrix_idx == 0)
                {
                    metadata->strides[limb] = lhs_stride;
                    metadata->coefficient_bytes[limb] = lhs_bytes;
                }
                else if (metadata->strides[limb] != lhs_stride ||
                         metadata->coefficient_bytes[limb] != lhs_bytes)
                {
                    return set_error("matrix batch limb layout differs");
                }
                const size_t flat = matrix_idx * metadata->limb_count + limb;
                metadata->left[flat] = lhs_pointer;
                metadata->outputs[flat] = out_pointer;
                if (rhs)
                {
                    int rhs_device = -1;
                    size_t rhs_stride = 0;
                    uint8_t rhs_bytes = 0;
                    const uint8_t *rhs_pointer = matrix_limb_ptr_by_id(rhs, 0, limb_id);
                    if (!rhs_pointer || matrix_limb_device(rhs, limb_id, &rhs_device) != 0 ||
                        !matrix_limb_metadata_by_id(rhs, limb_id, &rhs_stride, &rhs_bytes) ||
                        rhs_device != metadata->device || rhs_stride != lhs_stride || rhs_bytes != lhs_bytes)
                    {
                        return set_error("invalid right matrix batch limb");
                    }
                    metadata->right[flat] = rhs_pointer;
                }
            }
        }
        cudaError_t error = cudaSetDevice(metadata->device);
        if (error != cudaSuccess || !metadata->stream)
        {
            return error != cudaSuccess ? set_error(error) : set_error("null matrix batch stream");
        }
        for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
        {
            // Fresh outputs may allocate and initialize their descriptors on
            // different producer streams. Join every owner before this batch
            // writes its allocation on the first output's stream.
            const int output_status = matrix_wait_all_limb_streams(
                outputs[matrix_idx], metadata->device, metadata->stream, true);
            if (output_status != 0) return output_status;
            for (size_t limb = 0; limb < metadata->limb_count; ++limb)
            {
                const dim3 limb_id = metadata->limb_ids[limb];
                int status = matrix_wait_limb_stream(
                    left[matrix_idx], limb_id, metadata->device, metadata->stream,
                    false, views != nullptr);
                if (status == 0 && right)
                {
                    status = matrix_wait_limb_stream(
                        right[matrix_idx], limb_id, metadata->device, metadata->stream,
                        false, views != nullptr);
                }
                if (status != 0)
                {
                    return status;
                }
            }
        }
        return 0;
    }

    int finish_matrix_batch(
        const MatrixBatchMetadata &metadata,
        GpuMatrix *const *outputs,
        const GpuMatrix *const *left,
        const GpuMatrix *const *right)
    {
        // One batch stream has completed every output write. Its existing
        // output event also covers every input reader, without per-input lazy
        // event allocation. Capture this record before publishing reader waits.
        for (size_t matrix = 0; matrix < metadata.matrix_count; ++matrix) {
            const int status = matrix_record_all_limb_writes(outputs[matrix], metadata.stream, true);
            if (status != 0) return status;
        }
        const dim3 first = metadata.limb_ids[0];
        const cudaEvent_t completion = outputs[0]->exec_limb_states[first.x][first.y].write_done;
        const auto track_input = metadata.views.empty()
            ? matrix_track_limb_consumer : matrix_track_limb_consumer_readonly;
        for (size_t matrix_idx = 0; matrix_idx < metadata.matrix_count; ++matrix_idx)
        {
            for (size_t limb = 0; limb < metadata.limb_count; ++limb)
            {
                const dim3 limb_id = metadata.limb_ids[limb];
                int status = track_input(
                    left[matrix_idx], limb_id, metadata.device, metadata.stream, completion, true);
                if (status == 0 && right)
                {
                    status = track_input(
                        right[matrix_idx], limb_id, metadata.device, metadata.stream, completion, true);
                }
                if (status != 0) return status;
            }
        }
        return 0;
    }

}


extern "C" int gpu_matrix_retire_submitted_work(const GpuMatrix *output)
{
    if (!output || !output->ctx || !output->ctx->execution)
        return set_error("invalid interrupted matrix batch owner");
    GpuAllocationActivity activity(output->ctx->execution.get(), -1);
    // Batch kernels run on the first output's first active producer stream.
    // Retire all its producer streams as well, covering an interrupted setup or
    // format conversion without depending on partially updated write events.
    for (const auto &partition : output->exec_limb_states)
    {
        cudaStream_t seen[2 * GPU_RUNTIME_MAX_LIMBS]{};
        size_t count = 0;
        for (const auto &state : partition)
        {
            for (cudaStream_t stream : {state.stream, state.last_write_stream})
            {
                if (!stream || std::find(seen, seen + count, stream) != seen + count)
                    continue;
                if (count == 2 * GPU_RUNTIME_MAX_LIMBS)
                {
                    output->ctx->execution->memory_release_failed.store(true, std::memory_order_release);
                    output->ctx->execution->unretired_work.store(true, std::memory_order_release);
                    gpu_execution_mark_allocation_unknown(output->ctx->execution.get());
                    return set_error("too many interrupted batch producer streams");
                }
                seen[count++] = stream;
                const int status = gpu_context_retire_stream(output->ctx, state.device, stream);
                if (status != 0) return status;
            }
        }
    }
    return 0;
}

extern "C" int gpu_matrix_query_batch_workspace_bytes(
    const GpuContext *ctx, int level, size_t output_rows, size_t output_cols,
    size_t matrix_count, size_t product_count, GpuMatrixBatchOperation operation,
    int matrix_views, GpuMatrixBatchWorkspaceBytes *out)
{
    if (!ctx || !out || level < 0 || output_rows == 0 || output_cols == 0 ||
        ctx->gpu_ids.size() != 1 || (matrix_views != 0 && matrix_views != 1))
        return set_error("invalid matrix batch workspace query");
    GpuMatrixAllocationBytes allocation{};
    int status = gpu_matrix_query_allocation_bytes(
        ctx, level, output_rows, output_cols, GPU_POLY_FORMAT_EVAL, &allocation);
    if (status != 0) return status;
    MatrixBatchAllocationPlan plan{};
    status = plan_matrix_batch_allocation(
        matrix_count, static_cast<size_t>(level) + 1, product_count, operation,
        matrix_views != 0, &plan);
    if (status != 0) return status;
    *out = GpuMatrixBatchWorkspaceBytes{
        plan.bytes, plan.bytes <= allocation.aux_workspace_bytes ? 0 : plan.bytes,
        alignof(void *)};
    return 0;
}

extern "C" int gpu_matrix_binary_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *left,
    const GpuMatrix *const *right,
    const GpuMatrixBatchView *views,
    size_t matrix_count,
    int operation)
{
    if (!outputs || matrix_count == 0 || !outputs[0] || !outputs[0]->ctx ||
        !outputs[0]->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_binary_batch");
    GpuAllocationActivity activity(outputs[0]->ctx->execution.get(), -1);
    if (!right || (operation != 0 && operation != 1))
    {
        return set_error("invalid matrix binary batch operation");
    }
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(outputs, left, right, matrix_count, 0, &metadata, views);
    if (status != 0) return status;
    const size_t coefficients_per_limb = metadata.inner * metadata.n;
    const size_t total_coefficients = matrix_count * metadata.limb_count * coefficients_per_limb;
    const size_t pointer_count = matrix_count * metadata.limb_count;
    const uint8_t **d_left = nullptr;
    const uint8_t **d_right = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    MatrixBatchWorkspace workspace;
    status = workspace.acquire(outputs[0], metadata, GPU_MATRIX_BATCH_BINARY);
    if (status != 0) return status;
    d_left = reinterpret_cast<decltype(d_left)>(workspace.region(0));
    d_right = reinterpret_cast<decltype(d_right)>(workspace.region(1));
    d_outputs = reinterpret_cast<decltype(d_outputs)>(workspace.region(2));
    d_strides = reinterpret_cast<decltype(d_strides)>(workspace.region(3));
    d_bytes = reinterpret_cast<decltype(d_bytes)>(workspace.region(4));
    d_moduli = reinterpret_cast<decltype(d_moduli)>(workspace.region(5));
    auto *d_views = views ? static_cast<MatrixBatchGeometry *>(workspace.region(6)) : nullptr;
    cudaError_t error = cudaSuccess;
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_left, metadata.left.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_right, metadata.right.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count, cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess && d_views) error = cudaMemcpyAsync(
        d_views, metadata.views.data(), matrix_count * sizeof(MatrixBatchGeometry),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    const int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_binary_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_left, d_right, d_outputs, d_strides, d_bytes, d_moduli, metadata.limb_count,
        coefficients_per_limb, total_coefficients, metadata.n, operation, d_views, metadata.columns);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    status = finish_matrix_batch(metadata, outputs, left, right);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx) outputs[matrix_idx]->format = left[matrix_idx]->format;
    if (status != 0) return status;
    return workspace.complete();
}

extern "C" int gpu_matrix_negate_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    const GpuMatrixBatchView *views,
    size_t matrix_count)
{
    if (!outputs || matrix_count == 0 || !outputs[0] || !outputs[0]->ctx ||
        !outputs[0]->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_negate_batch");
    GpuAllocationActivity activity(outputs[0]->ctx->execution.get(), -1);
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(
        outputs, inputs, nullptr, matrix_count, 0, &metadata, views);
    if (status != 0) return status;
    const size_t coefficients_per_limb = metadata.inner * metadata.n;
    const size_t total_coefficients = matrix_count * metadata.limb_count * coefficients_per_limb;
    const size_t pointer_count = matrix_count * metadata.limb_count;
    const uint8_t **d_inputs = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    MatrixBatchWorkspace workspace;
    status = workspace.acquire(outputs[0], metadata, GPU_MATRIX_BATCH_NEGATE);
    if (status != 0) return status;
    d_inputs = reinterpret_cast<decltype(d_inputs)>(workspace.region(0));
    d_outputs = reinterpret_cast<decltype(d_outputs)>(workspace.region(1));
    d_strides = reinterpret_cast<decltype(d_strides)>(workspace.region(2));
    d_bytes = reinterpret_cast<decltype(d_bytes)>(workspace.region(3));
    d_moduli = reinterpret_cast<decltype(d_moduli)>(workspace.region(4));
    auto *d_columns = views
        ? static_cast<MatrixBatchGeometry *>(workspace.region(5)) : nullptr;
    cudaError_t error = cudaSuccess;
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_inputs, metadata.left.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count, cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess && d_columns) error = cudaMemcpyAsync(
        d_columns, metadata.views.data(), matrix_count * sizeof(MatrixBatchGeometry),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    const int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_negate_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_inputs, d_outputs, d_strides, d_bytes, d_moduli, metadata.limb_count,
        coefficients_per_limb, total_coefficients, metadata.n, d_columns, metadata.columns);
    cudaError_t launch_error = cudaGetLastError();
    if (launch_error != cudaSuccess)
    {
        return set_error(launch_error);
    }
    status = finish_matrix_batch(metadata, outputs, inputs, nullptr);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx) outputs[matrix_idx]->format = inputs[matrix_idx]->format;
    if (status != 0) return status;
    return workspace.complete();
}

extern "C" int gpu_matrix_validate_ring_automorphism(
    size_t ring_dimension,
    const size_t *indices,
    size_t matrix_count)
{
    if (ring_dimension == 0 || (ring_dimension & (ring_dimension - 1)) != 0)
        return set_error("ring automorphism requires a power-of-two ring dimension");
    if (matrix_count != 0 && !indices)
        return set_error("null ring automorphism indices");
    if (ring_dimension > SIZE_MAX / 2)
        return set_error("ring automorphism dimension overflow");
    for (size_t matrix = 0; matrix < matrix_count; ++matrix)
    {
        if (indices[matrix] == 0 || indices[matrix] >= 2 * ring_dimension || indices[matrix] % 2 == 0)
            return set_error("invalid ring automorphism index");
    }
    return 0;
}

extern "C" int gpu_matrix_ring_automorphism_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *inputs,
    const size_t *indices,
    const GpuMatrixBatchView *views,
    size_t matrix_count)
{
    if (!outputs || matrix_count == 0 || !outputs[0] || !outputs[0]->ctx ||
        !outputs[0]->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_ring_automorphism_batch");
    GpuAllocationActivity activity(outputs[0]->ctx->execution.get(), -1);
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(
        outputs, inputs, nullptr, matrix_count, 0, &metadata, views);
    if (status != 0) return status;

    status = gpu_matrix_validate_ring_automorphism(metadata.n, indices, matrix_count);
    if (status != 0) return status;
    const size_t coefficients_per_limb = metadata.inner * metadata.n;
    const size_t total_coefficients = matrix_count * metadata.limb_count * coefficients_per_limb;
    const size_t pointer_count = matrix_count * metadata.limb_count;
    const uint8_t **d_inputs = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    size_t *d_indices = nullptr;
    MatrixBatchWorkspace workspace;
    status = workspace.acquire(outputs[0], metadata, GPU_MATRIX_BATCH_AUTOMORPHISM);
    if (status != 0) return status;
    d_inputs = reinterpret_cast<decltype(d_inputs)>(workspace.region(0));
    d_outputs = reinterpret_cast<decltype(d_outputs)>(workspace.region(1));
    d_strides = reinterpret_cast<decltype(d_strides)>(workspace.region(2));
    d_bytes = reinterpret_cast<decltype(d_bytes)>(workspace.region(3));
    d_moduli = reinterpret_cast<decltype(d_moduli)>(workspace.region(4));
    d_indices = reinterpret_cast<decltype(d_indices)>(workspace.region(5));
    auto *d_columns = views
        ? static_cast<MatrixBatchGeometry *>(workspace.region(6)) : nullptr;
    cudaError_t error = cudaSuccess;
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_inputs, metadata.left.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count, cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_indices, indices, matrix_count * sizeof(size_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess && d_columns) error = cudaMemcpyAsync(
        d_columns, metadata.views.data(), matrix_count * sizeof(MatrixBatchGeometry),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error != cudaSuccess) { return set_error(error); }
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_ring_automorphism_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_inputs, d_outputs, d_strides, d_bytes, d_moduli, d_indices,
        metadata.limb_count, coefficients_per_limb, total_coefficients, metadata.n,
        d_columns, metadata.columns, inputs[0]->format == GPU_POLY_FORMAT_EVAL,
        static_cast<unsigned>(__builtin_ctz(static_cast<unsigned>(metadata.n))));
    error = cudaGetLastError();
    if (error != cudaSuccess) { return set_error(error); }
    status = finish_matrix_batch(metadata, outputs, inputs, nullptr);
    for (size_t matrix = 0; matrix < matrix_count; ++matrix) outputs[matrix]->format = inputs[matrix]->format;
    if (status != 0) return status;
    return workspace.complete();
}

extern "C" int gpu_matrix_mul_scalar_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *matrices,
    const GpuMatrix *const *scalars,
    const GpuMatrixBatchView *views,
    size_t matrix_count,
    const uint64_t *integer_residues)
{
    if (!outputs || matrix_count == 0 || !outputs[0] || !outputs[0]->ctx ||
        !outputs[0]->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_mul_scalar_batch");
    GpuAllocationActivity activity(outputs[0]->ctx->execution.get(), -1);
    if ((scalars == nullptr) == (integer_residues == nullptr))
        return set_error("scalar batch requires exactly one scalar representation");
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(
        outputs, matrices, scalars, matrix_count, integer_residues ? 0 : 2, &metadata, views);
    if (status != 0) return status;
    if (integer_residues)
        for (size_t matrix = 0; matrix < matrix_count; ++matrix)
            for (size_t limb = 0; limb < metadata.limb_count; ++limb)
                if (integer_residues[matrix * metadata.limb_count + limb] >= metadata.moduli[limb])
                    return set_error("integer scalar residue is not canonical");
    const size_t coefficients_per_limb = metadata.inner * metadata.n;
    const size_t total_coefficients = matrix_count * metadata.limb_count * coefficients_per_limb;
    if (total_coefficients == 0) return 0;
    const size_t pointer_count = matrix_count * metadata.limb_count;
    const uint8_t **d_matrices = nullptr;
    const uint8_t **d_scalars = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    MatrixBatchWorkspace workspace;
    status = workspace.acquire(outputs[0], metadata, GPU_MATRIX_BATCH_SCALAR);
    if (status != 0) return status;
    d_matrices = reinterpret_cast<decltype(d_matrices)>(workspace.region(0));
    d_scalars = reinterpret_cast<decltype(d_scalars)>(workspace.region(1));
    d_outputs = reinterpret_cast<decltype(d_outputs)>(workspace.region(2));
    d_strides = reinterpret_cast<decltype(d_strides)>(workspace.region(3));
    d_bytes = reinterpret_cast<decltype(d_bytes)>(workspace.region(4));
    d_moduli = reinterpret_cast<decltype(d_moduli)>(workspace.region(5));
    auto *d_columns = views
        ? static_cast<MatrixBatchGeometry *>(workspace.region(6)) : nullptr;
    cudaError_t error = cudaSuccess;
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_matrices, metadata.left.data(), pointer_count * sizeof(uint8_t *),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_scalars, integer_residues ? static_cast<const void *>(integer_residues) :
            static_cast<const void *>(metadata.right.data()), pointer_count * sizeof(uint64_t),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count,
        cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(
        d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess && d_columns) error = cudaMemcpyAsync(
        d_columns, metadata.views.data(), matrix_count * sizeof(MatrixBatchGeometry),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    constexpr int threads = 256;
    const int blocks = static_cast<int>((total_coefficients + threads - 1) / threads);
    matrix_scalar_mul_batch_kernel<<<blocks, threads, 0, metadata.stream>>>(
        d_matrices,
        d_scalars,
        d_outputs,
        d_strides,
        d_bytes,
        d_moduli,
        metadata.limb_count,
        coefficients_per_limb,
        total_coefficients,
        metadata.n,
        d_columns,
        metadata.columns,
        integer_residues ? reinterpret_cast<const uint64_t *>(d_scalars) : nullptr);
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    status = finish_matrix_batch(metadata, outputs, matrices, scalars);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx)
    {
        outputs[matrix_idx]->format = matrices[matrix_idx]->format;
    }
    if (status != 0) return status;
    return workspace.complete();
}

extern "C" int gpu_matrix_mul_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *left,
    const GpuMatrix *const *right,
    const GpuMatrixBatchView *views,
    size_t matrix_count)
{
    if (!outputs || matrix_count == 0 || !outputs[0] || !outputs[0]->ctx ||
        !outputs[0]->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_mul_batch");
    GpuAllocationActivity activity(outputs[0]->ctx->execution.get(), -1);
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(outputs, left, right, matrix_count, 1, &metadata, views);
    if (status != 0) return status;
    const size_t pointer_count = matrix_count * metadata.limb_count;
    if (pointer_count > 65535 ||
        metadata.columns > static_cast<size_t>(INT_MAX) * kMatmulTileN ||
        metadata.rows > static_cast<size_t>(65535) * kMatmulTileM)
    {
        return set_error("matrix multiplication batch exceeds CUDA grid dimensions");
    }
    const uint8_t **d_left = nullptr;
    const uint8_t **d_right = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    uint64_t *d_reciprocals = nullptr;
    std::vector<uint64_t> reciprocals(metadata.limb_count);
    bool use_thin_row_kernel = metadata.rows == 1;
    bool lazy_reduction = use_thin_row_kernel;
    if (use_thin_row_kernel)
    {
        for (size_t limb = 0; limb < metadata.limb_count; ++limb)
        {
            if (!matrix_barrett_u32_reciprocal(metadata.moduli[limb], &reciprocals[limb]))
            {
                use_thin_row_kernel = false;
                break;
            }
            lazy_reduction =
                lazy_reduction && matrix_lazy_dot_u64(metadata.inner, metadata.moduli[limb]);
        }
    }
    const size_t groups_per_matrix = use_thin_row_kernel
                                         ? (metadata.n + kThinMatmulWarpSize - 1) /
                                               kThinMatmulWarpSize
                                         : metadata.n;
    const size_t coefficient_groups =
        std::min(groups_per_matrix, static_cast<size_t>(65535) / pointer_count);
    MatrixBatchWorkspace workspace;
    status = workspace.acquire(outputs[0], metadata, GPU_MATRIX_BATCH_MULTIPLY);
    if (status != 0) return status;
    d_left = reinterpret_cast<decltype(d_left)>(workspace.region(0));
    d_right = reinterpret_cast<decltype(d_right)>(workspace.region(1));
    d_outputs = reinterpret_cast<decltype(d_outputs)>(workspace.region(2));
    d_strides = reinterpret_cast<decltype(d_strides)>(workspace.region(3));
    d_bytes = reinterpret_cast<decltype(d_bytes)>(workspace.region(4));
    d_moduli = reinterpret_cast<decltype(d_moduli)>(workspace.region(5));
    d_reciprocals = reinterpret_cast<decltype(d_reciprocals)>(workspace.region(6));
    auto *d_views = views ? static_cast<MatrixBatchGeometry *>(workspace.region(7)) : nullptr;
    cudaError_t error = cudaSuccess;
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_left, metadata.left.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_right, metadata.right.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_outputs, metadata.outputs.data(), pointer_count * sizeof(uint8_t *), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count, cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess) error = cudaMemcpyAsync(d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess && use_thin_row_kernel) error = cudaMemcpyAsync(d_reciprocals, reciprocals.data(), metadata.limb_count * sizeof(uint64_t), cudaMemcpyHostToDevice, metadata.stream);
    if (error == cudaSuccess && d_views) error = cudaMemcpyAsync(
        d_views, metadata.views.data(), matrix_count * sizeof(MatrixBatchGeometry),
        cudaMemcpyHostToDevice, metadata.stream);
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    if (use_thin_row_kernel)
    {
        const dim3 block(kThinMatmulWarpSize, kThinMatmulColumnsPerBlock, 1);
        const dim3 grid(
            static_cast<unsigned int>(
                (metadata.columns + kThinMatmulColumnsPerBlock - 1) /
                kThinMatmulColumnsPerBlock),
            1,
            static_cast<unsigned int>(pointer_count * coefficient_groups));
        matrix_thin_row_matmul_batch_kernel<<<grid, block, 0, metadata.stream>>>(
            d_left, d_right, d_outputs, d_strides, d_bytes, d_moduli, d_reciprocals,
            metadata.limb_count, metadata.inner, metadata.columns, metadata.n,
            coefficient_groups, lazy_reduction, d_views);
    }
    else
    {
        const dim3 block(kMatmulTileN, kMatmulTileM, 1);
        const dim3 grid(
            static_cast<unsigned int>((metadata.columns + kMatmulTileN - 1) / kMatmulTileN),
            static_cast<unsigned int>((metadata.rows + kMatmulTileM - 1) / kMatmulTileM),
            static_cast<unsigned int>(pointer_count * coefficient_groups));
        matrix_matmul_batch_kernel<<<grid, block, 0, metadata.stream>>>(
            d_left, d_right, d_outputs, d_strides, d_bytes, d_moduli, metadata.limb_count,
            metadata.rows, metadata.inner, metadata.columns, metadata.n, coefficient_groups, d_views);
    }
    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        return set_error(error);
    }
    status = finish_matrix_batch(metadata, outputs, left, right);
    for (size_t matrix_idx = 0; matrix_idx < matrix_count; ++matrix_idx) outputs[matrix_idx]->format = GPU_POLY_FORMAT_EVAL;
    if (status != 0) return status;
    return workspace.complete();
}

extern "C" int gpu_matrix_mul_accumulate_batch(
    GpuMatrix *const *outputs,
    const GpuMatrix *const *left,
    const GpuMatrix *const *right,
    const GpuMatrix *const *coefficients,
    const GpuMatrix *const *biases,
    const size_t *inner_dimensions,
    size_t matrix_count,
    size_t product_count,
    const GpuMatrixBatchView *views,
    const GpuMatrixRange *bias_view,
    const uint64_t *integer_residues)
{
    if (!outputs || matrix_count == 0 || !outputs[0] || !outputs[0]->ctx ||
        !outputs[0]->ctx->execution)
        return set_error("invalid execution owner in gpu_matrix_mul_accumulate_batch");
    GpuAllocationActivity activity(outputs[0]->ctx->execution.get(), -1);
    if (views) {
        auto *out = outputs[0];
        auto *ctx = out->ctx;
        if (matrix_count != 1 || product_count == 0 || !left || !right || !biases ||
            !integer_residues || out->level < 0 || out->format != GPU_POLY_FORMAT_EVAL || ctx->N <= 0)
            return set_error("invalid accumulate range arguments");
        const size_t limbs = static_cast<size_t>(out->level) + 1;
        if (limbs > GPU_RUNTIME_MAX_LIMBS || ctx->limb_gpu_ids.size() < limbs ||
            ctx->moduli.size() < limbs || product_count > SIZE_MAX / limbs)
            return set_error("invalid accumulate range basis");
        auto valid = [&](const GpuMatrix *matrix, const GpuMatrixRange &range) {
            return matrix && matrix->ctx == ctx && matrix->level == out->level &&
                matrix->format == GPU_POLY_FORMAT_EVAL && range.row_start <= range.row_end &&
                range.row_end <= matrix->rows && range.column_start <= range.column_end &&
                range.column_end <= matrix->cols && (matrix->cols == 0 || matrix->rows <= SIZE_MAX / matrix->cols);
        };
        const auto output_range = views[0].output;
        if (!valid(out, output_range)) return set_error("invalid accumulate output range");
        const size_t rows = output_range.row_end - output_range.row_start;
        const size_t columns = output_range.column_end - output_range.column_start;
        const size_t n = static_cast<size_t>(ctx->N);
        if ((columns && rows > SIZE_MAX / columns) || rows * columns > SIZE_MAX / n)
            return set_error("accumulate output extent overflow");
        using Descriptor = GpuMatrix::SharedLimbBuffer::DeviceDescriptor;
        std::vector<AccumulateTerm> terms(product_count);
        // Complete shape, alias and descriptor validation before queuing waits
        // or kernels. Empty dot products never dereference their empty owners.
        for (size_t i = 0; i < product_count; ++i) {
            const auto &v = views[i];
            if (!valid(left[i], v.left) || !valid(right[i], v.right) || left[i] == out || right[i] == out ||
                v.output.row_start != output_range.row_start || v.output.row_end != output_range.row_end ||
                v.output.column_start != output_range.column_start || v.output.column_end != output_range.column_end)
                return set_error("invalid accumulate product range or output alias");
            const size_t lr = v.left.row_end - v.left.row_start, lc = v.left.column_end - v.left.column_start;
            const size_t rr = v.right.row_end - v.right.row_start, rc = v.right.column_end - v.right.column_start;
            const unsigned scalar = lr == 1 && lc == 1 ? 1 : (rr == 1 && rc == 1 ? 2 : 0);
            if ((scalar == 1 && (rr != rows || rc != columns)) ||
                (scalar == 2 && (lr != rows || lc != columns)) ||
                (scalar == 0 && (lr != rows || rc != columns || lc != rr)))
                return set_error("accumulate products have incompatible dimensions");
            auto &term = terms[i];
            term.left_offset = v.left.row_start * left[i]->cols + v.left.column_start;
            term.right_offset = v.right.row_start * right[i]->cols + v.right.column_start;
            term.left_pitch = left[i]->cols; term.right_pitch = right[i]->cols;
            term.inner = scalar ? 1 : lc; term.scalar = scalar;
            for (size_t limb = 0; limb < limbs; ++limb) {
                const uint64_t residue = integer_residues[i * limbs + limb];
                if (residue >= ctx->moduli[limb]) return set_error("noncanonical accumulate coefficient");
                term.residues[limb] = residue;
            }
        }
        const auto *bias = biases[0];
        if (bias && (!bias_view || !valid(bias, *bias_view) || bias == out ||
            bias_view->row_end - bias_view->row_start != rows ||
            bias_view->column_end - bias_view->column_start != columns))
            return set_error("invalid accumulate bias range");
        if (rows == 0 || columns == 0) return 0;
        AccumulateRangeMetadata metadata{};
        int device = -1;
        auto descriptors = [&](const GpuMatrix *matrix, const Descriptor **table) -> int {
            for (size_t limb = 0; limb < limbs; ++limb) {
                const auto id = ctx->limb_gpu_ids[limb];
                if (id.x >= matrix->shared_limb_buffers.size()) return set_error("invalid accumulate partition");
                const auto &buffer = matrix->shared_limb_buffers[id.x];
                if (!buffer.device_descriptors || id.y >= buffer.limb_count)
                    return set_error("missing accumulate descriptors");
                if (device < 0) device = buffer.device;
                if (buffer.device != device) return set_error("accumulate inputs must share a device");
                if (limb == 0) *table = buffer.device_descriptors;
                else if (*table != buffer.device_descriptors) return set_error("accumulate descriptors span partitions");
            }
            return 0;
        };
        int status = descriptors(out, &metadata.output);
        if (status != 0) return status;
        if (bias) { status = descriptors(bias, &metadata.bias); if (status != 0) return status; }
        for (size_t i = 0; i < product_count; ++i) {
            if (terms[i].inner == 0) continue;
            status = descriptors(left[i], &terms[i].left); if (status != 0) return status;
            status = descriptors(right[i], &terms[i].right); if (status != 0) return status;
        }
        metadata.output_offset = output_range.row_start * out->cols + output_range.column_start;
        metadata.output_pitch = out->cols;
        if (bias) {
            metadata.bias_offset = bias_view->row_start * bias->cols + bias_view->column_start;
            metadata.bias_pitch = bias->cols;
        }
        for (size_t limb = 0; limb < limbs; ++limb) {
            metadata.indices[limb] = ctx->limb_gpu_ids[limb].y;
            metadata.moduli[limb] = ctx->moduli[limb];
            metadata.reciprocals[limb] = UINT64_MAX / ctx->moduli[limb];
        }
        cudaStream_t stream = nullptr;
        status = matrix_limb_stream(out, ctx->limb_gpu_ids[0], &stream);
        if (status != 0) return status;
        if (!stream) return set_error("missing accumulate output stream");
        cudaError_t error = cudaSetDevice(device); if (error != cudaSuccess) return set_error(error);
        status = matrix_wait_all_limb_streams(out, device, stream, true); if (status != 0) return status;
        if (bias) { status = matrix_wait_all_limb_streams(bias, device, stream, false, true); if (status != 0) return status; }
        for (size_t i = 0; i < product_count; ++i) {
            if (terms[i].inner == 0) continue;
            status = matrix_wait_all_limb_streams(left[i], device, stream, false, true); if (status != 0) return status;
            status = matrix_wait_all_limb_streams(right[i], device, stream, false, true); if (status != 0) return status;
        }
        const size_t count = rows * columns * n;
        const dim3 grid(static_cast<unsigned>(std::min(count / 256 + (count % 256 != 0), size_t{65535})), 1, static_cast<unsigned>(limbs));
        for (size_t start = 0; start < product_count; start += metadata.term_count) {
            metadata.term_count = std::min(kAccumulateTerms, product_count - start);
            std::copy_n(terms.data() + start, metadata.term_count, metadata.terms);
            metadata.accumulate = start != 0;
            if (start != 0) metadata.bias = nullptr;
            matrix_accumulate_range_kernel<<<grid, 256, 0, stream>>>(metadata, columns, count, n);
            error = cudaGetLastError(); if (error != cudaSuccess) return set_error(error);
        }
        status = matrix_record_all_limb_writes(out, stream, true); if (status != 0) return status;
        const auto first = ctx->limb_gpu_ids[0];
        const cudaEvent_t completion = out->exec_limb_states[first.x][first.y].write_done;
        for (size_t i = 0; i < product_count; ++i) {
            if (terms[i].inner == 0) continue;
            status = matrix_track_all_limb_consumers(left[i], device, stream, completion, true, true); if (status != 0) return status;
            status = matrix_track_all_limb_consumers(right[i], device, stream, completion, true, true); if (status != 0) return status;
        }
        return bias ? matrix_track_all_limb_consumers(bias, device, stream, completion, true, true) : 0;
    }
    if (bias_view || integer_residues) return set_error("accumulate residues require views");
    if (!outputs || !left || !right || !coefficients || !biases || !inner_dimensions ||
        matrix_count == 0 || product_count == 0)
    {
        return set_error("invalid multiply-accumulate batch arguments");
    }
    std::vector<const GpuMatrix *> first_left(matrix_count);
    std::vector<const GpuMatrix *> first_right(matrix_count);
    for (size_t matrix = 0; matrix < matrix_count; ++matrix)
    {
        first_left[matrix] = left[matrix * product_count];
        first_right[matrix] = right[matrix * product_count];
    }
    MatrixBatchMetadata metadata;
    int status = prepare_matrix_batch(
        outputs, first_left.data(), first_right.data(), matrix_count, 1, &metadata);
    if (status != 0) return status;
    if (metadata.rows != 1) return set_error("multiply-accumulate requires row matrices");
    const size_t product_total = matrix_count * product_count;
    const size_t product_pointer_count = product_total * metadata.limb_count;
    const size_t output_pointer_count = matrix_count * metadata.limb_count;
    if (output_pointer_count > 65535)
    {
        return set_error("multiply-accumulate batch exceeds CUDA grid depth");
    }
    std::vector<const uint8_t *> host_left(product_pointer_count);
    std::vector<const uint8_t *> host_right(product_pointer_count);
    std::vector<const uint8_t *> host_coefficients(product_pointer_count, nullptr);
    std::vector<const uint8_t *> host_biases(output_pointer_count, nullptr);
    size_t maximum_inner = 0;
    for (size_t matrix = 0; matrix < matrix_count; ++matrix)
    {
        for (size_t product = 0; product < product_count; ++product)
        {
            const size_t product_index = matrix * product_count + product;
            const GpuMatrix *lhs = left[product_index];
            const GpuMatrix *rhs = right[product_index];
            const GpuMatrix *coefficient = coefficients[product_index];
            const size_t inner = inner_dimensions[product_index];
            maximum_inner = std::max(maximum_inner, inner);
            if (!lhs || !rhs || lhs->ctx != metadata.context || rhs->ctx != metadata.context ||
                lhs->format != GPU_POLY_FORMAT_EVAL || rhs->format != GPU_POLY_FORMAT_EVAL ||
                lhs->level != metadata.level || rhs->level != metadata.level ||
                lhs->rows != 1 || lhs->cols != inner || rhs->rows != inner ||
                rhs->cols != metadata.columns ||
                (coefficient && (coefficient->ctx != metadata.context ||
                                 coefficient->level != metadata.level ||
                                 coefficient->format != GPU_POLY_FORMAT_EVAL ||
                                 coefficient->rows != 1 || coefficient->cols != 1)))
            {
                return set_error("invalid multiply-accumulate product layout");
            }
            for (size_t limb = 0; limb < metadata.limb_count; ++limb)
            {
                const dim3 limb_id = metadata.limb_ids[limb];
                const size_t pointer_index = product_index * metadata.limb_count + limb;
                host_left[pointer_index] = matrix_limb_ptr_by_id(lhs, 0, limb_id);
                host_right[pointer_index] = matrix_limb_ptr_by_id(rhs, 0, limb_id);
                if (coefficient)
                    host_coefficients[pointer_index] = matrix_limb_ptr_by_id(coefficient, 0, limb_id);
                int lhs_device = -1;
                int rhs_device = -1;
                size_t lhs_stride = 0;
                size_t rhs_stride = 0;
                uint8_t lhs_bytes = 0;
                uint8_t rhs_bytes = 0;
                if (!host_left[pointer_index] || !host_right[pointer_index] ||
                    (coefficient && !host_coefficients[pointer_index]) ||
                    matrix_limb_device(lhs, limb_id, &lhs_device) != 0 ||
                    matrix_limb_device(rhs, limb_id, &rhs_device) != 0 ||
                    !matrix_limb_metadata_by_id(lhs, limb_id, &lhs_stride, &lhs_bytes) ||
                    !matrix_limb_metadata_by_id(rhs, limb_id, &rhs_stride, &rhs_bytes) ||
                    lhs_device != metadata.device || rhs_device != metadata.device ||
                    lhs_stride != metadata.strides[limb] ||
                    rhs_stride != metadata.strides[limb] ||
                    lhs_bytes != metadata.coefficient_bytes[limb] ||
                    rhs_bytes != metadata.coefficient_bytes[limb])
                    return set_error("invalid multiply-accumulate limb pointer");
                status = matrix_wait_limb_stream(lhs, limb_id, metadata.device, metadata.stream);
                if (status == 0)
                    status = matrix_wait_limb_stream(rhs, limb_id, metadata.device, metadata.stream);
                if (status == 0 && coefficient)
                    status = matrix_wait_limb_stream(
                        coefficient, limb_id, metadata.device, metadata.stream);
                if (status != 0) return status;
            }
        }
        const GpuMatrix *bias = biases[matrix];
        if (bias && (bias->ctx != metadata.context || bias->level != metadata.level ||
                     bias->format != GPU_POLY_FORMAT_EVAL || bias->rows != 1 ||
                     bias->cols != metadata.columns))
            return set_error("invalid multiply-accumulate bias layout");
        for (size_t limb = 0; limb < metadata.limb_count && bias; ++limb)
        {
            const dim3 limb_id = metadata.limb_ids[limb];
            host_biases[matrix * metadata.limb_count + limb] =
                matrix_limb_ptr_by_id(bias, 0, limb_id);
            if (!host_biases[matrix * metadata.limb_count + limb])
                return set_error("invalid multiply-accumulate bias pointer");
            status = matrix_wait_limb_stream(bias, limb_id, metadata.device, metadata.stream);
            if (status != 0) return status;
        }
    }
    std::vector<uint64_t> reciprocals(metadata.limb_count);
    bool lazy_reduction = true;
    for (size_t limb = 0; limb < metadata.limb_count; ++limb)
    {
        if (!matrix_barrett_u32_reciprocal(metadata.moduli[limb], &reciprocals[limb]))
            return set_error("multiply-accumulate requires 32-bit CRT moduli");
        lazy_reduction = lazy_reduction &&
            matrix_lazy_dot_u64(maximum_inner, metadata.moduli[limb]);
    }
    const uint8_t **d_left = nullptr;
    const uint8_t **d_right = nullptr;
    const uint8_t **d_coefficients = nullptr;
    const uint8_t **d_biases = nullptr;
    uint8_t **d_outputs = nullptr;
    size_t *d_inner = nullptr;
    size_t *d_strides = nullptr;
    uint8_t *d_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    uint64_t *d_reciprocals = nullptr;
    MatrixBatchWorkspace workspace;
    status = workspace.acquire(outputs[0], metadata, GPU_MATRIX_BATCH_ACCUMULATE, product_count);
    if (status != 0) return status;
    d_left = reinterpret_cast<decltype(d_left)>(workspace.region(0));
    d_right = reinterpret_cast<decltype(d_right)>(workspace.region(1));
    d_coefficients = reinterpret_cast<decltype(d_coefficients)>(workspace.region(2));
    d_biases = reinterpret_cast<decltype(d_biases)>(workspace.region(3));
    d_outputs = reinterpret_cast<decltype(d_outputs)>(workspace.region(4));
    d_inner = reinterpret_cast<decltype(d_inner)>(workspace.region(5));
    d_strides = reinterpret_cast<decltype(d_strides)>(workspace.region(6));
    d_bytes = reinterpret_cast<decltype(d_bytes)>(workspace.region(7));
    d_moduli = reinterpret_cast<decltype(d_moduli)>(workspace.region(8));
    d_reciprocals = reinterpret_cast<decltype(d_reciprocals)>(workspace.region(9));
    cudaError_t error = cudaSuccess;
#define COPY_ASYNC(dst, src, bytes) if (error == cudaSuccess) error = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, metadata.stream)
    COPY_ASYNC(d_left, host_left.data(), product_pointer_count * sizeof(uint8_t *));
    COPY_ASYNC(d_right, host_right.data(), product_pointer_count * sizeof(uint8_t *));
    COPY_ASYNC(d_coefficients, host_coefficients.data(), product_pointer_count * sizeof(uint8_t *));
    COPY_ASYNC(d_biases, host_biases.data(), output_pointer_count * sizeof(uint8_t *));
    COPY_ASYNC(d_outputs, metadata.outputs.data(), output_pointer_count * sizeof(uint8_t *));
    COPY_ASYNC(d_inner, inner_dimensions, product_total * sizeof(size_t));
    COPY_ASYNC(d_strides, metadata.strides.data(), metadata.limb_count * sizeof(size_t));
    COPY_ASYNC(d_bytes, metadata.coefficient_bytes.data(), metadata.limb_count);
    COPY_ASYNC(d_moduli, metadata.moduli.data(), metadata.limb_count * sizeof(uint64_t));
    COPY_ASYNC(d_reciprocals, reciprocals.data(), metadata.limb_count * sizeof(uint64_t));
#undef COPY_ASYNC
    if (error != cudaSuccess) { return set_error(error); }
    const size_t groups_per_matrix = (metadata.n + kThinMatmulWarpSize - 1) / kThinMatmulWarpSize;
    const size_t coefficient_groups =
        std::min(groups_per_matrix, static_cast<size_t>(65535) / output_pointer_count);
    const dim3 block(kThinMatmulWarpSize, kThinMatmulColumnsPerBlock, 1);
    const dim3 grid(
        static_cast<unsigned int>((metadata.columns + kThinMatmulColumnsPerBlock - 1) /
                                  kThinMatmulColumnsPerBlock),
        1,
        static_cast<unsigned int>(output_pointer_count * coefficient_groups));
    matrix_thin_row_mul_accumulate_batch_kernel<<<grid, block, 0, metadata.stream>>>(
        d_left, d_right, d_coefficients, d_biases, d_outputs, d_inner, d_strides, d_bytes,
        d_moduli, d_reciprocals, metadata.limb_count, product_count, metadata.columns,
        metadata.n, coefficient_groups, lazy_reduction);
    error = cudaGetLastError();
    if (error != cudaSuccess) { return set_error(error); }
    for (size_t matrix = 0; matrix < matrix_count; ++matrix)
    {
        for (size_t limb = 0; limb < metadata.limb_count; ++limb)
        {
            const dim3 limb_id = metadata.limb_ids[limb];
            for (size_t product = 0; product < product_count; ++product)
            {
                const size_t index = matrix * product_count + product;
                status = matrix_track_limb_consumer(left[index], limb_id, metadata.device, metadata.stream);
                if (status == 0) status = matrix_track_limb_consumer(right[index], limb_id, metadata.device, metadata.stream);
                if (status == 0 && coefficients[index]) status = matrix_track_limb_consumer(coefficients[index], limb_id, metadata.device, metadata.stream);
                if (status != 0) { return status; }
            }
            if (biases[matrix])
            {
                status = matrix_track_limb_consumer(biases[matrix], limb_id, metadata.device, metadata.stream);
                if (status != 0) { return status; }
            }
            status = matrix_record_limb_write(outputs[matrix], limb_id, metadata.stream);
            if (status != 0) { return status; }
        }
        outputs[matrix]->format = GPU_POLY_FORMAT_EVAL;
    }
    return workspace.complete();
}
