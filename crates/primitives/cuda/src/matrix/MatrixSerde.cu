#include "gpu_prepared_plan.cuh"
#include "matrix/MatrixNTT.cuh"

static std::atomic<size_t> readback_prepared_acquisitions{0};
static std::atomic<size_t> rns_upload_prepared_acquisitions{0};
static std::atomic<size_t> reconstruction_prepared_acquisitions{0};

namespace
{
    constexpr int kMaxRnsLimbs = 64;
    constexpr int kMaxCoeffWords = 64;

    struct SerdeStreamRef
    {
        int device;
        cudaStream_t stream;
    };

    int serde_begin_private_stream(GpuMatrix *matrix, int device, GpuCudaResource &resource, cudaStream_t *stream)
    {
        const int acquired = resource.acquire(matrix->ctx, device, GPU_PREPARED_SUBMISSION_STREAM);
        if (acquired != 0) return acquired;
        *stream = resource.stream;
        cudaError_t error = cudaSuccess;
        cudaStream_t producer = nullptr;
        int status = matrix_limb_stream(matrix, matrix->ctx->limb_gpu_ids[0], &producer);
        if (status != 0) return status;
        const cudaEvent_t start = resource.event;
        // Include the producer's current boundary, not only an older matrix
        // writer event. This also gates existing matrices on benchmark starts.
        error = cudaEventRecord(start, producer);
        if (error == cudaSuccess) error = cudaStreamWaitEvent(*stream, start, 0);
        return error == cudaSuccess ? 0 : set_error(error);
    }

    int serde_finish_private_stream(GpuMatrix *matrix, int device, GpuCudaResource &resource, cudaStream_t &stream)
    {
        if (!stream) return 0;
        auto &owner = *matrix->ctx->execution;
        if (owner.unretired_work.load(std::memory_order_acquire)) {
            resource.quarantine();
            return set_error("compact serialization has unretired work");
        }
        // The caller has enqueued all temporary frees. Join their completion to
        // this matrix's producers, keeping unrelated owners independent.
        const cudaError_t recorded = cudaEventRecord(resource.event, stream);
        int status = recorded == cudaSuccess
            ? matrix_track_all_limb_consumers(matrix, device, stream, resource.event)
            : set_error(recorded);
        if (status != 0)
        {
            owner.memory_release_failed.store(true, std::memory_order_release);
            owner.unretired_work.store(true, std::memory_order_release);
            resource.quarantine();
            return status;
        }
        const int released = resource.release();
        if (released != 0)
        {
            owner.memory_release_failed.store(true, std::memory_order_release);
            owner.unretired_work.store(true, std::memory_order_release);
            return released;
        }
        stream = nullptr;
        return 0;
    }

    bool serde_checked_mul_size(size_t a, size_t b, size_t *out)
    {
        if (!out)
        {
            return false;
        }
        if (a != 0 && b > static_cast<size_t>(-1) / a)
        {
            return false;
        }
        *out = a * b;
        return true;
    }

    void serde_append_unique_stream(std::vector<SerdeStreamRef> &streams, int device, cudaStream_t stream)
    {
        if (!stream)
        {
            return;
        }
        for (const auto &entry : streams)
        {
            if (entry.device == device && entry.stream == stream)
            {
                return;
            }
        }
        streams.push_back(SerdeStreamRef{device, stream});
    }

    int serde_build_event_set_from_streams(
        GpuContext *ctx, const std::vector<SerdeStreamRef> &streams, GpuEventSet **out_events)
    {
        if (!out_events)
        {
            return set_error("invalid out_events in serde_build_event_set_from_streams");
        }
        *out_events = nullptr;
        if (streams.empty())
        {
            return 0;
        }

        auto *event_set = new GpuEventSet();
        event_set->execution = ctx->execution;
        event_set->entries.reserve(streams.size());
        for (const auto &entry : streams)
        {
            cudaError_t err = cudaSetDevice(entry.device);
            if (err != cudaSuccess)
            {
                gpu_event_set_destroy(event_set);
                return set_error(err);
            }

            auto resource = std::make_shared<GpuCudaResource>();
            const int status = resource->acquire(ctx, entry.device, GPU_PREPARED_COMPLETION_EVENT);
            if (status != 0) {
                gpu_event_set_destroy(event_set);
                return status;
            }
            err = cudaEventRecord(resource->event, entry.stream);
            if (err != cudaSuccess) {
                resource->quarantine();
                gpu_execution_mark_allocation_unknown(ctx->execution.get());
                gpu_event_set_destroy(event_set);
                return set_error(err);
            }
            event_set->entries.push_back(GpuEventSet::Entry{resource->event, entry.device, resource});
        }

        *out_events = event_set;
        return 0;
    }

    __device__ __forceinline__ uint64_t serde_mul_mod_u64_device(
        uint64_t a,
        uint64_t b,
        uint64_t modulus)
    {
        const unsigned __int128 prod =
            static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
        return static_cast<uint64_t>(prod % static_cast<unsigned __int128>(modulus));
    }

    __global__ void serde_pack_u64_limbs_to_packed_kernel(
        const uint64_t *src_words,
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *limbs,
        size_t limb_count,
        size_t poly_count,
        size_t n)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total = poly_count * limb_count * n;
        if (idx >= total) return;
        const size_t poly_limb = idx / n;
        const size_t poly_idx = poly_limb / limb_count;
        const auto limb = limbs[poly_limb % limb_count];
        matrix_store_limb_u64(
            limb.base, poly_idx, idx % n, limb.stride, limb.width, src_words[idx]);
    }

    __global__ void serde_unpack_packed_limbs_to_u64_kernel(
        const GpuMatrix::SharedLimbBuffer::DeviceDescriptor *limbs,
        size_t limb_count,
        uint64_t *dst_words,
        size_t poly_count,
        size_t n)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        const size_t total = poly_count * limb_count * n;
        if (idx >= total) return;
        const size_t poly_limb = idx / n;
        const size_t poly_idx = poly_limb / limb_count;
        const size_t limb_idx = poly_limb % limb_count;
        const auto limb = limbs[limb_idx];
        dst_words[idx] = matrix_load_limb_u64(
            limb.base, poly_idx, idx % n, limb.stride, limb.width);
    }

    __global__ void serde_reconstruct_rns_to_words_kernel(
        const uint8_t *const *limb_ptrs,
        const size_t *limb_strides,
        const uint8_t *limb_coeff_bytes,
        const uint64_t *moduli,
        const uint64_t *garner_inverses,
        int inverse_stride,
        int limb_count,
        size_t coeff_count,
        size_t n,
        int words_per_coeff,
        uint64_t *coeff_words_out,
        int *overflow_out)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= coeff_count)
        {
            return;
        }
        const size_t poly_idx = idx / n;
        const size_t coeff_idx = idx % n;

        uint64_t mixed_digits[kMaxRnsLimbs];
        uint64_t coeff_words[kMaxCoeffWords];

        for (int i = 0; i < limb_count; ++i)
        {
            const size_t stride = limb_strides[static_cast<size_t>(i)];
            const uint8_t coeff_bytes = limb_coeff_bytes[static_cast<size_t>(i)];
            mixed_digits[i] = matrix_load_limb_u64(
                                  limb_ptrs[i],
                                  poly_idx,
                                  coeff_idx,
                                  stride,
                                  coeff_bytes) %
                              moduli[i];
        }

        const size_t inverse_stride_sz = static_cast<size_t>(inverse_stride);
        for (int i = 1; i < limb_count; ++i)
        {
            const uint64_t qi = moduli[i];
            uint64_t t = mixed_digits[i];
            for (int j = 0; j < i; ++j)
            {
                const uint64_t xj_mod_qi = mixed_digits[j] % qi;
                const uint64_t diff =
                    t >= xj_mod_qi
                        ? (t - xj_mod_qi)
                        : static_cast<uint64_t>(
                              static_cast<unsigned __int128>(t) +
                              static_cast<unsigned __int128>(qi) -
                              static_cast<unsigned __int128>(xj_mod_qi));
                const uint64_t inv =
                    garner_inverses[static_cast<size_t>(j) * inverse_stride_sz +
                                    static_cast<size_t>(i)];
                t = serde_mul_mod_u64_device(diff, inv, qi);
            }
            mixed_digits[i] = t;
        }

        for (int w = 0; w < words_per_coeff; ++w)
        {
            coeff_words[w] = 0;
        }

        for (int i = limb_count - 1; i >= 0; --i)
        {
            const uint64_t qi = moduli[i];
            uint64_t carry = mixed_digits[i];
            for (int w = 0; w < words_per_coeff; ++w)
            {
                const unsigned __int128 term =
                    static_cast<unsigned __int128>(coeff_words[w]) *
                        static_cast<unsigned __int128>(qi) +
                    static_cast<unsigned __int128>(carry);
                coeff_words[w] = static_cast<uint64_t>(term);
                carry = static_cast<uint64_t>(term >> 64);
            }
            if (carry != 0)
            {
                atomicExch(overflow_out, 1);
            }
        }

        uint64_t *dst = coeff_words_out + idx * static_cast<size_t>(words_per_coeff);
        for (int w = 0; w < words_per_coeff; ++w)
        {
            dst[w] = coeff_words[w];
        }
    }

    __device__ __forceinline__ uint32_t serde_bit_width_u64_device(uint64_t value)
    {
        return value == 0 ? 0u : static_cast<uint32_t>(64 - __clzll(value));
    }

    __device__ __forceinline__ int serde_compare_words_desc_device(
        const uint64_t *lhs,
        const uint64_t *rhs,
        int words_per_coeff)
    {
        for (int w = words_per_coeff - 1; w >= 0; --w)
        {
            const uint64_t a = lhs[w];
            const uint64_t b = rhs[w];
            if (a > b)
            {
                return 1;
            }
            if (a < b)
            {
                return -1;
            }
        }
        return 0;
    }

    __global__ void serde_center_coeff_words_kernel(
        uint64_t *coeff_words,
        size_t n,
        int words_per_coeff,
        const uint64_t *modulus_words,
        const uint64_t *half_modulus_words,
        uint8_t *sign_bits_out,
        unsigned int *max_abs_bits_out)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= n)
        {
            return;
        }

        uint64_t *coeff = coeff_words + idx * static_cast<size_t>(words_per_coeff);
        const bool is_negative =
            serde_compare_words_desc_device(coeff, half_modulus_words, words_per_coeff) > 0;

        uint64_t borrow = 0;
        if (is_negative)
        {
            for (int w = 0; w < words_per_coeff; ++w)
            {
                const unsigned __int128 minuend =
                    static_cast<unsigned __int128>(modulus_words[w]);
                const unsigned __int128 subtrahend =
                    static_cast<unsigned __int128>(coeff[w]) + static_cast<unsigned __int128>(borrow);
                if (minuend >= subtrahend)
                {
                    coeff[w] = static_cast<uint64_t>(minuend - subtrahend);
                    borrow = 0;
                }
                else
                {
                    coeff[w] =
                        static_cast<uint64_t>(
                            minuend +
                            (static_cast<unsigned __int128>(1) << 64) -
                            subtrahend);
                    borrow = 1ULL;
                }
            }
        }

        sign_bits_out[idx] = static_cast<uint8_t>(is_negative ? 1 : 0);

        uint32_t abs_bit_width = 0;
        for (int w = words_per_coeff - 1; w >= 0; --w)
        {
            const uint64_t word = coeff[w];
            if (word != 0)
            {
                abs_bit_width = static_cast<uint32_t>(w) * 64u + serde_bit_width_u64_device(word);
                break;
            }
        }
        atomicMax(max_abs_bits_out, abs_bit_width);
    }

    __global__ void serde_check_centered_bound_batch_kernel(
        const uint8_t *const *matrix_limb_ptrs,
        const size_t *limb_strides,
        const uint8_t *limb_coeff_bytes,
        const uint64_t *moduli,
        const uint64_t *garner_inverses,
        int inverse_stride,
        int limb_count,
        size_t total_coefficients,
        size_t coefficients_per_matrix,
        size_t n,
        int words_per_coeff,
        const uint64_t *modulus_words,
        const uint64_t *half_modulus_words,
        const uint64_t *bound_words,
        int *accepted,
        uint8_t *canonical_payload,
        size_t canonical_dst_cols,
        size_t canonical_magnitude_bytes,
        size_t canonical_dst_row,
        size_t canonical_dst_col,
        size_t canonical_tile_cols)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= total_coefficients)
        {
            return;
        }
        const size_t matrix_idx = idx / coefficients_per_matrix;
        const size_t local_idx = idx % coefficients_per_matrix;
        if (local_idx >= coefficients_per_matrix)
        {
            return;
        }
        const size_t poly_idx = local_idx / n;
        const size_t coeff_idx = local_idx % n;

        uint64_t mixed_digits[kMaxRnsLimbs];
        uint64_t coeff_words[kMaxCoeffWords];
        const size_t pointer_base = matrix_idx * static_cast<size_t>(limb_count);
        for (int i = 0; i < limb_count; ++i)
        {
            mixed_digits[i] = matrix_load_limb_u64(
                                  matrix_limb_ptrs[pointer_base + static_cast<size_t>(i)],
                                  poly_idx,
                                  coeff_idx,
                                  limb_strides[static_cast<size_t>(i)],
                                  limb_coeff_bytes[static_cast<size_t>(i)]) %
                              moduli[i];
        }
        const size_t inverse_stride_sz = static_cast<size_t>(inverse_stride);
        for (int i = 1; i < limb_count; ++i)
        {
            const uint64_t qi = moduli[i];
            uint64_t t = mixed_digits[i];
            for (int j = 0; j < i; ++j)
            {
                const uint64_t xj_mod_qi = mixed_digits[j] % qi;
                const uint64_t diff =
                    t >= xj_mod_qi
                        ? t - xj_mod_qi
                        : static_cast<uint64_t>(
                              static_cast<unsigned __int128>(t) + qi - xj_mod_qi);
                t = serde_mul_mod_u64_device(
                    diff,
                    garner_inverses[static_cast<size_t>(j) * inverse_stride_sz +
                                    static_cast<size_t>(i)],
                    qi);
            }
            mixed_digits[i] = t;
        }
        for (int w = 0; w < words_per_coeff; ++w)
        {
            coeff_words[w] = 0;
        }
        for (int i = limb_count - 1; i >= 0; --i)
        {
            uint64_t carry = mixed_digits[i];
            for (int w = 0; w < words_per_coeff; ++w)
            {
                const unsigned __int128 term =
                    static_cast<unsigned __int128>(coeff_words[w]) * moduli[i] + carry;
                coeff_words[w] = static_cast<uint64_t>(term);
                carry = static_cast<uint64_t>(term >> 64);
            }
        }
        const bool is_negative = serde_compare_words_desc_device(
                coeff_words,
                half_modulus_words,
                words_per_coeff) > 0;
        if (is_negative)
        {
            uint64_t borrow = 0;
            for (int w = 0; w < words_per_coeff; ++w)
            {
                const unsigned __int128 minuend = modulus_words[w];
                const unsigned __int128 subtrahend =
                    static_cast<unsigned __int128>(coeff_words[w]) + borrow;
                if (minuend >= subtrahend)
                {
                    coeff_words[w] = static_cast<uint64_t>(minuend - subtrahend);
                    borrow = 0;
                }
                else
                {
                    coeff_words[w] = static_cast<uint64_t>(
                        minuend + (static_cast<unsigned __int128>(1) << 64) - subtrahend);
                    borrow = 1;
                }
            }
        }
        if (serde_compare_words_desc_device(coeff_words, bound_words, words_per_coeff) > 0)
        {
            atomicExch(&accepted[matrix_idx], 0);
        }
        if (canonical_payload)
        {
            bool is_zero = true;
            for (int w = 0; w < words_per_coeff; ++w)
            {
                is_zero = is_zero && coeff_words[w] == 0;
            }
            const size_t tile_row = poly_idx / canonical_tile_cols;
            const size_t tile_col = poly_idx % canonical_tile_cols;
            const size_t dst_poly =
                (canonical_dst_row + tile_row) * canonical_dst_cols +
                canonical_dst_col + tile_col;
            uint8_t *dst = canonical_payload +
                (dst_poly * n + coeff_idx) * (canonical_magnitude_bytes + 1);
            dst[0] = is_zero ? 0 : (is_negative ? 2 : 1);
            for (size_t byte = 0; byte < canonical_magnitude_bytes; ++byte)
            {
                const size_t word = byte / sizeof(uint64_t);
                const size_t shift = (byte % sizeof(uint64_t)) * 8;
                dst[1 + byte] = word < static_cast<size_t>(words_per_coeff)
                    ? static_cast<uint8_t>(coeff_words[word] >> shift)
                    : 0;
            }
        }
    }

    __global__ void serde_pack_centered_coeff_words_bits_kernel(
        const uint64_t *centered_abs_words,
        const uint8_t *sign_bits,
        size_t n,
        int words_per_coeff,
        uint32_t bit_width,
        uint8_t *payload_out,
        size_t payload_len)
    {
        const size_t byte_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (byte_idx >= payload_len)
        {
            return;
        }

        const size_t total_bits = n * static_cast<size_t>(bit_width);
        const size_t base_bit = byte_idx * 8;
        uint8_t out = 0;
        for (size_t k = 0; k < 8; ++k)
        {
            const size_t bit_idx = base_bit + k;
            if (bit_idx >= total_bits)
            {
                break;
            }
            const size_t coeff_idx = bit_idx / static_cast<size_t>(bit_width);
            const size_t coeff_bit = bit_idx % static_cast<size_t>(bit_width);
            uint8_t bit = 0;
            if (coeff_bit == static_cast<size_t>(bit_width - 1))
            {
                bit = static_cast<uint8_t>(sign_bits[coeff_idx] & 0x1u);
            }
            else
            {
                const size_t word_idx = coeff_bit / 64;
                const uint32_t bit_in_word = static_cast<uint32_t>(coeff_bit % 64);
                if (word_idx < static_cast<size_t>(words_per_coeff))
                {
                    const uint64_t word =
                        centered_abs_words[coeff_idx * static_cast<size_t>(words_per_coeff) + word_idx];
                    bit = static_cast<uint8_t>((word >> bit_in_word) & uint64_t{1});
                }
            }
            out |= static_cast<uint8_t>(bit << k);
        }
        payload_out[byte_idx] = out;
    }

    __global__ void serde_unpack_packed_coeffs_mod_kernel(
        const uint8_t *payload,
        size_t coeff_count,
        size_t n,
        uint32_t bit_width,
        const uint64_t *moduli,
        int limb_count,
        uint8_t *const *limb_ptrs,
        const size_t *limb_strides,
        const uint8_t *limb_coeff_bytes)
    {
        const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
        if (idx >= coeff_count)
        {
            return;
        }
        const size_t poly_idx = idx / n;
        const size_t coeff_idx = idx % n;
        if (bit_width == 0)
        {
            for (int limb = 0; limb < limb_count; ++limb)
            {
                const size_t limb_idx = static_cast<size_t>(limb);
                matrix_store_limb_u64(
                    limb_ptrs[limb_idx],
                    poly_idx,
                    coeff_idx,
                    limb_strides[limb_idx],
                    limb_coeff_bytes[limb_idx],
                    0);
            }
            return;
        }

        const size_t base_bit = idx * static_cast<size_t>(bit_width);
        const uint32_t magnitude_bits = bit_width > 0 ? (bit_width - 1) : 0;
        const size_t sign_bit_idx = base_bit + static_cast<size_t>(bit_width - 1);
        const uint8_t sign_byte = payload[sign_bit_idx / 8];
        const bool is_negative =
            ((sign_byte >> (sign_bit_idx % 8)) & 0x1u) != 0;
        for (int limb = 0; limb < limb_count; ++limb)
        {
            const uint64_t modulus = moduli[static_cast<size_t>(limb)];
            uint64_t residue = 0;
            for (int b = static_cast<int>(magnitude_bits) - 1; b >= 0; --b)
            {
                const size_t bit_idx = base_bit + static_cast<size_t>(b);
                const uint8_t byte_val = payload[bit_idx / 8];
                const uint8_t bit = static_cast<uint8_t>((byte_val >> (bit_idx % 8)) & 0x1u);
                const unsigned __int128 term =
                    static_cast<unsigned __int128>(residue) * static_cast<unsigned __int128>(2u) +
                    static_cast<unsigned __int128>(bit);
                residue =
                    static_cast<uint64_t>(term % static_cast<unsigned __int128>(modulus));
            }
            if (is_negative && residue != 0)
            {
                residue = modulus - residue;
            }
            const size_t limb_idx = static_cast<size_t>(limb);
            matrix_store_limb_u64(
                limb_ptrs[limb_idx],
                poly_idx,
                coeff_idx,
                limb_strides[limb_idx],
                limb_coeff_bytes[limb_idx],
                residue);
        }
    }

    bool serde_compute_modulus_words_le(
        const std::vector<uint64_t> &moduli,
        std::vector<uint64_t> *out_words)
    {
        if (!out_words)
        {
            return false;
        }
        out_words->clear();
        out_words->push_back(1u);
        for (uint64_t modulus : moduli)
        {
            unsigned __int128 carry = 0;
            for (size_t i = 0; i < out_words->size(); ++i)
            {
                const unsigned __int128 term =
                    static_cast<unsigned __int128>((*out_words)[i]) *
                        static_cast<unsigned __int128>(modulus) +
                    carry;
                (*out_words)[i] = static_cast<uint64_t>(term);
                carry = term >> 64;
            }
            if (carry != 0)
            {
                out_words->push_back(static_cast<uint64_t>(carry));
            }
        }
        while (out_words->size() > 1 && out_words->back() == 0)
        {
            out_words->pop_back();
        }
        return true;
    }

    void serde_shift_words_right_one_le(std::vector<uint64_t> *words)
    {
        if (!words || words->empty())
        {
            return;
        }
        uint64_t carry = 0;
        for (size_t i = words->size(); i-- > 0;)
        {
            const uint64_t current = (*words)[i];
            (*words)[i] = (current >> 1) | (carry << 63);
            carry = current & 1ULL;
        }
        while (words->size() > 1 && words->back() == 0)
        {
            words->pop_back();
        }
    }

    bool serde_get_matrix_shape(
        const GpuMatrix *mat,
        int *out_level,
        size_t *out_n,
        size_t *out_limb_count,
        size_t *out_poly_count)
    {
        if (!mat || !mat->ctx || !out_level || !out_n ||
            !out_limb_count || !out_poly_count)
        {
            return false;
        }
        if (mat->level < 0)
        {
            return false;
        }
        const int N = mat->ctx->N;
        if (N <= 0)
        {
            return false;
        }
        size_t poly_count = 0;
        if (!serde_checked_mul_size(mat->rows, mat->cols, &poly_count))
        {
            return false;
        }
        *out_level = mat->level;
        *out_n = static_cast<size_t>(N);
        *out_limb_count = static_cast<size_t>(mat->level + 1);
        *out_poly_count = poly_count;
        return true;
    }

    bool serde_compute_payload_len(size_t n, uint32_t max_bits, size_t *out_payload_len)
    {
        if (!out_payload_len)
        {
            return false;
        }
        if (max_bits == 0)
        {
            *out_payload_len = 0;
            return true;
        }
        size_t total_bits = 0;
        if (!serde_checked_mul_size(n, static_cast<size_t>(max_bits), &total_bits))
        {
            return false;
        }
        *out_payload_len = (total_bits + static_cast<size_t>(7)) / static_cast<size_t>(8);
        return true;
    }

}

#include "MatrixSerdeWorkspace.cu"

extern "C" int gpu_matrix_query_rns_workspace(
    GpuContext *ctx, int level, size_t rows, size_t columns,
    GpuPreparedWorkspaceLayout *out)
{
    if (!ctx || !out || ctx->gpu_ids.size() != 1 || ctx->N <= 0 || level < 0 ||
        static_cast<size_t>(level) >= ctx->moduli.size())
        return set_error("invalid single-device RNS transfer layout");
    size_t bytes = 0;
    if (!serde_checked_mul_size(rows, columns, &bytes) ||
        !serde_checked_mul_size(bytes, static_cast<size_t>(ctx->N), &bytes) ||
        !serde_checked_mul_size(bytes, static_cast<size_t>(level) + 1, &bytes) ||
        !serde_checked_mul_size(bytes, sizeof(uint64_t), &bytes))
        return set_error("RNS transfer workspace size overflow");
    *out = {bytes, alignof(uint64_t), GPU_PREPARED_TRANSFER_WORKSPACE};
    return 0;
}

extern "C" int gpu_matrix_load_rns_batch(
    GpuMatrix *mat,
    const uint8_t *bytes,
    size_t bytes_per_poly,
    int format,
    GpuEventSet **out_events)
{
    if (!mat || !mat->ctx || !mat->ctx->execution)
        return set_error("invalid allocation owner in gpu_matrix_load_rns_batch");
    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);

    if (mat) mat->host_observed_writer_ready.store(false, std::memory_order_release);
    if (!mat || !out_events)
    {
        return set_error("invalid gpu_matrix_load_rns_batch arguments");
    }
    *out_events = nullptr;
    if (!mat->ctx)
    {
        return set_error("invalid context in gpu_matrix_load_rns_batch");
    }

    GpuPolyFormat target_format;
    if (!parse_format(format, target_format))
    {
        return set_error("invalid format in gpu_matrix_load_rns_batch");
    }

    size_t count = 0;
    if (!serde_checked_mul_size(mat->rows, mat->cols, &count))
    {
        return set_error("matrix size overflow in gpu_matrix_load_rns_batch");
    }
    if (count > 0 && !bytes)
    {
        return set_error("null bytes in gpu_matrix_load_rns_batch");
    }

    if (mat->level < 0)
    {
        mat->format = target_format;
        return count == 0 ? 0 : set_error("invalid level in gpu_matrix_load_rns_batch");
    }
    const int N = mat->ctx->N;
    if (N <= 0)
    {
        return set_error("invalid ring dimension in gpu_matrix_load_rns_batch");
    }

    const size_t limb_count = static_cast<size_t>(mat->level + 1);
    size_t expected_words = 0;
    size_t expected_bytes = 0;
    if (!serde_checked_mul_size(limb_count, static_cast<size_t>(N), &expected_words) ||
        !serde_checked_mul_size(expected_words, sizeof(uint64_t), &expected_bytes))
    {
        return set_error("size overflow in gpu_matrix_load_rns_batch");
    }
    if (bytes_per_poly == 0 || bytes_per_poly % sizeof(uint64_t) != 0)
    {
        return set_error("bytes_per_poly must be a non-zero multiple of 8");
    }
    if (bytes_per_poly < expected_bytes)
    {
        return set_error("bytes_per_poly too small in gpu_matrix_load_rns_batch");
    }
    if (count == 0)
    {
        mat->format = target_format;
        return 0;
    }

    auto &limb_map = mat->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_load_rns_batch");
    }

    std::vector<SerdeStreamRef> streams;
    streams.reserve(limb_count);
    const size_t batch_limbs = mat->shared_limb_buffers.size() == 1 ? limb_count : 1;
    const size_t host_limb_bytes = static_cast<size_t>(N) * sizeof(uint64_t);
    size_t batch_poly_bytes = 0;
    size_t staging_bytes = 0;
    if (!serde_checked_mul_size(batch_limbs, host_limb_bytes, &batch_poly_bytes) ||
        !serde_checked_mul_size(count, batch_poly_bytes, &staging_bytes))
        return set_error("staging size overflow in gpu_matrix_load_rns_batch");
    for (size_t limb = 0; limb < limb_count; limb += batch_limbs)
    {
        const dim3 limb_id = limb_map[limb];
        const auto &buffer = mat->shared_limb_buffers[limb_id.x];
        const int device = buffer.device;
        cudaStream_t stream = nullptr;
        int status = matrix_limb_stream(mat, limb_id, &stream);
        if (status != 0) return status;
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) return set_error(err);
        for (size_t index = 0; index < batch_limbs; ++index)
        {
            status = matrix_wait_limb_stream(mat, limb_map[limb + index], device, stream);
            if (status != 0) return status;
        }
        GpuDeviceWorkspace staging;
        status = staging.acquire(mat->ctx, device, GPU_PREPARED_TRANSFER_WORKSPACE,
            staging_bytes, alignof(uint64_t), stream);
        if (status != 0) return status;
        auto *src_words_device = reinterpret_cast<uint64_t *>(staging.data);
        if (bytes_per_poly == batch_poly_bytes)
        {
            err = cudaMemcpyAsync(
                src_words_device, bytes, staging_bytes, cudaMemcpyHostToDevice, stream);
        }
        else
        {
            err = cudaMemcpy2DAsync(
                src_words_device,
                batch_poly_bytes,
                bytes + limb * host_limb_bytes,
                bytes_per_poly,
                batch_poly_bytes,
                count,
                cudaMemcpyHostToDevice,
                stream);
        }
        if (err == cudaSuccess)
        {
            const size_t total_coeff = staging_bytes / sizeof(uint64_t);
            const int threads = 256;
            const int blocks = static_cast<int>((total_coeff + threads - 1) / threads);
            serde_pack_u64_limbs_to_packed_kernel<<<blocks, threads, 0, stream>>>(
                src_words_device,
                buffer.device_descriptors + limb_id.y,
                batch_limbs,
                count,
                static_cast<size_t>(N));
            err = cudaGetLastError();
        }
        status = staging.release();
        if (err != cudaSuccess) return set_error(err);
        if (status != 0) return status;
        for (size_t index = 0; index < batch_limbs; ++index)
        {
            status = matrix_record_limb_write(mat, limb_map[limb + index], stream);
            if (status != 0) return status;
        }
        serde_append_unique_stream(streams, device, stream);
    }

    mat->format = target_format;
    return serde_build_event_set_from_streams(mat->ctx, streams, out_events);
}

extern "C" int gpu_matrix_rns_store_completion_events(const GpuMatrix *mat, size_t *out_count)
{
    if (!mat || !mat->ctx || !out_count || mat->level < 0)
        return set_error("invalid matrix RNS event query");
    *out_count = 0;
    if (mat->rows == 0 || mat->cols == 0) return 0;
    const size_t limbs = static_cast<size_t>(mat->level) + 1;
    if (mat->ctx->limb_gpu_ids.size() < limbs)
        return set_error("invalid matrix RNS event query basis");
    // One event per unpack/transfer batch is shared by all limb readers and
    // the returned snapshot, independently of the backing's producer streams.
    *out_count = mat->shared_limb_buffers.size() == 1 ? 1 : limbs;
    return 0;
}

extern "C" int gpu_matrix_store_rns_batch(
    const GpuMatrix *mat,
    uint8_t *bytes_out,
    size_t bytes_per_poly,
    int format,
    GpuEventSet **out_events)
{
    if (!mat || !mat->ctx || !mat->ctx->execution)
        return set_error("invalid allocation owner in gpu_matrix_store_rns_batch");
    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);

    if (!mat || !out_events)
    {
        return set_error("invalid gpu_matrix_store_rns_batch arguments");
    }
    *out_events = nullptr;
    if (!mat->ctx)
    {
        return set_error("invalid context in gpu_matrix_store_rns_batch");
    }

    GpuPolyFormat target_format;
    if (!parse_format(format, target_format))
    {
        return set_error("invalid format in gpu_matrix_store_rns_batch");
    }
    if (target_format != mat->format)
    {
        return set_error("format conversion is not supported in gpu_matrix_store_rns_batch");
    }

    size_t count = 0;
    if (!serde_checked_mul_size(mat->rows, mat->cols, &count))
    {
        return set_error("matrix size overflow in gpu_matrix_store_rns_batch");
    }
    if (count > 0 && !bytes_out)
    {
        return set_error("null bytes_out in gpu_matrix_store_rns_batch");
    }
    if (count == 0)
    {
        return 0;
    }

    if (mat->level < 0)
    {
        return set_error("invalid level in gpu_matrix_store_rns_batch");
    }
    const int N = mat->ctx->N;
    if (N <= 0)
    {
        return set_error("invalid ring dimension in gpu_matrix_store_rns_batch");
    }

    const size_t limb_count = static_cast<size_t>(mat->level + 1);
    size_t expected_words = 0;
    size_t expected_bytes = 0;
    if (!serde_checked_mul_size(limb_count, static_cast<size_t>(N), &expected_words) ||
        !serde_checked_mul_size(expected_words, sizeof(uint64_t), &expected_bytes))
    {
        return set_error("size overflow in gpu_matrix_store_rns_batch");
    }
    if (bytes_per_poly == 0 || bytes_per_poly % sizeof(uint64_t) != 0)
    {
        return set_error("bytes_per_poly must be a non-zero multiple of 8");
    }
    if (bytes_per_poly < expected_bytes)
    {
        return set_error("bytes_per_poly too small in gpu_matrix_store_rns_batch");
    }

    auto &limb_map = mat->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_store_rns_batch");
    }

    std::unique_ptr<GpuEventSet, decltype(&gpu_event_set_destroy)> events(
        new GpuEventSet(), gpu_event_set_destroy);
    events->execution = mat->ctx->execution;
    events->entries.reserve(limb_count);
    // Fleet matrices keep all CRT limbs on one device. Unpack that complete
    // allocation in one kernel and transfer each polynomial contiguously,
    // instead of allocating and scheduling a D2H copy for every CRT limb.
    const size_t batch_limbs = mat->shared_limb_buffers.size() == 1 ? limb_count : 1;
    const size_t host_limb_bytes = static_cast<size_t>(N) * sizeof(uint64_t);
    size_t batch_poly_bytes = 0;
    size_t staging_bytes = 0;
    if (!serde_checked_mul_size(batch_limbs, host_limb_bytes, &batch_poly_bytes) ||
        !serde_checked_mul_size(count, batch_poly_bytes, &staging_bytes))
        return set_error("staging size overflow in gpu_matrix_store_rns_batch");
    for (size_t limb = 0; limb < limb_count; limb += batch_limbs)
    {
        const dim3 limb_id = limb_map[limb];
        const auto &buffer = mat->shared_limb_buffers[limb_id.x];
        const int device = buffer.device;
        cudaStream_t stream = nullptr;
        int status = matrix_limb_stream(mat, limb_id, &stream);
        if (status != 0) return status;
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        for (size_t index = 0; index < batch_limbs; ++index)
        {
            status = matrix_wait_limb_stream(mat, limb_map[limb + index], device, stream, false, true);
            if (status != 0) return status;
        }
        GpuDeviceWorkspace staging;
        status = staging.acquire(mat->ctx, device, GPU_PREPARED_TRANSFER_WORKSPACE,
            staging_bytes, alignof(uint64_t), stream);
        if (status != 0) return status;
        auto *dst_words_device = reinterpret_cast<uint64_t *>(staging.data);
        const size_t total_coeff = staging_bytes / sizeof(uint64_t);
        const int threads = 256;
        const int blocks = static_cast<int>((total_coeff + threads - 1) / threads);
        serde_unpack_packed_limbs_to_u64_kernel<<<blocks, threads, 0, stream>>>(
            buffer.device_descriptors + limb_id.y,
            batch_limbs,
            dst_words_device,
            count,
            static_cast<size_t>(N));
        err = cudaGetLastError();
        if (err == cudaSuccess && bytes_per_poly == batch_poly_bytes)
        {
            err = cudaMemcpyAsync(
                bytes_out, dst_words_device, staging_bytes, cudaMemcpyDeviceToHost, stream);
        }
        else if (err == cudaSuccess)
        {
            err = cudaMemcpy2DAsync(
                bytes_out + limb * host_limb_bytes,
                bytes_per_poly,
                dst_words_device,
                batch_poly_bytes,
                batch_poly_bytes,
                count,
                cudaMemcpyDeviceToHost,
                stream);
        }
        status = staging.release();
        if (err != cudaSuccess) return set_error(err);
        if (status != 0) return status;
        // The whole batch finishes on this stream. Share its completion event
        // between the snapshot and all limb readers instead of claiming one
        // extra event for each limb whose producer uses a different stream.
        auto completion = std::make_shared<GpuCudaResource>();
        status = completion->acquire(mat->ctx, device, GPU_PREPARED_COMPLETION_EVENT);
        if (status != 0) return status;
        err = cudaEventRecord(completion->event, stream);
        if (err != cudaSuccess) {
            completion->quarantine();
            gpu_execution_mark_allocation_unknown(mat->ctx->execution.get());
            return set_error(err);
        }
        events->entries.push_back(GpuEventSet::Entry{completion->event, device, completion});
        for (size_t index = 0; index < batch_limbs; ++index)
        {
            status = matrix_track_limb_consumer(
                mat, limb_map[limb + index], device, stream, completion->event);
            if (status != 0) return status;
        }
    }

    *out_events = events.release();
    return 0;
}

extern "C" int gpu_matrix_store_const_coeff_batch(
    const GpuMatrix *mat,
    uint64_t *words_out,
    size_t words_per_poly,
    GpuEventSet **out_events)
{
    if (!mat || !mat->ctx || !mat->ctx->execution)
        return set_error("invalid allocation owner in gpu_matrix_store_const_coeff_batch");
    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);

    if (!mat || !out_events)
    {
        return set_error("invalid gpu_matrix_store_const_coeff_batch arguments");
    }
    *out_events = nullptr;
    if (!mat->ctx)
    {
        return set_error("invalid context in gpu_matrix_store_const_coeff_batch");
    }
    if (mat->format != GPU_POLY_FORMAT_COEFF)
    {
        return set_error("gpu_matrix_store_const_coeff_batch requires coeff format");
    }

    size_t count = 0;
    if (!serde_checked_mul_size(mat->rows, mat->cols, &count))
    {
        return set_error("matrix size overflow in gpu_matrix_store_const_coeff_batch");
    }
    if (count > 0 && !words_out)
    {
        return set_error("null words_out in gpu_matrix_store_const_coeff_batch");
    }
    if (count == 0)
    {
        return 0;
    }

    if (mat->level < 0)
    {
        return set_error("invalid level in gpu_matrix_store_const_coeff_batch");
    }
    const int N = mat->ctx->N;
    if (N <= 0)
    {
        return set_error("invalid ring dimension in gpu_matrix_store_const_coeff_batch");
    }
    const size_t limb_count = static_cast<size_t>(mat->level + 1);
    if (words_per_poly < limb_count)
    {
        return set_error("words_per_poly too small in gpu_matrix_store_const_coeff_batch");
    }

    size_t total_words = 0;
    if (!serde_checked_mul_size(count, words_per_poly, &total_words))
    {
        return set_error("output size overflow in gpu_matrix_store_const_coeff_batch");
    }
    std::fill_n(words_out, total_words, static_cast<uint64_t>(0));

    auto &limb_map = mat->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_matrix_store_const_coeff_batch");
    }

    std::vector<SerdeStreamRef> streams;
    streams.reserve(limb_count);
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        const uint8_t *src = matrix_limb_ptr_by_id(mat, 0, limb_id);
        if (!src)
        {
            return set_error("null matrix limb base pointer in gpu_matrix_store_const_coeff_batch");
        }

        int device = -1;
        cudaStream_t stream = nullptr;
        int status = matrix_limb_device(mat, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        status = matrix_limb_stream(mat, limb_id, &stream);
        if (status != 0)
        {
            return status;
        }

        size_t src_pitch = 0;
        uint8_t src_coeff_bytes = 0;
        if (!matrix_limb_metadata_by_id(mat, limb_id, &src_pitch, &src_coeff_bytes))
        {
            return set_error("invalid limb metadata in gpu_matrix_store_const_coeff_batch");
        }
        size_t src_width = 0;
        if (!serde_checked_mul_size(static_cast<size_t>(N), static_cast<size_t>(src_coeff_bytes), &src_width))
        {
            return set_error("size overflow in gpu_matrix_store_const_coeff_batch");
        }
        if (src_coeff_bytes == 0 || src_pitch < src_width)
        {
            return set_error("invalid source stride in gpu_matrix_store_const_coeff_batch");
        }

        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        status = matrix_wait_limb_stream(mat, limb_id, device, stream, false, true);
        if (status != 0)
        {
            return status;
        }

        uint8_t *dst =
            reinterpret_cast<uint8_t *>(words_out) + limb * sizeof(uint64_t);
        err = cudaMemcpy2DAsync(
            dst,
            words_per_poly * sizeof(uint64_t),
            src,
            src_pitch,
            src_coeff_bytes,
            count,
            cudaMemcpyDeviceToHost,
            stream);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }

        status = matrix_track_limb_consumer(mat, limb_id, device, stream);
        if (status != 0)
        {
            return status;
        }
        serde_append_unique_stream(streams, device, stream);
    }

    return serde_build_event_set_from_streams(mat->ctx, streams, out_events);
}

struct GpuPreparedConstCoeffReadbackLimb
{
    size_t ordinal;
    dim3 limb_id;
    int device;
    cudaStream_t stream;
    const uint8_t *source;
    size_t source_pitch;
    uint8_t source_width;
    cudaEvent_t completion;
    std::unique_ptr<GpuCudaResource> completion_resource;
    // Terminal-event generations. `armed` names the submission that may have
    // queued work for this limb, `recorded` names the submission whose
    // completion event was actually recorded on this limb's stream. A reclaim
    // decision is only valid for a generation whose armed limbs are all
    // recorded, so a partially submitted or unrecorded event can never be
    // mistaken for the current one.
    uint64_t armed_generation;
    uint64_t recorded_generation;
};

struct GpuPreparedConstCoeffReadback
{
    const GpuMatrix *matrix;
    uint64_t *words;
    size_t polynomial_count;
    size_t words_per_poly;
    size_t coefficient_index;
    size_t coefficient_count;
    std::vector<GpuPreparedConstCoeffReadbackLimb> limbs;
    uint64_t submission_generation;
    // Query and submit may be called from different host workers.  Protect
    // generation state as one transaction so a query can never observe a
    // partially armed submission.
    mutable std::mutex mutex;
};

extern "C" int gpu_matrix_prepare_const_coeff_readback(
    const GpuMatrix *mat,
    uint64_t *words_out,
    size_t words_per_poly,
    size_t coefficient_index,
    size_t coefficient_count,
    const GpuPreparedPlanDescriptor *plan,
    GpuPreparedConstCoeffReadback **out_plan)
{
    if (!out_plan)
        return set_error("null prepared coefficient readback output");
    *out_plan = nullptr;
    if (!mat || !mat->ctx || !mat->ctx->execution || mat->format != GPU_POLY_FORMAT_COEFF)
        return set_error("invalid prepared coefficient readback matrix");
    if (!plan || gpu_prepared_validate_descriptor(plan) != 0)
        return set_error("prepared coefficient readback requires a saved descriptor");
    if (mat->level < 0 || mat->ctx->N <= 0 || coefficient_count == 0 ||
        coefficient_index >= static_cast<size_t>(mat->ctx->N) ||
        coefficient_count > static_cast<size_t>(mat->ctx->N) - coefficient_index)
        return set_error("invalid prepared coefficient readback dimensions");

    size_t polynomial_count = 0;
    if (!serde_checked_mul_size(mat->rows, mat->cols, &polynomial_count))
        return set_error("prepared coefficient readback matrix size overflow");
    const size_t limb_count = static_cast<size_t>(mat->level) + 1;
    if (polynomial_count == 0 || !words_out ||
        words_per_poly < limb_count * coefficient_count)
        return set_error("invalid prepared coefficient readback output");
    size_t total_words = 0;
    if (!serde_checked_mul_size(polynomial_count, words_per_poly, &total_words))
        return set_error("prepared coefficient readback output size overflow");
    size_t total_bytes = 0;
    if (!serde_checked_mul_size(total_words, sizeof(uint64_t), &total_bytes))
        return set_error("prepared coefficient readback output byte size overflow");
    if (mat->ctx->limb_gpu_ids.size() < limb_count)
        return set_error("prepared coefficient readback limb metadata is incomplete");

    const GpuMatrix *owner = gpu_prepared_base_owner(mat);
    if (!owner || owner->ctx != mat->ctx || owner->level != mat->level)
        return set_error("prepared coefficient readback owner is invalid");
    if (plan->allocation_count != limb_count + 1 || plan->stream_count != limb_count)
        return set_error("prepared coefficient readback plan shape does not match its owner");
    const GpuPreparedResourceKey host_key{0, -1, -1, 0, 0, GPU_PREPARED_STAGE_READBACK};
    if (gpu_prepared_require_allocation(
            plan, 0, GPU_PREPARED_PINNED_HOST, &host_key, total_bytes, alignof(uint64_t)) != 0)
        return set_error("prepared coefficient readback plan destination does not match its owner");

    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);
    auto *prepared = new GpuPreparedConstCoeffReadback{
        mat, words_out, polynomial_count, words_per_poly,
        coefficient_index, coefficient_count, {}, 0, {}};
    try
    {
        prepared->limbs.reserve(limb_count);
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 limb_id = mat->ctx->limb_gpu_ids[limb];
            const uint8_t *source_base = matrix_limb_ptr_by_id(mat, 0, limb_id);
            if (!source_base)
                throw std::runtime_error("null prepared coefficient readback source");

            int device = -1;
            cudaStream_t stream = nullptr;
            int status = matrix_limb_device(mat, limb_id, &device);
            if (status != 0)
                throw std::runtime_error("invalid prepared coefficient readback limb device");
            status = matrix_limb_stream(mat, limb_id, &stream);
            if (status != 0 || !stream)
                throw std::runtime_error("invalid prepared coefficient readback stream");

            dim3 planned_limb{};
            GpuPreparedResourceKey key{};
            status = gpu_prepared_limb_key(
                owner->ctx, owner->level, limb, GPU_PREPARED_STAGE_READBACK, &planned_limb, &key);
            if (status != 0 || planned_limb.x != limb_id.x || planned_limb.y != limb_id.y ||
                key.device != device)
                throw std::runtime_error(
                    "prepared coefficient readback owner differs from its saved plan");
            status = gpu_prepared_require_allocation(
                plan, limb + 1, GPU_PREPARED_COMPLETION_EVENT, &key, 0, 1);
            if (status != 0)
                throw std::runtime_error("prepared coefficient readback completion claim missing");
            const auto &stream_entry = plan->streams[limb];
            if (stream_entry.origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
                std::memcmp(&stream_entry.key, &key, sizeof(key)) != 0 ||
                gpu_prepared_require_stream_slot(
                    owner->ctx, limb_id.x, stream, stream_entry.pool_slot) != 0)
                throw std::runtime_error(
                    "prepared coefficient readback stream differs from its saved descriptor");

            size_t source_pitch = 0;
            uint8_t source_width = 0;
            if (!matrix_limb_metadata_by_id(mat, limb_id, &source_pitch, &source_width) ||
                source_width == 0 ||
                source_pitch < static_cast<size_t>(mat->ctx->N) * source_width)
                throw std::runtime_error("invalid prepared coefficient readback layout");
            const uint8_t *source = source_base + coefficient_index * source_width;

            auto completion = std::make_unique<GpuCudaResource>();
            const int resource_status =
                completion->acquire(mat->ctx, device, GPU_PREPARED_COMPLETION_EVENT);
            if (resource_status != 0)
                throw std::runtime_error("failed to acquire prepared readback completion event");
            prepared->limbs.push_back(GpuPreparedConstCoeffReadbackLimb{
                limb,
                limb_id,
                device,
                stream,
                source,
                source_pitch,
                source_width,
                completion->event,
                std::move(completion),
                0,
                0,
            });
        }
    }
    catch (const std::exception &error)
    {
        delete prepared;
        return set_error(error.what());
    }
    *out_plan = prepared;
    readback_prepared_acquisitions.fetch_add(limb_count, std::memory_order_relaxed);
    return 0;
}

extern "C" void gpu_matrix_test_reset_serde_bind_counters()
{
    readback_prepared_acquisitions.store(0, std::memory_order_relaxed);
    rns_upload_prepared_acquisitions.store(0, std::memory_order_relaxed);
    reconstruction_prepared_acquisitions.store(0, std::memory_order_relaxed);
}

extern "C" size_t gpu_matrix_test_readback_prepared_acquisitions()
{
    return readback_prepared_acquisitions.load(std::memory_order_relaxed);
}

extern "C" size_t gpu_matrix_test_rns_upload_prepared_acquisitions()
{
    return rns_upload_prepared_acquisitions.load(std::memory_order_relaxed);
}

extern "C" size_t gpu_matrix_test_reconstruction_prepared_acquisitions()
{
    return reconstruction_prepared_acquisitions.load(std::memory_order_relaxed);
}

namespace
{
    // Reports a failed submission while still recording the terminal completion
    // of every limb that may already have queued work. The submission's own
    // cause is the more specific report; when a terminal record is refused the
    // generation stays unproven, so the pinned destination is retained instead
    // of freed while a queued copy could still read it.
    int fail_const_coeff_readback_submission(
        GpuPreparedConstCoeffReadback *plan, uint64_t generation, int status)
    {
        for (auto &limb : plan->limbs)
        {
            if (limb.armed_generation != generation || limb.recorded_generation == generation)
                continue;
            if (cudaSetDevice(limb.device) == cudaSuccess &&
                cudaEventRecord(limb.completion, limb.stream) == cudaSuccess)
                limb.recorded_generation = generation;
        }
        return status;
    }
}

extern "C" int gpu_matrix_submit_const_coeff_readback(
    GpuPreparedConstCoeffReadback *plan)
{
    if (!plan || !plan->matrix || !plan->matrix->ctx || !plan->words ||
        plan->limbs.empty())
        return set_error("invalid prepared coefficient readback plan");
    std::lock_guard<std::mutex> lock(plan->mutex);
    size_t total_words = 0;
    if (!serde_checked_mul_size(plan->polynomial_count, plan->words_per_poly, &total_words))
        return set_error("prepared coefficient readback output size overflow");
    std::fill_n(plan->words, total_words, static_cast<uint64_t>(0));

    const uint64_t generation = ++plan->submission_generation;
    for (auto &limb : plan->limbs)
    {
        cudaError_t error = cudaSetDevice(limb.device);
        if (error != cudaSuccess)
            return fail_const_coeff_readback_submission(plan, generation, set_error(error));
        int status = matrix_wait_limb_stream(
            plan->matrix, limb.limb_id, limb.device, limb.stream, true, true);
        if (status != 0)
            return fail_const_coeff_readback_submission(plan, generation, status);
        auto *destination = reinterpret_cast<uint8_t *>(plan->words) +
            limb.ordinal * plan->coefficient_count * sizeof(uint64_t);
        error = cudaMemcpy2DAsync(
            destination,
            plan->words_per_poly * sizeof(uint64_t),
            limb.source,
            limb.source_pitch,
            static_cast<size_t>(limb.source_width) * plan->coefficient_count,
            plan->polynomial_count,
            cudaMemcpyDeviceToHost,
            limb.stream);
        if (error != cudaSuccess)
            return fail_const_coeff_readback_submission(plan, generation, set_error(error));
        // The copy is queued: this limb now owes a terminal record for this
        // generation before the pinned destination may be released.
        limb.armed_generation = generation;
        error = cudaEventRecord(limb.completion, limb.stream);
        if (error != cudaSuccess)
            return fail_const_coeff_readback_submission(plan, generation, set_error(error));
        limb.recorded_generation = generation;
        status = matrix_track_limb_consumer_readonly(
            plan->matrix, limb.limb_id, limb.device, limb.stream, limb.completion, true);
        if (status != 0)
            return fail_const_coeff_readback_submission(plan, generation, status);
    }
    return 0;
}

extern "C" int gpu_matrix_wait_const_coeff_readback(
    const GpuPreparedConstCoeffReadback *plan)
{
    if (!plan || plan->limbs.empty())
        return set_error("invalid prepared coefficient readback wait");
    std::lock_guard<std::mutex> lock(plan->mutex);
    for (const auto &limb : plan->limbs)
    {
        const cudaError_t error = cudaEventSynchronize(limb.completion);
        if (error != cudaSuccess)
            return set_error(error);
    }
    return 0;
}

extern "C" int gpu_matrix_query_const_coeff_readback(
    const GpuPreparedConstCoeffReadback *plan, int *out_ready)
{
    if (!plan || !out_ready || plan->limbs.empty())
        return set_error("invalid prepared coefficient readback query");
    std::lock_guard<std::mutex> lock(plan->mutex);
    *out_ready = 1;
    int current = 0;
    cudaError_t error = cudaGetDevice(&current);
    if (error != cudaSuccess) return set_error(error);
    for (const auto &limb : plan->limbs)
    {
        // A limb that queued nothing for the newest submission holds a stale
        // event and cannot speak for it, and an armed limb whose terminal event
        // was never recorded is unproven rather than complete. Either way the
        // answer is "not ready", never a completion claim.
        if (limb.armed_generation != plan->submission_generation) continue;
        if (limb.recorded_generation != plan->submission_generation)
        {
            *out_ready = 0;
            continue;
        }
        error = cudaSetDevice(limb.device);
        if (error == cudaSuccess) error = cudaEventQuery(limb.completion);
        if (error == cudaErrorNotReady)
        {
            *out_ready = 0;
            error = cudaSuccess;
        }
        if (error != cudaSuccess) break;
    }
    const cudaError_t restored = cudaSetDevice(current);
    if (error == cudaSuccess) error = restored;
    return error == cudaSuccess ? 0 : set_error(error);
}

extern "C" void gpu_matrix_destroy_const_coeff_readback(
    GpuPreparedConstCoeffReadback *plan)
{
    if (!plan) return;
    delete plan;
}

namespace
{
    // Moves a plan's pinned host allocation to the context-owned reclaimer
    // behind one freshly recorded event per stream the plan is allowed to queue
    // work on. The records are taken after the caller stopped submitting, so
    // they dominate every queued operation, including work that a partial
    // submission left without a terminal event. Ownership only moves when the
    // reclaimer accepts it; a nonzero status leaves the allocation unretired
    // for the caller to leak instead of freeing it early.
    int defer_pinned_free_behind_plan_streams(
        GpuContext *ctx, const std::vector<SerdeStreamRef> &streams, void *pointer)
    {
        if (!ctx || !pointer)
            return set_error("invalid prepared pinned free request");
        if (streams.empty())
            return set_error("prepared plan has no submission stream");
        GpuEventSet *events = nullptr;
        const int status = serde_build_event_set_from_streams(ctx, streams, &events);
        if (status != 0) return status;
        return gpu_event_set_defer_pinned_free(ctx, events, pointer);
    }
}

extern "C" int gpu_matrix_defer_const_coeff_readback_pinned_free(
    const GpuPreparedConstCoeffReadback *plan, void *pointer)
{
    if (!plan || !plan->matrix || !plan->matrix->ctx || !plan->matrix->ctx->execution ||
        !pointer || plan->limbs.empty())
        return set_error("invalid prepared coefficient readback pinned free");
    std::lock_guard<std::mutex> lock(plan->mutex);
    std::vector<SerdeStreamRef> streams;
    try
    {
        streams.reserve(plan->limbs.size());
    }
    catch (const std::exception &error)
    {
        return set_error(error.what());
    }
    for (const auto &limb : plan->limbs)
        serde_append_unique_stream(streams, limb.device, limb.stream);
    return defer_pinned_free_behind_plan_streams(plan->matrix->ctx, streams, pointer);
}

struct GpuPreparedRnsUploadLimb
{
    size_t ordinal;
    dim3 limb_id;
    int device;
    cudaStream_t stream;
    GpuMatrix::SharedLimbBuffer::DeviceDescriptor *destination;
    size_t staging_bytes;
    size_t blocks;
    std::unique_ptr<GpuDeviceWorkspace> workspace;
    std::unique_ptr<GpuCudaResource> completion_resource;
    // Terminal-event generations. `armed` names the submission that may have
    // queued work for this limb, `recorded` names the submission whose
    // completion event was actually recorded on this limb's stream. A reclaim
    // decision is only valid for a generation whose armed limbs are all
    // recorded, so a partially submitted or unrecorded event can never be
    // mistaken for the current one.
    uint64_t armed_generation;
    uint64_t recorded_generation;
};

struct GpuPreparedRnsUpload
{
    GpuMatrix *matrix;
    const uint8_t *host_bytes;
    size_t bytes_per_poly;
    size_t polynomial_count;
    size_t limb_count;
    GpuPolyFormat format;
    bool transform_to_eval;
    GpuMatrixTransformPlan *transform;
    std::vector<GpuPreparedRnsUploadLimb> limbs;
    uint64_t submission_generation;
    // See GpuPreparedConstCoeffReadback::mutex.  The Rust busy bit prevents
    // ordinary overlap, but native query/submit calls still need a data-race
    // boundary when separate callers arrive concurrently.
    mutable std::mutex mutex;
};

extern "C" int gpu_matrix_prepare_rns_upload(
    GpuMatrix *mat,
    const uint8_t *bytes,
    size_t bytes_per_poly,
    int format,
    bool transform_to_eval,
    const GpuPreparedPlanDescriptor *plan,
    GpuPreparedRnsUpload **out_plan)
{
    if (!out_plan)
        return set_error("null prepared RNS upload output");
    *out_plan = nullptr;
    if (!mat || !mat->ctx || !mat->ctx->execution || !bytes || mat->level < 0 ||
        mat->ctx->N <= 0)
        return set_error("invalid prepared RNS upload arguments");
    if (!plan || gpu_prepared_validate_descriptor(plan) != 0)
        return set_error("prepared RNS upload requires a saved descriptor");
    GpuPolyFormat target_format;
    if (!parse_format(format, target_format))
        return set_error("invalid prepared RNS upload format");
    if (transform_to_eval && target_format != GPU_POLY_FORMAT_COEFF)
        return set_error("prepared RNS upload transform requires coefficient input");
    if (transform_to_eval && mat->format != GPU_POLY_FORMAT_EVAL)
        return set_error("prepared RNS upload transform requires evaluation destination");
    if (!transform_to_eval && mat->format != target_format)
        return set_error("prepared RNS upload destination format mismatch");

    size_t polynomial_count = 0;
    if (!serde_checked_mul_size(mat->rows, mat->cols, &polynomial_count) || polynomial_count == 0)
        return set_error("invalid prepared RNS upload matrix size");
    const size_t limb_count = static_cast<size_t>(mat->level) + 1;
    size_t expected_words = 0;
    size_t expected_bytes = 0;
    if (!serde_checked_mul_size(limb_count, static_cast<size_t>(mat->ctx->N), &expected_words) ||
        !serde_checked_mul_size(expected_words, sizeof(uint64_t), &expected_bytes) ||
        bytes_per_poly < expected_bytes)
        return set_error("prepared RNS upload byte span is too small");
    if (mat->ctx->limb_gpu_ids.size() < limb_count)
        return set_error("prepared RNS upload limb metadata is incomplete");

    const size_t host_limb_bytes = static_cast<size_t>(mat->ctx->N) * sizeof(uint64_t);
    size_t total_coefficients = 0;
    size_t staging_bytes = 0;
    size_t host_bytes = 0;
    if (!serde_checked_mul_size(polynomial_count, static_cast<size_t>(mat->ctx->N), &total_coefficients) ||
        !serde_checked_mul_size(total_coefficients, sizeof(uint64_t), &staging_bytes) ||
        !serde_checked_mul_size(polynomial_count, bytes_per_poly, &host_bytes) ||
        host_limb_bytes > bytes_per_poly)
        return set_error("prepared RNS upload host stride is invalid");
    const size_t expected_allocations = 1 + 2 * limb_count + (transform_to_eval ? 1 : 0);
    const size_t expected_streams = limb_count + (transform_to_eval ? 1 : 0);
    const GpuMatrix *owner = gpu_prepared_base_owner(mat);
    if (!owner || owner->ctx != mat->ctx || owner->level != mat->level)
        return set_error("prepared RNS upload owner is invalid");
    const GpuPreparedResourceKey host_key{0, -1, -1, 0, 0, GPU_PREPARED_STAGE_UPLOAD};
    if (plan->allocation_count != expected_allocations || plan->stream_count != expected_streams ||
        gpu_prepared_require_allocation(
            plan, 0, GPU_PREPARED_PINNED_HOST, &host_key, host_bytes, alignof(uint64_t)) != 0)
        return set_error("prepared RNS upload plan shape does not match its owner");

    // Validate every physical claim and stream before constructing the first
    // workspace. This keeps malformed later-limb descriptors side-effect free.
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = mat->ctx->limb_gpu_ids[limb];
        int device = -1;
        cudaStream_t stream = nullptr;
        dim3 planned_limb{};
        GpuPreparedResourceKey key{};
        if (matrix_limb_device(mat, limb_id, &device) != 0 ||
            matrix_limb_stream(mat, limb_id, &stream) != 0 || !stream ||
            gpu_prepared_limb_key(owner->ctx, owner->level, limb,
                GPU_PREPARED_STAGE_UPLOAD, &planned_limb, &key) != 0 ||
            planned_limb.x != limb_id.x || planned_limb.y != limb_id.y || key.device != device ||
            gpu_prepared_require_allocation(plan, 1 + 2 * limb,
                GPU_PREPARED_TRANSFER_WORKSPACE, &key, staging_bytes, alignof(uint64_t)) != 0 ||
            gpu_prepared_require_allocation(plan, 2 + 2 * limb,
                GPU_PREPARED_COMPLETION_EVENT, &key, 0, 1) != 0 ||
            plan->streams[limb].origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
            std::memcmp(&plan->streams[limb].key, &key, sizeof(key)) != 0 ||
            gpu_prepared_require_stream_slot(owner->ctx, limb_id.x, stream,
                plan->streams[limb].pool_slot) != 0)
            return set_error("prepared RNS upload descriptor has an invalid limb claim");
    }
    if (transform_to_eval)
    {
        size_t launch_count = 0;
        if (gpu_prepared_ntt_launch_table(static_cast<uint32_t>(mat->ctx->N), limb_count,
                polynomial_count, 1, nullptr, 0, &launch_count) != 0)
            return set_error("prepared RNS upload transform geometry is invalid");
        size_t geometry_bytes = 0;
        if (!serde_checked_mul_size(launch_count, sizeof(GpuPreparedNttLaunchLayout),
                &geometry_bytes))
            return set_error("prepared RNS upload transform geometry overflow");
        dim3 ntt_limb{};
        GpuPreparedResourceKey ntt_key{};
        if (gpu_prepared_limb_key(owner->ctx, owner->level, 0,
                GPU_PREPARED_STAGE_NTT, &ntt_limb, &ntt_key) != 0 ||
            gpu_prepared_require_allocation(plan, expected_allocations - 1,
                GPU_PREPARED_PLAN_HOST_ONLY, &ntt_key, geometry_bytes, alignof(void *)) != 0 ||
            plan->streams[limb_count].origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
            std::memcmp(&plan->streams[limb_count].key, &ntt_key, sizeof(ntt_key)) != 0)
            return set_error("prepared RNS upload transform descriptor is invalid");
        GpuPreparedPlanDescriptor ntt_layout{};
        ntt_layout.allocation_count = 1;
        ntt_layout.allocations[0] = plan->allocations[expected_allocations - 1];
        ntt_layout.stream_count = 1;
        ntt_layout.streams[0] = plan->streams[limb_count];
        ntt_layout.launch_count = plan->launch_count;
        if (ntt_layout.launch_count > GPU_PREPARED_PLAN_MAX_STREAMS)
            return set_error("prepared RNS upload transform launch table is too large");
        for (size_t index = 0; index < ntt_layout.launch_count; ++index)
            ntt_layout.launches[index] = plan->launches[index];
        GpuMatrixTransformPlan *validated_transform = nullptr;
        if (gpu_matrix_prepare_ntt_plan_with_layout(mat, nullptr, true,
                &ntt_layout, &validated_transform) != 0)
            return set_error("prepared RNS upload transform geometry differs");
        gpu_matrix_destroy_ntt_plan(validated_transform);
    }

    GpuAllocationActivity activity(mat->ctx->execution.get(), -1);
    auto *prepared = new GpuPreparedRnsUpload{
        mat, bytes, bytes_per_poly, polynomial_count, limb_count, target_format,
        transform_to_eval, nullptr, {}, 0, {}};
    try
    {
        prepared->limbs.reserve(limb_count);
        const size_t blocks = (total_coefficients + 255) / 256;
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const dim3 limb_id = mat->ctx->limb_gpu_ids[limb];
            if (limb_id.x >= mat->shared_limb_buffers.size())
                throw std::runtime_error("invalid prepared RNS upload partition");
            auto &buffer = mat->shared_limb_buffers[limb_id.x];
            if (!buffer.device_descriptors || limb_id.y >= buffer.limb_count)
                throw std::runtime_error("missing prepared RNS upload descriptors");
            int device = -1;
            cudaStream_t stream = nullptr;
            int status = matrix_limb_device(mat, limb_id, &device);
            if (status != 0)
                throw std::runtime_error("invalid prepared RNS upload device");
            status = matrix_limb_stream(mat, limb_id, &stream);
            if (status != 0 || !stream)
                throw std::runtime_error("invalid prepared RNS upload stream");

            dim3 planned_limb{};
            GpuPreparedResourceKey key{};
            status = gpu_prepared_limb_key(
                owner->ctx, owner->level, limb, GPU_PREPARED_STAGE_UPLOAD, &planned_limb, &key);
            if (status != 0 || planned_limb.x != limb_id.x || planned_limb.y != limb_id.y ||
                key.device != device)
                throw std::runtime_error("prepared RNS upload owner differs from its saved plan");
            status = gpu_prepared_require_allocation(
                plan, 1 + 2 * limb, GPU_PREPARED_TRANSFER_WORKSPACE, &key, staging_bytes,
                alignof(uint64_t));
            if (status != 0)
                throw std::runtime_error("prepared RNS upload staging claim missing");
            status = gpu_prepared_require_allocation(
                plan, 2 + 2 * limb, GPU_PREPARED_COMPLETION_EVENT, &key, 0, 1);
            if (status != 0)
                throw std::runtime_error("prepared RNS upload completion claim missing");
            const auto &stream_entry = plan->streams[limb];
            if (stream_entry.origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
                std::memcmp(&stream_entry.key, &key, sizeof(key)) != 0 ||
                gpu_prepared_require_stream_slot(
                    owner->ctx, limb_id.x, stream, stream_entry.pool_slot) != 0)
                throw std::runtime_error(
                    "prepared RNS upload stream differs from its saved descriptor");

            auto workspace = std::make_unique<GpuDeviceWorkspace>();
            status = workspace->acquire(
                mat->ctx, device, GPU_PREPARED_TRANSFER_WORKSPACE,
                staging_bytes, alignof(uint64_t), stream);
            if (status != 0)
                throw std::runtime_error("failed to acquire prepared RNS upload workspace");
            auto completion = std::make_unique<GpuCudaResource>();
            status = completion->acquire(mat->ctx, device, GPU_PREPARED_COMPLETION_EVENT);
            if (status != 0)
                throw std::runtime_error("failed to acquire prepared RNS upload completion event");
            prepared->limbs.push_back(GpuPreparedRnsUploadLimb{
                limb,
                limb_id,
                device,
                stream,
                buffer.device_descriptors + limb_id.y,
                staging_bytes,
                blocks,
                std::move(workspace),
                std::move(completion),
                0,
                0,
            });
            if (host_limb_bytes > bytes_per_poly)
                throw std::runtime_error("prepared RNS upload host stride is invalid");
        }
        if (transform_to_eval)
        {
            const GpuMatrixRange range{0, mat->rows, 0, mat->cols};
            GpuPreparedPlanDescriptor ntt_plan{};
            ntt_plan.allocation_count = 1;
            ntt_plan.allocations[0] = plan->allocations[expected_allocations - 1];
            ntt_plan.stream_count = 1;
            ntt_plan.streams[0] = plan->streams[limb_count];
            ntt_plan.launch_count = plan->launch_count;
            if (ntt_plan.launch_count > GPU_PREPARED_PLAN_MAX_STREAMS)
                throw std::runtime_error("prepared RNS upload transform launch table is too large");
            for (size_t index = 0; index < ntt_plan.launch_count; ++index)
                ntt_plan.launches[index] = plan->launches[index];
            const int status = gpu_matrix_prepare_ntt_plan_with_layout(
                mat, &range, true, &ntt_plan, &prepared->transform);
            if (status != 0)
                throw std::runtime_error("failed to prepare RNS upload NTT transform");
            GpuPreparedResourceKey transform_key{};
            dim3 planned_limb{};
            const int key_status = gpu_prepared_limb_key(
                owner->ctx, owner->level, 0, GPU_PREPARED_STAGE_NTT, &planned_limb,
                &transform_key);
            if (key_status != 0)
                throw std::runtime_error("invalid prepared RNS upload transform plan");
            const int geometry_status = gpu_prepared_require_allocation(
                plan, expected_allocations - 1, GPU_PREPARED_PLAN_HOST_ONLY, &transform_key,
                plan->allocations[expected_allocations - 1].bytes, alignof(void *));
            if (geometry_status != 0)
                throw std::runtime_error("prepared RNS upload transform geometry missing");
            const auto &transform_stream = plan->streams[limb_count];
            if (transform_stream.origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
                std::memcmp(&transform_stream.key, &transform_key, sizeof(transform_key)) != 0 ||
                gpu_prepared_require_stream_slot(
                    owner->ctx, planned_limb.x, prepared->transform->launches.front().stream,
                    transform_stream.pool_slot) != 0)
                throw std::runtime_error("prepared RNS upload transform stream is not planned");
        }
    }
    catch (const std::exception &error)
    {
        if (prepared->transform)
            gpu_matrix_destroy_ntt_plan(prepared->transform);
        delete prepared;
        return set_error(error.what());
    }
    *out_plan = prepared;
    rns_upload_prepared_acquisitions.fetch_add(
        2 * limb_count + (transform_to_eval ? 1 : 0), std::memory_order_relaxed);
    return 0;
}

namespace
{
    // Records this plan's terminal completion on every limb stream that already
    // queued work for `generation`. Called after the queued work and on every
    // failure return, so the pinned staging allocation can only be reclaimed
    // through an event that provably covers the queued copies.
    int record_rns_upload_terminal(GpuPreparedRnsUpload *plan, uint64_t generation)
    {
        for (auto &limb : plan->limbs)
        {
            if (limb.armed_generation != generation || limb.recorded_generation == generation)
                continue;
            cudaError_t error = cudaSetDevice(limb.device);
            if (error == cudaSuccess)
                error = cudaEventRecord(limb.completion_resource->event, limb.stream);
            if (error != cudaSuccess) return set_error(error);
            limb.recorded_generation = generation;
        }
        return 0;
    }

    // Reports a failed submission while still recording the terminal completion
    // of every limb that may already have queued work. The submission's own
    // cause is the more specific report; when a terminal record is refused the
    // generation stays unproven, so the pinned staging allocation is retained
    // instead of freed while a queued copy could still read it.
    int fail_rns_upload_submission(
        GpuPreparedRnsUpload *plan, uint64_t generation, int status)
    {
        // The terminal record's own status is deliberately not reported: the
        // submission cause is what the caller must see, and a refused record
        // leaves the generation unproven, which fails closed.
        record_rns_upload_terminal(plan, generation);
        return status;
    }
}

extern "C" int gpu_matrix_submit_rns_upload(GpuPreparedRnsUpload *plan)
{
    if (!plan || !plan->matrix || !plan->host_bytes || plan->limbs.empty())
        return set_error("invalid prepared RNS upload plan");
    std::lock_guard<std::mutex> lock(plan->mutex);
    const size_t host_limb_bytes = static_cast<size_t>(plan->matrix->ctx->N) * sizeof(uint64_t);
    const uint64_t generation = ++plan->submission_generation;
    for (auto &limb : plan->limbs)
    {
        cudaError_t error = cudaSetDevice(limb.device);
        if (error != cudaSuccess)
            return fail_rns_upload_submission(plan, generation, set_error(error));
        int status = matrix_wait_limb_stream(
            plan->matrix, limb.limb_id, limb.device, limb.stream, true, false);
        if (status != 0)
            return fail_rns_upload_submission(plan, generation, status);
        auto *staging = reinterpret_cast<uint64_t *>(limb.workspace->data);
        const auto *source = plan->host_bytes + limb.ordinal * host_limb_bytes;
        error = cudaMemcpy2DAsync(
            staging,
            limb.staging_bytes / plan->polynomial_count,
            source,
            plan->bytes_per_poly,
            host_limb_bytes,
            plan->polynomial_count,
            cudaMemcpyHostToDevice,
            limb.stream);
        if (error != cudaSuccess)
            return fail_rns_upload_submission(plan, generation, set_error(error));
        // The staged copy is queued: this limb now owes a terminal record for
        // this generation before the pinned staging allocation may be released.
        limb.armed_generation = generation;
        serde_pack_u64_limbs_to_packed_kernel<<<
            static_cast<unsigned int>(limb.blocks), 256, 0, limb.stream>>>(
            staging, limb.destination, 1, plan->polynomial_count,
            static_cast<size_t>(plan->matrix->ctx->N));
        error = cudaGetLastError();
        if (error != cudaSuccess)
            return fail_rns_upload_submission(plan, generation, set_error(error));
        status = matrix_record_limb_write(plan->matrix, limb.limb_id, limb.stream, true);
        if (status != 0)
            return fail_rns_upload_submission(plan, generation, status);
    }
    if (plan->transform_to_eval)
    {
        const int status = gpu_matrix_submit_ntt_plan(plan->transform, plan->matrix);
        if (status != 0)
            return fail_rns_upload_submission(plan, generation, status);
    }
    else
    {
        plan->matrix->format = plan->format;
    }
    // The terminal record is taken last so it also covers the optional
    // evaluation transform queued behind the per-limb copies.
    return record_rns_upload_terminal(plan, generation);
}

extern "C" int gpu_matrix_wait_rns_upload(const GpuPreparedRnsUpload *plan)
{
    if (!plan || plan->limbs.empty())
        return set_error("invalid prepared RNS upload wait");
    std::lock_guard<std::mutex> lock(plan->mutex);
    for (const auto &limb : plan->limbs)
    {
        const cudaError_t error = cudaEventSynchronize(limb.completion_resource->event);
        if (error != cudaSuccess)
            return set_error(error);
    }
    return 0;
}

extern "C" int gpu_matrix_query_rns_upload(
    const GpuPreparedRnsUpload *plan, int *out_ready)
{
    if (!plan || !out_ready || plan->limbs.empty())
        return set_error("invalid prepared RNS upload query");
    std::lock_guard<std::mutex> lock(plan->mutex);
    *out_ready = 1;
    int current = 0;
    cudaError_t error = cudaGetDevice(&current);
    if (error != cudaSuccess) return set_error(error);
    for (const auto &limb : plan->limbs)
    {
        // A limb that queued nothing for the newest submission holds a stale
        // event and cannot speak for it, and an armed limb whose terminal event
        // was never recorded is unproven rather than complete. Either way the
        // answer is "not ready", never a completion claim.
        if (limb.armed_generation != plan->submission_generation) continue;
        if (limb.recorded_generation != plan->submission_generation)
        {
            *out_ready = 0;
            continue;
        }
        error = cudaSetDevice(limb.device);
        if (error == cudaSuccess) error = cudaEventQuery(limb.completion_resource->event);
        if (error == cudaErrorNotReady)
        {
            *out_ready = 0;
            error = cudaSuccess;
        }
        if (error != cudaSuccess) break;
    }
    const cudaError_t restored = cudaSetDevice(current);
    if (error == cudaSuccess) error = restored;
    return error == cudaSuccess ? 0 : set_error(error);
}

extern "C" int gpu_matrix_defer_rns_upload_pinned_free(
    const GpuPreparedRnsUpload *plan, void *pointer)
{
    if (!plan || !plan->matrix || !plan->matrix->ctx || !plan->matrix->ctx->execution ||
        !pointer || plan->limbs.empty())
        return set_error("invalid prepared RNS upload pinned free");
    std::lock_guard<std::mutex> lock(plan->mutex);
    std::vector<SerdeStreamRef> streams;
    try
    {
        streams.reserve(plan->limbs.size());
    }
    catch (const std::exception &error)
    {
        return set_error(error.what());
    }
    for (const auto &limb : plan->limbs)
        serde_append_unique_stream(streams, limb.device, limb.stream);
    return defer_pinned_free_behind_plan_streams(plan->matrix->ctx, streams, pointer);
}

extern "C" void gpu_matrix_destroy_rns_upload(GpuPreparedRnsUpload *plan)
{
    if (!plan) return;
    if (plan->transform)
        gpu_matrix_destroy_ntt_plan(plan->transform);
    delete plan;
}

extern "C" int gpu_poly_store_compact_bytes(
    GpuMatrix *poly,
    uint8_t *payload_out,
    size_t payload_capacity,
    uint16_t *out_max_coeff_bits,
    uint16_t *out_bytes_per_coeff,
    size_t *out_payload_len)
{
    if (!poly || !poly->ctx || !poly->ctx->execution)
        return set_error("invalid allocation owner in gpu_poly_store_compact_bytes");
    GpuAllocationActivity activity(poly->ctx->execution.get(), -1);

    if (!poly || !payload_out || !out_max_coeff_bits || !out_bytes_per_coeff || !out_payload_len)
    {
        return set_error("invalid gpu_poly_store_compact_bytes arguments");
    }
    *out_max_coeff_bits = 0;
    *out_bytes_per_coeff = 0;
    *out_payload_len = 0;

    int level = -1;
    size_t n = 0;
    size_t limb_count = 0;
    size_t poly_count = 0;
    if (!serde_get_matrix_shape(poly, &level, &n, &limb_count, &poly_count))
    {
        return set_error("invalid matrix in gpu_poly_store_compact_bytes");
    }
    if (limb_count > poly->ctx->moduli.size())
    {
        return set_error("unexpected modulus count in gpu_poly_store_compact_bytes");
    }
    if (limb_count == 0 || n == 0 || poly_count == 0)
    {
        return 0;
    }
    if (limb_count > static_cast<size_t>(kMaxRnsLimbs))
    {
        return set_error("limb_count exceeds kernel limit in gpu_poly_store_compact_bytes");
    }
    size_t coeff_count = 0;
    if (!serde_checked_mul_size(poly_count, n, &coeff_count))
    {
        return set_error("coeff_count overflow in gpu_poly_store_compact_bytes");
    }

    if (poly->format == GPU_POLY_FORMAT_EVAL)
    {
        const int status = gpu_matrix_intt_all(poly);
        if (status != 0)
        {
            return status;
        }
    }
    if (poly->format != GPU_POLY_FORMAT_COEFF)
    {
        return set_error("gpu_poly_store_compact_bytes expects coeff format");
    }

    size_t total_bits_upper = 0;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        total_bits_upper += static_cast<size_t>(bit_width_u64(poly->ctx->moduli[limb]));
    }
    const size_t words_per_coeff =
        std::max<size_t>(
            1,
            (total_bits_upper + static_cast<size_t>(63)) / static_cast<size_t>(64));
    if (words_per_coeff > static_cast<size_t>(kMaxCoeffWords))
    {
        return set_error("words_per_coeff exceeds kernel limit in gpu_poly_store_compact_bytes");
    }

    auto &limb_map = poly->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_poly_store_compact_bytes");
    }

    std::vector<const uint8_t *> limb_ptrs(limb_count);
    std::vector<size_t> limb_strides(limb_count);
    std::vector<uint8_t> limb_coeff_bytes(limb_count);
    int common_device = -1;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        const uint8_t *limb_ptr = matrix_limb_ptr_by_id(poly, 0, limb_id);
        if (!limb_ptr)
        {
            return set_error("null matrix limb pointer in gpu_poly_store_compact_bytes");
        }
        int device = -1;
        int status = matrix_limb_device(poly, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        size_t limb_stride = 0;
        uint8_t limb_bytes = 0;
        if (!matrix_limb_metadata_by_id(poly, limb_id, &limb_stride, &limb_bytes))
        {
            return set_error("invalid limb metadata in gpu_poly_store_compact_bytes");
        }
        if (limb_bytes == 0 || limb_stride < n * static_cast<size_t>(limb_bytes))
        {
            return set_error("invalid limb stride in gpu_poly_store_compact_bytes");
        }
        if (common_device < 0)
        {
            common_device = device;
        }
        else if (common_device != device)
        {
            return set_error("gpu_poly_store_compact_bytes requires all limbs on a single GPU");
        }
        limb_ptrs[limb] = limb_ptr;
        limb_strides[limb] = limb_stride;
        limb_coeff_bytes[limb] = limb_bytes;
    }

    const size_t inverse_stride = poly->ctx->moduli.size();
    const std::vector<uint64_t> &inverse_table = poly->ctx->garner_inverse_table;
    if (inverse_table.size() != inverse_stride * inverse_stride)
    {
        return set_error("invalid cached inverse table in gpu_poly_store_compact_bytes");
    }
    std::vector<uint64_t> moduli_subset(poly->ctx->moduli.begin(), poly->ctx->moduli.begin() + limb_count);
    std::vector<uint64_t> modulus_words_host;
    if (!serde_compute_modulus_words_le(moduli_subset, &modulus_words_host))
    {
        return set_error("failed to compute modulus words in gpu_poly_store_compact_bytes");
    }
    if (modulus_words_host.size() > words_per_coeff)
    {
        return set_error("modulus words exceed kernel limit in gpu_poly_store_compact_bytes");
    }
    std::vector<uint64_t> half_modulus_words_host = modulus_words_host;
    serde_shift_words_right_one_le(&half_modulus_words_host);
    modulus_words_host.resize(words_per_coeff, 0);
    half_modulus_words_host.resize(words_per_coeff, 0);

    if (words_per_coeff > std::numeric_limits<size_t>::max() / coeff_count)
    {
        return set_error("coeff word length overflow in gpu_poly_store_compact_bytes");
    }
    const size_t coeff_word_len = coeff_count * words_per_coeff;

    cudaError_t err = cudaSetDevice(common_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    const uint8_t **d_limb_ptrs = nullptr;
    size_t *d_limb_strides = nullptr;
    uint8_t *d_limb_coeff_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    uint64_t *d_modulus_words = nullptr;
    uint64_t *d_half_modulus_words = nullptr;
    uint64_t *d_garner_inv = nullptr;
    uint64_t *d_coeff_words = nullptr;
    uint8_t *d_sign_bits = nullptr;
    int *d_overflow = nullptr;
    unsigned int *d_max_abs_bits = nullptr;
    uint8_t *d_payload = nullptr;
    cudaStream_t work_stream = nullptr;
    GpuCudaResource private_stream;
    CompactWorkspace workspace;
    auto release = [&]() {
        const int released = workspace.storage.release();
        const int finished = serde_finish_private_stream(poly, common_device, private_stream, work_stream);
        return released != 0 ? released : finished;
    };

    const int stream_status = serde_begin_private_stream(poly, common_device, private_stream, &work_stream);
    if (stream_status != 0)
    {
        release();
        return stream_status;
    }
    const int workspace_status = workspace.acquire(poly->ctx, common_device, level,
        poly->rows, poly->cols, 1, 0, 0, work_stream);
    if (workspace_status != 0) { release(); return workspace_status; }

    err = workspace.span(0, &d_modulus_words, words_per_coeff * sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_modulus_words,
        modulus_words_host.data(),
        words_per_coeff * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = workspace.span(1, &d_half_modulus_words, words_per_coeff * sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_half_modulus_words,
        half_modulus_words_host.data(),
        words_per_coeff * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const int wait_status = matrix_wait_limb_stream(
            poly,
            limb_map[limb],
            common_device,
            work_stream, false, true);
        if (wait_status != 0)
        {
            release();
            return wait_status;
        }
    }

    err = workspace.span(2, &d_limb_ptrs, limb_count * sizeof(const uint8_t *));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_ptrs,
        limb_ptrs.data(),
        limb_count * sizeof(const uint8_t *),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    err = workspace.span(3, &d_limb_strides, limb_count * sizeof(size_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_strides,
        limb_strides.data(),
        limb_count * sizeof(size_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = workspace.span(4, &d_limb_coeff_bytes, limb_count * sizeof(uint8_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_coeff_bytes,
        limb_coeff_bytes.data(),
        limb_count * sizeof(uint8_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    err = workspace.span(5, &d_moduli, moduli_subset.size() * sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_moduli,
        moduli_subset.data(),
        moduli_subset.size() * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    err = workspace.span(6, &d_garner_inv, inverse_table.size() * sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_garner_inv,
        inverse_table.data(),
        inverse_table.size() * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    err = workspace.span(7, &d_coeff_words, coeff_word_len * sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = workspace.span(8, &d_overflow, sizeof(int));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemsetAsync(d_overflow, 0, sizeof(int), work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = workspace.span(9, &d_sign_bits, coeff_count * sizeof(uint8_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = workspace.span(10, &d_max_abs_bits, sizeof(unsigned int));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemsetAsync(d_max_abs_bits, 0, sizeof(unsigned int), work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    const int threads = 256;
    const int blocks =
        static_cast<int>((coeff_count + static_cast<size_t>(threads) - 1) /
                         static_cast<size_t>(threads));
    serde_reconstruct_rns_to_words_kernel<<<blocks, threads, 0, work_stream>>>(
        d_limb_ptrs,
        d_limb_strides,
        d_limb_coeff_bytes,
        d_moduli,
        d_garner_inv,
        static_cast<int>(inverse_stride),
        static_cast<int>(limb_count),
        coeff_count,
        n,
        static_cast<int>(words_per_coeff),
        d_coeff_words,
        d_overflow);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    serde_center_coeff_words_kernel<<<blocks, threads, 0, work_stream>>>(
        d_coeff_words,
        coeff_count,
        static_cast<int>(words_per_coeff),
        d_modulus_words,
        d_half_modulus_words,
        d_sign_bits,
        d_max_abs_bits);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    int h_overflow = 0;
    unsigned int h_max_bits = 0;
    err = cudaMemcpyAsync(
        &h_overflow,
        d_overflow,
        sizeof(int),
        cudaMemcpyDeviceToHost,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        &h_max_bits,
        d_max_abs_bits,
        sizeof(unsigned int),
        cudaMemcpyDeviceToHost,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaStreamSynchronize(work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    if (h_overflow != 0)
    {
        release();
        return set_error("overflow in gpu_poly_store_compact_bytes");
    }
    if (h_max_bits > static_cast<unsigned int>(std::numeric_limits<uint16_t>::max()))
    {
        release();
        return set_error("max coeff bits exceed u16 range in gpu_poly_store_compact_bytes");
    }
    const unsigned int h_signed_bits =
        h_max_bits == 0 ? 0 : (h_max_bits + static_cast<unsigned int>(1));
    if (h_signed_bits > static_cast<unsigned int>(std::numeric_limits<uint16_t>::max()))
    {
        release();
        return set_error("centered max coeff bits exceed u16 range in gpu_poly_store_compact_bytes");
    }
    const unsigned int h_bytes_per_coeff =
        (h_signed_bits + static_cast<unsigned int>(7)) / static_cast<unsigned int>(8);
    if (h_bytes_per_coeff > static_cast<unsigned int>(std::numeric_limits<uint16_t>::max()))
    {
        release();
        return set_error("bytes_per_coeff exceed u16 range in gpu_poly_store_compact_bytes");
    }

    size_t payload_len = 0;
    if (!serde_compute_payload_len(coeff_count, h_signed_bits, &payload_len))
    {
        release();
        return set_error("payload length overflow in gpu_poly_store_compact_bytes");
    }
    if (payload_len > payload_capacity)
    {
        release();
        return set_error("payload_capacity too small in gpu_poly_store_compact_bytes");
    }

    if (payload_len > 0)
    {
        err =
            workspace.span(11, &d_payload, payload_len);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
        const int payload_blocks =
            static_cast<int>((payload_len + static_cast<size_t>(threads) - 1) /
                             static_cast<size_t>(threads));
        serde_pack_centered_coeff_words_bits_kernel<<<payload_blocks, threads, 0, work_stream>>>(
            d_coeff_words,
            d_sign_bits,
            coeff_count,
            static_cast<int>(words_per_coeff),
            h_signed_bits,
            d_payload,
            payload_len);
        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            payload_out,
            d_payload,
            payload_len,
            cudaMemcpyDeviceToHost,
            work_stream);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
    }

    err = cudaStreamSynchronize(work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    *out_max_coeff_bits = static_cast<uint16_t>(h_signed_bits);
    *out_bytes_per_coeff = static_cast<uint16_t>(h_bytes_per_coeff);
    *out_payload_len = payload_len;
    return release();
}

extern "C" int gpu_poly_load_compact_bytes(
    GpuMatrix *poly,
    const uint8_t *payload,
    size_t payload_len,
    uint16_t max_coeff_bits)
{
    if (!poly || !poly->ctx || !poly->ctx->execution)
        return set_error("invalid allocation owner in gpu_poly_load_compact_bytes");
    GpuAllocationActivity activity(poly->ctx->execution.get(), -1);

    if (!poly || !poly->ctx)
    {
        return set_error("invalid gpu_poly_load_compact_bytes arguments");
    }

    int level = -1;
    size_t n = 0;
    size_t limb_count = 0;
    size_t poly_count = 0;
    if (!serde_get_matrix_shape(poly, &level, &n, &limb_count, &poly_count))
    {
        return set_error("invalid matrix in gpu_poly_load_compact_bytes");
    }
    if (limb_count > poly->ctx->moduli.size())
    {
        return set_error("unexpected modulus count in gpu_poly_load_compact_bytes");
    }
    if (limb_count == 0 || n == 0 || poly_count == 0)
    {
        poly->format = GPU_POLY_FORMAT_COEFF;
        return 0;
    }
    if (limb_count > static_cast<size_t>(kMaxRnsLimbs))
    {
        return set_error("limb_count exceeds kernel limit in gpu_poly_load_compact_bytes");
    }

    size_t coeff_count = 0;
    if (!serde_checked_mul_size(poly_count, n, &coeff_count))
    {
        return set_error("coeff_count overflow in gpu_poly_load_compact_bytes");
    }

    if (max_coeff_bits == 0)
    {
        if (payload_len != 0)
        {
            return set_error("payload_len must be zero when max_coeff_bits is zero");
        }
    }
    else if (!payload)
    {
        return set_error("null payload in gpu_poly_load_compact_bytes");
    }

    size_t expected_payload_len = 0;
    if (!serde_compute_payload_len(
            coeff_count,
            static_cast<uint32_t>(max_coeff_bits),
            &expected_payload_len))
    {
        return set_error("payload length overflow in gpu_poly_load_compact_bytes");
    }
    if (payload_len != expected_payload_len)
    {
        return set_error("payload length mismatch in gpu_poly_load_compact_bytes");
    }

    auto &limb_map = poly->ctx->limb_gpu_ids;
    if (limb_map.size() < limb_count)
    {
        return set_error("unexpected limb mapping size in gpu_poly_load_compact_bytes");
    }

    std::vector<uint8_t *> limb_ptrs(limb_count);
    std::vector<size_t> limb_strides(limb_count);
    std::vector<uint8_t> limb_coeff_bytes(limb_count);
    int common_device = -1;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 limb_id = limb_map[limb];
        uint8_t *limb_ptr = matrix_limb_ptr_by_id(poly, 0, limb_id);
        if (!limb_ptr)
        {
            return set_error("null matrix limb pointer in gpu_poly_load_compact_bytes");
        }
        int device = -1;
        int status = matrix_limb_device(poly, limb_id, &device);
        if (status != 0)
        {
            return status;
        }
        size_t limb_stride = 0;
        uint8_t limb_bytes = 0;
        if (!matrix_limb_metadata_by_id(poly, limb_id, &limb_stride, &limb_bytes))
        {
            return set_error("invalid limb metadata in gpu_poly_load_compact_bytes");
        }
        if (limb_bytes == 0 || limb_stride < n * static_cast<size_t>(limb_bytes))
        {
            return set_error("invalid limb stride in gpu_poly_load_compact_bytes");
        }
        if (common_device < 0)
        {
            common_device = device;
        }
        else if (common_device != device)
        {
            return set_error("gpu_poly_load_compact_bytes requires all limbs on a single GPU");
        }
        limb_ptrs[limb] = limb_ptr;
        limb_strides[limb] = limb_stride;
        limb_coeff_bytes[limb] = limb_bytes;
    }

    cudaError_t err = cudaSetDevice(common_device);
    if (err != cudaSuccess)
    {
        return set_error(err);
    }

    uint8_t *d_payload = nullptr;
    uint8_t **d_limb_ptrs = nullptr;
    size_t *d_limb_strides = nullptr;
    uint8_t *d_limb_coeff_bytes = nullptr;
    uint64_t *d_moduli = nullptr;
    cudaStream_t work_stream = nullptr;
    GpuCudaResource private_stream;
    CompactWorkspace workspace;
    auto release = [&]() {
        const int released = workspace.storage.release();
        const int finished = serde_finish_private_stream(poly, common_device, private_stream, work_stream);
        return released != 0 ? released : finished;
    };

    const int stream_status = serde_begin_private_stream(poly, common_device, private_stream, &work_stream);
    if (stream_status != 0)
    {
        release();
        return stream_status;
    }
    const int workspace_status = workspace.acquire(poly->ctx, common_device, level,
        poly->rows, poly->cols, 1, 2, max_coeff_bits, work_stream);
    if (workspace_status != 0) { release(); return workspace_status; }

    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const int wait_status = matrix_wait_limb_stream(
            poly,
            limb_map[limb],
            common_device,
            work_stream);
        if (wait_status != 0)
        {
            release();
            return wait_status;
        }
    }

    if (payload_len > 0)
    {
        err = workspace.span(0, &d_payload, payload_len);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
        err = cudaMemcpyAsync(
            d_payload,
            payload,
            payload_len,
            cudaMemcpyHostToDevice,
            work_stream);
        if (err != cudaSuccess)
        {
            release();
            return set_error(err);
        }
    }

    err = workspace.span(1, &d_limb_ptrs, limb_count * sizeof(uint8_t *));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_ptrs,
        limb_ptrs.data(),
        limb_count * sizeof(uint8_t *),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    err = workspace.span(2, &d_limb_strides, limb_count * sizeof(size_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_strides,
        limb_strides.data(),
        limb_count * sizeof(size_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = workspace.span(3, &d_limb_coeff_bytes, limb_count * sizeof(uint8_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_limb_coeff_bytes,
        limb_coeff_bytes.data(),
        limb_count * sizeof(uint8_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    std::vector<uint64_t> moduli_subset(poly->ctx->moduli.begin(), poly->ctx->moduli.begin() + limb_count);
    err = workspace.span(4, &d_moduli, moduli_subset.size() * sizeof(uint64_t));
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    err = cudaMemcpyAsync(
        d_moduli,
        moduli_subset.data(),
        moduli_subset.size() * sizeof(uint64_t),
        cudaMemcpyHostToDevice,
        work_stream);
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }

    const int threads = 256;
    const int blocks =
        static_cast<int>((coeff_count + static_cast<size_t>(threads) - 1) /
                         static_cast<size_t>(threads));
    serde_unpack_packed_coeffs_mod_kernel<<<blocks, threads, 0, work_stream>>>(
        d_payload,
        coeff_count,
        n,
        static_cast<uint32_t>(max_coeff_bits),
        d_moduli,
        static_cast<int>(limb_count),
        d_limb_ptrs,
        d_limb_strides,
        d_limb_coeff_bytes);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        release();
        return set_error(err);
    }
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const int status = matrix_record_limb_write(poly, limb_map[limb], work_stream);
        if (status != 0)
        {
            release();
            return status;
        }
    }

    const int released = release();
    if (released != 0) return released;
    poly->format = GPU_POLY_FORMAT_COEFF;
    return 0;
}

extern "C" int gpu_matrix_store_compact_bytes(
    GpuMatrix *mat,
    uint8_t *payload_out,
    size_t payload_capacity,
    uint16_t *out_max_coeff_bits,
    uint16_t *out_bytes_per_coeff,
    size_t *out_payload_len)
{
    return gpu_poly_store_compact_bytes(
        mat,
        payload_out,
        payload_capacity,
        out_max_coeff_bits,
        out_bytes_per_coeff,
        out_payload_len);
}

extern "C" int gpu_matrix_load_compact_bytes(
    GpuMatrix *mat,
    const uint8_t *payload,
    size_t payload_len,
    uint16_t max_coeff_bits)
{
    return gpu_poly_load_compact_bytes(mat, payload, payload_len, max_coeff_bits);
}

// Prepared compact replay.  Unlike gpu_poly_load_compact_bytes this object
// owns its workspace, metadata and terminal event for its entire lifetime;
// submit only changes the bytes and codec width in the already pinned input
// slot.  In particular, no temporary vectors, private streams, or CUDA
// allocations are permitted on the submission path.
static std::atomic<size_t> compact_upload_legacy_prepare_calls{0};
static std::atomic<size_t> compact_upload_prepared_acquisitions{0};
struct GpuPreparedCompactUpload
{
    GpuMatrix *matrix;
    const uint8_t *host_payload;
    size_t payload_capacity;
    size_t coeff_count;
    size_t n;
    size_t limb_count;
    int device;
    cudaStream_t stream;
    CompactWorkspace workspace;
    std::unique_ptr<GpuCudaResource> completion;
    std::unique_ptr<GpuMatrixTransformPlan> transform;
    mutable std::mutex mutex;
    uint64_t generation;
    bool armed;
};

extern "C" int gpu_matrix_prepare_compact_upload(
    GpuMatrix *mat,
    const uint8_t *payload,
    size_t payload_capacity,
    uint16_t max_coeff_bits,
    const GpuPreparedPlanDescriptor *plan,
    GpuPreparedCompactUpload **out_plan)
{
    if (!out_plan) return set_error("null prepared compact upload output");
    *out_plan = nullptr;
    if (!mat || !mat->ctx || !mat->ctx->execution || !payload || mat->level < 0 ||
        mat->ctx->N <= 0 || payload_capacity == 0)
        return set_error("invalid prepared compact upload arguments");
    if (!plan || gpu_prepared_validate_descriptor(plan) != 0)
        return set_error("prepared compact upload requires a saved descriptor");
    const size_t limb_count = static_cast<size_t>(mat->level) + 1;
    size_t poly_count = 0, coeff_count = 0;
    if (!serde_checked_mul_size(mat->rows, mat->cols, &poly_count) ||
        !serde_checked_mul_size(poly_count, static_cast<size_t>(mat->ctx->N), &coeff_count) ||
        poly_count == 0 || limb_count > mat->ctx->limb_gpu_ids.size())
        return set_error("invalid prepared compact upload dimensions");
    uint32_t max_bits = 0;
    for (size_t limb = 0; limb < limb_count; ++limb)
        max_bits += bit_width_u64(mat->ctx->moduli[limb]);
    if (max_bits == 0 || max_bits > std::numeric_limits<uint16_t>::max())
        return set_error("prepared compact upload codec width is invalid");
    if (max_coeff_bits == 0 || max_coeff_bits > max_bits)
        return set_error("prepared compact upload codec width exceeds destination");
    size_t expected = 0;
    if (!serde_compute_payload_len(coeff_count, max_coeff_bits, &expected) ||
        payload_capacity < expected)
        return set_error("prepared compact upload staging is too small");

    const size_t expected_allocations = mat->format == GPU_POLY_FORMAT_EVAL ? 4 : 3;
    const size_t expected_streams = mat->format == GPU_POLY_FORMAT_EVAL ? 2 : 1;
    if (plan->allocation_count != expected_allocations || plan->stream_count != expected_streams)
        return set_error("prepared compact upload descriptor counts mismatch");
    const GpuPreparedResourceKey host_key{0, -1, -1, 0, 0, GPU_PREPARED_STAGE_UPLOAD};
    if (gpu_prepared_require_allocation(plan, 0, GPU_PREPARED_PINNED_HOST, &host_key,
            payload_capacity, 1) != 0)
        return set_error("prepared compact upload pinned staging claim differs from descriptor");

    int common_device = -1;
    cudaStream_t stream = nullptr;
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        int device = -1;
        cudaStream_t limb_stream = nullptr;
        if (matrix_limb_device(mat, mat->ctx->limb_gpu_ids[limb], &device) != 0 ||
            matrix_limb_stream(mat, mat->ctx->limb_gpu_ids[limb], &limb_stream) != 0 || !limb_stream)
            return set_error("prepared compact upload limb metadata is invalid");
        if (common_device < 0) common_device = device;
        if (common_device != device)
            return set_error("prepared compact upload requires one GPU");
        if (limb == 0) stream = limb_stream;
    }
    auto prepared = std::make_unique<GpuPreparedCompactUpload>();
    prepared->matrix = mat;
    prepared->host_payload = payload;
    prepared->payload_capacity = payload_capacity;
    prepared->coeff_count = coeff_count;
    prepared->n = static_cast<size_t>(mat->ctx->N);
    prepared->limb_count = limb_count;
    prepared->device = common_device;
    prepared->stream = stream;
    prepared->generation = 0;
    prepared->armed = false;
    dim3 upload_limb{};
    GpuPreparedResourceKey upload_key{};
    if (gpu_prepared_limb_key(mat->ctx, mat->level, 0, GPU_PREPARED_STAGE_UPLOAD,
            &upload_limb, &upload_key) != 0 || upload_limb.x != mat->ctx->limb_gpu_ids[0].x ||
        upload_limb.y != mat->ctx->limb_gpu_ids[0].y)
        return set_error("prepared compact upload owner key is invalid");
    GpuPreparedWorkspaceLayout workspace_layout{};
    if (gpu_matrix_query_compact_workspace(mat->ctx, mat->level, mat->rows, mat->cols,
            1, 2, static_cast<uint16_t>(max_bits), &workspace_layout) != 0 ||
        gpu_prepared_require_allocation(plan, 1, workspace_layout.kind, &upload_key,
            workspace_layout.bytes, workspace_layout.alignment) != 0 ||
        gpu_prepared_require_allocation(plan, 2, GPU_PREPARED_COMPLETION_EVENT, &upload_key,
            0, 1) != 0)
        return set_error("prepared compact upload resource claims differ from descriptor");
    if (plan->streams[0].origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
        std::memcmp(&plan->streams[0].key, &upload_key, sizeof(upload_key)) != 0 ||
        gpu_prepared_require_stream_slot(mat->ctx, upload_limb.x, stream,
            plan->streams[0].pool_slot) != 0)
        return set_error("prepared compact upload stream differs from descriptor");
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        size_t stride = 0;
        uint8_t width = 0;
        const dim3 id = mat->ctx->limb_gpu_ids[limb];
        if (!matrix_limb_ptr_by_id(mat, 0, id) ||
            !matrix_limb_metadata_by_id(mat, id, &stride, &width) || width == 0)
            return set_error("prepared compact upload destination metadata is invalid");
    }
    if (mat->format == GPU_POLY_FORMAT_EVAL)
    {
        size_t launch_count = 0, geometry_bytes = 0;
        if (gpu_prepared_ntt_launch_table(static_cast<uint32_t>(mat->ctx->N), limb_count,
                poly_count, 1, nullptr, 0, &launch_count) != 0 ||
            !serde_checked_mul_size(launch_count, sizeof(GpuPreparedNttLaunchLayout),
                &geometry_bytes))
            return set_error("prepared compact upload NTT geometry is invalid");
        dim3 ntt_limb{};
        GpuPreparedResourceKey ntt_key{};
        if (gpu_prepared_limb_key(mat->ctx, mat->level, 0,
                GPU_PREPARED_STAGE_NTT, &ntt_limb, &ntt_key) != 0 ||
            gpu_prepared_require_allocation(plan, 3, GPU_PREPARED_PLAN_HOST_ONLY,
                &ntt_key, geometry_bytes, alignof(void *)) != 0 ||
            plan->streams[1].origin != GPU_PREPARED_STREAM_CONTEXT_REUSED ||
            std::memcmp(&plan->streams[1].key, &ntt_key, sizeof(ntt_key)) != 0 ||
            gpu_prepared_require_stream_slot(mat->ctx, ntt_limb.x, stream,
                plan->streams[1].pool_slot) != 0)
            return set_error("prepared compact upload NTT descriptor differs");
    }
    const int workspace_status = prepared->workspace.acquire(
        mat->ctx, common_device, mat->level, mat->rows, mat->cols, 1, 2,
        static_cast<uint16_t>(max_bits), stream);
    if (workspace_status != 0) return workspace_status;
    prepared->completion = std::make_unique<GpuCudaResource>();
    if (prepared->completion->acquire(mat->ctx, common_device,
                                      GPU_PREPARED_COMPLETION_EVENT) != 0)
        return set_error("prepared compact upload completion event allocation failed");
    // The metadata arrays are immutable and copied once while the plan is
    // prepared.  This is the only host-side vector construction in the plan.
    cudaError_t err = cudaSetDevice(common_device);
    if (err != cudaSuccess) return set_error(err);
    uint8_t *d_ptrs = nullptr, *d_bytes = nullptr;
    size_t *d_strides = nullptr;
    uint64_t *d_moduli = nullptr;
    if (prepared->workspace.span(1, &d_ptrs, limb_count * sizeof(uint8_t *)) != cudaSuccess ||
        prepared->workspace.span(2, &d_strides, limb_count * sizeof(size_t)) != cudaSuccess ||
        prepared->workspace.span(3, &d_bytes, limb_count * sizeof(uint8_t)) != cudaSuccess ||
        prepared->workspace.span(4, &d_moduli, limb_count * sizeof(uint64_t)) != cudaSuccess)
        return set_error("prepared compact upload metadata workspace is invalid");
    std::vector<uint8_t *> ptrs(limb_count);
    std::vector<size_t> strides(limb_count);
    std::vector<uint8_t> widths(limb_count);
    for (size_t limb = 0; limb < limb_count; ++limb)
    {
        const dim3 id = mat->ctx->limb_gpu_ids[limb];
        ptrs[limb] = matrix_limb_ptr_by_id(mat, 0, id);
        if (!ptrs[limb] || !matrix_limb_metadata_by_id(mat, id, &strides[limb], &widths[limb]) ||
            widths[limb] == 0)
            return set_error("prepared compact upload destination metadata is invalid");
    }
    err = cudaMemcpyAsync(d_ptrs, ptrs.data(), limb_count * sizeof(uint8_t *),
                          cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(d_strides, strides.data(),
        limb_count * sizeof(size_t), cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(d_bytes, widths.data(),
        limb_count * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
    if (err == cudaSuccess) err = cudaMemcpyAsync(d_moduli, mat->ctx->moduli.data(),
        limb_count * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) return set_error(err);
    if (mat->format == GPU_POLY_FORMAT_EVAL)
    {
        GpuMatrixRange range{0, mat->rows, 0, mat->cols};
        GpuPreparedPlanDescriptor ntt_plan{};
        ntt_plan.allocation_count = 1;
        ntt_plan.allocations[0] = plan->allocations[3];
        ntt_plan.stream_count = 1;
        ntt_plan.streams[0] = plan->streams[1];
        ntt_plan.launch_count = plan->launch_count;
        for (size_t index = 0; index < plan->launch_count; ++index)
            ntt_plan.launches[index] = plan->launches[index];
        GpuMatrixTransformPlan *transform = nullptr;
        if (gpu_matrix_prepare_ntt_plan_with_layout(mat, &range, true, &ntt_plan, &transform) != 0 ||
            !transform)
            return set_error("prepared compact upload evaluation transform failed");
        prepared->transform.reset(transform);
    }
    *out_plan = prepared.release();
    compact_upload_prepared_acquisitions.fetch_add(
        expected_allocations - 1, std::memory_order_relaxed);
    return 0;
}

extern "C" void gpu_matrix_test_reset_compact_upload_bind_counters()
{
    compact_upload_legacy_prepare_calls.store(0, std::memory_order_relaxed);
    compact_upload_prepared_acquisitions.store(0, std::memory_order_relaxed);
}

extern "C" size_t gpu_matrix_test_legacy_compact_upload_prepare_calls()
{
    return compact_upload_legacy_prepare_calls.load(std::memory_order_relaxed);
}

extern "C" size_t gpu_matrix_test_compact_upload_prepared_acquisitions()
{
    return compact_upload_prepared_acquisitions.load(std::memory_order_relaxed);
}

extern "C" int gpu_matrix_submit_compact_upload(
    GpuPreparedCompactUpload *plan, uint16_t max_coeff_bits, size_t payload_len)
{
    if (!plan || !plan->matrix || !plan->host_payload) return set_error("invalid prepared compact upload");
    std::lock_guard<std::mutex> lock(plan->mutex);
    if (plan->armed) return set_error("prepared compact upload has unretired work");
    size_t expected = 0;
    if (max_coeff_bits == 0 || !serde_compute_payload_len(plan->coeff_count,
            max_coeff_bits, &expected) || payload_len != expected || payload_len > plan->payload_capacity)
        return set_error("prepared compact upload payload length mismatch");
    cudaError_t err = cudaSetDevice(plan->device);
    if (err == cudaSuccess)
        for (size_t limb = 0; limb < plan->limb_count; ++limb)
            if (matrix_wait_limb_stream(plan->matrix,
                    plan->matrix->ctx->limb_gpu_ids[limb], plan->device, plan->stream) != 0)
                return 1;
    if (err == cudaSuccess) err = cudaMemcpyAsync(plan->workspace.storage.data,
        plan->host_payload, payload_len, cudaMemcpyHostToDevice, plan->stream);
    if (err == cudaSuccess)
    {
        uint8_t *d_payload = nullptr, **d_ptrs = nullptr, *d_bytes = nullptr;
        size_t *d_strides = nullptr;
        uint64_t *d_moduli = nullptr;
        if (plan->workspace.span(0, &d_payload, payload_len) != cudaSuccess ||
            plan->workspace.span(1, &d_ptrs, plan->limb_count * sizeof(uint8_t *)) != cudaSuccess ||
            plan->workspace.span(2, &d_strides, plan->limb_count * sizeof(size_t)) != cudaSuccess ||
            plan->workspace.span(3, &d_bytes, plan->limb_count * sizeof(uint8_t)) != cudaSuccess ||
            plan->workspace.span(4, &d_moduli, plan->limb_count * sizeof(uint64_t)) != cudaSuccess)
            return set_error("prepared compact upload workspace span failed");
        const int blocks = static_cast<int>((plan->coeff_count + 255) / 256);
        serde_unpack_packed_coeffs_mod_kernel<<<blocks, 256, 0, plan->stream>>>(
            d_payload, plan->coeff_count, plan->n, max_coeff_bits, d_moduli,
            static_cast<int>(plan->limb_count), d_ptrs, d_strides, d_bytes);
        err = cudaGetLastError();
        if (err == cudaSuccess)
            for (size_t limb = 0; limb < plan->limb_count; ++limb)
                if (matrix_record_limb_write(plan->matrix, plan->matrix->ctx->limb_gpu_ids[limb], plan->stream) != 0)
                    return 1;
        if (err == cudaSuccess && plan->transform)
        {
            const int transform_status = gpu_matrix_submit_ntt_plan(
                plan->transform.get(), plan->matrix);
            if (transform_status != 0) return transform_status;
        }
    }
    if (err != cudaSuccess) return set_error(err);
    ++plan->generation;
    plan->armed = true;
    err = cudaEventRecord(plan->completion->event, plan->stream);
    if (err != cudaSuccess) return set_error(err);
    plan->matrix->format = plan->transform ? GPU_POLY_FORMAT_EVAL : GPU_POLY_FORMAT_COEFF;
    return 0;
}

extern "C" int gpu_matrix_query_compact_upload(
    const GpuPreparedCompactUpload *plan, int *out_ready)
{
    if (!plan || !out_ready) return set_error("invalid prepared compact upload query");
    std::lock_guard<std::mutex> lock(plan->mutex);
    *out_ready = 0;
    if (!plan->armed) return 0;
    const cudaError_t status = cudaEventQuery(plan->completion->event);
    if (status == cudaSuccess) { *out_ready = 1; const_cast<GpuPreparedCompactUpload *>(plan)->armed = false; return 0; }
    if (status == cudaErrorNotReady) return 0;
    return set_error(status);
}

extern "C" int gpu_matrix_wait_compact_upload(const GpuPreparedCompactUpload *plan)
{
    if (!plan) return set_error("invalid prepared compact upload wait");
    std::lock_guard<std::mutex> lock(plan->mutex);
    if (!plan->armed) return 0;
    const cudaError_t status = cudaEventSynchronize(plan->completion->event);
    if (status != cudaSuccess) return set_error(status);
    const_cast<GpuPreparedCompactUpload *>(plan)->armed = false;
    return 0;
}

extern "C" int gpu_matrix_defer_compact_upload_pinned_free(
    const GpuPreparedCompactUpload *plan, void *pointer)
{
    if (!plan || !plan->matrix || !plan->matrix->ctx ||
        !plan->matrix->ctx->execution || !pointer || !plan->stream)
        return set_error("invalid prepared compact upload pinned free");
    return gpu_defer_pinned_frees(plan->matrix->ctx, plan->device,
        plan->stream, &pointer, 1);
}

extern "C" void gpu_matrix_destroy_compact_upload(GpuPreparedCompactUpload *plan)
{
    delete plan;
}
