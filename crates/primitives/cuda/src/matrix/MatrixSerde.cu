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
    const size_t batch_limbs = mat->shared_limb_buffers.size() == 1 ? limbs : 1;
    std::vector<SerdeStreamRef> streams;
    streams.reserve(limbs);
    // A prepared owner retains the backing's streams even when its requested
    // shape becomes smaller. Query those immutable streams, not a fresh shape's
    // allocation class. No device selection, allocation or GPU wait is needed.
    for (size_t limb = 0; limb < limbs; limb += batch_limbs) {
        const dim3 id = mat->ctx->limb_gpu_ids[limb];
        if (id.x >= mat->shared_limb_buffers.size()) return set_error("invalid RNS partition");
        cudaStream_t stream = nullptr;
        int status = matrix_limb_stream(mat, id, &stream);
        if (status != 0) return status;
        for (size_t i = 0; i < batch_limbs; ++i) {
            cudaStream_t producer = nullptr;
            status = matrix_limb_stream(mat, mat->ctx->limb_gpu_ids[limb + i], &producer);
            if (status != 0) return status;
            // Each cross-stream consumer join consumes one temporary event.
            if (producer != stream) ++*out_count;
        }
        serde_append_unique_stream(streams, mat->shared_limb_buffers[id.x].device, stream);
    }
    *out_count += streams.size(); // One returned completion per D2H stream.
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

    std::vector<SerdeStreamRef> streams;
    streams.reserve(limb_count);
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
        for (size_t index = 0; index < batch_limbs; ++index)
        {
            status = matrix_track_limb_consumer(mat, limb_map[limb + index], device, stream);
            if (status != 0) return status;
        }
        serde_append_unique_stream(streams, device, stream);
    }

    return serde_build_event_set_from_streams(mat->ctx, streams, out_events);
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
