// The `tfhe.blind_rotation` subgraph kernel: the whole CMUX loop of a TFHE
// blind rotation in one cooperative launch.
//
// Semantics (crates/fhe/src/tfhe.rs `blind_rotation_subgraph`), exact modulo
// every CRT limb q_t of the ring:
//   acc = (initial_a; initial_b), a 2 x 1 evaluation-domain column
//   for i < n: e_i = ((mask_i * 2N + q / 2) div q) mod 2N
//              d   = X^{e_i} acc - acc
//              acc = acc + [key_a_i; key_b_i] * decompose(d)
//   result (acc_a, acc_b)
// `decompose` is mxx's exact per-limb balanced gadget decomposition: the
// coefficients of d in limb t (centered) split into `digits_per_tower`
// balanced base-2^b digits; digit row src_row * digits + t * digits_per_tower
// + j, used in every limb as a signed residue. The product and sums are in
// the evaluation domain after mxx's negacyclic NTT, so the result equals the
// graph's bit for bit.
//
// Work split: every iteration runs two phases separated by grid barriers:
// (1) the 2L (row, limb) blocks add the previous iteration's products to the
// accumulator and inverse-transform its difference; (2) the L * 2D (output
// limb, digit row) blocks decompose and forward-transform one digit row and
// add its products with both key rows into a 64-bit product accumulator.
// Each block has N / 4 threads running radix-4 register-blocked NTT passes.

#include "SubgraphKernel.cuh"

#include <cooperative_groups.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

namespace
{
    constexpr uint32_t kMaxLimbs = 4;
    constexpr uint32_t kMaxDegree = 2048;
    // Parameters registered by `TfheParams::gpu_blind_rotation_kernel`.
    enum Parameter : uint32_t
    {
        kLweDimension,
        kLogLweModulus,
        kBaseBits,
        kDigitsPerTower,
        kRetainedTowers,
        kParameterCount,
    };
    // Status codes written on invalid device data.
    constexpr uint32_t kInvalidMask = 11;

    struct Limb
    {
        uint64_t address;
        uint64_t coefficient_stride;
        uint32_t word_bytes;
        uint32_t modulus;
    };

    struct Arguments
    {
        // [row][limb]: row 0 is `a`, row 1 is `b`.
        Limb initial[2][kMaxLimbs];
        Limb output[2][kMaxLimbs];
        // Device tables of key member limbs, member major.
        const MxxRawMatrixLimb *key[2];
        const uint64_t *mask;
        int32_t mask_encoding;
        uint32_t lwe_dimension;
        uint32_t log_q;
        uint32_t limbs;
        uint32_t base_bits;
        uint32_t digits_per_tower;
        uint32_t digits;
        uint32_t *scratch;
        uint32_t *status;
        MxxNttTables ntt[kMaxLimbs];
        // floor(2^64 / q_t), for Barrett reduction of the key products.
        uint64_t barrett[kMaxLimbs];
    };

    __device__ __forceinline__ uint32_t add_mod(uint32_t a, uint32_t b, uint32_t q)
    {
        const uint32_t sum = a + b;
        return sum >= q ? sum - q : sum;
    }

    __device__ __forceinline__ uint32_t sub_mod(uint32_t a, uint32_t b, uint32_t q)
    {
        return a >= b ? a - b : a + q - b;
    }

    // a * w mod q with the 32-bit Shoup constant floor(w 2^32 / q), the high
    // word of mxx's 64-bit one.
    __device__ __forceinline__ uint32_t mul_shoup(uint32_t a, uint32_t w, uint32_t shoup, uint32_t q)
    {
        const uint32_t reduced = a * w - __umulhi(a, shoup) * q;
        return reduced >= q ? reduced - q : reduced;
    }

    __device__ __forceinline__ uint32_t load(const Limb &limb, uint32_t coefficient)
    {
        const uint64_t address = limb.address + coefficient * limb.coefficient_stride;
        return limb.word_bytes == 4 ?
            *reinterpret_cast<const uint32_t *>(address) :
            static_cast<uint32_t>(*reinterpret_cast<const uint64_t *>(address));
    }

    __device__ __forceinline__ void store(const Limb &limb, uint32_t coefficient, uint32_t value)
    {
        const uint64_t address = limb.address + coefficient * limb.coefficient_stride;
        if (limb.word_bytes == 4) *reinterpret_cast<uint32_t *>(address) = value;
        else *reinterpret_cast<uint64_t *>(address) = value;
    }

    // The residue of LWE mask coordinate `index` modulo 2^log_q, which alone
    // determines the ring exponent.
    __device__ __forceinline__ bool mask_residue(const Arguments &args, uint32_t index,
        uint64_t *residue)
    {
        const uint64_t mask = args.log_q == 64 ? ~0ULL : (1ULL << args.log_q) - 1;
        if (args.mask_encoding == 0 || args.mask_encoding == 1)
        {
            *residue = args.mask[index] & mask;
            return true;
        }
        if (args.mask_encoding < 3) return false;
        const uint64_t *value = args.mask + size_t(index) * (args.mask_encoding - 1);
        if (value[0] > 1) return false;
        const uint64_t magnitude = value[1] & mask;
        *residue = value[0] == 0 ? magnitude : (0 - magnitude) & mask;
        return true;
    }

    // One radix-RP pass of the negacyclic NTT over `values` (mxx's blocked
    // NTT pass): stage twiddle k is psi^{+-2k} for k < N/2.
    template <bool Forward, uint32_t RP>
    __device__ __forceinline__ void ntt_pass(uint32_t *values, uint32_t degree,
        uint32_t log_degree, uint32_t log_span, const uint32_t *twiddles,
        const uint32_t *shoup, uint32_t q)
    {
        constexpr uint32_t log_rp = RP == 4 ? 2 : 1;
        const uint32_t log_stride = Forward ? log_span - log_rp : log_span;
        const uint32_t stride = 1U << log_stride;
        for (uint32_t group = threadIdx.x; group < degree / RP; group += blockDim.x)
        {
            const uint32_t base =
                ((group >> log_stride) << (log_stride + log_rp)) | (group & (stride - 1));
            uint32_t x[RP];
#pragma unroll
            for (uint32_t k = 0; k < RP; ++k) x[k] = values[base + k * stride];
#pragma unroll
            for (uint32_t level = 0; level < log_rp; ++level)
            {
                const uint32_t log_half = Forward ? log_rp - level - 1 : level;
                const uint32_t half = 1U << log_half;
                const uint32_t log_length = log_half + 1 + log_stride;
#pragma unroll
                for (uint32_t k = 0; k < RP; ++k)
                {
                    if (k & half) continue;
                    const uint32_t offset = (base + k * stride) & ((1U << log_length) - 1);
                    const uint32_t t = offset << (log_degree - log_length);
                    const uint32_t lower = x[k], upper = x[k + half];
                    if constexpr (Forward)
                    {
                        x[k] = add_mod(lower, upper, q);
                        x[k + half] = mul_shoup(sub_mod(lower, upper, q), twiddles[t], shoup[t], q);
                    }
                    else
                    {
                        const uint32_t product = mul_shoup(upper, twiddles[t], shoup[t], q);
                        x[k] = add_mod(lower, product, q);
                        x[k + half] = sub_mod(lower, product, q);
                    }
                }
            }
#pragma unroll
            for (uint32_t k = 0; k < RP; ++k) values[base + k * stride] = x[k];
        }
    }

    // The stages of a whole transform over `values` in shared memory; a
    // shorter pass runs first (forward) or last (inverse) for an odd log N,
    // and full passes whose stride is at most 32 stay inside one warp.
    template <bool Forward, uint32_t LogDegree>
    __device__ __forceinline__ void ntt(uint32_t *values, const uint32_t *twiddles,
        const uint32_t *shoup, uint32_t q)
    {
        constexpr uint32_t degree = 1U << LogDegree;
        uint32_t remaining = LogDegree;
        uint32_t log_span = Forward ? LogDegree : 0;
        bool previous_local = false;
#pragma unroll
        for (uint32_t pass = 0; pass < (LogDegree + 1) / 2; ++pass)
        {
            const uint32_t bits = Forward && remaining % 2 != 0 ? 1 :
                (remaining < 2 ? remaining : 2);
            const uint32_t log_stride = Forward ? log_span - bits : log_span;
            const bool local = bits == 2 && log_stride <= 5 && degree / 4 >= 32;
            if (previous_local && local) __syncwarp();
            else __syncthreads();
            previous_local = local;
            if (bits == 2) ntt_pass<Forward, 4>(values, degree, LogDegree, log_span, twiddles, shoup, q);
            else ntt_pass<Forward, 2>(values, degree, LogDegree, log_span, twiddles, shoup, q);
            log_span = Forward ? log_span - bits : log_span + bits;
            remaining -= bits;
        }
        __syncthreads();
    }

    // psi^i and its 32-bit Shoup constant, for i < N, into shared memory.
    __device__ void stage_powers(uint32_t *powers, uint32_t *shoup, const uint64_t *table,
        const uint64_t *table_shoup, uint32_t degree)
    {
        for (uint32_t i = threadIdx.x; i < degree; i += blockDim.x)
        {
            powers[i] = static_cast<uint32_t>(table[i]);
            shoup[i] = static_cast<uint32_t>(table_shoup[i] >> 32);
        }
    }

    // Stage twiddles psi^{+-2k} (k < N/2) of `table` into shared memory.
    __device__ void stage_twiddles(uint32_t *twiddles, uint32_t *shoup, const uint64_t *table,
        const uint64_t *table_shoup, uint32_t degree)
    {
        for (uint32_t k = threadIdx.x; k < degree / 2; k += blockDim.x)
        {
            twiddles[k] = static_cast<uint32_t>(table[2 * k]);
            shoup[k] = static_cast<uint32_t>(table_shoup[2 * k] >> 32);
        }
    }

    __device__ __forceinline__ uint64_t barrett_reduce(uint64_t value, uint64_t barrett,
        uint32_t q)
    {
        uint64_t reduced = value - __umul64hi(value, barrett) * q;
        while (reduced >= q) reduced -= q;
        return reduced;
    }

    template <uint32_t LogDegree>
    __global__ void __launch_bounds__(1024) blind_rotation_kernel(
        const __grid_constant__ Arguments args)
    {
        extern __shared__ uint32_t shared[];
        constexpr uint32_t degree = 1U << LogDegree;
        // Shared: the transform, the stage twiddles of both directions, and
        // the powers psi^i (phase-2 limb) and psi^-i (phase-1 limb).
        uint32_t *values = shared;
        uint32_t *inverse_twiddles = values + degree;
        uint32_t *inverse_shoup = inverse_twiddles + degree / 2;
        uint32_t *forward_twiddles = inverse_shoup + degree / 2;
        uint32_t *forward_shoup = forward_twiddles + degree / 2;
        uint32_t *difference_powers = forward_shoup + degree / 2;
        uint32_t *difference_power_shoup = difference_powers + degree;
        uint32_t *inverse_powers = difference_power_shoup + degree;
        uint32_t *inverse_power_shoup = inverse_powers + degree;
        uint32_t *digit_powers = inverse_power_shoup + degree;
        uint32_t *digit_power_shoup = digit_powers + degree;
        const uint32_t limbs = args.limbs, digits = args.digits;
        const uint32_t rows = 2 * digits;
        // Scratch: the difference coefficients [limb][row][N] (32-bit), then
        // the product accumulator [row][limb][N] (64-bit): phase 2 adds each
        // digit row's reduced products into it, and phase 1 of the next
        // iteration adds it to the accumulator column and clears it.
        uint32_t *coefficients = args.scratch;
        auto *products = reinterpret_cast<unsigned long long *>(
            args.scratch + ((2 * limbs * degree + 1) & ~1U));
        cooperative_groups::grid_group grid = cooperative_groups::this_grid();
        const uint32_t block = blockIdx.x;
        // A block keeps the inverse twiddles of its phase-1 limb and the
        // forward twiddles of its phase-2 limb for the whole loop.
        const uint32_t difference_limb = block < 2 * limbs ? block / 2 : 0;
        const uint32_t digit_limb = block < limbs * rows ? block / rows : 0;
        stage_twiddles(inverse_twiddles, inverse_shoup, args.ntt[difference_limb].inverse,
            args.ntt[difference_limb].inverse_shoup, degree);
        stage_twiddles(forward_twiddles, forward_shoup, args.ntt[digit_limb].forward,
            args.ntt[digit_limb].forward_shoup, degree);
        stage_powers(difference_powers, difference_power_shoup, args.ntt[difference_limb].forward,
            args.ntt[difference_limb].forward_shoup, degree);
        stage_powers(inverse_powers, inverse_power_shoup, args.ntt[difference_limb].inverse,
            args.ntt[difference_limb].inverse_shoup, degree);
        stage_powers(digit_powers, digit_power_shoup, args.ntt[digit_limb].forward,
            args.ntt[digit_limb].forward_shoup, degree);
        // The accumulator column lives in the outputs.
        for (uint32_t k = block * blockDim.x + threadIdx.x; k < 2 * limbs * degree;
             k += gridDim.x * blockDim.x)
        {
            const uint32_t row = k / (limbs * degree), limb = k / degree % limbs, j = k % degree;
            store(args.output[row][limb], j, load(args.initial[row][limb], j));
            products[k] = 0;
        }
        grid.sync();
        for (uint32_t iteration = 0; iteration < args.lwe_dimension; ++iteration)
        {
            uint64_t residue = 0;
            if (!mask_residue(args, iteration, &residue))
            {
                if (block == 0 && threadIdx.x == 0) atomicCAS(args.status, 0U, kInvalidMask);
                return;
            }
            const uint32_t exponent = static_cast<uint32_t>(
                ((static_cast<unsigned __int128>(residue) * (2 * degree) +
                     (static_cast<unsigned __int128>(1) << (args.log_q - 1))) >> args.log_q) &
                (2 * degree - 1));
            // Phase 1: acc += previous products; d = X^e acc - acc, back to
            // coefficients. The other blocks prefetch this iteration's key
            // rows into L2 for phase 2.
            if (block >= 2 * limbs)
            {
                const uint32_t lines_per_row = (degree * 4 + 127) / 128;
                const uint32_t lines = 2 * limbs * rows * lines_per_row;
                for (uint32_t line = (block - 2 * limbs) * blockDim.x + threadIdx.x; line < lines;
                     line += (gridDim.x - 2 * limbs) * blockDim.x)
                {
                    const uint32_t row = line / (limbs * rows * lines_per_row);
                    const uint32_t limb = line / (rows * lines_per_row) % limbs;
                    const uint32_t digit_row = line / lines_per_row % rows;
                    const MxxRawMatrixLimb key = args.key[row][size_t(iteration) * limbs + limb];
                    const uint64_t address = key.address + digit_row * key.column_stride_bytes +
                        min((line % lines_per_row) * 128, (degree - 1) * 4);
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(address));
                }
            }
            else
            {
                const uint32_t row = block % 2, limb = block / 2;
                const Limb &acc = args.output[row][limb];
                const uint32_t q = acc.modulus;
                const uint64_t barrett = args.barrett[limb];
                const MxxNttTables &tables = args.ntt[limb];
                unsigned long long *pending = products + (size_t(row) * limbs + limb) * degree;
                for (uint32_t s = threadIdx.x; s < degree; s += blockDim.x)
                {
                    uint32_t value = load(acc, s);
                    if (iteration != 0)
                    {
                        value = add_mod(value,
                            static_cast<uint32_t>(barrett_reduce(pending[s], barrett, q)), q);
                        store(acc, s, value);
                        pending[s] = 0;
                    }
                    const uint32_t reversed = __brev(s) >> (32 - LogDegree);
                    const uint32_t power = exponent * (2 * reversed + 1) & (2 * degree - 1);
                    const uint32_t index = power < degree ? power : power - degree;
                    uint32_t rotated = mul_shoup(value, difference_powers[index],
                        difference_power_shoup[index], q);
                    if (power >= degree && rotated != 0) rotated = q - rotated;
                    values[s] = sub_mod(rotated, value, q);
                }
                ntt<false, LogDegree>(values, inverse_twiddles, inverse_shoup, q);
                const uint32_t inverse = static_cast<uint32_t>(*tables.degree_inverse);
                const uint32_t inverse_shoup_word =
                    static_cast<uint32_t>(*tables.degree_inverse_shoup >> 32);
                for (uint32_t i = threadIdx.x; i < degree; i += blockDim.x)
                {
                    const uint32_t scaled = mul_shoup(values[i], inverse, inverse_shoup_word, q);
                    coefficients[(limb * 2 + row) * degree + i] =
                        mul_shoup(scaled, inverse_powers[i], inverse_power_shoup[i], q);
                }
            }
            grid.sync();
            // Phase 2: digit row `digit_row` as residues of limb `limb`,
            // forward, times both key rows, added into the products.
            if (block < limbs * rows)
            {
                const uint32_t limb = block / rows, digit_row = block % rows;
                const uint32_t source_row = digit_row / digits;
                const uint32_t tower = digit_row % digits / args.digits_per_tower;
                const uint32_t position = digit_row % args.digits_per_tower;
                const int32_t tower_modulus = static_cast<int32_t>(args.output[0][tower].modulus);
                const uint32_t q = args.output[0][limb].modulus;
                const uint64_t barrett = args.barrett[limb];
                const int32_t base = 1 << args.base_bits, half = base / 2;
                for (uint32_t j = threadIdx.x; j < degree; j += blockDim.x)
                {
                    int32_t value = static_cast<int32_t>(
                        coefficients[(tower * 2 + source_row) * degree + j]);
                    if (2 * static_cast<int64_t>(value) > tower_modulus) value -= tower_modulus;
                    int32_t digit = 0;
                    for (uint32_t step = 0; step <= position; ++step)
                    {
                        const int32_t quotient = value >> args.base_bits;
                        const int32_t remainder = value & (base - 1);
                        if (remainder < half) { digit = remainder; value = quotient; }
                        else if (remainder > half) { digit = remainder - base; value = quotient + 1; }
                        else if ((quotient & 1) == 0) { digit = half; value = quotient; }
                        else { digit = -half; value = quotient + 1; }
                    }
                    const uint32_t residue_digit = digit < 0 ?
                        q - static_cast<uint32_t>(-digit) : static_cast<uint32_t>(digit);
                    values[j] = mul_shoup(residue_digit, digit_powers[j], digit_power_shoup[j], q);
                }
                ntt<true, LogDegree>(values, forward_twiddles, forward_shoup, q);
                for (uint32_t row = 0; row < 2; ++row)
                {
                    const MxxRawMatrixLimb key = args.key[row][size_t(iteration) * limbs + limb];
                    const uint64_t column = key.address + digit_row * key.column_stride_bytes;
                    unsigned long long *target = products + (size_t(row) * limbs + limb) * degree;
                    for (uint32_t j = threadIdx.x; j < degree; j += blockDim.x)
                    {
                        const uint64_t address = column + j * key.coefficient_stride_bytes;
                        const uint64_t entry = key.word_bytes == 4 ?
                            *reinterpret_cast<const uint32_t *>(address) :
                            *reinterpret_cast<const uint64_t *>(address);
                        atomicAdd(target + j, barrett_reduce(entry * values[j], barrett, q));
                    }
                }
            }
            grid.sync();
        }
        // The last iteration's products.
        for (uint32_t k = block * blockDim.x + threadIdx.x; k < 2 * limbs * degree;
             k += gridDim.x * blockDim.x)
        {
            const uint32_t row = k / (limbs * degree), limb = k / degree % limbs, j = k % degree;
            const Limb &acc = args.output[row][limb];
            const uint32_t sum = static_cast<uint32_t>(
                barrett_reduce(products[k], args.barrett[limb], acc.modulus));
            store(acc, j, add_mod(load(acc, j), sum, acc.modulus));
        }
    }

    uint32_t log2_exact(uint64_t value)
    {
        uint32_t log = 0;
        while ((1ULL << log) < value) ++log;
        return (1ULL << log) == value ? log : UINT32_MAX;
    }

    MxxGraphPatch address_patch(uint32_t offset, uint32_t binding)
    {
        return MxxGraphPatch{nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, offset,
            sizeof(uint64_t), binding, 0};
    }
}

extern "C" int mxx_fhe_tfhe_blind_rotation(const MxxSubgraphLaunch *launch)
{
    const auto fail = [](const char *message) { return gpu_set_last_error(message); };
    if (!launch || launch->input_count != 5 || launch->output_count != 2 ||
        launch->parameter_count != kParameterCount)
        return fail("tfhe.blind_rotation needs 5 inputs, 2 outputs and 5 parameters");
    const MxxSubgraphOperand *operands = launch->operands;
    const uint64_t *parameters = launch->parameters;
    const uint32_t degree = launch->degree, limbs = launch->limb_count;
    const uint32_t log_degree = log2_exact(degree);
    const uint32_t digits = static_cast<uint32_t>(
        parameters[kDigitsPerTower] * parameters[kRetainedTowers]);
    const uint32_t rows = 2 * digits;
    if (log_degree == UINT32_MAX || degree < 16 || degree > kMaxDegree || limbs == 0 ||
        limbs > kMaxLimbs || parameters[kRetainedTowers] > limbs || digits == 0 ||
        parameters[kBaseBits] == 0 || parameters[kBaseBits] > 30 ||
        parameters[kLogLweModulus] == 0 || parameters[kLogLweModulus] > 64 ||
        launch->scratch_bytes < (uint64_t(2 * limbs) * degree + 1) / 2 * 8 +
            uint64_t(2 * limbs) * degree * 8)
        return fail("tfhe.blind_rotation parameters are outside the kernel's range");
    const MxxSubgraphOperand &initial_a = operands[0], &initial_b = operands[1];
    const MxxSubgraphOperand &mask = operands[2], &key_a = operands[3], &key_b = operands[4];
    const MxxSubgraphOperand &output_a = operands[5], &output_b = operands[6];
    const auto one_by = [&](const MxxSubgraphOperand &operand, uint64_t columns) {
        return operand.matrix.rows == 1 && operand.matrix.columns == columns &&
            operand.matrix.limb_count == limbs;
    };
    if (initial_a.kind != MXX_SUBGRAPH_MATRIX || initial_b.kind != MXX_SUBGRAPH_MATRIX ||
        output_a.kind != MXX_SUBGRAPH_MATRIX || output_b.kind != MXX_SUBGRAPH_MATRIX ||
        mask.kind != MXX_SUBGRAPH_INTEGER_FAMILY || key_a.kind != MXX_SUBGRAPH_MATRIX_FAMILY ||
        key_b.kind != MXX_SUBGRAPH_MATRIX_FAMILY || !one_by(initial_a, 1) ||
        !one_by(initial_b, 1) || !one_by(output_a, 1) || !one_by(output_b, 1) ||
        !one_by(key_a, rows) || !one_by(key_b, rows) ||
        mask.integer_count != parameters[kLweDimension] ||
        key_a.family_count != parameters[kLweDimension] ||
        key_b.family_count != parameters[kLweDimension])
        return fail("tfhe.blind_rotation operands have unexpected shapes");
    Arguments args{};
    std::vector<MxxGraphPatch> patches;
    const MxxSubgraphOperand *matrices[2][2] = {{&initial_a, &initial_b}, {&output_a, &output_b}};
    for (uint32_t side = 0; side < 2; ++side)
        for (uint32_t row = 0; row < 2; ++row)
            for (uint32_t limb = 0; limb < limbs; ++limb)
            {
                const MxxRawMatrixLimb &source = matrices[side][row]->matrix.limbs[limb];
                if (source.modulus >= (1ULL << 31) || (source.word_bytes != 4 && source.word_bytes != 8))
                    return fail("tfhe.blind_rotation needs 31-bit CRT limbs");
                Limb &target = side == 0 ? args.initial[row][limb] : args.output[row][limb];
                target = Limb{source.address, source.coefficient_stride_bytes, source.word_bytes,
                    static_cast<uint32_t>(source.modulus)};
                const uint32_t offset = static_cast<uint32_t>(
                    (side == 0 ? offsetof(Arguments, initial) : offsetof(Arguments, output)) +
                    (row * kMaxLimbs + limb) * sizeof(Limb) + offsetof(Limb, address));
                patches.push_back(address_patch(offset, matrices[side][row]->binding + limb));
            }
    args.key[0] = key_a.family_table;
    args.key[1] = key_b.family_table;
    args.mask = static_cast<const uint64_t *>(mask.integers);
    args.mask_encoding = mask.integer_encoding;
    patches.push_back(address_patch(offsetof(Arguments, mask), mask.binding));
    args.lwe_dimension = static_cast<uint32_t>(parameters[kLweDimension]);
    args.log_q = static_cast<uint32_t>(parameters[kLogLweModulus]);
    args.limbs = limbs;
    args.base_bits = static_cast<uint32_t>(parameters[kBaseBits]);
    args.digits_per_tower = static_cast<uint32_t>(parameters[kDigitsPerTower]);
    args.digits = digits;
    args.scratch = static_cast<uint32_t *>(launch->scratch);
    patches.push_back(address_patch(offsetof(Arguments, scratch), launch->scratch_binding));
    args.status = launch->status;
    patches.push_back(address_patch(offsetof(Arguments, status), launch->status_binding));
    std::memcpy(args.ntt, launch->ntt, limbs * sizeof(MxxNttTables));
    for (uint32_t limb = 0; limb < limbs; ++limb)
    {
        const uint64_t q = args.output[0][limb].modulus;
        args.barrett[limb] = static_cast<uint64_t>((static_cast<unsigned __int128>(1) << 64) / q);
    }
    const uint32_t threads = std::max<uint32_t>(degree / 4, 32);
    const uint32_t blocks = std::max(2 * limbs, limbs * rows);
    const size_t shared = size_t(degree) * 9 * sizeof(uint32_t);
    const auto run = [&](auto kernel) {
        if (shared > 48 * 1024 &&
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(shared)) != cudaSuccess)
            return fail("tfhe.blind_rotation shared memory exceeds the device limit");
        int per_sm = 0, sms = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, shared) !=
                cudaSuccess ||
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount,
                launch->physical_device) != cudaSuccess ||
            uint64_t(per_sm) * sms < blocks)
            return fail("tfhe.blind_rotation grid does not fit one cooperative launch");
        return mxx_gpu_launch_cooperative_kernel(launch->context,
            static_cast<cudaStream_t>(launch->stream), kernel, dim3(blocks), dim3(threads),
            shared, patches.data(), patches.size(), args);
    };
    switch (log_degree)
    {
    case 4: return run(blind_rotation_kernel<4>);
    case 5: return run(blind_rotation_kernel<5>);
    case 6: return run(blind_rotation_kernel<6>);
    case 7: return run(blind_rotation_kernel<7>);
    case 8: return run(blind_rotation_kernel<8>);
    case 9: return run(blind_rotation_kernel<9>);
    case 10: return run(blind_rotation_kernel<10>);
    case 11: return run(blind_rotation_kernel<11>);
    default: return fail("tfhe.blind_rotation ring dimension is outside 16..2048");
    }
}
