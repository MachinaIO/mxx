#include "Runtime.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <memory>
#include <set>
#include <map>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>



static_assert(sizeof(MxxExportSlotHeader) == 40, "export slot ABI size");
static_assert(offsetof(MxxExportSlotHeader, ready) == 0, "export slot ready offset");
static_assert(offsetof(MxxExportSlotHeader, occurrence) == 8, "export slot occurrence offset");
static_assert(offsetof(MxxExportSlotHeader, artifact_offset) == 16, "export slot artifact offset");
static_assert(offsetof(MxxExportSlotHeader, payload_bytes) == 24, "export slot payload size offset");
static_assert(offsetof(MxxExportSlotHeader, site) == 32, "export slot site offset");
static_assert(offsetof(MxxExportSlotHeader, flags) == 36, "export slot flags offset");

// A graph copy between two GPUs, run by one of them through peer access.
// CUDA rebinds a peer memcpy node only while its operands keep their original
// mappings, but a kernel's pointer arguments may be rebound freely.
__global__ void mxx_peer_copy_kernel(uint8_t *destination, const uint8_t *source,
    uint64_t bytes)
{
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    const uint64_t first = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool vector = ((reinterpret_cast<uintptr_t>(destination) |
        reinterpret_cast<uintptr_t>(source) | bytes) & 15U) == 0;
    if (vector)
    {
        uint4 *to = reinterpret_cast<uint4 *>(destination);
        const uint4 *from = reinterpret_cast<const uint4 *>(source);
        for (uint64_t index = first; index < bytes / 16; index += stride) to[index] = from[index];
        return;
    }
    for (uint64_t index = first; index < bytes; index += stride) destination[index] = source[index];
}

// Copy the elements of a four-dimensional view between two layouts, each
// given by its byte strides. An artifact export gathers a strided physical
// view densely; a device import scatters it back.
__global__ void mxx_strided_copy_kernel(uint8_t *destination, const uint8_t *source,
    MxxStridedCopy copy)
{
    // Copy each element in the widest units its alignment allows, so that
    // writes into mapped host memory stay wide and coalesced.
    uint64_t alignment = reinterpret_cast<uintptr_t>(destination) |
        reinterpret_cast<uintptr_t>(source) | copy.element_bytes;
    for (int axis = 0; axis < 4; ++axis)
        alignment |= copy.source_stride[axis] | copy.destination_stride[axis];
    const uint32_t unit = (alignment & 15U) == 0 ? 16 : (alignment & 7U) == 0 ? 8
        : (alignment & 3U) == 0 ? 4 : 1;
    const uint64_t units_per_element = copy.element_bytes / unit;
    const uint64_t units = copy.extent[0] * copy.extent[1] * copy.extent[2] * copy.extent[3] *
        units_per_element;
    const uint64_t step = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t index = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < units; index += step)
    {
        uint64_t rest = index / units_per_element;
        const uint64_t within = (index % units_per_element) * unit;
        uint64_t from = within;
        uint64_t to = within;
        for (int axis = 3; axis >= 0; --axis)
        {
            const uint64_t coordinate = rest % copy.extent[axis];
            rest /= copy.extent[axis];
            from += coordinate * copy.source_stride[axis];
            to += coordinate * copy.destination_stride[axis];
        }
        switch (unit)
        {
        case 16:
            *reinterpret_cast<uint4 *>(destination + to) =
                *reinterpret_cast<const uint4 *>(source + from);
            break;
        case 8:
            *reinterpret_cast<uint64_t *>(destination + to) =
                *reinterpret_cast<const uint64_t *>(source + from);
            break;
        case 4:
            *reinterpret_cast<uint32_t *>(destination + to) =
                *reinterpret_cast<const uint32_t *>(source + from);
            break;
        default:
            destination[to] = source[from];
        }
    }
}

static uint32_t strided_copy_blocks(const MxxStridedCopy &copy, uint32_t threads)
{
    const uint64_t units = copy.extent[0] * copy.extent[1] * copy.extent[2] * copy.extent[3] *
        ((copy.element_bytes + 15) / 16);
    return static_cast<uint32_t>(
        std::max<uint64_t>(1, std::min<uint64_t>((units + threads - 1) / threads, 65535)));
}

__global__ void mxx_export_slot_publish_kernel(MxxExportSlotHeader *header,
    uint64_t occurrence, uint64_t artifact_offset, uint64_t payload_bytes,
    uint32_t site, uint32_t flags, const uint32_t *gate)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    // A failed status word marks the publication suppressed: the payload was
    // computed by a failed operation and must not be acted on.
    if (gate && *gate != 0) flags |= MXX_EXPORT_SLOT_SUPPRESSED;
    header->occurrence = occurrence;
    header->artifact_offset = artifact_offset;
    header->payload_bytes = payload_bytes;
    header->site = site;
    header->flags = flags;
    __threadfence_system();
    // A fenced aligned store: GPUs on PCIe have no native atomics on host
    // memory, and the host reads `ready` with an acquire load.
    *reinterpret_cast<volatile unsigned long long *>(&header->ready) = 1ULL;
}

#if !defined(MXX_GPU_BACKEND_HIP) && defined(CUDART_VERSION) && CUDART_VERSION >= 12000

__global__ void mxx_if_gate_kernel(const uint64_t *predicate,
    cudaGraphConditionalHandle handle)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        cudaGraphSetConditional(handle, *predicate != 0);
}

__global__ void mxx_while_gate_kernel(uint64_t *index,
    const uint64_t *limit, uint64_t max_iterations, uint32_t *status,
    cudaGraphConditionalHandle handle, bool advance)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (advance) ++*index;
    const uint64_t requested = *limit;
    if (requested > max_iterations)
    {
        atomicCAS(status, 0U, 4U);
        cudaGraphSetConditional(handle, 0U);
        return;
    }
    cudaGraphSetConditional(handle, *index < requested && *status == 0U);
}

#endif

struct PinnedHostReclaimer
{
    struct Job
    {
        int device;
        gpuEvent_t completion;
        std::vector<void *> pointers;
    };

    PinnedHostReclaimer()
    {
        // Start only after every member (including worker-visible flags) has
        // completed initialization, irrespective of declaration order.
        worker = std::thread(&PinnedHostReclaimer::run, this);
    }

    ~PinnedHostReclaimer()
    {
        shutdown();
    }

    int enqueue(int device, gpuEvent_t completion, std::vector<void *> &&pointers)
    {
        try
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping || joined)
            {
                record_failure_locked("pinned-host reclaimer is stopped");
                return 1;
            }
            pending.push_back(Job{device, completion, std::move(pointers)});
        }
        catch (const std::exception &error)
        {
            std::lock_guard<std::mutex> lock(mutex);
            record_failure_locked(error.what());
            return 1;
        }
        wake.notify_one();
        return 0;
    }

    void record_uncertain(const char *message)
    {
        std::lock_guard<std::mutex> lock(mutex);
        record_failure_locked(message);
    }

    int wait_idle()
    {
        std::unique_lock<std::mutex> lock(mutex);
        idle.wait(lock, [this]() { return pending.empty() && active == 0; });
        return failed ? 1 : 0;
    }

    std::string failure_message()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return failure_message_text.empty() ? "pinned-host reclamation failed"
                                             : failure_message_text;
    }

    void shutdown()
    {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (joined)
            {
                return;
            }
            stopping = true;
        }
        wake.notify_all();
        if (worker.joinable())
        {
            worker.join();
        }
        std::lock_guard<std::mutex> lock(mutex);
        joined = true;
    }

private:
    void record_failure_locked(const char *message)
    {
        failed = true;
        if (failure_message_text.empty())
        {
            failure_message_text = message ? message : "unknown pinned-host reclamation failure";
        }
    }

    void record_failure(const char *message)
    {
        std::lock_guard<std::mutex> lock(mutex);
        record_failure_locked(message);
    }

    void process(Job &job)
    {
        gpuError_t error = mxx_set_device(job.device);
        if (error == gpuSuccess)
        {
            error = gpuEventSynchronize(job.completion);
        }
        if (error != gpuSuccess)
        {
            record_failure(gpuGetErrorString(error));
            // The event and pointers are intentionally leaked. Once
            // synchronization is uncertain, freeing host memory could race
            // with an in-flight asynchronous copy.
            return;
        }

        error = gpuEventDestroy(job.completion);
        if (error != gpuSuccess)
        {
            record_failure(gpuGetErrorString(error));
            // Keep the pointers leaked when event destruction is uncertain.
            return;
        }

        for (void *pointer : job.pointers)
        {
            if (!pointer)
            {
                continue;
            }
            error = gpuFreeHost(pointer);
            if (error != gpuSuccess)
            {
                // Do not retry an uncertain free. The failed pointer is
                // leaked, while independent pointers can still be reclaimed.
                record_failure(gpuGetErrorString(error));
            }
        }
    }

    void run()
    {
        for (;;)
        {
            Job job{};
            {
                std::unique_lock<std::mutex> lock(mutex);
                wake.wait(lock, [this]() { return stopping || !pending.empty(); });
                if (pending.empty())
                {
                    if (stopping)
                    {
                        return;
                    }
                    continue;
                }
                job = std::move(pending.front());
                pending.pop_front();
                ++active;
            }

            process(job);

            {
                std::lock_guard<std::mutex> lock(mutex);
                --active;
                if (pending.empty() && active == 0)
                {
                    idle.notify_all();
                }
            }
        }
    }

    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable idle;
    std::deque<Job> pending;
    std::thread worker;
    size_t active = 0;
    bool stopping = false;
    bool joined = false;
    bool failed = false;
    std::string failure_message_text;
};

namespace
{
    thread_local std::string last_error;
    constexpr size_t MAX_TRACKED_GPU_DEVICES = 256;
    std::atomic<size_t> live_context_counts[MAX_TRACKED_GPU_DEVICES]{};
    std::atomic<uint64_t> context_generations[MAX_TRACKED_GPU_DEVICES]{};

    int set_error(const char *msg)
    {
        last_error = msg ? msg : "unknown error";
        return 1;
    }

    int set_error(gpuError_t err)
    {
        last_error = gpuGetErrorString(err);
        return err == gpuErrorMemoryAllocation ? GPU_STATUS_OUT_OF_MEMORY : 1;
    }

    int set_error(const std::exception &e)
    {
        return set_error(e.what());
    }

    void destroy_event_set(GpuEventSet *events)
    {
        if (!events)
        {
            return;
        }
        for (const auto &entry : events->entries)
        {
            (void)mxx_set_device(entry.device);
            (void)gpuEventDestroy(entry.event);
        }
        delete events;
    }

    int fence_release_streams(const GpuExecutionOwner *owner)
    {
        if (!owner)
        {
            return set_error("invalid GPU context");
        }
        for (size_t partition = 0; partition < owner->release_streams_by_partition.size(); ++partition)
        {
            const int device = owner->gpu_ids[partition];
            gpuStream_t stream = owner->release_streams_by_partition[partition];
            if (!stream)
            {
                continue;
            }
            gpuError_t err = mxx_set_device(device);
            if (err != gpuSuccess)
            {
                return set_error(err);
            }
            gpuEvent_t epoch = nullptr;
            err = gpuEventCreateWithFlags(&epoch, gpuEventDisableTiming);
            if (err == gpuSuccess)
            {
                err = gpuEventRecord(epoch, stream);
            }
            if (err == gpuSuccess)
            {
                err = gpuEventSynchronize(epoch);
            }
            if (epoch)
            {
                (void)gpuEventDestroy(epoch);
            }
            if (err != gpuSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    int wait_pinned_host_reclaimer(const GpuExecutionOwner *owner)
    {
        if (!owner || !owner->pinned_host_reclaimer)
        {
            return 0;
        }
        PinnedHostReclaimer *reclaimer = owner->pinned_host_reclaimer;
        const int status = reclaimer->wait_idle();
        if (status != 0)
        {
            const std::string message = reclaimer->failure_message();
            return set_error(message.c_str());
        }
        return 0;
    }

    int shutdown_pinned_host_reclaimer(GpuExecutionOwner *owner)
    {
        if (!owner || !owner->pinned_host_reclaimer)
        {
            return 0;
        }
        PinnedHostReclaimer *reclaimer = owner->pinned_host_reclaimer;
        reclaimer->shutdown();
        const int status = reclaimer->wait_idle();
        if (status != 0)
        {
            const std::string message = reclaimer->failure_message();
            delete reclaimer;
            owner->pinned_host_reclaimer = nullptr;
            return set_error(message.c_str());
        }
        delete reclaimer;
        owner->pinned_host_reclaimer = nullptr;
        return 0;
    }

    void destroy_context_streams(GpuExecutionOwner *owner)
    {
        if (!owner)
        {
            return;
        }
        const int stream_status = fence_release_streams(owner);
        const int reclaimer_status = shutdown_pinned_host_reclaimer(owner);
        if (stream_status != 0 || reclaimer_status != 0)
        {
            set_error("GPU context release cleanup failed");
        }
        for (size_t partition = 0; partition < owner->gpu_ids.size(); ++partition)
        {
            (void)mxx_set_device(owner->gpu_ids[partition]);
            if (partition < owner->release_streams_by_partition.size() &&
                owner->release_streams_by_partition[partition])
            {
                (void)gpuStreamDestroy(owner->release_streams_by_partition[partition]);
                owner->release_streams_by_partition[partition] = nullptr;
            }
            if (partition < owner->transfer_streams_by_partition.size() &&
                owner->transfer_streams_by_partition[partition])
            {
                (void)gpuStreamDestroy(owner->transfer_streams_by_partition[partition]);
                owner->transfer_streams_by_partition[partition] = nullptr;
            }
            if (partition < owner->compute_streams_by_partition.size())
            {
                for (gpuStream_t &stream : owner->compute_streams_by_partition[partition])
                {
                    if (stream)
                    {
                        (void)gpuStreamDestroy(stream);
                        stream = nullptr;
                    }
                }
            }
        }
        owner->release_streams_by_partition.clear();
        owner->compute_streams_by_partition.clear();
    }

    bool mod_inverse_u64(uint64_t a, uint64_t modulus, uint64_t &out_inv)
    {
        if (modulus == 0)
        {
            return false;
        }
        __int128 t = 0;
        __int128 new_t = 1;
        __int128 r = static_cast<__int128>(modulus);
        __int128 new_r = static_cast<__int128>(a % modulus);
        while (new_r != 0)
        {
            const __int128 q = r / new_r;

            const __int128 tmp_t = t - q * new_t;
            t = new_t;
            new_t = tmp_t;

            const __int128 tmp_r = r - q * new_r;
            r = new_r;
            new_r = tmp_r;
        }
        if (r != 1)
        {
            return false;
        }
        if (t < 0)
        {
            t += static_cast<__int128>(modulus);
        }
        out_inv = static_cast<uint64_t>(t);
        return true;
    }

    std::vector<uint64_t> compute_garner_inverse_table(const std::vector<uint64_t> &moduli, int limb_count)
    {
        const size_t count = static_cast<size_t>(limb_count);
        std::vector<uint64_t> inverse_table(count * count, 0);
        for (int i = 1; i < limb_count; ++i)
        {
            const uint64_t qi = moduli[static_cast<size_t>(i)];
            for (int j = 0; j < i; ++j)
            {
                const uint64_t qj = moduli[static_cast<size_t>(j)];
                uint64_t inv = 0;
                if (!mod_inverse_u64(qj % qi, qi, inv))
                {
                    throw std::runtime_error("CRT moduli must be pairwise coprime");
                }
                inverse_table[static_cast<size_t>(j) * count + static_cast<size_t>(i)] = inv;
            }
        }
        return inverse_table;
    }

    uint64_t mul_mod_u64_host(uint64_t a, uint64_t b, uint64_t mod)
    {
        const unsigned __int128 product = static_cast<unsigned __int128>(a) * b;
        return static_cast<uint64_t>(product % mod);
    }

    uint64_t shoup_reciprocal_u64_host(uint64_t value, uint64_t modulus)
    {
        return static_cast<uint64_t>(
            (static_cast<unsigned __int128>(value) << 64U) / modulus);
    }

    uint64_t pow_mod_u64_host(uint64_t base, uint64_t exp, uint64_t mod)
    {
        uint64_t result = 1 % mod;
        uint64_t cur = base % mod;
        uint64_t e = exp;
        while (e != 0)
        {
            if ((e & 1ULL) != 0)
            {
                result = mul_mod_u64_host(result, cur, mod);
            }
            cur = mul_mod_u64_host(cur, cur, mod);
            e >>= 1ULL;
        }
        return result;
    }

    uint64_t find_primitive_root_u64(uint64_t prime)
    {
        if (prime <= 2)
        {
            throw std::runtime_error("invalid prime for primitive root");
        }

        uint64_t phi = prime - 1;
        uint64_t n = phi;
        std::vector<uint64_t> factors;
        for (uint64_t d = 2; d * d <= n; ++d)
        {
            if (n % d != 0)
            {
                continue;
            }
            factors.push_back(d);
            while (n % d == 0)
            {
                n /= d;
            }
        }
        if (n > 1)
        {
            factors.push_back(n);
        }

        for (uint64_t candidate = 2; candidate < prime; ++candidate)
        {
            bool ok = true;
            for (uint64_t factor : factors)
            {
                if (pow_mod_u64_host(candidate, phi / factor, prime) == 1)
                {
                    ok = false;
                    break;
                }
            }
            if (ok)
            {
                return candidate;
            }
        }
        throw std::runtime_error("failed to find primitive root");
    }

    uint64_t compute_2nth_unity_root_u64(uint64_t prime, uint64_t n)
    {
        if (n == 0 || n > std::numeric_limits<uint64_t>::max() / 2)
        {
            throw std::runtime_error("invalid ring size while computing NTT root");
        }
        const uint64_t order = n * 2;
        if (prime % order != 1)
        {
            throw std::runtime_error("modulus is not congruent to 1 mod 2N");
        }

        const uint64_t generator = find_primitive_root_u64(prime);
        const uint64_t root = pow_mod_u64_host(generator, (prime - 1) / order, prime);
        if (pow_mod_u64_host(root, order, prime) != 1)
        {
            throw std::runtime_error("computed root does not have order dividing 2N");
        }
        if (pow_mod_u64_host(root, n, prime) != prime - 1)
        {
            throw std::runtime_error("computed root is not a primitive 2N-th root");
        }
        // OpenFHE canonicalizes a power-of-two root to the smallest primitive
        // root. Matching that choice keeps the GPU evaluation representation
        // bit-exact with DCRTPoly rather than merely internally invertible.
        uint64_t minimum_root = root;
        uint64_t odd_power = root;
        const uint64_t root_squared = mul_mod_u64_host(root, root, prime);
        for (uint64_t exponent = 3; exponent < order; exponent += 2)
        {
            odd_power = mul_mod_u64_host(odd_power, root_squared, prime);
            minimum_root = std::min(minimum_root, odd_power);
        }
        return minimum_root;
    }

    void validate_gpu_list(const std::vector<int> &gpu_list)
    {
        if (gpu_list.empty())
        {
            throw std::runtime_error("empty gpu list");
        }
        if (gpu_list.size() > GPU_RUNTIME_MAX_DIGITS)
        {
            throw std::runtime_error("gpu count exceeds supported maximum");
        }

        int device_count = 0;
        if (gpu_device_count(&device_count) != 0)
        {
            throw std::runtime_error("cannot query GPU devices");
        }
        if (device_count <= 0)
        {
            throw std::runtime_error("no GPU device available");
        }

        std::unordered_set<int> seen;
        seen.reserve(gpu_list.size());
        for (int id : gpu_list)
        {
            if (id < 0 || id >= device_count)
            {
                throw std::runtime_error("invalid gpu id in context creation");
            }
            if (!seen.insert(id).second)
            {
                throw std::runtime_error("duplicate gpu id in context creation");
            }
        }
    }

    std::vector<size_t> compute_decomp_counts_by_partition(size_t gpu_count, uint32_t dnum)
    {
        std::vector<size_t> counts(gpu_count, 0);
        for (uint32_t digit = 0; digit < dnum; ++digit)
        {
            counts[static_cast<size_t>(digit) % gpu_count] += 1;
        }
        return counts;
    }

    uint8_t modulus_coeff_bytes(uint64_t modulus)
    {
        if (modulus == 0)
        {
            throw std::runtime_error("zero modulus in limb metadata");
        }
        const uint32_t bit_width = static_cast<uint32_t>(64U - __builtin_clzll(modulus));
        return bit_width <= 32U ? 4U : 8U;
    }

    void build_limb_metadata(
        const std::vector<uint64_t> &moduli,
        size_t limb_count,
        size_t gpu_count,
        uint32_t dnum,
        std::vector<dim3> &limb_gpu_ids,
        std::vector<int> &limb_prime_ids,
        std::vector<GpuLimbType> &limb_types,
        std::vector<uint8_t> &limb_coeff_bytes)
    {
        if (moduli.size() < limb_count)
        {
            throw std::runtime_error("modulus metadata size mismatch in build_limb_metadata");
        }
        std::vector<uint32_t> next_local_index(gpu_count, 0);
        for (size_t limb = 0; limb < limb_count; ++limb)
        {
            const uint32_t digit = static_cast<uint32_t>(limb % static_cast<size_t>(dnum));
            const uint32_t partition = digit % static_cast<uint32_t>(gpu_count);
            const uint32_t local_index = next_local_index[partition]++;
            limb_gpu_ids[limb] = dim3(partition, local_index, 0);
            limb_prime_ids[limb] = static_cast<int>(limb);
            const uint8_t coeff_bytes = modulus_coeff_bytes(moduli[limb]);
            limb_coeff_bytes[limb] = coeff_bytes;
            limb_types[limb] = coeff_bytes <= 4 ? GPU_LIMB_U32 : GPU_LIMB_U64;
        }
    }

    void build_ntt_constants(
        const std::vector<uint64_t> &moduli,
        uint64_t n,
        std::vector<uint64_t> &n_inv_by_prime,
        std::vector<uint64_t> &root_by_prime,
        std::vector<uint64_t> &inv_root_by_prime)
    {
        const size_t count = moduli.size();
        n_inv_by_prime.assign(count, 0);
        root_by_prime.assign(count, 0);
        inv_root_by_prime.assign(count, 0);

        for (size_t i = 0; i < count; ++i)
        {
            const uint64_t modulus = moduli[i];
            if (modulus == 0)
            {
                throw std::runtime_error("zero modulus in gpu context creation");
            }

            uint64_t n_inv = 0;
            if (!mod_inverse_u64(n % modulus, modulus, n_inv))
            {
                throw std::runtime_error("failed to compute N inverse modulo prime");
            }

            const uint64_t root = compute_2nth_unity_root_u64(modulus, n);
            uint64_t inv_root = 0;
            if (!mod_inverse_u64(root % modulus, modulus, inv_root))
            {
                throw std::runtime_error("failed to compute inverse root modulo prime");
            }

            n_inv_by_prime[i] = n_inv;
            root_by_prime[i] = root;
            inv_root_by_prime[i] = inv_root;
        }
    }

    uint64_t parse_u64_or_default(const char *value, uint64_t default_value)
    {
        if (!value || value[0] == '\0')
        {
            return default_value;
        }
        errno = 0;
        char *end = nullptr;
        const unsigned long long parsed = std::strtoull(value, &end, 10);
        if (errno != 0 || !end || *end != '\0')
        {
            return default_value;
        }
        return static_cast<uint64_t>(parsed);
    }

    uint64_t mempool_release_threshold_bytes()
    {
        const uint64_t default_threshold = std::numeric_limits<uint64_t>::max();
        const char *env = std::getenv("MXX_CUDA_MEMPOOL_RELEASE_THRESHOLD_BYTES");
        return parse_u64_or_default(env, default_threshold);
    }

    void configure_default_mempool_release_threshold(const std::vector<int> &gpu_list)
    {
        uint64_t threshold = mempool_release_threshold_bytes();
        for (int device : gpu_list)
        {
            gpuMemPool_t pool = nullptr;
            gpuError_t err = gpuDeviceGetDefaultMemPool(&pool, mxx_physical_device(device));
            if (err != gpuSuccess)
            {
                throw std::runtime_error(gpuGetErrorString(err));
            }
            err = gpuMemPoolSetAttribute(pool, gpuMemPoolAttrReleaseThreshold, &threshold);
            if (err != gpuSuccess)
            {
                throw std::runtime_error(gpuGetErrorString(err));
            }
        }
    }

    // Cross-device graph copies go peer to peer where the hardware allows it:
    // each context's GPU gets access to every other physical GPU and to its
    // stream-ordered pool. Logical devices sharing one GPU need nothing, and
    // copies between GPUs without peer access are staged through host memory
    // (mxx_gpu_graph_builder_add_device_copy).
    void enable_peer_access(const std::vector<int> &gpu_list)
    {
        static std::mutex mutex;
        static std::set<std::pair<int, int>> enabled;
        std::lock_guard<std::mutex> lock(mutex);
        int physical_count = 0;
        if (gpuGetDeviceCount(&physical_count) != gpuSuccess)
        {
            (void)gpuGetLastError();
            return;
        }
        int current = 0;
        if (gpuGetDevice(&current) != gpuSuccess) current = -1;
        for (int device : gpu_list)
        {
            const int physical = mxx_physical_device(device);
            for (int peer = 0; peer < physical_count; ++peer)
            {
                int accessible = 0;
                if (peer == physical || enabled.count({physical, peer}) ||
                    gpuDeviceCanAccessPeer(&accessible, physical, peer) != gpuSuccess ||
                    !accessible)
                    continue;
                gpuMemPool_t pool = nullptr;
                gpuMemAccessDesc access{};
                access.location.type = gpuMemLocationTypeDevice;
                access.location.id = physical;
                access.flags = gpuMemAccessFlagsProtReadWrite;
                if (gpuSetDevice(physical) == gpuSuccess)
                {
                    const gpuError_t err = gpuDeviceEnablePeerAccess(peer, 0);
                    if (err == gpuSuccess || err == gpuErrorPeerAccessAlreadyEnabled)
                        enabled.insert({physical, peer});
                }
                // The peer's pool becomes readable and writable from this GPU.
                if (gpuDeviceGetDefaultMemPool(&pool, peer) == gpuSuccess)
                    (void)gpuMemPoolSetAccess(pool, &access, 1);
                (void)gpuGetLastError();
            }
        }
        if (current >= 0) (void)gpuSetDevice(current);
    }

    // Hardware peer capability alone does not authorize stream-ordered pool
    // access. Query the owner's pool direction before choosing a peer kernel.
    bool gpu_peer_pool_accessible(int executor, int owner)
    {
        if (executor == owner) return true;
        int capable = 0;
        gpuMemPool_t pool = nullptr;
        gpuMemLocation location{};
        location.type = gpuMemLocationTypeDevice;
        location.id = executor;
        gpuMemAccessFlags access{};
        const bool allowed =
            gpuDeviceCanAccessPeer(&capable, executor, owner) == gpuSuccess && capable &&
            gpuDeviceGetDefaultMemPool(&pool, owner) == gpuSuccess &&
            gpuMemPoolGetAccess(&access, pool, &location) == gpuSuccess &&
            access == gpuMemAccessFlagsProtReadWrite;
        if (!allowed) (void)gpuGetLastError();
        return allowed;
    }

    GpuNttDeviceConstants make_empty_ntt_device_constants(
        int device,
        size_t limb_count,
        uint32_t ring_dimension)
    {
        GpuNttDeviceConstants out{};
        out.device = device;
        out.limb_count = limb_count;
        out.ring_dimension = ring_dimension;
        out.twiddle_forward = nullptr;
        out.twiddle_inverse = nullptr;
        out.twiddle_shoup_forward = nullptr;
        out.twiddle_shoup_inverse = nullptr;
        out.moduli = nullptr;
        out.n_inv = nullptr;
        out.n_inv_shoup = nullptr;
        return out;
    }

    void free_ntt_device_constants_entry(GpuNttDeviceConstants &entry, gpuStream_t release = nullptr)
    {
        if (entry.device < 0)
        {
            return;
        }
        if (mxx_set_device(entry.device) != gpuSuccess)
        {
            return;
        }
        void *pointers[] = {entry.twiddle_forward, entry.twiddle_inverse,
            entry.twiddle_shoup_forward, entry.twiddle_shoup_inverse,
            entry.moduli, entry.n_inv, entry.n_inv_shoup};
        for (void *pointer : pointers)
        {
            if (pointer)
            {
                const gpuError_t status = release ? gpuFreeAsync(pointer, release) : gpuFree(pointer);
                if (status != gpuSuccess)
                {
                    set_error(status);
                }
            }
        }
        entry.twiddle_forward = nullptr;
        entry.twiddle_inverse = nullptr;
        entry.twiddle_shoup_forward = nullptr;
        entry.twiddle_shoup_inverse = nullptr;
        entry.moduli = nullptr;
        entry.n_inv = nullptr;
        entry.n_inv_shoup = nullptr;
    }

    void free_ntt_device_constants(std::vector<GpuNttDeviceConstants> &entries,
        const GpuExecutionOwner *owner = nullptr)
    {
        for (auto &entry : entries)
        {
            gpuStream_t release = nullptr;
            if (owner)
            {
                auto found = std::find(owner->gpu_ids.begin(), owner->gpu_ids.end(), entry.device);
                if (found == owner->gpu_ids.end())
                {
                    set_error("ring constant device is outside its execution owner");
                    continue;
                }
                release = owner->release_streams_by_partition[found - owner->gpu_ids.begin()];
            }
            free_ntt_device_constants_entry(entry, release);
        }
        entries.clear();
    }

    void upload_ntt_small_constants_to_device(
        int device,
        const std::vector<uint64_t> &limb_moduli,
        const std::vector<uint64_t> &limb_n_inv,
        const std::vector<uint64_t> &limb_n_inv_shoup,
        GpuNttDeviceConstants *out_entry)
    {
        const size_t limb_count = limb_moduli.size();
        if (limb_count == 0 || limb_count > GPU_RUNTIME_MAX_LIMBS)
        {
            throw std::runtime_error("invalid limb count in upload_ntt_small_constants_to_device");
        }
        if (limb_n_inv.size() != limb_count ||
            limb_n_inv_shoup.size() != limb_count)
        {
            throw std::runtime_error("inconsistent limb constants in upload_ntt_small_constants_to_device");
        }

        if (!out_entry)
        {
            throw std::runtime_error("null output entry in upload_ntt_small_constants_to_device");
        }
        gpuError_t err = mxx_set_device(device);
        if (err != gpuSuccess)
        {
            throw std::runtime_error(gpuGetErrorString(err));
        }
        const size_t limb_bytes = limb_count * sizeof(uint64_t);
        auto alloc_and_copy = [&](uint64_t **dst, const std::vector<uint64_t> &src)
        {
            err = gpuMalloc(reinterpret_cast<void **>(dst), limb_bytes);
            if (err != gpuSuccess) throw std::runtime_error(gpuGetErrorString(err));
            err = gpuMemcpy(*dst, src.data(), limb_bytes, gpuMemcpyHostToDevice);
            if (err != gpuSuccess) throw std::runtime_error(gpuGetErrorString(err));
        };
        try
        {
            alloc_and_copy(&out_entry->moduli, limb_moduli);
            alloc_and_copy(&out_entry->n_inv, limb_n_inv);
            alloc_and_copy(&out_entry->n_inv_shoup, limb_n_inv_shoup);
        }
        catch (...)
        {
            free_ntt_device_constants_entry(*out_entry);
            throw;
        }
    }

    void upload_ntt_twiddles_to_device(
        int device,
        const std::vector<uint64_t> &twiddle_forward,
        const std::vector<uint64_t> &twiddle_inverse,
        const std::vector<uint64_t> &twiddle_shoup_forward,
        const std::vector<uint64_t> &twiddle_shoup_inverse,
        GpuNttDeviceConstants *out_entry)
    {
        if (!out_entry)
        {
            throw std::runtime_error("null output entry in upload_ntt_twiddles_to_device");
        }
        const size_t limb_count = out_entry->limb_count;
        const size_t ring_dimension = out_entry->ring_dimension;
        if (limb_count == 0 || ring_dimension == 0)
        {
            throw std::runtime_error("invalid NTT constants shape in upload_ntt_twiddles_to_device");
        }
        const size_t twiddle_count = limb_count * ring_dimension;
        if (twiddle_forward.size() != twiddle_count ||
            twiddle_inverse.size() != twiddle_count ||
            twiddle_shoup_forward.size() != twiddle_count ||
            twiddle_shoup_inverse.size() != twiddle_count)
        {
            throw std::runtime_error("inconsistent twiddle constants in upload_ntt_twiddles_to_device");
        }

        gpuError_t err = mxx_set_device(device);
        if (err != gpuSuccess)
        {
            throw std::runtime_error(gpuGetErrorString(err));
        }

        const size_t twiddle_bytes = twiddle_count * sizeof(uint64_t);
        auto alloc_and_copy = [&](uint64_t **dst, const uint64_t *src)
        {
            gpuError_t local_err = gpuMalloc(reinterpret_cast<void **>(dst), twiddle_bytes);
            if (local_err != gpuSuccess)
            {
                throw std::runtime_error(gpuGetErrorString(local_err));
            }
            local_err = gpuMemcpy(*dst, src, twiddle_bytes, gpuMemcpyHostToDevice);
            if (local_err != gpuSuccess)
            {
                throw std::runtime_error(gpuGetErrorString(local_err));
            }
        };

        try
        {
            alloc_and_copy(&out_entry->twiddle_forward, twiddle_forward.data());
            alloc_and_copy(&out_entry->twiddle_inverse, twiddle_inverse.data());
            alloc_and_copy(&out_entry->twiddle_shoup_forward, twiddle_shoup_forward.data());
            alloc_and_copy(&out_entry->twiddle_shoup_inverse, twiddle_shoup_inverse.data());
        }
        catch (...)
        {
            free_ntt_device_constants_entry(*out_entry);
            throw;
        }
    }

    // Default-stream copies need an explicit edge to nonblocking owner
    // streams. This covers both pageable H2D staging and completed D2H reads
    // whose source will subsequently be reclaimed through an async free.
    gpuError_t order_default_stream_completion(const GpuExecutionOwner &owner, int device)
    {
        const auto found = std::find(owner.gpu_ids.begin(), owner.gpu_ids.end(), device);
        if (found == owner.gpu_ids.end())
            return gpuErrorInvalidDevice;
        gpuEvent_t copied = nullptr;
        gpuError_t error = mxx_set_device(device);
        if (error == gpuSuccess)
            error = gpuEventCreateWithFlags(&copied, gpuEventDisableTiming);
        if (error == gpuSuccess) error = gpuEventRecord(copied, nullptr);
        const auto wait = [&](gpuStream_t stream) {
            if (error == gpuSuccess && stream)
                error = gpuStreamWaitEvent(stream, copied, 0);
        };
        // A borrowed raw address does not identify which compute stream owns
        // its allocation. Protect every possible reclamation stream of this
        // execution owner on the same physical device, including logical
        // aliases. Other physical devices and execution owners remain free.
        for (size_t partition = 0; partition < owner.gpu_ids.size(); ++partition)
        {
            if (mxx_physical_device(owner.gpu_ids[partition]) != mxx_physical_device(device)) continue;
            for (gpuStream_t stream : owner.compute_streams_by_partition[partition]) wait(stream);
            wait(owner.transfer_streams_by_partition[partition]);
            wait(owner.release_streams_by_partition[partition]);
        }
        if (copied)
        {
            // Queued waits retain the event's recorded completion even after
            // handle destruction; keep the first error if cleanup also fails.
            const gpuError_t destroyed = gpuEventDestroy(copied);
            if (error == gpuSuccess) error = destroyed;
        }
        return error;
    }
}

GpuExecutionOwner::~GpuExecutionOwner()
{
    destroy_context_streams(this);
    if (registered)
    {
        for (int device : gpu_ids)
        {
            live_context_counts[static_cast<size_t>(device)].fetch_sub(1, std::memory_order_relaxed);
        }
    }
}

extern "C" int gpu_set_last_error(const char *msg)
{
    return set_error(msg);
}

extern "C" int gpu_set_last_error_runtime(int runtime_error)
{
    return set_error(static_cast<gpuError_t>(runtime_error));
}

extern "C"
{
    int gpu_context_create(
        uint32_t logN,
        uint32_t L,
        uint32_t dnum,
        const uint64_t *moduli,
        size_t moduli_len,
        const int *gpu_ids,
        size_t gpu_ids_len,
        size_t stream_pool_size,
        const GpuContext *related_context,
        GpuContext **out_ctx)
    {
        GpuContext *gpu_ctx = nullptr;
        try
        {
            if (!out_ctx || !moduli || moduli_len == 0 || stream_pool_size == 0)
            {
                return set_error("invalid context arguments");
            }
            *out_ctx = nullptr;
            if (moduli_len != static_cast<size_t>(L + 1))
            {
                return set_error("moduli_len must equal L + 1");
            }
            if (moduli_len > GPU_RUNTIME_MAX_LIMBS)
            {
                return set_error("moduli_len exceeds supported maximum");
            }
            if (logN == 0 || logN >= 31)
            {
                return set_error("logN must be between 1 and 30");
            }

            std::vector<int> gpu_list;
            if (gpu_ids_len == 0 || !gpu_ids)
            {
                gpu_list.push_back(0);
            }
            else
            {
                gpu_list.assign(gpu_ids, gpu_ids + gpu_ids_len);
            }

            validate_gpu_list(gpu_list);
            if (related_context &&
                (!related_context->execution || related_context->gpu_ids != gpu_list))
            {
                return set_error("related GPU rings must share device placement");
            }
            configure_default_mempool_release_threshold(gpu_list);
            enable_peer_access(gpu_list);
            const uint32_t resolved_dnum =
                dnum == 0 ? static_cast<uint32_t>(gpu_list.size()) : dnum;
            if (resolved_dnum == 0 || resolved_dnum > GPU_RUNTIME_MAX_DIGITS)
            {
                return set_error("invalid dnum in context creation");
            }

            std::vector<uint64_t> moduli_vec(moduli, moduli + moduli_len);
            std::vector<uint64_t> inverse_table =
                compute_garner_inverse_table(moduli_vec, static_cast<int>(moduli_len));
            const uint64_t n_u64 = uint64_t{1} << logN;

            std::vector<uint64_t> n_inv_by_prime;
            std::vector<uint64_t> root_by_prime;
            std::vector<uint64_t> inv_root_by_prime;
            build_ntt_constants(moduli_vec, n_u64, n_inv_by_prime, root_by_prime, inv_root_by_prime);

            const size_t limb_count = moduli_len;
            std::vector<dim3> limb_gpu_ids(limb_count, dim3{0, 0, 0});
            std::vector<int> limb_prime_ids(limb_count, -1);
            std::vector<GpuLimbType> limb_types(limb_count, GPU_LIMB_U64);
            std::vector<uint8_t> limb_coeff_bytes(limb_count, 0);
            build_limb_metadata(
                moduli_vec,
                limb_count,
                gpu_list.size(),
                resolved_dnum,
                limb_gpu_ids,
                limb_prime_ids,
                limb_types,
                limb_coeff_bytes);

            std::vector<size_t> decomp_counts_by_partition =
                compute_decomp_counts_by_partition(gpu_list.size(), resolved_dnum);

            std::vector<uint64_t> limb_moduli(limb_count, 0);
            std::vector<uint64_t> limb_root(limb_count, 0);
            std::vector<uint64_t> limb_inv_root(limb_count, 0);
            std::vector<uint64_t> limb_n_inv(limb_count, 0);
            std::vector<uint64_t> limb_n_inv_shoup(limb_count, 0);
            const size_t twiddle_count = limb_count * static_cast<size_t>(n_u64);
            std::vector<uint64_t> twiddle_forward(twiddle_count, 0);
            std::vector<uint64_t> twiddle_inverse(twiddle_count, 0);
            std::vector<uint64_t> twiddle_shoup_forward(twiddle_count, 0);
            std::vector<uint64_t> twiddle_shoup_inverse(twiddle_count, 0);
            for (size_t limb_idx = 0; limb_idx < limb_count; ++limb_idx)
            {
                const int primeid = limb_prime_ids[limb_idx];
                if (primeid < 0 || static_cast<size_t>(primeid) >= moduli_vec.size())
                {
                    throw std::runtime_error("invalid prime id in context creation");
                }
                const size_t prime_idx = static_cast<size_t>(primeid);
                const uint64_t modulus = moduli_vec[prime_idx];
                const uint64_t root = root_by_prime[prime_idx];
                const uint64_t inv_root = inv_root_by_prime[prime_idx];
                const uint64_t n_inv = n_inv_by_prime[prime_idx];
                limb_moduli[limb_idx] = modulus;
                limb_root[limb_idx] = root;
                limb_inv_root[limb_idx] = inv_root;
                limb_n_inv[limb_idx] = n_inv;
                limb_n_inv_shoup[limb_idx] = shoup_reciprocal_u64_host(n_inv, modulus);

                uint64_t forward_power = 1;
                uint64_t inverse_power = 1;
                const size_t limb_offset = limb_idx * static_cast<size_t>(n_u64);
                for (uint64_t exponent = 0; exponent < n_u64; ++exponent)
                {
                    const size_t offset = limb_offset + static_cast<size_t>(exponent);
                    twiddle_forward[offset] = forward_power;
                    twiddle_inverse[offset] = inverse_power;
                    twiddle_shoup_forward[offset] =
                        shoup_reciprocal_u64_host(forward_power, modulus);
                    twiddle_shoup_inverse[offset] =
                        shoup_reciprocal_u64_host(inverse_power, modulus);
                    forward_power = mul_mod_u64_host(forward_power, root, modulus);
                    inverse_power = mul_mod_u64_host(inverse_power, inv_root, modulus);
                }
            }

            gpu_ctx = new GpuContext();
            gpu_ctx->execution = related_context ? related_context->execution :
                std::make_shared<GpuExecutionOwner>();
            if (!related_context)
            {
                static std::atomic<uint64_t> next_identity{1};
                uint64_t identity = next_identity.load(std::memory_order_relaxed);
                do
                {
                    if (identity == std::numeric_limits<uint64_t>::max())
                    {
                        throw std::runtime_error("GPU execution identity space exhausted");
                    }
                } while (!next_identity.compare_exchange_weak(identity, identity + 1,
                    std::memory_order_relaxed));
                gpu_ctx->execution->identity = identity;
                gpu_ctx->execution->gpu_ids = gpu_list;
                gpu_ctx->execution->pinned_host_reclaimer = new PinnedHostReclaimer();
            }
            gpu_ctx->moduli = std::move(moduli_vec);
            gpu_ctx->barrett_reciprocals.reserve(gpu_ctx->moduli.size());
            for (const uint64_t modulus : gpu_ctx->moduli)
            {
                unsigned __int128 reciprocal = 0;
                if (modulus > 1 && modulus < (uint64_t{1} << 63))
                {
                    const unsigned __int128 maximum = ~static_cast<unsigned __int128>(0);
                    // floor(2^128/q), without representing 2^128 itself.
                    reciprocal = maximum / modulus + (maximum % modulus == modulus - 1);
                }
                gpu_ctx->barrett_reciprocals.push_back({
                    static_cast<uint64_t>(reciprocal), static_cast<uint64_t>(reciprocal >> 64)});
            }
            gpu_ctx->ntt_n_inv_by_prime = std::move(n_inv_by_prime);
            gpu_ctx->ntt_root_by_prime = std::move(root_by_prime);
            gpu_ctx->ntt_inv_root_by_prime = std::move(inv_root_by_prime);
            gpu_ctx->N = static_cast<int>(n_u64);
            gpu_ctx->level = static_cast<int>(L);
            gpu_ctx->gpu_ids = std::move(gpu_list);
            gpu_ctx->dnum = resolved_dnum;
            gpu_ctx->max_aux_limbs = GPU_RUNTIME_MAX_LIMBS;
            gpu_ctx->garner_inverse_table = std::move(inverse_table);
            gpu_ctx->limb_gpu_ids = std::move(limb_gpu_ids);
            gpu_ctx->limb_prime_ids = std::move(limb_prime_ids);
            gpu_ctx->limb_types = std::move(limb_types);
            gpu_ctx->limb_coeff_bytes = std::move(limb_coeff_bytes);
            gpu_ctx->decomp_counts_by_partition = std::move(decomp_counts_by_partition);
            if (!related_context)
            {
            gpu_ctx->execution->compute_streams_by_partition.resize(gpu_ctx->gpu_ids.size());
            gpu_ctx->execution->release_streams_by_partition.resize(gpu_ctx->gpu_ids.size(), nullptr);
            gpu_ctx->execution->transfer_streams_by_partition.resize(gpu_ctx->gpu_ids.size(), nullptr);
            for (size_t partition = 0; partition < gpu_ctx->gpu_ids.size(); ++partition)
            {
                const int device = gpu_ctx->gpu_ids[partition];
                gpuError_t err = mxx_set_device(device);
                if (err != gpuSuccess)
                {
                    throw std::runtime_error(gpuGetErrorString(err));
                }
                auto &streams = gpu_ctx->execution->compute_streams_by_partition[partition];
                streams.resize(stream_pool_size, nullptr);
                for (gpuStream_t &stream : streams)
                {
                    err = gpuStreamCreateWithFlags(&stream, gpuStreamNonBlocking);
                    if (err != gpuSuccess)
                    {
                        throw std::runtime_error(gpuGetErrorString(err));
                    }
                }
                err = gpuStreamCreateWithFlags(
                    &gpu_ctx->execution->release_streams_by_partition[partition],
                    gpuStreamNonBlocking);
                if (err != gpuSuccess)
                {
                    throw std::runtime_error(gpuGetErrorString(err));
                }
                err = gpuStreamCreateWithFlags(
                    &gpu_ctx->execution->transfer_streams_by_partition[partition],
                    gpuStreamNonBlocking);
                if (err != gpuSuccess)
                {
                    throw std::runtime_error(gpuGetErrorString(err));
                }
            }
            }
            gpu_ctx->ntt_device_constants.reserve(gpu_ctx->gpu_ids.size());
            for (int device : gpu_ctx->gpu_ids)
            {
                gpu_ctx->ntt_device_constants.push_back(
                    make_empty_ntt_device_constants(
                        device,
                        limb_count,
                        static_cast<uint32_t>(n_u64)));
                // Publish the owner before uploads so every exception path
                // can release partially initialized constants.
                auto &device_constants = gpu_ctx->ntt_device_constants.back();
                upload_ntt_small_constants_to_device(
                    device,
                    limb_moduli,
                    limb_n_inv,
                    limb_n_inv_shoup,
                    &device_constants);
                upload_ntt_twiddles_to_device(
                    device,
                    twiddle_forward,
                    twiddle_inverse,
                    twiddle_shoup_forward,
                    twiddle_shoup_inverse,
                    &device_constants);
                const gpuError_t ordered = order_default_stream_completion(*gpu_ctx->execution, device);
                if (ordered != gpuSuccess) throw std::runtime_error(gpuGetErrorString(ordered));
            }
            for (int device : gpu_ctx->gpu_ids)
            {
                if (device < 0 || static_cast<size_t>(device) >= MAX_TRACKED_GPU_DEVICES)
                {
                    throw std::runtime_error("GPU device ordinal exceeds live-context counter capacity");
                }
            }
            for (int device : gpu_ctx->gpu_ids)
            {
                if (!related_context)
                {
                    live_context_counts[static_cast<size_t>(device)].fetch_add(
                        1, std::memory_order_relaxed);
                }
                context_generations[static_cast<size_t>(device)].fetch_add(
                    1,
                    std::memory_order_release);
            }
            if (!related_context)
            {
                gpu_ctx->execution->registered = true;
            }
            *out_ctx = gpu_ctx;
            return 0;
        }
        catch (const std::exception &e)
        {
            if (gpu_ctx)
            {
                free_ntt_device_constants(gpu_ctx->ntt_device_constants);
                delete gpu_ctx;
            }
            return set_error(e);
        }
        catch (...)
        {
            if (gpu_ctx)
            {
                free_ntt_device_constants(gpu_ctx->ntt_device_constants);
                delete gpu_ctx;
            }
            return set_error("unknown exception in gpu_context_create");
        }
    }

    void gpu_context_destroy(GpuContext *ctx)
    {
        if (!ctx)
        {
            return;
        }
        const std::vector<int> gpu_ids = ctx->gpu_ids;
        free_ntt_device_constants(ctx->ntt_device_constants, ctx->execution.get());
        delete ctx;
        for (int device : gpu_ids)
        {
            context_generations[static_cast<size_t>(device)].fetch_add(
                1,
                std::memory_order_release);
        }
    }

    uint64_t gpu_context_execution_identity(const GpuContext *ctx)
    {
        return ctx ? ctx->execution->identity : 0;
    }

    int gpu_context_fence_releases(const GpuContext *ctx)
    {
        if (!ctx)
        {
            return set_error("invalid gpu_context_fence_releases arguments");
        }
        const int stream_status = fence_release_streams(ctx->execution.get());
        const int reclaimer_status = wait_pinned_host_reclaimer(ctx->execution.get());
        if (stream_status != 0)
        {
            return stream_status;
        }
        return reclaimer_status;
    }

    int gpu_defer_pinned_frees(
        GpuContext *ctx,
        int device,
        gpuStream_t stream,
        void *const *ptrs,
        size_t count)
    {
        if (!ctx || !ctx->execution->pinned_host_reclaimer || device < 0 ||
            (count != 0 && !ptrs))
        {
            return set_error("invalid gpu_defer_pinned_frees arguments");
        }
        if (count == 0)
        {
            return 0;
        }

        std::vector<void *> pointers;
        try
        {
            pointers.reserve(count);
            for (size_t index = 0; index < count; ++index)
            {
                if (ptrs[index])
                {
                    pointers.push_back(ptrs[index]);
                }
            }
        }
        catch (const std::exception &error)
        {
            ctx->execution->pinned_host_reclaimer->record_uncertain(error.what());
            return set_error(error);
        }
        if (pointers.empty())
        {
            return 0;
        }

        gpuError_t error = mxx_set_device(device);
        gpuEvent_t completion = nullptr;
        if (error == gpuSuccess)
        {
            error = gpuEventCreateWithFlags(&completion, gpuEventDisableTiming);
        }
        if (error == gpuSuccess)
        {
            error = gpuEventRecord(completion, stream);
        }
        if (error != gpuSuccess)
        {
            if (completion)
            {
                // The event may have been recorded before the error was
                // reported. Keep it leaked with the pointers rather than
                // destroying an event that could still be in flight.
                completion = nullptr;
            }
            ctx->execution->pinned_host_reclaimer->record_uncertain(gpuGetErrorString(error));
            return set_error(error);
        }

        const int enqueue_status =
            ctx->execution->pinned_host_reclaimer->enqueue(device, completion, std::move(pointers));
        if (enqueue_status != 0)
        {
            // Enqueue retains ownership on success. On failure, the event and
            // pointers intentionally remain leaked because their last
            // asynchronous use cannot be proven complete.
            return set_error("failed to enqueue deferred pinned-host free");
        }
        return 0;
    }

    int gpu_context_get_N(const GpuContext *ctx, int *out_N)
    {
        if (!ctx || !out_N)
        {
            return set_error("invalid gpu_context_get_N arguments");
        }
        *out_N = ctx->N;
        return 0;
    }

    // Graph-reserved physical memory currently mapped on `device`.
    int gpu_device_graph_memory_reserved(int device, size_t *out_reserved_bytes)
    {
        if (device < 0 || !out_reserved_bytes)
            return set_error("invalid gpu_device_graph_memory_reserved arguments");
        uint64_t reserved = 0;
        const gpuError_t err = gpuDeviceGetGraphMemAttribute(
            mxx_physical_device(device), gpuGraphMemAttrReservedMemCurrent, &reserved);
        if (err != gpuSuccess) return set_error(err);
        *out_reserved_bytes = static_cast<size_t>(reserved);
        return 0;
    }

    // Complete the work and pending frees of `ctx` on `device`, then return
    // the physical memory the Graph pool, the default pool, and the driver's
    // cache of destroyed Graph executables retain, so either pool can satisfy
    // the next allocation. Planning only: the host waits for this context's
    // streams on the device, not for other work on the GPU.
    int gpu_device_release_cached_memory(const GpuContext *ctx, int device)
    {
        if (!ctx || !ctx->execution || device < 0)
            return set_error("invalid gpu_device_release_cached_memory arguments");
        const GpuExecutionOwner &owner = *ctx->execution;
        const auto found = std::find(owner.gpu_ids.begin(), owner.gpu_ids.end(), device);
        if (found == owner.gpu_ids.end())
            return set_error("device is not owned by the GPU context");
        const size_t partition = static_cast<size_t>(found - owner.gpu_ids.begin());
        std::vector<gpuStream_t> streams;
        if (partition < owner.compute_streams_by_partition.size())
            streams = owner.compute_streams_by_partition[partition];
        if (partition < owner.release_streams_by_partition.size() &&
            owner.release_streams_by_partition[partition])
            streams.push_back(owner.release_streams_by_partition[partition]);
        const int physical = mxx_physical_device(device);
        gpuError_t err = mxx_set_device(device);
        // Synchronize the streams, not events recorded on them. Both wait
        // for the same work, including queued stream-ordered frees. Compute
        // Sanitizer, however, treats only a stream synchronization as
        // completing those frees. After an event wait, the whole-device
        // probe below makes the driver release destroyed Graph executables.
        // The sanitizer then reports each of their kernels as using the
        // freed buffers after the free.
        for (gpuStream_t stream : streams)
            if (err == gpuSuccess) err = gpuStreamSynchronize(stream);
        if (err == gpuSuccess) err = gpuDeviceGraphMemTrim(physical);
        gpuMemPool_t pool = nullptr;
        if (err == gpuSuccess) err = gpuDeviceGetDefaultMemPool(&pool, physical);
        if (err == gpuSuccess) err = gpuMemPoolTrimTo(pool, 0);
        if (err != gpuSuccess) return set_error(err);
        // The driver caches the device memory of destroyed Graph executables
        // (several KiB per kernel node) and returns it only to gpuMalloc,
        // not to pool or Graph allocations. A request for the whole device
        // cannot succeed, but makes the driver release that cache first.
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        err = gpuMemGetInfo(&free_bytes, &total_bytes);
        if (err != gpuSuccess) return set_error(err);
        void *probe = nullptr;
        if (gpuMalloc(&probe, total_bytes) == gpuSuccess)
        {
            err = gpuFree(probe);
            if (err != gpuSuccess) return set_error(err);
        }
        else
        {
            (void)gpuGetLastError();
        }
        return 0;
    }

    int gpu_default_mempool_get_usage(
        int device,
        size_t *out_used_current_bytes,
        size_t *out_used_high_bytes,
        size_t *out_reserved_current_bytes)
    {
        if (device < 0 || !out_used_current_bytes || !out_used_high_bytes ||
            !out_reserved_current_bytes)
        {
            return set_error("invalid gpu_default_mempool_get_usage arguments");
        }
        gpuMemPool_t pool = nullptr;
        gpuError_t err = gpuDeviceGetDefaultMemPool(&pool, mxx_physical_device(device));
        if (err != gpuSuccess)
        {
        return set_error(err);
        }
        uint64_t used_current = 0;
        uint64_t used_high = 0;
        uint64_t reserved_current = 0;
        err = gpuMemPoolGetAttribute(pool, gpuMemPoolAttrUsedMemCurrent, &used_current);
        if (err != gpuSuccess)
        {
        return set_error(err);
        }
        err = gpuMemPoolGetAttribute(pool, gpuMemPoolAttrUsedMemHigh, &used_high);
        if (err != gpuSuccess)
        {
        return set_error(err);
        }
        err = gpuMemPoolGetAttribute(pool, gpuMemPoolAttrReservedMemCurrent, &reserved_current);
        if (err != gpuSuccess)
        {
        return set_error(err);
        }
        if (used_current > std::numeric_limits<size_t>::max() ||
            used_high > std::numeric_limits<size_t>::max() ||
            reserved_current > std::numeric_limits<size_t>::max())
        {
            return set_error("default mempool usage does not fit size_t");
        }
        *out_used_current_bytes = static_cast<size_t>(used_current);
        *out_used_high_bytes = static_cast<size_t>(used_high);
        *out_reserved_current_bytes = static_cast<size_t>(reserved_current);
        return 0;
    }

    int gpu_device_context_state(int device, size_t *out_count, uint64_t *out_generation)
    {
        if (device < 0 || static_cast<size_t>(device) >= MAX_TRACKED_GPU_DEVICES ||
            !out_count || !out_generation)
        {
            return set_error("invalid gpu_device_context_state arguments");
        }
        const size_t index = static_cast<size_t>(device);
        const uint64_t generation_before =
            context_generations[index].load(std::memory_order_acquire);
        const size_t count = live_context_counts[index].load(std::memory_order_acquire);
        const uint64_t generation_after =
            context_generations[index].load(std::memory_order_acquire);
        *out_count = generation_before == generation_after
            ? count
            : std::numeric_limits<size_t>::max();
        *out_generation = generation_after;
        return 0;
    }

    int gpu_device_get_identity(
        int device,
        char *out_name,
        size_t name_capacity,
        char *out_uuid,
        size_t uuid_capacity,
        int *out_compute_major,
        int *out_compute_minor,
        size_t *out_total_global_memory,
        int *out_driver_version,
        int *out_runtime_version,
        uint64_t *out_context_generation,
        char *out_backend, size_t backend_capacity,
        char *out_architecture, size_t architecture_capacity,
        int *out_wave_size)
    {
        if (device < 0 || !out_name || name_capacity == 0 || !out_compute_major ||
            !out_uuid || uuid_capacity < 33 || !out_compute_minor ||
            !out_total_global_memory || !out_driver_version || !out_runtime_version ||
            !out_context_generation || !out_backend || backend_capacity < 5 ||
            !out_architecture || architecture_capacity == 0 || !out_wave_size)
        {
            return set_error("invalid gpu_device_get_identity arguments");
        }
        gpuDeviceProp properties{};
        gpuError_t err = gpuGetDeviceProperties(&properties, mxx_physical_device(device));
        if (err != gpuSuccess)
        {
        return set_error(err);
        }
        const size_t source_length = strnlen(properties.name, sizeof(properties.name));
        const size_t copied = std::min(source_length, name_capacity - 1);
        memcpy(out_name, properties.name, copied);
        out_name[copied] = '\0';
        const auto *uuid_bytes = reinterpret_cast<const unsigned char *>(&properties.uuid);
        static constexpr char hex[] = "0123456789abcdef";
        for (size_t index = 0; index < sizeof(properties.uuid); ++index)
        {
            out_uuid[index * 2] = hex[(uuid_bytes[index] >> 4) & 0xf];
            out_uuid[index * 2 + 1] = hex[uuid_bytes[index] & 0xf];
        }
        out_uuid[sizeof(properties.uuid) * 2] = '\0';
        err = gpuDriverGetVersion(out_driver_version);
        if (err != gpuSuccess)
        {
            return set_error(err);
        }
        err = gpuRuntimeGetVersion(out_runtime_version);
        if (err != gpuSuccess)
        {
            return set_error(err);
        }
        size_t live_contexts = 0;
        if (gpu_device_context_state(device, &live_contexts, out_context_generation) != 0)
        {
            return 1;
        }
#if defined(MXX_GPU_BACKEND_HIP)
        snprintf(out_backend, backend_capacity, "hip");
        snprintf(out_architecture, architecture_capacity, "%s", properties.gcnArchName);
        *out_compute_major = 0;
        *out_compute_minor = 0;
#else
        snprintf(out_backend, backend_capacity, "cuda");
        snprintf(out_architecture, architecture_capacity, "sm_%d%d", properties.major, properties.minor);
        *out_compute_major = properties.major;
        *out_compute_minor = properties.minor;
#endif
        *out_wave_size = properties.warpSize;
        *out_total_global_memory = properties.totalGlobalMem;
        return 0;
    }

    int gpu_event_set_wait(GpuEventSet *events)
    {
        if (!events)
        {
            return set_error("invalid gpu_event_set_wait arguments");
        }
        for (const auto &entry : events->entries)
        {
            gpuError_t err = mxx_set_device(entry.device);
            if (err != gpuSuccess)
            {
                return set_error(err);
            }
            err = gpuEventSynchronize(entry.event);
            if (err != gpuSuccess)
            {
                return set_error(err);
            }
        }
        return 0;
    }

    void gpu_event_set_destroy(GpuEventSet *events)
    {
        destroy_event_set(events);
    }

    static std::vector<int> &logical_device_table()
    {
        static std::vector<int> table;
        return table;
    }

    static thread_local int current_logical_device = -1;

    int gpu_configure_logical_devices(const int *physical, size_t count)
    {
        if (!physical || count == 0)
            return set_error("invalid logical device table");
        int physical_count = 0;
        gpuError_t err = gpuGetDeviceCount(&physical_count);
        if (err != gpuSuccess) return set_error(err);
        for (size_t index = 0; index < count; ++index)
            if (physical[index] < 0 || physical[index] >= physical_count)
                return set_error("logical device maps to an absent physical device");
        auto &table = logical_device_table();
        if (!table.empty())
            return std::equal(table.begin(), table.end(), physical, physical + count) &&
                table.size() == count ? 0 : set_error("logical device table is already configured");
        table.assign(physical, physical + count);
        return 0;
    }

    int mxx_physical_device(int logical)
    {
        const auto &table = logical_device_table();
        if (table.empty() || logical < 0) return logical;
        return static_cast<size_t>(logical) < table.size() ? table[logical] : -1;
    }

    gpuError_t mxx_set_device(int logical)
    {
        const int physical = mxx_physical_device(logical);
        if (physical < 0) return gpuErrorInvalidDevice;
        const gpuError_t err = gpuSetDevice(physical);
        if (err == gpuSuccess) current_logical_device = logical;
        return err;
    }

    gpuError_t mxx_get_device(int *logical)
    {
        int physical = 0;
        const gpuError_t err = gpuGetDevice(&physical);
        if (err != gpuSuccess) return err;
        // The last logical device selected on this thread, when it is still
        // current; otherwise the first logical device of that physical one.
        if (current_logical_device >= 0 && mxx_physical_device(current_logical_device) == physical)
        {
            *logical = current_logical_device;
            return gpuSuccess;
        }
        const auto &table = logical_device_table();
        if (table.empty()) { *logical = physical; return gpuSuccess; }
        const auto found = std::find(table.begin(), table.end(), physical);
        if (found == table.end()) return gpuErrorInvalidDevice;
        *logical = static_cast<int>(found - table.begin());
        return gpuSuccess;
    }

    int gpu_device_count(int *out_count)
    {
        if (!out_count)
        {
            return set_error("invalid gpu_device_count arguments");
        }
        int count = 0;
        gpuError_t err = gpuGetDeviceCount(&count);
        if (err == gpuErrorNoDevice)
        {
            *out_count = 0;
            return 0;
        }
        if (err != gpuSuccess)
        {
            return set_error(err);
        }
        const auto &table = logical_device_table();
        *out_count = table.empty() ? count : static_cast<int>(table.size());
        return 0;
    }

    int gpu_device_mem_info(int device, size_t *out_free, size_t *out_total)
    {
        if (!out_free || !out_total)
        {
            return set_error("invalid gpu_device_mem_info arguments");
        }
        int current = 0;
        gpuError_t err = gpuGetDevice(&current);
        if (err != gpuSuccess)
        {
            return set_error(err);
        }
        err = mxx_set_device(device);
        if (err != gpuSuccess)
        {
            return set_error(err);
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        err = gpuMemGetInfo(&free_bytes, &total_bytes);
        gpuError_t restore_err = gpuSetDevice(current);
        if (err != gpuSuccess)
        {
            return set_error(err);
        }
        if (restore_err != gpuSuccess)
        {
            return set_error(restore_err);
        }
        *out_free = free_bytes;
        *out_total = total_bytes;
        return 0;
    }

    int gpu_device_synchronize()
    {
        gpuError_t err = gpuDeviceSynchronize();
        if (err != gpuSuccess)
        {
        return set_error(err);
        }
        return 0;
    }

    const char *gpu_last_error()
    {
        return last_error.c_str();
    }

    void *gpu_pinned_alloc(size_t bytes)
    {
        try
        {
            if (bytes == 0)
            {
                return nullptr;
            }
            void *ptr = nullptr;
            gpuError_t err = gpuMallocHost(&ptr, bytes);
            if (err != gpuSuccess)
            {
                set_error(err);
                return nullptr;
            }
            return ptr;
        }
        catch (const std::exception &e)
        {
            set_error(e);
            return nullptr;
        }
        catch (...)
        {
            set_error("unknown exception in gpu_pinned_alloc");
            return nullptr;
        }
    }

    void gpu_pinned_free(void *ptr)
    {
        if (!ptr)
        {
            return;
        }
        gpuError_t err = gpuFreeHost(ptr);
        if (err != gpuSuccess)
        {
        set_error(err);
        }
    }

    int gpu_export_slot_alloc(int physical_device, size_t payload_capacity,
        void **out_host, void **out_device)
    {
        if (!out_host || !out_device ||
            payload_capacity > SIZE_MAX - sizeof(MxxExportSlotHeader))
            return set_error("invalid export slot allocation arguments");
        *out_host = nullptr;
        *out_device = nullptr;
        const gpuError_t device_error = mxx_set_device(physical_device);
        if (device_error != gpuSuccess) return set_error(device_error);
        void *host = nullptr;
        gpuError_t error = gpuHostAlloc(&host,
            sizeof(MxxExportSlotHeader) + payload_capacity, gpuHostAllocMapped);
        if (error != gpuSuccess) return set_error(error);
        void *device = nullptr;
        error = gpuHostGetDevicePointer(&device, host, 0);
        if (error != gpuSuccess)
        {
            (void)gpuFreeHost(host);
            return set_error(error);
        }
        memset(host, 0, sizeof(MxxExportSlotHeader));
        *out_host = host;
        *out_device = device;
        return 0;
    }

    // An export slot whose payload lives in device memory: the Graph copy
    // stays on the GPU and only the small header is host-mapped for the
    // publication the I/O worker observes.
    int gpu_export_slot_alloc_device(int physical_device, size_t payload_capacity,
        void **out_host, void **out_device_header, void **out_device_payload)
    {
        if (!out_host || !out_device_header || !out_device_payload)
            return set_error("invalid device export slot allocation arguments");
        *out_host = nullptr;
        *out_device_header = nullptr;
        *out_device_payload = nullptr;
        gpuError_t error = mxx_set_device(physical_device);
        if (error != gpuSuccess) return set_error(error);
        void *host = nullptr;
        error = gpuHostAlloc(&host, sizeof(MxxExportSlotHeader), gpuHostAllocMapped);
        if (error != gpuSuccess) return set_error(error);
        void *device_header = nullptr;
        error = gpuHostGetDevicePointer(&device_header, host, 0);
        void *payload = nullptr;
        if (error == gpuSuccess && payload_capacity != 0)
            error = gpuMalloc(&payload, payload_capacity);
        if (error != gpuSuccess)
        {
            (void)gpuFreeHost(host);
            return set_error(error);
        }
        memset(host, 0, sizeof(MxxExportSlotHeader));
        *out_host = host;
        *out_device_header = device_header;
        *out_device_payload = payload;
        return 0;
    }

    int gpu_device_memory_alloc(int physical_device, size_t bytes, void **out)
    {
        if (!out || bytes == 0) return set_error("invalid device memory allocation arguments");
        *out = nullptr;
        gpuError_t error = mxx_set_device(physical_device);
        if (error == gpuSuccess) error = gpuMalloc(out, bytes);
        return error == gpuSuccess ? 0 : set_error(error);
    }

    int gpu_device_memory_free(int physical_device, void *address)
    {
        if (!address) return 0;
        gpuError_t error = mxx_set_device(physical_device);
        if (error == gpuSuccess) error = gpuFree(address);
        return error == gpuSuccess ? 0 : set_error(error);
    }

    // Copy between device allocations, possibly on different GPUs, and return
    // only after the copy has completed.
    int gpu_device_memory_copy(
        int destination_device, void *destination, const void *source, size_t bytes)
    {
        if (bytes == 0) return 0;
        if (!destination || !source) return set_error("invalid device memory copy arguments");
        gpuError_t error = mxx_set_device(destination_device);
        gpuStream_t stream = nullptr;
        if (error == gpuSuccess) error = gpuStreamCreateWithFlags(&stream, gpuStreamNonBlocking);
        if (error != gpuSuccess) return set_error(error);
        error = gpuMemcpyAsync(destination, source, bytes, gpuMemcpyDefault, stream);
        const gpuError_t synchronized = gpuStreamSynchronize(stream);
        if (error == gpuSuccess) error = synchronized;
        const gpuError_t destroyed = gpuStreamDestroy(stream);
        if (error == gpuSuccess) error = destroyed;
        return error == gpuSuccess ? 0 : set_error(error);
    }

    int gpu_device_strided_copy(
        int device, void *destination, const void *source, MxxStridedCopy copy)
    {
        if (!destination || !source || copy.element_bytes == 0)
            return set_error("invalid device strided copy arguments");
        gpuError_t error = mxx_set_device(device);
        gpuStream_t stream = nullptr;
        if (error == gpuSuccess) error = gpuStreamCreateWithFlags(&stream, gpuStreamNonBlocking);
        if (error != gpuSuccess) return set_error(error);
        constexpr uint32_t threads = 256;
        mxx_strided_copy_kernel<<<strided_copy_blocks(copy, threads), threads, 0, stream>>>(
            static_cast<uint8_t *>(destination), static_cast<const uint8_t *>(source), copy);
        error = gpuGetLastError();
        const gpuError_t synchronized = gpuStreamSynchronize(stream);
        if (error == gpuSuccess) error = synchronized;
        const gpuError_t destroyed = gpuStreamDestroy(stream);
        if (error == gpuSuccess) error = destroyed;
        return error == gpuSuccess ? 0 : set_error(error);
    }

    int gpu_device_memory_download(
        int physical_device, const void *source, void *destination, size_t bytes)
    {
        if (bytes == 0) return 0;
        if (!destination || !source) return set_error("invalid device memory download arguments");
        gpuError_t error = mxx_set_device(physical_device);
        gpuStream_t stream = nullptr;
        if (error == gpuSuccess) error = gpuStreamCreateWithFlags(&stream, gpuStreamNonBlocking);
        if (error != gpuSuccess) return set_error(error);
        error = gpuMemcpyAsync(destination, source, bytes, gpuMemcpyDeviceToHost, stream);
        const gpuError_t synchronized = gpuStreamSynchronize(stream);
        if (error == gpuSuccess) error = synchronized;
        const gpuError_t destroyed = gpuStreamDestroy(stream);
        if (error == gpuSuccess) error = destroyed;
        return error == gpuSuccess ? 0 : set_error(error);
    }

    int gpu_export_slot_ready(const void *host_header, int *out_ready)
    {
        if (!host_header || !out_ready) return set_error("invalid export slot ready query");
        const auto *header = static_cast<const MxxExportSlotHeader *>(host_header);
        *out_ready = __atomic_load_n(&header->ready, __ATOMIC_ACQUIRE) == 1ULL;
        return 0;
    }

    int gpu_export_slot_reset(void *host_header)
    {
        if (!host_header) return set_error("invalid export slot reset");
        auto *header = static_cast<MxxExportSlotHeader *>(host_header);
        // The caller has joined GPU completion and all host I/O readers.
        // Publish zero last so the next observer cannot see stale metadata.
        header->occurrence = 0;
        header->artifact_offset = 0;
        header->payload_bytes = 0;
        header->site = 0;
        header->flags = 0;
        __atomic_store_n(&header->ready, 0ULL, __ATOMIC_RELEASE);
        return 0;
    }

    int gpu_device_buffer_alloc(void *stream_raw, size_t bytes, MxxGpuDeviceBuffer **out)
    {
        if (!stream_raw || !out || bytes == 0)
            return set_error("invalid gpu_device_buffer_alloc arguments");
        *out = nullptr;
        auto *buffer = new (std::nothrow) MxxGpuDeviceBuffer();
        if (!buffer)
            return set_error("failed to allocate device buffer owner");
        buffer->allocation_stream = reinterpret_cast<gpuStream_t>(stream_raw);
        buffer->bytes = bytes;
        gpuError_t status = mxx_get_device(&buffer->device);
        if (status == gpuSuccess)
        {
            status = gpuMallocAsync(
                reinterpret_cast<void **>(&buffer->address), bytes, buffer->allocation_stream);
        }
        if (status == gpuSuccess)
            // A new buffer starts zeroed: pool memory may hold another owner's bytes, and a
            // plan may read bytes no operation wrote, such as the members of the waves an
            // I/O trial skips.
            status = gpuMemsetAsync(buffer->address, 0, bytes, buffer->allocation_stream);
        if (status == gpuSuccess)
        {
            status = gpuEventCreateWithFlags(&buffer->producer, gpuEventDisableTiming);
        }
        if (status == gpuSuccess)
        {
            status = gpuEventRecord(buffer->producer, buffer->allocation_stream);
        }
        if (status != gpuSuccess)
        {
            if (buffer->producer) (void)gpuEventDestroy(buffer->producer);
            if (buffer->address) (void)gpuFreeAsync(buffer->address, buffer->allocation_stream);
            delete buffer;
            return set_error(status);
        }
        buffer->producer_valid = true;
        *out = buffer;
        return 0;
    }

    int gpu_device_buffer_address(
        const MxxGpuDeviceBuffer *buffer,
        size_t offset,
        size_t bytes,
        void **out_address)
    {
        if (!buffer || !out_address || offset > buffer->bytes || bytes > buffer->bytes - offset)
            return set_error("invalid gpu_device_buffer_address arguments");
        *out_address = buffer->address + offset;
        return 0;
    }

    int gpu_device_buffer_free(MxxGpuDeviceBuffer *buffer)
    {
        if (!buffer)
            return 0;
        gpuError_t status = mxx_set_device(buffer->device);
        if (status == gpuSuccess && buffer->producer_valid) {
            status = gpuStreamWaitEvent(buffer->allocation_stream, buffer->producer, 0);
        }
        if (status == gpuSuccess) {
            status = gpuFreeAsync(buffer->address, buffer->allocation_stream);
        }
        if (buffer->producer)
        {
            const gpuError_t event_status = gpuEventDestroy(buffer->producer);
            if (status == gpuSuccess) status = event_status;
        }
        delete buffer;
        return status == gpuSuccess ? 0 : set_error(status);
    }

    int gpu_device_buffer_upload(
        MxxGpuDeviceBuffer *buffer,
        size_t offset,
        const void *source,
        size_t bytes)
    {
        if (!buffer || !source || offset > buffer->bytes || bytes > buffer->bytes - offset)
            return set_error("invalid gpu_device_buffer_upload arguments");
        // The buffer is pool memory of its own device, which only that
        // device may name as a copy operand.
        gpuError_t status = mxx_set_device(buffer->device);
        if (status == gpuSuccess)
            status = gpuMemcpyAsync(buffer->address + offset, source, bytes,
                gpuMemcpyHostToDevice, buffer->allocation_stream);
        if (status == gpuSuccess)
            status = gpuEventRecord(buffer->producer, buffer->allocation_stream);
        if (status != gpuSuccess)
            return set_error(status);
        buffer->producer_valid = true;
        return 0;
    }

    int gpu_device_buffer_download(
        const MxxGpuDeviceBuffer *buffer,
        size_t offset,
        void *destination,
        size_t bytes)
    {
        if (!buffer || !destination || offset > buffer->bytes ||
            bytes > buffer->bytes - offset)
        {
            return set_error("invalid gpu_device_buffer_download arguments");
        }
        gpuError_t status = mxx_set_device(buffer->device);
        if (status == gpuSuccess && buffer->producer_valid)
            status = gpuStreamWaitEvent(buffer->allocation_stream, buffer->producer, 0);
        if (status == gpuSuccess)
        {
            status = gpuMemcpyAsync(
                destination,
                buffer->address + offset,
                bytes,
                gpuMemcpyDeviceToHost,
                buffer->allocation_stream);
        }
        if (status == gpuSuccess)
            status = gpuStreamSynchronize(buffer->allocation_stream);
        return status == gpuSuccess ? 0 : set_error(status);
    }

    int gpu_context_download_address(
        GpuContext *ctx, int device, const void *address,
        void *destination, size_t bytes)
    {
        if (!ctx || !ctx->execution || !address || !destination || bytes == 0 ||
            std::find(ctx->gpu_ids.begin(), ctx->gpu_ids.end(), device) == ctx->gpu_ids.end())
            return set_error("invalid resident address download arguments");
        gpuError_t status = mxx_set_device(device);
        gpuPointerAttributes attributes{};
        if (status == gpuSuccess) status = gpuPointerGetAttributes(&attributes, address);
        if (status != gpuSuccess) return set_error(status);
        if (attributes.device != mxx_physical_device(device) || attributes.type != gpuMemoryTypeDevice)
            return set_error("resident download address belongs to another device or memory type");
        if (bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(address))
            return set_error("resident download address overflows");
        status = gpuMemcpy(destination, address, bytes, gpuMemcpyDeviceToHost);
        if (status == gpuSuccess)
            status = order_default_stream_completion(*ctx->execution, device);
        return status == gpuSuccess ? 0 : set_error(status);
    }

    gpuError_t gpu_join_retirement_stream(int owner_device, void *retirement_stream_raw,
        int consumer_device, void *consumer_stream_raw)
    {
        if (!retirement_stream_raw || !consumer_stream_raw) return gpuErrorInvalidValue;
        if (retirement_stream_raw == consumer_stream_raw) return gpuSuccess;
        int previous_device = -1;
        gpuError_t error = mxx_get_device(&previous_device);
        gpuEvent_t retired = nullptr;
        if (error == gpuSuccess) error = mxx_set_device(owner_device);
        if (error == gpuSuccess) error = gpuEventCreateWithFlags(&retired, gpuEventDisableTiming);
        if (error == gpuSuccess)
            error = gpuEventRecord(retired, reinterpret_cast<gpuStream_t>(retirement_stream_raw));
        if (error == gpuSuccess) error = mxx_set_device(consumer_device);
        if (error == gpuSuccess) {
            error = gpuStreamWaitEvent(reinterpret_cast<gpuStream_t>(consumer_stream_raw), retired, 0);
        }
        if (retired) {
            const gpuError_t selected = mxx_set_device(owner_device);
            const gpuError_t destroyed = selected == gpuSuccess ? gpuEventDestroy(retired) : selected;
            if (error == gpuSuccess) error = destroyed;
        }
        if (previous_device >= 0) {
            const gpuError_t restored = mxx_set_device(previous_device);
            if (error == gpuSuccess) error = restored;
        }
        return error;
    }

    int gpu_device_buffer_record_compiled_use(MxxGpuDeviceBuffer *buffer,
        int consumer_device, void *consumer_stream_raw, void *completion_event_raw, bool written)
    {
        if (!buffer || !consumer_stream_raw || !completion_event_raw || consumer_device < 0) {
            set_error("invalid compiled buffer completion protection");
            return GPU_STATUS_LAUNCH_UNCERTAIN;
        }
        int previous_device = -1;
        gpuError_t status = mxx_get_device(&previous_device);
        if (status == gpuSuccess) status = mxx_set_device(buffer->device);
        if (status == gpuSuccess)
            status = gpuStreamWaitEvent(buffer->allocation_stream,
                reinterpret_cast<gpuEvent_t>(completion_event_raw), 0);
        if (status == gpuSuccess && written) {
            if (mxx_physical_device(buffer->device) != mxx_physical_device(consumer_device))
                status = gpuErrorInvalidDevice;
            if (status == gpuSuccess) status = mxx_set_device(consumer_device);
            if (status == gpuSuccess)
                status = gpuEventRecord(buffer->producer, reinterpret_cast<gpuStream_t>(consumer_stream_raw));
            if (status == gpuSuccess) buffer->producer_valid = true;
        }
        if (previous_device >= 0) {
            const gpuError_t restored = mxx_set_device(previous_device);
            if (status == gpuSuccess) status = restored;
        }
        if (status != gpuSuccess) {
            set_error(status);
            return GPU_STATUS_LAUNCH_UNCERTAIN;
        }
        return 0;
    }

    int gpu_device_buffer_wait_compiled_inputs(
        const MxxGpuDeviceBuffer *buffer,
        int consumer_device,
        void *consumer_stream_raw,
        bool read_only)
    {
        if (!buffer || !consumer_stream_raw || consumer_device < 0)
            return set_error("invalid gpu_device_buffer_wait_compiled_inputs arguments");
        gpuError_t status = read_only ? gpuSuccess : gpu_join_retirement_stream(buffer->device,
            reinterpret_cast<void *>(buffer->allocation_stream), consumer_device, consumer_stream_raw);
        if (status == gpuSuccess) status = mxx_set_device(consumer_device);
        if (status == gpuSuccess && buffer->producer_valid)
        {
            status = gpuStreamWaitEvent(
                reinterpret_cast<gpuStream_t>(consumer_stream_raw), buffer->producer, 0);
        }
        return status == gpuSuccess ? 0 : set_error(status);
    }

    int gpu_device_buffer_wait(const MxxGpuDeviceBuffer *buffer)
    {
        if (!buffer || !buffer->producer_valid)
            return buffer ? 0 : set_error("invalid gpu_device_buffer_wait arguments");
        gpuError_t status = mxx_set_device(buffer->device);
        if (status == gpuSuccess)
            status = gpuEventSynchronize(buffer->producer);
        return status == gpuSuccess ? 0 : set_error(status);
    }

    int gpu_context_get_compute_stream(
        const GpuContext *ctx,
        int physical_device,
        void **out_stream)
    {
        if (!ctx || !ctx->execution || !out_stream)
        {
            return set_error("invalid gpu_context_get_compute_stream arguments");
        }
        const auto found = std::find(ctx->execution->gpu_ids.begin(),
            ctx->execution->gpu_ids.end(), physical_device);
        if (found == ctx->execution->gpu_ids.end())
        {
            return set_error("physical device is not owned by the GPU context");
        }
        const size_t partition = static_cast<size_t>(found - ctx->execution->gpu_ids.begin());
        if (partition >= ctx->execution->compute_streams_by_partition.size() ||
            ctx->execution->compute_streams_by_partition[partition].empty())
        {
            return set_error("GPU context has no compute stream for physical device");
        }
        // Callers allocate on the stream or add Graph nodes that do not select
        // a device themselves, so the stream's device becomes current.
        if (mxx_set_device(physical_device) != gpuSuccess)
            return set_error(gpuGetLastError());
        const size_t index = ctx->execution->next_compute_stream.fetch_add(
            1, std::memory_order_relaxed) %
            ctx->execution->compute_streams_by_partition[partition].size();
        *out_stream = reinterpret_cast<void *>(
            ctx->execution->compute_streams_by_partition[partition][index]);
        return 0;
    }

    struct GraphPatchRecord
    {
        MxxGraphPatch patch{};
    };

    struct GraphKernelUpdateRecord
    {
        gpuGraphNode_t node = nullptr;
        gpuKernelNodeParams launch{};
        std::vector<std::vector<uint8_t>> arguments;
        std::vector<void *> argument_pointers;
        std::vector<GraphPatchRecord> patches;
        // The executable node holds `arguments` (from creation on); a bind
        // that leaves every patched byte unchanged skips the node update.
        bool bound = false;
        // The (logical) device current at creation: a node's kernel function
        // resolves on that GPU, so every update selects it again.
        int device = -1;
    };

    struct GraphMemcpyUpdateRecord
    {
        gpuGraphNode_t node = nullptr;
        void *source = nullptr;
        void *destination = nullptr;
        size_t bytes = 0;
        gpuMemcpyKind kind = gpuMemcpyDefault;
        std::vector<GraphPatchRecord> patches;
        void *bound_source = nullptr;
        void *bound_destination = nullptr;
        bool bound = false;
        // The (logical) device made current to create and update the node,
        // for memory (a stream-ordered pool or graph allocation) only that
        // GPU may name; -1 uses the graph's device.
        int device = -1;
    };

    struct GraphMemsetUpdateRecord
    {
        gpuGraphNode_t node = nullptr;
        gpuMemsetParams params{};
        GraphPatchRecord patch{};
        bool bound = false;
        // The (logical) device current at creation, selected for updates.
        int device = -1;
    };


    // Writes every patch of `record` into its argument bytes and reports
    // whether any byte changed. Returns nonzero on an invalid patch.
    int patch_kernel_record(GraphKernelUpdateRecord &record,
        const MxxGraphBindingValue *values, size_t count, bool &changed);

#if defined(MXX_GPU_BACKEND_HIP)
extern "C++" {
#include "HipGraphSchedule.h"
}
#endif

    struct MxxGpuGraphExec
    {
        int device = -1;
        std::shared_ptr<GpuExecutionOwner> execution;
        gpuStream_t default_stream = nullptr;
        gpuGraph_t graph = nullptr;
        gpuGraphExec_t exec = nullptr;
#if !defined(MXX_GPU_BACKEND_HIP)
        // Private reusable records for top-level operation branches. A launch
        // waits on every record before producing its fresh public completion.
        std::vector<gpuEvent_t> branch_completion_events;
#endif
#if defined(MXX_GPU_BACKEND_HIP)
        std::unique_ptr<HipGraphSchedule> schedule;
#endif
        std::vector<GraphKernelUpdateRecord> kernels;
        std::vector<GraphMemcpyUpdateRecord> memcpys;
        std::vector<GraphMemsetUpdateRecord> memsets;
        // Child-graph records are patched before the conditional child graph
        // is rebound; dropping them would retain stale plan-time pointers.
        std::vector<GraphKernelUpdateRecord> body_kernels;
        std::vector<GraphMemcpyUpdateRecord> body_memcpys;
        std::vector<GraphMemsetUpdateRecord> body_memsets;
        // Pinned host buffers staging copies between GPUs without peer access.
        std::vector<void *> host_staging;
    };

    struct MxxGpuGraphBuilder
    {
        GpuContext *context = nullptr;
        int device = -1;
        gpuStream_t stream = nullptr;
        gpuGraph_t graph = nullptr;
        gpuGraph_t root_graph = nullptr;
        bool conditional_body_active = false;
        // The device of the innermost conditional body graph. CUDA requires
        // every node of a body graph to reside on one device, so a node added
        // there is created on this device.
        int body_device = -1;
        MxxPreimageRetrySpec retry_spec{};
        void *retry_scratch = nullptr;
        void *retry_control = nullptr;
        void *retry_status = nullptr;
        bool generic_body_mode = false;
        std::vector<gpuGraphNode_t> body_terminals;
        // Last conditional node emitted into the open conditional body.
        // Concurrent sibling conditionals inside one body graph do not
        // complete on the device, so each is ordered after the previous one.
        gpuGraphNode_t last_body_conditional = nullptr;
        // One frame per open conditional body, innermost last. A body may
        // itself contain conditional operations, so the enclosing scope's
        // graph, operation state and WHILE control are restored on finish.
        struct ConditionalFrame
        {
            gpuGraph_t parent_graph = nullptr;
            gpuGraphNode_t conditional_node = nullptr;
#if !defined(MXX_GPU_BACKEND_HIP) && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
            cudaGraphConditionalHandle handle = 0;
#endif
            uint32_t parent_operation_index = 0;
            std::vector<gpuGraphNode_t> parent_operation_nodes;
            std::vector<MxxGraphBindingMapEntry> parent_binding_map;
            std::vector<gpuGraphNode_t> parent_body_terminals;
            bool parent_conditional_body_active = false;
            bool parent_generic_body_mode = false;
            int parent_body_device = -1;
            gpuGraphNode_t parent_last_body_conditional = nullptr;
            // WHILE control advanced by the body's tail gate; absent for IF.
            bool is_while = false;
            uint64_t *loop_index = nullptr;
            const uint64_t *loop_limit = nullptr;
            uint32_t *loop_status = nullptr;
            uint64_t loop_max_iterations = 0;
            uint32_t loop_index_binding = 0;
            uint32_t loop_limit_binding = 0;
            uint32_t loop_status_binding = 0;
        };
        std::vector<ConditionalFrame> conditional_frames;
#if defined(MXX_GPU_BACKEND_HIP)
        std::map<gpuGraphNode_t, HipGraphControl> hip_controls;
        std::vector<gpuGraph_t> hip_body_graphs;
#endif
        bool operation_active = false;
        uint32_t operation_index = 0;
        std::vector<gpuGraphNode_t> frontier;
        std::vector<gpuGraphNode_t> operation_nodes;
        std::vector<gpuGraphNode_t> terminals;
#if !defined(MXX_GPU_BACKEND_HIP)
        // Until successful instantiation transfers them to the executable,
        // the builder owns these events, including on partial construction.
        std::vector<gpuEvent_t> branch_completion_events;
#endif
        std::vector<GraphKernelUpdateRecord> kernels;
        std::vector<GraphMemcpyUpdateRecord> memcpys;
        std::vector<GraphMemsetUpdateRecord> memsets;
        std::vector<GraphKernelUpdateRecord> body_kernels;
        std::vector<GraphMemcpyUpdateRecord> body_memcpys;
        std::vector<GraphMemsetUpdateRecord> body_memsets;
        struct ResidentAddress { uint64_t address; size_t bytes; uint32_t binding; };
        std::vector<ResidentAddress> resident_addresses;
        std::vector<MxxGraphBindingMapEntry> binding_map;
        // Graph-owned scratch allocation and free nodes, addressed by token.
        // The caller orders each one after earlier memory nodes; CUDA may
        // place an allocation in the memory of any free ordered before it.
        std::vector<gpuGraphNode_t> memory_nodes;
        // Allocation nodes the next top-level operation must follow.
        std::vector<gpuGraphNode_t> pending_memory_dependencies;
        // Pinned buffers of staged device copies, moved to the executable.
        std::vector<void *> host_staging;
        // Staged copies between one pair of physical GPUs in one graph take
        // turns on two buffers: a copy reuses the buffer of the copy two
        // before it, after that copy's host-to-device node. Pinned memory is
        // then bounded by the GPU pairs, not by the number of copies.
        struct HostStagingSlot
        {
            void *buffer = nullptr;
            size_t bytes = 0;
            gpuGraphNode_t last_reader = nullptr;
        };
        struct HostStagingRing
        {
            HostStagingSlot slots[2];
            size_t next = 0;
        };
        std::map<std::tuple<int, int, gpuGraph_t>, HostStagingRing> staging_rings;
        ~MxxGpuGraphBuilder()
        {
            if (root_graph) (void)gpuGraphDestroy(root_graph);
#if !defined(MXX_GPU_BACKEND_HIP)
            if (!branch_completion_events.empty()) (void)mxx_set_device(device);
            for (auto event : branch_completion_events) (void)gpuEventDestroy(event);
#endif
#if defined(MXX_GPU_BACKEND_HIP)
            for (auto graph : hip_body_graphs) (void)gpuGraphDestroy(graph);
#endif
            for (void *buffer : host_staging) (void)gpuFreeHost(buffer);
        }
    };

    struct MxxGpuNativeEvent
    {
        int device = -1;
        gpuEvent_t event = nullptr;
    };

    int validate_patch_value(
        const MxxGraphPatch &patch,
        const MxxGraphBindingValue *values,
        size_t value_count,
        uint8_t *destination)
    {
        if (!values || patch.binding_index >= value_count || !destination)
        {
            return set_error("invalid CUDA graph patch binding");
        }
        const MxxGraphBindingValue &value = values[patch.binding_index];
        if (patch.target == MXX_GRAPH_PATCH_INTEGER_ENCODING) {
            if (patch.byte_count != sizeof(int)) return set_error("invalid integer encoding patch width");
            if (value.kind == 4) memcpy(destination, value.bytes + 8, sizeof(int));
            return 0;
        }
        if (patch.byte_count == 0 || patch.byte_count > sizeof(value.bytes) ||
            patch.byte_count != value.byte_count)
        {
            return set_error("CUDA graph binding width does not match its patch");
        }
        if (value.kind == MXX_GRAPH_BINDING_BYTES32 && value.byte_count != 32)
        {
            return set_error("Bytes32 graph binding has an invalid width");
        }
        if ((value.kind == MXX_GRAPH_BINDING_DEVICE_ADDRESS ||
             value.kind == MXX_GRAPH_BINDING_U64 ||
             value.kind == MXX_GRAPH_BINDING_I64) && value.byte_count != 8)
        {
            return set_error("word graph binding has an invalid width");
        }
        memcpy(destination, value.bytes, patch.byte_count);
        if (patch.address_addend != 0)
        {
            if (patch.byte_count != sizeof(uint64_t))
            {
                return set_error("address addend requires an eight-byte patch");
            }
            uint64_t address = 0;
            memcpy(&address, destination, sizeof(address));
            if (UINT64_MAX - address < patch.address_addend)
            {
                return set_error("CUDA graph address patch overflow");
            }
            address += patch.address_addend;
            memcpy(destination, &address, sizeof(address));
        }
        return 0;
    }

    int patch_kernel_record(GraphKernelUpdateRecord &record,
        const MxxGraphBindingValue *values, size_t count, bool &changed)
    {
        changed = false;
        for (const GraphPatchRecord &patch_record : record.patches)
        {
            const MxxGraphPatch &patch = patch_record.patch;
            if (patch.byte_offset > record.arguments[patch.argument_index].size() ||
                patch.byte_count > record.arguments[patch.argument_index].size() - patch.byte_offset)
                return 1;
            uint8_t *field = record.arguments[patch.argument_index].data() + patch.byte_offset;
            uint8_t previous[sizeof(MxxGraphBindingValue::bytes)];
            memcpy(previous, field, patch.byte_count);
            if (validate_patch_value(patch, values, count, field) != 0) return 1;
            changed = changed || memcmp(previous, field, patch.byte_count) != 0;
        }
        for (size_t index = 0; index < record.arguments.size(); ++index)
            record.argument_pointers[index] = record.arguments[index].data();
        record.launch.kernelParams = record.argument_pointers.data();
        return 0;
    }

    bool patch_duplicate(
        const std::vector<GraphPatchRecord> &patches,
        const MxxGraphPatch &candidate)
    {
        for (const GraphPatchRecord &record : patches)
        {
            const MxxGraphPatch &patch = record.patch;
            if (patch.target == candidate.target &&
                patch.argument_index == candidate.argument_index &&
                patch.byte_offset == candidate.byte_offset &&
                patch.byte_count == candidate.byte_count)
            {
                return true;
            }
        }
        return false;
    }

    int mxx_gpu_graph_builder_create(GpuContext *ctx, int device, void *stream,
        MxxGpuGraphBuilder **out_builder)
    {
        if (!ctx || !ctx->execution || !out_builder)
            return set_error("invalid explicit CUDA graph builder arguments");
        *out_builder = nullptr;
        if (!stream && gpu_context_get_compute_stream(ctx, device, &stream) != 0) return 1;
        if (mxx_set_device(device) != gpuSuccess) return set_error(gpuGetLastError());
        auto *builder = new (std::nothrow) MxxGpuGraphBuilder();
        if (!builder) return set_error("failed to allocate graph builder");
        builder->context = ctx;
        builder->device = device;
        builder->stream = reinterpret_cast<gpuStream_t>(stream);
        const gpuError_t error = gpuGraphCreate(&builder->graph, 0);
        if (error != gpuSuccess) { delete builder; return set_error(error); }
        builder->root_graph = builder->graph;
        *out_builder = builder;
        return 0;
    }

    // The explicit graph operation this thread is constructing, whatever
    // device context its current operation targets.
    static MxxGpuGraphBuilder *&thread_explicit_builder()
    {
        static thread_local MxxGpuGraphBuilder *builder = nullptr;
        return builder;
    }

    int mxx_gpu_graph_builder_begin_operation(MxxGpuGraphBuilder *builder,
        uint32_t operation_index, const uint32_t *predecessors, size_t predecessor_count)
    {
        if (!builder || builder->operation_active ||
            (builder->conditional_body_active && !builder->generic_body_mode) ||
            (predecessor_count && !predecessors))
            return set_error("invalid explicit graph operation start");
        const auto &available_terminals = builder->generic_body_mode ?
            builder->body_terminals : builder->terminals;
        for (size_t index = 0; index < predecessor_count; ++index)
            if (predecessors[index] >= available_terminals.size())
                return set_error("explicit graph predecessor token is unknown");
        try
        {
            auto &owner = *builder->context->execution;
            {
                std::lock_guard<std::mutex> lock(owner.graph_mutex);
                if (owner.explicit_builder && owner.explicit_builder != builder)
                    return set_error("another explicit graph operation is active");
                owner.explicit_builder = builder;
            }
            thread_explicit_builder() = builder;
            builder->frontier.clear();
            builder->operation_nodes.clear();
            builder->binding_map.clear();
            for (size_t index = 0; index < predecessor_count; ++index)
            {
                const uint32_t token = predecessors[index];
                auto node = available_terminals[token];
                if (std::find(builder->frontier.begin(), builder->frontier.end(), node) ==
                    builder->frontier.end()) builder->frontier.push_back(node);
            }
            if (!builder->generic_body_mode)
            {
                for (auto node : builder->pending_memory_dependencies)
                    if (std::find(builder->frontier.begin(), builder->frontier.end(), node) ==
                        builder->frontier.end()) builder->frontier.push_back(node);
                builder->pending_memory_dependencies.clear();
            }
            builder->operation_index = operation_index;
            builder->operation_active = true;
            return 0;
        }
        catch (const std::exception &error) { return set_error(error); }
    }

    int mxx_gpu_graph_builder_finish_operation(MxxGpuGraphBuilder *builder,
        uint32_t *out_terminal)
    {
        if (!builder || !builder->operation_active ||
            (builder->conditional_body_active && !builder->generic_body_mode) ||
            !out_terminal)
            return set_error("invalid explicit graph operation finish");
        gpuGraphNode_t terminal = nullptr;
        if (builder->operation_nodes.empty())
        {
            const gpuError_t error = gpuGraphAddEmptyNode(&terminal, builder->graph,
                builder->frontier.data(), builder->frontier.size());
            if (error != gpuSuccess) return set_error(error);
        }
        else terminal = builder->operation_nodes.back();
        auto &terminals = builder->generic_body_mode ?
            builder->body_terminals : builder->terminals;
        if (terminals.size() >= UINT32_MAX)
            return set_error("explicit graph operation token overflow");
        terminals.push_back(terminal);
        *out_terminal = static_cast<uint32_t>(terminals.size() - 1);
        builder->operation_active = false;
        if (!builder->generic_body_mode) {
            auto &owner = *builder->context->execution;
            std::lock_guard<std::mutex> lock(owner.graph_mutex);
            if (owner.explicit_builder == builder) owner.explicit_builder = nullptr;
            if (thread_explicit_builder() == builder) thread_explicit_builder() = nullptr;
        }
        builder->frontier.clear();
        builder->operation_nodes.clear();
        return 0;
    }

    int mxx_gpu_graph_builder_bind_resident_address(MxxGpuGraphBuilder *builder,
        uint64_t address, size_t bytes, uint32_t binding)
    {
        if (!builder || !address || !bytes) return set_error("invalid graph binding address");
        for (const auto &owner : builder->resident_addresses)
        {
            if (owner.binding == binding)
            {
                if (owner.address == address && owner.bytes == bytes) return 0;
                const std::string detail =
                    "graph binding identity changed: binding=" + std::to_string(binding) +
                    " old_address=" + std::to_string(owner.address) +
                    " old_bytes=" + std::to_string(owner.bytes) +
                    " new_address=" + std::to_string(address) +
                    " new_bytes=" + std::to_string(bytes);
                return set_error(detail.c_str());
            }
        }
        builder->resident_addresses.push_back({address, bytes, binding});
        return 0;
    }

    int normalize_builder_patch(const MxxGpuGraphBuilder *builder,
        MxxGraphPatch *patch, uint64_t captured_address)
    {
        if (!builder || !patch) return set_error("invalid explicit graph patch");
        if (!builder->binding_map.empty())
        {
            const auto mapping = std::find_if(builder->binding_map.begin(),
                builder->binding_map.end(), [patch](const auto &entry) {
                    return entry.local_binding == patch->binding_index;
                });
            if (mapping == builder->binding_map.end())
                return set_error("unmapped explicit graph local binding");
            patch->binding_index = mapping->global_binding;
        }
        const auto owner = std::find_if(builder->resident_addresses.begin(),
            builder->resident_addresses.end(), [patch](const auto &entry) {
                return entry.binding == patch->binding_index;
            });
        if (owner != builder->resident_addresses.end())
        {
            if (captured_address < owner->address ||
                captured_address - owner->address >= owner->bytes)
                return set_error("explicit graph pointer outside registered binding");
            if (UINT64_MAX - patch->address_addend < captured_address - owner->address)
                return set_error("explicit graph pointer addend overflow");
            patch->address_addend += captured_address - owner->address;
        }
        return 0;
    }

    int mxx_gpu_graph_builder_add_kernel(MxxGpuGraphBuilder *builder, const void *function,
        uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t block_y, uint32_t block_z,
        size_t shared_bytes, const void *const *arguments, const size_t *argument_sizes,
        size_t argument_count, const MxxGraphPatch *patches, size_t patch_count)
    {
        if (!builder || !builder->operation_active || !function || !grid_x || !grid_y || !grid_z ||
            !block_x || !block_y || !block_z ||
            (argument_count && (!arguments || !argument_sizes)) ||
            (patch_count && !patches)) return set_error("invalid explicit kernel node");
        try
        {
            GraphKernelUpdateRecord record;
            record.arguments.resize(argument_count);
            record.argument_pointers.resize(argument_count);
            for (size_t index = 0; index < argument_count; ++index)
            {
                if (!argument_sizes[index] || !arguments[index])
                    return set_error("empty explicit kernel argument");
                record.arguments[index].resize(argument_sizes[index]);
                memcpy(record.arguments[index].data(), arguments[index], argument_sizes[index]);
                record.argument_pointers[index] = record.arguments[index].data();
            }
            record.launch.func = const_cast<void *>(function);
            record.launch.gridDim = dim3(grid_x, grid_y, grid_z);
            record.launch.blockDim = dim3(block_x, block_y, block_z);
            record.launch.sharedMemBytes = static_cast<unsigned int>(shared_bytes);
            record.launch.kernelParams = record.argument_pointers.data();
            for (size_t index = 0; index < patch_count; ++index)
            {
                const auto &patch = patches[index];
                if ((patch.target != MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD &&
                    patch.target != MXX_GRAPH_PATCH_INTEGER_ENCODING) ||
                    patch.argument_index >= argument_count ||
                    patch.byte_offset > argument_sizes[patch.argument_index] ||
                    patch.byte_count > argument_sizes[patch.argument_index] - patch.byte_offset ||
                    patch_duplicate(record.patches, patch))
                    return set_error("invalid explicit kernel patch");
                MxxGraphPatch normalized = patch;
                uint64_t captured_address = 0;
                if (patch.byte_count == sizeof(uint64_t) &&
                    patch.target == MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD)
                    memcpy(&captured_address,
                        static_cast<const uint8_t *>(arguments[patch.argument_index]) + patch.byte_offset,
                        sizeof(captured_address));
                if (normalize_builder_patch(builder, &normalized, captured_address) != 0) return 1;
                record.patches.push_back(GraphPatchRecord{normalized});
            }
            gpuError_t error = mxx_get_device(&record.device);
            if (error == gpuSuccess)
                error = gpuGraphAddKernelNode(&record.node, builder->graph,
                    builder->frontier.data(), builder->frontier.size(), &record.launch);
            if (error != gpuSuccess) return set_error(error);
                // The node, and the executable instantiated from it, hold these bytes.
            record.bound = true;
            builder->frontier.assign(1, record.node);
            builder->operation_nodes.push_back(record.node);
            if (builder->conditional_body_active)
                builder->body_kernels.push_back(std::move(record));
            else builder->kernels.push_back(std::move(record));
            return 0;
        }
        catch (const std::exception &error) { return set_error(error); }
    }

    // A strided-copy kernel node. Patches name the destination and source
    // like the one-dimensional memcpy patches.
    int mxx_gpu_graph_builder_add_strided_copy(MxxGpuGraphBuilder *builder,
        void *destination, const void *source, MxxStridedCopy copy,
        const MxxGraphPatch *patches, size_t patch_count)
    {
        if (!builder || !builder->operation_active || !destination || !source ||
            copy.element_bytes == 0 || (patch_count && !patches))
            return set_error("invalid strided copy node");
        std::vector<MxxGraphPatch> kernel_patches;
        for (size_t index = 0; index < patch_count; ++index)
        {
            const auto &patch = patches[index];
            if ((patch.target != MXX_GRAPH_PATCH_MEMCPY_1D_SRC &&
                patch.target != MXX_GRAPH_PATCH_MEMCPY_1D_DST) ||
                patch.byte_count != sizeof(uint64_t))
                return set_error("invalid strided copy patch");
            MxxGraphPatch kernel_patch = patch;
            kernel_patch.target = MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD;
            kernel_patch.argument_index = patch.target == MXX_GRAPH_PATCH_MEMCPY_1D_DST ? 0 : 1;
            kernel_patch.byte_offset = 0;
            kernel_patches.push_back(kernel_patch);
        }
        const void *arguments[] = {&destination, &source, &copy};
        const size_t sizes[] = {sizeof(destination), sizeof(source), sizeof(copy)};
        constexpr uint32_t threads = 256;
        return mxx_gpu_graph_builder_add_kernel(builder,
            reinterpret_cast<const void *>(mxx_strided_copy_kernel),
            strided_copy_blocks(copy, threads), 1, 1, threads, 1, 1, 0, arguments, sizes, 3,
            kernel_patches.data(), kernel_patches.size());
    }

    int mxx_gpu_graph_builder_add_memcpy(MxxGpuGraphBuilder *builder,
        void *destination, const void *source, size_t bytes, int copy_kind,
        const MxxGraphPatch *patches, size_t patch_count)
    {
        if (!builder || !builder->operation_active || !destination || !source || !bytes ||
            (patch_count && !patches)) return set_error("invalid explicit memcpy node");
        // A device copy that does not stay within the graph's GPU becomes a
        // copy kernel on a GPU that reaches both operands: CUDA rebinds a peer
        // memcpy node only while its operands keep their mappings, and a
        // kernel's pointer arguments may be rebound freely. Without peer
        // access the memcpy node is kept and CUDA reports the unsupported
        // copy when the node is created.
        if (copy_kind == gpuMemcpyDefault || copy_kind == gpuMemcpyDeviceToDevice)
        {
            gpuPointerAttributes source_attributes{};
            gpuPointerAttributes destination_attributes{};
            int current = 0;
            auto reaches = [](int executor, int owner) {
                return executor == owner || gpu_peer_pool_accessible(executor, owner);
            };
            // A memcpy node takes the context current when it is added, and
            // the driver cannot run one whose operands are both on another
            // GPU. Only a copy within the graph's GPU stays a memcpy node; any
            // other copy is a kernel: on the operands' GPU when they share
            // one, otherwise on the graph's GPU. A conditional body keeps
            // every node on its GPU.
            const int graph_device = builder->conditional_body_active &&
                    builder->body_device >= 0 ?
                builder->body_device : mxx_physical_device(builder->device);
            if (gpuPointerGetAttributes(&source_attributes, source) == gpuSuccess &&
                gpuPointerGetAttributes(&destination_attributes, destination) == gpuSuccess &&
                gpuGetDevice(&current) == gpuSuccess &&
                source_attributes.type == gpuMemoryTypeDevice &&
                destination_attributes.type == gpuMemoryTypeDevice &&
                (source_attributes.device != graph_device ||
                    destination_attributes.device != graph_device))
            {
                const bool shared = source_attributes.device == destination_attributes.device;
                const int executor = !builder->conditional_body_active && shared ?
                    source_attributes.device : graph_device;
                if (reaches(executor, source_attributes.device) &&
                    reaches(executor, destination_attributes.device))
                {
                    std::vector<MxxGraphPatch> kernel_patches;
                    for (size_t index = 0; index < patch_count; ++index)
                    {
                        const auto &patch = patches[index];
                        if ((patch.target != MXX_GRAPH_PATCH_MEMCPY_1D_SRC &&
                            patch.target != MXX_GRAPH_PATCH_MEMCPY_1D_DST) ||
                            patch.byte_count != sizeof(uint64_t))
                            return set_error("invalid explicit memcpy patch");
                        MxxGraphPatch kernel_patch = patch;
                        kernel_patch.target = MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD;
                        kernel_patch.argument_index =
                            patch.target == MXX_GRAPH_PATCH_MEMCPY_1D_DST ? 0 : 1;
                        kernel_patch.byte_offset = 0;
                        kernel_patches.push_back(kernel_patch);
                    }
                    const uint64_t length = bytes;
                    const void *arguments[] = {&destination, &source, &length};
                    const size_t sizes[] = {sizeof(destination), sizeof(source), sizeof(length)};
                    constexpr uint32_t threads = 256;
                    const uint64_t units = (bytes + 15) / 16;
                    const uint32_t blocks = static_cast<uint32_t>(
                        std::max<uint64_t>(1, std::min<uint64_t>((units + threads - 1) / threads, 4096)));
                    if (executor != current)
                    {
                        const gpuError_t device_error = gpuSetDevice(executor);
                        if (device_error != gpuSuccess) return set_error(device_error);
                    }
                    const int status = mxx_gpu_graph_builder_add_kernel(builder,
                        reinterpret_cast<const void *>(mxx_peer_copy_kernel), blocks, 1, 1,
                        threads, 1, 1, 0, arguments, sizes, 3, kernel_patches.data(),
                        kernel_patches.size());
                    if (executor != current)
                    {
                        const gpuError_t device_error = gpuSetDevice(current);
                        if (device_error != gpuSuccess && status == 0)
                            return set_error(device_error);
                    }
                    return status;
                }
            }
            (void)gpuGetLastError();
        }
        GraphMemcpyUpdateRecord record;
        record.source = const_cast<void *>(source);
        record.destination = destination;
        record.bytes = bytes;
        record.kind = static_cast<gpuMemcpyKind>(copy_kind);
        for (size_t index = 0; index < patch_count; ++index)
        {
            const auto &patch = patches[index];
            if ((patch.target != MXX_GRAPH_PATCH_MEMCPY_1D_SRC &&
                patch.target != MXX_GRAPH_PATCH_MEMCPY_1D_DST) ||
                patch.byte_count != sizeof(uint64_t) || patch_duplicate(record.patches, patch))
                return set_error("invalid explicit memcpy patch");
            MxxGraphPatch normalized = patch;
            const uint64_t captured_address = reinterpret_cast<uint64_t>(
                patch.target == MXX_GRAPH_PATCH_MEMCPY_1D_SRC ? source : destination);
            if (normalize_builder_patch(builder, &normalized, captured_address) != 0) return 1;
            record.patches.push_back(GraphPatchRecord{normalized});
        }
        const gpuError_t error = gpuGraphAddMemcpyNode1D(&record.node, builder->graph,
            builder->frontier.data(), builder->frontier.size(), destination, source,
            bytes, record.kind);
        if (error != gpuSuccess) return set_error(error);
        record.bound_source = record.source;
        record.bound_destination = record.destination;
        record.bound = true;
        builder->frontier.assign(1, record.node);
        builder->operation_nodes.push_back(record.node);
        if (builder->conditional_body_active)
            builder->body_memcpys.push_back(std::move(record));
        else builder->memcpys.push_back(std::move(record));
        return 0;
    }

    static bool host_staged_copies_forced()
    {
        static const bool forced = [] {
            const char *value = std::getenv("MXX_GPU_HOST_STAGED_COPIES");
            return value && std::strcmp(value, "1") == 0;
        }();
        return forced;
    }

    int mxx_gpu_graph_builder_add_device_copy(MxxGpuGraphBuilder *builder,
        void *destination, int destination_device, const void *source, int source_device,
        size_t bytes, const MxxGraphPatch *patches, size_t patch_count)
    {
        if (!builder || !destination || !source || !bytes || (patch_count && !patches))
            return set_error("invalid explicit device copy");
        const int from = mxx_physical_device(source_device);
        const int to = mxx_physical_device(destination_device);
        bool direct = from == to;
        if (!direct)
        {
            direct = gpu_peer_pool_accessible(to, from);
        }
        if (source_device != destination_device && host_staged_copies_forced()) direct = false;
        // A stream-ordered pool or graph allocation is a valid copy operand
        // only with its own GPU current, so every node is created (and later
        // updated) with the GPU of the memory it touches current: the shared
        // GPU, the destination GPU of a peer copy (which may access the
        // source's pool), or each side of a copy staged through host memory.
        int current = -1;
        if (gpuGetDevice(&current) != gpuSuccess) return set_error(gpuGetLastError());
        auto &records = builder->conditional_body_active ? builder->body_memcpys : builder->memcpys;
        const auto add_on = [&](int device, void *to_address, const void *from_address,
                                gpuMemcpyKind kind, const MxxGraphPatch *node_patches,
                                size_t node_patch_count) {
            if (mxx_set_device(device) != gpuSuccess) return set_error(gpuGetLastError());
            const size_t before = records.size();
            const int status = mxx_gpu_graph_builder_add_memcpy(builder, to_address,
                from_address, bytes, kind, node_patches, node_patch_count);
            if (status == 0 && records.size() == before + 1) records.back().device = device;
            return status;
        };
        int status = 0;
        if (direct)
        {
            status = add_on(from == to ? source_device : destination_device, destination, source,
                from == to ? gpuMemcpyDeviceToDevice : gpuMemcpyDefault, patches, patch_count);
        }
        else
        {
            // Without peer access neither GPU may touch the other's memory, so
            // the copy passes through a pinned host buffer the executable owns.
            std::vector<MxxGraphPatch> source_patches, destination_patches;
            for (size_t index = 0; index < patch_count; ++index)
                (patches[index].target == MXX_GRAPH_PATCH_MEMCPY_1D_SRC ? source_patches :
                    destination_patches).push_back(patches[index]);
            auto &ring = builder->staging_rings[{from, to, builder->graph}];
            auto &slot = ring.slots[ring.next];
            ring.next ^= 1;
            if (slot.bytes < bytes)
            {
                // Earlier nodes keep reading the smaller buffer it replaces.
                void *staging = nullptr;
                const gpuError_t error = gpuMallocHost(&staging, bytes);
                if (error != gpuSuccess) return set_error(error);
                try { builder->host_staging.push_back(staging); }
                catch (const std::exception &exception)
                {
                    (void)gpuFreeHost(staging);
                    return set_error(exception);
                }
                slot = {staging, bytes, nullptr};
            }
            // Nodes are added in dependency order, so the edge from an earlier
            // node cannot close a cycle.
            if (slot.last_reader &&
                std::find(builder->frontier.begin(), builder->frontier.end(), slot.last_reader) ==
                    builder->frontier.end())
                builder->frontier.push_back(slot.last_reader);
            status = add_on(source_device, slot.buffer, source, gpuMemcpyDeviceToHost,
                source_patches.data(), source_patches.size());
            if (status == 0)
                status = add_on(destination_device, destination, slot.buffer,
                    gpuMemcpyHostToDevice, destination_patches.data(), destination_patches.size());
            if (status == 0) slot.last_reader = records.back().node;
        }
        if (gpuSetDevice(current) != gpuSuccess && status == 0)
            return set_error(gpuGetLastError());
        return status;
    }

    int mxx_gpu_graph_builder_add_memset(MxxGpuGraphBuilder *builder,
        void *destination, int value, size_t bytes, const MxxGraphPatch *patch)
    {
        if (!builder || !builder->operation_active || !destination || !bytes ||
            (patch && (patch->target != MXX_GRAPH_PATCH_MEMSET_1D_DST ||
                patch->byte_count != sizeof(uint64_t))))
            return set_error("invalid explicit memset node");
        GraphMemsetUpdateRecord record;
        record.params.dst = destination;
        record.params.value = value;
        record.params.elementSize = 1;
        record.params.width = bytes;
        record.params.height = 1;
        if (patch)
        {
            record.patch.patch = *patch;
            if (normalize_builder_patch(builder, &record.patch.patch,
                reinterpret_cast<uint64_t>(destination)) != 0) return 1;
        }
        gpuError_t error = mxx_get_device(&record.device);
        if (error == gpuSuccess)
            error = gpuGraphAddMemsetNode(&record.node, builder->graph,
                builder->frontier.data(), builder->frontier.size(), &record.params);
        if (error != gpuSuccess) return set_error(error);
        record.bound = true;
        builder->frontier.assign(1, record.node);
        builder->operation_nodes.push_back(record.node);
        if (patch)
        {
            if (builder->conditional_body_active)
                builder->body_memsets.push_back(std::move(record));
            else builder->memsets.push_back(std::move(record));
        }
        return 0;
    }

    int mxx_gpu_graph_builder_add_export_publish(MxxGpuGraphBuilder *builder,
        void *device_header, uint64_t occurrence, uint64_t artifact_offset,
        uint64_t payload_bytes, uint32_t site, uint32_t flags,
        uint32_t header_binding, const void *gate, uint32_t gate_binding)
    {
        if (!builder || !device_header || (flags & ~1U) != 0)
            return set_error("invalid explicit export publication");
        const void *arguments[] = {&device_header, &occurrence, &artifact_offset,
            &payload_bytes, &site, &flags, &gate};
        const size_t sizes[] = {sizeof(device_header), sizeof(occurrence),
            sizeof(artifact_offset), sizeof(payload_bytes), sizeof(site), sizeof(flags),
            sizeof(gate)};
        const MxxGraphPatch patches[] = {
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(device_header),
                header_binding, 0},
            {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 6, 0, sizeof(gate),
                gate_binding, 0}};
        return mxx_gpu_graph_builder_add_kernel(builder,
            reinterpret_cast<const void *>(mxx_export_slot_publish_kernel),
            1, 1, 1, 1, 1, 1, 0, arguments, sizes, 7, patches, gate ? 2 : 1);
    }

    static MxxGraphPatch mxx_direct_pointer_patch(uint32_t argument_index,
        uint32_t binding)
    {
        return MxxGraphPatch{nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
            argument_index, 0, sizeof(void *), binding, 0};
    }

#if defined(MXX_GPU_BACKEND_HIP)
extern "C++" {
#include "HipGraphSchedule.cu"
}
#endif

#if !defined(MXX_GPU_BACKEND_HIP) && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
    static int mxx_gpu_graph_builder_enter_generic_body(MxxGpuGraphBuilder *builder,
        cudaGraphConditionalHandle handle, cudaGraphConditionalNodeType type,
        const MxxGpuGraphBuilder::ConditionalFrame *loop)
    {
#if !defined(MXX_GPU_BACKEND_HIP) && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        cudaGraphNodeParams parameters{};
        parameters.type = cudaGraphNodeTypeConditional;
        parameters.conditional.handle = handle;
        parameters.conditional.type = type;
        parameters.conditional.size = 1;
        gpuGraphNode_t conditional = nullptr;
        // The node joins the scope that is open now: the root graph, or the
        // body graph of an enclosing conditional.
        const gpuError_t error = cudaGraphAddNode(&conditional, builder->graph,
            builder->frontier.data(), nullptr, builder->frontier.size(), &parameters);
        if (error != gpuSuccess) return set_error(error);
        if (!parameters.conditional.phGraph_out ||
            !parameters.conditional.phGraph_out[0])
            return set_error("CUDA returned no direct conditional body graph");
        MxxGpuGraphBuilder::ConditionalFrame frame;
        frame.parent_graph = builder->graph;
        frame.conditional_node = conditional;
        frame.handle = handle;
        frame.parent_operation_index = builder->operation_index;
        frame.parent_operation_nodes = std::move(builder->operation_nodes);
        frame.parent_binding_map = std::move(builder->binding_map);
        frame.parent_body_terminals = std::move(builder->body_terminals);
        frame.parent_conditional_body_active = builder->conditional_body_active;
        frame.parent_generic_body_mode = builder->generic_body_mode;
        frame.parent_body_device = builder->body_device;
        frame.parent_last_body_conditional = builder->last_body_conditional;
        if (loop)
        {
            frame.is_while = true;
            frame.loop_index = loop->loop_index;
            frame.loop_limit = loop->loop_limit;
            frame.loop_status = loop->loop_status;
            frame.loop_max_iterations = loop->loop_max_iterations;
            frame.loop_index_binding = loop->loop_index_binding;
            frame.loop_limit_binding = loop->loop_limit_binding;
            frame.loop_status_binding = loop->loop_status_binding;
        }
        builder->conditional_frames.push_back(std::move(frame));
        builder->graph = parameters.conditional.phGraph_out[0];
        builder->frontier.clear();
        builder->operation_nodes.clear();
        builder->binding_map.clear();
        builder->body_terminals.clear();
        builder->last_body_conditional = nullptr;
        builder->operation_active = false;
        builder->conditional_body_active = true;
        builder->generic_body_mode = true;
        {
            // The conditional node, and so its body, lives on the device that
            // was current when it was added.
            const gpuError_t device_error = gpuGetDevice(&builder->body_device);
            if (device_error != gpuSuccess) return set_error(device_error);
        }
        return 0;
#else
        (void)builder; (void)handle; (void)type; (void)loop;
        return GPU_STATUS_CONDITIONAL_UNSUPPORTED;
#endif
    }

    static void mxx_gpu_graph_builder_order_body_conditional(MxxGpuGraphBuilder *builder)
    {
        const auto previous = builder->last_body_conditional;
        if (builder->conditional_body_active && previous &&
            std::find(builder->frontier.begin(), builder->frontier.end(), previous) ==
                builder->frontier.end())
            builder->frontier.push_back(previous);
    }

#endif

    int mxx_gpu_graph_builder_begin_if(MxxGpuGraphBuilder *builder,
        const uint64_t *predicate, uint32_t predicate_binding)
    {
#if !defined(MXX_GPU_BACKEND_HIP) && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        if (!builder || !builder->operation_active ||
            (builder->conditional_body_active && !builder->generic_body_mode) ||
            !predicate) return set_error("invalid direct IF body");
        mxx_gpu_graph_builder_order_body_conditional(builder);
        cudaGraphConditionalHandle handle = 0;
        gpuError_t error = cudaGraphConditionalHandleCreate(&handle,
            builder->graph, 1U, cudaGraphCondAssignDefault);
        if (error != gpuSuccess) return set_error(error);
        const void *arguments[] = {&predicate, &handle};
        const size_t sizes[] = {sizeof(predicate), sizeof(handle)};
        const MxxGraphPatch patch = mxx_direct_pointer_patch(0, predicate_binding);
        int status = mxx_gpu_graph_builder_add_kernel(builder,
            reinterpret_cast<const void *>(mxx_if_gate_kernel),
            1, 1, 1, 1, 1, 1, 0, arguments, sizes, 2, &patch, 1);
        if (status != 0) return status;
        return mxx_gpu_graph_builder_enter_generic_body(
            builder, handle, cudaGraphCondTypeIf, nullptr);
#elif defined(MXX_GPU_BACKEND_HIP)
        return hip_graph_begin_control(builder, predicate, predicate_binding,
            nullptr, nullptr, 0, nullptr, 0, 0, 0);
#else
        (void)builder; (void)predicate; (void)predicate_binding;
        return GPU_STATUS_CONDITIONAL_UNSUPPORTED;
#endif
    }

    int mxx_gpu_graph_builder_begin_while(MxxGpuGraphBuilder *builder,
        uint64_t *index, const uint64_t *limit, uint64_t max_iterations,
        uint32_t *status_word, uint32_t index_binding, uint32_t limit_binding,
        uint32_t status_binding)
    {
#if !defined(MXX_GPU_BACKEND_HIP) && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        if (!builder || !builder->operation_active ||
            (builder->conditional_body_active && !builder->generic_body_mode) ||
            !index || !limit || !status_word || !max_iterations)
            return set_error("invalid direct WHILE body");
        mxx_gpu_graph_builder_order_body_conditional(builder);
        cudaGraphConditionalHandle handle = 0;
        gpuError_t error = cudaGraphConditionalHandleCreate(&handle,
            builder->graph, 1U, cudaGraphCondAssignDefault);
        if (error != gpuSuccess) return set_error(error);
        const bool advance = false;
        const void *arguments[] = {
            &index, &limit, &max_iterations, &status_word, &handle, &advance};
        const size_t sizes[] = {sizeof(index), sizeof(limit), sizeof(max_iterations),
            sizeof(status_word), sizeof(handle), sizeof(advance)};
        const MxxGraphPatch patches[] = {
            mxx_direct_pointer_patch(0, index_binding),
            mxx_direct_pointer_patch(1, limit_binding),
            mxx_direct_pointer_patch(3, status_binding)};
        int result = mxx_gpu_graph_builder_add_kernel(builder,
            reinterpret_cast<const void *>(mxx_while_gate_kernel),
            1, 1, 1, 1, 1, 1, 0, arguments, sizes, 6, patches, 3);
        if (result != 0) return result;
        MxxGpuGraphBuilder::ConditionalFrame loop;
        loop.loop_index = index;
        loop.loop_limit = limit;
        loop.loop_status = status_word;
        loop.loop_max_iterations = max_iterations;
        loop.loop_index_binding = index_binding;
        loop.loop_limit_binding = limit_binding;
        loop.loop_status_binding = status_binding;
        return mxx_gpu_graph_builder_enter_generic_body(
            builder, handle, cudaGraphCondTypeWhile, &loop);
#elif defined(MXX_GPU_BACKEND_HIP)
        return hip_graph_begin_control(builder, nullptr, 0, index, limit,
            max_iterations, status_word, index_binding, limit_binding, status_binding);
#else
        (void)builder; (void)index; (void)limit; (void)max_iterations;
        (void)status_word; (void)index_binding; (void)limit_binding; (void)status_binding;
        return GPU_STATUS_CONDITIONAL_UNSUPPORTED;
#endif
    }

    int mxx_gpu_graph_builder_finish_generic_body(MxxGpuGraphBuilder *builder)
    {
#if !defined(MXX_GPU_BACKEND_HIP) && defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        if (!builder || !builder->generic_body_mode || !builder->conditional_body_active ||
            builder->operation_active)
            return set_error("invalid direct conditional body completion");
        if (builder->body_terminals.empty())
        {
            gpuGraphNode_t empty = nullptr;
            const gpuError_t error = gpuGraphAddEmptyNode(
                &empty, builder->graph, nullptr, 0);
            if (error != gpuSuccess) return set_error(error);
            builder->body_terminals.push_back(empty);
        }
        if (builder->conditional_frames.empty())
            return set_error("direct conditional body has no open frame");
        auto frame = std::move(builder->conditional_frames.back());
        builder->conditional_frames.pop_back();
        if (frame.is_while)
        {
            // The tail gate lives in the body but patches control bound by
            // the enclosing WHILE operation.
            builder->frontier = builder->body_terminals;
            builder->binding_map = frame.parent_binding_map;
            builder->operation_active = true;
            const auto handle = frame.handle;
            const bool advance = true;
            const void *arguments[] = {&frame.loop_index, &frame.loop_limit,
                &frame.loop_max_iterations, &frame.loop_status, &handle, &advance};
            const size_t sizes[] = {sizeof(frame.loop_index), sizeof(frame.loop_limit),
                sizeof(frame.loop_max_iterations), sizeof(frame.loop_status),
                sizeof(handle), sizeof(advance)};
            const MxxGraphPatch patches[] = {
                mxx_direct_pointer_patch(0, frame.loop_index_binding),
                mxx_direct_pointer_patch(1, frame.loop_limit_binding),
                mxx_direct_pointer_patch(3, frame.loop_status_binding)};
            const int result = mxx_gpu_graph_builder_add_kernel(builder,
                reinterpret_cast<const void *>(mxx_while_gate_kernel),
                1, 1, 1, 1, 1, 1, 0, arguments, sizes, 6, patches, 3);
            if (result != 0) return result;
        }
        builder->graph = frame.parent_graph;
        builder->frontier.assign(1, frame.conditional_node);
        builder->operation_nodes = std::move(frame.parent_operation_nodes);
        builder->operation_nodes.push_back(frame.conditional_node);
        builder->binding_map = std::move(frame.parent_binding_map);
        builder->body_terminals = std::move(frame.parent_body_terminals);
        builder->operation_index = frame.parent_operation_index;
        builder->operation_active = true;
        builder->conditional_body_active = frame.parent_conditional_body_active;
        builder->generic_body_mode = frame.parent_generic_body_mode;
        builder->body_device = frame.parent_body_device;
        builder->last_body_conditional = builder->conditional_body_active ?
            frame.conditional_node : nullptr;
        return 0;
#elif defined(MXX_GPU_BACKEND_HIP)
        return hip_graph_finish_control(builder);
#else
        (void)builder;
        return GPU_STATUS_CONDITIONAL_UNSUPPORTED;
#endif
    }

    // The memory nodes named by `tokens`, appended to `dependencies`.
    int append_memory_dependencies(const MxxGpuGraphBuilder *builder, const uint32_t *tokens,
        size_t count, std::vector<gpuGraphNode_t> &dependencies)
    {
        if (count && !tokens) return set_error("graph memory dependencies are missing");
        for (size_t index = 0; index < count; ++index)
        {
            if (tokens[index] >= builder->memory_nodes.size())
                return set_error("graph memory dependency token is unknown");
            const gpuGraphNode_t node = builder->memory_nodes[tokens[index]];
            if (std::find(dependencies.begin(), dependencies.end(), node) == dependencies.end())
                dependencies.push_back(node);
        }
        return 0;
    }

    // An empty node ordered after the memory nodes `after` and after the
    // emitted top-level `operations`. Scratch the plan places in its own
    // memory is ordered by these nodes as CUDA orders Graph allocations: an
    // allocation follows the free of any memory it reuses.
    int mxx_gpu_graph_builder_add_memory_barrier(MxxGpuGraphBuilder *builder,
        const uint32_t *operations, size_t operation_count, const uint32_t *after,
        size_t after_count, uint32_t *out_token)
    {
        if (!builder || !out_token || (operation_count && !operations) ||
            builder->operation_active || builder->conditional_body_active ||
            builder->generic_body_mode || builder->graph != builder->root_graph)
            return set_error("invalid graph memory barrier");
        if (builder->memory_nodes.size() >= UINT32_MAX)
            return set_error("graph memory node token overflow");
        std::vector<gpuGraphNode_t> dependencies;
        if (append_memory_dependencies(builder, after, after_count, dependencies) != 0) return 1;
        for (size_t index = 0; index < operation_count; ++index)
        {
            if (operations[index] >= builder->terminals.size())
                return set_error("graph memory barrier operation is unknown");
            const gpuGraphNode_t terminal = builder->terminals[operations[index]];
            if (std::find(dependencies.begin(), dependencies.end(), terminal) ==
                dependencies.end()) dependencies.push_back(terminal);
        }
        gpuGraphNode_t node = nullptr;
        const gpuError_t error = gpuGraphAddEmptyNode(&node, builder->root_graph,
            dependencies.data(), dependencies.size());
        if (error != gpuSuccess) return set_error(error);
        builder->memory_nodes.push_back(node);
        *out_token = static_cast<uint32_t>(builder->memory_nodes.size() - 1);
        return 0;
    }

    // The next top-level operation starts after these memory barriers.
    int mxx_gpu_graph_builder_set_pending_memory_dependencies(MxxGpuGraphBuilder *builder,
        const uint32_t *tokens, size_t count)
    {
        if (!builder || (count && !tokens) || builder->operation_active ||
            builder->generic_body_mode)
            return set_error("invalid pending graph memory dependencies");
        builder->pending_memory_dependencies.clear();
        for (size_t index = 0; index < count; ++index)
        {
            if (tokens[index] >= builder->memory_nodes.size())
                return set_error("graph memory dependency token is unknown");
            builder->pending_memory_dependencies.push_back(builder->memory_nodes[tokens[index]]);
        }
        return 0;
    }

    int mxx_gpu_graph_builder_finish(MxxGpuGraphBuilder *builder, MxxGpuGraphExec **out_exec)
    {
        if (!builder || !out_exec || builder->operation_active ||
            builder->conditional_body_active || builder->terminals.empty())
            return set_error("invalid explicit graph finish");
        *out_exec = nullptr;
#if !defined(MXX_GPU_BACKEND_HIP)
        // Compute Sanitizer can miss parallel graph-branch completion when
        // tracking stream-ordered allocation lifetime. Export completion from
        // each top-level operation before any downstream empty-node join.
        // Native nodes inside an operation are ordered, so one record covers
        // its whole chain. Conditional bodies cannot contain event nodes:
        // their enclosing top-level operation is the legal record boundary.
        // Record single-operation graphs too: distinct executables can use the
        // same storage in consecutive planning or production submissions.
        // These are side nodes: no dependency is added between operations.
        {
            try { builder->branch_completion_events.reserve(builder->terminals.size()); }
            catch (const std::exception &error) { return set_error(error); }
            gpuError_t status = mxx_set_device(builder->device);
            if (status != gpuSuccess) return set_error(status);
            for (auto terminal : builder->terminals)
            {
                cudaGraphNodeType type;
                status = cudaGraphNodeGetType(terminal, &type);
                if (status != gpuSuccess) return set_error(status);
                if (type == cudaGraphNodeTypeEmpty) continue;
                gpuEvent_t completion = nullptr;
                status = gpuEventCreateWithFlags(&completion, gpuEventDisableTiming);
                if (status != gpuSuccess) return set_error(status);
                builder->branch_completion_events.push_back(completion);
                gpuGraphNode_t record = nullptr;
                status = cudaGraphAddEventRecordNode(&record, builder->root_graph,
                    &terminal, 1, completion);
                if (status != gpuSuccess) return set_error(status);
            }
        }
#endif
        auto *result = new (std::nothrow) MxxGpuGraphExec();
        if (!result) return set_error("failed to allocate explicit graph exec");
        result->device = builder->device;
        result->execution = builder->context->execution;
        result->default_stream = builder->stream;
        const gpuError_t error = gpuGraphInstantiateWithFlags(&result->exec,
            builder->root_graph, gpuGraphInstantiateFlagAutoFreeOnLaunch);
        if (error != gpuSuccess) { delete result; return set_error(error); }
#if defined(MXX_GPU_BACKEND_HIP)
        if (!builder->hip_controls.empty())
        {
            result->schedule.reset(new HipGraphSchedule());
            if (result->schedule->compile(builder) != 0)
            {
                (void)gpuGraphExecDestroy(result->exec);
                delete result;
                return 1;
            }
        }
#endif
        result->graph = builder->root_graph;
        builder->root_graph = nullptr;
        builder->graph = nullptr;
        result->kernels = std::move(builder->kernels);
        result->memcpys = std::move(builder->memcpys);
        result->memsets = std::move(builder->memsets);
        result->body_kernels = std::move(builder->body_kernels);
        result->body_memcpys = std::move(builder->body_memcpys);
        result->body_memsets = std::move(builder->body_memsets);
#if defined(MXX_GPU_BACKEND_HIP)
        if (result->schedule)
        {
            for (auto &record : result->body_kernels) result->kernels.push_back(std::move(record));
            for (auto &record : result->body_memcpys) result->memcpys.push_back(std::move(record));
            for (auto &record : result->body_memsets) result->memsets.push_back(std::move(record));
            result->body_kernels.clear();
            result->body_memcpys.clear();
            result->body_memsets.clear();
        }
#endif
#if !defined(MXX_GPU_BACKEND_HIP)
        result->branch_completion_events = std::move(builder->branch_completion_events);
        builder->branch_completion_events.clear();
#endif
        result->host_staging = std::move(builder->host_staging);
        builder->host_staging.clear();
        *out_exec = result;
        delete builder;
        return 0;
    }

    void mxx_gpu_graph_builder_destroy(MxxGpuGraphBuilder *builder)
    {
        if (builder && builder->context && builder->context->execution)
        {
            auto &owner = *builder->context->execution;
            std::lock_guard<std::mutex> lock(owner.graph_mutex);
            if (owner.explicit_builder == builder) owner.explicit_builder = nullptr;
        }
        if (thread_explicit_builder() == builder) thread_explicit_builder() = nullptr;
        delete builder;
    }

    MxxGpuGraphBuilder *mxx_gpu_graph_builder_for_stream(GpuContext *ctx, void *stream)
    {
        if (!ctx || !ctx->execution || !stream) return nullptr;
        const auto matches = [stream](MxxGpuGraphBuilder *builder) {
            return builder && builder->operation_active &&
                builder->stream == reinterpret_cast<gpuStream_t>(stream);
        };
        {
            auto &owner = *ctx->execution;
            std::lock_guard<std::mutex> lock(owner.graph_mutex);
            if (matches(owner.explicit_builder)) return owner.explicit_builder;
        }
        // An operation of a multi-device graph emits through the stream of
        // another device's context; its builder is the one this thread is
        // constructing.
        auto *builder = thread_explicit_builder();
        return builder && builder->operation_active ? builder : nullptr;
    }

    int mxx_gpu_graph_dispatch_kernel(GpuContext *ctx, void *stream,
        const void *function, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t block_y, uint32_t block_z, size_t shared_bytes,
        void **arguments, const size_t *argument_sizes, size_t argument_count,
        const MxxGraphPatch *patches, size_t patch_count, int cooperative)
    {
        if (!ctx || !stream || !function || (cooperative != 0 && cooperative != 1))
            return set_error("invalid kernel dispatch");
        if (auto *builder = mxx_gpu_graph_builder_for_stream(ctx, stream))
        {
            // A kernel node belongs to the context current when it is added,
            // and it must be the context of the stream the kernel was issued
            // to: an operation of a multi-device graph emits through another
            // device's stream while the graph's device is current.
            int current = -1;
            int target = -1;
            gpuError_t error = gpuGetDevice(&current);
            if (error == gpuSuccess)
                error = gpuStreamGetDevice(reinterpret_cast<gpuStream_t>(stream), &target);
            if (error != gpuSuccess) return set_error(error);
            if (target != current)
            {
                error = gpuSetDevice(target);
                if (error != gpuSuccess) return set_error(error);
            }
            const int status = mxx_gpu_graph_builder_add_kernel(builder, function, grid_x,
                grid_y, grid_z, block_x, block_y, block_z, shared_bytes,
                const_cast<const void *const *>(arguments), argument_sizes,
                argument_count, patches, patch_count);
            if (target != current)
            {
                const gpuError_t restore = gpuSetDevice(current);
                if (restore != gpuSuccess && status == 0) return set_error(restore);
            }
            if (status != 0 || !cooperative) return status;
            gpuLaunchAttributeValue value{};
            value.cooperative = 1;
            const gpuError_t attribute = gpuGraphKernelNodeSetAttribute(
                builder->operation_nodes.back(), gpuLaunchAttributeCooperative, &value);
            return attribute == gpuSuccess ? 0 : set_error(attribute);
        }
        {
            auto &owner = *ctx->execution;
            std::lock_guard<std::mutex> lock(owner.graph_mutex);
            if (owner.explicit_builder)
                return set_error("kernel dispatch escaped the active explicit graph operation");
        }
        const gpuError_t error = cooperative ?
            gpuLaunchCooperativeKernel(function, dim3(grid_x, grid_y, grid_z),
                dim3(block_x, block_y, block_z), arguments, shared_bytes,
                reinterpret_cast<gpuStream_t>(stream)) :
            gpuLaunchKernel(function, dim3(grid_x, grid_y, grid_z),
                dim3(block_x, block_y, block_z), arguments, shared_bytes,
                reinterpret_cast<gpuStream_t>(stream));
        return error == gpuSuccess ? 0 : set_error(error);
    }

    int mxx_gpu_graph_upload(MxxGpuGraphExec *exec, void *launch_stream)
    {
        if (!exec || !exec->exec || !launch_stream)
        {
            return set_error("invalid mxx_gpu_graph_upload arguments");
        }
        gpuError_t error = mxx_set_device(exec->device);
        if (error == gpuSuccess)
        {
#if defined(MXX_GPU_BACKEND_HIP)
            error = exec->schedule ? exec->schedule->upload(reinterpret_cast<gpuStream_t>(launch_stream)) :
                gpuGraphUpload(exec->exec, reinterpret_cast<gpuStream_t>(launch_stream));
#else
            error = gpuGraphUpload(exec->exec, reinterpret_cast<gpuStream_t>(launch_stream));
#endif
        }
        return error == gpuSuccess ? 0 : set_error(error);
    }

    // Run a node update with the node's creation device current, then make
    // `home` current again. A kernel resolves, and pool or graph memory is
    // valid, only on its own GPU.
    extern "C++" template <typename Update>
    gpuError_t update_on_device(int device, int home, Update update)
    {
        const bool select = device >= 0 && device != home;
        if (select)
        {
            const gpuError_t error = mxx_set_device(device);
            if (error != gpuSuccess) return error;
        }
        gpuError_t error = update();
        if (select)
        {
            const gpuError_t restore = mxx_set_device(home);
            if (error == gpuSuccess) error = restore;
        }
        return error;
    }

    // Describe a failed memcpy rebind: CUDA accepts new operands only on the
    // devices of the instantiated ones, so name both pairs.
    extern "C++" std::string memcpy_rebind_error(gpuError_t error, const void *bound_source,
        const void *bound_destination, const void *source, const void *destination,
        size_t bytes, gpuMemcpyKind kind)
    {
        auto device_of = [](const void *pointer) {
            gpuPointerAttributes attributes{};
            if (gpuPointerGetAttributes(&attributes, pointer) != gpuSuccess)
            {
                (void)gpuGetLastError();
                return std::string("?");
            }
            if (attributes.type == gpuMemoryTypeHost) return std::string("host");
            if (attributes.type == gpuMemoryTypeUnregistered) return std::string("unregistered");
            return "gpu" + std::to_string(attributes.device);
        };
        return std::string(gpuGetErrorString(error)) + " (memcpy rebind of " +
            std::to_string(bytes) + " bytes, kind " + std::to_string(static_cast<int>(kind)) +
            ": source " + device_of(bound_source) + " -> " + device_of(source) +
            ", destination " + device_of(bound_destination) + " -> " + device_of(destination) + ")";
    }

    int mxx_gpu_graph_bind(
        MxxGpuGraphExec *exec,
        const MxxGraphBindingValue *values,
        size_t count)
    {
        if (!exec || !exec->exec || (count != 0 && !values))
        {
            return set_error("invalid mxx_gpu_graph_bind arguments");
        }
        gpuError_t error = mxx_set_device(exec->device);
        if (error != gpuSuccess)
        {
            return set_error((std::string("graph bind device: ") + gpuGetErrorString(error)).c_str());
        }
#if defined(MXX_GPU_BACKEND_HIP)
        if (exec->schedule && exec->schedule->bind(values, count) != 0) return 1;
#endif
        // Body launch records are attached to the explicit child graph. They
        // must be patched as part of every bind just like top-level records;
        // otherwise conditional retry replay would dereference stale
        // plan-time pointers. CUDA accepts these updates on the retained
        // child graph before its parent executable is launched.
        // A node whose patched bytes are unchanged since its last update keeps
        // its parameters; replays with the same inputs update nothing.
        bool body_changed = false;
        for (auto &record : exec->body_kernels)
        {
            bool changed = false;
            if (patch_kernel_record(record, values, count, changed) != 0) return 1;
            if (record.bound && !changed) continue;
            error = update_on_device(record.device, exec->device,
                [&] { return gpuGraphKernelNodeSetParams(record.node, &record.launch); });
            if (error != gpuSuccess) return set_error(error);
            record.bound = true;
            body_changed = true;
        }
        // Kernel updates may have left another device current.
        error = mxx_set_device(exec->device);
        if (error != gpuSuccess)
            return set_error((std::string("graph bind device: ") + gpuGetErrorString(error)).c_str());
        for (auto &record : exec->body_memcpys)
        {
            void *source = record.source;
            void *destination = record.destination;
            for (const GraphPatchRecord &patch_record : record.patches)
            {
                uint64_t address = 0;
                if (validate_patch_value(patch_record.patch, values, count,
                                         reinterpret_cast<uint8_t *>(&address)) != 0)
                    return 1;
                if (patch_record.patch.target == MXX_GRAPH_PATCH_MEMCPY_1D_SRC)
                    source = reinterpret_cast<void *>(address);
                else
                    destination = reinterpret_cast<void *>(address);
            }
            if (record.bound && source == record.bound_source &&
                destination == record.bound_destination)
                continue;
            error = update_on_device(record.device, exec->device, [&] {
                return gpuGraphMemcpyNodeSetParams1D(
                    record.node, destination, source, record.bytes, record.kind);
            });
            if (error != gpuSuccess) return set_error(error);
            record.bound_source = source;
            record.bound_destination = destination;
            record.bound = true;
            body_changed = true;
        }
        for (auto &record : exec->body_memsets)
        {
            uint64_t address = 0;
            if (validate_patch_value(record.patch.patch, values, count,
                                     reinterpret_cast<uint8_t *>(&address)) != 0)
                return 1;
            if (record.bound && record.params.dst == reinterpret_cast<void *>(address)) continue;
            record.params.dst = reinterpret_cast<void *>(address);
            error = update_on_device(record.device, exec->device,
                [&] { return gpuGraphMemsetNodeSetParams(record.node, &record.params); });
            if (error != gpuSuccess) return set_error(error);
            record.bound = true;
            body_changed = true;
        }
        // Reconciling the executable with the child graph also resets every
        // top-level node to its graph parameters, so they are all reapplied.
        const bool reapply_top_level = body_changed;
        if (body_changed)
        {
            // Body records refer to the retained child graph nodes rather than
            // executable top-level nodes.  Force CUDA to reconcile the
            // modified child graph with the instantiated parent; never report
            // a successful bind while silently leaving a stale child graph.
#if !defined(MXX_GPU_BACKEND_HIP) && defined(CUDART_VERSION) && CUDART_VERSION >= 13000
            gpuGraphExecUpdateResultInfo update_info{};
            error = gpuGraphExecUpdate(exec->exec, exec->graph, &update_info);
            if (error != gpuSuccess || update_info.result != gpuGraphExecUpdateSuccess)
            {
                error = error == gpuSuccess ? gpuErrorGraphExecUpdateFailure : error;
                return set_error((std::string("graph exec update: ") + gpuGetErrorString(error)).c_str());
            }
#else
            gpuGraphNode_t update_error_node = nullptr;
            gpuGraphExecUpdateResult update_result = gpuGraphExecUpdateError;
            error = gpuGraphExecUpdate(exec->exec, exec->graph, &update_error_node, &update_result);
            if (error != gpuSuccess || update_result != gpuGraphExecUpdateSuccess)
            {
                error = error == gpuSuccess ? gpuErrorGraphExecUpdateFailure : error;
                return set_error((std::string("graph exec update: ") + gpuGetErrorString(error)).c_str());
            }
#endif
        }
        for (auto &record : exec->kernels)
        {
            bool changed = false;
            if (patch_kernel_record(record, values, count, changed) != 0) return 1;
            if (record.bound && !changed && !reapply_top_level) continue;
            error = update_on_device(record.device, exec->device, [&] {
                return gpuGraphExecKernelNodeSetParams(
#if defined(MXX_GPU_BACKEND_HIP)
                    exec->schedule ? exec->schedule->executable(record.node) : exec->exec,
                    exec->schedule ? exec->schedule->node(record.node) : record.node,
#else
                    exec->exec, record.node,
#endif
                    &record.launch);
            });
            if (error != gpuSuccess)
            {
                return set_error((std::string("kernel rebind: ") + gpuGetErrorString(error)).c_str());
            }
            record.bound = true;
        }
        // Kernel updates may have left another device current.
        error = mxx_set_device(exec->device);
        if (error != gpuSuccess)
            return set_error((std::string("graph bind device: ") + gpuGetErrorString(error)).c_str());
        for (auto &record : exec->memcpys)
        {
            void *source = record.source;
            void *destination = record.destination;
            for (const GraphPatchRecord &patch_record : record.patches)
            {
                uint64_t address = 0;
                if (validate_patch_value(patch_record.patch, values, count,
                        reinterpret_cast<uint8_t *>(&address)) != 0)
                {
                    return 1;
                }
                if (patch_record.patch.target == MXX_GRAPH_PATCH_MEMCPY_1D_SRC)
                    source = reinterpret_cast<void *>(address);
                else
                    destination = reinterpret_cast<void *>(address);
            }
            if (record.bound && !reapply_top_level && source == record.bound_source &&
                destination == record.bound_destination)
                continue;
            error = update_on_device(record.device, exec->device, [&] {
                return gpuGraphExecMemcpyNodeSetParams1D(
                    exec->exec, record.node, destination, source, record.bytes, record.kind);
            });
            if (error != gpuSuccess)
            {
                return set_error(memcpy_rebind_error(error, record.bound_source,
                    record.bound_destination, source, destination, record.bytes, record.kind).c_str());
            }
            record.bound_source = source;
            record.bound_destination = destination;
            record.bound = true;
        }
        for (auto &record : exec->memsets)
        {
            uint64_t address = 0;
            if (validate_patch_value(record.patch.patch, values, count,
                    reinterpret_cast<uint8_t *>(&address)) != 0)
            {
                return 1;
            }
            if (record.bound && !reapply_top_level &&
                record.params.dst == reinterpret_cast<void *>(address))
                continue;
            record.params.dst = reinterpret_cast<void *>(address);
            error = update_on_device(record.device, exec->device, [&] {
                return gpuGraphExecMemsetNodeSetParams(
#if defined(MXX_GPU_BACKEND_HIP)
                    exec->schedule ? exec->schedule->executable(record.node) : exec->exec,
                    exec->schedule ? exec->schedule->node(record.node) : record.node,
#else
                    exec->exec, record.node,
#endif
                    &record.params);
            });
            if (error != gpuSuccess)
            {
                return set_error((std::string("memset rebind: ") + gpuGetErrorString(error)).c_str());
            }
            record.bound = true;
        }
        error = mxx_set_device(exec->device);
        if (error != gpuSuccess)
            return set_error((std::string("graph bind device: ") + gpuGetErrorString(error)).c_str());
        return 0;
    }

    int mxx_gpu_graph_launch(
        MxxGpuGraphExec *exec,
        void *launch_stream,
        MxxGpuNativeEvent **out_event)
    {
        if (!exec || !exec->exec || !launch_stream || !out_event)
        {
            return set_error("invalid mxx_gpu_graph_launch arguments");
        }
        *out_event = nullptr;
        gpuError_t error = mxx_set_device(exec->device);
        const gpuStream_t stream = reinterpret_cast<gpuStream_t>(launch_stream);
        auto *event = new (std::nothrow) MxxGpuNativeEvent();
        if (!event)
        {
            return set_error("failed to allocate CUDA graph completion state");
        }
        event->device = exec->device;
        if (error == gpuSuccess)
        {
            error = gpuEventCreateWithFlags(&event->event, gpuEventDisableTiming);
        }
        bool launch_attempted = false;
        if (error == gpuSuccess)
        {
            launch_attempted = true;
#if defined(MXX_GPU_BACKEND_HIP)
            error = exec->schedule ? exec->schedule->launch(stream) : gpuGraphLaunch(exec->exec, stream);
#else
            error = gpuGraphLaunch(exec->exec, stream);
#endif
        }
#if !defined(MXX_GPU_BACKEND_HIP)
        // Snapshot this launch's private branch records before a later launch
        // can re-record them. The public event remains unique per submission.
        // This only joins completion; graph branches still run independently.
        if (error == gpuSuccess)
        {
            for (auto completion : exec->branch_completion_events)
            {
                error = gpuStreamWaitEvent(stream, completion, 0);
                if (error != gpuSuccess) break;
            }
        }
#endif
        if (error == gpuSuccess)
        {
            error = gpuEventRecord(event->event, stream);
        }
        if (error != gpuSuccess)
        {
            // Once submission has been attempted, returning an ordinary error
            // permits Rust to drop all bound owners. Drain only this exceptional
            // path first; successful launches remain fully asynchronous. If the
            // stream cannot be drained, do not return into unsafe owner release.
            if (launch_attempted)
            {
                // A failed launch or event record can leave the stream in an
                // unknown state.  Never terminate the host process here: the
                // caller must receive a distinct status so it can retain the
                // bound owners and perform its deferred-reclamation policy.
                const gpuError_t drain = gpuStreamSynchronize(stream);
                if (drain != gpuSuccess)
                {
                    if (event->event) (void)gpuEventDestroy(event->event);
                    delete event;
                    last_error = "CUDA graph launch completion is uncertain; bound owners must be retained";
                    return GPU_STATUS_LAUNCH_UNCERTAIN;
                }
            }
            if (event->event) (void)gpuEventDestroy(event->event);
            delete event;
            return set_error(error);
        }
        if (exec->execution) {
            for (size_t partition = 0; partition < exec->execution->gpu_ids.size(); ++partition) {
                error = mxx_set_device(exec->execution->gpu_ids[partition]);
                if (error == gpuSuccess) {
                    error = gpuStreamWaitEvent(exec->execution->release_streams_by_partition[partition], event->event, 0);
                }
                if (error != gpuSuccess) break;
            }
            if (error == gpuSuccess) error = mxx_set_device(exec->device);
            if (error != gpuSuccess) {
                // Submission succeeded. Rust must retain the graph and all
                // bound owners even if retirement protection cannot be queued.
                set_error(error);
                mxx_gpu_native_event_destroy(event);
                return GPU_STATUS_LAUNCH_UNCERTAIN;
            }
        }
        *out_event = event;
        return 0;
    }

    int mxx_gpu_graph_schedule_metrics(MxxGpuGraphExec *exec, uint64_t *reads,
        uint64_t *bytes, double *wait_seconds, uint64_t *launches, double *host_seconds)
    {
        if (!exec || !reads || !bytes || !wait_seconds || !launches || !host_seconds)
            return set_error("invalid graph schedule metrics");
        *reads = *bytes = *launches = 0;
        *wait_seconds = *host_seconds = 0;
#if defined(MXX_GPU_BACKEND_HIP)
        if (exec->schedule)
        {
            *reads = exec->schedule->control_reads;
            *bytes = exec->schedule->control_bytes;
            *wait_seconds = exec->schedule->control_wait_seconds;
            *launches = exec->schedule->region_launches;
            *host_seconds = exec->schedule->host_schedule_seconds;
        }
#endif
        return 0;
    }

    void mxx_gpu_graph_exec_destroy(MxxGpuGraphExec *exec)
    {
        if (!exec) return;
        (void)mxx_set_device(exec->device);
        if (exec->exec)
        {
            const gpuError_t error = gpuGraphExecDestroy(exec->exec);
            if (error != gpuSuccess) set_error(error);
            exec->exec = nullptr;
        }
        if (exec->graph)
        {
            const gpuError_t error = gpuGraphDestroy(exec->graph);
            if (error != gpuSuccess) set_error(error);
            exec->graph = nullptr;
        }
#if !defined(MXX_GPU_BACKEND_HIP)
        for (auto event : exec->branch_completion_events) (void)gpuEventDestroy(event);
#endif
        for (void *buffer : exec->host_staging) (void)gpuFreeHost(buffer);
        delete exec;
    }

    int mxx_gpu_stream_record_event(void *stream_raw, int device, MxxGpuNativeEvent **out_event)
    {
        if (!stream_raw || !out_event) return set_error("invalid stream event record arguments");
        *out_event = nullptr;
        auto *event = new (std::nothrow) MxxGpuNativeEvent();
        if (!event) return set_error("failed to allocate stream event state");
        event->device = device;
        gpuError_t error = mxx_set_device(device);
        if (error == gpuSuccess)
            error = gpuEventCreateWithFlags(&event->event, gpuEventDisableTiming);
        if (error == gpuSuccess)
            error = gpuEventRecord(event->event, reinterpret_cast<gpuStream_t>(stream_raw));
        if (error != gpuSuccess)
        {
            mxx_gpu_native_event_destroy(event);
            return set_error(error);
        }
        *out_event = event;
        return 0;
    }

    int mxx_gpu_native_event_wait(MxxGpuNativeEvent *event)
    {
        if (!event || !event->event) return set_error("invalid CUDA graph event");
        gpuError_t error = mxx_set_device(event->device);
        if (error == gpuSuccess) error = gpuEventSynchronize(event->event);
        return error == gpuSuccess ? 0 : set_error(error);
    }

    int mxx_gpu_native_event_enqueue_wait(MxxGpuNativeEvent *event, void *stream_raw)
    {
        if (!event || !event->event || !stream_raw)
            return set_error("invalid CUDA graph event wait arguments");
        gpuError_t error = mxx_set_device(event->device);
        if (error == gpuSuccess)
        {
            error = gpuStreamWaitEvent(
                reinterpret_cast<gpuStream_t>(stream_raw), event->event, 0);
        }
        return error == gpuSuccess ? 0 : set_error(error);
    }

    int mxx_gpu_native_event_raw(MxxGpuNativeEvent *event, void **out_event)
    {
        if (!event || !event->event || !out_event)
        {
            return set_error("invalid CUDA graph event raw-handle arguments");
        }
        *out_event = reinterpret_cast<void *>(event->event);
        return 0;
    }

    void mxx_gpu_native_event_destroy(MxxGpuNativeEvent *event)
    {
        if (!event) return;
        (void)mxx_set_device(event->device);
        if (event->event)
        {
            const gpuError_t error = gpuEventDestroy(event->event);
            if (error != gpuSuccess) set_error(error);
        }
        delete event;
    }

    int gpu_device_buffer_copy_from_address(
        const void *source,
        int source_device,
        MxxGpuDeviceBuffer *destination,
        size_t bytes,
        void *stream_raw,
        MxxGpuNativeEvent **out_event)
    {
        if (!source || !destination || !stream_raw || !out_event ||
            bytes > destination->bytes)
        {
            return set_error("invalid gpu_device_buffer_copy_from_address arguments");
        }
        *out_event = nullptr;
        auto *event = new (std::nothrow) MxxGpuNativeEvent();
        if (!event) return set_error("failed to allocate device copy completion state");
        event->device = destination->device;
        const gpuStream_t stream = reinterpret_cast<gpuStream_t>(stream_raw);
        gpuError_t status = mxx_set_device(destination->device);
        if (status == gpuSuccess)
            status = gpuEventCreateWithFlags(&event->event, gpuEventDisableTiming);
        if (status == gpuSuccess && destination->producer_valid &&
            stream != destination->allocation_stream)
        {
            status = gpuStreamWaitEvent(stream, destination->producer, 0);
        }
        if (status == gpuSuccess)
        {
            const int source_physical = mxx_physical_device(source_device);
            const int destination_physical = mxx_physical_device(destination->device);
            status = source_physical == destination_physical ?
                gpuMemcpyAsync(destination->address, source, bytes,
                    gpuMemcpyDeviceToDevice, stream) :
                gpuMemcpyPeerAsync(destination->address, destination_physical,
                    source, source_physical, bytes, stream);
        }
        if (status == gpuSuccess) status = gpuEventRecord(destination->producer, stream);
        if (status == gpuSuccess) status = gpuEventRecord(event->event, stream);
        if (status != gpuSuccess)
        {
            mxx_gpu_native_event_destroy(event);
            return set_error(status);
        }
        destination->producer_valid = true;
        *out_event = event;
        return 0;
    }

}
