#include "Runtime.cuh"

#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <exception>
#include <limits>
#include <mutex>
#include <set>
#include <map>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
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

__global__ void mxx_export_slot_publish_kernel(MxxExportSlotHeader *header,
    uint64_t occurrence, uint64_t artifact_offset, uint64_t payload_bytes,
    uint32_t site, uint32_t flags)
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    header->occurrence = occurrence;
    header->artifact_offset = artifact_offset;
    header->payload_bytes = payload_bytes;
    header->site = site;
    header->flags = flags;
    __threadfence_system();
    atomicExch_system(reinterpret_cast<unsigned long long *>(&header->ready), 1ULL);
}

__global__ void mxx_dynamic_export_claim_kernel(
    const MxxDynamicExportEntry *table, uint32_t *claims,
    uint32_t *claim_result, const uint64_t *occurrence,
    uint32_t *status, size_t count)
{
    if (threadIdx.x || blockIdx.x) return;
    *claim_result = 0;
    if (*status != 0U) return;
    const uint64_t index = *occurrence;
    if (index >= count)
    {
        atomicCAS(status, 0U, 1U);
        return;
    }
    const auto entry = table[index];
    if (!entry.header_address || !entry.payload_address ||
        entry.payload_bytes > entry.payload_capacity || entry.occurrence != index)
    {
        atomicCAS(status, 0U, 3U);
        return;
    }
    if (atomicCAS(&claims[index], 0U, 1U) != 0U)
    {
        atomicCAS(status, 0U, 2U);
        return;
    }
    *claim_result = 1;
}

__global__ void mxx_dynamic_export_copy_kernel(
    const MxxDynamicExportEntry *table, const uint32_t *claim_result,
    const uint64_t *occurrence, const uint8_t *source,
    size_t byte_offset)
{
    if (*claim_result != 1U) return;
    const auto entry = table[*occurrence];
    const size_t index = byte_offset +
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < entry.payload_bytes)
        reinterpret_cast<uint8_t *>(entry.payload_address)[index] = source[index];
}

__global__ void mxx_dynamic_export_publish_kernel(
    const MxxDynamicExportEntry *table, const uint32_t *claim_result,
    const uint64_t *occurrence)
{
    if (threadIdx.x || blockIdx.x || *claim_result != 1U) return;
    const auto entry = table[*occurrence];
    auto *header = reinterpret_cast<MxxExportSlotHeader *>(entry.header_address);
    header->occurrence = entry.occurrence;
    header->artifact_offset = entry.artifact_offset;
    header->payload_bytes = entry.payload_bytes;
    header->site = entry.site;
    header->flags = entry.flags;
    __threadfence_system();
    atomicExch_system(reinterpret_cast<unsigned long long *>(&header->ready), 1ULL);
}

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000

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
        cudaEvent_t completion;
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

    int enqueue(int device, cudaEvent_t completion, std::vector<void *> &&pointers)
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
        cudaError_t error = mxx_set_device(job.device);
        if (error == cudaSuccess)
        {
            error = cudaEventSynchronize(job.completion);
        }
        if (error != cudaSuccess)
        {
            record_failure(cudaGetErrorString(error));
            // The event and pointers are intentionally leaked. Once
            // synchronization is uncertain, freeing host memory could race
            // with an in-flight asynchronous copy.
            return;
        }

        error = cudaEventDestroy(job.completion);
        if (error != cudaSuccess)
        {
            record_failure(cudaGetErrorString(error));
            // Keep the pointers leaked when event destruction is uncertain.
            return;
        }

        for (void *pointer : job.pointers)
        {
            if (!pointer)
            {
                continue;
            }
            error = cudaFreeHost(pointer);
            if (error != cudaSuccess)
            {
                // Do not retry an uncertain free. The failed pointer is
                // leaked, while independent pointers can still be reclaimed.
                record_failure(cudaGetErrorString(error));
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

    int set_error(cudaError_t err)
    {
        last_error = cudaGetErrorString(err);
        return err == cudaErrorMemoryAllocation ? GPU_STATUS_OUT_OF_MEMORY : 1;
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
            mxx_set_device(entry.device);
            cudaEventDestroy(entry.event);
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
            cudaStream_t stream = owner->release_streams_by_partition[partition];
            if (!stream)
            {
                continue;
            }
            cudaError_t err = mxx_set_device(device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
            cudaEvent_t epoch = nullptr;
            err = cudaEventCreateWithFlags(&epoch, cudaEventDisableTiming);
            if (err == cudaSuccess)
            {
                err = cudaEventRecord(epoch, stream);
            }
            if (err == cudaSuccess)
            {
                err = cudaEventSynchronize(epoch);
            }
            if (epoch)
            {
                cudaEventDestroy(epoch);
            }
            if (err != cudaSuccess)
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
            mxx_set_device(owner->gpu_ids[partition]);
            if (partition < owner->release_streams_by_partition.size() &&
                owner->release_streams_by_partition[partition])
            {
                cudaStreamDestroy(owner->release_streams_by_partition[partition]);
                owner->release_streams_by_partition[partition] = nullptr;
            }
            if (partition < owner->compute_streams_by_partition.size())
            {
                for (cudaStream_t &stream : owner->compute_streams_by_partition[partition])
                {
                    if (stream)
                    {
                        cudaStreamDestroy(stream);
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
            throw std::runtime_error("cannot query CUDA devices");
        }
        if (device_count <= 0)
        {
            throw std::runtime_error("no CUDA device available");
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
            cudaMemPool_t pool = nullptr;
            cudaError_t err = cudaDeviceGetDefaultMemPool(&pool, mxx_physical_device(device));
            if (err != cudaSuccess)
            {
                throw std::runtime_error(cudaGetErrorString(err));
            }
            err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
            if (err != cudaSuccess)
            {
                throw std::runtime_error(cudaGetErrorString(err));
            }
        }
    }

    // Cross-device graph copies go peer to peer where the hardware allows it:
    // each context's GPU gets access to every other physical GPU and to its
    // stream-ordered pool. Logical devices sharing one GPU need nothing.
    void enable_peer_access(const std::vector<int> &gpu_list)
    {
        static std::mutex mutex;
        static std::set<std::pair<int, int>> enabled;
        std::lock_guard<std::mutex> lock(mutex);
        int physical_count = 0;
        if (cudaGetDeviceCount(&physical_count) != cudaSuccess)
        {
            cudaGetLastError();
            return;
        }
        int current = 0;
        if (cudaGetDevice(&current) != cudaSuccess) current = -1;
        for (int device : gpu_list)
        {
            const int physical = mxx_physical_device(device);
            for (int peer = 0; peer < physical_count; ++peer)
            {
                int accessible = 0;
                if (peer == physical || enabled.count({physical, peer}) ||
                    cudaDeviceCanAccessPeer(&accessible, physical, peer) != cudaSuccess ||
                    !accessible)
                    continue;
                cudaMemPool_t pool = nullptr;
                cudaMemAccessDesc access{};
                access.location.type = cudaMemLocationTypeDevice;
                access.location.id = physical;
                access.flags = cudaMemAccessFlagsProtReadWrite;
                if (cudaSetDevice(physical) == cudaSuccess)
                {
                    const cudaError_t err = cudaDeviceEnablePeerAccess(peer, 0);
                    if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled)
                        enabled.insert({physical, peer});
                }
                // The peer's pool becomes readable and writable from this GPU.
                if (cudaDeviceGetDefaultMemPool(&pool, peer) == cudaSuccess)
                    cudaMemPoolSetAccess(pool, &access, 1);
                cudaGetLastError();
            }
        }
        if (current >= 0) cudaSetDevice(current);
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

    void free_ntt_device_constants_entry(GpuNttDeviceConstants &entry, cudaStream_t release = nullptr)
    {
        if (entry.device < 0)
        {
            return;
        }
        if (mxx_set_device(entry.device) != cudaSuccess)
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
                const cudaError_t status = release ? cudaFreeAsync(pointer, release) : cudaFree(pointer);
                if (status != cudaSuccess)
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
            cudaStream_t release = nullptr;
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
        cudaError_t err = mxx_set_device(device);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        const size_t limb_bytes = limb_count * sizeof(uint64_t);
        auto alloc_and_copy = [&](uint64_t **dst, const std::vector<uint64_t> &src)
        {
            err = cudaMalloc(reinterpret_cast<void **>(dst), limb_bytes);
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
            err = cudaMemcpy(*dst, src.data(), limb_bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
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

        cudaError_t err = mxx_set_device(device);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        const size_t twiddle_bytes = twiddle_count * sizeof(uint64_t);
        auto alloc_and_copy = [&](uint64_t **dst, const uint64_t *src)
        {
            cudaError_t local_err = cudaMalloc(reinterpret_cast<void **>(dst), twiddle_bytes);
            if (local_err != cudaSuccess)
            {
                throw std::runtime_error(cudaGetErrorString(local_err));
            }
            local_err = cudaMemcpy(*dst, src, twiddle_bytes, cudaMemcpyHostToDevice);
            if (local_err != cudaSuccess)
            {
                throw std::runtime_error(cudaGetErrorString(local_err));
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

extern "C" int gpu_set_last_error_cuda(int cuda_error)
{
    return set_error(static_cast<cudaError_t>(cuda_error));
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
            for (size_t partition = 0; partition < gpu_ctx->gpu_ids.size(); ++partition)
            {
                const int device = gpu_ctx->gpu_ids[partition];
                cudaError_t err = mxx_set_device(device);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(cudaGetErrorString(err));
                }
                auto &streams = gpu_ctx->execution->compute_streams_by_partition[partition];
                streams.resize(stream_pool_size, nullptr);
                for (cudaStream_t &stream : streams)
                {
                    err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                    if (err != cudaSuccess)
                    {
                        throw std::runtime_error(cudaGetErrorString(err));
                    }
                }
                err = cudaStreamCreateWithFlags(
                    &gpu_ctx->execution->release_streams_by_partition[partition],
                    cudaStreamNonBlocking);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(cudaGetErrorString(err));
                }
            }
            }
            gpu_ctx->ntt_device_constants.reserve(gpu_ctx->gpu_ids.size());
            for (int device : gpu_ctx->gpu_ids)
            {
                GpuNttDeviceConstants device_constants =
                    make_empty_ntt_device_constants(
                        device,
                        limb_count,
                        static_cast<uint32_t>(n_u64));
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
                gpu_ctx->ntt_device_constants.push_back(device_constants);
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
        cudaStream_t stream,
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

        cudaError_t error = mxx_set_device(device);
        cudaEvent_t completion = nullptr;
        if (error == cudaSuccess)
        {
            error = cudaEventCreateWithFlags(&completion, cudaEventDisableTiming);
        }
        if (error == cudaSuccess)
        {
            error = cudaEventRecord(completion, stream);
        }
        if (error != cudaSuccess)
        {
            if (completion)
            {
                // The event may have been recorded before the error was
                // reported. Keep it leaked with the pointers rather than
                // destroying an event that could still be in flight.
                completion = nullptr;
            }
            ctx->execution->pinned_host_reclaimer->record_uncertain(cudaGetErrorString(error));
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
        cudaMemPool_t pool = nullptr;
        cudaError_t err = cudaDeviceGetDefaultMemPool(&pool, mxx_physical_device(device));
        if (err != cudaSuccess)
        {
        return set_error(err);
        }
        uint64_t used_current = 0;
        uint64_t used_high = 0;
        uint64_t reserved_current = 0;
        err = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_current);
        if (err != cudaSuccess)
        {
        return set_error(err);
        }
        err = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_high);
        if (err != cudaSuccess)
        {
        return set_error(err);
        }
        err = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved_current);
        if (err != cudaSuccess)
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
        uint64_t *out_context_generation)
    {
        if (device < 0 || !out_name || name_capacity == 0 || !out_compute_major ||
            !out_uuid || uuid_capacity < 33 || !out_compute_minor ||
            !out_total_global_memory || !out_driver_version || !out_runtime_version ||
            !out_context_generation)
        {
            return set_error("invalid gpu_device_get_identity arguments");
        }
        cudaDeviceProp properties{};
        cudaError_t err = cudaGetDeviceProperties(&properties, mxx_physical_device(device));
        if (err != cudaSuccess)
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
        err = cudaDriverGetVersion(out_driver_version);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = cudaRuntimeGetVersion(out_runtime_version);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        size_t live_contexts = 0;
        if (gpu_device_context_state(device, &live_contexts, out_context_generation) != 0)
        {
            return 1;
        }
        *out_compute_major = properties.major;
        *out_compute_minor = properties.minor;
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
            cudaError_t err = mxx_set_device(entry.device);
            if (err != cudaSuccess)
            {
                return set_error(err);
            }
            err = cudaEventSynchronize(entry.event);
            if (err != cudaSuccess)
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
        cudaError_t err = cudaGetDeviceCount(&physical_count);
        if (err != cudaSuccess) return set_error(err);
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

    cudaError_t mxx_set_device(int logical)
    {
        const int physical = mxx_physical_device(logical);
        if (physical < 0) return cudaErrorInvalidDevice;
        const cudaError_t err = cudaSetDevice(physical);
        if (err == cudaSuccess) current_logical_device = logical;
        return err;
    }

    cudaError_t mxx_get_device(int *logical)
    {
        int physical = 0;
        const cudaError_t err = cudaGetDevice(&physical);
        if (err != cudaSuccess) return err;
        // The last logical device selected on this thread, when it is still
        // current; otherwise the first logical device of that physical one.
        if (current_logical_device >= 0 && mxx_physical_device(current_logical_device) == physical)
        {
            *logical = current_logical_device;
            return cudaSuccess;
        }
        const auto &table = logical_device_table();
        if (table.empty()) { *logical = physical; return cudaSuccess; }
        const auto found = std::find(table.begin(), table.end(), physical);
        if (found == table.end()) return cudaErrorInvalidDevice;
        *logical = static_cast<int>(found - table.begin());
        return cudaSuccess;
    }

    int gpu_device_count(int *out_count)
    {
        if (!out_count)
        {
            return set_error("invalid gpu_device_count arguments");
        }
        int count = 0;
        cudaError_t err = cudaGetDeviceCount(&count);
        if (err == cudaErrorNoDevice)
        {
            *out_count = 0;
            return 0;
        }
        if (err != cudaSuccess)
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
        cudaError_t err = cudaGetDevice(&current);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        err = mxx_set_device(device);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        err = cudaMemGetInfo(&free_bytes, &total_bytes);
        cudaError_t restore_err = cudaSetDevice(current);
        if (err != cudaSuccess)
        {
            return set_error(err);
        }
        if (restore_err != cudaSuccess)
        {
            return set_error(restore_err);
        }
        *out_free = free_bytes;
        *out_total = total_bytes;
        return 0;
    }

    int gpu_device_synchronize()
    {
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
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
            cudaError_t err = cudaMallocHost(&ptr, bytes);
            if (err != cudaSuccess)
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
        cudaError_t err = cudaFreeHost(ptr);
        if (err != cudaSuccess)
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
        const cudaError_t device_error = mxx_set_device(physical_device);
        if (device_error != cudaSuccess) return set_error(device_error);
        void *host = nullptr;
        cudaError_t error = cudaHostAlloc(&host,
            sizeof(MxxExportSlotHeader) + payload_capacity, cudaHostAllocMapped);
        if (error != cudaSuccess) return set_error(error);
        void *device = nullptr;
        error = cudaHostGetDevicePointer(&device, host, 0);
        if (error != cudaSuccess)
        {
            cudaFreeHost(host);
            return set_error(error);
        }
        memset(host, 0, sizeof(MxxExportSlotHeader));
        *out_host = host;
        *out_device = device;
        return 0;
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
        buffer->allocation_stream = reinterpret_cast<cudaStream_t>(stream_raw);
        buffer->bytes = bytes;
        cudaError_t status = mxx_get_device(&buffer->device);
        if (status == cudaSuccess)
        {
            status = cudaMallocAsync(
                reinterpret_cast<void **>(&buffer->address), bytes, buffer->allocation_stream);
        }
        if (status == cudaSuccess)
        {
            status = cudaEventCreateWithFlags(&buffer->producer, cudaEventDisableTiming);
        }
        if (status == cudaSuccess)
        {
            status = cudaEventRecord(buffer->producer, buffer->allocation_stream);
        }
        if (status != cudaSuccess)
        {
            if (buffer->producer) cudaEventDestroy(buffer->producer);
            if (buffer->address) cudaFreeAsync(buffer->address, buffer->allocation_stream);
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
        cudaError_t status = mxx_set_device(buffer->device);
        if (status == cudaSuccess && buffer->producer_valid)
            status = cudaStreamWaitEvent(buffer->allocation_stream, buffer->producer, 0);
        if (status == cudaSuccess)
            status = cudaFreeAsync(buffer->address, buffer->allocation_stream);
        if (buffer->producer)
        {
            const cudaError_t event_status = cudaEventDestroy(buffer->producer);
            if (status == cudaSuccess) status = event_status;
        }
        delete buffer;
        return status == cudaSuccess ? 0 : set_error(status);
    }

    int gpu_device_buffer_upload(
        MxxGpuDeviceBuffer *buffer,
        size_t offset,
        const void *source,
        size_t bytes)
    {
        if (!buffer || !source || offset > buffer->bytes || bytes > buffer->bytes - offset)
            return set_error("invalid gpu_device_buffer_upload arguments");
        cudaError_t status = cudaMemcpyAsync(
            buffer->address + offset,
            source,
            bytes,
            cudaMemcpyHostToDevice,
            buffer->allocation_stream);
        if (status == cudaSuccess)
            status = cudaEventRecord(buffer->producer, buffer->allocation_stream);
        if (status != cudaSuccess)
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
        cudaError_t status = mxx_set_device(buffer->device);
        if (status == cudaSuccess && buffer->producer_valid)
            status = cudaStreamWaitEvent(buffer->allocation_stream, buffer->producer, 0);
        if (status == cudaSuccess)
        {
            status = cudaMemcpyAsync(
                destination,
                buffer->address + offset,
                bytes,
                cudaMemcpyDeviceToHost,
                buffer->allocation_stream);
        }
        if (status == cudaSuccess)
            status = cudaStreamSynchronize(buffer->allocation_stream);
        return status == cudaSuccess ? 0 : set_error(status);
    }

    int gpu_context_download_address(
        GpuContext *ctx, int device, const void *address,
        void *destination, size_t bytes)
    {
        if (!ctx || !address || !destination || bytes == 0 ||
            std::find(ctx->gpu_ids.begin(), ctx->gpu_ids.end(), device) == ctx->gpu_ids.end())
            return set_error("invalid resident address download arguments");
        cudaError_t status = mxx_set_device(device);
        cudaPointerAttributes attributes{};
        if (status == cudaSuccess) status = cudaPointerGetAttributes(&attributes, address);
        if (status != cudaSuccess) return set_error(status);
        if (attributes.device != mxx_physical_device(device) || attributes.type != cudaMemoryTypeDevice)
            return set_error("resident download address belongs to another device or memory type");
        if (bytes > UINTPTR_MAX - reinterpret_cast<uintptr_t>(address))
            return set_error("resident download address overflows");
        status = cudaMemcpy(destination, address, bytes, cudaMemcpyDeviceToHost);
        return status == cudaSuccess ? 0 : set_error(status);
    }

    int gpu_device_buffer_wait_compiled_inputs(
        const MxxGpuDeviceBuffer *buffer,
        int consumer_device,
        void *consumer_stream_raw,
        bool read_only)
    {
        (void)read_only;
        if (!buffer || !consumer_stream_raw || consumer_device < 0)
            return set_error("invalid gpu_device_buffer_wait_compiled_inputs arguments");
        cudaError_t status = mxx_set_device(consumer_device);
        if (status == cudaSuccess && buffer->producer_valid &&
            reinterpret_cast<cudaStream_t>(consumer_stream_raw) != buffer->allocation_stream)
        {
            status = cudaStreamWaitEvent(
                reinterpret_cast<cudaStream_t>(consumer_stream_raw), buffer->producer, 0);
        }
        return status == cudaSuccess ? 0 : set_error(status);
    }

    int gpu_device_buffer_wait(const MxxGpuDeviceBuffer *buffer)
    {
        if (!buffer || !buffer->producer_valid)
            return buffer ? 0 : set_error("invalid gpu_device_buffer_wait arguments");
        cudaError_t status = mxx_set_device(buffer->device);
        if (status == cudaSuccess)
            status = cudaEventSynchronize(buffer->producer);
        return status == cudaSuccess ? 0 : set_error(status);
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
        cudaGraphNode_t node = nullptr;
        cudaKernelNodeParams launch{};
        std::vector<std::vector<uint8_t>> arguments;
        std::vector<void *> argument_pointers;
        std::vector<GraphPatchRecord> patches;
    };

    struct GraphMemcpyUpdateRecord
    {
        cudaGraphNode_t node = nullptr;
        void *source = nullptr;
        void *destination = nullptr;
        size_t bytes = 0;
        cudaMemcpyKind kind = cudaMemcpyDefault;
        std::vector<GraphPatchRecord> patches;
    };

    struct GraphMemsetUpdateRecord
    {
        cudaGraphNode_t node = nullptr;
        cudaMemsetParams params{};
        GraphPatchRecord patch{};
    };

    struct MxxGpuGraphExec
    {
        int device = -1;
        cudaStream_t default_stream = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        std::vector<GraphKernelUpdateRecord> kernels;
        std::vector<GraphMemcpyUpdateRecord> memcpys;
        std::vector<GraphMemsetUpdateRecord> memsets;
        // Child-graph records are patched before the conditional child graph
        // is rebound; dropping them would retain stale plan-time pointers.
        std::vector<GraphKernelUpdateRecord> body_kernels;
        std::vector<GraphMemcpyUpdateRecord> body_memcpys;
        std::vector<GraphMemsetUpdateRecord> body_memsets;
    };

    struct MxxGpuGraphBuilder
    {
        GpuContext *context = nullptr;
        int device = -1;
        cudaStream_t stream = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraph_t root_graph = nullptr;
        bool conditional_body_active = false;
        MxxPreimageRetrySpec retry_spec{};
        void *retry_scratch = nullptr;
        void *retry_control = nullptr;
        void *retry_status = nullptr;
        bool generic_body_mode = false;
        std::vector<cudaGraphNode_t> body_terminals;
        // Last conditional node emitted into the open conditional body.
        // Concurrent sibling conditionals inside one body graph do not
        // complete on the device, so each is ordered after the previous one.
        cudaGraphNode_t last_body_conditional = nullptr;
        // One frame per open conditional body, innermost last. A body may
        // itself contain conditional operations, so the enclosing scope's
        // graph, operation state and WHILE control are restored on finish.
        struct ConditionalFrame
        {
            cudaGraph_t parent_graph = nullptr;
            cudaGraphNode_t conditional_node = nullptr;
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
            cudaGraphConditionalHandle handle = 0;
#endif
            uint32_t parent_operation_index = 0;
            std::vector<cudaGraphNode_t> parent_operation_nodes;
            std::vector<MxxGraphBindingMapEntry> parent_binding_map;
            std::vector<cudaGraphNode_t> parent_body_terminals;
            bool parent_conditional_body_active = false;
            bool parent_generic_body_mode = false;
            cudaGraphNode_t parent_last_body_conditional = nullptr;
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
        bool operation_active = false;
        uint32_t operation_index = 0;
        std::vector<cudaGraphNode_t> frontier;
        std::vector<cudaGraphNode_t> operation_nodes;
        std::vector<cudaGraphNode_t> terminals;
        std::vector<GraphKernelUpdateRecord> kernels;
        std::vector<GraphMemcpyUpdateRecord> memcpys;
        std::vector<GraphMemsetUpdateRecord> memsets;
        std::vector<GraphKernelUpdateRecord> body_kernels;
        std::vector<GraphMemcpyUpdateRecord> body_memcpys;
        std::vector<GraphMemsetUpdateRecord> body_memsets;
        struct ResidentAddress { uint64_t address; size_t bytes; uint32_t binding; };
        std::vector<ResidentAddress> resident_addresses;
        std::vector<MxxGraphBindingMapEntry> binding_map;
        ~MxxGpuGraphBuilder() { if (root_graph) cudaGraphDestroy(root_graph); }
    };

    struct MxxGpuNativeEvent
    {
        int device = -1;
        cudaEvent_t event = nullptr;
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
        if (mxx_set_device(device) != cudaSuccess) return set_error(cudaGetLastError());
        auto *builder = new (std::nothrow) MxxGpuGraphBuilder();
        if (!builder) return set_error("failed to allocate graph builder");
        builder->context = ctx;
        builder->device = device;
        builder->stream = reinterpret_cast<cudaStream_t>(stream);
        const cudaError_t error = cudaGraphCreate(&builder->graph, 0);
        if (error != cudaSuccess) { delete builder; return set_error(error); }
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
        cudaGraphNode_t terminal = nullptr;
        if (builder->operation_nodes.empty())
        {
            const cudaError_t error = cudaGraphAddEmptyNode(&terminal, builder->graph,
                builder->frontier.data(), builder->frontier.size());
            if (error != cudaSuccess) return set_error(error);
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
            const cudaError_t error = cudaGraphAddKernelNode(&record.node, builder->graph,
                builder->frontier.data(), builder->frontier.size(), &record.launch);
            if (error != cudaSuccess) return set_error(error);
            builder->frontier.assign(1, record.node);
            builder->operation_nodes.push_back(record.node);
            if (builder->conditional_body_active)
                builder->body_kernels.push_back(std::move(record));
            else builder->kernels.push_back(std::move(record));
            return 0;
        }
        catch (const std::exception &error) { return set_error(error); }
    }

    int mxx_gpu_graph_builder_add_memcpy(MxxGpuGraphBuilder *builder,
        void *destination, const void *source, size_t bytes, int copy_kind,
        const MxxGraphPatch *patches, size_t patch_count)
    {
        if (!builder || !builder->operation_active || !destination || !source || !bytes ||
            (patch_count && !patches)) return set_error("invalid explicit memcpy node");
        GraphMemcpyUpdateRecord record;
        record.source = const_cast<void *>(source);
        record.destination = destination;
        record.bytes = bytes;
        record.kind = static_cast<cudaMemcpyKind>(copy_kind);
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
        const cudaError_t error = cudaGraphAddMemcpyNode1D(&record.node, builder->graph,
            builder->frontier.data(), builder->frontier.size(), destination, source,
            bytes, record.kind);
        if (error != cudaSuccess) return set_error(error);
        builder->frontier.assign(1, record.node);
        builder->operation_nodes.push_back(record.node);
        if (builder->conditional_body_active)
            builder->body_memcpys.push_back(std::move(record));
        else builder->memcpys.push_back(std::move(record));
        return 0;
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
        const cudaError_t error = cudaGraphAddMemsetNode(&record.node, builder->graph,
            builder->frontier.data(), builder->frontier.size(), &record.params);
        if (error != cudaSuccess) return set_error(error);
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
        uint32_t header_binding)
    {
        if (!builder || !device_header || (flags & ~1U) != 0)
            return set_error("invalid explicit export publication");
        const void *arguments[] = {&device_header, &occurrence, &artifact_offset,
            &payload_bytes, &site, &flags};
        const size_t sizes[] = {sizeof(device_header), sizeof(occurrence),
            sizeof(artifact_offset), sizeof(payload_bytes), sizeof(site), sizeof(flags)};
        const MxxGraphPatch patch{nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
            0, 0, sizeof(device_header), header_binding, 0};
        return mxx_gpu_graph_builder_add_kernel(builder,
            reinterpret_cast<const void *>(mxx_export_slot_publish_kernel),
            1, 1, 1, 1, 1, 1, 0, arguments, sizes, 6, &patch, 1);
    }

    int mxx_gpu_graph_builder_add_dynamic_export(
        MxxGpuGraphBuilder *builder, const MxxDynamicExportEntry *table,
        uint32_t *claims, uint32_t *claim_result,
        const uint64_t *occurrence, const void *source,
        uint32_t *status, size_t count, size_t maximum_payload_bytes,
        uint32_t table_binding, uint32_t claims_binding,
        uint32_t claim_result_binding, uint32_t occurrence_binding,
        uint32_t source_binding, uint32_t status_binding)
    {
        if (!builder || !builder->operation_active || !table || !claims ||
            !claim_result || !occurrence || !source || !status ||
            !count || !maximum_payload_bytes)
            return set_error("invalid dynamic export graph arguments");
        auto pointer_patch = [](uint32_t argument, uint32_t binding) {
            return MxxGraphPatch{nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
                argument, 0, sizeof(void *), binding, 0};
        };
        const void *claim_arguments[] = {
            &table, &claims, &claim_result, &occurrence, &status, &count};
        const size_t claim_sizes[] = {
            sizeof(table), sizeof(claims), sizeof(claim_result),
            sizeof(occurrence), sizeof(status), sizeof(count)};
        const MxxGraphPatch claim_patches[] = {
            pointer_patch(0, table_binding), pointer_patch(1, claims_binding),
            pointer_patch(2, claim_result_binding),
            pointer_patch(3, occurrence_binding), pointer_patch(4, status_binding)};
        int result = mxx_gpu_graph_builder_add_kernel(builder,
            reinterpret_cast<const void *>(mxx_dynamic_export_claim_kernel),
            1, 1, 1, 1, 1, 1, 0, claim_arguments, claim_sizes, 6,
            claim_patches, std::size(claim_patches));
        if (result != 0) return result;
        const auto *source_bytes = static_cast<const uint8_t *>(source);
        const MxxGraphPatch copy_patches[] = {
            pointer_patch(0, table_binding), pointer_patch(1, claim_result_binding),
            pointer_patch(2, occurrence_binding), pointer_patch(3, source_binding)};
        constexpr size_t max_copy_chunk = 65535ULL * 256ULL;
        for (size_t offset = 0; offset < maximum_payload_bytes; offset += max_copy_chunk)
        {
            const size_t chunk = std::min(max_copy_chunk, maximum_payload_bytes - offset);
            const void *copy_arguments[] = {
                &table, &claim_result, &occurrence, &source_bytes, &offset};
            const size_t copy_sizes[] = {
                sizeof(table), sizeof(claim_result), sizeof(occurrence),
                sizeof(source_bytes), sizeof(offset)};
            result = mxx_gpu_graph_builder_add_kernel(builder,
                reinterpret_cast<const void *>(mxx_dynamic_export_copy_kernel),
                static_cast<uint32_t>((chunk + 255) / 256), 1, 1, 256, 1, 1,
                0, copy_arguments, copy_sizes, 5,
                copy_patches, std::size(copy_patches));
            if (result != 0) return result;
        }
        const void *publish_arguments[] = {&table, &claim_result, &occurrence};
        const size_t publish_sizes[] = {
            sizeof(table), sizeof(claim_result), sizeof(occurrence)};
        const MxxGraphPatch publish_patches[] = {
            pointer_patch(0, table_binding), pointer_patch(1, claim_result_binding),
            pointer_patch(2, occurrence_binding)};
        return mxx_gpu_graph_builder_add_kernel(builder,
            reinterpret_cast<const void *>(mxx_dynamic_export_publish_kernel),
            1, 1, 1, 1, 1, 1, 0,
            publish_arguments, publish_sizes, 3,
            publish_patches, std::size(publish_patches));
    }

    static MxxGraphPatch mxx_direct_pointer_patch(uint32_t argument_index,
        uint32_t binding)
    {
        return MxxGraphPatch{nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD,
            argument_index, 0, sizeof(void *), binding, 0};
    }

    static int mxx_gpu_graph_builder_enter_generic_body(MxxGpuGraphBuilder *builder,
        cudaGraphConditionalHandle handle, cudaGraphConditionalNodeType type,
        const MxxGpuGraphBuilder::ConditionalFrame *loop)
    {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        cudaGraphNodeParams parameters{};
        parameters.type = cudaGraphNodeTypeConditional;
        parameters.conditional.handle = handle;
        parameters.conditional.type = type;
        parameters.conditional.size = 1;
        cudaGraphNode_t conditional = nullptr;
        // The node joins the scope that is open now: the root graph, or the
        // body graph of an enclosing conditional.
        const cudaError_t error = cudaGraphAddNode(&conditional, builder->graph,
            builder->frontier.data(), nullptr, builder->frontier.size(), &parameters);
        if (error != cudaSuccess) return set_error(error);
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

    int mxx_gpu_graph_builder_begin_if(MxxGpuGraphBuilder *builder,
        const uint64_t *predicate, uint32_t predicate_binding)
    {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        if (!builder || !builder->operation_active ||
            (builder->conditional_body_active && !builder->generic_body_mode) ||
            !predicate) return set_error("invalid direct IF body");
        mxx_gpu_graph_builder_order_body_conditional(builder);
        cudaGraphConditionalHandle handle = 0;
        cudaError_t error = cudaGraphConditionalHandleCreate(&handle,
            builder->graph, 1U, cudaGraphCondAssignDefault);
        if (error != cudaSuccess) return set_error(error);
        const void *arguments[] = {&predicate, &handle};
        const size_t sizes[] = {sizeof(predicate), sizeof(handle)};
        const MxxGraphPatch patch = mxx_direct_pointer_patch(0, predicate_binding);
        int status = mxx_gpu_graph_builder_add_kernel(builder,
            reinterpret_cast<const void *>(mxx_if_gate_kernel),
            1, 1, 1, 1, 1, 1, 0, arguments, sizes, 2, &patch, 1);
        if (status != 0) return status;
        return mxx_gpu_graph_builder_enter_generic_body(
            builder, handle, cudaGraphCondTypeIf, nullptr);
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
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        if (!builder || !builder->operation_active ||
            (builder->conditional_body_active && !builder->generic_body_mode) ||
            !index || !limit || !status_word || !max_iterations)
            return set_error("invalid direct WHILE body");
        mxx_gpu_graph_builder_order_body_conditional(builder);
        cudaGraphConditionalHandle handle = 0;
        cudaError_t error = cudaGraphConditionalHandleCreate(&handle,
            builder->graph, 1U, cudaGraphCondAssignDefault);
        if (error != cudaSuccess) return set_error(error);
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
#else
        (void)builder; (void)index; (void)limit; (void)max_iterations;
        (void)status_word; (void)index_binding; (void)limit_binding; (void)status_binding;
        return GPU_STATUS_CONDITIONAL_UNSUPPORTED;
#endif
    }

    int mxx_gpu_graph_builder_finish_generic_body(MxxGpuGraphBuilder *builder)
    {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        if (!builder || !builder->generic_body_mode || !builder->conditional_body_active ||
            builder->operation_active)
            return set_error("invalid direct conditional body completion");
        if (builder->body_terminals.empty())
        {
            cudaGraphNode_t empty = nullptr;
            const cudaError_t error = cudaGraphAddEmptyNode(
                &empty, builder->graph, nullptr, 0);
            if (error != cudaSuccess) return set_error(error);
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
        builder->last_body_conditional = builder->conditional_body_active ?
            frame.conditional_node : nullptr;
        return 0;
#else
        (void)builder;
        return GPU_STATUS_CONDITIONAL_UNSUPPORTED;
#endif
    }

    int mxx_gpu_graph_builder_finish(MxxGpuGraphBuilder *builder, MxxGpuGraphExec **out_exec)
    {
        if (!builder || !out_exec || builder->operation_active ||
            builder->conditional_body_active || builder->terminals.empty())
            return set_error("invalid explicit graph finish");
        *out_exec = nullptr;
        auto *result = new (std::nothrow) MxxGpuGraphExec();
        if (!result) return set_error("failed to allocate explicit graph exec");
        result->device = builder->device;
        result->default_stream = builder->stream;
        const cudaError_t error = cudaGraphInstantiateWithFlags(&result->exec,
            builder->root_graph, cudaGraphInstantiateFlagAutoFreeOnLaunch);
        if (error != cudaSuccess) { delete result; return set_error(error); }
        result->graph = builder->root_graph;
        builder->root_graph = nullptr;
        builder->graph = nullptr;
        result->kernels = std::move(builder->kernels);
        result->memcpys = std::move(builder->memcpys);
        result->memsets = std::move(builder->memsets);
        result->body_kernels = std::move(builder->body_kernels);
        result->body_memcpys = std::move(builder->body_memcpys);
        result->body_memsets = std::move(builder->body_memsets);
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
                builder->stream == reinterpret_cast<cudaStream_t>(stream);
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
        const MxxGraphPatch *patches, size_t patch_count)
    {
        if (!ctx || !stream || !function) return set_error("invalid kernel dispatch");
        if (auto *builder = mxx_gpu_graph_builder_for_stream(ctx, stream))
        {
            return mxx_gpu_graph_builder_add_kernel(builder, function, grid_x, grid_y,
                grid_z, block_x, block_y, block_z, shared_bytes,
                const_cast<const void *const *>(arguments), argument_sizes,
                argument_count, patches, patch_count);
        }
        {
            auto &owner = *ctx->execution;
            std::lock_guard<std::mutex> lock(owner.graph_mutex);
            if (owner.explicit_builder)
                return set_error("kernel dispatch escaped the active explicit graph operation");
        }
        const cudaError_t error = cudaLaunchKernel(function,
            dim3(grid_x, grid_y, grid_z), dim3(block_x, block_y, block_z),
            arguments, shared_bytes, reinterpret_cast<cudaStream_t>(stream));
        return error == cudaSuccess ? 0 : set_error(error);
    }

    int mxx_gpu_graph_upload(MxxGpuGraphExec *exec, void *launch_stream)
    {
        if (!exec || !exec->exec || !launch_stream)
        {
            return set_error("invalid mxx_gpu_graph_upload arguments");
        }
        cudaError_t error = mxx_set_device(exec->device);
        if (error == cudaSuccess)
        {
            error = cudaGraphUpload(exec->exec, reinterpret_cast<cudaStream_t>(launch_stream));
        }
        return error == cudaSuccess ? 0 : set_error(error);
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
        cudaError_t error = mxx_set_device(exec->device);
        if (error != cudaSuccess)
        {
            return set_error(error);
        }
        // Body launch records are attached to the explicit child graph. They
        // must be patched as part of every bind just like top-level records;
        // otherwise conditional retry replay would dereference stale
        // plan-time pointers. CUDA accepts these updates on the retained
        // child graph before its parent executable is launched.
        for (auto &record : exec->body_kernels)
        {
            for (const GraphPatchRecord &patch_record : record.patches)
            {
                const MxxGraphPatch &patch = patch_record.patch;
                if (patch.byte_offset > record.arguments[patch.argument_index].size() ||
                    patch.byte_count > record.arguments[patch.argument_index].size() - patch.byte_offset ||
                    validate_patch_value(
                        patch,
                        values,
                        count,
                        record.arguments[patch.argument_index].data() + patch.byte_offset) != 0)
                    return 1;
            }
            for (size_t index = 0; index < record.arguments.size(); ++index)
                record.argument_pointers[index] = record.arguments[index].data();
            record.launch.kernelParams = record.argument_pointers.data();
            error = cudaGraphKernelNodeSetParams(record.node, &record.launch);
            if (error != cudaSuccess) return set_error(error);
        }
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
            error = cudaGraphMemcpyNodeSetParams1D(
                record.node, destination, source, record.bytes, record.kind);
            if (error != cudaSuccess) return set_error(error);
        }
        for (auto &record : exec->body_memsets)
        {
            uint64_t address = 0;
            if (validate_patch_value(record.patch.patch, values, count,
                                     reinterpret_cast<uint8_t *>(&address)) != 0)
                return 1;
            record.params.dst = reinterpret_cast<void *>(address);
            error = cudaGraphMemsetNodeSetParams(record.node, &record.params);
            if (error != cudaSuccess) return set_error(error);
        }
        if (!exec->body_kernels.empty() || !exec->body_memcpys.empty() ||
            !exec->body_memsets.empty())
        {
            // Body records refer to the retained child graph nodes rather than
            // executable top-level nodes.  Force CUDA to reconcile the
            // modified child graph with the instantiated parent; never report
            // a successful bind while silently leaving a stale child graph.
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13000
            cudaGraphExecUpdateResultInfo update_info{};
            error = cudaGraphExecUpdate(exec->exec, exec->graph, &update_info);
            if (error != cudaSuccess || update_info.result != cudaGraphExecUpdateSuccess)
                return set_error(error == cudaSuccess ? cudaErrorGraphExecUpdateFailure : error);
#else
            cudaGraphNode_t update_error_node = nullptr;
            cudaGraphExecUpdateResult update_result = cudaGraphExecUpdateError;
            error = cudaGraphExecUpdate(exec->exec, exec->graph, &update_error_node, &update_result);
            if (error != cudaSuccess || update_result != cudaGraphExecUpdateSuccess)
                return set_error(error == cudaSuccess ? cudaErrorGraphExecUpdateFailure : error);
#endif
        }
        for (auto &record : exec->kernels)
        {
            for (const GraphPatchRecord &patch_record : record.patches)
            {
                const MxxGraphPatch &patch = patch_record.patch;
                if (patch.byte_offset > record.arguments[patch.argument_index].size() ||
                    patch.byte_count > record.arguments[patch.argument_index].size() - patch.byte_offset ||
                    validate_patch_value(patch, values, count,
                        record.arguments[patch.argument_index].data() + patch.byte_offset) != 0)
                {
                    return 1;
                }
            }
            for (size_t index = 0; index < record.arguments.size(); ++index)
            {
                record.argument_pointers[index] = record.arguments[index].data();
            }
            record.launch.kernelParams = record.argument_pointers.data();
            error = cudaGraphExecKernelNodeSetParams(exec->exec, record.node, &record.launch);
            if (error != cudaSuccess)
            {
                return set_error(error);
            }
        }
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
            error = cudaGraphExecMemcpyNodeSetParams1D(
                exec->exec, record.node, destination, source, record.bytes, record.kind);
            if (error != cudaSuccess)
            {
                return set_error(error);
            }
        }
        for (auto &record : exec->memsets)
        {
            uint64_t address = 0;
            if (validate_patch_value(record.patch.patch, values, count,
                    reinterpret_cast<uint8_t *>(&address)) != 0)
            {
                return 1;
            }
            record.params.dst = reinterpret_cast<void *>(address);
            error = cudaGraphExecMemsetNodeSetParams(exec->exec, record.node, &record.params);
            if (error != cudaSuccess)
            {
                return set_error(error);
            }
        }
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
        cudaError_t error = mxx_set_device(exec->device);
        const cudaStream_t stream = reinterpret_cast<cudaStream_t>(launch_stream);
        auto *event = new (std::nothrow) MxxGpuNativeEvent();
        if (!event)
        {
            return set_error("failed to allocate CUDA graph completion state");
        }
        event->device = exec->device;
        if (error == cudaSuccess)
        {
            error = cudaEventCreateWithFlags(&event->event, cudaEventDisableTiming);
        }
        bool launch_attempted = false;
        if (error == cudaSuccess)
        {
            launch_attempted = true;
            error = cudaGraphLaunch(exec->exec, stream);
        }
        if (error == cudaSuccess)
        {
            error = cudaEventRecord(event->event, stream);
        }
        if (error != cudaSuccess)
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
                const cudaError_t drain = cudaStreamSynchronize(stream);
                if (drain != cudaSuccess)
                {
                    if (event->event) cudaEventDestroy(event->event);
                    delete event;
                    last_error = "CUDA graph launch completion is uncertain; bound owners must be retained";
                    return GPU_STATUS_LAUNCH_UNCERTAIN;
                }
            }
            if (event->event) cudaEventDestroy(event->event);
            delete event;
            return set_error(error);
        }
        *out_event = event;
        return 0;
    }

    void mxx_gpu_graph_exec_destroy(MxxGpuGraphExec *exec)
    {
        if (!exec) return;
        mxx_set_device(exec->device);
        if (exec->exec)
        {
            const cudaError_t error = cudaGraphExecDestroy(exec->exec);
            if (error != cudaSuccess) set_error(error);
            exec->exec = nullptr;
        }
        if (exec->graph)
        {
            const cudaError_t error = cudaGraphDestroy(exec->graph);
            if (error != cudaSuccess) set_error(error);
            exec->graph = nullptr;
        }
        delete exec;
    }

    int mxx_gpu_stream_record_event(void *stream_raw, int device, MxxGpuNativeEvent **out_event)
    {
        if (!stream_raw || !out_event) return set_error("invalid stream event record arguments");
        *out_event = nullptr;
        auto *event = new (std::nothrow) MxxGpuNativeEvent();
        if (!event) return set_error("failed to allocate stream event state");
        event->device = device;
        cudaError_t error = mxx_set_device(device);
        if (error == cudaSuccess)
            error = cudaEventCreateWithFlags(&event->event, cudaEventDisableTiming);
        if (error == cudaSuccess)
            error = cudaEventRecord(event->event, reinterpret_cast<cudaStream_t>(stream_raw));
        if (error != cudaSuccess)
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
        cudaError_t error = mxx_set_device(event->device);
        if (error == cudaSuccess) error = cudaEventSynchronize(event->event);
        return error == cudaSuccess ? 0 : set_error(error);
    }

    int mxx_gpu_native_event_enqueue_wait(MxxGpuNativeEvent *event, void *stream_raw)
    {
        if (!event || !event->event || !stream_raw)
            return set_error("invalid CUDA graph event wait arguments");
        cudaError_t error = mxx_set_device(event->device);
        if (error == cudaSuccess)
        {
            error = cudaStreamWaitEvent(
                reinterpret_cast<cudaStream_t>(stream_raw), event->event, 0);
        }
        return error == cudaSuccess ? 0 : set_error(error);
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
        mxx_set_device(event->device);
        if (event->event)
        {
            const cudaError_t error = cudaEventDestroy(event->event);
            if (error != cudaSuccess) set_error(error);
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
        const cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
        cudaError_t status = mxx_set_device(destination->device);
        if (status == cudaSuccess)
            status = cudaEventCreateWithFlags(&event->event, cudaEventDisableTiming);
        if (status == cudaSuccess && destination->producer_valid &&
            stream != destination->allocation_stream)
        {
            status = cudaStreamWaitEvent(stream, destination->producer, 0);
        }
        if (status == cudaSuccess)
        {
            const int source_physical = mxx_physical_device(source_device);
            const int destination_physical = mxx_physical_device(destination->device);
            status = source_physical == destination_physical ?
                cudaMemcpyAsync(destination->address, source, bytes,
                    cudaMemcpyDeviceToDevice, stream) :
                cudaMemcpyPeerAsync(destination->address, destination_physical,
                    source, source_physical, bytes, stream);
        }
        if (status == cudaSuccess) status = cudaEventRecord(destination->producer, stream);
        if (status == cudaSuccess) status = cudaEventRecord(event->event, stream);
        if (status != cudaSuccess)
        {
            mxx_gpu_native_event_destroy(event);
            return set_error(status);
        }
        destination->producer_valid = true;
        *out_event = event;
        return 0;
    }

}
