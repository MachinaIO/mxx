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
#include <map>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
__global__ void mxx_preimage_retry_gate_kernel(
    MxxPreimageStatus *status,
    uint32_t max_attempts,
    cudaGraphConditionalHandle handle,
    void *fixed_scratch,
    void *device_control)
{
    (void)fixed_scratch;
    (void)device_control;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (status->accepted == 1U)
    {
        cudaGraphSetConditional(handle, 0U);
        return;
    }
    if (status->accepted > 1U || status->error_code != MXX_PREIMAGE_SUCCESS)
    {
        // A malformed multi-accept or an upstream sampler error must never
        // turn into an unbounded conditional loop.
        status->error_code = status->error_code == MXX_PREIMAGE_SUCCESS
            ? MXX_PREIMAGE_EXHAUSTED
            : status->error_code;
        cudaGraphSetConditional(handle, 0U);
        return;
    }
    if (status->attempts >= max_attempts)
    {
        status->error_code = MXX_PREIMAGE_EXHAUSTED;
        cudaGraphSetConditional(handle, 0U);
        return;
    }
    auto *control = static_cast<MxxPreimageLaunchControl *>(device_control);
    if (control) control->attempt = status->attempts;
    cudaGraphSetConditional(handle, 1U);
}


#endif

__global__ void mxx_gpu_gather_u64_kernel(
    const uint64_t *source,
    const uint64_t *indices,
    uint64_t *destination,
    size_t count,
    size_t source_count)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
    {
        return;
    }
    // Index families are stored as signed i64 words.  Valid DSL indices are
    // non-negative and within the source family; guard the device access so a
    // malformed runtime index cannot turn into an out-of-bounds load.  The
    // gather remains fully asynchronous: invalid entries produce the neutral
    // zero value and no host readback is introduced here.
    const int64_t selected = static_cast<int64_t>(indices[index]);
    if (selected < 0 || static_cast<uint64_t>(selected) >= source_count)
    {
        destination[index] = 0;
        return;
    }
    destination[index] = source[static_cast<size_t>(selected)];
}

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
        cudaError_t error = cudaSetDevice(job.device);
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
            cudaSetDevice(entry.device);
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
            cudaError_t err = cudaSetDevice(device);
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
            cudaSetDevice(owner->gpu_ids[partition]);
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
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
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
        const uint32_t coeff_bytes = (bit_width + 7U) / 8U;
        if (coeff_bytes == 0 || coeff_bytes > 8U)
        {
            throw std::runtime_error("invalid modulus byte-width in limb metadata");
        }
        return static_cast<uint8_t>(coeff_bytes);
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
            cudaError_t err = cudaDeviceGetDefaultMemPool(&pool, device);
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

    size_t minimum_device_memory_budget_bytes(
        const std::vector<int> &gpu_list,
        uint32_t percent)
    {
        int original_device = 0;
        cudaError_t err = cudaGetDevice(&original_device);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        size_t minimum_budget = std::numeric_limits<size_t>::max();
        for (int device : gpu_list)
        {
            err = cudaSetDevice(device);
            if (err == cudaSuccess)
            {
                size_t free_bytes = 0;
                size_t total_bytes = 0;
                err = cudaMemGetInfo(&free_bytes, &total_bytes);
                if (err == cudaSuccess)
                {
                    // Split the calculation before multiplication to preserve
                    // floor(total_bytes * percent / 100) without overflowing.
                    const size_t budget =
                        (total_bytes / 100) * static_cast<size_t>(percent) +
                        ((total_bytes % 100) * static_cast<size_t>(percent)) / 100;
                    minimum_budget = std::min(minimum_budget, budget);
                    continue;
                }
            }

            const std::string error = cudaGetErrorString(err);
            cudaSetDevice(original_device);
            throw std::runtime_error(error);
        }

        err = cudaSetDevice(original_device);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        return minimum_budget;
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
        if (cudaSetDevice(entry.device) != cudaSuccess)
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
        cudaError_t err = cudaSetDevice(device);
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

        cudaError_t err = cudaSetDevice(device);
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
        uint32_t vram_percent,
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
            if (vram_percent == 0 || vram_percent > 100)
            {
                return set_error("GPU VRAM percentage must be between 1 and 100");
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
                (!related_context->execution || related_context->gpu_ids != gpu_list ||
                 related_context->execution->vram_percent != vram_percent))
            {
                return set_error("related GPU rings must share device placement and VRAM policy");
            }
            const size_t vram_budget_bytes =
                minimum_device_memory_budget_bytes(gpu_list, vram_percent);
            configure_default_mempool_release_threshold(gpu_list);
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
                gpu_ctx->execution->vram_budget_bytes = vram_budget_bytes;
                gpu_ctx->execution->vram_percent = vram_percent;
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
            gpu_ctx->vram_budget_bytes = gpu_ctx->execution->vram_budget_bytes;
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
                cudaError_t err = cudaSetDevice(device);
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

        cudaError_t error = cudaSetDevice(device);
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

    int gpu_context_get_vram_budget_bytes(const GpuContext *ctx, size_t *out_bytes)
    {
        if (!ctx || !out_bytes)
        {
            return set_error("invalid gpu_context_get_vram_budget_bytes arguments");
        }
        *out_bytes = ctx->vram_budget_bytes;
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
        cudaError_t err = cudaDeviceGetDefaultMemPool(&pool, device);
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

    int gpu_default_mempool_reset_used_high(int device)
    {
        if (device < 0 || static_cast<size_t>(device) >= MAX_TRACKED_GPU_DEVICES)
        {
            return set_error("invalid gpu_default_mempool_reset_used_high device");
        }
        if (live_context_counts[static_cast<size_t>(device)].load(std::memory_order_acquire) != 1)
        {
            return set_error(
                "default mempool high-water reset requires exactly one live mxx context");
        }
        cudaMemPool_t pool = nullptr;
        cudaError_t err = cudaDeviceGetDefaultMemPool(&pool, device);
        if (err != cudaSuccess)
        {
        return set_error(err);
        }
        uint64_t reset = 0;
        err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &reset);
        if (err != cudaSuccess)
        {
        return set_error(err);
        }
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
        cudaError_t err = cudaGetDeviceProperties(&properties, device);
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
            cudaError_t err = cudaSetDevice(entry.device);
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
        *out_count = count;
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
        err = cudaSetDevice(device);
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

    int gpu_device_reset()
    {
        cudaError_t err = cudaDeviceReset();
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
        cudaError_t status = cudaGetDevice(&buffer->device);
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
        cudaError_t status = cudaSetDevice(buffer->device);
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
        cudaError_t status = cudaSetDevice(buffer->device);
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

    int gpu_device_buffer_wait_compiled_inputs(
        const MxxGpuDeviceBuffer *buffer,
        int consumer_device,
        void *consumer_stream_raw,
        bool read_only)
    {
        (void)read_only;
        if (!buffer || !consumer_stream_raw || consumer_device < 0)
            return set_error("invalid gpu_device_buffer_wait_compiled_inputs arguments");
        cudaError_t status = cudaSetDevice(consumer_device);
        if (status == cudaSuccess && buffer->producer_valid &&
            reinterpret_cast<cudaStream_t>(consumer_stream_raw) != buffer->allocation_stream)
        {
            status = cudaStreamWaitEvent(
                reinterpret_cast<cudaStream_t>(consumer_stream_raw), buffer->producer, 0);
        }
        return status == cudaSuccess ? 0 : set_error(status);
    }

    int gpu_device_buffer_track_compiled_consumer(
        const MxxGpuDeviceBuffer *buffer,
        int consumer_device,
        void *consumer_stream_raw,
        void *completion_event_raw,
        bool read_only)
    {
        (void)consumer_stream_raw;
        (void)read_only;
        if (!buffer || !completion_event_raw || consumer_device < 0)
            return set_error("invalid gpu_device_buffer_track_compiled_consumer arguments");
        cudaError_t status = cudaSetDevice(buffer->device);
        if (status == cudaSuccess)
            status = cudaStreamWaitEvent(
                buffer->allocation_stream,
                reinterpret_cast<cudaEvent_t>(completion_event_raw),
                0);
        return status == cudaSuccess ? 0 : set_error(status);
    }

    int gpu_device_buffer_record_compiled_write(
        MxxGpuDeviceBuffer *buffer,
        void *stream_raw)
    {
        if (!buffer || !stream_raw)
            return set_error("invalid gpu_device_buffer_record_compiled_write arguments");
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        cudaError_t capture_error = cudaStreamIsCapturing(reinterpret_cast<cudaStream_t>(stream_raw), &capture);
        if (capture_error != cudaSuccess) return set_error(capture_error);
        if (capture != cudaStreamCaptureStatusNone) return 0;
        cudaError_t status = cudaEventRecord(
            buffer->producer,
            reinterpret_cast<cudaStream_t>(stream_raw));
        if (status != cudaSuccess)
            return set_error(status);
        buffer->producer_valid = true;
        return 0;
    }

    int gpu_device_buffer_wait(const MxxGpuDeviceBuffer *buffer)
    {
        if (!buffer || !buffer->producer_valid)
            return buffer ? 0 : set_error("invalid gpu_device_buffer_wait arguments");
        cudaError_t status = cudaSetDevice(buffer->device);
        if (status == cudaSuccess)
            status = cudaEventSynchronize(buffer->producer);
        return status == cudaSuccess ? 0 : set_error(status);
    }

    int gpu_device_buffer_prepare_external_for_capture(MxxGpuDeviceBuffer *buffer)
    {
        if (!buffer) return set_error("invalid capture buffer");
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        cudaError_t error = cudaStreamIsCapturing(buffer->allocation_stream, &capture);
        if (error != cudaSuccess) return set_error(error);
        if (capture != cudaStreamCaptureStatusNone) return set_error("buffer preparation must precede capture");
        const int status = gpu_device_buffer_wait(buffer);
        if (status != 0) return status;
        buffer->producer_valid = false;
        return 0;
    }

    int gpu_device_buffer_gather_u64(
        const MxxGpuDeviceBuffer *source,
        size_t source_offset,
        size_t source_count,
        const MxxGpuDeviceBuffer *indices,
        size_t indices_offset,
        size_t index_count,
        MxxGpuDeviceBuffer *destination,
        size_t destination_offset,
        void *stream_raw)
    {
        if (!source || !indices || !destination || !stream_raw ||
            source_offset > source->bytes ||
            source_count > (source->bytes - source_offset) / sizeof(uint64_t) ||
            indices_offset > indices->bytes ||
            index_count > (indices->bytes - indices_offset) / sizeof(uint64_t) ||
            destination_offset > destination->bytes ||
            index_count > (destination->bytes - destination_offset) / sizeof(uint64_t))
        {
            return set_error("invalid gpu_device_buffer_gather_u64 arguments");
        }
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
        cudaError_t status = cudaSetDevice(destination->device);
        if (status == cudaSuccess && source->producer_valid)
            status = cudaStreamWaitEvent(stream, source->producer, 0);
        if (status == cudaSuccess && indices->producer_valid)
            status = cudaStreamWaitEvent(stream, indices->producer, 0);
        if (status == cudaSuccess && index_count != 0)
        {
            const int threads = 256;
            const int blocks = static_cast<int>((index_count + threads - 1) / threads);
            mxx_gpu_gather_u64_kernel<<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const uint64_t *>(source->address) +
                    source_offset / sizeof(uint64_t),
                reinterpret_cast<const uint64_t *>(indices->address) +
                    indices_offset / sizeof(uint64_t),
                reinterpret_cast<uint64_t *>(destination->address) +
                    destination_offset / sizeof(uint64_t),
                index_count,
                source_count);
            status = cudaGetLastError();
        }
        if (status == cudaSuccess)
            status = cudaEventRecord(destination->producer, stream);
        if (status != cudaSuccess)
            return set_error(status);
        destination->producer_valid = true;
        return 0;
    }

    int gpu_device_buffer_copy_range(
        const MxxGpuDeviceBuffer *source,
        size_t source_offset,
        MxxGpuDeviceBuffer *destination,
        size_t destination_offset,
        size_t bytes,
        void *stream_raw)
    {
        if (!source || !destination || !stream_raw ||
            source_offset > source->bytes || bytes > source->bytes - source_offset ||
            destination_offset > destination->bytes ||
            bytes > destination->bytes - destination_offset)
        {
            return set_error("invalid gpu_device_buffer_copy_range arguments");
        }
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_raw);
        cudaError_t status = cudaSetDevice(destination->device);
        if (status == cudaSuccess && destination->producer_valid &&
            stream != destination->allocation_stream)
        {
            status = cudaStreamWaitEvent(stream, destination->producer, 0);
        }
        if (status == cudaSuccess && source->producer_valid)
        {
            // CUDA permits a stream on the destination device to wait on an
            // event recorded by a peer device.  This preserves the producer
            // dependency without a host synchronization or staging buffer.
            status = cudaStreamWaitEvent(stream, source->producer, 0);
        }
        if (status == cudaSuccess && source->device != destination->device)
        {
            int can_access = 0;
            status = cudaDeviceCanAccessPeer(&can_access, destination->device, source->device);
            if (status == cudaSuccess && !can_access)
                return set_error("peer access is unavailable for device values copy");
            if (status == cudaSuccess)
            {
                status = cudaDeviceEnablePeerAccess(source->device, 0);
                if (status == cudaErrorPeerAccessAlreadyEnabled)
                {
                    cudaGetLastError();
                    status = cudaSuccess;
                }
            }
        }
        if (status == cudaSuccess)
        {
            if (source->device == destination->device)
            {
                status = cudaMemcpyAsync(
                    destination->address + destination_offset,
                    source->address + source_offset,
                    bytes,
                    cudaMemcpyDeviceToDevice,
                    stream);
            }
            else
            {
                status = cudaMemcpyPeerAsync(
                    destination->address + destination_offset,
                    destination->device,
                    source->address + source_offset,
                    source->device,
                    bytes,
                    stream);
            }
        }
        if (status == cudaSuccess)
            status = cudaEventRecord(destination->producer, stream);
        if (status != cudaSuccess)
            return set_error(status);
        destination->producer_valid = true;
        return 0;
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

    struct GraphRetainedResource
    {
        void *resource;
        void (*destroy)(void *);
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

    struct MxxGpuGraphCapture
    {
        GpuContext *context = nullptr;
        int device = -1;
        cudaStream_t stream = nullptr;
        bool active = false;
        uint32_t current_binding_offset = 0;
        uint32_t next_binding_index = 0;
        std::vector<MxxGraphBindingMapEntry> current_binding_map;
        struct ResidentAddress { uint64_t address; size_t bytes; uint32_t binding; };
        std::vector<ResidentAddress> resident_addresses;
        std::vector<void *> initialized_control_status;
        std::vector<GraphKernelUpdateRecord> kernels;
        std::vector<GraphMemcpyUpdateRecord> memcpys;
        std::vector<GraphMemsetUpdateRecord> memsets;
        struct PreimageRetryRegistration
        {
            MxxPreimageRetrySpec spec{};
            void *fixed_scratch = nullptr;
            void *device_control = nullptr;
            void *device_status = nullptr;
            cudaGraph_t body_graph = nullptr;
            cudaGraphNode_t boundary = nullptr;
            // Launch-site registrations made on the conditional-body stream
            // belong to the body graph, not to the parent graph.  Retain the
            // exact records until the body is installed in the executable;
            // silently dropping these patches would leave replay pointing at
            // the capture-time scratch and status owners.
            std::vector<GraphKernelUpdateRecord> body_kernels;
            std::vector<GraphMemcpyUpdateRecord> body_memcpys;
            std::vector<GraphMemsetUpdateRecord> body_memsets;
            std::vector<GraphKernelUpdateRecord> gate_kernels;
        };
        std::vector<PreimageRetryRegistration> preimage_retries;
        // The body-capture API returns an opaque graph handle, so its
        // launch-site records are staged here between body-finish and the
        // matching add_preimage_retry_body call.
        std::vector<GraphKernelUpdateRecord> finished_body_kernels;
        std::vector<GraphMemcpyUpdateRecord> finished_body_memcpys;
        std::vector<GraphMemsetUpdateRecord> finished_body_memsets;
        std::vector<MxxGpuCaptureEvent *> captured_events;
        std::vector<GraphRetainedResource> retained_resources;
        ~MxxGpuGraphCapture()
        {
            for (const auto &resource : retained_resources) resource.destroy(resource.resource);
            for (auto &retry : preimage_retries)
                if (retry.body_graph) cudaGraphDestroy(retry.body_graph);
            for (MxxGpuCaptureEvent *event : captured_events)
                mxx_gpu_capture_event_release(event);
        }
    };

    struct MxxGpuGraphBodyCapture
    {
        MxxGpuGraphCapture *parent = nullptr;
        int device = -1;
        cudaStream_t stream = nullptr;
        bool active = false;
        std::vector<GraphKernelUpdateRecord> kernels;
        std::vector<GraphMemcpyUpdateRecord> memcpys;
        std::vector<GraphMemsetUpdateRecord> memsets;
    };

    int mxx_graph_retain_released_resource(GpuContext *ctx, void *resource, void (*destroy)(void *))
    {
        if (!ctx || !ctx->execution || !resource || !destroy) return 0;
        auto &owner = *ctx->execution;
        std::lock_guard<std::mutex> lock(owner.capture_mutex);
        if (!owner.capture_active || !owner.capture_handle) return 0;
        auto *capture = static_cast<MxxGpuGraphCapture *>(owner.capture_handle);
        capture->retained_resources.push_back({resource, destroy});
        return 1;
    }

    void clear_capture_state(MxxGpuGraphCapture *capture)
    {
        if (!capture || !capture->context || !capture->context->execution) return;
        auto &owner = *capture->context->execution;
        std::lock_guard<std::mutex> lock(owner.capture_mutex);
        if (owner.capture_stream == capture->stream)
        {
            owner.capture_stream = nullptr;
            owner.capture_device = -1;
            owner.capture_active = false;
            owner.capture_handle = nullptr;
        }
    }

    struct MxxGpuGraphExec
    {
        int device = -1;
        cudaStream_t default_stream = nullptr;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        std::vector<GraphKernelUpdateRecord> kernels;
        std::vector<GraphMemcpyUpdateRecord> memcpys;
        std::vector<GraphMemsetUpdateRecord> memsets;
        // Source child-graph records are retained separately from top-level
        // records. They are patched before the conditional child graph is
        // rebound; dropping them would silently retain capture-time pointers.
        std::vector<GraphKernelUpdateRecord> body_kernels;
        std::vector<GraphMemcpyUpdateRecord> body_memcpys;
        std::vector<GraphMemsetUpdateRecord> body_memsets;
        // Child graphs are cloned into conditional bodies. Retain the source
        // graphs until the executable is destroyed so CUDA never observes a
        // dangling child graph during graph update or replay.
        std::vector<cudaGraph_t> retained_body_graphs;
        std::vector<MxxGpuCaptureEvent *> captured_events;
        std::vector<GraphRetainedResource> retained_resources;
    };

    struct MxxGpuNativeEvent
    {
        int device = -1;
        cudaEvent_t event = nullptr;
    };

    void destroy_captured_graph(cudaGraph_t graph)
    {
        if (graph)
        {
            const cudaError_t error = cudaGraphDestroy(graph);
            if (error != cudaSuccess)
            {
                set_error(error);
            }
        }
    }

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

    int graph_capture_node_is_unique(const MxxGpuGraphCapture *capture, cudaGraphNode_t node)
    {
        for (const auto &record : capture->kernels)
        {
            if (record.node == node) return set_error("CUDA graph node registered twice");
        }
        for (const auto &record : capture->memcpys)
        {
            if (record.node == node) return set_error("CUDA graph node registered twice");
        }
        for (const auto &record : capture->memsets)
        {
            if (record.node == node) return set_error("CUDA graph node registered twice");
        }
        return 0;
    }

    int mxx_gpu_graph_capture_begin(
        GpuContext *ctx,
        int physical_device,
        void *raw_stream,
        MxxGpuGraphCapture **out_capture)
    {
        if (!ctx || !ctx->execution || !out_capture)
        {
            return set_error("invalid mxx_gpu_graph_capture_begin arguments");
        }
        *out_capture = nullptr;
        if (!raw_stream && gpu_context_get_compute_stream(ctx, physical_device, &raw_stream) != 0)
            return 1;
        auto *capture = new (std::nothrow) MxxGpuGraphCapture();
        if (!capture)
        {
            return set_error("failed to allocate CUDA graph capture state");
        }
        capture->context = ctx;
        capture->device = physical_device;
        capture->stream = reinterpret_cast<cudaStream_t>(raw_stream);
        cudaError_t error = cudaSetDevice(physical_device);
        if (error == cudaSuccess)
        {
            error = cudaStreamBeginCapture(capture->stream, cudaStreamCaptureModeThreadLocal);
        }
        if (error != cudaSuccess)
        {
            delete capture;
            return set_error(error);
        }
        {
            auto &owner = *ctx->execution;
            std::lock_guard<std::mutex> lock(owner.capture_mutex);
            if (owner.capture_active)
            {
                cudaGraph_t discarded = nullptr;
                cudaStreamEndCapture(capture->stream, &discarded);
                if (discarded) cudaGraphDestroy(discarded);
                delete capture;
                return set_error("another CUDA graph capture is already active for this execution owner");
            }
            owner.capture_stream = capture->stream;
            owner.capture_device = physical_device;
            owner.capture_active = true;
            ++owner.capture_generation;
            owner.capture_handle = capture;
        }
        capture->active = true;
        *out_capture = capture;
        return 0;
    }

    int mxx_gpu_graph_capture_stream(MxxGpuGraphCapture *capture, void **out_stream)
    {
        if (!capture || !capture->active || !out_stream || !capture->stream)
        {
            return set_error("invalid mxx_gpu_graph_capture_stream arguments");
        }
        *out_stream = reinterpret_cast<void *>(capture->stream);
        return 0;
    }

    int mxx_gpu_graph_capture_claim_binding_range(
        MxxGpuGraphCapture *capture,
        uint32_t count,
        uint32_t *out_offset)
    {
        if (!capture || !capture->active || !out_offset || count == 0)
            return set_error("invalid CUDA graph binding-range claim");
        if (count > UINT32_MAX - capture->next_binding_index)
            return set_error("CUDA graph binding index overflow");
        *out_offset = capture->next_binding_index;
        capture->next_binding_index += count;
        capture->current_binding_offset = *out_offset;
        capture->current_binding_map.clear();
        return 0;
    }

    int mxx_gpu_graph_capture_set_binding_offset(
        MxxGpuGraphCapture *capture,
        uint32_t offset)
    {
        if (!capture || !capture->active)
            return set_error("invalid CUDA graph binding-offset update");
        capture->current_binding_offset = offset;
        capture->current_binding_map.clear();
        return 0;
    }

    int mxx_gpu_graph_capture_set_binding_map(
        MxxGpuGraphCapture *capture,
        const MxxGraphBindingMapEntry *entries,
        size_t entry_count)
    {
        if (!capture || !capture->active || (entry_count != 0 && !entries))
            return set_error("invalid CUDA graph binding map");
        try
        {
            std::vector<MxxGraphBindingMapEntry> mapping;
            mapping.reserve(entry_count);
            for (size_t index = 0; index < entry_count; ++index)
            {
                const MxxGraphBindingMapEntry entry = entries[index];
                bool duplicate = false;
                for (const auto &existing : mapping)
                {
                    if (existing.local_binding != entry.local_binding) continue;
                    if (existing.global_binding != entry.global_binding)
                        return set_error("conflicting local CUDA graph binding identity");
                    // Re-registering one exact identity is harmless.
                    duplicate = true;
                    break;
                }
                if (!duplicate) mapping.push_back(entry);
            }
            capture->current_binding_map = std::move(mapping);
        }
        catch (const std::exception &exception)
        {
            return set_error(exception);
        }
        return 0;
    }

    int mxx_gpu_graph_capture_abort(MxxGpuGraphCapture *capture)
    {
        if (!capture)
        {
            return set_error("invalid mxx_gpu_graph_capture_abort arguments");
        }
        int status = 0;
        if (capture->active)
        {
            cudaError_t error = cudaSetDevice(capture->device);
            cudaGraph_t discarded = nullptr;
            if (error == cudaSuccess)
            {
                error = cudaStreamEndCapture(capture->stream, &discarded);
            }
            if (discarded)
            {
                const cudaError_t destroy_error = cudaGraphDestroy(discarded);
                if (error == cudaSuccess) error = destroy_error;
            }
            if (error != cudaSuccess)
            {
                status = set_error(error);
            }
            capture->active = false;
            clear_capture_state(capture);
        }
        delete capture;
        return status;
    }

    int mxx_gpu_graph_body_capture_begin(
        MxxGpuGraphCapture *capture,
        MxxGpuGraphBodyCapture **out_body)
    {
        if (!capture || !capture->active || !out_body)
            return set_error("invalid CUDA graph body capture arguments");
        *out_body = nullptr;
        cudaStream_t body_stream = nullptr;
        auto &owner = *capture->context->execution;
        {
            std::lock_guard<std::mutex> lock(owner.capture_mutex);
            const auto found = std::find(
                owner.gpu_ids.begin(), owner.gpu_ids.end(), capture->device);
            if (found == owner.gpu_ids.end())
                return set_error("body capture device is not owned by execution owner");
            const size_t partition = static_cast<size_t>(found - owner.gpu_ids.begin());
            if (partition >= owner.compute_streams_by_partition.size())
                return set_error("body capture has no compute stream partition");
            for (cudaStream_t candidate : owner.compute_streams_by_partition[partition])
            {
                if (candidate != capture->stream)
                {
                    body_stream = candidate;
                    break;
                }
            }
        }
        if (!body_stream)
            return set_error("conditional body capture requires a second compute stream");
        auto *body = new (std::nothrow) MxxGpuGraphBodyCapture();
        if (!body) return set_error("failed to allocate CUDA graph body capture state");
        body->parent = capture;
        body->device = capture->device;
        body->stream = body_stream;

        // Claim the body slot before beginning CUDA capture.  Starting the
        // capture first leaves a race in which two host callers can both
        // enter the same auxiliary stream and only one is rejected after
        // CUDA state has already been mutated.
        {
            std::lock_guard<std::mutex> lock(owner.capture_mutex);
            if (owner.capture_body_active || owner.capture_body_handle != nullptr)
            {
                delete body;
                return set_error("another CUDA graph body capture is already active");
            }
            owner.capture_body_stream = body_stream;
            owner.capture_body_device = body->device;
            owner.capture_body_active = true;
            owner.capture_body_handle = body;
        }
        cudaError_t error = cudaSetDevice(body->device);
        if (error == cudaSuccess)
            error = cudaStreamBeginCapture(body->stream, cudaStreamCaptureModeThreadLocal);
        if (error != cudaSuccess)
        {
            std::lock_guard<std::mutex> lock(owner.capture_mutex);
            if (owner.capture_body_handle == body)
            {
                owner.capture_body_stream = nullptr;
                owner.capture_body_device = -1;
                owner.capture_body_active = false;
                owner.capture_body_handle = nullptr;
            }
            delete body;
            return set_error(error);
        }
        body->active = true;
        *out_body = body;
        return 0;
    }

    int mxx_gpu_graph_body_capture_stream(
        MxxGpuGraphBodyCapture *body,
        void **out_stream)
    {
        if (!body || !body->active || !out_stream || !body->stream)
            return set_error("invalid CUDA graph body capture stream arguments");
        *out_stream = reinterpret_cast<void *>(body->stream);
        return 0;
    }

    int mxx_gpu_graph_body_capture_finish(
        MxxGpuGraphBodyCapture *body,
        void **out_graph)
    {
        if (!body || !out_graph)
            return set_error("invalid CUDA graph body capture finish arguments");
        *out_graph = nullptr;
        MxxGpuGraphCapture *parent = body->parent;
        const cudaStream_t body_stream = body->stream;
        cudaGraph_t graph = nullptr;
        cudaError_t error = cudaSetDevice(body->device);
        if (error == cudaSuccess && body->active)
        {
            error = cudaStreamEndCapture(body->stream, &graph);
            body->active = false;
        }
        if (parent && parent->context && parent->context->execution)
        {
            auto &owner = *parent->context->execution;
            std::lock_guard<std::mutex> lock(owner.capture_mutex);
            if (owner.capture_body_stream == body_stream)
            {
                owner.capture_body_stream = nullptr;
                owner.capture_body_device = -1;
                owner.capture_body_active = false;
                owner.capture_body_handle = nullptr;
            }
        }
        if (parent)
        {
            parent->finished_body_kernels = std::move(body->kernels);
            parent->finished_body_memcpys = std::move(body->memcpys);
            parent->finished_body_memsets = std::move(body->memsets);
        }
        delete body;
        if (error != cudaSuccess || !graph)
        {
            if (parent)
            {
                parent->finished_body_kernels.clear();
                parent->finished_body_memcpys.clear();
                parent->finished_body_memsets.clear();
            }
            if (graph) cudaGraphDestroy(graph);
            return error == cudaSuccess ? set_error("body capture produced no graph")
                                        : set_error(error);
        }
        *out_graph = reinterpret_cast<void *>(graph);
        return 0;
    }

    int mxx_gpu_graph_body_capture_abort(MxxGpuGraphBodyCapture *body)
    {
        if (!body) return set_error("invalid CUDA graph body capture abort arguments");
        int status = 0;
        if (body->active)
        {
            cudaError_t error = cudaSetDevice(body->device);
            cudaGraph_t discarded = nullptr;
            if (error == cudaSuccess)
                error = cudaStreamEndCapture(body->stream, &discarded);
            if (discarded) cudaGraphDestroy(discarded);
            if (error != cudaSuccess) status = set_error(error);
            body->active = false;
        }
        if (body->parent && body->parent->context && body->parent->context->execution)
        {
            auto &owner = *body->parent->context->execution;
            std::lock_guard<std::mutex> lock(owner.capture_mutex);
            if (owner.capture_body_stream == body->stream)
            {
                owner.capture_body_stream = nullptr;
                owner.capture_body_device = -1;
                owner.capture_body_active = false;
                owner.capture_body_handle = nullptr;
            }
        }
        delete body;
        return status;
    }

    void mxx_gpu_graph_body_destroy(void *graph)
    {
        if (graph) cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(graph));
    }

    int resolve_resident_patch(const MxxGpuGraphCapture *capture, uint64_t address, MxxGraphPatch &patch);

    int mxx_gpu_graph_capture_finish(
        MxxGpuGraphCapture *capture,
        MxxGpuGraphExec **out_exec)
    {
        if (!capture || !out_exec)
        {
            return set_error("invalid mxx_gpu_graph_capture_finish arguments");
        }
        *out_exec = nullptr;
        cudaGraph_t graph = nullptr;
        cudaError_t error = cudaSetDevice(capture->device);
        if (error == cudaSuccess && capture->active)
        {
            error = cudaStreamEndCapture(capture->stream, &graph);
            capture->active = false;
        }
        if (error != cudaSuccess || !graph)
        {
            if (graph) destroy_captured_graph(graph);
            clear_capture_state(capture);
            delete capture;
            return error == cudaSuccess ? set_error("CUDA graph capture produced no graph")
                                        : set_error(error);
        }

        size_t captured_node_count = 0;
        error = cudaGraphGetNodes(graph, nullptr, &captured_node_count);
        if (error != cudaSuccess || captured_node_count == 0)
        {
            destroy_captured_graph(graph);
            clear_capture_state(capture);
            delete capture;
            return error == cudaSuccess ? set_error("empty CUDA graph is not executable work")
                                        : set_error(error);
        }

        auto *result = new (std::nothrow) MxxGpuGraphExec();
        if (!result)
        {
            destroy_captured_graph(graph);
            clear_capture_state(capture);
            delete capture;
            return set_error("failed to allocate CUDA graph executable state");
        }
        result->device = capture->device;
        result->default_stream = capture->stream;
        result->graph = graph;

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12030
        // Insert each retry before its capture-time boundary. Downstream
        // consumers/status copies depend on that boundary, not the last node
        // of the whole graph.
        for (auto &registration : capture->preimage_retries)
        {
            cudaGraphConditionalHandle handle = 0;
            error = cudaGraphConditionalHandleCreate(
                &handle, graph, 1U, cudaGraphCondAssignDefault);
            if (error != cudaSuccess) break;

            auto *status = static_cast<MxxPreimageStatus *>(registration.device_status);
            uint32_t max_attempts = registration.spec.max_attempts;
            void *fixed_scratch = registration.fixed_scratch;
            void *device_control = registration.device_control;
            void *kernel_arguments[] = {
                &status,
                &max_attempts,
                &handle,
                &fixed_scratch,
                &device_control,
            };
            cudaKernelNodeParams gate_params{};
            gate_params.func = reinterpret_cast<void *>(mxx_preimage_retry_gate_kernel);
            gate_params.gridDim = dim3(1, 1, 1);
            gate_params.blockDim = dim3(1, 1, 1);
            gate_params.sharedMemBytes = 0;
            gate_params.kernelParams = kernel_arguments;
            gate_params.extra = nullptr;

            size_t dependency_count = 0;
            error = cudaGraphNodeGetDependencies(registration.boundary, nullptr,
#if CUDART_VERSION >= 13000
                nullptr,
#endif
                &dependency_count);
            std::vector<cudaGraphNode_t> dependencies;
            if (error == cudaSuccess && dependency_count != 0)
            {
                dependencies.resize(dependency_count);
                error = cudaGraphNodeGetDependencies(
                    registration.boundary, dependencies.data(),
#if CUDART_VERSION >= 13000
                    nullptr,
#endif
                    &dependency_count);
            }
            if (error != cudaSuccess) break;

            // CUDA supplies its own array of conditional body graph handles
            // in phGraph_out; it does not fill a caller-owned graph pointer.
            cudaGraph_t body = nullptr;
            cudaGraphNodeParams conditional_params{};
            conditional_params.type = cudaGraphNodeTypeConditional;
            conditional_params.conditional.handle = handle;
            conditional_params.conditional.type = cudaGraphCondTypeWhile;
            conditional_params.conditional.size = 1;
            conditional_params.conditional.phGraph_out = nullptr;
            cudaGraphNode_t conditional_node = nullptr;
            error = cudaGraphAddNode(
                &conditional_node,
                graph,
                dependencies.empty() ? nullptr : dependencies.data(),
                nullptr,
                dependencies.size(),
                &conditional_params);
            if (error != cudaSuccess) break;
            if (conditional_params.conditional.phGraph_out)
                body = conditional_params.conditional.phGraph_out[0];

            error = cudaGraphAddDependencies(graph, &conditional_node, &registration.boundary,
#if CUDART_VERSION >= 13000
                nullptr,
#endif
                1);
            if (error != cudaSuccess) break;

            if (!body)
            {
                error = cudaErrorInvalidValue;
                break;
            }
            cudaGraphNode_t body_child = nullptr;
            error = cudaGraphAddChildGraphNode(
                &body_child, body, nullptr, 0, registration.body_graph);
            if (error != cudaSuccess) break;
            // Child insertion clones the graph. Replay must patch nodes in
            // that owned clone, not the capture source graph.
            cudaGraph_t replay_body = nullptr;
            error = cudaGraphChildGraphNodeGetGraph(body_child, &replay_body);
            if (error != cudaSuccess) break;
            const auto remap_body_nodes = [&](auto &records) {
                for (auto &record : records)
                {
                    cudaGraphNode_t cloned = nullptr;
                    const cudaError_t remap_error = cudaGraphNodeFindInClone(
                        &cloned, record.node, replay_body);
                    if (remap_error != cudaSuccess) return remap_error;
                    record.node = cloned;
                }
                return cudaSuccess;
            };
            error = remap_body_nodes(registration.body_kernels);
            if (error == cudaSuccess) error = remap_body_nodes(registration.body_memcpys);
            if (error == cudaSuccess) error = remap_body_nodes(registration.body_memsets);
            if (error != cudaSuccess) break;
            cudaGraphNode_t body_node = nullptr;
            error = cudaGraphAddKernelNode(
                &body_node,
                body,
                &body_child,
                1,
                &gate_params);
            if (error != cudaSuccess) break;
            GraphKernelUpdateRecord gate_record;
            gate_record.node = body_node;
            gate_record.launch = gate_params;
            const size_t gate_argument_sizes[] = {
                sizeof(void *), sizeof(uint32_t), sizeof(cudaGraphConditionalHandle),
                sizeof(void *), sizeof(void *),
            };
            const MxxGraphPatch gate_patches[] = {
                {nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0, sizeof(void *),
                 registration.spec.status_binding_index, 0},
            };
            try
            {
                gate_record.arguments.resize(std::size(gate_argument_sizes));
                gate_record.argument_pointers.resize(std::size(gate_argument_sizes));
                for (size_t index = 0; index < std::size(gate_argument_sizes); ++index)
                {
                    gate_record.arguments[index].resize(gate_argument_sizes[index]);
                    memcpy(
                        gate_record.arguments[index].data(),
                        kernel_arguments[index],
                        gate_argument_sizes[index]);
                }
                gate_record.launch.kernelParams = gate_record.argument_pointers.data();
                gate_record.launch.extra = nullptr;
                for (MxxGraphPatch patch : gate_patches)
                {
                    if (!capture->resident_addresses.empty())
                    {
                        uint64_t address = 0;
                        memcpy(&address, kernel_arguments[patch.argument_index], sizeof(address));
                        const bool resident = std::any_of(
                            capture->resident_addresses.begin(), capture->resident_addresses.end(),
                            [address](const MxxGpuGraphCapture::ResidentAddress &owner) {
                                return address >= owner.address && address - owner.address < owner.bytes;
                            });
                        if (!resident) continue;
                        if (resolve_resident_patch(capture, address, patch) != 0)
                        { error = cudaErrorInvalidValue; break; }
                    }
                    gate_record.patches.push_back(GraphPatchRecord{patch});
                }
                registration.gate_kernels.push_back(std::move(gate_record));
            }
            catch (const std::exception &exception)
            {
                error = cudaErrorMemoryAllocation;
                set_error(exception);
                break;
            }
            // The conditional node owns the body graph and CUDA has populated
            // the body handle in phGraph_out. Do not destroy it separately.
        }
#elif defined(CUDART_VERSION)
        if (!capture->preimage_retries.empty())
        {
            error = cudaErrorNotSupported;
        }
#endif

        if (error != cudaSuccess)
        {
            result->exec = nullptr;
            destroy_captured_graph(graph);
            clear_capture_state(capture);
            delete result;
            delete capture;
            return set_error(error);
        }
        // Allocation nodes belong to the executable, not to the capture
        // exemplar. Replaying an allocation graph requires releasing its
        // preceding launch's internal allocations before allocating again.
        error = cudaGraphInstantiateWithFlags(&result->exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch);
        if (error != cudaSuccess)
        {
            result->exec = nullptr;
            destroy_captured_graph(graph);
            clear_capture_state(capture);
            delete result;
            delete capture;
            return set_error(error);
        }
        try
        {
            result->kernels = std::move(capture->kernels);
            result->memcpys = std::move(capture->memcpys);
            result->memsets = std::move(capture->memsets);
            for (auto &registration : capture->preimage_retries)
            {
                result->body_kernels.insert(
                    result->body_kernels.end(),
                    std::make_move_iterator(registration.body_kernels.begin()),
                    std::make_move_iterator(registration.body_kernels.end()));
                result->body_memcpys.insert(
                    result->body_memcpys.end(),
                    std::make_move_iterator(registration.body_memcpys.begin()),
                    std::make_move_iterator(registration.body_memcpys.end()));
                result->body_memsets.insert(
                    result->body_memsets.end(),
                    std::make_move_iterator(registration.body_memsets.begin()),
                    std::make_move_iterator(registration.body_memsets.end()));
                result->body_kernels.insert(
                    result->body_kernels.end(),
                    std::make_move_iterator(registration.gate_kernels.begin()),
                    std::make_move_iterator(registration.gate_kernels.end()));
            }
            for (auto &registration : capture->preimage_retries)
            {
                if (registration.body_graph)
                {
                    result->retained_body_graphs.push_back(registration.body_graph);
                    registration.body_graph = nullptr;
                }
            }
            result->captured_events = std::move(capture->captured_events);
            result->retained_resources = std::move(capture->retained_resources);
        }
        catch (const std::exception &exception)
        {
            cudaGraphExecDestroy(result->exec);
            destroy_captured_graph(result->graph);
            for (cudaGraph_t body : result->retained_body_graphs)
                if (body) cudaGraphDestroy(body);
            clear_capture_state(capture);
            delete result;
            delete capture;
            return set_error(exception);
        }
        clear_capture_state(capture);
        delete capture;
        *out_exec = result;
        return 0;
    }

    int mxx_graph_register_kernel_update(
        MxxGpuGraphCapture *capture,
        void *node_raw,
        const void *launch_raw,
        const size_t *argument_sizes,
        size_t argument_count,
        const MxxGraphPatch *patches,
        size_t patch_count)
    {
        if (!capture || !capture->active || !node_raw || !launch_raw ||
            (argument_count != 0 && (!argument_sizes || !static_cast<const cudaKernelNodeParams *>(launch_raw)->kernelParams)) ||
            (patch_count != 0 && !patches) ||
            static_cast<const cudaKernelNodeParams *>(launch_raw)->extra != nullptr)
        {
            return set_error("invalid CUDA graph kernel registration arguments");
        }
        auto *node = reinterpret_cast<cudaGraphNode_t>(node_raw);
        if (graph_capture_node_is_unique(capture, node) != 0)
        {
            return 1;
        }
        const auto *launch = static_cast<const cudaKernelNodeParams *>(launch_raw);
        GraphKernelUpdateRecord record;
        record.node = node;
        record.launch = *launch;
        try
        {
            record.arguments.resize(argument_count);
            record.argument_pointers.resize(argument_count);
            for (size_t index = 0; index < argument_count; ++index)
            {
                if (argument_sizes[index] == 0 || !launch->kernelParams[index])
                {
                    return set_error("kernel update contains an empty argument");
                }
                record.arguments[index].resize(argument_sizes[index]);
                memcpy(record.arguments[index].data(), launch->kernelParams[index], argument_sizes[index]);
            }
            record.launch.kernelParams = record.argument_pointers.data();
            record.launch.extra = nullptr;
            record.patches.reserve(patch_count);
            for (size_t index = 0; index < patch_count; ++index)
            {
                const MxxGraphPatch &patch = patches[index];
                if ((patch.target != MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD && patch.target != MXX_GRAPH_PATCH_INTEGER_ENCODING) ||
                    patch.argument_index >= argument_count ||
                    patch.byte_offset > argument_sizes[patch.argument_index] ||
                    patch.byte_count > argument_sizes[patch.argument_index] - patch.byte_offset ||
                    patch_duplicate(record.patches, patch))
                {
                    return set_error("invalid or duplicate kernel graph patch");
                }
                record.patches.push_back(GraphPatchRecord{patch});
            }
            capture->kernels.push_back(std::move(record));
        }
        catch (const std::exception &exception)
        {
            return set_error(exception);
        }
        return 0;
    }

    // Return the owner capture record for a launch site.  Matrix launch code
    // deliberately does not retain a graph handle; it only has its context
    // and the explicit stream on which it just enqueued work.  The owner lock
    // protects the short lookup while capture is active.  A null result is a
    // normal ordinary-execution path and must not change launch behavior.
    int graph_capture_for_stream(
        GpuContext *ctx,
        cudaStream_t stream,
        MxxGpuGraphCapture **out_capture,
        MxxGpuGraphBodyCapture **out_body)
    {
        if (!out_capture || !out_body)
        {
            return set_error("null graph capture output");
        }
        *out_capture = nullptr;
        *out_body = nullptr;
        if (!ctx || !ctx->execution || !stream)
        {
            return set_error("invalid graph capture stream lookup arguments");
        }
        auto &owner = *ctx->execution;
        std::lock_guard<std::mutex> lock(owner.capture_mutex);
        if (owner.capture_body_active && owner.capture_body_stream == stream &&
            owner.capture_body_handle != nullptr)
        {
            *out_capture = owner.capture_handle;
            *out_body = owner.capture_body_handle;
            return 0;
        }
        if (!owner.capture_active || owner.capture_stream != stream ||
            owner.capture_handle == nullptr)
        {
            return 0;
        }
        *out_capture = owner.capture_handle;
        return 0;
    }

    int graph_capture_latest_node(
        cudaStream_t stream,
        cudaGraphNodeType expected_type,
        cudaGraphNode_t *out_node)
    {
        if (!stream || !out_node)
        {
            return set_error("invalid graph capture node lookup arguments");
        }
        *out_node = nullptr;
        cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
        unsigned long long capture_id = 0;
        cudaGraph_t graph = nullptr;
        const cudaGraphNode_t *dependencies = nullptr;
        size_t dependency_count = 0;
#if CUDART_VERSION >= 13000
        cudaError_t error = cudaStreamGetCaptureInfo(
#else
        cudaError_t error = cudaStreamGetCaptureInfo_v2(
#endif
            stream,
            &status,
            &capture_id,
            &graph,
            &dependencies,
#if CUDART_VERSION >= 13000
            nullptr,
#endif
            &dependency_count);
        if (error != cudaSuccess)
        {
            return set_error(error);
        }
        if (status != cudaStreamCaptureStatusActive || !graph || dependency_count == 0)
        {
            return set_error("launch-site graph registration requires an active capture node");
        }
        // The registration call is made immediately after the corresponding
        // launch and before any event/lifetime operation.  CUDA exposes the
        // just-created top-level node as the sole current dependency on this
        // stream. Requiring that exact shape rejects ambiguous joins instead
        // of guessing from graph order or scanning unrelated nodes.
        if (dependency_count != 1 || !dependencies || !dependencies[0])
        {
            return set_error("launch-site graph registration found ambiguous capture dependencies");
        }
        cudaGraphNodeType actual_type = cudaGraphNodeTypeEmpty;
        error = cudaGraphNodeGetType(dependencies[0], &actual_type);
        if (error != cudaSuccess)
        {
            return set_error(error);
        }
        if (actual_type != expected_type)
        {
            return set_error("launch-site graph registration node type mismatch");
        }
        *out_node = dependencies[0];
        return 0;
    }

    int graph_capture_body_node_is_unique(
        const MxxGpuGraphBodyCapture *body,
        cudaGraphNode_t node)
    {
        if (!body || !node) return set_error("invalid body graph node registration");
        for (const auto &record : body->kernels)
            if (record.node == node) return set_error("duplicate body graph kernel registration");
        for (const auto &record : body->memcpys)
            if (record.node == node) return set_error("duplicate body graph memcpy registration");
        for (const auto &record : body->memsets)
            if (record.node == node) return set_error("duplicate body graph memset registration");
        return 0;
    }

    int mxx_gpu_graph_capture_bind_resident_address(
        MxxGpuGraphCapture *capture, uint64_t address, size_t bytes, uint32_t binding)
    {
        if (!capture || !capture->active || !address || !bytes)
            return set_error("invalid resident capture binding");
        // Binding IDs are schema identities.  Capture preparation can visit
        // the same owner from both the program-level schema and a native
        // payload, so replaying an exact registration is harmless and must
        // not create a second candidate in the address index.  A schema ID
        // may not, however, be rebound to a different range: that would make
        // a later patch depend on registration order.  Distinct IDs remain
        // independent even when they intentionally alias the same address.
        for (const auto &owner : capture->resident_addresses)
        {
            if (owner.binding != binding) continue;
            if (owner.address == address && owner.bytes == bytes) return 0;
            return set_error("conflicting resident capture binding identity");
        }
        try
        {
            capture->resident_addresses.push_back({address, bytes, binding});
        }
        catch (const std::exception &exception)
        {
            return set_error(exception);
        }
        return 0;
    }

    bool mxx_gpu_graph_control_status_needs_reset(GpuContext *ctx, void *stream, void *status)
    {
        MxxGpuGraphCapture *capture = nullptr;
        MxxGpuGraphBodyCapture *body = nullptr;
        if (graph_capture_for_stream(ctx, reinterpret_cast<cudaStream_t>(stream), &capture, &body) != 0 || !capture || capture->resident_addresses.empty())
            return true;
        for (void *initialized : capture->initialized_control_status)
            if (initialized == status) return false;
        capture->initialized_control_status.push_back(status);
        return true;
    }

    int resolve_resident_patch(const MxxGpuGraphCapture *capture, uint64_t address, MxxGraphPatch &patch)
    {
        // A mapped resident binding is a schema identity, not merely an
        // address range.  Two mapped schema slots are allowed to capture the
        // same address (for example, a base allocation and an overlapping
        // view), so choosing an arbitrary containing range silently redirects
        // one slot to the other on replay.  Prefer the exact binding recorded
        // in the patch whenever the capture supplied an explicit local-to-
        // global map.
        bool binding_was_registered = false;
        for (const auto &owner : capture->resident_addresses)
        {
            if (owner.binding != patch.binding_index) continue;
            binding_was_registered = true;
            if (address >= owner.address && address - owner.address < owner.bytes)
            {
                patch.address_addend = address - owner.address;
                return 0;
            }
            break;
        }

        // Unmapped primitive launches use operation-local binding IDs.  A
        // matrix copy expands one allocation into one patch per CRT limb, but
        // callers bind the allocation as a single owner range.  In that ABI,
        // the local limb ID is not the replay owner ID; the unique containing
        // owner is the only valid way to normalize it.  Explicit maps remain
        // strict so an owner/range mismatch cannot be hidden in a typed
        // resident capture.
        if (!capture->current_binding_map.empty() && binding_was_registered)
            return set_error("resident patch address is outside its binding range");

        // An unmapped capture may still rely on the address to identify its
        // schema slot.  A unique range is sufficient for that ABI.  If
        // aliases overlap, identity is required; selecting the smallest or
        // first address would merge distinct schema IDs and is therefore
        // invalid.
        const MxxGpuGraphCapture::ResidentAddress *selected = nullptr;
        for (const auto &owner : capture->resident_addresses)
        {
            if (address >= owner.address && address - owner.address < owner.bytes)
            {
                if (selected)
                {
                    return set_error("captured resident pointer has ambiguous overlapping bindings");
                }
                selected = &owner;
            }
        }
        if (selected)
        {
            patch.binding_index = selected->binding;
            patch.address_addend = address - selected->address;
            return 0;
        }
        if (binding_was_registered)
            return set_error("resident patch address is outside its binding range");
        return set_error("captured control pointer has no resident arena binding");
    }

    int mxx_graph_register_resident_descriptor_for_stream(
        GpuContext *ctx, void *stream_raw, const size_t *argument_sizes, size_t argument_count)
    {
        MxxGpuGraphCapture *capture = nullptr;
        MxxGpuGraphBodyCapture *body = nullptr;
        const auto stream = reinterpret_cast<cudaStream_t>(stream_raw);
        if (graph_capture_for_stream(ctx, stream, &capture, &body) != 0) return 1;
        if (!capture || capture->resident_addresses.empty()) return 0;
        cudaGraphNode_t node = nullptr;
        if (graph_capture_latest_node(stream, cudaGraphNodeTypeKernel, &node) != 0) return 1;
        cudaKernelNodeParams launch{};
        const cudaError_t error = cudaGraphKernelNodeGetParams(node, &launch);
        if (error != cudaSuccess) return set_error(error);
        uint64_t address = 0;
        memcpy(&address, launch.kernelParams[0], sizeof(address));
        const bool resident = std::any_of(capture->resident_addresses.begin(),
            capture->resident_addresses.end(), [address](const auto &owner) {
                return address >= owner.address && address - owner.address < owner.bytes;
            });
        if (!resident) return 0;
        MxxGraphPatch patch{nullptr, MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD, 0, 0,
            sizeof(void *), UINT32_MAX, 0};
        if (resolve_resident_patch(capture, address, patch) != 0) return 1;
        if (patch.binding_index < capture->current_binding_offset)
            return set_error("resident descriptor belongs to another binding scope");
        patch.binding_index -= capture->current_binding_offset;
        return mxx_graph_register_kernel_update_for_stream(ctx, stream_raw,
            argument_sizes, argument_count, &patch, 1);
    }

    int mxx_gpu_graph_capture_resolve_fixed_addresses(MxxGpuGraphCapture *capture)
    {
        if (!capture || !capture->active) return set_error("invalid fixed capture address resolution");
        int resolution_error = 0;
        const auto resolve = [capture, &resolution_error](uint64_t address, MxxGraphPatch &patch) {
            for (const auto &owner : capture->resident_addresses)
                if (address >= owner.address && address - owner.address < owner.bytes) {
                    const int result = resolve_resident_patch(capture, address, patch);
                    if (result != 0) resolution_error = result;
                    return true;
                }
            // Unbound pointers address graph-internal allocations, whose
            // destruction was transferred to this executable during capture.
            return false;
        };
        const auto resolve_records = [&](auto &kernels, auto &memcpys, auto &memsets) {
        for (auto &kernel : kernels) {
            kernel.patches.erase(std::remove_if(kernel.patches.begin(), kernel.patches.end(), [&](GraphPatchRecord &record) {
                auto &patch = record.patch;
                if (patch.target == MXX_GRAPH_PATCH_INTEGER_ENCODING) {
                    const bool bound = resolve(patch.address_addend, patch);
                    patch.address_addend = 0;
                    return !bound;
                }
                if (patch.byte_count != sizeof(uint64_t)) return false;
                uint64_t address = 0;
                memcpy(&address, kernel.arguments[patch.argument_index].data() + patch.byte_offset, sizeof(address));
                return !resolve(address, patch);
            }), kernel.patches.end());
        }
        for (auto &copy : memcpys) {
            copy.patches.erase(std::remove_if(copy.patches.begin(), copy.patches.end(), [&](GraphPatchRecord &record) {
                const void *address = record.patch.target == MXX_GRAPH_PATCH_MEMCPY_1D_SRC ? copy.source : copy.destination;
                return !resolve(reinterpret_cast<uint64_t>(address), record.patch);
            }), copy.patches.end());
        }
        memsets.erase(std::remove_if(memsets.begin(), memsets.end(), [&](GraphMemsetUpdateRecord &record) {
            return !resolve(reinterpret_cast<uint64_t>(record.params.dst), record.patch.patch);
        }), memsets.end());
        };
        resolve_records(capture->kernels, capture->memcpys, capture->memsets);
        for (auto &retry : capture->preimage_retries)
            resolve_records(retry.body_kernels, retry.body_memcpys, retry.body_memsets);
        return resolution_error;
    }

    int offset_graph_patches(
        const MxxGpuGraphCapture *capture,
        const MxxGraphPatch *patches,
        size_t patch_count,
        std::vector<MxxGraphPatch> *out)
    {
        if (!out || (patch_count != 0 && !patches))
            return set_error("invalid CUDA graph patch offset arguments");
        try
        {
            out->clear();
            if (patch_count != 0)
                out->assign(patches, patches + patch_count);
            for (MxxGraphPatch &patch : *out)
            {
                if (capture && !capture->current_binding_map.empty())
                {
                    const auto mapped = std::find_if(
                        capture->current_binding_map.begin(),
                        capture->current_binding_map.end(),
                        [&patch](const MxxGraphBindingMapEntry &entry) {
                            return entry.local_binding == patch.binding_index;
                        });
                    if (mapped == capture->current_binding_map.end())
                        return set_error("CUDA graph patch has no explicit local binding identity");
                    patch.binding_index = mapped->global_binding;
                }
                else if (capture)
                {
                    const uint32_t offset = capture->current_binding_offset;
                    if (patch.binding_index > UINT32_MAX - offset)
                        return set_error("CUDA graph binding index overflow");
                    patch.binding_index += offset;
                }
            }
        }
        catch (const std::exception &exception)
        {
            return set_error(exception);
        }
        return 0;
    }

    int register_body_kernel_update(
        MxxGpuGraphBodyCapture *body,
        cudaGraphNode_t node,
        const cudaKernelNodeParams &launch,
        const size_t *argument_sizes,
        size_t argument_count,
        const MxxGraphPatch *patches,
        size_t patch_count)
    {
        if (!body || !body->active || !node ||
            (argument_count != 0 && (!argument_sizes || !launch.kernelParams)) ||
            (patch_count != 0 && !patches))
            return set_error("invalid CUDA graph body kernel registration arguments");
        if (graph_capture_body_node_is_unique(body, node) != 0) return 1;
        GraphKernelUpdateRecord record;
        record.node = node;
        record.launch = launch;
        try
        {
            record.arguments.resize(argument_count);
            record.argument_pointers.resize(argument_count);
            for (size_t index = 0; index < argument_count; ++index)
            {
                if (argument_sizes[index] == 0 || !launch.kernelParams[index])
                    return set_error("body kernel update contains an empty argument");
                record.arguments[index].resize(argument_sizes[index]);
                memcpy(record.arguments[index].data(), launch.kernelParams[index], argument_sizes[index]);
            }
            record.launch.kernelParams = record.argument_pointers.data();
            record.launch.extra = nullptr;
            record.patches.reserve(patch_count);
            for (size_t index = 0; index < patch_count; ++index)
            {
                const MxxGraphPatch &patch = patches[index];
                if ((patch.target != MXX_GRAPH_PATCH_KERNEL_ARGUMENT_FIELD && patch.target != MXX_GRAPH_PATCH_INTEGER_ENCODING) ||
                    patch.argument_index >= argument_count ||
                    patch.byte_offset > argument_sizes[patch.argument_index] ||
                    patch.byte_count > argument_sizes[patch.argument_index] - patch.byte_offset ||
                    patch_duplicate(record.patches, patch))
                    return set_error("invalid or duplicate body kernel graph patch");
                record.patches.push_back(GraphPatchRecord{patch});
            }
            body->kernels.push_back(std::move(record));
        }
        catch (const std::exception &exception)
        {
            return set_error(exception);
        }
        return 0;
    }

    int register_body_memcpy_update(
        MxxGpuGraphBodyCapture *body,
        cudaGraphNode_t node,
        size_t fixed_bytes,
        int fixed_kind,
        const MxxGraphPatch *patches,
        size_t patch_count)
    {
        if (!body || !body->active || !node || (patch_count != 0 && !patches))
            return set_error("invalid CUDA graph body memcpy registration arguments");
        if (graph_capture_body_node_is_unique(body, node) != 0) return 1;
        cudaMemcpy3DParms params{};
        cudaError_t error = cudaGraphMemcpyNodeGetParams(node, &params);
        if (error != cudaSuccess) return set_error(error);
        if (params.extent.height != 1 || params.extent.depth != 1 ||
            params.extent.width != fixed_bytes || static_cast<int>(params.kind) != fixed_kind)
            return set_error("captured body memcpy does not match its fixed linear layout");
        GraphMemcpyUpdateRecord record;
        record.node = node;
        record.source = params.srcPtr.ptr;
        record.destination = params.dstPtr.ptr;
        record.bytes = fixed_bytes;
        record.kind = params.kind;
        try
        {
            record.patches.reserve(patch_count);
            for (size_t index = 0; index < patch_count; ++index)
            {
                const MxxGraphPatch &patch = patches[index];
                if ((patch.target != MXX_GRAPH_PATCH_MEMCPY_1D_SRC &&
                     patch.target != MXX_GRAPH_PATCH_MEMCPY_1D_DST) ||
                    patch.byte_count != sizeof(uint64_t) || patch_duplicate(record.patches, patch))
                    return set_error("invalid or duplicate body memcpy graph patch");
                record.patches.push_back(GraphPatchRecord{patch});
            }
            body->memcpys.push_back(std::move(record));
        }
        catch (const std::exception &exception)
        {
            return set_error(exception);
        }
        return 0;
    }

    int register_body_memset_update(
        MxxGpuGraphBodyCapture *body,
        cudaGraphNode_t node,
        const void *fixed_params_raw,
        const MxxGraphPatch *destination_patch)
    {
        if (!body || !body->active || !node || !fixed_params_raw || !destination_patch ||
            destination_patch->target != MXX_GRAPH_PATCH_MEMSET_1D_DST ||
            destination_patch->byte_count != sizeof(uint64_t))
            return set_error("invalid CUDA graph body memset registration arguments");
        if (graph_capture_body_node_is_unique(body, node) != 0) return 1;
        GraphMemsetUpdateRecord record;
        record.node = node;
        record.params = *static_cast<const cudaMemsetParams *>(fixed_params_raw);
        record.patch.patch = *destination_patch;
        body->memsets.push_back(std::move(record));
        return 0;
    }

    int mxx_graph_register_kernel_update_for_stream(
        GpuContext *ctx,
        void *stream_raw,
        const size_t *argument_sizes,
        size_t argument_count,
        const MxxGraphPatch *patches,
        size_t patch_count)
    {
        MxxGpuGraphCapture *capture = nullptr;
        MxxGpuGraphBodyCapture *body = nullptr;
        if (graph_capture_for_stream(
                ctx,
                reinterpret_cast<cudaStream_t>(stream_raw),
                &capture,
                &body) != 0)
        {
            return 1;
        }
        if (!capture)
        {
            return 0;
        }
        std::vector<MxxGraphPatch> normalized_patches;
        if (offset_graph_patches(capture, patches, patch_count, &normalized_patches) != 0)
            return 1;
        if (body)
        {
            cudaGraphNode_t node = nullptr;
            int status = graph_capture_latest_node(
                body->stream,
                cudaGraphNodeTypeKernel,
                &node);
            if (status != 0) return status;
            cudaKernelNodeParams launch{};
            cudaError_t error = cudaGraphKernelNodeGetParams(node, &launch);
            if (error != cudaSuccess) return set_error(error);
            return register_body_kernel_update(
                body,
                node,
                launch,
                argument_sizes,
                argument_count,
                normalized_patches.data(),
                normalized_patches.size());
        }
        cudaGraphNode_t node = nullptr;
        int status = graph_capture_latest_node(
            capture->stream,
            cudaGraphNodeTypeKernel,
            &node);
        if (status != 0)
        {
            return status;
        }
        cudaKernelNodeParams launch{};
        cudaError_t error = cudaGraphKernelNodeGetParams(node, &launch);
        if (error != cudaSuccess)
        {
            return set_error(error);
        }
        if (!capture->resident_addresses.empty())
        {
            for (auto &patch : normalized_patches)
            {
                uint64_t address = 0;
                if (patch.target == MXX_GRAPH_PATCH_INTEGER_ENCODING) address = patch.address_addend;
                else memcpy(&address, static_cast<const char *>(launch.kernelParams[patch.argument_index]) + patch.byte_offset, sizeof(address));
                const bool resident = std::any_of(
                    capture->resident_addresses.begin(),
                    capture->resident_addresses.end(),
                    [address](const MxxGpuGraphCapture::ResidentAddress &owner) {
                        return address >= owner.address && address - owner.address < owner.bytes;
                    });
                // A pointer outside the resident schema belongs to an
                // allocation created during capture.  Leave its patch
                // unresolved; resolve_fixed_addresses removes it after all
                // graph-local allocations have been identified.
                if (resident && resolve_resident_patch(capture, address, patch) != 0) return 1;
                if (patch.target == MXX_GRAPH_PATCH_INTEGER_ENCODING) patch.address_addend = 0;
            }
        }
        return mxx_graph_register_kernel_update(
            capture,
            reinterpret_cast<void *>(node),
            &launch,
            argument_sizes,
            argument_count,
            normalized_patches.data(),
            normalized_patches.size());
    }

    int mxx_graph_register_memcpy1d_update(
        MxxGpuGraphCapture *capture,
        void *node_raw,
        size_t fixed_bytes,
        int fixed_kind,
        const MxxGraphPatch *patches,
        size_t patch_count)
    {
        if (!capture || !capture->active || !node_raw ||
            (patch_count != 0 && !patches))
        {
            return set_error("invalid CUDA graph memcpy registration arguments");
        }
        auto *node = reinterpret_cast<cudaGraphNode_t>(node_raw);
        if (graph_capture_node_is_unique(capture, node) != 0)
        {
            return 1;
        }
        cudaMemcpy3DParms params{};
        cudaError_t error = cudaGraphMemcpyNodeGetParams(node, &params);
        if (error != cudaSuccess)
        {
            return set_error(error);
        }
        if (params.extent.height != 1 || params.extent.depth != 1 ||
            params.extent.width != fixed_bytes || static_cast<int>(params.kind) != fixed_kind)
        {
            return set_error("captured memcpy does not match its fixed linear layout");
        }
        GraphMemcpyUpdateRecord record;
        record.node = node;
        record.source = params.srcPtr.ptr;
        record.destination = params.dstPtr.ptr;
        record.bytes = fixed_bytes;
        record.kind = params.kind;
        try
        {
            record.patches.reserve(patch_count);
            for (size_t index = 0; index < patch_count; ++index)
            {
                const MxxGraphPatch &patch = patches[index];
                if ((patch.target != MXX_GRAPH_PATCH_MEMCPY_1D_SRC &&
                     patch.target != MXX_GRAPH_PATCH_MEMCPY_1D_DST) ||
                    patch.byte_count != sizeof(uint64_t) ||
                    patch_duplicate(record.patches, patch))
                {
                    return set_error("invalid or duplicate memcpy graph patch");
                }
                record.patches.push_back(GraphPatchRecord{patch});
            }
            capture->memcpys.push_back(std::move(record));
        }
        catch (const std::exception &exception)
        {
            return set_error(exception);
        }
        return 0;
    }

    int mxx_graph_register_memcpy1d_update_for_stream(
        GpuContext *ctx,
        void *stream_raw,
        size_t fixed_bytes,
        int fixed_kind,
        const MxxGraphPatch *patches,
        size_t patch_count)
    {
        MxxGpuGraphCapture *capture = nullptr;
        MxxGpuGraphBodyCapture *body = nullptr;
        if (graph_capture_for_stream(
                ctx,
                reinterpret_cast<cudaStream_t>(stream_raw),
                &capture,
                &body) != 0)
        {
            return 1;
        }
        if (!capture)
        {
            return 0;
        }
        std::vector<MxxGraphPatch> normalized_patches;
        if (offset_graph_patches(capture, patches, patch_count, &normalized_patches) != 0)
            return 1;
        if (body)
        {
            cudaGraphNode_t node = nullptr;
            int status = graph_capture_latest_node(
                body->stream,
                cudaGraphNodeTypeMemcpy,
                &node);
            if (status != 0) return status;
            return register_body_memcpy_update(
                body,
                node,
                fixed_bytes,
                fixed_kind,
                normalized_patches.data(),
                normalized_patches.size());
        }
        cudaGraphNode_t node = nullptr;
        int status = graph_capture_latest_node(
            capture->stream,
            cudaGraphNodeTypeMemcpy,
            &node);
        if (status != 0)
        {
            return status;
        }
        if (!capture->resident_addresses.empty())
        {
            cudaMemcpy3DParms params{};
            cudaError_t error = cudaGraphMemcpyNodeGetParams(node, &params);
            if (error != cudaSuccess) return set_error(error);
            for (auto &patch : normalized_patches)
            {
                const void *address = patch.target == MXX_GRAPH_PATCH_MEMCPY_1D_DST ? params.dstPtr.ptr : params.srcPtr.ptr;
                const uint64_t raw = reinterpret_cast<uint64_t>(address);
                const bool resident = std::any_of(
                    capture->resident_addresses.begin(),
                    capture->resident_addresses.end(),
                    [raw](const MxxGpuGraphCapture::ResidentAddress &owner) {
                        return raw >= owner.address && raw - owner.address < owner.bytes;
                    });
                if (resident && resolve_resident_patch(capture, raw, patch) != 0) return 1;
            }
        }
        return mxx_graph_register_memcpy1d_update(
            capture,
            reinterpret_cast<void *>(node),
            fixed_bytes,
            fixed_kind,
            normalized_patches.data(),
            normalized_patches.size());
    }

    int mxx_graph_register_memset1d_update(
        MxxGpuGraphCapture *capture,
        void *node_raw,
        const void *fixed_params_raw,
        const MxxGraphPatch *destination_patch)
    {
        if (!capture || !capture->active || !node_raw || !fixed_params_raw || !destination_patch ||
            destination_patch->target != MXX_GRAPH_PATCH_MEMSET_1D_DST ||
            destination_patch->byte_count != sizeof(uint64_t))
        {
            return set_error("invalid CUDA graph memset registration arguments");
        }
        auto *node = reinterpret_cast<cudaGraphNode_t>(node_raw);
        if (graph_capture_node_is_unique(capture, node) != 0)
        {
            return 1;
        }
        GraphMemsetUpdateRecord record;
        record.node = node;
        record.params = *static_cast<const cudaMemsetParams *>(fixed_params_raw);
        record.patch.patch = *destination_patch;
        capture->memsets.push_back(std::move(record));
        return 0;
    }

    int mxx_graph_register_memset1d_update_for_stream(
        GpuContext *ctx,
        void *stream_raw,
        const void *fixed_params_raw,
        const MxxGraphPatch *destination_patch)
    {
        MxxGpuGraphCapture *capture = nullptr;
        MxxGpuGraphBodyCapture *body = nullptr;
        if (graph_capture_for_stream(
                ctx,
                reinterpret_cast<cudaStream_t>(stream_raw),
                &capture,
                &body) != 0)
        {
            return 1;
        }
        if (!capture)
        {
            return 0;
        }
        std::vector<MxxGraphPatch> normalized_patches;
        if (offset_graph_patches(capture, destination_patch, 1, &normalized_patches) != 0)
            return 1;
        if (body)
        {
            cudaGraphNode_t node = nullptr;
            int status = graph_capture_latest_node(
                body->stream,
                cudaGraphNodeTypeMemset,
                &node);
            if (status != 0) return status;
            return register_body_memset_update(
                body, node, fixed_params_raw, normalized_patches.data());
        }
        cudaGraphNode_t node = nullptr;
        int status = graph_capture_latest_node(
            capture->stream,
            cudaGraphNodeTypeMemset,
            &node);
        if (status != 0)
        {
            return status;
        }
        cudaMemsetParams params{};
        cudaError_t error = cudaGraphMemsetNodeGetParams(node, &params);
        if (error != cudaSuccess)
        {
            return set_error(error);
        }
        if (!capture->resident_addresses.empty())
        {
            const uint64_t raw = reinterpret_cast<uint64_t>(params.dst);
            const bool resident = std::any_of(
                capture->resident_addresses.begin(),
                capture->resident_addresses.end(),
                [raw](const MxxGpuGraphCapture::ResidentAddress &owner) {
                    return raw >= owner.address && raw - owner.address < owner.bytes;
                });
            if (resident && resolve_resident_patch(capture, raw, normalized_patches[0]) != 0)
                return 1;
        }
        return mxx_graph_register_memset1d_update(
            capture,
            reinterpret_cast<void *>(node),
            fixed_params_raw ? fixed_params_raw : &params,
            normalized_patches.data());
    }

    int mxx_gpu_graph_upload(MxxGpuGraphExec *exec, void *launch_stream)
    {
        if (!exec || !exec->exec || !launch_stream)
        {
            return set_error("invalid mxx_gpu_graph_upload arguments");
        }
        cudaError_t error = cudaSetDevice(exec->device);
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
        cudaError_t error = cudaSetDevice(exec->device);
        if (error != cudaSuccess)
        {
            return set_error(error);
        }
        // Body launch records are attached to the captured child graph.  They
        // must be patched as part of every bind just like top-level records;
        // otherwise conditional retry replay would dereference stale
        // capture-time pointers.  CUDA accepts these updates on the retained
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
        cudaError_t error = cudaSetDevice(exec->device);
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

    int mxx_gpu_graph_add_preimage_retry_body(
        MxxGpuGraphCapture *capture,
        const MxxPreimageRetrySpec *spec,
        void *fixed_scratch,
        void *device_control,
        void *device_status,
        void *body_graph_raw)
    {
        if (!capture || !capture->active || !spec || !fixed_scratch || !device_control ||
            !device_status || spec->max_attempts == 0)
        {
            return set_error("invalid preimage retry graph arguments");
        }

#if !defined(CUDART_VERSION) || CUDART_VERSION < 12030
        return GPU_STATUS_CONDITIONAL_UNSUPPORTED;
#else
        if (spec->attempt_binding_index == UINT32_MAX ||
            spec->control_binding_index == UINT32_MAX ||
            spec->status_binding_index == UINT32_MAX ||
            spec->attempt_binding_index == spec->control_binding_index ||
            spec->attempt_binding_index == spec->status_binding_index ||
            spec->control_binding_index == spec->status_binding_index)
            return set_error("preimage retry binding indices must be distinct and valid");
        if (!body_graph_raw)
            return set_error("preimage retry registration requires a captured body graph");
        const auto body_graph = reinterpret_cast<cudaGraph_t>(body_graph_raw);
        size_t body_node_count = 0;
        cudaError_t error = cudaGraphGetNodes(body_graph, nullptr, &body_node_count);
        if (error != cudaSuccess) return set_error(error);
        if (body_node_count == 0)
            return set_error("preimage retry body must contain an attempt");
        std::vector<cudaGraphNode_t> body_nodes;
        try
        {
            body_nodes.resize(body_node_count);
            capture->preimage_retries.reserve(capture->preimage_retries.size() + 1);
        }
        catch (const std::exception &exception)
        {
            return set_error(exception);
        }
        error = cudaGraphGetNodes(body_graph, body_nodes.data(), &body_node_count);
        if (error != cudaSuccess) return set_error(error);
        for (cudaGraphNode_t node : body_nodes)
        {
            cudaGraphNodeType type;
            error = cudaGraphNodeGetType(node, &type);
            if (error != cudaSuccess) return set_error(error);
            if (type != cudaGraphNodeTypeKernel && type != cudaGraphNodeTypeMemcpy &&
                type != cudaGraphNodeTypeMemset && type != cudaGraphNodeTypeEmpty)
                return set_error("retry body contains allocation, host, event, or nested graph work");
            if (type == cudaGraphNodeTypeMemcpy)
            {
                cudaMemcpy3DParms parameters{};
                error = cudaGraphMemcpyNodeGetParams(node, &parameters);
                if (error != cudaSuccess) return set_error(error);
                if (parameters.kind != cudaMemcpyDeviceToDevice)
                    return set_error("retry body may only copy device-to-device");
            }
        }
        cudaStreamCaptureStatus capture_status;
        unsigned long long capture_id = 0;
        cudaGraph_t graph = nullptr;
        const cudaGraphNode_t *dependencies = nullptr;
        size_t dependency_count = 0;
#if CUDART_VERSION >= 13000
        error = cudaStreamGetCaptureInfo(
#else
        error = cudaStreamGetCaptureInfo_v2(
#endif
            capture->stream, &capture_status, &capture_id, &graph,
            &dependencies,
#if CUDART_VERSION >= 13000
            nullptr,
#endif
            &dependency_count);
        if (error != cudaSuccess) return set_error(error);
        if (capture_status != cudaStreamCaptureStatusActive || !graph)
            return set_error("retry registration requires an active capture graph");
        cudaGraphNode_t boundary = nullptr;
        error = cudaGraphAddEmptyNode(&boundary, graph, dependencies, dependency_count);
        if (error != cudaSuccess) return set_error(error);
        error = cudaStreamUpdateCaptureDependencies(
            capture->stream, &boundary,
#if CUDART_VERSION >= 13000
            nullptr,
#endif
            1, cudaStreamSetCaptureDependencies);
        if (error != cudaSuccess) return set_error(error);
        MxxGpuGraphCapture::PreimageRetryRegistration registration{};
        registration.spec = *spec;
        registration.fixed_scratch = fixed_scratch;
        registration.device_control = device_control;
        registration.device_status = device_status;
        registration.body_graph = reinterpret_cast<cudaGraph_t>(body_graph_raw);
        registration.boundary = boundary;
        registration.body_kernels = std::move(capture->finished_body_kernels);
        registration.body_memcpys = std::move(capture->finished_body_memcpys);
        registration.body_memsets = std::move(capture->finished_body_memsets);
        if (registration.body_kernels.empty() && registration.body_memcpys.empty() &&
            registration.body_memsets.empty())
            return set_error("preimage retry body has no registered launch schema");
        capture->preimage_retries.push_back(std::move(registration));
        return 0;
#endif
    }

    void mxx_gpu_graph_exec_destroy(MxxGpuGraphExec *exec)
    {
        if (!exec) return;
        cudaSetDevice(exec->device);
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
        for (cudaGraph_t body : exec->retained_body_graphs)
        {
            if (body)
            {
                const cudaError_t error = cudaGraphDestroy(body);
                if (error != cudaSuccess) set_error(error);
            }
        }
        exec->retained_body_graphs.clear();
        for (MxxGpuCaptureEvent *event : exec->captured_events)
        {
            mxx_gpu_capture_event_release(event);
        }
        exec->captured_events.clear();
        for (const auto &resource : exec->retained_resources) resource.destroy(resource.resource);
        exec->retained_resources.clear();
        delete exec;
    }

    int mxx_gpu_native_event_wait(MxxGpuNativeEvent *event)
    {
        if (!event || !event->event) return set_error("invalid CUDA graph event");
        cudaError_t error = cudaSetDevice(event->device);
        if (error == cudaSuccess) error = cudaEventSynchronize(event->event);
        return error == cudaSuccess ? 0 : set_error(error);
    }

    int mxx_gpu_native_event_enqueue_wait(MxxGpuNativeEvent *event, void *stream_raw)
    {
        if (!event || !event->event || !stream_raw)
            return set_error("invalid CUDA graph event wait arguments");
        cudaError_t error = cudaSetDevice(event->device);
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

    int mxx_gpu_native_event_query(MxxGpuNativeEvent *event, int *out_complete)
    {
        if (!event || !event->event || !out_complete)
            return set_error("invalid CUDA graph event query arguments");
        cudaError_t error = cudaSetDevice(event->device);
        if (error == cudaSuccess) error = cudaEventQuery(event->event);
        if (error == cudaSuccess)
        {
            *out_complete = 1;
            return 0;
        }
        if (error == cudaErrorNotReady)
        {
            *out_complete = 0;
            cudaGetLastError();
            return 0;
        }
        return set_error(error);
    }

    void mxx_gpu_native_event_destroy(MxxGpuNativeEvent *event)
    {
        if (!event) return;
        cudaSetDevice(event->device);
        if (event->event)
        {
            const cudaError_t error = cudaEventDestroy(event->event);
            if (error != cudaSuccess) set_error(error);
        }
        delete event;
    }

    int mxx_gpu_graph_memory_snapshot(
        int physical_device,
        MxxGraphMemorySnapshot *out_snapshot)
    {
        if (!out_snapshot || physical_device < 0)
            return set_error("invalid mxx_gpu_graph_memory_snapshot arguments");
        size_t used_current = 0;
        size_t used_high = 0;
        size_t reserved_current = 0;
        if (gpu_default_mempool_get_usage(
                physical_device, &used_current, &used_high, &reserved_current) != 0)
        {
            return 1;
        }
        cudaMemPool_t pool = nullptr;
        cudaError_t error = cudaSetDevice(physical_device);
        if (error == cudaSuccess) error = cudaDeviceGetDefaultMemPool(&pool, physical_device);
        uint64_t reserved_high = 0;
        if (error == cudaSuccess)
        {
            error = cudaMemPoolGetAttribute(
                pool, cudaMemPoolAttrReservedMemHigh, &reserved_high);
        }
        if (error != cudaSuccess) return set_error(error);
        out_snapshot->used_current = used_current;
        out_snapshot->used_high = used_high;
        out_snapshot->reserved_current = reserved_current;
        out_snapshot->reserved_high = reserved_high;
        return 0;
    }
}

static int register_capture_event(
    MxxGpuGraphCapture *capture,
    MxxGpuCaptureEvent *event)
{
    if (!capture || !event)
    {
        return gpu_set_last_error("invalid CUDA capture event registration");
    }
    try
    {
        ++event->references;
        capture->captured_events.push_back(event);
    }
    catch (const std::exception &exception)
    {
        --event->references;
        return gpu_set_last_error(exception.what());
    }
    return 0;
}

MxxGpuCaptureEvent *mxx_gpu_capture_event_create(GpuContext *ctx, int device)
{
    if (!ctx || !ctx->execution)
    {
        gpu_set_last_error("invalid GPU context for capture event");
        return nullptr;
    }
    auto &owner = *ctx->execution;
    std::lock_guard<std::mutex> lock(owner.capture_mutex);
    if (!owner.capture_active || !owner.capture_handle)
    {
        gpu_set_last_error("capture event requested outside an active graph capture");
        return nullptr;
    }
    auto *event = new (std::nothrow) MxxGpuCaptureEvent();
    if (!event)
    {
        gpu_set_last_error("failed to allocate CUDA capture event state");
        return nullptr;
    }
    cudaError_t error = cudaSetDevice(device);
    if (error == cudaSuccess)
    {
        error = cudaEventCreateWithFlags(&event->event, cudaEventDisableTiming);
    }
    if (error != cudaSuccess)
    {
        if (event->event) cudaEventDestroy(event->event);
        delete event;
        gpu_set_last_error_cuda(static_cast<int>(error));
        return nullptr;
    }
    event->capture_generation = owner.capture_generation;
    // The matrix keeps the initial reference. The active graph retains the
    // second reference until its executable is destroyed.
    if (register_capture_event(owner.capture_handle, event) != 0)
    {
        cudaEventDestroy(event->event);
        delete event;
        return nullptr;
    }
    return event;
}

void mxx_gpu_capture_event_release(MxxGpuCaptureEvent *event)
{
    if (!event || event->references == 0)
    {
        return;
    }
    --event->references;
    if (event->references != 0)
    {
        return;
    }
    if (event->event)
    {
        const cudaError_t error = cudaEventDestroy(event->event);
        if (error != cudaSuccess)
        {
            gpu_set_last_error_cuda(static_cast<int>(error));
        }
        event->event = nullptr;
    }
    delete event;
}
