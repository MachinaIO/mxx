#include "Runtime.cuh"
#include "gpu_admission.cuh"

#include <algorithm>
#include <array>
#include <unordered_map>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <deque>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
    // Two completed transfer buffers per device cover the load/store pipeline.
    // Capacity follows the largest active chunks, never graph or Family size.
    // Only bookkeeping holds this lock; allocation, copies and GPU work do not.
    struct PinnedHostPool
    {
        struct Block {
            void *pointer = nullptr;
            size_t bytes = 0;
            int device = 0;
            GpuPreparedPinnedLease *prepared = nullptr;
        };
        std::mutex mutex;
        std::unordered_map<void *, Block> allocated;
        std::unordered_map<int, std::array<Block, 2>> ready;

        ~PinnedHostPool()
        {
            for (const auto &device : ready)
            {
                GpuAllocationActivity activity(nullptr, device.first);
                for (const auto &block : device.second)
                    if (block.pointer && cudaFreeHost(block.pointer) != cudaSuccess)
                        gpu_device_mark_allocation_unknown(device.first);
            }
        }

        void *take(int device, size_t bytes)
        {
            std::lock_guard<std::mutex> lock(mutex);
            auto &slots = ready[device];
            Block *best = nullptr;
            for (auto &slot : slots)
                if (slot.pointer && slot.bytes >= bytes && (!best || slot.bytes < best->bytes))
                    best = &slot;
            if (!best) return nullptr;
            void *pointer = best->pointer;
            *best = {};
            return pointer;
        }

        cudaError_t release(void *pointer, int fallback_device)
        {
            GpuPreparedPinnedLease *prepared = nullptr;
            void *discard = pointer;
            int allocation_device = fallback_device;
            std::optional<GpuAllocationActivity> activity;
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto found = allocated.find(pointer);
                if (found != allocated.end()) allocation_device = found->second.device;
                // A portable pinned block can be dropped on another device's
                // thread. Track its recorded placement before changing cache
                // ownership, then retain the scope through any eviction/free.
                activity.emplace(nullptr, allocation_device);
                if (found != allocated.end())
                {
                    prepared = found->second.prepared;
                    if (prepared) {
                        found->second.prepared = nullptr;
                        discard = nullptr;
                    } else {
                        auto &slots = ready[found->second.device];
                        auto smallest = std::min_element(slots.begin(), slots.end(),
                            [](const Block &a, const Block &b) { return a.bytes < b.bytes; });
                        if (smallest->bytes <= found->second.bytes)
                        {
                            discard = smallest->pointer;
                            *smallest = found->second;
                        }
                        if (discard) allocated.erase(discard);
                    }
                }
            }
            // Recycling can destroy the last storage owner and re-enter this
            // pool to release its backing. Never invoke it under the pool lock.
            if (prepared) gpu_prepared_pinned_recycle(prepared);
            // Unregistered pointers belong to existing CUDA callers. Their
            // deferred destruction keeps its original cudaFreeHost behavior.
            const cudaError_t error = discard ? cudaFreeHost(discard) : cudaSuccess;
            if (error != cudaSuccess) gpu_device_mark_allocation_unknown(allocation_device);
            return error;
        }

        int mark_deferred(GpuContext *ctx, void *pointer)
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto found = allocated.find(pointer);
            if (found == allocated.end() || !found->second.prepared) return 0;
            return gpu_prepared_pinned_defer(found->second.prepared, ctx);
        }

        void clear(int device)
        {
            std::array<Block, 2> blocks{};
            {
                std::lock_guard<std::mutex> lock(mutex);
                auto found = ready.find(device);
                if (found == ready.end()) return;
                blocks = found->second;
                ready.erase(found);
                for (const auto &block : blocks) allocated.erase(block.pointer);
            }
            for (const auto &block : blocks)
                if (block.pointer && cudaFreeHost(block.pointer) != cudaSuccess)
                    gpu_device_mark_allocation_unknown(device);
        }
    };

    PinnedHostPool &pinned_host_pool()
    {
        static PinnedHostPool pool;
        return pool;
    }
}

struct PinnedHostReclaimer
{
    struct Job
    {
        int device;
        cudaEvent_t completion;
        std::vector<void *> pointers;
        std::shared_ptr<GpuCudaResource> resource;
    };

    explicit PinnedHostReclaimer(GpuExecutionOwner *execution_owner) : owner(execution_owner)
    {
        // Start only after every member (including worker-visible flags) has
        // completed initialization, irrespective of declaration order.
        worker = std::thread(&PinnedHostReclaimer::run, this);
    }

    ~PinnedHostReclaimer()
    {
        shutdown();
    }

    int enqueue(int device, cudaEvent_t completion, std::vector<void *> &&pointers,
                std::shared_ptr<GpuCudaResource> resource = nullptr)
    {
        GpuAllocationActivity activity(owner, device);
        try
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (stopping || joined)
            {
                record_failure_locked("pinned-host reclaimer is stopped");
                return 1;
            }
            pending.push_back(Job{device, completion, std::move(pointers), std::move(resource)});
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

    // Uses only the existing queue bookkeeping lock; never waits for CUDA or
    // for the worker to become idle. Negative means completion is uncertain.
    int query_idle()
    {
        std::lock_guard<std::mutex> lock(mutex);
        return failed ? -1 : (pending.empty() && active == 0 ? 1 : 0);
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
        gpu_execution_mark_allocation_unknown(owner);
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
        // Retirement runs independently of foreground setup/submission. Track
        // its complete device activity, including event waits and host frees,
        // without making setup closure race a foreground-allocation counter.
        // Epoch observations still require an idle reclaimer and stable device
        // counters; the initial epoch explicitly drains this worker.
        GpuAllocationActivity activity(nullptr, job.device);
        cudaError_t error = cudaSetDevice(job.device);
        if (error == cudaSuccess)
        {
            error = cudaEventSynchronize(job.completion);
        }
        if (error != cudaSuccess)
        {
            gpu_device_mark_allocation_unknown(job.device);
            record_failure(cudaGetErrorString(error));
            if (job.resource) job.resource->quarantine();
            // The event and pointers are intentionally leaked. Once
            // synchronization is uncertain, freeing host memory could race
            // with an in-flight asynchronous copy.
            return;
        }

        const int resource_status = job.resource ? job.resource->release() : 0;
        error = job.resource ? cudaSuccess : cudaEventDestroy(job.completion);
        job.resource.reset();
        if (resource_status != 0 || error != cudaSuccess)
        {
            gpu_device_mark_allocation_unknown(job.device);
            record_failure(resource_status != 0 ? "CUDA completion resource release failed" : cudaGetErrorString(error));
            // Keep the pointers leaked when event destruction is uncertain.
            return;
        }

        for (void *pointer : job.pointers)
        {
            if (!pointer)
            {
                continue;
            }
            error = pinned_host_pool().release(pointer, job.device);
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

    GpuExecutionOwner *owner;
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
    std::atomic<uint64_t> device_allocation_revisions[MAX_TRACKED_GPU_DEVICES]{};
    std::atomic<size_t> active_device_allocation_calls[MAX_TRACKED_GPU_DEVICES]{};
    std::atomic<bool> device_allocation_activity_unknown[MAX_TRACKED_GPU_DEVICES]{};

    void advance_allocation_revision(std::atomic<uint64_t> &revision,
                                     std::atomic<bool> &unknown)
    {
        if (revision.fetch_add(1, std::memory_order_seq_cst) == UINT64_MAX)
            unknown.store(true, std::memory_order_seq_cst);
    }

    void begin_device_allocation_activity(int device)
    {
        const size_t index = static_cast<size_t>(device);
        if (active_device_allocation_calls[index].fetch_add(1, std::memory_order_seq_cst) == SIZE_MAX)
            device_allocation_activity_unknown[index].store(true, std::memory_order_seq_cst);
        advance_allocation_revision(device_allocation_revisions[index],
                                    device_allocation_activity_unknown[index]);
    }

    void end_device_allocation_activity(int device)
    {
        const size_t index = static_cast<size_t>(device);
        advance_allocation_revision(device_allocation_revisions[index],
                                    device_allocation_activity_unknown[index]);
        active_device_allocation_calls[index].fetch_sub(1, std::memory_order_seq_cst);
    }

    int set_error(const char *msg)
    {
        last_error = msg ? msg : "unknown error";
        return 1;
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
        auto owner = events->execution;
        GpuAllocationActivity owner_activity(owner.get(), -1);
        for (const auto &entry : events->entries)
        {
            GpuAllocationActivity activity(nullptr, entry.device);
            cudaError_t error = cudaSetDevice(entry.device);
            if (error == cudaSuccess && !entry.resource) error = cudaEventDestroy(entry.event);
            if (error != cudaSuccess)
            {
                gpu_execution_mark_allocation_unknown(owner.get());
                gpu_device_mark_allocation_unknown(entry.device);
                set_error(cudaGetErrorString(error));
            }
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
                gpu_device_mark_allocation_unknown(device);
                return set_error(cudaGetErrorString(err));
            }
            if (partition >= owner->completion_events_by_partition.size() ||
                owner->completion_events_by_partition[partition].empty() ||
                !owner->completion_events_by_partition[partition].back())
                return set_error("release fence event was not prepared");
            const cudaEvent_t epoch = owner->completion_events_by_partition[partition].back();
            err = cudaEventRecord(epoch, stream);
            if (err == cudaSuccess) err = cudaEventSynchronize(epoch);
            if (err != cudaSuccess)
            {
                gpu_device_mark_allocation_unknown(device);
                return set_error(cudaGetErrorString(err));
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
            gpu_execution_mark_allocation_unknown(owner);
            for (int device : owner->gpu_ids) gpu_device_mark_allocation_unknown(device);
            set_error("GPU context release cleanup failed");
        }
        for (size_t partition = 0; partition < owner->gpu_ids.size(); ++partition)
        {
            const int device = owner->gpu_ids[partition];
            if (cudaSetDevice(device) != cudaSuccess)
            {
                gpu_execution_mark_allocation_unknown(owner);
                gpu_device_mark_allocation_unknown(device);
                continue;
            }
            if (partition < owner->timing_partitions.size()) {
                auto &timing = owner->timing_partitions[partition];
                const auto destroy_event = [&](cudaEvent_t event) {
                    if (event && cudaEventDestroy(event) != cudaSuccess) {
                        gpu_execution_mark_allocation_unknown(owner);
                        gpu_device_mark_allocation_unknown(device);
                    }
                };
                for (cudaEvent_t event : timing.before) destroy_event(event);
                for (cudaEvent_t event : timing.after) destroy_event(event);
                destroy_event(timing.start);
                destroy_event(timing.stop);
                if (timing.stream && cudaStreamDestroy(timing.stream) != cudaSuccess) {
                    gpu_execution_mark_allocation_unknown(owner);
                    gpu_device_mark_allocation_unknown(device);
                }
            }
            if (partition < owner->completion_events_by_partition.size()) {
                for (cudaEvent_t event : owner->completion_events_by_partition[partition])
                    if (event && cudaEventDestroy(event) != cudaSuccess) {
                        gpu_execution_mark_allocation_unknown(owner);
                        gpu_device_mark_allocation_unknown(device);
                    }
            }
            if (partition < owner->release_streams_by_partition.size() &&
                owner->release_streams_by_partition[partition])
            {
                if (cudaStreamDestroy(owner->release_streams_by_partition[partition]) != cudaSuccess)
                {
                    gpu_execution_mark_allocation_unknown(owner);
                    gpu_device_mark_allocation_unknown(device);
                }
                owner->release_streams_by_partition[partition] = nullptr;
            }
            if (partition < owner->compute_streams_by_partition.size())
            {
                for (cudaStream_t &stream : owner->compute_streams_by_partition[partition])
                {
                    if (stream)
                    {
                        if (cudaStreamDestroy(stream) != cudaSuccess)
                        {
                            gpu_execution_mark_allocation_unknown(owner);
                            gpu_device_mark_allocation_unknown(device);
                        }
                        stream = nullptr;
                    }
                }
            }
        }
        owner->timing_partitions.clear();
        owner->completion_events_by_partition.clear();
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

    GpuRingDeviceConstants make_empty_ring_device_constants(
        int device,
        size_t limb_count,
        uint32_t ring_dimension)
    {
        GpuRingDeviceConstants out{};
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
        out.garner_inverses = nullptr;
        return out;
    }

    void free_ring_device_constants_entry(GpuRingDeviceConstants &entry, cudaStream_t release = nullptr)
    {
        if (entry.device < 0)
        {
            return;
        }
        if (cudaSetDevice(entry.device) != cudaSuccess)
        {
            gpu_device_mark_allocation_unknown(entry.device);
            return;
        }
        void *pointers[] = {entry.twiddle_forward, entry.twiddle_inverse,
            entry.twiddle_shoup_forward, entry.twiddle_shoup_inverse,
            entry.moduli, entry.n_inv, entry.n_inv_shoup, entry.garner_inverses};
        for (void *pointer : pointers)
        {
            if (pointer)
            {
                const cudaError_t status = release ? cudaFreeAsync(pointer, release) : cudaFree(pointer);
                if (status != cudaSuccess)
                {
                    gpu_device_mark_allocation_unknown(entry.device);
                    set_error(cudaGetErrorString(status));
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
        entry.garner_inverses = nullptr;
    }

    void free_ring_device_constants(std::vector<GpuRingDeviceConstants> &entries,
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
            free_ring_device_constants_entry(entry, release);
        }
        entries.clear();
    }

    void upload_ring_small_constants_to_device(
        int device,
        const std::vector<uint64_t> &limb_moduli,
        const std::vector<uint64_t> &limb_n_inv,
        const std::vector<uint64_t> &limb_n_inv_shoup,
        const std::vector<uint64_t> &garner_inverses,
        GpuRingDeviceConstants *out_entry)
    {
        const size_t limb_count = limb_moduli.size();
        if (limb_count == 0 || limb_count > GPU_RUNTIME_MAX_LIMBS)
        {
            throw std::runtime_error("invalid limb count in upload_ring_small_constants_to_device");
        }
        if (limb_n_inv.size() != limb_count ||
            limb_n_inv_shoup.size() != limb_count || garner_inverses.size() != limb_count * limb_count)
        {
            throw std::runtime_error("inconsistent limb constants in upload_ring_small_constants_to_device");
        }

        if (!out_entry)
        {
            throw std::runtime_error("null output entry in upload_ring_small_constants_to_device");
        }
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        auto alloc_and_copy = [&](uint64_t **dst, const std::vector<uint64_t> &src)
        {
            const size_t bytes = src.size() * sizeof(uint64_t);
            err = cudaMalloc(reinterpret_cast<void **>(dst), bytes);
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
            err = cudaMemcpy(*dst, src.data(), bytes, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        };
        try
        {
            alloc_and_copy(&out_entry->moduli, limb_moduli);
            alloc_and_copy(&out_entry->n_inv, limb_n_inv);
            alloc_and_copy(&out_entry->n_inv_shoup, limb_n_inv_shoup);
            alloc_and_copy(&out_entry->garner_inverses, garner_inverses);
        }
        catch (...)
        {
            free_ring_device_constants_entry(*out_entry);
            throw;
        }
    }

    void upload_ntt_twiddles_to_device(
        int device,
        const std::vector<uint64_t> &twiddle_forward,
        const std::vector<uint64_t> &twiddle_inverse,
        const std::vector<uint64_t> &twiddle_shoup_forward,
        const std::vector<uint64_t> &twiddle_shoup_inverse,
        GpuRingDeviceConstants *out_entry)
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
            free_ring_device_constants_entry(*out_entry);
            throw;
        }
    }
}

GpuAllocationActivity::GpuAllocationActivity(GpuExecutionOwner *owner, int device)
    : owner_(owner), device_(device)
{
    if ((device < 0 && (device != -1 || !owner)) ||
        (device >= 0 && static_cast<size_t>(device) >= MAX_TRACKED_GPU_DEVICES))
    {
        gpu_execution_mark_allocation_unknown(owner);
        device_ = -2;
        return;
    }
    if (owner_)
    {
        if (owner_->active_allocation_calls.fetch_add(1, std::memory_order_seq_cst) == SIZE_MAX)
            owner_->allocation_activity_unknown.store(true, std::memory_order_seq_cst);
        advance_allocation_revision(owner_->allocation_revision, owner_->allocation_activity_unknown);
    }
    if (device_ >= 0) begin_device_allocation_activity(device_);
    else for (int member : owner_->gpu_ids) begin_device_allocation_activity(member);
}

GpuAllocationActivity::~GpuAllocationActivity()
{
    if (device_ == -2) return;
    if (device_ >= 0) end_device_allocation_activity(device_);
    else for (int member : owner_->gpu_ids) end_device_allocation_activity(member);
    if (owner_)
    {
        advance_allocation_revision(owner_->allocation_revision, owner_->allocation_activity_unknown);
        owner_->active_allocation_calls.fetch_sub(1, std::memory_order_seq_cst);
    }
}

void gpu_execution_mark_allocation_unknown(GpuExecutionOwner *owner)
{
    if (owner) owner->allocation_activity_unknown.store(true, std::memory_order_seq_cst);
}

void gpu_device_mark_allocation_unknown(int device)
{
    if (device >= 0 && static_cast<size_t>(device) < MAX_TRACKED_GPU_DEVICES)
        device_allocation_activity_unknown[static_cast<size_t>(device)].store(
            true, std::memory_order_seq_cst);
}

extern "C" int gpu_context_admission_is_required(const GpuContext *ctx)
{
    return ctx && ctx->execution &&
        ctx->execution->admission_required.load(std::memory_order_acquire) ? 1 : 0;
}

extern "C" int gpu_context_retire_stream(const GpuContext *ctx, int device, cudaStream_t stream)
{
    if (!ctx || !ctx->execution)
        return set_error("invalid GPU stream retirement context");
    GpuAllocationActivity activity(ctx->execution.get(), device);
    auto &owner = *ctx->execution;
    const auto fail = [&owner](const char *message) {
        owner.memory_release_failed.store(true, std::memory_order_release);
        owner.unretired_work.store(true, std::memory_order_release);
        return set_error(message);
    };
    if (owner.unretired_work.load(std::memory_order_acquire))
        return fail("GPU execution has unretired work; its resources remain quarantined");
    const auto found = std::find(owner.gpu_ids.begin(), owner.gpu_ids.end(), device);
    if (!stream || found == owner.gpu_ids.end())
        return fail("invalid GPU stream retirement placement");
    const size_t partition = static_cast<size_t>(found - owner.gpu_ids.begin());
    if (partition >= owner.compute_streams_by_partition.size() ||
        partition >= owner.release_streams_by_partition.size() ||
        !owner.release_streams_by_partition[partition])
        return fail("missing GPU stream retirement resources");
    int current = 0;
    cudaError_t error = cudaGetDevice(&current);
    if (error != cudaSuccess) return fail(cudaGetErrorString(error));
    error = cudaSetDevice(device);
    GpuCudaResource completion_resource;
    if (error == cudaSuccess) {
        const int status = completion_resource.acquire(ctx, device, GPU_PREPARED_COMPLETION_EVENT);
        if (status != 0) {
            cudaSetDevice(current);
            return fail("stream retirement requires a prepared completion event");
        }
    }
    const cudaEvent_t completion = completion_resource.event;
    if (error == cudaSuccess) error = cudaEventRecord(completion, stream);
    if (error == cudaSuccess)
        error = cudaStreamWaitEvent(owner.release_streams_by_partition[partition], completion, 0);
    if (error == cudaSuccess)
    {
        // All matrices keep their original producer stream. Joining the failed
        // consumer here protects both final release and subsequent in-place use,
        // even if the ordinary per-input/output completion update was interrupted.
        // This conservative fan-out is confined to a failed enqueue.
        for (cudaStream_t producer : owner.compute_streams_by_partition[partition])
        {
            if (!producer) { error = cudaErrorInvalidResourceHandle; break; }
            if (producer != stream)
            {
                error = cudaStreamWaitEvent(producer, completion, 0);
                if (error != cudaSuccess) break;
            }
        }
    }
    if (error != cudaSuccess) completion_resource.quarantine();
    const int released = completion_resource.release();
    if (released != 0 && error == cudaSuccess) {
        cudaSetDevice(current);
        return fail("failed to release stream retirement event");
    }
    const cudaError_t restored = cudaSetDevice(current);
    if (error == cudaSuccess) error = restored;
    return error == cudaSuccess ? 0 : fail(cudaGetErrorString(error));
}

GpuExecutionOwner::~GpuExecutionOwner()
{
    GpuAllocationActivity activity(this, -1);
    if (allocation_activity_unknown.load(std::memory_order_seq_cst) ||
        memory_release_failed.load(std::memory_order_acquire) ||
        unretired_work.load(std::memory_order_acquire))
        for (int device : gpu_ids) gpu_device_mark_allocation_unknown(device);
    destroy_context_streams(this);
    if (registered)
    {
        for (int device : gpu_ids)
        {
            if (live_context_counts[static_cast<size_t>(device)].fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                pinned_host_pool().clear(device);
            }
        }
    }
}

extern "C" int gpu_set_last_error(const char *msg)
{
    return set_error(msg);
}

// Measurement handles borrow the setup-owned group under timing_active. These
// joins do not replace the operands' production reader/release dependencies.
struct GpuDeviceTiming
{
    std::shared_ptr<GpuExecutionOwner> owner;
    bool stopped = false;
    bool failed = false;

    ~GpuDeviceTiming()
    {
        int current = 0;
        const bool restore = cudaGetDevice(&current) == cudaSuccess;
        if (owner) {
            GpuAllocationActivity activity(owner.get(), -1);
            // No host observer remains. Prior waits retain their recorded
            // event state when the next span rerecords these same handles.
            owner->timing_active.store(false, std::memory_order_release);
        }
        owner.reset();
        if (restore && cudaSetDevice(current) != cudaSuccess)
            gpu_device_mark_allocation_unknown(current);
    }
};

extern "C"
{
    int gpu_context_begin_device_timing(const GpuContext *ctx, GpuDeviceTiming **out_timing)
    {
        if (!ctx || !ctx->execution || !out_timing)
            return set_error("invalid GPU device timing context");
        *out_timing = nullptr;
        auto &owner = *ctx->execution;
        GpuAllocationActivity activity(&owner, -1);
        if (owner.memory_release_failed.load(std::memory_order_acquire) ||
            owner.unretired_work.load(std::memory_order_acquire))
            return set_error("cannot measure a failed GPU execution owner");
        int current = 0;
        cudaError_t error = cudaGetDevice(&current);
        if (error != cudaSuccess) return set_error(cudaGetErrorString(error));
        std::unique_ptr<GpuDeviceTiming> timing;
        try
        {
            timing = std::make_unique<GpuDeviceTiming>();
            bool expected = false;
            if (!owner.timing_active.compare_exchange_strong(
                    expected, true, std::memory_order_acq_rel))
                return set_error("GPU execution owner already has an active measurement");
            timing->owner = ctx->execution;
            if (owner.compute_streams_by_partition.size() != owner.gpu_ids.size() ||
                owner.release_streams_by_partition.size() != owner.gpu_ids.size())
                throw std::runtime_error("missing GPU timing stream partitions");
            if (owner.timing_partitions.size() != owner.gpu_ids.size())
                throw std::runtime_error("missing setup-owned GPU timing resources");
            for (auto &partition : owner.timing_partitions)
            {
                if (!partition.stream || !partition.start || !partition.stop ||
                    partition.before.size() != partition.participants.size() ||
                    partition.after.size() != partition.participants.size())
                    throw std::runtime_error("incomplete setup-owned GPU timing resources");
                error = cudaSetDevice(partition.device);
                for (size_t stream = 0;
                     error == cudaSuccess && stream < partition.participants.size(); ++stream)
                {
                    if (!partition.participants[stream] || !partition.before[stream] ||
                        !partition.after[stream])
                        throw std::runtime_error("missing GPU timing participant resources");
                    if (error == cudaSuccess)
                        error = cudaEventRecord(
                            partition.before[stream], partition.participants[stream]);
                    if (error == cudaSuccess)
                        error = cudaStreamWaitEvent(partition.stream, partition.before[stream], 0);
                }
                if (error == cudaSuccess)
                    error = cudaEventRecord(partition.start, partition.stream);
                if (error == cudaSuccess)
                {
                    for (cudaStream_t stream : partition.participants)
                    {
                        error = cudaStreamWaitEvent(stream, partition.start, 0);
                        if (error != cudaSuccess) break;
                    }
                }
                if (error != cudaSuccess) break;
            }
        }
        catch (const std::exception &exception)
        {
            gpu_execution_mark_allocation_unknown(&owner);
            timing.reset();
            cudaSetDevice(current);
            return set_error(exception.what());
        }
        const cudaError_t restored = cudaSetDevice(current);
        if (error == cudaSuccess) error = restored;
        if (error != cudaSuccess)
        {
            gpu_execution_mark_allocation_unknown(&owner);
            timing.reset();
            return set_error(cudaGetErrorString(error));
        }
        *out_timing = timing.release();
        return 0;
    }

    int gpu_device_timing_stop(GpuDeviceTiming *timing)
    {
        if (!timing || timing->failed)
            return set_error("invalid or failed GPU device timing");
        GpuAllocationActivity activity(timing->owner.get(), -1);
        if (timing->stopped) return 0;
        if (timing->owner->memory_release_failed.load(std::memory_order_acquire) ||
            timing->owner->unretired_work.load(std::memory_order_acquire))
        {
            timing->failed = true;
            return set_error("GPU execution failed during device timing");
        }
        int current = 0;
        cudaError_t error = cudaGetDevice(&current);
        if (error != cudaSuccess)
        {
            gpu_execution_mark_allocation_unknown(timing->owner.get());
            timing->failed = true;
            return set_error(cudaGetErrorString(error));
        }
        // Enqueue every device's stop before waiting for any device. Logical
        // waves may have disjoint streams and independent device timelines.
        for (auto &partition : timing->owner->timing_partitions)
        {
            error = cudaSetDevice(partition.device);
            for (size_t stream = 0;
                 error == cudaSuccess && stream < partition.participants.size(); ++stream)
            {
                error = cudaEventRecord(partition.after[stream], partition.participants[stream]);
                if (error == cudaSuccess)
                    error = cudaStreamWaitEvent(partition.stream, partition.after[stream], 0);
            }
            if (error == cudaSuccess)
                error = cudaEventRecord(partition.stop, partition.stream);
            if (error != cudaSuccess) break;
        }
        const cudaError_t restored = cudaSetDevice(current);
        if (error == cudaSuccess) error = restored;
        if (error != cudaSuccess)
        {
            gpu_execution_mark_allocation_unknown(timing->owner.get());
            timing->failed = true;
            return set_error(cudaGetErrorString(error));
        }
        timing->stopped = true;
        return 0;
    }

    int gpu_device_timing_elapsed(
        GpuDeviceTiming *timing, int *out_devices, double *out_seconds, size_t count)
    {
        if (!timing || !timing->stopped || timing->failed || !out_devices || !out_seconds ||
            count != timing->owner->timing_partitions.size())
            return set_error("invalid or incomplete GPU device timing result");
        GpuAllocationActivity activity(timing->owner.get(), -1);
        int current = 0;
        cudaError_t error = cudaGetDevice(&current);
        if (error != cudaSuccess) return set_error(cudaGetErrorString(error));
        for (size_t index = 0; index < count; ++index)
        {
            const auto &partition = timing->owner->timing_partitions[index];
            error = cudaSetDevice(partition.device);
            // This is the sole wait, at the explicit measurement boundary.
            if (error == cudaSuccess) error = cudaEventSynchronize(partition.stop);
            float milliseconds = 0;
            if (error == cudaSuccess)
                error = cudaEventElapsedTime(&milliseconds, partition.start, partition.stop);
            if (error != cudaSuccess) break;
            out_devices[index] = partition.device;
            out_seconds[index] = static_cast<double>(milliseconds) / 1000.0;
        }
        const cudaError_t restored = cudaSetDevice(current);
        if (error == cudaSuccess) error = restored;
        if (error != cudaSuccess)
        {
            gpu_execution_mark_allocation_unknown(timing->owner.get());
            timing->failed = true;
            return set_error(cudaGetErrorString(error));
        }
        if (timing->owner->memory_release_failed.load(std::memory_order_acquire) ||
            timing->owner->unretired_work.load(std::memory_order_acquire))
        {
            timing->failed = true;
            return set_error("GPU execution failed during device timing completion");
        }
        return 0;
    }

    void gpu_device_timing_destroy(GpuDeviceTiming *timing)
    {
        delete timing;
    }

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
        // Remain active through catch-path cleanup, before this context has
        // been registered or its persistent tables have been allocated.
        std::vector<std::unique_ptr<GpuAllocationActivity>> construction_activity;
        try
        {
            if (!out_ctx || !moduli || moduli_len == 0 || stream_pool_size == 0)
            {
                return set_error("invalid context arguments");
            }
            *out_ctx = nullptr;
            if (related_context && related_context->execution) {
                construction_activity.emplace_back(std::make_unique<GpuAllocationActivity>(
                    related_context->execution.get(), -1));
                const auto &owner = *related_context->execution;
                if (owner.admission_required.load(std::memory_order_acquire) ||
                    owner.pinned_admission_required.load(std::memory_order_acquire) ||
                    owner.transfer_admission_required.load(std::memory_order_acquire) ||
                    owner.resource_admission_required.load(std::memory_order_acquire))
                    return set_error("related GPU parameter contexts must be provisioned before admission");
            }
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

            construction_activity.reserve(gpu_list.size());
            for (int device : gpu_list)
            {
                if (device < 0 || static_cast<size_t>(device) >= MAX_TRACKED_GPU_DEVICES)
                    return set_error("GPU device ordinal exceeds activity counter capacity");
                construction_activity.emplace_back(std::make_unique<GpuAllocationActivity>(
                    related_context ? related_context->execution.get() : nullptr, device));
                advance_allocation_revision(context_generations[static_cast<size_t>(device)],
                    device_allocation_activity_unknown[static_cast<size_t>(device)]);
            }
            validate_gpu_list(gpu_list);
            if (related_context &&
                (!related_context->execution ||
                 related_context->execution->unretired_work.load(std::memory_order_acquire) ||
                 related_context->gpu_ids != gpu_list ||
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
                gpu_ctx->execution->pinned_host_reclaimer =
                    new PinnedHostReclaimer(gpu_ctx->execution.get());
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
            gpu_ctx->execution->completion_events_by_partition.resize(gpu_ctx->gpu_ids.size());
            gpu_ctx->execution->kernel_partitions.resize(gpu_ctx->gpu_ids.size());
            for (size_t partition = 0; partition < gpu_ctx->gpu_ids.size(); ++partition)
            {
                const int device = gpu_ctx->gpu_ids[partition];
                cudaError_t err = cudaSetDevice(device);
                if (err != cudaSuccess)
                {
                    throw std::runtime_error(cudaGetErrorString(err));
                }
                if (gpu_matrix_prepare_kernels(
                        &gpu_ctx->execution->kernel_partitions[partition]) != 0)
                    throw std::runtime_error(gpu_last_error());
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
                auto &events = gpu_ctx->execution->completion_events_by_partition[partition];
                events.resize(stream_pool_size + 1, nullptr);
                for (cudaEvent_t &event : events) {
                    err = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
                    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
                }
            }
            }
            if (!related_context)
            {
                auto &owner = *gpu_ctx->execution;
                owner.timing_partitions.resize(owner.gpu_ids.size());
                for (size_t index = 0; index < owner.gpu_ids.size(); ++index)
                {
                    auto &partition = owner.timing_partitions[index];
                    partition.device = owner.gpu_ids[index];
                    partition.participants = owner.compute_streams_by_partition[index];
                    partition.participants.push_back(owner.release_streams_by_partition[index]);
                    partition.before.resize(partition.participants.size(), nullptr);
                    partition.after.resize(partition.participants.size(), nullptr);
                    cudaError_t error = cudaSetDevice(partition.device);
                    if (error == cudaSuccess)
                        error = cudaStreamCreateWithFlags(&partition.stream, cudaStreamNonBlocking);
                    if (error == cudaSuccess) error = cudaEventCreate(&partition.start);
                    if (error == cudaSuccess) error = cudaEventCreate(&partition.stop);
                    for (size_t stream = 0;
                         error == cudaSuccess && stream < partition.participants.size(); ++stream)
                    {
                        error = cudaEventCreateWithFlags(&partition.before[stream], cudaEventDisableTiming);
                        if (error == cudaSuccess)
                            error = cudaEventCreateWithFlags(&partition.after[stream], cudaEventDisableTiming);
                    }
                    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
                }
            }
            gpu_ctx->ring_device_constants.reserve(gpu_ctx->gpu_ids.size());
            for (int device : gpu_ctx->gpu_ids)
            {
                GpuRingDeviceConstants device_constants =
                    make_empty_ring_device_constants(
                        device,
                        limb_count,
                        static_cast<uint32_t>(n_u64));
                upload_ring_small_constants_to_device(
                    device,
                    limb_moduli,
                    limb_n_inv,
                    limb_n_inv_shoup,
                    gpu_ctx->garner_inverse_table,
                    &device_constants);
                upload_ntt_twiddles_to_device(
                    device,
                    twiddle_forward,
                    twiddle_inverse,
                    twiddle_shoup_forward,
                    twiddle_shoup_inverse,
                    &device_constants);
                gpu_ctx->ring_device_constants.push_back(device_constants);
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
                free_ring_device_constants(gpu_ctx->ring_device_constants);
                delete gpu_ctx;
            }
            return set_error(e);
        }
        catch (...)
        {
            if (gpu_ctx)
            {
                free_ring_device_constants(gpu_ctx->ring_device_constants);
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
        // An event failure leaves device readers unbounded in time. Retain the
        // context itself (including NTT tables, streams and registration) rather
        // than let a last-owner drop free resources that they may still access.
        if (ctx->execution->unretired_work.load(std::memory_order_acquire)) return;
        auto owner = ctx->execution;
        int current = 0;
        const bool restore = cudaGetDevice(&current) == cudaSuccess;
        const std::vector<int> gpu_ids = ctx->gpu_ids;
        {
            GpuAllocationActivity activity(owner.get(), -1);
            free_ring_device_constants(ctx->ring_device_constants, owner.get());
            delete ctx;
            for (int device : gpu_ids)
            {
                advance_allocation_revision(context_generations[static_cast<size_t>(device)],
                    device_allocation_activity_unknown[static_cast<size_t>(device)]);
            }
        }
        owner.reset();
        // A finished measurement may own the final Rust parameter reference.
        // Its ordinary context cleanup must preserve the caller's device too.
        if (restore && cudaSetDevice(current) != cudaSuccess)
            gpu_device_mark_allocation_unknown(current);
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
        if (ctx->execution->unretired_work.load(std::memory_order_acquire))
            return set_error("GPU execution has unretired work; resources remain quarantined");
        GpuAllocationActivity activity(ctx->execution.get(), -1);
        const int stream_status = fence_release_streams(ctx->execution.get());
        const int reclaimer_status = wait_pinned_host_reclaimer(ctx->execution.get());
        if (stream_status != 0)
        {
            return stream_status;
        }
        return reclaimer_status;
    }

    int gpu_context_record_releases(const GpuContext *ctx, GpuEventSet **out_events)
    {
        if (!ctx || !ctx->execution || !out_events)
            return set_error("invalid gpu_context_record_releases arguments");
        GpuAllocationActivity activity(ctx->execution.get(), -1);
        *out_events = nullptr;
        if (ctx->execution->memory_release_failed.load(std::memory_order_acquire))
            return set_error("GPU memory release failed; allocations remain charged");
        int current = 0;
        cudaError_t error = cudaGetDevice(&current);
        if (error != cudaSuccess) return set_error(cudaGetErrorString(error));
        GpuEventSet *events = nullptr;
        try
        {
            events = new GpuEventSet{};
            events->execution = ctx->execution;
            events->entries.reserve(ctx->gpu_ids.size());
            if (ctx->execution->release_streams_by_partition.size() != ctx->gpu_ids.size())
                throw std::runtime_error("missing GPU release stream partition");
            for (size_t index = 0; index < ctx->gpu_ids.size(); ++index)
            {
                const cudaStream_t stream = ctx->execution->release_streams_by_partition[index];
                if (!stream) throw std::runtime_error("missing GPU release stream");
                error = cudaSetDevice(ctx->gpu_ids[index]);
                if (error != cudaSuccess) break;
                auto resource = std::make_shared<GpuCudaResource>();
                const int status = resource->acquire(ctx, ctx->gpu_ids[index], GPU_PREPARED_COMPLETION_EVENT);
                if (status != 0) {
                    destroy_event_set(events);
                    cudaSetDevice(current);
                    return status;
                }
                events->entries.push_back({resource->event, ctx->gpu_ids[index], resource});
                error = cudaEventRecord(resource->event, stream);
                if (error != cudaSuccess) break;
            }
        }
        catch (const std::exception &exception)
        {
            destroy_event_set(events);
            cudaSetDevice(current);
            return set_error(exception.what());
        }
        if (error == cudaSuccess) error = cudaSetDevice(current);
        if (error != cudaSuccess)
        {
            destroy_event_set(events);
            cudaSetDevice(current);
            return set_error(cudaGetErrorString(error));
        }
        *out_events = events;
        return 0;
    }

    int gpu_context_query_releases(const GpuContext *ctx, const GpuEventSet *events, int *out_ready)
    {
        if (!ctx || !ctx->execution || !events || !out_ready)
            return set_error("invalid gpu_context_query_releases arguments");
        if (events->execution != ctx->execution)
            return set_error("release completion belongs to another execution owner");
        *out_ready = 0;
        if (ctx->execution->memory_release_failed.load(std::memory_order_acquire))
            return set_error("GPU memory release failed; allocations remain charged");
        int current = 0;
        cudaError_t error = cudaGetDevice(&current);
        if (error != cudaSuccess) return set_error(cudaGetErrorString(error));
        bool ready = true;
        for (const auto &entry : events->entries)
        {
            error = cudaSetDevice(entry.device);
            if (error != cudaSuccess) break;
            error = cudaEventQuery(entry.event);
            if (error == cudaErrorNotReady)
            {
                ready = false;
                error = cudaSuccess;
            }
            if (error != cudaSuccess) break;
        }
        const cudaError_t restored = cudaSetDevice(current);
        if (error == cudaSuccess) error = restored;
        if (error != cudaSuccess)
        {
            gpu_execution_mark_allocation_unknown(ctx->execution.get());
            return set_error(cudaGetErrorString(error));
        }
        *out_ready = ready ? 1 : 0;
        return 0;
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

        GpuAllocationActivity activity(ctx->execution.get(), device);

        std::vector<void *> pointers;
        try
        {
            pointers.reserve(count);
            for (size_t index = 0; index < count; ++index)
            {
                if (ptrs[index])
                {
                    if (pinned_host_pool().mark_deferred(ctx, ptrs[index]) != 0)
                    {
                        ctx->execution->pinned_host_reclaimer->record_uncertain("foreign prepared pinned buffer");
                        return 1;
                    }
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

        auto resource = std::make_shared<GpuCudaResource>();
        const int resource_status = resource->acquire(ctx, device, GPU_PREPARED_COMPLETION_EVENT);
        if (resource_status != 0) {
            ctx->execution->pinned_host_reclaimer->record_uncertain("missing pinned retirement event");
            return resource_status;
        }
        cudaEvent_t completion = resource->event;
        cudaError_t error = cudaSetDevice(device);
        if (error == cudaSuccess) error = cudaEventRecord(completion, stream);
        if (error != cudaSuccess)
        {
            resource->quarantine();
            if (completion)
            {
                // The event may have been recorded before the error was
                // reported. Keep it leaked with the pointers rather than
                // destroying an event that could still be in flight.
                completion = nullptr;
            }
            ctx->execution->pinned_host_reclaimer->record_uncertain(cudaGetErrorString(error));
            return set_error(cudaGetErrorString(error));
        }

        resource->detach_execution();
        const int enqueue_status = ctx->execution->pinned_host_reclaimer->enqueue(
            device, completion, std::move(pointers), resource);
        if (enqueue_status != 0)
        {
            resource->quarantine();
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

    int gpu_context_observe_allocation_epoch(
        const GpuContext *ctx, int device, GpuAllocationEpochBoundary boundary,
        int external_pool_exclusive, GpuAllocationEpochEvidence *out,
        GpuAllocationEpochUnverified *out_reason)
    {
        if (!ctx || !ctx->execution || !out || !out_reason || device < 0 ||
            static_cast<size_t>(device) >= MAX_TRACKED_GPU_DEVICES ||
            (boundary != GPU_ALLOCATION_INITIAL_SETUP && boundary != GPU_ALLOCATION_REFRESH) ||
            (external_pool_exclusive != 0 && external_pool_exclusive != 1))
            return set_error("invalid allocation epoch arguments");
        *out = {};
        *out_reason = GPU_ALLOCATION_UNSUPPORTED_ACTIVITY;
        auto &owner = *ctx->execution;
        const auto found = std::find(owner.gpu_ids.begin(), owner.gpu_ids.end(), device);
        if (found == owner.gpu_ids.end())
            return set_error("allocation epoch device is outside execution owner");
        const size_t partition = static_cast<size_t>(found - owner.gpu_ids.begin());
        const size_t index = static_cast<size_t>(device);
        if (!external_pool_exclusive)
        {
            *out_reason = GPU_ALLOCATION_EXTERNAL_EXCLUSIVITY_REQUIRED;
            return 0;
        }
        if (owner.memory_release_failed.load(std::memory_order_acquire) ||
            owner.unretired_work.load(std::memory_order_acquire))
            return set_error("cannot observe allocation epoch of a quarantined execution");
        // Diagnostics remain available while activity coverage is incomplete.
        // They are independent atomic observations, never a coherent receipt;
        // unsupported observation performs no CUDA query or synchronization.
        out->device = device;
        out->boundary = boundary;
        out->execution_identity = owner.identity;
        out->context_generation = context_generations[index].load(std::memory_order_seq_cst);
        out->owner_revision = owner.allocation_revision.load(std::memory_order_seq_cst);
        out->device_revision = device_allocation_revisions[index].load(std::memory_order_seq_cst);
        if (!owner.allocation_tracking_complete.load(std::memory_order_acquire) ||
            owner.allocation_activity_unknown.load(std::memory_order_seq_cst) ||
            device_allocation_activity_unknown[index].load(std::memory_order_seq_cst))
            return 0;
        if (partition >= owner.compute_streams_by_partition.size() ||
            partition >= owner.release_streams_by_partition.size() ||
            !owner.release_streams_by_partition[partition])
            return set_error("allocation epoch is missing owner streams");
        if (live_context_counts[index].load(std::memory_order_seq_cst) != 1)
        {
            *out_reason = GPU_ALLOCATION_NONEXCLUSIVE_OWNER;
            return 0;
        }
        if (owner.timing_active.load(std::memory_order_acquire))
        {
            *out_reason = GPU_ALLOCATION_PENDING_WORK;
            return 0;
        }
        int current = 0;
        cudaError_t error = cudaGetDevice(&current);
        if (error != cudaSuccess) return set_error(cudaGetErrorString(error));
        auto restore = [&](int status) {
            const cudaError_t restored = cudaSetDevice(current);
            return status != 0 ? status : restored == cudaSuccess
                ? 0 : set_error(cudaGetErrorString(restored));
        };
        error = cudaSetDevice(device);
        if (error != cudaSuccess) return restore(set_error(cudaGetErrorString(error)));
        const auto visit_streams = [&](bool wait, bool all_devices, bool *ready) -> int {
            *ready = true;
            const size_t first_partition = all_devices ? 0 : partition;
            const size_t last_partition = all_devices ? owner.gpu_ids.size() : partition + 1;
            for (size_t stream_partition = first_partition; stream_partition < last_partition;
                 ++stream_partition)
            {
                if (stream_partition >= owner.compute_streams_by_partition.size() ||
                    stream_partition >= owner.release_streams_by_partition.size())
                    return set_error("allocation epoch is missing owner stream partitions");
                cudaError_t selected = cudaSetDevice(owner.gpu_ids[stream_partition]);
                if (selected != cudaSuccess) return set_error(cudaGetErrorString(selected));
                const auto &compute = owner.compute_streams_by_partition[stream_partition];
                for (size_t stream_index = 0; stream_index <= compute.size(); ++stream_index)
                {
                    const cudaStream_t stream = stream_index < compute.size()
                        ? compute[stream_index] : owner.release_streams_by_partition[stream_partition];
                    if (!stream) return set_error("allocation epoch contains a null stream");
                    cudaError_t status;
                    if (wait)
                    {
                        if (stream_partition >= owner.completion_events_by_partition.size() ||
                            stream_index >= owner.completion_events_by_partition[stream_partition].size())
                            return set_error("allocation epoch fence event was not prepared");
                        const cudaEvent_t completion = owner.completion_events_by_partition[stream_partition][stream_index];
                        if (!completion) return set_error("allocation epoch fence event is null");
                        status = cudaEventRecord(completion, stream);
                        if (status == cudaSuccess) status = cudaEventSynchronize(completion);
                    }
                    else status = cudaStreamQuery(stream);
                    if (status == cudaErrorNotReady) *ready = false;
                    else if (status != cudaSuccess)
                    {
                        gpu_execution_mark_allocation_unknown(&owner);
                        gpu_device_mark_allocation_unknown(owner.gpu_ids[stream_partition]);
                        return set_error(cudaGetErrorString(status));
                    }
                }
            }
            const cudaError_t selected = cudaSetDevice(device);
            if (selected != cudaSuccess) return set_error(cudaGetErrorString(selected));
            return 0;
        };
        if (boundary == GPU_ALLOCATION_INITIAL_SETUP)
        {
            // Explicit setup only: fence each owner stream, including ordinary
            // allocations on compute streams, then drain host reclamation. The
            // following snapshot begins after these synchronizations can trim
            // the default pool. No production or refresh caller uses this path.
            bool ready = false;
            int status = visit_streams(false, true, &ready);
            const int idle = owner.pinned_host_reclaimer ? owner.pinned_host_reclaimer->query_idle() : 1;
            if (idle < 0) return restore(set_error("allocation epoch reclamation is uncertain"));
            if (status == 0 && (!ready || idle == 0))
            {
                GpuAllocationActivity activity(&owner, -1);
                status = visit_streams(true, true, &ready);
                if (status == 0) status = wait_pinned_host_reclaimer(&owner);
            }
            if (status != 0) return restore(status);
        }
        GpuAllocationEpochEvidence evidence{};
        evidence.device = device;
        evidence.boundary = boundary;
        evidence.execution_identity = owner.identity;
        evidence.context_generation = context_generations[index].load(std::memory_order_seq_cst);
        evidence.owner_revision = owner.allocation_revision.load(std::memory_order_seq_cst);
        evidence.device_revision = device_allocation_revisions[index].load(std::memory_order_seq_cst);
        if (owner.active_allocation_calls.load(std::memory_order_seq_cst) != 0 ||
            active_device_allocation_calls[index].load(std::memory_order_seq_cst) != 0)
        {
            *out_reason = GPU_ALLOCATION_HOST_ACTIVITY;
            return restore(0);
        }
        bool ready = false;
        int status = visit_streams(false, false, &ready);
        if (status != 0) return restore(status);
        const int reclaimer_idle = owner.pinned_host_reclaimer
            ? owner.pinned_host_reclaimer->query_idle() : 1;
        if (reclaimer_idle < 0)
            return restore(set_error("allocation epoch has uncertain pinned reclamation"));
        if (!ready || reclaimer_idle == 0)
        {
            *out_reason = GPU_ALLOCATION_PENDING_WORK;
            return restore(0);
        }
        error = cudaMemGetInfo(&evidence.free_bytes, &evidence.total_bytes);
        size_t high_water = 0;
        if (error != cudaSuccess) return restore(set_error(cudaGetErrorString(error)));
        status = gpu_default_mempool_get_usage(device, &evidence.pool_used_bytes,
            &high_water, &evidence.pool_reserved_bytes);
        if (status != 0) return restore(status);
        int valid = 0;
        status = gpu_context_validate_allocation_epoch(ctx, &evidence, &valid);
        if (status != 0) return restore(status);
        if (!valid)
        {
            *out_reason = GPU_ALLOCATION_CHANGED;
            return restore(0);
        }
        if (evidence.free_bytes > evidence.total_bytes ||
            evidence.pool_reserved_bytes > evidence.total_bytes - evidence.free_bytes ||
            evidence.pool_used_bytes > evidence.pool_reserved_bytes)
            return restore(set_error("inconsistent allocation epoch counters"));
        const size_t outside_pool = evidence.total_bytes - evidence.free_bytes -
            evidence.pool_reserved_bytes;
        if (evidence.pool_used_bytes > SIZE_MAX - outside_pool)
            return restore(set_error("allocation epoch resident size overflow"));
        evidence.resident_bytes = outside_pool + evidence.pool_used_bytes;
        status = restore(0);
        if (status != 0) return status;
        *out = evidence;
        *out_reason = GPU_ALLOCATION_EPOCH_VERIFIED;
        return 0;
    }

    int gpu_context_validate_allocation_epoch(
        const GpuContext *ctx, const GpuAllocationEpochEvidence *evidence,
        int *out_current)
    {
        if (!ctx || !ctx->execution || !evidence || !out_current || evidence->device < 0 ||
            static_cast<size_t>(evidence->device) >= MAX_TRACKED_GPU_DEVICES)
            return set_error("invalid allocation epoch validation arguments");
        *out_current = 0;
        const auto &owner = *ctx->execution;
        if (std::find(owner.gpu_ids.begin(), owner.gpu_ids.end(), evidence->device) == owner.gpu_ids.end())
            return set_error("allocation epoch device is outside execution owner");
        if (owner.memory_release_failed.load(std::memory_order_acquire) ||
            owner.unretired_work.load(std::memory_order_acquire))
            return set_error("allocation epoch execution is quarantined");
        const size_t index = static_cast<size_t>(evidence->device);
        const int reclaimer_idle = owner.pinned_host_reclaimer
            ? owner.pinned_host_reclaimer->query_idle() : 1;
        if (reclaimer_idle < 0) return set_error("allocation epoch reclamation is uncertain");
        *out_current = evidence->execution_identity == owner.identity &&
            owner.allocation_tracking_complete.load(std::memory_order_acquire) &&
            !owner.allocation_activity_unknown.load(std::memory_order_seq_cst) &&
            !device_allocation_activity_unknown[index].load(std::memory_order_seq_cst) &&
            !owner.timing_active.load(std::memory_order_acquire) && reclaimer_idle == 1 &&
            live_context_counts[index].load(std::memory_order_seq_cst) == 1 &&
            owner.active_allocation_calls.load(std::memory_order_seq_cst) == 0 &&
            active_device_allocation_calls[index].load(std::memory_order_seq_cst) == 0 &&
            owner.allocation_revision.load(std::memory_order_seq_cst) == evidence->owner_revision &&
            device_allocation_revisions[index].load(std::memory_order_seq_cst) == evidence->device_revision &&
            context_generations[index].load(std::memory_order_seq_cst) == evidence->context_generation;
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
            return set_error(cudaGetErrorString(err));
        }
        uint64_t used_current = 0;
        uint64_t used_high = 0;
        uint64_t reserved_current = 0;
        err = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_current);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        err = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_high);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        err = cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved_current);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
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
        GpuAllocationActivity activity(nullptr, device);
        if (live_context_counts[static_cast<size_t>(device)].load(std::memory_order_acquire) != 1)
        {
            return set_error(
                "default mempool high-water reset requires exactly one live mxx context");
        }
        cudaMemPool_t pool = nullptr;
        cudaError_t err = cudaDeviceGetDefaultMemPool(&pool, device);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        uint64_t reset = 0;
        err = cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &reset);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        return 0;
    }

    int gpu_device_get_identity(
        int device,
        char *out_name,
        size_t name_capacity,
        int *out_compute_major,
        int *out_compute_minor,
        size_t *out_total_global_memory)
    {
        if (device < 0 || !out_name || name_capacity == 0 || !out_compute_major ||
            !out_compute_minor || !out_total_global_memory)
        {
            return set_error("invalid gpu_device_get_identity arguments");
        }
        cudaDeviceProp properties{};
        cudaError_t err = cudaGetDeviceProperties(&properties, device);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        const size_t source_length = strnlen(properties.name, sizeof(properties.name));
        const size_t copied = std::min(source_length, name_capacity - 1);
        memcpy(out_name, properties.name, copied);
        out_name[copied] = '\0';
        *out_compute_major = properties.major;
        *out_compute_minor = properties.minor;
        *out_total_global_memory = properties.totalGlobalMem;
        return 0;
    }

    int gpu_event_set_defer_pinned_free(GpuContext *ctx, GpuEventSet *events, void *pointer)
    {
        if (!ctx || !ctx->execution || !events || !pointer)
            return set_error("invalid gpu_event_set_defer_pinned_free arguments");
        if (events->execution != ctx->execution)
            return set_error("pinned free completion belongs to another execution owner");
        if (pinned_host_pool().mark_deferred(ctx, pointer) != 0)
        {
            ctx->execution->pinned_host_reclaimer->record_uncertain("foreign prepared pinned buffer");
            return 1;
        }
        GpuAllocationActivity activity(ctx->execution.get(), -1);
        if (events->entries.size() == 1)
        {
            // A whole-matrix H2D copy already has one completion event. Hand
            // that event to the reclaimer directly, without waiting behind
            // unrelated device allocations on the release stream.
            const auto entry = events->entries.front();
            try
            {
                if (entry.resource) entry.resource->detach_execution();
                const int status = ctx->execution->pinned_host_reclaimer->enqueue(
                    entry.device, entry.event, std::vector<void *>{pointer}, entry.resource);
                if (status == 0) delete events; // The reclaimer now owns the CUDA event.
                return status;
            }
            catch (const std::exception &error)
            {
                ctx->execution->pinned_host_reclaimer->record_uncertain(error.what());
                return set_error(error); // Keep the event and pointer owned on failure.
            }
        }
        // Join the H2D events on an existing release stream. Ownership passes
        // to the existing reclaimer; neither the caller nor a CUDA callback
        // frees pinned memory while a DMA still references it.
        const int device = ctx->execution->gpu_ids.front();
        cudaStream_t stream = ctx->execution->release_streams_by_partition.front();
        cudaError_t error = cudaSetDevice(device);
        for (const auto &entry : events->entries)
        {
            if (error == cudaSuccess)
            {
                error = cudaStreamWaitEvent(stream, entry.event, 0);
            }
        }
        if (error != cudaSuccess)
        {
            ctx->execution->pinned_host_reclaimer->record_uncertain(cudaGetErrorString(error));
            return set_error(cudaGetErrorString(error)); // Keep ownership if completion is uncertain.
        }
        void *pointers[] = {pointer};
        const int status = gpu_defer_pinned_frees(ctx, device, stream, pointers, 1);
        destroy_event_set(events);
        return status;
    }

    int gpu_event_set_wait(GpuEventSet *events)
    {
        if (!events)
        {
            return set_error("invalid gpu_event_set_wait arguments");
        }
        auto owner = events->execution;
        GpuAllocationActivity owner_activity(owner.get(), -1);
        for (const auto &entry : events->entries)
        {
            GpuAllocationActivity activity(nullptr, entry.device);
            cudaError_t err = cudaSetDevice(entry.device);
            if (err != cudaSuccess)
            {
                gpu_execution_mark_allocation_unknown(owner.get());
                gpu_device_mark_allocation_unknown(entry.device);
                return set_error(cudaGetErrorString(err));
            }
            err = cudaEventSynchronize(entry.event);
            if (err != cudaSuccess)
            {
                gpu_execution_mark_allocation_unknown(owner.get());
                gpu_device_mark_allocation_unknown(entry.device);
                return set_error(cudaGetErrorString(err));
            }
        }
        return 0;
    }

    void gpu_event_set_destroy(GpuEventSet *events)
    {
        int current = 0;
        const bool restore = cudaGetDevice(&current) == cudaSuccess;
        destroy_event_set(events);
        if (restore && cudaSetDevice(current) != cudaSuccess)
            gpu_device_mark_allocation_unknown(current);
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
            return set_error(cudaGetErrorString(err));
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
            return set_error(cudaGetErrorString(err));
        }
        err = cudaSetDevice(device);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        err = cudaMemGetInfo(&free_bytes, &total_bytes);
        cudaError_t restore_err = cudaSetDevice(current);
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        if (restore_err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(restore_err));
        }
        *out_free = free_bytes;
        *out_total = total_bytes;
        return 0;
    }

    int gpu_device_synchronize()
    {
        int device = -1;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) return set_error(cudaGetErrorString(err));
        GpuAllocationActivity activity(nullptr, device);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            gpu_device_mark_allocation_unknown(device);
            return set_error(cudaGetErrorString(err));
        }
        return 0;
    }

    int gpu_device_reset()
    {
        int device = -1;
        cudaError_t err = cudaGetDevice(&device);
        if (err != cudaSuccess) return set_error(cudaGetErrorString(err));
        GpuAllocationActivity activity(nullptr, device);
        // Reset invalidates every existing stream/event and cannot preserve a
        // previously registered owner's allocation attribution, even on success.
        gpu_device_mark_allocation_unknown(device);
        err = cudaDeviceReset();
        if (err != cudaSuccess)
        {
            return set_error(cudaGetErrorString(err));
        }
        return 0;
    }

    const char *gpu_last_error()
    {
        return last_error.c_str();
    }

    int gpu_pinned_bind_prepared(void *pointer, GpuPreparedPinnedLease *lease)
    {
        auto &pool = pinned_host_pool();
        std::lock_guard<std::mutex> lock(pool.mutex);
        const auto found = pool.allocated.find(pointer);
        if (!lease || found == pool.allocated.end() || found->second.prepared)
            return set_error("invalid or already claimed prepared pinned backing");
        found->second.prepared = lease;
        return 0;
    }

    void *gpu_pinned_alloc(GpuContext *ctx, size_t bytes, size_t alignment)
    {
        try
        {
            if (bytes == 0)
            {
                return nullptr;
            }
            if (!ctx || !ctx->execution || ctx->gpu_ids.empty() || alignment == 0 ||
                alignment > 256 || (alignment & (alignment - 1)) != 0 ||
                ctx->execution->unretired_work.load(std::memory_order_acquire))
            {
                set_error("invalid pinned allocation context or alignment");
                return nullptr;
            }
            const int device = ctx->gpu_ids.front();
            GpuAllocationActivity activity(ctx->execution.get(), device);
            void *prepared = nullptr;
            int handled = 0;
            if (gpu_prepared_pinned_claim(ctx, bytes, alignment, &prepared, &handled) != 0)
                return nullptr;
            if (handled) return prepared;
            cudaError_t err = cudaSetDevice(device);
            if (err != cudaSuccess)
            {
                set_error(cudaGetErrorString(err));
                return nullptr;
            }
            auto &pool = pinned_host_pool();
            if (void *cached = pool.take(device, bytes)) {
                if (reinterpret_cast<uintptr_t>(cached) % alignment == 0) return cached;
                pool.release(cached, device);
                set_error("cached pinned allocation does not meet alignment");
                return nullptr;
            }
            void *ptr = nullptr;
            err = cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable);
            if (err != cudaSuccess)
            {
                set_error(cudaGetErrorString(err));
                return nullptr;
            }
            if (reinterpret_cast<uintptr_t>(ptr) % alignment != 0) {
                if (cudaFreeHost(ptr) != cudaSuccess) gpu_device_mark_allocation_unknown(device);
                set_error("pinned allocation does not meet alignment");
                return nullptr;
            }
            try
            {
                std::lock_guard<std::mutex> lock(pool.mutex);
                pool.allocated.emplace(ptr, PinnedHostPool::Block{ptr, bytes, device});
            }
            catch (...)
            {
                if (cudaFreeHost(ptr) != cudaSuccess)
                    gpu_device_mark_allocation_unknown(device);
                throw;
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

    int gpu_pinned_free(void *ptr)
    {
        if (!ptr) return 0;
        int device = 0;
        const cudaError_t selected = cudaGetDevice(&device);
        if (selected != cudaSuccess) return set_error(cudaGetErrorString(selected));
        const cudaError_t error = pinned_host_pool().release(ptr, device);
        return error == cudaSuccess ? 0 : set_error(cudaGetErrorString(error));
    }
}
