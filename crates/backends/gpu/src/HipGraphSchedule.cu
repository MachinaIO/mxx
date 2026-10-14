// Included by Runtime.cu after its private graph definitions. HIP conditional
// operations are empty DAG nodes with reusable child schedules, never graph
// host callbacks. Only this explicit D2H control boundary waits on device work.

__global__ void mxx_hip_graph_barrier_kernel() {}

// HIP's executable scheduler can deadlock on redundant ancestor dependencies
// and on empty-node branches. Bypass no-op nodes without changing the
// happens-before relation. A many-to-many join uses one GPU barrier instead
// of expanding its incoming/outgoing edges into a quadratic cross product.
// Then remove transitive edges, preserving independent work and every node
// used for replay binding. This runs only on executable regions: conditional
// markers remain in the source scopes used by the host control scheduler.
static int hip_graph_prepare_executable(gpuGraph_t graph)
{
    try
    {
        size_t count = 0;
        auto error = gpuGraphGetNodes(graph, nullptr, &count);
        if (error != gpuSuccess) return set_error(error);
        std::vector<gpuGraphNode_t> nodes(count);
        error = gpuGraphGetNodes(graph, nodes.data(), &count);
        if (error != gpuSuccess) return set_error(error);
        size_t live = count;
        for (auto node : nodes)
        {
            hipGraphNodeType type;
            error = hipGraphNodeGetType(node, &type);
            if (error != gpuSuccess) return set_error(error);
            // Keep a single no-op when the entire region has no device work.
            if (type != hipGraphNodeTypeEmpty || live == 1) continue;
            size_t parent_count = 0, child_count = 0;
            error = hipGraphNodeGetDependencies(node, nullptr, &parent_count);
            if (error == gpuSuccess)
                error = hipGraphNodeGetDependentNodes(node, nullptr, &child_count);
            if (error != gpuSuccess) return set_error(error);
            std::vector<gpuGraphNode_t> parents(parent_count), children(child_count);
            error = hipGraphNodeGetDependencies(node, parents.data(), &parent_count);
            if (error == gpuSuccess)
                error = hipGraphNodeGetDependentNodes(node, children.data(), &child_count);
            if (error != gpuSuccess) return set_error(error);
            if (parent_count > 1 && child_count > 1)
            {
                gpuKernelNodeParams barrier{};
                barrier.func = reinterpret_cast<void *>(mxx_hip_graph_barrier_kernel);
                barrier.gridDim = barrier.blockDim = dim3(1);
                gpuGraphNode_t replacement = nullptr;
                error = gpuGraphAddKernelNode(&replacement, graph, parents.data(), parent_count, &barrier);
                if (error != gpuSuccess) return set_error(error);
                parents.assign(1, replacement);
            }
            else --live;
            for (auto child : children)
            {
                size_t existing_count = 0;
                error = hipGraphNodeGetDependencies(child, nullptr, &existing_count);
                if (error != gpuSuccess) return set_error(error);
                std::vector<gpuGraphNode_t> existing(existing_count);
                error = hipGraphNodeGetDependencies(child, existing.data(), &existing_count);
                if (error != gpuSuccess) return set_error(error);
                for (auto parent : parents)
                {
                    if (std::find(existing.begin(), existing.end(), parent) != existing.end()) continue;
                    error = hipGraphAddDependencies(graph, &parent, &child, 1);
                    if (error != gpuSuccess) return set_error(error);
                }
            }
            error = gpuGraphDestroyNode(node);
            if (error != gpuSuccess) return set_error(error);
        }
        error = gpuGraphGetNodes(graph, nullptr, &count);
        if (error != gpuSuccess) return set_error(error);
        nodes.resize(count);
        error = gpuGraphGetNodes(graph, nodes.data(), &count);
        if (error != gpuSuccess) return set_error(error);
        size_t edge_count = 0;
        error = gpuGraphGetEdges(graph, nullptr, nullptr, &edge_count);
        if (error != gpuSuccess) return set_error(error);
        std::vector<gpuGraphNode_t> from(edge_count), to(edge_count);
        error = gpuGraphGetEdges(graph, from.data(), to.data(), &edge_count);
        if (error != gpuSuccess) return set_error(error);
        std::map<gpuGraphNode_t, size_t> indices;
        for (size_t index = 0; index < count; ++index) indices.emplace(nodes[index], index);
        std::vector<std::vector<size_t>> parents(count), children(count);
        for (size_t edge = 0; edge < edge_count; ++edge)
        {
            const auto parent = indices.at(from[edge]), child = indices.at(to[edge]);
            parents[child].push_back(parent);
            children[parent].push_back(child);
        }
        std::vector<size_t> remaining(count), order, rank(count);
        order.reserve(count);
        for (size_t index = 0; index < count; ++index)
        {
            remaining[index] = parents[index].size();
            if (!remaining[index]) order.push_back(index);
        }
        for (size_t position = 0; position < order.size(); ++position)
        {
            const auto node = order[position];
            rank[node] = position;
            for (auto child : children[node])
                if (--remaining[child] == 0) order.push_back(child);
        }
        if (order.size() != count) return set_error("cyclic HIP graph");
        std::vector<size_t> visited(count, count), pending;
        std::vector<gpuGraphNode_t> redundant_from, redundant_to;
        for (size_t node = 0; node < count; ++node)
        {
            auto &predecessors = parents[node];
            if (predecessors.size() < 2) continue;
            std::sort(predecessors.begin(), predecessors.end(),
                [&](size_t left, size_t right) { return rank[left] > rank[right]; });
            const size_t earliest = rank[predecessors.back()];
            for (auto parent : predecessors)
            {
                if (visited[parent] == node)
                {
                    redundant_from.push_back(nodes[parent]);
                    redundant_to.push_back(nodes[node]);
                    continue;
                }
                visited[parent] = node;
                pending.push_back(parent);
                while (!pending.empty())
                {
                    const auto current = pending.back();
                    pending.pop_back();
                    for (auto ancestor : parents[current])
                    {
                        // Earlier ancestors cannot be another direct parent.
                        if (rank[ancestor] < earliest || visited[ancestor] == node) continue;
                        visited[ancestor] = node;
                        pending.push_back(ancestor);
                    }
                }
            }
        }
        if (!redundant_from.empty())
        {
            error = hipGraphRemoveDependencies(graph, redundant_from.data(),
                redundant_to.data(), redundant_from.size());
            if (error != gpuSuccess) return set_error(error);
        }
        return 0;
    }
    catch (const std::exception &error) { return set_error(error); }
}

__global__ void mxx_hip_control_kernel(const HipControlDescriptor *descriptors,
    uint64_t *records, size_t count, bool advance)
{
    const size_t lane = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    const auto control = descriptors[lane];
    if (advance && !records[lane]) return;
    if (control.predicate)
    {
        records[lane] = advance ? 0 : *control.predicate != 0;
        return;
    }
    if (advance) ++*control.index;
    if (*control.limit > control.maximum)
    {
        atomicCAS(control.status, 0U, 4U);
        records[lane] = 0;
        return;
    }
    records[lane] = *control.index < *control.limit && *control.status == 0U;
}

static int hip_graph_begin_control(MxxGpuGraphBuilder *builder,
    const uint64_t *predicate, uint32_t predicate_binding,
    uint64_t *index, const uint64_t *limit, uint64_t maximum,
    uint32_t *status, uint32_t index_binding, uint32_t limit_binding,
    uint32_t status_binding)
{
    if (!builder || !builder->operation_active ||
        (builder->conditional_body_active && !builder->generic_body_mode) ||
        (!predicate && (!index || !limit || !maximum || !status)))
        return set_error("invalid HIP control body");
    HipGraphControl control;
    control.predicate = predicate;
    control.index = index;
    control.limit = limit;
    control.maximum = maximum;
    control.status = status;
    const auto patch = [&](uint32_t field, uint32_t binding, const void *address) {
        auto value = mxx_direct_pointer_patch(field, binding);
        if (normalize_builder_patch(builder, &value, reinterpret_cast<uint64_t>(address)) != 0)
            return 1;
        control.patches.push_back(value);
        return 0;
    };
    if (predicate)
    {
        if (patch(0, predicate_binding, predicate) != 0) return 1;
    }
    else if (patch(1, index_binding, index) || patch(2, limit_binding, limit) ||
        patch(4, status_binding, status)) return 1;
    MxxGpuGraphBuilder::ConditionalFrame frame;
    frame.parent_graph = builder->graph;
    gpuError_t error = gpuGraphAddEmptyNode(&frame.conditional_node, builder->graph,
        builder->frontier.data(), builder->frontier.size());
    if (error != gpuSuccess) return set_error(error);
    error = gpuGraphCreate(&control.body, 0);
    if (error != gpuSuccess) return set_error(error);
    builder->hip_body_graphs.push_back(control.body);
    frame.parent_operation_index = builder->operation_index;
    frame.parent_operation_nodes = std::move(builder->operation_nodes);
    frame.parent_binding_map = std::move(builder->binding_map);
    frame.parent_body_terminals = std::move(builder->body_terminals);
    frame.parent_conditional_body_active = builder->conditional_body_active;
    frame.parent_generic_body_mode = builder->generic_body_mode;
    frame.parent_body_device = builder->body_device;
    frame.parent_last_body_conditional = builder->last_body_conditional;
    builder->hip_controls.emplace(frame.conditional_node, control);
    builder->conditional_frames.push_back(std::move(frame));
    builder->graph = control.body;
    builder->frontier.clear();
    builder->operation_nodes.clear();
    builder->binding_map.clear();
    builder->body_terminals.clear();
    builder->last_body_conditional = nullptr;
    builder->operation_active = false;
    builder->conditional_body_active = true;
    builder->generic_body_mode = true;
    error = gpuGetDevice(&builder->body_device);
    return error == gpuSuccess ? 0 : set_error(error);
}

static int hip_graph_finish_control(MxxGpuGraphBuilder *builder)
{
    if (!builder || !builder->generic_body_mode || !builder->conditional_body_active ||
        builder->operation_active || builder->conditional_frames.empty())
        return set_error("invalid HIP control body completion");
    auto frame = std::move(builder->conditional_frames.back());
    builder->conditional_frames.pop_back();
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
    builder->last_body_conditional = frame.parent_last_body_conditional;
    return 0;
}

HipGraphSchedule::~HipGraphSchedule()
{
    (void)mxx_set_device(device);
    for (auto &scope : scopes)
        for (auto &region : scope.second)
        {
            if (region.exec) (void)gpuGraphExecDestroy(region.exec);
            if (region.graph) (void)gpuGraphDestroy(region.graph);
            if (region.control_complete) (void)gpuEventDestroy(region.control_complete);
        }
    for (auto graph : source_bodies) (void)gpuGraphDestroy(graph);
    if (device_record) (void)gpuFreeAsync(device_record, allocation_stream);

    for (const auto &entry : scope_complete) (void)gpuEventDestroy(entry.second);
    if (device_descriptors) (void)gpuFreeAsync(device_descriptors, allocation_stream);

    if (descriptors_ready) (void)gpuEventDestroy(descriptors_ready);
    // Even an unsuccessful bind may have queued a descriptor upload. Retire
    // its pinned owners behind the allocation stream rather than waiting in
    // this non-D2H destructor or freeing pending transfer operands.
    std::vector<void *> pinned;
    if (host_record) pinned.push_back(host_record);
    if (host_descriptors) pinned.push_back(host_descriptors);
    if (!pinned.empty())
    {
        gpuEvent_t retired = nullptr;
        auto error = gpuEventCreateWithFlags(&retired, gpuEventDisableTiming);
        if (error == gpuSuccess) error = gpuEventRecord(retired, allocation_stream);
        if (error != gpuSuccess || !execution_owner || !execution_owner->pinned_host_reclaimer ||
            execution_owner->pinned_host_reclaimer->enqueue(device, retired, std::move(pinned)) != 0)
        {
            // Fail closed: unknown completion retains pointers and event.
            if (execution_owner && execution_owner->pinned_host_reclaimer)
                execution_owner->pinned_host_reclaimer->record_uncertain("HIP control owner retirement failed");
        }
    }
}

int HipGraphSchedule::compile(MxxGpuGraphBuilder *builder)
{
    device = builder->device;
    execution_owner = builder->context->execution;
    allocation_stream = builder->stream;
    root = builder->root_graph;
    controls = builder->hip_controls;
    if (compile_scope(root) != 0) return 1;
    gpuError_t error = gpuMallocAsync(reinterpret_cast<void **>(&device_record),
        record_capacity * sizeof(uint64_t), allocation_stream);
    if (error == gpuSuccess)
        error = gpuMallocHost(reinterpret_cast<void **>(&host_record), record_capacity * sizeof(uint64_t));
    if (error == gpuSuccess)
        error = gpuMallocAsync(reinterpret_cast<void **>(&device_descriptors),
            descriptor_nodes.size() * sizeof(HipControlDescriptor), allocation_stream);
    if (error == gpuSuccess)
        error = gpuMallocHost(reinterpret_cast<void **>(&host_descriptors),
            descriptor_nodes.size() * sizeof(HipControlDescriptor));
    if (error == gpuSuccess)
        error = gpuEventCreateWithFlags(&descriptors_ready, gpuEventDisableTiming);
    if (error != gpuSuccess) return set_error(error);
    source_bodies = std::move(builder->hip_body_graphs);
    builder->hip_body_graphs.clear();
    return 0;
}

int HipGraphSchedule::compile_scope(gpuGraph_t graph)
{
    size_t count = 0;
    gpuError_t error = gpuGraphGetNodes(graph, nullptr, &count);
    if (error != gpuSuccess) return set_error(error);
    std::vector<gpuGraphNode_t> all(count);
    error = gpuGraphGetNodes(graph, all.data(), &count);
    if (error != gpuSuccess) return set_error(error);
    size_t edge_count = 0;
    error = gpuGraphGetEdges(graph, nullptr, nullptr, &edge_count);
    if (error != gpuSuccess) return set_error(error);
    std::vector<gpuGraphNode_t> from(edge_count), to(edge_count);
    error = gpuGraphGetEdges(graph, from.data(), to.data(), &edge_count);
    if (error != gpuSuccess) return set_error(error);
    std::map<gpuGraphNode_t, size_t> remaining;
    std::map<gpuGraphNode_t, std::vector<gpuGraphNode_t>> followers;
    for (auto node : all) remaining[node] = 0;
    for (size_t edge = 0; edge < edge_count; ++edge)
    {
        ++remaining[to[edge]];
        followers[from[edge]].push_back(to[edge]);
    }
    auto &regions = scopes[graph];
    error = gpuEventCreateWithFlags(&scope_complete[graph], gpuEventDisableTiming);
    if (error != gpuSuccess) return set_error(error);
    // Consume every ready ordinary node before a control boundary. Independent
    // work keeps graph parallelism within each region. Removed cross-region
    // dependencies are supplied by the launch stream's submission order.
    const auto consume = [&](gpuGraphNode_t node) {
        remaining.erase(node);
        for (auto next : followers[node]) --remaining.at(next);
    };
    while (!remaining.empty())
    {
        std::set<gpuGraphNode_t> segment;
        bool progress = true;
        while (progress)
        {
            progress = false;
            for (auto it = remaining.begin(); it != remaining.end();)
            {
                auto node = it->first;
                if (it->second == 0 && controls.count(node) == 0)
                {
                    segment.insert(node);
                    ++it;
                    consume(node);
                    progress = true;
                }
                else ++it;
            }
        }
        if (!segment.empty())
        {
            regions.emplace_back();
            auto &region = regions.back();
            error = gpuGraphClone(&region.graph, graph);
            if (error != gpuSuccess) return set_error(error);
            std::map<gpuGraphNode_t, gpuGraphNode_t> clones;
            for (auto original : all)
            {
                gpuGraphNode_t clone = nullptr;
                error = gpuGraphNodeFindInClone(&clone, original, region.graph);
                if (error != gpuSuccess) return set_error(error);
                if (segment.count(original)) clones.emplace(original, clone);
                else
                {
                    error = gpuGraphDestroyNode(clone);
                    if (error != gpuSuccess) return set_error(error);
                }
            }
            if (hip_graph_prepare_executable(region.graph) != 0) return 1;
            error = gpuGraphInstantiateWithFlags(&region.exec, region.graph, 0);
            if (error != gpuSuccess) return set_error(error);
            for (auto entry : clones)
            {
                hipGraphNodeType type;
                error = hipGraphNodeGetType(entry.first, &type);
                if (error != gpuSuccess) return set_error(error);
                if (type != hipGraphNodeTypeEmpty)
                    nodes.emplace(entry.first, std::make_pair(region.exec, entry.second));
            }
        }
        auto selected = std::find_if(remaining.begin(), remaining.end(),
            [&](const auto &entry) { return entry.second == 0 && controls.count(entry.first); });
        if (selected == remaining.end())
        {
            if (!remaining.empty()) return set_error("cyclic HIP region graph");
            break;
        }
        HipGraphRegion boundary;
        // All currently ready control nodes are independent lanes. Their GPU
        // records are copied in one bounded D2H and observed by one event.
        for (const auto &entry : remaining)
            if (entry.second == 0 && controls.count(entry.first))
                boundary.controls.push_back(entry.first);
        boundary.record_offset = record_capacity;
        boundary.descriptor_offset = descriptor_nodes.size();
        record_capacity += boundary.controls.size();
        error = gpuEventCreateWithFlags(&boundary.control_complete, gpuEventDisableTiming);
        if (error != gpuSuccess) return set_error(error);
        descriptor_nodes.insert(descriptor_nodes.end(), boundary.controls.begin(), boundary.controls.end());
        const auto ready = boundary.controls;
        regions.push_back(std::move(boundary));
        for (auto control : ready)
        {
            if (compile_scope(controls.at(control).body) != 0) return 1;
            consume(control);
        }
    }
    return 0;
}

int HipGraphSchedule::bind(const MxxGraphBindingValue *values, size_t count)
{
    for (auto &entry : controls)
        for (const auto &patch : entry.second.patches)
        {
            uint64_t address = 0;
            if (validate_patch_value(patch, values, count, reinterpret_cast<uint8_t *>(&address)) != 0)
                return 1;
            auto &control = entry.second;
            switch (patch.argument_index)
            {
                case 0: control.predicate = reinterpret_cast<const uint64_t *>(address); break;
                case 1: control.index = reinterpret_cast<uint64_t *>(address); break;
                case 2: control.limit = reinterpret_cast<const uint64_t *>(address); break;
                case 4: control.status = reinterpret_cast<uint32_t *>(address); break;
                default: return set_error("invalid HIP control binding");
            }
        }
    // Bind is called only after the previous execute has joined. The pinned
    // descriptor table remains owned until its async upload and all consumers
    // complete; this is one table refresh per bind, never per loop iteration.
    for (size_t lane = 0; lane < descriptor_nodes.size(); ++lane)
    {
        const auto &control = controls.at(descriptor_nodes[lane]);
        host_descriptors[lane] = HipControlDescriptor{control.predicate, control.index,
            control.limit, control.status, control.maximum};
    }
    auto error = gpuMemcpyAsync(device_descriptors, host_descriptors,
        descriptor_nodes.size() * sizeof(HipControlDescriptor), gpuMemcpyHostToDevice, allocation_stream);
    if (error == gpuSuccess) error = gpuEventRecord(descriptors_ready, allocation_stream);
    if (error != gpuSuccess) return set_error(error);
    return 0;
}

gpuGraphExec_t HipGraphSchedule::executable(gpuGraphNode_t original) const
{
    const auto found = nodes.find(original);
    return found == nodes.end() ? nullptr : found->second.first;
}

gpuGraphNode_t HipGraphSchedule::node(gpuGraphNode_t original) const
{
    const auto found = nodes.find(original);
    return found == nodes.end() ? nullptr : found->second.second;
}

gpuError_t HipGraphSchedule::upload(gpuStream_t stream)
{
    for (auto &scope : scopes)
        for (auto &region : scope.second)
            if (region.exec)
            {
                const auto error = gpuGraphUpload(region.exec, stream);
                if (error != gpuSuccess) return error;
            }
    return gpuSuccess;
}

gpuError_t HipGraphSchedule::submit_controls(const HipGraphRegion &region,
    bool advance, gpuStream_t stream)
{
    const auto started = std::chrono::steady_clock::now();
    ++control_reads;
    const size_t count = region.controls.size();
    control_bytes += count * sizeof(uint64_t);
    const auto *descriptors = device_descriptors + region.descriptor_offset;
    auto *records = device_record + region.record_offset;
    void *arguments[] = {&descriptors, &records, const_cast<size_t *>(&count), &advance};
    auto error = gpuLaunchKernel(reinterpret_cast<const void *>(mxx_hip_control_kernel),
        dim3(static_cast<unsigned>((count + 127) / 128)), dim3(128), arguments, 0, stream);
    if (error == gpuSuccess)
        error = gpuMemcpyAsync(host_record + region.record_offset, records,
            count * sizeof(uint64_t), gpuMemcpyDeviceToHost, stream);
    if (error == gpuSuccess) error = gpuEventRecord(region.control_complete, stream);
    control_wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return error;
}

// Submit all ready scopes before awaiting a control transfer. Each active
// boundary has disjoint records and an event. Body queues come from the
// context's configured compute pool and remain alive with that owner.
// A completed body's event joins its parent before the next predicate check.
gpuError_t HipGraphSchedule::launch(gpuStream_t root_stream)
{
    control_reads = control_bytes = region_launches = 0;
    control_wait_seconds = host_schedule_seconds = 0;
    const auto started = std::chrono::steady_clock::now();
    struct ScopeRun
    {
        gpuGraph_t graph = nullptr;
        gpuStream_t stream = nullptr;
        ScopeRun *parent = nullptr;
        const HipGraphRegion *boundary = nullptr;
        size_t region = 0;
        size_t children = 0;
        bool active = false;
    };
    std::map<gpuGraph_t, ScopeRun> runs;
    std::deque<ScopeRun *> ready, pending;
    const auto found = std::find(execution_owner->gpu_ids.begin(), execution_owner->gpu_ids.end(), device);
    if (found == execution_owner->gpu_ids.end()) return gpuErrorInvalidDevice;
    const auto partition = static_cast<size_t>(found - execution_owner->gpu_ids.begin());
    std::vector<gpuStream_t> body_streams;
    for (auto stream : execution_owner->compute_streams_by_partition[partition])
        if (stream != root_stream) body_streams.push_back(stream);
    // A pool with one stream is an explicitly configured concurrency limit.
    if (body_streams.empty()) body_streams.push_back(root_stream);
    size_t next_stream = 0;
    const auto begin = [&](gpuGraph_t graph, ScopeRun *parent) {
        auto &run = runs[graph];
        if (run.active) return gpuErrorInvalidValue;
        run = ScopeRun{};
        run.graph = graph;
        run.parent = parent;
        run.stream = parent ? body_streams[next_stream++ % body_streams.size()] : root_stream;
        run.active = true;
        ready.push_back(&run);
        return gpuSuccess;
    };
    auto error = gpuStreamWaitEvent(root_stream, descriptors_ready, 0);
    if (error == gpuSuccess) error = begin(root, nullptr);
    while (error == gpuSuccess && (!ready.empty() || !pending.empty()))
    {
        while (!ready.empty() && error == gpuSuccess)
        {
            auto *run = ready.front();
            ready.pop_front();
            const auto &regions = scopes.at(run->graph);
            while (run->region < regions.size())
            {
                const auto &region = regions[run->region++];
                if (!region.exec)
                {
                    run->boundary = &region;
                    error = submit_controls(region, false, run->stream);
                    if (error == gpuSuccess) pending.push_back(run);
                    break;
                }
                ++region_launches;
                error = gpuGraphLaunch(region.exec, run->stream);
                if (error != gpuSuccess) break;
            }
            if (error != gpuSuccess) break;
            if (run->boundary) continue;
            error = gpuEventRecord(scope_complete.at(run->graph), run->stream);
            if (error != gpuSuccess) break;
            run->active = false;
            if (run->parent)
            {
                auto *parent = run->parent;
                error = gpuStreamWaitEvent(parent->stream, scope_complete.at(run->graph), 0);
                if (error != gpuSuccess) break;
                if (--parent->children == 0)
                {
                    error = submit_controls(*parent->boundary, true, parent->stream);
                    if (error == gpuSuccess) pending.push_back(parent);
                }
            }
        }
        if (error != gpuSuccess || pending.empty()) continue;
        // Drain any already-ready D2H first; if none is ready, wait on one
        // recorded transfer event instead of busy-polling the device.
        auto selected = pending.begin();
        for (auto candidate = pending.begin(); candidate != pending.end(); ++candidate)
        {
            const auto status = gpuEventQuery((*candidate)->boundary->control_complete);
            if (status == gpuSuccess) { selected = candidate; break; }
            if (status != gpuErrorNotReady) { error = status; break; }
        }
        if (error != gpuSuccess) break;
        auto *run = *selected;
        pending.erase(selected);
        const auto waited = std::chrono::steady_clock::now();
        error = gpuEventSynchronize(run->boundary->control_complete);
        control_wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - waited).count();
        if (error != gpuSuccess) break;
        const auto &boundary = *run->boundary;
        for (size_t lane = 0; lane < boundary.controls.size(); ++lane)
            if (host_record[boundary.record_offset + lane]) ++run->children;
        if (!run->children)
        {
            run->boundary = nullptr;
            ready.push_back(run);
            continue;
        }
        for (size_t lane = 0; lane < boundary.controls.size(); ++lane)
            if (host_record[boundary.record_offset + lane])
            {
                error = begin(controls.at(boundary.controls[lane]).body, run);
                if (error != gpuSuccess) break;
            }
    }
    host_schedule_seconds = std::max(0.0,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() - control_wait_seconds);
    return error;
}
