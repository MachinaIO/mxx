// Included by Runtime.cu after its private graph definitions. HIP conditional
// operations are empty DAG nodes with reusable child schedules, never graph
// host callbacks. Only this explicit D2H control boundary waits on device work.
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
        }
    for (auto graph : source_bodies) (void)gpuGraphDestroy(graph);
    if (device_record) (void)gpuFreeAsync(device_record, allocation_stream);

    if (control_complete) (void)gpuEventDestroy(control_complete);
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
    if (compile_scope(root, 0) != 0) return 1;
    gpuError_t error = gpuMallocAsync(reinterpret_cast<void **>(&device_record),
        record_capacity * sizeof(uint64_t), allocation_stream);
    if (error == gpuSuccess)
        error = gpuMallocHost(reinterpret_cast<void **>(&host_record), record_capacity * sizeof(uint64_t));
    if (error == gpuSuccess)
        error = gpuEventCreateWithFlags(&control_complete, gpuEventDisableTiming);
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

int HipGraphSchedule::compile_scope(gpuGraph_t graph, size_t record_base)
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
            error = gpuGraphInstantiateWithFlags(&region.exec, region.graph, 0);
            if (error != gpuSuccess) return set_error(error);
            for (auto entry : clones) nodes.emplace(entry.first, std::make_pair(region.exec, entry.second));
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
        boundary.record_offset = record_base;
        boundary.descriptor_offset = descriptor_nodes.size();
        record_capacity = std::max(record_capacity, record_base + boundary.controls.size());
        descriptor_nodes.insert(descriptor_nodes.end(), boundary.controls.begin(), boundary.controls.end());
        const auto ready = boundary.controls;
        const size_t child_record_base = record_base + ready.size();
        regions.push_back(std::move(boundary));
        for (auto control : ready)
        {
            if (compile_scope(controls.at(control).body, child_record_base) != 0) return 1;
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

gpuError_t HipGraphSchedule::read_controls(const HipGraphRegion &region,
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
    if (error == gpuSuccess) error = gpuEventRecord(control_complete, stream);
    if (error == gpuSuccess) error = gpuEventSynchronize(control_complete);
    control_wait_seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return error;
}

gpuError_t HipGraphSchedule::launch_scope(gpuGraph_t graph, gpuStream_t stream)
{
    for (const auto &region : scopes.at(graph))
    {
        if (region.exec)
        {
            ++region_launches;
            const auto error = gpuGraphLaunch(region.exec, stream);
            if (error != gpuSuccess) return error;
            continue;
        }
        auto error = read_controls(region, false, stream);
        if (error != gpuSuccess) return error;
        while (true)
        {
            bool any = false;
            // Each boundary owns disjoint record slots: nested control can
            // reuse its own slots without overwriting the parent's snapshot.
            for (size_t lane = 0; lane < region.controls.size(); ++lane)
                if (host_record[region.record_offset + lane])
                {
                    any = true;
                    error = launch_scope(controls.at(region.controls[lane]).body, stream);
                    if (error != gpuSuccess) return error;
                }
            if (!any) break;
            error = read_controls(region, true, stream);
            if (error != gpuSuccess) return error;
        }
    }
    return gpuSuccess;
}

gpuError_t HipGraphSchedule::launch(gpuStream_t stream)
{
    control_reads = control_bytes = region_launches = 0;
    control_wait_seconds = host_schedule_seconds = 0;
    const auto started = std::chrono::steady_clock::now();
    auto error = gpuStreamWaitEvent(stream, descriptors_ready, 0);
    if (error == gpuSuccess) error = launch_scope(root, stream);
    host_schedule_seconds = std::max(0.0,
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() - control_wait_seconds);
    return error;
}
