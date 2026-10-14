struct HipControlDescriptor
{
    const uint64_t *predicate = nullptr;
    uint64_t *index = nullptr;
    const uint64_t *limit = nullptr;
    uint32_t *status = nullptr;
    uint64_t maximum = 0;
};

// Private HIP schedule declarations. Included within Runtime.cu's C ABI scope;
// implementations are in HipGraphSchedule.cu, in the same translation unit.
struct HipGraphControl
{
    int device = -1;
    std::shared_ptr<GpuExecutionOwner> execution;
    gpuGraph_t body = nullptr;
    const uint64_t *predicate = nullptr;
    uint64_t *index = nullptr;
    const uint64_t *limit = nullptr;
    uint32_t *status = nullptr;
    uint64_t maximum = 0;
    std::vector<MxxGraphPatch> patches;
};

struct HipControlBatch
{
    int device = -1;
    std::vector<gpuGraphNode_t> controls;
    size_t record_offset = 0;
    gpuEvent_t control_complete = nullptr;
};

struct HipGraphRegion
{
    gpuGraph_t graph = nullptr;
    gpuGraphExec_t exec = nullptr;
    std::vector<HipControlBatch> batches;
};

struct HipGraphScope
{
    int device = -1;
    std::shared_ptr<GpuExecutionOwner> execution;
    gpuEvent_t complete = nullptr;
    std::vector<HipGraphRegion> regions;
};

// One upload per bind and one contiguous record/descriptor allocation per
// control device. No control kernel dereferences another GPU's metadata.
struct HipControlStorage
{
    std::shared_ptr<GpuExecutionOwner> execution;
    gpuStream_t allocation_stream = nullptr;
    HipControlDescriptor *host_descriptors = nullptr;
    HipControlDescriptor *device_descriptors = nullptr;
    uint64_t *device_record = nullptr;
    uint64_t *host_record = nullptr;
    gpuEvent_t descriptors_ready = nullptr;
    std::vector<gpuGraphNode_t> descriptor_nodes;
};

struct HipGraphSchedule
{
    uint64_t control_reads = 0;
    uint64_t control_bytes = 0;
    uint64_t region_launches = 0;
    double control_wait_seconds = 0;
    double host_schedule_seconds = 0;
    std::shared_ptr<GpuExecutionOwner> execution_owner;
    int device = -1;
    std::map<int, HipControlStorage> storage;
    std::map<gpuGraph_t, HipGraphScope> scopes;
    std::map<gpuGraphNode_t, HipGraphControl> controls;
    std::map<gpuGraphNode_t, std::pair<gpuGraphExec_t, gpuGraphNode_t>> nodes;
    std::vector<gpuGraph_t> source_bodies;
    gpuGraph_t root = nullptr;
    ~HipGraphSchedule();
    int compile(MxxGpuGraphBuilder *builder);
    int compile_scope(gpuGraph_t graph, int scope_device,
        const std::shared_ptr<GpuExecutionOwner> &execution);
    int bind(const MxxGraphBindingValue *values, size_t count);
    gpuGraphExec_t executable(gpuGraphNode_t original) const;
    gpuGraphNode_t node(gpuGraphNode_t original) const;
    gpuError_t upload(gpuStream_t stream);
    gpuError_t launch(gpuStream_t stream);
    gpuError_t drain();
    gpuError_t submit_controls(const HipControlBatch &batch, bool advance, gpuStream_t stream);
};
