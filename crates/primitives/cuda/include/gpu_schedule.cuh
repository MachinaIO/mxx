#pragma once
#include <stddef.h>
#include <stdint.h>

// A plan's native launch streams, not a scheduler-selected surrogate stream.
struct GpuPreparedSchedule;
struct GpuPreparedPlanRef {
    uint32_t kind;
    const void *plan;
};
extern "C" {
int gpu_prepared_schedule_create(const GpuPreparedPlanRef *plans, size_t count,
    GpuPreparedSchedule *const *dependencies, size_t dependency_count,
    GpuPreparedSchedule **out);
size_t gpu_prepared_schedule_stream_count(const GpuPreparedSchedule *schedule);
const void *gpu_prepared_schedule_stream_context(const GpuPreparedSchedule *schedule, size_t index);
int gpu_prepared_schedule_bind_dependencies(GpuPreparedSchedule *schedule,
    const GpuPreparedSchedule *const *dependencies, size_t count);
int gpu_prepared_schedule_provision(GpuPreparedSchedule *schedule);
int gpu_prepared_schedule_begin(const GpuPreparedSchedule *schedule);
int gpu_prepared_schedule_end(const GpuPreparedSchedule *schedule);
int gpu_prepared_schedule_is_ready(const GpuPreparedSchedule *schedule, bool *ready);
void gpu_prepared_schedule_destroy(GpuPreparedSchedule *schedule);
}
