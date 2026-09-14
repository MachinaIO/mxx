#pragma once

struct GpuMatrix;
struct GpuContext;

// Unit-test stream gate. No production Rust wrapper references these symbols.
extern "C" void *gpu_test_matrix_stream_gate(GpuMatrix *matrix);
extern "C" void *gpu_test_context_stream_gate(GpuContext *context);
extern "C" void *gpu_test_context_streams_gate(GpuContext *context);
extern "C" void gpu_test_release_stream_gate(void *gate);
