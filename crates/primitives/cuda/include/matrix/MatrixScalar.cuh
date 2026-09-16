#pragma once
#include "matrix/Matrix.cuh"

struct GpuPreparedThreshold;
struct GpuPreparedScalarPack;
struct GpuPreparedScalarBuffer;
struct GpuPreparedScalarOp;
struct GpuPreparedScalarMatrixSelect;
struct GpuPreparedScalarRef {
    const GpuPreparedScalarBuffer *owner;
    size_t index;
};

extern "C" {
int gpu_matrix_prepare_scalar_buffer(const GpuMatrix *anchor, size_t count, size_t words,
    uint64_t *host, GpuPreparedScalarBuffer **out);
int gpu_matrix_upload_scalar_buffer(const GpuPreparedScalarBuffer *buffer);
int gpu_matrix_wait_scalar_buffer(const GpuPreparedScalarBuffer *buffer);
int gpu_matrix_read_scalar_buffer(const GpuPreparedScalarBuffer *buffer);
void gpu_matrix_destroy_scalar_buffer(GpuPreparedScalarBuffer *buffer);
size_t gpu_matrix_scalar_op_workspace_bytes(size_t left_words, size_t right_words, size_t output_words, size_t candidate_count);
int gpu_matrix_prepare_scalar_op(int opcode, GpuPreparedScalarRef left, GpuPreparedScalarRef right,
    GpuPreparedScalarRef output, size_t bit, const GpuPreparedScalarRef *candidates, size_t candidate_count, GpuPreparedScalarOp **out);
int gpu_matrix_submit_scalar_op(const GpuPreparedScalarOp *plan);
void gpu_matrix_destroy_scalar_op(GpuPreparedScalarOp *plan);
size_t gpu_matrix_scalar_matrix_select_workspace_bytes(size_t count);
int gpu_matrix_prepare_scalar_matrix_select(GpuMatrix *output, GpuPreparedScalarRef selector,
    const GpuMatrix *const *sources, const GpuMatrixBatchView *views, size_t count, GpuPreparedScalarMatrixSelect **out);
int gpu_matrix_submit_scalar_matrix_select(const GpuPreparedScalarMatrixSelect *plan);
void gpu_matrix_destroy_scalar_matrix_select(GpuPreparedScalarMatrixSelect *plan);
int gpu_matrix_threshold_workspace_bytes(const GpuMatrix *source, size_t count,
    size_t plaintext_words, size_t *bytes);
int gpu_matrix_prepare_threshold(const GpuMatrix *source, size_t count,
    const uint64_t *plaintext, size_t plaintext_words, bool output_bool,
    GpuPreparedScalarBuffer *output, GpuPreparedThreshold **out);
int gpu_matrix_submit_threshold(const GpuPreparedThreshold *plan);
void gpu_matrix_destroy_threshold(GpuPreparedThreshold *plan);
size_t gpu_matrix_scalar_pack_workspace_bytes(size_t count);
int gpu_matrix_prepare_scalar_pack(GpuMatrix *output, const GpuPreparedScalarRef *values,
    size_t count, size_t coefficient_bits, GpuPreparedScalarPack **out);
int gpu_matrix_submit_scalar_pack(const GpuPreparedScalarPack *plan);
void gpu_matrix_destroy_scalar_pack(GpuPreparedScalarPack *plan);
}
