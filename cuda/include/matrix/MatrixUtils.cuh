#pragma once

#include "matrix/Matrix.cuh"

bool matrix_aux_slice_for_limb(const GpuMatrix *mat, const dim3 &limb_id, size_t bytes, void **out_ptr);
size_t matrix_align_up_size(size_t value, size_t alignment);
int matrix_acquire_aux_workspace(
    const GpuMatrix *aux_owner,
    const dim3 *aux_limb_id,
    size_t bytes,
    void **out_ptr,
    bool *out_shared);
int matrix_release_aux_workspace(void *ptr, bool from_shared);
