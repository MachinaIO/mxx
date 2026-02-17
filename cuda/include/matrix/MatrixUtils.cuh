#pragma once

#include "matrix/Matrix.cuh"

bool matrix_aux_slice_for_limb(const GpuMatrix *mat, const dim3 &limb_id, size_t bytes, void **out_ptr);
