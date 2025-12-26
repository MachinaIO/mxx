#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuContext GpuContext;
typedef struct GpuPoly GpuPoly;

int gpu_context_create(
    uint32_t logN,
    uint32_t L,
    uint32_t dnum,
    const uint64_t* moduli,
    size_t moduli_len,
    const int* gpu_ids,
    size_t gpu_ids_len,
    uint32_t batch,
    GpuContext** out_ctx
);

void gpu_context_destroy(GpuContext* ctx);

int gpu_context_get_N(const GpuContext* ctx, int* out_N);

int gpu_poly_create(GpuContext* ctx, int level, GpuPoly** out_poly);
void gpu_poly_destroy(GpuPoly* poly);

int gpu_poly_clone(const GpuPoly* src, GpuPoly** out_poly);
int gpu_poly_copy(GpuPoly* dst, const GpuPoly* src);

int gpu_poly_get_level(const GpuPoly* poly, int* out_level);

int gpu_poly_load_rns(GpuPoly* poly, const uint64_t* coeffs_flat, size_t coeffs_len);
int gpu_poly_store_rns(const GpuPoly* poly, uint64_t* coeffs_flat_out, size_t coeffs_len);

int gpu_poly_add(GpuPoly* out, const GpuPoly* a, const GpuPoly* b);
int gpu_poly_sub(GpuPoly* out, const GpuPoly* a, const GpuPoly* b);
int gpu_poly_mul(GpuPoly* out, const GpuPoly* a, const GpuPoly* b);

int gpu_poly_ntt(GpuPoly* poly, int batch);
int gpu_poly_intt(GpuPoly* poly, int batch);

const char* gpu_last_error();

#ifdef __cplusplus
}
#endif
