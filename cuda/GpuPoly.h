#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuContext GpuContext;
typedef struct GpuPoly GpuPoly;
typedef struct GpuEventSet GpuEventSet;
typedef enum GpuPolyFormat
{
    GPU_POLY_FORMAT_COEFF = 0,
    GPU_POLY_FORMAT_EVAL = 1,
} GpuPolyFormat;

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
int gpu_poly_clone_async(const GpuPoly* src, GpuPoly** out_poly, GpuEventSet** out_events);
int gpu_poly_copy(GpuPoly* dst, const GpuPoly* src);

int gpu_poly_get_level(const GpuPoly* poly, int* out_level);

int gpu_poly_load_rns(
    GpuPoly* poly,
    const uint64_t* rns_flat,
    size_t rns_len,
    int format);
int gpu_poly_store_rns(
    GpuPoly* poly,
    uint64_t* rns_flat_out,
    size_t rns_len,
    int format,
    GpuEventSet** out_events);
int gpu_poly_load_rns_batch(
    GpuPoly* const* polys,
    size_t poly_count,
    const uint8_t* bytes,
    size_t bytes_per_poly,
    int format);
int gpu_poly_store_rns_batch(
    GpuPoly* const* polys,
    size_t poly_count,
    uint8_t* bytes_out,
    size_t bytes_per_poly,
    int format,
    GpuEventSet** out_events);
int gpu_event_set_wait(GpuEventSet* events);
void gpu_event_set_destroy(GpuEventSet* events);

int gpu_poly_add(GpuPoly* out, const GpuPoly* a, const GpuPoly* b);
int gpu_poly_sub(GpuPoly* out, const GpuPoly* a, const GpuPoly* b);
int gpu_poly_mul(GpuPoly* out, const GpuPoly* a, const GpuPoly* b);
int gpu_block_add(GpuPoly* const* out, const GpuPoly* const* lhs, const GpuPoly* const* rhs, size_t count);
int gpu_block_sub(GpuPoly* const* out, const GpuPoly* const* lhs, const GpuPoly* const* rhs, size_t count);
int gpu_block_entrywise_mul(
    GpuPoly* const* out,
    const GpuPoly* const* lhs,
    const GpuPoly* const* rhs,
    size_t count);
int gpu_poly_decompose_base(
    const GpuPoly* src,
    uint32_t base_bits,
    GpuPoly* const* out_polys,
    size_t out_count);

int gpu_poly_ntt(GpuPoly* poly, int batch);
int gpu_poly_intt(GpuPoly* poly, int batch);
int gpu_device_synchronize();
int gpu_device_reset();
int gpu_device_count(int* out_count);
int gpu_device_mem_info(int device, size_t* out_free, size_t* out_total);

const char* gpu_last_error();

void* gpu_pinned_alloc(size_t bytes);
void gpu_pinned_free(void* ptr);

#ifdef __cplusplus
}
#endif
