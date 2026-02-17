#pragma once

#include <stddef.h>
#include <stdint.h>

#include "Runtime.cuh"

#ifdef __cplusplus
enum class PolyFormat
{
    Coeff,
    Eval,
};

struct GpuPoly
{
    CKKS::RNSPoly *poly;
    GpuContext *ctx;
    int level;
    PolyFormat format;
};
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuPoly GpuPoly;
typedef struct GpuMatrix GpuMatrix;
typedef enum GpuPolyFormat
{
    GPU_POLY_FORMAT_COEFF = 0,
    GPU_POLY_FORMAT_EVAL = 1,
} GpuPolyFormat;

typedef enum GpuMatrixSampleDist
{
    GPU_MATRIX_DIST_UNIFORM = 0,
    GPU_MATRIX_DIST_GAUSS = 1,
    GPU_MATRIX_DIST_BIT = 2,
    GPU_MATRIX_DIST_TERNARY = 3,
} GpuMatrixSampleDist;

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
int gpu_poly_store_compact_bytes(
    GpuPoly* poly,
    uint8_t* payload_out,
    size_t payload_capacity,
    uint16_t* out_max_coeff_bits,
    uint16_t* out_bytes_per_coeff,
    size_t* out_payload_len);
int gpu_poly_load_compact_bytes(
    GpuPoly* poly,
    const uint8_t* payload,
    size_t payload_len,
    uint16_t max_coeff_bits);
int gpu_poly_store_coeffs_words(
    GpuPoly* poly,
    uint64_t* coeff_words_out,
    size_t coeff_words_len,
    size_t words_per_coeff,
    int format);

int gpu_poly_add(GpuPoly* out, const GpuPoly* a, const GpuPoly* b);
int gpu_poly_sub(GpuPoly* out, const GpuPoly* a, const GpuPoly* b);
int gpu_poly_mul(GpuPoly* out, const GpuPoly* a, const GpuPoly* b);
int gpu_poly_equal(const GpuPoly* lhs, const GpuPoly* rhs, int* out_equal);
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

int gpu_matrix_create(GpuContext* ctx, int level, size_t rows, size_t cols, int format, GpuMatrix** out);
void gpu_matrix_destroy(GpuMatrix* mat);
int gpu_matrix_copy(GpuMatrix* dst, const GpuMatrix* src);
int gpu_matrix_entry_clone(
    const GpuMatrix* mat,
    size_t row,
    size_t col,
    GpuPoly** out_poly,
    GpuEventSet** out_events);
int gpu_matrix_copy_entry(GpuMatrix* mat, size_t row, size_t col, const GpuPoly* src);
int gpu_matrix_load_rns_batch(
    GpuMatrix* mat,
    const uint8_t* bytes,
    size_t bytes_per_poly,
    int format,
    GpuEventSet** out_events);
int gpu_matrix_store_rns_batch(
    const GpuMatrix* mat,
    uint8_t* bytes_out,
    size_t bytes_per_poly,
    int format,
    GpuEventSet** out_events);
int gpu_matrix_add(GpuMatrix* out, const GpuMatrix* lhs, const GpuMatrix* rhs);
int gpu_matrix_sub(GpuMatrix* out, const GpuMatrix* lhs, const GpuMatrix* rhs);
int gpu_matrix_mul(GpuMatrix* out, const GpuMatrix* lhs, const GpuMatrix* rhs);
int gpu_matrix_equal(const GpuMatrix* lhs, const GpuMatrix* rhs, int* out_equal);
int gpu_matrix_mul_timed(GpuMatrix* out, const GpuMatrix* lhs, const GpuMatrix* rhs, double* out_kernel_ms);
int gpu_matrix_mul_scalar(GpuMatrix* out, const GpuMatrix* lhs, const GpuPoly* scalar);
int gpu_matrix_copy_block(
    GpuMatrix* out,
    const GpuMatrix* src,
    size_t dst_row,
    size_t dst_col,
    size_t src_row,
    size_t src_col,
    size_t rows,
    size_t cols);
int gpu_matrix_fill_gadget(
    GpuMatrix* out,
    uint32_t base_bits);
int gpu_matrix_decompose_base(const GpuMatrix* src, uint32_t base_bits, GpuMatrix* out);
int gpu_matrix_gauss_samp_gq_arb_base(
    const GpuMatrix* src,
    uint32_t base_bits,
    double c,
    double dgg_stddev,
    uint64_t seed,
    GpuMatrix* out);
int gpu_matrix_sample_p1_full(
    const GpuMatrix* a_mat,
    const GpuMatrix* b_mat,
    const GpuMatrix* d_mat,
    const GpuMatrix* tp2,
    double sigma,
    double s,
    double dgg_stddev,
    uint64_t seed,
    GpuMatrix* out);
int gpu_matrix_sample_distribution(
    GpuMatrix* out,
    int dist_type,
    double sigma,
    uint64_t seed);

int gpu_poly_ntt(GpuPoly* poly, int batch);
int gpu_poly_intt(GpuPoly* poly, int batch);

#ifdef __cplusplus
}
#endif
