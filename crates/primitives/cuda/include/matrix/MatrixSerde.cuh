#pragma once

#include "matrix/Matrix.cuh"
#include "gpu_admission.cuh"
#include "gpu_prepared_plan.cuh"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct GpuPreparedConstCoeffReadback GpuPreparedConstCoeffReadback;

    int gpu_matrix_load_rns_batch(
        GpuMatrix *mat,
        const uint8_t *bytes,
        size_t bytes_per_poly,
        int format,
        GpuEventSet **out_events);

    int gpu_matrix_rns_store_completion_events(const GpuMatrix *mat, size_t *out_count);
    int gpu_matrix_store_rns_batch(
        const GpuMatrix *mat,
        uint8_t *bytes_out,
        size_t bytes_per_poly,
        int format,
        GpuEventSet **out_events);

    int gpu_matrix_store_const_coeff_batch(
        const GpuMatrix *mat,
        uint64_t *words_out,
        size_t words_per_poly,
        GpuEventSet **out_events);

    // Fixed coefficient-domain readback. Preparation captures the matrix
    // geometry, every limb's source descriptor/stream and one completion event
    // per source stream. Submission only replays the already validated 2-D
    // copies into the caller-owned pinned buffer.
    int gpu_matrix_prepare_const_coeff_readback(
        const GpuMatrix *mat,
        uint64_t *words_out,
        size_t words_per_poly,
        size_t coefficient_index,
        size_t coefficient_count,
        const GpuPreparedPlanDescriptor *plan,
        GpuPreparedConstCoeffReadback **out_plan);
    // Submission mutates only the per-submission terminal generations; the
    // recorded event of the newest generation is what a readiness query and the
    // pinned-free retirement both rely on.
    int gpu_matrix_submit_const_coeff_readback(
        GpuPreparedConstCoeffReadback *plan);
    int gpu_matrix_query_const_coeff_readback(
        const GpuPreparedConstCoeffReadback *plan,
        int *out_ready);
    int gpu_matrix_wait_const_coeff_readback(
        const GpuPreparedConstCoeffReadback *plan);
    // Hands the pinned readback destination to the context-owned reclaimer
    // behind fresh terminal events for every stream the plan submits on.
    int gpu_matrix_defer_const_coeff_readback_pinned_free(
        const GpuPreparedConstCoeffReadback *plan,
        void *pointer);
    void gpu_matrix_destroy_const_coeff_readback(
        GpuPreparedConstCoeffReadback *plan);

    typedef struct GpuPreparedRnsUpload GpuPreparedRnsUpload;
    typedef struct GpuPreparedCompactUpload GpuPreparedCompactUpload;
    int gpu_matrix_prepare_rns_upload(
        GpuMatrix *mat,
        const uint8_t *bytes,
        size_t bytes_per_poly,
        int format,
        bool transform_to_eval,
        const GpuPreparedPlanDescriptor *plan,
        GpuPreparedRnsUpload **out_plan);
    int gpu_matrix_submit_rns_upload(GpuPreparedRnsUpload *plan);
    int gpu_matrix_query_rns_upload(
        const GpuPreparedRnsUpload *plan,
        int *out_ready);
    int gpu_matrix_wait_rns_upload(const GpuPreparedRnsUpload *plan);
    // Hands the pinned staging allocation to the context-owned reclaimer behind
    // fresh terminal events for every stream the plan submits on.
    int gpu_matrix_defer_rns_upload_pinned_free(
        const GpuPreparedRnsUpload *plan,
        void *pointer);
    void gpu_matrix_destroy_rns_upload(GpuPreparedRnsUpload *plan);
    int gpu_matrix_prepare_compact_upload(
        GpuMatrix *mat, const uint8_t *payload, size_t payload_capacity,
        uint16_t max_coeff_bits, const GpuPreparedPlanDescriptor *plan,
        GpuPreparedCompactUpload **out_plan);
    int gpu_matrix_submit_compact_upload(
        GpuPreparedCompactUpload *plan, uint16_t max_coeff_bits, size_t payload_len);
    int gpu_matrix_query_compact_upload(
        const GpuPreparedCompactUpload *plan, int *out_ready);
    int gpu_matrix_wait_compact_upload(const GpuPreparedCompactUpload *plan);
    void gpu_matrix_destroy_compact_upload(GpuPreparedCompactUpload *plan);


    int gpu_matrix_store_compact_bytes(
        GpuMatrix *mat,
        uint8_t *payload_out,
        size_t payload_capacity,
        uint16_t *out_max_coeff_bits,
        uint16_t *out_bytes_per_coeff,
        size_t *out_payload_len);

    // Store through the retained matrix owner. Evaluation owners are restored
    // before return, including when the store itself fails.
    int gpu_matrix_store_compact_bytes_borrowed(
        GpuMatrix *mat,
        uint8_t *payload_out,
        size_t payload_capacity,
        uint16_t *out_max_coeff_bits,
        uint16_t *out_bytes_per_coeff,
        size_t *out_payload_len);

    int gpu_matrix_store_compact_bytes_batch(
        GpuMatrix *const *matrices,
        size_t matrix_count,
        uint8_t *const *payload_outputs,
        const size_t *payload_capacities,
        uint16_t *out_max_coeff_bits,
        uint16_t *out_bytes_per_coeff,
        size_t *out_payload_lengths);

    int gpu_matrix_load_compact_bytes(
        GpuMatrix *mat,
        const uint8_t *payload,
        size_t payload_len,
        uint16_t max_coeff_bits);

    int gpu_poly_store_compact_bytes(
        GpuMatrix *poly,
        uint8_t *payload_out,
        size_t payload_capacity,
        uint16_t *out_max_coeff_bits,
        uint16_t *out_bytes_per_coeff,
        size_t *out_payload_len);

    int gpu_poly_load_compact_bytes(
        GpuMatrix *poly,
        const uint8_t *payload,
        size_t payload_len,
        uint16_t max_coeff_bits);

#ifdef __cplusplus
}
#endif

// Exact unpack/pack span for one single-device RNS matrix transfer. This excludes
// the retained matrix, pinned host bytes and completion resources.
extern "C" int gpu_matrix_query_rns_workspace(
    GpuContext *ctx, int level, size_t rows, size_t columns,
    GpuPreparedWorkspaceLayout *out);

// kind: 0 scalar store, 1 homogeneous batch store, 2 scalar load.
// Store capacity uses the codec's modulus-derived signed-width bound; load uses
// the validated artifact width. Streams, events and host buffers are separate.
extern "C" int gpu_matrix_query_compact_workspace(
    GpuContext *ctx, int level, size_t rows, size_t columns, size_t matrices,
    int kind, uint16_t max_coeff_bits, GpuPreparedWorkspaceLayout *out);
