use super::{CircuitBenchEstimate, CircuitBenchSummary, scale_independent_summary};

fn estimate_summary(estimate: CircuitBenchEstimate) -> CircuitBenchSummary {
    CircuitBenchSummary::from_nanos(estimate.total_time, estimate.latency, estimate.max_parallelism)
        .with_peak_vram({
            #[cfg(feature = "gpu")]
            {
                estimate.peak_vram
            }
            #[cfg(not(feature = "gpu"))]
            {
                0
            }
        })
}

/// Size-aware benchmark model for native polynomial-vector arithmetic.
pub trait NativePolyMatrixBenchEstimator {
    fn estimate_poly_vector_mul(&self, vector_len: usize) -> CircuitBenchEstimate;

    fn estimate_vector_inner_product(&self, vector_len: usize) -> CircuitBenchEstimate;

    fn estimate_vector_add(&self, vector_len: usize) -> CircuitBenchEstimate;

    fn estimate_vector_sub(&self, vector_len: usize) -> CircuitBenchEstimate;

    fn estimate_vector_matrix_product(
        &self,
        inner_len: usize,
        rhs_cols: usize,
    ) -> CircuitBenchSummary {
        scale_independent_summary(
            estimate_summary(self.estimate_vector_inner_product(inner_len)),
            rhs_cols,
        )
    }

    fn estimate_row_parallel_matrix_product(
        &self,
        lhs_rows: usize,
        inner_len: usize,
        rhs_cols: usize,
    ) -> CircuitBenchSummary {
        scale_independent_summary(
            self.estimate_vector_matrix_product(inner_len, rhs_cols),
            lhs_rows,
        )
    }
}
