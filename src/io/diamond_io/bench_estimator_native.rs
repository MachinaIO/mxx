use crate::bench_estimator::{CircuitBenchEstimate, CircuitBenchSummary};

#[cfg(feature = "gpu")]
use std::{collections::HashMap, sync::RwLock};

#[cfg(feature = "gpu")]
use tracing::info;

#[cfg(feature = "gpu")]
use crate::{
    matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
    poly::{
        Poly, PolyParams,
        dcrt::{
            gpu::{GpuDCRTPoly, GpuDCRTPolyParams, gpu_device_sync},
            params::DCRTPolyParams,
        },
    },
    sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
};

use super::bench_estimator_utils::{scale_estimate, scale_summary};

/// Size-aware benchmark model for native polynomial-vector arithmetic.
///
/// Implementations should include the CPU-to-GPU upload of input data and GPU-to-CPU download of
/// output data in each returned operation estimate. DiamondIO then composes these measured units by
/// dependency structure rather than adding a separate transfer model.
pub trait DiamondIONativeBenchEstimator {
    /// Estimates multiplying one polynomial scalar into a vector of `vector_len` polynomials.
    fn estimate_poly_vector_mul(&self, vector_len: usize) -> CircuitBenchEstimate;

    /// Estimates one inner product between two vectors of `vector_len` polynomials.
    fn estimate_vector_inner_product(&self, vector_len: usize) -> CircuitBenchEstimate;

    /// Estimates adding two polynomial vectors of `vector_len` entries.
    fn estimate_vector_add(&self, vector_len: usize) -> CircuitBenchEstimate;

    /// Estimates subtracting two polynomial vectors of `vector_len` entries.
    fn estimate_vector_sub(&self, vector_len: usize) -> CircuitBenchEstimate;

    /// Estimates a row-vector by matrix product.
    ///
    /// The model computes each RHS column independently. The one-column inner-product estimate
    /// includes uploading both input vectors and downloading the one-polynomial output, so this
    /// composition still accounts for CPU/GPU transfer time per independent column.
    fn estimate_vector_matrix_product(
        &self,
        inner_len: usize,
        rhs_cols: usize,
    ) -> CircuitBenchSummary {
        scale_estimate(self.estimate_vector_inner_product(inner_len), rhs_cols)
    }

    /// Estimates an `lhs_rows x inner_len` by `inner_len x rhs_cols` product as independent row
    /// vector products. DiamondIO only needs this for small selector matrices during target
    /// construction; no dense matrix-matrix primitive is assumed.
    fn estimate_row_parallel_matrix_product(
        &self,
        lhs_rows: usize,
        inner_len: usize,
        rhs_cols: usize,
    ) -> CircuitBenchSummary {
        scale_summary(self.estimate_vector_matrix_product(inner_len, rhs_cols), lhs_rows)
    }
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
enum GpuNativeBenchOp {
    PolyVectorMul,
    VectorInnerProduct,
    VectorAdd,
    VectorSub,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
struct GpuNativeBenchKey {
    op: GpuNativeBenchOp,
    vector_len: usize,
}

/// Measured native-operation estimator for `GpuDCRTPolyMatrix`.
///
/// Each measured operation starts from compact CPU bytes, uploads the inputs to GPU matrices,
/// executes the GPU operation, and stores the result back to compact bytes. This deliberately
/// includes CPU-to-GPU input transfer and GPU-to-CPU output transfer in the operation unit used by
/// DiamondIO benchmark estimation.
#[cfg(feature = "gpu")]
#[derive(Debug)]
pub struct GpuDCRTPolyMatrixNativeBenchEstimator {
    pub params: GpuDCRTPolyParams,
    pub iterations: usize,
    cache: RwLock<HashMap<GpuNativeBenchKey, CircuitBenchEstimate>>,
}

#[cfg(feature = "gpu")]
impl GpuDCRTPolyMatrixNativeBenchEstimator {
    pub fn new(params: GpuDCRTPolyParams, iterations: usize) -> Self {
        Self { params, iterations: iterations.max(1), cache: RwLock::new(HashMap::new()) }
    }

    fn cpu_params(&self) -> DCRTPolyParams {
        DCRTPolyParams::new(
            self.params.ring_dimension(),
            self.params.crt_depth(),
            self.params.crt_bits(),
            self.params.base_bits(),
        )
    }

    fn sample_matrix_bytes(&self, nrow: usize, ncol: usize, is_ntt: bool) -> Vec<u8> {
        GpuDCRTPolyMatrix::zero_compact_bytes(
            &self.params,
            nrow,
            ncol,
            0,
            is_ntt,
            self.params.modulus_bits().try_into().expect("GPU modulus bits must fit in u16"),
        )
    }

    fn sample_poly_bytes(&self) -> Vec<u8> {
        let sampler = DCRTPolyUniformSampler::new();
        sampler.sample_poly(&self.cpu_params(), &DistType::FinRingDist).to_compact_bytes()
    }

    fn cached<F>(&self, key: GpuNativeBenchKey, measure: F) -> CircuitBenchEstimate
    where
        F: FnOnce(&Self) -> CircuitBenchEstimate,
    {
        if let Some(estimate) = self
            .cache
            .read()
            .expect("DiamondIO native bench cache read lock poisoned")
            .get(&key)
            .copied()
        {
            info!(?key, ?estimate, "DiamondIO native GPU bench cache hit");
            return estimate;
        }
        info!(?key, "DiamondIO native GPU bench cache miss; measuring operation");
        let estimate = measure(self);
        info!(?key, ?estimate, "DiamondIO native GPU bench measurement complete");
        self.cache
            .write()
            .expect("DiamondIO native bench cache write lock poisoned")
            .insert(key, estimate);
        estimate
    }

    fn measured<F>(&self, op: F) -> CircuitBenchEstimate
    where
        F: FnMut(),
    {
        gpu_device_sync();
        let measurement = crate::bench_estimator::benchmark_gate_operation(self.iterations, op);
        gpu_device_sync();
        CircuitBenchEstimate::new(measurement.time, measurement.time)
            .with_peak_vram(measurement.peak_vram)
    }
}

#[cfg(feature = "gpu")]
impl DiamondIONativeBenchEstimator for GpuDCRTPolyMatrixNativeBenchEstimator {
    fn estimate_poly_vector_mul(&self, vector_len: usize) -> CircuitBenchEstimate {
        assert!(vector_len > 0, "poly-vector multiplication requires a nonempty vector");
        self.cached(
            GpuNativeBenchKey { op: GpuNativeBenchOp::PolyVectorMul, vector_len },
            |estimator| {
                let scalar_bytes = estimator.sample_poly_bytes();
                let vector_bytes = estimator.sample_matrix_bytes(1, vector_len, true);
                info!(
                    vector_len,
                    scalar_bytes = scalar_bytes.len(),
                    vector_bytes = vector_bytes.len(),
                    vector_format = "eval",
                    "DiamondIO native GPU bench prepared poly-vector mul inputs"
                );
                estimator.measured(move || {
                    let scalar = GpuDCRTPoly::from_compact_bytes(&estimator.params, &scalar_bytes);
                    let vector =
                        GpuDCRTPolyMatrix::from_compact_bytes(&estimator.params, &vector_bytes);
                    let out = &vector * &scalar;
                    std::hint::black_box(out.into_compact_bytes());
                })
            },
        )
    }

    fn estimate_vector_inner_product(&self, vector_len: usize) -> CircuitBenchEstimate {
        assert!(vector_len > 0, "vector inner product requires a nonempty vector");
        self.cached(
            GpuNativeBenchKey { op: GpuNativeBenchOp::VectorInnerProduct, vector_len },
            |estimator| {
                let lhs_bytes = estimator.sample_matrix_bytes(1, vector_len, true);
                let rhs_bytes = estimator.sample_matrix_bytes(vector_len, 1, true);
                info!(
                    vector_len,
                    lhs_bytes = lhs_bytes.len(),
                    rhs_bytes = rhs_bytes.len(),
                    input_format = "eval",
                    "DiamondIO native GPU bench prepared vector inner-product inputs"
                );
                estimator.measured(move || {
                    let lhs = GpuDCRTPolyMatrix::from_compact_bytes(&estimator.params, &lhs_bytes);
                    let rhs = GpuDCRTPolyMatrix::from_compact_bytes(&estimator.params, &rhs_bytes);
                    let out = &lhs * &rhs;
                    std::hint::black_box(out.into_compact_bytes());
                })
            },
        )
    }

    fn estimate_vector_add(&self, vector_len: usize) -> CircuitBenchEstimate {
        assert!(vector_len > 0, "vector addition requires a nonempty vector");
        self.cached(
            GpuNativeBenchKey { op: GpuNativeBenchOp::VectorAdd, vector_len },
            |estimator| {
                let lhs_bytes = estimator.sample_matrix_bytes(1, vector_len, false);
                let rhs_bytes = estimator.sample_matrix_bytes(1, vector_len, false);
                info!(
                    vector_len,
                    lhs_bytes = lhs_bytes.len(),
                    rhs_bytes = rhs_bytes.len(),
                    input_format = "coeff",
                    "DiamondIO native GPU bench prepared vector add inputs"
                );
                estimator.measured(move || {
                    let lhs = GpuDCRTPolyMatrix::from_compact_bytes(&estimator.params, &lhs_bytes);
                    let rhs = GpuDCRTPolyMatrix::from_compact_bytes(&estimator.params, &rhs_bytes);
                    let out = lhs + &rhs;
                    std::hint::black_box(out.into_compact_bytes());
                })
            },
        )
    }

    fn estimate_vector_sub(&self, vector_len: usize) -> CircuitBenchEstimate {
        assert!(vector_len > 0, "vector subtraction requires a nonempty vector");
        self.cached(
            GpuNativeBenchKey { op: GpuNativeBenchOp::VectorSub, vector_len },
            |estimator| {
                let lhs_bytes = estimator.sample_matrix_bytes(1, vector_len, false);
                let rhs_bytes = estimator.sample_matrix_bytes(1, vector_len, false);
                info!(
                    vector_len,
                    lhs_bytes = lhs_bytes.len(),
                    rhs_bytes = rhs_bytes.len(),
                    input_format = "coeff",
                    "DiamondIO native GPU bench prepared vector sub inputs"
                );
                estimator.measured(move || {
                    let lhs = GpuDCRTPolyMatrix::from_compact_bytes(&estimator.params, &lhs_bytes);
                    let rhs = GpuDCRTPolyMatrix::from_compact_bytes(&estimator.params, &rhs_bytes);
                    let out = lhs - &rhs;
                    std::hint::black_box(out.into_compact_bytes());
                })
            },
        )
    }
}
