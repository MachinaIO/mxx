use std::{marker::PhantomData, time::Instant};

use crate::{
    bench_estimator::{
        BenchEstimator, CircuitBenchEstimate, benchmark_gate_operation,
        column_parallel_gate_estimate,
    },
    bgg::{encoding::BggEncoding, public_key::BggPublicKey},
    circuit::Evaluable,
    matrix::PolyMatrix,
    poly::Poly,
};
use num_bigint::BigUint;
use tracing::info;

fn per_gate_time_estimate(time: f64, peak_vram: usize) -> CircuitBenchEstimate {
    CircuitBenchEstimate::new(time, time).with_peak_vram(peak_vram)
}

/// Bench estimator for one scalar `BggEncoding`.
///
/// `NaiveBGGVecBenchEstimator` scales a scalar BGG estimate across slots, so the inner estimator
/// must model a single ordinary `BggEncoding` rather than a packed `BggPolyEncoding`.
#[derive(Debug, Clone)]
pub struct BggEncodingBenchEstimator<M>
where
    M: PolyMatrix,
{
    pub input_time: f64,
    pub input_peak_vram: usize,
    pub add_time: f64,
    pub add_peak_vram: usize,
    pub sub_time: f64,
    pub sub_peak_vram: usize,
    pub mul_time: f64,
    pub mul_peak_vram: usize,
    pub mul_rhs_column_count: usize,
    pub small_scalar_mul_time: f64,
    pub small_scalar_mul_peak_vram: usize,
    pub large_scalar_mul_time: f64,
    pub large_scalar_mul_peak_vram: usize,
    pub large_scalar_mul_rhs_column_count: usize,
    pub public_lut_time: f64,
    pub public_lut_peak_vram: usize,
    pub slot_transfer_time: f64,
    pub slot_transfer_peak_vram: usize,
    _m: PhantomData<M>,
}

impl<M> BggEncodingBenchEstimator<M>
where
    M: PolyMatrix,
{
    /// Benchmarks scalar BGG encoding operations.
    ///
    /// `public_lookup_op` must execute the representative public-LUT lookup for the concrete
    /// encoding layer being estimated. Scalar `BggEncoding` does not define that operation by
    /// itself, while vector and packed encodings provide concrete lookup evaluators.
    pub fn benchmark<R, F>(
        params: &<M::P as Poly>::Params,
        iterations: usize,
        public_lookup_op: F,
    ) -> Self
    where
        F: FnMut() -> R,
    {
        let matrix = M::gadget_matrix(params, 1);
        let pubkey = BggPublicKey::new(matrix.clone(), true);
        let plaintext_a = M::P::from_usize_to_constant(params, 2);
        let plaintext_b = M::P::from_usize_to_constant(params, 3);
        let enc_a = BggEncoding::new(matrix.clone(), pubkey.clone(), Some(plaintext_a));
        let enc_b = BggEncoding::new(matrix, pubkey, Some(plaintext_b));

        info!("BggEncodingBenchEstimator::benchmark starting add");
        let started = Instant::now();
        let add = benchmark_gate_operation(iterations, || enc_a.clone() + &enc_b);
        info!(
            elapsed = ?started.elapsed(),
            peak_vram = add.peak_vram,
            "BggEncodingBenchEstimator::benchmark finished add"
        );
        info!("BggEncodingBenchEstimator::benchmark starting sub");
        let started = Instant::now();
        let sub = benchmark_gate_operation(iterations, || enc_a.clone() - &enc_b);
        info!(
            elapsed = ?started.elapsed(),
            peak_vram = sub.peak_vram,
            "BggEncodingBenchEstimator::benchmark finished sub"
        );
        info!("BggEncodingBenchEstimator::benchmark starting mul");
        let started = Instant::now();
        let mul = benchmark_gate_operation(iterations, || enc_a.clone() * &enc_b);
        let mul_rhs_column_count = enc_b.pubkey.matrix.col_size();
        info!(
            elapsed = ?started.elapsed(),
            peak_vram = mul.peak_vram,
            mul_rhs_column_count,
            "BggEncodingBenchEstimator::benchmark finished mul"
        );
        info!("BggEncodingBenchEstimator::benchmark starting small scalar mul");
        let started = Instant::now();
        let small_scalar_mul =
            benchmark_gate_operation(iterations, || enc_a.small_scalar_mul(params, &[3u32]));
        info!(
            elapsed = ?started.elapsed(),
            peak_vram = small_scalar_mul.peak_vram,
            "BggEncodingBenchEstimator::benchmark finished small scalar mul"
        );
        info!("BggEncodingBenchEstimator::benchmark starting large scalar mul");
        let started = Instant::now();
        let large_scalar_mul = benchmark_gate_operation(iterations, || {
            enc_a.large_scalar_mul(params, &[BigUint::from(5u32)])
        });
        let large_scalar_mul_rhs_column_count = enc_a.pubkey.matrix.col_size();
        info!(
            elapsed = ?started.elapsed(),
            peak_vram = large_scalar_mul.peak_vram,
            large_scalar_mul_rhs_column_count,
            "BggEncodingBenchEstimator::benchmark finished large scalar mul"
        );

        // Scalar `BggEncoding` does not have its own slot-transfer evaluator. The naive-vector
        // wrapper supplies slot structure, so this single-scalar clone cost keeps the estimate type
        // complete without introducing packed `BggPolyEncoding` machinery in scalar callers.
        info!("BggEncodingBenchEstimator::benchmark starting slot-transfer placeholder");
        let started = Instant::now();
        let slot_transfer = benchmark_gate_operation(iterations, || enc_a.clone());
        info!(
            elapsed = ?started.elapsed(),
            peak_vram = slot_transfer.peak_vram,
            "BggEncodingBenchEstimator::benchmark finished slot-transfer placeholder"
        );

        info!("BggEncodingBenchEstimator::benchmark starting public-LUT lookup");
        let started = Instant::now();
        let mut public_lookup_op = public_lookup_op;
        let public_lut = benchmark_gate_operation(iterations, || public_lookup_op());
        info!(
            elapsed = ?started.elapsed(),
            peak_vram = public_lut.peak_vram,
            "BggEncodingBenchEstimator::benchmark finished public-LUT lookup"
        );

        Self {
            input_time: 0.0,
            input_peak_vram: 0,
            add_time: add.time,
            add_peak_vram: add.peak_vram,
            sub_time: sub.time,
            sub_peak_vram: sub.peak_vram,
            mul_time: mul.time,
            mul_peak_vram: mul.peak_vram,
            mul_rhs_column_count,
            small_scalar_mul_time: small_scalar_mul.time,
            small_scalar_mul_peak_vram: small_scalar_mul.peak_vram,
            large_scalar_mul_time: large_scalar_mul.time,
            large_scalar_mul_peak_vram: large_scalar_mul.peak_vram,
            large_scalar_mul_rhs_column_count,
            public_lut_time: public_lut.time,
            public_lut_peak_vram: public_lut.peak_vram,
            slot_transfer_time: slot_transfer.time,
            slot_transfer_peak_vram: slot_transfer.peak_vram,
            _m: PhantomData,
        }
    }
}

impl<M> BenchEstimator<BggEncoding<M>> for BggEncodingBenchEstimator<M>
where
    M: PolyMatrix,
{
    fn estimate_input(&self) -> CircuitBenchEstimate {
        per_gate_time_estimate(self.input_time, self.input_peak_vram)
    }

    fn estimate_add(&self) -> CircuitBenchEstimate {
        per_gate_time_estimate(self.add_time, self.add_peak_vram)
    }

    fn estimate_sub(&self) -> CircuitBenchEstimate {
        per_gate_time_estimate(self.sub_time, self.sub_peak_vram)
    }

    fn estimate_mul(&self) -> CircuitBenchEstimate {
        column_parallel_gate_estimate(self.mul_time, self.mul_peak_vram, self.mul_rhs_column_count)
    }

    fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchEstimate {
        per_gate_time_estimate(self.small_scalar_mul_time, self.small_scalar_mul_peak_vram)
    }

    fn estimate_large_scalar_mul(&self, _scalar: &[BigUint]) -> CircuitBenchEstimate {
        column_parallel_gate_estimate(
            self.large_scalar_mul_time,
            self.large_scalar_mul_peak_vram,
            self.large_scalar_mul_rhs_column_count,
        )
    }

    fn estimate_slot_transfer(&self, src_slots: &[(u32, Option<u32>)]) -> CircuitBenchEstimate {
        let _ = src_slots;
        per_gate_time_estimate(self.slot_transfer_time, self.slot_transfer_peak_vram)
    }

    fn estimate_slot_reduce(&self, input_count: usize, _num_slots: usize) -> CircuitBenchEstimate {
        assert!(input_count > 0, "slot_reduce input_count must be positive");
        let mut estimate =
            per_gate_time_estimate(self.slot_transfer_time, self.slot_transfer_peak_vram);
        estimate.total_time *= BigUint::from(input_count);
        estimate.max_parallelism *= BigUint::from(input_count);
        estimate
    }

    fn estimate_public_lookup(&self, _lut_id: usize) -> CircuitBenchEstimate {
        per_gate_time_estimate(self.public_lut_time, self.public_lut_peak_vram)
    }
}

#[cfg(test)]
mod tests {
    use super::BggEncodingBenchEstimator;
    use crate::{
        bench_estimator::BenchEstimator, bgg::encoding::BggEncoding,
        matrix::dcrt_poly::DCRTPolyMatrix,
    };
    use std::marker::PhantomData;

    fn nanos(value: u64) -> num_bigint::BigUint {
        num_bigint::BigUint::from(value)
    }

    #[test]
    fn bgg_encoding_bench_estimator_returns_scalar_gate_estimates() {
        let estimator = BggEncodingBenchEstimator::<DCRTPolyMatrix> {
            input_time: 0.0,
            input_peak_vram: 0,
            add_time: 1.0,
            add_peak_vram: 0,
            sub_time: 2.0,
            sub_peak_vram: 0,
            mul_time: 3.0,
            mul_peak_vram: 0,
            mul_rhs_column_count: 3,
            small_scalar_mul_time: 4.0,
            small_scalar_mul_peak_vram: 0,
            large_scalar_mul_time: 5.0,
            large_scalar_mul_peak_vram: 0,
            large_scalar_mul_rhs_column_count: 5,
            public_lut_time: 6.0,
            public_lut_peak_vram: 0,
            slot_transfer_time: 7.0,
            slot_transfer_peak_vram: 0,
            _m: PhantomData,
        };

        let add = <BggEncodingBenchEstimator<DCRTPolyMatrix> as BenchEstimator<
            BggEncoding<DCRTPolyMatrix>,
        >>::estimate_add(&estimator);
        let mul = <BggEncodingBenchEstimator<DCRTPolyMatrix> as BenchEstimator<
            BggEncoding<DCRTPolyMatrix>,
        >>::estimate_mul(&estimator);
        let large_scalar_mul = <BggEncodingBenchEstimator<DCRTPolyMatrix> as BenchEstimator<
            BggEncoding<DCRTPolyMatrix>,
        >>::estimate_large_scalar_mul(
            &estimator, &[num_bigint::BigUint::from(7u32)]
        );
        let public_lookup = <BggEncodingBenchEstimator<DCRTPolyMatrix> as BenchEstimator<
            BggEncoding<DCRTPolyMatrix>,
        >>::estimate_public_lookup(&estimator, 0);

        assert!(add.latency.is_finite());
        assert!(public_lookup.latency.is_finite());
        assert_eq!(add.total_time, nanos(1_000_000_000));
        assert_eq!(mul.total_time, nanos(3_000_000_000));
        assert_eq!(mul.latency, 1.0);
        assert_eq!(mul.max_parallelism, nanos(3));
        assert_eq!(large_scalar_mul.total_time, nanos(5_000_000_000));
        assert_eq!(large_scalar_mul.latency, 1.0);
        assert_eq!(large_scalar_mul.max_parallelism, nanos(5));
        assert_eq!(public_lookup.total_time, nanos(6_000_000_000));
    }
}
