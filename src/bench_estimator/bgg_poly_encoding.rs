use std::{marker::PhantomData, sync::Arc};

use crate::{
    bench_estimator::{BenchEstimator, CircuitBenchEstimate, benchmark_gate_operation},
    bgg::{poly_encoding::BggPolyEncoding, public_key::BggPublicKey},
    circuit::{Evaluable, gate::GateId},
    element::PolyElem,
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    slot_transfer::SlotTransferEvaluator,
};
use num_bigint::BigUint;
use tracing::debug;

fn validate_single_slot_shape(
    num_slots: usize,
    input_slot_counts: &[(&str, usize)],
    slot_transfer_len: usize,
) -> Result<(), String> {
    if num_slots == 0 {
        return Err("BggPolyEncodingBenchSamples::num_slots must be positive".to_string());
    }
    for (label, slot_count) in input_slot_counts {
        if *slot_count != 1 {
            return Err(format!(
                "BggPolyEncodingBenchSamples::{label} must contain exactly one slot for latency benchmarking"
            ));
        }
    }
    if slot_transfer_len != 1 {
        return Err(
            "BggPolyEncodingBenchSamples::slot_transfer_src_slots must contain exactly one destination slot for latency benchmarking"
                .to_string(),
        );
    }
    Ok(())
}

pub struct BggPolyEncodingBenchSamples<'a, M: PolyMatrix> {
    pub num_slots: usize,
    pub params: &'a <M::P as Poly>::Params,
    pub add_lhs: &'a BggPolyEncoding<M>,
    pub add_rhs: &'a BggPolyEncoding<M>,
    pub sub_lhs: &'a BggPolyEncoding<M>,
    pub sub_rhs: &'a BggPolyEncoding<M>,
    pub mul_lhs: &'a BggPolyEncoding<M>,
    pub mul_rhs: &'a BggPolyEncoding<M>,
    pub small_scalar_input: &'a BggPolyEncoding<M>,
    pub small_scalar: &'a [u32],
    pub large_scalar_input: &'a BggPolyEncoding<M>,
    pub large_scalar: &'a [BigUint],
    pub public_lut_one: &'a BggPolyEncoding<M>,
    pub public_lut_input: &'a BggPolyEncoding<M>,
    pub public_lut: &'a PublicLut<M::P>,
    pub public_lut_gate_id: GateId,
    pub public_lut_id: usize,
    pub slot_transfer_input: &'a BggPolyEncoding<M>,
    pub slot_transfer_src_slots: &'a [(u32, Option<u32>)],
    pub slot_transfer_gate_id: GateId,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolyEncodingChunkBenchMeasurement {
    // One runtime critical path for one slot. For `public_lut` this is one output chunk; for
    // `slot transfer` this is the sum of the stage critical paths for one destination slot.
    pub latency: f64,
    // The full work needed to finish one slot after expanding all repeated chunk work and adding
    // any one-time stage costs such as concat / reload boundaries.
    pub total_time: f64,
    // The maximum number of same-stage chunks that could run at once for one slot.
    pub max_parallelism: usize,
    pub peak_vram: usize,
}

pub trait PolyEncodingPublicLutBenchEstimator<M>:
    PltEvaluator<BggPolyEncoding<M>> + Send + Sync
where
    M: PolyMatrix,
{
    fn benchmark_public_lookup_chunk(
        &self,
        samples: &BggPolyEncodingBenchSamples<'_, M>,
        iterations: usize,
    ) -> PolyEncodingChunkBenchMeasurement;
}

pub trait PolyEncodingSlotTransferBenchEstimator<M>:
    SlotTransferEvaluator<BggPolyEncoding<M>> + Send + Sync
where
    M: PolyMatrix,
{
    fn benchmark_slot_transfer_chunk(
        &self,
        samples: &BggPolyEncodingBenchSamples<'_, M>,
        iterations: usize,
    ) -> PolyEncodingChunkBenchMeasurement;
}

impl<M: PolyMatrix> BggPolyEncodingBenchSamples<'_, M> {
    fn validate_single_slot_inputs(&self) -> Result<(), String> {
        let slot_counts = [
            ("add_lhs", self.add_lhs.num_slots()),
            ("add_rhs", self.add_rhs.num_slots()),
            ("sub_lhs", self.sub_lhs.num_slots()),
            ("sub_rhs", self.sub_rhs.num_slots()),
            ("mul_lhs", self.mul_lhs.num_slots()),
            ("mul_rhs", self.mul_rhs.num_slots()),
            ("small_scalar_input", self.small_scalar_input.num_slots()),
            ("large_scalar_input", self.large_scalar_input.num_slots()),
            ("public_lut_one", self.public_lut_one.num_slots()),
            ("public_lut_input", self.public_lut_input.num_slots()),
            ("slot_transfer_input", self.slot_transfer_input.num_slots()),
        ];
        validate_single_slot_shape(self.num_slots, &slot_counts, self.slot_transfer_src_slots.len())
    }

    fn assert_single_slot_inputs(&self) {
        self.validate_single_slot_inputs().unwrap_or_else(|message| panic!("{message}"));
    }
}

struct OwnedBggPolyEncodingBenchSamples<M: PolyMatrix> {
    num_slots: usize,
    params: <M::P as Poly>::Params,
    add_lhs: BggPolyEncoding<M>,
    add_rhs: BggPolyEncoding<M>,
    sub_lhs: BggPolyEncoding<M>,
    sub_rhs: BggPolyEncoding<M>,
    mul_lhs: BggPolyEncoding<M>,
    mul_rhs: BggPolyEncoding<M>,
    small_scalar_input: BggPolyEncoding<M>,
    small_scalar: Vec<u32>,
    large_scalar_input: BggPolyEncoding<M>,
    large_scalar: Vec<BigUint>,
    public_lut_one: BggPolyEncoding<M>,
    public_lut_input: BggPolyEncoding<M>,
    public_lut: PublicLut<M::P>,
    public_lut_gate_id: GateId,
    public_lut_id: usize,
    slot_transfer_input: BggPolyEncoding<M>,
    slot_transfer_src_slots: Vec<(u32, Option<u32>)>,
    slot_transfer_gate_id: GateId,
}

impl<M: PolyMatrix> OwnedBggPolyEncodingBenchSamples<M> {
    fn sample_encoding(
        params: &<M::P as Poly>::Params,
        pubkey_matrix: &M,
        plaintext_constant: usize,
    ) -> BggPolyEncoding<M> {
        let vector_bytes = vec![Arc::<[u8]>::from(pubkey_matrix.clone().into_compact_bytes())];
        let plaintext_bytes = Some(vec![Arc::<[u8]>::from(
            M::P::from_usize_to_constant(params, plaintext_constant).to_compact_bytes(),
        )]);
        BggPolyEncoding::new(
            params.clone(),
            vector_bytes,
            BggPublicKey::new(pubkey_matrix.clone(), true),
            plaintext_bytes,
        )
    }

    fn default_for_benchmark(num_slots: usize) -> Self
    where
        <M::P as Poly>::Params: Default,
    {
        assert!(num_slots > 0, "BggPolyEncodingBenchEstimator::benchmark num_slots must be > 0");
        let params = <M::P as Poly>::Params::default();
        let pubkey_matrix = M::gadget_matrix(&params, 1);
        let public_lut = PublicLut::new(
            &params,
            2,
            |params: &<M::P as Poly>::Params, x| {
                Some((x, <M::P as Poly>::Elem::constant(&params.modulus(), x + 1)))
            },
            None,
        );

        Self {
            num_slots,
            add_lhs: Self::sample_encoding(&params, &pubkey_matrix, 2),
            add_rhs: Self::sample_encoding(&params, &pubkey_matrix, 3),
            sub_lhs: Self::sample_encoding(&params, &pubkey_matrix, 4),
            sub_rhs: Self::sample_encoding(&params, &pubkey_matrix, 1),
            mul_lhs: Self::sample_encoding(&params, &pubkey_matrix, 2),
            mul_rhs: Self::sample_encoding(&params, &pubkey_matrix, 5),
            small_scalar_input: Self::sample_encoding(&params, &pubkey_matrix, 3),
            small_scalar: vec![3u32, 5u32],
            large_scalar_input: Self::sample_encoding(&params, &pubkey_matrix, 7),
            large_scalar: vec![BigUint::from(7u32)],
            public_lut_one: Self::sample_encoding(&params, &pubkey_matrix, 1),
            public_lut_input: Self::sample_encoding(&params, &pubkey_matrix, 2),
            public_lut,
            public_lut_gate_id: GateId(17),
            public_lut_id: 7,
            slot_transfer_input: Self::sample_encoding(&params, &pubkey_matrix, 4),
            slot_transfer_src_slots: vec![(0, Some(2u32))],
            slot_transfer_gate_id: GateId(19),
            params,
        }
    }

    fn borrowed(&self) -> BggPolyEncodingBenchSamples<'_, M> {
        BggPolyEncodingBenchSamples {
            num_slots: self.num_slots,
            params: &self.params,
            add_lhs: &self.add_lhs,
            add_rhs: &self.add_rhs,
            sub_lhs: &self.sub_lhs,
            sub_rhs: &self.sub_rhs,
            mul_lhs: &self.mul_lhs,
            mul_rhs: &self.mul_rhs,
            small_scalar_input: &self.small_scalar_input,
            small_scalar: &self.small_scalar,
            large_scalar_input: &self.large_scalar_input,
            large_scalar: &self.large_scalar,
            public_lut_one: &self.public_lut_one,
            public_lut_input: &self.public_lut_input,
            public_lut: &self.public_lut,
            public_lut_gate_id: self.public_lut_gate_id,
            public_lut_id: self.public_lut_id,
            slot_transfer_input: &self.slot_transfer_input,
            slot_transfer_src_slots: &self.slot_transfer_src_slots,
            slot_transfer_gate_id: self.slot_transfer_gate_id,
        }
    }
}

pub struct BggPolyEncodingBenchEstimator<M>
where
    M: PolyMatrix,
{
    pub num_slots: usize,
    pub input_time: f64,
    pub input_peak_vram: usize,
    pub add_time: f64,
    pub add_peak_vram: usize,
    pub sub_time: f64,
    pub sub_peak_vram: usize,
    pub mul_time: f64,
    pub mul_peak_vram: usize,
    pub small_scalar_mul_time: f64,
    pub small_scalar_mul_peak_vram: usize,
    pub large_scalar_mul_time: f64,
    pub large_scalar_mul_peak_vram: usize,
    pub public_lut_latency: f64,
    pub public_lut_total_time: f64,
    pub public_lut_peak_vram: usize,
    pub slot_transfer_latency: f64,
    pub slot_transfer_total_time: f64,
    pub slot_transfer_peak_vram: usize,
    pub slot_transfer_max_parallelism: usize,
    pub public_lut_max_parallelism: usize,
    _m: PhantomData<M>,
}

impl<M> BggPolyEncodingBenchEstimator<M>
where
    M: PolyMatrix,
{
    pub fn benchmark<PE, SE>(
        num_slots: usize,
        samples: Option<&BggPolyEncodingBenchSamples<'_, M>>,
        public_lut_evaluator: &PE,
        slot_transfer_evaluator: &SE,
        iterations: usize,
    ) -> Self
    where
        PE: PolyEncodingPublicLutBenchEstimator<M>,
        SE: PolyEncodingSlotTransferBenchEstimator<M>,
        <M::P as Poly>::Params: Default,
    {
        if let Some(samples) = samples {
            assert_eq!(
                samples.num_slots, num_slots,
                "BggPolyEncodingBenchEstimator::benchmark requires num_slots == samples.num_slots"
            );
            return Self::benchmark_with_samples(
                samples,
                public_lut_evaluator,
                slot_transfer_evaluator,
                iterations,
            );
        }

        let owned_samples = OwnedBggPolyEncodingBenchSamples::<M>::default_for_benchmark(num_slots);
        let borrowed_samples = owned_samples.borrowed();
        Self::benchmark_with_samples(
            &borrowed_samples,
            public_lut_evaluator,
            slot_transfer_evaluator,
            iterations,
        )
    }

    pub fn benchmark_with_samples<PE, SE>(
        samples: &BggPolyEncodingBenchSamples<'_, M>,
        public_lut_evaluator: &PE,
        slot_transfer_evaluator: &SE,
        iterations: usize,
    ) -> Self
    where
        PE: PolyEncodingPublicLutBenchEstimator<M>,
        SE: PolyEncodingSlotTransferBenchEstimator<M>,
    {
        samples.assert_single_slot_inputs();

        // All benchmark samples are single-slot. Higher-level slot-count scaling happens only in
        // `estimate_*`, so the benchmarked operation here should be the exact one-slot runtime
        // unit with no extra multiplication or normalization.
        let add_bench =
            benchmark_gate_operation(iterations, || samples.add_lhs.clone() + samples.add_rhs);
        debug!("BggPolyEncodingBenchEstimator::benchmark add_bench={:?}", add_bench);
        let sub_bench =
            benchmark_gate_operation(iterations, || samples.sub_lhs.clone() - samples.sub_rhs);
        debug!("BggPolyEncodingBenchEstimator::benchmark sub_bench={:?}", sub_bench);
        let mul_bench =
            benchmark_gate_operation(iterations, || samples.mul_lhs.clone() * samples.mul_rhs);
        debug!("BggPolyEncodingBenchEstimator::benchmark mul_bench={:?}", mul_bench);
        let small_scalar_mul_bench = benchmark_gate_operation(iterations, || {
            samples.small_scalar_input.small_scalar_mul(samples.params, samples.small_scalar)
        });
        debug!(
            "BggPolyEncodingBenchEstimator::benchmark small_scalar_mul_bench={:?}",
            small_scalar_mul_bench
        );
        let large_scalar_mul_bench = benchmark_gate_operation(iterations, || {
            samples.large_scalar_input.large_scalar_mul(samples.params, samples.large_scalar)
        });
        debug!(
            "BggPolyEncodingBenchEstimator::benchmark large_scalar_mul_bench={:?}",
            large_scalar_mul_bench
        );
        let public_lut_bench =
            public_lut_evaluator.benchmark_public_lookup_chunk(samples, iterations);
        debug!("BggPolyEncodingBenchEstimator::benchmark public_lut_bench={:?}", public_lut_bench);
        let slot_transfer_bench =
            slot_transfer_evaluator.benchmark_slot_transfer_chunk(samples, iterations);
        debug!(
            "BggPolyEncodingBenchEstimator::benchmark slot_transfer_bench={:?}",
            slot_transfer_bench
        );

        Self {
            num_slots: samples.num_slots,
            input_time: 0.0,
            input_peak_vram: 0,
            add_time: add_bench.time,
            add_peak_vram: add_bench.peak_vram,
            sub_time: sub_bench.time,
            sub_peak_vram: sub_bench.peak_vram,
            mul_time: mul_bench.time,
            mul_peak_vram: mul_bench.peak_vram,
            small_scalar_mul_time: small_scalar_mul_bench.time,
            small_scalar_mul_peak_vram: small_scalar_mul_bench.peak_vram,
            large_scalar_mul_time: large_scalar_mul_bench.time,
            large_scalar_mul_peak_vram: large_scalar_mul_bench.peak_vram,
            public_lut_latency: public_lut_bench.latency,
            public_lut_total_time: public_lut_bench.total_time,
            public_lut_peak_vram: public_lut_bench.peak_vram,
            slot_transfer_latency: slot_transfer_bench.latency,
            slot_transfer_total_time: slot_transfer_bench.total_time,
            slot_transfer_peak_vram: slot_transfer_bench.peak_vram,
            slot_transfer_max_parallelism: slot_transfer_bench.max_parallelism,
            public_lut_max_parallelism: public_lut_bench.max_parallelism,
            _m: PhantomData,
        }
    }
}

impl<M> BenchEstimator<BggPolyEncoding<M>> for BggPolyEncodingBenchEstimator<M>
where
    M: PolyMatrix,
{
    fn estimate_input(&self) -> CircuitBenchEstimate {
        CircuitBenchEstimate::new(
            self.input_time * self.num_slots as f64,
            self.input_time,
            0,
            self.input_peak_vram,
        )
    }

    fn estimate_add(&self) -> CircuitBenchEstimate {
        CircuitBenchEstimate::new(
            self.add_time * self.num_slots as f64,
            self.add_time,
            0,
            self.add_peak_vram,
        )
    }

    fn estimate_sub(&self) -> CircuitBenchEstimate {
        CircuitBenchEstimate::new(
            self.sub_time * self.num_slots as f64,
            self.sub_time,
            0,
            self.sub_peak_vram,
        )
    }

    fn estimate_mul(&self) -> CircuitBenchEstimate {
        CircuitBenchEstimate::new(
            self.mul_time * self.num_slots as f64,
            self.mul_time,
            0,
            self.mul_peak_vram,
        )
    }

    fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchEstimate {
        CircuitBenchEstimate::new(
            self.small_scalar_mul_time * self.num_slots as f64,
            self.small_scalar_mul_time,
            0,
            self.small_scalar_mul_peak_vram,
        )
    }

    fn estimate_large_scalar_mul(&self, _scalar: &[BigUint]) -> CircuitBenchEstimate {
        CircuitBenchEstimate::new(
            self.large_scalar_mul_time * self.num_slots as f64,
            self.large_scalar_mul_time,
            0,
            self.large_scalar_mul_peak_vram,
        )
    }

    fn estimate_slot_transfer(&self, src_slots: &[(u32, Option<u32>)]) -> CircuitBenchEstimate {
        assert_eq!(
            src_slots.len(),
            self.num_slots,
            "BggPolyEncodingBenchEstimator::estimate_slot_transfer requires src_slots.len() == num_slots"
        );
        CircuitBenchEstimate::new(
            self.slot_transfer_total_time * self.num_slots as f64,
            self.slot_transfer_latency,
            (self.slot_transfer_max_parallelism.max(1) as u128) * (self.num_slots as u128),
            self.slot_transfer_peak_vram,
        )
    }

    fn estimate_public_lookup(&self, _lut_id: usize) -> CircuitBenchEstimate {
        CircuitBenchEstimate::new(
            self.public_lut_total_time * self.num_slots as f64,
            self.public_lut_latency,
            (self.public_lut_max_parallelism.max(1) as u128) * (self.num_slots as u128),
            self.public_lut_peak_vram,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{BggPolyEncodingBenchEstimator, validate_single_slot_shape};
    use crate::{
        __PAIR, __TestState,
        bench_estimator::{BenchEstimator, CircuitBenchEstimate},
        matrix::dcrt_poly::DCRTPolyMatrix,
    };
    use num_bigint::BigUint;
    use sequential_test::sequential;
    use std::marker::PhantomData;

    fn test_estimator() -> BggPolyEncodingBenchEstimator<DCRTPolyMatrix> {
        BggPolyEncodingBenchEstimator {
            num_slots: 3,
            input_time: 0.0,
            input_peak_vram: 0,
            add_time: 1.0,
            add_peak_vram: 71,
            sub_time: 2.0,
            sub_peak_vram: 73,
            mul_time: 3.0,
            mul_peak_vram: 79,
            small_scalar_mul_time: 4.0,
            small_scalar_mul_peak_vram: 83,
            large_scalar_mul_time: 5.0,
            large_scalar_mul_peak_vram: 89,
            public_lut_latency: 6.0,
            public_lut_total_time: 12.0,
            public_lut_peak_vram: 97,
            slot_transfer_latency: 7.0,
            slot_transfer_total_time: 28.0,
            slot_transfer_peak_vram: 101,
            slot_transfer_max_parallelism: 4,
            public_lut_max_parallelism: 2,
            _m: PhantomData,
        }
    }

    #[test]
    #[sequential]
    fn test_bgg_poly_encoding_bench_estimator_scales_one_slot_latency_by_num_slots() {
        let estimator = test_estimator();

        assert_eq!(estimator.estimate_input(), CircuitBenchEstimate::new(0.0, 0.0, 0, 0));

        let add = estimator.estimate_add();
        assert_eq!(add, CircuitBenchEstimate::new(3.0, 1.0, 0, estimator.add_peak_vram));

        let sub = estimator.estimate_sub();
        assert_eq!(sub, CircuitBenchEstimate::new(6.0, 2.0, 0, estimator.sub_peak_vram));

        let mul = estimator.estimate_mul();
        assert_eq!(mul, CircuitBenchEstimate::new(9.0, 3.0, 0, estimator.mul_peak_vram));

        let small_scalar = estimator.estimate_small_scalar_mul(&[3u32, 5u32]);
        assert_eq!(
            small_scalar,
            CircuitBenchEstimate::new(12.0, 4.0, 0, estimator.small_scalar_mul_peak_vram)
        );

        let large_scalar = estimator.estimate_large_scalar_mul(&[BigUint::from(7u32)]);
        assert_eq!(
            large_scalar,
            CircuitBenchEstimate::new(15.0, 5.0, 0, estimator.large_scalar_mul_peak_vram)
        );

        let public_lookup = estimator.estimate_public_lookup(7);
        assert_eq!(
            public_lookup,
            CircuitBenchEstimate::new(36.0, 6.0, 6, estimator.public_lut_peak_vram)
        );

        let slot_transfer =
            estimator.estimate_slot_transfer(&[(0, None), (0, Some(2)), (0, Some(1))]);
        assert_eq!(
            slot_transfer,
            CircuitBenchEstimate::new(84.0, 7.0, 12, estimator.slot_transfer_peak_vram)
        );
    }

    #[test]
    #[sequential]
    fn test_bgg_poly_encoding_bench_estimator_rejects_multi_slot_custom_samples() {
        assert_eq!(
            validate_single_slot_shape(
                1,
                &[
                    ("add_lhs", 1),
                    ("add_rhs", 1),
                    ("sub_lhs", 1),
                    ("sub_rhs", 1),
                    ("mul_lhs", 1),
                    ("mul_rhs", 1),
                    ("small_scalar_input", 1),
                    ("large_scalar_input", 1),
                    ("public_lut_one", 1),
                    ("public_lut_input", 1),
                    ("slot_transfer_input", 1),
                ],
                2,
            ),
            Err(
                "BggPolyEncodingBenchSamples::slot_transfer_src_slots must contain exactly one destination slot for latency benchmarking"
                    .to_string()
            )
        );
    }
}
