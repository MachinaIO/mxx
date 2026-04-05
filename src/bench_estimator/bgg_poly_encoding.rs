use std::{marker::PhantomData, sync::Arc};

use crate::{
    bench_estimator::{BenchEstimator, CircuitBenchEstimate, measure_bench_operation},
    bgg::{poly_encoding::BggPolyEncoding, public_key::BggPublicKey},
    circuit::{Evaluable, gate::GateId},
    element::PolyElem,
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::{Poly, PolyParams},
    slot_transfer::SlotTransferEvaluator,
};
use num_bigint::BigUint;

fn per_slot_gate_estimate(latency: f64, num_slots: usize) -> CircuitBenchEstimate {
    CircuitBenchEstimate { latency, total_time: latency * num_slots as f64 }
}

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
    pub add_time: f64,
    pub sub_time: f64,
    pub mul_time: f64,
    pub small_scalar_mul_time: f64,
    pub large_scalar_mul_time: f64,
    pub public_lut_time: f64,
    pub slot_transfer_time: f64,
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
        PE: PltEvaluator<BggPolyEncoding<M>>,
        SE: SlotTransferEvaluator<BggPolyEncoding<M>>,
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
        PE: PltEvaluator<BggPolyEncoding<M>>,
        SE: SlotTransferEvaluator<BggPolyEncoding<M>>,
    {
        samples.assert_single_slot_inputs();

        let add_time =
            measure_bench_operation(iterations, || samples.add_lhs.clone() + samples.add_rhs);
        let sub_time =
            measure_bench_operation(iterations, || samples.sub_lhs.clone() - samples.sub_rhs);
        let mul_time =
            measure_bench_operation(iterations, || samples.mul_lhs.clone() * samples.mul_rhs);
        let small_scalar_mul_time = measure_bench_operation(iterations, || {
            samples.small_scalar_input.small_scalar_mul(samples.params, samples.small_scalar)
        });
        let large_scalar_mul_time = measure_bench_operation(iterations, || {
            samples.large_scalar_input.large_scalar_mul(samples.params, samples.large_scalar)
        });
        let public_lut_time = measure_bench_operation(iterations, || {
            public_lut_evaluator.public_lookup(
                samples.params,
                samples.public_lut,
                samples.public_lut_one,
                samples.public_lut_input,
                samples.public_lut_gate_id,
                samples.public_lut_id,
            )
        });
        let slot_transfer_time = measure_bench_operation(iterations, || {
            slot_transfer_evaluator.slot_transfer(
                samples.params,
                samples.slot_transfer_input,
                samples.slot_transfer_src_slots,
                samples.slot_transfer_gate_id,
            )
        });

        Self {
            num_slots: samples.num_slots,
            input_time: 0.0,
            add_time,
            sub_time,
            mul_time,
            small_scalar_mul_time,
            large_scalar_mul_time,
            public_lut_time,
            slot_transfer_time,
            _m: PhantomData,
        }
    }
}

impl<M> BenchEstimator<BggPolyEncoding<M>> for BggPolyEncodingBenchEstimator<M>
where
    M: PolyMatrix,
{
    fn estimate_input(&self) -> CircuitBenchEstimate {
        per_slot_gate_estimate(self.input_time, self.num_slots)
    }

    fn estimate_add(&self) -> CircuitBenchEstimate {
        per_slot_gate_estimate(self.add_time, self.num_slots)
    }

    fn estimate_sub(&self) -> CircuitBenchEstimate {
        per_slot_gate_estimate(self.sub_time, self.num_slots)
    }

    fn estimate_mul(&self) -> CircuitBenchEstimate {
        per_slot_gate_estimate(self.mul_time, self.num_slots)
    }

    fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchEstimate {
        per_slot_gate_estimate(self.small_scalar_mul_time, self.num_slots)
    }

    fn estimate_large_scalar_mul(&self, _scalar: &[BigUint]) -> CircuitBenchEstimate {
        per_slot_gate_estimate(self.large_scalar_mul_time, self.num_slots)
    }

    fn estimate_slot_transfer(&self, src_slots: &[(u32, Option<u32>)]) -> CircuitBenchEstimate {
        assert_eq!(
            src_slots.len(),
            self.num_slots,
            "BggPolyEncodingBenchEstimator::estimate_slot_transfer requires src_slots.len() == num_slots"
        );
        per_slot_gate_estimate(self.slot_transfer_time, self.num_slots)
    }

    fn estimate_public_lookup(&self, _lut_id: usize) -> CircuitBenchEstimate {
        per_slot_gate_estimate(self.public_lut_time, self.num_slots)
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
            add_time: 1.0,
            sub_time: 2.0,
            mul_time: 3.0,
            small_scalar_mul_time: 4.0,
            large_scalar_mul_time: 5.0,
            public_lut_time: 6.0,
            slot_transfer_time: 7.0,
            _m: PhantomData,
        }
    }

    #[test]
    #[sequential]
    fn test_bgg_poly_encoding_bench_estimator_scales_one_slot_latency_by_num_slots() {
        let estimator = test_estimator();

        assert_eq!(
            estimator.estimate_input(),
            CircuitBenchEstimate { total_time: 0.0, latency: 0.0 }
        );

        let add = estimator.estimate_add();
        assert!(add.latency >= 0.0);
        assert!((add.total_time - add.latency * 3.0).abs() < 1e-9);

        let sub = estimator.estimate_sub();
        assert!(sub.latency >= 0.0);
        assert!((sub.total_time - sub.latency * 3.0).abs() < 1e-9);

        let mul = estimator.estimate_mul();
        assert!(mul.latency >= 0.0);
        assert!((mul.total_time - mul.latency * 3.0).abs() < 1e-9);

        let small_scalar = estimator.estimate_small_scalar_mul(&[3u32, 5u32]);
        assert!(small_scalar.latency >= 0.0);
        assert!((small_scalar.total_time - small_scalar.latency * 3.0).abs() < 1e-9);

        let large_scalar = estimator.estimate_large_scalar_mul(&[BigUint::from(7u32)]);
        assert!(large_scalar.latency >= 0.0);
        assert!((large_scalar.total_time - large_scalar.latency * 3.0).abs() < 1e-9);

        let public_lookup = estimator.estimate_public_lookup(7);
        assert_eq!(public_lookup.latency, estimator.public_lut_time);
        assert!((public_lookup.total_time - public_lookup.latency * 3.0).abs() < 1e-9);

        let slot_transfer =
            estimator.estimate_slot_transfer(&[(0, None), (0, Some(2)), (0, Some(1))]);
        assert_eq!(slot_transfer.latency, estimator.slot_transfer_time);
        assert!((slot_transfer.total_time - slot_transfer.latency * 3.0).abs() < 1e-9);
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
