use std::marker::PhantomData;

use crate::{
    bgg::public_key::BggPublicKey,
    circuit::{Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::Poly,
    slot_transfer::SlotTransferEvaluator,
};
use num_bigint::BigUint;

use super::{BenchEstimator, CircuitBenchUnitEstimate, measure_bench_operation};

pub(crate) fn per_gate_time_estimate(time: f64) -> CircuitBenchUnitEstimate {
    CircuitBenchUnitEstimate { latency: time, total_time: time }
}

pub trait PublicLutSampleAuxBenchEstimator {
    fn sample_aux_matrices_lut_entry_time(&self) -> CircuitBenchUnitEstimate;
    fn sample_aux_matrices_lut_gate_time(&self) -> CircuitBenchUnitEstimate;

    fn benchmark_public_lut_gate_time<R, F>(&self, iterations: usize, op: F) -> f64
    where
        F: FnMut() -> R,
    {
        measure_bench_operation(iterations, op)
    }
}

pub trait SlotTransferSampleAuxBenchEstimator {
    fn sample_aux_matrices_slot_time(&self) -> CircuitBenchUnitEstimate;
    fn sample_aux_matrices_gate_time(&self) -> CircuitBenchUnitEstimate;

    fn benchmark_slot_transfer_gate_time<R, F>(&self, iterations: usize, op: F) -> f64
    where
        F: FnMut() -> R,
    {
        measure_bench_operation(iterations, op)
    }
}

pub struct BggPublicKeyBenchSamples<'a, M: PolyMatrix> {
    pub params: &'a <M::P as Poly>::Params,
    pub add_lhs: &'a BggPublicKey<M>,
    pub add_rhs: &'a BggPublicKey<M>,
    pub sub_lhs: &'a BggPublicKey<M>,
    pub sub_rhs: &'a BggPublicKey<M>,
    pub mul_lhs: &'a BggPublicKey<M>,
    pub mul_rhs: &'a BggPublicKey<M>,
    pub small_scalar_input: &'a BggPublicKey<M>,
    pub small_scalar: &'a [u32],
    pub large_scalar_input: &'a BggPublicKey<M>,
    pub large_scalar: &'a [BigUint],
    pub public_lut_one: &'a BggPublicKey<M>,
    pub public_lut_input: &'a BggPublicKey<M>,
    pub public_lut: &'a PublicLut<M::P>,
    pub public_lut_gate_id: GateId,
    pub public_lut_id: usize,
    pub slot_transfer_input: &'a BggPublicKey<M>,
    pub slot_transfer_src_slots: &'a [(u32, Option<u32>)],
    pub slot_transfer_gate_id: GateId,
}

pub struct BggPublicKeyBenchEstimator<M, PLE, STE>
where
    M: PolyMatrix,
    PLE: PublicLutSampleAuxBenchEstimator,
    STE: SlotTransferSampleAuxBenchEstimator,
{
    pub input_time: f64,
    pub add_time: f64,
    pub sub_time: f64,
    pub mul_time: f64,
    pub small_scalar_mul_time: f64,
    pub large_scalar_mul_time: f64,
    pub public_lut_time: f64,
    pub slot_transfer_time: f64,
    pub public_lut_estimator: PLE,
    pub slot_transfer_estimator: STE,
    _m: PhantomData<M>,
}

impl<M, PLE, STE> BggPublicKeyBenchEstimator<M, PLE, STE>
where
    M: PolyMatrix,
    PLE: PublicLutSampleAuxBenchEstimator,
    STE: SlotTransferSampleAuxBenchEstimator,
{
    pub fn benchmark<PE, SE>(
        samples: &BggPublicKeyBenchSamples<'_, M>,
        public_lut_evaluator: &PE,
        slot_transfer_evaluator: &SE,
        public_lut_estimator: PLE,
        slot_transfer_estimator: STE,
        iterations: usize,
    ) -> Self
    where
        PE: PltEvaluator<BggPublicKey<M>>,
        SE: SlotTransferEvaluator<BggPublicKey<M>>,
    {
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
            input_time: 0.0,
            add_time,
            sub_time,
            mul_time,
            small_scalar_mul_time,
            large_scalar_mul_time,
            public_lut_time,
            slot_transfer_time,
            public_lut_estimator,
            slot_transfer_estimator,
            _m: PhantomData,
        }
    }

    pub fn estimate_public_lut_sample_aux_matrices(
        &self,
        total_lut_entries: usize,
        total_lut_gates: usize,
    ) -> CircuitBenchUnitEstimate {
        let lut_entry_time = self.public_lut_estimator.sample_aux_matrices_lut_entry_time();
        let lut_gate_time = self.public_lut_estimator.sample_aux_matrices_lut_gate_time();
        CircuitBenchUnitEstimate {
            total_time: lut_entry_time.total_time * total_lut_entries as f64 +
                lut_gate_time.total_time * total_lut_gates as f64,
            latency: lut_entry_time.latency + lut_gate_time.latency,
        }
    }

    pub fn estimate_slot_transfer_sample_aux_matrices(
        &self,
        num_slots: usize,
        slot_transfer_gate_count: usize,
    ) -> CircuitBenchUnitEstimate {
        let slot_time = self.slot_transfer_estimator.sample_aux_matrices_slot_time();
        let gate_time = self.slot_transfer_estimator.sample_aux_matrices_gate_time();
        CircuitBenchUnitEstimate {
            total_time: slot_time.total_time * num_slots as f64 +
                gate_time.total_time * slot_transfer_gate_count as f64,
            latency: slot_time.latency + gate_time.latency,
        }
    }
}

impl<M, PLE, STE> BenchEstimator<BggPublicKey<M>> for BggPublicKeyBenchEstimator<M, PLE, STE>
where
    M: PolyMatrix,
    PLE: PublicLutSampleAuxBenchEstimator,
    STE: SlotTransferSampleAuxBenchEstimator,
{
    fn estimate_input(&self) -> CircuitBenchUnitEstimate {
        per_gate_time_estimate(self.input_time)
    }

    fn estimate_add(&self) -> CircuitBenchUnitEstimate {
        per_gate_time_estimate(self.add_time)
    }

    fn estimate_sub(&self) -> CircuitBenchUnitEstimate {
        per_gate_time_estimate(self.sub_time)
    }

    fn estimate_mul(&self) -> CircuitBenchUnitEstimate {
        per_gate_time_estimate(self.mul_time)
    }

    fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchUnitEstimate {
        per_gate_time_estimate(self.small_scalar_mul_time)
    }

    fn estimate_large_scalar_mul(
        &self,
        _scalar: &[num_bigint::BigUint],
    ) -> CircuitBenchUnitEstimate {
        per_gate_time_estimate(self.large_scalar_mul_time)
    }

    fn estimate_slot_transfer(&self, src_slots: &[(u32, Option<u32>)]) -> CircuitBenchUnitEstimate {
        let _ = src_slots;
        per_gate_time_estimate(self.slot_transfer_time)
    }

    fn estimate_public_lookup(&self, lut_id: usize) -> CircuitBenchUnitEstimate {
        let _ = lut_id;
        per_gate_time_estimate(self.public_lut_time)
    }
}

#[cfg(test)]
mod tests {
    use super::{BggPublicKeyBenchEstimator, BggPublicKeyBenchSamples};
    use crate::{
        __PAIR, __TestState,
        bench_estimator::{
            BenchEstimator, CircuitBenchUnitEstimate, PublicLutSampleAuxBenchEstimator,
            SlotTransferSampleAuxBenchEstimator,
        },
        bgg::public_key::BggPublicKey,
        circuit::gate::GateId,
        element::PolyElem,
        lookup::{PltEvaluator, PublicLut},
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        slot_transfer::SlotTransferEvaluator,
    };
    #[cfg(not(feature = "gpu"))]
    use crate::{
        lookup::ggh15_eval::GGH15BGGPubKeyPltEvaluator,
        sampler::{
            hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        slot_transfer::bgg_pubkey::BggPublicKeySTEvaluator,
    };
    #[cfg(not(feature = "gpu"))]
    use keccak_asm::Keccak256;
    use num_bigint::BigUint;
    use sequential_test::sequential;
    #[cfg(not(feature = "gpu"))]
    use tempfile::tempdir;

    struct DummyPublicLutEstimator;

    impl PublicLutSampleAuxBenchEstimator for DummyPublicLutEstimator {
        fn sample_aux_matrices_lut_entry_time(&self) -> CircuitBenchUnitEstimate {
            CircuitBenchUnitEstimate { latency: 3.0, total_time: 5.0 }
        }

        fn sample_aux_matrices_lut_gate_time(&self) -> CircuitBenchUnitEstimate {
            CircuitBenchUnitEstimate { latency: 7.0, total_time: 11.0 }
        }

        fn benchmark_public_lut_gate_time<R, F>(&self, _iterations: usize, _op: F) -> f64
        where
            F: FnMut() -> R,
        {
            panic!("BggPublicKeyBenchEstimator should benchmark public_lookup directly")
        }
    }

    struct DummySlotTransferEstimator;

    impl SlotTransferSampleAuxBenchEstimator for DummySlotTransferEstimator {
        fn sample_aux_matrices_slot_time(&self) -> CircuitBenchUnitEstimate {
            CircuitBenchUnitEstimate { latency: 19.0, total_time: 23.0 }
        }

        fn sample_aux_matrices_gate_time(&self) -> CircuitBenchUnitEstimate {
            CircuitBenchUnitEstimate { latency: 29.0, total_time: 31.0 }
        }

        fn benchmark_slot_transfer_gate_time<R, F>(&self, _iterations: usize, _op: F) -> f64
        where
            F: FnMut() -> R,
        {
            panic!("BggPublicKeyBenchEstimator should benchmark slot_transfer directly")
        }
    }

    struct DummyPubKeyPltEvaluator;

    impl PltEvaluator<BggPublicKey<DCRTPolyMatrix>> for DummyPubKeyPltEvaluator {
        fn public_lookup(
            &self,
            _params: &DCRTPolyParams,
            _plt: &PublicLut<DCRTPoly>,
            _one: &BggPublicKey<DCRTPolyMatrix>,
            input: &BggPublicKey<DCRTPolyMatrix>,
            _gate_id: GateId,
            _lut_id: usize,
        ) -> BggPublicKey<DCRTPolyMatrix> {
            input.clone()
        }
    }

    struct DummyPubKeySTEvaluator;

    impl SlotTransferEvaluator<BggPublicKey<DCRTPolyMatrix>> for DummyPubKeySTEvaluator {
        fn slot_transfer(
            &self,
            _params: &DCRTPolyParams,
            input: &BggPublicKey<DCRTPolyMatrix>,
            _src_slots: &[(u32, Option<u32>)],
            _gate_id: GateId,
        ) -> BggPublicKey<DCRTPolyMatrix> {
            input.clone()
        }
    }

    fn sample_pubkey(params: &DCRTPolyParams) -> BggPublicKey<DCRTPolyMatrix> {
        BggPublicKey::new(DCRTPolyMatrix::gadget_matrix(params, 1), true)
    }

    fn sample_public_lut(params: &DCRTPolyParams) -> PublicLut<DCRTPoly> {
        PublicLut::new(
            params,
            2,
            |params: &DCRTPolyParams, x| {
                Some((x, <DCRTPoly as Poly>::Elem::constant(&params.modulus(), x + 1)))
            },
            None,
        )
    }

    fn test_estimator() -> (
        BggPublicKeyBenchEstimator<
            DCRTPolyMatrix,
            DummyPublicLutEstimator,
            DummySlotTransferEstimator,
        >,
        DCRTPolyParams,
    ) {
        let params = DCRTPolyParams::default();
        let add_lhs = sample_pubkey(&params);
        let add_rhs = sample_pubkey(&params);
        let sub_lhs = sample_pubkey(&params);
        let sub_rhs = sample_pubkey(&params);
        let mul_lhs = sample_pubkey(&params);
        let mul_rhs = sample_pubkey(&params);
        let small_scalar_input = sample_pubkey(&params);
        let large_scalar_input = sample_pubkey(&params);
        let public_lut_one = sample_pubkey(&params);
        let public_lut_input = sample_pubkey(&params);
        let slot_transfer_input = sample_pubkey(&params);
        let public_lut = sample_public_lut(&params);
        let slot_transfer_src_slots = vec![(0, None), (1, Some(0))];
        let small_scalar = vec![3u32, 5u32];
        let large_scalar = vec![BigUint::from(7u32)];
        let samples = BggPublicKeyBenchSamples {
            params: &params,
            add_lhs: &add_lhs,
            add_rhs: &add_rhs,
            sub_lhs: &sub_lhs,
            sub_rhs: &sub_rhs,
            mul_lhs: &mul_lhs,
            mul_rhs: &mul_rhs,
            small_scalar_input: &small_scalar_input,
            small_scalar: &small_scalar,
            large_scalar_input: &large_scalar_input,
            large_scalar: &large_scalar,
            public_lut_one: &public_lut_one,
            public_lut_input: &public_lut_input,
            public_lut: &public_lut,
            public_lut_gate_id: GateId(11),
            public_lut_id: 7,
            slot_transfer_input: &slot_transfer_input,
            slot_transfer_src_slots: &slot_transfer_src_slots,
            slot_transfer_gate_id: GateId(13),
        };
        let estimator = BggPublicKeyBenchEstimator::benchmark(
            &samples,
            &DummyPubKeyPltEvaluator,
            &DummyPubKeySTEvaluator,
            DummyPublicLutEstimator,
            DummySlotTransferEstimator,
            1,
        );
        (estimator, params)
    }

    #[test]
    #[sequential]
    fn test_bgg_pubkey_bench_estimator_uses_requested_formulas_and_measured_gate_times() {
        let (estimator, _) = test_estimator();

        let public_lut = estimator.estimate_public_lut_sample_aux_matrices(4, 2);
        assert_eq!(public_lut, CircuitBenchUnitEstimate { total_time: 42.0, latency: 10.0 });

        let slot_transfer = estimator.estimate_slot_transfer_sample_aux_matrices(3, 2);
        assert_eq!(slot_transfer, CircuitBenchUnitEstimate { total_time: 131.0, latency: 48.0 });

        assert_eq!(
            estimator.estimate_input(),
            CircuitBenchUnitEstimate { total_time: 0.0, latency: 0.0 }
        );
        assert!(estimator.add_time >= 0.0);
        assert!(estimator.sub_time >= 0.0);
        assert!(estimator.mul_time >= 0.0);
        assert!(estimator.small_scalar_mul_time >= 0.0);
        assert!(estimator.large_scalar_mul_time >= 0.0);
        assert_eq!(estimator.estimate_add().latency, estimator.estimate_add().total_time);
        assert_eq!(estimator.estimate_sub().latency, estimator.estimate_sub().total_time);
        assert_eq!(estimator.estimate_mul().latency, estimator.estimate_mul().total_time);
        assert!(estimator.public_lut_time >= 0.0);
        assert!(estimator.slot_transfer_time >= 0.0);
        assert_eq!(
            estimator.estimate_public_lookup(7),
            CircuitBenchUnitEstimate {
                total_time: estimator.public_lut_time,
                latency: estimator.public_lut_time,
            }
        );
        assert_eq!(
            estimator.estimate_slot_transfer(&[(0, None), (1, Some(0))]),
            CircuitBenchUnitEstimate {
                total_time: estimator.slot_transfer_time,
                latency: estimator.slot_transfer_time,
            }
        );
    }

    #[cfg(not(feature = "gpu"))]
    #[test]
    #[sequential]
    fn test_real_pubkey_aux_estimators_return_nonnegative_unit_estimates() {
        let dir = tempdir().expect("temporary benchmark dir should be created");
        let plt_estimator = GGH15BGGPubKeyPltEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new([0x11u8; 32], 2, 4.578, 0.0, dir.path().to_path_buf());
        let lut_entry = plt_estimator.sample_aux_matrices_lut_entry_time();
        let lut_gate = plt_estimator.sample_aux_matrices_lut_gate_time();
        assert!(lut_entry.latency >= 0.0);
        assert_eq!(lut_entry.latency, lut_entry.total_time);
        assert!(lut_gate.latency >= 0.0);
        assert_eq!(lut_gate.latency, lut_gate.total_time);

        let st_estimator = BggPublicKeySTEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new([0x22u8; 32], 2, 2, 4.578, 0.0, dir.path().to_path_buf());
        let slot_time = st_estimator.sample_aux_matrices_slot_time();
        let gate_time = st_estimator.sample_aux_matrices_gate_time();
        assert!(slot_time.latency >= 0.0);
        assert_eq!(slot_time.latency, slot_time.total_time);
        assert!(gate_time.latency >= 0.0);
        assert_eq!(gate_time.latency, gate_time.total_time);
    }
}
