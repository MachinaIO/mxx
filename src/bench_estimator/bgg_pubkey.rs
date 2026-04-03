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
    use super::BggPublicKeyBenchEstimator;
    use crate::{
        __PAIR, __TestState,
        bench_estimator::{
            BenchEstimator, CircuitBenchUnitEstimate, PublicLutSampleAuxBenchEstimator,
            SlotTransferSampleAuxBenchEstimator,
        },
        matrix::dcrt_poly::DCRTPolyMatrix,
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
    use sequential_test::sequential;
    use std::marker::PhantomData;
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

    fn test_estimator() -> BggPublicKeyBenchEstimator<
        DCRTPolyMatrix,
        DummyPublicLutEstimator,
        DummySlotTransferEstimator,
    > {
        BggPublicKeyBenchEstimator {
            input_time: 0.0,
            add_time: 1.0,
            sub_time: 2.0,
            mul_time: 3.0,
            small_scalar_mul_time: 4.0,
            large_scalar_mul_time: 5.0,
            public_lut_time: 6.0,
            slot_transfer_time: 7.0,
            public_lut_estimator: DummyPublicLutEstimator,
            slot_transfer_estimator: DummySlotTransferEstimator,
            _m: PhantomData,
        }
    }

    #[test]
    #[sequential]
    fn test_bgg_pubkey_bench_estimator_uses_requested_formulas_and_measured_gate_times() {
        let estimator = test_estimator();

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
