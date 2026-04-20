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
use tracing::debug;

use super::{
    BenchEstimator, CircuitBenchEstimate, benchmark_gate_operation, measure_bench_operation,
};

pub(crate) fn per_gate_time_estimate(time: f64, peak_vram: usize) -> CircuitBenchEstimate {
    CircuitBenchEstimate::new(time, time).with_peak_vram(peak_vram)
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SampleAuxBenchEstimate {
    pub total_time: f64,
    pub latency: f64,
    pub compact_bytes: BigUint,
}

impl SampleAuxBenchEstimate {
    pub fn from_chunk(latency: f64, total_chunk_count: usize, chunk_compact_bytes: usize) -> Self {
        Self::from_chunk_with_base(
            latency,
            total_chunk_count,
            chunk_compact_bytes,
            BigUint::default(),
        )
    }

    pub fn from_chunk_with_base(
        latency: f64,
        total_chunk_count: usize,
        chunk_compact_bytes: usize,
        base_compact_bytes: BigUint,
    ) -> Self {
        Self {
            total_time: latency * total_chunk_count as f64,
            latency,
            compact_bytes: base_compact_bytes +
                BigUint::from(chunk_compact_bytes) * BigUint::from(total_chunk_count),
        }
    }
}

pub trait PublicLutSampleAuxBenchEstimator<M: PolyMatrix> {
    type Params;

    fn sample_aux_matrices_lut_entry_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate;
    fn sample_aux_matrices_lut_gate_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate;
    fn write_dummy_aux_for_poly_encode_bench(
        &self,
        params: &Self::Params,
        plt: &PublicLut<M::P>,
        used_inputs: &[u64],
        lut_id: usize,
        gate_id: GateId,
        error_sigma: f64,
    );

    fn benchmark_public_lut_gate_time<R, F>(&self, iterations: usize, op: F) -> f64
    where
        F: FnMut() -> R,
    {
        measure_bench_operation(iterations, op)
    }
}

pub trait SlotTransferSampleAuxBenchEstimator<M: PolyMatrix> {
    type Params;

    fn sample_aux_matrices_slot_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate;
    fn sample_aux_matrices_gate_time(&self, params: &Self::Params) -> SampleAuxBenchEstimate;
    fn write_dummy_aux_for_poly_encode_bench(
        &self,
        params: &Self::Params,
        gate_id: GateId,
        error_sigma: f64,
    );

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
    PLE: PublicLutSampleAuxBenchEstimator<M, Params = <M::P as Poly>::Params>,
    STE: SlotTransferSampleAuxBenchEstimator<M, Params = <M::P as Poly>::Params>,
{
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
    pub public_lut_time: f64,
    pub public_lut_peak_vram: usize,
    pub slot_transfer_time: f64,
    pub slot_transfer_peak_vram: usize,
    pub public_lut_estimator: PLE,
    pub slot_transfer_estimator: STE,
    _m: PhantomData<M>,
}

impl<M, PLE, STE> BggPublicKeyBenchEstimator<M, PLE, STE>
where
    M: PolyMatrix,
    PLE: PublicLutSampleAuxBenchEstimator<M, Params = <M::P as Poly>::Params>,
    STE: SlotTransferSampleAuxBenchEstimator<M, Params = <M::P as Poly>::Params>,
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
        let add_bench =
            benchmark_gate_operation(iterations, || samples.add_lhs.clone() + samples.add_rhs);
        debug!("BggPublicKeyBenchEstimator::benchmark add_bench={:?}", add_bench);
        let sub_bench =
            benchmark_gate_operation(iterations, || samples.sub_lhs.clone() - samples.sub_rhs);
        debug!("BggPublicKeyBenchEstimator::benchmark sub_bench={:?}", sub_bench);
        let mul_bench =
            benchmark_gate_operation(iterations, || samples.mul_lhs.clone() * samples.mul_rhs);
        debug!("BggPublicKeyBenchEstimator::benchmark mul_bench={:?}", mul_bench);
        let small_scalar_mul_bench = benchmark_gate_operation(iterations, || {
            samples.small_scalar_input.small_scalar_mul(samples.params, samples.small_scalar)
        });
        debug!(
            "BggPublicKeyBenchEstimator::benchmark small_scalar_mul_bench={:?}",
            small_scalar_mul_bench
        );
        let large_scalar_mul_bench = benchmark_gate_operation(iterations, || {
            samples.large_scalar_input.large_scalar_mul(samples.params, samples.large_scalar)
        });
        debug!(
            "BggPublicKeyBenchEstimator::benchmark large_scalar_mul_bench={:?}",
            large_scalar_mul_bench
        );
        let public_lut_bench = benchmark_gate_operation(iterations, || {
            public_lut_evaluator.public_lookup(
                samples.params,
                samples.public_lut,
                samples.public_lut_one,
                samples.public_lut_input,
                samples.public_lut_gate_id,
                samples.public_lut_id,
            )
        });
        debug!("BggPublicKeyBenchEstimator::benchmark public_lut_bench={:?}", public_lut_bench);
        let slot_transfer_bench = benchmark_gate_operation(iterations, || {
            slot_transfer_evaluator.slot_transfer(
                samples.params,
                samples.slot_transfer_input,
                samples.slot_transfer_src_slots,
                samples.slot_transfer_gate_id,
            )
        });
        debug!(
            "BggPublicKeyBenchEstimator::benchmark slot_transfer_bench={:?}",
            slot_transfer_bench
        );

        Self {
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
            public_lut_time: public_lut_bench.time,
            public_lut_peak_vram: public_lut_bench.peak_vram,
            slot_transfer_time: slot_transfer_bench.time,
            slot_transfer_peak_vram: slot_transfer_bench.peak_vram,
            public_lut_estimator,
            slot_transfer_estimator,
            _m: PhantomData,
        }
    }

    pub fn estimate_public_lut_sample_aux_matrices(
        &self,
        params: &<M::P as Poly>::Params,
        total_lut_entries: usize,
        total_lut_gates: usize,
    ) -> SampleAuxBenchEstimate {
        let lut_entry_time = self.public_lut_estimator.sample_aux_matrices_lut_entry_time(params);
        let lut_gate_time = self.public_lut_estimator.sample_aux_matrices_lut_gate_time(params);
        debug!(
            "BggPublicKeyBenchEstimator::estimate_public_lut_sample_aux_matrices components: total_lut_entries={}, total_lut_gates={}, lut_entry_time={:?}, lut_gate_time={:?}",
            total_lut_entries, total_lut_gates, lut_entry_time, lut_gate_time
        );
        let estimate = SampleAuxBenchEstimate {
            total_time: lut_entry_time.total_time * total_lut_entries as f64 +
                lut_gate_time.total_time * total_lut_gates as f64,
            latency: lut_entry_time.latency + lut_gate_time.latency,
            compact_bytes: &lut_entry_time.compact_bytes * BigUint::from(total_lut_entries) +
                &lut_gate_time.compact_bytes * BigUint::from(total_lut_gates),
        };
        debug!(
            "BggPublicKeyBenchEstimator::estimate_public_lut_sample_aux_matrices estimate={:?}",
            estimate
        );
        estimate
    }

    pub fn estimate_slot_transfer_sample_aux_matrices(
        &self,
        params: &<M::P as Poly>::Params,
        num_slots: usize,
        slot_transfer_gate_count: usize,
    ) -> SampleAuxBenchEstimate {
        let slot_time = self.slot_transfer_estimator.sample_aux_matrices_slot_time(params);
        let gate_time = self.slot_transfer_estimator.sample_aux_matrices_gate_time(params);
        debug!(
            "BggPublicKeyBenchEstimator::estimate_slot_transfer_sample_aux_matrices components: num_slots={}, slot_transfer_gate_count={}, slot_time={:?}, gate_time={:?}",
            num_slots, slot_transfer_gate_count, slot_time, gate_time
        );
        let estimate = SampleAuxBenchEstimate {
            total_time: slot_time.total_time * num_slots as f64 +
                gate_time.total_time * slot_transfer_gate_count as f64,
            latency: slot_time.latency + gate_time.latency,
            compact_bytes: &slot_time.compact_bytes * BigUint::from(num_slots) +
                &gate_time.compact_bytes * BigUint::from(slot_transfer_gate_count),
        };
        debug!(
            "BggPublicKeyBenchEstimator::estimate_slot_transfer_sample_aux_matrices estimate={:?}",
            estimate
        );
        estimate
    }
}

impl<M, PLE, STE> BenchEstimator<BggPublicKey<M>> for BggPublicKeyBenchEstimator<M, PLE, STE>
where
    M: PolyMatrix,
    PLE: PublicLutSampleAuxBenchEstimator<M, Params = <M::P as Poly>::Params>,
    STE: SlotTransferSampleAuxBenchEstimator<M, Params = <M::P as Poly>::Params>,
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
        per_gate_time_estimate(self.mul_time, self.mul_peak_vram)
    }

    fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchEstimate {
        per_gate_time_estimate(self.small_scalar_mul_time, self.small_scalar_mul_peak_vram)
    }

    fn estimate_large_scalar_mul(&self, _scalar: &[num_bigint::BigUint]) -> CircuitBenchEstimate {
        per_gate_time_estimate(self.large_scalar_mul_time, self.large_scalar_mul_peak_vram)
    }

    fn estimate_slot_transfer(&self, src_slots: &[(u32, Option<u32>)]) -> CircuitBenchEstimate {
        let _ = src_slots;
        per_gate_time_estimate(self.slot_transfer_time, self.slot_transfer_peak_vram)
    }

    fn estimate_public_lookup(&self, lut_id: usize) -> CircuitBenchEstimate {
        let _ = lut_id;
        per_gate_time_estimate(self.public_lut_time, self.public_lut_peak_vram)
    }
}

#[cfg(test)]
mod tests {
    use super::BggPublicKeyBenchEstimator;
    use crate::{
        __PAIR, __TestState,
        bench_estimator::{
            BenchEstimator, CircuitBenchEstimate, PublicLutSampleAuxBenchEstimator,
            SampleAuxBenchEstimate, SlotTransferSampleAuxBenchEstimator,
        },
        circuit::gate::GateId,
        lookup::PublicLut,
        matrix::dcrt_poly::DCRTPolyMatrix,
        poly::{Poly, dcrt::poly::DCRTPoly},
    };
    #[cfg(feature = "gpu")]
    use crate::{
        lookup::ggh15_eval::GGH15BGGPubKeyPltEvaluator,
        matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
        poly::dcrt::gpu::{GpuDCRTPolyParams, gpu_device_sync},
        sampler::{
            gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
            trapdoor::GpuDCRTPolyTrapdoorSampler,
        },
        slot_transfer::bgg_pubkey::BggPublicKeySTEvaluator,
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
    #[cfg(feature = "gpu")]
    use keccak_asm::Keccak256;
    #[cfg(not(feature = "gpu"))]
    use keccak_asm::Keccak256;
    use num_bigint::BigUint;
    use sequential_test::sequential;
    use std::marker::PhantomData;
    #[cfg(any(feature = "gpu", not(feature = "gpu")))]
    use tempfile::tempdir;

    struct DummyPublicLutEstimator;

    impl PublicLutSampleAuxBenchEstimator<DCRTPolyMatrix> for DummyPublicLutEstimator {
        type Params = <DCRTPoly as Poly>::Params;

        fn sample_aux_matrices_lut_entry_time(
            &self,
            _params: &Self::Params,
        ) -> SampleAuxBenchEstimate {
            SampleAuxBenchEstimate {
                latency: 3.0,
                total_time: 5.0,
                compact_bytes: BigUint::from(7u32),
            }
        }

        fn sample_aux_matrices_lut_gate_time(
            &self,
            _params: &Self::Params,
        ) -> SampleAuxBenchEstimate {
            SampleAuxBenchEstimate {
                latency: 7.0,
                total_time: 11.0,
                compact_bytes: BigUint::from(13u32),
            }
        }

        fn write_dummy_aux_for_poly_encode_bench(
            &self,
            _params: &Self::Params,
            _plt: &PublicLut<DCRTPoly>,
            _used_inputs: &[u64],
            _lut_id: usize,
            _gate_id: GateId,
            _error_sigma: f64,
        ) {
            panic!("dummy public-lut estimator does not write synthetic checkpoints")
        }

        fn benchmark_public_lut_gate_time<R, F>(&self, _iterations: usize, _op: F) -> f64
        where
            F: FnMut() -> R,
        {
            panic!("BggPublicKeyBenchEstimator should benchmark public_lookup directly")
        }
    }

    struct DummySlotTransferEstimator;

    impl SlotTransferSampleAuxBenchEstimator<DCRTPolyMatrix> for DummySlotTransferEstimator {
        type Params = <DCRTPoly as Poly>::Params;

        fn sample_aux_matrices_slot_time(&self, _params: &Self::Params) -> SampleAuxBenchEstimate {
            SampleAuxBenchEstimate {
                latency: 19.0,
                total_time: 23.0,
                compact_bytes: BigUint::from(29u32),
            }
        }

        fn sample_aux_matrices_gate_time(&self, _params: &Self::Params) -> SampleAuxBenchEstimate {
            SampleAuxBenchEstimate {
                latency: 29.0,
                total_time: 31.0,
                compact_bytes: BigUint::from(37u32),
            }
        }

        fn write_dummy_aux_for_poly_encode_bench(
            &self,
            _params: &Self::Params,
            _gate_id: GateId,
            _error_sigma: f64,
        ) {
            panic!("dummy slot-transfer estimator does not write synthetic checkpoints")
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
            input_peak_vram: 0,
            add_time: 1.0,
            add_peak_vram: 41,
            sub_time: 2.0,
            sub_peak_vram: 43,
            mul_time: 3.0,
            mul_peak_vram: 47,
            small_scalar_mul_time: 4.0,
            small_scalar_mul_peak_vram: 53,
            large_scalar_mul_time: 5.0,
            large_scalar_mul_peak_vram: 59,
            public_lut_time: 6.0,
            public_lut_peak_vram: 61,
            slot_transfer_time: 7.0,
            slot_transfer_peak_vram: 67,
            public_lut_estimator: DummyPublicLutEstimator,
            slot_transfer_estimator: DummySlotTransferEstimator,
            _m: PhantomData,
        }
    }

    #[test]
    #[sequential]
    fn test_bgg_pubkey_bench_estimator_uses_requested_formulas_and_measured_gate_times() {
        let estimator = test_estimator();
        let params = <DCRTPoly as Poly>::Params::default();

        let public_lut = estimator.estimate_public_lut_sample_aux_matrices(&params, 4, 2);
        assert_eq!(
            public_lut,
            SampleAuxBenchEstimate {
                total_time: 42.0,
                latency: 10.0,
                compact_bytes: BigUint::from(54u32),
            }
        );

        let slot_transfer = estimator.estimate_slot_transfer_sample_aux_matrices(&params, 3, 2);
        assert_eq!(
            slot_transfer,
            SampleAuxBenchEstimate {
                total_time: 131.0,
                latency: 48.0,
                compact_bytes: BigUint::from(161u32),
            }
        );

        assert_eq!(
            estimator.estimate_input(),
            CircuitBenchEstimate::new(0.0, 0.0).with_peak_vram(0)
        );
        assert!(estimator.add_time >= 0.0);
        assert!(estimator.sub_time >= 0.0);
        assert!(estimator.mul_time >= 0.0);
        assert!(estimator.small_scalar_mul_time >= 0.0);
        assert!(estimator.large_scalar_mul_time >= 0.0);
        assert_eq!(
            estimator.estimate_add(),
            CircuitBenchEstimate::new(estimator.add_time, estimator.add_time)
                .with_peak_vram(estimator.add_peak_vram)
        );
        assert_eq!(
            estimator.estimate_sub(),
            CircuitBenchEstimate::new(estimator.sub_time, estimator.sub_time)
                .with_peak_vram(estimator.sub_peak_vram)
        );
        assert_eq!(
            estimator.estimate_mul(),
            CircuitBenchEstimate::new(estimator.mul_time, estimator.mul_time)
                .with_peak_vram(estimator.mul_peak_vram)
        );
        assert!(estimator.public_lut_time >= 0.0);
        assert!(estimator.slot_transfer_time >= 0.0);
        assert_eq!(
            estimator.estimate_public_lookup(7),
            CircuitBenchEstimate::new(estimator.public_lut_time, estimator.public_lut_time)
                .with_peak_vram(estimator.public_lut_peak_vram)
        );
        assert_eq!(
            estimator.estimate_slot_transfer(&[(0, None), (1, Some(0))]),
            CircuitBenchEstimate::new(estimator.slot_transfer_time, estimator.slot_transfer_time)
                .with_peak_vram(estimator.slot_transfer_peak_vram)
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
        let params = <DCRTPoly as Poly>::Params::default();
        let lut_entry = plt_estimator.sample_aux_matrices_lut_entry_time(&params);
        let lut_gate = plt_estimator.sample_aux_matrices_lut_gate_time(&params);
        assert!(lut_entry.latency >= 0.0);
        assert!(lut_entry.total_time >= lut_entry.latency);
        assert!(lut_entry.compact_bytes > BigUint::default());
        assert!(lut_gate.latency >= 0.0);
        assert!(lut_gate.total_time >= lut_gate.latency);
        assert!(lut_gate.compact_bytes > BigUint::default());

        let st_estimator = BggPublicKeySTEvaluator::<
            DCRTPolyMatrix,
            DCRTPolyUniformSampler,
            DCRTPolyHashSampler<Keccak256>,
            DCRTPolyTrapdoorSampler,
        >::new([0x22u8; 32], 2, 2, 4.578, 0.0, dir.path().to_path_buf());
        let slot_time = st_estimator.sample_aux_matrices_slot_time(&params);
        let gate_time = st_estimator.sample_aux_matrices_gate_time(&params);
        assert!(slot_time.latency >= 0.0);
        assert!(slot_time.total_time >= slot_time.latency);
        assert!(slot_time.compact_bytes > BigUint::default());
        assert!(gate_time.latency >= 0.0);
        assert!(gate_time.total_time >= gate_time.latency);
        assert!(gate_time.compact_bytes > BigUint::default());
    }

    #[cfg(feature = "gpu")]
    #[test]
    #[sequential]
    fn test_real_gpu_pubkey_aux_estimators_use_explicit_params() {
        gpu_device_sync();
        let dir = tempdir().expect("temporary benchmark dir should be created");
        let params = GpuDCRTPolyParams::default();
        let plt_estimator = GGH15BGGPubKeyPltEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyUniformSampler,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new([0x11u8; 32], 2, 4.578, 0.0, dir.path().to_path_buf());
        let lut_entry = plt_estimator.sample_aux_matrices_lut_entry_time(&params);
        let lut_gate = plt_estimator.sample_aux_matrices_lut_gate_time(&params);
        assert!(lut_entry.latency >= 0.0);
        assert!(lut_entry.total_time >= lut_entry.latency);
        assert!(lut_entry.compact_bytes > BigUint::default());
        assert!(lut_gate.latency >= 0.0);
        assert!(lut_gate.total_time >= lut_gate.latency);
        assert!(lut_gate.compact_bytes > BigUint::default());

        let st_estimator = BggPublicKeySTEvaluator::<
            GpuDCRTPolyMatrix,
            GpuDCRTPolyUniformSampler,
            GpuDCRTPolyHashSampler<Keccak256>,
            GpuDCRTPolyTrapdoorSampler,
        >::new([0x22u8; 32], 2, 2, 4.578, 0.0, dir.path().to_path_buf());
        let slot_time = st_estimator.sample_aux_matrices_slot_time(&params);
        let gate_time = st_estimator.sample_aux_matrices_gate_time(&params);
        assert!(slot_time.latency >= 0.0);
        assert!(slot_time.total_time >= slot_time.latency);
        assert!(slot_time.compact_bytes > BigUint::default());
        assert!(gate_time.latency >= 0.0);
        assert!(gate_time.total_time >= gate_time.latency);
        assert!(gate_time.compact_bytes > BigUint::default());
        gpu_device_sync();
    }
}
