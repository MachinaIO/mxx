use crate::{
    bench_estimator::{
        BenchEstimator, CircuitBenchEstimate, PublicKeyAuxBenchEstimator, SampleAuxBenchEstimate,
        benchmark_gate_operation,
    },
    bgg::{
        encoding::BggEncoding,
        naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
        public_key::BggPublicKey,
    },
    matrix::PolyMatrix,
    poly::Poly,
};
use num_bigint::BigUint;

fn scale_single_slot_estimate(
    estimate: CircuitBenchEstimate,
    num_slots: usize,
) -> CircuitBenchEstimate {
    let total_time = estimate.total_time * num_slots as f64;
    let max_parallelism = estimate
        .max_parallelism
        .checked_mul(num_slots as u128)
        .expect("naive BGG vector benchmark parallelism overflowed while scaling by slot count");
    let scaled = CircuitBenchEstimate::new(total_time, estimate.latency)
        .with_max_parallelism(max_parallelism);
    #[cfg(feature = "gpu")]
    {
        // Peak VRAM is a per-wave resource estimate, not an amount of independent work, so it
        // stays equal to the single-slot measurement.
        scaled.with_peak_vram(estimate.peak_vram)
    }
    #[cfg(not(feature = "gpu"))]
    {
        scaled
    }
}

fn scale_single_slot_sample_aux(
    estimate: SampleAuxBenchEstimate,
    num_slots: usize,
) -> SampleAuxBenchEstimate {
    SampleAuxBenchEstimate {
        total_time: estimate.total_time * num_slots as f64,
        // Public LUT auxiliary matrices for different naive-vector slots are independent. With
        // enough GPUs, all slots can be sampled in the same wave, so latency stays at the scalar
        // slot latency while total work and persisted compact bytes scale by slot count.
        latency: estimate.latency,
        compact_bytes: estimate.compact_bytes * BigUint::from(num_slots),
    }
}

fn scale_single_slot_estimate_with_io(
    estimate: CircuitBenchEstimate,
    unit_count: usize,
    loads_per_unit: usize,
    stores_per_unit: usize,
    slot_load: CircuitBenchEstimate,
    slot_store: CircuitBenchEstimate,
) -> CircuitBenchEstimate {
    let unit_count_f64 = unit_count as f64;
    let loads_f64 = loads_per_unit as f64;
    let stores_f64 = stores_per_unit as f64;
    let total_time = estimate.total_time * unit_count_f64 +
        slot_load.total_time * loads_f64 * unit_count_f64 +
        slot_store.total_time * stores_f64 * unit_count_f64;
    let latency =
        estimate.latency + slot_load.latency * loads_f64 + slot_store.latency * stores_f64;
    let max_parallelism = estimate
        .max_parallelism
        .checked_mul(unit_count as u128)
        .and_then(|value| {
            value.checked_add(
                slot_load
                    .max_parallelism
                    .checked_mul(loads_per_unit as u128)?
                    .checked_mul(unit_count as u128)?,
            )
        })
        .and_then(|value| {
            value.checked_add(
                slot_store
                    .max_parallelism
                    .checked_mul(stores_per_unit as u128)?
                    .checked_mul(unit_count as u128)?,
            )
        })
        .expect("naive BGG vector benchmark parallelism overflowed while adding slot IO");
    let scaled =
        CircuitBenchEstimate::new(total_time, latency).with_max_parallelism(max_parallelism);
    #[cfg(feature = "gpu")]
    {
        scaled.with_peak_vram(estimate.peak_vram.max(slot_load.peak_vram).max(slot_store.peak_vram))
    }
    #[cfg(not(feature = "gpu"))]
    {
        scaled
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NaiveBGGVecSlotIoBenchEstimates {
    pub load: CircuitBenchEstimate,
    pub store: CircuitBenchEstimate,
}

/// Bench estimator for naive slotwise BGG vectors.
///
/// The wrapped scalar estimator is expected to measure or estimate one ordinary BGG slot. This
/// wrapper mirrors `BggPolyEncodingBenchEstimator`: latency remains the one-slot value, while
/// total work and available parallelism scale linearly with `num_slots`.
#[derive(Debug, Clone)]
pub struct NaiveBGGVecBenchEstimator<BE> {
    pub inner: BE,
    pub num_slots: usize,
    pub slot_io: NaiveBGGVecSlotIoBenchEstimates,
}

impl<BE> NaiveBGGVecBenchEstimator<BE> {
    pub fn new(inner: BE, num_slots: usize) -> Self {
        assert!(num_slots > 0, "num_slots must be positive");
        Self { inner, num_slots, slot_io: NaiveBGGVecSlotIoBenchEstimates::default() }
    }

    pub fn with_slot_io_estimates(
        inner: BE,
        num_slots: usize,
        slot_io: NaiveBGGVecSlotIoBenchEstimates,
    ) -> Self {
        assert!(num_slots > 0, "num_slots must be positive");
        Self { inner, num_slots, slot_io }
    }

    fn scale_with_io(
        &self,
        estimate: CircuitBenchEstimate,
        unit_count: usize,
        loads_per_unit: usize,
        stores_per_unit: usize,
    ) -> CircuitBenchEstimate {
        scale_single_slot_estimate_with_io(
            estimate,
            unit_count,
            loads_per_unit,
            stores_per_unit,
            self.slot_io.load,
            self.slot_io.store,
        )
    }
}

pub fn benchmark_public_key_vec_slot_io<M>(
    params: &<M::P as crate::poly::Poly>::Params,
    sample: &NaiveBGGPublicKeyVec<M>,
    iterations: usize,
) -> NaiveBGGVecSlotIoBenchEstimates
where
    M: PolyMatrix,
{
    let materialized = sample.key(0);
    let load = benchmark_gate_operation(iterations, || sample.key(0));
    let store = benchmark_gate_operation(iterations, || {
        NaiveBGGPublicKeyVec::new(params, vec![materialized.clone()])
    });
    NaiveBGGVecSlotIoBenchEstimates {
        load: CircuitBenchEstimate::new(load.time, load.time).with_peak_vram(load.peak_vram),
        store: CircuitBenchEstimate::new(store.time, store.time).with_peak_vram(store.peak_vram),
    }
}

pub fn benchmark_encoding_vec_slot_io<M>(
    params: &<M::P as crate::poly::Poly>::Params,
    sample: &NaiveBGGEncodingVec<M>,
    iterations: usize,
) -> NaiveBGGVecSlotIoBenchEstimates
where
    M: PolyMatrix,
{
    let materialized = sample.encoding(0);
    let load = benchmark_gate_operation(iterations, || sample.encoding(0));
    let store = benchmark_gate_operation(iterations, || {
        NaiveBGGEncodingVec::new(params, vec![materialized.clone()])
    });
    NaiveBGGVecSlotIoBenchEstimates {
        load: CircuitBenchEstimate::new(load.time, load.time).with_peak_vram(load.peak_vram),
        store: CircuitBenchEstimate::new(store.time, store.time).with_peak_vram(store.peak_vram),
    }
}

impl<M, BE> BenchEstimator<NaiveBGGPublicKeyVec<M>> for NaiveBGGVecBenchEstimator<BE>
where
    M: PolyMatrix,
    BE: BenchEstimator<BggPublicKey<M>>,
{
    fn estimate_input(&self) -> CircuitBenchEstimate {
        scale_single_slot_estimate(self.inner.estimate_input(), self.num_slots)
    }

    fn estimate_add(&self) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_add(), self.num_slots, 2, 1)
    }

    fn estimate_sub(&self) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_sub(), self.num_slots, 2, 1)
    }

    fn estimate_mul(&self) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_mul(), self.num_slots, 2, 1)
    }

    fn estimate_small_scalar_mul(&self, scalar: &[u32]) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_small_scalar_mul(scalar), self.num_slots, 1, 1)
    }

    fn estimate_large_scalar_mul(&self, scalar: &[BigUint]) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_large_scalar_mul(scalar), self.num_slots, 1, 1)
    }

    fn estimate_slot_transfer(&self, src_slots: &[(u32, Option<u32>)]) -> CircuitBenchEstimate {
        assert_eq!(
            src_slots.len(),
            self.num_slots,
            "NaiveBGGVecBenchEstimator::estimate_slot_transfer requires src_slots.len() == num_slots"
        );
        self.scale_with_io(self.inner.estimate_slot_transfer(&src_slots[..1]), self.num_slots, 1, 1)
    }

    fn estimate_slot_reduce(&self, input_count: usize, num_slots: usize) -> CircuitBenchEstimate {
        assert!(input_count > 0, "slot_reduce input_count must be positive");
        assert_eq!(
            num_slots, self.num_slots,
            "NaiveBGGVecBenchEstimator::estimate_slot_reduce requires num_slots == estimator num_slots"
        );
        self.scale_with_io(self.inner.estimate_slot_reduce(1, num_slots), input_count, num_slots, 1)
    }

    fn estimate_public_lookup(&self, lut_id: usize) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_public_lookup(lut_id), self.num_slots, 1, 1)
    }
}

impl<P, BE> PublicKeyAuxBenchEstimator<P> for NaiveBGGVecBenchEstimator<BE>
where
    P: Poly,
    BE: PublicKeyAuxBenchEstimator<P>,
{
    fn estimate_public_lut_sample_aux_matrices_for_circuit(
        &self,
        params: &P::Params,
        circuit: &crate::circuit::PolyCircuit<P>,
    ) -> SampleAuxBenchEstimate {
        scale_single_slot_sample_aux(
            self.inner.estimate_public_lut_sample_aux_matrices_for_circuit(params, circuit),
            self.num_slots,
        )
    }
}

impl<M, BE> BenchEstimator<NaiveBGGEncodingVec<M>> for NaiveBGGVecBenchEstimator<BE>
where
    M: PolyMatrix,
    BE: BenchEstimator<BggEncoding<M>>,
{
    fn estimate_input(&self) -> CircuitBenchEstimate {
        scale_single_slot_estimate(self.inner.estimate_input(), self.num_slots)
    }

    fn estimate_add(&self) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_add(), self.num_slots, 2, 1)
    }

    fn estimate_sub(&self) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_sub(), self.num_slots, 2, 1)
    }

    fn estimate_mul(&self) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_mul(), self.num_slots, 2, 1)
    }

    fn estimate_small_scalar_mul(&self, scalar: &[u32]) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_small_scalar_mul(scalar), self.num_slots, 1, 1)
    }

    fn estimate_large_scalar_mul(&self, scalar: &[BigUint]) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_large_scalar_mul(scalar), self.num_slots, 1, 1)
    }

    fn estimate_slot_transfer(&self, src_slots: &[(u32, Option<u32>)]) -> CircuitBenchEstimate {
        assert_eq!(
            src_slots.len(),
            self.num_slots,
            "NaiveBGGVecBenchEstimator::estimate_slot_transfer requires src_slots.len() == num_slots"
        );
        self.scale_with_io(self.inner.estimate_slot_transfer(&src_slots[..1]), self.num_slots, 1, 1)
    }

    fn estimate_slot_reduce(&self, input_count: usize, num_slots: usize) -> CircuitBenchEstimate {
        assert!(input_count > 0, "slot_reduce input_count must be positive");
        assert_eq!(
            num_slots, self.num_slots,
            "NaiveBGGVecBenchEstimator::estimate_slot_reduce requires num_slots == estimator num_slots"
        );
        self.scale_with_io(self.inner.estimate_slot_reduce(1, num_slots), input_count, num_slots, 1)
    }

    fn estimate_public_lookup(&self, lut_id: usize) -> CircuitBenchEstimate {
        self.scale_with_io(self.inner.estimate_public_lookup(lut_id), self.num_slots, 1, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::{NaiveBGGVecBenchEstimator, NaiveBGGVecSlotIoBenchEstimates};
    use crate::{
        __PAIR, __TestState,
        bench_estimator::{
            BenchEstimator, CircuitBenchEstimate, PublicLutSampleAuxBenchEstimator,
            benchmark_gate_operation,
        },
        bgg::{
            encoding::BggEncoding,
            naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
            public_key::BggPublicKey,
            sampler::{BGGEncodingSampler, BGGPublicKeySampler},
        },
        circuit::PolyGateType,
        element::PolyElem,
        lookup::{
            PltEvaluator, PublicLut,
            lwe::{
                LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator,
                NaiveLWEBGGEncodingVecPltEvaluator, NaiveLWEBGGPublicKeyVecPltEvaluator,
            },
        },
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        sampler::{
            PolyTrapdoorSampler, hash::DCRTPolyHashSampler, trapdoor::DCRTPolyTrapdoorSampler,
            uniform::DCRTPolyUniformSampler,
        },
        storage::write::{init_storage_system, storage_test_lock, wait_for_all_writes},
        utils::create_bit_random_poly,
    };
    use keccak_asm::Keccak256;
    use num_bigint::BigUint;
    use sequential_test::sequential;
    use std::{fs, path::Path, sync::Arc};

    const SIGMA: f64 = 4.578;

    #[derive(Debug, Clone)]
    struct DummyScalarBenchEstimator;

    impl BenchEstimator<BggPublicKey<DCRTPolyMatrix>> for DummyScalarBenchEstimator {
        fn estimate_input(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(1.0, 1.0).with_max_parallelism(1)
        }

        fn estimate_add(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(2.0, 2.0).with_max_parallelism(1)
        }

        fn estimate_sub(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(3.0, 3.0).with_max_parallelism(1)
        }

        fn estimate_mul(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(4.0, 4.0).with_max_parallelism(1)
        }

        fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(5.0, 5.0).with_max_parallelism(1)
        }

        fn estimate_large_scalar_mul(&self, _scalar: &[BigUint]) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(6.0, 6.0).with_max_parallelism(1)
        }

        fn estimate_slot_transfer(
            &self,
            _src_slots: &[(u32, Option<u32>)],
        ) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(7.0, 7.0).with_max_parallelism(1)
        }

        fn estimate_slot_reduce(
            &self,
            input_count: usize,
            _num_slots: usize,
        ) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(9.0 * input_count as f64, 9.0)
                .with_max_parallelism(4 * input_count as u128)
        }

        fn estimate_public_lookup(&self, _lut_id: usize) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(8.0, 8.0).with_max_parallelism(2)
        }
    }

    #[derive(Debug, Clone)]
    struct MeasuredEncodingPublicLookupEstimator {
        public_lookup_time: f64,
        public_lookup_peak_vram: usize,
    }

    impl BenchEstimator<BggEncoding<DCRTPolyMatrix>> for MeasuredEncodingPublicLookupEstimator {
        fn estimate_input(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(0.0, 0.0)
        }

        fn estimate_add(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(0.0, 0.0)
        }

        fn estimate_sub(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(0.0, 0.0)
        }

        fn estimate_mul(&self) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(0.0, 0.0)
        }

        fn estimate_small_scalar_mul(&self, _scalar: &[u32]) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(0.0, 0.0)
        }

        fn estimate_large_scalar_mul(&self, _scalar: &[BigUint]) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(0.0, 0.0)
        }

        fn estimate_slot_transfer(
            &self,
            _src_slots: &[(u32, Option<u32>)],
        ) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(0.0, 0.0)
        }

        fn estimate_slot_reduce(
            &self,
            _input_count: usize,
            _num_slots: usize,
        ) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(0.0, 0.0)
        }

        fn estimate_public_lookup(&self, _lut_id: usize) -> CircuitBenchEstimate {
            CircuitBenchEstimate::new(self.public_lookup_time, self.public_lookup_time)
                .with_peak_vram(self.public_lookup_peak_vram)
        }
    }

    fn lsb_lut(params: &DCRTPolyParams) -> PublicLut<DCRTPoly> {
        PublicLut::new(
            params,
            16,
            |params: &DCRTPolyParams, input| {
                Some((input & 1, <DCRTPoly as Poly>::Elem::constant(&params.modulus(), input & 1)))
            },
            Some((1, <DCRTPoly as Poly>::Elem::constant(&params.modulus(), 1))),
        )
    }

    fn prepare_clean_storage(dir_path: &str) {
        let dir = Path::new(dir_path);
        if dir.exists() {
            fs::remove_dir_all(dir).unwrap();
        }
        fs::create_dir_all(dir).unwrap();
        init_storage_system(dir.to_path_buf());
    }

    #[test]
    fn test_naive_bgg_vec_bench_estimator_scales_total_time_and_parallelism_by_slots() {
        let estimator = NaiveBGGVecBenchEstimator::new(DummyScalarBenchEstimator, 3);

        let public_lookup = <NaiveBGGVecBenchEstimator<_> as BenchEstimator<
            NaiveBGGPublicKeyVec<DCRTPolyMatrix>,
        >>::estimate_public_lookup(&estimator, 0);
        assert_eq!(public_lookup.latency, 8.0);
        assert_eq!(public_lookup.total_time, 24.0);
        assert_eq!(public_lookup.max_parallelism, 6);

        let slot_transfer = <NaiveBGGVecBenchEstimator<_> as BenchEstimator<
            NaiveBGGPublicKeyVec<DCRTPolyMatrix>,
        >>::estimate_slot_transfer(
            &estimator, &[(0, None), (1, None), (2, Some(3))]
        );
        assert_eq!(slot_transfer.latency, 7.0);
        assert_eq!(slot_transfer.total_time, 21.0);
        assert_eq!(slot_transfer.max_parallelism, 3);

        let slot_reduce = <NaiveBGGVecBenchEstimator<_> as BenchEstimator<
            NaiveBGGPublicKeyVec<DCRTPolyMatrix>,
        >>::estimate_slot_reduce(&estimator, 2, 3);
        assert_eq!(slot_reduce.latency, 9.0);
        assert_eq!(slot_reduce.total_time, 18.0);
        assert_eq!(slot_reduce.max_parallelism, 8);

        let routed_slot_reduce = <NaiveBGGVecBenchEstimator<_> as BenchEstimator<
            NaiveBGGPublicKeyVec<DCRTPolyMatrix>,
        >>::estimate_gate_bench(
            &estimator,
            &PolyGateType::SlotReduce { num_slots: 3, input_count: 2 },
        );
        assert_eq!(routed_slot_reduce.latency, 9.0);
        assert_eq!(routed_slot_reduce.total_time, 18.0);
        assert_eq!(routed_slot_reduce.max_parallelism, 8);
    }

    #[test]
    fn test_naive_bgg_vec_bench_estimator_adds_slot_load_and_store_time() {
        let slot_io = NaiveBGGVecSlotIoBenchEstimates {
            load: CircuitBenchEstimate::new(0.5, 0.5).with_max_parallelism(1),
            store: CircuitBenchEstimate::new(0.25, 0.25).with_max_parallelism(1),
        };
        let estimator = NaiveBGGVecBenchEstimator::with_slot_io_estimates(
            DummyScalarBenchEstimator,
            3,
            slot_io,
        );

        let add = <NaiveBGGVecBenchEstimator<_> as BenchEstimator<
            NaiveBGGPublicKeyVec<DCRTPolyMatrix>,
        >>::estimate_add(&estimator);
        assert_eq!(add.latency, 3.25);
        assert_eq!(add.total_time, 9.75);
        assert_eq!(add.max_parallelism, 12);

        let public_lookup = <NaiveBGGVecBenchEstimator<_> as BenchEstimator<
            NaiveBGGPublicKeyVec<DCRTPolyMatrix>,
        >>::estimate_public_lookup(&estimator, 0);
        assert_eq!(public_lookup.latency, 8.75);
        assert_eq!(public_lookup.total_time, 26.25);
        assert_eq!(public_lookup.max_parallelism, 12);

        let slot_reduce = <NaiveBGGVecBenchEstimator<_> as BenchEstimator<
            NaiveBGGPublicKeyVec<DCRTPolyMatrix>,
        >>::estimate_slot_reduce(&estimator, 2, 3);
        assert_eq!(slot_reduce.latency, 10.75);
        assert_eq!(slot_reduce.total_time, 21.5);
        assert_eq!(slot_reduce.max_parallelism, 16);
    }

    #[tokio::test]
    #[sequential]
    async fn test_naive_encoding_public_lut_estimate_uses_slot_zero_dummy_aux() {
        let _storage_lock = storage_test_lock().await;
        let params = DCRTPolyParams::default();
        let plt = lsb_lut(&params);
        let hash_key = [0x57u8; 32];
        let d = 2;
        let num_slots = 3;
        let dir_path = "test_data/test_naive_encoding_public_lut_estimate";
        prepare_clean_storage(dir_path);

        let pubkey_sampler =
            BGGPublicKeySampler::<_, DCRTPolyHashSampler<Keccak256>>::new(hash_key, d);
        let secrets = vec![create_bit_random_poly(&params); d];
        let plaintexts = vec![
            DCRTPoly::from_usize_to_constant(&params, 1),
            DCRTPoly::from_usize_to_constant(&params, 5),
        ];
        let pubkeys = pubkey_sampler.sample(&params, b"naive-bench-lwe-plt", &[true, true]);
        let encoding_sampler =
            BGGEncodingSampler::<DCRTPolyUniformSampler>::new(&params, &secrets, None);
        let encodings = encoding_sampler.sample(&params, &pubkeys, &plaintexts);

        let trapdoor_sampler = DCRTPolyTrapdoorSampler::new(&params, SIGMA);
        let (b_trapdoor, b) = trapdoor_sampler.trapdoor(&params, d);
        let secret_vec = DCRTPolyMatrix::from_poly_vec_row(&params, secrets);
        let c_b = secret_vec * &b;
        let pubkey_evaluator =
            NaiveLWEBGGPublicKeyVecPltEvaluator::new(LWEBGGPubKeyPltEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyHashSampler<Keccak256>,
                _,
            >::new(
                hash_key,
                trapdoor_sampler,
                Arc::new(b),
                Arc::new(b_trapdoor),
                dir_path.into(),
            ));

        pubkey_evaluator.write_dummy_aux_for_poly_encode_bench(
            &params,
            &plt,
            &[plaintexts[1].const_coeff_u64()],
            0,
            crate::circuit::gate::GateId(11),
            0.0,
        );
        wait_for_all_writes(Path::new(dir_path).to_path_buf()).await.unwrap();

        let encoding_evaluator =
            NaiveLWEBGGEncodingVecPltEvaluator::new(LWEBGGEncodingPltEvaluator::<
                DCRTPolyMatrix,
                DCRTPolyHashSampler<Keccak256>,
            >::new(
                hash_key, dir_path.into(), c_b
            ));
        let one = NaiveBGGEncodingVec::new(&params, vec![encodings[0].clone()]);
        let input = NaiveBGGEncodingVec::new(&params, vec![encodings[1].clone()]);
        let public_lookup_bench = benchmark_gate_operation(1, || {
            encoding_evaluator.public_lookup(
                &params,
                &plt,
                &one,
                &input,
                crate::circuit::gate::GateId(11),
                0,
            )
        });
        assert!(public_lookup_bench.time >= 0.0);

        let scalar_estimator = MeasuredEncodingPublicLookupEstimator {
            public_lookup_time: public_lookup_bench.time,
            public_lookup_peak_vram: public_lookup_bench.peak_vram,
        };
        let estimator = NaiveBGGVecBenchEstimator::new(scalar_estimator, num_slots);
        let estimate = <NaiveBGGVecBenchEstimator<_> as BenchEstimator<
            NaiveBGGEncodingVec<DCRTPolyMatrix>,
        >>::estimate_public_lookup(&estimator, 0);
        assert_eq!(estimate.latency, public_lookup_bench.time);
        assert_eq!(estimate.total_time, public_lookup_bench.time * num_slots as f64);
        #[cfg(feature = "gpu")]
        assert_eq!(estimate.peak_vram, public_lookup_bench.peak_vram);
    }
}
