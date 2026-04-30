use crate::{
    bench_estimator::{PublicLutSampleAuxBenchEstimator, SampleAuxBenchEstimate},
    bgg::naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
    circuit::{evaluable::Evaluable, gate::GateId},
    lookup::{PltEvaluator, PublicLut},
    matrix::PolyMatrix,
    poly::Poly,
    sampler::{PolyHashSampler, PolyTrapdoorSampler},
};
use rayon::prelude::*;

use super::{LWEBGGEncodingPltEvaluator, LWEBGGPubKeyPltEvaluator};

pub struct NaiveLWEBGGPublicKeyVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
{
    pub inner: LWEBGGPubKeyPltEvaluator<M, SH, ST>,
}

impl<M, SH, ST> NaiveLWEBGGPublicKeyVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    pub fn new(inner: LWEBGGPubKeyPltEvaluator<M, SH, ST>) -> Self {
        Self { inner }
    }

    pub fn sample_aux_matrices(&self, params: &<M::P as Poly>::Params) {
        self.inner.sample_aux_matrices(params);
    }
}

impl<M, SH, ST> PltEvaluator<NaiveBGGPublicKeyVec<M>>
    for NaiveLWEBGGPublicKeyVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    fn public_lookup(
        &self,
        params: &<NaiveBGGPublicKeyVec<M> as Evaluable>::Params,
        plt: &PublicLut<<NaiveBGGPublicKeyVec<M> as Evaluable>::P>,
        one: &NaiveBGGPublicKeyVec<M>,
        input: &NaiveBGGPublicKeyVec<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> NaiveBGGPublicKeyVec<M> {
        input.assert_compatible(one);
        let num_slots = input.num_slots();
        NaiveBGGPublicKeyVec::new(
            params,
            (0..num_slots)
                .into_par_iter()
                .map(|slot_idx| {
                    let input_key = input.key(slot_idx);
                    self.inner.public_lookup_for_slot(
                        params,
                        plt,
                        &input_key,
                        gate_id,
                        lut_id,
                        Some(slot_idx),
                    )
                })
                .collect::<Vec<_>>(),
        )
    }
}

impl<M, SH, ST> PublicLutSampleAuxBenchEstimator<M>
    for NaiveLWEBGGPublicKeyVecPltEvaluator<M, SH, ST>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    ST: PolyTrapdoorSampler<M = M> + Send + Sync,
    M::P: 'static,
{
    type Params = <M::P as Poly>::Params;

    fn estimate_public_lut_sample_aux_matrices(
        &self,
        params: &Self::Params,
        total_lut_entries: usize,
        total_lut_gates: usize,
    ) -> SampleAuxBenchEstimate {
        self.inner.estimate_public_lut_sample_aux_matrices(
            params,
            total_lut_entries,
            total_lut_gates,
        )
    }

    fn write_dummy_aux_for_poly_encode_bench(
        &self,
        params: &Self::Params,
        plt: &PublicLut<M::P>,
        used_inputs: &[u64],
        lut_id: usize,
        gate_id: GateId,
        error_sigma: f64,
    ) {
        // Only the slot-0 dummy auxiliary data is needed for benchmark warmup because the naive
        // vector benchmark estimator measures a single scalar slot and scales the estimate by the
        // logical slot count.
        self.inner.write_dummy_aux_for_poly_encode_bench(
            params,
            plt,
            used_inputs,
            lut_id,
            gate_id,
            error_sigma,
        );
    }
}

#[derive(Debug, Clone)]
pub struct NaiveLWEBGGEncodingVecPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub inner: LWEBGGEncodingPltEvaluator<M, SH>,
}

impl<M, SH> NaiveLWEBGGEncodingVecPltEvaluator<M, SH>
where
    M: PolyMatrix,
    SH: PolyHashSampler<[u8; 32], M = M>,
{
    pub fn new(inner: LWEBGGEncodingPltEvaluator<M, SH>) -> Self {
        Self { inner }
    }
}

impl<M, SH> PltEvaluator<NaiveBGGEncodingVec<M>> for NaiveLWEBGGEncodingVecPltEvaluator<M, SH>
where
    M: PolyMatrix + Send + Sync + 'static,
    SH: PolyHashSampler<[u8; 32], M = M> + Send + Sync,
    M::P: 'static,
    for<'a, 'b> &'a M: std::ops::Mul<&'b M, Output = M>,
{
    fn public_lookup(
        &self,
        params: &<NaiveBGGEncodingVec<M> as Evaluable>::Params,
        plt: &PublicLut<<NaiveBGGEncodingVec<M> as Evaluable>::P>,
        one: &NaiveBGGEncodingVec<M>,
        input: &NaiveBGGEncodingVec<M>,
        gate_id: GateId,
        lut_id: usize,
    ) -> NaiveBGGEncodingVec<M> {
        input.assert_compatible(one);
        let num_slots = input.num_slots();
        NaiveBGGEncodingVec::new(
            params,
            (0..num_slots)
                .into_par_iter()
                .map(|slot_idx| {
                    let input_encoding = input.encoding(slot_idx);
                    self.inner.public_lookup_for_slot(
                        params,
                        plt,
                        &input_encoding,
                        gate_id,
                        lut_id,
                        Some(slot_idx),
                    )
                })
                .collect::<Vec<_>>(),
        )
    }
}
