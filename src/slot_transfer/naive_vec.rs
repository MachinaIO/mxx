use crate::{
    bench_estimator::{SampleAuxBenchEstimate, SlotTransferSampleAuxBenchEstimator},
    bgg::naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
    circuit::{evaluable::Evaluable, gate::GateId},
    matrix::PolyMatrix,
    slot_transfer::SlotTransferEvaluator,
};
use rayon::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct NaiveBGGVecSlotTransferEvaluator;

impl NaiveBGGVecSlotTransferEvaluator {
    pub fn new() -> Self {
        Self
    }
}

impl<M: PolyMatrix> SlotTransferSampleAuxBenchEstimator<M> for NaiveBGGVecSlotTransferEvaluator {
    type Params = <M::P as crate::poly::Poly>::Params;

    fn sample_aux_matrices_slot_time(&self, _params: &Self::Params) -> SampleAuxBenchEstimate {
        SampleAuxBenchEstimate::default()
    }

    fn sample_aux_matrices_gate_time(&self, _params: &Self::Params) -> SampleAuxBenchEstimate {
        SampleAuxBenchEstimate::default()
    }

    fn write_dummy_aux_for_poly_encode_bench(
        &self,
        _params: &Self::Params,
        _gate_id: GateId,
        _error_sigma: f64,
    ) {
        // Naive slot transfer only shuffles already materialized per-slot BGG values, so it has no
        // auxiliary matrices to sample for benchmark warmup.
    }
}

impl<M: PolyMatrix> SlotTransferEvaluator<NaiveBGGPublicKeyVec<M>>
    for NaiveBGGVecSlotTransferEvaluator
{
    fn slot_transfer(
        &self,
        params: &<NaiveBGGPublicKeyVec<M> as Evaluable>::Params,
        input: &NaiveBGGPublicKeyVec<M>,
        src_slots: &[(u32, Option<u32>)],
        _gate_id: GateId,
    ) -> NaiveBGGPublicKeyVec<M> {
        NaiveBGGPublicKeyVec::new(
            src_slots
                .par_iter()
                .map(|(src_slot, scalar)| {
                    let key = input
                        .keys
                        .get(*src_slot as usize)
                        .unwrap_or_else(|| panic!("source slot {} out of range", src_slot));
                    match scalar {
                        Some(scalar) => key.small_scalar_mul(params, &[*scalar]),
                        None => key.clone(),
                    }
                })
                .collect(),
        )
    }
}

impl<M: PolyMatrix> SlotTransferEvaluator<NaiveBGGEncodingVec<M>>
    for NaiveBGGVecSlotTransferEvaluator
{
    fn slot_transfer(
        &self,
        params: &<NaiveBGGEncodingVec<M> as Evaluable>::Params,
        input: &NaiveBGGEncodingVec<M>,
        src_slots: &[(u32, Option<u32>)],
        _gate_id: GateId,
    ) -> NaiveBGGEncodingVec<M> {
        NaiveBGGEncodingVec::new(
            src_slots
                .par_iter()
                .map(|(src_slot, scalar)| {
                    let encoding = input
                        .encodings
                        .get(*src_slot as usize)
                        .unwrap_or_else(|| panic!("source slot {} out of range", src_slot));
                    match scalar {
                        Some(scalar) => encoding.small_scalar_mul(params, &[*scalar]),
                        None => encoding.clone(),
                    }
                })
                .collect(),
        )
    }
}
