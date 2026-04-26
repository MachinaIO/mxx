use crate::{
    bench_estimator::{SampleAuxBenchEstimate, SlotTransferSampleAuxBenchEstimator},
    bgg::naive_vec::{NaiveBGGEncodingVec, NaiveBGGPublicKeyVec},
    circuit::{evaluable::Evaluable, gate::GateId},
    matrix::PolyMatrix,
    poly::PolyParams,
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

    fn slot_reduce(
        &self,
        params: &<NaiveBGGPublicKeyVec<M> as Evaluable>::Params,
        inputs: &[NaiveBGGPublicKeyVec<M>],
        num_slots: usize,
        _gate_id: GateId,
    ) -> NaiveBGGPublicKeyVec<M> {
        assert!(num_slots > 0, "slot_reduce requires num_slots > 0");
        assert!(!inputs.is_empty(), "slot_reduce requires at least one input");
        assert!(
            inputs.len() <= num_slots,
            "slot_reduce input count {} exceeds num_slots {}",
            inputs.len(),
            num_slots
        );
        let ring_dim = params.ring_dimension() as usize;
        assert!(
            num_slots <= ring_dim,
            "slot_reduce num_slots {} exceeds ring dimension {}",
            num_slots,
            ring_dim
        );

        NaiveBGGPublicKeyVec::new(
            inputs
                .par_iter()
                .enumerate()
                .map(|(input_idx, input)| {
                    assert!(
                        input.num_slots() >= num_slots,
                        "slot_reduce input {} has {} slots, expected at least {}",
                        input_idx,
                        input.num_slots(),
                        num_slots
                    );
                    let mut terms = (0..num_slots)
                        .map(|src_slot| {
                            let mut scalar = vec![0u32; ring_dim];
                            scalar[src_slot] = 1;
                            input.keys[src_slot].small_scalar_mul(params, &scalar)
                        })
                        .collect::<Vec<_>>();
                    let mut reduced = terms.drain(..1).next().expect("slot_reduce needs a term");
                    for term in terms {
                        reduced = reduced + &term;
                    }
                    reduced
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

    fn slot_reduce(
        &self,
        params: &<NaiveBGGEncodingVec<M> as Evaluable>::Params,
        inputs: &[NaiveBGGEncodingVec<M>],
        num_slots: usize,
        _gate_id: GateId,
    ) -> NaiveBGGEncodingVec<M> {
        assert!(num_slots > 0, "slot_reduce requires num_slots > 0");
        assert!(!inputs.is_empty(), "slot_reduce requires at least one input");
        assert!(
            inputs.len() <= num_slots,
            "slot_reduce input count {} exceeds num_slots {}",
            inputs.len(),
            num_slots
        );
        let ring_dim = params.ring_dimension() as usize;
        assert!(
            num_slots <= ring_dim,
            "slot_reduce num_slots {} exceeds ring dimension {}",
            num_slots,
            ring_dim
        );

        NaiveBGGEncodingVec::new(
            inputs
                .par_iter()
                .enumerate()
                .map(|(input_idx, input)| {
                    assert!(
                        input.num_slots() >= num_slots,
                        "slot_reduce input {} has {} slots, expected at least {}",
                        input_idx,
                        input.num_slots(),
                        num_slots
                    );
                    let mut terms = (0..num_slots)
                        .map(|src_slot| {
                            let mut scalar = vec![0u32; ring_dim];
                            scalar[src_slot] = 1;
                            input.encodings[src_slot].small_scalar_mul(params, &scalar)
                        })
                        .collect::<Vec<_>>();
                    let mut reduced = terms.drain(..1).next().expect("slot_reduce needs a term");
                    for term in terms {
                        reduced = reduced + &term;
                    }
                    reduced
                })
                .collect(),
        )
    }
}
