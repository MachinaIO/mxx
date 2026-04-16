use super::{NegacyclicConvolutionContext, RingGswConvolution};
use crate::{
    circuit::{BatchedWire, PolyCircuit, SlotTransferSpec, SubCircuitParamValue, gate::GateId},
    gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext},
    poly::Poly,
};
use num_bigint::BigUint;
use rayon::prelude::*;

impl<P: Poly + 'static> NegacyclicConvolutionContext<P> for NestedRnsPolyContext {
    fn q_level_diagonal_product_param_bindings(
        &self,
        diagonal: usize,
        num_slots: usize,
    ) -> Vec<SubCircuitParamValue> {
        let rhs_binding =
            SubCircuitParamValue::SlotTransfer(SlotTransferSpec::rotation(diagonal, num_slots));
        let lhs_bindings = self
            .p_moduli
            .par_iter()
            .map(|&p_i| {
                let negative_scalar =
                    u32::try_from(p_i - 1).expect("signed slot-transfer scalar must fit in u32");
                SubCircuitParamValue::SlotTransfer(SlotTransferSpec::repeated(
                    diagonal,
                    num_slots,
                    diagonal,
                    Some(negative_scalar),
                ))
            })
            .collect::<Vec<_>>();
        let mut bindings = Vec::with_capacity(lhs_bindings.len() + 1);
        bindings.extend(lhs_bindings);
        bindings.push(rhs_binding);
        bindings
    }

    fn reduce_q_level_row(&self, row: &[GateId], circuit: &mut PolyCircuit<P>) -> Vec<GateId> {
        Self::reduce_q_level_row(self, row, circuit)
    }

    fn mul_q_level_rows(
        &self,
        left: &[GateId],
        right: &[GateId],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<GateId> {
        Self::mul_q_level_rows(self, left, right, circuit)
    }
}

impl<P: Poly + 'static> RingGswConvolution<P> for NestedRnsPoly<P> {
    fn from_diagonal_q_level_outputs(
        template: &Self,
        q_level_outputs: Vec<Vec<BatchedWire>>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        NestedRnsPoly::new(
            template.ctx.clone(),
            q_level_outputs.into_iter().map(BatchedWire::from_batches).collect::<Vec<_>>(),
            Some(template.level_offset),
            template.enable_levels,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn from_sparse_diagonal_q_level_output(
        template: &Self,
        target_q_idx: usize,
        q_level_output: Vec<BatchedWire>,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let active_levels = template.active_q_moduli().len();
        let mut inner =
            (0..active_levels).map(|_| template.ctx.zero_level_batch(circuit)).collect::<Vec<_>>();
        inner[target_q_idx] = BatchedWire::from_batches(q_level_output);
        NestedRnsPoly::new(
            template.ctx.clone(),
            inner,
            Some(template.level_offset),
            template.enable_levels,
            (0..active_levels)
                .map(|q_idx| {
                    if q_idx == target_q_idx { max_plaintext.clone() } else { BigUint::from(0u64) }
                })
                .collect::<Vec<_>>(),
        )
        .with_p_max_traces(
            (0..active_levels)
                .map(
                    |q_idx| {
                        if q_idx == target_q_idx {
                            p_max_trace.clone()
                        } else {
                            BigUint::from(0u64)
                        }
                    },
                )
                .collect::<Vec<_>>(),
        )
    }
}
