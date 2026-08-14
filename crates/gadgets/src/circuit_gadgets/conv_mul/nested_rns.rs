use super::{NegacyclicConvolutionContext, RingGswConvolution};
use crate::{
    circuit::{
        BatchedWire, PolyCircuit, SlotTransferSpec, SubCircuitParamSpec, SubCircuitParamValue,
        gate::GateId,
    },
    circuit_gadgets::arith::{NestedRnsPoly, NestedRnsPolyContext},
    poly::Poly,
};
use num_bigint::BigUint;
use rayon::prelude::*;

impl<P: Poly + 'static> NegacyclicConvolutionContext<P> for NestedRnsPolyContext {
    fn q_level_diagonal_product_param_specs(&self) -> Vec<SubCircuitParamSpec> {
        let mut specs = self
            .p_moduli
            .iter()
            .map(|&p_i| SubCircuitParamSpec::SlotTransfer {
                max_scalar: u32::try_from(p_i - 1)
                    .expect("signed slot-transfer scalar must fit in u32"),
            })
            .collect::<Vec<_>>();
        specs.push(SubCircuitParamSpec::SlotTransfer { max_scalar: 1 });
        specs
    }

    fn q_level_diagonal_product_param_bindings(
        &self,
        diagonal: usize,
        num_slots: usize,
    ) -> Vec<SubCircuitParamValue> {
        let lanes = self.q_moduli_depth;
        let rhs_binding = SubCircuitParamValue::SlotTransfer(SlotTransferSpec::rotation(
            diagonal * lanes,
            num_slots * lanes,
        ));
        let lhs_bindings = self
            .p_moduli
            .par_iter()
            .map(|&p_i| {
                let negative_scalar =
                    u32::try_from(p_i - 1).expect("signed slot-transfer scalar must fit in u32");
                SubCircuitParamValue::SlotTransfer(SlotTransferSpec::repeated_lanes(
                    diagonal,
                    num_slots,
                    lanes,
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

    fn reduce_q_level_row(
        &self,
        row: &[GateId],
        input_norms: &[BigUint],
        circuit: &mut PolyCircuit<P>,
    ) -> (Vec<GateId>, Vec<BigUint>) {
        Self::reduce_q_level_row(self, row, input_norms, circuit)
    }

    fn mul_q_level_rows(
        &self,
        left: &[GateId],
        right: &[GateId],
        left_norms: &[BigUint],
        right_norms: &[BigUint],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<GateId> {
        Self::mul_q_level_rows(self, left, right, left_norms, right_norms, circuit)
    }
}

impl<P: Poly + 'static> RingGswConvolution<P> for NestedRnsPoly<P> {
    fn physical_q_row_count(&self) -> usize {
        1
    }

    fn q_level_row_max_plaintext_norms(&self, physical_q_row: usize) -> Vec<BigUint> {
        assert_eq!(physical_q_row, 0, "packed nested-RNS has one physical q row");
        let active_levels =
            self.enable_levels.unwrap_or(self.ctx.q_moduli_depth - self.level_offset);
        let trace = self.p_max_traces[..active_levels].iter().max().cloned().unwrap_or_default();
        self.ctx.lookup_input_ranges_for_trace(&trace);
        vec![trace; self.ctx.p_moduli.len()]
    }

    fn from_diagonal_q_level_outputs(
        template: &Self,
        q_level_outputs: Vec<Vec<BatchedWire>>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        assert_eq!(q_level_outputs.len(), 1, "packed nested-RNS has one physical q row");
        NestedRnsPoly::new(
            template.ctx.clone(),
            BatchedWire::from_batches(q_level_outputs.into_iter().next().unwrap()),
            template.num_coefficient_slots,
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
        let _ = circuit;
        NestedRnsPoly::new(
            template.ctx.clone(),
            BatchedWire::from_batches(q_level_output),
            template.num_coefficient_slots,
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
