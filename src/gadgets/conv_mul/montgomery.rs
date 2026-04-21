// Intentionally left without an active Montgomery implementation for
// `NegacyclicConvolutionContext`; see the commented block below.

/* impl<P: Poly + 'static> NegacyclicConvolutionContext<P> for MontgomeryPolyContext<P> {
    fn q_level_diagonal_product_param_bindings(
        &self,
        diagonal: usize,
        num_slots: usize,
    ) -> Vec<SubCircuitParamValue> {
        let rhs_binding =
            SubCircuitParamValue::SlotTransfer(SlotTransferSpec::rotation(diagonal, num_slots));
        let negative_scalar = u32::try_from(self.modulus_u64.saturating_sub(1))
            .expect("Montgomery modulus must fit in u32 for negacyclic slot-transfer scalars");
        let lhs_bindings = (0..self.num_limbs)
            .into_par_iter()
            .map(|_| {
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

    fn reduce_q_level_row(&self, row: &[GateId], _circuit: &mut PolyCircuit<P>) -> Vec<GateId> {
        row.to_vec()
    }

    fn mul_q_level_rows(
        &self,
        left: &[GateId],
        right: &[GateId],
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<GateId> {
        assert_eq!(
            left.len(),
            self.num_limbs,
            "Montgomery q-level rows must have one wire per limb"
        );
        assert_eq!(
            right.len(),
            self.num_limbs,
            "Montgomery q-level rows must have one wire per limb"
        );
        let lhs = MontgomeryPoly::new(
            Arc::new(self.clone()),
            CarryArithPoly::new(self.carry_arith_ctx.clone(), left.to_vec()),
        );
        let rhs = MontgomeryPoly::new(
            Arc::new(self.clone()),
            CarryArithPoly::new(self.carry_arith_ctx.clone(), right.to_vec()),
        );
        lhs.mul(&rhs, circuit).value.limbs
    }
} */
