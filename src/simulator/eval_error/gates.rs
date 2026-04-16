use super::*;

impl PolyCircuit<DCRTPoly> {
    /// Evaluate the full circuit with concrete `ErrorNorm` inputs.
    ///
    /// This is the user-facing entry point for the extracted simulator. It creates the canonical
    /// "one" error and input error values from the simulator context, then delegates to
    /// `eval_max_error_norm`, which performs the actual layered propagation.
    pub fn simulate_max_error_norm<P: AffinePltEvaluator>(
        &self,
        ctx: Arc<SimulatorContext>,
        input_norm_bound: BigDecimal,
        input_size: usize,
        e_init_norm: &BigDecimal,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
    ) -> Vec<ErrorNorm>
    where
        ErrorNorm: Evaluable<P = DCRTPoly>,
    {
        let one_error = ErrorNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
        );
        let input_error = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_norm_bound.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
        );
        info!("e_init_norm bits {}", bigdecimal_bits_ceil(e_init_norm));
        info!("input_norm_bound bits {}", bigdecimal_bits_ceil(&input_norm_bound));
        let input_errors = vec![input_error; input_size];
        self.eval_max_error_norm(one_error, input_errors, plt_evaluator, slot_transfer_evaluator)
    }

    /// Evaluate a regular gate whose behavior is fully determined by the input error norms.
    ///
    /// These gates do not need LUT- or slot-transfer-specific evaluator hooks. The helper is split
    /// out so `eval_max_error_norm` can batch pure arithmetic gates separately from
    /// evaluator-backed gates inside the same execution layer.
    pub(super) fn eval_pure_regular_error_norm_gate(
        &self,
        gate_id: GateId,
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
    ) -> ErrorNorm {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::Add => {
                let left = self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                let right = self.clone_error_norm_value_for_gate(
                    gate.input_gates[1],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                left + &right
            }
            PolyGateType::Sub => {
                let left = self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                let right = self.clone_error_norm_value_for_gate(
                    gate.input_gates[1],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                left - &right
            }
            PolyGateType::Mul => {
                let left = self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                let right = self.clone_error_norm_value_for_gate(
                    gate.input_gates[1],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                left * &right
            }
            PolyGateType::SmallScalarMul { scalar } => {
                let scalar = scalar.resolve_small_scalar(&[]);
                self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                )
                .small_scalar_mul(&(), scalar)
            }
            PolyGateType::LargeScalarMul { scalar } => {
                let scalar = scalar.resolve_large_scalar(&[]);
                self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                )
                .large_scalar_mul(&(), scalar)
            }
            _ => panic!("gate {gate_id} is not pure in error-norm evaluation"),
        }
    }

    /// Evaluate a regular gate that may need an external affine evaluator.
    ///
    /// Public lookups and slot transfers use domain-specific bound logic that lives outside the
    /// generic `Evaluable` implementation. All other regular gates fall back to
    /// `eval_pure_regular_error_norm_gate`.
    pub(super) fn eval_evaluator_regular_error_norm_gate<P: AffinePltEvaluator>(
        &self,
        gate_id: GateId,
        one_error: &ErrorNorm,
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
    ) -> ErrorNorm {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::SlotTransfer { src_slots } => {
                let src_slots = src_slots.resolve_slot_transfer(&[]);
                let input = self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                slot_transfer_evaluator.expect("slot transfer evaluator missing").slot_transfer(
                    &(),
                    &input,
                    src_slots.as_ref(),
                    gate_id,
                )
            }
            PolyGateType::PubLut { lut_id } => {
                let lut_id = lut_id.resolve_public_lookup(&[]);
                let input = self.clone_error_norm_value_for_gate(
                    gate.input_gates[0],
                    wires,
                    input_gate_positions,
                    input_values,
                );
                let lookup = self.lookup_table(lut_id);
                plt_evaluator.expect("public lookup evaluator missing").public_lookup(
                    &(),
                    lookup.as_ref(),
                    one_error,
                    &input,
                    gate_id,
                    lut_id,
                )
            }
            _ => self.eval_pure_regular_error_norm_gate(
                gate_id,
                wires,
                input_gate_positions,
                input_values,
            ),
        }
    }

    /// Build the reverse lookup from input gate id to the corresponding `inputs` index.
    ///
    /// The concrete evaluator and the summary builder both preload input values separately from the
    /// temporary wire store, so they need a cheap way to tell whether a gate id refers to a circuit
    /// input or to an already-materialized internal wire.
    pub(super) fn error_norm_input_gate_positions(
        &self,
        input_gate_ids: &[GateId],
    ) -> ErrorNormInputGatePositions {
        let mut positions = HashMap::with_capacity(input_gate_ids.len());
        for (idx, gate_id) in input_gate_ids.iter().copied().enumerate() {
            positions.insert(gate_id, idx);
        }
        positions
    }

    /// Clone one concrete `ErrorNorm` either from temporary wire storage or from the original
    /// inputs.
    pub(super) fn clone_error_norm_value_for_gate(
        &self,
        gate_id: GateId,
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
    ) -> ErrorNorm {
        if let Some(value) = wires.get(gate_id) {
            return value.as_ref().clone();
        }
        if let Some(&input_idx) = input_gate_positions.get(&gate_id) {
            return input_values[input_idx]
                .as_ref()
                .unwrap_or_else(|| panic!("error-norm value missing for input gate {gate_id}"))
                .as_ref()
                .clone();
        }
        panic!("error-norm value missing for gate {gate_id}")
    }

    /// Clone only the matrix-norm component for one gate.
    ///
    /// Summary construction often needs just the matrix half of an `ErrorNorm`, so this helper
    /// keeps that access pattern explicit and avoids rebuilding the full value object in
    /// callers.
    pub(super) fn clone_error_norm_matrix_norm_for_gate(
        &self,
        gate_id: GateId,
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
    ) -> PolyMatrixNorm {
        if let Some(value) = wires.get(gate_id) {
            return value.matrix_norm.clone();
        }
        if let Some(&input_idx) = input_gate_positions.get(&gate_id) {
            return input_values[input_idx]
                .as_ref()
                .unwrap_or_else(|| {
                    panic!("error-norm matrix norm missing for input gate {gate_id}")
                })
                .matrix_norm
                .clone();
        }
        panic!("error-norm matrix norm missing for gate {gate_id}")
    }

    /// Clone only the plaintext-norm component for one concrete gate value.
    pub(super) fn clone_error_norm_plaintext_norm_for_value_gate(
        &self,
        gate_id: GateId,
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
    ) -> PolyNorm {
        if let Some(value) = wires.get(gate_id) {
            return value.plaintext_norm.clone();
        }
        if let Some(&input_idx) = input_gate_positions.get(&gate_id) {
            return input_values[input_idx]
                .as_ref()
                .unwrap_or_else(|| panic!("error-norm value missing for input gate {gate_id}"))
                .plaintext_norm
                .clone();
        }
        panic!("error-norm value missing for gate {gate_id}")
    }

    /// Materialize plaintext norms for an input-set used by a sub-circuit call.
    ///
    /// Shared-prefix sub-circuit calls reuse the same input-set across many call sites. This helper
    /// produces the flat plaintext profile that later becomes part of the summary-cache key.
    pub(super) fn collect_error_norm_plaintext_norms_for_value_input_set(
        &self,
        input_ids: &[BatchedWire],
        wires: &ErrorNormWireStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_values: &[Option<Arc<ErrorNorm>>],
    ) -> Vec<PolyNorm> {
        input_ids
            .par_iter()
            .flat_map_iter(|batch| batch.gate_ids())
            .map(|input_id| {
                self.clone_error_norm_plaintext_norm_for_value_gate(
                    input_id,
                    wires,
                    input_gate_positions,
                    input_values,
                )
            })
            .collect::<Vec<_>>()
    }

    pub(super) fn build_pure_regular_error_norm_expr(
        &self,
        gate_id: GateId,
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        param_bindings: &[SubCircuitParamValue],
        plt_evaluator: Option<&impl AffinePltEvaluator>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        node: &ErrorNormSummaryNode,
    ) -> ErrorNormSummaryExpr {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::Add | PolyGateType::Sub => {
                let left = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                let right = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[1],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                left.as_ref().add_bound(right.as_ref())
            }
            PolyGateType::Mul => {
                let left = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                let right = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[1],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                left.as_ref().mul_bound(right.as_ref())
            }
            PolyGateType::SmallScalarMul { scalar } => {
                let scalar = scalar.resolve_small_scalar(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                input.as_ref().small_scalar_mul_bound(scalar)
            }
            PolyGateType::LargeScalarMul { scalar } => {
                let scalar = scalar.resolve_large_scalar(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                input.as_ref().large_scalar_mul_bound(scalar)
            }
            _ => panic!("gate {gate_id} is not pure in error-norm summary eval"),
        }
    }

    pub(super) fn build_evaluator_regular_error_norm_expr<P: AffinePltEvaluator>(
        &self,
        gate_id: GateId,
        gate_exprs: &ErrorNormExprStore,
        input_gate_positions: &ErrorNormInputGatePositions,
        input_plaintext_norms: &[PolyNorm],
        one_error: &ErrorNorm,
        param_bindings: &[SubCircuitParamValue],
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        summary_cache: &ErrorNormSubCircuitSummaryCache,
        node: &ErrorNormSummaryNode,
    ) -> ErrorNormSummaryExpr {
        let gate = self.gate(gate_id);
        match &gate.gate_type {
            PolyGateType::SlotTransfer { src_slots } => {
                let src_slots = src_slots.resolve_slot_transfer(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                slot_transfer_evaluator
                    .expect("slot transfer evaluator missing")
                    .slot_transfer_affine(input.as_ref(), src_slots.as_ref(), gate_id)
            }
            PolyGateType::PubLut { lut_id } => {
                let lut_id = lut_id.resolve_public_lookup(param_bindings);
                let input = self.clone_error_norm_summary_expr_for_summary_gate_direct(
                    gate.input_gates[0],
                    gate_exprs,
                    input_gate_positions,
                    input_plaintext_norms,
                    one_error,
                    plt_evaluator,
                    slot_transfer_evaluator,
                    summary_cache,
                    node,
                );
                let lookup = self.lookup_table(lut_id);
                plt_evaluator.expect("public lookup evaluator missing").public_lookup_affine(
                    input.as_ref(),
                    lookup.as_ref(),
                    gate_id,
                    lut_id,
                )
            }
            _ => self.build_pure_regular_error_norm_expr(
                gate_id,
                gate_exprs,
                input_gate_positions,
                input_plaintext_norms,
                one_error,
                param_bindings,
                plt_evaluator,
                slot_transfer_evaluator,
                summary_cache,
                node,
            ),
        }
    }
}
