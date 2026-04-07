use crate::{
    circuit::{Evaluable, PolyCircuit, PolyGateType, gate::GateId},
    element::PolyElem,
    impl_binop_with_refs,
    lookup::{PltEvaluator, PublicLut, commit_eval::compute_padded_len},
    poly::dcrt::poly::DCRTPoly,
    simulator::{SimulatorContext, poly_matrix_norm::PolyMatrixNorm, poly_norm::PolyNorm},
    slot_transfer::SlotTransferEvaluator,
    utils::bigdecimal_bits_ceil,
};
use bigdecimal::BigDecimal;
use num_bigint::BigUint;
use num_traits::{FromPrimitive, One};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::{HashMap, HashSet},
    ops::{Add, Mul, Sub},
    sync::Arc,
};
use tracing::{debug, info};

fn gaussian_tail_bound_factor() -> BigDecimal {
    BigDecimal::from_f32(6.5).unwrap()
}

fn merge_optional_matrix_norm(
    left: Option<PolyMatrixNorm>,
    right: Option<PolyMatrixNorm>,
) -> Option<PolyMatrixNorm> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        (Some(left), None) => Some(left),
        (None, Some(right)) => Some(right),
        (None, None) => None,
    }
}

fn merge_gate_use_counts(
    mut left: HashMap<GateId, usize>,
    right: HashMap<GateId, usize>,
) -> HashMap<GateId, usize> {
    for (gate_id, count) in right {
        *left.entry(gate_id).or_insert(0) += count;
    }
    left
}

fn error_norm_plaintext_profile_bits(profile: &[PolyNorm]) -> Vec<u64> {
    profile.iter().map(|norm| bigdecimal_bits_ceil(&norm.norm)).collect()
}

impl PolyCircuit<DCRTPoly> {
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

    fn eval_max_error_norm<P: AffinePltEvaluator>(
        &self,
        one_error: ErrorNorm,
        inputs: Vec<ErrorNorm>,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
    ) -> Vec<ErrorNorm> {
        let input_gate_ids = self.sorted_input_gate_ids();
        assert_eq!(
            input_gate_ids.len(),
            inputs.len(),
            "number of provided inputs must match circuit inputs"
        );

        let output_set = self.output_gate_ids().iter().copied().collect::<HashSet<_>>();
        let mut remaining_use_count = self.error_norm_remaining_use_count();
        let mut wires = HashMap::new();
        wires.insert(GateId(0), one_error.clone());
        for (gate_id, input) in input_gate_ids.into_iter().zip(inputs.into_iter()) {
            wires.insert(gate_id, input);
        }

        let mut sub_circuit_summaries: HashMap<usize, Vec<ErrorNormSubCircuitSummaryRecord>> =
            HashMap::new();
        let mut cache_stats = ErrorNormSubCircuitCacheStats::default();
        for (gate_id, gate) in self.gates_in_id_order() {
            let gate_id = *gate_id;
            if wires.contains_key(&gate_id) {
                continue;
            }
            let value = match &gate.gate_type {
                PolyGateType::Input => {
                    panic!("input gate {gate_id} should already be preloaded");
                }
                PolyGateType::Add => {
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Add").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Add").clone();
                    left + &right
                }
                PolyGateType::Sub => {
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Sub").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Sub").clone();
                    left - &right
                }
                PolyGateType::Mul => {
                    let left =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Mul").clone();
                    let right =
                        wires.get(&gate.input_gates[1]).expect("wire missing for Mul").clone();
                    left * &right
                }
                PolyGateType::SmallScalarMul { scalar } => wires
                    .get(&gate.input_gates[0])
                    .expect("wire missing for SmallScalarMul")
                    .small_scalar_mul(&(), scalar),
                PolyGateType::LargeScalarMul { scalar } => wires
                    .get(&gate.input_gates[0])
                    .expect("wire missing for LargeScalarMul")
                    .large_scalar_mul(&(), scalar),
                PolyGateType::SlotTransfer { src_slots } => {
                    let input =
                        wires.get(&gate.input_gates[0]).expect("wire missing for SlotTransfer");
                    slot_transfer_evaluator.expect("slot transfer evaluator missing").slot_transfer(
                        &(),
                        input,
                        src_slots,
                        gate_id,
                    )
                }
                PolyGateType::PubLut { lut_id } => {
                    let input =
                        wires.get(&gate.input_gates[0]).expect("wire missing for Public Lookup");
                    let lookup = self.lookup_table(*lut_id);
                    plt_evaluator.expect("public lookup evaluator missing").public_lookup(
                        &(),
                        lookup.as_ref(),
                        &one_error,
                        input,
                        gate_id,
                        *lut_id,
                    )
                }
                PolyGateType::SubCircuitOutput { call_id, .. } => {
                    let call = self.sub_circuit_call_info(*call_id);
                    let sub_inputs = call
                        .inputs
                        .iter()
                        .map(|input_id| {
                            wires.get(input_id).expect("wire missing for sub-circuit input").clone()
                        })
                        .collect::<Vec<_>>();
                    let input_exprs = sub_inputs
                        .iter()
                        .enumerate()
                        .map(|(idx, input)| ErrorNormSummaryExpr::input(idx, input))
                        .collect::<Vec<_>>();
                    let summary_idx = self.ensure_cached_sub_circuit_summary(
                        call.sub_circuit_id,
                        &input_exprs,
                        &one_error,
                        &mut sub_circuit_summaries,
                        &mut cache_stats,
                        plt_evaluator,
                        slot_transfer_evaluator,
                    );
                    let sub_outputs = sub_circuit_summaries
                        .get(&call.sub_circuit_id)
                        .expect("error-norm sub-circuit summary cache missing")[summary_idx]
                        .summary
                        .evaluate(&sub_inputs);
                    for (output_gate_id, output) in
                        call.output_gate_ids.iter().copied().zip(sub_outputs.into_iter())
                    {
                        wires.insert(output_gate_id, output);
                    }
                    self.release_error_norm_inputs(
                        &call.inputs,
                        &output_set,
                        &mut remaining_use_count,
                        &mut wires,
                    );
                    wires.get(&gate_id).expect("sub-circuit output should be populated").clone()
                }
            };
            if !matches!(gate.gate_type, PolyGateType::SubCircuitOutput { .. }) {
                wires.insert(gate_id, value);
                self.release_error_norm_inputs(
                    &gate.input_gates,
                    &output_set,
                    &mut remaining_use_count,
                    &mut wires,
                );
            }
        }

        debug!(
            "error-norm sub-circuit summary cache totals hits={} misses={} cached_subcircuits={}",
            cache_stats.hits,
            cache_stats.misses,
            sub_circuit_summaries.len()
        );

        self.output_gate_ids()
            .par_iter()
            .map(|gate_id| wires.get(gate_id).expect("missing error-norm output wire").clone())
            .collect()
    }

    fn build_sub_circuit_summary<P: AffinePltEvaluator>(
        &self,
        one_error: &ErrorNorm,
        input_exprs: &[ErrorNormSummaryExpr],
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
        cache_stats: &mut ErrorNormSubCircuitCacheStats,
    ) -> ErrorNormSubCircuitSummary {
        let input_gate_ids = self.sorted_input_gate_ids();
        let mut gate_exprs = HashMap::new();
        gate_exprs.insert(GateId(0), ErrorNormSummaryExpr::constant(one_error.clone()));
        for (gate_id, input_expr) in input_gate_ids.iter().copied().zip(input_exprs.iter().cloned())
        {
            gate_exprs.insert(gate_id, input_expr);
        }

        let mut child_summaries: HashMap<usize, Vec<ErrorNormSubCircuitSummaryRecord>> =
            HashMap::new();
        for (gate_id, gate) in self.gates_in_id_order() {
            let gate_id = *gate_id;
            if gate_exprs.contains_key(&gate_id) {
                continue;
            }
            let expr = match &gate.gate_type {
                PolyGateType::Input => {
                    panic!(
                        "input gate {gate_id} should already be present in the error-norm summary"
                    )
                }
                PolyGateType::Add | PolyGateType::Sub => {
                    let left = gate_exprs
                        .get(&gate.input_gates[0])
                        .expect("left error-norm expression missing");
                    let right = gate_exprs
                        .get(&gate.input_gates[1])
                        .expect("right error-norm expression missing");
                    left.add_bound(right)
                }
                PolyGateType::Mul => {
                    let left = gate_exprs
                        .get(&gate.input_gates[0])
                        .expect("left error-norm expression missing");
                    let right = gate_exprs
                        .get(&gate.input_gates[1])
                        .expect("right error-norm expression missing");
                    left.mul_bound(right)
                }
                PolyGateType::SmallScalarMul { scalar } => {
                    let input = gate_exprs
                        .get(&gate.input_gates[0])
                        .expect("error-norm expression missing for SmallScalarMul");
                    input.small_scalar_mul_bound(scalar)
                }
                PolyGateType::LargeScalarMul { scalar } => {
                    let input = gate_exprs
                        .get(&gate.input_gates[0])
                        .expect("error-norm expression missing for LargeScalarMul");
                    input.large_scalar_mul_bound(scalar)
                }
                PolyGateType::SlotTransfer { src_slots } => {
                    let input = gate_exprs
                        .get(&gate.input_gates[0])
                        .expect("error-norm expression missing for SlotTransfer");
                    slot_transfer_evaluator
                        .expect("slot transfer evaluator missing")
                        .slot_transfer_affine(input, src_slots, gate_id)
                }
                PolyGateType::PubLut { lut_id } => {
                    let input = gate_exprs
                        .get(&gate.input_gates[0])
                        .expect("error-norm expression missing for Public Lookup");
                    let lookup = self.lookup_table(*lut_id);
                    plt_evaluator.expect("public lookup evaluator missing").public_lookup_affine(
                        input,
                        lookup.as_ref(),
                        gate_id,
                        *lut_id,
                    )
                }
                PolyGateType::SubCircuitOutput { call_id, .. } => {
                    let call = self.sub_circuit_call_info(*call_id);
                    let actual_inputs = call
                        .inputs
                        .iter()
                        .map(|input_id| {
                            gate_exprs
                                .get(input_id)
                                .expect("error-norm expression missing for sub-circuit input")
                                .clone()
                        })
                        .collect::<Vec<_>>();
                    let summary_idx = self.ensure_cached_sub_circuit_summary(
                        call.sub_circuit_id,
                        &actual_inputs,
                        one_error,
                        &mut child_summaries,
                        cache_stats,
                        plt_evaluator,
                        slot_transfer_evaluator,
                    );
                    let sub_outputs = child_summaries
                        .get(&call.sub_circuit_id)
                        .expect("error-norm child sub-circuit summary cache missing")[summary_idx]
                        .summary
                        .substitute(&actual_inputs);
                    for (output_gate_id, output_expr) in
                        call.output_gate_ids.iter().copied().zip(sub_outputs.into_iter())
                    {
                        gate_exprs.insert(output_gate_id, output_expr);
                    }
                    gate_exprs
                        .get(&gate_id)
                        .expect("error-norm sub-circuit output should be populated")
                        .clone()
                }
            };
            if !matches!(gate.gate_type, PolyGateType::SubCircuitOutput { .. }) {
                gate_exprs.insert(gate_id, expr);
            }
        }

        let output_exprs = self
            .output_gate_ids()
            .par_iter()
            .map(|output_id| {
                gate_exprs
                    .get(output_id)
                    .expect("missing error-norm expression for sub-circuit output")
                    .clone()
            })
            .collect::<Vec<_>>();
        ErrorNormSubCircuitSummary { output_exprs }
    }

    fn ensure_cached_sub_circuit_summary<P: AffinePltEvaluator>(
        &self,
        sub_circuit_id: usize,
        input_exprs: &[ErrorNormSummaryExpr],
        one_error: &ErrorNorm,
        summary_cache: &mut HashMap<usize, Vec<ErrorNormSubCircuitSummaryRecord>>,
        cache_stats: &mut ErrorNormSubCircuitCacheStats,
        plt_evaluator: Option<&P>,
        slot_transfer_evaluator: Option<&dyn AffineSlotTransferEvaluator>,
    ) -> usize {
        let input_plaintext_norms =
            input_exprs.iter().map(|input| input.plaintext_norm.clone()).collect::<Vec<_>>();
        let profile_bits = error_norm_plaintext_profile_bits(&input_plaintext_norms);
        let entries = summary_cache.entry(sub_circuit_id).or_default();
        if let Some(summary_idx) =
            entries.iter().position(|entry| entry.input_plaintext_norms == input_plaintext_norms)
        {
            cache_stats.hits += 1;
            debug!(
                "error-norm sub-circuit summary cache hit sub_circuit_id={} profile_bits={:?} cached_profiles={}",
                sub_circuit_id,
                profile_bits,
                entries.len()
            );
            return summary_idx;
        }

        cache_stats.misses += 1;
        let miss_kind = if entries.is_empty() { "first_call" } else { "new_plaintext_profile" };
        debug!(
            "error-norm sub-circuit summary cache miss sub_circuit_id={} miss_kind={} profile_bits={:?} cached_profiles_before={}",
            sub_circuit_id,
            miss_kind,
            profile_bits,
            entries.len()
        );

        let canonical_input_exprs = input_plaintext_norms
            .iter()
            .cloned()
            .enumerate()
            .map(|(idx, plaintext_norm)| {
                ErrorNormSummaryExpr::input_with_plaintext_norm(idx, plaintext_norm)
            })
            .collect::<Vec<_>>();
        let summary = self.registered_sub_circuit(sub_circuit_id).build_sub_circuit_summary(
            one_error,
            &canonical_input_exprs,
            plt_evaluator,
            slot_transfer_evaluator,
            cache_stats,
        );
        entries.push(ErrorNormSubCircuitSummaryRecord { input_plaintext_norms, summary });
        debug!(
            "error-norm sub-circuit summary cache store sub_circuit_id={} miss_kind={} cached_profiles_after={}",
            sub_circuit_id,
            miss_kind,
            entries.len()
        );
        entries.len() - 1
    }

    fn error_norm_remaining_use_count(&self) -> HashMap<GateId, usize> {
        let mut use_count = self
            .gates_in_id_order()
            .collect::<Vec<_>>()
            .par_iter()
            .filter(|(_, gate)| !matches!(gate.gate_type, PolyGateType::SubCircuitOutput { .. }))
            .fold(HashMap::new, |mut local, (_, gate)| {
                for input_id in &gate.input_gates {
                    *local.entry(*input_id).or_insert(0) += 1;
                }
                local
            })
            .reduce(HashMap::new, merge_gate_use_counts);
        let mut seen_call_ids = HashSet::new();
        for (_, gate) in self.gates_in_id_order() {
            let PolyGateType::SubCircuitOutput { call_id, .. } = gate.gate_type else {
                continue;
            };
            if !seen_call_ids.insert(call_id) {
                continue;
            }
            let call = self.sub_circuit_call_info(call_id);
            for input_id in call.inputs {
                *use_count.entry(input_id).or_insert(0) += 1;
            }
        }
        use_count
    }

    fn release_error_norm_inputs(
        &self,
        input_ids: &[GateId],
        output_set: &HashSet<GateId>,
        remaining_use_count: &mut HashMap<GateId, usize>,
        wires: &mut HashMap<GateId, ErrorNorm>,
    ) {
        for input_id in input_ids {
            if output_set.contains(input_id) {
                continue;
            }
            let Some(remaining) = remaining_use_count.get_mut(input_id) else {
                continue;
            };
            debug_assert!(
                *remaining > 0,
                "remaining error-norm use counter underflow for gate {input_id}"
            );
            *remaining -= 1;
            if *remaining == 0 {
                remaining_use_count.remove(input_id);
                wires.remove(input_id);
            }
        }
    }
}

// Note: h_norm and plaintext_norm computed here can be larger than the modulus `q`.
// In such a case, the error after circuit evaluation could be too large.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErrorNorm {
    pub plaintext_norm: PolyNorm,
    pub matrix_norm: PolyMatrixNorm,
}

impl ErrorNorm {
    pub fn new(plaintext_norm: PolyNorm, matrix_norm: PolyMatrixNorm) -> Self {
        debug_assert_eq!(plaintext_norm.ctx, matrix_norm.clone_ctx());
        Self { plaintext_norm, matrix_norm }
    }

    #[inline]
    pub fn ctx(&self) -> &SimulatorContext {
        &self.plaintext_norm.ctx
    }
    #[inline]
    pub fn clone_ctx(&self) -> Arc<SimulatorContext> {
        self.plaintext_norm.ctx.clone()
    }
}

impl_binop_with_refs!(ErrorNorm => Add::add(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm {
        plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
        matrix_norm: &self.matrix_norm + &rhs.matrix_norm
    }
});

// Note: norm of the subtraction result is bounded by a sum of the norms of the input matrices,
// i.e., |A-B| < |A| + |B|
impl_binop_with_refs!(ErrorNorm => Sub::sub(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm {
        plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
        matrix_norm: &self.matrix_norm + &rhs.matrix_norm
    }
});

impl_binop_with_refs!(ErrorNorm => Mul::mul(self, rhs: &ErrorNorm) -> ErrorNorm {
    debug_assert_eq!(self.ctx(), rhs.ctx());
    ErrorNorm {
        plaintext_norm: &self.plaintext_norm * &rhs.plaintext_norm,
        matrix_norm: &self.matrix_norm * PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().m_g) + rhs.matrix_norm.clone() * &self.plaintext_norm
    }
});

impl Evaluable for ErrorNorm {
    type Params = ();
    type P = DCRTPoly;
    type Compact = ErrorNorm;

    fn to_compact(self) -> Self::Compact {
        self
    }

    fn from_compact(_: &Self::Params, compact: &Self::Compact) -> Self {
        compact.clone()
    }

    fn small_scalar_mul(&self, _: &Self::Params, scalar: &[u32]) -> Self {
        let scalar_max = BigDecimal::from(*scalar.iter().max().unwrap());
        let scalar_poly = PolyNorm::new(self.clone_ctx(), scalar_max);
        ErrorNorm {
            matrix_norm: self.matrix_norm.clone() * &scalar_poly,
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }

    fn large_scalar_mul(&self, _: &Self::Params, scalar: &[BigUint]) -> Self {
        let scalar_max = scalar.iter().max().unwrap().clone();
        let scalar_bd = BigDecimal::from(num_bigint::BigInt::from(scalar_max));
        let scalar_poly = PolyNorm::new(self.clone_ctx(), scalar_bd);
        ErrorNorm {
            matrix_norm: self.matrix_norm.clone() *
                PolyMatrixNorm::gadget_decomposed(self.clone_ctx(), self.ctx().m_g),
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
        }
    }
}

pub trait AffinePltEvaluator: PltEvaluator<ErrorNorm> {
    fn public_lookup_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        plt: &PublicLut<DCRTPoly>,
        gate_id: GateId,
        lut_id: usize,
    ) -> ErrorNormSummaryExpr;
}

pub trait AffineSlotTransferEvaluator: SlotTransferEvaluator<ErrorNorm> {
    fn slot_transfer_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        src_slots: &[(u32, Option<u32>)],
        gate_id: GateId,
    ) -> ErrorNormSummaryExpr;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AffineErrorNormTerm {
    input_idx: usize,
    pending_scalar: BigDecimal,
    pending_poly: Option<PolyNorm>,
    matrix_factor: Option<PolyMatrixNorm>,
}

impl AffineErrorNormTerm {
    fn input(input_idx: usize) -> Self {
        Self {
            input_idx,
            pending_scalar: BigDecimal::one(),
            pending_poly: None,
            matrix_factor: None,
        }
    }

    fn transform_scalar_in_place(&mut self, scalar: &BigDecimal) {
        if let Some(matrix_factor) = &mut self.matrix_factor {
            *matrix_factor = matrix_factor.clone() * scalar;
        } else {
            self.pending_scalar *= scalar;
        }
    }

    fn transform_poly_in_place(&mut self, poly: &PolyNorm) {
        if let Some(matrix_factor) = &mut self.matrix_factor {
            *matrix_factor = matrix_factor.clone() * poly;
        } else {
            self.pending_poly = Some(match self.pending_poly.take() {
                Some(existing) => existing * poly.clone(),
                None => poly.clone(),
            });
        }
    }

    fn transform_matrix_in_place(&mut self, matrix: &PolyMatrixNorm) {
        let mut matrix = matrix.clone();
        if self.pending_scalar != BigDecimal::one() {
            matrix = matrix * &self.pending_scalar;
            self.pending_scalar = BigDecimal::one();
        }
        if let Some(poly) = self.pending_poly.take() {
            matrix = poly * matrix;
        }
        self.matrix_factor = Some(match self.matrix_factor.take() {
            Some(existing) => existing * matrix,
            None => matrix,
        });
    }

    fn split_cols_left_in_place(&mut self, left_col_size: usize) {
        let matrix_factor = self
            .matrix_factor
            .take()
            .expect("split_cols requires a matrix factor in the affine error-norm term");
        self.matrix_factor = Some(matrix_factor.split_cols(left_col_size).0);
    }

    fn compose_after(&self, outer: &AffineErrorNormTerm) -> Self {
        let mut composed = self.clone();
        composed.transform_scalar_in_place(&outer.pending_scalar);
        if let Some(poly) = &outer.pending_poly {
            composed.transform_poly_in_place(poly);
        }
        if let Some(matrix) = &outer.matrix_factor {
            composed.transform_matrix_in_place(matrix);
        }
        composed
    }

    fn apply_to_matrix(&self, input: &PolyMatrixNorm) -> PolyMatrixNorm {
        let mut value = input.clone();
        if self.pending_scalar != BigDecimal::one() {
            value = value * &self.pending_scalar;
        }
        if let Some(poly) = &self.pending_poly {
            value = value * poly;
        }
        if let Some(matrix) = &self.matrix_factor {
            value *= matrix.clone();
        }
        value
    }

    fn apply_to_const(&self, const_term: Option<&PolyMatrixNorm>) -> Option<PolyMatrixNorm> {
        let mut value = const_term.cloned()?;
        if self.pending_scalar != BigDecimal::one() {
            value = value * &self.pending_scalar;
        }
        if let Some(poly) = &self.pending_poly {
            value = value * poly;
        }
        if let Some(matrix) = &self.matrix_factor {
            value *= matrix.clone();
        }
        Some(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AffineErrorNormExpr {
    const_term: Option<PolyMatrixNorm>,
    input_terms: Vec<AffineErrorNormTerm>,
}

impl AffineErrorNormExpr {
    fn zero() -> Self {
        Self { const_term: None, input_terms: vec![] }
    }

    fn constant(const_term: PolyMatrixNorm) -> Self {
        Self { const_term: Some(const_term), input_terms: vec![] }
    }

    fn input(input_idx: usize) -> Self {
        Self { const_term: None, input_terms: vec![AffineErrorNormTerm::input(input_idx)] }
    }

    fn add_expr(&self, rhs: &Self) -> Self {
        let const_term = match (&self.const_term, &rhs.const_term) {
            (Some(left), Some(right)) => Some(left + right),
            (Some(left), None) => Some(left.clone()),
            (None, Some(right)) => Some(right.clone()),
            (None, None) => None,
        };
        let mut input_terms = self.input_terms.clone();
        input_terms.extend(rhs.input_terms.clone());
        Self { const_term, input_terms }
    }

    fn transform_scalar(&self, scalar: &BigDecimal) -> Self {
        let const_term = self.const_term.clone().map(|const_term| const_term * scalar);
        let input_terms = self
            .input_terms
            .par_iter()
            .map(|term| {
                let mut term = term.clone();
                term.transform_scalar_in_place(scalar);
                term
            })
            .collect();
        Self { const_term, input_terms }
    }

    fn transform_poly(&self, poly: &PolyNorm) -> Self {
        let const_term = self.const_term.clone().map(|const_term| const_term * poly);
        let input_terms = self
            .input_terms
            .par_iter()
            .map(|term| {
                let mut term = term.clone();
                term.transform_poly_in_place(poly);
                term
            })
            .collect();
        Self { const_term, input_terms }
    }

    fn transform_matrix(&self, matrix: &PolyMatrixNorm) -> Self {
        let const_term = self.const_term.clone().map(|const_term| const_term * matrix.clone());
        let input_terms = self
            .input_terms
            .par_iter()
            .map(|term| {
                let mut term = term.clone();
                term.transform_matrix_in_place(matrix);
                term
            })
            .collect();
        Self { const_term, input_terms }
    }

    fn split_cols_left(&self, left_col_size: usize) -> Self {
        let const_term =
            self.const_term.clone().map(|const_term| const_term.split_cols(left_col_size).0);
        let input_terms = self
            .input_terms
            .par_iter()
            .map(|term| {
                let mut term = term.clone();
                term.split_cols_left_in_place(left_col_size);
                term
            })
            .collect();
        Self { const_term, input_terms }
    }

    fn substitute_inputs(&self, actual_inputs: &[AffineErrorNormExpr]) -> Self {
        self.input_terms
            .par_iter()
            .map(|term| {
                let input_expr = actual_inputs
                    .get(term.input_idx)
                    .expect("affine error-norm input index out of range during substitution");
                input_expr.apply_outer_transform(term)
            })
            .reduce(Self::zero, |left, right| left.add_expr(&right))
            .add_expr(&Self { const_term: self.const_term.clone(), input_terms: vec![] })
    }

    fn apply_outer_transform(&self, transform: &AffineErrorNormTerm) -> Self {
        let const_term = transform.apply_to_const(self.const_term.as_ref());
        let input_terms =
            self.input_terms.par_iter().map(|term| term.compose_after(transform)).collect();
        Self { const_term, input_terms }
    }

    fn evaluate(&self, inputs: &[ErrorNorm]) -> PolyMatrixNorm {
        merge_optional_matrix_norm(
            self.const_term.clone(),
            self.input_terms
                .par_iter()
                .map(|term| {
                    let input = inputs
                        .get(term.input_idx)
                        .expect("affine error-norm input index out of range during evaluation");
                    Some(term.apply_to_matrix(&input.matrix_norm))
                })
                .reduce(|| None, merge_optional_matrix_norm),
        )
        .expect("affine error-norm expression must produce a matrix norm")
    }
}

#[derive(Debug, Clone)]
pub struct ErrorNormSummaryExpr {
    plaintext_norm: PolyNorm,
    matrix_expr: AffineErrorNormExpr,
}

impl ErrorNormSummaryExpr {
    fn constant(value: ErrorNorm) -> Self {
        Self {
            plaintext_norm: value.plaintext_norm,
            matrix_expr: AffineErrorNormExpr::constant(value.matrix_norm),
        }
    }

    fn input(input_idx: usize, input: &ErrorNorm) -> Self {
        Self::input_with_plaintext_norm(input_idx, input.plaintext_norm.clone())
    }

    fn input_with_plaintext_norm(input_idx: usize, plaintext_norm: PolyNorm) -> Self {
        Self { plaintext_norm, matrix_expr: AffineErrorNormExpr::input(input_idx) }
    }

    fn add_bound(&self, rhs: &Self) -> Self {
        Self {
            plaintext_norm: &self.plaintext_norm + &rhs.plaintext_norm,
            matrix_expr: self.matrix_expr.add_expr(&rhs.matrix_expr),
        }
    }

    fn mul_bound(&self, rhs: &Self) -> Self {
        Self {
            plaintext_norm: &self.plaintext_norm * &rhs.plaintext_norm,
            matrix_expr: self
                .matrix_expr
                .transform_matrix(&PolyMatrixNorm::gadget_decomposed(
                    self.plaintext_norm.ctx.clone(),
                    self.plaintext_norm.ctx.m_g,
                ))
                .add_expr(&rhs.matrix_expr.transform_poly(&self.plaintext_norm)),
        }
    }

    fn small_scalar_mul_bound(&self, scalar: &[u32]) -> Self {
        let scalar_max = BigDecimal::from(*scalar.iter().max().unwrap());
        let scalar_poly = PolyNorm::new(self.plaintext_norm.ctx.clone(), scalar_max.clone());
        Self {
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
            matrix_expr: self.matrix_expr.transform_scalar(&scalar_max),
        }
    }

    fn large_scalar_mul_bound(&self, scalar: &[BigUint]) -> Self {
        let scalar_max = scalar.iter().max().unwrap().clone();
        let scalar_bd = BigDecimal::from(num_bigint::BigInt::from(scalar_max));
        let scalar_poly = PolyNorm::new(self.plaintext_norm.ctx.clone(), scalar_bd);
        Self {
            plaintext_norm: self.plaintext_norm.clone() * &scalar_poly,
            matrix_expr: self.matrix_expr.transform_matrix(&PolyMatrixNorm::gadget_decomposed(
                self.plaintext_norm.ctx.clone(),
                self.plaintext_norm.ctx.m_g,
            )),
        }
    }

    fn substitute_inputs(&self, actual_inputs: &[ErrorNormSummaryExpr]) -> Self {
        let actual_matrix_exprs =
            actual_inputs.iter().map(|input| input.matrix_expr.clone()).collect::<Vec<_>>();
        Self {
            plaintext_norm: self.plaintext_norm.clone(),
            matrix_expr: self.matrix_expr.substitute_inputs(&actual_matrix_exprs),
        }
    }

    fn evaluate(&self, inputs: &[ErrorNorm]) -> ErrorNorm {
        ErrorNorm::new(self.plaintext_norm.clone(), self.matrix_expr.evaluate(inputs))
    }
}

#[derive(Debug, Clone)]
struct ErrorNormSubCircuitSummary {
    output_exprs: Vec<ErrorNormSummaryExpr>,
}

impl ErrorNormSubCircuitSummary {
    fn substitute(&self, actual_inputs: &[ErrorNormSummaryExpr]) -> Vec<ErrorNormSummaryExpr> {
        self.output_exprs.par_iter().map(|expr| expr.substitute_inputs(actual_inputs)).collect()
    }

    fn evaluate(&self, inputs: &[ErrorNorm]) -> Vec<ErrorNorm> {
        self.output_exprs.par_iter().map(|expr| expr.evaluate(inputs)).collect()
    }
}

#[derive(Debug, Clone)]
struct ErrorNormSubCircuitSummaryRecord {
    input_plaintext_norms: Vec<PolyNorm>,
    summary: ErrorNormSubCircuitSummary,
}

#[derive(Debug, Default, Clone)]
struct ErrorNormSubCircuitCacheStats {
    hits: usize,
    misses: usize,
}

#[derive(Debug, Clone)]
pub struct NormBggPolyEncodingSTEvaluator {
    pub const_term: PolyMatrixNorm,
    pub transfer_plaintext_multiplier: PolyMatrixNorm,
    pub input_vector_multiplier: PolyMatrixNorm,
}

impl NormBggPolyEncodingSTEvaluator {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        e_b0_sigma: f64,
        e_mat_sigma: &BigDecimal,
        secret_sigma: Option<BigDecimal>,
    ) -> Self {
        let c_b0_error_norm = PolyMatrixNorm::sample_gauss(
            ctx.clone(),
            1,
            ctx.m_b,
            BigDecimal::from_f64(e_b0_sigma).expect("e_b0_sigma must be finite"),
        );

        let b0_preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(1));
        info!(
            "{}",
            format!(
                "BGG poly-encoding slot-transfer preimage norm bits {}",
                bigdecimal_bits_ceil(&b0_preimage_norm)
            )
        );

        let matrix_norm_bits = |m: &PolyMatrixNorm| bigdecimal_bits_ceil(&m.poly_norm.norm);
        let log_matrix_norm_bits = |name: &str, m: &PolyMatrixNorm| {
            debug!(
                "NormBggPolyEncodingSTEvaluator::new {} matrix norm bits {}",
                name,
                matrix_norm_bits(m)
            );
        };
        log_matrix_norm_bits("c_b0_error_norm", &c_b0_error_norm);
        let s_vec = PolyMatrixNorm::new(
            ctx.clone(),
            1,
            ctx.secret_size,
            secret_sigma.unwrap_or(BigDecimal::one()),
            None,
        );
        log_matrix_norm_bits("s_vec", &s_vec);

        // `c_b0 * gate_preimage` with `B0 * gate_preimage = target + error`.
        let gate_preimage =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, b0_preimage_norm.clone(), None);
        log_matrix_norm_bits("gate_preimage", &gate_preimage);
        let gaussian_bound = gaussian_tail_bound_factor();
        let gate_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        log_matrix_norm_bits("gate_target_error", &gate_target_error);
        let gate_target_error_term = s_vec.clone() * &gate_target_error;
        log_matrix_norm_bits("gate_target_error_term", &gate_target_error_term);
        let c_b0_gate_term = c_b0_error_norm.clone() * &gate_preimage;
        log_matrix_norm_bits("c_b0_gate_term", &c_b0_gate_term);
        let const_term = &gate_target_error_term + &c_b0_gate_term;
        log_matrix_norm_bits("const_term", &const_term);

        // `((c_b0 * slot_preimage_b0) * slot_preimage_b1) * plaintext`.
        let slot_preimage_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, 2 * ctx.m_b, b0_preimage_norm.clone(), None);
        log_matrix_norm_bits("slot_preimage_b0", &slot_preimage_b0);
        let b1_preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(2));
        // `preimage_b1` targets the `B1` basis, whose trapdoor size is `2 * secret_size`.
        let slot_preimage_b1 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b * 2, ctx.m_g, b1_preimage_norm.clone(), None);
        log_matrix_norm_bits("slot_preimage_b1", &slot_preimage_b1);
        let slot_preimage_b0_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_b * 2,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        log_matrix_norm_bits("slot_preimage_b0_target_error", &slot_preimage_b0_target_error);
        let slot_preimage_b1_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size * 2,
            ctx.m_g,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        log_matrix_norm_bits("slot_preimage_b1_target_error", &slot_preimage_b1_target_error);
        let slot_secret_and_identity = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.secret_size * 2,
            BigDecimal::one(),
            None,
        );
        log_matrix_norm_bits("slot_secret_and_identity", &slot_secret_and_identity);
        let slot_stage1_error_term =
            s_vec.clone() * slot_secret_and_identity * slot_preimage_b1_target_error;
        log_matrix_norm_bits("slot_stage1_error_term", &slot_stage1_error_term);
        let slot_stage0_error_term =
            s_vec.clone() * slot_preimage_b0_target_error * slot_preimage_b1.clone();
        log_matrix_norm_bits("slot_stage0_error_term", &slot_stage0_error_term);
        let c_b0_transfer_term = c_b0_error_norm * slot_preimage_b0 * slot_preimage_b1;
        log_matrix_norm_bits("c_b0_transfer_term", &c_b0_transfer_term);
        let transfer_plaintext_multiplier = slot_stage1_error_term.clone() +
            slot_stage0_error_term.clone() +
            c_b0_transfer_term.clone();
        log_matrix_norm_bits("transfer_plaintext_multiplier", &transfer_plaintext_multiplier);

        // `input_vector * slot_a.decompose()`.
        let input_vector_multiplier = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        log_matrix_norm_bits("input_vector_multiplier", &input_vector_multiplier);

        info!("BGG poly-encoding slot-transfer const term bits {}", matrix_norm_bits(&const_term));
        info!(
            "BGG poly-encoding slot-transfer plaintext multiplier bits {}",
            matrix_norm_bits(&transfer_plaintext_multiplier)
        );
        info!(
            "BGG poly-encoding slot-transfer input multiplier bits {}",
            matrix_norm_bits(&input_vector_multiplier)
        );

        Self { const_term, transfer_plaintext_multiplier, input_vector_multiplier }
    }
}

impl SlotTransferEvaluator<ErrorNorm> for NormBggPolyEncodingSTEvaluator {
    fn slot_transfer(
        &self,
        _: &(),
        input: &ErrorNorm,
        src_slots: &[(u32, Option<u32>)],
        _: GateId,
    ) -> ErrorNorm {
        let scalar_max =
            src_slots.iter().map(|(_, scalar)| u64::from(scalar.unwrap_or(1))).max().unwrap_or(1);
        let scalar_bd = BigDecimal::from(scalar_max);
        let plaintext_norm = input.plaintext_norm.clone() * &scalar_bd;
        let input_vector_term =
            (input.matrix_norm.clone() * &self.input_vector_multiplier) * &scalar_bd;
        let transfer_plaintext_term = self.transfer_plaintext_multiplier.clone() * &plaintext_norm;
        let matrix_norm = &self.const_term + &input_vector_term + &transfer_plaintext_term;
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffineSlotTransferEvaluator for NormBggPolyEncodingSTEvaluator {
    fn slot_transfer_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        src_slots: &[(u32, Option<u32>)],
        _: GateId,
    ) -> ErrorNormSummaryExpr {
        let scalar_max =
            src_slots.iter().map(|(_, scalar)| u64::from(scalar.unwrap_or(1))).max().unwrap_or(1);
        let scalar_bd = BigDecimal::from(scalar_max);
        let plaintext_norm = input.plaintext_norm.clone() * &scalar_bd;
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&self.input_vector_multiplier)
            .transform_scalar(&scalar_bd)
            .add_expr(&AffineErrorNormExpr::constant(
                &self.const_term + &(self.transfer_plaintext_multiplier.clone() * &plaintext_norm),
            ));
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}

#[derive(Debug, Clone)]
pub struct NormPltLWEEvaluator {
    pub e_b_times_preimage: PolyMatrixNorm,
    pub preimage_lower: PolyMatrixNorm,
}

impl NormPltLWEEvaluator {
    pub fn new(ctx: Arc<SimulatorContext>, e_b_sigma: &BigDecimal) -> Self {
        let norm = compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None);
        let norm_bits = bigdecimal_bits_ceil(&norm);
        info!("{}", format!("preimage norm bits {}", norm_bits));
        let e_b_init = PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, e_b_sigma * 6, None);
        let e_b_times_preimage =
            &e_b_init * &PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, norm.clone(), None);
        let preimage_lower = PolyMatrixNorm::new(ctx.clone(), ctx.m_g, ctx.m_g, norm.clone(), None);
        info!(
            "LWE PLT const term norm bits {}",
            bigdecimal_bits_ceil(&e_b_times_preimage.poly_norm.norm)
        );
        info!(
            "LWE PLT e_input multiplier norm bits {}",
            bigdecimal_bits_ceil(&preimage_lower.poly_norm.norm)
        );
        Self { e_b_times_preimage, preimage_lower }
    }
}

impl PltEvaluator<ErrorNorm> for NormPltLWEEvaluator {
    fn public_lookup(
        &self,
        _: &(),
        plt: &PublicLut<DCRTPoly>,
        _: &ErrorNorm,
        input: &ErrorNorm,
        _: GateId,
        _: usize,
    ) -> ErrorNorm {
        let matrix_norm = &self.e_b_times_preimage + (&input.matrix_norm * &self.preimage_lower);
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.clone_ctx(), plaintext_bd);
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffinePltEvaluator for NormPltLWEEvaluator {
    fn public_lookup_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        plt: &PublicLut<DCRTPoly>,
        _: GateId,
        _: usize,
    ) -> ErrorNormSummaryExpr {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.plaintext_norm.ctx.clone(), plaintext_bd);
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&self.preimage_lower)
            .add_expr(&AffineErrorNormExpr::constant(self.e_b_times_preimage.clone()));
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}
#[derive(Debug, Clone)]
pub struct NormPltGGH15Evaluator {
    pub const_term: PolyMatrixNorm,
    pub e_input_multiplier: PolyMatrixNorm,
}

impl NormPltGGH15Evaluator {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        e_b_sigma: &BigDecimal,
        e_mat_sigma: &BigDecimal,
        secret_sigma: Option<BigDecimal>,
    ) -> Self {
        let dump_const_term_breakdown = std::env::var("MXX_SIM_GGH15_CONST_TERM_BREAKDOWN")
            .ok()
            .map(|raw| matches!(raw.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);
        let matrix_norm_bits = |m: &PolyMatrixNorm| bigdecimal_bits_ceil(&m.poly_norm.norm);
        let gaussian_bound = gaussian_tail_bound_factor();

        let preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None);
        info!("{}", format!("preimage norm bits {}", bigdecimal_bits_ceil(&preimage_norm)));
        let e_b_init =
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, e_b_sigma * &gaussian_bound, None);
        let s_vec = PolyMatrixNorm::new(
            ctx.clone(),
            1,
            ctx.secret_size,
            secret_sigma.unwrap_or(BigDecimal::one()),
            None,
        );
        // Corresponds to `preimage_gate1` sampled in `sample_gate_preimages_batch` stage1
        // from target `S_g * B1 + error` (B1 now has size d, so this is m_b x m_b).
        let preimage_gate1_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_b, preimage_norm.clone(), None);
        // Corresponds to stage1 Gaussian `error` in target `S_g * B1 + error`.
        let stage1_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_b,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        let gate1_from_eb = e_b_init.clone() * &preimage_gate1_from_b0;
        let gate1_from_s = s_vec.clone() * &stage1_target_error;
        // Corresponds to the error part of `c_b0 * preimage_gate1`.
        let gate1_error_total = &gate1_from_eb + &gate1_from_s;
        let gate1_total_bits = matrix_norm_bits(&gate1_error_total);
        let gate1_from_eb_bits = matrix_norm_bits(&gate1_from_eb);
        let gate1_from_s_bits = matrix_norm_bits(&gate1_from_s);

        // Corresponds to `gy.decompose()` in `public_lookup`.
        let gy_decomposed = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        // Corresponds to `v_idx` in `public_lookup`.
        let v_idx = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        // Corresponds to the vertically stacked
        // `small_decomposed_identity_chunk_from_scalar(...)` blocks used in vx accumulation.
        let small_decomposed_identity_chunks = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_g * ctx.log_base_q_small,
            ctx.m_g,
            ctx.base.clone() - BigDecimal::from(1u64),
            Some((ctx.m_g - 1) * ctx.log_base_q_small),
        );
        // Corresponds to `(small_decomposed_identity_chunk_from_scalar * v_idx)` in
        // `public_lookup`.
        let small_times_v = small_decomposed_identity_chunks * &v_idx;

        // Corresponds to `preimage_gate2_identity` (B0 preimage for identity/out term).
        let preimage_gate2_identity_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_norm.clone(), None);
        // Corresponds to `preimage_gate2_gy` (B0 preimage for gy term).
        let preimage_gate2_gy_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_norm.clone(), None);
        // Corresponds to `preimage_gate2_v` (B0 preimage for v_idx term).
        let preimage_gate2_v_from_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_norm.clone(), None);
        // Corresponds to concatenated `preimage_gate2_vx_chunk` blocks.
        let preimage_gate2_vx_from_b0 = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_b,
            ctx.m_g * ctx.log_base_q_small,
            preimage_norm.clone(),
            None,
        );
        // Corresponds to Gaussian `error` added in stage2 target
        // `S_g * w_block_identity + out_matrix + error`.
        let stage2_identity_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        // Corresponds to Gaussian `error` added in stage3 target
        // `S_g * w_block_gy - gadget + error`.
        let stage3_gy_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        // Corresponds to Gaussian `error` added in stage4 target
        // `S_g * w_block_v - (input_matrix * u_g_decomposed) + error`.
        let stage4_v_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g,
            e_mat_sigma * &gaussian_bound,
            None,
        );
        // Corresponds to Gaussian `error` added in stage5 target
        // `S_g * w_block_vx + (u_g_matrix * gadget_small) + error`.
        let stage5_vx_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g * ctx.log_base_q_small,
            e_mat_sigma * &gaussian_bound,
            None,
        );

        let gate2_identity_from_eb = e_b_init.clone() * &preimage_gate2_identity_from_b0;
        let gate2_identity_from_s = s_vec.clone() * &stage2_identity_target_error;
        let gate2_identity_total = &gate2_identity_from_eb + &gate2_identity_from_s;

        let gate2_gy_from_eb = e_b_init.clone() * &preimage_gate2_gy_from_b0;
        let gate2_gy_from_s = s_vec.clone() * &stage3_gy_target_error;
        let gate2_gy_total = &gate2_gy_from_eb + &gate2_gy_from_s;

        let gate2_v_from_eb = e_b_init.clone() * &preimage_gate2_v_from_b0;
        let gate2_v_from_s = s_vec.clone() * &stage4_v_target_error;
        let gate2_v_total = &gate2_v_from_eb + &gate2_v_from_s;

        let gate2_vx_from_eb = e_b_init.clone() * &preimage_gate2_vx_from_b0;
        let gate2_vx_from_s = s_vec.clone() * &stage5_vx_target_error;
        let gate2_vx_total = &gate2_vx_from_eb + &gate2_vx_from_s;

        // Corresponds to
        // `c_b0 * (preimage_gate2_gy * gy_decomposed + preimage_gate2_v * v_idx + vx_product_acc *
        // v_idx)`.
        let const_term_gate2_gy_total = gate2_gy_total.clone() * gy_decomposed.clone();
        let const_term_gate2_v_total = gate2_v_total.clone() * v_idx.clone();
        let const_term_gate2_vx_total = gate2_vx_total.clone() * small_times_v.clone();
        let mut const_term_gate2_t_total = const_term_gate2_gy_total.clone();
        const_term_gate2_t_total += const_term_gate2_v_total.clone();
        const_term_gate2_t_total += const_term_gate2_vx_total.clone();
        // Corresponds to `c_b0 * preimage_gate2_identity`.
        let const_term_gate2_identity_total = gate2_identity_total.clone();

        // Corresponds to the stored `preimage_lut` loaded in `public_lookup`.
        // In the GGH15 public-key evaluator, `sample_lut_preimages` already samples this matrix
        // from a target that includes identity + gy + v + vx components, and
        // `public_lookup` subtracts `preimage_gate1 * preimage_lut` without additional
        // multipliers.
        let preimage_lut_total =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, preimage_norm.clone(), None);
        // Corresponds to subtraction term
        // `c_b0 * (preimage_gate1 * preimage_lut)` in `public_lookup`.
        let const_term_lut_subtraction_total = gate1_error_total.clone() * preimage_lut_total;

        let mut const_term = const_term_gate2_identity_total.clone();
        const_term += const_term_gate2_t_total.clone();
        const_term += const_term_lut_subtraction_total.clone();
        info!(
            "{}",
            format!(
                "GGH15 PLT const term norm bits {}",
                bigdecimal_bits_ceil(&const_term.poly_norm.norm)
            )
        );

        if dump_const_term_breakdown {
            info!(
                "GGH15 const term breakdown bits: gate1_total={} gate1_from_eb={} gate1_from_s={} gate2_identity_total={} gate2_identity_from_eb={} gate2_identity_from_s={} gate2_gy_total={} gate2_gy_from_eb={} gate2_gy_from_s={} gate2_v_total={} gate2_v_from_eb={} gate2_v_from_s={} gate2_vx_total={} gate2_vx_from_eb={} gate2_vx_from_s={} term_gate2_identity={} term_gate2_gy={} term_gate2_v={} term_gate2_vx={} term_gate2_t={} term_lut_subtraction={} const_total={}",
                gate1_total_bits,
                gate1_from_eb_bits,
                gate1_from_s_bits,
                matrix_norm_bits(&gate2_identity_total),
                matrix_norm_bits(&gate2_identity_from_eb),
                matrix_norm_bits(&gate2_identity_from_s),
                matrix_norm_bits(&gate2_gy_total),
                matrix_norm_bits(&gate2_gy_from_eb),
                matrix_norm_bits(&gate2_gy_from_s),
                matrix_norm_bits(&gate2_v_total),
                matrix_norm_bits(&gate2_v_from_eb),
                matrix_norm_bits(&gate2_v_from_s),
                matrix_norm_bits(&gate2_vx_total),
                matrix_norm_bits(&gate2_vx_from_eb),
                matrix_norm_bits(&gate2_vx_from_s),
                matrix_norm_bits(&const_term_gate2_identity_total),
                matrix_norm_bits(&const_term_gate2_gy_total),
                matrix_norm_bits(&const_term_gate2_v_total),
                matrix_norm_bits(&const_term_gate2_vx_total),
                matrix_norm_bits(&const_term_gate2_t_total),
                matrix_norm_bits(&const_term_lut_subtraction_total),
                matrix_norm_bits(&const_term)
            );
        }

        // Corresponds to `input.vector * u_g_decomposed * v_idx` in `public_lookup`.
        let e_input_multiplier = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g) * &v_idx;
        info!(
            "{}",
            format!(
                "GGH15 PLT e_input multiplier norm bits {}",
                bigdecimal_bits_ceil(&e_input_multiplier.poly_norm.norm)
            )
        );

        Self { const_term, e_input_multiplier }
    }
}

impl PltEvaluator<ErrorNorm> for NormPltGGH15Evaluator {
    fn public_lookup(
        &self,
        _: &(),
        plt: &PublicLut<DCRTPoly>,
        _: &ErrorNorm,
        input: &ErrorNorm,
        _: GateId,
        _: usize,
    ) -> ErrorNorm {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.clone_ctx(), plaintext_bd);
        let matrix_norm = &self.const_term + &input.matrix_norm * &self.e_input_multiplier;
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffinePltEvaluator for NormPltGGH15Evaluator {
    fn public_lookup_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        plt: &PublicLut<DCRTPoly>,
        _: GateId,
        _: usize,
    ) -> ErrorNormSummaryExpr {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.plaintext_norm.ctx.clone(), plaintext_bd);
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&self.e_input_multiplier)
            .add_expr(&AffineErrorNormExpr::constant(self.const_term.clone()));
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}

#[derive(Debug, Clone)]
pub struct NormPltCommitEvaluator {
    pub lut_term: PolyMatrixNorm,
}

impl NormPltCommitEvaluator {
    pub fn new(
        ctx: Arc<SimulatorContext>,
        error_sigma: &BigDecimal,
        tree_base: usize,
        circuit: &PolyCircuit<DCRTPoly>,
    ) -> Self {
        let lut_vector_len = circuit.lut_vector_len_with_subcircuits();
        let padded_len = compute_padded_len(tree_base, lut_vector_len);
        debug!(
            "NormPltCommitEvaluator padded_len={} lut_vector_len={}",
            padded_len, lut_vector_len
        );
        let preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, None);
        let t_bottom = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_b,
            tree_base * ctx.m_b * ctx.m_g * ctx.m_g,
            preimage_norm.clone(),
            None,
        );
        let j_mat = PolyMatrixNorm::new(
            ctx.clone(),
            t_bottom.ncol,
            ctx.m_b * ctx.log_base_q,
            ctx.base.clone() - BigDecimal::from(1u64),
            None,
        );
        let verifier_base = t_bottom * &j_mat;
        let verifier_norm = verifier_base *
            PolyMatrixNorm::gadget_decomposed_with_secret_size(ctx.clone(), ctx.m_b, ctx.m_b);
        let t_top = PolyMatrixNorm::new(
            ctx.clone(),
            tree_base * ctx.m_b * ctx.m_b * ctx.m_g,
            tree_base * ctx.m_b * ctx.m_g * ctx.m_g,
            preimage_norm.clone(),
            None,
        );
        let t_top_j_mat = &t_top * &j_mat;
        let msg_tensor_identity = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.m_b,
            t_top.nrow,
            ctx.base.clone() - BigDecimal::from(1u64),
            Some(ctx.m_b - 1),
        );
        let opening_base = &msg_tensor_identity * t_top_j_mat;
        let j_mat_last = PolyMatrixNorm::new(
            ctx.clone(),
            tree_base * ctx.m_b * ctx.m_g * ctx.m_g,
            ctx.m_b,
            ctx.base.clone() - BigDecimal::from(1u64),
            Some(tree_base * ctx.m_b * ctx.m_g * ctx.m_g - ctx.m_b),
        );
        let opening_base_last = &msg_tensor_identity * &t_top * &j_mat_last;
        let log_tree_base_len = {
            let mut padded_len = padded_len;
            let mut depth = 0;
            while padded_len > 1 {
                debug_assert!(padded_len % tree_base == 0);
                padded_len /= tree_base;
                depth += 1;
            }
            depth
        };
        let opening_norm = {
            let lhs = opening_base *
                PolyMatrixNorm::gadget_decomposed_with_secret_size(ctx.clone(), ctx.m_b, ctx.m_b) *
                (log_tree_base_len - 1);
            lhs + opening_base_last
        };

        let gaussian_bound = gaussian_tail_bound_factor();
        let init_error =
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_b, error_sigma * &gaussian_bound, None);
        let preimage =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, verifier_norm.nrow, preimage_norm, None);
        let lut_term = &init_error * preimage * verifier_norm + init_error * opening_norm;
        info!("lut_term norm bits {}", bigdecimal_bits_ceil(&lut_term.poly_norm.norm));
        Self { lut_term }
    }
}

impl PltEvaluator<ErrorNorm> for NormPltCommitEvaluator {
    fn public_lookup(
        &self,
        _: &<ErrorNorm as Evaluable>::Params,
        plt: &PublicLut<<ErrorNorm as Evaluable>::P>,
        _: &ErrorNorm,
        input: &ErrorNorm,
        _: GateId,
        _: usize,
    ) -> ErrorNorm {
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.clone_ctx(), plaintext_bd);
        let ctx = input.clone_ctx();
        let m_b = ctx.m_b;
        let m_g = ctx.m_g;
        let matrix_norm =
            &self.lut_term + &input.matrix_norm * PolyMatrixNorm::gadget_decomposed(ctx, m_b);
        // info!("matrix_norm norm bits {}", bigdecimal_bits_ceil(&matrix_norm.poly_norm.norm));
        let (matrix_norm, _) = matrix_norm.split_cols(m_g);
        ErrorNorm { matrix_norm, plaintext_norm }
    }
}

impl AffinePltEvaluator for NormPltCommitEvaluator {
    fn public_lookup_affine(
        &self,
        input: &ErrorNormSummaryExpr,
        plt: &PublicLut<DCRTPoly>,
        _: GateId,
        _: usize,
    ) -> ErrorNormSummaryExpr {
        let ctx = self.lut_term.clone_ctx();
        let m_b = ctx.m_b;
        let m_g = ctx.m_g;
        let plaintext_bd =
            BigDecimal::from(num_bigint::BigInt::from(plt.max_output_row().1.value().clone()));
        let plaintext_norm = PolyNorm::new(input.plaintext_norm.ctx.clone(), plaintext_bd);
        let matrix_expr = input
            .matrix_expr
            .transform_matrix(&PolyMatrixNorm::gadget_decomposed(ctx, m_b))
            .add_expr(&AffineErrorNormExpr::constant(self.lut_term.clone()))
            .split_cols_left(m_g);
        ErrorNormSummaryExpr { plaintext_norm, matrix_expr }
    }
}

pub fn compute_preimage_norm(
    ring_dim_sqrt: &BigDecimal,
    m_g: u64,
    base: &BigDecimal,
    b_nrow: Option<usize>,
) -> BigDecimal {
    let c_0 = BigDecimal::from_f64(1.8).unwrap();
    let c_1 = BigDecimal::from_f64(4.7).unwrap();
    let sigma = BigDecimal::from_f64(4.578).unwrap();
    let two_sqrt = BigDecimal::from(2).sqrt().unwrap();
    let m_g_sqrt = BigDecimal::from(m_g).sqrt().expect("sqrt(m_g) failed");
    let b_nrow = b_nrow.unwrap_or(1);
    let term = BigDecimal::from(b_nrow as u64).sqrt().unwrap() * ring_dim_sqrt.clone() * m_g_sqrt +
        two_sqrt * ring_dim_sqrt +
        c_1;
    let preimage_norm =
        c_0 * BigDecimal::from_f32(6.5).unwrap() * sigma.clone() * ((base + 1) * sigma) * term;
    // let preimage_norm_bits = bigdecimal_bits_ceil(&preimage_norm);
    // info!("{}", format!("preimage norm bits {}", preimage_norm_bits));
    preimage_norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::PolyCircuit,
        lookup::PublicLut,
        poly::{
            Poly,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        simulator::SimulatorContext,
        slot_transfer::SlotTransferEvaluator,
    };
    use bigdecimal::BigDecimal;

    fn make_ctx() -> Arc<SimulatorContext> {
        // secpar_sqrt=50, ring_dim_sqrt=1024, base=32, log_base_q=(128/32)*7 = 28
        Arc::new(SimulatorContext::new(
            BigDecimal::from(1024u64), // ring_dim_sqrt
            BigDecimal::from(32u64),   // base
            2,
            28, // log_base_q
            3,  // log_base_q_small
        ))
    }

    fn simulate_max_error_norm_via_generic_eval_reference<P: AffinePltEvaluator>(
        circuit: &PolyCircuit<DCRTPoly>,
        ctx: Arc<SimulatorContext>,
        input_norm_bound: BigDecimal,
        input_size: usize,
        e_init_norm: &BigDecimal,
        plt_evaluator: Option<&P>,
    ) -> Vec<ErrorNorm> {
        let one_error = ErrorNorm::new(
            PolyNorm::one(ctx.clone()),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, e_init_norm.clone(), None),
        );
        let input_error = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_norm_bound),
            PolyMatrixNorm::new(ctx, 1, one_error.ctx().m_g, e_init_norm.clone(), None),
        );
        circuit.eval(&(), one_error, vec![input_error; input_size], plt_evaluator, None, None)
    }

    const E_B_SIGMA: f64 = 4.0;
    const E_INIT_NORM: u32 = 1 << 14;

    #[test]
    fn test_wire_norm_addition() {
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2);
        let out_gid = circuit.add_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = BigDecimal::from(5u64);

        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &BigDecimal::from(E_INIT_NORM),
            None::<&NormPltLWEEvaluator>,
            None,
        );
        assert_eq!(out.len(), 1);
        // Build expected from input wires and add them
        let in_wire = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
        );
        let expected = &in_wire + &in_wire;
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_subtraction() {
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2);
        let out_gid = circuit.sub_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = BigDecimal::from(5u64);
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &BigDecimal::from(E_INIT_NORM),
            None::<&NormPltLWEEvaluator>,
            None,
        );
        assert_eq!(out.len(), 1);
        let in_wire = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
        );
        let expected = &in_wire - &in_wire; // subtraction bound equals addition bound
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_multiplication() {
        // ctx: secpar_sqrt=50, ring_dim_sqrt=1024, base=32, log_base_q=28
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2);
        let out_gid = circuit.mul_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = BigDecimal::from(5u64);
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &BigDecimal::from(E_INIT_NORM),
            None::<&NormPltLWEEvaluator>,
            None,
        );
        assert_eq!(out.len(), 1);

        // Build expected = in_wire * in_wire
        let in_wire = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), input_bound),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(E_INIT_NORM), None),
        );
        let expected = &in_wire * &in_wire;
        assert_eq!(out[0], expected);
    }

    #[test]
    fn test_wire_norm_simulator_multiplication_matches_generic_eval() {
        let ctx = make_ctx();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let ins = circuit.input(2);
        let out_gid = circuit.mul_gate(ins[0], ins[1]);
        circuit.output(vec![out_gid]);
        let input_bound = BigDecimal::from(5u64);
        let e_init_norm = BigDecimal::from(E_INIT_NORM);
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
            None,
        );
        let generic = simulate_max_error_norm_via_generic_eval_reference(
            &circuit,
            ctx,
            input_bound,
            2,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
        );
        assert_eq!(out, generic);
    }

    #[test]
    fn test_wire_norm_slot_transfer_matches_bgg_poly_encoding_bound() {
        let ctx = make_ctx();
        let e_b0_sigma = 11.0;
        let c_b0_error_norm = PolyMatrixNorm::sample_gauss(
            ctx.clone(),
            1,
            ctx.m_b,
            BigDecimal::from_f64(e_b0_sigma).unwrap(),
        );
        let evaluator = NormBggPolyEncodingSTEvaluator::new(
            ctx.clone(),
            e_b0_sigma,
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            None,
        );
        let input = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), BigDecimal::from(5u64)),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(7u64), None),
        );
        let src_slots = [(2, None), (0, Some(3)), (1, Some(2))];

        let out = evaluator.slot_transfer(&(), &input, &src_slots, GateId(0));

        let b0_preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(1));
        let s_vec = PolyMatrixNorm::new(ctx.clone(), 1, ctx.secret_size, BigDecimal::one(), None);
        let gate_preimage =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, ctx.m_g, b0_preimage_norm.clone(), None);
        let gate_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_g,
            BigDecimal::from_f64(E_B_SIGMA * 6.5).unwrap(),
            None,
        );
        let slot_preimage_b0 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b, 2 * ctx.m_b, b0_preimage_norm.clone(), None);
        let b1_preimage_norm =
            compute_preimage_norm(&ctx.ring_dim_sqrt, ctx.m_g as u64, &ctx.base, Some(2));
        let slot_preimage_b1 =
            PolyMatrixNorm::new(ctx.clone(), ctx.m_b * 2, ctx.m_g, b1_preimage_norm.clone(), None);
        let slot_preimage_b0_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.m_b * 2,
            BigDecimal::from_f64(E_B_SIGMA * 6.5).unwrap(),
            None,
        );
        let slot_preimage_b1_target_error = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size * 2,
            ctx.m_g,
            BigDecimal::from_f64(E_B_SIGMA * 6.5).unwrap(),
            None,
        );
        let slot_secret_and_identity = PolyMatrixNorm::new(
            ctx.clone(),
            ctx.secret_size,
            ctx.secret_size * 2,
            BigDecimal::one(),
            None,
        );
        let scalar_bd = BigDecimal::from(3u64);
        let input_vector_multiplier = PolyMatrixNorm::gadget_decomposed(ctx.clone(), ctx.m_g);
        let plaintext_norm = input.plaintext_norm.clone() * &scalar_bd;
        let const_term =
            s_vec.clone() * &gate_target_error + c_b0_error_norm.clone() * &gate_preimage;
        let transfer_plaintext_multiplier =
            s_vec.clone() * slot_secret_and_identity * slot_preimage_b1_target_error +
                s_vec.clone() * slot_preimage_b0_target_error * slot_preimage_b1.clone() +
                c_b0_error_norm * slot_preimage_b0 * slot_preimage_b1;
        let matrix_norm = const_term +
            (input.matrix_norm.clone() * &input_vector_multiplier) * &scalar_bd +
            transfer_plaintext_multiplier * &plaintext_norm;

        assert_eq!(out, ErrorNorm { plaintext_norm, matrix_norm });
    }

    #[test]
    fn test_wire_norm_slot_transfer_bound_is_independent_of_slot_count() {
        let ctx = make_ctx();
        let e_b0_sigma = 9.0;
        let evaluator = NormBggPolyEncodingSTEvaluator::new(
            ctx.clone(),
            e_b0_sigma,
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            None,
        );
        let input = ErrorNorm::new(
            PolyNorm::new(ctx.clone(), BigDecimal::from(4u64)),
            PolyMatrixNorm::new(ctx.clone(), 1, ctx.m_g, BigDecimal::from(6u64), None),
        );

        let out_single = evaluator.slot_transfer(&(), &input, &[(0, Some(2))], GateId(0));
        let out_many = evaluator.slot_transfer(
            &(),
            &input,
            &[(0, Some(2)), (1, Some(2)), (2, Some(2))],
            GateId(1),
        );

        assert_eq!(out_single, out_many);
    }

    #[test]
    fn test_wire_norm_lwe_plt_bounds() {
        // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new(
            &params,
            2,
            |params, idx| match idx {
                0 => Some((
                    0,
                    DCRTPoly::from_usize_to_constant(params, 5)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                1 => Some((
                    1,
                    DCRTPoly::from_usize_to_constant(params, 7)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        // Circuit: out = PLT(in)
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt);
        let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![out_gate]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(5u64);
        let plt_evaluator =
            NormPltLWEEvaluator::new(ctx.clone(), &BigDecimal::from_f64(E_B_SIGMA).unwrap());
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            1,
            &BigDecimal::from(E_INIT_NORM),
            Some(&plt_evaluator),
            None,
        );
        assert_eq!(out.len(), 1);
        // Bound must be max output coeff across LUT entries (7)
        assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
    }

    #[test]
    fn test_wire_norm_ggh15_plt_bounds() {
        // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new(
            &params,
            2,
            |params, idx| match idx {
                0 => Some((
                    0,
                    DCRTPoly::from_usize_to_constant(params, 5)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                1 => Some((
                    1,
                    DCRTPoly::from_usize_to_constant(params, 7)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        // Circuit: out = PLT(in)
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt);
        let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![out_gate]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(5u64);
        let plt_evaluator = NormPltGGH15Evaluator::new(
            ctx.clone(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            None,
        );
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            1,
            &BigDecimal::from(E_INIT_NORM),
            Some(&plt_evaluator),
            None,
        );
        assert_eq!(out.len(), 1);
        // Bound must be max output coeff across LUT entries (7)
        assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
    }

    #[test]
    fn test_wire_norm_simulator_ggh15_plt_uses_lut_plaintext_bound() {
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new(
            &params,
            2,
            |params, idx| match idx {
                0 => Some((
                    0,
                    DCRTPoly::from_usize_to_constant(params, 5)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                1 => Some((
                    1,
                    DCRTPoly::from_usize_to_constant(params, 7)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt);
        let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![out_gate]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(5u64);
        let plt_evaluator = NormPltGGH15Evaluator::new(
            ctx.clone(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            None,
        );
        let out = circuit.simulate_max_error_norm(
            ctx,
            input_bound,
            1,
            &BigDecimal::from(E_INIT_NORM),
            Some(&plt_evaluator),
            None,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
    }

    #[test]
    fn test_wire_norm_simulator_sub_circuit_matches_generic_eval() {
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new(
            &params,
            2,
            |params, idx| match idx {
                0 => Some((
                    0,
                    DCRTPoly::from_usize_to_constant(params, 3)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                1 => Some((
                    1,
                    DCRTPoly::from_usize_to_constant(params, 5)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(1);
        let squared = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[0]);
        let sub_out = sub_circuit.add_gate(squared, sub_inputs[0]);
        sub_circuit.output(vec![sub_out]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2);
        let sub_circuit_id = circuit.register_sub_circuit(sub_circuit);
        let left = circuit.call_sub_circuit(sub_circuit_id, &[inputs[0]]);
        let right = circuit.call_sub_circuit(sub_circuit_id, &[inputs[1]]);
        let summed = circuit.add_gate(left[0], right[0]);
        let plt_id = circuit.register_public_lookup(plt);
        let out = circuit.public_lookup_gate(summed, plt_id);
        circuit.output(vec![out]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(13u64);
        let e_init_norm = BigDecimal::from(E_INIT_NORM);
        let plt_evaluator = NormPltGGH15Evaluator::new(
            ctx.clone(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            None,
        );

        let simulated = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &e_init_norm,
            Some(&plt_evaluator),
            None,
        );
        let generic = simulate_max_error_norm_via_generic_eval_reference(
            &circuit,
            ctx,
            input_bound,
            2,
            &e_init_norm,
            Some(&plt_evaluator),
        );

        assert_eq!(simulated, generic);
    }

    #[test]
    fn test_wire_norm_simulator_sub_circuit_recomputes_for_new_plaintext_profile() {
        let mut sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let sub_inputs = sub_circuit.input(1);
        let squared = sub_circuit.mul_gate(sub_inputs[0], sub_inputs[0]);
        let sub_out = sub_circuit.add_gate(squared, sub_inputs[0]);
        sub_circuit.output(vec![sub_out]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        let doubled = circuit.add_gate(inputs[0], inputs[0]);
        let sub_circuit_id = circuit.register_sub_circuit(sub_circuit);
        let left = circuit.call_sub_circuit(sub_circuit_id, &[inputs[0]]);
        let right = circuit.call_sub_circuit(sub_circuit_id, &[doubled]);
        let out = circuit.add_gate(left[0], right[0]);
        circuit.output(vec![out]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(13u64);
        let e_init_norm = BigDecimal::from(E_INIT_NORM);

        let simulated = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            1,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
            None,
        );
        let generic = simulate_max_error_norm_via_generic_eval_reference(
            &circuit,
            ctx,
            input_bound,
            1,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
        );

        assert_eq!(simulated, generic);
    }

    #[test]
    fn test_wire_norm_simulator_nested_sub_circuit_matches_generic_eval() {
        let mut inner_sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let inner_inputs = inner_sub_circuit.input(1);
        let inner_out = inner_sub_circuit.add_gate(inner_inputs[0], inner_inputs[0]);
        inner_sub_circuit.output(vec![inner_out]);

        let mut outer_sub_circuit = PolyCircuit::<DCRTPoly>::new();
        let outer_inputs = outer_sub_circuit.input(2);
        let inner_sub_circuit_id = outer_sub_circuit.register_sub_circuit(inner_sub_circuit);
        let inner_from_second =
            outer_sub_circuit.call_sub_circuit(inner_sub_circuit_id, &[outer_inputs[1]]);
        let outer_out = outer_sub_circuit.add_gate(outer_inputs[0], inner_from_second[0]);
        outer_sub_circuit.output(vec![outer_out]);

        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2);
        let outer_sub_circuit_id = circuit.register_sub_circuit(outer_sub_circuit);
        let left = circuit.call_sub_circuit(outer_sub_circuit_id, &[inputs[0], inputs[1]]);
        let right = circuit.call_sub_circuit(outer_sub_circuit_id, &[inputs[1], inputs[0]]);
        let out = circuit.add_gate(left[0], right[0]);
        circuit.output(vec![out]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(11u64);
        let e_init_norm = BigDecimal::from(E_INIT_NORM);

        let simulated = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            2,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
            None,
        );
        let generic = simulate_max_error_norm_via_generic_eval_reference(
            &circuit,
            ctx,
            input_bound,
            2,
            &e_init_norm,
            None::<&NormPltLWEEvaluator>,
        );

        assert_eq!(simulated, generic);
    }

    #[test]
    fn test_wire_norm_commit_plt_bounds() {
        // Build a tiny LUT on DCRTPoly where the maximum output coeff is known (e.g., 7)
        let params = DCRTPolyParams::default();
        let plt = PublicLut::<DCRTPoly>::new(
            &params,
            2,
            |params, idx| match idx {
                0 => Some((
                    0,
                    DCRTPoly::from_usize_to_constant(params, 5)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                1 => Some((
                    1,
                    DCRTPoly::from_usize_to_constant(params, 7)
                        .coeffs()
                        .into_iter()
                        .next()
                        .expect("constant-term coefficient must exist"),
                )),
                _ => unreachable!("index out of range for test LUT"),
            },
            None,
        );

        // Circuit: out = PLT(in)
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(1);
        let plt_id = circuit.register_public_lookup(plt);
        let out_gate = circuit.public_lookup_gate(inputs[0], plt_id);
        circuit.output(vec![out_gate]);

        let ctx = make_ctx();
        let input_bound = BigDecimal::from(5u64);
        let tree_base = 2;
        let plt_evaluator = NormPltCommitEvaluator::new(
            ctx.clone(),
            &BigDecimal::from_f64(E_B_SIGMA).unwrap(),
            tree_base,
            &circuit,
        );
        let out = circuit.simulate_max_error_norm(
            ctx.clone(),
            input_bound.clone(),
            1,
            &BigDecimal::from(E_INIT_NORM),
            Some(&plt_evaluator),
            None,
        );
        assert_eq!(out.len(), 1);
        // Bound must be max output coeff across LUT entries (7)
        assert_eq!(out[0].plaintext_norm.norm, BigDecimal::from(7u64));
    }
}
