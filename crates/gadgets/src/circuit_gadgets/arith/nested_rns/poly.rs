use super::{context::nested_rns_level_from_wires, *};
use crate::{
    circuit_gadgets::arith::{
        BinaryPlannerResult, DecomposeArithmeticGadget, ModularArithmeticContext,
        ModularArithmeticGadget, ModularArithmeticPlanner,
    },
    matrix::PolyMatrix,
    utils::mod_inverse,
};
use num_traits::ToPrimitive;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NestedRnsPlannerMetadata {
    pub max_plaintexts: Vec<BigUint>,
    pub p_max_traces: Vec<BigUint>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NestedRnsAddPlanKey {
    pub pre_full_reduce: bool,
    pub reduce_levels: Vec<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NestedRnsSubPlanKey {
    pub pre_full_reduce: bool,
    pub reduce_levels: Vec<bool>,
    pub trace_multipliers: Vec<BigUint>,
}

/// Build the large-scalar bindings used by the subtraction helper that adds trace offsets first.
///
/// Nested-RNS subtraction works in nonnegative residues, so the left operand is shifted by a
/// multiple of `p_i` before subtracting the right operand. Keeping this helper local makes the
/// binding layout live next to the only call site that needs it.
fn sub_with_trace_offset_param_bindings(
    offset_multiplier: &BigUint,
    p_moduli: &[u64],
) -> Vec<SubCircuitParamValue> {
    p_moduli
        .par_iter()
        .map(|&p_i| {
            SubCircuitParamValue::LargeScalarMul(vec![offset_multiplier * BigUint::from(p_i)])
        })
        .collect()
}

impl<P: Poly> NestedRnsPoly<P> {
    pub fn physical_slots(&self) -> usize {
        self.window.physical_slots(self.num_coefficient_slots)
    }

    /// Construct a nested-RNS polynomial from already-built q-level batches plus metadata.
    ///
    /// All higher-level constructors eventually funnel through here so the invariant checks on
    /// the active window, packed width, `max_plaintexts`, and `p_max_traces` stay centralized.
    pub fn new(
        ctx: Arc<NestedRnsPolyContext>,
        inner: BatchedWire,
        num_coefficient_slots: usize,
        window: CrtWindow,
        max_plaintexts: Vec<BigUint>,
    ) -> Self {
        assert!(num_coefficient_slots > 0, "nested-RNS coefficient slot count must be positive");
        let window = CrtWindow::new(window.offset, window.depth, ctx.q_moduli_depth);
        let _physical_slots = window.physical_slots(num_coefficient_slots);
        assert_eq!(max_plaintexts.len(), window.depth, "plaintext bounds must match CRT window");
        let p_max_traces = vec![ctx.reduced_p_max_trace(); window.depth];
        let poly = Self {
            ctx,
            inner,
            num_coefficient_slots,
            window,
            max_plaintexts,
            p_max_traces,
            _p: PhantomData,
        };
        poly.validate_representation();
        poly
    }

    /// Replace the carried trace metadata while preserving the underlying wire layout.
    ///
    /// This is used by helpers that know their exact post-operation trace bounds and want to return
    /// a new `NestedRnsPoly` without re-deriving all other metadata.
    pub(crate) fn with_p_max_traces(mut self, p_max_traces: Vec<BigUint>) -> Self {
        self.p_max_traces = p_max_traces;
        self.validate_representation();
        self
    }

    /// Allocate a fresh circuit input in nested-RNS form.
    pub fn input(
        ctx: Arc<NestedRnsPolyContext>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let window = CrtWindow::new(window.offset, window.depth, ctx.q_moduli_depth);
        let inner = circuit.input(ctx.p_moduli.len());
        let max_plaintexts = ctx.q_moduli[window.offset..window.end()]
            .par_iter()
            .map(|&q_i| BigUint::from(q_i - 1))
            .collect();
        Self::new(ctx, inner, num_coefficient_slots, window, max_plaintexts)
    }

    /// Allocate a fresh input while preserving explicit plaintext and trace metadata.
    ///
    /// Support sub-circuit builders and metadata-preserving transforms use this when the new wires
    /// should behave exactly like an existing nested-RNS value from the bound-tracking perspective.
    pub(crate) fn input_with_metadata(
        ctx: Arc<NestedRnsPolyContext>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let window = CrtWindow::new(window.offset, window.depth, ctx.q_moduli_depth);
        assert_eq!(max_plaintexts.len(), window.depth);
        assert_eq!(p_max_traces.len(), window.depth);
        let inner = circuit.input(ctx.p_moduli.len());
        Self::new(ctx, inner, num_coefficient_slots, window, max_plaintexts)
            .with_p_max_traces(p_max_traces)
    }

    fn planner_metadata(&self) -> NestedRnsPlannerMetadata {
        NestedRnsPlannerMetadata {
            max_plaintexts: self.max_plaintexts.clone(),
            p_max_traces: self.p_max_traces.clone(),
        }
    }

    fn normalized_planner_metadata(
        ctx: &NestedRnsPolyContext,
        window: CrtWindow,
    ) -> NestedRnsPlannerMetadata {
        let (max_plaintexts, p_max_traces) = ctx.full_reduce_output_metadata(window);
        NestedRnsPlannerMetadata { max_plaintexts, p_max_traces }
    }

    /// Lazily reduce the packed wire when at least one q-level is selected.
    ///
    /// The LUT acts on every physical lane in one sub-circuit call. Metadata is therefore reset for
    /// every nonzero active lane, while proven-zero lanes retain their sparse zero bound.
    fn lazy_reduce_selected_levels(
        &self,
        reduce_levels: &[bool],
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let levels = self.resolve_enable_levels();
        assert_eq!(reduce_levels.len(), levels);
        if !reduce_levels.iter().any(|&flag| flag) {
            return self.clone();
        }

        let reduced_trace = self.ctx.reduced_p_max_trace();
        let p_max_traces = self
            .max_plaintexts
            .iter()
            .map(
                |bound| {
                    if bound == &BigUint::ZERO { BigUint::ZERO } else { reduced_trace.clone() }
                },
            )
            .collect();
        let inner = nested_rns_level_from_wires(circuit.call_sub_circuit_with_max_plaintext_norms(
            self.ctx.lazy_reduce_id,
            &[self.inner],
            self.lookup_input_ranges(),
        ));
        Self::new(
            self.ctx.clone(),
            inner,
            self.num_coefficient_slots,
            self.window,
            self.max_plaintexts.clone(),
        )
        .with_p_max_traces(p_max_traces)
    }

    /// Reduce exactly those q-levels whose current trace metadata says the lazy-reduce LUT is
    /// needed.
    pub(crate) fn lazy_reduce_if_unreduced(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let threshold = self.ctx.unreduced_trace_threshold();
        let reduce_levels = self.p_max_traces[..self.resolve_enable_levels()]
            .par_iter()
            .map(|trace| trace >= &threshold)
            .collect::<Vec<_>>();
        self.lazy_reduce_selected_levels(&reduce_levels, circuit)
    }

    /// Return the fully reduced trace bound for every currently active q-level.
    fn reduced_p_max_traces(&self) -> Vec<BigUint> {
        vec![self.ctx.reduced_p_max_trace(); self.resolve_enable_levels()]
    }

    /// Translate the explicit per-q-level trace invariant into the existing sub-circuit input
    /// contract. Lookup boundaries currently accept only canonical, nonnegative constant
    /// polynomials, so this is a coefficient bound rather than a matrix norm.
    fn lookup_input_ranges(&self) -> Vec<SubCircuitInputMaxPlaintextNormRange> {
        SubCircuitInputMaxPlaintextNormRange::compress(&self.lookup_input_norms())
    }

    fn lookup_input_norms(&self) -> Vec<BigUint> {
        let trace = self.p_max_traces[..self.resolve_enable_levels()]
            .iter()
            .max()
            .cloned()
            .unwrap_or(BigUint::ZERO);
        self.ctx.lookup_input_ranges_for_trace(&trace);
        vec![trace; self.ctx.p_moduli.len()]
    }

    fn combined_lookup_input_ranges(
        &self,
        other: &Self,
    ) -> Vec<SubCircuitInputMaxPlaintextNormRange> {
        let mut norms = self.lookup_input_norms();
        norms.extend(other.lookup_input_norms());
        SubCircuitInputMaxPlaintextNormRange::compress(&norms)
    }

    /// Predict the post-addition trace bounds without changing the underlying wires.
    fn compute_add_output_p_max_traces(&self, other: &Self) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        self.p_max_traces[..levels]
            .par_iter()
            .zip(other.p_max_traces[..levels].par_iter())
            .map(|(left_trace, right_trace)| left_trace + right_trace)
            .collect()
    }

    /// Convert a trace bound into the offset multiplier required by subtraction.
    ///
    /// Subtraction adds enough multiples of `p_max` to the left operand to keep all residues
    /// nonnegative before applying the helper sub-circuit.
    fn trace_multiplier(&self, trace: &BigUint) -> BigUint {
        (trace + BigUint::from(self.ctx.p_max - 1)) / BigUint::from(self.ctx.p_max)
    }

    /// Predict the post-subtraction trace bounds, including the nonnegative offset shift.
    fn compute_sub_output_p_max_traces(&self, other: &Self) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        let p_max = BigUint::from(self.ctx.p_max);
        let uniform_multiplier = other
            .p_max_traces
            .iter()
            .map(|trace| self.trace_multiplier(trace))
            .max()
            .unwrap_or(BigUint::ZERO);
        self.p_max_traces[..levels]
            .par_iter()
            .zip(self.max_plaintexts[..levels].par_iter())
            .zip(other.max_plaintexts[..levels].par_iter())
            .map(|((left_trace, left_bound), right_bound)| {
                if left_bound == &BigUint::ZERO && right_bound == &BigUint::ZERO {
                    BigUint::ZERO
                } else {
                    left_trace + &uniform_multiplier * &p_max
                }
            })
            .collect()
    }

    /// Assert that the tracked traces still fit inside the lookup tables installed by the context.
    ///
    /// Helpers that rely on `lut_mod_p_*` tables call this before dispatch so any metadata bug
    /// fails immediately instead of silently generating an out-of-domain lookup.
    fn assert_p_max_traces_within_lut_map_size(&self, traces: &[BigUint], message: &str) {
        assert!(
            traces.iter().all(|trace| trace < &self.ctx.lut_mod_p_max_map_size),
            "{}: p_max_traces={:?}, lut_mod_p_max_map_size={}",
            message,
            traces,
            self.ctx.lut_mod_p_max_map_size
        );
    }

    fn coefficient_rotation(&self, src_slots: &[(u32, Option<Vec<u64>>)]) -> Option<usize> {
        if src_slots.is_empty() || src_slots.iter().any(|(_, scalar)| scalar.is_some()) {
            return None;
        }
        let n = self.num_coefficient_slots;
        let first = usize::try_from(src_slots[0].0).ok()?;
        let diagonal = (n - first % n) % n;
        src_slots
            .iter()
            .enumerate()
            .all(|(dst, (src, _))| usize::try_from(*src).ok() == Some((dst + n - diagonal) % n))
            .then_some(diagonal)
    }

    fn expand_slot_transfer_for_residue(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
        p_j: u64,
    ) -> Vec<(u32, Option<u32>)> {
        let lanes = self.window.depth;
        let levels = self.resolve_enable_levels();
        src_slots
            .iter()
            .enumerate()
            .flat_map(|(dst_c, (src_c, scalars))| {
                assert!(
                    usize::try_from(*src_c).expect("source coefficient block must fit usize") <
                        self.num_coefficient_slots,
                    "source coefficient block exceeds packed width"
                );
                if let Some(scalars) = scalars {
                    assert_eq!(scalars.len(), levels, "slot scalar depth must match active levels");
                }
                (0..lanes).map(move |g| {
                    let src =
                        usize::try_from(*src_c).expect("source block must fit usize") * lanes + g;
                    let scalar = scalars.as_ref().map(|values| {
                        u32::try_from(values[g] % p_j).expect("slot-transfer scalar must fit u32")
                    });
                    let _dst = dst_c * lanes + g;
                    (u32::try_from(src).expect("physical source slot must fit u32"), scalar)
                })
            })
            .collect()
    }

    fn identity_repeated_lane_scalars(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
        residue_modulus: u64,
    ) -> Option<Vec<Option<u32>>> {
        if src_slots.len() != self.num_coefficient_slots {
            return None;
        }
        let lanes = self.window.depth;
        let levels = self.resolve_enable_levels();
        let mut repeated: Option<Vec<Option<u32>>> = None;
        for (block, (source, scalars)) in src_slots.iter().enumerate() {
            if usize::try_from(*source).ok() != Some(block) {
                return None;
            }
            let lane_scalars = match scalars {
                Some(values) if values.len() == levels && values.len() == lanes => values
                    .iter()
                    .map(|value| u32::try_from(value % residue_modulus).ok().map(Some))
                    .collect::<Option<Vec<_>>>()?,
                Some(_) => return None,
                None => vec![None; lanes],
            };
            match &repeated {
                Some(expected) if expected != &lane_scalars => return None,
                Some(_) => {}
                None => repeated = Some(lane_scalars),
            }
        }
        repeated
    }

    fn lane_scalar_mul_gate(
        &self,
        gate: GateId,
        active_scalars: &[u64],
        residue_modulus: u64,
        circuit: &mut PolyCircuit<P>,
    ) -> GateId {
        assert_eq!(active_scalars.len(), self.resolve_enable_levels());
        let lane_scalars = active_scalars
            .iter()
            .map(|scalar| {
                Some(u32::try_from(scalar % residue_modulus).expect("lane scalar must fit u32"))
            })
            .collect();
        circuit
            .slot_identity_repeated_lanes_gate(gate, self.num_coefficient_slots, lane_scalars)
            .as_single_wire()
    }

    /// Apply one coefficient-block slot plan lane-wise, automatically reducing first when needed.
    ///
    /// The operation preserves the original behavior: first ensure the predicted plaintext bound
    /// fits under `p_full`, then lazy-reduce any unreduced traces, and finally run the
    /// lane-packed slot-transfer plus lazy-reduce helper.
    pub fn slot_transfer(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let mut operand = self.clone();
        let predicted_bounds = self.compute_slot_transfer_output_bounds(src_slots);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            operand = self.full_reduce(circuit);
        }
        operand = operand.lazy_reduce_if_unreduced(circuit);
        let final_bounds = operand.compute_slot_transfer_output_bounds(src_slots);
        operand.assert_bounds_within_p_full(
            &final_bounds,
            "slot_transfer output exceeds p_full even after automatic full_reduce",
        );

        assert_eq!(src_slots.len(), operand.num_coefficient_slots);
        let rotation = operand.coefficient_rotation(src_slots);
        let transferred = operand
            .inner
            .gate_ids()
            .zip(operand.ctx.p_moduli.iter().copied())
            .map(|(gate_id, p_j)| {
                if let Some(diagonal) = rotation {
                    circuit.slot_rotation_gate(
                        gate_id,
                        diagonal * operand.window.depth,
                        operand.window.physical_slots(operand.num_coefficient_slots),
                    )
                } else {
                    if let Some(lane_scalars) =
                        operand.identity_repeated_lane_scalars(src_slots, p_j)
                    {
                        circuit.slot_identity_repeated_lanes_gate(
                            gate_id,
                            operand.num_coefficient_slots,
                            lane_scalars,
                        )
                    } else {
                        let expanded = operand.expand_slot_transfer_for_residue(src_slots, p_j);
                        circuit.slot_transfer_gate(gate_id, &expanded)
                    }
                }
            })
            .collect::<Vec<_>>();
        let inner = nested_rns_level_from_wires(circuit.call_sub_circuit_with_max_plaintext_norms(
            operand.ctx.lazy_reduce_id,
            &transferred,
            operand.lookup_input_ranges(),
        ));
        Self::new(
            operand.ctx.clone(),
            inner,
            operand.num_coefficient_slots,
            operand.window,
            final_bounds,
        )
        .with_p_max_traces(operand.reduced_p_max_traces())
    }

    /// Repack into another contiguous CRT window using exactly `N * target.depth` lanes.
    pub(crate) fn repack_window(&self, target: CrtWindow, circuit: &mut PolyCircuit<P>) -> Self {
        let target = CrtWindow::new(target.offset, target.depth, self.ctx.q_moduli_depth);
        let operand = self.lazy_reduce_if_unreduced(circuit);
        if target == operand.window {
            return operand;
        }
        let inner = {
            let plan = (0..operand.num_coefficient_slots)
                .flat_map(|coefficient| {
                    (0..target.depth).map(move |target_local| {
                        let global = target.offset + target_local;
                        if global >= operand.window.offset && global < operand.window.end() {
                            let source_local = global - operand.window.offset;
                            let source = coefficient * operand.window.depth + source_local;
                            (u32::try_from(source).expect("physical slot must fit u32"), None)
                        } else {
                            (0, Some(0))
                        }
                    })
                })
                .collect::<Vec<_>>();
            operand
                .inner
                .gate_ids()
                .map(|gate| circuit.slot_transfer_gate(gate, &plan).as_single_wire())
                .collect::<Vec<_>>()
        };
        let mut max_plaintexts = vec![BigUint::ZERO; target.depth];
        let mut p_max_traces = vec![BigUint::ZERO; target.depth];
        for target_local in 0..target.depth {
            let global = target.offset + target_local;
            if global >= operand.window.offset && global < operand.window.end() {
                let source_local = global - operand.window.offset;
                max_plaintexts[target_local] = operand.max_plaintexts[source_local].clone();
                p_max_traces[target_local] = operand.p_max_traces[source_local].clone();
            }
        }
        Self::new(
            operand.ctx.clone(),
            BatchedWire::from_batches(inner),
            operand.num_coefficient_slots,
            target,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    pub fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible_layout(other);
        let mut left = self.clone();
        let mut right = other.clone();
        let predicted_bounds =
            self.compute_binary_output_bounds(other, &|left, right, _| left + right);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            left = self.full_reduce(circuit);
            right = other.full_reduce(circuit);
        }

        let predicted_traces = left.compute_add_output_p_max_traces(&right);
        let reduce_levels = predicted_traces
            .iter()
            .map(|trace| trace >= &left.ctx.lut_mod_p_max_map_size)
            .collect::<Vec<_>>();
        left = left.lazy_reduce_selected_levels(&reduce_levels, circuit);
        right = right.lazy_reduce_selected_levels(&reduce_levels, circuit);

        let final_bounds =
            left.compute_binary_output_bounds(&right, &|left, right, _| left + right);
        left.assert_bounds_within_p_full(
            &final_bounds,
            "additive operation output exceeds p_full even after automatic full_reduce",
        );
        let final_traces = left.compute_add_output_p_max_traces(&right);
        left.assert_p_max_traces_within_lut_map_size(
            &final_traces,
            "additive operation output exceeds lut_mod_p_map_size even after pre-reduction",
        );
        left.call_uniform_binary_subcircuit(
            &right,
            circuit,
            self.ctx.add_without_reduce_id,
            final_bounds,
            final_traces,
        )
    }

    pub fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.assert_compatible_layout(other);
        let mut left = self.clone();
        let mut right = other.clone();
        let predicted_bounds = self.compute_binary_output_bounds(other, &|left, _right, q_i| {
            left + BigUint::from(q_i - 1)
        });
        if self.bounds_exceed_p_full(&predicted_bounds) {
            left = self.full_reduce(circuit);
            right = other.full_reduce(circuit);
        }

        let predicted_traces = left.compute_sub_output_p_max_traces(&right);
        let reduce_levels = predicted_traces
            .iter()
            .map(|trace| trace >= &left.ctx.lut_mod_p_max_map_size)
            .collect::<Vec<_>>();
        left = left.lazy_reduce_selected_levels(&reduce_levels, circuit);
        right = right.lazy_reduce_selected_levels(&reduce_levels, circuit);

        let final_bounds = left.compute_binary_output_bounds(&right, &|left, right, q_i| {
            if left == &BigUint::ZERO && right == &BigUint::ZERO {
                BigUint::ZERO
            } else {
                left + BigUint::from(q_i - 1)
            }
        });
        left.assert_bounds_within_p_full(
            &final_bounds,
            "subtractive operation output exceeds p_full even after automatic full_reduce",
        );
        let final_traces = left.compute_sub_output_p_max_traces(&right);
        left.assert_p_max_traces_within_lut_map_size(
            &final_traces,
            "subtractive operation output exceeds lut_mod_p_map_size even after pre-reduction",
        );
        let proven_zero = left
            .max_plaintexts
            .iter()
            .zip(&right.max_plaintexts)
            .map(|(l, r)| l == &BigUint::ZERO && r == &BigUint::ZERO)
            .collect::<Vec<_>>();
        let result = left.call_sub_with_trace_offsets(
            &right,
            circuit,
            final_bounds.clone(),
            final_traces.clone(),
        );
        if !proven_zero.iter().any(|x| *x) {
            result
        } else {
            let scalars = proven_zero.into_iter().map(|zero| u64::from(!zero)).collect::<Vec<_>>();
            let plan = (0..left.num_coefficient_slots)
                .map(|c| {
                    (
                        u32::try_from(c).expect("coefficient block must fit u32"),
                        Some(scalars.clone()),
                    )
                })
                .collect::<Vec<_>>();
            result.slot_transfer(&plan, circuit)
        }
    }

    pub fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let left = self.lazy_reduce_if_unreduced(circuit);
        let right = other.lazy_reduce_if_unreduced(circuit);
        left.apply_binary_operation(&right, circuit, |left, right, _| left * right)
    }

    pub fn mul_right_sparse(
        &self,
        other: &Self,
        right_q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.assert_compatible_layout(other);
        let levels = self.resolve_enable_levels();
        other.assert_sparse_at_q_idx(right_q_idx);

        let mut left = self.clone();
        let mut right = other.clone();
        let mut predicted_bounds = vec![BigUint::ZERO; levels];
        predicted_bounds[right_q_idx] =
            &self.max_plaintexts[right_q_idx] * &other.max_plaintexts[right_q_idx];
        if self.bounds_exceed_p_full(&predicted_bounds) {
            left = self.full_reduce(circuit);
            right = other.full_reduce(circuit);
        }

        left = left.lazy_reduce_if_unreduced(circuit);
        right = right.lazy_reduce_if_unreduced(circuit);

        let mut final_bounds = vec![BigUint::ZERO; levels];
        final_bounds[right_q_idx] =
            &left.max_plaintexts[right_q_idx] * &right.max_plaintexts[right_q_idx];
        left.assert_bounds_within_p_full(
            &final_bounds,
            "mul_right_sparse output exceeds p_full even after automatic full_reduce",
        );

        let mut final_traces = vec![BigUint::ZERO; levels];
        final_traces[right_q_idx] = left.ctx.reduced_p_max_trace();
        left.call_sparse_right_subcircuit(&right, right_q_idx, circuit, final_bounds, final_traces)
    }

    pub fn full_reduce(&self, circuit: &mut PolyCircuit<P>) -> Self {
        let operand = self.lazy_reduce_if_unreduced(circuit);
        let levels = self.resolve_enable_levels();
        let (ys, w) = operand.decomposition_terms(circuit);
        let p_depth = self.ctx.p_moduli.len();
        let mut sum = vec![circuit.const_zero_gate(); p_depth];
        for (y_idx, &y) in ys.iter().enumerate() {
            let scalars = (0..levels)
                .map(|a| {
                    (&self.ctx.p_over_pis[y_idx] %
                        BigUint::from(self.ctx.q_moduli[self.window.offset + a]))
                    .to_u64()
                    .expect("full-reduce scalar must fit u64")
                })
                .collect::<Vec<_>>();
            let scaled = self
                .ctx
                .p_moduli
                .iter()
                .map(|&p_i| self.lane_scalar_mul_gate(y, &scalars, p_i, circuit))
                .collect::<Vec<_>>();
            let reduced = circuit.call_sub_circuit_with_max_plaintext_norms(
                self.ctx.lazy_reduce_id,
                &scaled,
                self.ctx.canonical_residue_scaled_input_ranges(),
            );
            for (acc, term) in sum.iter_mut().zip(reduced) {
                *acc = circuit.add_gate(*acc, term);
            }
        }
        let p_scalars = (0..levels)
            .map(|a| {
                (&self.ctx.p_full % BigUint::from(self.ctx.q_moduli[self.window.offset + a]))
                    .to_u64()
                    .expect("full-reduce scalar must fit u64")
            })
            .collect::<Vec<_>>();
        let scaled_w = self
            .ctx
            .p_moduli
            .iter()
            .map(|&p_i| self.lane_scalar_mul_gate(w, &p_scalars, p_i, circuit))
            .collect::<Vec<_>>();
        let raw = sum
            .into_iter()
            .zip(scaled_w)
            .zip(self.ctx.p_moduli.iter().copied())
            .map(|((sum, w_term), p_i)| {
                let offset = circuit
                    .const_digits(&[u32::try_from(self.ctx.p_moduli.len() as u64 * p_i)
                        .expect("full-reduce offset must fit u32")]);
                let shifted = circuit.add_gate(sum, offset);
                circuit.sub_gate(shifted, w_term)
            })
            .collect::<Vec<_>>();
        let result_inner =
            nested_rns_level_from_wires(circuit.call_sub_circuit_with_max_plaintext_norms(
                self.ctx.lazy_reduce_id,
                &raw,
                self.ctx.full_reduce_raw_input_ranges(),
            ));
        let max_plaintexts = (0..levels)
            .map(|local_idx| {
                if operand.max_plaintexts[local_idx] == BigUint::ZERO {
                    BigUint::ZERO
                } else {
                    self.ctx.full_reduce_max_plaintexts[self.window.offset + local_idx].clone()
                }
            })
            .collect::<Vec<_>>();
        let p_max_traces = (0..levels)
            .map(|a| {
                if operand.max_plaintexts[a] == BigUint::ZERO {
                    BigUint::ZERO
                } else {
                    operand.ctx.reduced_p_max_trace()
                }
            })
            .collect();
        Self::new(
            self.ctx.clone(),
            result_inner,
            self.num_coefficient_slots,
            self.window,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    pub fn const_mul(&self, tower_constants: &[u64], circuit: &mut PolyCircuit<P>) -> Self {
        let levels = self.resolve_enable_levels();
        assert_eq!(tower_constants.len(), levels);
        let mut operand = self.clone();
        let predicted_bounds = self.compute_const_mul_output_bounds(tower_constants);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            operand = self.full_reduce(circuit);
        }
        operand = operand.lazy_reduce_if_unreduced(circuit);
        let final_bounds = operand.compute_const_mul_output_bounds(tower_constants);
        operand.assert_bounds_within_p_full(
            &final_bounds,
            "const_mul output exceeds p_full even after automatic full_reduce",
        );
        let plan = (0..operand.num_coefficient_slots)
            .map(|c| {
                (
                    u32::try_from(c).expect("coefficient block must fit u32"),
                    Some(tower_constants.to_vec()),
                )
            })
            .collect::<Vec<_>>();
        let scaled = operand.slot_transfer(&plan, circuit);
        scaled.packed_with_metadata(scaled.inner, final_bounds, operand.reduced_p_max_traces())
    }

    /// Multiply every physical lane by one scalar without introducing a slot-transfer gate.
    ///
    /// Modulus-basis conversion uses this only after isolating and moving a single nonzero lane.
    /// All other lanes therefore stay literal zero, while the arithmetic scalar operation avoids
    /// an unnecessary BGG slot-transfer artifact.
    pub(crate) fn uniform_const_mul(&self, scalar: u64, circuit: &mut PolyCircuit<P>) -> Self {
        let constants = vec![scalar; self.resolve_enable_levels()];
        let mut operand = self.clone();
        let predicted_bounds = self.compute_const_mul_output_bounds(&constants);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            operand = self.full_reduce(circuit);
        }
        operand = operand.lazy_reduce_if_unreduced(circuit);
        let final_bounds = operand.compute_const_mul_output_bounds(&constants);
        operand.assert_bounds_within_p_full(
            &final_bounds,
            "uniform const_mul output exceeds p_full even after automatic full_reduce",
        );
        let scaled = operand
            .inner
            .gate_ids()
            .zip(operand.ctx.p_moduli.iter().copied())
            .map(|(gate, p_i)| {
                let scalar = BigUint::from(scalar % p_i);
                circuit.large_scalar_mul(gate, std::slice::from_ref(&scalar))
            })
            .collect::<Vec<_>>();
        let inner = nested_rns_level_from_wires(circuit.call_sub_circuit_with_max_plaintext_norms(
            operand.ctx.lazy_reduce_id,
            &scaled,
            operand.ctx.canonical_residue_scaled_input_ranges(),
        ));
        operand.packed_with_metadata(inner, final_bounds, operand.reduced_p_max_traces())
    }

    pub fn gadget_vector(
        ctx: Arc<NestedRnsPolyContext>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        let _ = encoding::resolve_nested_rns_active_window(ctx.as_ref(), window);
        let chunk_width = ctx.p_moduli.len() + 1;
        let gadget_values = ctx.gadget_values[window.offset..window.end()]
            .iter()
            .flat_map(|level_values| level_values.iter().cloned())
            .collect::<Vec<_>>();
        gadget_values
            .into_iter()
            .enumerate()
            .map(|(idx, value)| {
                Self::sparse_constant_level_poly(
                    ctx.clone(),
                    num_coefficient_slots,
                    window,
                    idx / chunk_width,
                    &value,
                    circuit,
                )
            })
            .collect()
    }

    pub fn gadget_decompose(&self, circuit: &mut PolyCircuit<P>) -> Vec<Self> {
        let operand = if self.bounds_exceed_p_full(&self.max_plaintexts) {
            self.full_reduce(circuit)
        } else {
            self.clone()
        };
        operand.assert_p_max_traces_within_lut_map_size(
            &operand.p_max_traces[..operand.resolve_enable_levels()],
            "gadget_decompose input exceeds lut_mod_p_map_size",
        );
        let levels = operand.resolve_enable_levels();
        let p_moduli_depth = operand.ctx.p_moduli.len();
        let w_bound =
            BigUint::from(u64::try_from(p_moduli_depth).expect("p_moduli length must fit in u64"));
        let mut decomposition = Vec::with_capacity(levels * (p_moduli_depth + 1));
        let outputs = circuit.call_sub_circuit_with_max_plaintext_norms(
            operand.ctx.gadget_decompose_id,
            &[operand.inner],
            operand.lookup_input_ranges(),
        );
        for q_idx in 0..levels {
            for p_idx in 0..p_moduli_depth {
                let y_bound = BigUint::from(operand.ctx.p_moduli[p_idx] - 1);
                let start = p_idx * p_moduli_depth;
                let y_row = nested_rns_level_from_wires(
                    outputs[start..start + p_moduli_depth].iter().copied(),
                );
                decomposition.push(operand.sparse_level_poly_from_row(
                    q_idx,
                    y_row,
                    y_bound.clone(),
                    y_bound,
                    circuit,
                ));
            }
            let w_start = p_moduli_depth * p_moduli_depth;
            let w_row = nested_rns_level_from_wires(
                outputs[w_start..w_start + p_moduli_depth].iter().copied(),
            );
            decomposition.push(operand.sparse_level_poly_from_row(
                q_idx,
                w_row,
                w_bound.clone(),
                w_bound.clone(),
                circuit,
            ));
        }

        decomposition
    }

    pub fn conv_mul_right_decomposed_many(
        &self,
        params: &P::Params,
        left_rows: &[&[Self]],
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self>
    where
        P: 'static,
    {
        if left_rows.is_empty() {
            return vec![];
        }

        let levels = self.resolve_enable_levels();
        let p_moduli_depth = self.ctx.p_moduli.len();
        let chunk_width = p_moduli_depth + 1;
        let gadget_len = levels * chunk_width;
        for (row_idx, row) in left_rows.iter().enumerate() {
            assert_eq!(row.len(), gadget_len, "left row {} length mismatch", row_idx);
            for (entry_idx, entry) in row.iter().enumerate() {
                entry.assert_compatible_layout(self);
                assert!(
                    Arc::ptr_eq(&entry.ctx, &self.ctx),
                    "conv_mul_right_decomposed_many requires left row {} entry {} to share the NestedRnsPolyContext with right",
                    row_idx,
                    entry_idx
                );
            }
        }

        let right = self.prepare_for_decomposed_conv(circuit);
        let prepared_left_rows = left_rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| entry.prepare_for_decomposed_conv(circuit))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let row_count = prepared_left_rows.len();

        let lhs_trace_bound = prepared_left_rows
            .iter()
            .flat_map(|row| row.iter())
            .flat_map(|entry| entry.p_max_traces[..entry.resolve_enable_levels()].iter())
            .max()
            .cloned()
            .unwrap_or(BigUint::ZERO);
        right.ctx.lookup_input_ranges_for_trace(&lhs_trace_bound);
        let lhs_input_norms = vec![lhs_trace_bound; p_moduli_depth];
        let rhs_trace_bound = BigUint::from(right.ctx.p_moduli.iter().copied().max().unwrap() - 1);
        right.ctx.lookup_input_ranges_for_trace(&rhs_trace_bound);
        let rhs_input_norms = vec![rhs_trace_bound; p_moduli_depth];

        let term_subcircuit = negacyclic_conv_mul_right_decomposed_term_many_subcircuit::<P>(
            circuit,
            right.ctx.as_ref(),
            row_count,
            num_slots,
            right.window.depth,
            &lhs_input_norms,
            &rhs_input_norms,
        );
        let term_subcircuit_id = circuit.register_sub_circuit(term_subcircuit);

        let flat_term_output_templates =
            encoding::map_nested_rns_values(row_count * gadget_len, |flat_idx| {
                let row_idx = flat_idx / gadget_len;
                let global_idx = flat_idx % gadget_len;
                let left = &prepared_left_rows[row_idx][global_idx];
                Self::conv_mul_right_decomposed_output_template(
                    circuit,
                    params,
                    left,
                    global_idx / chunk_width,
                    global_idx % chunk_width,
                    num_slots,
                )
            });
        let term_output_templates = flat_term_output_templates
            .chunks(gadget_len)
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();

        let mut row_terms = vec![Vec::with_capacity(gadget_len); row_count];
        for q_idx in 0..levels {
            let (ys, w) = right.decomposition_terms(circuit);
            for term_idx in 0..chunk_width {
                let global_idx = q_idx * chunk_width + term_idx;
                let term_gate = if term_idx < p_moduli_depth { ys[term_idx] } else { w };
                let term_row = vec![BatchedWire::single(term_gate); p_moduli_depth];
                let mut inputs = Vec::with_capacity(row_count + p_moduli_depth);
                let mut input_norms = Vec::with_capacity((row_count + 1) * p_moduli_depth);
                for row in &prepared_left_rows {
                    inputs.push(row[global_idx].inner);
                    input_norms.extend(row[global_idx].lookup_input_norms());
                }
                inputs.extend_from_slice(&term_row);
                let term_bound = if term_idx < p_moduli_depth {
                    BigUint::from(right.ctx.p_moduli[term_idx] - 1)
                } else {
                    BigUint::from(p_moduli_depth)
                };
                input_norms.extend(vec![term_bound; p_moduli_depth]);
                let outputs = circuit.call_sub_circuit_with_max_plaintext_norms(
                    term_subcircuit_id,
                    &inputs,
                    SubCircuitInputMaxPlaintextNormRange::compress(&input_norms),
                );
                for row_idx in 0..row_count {
                    let start = row_idx * p_moduli_depth;
                    let output_template = &term_output_templates[row_idx][global_idx];
                    row_terms[row_idx].push(Self::sparse_level_poly_from_row_with_metadata(
                        self.ctx.clone(),
                        self.num_coefficient_slots,
                        self.window,
                        q_idx,
                        nested_rns_level_from_wires(
                            outputs[start..start + p_moduli_depth].iter().copied(),
                        ),
                        output_template.max_plaintexts[q_idx].clone(),
                        output_template.p_max_traces[q_idx].clone(),
                        circuit,
                    ));
                }
            }
        }

        row_terms
            .into_iter()
            .map(|mut terms| {
                let mut acc = terms
                    .pop()
                    .expect("conv_mul_right_decomposed_many requires at least one gadget term");
                for term in terms {
                    acc = acc.add(&term, circuit);
                }
                acc
            })
            .collect()
    }

    pub fn conv_mul_right_decomposed(
        &self,
        params: &P::Params,
        left_row: &[Self],
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self
    where
        P: 'static,
    {
        self.conv_mul_right_decomposed_many(params, &[left_row], num_slots, circuit)
            .into_iter()
            .next()
            .expect("conv_mul_right_decomposed must produce one output row")
    }

    fn prepare_for_decomposed_conv(&self, circuit: &mut PolyCircuit<P>) -> Self {
        if self.bounds_exceed_p_full(&self.max_plaintexts) {
            self.full_reduce(circuit)
        } else {
            self.assert_p_max_traces_within_lut_map_size(
                &self.p_max_traces[..self.resolve_enable_levels()],
                "decomposed convolution input exceeds lut_mod_p_map_size",
            );
            self.clone()
        }
    }

    fn sparse_decomposed_term_input_template(
        ctx: Arc<NestedRnsPolyContext>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        target_q_idx: usize,
        term_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let target_row = circuit.input(ctx.p_moduli.len());
        let (max_plaintext, p_max_trace) = if term_idx < ctx.p_moduli.len() {
            let bound = BigUint::from(ctx.p_moduli[term_idx] - 1);
            (bound.clone(), bound)
        } else {
            let bound =
                BigUint::from(u64::try_from(ctx.p_moduli.len()).expect("p_moduli length fits u64"));
            (bound.clone(), bound)
        };
        Self::sparse_level_poly_from_row_with_metadata(
            ctx,
            num_coefficient_slots,
            window,
            target_q_idx,
            target_row,
            max_plaintext,
            p_max_trace,
            circuit,
        )
    }

    fn conv_mul_right_decomposed_output_template(
        source_circuit: &PolyCircuit<P>,
        params: &P::Params,
        left: &Self,
        target_q_idx: usize,
        term_idx: usize,
        num_slots: usize,
    ) -> Self
    where
        P: 'static,
    {
        let mut template_circuit = source_circuit.fresh_sub_circuit();
        let template_ctx = left.ctx.clone();
        let lhs = Self::input_with_metadata(
            template_ctx.clone(),
            left.num_coefficient_slots,
            left.window,
            left.max_plaintexts.clone(),
            left.p_max_traces.clone(),
            &mut template_circuit,
        );
        let rhs = Self::sparse_decomposed_term_input_template(
            template_ctx,
            left.num_coefficient_slots,
            lhs.window,
            target_q_idx,
            term_idx,
            &mut template_circuit,
        );
        negacyclic_conv_mul_right_sparse(
            params,
            &mut template_circuit,
            &lhs,
            &rhs,
            target_q_idx,
            num_slots,
        )
    }

    pub(crate) fn prepare_for_reconstruct(&self, circuit: &mut PolyCircuit<P>) -> Self {
        if self.bounds_exceed_p_full(&self.max_plaintexts) {
            self.full_reduce(circuit)
        } else {
            self.lazy_reduce_if_unreduced(circuit)
        }
    }

    pub fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        let operand = self.prepare_for_reconstruct(circuit);
        let levels = operand.resolve_enable_levels();
        let active_moduli = operand.active_q_moduli();
        let active_modulus =
            active_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
        let (ys, w) = operand.decomposition_terms(circuit);
        let mut x_prime = circuit.const_zero_gate();
        for (p_idx, y_i) in ys.into_iter().enumerate() {
            let scaled = circuit.large_scalar_mul(y_i, &[operand.ctx.p_over_pis[p_idx].clone()]);
            x_prime = circuit.add_gate(x_prime, scaled);
        }
        let pv = circuit.large_scalar_mul(w, &[operand.ctx.p_full.clone()]);
        x_prime = circuit.sub_gate(x_prime, pv);

        let lanes = operand.window.depth;
        let mut sum_mod_q = circuit.const_zero_gate();
        for a in 0..levels {
            let q_i_big = BigUint::from(active_moduli[a]);
            let q_over_qi = &active_modulus / &q_i_big;
            let q_over_qi_mod = &q_over_qi % &q_i_big;
            let inv = mod_inverse(
                q_over_qi_mod.to_u64().expect("CRT residue must fit in u64"),
                active_moduli[a],
            )
            .expect("CRT modulus must be invertible within the active range");
            let reconst_coeff = (&q_over_qi * BigUint::from(inv)) % &active_modulus;
            let broadcast = (0..operand.num_coefficient_slots * lanes)
                .map(|dst| {
                    let coefficient = dst / lanes;
                    let src = coefficient * lanes + a;
                    (u32::try_from(src).expect("physical slot must fit u32"), None)
                })
                .collect::<Vec<_>>();
            let homogeneous = circuit.slot_transfer_gate(x_prime, &broadcast);
            let sum_q_k_scaled = circuit.large_scalar_mul(homogeneous, &[reconst_coeff]);
            sum_mod_q = circuit.add_gate(sum_mod_q, sum_q_k_scaled);
        }
        sum_mod_q.as_single_wire()
    }

    /// Reconstructs each active q-level only into its coefficient's q1 anchor.
    ///
    /// The returned wire is intentionally not a full-lane reconstruction: for
    /// coefficient `c`, only slot `c * active_crt_depth` is authoritative.  This
    /// path is for Tall's anchor consumer, which reads exactly those slots.  It
    /// uses one compact block-wise anchor reduction rather than one rotation per
    /// active CRT lane, while leaving [`Self::reconstruct`] unchanged for callers
    /// that require every lane to be valid.
    pub fn reconstruct_q1_anchors(&self, circuit: &mut PolyCircuit<P>) -> Q1AnchorReconstruction {
        let operand = self.prepare_for_reconstruct(circuit);
        let levels = operand.resolve_enable_levels();
        let active_moduli = operand.active_q_moduli();
        let active_modulus =
            active_moduli.iter().fold(BigUint::from(1u64), |acc, &q_i| acc * BigUint::from(q_i));
        let (ys, w) = operand.decomposition_terms(circuit);
        let mut x_prime = circuit.const_zero_gate();
        for (p_idx, y_i) in ys.into_iter().enumerate() {
            let scaled = circuit.large_scalar_mul(y_i, &[operand.ctx.p_over_pis[p_idx].clone()]);
            x_prime = circuit.add_gate(x_prime, scaled);
        }
        let pv = circuit.large_scalar_mul(w, &[operand.ctx.p_full.clone()]);
        x_prime = circuit.sub_gate(x_prime, pv);

        let lanes = operand.window.depth;
        let mut reconst_coefficients = Vec::with_capacity(levels);
        for a in 0..levels {
            let q_i_big = BigUint::from(active_moduli[a]);
            let q_over_qi = &active_modulus / &q_i_big;
            let q_over_qi_mod = &q_over_qi % &q_i_big;
            let inv = mod_inverse(
                q_over_qi_mod.to_u64().expect("CRT residue must fit in u64"),
                active_moduli[a],
            )
            .expect("CRT modulus must be invertible within the active range");
            let reconst_coeff = (&q_over_qi * BigUint::from(inv)) % &active_modulus;
            reconst_coefficients.push(reconst_coeff);
        }
        reconst_coefficients.resize(lanes, BigUint::ZERO);
        let anchors = circuit.slot_anchor_reduce_gate(
            x_prime,
            operand.num_coefficient_slots,
            reconst_coefficients,
        );
        Q1AnchorReconstruction {
            anchor_wire: anchors.as_single_wire(),
            coefficient_slots: operand.num_coefficient_slots,
            q_moduli_depth: lanes,
        }
    }

    pub fn benchmark_multiplication_tree(
        ctx: Arc<NestedRnsPolyContext>,
        circuit: &mut PolyCircuit<P>,
        height: usize,
        window: CrtWindow,
    ) {
        let num_inputs =
            1usize.checked_shl(height as u32).expect("height is too large to represent 2^h inputs");
        let mut current_layer: Vec<NestedRnsPoly<P>> = (0..num_inputs)
            .map(|_| NestedRnsPoly::input(ctx.clone(), 1, window, circuit))
            .collect();
        while current_layer.len() > 1 {
            let mut next_layer = Vec::with_capacity(current_layer.len() / 2);
            for pair in current_layer.chunks(2) {
                let parent = pair[0].mul(&pair[1], circuit);
                next_layer.push(parent);
            }
            current_layer = next_layer;
        }
        let root = current_layer.pop().expect("multiplication tree must contain at least one node");
        let out = root.reconstruct(circuit);
        circuit.output(vec![out]);
    }

    /// Execute one lane-uniform binary helper subcircuit on the packed value.
    ///
    /// This helper is intentionally thin: higher-level operations compute the post-operation
    /// bounds first, then hand those exact bounds here so this function only performs the wiring.
    fn call_binary_subcircuit(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        subcircuit_id: usize,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let result_inner =
            nested_rns_level_from_wires(circuit.call_sub_circuit_with_max_plaintext_norms(
                subcircuit_id,
                &[self.inner, other.inner],
                self.combined_lookup_input_ranges(other),
            ));
        Self::new(
            self.ctx.clone(),
            result_inner,
            self.num_coefficient_slots,
            self.window,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn call_sparse_right_subcircuit(
        &self,
        other: &Self,
        target_q_idx: usize,
        circuit: &mut PolyCircuit<P>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let product = self.call_product(other, circuit, max_plaintexts, p_max_traces);
        let mut scalars = vec![0u64; self.resolve_enable_levels()];
        scalars[target_q_idx] = 1;
        let plan = (0..self.num_coefficient_slots)
            .map(|c| {
                (u32::try_from(c).expect("coefficient block must fit u32"), Some(scalars.clone()))
            })
            .collect::<Vec<_>>();
        product.slot_transfer(&plan, circuit)
    }

    fn call_product(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let left_norms = self.lookup_input_norms();
        let right_norms = other.lookup_input_norms();
        let products = self
            .inner
            .gate_ids()
            .zip(other.inner.gate_ids())
            .map(|(left, right)| circuit.mul_gate(left, right))
            .collect::<Vec<_>>();
        let product_norms = left_norms
            .iter()
            .zip(&right_norms)
            .map(|(left, right)| left * right)
            .collect::<Vec<_>>();
        let ranges = self.ctx.checked_lookup_input_ranges(product_norms);
        let inner = nested_rns_level_from_wires(circuit.call_sub_circuit_with_max_plaintext_norms(
            self.ctx.lazy_reduce_id,
            products,
            ranges,
        ));
        self.packed_with_metadata(inner, max_plaintexts, p_max_traces)
    }

    fn call_uniform_binary_subcircuit(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        subcircuit_id: usize,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        self.call_binary_subcircuit(other, circuit, subcircuit_id, max_plaintexts, p_max_traces)
    }

    fn packed_with_metadata(
        &self,
        inner: BatchedWire,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        Self::new(self.ctx.clone(), inner, self.num_coefficient_slots, self.window, max_plaintexts)
            .with_p_max_traces(p_max_traces)
    }

    /// Subtraction cannot stay within the lazy range by raw `left - right`, because the borrowed
    /// amount depends on how large the unreduced right trace may be. This helper routes every
    /// packed lane through the dedicated subcircuit that first shifts the left operand by a
    /// multiple of `p_i`, then subtracts the right operand.
    ///
    /// Callers provide the already-computed output metadata, so the only logic here is choosing
    /// one uniform offset multiplier and wiring the matching parameter bindings.
    fn call_sub_with_trace_offsets(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let offset_multiplier = other
            .p_max_traces
            .iter()
            .map(|trace| self.trace_multiplier(trace))
            .max()
            .unwrap_or(BigUint::ZERO);
        let bindings = sub_with_trace_offset_param_bindings(&offset_multiplier, &self.ctx.p_moduli);
        let outputs = circuit.call_sub_circuit_with_bindings_and_max_plaintext_norms(
            self.ctx.sub_with_trace_offsets_id,
            &[self.inner, other.inner],
            &bindings,
            self.combined_lookup_input_ranges(other),
        );
        self.packed_with_metadata(
            nested_rns_level_from_wires(outputs),
            max_plaintexts,
            p_max_traces,
        )
    }

    fn apply_binary_operation<FB>(
        &self,
        other: &Self,
        circuit: &mut PolyCircuit<P>,
        output_bound: FB,
    ) -> Self
    where
        FB: Fn(&BigUint, &BigUint, u64) -> BigUint,
    {
        self.assert_compatible_layout(other);
        let mut left = self.clone();
        let mut right = other.clone();
        let predicted_bounds = self.compute_binary_output_bounds(other, &output_bound);
        if self.bounds_exceed_p_full(&predicted_bounds) {
            let levels = self.resolve_enable_levels();
            let context_reduced_bounds =
                &self.ctx.full_reduce_max_plaintexts[self.window.offset..self.window.end()];
            let reduced_bounds_for = |bounds: &[BigUint]| {
                bounds[..levels]
                    .iter()
                    .zip(context_reduced_bounds)
                    .map(
                        |(bound, reduced)| {
                            if bound == &BigUint::ZERO { BigUint::ZERO } else { reduced.clone() }
                        },
                    )
                    .collect::<Vec<_>>()
            };
            let reduced_left_bounds = reduced_bounds_for(&self.max_plaintexts);
            let reduced_right_bounds = reduced_bounds_for(&other.max_plaintexts);
            let bounds_with = |left_bounds: &[BigUint], right_bounds: &[BigUint]| {
                (0..levels)
                    .map(|q_idx| {
                        output_bound(
                            &left_bounds[q_idx],
                            &right_bounds[q_idx],
                            self.ctx.q_moduli[self.window.offset + q_idx],
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let left_reduced_bounds = bounds_with(&reduced_left_bounds, &other.max_plaintexts);
            let right_reduced_bounds = bounds_with(&self.max_plaintexts, &reduced_right_bounds);
            let left_reduction_suffices = !self.bounds_exceed_p_full(&left_reduced_bounds);
            let right_reduction_suffices = !self.bounds_exceed_p_full(&right_reduced_bounds);

            match (left_reduction_suffices, right_reduction_suffices) {
                (true, false) => left = self.full_reduce(circuit),
                (false, true) => right = other.full_reduce(circuit),
                (true, true) => {
                    if left_reduced_bounds.iter().max() <= right_reduced_bounds.iter().max() {
                        left = self.full_reduce(circuit);
                    } else {
                        right = other.full_reduce(circuit);
                    }
                }
                (false, false) => {
                    left = self.full_reduce(circuit);
                    right = other.full_reduce(circuit);
                }
            }
        }
        let final_bounds = left.compute_binary_output_bounds(&right, &output_bound);
        left.assert_bounds_within_p_full(
            &final_bounds,
            "binary operation output exceeds p_full even after automatic full_reduce",
        );
        left.call_product(&right, circuit, final_bounds, left.reduced_p_max_traces())
    }

    /// Materialize the `y_i` digits and rounding accumulator `w` for one active q-level.
    ///
    /// The returned pair is consumed by reconstruction and decomposed-convolution code:
    /// reconstruction multiplies each `y_i` by `p / p_i` and subtracts `w * p`, while the
    /// convolution path uses the same terms as gadget digits. The helper deliberately hides the
    /// subcircuit call shape so those higher-level routines can reason in terms of algebraic
    /// objects instead of raw wire layout.
    pub(crate) fn decomposition_terms(
        &self,
        circuit: &mut PolyCircuit<P>,
    ) -> (Vec<GateId>, GateId) {
        let outputs = circuit.call_sub_circuit_with_max_plaintext_norms(
            self.ctx.decomposition_terms_id,
            &[self.inner],
            self.lookup_input_ranges(),
        );
        let p_moduli_depth = self.ctx.p_moduli.len();
        (
            outputs[..p_moduli_depth].iter().copied().map(BatchedWire::as_single_wire).collect(),
            outputs[p_moduli_depth].as_single_wire(),
        )
    }

    pub(crate) fn sparse_constant_level_poly(
        ctx: Arc<NestedRnsPolyContext>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        target_q_idx: usize,
        value: &BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let p_moduli = ctx.p_moduli.clone();
        let value_for_residues = value.clone();
        let residues = encoding::map_nested_rns_values(p_moduli.len(), move |idx| {
            &value_for_residues % BigUint::from(p_moduli[idx])
        });
        let p_max_trace = residues.iter().cloned().max().unwrap_or(BigUint::ZERO);
        let row = residues
            .into_iter()
            .map(|residue| const_biguint_gate(circuit, &residue))
            .collect::<Vec<_>>();
        Self::sparse_level_poly_from_row_with_metadata(
            ctx,
            num_coefficient_slots,
            window,
            target_q_idx,
            nested_rns_level_from_wires(row),
            value.clone(),
            p_max_trace,
            circuit,
        )
    }

    /// Wrap one explicit p-moduli row as a sparse nested-RNS value whose non-zero mass is known to
    /// live at a single q-level.
    ///
    /// Higher-level callers use this in two different situations:
    /// - gadget/vector constructors create constant sparse rows that are already fully known, and
    /// - decomposed convolution synthesizes placeholder rows that represent one decomposition term.
    ///
    /// The method does not transform the row at all; it only attaches the precise metadata that
    /// downstream arithmetic depends on: which q-level is active, what the plaintext bound is at
    /// that level, and how large the unreduced p-trace may be before a lazy reduction is required.
    pub(crate) fn sparse_level_poly_from_row_with_metadata(
        ctx: Arc<NestedRnsPolyContext>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        target_q_idx: usize,
        target_row: BatchedWire,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        let window = CrtWindow::new(window.offset, window.depth, ctx.q_moduli_depth);
        let mut max_plaintexts = vec![BigUint::ZERO; window.depth];
        let mut p_max_traces = vec![BigUint::ZERO; window.depth];
        max_plaintexts[target_q_idx] = max_plaintext;
        p_max_traces[target_q_idx] = p_max_trace;
        let lanes = window.depth;
        let lane_scalars =
            (0..lanes).map(|lane| Some(u32::from(lane == target_q_idx))).collect::<Vec<_>>();
        let inner = target_row
            .gate_ids()
            .map(|gate| {
                circuit
                    .slot_identity_repeated_lanes_gate(
                        gate,
                        num_coefficient_slots,
                        lane_scalars.clone(),
                    )
                    .as_single_wire()
            })
            .collect::<Vec<_>>();

        Self::new(
            ctx,
            BatchedWire::from_batches(inner),
            num_coefficient_slots,
            window,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn sparse_level_poly_from_row(
        &self,
        target_q_idx: usize,
        target_row: BatchedWire,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::sparse_level_poly_from_row_with_metadata(
            self.ctx.clone(),
            self.num_coefficient_slots,
            self.window,
            target_q_idx,
            target_row,
            max_plaintext,
            p_max_trace,
            circuit,
        )
    }

    fn compute_binary_output_bounds<F>(&self, other: &Self, output_bound: &F) -> Vec<BigUint>
    where
        F: Fn(&BigUint, &BigUint, u64) -> BigUint,
    {
        let levels = self.resolve_enable_levels();
        (0..levels)
            .map(|q_idx| {
                output_bound(
                    &self.max_plaintexts[q_idx],
                    &other.max_plaintexts[q_idx],
                    self.ctx.q_moduli[self.window.offset + q_idx],
                )
            })
            .collect()
    }

    fn compute_const_mul_output_bounds(&self, tower_constants: &[u64]) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        (0..levels)
            .map(|q_idx| {
                &self.max_plaintexts[q_idx] *
                    BigUint::from(
                        tower_constants[q_idx] % self.ctx.q_moduli[self.window.offset + q_idx],
                    )
            })
            .collect()
    }

    fn compute_slot_transfer_output_bounds(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
    ) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        let tower_scales = self.compute_slot_transfer_tower_scales(src_slots);
        (0..levels).map(|q_idx| &self.max_plaintexts[q_idx] * &tower_scales[q_idx]).collect()
    }

    fn compute_slot_transfer_tower_scales(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
    ) -> Vec<BigUint> {
        let levels = self.resolve_enable_levels();
        (0..levels)
            .map(|q_idx| {
                src_slots
                    .iter()
                    .map(|(_src_slot, slot_scalars)| {
                        let scalar = slot_scalars.as_ref().map_or(1u64, |slot_scalars| {
                            let residue = *slot_scalars.get(q_idx).unwrap_or_else(|| {
                                panic!(
                                    "slot scalar depth {} does not cover q_moduli_idx {}",
                                    slot_scalars.len(),
                                    q_idx
                                )
                            });
                            residue % self.ctx.q_moduli[self.window.offset + q_idx]
                        });
                        BigUint::from(scalar)
                    })
                    .max()
                    .unwrap_or(BigUint::ZERO)
            })
            .collect()
    }

    fn bounds_exceed_p_full(&self, bounds: &[BigUint]) -> bool {
        bounds.iter().any(|bound| bound >= &self.ctx.p_full)
    }

    fn assert_bounds_within_p_full(&self, bounds: &[BigUint], message: &str) {
        assert!(
            !self.bounds_exceed_p_full(bounds),
            "{}: max_plaintexts={:?}, p_full={}",
            message,
            bounds,
            self.ctx.p_full
        );
    }

    fn assert_compatible_layout(&self, other: &Self) {
        assert!(Arc::ptr_eq(&self.ctx, &other.ctx), "nested-RNS operands must share one context");
        assert_eq!(
            self.num_coefficient_slots, other.num_coefficient_slots,
            "mismatched coefficient slot counts"
        );
        assert_eq!(
            self.window, other.window,
            "mismatched CRT windows: left={:?}, right={:?}",
            self.window, other.window
        );
    }

    fn assert_sparse_at_q_idx(&self, target_q_idx: usize) {
        let levels = self.resolve_enable_levels();
        assert!(
            target_q_idx < levels,
            "mul_right_sparse target q_idx {} exceeds active levels {}",
            target_q_idx,
            levels
        );
        for q_idx in 0..levels {
            if q_idx != target_q_idx {
                assert!(
                    self.max_plaintexts[q_idx] == BigUint::ZERO,
                    "mul_right_sparse requires the right operand to be zero outside q_idx"
                );
            }
        }
        assert!(
            self.max_plaintexts[target_q_idx] != BigUint::ZERO,
            "mul_right_sparse requires a non-zero bound at q_idx"
        );
    }

    fn resolve_enable_levels(&self) -> usize {
        self.window.depth
    }

    fn validate_representation(&self) {
        let window = CrtWindow::new(self.window.offset, self.window.depth, self.ctx.q_moduli_depth);
        let _physical_slots = window.physical_slots(self.num_coefficient_slots);
        assert_eq!(self.max_plaintexts.len(), window.depth);
        assert_eq!(self.p_max_traces.len(), window.depth);
        assert_eq!(self.inner.len(), self.ctx.p_moduli.len());
    }

    pub fn active_q_moduli(&self) -> Vec<u64> {
        self.ctx.q_moduli[self.window.offset..self.window.end()].to_vec()
    }
}

/// Materialize a `BigUint` constant as one circuit gate, choosing the cheapest encoding path.
fn const_biguint_gate<P: Poly>(circuit: &mut PolyCircuit<P>, value: &BigUint) -> GateId {
    if let Some(value_u32) = value.to_u32() {
        circuit.const_digits(&[value_u32]).as_single_wire()
    } else {
        let one = circuit.const_one_gate();
        circuit.large_scalar_mul(one, std::slice::from_ref(value)).as_single_wire()
    }
}

impl<P: Poly + 'static> ModularArithmeticGadget<P> for NestedRnsPoly<P> {
    type Context = NestedRnsPolyContext;

    fn context(&self) -> &Arc<Self::Context> {
        &self.ctx
    }

    fn crt_window(&self) -> CrtWindow {
        self.window
    }

    fn max_plaintexts(&self) -> &[BigUint] {
        &self.max_plaintexts
    }

    fn p_max_traces(&self) -> &[BigUint] {
        &self.p_max_traces
    }

    fn num_coefficient_slots(&self) -> usize {
        self.num_coefficient_slots
    }

    fn input(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        NestedRnsPoly::input(ctx, num_coefficient_slots, window, circuit)
    }

    fn input_with_metadata(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        NestedRnsPoly::input_with_metadata(
            ctx,
            num_coefficient_slots,
            window,
            max_plaintexts,
            p_max_traces,
            circuit,
        )
    }

    fn active_q_moduli(&self) -> Vec<u64> {
        self.active_q_moduli()
    }

    fn flatten(&self) -> Vec<BatchedWire> {
        self.inner.gate_ids().map(BatchedWire::single).collect()
    }

    fn from_flat_outputs(
        template: &Self,
        outputs: &[GateId],
        max_plaintexts: Vec<BigUint>,
        p_max_traces: Vec<BigUint>,
    ) -> Self {
        let p_moduli_depth = template.ctx.p_moduli.len();
        assert_eq!(
            outputs.len(),
            p_moduli_depth,
            "flattened packed nested-RNS output size must match p_moduli_depth"
        );
        NestedRnsPoly::new(
            template.ctx.clone(),
            BatchedWire::from_batches(outputs.iter().copied()),
            template.num_coefficient_slots,
            template.window,
            max_plaintexts,
        )
        .with_p_max_traces(p_max_traces)
    }

    fn q_level_row_batch(&self, _q_idx: usize) -> BatchedWire {
        self.inner
    }

    fn sparse_level_poly_with_metadata(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        target_q_idx: usize,
        target_row: BatchedWire,
        max_plaintext: BigUint,
        p_max_trace: BigUint,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::sparse_level_poly_from_row_with_metadata(
            ctx,
            num_coefficient_slots,
            window,
            target_q_idx,
            target_row,
            max_plaintext,
            p_max_trace,
            circuit,
        )
    }

    fn slot_transfer(
        &self,
        src_slots: &[(u32, Option<Vec<u64>>)],
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.slot_transfer(src_slots, circuit)
    }

    fn add(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.add(other, circuit)
    }

    fn sub(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.sub(other, circuit)
    }

    fn mul(&self, other: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        self.mul(other, circuit)
    }

    fn mul_right_sparse(
        &self,
        other: &Self,
        rhs_q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        self.mul_right_sparse(other, rhs_q_idx, circuit)
    }

    fn full_reduce(&self, circuit: &mut PolyCircuit<P>) -> Self {
        self.full_reduce(circuit)
    }

    fn prepare_for_reconstruct(&self, circuit: &mut PolyCircuit<P>) -> Self {
        self.prepare_for_reconstruct(circuit)
    }

    fn const_mul(&self, tower_constants: &[u64], circuit: &mut PolyCircuit<P>) -> Self {
        self.const_mul(tower_constants, circuit)
    }

    fn reconstruct(&self, circuit: &mut PolyCircuit<P>) -> GateId {
        self.reconstruct(circuit)
    }
}

impl<P: Poly + 'static> ModularArithmeticPlanner<P> for NestedRnsPoly<P> {
    type Metadata = NestedRnsPlannerMetadata;
    type AddPlanKey = NestedRnsAddPlanKey;
    type SubPlanKey = NestedRnsSubPlanKey;

    fn metadata(entry: &Self) -> Self::Metadata {
        entry.planner_metadata()
    }

    fn normalized_metadata(ctx: &Self::Context, window: CrtWindow) -> Self::Metadata {
        Self::normalized_planner_metadata(ctx, window)
    }

    fn input_with_planner_metadata(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        metadata: &Self::Metadata,
        circuit: &mut PolyCircuit<P>,
    ) -> Self {
        Self::input_with_metadata(
            ctx,
            num_coefficient_slots,
            window,
            metadata.max_plaintexts.clone(),
            metadata.p_max_traces.clone(),
            circuit,
        )
    }

    fn from_flat_outputs_with_planner_metadata(
        template: &Self,
        outputs: &[GateId],
        metadata: &Self::Metadata,
    ) -> Self {
        Self::from_flat_outputs(
            template,
            outputs,
            metadata.max_plaintexts.clone(),
            metadata.p_max_traces.clone(),
        )
    }

    fn compute_add_plan_and_output(
        left: &Self,
        right: &Self,
    ) -> BinaryPlannerResult<Self::AddPlanKey, Self::Metadata> {
        debug_assert_eq!(left.max_plaintexts.len(), right.max_plaintexts.len());
        let p_full =
            <NestedRnsPolyContext as ModularArithmeticContext<P>>::plaintext_capacity_bound(
                left.ctx.as_ref(),
            );
        let pre_full_reduce = left
            .max_plaintexts
            .par_iter()
            .zip(right.max_plaintexts.par_iter())
            .any(|(lhs_bound, rhs_bound)| lhs_bound + rhs_bound > p_full);
        let left_before_reduce = if pre_full_reduce {
            Self::normalized_planner_metadata(left.ctx.as_ref(), left.window)
        } else {
            left.planner_metadata()
        };
        let right_before_reduce = if pre_full_reduce {
            Self::normalized_planner_metadata(right.ctx.as_ref(), right.window)
        } else {
            right.planner_metadata()
        };
        let reduce_levels = left_before_reduce
            .p_max_traces
            .par_iter()
            .zip(right_before_reduce.p_max_traces.par_iter())
            .map(|(lhs_trace, rhs_trace)| lhs_trace + rhs_trace >= left.ctx.lut_mod_p_max_map_size)
            .collect::<Vec<_>>();
        let reduced_trace = left.ctx.reduced_p_max_trace();
        let reduce_traces = |traces: &[BigUint]| {
            traces
                .iter()
                .zip(reduce_levels.iter())
                .map(|(trace, reduce)| if *reduce { reduced_trace.clone() } else { trace.clone() })
                .collect::<Vec<_>>()
        };
        let left_after_reduce = NestedRnsPlannerMetadata {
            max_plaintexts: left_before_reduce.max_plaintexts.clone(),
            p_max_traces: reduce_traces(&left_before_reduce.p_max_traces),
        };
        let right_after_reduce = NestedRnsPlannerMetadata {
            max_plaintexts: right_before_reduce.max_plaintexts.clone(),
            p_max_traces: reduce_traces(&right_before_reduce.p_max_traces),
        };
        BinaryPlannerResult {
            cache_key: NestedRnsAddPlanKey { pre_full_reduce, reduce_levels },
            output_metadata: NestedRnsPlannerMetadata {
                max_plaintexts: left_after_reduce
                    .max_plaintexts
                    .par_iter()
                    .zip(right_after_reduce.max_plaintexts.par_iter())
                    .map(|(lhs_bound, rhs_bound)| lhs_bound + rhs_bound)
                    .collect(),
                p_max_traces: left_after_reduce
                    .p_max_traces
                    .par_iter()
                    .zip(right_after_reduce.p_max_traces.par_iter())
                    .map(|(lhs_trace, rhs_trace)| lhs_trace + rhs_trace)
                    .collect(),
            },
        }
    }

    fn compute_sub_plan_and_output(
        left: &Self,
        right: &Self,
    ) -> BinaryPlannerResult<Self::SubPlanKey, Self::Metadata> {
        debug_assert_eq!(left.max_plaintexts.len(), right.max_plaintexts.len());
        let p_full =
            <NestedRnsPolyContext as ModularArithmeticContext<P>>::plaintext_capacity_bound(
                left.ctx.as_ref(),
            );
        let pre_full_reduce = left
            .active_q_moduli()
            .par_iter()
            .enumerate()
            .any(|(q_idx, &q_i)| &left.max_plaintexts[q_idx] + BigUint::from(q_i - 1) > p_full);
        let left_before_reduce = if pre_full_reduce {
            Self::normalized_planner_metadata(left.ctx.as_ref(), left.window)
        } else {
            left.planner_metadata()
        };
        let right_before_reduce = if pre_full_reduce {
            Self::normalized_planner_metadata(right.ctx.as_ref(), right.window)
        } else {
            right.planner_metadata()
        };
        let p_max_minus_one = left.ctx.reduced_p_max_trace();
        let p_max = &p_max_minus_one + BigUint::from(1u64);
        let trace_multiplier = |trace: &BigUint| (trace + &p_max_minus_one) / &p_max;
        let predicted_traces = left_before_reduce
            .p_max_traces
            .par_iter()
            .zip(right_before_reduce.p_max_traces.par_iter())
            .map(|(lhs_trace, rhs_trace)| lhs_trace + trace_multiplier(rhs_trace) * &p_max)
            .collect::<Vec<_>>();
        let reduce_levels = predicted_traces
            .par_iter()
            .map(|trace| trace >= &left.ctx.lut_mod_p_max_map_size)
            .collect::<Vec<_>>();
        let reduced_trace = left.ctx.reduced_p_max_trace();
        let reduce_traces = |traces: &[BigUint]| {
            traces
                .iter()
                .zip(reduce_levels.iter())
                .map(|(trace, reduce)| if *reduce { reduced_trace.clone() } else { trace.clone() })
                .collect::<Vec<_>>()
        };
        let left_after_reduce = NestedRnsPlannerMetadata {
            max_plaintexts: left_before_reduce.max_plaintexts.clone(),
            p_max_traces: reduce_traces(&left_before_reduce.p_max_traces),
        };
        let right_after_reduce = NestedRnsPlannerMetadata {
            max_plaintexts: right_before_reduce.max_plaintexts.clone(),
            p_max_traces: reduce_traces(&right_before_reduce.p_max_traces),
        };
        let trace_multipliers =
            right_after_reduce.p_max_traces.par_iter().map(trace_multiplier).collect::<Vec<_>>();
        BinaryPlannerResult {
            cache_key: NestedRnsSubPlanKey {
                pre_full_reduce,
                reduce_levels,
                trace_multipliers: trace_multipliers.clone(),
            },
            output_metadata: NestedRnsPlannerMetadata {
                max_plaintexts: left_after_reduce
                    .max_plaintexts
                    .par_iter()
                    .zip(left.active_q_moduli().par_iter())
                    .map(|(lhs_bound, &q_i)| lhs_bound + BigUint::from(q_i - 1))
                    .collect(),
                p_max_traces: left_after_reduce
                    .p_max_traces
                    .par_iter()
                    .zip(trace_multipliers.par_iter())
                    .map(|(lhs_trace, multiplier)| lhs_trace + multiplier * &p_max)
                    .collect(),
            },
        }
    }

    fn normalize_mul_input(entry: &Self, circuit: &mut PolyCircuit<P>) -> Self {
        let reduced_metadata = Self::normalized_planner_metadata(entry.ctx.as_ref(), entry.window);
        let needs_full_reduce = entry
            .max_plaintexts
            .iter()
            .zip(reduced_metadata.max_plaintexts.iter())
            .any(|(current, reduced)| current > reduced);
        let needs_trace_reduce = entry
            .p_max_traces
            .iter()
            .zip(reduced_metadata.p_max_traces.iter())
            .any(|(current, reduced)| current > reduced);
        if needs_full_reduce {
            entry.full_reduce(circuit)
        } else if needs_trace_reduce {
            entry.prepare_for_reconstruct(circuit)
        } else {
            entry.clone()
        }
    }
}

impl<P: Poly + 'static> DecomposeArithmeticGadget<P> for NestedRnsPoly<P> {
    fn gadget_matrix<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        window: CrtWindow,
    ) -> M {
        M::from_poly_vec_row(
            params,
            nested_rns_gadget_vector::<P, M>(params, ctx, window).get_row(0),
        )
    }

    fn gadget_decomposed<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        target: &M,
        window: CrtWindow,
    ) -> M {
        nested_rns_gadget_decomposed(params, ctx, target, window)
    }

    fn gadget_constant_coeffs<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        window: CrtWindow,
    ) -> Vec<BigUint> {
        let active_q_moduli = encoding::resolve_nested_rns_active_window(ctx, window);
        let reconst_coeffs = encoding::nested_rns_level_reconstruction_coeffs(&active_q_moduli);
        let chunk_width =
            <NestedRnsPolyContext as ModularArithmeticContext<P>>::decomposition_len(ctx);
        let mut constants = Vec::with_capacity(active_q_moduli.len() * chunk_width);
        for (q_idx, level_values) in
            ctx.gadget_values[window.offset..window.end()].iter().enumerate()
        {
            for residue in level_values {
                let row = ctx
                    .p_moduli
                    .iter()
                    .map(|&p_i| (residue % BigUint::from(p_i)).to_u64().expect("row residue fits"))
                    .collect::<Vec<_>>();
                constants.push(encoding::nested_rns_sparse_level_slot_value::<P>(
                    params,
                    ctx,
                    &reconst_coeffs[q_idx],
                    &row,
                ));
            }
        }
        constants
    }

    fn gadget_decomposed_constant_tower_coeffs<M: PolyMatrix<P = P>>(
        params: &P::Params,
        ctx: &Self::Context,
        constant: BigUint,
        window: CrtWindow,
    ) -> Vec<Vec<u64>> {
        let active_q_moduli = encoding::resolve_nested_rns_active_window(ctx, window);
        let chunk_width =
            <NestedRnsPolyContext as ModularArithmeticContext<P>>::decomposition_len(ctx);
        let reconst_coeffs = encoding::nested_rns_level_reconstruction_coeffs(&active_q_moduli);
        let mut output = Vec::with_capacity(active_q_moduli.len() * chunk_width);
        for (q_idx, &q_i) in active_q_moduli.iter().enumerate() {
            let q_i_big = BigUint::from(q_i);
            let input_residue =
                (&constant % &q_i_big).to_u64().expect("q-level residue must fit in u64");
            let input_row = ctx.p_moduli.iter().map(|&p_i| input_residue % p_i).collect::<Vec<_>>();
            let (ys, w) = encoding::nested_rns_decomposition_terms_from_row(ctx, &input_row);
            for digit_idx in 0..chunk_width {
                let scalar = if digit_idx < ctx.p_moduli.len() {
                    ys[digit_idx].to_u64().expect("decomposition digit must fit in u64")
                } else {
                    w.to_u64().expect("rounding digit must fit in u64")
                };
                let encoded_row = ctx.p_moduli.iter().map(|&p_i| scalar % p_i).collect::<Vec<_>>();
                let coeff = encoding::nested_rns_sparse_level_slot_value::<P>(
                    params,
                    ctx,
                    &reconst_coeffs[q_idx],
                    &encoded_row,
                );
                output.push(
                    active_q_moduli
                        .iter()
                        .map(|&tower_q| {
                            (&coeff % BigUint::from(tower_q))
                                .to_u64()
                                .expect("decomposition tower residue must fit in u64")
                        })
                        .collect::<Vec<_>>(),
                );
            }
        }
        output
    }

    fn gadget_decomposition_norm_bound(ctx: &Self::Context, window: CrtWindow) -> BigUint {
        let levels =
            <NestedRnsPolyContext as ModularArithmeticContext<P>>::validate_window(ctx, window)
                .depth;
        BigUint::from(
            u64::try_from(
                levels *
                    <NestedRnsPolyContext as ModularArithmeticContext<P>>::decomposition_len(ctx),
            )
            .expect("gadget decomposition width must fit in u64"),
        )
    }

    fn randomizer_decomposition_norm_bound(ctx: &Self::Context, _window: CrtWindow) -> BigUint {
        BigUint::from(
            *ctx.p_moduli
                .iter()
                .max()
                .expect("NestedRnsPolyContext requires at least one p modulus"),
        )
    }

    fn gadget_vector(
        ctx: Arc<Self::Context>,
        num_coefficient_slots: usize,
        window: CrtWindow,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        NestedRnsPoly::gadget_vector(ctx, num_coefficient_slots, window, circuit)
    }

    fn gadget_decompose(&self, circuit: &mut PolyCircuit<P>) -> Vec<Self> {
        self.gadget_decompose(circuit)
    }

    fn decomposition_terms_for_level(
        &self,
        _q_idx: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> (Vec<GateId>, GateId) {
        self.decomposition_terms(circuit)
    }

    fn conv_mul_right_decomposed_many(
        &self,
        params: &P::Params,
        left_rows: &[&[Self]],
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> Vec<Self> {
        self.conv_mul_right_decomposed_many(params, left_rows, num_slots, circuit)
    }

    fn mul_rows_with_decomposed_rhs(
        params: &P::Params,
        lhs_row0: &[Self],
        lhs_row1: &[Self],
        rhs_top: &Self,
        rhs_bottom: &Self,
        num_slots: usize,
        circuit: &mut PolyCircuit<P>,
    ) -> [Self; 2] {
        super::decomposed_mul::mul_rows_with_decomposed_rhs(
            params, lhs_row0, lhs_row1, rhs_top, rhs_bottom, num_slots, circuit,
        )
    }
}

#[cfg(test)]
mod full_reduce_tests {
    use super::*;
    use crate::{
        circuit_gadgets::arith::{CrtWindow, DEFAULT_MAX_UNREDUCED_MULS},
        poly::{
            PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
        test_utils::{PolyVec, execute_polyvec_circuit},
    };

    const SCALE: u64 = 1 << 8;

    fn parameters() -> DCRTPolyParams {
        DCRTPolyParams::new(2, 3, 12, 6)
    }

    /// Smallest p-basis width supporting the full-reduce test parameters under the default
    /// unreduced-multiplication budget.
    fn test_p_moduli_bits() -> usize {
        super::super::encoding::minimum_p_moduli_bits(
            *parameters().to_crt().0.iter().max().expect("nonempty CRT basis"),
            DEFAULT_MAX_UNREDUCED_MULS,
        )
        .expect("test parameters support a p basis")
    }

    fn run_explicit_full_reduce(values: &[BigUint], window: CrtWindow) -> Vec<BigUint> {
        let parameters = parameters();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let context = Arc::new(NestedRnsPolyContext::setup(
            &mut circuit,
            &parameters,
            test_p_moduli_bits(),
            DEFAULT_MAX_UNREDUCED_MULS,
            SCALE,
            false,
            None,
        ));
        let outputs = values
            .iter()
            .map(|_| {
                let input = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
                input.full_reduce(&mut circuit).reconstruct(&mut circuit)
            })
            .collect::<Vec<_>>();
        circuit.output(outputs);

        let inputs = values
            .iter()
            .flat_map(|value| {
                encode_nested_rns_poly::<DCRTPoly>(
                    context.p_moduli_bits,
                    context.max_unreduced_muls,
                    &parameters,
                    std::slice::from_ref(value),
                    window,
                )
            })
            .map(|lanes| {
                assert_eq!(lanes.len(), window.depth);
                PolyVec(
                    lanes
                        .into_iter()
                        .map(|lane| DCRTPoly::from_biguint_to_constant(&parameters, lane))
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let outputs = execute_polyvec_circuit(
            "nested-rns-full-reduce-runtime",
            &parameters,
            &circuit,
            inputs,
            window.depth,
        );
        outputs.into_iter().map(|output| output.0[0].coeffs_biguints()[0].clone()).collect()
    }

    #[test]
    fn explicit_full_reduce_matches_inputs_for_full_and_offset_windows() {
        let parameters = parameters();
        let (q_moduli, _, depth) = parameters.to_crt();
        let windows = [
            CrtWindow::full(depth),
            CrtWindow::new(0, 1, depth),
            CrtWindow::new(1, 1, depth),
            CrtWindow::new(1, 2, depth),
        ];
        for window in windows {
            let modulus = q_moduli[window.offset..window.end()]
                .iter()
                .fold(BigUint::from(1u8), |acc, &q_i| acc * BigUint::from(q_i));
            let values = [
                BigUint::ZERO,
                BigUint::from(1u8),
                BigUint::from(123u16),
                &modulus - BigUint::from(1u8),
            ];
            let actual = run_explicit_full_reduce(&values, window);
            for (actual, value) in actual.into_iter().zip(values) {
                assert_eq!(
                    actual % &modulus,
                    value % &modulus,
                    "full_reduce mismatch for window={window:?}"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        circuit::{GateParamSource, PolyGateKind, PolyGateType, SlotTransferSpec},
        circuit_gadgets::arith::DecomposeArithmeticGadget,
        test_utils::{diagonal_matrix, execute_circuit_with_shape},
        utils::{ceil_biguint_nth_root, gen_biguint_for_modulus, pow_biguint_usize},
    };
    use mxx_primitives::{
        matrix::{PolyMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{
            Poly as ConcretePoly, PolyParams,
            dcrt::{params::DCRTPolyParams, poly::DCRTPoly},
        },
    };
    use num_traits::{One, ToPrimitive, Zero};

    const SCALE: u64 = 1 << 8;

    fn test_parameters() -> DCRTPolyParams {
        DCRTPolyParams::new(2, 3, 12, 6)
    }

    /// Smallest p-basis width supporting the shared test parameters under the default
    /// unreduced-multiplication budget, so budget changes never silently starve these tests.
    fn test_p_moduli_bits() -> usize {
        super::super::encoding::minimum_p_moduli_bits(
            *test_parameters().to_crt().0.iter().max().expect("nonempty CRT basis"),
            DEFAULT_MAX_UNREDUCED_MULS,
        )
        .expect("test parameters support a p basis")
    }

    fn create_context(
        circuit: &mut PolyCircuit<DCRTPoly>,
        q_level: Option<usize>,
    ) -> (DCRTPolyParams, Arc<NestedRnsPolyContext>) {
        let parameters = test_parameters();
        let context = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            &parameters,
            test_p_moduli_bits(),
            DEFAULT_MAX_UNREDUCED_MULS,
            SCALE,
            false,
            q_level,
        ));
        (parameters, context)
    }

    fn create_context_with_config(
        circuit: &mut PolyCircuit<DCRTPoly>,
        q_level: Option<usize>,
        p_moduli_bits: usize,
        max_unreduced_muls: usize,
    ) -> (DCRTPolyParams, Arc<NestedRnsPolyContext>) {
        let parameters = DCRTPolyParams::new(2, 3, 18, 6);
        let context = Arc::new(NestedRnsPolyContext::setup(
            circuit,
            &parameters,
            p_moduli_bits,
            max_unreduced_muls,
            SCALE,
            false,
            q_level,
        ));
        (parameters, context)
    }

    fn active_modulus(parameters: &DCRTPolyParams, window: CrtWindow) -> BigUint {
        let (q_moduli, _, _) = parameters.to_crt();
        q_moduli[window.offset..window.end()]
            .par_iter()
            .copied()
            .map(BigUint::from)
            .reduce(BigUint::one, |left, right| left * right)
    }

    fn encode_value(
        context: &NestedRnsPolyContext,
        parameters: &DCRTPolyParams,
        value: &BigUint,
        window: CrtWindow,
    ) -> Vec<DCRTPolyMatrix> {
        encode_nested_rns_poly::<DCRTPoly>(
            context.p_moduli_bits,
            context.max_unreduced_muls,
            parameters,
            std::slice::from_ref(value),
            window,
        )
        .into_iter()
        .map(|lanes| {
            diagonal_matrix(
                parameters,
                lanes.into_iter().map(|lane| DCRTPoly::from_biguint_to_constant(parameters, lane)),
            )
        })
        .collect()
    }

    fn execute_constant_output(
        name: &str,
        parameters: &DCRTPolyParams,
        circuit: &PolyCircuit<DCRTPoly>,
        inputs: impl IntoIterator<Item = DCRTPolyMatrix>,
    ) -> BigUint {
        let inputs = inputs.into_iter().collect::<Vec<_>>();
        let wire_size = inputs.first().expect("nested-RNS circuit requires inputs").row_size();
        let outputs =
            execute_circuit_with_shape(name, parameters, circuit, &inputs, (wire_size, wire_size));
        assert_eq!(outputs.len(), 1);
        outputs[0].entry(0, 0).coeffs_biguints()[0].clone()
    }

    fn random_value(modulus: &BigUint) -> BigUint {
        let mut rng = rand::rng();
        gen_biguint_for_modulus(&mut rng, modulus)
    }

    #[derive(Clone, Copy)]
    enum BinaryOperation {
        Add,
        Sub,
        Mul,
    }

    fn test_binary_case(
        operation: BinaryOperation,
        window: CrtWindow,
        left_value: BigUint,
        right_value: BigUint,
    ) {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (parameters, context) = create_context(&mut circuit, None);
        let modulus = active_modulus(&parameters, window);
        let left = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
        let right = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
        let result = match operation {
            BinaryOperation::Add => left.add(&right, &mut circuit),
            BinaryOperation::Sub => left.sub(&right, &mut circuit),
            BinaryOperation::Mul => left.mul(&right, &mut circuit),
        };
        let output = result.reconstruct(&mut circuit);
        circuit.output([output]);

        let mut inputs = encode_value(&context, &parameters, &left_value, window);
        inputs.extend(encode_value(&context, &parameters, &right_value, window));
        let actual =
            execute_constant_output("nested-rns-binary-runtime", &parameters, &circuit, inputs);
        let expected = match operation {
            BinaryOperation::Add => left_value + right_value,
            BinaryOperation::Sub => left_value + &modulus - right_value,
            BinaryOperation::Mul => left_value * right_value,
        };
        assert_eq!(actual % &modulus, expected % modulus);
    }

    fn run_binary_cases(operation: BinaryOperation, window: CrtWindow) {
        let parameters = test_parameters();
        let modulus = active_modulus(&parameters, window);
        let boundary = &modulus - BigUint::one();
        let cases = match operation {
            BinaryOperation::Add => vec![
                (boundary.clone(), boundary.clone()),
                (random_value(&modulus), random_value(&modulus)),
            ],
            BinaryOperation::Sub => {
                vec![(BigUint::zero(), boundary), (random_value(&modulus), random_value(&modulus))]
            }
            BinaryOperation::Mul => {
                vec![(boundary.clone(), boundary), (random_value(&modulus), random_value(&modulus))]
            }
        };
        cases.into_par_iter().for_each(|(left, right)| {
            test_binary_case(operation, window, left, right);
        });
    }

    #[test]
    fn add_sub_mul_match_boundary_and_random_arithmetic_at_runtime() {
        [BinaryOperation::Add, BinaryOperation::Sub, BinaryOperation::Mul]
            .into_par_iter()
            .for_each(|operation| run_binary_cases(operation, CrtWindow::full(3)));
    }

    #[test]
    fn arithmetic_respects_partial_level_windows_at_runtime() {
        [BinaryOperation::Add, BinaryOperation::Sub, BinaryOperation::Mul]
            .into_par_iter()
            .for_each(|operation| run_binary_cases(operation, CrtWindow::new(1, 2, 3)));
    }

    #[test]
    fn packed_arithmetic_shape_is_independent_of_active_window_position() {
        fn counts(level_offset: usize) -> std::collections::HashMap<PolyGateKind, usize> {
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let (_, context) = create_context(&mut circuit, None);
            let window = CrtWindow::new(level_offset, 1, context.q_moduli_depth);
            let left = NestedRnsPoly::input(context.clone(), 2, window, &mut circuit);
            let right = NestedRnsPoly::input(context, 2, window, &mut circuit);
            let output = left.mul(&right, &mut circuit).full_reduce(&mut circuit);
            circuit.output([output.inner]);
            circuit.count_gates_by_type_vec()
        }

        assert_eq!(counts(0), counts(1));
    }

    fn sparse_gadget_entry(
        context: Arc<NestedRnsPolyContext>,
        target_q_idx: usize,
        circuit: &mut PolyCircuit<DCRTPoly>,
    ) -> NestedRnsPoly<DCRTPoly> {
        let value = context.gadget_values[target_q_idx][0].clone();
        NestedRnsPoly::sparse_constant_level_poly(
            context.clone(),
            1,
            CrtWindow::full(context.q_moduli_depth),
            target_q_idx,
            &value,
            circuit,
        )
    }

    #[test]
    fn sparse_multiplication_matches_generic_multiplication_at_runtime() {
        let target_q_idx = 1;
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (parameters, context) = create_context(&mut circuit, None);
        let window = CrtWindow::full(context.q_moduli_depth);
        let input = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
        let sparse = sparse_gadget_entry(context.clone(), target_q_idx, &mut circuit);
        let sparse_product = input.mul_right_sparse(&sparse, target_q_idx, &mut circuit);
        let generic_product = input.mul(&sparse, &mut circuit);
        let sparse_output = sparse_product.reconstruct(&mut circuit);
        let generic_output = generic_product.reconstruct(&mut circuit);
        circuit.output([sparse_output, generic_output]);

        let modulus = active_modulus(&parameters, window);
        [BigUint::zero(), BigUint::from(7u8), random_value(&modulus)].into_par_iter().for_each(
            |value| {
                let inputs = encode_value(&context, &parameters, &value, window);
                let wire_size = inputs[0].row_size();
                let outputs = execute_circuit_with_shape(
                    "nested-rns-sparse-runtime",
                    &parameters,
                    &circuit,
                    &inputs,
                    (wire_size, wire_size),
                );
                assert_eq!(outputs[0], outputs[1]);
            },
        );

        let (generic_counts, sparse_counts) = rayon::join(
            || {
                let mut generic_circuit = PolyCircuit::<DCRTPoly>::new();
                let (_, generic_context) = create_context(&mut generic_circuit, None);
                let window = CrtWindow::full(generic_context.q_moduli_depth);
                let left =
                    NestedRnsPoly::input(generic_context.clone(), 1, window, &mut generic_circuit);
                let right =
                    sparse_gadget_entry(generic_context, target_q_idx, &mut generic_circuit);
                let product = left.mul(&right, &mut generic_circuit);
                let output = product.reconstruct(&mut generic_circuit);
                generic_circuit.output([output]);
                generic_circuit.count_gates_by_type_vec()
            },
            || {
                let mut sparse_circuit = PolyCircuit::<DCRTPoly>::new();
                let (_, sparse_context) = create_context(&mut sparse_circuit, None);
                let window = CrtWindow::full(sparse_context.q_moduli_depth);
                let left =
                    NestedRnsPoly::input(sparse_context.clone(), 1, window, &mut sparse_circuit);
                let right = sparse_gadget_entry(sparse_context, target_q_idx, &mut sparse_circuit);
                let product = left.mul_right_sparse(&right, target_q_idx, &mut sparse_circuit);
                let output = product.reconstruct(&mut sparse_circuit);
                sparse_circuit.output([output]);
                sparse_circuit.count_gates_by_type_vec()
            },
        );
        assert_eq!(
            sparse_counts.get(&PolyGateKind::Mul).copied().unwrap_or_default(),
            generic_counts.get(&PolyGateKind::Mul).copied().unwrap_or_default(),
            "packed sparse multiplication shares the same lane-uniform multiplication kernel",
        );
    }

    #[test]
    #[should_panic(
        expected = "mul_right_sparse requires the right operand to be zero outside q_idx"
    )]
    fn sparse_multiplication_rejects_a_dense_right_operand() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, context) = create_context(&mut circuit, None);
        let window = CrtWindow::full(context.q_moduli_depth);
        let left = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
        let right = NestedRnsPoly::input(context, 1, window, &mut circuit);
        let _ = left.mul_right_sparse(&right, 0, &mut circuit);
    }

    #[test]
    fn gadget_decomposition_recomposes_runtime_and_native_values() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (parameters, context) = create_context(&mut circuit, Some(2));
        let window = CrtWindow::new(0, 2, context.q_moduli_depth);
        let input = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
        let gadget = NestedRnsPoly::gadget_vector(context.clone(), 1, window, &mut circuit);
        let decomposition = input.gadget_decompose(&mut circuit);
        assert_eq!(gadget.len(), decomposition.len());
        let mut terms = gadget.iter().zip(&decomposition);
        let (first_gadget, first_digit) = terms.next().expect("gadget vector is non-empty");
        let mut recomposed = first_gadget.mul(first_digit, &mut circuit);
        for (gadget, digit) in terms {
            let product = gadget.mul(digit, &mut circuit);
            recomposed = recomposed.add(&product, &mut circuit);
        }
        let output = recomposed.reconstruct(&mut circuit);
        circuit.output([output]);
        let modulus = active_modulus(&parameters, window);
        let value = random_value(&modulus);
        let inputs = encode_value(&context, &parameters, &value, window);
        let actual = execute_constant_output(
            "nested-rns-decomposition-runtime",
            &parameters,
            &circuit,
            inputs,
        );
        assert_eq!(actual % &modulus, value.clone() % &modulus);

        let target = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            vec![vec![DCRTPoly::from_biguints(
                &parameters,
                &[BigUint::from(3u8), BigUint::from(5u8)],
            )]],
        );
        let mut decomposition_circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, decomposition_context) = create_context(&mut decomposition_circuit, Some(1));
        let decomposition_window = CrtWindow::new(0, 1, decomposition_context.q_moduli_depth);
        let ring_dimension = parameters.ring_dimension() as usize;
        let inputs = (0..target.row_size())
            .map(|_| {
                (0..target.col_size())
                    .map(|_| {
                        NestedRnsPoly::input(
                            decomposition_context.clone(),
                            ring_dimension,
                            decomposition_window,
                            &mut decomposition_circuit,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let decomposed = inputs
            .iter()
            .map(|row| {
                row.iter()
                    .map(|entry| entry.gadget_decompose(&mut decomposition_circuit))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let gadget_len = decomposed[0][0].len();
        let mut output_gates =
            Vec::with_capacity(target.row_size() * target.col_size() * gadget_len);
        for row in &decomposed {
            for digit_index in 0..gadget_len {
                for column in row {
                    output_gates.push(column[digit_index].reconstruct(&mut decomposition_circuit));
                }
            }
        }
        decomposition_circuit.output(output_gates);

        let mut encoded_inputs = Vec::new();
        for row_index in 0..target.row_size() {
            for column_index in 0..target.col_size() {
                let coefficients = target.entry(row_index, column_index).coeffs_biguints();
                encoded_inputs.extend(
                    encode_nested_rns_poly::<DCRTPoly>(
                        decomposition_context.p_moduli_bits,
                        decomposition_context.max_unreduced_muls,
                        &parameters,
                        &coefficients,
                        decomposition_window,
                    )
                    .into_iter()
                    .map(|lanes| {
                        diagonal_matrix(
                            &parameters,
                            lanes.into_iter().map(|value| {
                                DCRTPoly::from_biguint_to_constant(&parameters, value)
                            }),
                        )
                    }),
                );
            }
        }
        let wire_size = ring_dimension * decomposition_window.depth;
        let runtime_outputs = execute_circuit_with_shape(
            "nested-rns-matrix-decomposition-runtime",
            &parameters,
            &decomposition_circuit,
            &encoded_inputs,
            (wire_size, wire_size),
        );
        let runtime_polys = runtime_outputs
            .into_par_iter()
            .map(|matrix| {
                DCRTPoly::from_biguints(
                    &parameters,
                    &(0..ring_dimension)
                        .map(|slot| {
                            let physical = slot * decomposition_window.depth;
                            matrix.entry(physical, physical).coeffs_biguints()[0].clone()
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .collect::<Vec<_>>();
        let runtime_decomposition = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            runtime_polys.chunks(target.col_size()).map(|row| row.to_vec()).collect(),
        );
        let native_decomposition = NestedRnsPoly::<DCRTPoly>::gadget_decomposed::<DCRTPolyMatrix>(
            &parameters,
            &decomposition_context,
            &target,
            decomposition_window,
        );
        assert_eq!(runtime_decomposition.size(), native_decomposition.size());
        let coefficient_modulus = parameters.modulus();
        let columns = runtime_decomposition.col_size();
        (0..runtime_decomposition.row_size() * columns)
            .into_par_iter()
            .for_each(|index| {
                let row = index / columns;
                let column = index % columns;
                let reduce_coefficients = |polynomial: DCRTPoly| {
                    polynomial
                        .coeffs_biguints()
                        .into_iter()
                        .map(|coefficient| coefficient % coefficient_modulus.as_ref())
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    reduce_coefficients(runtime_decomposition.entry(row, column)),
                    reduce_coefficients(native_decomposition.entry(row, column)),
                    "runtime and native decomposition must have the same coefficients modulo q at ({row}, {column})"
                );
            });
    }

    #[test]
    fn gadget_decomposed_preserves_random_matrix_entry_layout() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (parameters, context) = create_context(&mut circuit, Some(1));
        let window = CrtWindow::new(0, 1, context.q_moduli_depth);
        let modulus = parameters.modulus();
        let entries = (0..6)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::rng();
                DCRTPoly::from_biguint_to_constant(
                    &parameters,
                    gen_biguint_for_modulus(&mut rng, modulus.as_ref()),
                )
            })
            .collect::<Vec<_>>();
        let target = DCRTPolyMatrix::from_poly_vec(
            &parameters,
            entries.chunks(3).map(|row| row.to_vec()).collect(),
        );
        let decomposed = NestedRnsPoly::<DCRTPoly>::gadget_decomposed::<DCRTPolyMatrix>(
            &parameters,
            &context,
            &target,
            window,
        );
        let gadget_len = decomposed.row_size() / target.row_size();
        assert_eq!(decomposed.col_size(), target.col_size());
        assert!(gadget_len > 0);

        (0..target.row_size() * target.col_size()).into_par_iter().for_each(|entry_index| {
            let row = entry_index / target.col_size();
            let column = entry_index % target.col_size();
            let single =
                DCRTPolyMatrix::from_poly_vec(&parameters, vec![vec![target.entry(row, column)]]);
            let expected = NestedRnsPoly::<DCRTPoly>::gadget_decomposed::<DCRTPolyMatrix>(
                &parameters,
                &context,
                &single,
                window,
            );
            assert_eq!(expected.size(), (gadget_len, 1));
            for digit in 0..gadget_len {
                // This test isolates matrix row/column placement. Preservation of every ring
                // coefficient is covered by
                // `gadget_decomposition_recomposes_runtime_and_native_values`.
                let actual_constant =
                    decomposed.entry(row * gadget_len + digit, column).coeffs_biguints()[0].clone();
                let expected_constant = expected.entry(digit, 0).coeffs_biguints()[0].clone();
                assert_eq!(
                    actual_constant, expected_constant,
                    "decomposed matrix entry ({row}, {column}), digit {digit}"
                );
            }
        });
    }

    #[test]
    fn unreduced_decomposition_matches_explicit_lazy_reduction_at_runtime() {
        fn build(
            explicit_reduce: bool,
        ) -> (DCRTPolyParams, Arc<NestedRnsPolyContext>, PolyCircuit<DCRTPoly>) {
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let (parameters, context) = create_context(&mut circuit, Some(1));
            let window = CrtWindow::new(0, 1, context.q_moduli_depth);
            let left = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
            let right = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
            let sum = left.add(&right, &mut circuit);
            let sum =
                if explicit_reduce { sum.lazy_reduce_if_unreduced(&mut circuit) } else { sum };
            let outputs = sum
                .gadget_decompose(&mut circuit)
                .into_iter()
                .map(|term| term.reconstruct(&mut circuit))
                .collect::<Vec<_>>();
            circuit.output(outputs);
            (parameters, context, circuit)
        }

        let (parameters, automatic_context, automatic) = build(false);
        let (_, manual_context, manual) = build(true);
        let window = CrtWindow::new(0, 1, automatic_context.q_moduli_depth);
        let modulus = active_modulus(&parameters, window);
        let left = random_value(&modulus);
        let right = random_value(&modulus);
        let mut automatic_inputs = encode_value(&automatic_context, &parameters, &left, window);
        automatic_inputs.extend(encode_value(&automatic_context, &parameters, &right, window));
        let mut manual_inputs = encode_value(&manual_context, &parameters, &left, window);
        manual_inputs.extend(encode_value(&manual_context, &parameters, &right, window));
        let automatic_wire_size = automatic_inputs[0].row_size();
        let automatic_outputs = execute_circuit_with_shape(
            "nested-rns-auto-decompose-runtime",
            &parameters,
            &automatic,
            &automatic_inputs,
            (automatic_wire_size, automatic_wire_size),
        );
        let manual_wire_size = manual_inputs[0].row_size();
        let manual_outputs = execute_circuit_with_shape(
            "nested-rns-manual-decompose-runtime",
            &parameters,
            &manual,
            &manual_inputs,
            (manual_wire_size, manual_wire_size),
        );
        assert_eq!(automatic_outputs, manual_outputs);
        assert!(
            automatic.non_free_depth() <= manual.non_free_depth(),
            "automatic reduction must not increase non-free depth"
        );
    }

    #[test]
    fn reconstruct_auto_reduce_matches_explicit_full_reduce_at_runtime() {
        fn build(
            explicit_reduce: bool,
        ) -> (DCRTPolyParams, Arc<NestedRnsPolyContext>, PolyCircuit<DCRTPoly>) {
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let (parameters, context) = create_context(&mut circuit, Some(1));
            let window = CrtWindow::new(0, 1, context.q_moduli_depth);
            let mut input = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
            input.max_plaintexts = vec![context.p_full.clone()];
            let input = if explicit_reduce { input.full_reduce(&mut circuit) } else { input };
            let output = input.reconstruct(&mut circuit);
            circuit.output([output]);
            (parameters, context, circuit)
        }

        let (parameters, automatic_context, automatic) = build(false);
        let (_, manual_context, manual) = build(true);
        let value = BigUint::from(123u16);
        let window = CrtWindow::new(0, 1, automatic_context.q_moduli_depth);
        let automatic_inputs = encode_value(&automatic_context, &parameters, &value, window);
        let manual_inputs = encode_value(&manual_context, &parameters, &value, window);
        let automatic_wire_size = automatic_inputs[0].row_size();
        let automatic_output = execute_circuit_with_shape(
            "nested-rns-auto-reconstruct-runtime",
            &parameters,
            &automatic,
            &automatic_inputs,
            (automatic_wire_size, automatic_wire_size),
        );
        let manual_wire_size = manual_inputs[0].row_size();
        let manual_output = execute_circuit_with_shape(
            "nested-rns-manual-reconstruct-runtime",
            &parameters,
            &manual,
            &manual_inputs,
            (manual_wire_size, manual_wire_size),
        );
        assert_eq!(automatic_output, manual_output);
        assert_eq!(automatic.count_gates_by_type_vec(), manual.count_gates_by_type_vec());
    }

    #[test]
    fn q1_anchor_reconstruction_matches_full_reconstruction_on_active_window() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (parameters, context) = create_context(&mut circuit, None);
        let window = CrtWindow::new(1, 2, context.q_moduli_depth);
        let input = NestedRnsPoly::input(context.clone(), 2, window, &mut circuit);
        let full = input.reconstruct(&mut circuit);
        let anchors = input.reconstruct_q1_anchors(&mut circuit);
        circuit.output([full, anchors.anchor_wire()]);

        let values = [BigUint::from(17u8), BigUint::from(83u8)];
        let inputs = encode_nested_rns_poly::<DCRTPoly>(
            context.p_moduli_bits,
            context.max_unreduced_muls,
            &parameters,
            &values,
            window,
        )
        .into_iter()
        .map(|lanes| {
            diagonal_matrix(
                &parameters,
                lanes
                    .into_iter()
                    .map(|value| DCRTPoly::from_biguint_to_constant(&parameters, value)),
            )
        })
        .collect::<Vec<_>>();
        let physical_slots = 2 * window.depth;
        let outputs = execute_circuit_with_shape(
            "nested-rns-q1-anchor-reconstruction-runtime",
            &parameters,
            &circuit,
            &inputs,
            (physical_slots, physical_slots),
        );
        for coefficient in 0..values.len() {
            let anchor = anchors.anchor_slot(coefficient);
            assert_eq!(
                outputs[1].entry(anchor, anchor).coeffs_biguints(),
                outputs[0].entry(anchor, anchor).coeffs_biguints(),
                "q1 anchor for coefficient {coefficient} must equal full reconstruction"
            );
        }
    }

    #[test]
    fn q1_anchor_reconstruction_uses_one_compact_anchor_reduction() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, context) = create_context(&mut circuit, None);
        let window = CrtWindow::new(1, 2, context.q_moduli_depth);
        let input = NestedRnsPoly::input(context, 3, window, &mut circuit);
        let anchors = input.reconstruct_q1_anchors(&mut circuit);
        circuit.output([anchors.anchor_wire()]);

        let transfers = circuit
            .gates_in_id_order()
            .filter_map(|(_, gate)| match &gate.gate_type {
                PolyGateType::SlotTransfer { src_slots } => Some(src_slots),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(transfers.len(), 1, "one reduction serves every active CRT lane");
        let GateParamSource::Const(SlotTransferSpec::AnchorReduce { num_blocks, lane_scalars }) =
            transfers[0]
        else {
            panic!("q1 reconstruction must use the compact anchor reduction")
        };
        assert_eq!(*num_blocks, 3);
        assert_eq!(lane_scalars.len(), window.depth);
    }

    #[test]
    fn repeated_lane_masks_stay_compact_and_irregular_masks_fall_back_to_explicit() {
        let mut compact_circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, compact_context) = create_context(&mut compact_circuit, Some(2));
        let compact_window = CrtWindow::new(0, 2, compact_context.q_moduli_depth);
        let compact_input = NestedRnsPoly::input(
            compact_context.clone(),
            1 << 12,
            compact_window,
            &mut compact_circuit,
        );
        let _ = compact_input.const_mul(&[3, 5], &mut compact_circuit);
        let compact_specs = compact_circuit
            .gates_in_id_order()
            .filter_map(|(_, gate)| match &gate.gate_type {
                PolyGateType::SlotTransfer {
                    src_slots:
                        GateParamSource::Const(SlotTransferSpec::IdentityRepeatedLanes {
                            num_blocks,
                            lane_scalars,
                        }),
                } => Some((*num_blocks, lane_scalars)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(compact_specs.len(), compact_context.p_moduli.len());
        assert!(compact_specs.iter().all(|(blocks, scalars)| {
            *blocks == 1 << 12 && scalars.len() == compact_window.depth
        }));

        let mut repack_circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, repack_context) = create_context(&mut repack_circuit, Some(2));
        let repack_window = CrtWindow::new(0, 2, repack_context.q_moduli_depth);
        let repack_input = NestedRnsPoly::input(
            repack_context.clone(),
            1 << 12,
            repack_window,
            &mut repack_circuit,
        );
        let _ = repack_input.repack_window(repack_window, &mut repack_circuit);
        let repack_specs = repack_circuit
            .gates_in_id_order()
            .filter_map(|(_, gate)| match &gate.gate_type {
                PolyGateType::SlotTransfer {
                    src_slots:
                        GateParamSource::Const(SlotTransferSpec::IdentityRepeatedLanes {
                            num_blocks,
                            lane_scalars,
                        }),
                } => Some((*num_blocks, lane_scalars)),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(repack_specs.is_empty());

        let mut irregular_circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, irregular_context) = create_context(&mut irregular_circuit, Some(2));
        let irregular_window = CrtWindow::new(0, 2, irregular_context.q_moduli_depth);
        let irregular_input = NestedRnsPoly::input(
            irregular_context.clone(),
            2,
            irregular_window,
            &mut irregular_circuit,
        );
        let _ = irregular_input
            .slot_transfer(&[(0, Some(vec![1, 2])), (1, Some(vec![3, 4]))], &mut irregular_circuit);
        let explicit_lengths = irregular_circuit
            .gates_in_id_order()
            .filter_map(|(_, gate)| match &gate.gate_type {
                PolyGateType::SlotTransfer {
                    src_slots: GateParamSource::Const(SlotTransferSpec::Explicit(mapping)),
                } => Some(mapping.len()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(explicit_lengths.len(), irregular_context.p_moduli.len());
        assert!(explicit_lengths.iter().all(|length| *length == 4));
    }

    #[test]
    fn const_mul_and_slot_transfer_match_runtime_values_and_track_bounds() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (parameters, context) = create_context(&mut circuit, Some(2));
        let window = CrtWindow::new(0, 2, context.q_moduli_depth);
        let input = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
        let constants = [3u64, 5u64];
        let product = input.const_mul(&constants, &mut circuit);
        let output = product.reconstruct(&mut circuit);
        circuit.output([output]);
        let modulus = active_modulus(&parameters, window);
        let value = random_value(&modulus);
        let inputs = encode_value(&context, &parameters, &value, window);
        let actual =
            execute_constant_output("nested-rns-const-mul-runtime", &parameters, &circuit, inputs);
        for (q_index, q_modulus) in context.q_moduli.iter().take(2).copied().enumerate() {
            let q_modulus = BigUint::from(q_modulus);
            assert_eq!(
                &actual % &q_modulus,
                ((&value % &q_modulus) * BigUint::from(constants[q_index])) % &q_modulus
            );
        }

        let mut slot_circuit = PolyCircuit::<DCRTPoly>::new();
        let (slot_parameters, slot_context) = create_context(&mut slot_circuit, Some(1));
        let slot_window = CrtWindow::new(0, 1, slot_context.q_moduli_depth);
        let slot_input =
            NestedRnsPoly::input(slot_context.clone(), 3, slot_window, &mut slot_circuit);
        let transferred = slot_input
            .slot_transfer(&[(0, Some(vec![1])), (1, Some(vec![2])), (2, None)], &mut slot_circuit);
        assert_eq!(
            transferred.max_plaintexts,
            vec![BigUint::from(slot_context.q_moduli[0] - 1) * BigUint::from(2u8)]
        );
        let slot_output = transferred.reconstruct(&mut slot_circuit);
        slot_circuit.output([slot_output]);
        let values = [BigUint::from(2u8), BigUint::from(3u8), BigUint::from(5u8)];
        let slot_inputs = encode_nested_rns_poly::<DCRTPoly>(
            slot_context.p_moduli_bits,
            slot_context.max_unreduced_muls,
            &slot_parameters,
            &values,
            slot_window,
        )
        .into_iter()
        .map(|lanes| {
            diagonal_matrix(
                &slot_parameters,
                lanes
                    .into_iter()
                    .map(|value| DCRTPoly::from_biguint_to_constant(&slot_parameters, value)),
            )
        })
        .collect::<Vec<_>>();
        let actual = execute_circuit_with_shape(
            "nested-rns-slot-transfer-runtime",
            &slot_parameters,
            &slot_circuit,
            &slot_inputs,
            (3 * slot_window.depth, 3 * slot_window.depth),
        );
        let expected_values =
            [values[0].clone(), values[1].clone() * BigUint::from(2u8), values[2].clone()];
        let expected = diagonal_matrix(
            &slot_parameters,
            expected_values.into_iter().flat_map(|value| {
                std::iter::repeat_n(
                    DCRTPoly::from_biguint_to_constant(&slot_parameters, value),
                    slot_window.depth,
                )
            }),
        );
        assert_eq!(actual, vec![expected]);
    }

    #[test]
    fn input_reduction_and_q_level_metadata_are_consistent() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (parameters, context) = create_context(&mut circuit, Some(2));
        assert_eq!(context.q_moduli_depth, parameters.to_crt().2);
        assert_eq!(context.q_moduli.len(), parameters.to_crt().2);
        let window = CrtWindow::new(0, 2, context.q_moduli_depth);
        let input = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
        assert_eq!(
            input.max_plaintexts,
            context.q_moduli.iter().take(2).map(|q| BigUint::from(q - 1)).collect::<Vec<_>>()
        );
        let reduced = input.full_reduce(&mut circuit);
        assert_eq!(reduced.max_plaintexts, context.full_reduce_max_plaintexts[..2]);

        let mut offset_circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, offset_context) = create_context(&mut offset_circuit, None);
        let offset_input = NestedRnsPoly::input(
            offset_context.clone(),
            1,
            CrtWindow::new(1, offset_context.q_moduli_depth - 1, offset_context.q_moduli_depth),
            &mut offset_circuit,
        );
        assert_eq!(offset_input.max_plaintexts.len(), offset_context.q_moduli_depth - 1);
    }

    #[test]
    fn compact_encoding_matches_polynomial_encoding() {
        let parameters = test_parameters();
        let (_, _, depth) = parameters.to_crt();
        let window = CrtWindow::new(0, 2, depth);
        let value = BigUint::from(12345u64);
        let expected = encode_nested_rns_poly::<DCRTPoly>(
            test_p_moduli_bits(),
            DEFAULT_MAX_UNREDUCED_MULS,
            &parameters,
            std::slice::from_ref(&value),
            window,
        )
        .into_par_iter()
        .map(|lanes| {
            lanes
                .into_iter()
                .map(|value| {
                    DCRTPoly::from_biguint_to_constant(&parameters, value).to_compact_bytes()
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
        let actual = encode_nested_rns_poly_compact_bytes::<DCRTPoly>(
            test_p_moduli_bits(),
            DEFAULT_MAX_UNREDUCED_MULS,
            &parameters,
            std::slice::from_ref(&value),
            window,
        );
        assert_eq!(actual, expected);

        let offset = encode_nested_rns_poly::<DCRTPoly>(
            test_p_moduli_bits(),
            DEFAULT_MAX_UNREDUCED_MULS,
            &parameters,
            &[BigUint::from(1u64)],
            CrtWindow::new(1, depth - 1, depth),
        );
        assert!(offset.iter().all(|lanes| {
            lanes.len() == depth - 1 && lanes.iter().all(|lane| lane == &BigUint::from(1u64))
        }));
    }

    #[test]
    #[should_panic(expected = "mismatched CRT windows")]
    fn binary_operations_reject_mismatched_level_windows() {
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let (_, context) = create_context(&mut circuit, None);
        let left = NestedRnsPoly::input(context.clone(), 1, CrtWindow::new(0, 1, 3), &mut circuit);
        let right = NestedRnsPoly::input(context, 1, CrtWindow::new(0, 2, 3), &mut circuit);
        let _ = left.add(&right, &mut circuit);
    }

    #[test]
    fn sample_crt_primes_satisfies_the_configured_unreduced_multiplication_budget() {
        let q_max = 43u64;
        let p_moduli_bits = 7usize;
        let max_unreduced_muls = 4usize;
        let p_moduli =
            super::super::encoding::sample_crt_primes(p_moduli_bits, q_max, max_unreduced_muls);
        let product =
            p_moduli.par_iter().copied().map(BigUint::from).reduce(BigUint::one, |a, b| a * b);
        let sum = p_moduli.par_iter().copied().sum::<u64>();
        let bound =
            super::super::encoding::sample_crt_primes_mul_budget_bound(sum, p_moduli.len(), q_max);
        assert!(pow_biguint_usize(&bound, max_unreduced_muls) < product);

        let default_product = super::super::encoding::sample_crt_primes(
            p_moduli_bits,
            q_max,
            DEFAULT_MAX_UNREDUCED_MULS,
        )
        .par_iter()
        .copied()
        .map(BigUint::from)
        .reduce(BigUint::one, |a, b| a * b);
        assert!(default_product < product);
    }

    #[test]
    fn sequential_add_inserts_only_the_required_full_reduction() {
        let q_level = Some(1usize);
        let mut setup = PolyCircuit::<DCRTPoly>::new();
        let (_, context) = create_context(&mut setup, q_level);
        let reduced_bound = context.full_reduce_max_plaintexts[0].clone();
        let (operand_count, operand_bound) = (2usize..=8)
            .find_map(|count| {
                let count = BigUint::from(count);
                let bound = (&context.p_full + &count - BigUint::one()) / &count;
                let pre_last = &bound * (&count - BigUint::one());
                if pre_last < context.p_full &&
                    &bound * &count >= context.p_full &&
                    &reduced_bound + &bound < context.p_full
                {
                    Some((count.to_usize().expect("small operand count"), bound))
                } else {
                    None
                }
            })
            .expect("a short addition chain must trigger exactly one reduction");

        let build = |manual: bool| {
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let (_, context) = create_context(&mut circuit, q_level);
            let window = CrtWindow::new(
                0,
                q_level.expect("test uses one active level"),
                context.q_moduli_depth,
            );
            let inputs = (0..operand_count)
                .map(|_| {
                    let input = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
                    NestedRnsPoly::new(
                        input.ctx.clone(),
                        input.inner.clone(),
                        input.num_coefficient_slots,
                        input.window,
                        vec![operand_bound.clone()],
                    )
                })
                .collect::<Vec<_>>();
            let mut sum = inputs[0].clone();
            if manual {
                for input in inputs.iter().skip(1).take(operand_count - 2) {
                    sum = sum.add(input, &mut circuit);
                }
                sum = sum.full_reduce(&mut circuit);
                sum = sum.add(
                    &inputs.last().expect("last input").full_reduce(&mut circuit),
                    &mut circuit,
                );
            } else {
                for input in inputs.iter().skip(1) {
                    sum = sum.add(input, &mut circuit);
                }
            }
            let result = sum.clone();
            let output = result.reconstruct(&mut circuit);
            circuit.output([output]);
            (result, circuit)
        };
        let (automatic_result, automatic) = build(false);
        let (manual_result, manual) = build(true);
        assert_eq!(automatic_result.max_plaintexts, manual_result.max_plaintexts);
        assert_eq!(automatic_result.p_max_traces, manual_result.p_max_traces);
        assert_eq!(automatic.count_gates_by_type_vec(), manual.count_gates_by_type_vec());
        assert_eq!(automatic.non_free_depth_contributions(), manual.non_free_depth_contributions());
    }

    #[test]
    fn sequential_mul_reduces_only_the_operand_required_by_the_multiplication_budget() {
        let q_level = Some(1usize);
        let p_moduli_bits = 10usize;
        let max_unreduced_muls = 4usize;
        let mut setup = PolyCircuit::<DCRTPoly>::new();
        let (_, context) =
            create_context_with_config(&mut setup, q_level, p_moduli_bits, max_unreduced_muls);
        let operand_count = max_unreduced_muls + 1;
        let operand_bound = ceil_biguint_nth_root(&context.p_full, operand_count);
        assert!(pow_biguint_usize(&operand_bound, operand_count - 1) < context.p_full);
        assert!(pow_biguint_usize(&operand_bound, operand_count) >= context.p_full);

        let build = |manual: bool| {
            let mut circuit = PolyCircuit::<DCRTPoly>::new();
            let (_, context) = create_context_with_config(
                &mut circuit,
                q_level,
                p_moduli_bits,
                max_unreduced_muls,
            );
            let window = CrtWindow::new(
                0,
                q_level.expect("test uses one active level"),
                context.q_moduli_depth,
            );
            let inputs = (0..operand_count)
                .map(|_| {
                    let input = NestedRnsPoly::input(context.clone(), 1, window, &mut circuit);
                    NestedRnsPoly::new(
                        input.ctx.clone(),
                        input.inner.clone(),
                        input.num_coefficient_slots,
                        input.window,
                        vec![operand_bound.clone()],
                    )
                })
                .collect::<Vec<_>>();
            let mut product = inputs[0].clone();
            if manual {
                for input in inputs.iter().skip(1).take(operand_count - 2) {
                    product = product.mul(input, &mut circuit);
                }
                product = product.full_reduce(&mut circuit);
                product = product.mul(inputs.last().expect("last input"), &mut circuit);
            } else {
                for input in inputs.iter().skip(1) {
                    product = product.mul(input, &mut circuit);
                }
            }
            let result = product.clone();
            let output = result.reconstruct(&mut circuit);
            circuit.output([output]);
            (result, circuit)
        };
        let (automatic_result, automatic) = build(false);
        let (manual_result, manual) = build(true);
        assert_eq!(automatic_result.max_plaintexts, manual_result.max_plaintexts);
        assert_eq!(automatic_result.p_max_traces, manual_result.p_max_traces);
        assert_eq!(automatic.count_gates_by_type_vec(), manual.count_gates_by_type_vec());
        assert_eq!(automatic.non_free_depth_contributions(), manual.non_free_depth_contributions());
    }
}
