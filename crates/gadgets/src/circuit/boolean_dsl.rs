//! Parameterized layered Boolean circuits represented by flattened DSL families.

use mxx_dsl::{
    BodyTraceRemapper, Bool, DslContext, DslError, EvaluateIntConstructionTrace, Family,
    GatherConstructionTrace, Int, LoopConstructionTrace, LoopIndex, Mat, Parallel,
    RemapConstructionTrace, Sequential, parallel_zip_bundle, parallel_zip_bundle_result,
    parallel_zip_bundle_result_traced,
};
use mxx_ir_core::{IntExpr, ValueHandle};

pub const BOOLEAN_INSTANCE_INPUT: &str = "boolean-instance";
pub const BOOLEAN_WITNESS_INPUT: &str = "boolean-witness";

fn int_expr_add(left: IntExpr, right: IntExpr) -> IntExpr {
    IntExpr::Add(Box::new(left), Box::new(right))
}

fn int_expr_mul(left: IntExpr, right: IntExpr) -> IntExpr {
    IntExpr::Mul(Box::new(left), Box::new(right))
}

fn bool_and(left: Bool, right: Bool) -> Bool {
    left.to_int().mul(right.to_int()).equal(Int::constant(1))
}

fn bool_or(left: Bool, right: Bool) -> Bool {
    Int::constant(1).less_equal(left.to_int().add(right.to_int()))
}

fn bool_value(value: bool) -> Bool {
    Bool::constant(value).to_int().equal(Int::constant(1))
}

fn bool_all(values: impl IntoIterator<Item = Bool>) -> Bool {
    values.into_iter().fold(bool_value(true), bool_and)
}

fn bool_exactly_one(values: impl IntoIterator<Item = Bool>) -> Bool {
    values.into_iter().map(Bool::to_int).fold(Int::constant(0), Int::add).equal(Int::constant(1))
}

/// Symbolic dimensions of the rectangular Boolean-circuit representation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BooleanCircuitFamilyParams {
    pub instance_width: IntExpr,
    pub witness_width: IntExpr,
    pub depth: IntExpr,
    pub max_layer_width: IntExpr,
}

impl BooleanCircuitFamilyParams {
    pub const INSTANCE_WIDTH_PARAMETER: &str = "instance_width";
    pub const WITNESS_WIDTH_PARAMETER: &str = "witness_width";
    pub const DEPTH_PARAMETER: &str = "depth";
    pub const MAX_LAYER_WIDTH_PARAMETER: &str = "max_layer_width";

    pub fn new(
        instance_width: impl Into<IntExpr>,
        witness_width: impl Into<IntExpr>,
        depth: impl Into<IntExpr>,
        max_layer_width: impl Into<IntExpr>,
    ) -> Self {
        Self {
            instance_width: instance_width.into(),
            witness_width: witness_width.into(),
            depth: depth.into(),
            max_layer_width: max_layer_width.into(),
        }
    }

    /// Declares the canonical symbolic circuit parameters on a graph context.
    pub fn declare(context: DslContext) -> (DslContext, Self) {
        let context = context
            .int_parameter(Self::INSTANCE_WIDTH_PARAMETER)
            .int_parameter(Self::WITNESS_WIDTH_PARAMETER)
            .int_parameter(Self::DEPTH_PARAMETER)
            .int_parameter(Self::MAX_LAYER_WIDTH_PARAMETER);
        let params = Self::new(
            IntExpr::Var(Self::INSTANCE_WIDTH_PARAMETER.to_owned()),
            IntExpr::Var(Self::WITNESS_WIDTH_PARAMETER.to_owned()),
            IntExpr::Var(Self::DEPTH_PARAMETER.to_owned()),
            IntExpr::Var(Self::MAX_LAYER_WIDTH_PARAMETER.to_owned()),
        );
        (context, params)
    }

    pub fn input_width(&self) -> IntExpr {
        int_expr_add(self.instance_width.clone(), self.witness_width.clone())
    }

    pub fn flattened_gate_count(&self) -> IntExpr {
        int_expr_mul(self.depth.clone(), self.max_layer_width.clone())
    }

    fn flattened_index(&self, layer: &LoopIndex, slot: &LoopIndex) -> IntExpr {
        int_expr_add(
            int_expr_mul(layer.expression(), self.max_layer_width.clone()),
            slot.expression(),
        )
    }
}

/// Public runtime circuit data in a rectangular, flattened representation.
///
/// `gate_kinds`, `left_sources`, and `right_sources` use row-major
/// `layer * max_layer_width + slot` indexing. Entries at or after a layer's active count must use
/// the canonical all-zero padding record. `output_sources` has one element so the output index is
/// still represented as ordinary public runtime family data.
#[derive(Clone)]
pub struct BooleanCircuitFamilyInputs {
    pub active_gate_counts: Family<Int>,
    pub gate_kinds: Family<Int>,
    pub left_sources: Family<Int>,
    pub right_sources: Family<Int>,
    pub output_sources: Family<Int>,
}

impl BooleanCircuitFamilyInputs {
    pub fn protocol_inputs(context: &DslContext, params: &BooleanCircuitFamilyParams) -> Self {
        let flattened_count = params.flattened_gate_count();
        Self {
            active_gate_counts: context
                .int_family_input("circuit-active-gate-count", params.depth.clone()),
            gate_kinds: context.int_family_input("circuit-gate-kind", flattened_count.clone()),
            left_sources: context.int_family_input("circuit-left-source", flattened_count.clone()),
            right_sources: context.int_family_input("circuit-right-source", flattened_count),
            output_sources: context.int_family_input("circuit-output-source", 1),
        }
    }

    pub fn output_source(&self) -> Int {
        self.output_sources.get(Int::constant(0))
    }
}

#[derive(Clone)]
pub struct GateSlot {
    pub layer: LoopIndex,
    pub index: LoopIndex,
}

pub trait BooleanLayerGate<T> {
    type ConstructionTrace: RemapConstructionTrace;

    /// Builds the six candidates in opcode order: false, true, copy, not, and, xor.
    ///
    /// A single call lets handlers share common executable subexpressions across candidates.
    fn candidates(
        &self,
        slot: GateSlot,
        left: T,
        right: T,
    ) -> Result<([T; 6], Self::ConstructionTrace), DslError>;
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct SelectConstructionTrace<const N: usize> {
    pub selector: ValueHandle,
    pub branches: [ValueHandle; N],
    pub output: ValueHandle,
}

impl<const N: usize> RemapConstructionTrace for SelectConstructionTrace<N> {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            selector: self.selector.remap_current_body(map)?,
            branches: self.branches.remap_current_body(map)?,
            output: self.output.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct MatrixBooleanGateSlotConstructionTrace<G> {
    pub opcode: ValueHandle,
    pub left: ValueHandle,
    pub right: ValueHandle,
    pub gate: G,
    pub candidate_select: SelectConstructionTrace<6>,
    pub active_gate_count: ValueHandle,
    pub active_select: SelectConstructionTrace<2>,
}

impl<G: RemapConstructionTrace> RemapConstructionTrace
    for MatrixBooleanGateSlotConstructionTrace<G>
{
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            opcode: self.opcode.remap_current_body(map)?,
            left: self.left.remap_current_body(map)?,
            right: self.right.remap_current_body(map)?,
            gate: self.gate.remap_current_body(map)?,
            candidate_select: self.candidate_select.remap_current_body(map)?,
            active_gate_count: self.active_gate_count.remap_current_body(map)?,
            active_select: self.active_select.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct MatrixBooleanLayerBodyConstructionTrace<G> {
    pub body_state: ValueHandle,
    pub body_active_gate_counts: ValueHandle,
    pub body_gate_kinds: ValueHandle,
    pub body_left_sources: ValueHandle,
    pub body_right_sources: ValueHandle,
    pub active_gate_count: GatherConstructionTrace,
    pub metadata: LayerMetadataConstructionTrace,
    pub left_values: LoopConstructionTrace<GatherConstructionTrace>,
    pub right_values: LoopConstructionTrace<GatherConstructionTrace>,
    pub gate_slots: LoopConstructionTrace<MatrixBooleanGateSlotConstructionTrace<G>>,
    pub body_output: ValueHandle,
}

impl<G: RemapConstructionTrace> RemapConstructionTrace
    for MatrixBooleanLayerBodyConstructionTrace<G>
{
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            body_state: self.body_state.remap_current_body(map)?,
            body_active_gate_counts: self.body_active_gate_counts.remap_current_body(map)?,
            body_gate_kinds: self.body_gate_kinds.remap_current_body(map)?,
            body_left_sources: self.body_left_sources.remap_current_body(map)?,
            body_right_sources: self.body_right_sources.remap_current_body(map)?,
            active_gate_count: self.active_gate_count.remap_current_body(map)?,
            metadata: self.metadata.remap_current_body(map)?,
            left_values: self.left_values.remap_current_body(map)?,
            right_values: self.right_values.remap_current_body(map)?,
            gate_slots: self.gate_slots.remap_current_body(map)?,
            body_output: self.body_output.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct LayerMetadataConstructionTrace {
    pub flattened_indices: LoopConstructionTrace<EvaluateIntConstructionTrace>,
    pub gate_kinds: LoopConstructionTrace<GatherConstructionTrace>,
    pub left_sources: LoopConstructionTrace<GatherConstructionTrace>,
    pub right_sources: LoopConstructionTrace<GatherConstructionTrace>,
}

impl RemapConstructionTrace for LayerMetadataConstructionTrace {
    fn remap_current_body(self, map: &BodyTraceRemapper<'_>) -> Result<Self, DslError> {
        Ok(Self {
            flattened_indices: self.flattened_indices.remap_current_body(map)?,
            gate_kinds: self.gate_kinds.remap_current_body(map)?,
            left_sources: self.left_sources.remap_current_body(map)?,
            right_sources: self.right_sources.remap_current_body(map)?,
        })
    }
}

#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct MatrixBooleanLayerConstructionTrace<G> {
    pub initial_state: ValueHandle,
    pub active_gate_counts: ValueHandle,
    pub gate_kinds: ValueHandle,
    pub left_sources: ValueHandle,
    pub right_sources: ValueHandle,
    pub layer_scan: LoopConstructionTrace<MatrixBooleanLayerBodyConstructionTrace<G>>,
}

fn layer_metadata(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    layer: &LoopIndex,
    gate_kinds: Family<Int>,
    left_sources: Family<Int>,
    right_sources: Family<Int>,
) -> Result<
    (Family<Int>, Family<Int>, Family<Int>, Family<Int>, LayerMetadataConstructionTrace),
    DslError,
> {
    let (flattened_indices, flattened_indices_trace) =
        Parallel::range(params.max_layer_width.clone()).map_values_traced(|slot| {
            context.evaluate_int_traced(params.flattened_index(layer, &slot))
        })?;
    let (kinds, gate_kinds_trace) = gate_kinds.parallel_gather_traced(flattened_indices.clone())?;
    let (left_indices, left_sources_trace) =
        left_sources.parallel_gather_traced(flattened_indices.clone())?;
    let (right_indices, right_sources_trace) =
        right_sources.parallel_gather_traced(flattened_indices.clone())?;
    Ok((
        flattened_indices,
        kinds,
        left_indices,
        right_indices,
        LayerMetadataConstructionTrace {
            flattened_indices: flattened_indices_trace,
            gate_kinds: gate_kinds_trace,
            left_sources: left_sources_trace,
            right_sources: right_sources_trace,
        },
    ))
}

pub fn evaluate_boolean_matrix_family<H>(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: Family<Mat>,
    handler: H,
) -> Result<(Family<Mat>, MatrixBooleanLayerConstructionTrace<H::ConstructionTrace>), DslError>
where
    H: BooleanLayerGate<Mat> + Clone,
{
    let initial_state = preceding.value_handle().clone();
    let active_gate_counts = circuit.active_gate_counts.value_handle().clone();
    let gate_kinds = circuit.gate_kinds.value_handle().clone();
    let left_sources = circuit.left_sources.value_handle().clone();
    let right_sources = circuit.right_sources.value_handle().clone();
    let invariants = (
        circuit.active_gate_counts,
        (circuit.gate_kinds, (circuit.left_sources, circuit.right_sources)),
    );
    let (final_state, layer_scan) = Sequential::range(params.depth.clone()).scan_traced(
        preceding,
        invariants,
        |layer, preceding, (active_gate_counts, (gate_kinds, (left_sources, right_sources)))| {
            let active_count_index = layer.as_int();
            let active_count_index_handle = active_count_index.value_handle().clone();
            let active_gate_counts_handle = active_gate_counts.value_handle().clone();
            let active_count = active_gate_counts.get(active_count_index);
            let body_state = preceding.value_handle().clone();
            let body_active_gate_counts = active_gate_counts.value_handle().clone();
            let body_gate_kinds = gate_kinds.value_handle().clone();
            let body_left_sources = left_sources.value_handle().clone();
            let body_right_sources = right_sources.value_handle().clone();
            let active_gate_count = GatherConstructionTrace {
                index: active_count_index_handle,
                sources: vec![active_gate_counts_handle],
                outputs: vec![active_count.value_handle().clone()],
            };
            let (_, kinds, left_indices, right_indices, metadata) =
                layer_metadata(context, params, &layer, gate_kinds, left_sources, right_sources)?;
            let (left_values, left_values_trace) =
                preceding.clone().parallel_gather_traced(left_indices)?;
            let (right_values, right_values_trace) =
                preceding.parallel_gather_traced(right_indices)?;
            let layer_handler = handler.clone();
            let (output, gate_slots) = parallel_zip_bundle_result_traced(
                (kinds, left_values, right_values),
                move |index, (kind, left, right)| {
                    let slot = GateSlot { layer: layer.clone(), index: index.clone() };
                    let opcode = kind.value_handle().clone();
                    let left_handle = left.value_handle().clone();
                    let right_handle = right.value_handle().clone();
                    let (candidates, gate) = layer_handler.candidates(slot, left, right)?;
                    let constant_false = candidates[0].clone();
                    let candidate_handles =
                        candidates.each_ref().map(|value| value.value_handle().clone());
                    let selected = kind.select(candidates.into_iter().collect())?;
                    let active =
                        index.as_int().less_equal(active_count.clone().sub(Int::constant(1)));
                    let active = active.to_int();
                    let active_selector = active.value_handle().clone();
                    let active_branches =
                        [constant_false.value_handle().clone(), selected.value_handle().clone()];
                    let output = active.select(vec![constant_false, selected])?;
                    Ok((
                        output.clone(),
                        MatrixBooleanGateSlotConstructionTrace {
                            opcode: opcode.clone(),
                            left: left_handle,
                            right: right_handle,
                            gate,
                            candidate_select: SelectConstructionTrace {
                                selector: opcode,
                                branches: candidate_handles,
                                output: active_branches[1].clone(),
                            },
                            active_gate_count: active_count.value_handle().clone(),
                            active_select: SelectConstructionTrace {
                                selector: active_selector,
                                branches: active_branches,
                                output: output.value_handle().clone(),
                            },
                        },
                    ))
                },
            )?;
            let body_output = output.value_handle().clone();
            Ok((
                output,
                MatrixBooleanLayerBodyConstructionTrace {
                    body_state,
                    body_active_gate_counts,
                    body_gate_kinds,
                    body_left_sources,
                    body_right_sources,
                    active_gate_count,
                    metadata,
                    left_values: left_values_trace,
                    right_values: right_values_trace,
                    gate_slots,
                    body_output,
                },
            ))
        },
    )?;
    let trace = MatrixBooleanLayerConstructionTrace {
        initial_state,
        active_gate_counts,
        gate_kinds,
        left_sources,
        right_sources,
        layer_scan,
    };
    Ok((final_state, trace))
}

pub fn evaluate_boolean_family(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: Family<Bool>,
) -> Result<Family<Bool>, DslError> {
    let invariants = (
        circuit.active_gate_counts,
        (circuit.gate_kinds, (circuit.left_sources, circuit.right_sources)),
    );
    Sequential::range(params.depth.clone()).scan(
        preceding,
        invariants,
        |layer, preceding, (active_gate_counts, (gate_kinds, (left_sources, right_sources)))| {
            let active_count = active_gate_counts.get(layer.as_int());
            let (_, kinds, left_indices, right_indices, _) =
                layer_metadata(context, params, &layer, gate_kinds, left_sources, right_sources)?;
            let left_values = preceding.clone().parallel_gather(left_indices)?;
            let right_values = preceding.parallel_gather(right_indices)?;
            parallel_zip_bundle_result(
                (kinds, left_values, right_values),
                move |index, (kind, left, right)| {
                    let not = left.clone().to_int().equal(Int::constant(0));
                    let and =
                        left.clone().to_int().mul(right.clone().to_int()).equal(Int::constant(1));
                    let xor =
                        left.clone().to_int().add(right.clone().to_int()).equal(Int::constant(1));
                    let selected = kind.select_bool(vec![
                        bool_value(false),
                        bool_value(true),
                        left,
                        not,
                        and,
                        xor,
                    ])?;
                    index
                        .as_int()
                        .less_equal(active_count.clone().sub(Int::constant(1)))
                        .to_int()
                        .select_bool(vec![bool_value(false), selected])
                },
            )
        },
    )
}

pub fn select_boolean_output(
    circuit: &BooleanCircuitFamilyInputs,
    final_layer: &Family<Bool>,
) -> Bool {
    final_layer.get(circuit.output_source())
}

pub fn select_boolean_matrix_output(
    circuit: &BooleanCircuitFamilyInputs,
    final_layer: &Family<Mat>,
) -> Mat {
    final_layer.get(circuit.output_source())
}

fn gate_record_valid(kind: Int, left: Int, right: Int, previous_width: Int) -> Bool {
    let zero = Int::constant(0);
    let left_in_range = bool_and(
        zero.clone().less_equal(left.clone()),
        left.clone().less_equal(previous_width.clone().sub(Int::constant(1))),
    );
    let right_in_range = bool_and(
        zero.clone().less_equal(right.clone()),
        right.clone().less_equal(previous_width.sub(Int::constant(1))),
    );
    let left_zero = left.equal(zero.clone());
    let right_zero = right.equal(zero);
    bool_exactly_one([
        bool_all([kind.clone().equal(Int::constant(0)), left_zero.clone(), right_zero.clone()]),
        bool_all([kind.clone().equal(Int::constant(1)), left_zero, right_zero.clone()]),
        bool_all([kind.clone().equal(Int::constant(2)), left_in_range.clone(), right_zero.clone()]),
        bool_all([kind.clone().equal(Int::constant(3)), left_in_range.clone(), right_zero]),
        bool_all([
            kind.clone().equal(Int::constant(4)),
            left_in_range.clone(),
            right_in_range.clone(),
        ]),
        bool_all([kind.equal(Int::constant(5)), left_in_range, right_in_range]),
    ])
}

fn reduce_bool_family(values: Family<Bool>, count: IntExpr) -> Result<Bool, DslError> {
    Sequential::range(count).scan(bool_value(true), values, |index, result, values| {
        Ok(bool_and(result, values.get(index.as_int())))
    })
}

/// Builds the authoritative sampler-free well-formedness predicate.
///
/// The initial input family and every carried layer have `max_layer_width` entries. Therefore the
/// predicate requires `instance_width + witness_width <= max_layer_width`; unused input and gate
/// slots use canonical zero padding.
pub fn boolean_circuit_validity_predicate(
    context: DslContext,
) -> Result<mxx_dsl::PurePredicateSpec, DslError> {
    let (context, params) = BooleanCircuitFamilyParams::declare(context);
    let circuit = BooleanCircuitFamilyInputs::protocol_inputs(&context, &params);
    let instance_width = context.evaluate_int(params.instance_width.clone());
    let witness_width = context.evaluate_int(params.witness_width.clone());
    let input_width = context.evaluate_int(params.input_width());
    let depth = context.evaluate_int(params.depth.clone());
    let max_width = context.evaluate_int(params.max_layer_width.clone());
    let initial_validity =
        Parallel::range(params.max_layer_width.clone()).map_values(|_| bool_value(true))?;
    let invariants = (
        circuit.active_gate_counts.clone(),
        (circuit.gate_kinds.clone(), (circuit.left_sources.clone(), circuit.right_sources.clone())),
    );
    let (slot_validity, final_active_count) = Sequential::range(params.depth.clone()).scan(
        (initial_validity, input_width.clone()),
        invariants,
        |layer,
         (previous_validity, previous_width),
         (active_gate_counts, (gate_kinds, (left_sources, right_sources)))| {
            let active_count = active_gate_counts.get(layer.as_int());
            let (_, kinds, left, right, _) =
                layer_metadata(&context, &params, &layer, gate_kinds, left_sources, right_sources)?;
            let records = parallel_zip_bundle((kinds, left, right), {
                let active_count = active_count.clone();
                move |slot, (kind, left, right)| {
                    let active =
                        slot.as_int().less_equal(active_count.clone().sub(Int::constant(1)));
                    active
                        .to_int()
                        .select_bool(vec![
                            bool_all([
                                kind.clone().equal(Int::constant(0)),
                                left.clone().equal(Int::constant(0)),
                                right.clone().equal(Int::constant(0)),
                            ]),
                            gate_record_valid(kind, left, right, previous_width.clone()),
                        ])
                        .expect("two boolean branches")
                }
            })?;
            let active_count_valid = bool_and(
                Int::constant(1).less_equal(active_count.clone()),
                active_count.clone().less_equal(max_width.clone()),
            );
            let validity = parallel_zip_bundle(
                (previous_validity, records),
                move |_, (previous, current)| {
                    bool_all([previous, current, active_count_valid.clone()])
                },
            )?;
            Ok((validity, active_count))
        },
    )?;
    let records_valid = reduce_bool_family(slot_validity, params.max_layer_width.clone())?;
    let output_source = circuit.output_source();
    let output_valid = bool_and(
        Int::constant(0).less_equal(output_source.clone()),
        output_source.less_equal(final_active_count.sub(Int::constant(1))),
    );
    let params_valid = bool_all([
        Int::constant(0).less_equal(instance_width),
        Int::constant(0).less_equal(witness_width),
        Int::constant(1).less_equal(input_width.clone()),
        Int::constant(1).less_equal(depth),
        Int::constant(1).less_equal(max_width.clone()),
        input_width.less_equal(max_width),
    ]);
    mxx_dsl::PurePredicateSpec::new(
        context
            .bool_output("valid", bool_all([params_valid, records_valid, output_valid]))?
            .build()?,
    )
}

/// Builds a sampler-free ideal evaluator for the symbolic circuit family.
///
/// Instance and witness inputs are separate integer families of length `max_layer_width`, so a
/// protocol can expose the witness only to its final stage. The first `instance_width` and
/// `witness_width` entries, respectively, must be zero or one; remaining entries must be canonical
/// zero padding. The predicate is true exactly when both inputs are canonical and the selected
/// circuit output is true. Circuit-data validity is checked by the separate validity predicate.
pub fn boolean_circuit_satisfaction_predicate(
    context: DslContext,
) -> Result<mxx_dsl::PurePredicateSpec, DslError> {
    let (context, params) = BooleanCircuitFamilyParams::declare(context);
    let circuit = BooleanCircuitFamilyInputs::protocol_inputs(&context, &params);
    let instance_width = context.evaluate_int(params.instance_width.clone());
    let witness_width = context.evaluate_int(params.witness_width.clone());
    let encoded_instances =
        context.int_family_input(BOOLEAN_INSTANCE_INPUT, params.max_layer_width.clone());
    let encoded_witnesses =
        context.int_family_input(BOOLEAN_WITNESS_INPUT, params.max_layer_width.clone());
    let canonical_family = |values: Family<Int>, width: Int| {
        values.parallel_map({
            move |slot, value| {
                let active = slot.as_int().less_equal(width.clone().sub(Int::constant(1)));
                let binary = bool_or(
                    value.clone().equal(Int::constant(0)),
                    value.clone().equal(Int::constant(1)),
                );
                active
                    .to_int()
                    .select_bool(vec![value.equal(Int::constant(0)), binary])
                    .expect("two boolean branches")
                    .to_int()
            }
        })
    };
    let instance_validity = canonical_family(encoded_instances.clone(), instance_width.clone())?;
    let witness_validity = canonical_family(encoded_witnesses.clone(), witness_width.clone())?;
    let input_validity =
        parallel_zip_bundle((instance_validity, witness_validity), |_, (instance, witness)| {
            instance.mul(witness).equal(Int::constant(1))
        })?;
    let input_width = context.evaluate_int(params.input_width());
    let witness_indices = encoded_instances.clone().parallel_map({
        let instance_width = instance_width.clone();
        let input_width = input_width.clone();
        move |slot, value| {
            let index = slot.as_int();
            let witness_active =
                index.clone().less_equal(input_width.clone().sub(Int::constant(1))).to_int().sub(
                    index.clone().less_equal(instance_width.clone().sub(Int::constant(1))).to_int(),
                );
            witness_active.mul(index.sub(instance_width.clone())).add(value.mul(Int::constant(0)))
        }
    })?;
    let shifted_witnesses = encoded_witnesses.parallel_gather(witness_indices)?;
    let inputs = parallel_zip_bundle_result(
        (encoded_instances, shifted_witnesses),
        move |slot, (instance, witness)| {
            let index = slot.as_int();
            let instance_active =
                index.clone().less_equal(instance_width.clone().sub(Int::constant(1)));
            let witness_active = index.less_equal(input_width.clone().sub(Int::constant(1)));
            let value = instance_active.to_int().select_int(vec![
                witness_active.to_int().select_int(vec![Int::constant(0), witness])?,
                instance,
            ])?;
            Ok::<_, DslError>(value.equal(Int::constant(1)))
        },
    )?;
    let final_layer = evaluate_boolean_family(&context, &params, circuit.clone(), inputs)?;
    let output = select_boolean_output(&circuit, &final_layer);
    let inputs_valid = reduce_bool_family(input_validity, params.max_layer_width)?;
    mxx_dsl::PurePredicateSpec::new(
        context.bool_output("satisfied", bool_and(inputs_valid, output))?.build()?,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::{
        BooleanCircuitData, BooleanCircuitShape, BooleanGateData, BooleanGateKind,
    };
    use mxx_ir_core::{ParamEnv, node::NodeKind};
    use mxx_primitives::poly::dcrt::params::DCRTPolyParams;
    use mxx_runtime::{
        RuntimeValue,
        artifact::MemoryArtifactStore,
        backend::poly::{CpuDcrtBackend, cpu_backend},
        execute,
        transcript::SamplingMode,
    };
    use std::collections::BTreeMap;

    fn bindings() -> ParamEnv {
        bindings_for(1, 1, 2, 3)
    }

    fn bindings_for(
        instance_width: i32,
        witness_width: i32,
        depth: i32,
        max_layer_width: i32,
    ) -> ParamEnv {
        ParamEnv {
            integers: BTreeMap::from([
                (
                    BooleanCircuitFamilyParams::INSTANCE_WIDTH_PARAMETER.to_owned(),
                    instance_width.into(),
                ),
                (
                    BooleanCircuitFamilyParams::WITNESS_WIDTH_PARAMETER.to_owned(),
                    witness_width.into(),
                ),
                (BooleanCircuitFamilyParams::DEPTH_PARAMETER.to_owned(), depth.into()),
                (
                    BooleanCircuitFamilyParams::MAX_LAYER_WIDTH_PARAMETER.to_owned(),
                    max_layer_width.into(),
                ),
            ]),
            ..ParamEnv::default()
        }
    }

    fn runtime_family(values: &[i32]) -> RuntimeValue<CpuDcrtBackend> {
        RuntimeValue::IndexedFamily(
            values.iter().map(|value| RuntimeValue::Int((*value).into())).collect(),
        )
    }

    fn execute_predicate(
        predicate: &mxx_dsl::PurePredicateSpec,
        bindings: &ParamEnv,
        inputs: BTreeMap<String, RuntimeValue<CpuDcrtBackend>>,
        output: &str,
    ) -> bool {
        let validated = mxx_ir_core::validate(&predicate.graph, bindings).unwrap();
        let result = execute(
            &validated,
            &mut cpu_backend([DCRTPolyParams::new(8, 1, 20, 4)]),
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        matches!(result.outputs[output], RuntimeValue::Bool(true))
    }

    #[test]
    fn symbolic_boolean_evaluation_uses_scan_and_dynamic_flattened_access() {
        let (context, params) =
            BooleanCircuitFamilyParams::declare(DslContext::new("symbolic-boolean"));
        let circuit = BooleanCircuitFamilyInputs::protocol_inputs(&context, &params);
        let encoded_inputs = context.int_family_input("inputs", params.max_layer_width.clone());
        let inputs =
            parallel_zip_bundle((encoded_inputs.clone(), encoded_inputs), |_, (value, _)| {
                value.equal(Int::constant(1))
            })
            .unwrap();
        let output = evaluate_boolean_family(&context, &params, circuit.clone(), inputs).unwrap();
        let selected = select_boolean_output(&circuit, &output);
        let graph = context.bool_output("result", selected).unwrap().build().unwrap();
        graph.validate(&bindings()).unwrap();

        assert_eq!(
            graph
                .graph
                .root_scope()
                .nodes()
                .iter()
                .filter(|node| matches!(node.kind(), NodeKind::SequentialLoop(_)))
                .count(),
            1
        );
        assert!(
            graph
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| { matches!(node.kind(), NodeKind::FamilyGetDynamic) })
        );
        assert!(
            graph
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| { matches!(node.kind(), NodeKind::ParallelLoop(_)) })
        );
    }

    #[test]
    fn symbolic_boolean_predicates_validate_for_multiple_shapes_without_rebuilding() {
        let validity =
            boolean_circuit_validity_predicate(DslContext::new("symbolic-validity")).unwrap();
        let satisfaction =
            boolean_circuit_satisfaction_predicate(DslContext::new("symbolic-satisfaction"))
                .unwrap();
        mxx_ir_core::validate(&validity.graph, &bindings()).unwrap();
        mxx_ir_core::validate(&satisfaction.graph, &bindings()).unwrap();

        let mut second = bindings();
        second.integers.insert(BooleanCircuitFamilyParams::DEPTH_PARAMETER.to_owned(), 4.into());
        second
            .integers
            .insert(BooleanCircuitFamilyParams::MAX_LAYER_WIDTH_PARAMETER.to_owned(), 5.into());
        mxx_ir_core::validate(&validity.graph, &second).unwrap();
        mxx_ir_core::validate(&satisfaction.graph, &second).unwrap();
    }

    #[test]
    fn symbolic_boolean_predicates_execute_the_dynamic_circuit() {
        let circuit_inputs = BTreeMap::from([
            ("circuit-active-gate-count".to_owned(), runtime_family(&[2, 1])),
            ("circuit-gate-kind".to_owned(), runtime_family(&[4, 5, 0, 2, 0, 0])),
            ("circuit-left-source".to_owned(), runtime_family(&[0, 0, 0, 0, 0, 0])),
            ("circuit-right-source".to_owned(), runtime_family(&[1, 1, 0, 0, 0, 0])),
            ("circuit-output-source".to_owned(), runtime_family(&[0])),
        ]);

        let validity =
            boolean_circuit_validity_predicate(DslContext::new("runtime-symbolic-validity"))
                .unwrap();
        assert!(execute_predicate(&validity, &bindings(), circuit_inputs.clone(), "valid",));

        for (instance_width, witness_width) in [(-1, 2), (2, -1)] {
            let mut invalid_bindings = bindings();
            invalid_bindings.integers.insert(
                BooleanCircuitFamilyParams::INSTANCE_WIDTH_PARAMETER.to_owned(),
                instance_width.into(),
            );
            invalid_bindings.integers.insert(
                BooleanCircuitFamilyParams::WITNESS_WIDTH_PARAMETER.to_owned(),
                witness_width.into(),
            );
            assert!(!execute_predicate(
                &validity,
                &invalid_bindings,
                circuit_inputs.clone(),
                "valid",
            ));
        }

        let satisfaction = boolean_circuit_satisfaction_predicate(DslContext::new(
            "runtime-symbolic-satisfaction",
        ))
        .unwrap();
        let mut satisfaction_inputs = circuit_inputs;
        satisfaction_inputs.insert(BOOLEAN_INSTANCE_INPUT.to_owned(), runtime_family(&[1, 0, 0]));
        satisfaction_inputs.insert(BOOLEAN_WITNESS_INPUT.to_owned(), runtime_family(&[1, 0, 0]));
        assert!(execute_predicate(&satisfaction, &bindings(), satisfaction_inputs, "satisfied",));
    }

    #[test]
    fn one_predicate_graph_accepts_distinct_active_widths_and_outputs() {
        let validity =
            boolean_circuit_validity_predicate(DslContext::new("dynamic-width-validity")).unwrap();
        let bindings = bindings_for(1, 1, 2, 3);
        let inputs = |active: &[i32], kinds: &[i32], output: i32| {
            BTreeMap::from([
                ("circuit-active-gate-count".to_owned(), runtime_family(active)),
                ("circuit-gate-kind".to_owned(), runtime_family(kinds)),
                ("circuit-left-source".to_owned(), runtime_family(&[0; 6])),
                ("circuit-right-source".to_owned(), runtime_family(&[0; 6])),
                ("circuit-output-source".to_owned(), runtime_family(&[output])),
            ])
        };
        assert!(execute_predicate(
            &validity,
            &bindings,
            inputs(&[2, 1], &[2, 3, 0, 2, 0, 0], 0),
            "valid",
        ));
        assert!(execute_predicate(
            &validity,
            &bindings,
            inputs(&[1, 3], &[2, 0, 0, 2, 3, 5], 2),
            "valid",
        ));
    }

    #[test]
    fn symbolic_evaluator_matches_all_gate_kinds_and_rejects_malformed_data() {
        let params = bindings_for(2, 1, 2, 6);
        let shape = BooleanCircuitShape {
            instance_width: 2,
            witness_width: 1,
            depth: 2,
            max_layer_width: 6,
        };
        let circuit = BooleanCircuitData {
            layers: vec![
                vec![
                    BooleanGateData { kind: BooleanGateKind::ConstantFalse, left: 0, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::ConstantTrue, left: 0, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::Copy, left: 0, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::Not, left: 1, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::And, left: 0, right: 1 },
                    BooleanGateData { kind: BooleanGateKind::Xor, left: 0, right: 1 },
                ],
                vec![
                    BooleanGateData { kind: BooleanGateKind::Copy, left: 4, right: 0 },
                    BooleanGateData { kind: BooleanGateKind::Xor, left: 2, right: 3 },
                ],
            ],
            output_source: 1,
        };
        let expected = circuit.evaluate(&shape, &[true, false], &[true]).unwrap();

        let validity =
            boolean_circuit_validity_predicate(DslContext::new("all-gates-validity")).unwrap();
        let satisfaction =
            boolean_circuit_satisfaction_predicate(DslContext::new("all-gates-satisfaction"))
                .unwrap();
        let kinds = vec![0, 1, 2, 3, 4, 5, 2, 5, 0, 0, 0, 0];
        let left = vec![0, 0, 0, 1, 0, 0, 4, 2, 0, 0, 0, 0];
        let right = vec![0, 0, 0, 0, 1, 1, 0, 3, 0, 0, 0, 0];
        let make_inputs = |kinds: &[i32], left: &[i32], right: &[i32], output: i32| {
            BTreeMap::from([
                ("circuit-active-gate-count".to_owned(), runtime_family(&[6, 2])),
                ("circuit-gate-kind".to_owned(), runtime_family(kinds)),
                ("circuit-left-source".to_owned(), runtime_family(left)),
                ("circuit-right-source".to_owned(), runtime_family(right)),
                ("circuit-output-source".to_owned(), runtime_family(&[output])),
            ])
        };
        let valid_inputs = make_inputs(&kinds, &left, &right, 1);
        assert!(execute_predicate(&validity, &params, valid_inputs.clone(), "valid"));
        let mut ideal_inputs = valid_inputs;
        ideal_inputs.insert(BOOLEAN_INSTANCE_INPUT.to_owned(), runtime_family(&[1, 0, 0, 0, 0, 0]));
        ideal_inputs.insert(BOOLEAN_WITNESS_INPUT.to_owned(), runtime_family(&[1, 0, 0, 0, 0, 0]));
        assert_eq!(execute_predicate(&satisfaction, &params, ideal_inputs, "satisfied"), expected,);

        let mut noncanonical_padding = kinds.clone();
        noncanonical_padding[8] = 1;
        assert!(!execute_predicate(
            &validity,
            &params,
            make_inputs(&noncanonical_padding, &left, &right, 1),
            "valid",
        ));
        let mut out_of_range = left.clone();
        out_of_range[2] = 3;
        assert!(!execute_predicate(
            &validity,
            &params,
            make_inputs(&kinds, &out_of_range, &right, 1),
            "valid",
        ));
        let mut invalid_kind = kinds.clone();
        invalid_kind[0] = 6;
        assert!(!execute_predicate(
            &validity,
            &params,
            make_inputs(&invalid_kind, &left, &right, 1),
            "valid",
        ));
        assert!(!execute_predicate(
            &validity,
            &params,
            make_inputs(&kinds, &left, &right, 2),
            "valid",
        ));
    }

    #[test]
    fn every_gate_output_matches_the_ideal_evaluator_for_every_boolean_input() {
        let params = bindings_for(2, 0, 1, 6);
        let circuit = BooleanCircuitData {
            layers: vec![vec![
                BooleanGateData { kind: BooleanGateKind::ConstantFalse, left: 0, right: 0 },
                BooleanGateData { kind: BooleanGateKind::ConstantTrue, left: 0, right: 0 },
                BooleanGateData { kind: BooleanGateKind::Copy, left: 0, right: 0 },
                BooleanGateData { kind: BooleanGateKind::Not, left: 0, right: 0 },
                BooleanGateData { kind: BooleanGateKind::And, left: 0, right: 1 },
                BooleanGateData { kind: BooleanGateKind::Xor, left: 0, right: 1 },
            ]],
            output_source: 0,
        };
        let satisfaction =
            boolean_circuit_satisfaction_predicate(DslContext::new("observable-all-gates"))
                .unwrap();
        let kinds = runtime_family(&[0, 1, 2, 3, 4, 5]);
        let left = runtime_family(&[0, 0, 0, 0, 0, 0]);
        let right = runtime_family(&[0, 0, 0, 0, 1, 1]);

        for left_input in [false, true] {
            for right_input in [false, true] {
                for output_slot in 0..6 {
                    let shape = BooleanCircuitShape {
                        instance_width: 2,
                        witness_width: 0,
                        depth: 1,
                        max_layer_width: 6,
                    };
                    let mut selected = circuit.clone();
                    selected.output_source = output_slot;
                    let expected =
                        selected.evaluate(&shape, &[left_input, right_input], &[]).unwrap();
                    let inputs = BTreeMap::from([
                        ("circuit-active-gate-count".to_owned(), runtime_family(&[6])),
                        ("circuit-gate-kind".to_owned(), kinds.clone()),
                        ("circuit-left-source".to_owned(), left.clone()),
                        ("circuit-right-source".to_owned(), right.clone()),
                        ("circuit-output-source".to_owned(), runtime_family(&[output_slot as i32])),
                        (
                            BOOLEAN_INSTANCE_INPUT.to_owned(),
                            runtime_family(&[
                                i32::from(left_input),
                                i32::from(right_input),
                                0,
                                0,
                                0,
                                0,
                            ]),
                        ),
                        (BOOLEAN_WITNESS_INPUT.to_owned(), runtime_family(&[0, 0, 0, 0, 0, 0])),
                    ]);
                    assert_eq!(
                        execute_predicate(&satisfaction, &params, inputs, "satisfied"),
                        expected,
                        "gate slot {output_slot} disagreed for inputs ({left_input}, {right_input})",
                    );
                }
            }
        }
    }
}
