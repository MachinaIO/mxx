//! Parameterized layered Boolean circuits represented by flattened DSL families.

use mxx_dsl::{Bool, DslContext, DslError, Family, Int, Mat, iterate, parallel, select};
use mxx_ir_core::IntExpr;

pub const BOOLEAN_INSTANCE_INPUT: &str = "boolean-instance";
pub const BOOLEAN_WITNESS_INPUT: &str = "boolean-witness";

fn bool_all(values: impl IntoIterator<Item = Bool>) -> Bool {
    values.into_iter().fold(Bool::constant(true), |left, right| left & right)
}

fn bool_exactly_one(values: impl IntoIterator<Item = Bool>) -> Bool {
    values.into_iter().map(Bool::to_int).fold(Int::constant(0), Int::add).equal(1)
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
        &self.instance_width + &self.witness_width
    }

    pub fn flattened_gate_count(&self) -> IntExpr {
        &self.depth * &self.max_layer_width
    }

    fn flattened_index(&self, layer: &Int, slot: &Int) -> Int {
        layer * Int::evaluate(self.max_layer_width.clone()) + slot
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
        self.output_sources.at(0)
    }
}

#[derive(Clone)]
pub struct GateSlot {
    pub layer: Int,
    pub index: Int,
}

pub trait BooleanLayerGate<T> {
    /// Builds the six candidates in opcode order: false, true, copy, not, and, xor.
    ///
    /// A single call lets handlers share common executable subexpressions across candidates.
    fn candidates(&self, slot: GateSlot, left: T, right: T) -> Result<[T; 6], DslError>;
}

/// Optional owner hooks for matrix-valued Boolean evaluation.
///
/// The generic gadget does not assign symbolic meaning.  An owning implementation may retain
/// frozen derivation references on the initial family and each selected lane value; the Rust
/// operational checker validates and applies those references.
pub trait BooleanMatrixLayerGate: BooleanLayerGate<Mat> {
    fn retain_initial_family(&self, family: Family<Mat>) -> Result<Family<Mat>, DslError> {
        Ok(family)
    }

    fn retain_selected_value(&self, value: Mat) -> Result<Mat, DslError> {
        Ok(value)
    }
}

pub fn evaluate_boolean_matrix_family<H>(
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: Family<Mat>,
    handler: H,
) -> Result<Family<Mat>, DslError>
where
    H: BooleanMatrixLayerGate,
{
    let preceding = handler.retain_initial_family(preceding)?;
    iterate(params.depth.clone(), preceding, |layer, preceding| {
        let active_count = circuit.active_gate_counts.at(&layer);
        parallel(params.max_layer_width.clone(), |index| {
            let flat = params.flattened_index(&layer, &index);
            let kind = circuit.gate_kinds.at(&flat);
            let left = preceding.at(circuit.left_sources.at(&flat));
            let right = preceding.at(circuit.right_sources.at(&flat));
            let slot = GateSlot { layer: layer.clone(), index: index.clone() };
            let candidates = handler.candidates(slot, left, right)?;
            let constant_false = candidates[0].clone();
            let selected = select(kind, candidates.into_iter().collect())?;
            let active = index.less_equal(&active_count - 1);
            let selected = select(active, vec![constant_false, selected])?;
            handler.retain_selected_value(selected)
        })
    })
}

pub fn evaluate_boolean_family(
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: Family<Bool>,
) -> Result<Family<Bool>, DslError> {
    iterate(params.depth.clone(), preceding, |layer, preceding| {
        let active_count = circuit.active_gate_counts.at(&layer);
        parallel(params.max_layer_width.clone(), |index| {
            let flat = params.flattened_index(&layer, &index);
            let kind = circuit.gate_kinds.at(&flat);
            let left = preceding.at(circuit.left_sources.at(&flat));
            let right = preceding.at(circuit.right_sources.at(&flat));
            let not = !&left;
            let and = &left & &right;
            let xor = &left ^ right;
            let selected = select(
                kind,
                vec![Bool::constant(false), Bool::constant(true), left, not, and, xor],
            )?;
            let active = index.less_equal(&active_count - 1);
            select(active, vec![Bool::constant(false), selected])
        })
    })
}

pub fn select_boolean_output(
    circuit: &BooleanCircuitFamilyInputs,
    final_layer: &Family<Bool>,
) -> Bool {
    final_layer.at(circuit.output_source())
}

pub fn select_boolean_matrix_output(
    circuit: &BooleanCircuitFamilyInputs,
    final_layer: &Family<Mat>,
) -> Mat {
    final_layer.at(circuit.output_source())
}

fn gate_record_valid(kind: Int, left: Int, right: Int, previous_width: Int) -> Bool {
    let zero = Int::constant(0);
    let left_in_range =
        zero.clone().less_equal(left.clone()) & left.clone().less_equal(&previous_width - 1);
    let right_in_range =
        zero.clone().less_equal(right.clone()) & right.clone().less_equal(previous_width - 1);
    let left_zero = left.equal(zero.clone());
    let right_zero = right.equal(zero);
    bool_exactly_one([
        bool_all([kind.clone().equal(0), left_zero.clone(), right_zero.clone()]),
        bool_all([kind.clone().equal(1), left_zero, right_zero.clone()]),
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
    iterate(count, Bool::constant(true), |index, result| Ok(result & values.at(index)))
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
    let initial_validity = parallel(params.max_layer_width.clone(), |_| Ok(Bool::constant(true)))?;
    let (slot_validity, final_active_count) = iterate(
        params.depth.clone(),
        (initial_validity, input_width.clone()),
        |layer, (previous_validity, previous_width)| {
            let active_count = circuit.active_gate_counts.at(&layer);
            let active_count_valid = Int::constant(1).less_equal(active_count.clone()) &
                active_count.clone().less_equal(max_width.clone());
            let validity = parallel(params.max_layer_width.clone(), |slot| {
                let flat = params.flattened_index(&layer, &slot);
                let kind = circuit.gate_kinds.at(&flat);
                let left = circuit.left_sources.at(&flat);
                let right = circuit.right_sources.at(&flat);
                let active = slot.clone().less_equal(&active_count - 1);
                let record_valid = select(
                    active,
                    vec![
                        bool_all([
                            kind.clone().equal(0),
                            left.clone().equal(0),
                            right.clone().equal(0),
                        ]),
                        gate_record_valid(kind, left, right, previous_width.clone()),
                    ],
                )?;
                Ok(bool_all([previous_validity.at(slot), record_valid, active_count_valid.clone()]))
            })?;
            Ok((validity, active_count))
        },
    )?;
    let records_valid = reduce_bool_family(slot_validity, params.max_layer_width.clone())?;
    let output_source = circuit.output_source();
    let output_valid = Int::constant(0).less_equal(output_source.clone()) &
        output_source.less_equal(final_active_count - 1);
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
            .build()?
            .graph,
    )
    .map_err(Into::into)
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
    let canonical_family = |values: &Family<Int>, width: &Int| {
        parallel(params.max_layer_width.clone(), |slot| {
            let value = values.at(&slot);
            let active = slot.less_equal(width - 1);
            let binary = value.clone().equal(0) | value.clone().equal(1);
            select(active, vec![value.equal(0), binary])
        })
    };
    let instance_validity = canonical_family(&encoded_instances, &instance_width)?;
    let witness_validity = canonical_family(&encoded_witnesses, &witness_width)?;
    let input_validity = parallel(params.max_layer_width.clone(), |index| {
        Ok(instance_validity.at(&index) & witness_validity.at(index))
    })?;
    let input_width = context.evaluate_int(params.input_width());
    let inputs = parallel(params.max_layer_width.clone(), |index| {
        let instance = encoded_instances.at(&index);
        let instance_active = index.clone().less_equal(&instance_width - 1);
        let input_active = index.clone().less_equal(&input_width - 1);
        let witness_active = input_active.clone().to_int() - instance_active.clone().to_int();
        // Inactive lanes read index zero because selection evaluates both candidate values.
        let witness_index = witness_active * (index - &instance_width);
        let witness = encoded_witnesses.at(witness_index);
        let value = select(
            instance_active,
            vec![select(input_active, vec![Int::constant(0), witness])?, instance],
        )?;
        Ok(value.equal(1))
    })?;
    let final_layer = evaluate_boolean_family(&params, circuit.clone(), inputs)?;
    let output = select_boolean_output(&circuit, &final_layer);
    let inputs_valid = reduce_bool_family(input_validity, params.max_layer_width)?;
    mxx_dsl::PurePredicateSpec::new(
        context.bool_output("satisfied", inputs_valid & output)?.build()?.graph,
    )
    .map_err(Into::into)
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
            &mut cpu_backend([DCRTPolyParams::new(8, 1, 20, 4, None, None)]),
            inputs,
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
        )
        .unwrap();
        matches!(result.outputs[output], RuntimeValue::Bool(true))
    }

    #[test]
    fn symbolic_boolean_evaluation_uses_iteration_and_dynamic_flattened_access() {
        let (context, params) =
            BooleanCircuitFamilyParams::declare(DslContext::new("symbolic-boolean"));
        let circuit = BooleanCircuitFamilyInputs::protocol_inputs(&context, &params);
        let encoded_inputs = context.int_family_input("inputs", params.max_layer_width.clone());
        let inputs =
            parallel(params.max_layer_width.clone(), |index| Ok(encoded_inputs.at(index).equal(1)))
                .unwrap();
        let output = evaluate_boolean_family(&params, circuit.clone(), inputs).unwrap();
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
