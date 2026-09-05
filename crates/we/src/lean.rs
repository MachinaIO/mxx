//! Application-owned linking of mechanically exported workflow roots.
//! This assembles a proposition; it does not assert or assume its noise conclusion.
pub mod check;
pub mod diamond;
pub mod numeric;

use crate::WitnessEncryptionProtocolDecl;
use mxx_correctness::{
    ComparatorSpec, InputContract, InputValueContract, ProtocolInputDestination, ProtocolInputId,
    StageId,
};
use mxx_ir_core::{
    lean::{BoundaryValue, LeanArtifact},
    types::ConcreteWireType,
};
use std::collections::BTreeMap;

pub struct ExportedRoots {
    pub stages: BTreeMap<StageId, LeanArtifact>,
    pub requirements: Vec<LeanArtifact>,
    pub ideal: LeanArtifact,
}

fn tuple(values: &[String]) -> String {
    match values {
        [] => "()".into(),
        [value] => value.clone(),
        _ => format!("({}, ())", values.join(", ")),
    }
}

fn project(value: &BoundaryValue, binder: &str) -> String {
    format!("{binder}{}", value.projection.strip_prefix("outputs").expect("output projection"))
}

fn input_contract_predicate(
    contract: &InputValueContract,
    ty: &ConcreteWireType,
    bindings: &mxx_ir_core::ParamEnv,
    value: &str,
    depth: usize,
) -> Result<String, String> {
    let evaluate = |expression: &mxx_ir_core::IntExpr| {
        expression.evaluate(bindings).map_err(|error| error.to_string())
    };
    match (contract, ty) {
        (InputValueContract::IntegerRange { lower, upper }, ConcreteWireType::Int) => {
            let lower = evaluate(lower)?;
            let upper = evaluate(upper)?;
            if lower > upper {
                return Err("input contract has reversed integer range".into());
            }
            Ok(format!("(({lower} : Int) ≤ {value} ∧ {value} ≤ ({upper} : Int))"))
        }
        (InputValueContract::Boolean, ConcreteWireType::Bool) => Ok("True".into()),
        (InputValueContract::Bytes { length }, ConcreteWireType::Bytes { length: actual }) => {
            let expected = usize::try_from(evaluate(length)?)
                .map_err(|_| "input contract byte length is not a valid size")?;
            if expected != *actual {
                return Err("input contract byte length/type mismatch".into());
            }
            Ok(format!("({value}).size = {expected}"))
        }
        (
            InputValueContract::Family { count, element },
            ConcreteWireType::IndexedFamily { count: actual, element: actual_element },
        ) => {
            let expected = usize::try_from(evaluate(count)?)
                .map_err(|_| "input contract family count is not a valid size")?;
            if expected != *actual {
                return Err("input contract family count/type mismatch".into());
            }
            let index = format!("contract_i_{depth}");
            let body = input_contract_predicate(
                element,
                actual_element,
                bindings,
                &format!("({value} {index})"),
                depth + 1,
            )?;
            Ok(format!("(∀ {index} : Fin {expected}, {body})"))
        }
        (
            InputValueContract::MatrixExact { .. } |
            InputValueContract::MatrixBounded { .. } |
            InputValueContract::MatrixLarge { .. } |
            InputValueContract::Trapdoor { .. },
            _,
        ) => Err("unsupported external input contract variant".into()),
        _ => Err("external input contract/type mismatch".into()),
    }
}

fn input_contract_conditions(
    contract: &InputContract,
    externals: &BTreeMap<ProtocolInputId, (String, ConcreteWireType)>,
    bindings: &mxx_ir_core::ParamEnv,
) -> Result<Vec<String>, String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut conditions = Vec::new();
    for entry in &contract.inputs {
        if !seen.insert(&entry.id) {
            return Err("duplicate external input contract ID".into());
        }
        let (field, ty) = externals.get(&entry.id).ok_or("unknown external input contract ID")?;
        conditions.push(input_contract_predicate(
            &entry.value,
            ty,
            bindings,
            &format!("external.{field}"),
            0,
        )?);
    }
    if seen.len() != externals.len() {
        return Err("missing external input contract".into());
    }
    Ok(conditions)
}

/// Assemble shared externals, producer artifacts, all requirement runs and the ideal run.
/// The backend artifact binds the fixed context to the layouts checked during export.
pub fn assemble_claim(
    declaration: &WitnessEncryptionProtocolDecl,
    roots: &ExportedRoots,
    bindings: &mxx_ir_core::ParamEnv,
    backend: &mxx_runtime::lean::LeanBackendArtifact,
) -> Result<String, String> {
    let backend_bindings = backend.exporter_bindings();
    let backend_module = backend.module_name();
    let backend = backend.context_name();
    let protocol = declaration.protocol();
    let bundle = &protocol.bundle;
    let check = |graph: &mxx_ir_core::Graph, artifact: &LeanArtifact| -> Result<(), String> {
        if artifact.backend_layouts.iter().any(|layout| !backend_bindings.contains(layout)) {
            return Err("generated backend does not cover exported layouts".into());
        }
        let expected =
            mxx_ir_core::encoding::spec_hash(graph, bindings).map_err(|error| error.to_string())?;
        if artifact.spec_hash != expected {
            return Err("generated graph or compile bindings mismatch".into());
        }
        Ok(())
    };
    let mut entries: Vec<(&LeanArtifact, String)> = Vec::new();
    let mut stage_positions = BTreeMap::new();
    for (index, stage) in bundle.workflow.stages.iter().enumerate() {
        stage_positions.insert(stage.id.clone(), index);
        let artifact = roots.stages.get(&stage.id).ok_or("missing generated stage")?;
        check(&stage.graph, artifact)?;
        entries.push((artifact, format!("stage_{index}")));
    }
    if roots.requirements.len() != bundle.requirements.len() ||
        bundle.precondition_spec.requirement_outputs.len() != bundle.requirements.len()
    {
        return Err("requirement root count mismatch".into());
    }
    let requirement_start = entries.len();
    for (requirement, artifact) in bundle.requirements.iter().zip(&roots.requirements) {
        check(&requirement.graph, artifact)?;
    }
    check(&bundle.ideal.graph, &roots.ideal)?;
    entries.extend(
        roots
            .requirements
            .iter()
            .enumerate()
            .map(|(index, artifact)| (artifact, format!("requirement_{index}"))),
    );
    let ideal_position = entries.len();
    entries.push((&roots.ideal, "ideal".into()));
    let mut inputs = entries
        .iter()
        .map(|(artifact, _)| vec![None::<String>; artifact.root.input_count])
        .collect::<Vec<_>>();
    let mut external_fields = Vec::new();
    let mut external_ids = std::collections::BTreeSet::new();
    let mut external_contract_fields = BTreeMap::new();
    for (external_index, binding) in bundle.input_bindings.iter().enumerate() {
        if !external_ids.insert(binding.input.clone()) {
            return Err("duplicate external input ID".into());
        }
        let field = format!("input_{external_index}");
        let mut external_type = None;
        for destination in &binding.destinations {
            let (position, name) = match destination {
                ProtocolInputDestination::WorkflowStage { stage, input } => {
                    (*stage_positions.get(stage).ok_or("unknown input stage")?, input.0.as_str())
                }
                ProtocolInputDestination::Requirement { requirement, input } => {
                    if *requirement >= roots.requirements.len() {
                        return Err("unknown requirement destination".into());
                    }
                    (requirement_start + requirement, input.as_str())
                }
                ProtocolInputDestination::Ideal { input } => (ideal_position, input.as_str()),
            };
            let value =
                entries[position].0.root.inputs.get(name).ok_or("missing input destination")?;
            if external_type.as_ref().is_some_and(|ty| ty != &value.wire_type) {
                return Err("external input destination type mismatch".into());
            }
            external_type = Some(value.wire_type.clone());
            if inputs[position][value.tuple_index].replace(format!("external.{field}")).is_some() {
                return Err("duplicate input destination".into());
            }
            if !external_fields.iter().any(|(name, _)| name == &field) {
                external_fields.push((field.clone(), value.lean_type.clone()));
            }
        }
        let external_type = external_type.ok_or("external input has no destination")?;
        external_contract_fields.insert(binding.input.clone(), (field, external_type));
    }
    let contract_conditions =
        input_contract_conditions(&bundle.input_contract, &external_contract_fields, bindings)?;
    for (position, stage) in bundle.workflow.stages.iter().enumerate() {
        for binding in &stage.bindings {
            let producer_position =
                *stage_positions.get(&binding.producer_stage).ok_or("unknown producer")?;
            if producer_position >= position {
                return Err("artifact producer must precede consumer".into());
            }
            let producer = entries[producer_position]
                .0
                .root
                .outputs
                .get(&binding.producer_output.0)
                .ok_or("missing producer output")?;
            let consumer = entries[position]
                .0
                .root
                .inputs
                .get(&binding.consumer_input.0)
                .ok_or("missing artifact consumer input")?;
            if producer.wire_type != consumer.wire_type {
                return Err("artifact value type mismatch".into());
            }
            let value = project(producer, &format!("execution.{}", entries[producer_position].1));
            if inputs[position][consumer.tuple_index].replace(value).is_some() {
                return Err("duplicate artifact/external binding".into());
            }
        }
    }
    let mut source = format!("import {backend_module}\nimport MxxWe.Decoder\n");
    for (artifact, _) in &entries {
        source.push_str(&format!("import {}\n", artifact.module_name));
    }
    source.push_str("\nnamespace GeneratedClaim\n\nstructure ExternalInputs where\n");
    for (field, ty) in external_fields {
        source.push_str(&format!("  {field} : {ty}\n"));
    }
    source.push_str(&format!(
        "\ndef ValidExternals (external : ExternalInputs) : Prop :=\n  {}\n",
        if contract_conditions.is_empty() {
            "True".into()
        } else {
            contract_conditions.join(" ∧\n  ")
        }
    ));
    source.push_str("\nstructure Execution where\n");
    for (artifact, field) in &entries {
        source.push_str(&format!("  {field} : {}\n", artifact.root.output_type));
    }
    let mut conditions = vec!["ValidExternals external".into()];
    for (position, (artifact, field)) in entries.iter().enumerate() {
        let root = &artifact.root;
        let values = inputs[position]
            .iter()
            .cloned()
            .collect::<Option<Vec<_>>>()
            .ok_or("unbound generated root input")?;
        let params = root
            .parameters
            .iter()
            .map(|(name, field)| {
                let value = field.root_value.clone().unwrap_or_else(|| "(0 : Int)".into());
                format!("{name} := {value}")
            })
            .collect::<Vec<_>>()
            .join(", ");
        let context = format!(
            "{}{}",
            if root.requires_backend { format!(" {backend}") } else { String::new() },
            if root.requires_hash_model { " hashModel" } else { "" }
        );
        source.push_str(&format!(
            "\ndef {field}_params : {} := {{ {params} }}\n",
            root.parameter_type
        ));
        conditions.push(format!(
            "{}{} {}_params ({}) execution.{}",
            root.relation,
            context,
            field,
            tuple(&values),
            field
        ));
    }
    for (index, name) in bundle.precondition_spec.requirement_outputs.iter().enumerate() {
        let root = &roots.requirements[index].root;
        let output = root.outputs.get(name).ok_or("missing requirement Boolean output")?;
        if output.lean_type != "Bool" {
            return Err("requirement output is not Boolean".into());
        }
        conditions
            .push(format!("{} = true", project(output, &format!("execution.requirement_{index}"))));
    }
    source.push_str(&format!("\ndef Runs (hashModel : MxxRuntime.HashModel) (external : ExternalInputs)\n    (execution : Execution) : Prop :=\n  {}\n", conditions.join(" ∧\n  ")));
    let ComparatorSpec::Equality { endpoints } = &bundle.comparator else {
        return Err("unsupported comparator".into());
    };
    if endpoints.len() != 1 ||
        bundle.endpoints.entries.len() != 1 ||
        bundle.operational_decoder_targets.len() != 1
    {
        return Err("WE claim currently requires one exact operational endpoint".into());
    }
    let endpoint = &bundle.endpoints.entries[0];
    let comparison = &endpoints[0];
    if comparison.endpoint != endpoint.spec ||
        comparison.actual_input != endpoint.workflow_output.output ||
        comparison.ideal_input != endpoint.ideal_output
    {
        return Err("comparator endpoint mismatch".into());
    }
    let actual_position =
        *stage_positions.get(&endpoint.workflow_output.stage).ok_or("missing endpoint stage")?;
    let actual = entries[actual_position]
        .0
        .root
        .outputs
        .get(&endpoint.workflow_output.output)
        .ok_or("missing actual endpoint")?;
    let ideal =
        roots.ideal.root.outputs.get(&endpoint.ideal_output).ok_or("missing ideal endpoint")?;
    if actual.lean_type != "Bool" || ideal.lean_type != "Bool" {
        return Err("WE endpoint must be Boolean".into());
    }
    let target = &bundle.operational_decoder_targets[0];
    if target.decoder_stage != endpoint.workflow_output.stage ||
        target.decoder_node != actual.wire.node ||
        !matches!(target.kind, mxx_correctness::OperationalDecoderKind::BooleanInterval)
    {
        return Err("operational decoder does not identify the actual endpoint".into());
    }
    let residual_position =
        *stage_positions.get(&target.residual_stage).ok_or("missing residual stage")?;
    let residual = entries[residual_position]
        .0
        .root
        .outputs
        .get(&target.residual_output)
        .ok_or("missing residual output")?;
    let mxx_ir_core::types::ConcreteWireType::Matrix(matrix) = &residual.wire_type else {
        return Err("residual must be a matrix".into());
    };
    if !matrix.is_scalar() || matrix.ring_dimension == 0 {
        return Err("residual must be a scalar polynomial".into());
    }
    let q = &matrix.modulus;
    let ideal_value = project(ideal, "execution.ideal");
    let actual_value = project(actual, &format!("execution.{}", entries[actual_position].1));
    let residual_value = project(residual, &format!("execution.{}", entries[residual_position].1));
    source.push_str(&format!("\nnoncomputable def observedResidual (execution : Execution) : Int :=\n  Mxx.Primitives.centeredLift {q}\n    ((({residual_value}) 0 0).coeff ⟨0, by decide⟩ -\n      (MxxWe.messageCenter {q} {ideal_value} : ZMod {q}))\n\n"));
    source.push_str(&format!("/-- The application proof must establish this proposition; no noise premise is assumed. -/\ndef CorrectnessClaim : Prop :=\n  ∀ hashModel external execution, Runs hashModel external execution →\n    (observedResidual execution).natAbs < MxxWe.decoderRadius {q} ∧\n    {actual_value} = {ideal_value}\n\nend GeneratedClaim\n"));
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_correctness::InputContractEntry;
    use mxx_ir_core::{IntExpr, ParamEnv};

    fn bit_range() -> InputValueContract {
        InputValueContract::IntegerRange {
            lower: IntExpr::constant(0),
            upper: IntExpr::constant(1),
        }
    }

    #[test]
    fn input_contract_family_is_symbolic_and_inclusive() {
        let contract = InputValueContract::Family {
            count: IntExpr::constant(1_000_000),
            element: Box::new(bit_range()),
        };
        let ty = ConcreteWireType::IndexedFamily {
            count: 1_000_000,
            element: Box::new(ConcreteWireType::Int),
        };
        let predicate =
            input_contract_predicate(&contract, &ty, &ParamEnv::default(), "external.bits", 0)
                .unwrap();
        assert_eq!(predicate.matches("∀").count(), 1);
        assert!(predicate.contains("Fin 1000000"));
        assert!(predicate.contains("(0 : Int) ≤ (external.bits contract_i_0)"));
        assert!(predicate.contains("(external.bits contract_i_0) ≤ (1 : Int)"));
        assert!(predicate.len() < 200);
    }

    #[test]
    fn input_contract_types_and_sizes_are_checked() {
        let env = ParamEnv::default();
        assert!(
            input_contract_predicate(&bit_range(), &ConcreteWireType::Bool, &env, "x", 0).is_err()
        );
        assert_eq!(
            input_contract_predicate(
                &InputValueContract::Boolean,
                &ConcreteWireType::Bool,
                &env,
                "x",
                0
            )
            .unwrap(),
            "True"
        );
        let bytes = InputValueContract::Bytes { length: IntExpr::constant(32) };
        assert_eq!(
            input_contract_predicate(&bytes, &ConcreteWireType::Bytes { length: 32 }, &env, "x", 0)
                .unwrap(),
            "(x).size = 32"
        );
        assert!(
            input_contract_predicate(&bytes, &ConcreteWireType::Bytes { length: 31 }, &env, "x", 0)
                .is_err()
        );
        for count in [-1, 2] {
            let family = InputValueContract::Family {
                count: IntExpr::constant(count),
                element: Box::new(bit_range()),
            };
            let ty = ConcreteWireType::IndexedFamily {
                count: 1,
                element: Box::new(ConcreteWireType::Int),
            };
            assert!(input_contract_predicate(&family, &ty, &env, "x", 0).is_err());
        }
    }

    #[test]
    fn input_contract_mapping_uses_exact_ids_and_coverage() {
        let id = ProtocolInputId::from("raw_bits");
        let externals = BTreeMap::from([(id.clone(), ("input_7".into(), ConcreteWireType::Int))]);
        let entry = InputContractEntry { id, name: "not_the_identity".into(), value: bit_range() };
        let mut contract = InputContract { inputs: vec![entry.clone()] };
        let env = ParamEnv::default();
        assert!(
            input_contract_conditions(&contract, &externals, &env).unwrap()[0]
                .contains("external.input_7")
        );
        contract.inputs.push(entry);
        assert!(input_contract_conditions(&contract, &externals, &env).is_err());
        contract.inputs.clear();
        assert!(input_contract_conditions(&contract, &externals, &env).is_err());
        contract.inputs.push(InputContractEntry {
            id: ProtocolInputId::from("unknown"),
            name: "raw_bits".into(),
            value: bit_range(),
        });
        assert!(input_contract_conditions(&contract, &externals, &env).is_err());
    }
}
