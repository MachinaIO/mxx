//! Application-independent linking of exported graphs and threshold correctness claims.
//! Claims are propositions to prove, never assumed conclusions.
use super::{BackendLayout, BoundaryValue, LeanArtifact};
use crate::{Graph, IntExpr, ParamEnv, types::ConcreteWireType};
use std::collections::BTreeSet;

/// Supported external-input predicates. Recursive families stay symbolic in Lean.
#[derive(Clone, Debug)]
pub enum InputContract {
    IntegerRange { lower: IntExpr, upper: IntExpr },
    Boolean,
    Bytes { length: IntExpr },
    Family { count: IntExpr, element: Box<InputContract> },
}

/// A named input or output of a root, identified by its position in `LinkedClaim::roots`.
#[derive(Clone, Debug)]
pub struct Port {
    pub root: usize,
    pub name: String,
}

pub struct ClaimRoot<'a> {
    pub graph: &'a Graph,
    pub artifact: &'a LeanArtifact,
    pub field: String,
}

pub struct ExternalInput {
    pub contract: InputContract,
    pub destinations: Vec<Port>,
}

pub struct Link {
    pub producer: Port,
    pub consumer: Port,
}

/// Names supplied by the owning semantic Lean packages. No semantic premise is added.
pub struct ClaimSemantics<'a> {
    pub imports: &'a [&'a str],
    pub hash_model_type: &'a str,
    pub centered_lift: &'a str,
    pub message_center: &'a str,
    pub decoder_radius: &'a str,
}

pub struct ClaimBackend<'a> {
    pub module_name: &'a str,
    pub context_name: &'a str,
    pub layouts: &'a [BackendLayout],
}

/// Shared externals, acyclic graph connections and one Boolean threshold endpoint.
/// The residual is a scalar polynomial; the conclusion bounds its constant coefficient
/// relative to the ideal Boolean's configured message center and equates both outputs.
pub struct LinkedClaim<'a> {
    pub roots: Vec<ClaimRoot<'a>>,
    pub externals: Vec<ExternalInput>,
    pub links: Vec<Link>,
    pub requirements: Vec<Port>,
    pub actual: Port,
    pub ideal: Port,
    pub residual: Port,
}

fn tuple(values: &[String]) -> String {
    match values {
        [] => "()".into(),
        [value] => value.clone(),
        _ => format!("({}, ())", values.join(", ")),
    }
}

fn project(value: &BoundaryValue, binder: &str) -> String {
    // Every boundary has already been checked against its graph and tuple layout.
    format!("{binder}{}", &value.projection["outputs".len()..])
}

fn input_contract_predicate(
    contract: &InputContract,
    ty: &ConcreteWireType,
    bindings: &crate::ParamEnv,
    value: &str,
    depth: usize,
) -> Result<String, String> {
    let evaluate = |expression: &crate::IntExpr| {
        expression.evaluate(bindings).map_err(|error| error.to_string())
    };
    match (contract, ty) {
        (InputContract::IntegerRange { lower, upper }, ConcreteWireType::Int) => {
            let lower = evaluate(lower)?;
            let upper = evaluate(upper)?;
            if lower > upper {
                return Err("input contract has reversed integer range".into());
            }
            Ok(format!("(({lower} : Int) ≤ {value} ∧ {value} ≤ ({upper} : Int))"))
        }
        (InputContract::Boolean, ConcreteWireType::Bool) => Ok("True".into()),
        (InputContract::Bytes { length }, ConcreteWireType::Bytes { length: actual }) => {
            let expected = usize::try_from(evaluate(length)?)
                .map_err(|_| "input contract byte length is not a valid size")?;
            if expected != *actual {
                return Err("input contract byte length/type mismatch".into());
            }
            Ok(format!("({value}).size = {expected}"))
        }
        (
            InputContract::Family { count, element },
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
        _ => Err("external input contract/type mismatch".into()),
    }
}

fn check_root(
    root: &ClaimRoot<'_>,
    bindings: &ParamEnv,
    backend: &ClaimBackend<'_>,
) -> Result<(), String> {
    let artifact = root.artifact;
    if artifact.backend_layouts.iter().any(|layout| !backend.layouts.contains(layout)) {
        return Err("generated backend does not cover exported layouts".into());
    }
    let expected = crate::encoding::spec_hash(root.graph, bindings).map_err(|e| e.to_string())?;
    if artifact.spec_hash != expected {
        return Err("generated graph or compile bindings mismatch".into());
    }
    let scope = root.graph.root_scope();
    let input_wires = super::scope_input_wires(scope);
    let output_wires = scope.outputs();
    if artifact.root.input_count != input_wires.len() ||
        artifact.root.output_count != output_wires.len() ||
        artifact.root.inputs.len() != input_wires.len() ||
        artifact.root.outputs.len() != root.graph.outputs().len()
    {
        return Err("generated root boundary count mismatch".into());
    }
    for (is_input, values, wires) in [
        (true, &artifact.root.inputs, input_wires.as_slice()),
        (false, &artifact.root.outputs, output_wires),
    ] {
        for (name, value) in values {
            if wires.get(value.tuple_index) != Some(&value.wire) ||
                value.projection !=
                    super::tuple_projection(
                        if is_input { "inputs" } else { "outputs" },
                        value.tuple_index,
                        wires.len(),
                    )
            {
                return Err("generated boundary projection mismatch".into());
            }
            let node = scope.node(value.wire.node).ok_or("generated boundary node missing")?;
            if is_input {
                if !matches!(node.kind(), crate::node::NodeKind::Input { name: expected, .. } if expected == name)
                {
                    return Err("generated input identity mismatch".into());
                }
            } else if root.graph.outputs().get(name).map(|output| output.value) != Some(value.wire)
            {
                return Err("generated output identity mismatch".into());
            }
            let ty = node
                .output_types()
                .get(value.wire.port.0 as usize)
                .ok_or("generated boundary port missing")?;
            if crate::validate::concretize_wire_type(
                ty,
                bindings,
                &crate::FrozenGraphScopeId::Root,
                value.wire.node,
            )
            .map_err(|e| e.to_string())? !=
                value.wire_type
            {
                return Err("generated boundary wire type mismatch".into());
            }
        }
    }
    Ok(())
}

fn output<'a>(claim: &'a LinkedClaim<'_>, port: &Port) -> Result<&'a BoundaryValue, String> {
    claim
        .roots
        .get(port.root)
        .ok_or("unknown output root")?
        .artifact
        .root
        .outputs
        .get(&port.name)
        .ok_or_else(|| "missing output port".into())
}

/// Assemble a claim after checking graph identities, backend coverage and exact boundary wiring.
pub fn assemble_claim(
    claim: &LinkedClaim<'_>,
    bindings: &ParamEnv,
    backend: &ClaimBackend<'_>,
    semantics: &ClaimSemantics<'_>,
) -> Result<String, String> {
    let mut fields = BTreeSet::new();
    for root in &claim.roots {
        if super::valid_identifier(&root.field).is_err() || !fields.insert(&root.field) {
            return Err("invalid or duplicate execution field".into());
        }
        check_root(root, bindings, backend)?;
    }
    let entries =
        claim.roots.iter().map(|root| (root.artifact, root.field.clone())).collect::<Vec<_>>();
    let mut inputs = entries
        .iter()
        .map(|(artifact, _)| vec![None::<String>; artifact.root.input_count])
        .collect::<Vec<_>>();
    let mut external_fields = Vec::new();
    let mut contract_conditions = Vec::new();
    for (index, external) in claim.externals.iter().enumerate() {
        let field = format!("input_{index}");
        let mut external_type = None;
        let mut lean_type = None;
        for destination in &external.destinations {
            let root = entries.get(destination.root).ok_or("unknown input root")?.0;
            let value =
                root.root.inputs.get(&destination.name).ok_or("missing input destination")?;
            if external_type.as_ref().is_some_and(|ty| ty != &value.wire_type) ||
                lean_type.as_ref().is_some_and(|ty| ty != &value.lean_type)
            {
                return Err("external input destination type mismatch".into());
            }
            external_type = Some(value.wire_type.clone());
            lean_type = Some(value.lean_type.clone());
            if inputs[destination.root][value.tuple_index]
                .replace(format!("external.{field}"))
                .is_some()
            {
                return Err("duplicate input destination".into());
            }
        }
        let ty = external_type.ok_or("external input has no destination")?;
        contract_conditions.push(input_contract_predicate(
            &external.contract,
            &ty,
            bindings,
            &format!("external.{field}"),
            0,
        )?);
        external_fields.push((field, lean_type.expect("typed destination")));
    }
    for link in &claim.links {
        if link.producer.root >= link.consumer.root {
            return Err("artifact producer must precede consumer".into());
        }
        let producer = output(claim, &link.producer)?;
        let consumer = entries
            .get(link.consumer.root)
            .ok_or("unknown consumer root")?
            .0
            .root
            .inputs
            .get(&link.consumer.name)
            .ok_or("missing artifact consumer input")?;
        if producer.wire_type != consumer.wire_type || producer.lean_type != consumer.lean_type {
            return Err("artifact value type mismatch".into());
        }
        let value = project(producer, &format!("execution.«{}»", entries[link.producer.root].1));
        if inputs[link.consumer.root][consumer.tuple_index].replace(value).is_some() {
            return Err("duplicate artifact/external binding".into());
        }
    }
    let backend_module = backend.module_name;
    let backend = backend.context_name;
    let mut source = format!("import {backend_module}\n");
    for import in semantics.imports {
        source.push_str(&format!("import {import}\n"));
    }
    for (artifact, _) in &entries {
        source.push_str(&format!("import {}\n", artifact.module_name));
    }
    source.push_str("\nnamespace GeneratedClaim\n\nstructure ExternalInputs where\n");
    for (field, ty) in external_fields {
        source.push_str(&format!("  {field} : {ty}\n"));
    }
    source.push_str(&format!(
        "\ndef ValidExternals ({} : ExternalInputs) : Prop :=\n  {}\n",
        if contract_conditions.iter().any(|condition| condition.contains("external.")) {
            "external"
        } else {
            "_"
        },
        if contract_conditions.is_empty() {
            "True".into()
        } else {
            contract_conditions.join(" ∧\n  ")
        }
    ));
    source.push_str("\nstructure Execution where\n");
    for (artifact, field) in &entries {
        source.push_str(&format!("  «{field}» : {}\n", artifact.root.output_type));
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
                format!("«{name}» := {value}")
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
            "{}{} {}_params ({}) execution.«{}»",
            root.relation,
            context,
            field,
            tuple(&values),
            field
        ));
    }
    for requirement in &claim.requirements {
        let value = output(claim, requirement)?;
        if value.wire_type != ConcreteWireType::Bool || value.lean_type != "Bool" {
            return Err("requirement output is not Boolean".into());
        }
        conditions.push(format!(
            "{} = true",
            project(value, &format!("execution.«{}»", entries[requirement.root].1))
        ));
    }
    let hash_binder = if entries.iter().any(|(artifact, _)| artifact.root.requires_hash_model) {
        "hashModel"
    } else {
        "_"
    };
    source.push_str(&format!("\ndef Runs ({hash_binder} : {}) (external : ExternalInputs)\n    (execution : Execution) : Prop :=\n  {}\n", semantics.hash_model_type, conditions.join(" ∧\n  ")));
    let actual = output(claim, &claim.actual)?;
    let ideal = output(claim, &claim.ideal)?;
    if actual.wire_type != ConcreteWireType::Bool ||
        ideal.wire_type != ConcreteWireType::Bool ||
        actual.lean_type != "Bool" ||
        ideal.lean_type != "Bool"
    {
        return Err("claim endpoint must be Boolean".into());
    }
    let residual = output(claim, &claim.residual)?;
    let ConcreteWireType::Matrix(matrix) = &residual.wire_type else {
        return Err("residual must be a matrix".into());
    };
    if !matrix.is_scalar() || matrix.ring_dimension == 0 {
        return Err("residual must be a scalar polynomial".into());
    }
    let q = &matrix.modulus;
    let ideal_value = project(ideal, &format!("execution.«{}»", entries[claim.ideal.root].1));
    let actual_value = project(actual, &format!("execution.«{}»", entries[claim.actual.root].1));
    let residual_value =
        project(residual, &format!("execution.«{}»", entries[claim.residual.root].1));
    let centered_lift = semantics.centered_lift;
    let message_center = semantics.message_center;
    let decoder_radius = semantics.decoder_radius;
    source.push_str(&format!("\nnoncomputable def observedResidual (execution : Execution) : Int :=\n  {centered_lift} {q}\n    ((({residual_value}) 0 0).coeff ⟨0, by decide⟩ -\n      ({message_center} {q} {ideal_value} : ZMod {q}))\n\n"));
    source.push_str(&format!("/-- The application proof must establish this proposition; no noise premise is assumed. -/\ndef CorrectnessClaim : Prop :=\n  ∀ hashModel external execution, Runs hashModel external execution →\n    (observedResidual execution).natAbs < {decoder_radius} {q} ∧\n    {actual_value} = {ideal_value}\n\nend GeneratedClaim\n"));
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IntExpr, ParamEnv};

    fn exported_graph() -> (Graph, LeanArtifact) {
        use crate::{
            GraphOutput, NodeHandle,
            node::{ConstantMatrix, NodeKind},
            types::{MatrixType, WireType},
        };
        let bit = NodeHandle::new(
            NodeKind::Input { name: "bit".into(), wire_type: WireType::Bool, artifact: None },
            vec![],
            vec![WireType::Bool],
        )
        .output(0)
        .unwrap();
        let matrix = MatrixType {
            modulus: 17.into(),
            ring_dimension: 2.into(),
            rows: 1.into(),
            columns: 1.into(),
        };
        let residual = NodeHandle::new(
            NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value: ConstantMatrix::Zero },
            vec![],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "linked-claim",
            vec![],
            std::collections::BTreeMap::from([
                ("bit".into(), GraphOutput { value: bit, confidentiality: None }),
                ("residual".into(), GraphOutput { value: residual, confidentiality: None }),
            ]),
            vec![],
            vec![],
            Default::default(),
        )
        .unwrap()
        .0;
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact =
            super::super::export(&validated, &super::super::ExportOptions::default()).unwrap();
        (graph, artifact)
    }

    fn linked<'a>(graph: &'a Graph, artifact: &'a LeanArtifact) -> LinkedClaim<'a> {
        LinkedClaim {
            roots: vec![
                ClaimRoot { graph, artifact, field: "producer".into() },
                ClaimRoot { graph, artifact, field: "ideal".into() },
            ],
            externals: vec![ExternalInput {
                contract: InputContract::Boolean,
                destinations: vec![
                    Port { root: 0, name: "bit".into() },
                    Port { root: 1, name: "bit".into() },
                ],
            }],
            links: vec![],
            requirements: vec![],
            actual: Port { root: 0, name: "bit".into() },
            ideal: Port { root: 1, name: "bit".into() },
            residual: Port { root: 0, name: "residual".into() },
        }
    }

    fn render(claim: &LinkedClaim<'_>) -> Result<String, String> {
        assemble_claim(
            claim,
            &ParamEnv::default(),
            &ClaimBackend { module_name: "Backend", context_name: "Backend.context", layouts: &[] },
            &ClaimSemantics {
                imports: &["OtherApplication.Semantics"],
                hash_model_type: "OtherApplication.HashModel",
                centered_lift: "OtherApplication.centeredLift",
                message_center: "OtherApplication.messageCenter",
                decoder_radius: "OtherApplication.decoderRadius",
            },
        )
    }

    #[test]
    fn linked_claim_uses_configured_semantics_and_shared_externals() {
        let (graph, artifact) = exported_graph();
        let source = render(&linked(&graph, &artifact)).unwrap();
        assert!(source.contains("OtherApplication.messageCenter 17"));
        assert!(source.contains("OtherApplication.decoderRadius 17"));
        assert!(!source.contains("MxxWe"));
        assert_eq!(source.matches("(external.input_0)").count(), 2);
        let (runs, conclusion) = source.split_once("def CorrectnessClaim").unwrap();
        assert!(!runs.contains(".natAbs <"));
        assert!(conclusion.contains("(observedResidual execution).natAbs <"));
        assert!(conclusion.contains("execution.«producer»"));
        assert!(conclusion.contains("execution.«ideal»"));
    }

    #[test]
    fn linked_claim_quotes_execution_fields_in_declarations_links_and_conclusions() {
        let (graph, artifact) = exported_graph();
        let mut claim = linked(&graph, &artifact);
        claim.roots[0].field = "match".into();
        claim.roots[1].field = "namespace".into();
        claim.externals[0].destinations.pop();
        claim.links.push(Link {
            producer: claim.actual.clone(),
            consumer: Port { root: 1, name: "bit".into() },
        });
        claim.requirements.push(claim.actual.clone());
        let source = assemble_claim(
            &claim,
            &ParamEnv::default(),
            &ClaimBackend {
                module_name: "MxxRuntime",
                context_name: "unusedBackend",
                layouts: &[],
            },
            &ClaimSemantics {
                imports: &["Decoder"],
                hash_model_type: "MxxRuntime.HashModel",
                centered_lift: "Mxx.Primitives.centeredLift",
                message_center: "MxxWe.messageCenter",
                decoder_radius: "MxxWe.decoderRadius",
            },
        )
        .unwrap();
        assert!(source.contains("  «match» :"));
        assert!(source.contains("  «namespace» :"));
        assert!(source.contains("namespace_params (execution.«match».1) execution.«namespace»"));
        assert!(source.contains("execution.«match».1 = true"));
        assert!(source.contains("execution.«match».1 = execution.«namespace».1"));
        assert!(source.contains("execution.«match».2.1"));
        assert!(!source.contains("execution.match"));
        assert!(!source.contains("execution.namespace"));
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/lean_ir_fixtures/claim_identifiers");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join(format!("{}.lean", artifact.module_name)), &artifact.source)
            .unwrap();
        std::fs::write(directory.join("Claim.lean"), source).unwrap();
    }

    #[test]
    fn linked_claim_rejects_missing_duplicate_backward_and_mistyped_links() {
        let (graph, artifact) = exported_graph();
        let mut claim = linked(&graph, &artifact);
        claim.externals[0].destinations.pop();
        assert!(render(&claim).unwrap_err().contains("unbound"));
        claim.links.push(Link {
            producer: claim.actual.clone(),
            consumer: Port { root: 1, name: "bit".into() },
        });
        assert!(render(&claim).is_ok());
        claim.links.push(Link {
            producer: claim.actual.clone(),
            consumer: Port { root: 1, name: "bit".into() },
        });
        assert!(render(&claim).unwrap_err().contains("duplicate"));
        claim.links.pop();
        claim.links[0].producer = claim.residual.clone();
        assert!(render(&claim).unwrap_err().contains("type mismatch"));
        claim.links[0].producer = claim.ideal.clone();
        assert!(render(&claim).unwrap_err().contains("precede"));
        claim.links.clear();
        claim.externals[0].destinations.push(Port { root: 1, name: "bit".into() });
        claim.requirements.push(claim.residual.clone());
        assert!(render(&claim).unwrap_err().contains("not Boolean"));
        claim.requirements.clear();
        claim.actual = claim.residual.clone();
        assert!(render(&claim).unwrap_err().contains("must be Boolean"));
    }

    #[test]
    fn linked_claim_rejects_mismatched_graph_layout_and_boundary_metadata() {
        let (graph, artifact) = exported_graph();
        let mut wrong = artifact.clone();
        wrong.spec_hash.0[0] ^= 1;
        assert!(render(&linked(&graph, &wrong)).unwrap_err().contains("bindings mismatch"));
        wrong = artifact.clone();
        wrong.backend_layouts.push(BackendLayout {
            modulus: 17.into(),
            ring_dimension: 2,
            base: 2.into(),
            regular_digits: 1,
        });
        assert!(render(&linked(&graph, &wrong)).unwrap_err().contains("cover exported layouts"));
        wrong = artifact.clone();
        wrong.root.outputs.get_mut("bit").unwrap().projection = "outputs.2.2.1".into();
        assert!(render(&linked(&graph, &wrong)).unwrap_err().contains("projection mismatch"));
        wrong = artifact.clone();
        wrong.root.inputs.get_mut("bit").unwrap().tuple_index = usize::MAX;
        assert!(render(&linked(&graph, &wrong)).unwrap_err().contains("projection mismatch"));
        wrong = artifact.clone();
        wrong.root.outputs.get_mut("bit").unwrap().wire_type = ConcreteWireType::Int;
        assert!(render(&linked(&graph, &wrong)).unwrap_err().contains("wire type mismatch"));
    }

    fn bit_range() -> InputContract {
        InputContract::IntegerRange { lower: IntExpr::constant(0), upper: IntExpr::constant(1) }
    }

    #[test]
    fn input_contract_family_is_symbolic_and_inclusive() {
        let contract = InputContract::Family {
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
                &InputContract::Boolean,
                &ConcreteWireType::Bool,
                &env,
                "x",
                0
            )
            .unwrap(),
            "True"
        );
        let bytes = InputContract::Bytes { length: IntExpr::constant(32) };
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
            let family = InputContract::Family {
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
}
