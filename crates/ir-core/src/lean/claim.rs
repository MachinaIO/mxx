//! Application-independent linking of exported graphs and their correctness claims.
//! Claims are propositions to prove, never assumed conclusions. A claim states only what the
//! protocol computes: the workflow endpoint equals the ideal one, for every execution or except
//! with a bounded probability. Noise bounds and other margins are proof obligations, not part of
//! the statement.
use super::{BoundaryValue, LeanArtifact};
use crate::{Graph, IntExpr, ParamEnv, types::ConcreteWireType};
use std::collections::BTreeSet;

/// Supported external-input predicates. Recursive families stay symbolic in Lean.
#[derive(Clone, Debug)]
pub enum InputContract {
    IntegerRange {
        lower: IntExpr,
        upper: IntExpr,
    },
    Boolean,
    /// A scalar polynomial matrix whose value is exactly zero or one.
    BooleanPolynomial,
    /// A scalar polynomial equal to a power of the negacyclic indeterminate.
    /// Exponents in the second half represent negative monomials; the stride
    /// identifies the actual native subring used by a strided encoding.
    SignedMonomial {
        exponent_stride: usize,
    },
    /// Boolean scalar polynomials, exactly one selected in each contiguous block.
    OneHotPolynomialFamily {
        block_lengths: Vec<usize>,
    },
    Bytes {
        length: IntExpr,
    },
    Family {
        count: IntExpr,
        element: Box<InputContract>,
    },
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

/// The generated module holding the gadget layouts of every root, and its context.
pub const BACKEND_MODULE: &str = "Backend";
pub const BACKEND_CONTEXT: &str = "Backend.backend";

/// Application-independent endpoint semantics, stated in the conclusion only.
#[derive(Clone, Debug)]
pub enum Endpoint {
    /// Exact equality of Boolean, integer, or family endpoints.
    Exact,
    /// Matrix endpoints within `bound` of each other (`Mxx.Primitives.Approx`).
    MatrixApprox { bound: num_bigint::BigUint },
}

/// Shared externals, acyclic graph connections and one typed endpoint.
pub struct LinkedClaim<'a> {
    pub roots: Vec<ClaimRoot<'a>>,
    pub externals: Vec<ExternalInput>,
    pub links: Vec<Link>,
    pub requirements: Vec<Port>,
    pub actual: Port,
    pub ideal: Port,
    pub endpoint: Endpoint,
    /// `None` states correctness of every execution. `Some(k)` reads every sampled coefficient
    /// from a sampling tape, stage `i` at site prefix `[i]`, and states that for every hash model
    /// and external input the tapes with a failing execution have `MxxRuntime.tapeMeasure` at
    /// most `2^-k`. The roots must then be exported with
    /// [`super::ExportOptions::sampling_tape`].
    pub failure_probability_log2: Option<u32>,
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
        (InputContract::BooleanPolynomial, ConcreteWireType::Matrix(matrix)) => {
            if !matrix.is_scalar() || matrix.ring.ring_dimension() as usize == 0 {
                return Err("Boolean polynomial contract requires a nonempty scalar matrix".into());
            }
            Ok(format!("(({value}) 0 0 = 0 ∨ ({value}) 0 0 = 1)"))
        }
        (InputContract::SignedMonomial { exponent_stride }, ConcreteWireType::Matrix(matrix)) => {
            if !matrix.is_scalar() || matrix.ring.ring_dimension() as usize == 0 {
                return Err("signed monomial contract requires a nonempty scalar matrix".into());
            }
            let n = matrix.ring.ring_dimension() as usize;
            if *exponent_stride == 0 || matrix.ring.ring_dimension() as usize % exponent_stride != 0
            {
                return Err(
                    "monomial exponent stride must be positive and divide the ring dimension"
                        .into(),
                );
            }
            let q = matrix.ring.modulus();
            Ok(format!(
                "(∃ exponent : Fin (2 * ({n} / {exponent_stride})), ({value}) 0 0 = AdjoinRoot.root (Mxx.Primitives.negacyclicModulus {n} (ZMod {q})) ^ ({exponent_stride} * exponent.val))"
            ))
        }
        (
            InputContract::OneHotPolynomialFamily { block_lengths },
            ConcreteWireType::IndexedFamily { count, element },
        ) => {
            let total = block_lengths
                .iter()
                .try_fold(0usize, |sum, length| sum.checked_add(*length))
                .ok_or("one-hot family size overflows")?;
            if total != *count || block_lengths.is_empty() || block_lengths.contains(&0) {
                return Err("one-hot family blocks do not partition its nonempty type".into());
            }
            let element_predicate = input_contract_predicate(
                &InputContract::BooleanPolynomial,
                element,
                bindings,
                &format!("({value} i)"),
                depth + 1,
            )?;
            let mut conditions = vec![format!("(∀ i : Fin {count}, {element_predicate})")];
            let mut offset = 0;
            for length in block_lengths {
                conditions.push(format!(
                    "(∃ selected : Fin {length}, ∀ i : Fin {length}, ({value} ⟨{offset} + i.val, by omega⟩) 0 0 = if i = selected then 1 else 0)"
                ));
                offset += length;
            }
            Ok(format!("({})", conditions.join(" ∧ ")))
        }
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

fn check_root(root: &ClaimRoot<'_>, bindings: &ParamEnv) -> Result<(), String> {
    let artifact = root.artifact;
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

/// Assemble a claim after checking graph identities and exact boundary wiring. Roots that use
/// gadgets read their layouts from [`BACKEND_MODULE`], which must cover every root's layouts.
pub fn assemble_claim(claim: &LinkedClaim<'_>, bindings: &ParamEnv) -> Result<String, String> {
    let mut fields = BTreeSet::new();
    for root in &claim.roots {
        if super::valid_identifier(&root.field).is_err() || !fields.insert(&root.field) {
            return Err("invalid or duplicate execution field".into());
        }
        check_root(root, bindings)?;
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
    let backend = BACKEND_CONTEXT;
    let mut source = format!("import {BACKEND_MODULE}\n");
    for (artifact, _) in &entries {
        source.push_str(&format!("import {}\n", artifact.module_name));
    }
    source.push_str("\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace GeneratedClaim\n\nstructure ExternalInputs where\n");
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
        if root.requires_sampling_tape != claim.failure_probability_log2.is_some() {
            return Err("root sampling tape does not match the failure probability".into());
        }
        let context = format!(
            "{}{}{}",
            if root.requires_backend { format!(" {backend}") } else { String::new() },
            if root.requires_hash_model { " hashModel" } else { "" },
            if root.requires_sampling_tape { format!(" tape [{position}]") } else { String::new() }
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
    let tape_binder = if claim.failure_probability_log2.is_some() {
        " (tape : MxxRuntime.SampleTape)"
    } else {
        ""
    };
    source.push_str(&format!("\ndef Runs ({hash_binder} : {}) (external : ExternalInputs){tape_binder}\n    (execution : Execution) : Prop :=\n  {}\n", "MxxRuntime.HashModel", conditions.join(" ∧\n  ")));
    // The success condition of one execution, quantified over every execution or bounded in
    // probability over sampling tapes.
    let correctness = |success: String| match claim.failure_probability_log2 {
        None => format!(
            "def CorrectnessClaim : Prop :=\n  ∀ hashModel external execution, Runs hashModel external execution →\n    {success}\n\nend GeneratedClaim\n"
        ),
        Some(bits) => format!(
            "def CorrectnessClaim : Prop :=\n  ∀ hashModel external,\n    MxxRuntime.tapeMeasure {{tape | ∃ execution, Runs hashModel external tape execution ∧\n      ¬ ({success})}} ≤ (2 : ENNReal)⁻¹ ^ {bits}\n\nend GeneratedClaim\n"
        ),
    };
    let actual = output(claim, &claim.actual)?;
    let ideal = output(claim, &claim.ideal)?;
    if actual.wire_type != ideal.wire_type || actual.lean_type != ideal.lean_type {
        return Err("endpoint type mismatch".into());
    }
    let actual_value = project(actual, &format!("execution.«{}»", entries[claim.actual.root].1));
    let ideal_value = project(ideal, &format!("execution.«{}»", entries[claim.ideal.root].1));
    let success = match &claim.endpoint {
        Endpoint::Exact => {
            let exact = |ty: &ConcreteWireType| {
                matches!(ty, ConcreteWireType::Bool | ConcreteWireType::Int)
            };
            let supported = match &actual.wire_type {
                ConcreteWireType::IndexedFamily { element, .. } => exact(element),
                ty => exact(ty),
            };
            if !supported {
                return Err("exact endpoint must be a Boolean, integer, or family of either".into());
            }
            format!("{actual_value} = {ideal_value}")
        }
        Endpoint::MatrixApprox { bound } => {
            let ConcreteWireType::Matrix(matrix) = &actual.wire_type else {
                return Err("approximation endpoint must be a matrix".into());
            };
            if matrix.ring.ring_dimension() as usize == 0 || matrix.rows == 0 || matrix.columns == 0
            {
                return Err("approximation endpoint must be nonempty".into());
            }
            format!("Mxx.Primitives.Approx ({actual_value}) ({ideal_value}) {bound}")
        }
    };
    source.push_str(match claim.failure_probability_log2 {
        None => "\n/-- Every execution's endpoint equals the ideal one. -/\n",
        Some(_) => "\n/-- Executions whose endpoint differs from the ideal one are rare. -/\n",
    });
    source.push_str(&correctness(success));
    Ok(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{IntExpr, ParamEnv};

    fn exported_graph() -> (Graph, LeanArtifact) {
        exported_graph_with(false)
    }

    fn exported_graph_with(sampling_tape: bool) -> (Graph, LeanArtifact) {
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
        let matrix =
            MatrixType { ring: crate::ring::test_ring(17, 2), rows: 1.into(), columns: 1.into() };
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
                ("bit".into(), GraphOutput { value: bit, availability: None }),
                ("residual".into(), GraphOutput { value: residual, availability: None }),
            ]),
            vec![],
            vec![],
            Default::default(),
        )
        .unwrap()
        .0;
        let validated = crate::ring::test_validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = super::super::export(
            &validated,
            &super::super::ExportOptions { sampling_tape, ..Default::default() },
        )
        .unwrap();
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
            endpoint: Endpoint::Exact,
            failure_probability_log2: None,
        }
    }

    fn render(claim: &LinkedClaim<'_>) -> Result<String, String> {
        assemble_claim(claim, &ParamEnv::default())
    }

    #[test]
    fn linked_claim_states_endpoint_equality_over_shared_externals() {
        let (graph, artifact) = exported_graph();
        let source = render(&linked(&graph, &artifact)).unwrap();
        assert!(source.contains("import Backend\n"));
        assert!(source.contains("(_ : MxxRuntime.HashModel)"));
        assert_eq!(source.matches("(external.input_0)").count(), 2);
        let (_, conclusion) = source.split_once("def CorrectnessClaim").unwrap();
        assert!(conclusion.contains(
            "Runs hashModel external execution →\n    execution.«producer».1 = execution.«ideal».1"
        ));
    }

    #[test]
    fn failure_probability_bounds_the_tape_measure_of_failing_runs() {
        let (graph, artifact) = exported_graph_with(true);
        let mut claim = linked(&graph, &artifact);
        claim.failure_probability_log2 = Some(40);
        let source = render(&claim).unwrap();
        let (runs, conclusion) = source.split_once("def CorrectnessClaim").unwrap();
        assert!(runs.contains("(external : ExternalInputs) (tape : MxxRuntime.SampleTape)"));
        assert!(runs.contains("tape [0] producer_params"));
        assert!(runs.contains("tape [1] ideal_params"));
        assert!(conclusion.contains(
            "MxxRuntime.tapeMeasure {tape | ∃ execution, Runs hashModel external tape execution ∧"
        ));
        assert!(conclusion.contains("¬ (execution.«producer».1 = execution.«ideal».1)"));
        assert!(conclusion.contains("≤ (2 : ENNReal)⁻¹ ^ 40"));
        // Tape-reading roots and a deterministic claim do not mix, in either direction.
        claim.failure_probability_log2 = None;
        assert!(render(&claim).unwrap_err().contains("sampling tape"));
        let (graph, artifact) = exported_graph();
        let mut claim = linked(&graph, &artifact);
        claim.failure_probability_log2 = Some(40);
        assert!(render(&claim).unwrap_err().contains("sampling tape"));
    }

    #[test]
    fn matrix_approximation_is_a_conclusion_on_exact_matching_endpoints() {
        let (graph, artifact) = exported_graph();
        let mut claim = linked(&graph, &artifact);
        claim.actual = Port { root: 0, name: "residual".into() };
        claim.ideal = Port { root: 1, name: "residual".into() };
        claim.endpoint = Endpoint::MatrixApprox { bound: 7u8.into() };
        let source = render(&claim).unwrap();
        let (runs, conclusion) = source.split_once("def CorrectnessClaim").unwrap();
        assert!(!runs.contains("Mxx.Primitives.Approx"));
        assert!(conclusion.contains("Mxx.Primitives.Approx"));
        assert!(conclusion.contains("execution.«producer»"));
        assert!(conclusion.contains("execution.«ideal»"));
        assert!(conclusion.contains(") 7"));
        claim.ideal.name = "bit".into();
        assert!(render(&claim).unwrap_err().contains("type mismatch"));
    }

    #[test]
    fn exact_endpoints_are_boolean_integer_or_family_values() {
        let (graph, artifact) = exported_graph();
        let mut claim = linked(&graph, &artifact);
        claim.actual = Port { root: 0, name: "residual".into() };
        assert!(render(&claim).unwrap_err().contains("type mismatch"));
        claim.ideal = Port { root: 1, name: "residual".into() };
        assert!(render(&claim).unwrap_err().contains("exact endpoint"));
    }

    #[test]
    fn polynomial_input_contracts_restrict_values_and_check_scalar_shapes() {
        let (_, artifact) = exported_graph();
        let ty = &artifact.root.outputs["residual"].wire_type;
        let env = ParamEnv::default();
        let boolean =
            input_contract_predicate(&InputContract::BooleanPolynomial, ty, &env, "input", 0)
                .unwrap();
        assert_eq!(boolean, "((input) 0 0 = 0 ∨ (input) 0 0 = 1)");
        let monomial = input_contract_predicate(
            &InputContract::SignedMonomial { exponent_stride: 1 },
            ty,
            &env,
            "input",
            0,
        )
        .unwrap();
        assert!(monomial.contains("Fin (2 * (2 / 1))"));
        assert!(monomial.contains("negacyclicModulus 2 (ZMod 17)"));
        let strided = input_contract_predicate(
            &InputContract::SignedMonomial { exponent_stride: 2 },
            ty,
            &env,
            "input",
            0,
        )
        .unwrap();
        assert!(strided.contains("^ (2 * exponent.val)"));
        for stride in [0, 3] {
            assert!(
                input_contract_predicate(
                    &InputContract::SignedMonomial { exponent_stride: stride },
                    ty,
                    &env,
                    "input",
                    0
                )
                .is_err()
            );
        }
        let ConcreteWireType::Matrix(mut nonscalar) = ty.clone() else { unreachable!() };
        nonscalar.columns = 2;
        for contract in
            [InputContract::BooleanPolynomial, InputContract::SignedMonomial { exponent_stride: 1 }]
        {
            assert!(
                input_contract_predicate(&contract, &ConcreteWireType::Bool, &env, "input", 0)
                    .is_err()
            );
            assert!(
                input_contract_predicate(
                    &contract,
                    &ConcreteWireType::Matrix(nonscalar.clone()),
                    &env,
                    "input",
                    0
                )
                .is_err()
            );
        }
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
        let source = render(&claim).unwrap();
        assert!(source.contains("  «match» :"));
        assert!(source.contains("  «namespace» :"));
        assert!(source.contains("namespace_params (execution.«match».1) execution.«namespace»"));
        assert!(source.contains("execution.«match».1 = true"));
        assert!(source.contains("execution.«match».1 = execution.«namespace».1"));
        assert!(!source.contains("execution.match"));
        assert!(!source.contains("execution.namespace"));
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/lean_ir_fixtures/claim_identifiers");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join(format!("{}.lean", artifact.module_name)), &artifact.source)
            .unwrap();
        std::fs::write(
            directory.join(format!("{BACKEND_MODULE}.lean")),
            super::super::backend::render_backend(&[]),
        )
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
        claim.links[0].producer = Port { root: 0, name: "residual".into() };
        assert!(render(&claim).unwrap_err().contains("type mismatch"));
        claim.links[0].producer = claim.ideal.clone();
        assert!(render(&claim).unwrap_err().contains("precede"));
        claim.links.clear();
        claim.externals[0].destinations.push(Port { root: 1, name: "bit".into() });
        claim.requirements.push(Port { root: 0, name: "residual".into() });
        assert!(render(&claim).unwrap_err().contains("not Boolean"));
        claim.requirements.clear();
        claim.actual = Port { root: 0, name: "residual".into() };
        assert!(render(&claim).unwrap_err().contains("type mismatch"));
    }

    #[test]
    fn linked_claim_rejects_mismatched_graph_and_boundary_metadata() {
        let (graph, artifact) = exported_graph();
        let mut wrong = artifact.clone();
        wrong.spec_hash.0[0] ^= 1;
        assert!(render(&linked(&graph, &wrong)).unwrap_err().contains("bindings mismatch"));
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
