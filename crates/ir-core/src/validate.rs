use crate::{
    artifact::{ArtifactType, Manifest, ManifestArtifact, ProductionId, validate_manifest},
    checks::{
        CheckError, ElaborationWarning, WarningKind, check_add_shape, check_topological,
        multiplication_type,
    },
    expr::{ExprError, IndexExpr, IndexMap, IntExpr, ParamEnv},
    graph::{CompileParameterKind, FrozenGraphScopeId, Graph, GraphScope, NodeHandle},
    node::{ConcatAxis, ConstantMatrix, GridInputMode, IntBinaryOp, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType, MatrixType, NodeId, Port, WireRef, WireType},
};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LivenessSchedule {
    pub last_use: BTreeMap<WireRef, usize>,
    pub retained: BTreeSet<WireRef>,
}

#[derive(Clone, Debug)]
pub struct ValidatedScope {
    pub execution_order: Vec<NodeHandle>,
    pub liveness: LivenessSchedule,
    pub wire_types: BTreeMap<WireRef, ConcreteWireType>,
    pub artifact_inputs: BTreeMap<WireRef, ManifestArtifact>,
}

#[derive(Clone, Debug)]
pub struct ValidatedGraph {
    pub source: Graph,
    pub bindings: ParamEnv,
    pub scopes: BTreeMap<FrozenGraphScopeId, ValidatedScope>,
    pub warnings: Vec<ElaborationWarning>,
}

impl ValidatedGraph {
    pub fn scope(&self, id: &FrozenGraphScopeId) -> Option<&ValidatedScope> {
        self.scopes.get(id)
    }

    pub fn root_scope(&self) -> &ValidatedScope {
        self.scope(&FrozenGraphScopeId::Root).expect("validated graph has a root scope")
    }
}

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error(transparent)]
    Expression(#[from] ExprError),
    #[error(transparent)]
    Check(#[from] CheckError),
    #[error("missing compile binding: {0}")]
    MissingBinding(String),
    #[error("scope {scope:?}, node {node:?}: unavailable wire {wire:?}")]
    MissingWire { scope: FrozenGraphScopeId, node: NodeId, wire: WireRef },
    #[error("graph scope is unavailable: {scope:?}")]
    MissingScope { scope: FrozenGraphScopeId },
    #[error("scope {scope:?}, node {node:?}: {message}")]
    Node { scope: FrozenGraphScopeId, node: NodeId, message: String },
    #[error("artifact production is unavailable: {0:?}")]
    MissingManifest(ProductionId),
    #[error("artifact manifest uses IR version {actual}, expected {expected}")]
    ManifestVersion { expected: u32, actual: u32 },
    #[error("artifact manifest key {key:?} does not match production id {manifest:?}")]
    ManifestProductionMismatch { key: ProductionId, manifest: ProductionId },
    #[error("artifact manifest {production:?} is invalid: {message}")]
    InvalidManifest { production: ProductionId, message: String },
    #[error("artifact {artifact} is unavailable in production {production:?}")]
    MissingArtifact { production: ProductionId, artifact: String },
    #[error("undeclared compile variable: {0}")]
    UndeclaredCompileVariable(String),
    #[error("loop-index binder {slot} is not available in scope {scope:?}")]
    IllegalLoopIndex { scope: FrozenGraphScopeId, slot: u32 },
    #[error("loop-index binder {slot} occurs in a structural type in scope {scope:?}")]
    StructuralLoopIndex { scope: FrozenGraphScopeId, slot: u32 },
    #[error("parameter constraint failed: {0}")]
    ParameterConstraint(String),
}

pub fn validate_structure(graph: &Graph) -> Result<(), ValidationError> {
    check_topological(graph)?;
    let mut root_declared =
        graph.parameters().iter().map(|parameter| parameter.name.clone()).collect::<BTreeSet<_>>();
    root_declared.extend(graph.real_constants().keys().cloned());
    let mut declared_by_scope = graph
        .scopes()
        .keys()
        .cloned()
        .map(|scope| (scope, BTreeSet::new()))
        .collect::<BTreeMap<_, _>>();
    declared_by_scope.insert(FrozenGraphScopeId::Root, root_declared);
    loop {
        let mut changed = false;
        for (scope_id, scope) in graph.scopes() {
            let parent_declared = declared_by_scope.get(scope_id).cloned().unwrap_or_default();
            for (position, node) in scope.nodes().iter().enumerate() {
                let node_id = NodeId(position as u64);
                if let NodeKind::SubgraphCall(call) = node.kind() {
                    if call.canonical_input_exclusive_uppers.len() != node.arguments().len() {
                        return Err(ValidationError::Node {
                            scope: scope_id.clone(),
                            node: node_id,
                            message: "canonical input exclusive upper bound count does not match call arguments"
                                .to_owned(),
                        });
                    }
                    if call
                        .canonical_input_exclusive_uppers
                        .iter()
                        .any(|upper| upper.as_ref().is_some_and(|upper| upper.is_zero()))
                    {
                        return Err(ValidationError::Node {
                            scope: scope_id.clone(),
                            node: node_id,
                            message: "canonical input exclusive upper bounds must be positive"
                                .to_owned(),
                        });
                    }
                    if call.canonical_input_exclusive_uppers.iter().zip(node.arguments()).any(
                        |(upper, argument)| {
                            upper.is_some() && !matches!(argument.wire_type(), WireType::Matrix(_))
                        },
                    ) {
                        return Err(ValidationError::Node {
                            scope: scope_id.clone(),
                            node: node_id,
                            message: "canonical input exclusive upper bounds require matrix call arguments"
                                .to_owned(),
                        });
                    }
                }
                let Some(child) = graph.child_scope_id(scope_id, node_id) else {
                    continue;
                };
                let binding_names = match node.kind() {
                    NodeKind::SubgraphCall(call) => &call.bindings,
                    NodeKind::SequentialLoop(loop_spec) => &loop_spec.bindings,
                    NodeKind::ParallelGrid(grid_spec) => &grid_spec.bindings,
                    _ => continue,
                };
                let target = declared_by_scope.entry(child.clone()).or_default();
                let before = target.len();
                target.extend(parent_declared.iter().cloned());
                target.extend(binding_names.iter().map(|(name, _)| name.clone()));
                changed |= target.len() != before;
            }
        }
        if !changed {
            break;
        }
    }

    let empty_dependencies = graph
        .scopes()
        .keys()
        .cloned()
        .map(|scope| (scope, BTreeSet::<String>::new()))
        .collect::<BTreeMap<_, _>>();
    let mut loop_dependent_by_scope = empty_dependencies.clone();
    loop {
        let mut next = empty_dependencies.clone();
        for (scope_id, scope) in graph.scopes() {
            let parent = loop_dependent_by_scope.get(scope_id).cloned().unwrap_or_default();
            for (position, node) in scope.nodes().iter().enumerate() {
                let Some(child) = graph.child_scope_id(scope_id, NodeId(position as u64)) else {
                    continue;
                };
                let bindings = match node.kind() {
                    NodeKind::SubgraphCall(call) => &call.bindings,
                    NodeKind::SequentialLoop(loop_spec) => &loop_spec.bindings,
                    NodeKind::ParallelGrid(grid_spec) => &grid_spec.bindings,
                    _ => continue,
                };
                next.entry(child).or_default().extend(child_loop_dependencies(&parent, bindings));
            }
        }
        if next == loop_dependent_by_scope {
            break;
        }
        loop_dependent_by_scope = next;
    }

    for (scope_id, scope) in graph.scopes() {
        let allowed_slots = structural_loop_slots(graph, scope_id);
        for (position, node) in scope.nodes().iter().enumerate() {
            let kind = serde_json::to_value(node.kind()).expect("NodeKind is serializable");
            let mut variables = BTreeSet::new();
            let mut loop_slots = BTreeSet::new();
            collect_expression_references(&kind, &mut variables, &mut loop_slots);
            let declared = &declared_by_scope[scope_id];
            if let Some(name) = variables.iter().find(|name| !declared.contains(*name)) {
                return Err(ValidationError::UndeclaredCompileVariable(name.clone()));
            }
            let mut node_allowed_slots = allowed_slots.clone();
            match node.kind() {
                NodeKind::SequentialLoop(loop_spec) => {
                    let value =
                        serde_json::to_value(&loop_spec.count).expect("IntExpr is serializable");
                    let mut ignored = BTreeSet::new();
                    let mut count_slots = BTreeSet::new();
                    collect_expression_references(&value, &mut ignored, &mut count_slots);
                    if count_slots.contains(&loop_spec.index_slot) {
                        return Err(ValidationError::IllegalLoopIndex {
                            scope: scope_id.clone(),
                            slot: loop_spec.index_slot,
                        });
                    }
                    node_allowed_slots.insert(loop_spec.index_slot);
                }
                NodeKind::ParallelGrid(grid_spec) => {
                    node_allowed_slots.extend(grid_spec.index_slots.iter().copied());
                }
                _ => {}
            }
            if let Some(slot) =
                loop_slots.iter().copied().find(|slot| !node_allowed_slots.contains(slot))
            {
                return Err(ValidationError::IllegalLoopIndex { scope: scope_id.clone(), slot });
            }
            if matches!(node.kind(), NodeKind::FamilyGetStatic { .. }) && !loop_slots.is_empty() {
                return node_error(
                    scope_id,
                    NodeId(position as u64),
                    "loop-dependent family access must use FamilyGetDynamic",
                );
            }
            if matches!(node.kind(), NodeKind::FamilyGetStatic { .. }) &&
                variables.iter().any(|name| loop_dependent_by_scope[scope_id].contains(name))
            {
                return node_error(
                    scope_id,
                    NodeId(position as u64),
                    "loop-dependent family access must use FamilyGetDynamic",
                );
            }
            if let NodeKind::FamilyGetStatic { indices } = node.kind() {
                if indices.iter().any(index_expr_has_loop_or_parameter) {
                    return node_error(
                        scope_id,
                        NodeId(position as u64),
                        "loop-dependent family access must use FamilyGetDynamic",
                    );
                }
            }
            for wire_type in node.output_types() {
                let value = serde_json::to_value(wire_type).expect("WireType is serializable");
                let mut ignored = BTreeSet::new();
                let mut structural_slots = BTreeSet::new();
                collect_expression_references(&value, &mut ignored, &mut structural_slots);
                if let Some(slot) = structural_slots.into_iter().next() {
                    return Err(ValidationError::StructuralLoopIndex {
                        scope: scope_id.clone(),
                        slot,
                    });
                }
            }
        }
    }
    Ok(())
}

fn structural_loop_slots(graph: &Graph, scope: &FrozenGraphScopeId) -> BTreeSet<u32> {
    let mut slots = BTreeSet::new();
    let mut current = scope;
    loop {
        let (parent, owner) = match current {
            FrozenGraphScopeId::ParallelBody { parent, owner } |
            FrozenGraphScopeId::SequentialBody { parent, owner } => (parent, owner),
            _ => break,
        };
        match graph.scope(parent).and_then(|scope| scope.node(*owner)).map(NodeHandle::kind) {
            Some(NodeKind::SequentialLoop(loop_spec)) => {
                slots.insert(loop_spec.index_slot);
            }
            Some(NodeKind::ParallelGrid(grid_spec)) => {
                slots.extend(grid_spec.index_slots.iter().copied());
            }
            _ => {}
        }
        current = parent;
    }
    slots
}

fn sequential_loop_slots(graph: &Graph, scope: &FrozenGraphScopeId) -> BTreeSet<u32> {
    let mut slots = BTreeSet::new();
    let mut current = scope;
    loop {
        let (parent, owner) = match current {
            FrozenGraphScopeId::SequentialBody { parent, owner } => (parent, owner),
            FrozenGraphScopeId::ParallelBody { parent, owner } => {
                current = parent;
                let _ = owner;
                continue;
            }
            _ => break,
        };
        if let Some(NodeKind::SequentialLoop(spec)) =
            graph.scope(parent).and_then(|scope| scope.node(*owner)).map(NodeHandle::kind)
        {
            slots.insert(spec.index_slot);
        }
        current = parent;
    }
    slots
}

fn collect_expression_references(
    value: &serde_json::Value,
    variables: &mut BTreeSet<String>,
    loop_slots: &mut BTreeSet<u32>,
) {
    match value {
        serde_json::Value::Object(fields) => {
            match (fields.get("tag").and_then(serde_json::Value::as_str), fields.get("value")) {
                (Some("Var"), Some(serde_json::Value::String(name))) => {
                    variables.insert(name.clone());
                }
                (Some("LoopIndex"), Some(slot)) => {
                    if let Some(slot) = slot.as_u64().and_then(|slot| u32::try_from(slot).ok()) {
                        loop_slots.insert(slot);
                    }
                }
                _ => {}
            }
            for child in fields.values() {
                collect_expression_references(child, variables, loop_slots);
            }
        }
        serde_json::Value::Array(values) => {
            for child in values {
                collect_expression_references(child, variables, loop_slots);
            }
        }
        _ => {}
    }
}

fn child_loop_dependencies(
    parent: &BTreeSet<String>,
    bindings: &[(String, IntExpr)],
) -> BTreeSet<String> {
    let binding_names = bindings.iter().map(|(name, _)| name).collect::<BTreeSet<_>>();
    let mut child = parent
        .iter()
        .filter(|name| !binding_names.contains(name))
        .cloned()
        .collect::<BTreeSet<_>>();
    for (name, expression) in bindings {
        let value = serde_json::to_value(expression).expect("IntExpr is serializable");
        let mut variables = BTreeSet::new();
        let mut loop_slots = BTreeSet::new();
        collect_expression_references(&value, &mut variables, &mut loop_slots);
        if !loop_slots.is_empty() || variables.iter().any(|variable| parent.contains(variable)) {
            child.insert(name.clone());
        }
    }
    child
}

pub fn validate(graph: &Graph, bindings: &ParamEnv) -> Result<ValidatedGraph, ValidationError> {
    validate_with_manifests(graph, bindings, &BTreeMap::new())
}

pub fn validate_with_manifests(
    graph: &Graph,
    bindings: &ParamEnv,
    manifests: &BTreeMap<ProductionId, Manifest>,
) -> Result<ValidatedGraph, ValidationError> {
    validate_structure(graph)?;
    check_manifests(manifests)?;
    check_bindings(graph, bindings)?;
    check_topological(graph)?;
    // This is the single source for parameter-only validity used by operational checking.
    // The remaining validation derives concrete types, checks wire flow and
    // shapes, and constructs execution/liveness data.
    crate::constraints::evaluate_param_constraints(graph, bindings)?;

    let mut warnings = Vec::new();
    let mut scopes = BTreeMap::new();
    let mut scope_bindings = BTreeMap::new();
    collect_scope_bindings(
        graph,
        &FrozenGraphScopeId::Root,
        bindings.clone(),
        &mut scope_bindings,
    )?;
    for (scope_id, scope) in graph.scopes() {
        let scope_env = scope_bindings
            .get(scope_id)
            .ok_or_else(|| ValidationError::MissingScope { scope: scope_id.clone() })?;
        let validated =
            validate_scope(graph, scope_id, scope, scope_env, manifests, &mut warnings)?;
        scopes.insert(scope_id.clone(), validated);
    }
    validate_structural_boundaries(graph, bindings, &scopes)?;
    Ok(ValidatedGraph { source: graph.clone(), bindings: bindings.clone(), scopes, warnings })
}

fn collect_scope_bindings(
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    env: ParamEnv,
    output: &mut BTreeMap<FrozenGraphScopeId, ParamEnv>,
) -> Result<(), ValidationError> {
    output.insert(scope_id.clone(), env.clone());
    let scope = graph
        .scope(scope_id)
        .ok_or_else(|| ValidationError::MissingScope { scope: scope_id.clone() })?;
    for (position, node) in scope.nodes().iter().enumerate() {
        let node_id = NodeId(position as u64);
        let Some(child_id) = graph.child_scope_id(scope_id, node_id) else {
            continue;
        };
        let child_env = match node.kind() {
            NodeKind::SubgraphCall(call) => child_bindings(&env, &call.bindings)?,
            NodeKind::SequentialLoop(loop_spec) => {
                let mut loop_env = env.clone();
                loop_env.loop_indices.insert(loop_spec.index_slot, BigInt::zero());
                child_bindings(&loop_env, &loop_spec.bindings)?
            }
            NodeKind::ParallelGrid(grid_spec) => {
                let mut grid_env = env.clone();
                for slot in &grid_spec.index_slots {
                    grid_env.loop_indices.insert(*slot, BigInt::zero());
                }
                child_bindings(&grid_env, &grid_spec.bindings)?
            }
            _ => continue,
        };
        collect_scope_bindings(graph, &child_id, child_env, output)?;
    }
    Ok(())
}

fn check_manifests(manifests: &BTreeMap<ProductionId, Manifest>) -> Result<(), ValidationError> {
    for (production, manifest) in manifests {
        if &manifest.production_id != production {
            return Err(ValidationError::ManifestProductionMismatch {
                key: production.clone(),
                manifest: manifest.production_id.clone(),
            });
        }
        validate_manifest(manifest).map_err(|error| ValidationError::InvalidManifest {
            production: production.clone(),
            message: error.to_string(),
        })?;
    }
    Ok(())
}

fn validate_scope(
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    scope: &GraphScope,
    bindings: &ParamEnv,
    manifests: &BTreeMap<ProductionId, Manifest>,
    warnings: &mut Vec<ElaborationWarning>,
) -> Result<ValidatedScope, ValidationError> {
    let mut wire_types = BTreeMap::new();
    let mut artifact_inputs = BTreeMap::new();
    for (position, handle) in scope.nodes().iter().enumerate() {
        let node = NodeView {
            id: NodeId(position as u64),
            kind: handle.kind(),
            args: scope.arguments(handle).expect("frozen arguments are local"),
            output_types: handle.output_types(),
        };
        validate_node(
            graph,
            scope_id,
            &node,
            bindings,
            manifests,
            &mut wire_types,
            &mut artifact_inputs,
            warnings,
        )?;
    }
    let retained = if *scope_id == FrozenGraphScopeId::Root {
        BTreeSet::new()
    } else {
        scope.outputs().iter().copied().collect()
    };
    let liveness = liveness(scope, retained);
    Ok(ValidatedScope {
        execution_order: scope.nodes().to_vec(),
        liveness,
        wire_types,
        artifact_inputs,
    })
}

struct NodeView<'a> {
    id: NodeId,
    kind: &'a NodeKind,
    args: Vec<WireRef>,
    output_types: &'a [WireType],
}

fn validate_node(
    graph: &Graph,
    scope: &FrozenGraphScopeId,
    node: &NodeView<'_>,
    env: &ParamEnv,
    manifests: &BTreeMap<ProductionId, Manifest>,
    values: &mut BTreeMap<WireRef, ConcreteWireType>,
    artifact_inputs: &mut BTreeMap<WireRef, ManifestArtifact>,
    warnings: &mut Vec<ElaborationWarning>,
) -> Result<(), ValidationError> {
    let sequential_slots = sequential_loop_slots(graph, scope);
    let inferred = match node.kind {
        NodeKind::Input { wire_type, artifact, .. } => {
            require_arity(scope, node, 0)?;
            let declared = concrete_wire(wire_type, env, scope, node.id)?;
            if let Some(artifact) = artifact {
                let manifest = manifests.get(&artifact.production_id).ok_or_else(|| {
                    ValidationError::MissingManifest(artifact.production_id.clone())
                })?;
                if manifest.ir_version != crate::encoding::IR_VERSION {
                    return Err(ValidationError::ManifestVersion {
                        expected: crate::encoding::IR_VERSION,
                        actual: manifest.ir_version,
                    });
                }
                let stored = manifest.artifacts.get(&artifact.artifact_name).ok_or_else(|| {
                    ValidationError::MissingArtifact {
                        production: artifact.production_id.clone(),
                        artifact: artifact.artifact_name.clone(),
                    }
                })?;
                if artifact.confidentiality != stored.confidentiality {
                    return node_error(
                        scope,
                        node.id,
                        "artifact confidentiality does not match manifest",
                    );
                }
                let (element, shape) = family_element_shape(&declared);
                if ArtifactType::from_wire_type(element).as_ref() != Some(&stored.artifact_type) ||
                    shape != stored.family_shape.as_deref()
                {
                    return node_error(scope, node.id, "artifact type does not match manifest");
                }
                artifact_inputs.insert(WireRef { node: node.id, port: Port(0) }, stored.clone());
            }
            vec![declared]
        }
        NodeKind::ConstantInt(_) | NodeKind::EvaluateInt(_) => {
            require_arity(scope, node, 0)?;
            if let NodeKind::EvaluateInt(value) = node.kind {
                value.evaluate(env)?;
            }
            vec![ConcreteWireType::ConstantInt]
        }
        NodeKind::ConstantReal(value) => {
            require_arity(scope, node, 0)?;
            value.evaluate_f64(env)?;
            vec![ConcreteWireType::ConstantReal]
        }
        NodeKind::ConstantBool(_) => {
            require_arity(scope, node, 0)?;
            vec![ConcreteWireType::ConstantBool]
        }
        NodeKind::ConstantMatrix { matrix_type, value } => {
            require_arity(scope, node, 0)?;
            let matrix = concrete_matrix(matrix_type, env, scope, node.id)?;
            validate_constant(value, &matrix, env, scope, node.id)?;
            vec![ConcreteWireType::Matrix(matrix)]
        }
        NodeKind::GadgetTrapdoor { matrix_type, base } => {
            require_arity(scope, node, 0)?;
            let matrix = concrete_matrix(matrix_type, env, scope, node.id)?;
            let gadget_base = base.evaluate(env)?;
            if gadget_base <= BigInt::one() ||
                matrix.rows == 0 ||
                !matrix.columns.is_multiple_of(matrix.rows)
            {
                return node_error(scope, node.id, "invalid gadget trapdoor dimensions or base");
            }
            let digit_count = matrix.columns / matrix.rows;
            vec![ConcreteWireType::Trapdoor {
                matrix,
                sigma: crate::RealExpr::FromInt(IntExpr::constant(gadget_base.clone())),
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound: BigInt::zero(),
            }]
        }
        NodeKind::TrapdoorPublic => {
            require_arity(scope, node, 1)?;
            let ConcreteWireType::Trapdoor { matrix, .. } = argument(scope, values, node, 0)?
            else {
                return node_error(scope, node.id, "trapdoor public projection requires a trapdoor");
            };
            vec![ConcreteWireType::Matrix(matrix.clone())]
        }
        NodeKind::IntBinary(operation) => {
            require_arity(scope, node, 2)?;
            require_scalar(scope, values, node, 0, is_integer, "integer")?;
            require_scalar(scope, values, node, 1, is_integer, "integer")?;
            let _runtime_zero_check =
                matches!(operation, IntBinaryOp::Divide | IntBinaryOp::Remainder);
            vec![ConcreteWireType::Int]
        }
        NodeKind::IntCompare(_) => {
            require_arity(scope, node, 2)?;
            require_scalar(scope, values, node, 0, is_integer, "integer")?;
            require_scalar(scope, values, node, 1, is_integer, "integer")?;
            vec![ConcreteWireType::Bool]
        }
        NodeKind::BitExtract { bit } => {
            require_arity(scope, node, 1)?;
            require_scalar(scope, values, node, 0, is_integer, "integer")?;
            if bit.evaluate(env)?.is_negative() {
                return node_error(scope, node.id, "bit position must be nonnegative");
            }
            vec![ConcreteWireType::Bool]
        }
        NodeKind::IntToReal => {
            require_arity(scope, node, 1)?;
            require_scalar(scope, values, node, 0, is_integer, "integer")?;
            vec![ConcreteWireType::Real]
        }
        NodeKind::BoolToInt => {
            require_arity(scope, node, 1)?;
            require_scalar(scope, values, node, 0, is_boolean, "boolean")?;
            vec![ConcreteWireType::Int]
        }
        NodeKind::RealBinary(_) => {
            require_arity(scope, node, 2)?;
            require_scalar(scope, values, node, 0, is_real, "real")?;
            require_scalar(scope, values, node, 1, is_real, "real")?;
            vec![ConcreteWireType::Real]
        }
        NodeKind::RealSqrt => {
            require_arity(scope, node, 1)?;
            require_scalar(scope, values, node, 0, is_real, "real")?;
            vec![ConcreteWireType::Real]
        }
        NodeKind::MatrixBinary(operation) => {
            require_arity(scope, node, 2)?;
            let left = matrix_argument(scope, values, node, 0)?;
            let right = matrix_argument(scope, values, node, 1)?;
            let output = match operation {
                MatrixBinaryOp::Add | MatrixBinaryOp::Subtract => {
                    check_add_shape(&left, &right)?;
                    left
                }
                MatrixBinaryOp::Multiply => multiplication_type(&left, &right)?,
            };
            vec![ConcreteWireType::Matrix(output)]
        }
        NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
            if coefficients.is_empty() {
                return Err(ValidationError::Node {
                    scope: scope.to_owned(),
                    node: node.id,
                    message: "multi-row GEMM requires at least one product".to_owned(),
                });
            }
            require_arity(scope, node, coefficients.len() * 2 + usize::from(*has_bias))?;
            let mut output = None;
            for (product, coefficient) in coefficients.iter().enumerate() {
                coefficient.evaluate(env)?;
                let left = matrix_argument(scope, values, node, 2 * product)?;
                let right = matrix_argument(scope, values, node, 2 * product + 1)?;
                let product_type = multiplication_type(&left, &right)?;
                if let Some(expected) = &output {
                    check_add_shape(expected, &product_type)?;
                } else {
                    output = Some(product_type);
                }
            }
            let output = output.expect("nonempty coefficients");
            if *has_bias {
                let bias = matrix_argument(scope, values, node, coefficients.len() * 2)?;
                check_add_shape(&output, &bias)?;
            }
            vec![ConcreteWireType::Matrix(output)]
        }
        NodeKind::MatrixNegate | NodeKind::MatrixScale { .. } => {
            require_arity(scope, node, 1)?;
            if let NodeKind::MatrixScale { scalar } = node.kind {
                scalar.evaluate(env)?;
            }
            vec![ConcreteWireType::Matrix(matrix_argument(scope, values, node, 0)?)]
        }
        NodeKind::RingAutomorphism { index } => {
            require_arity(scope, node, 1)?;
            let input = matrix_argument(scope, values, node, 0)?;
            if !input.ring_dimension.is_power_of_two() {
                return node_error(
                    scope,
                    node.id,
                    "ring automorphism requires a power-of-two ring dimension",
                );
            }
            let index = index.evaluate(env)?;
            let upper = BigInt::from(input.ring_dimension) * BigInt::from(2_u8);
            if index <= BigInt::from(0_u8) || index >= upper || (&index % 2_u8).is_zero() {
                return node_error(
                    scope,
                    node.id,
                    "ring automorphism index must be odd and lie in 1..2*ring_dimension",
                );
            }
            vec![ConcreteWireType::Matrix(input)]
        }
        NodeKind::Transpose => {
            require_arity(scope, node, 1)?;
            let input = matrix_argument(scope, values, node, 0)?;
            vec![ConcreteWireType::Matrix(ConcreteMatrixType {
                rows: input.columns,
                columns: input.rows,
                ..input
            })]
        }
        NodeKind::Slice { rows, columns } => {
            require_arity(scope, node, 1)?;
            let input = matrix_argument(scope, values, node, 0)?;
            vec![ConcreteWireType::Matrix(sliced_type(
                &input,
                rows.as_ref(),
                columns.as_ref(),
                env,
                scope,
                node.id,
            )?)]
        }
        NodeKind::Tensor => {
            require_arity(scope, node, 2)?;
            let left = matrix_argument(scope, values, node, 0)?;
            let right = matrix_argument(scope, values, node, 1)?;
            crate::checks::check_same_ring(&left, &right)?;
            vec![ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: left.modulus,
                ring_dimension: left.ring_dimension,
                rows: left.rows.saturating_mul(right.rows),
                columns: left.columns.saturating_mul(right.columns),
            })]
        }
        NodeKind::Concat { axis } => {
            let inputs = (0..node.args.len())
                .map(|index| matrix_argument(scope, values, node, index))
                .collect::<Result<Vec<_>, _>>()?;
            vec![ConcreteWireType::Matrix(concat_type(&inputs, *axis, scope, node.id)?)]
        }
        NodeKind::UniformResidueSample { matrix_type } => {
            require_arity(scope, node, 0)?;
            vec![ConcreteWireType::Matrix(concrete_matrix(matrix_type, env, scope, node.id)?)]
        }
        NodeKind::UniformIntervalSample { matrix_type, range } => {
            require_arity(scope, node, 0)?;
            let minimum = range.minimum.evaluate(env)?;
            let maximum = range.maximum.evaluate(env)?;
            if minimum > maximum {
                return node_error(scope, node.id, "uniform sample range is empty");
            }
            let matrix_type = concrete_matrix(matrix_type, env, scope, node.id)?;
            let is_ternary = minimum == BigInt::from(-1) && maximum == BigInt::from(1);
            let is_bit = minimum == BigInt::from(0) && maximum == BigInt::from(1);
            if !is_ternary && !is_bit {
                return node_error(
                    scope,
                    node.id,
                    "uniform interval must be [-1, 1] or [0, 1]; use uniform_residue for R_q",
                );
            }
            vec![ConcreteWireType::Matrix(matrix_type)]
        }
        NodeKind::GaussianSample { matrix_type, sigma, max_coefficient_bound } => {
            require_arity(scope, node, 0)?;
            require_nonnegative_real(sigma.evaluate_f64(env)?, scope, node.id, "Gaussian sigma")?;
            let max_coefficient_bound = max_coefficient_bound.evaluate(env)?;
            if max_coefficient_bound.is_negative() {
                return node_error(scope, node.id, "Gaussian coefficient bound must be nonnegative");
            }
            vec![ConcreteWireType::Matrix(concrete_matrix(matrix_type, env, scope, node.id)?)]
        }
        NodeKind::HashSample {
            matrix_type,
            tag_expressions,
            tag_decimal_expressions,
            tag_u64_le_expressions,
            ..
        } => {
            if argument(scope, values, node, 0)? != &(ConcreteWireType::Bytes { length: 32 }) {
                return node_error(scope, node.id, "hash sampling requires a 32-byte key");
            }
            for index in 1..node.args.len() {
                require_scalar(scope, values, node, index, is_integer, "integer")?;
            }
            for expression in tag_expressions.iter().chain(tag_decimal_expressions) {
                expression.evaluate(env)?;
            }
            for expression in tag_u64_le_expressions {
                if expression.evaluate(env)?.to_u64().is_none() {
                    return node_error(scope, node.id, "little-endian hash tag must fit in u64");
                }
            }
            vec![ConcreteWireType::Matrix(concrete_matrix(matrix_type, env, scope, node.id)?)]
        }
        NodeKind::TrapdoorSample {
            matrix_type,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => {
            require_arity(scope, node, 0)?;
            let sigma = sigma.close(env)?;
            require_positive_real(
                sigma.evaluate_f64(&ParamEnv::default())?,
                scope,
                node.id,
                "trapdoor sigma",
            )?;
            let gadget_base = gadget_base.evaluate(env)?.abs();
            if gadget_base <= BigInt::one() {
                return node_error(scope, node.id, "gadget base must be greater than one");
            }
            let digit_count =
                positive_usize(digit_count.evaluate(env)?, "trapdoor digit count", scope, node.id)?;
            let preimage_max_coefficient_bound = preimage_max_coefficient_bound.evaluate(env)?;
            if preimage_max_coefficient_bound.is_negative() {
                return node_error(scope, node.id, "preimage coefficient bound must be nonnegative");
            }
            let matrix = concrete_matrix(matrix_type, env, scope, node.id)?;
            let expected_columns = matrix
                .rows
                .checked_mul(digit_count.saturating_add(2))
                .ok_or_else(|| ValidationError::Node {
                    scope: scope.clone(),
                    node: node.id,
                    message: "trapdoor width overflow".to_owned(),
                })?;
            if matrix.columns != expected_columns {
                return node_error(
                    scope,
                    node.id,
                    "trapdoor columns must equal rows * (digit_count + 2)",
                );
            }
            vec![
                ConcreteWireType::Matrix(matrix.clone()),
                ConcreteWireType::Trapdoor {
                    matrix,
                    sigma,
                    gadget_base,
                    digit_count,
                    preimage_max_coefficient_bound,
                },
            ]
        }
        NodeKind::PreimageSample { matrix_type, max_coefficient_bound } => {
            // The three arguments instantiate the typed equation B*K=T: the public matrix B,
            // trapdoor metadata for B, and target T determine the witness shape checked below.
            require_arity(scope, node, 3)?;
            if max_coefficient_bound.evaluate(env)?.is_negative() {
                return node_error(scope, node.id, "preimage coefficient bound must be nonnegative");
            }
            let public = matrix_argument(scope, values, node, 0)?;
            let trapdoor = trapdoor_argument(scope, values, node, 1)?;
            if public != trapdoor {
                return node_error(
                    scope,
                    node.id,
                    "preimage public matrix does not match its trapdoor type",
                );
            }
            let target = matrix_argument(scope, values, node, 2)?;
            let output = concrete_matrix(matrix_type, env, scope, node.id)?;
            // Matrix multiplication fixes the witness dimensions, and equality of the resulting
            // shape with T is the structural part of the relation B*K=T.
            check_add_shape(&multiplication_type(&trapdoor, &output)?, &target)?;
            vec![ConcreteWireType::Preimage(output)]
        }
        NodeKind::ApplyPreimage => {
            // ApplyPreimage is the relation-consuming edge: numerically it computes A*K, but its
            // typed right input is what authorizes consuming the equation B*K=T. A plain
            // MatrixBinary::Multiply follows the same arithmetic without this authorization.
            require_arity(scope, node, 2)?;
            let left = matrix_argument(scope, values, node, 0)?;
            let right = matrix_argument(scope, values, node, 1)?;
            // The output shape is the usual product shape for A*K; the relation marker is gone.
            vec![ConcreteWireType::Matrix(multiplication_type(&left, &right)?)]
        }
        NodeKind::FamilyPreimageSample { matrix_type, max_coefficient_bound } => {
            // A source/trapdoor family indexed by i is paired with a target family indexed by
            // (i,j), yielding witnesses K_i,j that satisfy B_i*K_i,j=T_i,j independently.
            require_arity(scope, node, 3)?;
            if max_coefficient_bound.evaluate(env)?.is_negative() {
                return node_error(scope, node.id, "preimage coefficient bound must be nonnegative");
            }
            let public = argument(scope, values, node, 0)?.clone();
            let trapdoor = argument(scope, values, node, 1)?.clone();
            let public_shape = family_parts(&public).map(|(_, shape)| shape).unwrap_or_default();
            let trapdoor_shape =
                family_parts(&trapdoor).map(|(_, shape)| shape).unwrap_or_default();
            if public_shape != trapdoor_shape ||
                (!is_family(&public) && !matches!(public, ConcreteWireType::Matrix(_))) ||
                (!is_family(&trapdoor) && !matches!(trapdoor, ConcreteWireType::Trapdoor { .. }))
            {
                return node_error(scope, node.id, "preimage public/trapdoor family does not match");
            }
            let target = family_argument(scope, values, node, 2)?;
            let output = concrete_matrix(matrix_type, env, scope, node.id)?;
            let public_element =
                family_parts(&public).map(|(element, _)| element).unwrap_or_else(|| public.clone());
            let (target_element, target_shape) =
                family_parts(&target).ok_or_else(|| ValidationError::Node {
                    scope: scope.to_owned(),
                    node: node.id,
                    message: "family preimage target must be a family".into(),
                })?;
            if public_shape != trapdoor_shape ||
                target_shape.len() != public_shape.len() + 1 ||
                target_shape[..public_shape.len()] != public_shape ||
                !matches!(public_element, ConcreteWireType::Matrix(_)) ||
                !matches!(target_element, ConcreteWireType::Matrix(_))
            {
                return node_error(
                    scope,
                    node.id,
                    "family preimage shapes or element types do not match",
                );
            }
            let trapdoor_matrix = family_parts(&trapdoor)
                .and_then(|_| matrix_from_family(&trapdoor))
                .or_else(|| match trapdoor {
                    ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix),
                    _ => None,
                })
                .ok_or_else(|| ValidationError::Node {
                    scope: scope.to_owned(),
                    node: node.id,
                    message: "preimage trapdoor type is invalid".into(),
                })?;
            check_add_shape(
                &multiplication_type(&trapdoor_matrix, &output)?,
                &matrix_from_family(&target).expect("target matrix family checked"),
            )?;
            // The output keeps every source axis and the target's final branch axis, so relation
            // identity is preserved at each concrete family coordinate.
            vec![ConcreteWireType::Family {
                element: Box::new(ConcreteWireType::Preimage(output)),
                shape: target_shape,
            }]
        }
        NodeKind::GadgetDecompose { base, digit_count, .. } => {
            // GadgetDecompose creates a typed witness K whose row expansion is chosen so the
            // universal gadget equation G*K=T is dimensionally valid.
            require_arity(scope, node, 1)?;
            let input = matrix_argument(scope, values, node, 0)?;
            let base = base.evaluate(env)?;
            if base <= BigInt::one() {
                return node_error(scope, node.id, "gadget base must be greater than one");
            }
            let digits = positive_usize(
                digit_count.evaluate(env)?,
                "decomposition digit count",
                scope,
                node.id,
            )?;
            let rows = input.rows.checked_mul(digits).ok_or_else(|| ValidationError::Node {
                scope: scope.clone(),
                node: node.id,
                message: "gadget decomposition row count overflow".to_owned(),
            })?;
            // Only the witness row count changes: multiplying G (with digit-expanded columns) by
            // K reconstructs the original target T without changing its column count.
            vec![ConcreteWireType::Preimage(ConcreteMatrixType { rows, ..input })]
        }
        NodeKind::MaterializePreimageExact => {
            // This conversion is an identity on the runtime matrix K, but drops B*K=T from the
            // type. Exact-target validation outside this structural pass is what makes that drop
            // safe; a noisy target cannot be hidden by this node.
            require_arity(scope, node, 1)?;
            let input = preimage_argument(scope, values, node, 0)?;
            vec![ConcreteWireType::Matrix(input)]
        }
        NodeKind::PreimageBinary(operation) => {
            // Each typed branch computes a new witness while retaining a common-source relation;
            // the operation equations are B*(K')=T' and are encoded by PreimageBinaryOp.
            require_arity(scope, node, 2)?;
            let left = preimage_argument(scope, values, node, 0)?;
            let output = match operation {
                crate::node::PreimageBinaryOp::Add => {
                    let right = preimage_argument(scope, values, node, 1)?;
                    check_add_shape(&left, &right)?;
                    left
                }
                crate::node::PreimageBinaryOp::RightMultiplyExact => {
                    let right = matrix_argument(scope, values, node, 1)?;
                    multiplication_type(&left, &right)?
                }
                crate::node::PreimageBinaryOp::ComposeExactDecomposition => {
                    let right = preimage_argument(scope, values, node, 1)?;
                    multiplication_type(&left, &right)?
                }
            };
            vec![ConcreteWireType::Preimage(output)]
        }
        NodeKind::PreimageConcatColumns => {
            // Column concatenation maps K_j with B*K_j=T_j to K=[K_1|...|K_n] and
            // T=[T_1|...|T_n], preserving B*K=T componentwise.
            if node.args.is_empty() {
                return node_error(scope, node.id, "preimage concat requires at least one input");
            }
            let inputs = (0..node.args.len())
                .map(|index| preimage_argument(scope, values, node, index))
                .collect::<Result<Vec<_>, _>>()?;
            vec![ConcreteWireType::Preimage(concat_type(
                &inputs,
                crate::node::ConcatAxis::Columns,
                scope,
                node.id,
            )?)]
        }
        NodeKind::DecompositionEntry { row, column } => {
            // The scalar output is the coordinate K[row,column] of a witness satisfying G*K=T;
            // bounds are checked before the decomposition relation can be forgotten.
            require_arity(scope, node, 1)?;
            let input = preimage_argument(scope, values, node, 0)?;
            let row = nonnegative_usize(row.evaluate(env)?, "decomposition row", scope, node.id)?;
            let column =
                nonnegative_usize(column.evaluate(env)?, "decomposition column", scope, node.id)?;
            if row >= input.rows || column >= input.columns {
                return node_error(scope, node.id, "decomposition entry is out of bounds");
            }
            vec![ConcreteWireType::Matrix(ConcreteMatrixType { rows: 1, columns: 1, ..input })]
        }
        NodeKind::ExtractCoefficient { position, .. } => {
            require_arity(scope, node, 1)?;
            let input = matrix_argument(scope, values, node, 0)?;
            let position =
                nonnegative_usize(position.evaluate(env)?, "coefficient position", scope, node.id)?;
            if !input.is_scalar() || position >= input.ring_dimension {
                return node_error(scope, node.id, "coefficient extraction position is invalid");
            }
            vec![ConcreteWireType::Int]
        }
        NodeKind::LiftIntegerToConstantPolynomial { matrix_type } => {
            require_arity(scope, node, 1)?;
            if !matches!(values.get(&node.args[0]), Some(ConcreteWireType::Int)) {
                return node_error(scope, node.id, "constant-polynomial lift requires an integer");
            }
            let matrix = concrete_matrix(matrix_type, env, scope, node.id)?;
            if !matrix.is_scalar() || matrix.modulus <= BigInt::zero() || matrix.ring_dimension == 0
            {
                return node_error(
                    scope,
                    node.id,
                    "constant-polynomial lift requires a positive scalar matrix type",
                );
            }
            vec![ConcreteWireType::Matrix(matrix)]
        }
        NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
            require_arity(scope, node, 1)?;
            let input = matrix_argument(scope, values, node, 0)?;
            let count = positive_usize(length.evaluate(env)?, "decode length", scope, node.id)?;
            if !input.is_scalar() ||
                count > input.ring_dimension ||
                plaintext_modulus.evaluate(env)? <= BigInt::one()
            {
                return node_error(scope, node.id, "invalid threshold decoding parameters");
            }
            vec![if *output_bool { ConcreteWireType::Bool } else { ConcreteWireType::Int }; count]
        }
        NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => {
            if node.args.is_empty() ||
                node.args.len() != plaintext_moduli.len() ||
                node.args.len() != reconstruction_coefficients.len()
            {
                return node_error(scope, node.id, "CRT metadata count does not match inputs");
            }
            let first = matrix_argument(scope, values, node, 0)?;
            if first.rows != 1 {
                return node_error(
                    scope,
                    node.id,
                    "CRT inputs must have identical one-row matrix types",
                );
            }
            for index in 1..node.args.len() {
                if matrix_argument(scope, values, node, index)? != first {
                    return node_error(
                        scope,
                        node.id,
                        "CRT inputs must have identical one-row matrix types",
                    );
                }
            }
            for modulus in plaintext_moduli {
                let value = modulus.evaluate(env)?;
                if value <= BigInt::one() || value >= first.modulus {
                    return node_error(scope, node.id, "invalid CRT plaintext modulus");
                }
            }
            for coefficient in reconstruction_coefficients {
                let value = coefficient.evaluate(env)?;
                if value.is_negative() || value >= first.modulus {
                    return node_error(scope, node.id, "invalid CRT reconstruction coefficient");
                }
            }
            vec![ConcreteWireType::Matrix(first)]
        }
        NodeKind::PackPolynomialCoefficients { matrix_type, coefficient_bits } => {
            require_arity(scope, node, 1)?;
            let output = concrete_matrix(matrix_type, env, scope, node.id)?;
            if !output.is_scalar() {
                return node_error(scope, node.id, "packed polynomial output must be a 1x1 matrix");
            }
            let coefficient_bits = positive_usize(
                coefficient_bits.evaluate(env)?,
                "coefficient bit width",
                scope,
                node.id,
            )?;
            if (BigInt::one() << coefficient_bits) < output.modulus {
                return node_error(
                    scope,
                    node.id,
                    "coefficient bit width does not cover the modulus",
                );
            }
            let expected_count =
                output.ring_dimension.checked_mul(coefficient_bits).ok_or_else(|| {
                    ValidationError::Node {
                        scope: scope.clone(),
                        node: node.id,
                        message: "packed polynomial bit count overflow".to_owned(),
                    }
                })?;
            match argument(scope, values, node, 0)? {
                ConcreteWireType::Family { element, shape }
                    if element.as_ref() == &ConcreteWireType::Bool &&
                        shape == &vec![expected_count] => {}
                _ => {
                    return node_error(
                        scope,
                        node.id,
                        "packed polynomial requires the exact boolean coefficient-bit family",
                    )
                }
            }
            vec![ConcreteWireType::Matrix(output)]
        }
        NodeKind::SubgraphCall(_) | NodeKind::SequentialLoop(_) => node
            .output_types
            .iter()
            .map(|ty| concrete_wire(ty, env, scope, node.id))
            .collect::<Result<Vec<_>, _>>()?,
        NodeKind::FamilyPack { shape } => {
            // Packing lays out ∏_a n_a scalar elements in row-major order, so a rank-r shape
            // describes coordinates u and the flat offset sum_a u_a*∏_{b>a} n_b.
            let shape = concrete_shape(shape, env, scope, node.id)?;
            let count = shape_product(&shape, scope, node.id)?;
            if node.args.len() != count || count == 0 {
                return node_error(scope, node.id, "family pack argument count mismatch");
            }
            let first = argument(scope, values, node, 0)?.clone();
            if is_family(&first) {
                return node_error(scope, node.id, "family members must have one non-family type");
            }
            for index in 1..count {
                if argument(scope, values, node, index)? != &first {
                    return node_error(
                        scope,
                        node.id,
                        "family members must have one non-family type",
                    );
                }
            }
            vec![ConcreteWireType::Family { element: Box::new(first), shape }]
        }
        NodeKind::FamilyGetStatic { indices } => {
            // Static access evaluates the row-major coordinate u and returns exactly X[u].
            require_arity(scope, node, 1)?;
            let ConcreteWireType::Family { element, shape } =
                argument(scope, values, node, 0)?.clone()
            else {
                return node_error(scope, node.id, "family access requires a rank-N family");
            };
            if indices.len() != shape.len() {
                return node_error(scope, node.id, "family index rank mismatch");
            }
            for (axis, index) in indices.iter().enumerate() {
                if nonnegative_usize(index.evaluate(env)?, "family index", scope, node.id)? >=
                    shape[axis]
                {
                    return node_error(scope, node.id, "family index is out of range");
                }
            }
            vec![*element]
        }
        NodeKind::FamilyGetDynamic { rank } => {
            // Dynamic access accepts runtime coordinates u and returns X[u]; bounds remain a
            // runtime concern, while rank and scalar-index types are checked here.
            require_arity(scope, node, rank.saturating_add(1))?;
            let ConcreteWireType::Family { element, shape } =
                argument(scope, values, node, 0)?.clone()
            else {
                return node_error(scope, node.id, "family access requires a rank-N family");
            };
            if shape.len() != *rank {
                return node_error(scope, node.id, "family rank mismatch");
            }
            for index in 0..*rank {
                require_scalar(scope, values, node, index + 1, is_integer, "integer")?;
            }
            warnings.push(runtime_bounds_warning(node.id, "family index is checked at runtime"));
            vec![*element]
        }
        NodeKind::FamilySelectAxis { axis } => {
            // Selecting axis a removes that extent. For each remaining coordinate u, the selected
            // value is X[u with coordinate a replaced by selector(u)].
            require_arity(scope, node, 2)?;
            let family = family_argument(scope, values, node, 0)?;
            let (element, shape) = family_parts(&family).ok_or_else(|| ValidationError::Node {
                scope: scope.to_owned(),
                node: node.id,
                message: "family selection requires a family".into(),
            })?;
            if *axis >= shape.len() {
                return node_error(scope, node.id, "family axis is out of range");
            }
            let selector = argument(scope, values, node, 1)?;
            let mut output_shape = shape;
            output_shape.remove(*axis);
            let valid_selector =
                matches!(selector, ConcreteWireType::Int | ConcreteWireType::ConstantInt) ||
                    family_parts(selector).is_some_and(|(selector_element, selector_shape)| {
                        matches!(
                            selector_element,
                            ConcreteWireType::Int | ConcreteWireType::ConstantInt
                        ) && selector_shape == output_shape
                    });
            if !valid_selector {
                return node_error(
                    scope,
                    node.id,
                    "family selector must be scalar or an exact prefix/suffix integer family",
                );
            }
            if output_shape.is_empty() {
                vec![element]
            } else {
                vec![ConcreteWireType::Family { element: Box::new(element), shape: output_shape }]
            }
        }
        NodeKind::FamilyReindex { output_shape, map } => {
            // Reindexing defines Y[u]=X[f(u)] for every output coordinate u; only the index map
            // and output shape change, while the element (including Preimage) type is preserved.
            require_arity(scope, node, 1)?;
            let shape = concrete_shape(output_shape, env, scope, node.id)?;
            let input = family_argument(scope, values, node, 0)?;
            let (element, input_shape) =
                family_parts(&input).ok_or_else(|| ValidationError::Node {
                    scope: scope.to_owned(),
                    node: node.id,
                    message: "family reindex requires a family".into(),
                })?;
            validate_index_map(
                map,
                shape.len(),
                &input_shape,
                env,
                scope,
                node.id,
                &sequential_slots,
            )?;
            vec![ConcreteWireType::Family { element: Box::new(element), shape }]
        }
        NodeKind::FamilyGather { output_shape, input_rank } => {
            // Gather forms f(u)=(s_0[u],...,s_{r-1}[u]) from one selector family per input axis,
            // then returns Y[u]=X[f(u)] with the declared output shape.
            require_arity(scope, node, input_rank.saturating_add(1))?;
            let shape = concrete_shape(output_shape, env, scope, node.id)?;
            let input = family_argument(scope, values, node, 0)?;
            let (element, input_shape) =
                family_parts(&input).ok_or_else(|| ValidationError::Node {
                    scope: scope.to_owned(),
                    node: node.id,
                    message: "family gather requires a family".into(),
                })?;
            if input_shape.len() != *input_rank {
                return node_error(scope, node.id, "family gather input rank mismatch");
            }
            for index in 0..*input_rank {
                let selector = family_argument(scope, values, node, index + 1)?;
                let (selector_element, selector_shape) =
                    family_parts(&selector).ok_or_else(|| ValidationError::Node {
                        scope: scope.to_owned(),
                        node: node.id,
                        message: "family gather selectors must be families".into(),
                    })?;
                if !matches!(
                    selector_element,
                    ConcreteWireType::Int | ConcreteWireType::ConstantInt
                ) || selector_shape != shape
                {
                    return node_error(scope, node.id, "family gather selector shape mismatch");
                }
            }
            vec![ConcreteWireType::Family { element: Box::new(element), shape }]
        }
        // A grid body computes Y[u] once for each u in ∏_a [0,n_a); concrete output family
        // types therefore wrap each body output with exactly the grid shape.
        NodeKind::ParallelGrid(_) => node
            .output_types
            .iter()
            .map(|ty| concrete_wire(ty, env, scope, node.id))
            .collect::<Result<Vec<_>, _>>()?,
        NodeKind::Select { count } => {
            require_scalar(scope, values, node, 0, is_integer, "integer")?;
            let count =
                positive_usize(count.evaluate(env)?, "select branch count", scope, node.id)?;
            if node.args.len() != count.saturating_add(1) {
                return node_error(scope, node.id, "select branch count does not match arguments");
            }
            let first = argument(scope, values, node, 1)?.clone();
            for index in 2..node.args.len() {
                if argument(scope, values, node, index)? != &first {
                    return node_error(scope, node.id, "select branches have different types");
                }
            }
            warnings.push(runtime_bounds_warning(node.id, "select index is checked at runtime"));
            vec![first]
        }
    };

    let declared = node
        .output_types
        .iter()
        .map(|ty| concrete_wire(ty, env, scope, node.id))
        .collect::<Result<Vec<_>, _>>()?;
    if inferred != declared {
        return node_error(
            scope,
            node.id,
            &format!("declared output types {declared:?} do not match inferred types {inferred:?}"),
        );
    }
    for (port, ty) in inferred.into_iter().enumerate() {
        values.insert(WireRef { node: node.id, port: Port(port as u32) }, ty);
    }
    Ok(())
}

fn validate_structural_boundaries(
    graph: &Graph,
    env: &ParamEnv,
    scopes: &BTreeMap<FrozenGraphScopeId, ValidatedScope>,
) -> Result<(), ValidationError> {
    for (scope_id, scope) in graph.scopes() {
        let validated = &scopes[scope_id];
        for (position, handle) in scope.nodes().iter().enumerate() {
            let node_id = NodeId(position as u64);
            let Some(child_id) = graph.child_scope_id(scope_id, node_id) else { continue };
            let child_scope = graph.scope(&child_id).ok_or_else(|| ValidationError::Node {
                scope: scope_id.clone(),
                node: node_id,
                message: format!("missing child scope {child_id:?}"),
            })?;
            let child = &scopes[&child_id];
            let args = scope.arguments(handle).expect("frozen arguments");
            if let NodeKind::SequentialLoop(loop_spec) = handle.kind() {
                let _ = nonnegative_usize(
                    loop_spec.count.evaluate(env)?,
                    "sequential loop count",
                    scope_id,
                    node_id,
                )?;
                let mut loop_env = env.clone();
                loop_env.loop_indices.insert(loop_spec.index_slot, BigInt::zero());
                let _ = child_bindings(&loop_env, &loop_spec.bindings)?;
                if loop_spec.carried_count == 0 || loop_spec.carried_count > args.len() {
                    return node_error(scope_id, node_id, "invalid sequential carried count");
                }
                if child_scope.inputs().len() != args.len() {
                    return node_error(
                        scope_id,
                        node_id,
                        "sequential body input count does not match arguments",
                    );
                }
                if child_scope.outputs().len() != loop_spec.carried_count ||
                    handle.output_types().len() != loop_spec.carried_count
                {
                    return node_error(
                        scope_id,
                        node_id,
                        "sequential body output count does not match carried count",
                    );
                }
                for (arg, input) in args.iter().zip(child_scope.inputs()) {
                    let outer = validated.wire_types.get(arg).ok_or_else(|| {
                        ValidationError::MissingWire {
                            scope: scope_id.clone(),
                            node: node_id,
                            wire: *arg,
                        }
                    })?;
                    let inner = child.wire_types.get(input).ok_or_else(|| {
                        ValidationError::MissingWire {
                            scope: child_id.clone(),
                            node: input.node,
                            wire: *input,
                        }
                    })?;
                    if !structural_type_compatible(outer, inner) {
                        return node_error(scope_id, node_id, "sequential body input type mismatch");
                    }
                }
                for port in 0..loop_spec.carried_count {
                    let initial_type = &validated.wire_types[&args[port]];
                    let body_type = &child.wire_types[&child_scope.outputs()[port]];
                    let output_type =
                        &validated.wire_types[&WireRef { node: node_id, port: Port(port as u32) }];
                    if !structural_type_compatible(body_type, output_type) ||
                        !structural_type_compatible(initial_type, output_type)
                    {
                        return node_error(scope_id, node_id, "sequential carried type mismatch");
                    }
                }
                continue;
            }
            let modes = match handle.kind() {
                NodeKind::SubgraphCall(call) => {
                    let _ = child_bindings(env, &call.bindings)?;
                    vec![GridInputMode::Broadcast; args.len()]
                }
                NodeKind::ParallelGrid(grid_spec) => {
                    // A reindexed input is transported by X_u=X[f(u)], while a broadcast input is
                    // X_u=X. Validate f against the input family shape before binding body input.
                    if grid_spec.input_modes.len() != args.len() {
                        return node_error(
                            scope_id,
                            node_id,
                            "parallel grid input mode count mismatch",
                        );
                    }
                    let output_shape = concrete_shape(&grid_spec.shape, env, scope_id, node_id)?;
                    for (arg, mode) in args.iter().zip(&grid_spec.input_modes) {
                        if let crate::node::GridInputMode::Reindex { map } = mode {
                            if map.input_indices.iter().any(index_expr_has_loop_index) {
                                return node_error(
                                    scope_id,
                                    node_id,
                                    "parallel grid reindex maps cannot use loop indices",
                                );
                            }
                            let outer = validated.wire_types.get(arg).ok_or_else(|| {
                                ValidationError::MissingWire {
                                    scope: scope_id.clone(),
                                    node: node_id,
                                    wire: *arg,
                                }
                            })?;
                            let (_, input_shape) =
                                family_parts(outer).ok_or_else(|| ValidationError::Node {
                                    scope: scope_id.clone(),
                                    node: node_id,
                                    message: "parallel grid reindex input must be a family".into(),
                                })?;
                            validate_index_map(
                                map,
                                output_shape.len(),
                                &input_shape,
                                env,
                                scope_id,
                                node_id,
                                &BTreeSet::new(),
                            )?;
                        }
                    }
                    grid_spec.input_modes.clone()
                }
                _ => unreachable!(),
            };
            if child_scope.inputs().len() != args.len() {
                return node_error(
                    scope_id,
                    node_id,
                    "child input count does not match call arguments",
                );
            }
            for (index, ((arg, input), mode)) in
                args.iter().zip(child_scope.inputs()).zip(modes).enumerate()
            {
                let outer =
                    validated.wire_types.get(arg).ok_or_else(|| ValidationError::MissingWire {
                        scope: scope_id.clone(),
                        node: node_id,
                        wire: *arg,
                    })?;
                let expected =
                    child.wire_types.get(input).ok_or_else(|| ValidationError::MissingWire {
                        scope: child_id.clone(),
                        node: input.node,
                        wire: *input,
                    })?;
                let actual = match handle.kind() {
                    NodeKind::ParallelGrid(spec)
                        if matches!(spec.input_modes[index], GridInputMode::Reindex { .. }) =>
                    {
                        match outer {
                            ConcreteWireType::Family { element, .. } => element.as_ref(),
                            _ => {
                                return node_error(
                                    scope_id,
                                    node_id,
                                    "parallel grid reindex input must be a family",
                                )
                            }
                        }
                    }
                    _ => match mode {
                        // Broadcast preserves the complete value, including a typed preimage
                        // relation; reindexing above has selected one family element.
                        GridInputMode::Broadcast => outer,
                        GridInputMode::Reindex { .. } => unreachable!("reindex handled above"),
                    },
                };
                if actual != expected {
                    return node_error(scope_id, node_id, "child input type mismatch");
                }
            }
            if child_scope.outputs().len() != handle.output_types().len() {
                return node_error(scope_id, node_id, "child output count mismatch");
            }
            for (port, output) in child_scope.outputs().iter().enumerate() {
                let child_type = &child.wire_types[output];
                let call_type =
                    &validated.wire_types[&WireRef { node: node_id, port: Port(port as u32) }];
                let expected = if let NodeKind::ParallelGrid(spec) = handle.kind() {
                    // A single body result Y is lifted to Y[u] over every grid coordinate, so
                    // the call output must be exactly the body element wrapped by this shape.
                    let shape = concrete_shape(&spec.shape, env, scope_id, node_id)?;
                    ConcreteWireType::Family { element: Box::new(child_type.clone()), shape }
                } else {
                    child_type.clone()
                };
                if *call_type != expected {
                    return node_error(scope_id, node_id, "child output type mismatch");
                }
            }
        }
    }
    Ok(())
}

fn liveness(scope: &GraphScope, mut retained: BTreeSet<WireRef>) -> LivenessSchedule {
    let mut last_use = BTreeMap::new();
    for (index, node) in scope.nodes().iter().enumerate() {
        for argument in scope.arguments(node).expect("frozen arguments") {
            last_use.insert(argument, index);
        }
    }
    retained.extend(scope.outputs().iter().copied());
    LivenessSchedule { last_use, retained }
}

fn argument<'a>(
    scope: &FrozenGraphScopeId,
    values: &'a BTreeMap<WireRef, ConcreteWireType>,
    node: &NodeView<'_>,
    index: usize,
) -> Result<&'a ConcreteWireType, ValidationError> {
    let wire = *node.args.get(index).ok_or_else(|| ValidationError::Node {
        scope: scope.clone(),
        node: node.id,
        message: format!("missing argument {index}"),
    })?;
    values.get(&wire).ok_or_else(|| ValidationError::MissingWire {
        scope: scope.clone(),
        node: node.id,
        wire,
    })
}

fn matrix_argument(
    scope: &FrozenGraphScopeId,
    values: &BTreeMap<WireRef, ConcreteWireType>,
    node: &NodeView<'_>,
    index: usize,
) -> Result<ConcreteMatrixType, ValidationError> {
    argument(scope, values, node, index)?.matrix_type().cloned().ok_or_else(|| {
        ValidationError::Node {
            scope: scope.clone(),
            node: node.id,
            message: "expected matrix argument".to_owned(),
        }
    })
}

fn preimage_argument(
    scope: &FrozenGraphScopeId,
    values: &BTreeMap<WireRef, ConcreteWireType>,
    node: &NodeView<'_>,
    index: usize,
) -> Result<ConcreteMatrixType, ValidationError> {
    match argument(scope, values, node, index)? {
        ConcreteWireType::Preimage(matrix) => Ok(matrix.clone()),
        _ => node_error(scope, node.id, "expected preimage argument"),
    }
}

fn trapdoor_argument(
    scope: &FrozenGraphScopeId,
    values: &BTreeMap<WireRef, ConcreteWireType>,
    node: &NodeView<'_>,
    index: usize,
) -> Result<ConcreteMatrixType, ValidationError> {
    match argument(scope, values, node, index)? {
        ConcreteWireType::Trapdoor { matrix, .. } => Ok(matrix.clone()),
        _ => node_error(scope, node.id, "expected trapdoor argument"),
    }
}

fn require_scalar(
    scope: &FrozenGraphScopeId,
    values: &BTreeMap<WireRef, ConcreteWireType>,
    node: &NodeView<'_>,
    index: usize,
    predicate: fn(&ConcreteWireType) -> bool,
    label: &str,
) -> Result<(), ValidationError> {
    if predicate(argument(scope, values, node, index)?) {
        Ok(())
    } else {
        node_error(scope, node.id, &format!("expected {label} scalar"))
    }
}

fn require_arity(
    scope: &FrozenGraphScopeId,
    node: &NodeView<'_>,
    expected: usize,
) -> Result<(), ValidationError> {
    if node.args.len() == expected {
        Ok(())
    } else {
        node_error(
            scope,
            node.id,
            &format!("expected {expected} arguments, got {}", node.args.len()),
        )
    }
}

fn concrete_wire(
    ty: &WireType,
    env: &ParamEnv,
    scope: &FrozenGraphScopeId,
    node: NodeId,
) -> Result<ConcreteWireType, ValidationError> {
    Ok(match ty {
        WireType::ConstantInt => ConcreteWireType::ConstantInt,
        WireType::ConstantReal => ConcreteWireType::ConstantReal,
        WireType::ConstantBool => ConcreteWireType::ConstantBool,
        WireType::Int => ConcreteWireType::Int,
        WireType::Real => ConcreteWireType::Real,
        WireType::Bool => ConcreteWireType::Bool,
        WireType::Bytes { length } => ConcreteWireType::Bytes {
            length: nonnegative_usize(length.evaluate(env)?, "byte length", scope, node)?,
        },
        WireType::TypedBlob { type_name, schema_hash } => {
            ConcreteWireType::TypedBlob { type_name: type_name.clone(), schema_hash: *schema_hash }
        }
        WireType::Matrix(matrix) => {
            ConcreteWireType::Matrix(concrete_matrix(matrix, env, scope, node)?)
        }
        // A Preimage wire is deliberately distinct from Matrix: the former carries a witness K
        // for some registered B*K=T, whereas the latter carries only the runtime matrix value.
        WireType::Preimage(matrix) => {
            ConcreteWireType::Preimage(concrete_matrix(matrix, env, scope, node)?)
        }
        WireType::Trapdoor {
            matrix,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => ConcreteWireType::Trapdoor {
            matrix: concrete_matrix(matrix, env, scope, node)?,
            sigma: sigma.close(env)?,
            gadget_base: gadget_base.evaluate(env)?,
            digit_count: positive_usize(digit_count.evaluate(env)?, "digit count", scope, node)?,
            preimage_max_coefficient_bound: preimage_max_coefficient_bound.evaluate(env)?,
        },
        WireType::Family { element, shape } => {
            let element = concrete_wire(element, env, scope, node)?;
            if is_family(&element) {
                return node_error(scope, node, "nested families are unsupported");
            }
            ConcreteWireType::Family {
                element: Box::new(element),
                shape: concrete_shape(shape, env, scope, node)?,
            }
        }
    })
}

fn is_family(ty: &ConcreteWireType) -> bool {
    matches!(ty, ConcreteWireType::Family { .. })
}

fn family_parts(ty: &ConcreteWireType) -> Option<(ConcreteWireType, Vec<usize>)> {
    // Family metadata is separated from its element so rank and coordinate equations can be
    // checked without changing the element identity (including a Preimage relation marker).
    match ty {
        ConcreteWireType::Family { element, shape } => {
            Some((element.as_ref().clone(), shape.clone()))
        }
        _ => None,
    }
}

fn validate_index_map(
    map: &IndexMap,
    output_rank: usize,
    input_shape: &[usize],
    env: &ParamEnv,
    scope: &FrozenGraphScopeId,
    node: NodeId,
    sequential_slots: &BTreeSet<u32>,
) -> Result<(), ValidationError> {
    // Every input axis must receive one coordinate expression. A concrete expression is checked
    // against its input extent; an axis-dependent expression is retained for runtime checking.
    if map.input_indices.len() != input_shape.len() {
        return node_error(scope, node, "index map arity does not match input family rank");
    }
    for (axis, expression) in map.input_indices.iter().enumerate() {
        let value =
            validate_index_expr(expression, output_rank, env, scope, node, sequential_slots)?;
        if let Some(value) = value {
            let Some(index) = value.to_usize() else {
                return node_error(scope, node, "index map result must be nonnegative");
            };
            if index >= input_shape[axis] {
                return node_error(scope, node, "index map result is outside input shape");
            }
        }
    }
    Ok(())
}

fn validate_index_expr(
    expression: &IndexExpr,
    output_rank: usize,
    env: &ParamEnv,
    scope: &FrozenGraphScopeId,
    node: NodeId,
    sequential_slots: &BTreeSet<u32>,
) -> Result<Option<BigInt>, ValidationError> {
    // `Some(v)` means the coordinate is compile-time evaluable; `None` means it depends on an
    // output axis or loop binder and must remain a symbolic coordinate equation.
    let value = match expression {
        IndexExpr::Axis(axis) => {
            if *axis >= output_rank {
                return node_error(scope, node, "index map axis is outside output rank");
            }
            None
        }
        IndexExpr::Parameter(name) => {
            Some(env.integers.get(name).cloned().ok_or_else(|| ValidationError::Node {
                scope: scope.clone(),
                node,
                message: format!("index map parameter {name} is unresolved"),
            })?)
        }
        IndexExpr::LoopIndex(slot) => {
            if !sequential_slots.contains(slot) {
                return node_error(
                    scope,
                    node,
                    "index map loop index is not from a sequential loop",
                );
            }
            None
        }
        IndexExpr::Constant(value) => Some(value.clone()),
        IndexExpr::Add(lhs, rhs) => {
            index_binary(lhs, rhs, output_rank, env, scope, node, sequential_slots, |a, b| a + b)?
        }
        IndexExpr::Subtract(lhs, rhs) => {
            index_binary(lhs, rhs, output_rank, env, scope, node, sequential_slots, |a, b| a - b)?
        }
        IndexExpr::Multiply(lhs, rhs) => {
            index_binary(lhs, rhs, output_rank, env, scope, node, sequential_slots, |a, b| a * b)?
        }
        IndexExpr::Divide(lhs, rhs) => index_binary_checked(
            lhs,
            rhs,
            output_rank,
            env,
            scope,
            node,
            sequential_slots,
            |a, b| a / b,
        )?,
        IndexExpr::Remainder(lhs, rhs) => index_binary_checked(
            lhs,
            rhs,
            output_rank,
            env,
            scope,
            node,
            sequential_slots,
            |a, b| a % b,
        )?,
        IndexExpr::Equal(lhs, rhs) => {
            index_binary(lhs, rhs, output_rank, env, scope, node, sequential_slots, |a, b| {
                BigInt::from(a == b)
            })?
        }
        IndexExpr::Less(lhs, rhs) => {
            index_binary(lhs, rhs, output_rank, env, scope, node, sequential_slots, |a, b| {
                BigInt::from(a < b)
            })?
        }
        IndexExpr::LessEqual(lhs, rhs) => {
            index_binary(lhs, rhs, output_rank, env, scope, node, sequential_slots, |a, b| {
                BigInt::from(a <= b)
            })?
        }
        IndexExpr::Log2Ceil(value) => {
            validate_index_expr(value, output_rank, env, scope, node, sequential_slots)?;
            None
        }
        IndexExpr::Select { selector, branches } => {
            if branches.is_empty() {
                return node_error(scope, node, "index map select must have at least one branch");
            }
            let selector =
                validate_index_expr(selector, output_rank, env, scope, node, sequential_slots)?;
            let branches = branches
                .iter()
                .map(|branch| {
                    validate_index_expr(branch, output_rank, env, scope, node, sequential_slots)
                })
                .collect::<Result<Vec<_>, _>>()?;
            match selector.and_then(|value| value.to_usize()) {
                Some(index) => branches.get(index).cloned().flatten(),
                None => None,
            }
        }
    };
    Ok(value)
}

fn index_expr_has_loop_index(expression: &IndexExpr) -> bool {
    match expression {
        IndexExpr::LoopIndex(_) => true,
        IndexExpr::Add(lhs, rhs) |
        IndexExpr::Subtract(lhs, rhs) |
        IndexExpr::Multiply(lhs, rhs) |
        IndexExpr::Divide(lhs, rhs) |
        IndexExpr::Remainder(lhs, rhs) |
        IndexExpr::Equal(lhs, rhs) |
        IndexExpr::Less(lhs, rhs) |
        IndexExpr::LessEqual(lhs, rhs) => {
            index_expr_has_loop_index(lhs) || index_expr_has_loop_index(rhs)
        }
        IndexExpr::Select { selector, branches } => {
            index_expr_has_loop_index(selector) || branches.iter().any(index_expr_has_loop_index)
        }
        IndexExpr::Log2Ceil(value) => index_expr_has_loop_index(value),
        IndexExpr::Axis(_) | IndexExpr::Parameter(_) | IndexExpr::Constant(_) => false,
    }
}

fn index_expr_has_loop_or_parameter(expression: &IndexExpr) -> bool {
    match expression {
        IndexExpr::Axis(_) | IndexExpr::LoopIndex(_) | IndexExpr::Parameter(_) => true,
        IndexExpr::Log2Ceil(value) => index_expr_has_loop_or_parameter(value),
        IndexExpr::Add(lhs, rhs) |
        IndexExpr::Subtract(lhs, rhs) |
        IndexExpr::Multiply(lhs, rhs) |
        IndexExpr::Divide(lhs, rhs) |
        IndexExpr::Remainder(lhs, rhs) |
        IndexExpr::Equal(lhs, rhs) |
        IndexExpr::Less(lhs, rhs) |
        IndexExpr::LessEqual(lhs, rhs) => {
            index_expr_has_loop_or_parameter(lhs) || index_expr_has_loop_or_parameter(rhs)
        }
        IndexExpr::Select { selector, branches } => {
            index_expr_has_loop_or_parameter(selector) ||
                branches.iter().any(index_expr_has_loop_or_parameter)
        }
        IndexExpr::Constant(_) => false,
    }
}

fn index_binary(
    lhs: &IndexExpr,
    rhs: &IndexExpr,
    output_rank: usize,
    env: &ParamEnv,
    scope: &FrozenGraphScopeId,
    node: NodeId,
    sequential_slots: &BTreeSet<u32>,
    operation: impl FnOnce(BigInt, BigInt) -> BigInt,
) -> Result<Option<BigInt>, ValidationError> {
    // For compile-time operands this computes f(u)=lhs(u) op rhs(u); symbolic operands remain
    // unresolved so the same coordinate equation can be checked at execution time.
    let lhs = validate_index_expr(lhs, output_rank, env, scope, node, sequential_slots)?;
    let rhs = validate_index_expr(rhs, output_rank, env, scope, node, sequential_slots)?;
    Ok(lhs.zip(rhs).map(|(lhs, rhs)| operation(lhs, rhs)))
}

fn index_binary_checked(
    lhs: &IndexExpr,
    rhs: &IndexExpr,
    output_rank: usize,
    env: &ParamEnv,
    scope: &FrozenGraphScopeId,
    node: NodeId,
    sequential_slots: &BTreeSet<u32>,
    operation: impl FnOnce(BigInt, BigInt) -> BigInt,
) -> Result<Option<BigInt>, ValidationError> {
    // This is the same f(u)=lhs(u) op rhs(u) calculation with an explicit nonzero divisor guard
    // for division and remainder.
    let lhs = validate_index_expr(lhs, output_rank, env, scope, node, sequential_slots)?;
    let rhs = validate_index_expr(rhs, output_rank, env, scope, node, sequential_slots)?;
    if rhs.as_ref().is_some_and(Zero::is_zero) {
        return node_error(scope, node, "index map division by zero");
    }
    Ok(lhs.zip(rhs).map(|(lhs, rhs)| operation(lhs, rhs)))
}

fn family_argument(
    scope: &FrozenGraphScopeId,
    values: &BTreeMap<WireRef, ConcreteWireType>,
    node: &NodeView<'_>,
    index: usize,
) -> Result<ConcreteWireType, ValidationError> {
    // Family-valued arguments retain their complete coordinate domain; selecting an element is a
    // separate operation because it changes X[u] into one concrete X[u_0].
    let value = argument(scope, values, node, index)?.clone();
    if is_family(&value) {
        Ok(value)
    } else {
        node_error(scope, node.id, "expected family argument")
    }
}

fn matrix_from_family(ty: &ConcreteWireType) -> Option<ConcreteMatrixType> {
    // This projection reads the common matrix schema of every family element; it does not select
    // a coordinate and therefore does not alter the family relation or shape.
    family_parts(ty).and_then(|(element, _)| element.matrix_type().cloned())
}

fn concrete_shape(
    shape: &[IntExpr],
    env: &ParamEnv,
    scope: &FrozenGraphScopeId,
    node: NodeId,
) -> Result<Vec<usize>, ValidationError> {
    // Each symbolic extent n_a is evaluated independently; the resulting vector is the concrete
    // Cartesian domain shape `(n_0,...,n_{r-1})` used by all rank-N equations below.
    if shape.is_empty() {
        return node_error(scope, node, "family shape must be nonempty");
    }
    shape
        .iter()
        .map(|extent| nonnegative_usize(extent.evaluate(env)?, "family extent", scope, node))
        .collect()
}

fn shape_product(
    shape: &[usize],
    scope: &FrozenGraphScopeId,
    node: NodeId,
) -> Result<usize, ValidationError> {
    // Flattened family cardinality is the product |I|=∏_a n_a, with checked multiplication to
    // keep the coordinate domain finite and representable.
    shape.iter().try_fold(1usize, |product, extent| product.checked_mul(*extent)).ok_or_else(|| {
        ValidationError::Node {
            scope: scope.to_owned(),
            node,
            message: "family shape product overflow".to_owned(),
        }
    })
}

fn concrete_matrix(
    matrix: &MatrixType,
    env: &ParamEnv,
    scope: &FrozenGraphScopeId,
    node: NodeId,
) -> Result<ConcreteMatrixType, ValidationError> {
    let modulus = matrix.modulus.evaluate(env)?;
    if modulus <= BigInt::one() {
        return node_error(scope, node, "matrix modulus must exceed one");
    }
    Ok(ConcreteMatrixType {
        modulus,
        ring_dimension: positive_usize(
            matrix.ring_dimension.evaluate(env)?,
            "ring dimension",
            scope,
            node,
        )?,
        rows: positive_usize(matrix.rows.evaluate(env)?, "matrix rows", scope, node)?,
        columns: positive_usize(matrix.columns.evaluate(env)?, "matrix columns", scope, node)?,
    })
}

fn validate_constant(
    value: &ConstantMatrix,
    matrix: &ConcreteMatrixType,
    env: &ParamEnv,
    scope: &FrozenGraphScopeId,
    node: NodeId,
) -> Result<(), ValidationError> {
    match value {
        ConstantMatrix::UnitRow { index }
            if nonnegative_usize(index.evaluate(env)?, "unit-row index", scope, node)? >=
                matrix.columns =>
        {
            node_error(scope, node, "unit-row index is out of range")
        }
        ConstantMatrix::UnitColumn { index }
            if nonnegative_usize(index.evaluate(env)?, "unit-column index", scope, node)? >=
                matrix.rows =>
        {
            node_error(scope, node, "unit-column index is out of range")
        }
        ConstantMatrix::Gadget { base, .. } if base.evaluate(env)?.abs() <= BigInt::one() => {
            node_error(scope, node, "gadget base must exceed one")
        }
        ConstantMatrix::PowerOfBase { base, exponent }
            if base.evaluate(env)?.is_zero() || exponent.evaluate(env)?.is_negative() =>
        {
            node_error(scope, node, "invalid power-of-base constant")
        }
        ConstantMatrix::Rotation { exponent }
            if nonnegative_usize(exponent.evaluate(env)?, "rotation exponent", scope, node)? >=
                matrix.ring_dimension =>
        {
            node_error(scope, node, "rotation exponent is out of range")
        }
        ConstantMatrix::Polynomial { coefficients }
            if coefficients.len() > matrix.ring_dimension =>
        {
            node_error(scope, node, "constant polynomial exceeds the ring dimension")
        }
        ConstantMatrix::Polynomial { coefficients } => {
            for coefficient in coefficients {
                coefficient.evaluate(env)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn concat_type(
    inputs: &[ConcreteMatrixType],
    axis: ConcatAxis,
    scope: &FrozenGraphScopeId,
    node: NodeId,
) -> Result<ConcreteMatrixType, ValidationError> {
    let Some(first) = inputs.first() else {
        return node_error(scope, node, "concat requires input");
    };
    for input in &inputs[1..] {
        crate::checks::check_same_ring(first, input)?;
        if matches!(axis, ConcatAxis::Rows) && input.columns != first.columns ||
            matches!(axis, ConcatAxis::Columns) && input.rows != first.rows
        {
            return node_error(scope, node, "concat input shapes are incompatible");
        }
    }
    let (rows, columns) = match axis {
        ConcatAxis::Rows => (inputs.iter().map(|x| x.rows).sum(), first.columns),
        ConcatAxis::Columns => (first.rows, inputs.iter().map(|x| x.columns).sum()),
        ConcatAxis::Diagonal => {
            (inputs.iter().map(|x| x.rows).sum(), inputs.iter().map(|x| x.columns).sum())
        }
    };
    Ok(ConcreteMatrixType { rows, columns, ..first.clone() })
}

fn sliced_type(
    input: &ConcreteMatrixType,
    rows: Option<&crate::node::IndexRange>,
    columns: Option<&crate::node::IndexRange>,
    env: &ParamEnv,
    scope: &FrozenGraphScopeId,
    node: NodeId,
) -> Result<ConcreteMatrixType, ValidationError> {
    let evaluate = |range: &crate::node::IndexRange| -> Result<(usize, usize), ValidationError> {
        Ok((
            nonnegative_usize(range.start.evaluate(env)?, "slice start", scope, node)?,
            nonnegative_usize(range.end.evaluate(env)?, "slice end", scope, node)?,
        ))
    };
    let rows = rows.map(evaluate).transpose()?;
    let columns = columns.map(evaluate).transpose()?;
    if rows.is_some_and(|(start, end)| start >= end || end > input.rows) ||
        columns.is_some_and(|(start, end)| start >= end || end > input.columns)
    {
        return node_error(scope, node, "slice range is invalid");
    }
    Ok(ConcreteMatrixType {
        rows: rows.map_or(input.rows, |(start, end)| end - start),
        columns: columns.map_or(input.columns, |(start, end)| end - start),
        ..input.clone()
    })
}

fn check_bindings(graph: &Graph, env: &ParamEnv) -> Result<(), ValidationError> {
    for parameter in graph.parameters() {
        let present = match parameter.kind {
            CompileParameterKind::Integer => env.integers.contains_key(&parameter.name),
            CompileParameterKind::Real => env.reals.contains_key(&parameter.name),
        };
        if !present {
            return Err(ValidationError::MissingBinding(parameter.name.clone()));
        }
    }
    Ok(())
}

fn child_bindings(
    parent: &ParamEnv,
    bindings: &[(String, IntExpr)],
) -> Result<ParamEnv, ValidationError> {
    let mut child = parent.clone();
    for (name, expression) in bindings {
        child.integers.insert(name.clone(), expression.evaluate(parent)?);
    }
    Ok(child)
}

fn family_element_shape(ty: &ConcreteWireType) -> (&ConcreteWireType, Option<&[usize]>) {
    match ty {
        ConcreteWireType::Family { element, shape } => (element, Some(shape.as_slice())),
        scalar => (scalar, None),
    }
}

fn positive_usize(
    value: BigInt,
    label: &str,
    scope: &FrozenGraphScopeId,
    node: NodeId,
) -> Result<usize, ValidationError> {
    value.to_usize().filter(|value| *value > 0).ok_or_else(|| ValidationError::Node {
        scope: scope.clone(),
        node,
        message: format!("{label} must be a positive usize"),
    })
}

fn nonnegative_usize(
    value: BigInt,
    label: &str,
    scope: &FrozenGraphScopeId,
    node: NodeId,
) -> Result<usize, ValidationError> {
    value.to_usize().ok_or_else(|| ValidationError::Node {
        scope: scope.clone(),
        node,
        message: format!("{label} must be a nonnegative usize"),
    })
}

fn require_positive_real(
    value: f64,
    scope: &FrozenGraphScopeId,
    node: NodeId,
    label: &str,
) -> Result<(), ValidationError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        node_error(scope, node, &format!("{label} must be finite and positive"))
    }
}

fn require_nonnegative_real(
    value: f64,
    scope: &FrozenGraphScopeId,
    node: NodeId,
    label: &str,
) -> Result<(), ValidationError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        node_error(scope, node, &format!("{label} must be finite and nonnegative"))
    }
}

fn node_error<T>(
    scope: &FrozenGraphScopeId,
    node: NodeId,
    message: &str,
) -> Result<T, ValidationError> {
    Err(ValidationError::Node { scope: scope.clone(), node, message: message.to_owned() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::{CompileParameter, GraphOutput, SubgraphHandle, ValueHandle},
        node::{GridInputMode, ParallelGrid, SampleRange, SequentialLoop},
        with_new_construction_scope,
    };

    fn matrix_type(modulus: i64, rows: i64, columns: i64) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(modulus),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn input(name: &str, matrix_type: MatrixType) -> ValueHandle {
        let wire_type = WireType::Matrix(matrix_type);
        NodeHandle::new(
            NodeKind::Input { name: name.to_owned(), wire_type: wire_type.clone(), artifact: None },
            Vec::new(),
            vec![wire_type],
        )
        .output(0)
        .expect("matrix input")
    }

    fn value(
        kind: NodeKind,
        arguments: Vec<ValueHandle>,
        output_types: Vec<WireType>,
    ) -> ValueHandle {
        NodeHandle::new(kind, arguments, output_types).output(0).expect("node has an output")
    }

    fn graph(name: &str, output: ValueHandle) -> Graph {
        Graph::freeze(
            name,
            Vec::new(),
            BTreeMap::from([(
                "output".to_owned(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze graph")
        .0
    }

    fn node_message(error: ValidationError) -> String {
        match error {
            ValidationError::Node { message, .. } => message,
            ValidationError::ParameterConstraint(message) => message,
            other => panic!("expected node validation error, got {other:?}"),
        }
    }

    #[test]
    fn matrix_arithmetic_rejects_shape_and_ring_mismatches() {
        let left = input("left", matrix_type(17, 1, 2));
        let right = input("right", matrix_type(17, 2, 1));
        let sum = value(
            NodeKind::MatrixBinary(MatrixBinaryOp::Add),
            vec![left, right],
            vec![WireType::Matrix(matrix_type(17, 1, 2))],
        );
        assert!(matches!(
            validate(&graph("shape-mismatch", sum), &ParamEnv::default()),
            Err(ValidationError::Check(CheckError::ShapeMismatch { .. }))
        ));

        let left = input("left", matrix_type(17, 1, 1));
        let right = input("right", matrix_type(19, 1, 1));
        let product = value(
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            vec![left, right],
            vec![WireType::Matrix(matrix_type(17, 1, 1))],
        );
        assert!(matches!(
            validate(&graph("ring-mismatch", product), &ParamEnv::default()),
            Err(ValidationError::Check(CheckError::RingMismatch))
        ));
    }

    #[test]
    fn samplers_and_structural_views_reject_invalid_parameters() {
        let gaussian_type = matrix_type(17, 1, 1);
        let gaussian = value(
            NodeKind::GaussianSample {
                matrix_type: gaussian_type.clone(),
                sigma: crate::RealExpr::from_integer(-1),
                max_coefficient_bound: IntExpr::constant(0),
            },
            Vec::new(),
            vec![WireType::Matrix(gaussian_type)],
        );
        assert!(
            node_message(
                validate(&graph("negative-gaussian", gaussian), &ParamEnv::default()).unwrap_err()
            )
            .contains("Gaussian sigma")
        );

        let uniform_type = matrix_type(17, 1, 1);
        let uniform = value(
            NodeKind::UniformIntervalSample {
                matrix_type: uniform_type.clone(),
                range: SampleRange { minimum: IntExpr::constant(2), maximum: IntExpr::constant(1) },
            },
            Vec::new(),
            vec![WireType::Matrix(uniform_type)],
        );
        assert!(
            node_message(
                validate(&graph("empty-uniform", uniform), &ParamEnv::default()).unwrap_err()
            )
            .contains("uniform range")
        );

        let unsupported_interval_type = matrix_type(17, 1, 1);
        let unsupported_interval = value(
            NodeKind::UniformIntervalSample {
                matrix_type: unsupported_interval_type.clone(),
                range: SampleRange { minimum: IntExpr::constant(2), maximum: IntExpr::constant(3) },
            },
            Vec::new(),
            vec![WireType::Matrix(unsupported_interval_type)],
        );
        assert!(
            node_message(
                validate(
                    &graph("unsupported-uniform-interval", unsupported_interval),
                    &ParamEnv::default()
                )
                .unwrap_err()
            )
            .contains("uniform interval")
        );

        let residue_type = matrix_type(17, 1, 1);
        let residue = value(
            NodeKind::UniformResidueSample { matrix_type: residue_type.clone() },
            Vec::new(),
            vec![WireType::Matrix(residue_type)],
        );
        assert!(validate(&graph("uniform-residue", residue), &ParamEnv::default()).is_ok());
    }

    #[test]
    fn trapdoor_digit_count_must_be_positive_in_nodes_and_wire_types() {
        let trapdoor_sample = |digit_count: i64| {
            let matrix = matrix_type(17, 1, digit_count + 2);
            let trapdoor = WireType::Trapdoor {
                matrix: matrix.clone(),
                sigma: crate::RealExpr::from_integer(1),
                gadget_base: IntExpr::constant(2),
                digit_count: IntExpr::constant(digit_count),
                preimage_max_coefficient_bound: IntExpr::constant(8),
            };
            NodeHandle::new(
                NodeKind::TrapdoorSample {
                    matrix_type: matrix.clone(),
                    sigma: crate::RealExpr::from_integer(1),
                    gadget_base: IntExpr::constant(2),
                    digit_count: IntExpr::constant(digit_count),
                    preimage_max_coefficient_bound: IntExpr::constant(8),
                },
                Vec::new(),
                vec![WireType::Matrix(matrix), trapdoor],
            )
            .output(0)
            .expect("trapdoor sample matrix output")
        };

        assert!(
            node_message(
                validate(&graph("zero-trapdoor-digits", trapdoor_sample(0)), &ParamEnv::default())
                    .unwrap_err()
            )
            .contains("digit count")
        );
        validate(&graph("positive-trapdoor-digits", trapdoor_sample(1)), &ParamEnv::default())
            .expect("a trapdoor with one gadget digit is valid");

        let trapdoor_input = |digit_count: i64| {
            let matrix = matrix_type(17, 1, digit_count + 2);
            let wire_type = WireType::Trapdoor {
                matrix,
                sigma: crate::RealExpr::from_integer(1),
                gadget_base: IntExpr::constant(2),
                digit_count: IntExpr::constant(digit_count),
                preimage_max_coefficient_bound: IntExpr::constant(8),
            };
            value(
                NodeKind::Input {
                    name: format!("trapdoor-{digit_count}"),
                    wire_type: wire_type.clone(),
                    artifact: None,
                },
                Vec::new(),
                vec![wire_type],
            )
        };

        assert!(
            node_message(
                validate(
                    &graph("zero-trapdoor-wire-digits", trapdoor_input(0)),
                    &ParamEnv::default()
                )
                .unwrap_err()
            )
            .contains("trapdoor digit count")
        );
        validate(&graph("positive-trapdoor-wire-digits", trapdoor_input(1)), &ParamEnv::default())
            .expect("a trapdoor wire with one gadget digit is valid");
    }

    #[test]
    fn crt_and_decode_validate_all_metadata_against_the_input_type() {
        let level = input("level", matrix_type(257, 1, 2));
        let crt = value(
            NodeKind::CrtRecompose {
                plaintext_moduli: vec![IntExpr::constant(3), IntExpr::constant(5)],
                reconstruction_coefficients: vec![IntExpr::constant(1)],
            },
            vec![level.clone()],
            vec![WireType::Matrix(matrix_type(257, 1, 2))],
        );
        assert_eq!(
            node_message(validate(&graph("invalid-crt", crt), &ParamEnv::default()).unwrap_err()),
            "CRT metadata count does not match inputs"
        );

        let decode = NodeHandle::new(
            NodeKind::ThresholdDecode {
                plaintext_modulus: IntExpr::constant(1),
                length: IntExpr::constant(9),
                output_bool: false,
            },
            vec![level],
            vec![WireType::Int; 9],
        )
        .output(0)
        .expect("decode output");
        assert!(
            node_message(
                validate(&graph("invalid-decode", decode), &ParamEnv::default()).unwrap_err()
            )
            .contains("plaintext modulus")
        );
    }

    #[test]
    fn packed_polynomial_validation_requires_canonical_width_and_exact_bool_count() {
        let bools = (0..8)
            .map(|_| value(NodeKind::ConstantBool(false), Vec::new(), vec![WireType::ConstantBool]))
            .collect::<Vec<_>>();
        let family = value(
            NodeKind::FamilyPack { shape: vec![IntExpr::constant(8)] },
            bools,
            vec![WireType::Family {
                element: Box::new(WireType::ConstantBool),
                shape: vec![IntExpr::constant(8)],
            }],
        );
        let packed = value(
            NodeKind::PackPolynomialCoefficients {
                matrix_type: matrix_type(257, 1, 1),
                coefficient_bits: IntExpr::constant(8),
            },
            vec![family],
            vec![WireType::Matrix(matrix_type(257, 1, 1))],
        );
        assert_eq!(
            node_message(
                validate(&graph("invalid-packed-polynomial", packed), &ParamEnv::default())
                    .unwrap_err()
            ),
            "coefficient bit width does not cover the modulus"
        );
    }

    #[test]
    fn parallel_grid_preserves_nested_family_body_output() {
        let matrix = matrix_type(17, 1, 1);
        let (body, _) = with_new_construction_scope(|scope| {
            let first = value(
                NodeKind::ConstantMatrix {
                    matrix_type: matrix.clone(),
                    value: ConstantMatrix::Zero,
                },
                Vec::new(),
                vec![WireType::Matrix(matrix.clone())],
            );
            (
                SubgraphHandle::new(
                    "legacy-loop-family-body",
                    scope,
                    Vec::new(),
                    vec![first.clone()],
                )
                .unwrap(),
                first,
            )
        });
        let output = NodeHandle::parallel_grid(
            body,
            Vec::new(),
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(matrix.clone())),
                shape: vec![IntExpr::Var("count".into())],
            }],
            ParallelGrid {
                shape: vec![IntExpr::Var("count".into())],
                index_slots: vec![0],
                bindings: Vec::new(),
                input_modes: Vec::new(),
            },
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "legacy-loop-family",
            vec![CompileParameter { name: "count".into(), kind: CompileParameterKind::Integer }],
            BTreeMap::from([("out".into(), GraphOutput { value: output, confidentiality: None })]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let mut environment = ParamEnv::default();
        environment.integers.insert("count".into(), 3.into());
        let validated = validate(&graph, &environment).unwrap();
        assert_eq!(
            validated.root_scope().wire_types[&WireRef { node: NodeId(0), port: Port(0) }],
            ConcreteWireType::Family {
                element: Box::new(ConcreteWireType::Matrix(ConcreteMatrixType::scalar(
                    17.into(),
                    8
                ))),
                shape: vec![3],
            }
        );
    }

    #[test]
    fn family_validation_rejects_heterogeneous_members_and_warns_for_dynamic_indices() {
        let left = input("left", matrix_type(17, 1, 1));
        let right = input("right", matrix_type(17, 2, 1));
        let heterogeneous = value(
            NodeKind::FamilyPack { shape: vec![IntExpr::constant(2)] },
            vec![left, right],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(matrix_type(17, 1, 1))),
                shape: vec![IntExpr::constant(2)],
            }],
        );
        assert_eq!(
            node_message(
                validate(&graph("heterogeneous-family", heterogeneous), &ParamEnv::default())
                    .unwrap_err()
            ),
            "family members must have one non-family type"
        );

        let first = input("first", matrix_type(17, 1, 1));
        let second = input("second", matrix_type(17, 1, 1));
        let family_type = WireType::Family {
            element: Box::new(WireType::Matrix(matrix_type(17, 1, 1))),
            shape: vec![IntExpr::constant(2)],
        };
        let family = value(
            NodeKind::FamilyPack { shape: vec![IntExpr::constant(2)] },
            vec![first, second],
            vec![family_type],
        );
        let index =
            value(NodeKind::ConstantInt(BigInt::from(1)), Vec::new(), vec![WireType::ConstantInt]);
        let selected = value(
            NodeKind::FamilyGetDynamic { rank: 1 },
            vec![family, index],
            vec![WireType::Matrix(matrix_type(17, 1, 1))],
        );
        let validated = validate(&graph("dynamic-family", selected), &ParamEnv::default()).unwrap();
        assert_eq!(validated.warnings.len(), 1);
        assert_eq!(validated.warnings[0].kind, WarningKind::RuntimeSelectBoundsCheck);
    }

    #[test]
    fn loop_dependent_family_access_requires_dynamic_indexing() {
        let family_type = WireType::Family {
            element: Box::new(WireType::Int),
            shape: vec![IntExpr::constant(2)],
        };
        let family = value(
            NodeKind::Input {
                name: "family".to_owned(),
                wire_type: family_type.clone(),
                artifact: None,
            },
            Vec::new(),
            vec![family_type.clone()],
        );

        let parallel = |dynamic, aliased| {
            let (inputs, selected, scope) = crate::with_new_construction_scope(|scope| {
                let input = value(
                    NodeKind::Input {
                        name: "family".to_owned(),
                        wire_type: family_type.clone(),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![family_type.clone()],
                );
                let loop_index =
                    if aliased { IntExpr::Var("i".to_owned()) } else { IntExpr::LoopIndex(0) };
                let selected = if dynamic {
                    let index = value(
                        NodeKind::EvaluateInt(loop_index.clone()),
                        Vec::new(),
                        vec![WireType::ConstantInt],
                    );
                    value(
                        NodeKind::FamilyGetDynamic { rank: 1 },
                        vec![input.clone(), index],
                        vec![WireType::Int],
                    )
                } else {
                    value(
                        NodeKind::FamilyGetStatic {
                            indices: vec![IndexExpr::try_from(loop_index.clone()).unwrap()],
                        },
                        vec![input.clone()],
                        vec![WireType::Int],
                    )
                };
                (vec![input], selected, scope)
            });
            let body = crate::SubgraphHandle::new(
                if dynamic { "dynamic-parallel" } else { "static-parallel" },
                scope,
                inputs,
                vec![selected],
            )
            .unwrap();
            NodeHandle::parallel_grid(
                body,
                vec![family.clone()],
                vec![WireType::Family {
                    element: Box::new(WireType::Int),
                    shape: vec![IntExpr::constant(2)],
                }],
                ParallelGrid {
                    shape: vec![IntExpr::constant(2)],
                    index_slots: vec![0],
                    bindings: if aliased {
                        vec![("i".to_owned(), IntExpr::LoopIndex(0))]
                    } else {
                        Vec::new()
                    },
                    input_modes: vec![GridInputMode::Broadcast],
                },
            )
            .output(0)
            .unwrap()
        };

        let static_parallel = graph("static-parallel", parallel(false, false));
        assert_eq!(
            node_message(validate(&static_parallel, &ParamEnv::default()).unwrap_err()),
            "loop-dependent family access must use FamilyGetDynamic"
        );
        validate(&graph("dynamic-parallel", parallel(true, false)), &ParamEnv::default()).unwrap();
        assert_eq!(
            node_message(
                validate(
                    &graph("aliased-static-parallel", parallel(false, true)),
                    &ParamEnv::default()
                )
                .unwrap_err()
            ),
            "loop-dependent family access must use FamilyGetDynamic"
        );

        let sequential = |dynamic, aliased| {
            let (inputs, selected, scope) = crate::with_new_construction_scope(|scope| {
                let state = value(
                    NodeKind::Input {
                        name: "state".to_owned(),
                        wire_type: WireType::Int,
                        artifact: None,
                    },
                    Vec::new(),
                    vec![WireType::Int],
                );
                let invariant = value(
                    NodeKind::Input {
                        name: "family".to_owned(),
                        wire_type: family_type.clone(),
                        artifact: None,
                    },
                    Vec::new(),
                    vec![family_type.clone()],
                );
                let loop_index =
                    if aliased { IntExpr::Var("i".to_owned()) } else { IntExpr::LoopIndex(0) };
                let selected = if dynamic {
                    let index = value(
                        NodeKind::EvaluateInt(loop_index.clone()),
                        Vec::new(),
                        vec![WireType::ConstantInt],
                    );
                    value(
                        NodeKind::FamilyGetDynamic { rank: 1 },
                        vec![invariant.clone(), index],
                        vec![WireType::Int],
                    )
                } else {
                    value(
                        NodeKind::FamilyGetStatic {
                            indices: vec![IndexExpr::try_from(loop_index.clone()).unwrap()],
                        },
                        vec![invariant.clone()],
                        vec![WireType::Int],
                    )
                };
                (vec![state, invariant], selected, scope)
            });
            let body = crate::SubgraphHandle::new(
                if dynamic { "dynamic-sequential" } else { "static-sequential" },
                scope,
                inputs,
                vec![selected],
            )
            .unwrap();
            let initial = value(
                NodeKind::ConstantInt(BigInt::from(0)),
                Vec::new(),
                vec![WireType::ConstantInt],
            );
            NodeHandle::sequential_loop(
                body,
                vec![initial, family.clone()],
                vec![WireType::Int],
                SequentialLoop {
                    count: IntExpr::constant(2),
                    index_slot: 0,
                    bindings: if aliased {
                        vec![("i".to_owned(), IntExpr::LoopIndex(0))]
                    } else {
                        Vec::new()
                    },
                    carried_count: 1,
                },
            )
            .output(0)
            .unwrap()
        };

        let static_sequential = graph("static-sequential", sequential(false, false));
        assert_eq!(
            node_message(validate(&static_sequential, &ParamEnv::default()).unwrap_err()),
            "loop-dependent family access must use FamilyGetDynamic"
        );
        validate(&graph("dynamic-sequential", sequential(true, false)), &ParamEnv::default())
            .unwrap();
        assert_eq!(
            node_message(
                validate(
                    &graph("aliased-static-sequential", sequential(false, true)),
                    &ParamEnv::default()
                )
                .unwrap_err()
            ),
            "loop-dependent family access must use FamilyGetDynamic"
        );
    }

    #[test]
    fn loop_dependency_bindings_respect_shadowing_and_transitive_aliases() {
        let parent = BTreeSet::from(["i".to_owned()]);
        assert!(
            child_loop_dependencies(&parent, &[("i".to_owned(), IntExpr::constant(0))]).is_empty()
        );
        assert_eq!(
            child_loop_dependencies(&parent, &[("j".to_owned(), IntExpr::Var("i".to_owned()))]),
            BTreeSet::from(["i".to_owned(), "j".to_owned()])
        );
        assert_eq!(
            child_loop_dependencies(&parent, &[("i".to_owned(), IntExpr::Var("i".to_owned()))]),
            BTreeSet::from(["i".to_owned()])
        );
    }

    #[test]
    fn subgraph_call_requires_one_positive_canonical_upper_per_argument() {
        let body = crate::with_new_construction_scope(|scope| {
            let value = input("body-value", matrix_type(17, 1, 1));
            crate::SubgraphHandle::new("bounded-subgraph", scope, vec![value.clone()], vec![value])
                .expect("subgraph")
        });
        let outer = input("outer-value", matrix_type(17, 1, 1));
        let missing =
            NodeHandle::subgraph_call(body.clone(), vec![outer.clone()], Vec::new(), Vec::new())
                .output(0)
                .expect("output");
        assert_eq!(
            node_message(
                validate(&graph("missing-upper", missing), &ParamEnv::default()).unwrap_err()
            ),
            "canonical input exclusive upper bound count does not match call arguments"
        );
        let zero = NodeHandle::subgraph_call(
            body,
            vec![outer],
            Vec::new(),
            vec![Some(num_bigint::BigUint::zero())],
        )
        .output(0)
        .expect("output");
        assert_eq!(
            node_message(validate(&graph("zero-upper", zero), &ParamEnv::default()).unwrap_err()),
            "canonical input exclusive upper bounds must be positive"
        );
    }

    #[test]
    fn sequential_loop_validation_rejects_a_carried_arity_mismatch() {
        let initial =
            value(NodeKind::ConstantInt(BigInt::from(0)), Vec::new(), vec![WireType::ConstantInt]);
        let (body, scope) = crate::with_new_construction_scope(|scope| {
            let input = NodeHandle::new(
                NodeKind::Input {
                    name: "state".to_owned(),
                    wire_type: WireType::Int,
                    artifact: None,
                },
                Vec::new(),
                vec![WireType::Int],
            )
            .output(0)
            .unwrap();
            (input, scope)
        });
        let body = crate::SubgraphHandle::new(
            "invalid-sequential-body",
            scope,
            vec![body.clone()],
            vec![body],
        )
        .unwrap();
        let output = NodeHandle::sequential_loop(
            body,
            vec![initial],
            vec![WireType::Int],
            SequentialLoop {
                count: IntExpr::constant(1),
                index_slot: 0,
                bindings: Vec::new(),
                carried_count: 2,
            },
        )
        .output(0)
        .unwrap();
        assert_eq!(
            node_message(
                validate(&graph("invalid-sequential", output), &ParamEnv::default()).unwrap_err()
            ),
            "invalid sequential carried count"
        );
    }
}

fn runtime_bounds_warning(node: NodeId, message: &str) -> ElaborationWarning {
    ElaborationWarning {
        node,
        kind: WarningKind::RuntimeSelectBoundsCheck,
        message: message.to_owned(),
    }
}

fn is_integer(ty: &ConcreteWireType) -> bool {
    matches!(ty, ConcreteWireType::Int | ConcreteWireType::ConstantInt)
}

fn structural_type_compatible(actual: &ConcreteWireType, expected: &ConcreteWireType) -> bool {
    actual == expected ||
        matches!(
            (actual, expected),
            (ConcreteWireType::ConstantInt, ConcreteWireType::Int) |
                (ConcreteWireType::ConstantBool, ConcreteWireType::Bool) |
                (ConcreteWireType::ConstantReal, ConcreteWireType::Real)
        )
}

fn is_real(ty: &ConcreteWireType) -> bool {
    matches!(ty, ConcreteWireType::Real | ConcreteWireType::ConstantReal)
}

fn is_boolean(ty: &ConcreteWireType) -> bool {
    matches!(ty, ConcreteWireType::Bool | ConcreteWireType::ConstantBool)
}
