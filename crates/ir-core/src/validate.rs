use crate::{
    artifact::{ArtifactType, Manifest, ManifestArtifact, ProductionId, validate_manifest},
    checks::{
        CheckError, ElaborationWarning, WarningKind, check_add_shape, check_topological,
        multiplication_type,
    },
    expr::{ExprError, IntExpr, ParamEnv},
    graph::{CompileParameterKind, FrozenGraphScopeId, Graph, GraphScope, NodeHandle},
    node::{ConcatAxis, ConstantMatrix, IntBinaryOp, LoopInputMode, MatrixBinaryOp, NodeKind},
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
                let Some(child) = graph.child_scope_id(scope_id, NodeId(position as u64)) else {
                    continue;
                };
                let binding_names = match node.kind() {
                    NodeKind::SubgraphCall(call) => &call.bindings,
                    NodeKind::ParallelLoop(loop_spec) => &loop_spec.bindings,
                    _ => continue,
                };
                let target = declared_by_scope.entry(child).or_default();
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

    for (scope_id, scope) in graph.scopes() {
        let allowed_slots = structural_loop_slots(graph, scope_id);
        for node in scope.nodes() {
            let kind = serde_json::to_value(node.kind()).expect("NodeKind is serializable");
            let mut variables = BTreeSet::new();
            let mut loop_slots = BTreeSet::new();
            collect_expression_references(&kind, &mut variables, &mut loop_slots);
            let declared = &declared_by_scope[scope_id];
            if let Some(name) = variables.into_iter().find(|name| !declared.contains(name)) {
                return Err(ValidationError::UndeclaredCompileVariable(name));
            }
            if let Some(slot) = loop_slots.into_iter().find(|slot| !allowed_slots.contains(slot)) {
                return Err(ValidationError::IllegalLoopIndex { scope: scope_id.clone(), slot });
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
    while let FrozenGraphScopeId::ParallelBody { parent, owner } = current {
        if let Some(NodeKind::ParallelLoop(loop_spec)) =
            graph.scope(parent).and_then(|scope| scope.node(*owner)).map(NodeHandle::kind)
        {
            slots.insert(loop_spec.index_slot);
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
        let validated = validate_scope(scope_id, scope, scope_env, manifests, &mut warnings)?;
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
            NodeKind::ParallelLoop(loop_spec) => {
                let mut loop_env = env.clone();
                loop_env.loop_indices.insert(loop_spec.index_slot, BigInt::zero());
                child_bindings(&loop_env, &loop_spec.bindings)?
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
    scope: &FrozenGraphScopeId,
    node: &NodeView<'_>,
    env: &ParamEnv,
    manifests: &BTreeMap<ProductionId, Manifest>,
    values: &mut BTreeMap<WireRef, ConcreteWireType>,
    artifact_inputs: &mut BTreeMap<WireRef, ManifestArtifact>,
    warnings: &mut Vec<ElaborationWarning>,
) -> Result<(), ValidationError> {
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
                let (element, count) = family_element(&declared);
                if ArtifactType::from_wire_type(element).as_ref() != Some(&stored.artifact_type) ||
                    count != stored.family_count
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
            let gadget_base = base.evaluate(env)?.abs();
            if gadget_base <= BigInt::one() || !matrix.columns.is_multiple_of(matrix.rows) {
                return node_error(scope, node.id, "invalid gadget trapdoor dimensions or base");
            }
            let digit_count = matrix.columns / matrix.rows;
            vec![ConcreteWireType::Trapdoor {
                matrix,
                sigma: crate::RealExpr::FromInt(IntExpr::constant(gadget_base.clone())),
                gadget_base,
                digit_count,
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
        NodeKind::MatrixNegate | NodeKind::MatrixScale { .. } => {
            require_arity(scope, node, 1)?;
            if let NodeKind::MatrixScale { scalar } = node.kind {
                scalar.evaluate(env)?;
            }
            vec![ConcreteWireType::Matrix(matrix_argument(scope, values, node, 0)?)]
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
        NodeKind::Reshape { rows, columns } => {
            require_arity(scope, node, 1)?;
            let input = matrix_argument(scope, values, node, 0)?;
            let rows = positive_usize(rows.evaluate(env)?, "reshape rows", scope, node.id)?;
            let columns =
                positive_usize(columns.evaluate(env)?, "reshape columns", scope, node.id)?;
            if rows.saturating_mul(columns) != input.rows.saturating_mul(input.columns) {
                return node_error(scope, node.id, "reshape changes the element count");
            }
            vec![ConcreteWireType::Matrix(ConcreteMatrixType { rows, columns, ..input })]
        }
        NodeKind::UniformSample { matrix_type, range } => {
            require_arity(scope, node, 0)?;
            if range.minimum.evaluate(env)? > range.maximum.evaluate(env)? {
                return node_error(scope, node.id, "uniform sample range is empty");
            }
            vec![ConcreteWireType::Matrix(concrete_matrix(matrix_type, env, scope, node.id)?)]
        }
        NodeKind::GaussianSample { matrix_type, sigma } => {
            require_arity(scope, node, 0)?;
            require_nonnegative_real(sigma.evaluate_f64(env)?, scope, node.id, "Gaussian sigma")?;
            vec![ConcreteWireType::Matrix(concrete_matrix(matrix_type, env, scope, node.id)?)]
        }
        NodeKind::HashSample {
            matrix_type,
            tag_expressions,
            tag_decimal_expressions,
            tag_u64_le_expressions,
            base,
            digit_count,
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
            if base
                .as_ref()
                .is_some_and(|base| base.evaluate(env).is_ok_and(|v| v.abs() <= BigInt::one()))
            {
                return node_error(scope, node.id, "gadget base must be greater than one");
            }
            if let Some(count) = digit_count {
                positive_usize(count.evaluate(env)?, "decomposition digit count", scope, node.id)?;
            }
            vec![ConcreteWireType::Matrix(concrete_matrix(matrix_type, env, scope, node.id)?)]
        }
        NodeKind::TrapdoorSample { matrix_type, sigma, gadget_base, digit_count } => {
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
                ConcreteWireType::Trapdoor { matrix, sigma, gadget_base, digit_count },
            ]
        }
        NodeKind::PreimageSample { matrix_type } => {
            require_arity(scope, node, 3)?;
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
            check_add_shape(&multiplication_type(&trapdoor, &output)?, &target)?;
            vec![ConcreteWireType::Preimage(output)]
        }
        NodeKind::GadgetDecompose { base, digit_count, .. } => {
            require_arity(scope, node, 1)?;
            let input = matrix_argument(scope, values, node, 0)?;
            let base = base.evaluate(env)?.abs();
            if base <= BigInt::one() {
                return node_error(scope, node.id, "gadget base must be greater than one");
            }
            let digits = decomposition_digits(
                digit_count.as_ref(),
                &input.modulus,
                &base,
                env,
                scope,
                node.id,
            )?;
            vec![ConcreteWireType::Preimage(ConcreteMatrixType {
                rows: input.rows.saturating_mul(digits),
                ..input
            })]
        }
        NodeKind::ExtractCoefficient { position } | NodeKind::ConstantCoefficient { position } => {
            require_arity(scope, node, 1)?;
            let input = matrix_argument(scope, values, node, 0)?;
            let position =
                nonnegative_usize(position.evaluate(env)?, "coefficient position", scope, node.id)?;
            if !input.is_scalar() || position >= input.ring_dimension {
                return node_error(scope, node.id, "coefficient extraction position is invalid");
            }
            if matches!(node.kind, NodeKind::ExtractCoefficient { .. }) {
                vec![ConcreteWireType::Int]
            } else {
                vec![ConcreteWireType::Matrix(input)]
            }
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
        NodeKind::SubgraphCall(_) | NodeKind::ParallelLoop(_) => node
            .output_types
            .iter()
            .map(|ty| concrete_wire(ty, env, scope, node.id))
            .collect::<Result<Vec<_>, _>>()?,
        NodeKind::FamilyPack { count } => {
            let count = positive_usize(count.evaluate(env)?, "family count", scope, node.id)?;
            if node.args.len() != count || count == 0 {
                return node_error(scope, node.id, "family pack argument count mismatch");
            }
            let first = argument(scope, values, node, 0)?.clone();
            if matches!(first, ConcreteWireType::IndexedFamily { .. }) {
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
            vec![ConcreteWireType::IndexedFamily { element: Box::new(first), count }]
        }
        NodeKind::FamilyGetStatic { index } => {
            require_arity(scope, node, 1)?;
            let ConcreteWireType::IndexedFamily { element, count } =
                argument(scope, values, node, 0)?.clone()
            else {
                return node_error(scope, node.id, "family access requires an indexed family");
            };
            if nonnegative_usize(index.evaluate(env)?, "family index", scope, node.id)? >= count {
                return node_error(scope, node.id, "family index is out of range");
            }
            vec![*element]
        }
        NodeKind::FamilyGetDynamic => {
            require_arity(scope, node, 2)?;
            let ConcreteWireType::IndexedFamily { element, .. } =
                argument(scope, values, node, 0)?.clone()
            else {
                return node_error(scope, node.id, "family access requires an indexed family");
            };
            require_scalar(scope, values, node, 1, is_integer, "integer")?;
            warnings.push(runtime_bounds_warning(node.id, "family index is checked at runtime"));
            vec![*element]
        }
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
            let modes = match handle.kind() {
                NodeKind::SubgraphCall(call) => {
                    let _ = child_bindings(env, &call.bindings)?;
                    vec![LoopInputMode::Broadcast; args.len()]
                }
                NodeKind::ParallelLoop(loop_spec) => {
                    let count = nonnegative_usize(
                        loop_spec.count.evaluate(env)?,
                        "parallel count",
                        scope_id,
                        node_id,
                    )?;
                    if count < loop_spec.minimum_count {
                        return node_error(scope_id, node_id, "parallel count is below its minimum");
                    }
                    if loop_spec.input_modes.len() != args.len() {
                        return node_error(scope_id, node_id, "parallel input mode count mismatch");
                    }
                    let mut loop_env = env.clone();
                    loop_env.loop_indices.insert(loop_spec.index_slot, BigInt::zero());
                    let _ = child_bindings(&loop_env, &loop_spec.bindings)?;
                    loop_spec.input_modes.clone()
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
            for ((arg, input), mode) in args.iter().zip(child_scope.inputs()).zip(modes) {
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
                let actual = match mode {
                    LoopInputMode::Broadcast => outer,
                    LoopInputMode::Zip | LoopInputMode::ZipOffset { .. } => {
                        let ConcreteWireType::IndexedFamily { element, count } = outer else {
                            return node_error(scope_id, node_id, "zipped input is not a family");
                        };
                        if let NodeKind::ParallelLoop(spec) = handle.kind() {
                            let iterations = nonnegative_usize(
                                spec.count.evaluate(env)?,
                                "parallel count",
                                scope_id,
                                node_id,
                            )?;
                            let offset = match mode {
                                LoopInputMode::Zip => 0,
                                LoopInputMode::ZipOffset { offset } => offset,
                                _ => 0,
                            };
                            if *count < iterations.saturating_add(offset) {
                                return node_error(scope_id, node_id, "zipped family is too short");
                            }
                        }
                        element.as_ref()
                    }
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
                let expected = if matches!(handle.kind(), NodeKind::ParallelLoop(_)) {
                    let NodeKind::ParallelLoop(spec) = handle.kind() else { unreachable!() };
                    ConcreteWireType::IndexedFamily {
                        element: Box::new(child_type.clone()),
                        count: nonnegative_usize(
                            spec.count.evaluate(env)?,
                            "parallel count",
                            scope_id,
                            node_id,
                        )?,
                    }
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
        WireType::Preimage(matrix) => {
            ConcreteWireType::Preimage(concrete_matrix(matrix, env, scope, node)?)
        }
        WireType::Trapdoor { matrix, sigma, gadget_base, digit_count } => {
            ConcreteWireType::Trapdoor {
                matrix: concrete_matrix(matrix, env, scope, node)?,
                sigma: sigma.close(env)?,
                gadget_base: gadget_base.evaluate(env)?,
                digit_count: positive_usize(
                    digit_count.evaluate(env)?,
                    "digit count",
                    scope,
                    node,
                )?,
            }
        }
        WireType::IndexedFamily { element, count } => {
            let element = concrete_wire(element, env, scope, node)?;
            if matches!(element, ConcreteWireType::IndexedFamily { .. }) {
                return node_error(scope, node, "nested indexed families are unsupported");
            }
            ConcreteWireType::IndexedFamily {
                element: Box::new(element),
                count: nonnegative_usize(count.evaluate(env)?, "family count", scope, node)?,
            }
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

fn decomposition_digits(
    explicit: Option<&IntExpr>,
    modulus: &BigInt,
    base: &BigInt,
    env: &ParamEnv,
    scope: &FrozenGraphScopeId,
    node: NodeId,
) -> Result<usize, ValidationError> {
    if let Some(explicit) = explicit {
        return positive_usize(explicit.evaluate(env)?, "decomposition digit count", scope, node);
    }
    let mut power = BigInt::one();
    let mut digits = 0usize;
    while power < *modulus {
        power *= base;
        digits = digits.saturating_add(1);
    }
    Ok(digits.max(1))
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

fn family_element(ty: &ConcreteWireType) -> (&ConcreteWireType, Option<usize>) {
    match ty {
        ConcreteWireType::IndexedFamily { element, count } => (element, Some(*count)),
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

fn is_real(ty: &ConcreteWireType) -> bool {
    matches!(ty, ConcreteWireType::Real | ConcreteWireType::ConstantReal)
}

fn is_boolean(ty: &ConcreteWireType) -> bool {
    matches!(ty, ConcreteWireType::Bool | ConcreteWireType::ConstantBool)
}
