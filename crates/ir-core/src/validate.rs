use crate::{
    artifact::{ArtifactType, Manifest, ManifestArtifact, ProductionId, validate_manifest},
    checks::{
        CheckError, ElaborationWarning, WarningKind, check_add_shape, check_topological,
        multiplication_type,
    },
    expr::{ExprError, IntExpr, ParamEnv},
    graph::{CompileParameterKind, Graph},
    node::{
        ConcatAxis, ConstantMatrix, IntBinaryOp, LoopInputMode, MatrixBinaryOp, Node, NodeKind,
        RealBinaryOp,
    },
    types::{
        ConcreteMatrixType, ConcreteWireType, InstantiationFrame, MatrixType, NodeId, Port, WireId,
        WireRef, WireType,
    },
};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValidatedGraph {
    pub source: Graph,
    pub bindings: ParamEnv,
    pub wires: BTreeMap<WireId, ConcreteWireType>,
    /// Exact manifest descriptors approved for artifact-backed input wires.
    ///
    /// Runtime consumers must use these descriptors rather than reconstructing
    /// weaker descriptors from the declared wire type.
    pub artifact_inputs: BTreeMap<WireId, ManifestArtifact>,
    pub outputs: BTreeMap<String, WireRef>,
    pub warnings: Vec<ElaborationWarning>,
}

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("node {node:?} at instantiation path {instantiation_path:?}: {source}")]
    Context {
        node: NodeId,
        instantiation_path: Vec<InstantiationFrame>,
        #[source]
        source: Box<ValidationError>,
    },
    #[error(transparent)]
    Expression(#[from] ExprError),
    #[error(transparent)]
    Check(#[from] CheckError),
    #[error("missing compile binding: {0}")]
    MissingBinding(String),
    #[error("node {node:?} refers to unavailable wire {wire:?}")]
    MissingWire { node: NodeId, wire: WireRef },
    #[error("node {0:?} requires a matrix argument")]
    ExpectedMatrix(NodeId),
    #[error("node {0:?} requires a trapdoor argument")]
    ExpectedTrapdoor(NodeId),
    #[error("artifact production is unavailable: {0:?}")]
    MissingManifest(ProductionId),
    #[error(
        "artifact manifest uses Graph IR version {actual}, but this runtime requires version {expected}"
    )]
    ManifestVersion { expected: u32, actual: u32 },
    #[error(
        "artifact manifest map key {key:?} does not match the manifest production id {manifest:?}"
    )]
    ManifestProductionMismatch { key: ProductionId, manifest: ProductionId },
    #[error("artifact manifest {production:?} is invalid: {message}")]
    InvalidManifest { production: ProductionId, message: String },
    #[error("artifact {artifact} is unavailable in production {production:?}")]
    MissingArtifact { production: ProductionId, artifact: String },
    #[error("node {node:?}: {message}")]
    Node { node: NodeId, message: String },
}

pub fn validate(graph: &Graph, bindings: &ParamEnv) -> Result<ValidatedGraph, ValidationError> {
    validate_with_manifests(graph, bindings, &BTreeMap::new())
}

pub fn validate_with_manifests(
    graph: &Graph,
    bindings: &ParamEnv,
    manifests: &BTreeMap<ProductionId, Manifest>,
) -> Result<ValidatedGraph, ValidationError> {
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
    check_bindings(graph, bindings)?;
    check_topological(graph)?;
    let mut validator = Validator {
        manifests,
        wires: BTreeMap::new(),
        artifact_inputs: BTreeMap::new(),
        warnings: Vec::new(),
    };
    let root = validator.validate_instance(graph, bindings, Vec::new(), &BTreeMap::new())?;
    for (name, wire) in &graph.outputs {
        if !root.values.contains_key(wire) {
            return Err(ValidationError::Node {
                node: wire.node,
                message: format!("graph output {name} refers to an unavailable port"),
            });
        }
        if let Some(Node { kind: NodeKind::Output { name: declared_name, .. }, .. }) =
            graph.node(wire.node) &&
            declared_name != name
        {
            return Err(ValidationError::Node {
                node: wire.node,
                message: format!(
                    "graph output key {name} does not match Output node name {declared_name}"
                ),
            });
        }
    }
    Ok(ValidatedGraph {
        source: graph.clone(),
        bindings: bindings.clone(),
        wires: validator.wires,
        artifact_inputs: validator.artifact_inputs,
        outputs: graph.outputs.clone(),
        warnings: validator.warnings,
    })
}

struct Instance {
    values: BTreeMap<WireRef, ConcreteWireType>,
}

struct Validator<'a> {
    manifests: &'a BTreeMap<ProductionId, Manifest>,
    wires: BTreeMap<WireId, ConcreteWireType>,
    artifact_inputs: BTreeMap<WireId, ManifestArtifact>,
    warnings: Vec<ElaborationWarning>,
}

impl Validator<'_> {
    fn validate_instance(
        &mut self,
        graph: &Graph,
        bindings: &ParamEnv,
        path: Vec<InstantiationFrame>,
        input_overrides: &BTreeMap<String, ConcreteWireType>,
    ) -> Result<Instance, ValidationError> {
        check_bindings(graph, bindings)?;
        check_topological(graph).map_err(|error| contextualize_check(error, &path))?;
        let mut values = BTreeMap::new();
        for node in &graph.nodes {
            self.validate_node(graph, bindings, &path, input_overrides, &mut values, node)
                .map_err(|error| contextualize_unless_present(node.id, &path, error))?;
        }
        Ok(Instance { values })
    }

    fn validate_node(
        &mut self,
        graph: &Graph,
        bindings: &ParamEnv,
        path: &[InstantiationFrame],
        input_overrides: &BTreeMap<String, ConcreteWireType>,
        values: &mut BTreeMap<WireRef, ConcreteWireType>,
        node: &Node,
    ) -> Result<(), ValidationError> {
        match &node.kind {
            NodeKind::Input { name, wire_type, artifact } => {
                if let Some(value) = input_overrides.get(name) {
                    let declared = concrete_wire(wire_type, bindings)?;
                    if value != &declared {
                        return self.node_error(
                            node.id,
                            "subgraph input binding type does not match its declaration",
                        );
                    }
                    self.insert(values, path, node.id, 0, value.clone());
                    return Ok(());
                }
                if let Some(artifact) = artifact {
                    let manifest =
                        self.manifests.get(&artifact.production_id).ok_or_else(|| {
                            ValidationError::MissingManifest(artifact.production_id.clone())
                        })?;
                    if manifest.ir_version != crate::encoding::IR_VERSION {
                        return Err(ValidationError::ManifestVersion {
                            expected: crate::encoding::IR_VERSION,
                            actual: manifest.ir_version,
                        });
                    }
                    if manifest.production_id != artifact.production_id {
                        return Err(ValidationError::ManifestProductionMismatch {
                            key: artifact.production_id.clone(),
                            manifest: manifest.production_id.clone(),
                        });
                    }
                    let stored =
                        manifest.artifacts.get(&artifact.artifact_name).ok_or_else(|| {
                            ValidationError::MissingArtifact {
                                production: artifact.production_id.clone(),
                                artifact: artifact.artifact_name.clone(),
                            }
                        })?;
                    let declared = concrete_wire(wire_type, bindings)?;
                    if artifact.confidentiality != stored.confidentiality {
                        return self.node_error(
                            node.id,
                            "artifact confidentiality does not match manifest",
                        );
                    }
                    let (element, family_count) = match &declared {
                        ConcreteWireType::IndexedFamily { element, count } => {
                            (element.as_ref(), Some(*count))
                        }
                        scalar => (scalar, None),
                    };
                    let declared_artifact_type =
                        ArtifactType::from_wire_type(element).ok_or_else(|| {
                            ValidationError::Node {
                                node: node.id,
                                message: "artifact input has an unsupported wire type".to_owned(),
                            }
                        })?;
                    if declared_artifact_type != stored.artifact_type {
                        return self
                            .node_error(node.id, "declared artifact type does not match manifest");
                    }
                    if family_count != stored.family_count {
                        return self.node_error(node.id, "artifact family count mismatch");
                    }
                    self.artifact_inputs.insert(
                        WireId {
                            instantiation_path: path.to_vec(),
                            wire: WireRef { node: node.id, port: Port(0) },
                        },
                        stored.clone(),
                    );
                    self.insert(values, path, node.id, 0, declared);
                } else {
                    self.insert(values, path, node.id, 0, concrete_wire(wire_type, bindings)?);
                }
            }
            NodeKind::Output { artifact_confidentiality, .. } => {
                if node.args.len() != 1 {
                    return self.node_error(node.id, "output requires exactly one argument");
                }
                let value = self.argument(values, node, 0)?.clone();
                if artifact_confidentiality.is_some() {
                    let element = match &value {
                        ConcreteWireType::IndexedFamily { element, .. } => element.as_ref(),
                        scalar => scalar,
                    };
                    if ArtifactType::from_wire_type(element).is_none() {
                        return self.node_error(
                            node.id,
                            "persisted output has an unsupported artifact type",
                        );
                    }
                }
                self.insert(values, path, node.id, 0, value);
            }
            NodeKind::ConstantInt(_) => {
                self.insert(values, path, node.id, 0, ConcreteWireType::ConstantInt);
            }
            NodeKind::EvaluateInt(value) => {
                value.evaluate(bindings)?;
                self.insert(values, path, node.id, 0, ConcreteWireType::ConstantInt);
            }
            NodeKind::ConstantReal(value) => {
                value.evaluate_f64(bindings)?;
                self.insert(values, path, node.id, 0, ConcreteWireType::ConstantReal);
            }
            NodeKind::ConstantBool(_) => {
                self.insert(values, path, node.id, 0, ConcreteWireType::ConstantBool);
            }
            NodeKind::ConstantMatrix { matrix_type, value } => {
                let matrix = concrete_matrix(matrix_type, bindings)?;
                validate_constant(value, &matrix, bindings, node.id)?;
                self.insert(values, path, node.id, 0, ConcreteWireType::Matrix(matrix));
            }
            NodeKind::GadgetTrapdoor { matrix_type, base } => {
                let matrix = concrete_matrix(matrix_type, bindings)?;
                let gadget_base = base.evaluate(bindings)?.abs();
                if gadget_base <= BigInt::one() {
                    return self.node_error(node.id, "gadget base must be greater than one");
                }
                if !matrix.columns.is_multiple_of(matrix.rows) {
                    return self
                        .node_error(node.id, "gadget trapdoor columns must be divisible by rows");
                }
                let digit_count = matrix.columns / matrix.rows;
                self.insert(
                    values,
                    path,
                    node.id,
                    0,
                    ConcreteWireType::Trapdoor {
                        matrix,
                        sigma: crate::expr::RealExpr::FromInt(IntExpr::constant(
                            gadget_base.clone(),
                        )),
                        gadget_base,
                        digit_count,
                    },
                );
            }
            NodeKind::TrapdoorPublic => {
                let trapdoor = self.argument(values, node, 0)?;
                let ConcreteWireType::Trapdoor { matrix, .. } = trapdoor else {
                    return self
                        .node_error(node.id, "trapdoor public projection requires a trapdoor");
                };
                self.insert(values, path, node.id, 0, ConcreteWireType::Matrix(matrix.clone()));
            }
            NodeKind::IntBinary(operation) => {
                self.require_scalar(values, node, 0, is_integer, "integer")?;
                self.require_scalar(values, node, 1, is_integer, "integer")?;
                if matches!(operation, IntBinaryOp::Divide | IntBinaryOp::Remainder) {
                    // Zero is a defined runtime error, so validation only checks types.
                }
                self.insert(values, path, node.id, 0, ConcreteWireType::Int);
            }
            NodeKind::IntCompare(_) => {
                self.require_scalar(values, node, 0, is_integer, "integer")?;
                self.require_scalar(values, node, 1, is_integer, "integer")?;
                self.insert(values, path, node.id, 0, ConcreteWireType::Bool);
            }
            NodeKind::BitExtract { bit } => {
                self.require_scalar(values, node, 0, is_integer, "integer")?;
                if bit.evaluate(bindings)?.is_negative() {
                    return self.node_error(node.id, "bit position must be nonnegative");
                }
                self.insert(values, path, node.id, 0, ConcreteWireType::Bool);
            }
            NodeKind::IntToReal => {
                self.require_scalar(values, node, 0, is_integer, "integer")?;
                self.insert(values, path, node.id, 0, ConcreteWireType::Real);
            }
            NodeKind::BoolToInt => {
                self.require_scalar(values, node, 0, is_boolean, "boolean")?;
                self.insert(values, path, node.id, 0, ConcreteWireType::Int);
            }
            NodeKind::RealBinary(operation) => {
                self.require_scalar(values, node, 0, is_real, "real")?;
                self.require_scalar(values, node, 1, is_real, "real")?;
                if matches!(operation, RealBinaryOp::Divide) {
                    // Zero is a defined runtime error.
                }
                self.insert(values, path, node.id, 0, ConcreteWireType::Real);
            }
            NodeKind::RealSqrt => {
                self.require_scalar(values, node, 0, is_real, "real")?;
                self.insert(values, path, node.id, 0, ConcreteWireType::Real);
            }
            NodeKind::MatrixBinary(operation) => {
                let left = self.matrix_argument(values, node, 0)?;
                let right = self.matrix_argument(values, node, 1)?;
                let output = match operation {
                    MatrixBinaryOp::Add | MatrixBinaryOp::Subtract => {
                        check_add_shape(&left, &right)?;
                        left
                    }
                    MatrixBinaryOp::Multiply => multiplication_type(&left, &right)?,
                };
                self.insert(values, path, node.id, 0, ConcreteWireType::Matrix(output));
            }
            NodeKind::MatrixNegate | NodeKind::MatrixScale { .. } => {
                if let NodeKind::MatrixScale { scalar } = &node.kind {
                    scalar.evaluate(bindings)?;
                }
                let input = self.matrix_argument(values, node, 0)?;
                self.insert(values, path, node.id, 0, ConcreteWireType::Matrix(input));
            }
            NodeKind::Transpose => {
                let input = self.matrix_argument(values, node, 0)?;
                self.insert(
                    values,
                    path,
                    node.id,
                    0,
                    ConcreteWireType::Matrix(ConcreteMatrixType {
                        rows: input.columns,
                        columns: input.rows,
                        ..input
                    }),
                );
            }
            NodeKind::Slice { rows, columns } => {
                let input = self.matrix_argument(values, node, 0)?;
                let output = sliced_type(&input, rows.as_ref(), columns.as_ref(), node.id)?;
                self.insert(values, path, node.id, 0, ConcreteWireType::Matrix(output));
            }
            NodeKind::Tensor => {
                let left = self.matrix_argument(values, node, 0)?;
                let right = self.matrix_argument(values, node, 1)?;
                crate::checks::check_same_ring(&left, &right)?;
                self.insert(
                    values,
                    path,
                    node.id,
                    0,
                    ConcreteWireType::Matrix(ConcreteMatrixType {
                        modulus: left.modulus,
                        ring_dimension: left.ring_dimension,
                        rows: left.rows.saturating_mul(right.rows),
                        columns: left.columns.saturating_mul(right.columns),
                    }),
                );
            }
            NodeKind::Concat { axis } => {
                let inputs = (0..node.args.len())
                    .map(|index| self.matrix_argument(values, node, index))
                    .collect::<Result<Vec<_>, _>>()?;
                let output = concat_type(&inputs, *axis, node.id)?;
                self.insert(values, path, node.id, 0, ConcreteWireType::Matrix(output));
            }
            NodeKind::Reshape { rows, columns } => {
                let input = self.matrix_argument(values, node, 0)?;
                let rows = positive_usize(rows.evaluate(bindings)?, "reshape rows", node.id)?;
                let columns =
                    positive_usize(columns.evaluate(bindings)?, "reshape columns", node.id)?;
                if rows.saturating_mul(columns) != input.rows.saturating_mul(input.columns) {
                    return self.node_error(node.id, "reshape changes the element count");
                }
                self.insert(
                    values,
                    path,
                    node.id,
                    0,
                    ConcreteWireType::Matrix(ConcreteMatrixType { rows, columns, ..input }),
                );
            }
            NodeKind::UniformSample { matrix_type, range } => {
                if range.minimum > range.maximum {
                    return self.node_error(node.id, "uniform sample range is empty");
                }
                self.insert(
                    values,
                    path,
                    node.id,
                    0,
                    ConcreteWireType::Matrix(concrete_matrix(matrix_type, bindings)?),
                );
            }
            NodeKind::GaussianSample { matrix_type, sigma } => {
                require_nonnegative_real(sigma.evaluate_f64(bindings)?, node.id, "Gaussian sigma")?;
                self.insert(
                    values,
                    path,
                    node.id,
                    0,
                    ConcreteWireType::Matrix(concrete_matrix(matrix_type, bindings)?),
                );
            }
            NodeKind::HashSample {
                matrix_type,
                variant: _,
                tag_prefix: _,
                tag_expressions,
                tag_decimal_expressions,
                tag_u64_le_expressions,
                base,
                digit_count,
            } => {
                let key = self.argument(values, node, 0)?;
                if *key != (ConcreteWireType::Bytes { length: 32 }) {
                    return self.node_error(node.id, "hash sampling requires a 32-byte key");
                }
                for index in 1..node.args.len() {
                    self.require_scalar(values, node, index, is_integer, "integer")?;
                }
                for expression in tag_expressions {
                    expression.evaluate(bindings)?;
                }
                for expression in tag_decimal_expressions {
                    expression.evaluate(bindings)?;
                }
                for expression in tag_u64_le_expressions {
                    if expression.evaluate(bindings)?.to_u64().is_none() {
                        return self.node_error(
                            node.id,
                            "little-endian hash tag component must fit in u64",
                        );
                    }
                }
                if let Some(base) = base &&
                    base.evaluate(bindings)?.abs() <= BigInt::one()
                {
                    return self.node_error(node.id, "gadget base must be greater than one");
                }
                if let Some(digit_count) = digit_count {
                    positive_usize(
                        digit_count.evaluate(bindings)?,
                        "decomposition digit count",
                        node.id,
                    )?;
                }
                self.insert(
                    values,
                    path,
                    node.id,
                    0,
                    ConcreteWireType::Matrix(concrete_matrix(matrix_type, bindings)?),
                );
            }
            NodeKind::TrapdoorSample { matrix_type, sigma, gadget_base, digit_count } => {
                let sigma = sigma.close(bindings)?;
                require_positive_real(
                    sigma.evaluate_f64(&ParamEnv::default())?,
                    node.id,
                    "trapdoor sigma",
                )?;
                let gadget_base = gadget_base.evaluate(bindings)?.abs();
                if gadget_base <= BigInt::one() {
                    return self.node_error(node.id, "gadget base must be greater than one");
                }
                let digit_count = positive_usize(
                    digit_count.evaluate(bindings)?,
                    "trapdoor digit count",
                    node.id,
                )?;
                let matrix = concrete_matrix(matrix_type, bindings)?;
                let expected_columns = matrix
                    .rows
                    .checked_mul(digit_count.checked_add(2).ok_or_else(|| {
                        ValidationError::Node {
                            node: node.id,
                            message: "trapdoor public matrix width overflow".to_owned(),
                        }
                    })?)
                    .ok_or_else(|| ValidationError::Node {
                        node: node.id,
                        message: "trapdoor public matrix width overflow".to_owned(),
                    })?;
                if matrix.columns != expected_columns {
                    return self.node_error(
                        node.id,
                        "trapdoor public matrix columns must equal rows * (digit_count + 2)",
                    );
                }
                self.insert(values, path, node.id, 0, ConcreteWireType::Matrix(matrix.clone()));
                self.insert(
                    values,
                    path,
                    node.id,
                    1,
                    ConcreteWireType::Trapdoor { matrix, sigma, gadget_base, digit_count },
                );
            }
            NodeKind::PreimageSample { matrix_type } => {
                let trapdoor = self.trapdoor_argument(values, node, 0)?;
                let target = self.matrix_argument(values, node, 1)?;
                let output = concrete_matrix(matrix_type, bindings)?;
                let product = multiplication_type(&trapdoor, &output)?;
                check_add_shape(&product, &target)?;
                self.insert(values, path, node.id, 0, ConcreteWireType::Preimage(output));
            }
            NodeKind::GadgetDecompose { base, small: _, digit_count } => {
                let input = self.matrix_argument(values, node, 0)?;
                let base = base.evaluate(bindings)?.abs();
                if base <= BigInt::one() {
                    return self.node_error(node.id, "gadget base must be greater than one");
                }
                let digits = decomposition_digits(
                    digit_count.as_ref(),
                    &input.modulus,
                    &base,
                    bindings,
                    node.id,
                )?;
                let output = ConcreteMatrixType {
                    rows: input.rows.saturating_mul(digits),
                    columns: input.columns,
                    ..input
                };
                self.insert(values, path, node.id, 0, ConcreteWireType::Preimage(output));
            }
            NodeKind::ModDown { target_modulus } | NodeKind::ModUp { target_modulus } => {
                let input = self.matrix_argument(values, node, 0)?;
                let target = target_modulus.evaluate(bindings)?;
                if target <= BigInt::one() {
                    return self.node_error(node.id, "target modulus must be greater than one");
                }
                match &node.kind {
                    NodeKind::ModDown { .. } if target >= input.modulus => {
                        return self.node_error(node.id, "mod-down target must be smaller");
                    }
                    NodeKind::ModUp { .. } if target <= input.modulus => {
                        return self.node_error(node.id, "mod-up target must be larger");
                    }
                    _ => {}
                }
                self.insert(
                    values,
                    path,
                    node.id,
                    0,
                    ConcreteWireType::Matrix(ConcreteMatrixType { modulus: target, ..input }),
                );
            }
            NodeKind::ExtractCoefficient { position } => {
                let input = self.matrix_argument(values, node, 0)?;
                if !input.is_scalar() {
                    return self.node_error(node.id, "extract coefficient requires a 1x1 matrix");
                }
                let position = position.evaluate(bindings)?;
                if position.is_negative() ||
                    position.to_usize().is_none_or(|position| position >= input.ring_dimension)
                {
                    return self.node_error(node.id, "coefficient position is out of range");
                }
                self.insert(values, path, node.id, 0, ConcreteWireType::Int);
            }
            NodeKind::ConstantCoefficient { position } => {
                let input = self.matrix_argument(values, node, 0)?;
                if !input.is_scalar() {
                    return self.node_error(node.id, "constant coefficient requires a 1x1 matrix");
                }
                let position = position.evaluate(bindings)?;
                if position.is_negative() ||
                    position.to_usize().is_none_or(|position| position >= input.ring_dimension)
                {
                    return self.node_error(node.id, "coefficient position is out of range");
                }
                self.insert(values, path, node.id, 0, ConcreteWireType::Matrix(input));
            }
            NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
                let input = self.matrix_argument(values, node, 0)?;
                if !input.is_scalar() {
                    return self.node_error(node.id, "threshold decode requires a 1x1 matrix");
                }
                if plaintext_modulus.evaluate(bindings)? <= BigInt::one() {
                    return self.node_error(node.id, "plaintext modulus must be greater than one");
                }
                let count = positive_usize(length.evaluate(bindings)?, "decode length", node.id)?;
                if count > input.ring_dimension {
                    return self.node_error(node.id, "decode length exceeds ring dimension");
                }
                let output =
                    if *output_bool { ConcreteWireType::Bool } else { ConcreteWireType::Int };
                for port in 0..count {
                    self.insert(values, path, node.id, port as u32, output.clone());
                }
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => {
                if node.args.is_empty() ||
                    node.args.len() != plaintext_moduli.len() ||
                    node.args.len() != reconstruction_coefficients.len()
                {
                    return self.node_error(
                        node.id,
                        "CRT recomposition requires one modulus and reconstruction coefficient per input",
                    );
                }
                let first = self.matrix_argument(values, node, 0)?;
                if first.rows != 1 {
                    return self
                        .node_error(node.id, "CRT recomposition inputs must be one-row matrices");
                }
                for index in 1..node.args.len() {
                    if self.matrix_argument(values, node, index)? != first {
                        return self.node_error(
                            node.id,
                            "CRT recomposition inputs must have identical matrix types",
                        );
                    }
                }
                let full_modulus = first.modulus.clone();
                for modulus in plaintext_moduli {
                    let modulus = modulus.evaluate(bindings)?;
                    if modulus <= BigInt::one() || modulus >= full_modulus {
                        return self.node_error(
                            node.id,
                            "CRT plaintext moduli must be between one and the full modulus",
                        );
                    }
                }
                for coefficient in reconstruction_coefficients {
                    let coefficient = coefficient.evaluate(bindings)?;
                    if coefficient.sign() == num_bigint::Sign::Minus || coefficient >= full_modulus
                    {
                        return self.node_error(
                            node.id,
                            "CRT reconstruction coefficients must be full-modulus residues",
                        );
                    }
                }
                self.insert(values, path, node.id, 0, ConcreteWireType::Matrix(first));
            }
            NodeKind::SubgraphCall(call) => {
                let child =
                    graph.subgraphs.get(&call.graph).ok_or_else(|| ValidationError::Node {
                        node: node.id,
                        message: format!("subgraph {} does not exist", call.graph),
                    })?;
                let child_bindings = child_bindings(bindings, &call.bindings)?;
                let overrides = input_overrides_for(child, node, values)?;
                let mut child_path = path.to_vec();
                child_path.push(InstantiationFrame { call: node.id, loop_index: None });
                let instance =
                    self.validate_instance(child, &child_bindings, child_path, &overrides)?;
                self.insert_child_outputs(values, path, node.id, child, &instance)?;
            }
            NodeKind::ParallelLoop(loop_node) => {
                let child =
                    graph.subgraphs.get(&loop_node.graph).ok_or_else(|| ValidationError::Node {
                        node: node.id,
                        message: format!("parallel-loop body {} does not exist", loop_node.graph),
                    })?;
                let count =
                    nonnegative_usize(loop_node.count.evaluate(bindings)?, "loop count", node.id)?;
                if count < loop_node.minimum_count {
                    let message = format!(
                        "loop count must be at least {}, got {count}",
                        loop_node.minimum_count
                    );
                    return self.node_error(node.id, &message);
                }
                let overrides =
                    loop_input_overrides(child, node, values, &loop_node.input_modes, count)?;
                let tainted = loop_tainted_variables(loop_node);
                ensure_loop_structure_independent(child, &tainted, node.id)?;
                let mut iteration_bindings = bindings.clone();
                iteration_bindings
                    .integers
                    .insert(loop_node.index_variable.clone(), BigInt::zero());
                let child_bindings = child_bindings(&iteration_bindings, &loop_node.bindings)?;
                let mut child_path = path.to_vec();
                child_path.push(InstantiationFrame { call: node.id, loop_index: Some(0) });
                let instance =
                    self.validate_instance(child, &child_bindings, child_path, &overrides)?;
                for (port, (_, wire)) in child.outputs.iter().enumerate() {
                    let element = instance
                        .values
                        .get(wire)
                        .cloned()
                        .ok_or(ValidationError::MissingWire { node: node.id, wire: *wire })?;
                    if matches!(element, ConcreteWireType::IndexedFamily { .. }) {
                        return self
                            .node_error(node.id, "nested indexed families are not supported");
                    }
                    self.insert(
                        values,
                        path,
                        node.id,
                        port as u32,
                        ConcreteWireType::IndexedFamily { element: Box::new(element), count },
                    );
                }
            }
            NodeKind::FamilyPack { count } => {
                let count = positive_usize(count.evaluate(bindings)?, "family count", node.id)?;
                if node.args.len() != count {
                    return self.node_error(node.id, "family pack argument count mismatch");
                }
                let element = self.argument(values, node, 0)?.clone();
                if matches!(element, ConcreteWireType::IndexedFamily { .. }) {
                    return self.node_error(node.id, "nested indexed families are not supported");
                }
                for index in 1..node.args.len() {
                    if self.argument(values, node, index)? != &element {
                        return self.node_error(node.id, "family members must have identical types");
                    }
                }
                self.insert(
                    values,
                    path,
                    node.id,
                    0,
                    ConcreteWireType::IndexedFamily { element: Box::new(element), count },
                );
            }
            NodeKind::FamilyGetStatic { index } => {
                let family = self.argument(values, node, 0)?.clone();
                let ConcreteWireType::IndexedFamily { element, count } = family else {
                    return self.node_error(node.id, "family access requires an indexed family");
                };
                let index = nonnegative_usize(index.evaluate(bindings)?, "family index", node.id)?;
                if index >= count {
                    return self.node_error(node.id, "family index is out of range");
                }
                self.insert(values, path, node.id, 0, *element);
            }
            NodeKind::FamilyGetDynamic => {
                let family = self.argument(values, node, 0)?.clone();
                let ConcreteWireType::IndexedFamily { element, .. } = family else {
                    return self.node_error(node.id, "family access requires an indexed family");
                };
                self.require_scalar(values, node, 1, is_integer, "integer")?;
                self.warnings.push(ElaborationWarning {
                    node: node.id,
                    kind: WarningKind::RuntimeSelectBoundsCheck,
                    message: "family index is checked at runtime".to_owned(),
                });
                self.insert(values, path, node.id, 0, *element);
            }
            NodeKind::Select { count } => {
                self.require_scalar(values, node, 0, is_integer, "integer")?;
                let count =
                    positive_usize(count.evaluate(bindings)?, "select branch count", node.id)?;
                if node.args.len() != count.saturating_add(1) {
                    return self.node_error(node.id, "select branch count does not match arguments");
                }
                let first = self.argument(values, node, 1)?.clone();
                for index in 2..node.args.len() {
                    if self.argument(values, node, index)? != &first {
                        return self.node_error(node.id, "select branches have different types");
                    }
                }
                self.warnings.push(ElaborationWarning {
                    node: node.id,
                    kind: WarningKind::RuntimeSelectBoundsCheck,
                    message: "select index is checked at runtime".to_owned(),
                });
                self.insert(values, path, node.id, 0, first);
            }
        }
        Ok(())
    }

    fn insert_child_outputs(
        &mut self,
        values: &mut BTreeMap<WireRef, ConcreteWireType>,
        path: &[InstantiationFrame],
        node: NodeId,
        child: &Graph,
        instance: &Instance,
    ) -> Result<(), ValidationError> {
        for (port, (_, wire)) in child.outputs.iter().enumerate() {
            let ty = instance
                .values
                .get(wire)
                .cloned()
                .ok_or(ValidationError::MissingWire { node, wire: *wire })?;
            self.insert(values, path, node, port as u32, ty);
        }
        Ok(())
    }

    fn insert(
        &mut self,
        values: &mut BTreeMap<WireRef, ConcreteWireType>,
        path: &[InstantiationFrame],
        node: NodeId,
        port: u32,
        value: ConcreteWireType,
    ) {
        let wire = WireRef { node, port: Port(port) };
        values.insert(wire, value.clone());
        self.wires.insert(WireId { instantiation_path: path.to_vec(), wire }, value);
    }

    fn argument<'a>(
        &self,
        values: &'a BTreeMap<WireRef, ConcreteWireType>,
        node: &Node,
        index: usize,
    ) -> Result<&'a ConcreteWireType, ValidationError> {
        let wire = *node.args.get(index).ok_or(ValidationError::MissingWire {
            node: node.id,
            wire: WireRef { node: node.id, port: Port(index as u32) },
        })?;
        values.get(&wire).ok_or(ValidationError::MissingWire { node: node.id, wire })
    }

    fn matrix_argument(
        &self,
        values: &BTreeMap<WireRef, ConcreteWireType>,
        node: &Node,
        index: usize,
    ) -> Result<ConcreteMatrixType, ValidationError> {
        match self.argument(values, node, index)? {
            ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                Ok(matrix.clone())
            }
            _ => Err(ValidationError::ExpectedMatrix(node.id)),
        }
    }

    fn trapdoor_argument(
        &self,
        values: &BTreeMap<WireRef, ConcreteWireType>,
        node: &Node,
        index: usize,
    ) -> Result<ConcreteMatrixType, ValidationError> {
        match self.argument(values, node, index)? {
            ConcreteWireType::Trapdoor { matrix, .. } => Ok(matrix.clone()),
            _ => Err(ValidationError::ExpectedTrapdoor(node.id)),
        }
    }

    fn require_scalar(
        &self,
        values: &BTreeMap<WireRef, ConcreteWireType>,
        node: &Node,
        index: usize,
        predicate: fn(&ConcreteWireType) -> bool,
        expected: &str,
    ) -> Result<(), ValidationError> {
        if predicate(self.argument(values, node, index)?) {
            Ok(())
        } else {
            self.node_error(node.id, &format!("expected {expected} scalar argument"))
        }
    }

    fn node_error<T>(&self, node: NodeId, message: &str) -> Result<T, ValidationError> {
        Err(ValidationError::Node { node, message: message.to_owned() })
    }
}

fn input_overrides_for(
    child: &Graph,
    call: &Node,
    values: &BTreeMap<WireRef, ConcreteWireType>,
) -> Result<BTreeMap<String, ConcreteWireType>, ValidationError> {
    let inputs = child
        .nodes
        .iter()
        .filter_map(|node| match &node.kind {
            NodeKind::Input { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if inputs.len() != call.args.len() {
        return Err(ValidationError::Node {
            node: call.id,
            message: "subgraph input count does not match call arguments".to_owned(),
        });
    }
    inputs
        .into_iter()
        .zip(&call.args)
        .map(|(name, wire)| {
            values
                .get(wire)
                .cloned()
                .map(|value| (name, value))
                .ok_or(ValidationError::MissingWire { node: call.id, wire: *wire })
        })
        .collect()
}

fn loop_input_overrides(
    child: &Graph,
    call: &Node,
    values: &BTreeMap<WireRef, ConcreteWireType>,
    modes: &[LoopInputMode],
    count: usize,
) -> Result<BTreeMap<String, ConcreteWireType>, ValidationError> {
    let inputs = child
        .nodes
        .iter()
        .filter_map(|node| match &node.kind {
            NodeKind::Input { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if inputs.len() != call.args.len() || modes.len() != call.args.len() {
        return Err(ValidationError::Node {
            node: call.id,
            message: "parallel-loop input modes do not match body inputs".to_owned(),
        });
    }
    inputs
        .into_iter()
        .zip(&call.args)
        .zip(modes)
        .map(|((name, wire), mode)| {
            let value = values
                .get(wire)
                .cloned()
                .ok_or(ValidationError::MissingWire { node: call.id, wire: *wire })?;
            let value = match mode {
                LoopInputMode::Broadcast => value,
                LoopInputMode::Zip | LoopInputMode::ZipOffset { .. } => {
                    let ConcreteWireType::IndexedFamily { element, count: family_count } = value
                    else {
                        return Err(ValidationError::Node {
                            node: call.id,
                            message: "zipped parallel-loop input is not an indexed family"
                                .to_owned(),
                        });
                    };
                    let offset = match mode {
                        LoopInputMode::Zip => 0,
                        LoopInputMode::ZipOffset { offset } => *offset,
                        LoopInputMode::Broadcast => unreachable!(),
                    };
                    let required =
                        count.checked_add(offset).ok_or_else(|| ValidationError::Node {
                            node: call.id,
                            message: "zipped parallel-loop input range overflow".to_owned(),
                        })?;
                    if (offset == 0 && family_count != count) ||
                        (offset != 0 && family_count < required)
                    {
                        return Err(ValidationError::Node {
                            node: call.id,
                            message: "zipped parallel-loop input count mismatch".to_owned(),
                        });
                    }
                    *element
                }
            };
            Ok((name, value))
        })
        .collect()
}

fn loop_tainted_variables(loop_node: &crate::node::ParallelLoop) -> BTreeSet<String> {
    let mut tainted = BTreeSet::from([loop_node.index_variable.clone()]);
    for (name, expression) in &loop_node.bindings {
        if tainted.iter().any(|variable| expression.contains_variable(variable)) {
            tainted.insert(name.clone());
        }
    }
    tainted
}

fn ensure_loop_structure_independent(
    graph: &Graph,
    tainted: &BTreeSet<String>,
    loop_node: NodeId,
) -> Result<(), ValidationError> {
    let reject = |description: &str| {
        Err(ValidationError::Node {
            node: loop_node,
            message: format!("parallel-loop index affects structural field {description}"),
        })
    };
    for wire_type in graph.input_types.values() {
        if wire_type_depends_on(wire_type, tainted) {
            return reject("input type");
        }
    }
    for node in &graph.nodes {
        let depends = match &node.kind {
            NodeKind::Input { wire_type, .. } => wire_type_depends_on(wire_type, tainted),
            NodeKind::ConstantMatrix { matrix_type, value } => {
                matrix_type_depends_on(matrix_type, tainted) ||
                    match value {
                        ConstantMatrix::UnitRow { index } |
                        ConstantMatrix::UnitColumn { index } => int_depends_on(index, tainted),
                        ConstantMatrix::Gadget { base, .. } => int_depends_on(base, tainted),
                        ConstantMatrix::PowerOfBase { base, exponent } => {
                            int_depends_on(base, tainted) || int_depends_on(exponent, tainted)
                        }
                        // Rotation is a total, type-invariant value operation.
                        ConstantMatrix::Rotation { .. } |
                        ConstantMatrix::Zero |
                        ConstantMatrix::Identity => false,
                    }
            }
            NodeKind::GadgetTrapdoor { matrix_type, base } => {
                matrix_type_depends_on(matrix_type, tainted) || int_depends_on(base, tainted)
            }
            NodeKind::BitExtract { bit } => int_depends_on(bit, tainted),
            NodeKind::Reshape { rows, columns } => {
                int_depends_on(rows, tainted) || int_depends_on(columns, tainted)
            }
            NodeKind::UniformSample { matrix_type, .. } => {
                matrix_type_depends_on(matrix_type, tainted)
            }
            NodeKind::GaussianSample { matrix_type, sigma } => {
                matrix_type_depends_on(matrix_type, tainted) || real_depends_on(sigma, tainted)
            }
            NodeKind::HashSample { matrix_type, base, digit_count, .. } => {
                matrix_type_depends_on(matrix_type, tainted) ||
                    base.as_ref().is_some_and(|base| int_depends_on(base, tainted)) ||
                    digit_count.as_ref().is_some_and(|count| int_depends_on(count, tainted))
            }
            NodeKind::TrapdoorSample { matrix_type, sigma, gadget_base, digit_count } => {
                matrix_type_depends_on(matrix_type, tainted) ||
                    real_depends_on(sigma, tainted) ||
                    int_depends_on(gadget_base, tainted) ||
                    int_depends_on(digit_count, tainted)
            }
            NodeKind::PreimageSample { matrix_type } => {
                matrix_type_depends_on(matrix_type, tainted)
            }
            NodeKind::GadgetDecompose { base, digit_count, .. } => {
                int_depends_on(base, tainted) ||
                    digit_count.as_ref().is_some_and(|count| int_depends_on(count, tainted))
            }
            NodeKind::ModDown { target_modulus } | NodeKind::ModUp { target_modulus } => {
                int_depends_on(target_modulus, tainted)
            }
            NodeKind::ExtractCoefficient { position } |
            NodeKind::ConstantCoefficient { position } => int_depends_on(position, tainted),
            NodeKind::ThresholdDecode { plaintext_modulus, length, .. } => {
                int_depends_on(plaintext_modulus, tainted) || int_depends_on(length, tainted)
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => {
                plaintext_moduli
                    .iter()
                    .chain(reconstruction_coefficients)
                    .any(|value| int_depends_on(value, tainted))
            }
            NodeKind::SubgraphCall(call) => {
                if let Some(child) = graph.subgraphs.get(&call.graph) {
                    let child_tainted = remap_tainted_variables(tainted, &call.bindings);
                    ensure_loop_structure_independent(child, &child_tainted, loop_node)?;
                }
                false
            }
            NodeKind::ParallelLoop(nested) => {
                if int_depends_on(&nested.count, tainted) {
                    true
                } else {
                    if let Some(child) = graph.subgraphs.get(&nested.graph) {
                        let mut inherited = tainted.clone();
                        inherited.remove(&nested.index_variable);
                        let child_tainted = remap_tainted_variables(&inherited, &nested.bindings);
                        ensure_loop_structure_independent(child, &child_tainted, loop_node)?;
                    }
                    false
                }
            }
            NodeKind::FamilyPack { count } => int_depends_on(count, tainted),
            NodeKind::FamilyGetStatic { index } => int_depends_on(index, tainted),
            NodeKind::Select { count } => int_depends_on(count, tainted),
            NodeKind::ConstantReal(value) => real_depends_on(value, tainted),
            NodeKind::MatrixScale { .. } |
            NodeKind::TrapdoorPublic |
            NodeKind::ConstantInt(_) |
            NodeKind::EvaluateInt(_) |
            NodeKind::ConstantBool(_) |
            NodeKind::Output { .. } |
            NodeKind::IntBinary(_) |
            NodeKind::IntCompare(_) |
            NodeKind::IntToReal |
            NodeKind::BoolToInt |
            NodeKind::RealBinary(_) |
            NodeKind::RealSqrt |
            NodeKind::MatrixBinary(_) |
            NodeKind::MatrixNegate |
            NodeKind::Transpose |
            NodeKind::Slice { .. } |
            NodeKind::Tensor |
            NodeKind::Concat { .. } |
            NodeKind::FamilyGetDynamic => false,
        };
        if depends {
            return reject(&format!("node {}", node.id.0));
        }
    }
    Ok(())
}

fn remap_tainted_variables(
    inherited: &BTreeSet<String>,
    bindings: &[(String, IntExpr)],
) -> BTreeSet<String> {
    let mut tainted = inherited.clone();
    for (name, expression) in bindings {
        if int_depends_on(expression, inherited) {
            tainted.insert(name.clone());
        } else {
            tainted.remove(name);
        }
    }
    tainted
}

fn int_depends_on(expression: &IntExpr, variables: &BTreeSet<String>) -> bool {
    variables.iter().any(|variable| expression.contains_variable(variable))
}

fn real_depends_on(expression: &crate::expr::RealExpr, variables: &BTreeSet<String>) -> bool {
    variables.iter().any(|variable| expression.contains_variable(variable))
}

fn matrix_type_depends_on(matrix: &MatrixType, variables: &BTreeSet<String>) -> bool {
    int_depends_on(&matrix.modulus, variables) ||
        int_depends_on(&matrix.ring_dimension, variables) ||
        int_depends_on(&matrix.rows, variables) ||
        int_depends_on(&matrix.columns, variables)
}

fn wire_type_depends_on(wire_type: &WireType, variables: &BTreeSet<String>) -> bool {
    match wire_type {
        WireType::Bytes { length } => int_depends_on(length, variables),
        WireType::TypedBlob { .. } => false,
        WireType::Matrix(matrix) | WireType::Preimage(matrix) => {
            matrix_type_depends_on(matrix, variables)
        }
        WireType::Trapdoor { matrix, sigma, gadget_base, digit_count } => {
            matrix_type_depends_on(matrix, variables) ||
                real_depends_on(sigma, variables) ||
                int_depends_on(gadget_base, variables) ||
                int_depends_on(digit_count, variables)
        }
        WireType::IndexedFamily { element, count } => {
            int_depends_on(count, variables) || wire_type_depends_on(element, variables)
        }
        WireType::ConstantInt |
        WireType::ConstantReal |
        WireType::ConstantBool |
        WireType::Int |
        WireType::Real |
        WireType::Bool => false,
    }
}

fn child_bindings(
    parent: &ParamEnv,
    declared: &[(String, IntExpr)],
) -> Result<ParamEnv, ValidationError> {
    let mut bindings = parent.clone();
    for (name, expression) in declared {
        bindings.integers.insert(name.clone(), expression.evaluate(parent)?);
    }
    Ok(bindings)
}

fn check_bindings(graph: &Graph, env: &ParamEnv) -> Result<(), ValidationError> {
    for parameter in &graph.parameters {
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

fn concrete_wire(
    wire_type: &WireType,
    env: &ParamEnv,
) -> Result<ConcreteWireType, ValidationError> {
    Ok(match wire_type {
        WireType::ConstantInt => ConcreteWireType::ConstantInt,
        WireType::ConstantReal => ConcreteWireType::ConstantReal,
        WireType::ConstantBool => ConcreteWireType::ConstantBool,
        WireType::Int => ConcreteWireType::Int,
        WireType::Real => ConcreteWireType::Real,
        WireType::Bool => ConcreteWireType::Bool,
        WireType::Bytes { length } => ConcreteWireType::Bytes {
            length: positive_usize(length.evaluate(env)?, "byte-string length", NodeId(0))?,
        },
        WireType::TypedBlob { type_name, schema_hash } => {
            ConcreteWireType::TypedBlob { type_name: type_name.clone(), schema_hash: *schema_hash }
        }
        WireType::Matrix(matrix) => ConcreteWireType::Matrix(concrete_matrix(matrix, env)?),
        WireType::Trapdoor { matrix, sigma, gadget_base, digit_count } => {
            let sigma = sigma.close(env)?;
            require_positive_real(
                sigma.evaluate_f64(&ParamEnv::default())?,
                NodeId(0),
                "trapdoor sigma",
            )?;
            let gadget_base = gadget_base.evaluate(env)?.abs();
            if gadget_base <= BigInt::one() {
                return Err(ValidationError::Node {
                    node: NodeId(0),
                    message: "gadget base must be greater than one".to_owned(),
                });
            }
            let digit_count =
                positive_usize(digit_count.evaluate(env)?, "trapdoor digit count", NodeId(0))?;
            ConcreteWireType::Trapdoor {
                matrix: concrete_matrix(matrix, env)?,
                sigma,
                gadget_base,
                digit_count,
            }
        }
        WireType::Preimage(matrix) => ConcreteWireType::Preimage(concrete_matrix(matrix, env)?),
        WireType::IndexedFamily { element, count } => {
            let count = nonnegative_usize(count.evaluate(env)?, "indexed family count", NodeId(0))?;
            let element = concrete_wire(element, env)?;
            if matches!(element, ConcreteWireType::IndexedFamily { .. }) {
                return Err(ValidationError::Node {
                    node: NodeId(0),
                    message: "nested indexed families are not supported".to_owned(),
                });
            }
            ConcreteWireType::IndexedFamily { element: Box::new(element), count }
        }
    })
}

fn concrete_matrix(
    matrix: &MatrixType,
    env: &ParamEnv,
) -> Result<ConcreteMatrixType, ValidationError> {
    let modulus = matrix.modulus.evaluate(env)?;
    if modulus <= BigInt::one() {
        return Err(ValidationError::Node {
            node: NodeId(0),
            message: "matrix modulus must be greater than one".to_owned(),
        });
    }
    Ok(ConcreteMatrixType {
        modulus,
        ring_dimension: positive_usize(
            matrix.ring_dimension.evaluate(env)?,
            "ring dimension",
            NodeId(0),
        )?,
        rows: positive_usize(matrix.rows.evaluate(env)?, "matrix rows", NodeId(0))?,
        columns: positive_usize(matrix.columns.evaluate(env)?, "matrix columns", NodeId(0))?,
    })
}

fn validate_constant(
    value: &ConstantMatrix,
    matrix: &ConcreteMatrixType,
    env: &ParamEnv,
    node: NodeId,
) -> Result<(), ValidationError> {
    match value {
        ConstantMatrix::UnitRow { index } => {
            let index = nonnegative_usize(index.evaluate(env)?, "unit-row index", node)?;
            if index >= matrix.columns {
                return Err(ValidationError::Node {
                    node,
                    message: "unit-row index is out of range".to_owned(),
                });
            }
        }
        ConstantMatrix::UnitColumn { index } => {
            let index = nonnegative_usize(index.evaluate(env)?, "unit-column index", node)?;
            if index >= matrix.rows {
                return Err(ValidationError::Node {
                    node,
                    message: "unit-column index is out of range".to_owned(),
                });
            }
        }
        ConstantMatrix::Gadget { base, .. } => {
            if base.evaluate(env)?.abs() <= BigInt::one() {
                return Err(ValidationError::Node {
                    node,
                    message: "gadget base must be greater than one".to_owned(),
                });
            }
        }
        ConstantMatrix::PowerOfBase { base, exponent } => {
            if base.evaluate(env)?.is_zero() || exponent.evaluate(env)?.is_negative() {
                return Err(ValidationError::Node {
                    node,
                    message: "power-of-base constant has invalid parameters".to_owned(),
                });
            }
        }
        ConstantMatrix::Rotation { exponent } => {
            let exponent = nonnegative_usize(exponent.evaluate(env)?, "rotation exponent", node)?;
            if exponent >= matrix.ring_dimension {
                return Err(ValidationError::Node {
                    node,
                    message: "rotation exponent is out of range".to_owned(),
                });
            }
        }
        ConstantMatrix::Zero | ConstantMatrix::Identity => {}
    }
    Ok(())
}

fn concat_type(
    inputs: &[ConcreteMatrixType],
    axis: ConcatAxis,
    node: NodeId,
) -> Result<ConcreteMatrixType, ValidationError> {
    let Some(first) = inputs.first() else {
        return Err(ValidationError::Node {
            node,
            message: "concat requires at least one input".to_owned(),
        });
    };
    for input in &inputs[1..] {
        crate::checks::check_same_ring(first, input)?;
        let valid = match axis {
            ConcatAxis::Rows => input.columns == first.columns,
            ConcatAxis::Columns => input.rows == first.rows,
            ConcatAxis::Diagonal => true,
        };
        if !valid {
            return Err(ValidationError::Node {
                node,
                message: "concat input shapes are incompatible".to_owned(),
            });
        }
    }
    let (rows, columns) = match axis {
        ConcatAxis::Rows => (inputs.iter().map(|input| input.rows).sum(), first.columns),
        ConcatAxis::Columns => (first.rows, inputs.iter().map(|input| input.columns).sum()),
        ConcatAxis::Diagonal => (
            inputs.iter().map(|input| input.rows).sum(),
            inputs.iter().map(|input| input.columns).sum(),
        ),
    };
    Ok(ConcreteMatrixType { rows, columns, ..first.clone() })
}

fn sliced_type(
    input: &ConcreteMatrixType,
    rows: Option<&crate::node::IndexRange>,
    columns: Option<&crate::node::IndexRange>,
    node: NodeId,
) -> Result<ConcreteMatrixType, ValidationError> {
    if rows.is_some_and(|range| range.start >= range.end || range.end > input.rows) ||
        columns.is_some_and(|range| range.start >= range.end || range.end > input.columns)
    {
        return Err(ValidationError::Node { node, message: "slice range is invalid".to_owned() });
    }
    Ok(ConcreteMatrixType {
        rows: rows.map_or(input.rows, |range| range.end - range.start),
        columns: columns.map_or(input.columns, |range| range.end - range.start),
        ..input.clone()
    })
}

fn decomposition_digits(
    explicit: Option<&IntExpr>,
    modulus: &BigInt,
    base: &BigInt,
    env: &ParamEnv,
    node: NodeId,
) -> Result<usize, ValidationError> {
    if let Some(explicit) = explicit {
        return positive_usize(explicit.evaluate(env)?, "decomposition digit count", node);
    }
    let mut power = BigInt::one();
    let mut digits = 0usize;
    while power < *modulus {
        power *= base;
        digits = digits.saturating_add(1);
    }
    Ok(digits.max(1))
}

fn positive_usize(value: BigInt, label: &str, node: NodeId) -> Result<usize, ValidationError> {
    value.to_usize().filter(|value| *value > 0).ok_or_else(|| ValidationError::Node {
        node,
        message: format!("{label} must be a positive usize"),
    })
}

fn nonnegative_usize(value: BigInt, label: &str, node: NodeId) -> Result<usize, ValidationError> {
    value.to_usize().ok_or_else(|| ValidationError::Node {
        node,
        message: format!("{label} must be a nonnegative usize"),
    })
}

fn require_positive_real(value: f64, node: NodeId, label: &str) -> Result<(), ValidationError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(ValidationError::Node { node, message: format!("{label} must be finite and positive") })
    }
}

fn require_nonnegative_real(value: f64, node: NodeId, label: &str) -> Result<(), ValidationError> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(ValidationError::Node {
            node,
            message: format!("{label} must be finite and nonnegative"),
        })
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

fn contextualize(
    node: NodeId,
    path: &[InstantiationFrame],
    source: ValidationError,
) -> ValidationError {
    ValidationError::Context { node, instantiation_path: path.to_vec(), source: Box::new(source) }
}

fn contextualize_unless_present(
    node: NodeId,
    path: &[InstantiationFrame],
    error: ValidationError,
) -> ValidationError {
    if matches!(error, ValidationError::Context { .. }) {
        error
    } else {
        contextualize(node, path, error)
    }
}

fn contextualize_check(error: CheckError, path: &[InstantiationFrame]) -> ValidationError {
    let node = match &error {
        CheckError::DuplicateNode(node) |
        CheckError::NotTopological { node, .. } |
        CheckError::InvalidOutput { node, .. } => *node,
        _ => NodeId(0),
    };
    contextualize(node, path, ValidationError::Check(error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        graph::{CompileParameter, CompileParameterKind, Graph},
        node::{LoopInputMode, Node, ParallelLoop, SampleRange},
        types::MatrixType,
    };

    fn matrix_type(rows: i64, columns: i64) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    #[test]
    fn validates_matrix_shapes_without_symbolic_analysis() {
        let graph = Graph {
            name: "core-validation".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(0),
                    kind: NodeKind::UniformSample {
                        matrix_type: matrix_type(2, 3),
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Transpose,
                    args: vec![WireRef { node: NodeId(0), port: Port(0) }],
                },
            ],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(1), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let validated = validate(&graph, &ParamEnv::default()).expect("valid core graph");
        assert_eq!(
            validated
                .wires
                .get(&WireId {
                    instantiation_path: Vec::new(),
                    wire: WireRef { node: NodeId(1), port: Port(0) },
                })
                .and_then(ConcreteWireType::matrix_type)
                .map(|matrix| (matrix.rows, matrix.columns)),
            Some((3, 2))
        );
    }

    #[test]
    fn gaussian_sigma_accepts_zero_and_rejects_negative_values() {
        let graph = |sigma| Graph {
            name: "gaussian-sigma".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(0),
                kind: NodeKind::GaussianSample {
                    matrix_type: matrix_type(1, 1),
                    sigma: crate::RealExpr::from_f64_exact(sigma).expect("finite sigma"),
                },
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(0), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };

        validate(&graph(0.0), &ParamEnv::default()).expect("zero sigma is a zero sample");
        assert!(validate(&graph(-1.0), &ParamEnv::default()).is_err());
    }

    #[test]
    fn crt_recompose_rejects_invalid_moduli_and_coefficients() {
        let graph = |plaintext_moduli: Vec<i64>, coefficients: Vec<i64>| Graph {
            name: "crt-validation".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![
                Node {
                    id: NodeId(0),
                    kind: NodeKind::UniformSample {
                        matrix_type: matrix_type(1, 2),
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(1),
                    kind: NodeKind::UniformSample {
                        matrix_type: matrix_type(1, 2),
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::CrtRecompose {
                        plaintext_moduli: plaintext_moduli
                            .into_iter()
                            .map(IntExpr::constant)
                            .collect(),
                        reconstruction_coefficients: coefficients
                            .into_iter()
                            .map(IntExpr::constant)
                            .collect(),
                    },
                    args: vec![
                        WireRef { node: NodeId(0), port: Port(0) },
                        WireRef { node: NodeId(1), port: Port(0) },
                    ],
                },
            ],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(2), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        validate(&graph(vec![3, 5], vec![6, 7]), &ParamEnv::default()).unwrap();
        for invalid in [
            graph(vec![0, 5], vec![6, 7]),
            graph(vec![17, 5], vec![6, 7]),
            graph(vec![3, 5], vec![-1, 7]),
            graph(vec![3, 5], vec![17, 7]),
        ] {
            assert!(validate(&invalid, &ParamEnv::default()).is_err());
        }
    }

    fn rotation_loop(count: usize) -> Graph {
        let body = Graph {
            name: "rotation-body".to_owned(),
            parameters: vec![CompileParameter {
                name: "i".to_owned(),
                kind: CompileParameterKind::Integer,
            }],
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(0),
                kind: NodeKind::ConstantMatrix {
                    matrix_type: matrix_type(2, 2),
                    value: ConstantMatrix::Rotation { exponent: IntExpr::Var("i".to_owned()) },
                },
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(0), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let mut graph = Graph {
            name: "rotation-loop".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(10),
                kind: NodeKind::ParallelLoop(ParallelLoop {
                    graph: body.name.clone(),
                    count: IntExpr::constant(count),
                    minimum_count: 0,
                    index_variable: "i".to_owned(),
                    bindings: Vec::new(),
                    input_modes: Vec::new(),
                }),
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(10), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(body.name.clone(), Box::new(body));
        graph
    }

    #[test]
    fn parallel_loop_metadata_is_constant_in_cardinality_and_accepts_zero() {
        for count in [0, 8] {
            let validated =
                validate(&rotation_loop(count), &ParamEnv::default()).expect("loop validation");
            assert_eq!(validated.wires.len(), 2);
            assert!(matches!(
                validated.wires.get(&WireId {
                    instantiation_path: Vec::new(),
                    wire: WireRef { node: NodeId(10), port: Port(0) },
                }),
                Some(ConcreteWireType::IndexedFamily {
                    element,
                    count: actual_count,
                }) if matches!(element.as_ref(), ConcreteWireType::Matrix(_))
                    && *actual_count == count
            ));
        }
    }

    #[test]
    fn rotation_constant_rejects_exponents_at_or_beyond_the_ring_dimension() {
        let mut valid = rotation_loop(1);
        let body = valid.subgraphs.get_mut("rotation-body").expect("body");
        let NodeKind::ConstantMatrix { value, .. } = &mut body.nodes[0].kind else {
            panic!("rotation body");
        };
        *value = ConstantMatrix::Rotation { exponent: IntExpr::constant(7) };
        validate(&valid, &ParamEnv::default())
            .expect("the largest valid exponent is ring_dimension - 1");

        let mut invalid = valid;
        let body = invalid.subgraphs.get_mut("rotation-body").expect("body");
        let NodeKind::ConstantMatrix { value, .. } = &mut body.nodes[0].kind else {
            panic!("rotation body");
        };
        *value = ConstantMatrix::Rotation { exponent: IntExpr::constant(8) };
        let error = validate(&invalid, &ParamEnv::default())
            .expect_err("rotation exponent equals ring_dimension");
        assert!(error.to_string().contains("rotation exponent is out of range"));
    }

    #[test]
    fn parallel_loop_rejects_index_dependent_structure() {
        let mut graph = rotation_loop(3);
        let body = graph.subgraphs.get_mut("rotation-body").expect("body");
        body.nodes[0].kind = NodeKind::ConstantMatrix {
            matrix_type: MatrixType {
                rows: IntExpr::Add(
                    Box::new(IntExpr::Var("i".to_owned())),
                    Box::new(IntExpr::constant(1)),
                ),
                ..matrix_type(2, 2)
            },
            value: ConstantMatrix::Zero,
        };
        let error =
            validate(&graph, &ParamEnv::default()).expect_err("index-dependent shape rejection");
        assert!(error.to_string().contains("parallel-loop index affects structural field"));
    }

    #[test]
    fn parallel_loop_rejects_index_dependent_structure_in_nested_subgraph() {
        let mut graph = rotation_loop(3);
        let body = graph.subgraphs.get_mut("rotation-body").expect("body");
        let nested = Graph {
            name: "nested-shape".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(0),
                kind: NodeKind::ConstantMatrix {
                    matrix_type: MatrixType {
                        rows: IntExpr::Add(
                            Box::new(IntExpr::Var("i".to_owned())),
                            Box::new(IntExpr::constant(1)),
                        ),
                        ..matrix_type(2, 2)
                    },
                    value: ConstantMatrix::Zero,
                },
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(0), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        body.nodes[0].kind = NodeKind::SubgraphCall(crate::node::SubgraphCall {
            graph: nested.name.clone(),
            bindings: Vec::new(),
        });
        body.subgraphs.insert(nested.name.clone(), Box::new(nested));

        let error =
            validate(&graph, &ParamEnv::default()).expect_err("nested shape must be rejected");
        assert!(error.to_string().contains("parallel-loop index affects structural field"));
    }

    #[test]
    fn parallel_loop_allows_nested_value_binding_to_vary_by_index() {
        let mut graph = rotation_loop(3);
        let body = graph.subgraphs.get_mut("rotation-body").expect("body");
        let nested = Graph {
            name: "nested-value".to_owned(),
            parameters: vec![CompileParameter {
                name: "rotation".to_owned(),
                kind: CompileParameterKind::Integer,
            }],
            input_types: BTreeMap::new(),
            nodes: vec![Node {
                id: NodeId(0),
                kind: NodeKind::ConstantMatrix {
                    matrix_type: matrix_type(2, 2),
                    value: ConstantMatrix::Rotation {
                        exponent: IntExpr::Var("rotation".to_owned()),
                    },
                },
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(0), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        body.nodes[0].kind = NodeKind::SubgraphCall(crate::node::SubgraphCall {
            graph: nested.name.clone(),
            bindings: vec![("rotation".to_owned(), IntExpr::Var("i".to_owned()))],
        });
        body.subgraphs.insert(nested.name.clone(), Box::new(nested));

        validate(&graph, &ParamEnv::default()).expect("value-only binding may vary");
    }

    #[test]
    fn zipped_parallel_loop_requires_matching_family_cardinality() {
        let input_type = WireType::Matrix(matrix_type(2, 2));
        let body = Graph {
            name: "zip-body".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::from([("value".to_owned(), input_type.clone())]),
            nodes: vec![Node {
                id: NodeId(0),
                kind: NodeKind::Input {
                    name: "value".to_owned(),
                    wire_type: input_type.clone(),
                    artifact: None,
                },
                args: Vec::new(),
            }],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(0), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let family_type =
            WireType::IndexedFamily { element: Box::new(input_type), count: IntExpr::constant(2) };
        let mut graph = Graph {
            name: "bad-zip-loop".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::from([("family".to_owned(), family_type.clone())]),
            nodes: vec![
                Node {
                    id: NodeId(0),
                    kind: NodeKind::Input {
                        name: "family".to_owned(),
                        wire_type: family_type,
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ParallelLoop(ParallelLoop {
                        graph: body.name.clone(),
                        count: IntExpr::constant(3),
                        minimum_count: 0,
                        index_variable: "i".to_owned(),
                        bindings: Vec::new(),
                        input_modes: vec![LoopInputMode::Zip],
                    }),
                    args: vec![WireRef { node: NodeId(0), port: Port(0) }],
                },
            ],
            outputs: BTreeMap::from([(
                "out".to_owned(),
                WireRef { node: NodeId(1), port: Port(0) },
            )]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        graph.subgraphs.insert(body.name.clone(), Box::new(body));
        let error = validate(&graph, &ParamEnv::default()).expect_err("zip count mismatch");
        assert!(error.to_string().contains("zipped parallel-loop input count mismatch"));

        let offset_family = WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(matrix_type(2, 2))),
            count: IntExpr::constant(4),
        };
        graph.input_types.insert("family".to_owned(), offset_family.clone());
        graph.nodes[0].kind =
            NodeKind::Input { name: "family".to_owned(), wire_type: offset_family, artifact: None };
        if let NodeKind::ParallelLoop(loop_node) = &mut graph.nodes[1].kind {
            loop_node.input_modes = vec![LoopInputMode::ZipOffset { offset: 1 }];
        } else {
            unreachable!()
        }
        validate(&graph, &ParamEnv::default())
            .expect("an offset zip may consume a bounded suffix of a larger family");
        if let NodeKind::ParallelLoop(loop_node) = &mut graph.nodes[1].kind {
            loop_node.input_modes = vec![LoopInputMode::ZipOffset { offset: 2 }];
        } else {
            unreachable!()
        }
        let error = validate(&graph, &ParamEnv::default()).expect_err("offset zip exceeds family");
        assert!(error.to_string().contains("zipped parallel-loop input count mismatch"));
    }
}
