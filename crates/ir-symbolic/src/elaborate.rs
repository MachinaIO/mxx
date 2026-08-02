use crate::{
    atom::{
        AssumedMetadata, Atom, AtomClass, AtomId, AtomKind, AtomTable, DeclaredDependencies,
        DeclaredDependencyRef, ExternalSourceKind, ParallelIndex, PreimageRelation,
        SelectionDomain, SelectionDomainRef, SourceKind, SymbolicInstantiationFrame,
    },
    checks::ElaborationWarning,
    expression::{
        ExpressionError, IndexRange, SymbolicExprArena, SymbolicExprId, SymbolicExprNode,
    },
    manifest::{Manifest, ManifestArtifact},
    overlay::{
        DeclaredDependencyLabels, ExactAtomRef, PendingSymbolicExpr, PendingSymbolicExprNode,
        SymbolicOverlay, SymbolicValueRef, VirtualKind,
    },
    rewrite::{RewriteError, rewrite_expression},
};
use mxx_ir_core::{
    FrozenGraphScopeId, Graph, ScopedWireRef,
    expr::{ExprError, ParamEnv},
    node::{HashVariant, LoopInputMode, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType, NodeId, Port, WireRef},
    validate::{ValidatedGraph, ValidatedScope, ValidationError},
};
use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum SymbolicFamily {
    ExactMembers(Vec<SymbolicExprId>),
    StructuralTemplate { count: usize, template: SymbolicExprId, index_slot: u32 },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ElaboratedWire {
    pub wire_type: ConcreteWireType,
    pub expression: Option<SymbolicExprId>,
    pub family: Option<SymbolicFamily>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ElaboratedScope {
    pub wires: BTreeMap<WireRef, ElaboratedWire>,
}

#[derive(Clone, Debug)]
pub struct ElaboratedGraph {
    pub name: String,
    pub source: Graph,
    pub bindings: ParamEnv,
    pub scopes: BTreeMap<FrozenGraphScopeId, ElaboratedScope>,
    pub outputs: BTreeMap<String, ScopedWireRef>,
    pub atoms: AtomTable,
    pub expressions: SymbolicExprArena,
    pub preimage_relations: Vec<PreimageRelation>,
    pub warnings: Vec<ElaborationWarning>,
    pub decode_targets: Vec<DecodeTarget>,
    pub assumption_digest: Option<[u8; 32]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DecodeTarget {
    pub input: ScopedWireRef,
    pub plaintext_modulus: BigInt,
    pub length: usize,
}

impl ElaboratedGraph {
    pub fn scope(&self, id: &FrozenGraphScopeId) -> Option<&ElaboratedScope> {
        self.scopes.get(id)
    }

    pub fn wire(&self, reference: &ScopedWireRef) -> Option<&ElaboratedWire> {
        self.scope(&reference.scope)?.wires.get(&reference.wire)
    }
}

#[derive(Debug, Error)]
pub enum ElaborationError {
    #[error(transparent)]
    ParameterExpression(#[from] ExprError),
    #[error(transparent)]
    Validation(#[from] ValidationError),
    #[error(transparent)]
    SymbolicExpression(#[from] ExpressionError),
    #[error(transparent)]
    Rewrite(#[from] RewriteError),
    #[error("scope {scope:?}, node {node:?}: {message}")]
    Node { scope: FrozenGraphScopeId, node: NodeId, message: String },
    #[error("symbolic overlay: {0}")]
    Overlay(String),
    #[error("symbolic manifests are not self-consistent: {0}")]
    Manifest(String),
}

pub fn elaborate(graph: &Graph, bindings: &ParamEnv) -> Result<ElaboratedGraph, ElaborationError> {
    let validated = mxx_ir_core::validate(graph, bindings)?;
    elaborate_validated(&validated, &[], &SymbolicOverlay::default())
}

pub fn elaborate_with_manifests(
    graph: &Graph,
    bindings: &ParamEnv,
    manifests: &[Manifest],
) -> Result<ElaboratedGraph, ElaborationError> {
    let validated = mxx_ir_core::validate(graph, bindings)?;
    elaborate_validated(&validated, manifests, &SymbolicOverlay::default())
}

pub fn elaborate_with_overlay(
    graph: &Graph,
    bindings: &ParamEnv,
    manifests: &[Manifest],
    overlay: &SymbolicOverlay,
) -> Result<ElaboratedGraph, ElaborationError> {
    let validated = mxx_ir_core::validate(graph, bindings)?;
    elaborate_validated(&validated, manifests, overlay)
}

pub fn elaborate_validated(
    validated: &ValidatedGraph,
    manifests: &[Manifest],
    overlay: &SymbolicOverlay,
) -> Result<ElaboratedGraph, ElaborationError> {
    overlay.validate().map_err(ElaborationError::Overlay)?;
    let mut state = State {
        graph: validated,
        overlay,
        atoms: AtomTable::default(),
        expressions: SymbolicExprArena::default(),
        relations: Vec::new(),
        scopes: BTreeMap::new(),
        warnings: Vec::new(),
        decode_targets: Vec::new(),
        imported_artifacts: BTreeMap::new(),
    };
    state.import_manifests(manifests)?;
    state.insert_virtual_atoms()?;
    state.elaborate_scope(&FrozenGraphScopeId::Root, validated.root_scope())?;
    state.apply_declared_preimages()?;
    state.rewrite_all_roots()?;
    let outputs = validated
        .source
        .outputs()
        .iter()
        .map(|(name, output)| {
            (name.clone(), ScopedWireRef { scope: FrozenGraphScopeId::Root, wire: output.value })
        })
        .collect();
    Ok(ElaboratedGraph {
        name: validated.source.name().to_owned(),
        source: validated.source.clone(),
        bindings: validated.bindings.clone(),
        scopes: state.scopes,
        outputs,
        atoms: state.atoms,
        expressions: state.expressions,
        preimage_relations: state.relations,
        warnings: state.warnings,
        decode_targets: state.decode_targets,
        assumption_digest: overlay.digest().map_err(ElaborationError::Overlay)?,
    })
}

struct State<'a> {
    graph: &'a ValidatedGraph,
    overlay: &'a SymbolicOverlay,
    atoms: AtomTable,
    expressions: SymbolicExprArena,
    relations: Vec<PreimageRelation>,
    scopes: BTreeMap<FrozenGraphScopeId, ElaboratedScope>,
    warnings: Vec<ElaborationWarning>,
    decode_targets: Vec<DecodeTarget>,
    imported_artifacts: BTreeMap<(mxx_ir_core::artifact::ProductionId, String), ManifestArtifact>,
}

#[derive(Clone, Default)]
struct SymbolicOutput {
    expression: Option<SymbolicExprId>,
    family: Option<SymbolicFamily>,
}

impl State<'_> {
    fn import_manifests(&mut self, manifests: &[Manifest]) -> Result<(), ElaborationError> {
        for manifest in manifests {
            let imported =
                crate::manifest::import_manifest(manifest, &mut self.expressions, &mut self.atoms)
                    .map_err(|error| ElaborationError::Manifest(error.to_string()))?;
            self.relations.extend(imported.preimage_relations);
            for (name, artifact) in imported.artifacts {
                let key = (manifest.production_id.clone(), name);
                if let Some(existing) = self.imported_artifacts.insert(key, artifact.clone()) &&
                    existing != artifact
                {
                    return Err(ElaborationError::Manifest(
                        "conflicting imported symbolic artifact".to_owned(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn insert_virtual_atoms(&mut self) -> Result<(), ElaborationError> {
        for (id, declaration) in &self.overlay.virtual_atoms {
            let matrix_type = concrete_matrix(&declaration.matrix_type, &self.graph.bindings)?;
            let (kind, metadata) = match &declaration.kind {
                VirtualKind::Large => (AtomKind::Large, None),
                VirtualKind::Bounded {
                    norm,
                    is_const_poly,
                    zero_rows,
                    dependencies,
                    clt_ready,
                } => {
                    let zero_rows = zero_rows
                        .as_ref()
                        .map(|rows| rows.evaluate(&self.graph.bindings))
                        .transpose()?
                        .map(|rows| {
                            rows.to_usize().ok_or_else(|| {
                                ElaborationError::Overlay(
                                    "virtual zero_rows must be a nonnegative usize".to_owned(),
                                )
                            })
                        })
                        .transpose()?;
                    if zero_rows.is_some_and(|rows| rows > matrix_type.rows) {
                        return Err(ElaborationError::Overlay(
                            "virtual zero_rows exceeds matrix rows".to_owned(),
                        ));
                    }
                    let dependencies = match dependencies {
                        DeclaredDependencyLabels::Known(labels) => DeclaredDependencies::Known(
                            labels.iter().cloned().map(DeclaredDependencyRef::Local).collect(),
                        ),
                        DeclaredDependencyLabels::Unknown => DeclaredDependencies::Unknown,
                    };
                    (
                        AtomKind::Bounded,
                        Some(AssumedMetadata {
                            norm: norm.close(&self.graph.bindings)?,
                            is_const_poly: *is_const_poly,
                            zero_rows,
                            dependencies,
                            clt_ready: *clt_ready,
                        }),
                    )
                }
            };
            self.insert_atom(Atom {
                id: AtomId::Virtual(*id),
                class: AtomClass::Assumed { metadata },
                kind,
                matrix_type,
            })?;
        }
        Ok(())
    }

    fn elaborate_scope(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        validated: &ValidatedScope,
    ) -> Result<(), ElaborationError> {
        if self.scopes.contains_key(scope_id) {
            return Ok(());
        }
        let source_scope = self.graph.source.scope(scope_id).expect("validated source scope");
        let mut wires = BTreeMap::new();
        self.scopes.insert(scope_id.clone(), ElaboratedScope { wires: BTreeMap::new() });
        for (position, handle) in validated.execution_order.iter().enumerate() {
            let node = NodeId(position as u64);
            let args = source_scope.arguments(handle).expect("frozen arguments");
            let outputs = handle
                .output_types()
                .iter()
                .enumerate()
                .map(|(port, _)| WireRef { node, port: Port(port as u32) })
                .collect::<Vec<_>>();
            let values = self.elaborate_node(
                scope_id,
                node,
                handle.kind(),
                &args,
                &outputs,
                validated,
                &wires,
            )?;
            for (port, output) in outputs.iter().enumerate() {
                let symbolic = values.get(port).cloned().unwrap_or_default();
                wires.insert(
                    *output,
                    ElaboratedWire {
                        wire_type: validated.wire_types[output].clone(),
                        expression: symbolic.expression,
                        family: symbolic.family,
                    },
                );
            }
            self.scopes.insert(scope_id.clone(), ElaboratedScope { wires: wires.clone() });
            for output in outputs {
                let target = ScopedWireRef { scope: scope_id.clone(), wire: output };
                if self.overlay.assumptions.contains_key(&target) {
                    self.apply_assumption(&target)?;
                }
            }
            wires = self.scopes.get(scope_id).expect("current scope exists").wires.clone();
        }
        self.scopes.insert(scope_id.clone(), ElaboratedScope { wires });
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn elaborate_node(
        &mut self,
        scope: &FrozenGraphScopeId,
        node: NodeId,
        kind: &NodeKind,
        args: &[WireRef],
        outputs: &[WireRef],
        validated: &ValidatedScope,
        wires: &BTreeMap<WireRef, ElaboratedWire>,
    ) -> Result<Vec<SymbolicOutput>, ElaborationError> {
        let output_type = |port: usize| validated.wire_types[&outputs[port]].clone();
        let local = |port: usize| {
            AtomId::Local(ScopedWireRef { scope: scope.clone(), wire: outputs[port] })
        };

        let result = match kind {
            NodeKind::Input { artifact, .. } => {
                let ty = output_type(0);
                let imported = artifact.as_ref().and_then(|artifact| {
                    self.imported_artifacts
                        .get(&(artifact.production_id.clone(), artifact.artifact_name.clone()))
                        .cloned()
                });
                if let Some(imported) = imported {
                    if imported.wire_type != ty {
                        return Err(self.node_error(
                            scope,
                            node,
                            "symbolic artifact type does not match executable input type",
                        ));
                    }
                    vec![SymbolicOutput {
                        expression: imported.expression,
                        family: imported.family,
                    }]
                } else if let Some(matrix_type) = matrix_type(&ty).cloned() {
                    let id = local(0);
                    let (atom_kind, source) = match ty {
                        ConcreteWireType::Preimage(_) => (
                            AtomKind::Bounded,
                            SourceKind::External { kind: ExternalSourceKind::Preimage },
                        ),
                        ConcreteWireType::Trapdoor { .. } => (
                            AtomKind::Bounded,
                            SourceKind::External { kind: ExternalSourceKind::TrapdoorUniform },
                        ),
                        _ => (
                            AtomKind::Large,
                            SourceKind::External { kind: ExternalSourceKind::Matrix },
                        ),
                    };
                    self.insert_source_atom(id.clone(), atom_kind, matrix_type.clone(), source)?;
                    if matches!(ty, ConcreteWireType::Trapdoor { .. }) {
                        self.insert_source_atom(
                            AtomId::TrapdoorPublic(ScopedWireRef {
                                scope: scope.clone(),
                                wire: outputs[0],
                            }),
                            AtomKind::Large,
                            matrix_type,
                            SourceKind::External { kind: ExternalSourceKind::Matrix },
                        )?;
                    }
                    let expression = self.expressions.atom(id, &self.atoms)?;
                    let family = match ty {
                        ConcreteWireType::IndexedFamily { count, .. } => {
                            Some(SymbolicFamily::StructuralTemplate {
                                count,
                                template: expression,
                                index_slot: u32::MAX,
                            })
                        }
                        _ => None,
                    };
                    vec![SymbolicOutput { expression: Some(expression), family }]
                } else {
                    vec![SymbolicOutput::default()]
                }
            }
            NodeKind::ConstantMatrix { value, .. } => {
                let ty = matrix_type(&output_type(0)).expect("validated matrix").clone();
                let expression = if matches!(value, mxx_ir_core::node::ConstantMatrix::Zero) {
                    self.expressions.zero(ty)?
                } else {
                    let id = local(0);
                    let kind = if matches!(
                        value,
                        mxx_ir_core::node::ConstantMatrix::Gadget { small: false, .. }
                    ) {
                        // A full gadget row reaches the modulus scale and is
                        // part of the encoded signal, not a bounded error.
                        AtomKind::Large
                    } else {
                        AtomKind::Bounded
                    };
                    self.insert_source_atom(
                        id.clone(),
                        kind,
                        ty,
                        SourceKind::ConstantMatrix { value: value.clone() },
                    )?;
                    self.expressions.atom(id, &self.atoms)?
                };
                vec![matrix_output(expression)]
            }
            NodeKind::UniformSample { range, .. } => {
                let minimum = range.minimum.evaluate(&self.graph.bindings)?;
                let maximum = range.maximum.evaluate(&self.graph.bindings)?;
                let ty = matrix_type(&output_type(0)).expect("validated matrix").clone();
                let maximum_absolute = minimum.abs().max(maximum.abs());
                let atom_kind = if &maximum_absolute * BigInt::from(2u8) >= ty.modulus {
                    AtomKind::Large
                } else {
                    AtomKind::Bounded
                };
                let id = local(0);
                self.insert_source_atom(
                    id.clone(),
                    atom_kind,
                    ty,
                    SourceKind::UniformSample { minimum, maximum },
                )?;
                let expression = self.expressions.atom(id, &self.atoms)?;
                vec![matrix_output(expression)]
            }
            NodeKind::GaussianSample { sigma, .. } => {
                let id = local(0);
                self.insert_source_atom(
                    id.clone(),
                    AtomKind::Bounded,
                    matrix_type(&output_type(0)).expect("validated matrix").clone(),
                    SourceKind::GaussianSample { sigma: sigma.close(&self.graph.bindings)? },
                )?;
                let expression = self.expressions.atom(id, &self.atoms)?;
                vec![matrix_output(expression)]
            }
            NodeKind::HashSample { variant, base, digit_count, .. } => {
                let id = local(0);
                let base =
                    base.as_ref().map(|value| value.evaluate(&self.graph.bindings)).transpose()?;
                let digit_count = digit_count
                    .as_ref()
                    .map(|value| value.evaluate(&self.graph.bindings))
                    .transpose()?
                    .map(|value| value.to_usize().expect("validated digit count"));
                self.insert_source_atom(
                    id.clone(),
                    if matches!(variant, HashVariant::Plain) {
                        AtomKind::Large
                    } else {
                        AtomKind::Bounded
                    },
                    matrix_type(&output_type(0)).expect("validated matrix").clone(),
                    SourceKind::HashSample { variant: *variant, base, digit_count },
                )?;
                let expression = self.expressions.atom(id, &self.atoms)?;
                vec![matrix_output(expression)]
            }
            NodeKind::TrapdoorSample { sigma, gadget_base, digit_count, .. } => {
                let public = local(0);
                let trapdoor = local(1);
                let ty = matrix_type(&output_type(0)).expect("validated public matrix").clone();
                let sigma = sigma.close(&self.graph.bindings)?;
                let base = gadget_base.evaluate(&self.graph.bindings)?.abs();
                let digits = digit_count
                    .evaluate(&self.graph.bindings)?
                    .to_usize()
                    .expect("validated digit count");
                self.insert_source_atom(
                    public.clone(),
                    AtomKind::Large,
                    ty.clone(),
                    SourceKind::TrapdoorUniform {
                        sigma: sigma.clone(),
                        gadget_base: base.clone(),
                        digit_count: digits,
                    },
                )?;
                self.insert_source_atom(
                    trapdoor.clone(),
                    AtomKind::Bounded,
                    ty,
                    SourceKind::TrapdoorUniform { sigma, gadget_base: base, digit_count: digits },
                )?;
                vec![
                    matrix_output(self.expressions.atom(public, &self.atoms)?),
                    matrix_output(self.expressions.atom(trapdoor, &self.atoms)?),
                ]
            }
            NodeKind::GadgetTrapdoor { base, .. } => {
                let id = local(0);
                let ty = matrix_type(&output_type(0)).expect("validated trapdoor").clone();
                let digit_count = ty.columns / ty.rows;
                self.insert_source_atom(
                    id.clone(),
                    AtomKind::Bounded,
                    ty,
                    SourceKind::GadgetDecomposition {
                        base: base.evaluate(&self.graph.bindings)?.abs(),
                        digit_count,
                        small: false,
                    },
                )?;
                let expression = self.expressions.atom(id, &self.atoms)?;
                vec![matrix_output(expression)]
            }
            NodeKind::TrapdoorPublic => {
                let expression = self.trapdoor_public_expression(scope, args[0], wires)?;
                vec![matrix_output(expression)]
            }
            NodeKind::PreimageSample { .. } => {
                let left_expression = self.matrix_expression(scope, node, args[0], wires)?;
                let left_matrix = self.exact_atom_expression(left_expression).ok_or_else(|| {
                    self.node_error(
                        scope,
                        node,
                        "preimage public input does not identify one exact atom",
                    )
                })?;
                let target = self.matrix_expression(scope, node, args[2], wires)?;
                let id = local(0);
                let output = matrix_type(&output_type(0)).expect("validated preimage").clone();
                let (sigma, base, digits, public_rows) = match &validated.wire_types[&args[1]] {
                    ConcreteWireType::Trapdoor { matrix, sigma, gadget_base, digit_count } => {
                        (sigma.clone(), gadget_base.clone(), *digit_count, matrix.rows)
                    }
                    _ => return Err(self.node_error(scope, node, "preimage input is not trapdoor")),
                };
                self.insert_source_atom(
                    id.clone(),
                    AtomKind::Bounded,
                    output.clone(),
                    SourceKind::PreimageSample {
                        trapdoor_sigma: sigma,
                        gadget_base: base,
                        digit_count: digits,
                        public_matrix_rows: public_rows,
                        target_block_rows: output.rows,
                        zero_rows: None,
                    },
                )?;
                let expression = self.expressions.atom(id.clone(), &self.atoms)?;
                self.insert_relation(PreimageRelation {
                    left_matrix,
                    preimage: id,
                    product: target,
                })?;
                vec![matrix_output(expression)]
            }
            NodeKind::GadgetDecompose { base, small, digit_count } => {
                let id = local(0);
                let ty = matrix_type(&output_type(0)).expect("validated decomposition").clone();
                let input_ty = matrix_type(&validated.wire_types[&args[0]])
                    .expect("validated decomposition input");
                let digits = digit_count
                    .as_ref()
                    .map(|value| value.evaluate(&self.graph.bindings))
                    .transpose()?
                    .and_then(|value| value.to_usize())
                    .unwrap_or_else(|| ty.rows / input_ty.rows);
                self.insert_source_atom(
                    id.clone(),
                    AtomKind::Bounded,
                    ty,
                    SourceKind::GadgetDecomposition {
                        base: base.evaluate(&self.graph.bindings)?.abs(),
                        digit_count: digits,
                        small: *small,
                    },
                )?;
                let expression = self.expressions.atom(id, &self.atoms)?;
                vec![matrix_output(expression)]
            }
            NodeKind::MatrixBinary(operation) => {
                let left = self.matrix_expression(scope, node, args[0], wires)?;
                let right = self.matrix_expression(scope, node, args[1], wires)?;
                let ty = matrix_type(&output_type(0)).expect("validated matrix result").clone();
                let expression = match operation {
                    MatrixBinaryOp::Add => self.expressions.add(ty, [left, right])?,
                    MatrixBinaryOp::Subtract => self.expressions.subtract(ty, left, right)?,
                    MatrixBinaryOp::Multiply => {
                        self.expressions.multiply(ty, [left, right], &self.atoms)?
                    }
                };
                vec![matrix_output(expression)]
            }
            NodeKind::MatrixNegate => {
                let value = self.matrix_expression(scope, node, args[0], wires)?;
                vec![matrix_output(self.expressions.scale(-BigInt::from(1u8), value)?)]
            }
            NodeKind::MatrixScale { scalar } => {
                let value = self.matrix_expression(scope, node, args[0], wires)?;
                let scalar = scalar.evaluate(&self.graph.bindings)?;
                vec![matrix_output(self.expressions.scale(scalar, value)?)]
            }
            NodeKind::Transpose => {
                let value = self.matrix_expression(scope, node, args[0], wires)?;
                let ty = matrix_type(&output_type(0)).expect("validated transpose").clone();
                vec![matrix_output(self.expressions.transpose(ty, value)?)]
            }
            NodeKind::Slice { rows, columns } => {
                let range = |range: &mxx_ir_core::node::IndexRange| -> Result<IndexRange, ElaborationError> {
                    Ok(IndexRange {
                        start: range.start.evaluate(&self.graph.bindings)?.to_usize().expect("validated"),
                        end: range.end.evaluate(&self.graph.bindings)?.to_usize().expect("validated"),
                    })
                };
                let value = self.matrix_expression(scope, node, args[0], wires)?;
                let ty = matrix_type(&output_type(0)).expect("validated slice").clone();
                let expression = self.expressions.slice(
                    ty,
                    value,
                    rows.as_ref().map(range).transpose()?,
                    columns.as_ref().map(range).transpose()?,
                )?;
                vec![matrix_output(expression)]
            }
            NodeKind::Tensor => {
                let left = self.matrix_expression(scope, node, args[0], wires)?;
                let right = self.matrix_expression(scope, node, args[1], wires)?;
                let ty = matrix_type(&output_type(0)).expect("validated tensor").clone();
                vec![matrix_output(self.expressions.tensor(ty, left, right)?)]
            }
            NodeKind::Concat { axis } => {
                let inputs = args
                    .iter()
                    .map(|argument| self.matrix_expression(scope, node, *argument, wires))
                    .collect::<Result<Vec<_>, _>>()?;
                let ty = matrix_type(&output_type(0)).expect("validated concat").clone();
                vec![matrix_output(self.expressions.concat(ty, *axis, inputs)?)]
            }
            NodeKind::Reshape { .. } => {
                let value = self.matrix_expression(scope, node, args[0], wires)?;
                let ty = matrix_type(&output_type(0)).expect("validated reshape").clone();
                vec![matrix_output(self.expressions.reshape(ty, value)?)]
            }
            NodeKind::ConstantCoefficient { position } => {
                let value = self.matrix_expression(scope, node, args[0], wires)?;
                let position = position
                    .evaluate(&self.graph.bindings)?
                    .to_usize()
                    .expect("validated coefficient position");
                let ty = matrix_type(&output_type(0)).expect("validated coefficient").clone();
                vec![matrix_output(self.expressions.constant_coefficient(ty, value, position)?)]
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => {
                let inputs = args
                    .iter()
                    .map(|argument| self.matrix_expression(scope, node, *argument, wires))
                    .collect::<Result<Vec<_>, _>>()?;
                let moduli = plaintext_moduli
                    .iter()
                    .map(|value| value.evaluate(&self.graph.bindings))
                    .collect::<Result<Vec<_>, _>>()?;
                let coefficients = reconstruction_coefficients
                    .iter()
                    .map(|value| value.evaluate(&self.graph.bindings))
                    .collect::<Result<Vec<_>, _>>()?;
                for (argument, plaintext_modulus) in args.iter().zip(&moduli) {
                    let input_type =
                        matrix_type(&validated.wire_types[argument]).expect("validated CRT input");
                    self.decode_targets.push(DecodeTarget {
                        input: ScopedWireRef { scope: scope.clone(), wire: *argument },
                        plaintext_modulus: plaintext_modulus.clone(),
                        length: input_type.ring_dimension * input_type.columns,
                    });
                }
                let ty = matrix_type(&output_type(0)).expect("validated CRT output").clone();
                let expression =
                    self.expressions.crt_recompose(ty, inputs, moduli, coefficients)?;
                vec![matrix_output(expression)]
            }
            NodeKind::ThresholdDecode { plaintext_modulus, length, .. } => {
                self.decode_targets.push(DecodeTarget {
                    input: ScopedWireRef { scope: scope.clone(), wire: args[0] },
                    plaintext_modulus: plaintext_modulus.evaluate(&self.graph.bindings)?,
                    length: length.evaluate(&self.graph.bindings)?.to_usize().expect("validated"),
                });
                outputs.iter().map(|_| SymbolicOutput::default()).collect()
            }
            NodeKind::FamilyPack { .. } => {
                let members = args
                    .iter()
                    .map(|argument| self.matrix_expression(scope, node, *argument, wires))
                    .collect::<Result<Vec<_>, _>>()?;
                let matrix =
                    matrix_type(&output_type(0)).expect("validated family element").clone();
                let domain = SelectionDomainRef::Local(SelectionDomain {
                    index_wire: ScopedWireRef { scope: scope.clone(), wire: outputs[0] },
                    instantiation_path: Vec::new(),
                    count: members.len() as u64,
                    modulus: matrix.modulus.clone(),
                    ring_dimension: matrix.ring_dimension,
                });
                let expression = self.expressions.select(matrix, domain, members.clone())?;
                vec![SymbolicOutput {
                    expression: Some(expression),
                    family: Some(SymbolicFamily::ExactMembers(members)),
                }]
            }
            NodeKind::FamilyGetStatic { index } => {
                let index = index
                    .evaluate(&self.graph.bindings)?
                    .to_usize()
                    .expect("validated family index");
                let expression = match wires.get(&args[0]).and_then(|wire| wire.family.as_ref()) {
                    Some(SymbolicFamily::ExactMembers(members)) => members[index],
                    Some(SymbolicFamily::StructuralTemplate { template, .. }) => self
                        .specialize_expression(
                            *template,
                            ParallelIndex::Static(index as u64),
                            0,
                            &mut BTreeMap::new(),
                        )?,
                    None => {
                        return Err(self.node_error(scope, node, "family has no symbolic state"))
                    }
                };
                vec![matrix_output(expression)]
            }
            NodeKind::FamilyGetDynamic => {
                let expression = match wires.get(&args[0]).and_then(|wire| wire.family.as_ref()) {
                    Some(SymbolicFamily::ExactMembers(members)) => {
                        let matrix =
                            matrix_type(&output_type(0)).expect("validated family element").clone();
                        let domain = SelectionDomainRef::Local(SelectionDomain {
                            index_wire: ScopedWireRef { scope: scope.clone(), wire: args[1] },
                            instantiation_path: Vec::new(),
                            count: members.len() as u64,
                            modulus: matrix.modulus.clone(),
                            ring_dimension: matrix.ring_dimension,
                        });
                        self.expressions.select(matrix, domain, members.clone())?
                    }
                    Some(SymbolicFamily::StructuralTemplate { template, .. }) => self
                        .specialize_expression(
                            *template,
                            ParallelIndex::Dynamic(ScopedWireRef {
                                scope: scope.clone(),
                                wire: args[1],
                            }),
                            0,
                            &mut BTreeMap::new(),
                        )?,
                    None => {
                        return Err(self.node_error(scope, node, "family has no symbolic state"))
                    }
                };
                vec![matrix_output(expression)]
            }
            NodeKind::Select { count } => {
                let branches = args[1..]
                    .iter()
                    .map(|branch| self.matrix_expression(scope, node, *branch, wires))
                    .collect::<Result<Vec<_>, _>>()?;
                let matrix = matrix_type(&output_type(0)).expect("validated select").clone();
                let domain = SelectionDomainRef::Local(SelectionDomain {
                    index_wire: ScopedWireRef { scope: scope.clone(), wire: args[0] },
                    instantiation_path: Vec::new(),
                    count: count.evaluate(&self.graph.bindings)?.to_u64().expect("validated"),
                    modulus: matrix.modulus.clone(),
                    ring_dimension: matrix.ring_dimension,
                });
                vec![matrix_output(self.expressions.select(matrix, domain, branches)?)]
            }
            NodeKind::SubgraphCall(_) | NodeKind::ParallelLoop(_) => {
                self.elaborate_structural(scope, node, kind, args, outputs, wires)?
            }
            NodeKind::ConstantInt(_) |
            NodeKind::EvaluateInt(_) |
            NodeKind::ConstantReal(_) |
            NodeKind::ConstantBool(_) |
            NodeKind::IntBinary(_) |
            NodeKind::IntCompare(_) |
            NodeKind::BitExtract { .. } |
            NodeKind::IntToReal |
            NodeKind::BoolToInt |
            NodeKind::RealBinary(_) |
            NodeKind::RealSqrt |
            NodeKind::ExtractCoefficient { .. } => {
                outputs.iter().map(|_| SymbolicOutput::default()).collect()
            }
        };
        Ok(result)
    }

    fn elaborate_structural(
        &mut self,
        scope: &FrozenGraphScopeId,
        node: NodeId,
        kind: &NodeKind,
        args: &[WireRef],
        outputs: &[WireRef],
        wires: &BTreeMap<WireRef, ElaboratedWire>,
    ) -> Result<Vec<SymbolicOutput>, ElaborationError> {
        let child_id = self
            .graph
            .source
            .child_scope_id(scope, node)
            .ok_or_else(|| self.node_error(scope, node, "missing structural child scope"))?;
        let child_validated = self
            .graph
            .scope(&child_id)
            .cloned()
            .ok_or_else(|| self.node_error(scope, node, "missing validated child scope"))?;
        self.elaborate_scope(&child_id, &child_validated)?;
        let child_source = self.graph.source.scope(&child_id).expect("validated child scope");
        let modes = match kind {
            NodeKind::ParallelLoop(spec) => spec.input_modes.clone(),
            NodeKind::SubgraphCall(_) => vec![LoopInputMode::Broadcast; args.len()],
            _ => unreachable!(),
        };
        let call_site =
            ScopedWireRef { scope: scope.clone(), wire: WireRef { node, port: Port(0) } };
        let frame = match kind {
            NodeKind::ParallelLoop(spec) => SymbolicInstantiationFrame::ParallelIteration {
                call_site: call_site.clone(),
                index_slot: spec.index_slot,
                index: ParallelIndex::Template,
                index_offset: 0,
            },
            NodeKind::SubgraphCall(_) => SymbolicInstantiationFrame::Call(call_site.clone()),
            _ => unreachable!(),
        };
        let loop_count = match kind {
            NodeKind::ParallelLoop(spec) => Some(
                spec.count
                    .evaluate(&self.graph.bindings)?
                    .to_usize()
                    .expect("validated parallel count"),
            ),
            _ => None,
        };
        let mut substitutions = BTreeMap::new();
        let mut wire_substitutions = BTreeMap::new();
        for (((child_input, parent_argument), mode), input_index) in
            child_source.inputs().iter().zip(args).zip(modes).zip(0usize..)
        {
            let child_reference = ScopedWireRef { scope: child_id.clone(), wire: *child_input };
            let parent_reference = ScopedWireRef { scope: scope.clone(), wire: *parent_argument };
            wire_substitutions.insert(child_reference.clone(), parent_reference);
            let child_atom = AtomId::Local(child_reference);
            let parent = wires.get(parent_argument).cloned().ok_or_else(|| {
                self.node_error(scope, node, "structural argument is unavailable")
            })?;
            let replacement = match &mode {
                LoopInputMode::Broadcast => parent.expression,
                LoopInputMode::Zip | LoopInputMode::ZipOffset { .. } => match parent.family {
                    Some(SymbolicFamily::ExactMembers(members)) => {
                        let offset = match mode {
                            LoopInputMode::Zip => 0,
                            LoopInputMode::ZipOffset { offset } => offset,
                            LoopInputMode::Broadcast => unreachable!(),
                        };
                        let count = loop_count.expect("parallel zip input");
                        let branches = members
                            .get(offset..offset + count)
                            .ok_or_else(|| {
                                self.node_error(scope, node, "parallel zip family range is invalid")
                            })?
                            .to_vec();
                        let matrix = matrix_type(&child_validated.wire_types[child_input])
                            .expect("validated matrix family argument")
                            .clone();
                        let domain = SelectionDomainRef::Local(SelectionDomain {
                            index_wire: call_site.clone(),
                            instantiation_path: vec![frame.clone()],
                            count: count as u64,
                            modulus: matrix.modulus.clone(),
                            ring_dimension: matrix.ring_dimension,
                        });
                        Some(self.expressions.select(matrix, domain, branches)?)
                    }
                    Some(SymbolicFamily::StructuralTemplate { template, .. }) => {
                        let offset = match mode {
                            LoopInputMode::Zip => 0,
                            LoopInputMode::ZipOffset { offset } => offset as u64,
                            LoopInputMode::Broadcast => unreachable!(),
                        };
                        Some(self.specialize_expression(
                            template,
                            ParallelIndex::Template,
                            offset,
                            &mut BTreeMap::new(),
                        )?)
                    }
                    None => parent.expression,
                },
            };
            if let Some(replacement) = replacement {
                substitutions.insert(child_atom, replacement);
            } else if matrix_type(&child_validated.wire_types[child_input]).is_some() {
                return Err(self.node_error(
                    scope,
                    node,
                    &format!("matrix structural argument {input_index} has no expression"),
                ));
            }
        }
        let child_outputs = self.scopes[&child_id].wires.clone();
        let mut memo = BTreeMap::new();
        let mut result = Vec::with_capacity(outputs.len());
        for (port, child_output) in child_source.outputs().iter().enumerate() {
            let output = &child_outputs[child_output];
            let expression = output
                .expression
                .map(|expression| {
                    self.instantiate_expression(
                        expression,
                        &child_id,
                        &frame,
                        &substitutions,
                        &wire_substitutions,
                        &mut memo,
                    )
                })
                .transpose()?;
            let family = match kind {
                NodeKind::ParallelLoop(spec) => {
                    let count = loop_count.expect("parallel loop count");
                    expression.map(|template| SymbolicFamily::StructuralTemplate {
                        count,
                        template,
                        index_slot: spec.index_slot,
                    })
                }
                NodeKind::SubgraphCall(_) => output
                    .family
                    .as_ref()
                    .map(|family| {
                        self.instantiate_family(
                            family,
                            &child_id,
                            &frame,
                            &substitutions,
                            &wire_substitutions,
                            &mut memo,
                        )
                    })
                    .transpose()?,
                _ => unreachable!(),
            };
            let _ = port;
            result.push(SymbolicOutput { expression, family });
        }
        self.instantiate_relations(
            &child_id,
            &frame,
            &substitutions,
            &wire_substitutions,
            &mut memo,
        )?;
        Ok(result)
    }

    fn instantiate_family(
        &mut self,
        family: &SymbolicFamily,
        child_scope: &FrozenGraphScopeId,
        frame: &SymbolicInstantiationFrame,
        substitutions: &BTreeMap<AtomId, SymbolicExprId>,
        wire_substitutions: &BTreeMap<ScopedWireRef, ScopedWireRef>,
        memo: &mut BTreeMap<SymbolicExprId, SymbolicExprId>,
    ) -> Result<SymbolicFamily, ElaborationError> {
        Ok(match family {
            SymbolicFamily::ExactMembers(members) => SymbolicFamily::ExactMembers(
                members
                    .iter()
                    .map(|member| {
                        self.instantiate_expression(
                            *member,
                            child_scope,
                            frame,
                            substitutions,
                            wire_substitutions,
                            memo,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            SymbolicFamily::StructuralTemplate { count, template, index_slot } => {
                SymbolicFamily::StructuralTemplate {
                    count: *count,
                    template: self.instantiate_expression(
                        *template,
                        child_scope,
                        frame,
                        substitutions,
                        wire_substitutions,
                        memo,
                    )?,
                    index_slot: *index_slot,
                }
            }
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn instantiate_expression(
        &mut self,
        source: SymbolicExprId,
        child_scope: &FrozenGraphScopeId,
        frame: &SymbolicInstantiationFrame,
        substitutions: &BTreeMap<AtomId, SymbolicExprId>,
        wire_substitutions: &BTreeMap<ScopedWireRef, ScopedWireRef>,
        memo: &mut BTreeMap<SymbolicExprId, SymbolicExprId>,
    ) -> Result<SymbolicExprId, ElaborationError> {
        if let Some(value) = memo.get(&source) {
            return Ok(*value);
        }
        let record = self
            .expressions
            .get(source)
            .cloned()
            .ok_or(ExpressionError::MissingExpression(source))?;
        let result = match record.node {
            SymbolicExprNode::Zero => self.expressions.zero(record.matrix_type)?,
            SymbolicExprNode::Atom(atom) => {
                if let Some(replacement) = substitutions.get(&atom) {
                    *replacement
                } else {
                    let atom = self.instantiate_atom(&atom, child_scope, frame)?;
                    self.expressions.atom(atom, &self.atoms)?
                }
            }
            SymbolicExprNode::Add(children) => {
                let children = self.instantiate_children(
                    children,
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )?;
                self.expressions.add(record.matrix_type, children)?
            }
            SymbolicExprNode::Scale { coefficient, value } => {
                let value = self.instantiate_expression(
                    value,
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )?;
                self.expressions.scale(coefficient, value)?
            }
            SymbolicExprNode::Mul(children) => {
                let children = self.instantiate_children(
                    children,
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )?;
                self.expressions.multiply(record.matrix_type, children, &self.atoms)?
            }
            SymbolicExprNode::Tensor { left, right } => {
                let [left, right] = self.instantiate_pair(
                    [left, right],
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )?;
                self.expressions.tensor(record.matrix_type, left, right)?
            }
            SymbolicExprNode::Concat { axis, inputs } => {
                let inputs = self.instantiate_children(
                    inputs,
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )?;
                self.expressions.concat(record.matrix_type, axis, inputs)?
            }
            SymbolicExprNode::Select { domain, branches } => {
                let branches = self.instantiate_children(
                    branches,
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )?;
                let domain = instantiate_domain(domain, child_scope, frame, wire_substitutions);
                self.expressions.select(record.matrix_type, domain, branches)?
            }
            SymbolicExprNode::Transpose(value) => {
                let value = self.instantiate_expression(
                    value,
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )?;
                self.expressions.transpose(record.matrix_type, value)?
            }
            SymbolicExprNode::Slice { value, rows, columns } => {
                let value = self.instantiate_expression(
                    value,
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )?;
                self.expressions.slice(record.matrix_type, value, rows, columns)?
            }
            SymbolicExprNode::Reshape { value, .. } => {
                let value = self.instantiate_expression(
                    value,
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )?;
                self.expressions.reshape(record.matrix_type, value)?
            }
            SymbolicExprNode::ConstantCoefficient { value, position } => {
                let value = self.instantiate_expression(
                    value,
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )?;
                self.expressions.constant_coefficient(record.matrix_type, value, position)?
            }
            SymbolicExprNode::CrtRecompose {
                inputs,
                plaintext_moduli,
                reconstruction_coefficients,
            } => {
                let inputs = self.instantiate_children(
                    inputs,
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )?;
                self.expressions.crt_recompose(
                    record.matrix_type,
                    inputs,
                    plaintext_moduli,
                    reconstruction_coefficients,
                )?
            }
        };
        memo.insert(source, result);
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn instantiate_children(
        &mut self,
        children: Vec<SymbolicExprId>,
        child_scope: &FrozenGraphScopeId,
        frame: &SymbolicInstantiationFrame,
        substitutions: &BTreeMap<AtomId, SymbolicExprId>,
        wire_substitutions: &BTreeMap<ScopedWireRef, ScopedWireRef>,
        memo: &mut BTreeMap<SymbolicExprId, SymbolicExprId>,
    ) -> Result<Vec<SymbolicExprId>, ElaborationError> {
        children
            .into_iter()
            .map(|child| {
                self.instantiate_expression(
                    child,
                    child_scope,
                    frame,
                    substitutions,
                    wire_substitutions,
                    memo,
                )
            })
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    fn instantiate_pair(
        &mut self,
        children: [SymbolicExprId; 2],
        child_scope: &FrozenGraphScopeId,
        frame: &SymbolicInstantiationFrame,
        substitutions: &BTreeMap<AtomId, SymbolicExprId>,
        wire_substitutions: &BTreeMap<ScopedWireRef, ScopedWireRef>,
        memo: &mut BTreeMap<SymbolicExprId, SymbolicExprId>,
    ) -> Result<[SymbolicExprId; 2], ElaborationError> {
        let [left, right] = children;
        Ok([
            self.instantiate_expression(
                left,
                child_scope,
                frame,
                substitutions,
                wire_substitutions,
                memo,
            )?,
            self.instantiate_expression(
                right,
                child_scope,
                frame,
                substitutions,
                wire_substitutions,
                memo,
            )?,
        ])
    }

    fn instantiate_atom(
        &mut self,
        id: &AtomId,
        child_scope: &FrozenGraphScopeId,
        frame: &SymbolicInstantiationFrame,
    ) -> Result<AtomId, ElaborationError> {
        let instantiated = match id {
            AtomId::Local(template) | AtomId::TrapdoorPublic(template)
                if &template.scope == child_scope =>
            {
                AtomId::Instantiated {
                    template: template.clone(),
                    instantiation_path: vec![frame.clone()],
                }
            }
            AtomId::Instantiated { template, instantiation_path } => {
                let mut path = instantiation_path.clone();
                path.push(frame.clone());
                AtomId::Instantiated { template: template.clone(), instantiation_path: path }
            }
            _ => return Ok(id.clone()),
        };
        if self.atoms.contains_key(&instantiated) {
            return Ok(instantiated);
        }
        let source = self.atoms.get(id).cloned().ok_or_else(|| {
            ElaborationError::Overlay(format!("missing atom during instantiation: {id:?}"))
        })?;
        self.insert_atom(Atom { id: instantiated.clone(), ..source })?;
        Ok(instantiated)
    }

    #[allow(clippy::too_many_arguments)]
    fn instantiate_relations(
        &mut self,
        child_scope: &FrozenGraphScopeId,
        frame: &SymbolicInstantiationFrame,
        substitutions: &BTreeMap<AtomId, SymbolicExprId>,
        wire_substitutions: &BTreeMap<ScopedWireRef, ScopedWireRef>,
        memo: &mut BTreeMap<SymbolicExprId, SymbolicExprId>,
    ) -> Result<(), ElaborationError> {
        let source_relations = self.relations.clone();
        for relation in source_relations {
            if !atom_belongs_to_scope(&relation.preimage, child_scope) {
                continue;
            }
            let left_matrix = self.instantiate_exact_atom(
                &relation.left_matrix,
                child_scope,
                frame,
                substitutions,
            )?;
            let preimage =
                self.instantiate_exact_atom(&relation.preimage, child_scope, frame, substitutions)?;
            let product = self.instantiate_expression(
                relation.product,
                child_scope,
                frame,
                substitutions,
                wire_substitutions,
                memo,
            )?;
            self.insert_relation(PreimageRelation { left_matrix, preimage, product })?;
        }
        Ok(())
    }

    fn instantiate_exact_atom(
        &mut self,
        atom: &AtomId,
        child_scope: &FrozenGraphScopeId,
        frame: &SymbolicInstantiationFrame,
        substitutions: &BTreeMap<AtomId, SymbolicExprId>,
    ) -> Result<AtomId, ElaborationError> {
        if let Some(expression) = substitutions.get(atom) {
            return self.exact_atom_expression(*expression).ok_or_else(|| {
                ElaborationError::Overlay(
                    "preimage relation input was bound to a non-atomic expression".to_owned(),
                )
            });
        }
        self.instantiate_atom(atom, child_scope, frame)
    }

    fn specialize_expression(
        &mut self,
        source: SymbolicExprId,
        index: ParallelIndex,
        index_offset: u64,
        memo: &mut BTreeMap<SymbolicExprId, SymbolicExprId>,
    ) -> Result<SymbolicExprId, ElaborationError> {
        if let Some(value) = memo.get(&source) {
            return Ok(*value);
        }
        let record = self
            .expressions
            .get(source)
            .cloned()
            .ok_or(ExpressionError::MissingExpression(source))?;
        let result = match record.node {
            SymbolicExprNode::Zero => self.expressions.zero(record.matrix_type)?,
            SymbolicExprNode::Atom(atom) => {
                let atom = self.specialize_atom(&atom, &index, index_offset)?;
                self.expressions.atom(atom, &self.atoms)?
            }
            SymbolicExprNode::Add(children) => {
                let children = self.specialize_children(children, &index, index_offset, memo)?;
                self.expressions.add(record.matrix_type, children)?
            }
            SymbolicExprNode::Scale { coefficient, value } => {
                let value = self.specialize_expression(value, index.clone(), index_offset, memo)?;
                self.expressions.scale(coefficient, value)?
            }
            SymbolicExprNode::Mul(children) => {
                let children = self.specialize_children(children, &index, index_offset, memo)?;
                self.expressions.multiply(record.matrix_type, children, &self.atoms)?
            }
            SymbolicExprNode::Tensor { left, right } => {
                let left = self.specialize_expression(left, index.clone(), index_offset, memo)?;
                let right = self.specialize_expression(right, index.clone(), index_offset, memo)?;
                self.expressions.tensor(record.matrix_type, left, right)?
            }
            SymbolicExprNode::Concat { axis, inputs } => {
                let inputs = self.specialize_children(inputs, &index, index_offset, memo)?;
                self.expressions.concat(record.matrix_type, axis, inputs)?
            }
            SymbolicExprNode::Select { domain, branches } => {
                if let Some(branch) = parallel_family_branch(&domain, &index, index_offset) {
                    let selected = *branches.get(branch).ok_or_else(|| {
                        ElaborationError::Overlay(
                            "parallel family specialization is out of range".to_owned(),
                        )
                    })?;
                    let result =
                        self.specialize_expression(selected, index.clone(), index_offset, memo)?;
                    memo.insert(source, result);
                    return Ok(result);
                }
                let branches = self.specialize_children(branches, &index, index_offset, memo)?;
                let domain = specialize_domain(domain, &index, index_offset);
                self.expressions.select(record.matrix_type, domain, branches)?
            }
            SymbolicExprNode::Transpose(value) => {
                let value = self.specialize_expression(value, index.clone(), index_offset, memo)?;
                self.expressions.transpose(record.matrix_type, value)?
            }
            SymbolicExprNode::Slice { value, rows, columns } => {
                let value = self.specialize_expression(value, index.clone(), index_offset, memo)?;
                self.expressions.slice(record.matrix_type, value, rows, columns)?
            }
            SymbolicExprNode::Reshape { value, .. } => {
                let value = self.specialize_expression(value, index.clone(), index_offset, memo)?;
                self.expressions.reshape(record.matrix_type, value)?
            }
            SymbolicExprNode::ConstantCoefficient { value, position } => {
                let value = self.specialize_expression(value, index.clone(), index_offset, memo)?;
                self.expressions.constant_coefficient(record.matrix_type, value, position)?
            }
            SymbolicExprNode::CrtRecompose {
                inputs,
                plaintext_moduli,
                reconstruction_coefficients,
            } => {
                let inputs = self.specialize_children(inputs, &index, index_offset, memo)?;
                self.expressions.crt_recompose(
                    record.matrix_type,
                    inputs,
                    plaintext_moduli,
                    reconstruction_coefficients,
                )?
            }
        };
        memo.insert(source, result);
        Ok(result)
    }

    fn specialize_children(
        &mut self,
        children: Vec<SymbolicExprId>,
        index: &ParallelIndex,
        index_offset: u64,
        memo: &mut BTreeMap<SymbolicExprId, SymbolicExprId>,
    ) -> Result<Vec<SymbolicExprId>, ElaborationError> {
        children
            .into_iter()
            .map(|child| self.specialize_expression(child, index.clone(), index_offset, memo))
            .collect()
    }

    fn specialize_atom(
        &mut self,
        id: &AtomId,
        index: &ParallelIndex,
        index_offset: u64,
    ) -> Result<AtomId, ElaborationError> {
        let AtomId::Instantiated { template, instantiation_path } = id else {
            return Ok(id.clone());
        };
        let mut path = instantiation_path.clone();
        if !specialize_path(&mut path, index, index_offset) {
            return Ok(id.clone());
        }
        let specialized =
            AtomId::Instantiated { template: template.clone(), instantiation_path: path };
        if !self.atoms.contains_key(&specialized) {
            let source = self.atoms.get(id).cloned().ok_or_else(|| {
                ElaborationError::Overlay(format!("missing template atom: {id:?}"))
            })?;
            self.insert_atom(Atom { id: specialized.clone(), ..source })?;
            self.specialize_relations_for(id, &specialized, index, index_offset)?;
        }
        Ok(specialized)
    }

    fn specialize_relations_for(
        &mut self,
        source_atom: &AtomId,
        specialized_atom: &AtomId,
        index: &ParallelIndex,
        index_offset: u64,
    ) -> Result<(), ElaborationError> {
        let relations = self.relations.clone();
        for relation in relations.into_iter().filter(|relation| &relation.preimage == source_atom) {
            let left_matrix = self.specialize_atom(&relation.left_matrix, index, index_offset)?;
            let mut memo = BTreeMap::new();
            let product = self.specialize_expression(
                relation.product,
                index.clone(),
                index_offset,
                &mut memo,
            )?;
            self.insert_relation(PreimageRelation {
                left_matrix,
                preimage: specialized_atom.clone(),
                product,
            })?;
        }
        Ok(())
    }

    fn apply_assumption(&mut self, target: &ScopedWireRef) -> Result<(), ElaborationError> {
        let pending = self.overlay.assumptions.get(target).expect("caller checked assumption");
        let expression = self.convert_pending_expression(pending)?;
        let wire = self
            .scopes
            .get_mut(&target.scope)
            .and_then(|scope| scope.wires.get_mut(&target.wire))
            .ok_or_else(|| {
                ElaborationError::Overlay("assumption target is unavailable".to_owned())
            })?;
        let expected = matrix_type(&wire.wire_type).ok_or_else(|| {
            ElaborationError::Overlay("assumption target is not a matrix".to_owned())
        })?;
        if self.expressions.matrix_type(expression)? != expected {
            return Err(ElaborationError::Overlay(
                "assumption expression type does not match target".to_owned(),
            ));
        }
        wire.expression = Some(expression);
        wire.family = None;
        Ok(())
    }

    fn convert_pending_expression(
        &mut self,
        pending: &PendingSymbolicExpr,
    ) -> Result<SymbolicExprId, ElaborationError> {
        let matrix_type = concrete_matrix(&pending.matrix_type, &self.graph.bindings)?;
        Ok(match &pending.node {
            PendingSymbolicExprNode::Zero => self.expressions.zero(matrix_type)?,
            PendingSymbolicExprNode::Value(value) => match value {
                SymbolicValueRef::Local(reference) => self
                    .scopes
                    .get(&reference.scope)
                    .and_then(|scope| scope.wires.get(&reference.wire))
                    .and_then(|wire| wire.expression)
                    .ok_or_else(|| {
                        ElaborationError::Overlay(format!(
                            "assumption value is unavailable: {reference:?}"
                        ))
                    })?,
                SymbolicValueRef::Virtual(id) => {
                    self.expressions.atom(AtomId::Virtual(*id), &self.atoms)?
                }
                SymbolicValueRef::ImportedAtom { production_id, manifest_atom_id } => {
                    self.expressions.atom(
                        AtomId::Imported {
                            production_id: production_id.clone(),
                            manifest_atom_id: *manifest_atom_id,
                        },
                        &self.atoms,
                    )?
                }
            },
            PendingSymbolicExprNode::Add(children) => {
                let children = children
                    .iter()
                    .map(|child| self.convert_pending_expression(child))
                    .collect::<Result<Vec<_>, _>>()?;
                self.expressions.add(matrix_type, children)?
            }
            PendingSymbolicExprNode::Scale { coefficient, value } => {
                let value = self.convert_pending_expression(value)?;
                let result =
                    self.expressions.scale(coefficient.evaluate(&self.graph.bindings)?, value)?;
                if self.expressions.matrix_type(result)? != &matrix_type {
                    return Err(ElaborationError::Overlay(
                        "assumption scale type mismatch".to_owned(),
                    ));
                }
                result
            }
            PendingSymbolicExprNode::Mul(children) => {
                let children = children
                    .iter()
                    .map(|child| self.convert_pending_expression(child))
                    .collect::<Result<Vec<_>, _>>()?;
                self.expressions.multiply(matrix_type, children, &self.atoms)?
            }
        })
    }

    fn apply_declared_preimages(&mut self) -> Result<(), ElaborationError> {
        for relation in &self.overlay.preimage_relations {
            let left_matrix = self.exact_atom(&relation.left_matrix)?;
            let preimage = self.exact_atom(&relation.preimage)?;
            let product = self.convert_pending_expression(&relation.product)?;
            self.insert_relation(PreimageRelation { left_matrix, preimage, product })?;
        }
        Ok(())
    }

    fn exact_atom(&self, reference: &ExactAtomRef) -> Result<AtomId, ElaborationError> {
        match reference {
            ExactAtomRef::Virtual(id) => Ok(AtomId::Virtual(*id)),
            ExactAtomRef::Imported { production_id, manifest_atom_id } => Ok(AtomId::Imported {
                production_id: production_id.clone(),
                manifest_atom_id: *manifest_atom_id,
            }),
            ExactAtomRef::Local(reference) => {
                let expression = self
                    .scopes
                    .get(&reference.scope)
                    .and_then(|scope| scope.wires.get(&reference.wire))
                    .and_then(|wire| wire.expression)
                    .ok_or_else(|| {
                        ElaborationError::Overlay("exact local atom is unavailable".to_owned())
                    })?;
                self.exact_atom_expression(expression).ok_or_else(|| {
                    ElaborationError::Overlay(
                        "exact local reference is not one unviewed atom".to_owned(),
                    )
                })
            }
        }
    }

    fn rewrite_all_roots(&mut self) -> Result<(), ElaborationError> {
        let relation_snapshot = self.relations.clone();
        for relation in &mut self.relations {
            relation.product = rewrite_expression(
                relation.product,
                &mut self.expressions,
                &self.atoms,
                &relation_snapshot,
            )?;
        }
        let relations = self.relations.clone();
        for scope in self.scopes.values_mut() {
            for wire in scope.wires.values_mut() {
                if let Some(expression) = wire.expression {
                    wire.expression = Some(rewrite_expression(
                        expression,
                        &mut self.expressions,
                        &self.atoms,
                        &relations,
                    )?);
                }
                if let Some(family) = &mut wire.family {
                    match family {
                        SymbolicFamily::ExactMembers(members) => {
                            for member in members {
                                *member = rewrite_expression(
                                    *member,
                                    &mut self.expressions,
                                    &self.atoms,
                                    &relations,
                                )?;
                            }
                        }
                        SymbolicFamily::StructuralTemplate { template, .. } => {
                            *template = rewrite_expression(
                                *template,
                                &mut self.expressions,
                                &self.atoms,
                                &relations,
                            )?;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn trapdoor_public_expression(
        &mut self,
        scope: &FrozenGraphScopeId,
        trapdoor: WireRef,
        wires: &BTreeMap<WireRef, ElaboratedWire>,
    ) -> Result<SymbolicExprId, ElaborationError> {
        if trapdoor.port != Port(0) {
            let public = WireRef { node: trapdoor.node, port: Port(0) };
            if let Some(expression) = wires.get(&public).and_then(|wire| wire.expression) {
                return Ok(expression);
            }
        }
        let id = AtomId::TrapdoorPublic(ScopedWireRef { scope: scope.clone(), wire: trapdoor });
        if self.atoms.contains_key(&id) {
            return Ok(self.expressions.atom(id, &self.atoms)?);
        }
        wires
            .get(&trapdoor)
            .and_then(|wire| wire.expression)
            .ok_or_else(|| self.node_error(scope, trapdoor.node, "trapdoor public is unavailable"))
    }

    fn matrix_expression(
        &self,
        scope: &FrozenGraphScopeId,
        node: NodeId,
        wire: WireRef,
        wires: &BTreeMap<WireRef, ElaboratedWire>,
    ) -> Result<SymbolicExprId, ElaborationError> {
        wires.get(&wire).and_then(|wire| wire.expression).ok_or_else(|| {
            self.node_error(scope, node, "matrix argument has no symbolic expression")
        })
    }

    fn exact_atom_expression(&self, expression: SymbolicExprId) -> Option<AtomId> {
        match &self.expressions.get(expression)?.node {
            SymbolicExprNode::Atom(atom) => Some(atom.clone()),
            _ => None,
        }
    }

    fn insert_source_atom(
        &mut self,
        id: AtomId,
        kind: AtomKind,
        matrix_type: ConcreteMatrixType,
        source: SourceKind,
    ) -> Result<(), ElaborationError> {
        self.insert_atom(Atom { id, class: AtomClass::Source { source }, kind, matrix_type })
    }

    fn insert_atom(&mut self, atom: Atom) -> Result<(), ElaborationError> {
        if let Some(existing) = self.atoms.insert(atom.clone()) &&
            existing != atom
        {
            return Err(ElaborationError::Overlay("symbolic atom identity collision".to_owned()));
        }
        Ok(())
    }

    fn insert_relation(&mut self, relation: PreimageRelation) -> Result<(), ElaborationError> {
        if let Some(existing) = self.relations.iter().find(|existing| {
            existing.left_matrix == relation.left_matrix && existing.preimage == relation.preimage
        }) {
            if existing != &relation {
                return Err(ElaborationError::Overlay(
                    "preimage relation is declared more than once".to_owned(),
                ));
            }
            return Ok(());
        }
        self.relations.push(relation);
        Ok(())
    }

    fn node_error(
        &self,
        scope: &FrozenGraphScopeId,
        node: NodeId,
        message: &str,
    ) -> ElaborationError {
        ElaborationError::Node { scope: scope.clone(), node, message: message.to_owned() }
    }
}

fn matrix_output(expression: SymbolicExprId) -> SymbolicOutput {
    SymbolicOutput { expression: Some(expression), family: None }
}

fn matrix_type(ty: &ConcreteWireType) -> Option<&ConcreteMatrixType> {
    match ty {
        ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => Some(matrix),
        ConcreteWireType::Trapdoor { matrix, .. } => Some(matrix),
        ConcreteWireType::IndexedFamily { element, .. } => matrix_type(element),
        _ => None,
    }
}

fn concrete_matrix(
    matrix_type: &mxx_ir_core::types::MatrixType,
    bindings: &ParamEnv,
) -> Result<ConcreteMatrixType, ElaborationError> {
    Ok(ConcreteMatrixType {
        modulus: matrix_type.modulus.evaluate(bindings)?,
        ring_dimension: matrix_type
            .ring_dimension
            .evaluate(bindings)?
            .to_usize()
            .ok_or_else(|| ElaborationError::Overlay("ring dimension is not a usize".to_owned()))?,
        rows: matrix_type
            .rows
            .evaluate(bindings)?
            .to_usize()
            .ok_or_else(|| ElaborationError::Overlay("matrix rows are not a usize".to_owned()))?,
        columns: matrix_type.columns.evaluate(bindings)?.to_usize().ok_or_else(|| {
            ElaborationError::Overlay("matrix columns are not a usize".to_owned())
        })?,
    })
}

fn instantiate_domain(
    domain: SelectionDomainRef,
    child_scope: &FrozenGraphScopeId,
    frame: &SymbolicInstantiationFrame,
    substitutions: &BTreeMap<ScopedWireRef, ScopedWireRef>,
) -> SelectionDomainRef {
    match domain {
        SelectionDomainRef::Local(mut domain) => {
            if domain.index_wire.scope == *child_scope {
                if let Some(replacement) = substitutions.get(&domain.index_wire) {
                    domain.index_wire = replacement.clone();
                }
                domain.instantiation_path.push(frame.clone());
            }
            SelectionDomainRef::Local(domain)
        }
        imported => imported,
    }
}

fn specialize_domain(
    mut domain: SelectionDomainRef,
    index: &ParallelIndex,
    index_offset: u64,
) -> SelectionDomainRef {
    if let SelectionDomainRef::Local(domain) = &mut domain {
        specialize_path(&mut domain.instantiation_path, index, index_offset);
    }
    domain
}

fn parallel_family_branch(
    domain: &SelectionDomainRef,
    index: &ParallelIndex,
    index_offset: u64,
) -> Option<usize> {
    let SelectionDomainRef::Local(domain) = domain else { return None };
    let SymbolicInstantiationFrame::ParallelIteration {
        call_site,
        index: ParallelIndex::Template,
        index_offset: stored_offset,
        ..
    } = domain.instantiation_path.last()?
    else {
        return None;
    };
    if &domain.index_wire != call_site {
        return None;
    }
    match index {
        ParallelIndex::Static(index) => {
            usize::try_from(index.checked_add(*stored_offset)?.checked_add(index_offset)?).ok()
        }
        ParallelIndex::Template | ParallelIndex::Dynamic(_) => None,
    }
}

fn specialize_path(
    path: &mut [SymbolicInstantiationFrame],
    index: &ParallelIndex,
    index_offset: u64,
) -> bool {
    for frame in path.iter_mut().rev() {
        if let SymbolicInstantiationFrame::ParallelIteration {
            index: selected,
            index_offset: stored_offset,
            ..
        } = frame &&
            matches!(selected, ParallelIndex::Template)
        {
            let combined_offset = stored_offset
                .checked_add(index_offset)
                .expect("validated parallel offset must fit u64");
            match index {
                ParallelIndex::Static(index) => {
                    *selected = ParallelIndex::Static(
                        index
                            .checked_add(combined_offset)
                            .expect("validated parallel index must fit u64"),
                    );
                    *stored_offset = 0;
                }
                ParallelIndex::Template | ParallelIndex::Dynamic(_) => {
                    *selected = index.clone();
                    *stored_offset = combined_offset;
                }
            }
            return true;
        }
    }
    false
}

fn atom_belongs_to_scope(atom: &AtomId, scope: &FrozenGraphScopeId) -> bool {
    matches!(
        atom,
        AtomId::Local(reference) | AtomId::TrapdoorPublic(reference) if &reference.scope == scope
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        GraphOutput, IntExpr, NodeHandle, ValueHandle, WireType,
        graph::Graph,
        node::{ConcatAxis, NodeKind},
        types::MatrixType,
    };
    use std::collections::BTreeMap;

    fn matrix_type(rows: i64, columns: i64) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(257),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn input(name: &str, rows: i64, columns: i64) -> ValueHandle {
        let wire_type = WireType::Matrix(matrix_type(rows, columns));
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
        NodeHandle::new(kind, arguments, output_types).output(0).expect("node output")
    }

    fn graph(name: &str, outputs: impl IntoIterator<Item = (&'static str, ValueHandle)>) -> Graph {
        Graph::freeze(
            name,
            Vec::new(),
            outputs
                .into_iter()
                .map(|(name, value)| {
                    (name.to_owned(), GraphOutput { value, confidentiality: None })
                })
                .collect(),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .expect("freeze graph")
        .0
    }

    fn output_node<'a>(graph: &'a ElaboratedGraph, name: &str) -> &'a SymbolicExprNode {
        let expression = graph
            .wire(&graph.outputs[name])
            .and_then(|wire| wire.expression)
            .expect("matrix output expression");
        &graph.expressions.get(expression).expect("expression record").node
    }

    #[test]
    fn active_structural_nodes_elaborate_to_typed_symbolic_nodes() {
        let rectangular = input("rectangular", 2, 3);
        let transposed = value(
            NodeKind::Transpose,
            vec![rectangular.clone()],
            vec![WireType::Matrix(matrix_type(3, 2))],
        );
        let reshaped = value(
            NodeKind::Reshape { rows: IntExpr::constant(1), columns: IntExpr::constant(6) },
            vec![rectangular],
            vec![WireType::Matrix(matrix_type(1, 6))],
        );
        let coefficient = value(
            NodeKind::ConstantCoefficient { position: IntExpr::constant(3) },
            vec![input("polynomial", 1, 1)],
            vec![WireType::Matrix(matrix_type(1, 1))],
        );
        let concatenated = value(
            NodeKind::Concat { axis: ConcatAxis::Rows },
            vec![input("top", 1, 2), input("bottom", 2, 2)],
            vec![WireType::Matrix(matrix_type(3, 2))],
        );
        let elaborated = elaborate(
            &graph(
                "active-structural-nodes",
                [
                    ("transpose", transposed),
                    ("reshape", reshaped),
                    ("coefficient", coefficient),
                    ("concat", concatenated),
                ],
            ),
            &ParamEnv::default(),
        )
        .expect("elaboration");

        assert!(matches!(output_node(&elaborated, "transpose"), SymbolicExprNode::Transpose(_)));
        assert!(matches!(
            output_node(&elaborated, "reshape"),
            SymbolicExprNode::Reshape { rows: 1, columns: 6, .. }
        ));
        assert!(matches!(
            output_node(&elaborated, "coefficient"),
            SymbolicExprNode::ConstantCoefficient { position: 3, .. }
        ));
        assert!(matches!(
            output_node(&elaborated, "concat"),
            SymbolicExprNode::Concat { axis: ConcatAxis::Rows, inputs } if inputs.len() == 2
        ));
    }

    #[test]
    fn crt_and_threshold_decode_elaboration_preserve_normative_metadata() {
        let crt = value(
            NodeKind::CrtRecompose {
                plaintext_moduli: vec![IntExpr::constant(3), IntExpr::constant(5)],
                reconstruction_coefficients: vec![IntExpr::constant(7), IntExpr::constant(11)],
            },
            vec![input("level-0", 1, 2), input("level-1", 1, 2)],
            vec![WireType::Matrix(matrix_type(1, 2))],
        );
        let decode_handle = NodeHandle::new(
            NodeKind::ThresholdDecode {
                plaintext_modulus: IntExpr::constant(3),
                length: IntExpr::constant(2),
                output_bool: false,
            },
            vec![input("encoded", 1, 1)],
            vec![WireType::Int, WireType::Int],
        );
        let decoded = decode_handle.output(0).expect("first decoded coefficient");
        let elaborated = elaborate(
            &graph("crt-and-decode", [("crt", crt), ("decoded", decoded)]),
            &ParamEnv::default(),
        )
        .expect("elaboration");

        assert!(matches!(
            output_node(&elaborated, "crt"),
            SymbolicExprNode::CrtRecompose {
                plaintext_moduli,
                reconstruction_coefficients,
                inputs,
            } if plaintext_moduli == &vec![BigInt::from(3), BigInt::from(5)] &&
                reconstruction_coefficients == &vec![BigInt::from(7), BigInt::from(11)] &&
                inputs.len() == 2
        ));
        assert_eq!(elaborated.decode_targets.len(), 3);
        let decode_metadata = elaborated
            .decode_targets
            .iter()
            .map(|target| (target.plaintext_modulus.clone(), target.length))
            .collect::<Vec<_>>();
        assert_eq!(
            decode_metadata,
            vec![(BigInt::from(3), 16), (BigInt::from(5), 16), (BigInt::from(3), 2)]
        );
    }
}
