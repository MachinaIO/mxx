use crate::{
    atom::{
        Atom, AtomClass, AtomId, AtomKind, AtomTable, ConcatAxis, DefExpr, InstantiationFrame,
        PreimageRefs, SelectionDomain, TargetRef,
    },
    bounds::{BoundError, sum_norm, term_norm},
    checks::{
        CheckError, ElaborationWarning, WarningKind, check_add_shape, check_mod_down_normal_form,
        check_topological, is_reduced, multiplication_type,
    },
    expr::{ExprError, ParamEnv, RealExpr},
    graph::Graph,
    manifest::{
        ImportedManifest, InterpretationDigest, Manifest, import_manifest,
        merge_manifest_projections,
    },
    node::{ConstantMatrix, HashVariant, MatrixBinaryOp, Node, NodeKind},
    overlay::{
        AssumedTermListId, AtomRef, ExpectedEntry, FoldGroup, LoopIndexMatcher, OverlayTerm,
        PortMatcher, Reinterpretation, SymbolicOverlay, VirtualKind, overlay_hashes,
        selector_matches, validate_overlay,
    },
    rewrite::{RewriteError, TargetTermLists, rewrite_preimages},
    term::{Factor, Term, TermError, TermList, ViewDescriptor},
    types::{
        ConcreteMatrixType, ConcreteWireType, MatrixType, NodeId, Port, WireId, WireRef, WireType,
    },
    ubound::UBound,
};
use num_bigint::BigInt;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ElaboratedWire {
    pub wire_type: ConcreteWireType,
    pub terms: Option<TermList>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ElaboratedGraph {
    pub name: String,
    pub source: Graph,
    pub bindings: ParamEnv,
    pub wires: BTreeMap<WireId, ElaboratedWire>,
    pub outputs: BTreeMap<String, WireRef>,
    pub atoms: AtomTable,
    pub warnings: Vec<ElaborationWarning>,
    pub ring_expansion: UBound,
    pub target_terms: BTreeMap<TargetRef, TermList>,
    #[serde(with = "crate::serde_support::optional_hex32")]
    pub overlay_hash: Option<[u8; 32]>,
    #[serde(with = "crate::serde_support::optional_hex32")]
    pub assumption_hash: Option<[u8; 32]>,
    #[serde(with = "crate::serde_support::hex32_set")]
    pub assumption_digests: BTreeSet<[u8; 32]>,
    #[serde(with = "crate::manifest::interpretation_digest_map")]
    pub interpretation_digests: BTreeMap<crate::atom::ProductionId, InterpretationDigest>,
}

impl ElaboratedGraph {
    pub fn wire_terms(&self) -> BTreeMap<WireId, TermList> {
        self.wires
            .iter()
            .filter_map(|(wire, elaborated)| {
                elaborated.terms.clone().map(|terms| (wire.clone(), terms))
            })
            .collect()
    }

    pub fn manifest_metadata(&self) -> crate::manifest::ManifestMetadata {
        crate::manifest::ManifestMetadata {
            overlay_hash: self.overlay_hash,
            assumption_hash: self.assumption_hash,
            assumption_digests: self.assumption_digests.clone(),
            interpretation_digests: self.interpretation_digests.clone(),
        }
    }
}

#[derive(Debug, Error)]
pub enum ElaborationError {
    #[error("node {node:?} at instantiation path {instantiation_path:?}: {source}")]
    Context {
        node: NodeId,
        instantiation_path: Vec<InstantiationFrame>,
        #[source]
        source: Box<ElaborationError>,
    },
    #[error(transparent)]
    Expression(#[from] ExprError),
    #[error(transparent)]
    Check(#[from] CheckError),
    #[error(transparent)]
    Terms(#[from] TermError),
    #[error(transparent)]
    Rewrite(#[from] RewriteError),
    #[error(transparent)]
    SymbolicBound(#[from] BoundError),
    #[error(transparent)]
    Bound(#[from] crate::ubound::UBoundError),
    #[error("node {node:?}: {message}")]
    Node { node: NodeId, message: String },
    #[error("wire {wire:?} is unavailable while elaborating node {node:?}")]
    MissingWire { node: NodeId, wire: WireRef },
    #[error("node {0:?} does not produce a matrix wire")]
    ExpectedMatrix(NodeId),
    #[error("compile parameter {0} is not bound with the declared kind")]
    MissingBinding(String),
    #[error("manifest {0:?} was not supplied")]
    MissingManifest(crate::atom::ProductionId),
    #[error("manifest artifact {artifact} does not exist in {production:?}")]
    MissingArtifact { production: crate::atom::ProductionId, artifact: String },
    #[error("structural node {0:?} is not valid in this graph position")]
    InvalidStructuralNode(NodeId),
    #[error("symbolic overlay entry {entry:?}: {message}")]
    Overlay { entry: Option<usize>, message: String },
}

#[derive(Clone)]
struct MatrixValue {
    ty: ConcreteMatrixType,
    terms: TermList,
}

#[derive(Clone)]
struct TrapdoorValue {
    ty: ConcreteWireType,
    uniform: AtomId,
    sigma: UBound,
}

#[derive(Clone)]
enum Value {
    Scalar(ConcreteWireType),
    Matrix(MatrixValue),
    Preimage(MatrixValue),
    Trapdoor(TrapdoorValue),
}

pub fn elaborate(graph: &Graph, bindings: &ParamEnv) -> Result<ElaboratedGraph, ElaborationError> {
    elaborate_with_overlay(graph, bindings, &[], &SymbolicOverlay::default())
}

pub fn elaborate_with_manifests(
    graph: &Graph,
    bindings: &ParamEnv,
    manifests: &[Manifest],
) -> Result<ElaboratedGraph, ElaborationError> {
    elaborate_with_overlay(graph, bindings, manifests, &SymbolicOverlay::default())
}

pub fn elaborate_with_overlay(
    graph: &Graph,
    bindings: &ParamEnv,
    manifests: &[Manifest],
    overlay: &SymbolicOverlay,
) -> Result<ElaboratedGraph, ElaborationError> {
    validate_overlay(overlay)
        .map_err(|message| ElaborationError::Overlay { entry: None, message })?;
    let (overlay_hash, assumption_hash) = overlay_hashes(overlay)
        .map_err(|message| ElaborationError::Overlay { entry: None, message })?;
    check_bindings(graph, bindings)?;
    for (name, declaration) in &overlay.virtual_atoms {
        concrete_matrix(&declaration.matrix_type, bindings).map_err(|error| {
            ElaborationError::Overlay {
                entry: None,
                message: format!("virtual atom {name} has an invalid matrix type: {error}"),
            }
        })?;
        if let VirtualKind::Bounded { norm } = &declaration.kind {
            evaluate_bound(norm, bindings).map_err(|error| ElaborationError::Overlay {
                entry: None,
                message: format!("virtual atom {name} has an invalid bound: {error}"),
            })?;
        }
    }
    check_topological(graph).map_err(|error| contextualize_check(error, &[]))?;
    let merged = merge_manifest_projections(manifests)
        .map_err(|error| ElaborationError::Node { node: NodeId(0), message: error.to_string() })?;
    let imported = merged
        .iter()
        .map(|(id, manifest)| {
            import_manifest(manifest).map(|imported| (id.clone(), imported)).map_err(|error| {
                ElaborationError::Node { node: NodeId(0), message: error.to_string() }
            })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let mut assumption_digests = BTreeSet::new();
    let mut interpretation_digests = BTreeMap::new();
    let mut target_terms = BTreeMap::new();
    for manifest in imported.values() {
        assumption_digests.extend(manifest.assumption_digests.iter().copied());
        for (target, terms) in &manifest.term_lists {
            if let Some(existing) = target_terms.insert(target.clone(), terms.clone()) &&
                existing != *terms
            {
                return Err(ElaborationError::Node {
                    node: NodeId(0),
                    message: format!("embedded manifest term list {target:?} disagrees"),
                });
            }
        }
        for (production, digest) in &manifest.interpretation_digests {
            interpretation_digests.insert(production.clone(), *digest);
        }
    }
    if let Some(hash) = assumption_hash {
        assumption_digests.insert(hash);
    }
    let mut state = State {
        bindings,
        manifests: &imported,
        graph,
        input_overrides: BTreeMap::new(),
        values: BTreeMap::new(),
        all_wires: BTreeMap::new(),
        wire_terms: BTreeMap::new(),
        atoms: AtomTable::default(),
        warnings: Vec::new(),
        path: Vec::new(),
        ring_expansion: ring_expansion(graph, bindings)?,
        assumed_terms: BTreeMap::new(),
        overlay,
        overlay_matches: vec![0; overlay.entries.len()],
        used_virtual_atoms: BTreeSet::new(),
        used_assumed_terms: BTreeSet::new(),
        creating_virtual_atoms: BTreeSet::new(),
        active_overlay_wire: None,
        selection_domains: BTreeMap::new(),
    };
    for imported in imported.values() {
        for atom in imported.atoms.values() {
            if let Some(existing) = state.atoms.insert(atom.clone()) &&
                existing != *atom
            {
                return Err(ElaborationError::Node {
                    node: NodeId(0),
                    message: format!("embedded manifest atom {:?} disagrees", atom.id),
                });
            }
        }
    }
    for node in &graph.nodes {
        state.elaborate_node_with_context(node)?;
    }
    for (index, count) in state.overlay_matches.iter().enumerate() {
        if *count == 0 {
            state.warnings.push(ElaborationWarning {
                node: overlay.entries[index].0.node,
                kind: WarningKind::UnusedOverlaySelector,
                message: format!("symbolic overlay selector {index} matched no wire"),
            });
        }
    }
    for name in overlay.virtual_atoms.keys() {
        if !state.used_virtual_atoms.contains(name) {
            state.warnings.push(ElaborationWarning {
                node: NodeId(0),
                kind: WarningKind::UnusedVirtualAtom,
                message: format!("virtual atom {name} is unused"),
            });
        }
    }
    for id in overlay.term_lists.keys() {
        if !state.used_assumed_terms.contains(id) {
            state.warnings.push(ElaborationWarning {
                node: NodeId(0),
                kind: WarningKind::UnusedAssumedTermList,
                message: format!("assumed term list {} is unused", id.0),
            });
        }
    }
    for (name, wire) in &graph.outputs {
        if !state.values.contains_key(wire) {
            return Err(contextualize(
                wire.node,
                &[],
                ElaborationError::Node {
                    node: wire.node,
                    message: format!(
                        "graph output {name} refers to unavailable port {}",
                        wire.port.0
                    ),
                },
            ));
        }
    }
    let wires = state
        .all_wires
        .iter()
        .map(|(wire, value)| {
            let elaborated = match value {
                Value::Scalar(ty) => ElaboratedWire { wire_type: ty.clone(), terms: None },
                Value::Matrix(matrix) => ElaboratedWire {
                    wire_type: ConcreteWireType::Matrix(matrix.ty.clone()),
                    terms: Some(matrix.terms.clone()),
                },
                Value::Preimage(matrix) => ElaboratedWire {
                    wire_type: ConcreteWireType::Preimage(matrix.ty.clone()),
                    terms: Some(matrix.terms.clone()),
                },
                Value::Trapdoor(trapdoor) => {
                    ElaboratedWire { wire_type: trapdoor.ty.clone(), terms: None }
                }
            };
            (wire.clone(), elaborated)
        })
        .collect();
    target_terms.extend(state.assumed_terms.clone());

    Ok(ElaboratedGraph {
        name: graph.name.clone(),
        source: graph.clone(),
        bindings: bindings.clone(),
        wires,
        outputs: graph.outputs.clone(),
        atoms: state.atoms,
        warnings: state.warnings,
        ring_expansion: state.ring_expansion,
        target_terms,
        overlay_hash,
        assumption_hash,
        assumption_digests,
        interpretation_digests,
    })
}

struct State<'a> {
    graph: &'a Graph,
    bindings: &'a ParamEnv,
    manifests: &'a BTreeMap<crate::atom::ProductionId, ImportedManifest>,
    input_overrides: BTreeMap<String, Value>,
    values: BTreeMap<WireRef, Value>,
    all_wires: BTreeMap<WireId, Value>,
    wire_terms: BTreeMap<WireId, TermList>,
    atoms: AtomTable,
    warnings: Vec<ElaborationWarning>,
    path: Vec<InstantiationFrame>,
    ring_expansion: UBound,
    assumed_terms: BTreeMap<TargetRef, TermList>,
    overlay: &'a SymbolicOverlay,
    overlay_matches: Vec<usize>,
    used_virtual_atoms: BTreeSet<String>,
    used_assumed_terms: BTreeSet<AssumedTermListId>,
    creating_virtual_atoms: BTreeSet<String>,
    active_overlay_wire: Option<WireId>,
    selection_domains: BTreeMap<WireId, SelectionDomain>,
}

impl State<'_> {
    fn elaborate_node_with_context(&mut self, node: &Node) -> Result<(), ElaborationError> {
        let path = self.path.clone();
        self.elaborate_node(node)
            .and_then(|()| self.apply_overlay_to_node(node.id))
            .map_err(|error| contextualize_unless_present(node.id, &path, error))
    }

    fn elaborate_node(&mut self, node: &Node) -> Result<(), ElaborationError> {
        match &node.kind {
            NodeKind::Input { name, wire_type, artifact } => {
                self.input(node, name, wire_type, artifact.as_ref())?
            }
            NodeKind::Output { .. } => {
                if node.args.is_empty() {
                    return self.node_error(node.id, "output requires at least one value");
                }
                let expected_type = value_type(self.argument(node, 0)?);
                for port in 0..node.args.len() {
                    let value = self.argument(node, port)?.clone();
                    if value_type(&value) != expected_type {
                        return self
                            .node_error(node.id, "output family members must have identical types");
                    }
                    self.insert_value(node.id, port as u32, value);
                }
            }
            NodeKind::ConstantInt(_) => {
                self.insert_value(node.id, 0, Value::Scalar(ConcreteWireType::ConstantInt));
            }
            NodeKind::ConstantReal(_) => {
                self.insert_value(node.id, 0, Value::Scalar(ConcreteWireType::ConstantReal));
            }
            NodeKind::ConstantBool(_) => {
                self.insert_value(node.id, 0, Value::Scalar(ConcreteWireType::ConstantBool));
            }
            NodeKind::ConstantMatrix { matrix_type, value } => {
                let ty = concrete_matrix(matrix_type, self.bindings)?;
                let terms = if matches!(value, ConstantMatrix::Zero) {
                    TermList::zero()
                } else {
                    let norm = constant_norm(value, &ty, self.bindings)?;
                    let id = self.constant_atom(value, &ty, norm)?;
                    TermList::atom(id)
                };
                self.insert_matrix(node.id, 0, ty, terms, false);
            }
            NodeKind::GadgetTrapdoor { matrix_type, base } => {
                let ty = concrete_matrix(matrix_type, self.bindings)?;
                let gadget = self.gadget_atom(&ty, base, false)?;
                let sigma = UBound::from_integer(&base.evaluate(self.bindings)?.abs())?;
                self.insert_value(
                    node.id,
                    0,
                    Value::Trapdoor(TrapdoorValue {
                        ty: ConcreteWireType::Trapdoor {
                            matrix: ty,
                            sigma: RealExpr::FromInt(base.clone()),
                        },
                        uniform: gadget,
                        sigma,
                    }),
                );
            }
            NodeKind::IntBinary(_) |
            NodeKind::IntToReal |
            NodeKind::BoolToInt |
            NodeKind::RealBinary(_) |
            NodeKind::RealSqrt => self.scalar_operation(node)?,
            NodeKind::IntCompare(_) => {
                self.require_scalar(node, 0, is_integer, "integer")?;
                self.require_scalar(node, 1, is_integer, "integer")?;
                self.insert_value(node.id, 0, Value::Scalar(ConcreteWireType::Bool));
            }
            NodeKind::BitExtract { bit } => {
                self.require_scalar(node, 0, is_integer, "integer")?;
                if bit.evaluate(self.bindings)?.sign() == num_bigint::Sign::Minus {
                    return self.node_error(node.id, "bit position must be nonnegative");
                }
                self.insert_value(node.id, 0, Value::Scalar(ConcreteWireType::Bool));
            }
            NodeKind::MatrixBinary(operation) => self.matrix_binary(node, *operation)?,
            NodeKind::MatrixNegate => {
                let input = self.matrix_argument(node, 0)?;
                self.insert_matrix(node.id, 0, input.ty, input.terms.negate(), false);
            }
            NodeKind::MatrixScale { scalar } => {
                let input = self.matrix_argument(node, 0)?;
                let scalar = scalar.evaluate(self.bindings)?;
                let terms = input.terms.scale(&scalar, &self.atoms)?;
                self.insert_matrix(node.id, 0, input.ty, terms, false);
            }
            NodeKind::Transpose => {
                let input = self.matrix_argument(node, 0)?;
                self.warn_viewed_preimages(node.id, &input.terms);
                let ty = ConcreteMatrixType {
                    rows: input.ty.columns,
                    columns: input.ty.rows,
                    ..input.ty
                };
                let terms = input.terms.transpose(&self.atoms)?;
                self.insert_matrix(node.id, 0, ty, terms, false);
            }
            NodeKind::Slice { rows, columns } => {
                let input = self.matrix_argument(node, 0)?;
                self.warn_viewed_preimages(node.id, &input.terms);
                let ty = sliced_type(&input.ty, rows.as_ref(), columns.as_ref())?;
                let terms = input.terms.slice(*rows, *columns, &self.atoms)?;
                self.insert_matrix(node.id, 0, ty, terms, false);
            }
            NodeKind::Tensor => self.derived_matrix(node, "tensor", None)?,
            NodeKind::Concat { axis } => self.derived_matrix(node, "concat", Some(*axis))?,
            NodeKind::Reshape { rows, columns } => {
                let input = self.matrix_argument(node, 0)?;
                self.require_reduced(node.id, &input.terms)?;
                let rows = positive_usize(rows.evaluate(self.bindings)?, "reshape rows", node.id)?;
                let columns =
                    positive_usize(columns.evaluate(self.bindings)?, "reshape columns", node.id)?;
                if rows.saturating_mul(columns) != input.ty.rows.saturating_mul(input.ty.columns) {
                    return self.node_error(node.id, "reshape changes the element count");
                }
                let input_atom = self.definition_atom(node.id, 1, &input)?;
                let kind =
                    self.atoms.get(&input_atom).expect("definition atom was inserted").kind.clone();
                let ty = ConcreteMatrixType { rows, columns, ..input.ty };
                let id = self.derived_atom(
                    node.id,
                    0,
                    ty.clone(),
                    DefExpr::Reshape { input: input_atom.clone(), rows, columns },
                    BTreeSet::from([input_atom]),
                    kind,
                );
                self.insert_matrix(node.id, 0, ty, TermList::atom(id), false);
            }
            NodeKind::UniformSample { matrix_type, range } => {
                let ty = concrete_matrix(matrix_type, self.bindings)?;
                if range.minimum > range.maximum {
                    return self.node_error(node.id, "uniform sample range is empty");
                }
                let maximum = range.minimum.abs().max(range.maximum.abs());
                let kind = if &maximum * BigInt::from(2) >= ty.modulus {
                    AtomKind::Large
                } else {
                    AtomKind::Bounded { norm: UBound::from_integer(&maximum)? }
                };
                let id = self.source_atom(node.id, 0, ty.clone(), kind, None);
                self.insert_matrix(node.id, 0, ty, TermList::atom(id), false);
            }
            NodeKind::GaussianSample { matrix_type, sigma } => {
                let ty = concrete_matrix(matrix_type, self.bindings)?;
                let norm = gaussian_norm(evaluate_bound(sigma, self.bindings)?, &ty);
                let id = self.source_atom(node.id, 0, ty.clone(), AtomKind::Bounded { norm }, None);
                self.insert_matrix(node.id, 0, ty, TermList::atom(id), false);
            }
            NodeKind::HashSample {
                matrix_type,
                variant,
                tag_prefix: _,
                tag_expressions,
                base,
                digit_count,
            } => {
                for expression in tag_expressions {
                    expression.evaluate(self.bindings)?;
                }
                self.hash_sample(node, matrix_type, *variant, base.as_ref(), digit_count.as_ref())?
            }
            NodeKind::TrapdoorSample { matrix_type, sigma } => {
                let ty = concrete_matrix(matrix_type, self.bindings)?;
                let uniform = self.source_atom(node.id, 0, ty.clone(), AtomKind::Large, None);
                self.insert_matrix(node.id, 0, ty.clone(), TermList::atom(uniform.clone()), false);
                self.insert_value(
                    node.id,
                    1,
                    Value::Trapdoor(TrapdoorValue {
                        ty: ConcreteWireType::Trapdoor { matrix: ty, sigma: sigma.clone() },
                        uniform,
                        sigma: evaluate_bound(sigma, self.bindings)?,
                    }),
                );
            }
            NodeKind::PreimageSample { matrix_type } => {
                let ty = concrete_matrix(matrix_type, self.bindings)?;
                let trapdoor = self.trapdoor_argument(node, 0)?;
                let target = *node.args.get(1).ok_or(ElaborationError::MissingWire {
                    node: node.id,
                    wire: WireRef { node: node.id, port: Port(1) },
                })?;
                let target_value = self.matrix_argument(node, 1)?;
                let uniform_type =
                    trapdoor.ty.matrix_type().expect("trapdoor always has a matrix type");
                let product = multiplication_type(uniform_type, &ty)?;
                check_add_shape(&product, &target_value.ty)?;
                let refs =
                    PreimageRefs { uniform: trapdoor.uniform, target: TargetRef::Local(target) };
                let id = self.source_atom(
                    node.id,
                    0,
                    ty.clone(),
                    AtomKind::Bounded { norm: trapdoor.sigma },
                    Some(refs),
                );
                self.insert_matrix(node.id, 0, ty, TermList::atom(id), true);
            }
            NodeKind::GadgetDecompose { base, small, digit_count } => {
                let target = *node.args.first().ok_or(ElaborationError::MissingWire {
                    node: node.id,
                    wire: WireRef { node: node.id, port: Port(0) },
                })?;
                let input = self.matrix_argument(node, 0)?;
                let base_value = base.evaluate(self.bindings)?;
                if base_value <= BigInt::one() {
                    return self.node_error(node.id, "gadget base must be greater than one");
                }
                let digits = decomposition_digits(
                    digit_count.as_ref(),
                    &input.ty.modulus,
                    &base_value,
                    self.bindings,
                    node.id,
                )?;
                let gadget_ty = ConcreteMatrixType {
                    columns: input.ty.rows.saturating_mul(digits),
                    ..input.ty.clone()
                };
                let output_ty = ConcreteMatrixType {
                    rows: gadget_ty.columns,
                    columns: input.ty.columns,
                    ..input.ty.clone()
                };
                let gadget = self.gadget_atom(&gadget_ty, base, *small)?;
                let norm = UBound::from_ratio(&base_value, &BigInt::from(2))?;
                let id = self.source_atom(
                    node.id,
                    0,
                    output_ty.clone(),
                    AtomKind::Bounded { norm },
                    Some(PreimageRefs { uniform: gadget, target: TargetRef::Local(target) }),
                );
                self.insert_matrix(node.id, 0, output_ty, TermList::atom(id), true);
            }
            NodeKind::ModDown { target_modulus } => {
                self.mod_down(node, target_modulus.evaluate(self.bindings)?)?
            }
            NodeKind::ModUp { target_modulus } => {
                self.mod_up(node, target_modulus.evaluate(self.bindings)?)?
            }
            NodeKind::ExtractCoefficient { position } => {
                let input = self.matrix_argument(node, 0)?;
                self.require_reduced(node.id, &input.terms)?;
                if !input.ty.is_scalar() {
                    return self.node_error(node.id, "extract coefficient requires a 1x1 matrix");
                }
                let position = position.evaluate(self.bindings)?;
                if position.sign() == num_bigint::Sign::Minus ||
                    position >= BigInt::from(input.ty.ring_dimension)
                {
                    return self.node_error(node.id, "coefficient position is out of range");
                }
                self.insert_value(node.id, 0, Value::Scalar(ConcreteWireType::Int));
            }
            NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
                let input = self.matrix_argument(node, 0)?;
                self.require_reduced(node.id, &input.terms)?;
                if !input.ty.is_scalar() {
                    return self.node_error(node.id, "threshold decode requires a 1x1 matrix");
                }
                let count =
                    positive_usize(length.evaluate(self.bindings)?, "decode length", node.id)?;
                if count > input.ty.ring_dimension {
                    return self.node_error(node.id, "decode length exceeds ring dimension");
                }
                if plaintext_modulus.evaluate(self.bindings)? <= BigInt::one() {
                    return self.node_error(node.id, "plaintext modulus must be greater than one");
                }
                let ty = if *output_bool { ConcreteWireType::Bool } else { ConcreteWireType::Int };
                for port in 0..count {
                    self.insert_value(node.id, port as u32, Value::Scalar(ty.clone()));
                }
            }
            NodeKind::Select { count } => self.select(node, count.evaluate(self.bindings)?)?,
            NodeKind::SubgraphCall(call) => self.subgraph_call(node, call)?,
            NodeKind::ParallelLoop(loop_node) => self.parallel_loop(node, loop_node)?,
        }
        Ok(())
    }

    fn input(
        &mut self,
        node: &Node,
        name: &str,
        wire_type: &WireType,
        artifact: Option<&crate::node::ArtifactInput>,
    ) -> Result<(), ElaborationError> {
        if let Some(value) = self.input_overrides.get(name).cloned() {
            self.insert_value(node.id, 0, value);
            return Ok(());
        }
        if let Some(artifact) = artifact {
            let imported = self
                .manifests
                .get(&artifact.production_id)
                .ok_or_else(|| ElaborationError::MissingManifest(artifact.production_id.clone()))?;
            let stored = imported.artifacts.get(&artifact.artifact_name).ok_or_else(|| {
                ElaborationError::MissingArtifact {
                    production: artifact.production_id.clone(),
                    artifact: artifact.artifact_name.clone(),
                }
            })?;
            let declared_type = concrete_wire(wire_type, self.bindings)?
                .matrix_type()
                .cloned()
                .ok_or_else(|| ElaborationError::Node {
                    node: node.id,
                    message: "manifest artifacts must be declared with a matrix-like type"
                        .to_owned(),
                })?;
            if declared_type != stored.wire_type {
                return self
                    .node_error(node.id, "declared artifact type does not match the manifest");
            }
            if let Some(family) = &stored.family {
                let expected = artifact
                    .family_count
                    .as_ref()
                    .map(|count| count.evaluate(self.bindings))
                    .transpose()?
                    .and_then(|count| count.to_usize())
                    .unwrap_or(family.len());
                if expected != family.len() {
                    return self.node_error(node.id, "artifact family count mismatch");
                }
                for (port, terms) in family.iter().enumerate() {
                    self.insert_matrix(
                        node.id,
                        port as u32,
                        stored.wire_type.clone(),
                        terms.clone(),
                        matches!(wire_type, WireType::Preimage(_)),
                    );
                }
            } else {
                if artifact.family_count.is_some() {
                    return self.node_error(
                        node.id,
                        "artifact declares a family count but the manifest entry is singular",
                    );
                }
                self.insert_matrix(
                    node.id,
                    0,
                    stored.wire_type.clone(),
                    stored.terms.clone(),
                    matches!(wire_type, WireType::Preimage(_)),
                );
            }
            return Ok(());
        }
        match concrete_wire(wire_type, self.bindings)? {
            ConcreteWireType::Matrix(ty) => {
                let atom = self.source_atom(node.id, 0, ty.clone(), AtomKind::Large, None);
                self.insert_matrix(node.id, 0, ty, TermList::atom(atom), false);
            }
            ConcreteWireType::Preimage(ty) => {
                let atom = self.source_atom(
                    node.id,
                    0,
                    ty.clone(),
                    AtomKind::Bounded { norm: modulus_half(&ty.modulus)? },
                    None,
                );
                self.insert_matrix(node.id, 0, ty, TermList::atom(atom), true);
            }
            ConcreteWireType::Trapdoor { matrix, sigma } => {
                let uniform = self.source_atom(node.id, 1, matrix.clone(), AtomKind::Large, None);
                let sigma_bound = evaluate_bound(&sigma, self.bindings)?;
                self.insert_value(
                    node.id,
                    0,
                    Value::Trapdoor(TrapdoorValue {
                        ty: ConcreteWireType::Trapdoor { matrix, sigma: sigma.clone() },
                        uniform,
                        sigma: sigma_bound,
                    }),
                );
            }
            scalar => self.insert_value(node.id, 0, Value::Scalar(scalar)),
        }
        Ok(())
    }

    fn scalar_operation(&mut self, node: &Node) -> Result<(), ElaborationError> {
        let ty = match node.kind {
            NodeKind::IntBinary(_) => {
                self.require_scalar(node, 0, is_integer, "integer")?;
                self.require_scalar(node, 1, is_integer, "integer")?;
                ConcreteWireType::Int
            }
            NodeKind::BoolToInt => {
                self.require_scalar(node, 0, is_boolean, "boolean")?;
                ConcreteWireType::Int
            }
            NodeKind::IntToReal => {
                self.require_scalar(node, 0, is_integer, "integer")?;
                ConcreteWireType::Real
            }
            NodeKind::RealBinary(_) => {
                self.require_scalar(node, 0, is_real, "real")?;
                self.require_scalar(node, 1, is_real, "real")?;
                ConcreteWireType::Real
            }
            NodeKind::RealSqrt => {
                self.require_scalar(node, 0, is_real, "real")?;
                ConcreteWireType::Real
            }
            _ => return self.node_error(node.id, "invalid scalar operation"),
        };
        self.insert_value(node.id, 0, Value::Scalar(ty));
        Ok(())
    }

    fn require_scalar(
        &self,
        node: &Node,
        index: usize,
        predicate: fn(&ConcreteWireType) -> bool,
        expected: &str,
    ) -> Result<(), ElaborationError> {
        let Value::Scalar(ty) = self.argument(node, index)? else {
            return self.node_error(node.id, &format!("expected {expected} scalar argument"));
        };
        if predicate(ty) {
            Ok(())
        } else {
            self.node_error(node.id, &format!("expected {expected} scalar argument"))
        }
    }

    fn matrix_binary(
        &mut self,
        node: &Node,
        operation: MatrixBinaryOp,
    ) -> Result<(), ElaborationError> {
        let left = self.matrix_argument(node, 0)?;
        let right = self.matrix_argument(node, 1)?;
        let (ty, terms) = match operation {
            MatrixBinaryOp::Add => {
                check_add_shape(&left.ty, &right.ty)?;
                (left.ty.clone(), left.terms.add(&right.terms, &self.atoms)?)
            }
            MatrixBinaryOp::Subtract => {
                check_add_shape(&left.ty, &right.ty)?;
                (left.ty.clone(), left.terms.sub(&right.terms, &self.atoms)?)
            }
            MatrixBinaryOp::Multiply => {
                let ty = multiplication_type(&left.ty, &right.ty)?;
                let expanded = left.terms.multiply(&right.terms, &self.atoms)?;
                let terms = rewrite_preimages(expanded, &self.atoms, self)?;
                (ty, terms)
            }
        };
        self.insert_matrix(node.id, 0, ty, terms, false);
        Ok(())
    }

    fn hash_sample(
        &mut self,
        node: &Node,
        matrix_type: &MatrixType,
        variant: HashVariant,
        base: Option<&crate::expr::IntExpr>,
        digit_count: Option<&crate::expr::IntExpr>,
    ) -> Result<(), ElaborationError> {
        let Value::Scalar(ConcreteWireType::Bytes { length: 32 }) = self.argument(node, 0)? else {
            return self.node_error(node.id, "hash sampling requires a 32-byte key");
        };
        for index in 1..node.args.len() {
            self.require_scalar(node, index, is_integer, "integer hash-tag component")?;
        }
        let ty = concrete_matrix(matrix_type, self.bindings)?;
        match variant {
            HashVariant::Plain => {
                let id = self.source_atom(node.id, 0, ty.clone(), AtomKind::Large, None);
                self.insert_matrix(node.id, 0, ty, TermList::atom(id), false);
            }
            HashVariant::Decomposed | HashVariant::SmallDecomposed => {
                let base = base.ok_or_else(|| ElaborationError::Node {
                    node: node.id,
                    message: "decomposed hash requires a base".to_owned(),
                })?;
                let base_value = base.evaluate(self.bindings)?.abs();
                if base_value <= BigInt::one() {
                    return self
                        .node_error(node.id, "decomposed hash base must be greater than one");
                }
                let digits = decomposition_digits(
                    digit_count,
                    &ty.modulus,
                    &base_value,
                    self.bindings,
                    node.id,
                )?;
                if ty.rows % digits != 0 {
                    return self.node_error(
                        node.id,
                        "decomposed hash output rows must be divisible by gadget digits",
                    );
                }
                let target_ty = ConcreteMatrixType {
                    rows: ty.rows / digits,
                    columns: ty.columns,
                    ..ty.clone()
                };
                let gadget_ty = ConcreteMatrixType {
                    rows: target_ty.rows,
                    columns: ty.rows,
                    ..target_ty.clone()
                };
                let target = self.source_atom(node.id, 1, target_ty.clone(), AtomKind::Large, None);
                let target_wire = WireRef { node: node.id, port: Port(1) };
                let target_terms = TermList::atom(target);
                let target_id = WireId { instantiation_path: self.path.clone(), wire: target_wire };
                self.wire_terms.insert(target_id.clone(), target_terms.clone());
                self.all_wires.insert(
                    target_id,
                    Value::Matrix(MatrixValue { ty: target_ty, terms: target_terms }),
                );
                let gadget = self.gadget_atom(
                    &gadget_ty,
                    base,
                    matches!(variant, HashVariant::SmallDecomposed),
                )?;
                let norm = UBound::from_ratio(&base_value, &BigInt::from(2))?;
                let id = self.source_atom(
                    node.id,
                    0,
                    ty.clone(),
                    AtomKind::Bounded { norm },
                    Some(PreimageRefs { uniform: gadget, target: TargetRef::Local(target_wire) }),
                );
                self.insert_matrix(node.id, 0, ty, TermList::atom(id), true);
            }
        }
        Ok(())
    }

    fn derived_matrix(
        &mut self,
        node: &Node,
        operation: &str,
        axis: Option<ConcatAxis>,
    ) -> Result<(), ElaborationError> {
        let inputs = (0..node.args.len())
            .map(|index| self.matrix_argument(node, index))
            .collect::<Result<Vec<_>, _>>()?;
        for input in &inputs {
            self.require_reduced(node.id, &input.terms)?;
        }
        let first = inputs.first().ok_or_else(|| ElaborationError::Node {
            node: node.id,
            message: format!("{operation} requires at least one input"),
        })?;
        let mut ty = first.ty.clone();
        match operation {
            "tensor" => {
                if inputs.len() != 2 {
                    return self.node_error(node.id, "tensor requires two inputs");
                }
                crate::checks::check_same_ring(&inputs[0].ty, &inputs[1].ty)?;
                ty.rows = ty.rows.saturating_mul(inputs[1].ty.rows);
                ty.columns = ty.columns.saturating_mul(inputs[1].ty.columns);
            }
            "concat" => {
                for input in &inputs[1..] {
                    crate::checks::check_same_ring(&ty, &input.ty)?;
                    match axis.expect("concat axis") {
                        ConcatAxis::Rows if ty.columns == input.ty.columns => {
                            ty.rows += input.ty.rows;
                        }
                        ConcatAxis::Columns if ty.rows == input.ty.rows => {
                            ty.columns += input.ty.columns;
                        }
                        ConcatAxis::Diagonal => {
                            ty.rows += input.ty.rows;
                            ty.columns += input.ty.columns;
                        }
                        _ => return self.node_error(node.id, "concat shape mismatch"),
                    }
                }
            }
            _ => unreachable!(),
        }
        let mut input_atoms = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            input_atoms.push(self.definition_atom(
                node.id,
                u32::try_from(index + 1).map_err(|_| ElaborationError::Node {
                    node: node.id,
                    message: "derived input count exceeds the atom port range".to_owned(),
                })?,
                input,
            )?);
        }
        let definition = match operation {
            "tensor" => {
                DefExpr::Tensor { left: input_atoms[0].clone(), right: input_atoms[1].clone() }
            }
            "concat" => {
                DefExpr::Concat { inputs: input_atoms.clone(), axis: axis.expect("concat axis") }
            }
            _ => unreachable!(),
        };
        let kind = if inputs.iter().any(|input| {
            input.terms.terms.iter().any(|term| {
                term.factors.iter().any(|factor| {
                    self.atoms
                        .get(&factor.atom)
                        .is_some_and(|atom| matches!(atom.kind, AtomKind::Large))
                })
            })
        }) {
            AtomKind::Large
        } else {
            let norms = inputs
                .iter()
                .map(|input| {
                    sum_norm(&input.terms, &self.atoms, &self.ring_expansion)
                        .map_err(ElaborationError::from)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let norm = match operation {
                "tensor" => self.ring_expansion.mul(&norms[0]).mul(&norms[1]),
                "concat" => {
                    norms.into_iter().fold(UBound::zero(), |acc, norm| UBound::max(&acc, &norm))
                }
                _ => unreachable!(),
            };
            AtomKind::Bounded { norm }
        };
        let id = self.derived_atom(
            node.id,
            0,
            ty.clone(),
            definition,
            input_atoms.into_iter().collect(),
            kind,
        );
        self.insert_matrix(node.id, 0, ty, TermList::atom(id), false);
        Ok(())
    }

    fn definition_atom(
        &mut self,
        node: NodeId,
        port: u32,
        input: &MatrixValue,
    ) -> Result<AtomId, ElaborationError> {
        if let [term] = input.terms.terms.as_slice() &&
            term.coefficient == BigInt::one() &&
            let [factor] = term.factors.as_slice() &&
            factor.view.is_none()
        {
            return Ok(factor.atom.clone());
        }
        let kind = derived_kind(&input.terms, &self.atoms, &self.ring_expansion)?;
        Ok(self.derived_atom(
            node,
            port,
            input.ty.clone(),
            DefExpr::TermList(input.terms.clone()),
            dependencies(&input.terms, &self.atoms),
            kind,
        ))
    }

    fn select(&mut self, node: &Node, count: BigInt) -> Result<(), ElaborationError> {
        let count = positive_usize(count, "select count", node.id)?;
        if node.args.len() != count + 1 {
            return self.node_error(node.id, "select argument count mismatch");
        }
        self.require_scalar(node, 0, is_integer, "integer select index")?;
        let branches = (1..=count)
            .map(|index| self.matrix_argument(node, index))
            .collect::<Result<Vec<_>, _>>()?;
        let ty = branches
            .first()
            .ok_or_else(|| ElaborationError::Node {
                node: node.id,
                message: "select has no branches".to_owned(),
            })?
            .ty
            .clone();
        for branch in &branches[1..] {
            check_add_shape(&ty, &branch.ty)?;
        }
        let index_wire = node.args[0];
        let domain = SelectionDomain {
            index_wire,
            instantiation_path: self.path.clone(),
            count: count as u64,
            modulus: ty.modulus.clone(),
            ring_dimension: ty.ring_dimension,
        };
        self.selection_domains.insert(
            WireId {
                instantiation_path: self.path.clone(),
                wire: WireRef { node: node.id, port: Port(0) },
            },
            domain.clone(),
        );
        let scalar = ConcreteMatrixType::scalar(ty.modulus.clone(), ty.ring_dimension);
        let mut terms = Vec::new();
        for (branch, input) in branches.iter().enumerate() {
            let indicator = AtomId::Indicator { domain: domain.clone(), branch: branch as u64 };
            if !self.atoms.contains_key(&indicator) {
                self.atoms.insert(Atom {
                    id: indicator.clone(),
                    class: AtomClass::Derived {
                        definition: DefExpr::Indicator { index_wire, branch: branch as u64 },
                    },
                    kind: AtomKind::Bounded { norm: UBound::one() },
                    matrix_type: scalar.clone(),
                    dependencies: BTreeSet::new(),
                    preimage_refs: None,
                });
            }
            for term in &input.terms.terms {
                let mut factors = vec![Factor { atom: indicator.clone(), view: None }];
                factors.extend(term.factors.iter().cloned());
                terms.push(Term { coefficient: term.coefficient.clone(), factors });
            }
        }
        let terms = TermList { terms }.canonicalize(&self.atoms)?;
        if !self.index_proven_in_range(index_wire, count) {
            self.warnings.push(ElaborationWarning {
                node: node.id,
                kind: WarningKind::RuntimeSelectBoundsCheck,
                message: "select index requires a runtime bounds check".to_owned(),
            });
        }
        self.insert_matrix(node.id, 0, ty, terms, false);
        Ok(())
    }

    fn index_proven_in_range(&self, wire: WireRef, count: usize) -> bool {
        let Some(node) = self.graph.node(wire.node) else {
            return false;
        };
        match &node.kind {
            NodeKind::ConstantInt(value) => {
                value.sign() != num_bigint::Sign::Minus &&
                    value.to_usize().is_some_and(|value| value < count)
            }
            NodeKind::BoolToInt | NodeKind::BitExtract { .. } => count >= 2,
            NodeKind::IntBinary(crate::node::IntBinaryOp::Remainder) => {
                let Some(divisor) = node.args.get(1).and_then(|wire| self.graph.node(wire.node))
                else {
                    return false;
                };
                matches!(
                    &divisor.kind,
                    NodeKind::ConstantInt(value)
                        if value > &BigInt::zero() &&
                            value.to_usize().is_some_and(|value| value <= count)
                )
            }
            _ => false,
        }
    }

    fn subgraph_call(
        &mut self,
        node: &Node,
        call: &crate::node::SubgraphCall,
    ) -> Result<(), ElaborationError> {
        let graph = self.graph.subgraphs.get(&call.graph).cloned().ok_or_else(|| {
            ElaborationError::Node {
                node: node.id,
                message: format!("subgraph {} does not exist", call.graph),
            }
        })?;
        let env = self.child_bindings(&call.bindings, None)?;
        let outputs = self.elaborate_child(
            &graph,
            env,
            node,
            InstantiationFrame { call: node.id, loop_index: None },
        )?;
        for (port, value) in outputs.into_iter().enumerate() {
            self.insert_value(node.id, port as u32, value);
        }
        Ok(())
    }

    fn parallel_loop(
        &mut self,
        node: &Node,
        loop_node: &crate::node::ParallelLoop,
    ) -> Result<(), ElaborationError> {
        let graph = self.graph.subgraphs.get(&loop_node.graph).cloned().ok_or_else(|| {
            ElaborationError::Node {
                node: node.id,
                message: format!("subgraph {} does not exist", loop_node.graph),
            }
        })?;
        let count = positive_usize(
            loop_node.count.evaluate(self.bindings)?,
            "parallel-loop count",
            node.id,
        )?;
        let mut reference_types: Option<Vec<ConcreteWireType>> = None;
        for index in 0..count {
            let mut env =
                self.child_bindings(&loop_node.bindings, Some((&loop_node.index_variable, index)))?;
            env.integers.insert(loop_node.index_variable.clone(), BigInt::from(index));
            let outputs = self.elaborate_child(
                &graph,
                env,
                node,
                InstantiationFrame { call: node.id, loop_index: Some(index as u64) },
            )?;
            let output_types = outputs.iter().map(value_type).collect::<Vec<_>>();
            if let Some(reference) = &reference_types {
                if reference != &output_types {
                    return self.node_error(
                        node.id,
                        "parallel-loop output shapes depend on the loop index",
                    );
                }
            } else {
                reference_types = Some(output_types);
            }
            let width = outputs.len();
            for (output_index, value) in outputs.into_iter().enumerate() {
                self.insert_value(node.id, (index * width + output_index) as u32, value);
            }
        }
        Ok(())
    }

    fn child_bindings(
        &self,
        bindings: &[(String, crate::expr::IntExpr)],
        loop_index: Option<(&str, usize)>,
    ) -> Result<ParamEnv, ElaborationError> {
        let mut env = self.bindings.clone();
        if let Some((name, index)) = loop_index {
            env.integers.insert(name.to_owned(), BigInt::from(index));
        }
        for (name, expression) in bindings {
            env.integers.insert(name.clone(), expression.evaluate(&env)?);
        }
        Ok(env)
    }

    fn elaborate_child(
        &mut self,
        graph: &Graph,
        bindings: ParamEnv,
        call_node: &Node,
        frame: InstantiationFrame,
    ) -> Result<Vec<Value>, ElaborationError> {
        check_bindings(graph, &bindings)?;
        let mut path = self.path.clone();
        path.push(frame);
        check_topological(graph).map_err(|error| contextualize_check(error, &path))?;
        let input_nodes = graph
            .nodes
            .iter()
            .filter_map(|node| match &node.kind {
                NodeKind::Input { name, .. } => Some((name.clone(), node)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if input_nodes.len() != call_node.args.len() {
            return self
                .node_error(call_node.id, "subgraph input count does not match call arguments");
        }
        let input_overrides = input_nodes
            .iter()
            .zip(&call_node.args)
            .map(|((name, _), wire)| {
                self.values
                    .get(wire)
                    .cloned()
                    .map(|value| (name.clone(), value))
                    .ok_or(ElaborationError::MissingWire { node: call_node.id, wire: *wire })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let child_gamma = ring_expansion(graph, &bindings)?;
        let mut child = State {
            graph,
            bindings: &bindings,
            manifests: self.manifests,
            input_overrides,
            values: BTreeMap::new(),
            all_wires: self.all_wires.clone(),
            wire_terms: self.wire_terms.clone(),
            atoms: self.atoms.clone(),
            warnings: self.warnings.clone(),
            path,
            ring_expansion: UBound::max(&self.ring_expansion, &child_gamma),
            assumed_terms: self.assumed_terms.clone(),
            overlay: self.overlay,
            overlay_matches: self.overlay_matches.clone(),
            used_virtual_atoms: self.used_virtual_atoms.clone(),
            used_assumed_terms: self.used_assumed_terms.clone(),
            creating_virtual_atoms: self.creating_virtual_atoms.clone(),
            active_overlay_wire: None,
            selection_domains: self.selection_domains.clone(),
        };
        for child_node in &graph.nodes {
            child.elaborate_node_with_context(child_node)?;
        }
        let outputs = graph
            .outputs
            .values()
            .map(|wire| {
                child
                    .values
                    .get(wire)
                    .cloned()
                    .ok_or(ElaborationError::MissingWire { node: call_node.id, wire: *wire })
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.atoms = child.atoms;
        self.wire_terms = child.wire_terms;
        self.all_wires = child.all_wires;
        self.warnings = child.warnings;
        self.ring_expansion = child.ring_expansion;
        self.assumed_terms = child.assumed_terms;
        self.overlay_matches = child.overlay_matches;
        self.used_virtual_atoms = child.used_virtual_atoms;
        self.used_assumed_terms = child.used_assumed_terms;
        self.creating_virtual_atoms = child.creating_virtual_atoms;
        self.selection_domains = child.selection_domains;
        Ok(outputs)
    }

    fn mod_down(&mut self, node: &Node, target: BigInt) -> Result<(), ElaborationError> {
        let input = self.matrix_argument(node, 0)?;
        if target <= BigInt::one() || target >= input.ty.modulus {
            return self.node_error(node.id, "mod-down target must satisfy 1 < p < q");
        }
        self.require_reduced(node.id, &input.terms)?;
        check_mod_down_normal_form(&input.terms, &self.atoms)?;
        let output_ty = ConcreteMatrixType { modulus: target.clone(), ..input.ty.clone() };
        let mut signal = Vec::new();
        let mut bounded = Vec::new();
        let mut prefix_rounding = UBound::zero();
        for term in &input.terms.terms {
            let large_position = term.factors.iter().position(|factor| {
                self.atoms
                    .get(&factor.atom)
                    .is_some_and(|atom| matches!(atom.kind, AtomKind::Large))
            });
            if let Some(position) = large_position {
                let prefix = Term {
                    coefficient: term.coefficient.clone(),
                    factors: term.factors[..position].to_vec(),
                };
                let mut contribution = term_norm(&prefix, &self.atoms, &self.ring_expansion)?
                    .mul(&self.ring_expansion);
                if let Some(last) = prefix.factors.last() {
                    let last_atom = self
                        .atoms
                        .get(&last.atom)
                        .ok_or_else(|| TermError::MissingAtom(last.atom.clone()))?;
                    if !last_atom.matrix_type.is_scalar() {
                        contribution = contribution
                            .mul(&UBound::from_u64(last_atom.matrix_type.columns as u64));
                    }
                }
                prefix_rounding = prefix_rounding
                    .add(&contribution.mul(&UBound::from_ratio(&BigInt::one(), &BigInt::from(2))?));
                let mut factors = term.factors.clone();
                for factor in &mut factors[..position] {
                    factor.view.get_or_insert_with(ViewDescriptor::default).modulus_cast =
                        Some(target.clone());
                }
                let source = factors[position].atom.clone();
                let source_atom = self
                    .atoms
                    .get(&source)
                    .ok_or_else(|| TermError::MissingAtom(source.clone()))?;
                let image = self.derived_atom(
                    node.id,
                    (signal.len() + 1) as u32,
                    ConcreteMatrixType {
                        modulus: target.clone(),
                        ..source_atom.matrix_type.clone()
                    },
                    DefExpr::ModDownImage {
                        source: source.clone(),
                        source_modulus: input.ty.modulus.clone(),
                        target_modulus: target.clone(),
                    },
                    BTreeSet::from([source]),
                    AtomKind::Large,
                );
                factors[position] = Factor { atom: image, view: None };
                signal.push(Term { coefficient: term.coefficient.clone(), factors });
            } else {
                bounded.push(term.clone());
            }
        }
        self.warn_viewed_preimages(node.id, &input.terms);
        let bounded_terms = TermList { terms: bounded };
        let scaled = sum_norm(&bounded_terms, &self.atoms, &self.ring_expansion)?
            .mul(&UBound::from_ratio(&target, &input.ty.modulus)?);
        let error_norm = UBound::from_ratio(&BigInt::one(), &BigInt::from(2))?
            .add(&prefix_rounding)
            .add(&scaled);
        let signal_terms = TermList { terms: signal.clone() };
        let error = self.derived_atom(
            node.id,
            0,
            output_ty.clone(),
            DefExpr::ModDownError {
                input: node.args[0],
                signal: signal_terms,
                source_modulus: input.ty.modulus.clone(),
                target_modulus: target.clone(),
            },
            dependencies(&input.terms, &self.atoms),
            AtomKind::Bounded { norm: error_norm },
        );
        signal.push(Term {
            coefficient: BigInt::one(),
            factors: vec![Factor { atom: error, view: None }],
        });
        self.insert_matrix(
            node.id,
            0,
            output_ty,
            TermList { terms: signal }.canonicalize(&self.atoms)?,
            false,
        );
        Ok(())
    }

    fn mod_up(&mut self, node: &Node, target: BigInt) -> Result<(), ElaborationError> {
        let input = self.matrix_argument(node, 0)?;
        if target <= input.ty.modulus {
            return self.node_error(node.id, "mod-up target must be greater than q");
        }
        self.require_reduced(node.id, &input.terms)?;
        let output_ty = ConcreteMatrixType { modulus: target.clone(), ..input.ty.clone() };
        let mut lifted = input.terms.clone();
        for term in &mut lifted.terms {
            for factor in &mut term.factors {
                let atom = self
                    .atoms
                    .get(&factor.atom)
                    .ok_or_else(|| TermError::MissingAtom(factor.atom.clone()))?
                    .clone();
                match atom.kind {
                    AtomKind::Bounded { .. } => {
                        factor.view.get_or_insert_with(ViewDescriptor::default).modulus_cast =
                            Some(target.clone());
                    }
                    AtomKind::Large => {
                        let source = factor.atom.clone();
                        factor.atom = self.derived_atom(
                            node.id,
                            (self.atoms.len() + 1) as u32,
                            ConcreteMatrixType { modulus: target.clone(), ..atom.matrix_type },
                            DefExpr::ModUpLift {
                                source: source.clone(),
                                source_modulus: input.ty.modulus.clone(),
                                target_modulus: target.clone(),
                            },
                            BTreeSet::from([source]),
                            AtomKind::Large,
                        );
                        factor.view = None;
                    }
                }
            }
        }
        self.warn_viewed_preimages(node.id, &input.terms);
        let integer_norm =
            int_norm_sum(&input.terms, &self.atoms, &self.ring_expansion, &input.ty.modulus)?;
        let u_norm = integer_norm
            .div(&UBound::from_integer(&input.ty.modulus)?)?
            .add(&UBound::from_ratio(&BigInt::one(), &BigInt::from(2))?);
        let error = self.derived_atom(
            node.id,
            0,
            output_ty.clone(),
            DefExpr::ModUpError {
                input: node.args[0],
                lifted: lifted.clone(),
                source_modulus: input.ty.modulus.clone(),
                target_modulus: target.clone(),
            },
            dependencies(&input.terms, &self.atoms),
            AtomKind::Bounded { norm: u_norm },
        );
        lifted.terms.push(Term {
            coefficient: input.ty.modulus.clone(),
            factors: vec![Factor { atom: error, view: None }],
        });
        self.insert_matrix(node.id, 0, output_ty, lifted.canonicalize(&self.atoms)?, false);
        Ok(())
    }

    fn apply_overlay_to_node(&mut self, node: NodeId) -> Result<(), ElaborationError> {
        if self.overlay.entries.is_empty() {
            return Ok(());
        }
        let wires = self
            .all_wires
            .keys()
            .filter(|wire| wire.instantiation_path == self.path && wire.wire.node == node)
            .cloned()
            .collect::<Vec<_>>();
        for wire_id in wires {
            let matches = self
                .overlay
                .entries
                .iter()
                .enumerate()
                .filter_map(|(index, (selector, reinterpretation))| {
                    selector_matches(
                        selector,
                        &wire_id.instantiation_path,
                        wire_id.wire.node,
                        wire_id.wire.port.0,
                    )
                    .map(|bindings| (index, bindings, reinterpretation.clone()))
                })
                .collect::<Vec<_>>();
            if matches.len() > 1 {
                return Err(ElaborationError::Overlay {
                    entry: None,
                    message: format!(
                        "multiple overlay selectors resolve to concrete wire {wire_id:?}"
                    ),
                });
            }
            let Some((entry_index, bindings, reinterpretation)) = matches.into_iter().next() else {
                continue;
            };
            self.overlay_matches[entry_index] += 1;
            let value = self
                .all_wires
                .get(&wire_id)
                .cloned()
                .ok_or(ElaborationError::MissingWire { node, wire: wire_id.wire })?;
            let (matrix, preimage) = match value {
                Value::Matrix(matrix) => (matrix, false),
                Value::Preimage(matrix) => (matrix, true),
                _ => {
                    return Err(ElaborationError::Overlay {
                        entry: Some(entry_index),
                        message: format!("selected wire {wire_id:?} is not matrix-like"),
                    });
                }
            };
            self.active_overlay_wire = Some(wire_id.clone());
            let result = match reinterpretation {
                Reinterpretation::Unfold(spec) => {
                    self.apply_unfold(entry_index, &wire_id, matrix, preimage, &bindings, &spec)?
                }
                Reinterpretation::Fold(spec) => {
                    self.apply_fold(entry_index, &wire_id, matrix, &bindings, &spec)?
                }
            };
            self.active_overlay_wire = None;
            let matrix = result;
            let value = if preimage {
                Value::Preimage(matrix.clone())
            } else {
                Value::Matrix(matrix.clone())
            };
            self.wire_terms.insert(wire_id.clone(), matrix.terms);
            self.all_wires.insert(wire_id.clone(), value.clone());
            self.values.insert(wire_id.wire, value);
        }
        Ok(())
    }

    fn apply_unfold(
        &mut self,
        entry: usize,
        wire: &WireId,
        current: MatrixValue,
        preimage_wire: bool,
        bindings: &BTreeMap<String, u64>,
        spec: &crate::overlay::UnfoldSpec,
    ) -> Result<MatrixValue, ElaborationError> {
        let single_source = current.terms.terms.as_slice().first().is_some_and(|term| {
            current.terms.terms.len() == 1 &&
                term.coefficient == BigInt::one() &&
                term.factors.len() == 1 &&
                term.factors[0].view.is_none() &&
                self.atoms
                    .get(&term.factors[0].atom)
                    .is_some_and(|atom| matches!(atom.class, AtomClass::Source))
        });
        if !single_source && !spec.replace_derived {
            return Err(self.overlay_error(
                entry,
                "unfolding a derived description requires replace_derived: true",
            ));
        }
        if !single_source {
            self.warnings.push(ElaborationWarning {
                node: wire.wire.node,
                kind: WarningKind::ReplacedDerivedDescription,
                message: format!("overlay entry {entry} replaced a derived description"),
            });
        }
        if preimage_wire ||
            current.terms.terms.iter().any(|term| {
                term.factors.iter().any(|factor| {
                    self.atoms.get(&factor.atom).is_some_and(|atom| atom.preimage_refs.is_some())
                })
            })
        {
            self.warnings.push(ElaborationWarning {
                node: wire.wire.node,
                kind: WarningKind::DroppedPreimageReferences,
                message: format!("overlay entry {entry} discarded preimage references"),
            });
        }
        let assumed = self.resolve_assumed_terms(&spec.new_terms, bindings)?;
        self.check_term_list_type(entry, &assumed, &current.ty)?;
        let current_large = current.terms.contains_large(&self.atoms)?;
        let assumed_large = assumed.contains_large(&self.atoms)?;
        if current_large != assumed_large {
            return Err(self.overlay_error(
                entry,
                "unfold kind character differs from the current description",
            ));
        }
        if !current_large {
            let old = sum_norm(&current.terms, &self.atoms, &self.ring_expansion)?;
            let new = sum_norm(&assumed, &self.atoms, &self.ring_expansion)?;
            if new < old {
                self.warnings.push(ElaborationWarning {
                    node: wire.wire.node,
                    kind: WarningKind::StrengthenedBound,
                    message: format!("overlay entry {entry} assumes a tighter bound"),
                });
            }
        }
        let terms = rewrite_preimages(assumed, &self.atoms, self)?;
        Ok(MatrixValue { ty: current.ty, terms })
    }

    fn apply_fold(
        &mut self,
        entry: usize,
        wire: &WireId,
        current: MatrixValue,
        bindings: &BTreeMap<String, u64>,
        spec: &crate::overlay::FoldSpec,
    ) -> Result<MatrixValue, ElaborationError> {
        let positions = spec
            .expected
            .entries
            .iter()
            .map(|expected| self.resolve_expected_entry(entry, expected, bindings))
            .collect::<Result<Vec<_>, _>>()?;
        let expected = TermList { terms: positions.iter().flatten().cloned().collect() }
            .canonicalize(&self.atoms)?;
        if expected != current.terms {
            return Err(self.overlay_error(
                entry,
                &format!(
                    "fold intent mismatch: expected {:?}, actual {:?}",
                    expected, current.terms
                ),
            ));
        }
        self.check_fold_partition(entry, positions.len(), &spec.groups)?;
        let mut output = Vec::new();
        for (group_index, group) in spec.groups.iter().enumerate() {
            let members = group
                .terms()
                .iter()
                .flat_map(|position| positions[*position].iter().cloned())
                .collect::<Vec<_>>();
            if members.is_empty() {
                continue;
            }
            match group {
                FoldGroup::Keep { .. } => output.extend(members),
                FoldGroup::Residual { .. } => {
                    let folded = TermList { terms: members };
                    if folded.contains_large(&self.atoms)? {
                        return Err(self.overlay_error(entry, "Residual fold contains a large atom"));
                    }
                    let atom = self.create_fold_atom(
                        entry,
                        wire,
                        group_index,
                        current.ty.clone(),
                        folded,
                    )?;
                    output.push(Term {
                        coefficient: BigInt::one(),
                        factors: vec![Factor { atom, view: None }],
                    });
                }
                FoldGroup::Signal { suffix_len, .. } => {
                    let suffix_len = usize::try_from(*suffix_len).map_err(|_| {
                        self.overlay_error(entry, "Signal suffix length does not fit usize")
                    })?;
                    let first_suffix = signal_suffix(&members[0], suffix_len, &self.atoms)
                        .ok_or_else(|| self.overlay_error(entry, "Signal suffix is too long"))?;
                    let mut prefixes = Vec::new();
                    for member in members {
                        let suffix =
                            signal_suffix(&member, suffix_len, &self.atoms).ok_or_else(|| {
                                self.overlay_error(entry, "Signal suffix is too long")
                            })?;
                        if suffix != first_suffix {
                            return Err(self.overlay_error(
                                entry,
                                "Signal fold members do not share an identical suffix",
                            ));
                        }
                        let prefix_len = member.factors.len() - suffix.len();
                        let prefix = Term {
                            coefficient: member.coefficient,
                            factors: member.factors[..prefix_len].to_vec(),
                        };
                        if prefix.factors.iter().any(|factor| {
                            self.atoms
                                .get(&factor.atom)
                                .is_some_and(|atom| matches!(atom.kind, AtomKind::Large))
                        }) {
                            return Err(self
                                .overlay_error(entry, "Signal fold prefix contains a large atom"));
                        }
                        prefixes.push(prefix);
                    }
                    let folded = TermList { terms: prefixes }.canonicalize(&self.atoms)?;
                    if folded.terms.is_empty() {
                        continue;
                    }
                    let prefix_ty = if folded.terms[0].factors.is_empty() {
                        ConcreteMatrixType::scalar(
                            current.ty.modulus.clone(),
                            current.ty.ring_dimension,
                        )
                    } else {
                        self.term_type(entry, &folded.terms[0])?
                    };
                    let atom =
                        self.create_fold_atom(entry, wire, group_index, prefix_ty, folded)?;
                    let mut factors = vec![Factor { atom, view: None }];
                    factors.extend(first_suffix);
                    output.push(Term { coefficient: BigInt::one(), factors });
                }
            }
        }
        Ok(MatrixValue {
            ty: current.ty,
            terms: TermList { terms: output }.canonicalize(&self.atoms)?,
        })
    }

    fn create_fold_atom(
        &mut self,
        entry: usize,
        wire: &WireId,
        group_index: usize,
        matrix_type: ConcreteMatrixType,
        definition: TermList,
    ) -> Result<AtomId, ElaborationError> {
        if definition.terms.iter().any(|term| {
            term.factors.iter().any(|factor| {
                self.atoms.get(&factor.atom).is_some_and(|atom| atom.preimage_refs.is_some())
            })
        }) {
            self.warnings.push(ElaborationWarning {
                node: wire.wire.node,
                kind: WarningKind::DroppedPreimageReferences,
                message: format!("overlay fold entry {entry} absorbed preimage references"),
            });
        }
        let norm = sum_norm(&definition, &self.atoms, &self.ring_expansion)?;
        let id = AtomId::Overlay {
            instantiation_path: wire.instantiation_path.clone(),
            node: wire.wire.node,
            port: wire.wire.port.0,
            group_index: u32::try_from(group_index)
                .map_err(|_| self.overlay_error(entry, "fold group index exceeds u32"))?,
        };
        let atom = Atom {
            id: id.clone(),
            class: AtomClass::Derived { definition: DefExpr::Fold(definition.clone()) },
            kind: AtomKind::Bounded { norm },
            matrix_type,
            dependencies: dependencies(&definition, &self.atoms),
            preimage_refs: None,
        };
        if let Some(existing) = self.atoms.insert(atom.clone()) &&
            existing != atom
        {
            return Err(self.overlay_error(entry, "fold atom identity has conflicting definitions"));
        }
        Ok(id)
    }

    fn resolve_expected_entry(
        &mut self,
        entry: usize,
        expected: &ExpectedEntry,
        bindings: &BTreeMap<String, u64>,
    ) -> Result<Vec<Term>, ElaborationError> {
        match expected {
            ExpectedEntry::Term(term) => {
                let term = self.resolve_overlay_term(entry, term, bindings, true)?;
                Ok((!term.coefficient.is_zero()).then_some(term).into_iter().collect())
            }
            ExpectedEntry::IndicatorSum { select, index_var, body } => {
                let path = self.resolve_reference_path(entry, &select.path, bindings)?;
                let select_wire = WireId {
                    instantiation_path: path.clone(),
                    wire: WireRef { node: select.node, port: Port(0) },
                };
                if self.active_overlay_wire.as_ref().is_some_and(|target| {
                    target.instantiation_path == select_wire.instantiation_path &&
                        target.wire.node == select_wire.wire.node
                }) {
                    return Err(self
                        .overlay_error(entry, "IndicatorSum select must precede the target wire"));
                }
                let domain =
                    self.selection_domains.get(&select_wire).cloned().ok_or_else(|| {
                        self.overlay_error(
                            entry,
                            "IndicatorSum selector does not address an elaborated Select",
                        )
                    })?;
                if domain.count == 0 {
                    return Err(self.overlay_error(entry, "IndicatorSum domain is empty"));
                }
                let mut result = Vec::new();
                for branch in 0..domain.count {
                    let mut branch_bindings = bindings.clone();
                    branch_bindings.insert(index_var.clone(), branch);
                    let mut term =
                        self.resolve_overlay_term(entry, body, &branch_bindings, true)?;
                    if term.coefficient.is_zero() {
                        continue;
                    }
                    term.factors.insert(
                        0,
                        Factor {
                            atom: AtomId::Indicator { domain: domain.clone(), branch },
                            view: None,
                        },
                    );
                    result.push(term);
                }
                Ok(result)
            }
        }
    }

    fn resolve_assumed_terms(
        &mut self,
        id: &AssumedTermListId,
        bindings: &BTreeMap<String, u64>,
    ) -> Result<TermList, ElaborationError> {
        self.used_assumed_terms.insert(id.clone());
        let target = TargetRef::Assumed(id.clone());
        if let Some(terms) = self.assumed_terms.get(&target) {
            return Ok(terms.clone());
        }
        let declaration = self.overlay.term_lists.get(id).cloned().ok_or_else(|| {
            self.overlay_error_without_entry(&format!("assumed term list {} is undeclared", id.0))
        })?;
        let terms = TermList {
            terms: declaration
                .terms
                .iter()
                .map(|term| self.resolve_overlay_term(0, term, bindings, false))
                .collect::<Result<_, _>>()?,
        }
        .canonicalize(&self.atoms)?;
        self.assumed_terms.insert(target, terms.clone());
        Ok(terms)
    }

    fn resolve_overlay_term(
        &mut self,
        entry: usize,
        term: &OverlayTerm,
        bindings: &BTreeMap<String, u64>,
        allow_node: bool,
    ) -> Result<Term, ElaborationError> {
        let coefficient = term.coefficient.evaluate(self.bindings)?;
        let factors = term
            .factors
            .iter()
            .map(|factor| {
                Ok(Factor {
                    atom: self.resolve_atom_ref(entry, &factor.atom, bindings, allow_node)?,
                    view: factor.view.clone(),
                })
            })
            .collect::<Result<_, ElaborationError>>()?;
        Ok(Term { coefficient, factors })
    }

    fn resolve_atom_ref(
        &mut self,
        entry: usize,
        reference: &AtomRef,
        bindings: &BTreeMap<String, u64>,
        allow_node: bool,
    ) -> Result<AtomId, ElaborationError> {
        match reference {
            AtomRef::Constant { kind, params } => {
                let id = AtomId::Constant { kind: kind.clone(), params: params.clone() };
                if !self.atoms.contains_key(&id) {
                    if kind != "matrix" || params.len() != 5 {
                        return Err(self.overlay_error(
                            entry,
                            &format!("constant atom {id:?} has an unsupported identity"),
                        ));
                    }
                    let value =
                        serde_json::from_str::<ConstantMatrix>(&params[0]).map_err(|error| {
                            self.overlay_error(
                                entry,
                                &format!("constant atom value is invalid: {error}"),
                            )
                        })?;
                    let parse_integer = |value: &str, label: &str| {
                        value.parse::<BigInt>().map_err(|error| {
                            self.overlay_error(
                                entry,
                                &format!("constant atom {label} is invalid: {error}"),
                            )
                        })
                    };
                    let parse_usize = |value: &str, label: &str| {
                        value.parse::<usize>().map_err(|error| {
                            self.overlay_error(
                                entry,
                                &format!("constant atom {label} is invalid: {error}"),
                            )
                        })
                    };
                    let matrix_type = ConcreteMatrixType {
                        modulus: parse_integer(&params[1], "modulus")?,
                        ring_dimension: parse_usize(&params[2], "ring dimension")?,
                        rows: parse_usize(&params[3], "rows")?,
                        columns: parse_usize(&params[4], "columns")?,
                    };
                    if matrix_type.modulus <= BigInt::one() ||
                        matrix_type.ring_dimension == 0 ||
                        matrix_type.rows == 0 ||
                        matrix_type.columns == 0
                    {
                        return Err(self.overlay_error(
                            entry,
                            "constant atom matrix type must have positive dimensions and modulus > 1",
                        ));
                    }
                    let norm = constant_norm(&value, &matrix_type, self.bindings)?;
                    self.atoms.insert(Atom {
                        id: id.clone(),
                        class: AtomClass::Source,
                        kind: AtomKind::Bounded { norm },
                        matrix_type,
                        dependencies: BTreeSet::new(),
                        preimage_refs: None,
                    });
                }
                Ok(id)
            }
            AtomRef::Imported { production_id, manifest_atom_id } => {
                let id = AtomId::Imported {
                    production_id: production_id.clone(),
                    manifest_atom_id: *manifest_atom_id,
                };
                self.atoms.contains_key(&id).then_some(id.clone()).ok_or_else(|| {
                    self.overlay_error(entry, &format!("imported atom {id:?} is absent"))
                })
            }
            AtomRef::Virtual { name } => self.ensure_virtual_atom(entry, name, bindings),
            AtomRef::Node { path, node, port } => {
                if !allow_node {
                    return Err(self.overlay_error(
                        entry,
                        "Node references are forbidden in global declarations",
                    ));
                }
                let path = self.resolve_reference_path(entry, path, bindings)?;
                let port = match port {
                    PortMatcher::Concrete(port) => *port,
                    PortMatcher::Affine { var, stride, offset } => {
                        let value = bindings.get(var).ok_or_else(|| {
                            self.overlay_error(entry, &format!("port variable {var} is unbound"))
                        })?;
                        let port = u64::from(*stride)
                            .checked_mul(*value)
                            .and_then(|port| port.checked_add(u64::from(*offset)))
                            .ok_or_else(|| {
                                self.overlay_error(entry, "resolved port arithmetic overflow")
                            })?;
                        u32::try_from(port)
                            .map_err(|_| self.overlay_error(entry, "resolved port exceeds u32"))?
                    }
                };
                let wire = WireId {
                    instantiation_path: path,
                    wire: WireRef { node: *node, port: Port(port) },
                };
                if self.active_overlay_wire.as_ref().is_some_and(|target| {
                    target.instantiation_path == wire.instantiation_path &&
                        target.wire.node == wire.wire.node
                }) {
                    return Err(self.overlay_error(
                        entry,
                        "a wire never precedes another port of its own node",
                    ));
                }
                let terms = self.wire_terms.get(&wire).ok_or_else(|| {
                    self.overlay_error(entry, &format!("referenced wire {wire:?} is unavailable"))
                })?;
                let [term] = terms.terms.as_slice() else {
                    return Err(self.overlay_error(entry, "Node reference is not a single term"));
                };
                let [factor] = term.factors.as_slice() else {
                    return Err(self.overlay_error(entry, "Node reference is not a single factor"));
                };
                if term.coefficient != BigInt::one() || factor.view.is_some() {
                    return Err(self.overlay_error(
                        entry,
                        "Node reference must have coefficient one and no view",
                    ));
                }
                Ok(factor.atom.clone())
            }
        }
    }

    fn ensure_virtual_atom(
        &mut self,
        entry: usize,
        name: &str,
        bindings: &BTreeMap<String, u64>,
    ) -> Result<AtomId, ElaborationError> {
        let id = AtomId::Virtual { name: name.to_owned() };
        self.used_virtual_atoms.insert(name.to_owned());
        if self.atoms.contains_key(&id) {
            return Ok(id);
        }
        if !self.creating_virtual_atoms.insert(name.to_owned()) {
            return Err(self.overlay_error(entry, "virtual preimage declarations are cyclic"));
        }
        let declaration = self.overlay.virtual_atoms.get(name).cloned().ok_or_else(|| {
            self.overlay_error(entry, &format!("virtual atom {name} is undeclared"))
        })?;
        let matrix_type = concrete_matrix(&declaration.matrix_type, self.bindings)?;
        let kind = match declaration.kind {
            VirtualKind::Large => AtomKind::Large,
            VirtualKind::Bounded { norm } => {
                AtomKind::Bounded { norm: evaluate_bound(&norm, self.bindings)? }
            }
        };
        let preimage_refs = declaration
            .preimage
            .as_ref()
            .map(|preimage| {
                if !self.overlay.term_lists.contains_key(&preimage.target) {
                    return Err(self.overlay_error(
                        entry,
                        &format!("assumed target {} is undeclared", preimage.target.0),
                    ));
                }
                Ok(PreimageRefs {
                    uniform: self.resolve_atom_ref(entry, &preimage.uniform, bindings, false)?,
                    target: TargetRef::Assumed(preimage.target.clone()),
                })
            })
            .transpose()?;
        self.atoms.insert(Atom {
            id: id.clone(),
            class: AtomClass::Assumed,
            kind,
            matrix_type: matrix_type.clone(),
            dependencies: BTreeSet::new(),
            preimage_refs: preimage_refs.clone(),
        });
        if let Some(preimage) = &declaration.preimage {
            let uniform_type = self
                .atoms
                .get(&preimage_refs.as_ref().expect("preimage refs exist").uniform)
                .expect("resolved uniform atom exists")
                .matrix_type
                .clone();
            let target_type = multiplication_type(&uniform_type, &matrix_type)?;
            let target_terms = self.resolve_assumed_terms(&preimage.target, bindings)?;
            if let Err(error) = self.check_term_list_type(entry, &target_terms, &target_type) {
                self.atoms.remove(&id);
                self.creating_virtual_atoms.remove(name);
                return Err(error);
            }
        }
        self.creating_virtual_atoms.remove(name);
        Ok(id)
    }

    fn resolve_reference_path(
        &self,
        entry: usize,
        path: &[crate::overlay::FrameMatcher],
        bindings: &BTreeMap<String, u64>,
    ) -> Result<Vec<InstantiationFrame>, ElaborationError> {
        path.iter()
            .map(|frame| {
                let loop_index = match &frame.loop_index {
                    LoopIndexMatcher::Concrete(index) => Some(*index),
                    LoopIndexMatcher::Var(name) => Some(*bindings.get(name).ok_or_else(|| {
                        self.overlay_error(entry, &format!("path variable {name} is unbound"))
                    })?),
                    LoopIndexMatcher::Any => {
                        return Err(
                            self.overlay_error(entry, "Any is ambiguous in an atom reference path")
                        );
                    }
                };
                Ok(InstantiationFrame { call: frame.call, loop_index })
            })
            .collect()
    }

    fn check_fold_partition(
        &self,
        entry: usize,
        positions: usize,
        groups: &[FoldGroup],
    ) -> Result<(), ElaborationError> {
        let mut seen = BTreeSet::new();
        for group in groups {
            for position in group.terms() {
                if *position >= positions {
                    return Err(self.overlay_error(entry, "fold group position is out of range"));
                }
                if !seen.insert(*position) {
                    return Err(self.overlay_error(entry, "fold groups overlap"));
                }
            }
        }
        if seen.len() != positions {
            return Err(self.overlay_error(entry, "fold groups do not partition expected positions"));
        }
        Ok(())
    }

    fn check_term_list_type(
        &self,
        entry: usize,
        terms: &TermList,
        expected: &ConcreteMatrixType,
    ) -> Result<(), ElaborationError> {
        for term in &terms.terms {
            if &self.term_type(entry, term)? != expected {
                return Err(self.overlay_error(entry, "assumed term has the wrong matrix type"));
            }
        }
        Ok(())
    }

    fn term_type(&self, entry: usize, term: &Term) -> Result<ConcreteMatrixType, ElaborationError> {
        let mut result: Option<ConcreteMatrixType> = None;
        for factor in &term.factors {
            let atom = self
                .atoms
                .get(&factor.atom)
                .ok_or_else(|| self.overlay_error(entry, "term references an absent atom"))?;
            let ty = viewed_type(atom.matrix_type.clone(), factor.view.as_ref());
            result = Some(match result {
                None => ty,
                Some(current) => multiplication_type(&current, &ty)?,
            });
        }
        result
            .ok_or_else(|| self.overlay_error(entry, "empty-factor overlay terms are unsupported"))
    }

    fn overlay_error(&self, entry: usize, message: &str) -> ElaborationError {
        ElaborationError::Overlay { entry: Some(entry), message: message.to_owned() }
    }

    fn overlay_error_without_entry(&self, message: &str) -> ElaborationError {
        ElaborationError::Overlay { entry: None, message: message.to_owned() }
    }

    fn argument(&self, node: &Node, index: usize) -> Result<&Value, ElaborationError> {
        let wire = *node.args.get(index).ok_or(ElaborationError::MissingWire {
            node: node.id,
            wire: WireRef { node: node.id, port: Port(index as u32) },
        })?;
        self.values.get(&wire).ok_or(ElaborationError::MissingWire { node: node.id, wire })
    }

    fn matrix_argument(&self, node: &Node, index: usize) -> Result<MatrixValue, ElaborationError> {
        match self.argument(node, index)? {
            Value::Matrix(value) | Value::Preimage(value) => Ok(value.clone()),
            _ => Err(ElaborationError::ExpectedMatrix(node.id)),
        }
    }

    fn trapdoor_argument(
        &self,
        node: &Node,
        index: usize,
    ) -> Result<TrapdoorValue, ElaborationError> {
        match self.argument(node, index)? {
            Value::Trapdoor(value) => Ok(value.clone()),
            _ => self.node_error(node.id, "expected a trapdoor argument"),
        }
    }

    fn insert_matrix(
        &mut self,
        node: NodeId,
        port: u32,
        ty: ConcreteMatrixType,
        terms: TermList,
        preimage: bool,
    ) {
        let wire = WireRef { node, port: Port(port) };
        let value = if preimage {
            Value::Preimage(MatrixValue { ty, terms: terms.clone() })
        } else {
            Value::Matrix(MatrixValue { ty, terms: terms.clone() })
        };
        let id = WireId { instantiation_path: self.path.clone(), wire };
        self.wire_terms.insert(id.clone(), terms);
        self.all_wires.insert(id, value.clone());
        self.values.insert(wire, value);
    }

    fn insert_value(&mut self, node: NodeId, port: u32, value: Value) {
        let wire = WireRef { node, port: Port(port) };
        let id = WireId { instantiation_path: self.path.clone(), wire };
        if let Value::Matrix(matrix) | Value::Preimage(matrix) = &value {
            self.wire_terms.insert(id.clone(), matrix.terms.clone());
        }
        self.all_wires.insert(id, value.clone());
        self.values.insert(wire, value);
    }

    fn source_atom(
        &mut self,
        node: NodeId,
        port: u32,
        matrix_type: ConcreteMatrixType,
        kind: AtomKind,
        preimage_refs: Option<PreimageRefs>,
    ) -> AtomId {
        let id = AtomId::Local { instantiation_path: self.path.clone(), node, port };
        self.atoms.insert(Atom {
            id: id.clone(),
            class: AtomClass::Source,
            kind,
            matrix_type,
            dependencies: BTreeSet::new(),
            preimage_refs,
        });
        id
    }

    fn derived_atom(
        &mut self,
        node: NodeId,
        port: u32,
        matrix_type: ConcreteMatrixType,
        definition: DefExpr,
        dependencies: BTreeSet<AtomId>,
        kind: AtomKind,
    ) -> AtomId {
        let id = AtomId::Local { instantiation_path: self.path.clone(), node, port };
        self.atoms.insert(Atom {
            id: id.clone(),
            class: AtomClass::Derived { definition },
            kind,
            matrix_type,
            dependencies,
            preimage_refs: None,
        });
        id
    }

    fn constant_atom(
        &mut self,
        value: &ConstantMatrix,
        ty: &ConcreteMatrixType,
        norm: UBound,
    ) -> Result<AtomId, ElaborationError> {
        let params = vec![
            serde_json::to_string(value).map_err(|error| ElaborationError::Node {
                node: NodeId(0),
                message: error.to_string(),
            })?,
            ty.modulus.to_string(),
            ty.ring_dimension.to_string(),
            ty.rows.to_string(),
            ty.columns.to_string(),
        ];
        let id = AtomId::Constant { kind: "matrix".to_owned(), params };
        if !self.atoms.contains_key(&id) {
            self.atoms.insert(Atom {
                id: id.clone(),
                class: AtomClass::Source,
                kind: AtomKind::Bounded { norm },
                matrix_type: ty.clone(),
                dependencies: BTreeSet::new(),
                preimage_refs: None,
            });
        }
        Ok(id)
    }

    fn gadget_atom(
        &mut self,
        ty: &ConcreteMatrixType,
        base: &crate::expr::IntExpr,
        small: bool,
    ) -> Result<AtomId, ElaborationError> {
        let value = ConstantMatrix::Gadget { base: base.clone(), small };
        self.constant_atom(&value, ty, modulus_half(&ty.modulus)?)
    }

    fn require_reduced(&self, node: NodeId, terms: &TermList) -> Result<(), ElaborationError> {
        if is_reduced(terms, &self.atoms)? {
            Ok(())
        } else {
            Err(ElaborationError::Node {
                node,
                message: CheckError::RequiresReducedInput.to_string(),
            })
        }
    }

    fn warn_viewed_preimages(&mut self, node: NodeId, terms: &TermList) {
        if terms.terms.iter().any(|term| {
            term.factors.iter().any(|factor| {
                self.atoms.get(&factor.atom).is_some_and(|atom| atom.preimage_refs.is_some())
            })
        }) {
            self.warnings.push(ElaborationWarning {
                node,
                kind: WarningKind::DroppedPreimageReferences,
                message: "a viewed or modulus-converted preimage occurrence does not retain its rewrite identity".to_owned(),
            });
        }
    }

    fn node_error<T>(&self, node: NodeId, message: &str) -> Result<T, ElaborationError> {
        Err(ElaborationError::Node { node, message: message.to_owned() })
    }
}

impl TargetTermLists for State<'_> {
    fn resolve(&self, preimage: &AtomId, target: &TargetRef) -> Option<&TermList> {
        match target {
            TargetRef::Local(wire) => {
                let instantiation_path = match preimage {
                    AtomId::Local { instantiation_path, .. } => instantiation_path.clone(),
                    _ => Vec::new(),
                };
                self.wire_terms.get(&WireId { instantiation_path, wire: *wire })
            }
            TargetRef::Imported { production_id, .. } => self
                .manifests
                .get(production_id)
                .and_then(|manifest| manifest.term_lists.get(target)),
            TargetRef::Assumed(_) => self.assumed_terms.get(target),
        }
    }
}

fn signal_suffix(term: &Term, suffix_len: usize, atoms: &AtomTable) -> Option<Vec<Factor>> {
    let non_scalar_positions = term
        .factors
        .iter()
        .enumerate()
        .filter_map(|(index, factor)| {
            atoms
                .get(&factor.atom)
                .is_some_and(|atom| !atom.matrix_type.is_scalar())
                .then_some(index)
        })
        .collect::<Vec<_>>();
    if suffix_len > non_scalar_positions.len() {
        return None;
    }
    let start = if suffix_len == 0 {
        term.factors.len()
    } else {
        non_scalar_positions[non_scalar_positions.len() - suffix_len]
    };
    Some(term.factors[start..].to_vec())
}

fn viewed_type(mut ty: ConcreteMatrixType, view: Option<&ViewDescriptor>) -> ConcreteMatrixType {
    let Some(view) = view else {
        return ty;
    };
    if view.transpose {
        std::mem::swap(&mut ty.rows, &mut ty.columns);
    }
    if let Some(rows) = &view.row_range {
        ty.rows = rows.end.saturating_sub(rows.start);
    }
    if let Some(columns) = &view.column_range {
        ty.columns = columns.end.saturating_sub(columns.start);
    }
    if let Some(modulus) = &view.modulus_cast {
        ty.modulus = modulus.clone();
    }
    ty
}

fn contextualize(
    node: NodeId,
    instantiation_path: &[InstantiationFrame],
    source: ElaborationError,
) -> ElaborationError {
    ElaborationError::Context {
        node,
        instantiation_path: instantiation_path.to_vec(),
        source: Box::new(source),
    }
}

fn contextualize_unless_present(
    node: NodeId,
    instantiation_path: &[InstantiationFrame],
    error: ElaborationError,
) -> ElaborationError {
    if matches!(&error, ElaborationError::Context { .. }) {
        error
    } else {
        contextualize(node, instantiation_path, error)
    }
}

fn contextualize_check(
    error: CheckError,
    instantiation_path: &[InstantiationFrame],
) -> ElaborationError {
    let node = match &error {
        CheckError::Core(mxx_ir_core::checks::CheckError::DuplicateNode(node)) |
        CheckError::Core(mxx_ir_core::checks::CheckError::NotTopological { node, .. }) |
        CheckError::Core(mxx_ir_core::checks::CheckError::InvalidOutput { node, .. }) => *node,
        _ => NodeId(0),
    };
    contextualize(node, instantiation_path, ElaborationError::Check(error))
}

fn check_bindings(graph: &Graph, env: &ParamEnv) -> Result<(), ElaborationError> {
    for parameter in &graph.parameters {
        let present = match parameter.kind {
            crate::graph::CompileParameterKind::Integer => {
                env.integers.contains_key(&parameter.name)
            }
            crate::graph::CompileParameterKind::Real => env.reals.contains_key(&parameter.name),
        };
        if !present {
            return Err(ElaborationError::MissingBinding(parameter.name.clone()));
        }
    }
    Ok(())
}

fn value_type(value: &Value) -> ConcreteWireType {
    match value {
        Value::Scalar(ty) => ty.clone(),
        Value::Matrix(matrix) => ConcreteWireType::Matrix(matrix.ty.clone()),
        Value::Preimage(matrix) => ConcreteWireType::Preimage(matrix.ty.clone()),
        Value::Trapdoor(trapdoor) => trapdoor.ty.clone(),
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

fn concrete_wire(ty: &WireType, env: &ParamEnv) -> Result<ConcreteWireType, ElaborationError> {
    Ok(match ty {
        WireType::ConstantInt => ConcreteWireType::ConstantInt,
        WireType::ConstantReal => ConcreteWireType::ConstantReal,
        WireType::ConstantBool => ConcreteWireType::ConstantBool,
        WireType::Int => ConcreteWireType::Int,
        WireType::Real => ConcreteWireType::Real,
        WireType::Bool => ConcreteWireType::Bool,
        WireType::Bytes { length } => ConcreteWireType::Bytes {
            length: positive_usize(length.evaluate(env)?, "byte-string length", NodeId(0))?,
        },
        WireType::Matrix(matrix) => ConcreteWireType::Matrix(concrete_matrix(matrix, env)?),
        WireType::Trapdoor { matrix, sigma } => ConcreteWireType::Trapdoor {
            matrix: concrete_matrix(matrix, env)?,
            sigma: sigma.clone(),
        },
        WireType::Preimage(matrix) => ConcreteWireType::Preimage(concrete_matrix(matrix, env)?),
    })
}

fn evaluate_bound(expression: &RealExpr, env: &ParamEnv) -> Result<UBound, ElaborationError> {
    Ok(match expression {
        RealExpr::Rational(value) => UBound::from_ratio(value.numerator(), value.denominator())?,
        RealExpr::Var(name) => {
            let value =
                env.reals.get(name).ok_or_else(|| ExprError::UnboundVariable(name.clone()))?;
            UBound::from_ratio(value.numerator(), value.denominator())?
        }
        RealExpr::FromInt(value) => UBound::from_integer(&value.evaluate(env)?)?,
        RealExpr::Add(lhs, rhs) => evaluate_bound(lhs, env)?.add(&evaluate_bound(rhs, env)?),
        RealExpr::Sub(lhs, rhs) => evaluate_bound(lhs, env)?.sub(&evaluate_bound(rhs, env)?)?,
        RealExpr::Mul(lhs, rhs) => evaluate_bound(lhs, env)?.mul(&evaluate_bound(rhs, env)?),
        RealExpr::Div(lhs, rhs) => evaluate_bound(lhs, env)?.div(&evaluate_bound(rhs, env)?)?,
        RealExpr::Sqrt(value) => evaluate_bound(value, env)?.sqrt(),
    })
}

fn concrete_matrix(
    ty: &MatrixType,
    env: &ParamEnv,
) -> Result<ConcreteMatrixType, ElaborationError> {
    let modulus = ty.modulus.evaluate(env)?;
    if modulus <= BigInt::one() {
        return Err(ElaborationError::Node {
            node: NodeId(0),
            message: "matrix modulus must be greater than one".to_owned(),
        });
    }
    Ok(ConcreteMatrixType {
        modulus,
        ring_dimension: positive_usize(
            ty.ring_dimension.evaluate(env)?,
            "ring dimension",
            NodeId(0),
        )?,
        rows: positive_usize(ty.rows.evaluate(env)?, "matrix rows", NodeId(0))?,
        columns: positive_usize(ty.columns.evaluate(env)?, "matrix columns", NodeId(0))?,
    })
}

fn positive_usize(value: BigInt, label: &str, node: NodeId) -> Result<usize, ElaborationError> {
    value.to_usize().filter(|value| *value > 0).ok_or_else(|| ElaborationError::Node {
        node,
        message: format!("{label} must be a positive usize"),
    })
}

fn sliced_type(
    input: &ConcreteMatrixType,
    rows: Option<&crate::term::IndexRange>,
    columns: Option<&crate::term::IndexRange>,
) -> Result<ConcreteMatrixType, ElaborationError> {
    let row_count = rows.map_or(input.rows, |range| range.end.saturating_sub(range.start));
    let column_count = columns.map_or(input.columns, |range| range.end.saturating_sub(range.start));
    if rows.is_some_and(|range| range.start >= range.end || range.end > input.rows) ||
        columns.is_some_and(|range| range.start >= range.end || range.end > input.columns)
    {
        return Err(ElaborationError::Node {
            node: NodeId(0),
            message: "slice range is invalid".to_owned(),
        });
    }
    Ok(ConcreteMatrixType { rows: row_count, columns: column_count, ..input.clone() })
}

fn constant_norm(
    value: &ConstantMatrix,
    ty: &ConcreteMatrixType,
    env: &ParamEnv,
) -> Result<UBound, ElaborationError> {
    Ok(match value {
        ConstantMatrix::Zero => UBound::zero(),
        ConstantMatrix::Identity |
        ConstantMatrix::UnitRow { .. } |
        ConstantMatrix::UnitColumn { .. } |
        ConstantMatrix::Rotation { .. } => UBound::one(),
        ConstantMatrix::Gadget { .. } => modulus_half(&ty.modulus)?,
        ConstantMatrix::PowerOfBase { base, exponent } => {
            let base = base.evaluate(env)?.abs();
            let exponent =
                exponent.evaluate(env)?.to_u32().ok_or_else(|| ElaborationError::Node {
                    node: NodeId(0),
                    message: "power exponent must be a u32".to_owned(),
                })?;
            UBound::from_integer(&base.pow(exponent))?
        }
    })
}

fn gaussian_norm(sigma: UBound, ty: &ConcreteMatrixType) -> UBound {
    sigma.mul(&UBound::from_u64(ty.ring_dimension as u64).sqrt()).mul(&UBound::from_u64(8))
}

fn gadget_digits(modulus: &BigInt, base: &BigInt) -> usize {
    let mut power = BigInt::one();
    let mut digits = 0usize;
    while power < *modulus {
        power *= base;
        digits = digits.saturating_add(1);
    }
    digits.max(1)
}

fn decomposition_digits(
    explicit: Option<&crate::expr::IntExpr>,
    modulus: &BigInt,
    base: &BigInt,
    env: &ParamEnv,
    node: NodeId,
) -> Result<usize, ElaborationError> {
    match explicit {
        Some(value) => positive_usize(value.evaluate(env)?, "decomposition digit count", node),
        None => Ok(gadget_digits(modulus, base)),
    }
}

fn modulus_half(modulus: &BigInt) -> Result<UBound, crate::ubound::UBoundError> {
    UBound::from_ratio(modulus, &BigInt::from(2))
}

fn dependencies(terms: &TermList, atoms: &AtomTable) -> BTreeSet<AtomId> {
    terms
        .terms
        .iter()
        .flat_map(|term| term.factors.iter().map(|factor| factor.atom.clone()))
        .flat_map(|id| {
            let mut dependencies =
                atoms.get(&id).map_or_else(BTreeSet::new, |atom| atom.dependencies.clone());
            dependencies.insert(id);
            dependencies
        })
        .collect()
}

fn derived_kind(
    terms: &TermList,
    atoms: &AtomTable,
    gamma: &UBound,
) -> Result<AtomKind, ElaborationError> {
    if terms.contains_large(atoms)? {
        Ok(AtomKind::Large)
    } else {
        Ok(AtomKind::Bounded { norm: sum_norm(terms, atoms, gamma)? })
    }
}

fn int_norm_sum(
    terms: &TermList,
    atoms: &AtomTable,
    gamma: &UBound,
    modulus: &BigInt,
) -> Result<UBound, ElaborationError> {
    let half = modulus_half(modulus)?;
    let mut shadow = atoms.clone();
    for atom in shadow.values().cloned().collect::<Vec<_>>() {
        let capped = match atom.kind {
            AtomKind::Large => half.clone(),
            AtomKind::Bounded { norm } => UBound::min(&norm, &half),
        };
        shadow.get_mut(&atom.id).expect("atom exists").kind = AtomKind::Bounded { norm: capped };
    }
    let mut total = UBound::zero();
    for term in &terms.terms {
        total = total.add(&term_norm(term, &shadow, gamma)?);
    }
    Ok(total)
}

fn ring_expansion(graph: &Graph, env: &ParamEnv) -> Result<UBound, ElaborationError> {
    let maximum = graph
        .nodes
        .iter()
        .filter_map(|node| match &node.kind {
            NodeKind::Input {
                wire_type:
                    WireType::Matrix(matrix) |
                    WireType::Preimage(matrix) |
                    WireType::Trapdoor { matrix, .. },
                ..
            } => Some(&matrix.ring_dimension),
            NodeKind::ConstantMatrix { matrix_type, .. } |
            NodeKind::UniformSample { matrix_type, .. } |
            NodeKind::GaussianSample { matrix_type, .. } |
            NodeKind::HashSample { matrix_type, .. } |
            NodeKind::TrapdoorSample { matrix_type, .. } |
            NodeKind::PreimageSample { matrix_type } => Some(&matrix_type.ring_dimension),
            NodeKind::GadgetTrapdoor { matrix_type, .. } => Some(&matrix_type.ring_dimension),
            _ => None,
        })
        .map(|dimension| dimension.evaluate(env))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or_else(BigInt::one);
    Ok(UBound::from_integer(&maximum)?.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        expr::{IntExpr, Rational, RealExpr},
        graph::Graph,
        manifest::{ExportArtifact, export_manifest, import_manifest, production_id},
        node::{ArtifactInput, MatrixBinaryOp, Node, NodeKind, SampleRange},
        overlay::{
            AssumedPreimage, AssumedTermList, AtomRef, ExpectedEntry, ExpectedTermList, FactorRef,
            FoldGroup, FoldSpec, OverlayTerm, PortMatcher, Reinterpretation, SelectNodeSelector,
            UnfoldSpec, VirtualAtomDecl, VirtualKind, WireSelector,
        },
    };

    fn wire(node: u64, port: u32) -> WireRef {
        WireRef { node: NodeId(node), port: Port(port) }
    }

    fn root_cause(error: &ElaborationError) -> &ElaborationError {
        match error {
            ElaborationError::Context { source, .. } => root_cause(source),
            _ => error,
        }
    }

    fn matrix_type() -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(2),
            columns: IntExpr::constant(2),
        }
    }

    fn bounded_sample(id: u64) -> Node {
        Node {
            id: NodeId(id),
            kind: NodeKind::UniformSample {
                matrix_type: matrix_type(),
                range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
            },
            args: Vec::new(),
        }
    }

    fn empty_graph(name: &str, nodes: Vec<Node>, output: WireRef) -> Graph {
        Graph {
            name: name.to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes,
            outputs: BTreeMap::from([("out".to_owned(), output)]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        }
    }

    fn elaborated_terms(graph: &ElaboratedGraph, wire: WireRef) -> &TermList {
        graph
            .wires
            .get(&WireId { instantiation_path: Vec::new(), wire })
            .and_then(|wire| wire.terms.as_ref())
            .expect("matrix wire")
    }

    #[test]
    fn transpose_twice_restores_view_free_identity() {
        let graph = empty_graph(
            "transpose",
            vec![
                bounded_sample(1),
                Node { id: NodeId(2), kind: NodeKind::Transpose, args: vec![wire(1, 0)] },
                Node { id: NodeId(3), kind: NodeKind::Transpose, args: vec![wire(2, 0)] },
            ],
            wire(3, 0),
        );
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        assert_eq!(
            elaborated_terms(&elaborated, wire(1, 0)),
            elaborated_terms(&elaborated, wire(3, 0))
        );
    }

    #[test]
    fn tensor_and_concat_retain_exact_defining_expressions_and_norm_rules() {
        let mut left = bounded_sample(1);
        let mut right = bounded_sample(2);
        for node in [&mut left, &mut right] {
            let NodeKind::UniformSample { range, .. } = &mut node.kind else { unreachable!() };
            range.minimum = BigInt::from(-2);
            range.maximum = BigInt::from(2);
        }
        let tensor =
            Node { id: NodeId(3), kind: NodeKind::Tensor, args: vec![wire(1, 0), wire(2, 0)] };
        let concat = Node {
            id: NodeId(4),
            kind: NodeKind::Concat { axis: ConcatAxis::Rows },
            args: vec![wire(1, 0), wire(2, 0)],
        };
        let reshape = Node {
            id: NodeId(5),
            kind: NodeKind::Reshape { rows: IntExpr::constant(1), columns: IntExpr::constant(8) },
            args: vec![wire(4, 0)],
        };
        let graph = Graph {
            name: "derived-definitions".to_owned(),
            parameters: Vec::new(),
            input_types: BTreeMap::new(),
            nodes: vec![left, right, tensor, concat, reshape],
            outputs: BTreeMap::from([
                ("tensor".to_owned(), wire(3, 0)),
                ("concat".to_owned(), wire(4, 0)),
                ("reshape".to_owned(), wire(5, 0)),
            ]),
            subgraphs: BTreeMap::new(),
            real_constants: BTreeMap::new(),
        };
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let tensor_atom = &elaborated_terms(&elaborated, wire(3, 0)).terms[0].factors[0].atom;
        let concat_atom = &elaborated_terms(&elaborated, wire(4, 0)).terms[0].factors[0].atom;
        let reshape_atom = &elaborated_terms(&elaborated, wire(5, 0)).terms[0].factors[0].atom;
        let tensor = elaborated.atoms.get(tensor_atom).expect("tensor atom");
        let concat = elaborated.atoms.get(concat_atom).expect("concat atom");
        let reshape = elaborated.atoms.get(reshape_atom).expect("reshape atom");
        assert!(matches!(tensor.class, AtomClass::Derived { definition: DefExpr::Tensor { .. } }));
        assert!(matches!(
            concat.class,
            AtomClass::Derived { definition: DefExpr::Concat { axis: ConcatAxis::Rows, .. } }
        ));
        assert!(matches!(
            reshape.class,
            AtomClass::Derived { definition: DefExpr::Reshape { rows: 1, columns: 8, .. } }
        ));
        let AtomKind::Bounded { norm: tensor_norm } = &tensor.kind else {
            panic!("bounded tensor")
        };
        let AtomKind::Bounded { norm: concat_norm } = &concat.kind else {
            panic!("bounded concat")
        };
        assert_eq!(tensor_norm, &elaborated.ring_expansion.mul(&UBound::from_u64(4)));
        assert_eq!(concat_norm, &UBound::from_u64(2));
        assert_eq!(reshape.kind, concat.kind);
    }

    #[test]
    fn correlated_selects_eliminate_cross_terms_and_rewrite_diagonal() {
        let sigma =
            RealExpr::Rational(Rational::new(BigInt::from(3), BigInt::from(1)).expect("rational"));
        let mut nodes = vec![Node {
            id: NodeId(1),
            kind: NodeKind::ConstantInt(BigInt::from(0)),
            args: Vec::new(),
        }];
        for id in [2, 3] {
            nodes.push(Node {
                id: NodeId(id),
                kind: NodeKind::TrapdoorSample { matrix_type: matrix_type(), sigma: sigma.clone() },
                args: Vec::new(),
            });
        }
        nodes.extend([bounded_sample(4), bounded_sample(5)]);
        for (id, trapdoor, target) in [(6, 2, 4), (7, 3, 5)] {
            nodes.push(Node {
                id: NodeId(id),
                kind: NodeKind::PreimageSample { matrix_type: matrix_type() },
                args: vec![wire(trapdoor, 1), wire(target, 0)],
            });
        }
        nodes.push(Node {
            id: NodeId(8),
            kind: NodeKind::Select { count: IntExpr::constant(2) },
            args: vec![wire(1, 0), wire(2, 0), wire(3, 0)],
        });
        nodes.push(Node {
            id: NodeId(9),
            kind: NodeKind::Select { count: IntExpr::constant(2) },
            args: vec![wire(1, 0), wire(6, 0), wire(7, 0)],
        });
        nodes.push(Node {
            id: NodeId(10),
            kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            args: vec![wire(8, 0), wire(9, 0)],
        });
        let graph = empty_graph("correlated", nodes, wire(10, 0));
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let output = elaborated_terms(&elaborated, wire(10, 0));
        assert_eq!(output.terms.len(), 2);
        for term in &output.terms {
            assert_eq!(term.factors.len(), 2);
            assert!(matches!(term.factors[0].atom, AtomId::Indicator { .. }));
            assert!(matches!(term.factors[1].atom, AtomId::Local { node: NodeId(4 | 5), .. }));
        }
    }

    fn large_term_count(elaborated: &ElaboratedGraph, output: WireRef) -> usize {
        elaborated_terms(elaborated, output)
            .terms
            .iter()
            .filter(|term| {
                term.factors.iter().any(|factor| {
                    elaborated
                        .atoms
                        .get(&factor.atom)
                        .is_some_and(|atom| matches!(atom.kind, AtomKind::Large))
                })
            })
            .count()
    }

    #[test]
    fn correlated_select_with_mismatched_preimage_branch_leaves_large_term() {
        let sigma =
            RealExpr::Rational(Rational::new(BigInt::from(3), BigInt::one()).expect("rational"));
        let mut nodes = vec![Node {
            id: NodeId(1),
            kind: NodeKind::ConstantInt(BigInt::from(0)),
            args: Vec::new(),
        }];
        for id in [2, 3] {
            nodes.push(Node {
                id: NodeId(id),
                kind: NodeKind::TrapdoorSample { matrix_type: matrix_type(), sigma: sigma.clone() },
                args: Vec::new(),
            });
        }
        nodes.extend([bounded_sample(4), bounded_sample(5)]);
        nodes.push(Node {
            id: NodeId(6),
            kind: NodeKind::PreimageSample { matrix_type: matrix_type() },
            args: vec![wire(2, 1), wire(4, 0)],
        });
        // This branch deliberately references A_0 while it is paired with A_1.
        nodes.push(Node {
            id: NodeId(7),
            kind: NodeKind::PreimageSample { matrix_type: matrix_type() },
            args: vec![wire(2, 1), wire(5, 0)],
        });
        nodes.push(Node {
            id: NodeId(8),
            kind: NodeKind::Select { count: IntExpr::constant(2) },
            args: vec![wire(1, 0), wire(2, 0), wire(3, 0)],
        });
        nodes.push(Node {
            id: NodeId(9),
            kind: NodeKind::Select { count: IntExpr::constant(2) },
            args: vec![wire(1, 0), wire(6, 0), wire(7, 0)],
        });
        nodes.push(Node {
            id: NodeId(10),
            kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            args: vec![wire(8, 0), wire(9, 0)],
        });
        let graph = empty_graph("mismatched-selected-preimage", nodes, wire(10, 0));
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        assert_eq!(large_term_count(&elaborated, wire(10, 0)), 1);
    }

    #[test]
    fn independently_selected_uniform_and_preimage_leave_cross_branch_large_terms() {
        let sigma =
            RealExpr::Rational(Rational::new(BigInt::from(3), BigInt::one()).expect("rational"));
        let mut nodes = vec![
            Node { id: NodeId(1), kind: NodeKind::ConstantInt(BigInt::from(0)), args: Vec::new() },
            Node { id: NodeId(2), kind: NodeKind::ConstantInt(BigInt::from(1)), args: Vec::new() },
        ];
        for id in [3, 4] {
            nodes.push(Node {
                id: NodeId(id),
                kind: NodeKind::TrapdoorSample { matrix_type: matrix_type(), sigma: sigma.clone() },
                args: Vec::new(),
            });
        }
        nodes.extend([bounded_sample(5), bounded_sample(6)]);
        for (id, trapdoor, target) in [(7, 3, 5), (8, 4, 6)] {
            nodes.push(Node {
                id: NodeId(id),
                kind: NodeKind::PreimageSample { matrix_type: matrix_type() },
                args: vec![wire(trapdoor, 1), wire(target, 0)],
            });
        }
        nodes.push(Node {
            id: NodeId(9),
            kind: NodeKind::Select { count: IntExpr::constant(2) },
            args: vec![wire(1, 0), wire(3, 0), wire(4, 0)],
        });
        nodes.push(Node {
            id: NodeId(10),
            kind: NodeKind::Select { count: IntExpr::constant(2) },
            args: vec![wire(2, 0), wire(7, 0), wire(8, 0)],
        });
        nodes.push(Node {
            id: NodeId(11),
            kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            args: vec![wire(9, 0), wire(10, 0)],
        });
        let graph = empty_graph("independent-selected-preimages", nodes, wire(11, 0));
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        assert_eq!(large_term_count(&elaborated, wire(11, 0)), 2);
    }

    #[test]
    fn same_index_selects_over_different_rings_use_distinct_indicator_families() {
        let mut other_ring = matrix_type();
        other_ring.modulus = IntExpr::constant(19);
        let mut graph = empty_graph(
            "ring-scoped-selects",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantInt(BigInt::from(0)),
                    args: Vec::new(),
                },
                bounded_sample(2),
                bounded_sample(3),
                Node {
                    id: NodeId(4),
                    kind: NodeKind::UniformSample {
                        matrix_type: other_ring.clone(),
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(5),
                    kind: NodeKind::UniformSample {
                        matrix_type: other_ring,
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(6),
                    kind: NodeKind::Select { count: IntExpr::constant(2) },
                    args: vec![wire(1, 0), wire(2, 0), wire(3, 0)],
                },
                Node {
                    id: NodeId(7),
                    kind: NodeKind::Select { count: IntExpr::constant(2) },
                    args: vec![wire(1, 0), wire(4, 0), wire(5, 0)],
                },
            ],
            wire(6, 0),
        );
        graph.outputs.insert("other".to_owned(), wire(7, 0));
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let first = &elaborated_terms(&elaborated, wire(6, 0)).terms[0].factors[0].atom;
        let second = &elaborated_terms(&elaborated, wire(7, 0)).terms[0].factors[0].atom;
        assert_ne!(first, second);
        assert!(matches!(
            (first, second),
            (
                AtomId::Indicator {
                    domain: first_domain,
                    ..
                },
                AtomId::Indicator {
                    domain: second_domain,
                    ..
                }
            ) if first_domain.index_wire == second_domain.index_wire
                && first_domain.modulus != second_domain.modulus
        ));
    }

    #[test]
    fn repeated_subgraph_calls_stamp_distinct_atom_paths() {
        let body = empty_graph("body", vec![bounded_sample(1)], wire(1, 0));
        let mut graph = empty_graph(
            "parent",
            vec![
                Node {
                    id: NodeId(10),
                    kind: NodeKind::SubgraphCall(crate::node::SubgraphCall {
                        graph: "body".to_owned(),
                        bindings: Vec::new(),
                    }),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(11),
                    kind: NodeKind::SubgraphCall(crate::node::SubgraphCall {
                        graph: "body".to_owned(),
                        bindings: Vec::new(),
                    }),
                    args: Vec::new(),
                },
            ],
            wire(11, 0),
        );
        graph.subgraphs.insert("body".to_owned(), Box::new(body));
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let first = elaborated_terms(&elaborated, wire(10, 0)).terms[0].factors[0].atom.clone();
        let second = elaborated_terms(&elaborated, wire(11, 0)).terms[0].factors[0].atom.clone();
        assert_ne!(first, second);
        assert!(matches!(
            first,
            AtomId::Local {
                instantiation_path,
                ..
            } if instantiation_path == vec![InstantiationFrame {
                call: NodeId(10),
                loop_index: None,
            }]
        ));
    }

    #[test]
    fn child_failure_reports_the_child_node_and_instantiation_path() {
        let mut narrow = matrix_type();
        narrow.columns = IntExpr::constant(1);
        let child = empty_graph(
            "invalid-child",
            vec![
                bounded_sample(1),
                Node {
                    id: NodeId(2),
                    kind: NodeKind::UniformSample {
                        matrix_type: narrow,
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
            ],
            wire(3, 0),
        );
        let mut graph = empty_graph(
            "root",
            vec![Node {
                id: NodeId(10),
                kind: NodeKind::SubgraphCall(crate::node::SubgraphCall {
                    graph: "invalid-child".to_owned(),
                    bindings: Vec::new(),
                }),
                args: Vec::new(),
            }],
            wire(10, 0),
        );
        graph.subgraphs.insert("invalid-child".to_owned(), Box::new(child));

        let error = elaborate(&graph, &ParamEnv::default()).expect_err("child shape mismatch");
        let ElaborationError::Context { node, instantiation_path, source } = &error else {
            panic!("contextual error: {error}")
        };
        assert_eq!(*node, NodeId(3));
        assert_eq!(
            instantiation_path,
            &[InstantiationFrame { call: NodeId(10), loop_index: None }]
        );
        assert!(matches!(
            root_cause(source),
            ElaborationError::Check(CheckError::Core(
                mxx_ir_core::checks::CheckError::ShapeMismatch { .. }
            ))
        ));
    }

    #[test]
    fn mod_up_caps_a_large_bounded_preimage_and_drops_its_references() {
        let target_type = MatrixType {
            rows: IntExpr::constant(2),
            columns: IntExpr::constant(1),
            ..matrix_type()
        };
        let graph = empty_graph(
            "mod-up-cap",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::UniformSample {
                        matrix_type: target_type,
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::GadgetDecompose {
                        base: IntExpr::constant(20),
                        small: false,
                        digit_count: None,
                    },
                    args: vec![wire(1, 0)],
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::ModUp { target_modulus: IntExpr::constant(257) },
                    args: vec![wire(2, 0)],
                },
            ],
            wire(3, 0),
        );
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        assert!(elaborated.warnings.iter().any(|warning| {
            warning.node == NodeId(3) && warning.kind == WarningKind::DroppedPreimageReferences
        }));

        let output = elaborated_terms(&elaborated, wire(3, 0));
        let cast_term = output
            .terms
            .iter()
            .find(|term| {
                term.coefficient == BigInt::one() &&
                    term.factors.iter().any(|factor| factor.view.is_some())
            })
            .expect("cast preimage term");
        let cast_factor = &cast_term.factors[0];
        assert_eq!(
            cast_factor.view.as_ref().and_then(|view| view.modulus_cast.as_ref()),
            Some(&BigInt::from(257))
        );
        let source = elaborated.atoms.get(&cast_factor.atom).expect("source atom");
        assert!(source.preimage_refs.is_some());
        assert_eq!(
            crate::bounds::term_norm(cast_term, &elaborated.atoms, &elaborated.ring_expansion)
                .expect("bounded cast"),
            UBound::from_ratio(&BigInt::from(17), &BigInt::from(2)).expect("valid cap")
        );
    }

    #[test]
    fn mod_down_casts_indicator_and_bounded_prefix_into_target_ring() {
        let scalar_type = MatrixType {
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
            ..matrix_type()
        };
        let graph = empty_graph(
            "mod-down-indicator-prefix",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantInt(BigInt::from(0)),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::UniformSample {
                        matrix_type: scalar_type.clone(),
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::UniformSample {
                        matrix_type: scalar_type,
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(4),
                    kind: NodeKind::Select { count: IntExpr::constant(2) },
                    args: vec![wire(1, 0), wire(2, 0), wire(3, 0)],
                },
                Node {
                    id: NodeId(5),
                    kind: NodeKind::TrapdoorSample {
                        matrix_type: matrix_type(),
                        sigma: RealExpr::Rational(
                            Rational::new(BigInt::from(3), BigInt::one()).expect("rational"),
                        ),
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(6),
                    kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                    args: vec![wire(4, 0), wire(5, 0)],
                },
                Node {
                    id: NodeId(7),
                    kind: NodeKind::ModDown { target_modulus: IntExpr::constant(7) },
                    args: vec![wire(6, 0)],
                },
            ],
            wire(7, 0),
        );
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("elaboration");
        let output = elaborated_terms(&elaborated, wire(7, 0));
        assert!(output.terms.iter().all(|term| !term.coefficient.is_zero()));
        let signal_terms = output
            .terms
            .iter()
            .filter(|term| {
                term.factors.iter().any(|factor| {
                    elaborated
                        .atoms
                        .get(&factor.atom)
                        .is_some_and(|atom| matches!(atom.kind, AtomKind::Large))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(signal_terms.len(), 2);
        for term in signal_terms {
            assert_eq!(term.factors.len(), 3);
            assert!(
                term.factors[..2]
                    .iter()
                    .any(|factor| matches!(factor.atom, AtomId::Indicator { .. }))
            );
            for factor in &term.factors[..2] {
                assert_eq!(
                    factor.view.as_ref().and_then(|view| view.modulus_cast.as_ref()),
                    Some(&BigInt::from(7))
                );
            }
            let tail = elaborated.atoms.get(&term.factors[2].atom).expect("mod-down image");
            assert_eq!(tail.matrix_type.modulus, BigInt::from(7));
            assert!(matches!(
                tail.class,
                AtomClass::Derived { definition: DefExpr::ModDownImage { .. } }
            ));
        }
        let error = output
            .terms
            .iter()
            .find_map(|term| {
                (term.factors.len() == 1)
                    .then(|| elaborated.atoms.get(&term.factors[0].atom))
                    .flatten()
            })
            .expect("mod-down error");
        assert!(matches!(
            error.class,
            AtomClass::Derived { definition: DefExpr::ModDownError { .. } }
        ));
        assert!(matches!(error.kind, AtomKind::Bounded { .. }));
    }

    #[test]
    fn mod_down_rejects_large_factor_before_bounded_matrix_tail() {
        let graph = empty_graph(
            "invalid-mod-down-tail",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::TrapdoorSample {
                        matrix_type: matrix_type(),
                        sigma: RealExpr::Rational(
                            Rational::new(BigInt::from(3), BigInt::one()).expect("rational"),
                        ),
                    },
                    args: Vec::new(),
                },
                bounded_sample(2),
                Node {
                    id: NodeId(3),
                    kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                Node {
                    id: NodeId(4),
                    kind: NodeKind::ModDown { target_modulus: IntExpr::constant(7) },
                    args: vec![wire(3, 0)],
                },
            ],
            wire(4, 0),
        );
        let error = elaborate(&graph, &ParamEnv::default()).expect_err("invalid normal form");
        assert!(matches!(
            root_cause(&error),
            ElaborationError::Check(CheckError::InvalidModDownNormalForm)
        ));
    }

    #[test]
    fn select_rejects_mismatched_family_shapes() {
        let mut narrow = matrix_type();
        narrow.columns = IntExpr::constant(1);
        let graph = empty_graph(
            "invalid-select",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantInt(BigInt::from(0)),
                    args: Vec::new(),
                },
                bounded_sample(2),
                Node {
                    id: NodeId(3),
                    kind: NodeKind::UniformSample {
                        matrix_type: narrow,
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(4),
                    kind: NodeKind::Select { count: IntExpr::constant(2) },
                    args: vec![wire(1, 0), wire(2, 0), wire(3, 0)],
                },
            ],
            wire(4, 0),
        );
        let error = elaborate(&graph, &ParamEnv::default()).expect_err("shape mismatch");
        assert!(matches!(
            root_cause(&error),
            ElaborationError::Check(CheckError::Core(
                mxx_ir_core::checks::CheckError::ShapeMismatch { .. }
            ))
        ));
    }

    #[test]
    fn select_marks_only_indices_that_are_not_statically_proven_in_range() {
        let graph_for_index = |name: &str, index: Node| {
            empty_graph(
                name,
                vec![
                    index,
                    bounded_sample(2),
                    bounded_sample(3),
                    Node {
                        id: NodeId(4),
                        kind: NodeKind::Select { count: IntExpr::constant(2) },
                        args: vec![wire(1, 0), wire(2, 0), wire(3, 0)],
                    },
                ],
                wire(4, 0),
            )
        };
        let in_range = graph_for_index(
            "constant-in-range-select",
            Node { id: NodeId(1), kind: NodeKind::ConstantInt(BigInt::one()), args: Vec::new() },
        );
        let elaborated = elaborate(&in_range, &ParamEnv::default()).expect("in-range select");
        assert!(elaborated.warnings.is_empty());

        let out_of_range = graph_for_index(
            "constant-out-of-range-select",
            Node { id: NodeId(1), kind: NodeKind::ConstantInt(BigInt::from(2)), args: Vec::new() },
        );
        let elaborated =
            elaborate(&out_of_range, &ParamEnv::default()).expect("runtime-checked select");
        assert!(elaborated.warnings.iter().any(|warning| {
            warning.node == NodeId(4) && warning.kind == WarningKind::RuntimeSelectBoundsCheck
        }));
    }

    #[test]
    fn select_range_proof_understands_boolean_and_euclidean_remainder_indices() {
        let mut remainder_graph = empty_graph(
            "remainder-select",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "index".to_owned(),
                        wire_type: WireType::Int,
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(2),
                    kind: NodeKind::ConstantInt(BigInt::from(2)),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::IntBinary(crate::node::IntBinaryOp::Remainder),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
                bounded_sample(4),
                bounded_sample(5),
                Node {
                    id: NodeId(6),
                    kind: NodeKind::Select { count: IntExpr::constant(2) },
                    args: vec![wire(3, 0), wire(4, 0), wire(5, 0)],
                },
            ],
            wire(6, 0),
        );
        remainder_graph.input_types.insert("index".to_owned(), WireType::Int);
        let elaborated =
            elaborate(&remainder_graph, &ParamEnv::default()).expect("remainder select");
        assert!(elaborated.warnings.is_empty());

        let bool_graph = empty_graph(
            "boolean-select",
            vec![
                Node { id: NodeId(1), kind: NodeKind::ConstantBool(true), args: Vec::new() },
                Node { id: NodeId(2), kind: NodeKind::BoolToInt, args: vec![wire(1, 0)] },
                bounded_sample(3),
                bounded_sample(4),
                Node {
                    id: NodeId(5),
                    kind: NodeKind::Select { count: IntExpr::constant(2) },
                    args: vec![wire(2, 0), wire(3, 0), wire(4, 0)],
                },
            ],
            wire(5, 0),
        );
        let elaborated = elaborate(&bool_graph, &ParamEnv::default()).expect("boolean select");
        assert!(elaborated.warnings.is_empty());
    }

    #[test]
    fn child_select_bounds_warning_is_propagated() {
        let mut child = empty_graph(
            "dynamic-select-child",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::Input {
                        name: "index".to_owned(),
                        wire_type: WireType::Int,
                        artifact: None,
                    },
                    args: Vec::new(),
                },
                bounded_sample(2),
                bounded_sample(3),
                Node {
                    id: NodeId(4),
                    kind: NodeKind::Select { count: IntExpr::constant(2) },
                    args: vec![wire(1, 0), wire(2, 0), wire(3, 0)],
                },
            ],
            wire(4, 0),
        );
        child.input_types.insert("index".to_owned(), WireType::Int);
        let mut graph = empty_graph(
            "select-parent",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantInt(BigInt::zero()),
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(10),
                    kind: NodeKind::SubgraphCall(crate::node::SubgraphCall {
                        graph: "dynamic-select-child".to_owned(),
                        bindings: Vec::new(),
                    }),
                    args: vec![wire(1, 0)],
                },
            ],
            wire(10, 0),
        );
        graph.subgraphs.insert("dynamic-select-child".to_owned(), Box::new(child));
        let elaborated = elaborate(&graph, &ParamEnv::default()).expect("child select");
        assert!(elaborated.warnings.iter().any(|warning| {
            warning.node == NodeId(4) && warning.kind == WarningKind::RuntimeSelectBoundsCheck
        }));
    }

    #[test]
    fn graph_json_round_trip_preserves_elaboration() {
        let graph = empty_graph(
            "json-round-trip",
            vec![
                bounded_sample(1),
                bounded_sample(2),
                Node {
                    id: NodeId(3),
                    kind: NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
            ],
            wire(3, 0),
        );
        let json = serde_json::to_vec(&graph).expect("serialize graph");
        let decoded: Graph = serde_json::from_slice(&json).expect("deserialize graph");
        let original = elaborate(&graph, &ParamEnv::default()).expect("original");
        let round_trip = elaborate(&decoded, &ParamEnv::default()).expect("round trip");
        assert_eq!(original, round_trip);
    }

    #[test]
    fn graph_output_rejects_an_unavailable_port() {
        let graph = empty_graph("bad-output-port", vec![bounded_sample(1)], wire(1, 1));
        let error = elaborate(&graph, &ParamEnv::default()).expect_err("invalid output port");
        assert!(matches!(root_cause(&error), ElaborationError::Node { node: NodeId(1), .. }));
    }

    #[test]
    fn output_family_requires_identical_member_types() {
        let mut scalar_type = matrix_type();
        scalar_type.rows = IntExpr::constant(1);
        scalar_type.columns = IntExpr::constant(1);
        let graph = empty_graph(
            "bad-output-family",
            vec![
                bounded_sample(1),
                Node {
                    id: NodeId(2),
                    kind: NodeKind::UniformSample {
                        matrix_type: scalar_type,
                        range: SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
                    },
                    args: Vec::new(),
                },
                Node {
                    id: NodeId(3),
                    kind: NodeKind::Output { name: "family".to_owned() },
                    args: vec![wire(1, 0), wire(2, 0)],
                },
            ],
            wire(3, 0),
        );
        let error = elaborate(&graph, &ParamEnv::default()).expect_err("output type mismatch");
        assert!(matches!(root_cause(&error), ElaborationError::Node { node: NodeId(3), .. }));
    }

    fn matrix_input(id: u64, name: &str, preimage: bool) -> Node {
        let matrix = matrix_type();
        Node {
            id: NodeId(id),
            kind: NodeKind::Input {
                name: name.to_owned(),
                wire_type: if preimage {
                    WireType::Preimage(matrix)
                } else {
                    WireType::Matrix(matrix)
                },
                artifact: None,
            },
            args: Vec::new(),
        }
    }

    fn virtual_decl(matrix_type: MatrixType, kind: VirtualKind) -> VirtualAtomDecl {
        VirtualAtomDecl { matrix_type, kind, preimage: None }
    }

    fn virtual_factor(name: &str) -> FactorRef {
        FactorRef { atom: AtomRef::Virtual { name: name.to_owned() }, view: None }
    }

    fn overlay_term(coefficient: i64, factors: Vec<FactorRef>) -> OverlayTerm {
        OverlayTerm { coefficient: IntExpr::constant(coefficient), factors }
    }

    fn assumed_id(name: &str) -> AssumedTermListId {
        AssumedTermListId(name.to_owned())
    }

    fn selector(node: u64) -> WireSelector {
        WireSelector { path: Vec::new(), node: NodeId(node), port: 0 }
    }

    fn unfold_overlay() -> SymbolicOverlay {
        let mut scalar = matrix_type();
        scalar.rows = IntExpr::constant(1);
        scalar.columns = IntExpr::constant(1);
        SymbolicOverlay {
            virtual_atoms: BTreeMap::from([
                ("B".to_owned(), virtual_decl(matrix_type(), VirtualKind::Large)),
                (
                    "s".to_owned(),
                    virtual_decl(
                        scalar,
                        VirtualKind::Bounded { norm: RealExpr::FromInt(IntExpr::constant(1)) },
                    ),
                ),
                (
                    "e".to_owned(),
                    virtual_decl(
                        matrix_type(),
                        VirtualKind::Bounded { norm: RealExpr::FromInt(IntExpr::constant(1)) },
                    ),
                ),
            ]),
            term_lists: BTreeMap::from([
                (
                    assumed_id("B"),
                    AssumedTermList { terms: vec![overlay_term(1, vec![virtual_factor("B")])] },
                ),
                (
                    assumed_id("c"),
                    AssumedTermList {
                        terms: vec![
                            overlay_term(1, vec![virtual_factor("s"), virtual_factor("B")]),
                            overlay_term(1, vec![virtual_factor("e")]),
                        ],
                    },
                ),
            ]),
            entries: vec![
                (
                    selector(1),
                    Reinterpretation::Unfold(UnfoldSpec {
                        new_terms: assumed_id("B"),
                        replace_derived: false,
                    }),
                ),
                (
                    selector(2),
                    Reinterpretation::Unfold(UnfoldSpec {
                        new_terms: assumed_id("c"),
                        replace_derived: false,
                    }),
                ),
            ],
        }
    }

    fn two_input_graph() -> Graph {
        let mut graph = empty_graph(
            "overlay-unfold",
            vec![matrix_input(1, "B", false), matrix_input(2, "c", false)],
            wire(2, 0),
        );
        graph.input_types.insert("B".to_owned(), WireType::Matrix(matrix_type()));
        graph.input_types.insert("c".to_owned(), WireType::Matrix(matrix_type()));
        graph
    }

    #[test]
    fn unfold_replaces_sources_with_shared_virtual_atoms_and_hashes_ignore_entry_order() {
        let graph = two_input_graph();
        let overlay = unfold_overlay();
        let elaborated =
            elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &overlay).expect("unfold");
        let b_terms = elaborated_terms(&elaborated, wire(1, 0));
        let c_terms = elaborated_terms(&elaborated, wire(2, 0));
        assert_eq!(b_terms.terms[0].factors[0].atom, AtomId::Virtual { name: "B".to_owned() });
        assert_eq!(c_terms.terms.len(), 2);
        assert!(elaborated.assumption_hash.is_some());
        assert!(elaborated.assumption_digests.contains(&elaborated.assumption_hash.unwrap()));

        let mut reordered = overlay.clone();
        reordered.entries.reverse();
        let reordered =
            elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &reordered).expect("reorder");
        assert_eq!(elaborated.overlay_hash, reordered.overlay_hash);
        assert_eq!(elaborated.assumption_hash, reordered.assumption_hash);
        assert_eq!(elaborated.wires, reordered.wires);
    }

    #[test]
    fn unfold_enforces_kind_character_and_replace_derived_discipline() {
        let graph = empty_graph(
            "replace-derived",
            vec![
                bounded_sample(1),
                bounded_sample(2),
                Node {
                    id: NodeId(3),
                    kind: NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
            ],
            wire(3, 0),
        );
        let overlay = SymbolicOverlay {
            virtual_atoms: BTreeMap::from([(
                "e".to_owned(),
                virtual_decl(
                    matrix_type(),
                    VirtualKind::Bounded { norm: RealExpr::FromInt(IntExpr::constant(1)) },
                ),
            )]),
            term_lists: BTreeMap::from([(
                assumed_id("e"),
                AssumedTermList { terms: vec![overlay_term(1, vec![virtual_factor("e")])] },
            )]),
            entries: vec![(
                selector(3),
                Reinterpretation::Unfold(UnfoldSpec {
                    new_terms: assumed_id("e"),
                    replace_derived: false,
                }),
            )],
        };
        let error = elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &overlay)
            .expect_err("derived replacement requires opt-in");
        assert!(error.to_string().contains("replace_derived"));

        let mut accepted = overlay;
        let Reinterpretation::Unfold(spec) = &mut accepted.entries[0].1 else { unreachable!() };
        spec.replace_derived = true;
        let elaborated =
            elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &accepted).expect("opt-in");
        assert!(
            elaborated
                .warnings
                .iter()
                .any(|warning| { warning.kind == WarningKind::ReplacedDerivedDescription })
        );

        let bounded_graph = empty_graph("bounded", vec![bounded_sample(1)], wire(1, 0));
        let large_overlay = SymbolicOverlay {
            virtual_atoms: BTreeMap::from([(
                "large".to_owned(),
                virtual_decl(matrix_type(), VirtualKind::Large),
            )]),
            term_lists: BTreeMap::from([(
                assumed_id("large"),
                AssumedTermList { terms: vec![overlay_term(1, vec![virtual_factor("large")])] },
            )]),
            entries: vec![(
                selector(1),
                Reinterpretation::Unfold(UnfoldSpec {
                    new_terms: assumed_id("large"),
                    replace_derived: false,
                }),
            )],
        };
        let error =
            elaborate_with_overlay(&bounded_graph, &ParamEnv::default(), &[], &large_overlay)
                .expect_err("bounded cannot become large");
        assert!(error.to_string().contains("kind character"));
    }

    #[test]
    fn unfold_of_preimage_wire_is_auditable_and_cycles_fail_structurally() {
        let mut graph =
            empty_graph("preimage-unfold", vec![matrix_input(1, "k", true)], wire(1, 0));
        graph.input_types.insert("k".to_owned(), WireType::Preimage(matrix_type()));
        let overlay = SymbolicOverlay {
            virtual_atoms: BTreeMap::from([(
                "k".to_owned(),
                virtual_decl(
                    matrix_type(),
                    VirtualKind::Bounded { norm: RealExpr::FromInt(IntExpr::constant(1)) },
                ),
            )]),
            term_lists: BTreeMap::from([(
                assumed_id("k"),
                AssumedTermList { terms: vec![overlay_term(1, vec![virtual_factor("k")])] },
            )]),
            entries: vec![(
                selector(1),
                Reinterpretation::Unfold(UnfoldSpec {
                    new_terms: assumed_id("k"),
                    replace_derived: false,
                }),
            )],
        };
        let elaborated =
            elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &overlay).expect("unfold");
        assert!(
            elaborated
                .warnings
                .iter()
                .any(|warning| { warning.kind == WarningKind::DroppedPreimageReferences })
        );

        let mut cyclic = overlay;
        cyclic.virtual_atoms.get_mut("k").unwrap().preimage = Some(AssumedPreimage {
            uniform: AtomRef::Virtual { name: "k".to_owned() },
            target: assumed_id("k"),
        });
        let error =
            elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &cyclic).expect_err("cycle");
        assert!(error.to_string().contains("cycle"));
    }

    #[test]
    fn assumed_preimage_types_are_checked_and_virtual_uniforms_rewrite() {
        let mut graph =
            empty_graph("assumed-preimage", vec![matrix_input(1, "c", false)], wire(1, 0));
        graph.input_types.insert("c".to_owned(), WireType::Matrix(matrix_type()));
        let overlay = SymbolicOverlay {
            virtual_atoms: BTreeMap::from([
                ("A".to_owned(), virtual_decl(matrix_type(), VirtualKind::Large)),
                (
                    "K".to_owned(),
                    VirtualAtomDecl {
                        matrix_type: matrix_type(),
                        kind: VirtualKind::Bounded {
                            norm: RealExpr::FromInt(IntExpr::constant(1)),
                        },
                        preimage: Some(AssumedPreimage {
                            uniform: AtomRef::Virtual { name: "A".to_owned() },
                            target: assumed_id("target"),
                        }),
                    },
                ),
                ("target".to_owned(), virtual_decl(matrix_type(), VirtualKind::Large)),
            ]),
            term_lists: BTreeMap::from([
                (
                    assumed_id("target"),
                    AssumedTermList {
                        terms: vec![overlay_term(1, vec![virtual_factor("target")])],
                    },
                ),
                (
                    assumed_id("c"),
                    AssumedTermList {
                        terms: vec![overlay_term(
                            1,
                            vec![virtual_factor("A"), virtual_factor("K")],
                        )],
                    },
                ),
            ]),
            entries: vec![(
                selector(1),
                Reinterpretation::Unfold(UnfoldSpec {
                    new_terms: assumed_id("c"),
                    replace_derived: false,
                }),
            )],
        };
        let elaborated =
            elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &overlay).expect("rewrite");
        assert_eq!(
            elaborated_terms(&elaborated, wire(1, 0)),
            &TermList::atom(AtomId::Virtual { name: "target".to_owned() })
        );

        let mut invalid = overlay;
        let mut scalar = matrix_type();
        scalar.rows = IntExpr::constant(1);
        scalar.columns = IntExpr::constant(1);
        invalid.virtual_atoms.get_mut("target").unwrap().matrix_type = scalar;
        let error = elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &invalid)
            .expect_err("target type mismatch");
        assert!(error.to_string().contains("wrong matrix type"));
    }

    #[test]
    fn residual_fold_retains_an_exact_derived_definition() {
        let graph = empty_graph(
            "fold",
            vec![
                bounded_sample(1),
                Node {
                    id: NodeId(2),
                    kind: NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                    args: vec![wire(1, 0), wire(1, 0)],
                },
            ],
            wire(2, 0),
        );
        let expected = OverlayTerm {
            coefficient: IntExpr::constant(2),
            factors: vec![FactorRef {
                atom: AtomRef::Node {
                    path: Vec::new(),
                    node: NodeId(1),
                    port: PortMatcher::Concrete(0),
                },
                view: None,
            }],
        };
        let overlay = SymbolicOverlay {
            entries: vec![(
                selector(2),
                Reinterpretation::Fold(FoldSpec {
                    expected: ExpectedTermList { entries: vec![ExpectedEntry::Term(expected)] },
                    groups: vec![FoldGroup::Residual { terms: BTreeSet::from([0]) }],
                }),
            )],
            ..SymbolicOverlay::default()
        };
        let elaborated =
            elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &overlay).expect("fold");
        let atom_id = &elaborated_terms(&elaborated, wire(2, 0)).terms[0].factors[0].atom;
        let atom = elaborated.atoms.get(atom_id).expect("fold atom");
        assert!(matches!(atom.class, AtomClass::Derived { definition: DefExpr::Fold(_) }));
        assert!(elaborated.assumption_hash.is_none());
        assert!(elaborated.overlay_hash.is_some());
    }

    #[test]
    fn signal_fold_keeps_the_common_suffix_and_folds_only_the_prefix() {
        let graph = empty_graph(
            "signal-fold",
            vec![
                bounded_sample(1),
                bounded_sample(2),
                Node {
                    id: NodeId(3),
                    kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                    args: vec![wire(1, 0), wire(2, 0)],
                },
            ],
            wire(3, 0),
        );
        let node_factor = |node| FactorRef {
            atom: AtomRef::Node {
                path: Vec::new(),
                node: NodeId(node),
                port: PortMatcher::Concrete(0),
            },
            view: None,
        };
        let overlay = SymbolicOverlay {
            entries: vec![(
                selector(3),
                Reinterpretation::Fold(FoldSpec {
                    expected: ExpectedTermList {
                        entries: vec![ExpectedEntry::Term(overlay_term(
                            1,
                            vec![node_factor(1), node_factor(2)],
                        ))],
                    },
                    groups: vec![FoldGroup::Signal { terms: BTreeSet::from([0]), suffix_len: 1 }],
                }),
            )],
            ..SymbolicOverlay::default()
        };
        let elaborated =
            elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &overlay).expect("fold");
        let output = elaborated_terms(&elaborated, wire(3, 0));
        assert_eq!(output.terms.len(), 1);
        assert_eq!(output.terms[0].factors.len(), 2);
        let fold = elaborated.atoms.get(&output.terms[0].factors[0].atom).expect("fold atom");
        let AtomClass::Derived { definition: DefExpr::Fold(definition) } = &fold.class else {
            panic!("signal prefix must be retained as a fold definition")
        };
        assert_eq!(definition.terms[0].factors.len(), 1);
        assert_eq!(
            output.terms[0].factors[1].atom,
            AtomId::Local { instantiation_path: Vec::new(), node: NodeId(2), port: 0 }
        );
    }

    #[test]
    fn indicator_sum_expands_affine_ports_as_one_fold_position() {
        let graph = empty_graph(
            "indicator-overlay",
            vec![
                Node {
                    id: NodeId(1),
                    kind: NodeKind::ConstantInt(BigInt::zero()),
                    args: Vec::new(),
                },
                bounded_sample(2),
                bounded_sample(3),
                Node {
                    id: NodeId(4),
                    kind: NodeKind::Output { name: "branches".to_owned() },
                    args: vec![wire(2, 0), wire(3, 0)],
                },
                Node {
                    id: NodeId(5),
                    kind: NodeKind::Select { count: IntExpr::constant(2) },
                    args: vec![wire(1, 0), wire(4, 0), wire(4, 1)],
                },
                bounded_sample(6),
                Node {
                    id: NodeId(7),
                    kind: NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
                    args: vec![wire(5, 0), wire(6, 0)],
                },
            ],
            wire(7, 0),
        );
        let overlay = SymbolicOverlay {
            entries: vec![(
                selector(7),
                Reinterpretation::Fold(FoldSpec {
                    expected: ExpectedTermList {
                        entries: vec![ExpectedEntry::IndicatorSum {
                            select: SelectNodeSelector { path: Vec::new(), node: NodeId(5) },
                            index_var: "i".to_owned(),
                            body: overlay_term(
                                1,
                                vec![
                                    FactorRef {
                                        atom: AtomRef::Node {
                                            path: Vec::new(),
                                            node: NodeId(4),
                                            port: PortMatcher::Affine {
                                                var: "i".to_owned(),
                                                stride: 1,
                                                offset: 0,
                                            },
                                        },
                                        view: None,
                                    },
                                    FactorRef {
                                        atom: AtomRef::Node {
                                            path: Vec::new(),
                                            node: NodeId(6),
                                            port: PortMatcher::Concrete(0),
                                        },
                                        view: None,
                                    },
                                ],
                            ),
                        }],
                    },
                    groups: vec![FoldGroup::Residual { terms: BTreeSet::from([0]) }],
                }),
            )],
            ..SymbolicOverlay::default()
        };
        let elaborated =
            elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &overlay).expect("fold");
        let fold = elaborated
            .atoms
            .get(&elaborated_terms(&elaborated, wire(7, 0)).terms[0].factors[0].atom)
            .expect("fold atom");
        let AtomClass::Derived { definition: DefExpr::Fold(definition) } = &fold.class else {
            panic!("residual fold")
        };
        assert_eq!(definition.terms.len(), 2);
        assert!(definition.terms.iter().all(|term| {
            term.factors.iter().any(|factor| matches!(factor.atom, AtomId::Indicator { .. }))
        }));
    }

    #[test]
    fn overlay_rejects_same_node_references_and_duplicate_targets() {
        let graph = empty_graph(
            "same-node",
            vec![
                bounded_sample(1),
                bounded_sample(2),
                Node {
                    id: NodeId(3),
                    kind: NodeKind::Output { name: "family".to_owned() },
                    args: vec![wire(1, 0), wire(2, 0)],
                },
            ],
            wire(3, 0),
        );
        let same_node_term = OverlayTerm {
            coefficient: IntExpr::constant(1),
            factors: vec![FactorRef {
                atom: AtomRef::Node {
                    path: Vec::new(),
                    node: NodeId(3),
                    port: PortMatcher::Concrete(1),
                },
                view: None,
            }],
        };
        let overlay = SymbolicOverlay {
            entries: vec![(
                selector(3),
                Reinterpretation::Fold(FoldSpec {
                    expected: ExpectedTermList {
                        entries: vec![ExpectedEntry::Term(same_node_term)],
                    },
                    groups: vec![FoldGroup::Keep { terms: BTreeSet::from([0]) }],
                }),
            )],
            ..SymbolicOverlay::default()
        };
        let error = elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &overlay)
            .expect_err("same-node reference");
        assert!(error.to_string().contains("own node"));

        let graph = empty_graph("duplicate", vec![bounded_sample(1)], wire(1, 0));
        let fold = Reinterpretation::Fold(FoldSpec {
            expected: ExpectedTermList::default(),
            groups: Vec::new(),
        });
        let duplicate = SymbolicOverlay {
            entries: vec![(selector(1), fold.clone()), (selector(1), fold)],
            ..SymbolicOverlay::default()
        };
        let error = elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &duplicate)
            .expect_err("duplicate");
        assert!(error.to_string().contains("multiple overlay selectors"));
    }

    #[test]
    fn unused_overlay_declarations_and_selectors_are_reported() {
        let graph = empty_graph("unused", vec![bounded_sample(1)], wire(1, 0));
        let overlay = SymbolicOverlay {
            virtual_atoms: BTreeMap::from([(
                "unused".to_owned(),
                virtual_decl(matrix_type(), VirtualKind::Large),
            )]),
            term_lists: BTreeMap::from([(assumed_id("unused"), AssumedTermList::default())]),
            entries: vec![(
                selector(99),
                Reinterpretation::Fold(FoldSpec {
                    expected: ExpectedTermList::default(),
                    groups: Vec::new(),
                }),
            )],
        };
        let elaborated =
            elaborate_with_overlay(&graph, &ParamEnv::default(), &[], &overlay).expect("warnings");
        assert!(
            elaborated
                .warnings
                .iter()
                .any(|warning| { warning.kind == WarningKind::UnusedOverlaySelector })
        );
        assert!(
            elaborated
                .warnings
                .iter()
                .any(|warning| { warning.kind == WarningKind::UnusedVirtualAtom })
        );
        assert!(
            elaborated
                .warnings
                .iter()
                .any(|warning| { warning.kind == WarningKind::UnusedAssumedTermList })
        );
    }

    #[test]
    fn manifest_reexport_preserves_assumption_provenance_and_atom_origin() {
        let producer_graph = two_input_graph();
        let producer_elaboration =
            elaborate_with_overlay(&producer_graph, &ParamEnv::default(), &[], &unfold_overlay())
                .expect("producer");
        let producer = production_id(crate::atom::SpecHash([21; 32]), [22; 32]);
        let matrix = ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 8,
            rows: 2,
            columns: 2,
        };
        let producer_manifest = export_manifest(
            producer.clone(),
            &BTreeMap::from([(
                "c".to_owned(),
                ExportArtifact {
                    wire: WireId { instantiation_path: Vec::new(), wire: wire(2, 0) },
                    wire_type: matrix.clone(),
                    family: None,
                    content_hash: None,
                    layout: None,
                },
            )]),
            &producer_elaboration.wire_terms(),
            &producer_elaboration.target_terms,
            &producer_elaboration.atoms,
            &producer_elaboration.manifest_metadata(),
        )
        .expect("producer manifest");

        let mut consumer_graph = empty_graph(
            "consumer",
            vec![Node {
                id: NodeId(1),
                kind: NodeKind::Input {
                    name: "c".to_owned(),
                    wire_type: WireType::Matrix(matrix_type()),
                    artifact: Some(ArtifactInput {
                        production_id: producer.clone(),
                        artifact_name: "c".to_owned(),
                        family_count: None,
                    }),
                },
                args: Vec::new(),
            }],
            wire(1, 0),
        );
        consumer_graph.input_types.insert("c".to_owned(), WireType::Matrix(matrix_type()));
        let consumer = elaborate_with_manifests(
            &consumer_graph,
            &ParamEnv::default(),
            std::slice::from_ref(&producer_manifest),
        )
        .expect("consumer");
        let producer_assumption = producer_manifest.assumption_hash.expect("assumption");
        assert!(consumer.assumption_digests.contains(&producer_assumption));
        assert!(consumer.assumption_hash.is_none());

        let intermediary = production_id(crate::atom::SpecHash([23; 32]), [24; 32]);
        let intermediary_manifest = export_manifest(
            intermediary,
            &BTreeMap::from([(
                "c".to_owned(),
                ExportArtifact {
                    wire: WireId { instantiation_path: Vec::new(), wire: wire(1, 0) },
                    wire_type: matrix,
                    family: None,
                    content_hash: None,
                    layout: None,
                },
            )]),
            &consumer.wire_terms(),
            &consumer.target_terms,
            &consumer.atoms,
            &consumer.manifest_metadata(),
        )
        .expect("intermediary manifest");
        assert!(intermediary_manifest.assumption_digests.contains(&producer_assumption));
        assert!(intermediary_manifest.atoms.keys().any(|reference| {
            matches!(
                reference,
                crate::manifest::ManifestAtomRef::Imported {
                    production_id,
                    ..
                } if production_id == &producer
            )
        }));
        let imported = import_manifest(&intermediary_manifest).expect("standalone intermediary");
        assert!(imported.atoms.iter().any(|(id, _)| {
            matches!(
                id,
                AtomId::Imported {
                    production_id,
                    ..
                } if production_id == &producer
            )
        }));
    }
}
