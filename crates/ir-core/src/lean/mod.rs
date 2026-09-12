//! Mechanical extraction of a validated frozen graph into ordinary Lean relations.
//!
//! This module is deliberately application agnostic.  It does not construct a graph, inspect
//! Diamond roles, or provide semantic fallbacks for an operation it cannot translate.  Primitive
//! names are supplied by the package owning those primitive relations.

pub mod claim;
#[cfg(test)]
mod fixtures;
pub mod protocol;

use crate::{
    expr::{IntExpr, RealExpr},
    graph::{FrozenGraphScopeId, Graph, GraphScope},
    node::{ConstantMatrix, IntBinaryOp, IntCompareOp, MatrixBinaryOp, NodeKind},
    types::{NodeId, WireRef},
    validate::ValidatedGraph,
};
use sha2::{Digest, Sha256};
use std::{
    cell::{Cell, RefCell},
    collections::{BTreeMap, BTreeSet},
    rc::Rc,
};
use thiserror::Error;

/// Names of relations/functions supplied by the concrete primitive Lean package.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrimitiveNames {
    pub matrix_type: String,
    pub trapdoor_type: String,
    pub preimage_type: String,
    pub small_matrix_type: String,
    pub blob_type: String,
    pub matrix_add: String,
    pub matrix_sub: String,
    pub matrix_mul: String,
    pub matrix_mul_scalar_left: String,
    pub matrix_mul_scalar_right: String,
    pub matrix_neg: String,
    pub matrix_scale: String,
    pub matrix_polynomial: String,
    pub transpose: String,
    pub matrix_slice: String,
    pub concat_rows: String,
    pub concat_columns: String,
    pub concat_diagonal: String,
    pub trapdoor_sample: String,
    pub preimage_sample: String,
    pub uniform_residue_sample: String,
    pub uniform_interval_sample: String,
    pub gaussian_sample: String,
    pub hash_sample: String,
    pub gadget_trapdoor: String,
    pub gadget_decompose: String,
    pub matrix_mul_accumulate: String,
    pub extract_coefficient: String,
    pub lift_integer: String,
    pub threshold_decode: String,
    pub crt_recompose: String,
    pub modulus_switch: String,
    pub modulus_reduce: String,
    pub centered_rebase: String,
    pub rns_mod_up: String,
    pub rns_mod_down: String,
    pub centered_extend: String,
    pub block_mod_switch: String,
    pub ring_automorphism: String,
    pub pack_polynomial: String,
    pub polynomial_from_values: String,
    pub polynomial_values: String,
    pub family_pack: String,
    pub family_get_static: String,
    pub family_get_dynamic: String,
    pub select: String,
    pub trapdoor_public: String,
    pub int_divisible: String,
}

impl Default for PrimitiveNames {
    fn default() -> Self {
        Self {
            matrix_type: "Mxx.Primitives.ExactMatrix".into(),
            trapdoor_type: "MxxRuntime.TrapdoorValue".into(),
            preimage_type: "Mxx.Primitives.ExactMatrix".into(),
            small_matrix_type: "Mxx.Primitives.ExactMatrix".into(),
            blob_type: "MxxRuntime.Blob".into(),
            matrix_add: "MxxRuntime.matrixAdd".into(),
            matrix_sub: "MxxRuntime.matrixSub".into(),
            matrix_mul: "MxxRuntime.matrixMul".into(),
            matrix_mul_scalar_left: "MxxRuntime.matrixMulScalarLeft".into(),
            matrix_mul_scalar_right: "MxxRuntime.matrixMulScalarRight".into(),
            matrix_neg: "MxxRuntime.matrixNeg".into(),
            matrix_scale: "MxxRuntime.matrixScale".into(),
            matrix_polynomial: "MxxRuntime.matrixPolynomial".into(),
            transpose: "MxxRuntime.transpose".into(),
            matrix_slice: "MxxRuntime.sliceMatrix".into(),
            concat_rows: "MxxRuntime.concatRows".into(),
            concat_columns: "MxxRuntime.concatColumns".into(),
            concat_diagonal: "MxxRuntime.concatDiagonal".into(),
            trapdoor_sample: "MxxRuntime.trapdoorSample".into(),
            preimage_sample: "MxxRuntime.preimageRunsDispatched".into(),
            uniform_residue_sample: "MxxRuntime.uniformResidueSample".into(),
            uniform_interval_sample: "MxxRuntime.uniformIntervalSample".into(),
            gaussian_sample: "MxxRuntime.gaussianSample".into(),
            hash_sample: "MxxRuntime.hashSample".into(),
            gadget_trapdoor: "MxxRuntime.gadgetTrapdoorRuns".into(),
            gadget_decompose: "MxxRuntime.gadgetDecomposeRuns".into(),
            matrix_mul_accumulate: "MxxRuntime.matrixMulAccumulate".into(),
            extract_coefficient: "MxxRuntime.extractCoefficient".into(),
            lift_integer: "MxxRuntime.liftInteger".into(),
            threshold_decode: "MxxRuntime.thresholdDecode".into(),
            crt_recompose: "MxxRuntime.crtRecomposeLevel".into(),
            modulus_switch: "MxxRuntime.modulusSwitchRuns".into(),
            modulus_reduce: "MxxRuntime.modulusReduceRuns".into(),
            centered_rebase: "MxxRuntime.centeredRebaseRuns".into(),
            rns_mod_up: "MxxRuntime.rnsModUpRuns".into(),
            rns_mod_down: "MxxRuntime.rnsModDownRuns".into(),
            centered_extend: "MxxRuntime.centeredExtendRuns".into(),
            block_mod_switch: "MxxRuntime.blockModSwitchRuns".into(),
            ring_automorphism: "MxxRuntime.ringAutomorphismRuns".into(),
            pack_polynomial: "MxxRuntime.packPolynomial".into(),
            polynomial_from_values: "MxxRuntime.polynomialFromValues".into(),
            polynomial_values: "MxxRuntime.polynomialValues".into(),
            family_pack: "MxxRuntime.familyPack".into(),
            family_get_static: "MxxRuntime.familyGetStatic".into(),
            family_get_dynamic: "MxxRuntime.familyGetDynamic".into(),
            select: "MxxRuntime.select".into(),
            trapdoor_public: "MxxRuntime.trapdoorPublic".into(),
            int_divisible: "MxxRuntime.intDivisible".into(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendLayout {
    pub modulus: num_bigint::BigInt,
    pub ring_dimension: usize,
    pub base: num_bigint::BigInt,
    pub regular_digits: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExportOptions {
    pub module_name: String,
    pub namespace: String,
    /// Import containing the concrete primitive relations.
    pub runtime_import: String,
    pub primitives: PrimitiveNames,
    /// Layout metadata from the same concrete backend setup used by the Lean context.
    pub backend_layouts: Vec<BackendLayout>,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            module_name: "GeneratedProgram".into(),
            namespace: "Generated".into(),
            runtime_import: "MxxRuntime".into(),
            primitives: Default::default(),
            backend_layouts: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceMapEntry {
    pub scope: FrozenGraphScopeId,
    pub node: Option<NodeId>,
    pub port: Option<u32>,
    pub generated: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct SourceMap {
    pub entries: Vec<SourceMapEntry>,
}

/// A named boundary value, located in the complete stored scope tuple.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BoundaryValue {
    pub wire: WireRef,
    pub wire_type: crate::types::ConcreteWireType,
    pub lean_type: String,
    pub tuple_index: usize,
    /// Projection from the binder `inputs` or `outputs` of the root relation.
    pub projection: String,
}

/// Mechanical root interface for workflow linking and application claim assembly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParameterField {
    pub lean_type: String,
    /// Exact root binding when present. None denotes a child-only field: lexical calls
    /// initialize it before use, so its initial root record value is not read.
    pub root_value: Option<String>,
}

/// Mechanical root interface for workflow linking and application claim assembly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RootBoundary {
    pub relation: String,
    pub input_type: String,
    pub output_type: String,
    pub input_count: usize,
    pub output_count: usize,
    pub inputs: BTreeMap<String, BoundaryValue>,
    pub outputs: BTreeMap<String, BoundaryValue>,
    pub requires_backend: bool,
    pub requires_hash_model: bool,
    pub parameter_type: String,
    pub parameters: BTreeMap<String, ParameterField>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScopeProofShape {
    /// Named witness view; leaf execution relations retain their existential telescope.
    pub witness_record: Option<String>,
    /// Exact body at one fixed witness, for composing intermediate-node proofs.
    pub body_relation: Option<String>,
    /// Checked conversion from the execution relation to the named witness view.
    pub runs_iff_body: String,
    /// Emitted value right-hand sides; references name earlier entries or root arguments.
    pub value_expressions: BTreeMap<String, String>,
    /// Witness fields, also the leaf existential binders, in exact exported order. Pure lets
    /// are deliberately absent: Lean unfolds them while eliminating `Runs`.
    pub witnesses: Vec<(String, String)>,
    /// Positions in the right-associated conjunction contributed by each node,
    /// including its guards. The final output equation is at `constraint_count`.
    pub node_constraints: BTreeMap<NodeId, std::ops::Range<usize>>,
    /// Checked accessors for each node's facts, in local constraint order.
    /// Consumers apply each name to a body proof obtained through `runs_iff_body`.
    pub node_facts: BTreeMap<NodeId, Vec<String>>,
    /// Checked accessor for the final output equation of a record-backed scope.
    pub output_fact: Option<String>,
    pub constraint_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeanArtifact {
    pub backend_layouts: Vec<BackendLayout>,
    pub module_name: String,
    pub source: String,
    pub source_map: SourceMap,
    pub root: RootBoundary,
    pub scope_relations: BTreeMap<FrozenGraphScopeId, String>,
    pub scope_proofs: BTreeMap<FrozenGraphScopeId, ScopeProofShape>,
    pub digest: [u8; 32],
    pub spec_hash: crate::artifact::SpecHash,
    pub static_node_visits: usize,
}

#[derive(Debug, Error)]
pub enum ExportError {
    #[error("graph encoding error: {0}")]
    Encoding(String),
    #[error("backend layout error: {0}")]
    BackendLayout(String),
    #[error("scope {scope:?}, node {node:?}: unsupported operation {operation}")]
    UnsupportedOperation { scope: FrozenGraphScopeId, node: NodeId, operation: String },
    #[error("scope {scope:?}, node {node:?}: missing frozen arguments")]
    MissingArguments { scope: FrozenGraphScopeId, node: NodeId },
    #[error("scope {scope:?}, node {node:?}: output port {port} is unavailable")]
    MissingOutput { scope: FrozenGraphScopeId, node: NodeId, port: u32 },
    #[error("scope {scope:?}: child scope is unavailable for node {node:?}")]
    MissingChildScope { scope: FrozenGraphScopeId, node: NodeId },
    #[error("invalid graph parameter name {0:?}")]
    InvalidIdentifier(String),
    #[error("distinct graph scopes map to the same Lean declaration {0:?}")]
    ScopeNameCollision(String),
    #[error("loop-index binder {0} is unavailable at its use site")]
    MissingLoopIndex(u32),
}

#[derive(Clone, Debug, Default)]
struct LexicalEnv {
    vars: BTreeMap<String, String>,
    loop_indices: BTreeMap<u32, String>,
    loop_index_nats: BTreeMap<u32, String>,
    missing_loop_index: Cell<Option<u32>>,
    // Cloned child environments share references, but record the resolved lexical name:
    // a child's shadowing index `i` must not mark an outer `i_slot` as used.
    referenced_loop_names: Rc<RefCell<BTreeSet<String>>>,
}

impl LexicalEnv {
    fn expr(&self, expr: &IntExpr) -> String {
        match expr {
            IntExpr::Const(value) => value.to_string(),
            IntExpr::Var(name) => {
                self.vars.get(name).cloned().unwrap_or_else(|| format!("params.«{name}»"))
            }
            IntExpr::LoopIndex(slot) => {
                if let Some(name) = self.loop_index_nats.get(slot) {
                    self.referenced_loop_names.borrow_mut().insert(name.clone());
                }
                self.loop_indices.get(slot).cloned().unwrap_or_else(|| {
                    self.missing_loop_index.set(Some(*slot));
                    format!("__missing_loop_index_{slot}")
                })
            }
            IntExpr::Add(a, b) => format!("({} + {})", self.expr(a), self.expr(b)),
            IntExpr::Sub(a, b) => format!("({} - {})", self.expr(a), self.expr(b)),
            IntExpr::Mul(a, b) => format!("({} * {})", self.expr(a), self.expr(b)),
            IntExpr::Div(a, b) => {
                format!("MxxIR.exactDiv ({}) ({})", self.expr(a), self.expr(b))
            }
            IntExpr::FloorDiv(a, b) => format!("Int.fdiv ({}) ({})", self.expr(a), self.expr(b)),
            IntExpr::Rem(a, b) => format!("Int.fmod ({}) ({})", self.expr(a), self.expr(b)),
            IntExpr::Select { selector, branches } => format!(
                "MxxIR.selectInt ({}) [{}]",
                self.expr(selector),
                branches
                    .iter()
                    .map(|branch| format!("({})", self.expr(branch)))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            IntExpr::RoundDiv(a, b) => {
                format!("MxxIR.roundDiv ({}) ({})", self.expr(a), self.expr(b))
            }
            IntExpr::Log2Ceil(a) => format!("MxxIR.log2Ceil ({})", self.expr(a)),
        }
    }

    fn real_expr(&self, expr: &RealExpr) -> String {
        match expr {
            RealExpr::Rational(r) => format!("({}/{})", r.numerator(), r.denominator()),
            RealExpr::Var(name) => {
                self.vars.get(name).cloned().unwrap_or_else(|| format!("params.«{name}»"))
            }
            RealExpr::FromInt(value) => self.expr(value),
            RealExpr::Add(a, b) => format!("({} + {})", self.real_expr(a), self.real_expr(b)),
            RealExpr::Sub(a, b) => format!("({} - {})", self.real_expr(a), self.real_expr(b)),
            RealExpr::Mul(a, b) => format!("({} * {})", self.real_expr(a), self.real_expr(b)),
            RealExpr::Div(a, b) => format!("({} / {})", self.real_expr(a), self.real_expr(b)),
            RealExpr::Sqrt(a) => format!("Real.sqrt {}", self.real_expr(a)),
        }
    }

    fn take_missing_loop_index(&self) -> Option<u32> {
        self.missing_loop_index.take()
    }
}

/// Conditions required by the source expression evaluator.  These are emitted into the
/// generated relation instead of allowing Lean's total integer operators to silently define an
/// invalid IR expression.
fn expression_guards(expr: &IntExpr, env: &LexicalEnv) -> Vec<String> {
    let mut guards = Vec::new();
    fn visit(expr: &IntExpr, env: &LexicalEnv, guards: &mut Vec<String>) {
        match expr {
            IntExpr::Const(_) | IntExpr::Var(_) | IntExpr::LoopIndex(_) => {}
            IntExpr::Add(a, b) | IntExpr::Sub(a, b) | IntExpr::Mul(a, b) => {
                visit(a, env, guards);
                visit(b, env, guards);
            }
            IntExpr::Div(a, b) => {
                visit(a, env, guards);
                visit(b, env, guards);
                let numerator = env.expr(a);
                let denominator = env.expr(b);
                guards.push(format!("{} ≠ 0", denominator));
                guards.push(format!("{} % {} = 0", numerator, denominator));
            }
            IntExpr::FloorDiv(a, b) | IntExpr::Rem(a, b) => {
                visit(a, env, guards);
                visit(b, env, guards);
                guards.push(format!("({}) ≠ 0", env.expr(b)));
            }
            IntExpr::Select { selector, branches } => {
                visit(selector, env, guards);
                let selector = env.expr(selector);
                guards.push(format!("0 ≤ ({selector})"));
                guards.push(format!("({selector}) < {}", branches.len()));
                for (index, branch) in branches.iter().enumerate() {
                    let mut branch_guards = Vec::new();
                    visit(branch, env, &mut branch_guards);
                    for guard in branch_guards {
                        guards.push(format!("(({selector}) = {index} → ({guard}))"));
                    }
                }
            }
            IntExpr::RoundDiv(a, b) => {
                visit(a, env, guards);
                visit(b, env, guards);
                guards.push(format!("{} > 0", env.expr(b)));
            }
            IntExpr::Log2Ceil(a) => {
                visit(a, env, guards);
                guards.push(format!("{} ≥ 1", env.expr(a)));
            }
        }
    }
    visit(expr, env, &mut guards);
    guards
}

fn append_expression_guards(expr: &IntExpr, env: &LexicalEnv, relations: &mut Vec<String>) {
    relations.extend(expression_guards(expr, env));
}

/// Export one validated frozen IR into a deterministic Lean module.
pub fn export(
    validated: &ValidatedGraph,
    options: &ExportOptions,
) -> Result<LeanArtifact, ExportError> {
    let emitter = Emitter::new(validated, options);
    emitter.emit()
}

struct Emitter<'a> {
    graph: &'a Graph,
    validated: &'a ValidatedGraph,
    options: &'a ExportOptions,
    source: String,
    source_map: SourceMap,
    scopes: BTreeMap<FrozenGraphScopeId, String>,
    params: BTreeSet<String>,
    static_node_visits: usize,
    indent: usize,
    requires_backend: bool,
    requires_hash_model: bool,
    layout_environments: BTreeMap<FrozenGraphScopeId, Vec<crate::expr::ParamEnv>>,
    current_wire_types: BTreeMap<String, String>,
    current_scope_values: Vec<(String, String)>,
    current_referenced_wires: BTreeSet<String>,
    current_anonymous_lets: BTreeSet<String>,
    current_uses_hash_model: bool,
    current_witnesses: Vec<(String, String)>,
    current_runs_record: bool,
    current_witness_bindings: Vec<std::ops::Range<usize>>,
    current_value_expressions: BTreeMap<String, String>,
    scope_proofs: BTreeMap<FrozenGraphScopeId, ScopeProofShape>,
}

impl<'a> Emitter<'a> {
    fn new(validated: &'a ValidatedGraph, options: &'a ExportOptions) -> Self {
        let graph = &validated.source;
        let scopes = graph.scopes().keys().map(|id| (id.clone(), scope_name(id))).collect();
        let mut params = graph.parameters().iter().map(|p| p.name.clone()).collect::<BTreeSet<_>>();
        for scope in graph.scopes().values() {
            for node in scope.nodes() {
                let bindings = match node.kind() {
                    NodeKind::SubgraphCall(call) => &call.bindings,
                    NodeKind::ParallelLoop(spec) => &spec.bindings,
                    NodeKind::SequentialLoop(spec) => &spec.bindings,
                    _ => continue,
                };
                params.extend(bindings.iter().map(|(name, _)| name.clone()));
            }
        }
        Self {
            graph,
            validated,
            options,
            source: String::new(),
            source_map: Default::default(),
            scopes,
            params,
            static_node_visits: 0,
            indent: 1,
            requires_backend: graph.scopes().values().any(|scope| {
                scope.nodes().iter().any(|node| {
                    matches!(
                        node.kind(),
                        NodeKind::GadgetTrapdoor { .. } |
                            NodeKind::GadgetDecompose { .. } |
                            NodeKind::TrapdoorSample { .. } |
                            NodeKind::PreimageSample { .. } |
                            NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { .. }, .. }
                    )
                })
            }),
            layout_environments: BTreeMap::new(),
            current_wire_types: BTreeMap::new(),
            current_scope_values: Vec::new(),
            current_referenced_wires: BTreeSet::new(),
            current_anonymous_lets: BTreeSet::new(),
            current_uses_hash_model: false,
            current_witnesses: Vec::new(),
            current_runs_record: false,
            current_witness_bindings: Vec::new(),
            current_value_expressions: BTreeMap::new(),
            scope_proofs: BTreeMap::new(),
            requires_hash_model: graph.scopes().values().any(|scope| {
                scope.nodes().iter().any(|node| matches!(node.kind(), NodeKind::HashSample { .. }))
            }),
        }
    }

    fn emit(mut self) -> Result<LeanArtifact, ExportError> {
        let mut scope_names = BTreeSet::new();
        for name in self.scopes.values() {
            if !scope_names.insert(name) {
                return Err(ExportError::ScopeNameCollision(name.clone()));
            }
        }
        let mut root_env = self.validated.bindings.clone();
        root_env.loop_indices.clear();
        self.collect_layout_environments(&FrozenGraphScopeId::Root, root_env);
        for (index, layout) in self.options.backend_layouts.iter().enumerate() {
            if self.options.backend_layouts[..index].iter().any(|other| {
                other.modulus == layout.modulus &&
                    other.ring_dimension == layout.ring_dimension &&
                    other != layout
            }) {
                return Err(ExportError::BackendLayout("conflicting backend ring layouts".into()));
            }
        }
        self.source.push_str(&format!(
            "import MxxIR\nimport {}\n\nset_option maxRecDepth 16384\nset_option maxHeartbeats 2000000\n\nnamespace {}\n\n",
            self.options.runtime_import, self.options.namespace
        ));
        self.emit_params()?;
        // Lean declarations are not mutually recursive.  Emit every structural/named child
        // before the scope that calls it, using the frozen child links rather than relying on
        // lexical ordering of scope names.
        let mut emitted = BTreeSet::new();
        let scope_ids = self.graph.scopes().keys().cloned().collect::<Vec<_>>();
        for scope_id in scope_ids {
            self.emit_scope_postorder(&scope_id, &mut emitted)?;
        }
        self.source.push_str("\nend ");
        self.source.push_str(&self.options.namespace);
        self.source.push('\n');
        let spec_hash = crate::encoding::spec_hash(self.graph, &self.validated.bindings)
            .map_err(|error| ExportError::Encoding(error.to_string()))?;
        let mut hasher = Sha256::new();
        hasher.update(self.source.as_bytes());
        hasher.update(spec_hash.0);
        let digest = hasher.finalize();
        let mut digest_array = [0u8; 32];
        digest_array.copy_from_slice(&digest);
        let root_scope = self.graph.scope(&FrozenGraphScopeId::Root).expect("frozen root");
        let root_types = &self.validated.scopes[&FrozenGraphScopeId::Root].wire_types;
        let input_wires = scope_input_wires(root_scope);
        let output_wires = root_scope.outputs();
        let input_types =
            input_wires.iter().map(|w| self.lean_type(&root_types[w])).collect::<Vec<_>>();
        let output_types =
            output_wires.iter().map(|w| self.lean_type(&root_types[w])).collect::<Vec<_>>();
        let inputs = input_wires
            .iter()
            .enumerate()
            .map(|(index, wire)| {
                let NodeKind::Input { name, .. } =
                    root_scope.node(wire.node).expect("input node").kind()
                else {
                    unreachable!("root inputs are discovered from Input nodes")
                };
                (
                    name.clone(),
                    BoundaryValue {
                        wire: *wire,
                        wire_type: root_types[wire].clone(),
                        lean_type: input_types[index].clone(),
                        tuple_index: index,
                        projection: tuple_projection("inputs", index, input_wires.len()),
                    },
                )
            })
            .collect();
        let output_positions =
            output_wires.iter().enumerate().map(|(i, w)| (*w, i)).collect::<BTreeMap<_, _>>();
        let outputs = self
            .graph
            .outputs()
            .iter()
            .map(|(name, output)| {
                let index = output_positions[&output.value];
                (
                    name.clone(),
                    BoundaryValue {
                        wire: output.value,
                        wire_type: root_types[&output.value].clone(),
                        lean_type: output_types[index].clone(),
                        tuple_index: index,
                        projection: tuple_projection("outputs", index, output_wires.len()),
                    },
                )
            })
            .collect();
        let root = RootBoundary {
            relation: format!(
                "{}.{}",
                self.options.namespace,
                self.scopes[&FrozenGraphScopeId::Root]
            ),
            input_type: tuple_type(&input_types),
            output_type: tuple_type(&output_types),
            input_count: input_wires.len(),
            output_count: output_wires.len(),
            inputs,
            outputs,
            requires_backend: self.requires_backend,
            requires_hash_model: self.requires_hash_model,
            parameter_type: format!("{}.Params", self.options.namespace),
            parameters: if self.params.is_empty() {
                BTreeMap::from([(
                    "unit".into(),
                    ParameterField { lean_type: "Unit".into(), root_value: Some("()".into()) },
                )])
            } else {
                self.params
                    .iter()
                    .map(|name| {
                        let real = self.graph.parameters().iter().any(|parameter| {
                            parameter.name == *name &&
                                parameter.kind == crate::graph::CompileParameterKind::Real
                        });
                        let root_value = if real {
                            self.validated.bindings.reals.get(name).map(|value| {
                                format!("({} / {} : Rat)", value.numerator(), value.denominator())
                            })
                        } else {
                            self.validated
                                .bindings
                                .integers
                                .get(name)
                                .map(|value| format!("({value} : Int)"))
                        };
                        (
                            name.clone(),
                            ParameterField {
                                lean_type: if real { "Rat" } else { "Int" }.into(),
                                root_value,
                            },
                        )
                    })
                    .collect()
            },
        };
        Ok(LeanArtifact {
            backend_layouts: self.options.backend_layouts.clone(),
            module_name: self.options.module_name.clone(),
            source: self.source,
            source_map: self.source_map,
            root,
            scope_relations: self
                .scopes
                .iter()
                .map(|(scope, name)| (scope.clone(), format!("{}.{name}", self.options.namespace)))
                .collect(),
            scope_proofs: self.scope_proofs,
            digest: digest_array,
            spec_hash,
            static_node_visits: self.static_node_visits,
        })
    }

    fn emit_scope_postorder(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        emitted: &mut BTreeSet<FrozenGraphScopeId>,
    ) -> Result<(), ExportError> {
        if emitted.contains(scope_id) {
            return Ok(());
        }
        let children = self
            .graph
            .scope(scope_id)
            .expect("scope key came from graph")
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(position, _)| {
                self.graph.child_scope_id(scope_id, NodeId(position as u64))
            })
            .collect::<Vec<_>>();
        for child in children {
            self.emit_scope_postorder(&child, emitted)?;
        }
        self.emit_scope(scope_id)?;
        emitted.insert(scope_id.clone());
        Ok(())
    }

    fn emit_params(&mut self) -> Result<(), ExportError> {
        // Quote every field and its uses, so Lean keywords remain ordinary IR parameter names.
        // The identifier validation excludes delimiters that could terminate the quoted name.
        self.source.push_str("structure Params where\n");
        for parameter in self.graph.parameters() {
            valid_identifier(&parameter.name)?;
            let ty = match parameter.kind {
                crate::graph::CompileParameterKind::Integer => "Int",
                crate::graph::CompileParameterKind::Real => "Rat",
            };
            self.source.push_str(&format!("  «{}» : {}\n", parameter.name, ty));
        }
        for name in self
            .params
            .iter()
            .filter(|name| !self.graph.parameters().iter().any(|p| &p.name == *name))
        {
            valid_identifier(name)?;
            self.source.push_str(&format!("  «{}» : Int\n", name));
        }
        if self.params.is_empty() {
            self.source.push_str("  unit : Unit\n");
        }
        self.source.push('\n');
        Ok(())
    }

    fn emit_scope(&mut self, scope_id: &FrozenGraphScopeId) -> Result<(), ExportError> {
        self.indent = 1;
        self.current_witnesses.clear();
        self.current_witness_bindings.clear();
        self.current_value_expressions.clear();
        let scope = self.graph.scope(scope_id).expect("scope key came from graph");
        self.current_runs_record = matches!(scope_id, FrozenGraphScopeId::Root) ||
            scope.nodes().iter().any(|node| {
                matches!(node.kind(), NodeKind::ParallelLoop(_) | NodeKind::SequentialLoop(_))
            });
        let validated = self.validated.scopes.get(scope_id).expect("validated scope");
        let inputs = scope_input_wires(scope);
        if !matches!(scope_id, FrozenGraphScopeId::Root) {
            for (position, node) in scope.nodes().iter().enumerate() {
                if matches!(node.kind(), NodeKind::Input { .. }) {
                    let wire =
                        WireRef { node: NodeId(position as u64), port: crate::types::Port(0) };
                    if !inputs.contains(&wire) {
                        return self.unsupported(
                            scope_id,
                            NodeId(position as u64),
                            node.kind(),
                            "child Input is not declared in the formal scope inputs",
                        );
                    }
                }
            }
        }
        let input_types = inputs
            .iter()
            .map(|wire| self.lean_type(&validated.wire_types[wire]))
            .collect::<Vec<_>>();
        let output_types = scope
            .outputs()
            .iter()
            .map(|wire| self.lean_type(&validated.wire_types[wire]))
            .collect::<Vec<_>>();
        let input_ty = tuple_type(&input_types);
        let output_ty = tuple_type(&output_types);
        self.current_wire_types = validated
            .wire_types
            .iter()
            .map(|(wire, ty)| (wire_name(*wire), self.lean_type(ty)))
            .collect();
        // Preserve every computation, including retained dead roots, but name a pure result
        // only when a node consumes it or the scope returns it. Anonymous lets keep their RHS.
        self.current_referenced_wires = scope
            .outputs()
            .iter()
            .copied()
            .chain(
                scope.nodes().iter().flat_map(|node| scope.arguments(node).into_iter().flatten()),
            )
            .map(wire_name)
            .collect();
        self.current_anonymous_lets.clear();
        self.current_scope_values.clear();
        self.current_uses_hash_model = false;
        let declaration_start = self.source.len();
        // Keep the configuration argument part of the generated relation even for a closed graph;
        // the underscore binding suppresses Lean's unused-binder warning without changing its
        // semantics, and parameterized expressions still refer to `params` directly below.
        self.source.push_str("  let _params := params\n");
        if self.requires_backend {
            self.source.push_str("  let _backend := backend\n");
        }
        let mut env = LexicalEnv {
            vars: self
                .params
                .iter()
                .map(|name| (name.clone(), format!("params.«{name}»")))
                .collect(),
            loop_indices: scope_loop_slots(self.graph, scope_id)
                .into_iter()
                .map(|slot| (slot, format!("(Int.ofNat i_{slot})")))
                .collect(),
            loop_index_nats: scope_loop_slots(self.graph, scope_id)
                .into_iter()
                .map(|slot| (slot, format!("i_{slot}")))
                .collect(),
            missing_loop_index: Cell::new(None),
            referenced_loop_names: Rc::default(),
        };
        for (pos, wire) in inputs.iter().enumerate() {
            let value = tuple_projection("inputs", pos, inputs.len());
            self.let_output(&wire_name(*wire), &value);
        }
        let mut existentials = Vec::<(String, String)>::new();
        let mut relations = Vec::<String>::new();
        let mut node_constraints = BTreeMap::new();
        for (position, node) in scope.nodes().iter().enumerate() {
            self.static_node_visits += 1;
            let node_id = NodeId(position as u64);
            let arguments = scope
                .arguments(node)
                .ok_or(ExportError::MissingArguments { scope: scope_id.clone(), node: node_id })?;
            let first_constraint = relations.len();
            self.emit_node(
                scope_id,
                scope,
                node_id,
                node.kind(),
                &arguments,
                &mut env,
                &mut existentials,
                &mut relations,
            )?;
            node_constraints.insert(node_id, first_constraint..relations.len());
            if let Some(slot) = env.take_missing_loop_index() {
                return Err(ExportError::MissingLoopIndex(slot));
            }
        }
        let node_facts = node_constraints
            .iter()
            .map(|(node, range)| {
                (
                    *node,
                    range
                        .clone()
                        .enumerate()
                        .map(|(local, _)| {
                            format!(
                                "{}.{}.node_{}.fact_{local}",
                                self.options.namespace, self.scopes[scope_id], node.0
                            )
                        })
                        .collect(),
                )
            })
            .collect();
        self.scope_proofs.insert(
            scope_id.clone(),
            ScopeProofShape {
                witness_record: Some({
                    format!("{}.{}.Witness", self.options.namespace, self.scopes[scope_id])
                }),
                body_relation: Some(format!(
                    "{}.{}.body",
                    self.options.namespace, self.scopes[scope_id]
                )),
                runs_iff_body: format!(
                    "{}.{}.runs_iff_body",
                    self.options.namespace, self.scopes[scope_id]
                ),
                value_expressions: self.current_value_expressions.clone(),
                witnesses: self.current_witnesses.clone(),
                node_constraints: node_constraints.clone(),
                node_facts,
                output_fact: Some({
                    format!("{}.{}.output_fact", self.options.namespace, self.scopes[scope_id])
                }),
                constraint_count: relations.len(),
            },
        );
        let output =
            tuple_expr(&scope.outputs().iter().map(|wire| wire_name(*wire)).collect::<Vec<_>>());
        let conclusion = format!("outputs = {}", output);
        // Keep both the existential telescope and each conjunction fragment small
        // during elaboration. Reducible suffix calls preserve the exact right-associated
        // And proposition and original witness order by definitional equality.
        let mut constraint_call = conclusion.clone();
        let mut constraint_definition = String::new();
        // Include the output equation in the final fragment even for a closed pure scope.
        let fragment_count = relations.len().max(1).div_ceil(32);
        for part in (0..fragment_count).rev() {
            let start = (part * 32).min(relations.len());
            let end = ((part + 1) * 32).min(relations.len());
            let body = if start == end {
                constraint_call
            } else {
                format!("{} ∧ {}", relations[start..end].join(" ∧ "), constraint_call)
            };
            let (definition, call) = self.scope_constraints(scope_id, &body, &output_ty, part);
            constraint_definition.push_str(&definition);
            constraint_call = call;
        }
        self.source.push_str(&format!("{}{constraint_call}\n\n", "  ".repeat(self.indent)));
        let loop_binders =
            loop_parameters(self.graph, scope_id, &env.referenced_loop_names.borrow());
        // Scope arity and positional arguments are unchanged; only truly unused names disappear.
        let final_header = format!(
            "def {} {}{}(params : Params){} ({} : {}) (outputs : {}) : Prop :=\n",
            self.scopes[scope_id],
            if self.requires_backend { "(backend : MxxRuntime.BackendContext) " } else { "" },
            if !self.requires_hash_model {
                ""
            } else if self.current_uses_hash_model {
                "(hashModel : MxxRuntime.HashModel) "
            } else {
                "(_ : MxxRuntime.HashModel) "
            },
            loop_binders,
            if inputs.is_empty() { "_" } else { "inputs" },
            input_ty,
            output_ty
        );
        // Large records need projections and recursors, not size or constructor
        // injectivity theorems; generating the latter dominates Lean elaboration.
        let record = {
            let fields = self
                .current_witnesses
                .iter()
                .map(|(name, ty)| format!("  {name} : {ty}\n"))
                .collect::<String>();
            format!(
                "set_option genInjectivity false in\nset_option genSizeOf false in\nstructure {}.Witness where\n{fields}\n",
                self.scopes[scope_id]
            )
        };
        {
            let name = &self.scopes[scope_id];
            let body_header =
                final_header.replacen(&format!("def {name} "), &format!("abbrev {name}.body "), 1);
            let witness_name =
                if self.current_witnesses.is_empty() { "_witness" } else { "witness" };
            let body_header = format!(
                "{}({witness_name} : {name}.Witness) : Prop :=\n",
                body_header.strip_suffix(": Prop :=\n").expect("scope header")
            );
            // Reuse the single emitted body. Binder spans come directly from
            // bind_existential, so reconstruction never parses generated Lean text.
            let legacy_body = if self.current_runs_record {
                None
            } else {
                let mut body = String::new();
                let mut cursor = declaration_start;
                for (depth, ((field, ty), span)) in
                    self.current_witnesses.iter().zip(&self.current_witness_bindings).enumerate()
                {
                    for line in self.source[cursor..span.start].split_inclusive('\n') {
                        body.push_str(&"  ".repeat(depth));
                        body.push_str(line);
                    }
                    body.push_str(&format!("{}∃ ({field} : {ty}),\n", "  ".repeat(depth + 1)));
                    cursor = span.end;
                }
                for line in self.source[cursor..].split_inclusive('\n') {
                    body.push_str(&"  ".repeat(self.current_witnesses.len()));
                    body.push_str(line);
                }
                Some(body)
            };
            self.source
                .insert_str(declaration_start, &(record + &constraint_definition + &body_header));
            let mut wrapper_header = final_header
                .replace("(_ : MxxRuntime.HashModel)", "(hashModel : MxxRuntime.HashModel)");
            let backend = if self.requires_backend { "backend " } else { "" };
            let hash = if self.requires_hash_model { "hashModel " } else { "" };
            let input = if inputs.is_empty() { "()" } else { "inputs" };
            let loop_names = scope_loop_slots(self.graph, scope_id)
                .into_iter()
                .map(|slot| format!("i_{slot}"))
                .collect::<Vec<_>>();
            if !loop_binders.is_empty() {
                let named_binders =
                    loop_names.iter().map(|name| format!(" ({name} : Nat)")).collect::<String>();
                wrapper_header = wrapper_header.replacen(&loop_binders, &named_binders, 1);
            }
            let loop_arguments =
                loop_names.iter().map(|name| format!("{name} ")).collect::<String>();
            if let Some(body) = legacy_body {
                self.source.push_str(&final_header);
                self.source.push_str(&body);
            } else {
                self.source.push_str(&format!("{wrapper_header}  ∃ witness : {name}.Witness,\n    {name}.body {backend}{hash}params {loop_arguments}{input} outputs witness\n\n"));
            }
            let equivalence_header = wrapper_header.replacen(
                &format!("def {name} "),
                &format!("theorem {name}.runs_iff_body "),
                1,
            );
            let equivalence_header =
                equivalence_header.strip_suffix(": Prop :=\n").expect("scope header");
            self.source.push_str(&format!("{equivalence_header}: {name} {backend}{hash}params {loop_arguments}{input} outputs ↔ (∃ witness : {name}.Witness, {name}.body {backend}{hash}params {loop_arguments}{input} outputs witness) := by\n"));
            if self.current_runs_record {
                self.source.push_str("  exact Iff.rfl\n\n");
            } else {
                let fields = self
                    .current_witnesses
                    .iter()
                    .map(|(name, _)| name.as_str())
                    .collect::<Vec<_>>();
                let names = fields.join(", ");
                self.source.push_str("  constructor\n  · intro h\n");
                if !fields.is_empty() {
                    self.source.push_str(&format!("    obtain ⟨{names}, h⟩ := h\n"));
                }
                self.source.push_str(&format!("    exact ⟨⟨{names}⟩, h⟩\n  · rintro ⟨witness, h⟩\n    cases witness with\n    | mk {names_space} =>\n", names_space=fields.join(" ")));
                if fields.is_empty() {
                    self.source.push_str("      exact h\n\n");
                } else {
                    self.source.push_str(&format!("      exact ⟨{names}, h⟩\n\n"));
                }
            }
            // Keep conjunction representation details inside core-generated, Lean-checked
            // proof accessors. Lean infers their exact propositions from the body.
            let accessor_context = format!(
                "{}{}{{params : Params}}{} {{inputs : {input_ty}}} {{outputs : {output_ty}}} {{witness : {name}.Witness}}",
                if self.requires_backend { "{backend : MxxRuntime.BackendContext} " } else { "" },
                if self.requires_hash_model { "{hashModel : MxxRuntime.HashModel} " } else { "" },
                loop_names.iter().map(|name| format!(" {{{name} : Nat}}")).collect::<String>(),
            );
            let body_call =
                format!("{name}.body {backend}{hash}params {loop_arguments}inputs outputs witness");
            let accessor_arguments = format!(
                "{}{}(params := params) {}(inputs := inputs) (outputs := outputs) (witness := witness)",
                if self.requires_backend { "(backend := backend) " } else { "" },
                if self.requires_hash_model { "(hashModel := hashModel) " } else { "" },
                loop_names.iter().map(|name| format!("({name} := {name}) ")).collect::<String>(),
            );
            // Each suffix accessor consumes one conjunction link, avoiding quadratic
            // projection strings and repeated copies of the body's let telescope.
            for position in 1..=relations.len() {
                let previous = if position == 1 {
                    "facts".to_owned()
                } else {
                    format!("({name}.facts_tail_{} {accessor_arguments} facts)", position - 1)
                };
                self.source.push_str(&format!(
                    "private def {name}.facts_tail_{position} {accessor_context} (facts : {body_call}) :=\n  {previous}.2\n\n"
                ));
            }
            for (node, range) in &node_constraints {
                for (local, position) in range.clone().enumerate() {
                    let suffix = if position == 0 {
                        "facts".to_owned()
                    } else {
                        format!("({name}.facts_tail_{position} {accessor_arguments} facts)")
                    };
                    self.source.push_str(&format!(
                        "def {name}.node_{}.fact_{local} {accessor_context} (facts : {body_call}) :=\n  {suffix}.1\n\n", node.0,
                    ));
                }
            }
            let suffix = if relations.is_empty() {
                "facts".to_owned()
            } else {
                format!("{name}.facts_tail_{} {accessor_arguments} facts", relations.len())
            };
            self.source.push_str(&format!(
                "def {name}.output_fact {accessor_context} (facts : {body_call}) :=\n  {suffix}\n\n",
            ));
        }
        Ok(())
    }

    fn scope_constraints(
        &self,
        scope_id: &FrozenGraphScopeId,
        body: &str,
        output_ty: &str,
        part: usize,
    ) -> (String, String) {
        // Quoted parameter fields and string payloads are not local variables. In
        // particular params.«backend» must not introduce a free backend argument.
        let mut literal_end = None;
        let mut escaped = false;
        let identifiers = body
            .chars()
            .map(|character| {
                if let Some(end) = literal_end {
                    if escaped {
                        escaped = false;
                    } else if end == '"' && character == '\\' {
                        escaped = true;
                    } else if character == end {
                        literal_end = None;
                    }
                    ' '
                } else if character == '«' || character == '"' {
                    literal_end = Some(if character == '«' { '»' } else { '"' });
                    ' '
                } else {
                    character
                }
            })
            .collect::<String>();
        let referenced = identifiers
            .split(|character: char| !character.is_ascii_alphanumeric() && character != '_')
            .collect::<BTreeSet<_>>();
        let mut constraint_arguments = Vec::new();
        if referenced.contains("backend") {
            constraint_arguments
                .push(("backend".to_owned(), "MxxRuntime.BackendContext".to_owned()));
        }
        if referenced.contains("hashModel") {
            constraint_arguments.push(("hashModel".to_owned(), "MxxRuntime.HashModel".to_owned()));
        }
        if referenced.contains("params") {
            constraint_arguments.push(("params".to_owned(), "Params".to_owned()));
        }
        for slot in scope_loop_slots(self.graph, scope_id) {
            let name = format!("i_{slot}");
            if referenced.contains(name.as_str()) {
                constraint_arguments.push((name, "Nat".to_owned()));
            }
        }
        constraint_arguments.extend(
            self.current_scope_values
                .iter()
                .filter(|(name, _)| referenced.contains(name.as_str()))
                .cloned(),
        );
        constraint_arguments.push(("outputs".to_owned(), output_ty.to_owned()));
        let constraint_name = format!("{}.constraints_{part}", self.scopes[scope_id]);
        let constraint_header = constraint_arguments
            .iter()
            .map(|(name, ty)| format!("({name} : {ty})"))
            .collect::<Vec<_>>()
            .join(" ");
        let constraint_call = constraint_arguments
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let constraint_definition = format!(
            "abbrev {constraint_name} {constraint_header} : Prop :=\n  {}\n\n",
            indent_continuation(body, 1)
        );
        (constraint_definition, format!("{constraint_name} {constraint_call}"))
    }

    fn emit_node(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        scope: &GraphScope,
        node_id: NodeId,
        kind: &NodeKind,
        args: &[WireRef],
        env: &mut LexicalEnv,
        existentials: &mut Vec<(String, String)>,
        relations: &mut Vec<String>,
    ) -> Result<(), ExportError> {
        let output =
            |port: u32| wire_name(WireRef { node: node_id, port: crate::types::Port(port) });
        let arg = |i: usize| {
            args.get(i)
                .map(|w| wire_name(*w))
                .ok_or(ExportError::MissingArguments { scope: scope_id.clone(), node: node_id })
        };
        match kind {
            NodeKind::Input { .. } => {}
            NodeKind::ConstantInt(value) => self.let_output(&output(0), &value.to_string()),
            NodeKind::ConstantBool(value) => {
                self.let_output(&output(0), if *value { "true" } else { "false" })
            }
            NodeKind::ConstantMatrix {
                value: ConstantMatrix::Gadget { base, small: false },
                ..
            } => {
                let wire = WireRef { node: node_id, port: crate::types::Port(0) };
                let layout = self.require_layout(scope_id, wire, Some(base), None)?;
                let crate::types::ConcreteWireType::Matrix(matrix) =
                    &self.validated.scopes[scope_id].wire_types[&wire]
                else {
                    unreachable!()
                };
                if matrix.rows.checked_mul(layout.regular_digits) != Some(matrix.columns) {
                    return Err(ExportError::BackendLayout(
                        "public gadget width disagrees with backend layout".into(),
                    ));
                }
                let ty = self.output_type(scope, node_id, 0);
                append_expression_guards(base, env, relations);
                self.bind_existential(&output(0), &ty);
                relations.push(format!(
                    "MxxRuntime.gadgetMatrixRuns backend {} {} {}",
                    env.expr(base),
                    layout.regular_digits,
                    output(0)
                ));
            }
            NodeKind::ConstantMatrix { value, .. } => {
                let ty = self.output_type(scope, node_id, 0);
                let wire = WireRef { node: node_id, port: crate::types::Port(0) };
                let crate::types::ConcreteWireType::Matrix(matrix) =
                    &self.validated.scopes[scope_id].wire_types[&wire]
                else {
                    return self.unsupported(
                        scope_id,
                        node_id,
                        kind,
                        "constant output must be a matrix",
                    );
                };
                let term = match value {
                    ConstantMatrix::Zero => format!("(0 : {ty})"),
                    ConstantMatrix::Identity if matrix.rows == matrix.columns => {
                        format!("(1 : {ty})")
                    }
                    ConstantMatrix::UnitRow { index } if matrix.rows == 1 => {
                        append_expression_guards(index, env, relations);
                        relations.push(format!("0 ≤ ({})", env.expr(index)));
                        relations.push(format!("({}) < {}", env.expr(index), matrix.columns));
                        format!("(MxxRuntime.unitRow ({}) : {ty})", env.expr(index))
                    }
                    ConstantMatrix::UnitColumn { index } if matrix.columns == 1 => {
                        append_expression_guards(index, env, relations);
                        relations.push(format!("0 ≤ ({})", env.expr(index)));
                        relations.push(format!("({}) < {}", env.expr(index), matrix.rows));
                        format!("(MxxRuntime.unitColumn ({}) : {ty})", env.expr(index))
                    }
                    ConstantMatrix::PowerOfBase { base, exponent }
                        if matrix.rows == 1 && matrix.columns == 1 =>
                    {
                        append_expression_guards(base, env, relations);
                        append_expression_guards(exponent, env, relations);
                        relations.push(format!("0 ≤ ({})", env.expr(exponent)));
                        relations.push(format!("({}) ≤ 4294967295", env.expr(exponent)));
                        format!(
                            "({} (({}) ^ ({}).toNat) : {ty})",
                            self.options.primitives.lift_integer,
                            env.expr(base),
                            env.expr(exponent)
                        )
                    }
                    ConstantMatrix::Rotation { exponent }
                        if matrix.rows == 1 && matrix.columns == 1 =>
                    {
                        append_expression_guards(exponent, env, relations);
                        relations.push(format!("0 ≤ ({})", env.expr(exponent)));
                        relations.push(format!(
                            "({}) < {}",
                            env.expr(exponent),
                            matrix.ring_dimension
                        ));
                        format!("(MxxRuntime.rotationPolynomial ({}) : {ty})", env.expr(exponent))
                    }
                    ConstantMatrix::Polynomial { coefficients }
                        if matrix.rows == 1 && matrix.columns == 1 =>
                    {
                        for coefficient in coefficients {
                            append_expression_guards(coefficient, env, relations);
                        }
                        let coefficients = coefficients
                            .iter()
                            .map(|c| format!("({})", env.expr(c)))
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!(
                            "({} [{coefficients}] : {ty})",
                            self.options.primitives.matrix_polynomial
                        )
                    }
                    _ => {
                        return self.unsupported(
                            scope_id,
                            node_id,
                            kind,
                            "unsupported constant payload or shape",
                        )
                    }
                };
                self.let_output(&output(0), &term);
            }
            NodeKind::EvaluateInt(expr) => {
                append_expression_guards(expr, env, relations);
                self.let_output(&output(0), &env.expr(expr));
            }
            NodeKind::ConstantReal(expr) => {
                let value = env.real_expr(expr);
                self.let_output(&output(0), &value);
            }
            NodeKind::IntBinary(op) => {
                let a = arg(0)?;
                let b = arg(1)?;
                let term = match op {
                    IntBinaryOp::Add => format!("({a} + {b})"),
                    IntBinaryOp::Subtract => format!("({a} - {b})"),
                    IntBinaryOp::Multiply => format!("({a} * {b})"),
                    // Runtime integer arithmetic is Euclidean division.  The only partial
                    // condition is a non-zero divisor; exact compile-time `IntExpr::Div`
                    // remains separate and is guarded by its own expression semantics.
                    IntBinaryOp::Divide => {
                        relations.push(format!("{} ≠ 0", b));
                        format!("({a} / {b})")
                    }
                    IntBinaryOp::Remainder => {
                        relations.push(format!("{} ≠ 0", b));
                        format!("({a} % {b})")
                    }
                };
                self.let_output(&output(0), &term);
            }
            NodeKind::IntCompare(op) => {
                let a = arg(0)?;
                let b = arg(1)?;
                let term = match op {
                    IntCompareOp::Equal => format!("decide ({a} = {b})"),
                    IntCompareOp::Less => format!("decide ({a} < {b})"),
                    IntCompareOp::LessEqual => format!("decide ({a} ≤ {b})"),
                };
                self.let_output(&output(0), &term);
            }
            NodeKind::BoolToInt => {
                let a = arg(0)?;
                self.let_output(&output(0), &format!("if {a} then 1 else 0"));
            }
            NodeKind::BitExtract { bit } => {
                let a = arg(0)?;
                append_expression_guards(bit, env, relations);
                relations.push(format!("0 ≤ ({})", env.expr(bit)));
                self.let_output(
                    &output(0),
                    &format!("decide (({a} / (2 ^ (Int.toNat ({})))) % 2 = 1)", env.expr(bit)),
                );
            }
            NodeKind::MatrixBinary(op) => {
                let a = arg(0)?;
                let b = arg(1)?;
                let f = match op {
                    MatrixBinaryOp::Add => &self.options.primitives.matrix_add,
                    MatrixBinaryOp::Subtract => &self.options.primitives.matrix_sub,
                    MatrixBinaryOp::Multiply => self.matrix_multiply_name(scope_id, args),
                };
                self.let_output(&output(0), &format!("{f} {a} {b}"));
            }
            NodeKind::MatrixMulSmallRhs => {
                let a = arg(0)?;
                let b = arg(1)?;
                // The validated Preimage wire is an ordinary matrix at this operation boundary;
                // its coefficient bound is a construction invariant, while multiplication itself
                // has the same exact matrix semantics as the ordinary operator.
                self.let_output(
                    &output(0),
                    &format!("{} {} {}", self.matrix_multiply_name(scope_id, args), a, b),
                );
            }
            NodeKind::MatrixNegate => {
                let a = arg(0)?;
                self.let_output(
                    &output(0),
                    &format!("{} {}", self.options.primitives.matrix_neg, a),
                );
            }
            NodeKind::MatrixScale { scalar } => {
                let a = arg(0)?;
                append_expression_guards(scalar, env, relations);
                self.let_output(
                    &output(0),
                    &format!("{} {} {}", self.options.primitives.matrix_scale, env.expr(scalar), a),
                );
            }
            NodeKind::Transpose => {
                let a = arg(0)?;
                self.let_output(
                    &output(0),
                    &format!("{} {}", self.options.primitives.transpose, a),
                );
            }
            NodeKind::GadgetTrapdoor { base, .. } => {
                let wire = WireRef { node: node_id, port: crate::types::Port(0) };
                let layout = self.require_layout(scope_id, wire, Some(base), None)?;
                let crate::types::ConcreteWireType::Trapdoor {
                    matrix,
                    sigma,
                    preimage_max_coefficient_bound,
                    ..
                } = &self.validated.scopes[scope_id].wire_types[&wire]
                else {
                    unreachable!()
                };
                if matrix.rows.checked_mul(layout.regular_digits) != Some(matrix.columns) {
                    return Err(ExportError::BackendLayout(
                        "public gadget trapdoor width disagrees with backend layout".into(),
                    ));
                }
                let sigma = env.real_expr(sigma);
                let cutoff = preimage_max_coefficient_bound.to_string();
                append_expression_guards(base, env, relations);
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "{} backend {} {} {} {} {}",
                    self.options.primitives.gadget_trapdoor,
                    sigma,
                    env.expr(base),
                    layout.regular_digits,
                    cutoff,
                    output(0)
                ));
            }
            NodeKind::GadgetDecompose { base, digit_count, small } => {
                if *small {
                    return self.unsupported(
                        scope_id,
                        node_id,
                        kind,
                        "small gadget decomposition needs a runtime relation with its centered-digit semantics",
                    );
                }
                let a = arg(0)?;
                self.require_layout(scope_id, args[0], Some(base), Some(digit_count))?;
                append_expression_guards(base, env, relations);
                append_expression_guards(digit_count, env, relations);
                let decomposition = output(0);
                let ty = self.output_type(scope, node_id, 0);
                self.bind_existential(&decomposition, &ty);
                existentials.push((decomposition.clone(), ty));
                relations.push(format!(
                    "{} backend {} {} {} {}",
                    self.options.primitives.gadget_decompose,
                    env.expr(base),
                    env.expr(digit_count),
                    a,
                    decomposition
                ));
            }
            NodeKind::TrapdoorPublic => {
                let a = arg(0)?;
                self.let_output(
                    &output(0),
                    &format!("{} {a}", self.options.primitives.trapdoor_public),
                );
            }
            NodeKind::TrapdoorSample {
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
                ..
            } => {
                self.require_layout(
                    scope_id,
                    WireRef { node: node_id, port: crate::types::Port(1) },
                    Some(gadget_base),
                    Some(digit_count),
                )?;
                append_expression_guards(gadget_base, env, relations);
                append_expression_guards(digit_count, env, relations);
                append_expression_guards(preimage_max_coefficient_bound, env, relations);
                let p = output(0);
                let t = output(1);
                let ty0 = self.output_type(scope, node_id, 0);
                let ty1 = self.output_type(scope, node_id, 1);
                self.bind_existential(&t, &ty1);
                self.bind_existential(&p, &ty0);
                existentials.push((t.clone(), ty1));
                existentials.push((p.clone(), ty0));
                relations.push(format!(
                    "{} backend {} {} {} {} {} {}",
                    self.options.primitives.trapdoor_sample,
                    env.real_expr(sigma),
                    env.expr(gadget_base),
                    env.expr(digit_count),
                    env.expr(preimage_max_coefficient_bound),
                    p,
                    t
                ));
            }
            NodeKind::PreimageSample { max_coefficient_bound, .. } => {
                self.require_layout(scope_id, args[1], None, None)?;
                append_expression_guards(max_coefficient_bound, env, relations);
                let k = output(0);
                let ty = self.output_type(scope, node_id, 0);
                self.bind_existential(&k, &ty);
                existentials.push((k.clone(), ty));
                let public = arg(0)?;
                let trapdoor = arg(1)?;
                let target = arg(2)?;
                relations.push(format!("0 ≤ ({})", env.expr(max_coefficient_bound)));
                relations.push(format!(
                    "{} backend {} {} {} (Int.toNat ({})) {}",
                    self.options.primitives.preimage_sample,
                    public,
                    trapdoor,
                    target,
                    env.expr(max_coefficient_bound),
                    k
                ));
            }
            NodeKind::UniformResidueSample { .. } => self.sample_one(
                scope,
                node_id,
                existentials,
                relations,
                &self.options.primitives.uniform_residue_sample,
                &[],
            ),
            NodeKind::UniformIntervalSample { range, .. } => {
                append_expression_guards(&range.minimum, env, relations);
                append_expression_guards(&range.maximum, env, relations);
                let lo = env.expr(&range.minimum);
                let hi = env.expr(&range.maximum);
                self.sample_one(
                    scope,
                    node_id,
                    existentials,
                    relations,
                    &self.options.primitives.uniform_interval_sample,
                    &[lo, hi],
                );
            }
            NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
                append_expression_guards(max_coefficient_bound, env, relations);
                let sigma = env.real_expr(sigma);
                let bound = env.expr(max_coefficient_bound);
                self.sample_one(
                    scope,
                    node_id,
                    existentials,
                    relations,
                    &self.options.primitives.gaussian_sample,
                    &[sigma, bound],
                );
            }
            NodeKind::HashSample { variant, tag_prefix, tag_components, .. } => {
                if *variant != crate::node::HashVariant::Plain {
                    return self.unsupported(
                        scope_id,
                        node_id,
                        kind,
                        "decomposed hash variants need their backend decomposition relation",
                    );
                }
                let key = arg(0)?;
                self.current_uses_hash_model = true;
                let components = tag_components
                    .iter()
                    .map(|component| {
                        use crate::node::HashTagComponent;
                        match component {
                            HashTagComponent::Bytes(bytes) => format!(
                                ".bytes [{}]",
                                bytes.iter().map(u8::to_string).collect::<Vec<_>>().join(", ")
                            ),
                            HashTagComponent::Integer(expression) |
                            HashTagComponent::Decimal(expression) |
                            HashTagComponent::U64Le(expression) => {
                                append_expression_guards(expression, env, relations);
                                let constructor = match component {
                                    HashTagComponent::Integer(_) => "integer",
                                    HashTagComponent::Decimal(_) => "decimal",
                                    _ => "u64Le",
                                };
                                format!(".{constructor} ({})", env.expr(expression))
                            }
                            HashTagComponent::Operand(index) => {
                                format!(".integer {}", wire_name(args[*index]))
                            }
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                let prefix = format!(
                    "[{}]",
                    tag_prefix.iter().map(u8::to_string).collect::<Vec<_>>().join(", ")
                );
                self.sample_one(
                    scope,
                    node_id,
                    existentials,
                    relations,
                    &self.options.primitives.hash_sample,
                    &["hashModel".into(), prefix, format!("[{components}]"), key],
                );
            }
            NodeKind::ModulusSwitch { modulus } |
            NodeKind::ModulusReduce { modulus } |
            NodeKind::CenteredRebase { modulus } => {
                append_expression_guards(modulus, env, relations);
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                let relation = if matches!(kind, NodeKind::ModulusSwitch { .. }) {
                    &self.options.primitives.modulus_switch
                } else if matches!(kind, NodeKind::CenteredRebase { .. }) {
                    &self.options.primitives.centered_rebase
                } else {
                    &self.options.primitives.modulus_reduce
                };
                relations.push(format!("{relation} {} {}", arg(0)?, output(0)));
            }
            NodeKind::RnsModUp { modulus, source_moduli, digit_size, normalize } => {
                append_expression_guards(modulus, env, relations);
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                let basis = source_moduli.iter().map(u64::to_string).collect::<Vec<_>>().join(", ");
                relations.push(format!(
                    "{} [{}] {} {} {} {}",
                    self.options.primitives.rns_mod_up,
                    basis,
                    digit_size,
                    normalize,
                    arg(0)?,
                    output(0)
                ));
            }
            NodeKind::RnsModDown { modulus, source_moduli, plaintext_modulus } => {
                append_expression_guards(modulus, env, relations);
                append_expression_guards(plaintext_modulus, env, relations);
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                let basis = source_moduli.iter().map(u64::to_string).collect::<Vec<_>>().join(", ");
                relations.push(format!(
                    "{} [{}] ({}) {} {}",
                    self.options.primitives.rns_mod_down,
                    basis,
                    env.expr(plaintext_modulus),
                    arg(0)?,
                    output(0)
                ));
            }
            NodeKind::CenteredExtend { modulus } => {
                append_expression_guards(modulus, env, relations);
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "{} {} {}",
                    self.options.primitives.centered_extend,
                    arg(0)?,
                    output(0)
                ));
            }
            NodeKind::BlockModSwitch { modulus, plaintext_modulus } => {
                append_expression_guards(modulus, env, relations);
                append_expression_guards(plaintext_modulus, env, relations);
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "{} ({}) {} {}",
                    self.options.primitives.block_mod_switch,
                    env.expr(plaintext_modulus),
                    arg(0)?,
                    output(0)
                ));
            }
            NodeKind::RingAutomorphism { index } => {
                append_expression_guards(index, env, relations);
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "{} ({}) {} {}",
                    self.options.primitives.ring_automorphism,
                    env.expr(index),
                    arg(0)?,
                    output(0)
                ));
            }
            NodeKind::CrtRecompose { modulus, plaintext_moduli, reconstruction_coefficients } => {
                append_expression_guards(modulus, env, relations);
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                let mut terms = Vec::new();
                for (index, (plain, coefficient)) in
                    plaintext_moduli.iter().zip(reconstruction_coefficients).enumerate()
                {
                    append_expression_guards(plain, env, relations);
                    append_expression_guards(coefficient, env, relations);
                    relations.push(format!("0 < ({})", env.expr(plain)));
                    terms.push(format!(
                        "{} ({}) ({}) {}",
                        self.options.primitives.crt_recompose,
                        env.expr(plain),
                        env.expr(coefficient),
                        arg(index)?
                    ));
                }
                relations.push(format!(
                    "{} = {}",
                    output(0),
                    terms
                        .into_iter()
                        .map(|term| format!("({term})"))
                        .collect::<Vec<_>>()
                        .join(" + ")
                ));
            }
            NodeKind::MatrixMulAccumulate { .. } => {
                return self.unsupported(
                    scope_id,
                    node_id,
                    kind,
                    "matrix fused operation requires concrete runtime relation",
                )
            }
            NodeKind::Slice { rows, columns } => {
                let input = arg(0)?;
                let input_wire = args[0];
                let Some(crate::types::ConcreteWireType::Matrix(input_ty)) =
                    self.validated.scopes[scope_id].wire_types.get(&input_wire)
                else {
                    return self.unsupported(
                        scope_id,
                        node_id,
                        kind,
                        "slice requires a matrix input",
                    );
                };
                let output_name = output(0);
                let output_ty = self.output_type(scope, node_id, 0);
                self.bind_existential(&output_name, &output_ty);
                existentials.push((output_name.clone(), output_ty));
                let (row_start, row_end) =
                    self.slice_bounds(rows.as_ref(), input_ty.rows, env, relations);
                let (column_start, column_end) =
                    self.slice_bounds(columns.as_ref(), input_ty.columns, env, relations);
                relations.push(format!(
                    "{} {} {} {} {} {}",
                    self.options.primitives.matrix_slice,
                    input,
                    row_start,
                    row_end,
                    column_start,
                    column_end,
                ));
                relations.last_mut().unwrap().push_str(&format!(" {}", output_name));
            }
            NodeKind::Concat { axis } => {
                if args.is_empty() {
                    return self.unsupported(scope_id, node_id, kind, "concat requires an input");
                }
                if args.len() == 1 {
                    self.let_output(&output(0), &wire_name(args[0]));
                } else {
                    let output_name = output(0);
                    let output_ty = self.output_type(scope, node_id, 0);
                    self.bind_existential(&output_name, &output_ty);
                    existentials.push((output_name.clone(), output_ty));
                    let mut left = wire_name(args[0]);
                    let mut left_ty = match self.validated.scopes[scope_id].wire_types.get(&args[0])
                    {
                        Some(crate::types::ConcreteWireType::Matrix(ty)) => ty.clone(),
                        _ => {
                            return self.unsupported(
                                scope_id,
                                node_id,
                                kind,
                                "concat requires matrices",
                            )
                        }
                    };
                    for (index, right_wire) in args.iter().enumerate().skip(1) {
                        let right = wire_name(*right_wire);
                        let right_ty =
                            match self.validated.scopes[scope_id].wire_types.get(right_wire) {
                                Some(crate::types::ConcreteWireType::Matrix(ty)) => ty,
                                _ => {
                                    return self.unsupported(
                                        scope_id,
                                        node_id,
                                        kind,
                                        "concat requires matrices",
                                    )
                                }
                            };
                        let is_last = index + 1 == args.len();
                        let result_name = if is_last {
                            output_name.clone()
                        } else {
                            let name = format!("w_{}_concat_{}", node_id.0, index);
                            let result_ty = match axis {
                                crate::node::ConcatAxis::Rows => crate::types::ConcreteMatrixType {
                                    rows: left_ty.rows + right_ty.rows,
                                    ..left_ty.clone()
                                },
                                crate::node::ConcatAxis::Columns => {
                                    crate::types::ConcreteMatrixType {
                                        columns: left_ty.columns + right_ty.columns,
                                        ..left_ty.clone()
                                    }
                                }
                                crate::node::ConcatAxis::Diagonal => {
                                    crate::types::ConcreteMatrixType {
                                        rows: left_ty.rows + right_ty.rows,
                                        columns: left_ty.columns + right_ty.columns,
                                        ..left_ty.clone()
                                    }
                                }
                            };
                            let result_ty =
                                self.lean_type(&crate::types::ConcreteWireType::Matrix(result_ty));
                            self.bind_existential(&name, &result_ty);
                            existentials.push((name.clone(), result_ty));
                            name
                        };
                        let relation = match axis {
                            crate::node::ConcatAxis::Rows => &self.options.primitives.concat_rows,
                            crate::node::ConcatAxis::Columns => {
                                &self.options.primitives.concat_columns
                            }
                            crate::node::ConcatAxis::Diagonal => {
                                &self.options.primitives.concat_diagonal
                            }
                        };
                        relations.push(format!("{} {} {} {}", relation, left, right, result_name));
                        left = result_name;
                        left_ty = match axis {
                            crate::node::ConcatAxis::Rows => crate::types::ConcreteMatrixType {
                                rows: left_ty.rows + right_ty.rows,
                                ..left_ty
                            },
                            crate::node::ConcatAxis::Columns => crate::types::ConcreteMatrixType {
                                columns: left_ty.columns + right_ty.columns,
                                ..left_ty
                            },
                            crate::node::ConcatAxis::Diagonal => crate::types::ConcreteMatrixType {
                                rows: left_ty.rows + right_ty.rows,
                                columns: left_ty.columns + right_ty.columns,
                                ..left_ty
                            },
                        };
                    }
                }
            }
            NodeKind::ExtractCoefficient { position, .. } => {
                append_expression_guards(position, env, relations);
                self.bind_existential(&output(0), "Int");
                relations.push(format!(
                    "{} {} {} {}",
                    self.options.primitives.extract_coefficient,
                    env.expr(position),
                    arg(0)?,
                    output(0)
                ));
            }
            NodeKind::LiftIntegerToConstantPolynomial { .. } => {
                self.let_output(
                    &output(0),
                    &format!("{} {}", self.options.primitives.lift_integer, arg(0)?),
                );
            }
            NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } => {
                append_expression_guards(coefficient_bits, env, relations);
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "{} ({}) {} {}",
                    self.options.primitives.pack_polynomial,
                    env.expr(coefficient_bits),
                    arg(0)?,
                    output(0)
                ));
            }
            NodeKind::PolynomialFromValues { evaluation, .. } => {
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "{} {} {} {}",
                    self.options.primitives.polynomial_from_values,
                    evaluation,
                    arg(0)?,
                    output(0)
                ));
            }
            NodeKind::PolynomialValues { evaluation } => {
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "{} {} {} {}",
                    self.options.primitives.polynomial_values,
                    evaluation,
                    arg(0)?,
                    output(0)
                ));
            }
            NodeKind::Tensor => {
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "MxxRuntime.tensorRuns {} {} {}",
                    arg(0)?,
                    arg(1)?,
                    output(0)
                ));
            }
            NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
                append_expression_guards(plaintext_modulus, env, relations);
                append_expression_guards(length, env, relations);
                let count = scope.node(node_id).expect("decoder node").output_types().len();
                relations.push(format!("({}) = {count}", env.expr(length)));
                for port in 0..count {
                    let name = output(port as u32);
                    let decoded =
                        if *output_bool { format!("{name}_decoded") } else { name.clone() };
                    self.bind_existential(&decoded, "Int");
                    relations.push(format!(
                        "{} ({}) ({}) {port} {} {decoded}",
                        self.options.primitives.threshold_decode,
                        env.expr(plaintext_modulus),
                        env.expr(length),
                        arg(0)?,
                    ));
                    if *output_bool {
                        self.let_output(&name, &format!("decide ({decoded} ≠ 0)"));
                    }
                }
            }
            NodeKind::IntToReal | NodeKind::RealBinary(_) | NodeKind::RealSqrt => {
                return self.unsupported(
                    scope_id,
                    node_id,
                    kind,
                    "operation is not yet in the generic primitive interface",
                )
            }
            NodeKind::FamilyPack { count } => {
                let values =
                    args.iter().map(|wire| wire_name(*wire)).collect::<Vec<_>>().join(", ");
                append_expression_guards(count, env, relations);
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "{} ({}) [{}] {}",
                    self.options.primitives.family_pack,
                    env.expr(count),
                    values,
                    output(0)
                ));
            }
            NodeKind::FamilyGetStatic { index } => {
                let family = arg(0)?;
                append_expression_guards(index, env, relations);
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "{} {} ({}) {}",
                    self.options.primitives.family_get_static,
                    family,
                    env.expr(index),
                    output(0)
                ));
            }
            NodeKind::FamilyGetDynamic => {
                let family = arg(0)?;
                let index = arg(1)?;
                let family_wire = args.first().expect("family argument");
                let family_count = match self.validated.scopes[scope_id].wire_types.get(family_wire)
                {
                    Some(crate::types::ConcreteWireType::IndexedFamily { count, .. }) => {
                        count.to_string()
                    }
                    _ => {
                        return self.unsupported(
                            scope_id,
                            node_id,
                            kind,
                            "dynamic access requires an indexed family",
                        )
                    }
                };
                relations.push(format!("0 ≤ {}", index));
                relations.push(format!("{} < {}", index, family_count));
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "{} {} {} {}",
                    self.options.primitives.family_get_dynamic,
                    family,
                    index,
                    output(0)
                ));
            }
            NodeKind::Select { count } => {
                let selector = arg(0)?;
                append_expression_guards(count, env, relations);
                let branches =
                    args.iter().skip(1).map(|wire| wire_name(*wire)).collect::<Vec<_>>().join(", ");
                relations.push(format!("0 ≤ {}", selector));
                relations.push(format!("{} < {}", selector, env.expr(count)));
                relations.push(format!("{} = {}", env.expr(count), args.len() - 1));
                self.bind_existential(&output(0), &self.output_type(scope, node_id, 0));
                relations.push(format!(
                    "{} {} [{}] {}",
                    self.options.primitives.select,
                    selector,
                    branches,
                    output(0)
                ));
            }
            NodeKind::SubgraphCall(_) => self.emit_child_call(
                scope_id,
                scope,
                node_id,
                args,
                env,
                existentials,
                relations,
                false,
            )?,
            NodeKind::ParallelLoop(_) => self.emit_child_call(
                scope_id,
                scope,
                node_id,
                args,
                env,
                existentials,
                relations,
                true,
            )?,
            NodeKind::SequentialLoop(_) => {
                self.emit_sequential(scope_id, scope, node_id, args, env, existentials, relations)?
            }
        }
        for port in 0..scope.node(node_id).map(|n| n.output_types().len()).unwrap_or(0) {
            self.source_map.entries.push(SourceMapEntry {
                scope: scope_id.clone(),
                node: Some(node_id),
                port: Some(port as u32),
                generated: if self.current_anonymous_lets.contains(&output(port as u32)) {
                    "_".into()
                } else {
                    output(port as u32)
                },
            });
        }
        Ok(())
    }

    fn let_output(&mut self, name: &str, term: &str) {
        self.current_value_expressions.insert(name.to_owned(), term.to_owned());
        let ty = &self.current_wire_types[name];
        let binder = if self.current_referenced_wires.contains(name) {
            name
        } else {
            self.current_anonymous_lets.insert(name.into());
            "_"
        };
        if binder != "_" {
            self.current_scope_values.push((name.to_owned(), ty.clone()));
        }
        self.source
            .push_str(&format!("{}let {binder} : {ty} := {term}\n", "  ".repeat(self.indent)));
    }
    fn matrix_multiply_name(&self, scope: &FrozenGraphScopeId, args: &[WireRef]) -> &str {
        use crate::types::ConcreteWireType;
        let scalar = |wire: &WireRef| match &self.validated.scopes[scope].wire_types[wire] {
            ConcreteWireType::Matrix(matrix) |
            ConcreteWireType::SmallMatrix { matrix, .. } |
            ConcreteWireType::Preimage { matrix, .. } => matrix.is_scalar(),
            _ => false,
        };
        if scalar(&args[0]) {
            &self.options.primitives.matrix_mul_scalar_left
        } else if scalar(&args[1]) {
            &self.options.primitives.matrix_mul_scalar_right
        } else {
            &self.options.primitives.matrix_mul
        }
    }
    fn collect_layout_environments(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        env: crate::expr::ParamEnv,
    ) {
        let seen = self.layout_environments.entry(scope_id.clone()).or_default();
        if seen.contains(&env) {
            return;
        }
        seen.push(env.clone());
        let scope = self.graph.scope(scope_id).expect("validated scope");
        for (position, node) in scope.nodes().iter().enumerate() {
            let Some(child_id) = self.graph.child_scope_id(scope_id, NodeId(position as u64))
            else {
                continue;
            };
            let bindings = match node.kind() {
                NodeKind::SubgraphCall(call) => &call.bindings,
                NodeKind::ParallelLoop(body) => &body.bindings,
                NodeKind::SequentialLoop(body) => &body.bindings,
                _ => continue,
            };
            let mut child = env.clone();
            // All overrides read the same parent environment. Missing loop indices and
            // loop-dependent parameters stay unresolved, never a representative lane zero.
            for (name, expression) in bindings {
                match expression.evaluate(&env) {
                    Ok(value) => {
                        child.integers.insert(name.clone(), value);
                    }
                    Err(_) => {
                        child.integers.remove(name);
                    }
                }
            }
            self.collect_layout_environments(&child_id, child);
        }
    }
    fn require_layout(
        &self,
        scope: &FrozenGraphScopeId,
        wire: WireRef,
        base: Option<&IntExpr>,
        digits: Option<&IntExpr>,
    ) -> Result<BackendLayout, ExportError> {
        use crate::types::ConcreteWireType;
        let ty = &self.validated.scopes[scope].wire_types[&wire];
        let matrix = match ty {
            ConcreteWireType::Matrix(m) |
            ConcreteWireType::SmallMatrix { matrix: m, .. } |
            ConcreteWireType::Preimage { matrix: m, .. } |
            ConcreteWireType::Trapdoor { matrix: m, .. } => m,
            _ => {
                return Err(ExportError::BackendLayout("layout operand is not a ring value".into()))
            }
        };
        let layout = self
            .options
            .backend_layouts
            .iter()
            .find(|layout| {
                layout.modulus == matrix.modulus && layout.ring_dimension == matrix.ring_dimension
            })
            .ok_or_else(|| {
                ExportError::BackendLayout(format!(
                    "missing ring ({}, {})",
                    matrix.modulus, matrix.ring_dimension
                ))
            })?;
        for env in &self.layout_environments[scope] {
            let evaluate = |expression: &IntExpr| {
                expression.evaluate(env).map_err(|error| {
                    ExportError::BackendLayout(format!("nonuniform layout payload: {error}"))
                })
            };
            if let Some(base) = base {
                if evaluate(base)? != layout.base {
                    return Err(ExportError::BackendLayout("gadget base mismatch".into()));
                }
            }
            if let Some(digits) = digits {
                if evaluate(digits)? != num_bigint::BigInt::from(layout.regular_digits) {
                    return Err(ExportError::BackendLayout("gadget digit count mismatch".into()));
                }
            }
        }
        if let ConcreteWireType::Trapdoor { gadget_base, digit_count, .. } = ty {
            if gadget_base != &layout.base || *digit_count != layout.regular_digits {
                return Err(ExportError::BackendLayout("trapdoor layout mismatch".into()));
            }
        }
        Ok(layout.clone())
    }
    fn bind_existential(&mut self, name: &str, ty: &str) {
        self.current_value_expressions.insert(
            name.to_owned(),
            if self.current_runs_record { format!("witness.{name}") } else { name.to_owned() },
        );
        self.current_scope_values.push((name.to_owned(), ty.to_owned()));
        self.current_witnesses.push((name.to_owned(), ty.to_owned()));
        let start = self.source.len();
        self.source.push_str(&format!("  let {name} := witness.{name}\n"));
        self.current_witness_bindings.push(start..self.source.len());
    }
    fn output_type(&self, scope: &GraphScope, node: NodeId, port: u32) -> String {
        let wire = WireRef { node, port: crate::types::Port(port) };
        self.validated.scopes[scope.id()]
            .wire_types
            .get(&wire)
            .map(|ty| self.lean_type(ty))
            .unwrap_or_else(|| "Unit".into())
    }
    fn slice_bounds(
        &self,
        range: Option<&crate::node::IndexRange>,
        extent: usize,
        env: &LexicalEnv,
        relations: &mut Vec<String>,
    ) -> (String, String) {
        let (start, end) = range
            .map(|range| {
                append_expression_guards(&range.start, env, relations);
                append_expression_guards(&range.end, env, relations);
                (env.expr(&range.start), env.expr(&range.end))
            })
            .unwrap_or_else(|| ("0".into(), extent.to_string()));
        relations.push(format!("0 ≤ {}", start));
        relations.push(format!("{} < {}", start, end));
        relations.push(format!("{} ≤ {}", end, extent));
        (start, end)
    }
    fn sample_one(
        &mut self,
        scope: &GraphScope,
        node: NodeId,
        existentials: &mut Vec<(String, String)>,
        relations: &mut Vec<String>,
        relation: &str,
        args: &[String],
    ) {
        let name = wire_name(WireRef { node, port: crate::types::Port(0) });
        let ty = self.output_type(scope, node, 0);
        self.bind_existential(&name, &ty);
        existentials.push((name.clone(), ty));
        let mut terms = vec![relation.to_owned()];
        terms.extend(args.iter().map(|arg| format!("({arg})")));
        terms.push(name);
        relations.push(terms.join(" "));
    }
    fn emit_child_call(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        scope: &GraphScope,
        node: NodeId,
        args: &[WireRef],
        env: &LexicalEnv,
        existentials: &mut Vec<(String, String)>,
        relations: &mut Vec<String>,
        parallel: bool,
    ) -> Result<(), ExportError> {
        let child = self
            .graph
            .child_scope_id(scope_id, node)
            .ok_or(ExportError::MissingChildScope { scope: scope_id.clone(), node })?;
        self.current_uses_hash_model |= self.requires_hash_model;
        let child_name = format!(
            "{}{}{}",
            self.scopes[&child],
            if self.requires_backend { " backend" } else { "" },
            if self.requires_hash_model { " hashModel" } else { "" }
        );
        let node_ref = scope.node(node).expect("node");
        let outputs = (0..node_ref.output_types().len())
            .map(|port| wire_name(WireRef { node, port: crate::types::Port(port as u32) }))
            .collect::<Vec<_>>();
        for (port, name) in outputs.iter().enumerate() {
            let ty = self.output_type(scope, node, port as u32);
            self.bind_existential(name, &ty);
            existentials.push((name.clone(), ty));
        }
        let (bindings, input_modes, count) = match node_ref.kind() {
            NodeKind::SubgraphCall(call) => (&call.bindings, None, None),
            NodeKind::ParallelLoop(spec) => {
                (&spec.bindings, Some(&spec.input_modes), Some(env.expr(&spec.count)))
            }
            _ => unreachable!(),
        };
        let parallel_count = if let Some(count_expression) = &count {
            let wire = WireRef { node, port: crate::types::Port(0) };
            let Some(crate::types::ConcreteWireType::IndexedFamily { count, .. }) =
                self.validated.scopes[scope_id].wire_types.get(&wire)
            else {
                return self.unsupported(
                    scope_id,
                    node,
                    node_ref.kind(),
                    "parallel scope needs an indexed output",
                );
            };
            relations.push(format!("({count_expression}) = {count}"));
            Some(*count)
        } else {
            None
        };
        if let Some(slot) = env.take_missing_loop_index() {
            return Err(ExportError::MissingLoopIndex(slot));
        }
        let mut child_env = env.clone();
        if parallel {
            if let Some(slot) = loop_owner_slot(
                self.graph,
                &self.graph.child_scope_id(scope_id, node).expect("child scope"),
            ) {
                child_env.loop_indices.insert(slot, "(Int.ofNat i.val)".into());
                child_env.loop_index_nats.insert(slot, "i".into());
            }
        }
        let mut binding_guards = Vec::new();
        for (_, value) in bindings {
            append_expression_guards(value, &child_env, &mut binding_guards);
        }
        if let Some(count) = match node_ref.kind() {
            NodeKind::ParallelLoop(spec) => Some(&spec.count),
            _ => None,
        } {
            append_expression_guards(count, env, relations);
        }
        let child_params = params_update(&child_env, bindings);
        if let Some(slot) = child_env.take_missing_loop_index() {
            return Err(ExportError::MissingLoopIndex(slot));
        }
        let input = if parallel {
            let modes = input_modes.expect("parallel modes");
            tuple_expr(
                &args
                    .iter()
                    .enumerate()
                    .map(|(position, wire)| match modes[position] {
                        crate::node::LoopInputMode::Broadcast => wire_name(*wire),
                        crate::node::LoopInputMode::Zip => {
                            let crate::types::ConcreteWireType::IndexedFamily { count, .. } =
                                &self.validated.scopes[scope_id].wire_types[wire]
                            else {
                                unreachable!("validated zip input is an indexed family")
                            };
                            if Some(*count) == parallel_count {
                                format!("({} i)", wire_name(*wire))
                            } else {
                                // A zip may read a prefix of a longer family. Validation proves
                                // its length covers the loop; Lean still needs the Fin coercion.
                                format!("({} ⟨i.val, by omega⟩)", wire_name(*wire))
                            }
                        }
                        crate::node::LoopInputMode::ZipOffset { offset } => {
                            format!("({} ⟨i.val + {}, by omega⟩)", wire_name(*wire), offset)
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            tuple_expr(&args.iter().map(|wire| wire_name(*wire)).collect::<Vec<_>>())
        };
        let output = tuple_expr(
            &outputs
                .iter()
                .map(|name| if parallel { format!("({} i)", name) } else { name.clone() })
                .collect::<Vec<_>>(),
        );
        let call = if parallel {
            format!(
                "(∀ i : Fin {}, {}{} {} {} {} {})",
                parallel_count.expect("count"),
                if binding_guards.is_empty() {
                    String::new()
                } else {
                    format!("{} ∧ ", binding_guards.join(" ∧ "))
                },
                child_name,
                child_params,
                child_index_args(self.graph, &child, &child_env),
                input,
                output
            )
        } else {
            relations.extend(binding_guards);
            format!("{} {} {} {}", child_name, child_params, input, output)
        };
        relations.push(call);
        Ok(())
    }
    fn emit_sequential(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        scope: &GraphScope,
        node: NodeId,
        args: &[WireRef],
        env: &LexicalEnv,
        existentials: &mut Vec<(String, String)>,
        relations: &mut Vec<String>,
    ) -> Result<(), ExportError> {
        let child = self
            .graph
            .child_scope_id(scope_id, node)
            .ok_or(ExportError::MissingChildScope { scope: scope_id.clone(), node })?;
        self.current_uses_hash_model |= self.requires_hash_model;
        let child_name = format!(
            "{}{}{}",
            self.scopes[&child],
            if self.requires_backend { " backend" } else { "" },
            if self.requires_hash_model { " hashModel" } else { "" }
        );
        let kind = scope.node(node).expect("node").kind();
        let NodeKind::SequentialLoop(spec) = kind else { unreachable!() };
        let state_values =
            args[..spec.carried_count].iter().map(|w| wire_name(*w)).collect::<Vec<_>>();
        let state = tuple_expr(&state_values);
        let invariants =
            args[spec.carried_count..].iter().map(|w| wire_name(*w)).collect::<Vec<_>>();
        let final_state = tuple_expr(
            &(0..spec.carried_count)
                .map(|port| wire_name(WireRef { node, port: crate::types::Port(port as u32) }))
                .collect::<Vec<_>>(),
        );
        for port in 0..spec.carried_count {
            let output = wire_name(WireRef { node, port: crate::types::Port(port as u32) });
            let ty = self.output_type(scope, node, port as u32);
            self.bind_existential(&output, &ty);
            existentials.push((output, ty));
        }
        let mut child_env = env.clone();
        if let Some(slot) = loop_owner_slot(self.graph, &child) {
            child_env.loop_indices.insert(slot, "(Int.ofNat i)".into());
            child_env.loop_index_nats.insert(slot, "i".into());
        }
        let mut binding_guards = Vec::new();
        for (_, value) in &spec.bindings {
            append_expression_guards(value, &child_env, &mut binding_guards);
        }
        append_expression_guards(&spec.count, env, relations);
        let child_params = params_update(&child_env, &spec.bindings);
        if let Some(slot) = child_env.take_missing_loop_index() {
            return Err(ExportError::MissingLoopIndex(slot));
        }
        let current_values = (0..spec.carried_count)
            .map(|position| tuple_projection("current", position, spec.carried_count))
            .collect::<Vec<_>>();
        let child_input = tuple_expr(
            &current_values.into_iter().chain(invariants.iter().cloned()).collect::<Vec<_>>(),
        );
        let body = format!(
            "fun (i : Nat) (current next : {}) => {}{} {} {} {} next",
            tuple_type(
                &args[..spec.carried_count]
                    .iter()
                    .map(|wire| self.lean_type(&self.validated.scopes[scope_id].wire_types[wire]))
                    .collect::<Vec<_>>()
            ),
            if binding_guards.is_empty() {
                String::new()
            } else {
                format!("{} ∧ ", binding_guards.join(" ∧ "))
            },
            child_name,
            child_params,
            child_index_args(self.graph, &child, &child_env),
            child_input
        );
        relations.push(format!("0 ≤ ({})", env.expr(&spec.count)));
        relations.push(format!(
            "MxxIR.IterRuns ({body}) (Int.toNat ({})) {} {}",
            env.expr(&spec.count),
            state,
            final_state
        ));
        if let Some(slot) = env.take_missing_loop_index() {
            return Err(ExportError::MissingLoopIndex(slot));
        }
        Ok(())
    }
    fn unsupported<T>(
        &self,
        scope: &FrozenGraphScopeId,
        node: NodeId,
        kind: &NodeKind,
        operation: &str,
    ) -> Result<T, ExportError> {
        Err(ExportError::UnsupportedOperation {
            scope: scope.clone(),
            node,
            operation: format!("{operation}: {kind:?}"),
        })
    }
    fn lean_type(&self, ty: &crate::types::ConcreteWireType) -> String {
        match ty {
            crate::types::ConcreteWireType::ConstantInt | crate::types::ConcreteWireType::Int => {
                "Int".into()
            }
            crate::types::ConcreteWireType::ConstantReal | crate::types::ConcreteWireType::Real => {
                "Rat".into()
            }
            crate::types::ConcreteWireType::ConstantBool | crate::types::ConcreteWireType::Bool => {
                "Bool".into()
            }
            crate::types::ConcreteWireType::Bytes { .. } => "ByteArray".into(),
            crate::types::ConcreteWireType::TypedBlob { .. } => {
                self.options.primitives.blob_type.clone()
            }
            crate::types::ConcreteWireType::Matrix(matrix) => {
                self.matrix_type(matrix, &self.options.primitives.matrix_type)
            }
            crate::types::ConcreteWireType::Trapdoor { matrix, .. } => format!(
                "{} ({}) Unit",
                self.options.primitives.trapdoor_type,
                self.matrix_type(matrix, &self.options.primitives.matrix_type)
            ),
            crate::types::ConcreteWireType::SmallMatrix { matrix, .. } => {
                self.matrix_type(matrix, &self.options.primitives.small_matrix_type)
            }
            crate::types::ConcreteWireType::Preimage { matrix, .. } => {
                self.matrix_type(matrix, &self.options.primitives.preimage_type)
            }
            crate::types::ConcreteWireType::IndexedFamily { element, count } => {
                format!("Fin {} → {}", count, self.lean_type(element))
            }
        }
    }

    fn matrix_type(&self, matrix: &crate::types::ConcreteMatrixType, prefix: &str) -> String {
        format!(
            "{} {} {} {} {}",
            prefix, matrix.modulus, matrix.ring_dimension, matrix.rows, matrix.columns
        )
    }
}

fn valid_identifier(name: &str) -> Result<(), ExportError> {
    if name.is_empty() ||
        !name.chars().enumerate().all(|(i, c)| {
            c == '_' || c.is_ascii_alphanumeric() && (i > 0 || !c.is_ascii_digit())
        })
    {
        return Err(ExportError::InvalidIdentifier(name.to_owned()));
    }
    Ok(())
}
fn scope_name(id: &FrozenGraphScopeId) -> String {
    match id {
        FrozenGraphScopeId::Root => "generatedRoot".into(),
        FrozenGraphScopeId::Subgraph { canonical_name } => {
            format!("scope_{}", sanitize(canonical_name))
        }
        FrozenGraphScopeId::ParallelBody { parent, owner } => {
            format!("parallel_{}_{}", scope_name(parent), owner.0)
        }
        FrozenGraphScopeId::SequentialBody { parent, owner } => {
            format!("sequential_{}_{}", scope_name(parent), owner.0)
        }
    }
}
fn sanitize(name: &str) -> String {
    name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect()
}
fn wire_name(wire: WireRef) -> String {
    format!("w_{}_{}", wire.node.0, wire.port.0)
}
fn tuple_type(types: &[String]) -> String {
    match types {
        [] => "Unit".into(),
        [one] => one.clone(),
        _ => types.iter().rev().fold("Unit".into(), |tail, ty| {
            let ty = if ty.contains('→') { format!("({ty})") } else { ty.clone() };
            format!("{} × {}", ty, tail)
        }),
    }
}
fn tuple_expr(values: &[String]) -> String {
    match values {
        [] => "()".into(),
        [one] => one.clone(),
        _ => values.iter().rev().fold("()".into(), |tail, value| format!("({}, {})", value, tail)),
    }
}
fn tuple_projection(name: &str, position: usize, total: usize) -> String {
    if total == 1 {
        return name.into();
    }
    let mut current = name.to_owned();
    for _ in 0..position {
        current.push_str(".2");
    }
    if position + 1 == total {
        current.push_str(".1");
    } else {
        current.push_str(".1");
    }
    current
}
fn scope_input_wires(scope: &GraphScope) -> Vec<WireRef> {
    // Graph::freeze intentionally gives the root no formal input vector: runtime root inputs are
    // discovered from named Input nodes.  Child scopes, in contrast, have an authoritative formal
    // vector supplied by their SubgraphHandle and must not infer new formals from body nodes.
    if matches!(scope.id(), FrozenGraphScopeId::Root) {
        let mut inputs = Vec::new();
        for (position, node) in scope.nodes().iter().enumerate() {
            if matches!(node.kind(), NodeKind::Input { .. }) {
                inputs.push(WireRef { node: NodeId(position as u64), port: crate::types::Port(0) });
            }
        }
        inputs
    } else {
        scope.inputs().to_vec()
    }
}
fn loop_parameters(
    graph: &Graph,
    id: &FrozenGraphScopeId,
    referenced: &BTreeSet<String>,
) -> String {
    scope_loop_slots(graph, id)
        .into_iter()
        .map(|slot| {
            let name = format!("i_{slot}");
            format!(" ({} : Nat)", if referenced.contains(&name) { &name } else { "_" })
        })
        .collect()
}
fn scope_loop_slots(graph: &Graph, id: &FrozenGraphScopeId) -> Vec<u32> {
    let mut slots = match id {
        FrozenGraphScopeId::Root | FrozenGraphScopeId::Subgraph { .. } => Vec::new(),
        FrozenGraphScopeId::ParallelBody { parent, owner } |
        FrozenGraphScopeId::SequentialBody { parent, owner } => {
            let mut outer = scope_loop_slots(graph, parent);
            let slot =
                graph.scope(parent).and_then(|scope| scope.node(*owner)).and_then(
                    |node| match node.kind() {
                        NodeKind::ParallelLoop(spec) => Some(spec.index_slot),
                        NodeKind::SequentialLoop(spec) => Some(spec.index_slot),
                        _ => None,
                    },
                );
            if let Some(slot) = slot {
                outer.push(slot);
            }
            outer
        }
    };
    slots.dedup();
    slots
}
fn loop_owner_slot(graph: &Graph, id: &FrozenGraphScopeId) -> Option<u32> {
    match id {
        FrozenGraphScopeId::ParallelBody { parent, owner } |
        FrozenGraphScopeId::SequentialBody { parent, owner } => graph
            .scope(parent)
            .and_then(|scope| scope.node(*owner))
            .and_then(|node| match node.kind() {
                NodeKind::ParallelLoop(spec) => Some(spec.index_slot),
                NodeKind::SequentialLoop(spec) => Some(spec.index_slot),
                _ => None,
            }),
        _ => None,
    }
}
fn child_index_args(graph: &Graph, child: &FrozenGraphScopeId, env: &LexicalEnv) -> String {
    scope_loop_slots(graph, child)
        .into_iter()
        .map(|slot| {
            let name =
                env.loop_index_nats.get(&slot).cloned().unwrap_or_else(|| format!("i_{slot}"));
            env.referenced_loop_names.borrow_mut().insert(name.clone());
            name
        })
        .collect::<Vec<_>>()
        .join(" ")
}
fn indent_continuation(value: &str, indent: usize) -> String {
    value.replace(" ∧ ", &format!(" ∧\n{}", "  ".repeat(indent)))
}

fn params_update(env: &LexicalEnv, bindings: &[(String, IntExpr)]) -> String {
    if bindings.is_empty() {
        return "params".into();
    }
    let updates = bindings
        .iter()
        .map(|(name, value)| format!("«{}» := {}", name, env.expr(value)))
        .collect::<Vec<_>>();
    format!("{{ params with {} }}", updates.join(", "))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        expr::ParamEnv,
        graph::{
            CompileParameter, GraphOutput, NodeHandle, SubgraphHandle, with_new_construction_scope,
        },
        node::{IndexRange, LoopInputMode, MatrixBinaryOp, ParallelLoop, SequentialLoop},
        types::{MatrixType, WireType},
    };
    use num_bigint::BigInt;
    use std::collections::BTreeMap;

    fn scalar_input(name: &str) -> crate::graph::ValueHandle {
        NodeHandle::new(
            NodeKind::Input { name: name.into(), wire_type: WireType::Int, artifact: None },
            vec![],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap()
    }

    #[test]
    fn export_supports_integer_expression_semantics() {
        for (expression, expected) in [
            (IntExpr::FloorDiv(Box::new(7.into()), Box::new(2.into())), "Int.fdiv (7) (2)"),
            (IntExpr::Rem(Box::new(7.into()), Box::new(2.into())), "Int.fmod (7) (2)"),
            (
                IntExpr::Select { selector: Box::new(0.into()), branches: vec![7.into()] },
                "MxxIR.selectInt (0) [(7)]",
            ),
        ] {
            let value = NodeHandle::new(
                NodeKind::EvaluateInt(expression),
                vec![],
                vec![WireType::ConstantInt],
            )
            .output(0)
            .unwrap();
            let (graph, _) = Graph::freeze(
                "supported-expression",
                vec![],
                BTreeMap::from([("value".into(), GraphOutput { value, confidentiality: None })]),
                vec![],
                vec![],
                BTreeMap::new(),
            )
            .unwrap();
            let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
            let artifact = export(&validated, &ExportOptions::default()).unwrap();
            assert!(artifact.source.contains(expected));
        }
    }

    #[test]
    fn export_boundary_preserves_retained_roots_and_output_aliases() {
        let x = scalar_input("source");
        let retained =
            NodeHandle::new(NodeKind::ConstantInt(9.into()), vec![], vec![WireType::ConstantInt])
                .output(0)
                .unwrap();
        let (graph, _) = Graph::freeze(
            "boundary",
            vec![],
            BTreeMap::from([
                ("first".into(), GraphOutput { value: x.clone(), confidentiality: None }),
                ("alias".into(), GraphOutput { value: x, confidentiality: None }),
            ]),
            vec![retained],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        let root = graph.scope(&FrozenGraphScopeId::Root).unwrap();
        assert_eq!(artifact.root.output_count, root.outputs().len());
        assert_eq!(artifact.root.output_count, 3);
        assert_eq!(artifact.root.output_type, "Int × Int × Int × Unit");
        assert_eq!(artifact.root.inputs["source"].projection, "inputs");
        assert_eq!(artifact.root.outputs["first"], artifact.root.outputs["alias"]);
        let first = &artifact.root.outputs["first"];
        assert_eq!(root.outputs()[first.tuple_index], graph.outputs()["first"].value);
        assert_eq!(first.projection, "outputs.2.1");
        assert_eq!(artifact.root.relation, "Generated.generatedRoot");
        assert_eq!(artifact.root.parameter_type, "Generated.Params");
        assert_eq!(artifact.root.parameters["unit"].root_value.as_deref(), Some("()"));
    }

    #[test]
    fn export_refresh_transforms_with_heterogeneous_crt_sources() {
        let matrix = |modulus| MatrixType {
            modulus: IntExpr::constant(modulus),
            ring_dimension: IntExpr::constant(2),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let source = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: matrix(45),
                value: ConstantMatrix::Polynomial {
                    coefficients: vec![IntExpr::constant(44), IntExpr::constant(23)],
                },
            },
            vec![],
            vec![WireType::Matrix(matrix(45))],
        )
        .output(0)
        .unwrap();
        let switched = NodeHandle::new(
            NodeKind::ModulusSwitch { modulus: IntExpr::constant(15) },
            vec![source.clone()],
            vec![WireType::Matrix(matrix(15))],
        )
        .output(0)
        .unwrap();
        let reduced = NodeHandle::new(
            NodeKind::ModulusReduce { modulus: IntExpr::constant(5) },
            vec![source],
            vec![WireType::Matrix(matrix(5))],
        )
        .output(0)
        .unwrap();
        let reduced = NodeHandle::new(
            NodeKind::CenteredRebase { modulus: IntExpr::constant(5) },
            vec![reduced],
            vec![WireType::Matrix(matrix(5))],
        )
        .output(0)
        .unwrap();
        let conjugate = NodeHandle::new(
            NodeKind::RingAutomorphism { index: IntExpr::constant(3) },
            vec![reduced],
            vec![WireType::Matrix(matrix(5))],
        )
        .output(0)
        .unwrap();
        let value = NodeHandle::new(
            NodeKind::CrtRecompose {
                modulus: IntExpr::constant(45),
                plaintext_moduli: vec![IntExpr::constant(3), IntExpr::constant(5)],
                reconstruction_coefficients: vec![IntExpr::constant(10), IntExpr::constant(6)],
            },
            vec![switched, conjugate],
            vec![WireType::Matrix(matrix(45))],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "refresh_transforms",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(artifact.source.contains("modulusSwitchRuns"));
        assert!(artifact.source.contains("modulusReduceRuns"));
        assert!(artifact.source.contains("centeredRebaseRuns"));
        assert!(artifact.source.contains("ringAutomorphismRuns"));
        assert_eq!(artifact.source.matches("crtRecomposeLevel").count(), 2);
        assert!(artifact.source.contains("abbrev generatedRoot.constraints"));
    }

    #[test]
    fn test_export_rns_conversions_preserves_basis_and_normalization() {
        let matrix = |modulus| MatrixType {
            modulus: IntExpr::constant(modulus),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let source = NodeHandle::new(
            NodeKind::ConstantMatrix { matrix_type: matrix(17), value: ConstantMatrix::Zero },
            vec![],
            vec![WireType::Matrix(matrix(17))],
        )
        .output(0)
        .unwrap();
        let lifted = NodeHandle::new(
            NodeKind::RnsModUp {
                modulus: IntExpr::constant(17 * 97),
                source_moduli: vec![17],
                digit_size: 1,
                normalize: false,
            },
            vec![source],
            vec![WireType::Matrix(matrix(17 * 97))],
        )
        .output(0)
        .unwrap();
        let value = NodeHandle::new(
            NodeKind::RnsModDown {
                modulus: IntExpr::constant(17),
                source_moduli: vec![17, 97],
                plaintext_modulus: IntExpr::constant(2),
            },
            vec![lifted],
            vec![WireType::Matrix(matrix(17))],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "rns_conversions",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(artifact.source.contains("rnsModUpRuns [17] 1 false"));
        assert!(artifact.source.contains("rnsModDownRuns [17, 97] (2)"));
    }

    #[test]
    fn export_constant_polynomial_preserves_payload_and_expression_guards() {
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(2),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let value = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: matrix.clone(),
                value: ConstantMatrix::Polynomial {
                    coefficients: vec![
                        IntExpr::constant(-3),
                        IntExpr::Div(
                            Box::new(IntExpr::constant(8)),
                            Box::new(IntExpr::constant(2)),
                        ),
                    ],
                },
            },
            vec![],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "constants",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(artifact.source.contains("matrixPolynomial [(-3), (MxxIR.exactDiv (8) (2))]"));
        let normalized = artifact.source.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(normalized.contains("2 ≠ 0 ∧ 8 % 2 = 0"));
    }

    #[test]
    fn export_uses_one_lexical_wire_for_both_subtraction_operands() {
        let x = scalar_input("x");
        let y = NodeHandle::new(
            NodeKind::IntBinary(IntBinaryOp::Subtract),
            vec![x.clone(), x],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "ssa",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: y, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(artifact.source.contains("w_0_0 + ") == false);
        assert!(
            artifact.source.contains("w_1_0 - w_1_0") || artifact.source.contains("w_0_0 - w_0_0")
        );
        assert_eq!(
            artifact
                .source_map
                .entries
                .iter()
                .filter(|entry| entry.node == Some(NodeId(0)))
                .count(),
            1
        );
    }

    #[test]
    fn export_matrix_addition_is_taken_from_frozen_nodes() {
        let matrix = MatrixType {
            modulus: IntExpr::constant(BigInt::from(17)),
            ring_dimension: IntExpr::constant(2),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let ty = WireType::Matrix(matrix);
        let x = NodeHandle::new(
            NodeKind::Input { name: "x".into(), wire_type: ty.clone(), artifact: None },
            vec![],
            vec![ty.clone()],
        )
        .output(0)
        .unwrap();
        let y = NodeHandle::new(
            NodeKind::MatrixBinary(MatrixBinaryOp::Add),
            vec![x.clone(), x],
            vec![ty],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "matrix",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: y, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(artifact.source.contains("matrixAdd w_0_0 w_0_0"));
    }

    #[test]
    fn export_keeps_named_child_and_parallel_body_structural() {
        let child = with_new_construction_scope(|scope| {
            let value = scalar_input("value");
            SubgraphHandle::new("identity", scope, vec![value.clone()], vec![value]).unwrap()
        });
        let family = NodeHandle::new(
            NodeKind::Input {
                name: "family".into(),
                wire_type: WireType::IndexedFamily {
                    element: Box::new(WireType::Int),
                    count: IntExpr::constant(4),
                },
                artifact: None,
            },
            vec![],
            vec![WireType::IndexedFamily {
                element: Box::new(WireType::Int),
                count: IntExpr::constant(4),
            }],
        )
        .output(0)
        .unwrap();
        let parallel = NodeHandle::parallel_loop(
            child,
            vec![family],
            vec![WireType::IndexedFamily {
                element: Box::new(WireType::Int),
                count: IntExpr::constant(4),
            }],
            ParallelLoop {
                count: IntExpr::constant(4),
                minimum_count: 0,
                index_slot: 0,
                bindings: vec![],
                input_modes: vec![LoopInputMode::Zip],
            },
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "parallel",
            vec![],
            BTreeMap::from([(
                "out".into(),
                GraphOutput { value: parallel, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(artifact.source.contains("parallel_generatedRoot"));
        assert!(artifact.source.contains("∀ i : Fin 4"));
        assert!(artifact.source.contains("parallel_generatedRoot_1 params i "));
        assert!(artifact.source.contains("(_ : Nat) (inputs : Int)"));
        assert_eq!(artifact.static_node_visits, 3);
    }

    #[test]
    fn export_zip_prefix_preserves_source_indices_and_checks_family_extent() {
        for count in [0, 1, 3, 4] {
            let child = with_new_construction_scope(|scope| {
                let value = scalar_input("value");
                SubgraphHandle::new("prefix-identity", scope, vec![value.clone()], vec![value])
                    .unwrap()
            });
            let family_type =
                WireType::IndexedFamily { element: Box::new(WireType::Int), count: 3.into() };
            let family = NodeHandle::new(
                NodeKind::Input {
                    name: "family".into(),
                    wire_type: family_type.clone(),
                    artifact: None,
                },
                vec![],
                vec![family_type],
            )
            .output(0)
            .unwrap();
            let output = NodeHandle::parallel_loop(
                child,
                vec![family],
                vec![WireType::IndexedFamily {
                    element: Box::new(WireType::Int),
                    count: count.into(),
                }],
                ParallelLoop {
                    count: count.into(),
                    minimum_count: 0,
                    index_slot: 0,
                    bindings: vec![],
                    input_modes: vec![LoopInputMode::Zip],
                },
            )
            .output(0)
            .unwrap();
            let (graph, _) = Graph::freeze(
                "prefix",
                vec![],
                BTreeMap::from([(
                    "out".into(),
                    GraphOutput { value: output, confidentiality: None },
                )]),
                vec![],
                vec![],
                BTreeMap::new(),
            )
            .unwrap();
            let validated = crate::validate(&graph, &ParamEnv::default());
            if count == 4 {
                assert!(validated.is_err(), "zip must not read beyond the source family");
                continue;
            }
            let artifact = export(&validated.unwrap(), &ExportOptions::default()).unwrap();
            assert!(artifact.source.contains(&format!("∀ i : Fin {count}")));
            if count == 3 {
                assert!(artifact.source.contains("(w_0_0 i)"));
                assert!(!artifact.source.contains("⟨i.val, by omega⟩"));
            } else {
                assert!(artifact.source.contains("(w_0_0 ⟨i.val, by omega⟩)"));
            }
            let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../test_data/lean_ir_fixtures")
                .join(format!("zip_prefix_{count}"));
            std::fs::create_dir_all(&directory).unwrap();
            std::fs::write(directory.join("Generated.lean"), artifact.source).unwrap();
        }
    }

    #[test]
    fn export_scans_root_inputs_but_keeps_zero_input_child_unit() {
        let child = with_new_construction_scope(|scope| {
            let value = NodeHandle::new(
                NodeKind::ConstantInt(BigInt::from(7)),
                vec![],
                vec![WireType::ConstantInt],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("zero_input", scope, vec![], vec![value]).unwrap()
        });
        let root_input = scalar_input("root_input");
        let call = NodeHandle::subgraph_call(child, vec![], vec![], vec![]).output(0).unwrap();
        let (graph, _) = Graph::freeze(
            "root-child-inputs",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: call, confidentiality: None })]),
            vec![],
            vec![root_input],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(artifact.source.contains("(_ : Unit)"));
        assert!(artifact.source.contains("scope_zero_input params ()"));
        assert!(artifact.source.contains("(inputs : Int)"));
    }

    #[test]
    fn expression_selection_guards_only_the_selected_branch() {
        let env = LexicalEnv::default();
        let expression = IntExpr::Select {
            selector: Box::new(IntExpr::constant(0)),
            branches: vec![
                IntExpr::constant(7),
                IntExpr::FloorDiv(Box::new(IntExpr::constant(-3)), Box::new(IntExpr::constant(0))),
            ],
        };
        assert_eq!(expression.evaluate(&ParamEnv::default()).unwrap(), BigInt::from(7));
        assert_eq!(
            expression_guards(&expression, &env),
            vec!["0 ≤ (0)", "(0) < 2", "((0) = 1 → ((0) ≠ 0))"]
        );
        assert!(env.expr(&expression).contains("Int.fdiv"));
        let value =
            NodeHandle::new(NodeKind::EvaluateInt(expression), vec![], vec![WireType::ConstantInt])
                .output(0)
                .unwrap();
        let (graph, _) = Graph::freeze(
            "lazy_select",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        let normalized = artifact.source.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(normalized.contains("((0) = 1 → ((0) ≠ 0)) ∧ outputs ="));

        let remainder =
            IntExpr::Rem(Box::new(IntExpr::constant(3)), Box::new(IntExpr::constant(-2)));
        assert_eq!(remainder.evaluate(&ParamEnv::default()).unwrap(), BigInt::from(-1));
        assert_eq!(env.expr(&remainder), "Int.fmod (3) (-2)");
    }

    #[test]
    fn loop_reference_tracking_respects_shadowing() {
        let mut outer = LexicalEnv::default();
        outer.loop_indices.insert(7, "(Int.ofNat i_7)".into());
        outer.loop_index_nats.insert(7, "i_7".into());
        assert!(outer.referenced_loop_names.borrow().is_empty());
        let mut child = outer.clone();
        child.loop_indices.insert(7, "(Int.ofNat i)".into());
        child.loop_index_nats.insert(7, "i".into());
        assert_eq!(child.expr(&IntExpr::LoopIndex(7)), "(Int.ofNat i)");
        assert!(!outer.referenced_loop_names.borrow().contains("i_7"));
        assert_eq!(outer.expr(&IntExpr::LoopIndex(7)), "(Int.ofNat i_7)");
        assert!(outer.referenced_loop_names.borrow().contains("i_7"));
    }

    #[test]
    fn export_preserves_child_formal_order_at_call_site() {
        let child = with_new_construction_scope(|scope| {
            let first = scalar_input("first");
            let second = scalar_input("second");
            SubgraphHandle::new("ordered", scope, vec![first.clone(), second], vec![first]).unwrap()
        });
        let first = scalar_input("first");
        let second = scalar_input("second");
        let call = NodeHandle::subgraph_call(
            child,
            vec![second.clone(), first.clone()],
            vec![],
            vec![None, None],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "formal-order",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: call, confidentiality: None })]),
            vec![],
            vec![first, second],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(
            artifact
                .source
                .contains("def scope_ordered (params : Params) (inputs : Int × Int × Unit)")
        );
        assert!(artifact.source.contains("scope_ordered params (w_0_0, (w_1_0, ()))"));
        assert!(artifact.source.contains("let w_0_0 : Int := inputs.1"));
        assert!(artifact.source.contains("let _ : Int := inputs.2.1"));
        assert!(artifact.source_map.entries.iter().any(|entry| {
            entry.scope != FrozenGraphScopeId::Root &&
                entry.generated == "_" &&
                entry.port == Some(0)
        }));
    }

    #[test]
    fn export_hash_model_names_only_actual_hash_and_child_call_users() {
        let child = with_new_construction_scope(|scope| {
            let value = scalar_input("value");
            SubgraphHandle::new("plain", scope, vec![value.clone()], vec![value]).unwrap()
        });
        let scalar = scalar_input("scalar");
        let call =
            NodeHandle::subgraph_call(child, vec![scalar], vec![], vec![None]).output(0).unwrap();
        let key_type = WireType::Bytes { length: 32.into() };
        let key = NodeHandle::new(
            NodeKind::Input { name: "key".into(), wire_type: key_type.clone(), artifact: None },
            vec![],
            vec![key_type],
        )
        .output(0)
        .unwrap();
        let matrix = MatrixType {
            modulus: 17.into(),
            ring_dimension: 2.into(),
            rows: 1.into(),
            columns: 1.into(),
        };
        let sampled = NodeHandle::new(
            NodeKind::HashSample {
                matrix_type: matrix.clone(),
                variant: crate::node::HashVariant::Plain,
                tag_prefix: vec![],
                tag_components: Vec::new(),
                base: None,
                digit_count: None,
            },
            vec![key],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "hash-binders",
            vec![],
            BTreeMap::from([
                ("plain".into(), GraphOutput { value: call, confidentiality: None }),
                ("sampled".into(), GraphOutput { value: sampled, confidentiality: None }),
            ]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(artifact.source.contains("def scope_plain (_ : MxxRuntime.HashModel)"));
        assert!(artifact.source.contains("def generatedRoot (hashModel : MxxRuntime.HashModel)"));
        assert!(artifact.source.contains("scope_plain hashModel params"));
        assert!(artifact.source.contains("hashSample (hashModel)"));
    }

    #[test]
    fn export_keeps_two_bindings_to_one_scope_definition() {
        let child = with_new_construction_scope(|scope| {
            let value = NodeHandle::new(
                NodeKind::ConstantInt(BigInt::from(7)),
                vec![],
                vec![WireType::ConstantInt],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("bound", scope, vec![], vec![value]).unwrap()
        });
        let first = NodeHandle::subgraph_call(
            child.clone(),
            vec![],
            vec![("k".into(), IntExpr::constant(1))],
            vec![],
        )
        .output(0)
        .unwrap();
        let second = NodeHandle::subgraph_call(
            child,
            vec![],
            vec![("k".into(), IntExpr::constant(2))],
            vec![],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "two-bindings",
            vec![],
            BTreeMap::from([
                ("first".into(), GraphOutput { value: first, confidentiality: None }),
                ("second".into(), GraphOutput { value: second, confidentiality: None }),
            ]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert_eq!(artifact.source.matches("def scope_bound ").count(), 1);
        assert_eq!(artifact.root.parameters["k"].lean_type, "Int");
        assert_eq!(artifact.root.parameters["k"].root_value, None);
        assert!(artifact.source.contains("scope_bound { params with «k» := 1 } ()"));
        assert!(artifact.source.contains("scope_bound { params with «k» := 2 } ()"));
    }

    #[test]
    fn backend_layout_uses_lexical_child_bindings() {
        let child = with_new_construction_scope(|scope| {
            let matrix = MatrixType {
                modulus: 17.into(),
                ring_dimension: 2.into(),
                rows: 1.into(),
                columns: 2.into(),
            };
            let value = NodeHandle::new(
                NodeKind::ConstantMatrix {
                    matrix_type: matrix.clone(),
                    value: ConstantMatrix::Gadget { base: IntExpr::Var("b".into()), small: false },
                },
                vec![],
                vec![WireType::Matrix(matrix)],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("lexical_gadget", scope, vec![], vec![value]).unwrap()
        });
        for (root_base, child_base, accepted) in [(16, 32, false), (32, 16, true)] {
            let value = NodeHandle::subgraph_call(
                child.clone(),
                vec![],
                vec![("b".into(), IntExpr::Var("c".into())), ("c".into(), IntExpr::constant(32))],
                vec![],
            )
            .output(0)
            .unwrap();
            let second = NodeHandle::subgraph_call(
                child.clone(),
                vec![],
                vec![("b".into(), IntExpr::constant(16))],
                vec![],
            )
            .output(0)
            .unwrap();
            let graph = Graph::freeze(
                "lexical-layout",
                vec![CompileParameter {
                    name: "c".into(),
                    kind: crate::graph::CompileParameterKind::Integer,
                }],
                BTreeMap::from([
                    ("gadget".into(), GraphOutput { value, confidentiality: None }),
                    ("second".into(), GraphOutput { value: second, confidentiality: None }),
                ]),
                vec![],
                vec![],
                BTreeMap::new(),
            )
            .unwrap()
            .0;
            let env = ParamEnv {
                integers: BTreeMap::from([
                    ("b".into(), root_base.into()),
                    ("c".into(), child_base.into()),
                ]),
                ..ParamEnv::default()
            };
            let validated = crate::validate(&graph, &env).unwrap();
            let options = ExportOptions {
                backend_layouts: vec![BackendLayout {
                    modulus: 17.into(),
                    ring_dimension: 2,
                    base: 16.into(),
                    regular_digits: 2,
                }],
                ..ExportOptions::default()
            };
            let result = export(&validated, &options);
            assert_eq!(result.is_ok(), accepted, "{result:?}");
            if let Ok(artifact) = result {
                assert!(artifact.root.requires_backend);
                assert!(
                    artifact
                        .source
                        .contains("scope_lexical_gadget backend { params with «b» := 16 }")
                );
            }
        }
    }

    #[test]
    fn export_parenthesizes_nested_integer_helper_arguments() {
        let c = IntExpr::constant;
        let expressions = [
            IntExpr::Div(
                Box::new(IntExpr::FloorDiv(Box::new(c(8)), Box::new(c(2)))),
                Box::new(c(2)),
            ),
            IntExpr::RoundDiv(
                Box::new(IntExpr::Rem(Box::new(c(7)), Box::new(c(4)))),
                Box::new(c(2)),
            ),
            IntExpr::Log2Ceil(Box::new(IntExpr::Select {
                selector: Box::new(c(0)),
                branches: vec![
                    IntExpr::FloorDiv(Box::new(c(16)), Box::new(c(2))),
                    IntExpr::Div(Box::new(c(1)), Box::new(c(0))),
                ],
            })),
        ];
        let expected = [2, 2, 3];
        let outputs = expressions
            .into_iter()
            .zip(expected)
            .enumerate()
            .map(|(index, (expr, expected))| {
                assert_eq!(expr.evaluate(&ParamEnv::default()).unwrap(), BigInt::from(expected));
                let value = NodeHandle::new(
                    NodeKind::EvaluateInt(expr),
                    vec![],
                    vec![WireType::ConstantInt],
                )
                .output(0)
                .unwrap();
                (format!("out{index}"), GraphOutput { value, confidentiality: None })
            })
            .collect();
        let (graph, _) =
            Graph::freeze("nested_helpers", vec![], outputs, vec![], vec![], BTreeMap::new())
                .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(artifact.source.contains("MxxIR.exactDiv (Int.fdiv (8) (2)) (2)"));
        assert!(artifact.source.contains("MxxIR.roundDiv (Int.fmod (7) (4)) (2)"));
        assert!(artifact.source.contains("MxxIR.log2Ceil (MxxIR.selectInt"));
    }

    #[test]
    fn constraint_suffixes_preserve_all_guards_and_ignore_quoted_fields() {
        let outputs = (0..33)
            .map(|index| {
                let value = NodeHandle::new(
                    NodeKind::EvaluateInt(IntExpr::FloorDiv(
                        Box::new(IntExpr::constant(index)),
                        Box::new(IntExpr::Var("backend".into())),
                    )),
                    vec![],
                    vec![WireType::ConstantInt],
                )
                .output(0)
                .unwrap();
                (format!("out{index:02}"), GraphOutput { value, confidentiality: None })
            })
            .collect();
        let (graph, _) = Graph::freeze(
            "constraint_suffixes",
            vec![CompileParameter {
                name: "backend".into(),
                kind: crate::graph::CompileParameterKind::Integer,
            }],
            outputs,
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let mut bindings = ParamEnv::default();
        bindings.integers.insert("backend".into(), 1.into());
        let validated = crate::validate(&graph, &bindings).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert_eq!(artifact.source.matches("(params.«backend») ≠ 0").count(), 33);
        assert!(artifact.source.contains("abbrev generatedRoot.constraints_1"));
        assert!(!artifact.source.contains("(backend : MxxRuntime.BackendContext)"));
        assert_eq!(artifact.scope_relations[&FrozenGraphScopeId::Root], artifact.root.relation);
        let proof = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
        assert_eq!(proof.witness_record.as_deref(), Some("Generated.generatedRoot.Witness"));
        assert_eq!(proof.constraint_count, 33);
        assert!(proof.witnesses.is_empty());
        assert_eq!(proof.node_constraints.values().map(|range| range.len()).sum::<usize>(), 33);
    }

    #[test]
    fn export_emits_exact_division_and_remainder_guards() {
        let expr =
            IntExpr::Div(Box::new(IntExpr::constant(-6)), Box::new(IntExpr::Var("den".into())));
        let value =
            NodeHandle::new(NodeKind::EvaluateInt(expr), vec![], vec![WireType::ConstantInt])
                .output(0)
                .unwrap();
        let (graph, _) = Graph::freeze(
            "exact-div",
            vec![CompileParameter {
                name: "den".into(),
                kind: crate::graph::CompileParameterKind::Integer,
            }],
            BTreeMap::from([("out".into(), GraphOutput { value, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let mut bindings = ParamEnv::default();
        bindings.integers.insert("den".into(), BigInt::from(3));
        let validated = crate::validate(&graph, &bindings).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(artifact.source.contains("MxxIR.exactDiv (-6) (params.«den»)"));
        assert!(
            artifact.source.contains("params.«den» ≠ 0") &&
                artifact.source.contains("-6 % params.«den» = 0")
        );
    }

    #[test]
    fn export_rejects_colliding_sanitized_subgraph_names() {
        let outputs = ["a-b", "a_b"]
            .into_iter()
            .map(|name| {
                let child = with_new_construction_scope(|scope| {
                    let value = NodeHandle::new(
                        NodeKind::ConstantInt(BigInt::from(7)),
                        vec![],
                        vec![WireType::ConstantInt],
                    )
                    .output(0)
                    .unwrap();
                    SubgraphHandle::new(name, scope, vec![], vec![value]).unwrap()
                });
                let value =
                    NodeHandle::subgraph_call(child, vec![], vec![], vec![]).output(0).unwrap();
                (name.to_owned(), GraphOutput { value, confidentiality: None })
            })
            .collect();
        let (graph, _) =
            Graph::freeze("scope-collision", vec![], outputs, vec![], vec![], BTreeMap::new())
                .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        assert!(matches!(
            export(&validated, &ExportOptions::default()),
            Err(ExportError::ScopeNameCollision(name)) if name == "scope_a_b"
        ));
    }

    #[test]
    fn scope_projection_metadata_matches_slice_and_following_node() {
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(2),
            rows: IntExpr::constant(2),
            columns: IntExpr::constant(2),
        };
        let input = NodeHandle::new(
            NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value: ConstantMatrix::Zero },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let sliced_type = MatrixType { rows: IntExpr::constant(1), ..matrix };
        let slice = NodeHandle::new(
            NodeKind::Slice {
                rows: Some(IndexRange { start: IntExpr::constant(0), end: IntExpr::constant(1) }),
                columns: None,
            },
            vec![input],
            vec![WireType::Matrix(sliced_type.clone())],
        )
        .output(0)
        .unwrap();
        let result = NodeHandle::new(
            NodeKind::RingAutomorphism { index: IntExpr::constant(1) },
            vec![slice],
            vec![WireType::Matrix(sliced_type)],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "slice_projections",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: result, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        let proof = &artifact.scope_proofs[&FrozenGraphScopeId::Root];
        assert_eq!(proof.node_constraints[&NodeId(1)], 0..7);
        assert_eq!(proof.node_constraints[&NodeId(2)], 7..8);
        assert_eq!(proof.body_relation.as_deref(), Some("Generated.generatedRoot.body"));
        assert_eq!(proof.value_expressions["w_1_0"], "witness.w_1_0");
        assert_eq!(proof.constraint_count, 8);
        assert_eq!(
            proof.node_facts[&NodeId(1)].last().unwrap(),
            "Generated.generatedRoot.node_1.fact_6"
        );
        assert_eq!(proof.output_fact.as_deref(), Some("Generated.generatedRoot.output_fact"));
        let mut fixture = format!(
            "{}\ntheorem sliceProjectionForward (params : Generated.Params) (output : Mxx.Primitives.ExactMatrix 17 2 1 2)\n    (h : Generated.generatedRoot params () output) :\n    ∃ sliced transformed : Mxx.Primitives.ExactMatrix 17 2 1 2,\n      MxxRuntime.sliceMatrix (0 : Mxx.Primitives.ExactMatrix 17 2 2 2) 0 1 0 2 sliced ∧\n      MxxRuntime.ringAutomorphismRuns 1 sliced transformed ∧ output = transformed := by\n  obtain ⟨witness, h⟩ := h\n  have hs := {}\n  have ht := {}\n  have ho := {}\n  rcases witness with ⟨sliced, transformed⟩\n  exact ⟨sliced, transformed, hs, ht, ho⟩\n",
            artifact.source,
            format!(
                "{} (params := params) (inputs := ()) (outputs := output) (witness := witness) h",
                proof.node_facts[&NodeId(1)].last().unwrap()
            ),
            format!(
                "{} (params := params) (inputs := ()) (outputs := output) (witness := witness) h",
                proof.node_facts[&NodeId(2)][0]
            ),
            format!(
                "{} (params := params) (inputs := ()) (outputs := output) (witness := witness) h",
                proof.output_fact.as_ref().unwrap()
            )
        );
        fixture.push_str("\nexample (params : Generated.Params) (output : Mxx.Primitives.ExactMatrix 17 2 1 2) :\n    Generated.generatedRoot params () output ↔\n    ∃ sliced transformed : Mxx.Primitives.ExactMatrix 17 2 1 2,\n      MxxRuntime.sliceMatrix (0 : Mxx.Primitives.ExactMatrix 17 2 2 2) 0 1 0 2 sliced ∧\n      MxxRuntime.ringAutomorphismRuns 1 sliced transformed ∧ output = transformed := by\n  constructor\n  · exact sliceProjectionForward params output\n  · rintro ⟨sliced, transformed, hs, ht, ho⟩\n    exact ⟨⟨sliced, transformed⟩, by decide, by decide, by decide, by decide, by decide, by decide, hs, ht, ho⟩\n");
        fixture.push_str("\nexample (params : Generated.Params) (output : Mxx.Primitives.ExactMatrix 17 2 1 2) :\n    Generated.generatedRoot params () output ↔ ∃ witness, Generated.generatedRoot.body params () output witness := Iff.rfl\n");
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/lean_scope_projections");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("SliceProjections.lean"), fixture).unwrap();
    }

    #[test]
    fn test_leaf_scope_named_facts_preserve_existential_runs() {
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(2),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let family_type = WireType::IndexedFamily {
            element: Box::new(WireType::Matrix(matrix.clone())),
            count: IntExpr::constant(2),
        };
        let decomposed_type = WireType::Preimage {
            matrix: MatrixType { rows: IntExpr::constant(2), ..matrix.clone() },
            max_coefficient_bound: IntExpr::constant(1),
        };
        let child = with_new_construction_scope(|scope| {
            let family = NodeHandle::new(
                NodeKind::Input {
                    name: "captured".into(),
                    wire_type: family_type.clone(),
                    artifact: None,
                },
                vec![],
                vec![family_type.clone()],
            )
            .output(0)
            .unwrap();
            let selected = NodeHandle::new(
                NodeKind::FamilyGetStatic { index: IntExpr::constant(1) },
                vec![family.clone()],
                vec![WireType::Matrix(matrix)],
            )
            .output(0)
            .unwrap();
            let decomposed = NodeHandle::new(
                NodeKind::GadgetDecompose {
                    base: IntExpr::constant(2),
                    digit_count: IntExpr::constant(2),
                    small: false,
                },
                vec![selected],
                vec![decomposed_type.clone()],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("captured_decomposition", scope, vec![family], vec![decomposed])
                .unwrap()
        });
        let family = NodeHandle::new(
            NodeKind::Input {
                name: "members".into(),
                wire_type: family_type.clone(),
                artifact: None,
            },
            vec![],
            vec![family_type],
        )
        .output(0)
        .unwrap();
        let parallel = NodeHandle::parallel_loop(
            child,
            vec![family],
            vec![WireType::IndexedFamily {
                element: Box::new(decomposed_type),
                count: IntExpr::constant(2),
            }],
            ParallelLoop {
                count: IntExpr::constant(2),
                minimum_count: 0,
                index_slot: 0,
                bindings: vec![],
                input_modes: vec![LoopInputMode::Broadcast],
            },
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "leaf_facts",
            vec![],
            BTreeMap::from([(
                "out".into(),
                GraphOutput { value: parallel, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(
            &validated,
            &ExportOptions {
                backend_layouts: vec![BackendLayout {
                    modulus: 17.into(),
                    ring_dimension: 2,
                    base: 2.into(),
                    regular_digits: 2,
                }],
                ..ExportOptions::default()
            },
        )
        .unwrap();
        let (scope_id, shape) = artifact
            .scope_proofs
            .iter()
            .find(|(id, _)| matches!(id, FrozenGraphScopeId::ParallelBody { .. }))
            .unwrap();
        assert_eq!(shape.witnesses.len(), 2);
        assert_eq!(shape.node_facts.values().map(Vec::len).sum::<usize>(), shape.constraint_count);
        assert_eq!(
            artifact.static_node_visits,
            graph.scopes().values().map(|scope| scope.nodes().len()).sum::<usize>()
        );
        let relation = &artifact.scope_relations[scope_id];
        let run_definition = artifact
            .source
            .split(&format!("def {} ", relation.strip_prefix("Generated.").unwrap()))
            .nth(1)
            .unwrap()
            .split("theorem ")
            .next()
            .unwrap();
        assert!(run_definition.contains("∃ ("));
        assert!(!run_definition.contains("∃ witness :"));
        let node = |predicate: fn(&NodeKind) -> bool| {
            graph
                .scope(scope_id)
                .unwrap()
                .nodes()
                .iter()
                .enumerate()
                .find(|(_, node)| predicate(node.kind()))
                .map(|(index, _)| NodeId(index as u64))
                .unwrap()
        };
        let get = node(|kind| matches!(kind, NodeKind::FamilyGetStatic { .. }));
        let decompose = node(|kind| matches!(kind, NodeKind::GadgetDecompose { .. }));
        let record = shape.witness_record.as_ref().unwrap();
        let body = shape.body_relation.as_ref().unwrap();
        let equiv = &shape.runs_iff_body;
        let get_fact = shape.node_facts[&get].last().unwrap();
        let decomposition_fact = shape.node_facts[&decompose].last().unwrap();
        let output_fact = shape.output_fact.as_ref().unwrap();
        let selected = &shape.witnesses[0].0;
        let decomposed = &shape.witnesses[1].0;
        let mut fixture = artifact.source.clone();
        fixture.push_str(&format!(r#"
example (backend : MxxRuntime.BackendContext) (params : Generated.Params) (i : Nat)
    (inputs : Fin 2 → Mxx.Primitives.ExactMatrix 17 2 1 1)
    (outputs : Mxx.Primitives.ExactMatrix 17 2 2 1)
    (h : {relation} backend params i inputs outputs) :
    ∃ witness : {record},
      MxxRuntime.familyGetStatic inputs 1 witness.{selected} ∧
      MxxRuntime.gadgetDecomposeRuns backend 2 2 witness.{selected} witness.{decomposed} ∧
      outputs = witness.{decomposed} := by
  obtain ⟨witness, facts⟩ := ({equiv} backend params i inputs outputs).mp h
  exact ⟨witness,
    {get_fact} (backend := backend) (params := params) (i_0 := i) (inputs := inputs) (outputs := outputs) (witness := witness) facts,
    {decomposition_fact} (backend := backend) (params := params) (i_0 := i) (inputs := inputs) (outputs := outputs) (witness := witness) facts,
    {output_fact} (backend := backend) (params := params) (i_0 := i) (inputs := inputs) (outputs := outputs) (witness := witness) facts⟩

example (backend : MxxRuntime.BackendContext) (params : Generated.Params) (i : Nat)
    (inputs : Fin 2 → Mxx.Primitives.ExactMatrix 17 2 1 1)
    (outputs : Mxx.Primitives.ExactMatrix 17 2 2 1)
    (witness : {record}) (facts : {body} backend params i inputs outputs witness) :
    {relation} backend params i inputs outputs :=
  ({equiv} backend params i inputs outputs).mpr ⟨witness, facts⟩
"#));
        let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../test_data/lean_leaf_facts");
        std::fs::create_dir_all(&directory).unwrap();
        std::fs::write(directory.join("LeafFacts.lean"), fixture).unwrap();
    }

    #[test]
    fn invalid_slice_range_is_rejected_before_export() {
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(2),
            rows: IntExpr::constant(2),
            columns: IntExpr::constant(2),
        };
        let input = NodeHandle::new(
            NodeKind::Input {
                name: "matrix".into(),
                wire_type: WireType::Matrix(matrix.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let slice = NodeHandle::new(
            NodeKind::Slice {
                rows: Some(IndexRange { start: IntExpr::constant(2), end: IntExpr::constant(2) }),
                columns: None,
            },
            vec![input],
            vec![WireType::Matrix(MatrixType { rows: IntExpr::constant(1), ..matrix })],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "invalid-slice",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: slice, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let error = crate::validate(&graph, &ParamEnv::default()).unwrap_err().to_string();
        assert!(error.contains("row slice must be nonempty"));
    }

    #[test]
    fn export_keeps_two_carried_values_in_one_iteration_state() {
        let (body, first, second) = with_new_construction_scope(|scope| {
            let first = scalar_input("first");
            let second = scalar_input("second");
            let body = SubgraphHandle::new(
                "step",
                scope,
                vec![first.clone(), second.clone()],
                vec![first.clone(), second.clone()],
            )
            .unwrap();
            (body, first, second)
        });
        let _ = (first, second);
        let initial_first = scalar_input("initial-first");
        let initial_second = scalar_input("initial-second");
        let loop_output = NodeHandle::sequential_loop(
            body,
            vec![initial_first, initial_second],
            vec![WireType::Int, WireType::Int],
            SequentialLoop {
                count: IntExpr::constant(3),
                index_slot: 0,
                bindings: vec![],
                carried_count: 2,
            },
        );
        let first_output = loop_output.output(0).unwrap();
        let second_output = loop_output.output(1).unwrap();
        let (graph, _) = Graph::freeze(
            "sequential",
            vec![],
            BTreeMap::from([
                ("first".into(), GraphOutput { value: first_output, confidentiality: None }),
                ("second".into(), GraphOutput { value: second_output, confidentiality: None }),
            ]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        let artifact = export(&validated, &ExportOptions::default()).unwrap();
        assert!(artifact.source.contains("MxxIR.IterRuns"));
        assert!(artifact.source.contains("current") && artifact.source.contains("next"));
        assert!(artifact.source.contains("sequential_generatedRoot_2 params i"));
        assert!(artifact.source.contains("current.1") && artifact.source.contains("current.2"));
    }

    #[test]
    fn export_preserves_trapdoor_pair_and_three_preimage_operands() {
        let trapdoor_matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(2),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(3),
        };
        let target_matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(2),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let preimage_matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(2),
            rows: IntExpr::constant(3),
            columns: IntExpr::constant(1),
        };
        let trapdoor = NodeHandle::new(
            NodeKind::TrapdoorSample {
                matrix_type: trapdoor_matrix.clone(),
                sigma: crate::expr::RealExpr::from(1),
                gadget_base: IntExpr::constant(2),
                digit_count: IntExpr::constant(1),
                preimage_max_coefficient_bound: IntExpr::constant(4),
            },
            vec![],
            vec![
                WireType::Matrix(trapdoor_matrix.clone()),
                WireType::Trapdoor {
                    matrix: trapdoor_matrix.clone(),
                    sigma: crate::expr::RealExpr::from(1),
                    gadget_base: IntExpr::constant(2),
                    digit_count: IntExpr::constant(1),
                    preimage_max_coefficient_bound: IntExpr::constant(4),
                },
            ],
        );
        let public = trapdoor.output(0).unwrap();
        let token = trapdoor.output(1).unwrap();
        let target = NodeHandle::new(
            NodeKind::Input {
                name: "target".into(),
                wire_type: WireType::Matrix(target_matrix.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(target_matrix)],
        )
        .output(0)
        .unwrap();
        let preimage = NodeHandle::new(
            NodeKind::PreimageSample {
                matrix_type: preimage_matrix.clone(),
                max_coefficient_bound: IntExpr::constant(4),
            },
            vec![public.clone(), token, target],
            vec![WireType::Preimage {
                matrix: preimage_matrix,
                max_coefficient_bound: IntExpr::constant(4),
            }],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "preimage",
            vec![],
            BTreeMap::from([(
                "out".into(),
                GraphOutput { value: preimage, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let validated = crate::validate(&graph, &ParamEnv::default()).unwrap();
        assert!(matches!(
            export(&validated, &ExportOptions::default()),
            Err(ExportError::BackendLayout(_))
        ));
        let options = ExportOptions {
            backend_layouts: vec![BackendLayout {
                modulus: 17.into(),
                ring_dimension: 2,
                base: 2.into(),
                regular_digits: 1,
            }],
            ..ExportOptions::default()
        };
        let artifact = export(&validated, &options).unwrap();
        assert!(artifact.root.requires_backend);
        let mut incompatible = options.clone();
        incompatible.backend_layouts[0].base = 32.into();
        assert!(matches!(export(&validated, &incompatible), Err(ExportError::BackendLayout(_))));
        let mut conflicting = options.clone();
        conflicting.backend_layouts.push(incompatible.backend_layouts[0].clone());
        assert!(matches!(export(&validated, &conflicting), Err(ExportError::BackendLayout(_))));
        assert!(artifact.source.contains("trapdoorSample backend"));
        assert!(
            artifact
                .source
                .contains("preimageRunsDispatched backend w_0_0 w_0_1 w_1_0 (Int.toNat (4)) w_2_0")
        );
    }

    #[test]
    fn mismatched_public_preimage_is_rejected_before_export() {
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(2),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let trapdoor_matrix = MatrixType { columns: IntExpr::constant(3), ..matrix.clone() };
        let trapdoor = NodeHandle::new(
            NodeKind::TrapdoorSample {
                matrix_type: trapdoor_matrix.clone(),
                sigma: crate::expr::RealExpr::from(1),
                gadget_base: IntExpr::constant(2),
                digit_count: IntExpr::constant(1),
                preimage_max_coefficient_bound: IntExpr::constant(4),
            },
            vec![],
            vec![
                WireType::Matrix(trapdoor_matrix.clone()),
                WireType::Trapdoor {
                    matrix: trapdoor_matrix.clone(),
                    sigma: crate::expr::RealExpr::from(1),
                    gadget_base: IntExpr::constant(2),
                    digit_count: IntExpr::constant(1),
                    preimage_max_coefficient_bound: IntExpr::constant(4),
                },
            ],
        );
        let wrong_public_matrix =
            MatrixType { columns: IntExpr::constant(2), ..trapdoor_matrix.clone() };
        let wrong_public = NodeHandle::new(
            NodeKind::Input {
                name: "wrong".into(),
                wire_type: WireType::Matrix(wrong_public_matrix.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(wrong_public_matrix)],
        )
        .output(0)
        .unwrap();
        let target = NodeHandle::new(
            NodeKind::Input {
                name: "target".into(),
                wire_type: WireType::Matrix(matrix.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let preimage_ty = WireType::Preimage {
            matrix: MatrixType { rows: IntExpr::constant(3), ..matrix },
            max_coefficient_bound: IntExpr::constant(4),
        };
        let out = NodeHandle::new(
            NodeKind::PreimageSample {
                matrix_type: if let WireType::Preimage { matrix, .. } = &preimage_ty {
                    matrix.clone()
                } else {
                    unreachable!()
                },
                max_coefficient_bound: IntExpr::constant(4),
            },
            vec![wrong_public, trapdoor.output(1).unwrap(), target],
            vec![preimage_ty],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "mismatched-preimage",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: out, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let error = crate::validate(&graph, &ParamEnv::default()).unwrap_err().to_string();
        assert!(error.contains("preimage public matrix does not match"));
    }
}
