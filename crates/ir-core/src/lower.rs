//! Lowering a validated lattice graph to a core structured binary (`mxx-core`).
//!
//! The lattice DSL keeps rings, moduli, and matrix shapes on its own types and checks them in
//! validation; [`lower`] then hands the core only sizes, leaves, and loops, at the graph's
//! concrete bindings.
//!
//! # Encoding
//!
//! A matrix over a ring with CRT moduli `q_0, ..., q_{L-1}` and dimension `N` is a family of `L`
//! limbs ([`matrix_type`]). Limb `l` is a buffer of `rows * columns * N` little-endian `u64`
//! residues modulo `q_l`, canonical in `0..q_l`: entry `(r, c)` in row-major order, and in each
//! entry the coefficients of `X^0, ..., X^{N-1}` (the coefficient domain).
//!
//! # Lowering
//!
//! An operation on matrices that works limb by limb (addition, subtraction, negation, scaling by
//! an integer, and the matrix product) is an independent loop over the limbs. Iteration `l` takes
//! limb `l` of each operand with the built-in `get` and runs one leaf ([`leaves`]) with the
//! modulus `q_l`, chosen by the compile expression `Select(l, [q_0, ..., q_{L-1}])`, and any other
//! limb-dependent constant chosen the same way. The loop's family of limbs is the result matrix.
//!
//! Structure maps to the core's loops and built-in leaves: a parallel loop is an independent loop
//! whose body takes each zipped family whole and reads its member with `get` (`get_dynamic` for a
//! gathered one), a sequential loop is a loop with carried values, a subgraph call without
//! bindings is a loop of one iteration, and family packing, indexing, and selection are the
//! built-ins of the same name. Loop index slot `s` of the graph is core slot
//! `s + 1`; slot `0` is the limb loops'. Runtime integers are `Int`s of one word.
//!
//! Only the node kinds handled in [`lower`] are supported so far, and only loop bodies whose
//! types do not depend on the loop index; anything else is reported as
//! [`LowerError::Unsupported`].

pub mod leaves;
mod rv64_leaves;

use crate::{
    expr::IntExpr as GraphExpr,
    graph::FrozenGraphScopeId,
    node::{
        ConcatAxis, ConstantMatrix, HashTagComponent, HashVariant, IntBinaryOp, IntCompareOp,
        LoopInputMode, MatrixBinaryOp, NodeKind, RealBinaryOp,
    },
    types::{
        CoefficientBoundDomain, ConcreteMatrixType, ConcreteWireType, Port, TrapdoorInstantiation,
        WireRef,
    },
    validate::ValidatedGraph,
};
use mxx_core::{
    expr::IntExpr,
    leaf::{Builtin, Leaf},
    program::{DefinitionBuilder, Loop, Program, ProgramBuilder, Value},
    types::CoreType,
};
use num_bigint::BigInt;
use num_integer::Integer;
use std::{collections::BTreeMap, sync::Arc};

#[derive(Debug, thiserror::Error)]
pub enum LowerError {
    #[error("{0} is not lowered to the core yet")]
    Unsupported(String),
    #[error("wire {0:?} has no validated type")]
    MissingType(WireRef),
    #[error("wire {0:?} is read before it is produced")]
    Unproduced(WireRef),
    #[error("expression: {0}")]
    Expression(String),
}

/// The core type of a matrix: a family of limbs.
pub fn matrix_type(matrix: &ConcreteMatrixType) -> CoreType {
    let words = matrix.rows * matrix.columns * matrix.ring.ring_dimension() as usize;
    CoreType::family(CoreType::buffer(8, words), matrix.ring.crt_depth())
}

/// The parts of a trapdoor value, per limb: its public matrix, then, for a sampled trapdoor, its
/// secret block of the type returned: `[R; E]` (`2 rows x rows D`) of a computational (MP12)
/// trapdoor, or `T` (`m̄ x D`, one row) of a statistical one. The gadget trapdoor
/// (`GadgetTrapdoor`, as many columns as rows times digits) is its public matrix alone.
pub fn trapdoor_parts(
    ty: &ConcreteWireType,
) -> Option<(ConcreteMatrixType, Option<ConcreteMatrixType>)> {
    let ConcreteWireType::Trapdoor { matrix, digit_count, instantiation, .. } = ty else {
        return None;
    };
    let rows = matrix.rows;
    let mut secret = matrix.clone();
    secret.columns = rows * digit_count;
    match instantiation {
        TrapdoorInstantiation::Computational if matrix.columns == rows * digit_count => {
            Some((matrix.clone(), None))
        }
        TrapdoorInstantiation::Computational if matrix.columns == rows * (digit_count + 2) => {
            secret.rows = 2 * rows;
            Some((matrix.clone(), Some(secret)))
        }
        TrapdoorInstantiation::Statistical { m_bar }
            if rows == 1 && matrix.columns == m_bar + digit_count =>
        {
            secret.rows = *m_bar;
            Some((matrix.clone(), Some(secret)))
        }
        _ => None,
    }
}

/// The core type of a value of the graph.
pub fn core_type(ty: &ConcreteWireType, int_words: u32) -> Result<CoreType, LowerError> {
    Ok(match ty {
        // Bounded matrices keep the residues of their coefficients, like any matrix.
        ConcreteWireType::Matrix(matrix) |
        ConcreteWireType::SmallMatrix { matrix, .. } |
        ConcreteWireType::Preimage { matrix, .. } => matrix_type(matrix),
        ConcreteWireType::IndexedFamily { element, count } => {
            CoreType::family(core_type(element, int_words)?, *count as u64)
        }
        ConcreteWireType::Int | ConcreteWireType::ConstantInt => CoreType::Int { words: int_words },
        ConcreteWireType::Bytes { length } => CoreType::bytes(*length),
        ConcreteWireType::Bool | ConcreteWireType::ConstantBool => CoreType::Bool,
        ConcreteWireType::Real | ConcreteWireType::ConstantReal => leaves::REAL,
        ConcreteWireType::Trapdoor { .. } => match trapdoor_parts(ty) {
            Some((public, None)) => matrix_type(&public),
            Some((public, Some(secret))) => {
                let n = public.ring.ring_dimension() as usize;
                let words = (public.rows * public.columns + secret.rows * secret.columns) * n;
                CoreType::family(CoreType::buffer(8, words), public.ring.crt_depth())
            }
            None => return Err(LowerError::Unsupported(format!("a trapdoor of type {ty:?}"))),
        },
        other => return Err(LowerError::Unsupported(format!("a value of type {other:?}"))),
    })
}

/// The leaves a lowering uses, created once each.
/// The fast RNS base conversions.
#[derive(Clone, Copy)]
enum Conversion {
    Up { digit_size: usize, normalize: bool },
    Down { t: u64 },
}

struct Leaves {
    add: Arc<Leaf>,
    sub: Arc<Leaf>,
    neg: Arc<Leaf>,
    scale: Arc<Leaf>,
    poly_mul_acc: Arc<Leaf>,
    int_constant: Arc<Leaf>,
    int_binary: Arc<Leaf>,
    int_compare: Arc<Leaf>,
    bit_extract: Arc<Leaf>,
    bool_to_int: Arc<Leaf>,
    transpose: Arc<Leaf>,
    automorphism: Arc<Leaf>,
    multiply_monomial: Arc<Leaf>,
    uniform_mod: Arc<Leaf>,
    interval_mod: Arc<Leaf>,
    lift_int: Arc<Leaf>,
    threshold_decode: Arc<Leaf>,
    crt_map: Arc<Leaf>,
    fast_base_conversion: Arc<Leaf>,
    ntt: Arc<Leaf>,
    crt_values: Arc<Leaf>,
    int_residues: Arc<Leaf>,
    int_mat_vec: Arc<Leaf>,
    round_lift: Arc<Leaf>,
    pack_bits: Arc<Leaf>,
    hash_ints: Arc<Leaf>,
    hash_tag: Arc<Leaf>,
    real_op: Arc<Leaf>,
    int_to_real: Arc<Leaf>,
    cdt_mod: Arc<Leaf>,
    hash_uniform: Arc<Leaf>,
    gadget_decompose: Arc<Leaf>,
    trapdoor_preimage: Arc<Leaf>,
    statistical_trapgen: Arc<Leaf>,
    zero: Arc<Leaf>,
    block_diagonal: Arc<Leaf>,
}

/// The loop index slot of the limb loops.
const LIMB: u32 = 0;

/// The loop index slot of subgraph calls, which are loops of one iteration. A call inside a call's
/// body would reuse it, which validation rejects.
const SUBGRAPH: u32 = u32::MAX - 1;

/// The loop index slots of the loops of a matrix product (`PRODUCT - d` for dimension `d`) and of
/// its sum.
const PRODUCT: u32 = u32::MAX - 10;
const PRODUCT_SUM: u32 = u32::MAX - 3;

/// The loop index slot of preimage retries, whose body is one leaf call.
const RETRY: u32 = u32::MAX - 2;

/// The most candidates a preimage draws; the run faults when all are rejected (an `until` loop
/// faults after its last iteration). Each is rejected with the probability that a coefficient
/// exceeds the bound (small for the default cutoff of 6.5 widths), or that the leaf's stream runs
/// out.
const PREIMAGE_ATTEMPTS: u64 = 32;

/// The draws of `T` a statistical TrapGen makes before faulting, as the CPU backend's
/// `TRAPGEN_ATTEMPTS`.
const TRAPGEN_ATTEMPTS: u64 = 8;

/// The core slot of graph loop index slot `slot`.
fn core_slot(slot: u32) -> u32 {
    slot + 1
}

struct Lowering<'a> {
    graph: &'a ValidatedGraph,
    /// The parameters in scope: the graph's bindings, and the bindings of the calls and loops
    /// being lowered whose values do not depend on a loop index.
    env: crate::expr::ParamEnv,
    /// The bindings in scope whose values depend on a loop index, as expressions over it.
    subst: BTreeMap<String, GraphExpr>,
    /// The words of every integer: see [`int_words`].
    int_words: u32,
    leaves: Leaves,
    builder: ProgramBuilder,
}

/// The words of the lowering's integers: two's complement wide enough for the product of two
/// values below the largest modulus of the graph's rings, so the coefficient values a graph reads
/// out of its rings and their products fit; arithmetic wraps beyond that width.
pub fn int_words(graph: &ValidatedGraph) -> u32 {
    let bits = graph.resolved_rings.values().map(|ring| ring.modulus().bits()).max().unwrap_or(1);
    u32::try_from((2 * bits + 2).div_ceil(64)).unwrap_or(u32::MAX).clamp(1, 20)
}

/// Lowers `graph` at its bindings, with integers of [`int_words`] words.
pub fn lower(graph: &ValidatedGraph) -> Result<Program, LowerError> {
    lower_with_int_words(graph, int_words(graph))
}

/// Lowers `graph` with integers of `w` words, at least [`int_words`]: programs that exchange
/// integers, such as the algorithms of one scheme, use one width.
pub fn lower_with_int_words(graph: &ValidatedGraph, w: u32) -> Result<Program, LowerError> {
    if !(int_words(graph)..=20).contains(&w) {
        return Err(LowerError::Unsupported(format!(
            "{w}-word integers, outside {}..=20 words",
            int_words(graph)
        )));
    }
    let mut lowering = Lowering {
        graph,
        env: graph.bindings.clone(),
        subst: BTreeMap::new(),
        int_words: w,
        leaves: Leaves {
            add: leaves::add_mod(),
            sub: leaves::sub_mod(),
            neg: leaves::neg_mod(),
            scale: leaves::scale_mod(),
            poly_mul_acc: leaves::poly_mul_acc(),
            int_constant: leaves::int_constant(w),
            int_binary: leaves::int_binary(w),
            int_compare: leaves::int_compare(w),
            bit_extract: leaves::bit_extract(w),
            bool_to_int: leaves::bool_to_int(w),
            transpose: leaves::transpose(),
            automorphism: leaves::automorphism(),
            multiply_monomial: leaves::multiply_monomial(w),
            uniform_mod: leaves::uniform_mod(),
            interval_mod: leaves::interval_mod(),
            lift_int: leaves::lift_int(w),
            threshold_decode: leaves::threshold_decode(w),
            crt_map: leaves::crt_map(),
            fast_base_conversion: leaves::fast_base_conversion(),
            ntt: leaves::ntt(),
            crt_values: leaves::crt_values(w),
            int_residues: leaves::int_residues(w),
            int_mat_vec: leaves::int_mat_vec(w),
            round_lift: leaves::round_lift(),
            pack_bits: leaves::pack_bits(),
            hash_ints: leaves::hash_ints(w),
            hash_tag: leaves::hash_tag(w),
            real_op: leaves::real_op(),
            int_to_real: leaves::int_to_real(w),
            cdt_mod: leaves::cdt_mod(),
            hash_uniform: leaves::hash_uniform(),
            gadget_decompose: leaves::gadget_decompose(),
            trapdoor_preimage: leaves::trapdoor_preimage(),
            statistical_trapgen: leaves::statistical_trapgen(),
            zero: leaves::zero(),
            block_diagonal: leaves::block_diagonal(),
        },
        builder: ProgramBuilder::new(graph.source.name()),
    };
    let root_id = FrozenGraphScopeId::Root;
    let source = graph.source.scope(&root_id).expect("validated graph has a root scope");
    let scope = graph.root_scope();
    // The graph's inputs are the root definition's parameters.
    let mut values: BTreeMap<WireRef, Value> = BTreeMap::new();
    let mut input_names = Vec::new();
    let mut input_types = Vec::new();
    for node in &scope.execution_order {
        if let NodeKind::Input { name, artifact: None, .. } = node.kind() {
            let wire = WireRef { node: source.node_id(node).expect("frozen node"), port: Port(0) };
            values.insert(wire, Value::Param(input_names.len()));
            input_types.push(core_type(lowering.wire_type(&root_id, wire)?, w)?);
            input_names.push(name.clone());
        }
    }
    let mut root = DefinitionBuilder::new("root", input_types);
    lowering.scope(&root_id, &mut root, &mut values)?;
    let mut output_names = Vec::new();
    let mut outputs = Vec::new();
    for (name, output) in graph.source.outputs() {
        outputs.push(*values.get(&output.value).ok_or(LowerError::Unproduced(output.value))?);
        output_names.push(name.clone());
    }
    Ok(lowering.builder.finish(input_names, root.finish(outputs), output_names))
}

impl Lowering<'_> {
    /// The encoding of `value` at the lowering's width.
    fn int_bytes(&self, value: &BigInt) -> Result<Vec<u8>, LowerError> {
        let words = self.int_words as usize;
        let mut bytes = value.to_signed_bytes_le();
        if bytes.len() > 8 * words {
            return Err(LowerError::Unsupported(format!(
                "the integer {value} above the lowering's {words}-word integers"
            )));
        }
        let fill = if value.sign() == num_bigint::Sign::Minus { 0xff } else { 0 };
        bytes.resize(8 * words, fill);
        Ok(bytes)
    }

    /// The integer `value` as a constant of the lowering's width.
    fn int_value(
        &self,
        definition: &mut DefinitionBuilder,
        value: &BigInt,
    ) -> Result<Value, LowerError> {
        let leaf =
            leaves::constant(CoreType::Int { words: self.int_words }, self.int_bytes(value)?);
        Ok(definition.builtin(leaf, Vec::new()))
    }

    /// The family of the integers `values` as a constant of the lowering's width.
    fn int_family(
        &self,
        definition: &mut DefinitionBuilder,
        values: &[BigInt],
    ) -> Result<Value, LowerError> {
        let bytes = values
            .iter()
            .map(|value| self.int_bytes(value))
            .collect::<Result<Vec<_>, _>>()?
            .concat();
        let ty = CoreType::family(CoreType::Int { words: self.int_words }, values.len() as u64);
        let leaf = leaves::constant(ty, bytes);
        Ok(definition.builtin(leaf, Vec::new()))
    }

    fn wire_type(
        &self,
        scope: &FrozenGraphScopeId,
        wire: WireRef,
    ) -> Result<&ConcreteWireType, LowerError> {
        self.graph
            .scope(scope)
            .and_then(|s| s.wire_types.get(&wire))
            .ok_or(LowerError::MissingType(wire))
    }

    /// A compile expression of the graph as a core expression: evaluated at the bindings unless it
    /// reads a loop index.
    fn expr(&self, expr: &GraphExpr) -> Result<IntExpr, LowerError> {
        let fail = |e: String| LowerError::Expression(e);
        let substituted = substitute_vars(expr, &self.subst);
        let expr = &substituted;
        if !expr.contains_loop_index() {
            let value = expr.evaluate_in(self).map_err(|e| fail(e.to_string()))?;
            return i128::try_from(value).map(IntExpr::Const).map_err(|e| fail(e.to_string()));
        }
        let pair =
            |a: &GraphExpr, b: &GraphExpr| Ok::<_, LowerError>((self.expr(a)?, self.expr(b)?));
        Ok(match expr {
            GraphExpr::LoopIndex(slot) => IntExpr::LoopIndex(core_slot(*slot)),
            GraphExpr::Add(a, b) => {
                let (a, b) = pair(a, b)?;
                a + b
            }
            GraphExpr::Sub(a, b) => {
                let (a, b) = pair(a, b)?;
                a - b
            }
            GraphExpr::Mul(a, b) => {
                let (a, b) = pair(a, b)?;
                a * b
            }
            GraphExpr::FloorDiv(a, b) => {
                let (a, b) = pair(a, b)?;
                a.floor_div(b)
            }
            GraphExpr::Rem(a, b) => {
                let (a, b) = pair(a, b)?;
                a.floor_rem(b)
            }
            GraphExpr::Select { selector, branches } => IntExpr::Select {
                selector: Box::new(self.expr(selector)?),
                branches: branches.iter().map(|b| self.expr(b)).collect::<Result<_, _>>()?,
            },
            other => {
                return Err(LowerError::Unsupported(format!("the loop-index expression {other:?}")))
            }
        })
    }

    /// Lowers the nodes of `scope_id` into `definition`. `values` holds the scope's inputs.
    fn scope(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        definition: &mut DefinitionBuilder,
        values: &mut BTreeMap<WireRef, Value>,
    ) -> Result<(), LowerError> {
        let source = self.graph.source.scope(scope_id).expect("validated scope");
        let scope = self.graph.scope(scope_id).expect("validated scope");
        for node in &scope.execution_order {
            let id = source.node_id(node).expect("frozen node");
            let out = WireRef { node: id, port: Port(0) };
            if values.contains_key(&out) {
                continue;
            }
            if *scope_id != FrozenGraphScopeId::Root &&
                node.output_types().iter().any(|t| t.contains_loop_index())
            {
                return Err(LowerError::Unsupported(
                    "a loop body whose types depend on the loop index".into(),
                ));
            }
            let arguments: Vec<WireRef> = source.arguments(node).expect("frozen arguments");
            let operands = arguments
                .iter()
                .map(|wire| values.get(wire).copied().ok_or(LowerError::Unproduced(*wire)))
                .collect::<Result<Vec<_>, _>>()?;
            let outputs: Vec<Value> = match node.kind() {
                NodeKind::Input { .. } => {
                    return Err(LowerError::Unsupported("an input inside a loop body".into()))
                }
                NodeKind::ConstantInt(value) => vec![self.int_value(definition, value)?],
                NodeKind::ConstantReal(value) => {
                    let value = value
                        .evaluate_f64(&self.env)
                        .map_err(|e| LowerError::Expression(e.to_string()))?;
                    let leaf =
                        leaves::constant(leaves::REAL, value.to_bits().to_le_bytes().to_vec());
                    vec![definition.builtin(leaf, Vec::new())]
                }
                NodeKind::IntToReal => {
                    definition.leaf(&self.leaves.int_to_real, vec![self.int_words.into()], operands)
                }
                NodeKind::RealBinary(op) => {
                    let code: i64 = match op {
                        RealBinaryOp::Add => 0,
                        RealBinaryOp::Subtract => 1,
                        RealBinaryOp::Multiply => 2,
                        RealBinaryOp::Divide => 3,
                    };
                    definition.leaf(&self.leaves.real_op, vec![code.into()], operands)
                }
                NodeKind::RealSqrt => definition.leaf(
                    &self.leaves.real_op,
                    vec![4.into()],
                    vec![operands[0], operands[0]],
                ),
                NodeKind::ConstantBool(value) => {
                    let leaf =
                        leaves::constant(CoreType::Bool, u64::from(*value).to_le_bytes().to_vec());
                    vec![definition.builtin(leaf, Vec::new())]
                }
                NodeKind::EvaluateInt(expr) => {
                    if expr.contains_loop_index() {
                        let value = self.expr(expr)?;
                        definition.leaf(
                            &self.leaves.int_constant,
                            vec![self.int_words.into(), value],
                            Vec::new(),
                        )
                    } else {
                        let value = expr
                            .evaluate_in(self)
                            .map_err(|e| LowerError::Expression(e.to_string()))?;
                        vec![self.int_value(definition, &value)?]
                    }
                }
                NodeKind::FamilyPack { .. } => vec![definition.builtin(Builtin::Pack, operands)],
                NodeKind::FamilyGetStatic { index } => {
                    vec![definition.builtin(Builtin::Get { index: self.expr(index)? }, operands)]
                }
                NodeKind::FamilyGetDynamic => {
                    vec![definition.builtin(Builtin::GetDynamic, operands)]
                }
                NodeKind::Select { .. } => vec![definition.builtin(Builtin::Select, operands)],
                NodeKind::IntBinary(op) => {
                    let code: i64 = match op {
                        IntBinaryOp::Add => 0,
                        IntBinaryOp::Subtract => 1,
                        IntBinaryOp::Multiply => 2,
                        IntBinaryOp::Divide => 3,
                        IntBinaryOp::Remainder => 4,
                    };
                    definition.leaf(
                        &self.leaves.int_binary,
                        vec![self.int_words.into(), code.into()],
                        operands,
                    )
                }
                NodeKind::IntCompare(op) => {
                    let code: i64 = match op {
                        IntCompareOp::Equal => 0,
                        IntCompareOp::Less => 1,
                        IntCompareOp::LessEqual => 2,
                    };
                    definition.leaf(
                        &self.leaves.int_compare,
                        vec![self.int_words.into(), code.into()],
                        operands,
                    )
                }
                NodeKind::BitExtract { bit } => definition.leaf(
                    &self.leaves.bit_extract,
                    vec![self.int_words.into(), self.expr(bit)?],
                    operands,
                ),
                NodeKind::BoolToInt => {
                    definition.leaf(&self.leaves.bool_to_int, vec![self.int_words.into()], operands)
                }
                NodeKind::ConstantMatrix { value, .. } => {
                    let ConcreteWireType::Matrix(matrix) = self.wire_type(scope_id, out)?.clone()
                    else {
                        return Err(LowerError::Unsupported(
                            "a constant that is not a matrix".into(),
                        ));
                    };
                    let bytes = self.constant_matrix(&matrix, value)?;
                    let leaf = leaves::constant(matrix_type(&matrix), bytes);
                    vec![definition.builtin(leaf, Vec::new())]
                }
                NodeKind::UniformResidueSample { .. } |
                NodeKind::UniformIntervalSample { .. } |
                NodeKind::GaussianSample { .. } => {
                    vec![self.sample(scope_id, node.kind(), out, definition)?]
                }
                NodeKind::GadgetTrapdoor { base, .. } => {
                    let ty = self.wire_type(scope_id, out)?.clone();
                    let Some((public, None)) = trapdoor_parts(&ty) else {
                        return Err(LowerError::Unsupported(format!("a trapdoor of type {ty:?}")));
                    };
                    let gadget = ConstantMatrix::Gadget { base: base.clone(), small: false };
                    let bytes = self.constant_matrix(&public, &gadget)?;
                    vec![
                        definition
                            .builtin(leaves::constant(matrix_type(&public), bytes), Vec::new()),
                    ]
                }
                NodeKind::TrapdoorPublic => {
                    let ty = self.wire_type(scope_id, arguments[0])?.clone();
                    match trapdoor_parts(&ty) {
                        Some((_, None)) => vec![operands[0]],
                        Some((public, Some(_))) => {
                            let length = (public.rows *
                                public.columns *
                                public.ring.ring_dimension() as usize *
                                8) as i128;
                            let input = core_type(&ty, self.int_words)?;
                            vec![limb_loop(
                                &mut self.builder,
                                definition,
                                vec![input],
                                &[true],
                                public.ring.crt_depth(),
                                vec![operands[0]],
                                Box::new(move |body: &mut DefinitionBuilder, limbs: Vec<Value>| {
                                    body.builtin(
                                        Builtin::Window {
                                            offset: 0.into(),
                                            runs: 1.into(),
                                            length: length.into(),
                                            stride: 0.into(),
                                        },
                                        limbs,
                                    )
                                }),
                            )]
                        }
                        None => {
                            return Err(LowerError::Unsupported(format!(
                                "a trapdoor of type {ty:?}"
                            )));
                        }
                    }
                }
                // Under the gadget trapdoor a preimage is the target's gadget decomposition, as the
                // executors compute it; under a sampled trapdoor it is drawn as the CPU backend
                // draws it.
                NodeKind::PreimageSample { max_coefficient_bound, .. } => {
                    let trapdoor = self.wire_type(scope_id, arguments[1])?.clone();
                    let ConcreteWireType::Trapdoor { gadget_base, digit_count, .. } = &trapdoor
                    else {
                        unreachable!("validated trapdoor")
                    };
                    let ConcreteWireType::Matrix(target) =
                        self.wire_type(scope_id, arguments[2])?.clone()
                    else {
                        unreachable!("validated target")
                    };
                    let base = u64::try_from(gadget_base)
                        .map_err(|e| LowerError::Expression(e.to_string()))?;
                    match trapdoor_parts(&trapdoor) {
                        Some((_, None)) => vec![self.gadget_decompose(
                            definition,
                            &target,
                            operands[2],
                            base,
                            *digit_count,
                            false,
                        )?],
                        Some((public, Some(_))) => {
                            let bound = max_coefficient_bound
                                .evaluate_in(self)
                                .map_err(|e| LowerError::Expression(e.to_string()))?;
                            vec![self.trapdoor_preimage(
                                &trapdoor,
                                &public,
                                &target,
                                &bound,
                                definition,
                                [operands[1], operands[2]],
                            )?]
                        }
                        None => {
                            return Err(LowerError::Unsupported(format!(
                                "a preimage under a trapdoor of type {trapdoor:?}"
                            )));
                        }
                    }
                }
                NodeKind::TrapdoorSample {
                    matrix_type: node_matrix, sigma, gadget_base, ..
                } => {
                    let ty = self.wire_type(scope_id, WireRef { node: id, port: Port(1) })?.clone();
                    self.trapdoor_sample(&ty, node_matrix, sigma, gadget_base, definition)?
                }
                NodeKind::HashSample {
                    variant,
                    tag_prefix,
                    tag_components,
                    base,
                    digit_count,
                    ..
                } => {
                    let matrix = match self.wire_type(scope_id, out)? {
                        ConcreteWireType::Matrix(m) |
                        ConcreteWireType::SmallMatrix { matrix: m, .. } => m.clone(),
                        _ => {
                            return Err(LowerError::Unsupported(
                                "a hash sample that is not a matrix".into(),
                            ))
                        }
                    };
                    let (tag, tag_words) =
                        self.hash_tag(definition, tag_prefix, tag_components, &operands)?;
                    match (variant, base, digit_count) {
                        (HashVariant::Plain, None, _) => {
                            vec![self.hash_matrix(definition, &matrix, operands[0], tag, tag_words)]
                        }
                        (
                            HashVariant::Decomposed | HashVariant::SmallDecomposed,
                            Some(base),
                            Some(digit_count),
                        ) => {
                            // The decomposition of a hash sample of `rows / digit_count` rows.
                            let evaluate = |e: &GraphExpr| {
                                e.evaluate_in(self)
                                    .map_err(|e| LowerError::Expression(e.to_string()))
                            };
                            let base = u64::try_from(evaluate(base)?)
                                .map_err(|e| LowerError::Expression(e.to_string()))?;
                            let digit_count = usize::try_from(evaluate(digit_count)?)
                                .map_err(|e| LowerError::Expression(e.to_string()))?;
                            if digit_count == 0 || !matrix.rows.is_multiple_of(digit_count) {
                                return Err(LowerError::Unsupported(
                                    "decomposed hash rows not divisible by the digits".into(),
                                ));
                            }
                            let source =
                                ConcreteMatrixType { rows: matrix.rows / digit_count, ..matrix };
                            let sample =
                                self.hash_matrix(definition, &source, operands[0], tag, tag_words);
                            vec![self.gadget_decompose(
                                definition,
                                &source,
                                sample,
                                base,
                                digit_count,
                                *variant == HashVariant::SmallDecomposed,
                            )?]
                        }
                        _ => {
                            return Err(LowerError::Unsupported(
                                "a hash sample with inconsistent decomposition metadata".into(),
                            ))
                        }
                    }
                }
                NodeKind::GadgetDecompose { base, small, digit_count } => {
                    let ConcreteWireType::Matrix(input) =
                        self.wire_type(scope_id, arguments[0])?.clone()
                    else {
                        return Err(LowerError::Unsupported(
                            "decomposing a value that is not a matrix".into(),
                        ));
                    };
                    let evaluate = |e: &GraphExpr| {
                        e.evaluate_in(self).map_err(|e| LowerError::Expression(e.to_string()))
                    };
                    let base = u64::try_from(evaluate(base)?)
                        .map_err(|e| LowerError::Expression(e.to_string()))?;
                    let digit_count = usize::try_from(evaluate(digit_count)?)
                        .map_err(|e| LowerError::Expression(e.to_string()))?;
                    vec![self.gadget_decompose(
                        definition,
                        &input,
                        operands[0],
                        base,
                        digit_count,
                        *small,
                    )?]
                }
                NodeKind::LiftIntegerToConstantPolynomial { .. } => {
                    let ConcreteWireType::Matrix(matrix) = self.wire_type(scope_id, out)?.clone()
                    else {
                        return Err(LowerError::Unsupported("a lift that is not a matrix".into()));
                    };
                    let n = matrix.ring.ring_dimension() as usize;
                    let leaf = self.leaves.lift_int.clone();
                    let modulus = per_limb(matrix.ring.crt_moduli().to_vec());
                    let words = self.int_words;
                    vec![limb_loop(
                        &mut self.builder,
                        definition,
                        vec![CoreType::Int { words }],
                        &[false],
                        matrix.ring.crt_depth(),
                        operands,
                        Box::new(move |body: &mut DefinitionBuilder, m: Vec<Value>| {
                            body.leaf(
                                &leaf,
                                vec![n.into(), modulus, matrix.rows.into(), words.into()],
                                m,
                            )[0]
                        }),
                    )]
                }
                NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool, family } => {
                    let ConcreteWireType::Matrix(matrix) =
                        self.wire_type(scope_id, arguments[0])?.clone()
                    else {
                        return Err(LowerError::Unsupported(
                            "decoding a value that is not a matrix".into(),
                        ));
                    };
                    let t = self.expr(plaintext_modulus)?;
                    let length = self.expr(length)?;
                    let count = usize::try_from(
                        length
                            .eval(&Default::default())
                            .map_err(|e| LowerError::Expression(e.to_string()))?,
                    )
                    .map_err(|e| LowerError::Expression(e.to_string()))?;
                    let depth = matrix.ring.crt_depth();
                    let table: Vec<u8> =
                        matrix.ring.crt_moduli().iter().flat_map(|q| q.to_le_bytes()).collect();
                    let moduli = leaves::constant(CoreType::buffer(8, depth), table);
                    let moduli = definition.builtin(moduli, Vec::new());
                    let limb_words =
                        matrix.rows * matrix.columns * matrix.ring.ring_dimension() as usize;
                    let decoded = definition.leaf(
                        &self.leaves.threshold_decode,
                        vec![
                            (matrix.ring.ring_dimension() as usize).into(),
                            depth.into(),
                            t,
                            length,
                            limb_words.into(),
                            self.int_words.into(),
                        ],
                        vec![operands[0], moduli],
                    )[0];
                    if *family {
                        vec![decoded]
                    } else {
                        let zero = self.int_value(definition, &BigInt::from(0))?;
                        (0..count)
                            .map(|k| {
                                let value = definition
                                    .builtin(Builtin::Get { index: k.into() }, vec![decoded]);
                                if *output_bool {
                                    definition.leaf(
                                        &self.leaves.int_compare,
                                        vec![self.int_words.into(), 1.into()],
                                        vec![zero, value],
                                    )[0]
                                } else {
                                    value
                                }
                            })
                            .collect()
                    }
                }
                NodeKind::ParallelLoop(lp) => {
                    self.parallel(scope_id, id, lp, definition, operands, &arguments)?
                }
                NodeKind::SubgraphCall(call) => {
                    let body_id = self.graph.source.child_scope_id(scope_id, id).expect("subgraph");
                    let params = arguments
                        .iter()
                        .map(|wire| core_type(self.wire_type(scope_id, *wire)?, self.int_words))
                        .collect::<Result<Vec<_>, _>>()?;
                    let (body, outputs) =
                        self.body(&body_id, &call.bindings, params, |d, i| d.param(i))?;
                    definition
                        .add_loop(
                            Loop {
                                count: 1.into(),
                                index_slot: SUBGRAPH,
                                body,
                                bindings: Vec::new(),
                                carried: Vec::new(),
                                inputs: operands,
                                until: false,
                            },
                            outputs,
                        )
                        .into_iter()
                        .map(|family| {
                            definition.builtin(Builtin::Get { index: 0.into() }, vec![family])
                        })
                        .collect()
                }
                NodeKind::SequentialLoop(lp) => {
                    self.sequential(scope_id, id, lp, definition, operands, &arguments)?
                }
                NodeKind::Sha256BitColumns { seed, rows, columns, index_bits } => {
                    // A function of the node's metadata only: constants.
                    let (packed, counts) =
                        crate::hash_columns::sha256_bit_columns(seed, *rows, *columns, *index_bits);
                    vec![
                        self.int_family(definition, &packed)?,
                        self.int_family(definition, &counts)?,
                    ]
                }
                NodeKind::HashIntFamily { count, modulus, tag_prefix, tag_components } => {
                    let evaluate = |e: &GraphExpr| {
                        e.evaluate_in(self).map_err(|e| LowerError::Expression(e.to_string()))
                    };
                    let (count, modulus) = (evaluate(count)?, evaluate(modulus)?);
                    let bits = modulus.bits().saturating_sub(1) as usize;
                    if bits == 0 || modulus != BigInt::from(1) << bits {
                        return Err(LowerError::Unsupported(
                            "hash integers modulo a non-power of two".into(),
                        ));
                    }
                    if bits >= 64 * self.int_words as usize {
                        return Err(LowerError::Unsupported(
                            "hash integers above the lowering's integers".into(),
                        ));
                    }
                    let count = usize::try_from(count)
                        .map_err(|e| LowerError::Expression(e.to_string()))?;
                    let width = bits.div_ceil(8);
                    let high_mask: u64 =
                        if bits.is_multiple_of(8) { 0xff } else { (1 << (bits % 8)) - 1 };
                    let (tag_value, tag_words) =
                        self.hash_tag(definition, tag_prefix, tag_components, &operands)?;
                    definition.leaf(
                        &self.leaves.hash_ints,
                        vec![
                            count.into(),
                            width.into(),
                            high_mask.into(),
                            tag_words.into(),
                            self.int_words.into(),
                        ],
                        vec![operands[0], tag_value],
                    )
                }
                NodeKind::CrtRecompose {
                    plaintext_moduli, reconstruction_coefficients, ..
                } => {
                    vec![self.crt_recompose(
                        scope_id,
                        out,
                        &arguments,
                        plaintext_moduli,
                        reconstruction_coefficients,
                        definition,
                        operands,
                    )?]
                }
                NodeKind::PackPolynomialCoefficients { coefficient_bits, .. } => {
                    let ConcreteWireType::Matrix(matrix) = self.wire_type(scope_id, out)?.clone()
                    else {
                        return Err(LowerError::Unsupported("packing into a non-matrix".into()));
                    };
                    let bits = usize::try_from(
                        coefficient_bits
                            .evaluate_in(self)
                            .map_err(|e| LowerError::Expression(e.to_string()))?,
                    )
                    .map_err(|e| LowerError::Expression(e.to_string()))?;
                    if bits > 64 * leaves::CRT_MAP_WORDS {
                        return Err(LowerError::Unsupported(format!("{bits}-bit coefficients")));
                    }
                    let moduli = matrix.ring.crt_moduli().to_vec();
                    let bytes = moduli.iter().flat_map(|q| q.to_le_bytes()).collect();
                    let table = leaves::constant(CoreType::buffer(8, moduli.len()), bytes);
                    let table = definition.builtin(table, Vec::new());
                    definition.leaf(
                        &self.leaves.pack_bits,
                        vec![
                            (matrix.ring.ring_dimension() as usize).into(),
                            bits.into(),
                            moduli.len().into(),
                        ],
                        vec![operands[0], table],
                    )
                }
                NodeKind::PolynomialValues { .. } |
                NodeKind::PolynomialFromValues { .. } |
                NodeKind::ExtractCoefficient { .. } => vec![self.polynomial_values(
                    scope_id,
                    node.kind(),
                    out,
                    arguments[0],
                    definition,
                    operands[0],
                )?],
                NodeKind::IntMatrixVectorProduct { transpose } => {
                    let count = |wire: WireRef| match self.wire_type(scope_id, wire) {
                        Ok(ConcreteWireType::IndexedFamily { count, .. }) => {
                            usize::try_from(*count)
                                .map_err(|e| LowerError::Expression(e.to_string()))
                        }
                        _ => Err(LowerError::Unsupported("a product of non-families".into())),
                    };
                    let (entries, inner) = (count(arguments[0])?, count(arguments[1])?);
                    let outer = entries.checked_div(inner).unwrap_or(0);
                    definition.leaf(
                        &self.leaves.int_mat_vec,
                        vec![
                            outer.into(),
                            inner.into(),
                            usize::from(*transpose).into(),
                            self.int_words.into(),
                        ],
                        operands,
                    )
                }
                NodeKind::ModulusSwitch { .. } |
                NodeKind::ModulusReduce { .. } |
                NodeKind::CenteredRebase { .. } |
                NodeKind::CenteredRoundDivide { .. } |
                NodeKind::BlockModSwitch { .. } |
                NodeKind::RnsModUp { .. } |
                NodeKind::RnsModDown { .. } => {
                    vec![self.ring_conversion(
                        scope_id,
                        node.kind(),
                        out,
                        arguments[0],
                        definition,
                        operands[0],
                    )?]
                }
                kind => vec![
                    self.matrix_operation(scope_id, kind, out, &arguments, definition, operands)?,
                ],
            };
            for (port, value) in outputs.into_iter().enumerate() {
                values.insert(WireRef { node: id, port: Port(port as u32) }, value);
            }
        }
        Ok(())
    }

    /// A body definition, lowered with the bindings of its call or loop in scope: its parameters,
    /// how each scope input reads them, and its outputs. A binding whose value does not depend on
    /// a loop index joins the parameters; any other is substituted into the body's expressions.
    fn body(
        &mut self,
        body_id: &FrozenGraphScopeId,
        bindings: &[(String, GraphExpr)],
        params: Vec<CoreType>,
        read: impl Fn(&mut DefinitionBuilder, usize) -> Value,
    ) -> Result<(mxx_core::program::DefinitionId, usize), LowerError> {
        let (env, subst) = (self.env.clone(), self.subst.clone());
        for (name, expr) in bindings {
            let expr = substitute_vars(expr, &subst);
            if expr.contains_loop_index() {
                self.subst.insert(name.clone(), expr);
            } else {
                let value =
                    expr.evaluate(&env).map_err(|e| LowerError::Expression(e.to_string()))?;
                self.env.integers.insert(name.clone(), value);
                self.subst.remove(name);
            }
        }
        let lowered = self.body_in_scope(body_id, params, read);
        (self.env, self.subst) = (env, subst);
        lowered
    }

    fn body_in_scope(
        &mut self,
        body_id: &FrozenGraphScopeId,
        params: Vec<CoreType>,
        read: impl Fn(&mut DefinitionBuilder, usize) -> Value,
    ) -> Result<(mxx_core::program::DefinitionId, usize), LowerError> {
        let source = self.graph.source.scope(body_id).expect("validated body");
        let mut definition = DefinitionBuilder::new(format!("{body_id:?}"), params);
        let mut values = BTreeMap::new();
        for (i, wire) in source.inputs().iter().enumerate() {
            let value = read(&mut definition, i);
            values.insert(*wire, value);
        }
        self.scope(body_id, &mut definition, &mut values)?;
        let outputs = source
            .outputs()
            .iter()
            .map(|wire| values.get(wire).copied().ok_or(LowerError::Unproduced(*wire)))
            .collect::<Result<Vec<_>, _>>()?;
        let count = outputs.len();
        Ok((self.builder.add_definition(definition.finish(outputs)), count))
    }

    fn parallel(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        node: crate::types::NodeId,
        lp: &crate::node::ParallelLoop,
        definition: &mut DefinitionBuilder,
        operands: Vec<Value>,
        arguments: &[WireRef],
    ) -> Result<Vec<Value>, LowerError> {
        let body_id = self.graph.source.child_scope_id(scope_id, node).expect("parallel body");
        let slot = core_slot(lp.index_slot);
        let params = arguments
            .iter()
            .map(|wire| core_type(self.wire_type(scope_id, *wire)?, self.int_words))
            .collect::<Result<Vec<_>, _>>()?;
        let modes = lp.input_modes.clone();
        let index = IntExpr::LoopIndex(slot);
        let member = move |mode: LoopInputMode| match mode {
            LoopInputMode::Zip => Some(index.clone()),
            LoopInputMode::ZipOffset { offset } => Some(index.clone() + offset),
            LoopInputMode::ZipQuotient { divisor } => Some(index.clone().floor_div(divisor)),
            LoopInputMode::Broadcast | LoopInputMode::Gather { .. } => None,
        };
        let (body, outputs) = self.body(&body_id, &lp.bindings, params, |d, i| {
            let param = d.param(i);
            match modes[i] {
                LoopInputMode::Broadcast => param,
                LoopInputMode::Gather { index_argument } => {
                    let family = d.param(index_argument);
                    let at = member(modes[index_argument]).expect("a gather index is zipped");
                    let index = d.builtin(Builtin::Get { index: at }, vec![family]);
                    d.builtin(Builtin::GetDynamic, vec![param, index])
                }
                mode => {
                    d.builtin(Builtin::Get { index: member(mode).expect("zipped") }, vec![param])
                }
            }
        })?;
        let count = self.expr(&lp.count)?;
        Ok(definition.add_loop(
            Loop {
                count,
                index_slot: slot,
                body,
                bindings: Vec::new(),
                carried: Vec::new(),
                inputs: operands,
                until: false,
            },
            outputs,
        ))
    }

    fn sequential(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        node: crate::types::NodeId,
        lp: &crate::node::SequentialLoop,
        definition: &mut DefinitionBuilder,
        mut operands: Vec<Value>,
        arguments: &[WireRef],
    ) -> Result<Vec<Value>, LowerError> {
        let body_id = self.graph.source.child_scope_id(scope_id, node).expect("sequential body");
        let params = arguments
            .iter()
            .map(|wire| core_type(self.wire_type(scope_id, *wire)?, self.int_words))
            .collect::<Result<Vec<_>, _>>()?;
        let (body, _) = self.body(&body_id, &lp.bindings, params, |d, i| d.param(i))?;
        let inputs = operands.split_off(lp.carried_count);
        let count = self.expr(&lp.count)?;
        Ok(definition.add_loop(
            Loop {
                count,
                index_slot: core_slot(lp.index_slot),
                body,
                bindings: Vec::new(),
                carried: operands,
                inputs,
                until: false,
            },
            0,
        ))
    }

    /// The tag of a hash sample, the prefix and then each component framed as the backends frame
    /// it, as a buffer of its byte length and its bytes, and the buffer's words. A tag whose
    /// components are all known when lowering is a constant; otherwise `leaves::hash_tag` builds
    /// it at run time from a template and the integer operands: the sampler's arguments
    /// (`operands`, the key first) and the compile-time expressions that read loop indices.
    fn hash_tag(
        &mut self,
        definition: &mut DefinitionBuilder,
        prefix: &[u8],
        components: &[HashTagComponent],
        operands: &[Value],
    ) -> Result<(Value, usize), LowerError> {
        let words = self.int_words;
        let integer = |tag: &mut Vec<u8>, value: &BigInt| {
            let (sign, bytes) = value.to_bytes_be();
            tag.push(u8::from(sign == num_bigint::Sign::Minus));
            tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
            tag.extend_from_slice(&bytes);
        };
        // The template: literal bytes, and items `[kind, operand]` with their largest encoding.
        let mut template: Vec<u64> = Vec::new();
        let mut literal: Vec<u8> = prefix.to_vec();
        let mut dynamic: Vec<Value> = Vec::new();
        let mut capacity = 0usize;
        let flush = |template: &mut Vec<u64>, literal: &mut Vec<u8>, capacity: &mut usize| {
            if !literal.is_empty() {
                template.push(0);
                template.push(literal.len() as u64);
                template.extend(literal.chunks(8).map(|c| {
                    let mut w = [0u8; 8];
                    w[..c.len()].copy_from_slice(c);
                    u64::from_le_bytes(w)
                }));
                *capacity += literal.len();
                literal.clear();
            }
        };
        for component in components {
            // The kind and value of a component that is not a literal.
            let runtime: Option<(u64, Value)> = match component {
                HashTagComponent::Bytes(bytes) => {
                    literal.push(0);
                    literal.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
                    literal.extend_from_slice(bytes);
                    None
                }
                HashTagComponent::Integer(e) |
                HashTagComponent::Decimal(e) |
                HashTagComponent::U64Le(e) => {
                    let kind = match component {
                        HashTagComponent::Integer(_) => 1,
                        HashTagComponent::Decimal(_) => 2,
                        _ => 3,
                    };
                    if e.contains_loop_index() {
                        let value = self.expr(e)?;
                        let value = definition.leaf(
                            &self.leaves.int_constant,
                            vec![words.into(), value],
                            Vec::new(),
                        )[0];
                        Some((kind, value))
                    } else {
                        let value = e
                            .evaluate_in(self)
                            .map_err(|e| LowerError::Expression(e.to_string()))?;
                        match kind {
                            1 => {
                                literal.push(1);
                                integer(&mut literal, &value);
                            }
                            2 => {
                                let decimal = value.to_string();
                                literal.push(2);
                                literal.extend_from_slice(&(decimal.len() as u64).to_be_bytes());
                                literal.extend_from_slice(decimal.as_bytes());
                            }
                            _ => {
                                let value = u64::try_from(value)
                                    .map_err(|e| LowerError::Expression(e.to_string()))?;
                                literal.push(3);
                                literal.extend_from_slice(&value.to_le_bytes());
                            }
                        }
                        None
                    }
                }
                HashTagComponent::Operand(index) => Some((1, operands[*index])),
                HashTagComponent::OperandU64Le(index) => Some((3, operands[*index])),
            };
            if let Some((kind, value)) = runtime {
                flush(&mut template, &mut literal, &mut capacity);
                template.push(kind);
                template.push(dynamic.len() as u64);
                dynamic.push(value);
                capacity += match kind {
                    1 => 10 + 8 * words as usize,
                    2 => 11 + 20 * words as usize,
                    _ => 9,
                };
            }
        }
        if dynamic.is_empty() {
            let mut bytes = (literal.len() as u64).to_le_bytes().to_vec();
            bytes.extend_from_slice(&literal);
            bytes.resize(bytes.len().div_ceil(8) * 8, 0);
            let tag_words = bytes.len() / 8;
            let leaf = leaves::constant(CoreType::buffer(8, tag_words), bytes);
            return Ok((definition.builtin(leaf, Vec::new()), tag_words));
        }
        flush(&mut template, &mut literal, &mut capacity);
        let tag_words = 1 + capacity.div_ceil(8);
        let bytes = template.iter().flat_map(|w| w.to_le_bytes()).collect();
        let leaf = leaves::constant(CoreType::buffer(8, template.len()), bytes);
        let template_value = definition.builtin(leaf, Vec::new());
        let count = dynamic.len();
        let family = definition.builtin(Builtin::Pack, dynamic);
        let tag = definition.leaf(
            &self.leaves.hash_tag,
            vec![words.into(), template.len().into(), tag_words.into(), count.into()],
            vec![template_value, family],
        )[0];
        Ok((tag, tag_words))
    }

    /// The residues of a constant matrix, in the core encoding.
    fn constant_matrix(
        &self,
        matrix: &ConcreteMatrixType,
        value: &ConstantMatrix,
    ) -> Result<Vec<u8>, LowerError> {
        let evaluate =
            |e: &GraphExpr| e.evaluate_in(self).map_err(|e| LowerError::Expression(e.to_string()));
        let index = |e: &GraphExpr| -> Result<usize, LowerError> {
            usize::try_from(evaluate(e)?).map_err(|e| LowerError::Expression(e.to_string()))
        };
        let n = matrix.ring.ring_dimension() as usize;
        let (rows, columns) = (matrix.rows, matrix.columns);
        // Every entry's integer coefficients, row-major.
        let mut entries = vec![vec![BigInt::from(0); n]; rows * columns];
        let shape = |ok: bool| {
            if ok {
                Ok(())
            } else {
                Err(LowerError::Unsupported(format!("{value:?} at {rows}x{columns}")))
            }
        };
        match value {
            ConstantMatrix::Zero => {}
            ConstantMatrix::Identity => {
                shape(rows == columns)?;
                for i in 0..rows {
                    entries[i * columns + i][0] = BigInt::from(1);
                }
            }
            ConstantMatrix::UnitRow { index: i } => {
                shape(rows == 1)?;
                entries[index(i)?][0] = BigInt::from(1);
            }
            ConstantMatrix::UnitColumn { index: i } => {
                shape(columns == 1)?;
                entries[index(i)?][0] = BigInt::from(1);
            }
            ConstantMatrix::Gadget { base, small } => {
                // The CRT gadgets, as the backends build them. Each row `i` has a block of `D`
                // digits. Without `small`, the block has, for each of the first `D / k` limbs `j`,
                // `k` entries `e_j base^d`, where `e_j` is the CRT idempotent of limb `j` (1 modulo
                // `q_j`, 0 modulo the others) and `k` the digits of one tower; with `small`, it is
                // `(1, base, ..., base^(k-1))`.
                let base = evaluate(base)?;
                let bits = matrix
                    .ring
                    .crt_moduli()
                    .iter()
                    .map(|q| 64 - q.leading_zeros() as u64)
                    .max()
                    .unwrap_or(0);
                let base_bits = base.bits().saturating_sub(1);
                shape(base_bits > 0 && base == BigInt::from(1) << base_bits)?;
                let per_tower = bits.div_ceil(base_bits) as usize;
                shape(columns.is_multiple_of(rows))?;
                let digits = columns / rows;
                let modulus = matrix.ring.modulus();
                if *small {
                    shape(digits == per_tower)?;
                    for i in 0..rows {
                        let mut power = BigInt::from(1);
                        for d in 0..digits {
                            entries[i * columns + i * digits + d][0] = power.mod_floor(&modulus);
                            power *= &base;
                        }
                    }
                } else {
                    let towers = digits / per_tower;
                    shape(
                        digits.is_multiple_of(per_tower) &&
                            towers >= 1 &&
                            towers <= matrix.ring.crt_depth(),
                    )?;
                    for (j, q) in matrix.ring.crt_moduli().iter().take(towers).enumerate() {
                        let q = BigInt::from(*q);
                        let cofactor = &modulus / &q;
                        let inverse = cofactor.modpow(&(&q - 2), &q);
                        let idempotent = (&cofactor * inverse).mod_floor(&modulus);
                        for i in 0..rows {
                            let mut power = BigInt::from(1);
                            for k in 0..per_tower {
                                entries[i * columns + i * digits + j * per_tower + k][0] =
                                    (&idempotent * &power).mod_floor(&modulus);
                                power *= &base;
                            }
                        }
                    }
                }
            }
            ConstantMatrix::Rotation { exponent } => {
                shape(rows == 1 && columns == 1)?;
                let exponent = index(exponent)?;
                shape(exponent < n)?;
                entries[0][exponent] = BigInt::from(1);
            }
            ConstantMatrix::PowerOfBase { base, exponent } => {
                shape(rows == 1 && columns == 1)?;
                let exponent = u32::try_from(index(exponent)?)
                    .map_err(|e| LowerError::Expression(e.to_string()))?;
                entries[0][0] = evaluate(base)?.pow(exponent);
            }
            ConstantMatrix::Polynomial { coefficients } => {
                shape(rows == 1 && columns == 1 && coefficients.len() <= n)?;
                for (k, c) in coefficients.iter().enumerate() {
                    entries[0][k] = evaluate(c)?;
                }
            }
            ConstantMatrix::AllCoefficients { value } => {
                let value = evaluate(value)?;
                for entry in &mut entries {
                    entry.fill(value.clone());
                }
            }
        }
        let mut bytes = Vec::with_capacity(8 * n * rows * columns * matrix.ring.crt_depth());
        for q in matrix.ring.crt_moduli() {
            let q = BigInt::from(*q);
            for entry in &entries {
                for coefficient in entry {
                    let residue =
                        u64::try_from(coefficient.mod_floor(&q)).expect("a residue fits a word");
                    bytes.extend_from_slice(&residue.to_le_bytes());
                }
            }
        }
        Ok(bytes)
    }

    /// A uniform sample: per limb from its own coins for residues uniform modulo `Q`, or from
    /// coins every limb shares for integers uniform on an interval.
    fn sample(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        kind: &NodeKind,
        out: WireRef,
        definition: &mut DefinitionBuilder,
    ) -> Result<Value, LowerError> {
        let matrix = match self.wire_type(scope_id, out)? {
            ConcreteWireType::Matrix(matrix) | ConcreteWireType::SmallMatrix { matrix, .. } => {
                matrix.clone()
            }
            _ => return Err(LowerError::Unsupported("a sample that is not a matrix".into())),
        };
        self.sample_matrix(kind, &matrix, definition)
    }

    /// A computational (MP12) trapdoor of type `ty`: `R` and `E` Gaussian of width `sigma`,
    /// truncated at `13 sigma` (within `2^-120` of the untruncated width in statistical distance),
    /// `ā` uniform, and the public matrix `A = [ā | I | G - (ā R + E)]`, so that
    /// `A [R; E; I] = G`; returns the public matrix and the trapdoor (per limb, `A ‖ R ‖ E`).
    fn trapdoor_sample(
        &mut self,
        ty: &ConcreteWireType,
        node_matrix: &crate::types::MatrixType,
        sigma: &crate::RealExpr,
        gadget_base: &GraphExpr,
        definition: &mut DefinitionBuilder,
    ) -> Result<Vec<Value>, LowerError> {
        let Some((public, Some(secret))) = trapdoor_parts(ty) else {
            return Err(LowerError::Unsupported(format!("a trapdoor sample of type {ty:?}")));
        };
        if let ConcreteWireType::Trapdoor {
            instantiation: TrapdoorInstantiation::Statistical { .. },
            ..
        } = ty
        {
            return self.statistical_trapdoor(
                ty,
                &public,
                &secret,
                node_matrix,
                gadget_base,
                definition,
            );
        }
        let mut r = secret;
        r.rows = public.rows;
        let width =
            sigma.evaluate_f64(&self.env).map_err(|e| LowerError::Expression(e.to_string()))?;
        let bound = BigInt::from((13.0 * width).ceil() as u64);
        let gaussian = NodeKind::GaussianSample {
            matrix_type: node_matrix.clone(),
            sigma: sigma.clone(),
            max_coefficient_bound: GraphExpr::constant(bound),
            compact: false,
        };
        let rows = public.rows;
        let mut square = public.clone();
        square.columns = rows;
        let m = |t: &ConcreteMatrixType| ConcreteWireType::Matrix(t.clone());
        let r_value = self.sample_matrix(&gaussian, &r, definition)?;
        let e_value = self.sample_matrix(&gaussian, &r, definition)?;
        let uniform = NodeKind::UniformResidueSample { matrix_type: node_matrix.clone() };
        let a_bar = self.sample_matrix(&uniform, &square, definition)?;
        let gadget = ConstantMatrix::Gadget { base: gadget_base.clone(), small: false };
        let g_bytes = self.constant_matrix(&r, &gadget)?;
        let g = definition.builtin(leaves::constant(matrix_type(&r), g_bytes), Vec::new());
        let i_bytes = self.constant_matrix(&square, &ConstantMatrix::Identity)?;
        let identity =
            definition.builtin(leaves::constant(matrix_type(&square), i_bytes), Vec::new());
        let multiply = NodeKind::MatrixBinary(MatrixBinaryOp::Multiply);
        let a_r =
            self.matrix_op(&multiply, &r, &[m(&square), m(&r)], definition, vec![a_bar, r_value])?;
        let add = NodeKind::MatrixBinary(MatrixBinaryOp::Add);
        let a_r_e = self.matrix_op(&add, &r, &[m(&r), m(&r)], definition, vec![a_r, e_value])?;
        let subtract = NodeKind::MatrixBinary(MatrixBinaryOp::Subtract);
        let a1 = self.matrix_op(&subtract, &r, &[m(&r), m(&r)], definition, vec![g, a_r_e])?;
        let concat = NodeKind::Concat { axis: ConcatAxis::Columns };
        let a = self.matrix_op(
            &concat,
            &public,
            &[m(&square), m(&square), m(&r)],
            definition,
            vec![a_bar, identity, a1],
        )?;
        let limbs = public.ring.crt_depth();
        let trapdoor = limb_loop(
            &mut self.builder,
            definition,
            vec![matrix_type(&public), matrix_type(&r), matrix_type(&r)],
            &[true, true, true],
            limbs,
            vec![a, r_value, e_value],
            Box::new(|body: &mut DefinitionBuilder, limbs: Vec<Value>| {
                body.builtin(Builtin::Concat, limbs)
            }),
        );
        Ok(vec![a, trapdoor])
    }

    /// The per-limb body of the product `left right` of two matrices: loops over the rows `i`
    /// and columns `j` of the result, and in each a carried loop over the inner dimension `k`
    /// accumulating `A[i][k] B[k][j]` with `leaves::poly_mul_acc`.
    fn matrix_product(
        &mut self,
        n: usize,
        modulus: IntExpr,
        left: &ConcreteMatrixType,
        right: &ConcreteMatrixType,
    ) -> LimbBody<'static> {
        let entry = (8 * n) as i128;
        let (i, j, k) = (
            IntExpr::LoopIndex(PRODUCT),
            IntExpr::LoopIndex(PRODUCT - 1),
            IntExpr::LoopIndex(PRODUCT_SUM),
        );
        self.product_loops(
            n,
            modulus,
            vec![left.rows, right.columns],
            Some(left.columns),
            (i * left.columns as i128 + k.clone()) * entry,
            (k * right.columns as i128 + j) * entry,
            left.rows * left.columns * n * 8,
            right.rows * right.columns * n * 8,
        )
    }

    /// The per-limb body of a family of polynomial products over the nested loops `dims` (slot
    /// `PRODUCT - d` for dimension `d`, outermost first): each entry is `A[a_offset] B[b_offset]`
    /// (byte offsets of `N`-coefficient entries in the inputs of `a_bytes` and `b_bytes` bytes),
    /// summed over a carried loop of `sum` iterations (slot `PRODUCT_SUM`) when given, and the
    /// nested families are concatenated into the result's buffer, row-major.
    #[allow(clippy::too_many_arguments)]
    fn product_loops(
        &mut self,
        n: usize,
        modulus: IntExpr,
        dims: Vec<usize>,
        sum: Option<usize>,
        a_offset: IntExpr,
        b_offset: IntExpr,
        a_bytes: usize,
        b_bytes: usize,
    ) -> LimbBody<'static> {
        let poly = CoreType::buffer(8, n);
        let (a_ty, b_ty) = (CoreType::buffer(1, a_bytes), CoreType::buffer(1, b_bytes));
        let leaf = self.leaves.poly_mul_acc.clone();
        let zero_bytes = vec![0u8; 8 * n];
        let window = |offset: IntExpr| Builtin::Window {
            offset,
            runs: 1.into(),
            length: ((8 * n) as i128).into(),
            stride: 0.into(),
        };
        let arguments: Vec<IntExpr> = vec![n.into(), modulus];
        // The entry: one product, or the sum over the carried loop.
        let mut entry = DefinitionBuilder::new("product_entry", vec![a_ty.clone(), b_ty.clone()]);
        let (pa, pb) = (entry.param(0), entry.param(1));
        let zero = entry.builtin(leaves::constant(poly.clone(), zero_bytes.clone()), Vec::new());
        let value = match sum {
            Some(count) => {
                let mut term = DefinitionBuilder::new(
                    "product_term",
                    vec![poly.clone(), a_ty.clone(), b_ty.clone()],
                );
                let (acc, ta, tb) = (term.param(0), term.param(1), term.param(2));
                let x = term.builtin(window(a_offset.clone()), vec![ta]);
                let y = term.builtin(window(b_offset.clone()), vec![tb]);
                let next = term.leaf(&leaf, arguments.clone(), vec![acc, x, y])[0];
                let term = self.builder.add_definition(term.finish(vec![next]));
                entry.add_loop(
                    Loop {
                        count: count.into(),
                        index_slot: PRODUCT_SUM,
                        body: term,
                        bindings: Vec::new(),
                        carried: vec![zero],
                        inputs: vec![pa, pb],
                        until: false,
                    },
                    0,
                )[0]
            }
            None => {
                let x = entry.builtin(window(a_offset), vec![pa]);
                let y = entry.builtin(window(b_offset), vec![pb]);
                entry.leaf(&leaf, arguments, vec![zero, x, y])[0]
            }
        };
        let mut inner = self.builder.add_definition(entry.finish(vec![value]));
        // The loops, innermost first; the outermost is added to the limb's body.
        for d in (1..dims.len()).rev() {
            let mut level =
                DefinitionBuilder::new("product_loop", vec![a_ty.clone(), b_ty.clone()]);
            let (la, lb) = (level.param(0), level.param(1));
            let family = level.add_loop(
                Loop {
                    count: dims[d].into(),
                    index_slot: PRODUCT - d as u32,
                    body: inner,
                    bindings: Vec::new(),
                    carried: Vec::new(),
                    inputs: vec![la, lb],
                    until: false,
                },
                1,
            )[0];
            inner = self.builder.add_definition(level.finish(vec![family]));
        }
        let outer = dims[0];
        Box::new(move |body: &mut DefinitionBuilder, limbs: Vec<Value>| {
            let family = body.add_loop(
                Loop {
                    count: outer.into(),
                    index_slot: PRODUCT,
                    body: inner,
                    bindings: Vec::new(),
                    carried: Vec::new(),
                    inputs: limbs,
                    until: false,
                },
                1,
            )[0];
            body.builtin(Builtin::Concat, vec![family])
        })
    }

    /// The shape of the preimages of `columns` columns under the sampled trapdoor of type `ty`
    /// (public matrix of type `public`), its gadget base, TrapGen width `σ`, and preimage width:
    /// the trapdoor's, or the CPU sampler's minimum, `1.8 (b + 1) σ² (sqrt(n rows D) + sqrt(2 n) +
    /// 4.7)` for a computational trapdoor and `sqrt(c² (1 + ŝ²) + σ²)` with `c = (b + 1) σ` and
    /// `ŝ = σ sqrt(n) (sqrt(m̄) + sqrt(D) + sqrt(max(0, ln(512 n))))` for a statistical one.
    fn sampled_shape(
        &self,
        ty: &ConcreteWireType,
        public: &ConcreteMatrixType,
        columns: usize,
    ) -> Result<(mxx_lattice_leaves::preimage::Shape, u64, f64, f64), LowerError> {
        let ConcreteWireType::Trapdoor {
            sigma,
            gadget_base,
            digit_count,
            preimage_sigma,
            instantiation,
            ..
        } = ty
        else {
            unreachable!("a trapdoor")
        };
        let real = |e: &crate::RealExpr| {
            e.evaluate_f64(&self.env).map_err(|e| LowerError::Expression(e.to_string()))
        };
        let sigma = real(sigma)?;
        let base = u64::try_from(gadget_base).map_err(|e| LowerError::Expression(e.to_string()))?;
        let moduli = public.ring.crt_moduli();
        let bits = moduli.iter().map(|q| 64 - q.leading_zeros() as usize).max().unwrap_or(0);
        if !base.is_power_of_two() || base < 2 {
            return Err(LowerError::Unsupported(format!("a gadget base {base}")));
        }
        let tower_digits = bits.div_ceil(base.trailing_zeros() as usize);
        let n = public.ring.ring_dimension() as usize;
        let statistical = match instantiation {
            TrapdoorInstantiation::Statistical { m_bar } => *m_bar,
            TrapdoorInstantiation::Computational => 0,
        };
        let shape = mxx_lattice_leaves::preimage::Shape {
            n,
            towers: moduli.len(),
            rows: public.rows,
            digits: *digit_count,
            tower_digits,
            columns,
            statistical,
        };
        if *digit_count != tower_digits * moduli.len() ||
            tower_digits > mxx_lattice_leaves::gauss::MAX_DIGITS ||
            n < 2
        {
            return Err(LowerError::Unsupported(format!(
                "a sampled trapdoor of {} digits over {} towers of {tower_digits} digits",
                digit_count,
                moduli.len()
            )));
        }
        let c = (base as f64 + 1.0) * sigma;
        let width = match (preimage_sigma, statistical) {
            (Some(width), _) => real(width)?,
            (None, 0) => {
                1.8 * c *
                    sigma *
                    (((n * public.rows * digit_count) as f64).sqrt() +
                        (2.0 * n as f64).sqrt() +
                        4.7)
            }
            (None, m_bar) => {
                let deviation = (n as f64 * 512.0).ln().max(0.0);
                let s1 = sigma *
                    (n as f64).sqrt() *
                    ((m_bar as f64).sqrt() + (*digit_count as f64).sqrt() + deviation.sqrt());
                (c * c * (1.0 + s1 * s1) + sigma * sigma).sqrt()
            }
        };
        Ok((shape, base, sigma, width))
    }

    /// A statistical trapdoor of type `ty` (public matrix of type `public`, `T` of type `t`): `T`
    /// drawn by `leaves::statistical_trapgen` and redrawn by an `until` loop of at most
    /// [`TRAPGEN_ATTEMPTS`] iterations while the perturbation covariance of its preimages is not
    /// positive definite (faulting after the last, where the CPU backend reports an error), `ā`
    /// uniform, and `A = [ā | g - ā T]`, so `A [T; I] = g`; returns the public matrix and the
    /// trapdoor (per limb, `A ‖ T`).
    fn statistical_trapdoor(
        &mut self,
        ty: &ConcreteWireType,
        public: &ConcreteMatrixType,
        t: &ConcreteMatrixType,
        node_matrix: &crate::types::MatrixType,
        gadget_base: &GraphExpr,
        definition: &mut DefinitionBuilder,
    ) -> Result<Vec<Value>, LowerError> {
        let (shape, base, sigma, width) = self.sampled_shape(ty, public, 1)?;
        let (n, moduli, m_bar) = (shape.n, public.ring.crt_moduli().to_vec(), shape.statistical);
        let samples = (m_bar * shape.digits * n) as u64;
        let budget = 32 * samples + 65_536;
        let steps = leaves::statistical_trapgen_steps(&shape, budget);
        let scratch = mxx_lattice_leaves::preimage::trapgen_scratch_words(n, m_bar, shape.digits);
        let arguments: Vec<IntExpr> = [
            n as u64,
            moduli.len() as u64,
            m_bar as u64,
            shape.digits as u64,
            base,
            sigma.to_bits(),
            width.to_bits(),
            budget,
            steps,
            scratch as u64,
        ]
        .iter()
        .map(|w| IntExpr::from(i128::from(*w)))
        .collect();
        let t_value = self.retry(
            definition,
            t,
            TRAPGEN_ATTEMPTS,
            Vec::new(),
            Vec::new(),
            &moduli,
            "trapgen_attempt",
            |leaves| leaves.statistical_trapgen.clone(),
            arguments,
        )?;
        let mut a_bar_type = public.clone();
        a_bar_type.columns = m_bar;
        let uniform = NodeKind::UniformResidueSample { matrix_type: node_matrix.clone() };
        let a_bar = self.sample_matrix(&uniform, &a_bar_type, definition)?;
        let mut g_type = public.clone();
        g_type.columns = shape.digits;
        let gadget = ConstantMatrix::Gadget { base: gadget_base.clone(), small: false };
        let g_bytes = self.constant_matrix(&g_type, &gadget)?;
        let g = definition.builtin(leaves::constant(matrix_type(&g_type), g_bytes), Vec::new());
        let m = |x: &ConcreteMatrixType| ConcreteWireType::Matrix(x.clone());
        let multiply = NodeKind::MatrixBinary(MatrixBinaryOp::Multiply);
        let a_t = self.matrix_op(
            &multiply,
            &g_type,
            &[m(&a_bar_type), m(t)],
            definition,
            vec![a_bar, t_value],
        )?;
        let subtract = NodeKind::MatrixBinary(MatrixBinaryOp::Subtract);
        let a1 = self.matrix_op(
            &subtract,
            &g_type,
            &[m(&g_type), m(&g_type)],
            definition,
            vec![g, a_t],
        )?;
        let concat = NodeKind::Concat { axis: ConcatAxis::Columns };
        let a = self.matrix_op(
            &concat,
            public,
            &[m(&a_bar_type), m(&g_type)],
            definition,
            vec![a_bar, a1],
        )?;
        let trapdoor = limb_loop(
            &mut self.builder,
            definition,
            vec![matrix_type(public), matrix_type(t)],
            &[true, true],
            moduli.len(),
            vec![a, t_value],
            Box::new(|body: &mut DefinitionBuilder, limbs: Vec<Value>| {
                body.builtin(Builtin::Concat, limbs)
            }),
        );
        Ok(vec![a, trapdoor])
    }

    /// An `until` loop of at most `attempts` iterations, each a call of the leaf `leaf` (the
    /// arguments `arguments`) on `inputs` (of types `input_types`), the moduli, and 32 fresh coin
    /// bytes, whose outputs are a value of type `result` and the stop flag; the loop starts from
    /// zeros and faults after its last iteration.
    #[allow(clippy::too_many_arguments)]
    fn retry(
        &mut self,
        definition: &mut DefinitionBuilder,
        result: &ConcreteMatrixType,
        attempts: u64,
        input_types: Vec<CoreType>,
        inputs: Vec<Value>,
        moduli: &[u64],
        name: &str,
        leaf: impl Fn(&Leaves) -> Arc<Leaf>,
        arguments: Vec<IntExpr>,
    ) -> Result<Value, LowerError> {
        let result_type = matrix_type(result);
        let mut params = vec![result_type];
        params.extend(input_types);
        params.push(CoreType::buffer(8, moduli.len()));
        let count = params.len();
        let mut body = DefinitionBuilder::new(name, params);
        let coins = body.builtin(Builtin::Coins { bytes: 32.into() }, Vec::new());
        let mut body_inputs: Vec<Value> = (1..count).map(|i| body.param(i)).collect();
        body_inputs.push(coins);
        let outputs = body.leaf(&leaf(&self.leaves), arguments, body_inputs);
        let body = self.builder.add_definition(body.finish(outputs));
        let modulus_words = moduli.iter().flat_map(|q| q.to_le_bytes()).collect();
        let moduli_value = definition.builtin(
            leaves::constant(CoreType::buffer(8, moduli.len()), modulus_words),
            Vec::new(),
        );
        let limb_words = result.rows * result.columns * result.ring.ring_dimension() as usize;
        let first = definition.leaf(
            &self.leaves.zero,
            vec![limb_words.into(), moduli.len().into()],
            Vec::new(),
        )[0];
        let mut loop_inputs = inputs;
        loop_inputs.push(moduli_value);
        Ok(definition.add_loop(
            Loop {
                count: attempts.into(),
                index_slot: RETRY,
                body,
                bindings: Vec::new(),
                carried: vec![first],
                inputs: loop_inputs,
                until: true,
            },
            0,
        )[0])
    }

    /// A preimage of `target` under the computational trapdoor `operands[0]` (of type `ty`, public
    /// matrix of type `public`), as the CPU backend samples it: candidates of
    /// `leaves::trapdoor_preimage` from fresh coins, redrawn by an `until` loop of at most
    /// [`PREIMAGE_ATTEMPTS`] iterations until one is within `bound` (and its stream sufficed),
    /// faulting when none is. The
    /// width is the trapdoor's preimage width, or the computational sampler's minimum
    /// `1.8 (b + 1) σ² (sqrt(n rows D) + sqrt(2 n) + 4.7)`.
    fn trapdoor_preimage(
        &mut self,
        ty: &ConcreteWireType,
        public: &ConcreteMatrixType,
        target: &ConcreteMatrixType,
        bound: &BigInt,
        definition: &mut DefinitionBuilder,
        operands: [Value; 2],
    ) -> Result<Value, LowerError> {
        let (shape, base, sigma, width) = self.sampled_shape(ty, public, target.columns)?;
        if target.rows != public.rows {
            return Err(LowerError::Unsupported("a preimage of a target of other rows".into()));
        }
        let (n, moduli) = (shape.n, public.ring.crt_moduli().to_vec());
        let bound = u64::try_from(bound).unwrap_or(u64::MAX);
        let budget = 32 * shape.gaussian_samples() as u64 + 65_536;
        let steps = leaves::trapdoor_preimage_steps(&shape, budget);
        let scratch = mxx_lattice_leaves::preimage::scratch_words(&shape);
        let arguments: Vec<IntExpr> = [
            n as u64,
            moduli.len() as u64,
            public.rows as u64,
            shape.digits as u64,
            shape.tower_digits as u64,
            target.columns as u64,
            shape.secret_rows() as u64,
            u64::from(shape.statistical > 0),
            base,
            sigma.to_bits(),
            width.to_bits(),
            bound,
            budget,
            steps,
            scratch as u64,
        ]
        .iter()
        .map(|w| IntExpr::from(i128::from(*w)))
        .collect();
        let mut preimage = public.clone();
        preimage.rows = public.columns;
        preimage.columns = target.columns;
        let input_types = vec![core_type(ty, self.int_words)?, matrix_type(target)];
        self.retry(
            definition,
            &preimage,
            PREIMAGE_ATTEMPTS,
            input_types,
            operands.to_vec(),
            &moduli,
            "preimage_attempt",
            |leaves| leaves.trapdoor_preimage.clone(),
            arguments,
        )
    }

    /// A sample of type `matrix` (the sampler's own matrix type is not read).
    fn sample_matrix(
        &mut self,
        kind: &NodeKind,
        matrix: &ConcreteMatrixType,
        definition: &mut DefinitionBuilder,
    ) -> Result<Value, LowerError> {
        let matrix = matrix.clone();
        let words = matrix.rows * matrix.columns * matrix.ring.ring_dimension() as usize;
        let moduli = matrix.ring.crt_moduli().to_vec();
        let modulus = per_limb(moduli.clone());
        let coins = Builtin::Coins { bytes: (16 * words).into() };
        match kind {
            NodeKind::UniformResidueSample { .. } => {
                let leaf = self.leaves.uniform_mod.clone();
                Ok(limb_loop(
                    &mut self.builder,
                    definition,
                    Vec::new(),
                    &[],
                    moduli.len(),
                    Vec::new(),
                    Box::new(move |body: &mut DefinitionBuilder, _| {
                        let coins = body.builtin(coins, Vec::new());
                        body.leaf(&leaf, vec![words.into(), modulus], vec![coins])[0]
                    }),
                ))
            }
            NodeKind::UniformIntervalSample { range, .. } => {
                let evaluate = |e: &GraphExpr| {
                    e.evaluate_in(self).map_err(|e| LowerError::Expression(e.to_string()))
                };
                let (minimum, maximum) = (evaluate(&range.minimum)?, evaluate(&range.maximum)?);
                let width = u64::try_from(&maximum - &minimum + 1)
                    .ok()
                    .filter(|w| *w > 0 && *w < 1 << 63)
                    .ok_or_else(|| {
                        LowerError::Unsupported(format!("the interval [{minimum}, {maximum}]"))
                    })?;
                let minimum_mod_q = per_limb(
                    moduli
                        .iter()
                        .map(|q| {
                            u64::try_from(minimum.mod_floor(&BigInt::from(*q))).expect("a residue")
                        })
                        .collect(),
                );
                let shared = definition.builtin(coins, Vec::new());
                let leaf = self.leaves.interval_mod.clone();
                Ok(limb_loop(
                    &mut self.builder,
                    definition,
                    vec![CoreType::bytes(16 * words)],
                    &[false],
                    moduli.len(),
                    vec![shared],
                    Box::new(move |body: &mut DefinitionBuilder, coins: Vec<Value>| {
                        let arguments = vec![words.into(), modulus, minimum_mod_q, width.into()];
                        body.leaf(&leaf, arguments, coins)[0]
                    }),
                ))
            }
            NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
                let sigma = sigma
                    .evaluate_f64(&self.env)
                    .map_err(|e| LowerError::Expression(e.to_string()))?;
                let bound = u64::try_from(
                    max_coefficient_bound
                        .evaluate_in(self)
                        .map_err(|e| LowerError::Expression(e.to_string()))?,
                )
                .map_err(|e| LowerError::Expression(e.to_string()))?;
                let table = gaussian_table(sigma, bound);
                let entries = table.len();
                let table = leaves::constant(
                    CoreType::buffer(8, entries),
                    table.iter().flat_map(|w| w.to_le_bytes()).collect(),
                );
                let table = definition.builtin(table, Vec::new());
                let coins =
                    definition.builtin(Builtin::Coins { bytes: (8 * words).into() }, Vec::new());
                let offset = per_limb(
                    moduli
                        .iter()
                        .map(|q| {
                            u64::try_from((-BigInt::from(bound)).mod_floor(&BigInt::from(*q)))
                                .expect("a residue")
                        })
                        .collect(),
                );
                let leaf = self.leaves.cdt_mod.clone();
                Ok(limb_loop(
                    &mut self.builder,
                    definition,
                    vec![CoreType::bytes(8 * words), CoreType::buffer(8, entries)],
                    &[false, false],
                    moduli.len(),
                    vec![coins, table],
                    Box::new(move |body: &mut DefinitionBuilder, inputs: Vec<Value>| {
                        body.leaf(
                            &leaf,
                            vec![words.into(), modulus, entries.into(), offset],
                            inputs,
                        )[0]
                    }),
                ))
            }
            _ => unreachable!("only samples reach here"),
        }
    }

    /// An operation on matrices that works limb by limb.
    /// A ring conversion as an exact CRT map (`leaves::crt_map`), or as a selection of limbs for
    /// a reduction to a subset of the basis and for a rebase that keeps the basis.
    fn ring_conversion(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        kind: &NodeKind,
        out: WireRef,
        argument: WireRef,
        definition: &mut DefinitionBuilder,
        input: Value,
    ) -> Result<Value, LowerError> {
        let matrix = |ty: &ConcreteWireType| match ty {
            ConcreteWireType::Matrix(m) |
            ConcreteWireType::SmallMatrix { matrix: m, .. } |
            ConcreteWireType::Preimage { matrix: m, .. } => Ok(m.clone()),
            _ => Err(LowerError::Unsupported("a ring conversion of a non-matrix".into())),
        };
        let source = matrix(self.wire_type(scope_id, argument)?)?;
        let destination = matrix(self.wire_type(scope_id, out)?)?;
        let (from, to) =
            (source.ring.crt_moduli().to_vec(), destination.ring.crt_moduli().to_vec());
        let selection = |definition: &mut DefinitionBuilder| {
            let limbs = to
                .iter()
                .map(|q| {
                    let index = from.iter().position(|p| p == q).expect("a subset of the basis");
                    definition.builtin(Builtin::Get { index: index.into() }, vec![input])
                })
                .collect();
            definition.builtin(Builtin::Pack, limbs)
        };
        // Bounded digits per CRT limb are not one integer; validation lets them keep their basis
        // only, and so does the copy.
        let per_limb_digits = matches!(
            self.wire_type(scope_id, argument)?,
            ConcreteWireType::SmallMatrix { bound_domain: CoefficientBoundDomain::PerCrtLimb, .. } |
                ConcreteWireType::Preimage {
                    bound_domain: CoefficientBoundDomain::PerCrtLimb,
                    ..
                }
        );
        let mut divisor = vec![0u64; leaves::CRT_MAP_WORDS];
        let big = |value: &GraphExpr, divisor: &mut Vec<u64>| -> Result<(), LowerError> {
            let value =
                value.evaluate_in(self).map_err(|e| LowerError::Expression(e.to_string()))?;
            let (_, digits) = value.to_u64_digits();
            if digits.len() > leaves::CRT_MAP_WORDS {
                return Err(LowerError::Unsupported("a constant above the leaf's width".into()));
            }
            divisor[..digits.len()].copy_from_slice(&digits);
            Ok(())
        };
        let mode = match kind {
            NodeKind::ModulusReduce { .. } => return Ok(selection(definition)),
            NodeKind::CenteredRebase { .. } if from == to || per_limb_digits => {
                return Ok(selection(definition));
            }
            NodeKind::ModulusSwitch { .. } => 0,
            NodeKind::CenteredRebase { .. } => 1,
            NodeKind::CenteredRoundDivide { divisor: d } => {
                big(d, &mut divisor)?;
                2
            }
            NodeKind::BlockModSwitch { plaintext_modulus, .. } => {
                big(plaintext_modulus, &mut divisor)?;
                3
            }
            NodeKind::RnsModUp { digit_size, normalize, .. } => {
                return self.fast_base_conversion(
                    definition,
                    input,
                    &source,
                    &to,
                    Conversion::Up { digit_size: *digit_size, normalize: *normalize },
                );
            }
            NodeKind::RnsModDown { plaintext_modulus, .. } => {
                let t = u64::try_from(
                    plaintext_modulus
                        .evaluate_in(self)
                        .map_err(|e| LowerError::Expression(e.to_string()))?,
                )
                .map_err(|e| LowerError::Expression(e.to_string()))?;
                return self.fast_base_conversion(
                    definition,
                    input,
                    &source,
                    &to,
                    Conversion::Down { t },
                );
            }
            _ => unreachable!("a ring conversion"),
        };
        if from.len() > 16 || to.len() > 16 {
            return Err(LowerError::Unsupported("a CRT basis of more than 16 primes".into()));
        }
        let table =
            |moduli: &[u64]| -> Vec<u8> { moduli.iter().flat_map(|q| q.to_le_bytes()).collect() };
        let constant = |definition: &mut DefinitionBuilder, words: usize, bytes: Vec<u8>| {
            let leaf = leaves::constant(CoreType::buffer(8, words), bytes);
            definition.builtin(leaf, Vec::new())
        };
        let source_moduli = constant(definition, from.len(), table(&from));
        let destination_moduli = constant(definition, to.len(), table(&to));
        let divisor = constant(definition, divisor.len(), table(&divisor));
        let limb_words = source.rows * source.columns * source.ring.ring_dimension() as usize;
        Ok(definition.leaf(
            &self.leaves.crt_map,
            vec![limb_words.into(), from.len().into(), to.len().into(), mode.into()],
            vec![input, source_moduli, destination_moduli, divisor],
        )[0])
    }

    /// OpenFHE's fast RNS base conversion (`leaves::fast_base_conversion`), with its public tables
    /// computed here as the CPU executor's native code computes them.
    fn fast_base_conversion(
        &mut self,
        definition: &mut DefinitionBuilder,
        input: Value,
        source: &ConcreteMatrixType,
        to: &[u64],
        conversion: Conversion,
    ) -> Result<Value, LowerError> {
        let from = source.ring.crt_moduli();
        let mul = |a: u64, b: u64, q: u64| ((a as u128 * b as u128) % q as u128) as u64;
        let inverse = |a: u64, q: u64| {
            let (mut result, mut base, mut exponent) = (1u64, a % q, q - 2);
            while exponent > 0 {
                if exponent & 1 == 1 {
                    result = mul(result, base, q);
                }
                base = mul(base, base, q);
                exponent >>= 1;
            }
            result
        };
        let product = |members: &mut dyn Iterator<Item = u64>, q: u64| {
            members.fold(1u64, |acc, p| mul(acc, p % q, q))
        };
        let (groups, group, down): (Vec<Vec<usize>>, Vec<u64>, Option<u64>) = match conversion {
            Conversion::Up { digit_size, .. } => {
                let groups: Vec<Vec<usize>> = (0..from.len())
                    .collect::<Vec<_>>()
                    .chunks(digit_size)
                    .map(<[usize]>::to_vec)
                    .collect();
                (groups, (0..from.len()).map(|i| (i / digit_size) as u64).collect(), None)
            }
            Conversion::Down { t } => {
                let dropped = (0..from.len()).filter(|i| !to.contains(&from[*i])).collect();
                (vec![dropped], vec![0; from.len()], Some(t))
            }
        };
        let mut scale = vec![0u64; from.len()];
        let mut weight = vec![0u64; from.len() * to.len()];
        for members in &groups {
            for &i in members {
                let p = from[i];
                let others = members.iter().filter(|j| **j != i).map(|j| from[*j]);
                let mut denominator = product(&mut others.clone(), p);
                if let Conversion::Up { normalize: true, .. } = conversion {
                    let outside = (0..from.len()).filter(|j| !members.contains(j)).map(|j| from[j]);
                    denominator = mul(denominator, product(&mut outside.into_iter(), p), p);
                }
                scale[i] = inverse(denominator, p);
                if let Some(t) = down {
                    scale[i] = mul(scale[i], p - inverse(t % p, p), p);
                }
                for (m, q) in to.iter().enumerate() {
                    weight[i * to.len() + m] = product(&mut others.clone(), *q);
                }
            }
        }
        let mut down_table = vec![0u64; 3 * to.len()];
        if let Some(t) = down {
            for (m, q) in to.iter().enumerate() {
                let dropped = &mut groups[0].iter().map(|i| from[*i]);
                down_table[3 * m] = t % q;
                down_table[3 * m + 1] = inverse(product(dropped, *q), *q);
                down_table[3 * m + 2] =
                    from.iter().position(|p| p == q).expect("a subset of the basis") as u64;
            }
        }
        // The source and destination moduli, then the group, scale, weight, and down tables.
        let tables = [from.to_vec(), to.to_vec(), group, scale, weight, down_table];
        let mut inputs = vec![input];
        for words in tables {
            let bytes = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let leaf = leaves::constant(CoreType::buffer(8, words.len()), bytes);
            inputs.push(definition.builtin(leaf, Vec::new()));
        }
        let limb_words = source.rows * source.columns * source.ring.ring_dimension() as usize;
        Ok(definition.leaf(
            &self.leaves.fast_base_conversion,
            vec![
                limb_words.into(),
                from.len().into(),
                to.len().into(),
                groups.len().into(),
                usize::from(down.is_some()).into(),
            ],
            inputs,
        )[0])
    }

    /// Polynomial values as one-word integers, and back: coefficients, OpenFHE's native
    /// evaluation slots (`leaves::ntt` with OpenFHE's root of unity), and one coefficient.
    fn polynomial_values(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        kind: &NodeKind,
        out: WireRef,
        argument: WireRef,
        definition: &mut DefinitionBuilder,
        input: Value,
    ) -> Result<Value, LowerError> {
        let matrix_wire =
            if matches!(kind, NodeKind::PolynomialFromValues { .. }) { out } else { argument };
        let ConcreteWireType::Matrix(matrix) = self.wire_type(scope_id, matrix_wire)?.clone()
        else {
            return Err(LowerError::Unsupported("polynomial values of a non-matrix".into()));
        };
        let moduli = matrix.ring.crt_moduli().to_vec();
        let modulus: BigInt = moduli.iter().map(|q| BigInt::from(*q)).product();
        if modulus.bits() >= 64 * u64::from(self.int_words) || moduli.len() > 16 {
            return Err(LowerError::Unsupported(
                "polynomial values above the lowering's integers".into(),
            ));
        }
        let n = matrix.ring.ring_dimension() as usize;
        let depth = moduli.len();
        let table = |words: Vec<u64>, ty: CoreType, d: &mut DefinitionBuilder| {
            let leaf = leaves::constant(ty, words.iter().flat_map(|w| w.to_le_bytes()).collect());
            d.builtin(leaf, Vec::new())
        };
        let moduli_value = table(moduli.clone(), CoreType::buffer(8, depth), definition);
        let ntt = |this: &mut Self, definition: &mut DefinitionBuilder, limbs: Value, inverse| {
            let tables: Vec<NttTable> = moduli.iter().map(|q| NttTable::new(*q, n)).collect();
            let words = tables
                .iter()
                .flat_map(|t| if inverse { t.inverse.clone() } else { t.forward.clone() })
                .collect();
            let ty = CoreType::family(CoreType::buffer(8, n), depth as u64);
            let table = table(words, ty, definition);
            let leaf = this.leaves.ntt.clone();
            let q = per_limb(moduli.clone());
            let n_inverse = per_limb(tables.iter().map(|t| t.n_inverse).collect());
            let limb = CoreType::buffer(8, n);
            limb_loop(
                &mut this.builder,
                definition,
                vec![
                    CoreType::family(limb.clone(), depth as u64),
                    CoreType::family(limb, depth as u64),
                ],
                &[true, true],
                depth,
                vec![limbs, table],
                Box::new(move |body: &mut DefinitionBuilder, m: Vec<Value>| {
                    body.leaf(
                        &leaf,
                        vec![n.into(), q, 1.into(), usize::from(inverse).into(), n_inverse],
                        m,
                    )[0]
                }),
            )
        };
        Ok(match kind {
            NodeKind::PolynomialValues { evaluation } => {
                let limbs = if *evaluation { ntt(self, definition, input, false) } else { input };
                definition.leaf(
                    &self.leaves.crt_values,
                    vec![n.into(), depth.into(), self.int_words.into()],
                    vec![limbs, moduli_value],
                )[0]
            }
            NodeKind::PolynomialFromValues { evaluation, .. } => {
                let limbs = definition.leaf(
                    &self.leaves.int_residues,
                    vec![n.into(), depth.into(), self.int_words.into()],
                    vec![input, moduli_value],
                )[0];
                if *evaluation { ntt(self, definition, limbs, true) } else { limbs }
            }
            NodeKind::ExtractCoefficient { position, .. } => {
                let position = usize::try_from(
                    position
                        .evaluate_in(self)
                        .map_err(|e| LowerError::Expression(e.to_string()))?,
                )
                .map_err(|e| LowerError::Expression(e.to_string()))?;
                let words = (0..depth)
                    .map(|l| {
                        let limb =
                            definition.builtin(Builtin::Get { index: l.into() }, vec![input]);
                        definition.builtin(
                            Builtin::Window {
                                offset: (8 * position).into(),
                                runs: 1.into(),
                                length: 8.into(),
                                stride: 0.into(),
                            },
                            vec![limb],
                        )
                    })
                    .collect();
                let limbs = definition.builtin(Builtin::Pack, words);
                let values = definition.leaf(
                    &self.leaves.crt_values,
                    vec![1.into(), depth.into(), self.int_words.into()],
                    vec![limbs, moduli_value],
                )[0];
                definition.builtin(Builtin::Get { index: 0.into() }, vec![values])
            }
            _ => unreachable!("a polynomial value conversion"),
        })
    }

    /// `sum_i c_i (round(t_i x_i / Q_i) mod t_i)` in the destination ring: one
    /// `leaves::round_lift` per level, summed limb by limb.
    #[allow(clippy::too_many_arguments)]
    fn crt_recompose(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        out: WireRef,
        arguments: &[WireRef],
        plaintext_moduli: &[GraphExpr],
        reconstruction_coefficients: &[GraphExpr],
        definition: &mut DefinitionBuilder,
        operands: Vec<Value>,
    ) -> Result<Value, LowerError> {
        let ConcreteWireType::Matrix(destination) = self.wire_type(scope_id, out)?.clone() else {
            return Err(LowerError::Unsupported("a CRT recomposition to a non-matrix".into()));
        };
        let evaluate =
            |e: &GraphExpr| e.evaluate_in(self).map_err(|e| LowerError::Expression(e.to_string()));
        let plaintext_moduli =
            plaintext_moduli.iter().map(evaluate).collect::<Result<Vec<_>, _>>()?;
        let reconstruction_coefficients =
            reconstruction_coefficients.iter().map(evaluate).collect::<Result<Vec<_>, _>>()?;
        let to = destination.ring.crt_moduli().to_vec();
        let table = |definition: &mut DefinitionBuilder, words: &[u64]| {
            let bytes = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let leaf = leaves::constant(CoreType::buffer(8, words.len()), bytes);
            definition.builtin(leaf, Vec::new())
        };
        let destination_moduli = table(definition, &to);
        let words =
            destination.rows * destination.columns * destination.ring.ring_dimension() as usize;
        let mut sum: Option<Value> = None;
        for (index, (wire, level)) in arguments.iter().zip(operands).enumerate() {
            let ConcreteWireType::Matrix(source) = self.wire_type(scope_id, *wire)?.clone() else {
                return Err(LowerError::Unsupported("a CRT level that is not a matrix".into()));
            };
            let from = source.ring.crt_moduli().to_vec();
            let t = u64::try_from(plaintext_moduli[index].clone())
                .ok()
                .filter(|t| *t < 1 << 63)
                .ok_or_else(|| {
                    LowerError::Unsupported("a plaintext modulus of 2^63 or more".into())
                })?;
            if from.len() > 16 {
                return Err(LowerError::Unsupported("a CRT basis of more than 16 primes".into()));
            }
            let c = reconstruction_coefficients[index].clone();
            let residues: Vec<u64> = to
                .iter()
                .map(|q| {
                    let q = BigInt::from(*q);
                    u64::try_from(c.mod_floor(&q)).expect("a residue fits a word")
                })
                .collect();
            let source_moduli = table(definition, &from);
            let residues = table(definition, &residues);
            let lifted = definition.leaf(
                &self.leaves.round_lift,
                vec![words.into(), from.len().into(), to.len().into(), t.into()],
                vec![level, source_moduli, destination_moduli, residues],
            )[0];
            sum = Some(match sum {
                None => lifted,
                Some(previous) => {
                    let leaf = self.leaves.add.clone();
                    let modulus = per_limb(to.clone());
                    let limb = CoreType::buffer(8, words);
                    limb_loop(
                        &mut self.builder,
                        definition,
                        vec![
                            CoreType::family(limb.clone(), to.len() as u64),
                            CoreType::family(limb, to.len() as u64),
                        ],
                        &[true, true],
                        to.len(),
                        vec![previous, lifted],
                        Box::new(move |body: &mut DefinitionBuilder, m: Vec<Value>| {
                            body.leaf(&leaf, vec![words.into(), modulus], m)[0]
                        }),
                    )
                }
            });
        }
        sum.ok_or_else(|| LowerError::Unsupported("a CRT recomposition of no levels".into()))
    }

    /// A matrix of residues uniform modulo the ring modulus from a key and a tag
    /// (`leaves::hash_uniform`).
    fn hash_matrix(
        &mut self,
        definition: &mut DefinitionBuilder,
        matrix: &ConcreteMatrixType,
        key: Value,
        tag: Value,
        tag_words: usize,
    ) -> Value {
        let modulus = matrix.ring.modulus();
        let bits = modulus.bits() as usize;
        let width = bits.div_ceil(8).max(1);
        let high_mask: u64 = if bits.is_multiple_of(8) { 0xff } else { (1 << (bits % 8)) - 1 };
        let mut modulus_bytes = modulus.to_bytes_le().1;
        modulus_bytes.resize(modulus_bytes.len().div_ceil(8).max(1) * 8, 0);
        let constant = |definition: &mut DefinitionBuilder, bytes: Vec<u8>| {
            let ty = CoreType::buffer(8, bytes.len() / 8);
            definition.builtin(leaves::constant(ty, bytes), Vec::new())
        };
        let modulus_words = modulus_bytes.len() / 8;
        let moduli = constant(
            definition,
            matrix.ring.crt_moduli().iter().flat_map(|q| q.to_le_bytes()).collect(),
        );
        let modulus_value = constant(definition, modulus_bytes);
        let n = matrix.ring.ring_dimension() as usize;
        definition.leaf(
            &self.leaves.hash_uniform,
            vec![
                n.into(),
                matrix.ring.crt_depth().into(),
                matrix.rows.into(),
                matrix.columns.into(),
                width.into(),
                high_mask.into(),
                (matrix.rows * matrix.columns * n).into(),
                tag_words.into(),
                modulus_words.into(),
            ],
            vec![key, tag, moduli, modulus_value],
        )[0]
    }

    /// The balanced CRT gadget decomposition of `value`, of type `input`, into `digit_count`
    /// base-`base` digits per row (`leaves::gadget_decompose`): over the leading towers without
    /// `small` (dropping the rest, ePrint 2024/909, section 3.2), or per limb with `small`.
    fn gadget_decompose(
        &mut self,
        definition: &mut DefinitionBuilder,
        input: &ConcreteMatrixType,
        value: Value,
        base: u64,
        digit_count: usize,
        small: bool,
    ) -> Result<Value, LowerError> {
        let moduli = input.ring.crt_moduli().to_vec();
        let depth = moduli.len();
        let bits = moduli.iter().map(|q| 64 - q.leading_zeros() as usize).max().unwrap_or(0);
        if !base.is_power_of_two() || base < 2 {
            return Err(LowerError::Unsupported(format!("a gadget base {base}")));
        }
        let per_tower = bits.div_ceil(base.trailing_zeros() as usize);
        let retained = if small { depth } else { digit_count / per_tower };
        let valid = if small {
            digit_count == per_tower
        } else {
            digit_count > 0 && digit_count.is_multiple_of(per_tower) && retained <= depth
        };
        if !valid {
            return Err(LowerError::Unsupported(format!(
                "a gadget decomposition of {digit_count} digits over {depth} towers"
            )));
        }
        // The dropped towers' correction tables.
        let mul = |a: u64, b: u64, q: u64| ((a as u128 * b as u128) % q as u128) as u64;
        let cofactor = |p: usize, q: u64| {
            (retained..depth).filter(|r| *r != p).fold(1u64, |acc, r| mul(acc, moduli[r] % q, q))
        };
        let mut twist = vec![0u64; depth];
        let mut weight = vec![0u64; depth * depth];
        for p in retained..depth {
            let prime = moduli[p];
            let (mut inverse, mut power, mut exponent) = (1u64, cofactor(p, prime), prime - 2);
            while exponent > 0 {
                if exponent & 1 == 1 {
                    inverse = mul(inverse, power, prime);
                }
                power = mul(power, power, prime);
                exponent >>= 1;
            }
            twist[p] = inverse;
            for t in 0..retained {
                weight[p * depth + t] = cofactor(p, moduli[t]);
            }
        }
        let table = |definition: &mut DefinitionBuilder, words: &[u64]| {
            let bytes = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let leaf = leaves::constant(CoreType::buffer(8, words.len()), bytes);
            definition.builtin(leaf, Vec::new())
        };
        let moduli_value = table(definition, &moduli);
        let twist = table(definition, &twist);
        let weight = table(definition, &weight);
        let n = input.ring.ring_dimension() as usize;
        let words_in = input.rows * input.columns * n;
        Ok(definition.leaf(
            &self.leaves.gadget_decompose,
            vec![
                n.into(),
                depth.into(),
                input.rows.into(),
                input.columns.into(),
                base.into(),
                per_tower.into(),
                words_in.into(),
                (words_in * digit_count).into(),
                retained.into(),
                usize::from(small).into(),
            ],
            vec![value, moduli_value, twist, weight],
        )[0])
    }

    fn matrix_operation(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        kind: &NodeKind,
        out: WireRef,
        arguments: &[WireRef],
        definition: &mut DefinitionBuilder,
        inputs: Vec<Value>,
    ) -> Result<Value, LowerError> {
        let unsupported = || {
            LowerError::Unsupported(format!(
                "node kind {}",
                format!("{kind:?}").chars().take(60).collect::<String>()
            ))
        };
        let result = match self.wire_type(scope_id, out)? {
            ConcreteWireType::Matrix(m) => m.clone(),
            _ => return Err(unsupported()),
        };
        let operands = arguments
            .iter()
            .map(|wire| self.wire_type(scope_id, *wire).cloned())
            .collect::<Result<Vec<_>, _>>()?;
        self.matrix_op(kind, &result, &operands, definition, inputs)
    }

    /// A matrix operation of result type `result` over operands of types `operands`.
    fn matrix_op(
        &mut self,
        kind: &NodeKind,
        result: &ConcreteMatrixType,
        operands: &[ConcreteWireType],
        definition: &mut DefinitionBuilder,
        inputs: Vec<Value>,
    ) -> Result<Value, LowerError> {
        let unsupported = || {
            LowerError::Unsupported(format!(
                "node kind {}",
                format!("{kind:?}").chars().take(60).collect::<String>()
            ))
        };
        let result = result.clone();
        let ring = result.ring.clone();
        let n = ring.ring_dimension() as usize;
        let words = result.rows * result.columns * n;
        let modulus = per_limb(ring.crt_moduli().to_vec());
        let matrices: Vec<Option<ConcreteMatrixType>> = operands
            .iter()
            .map(|ty| match ty {
                ConcreteWireType::Matrix(m) |
                ConcreteWireType::SmallMatrix { matrix: m, .. } |
                ConcreteWireType::Preimage { matrix: m, .. } => Some(m.clone()),
                _ => None,
            })
            .collect();
        let leaf = |leaf: &Arc<Leaf>, arguments: Vec<IntExpr>| {
            let leaf = leaf.clone();
            move |body: &mut DefinitionBuilder, limbs: Vec<Value>| {
                body.leaf(&leaf, arguments, limbs)[0]
            }
        };
        let entries = result.rows * result.columns;
        let body: LimbBody<'static> = match kind {
            NodeKind::MatrixBinary(MatrixBinaryOp::Add) => {
                Box::new(leaf(&self.leaves.add, vec![words.into(), modulus]))
            }
            NodeKind::MatrixBinary(MatrixBinaryOp::Subtract) => {
                Box::new(leaf(&self.leaves.sub, vec![words.into(), modulus]))
            }
            NodeKind::MatrixNegate => Box::new(leaf(&self.leaves.neg, vec![words.into(), modulus])),
            NodeKind::MatrixScale { scalar } => {
                // A scalar that reads a loop index (through a binding) is reduced per iteration.
                let reduced: Vec<IntExpr> =
                    if substitute_vars(scalar, &self.subst).contains_loop_index() {
                        let scalar = self.expr(scalar)?;
                        ring.crt_moduli()
                            .iter()
                            .map(|q| scalar.clone().floor_rem(IntExpr::from(i128::from(*q))))
                            .collect()
                    } else {
                        let scalar = scalar
                            .evaluate_in(self)
                            .map_err(|e| LowerError::Expression(e.to_string()))?;
                        ring.crt_moduli()
                            .iter()
                            .map(|q| {
                                let q = BigInt::from(*q);
                                IntExpr::from(
                                    i128::try_from(((&scalar % &q) + &q) % &q)
                                        .expect("a residue fits a word"),
                                )
                            })
                            .collect()
                    };
                let reduced = IntExpr::Select {
                    selector: Box::new(IntExpr::LoopIndex(LIMB)),
                    branches: reduced,
                };
                Box::new(leaf(&self.leaves.scale, vec![words.into(), modulus, reduced]))
            }
            // A bounded right-hand side holds residues like any matrix, so its product is the
            // ordinary one.
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) | NodeKind::MatrixMulSmallRhs => {
                let (left, right) = (
                    matrices[0].clone().ok_or_else(unsupported)?,
                    matrices[1].clone().ok_or_else(unsupported)?,
                );
                self.matrix_product(n, modulus, &left, &right)
            }
            NodeKind::Tensor => {
                let (left, right) = (
                    matrices[0].clone().ok_or_else(unsupported)?,
                    matrices[1].clone().ok_or_else(unsupported)?,
                );
                let entry = (8 * n) as i128;
                let l = |d: u32| IntExpr::LoopIndex(PRODUCT - d);
                // Entry ((i1, i2), (j1, j2)) is A[i1][j1] B[i2][j2]: loops i1, i2, j1, j2.
                self.product_loops(
                    n,
                    modulus,
                    vec![left.rows, right.rows, left.columns, right.columns],
                    None,
                    (l(0) * left.columns as i128 + l(2)) * entry,
                    (l(1) * right.columns as i128 + l(3)) * entry,
                    left.rows * left.columns * n * 8,
                    right.rows * right.columns * n * 8,
                )
            }
            NodeKind::BlockDiagonal { count } => {
                let input = matrices[0].clone().ok_or_else(unsupported)?;
                Box::new(leaf(
                    &self.leaves.block_diagonal,
                    vec![n.into(), input.rows.into(), input.columns.into(), (*count).into()],
                ))
            }
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                // bias + sum_t c_t left_t right_t, one limb at a time.
                let reduced = coefficients
                    .iter()
                    .map(|c| {
                        let c = c
                            .evaluate_in(self)
                            .map_err(|e| LowerError::Expression(e.to_string()))?;
                        Ok(per_limb(
                            ring.crt_moduli()
                                .iter()
                                .map(|q| {
                                    u64::try_from(c.mod_floor(&BigInt::from(*q)))
                                        .expect("a residue")
                                })
                                .collect(),
                        ))
                    })
                    .collect::<Result<Vec<_>, LowerError>>()?;
                let shapes = (0..coefficients.len())
                    .map(|t| {
                        let (l, r) = (
                            matrices[2 * t].clone().ok_or_else(unsupported)?,
                            matrices[2 * t + 1].clone().ok_or_else(unsupported)?,
                        );
                        Ok((l.rows, l.columns, r.columns))
                    })
                    .collect::<Result<Vec<_>, LowerError>>()?;
                let products = (0..shapes.len())
                    .map(|t| {
                        let (l, r) = (
                            matrices[2 * t].clone().ok_or_else(unsupported)?,
                            matrices[2 * t + 1].clone().ok_or_else(unsupported)?,
                        );
                        Ok(self.matrix_product(n, modulus.clone(), &l, &r))
                    })
                    .collect::<Result<Vec<_>, LowerError>>()?;
                let (scale, add) = (self.leaves.scale.clone(), self.leaves.add.clone());
                let has_bias = *has_bias;
                Box::new(move |body: &mut DefinitionBuilder, limbs: Vec<Value>| {
                    let mut sum = has_bias.then(|| limbs[2 * shapes.len()]);
                    for (t, (product, c)) in products.into_iter().zip(reduced).enumerate() {
                        let product = product(body, vec![limbs[2 * t], limbs[2 * t + 1]]);
                        let scaled = body.leaf(
                            &scale,
                            vec![words.into(), modulus.clone(), c],
                            vec![product],
                        )[0];
                        sum = Some(match sum {
                            Some(previous) => body.leaf(
                                &add,
                                vec![words.into(), modulus.clone()],
                                vec![previous, scaled],
                            )[0],
                            None => scaled,
                        });
                    }
                    sum.expect("a multiply-accumulate has a product or a bias")
                })
            }
            NodeKind::Transpose => {
                let input = matrices[0].clone().ok_or_else(unsupported)?;
                Box::new(leaf(
                    &self.leaves.transpose,
                    vec![n.into(), input.rows.into(), input.columns.into()],
                ))
            }
            NodeKind::RingAutomorphism { index } => {
                let k = self.expr(index)?;
                Box::new(leaf(
                    &self.leaves.automorphism,
                    vec![n.into(), modulus, k, entries.into()],
                ))
            }
            NodeKind::MultiplyMonomial => Box::new(leaf(
                &self.leaves.multiply_monomial,
                vec![n.into(), modulus, entries.into()],
            )),
            NodeKind::Slice { rows, columns } => {
                let input = matrices[0].clone().ok_or_else(unsupported)?;
                // The start may depend on loop indices; validation fixed the result's shape.
                let start = |range: &Option<crate::node::IndexRange>| match range {
                    Some(r) => self.expr(&r.start),
                    None => Ok(IntExpr::from(0)),
                };
                let (r0, c0) = (start(rows)?, start(columns)?);
                let entry = 8 * n as i128;
                let window = Builtin::Window {
                    offset: (r0 * IntExpr::from(input.columns as i128) + c0) * IntExpr::from(entry),
                    runs: result.rows.into(),
                    length: (result.columns as i128 * entry).into(),
                    stride: (input.columns as i128 * entry).into(),
                };
                Box::new(move |body: &mut DefinitionBuilder, limbs: Vec<Value>| {
                    body.builtin(window, limbs)
                })
            }
            NodeKind::Concat { axis: ConcatAxis::Rows } => {
                Box::new(|body: &mut DefinitionBuilder, limbs: Vec<Value>| {
                    body.builtin(Builtin::Concat, limbs)
                })
            }
            NodeKind::Concat { axis: ConcatAxis::Columns } => {
                let columns: Vec<usize> = matrices
                    .iter()
                    .map(|m| m.as_ref().map(|m| m.columns))
                    .collect::<Option<_>>()
                    .ok_or_else(unsupported)?;
                let rows = result.rows;
                Box::new(move |body: &mut DefinitionBuilder, limbs: Vec<Value>| {
                    // Row r of the result is row r of every input, in order.
                    let mut pieces = Vec::new();
                    for r in 0..rows {
                        for (limb, width) in limbs.iter().zip(&columns) {
                            let length = (width * n * 8) as i128;
                            pieces.push(body.builtin(
                                Builtin::Window {
                                    offset: (r as i128 * length).into(),
                                    runs: 1.into(),
                                    length: length.into(),
                                    stride: 0.into(),
                                },
                                vec![*limb],
                            ));
                        }
                    }
                    body.builtin(Builtin::Concat, pieces)
                })
            }
            _ => return Err(unsupported()),
        };
        let input_types = operands
            .iter()
            .map(|ty| core_type(ty, self.int_words))
            .collect::<Result<Vec<_>, _>>()?;
        let per_limb_inputs: Vec<bool> = matrices.iter().map(Option::is_some).collect();
        Ok(limb_loop(
            &mut self.builder,
            definition,
            input_types,
            &per_limb_inputs,
            ring.crt_depth(),
            inputs,
            body,
        ))
    }
}

/// The cumulative distribution table of the discrete Gaussian `exp(-x^2 / (2 sigma^2))` on
/// `[-bound, bound]`: entry `k` is `2^63 P(X <= k - bound)` for `k < 2 bound`, so a 63-bit uniform
/// `u` falls at `X = #{k : u >= table[k]} - bound`. The table is computed in `f64`, so each
/// probability is exact to about `2^-50` relative.
fn gaussian_table(sigma: f64, bound: u64) -> Vec<u64> {
    let bound = bound as i64;
    let weights: Vec<f64> = (-bound..=bound)
        .map(|x| (-(x as f64) * (x as f64) / (2.0 * sigma * sigma)).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let scale = (1u64 << 63) as f64;
    let mut cumulative = 0.0;
    weights[..weights.len() - 1]
        .iter()
        .map(|w| {
            cumulative += w;
            ((cumulative / total) * scale).min(scale - 1.0) as u64
        })
        .collect()
}

/// Builds one limb of a result from the limb (or whole value) of each input.
type LimbBody<'a> = Box<dyn FnOnce(&mut DefinitionBuilder, Vec<Value>) -> Value + 'a>;

/// The tables of OpenFHE's negacyclic NTT over one prime `q`: `psi^brv(k)` and `psi^-brv(k)` for
/// `k < n`, with `psi` OpenFHE's 2n-th root of unity (`RootOfUnity`: the least primitive one),
/// and `n^-1 mod q`.
struct NttTable {
    forward: Vec<u64>,
    inverse: Vec<u64>,
    n_inverse: u64,
}

impl NttTable {
    fn new(q: u64, n: usize) -> Self {
        let mul = |a: u64, b: u64| ((a as u128 * b as u128) % q as u128) as u64;
        let pow = |mut base: u64, mut exponent: u64| {
            let mut result = 1u64;
            while exponent > 0 {
                if exponent & 1 == 1 {
                    result = mul(result, base);
                }
                base = mul(base, base);
                exponent >>= 1;
            }
            result
        };
        let order = 2 * n as u64;
        // A primitive 2n-th root: its n-th power is -1. The least one is among its odd powers.
        let root = (2..q)
            .map(|g| pow(g, (q - 1) / order))
            .find(|r| pow(*r, n as u64) == q - 1)
            .expect("q = 1 mod 2n has a primitive 2n-th root of unity");
        let psi = (1..order).step_by(2).map(|k| pow(root, k)).min().expect("n > 0");
        let psi_inverse = pow(psi, q - 2);
        let bits = n.trailing_zeros();
        let reverse =
            |k: usize| if bits == 0 { 0 } else { k.reverse_bits() >> (usize::BITS - bits) };
        Self {
            forward: (0..n).map(|k| pow(psi, reverse(k) as u64)).collect(),
            inverse: (0..n).map(|k| pow(psi_inverse, reverse(k) as u64)).collect(),
            n_inverse: pow(n as u64 % q, q - 2),
        }
    }
}

/// Evaluation of a graph expression with the bindings in scope of a lowering: those that depend
/// on a loop index are substituted first, so an expression that reads one fails to evaluate
/// instead of taking an outer value.
trait EvaluateIn {
    fn evaluate_in(&self, lowering: &Lowering<'_>) -> Result<BigInt, crate::expr::ExprError>;
}

impl EvaluateIn for GraphExpr {
    fn evaluate_in(&self, lowering: &Lowering<'_>) -> Result<BigInt, crate::expr::ExprError> {
        substitute_vars(self, &lowering.subst).evaluate(&lowering.env)
    }
}

/// `expr` with each variable bound in `subst` replaced by its expression.
fn substitute_vars(expr: &GraphExpr, subst: &BTreeMap<String, GraphExpr>) -> GraphExpr {
    if subst.is_empty() {
        return expr.clone();
    }
    let sub = |e: &GraphExpr| Box::new(substitute_vars(e, subst));
    match expr {
        GraphExpr::Var(name) => subst.get(name).cloned().unwrap_or_else(|| expr.clone()),
        GraphExpr::Add(a, b) => GraphExpr::Add(sub(a), sub(b)),
        GraphExpr::Sub(a, b) => GraphExpr::Sub(sub(a), sub(b)),
        GraphExpr::Mul(a, b) => GraphExpr::Mul(sub(a), sub(b)),
        GraphExpr::Div(a, b) => GraphExpr::Div(sub(a), sub(b)),
        GraphExpr::FloorDiv(a, b) => GraphExpr::FloorDiv(sub(a), sub(b)),
        GraphExpr::Rem(a, b) => GraphExpr::Rem(sub(a), sub(b)),
        GraphExpr::RoundDiv(a, b) => GraphExpr::RoundDiv(sub(a), sub(b)),
        GraphExpr::Log2Ceil(a) => GraphExpr::Log2Ceil(sub(a)),
        GraphExpr::Select { selector, branches } => GraphExpr::Select {
            selector: sub(selector),
            branches: branches.iter().map(|b| substitute_vars(b, subst)).collect(),
        },
        GraphExpr::RingCrtModulus { ring, index } => {
            GraphExpr::RingCrtModulus { ring: ring.clone(), index: sub(index) }
        }
        GraphExpr::Const(_) |
        GraphExpr::LoopIndex(_) |
        GraphExpr::RingModulus(_) |
        GraphExpr::RingCrtDepth(_) => expr.clone(),
    }
}

/// `constants[l]` for limb `l`.
fn per_limb(constants: Vec<u64>) -> IntExpr {
    IntExpr::Select {
        selector: Box::new(IntExpr::LoopIndex(LIMB)),
        branches: constants.into_iter().map(IntExpr::from).collect(),
    }
}

/// An independent loop over `limbs` limbs whose body gets limb `l` of each matrix input (the
/// other inputs whole) and builds the result limb with `body`.
fn limb_loop(
    builder: &mut ProgramBuilder,
    definition: &mut DefinitionBuilder,
    input_types: Vec<CoreType>,
    per_limb: &[bool],
    limbs: usize,
    inputs: Vec<Value>,
    body: LimbBody<'_>,
) -> Value {
    let mut definition_body = DefinitionBuilder::new("limb", input_types);
    let operands: Vec<Value> = per_limb
        .iter()
        .enumerate()
        .map(|(i, by_limb)| {
            let input = definition_body.param(i);
            if *by_limb {
                definition_body
                    .builtin(Builtin::Get { index: IntExpr::LoopIndex(LIMB) }, vec![input])
            } else {
                input
            }
        })
        .collect();
    let limb = body(&mut definition_body, operands);
    let body = builder.add_definition(definition_body.finish(vec![limb]));
    definition.add_loop(
        Loop {
            count: limbs.into(),
            index_slot: LIMB,
            body,
            bindings: Vec::new(),
            carried: Vec::new(),
            inputs,
            until: false,
        },
        1,
    )[0]
}
