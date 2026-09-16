//! Fixed topology lowering for the prepared GPU compiler.

use mxx_ir_core::{
    ParamEnv, ValidatedGraph,
    expr::{IntExpr, RealExpr},
    graph::FrozenGraphScopeId,
    node::{
        ConcatAxis, IndexRange, IntBinaryOp, IntCompareOp, LoopInputMode, MatrixBinaryOp, NodeKind,
        RealBinaryOp,
    },
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

#[path = "gpu_prepared_scope.rs"]
mod scope;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedLoweringError {
    WrongNodeKind(&'static str),
    InvalidContract(&'static str),
    InvalidIndex,
    InvalidRange,
    InvalidLoop,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum PreparedFormat {
    Coefficient,
    Evaluation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValueLocation {
    pub owner: u64,
    pub rows: Range<usize>,
    pub columns: Range<usize>,
    pub level: usize,
    pub format: PreparedFormat,
    pub device: i32,
}

impl ValueLocation {
    pub fn shape(&self) -> (usize, usize) {
        (self.rows.end - self.rows.start, self.columns.end - self.columns.start)
    }

    fn same_owner(&self, other: &Self) -> bool {
        self.owner == other.owner &&
            self.level == other.level &&
            self.format == other.format &&
            self.device == other.device
    }
}

/// Stable logical-to-physical binding identity. Storage is populated by the
/// append-only provisioning transaction after lowering selects the topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PreparedBindingId {
    pub owner: u64,
    pub device: i32,
    pub instance: usize,
    pub storage: Option<PreparedStorageBinding>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PreparedStorageBinding {
    pub storage_id: u64,
    pub slot_id: u64,
    pub slot_index: usize,
    pub context: usize,
    /// CRT basis identity carried with the physical slot.  Context identity
    /// alone is insufficient when one context owns several bases.
    pub basis: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedCopy {
    pub source: ValueLocation,
    pub destination: ValueLocation,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedView {
    Alias(ValueLocation),
    TransposeAlias(ValueLocation),
    FixedCopies(Box<[FixedCopy]>),
}

fn require_node<'a>(
    kind: &'a NodeKind,
    expected: &'static str,
) -> Result<&'a NodeKind, PreparedLoweringError> {
    if kind_name(kind) == expected {
        Ok(kind)
    } else {
        Err(PreparedLoweringError::WrongNodeKind(expected))
    }
}

fn kind_name(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::Input { .. } => "Input",
        NodeKind::ConstantInt(_) => "ConstantInt",
        NodeKind::EvaluateInt(_) => "EvaluateInt",
        NodeKind::ConstantReal(_) => "ConstantReal",
        NodeKind::ConstantBool(_) => "ConstantBool",
        NodeKind::ConstantMatrix { .. } => "ConstantMatrix",
        NodeKind::GadgetTrapdoor { .. } => "GadgetTrapdoor",
        NodeKind::TrapdoorPublic => "TrapdoorPublic",
        NodeKind::IntBinary(_) => "IntBinary",
        NodeKind::IntCompare(_) => "IntCompare",
        NodeKind::BitExtract { .. } => "BitExtract",
        NodeKind::IntToReal => "IntToReal",
        NodeKind::BoolToInt => "BoolToInt",
        NodeKind::RealBinary(_) => "RealBinary",
        NodeKind::RealSqrt => "RealSqrt",
        NodeKind::MatrixBinary(_) => "MatrixBinary",
        NodeKind::MatrixMulAccumulate { .. } => "MatrixMulAccumulate",
        NodeKind::MatrixMulSmallRhs => "MatrixMulSmallRhs",
        NodeKind::MatrixNegate => "MatrixNegate",
        NodeKind::MatrixScale { .. } => "MatrixScale",
        NodeKind::RingAutomorphism { .. } => "RingAutomorphism",
        NodeKind::ModulusSwitch { .. } => "ModulusSwitch",
        NodeKind::ModulusReduce { .. } => "ModulusReduce",
        NodeKind::CenteredRebase { .. } => "CenteredRebase",
        NodeKind::RnsModUp { .. } => "RnsModUp",
        NodeKind::RnsModDown { .. } => "RnsModDown",
        NodeKind::CenteredExtend { .. } => "CenteredExtend",
        NodeKind::BlockModSwitch { .. } => "BlockModSwitch",
        NodeKind::Transpose => "Transpose",
        NodeKind::Slice { .. } => "Slice",
        NodeKind::Tensor => "Tensor",
        NodeKind::Concat { .. } => "Concat",
        NodeKind::UniformResidueSample { .. } => "UniformResidueSample",
        NodeKind::UniformIntervalSample { .. } => "UniformIntervalSample",
        NodeKind::GaussianSample { .. } => "GaussianSample",
        NodeKind::HashSample { .. } => "HashSample",
        NodeKind::TrapdoorSample { .. } => "TrapdoorSample",
        NodeKind::PreimageSample { .. } => "PreimageSample",
        NodeKind::GadgetDecompose { .. } => "GadgetDecompose",
        NodeKind::ExtractCoefficient { .. } => "ExtractCoefficient",
        NodeKind::LiftIntegerToConstantPolynomial { .. } => "LiftIntegerToConstantPolynomial",
        NodeKind::ThresholdDecode { .. } => "ThresholdDecode",
        NodeKind::CrtRecompose { .. } => "CrtRecompose",
        NodeKind::PackPolynomialCoefficients { .. } => "PackPolynomialCoefficients",
        NodeKind::PolynomialFromValues { .. } => "PolynomialFromValues",
        NodeKind::PolynomialValues { .. } => "PolynomialValues",
        NodeKind::SubgraphCall(_) => "SubgraphCall",
        NodeKind::ParallelLoop(_) => "ParallelLoop",
        NodeKind::SequentialLoop(_) => "SequentialLoop",
        NodeKind::FamilyPack { .. } => "FamilyPack",
        NodeKind::FamilyGetStatic { .. } => "FamilyGetStatic",
        NodeKind::FamilyGetDynamic => "FamilyGetDynamic",
        NodeKind::Select { .. } => "Select",
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ScalarValue {
    Int(BigInt),
    Real(f64),
    Bool(bool),
    Runtime(usize),
    /// A value already materialized in the prepared scalar slot table.
    Slot(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalarOpcode {
    IntBinary(IntBinaryOp),
    IntCompare(IntCompareOp),
    BitExtract,
    IntToReal,
    BoolToInt,
    RealBinary(RealBinaryOp),
    RealSqrt,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarInstruction {
    pub opcode: ScalarOpcode,
    pub operands: Box<[usize]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedScalar {
    pub constants: Box<[ScalarValue]>,
    pub instructions: Box<[ScalarInstruction]>,
    pub result: usize,
}

fn constant_int(
    expression: &IntExpr,
    bindings: &ParamEnv,
) -> Result<ScalarValue, PreparedLoweringError> {
    expression
        .evaluate(bindings)
        .map(ScalarValue::Int)
        .map_err(|_| PreparedLoweringError::InvalidContract("integer expression is not constant"))
}

fn constant_real(
    expression: &RealExpr,
    bindings: &ParamEnv,
) -> Result<ScalarValue, PreparedLoweringError> {
    expression
        .evaluate_f64(bindings)
        .map(ScalarValue::Real)
        .map_err(|_| PreparedLoweringError::InvalidContract("real expression is not constant"))
}

/// Lower one scalar node. Expressions become constants; runtime scalar inputs
/// remain bytecode operands and are never re-parsed during submit.
pub fn lower_scalar(
    kind: &NodeKind,
    bindings: &ParamEnv,
    operand_slots: &[usize],
) -> Result<PreparedScalar, PreparedLoweringError> {
    let mut constants = Vec::new();
    let mut instructions = Vec::new();
    let push_constant = |constants: &mut Vec<ScalarValue>, value: ScalarValue| {
        let index = constants.len();
        constants.push(value);
        index
    };
    let opcode = match kind {
        NodeKind::ConstantInt(value) => {
            let result = push_constant(&mut constants, ScalarValue::Int(value.clone()));
            return Ok(PreparedScalar {
                constants: constants.into_boxed_slice(),
                instructions: instructions.into_boxed_slice(),
                result,
            });
        }
        NodeKind::EvaluateInt(expression) => {
            let result = push_constant(&mut constants, constant_int(expression, bindings)?);
            return Ok(PreparedScalar {
                constants: constants.into_boxed_slice(),
                instructions: instructions.into_boxed_slice(),
                result,
            });
        }
        NodeKind::ConstantReal(expression) => {
            let result = push_constant(&mut constants, constant_real(expression, bindings)?);
            return Ok(PreparedScalar {
                constants: constants.into_boxed_slice(),
                instructions: instructions.into_boxed_slice(),
                result,
            });
        }
        NodeKind::ConstantBool(value) => {
            let result = push_constant(&mut constants, ScalarValue::Bool(*value));
            return Ok(PreparedScalar {
                constants: constants.into_boxed_slice(),
                instructions: instructions.into_boxed_slice(),
                result,
            });
        }
        NodeKind::IntBinary(operation) => ScalarOpcode::IntBinary(*operation),
        NodeKind::IntCompare(operation) => ScalarOpcode::IntCompare(*operation),
        NodeKind::BitExtract { .. } => ScalarOpcode::BitExtract,
        NodeKind::IntToReal => ScalarOpcode::IntToReal,
        NodeKind::BoolToInt => ScalarOpcode::BoolToInt,
        NodeKind::RealBinary(operation) => ScalarOpcode::RealBinary(*operation),
        NodeKind::RealSqrt => ScalarOpcode::RealSqrt,
        NodeKind::Input { .. } |
        NodeKind::ConstantMatrix { .. } |
        NodeKind::GadgetTrapdoor { .. } |
        NodeKind::TrapdoorPublic |
        NodeKind::MatrixBinary(_) |
        NodeKind::MatrixMulAccumulate { .. } |
        NodeKind::MatrixMulSmallRhs |
        NodeKind::MatrixNegate |
        NodeKind::MatrixScale { .. } |
        NodeKind::RingAutomorphism { .. } |
        NodeKind::ModulusSwitch { .. } |
        NodeKind::ModulusReduce { .. } |
        NodeKind::CenteredRebase { .. } |
        NodeKind::RnsModUp { .. } |
        NodeKind::RnsModDown { .. } |
        NodeKind::CenteredExtend { .. } |
        NodeKind::BlockModSwitch { .. } |
        NodeKind::Transpose |
        NodeKind::Slice { .. } |
        NodeKind::Tensor |
        NodeKind::Concat { .. } |
        NodeKind::UniformResidueSample { .. } |
        NodeKind::UniformIntervalSample { .. } |
        NodeKind::GaussianSample { .. } |
        NodeKind::HashSample { .. } |
        NodeKind::TrapdoorSample { .. } |
        NodeKind::PreimageSample { .. } |
        NodeKind::GadgetDecompose { .. } |
        NodeKind::ExtractCoefficient { .. } |
        NodeKind::LiftIntegerToConstantPolynomial { .. } |
        NodeKind::ThresholdDecode { .. } |
        NodeKind::CrtRecompose { .. } |
        NodeKind::PackPolynomialCoefficients { .. } |
        NodeKind::PolynomialFromValues { .. } |
        NodeKind::PolynomialValues { .. } |
        NodeKind::SubgraphCall(_) |
        NodeKind::ParallelLoop(_) |
        NodeKind::SequentialLoop(_) |
        NodeKind::FamilyPack { .. } |
        NodeKind::FamilyGetStatic { .. } |
        NodeKind::FamilyGetDynamic |
        NodeKind::Select { .. } => {
            return Err(PreparedLoweringError::WrongNodeKind("scalar node"));
        }
    };
    if operand_slots.is_empty() {
        return Err(PreparedLoweringError::InvalidContract(
            "runtime scalar operation has no operands",
        ));
    }
    constants.extend(operand_slots.iter().copied().map(ScalarValue::Slot));
    let operands = (0..operand_slots.len()).collect::<Vec<_>>().into_boxed_slice();
    instructions.push(ScalarInstruction { opcode, operands });
    let result = constants.len();
    Ok(PreparedScalar {
        constants: constants.into_boxed_slice(),
        instructions: instructions.into_boxed_slice(),
        result,
    })
}

pub fn lower_slice(
    kind: &NodeKind,
    bindings: &ParamEnv,
    source: ValueLocation,
    destination: ValueLocation,
) -> Result<PreparedView, PreparedLoweringError> {
    let NodeKind::Slice { rows, columns } = kind else {
        return Err(PreparedLoweringError::WrongNodeKind("Slice"));
    };
    let resolve_index = |expression: &IntExpr| {
        expression
            .evaluate(bindings)
            .ok()
            .and_then(|value| value.to_usize())
            .ok_or(PreparedLoweringError::InvalidIndex)
    };
    let resolve_range = |range: &IndexRange| {
        let start = resolve_index(&range.start)?;
        let end = resolve_index(&range.end)?;
        if start >= end {
            return Err(PreparedLoweringError::InvalidRange);
        }
        Ok(start..end)
    };
    let source_rows = rows.as_ref().map_or_else(
        || Ok(source.rows.clone()),
        |range| {
            let range = resolve_range(range)?;
            Ok(source.rows.start + range.start..source.rows.start + range.end)
        },
    )?;
    let source_columns = columns.as_ref().map_or_else(
        || Ok(source.columns.clone()),
        |range| {
            let range = resolve_range(range)?;
            Ok(source.columns.start + range.start..source.columns.start + range.end)
        },
    )?;
    if source_rows.end - source_rows.start != destination.rows.end - destination.rows.start ||
        source_columns.end - source_columns.start !=
            destination.columns.end - destination.columns.start
    {
        return Err(PreparedLoweringError::InvalidRange);
    }
    let selected_source =
        ValueLocation { rows: source_rows.clone(), columns: source_columns.clone(), ..source };
    if selected_source.same_owner(&destination) {
        return Ok(PreparedView::Alias(ValueLocation {
            rows: source_rows,
            columns: source_columns,
            ..destination
        }));
    }
    Ok(PreparedView::FixedCopies(
        vec![FixedCopy { source: selected_source, destination }].into_boxed_slice(),
    ))
}

pub fn lower_transpose(
    kind: &NodeKind,
    source: ValueLocation,
    destination: ValueLocation,
) -> Result<PreparedView, PreparedLoweringError> {
    require_node(kind, "Transpose")?;
    let (rows, columns) = source.shape();
    if destination.shape() != (columns, rows) {
        return Err(PreparedLoweringError::InvalidRange);
    }
    let transposed_source = ValueLocation { rows: source.columns, columns: source.rows, ..source };
    if transposed_source.same_owner(&destination) {
        return Ok(PreparedView::TransposeAlias(ValueLocation {
            rows: transposed_source.rows,
            columns: transposed_source.columns,
            ..destination
        }));
    }
    Ok(PreparedView::FixedCopies(
        vec![FixedCopy { source: transposed_source, destination }].into_boxed_slice(),
    ))
}

pub fn lower_concat(
    kind: &NodeKind,
    inputs: &[ValueLocation],
    destination: ValueLocation,
) -> Result<PreparedView, PreparedLoweringError> {
    let NodeKind::Concat { axis } = kind else {
        return Err(PreparedLoweringError::WrongNodeKind("Concat"));
    };
    if inputs.is_empty() {
        return Err(PreparedLoweringError::InvalidContract("concat needs an input"));
    }
    let aliasable = destination.same_owner(&inputs[0]) &&
        inputs.iter().all(|input| input.same_owner(&inputs[0])) &&
        match axis {
            ConcatAxis::Rows => inputs.iter().all(|input| input.columns == inputs[0].columns),
            ConcatAxis::Columns | ConcatAxis::Diagonal => {
                inputs.iter().all(|input| input.rows == inputs[0].rows)
            }
        };
    if aliasable && *axis != ConcatAxis::Diagonal {
        return Ok(PreparedView::Alias(destination));
    }
    let mut row_offset = destination.rows.start;
    let mut column_offset = destination.columns.start;
    let copies = inputs
        .iter()
        .map(|input| {
            let rows = input.rows.end - input.rows.start;
            let columns = input.columns.end - input.columns.start;
            let (target_rows, target_columns) = match axis {
                ConcatAxis::Rows => {
                    let target = row_offset..row_offset + rows;
                    row_offset += rows;
                    (target, destination.columns.clone())
                }
                ConcatAxis::Columns => {
                    let target = column_offset..column_offset + columns;
                    column_offset += columns;
                    (destination.rows.clone(), target)
                }
                ConcatAxis::Diagonal => {
                    let target_rows = row_offset..row_offset + rows;
                    let target_columns = column_offset..column_offset + columns;
                    row_offset += rows;
                    column_offset += columns;
                    (target_rows, target_columns)
                }
            };
            FixedCopy {
                source: input.clone(),
                destination: ValueLocation {
                    owner: destination.owner,
                    rows: target_rows,
                    columns: target_columns,
                    level: destination.level,
                    format: destination.format,
                    device: destination.device,
                },
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Ok(PreparedView::FixedCopies(copies))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedFamily {
    pub members: Box<[ValueLocation]>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedSelection {
    Static { location: ValueLocation },
    Dynamic { candidates: Box<[ValueLocation]>, selector: WireRef },
    Select { candidates: Box<[ValueLocation]>, selector: WireRef },
    ScalarStatic { slot: usize },
    ScalarDynamic { candidates: Box<[usize]>, selector: WireRef },
    ScalarSelect { candidates: Box<[usize]>, selector: WireRef },
}

impl PreparedSelection {
    pub fn with_selector(self, selector: WireRef) -> Self {
        match self {
            Self::Dynamic { candidates, .. } => Self::Dynamic { candidates, selector },
            Self::Select { candidates, .. } => Self::Select { candidates, selector },
            Self::ScalarDynamic { candidates, .. } => Self::ScalarDynamic { candidates, selector },
            Self::ScalarSelect { candidates, .. } => Self::ScalarSelect { candidates, selector },
            static_selection @ Self::Static { .. } => static_selection,
            static_selection @ Self::ScalarStatic { .. } => static_selection,
        }
    }
}

pub fn lower_scalar_family(
    kind: &NodeKind,
    candidates: Box<[usize]>,
    selector: WireRef,
    static_index: Option<usize>,
) -> Result<PreparedSelection, PreparedLoweringError> {
    match kind {
        NodeKind::FamilyGetStatic { .. } => Ok(PreparedSelection::ScalarStatic {
            slot: candidates
                .get(static_index.ok_or(PreparedLoweringError::InvalidIndex)?)
                .copied()
                .ok_or(PreparedLoweringError::InvalidIndex)?,
        }),
        NodeKind::FamilyGetDynamic => Ok(PreparedSelection::ScalarDynamic { candidates, selector }),
        NodeKind::Select { .. } => Ok(PreparedSelection::ScalarSelect { candidates, selector }),
        _ => Err(PreparedLoweringError::WrongNodeKind("scalar family selection node")),
    }
}

pub fn lower_family(
    kind: &NodeKind,
    family: &PreparedFamily,
    static_index: Option<usize>,
) -> Result<PreparedSelection, PreparedLoweringError> {
    match kind {
        NodeKind::FamilyGetStatic { .. } => {
            let index = static_index.ok_or(PreparedLoweringError::InvalidIndex)?;
            let location =
                family.members.get(index).cloned().ok_or(PreparedLoweringError::InvalidIndex)?;
            Ok(PreparedSelection::Static { location })
        }
        NodeKind::FamilyGetDynamic => Ok(PreparedSelection::Dynamic {
            candidates: family.members.clone(),
            selector: WireRef { node: NodeId(0), port: Port(0) },
        }),
        NodeKind::Select { .. } => Ok(PreparedSelection::Select {
            candidates: family.members.clone(),
            selector: WireRef { node: NodeId(0), port: Port(0) },
        }),
        _ => Err(PreparedLoweringError::WrongNodeKind("family selection node")),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedTopologyNode {
    pub id: u32,
    pub command: PreparedCommandRequirement,
    pub stream: u32,
    pub waits: Box<[u32]>,
    pub completion: u32,
}

/// The fixed kind of work represented by a topology node.  This is deliberately
/// independent of the IR spelling: an input or constant is consumed during
/// warmup, views remain aliases, and only the remaining kinds need a fixed
/// command at replay time.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedOperation {
    Warmup,
    Scalar,
    Alias,
    Selection,
    ParallelLoop,
    SequentialLoop,
    Gpu(PreparedGpuOperation),
}

/// Native operation required by a prepared topology node.  Keeping this as a
/// closed enum makes the warmup contract structural: adding a node kind must
/// add its fixed command requirement, rather than relying on a graph name or
/// a string-based shape test.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedGpuOperation {
    MatrixBinary(MatrixBinaryOp),
    Transpose,
    ConcatRows,
    FixedCopies,
    MatrixMulAccumulate,
    MatrixMulSmallRhs,
    MatrixNegate,
    MatrixScale,
    RingAutomorphism,
    ModulusSwitch,
    ModulusReduce,
    CenteredRebase,
    RnsModUp,
    RnsModDown,
    CenteredExtend,
    BlockModSwitch,
    Tensor,
    UniformResidueSample,
    UniformIntervalSample,
    GaussianSample,
    HashSample,
    HashCompactDecompose,
    TrapdoorSample,
    PreimageSample,
    GadgetDecompose,
    ExtractCoefficient,
    LiftIntegerToConstantPolynomial,
    ThresholdDecode,
    CrtRecompose,
    RnsUpload,
    RnsReadback,
    PackPolynomialCoefficients,
    SubgraphCall,
    FamilyPack,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedCommandRequirement {
    pub operation: PreparedOperation,
    pub inputs: usize,
    pub outputs: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedTopologyEdge {
    pub from: u32,
    pub to: u32,
    pub event: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Default)]
pub struct PreparedTopology {
    pub nodes: Box<[PreparedTopologyNode]>,
    pub edges: Box<[PreparedTopologyEdge]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedOutputKind {
    Matrix,
    SmallMatrix,
    Scalar,
    HostReconstruction,
    Family,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedOutputBinding {
    pub name: String,
    pub wire: WireRef,
    pub kind: PreparedOutputKind,
}

/// A logical leaf of a root family input.  The public runtime value remains a
/// nested `IndexedFamily`; prepared replay binds each leaf to one fixed input
/// slot using this immutable root/path descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedInputLeaf {
    pub root: WireRef,
    pub path: Box<[usize]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedNodeSource {
    pub kind: NodeKind,
    pub environment: ParamEnv,
    pub variants: Box<[NodeKind]>,
    pub variant_indices: Box<[usize]>,
    pub variant_input_types: Box<[Box<[ConcreteWireType]>]>,
    pub variant_output_types: Box<[Box<[ConcreteWireType]>]>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PreparedReplayStep {
    Node(u32),
    Subgraph {
        body: Box<[PreparedReplayStep]>,
    },
    Parallel {
        counts: Box<[usize]>,
        waves: Box<[Box<[PreparedReplayStep]>]>,
    },
    Sequential {
        count: usize,
        counts: Box<[usize]>,
        offsets: Box<[usize]>,
        banks: [Box<[PreparedReplayStep]>; 2],
        tail: Box<[PreparedReplayStep]>,
    },
}

impl PreparedNodeSource {
    pub fn kind(&self) -> &NodeKind {
        &self.kind
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PreparedProgram {
    pub replay: Box<[PreparedReplayStep]>,
    /// Warmup-only source metadata, retained for native command construction.
    pub node_sources: BTreeMap<u32, PreparedNodeSource>,
    pub wire_types: BTreeMap<WireRef, ConcreteWireType>,
    /// Number of concurrently reusable execution instances reserved for this
    /// graph. Warmup fills this from the admitted wave width.
    pub instance_count: usize,
    pub topology: PreparedTopology,
    pub values: BTreeMap<WireRef, ValueLocation>,
    pub inputs: Box<[WireRef]>,
    pub outputs: Box<[WireRef]>,
    pub bindings: BTreeMap<WireRef, PreparedBindingId>,
    pub instance_bindings: Box<[BTreeMap<WireRef, PreparedBindingId>]>,
    pub instance_storage_bindings: Box<[BTreeMap<PreparedBindingId, PreparedStorageBinding>]>,
    pub node_bindings: BTreeMap<u32, (Box<[WireRef]>, Box<[WireRef]>)>,
    pub scalar_commands: BTreeMap<u32, PreparedScalar>,
    /// Stable scalar slots assigned by lowering.  Replay uses these bindings
    /// for inputs, constants, and intermediate results instead of inferring
    /// positions from a node's local argument count.
    pub scalar_slots: BTreeMap<WireRef, usize>,
    pub scalar_slot_count: usize,
    pub scalar_initializers: BTreeMap<usize, ScalarValue>,
    /// Maximum accepted host integer width for each root input. Zero denotes
    /// a non-integer input. The width is fixed by warmup and is checked before
    /// a live instance is acquired.
    pub scalar_input_max_words: Box<[usize]>,
    /// Scalar wires produced on the device (or depending on one). They must
    /// never acquire a duplicate host bytecode command.
    pub device_scalar_wires: BTreeSet<WireRef>,
    pub input_names: Box<[(String, WireRef)]>,
    /// Runtime binding order after expanding root-family leaves. Root scalar,
    /// matrix, byte, and compact inputs appear directly; family roots are
    /// replaced by their leaf wires.
    pub runtime_input_wires: Box<[WireRef]>,
    /// Root-input index for each runtime leaf, fixed during warmup so replay
    /// can bind directly without rebuilding a root/path map.
    pub runtime_input_roots: Box<[usize]>,
    pub input_leaf_bindings: BTreeMap<WireRef, PreparedInputLeaf>,
    pub output_names: Box<[(String, WireRef)]>,
    pub output_bindings: Box<[PreparedOutputBinding]>,
    pub view_commands: BTreeMap<u32, PreparedView>,
    pub selection_commands: BTreeMap<u32, PreparedSelection>,
    /// The fixed members captured by each FamilyPack.  Family selection
    /// commands refer to this table by their family wire; replay never has to
    /// inspect the family node or rebuild its candidate list.
    pub family_members: BTreeMap<WireRef, Box<[ValueLocation]>>,
    pub family_wires: BTreeMap<WireRef, Box<[WireRef]>>,
}

fn prepared_operation(kind: &NodeKind) -> PreparedOperation {
    match kind {
        NodeKind::Input { .. } |
        NodeKind::ConstantInt(_) |
        NodeKind::EvaluateInt(_) |
        NodeKind::ConstantReal(_) |
        NodeKind::ConstantBool(_) |
        NodeKind::ConstantMatrix { .. } |
        NodeKind::GadgetTrapdoor { .. } |
        NodeKind::TrapdoorPublic => PreparedOperation::Warmup,
        NodeKind::IntBinary(_) |
        NodeKind::IntCompare(_) |
        NodeKind::BitExtract { .. } |
        NodeKind::IntToReal |
        NodeKind::BoolToInt |
        NodeKind::RealBinary(_) |
        NodeKind::RealSqrt => PreparedOperation::Scalar,
        NodeKind::Transpose => PreparedOperation::Gpu(PreparedGpuOperation::Transpose),
        NodeKind::Concat { axis: ConcatAxis::Rows } => {
            PreparedOperation::Gpu(PreparedGpuOperation::ConcatRows)
        }
        NodeKind::Slice { .. } | NodeKind::Concat { .. } => PreparedOperation::Alias,
        NodeKind::FamilyGetStatic { .. } | NodeKind::FamilyGetDynamic | NodeKind::Select { .. } => {
            PreparedOperation::Selection
        }
        NodeKind::ParallelLoop(_) => PreparedOperation::ParallelLoop,
        NodeKind::SequentialLoop(_) => PreparedOperation::SequentialLoop,
        NodeKind::MatrixBinary(operation) => {
            PreparedOperation::Gpu(PreparedGpuOperation::MatrixBinary(*operation))
        }
        NodeKind::MatrixMulAccumulate { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::MatrixMulAccumulate)
        }
        NodeKind::MatrixMulSmallRhs => {
            PreparedOperation::Gpu(PreparedGpuOperation::MatrixMulSmallRhs)
        }
        NodeKind::MatrixNegate => PreparedOperation::Gpu(PreparedGpuOperation::MatrixNegate),
        NodeKind::MatrixScale { .. } => PreparedOperation::Gpu(PreparedGpuOperation::MatrixScale),
        NodeKind::RingAutomorphism { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::RingAutomorphism)
        }
        NodeKind::ModulusSwitch { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::ModulusSwitch)
        }
        NodeKind::ModulusReduce { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::ModulusReduce)
        }
        NodeKind::CenteredRebase { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::CenteredRebase)
        }
        NodeKind::RnsModUp { .. } => PreparedOperation::Gpu(PreparedGpuOperation::RnsModUp),
        NodeKind::RnsModDown { .. } => PreparedOperation::Gpu(PreparedGpuOperation::RnsModDown),
        NodeKind::CenteredExtend { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::CenteredExtend)
        }
        NodeKind::BlockModSwitch { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::BlockModSwitch)
        }
        NodeKind::Tensor => PreparedOperation::Gpu(PreparedGpuOperation::Tensor),
        NodeKind::UniformResidueSample { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::UniformResidueSample)
        }
        NodeKind::UniformIntervalSample { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::UniformIntervalSample)
        }
        NodeKind::GaussianSample { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::GaussianSample)
        }
        NodeKind::HashSample { variant, .. } => match variant {
            mxx_ir_core::node::HashVariant::Plain => {
                PreparedOperation::Gpu(PreparedGpuOperation::HashSample)
            }
            mxx_ir_core::node::HashVariant::Decomposed |
            mxx_ir_core::node::HashVariant::SmallDecomposed => {
                PreparedOperation::Gpu(PreparedGpuOperation::HashCompactDecompose)
            }
        },
        NodeKind::TrapdoorSample { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::TrapdoorSample)
        }
        NodeKind::PreimageSample { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::PreimageSample)
        }
        NodeKind::GadgetDecompose { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::GadgetDecompose)
        }
        NodeKind::ExtractCoefficient { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::ExtractCoefficient)
        }
        NodeKind::LiftIntegerToConstantPolynomial { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::LiftIntegerToConstantPolynomial)
        }
        NodeKind::ThresholdDecode { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::ThresholdDecode)
        }
        NodeKind::CrtRecompose { .. } => PreparedOperation::Gpu(PreparedGpuOperation::CrtRecompose),
        NodeKind::PackPolynomialCoefficients { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::PackPolynomialCoefficients)
        }
        NodeKind::PolynomialFromValues { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::RnsUpload)
        }
        NodeKind::PolynomialValues { .. } => {
            PreparedOperation::Gpu(PreparedGpuOperation::RnsReadback)
        }
        NodeKind::SubgraphCall(_) => PreparedOperation::ParallelLoop,
        NodeKind::FamilyPack { .. } => PreparedOperation::Selection,
    }
}

impl PreparedTopology {
    /// Return the immutable replay order.  Dependencies are represented by
    /// completion event IDs in `waits`; this ordering is only a submission
    /// order and never waits on the host.
    pub fn replay_order(&self) -> Result<Box<[u32]>, PreparedLoweringError> {
        let ids = self.nodes.iter().map(|node| node.id).collect::<BTreeSet<_>>();
        let mut indegree = self
            .nodes
            .iter()
            .map(|node| {
                let dependencies = node
                    .waits
                    .iter()
                    .copied()
                    .filter(|wait| ids.contains(wait))
                    .collect::<BTreeSet<_>>();
                (node.id, dependencies.len())
            })
            .collect::<BTreeMap<_, _>>();
        let mut ready = self
            .nodes
            .iter()
            .filter(|node| indegree[&node.id] == 0)
            .map(|node| node.id)
            .collect::<BTreeSet<_>>();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(id) = ready.pop_first() {
            order.push(id);
            for node in &self.nodes {
                if node.waits.iter().any(|wait| *wait == id) {
                    let remaining =
                        indegree.get_mut(&node.id).expect("topology node has an indegree entry");
                    *remaining -= 1;
                    if *remaining == 0 {
                        ready.insert(node.id);
                    }
                }
            }
        }
        if order.len() != self.nodes.len() {
            return Err(PreparedLoweringError::InvalidContract("topology contains a cycle"));
        }
        Ok(order.into_boxed_slice())
    }
}

fn matrix_location(
    owner: u64,
    ty: &ConcreteWireType,
    format: PreparedFormat,
) -> Option<ValueLocation> {
    let matrix = ty.matrix_type()?;
    Some(ValueLocation {
        owner,
        rows: 0..matrix.rows,
        columns: 0..matrix.columns,
        level: 0,
        format,
        device: 0,
    })
}

/// Lower a validated scope into the fixed topology consumed by warmup.  This
/// pass is deliberately independent of native allocation: the resulting wire
/// locations and event DAG are the stable contract used by later provisioning.
pub fn lower_graph(
    graph: &ValidatedGraph,
    wave_bound: std::num::NonZeroUsize,
) -> Result<PreparedProgram, PreparedLoweringError> {
    let mut program = scope::instantiate(graph, wave_bound)?;
    let mut device_scalars = BTreeSet::new();
    for node in &program.topology.nodes {
        let Some(source) = program.node_sources.get(&node.id) else { continue };
        let (arguments, outputs) = program.node_bindings.get(&node.id).cloned().unwrap_or_default();
        if matches!(source.kind, NodeKind::PackPolynomialCoefficients { .. }) {
            for argument in &arguments {
                let members = family_leaf_wires(&program, *argument);
                device_scalars.extend(
                    members.into_iter().filter(|wire| program.scalar_slots.contains_key(wire)),
                );
            }
        }
        let threshold = matches!(source.kind, NodeKind::ThresholdDecode { .. });
        let dependent = arguments.iter().any(|wire| device_scalars.contains(wire));
        if !threshold && !dependent {
            continue;
        }
        for wire in outputs {
            if matches!(
                program.wire_types.get(&wire),
                Some(
                    ConcreteWireType::Int |
                        ConcreteWireType::Bool |
                        ConcreteWireType::Real |
                        ConcreteWireType::IndexedFamily { .. }
                )
            ) {
                device_scalars.insert(wire);
            }
        }
    }
    program.device_scalar_wires = device_scalars;
    // Device-derived chains cannot jump back to host bytecode. Include their
    // scalar ancestors (runtime inputs/constants included), then propagate to
    // aliases and families until the finite wire set is closed.
    loop {
        let previous = program.device_scalar_wires.len();
        let device_slots = program
            .device_scalar_wires
            .iter()
            .filter_map(|wire| program.scalar_slots.get(wire))
            .copied()
            .collect::<BTreeSet<_>>();
        program.device_scalar_wires.extend(
            program
                .scalar_slots
                .iter()
                .filter_map(|(wire, slot)| device_slots.contains(slot).then_some(*wire)),
        );
        for node in &program.topology.nodes {
            let Some((arguments, outputs)) = program.node_bindings.get(&node.id) else { continue };
            let scalar = |wire: &WireRef| {
                matches!(
                    program.wire_types.get(wire),
                    Some(
                        ConcreteWireType::Int |
                            ConcreteWireType::Bool |
                            ConcreteWireType::Real |
                            ConcreteWireType::ConstantInt |
                            ConcreteWireType::ConstantBool |
                            ConcreteWireType::ConstantReal |
                            ConcreteWireType::IndexedFamily { .. }
                    )
                )
            };
            if arguments
                .iter()
                .chain(outputs.iter())
                .any(|wire| program.device_scalar_wires.contains(wire))
            {
                for wire in arguments.iter().chain(outputs.iter()).filter(|wire| scalar(wire)) {
                    program.device_scalar_wires.insert(*wire);
                }
            }
        }
        if program.device_scalar_wires.len() == previous {
            break;
        }
    }
    Ok(program)
}

pub fn family_leaf_wires(program: &PreparedProgram, wire: WireRef) -> Vec<WireRef> {
    let Some(members) = program.family_wires.get(&wire) else { return vec![wire] };
    members.iter().flat_map(|member| family_leaf_wires(program, *member)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        node::{
            ConcatAxis, ConstantMatrix, HashTagComponent, HashVariant, IntBinaryOp, IntCompareOp,
            LoopInputMode, MatrixBinaryOp, ParallelLoop, SampleRange, SequentialLoop, SubgraphCall,
        },
        types::{MatrixType, WireType},
    };

    fn location(owner: u64, rows: Range<usize>, columns: Range<usize>) -> ValueLocation {
        ValueLocation {
            owner,
            rows,
            columns,
            level: 2,
            format: PreparedFormat::Evaluation,
            device: 3,
        }
    }

    fn matrix_type() -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        }
    }

    #[test]
    fn prepared_binding_retains_binary_semantics_and_slot_identity() {
        assert_eq!(
            prepared_operation(&NodeKind::MatrixBinary(MatrixBinaryOp::Subtract)),
            PreparedOperation::Gpu(PreparedGpuOperation::MatrixBinary(MatrixBinaryOp::Subtract))
        );
        let binding = PreparedBindingId {
            owner: 7,
            device: 3,
            instance: 1,
            storage: Some(PreparedStorageBinding {
                storage_id: 11,
                slot_id: 13,
                slot_index: 2,
                context: 17,
                basis: 3,
            }),
        };
        let storage = binding.storage.unwrap();
        assert_eq!((storage.storage_id, storage.slot_id, storage.slot_index), (11, 13, 2));
    }

    #[test]
    fn every_node_kind_has_an_explicit_lowering_category() {
        let z = || IntExpr::constant(0);
        let one = || IntExpr::constant(1);
        let real_one = || RealExpr::from_integer(1);
        let all = vec![
            (
                "Input",
                NodeKind::Input { name: "x".into(), wire_type: WireType::Int, artifact: None },
            ),
            ("ConstantInt", NodeKind::ConstantInt(BigInt::from(1))),
            ("EvaluateInt", NodeKind::EvaluateInt(z())),
            ("ConstantReal", NodeKind::ConstantReal(real_one())),
            ("ConstantBool", NodeKind::ConstantBool(true)),
            (
                "ConstantMatrix",
                NodeKind::ConstantMatrix {
                    matrix_type: matrix_type(),
                    value: ConstantMatrix::Zero,
                },
            ),
            (
                "GadgetTrapdoor",
                NodeKind::GadgetTrapdoor { matrix_type: matrix_type(), base: one() },
            ),
            ("TrapdoorPublic", NodeKind::TrapdoorPublic),
            ("IntBinary", NodeKind::IntBinary(IntBinaryOp::Add)),
            ("IntCompare", NodeKind::IntCompare(IntCompareOp::Equal)),
            ("BitExtract", NodeKind::BitExtract { bit: z() }),
            ("IntToReal", NodeKind::IntToReal),
            ("BoolToInt", NodeKind::BoolToInt),
            ("RealBinary", NodeKind::RealBinary(RealBinaryOp::Add)),
            ("RealSqrt", NodeKind::RealSqrt),
            ("MatrixBinary", NodeKind::MatrixBinary(MatrixBinaryOp::Add)),
            (
                "MatrixMulAccumulate",
                NodeKind::MatrixMulAccumulate { coefficients: vec![one()], has_bias: true },
            ),
            ("MatrixMulSmallRhs", NodeKind::MatrixMulSmallRhs),
            ("MatrixNegate", NodeKind::MatrixNegate),
            ("MatrixScale", NodeKind::MatrixScale { scalar: one() }),
            ("RingAutomorphism", NodeKind::RingAutomorphism { index: one() }),
            ("ModulusSwitch", NodeKind::ModulusSwitch { modulus: one() }),
            ("ModulusReduce", NodeKind::ModulusReduce { modulus: one() }),
            ("CenteredRebase", NodeKind::CenteredRebase { modulus: one() }),
            (
                "RnsModUp",
                NodeKind::RnsModUp {
                    modulus: one(),
                    source_moduli: vec![17],
                    digit_size: 1,
                    normalize: true,
                },
            ),
            (
                "RnsModDown",
                NodeKind::RnsModDown {
                    modulus: one(),
                    source_moduli: vec![17],
                    plaintext_modulus: one(),
                },
            ),
            ("CenteredExtend", NodeKind::CenteredExtend { modulus: one() }),
            (
                "BlockModSwitch",
                NodeKind::BlockModSwitch { modulus: one(), plaintext_modulus: one() },
            ),
            ("Transpose", NodeKind::Transpose),
            ("Slice", NodeKind::Slice { rows: None, columns: None }),
            ("Tensor", NodeKind::Tensor),
            ("Concat", NodeKind::Concat { axis: ConcatAxis::Rows }),
            ("UniformResidueSample", NodeKind::UniformResidueSample { matrix_type: matrix_type() }),
            (
                "UniformIntervalSample",
                NodeKind::UniformIntervalSample {
                    matrix_type: matrix_type(),
                    range: SampleRange { minimum: z(), maximum: one() },
                },
            ),
            (
                "GaussianSample",
                NodeKind::GaussianSample {
                    matrix_type: matrix_type(),
                    sigma: real_one(),
                    max_coefficient_bound: one(),
                },
            ),
            (
                "HashSample",
                NodeKind::HashSample {
                    matrix_type: matrix_type(),
                    variant: HashVariant::Plain,
                    tag_prefix: vec![],
                    tag_components: vec![HashTagComponent::Integer(z())],
                    base: None,
                    digit_count: None,
                },
            ),
            (
                "TrapdoorSample",
                NodeKind::TrapdoorSample {
                    matrix_type: matrix_type(),
                    sigma: real_one(),
                    gadget_base: one(),
                    digit_count: one(),
                    preimage_max_coefficient_bound: one(),
                },
            ),
            (
                "PreimageSample",
                NodeKind::PreimageSample {
                    matrix_type: matrix_type(),
                    max_coefficient_bound: one(),
                },
            ),
            (
                "GadgetDecompose",
                NodeKind::GadgetDecompose { base: one(), small: false, digit_count: one() },
            ),
            (
                "ExtractCoefficient",
                NodeKind::ExtractCoefficient {
                    position: z(),
                    canonical_input_exclusive_upper: None,
                },
            ),
            (
                "LiftIntegerToConstantPolynomial",
                NodeKind::LiftIntegerToConstantPolynomial { matrix_type: matrix_type() },
            ),
            (
                "ThresholdDecode",
                NodeKind::ThresholdDecode {
                    plaintext_modulus: one(),
                    length: one(),
                    output_bool: true,
                },
            ),
            (
                "CrtRecompose",
                NodeKind::CrtRecompose {
                    modulus: one(),
                    plaintext_moduli: vec![one()],
                    reconstruction_coefficients: vec![one()],
                },
            ),
            (
                "PackPolynomialCoefficients",
                NodeKind::PackPolynomialCoefficients {
                    matrix_type: matrix_type(),
                    coefficient_bits: one(),
                },
            ),
            (
                "PolynomialFromValues",
                NodeKind::PolynomialFromValues { matrix_type: matrix_type(), evaluation: true },
            ),
            ("PolynomialValues", NodeKind::PolynomialValues { evaluation: true }),
            (
                "SubgraphCall",
                NodeKind::SubgraphCall(SubgraphCall {
                    definition: "body".into(),
                    bindings: vec![],
                    canonical_input_exclusive_uppers: vec![],
                }),
            ),
            (
                "ParallelLoop",
                NodeKind::ParallelLoop(ParallelLoop {
                    count: one(),
                    minimum_count: 0,
                    index_slot: 0,
                    bindings: vec![],
                    input_modes: vec![LoopInputMode::Broadcast],
                }),
            ),
            (
                "SequentialLoop",
                NodeKind::SequentialLoop(SequentialLoop {
                    count: one(),
                    index_slot: 0,
                    bindings: vec![],
                    carried_count: 1,
                }),
            ),
            ("FamilyPack", NodeKind::FamilyPack { count: one() }),
            ("FamilyGetStatic", NodeKind::FamilyGetStatic { index: z() }),
            ("FamilyGetDynamic", NodeKind::FamilyGetDynamic),
            ("Select", NodeKind::Select { count: one() }),
        ];
        for (expected, kind) in all {
            assert_eq!(kind_name(&kind), expected, "unclassified NodeKind: {kind:?}");
        }
    }

    #[test]
    fn scalar_nodes_have_constant_or_fixed_bytecode_lowering() {
        let env = ParamEnv::default();
        assert_eq!(
            lower_scalar(&NodeKind::ConstantInt(BigInt::from(7)), &env, &[]).unwrap().result,
            0
        );
        assert_eq!(lower_scalar(&NodeKind::ConstantBool(true), &env, &[]).unwrap().result, 0);
        let evaluated =
            lower_scalar(&NodeKind::EvaluateInt(IntExpr::constant(11)), &env, &[]).unwrap();
        assert_eq!(evaluated.constants.as_ref(), &[ScalarValue::Int(BigInt::from(11))]);
        let real = lower_scalar(
            &NodeKind::ConstantReal(RealExpr::from_f64_exact(1.25).unwrap()),
            &env,
            &[],
        )
        .unwrap();
        assert_eq!(real.constants.as_ref(), &[ScalarValue::Real(1.25)]);
        for kind in [
            NodeKind::IntBinary(IntBinaryOp::Add),
            NodeKind::IntCompare(IntCompareOp::Equal),
            NodeKind::BitExtract { bit: IntExpr::constant(0) },
            NodeKind::IntToReal,
            NodeKind::BoolToInt,
            NodeKind::RealBinary(RealBinaryOp::Multiply),
            NodeKind::RealSqrt,
        ] {
            let scalar = lower_scalar(&kind, &env, &[0, 1]).unwrap();
            assert_eq!(scalar.instructions.len(), 1);
            assert_eq!(scalar.constants.as_ref(), &[ScalarValue::Slot(0), ScalarValue::Slot(1)]);
        }
    }

    #[test]
    fn views_distinguish_aliases_and_fixed_copies() {
        let source = location(1, 0..4, 0..4);
        let mut coefficient = source.clone();
        coefficient.format = PreparedFormat::Coefficient;
        assert_eq!(coefficient.shape(), (4, 4));
        let slice = NodeKind::Slice { rows: None, columns: None };
        assert!(matches!(
            lower_slice(&slice, &ParamEnv::default(), source.clone(), location(1, 0..4, 0..4))
                .unwrap(),
            PreparedView::Alias(_)
        ));
        let sliced = NodeKind::Slice {
            rows: Some(mxx_ir_core::node::IndexRange {
                start: IntExpr::constant(1),
                end: IntExpr::constant(3),
            }),
            columns: None,
        };
        let PreparedView::Alias(sliced) =
            lower_slice(&sliced, &ParamEnv::default(), source.clone(), location(1, 0..2, 0..4))
                .unwrap()
        else {
            panic!("slice must be an alias");
        };
        assert_eq!(sliced.rows, 1..3);
        assert!(matches!(
            lower_slice(&slice, &ParamEnv::default(), source.clone(), location(9, 0..4, 0..4))
                .unwrap(),
            PreparedView::FixedCopies(copies) if copies.len() == 1
        ));
        assert!(matches!(
            lower_transpose(&NodeKind::Transpose, source.clone(), location(1, 0..4, 0..4)).unwrap(),
            PreparedView::TransposeAlias(_)
        ));
        assert!(matches!(
            lower_transpose(&NodeKind::Transpose, source.clone(), location(9, 0..4, 0..4))
                .unwrap(),
            PreparedView::FixedCopies(copies) if copies.len() == 1
        ));
        let other = location(2, 0..4, 0..4);
        let concat = NodeKind::Concat { axis: ConcatAxis::Diagonal };
        assert!(matches!(
            lower_concat(&concat, &[source, other], location(3, 0..8, 0..8)).unwrap(),
            PreparedView::FixedCopies(_)
        ));
        let source = location(1, 0..4, 0..4);
        let same_owner = location(1, 0..4, 0..4);
        assert!(matches!(
            lower_concat(
                &NodeKind::Concat { axis: ConcatAxis::Rows },
                &[source.clone(), same_owner.clone()],
                location(1, 0..8, 0..4)
            )
            .unwrap(),
            PreparedView::Alias(_)
        ));
        assert!(matches!(
            lower_concat(
                &NodeKind::Concat { axis: ConcatAxis::Columns },
                &[source, same_owner],
                location(9, 0..4, 0..8)
            )
            .unwrap(),
            PreparedView::FixedCopies(copies) if copies.len() == 2
        ));
    }

    #[test]
    fn family_and_selection_lower_to_fixed_descriptors() {
        let family = PreparedFamily {
            members: vec![location(1, 0..1, 0..1), location(2, 0..1, 0..1)].into_boxed_slice(),
        };
        assert!(matches!(
            lower_family(
                &NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
                &family,
                Some(1)
            )
            .unwrap(),
            PreparedSelection::Static { .. }
        ));
        assert!(matches!(
            lower_family(&NodeKind::FamilyGetDynamic, &family, None).unwrap(),
            PreparedSelection::Dynamic { candidates, .. } if candidates.len() == 2
        ));
        assert!(matches!(
            lower_family(&NodeKind::Select { count: IntExpr::constant(2) }, &family, None).unwrap(),
            PreparedSelection::Select { .. }
        ));
        assert!(matches!(
            lower_scalar_family(
                &NodeKind::FamilyGetDynamic,
                vec![3, 4].into_boxed_slice(),
                WireRef { node: NodeId(9), port: Port(0) },
                None,
            )
            .unwrap(),
            PreparedSelection::ScalarDynamic { candidates, .. } if candidates.as_ref() == [3, 4]
        ));
        assert!(matches!(
            lower_scalar_family(
                &NodeKind::FamilyGetStatic { index: IntExpr::constant(1) },
                vec![3, 4].into_boxed_slice(),
                WireRef { node: NodeId(9), port: Port(0) },
                Some(1),
            )
            .unwrap(),
            PreparedSelection::ScalarStatic { slot: 4 }
        ));
    }

    #[test]
    fn topology_replay_order_keeps_independent_nodes_parallel() {
        let topology = PreparedTopology {
            nodes: vec![
                PreparedTopologyNode {
                    id: 10,
                    command: PreparedCommandRequirement {
                        operation: PreparedOperation::Gpu(PreparedGpuOperation::MatrixBinary(
                            MatrixBinaryOp::Add,
                        )),
                        inputs: 1,
                        outputs: 1,
                    },
                    stream: 0,
                    waits: Box::new([]),
                    completion: 10,
                },
                PreparedTopologyNode {
                    id: 20,
                    command: PreparedCommandRequirement {
                        operation: PreparedOperation::Gpu(PreparedGpuOperation::Tensor),
                        inputs: 1,
                        outputs: 1,
                    },
                    stream: 1,
                    waits: Box::new([]),
                    completion: 20,
                },
                PreparedTopologyNode {
                    id: 30,
                    command: PreparedCommandRequirement {
                        operation: PreparedOperation::Gpu(
                            PreparedGpuOperation::MatrixMulAccumulate,
                        ),
                        inputs: 2,
                        outputs: 1,
                    },
                    stream: 2,
                    waits: vec![10, 20].into_boxed_slice(),
                    completion: 30,
                },
            ]
            .into_boxed_slice(),
            edges: vec![
                PreparedTopologyEdge { from: 10, to: 30, event: 10 },
                PreparedTopologyEdge { from: 20, to: 30, event: 20 },
            ]
            .into_boxed_slice(),
        };
        let order = topology.replay_order().unwrap();
        assert_eq!(order.last(), Some(&30));
        assert!(order[..2].contains(&10));
        assert!(order[..2].contains(&20));
    }
}
