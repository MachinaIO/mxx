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
use mxx_primitives::{
    matrix::gpu_dcrt_poly::{
        GpuPreparedAccumulateLayout, GpuPreparedRequest, PreparedAllocationLayout,
        PreparedOwnerLayout, PreparedPlanLayout, PreparedStreamFootprint,
    },
    sampler::trapdoor::gpu::{GpuPreparedPreimageLayout, GpuPreparedTrapdoorLayout},
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

/// A stable reference to one logical prepared value.  The index is assigned by
/// lowering and is never searched for by name during replay.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedDescriptorRef {
    pub wire: WireRef,
    pub instance: usize,
    pub store: Option<usize>,
    pub kind: PreparedDescriptorKind,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedDescriptorKind {
    Matrix,
    Scalar,
    Host,
    Family,
}

/// Pure metadata for one physical value class.  This is intentionally a
/// logical store description; native allocation and slot assignment remain a
/// later provisioning concern.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedStorePlan {
    pub wire: WireRef,
    pub wire_type: Option<ConcreteWireType>,
    pub location: ValueLocation,
    pub instance: usize,
    pub capacity_rows: usize,
    pub capacity_columns: usize,
    /// Logical command shape remains distinct from the colored owner's maximum
    /// capacity. Native stage planners consume this exact shape; allocation
    /// claims consume the capacity fields above.
    pub logical_rows: usize,
    pub logical_columns: usize,
    pub alignment: usize,
}

/// Logical stage identity captured by lowering. The native schedule remains
/// the sole authority for selecting concrete CUDA streams; no host-side stream
/// number is inferred here.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum PreparedStageRole {
    Matrix,
    Transfer,
    Sampling,
    Scalar,
    Selection,
    View,
    Schedule,
    Control,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PreparedStreamKey {
    pub stage: PreparedStageRole,
    pub instance: usize,
    /// All matrix descriptors participating in the command. Native binding
    /// resolves these immutable store identities to context/partition/device/
    /// limb placement; lowering never invents a CUDA stream or partition.
    pub stores: Box<[usize]>,
    pub placements: Box<[PreparedPlacementKey]>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PreparedPlacementKey {
    pub store: usize,
    pub owner: u64,
    pub device: i32,
    pub level: usize,
    pub format: PreparedFormat,
}

/// Closed native stage classification carried by the pure lowering plan. The
/// recipe contains structural identities only; it never contains a backend,
/// context pointer, CUDA handle, or live owner.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedNativeStage {
    Matrix(PreparedGpuOperation),
    Transfer(PreparedGpuOperation),
    Sampling(PreparedGpuOperation),
    ScalarBuffer,
    ScalarOp,
    ScalarMatrixSelect,
    Threshold,
    ScalarPack,
    Selection,
    View,
    Schedule,
    Control,
}

/// Structural shape of a scalar native resource.  Widths here describe the
/// initial prepared generation; arbitrary-width runtime integers may grow that
/// generation at the explicit input boundary and are accounted for separately
/// by the scalar capacity ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedScalarResource {
    Buffer { count: usize, words: usize, pinned_host_bytes: usize },
    Op { left_words: usize, right_words: usize, output_words: usize, candidate_count: usize },
    MatrixSelect { rows: usize, columns: usize, level: usize, count: usize },
    Threshold { count: usize, plaintext_words: usize },
    Pack { count: usize, coefficient_bits: usize, output_format: i32 },
}

/// Metadata for the fixed replay uploader associated with a command output.
/// This is deliberately independent of any native object: the resolver turns
/// it into a saved primitive descriptor and exact admission claims.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedReplayUploadRecipe {
    Matrix {
        rows: usize,
        columns: usize,
        level: usize,
        format: i32,
        payload_capacity: usize,
        codec_capacity: usize,
    },
    Small {
        rows: usize,
        columns: usize,
        level: usize,
        payload_bytes: usize,
        codec_capacity: usize,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PreparedOwnerKey {
    pub owner: u64,
    pub device: i32,
    pub instance: usize,
    pub level: usize,
    pub format: PreparedFormat,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedNativeRecipe {
    pub node: u32,
    /// The complete validated node payload, including ranges, words/bytes
    /// derivation inputs, distribution/phase parameters, scalar thresholds,
    /// and small-RHS/trapdoor/preimage settings. Keeping the typed payload
    /// intact prevents a later binder from replacing any of those values with
    /// a default or an observed runtime value.
    pub source: Option<PreparedNodeSource>,
    pub stage: PreparedNativeStage,
    pub inputs: Box<[usize]>,
    pub outputs: Box<[usize]>,
    pub owners: Box<[PreparedOwnerKey]>,
    pub completion: Option<u32>,
    /// Threshold decoding first copies its source into a fixed 1x1
    /// coefficient-domain matrix.  The store and owner are planned here so
    /// replay can consume the admitted matrix slot instead of constructing an
    /// unplanned staging owner.
    pub matrix_staging: Option<PreparedOwnerKey>,
    /// Exact scalar sub-stage contract. Matrix stages leave this unset.
    pub scalar: Option<PreparedScalarResource>,
    pub replay_upload: Option<PreparedReplayUploadRecipe>,
}

/// One scalar backing that is materialized during warmup.  Scalar values are
/// not matrix stores, but their native buffers still belong to the same
/// resolver transaction and therefore carry an explicit owner/context.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedScalarBufferPlan {
    pub wire: WireRef,
    pub instance: usize,
    pub owner: PreparedOwnerKey,
    pub count: usize,
    pub words: usize,
    pub pinned_host_bytes: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedCommandPlan {
    pub node: u32,
    pub instance: usize,
    pub operation: PreparedOperation,
    pub inputs: Box<[PreparedDescriptorRef]>,
    pub outputs: Box<[PreparedDescriptorRef]>,
    pub stream: Option<PreparedStreamKey>,
    pub waits: Box<[u32]>,
    pub completion: u32,
    pub recipe: PreparedNativeRecipe,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedSchedulePlan {
    pub instance: usize,
    pub stream: PreparedStreamKey,
    pub command_group: Box<[u32]>,
    pub waits: Box<[u32]>,
    pub completions: Box<[u32]>,
    pub recipe: PreparedNativeRecipe,
}

/// Pure preparation metadata consumed by the later native provisioning pass.
/// Building this value performs no CUDA calls, allocations, event creation,
/// stream creation, or kernel launches.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PreparedResourcePlan {
    /// Number of independent physical execution instances represented by the
    /// store, command, and schedule entries below.
    pub instance_count: usize,
    pub stores: Box<[PreparedStorePlan]>,
    pub scalar_buffers: Box<[PreparedScalarBufferPlan]>,
    pub commands: Box<[PreparedCommandPlan]>,
    pub schedules: Box<[PreparedSchedulePlan]>,
}

/// A physical owner resolved during warmup.  The native owner is deliberately
/// absent: this record is only the value layout needed by the later bind pass.
#[derive(Clone, Debug)]
pub struct PreparedResolvedOwner {
    pub key: PreparedOwnerKey,
    pub layout: PreparedOwnerLayout,
    /// The exact matrix slot admitted for this owner during warmup.  Replay
    /// must use this identity directly; it must not rediscover an owner by
    /// scanning the accepted inventory.
    pub slot: Option<PreparedSlotRef>,
}

impl PartialEq for PreparedResolvedOwner {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key &&
            self.layout.execution_owner_identity() == other.layout.execution_owner_identity() &&
            self.layout.stream_ordinal_base() == other.layout.stream_ordinal_base() &&
            self.layout.execution_class() == other.layout.execution_class() &&
            self.layout.partition_count() == other.layout.partition_count()
    }
}

/// An exact slot selected during the one warmup admission transaction.
///
/// The request is copied from the accepted storage identity; replay must pass
/// this value through the containing region and never rediscover a slot by
/// scanning the global inventory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreparedSlotRef {
    pub device: i32,
    pub request: GpuPreparedRequest,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedAllocationClaim {
    pub command: u32,
    pub ordinal: usize,
    pub store: Option<usize>,
    pub layout: PreparedAllocationLayout,
    pub slot: Option<PreparedSlotRef>,
}

/// One allocation claim belonging to a fused sampler tape.  Composite plans
/// cannot be represented by the command-level native descriptor alone: their
/// internal owners and phase resources are consumed in a second, fixed order.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedCompositeClaim {
    pub command: u32,
    pub ordinal: usize,
    pub claim: mxx_primitives::matrix::gpu_dcrt_poly::GpuTracedClaim,
    /// The native descriptor which produced this claim, when it is a native
    /// allocation.  Keeping the key beside the traced claim lets admission
    /// match stream footprints by device/partition/limb/role instead of by a
    /// fragile ordinal among unrelated workspace claims.
    pub layout: Option<mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationLayout>,
    pub slot: Option<PreparedSlotRef>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedCompositeStream {
    pub command: u32,
    pub ordinal: usize,
    pub layout: PreparedStreamFootprint,
    pub slot: Option<PreparedSlotRef>,
}

/// Validate the closed composite allocation/stream contract without touching
/// CUDA. Every native submission stream must have one and only one matching
/// allocation key, including its device, partition, limb and role fields.
pub fn validate_composite_stream_claims(
    allocations: &[PreparedCompositeClaim],
    streams: &[PreparedCompositeStream],
) -> Result<(), String> {
    let stream_allocations = allocations
        .iter()
        .filter(|allocation| {
            allocation.claim.kind() ==
                mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotKind::SubmissionStream
        })
        .collect::<Vec<_>>();
    let submitted_streams =
        streams.iter().filter(|stream| stream.layout.origin == 1).collect::<Vec<_>>();
    if stream_allocations.len() != submitted_streams.len() {
        return Err(format!(
            "composite stream/allocation claim count mismatch: {} != {}",
            submitted_streams.len(),
            stream_allocations.len()
        ));
    }
    let mut consumed = vec![false; stream_allocations.len()];
    for stream in submitted_streams {
        let matches = stream_allocations
            .iter()
            .enumerate()
            .filter(|(index, allocation)| {
                !consumed[*index] &&
                    allocation.layout.is_some_and(|layout| layout.key == stream.layout.key)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(format!(
                "composite stream key {:?} has {} matching allocation claims",
                stream.layout.key,
                matches.len()
            ));
        }
        consumed[matches[0]] = true;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedStreamClaim {
    pub command: u32,
    pub ordinal: usize,
    pub layout: PreparedStreamFootprint,
    pub slot: Option<PreparedSlotRef>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedResolvedCommand {
    pub command: PreparedCommandPlan,
    pub native: Option<PreparedPlanLayout>,
    /// Complete ordered descriptor bundle for a fused accumulate tape.
    /// `native` remains the command-level compatibility view for accounting;
    /// binders must consume this bundle for the actual sub-stages.
    pub accumulate: Option<GpuPreparedAccumulateLayout>,
    /// Complete sampler bundles resolved during warmup.  These are retained
    /// beside the command-level accounting view so binders cannot fall back to
    /// live geometry planning for composite trapdoor/preimage commands.
    pub preimage: Option<GpuPreparedPreimageLayout>,
    pub trapdoor: Option<GpuPreparedTrapdoorLayout>,
    pub allocations: Box<[PreparedAllocationClaim]>,
    pub composite_allocations: Box<[PreparedCompositeClaim]>,
    pub composite_streams: Box<[PreparedCompositeStream]>,
    pub streams: Box<[PreparedStreamClaim]>,
    pub replay_upload: Option<PreparedResolvedReplayUpload>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedResolvedReplayUpload {
    pub recipe: PreparedReplayUploadRecipe,
    pub layout: PreparedPlanLayout,
    pub allocations: Box<[PreparedAllocationClaim]>,
    pub streams: Box<[PreparedStreamClaim]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedResolvedSchedule {
    pub schedule: PreparedSchedulePlan,
    pub native: PreparedPlanLayout,
    pub allocations: Box<[PreparedAllocationClaim]>,
    pub streams: Box<[PreparedStreamClaim]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedResolvedScalarBuffer {
    pub plan: PreparedScalarBufferPlan,
    pub layout: PreparedPlanLayout,
    pub slots: Box<[Option<PreparedSlotRef>]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedResolvedResources {
    pub owners: Box<[PreparedResolvedOwner]>,
    pub scalar_buffers: Box<[PreparedResolvedScalarBuffer]>,
    pub commands: Box<[PreparedResolvedCommand]>,
    pub schedules: Box<[PreparedResolvedSchedule]>,
}

fn first_matrix_wire(program: &GpuPreparation, wire: WireRef) -> Option<WireRef> {
    if program.values.contains_key(&wire) {
        return Some(wire);
    }
    program
        .family_wires
        .get(&wire)
        .and_then(|members| members.iter().find_map(|member| first_matrix_wire(program, *member)))
}

pub fn physical_command_matches(
    command: &PreparedCommandPlan,
    node: u32,
    instance: usize,
    device: i32,
) -> bool {
    command.node == node &&
        command.instance == instance &&
        command.recipe.owners.iter().any(|owner| owner.device == device)
}

impl PreparedResolvedResources {
    /// Select the immutable resource slice for one physical shard. Every
    /// device-expanded command has owner keys for exactly one device, so a
    /// binder cannot accidentally consume the first device's layout for a
    /// different shard.
    pub fn for_device(&self, device: i32) -> Self {
        let owners = self
            .owners
            .iter()
            .filter(|owner| owner.key.device == device)
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let commands = self
            .commands
            .iter()
            .filter(|command| {
                command.command.recipe.owners.iter().any(|owner| owner.device == device)
            })
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let schedules = self
            .schedules
            .iter()
            .filter(|schedule| {
                schedule.schedule.recipe.owners.iter().any(|owner| owner.device == device)
            })
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let scalar_buffers = self
            .scalar_buffers
            .iter()
            .filter(|buffer| buffer.plan.owner.device == device)
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self { owners, scalar_buffers, commands, schedules }
    }
}

/// Backend hook used only at warmup. Implementations must call the primitive
/// metadata-only owner/stage planners and return their saved descriptors; they
/// must not create owners, reserve storage, or submit CUDA work.
pub trait PreparedResourceBackend {
    fn plan_owner(
        &self,
        key: &PreparedOwnerKey,
        store: &PreparedStorePlan,
        stream_ordinal_base: usize,
    ) -> Result<(PreparedOwnerLayout, usize), String>;

    fn plan_stage(
        &self,
        recipe: &PreparedNativeRecipe,
        stores: &[PreparedStorePlan],
        owners: &[PreparedResolvedOwner],
    ) -> Result<Option<PreparedPlanLayout>, String>;

    /// Plan composite sampler descriptors in the same warmup pass as the
    /// command-level accounting descriptor. Simple stages leave both values
    /// unset; trapdoor/preimage stages retain their complete ordered bundles
    /// for the native binder.
    fn plan_sampler_bundles(
        &self,
        _recipe: &PreparedNativeRecipe,
        _stores: &[PreparedStorePlan],
        _owners: &[PreparedResolvedOwner],
    ) -> Result<(Option<GpuPreparedPreimageLayout>, Option<GpuPreparedTrapdoorLayout>), String>
    {
        Ok((None, None))
    }

    fn plan_accumulate(
        &self,
        _recipe: &PreparedNativeRecipe,
        _stores: &[PreparedStorePlan],
        _owners: &[PreparedResolvedOwner],
    ) -> Result<Option<GpuPreparedAccumulateLayout>, String> {
        Ok(None)
    }

    fn plan_replay_upload(
        &self,
        recipe: &PreparedNativeRecipe,
        replay: &PreparedReplayUploadRecipe,
        stores: &[PreparedStorePlan],
        owners: &[PreparedResolvedOwner],
    ) -> Result<PreparedPlanLayout, String>;

    fn plan_schedule(
        &self,
        recipe: &PreparedNativeRecipe,
        members: &[PreparedPlanLayout],
    ) -> Result<PreparedPlanLayout, String>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
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
        call: Option<NodeId>,
        body: Box<[PreparedReplayStep]>,
    },
    Parallel {
        call: NodeId,
        counts: Box<[usize]>,
        waves: Box<[Box<[PreparedReplayStep]>]>,
    },
    Sequential {
        call: NodeId,
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
pub struct GpuPreparation {
    pub replay: Box<[PreparedReplayStep]>,
    /// Warmup-only source metadata, retained for native command construction.
    pub node_sources: BTreeMap<u32, PreparedNodeSource>,
    pub wire_types: BTreeMap<WireRef, ConcreteWireType>,
    /// Number of concurrently reusable execution instances reserved for this
    /// graph. Warmup fills this from `ExecutionConfig::max_live_gpu_executions`;
    /// it is independent of the graph's parallel-loop wave width.
    pub instance_count: usize,
    pub topology: PreparedTopology,
    pub values: BTreeMap<WireRef, ValueLocation>,
    pub inputs: Box<[WireRef]>,
    pub outputs: Box<[WireRef]>,
    /// Matrix/scalar wires retained by the prepared trace descriptors. These
    /// are populated during warmup so trace capture never needs to rediscover
    /// graph values or allocate a retention map during execute.
    pub trace_wires: Box<[WireRef]>,
    /// Stable ordinary identities for wires emitted from the root scope.
    /// Prepared execution uses these keys directly for trace/transcript
    /// publication instead of exposing virtual topology node ids.
    pub trace_keys: BTreeMap<WireRef, mxx_ir_core::types::WireId>,
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
    /// Maximum physical capacity per canonical owner after storage coloring.
    pub storage_capacities: BTreeMap<u64, (usize, usize)>,
    /// Immutable metadata-only resource view. Native bind/provisioning may
    /// consume this plan later, but lowering itself stays side-effect free.
    pub resource_plan: PreparedResourcePlan,
    /// The resolver's immutable result retained across the warmup boundary.
    /// Slot references are populated by the backend's single admission pass;
    /// replay consumes this table instead of rebuilding native metadata.
    pub resolved_resources: Option<PreparedResolvedResources>,
}

impl PreparedResourcePlan {
    /// Verify that every native command has a concrete structural descriptor
    /// before warmup publishes it. Non-GPU/control nodes intentionally have no
    /// stream footprint and therefore do not participate in this check.
    pub fn validate_for_warmup(&self) -> Result<(), PreparedLoweringError> {
        for command in &self.commands {
            let expected_stage = if matches!(command.operation, PreparedOperation::Selection) {
                match command.recipe.scalar {
                    Some(PreparedScalarResource::MatrixSelect { .. }) => {
                        PreparedNativeStage::ScalarMatrixSelect
                    }
                    Some(PreparedScalarResource::Op { .. }) => PreparedNativeStage::ScalarOp,
                    _ => PreparedNativeStage::Selection,
                }
            } else {
                prepared_native_stage(&command.operation)
            };
            if command.recipe.stage != expected_stage {
                return Err(PreparedLoweringError::InvalidContract(
                    "native recipe stage does not match command operation",
                ));
            }
            if matches!(command.operation, PreparedOperation::Gpu(_)) {
                let Some(ref stream) = command.stream else {
                    return Err(PreparedLoweringError::InvalidContract(
                        "GPU command has no structural stream key",
                    ));
                };
                if stream.stores.is_empty() ||
                    stream.stores.len() != stream.placements.len() ||
                    command.inputs.iter().any(|descriptor| {
                        descriptor.kind == PreparedDescriptorKind::Matrix &&
                            descriptor.store.is_none_or(|index| self.stores.get(index).is_none())
                    }) ||
                    command.outputs.iter().any(|descriptor| {
                        descriptor.kind == PreparedDescriptorKind::Matrix &&
                            descriptor.store.is_none_or(|index| self.stores.get(index).is_none())
                    })
                {
                    return Err(PreparedLoweringError::InvalidContract(
                        "GPU command has an unresolved matrix descriptor",
                    ));
                }
                if command.recipe.owners.is_empty() {
                    return Err(PreparedLoweringError::InvalidContract(
                        "GPU command has no native owner key",
                    ));
                }
                if matches!(
                    command.operation,
                    PreparedOperation::Gpu(PreparedGpuOperation::ThresholdDecode)
                ) {
                    let Some(staging) = command.recipe.matrix_staging else {
                        return Err(PreparedLoweringError::InvalidContract(
                            "threshold command has no planned matrix staging owner",
                        ));
                    };
                    if !command.recipe.owners.contains(&staging) ||
                        !stream.placements.iter().any(|placement| {
                            placement.owner == staging.owner &&
                                placement.device == staging.device &&
                                placement.level == staging.level &&
                                placement.format == staging.format
                        })
                    {
                        return Err(PreparedLoweringError::InvalidContract(
                            "threshold staging owner is absent from its stream",
                        ));
                    }
                }
                if stream.stores.iter().zip(&stream.placements).any(|(index, placement)| {
                    self.stores.get(*index).is_none_or(|store| {
                        store.instance != stream.instance ||
                            placement.store != *index ||
                            placement.owner != store.location.owner ||
                            placement.device != store.location.device ||
                            placement.level != store.location.level ||
                            placement.format != store.location.format
                    })
                }) {
                    return Err(PreparedLoweringError::InvalidContract(
                        "GPU stream key does not match its stores",
                    ));
                }
            }
        }
        if self.schedules.iter().any(|schedule| {
            schedule.stream.stores.is_empty() ||
                schedule.command_group.is_empty() ||
                schedule.recipe.stage != PreparedNativeStage::Schedule ||
                schedule.recipe.owners.is_empty()
        }) {
            return Err(PreparedLoweringError::InvalidContract(
                "GPU schedule has no structural stream group",
            ));
        }
        Ok(())
    }

    pub fn from_preparation(program: &GpuPreparation) -> Result<Self, PreparedLoweringError> {
        let mut seeds = Vec::<(WireRef, ValueLocation, usize, usize)>::new();
        for (wire, location) in &program.values {
            let key = (location.owner, location.device, location.level, location.format);
            if let Some((_, (_, _, existing_rows, existing_columns))) =
                seeds.iter_mut().enumerate().find(|(_, (_, candidate, _, _))| {
                    (candidate.owner, candidate.device, candidate.level, candidate.format) == key
                })
            {
                *existing_rows = (*existing_rows).max(location.rows.end);
                *existing_columns = (*existing_columns).max(location.columns.end);
            } else {
                let (capacity_rows, capacity_columns) = program
                    .storage_capacities
                    .get(&location.owner)
                    .copied()
                    .unwrap_or((location.rows.end, location.columns.end));
                seeds.push((*wire, location.clone(), capacity_rows, capacity_columns));
            }
        }
        let instances = program.instance_count.max(1);
        let mut stores = Vec::with_capacity(seeds.len() * instances);
        // Resolve every wire through its canonical physical identity.  A view
        // or another alias of the same colored owner must point at the one
        // store entry; using the first wire as the key would leave later
        // aliases unresolved (and could tempt the binder to allocate again).
        let mut store_index = BTreeMap::<(u64, i32, usize, PreparedFormat, usize), usize>::new();
        for instance in 0..instances {
            for (wire, location, capacity_rows, capacity_columns) in &seeds {
                let index = stores.len();
                let mut descriptor = location.clone();
                descriptor.rows = 0..*capacity_rows;
                descriptor.columns = 0..*capacity_columns;
                stores.push(PreparedStorePlan {
                    wire: *wire,
                    wire_type: program.wire_types.get(wire).cloned(),
                    location: descriptor,
                    instance,
                    capacity_rows: *capacity_rows,
                    capacity_columns: *capacity_columns,
                    logical_rows: location.rows.end,
                    logical_columns: location.columns.end,
                    alignment: 8,
                });
                store_index.insert(
                    (location.owner, location.device, location.level, location.format, instance),
                    index,
                );
            }
        }
        // Threshold replay uses a coefficient-domain 1x1 matrix as the input
        // to the native scalar kernel. It is not an IR value, so reserve it as
        // an explicit synthetic store before constructing command streams.
        let mut next_staging_owner = program
            .values
            .values()
            .map(|location| location.owner)
            .chain(program.storage_capacities.keys().copied())
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or(PreparedLoweringError::InvalidRange)?;
        let mut threshold_staging_stores = BTreeMap::<(u32, usize), usize>::new();
        for instance in 0..instances {
            for node in &program.topology.nodes {
                if !matches!(
                    node.command.operation,
                    PreparedOperation::Gpu(PreparedGpuOperation::ThresholdDecode)
                ) {
                    continue;
                }
                let source_wire = program
                    .node_bindings
                    .get(&node.id)
                    .and_then(|(inputs, _)| inputs.first())
                    .copied()
                    .and_then(|wire| first_matrix_wire(program, wire))
                    .ok_or(PreparedLoweringError::InvalidRange)?;
                let source =
                    program.values.get(&source_wire).ok_or(PreparedLoweringError::InvalidRange)?;
                let source_store = *store_index
                    .get(&(source.owner, source.device, source.level, source.format, instance))
                    .ok_or(PreparedLoweringError::InvalidRange)?;
                let owner = next_staging_owner;
                next_staging_owner =
                    next_staging_owner.checked_add(1).ok_or(PreparedLoweringError::InvalidRange)?;
                let location = ValueLocation {
                    owner,
                    rows: 0..1,
                    columns: 0..1,
                    level: source.level,
                    format: PreparedFormat::Coefficient,
                    device: source.device,
                };
                let index = stores.len();
                stores.push(PreparedStorePlan {
                    wire: source_wire,
                    wire_type: stores[source_store].wire_type.clone(),
                    location,
                    instance,
                    capacity_rows: 1,
                    capacity_columns: 1,
                    logical_rows: 1,
                    logical_columns: 1,
                    alignment: 8,
                });
                threshold_staging_stores.insert((node.id, instance), index);
            }
        }
        // Scalar backings are fixed warmup resources as well.  Keep one
        // descriptor per scalar wire and execution instance; runtime BigInt
        // growth is charged separately by the execution ledger.
        let mut scalar_counts = BTreeMap::<WireRef, usize>::new();
        for node in &program.topology.nodes {
            let Some(source) = program.node_sources.get(&node.id) else { continue };
            let NodeKind::ThresholdDecode { length, .. } = source.kind() else { continue };
            let Some(output) =
                program.node_bindings.get(&node.id).and_then(|(_, outputs)| outputs.first())
            else {
                continue
            };
            if !program.scalar_slots.contains_key(output) {
                continue;
            }
            let count = length
                .evaluate(&source.environment)
                .ok()
                .and_then(|value| value.to_usize())
                .unwrap_or(1);
            scalar_counts
                .entry(*output)
                .and_modify(|existing| *existing = (*existing).max(count))
                .or_insert(count);
        }
        let scalar_width = |wire: &WireRef| {
            program
                .scalar_slots
                .get(wire)
                .and_then(|slot| program.scalar_initializers.get(slot))
                .map(|value| match value {
                    ScalarValue::Int(value) => value.bits().div_ceil(64).saturating_add(1) as usize,
                    _ => 1,
                })
                .unwrap_or(1)
        };
        let mut scalar_buffers = Vec::new();
        for instance in 0..instances {
            let Some(anchor) = stores.iter().find(|store| store.instance == instance) else {
                continue;
            };
            for wire in program.scalar_slots.keys().copied() {
                let words = scalar_width(&wire).max(1);
                let count = scalar_counts.get(&wire).copied().unwrap_or(1).max(1);
                let pinned_host_bytes = count
                    .checked_mul(words.checked_add(1).ok_or(PreparedLoweringError::InvalidRange)?)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<u64>()))
                    .ok_or(PreparedLoweringError::InvalidRange)?;
                scalar_buffers.push(PreparedScalarBufferPlan {
                    wire,
                    instance,
                    owner: PreparedOwnerKey {
                        owner: anchor.location.owner,
                        device: anchor.location.device,
                        instance,
                        level: anchor.location.level,
                        format: anchor.location.format,
                    },
                    count,
                    words,
                    pinned_host_bytes,
                });
            }
        }
        fn descriptors(
            program: &GpuPreparation,
            wire: WireRef,
            instance: usize,
            store_index: &BTreeMap<(u64, i32, usize, PreparedFormat, usize), usize>,
            out: &mut Vec<PreparedDescriptorRef>,
        ) {
            if let Some(members) = program.family_wires.get(&wire) {
                for member in members {
                    descriptors(program, *member, instance, store_index, out);
                }
                return;
            }
            let kind = if program.values.contains_key(&wire) {
                PreparedDescriptorKind::Matrix
            } else if program.scalar_slots.contains_key(&wire) {
                PreparedDescriptorKind::Scalar
            } else if matches!(
                program.wire_types.get(&wire),
                Some(ConcreteWireType::IndexedFamily { .. })
            ) {
                PreparedDescriptorKind::Family
            } else {
                PreparedDescriptorKind::Host
            };
            let store = program.values.get(&wire).and_then(|location| {
                store_index
                    .get(&(
                        location.owner,
                        location.device,
                        location.level,
                        location.format,
                        instance,
                    ))
                    .copied()
            });
            out.push(PreparedDescriptorRef { wire, instance, store, kind });
        }
        fn stream_key(
            node: &PreparedTopologyNode,
            inputs: &[PreparedDescriptorRef],
            outputs: &[PreparedDescriptorRef],
            stores: &[PreparedStorePlan],
            instance: usize,
        ) -> Option<PreparedStreamKey> {
            if !matches!(node.command.operation, PreparedOperation::Gpu(_)) {
                return None;
            }
            let mut store_ids = inputs
                .iter()
                .chain(outputs.iter())
                .filter_map(|descriptor| descriptor.store)
                .collect::<Vec<_>>();
            store_ids.sort_unstable();
            store_ids.dedup();
            let placements = store_ids
                .iter()
                .filter_map(|index| {
                    stores.get(*index).map(|store| PreparedPlacementKey {
                        store: *index,
                        owner: store.location.owner,
                        device: store.location.device,
                        level: store.location.level,
                        format: store.location.format,
                    })
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            Some(PreparedStreamKey {
                stage: prepared_stage_role(&node.command.operation),
                instance,
                stores: store_ids.into_boxed_slice(),
                placements,
            })
        }
        let mut commands = Vec::with_capacity(program.topology.nodes.len() * instances);
        let mut schedules = Vec::<PreparedSchedulePlan>::new();
        for instance in 0..instances {
            for node in &program.topology.nodes {
                let (input_wires, output_wires) =
                    program.node_bindings.get(&node.id).cloned().unwrap_or_default();
                let mut inputs = Vec::new();
                let mut outputs = Vec::new();
                for wire in &input_wires {
                    descriptors(program, *wire, instance, &store_index, &mut inputs);
                }
                for wire in &output_wires {
                    descriptors(program, *wire, instance, &store_index, &mut outputs);
                }
                let stream = stream_key(node, &inputs, &outputs, &stores, instance);
                let stream = match stream {
                    Some(mut stream) => {
                        if let Some(&staging_store) =
                            threshold_staging_stores.get(&(node.id, instance))
                        {
                            if !stream.stores.contains(&staging_store) {
                                let mut store_ids = stream.stores.to_vec();
                                store_ids.push(staging_store);
                                store_ids.sort_unstable();
                                store_ids.dedup();
                                let placements = store_ids
                                    .iter()
                                    .filter_map(|index| {
                                        stores.get(*index).map(|store| PreparedPlacementKey {
                                            store: *index,
                                            owner: store.location.owner,
                                            device: store.location.device,
                                            level: store.location.level,
                                            format: store.location.format,
                                        })
                                    })
                                    .collect::<Vec<_>>();
                                stream.stores = store_ids.into_boxed_slice();
                                stream.placements = placements.into_boxed_slice();
                            }
                        }
                        Some(stream)
                    }
                    None => None,
                };
                if let Some(stream_key) = stream.clone() {
                    let existing = schedules.iter_mut().find(|schedule| {
                        schedule.instance == instance &&
                            schedule.stream.stage == stream_key.stage &&
                            schedule.stream.stores == stream_key.stores &&
                            schedule.stream.placements == stream_key.placements
                    });
                    if let Some(schedule) = existing {
                        let mut command_group = schedule.command_group.to_vec();
                        command_group.push(node.id);
                        schedule.command_group = command_group.into_boxed_slice();
                        let mut waits = schedule.waits.to_vec();
                        for wait in node.waits.iter().copied() {
                            if !waits.contains(&wait) {
                                waits.push(wait);
                            }
                        }
                        schedule.waits = waits.into_boxed_slice();
                        let mut completions = schedule.completions.to_vec();
                        completions.push(node.completion);
                        schedule.completions = completions.into_boxed_slice();
                    } else {
                        let schedule_recipe = PreparedNativeRecipe {
                            node: node.id,
                            source: program.node_sources.get(&node.id).cloned(),
                            stage: PreparedNativeStage::Schedule,
                            inputs: stream_key.stores.clone(),
                            outputs: Box::new([]),
                            owners: stream_key
                                .placements
                                .iter()
                                .map(|placement| PreparedOwnerKey {
                                    owner: placement.owner,
                                    device: placement.device,
                                    instance,
                                    level: placement.level,
                                    format: placement.format,
                                })
                                .collect::<Vec<_>>()
                                .into_boxed_slice(),
                            completion: None,
                            matrix_staging: None,
                            scalar: None,
                            replay_upload: None,
                        };
                        schedules.push(PreparedSchedulePlan {
                            instance,
                            stream: stream_key,
                            command_group: vec![node.id].into_boxed_slice(),
                            waits: node.waits.clone(),
                            completions: vec![node.completion].into_boxed_slice(),
                            recipe: schedule_recipe,
                        });
                    }
                }
                let mut recipe = prepared_recipe(
                    node.id,
                    program.node_sources.get(&node.id),
                    &node.command.operation,
                    &inputs,
                    &outputs,
                    stream.as_ref(),
                    Some(node.completion),
                );
                if let Some(&staging_store) = threshold_staging_stores.get(&(node.id, instance)) {
                    let store =
                        stores.get(staging_store).ok_or(PreparedLoweringError::InvalidRange)?;
                    recipe.matrix_staging = Some(PreparedOwnerKey {
                        owner: store.location.owner,
                        device: store.location.device,
                        instance,
                        level: store.location.level,
                        format: store.location.format,
                    });
                }
                if matches!(node.command.operation, PreparedOperation::Scalar) {
                    let width = |wire: &WireRef| {
                        program
                            .scalar_slots
                            .get(wire)
                            .and_then(|slot| program.scalar_initializers.get(slot))
                            .map(|value| match value {
                                ScalarValue::Int(value) => {
                                    value.bits().div_ceil(64).saturating_add(1) as usize
                                }
                                _ => 1,
                            })
                            .unwrap_or(1)
                    };
                    recipe.scalar = Some(PreparedScalarResource::Op {
                        left_words: input_wires.first().map(width).unwrap_or(1),
                        right_words: input_wires.get(1).map(width).unwrap_or(0),
                        output_words: output_wires.first().map(width).unwrap_or(1),
                        candidate_count: 0,
                    });
                }
                // Scalar commands do not have matrix descriptor stores in
                // their input/output list.  Still bind them to the canonical
                // matrix owner for this instance so the native planner can
                // resolve its context, device and stream deterministically.
                if matches!(node.command.operation, PreparedOperation::Scalar) ||
                    matches!(
                        recipe.stage,
                        PreparedNativeStage::ScalarPack | PreparedNativeStage::Threshold
                    )
                {
                    if let Some(store) = stores.iter().find(|store| store.instance == instance) {
                        let owner = recipe.matrix_staging.unwrap_or(PreparedOwnerKey {
                            owner: store.location.owner,
                            device: store.location.device,
                            instance,
                            level: store.location.level,
                            format: store.location.format,
                        });
                        recipe.owners = vec![PreparedOwnerKey {
                            owner: owner.owner,
                            device: owner.device,
                            instance,
                            level: owner.level,
                            format: owner.format,
                        }]
                        .into_boxed_slice();
                    }
                }
                if matches!(node.command.operation, PreparedOperation::Selection) {
                    if let Some(
                        PreparedSelection::ScalarStatic { .. } |
                        PreparedSelection::ScalarDynamic { .. } |
                        PreparedSelection::ScalarSelect { .. },
                    ) = program.selection_commands.get(&node.id)
                    {
                        recipe.stage = PreparedNativeStage::ScalarOp;
                        let candidate_count = match program.selection_commands.get(&node.id) {
                            Some(PreparedSelection::ScalarStatic { .. }) => 1,
                            Some(PreparedSelection::ScalarDynamic { candidates, .. }) |
                            Some(PreparedSelection::ScalarSelect { candidates, .. }) => {
                                candidates.len()
                            }
                            _ => 0,
                        };
                        recipe.scalar = Some(PreparedScalarResource::Op {
                            left_words: 1,
                            right_words: 0,
                            output_words: 1,
                            candidate_count,
                        });
                        if let Some(store) = stores.iter().find(|store| store.instance == instance)
                        {
                            recipe.owners = vec![PreparedOwnerKey {
                                owner: store.location.owner,
                                device: store.location.device,
                                instance,
                                level: store.location.level,
                                format: store.location.format,
                            }]
                            .into_boxed_slice();
                        }
                    }
                    if let Some(
                        PreparedSelection::Dynamic { candidates, selector } |
                        PreparedSelection::Select { candidates, selector },
                    ) = program.selection_commands.get(&node.id)
                    {
                        if program.scalar_slots.contains_key(selector) {
                            if let Some(output) =
                                outputs.first().and_then(|wire| program.values.get(&wire.wire))
                            {
                                recipe.stage = PreparedNativeStage::ScalarMatrixSelect;
                                recipe.scalar = Some(PreparedScalarResource::MatrixSelect {
                                    rows: output.rows.end - output.rows.start,
                                    columns: output.columns.end - output.columns.start,
                                    level: output.level,
                                    count: candidates.len(),
                                });
                                if let Some(store) =
                                    stores.iter().find(|store| store.instance == instance)
                                {
                                    recipe.owners = vec![PreparedOwnerKey {
                                        owner: store.location.owner,
                                        device: store.location.device,
                                        instance,
                                        level: store.location.level,
                                        format: store.location.format,
                                    }]
                                    .into_boxed_slice();
                                }
                            }
                        }
                    }
                }
                if let Some(replay) =
                    prepared_replay_upload_recipe(program, &node.command.operation, &outputs)
                {
                    recipe.replay_upload = Some(replay?);
                }
                commands.push(PreparedCommandPlan {
                    node: node.id,
                    instance,
                    operation: node.command.operation.clone(),
                    inputs: inputs.into_boxed_slice(),
                    outputs: outputs.into_boxed_slice(),
                    stream,
                    waits: node.waits.clone(),
                    completion: node.completion,
                    recipe,
                });
            }
        }
        let plan = Self {
            instance_count: instances,
            stores: stores.into_boxed_slice(),
            scalar_buffers: scalar_buffers.into_boxed_slice(),
            commands: commands.into_boxed_slice(),
            schedules: schedules.into_boxed_slice(),
        };
        plan.validate_for_warmup()?;
        Ok(plan)
    }

    /// Build one physical resource view for every device in a fleet.  The
    /// logical lowering uses device zero as its neutral placement, but native
    /// owners, streams and staging claims are device-context specific.  Keep
    /// the logical graph unchanged and expand its resource plan only at the
    /// warmup boundary where the fleet's physical devices are known.
    pub fn from_preparation_for_devices(
        program: &GpuPreparation,
        devices: &[i32],
    ) -> Result<Self, PreparedLoweringError> {
        if devices.is_empty() {
            return Self::from_preparation(program);
        }
        if devices.iter().copied().collect::<BTreeSet<_>>().len() != devices.len() {
            return Err(PreparedLoweringError::InvalidContract(
                "physical device expansion contains duplicate devices",
            ));
        }
        let mut plans = Vec::with_capacity(devices.len());
        for device in devices.iter().copied() {
            let mut physical = program.clone();
            for location in physical.values.values_mut() {
                location.device = device;
            }
            plans.push(Self::from_preparation(&physical)?);
        }
        Self::merge_physical_plans(plans)
    }

    fn merge_physical_plans(plans: Vec<Self>) -> Result<Self, PreparedLoweringError> {
        let instance_count = plans.first().map(|plan| plan.instance_count).unwrap_or(0);
        let mut stores = Vec::new();
        let mut commands = Vec::new();
        let mut schedules = Vec::new();
        let mut scalar_buffers = Vec::new();
        for plan in plans {
            let store_offset = stores.len();
            stores.extend(plan.stores.iter().cloned());
            let remap_store = |index: usize| index + store_offset;
            let remap_descriptor = |descriptor: PreparedDescriptorRef| PreparedDescriptorRef {
                store: descriptor.store.map(remap_store),
                ..descriptor
            };
            let remap_stream = |stream: &PreparedStreamKey| PreparedStreamKey {
                stores: stream.stores.iter().copied().map(remap_store).collect(),
                placements: stream
                    .placements
                    .iter()
                    .map(|placement| PreparedPlacementKey {
                        store: remap_store(placement.store),
                        ..*placement
                    })
                    .collect(),
                ..stream.clone()
            };
            for mut command in plan.commands.iter().cloned() {
                command.inputs = command.inputs.iter().copied().map(remap_descriptor).collect();
                command.outputs = command.outputs.iter().copied().map(remap_descriptor).collect();
                command.stream = command.stream.as_ref().map(remap_stream);
                command.recipe.inputs =
                    command.recipe.inputs.iter().copied().map(remap_store).collect();
                command.recipe.outputs =
                    command.recipe.outputs.iter().copied().map(remap_store).collect();
                commands.push(command);
            }
            for mut schedule in plan.schedules.iter().cloned() {
                schedule.stream = remap_stream(&schedule.stream);
                schedule.recipe.inputs =
                    schedule.recipe.inputs.iter().copied().map(remap_store).collect();
                schedule.recipe.outputs =
                    schedule.recipe.outputs.iter().copied().map(remap_store).collect();
                schedules.push(schedule);
            }
            scalar_buffers.extend(plan.scalar_buffers.iter().cloned());
        }
        let merged = Self {
            instance_count,
            stores: stores.into_boxed_slice(),
            commands: commands.into_boxed_slice(),
            schedules: schedules.into_boxed_slice(),
            scalar_buffers: scalar_buffers.into_boxed_slice(),
        };
        merged.validate_for_warmup()?;
        Ok(merged)
    }

    /// Resolve the pure logical plan into exact primitive descriptors during
    /// warmup. The backend hook is intentionally the only place that knows how
    /// to obtain a ring context; this pass itself stores no native owner and
    /// performs no provisioning. Every owner and command is resolved exactly
    /// once, preserving multiplicity across instances and stages.
    pub fn resolve_prepared_resources<B: PreparedResourceBackend>(
        &self,
        backend: &B,
    ) -> Result<PreparedResolvedResources, String> {
        self.validate_for_warmup()
            .map_err(|error| format!("invalid prepared resource plan: {error:?}"))?;
        let mut owner_keys = self
            .stores
            .iter()
            .map(|store| PreparedOwnerKey {
                owner: store.location.owner,
                device: store.location.device,
                instance: store.instance,
                level: store.location.level,
                format: store.location.format,
            })
            .collect::<Vec<_>>();
        for command in &self.commands {
            owner_keys.extend(command.recipe.owners.iter().copied());
        }
        owner_keys.extend(self.scalar_buffers.iter().map(|buffer| buffer.owner));
        owner_keys.sort_unstable();
        owner_keys.dedup();
        let mut cursors = BTreeMap::<(u64, i32, usize), usize>::new();
        let mut owners = Vec::with_capacity(owner_keys.len());
        for key in owner_keys {
            let store = self
                .stores
                .iter()
                .find(|store| {
                    store.location.owner == key.owner &&
                        store.location.device == key.device &&
                        store.instance == key.instance &&
                        store.location.level == key.level &&
                        store.location.format == key.format
                })
                .ok_or_else(|| format!("prepared owner {} has no matrix store", key.owner))?;
            let cursor = cursors.entry((key.owner, key.device, key.instance)).or_default();
            let (layout, consumed) = backend.plan_owner(&key, store, *cursor)?;
            *cursor = cursor.checked_add(consumed).ok_or("prepared owner cursor overflow")?;
            owners.push(PreparedResolvedOwner { key, layout, slot: None });
        }
        let mut scalar_buffers = Vec::with_capacity(self.scalar_buffers.len());
        for buffer in &self.scalar_buffers {
            let recipe = PreparedNativeRecipe {
                node: u32::MAX,
                source: None,
                stage: PreparedNativeStage::ScalarBuffer,
                inputs: Box::new([]),
                outputs: Box::new([]),
                owners: vec![buffer.owner].into_boxed_slice(),
                completion: None,
                matrix_staging: None,
                scalar: Some(PreparedScalarResource::Buffer {
                    count: buffer.count,
                    words: buffer.words,
                    pinned_host_bytes: buffer.pinned_host_bytes,
                }),
                replay_upload: None,
            };
            let layout = backend
                .plan_stage(&recipe, &self.stores, &owners)?
                .ok_or("scalar buffer native plan missing")?;
            let slots =
                layout.allocations().iter().map(|_| None).collect::<Vec<_>>().into_boxed_slice();
            scalar_buffers.push(PreparedResolvedScalarBuffer {
                plan: buffer.clone(),
                layout,
                slots,
            });
        }
        let mut commands = Vec::with_capacity(self.commands.len());
        for command in &self.commands {
            let native = backend.plan_stage(&command.recipe, &self.stores, &owners)?;
            let (preimage, trapdoor) =
                backend.plan_sampler_bundles(&command.recipe, &self.stores, &owners)?;
            let (composite_claims, composite_layouts) = preimage
                .as_ref()
                .map(|layout| (layout.claims(), layout.claim_layouts()))
                .or_else(|| {
                    trapdoor.as_ref().map(|layout| (layout.claims(), layout.claim_layouts()))
                })
                .unwrap_or_default();
            if composite_claims.len() != composite_layouts.len() {
                return Err(format!(
                    "prepared node {} composite claim/layout table mismatch",
                    command.node
                ));
            }
            let composite_allocations = composite_claims
                .into_iter()
                .zip(composite_layouts)
                .enumerate()
                .map(|(ordinal, (claim, layout))| PreparedCompositeClaim {
                    command: command.node,
                    ordinal,
                    claim,
                    layout,
                    slot: None,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let composite_streams = preimage
                .as_ref()
                .map(|layout| layout.streams())
                .or_else(|| trapdoor.as_ref().map(|layout| layout.streams()))
                .unwrap_or_default()
                .into_iter()
                .enumerate()
                .map(|(ordinal, layout)| PreparedCompositeStream {
                    command: command.node,
                    ordinal,
                    layout,
                    slot: None,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            if !composite_allocations.is_empty() {
                validate_composite_stream_claims(&composite_allocations, &composite_streams)
                    .map_err(|error| format!("prepared node {}: {error}", command.node))?;
            }
            let accumulate = backend.plan_accumulate(&command.recipe, &self.stores, &owners)?;
            let store = command
                .recipe
                .outputs
                .first()
                .copied()
                .or_else(|| command.recipe.inputs.first().copied());
            let composite = preimage.is_some() || trapdoor.is_some();
            let (allocations, streams) = if composite {
                // Composite binders consume the flattened substage table below;
                // retaining the command-level sampler allocation would reserve
                // it twice and leave an unconsumed permit entry.
                (
                    Box::<[PreparedAllocationClaim]>::default(),
                    Box::<[PreparedStreamClaim]>::default(),
                )
            } else {
                match native.as_ref() {
                    Some(native) => {
                        let allocations = native
                            .allocations()
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(ordinal, layout)| PreparedAllocationClaim {
                                command: command.node,
                                ordinal,
                                store,
                                layout,
                                slot: None,
                            })
                            .collect::<Vec<_>>()
                            .into_boxed_slice();
                        let streams = native
                            .streams()
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(ordinal, layout)| PreparedStreamClaim {
                                command: command.node,
                                ordinal,
                                layout,
                                slot: None,
                            })
                            .collect::<Vec<_>>()
                            .into_boxed_slice();
                        (allocations, streams)
                    }
                    None if matches!(
                        command.recipe.stage,
                        PreparedNativeStage::Control |
                            PreparedNativeStage::View |
                            PreparedNativeStage::Selection
                    ) =>
                    {
                        (
                            Box::<[PreparedAllocationClaim]>::default(),
                            Box::<[PreparedStreamClaim]>::default(),
                        )
                    }
                    None => return Err(format!("native plan missing for command {}", command.node)),
                }
            };
            let replay_upload = command
                .recipe
                .replay_upload
                .as_ref()
                .map(|replay| -> Result<PreparedResolvedReplayUpload, String> {
                    let layout = backend.plan_replay_upload(
                        &command.recipe,
                        replay,
                        &self.stores,
                        &owners,
                    )?;
                    let replay_allocations = layout
                        .allocations()
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(ordinal, layout)| PreparedAllocationClaim {
                            command: command.node,
                            ordinal,
                            store,
                            layout,
                            slot: None,
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    let replay_streams = layout
                        .streams()
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(ordinal, layout)| PreparedStreamClaim {
                            command: command.node,
                            ordinal,
                            layout,
                            slot: None,
                        })
                        .collect::<Vec<_>>()
                        .into_boxed_slice();
                    Ok(PreparedResolvedReplayUpload {
                        recipe: replay.clone(),
                        layout,
                        allocations: replay_allocations,
                        streams: replay_streams,
                    })
                })
                .transpose()?;
            commands.push(PreparedResolvedCommand {
                command: command.clone(),
                native,
                accumulate,
                preimage,
                trapdoor,
                allocations,
                composite_allocations,
                composite_streams,
                streams,
                replay_upload,
            });
        }
        let mut schedules = Vec::with_capacity(self.schedules.len());
        for schedule in &self.schedules {
            let members = schedule
                .command_group
                .iter()
                .filter_map(|node| {
                    commands.iter().find(|command| {
                        command.command.node == *node &&
                            command
                                .command
                                .recipe
                                .owners
                                .iter()
                                .any(|owner| schedule.recipe.owners.contains(owner))
                    })
                })
                .filter_map(|command| command.native.clone())
                .collect::<Vec<_>>();
            let native = backend.plan_schedule(&schedule.recipe, &members)?;
            let streams = native
                .streams()
                .iter()
                .copied()
                .enumerate()
                .map(|(ordinal, layout)| PreparedStreamClaim {
                    command: schedule.command_group.first().copied().unwrap_or_default(),
                    ordinal,
                    layout,
                    slot: None,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            let allocations = native
                .allocations()
                .iter()
                .copied()
                .enumerate()
                .map(|(ordinal, layout)| PreparedAllocationClaim {
                    command: schedule.command_group.first().copied().unwrap_or_default(),
                    ordinal,
                    store: schedule.recipe.inputs.first().copied(),
                    layout,
                    slot: None,
                })
                .collect::<Vec<_>>()
                .into_boxed_slice();
            schedules.push(PreparedResolvedSchedule {
                schedule: schedule.clone(),
                native,
                allocations,
                streams,
            });
        }
        Ok(PreparedResolvedResources {
            owners: owners.into_boxed_slice(),
            scalar_buffers: scalar_buffers.into_boxed_slice(),
            commands: commands.into_boxed_slice(),
            schedules: schedules.into_boxed_slice(),
        })
    }
}

fn prepared_stage_role(operation: &PreparedOperation) -> PreparedStageRole {
    match operation {
        PreparedOperation::Gpu(operation) => match operation {
            PreparedGpuOperation::MatrixBinary(_) |
            PreparedGpuOperation::MatrixMulAccumulate |
            PreparedGpuOperation::MatrixMulSmallRhs |
            PreparedGpuOperation::MatrixNegate |
            PreparedGpuOperation::MatrixScale |
            PreparedGpuOperation::RingAutomorphism |
            PreparedGpuOperation::ModulusSwitch |
            PreparedGpuOperation::ModulusReduce |
            PreparedGpuOperation::CenteredRebase |
            PreparedGpuOperation::RnsModUp |
            PreparedGpuOperation::RnsModDown |
            PreparedGpuOperation::CenteredExtend |
            PreparedGpuOperation::BlockModSwitch |
            PreparedGpuOperation::Tensor |
            PreparedGpuOperation::Transpose |
            PreparedGpuOperation::ConcatRows |
            PreparedGpuOperation::FixedCopies |
            PreparedGpuOperation::GadgetDecompose |
            PreparedGpuOperation::ExtractCoefficient |
            PreparedGpuOperation::LiftIntegerToConstantPolynomial |
            PreparedGpuOperation::ThresholdDecode |
            PreparedGpuOperation::CrtRecompose |
            PreparedGpuOperation::PackPolynomialCoefficients => PreparedStageRole::Matrix,
            PreparedGpuOperation::UniformResidueSample |
            PreparedGpuOperation::UniformIntervalSample |
            PreparedGpuOperation::GaussianSample |
            PreparedGpuOperation::HashSample |
            PreparedGpuOperation::HashCompactDecompose |
            PreparedGpuOperation::TrapdoorSample |
            PreparedGpuOperation::PreimageSample => PreparedStageRole::Sampling,
            PreparedGpuOperation::RnsUpload | PreparedGpuOperation::RnsReadback => {
                PreparedStageRole::Transfer
            }
        },
        PreparedOperation::Scalar => PreparedStageRole::Scalar,
        PreparedOperation::Alias => PreparedStageRole::View,
        PreparedOperation::Selection => PreparedStageRole::Selection,
        PreparedOperation::ParallelLoop | PreparedOperation::SequentialLoop => {
            PreparedStageRole::Control
        }
        PreparedOperation::Warmup => PreparedStageRole::Control,
    }
}

fn prepared_native_stage(operation: &PreparedOperation) -> PreparedNativeStage {
    match operation {
        PreparedOperation::Gpu(operation) => match operation {
            PreparedGpuOperation::RnsUpload | PreparedGpuOperation::RnsReadback => {
                PreparedNativeStage::Transfer(*operation)
            }
            PreparedGpuOperation::UniformResidueSample |
            PreparedGpuOperation::UniformIntervalSample |
            PreparedGpuOperation::GaussianSample |
            PreparedGpuOperation::HashSample |
            PreparedGpuOperation::HashCompactDecompose |
            PreparedGpuOperation::TrapdoorSample |
            PreparedGpuOperation::PreimageSample => PreparedNativeStage::Sampling(*operation),
            PreparedGpuOperation::ThresholdDecode => PreparedNativeStage::Threshold,
            PreparedGpuOperation::PackPolynomialCoefficients => PreparedNativeStage::ScalarPack,
            PreparedGpuOperation::MatrixBinary(_) |
            PreparedGpuOperation::Transpose |
            PreparedGpuOperation::ConcatRows |
            PreparedGpuOperation::FixedCopies |
            PreparedGpuOperation::MatrixMulAccumulate |
            PreparedGpuOperation::MatrixMulSmallRhs |
            PreparedGpuOperation::MatrixNegate |
            PreparedGpuOperation::MatrixScale |
            PreparedGpuOperation::RingAutomorphism |
            PreparedGpuOperation::ModulusSwitch |
            PreparedGpuOperation::ModulusReduce |
            PreparedGpuOperation::CenteredRebase |
            PreparedGpuOperation::RnsModUp |
            PreparedGpuOperation::RnsModDown |
            PreparedGpuOperation::CenteredExtend |
            PreparedGpuOperation::BlockModSwitch |
            PreparedGpuOperation::Tensor |
            PreparedGpuOperation::GadgetDecompose |
            PreparedGpuOperation::ExtractCoefficient |
            PreparedGpuOperation::LiftIntegerToConstantPolynomial |
            PreparedGpuOperation::CrtRecompose => PreparedNativeStage::Matrix(*operation),
        },
        PreparedOperation::Scalar => PreparedNativeStage::ScalarOp,
        // Selection is refined to ScalarMatrixSelect in `from_preparation`
        // once its fixed candidate table is available. Matrix selections keep
        // the ordinary selection stage and are handled by the matrix binder.
        PreparedOperation::Selection => PreparedNativeStage::Selection,
        PreparedOperation::Alias => PreparedNativeStage::View,
        PreparedOperation::ParallelLoop | PreparedOperation::SequentialLoop => {
            PreparedNativeStage::Control
        }
        PreparedOperation::Warmup => PreparedNativeStage::Control,
    }
}

fn prepared_replay_upload_recipe(
    program: &GpuPreparation,
    operation: &PreparedOperation,
    outputs: &[PreparedDescriptorRef],
) -> Option<Result<PreparedReplayUploadRecipe, PreparedLoweringError>> {
    let output = outputs.first()?;
    let location = program.values.get(&output.wire)?;
    let wire_type = program.wire_types.get(&output.wire)?;
    match operation {
        PreparedOperation::Gpu(
            PreparedGpuOperation::UniformResidueSample |
            PreparedGpuOperation::UniformIntervalSample |
            PreparedGpuOperation::GaussianSample |
            PreparedGpuOperation::TrapdoorSample,
        ) => {
            let matrix = wire_type.matrix_type()?;
            let coefficient_count = matrix
                .rows
                .checked_mul(matrix.columns)
                .and_then(|count| count.checked_mul(matrix.ring_dimension))
                .ok_or(PreparedLoweringError::InvalidRange);
            let bits = matrix.modulus.bits() as usize;
            let payload_capacity = coefficient_count.and_then(|count| {
                count
                    .checked_mul(bits)
                    .map(|bits| bits.div_ceil(8))
                    .ok_or(PreparedLoweringError::InvalidRange)
            });
            Some(payload_capacity.map(|payload_capacity| PreparedReplayUploadRecipe::Matrix {
                rows: matrix.rows,
                columns: matrix.columns,
                level: location.level,
                format: match location.format {
                    PreparedFormat::Coefficient => 0,
                    PreparedFormat::Evaluation => 1,
                },
                codec_capacity: payload_capacity.saturating_add(128),
                payload_capacity,
            }))
        }
        PreparedOperation::Gpu(PreparedGpuOperation::PreimageSample) => {
            let (matrix, bound) = match wire_type {
                ConcreteWireType::Preimage { matrix, max_coefficient_bound } |
                ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound } => {
                    (matrix, max_coefficient_bound)
                }
                _ => {
                    return Some(Err(PreparedLoweringError::InvalidContract(
                        "preimage replay output is not a bounded matrix",
                    )))
                }
            };
            let magnitude_bytes = bound.to_bytes_le().1.len().max(1);
            let payload_bytes = matrix
                .rows
                .checked_mul(matrix.columns)
                .and_then(|count| count.checked_mul(matrix.ring_dimension))
                .and_then(|count| count.checked_mul(1usize.checked_add(magnitude_bytes)?))
                .ok_or(PreparedLoweringError::InvalidRange);
            Some(payload_bytes.map(|payload_bytes| PreparedReplayUploadRecipe::Small {
                rows: matrix.rows,
                columns: matrix.columns,
                level: location.level,
                codec_capacity: payload_bytes.saturating_add(49).saturating_add(magnitude_bytes),
                payload_bytes,
            }))
        }
        _ => None,
    }
}

fn prepared_recipe(
    node: u32,
    source: Option<&PreparedNodeSource>,
    operation: &PreparedOperation,
    inputs: &[PreparedDescriptorRef],
    outputs: &[PreparedDescriptorRef],
    stream: Option<&PreparedStreamKey>,
    completion: Option<u32>,
) -> PreparedNativeRecipe {
    let mut input_stores =
        inputs.iter().filter_map(|descriptor| descriptor.store).collect::<Vec<_>>();
    let mut output_stores =
        outputs.iter().filter_map(|descriptor| descriptor.store).collect::<Vec<_>>();
    if !matches!(operation, PreparedOperation::Gpu(PreparedGpuOperation::MatrixMulAccumulate)) {
        input_stores.sort_unstable();
        input_stores.dedup();
    }
    output_stores.sort_unstable();
    output_stores.dedup();
    let mut owners = stream
        .into_iter()
        .flat_map(|stream| stream.placements.iter())
        .map(|placement| PreparedOwnerKey {
            owner: placement.owner,
            device: placement.device,
            instance: stream.map_or(0, |stream| stream.instance),
            level: placement.level,
            format: placement.format,
        })
        .collect::<Vec<_>>();
    owners.sort_unstable();
    owners.dedup();
    PreparedNativeRecipe {
        node,
        source: source.cloned(),
        stage: prepared_native_stage(operation),
        inputs: input_stores.into_boxed_slice(),
        outputs: output_stores.into_boxed_slice(),
        owners: owners.into_boxed_slice(),
        completion,
        matrix_staging: None,
        scalar: match operation {
            PreparedOperation::Scalar => Some(PreparedScalarResource::Op {
                left_words: 1,
                right_words: 1,
                output_words: 1,
                candidate_count: 0,
            }),
            PreparedOperation::Gpu(PreparedGpuOperation::ThresholdDecode) => {
                Some(PreparedScalarResource::Threshold { count: 1, plaintext_words: 1 })
            }
            PreparedOperation::Gpu(PreparedGpuOperation::PackPolynomialCoefficients) => {
                Some(PreparedScalarResource::Pack {
                    count: 1,
                    coefficient_bits: 0,
                    output_format: 1,
                })
            }
            _ => None,
        },
        replay_upload: None,
    }
}

impl GpuPreparation {
    pub fn resource_plan(&self) -> &PreparedResourcePlan {
        &self.resource_plan
    }
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
) -> Result<GpuPreparation, PreparedLoweringError> {
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
    program.resource_plan = PreparedResourcePlan::from_preparation(&program)?;
    program.resource_plan.validate_for_warmup()?;
    Ok(program)
}

pub fn family_leaf_wires(program: &GpuPreparation, wire: WireRef) -> Vec<WireRef> {
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
        types::{ConcreteMatrixType, MatrixType, WireType},
    };
    use mxx_primitives::{
        matrix::gpu_dcrt_poly::{
            GpuPreparedSlotKind, GpuPreparedWorkspaceLayout, GpuTracedClaim, PreparedResourceKey,
        },
        sampler::trapdoor::gpu::{PreimageStage, TrapdoorStage},
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

    fn threshold_program() -> GpuPreparation {
        let source = WireRef { node: NodeId(60), port: Port(0) };
        let output = WireRef { node: NodeId(61), port: Port(0) };
        let node_id = 62;
        let matrix = ConcreteMatrixType::scalar(BigInt::from(17), 8);
        let mut program = GpuPreparation::default();
        program.values.insert(
            source,
            ValueLocation {
                owner: 70,
                rows: 0..1,
                columns: 0..1,
                level: 0,
                format: PreparedFormat::Evaluation,
                device: 3,
            },
        );
        program.wire_types.insert(source, ConcreteWireType::Matrix(matrix));
        program.wire_types.insert(output, ConcreteWireType::Bool);
        program
            .node_bindings
            .insert(node_id, (vec![source].into_boxed_slice(), vec![output].into_boxed_slice()));
        program.node_sources.insert(
            node_id,
            PreparedNodeSource {
                kind: NodeKind::ThresholdDecode {
                    plaintext_modulus: IntExpr::constant(17),
                    length: IntExpr::constant(8),
                    output_bool: true,
                },
                environment: ParamEnv::default(),
                variants: Box::new([]),
                variant_indices: Box::new([]),
                variant_input_types: Box::new([]),
                variant_output_types: Box::new([]),
            },
        );
        program.topology.nodes = vec![PreparedTopologyNode {
            id: node_id,
            command: PreparedCommandRequirement {
                operation: PreparedOperation::Gpu(PreparedGpuOperation::ThresholdDecode),
                inputs: 1,
                outputs: 1,
            },
            stream: 0,
            waits: Box::new([]),
            completion: 63,
        }]
        .into_boxed_slice();
        program
    }

    fn composite_stream_claim(
        command: u32,
        ordinal: usize,
        device: i32,
        partition: i32,
        role: i32,
    ) -> (PreparedCompositeClaim, PreparedCompositeStream) {
        let key = PreparedResourceKey {
            execution_owner_identity: 99,
            partition,
            device,
            limb_x: ordinal as u32,
            limb_y: (ordinal + 1) as u32,
            role,
        };
        let layout = PreparedAllocationLayout {
            key,
            kind: GpuPreparedSlotKind::SubmissionStream as i32,
            rows: 0,
            columns: 0,
            bytes: 0,
            alignment: 1,
            level: -1,
            format: -1,
        };
        (
            PreparedCompositeClaim {
                command,
                ordinal,
                claim: GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::SubmissionStream,
                    bytes: 0,
                    alignment: 1,
                }),
                layout: Some(layout),
                slot: None,
            },
            PreparedCompositeStream {
                command,
                ordinal,
                layout: PreparedStreamFootprint { key, origin: 1, pool_slot: ordinal },
                slot: None,
            },
        )
    }

    #[test]
    fn composite_bind_tapes_and_stream_claims_are_closed_without_a_device() {
        let preimage = GpuPreparedPreimageLayout::bind_entries();
        assert_eq!(preimage.len(), 42);
        assert_eq!(
            preimage[..16].iter().map(|entry| entry.owner).collect::<Vec<_>>(),
            (0..16).map(Some).collect::<Vec<_>>()
        );
        assert_eq!(
            preimage[16..40].iter().map(|entry| entry.stage).collect::<Vec<_>>(),
            PreimageStage::ALL.iter().map(|stage| Some(stage.index())).collect::<Vec<_>>()
        );
        assert_eq!(
            preimage[16..40].iter().map(|entry| entry.kind).collect::<Vec<_>>(),
            PreimageStage::ALL.iter().map(|stage| stage.kind()).collect::<Vec<_>>()
        );
        assert_eq!(
            mxx_primitives::sampler::trapdoor::gpu::PREIMAGE_STAGE_KINDS.to_vec(),
            PreimageStage::ALL.iter().map(|stage| stage.kind()).collect::<Vec<_>>()
        );
        assert_eq!(
            preimage[40].kind,
            mxx_primitives::sampler::trapdoor::gpu::GpuPreparedPreimageEntryKind::Phase
        );
        assert_eq!(
            preimage[41].kind,
            mxx_primitives::sampler::trapdoor::gpu::GpuPreparedPreimageEntryKind::Cutoff
        );

        let trapdoor = GpuPreparedTrapdoorLayout::bind_entries();
        assert_eq!(trapdoor.len(), 30);
        assert_eq!(
            trapdoor[..13].iter().map(|entry| entry.owner).collect::<Vec<_>>(),
            (0..13).map(Some).collect::<Vec<_>>()
        );
        assert_eq!(
            trapdoor[13..].iter().map(|entry| entry.stage).collect::<Vec<_>>(),
            TrapdoorStage::ALL.into_iter().map(Some).collect::<Vec<_>>()
        );
        assert_eq!(
            trapdoor[13..].iter().map(|entry| entry.kind).collect::<Vec<_>>(),
            TrapdoorStage::ALL.into_iter().map(|stage| stage.kind()).collect::<Vec<_>>()
        );
        assert_eq!(
            mxx_primitives::sampler::trapdoor::gpu::TRAPDOOR_STAGE_KINDS.to_vec(),
            TrapdoorStage::ALL.into_iter().map(|stage| stage.kind()).collect::<Vec<_>>()
        );
        let stage_entries = &trapdoor[13..];
        assert_eq!(stage_entries.len(), TrapdoorStage::ALL.len());
        assert!(
            stage_entries.iter().map(|entry| entry.stage).collect::<BTreeSet<_>>().len() ==
                TrapdoorStage::ALL.len()
        );

        // Keep descriptor, claim, and stream ordinals one-to-one. This uses
        // only structural values, so it catches a reordered or duplicated
        // stage without requiring a CUDA device or live allocation.
        let mut stage_claims = Vec::with_capacity(TrapdoorStage::ALL.len());
        let mut stage_streams = Vec::with_capacity(TrapdoorStage::ALL.len());
        for (ordinal, stage) in TrapdoorStage::ALL.into_iter().enumerate() {
            assert_eq!(stage_entries[ordinal].stage, Some(stage));
            assert_eq!(stage_entries[ordinal].kind, stage.kind());
            let (claim, stream) = composite_stream_claim(
                8,
                ordinal,
                3 + (ordinal as i32 % 2),
                ordinal as i32 % 3,
                stage.index() as i32,
            );
            stage_claims.push(claim);
            stage_streams.push(stream);
        }
        validate_composite_stream_claims(&stage_claims, &stage_streams).unwrap();

        // Interleave a non-stream claim and use two devices/partitions. The
        // validator must consume each exact physical stream key once.
        let (first_claim, first_stream) = composite_stream_claim(7, 0, 3, 0, 41);
        let (second_claim, second_stream) = composite_stream_claim(7, 1, 8, 2, 42);
        let allocations = [
            PreparedCompositeClaim {
                command: 7,
                ordinal: 0,
                claim: GpuTracedClaim::matrix(1, 1, 0, true),
                layout: None,
                slot: None,
            },
            first_claim,
            PreparedCompositeClaim {
                command: 7,
                ordinal: 2,
                claim: GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                }),
                layout: None,
                slot: None,
            },
            second_claim,
        ];
        let streams = [second_stream, first_stream];
        validate_composite_stream_claims(&allocations, &streams).unwrap();

        let (_, mut reused_stream) = composite_stream_claim(7, 3, 3, 0, 43);
        reused_stream.layout.origin = 0;
        validate_composite_stream_claims(
            &allocations,
            &[streams[0].clone(), streams[1].clone(), reused_stream],
        )
        .unwrap();

        let mut duplicate = streams;
        duplicate[1].layout.key = duplicate[0].layout.key;
        assert!(validate_composite_stream_claims(&allocations, &duplicate).is_err());
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
            assert!(
                matches!(
                    prepared_operation(&kind),
                    PreparedOperation::Warmup |
                        PreparedOperation::Scalar |
                        PreparedOperation::Alias |
                        PreparedOperation::Selection |
                        PreparedOperation::ParallelLoop |
                        PreparedOperation::SequentialLoop |
                        PreparedOperation::Gpu(_)
                ),
                "NodeKind has no explicit prepared replay category: {expected}"
            );
        }
    }

    #[test]
    fn structural_variants_select_one_explicit_replay_category() {
        let one = || IntExpr::constant(1);
        let real_one = || RealExpr::from_integer(1);
        let matrix = || matrix_type();
        let cases = [
            (
                "row concat",
                NodeKind::Concat { axis: ConcatAxis::Rows },
                PreparedOperation::Gpu(PreparedGpuOperation::ConcatRows),
            ),
            (
                "diagonal concat",
                NodeKind::Concat { axis: ConcatAxis::Diagonal },
                PreparedOperation::Alias,
            ),
            (
                "slice alias",
                NodeKind::Slice { rows: None, columns: None },
                PreparedOperation::Alias,
            ),
            (
                "plain hash",
                NodeKind::HashSample {
                    matrix_type: matrix(),
                    variant: HashVariant::Plain,
                    tag_prefix: vec![],
                    tag_components: vec![],
                    base: None,
                    digit_count: None,
                },
                PreparedOperation::Gpu(PreparedGpuOperation::HashSample),
            ),
            (
                "decomposed hash",
                NodeKind::HashSample {
                    matrix_type: matrix(),
                    variant: HashVariant::Decomposed,
                    tag_prefix: vec![],
                    tag_components: vec![],
                    base: Some(one()),
                    digit_count: Some(one()),
                },
                PreparedOperation::Gpu(PreparedGpuOperation::HashCompactDecompose),
            ),
            (
                "small decomposed hash",
                NodeKind::HashSample {
                    matrix_type: matrix(),
                    variant: HashVariant::SmallDecomposed,
                    tag_prefix: vec![],
                    tag_components: vec![],
                    base: Some(one()),
                    digit_count: Some(one()),
                },
                PreparedOperation::Gpu(PreparedGpuOperation::HashCompactDecompose),
            ),
            (
                "parallel loop",
                NodeKind::ParallelLoop(ParallelLoop {
                    count: one(),
                    minimum_count: 0,
                    index_slot: 0,
                    bindings: vec![],
                    input_modes: vec![],
                }),
                PreparedOperation::ParallelLoop,
            ),
            (
                "sequential loop",
                NodeKind::SequentialLoop(SequentialLoop {
                    count: one(),
                    index_slot: 0,
                    bindings: vec![],
                    carried_count: 1,
                }),
                PreparedOperation::SequentialLoop,
            ),
            (
                "subgraph call",
                NodeKind::SubgraphCall(SubgraphCall {
                    definition: "body".into(),
                    bindings: vec![],
                    canonical_input_exclusive_uppers: vec![],
                }),
                PreparedOperation::ParallelLoop,
            ),
            ("family pack", NodeKind::FamilyPack { count: one() }, PreparedOperation::Selection),
            (
                "trapdoor sample",
                NodeKind::TrapdoorSample {
                    matrix_type: matrix(),
                    sigma: real_one(),
                    gadget_base: one(),
                    digit_count: one(),
                    preimage_max_coefficient_bound: one(),
                },
                PreparedOperation::Gpu(PreparedGpuOperation::TrapdoorSample),
            ),
            (
                "preimage sample",
                NodeKind::PreimageSample { matrix_type: matrix(), max_coefficient_bound: one() },
                PreparedOperation::Gpu(PreparedGpuOperation::PreimageSample),
            ),
        ];
        for (name, kind, expected) in cases {
            assert_eq!(prepared_operation(&kind), expected, "wrong prepared category for {name}");
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

    #[test]
    fn resource_plan_uses_fixed_descriptor_indices_without_native_work() {
        let input = WireRef { node: NodeId(1), port: Port(0) };
        let output = WireRef { node: NodeId(2), port: Port(0) };
        let mut program = GpuPreparation::default();
        program.values.insert(
            input,
            ValueLocation {
                owner: 7,
                rows: 0..2,
                columns: 0..3,
                level: 1,
                format: PreparedFormat::Coefficient,
                device: 2,
            },
        );
        program.values.insert(
            output,
            ValueLocation {
                owner: 8,
                rows: 0..2,
                columns: 0..3,
                level: 1,
                format: PreparedFormat::Evaluation,
                device: 2,
            },
        );
        program
            .node_bindings
            .insert(9, (vec![input].into_boxed_slice(), vec![output].into_boxed_slice()));
        program.topology.nodes = vec![PreparedTopologyNode {
            id: 9,
            command: PreparedCommandRequirement {
                operation: PreparedOperation::Gpu(PreparedGpuOperation::MatrixBinary(
                    MatrixBinaryOp::Add,
                )),
                inputs: 1,
                outputs: 1,
            },
            stream: 3,
            waits: Box::new([]),
            completion: 90,
        }]
        .into_boxed_slice();

        let plan = PreparedResourcePlan::from_preparation(&program).unwrap();
        assert_eq!(plan.stores.len(), 2);
        assert_eq!(plan.commands[0].inputs[0].store, Some(0));
        assert_eq!(plan.commands[0].outputs[0].store, Some(1));
        assert_eq!(plan.commands[0].inputs[0].kind, PreparedDescriptorKind::Matrix);
        assert_eq!(plan.commands[0].stream.as_ref().unwrap().stores.as_ref(), [0, 1]);
        assert_eq!(plan.schedules[0].command_group.as_ref(), [9]);
        assert_eq!(plan.schedules[0].completions.as_ref(), [90]);
        plan.validate_for_warmup().unwrap();
    }

    #[test]
    fn resource_plan_flattens_families_and_allows_host_outputs() {
        let root = WireRef { node: NodeId(1), port: Port(0) };
        let leaf_a = WireRef { node: NodeId(u64::MAX), port: Port(0) };
        let leaf_b = WireRef { node: NodeId(u64::MAX - 1), port: Port(0) };
        let output = WireRef { node: NodeId(2), port: Port(0) };
        let mut program = GpuPreparation::default();
        for (wire, owner) in [(leaf_a, 10), (leaf_b, 11)] {
            program.values.insert(
                wire,
                ValueLocation {
                    owner,
                    rows: 0..1,
                    columns: 0..2,
                    level: 0,
                    format: PreparedFormat::Evaluation,
                    device: 3,
                },
            );
        }
        program.family_wires.insert(root, vec![leaf_a, leaf_b].into_boxed_slice());
        program
            .node_bindings
            .insert(2, (vec![root].into_boxed_slice(), vec![output].into_boxed_slice()));
        program.topology.nodes = vec![PreparedTopologyNode {
            id: 2,
            command: PreparedCommandRequirement {
                operation: PreparedOperation::Gpu(PreparedGpuOperation::RnsReadback),
                inputs: 1,
                outputs: 1,
            },
            stream: 0,
            waits: Box::new([]),
            completion: 12,
        }]
        .into_boxed_slice();
        let plan = PreparedResourcePlan::from_preparation(&program).unwrap();
        assert_eq!(plan.commands[0].inputs.len(), 2);
        assert!(plan.commands[0].inputs.iter().all(|descriptor| {
            descriptor.kind == PreparedDescriptorKind::Matrix && descriptor.store.is_some()
        }));
        assert_eq!(plan.commands[0].outputs[0].kind, PreparedDescriptorKind::Host);
        plan.validate_for_warmup().unwrap();
    }

    #[test]
    fn resource_plan_preserves_reused_capacity_and_slot_multiplicity() {
        let wire = WireRef { node: NodeId(4), port: Port(0) };
        let mut program = GpuPreparation { instance_count: 2, ..GpuPreparation::default() };
        program.values.insert(
            wire,
            ValueLocation {
                owner: 7,
                rows: 0..2,
                columns: 0..3,
                level: 1,
                format: PreparedFormat::Evaluation,
                device: 4,
            },
        );
        program.storage_capacities.insert(7, (9, 11));
        program.node_bindings.insert(4, (Box::new([]), vec![wire].into_boxed_slice()));
        program.topology.nodes = vec![PreparedTopologyNode {
            id: 4,
            command: PreparedCommandRequirement {
                operation: PreparedOperation::Gpu(PreparedGpuOperation::MatrixNegate),
                inputs: 0,
                outputs: 1,
            },
            stream: 0,
            waits: Box::new([]),
            completion: 5,
        }]
        .into_boxed_slice();
        let plan = PreparedResourcePlan::from_preparation(&program).unwrap();
        assert_eq!(plan.instance_count, 2);
        assert_eq!(plan.stores.len(), 2);
        assert_eq!(plan.stores[0].capacity_rows, 9);
        assert_eq!(plan.stores[0].capacity_columns, 11);
        assert_eq!(plan.stores[0].location.rows, 0..9);
        assert_eq!(plan.stores[1].instance, 1);
        assert_eq!(plan.commands.len(), 2);
        assert_eq!(plan.commands[0].stream.as_ref().unwrap().stores.as_ref(), [0]);
        assert_eq!(plan.commands[1].stream.as_ref().unwrap().stores.as_ref(), [1]);
    }

    #[test]
    fn threshold_resource_plan_reserves_a_distinct_coefficient_staging_owner() {
        let plan = PreparedResourcePlan::from_preparation(&threshold_program()).unwrap();
        assert_eq!(plan.stores.len(), 2);
        let staging = plan
            .stores
            .iter()
            .find(|store| store.location.owner != 70)
            .expect("threshold staging store");
        assert_eq!(staging.location.rows, 0..1);
        assert_eq!(staging.location.columns, 0..1);
        assert_eq!(staging.location.format, PreparedFormat::Coefficient);
        assert_eq!(staging.location.device, 3);
        let command = &plan.commands[0];
        let staging_owner = command.recipe.matrix_staging.expect("threshold staging owner");
        assert_eq!(staging_owner.owner, staging.location.owner);
        assert_eq!(staging_owner.format, PreparedFormat::Coefficient);
        assert!(command.recipe.owners.contains(&staging_owner));
        assert_eq!(command.stream.as_ref().unwrap().stores.len(), 2);
        plan.validate_for_warmup().unwrap();
    }

    #[test]
    fn threshold_resource_plan_expands_staging_per_physical_device() {
        let plan =
            PreparedResourcePlan::from_preparation_for_devices(&threshold_program(), &[2, 7])
                .unwrap();
        let staging_stores = plan
            .stores
            .iter()
            .enumerate()
            .filter(|(_, store)| store.location.format == PreparedFormat::Coefficient)
            .collect::<Vec<_>>();
        assert_eq!(staging_stores.len(), 2);
        assert_eq!(
            staging_stores.iter().map(|(_, store)| store.location.device).collect::<Vec<_>>(),
            [2, 7]
        );

        for device in [2, 7] {
            let [(staging_index, staging)] = staging_stores
                .iter()
                .filter(|(_, store)| store.location.device == device)
                .copied()
                .collect::<Vec<_>>()[..]
            else {
                panic!("expected exactly one threshold staging store on device {device}");
            };
            let [command] = plan
                .commands
                .iter()
                .filter(|command| {
                    command.recipe.matrix_staging.is_some_and(|owner| owner.device == device)
                })
                .collect::<Vec<_>>()[..]
            else {
                panic!("expected exactly one threshold command on device {device}");
            };
            let staging_owner = command.recipe.matrix_staging.unwrap();
            assert_eq!(staging_owner.owner, staging.location.owner);
            assert_eq!(staging_owner.device, device);
            assert_eq!(
                command.recipe.owners.iter().filter(|owner| **owner == staging_owner).count(),
                1
            );
            let stream = command.stream.as_ref().unwrap();
            assert!(stream.stores.contains(&staging_index));
            assert_eq!(
                stream
                    .placements
                    .iter()
                    .filter(|placement| placement.store == staging_index)
                    .collect::<Vec<_>>(),
                [&PreparedPlacementKey {
                    store: staging_index,
                    owner: staging.location.owner,
                    device,
                    level: staging.location.level,
                    format: PreparedFormat::Coefficient,
                }]
            );
        }
        plan.validate_for_warmup().unwrap();
    }

    #[test]
    fn threshold_validation_rejects_missing_staging_contract() {
        let valid = PreparedResourcePlan::from_preparation(&threshold_program()).unwrap();

        let mut missing_owner = valid.clone();
        missing_owner.commands[0].recipe.matrix_staging = None;
        assert!(missing_owner.validate_for_warmup().is_err());

        let staging_index = valid.commands[0]
            .stream
            .as_ref()
            .unwrap()
            .stores
            .iter()
            .copied()
            .find(|index| valid.stores[*index].location.format == PreparedFormat::Coefficient)
            .unwrap();
        let mut missing_placement = valid;
        let stream = missing_placement.commands[0].stream.as_mut().unwrap();
        stream.stores =
            stream.stores.iter().copied().filter(|index| *index != staging_index).collect();
        stream.placements = stream
            .placements
            .iter()
            .copied()
            .filter(|placement| placement.store != staging_index)
            .collect();
        assert!(missing_placement.validate_for_warmup().is_err());
    }

    #[test]
    fn resource_plan_expands_owner_and_stream_claims_per_physical_device() {
        let wire = WireRef { node: NodeId(40), port: Port(0) };
        let mut program = GpuPreparation::default();
        program.values.insert(
            wire,
            ValueLocation {
                owner: 70,
                rows: 0..2,
                columns: 0..3,
                level: 1,
                format: PreparedFormat::Evaluation,
                device: 0,
            },
        );
        program.storage_capacities.insert(70, (2, 3));
        program.node_bindings.insert(40, (Box::new([]), vec![wire].into_boxed_slice()));
        program.topology.nodes = vec![PreparedTopologyNode {
            id: 40,
            command: PreparedCommandRequirement {
                operation: PreparedOperation::Gpu(PreparedGpuOperation::MatrixNegate),
                inputs: 0,
                outputs: 1,
            },
            stream: 0,
            waits: Box::new([]),
            completion: 41,
        }]
        .into_boxed_slice();

        let plan = PreparedResourcePlan::from_preparation_for_devices(&program, &[2, 7]).unwrap();
        assert_eq!(plan.stores.len(), 2);
        assert_eq!(plan.commands.len(), 2);
        assert_eq!(
            plan.stores.iter().map(|store| store.location.device).collect::<Vec<_>>(),
            [2, 7]
        );
        assert_eq!(
            plan.commands
                .iter()
                .map(|command| command.stream.as_ref().unwrap().placements[0].device)
                .collect::<Vec<_>>(),
            [2, 7]
        );
        assert_ne!(
            plan.commands[0].stream.as_ref().unwrap().stores[0],
            plan.commands[1].stream.as_ref().unwrap().stores[0]
        );
        assert!(PreparedResourcePlan::from_preparation_for_devices(&program, &[2, 2]).is_err());
    }

    #[test]
    fn replay_mapping_selects_each_physical_command_deterministically() {
        let command = |device, instance| PreparedCommandPlan {
            node: 50,
            instance,
            operation: PreparedOperation::Gpu(PreparedGpuOperation::MatrixNegate),
            inputs: Box::new([]),
            outputs: Box::new([]),
            stream: None,
            waits: Box::new([]),
            completion: 51,
            recipe: PreparedNativeRecipe {
                node: 50,
                source: None,
                stage: PreparedNativeStage::Matrix(PreparedGpuOperation::MatrixNegate),
                inputs: Box::new([]),
                outputs: Box::new([]),
                owners: vec![PreparedOwnerKey {
                    owner: 80,
                    device,
                    instance,
                    level: 0,
                    format: PreparedFormat::Evaluation,
                }]
                .into_boxed_slice(),
                completion: Some(51),
                matrix_staging: None,
                scalar: None,
                replay_upload: None,
            },
        };
        let commands = [command(2, 0), command(7, 0), command(2, 1), command(7, 1)];
        assert!(physical_command_matches(&commands[0], 50, 0, 2));
        assert!(!physical_command_matches(&commands[0], 50, 0, 7));
        assert!(physical_command_matches(&commands[1], 50, 0, 7));
        assert!(!physical_command_matches(&commands[0], 50, 1, 2));
        assert!(physical_command_matches(&commands[2], 50, 1, 2));
        assert!(physical_command_matches(&commands[3], 50, 1, 7));
    }

    #[test]
    fn resource_plan_anchors_streams_on_all_matrix_descriptors() {
        let host = WireRef { node: NodeId(5), port: Port(0) };
        let matrix = WireRef { node: NodeId(6), port: Port(0) };
        let mut program = GpuPreparation::default();
        program.values.insert(
            matrix,
            ValueLocation {
                owner: 22,
                rows: 0..1,
                columns: 0..1,
                level: 0,
                format: PreparedFormat::Evaluation,
                device: 8,
            },
        );
        program
            .node_bindings
            .insert(6, (vec![host].into_boxed_slice(), vec![matrix].into_boxed_slice()));
        program.topology.nodes = vec![PreparedTopologyNode {
            id: 6,
            command: PreparedCommandRequirement {
                operation: PreparedOperation::Gpu(PreparedGpuOperation::RnsUpload),
                inputs: 1,
                outputs: 1,
            },
            stream: 0,
            waits: vec![3].into_boxed_slice(),
            completion: 4,
        }]
        .into_boxed_slice();
        let plan = PreparedResourcePlan::from_preparation(&program).unwrap();
        let stream = plan.commands[0].stream.as_ref().unwrap();
        assert_eq!(stream.stores.len(), 1);
        assert_eq!(stream.placements[0].device, 8);
        assert_eq!(plan.schedules[0].waits.as_ref(), [3]);
        assert_eq!(plan.schedules[0].completions.as_ref(), [4]);
        plan.validate_for_warmup().unwrap();
    }

    #[test]
    fn resource_plan_shares_only_proven_alias_owner() {
        let first = WireRef { node: NodeId(7), port: Port(0) };
        let alias = WireRef { node: NodeId(8), port: Port(0) };
        let mut program = GpuPreparation::default();
        for (wire, rows, columns) in [(first, 0..2, 0..3), (alias, 0..1, 0..2)] {
            program.values.insert(
                wire,
                ValueLocation {
                    owner: 42,
                    rows,
                    columns,
                    level: 1,
                    format: PreparedFormat::Evaluation,
                    device: 5,
                },
            );
        }
        program
            .node_bindings
            .insert(9, (vec![first, alias].into_boxed_slice(), vec![alias].into_boxed_slice()));
        program.topology.nodes = vec![PreparedTopologyNode {
            id: 9,
            command: PreparedCommandRequirement {
                operation: PreparedOperation::Gpu(PreparedGpuOperation::MatrixBinary(
                    MatrixBinaryOp::Add,
                )),
                inputs: 2,
                outputs: 1,
            },
            stream: 0,
            waits: Box::new([]),
            completion: 10,
        }]
        .into_boxed_slice();
        let plan = PreparedResourcePlan::from_preparation(&program).unwrap();
        assert_eq!(plan.stores.len(), 1);
        assert_eq!(plan.commands[0].inputs[0].store, Some(0));
        assert_eq!(plan.commands[0].inputs[1].store, Some(0));
        assert_eq!(plan.commands[0].outputs[0].store, Some(0));
        assert_eq!(plan.stores[0].location.rows, 0..2);
        assert_eq!(plan.stores[0].location.columns, 0..3);
    }

    #[test]
    fn resource_plan_carries_closed_native_recipes_without_runtime_handles() {
        let source = include_str!("gpu_prepared_lowering.rs")
            .split("#[cfg(test)]")
            .next()
            .expect("lowering implementation before tests");
        assert!(source.contains("pub enum PreparedNativeStage"));
        assert!(source.contains("pub struct PreparedNativeRecipe"));
        assert!(source.contains("pub trait PreparedResourceBackend"));
        assert!(source.contains("pub fn resolve_prepared_resources<B: PreparedResourceBackend"));
        assert!(source.contains("stream_ordinal_base"));
        assert!(!source.contains("cudaStream_t"));
        assert!(!source.contains("cudaEvent_t"));
    }

    #[test]
    fn scalar_native_recipes_are_structural_and_width_explicit() {
        let resources = [
            PreparedScalarResource::Buffer { count: 1, words: 1, pinned_host_bytes: 16 },
            PreparedScalarResource::Op {
                left_words: 1,
                right_words: 1,
                output_words: 1,
                candidate_count: 3,
            },
            PreparedScalarResource::MatrixSelect { rows: 2, columns: 4, level: 1, count: 3 },
            PreparedScalarResource::Threshold { count: 8, plaintext_words: 2 },
            PreparedScalarResource::Pack { count: 32, coefficient_bits: 1, output_format: 1 },
        ];
        assert_eq!(
            resources[0],
            PreparedScalarResource::Buffer { count: 1, words: 1, pinned_host_bytes: 16 }
        );
        assert!(matches!(resources[1], PreparedScalarResource::Op { candidate_count: 3, .. }));
        assert!(matches!(resources[2], PreparedScalarResource::MatrixSelect { count: 3, .. }));
        assert!(matches!(
            resources[3],
            PreparedScalarResource::Threshold { plaintext_words: 2, .. }
        ));
        assert!(matches!(resources[4], PreparedScalarResource::Pack { coefficient_bits: 1, .. }));
    }
}
