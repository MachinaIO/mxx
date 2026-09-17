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
        GpuPreparedAccumulateLayout, GpuPreparedRequest, GpuPreparedStorage,
        PreparedAllocationLayout, PreparedOwnerLayout, PreparedPlanLayout, PreparedResourceKey,
        PreparedStreamFootprint,
    },
    sampler::trapdoor::gpu::{GpuPreparedPreimageLayout, GpuPreparedTrapdoorLayout},
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    sync::Arc,
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

/// The semantic site of a matrix value.  A site is deliberately tagged so
/// that ordinary wires, finite variant outputs, and host readback staging can
/// never be confused merely because their physical geometry happens to match.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum MatrixSite {
    Ordinary { wire: WireRef, device: i32, instance: usize },
    Variant { node: u32, variant: usize, port: usize, device: i32, instance: usize },
    HostStaging { node: u32, instance: usize, device: i32 },
}

/// Stable identity for one finalized matrix site.  `location` is the physical
/// owner/capacity view; the tagged `site` remains the semantic identity.  The
/// two are kept separate so aliasing and storage colouring cannot silently
/// turn one logical value into another.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct FinalizedMatrixId(pub u64);

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct BasisId(pub u64);

/// Fully resolved matrix view geometry.  The physical owner is not inferred
/// from this layout; it is only the immutable semantic view consumed by
/// descriptors and replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixViewLayout {
    pub rows: Range<usize>,
    pub columns: Range<usize>,
    pub row_stride: usize,
    pub column_stride: usize,
    pub ring_rows: usize,
    pub ring_columns: usize,
}

/// Canonical physical owner/capacity record.  It is stored once by the
/// finalized matrix table; resource plans refer to it through
/// `FinalizedMatrixId` instead of deciding owner identity from a tuple.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalizedMatrixPhysical {
    pub owner: u64,
    pub device: i32,
    pub level: usize,
    pub format: PreparedFormat,
    pub context_identity: usize,
    pub capacity_rows: usize,
    pub capacity_columns: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FinalizedMatrixIdentity {
    pub site: MatrixSite,
    pub physical: FinalizedMatrixPhysical,
    pub basis: BasisId,
    pub view: MatrixViewLayout,
    pub execution_instance: usize,
}

pub(crate) fn finalized_matrix_location(identity: &FinalizedMatrixIdentity) -> ValueLocation {
    ValueLocation {
        owner: identity.physical.owner,
        rows: identity.view.rows.clone(),
        columns: identity.view.columns.clone(),
        level: identity.physical.level,
        format: identity.physical.format,
        device: identity.physical.device,
    }
}

/// The sole finalized matrix identity authority.  All native descriptors and
/// stores are derived from this table; callers must not maintain parallel
/// ordinary/variant/staging state maps.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct FinalizedMatrixIdTable {
    pub by_site: BTreeMap<MatrixSite, FinalizedMatrixId>,
    pub identities: BTreeMap<FinalizedMatrixId, FinalizedMatrixIdentity>,
    next: u64,
}

pub fn store_owner_key(store: &PreparedStorePlan) -> PreparedOwnerKey {
    PreparedOwnerKey { matrix_id: store.matrix_id, instance: store.instance }
}

impl FinalizedMatrixIdTable {
    pub fn clear(&mut self) {
        self.by_site.clear();
        self.identities.clear();
        self.next = 0;
    }

    pub fn insert(
        &mut self,
        site: MatrixSite,
        location: ValueLocation,
        context_identity: usize,
        capacity_rows: usize,
        capacity_columns: usize,
    ) -> Result<FinalizedMatrixId, String> {
        let view_location = location.clone();
        let candidate = FinalizedMatrixIdentity {
            site,
            physical: FinalizedMatrixPhysical {
                owner: location.owner,
                device: location.device,
                level: location.level,
                format: location.format,
                context_identity,
                capacity_rows,
                capacity_columns,
            },
            // Context identity selects the parameter set; include the
            // finalized level/format so CRT basis content/order cannot alias
            // across views that happen to share a context.
            basis: BasisId(
                (context_identity as u64)
                    .wrapping_mul(1_000_003)
                    .wrapping_add(view_location.level as u64)
                    .wrapping_mul(31)
                    .wrapping_add(match view_location.format {
                        PreparedFormat::Coefficient => 0,
                        PreparedFormat::Evaluation => 1,
                    }),
            ),
            view: MatrixViewLayout {
                rows: view_location.rows.clone(),
                columns: view_location.columns.clone(),
                row_stride: view_location.columns.end,
                column_stride: 1,
                ring_rows: view_location.rows.end,
                ring_columns: view_location.columns.end,
            },
            execution_instance: match site {
                MatrixSite::Ordinary { instance, .. } |
                MatrixSite::Variant { instance, .. } |
                MatrixSite::HostStaging { instance, .. } => instance,
            },
        };
        if let Some(id) = self.by_site.get(&site).copied() {
            let existing = self
                .identities
                .get(&id)
                .ok_or_else(|| "finalized matrix site index is corrupt".to_owned())?;
            return if existing == &candidate {
                Ok(id)
            } else {
                Err(format!("conflicting finalized matrix identity for site {site:?}"))
            };
        }
        let id = {
            let id = FinalizedMatrixId(self.next);
            self.next = self.next.checked_add(1).expect("finalized matrix id overflow");
            id
        };
        self.by_site.insert(site, id);
        self.identities.insert(id, candidate);
        Ok(id)
    }

    pub fn id(&self, site: MatrixSite) -> Option<FinalizedMatrixId> {
        self.by_site.get(&site).copied()
    }

    pub fn identity(&self, id: FinalizedMatrixId) -> Option<&FinalizedMatrixIdentity> {
        self.identities.get(&id)
    }

    /// Resolve the same tagged semantic site for one concrete execution
    /// instance. Instance expansion is part of identity, not stream order.
    pub fn for_execution_instance(
        &self,
        id: FinalizedMatrixId,
        instance: usize,
    ) -> Option<FinalizedMatrixId> {
        let site = self.identity(id)?.site;
        let site = match site {
            MatrixSite::Ordinary { wire, device, .. } => {
                MatrixSite::Ordinary { wire, device, instance }
            }
            MatrixSite::Variant { node, variant, port, device, .. } => {
                MatrixSite::Variant { node, variant, port, device, instance }
            }
            MatrixSite::HostStaging { node, device, .. } => {
                MatrixSite::HostStaging { node, device, instance }
            }
        };
        self.id(site)
    }

    pub fn ordinary_for_instance(
        &self,
        wire: WireRef,
        device: i32,
        instance: usize,
    ) -> Option<&FinalizedMatrixIdentity> {
        self.id(MatrixSite::Ordinary { wire, device, instance }).and_then(|id| self.identity(id))
    }

    pub fn variant(
        &self,
        node: u32,
        variant: usize,
        port: usize,
        device: i32,
    ) -> Option<&FinalizedMatrixIdentity> {
        self.variant_for_instance(node, variant, port, device, 0)
    }

    pub fn variant_for_instance(
        &self,
        node: u32,
        variant: usize,
        port: usize,
        device: i32,
        instance: usize,
    ) -> Option<&FinalizedMatrixIdentity> {
        self.id(MatrixSite::Variant { node, variant, port, device, instance })
            .and_then(|id| self.identity(id))
    }

    pub fn host_staging(
        &self,
        node: u32,
        instance: usize,
        device: i32,
    ) -> Option<&FinalizedMatrixIdentity> {
        self.id(MatrixSite::HostStaging { node, instance, device }).and_then(|id| self.identity(id))
    }

    pub fn iter_variants(&self) -> impl Iterator<Item = (MatrixSite, &FinalizedMatrixIdentity)> {
        self.identities.iter().filter_map(|(_, identity)| {
            matches!(identity.site, MatrixSite::Variant { .. }).then_some((identity.site, identity))
        })
    }

    pub fn iter_host_staging(
        &self,
    ) -> impl Iterator<Item = (MatrixSite, &FinalizedMatrixIdentity)> {
        self.identities.iter().filter_map(|(_, identity)| {
            matches!(identity.site, MatrixSite::HostStaging { .. })
                .then_some((identity.site, identity))
        })
    }
}

fn exact_common_instance_identity<'a>(
    table: &'a FinalizedMatrixIdTable,
    wire: WireRef,
    device: i32,
    instance_count: usize,
) -> Option<&'a FinalizedMatrixIdentity> {
    let mut identities =
        (0..instance_count).map(|instance| table.ordinary_for_instance(wire, device, instance));
    let first = identities.next()??;
    identities
        .all(|identity| {
            identity.is_some_and(|candidate| {
                candidate.physical == first.physical && candidate.view == first.view
            })
        })
        .then_some(first)
}

/// Stable logical-to-physical binding identity. Storage is populated by the
/// append-only provisioning transaction after lowering selects the topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct PreparedBindingId {
    pub matrix_id: FinalizedMatrixId,
    pub instance: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedViewRange {
    pub rows: Range<usize>,
    pub columns: Range<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedCopy {
    pub source: PreparedViewRange,
    pub destination: PreparedViewRange,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PreparedViewShape {
    Alias(PreparedViewRange),
    TransposeAlias(PreparedViewRange),
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
) -> Result<PreparedViewShape, PreparedLoweringError> {
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
        return Ok(PreparedViewShape::Alias(PreparedViewRange {
            rows: source_rows,
            columns: source_columns,
        }));
    }
    Ok(PreparedViewShape::FixedCopies(
        vec![FixedCopy {
            source: PreparedViewRange {
                rows: selected_source.rows,
                columns: selected_source.columns,
            },
            destination: PreparedViewRange { rows: destination.rows, columns: destination.columns },
        }]
        .into_boxed_slice(),
    ))
}

pub fn lower_transpose(
    kind: &NodeKind,
    source: ValueLocation,
    destination: ValueLocation,
) -> Result<PreparedViewShape, PreparedLoweringError> {
    require_node(kind, "Transpose")?;
    let (rows, columns) = source.shape();
    if destination.shape() != (columns, rows) {
        return Err(PreparedLoweringError::InvalidRange);
    }
    let transposed_source = ValueLocation { rows: source.columns, columns: source.rows, ..source };
    if transposed_source.same_owner(&destination) {
        return Ok(PreparedViewShape::TransposeAlias(PreparedViewRange {
            rows: transposed_source.rows,
            columns: transposed_source.columns,
        }));
    }
    Ok(PreparedViewShape::FixedCopies(
        vec![FixedCopy {
            source: PreparedViewRange {
                rows: transposed_source.rows,
                columns: transposed_source.columns,
            },
            destination: PreparedViewRange { rows: destination.rows, columns: destination.columns },
        }]
        .into_boxed_slice(),
    ))
}

pub fn lower_concat(
    kind: &NodeKind,
    inputs: &[ValueLocation],
    destination: ValueLocation,
) -> Result<PreparedViewShape, PreparedLoweringError> {
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
        return Ok(PreparedViewShape::Alias(PreparedViewRange {
            rows: destination.rows,
            columns: destination.columns,
        }));
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
                source: PreparedViewRange {
                    rows: input.rows.clone(),
                    columns: input.columns.clone(),
                },
                destination: PreparedViewRange { rows: target_rows, columns: target_columns },
            }
        })
        .collect::<Vec<_>>()
        .into_boxed_slice();
    Ok(PreparedViewShape::FixedCopies(copies))
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
    pub matrix_id: Option<FinalizedMatrixId>,
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
    pub matrix_id: FinalizedMatrixId,
    pub wire: WireRef,
    pub wire_type: Option<ConcreteWireType>,
    pub instance: usize,
    /// Logical command shape remains distinct from the colored owner's maximum
    /// capacity. Native stage planners consume this exact shape; allocation
    /// claims consume the capacity fields above.
    pub logical_rows: usize,
    pub logical_columns: usize,
}

/// Resolve the one matrix store a native recipe owns. A recipe with several
/// outputs or several input-only stores is not safely bindable by ordinal;
/// callers must reject it before provisioning rather than silently selecting
/// the first entry.
pub(crate) fn exact_recipe_store(
    recipe: &PreparedNativeRecipe,
    stores: &[PreparedStorePlan],
) -> Result<Option<usize>, String> {
    let unique = |indices: &[usize]| -> Result<Option<usize>, String> {
        let mut candidates = indices.to_vec();
        if candidates.iter().any(|index| stores.get(*index).is_none()) {
            return Err(format!("prepared node {} has an invalid recipe store", recipe.node));
        }
        candidates.sort_unstable();
        candidates.dedup();
        match candidates.as_slice() {
            [] => Ok(None),
            [store] => Ok(Some(*store)),
            _ => Err(format!("prepared node {} has ambiguous recipe stores", recipe.node)),
        }
    };
    if let Some(store) = unique(&recipe.outputs)? {
        return Ok(Some(store));
    }
    if let Some(store) = unique(&recipe.inputs)? {
        return Ok(Some(store));
    }
    // Scalar-native recipes have no matrix descriptors, but their single
    // explicit owner is still exact provenance for auxiliary workspaces.
    // Resolve that owner only when it identifies one logical matrix store;
    // multiple owners remain an intentional ambiguity and are rejected.
    let owner_stores = recipe
        .owners
        .iter()
        .flat_map(|owner| {
            stores.iter().enumerate().filter_map(move |(index, store)| {
                (store.matrix_id == owner.matrix_id && store.instance == owner.instance)
                    .then_some(index)
            })
        })
        .collect::<Vec<_>>();
    unique(&owner_stores)
}

/// Resolve the physical store for one native allocation. Matrix plans can
/// contain a source-shaped scratch allocation in addition to their output;
/// the allocation geometry is authoritative for that claim, while auxiliary
/// workspaces retain the recipe's exact output/input provenance.
pub(crate) fn recipe_store_for_layout(
    recipe: &PreparedNativeRecipe,
    stores: &[PreparedStorePlan],
    layout: &PreparedAllocationLayout,
) -> Result<Option<usize>, String> {
    if layout.kind != 0 {
        return exact_recipe_store(recipe, stores);
    }
    if !matches!(layout.format, 0 | 1) {
        return Err(format!("prepared node {} matrix layout has invalid format", recipe.node));
    }
    let mut candidates = recipe
        .inputs
        .iter()
        .chain(recipe.outputs.iter())
        .copied()
        .filter(|index| {
            stores.get(*index).is_some_and(|store| {
                store.logical_rows == layout.rows && store.logical_columns == layout.columns
            })
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    candidates.dedup();
    match candidates.as_slice() {
        [store] => Ok(Some(*store)),
        [] => exact_recipe_store(recipe, stores),
        _ => Err(format!("prepared node {} matrix allocation has ambiguous stores", recipe.node)),
    }
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
    pub matrix_id: FinalizedMatrixId,
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

/// Structural shape of a scalar native resource. Widths are fixed by the
/// operation-aware warmup projection; runtime values outside that projection
/// are rejected at the input boundary.
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
    pub matrix_id: FinalizedMatrixId,
    pub instance: usize,
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
    /// Threshold decoding first copies its source into a dedicated
    /// coefficient-domain staging owner. The store and owner are planned here
    /// so replay can consume the admitted matrix slot instead of constructing
    /// an unplanned staging owner.
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
    pub finalized_matrices: FinalizedMatrixIdTable,
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
    /// Completion resources reserved for the one synthesized root-input copy
    /// owned by this physical store.  Input copies are not IR commands, so
    /// their native claims live beside the owner rather than being recovered
    /// by scanning the inventory at bind time.
    pub input_copy_slots: Vec<PreparedSlotRef>,
}

impl PartialEq for PreparedResolvedOwner {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key &&
            self.layout.execution_owner_identity() == other.layout.execution_owner_identity() &&
            self.layout.execution_class() == other.layout.execution_class() &&
            self.layout.partition_count() == other.layout.partition_count()
    }
}

/// An exact slot selected during the one warmup admission transaction.
///
/// The request is copied from the accepted storage identity; replay must pass
/// this value through the containing region and never rediscover a slot by
/// scanning the global inventory.
#[derive(Clone)]
pub struct PreparedSlotRef {
    /// Canonical native storage identity selected during warmup.  Binding
    /// carries this owner directly; no region inventory rematching is needed.
    pub storage: Arc<GpuPreparedStorage>,
    pub device: i32,
    pub request: PreparedSlotRequest,
}

impl std::fmt::Debug for PreparedSlotRef {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedSlotRef")
            .field("storage", &self.storage.identity())
            .field("device", &self.device)
            .field("request", &self.request)
            .finish()
    }
}

impl PartialEq for PreparedSlotRef {
    fn eq(&self, other: &Self) -> bool {
        self.storage.identity() == other.storage.identity() &&
            self.device == other.device &&
            self.request == other.request
    }
}

impl Eq for PreparedSlotRef {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PreparedSlotRequest {
    Matrix(GpuPreparedRequest),
    Workspace(mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedWorkspaceRequest),
}

impl PreparedSlotRef {
    pub fn matrix_request(&self) -> Option<GpuPreparedRequest> {
        match self.request {
            PreparedSlotRequest::Matrix(request) => Some(request),
            PreparedSlotRequest::Workspace(_) => None,
        }
    }

    pub fn workspace_request(
        &self,
    ) -> Option<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedWorkspaceRequest> {
        match self.request {
            PreparedSlotRequest::Matrix(_) => None,
            PreparedSlotRequest::Workspace(request) => Some(request),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedAllocationClaim {
    pub command: u32,
    pub ordinal: usize,
    pub store: Option<usize>,
    pub layout: PreparedAllocationLayout,
    /// Exact matrix owner layout resolved during warmup.  Workspace claims
    /// leave this unset; matrix provisioning must never regenerate it from a
    /// stream ordinal or a partial resource-key match.
    pub owner_layout: Option<PreparedOwnerLayout>,
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
    /// Exact logical store used when the native claim does not carry a
    /// physical allocation layout (for example a matrix phase claim).
    pub store: Option<usize>,
    /// The native descriptor which produced this claim, when it is a native
    /// allocation.  Keeping the key beside the traced claim lets admission
    /// match stream footprints by device/partition/limb/role instead of by a
    /// fragile ordinal among unrelated workspace claims.
    pub layout: Option<mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationLayout>,
    pub owner_layout: Option<PreparedOwnerLayout>,
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
    /// Canonical prepared schedule resource selected during warmup. Runtime
    /// provisioning must consume this identity directly; it must not
    /// rediscover a schedule from node numbers or stream geometry.
    pub schedule_id: Option<usize>,
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
pub enum PreparedScheduleResource {
    Allocation(PreparedAllocationClaim),
    Stream(PreparedStreamClaim),
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedResolvedSchedule {
    pub schedule: PreparedSchedulePlan,
    pub resources: Box<[PreparedScheduleResource]>,
}

impl PreparedResolvedSchedule {
    pub fn schedule_allocations_len(&self) -> usize {
        self.resources
            .iter()
            .filter(|resource| matches!(resource, PreparedScheduleResource::Allocation(_)))
            .count()
    }

    pub fn allocation(&self, index: usize) -> Option<&PreparedAllocationClaim> {
        self.resources
            .iter()
            .filter_map(|resource| match resource {
                PreparedScheduleResource::Allocation(allocation) => Some(allocation),
                PreparedScheduleResource::Stream(_) => None,
            })
            .nth(index)
    }

    pub fn allocation_mut(&mut self, index: usize) -> Option<&mut PreparedAllocationClaim> {
        self.resources
            .iter_mut()
            .filter_map(|resource| match resource {
                PreparedScheduleResource::Allocation(allocation) => Some(allocation),
                PreparedScheduleResource::Stream(_) => None,
            })
            .nth(index)
    }

    pub fn allocation_claims(&self) -> impl Iterator<Item = &PreparedAllocationClaim> {
        self.resources.iter().filter_map(|resource| match resource {
            PreparedScheduleResource::Allocation(allocation) => Some(allocation),
            PreparedScheduleResource::Stream(_) => None,
        })
    }

    pub fn stream_claims(&self) -> impl Iterator<Item = &PreparedStreamClaim> {
        self.resources.iter().filter_map(|resource| match resource {
            PreparedScheduleResource::Allocation(_) => None,
            PreparedScheduleResource::Stream(stream) => Some(stream),
        })
    }

    pub fn stream_claims_mut(&mut self) -> impl Iterator<Item = &mut PreparedStreamClaim> {
        self.resources.iter_mut().filter_map(|resource| match resource {
            PreparedScheduleResource::Allocation(_) => None,
            PreparedScheduleResource::Stream(stream) => Some(stream),
        })
    }

    pub fn stream_claim_mut(&mut self, index: usize) -> Option<&mut PreparedStreamClaim> {
        self.resources
            .iter_mut()
            .filter_map(|resource| match resource {
                PreparedScheduleResource::Allocation(_) => None,
                PreparedScheduleResource::Stream(stream) => Some(stream),
            })
            .nth(index)
    }
}

/// One native member stream together with the finalized store identity that
/// owns it. Multiple member commands may expose the same physical stream when
/// they are aliases of one store; candidates resolving to different stores
/// must never be collapsed by stream geometry alone.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PreparedScheduleMemberCandidate {
    pub(super) member: usize,
    pub(super) stream: PreparedStreamFootprint,
    pub(super) store: usize,
    pub(super) owner: u64,
    pub(super) context_identity: usize,
    pub(super) device: i32,
    pub(super) instance: usize,
    pub(super) level: usize,
    pub(super) format: PreparedFormat,
}

/// Compare a saved schedule stream by its complete physical identity.  Role,
/// origin, and pool slot are part of the identity: matching only owner/device
/// would conflate distinct stream completions.
fn same_prepared_stream_physical(
    left: &PreparedStreamFootprint,
    right: &PreparedStreamFootprint,
) -> bool {
    left.key == right.key && left.origin == right.origin && left.pool_slot == right.pool_slot
}

pub(super) fn same_prepared_stream_placement(
    left: &PreparedStreamFootprint,
    right: &PreparedStreamFootprint,
) -> bool {
    left.key.execution_owner_identity == right.key.execution_owner_identity &&
        left.key.context_identity == right.key.context_identity &&
        left.key.instance == right.key.instance &&
        left.key.partition == right.key.partition &&
        left.key.device == right.key.device &&
        left.key.limb_x == right.key.limb_x &&
        left.key.limb_y == right.key.limb_y &&
        left.origin == right.origin &&
        left.pool_slot == right.pool_slot
}

fn validate_prepared_resource_key(
    key: &mxx_primitives::matrix::gpu_dcrt_poly::PreparedResourceKey,
    label: &str,
) -> Result<(), String> {
    let host_partition = key.partition < 0;
    let host_device = key.device < 0;
    if host_partition != host_device {
        return Err(format!(
            "{label} has a mixed host sentinel resource key (partition={}, device={})",
            key.partition, key.device
        ));
    }
    Ok(())
}

fn validate_prepared_layout_keys(layout: &PreparedPlanLayout, label: &str) -> Result<(), String> {
    for allocation in layout.allocations() {
        validate_prepared_resource_key(&allocation.key, label)?;
    }
    for stream in layout.streams() {
        validate_prepared_resource_key(&stream.key, label)?;
    }
    Ok(())
}

pub(super) fn unique_prepared_schedule_member(
    candidates: &[PreparedScheduleMemberCandidate],
    target: &PreparedStreamFootprint,
) -> Result<usize, String> {
    let matches = candidates
        .iter()
        .filter(|candidate| same_prepared_stream_placement(&candidate.stream, target))
        .collect::<Vec<_>>();
    let Some(first) = matches.first() else {
        return Err("schedule stream has no exact member provenance".into());
    };
    let equivalent = matches.iter().all(|candidate| {
        candidate.store == first.store &&
            candidate.owner == first.owner &&
            candidate.context_identity == first.context_identity &&
            candidate.device == first.device &&
            candidate.instance == first.instance &&
            candidate.level == first.level &&
            candidate.format == first.format &&
            same_prepared_stream_placement(&candidate.stream, &first.stream)
    });
    if !equivalent {
        return Err("schedule stream has ambiguous member provenance".into());
    }
    // Aliased member commands are valid, but choose their canonical member
    // deterministically so allocation and completion provenance agree.
    Ok(matches
        .iter()
        .min_by_key(|candidate| (candidate.store, candidate.member))
        .expect("nonempty schedule member matches")
        .member)
}

fn schedule_member_matches(
    schedule: &PreparedSchedulePlan,
    command: &PreparedCommandPlan,
    node: u32,
) -> bool {
    command.node == node &&
        command.instance == schedule.instance &&
        command.stream.as_ref() == Some(&schedule.stream) &&
        command.recipe.owners.iter().any(|owner| schedule.recipe.owners.contains(owner))
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedResolvedScalarBuffer {
    pub plan: PreparedScalarBufferPlan,
    pub layout: PreparedPlanLayout,
    pub slots: Box<[Option<PreparedSlotRef>]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PreparedResolvedResources {
    pub finalized_matrices: FinalizedMatrixIdTable,
    pub owners: Box<[PreparedResolvedOwner]>,
    pub scalar_buffers: Box<[PreparedResolvedScalarBuffer]>,
    pub commands: Box<[PreparedResolvedCommand]>,
    pub schedules: Box<[PreparedResolvedSchedule]>,
}

pub(super) fn first_matrix_wire(program: &GpuPreparation, wire: WireRef) -> Option<WireRef> {
    if program.values.contains_key(&wire) {
        return Some(wire);
    }
    program
        .family_wires
        .get(&wire)
        .and_then(|members| members.iter().find_map(|member| first_matrix_wire(program, *member)))
}

pub fn physical_command_matches(
    finalized_matrices: &FinalizedMatrixIdTable,
    command: &PreparedCommandPlan,
    node: u32,
    instance: usize,
    device: i32,
) -> bool {
    command.node == node &&
        command.instance == instance &&
        command.recipe.owners.iter().any(|owner| {
            finalized_matrices
                .identity(owner.matrix_id)
                .is_some_and(|identity| identity.physical.device == device)
        })
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
            .filter(|owner| {
                self.finalized_matrices
                    .identity(owner.key.matrix_id)
                    .is_some_and(|identity| identity.physical.device == device)
            })
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let commands = self
            .commands
            .iter()
            .filter(|command| {
                command.command.recipe.owners.iter().any(|owner| {
                    self.finalized_matrices
                        .identity(owner.matrix_id)
                        .is_some_and(|identity| identity.physical.device == device)
                })
            })
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let schedules = self
            .schedules
            .iter()
            .filter(|schedule| {
                schedule.schedule.recipe.owners.iter().any(|owner| {
                    self.finalized_matrices
                        .identity(owner.matrix_id)
                        .is_some_and(|identity| identity.physical.device == device)
                })
            })
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let scalar_buffers = self
            .scalar_buffers
            .iter()
            .filter(|buffer| {
                self.finalized_matrices
                    .identity(buffer.plan.owner.matrix_id)
                    .is_some_and(|identity| identity.physical.device == device)
            })
            .cloned()
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            finalized_matrices: self.finalized_matrices.clone(),
            owners,
            scalar_buffers,
            commands,
            schedules,
        }
    }
}

/// Backend hook used only at warmup. Implementations must call the primitive
/// metadata-only owner/stage planners and return their saved descriptors; they
/// must not create owners, reserve storage, or submit CUDA work.
pub trait PreparedResourceBackend {
    fn plan_owner(
        &self,
        finalized_matrices: &FinalizedMatrixIdTable,
        key: &PreparedOwnerKey,
        store: &PreparedStorePlan,
    ) -> Result<PreparedOwnerLayout, String>;

    fn plan_stage(
        &self,
        finalized_matrices: &FinalizedMatrixIdTable,
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
        _finalized_matrices: &FinalizedMatrixIdTable,
        _recipe: &PreparedNativeRecipe,
        _stores: &[PreparedStorePlan],
        _owners: &[PreparedResolvedOwner],
    ) -> Result<(Option<GpuPreparedPreimageLayout>, Option<GpuPreparedTrapdoorLayout>), String>
    {
        Ok((None, None))
    }

    fn plan_accumulate(
        &self,
        _finalized_matrices: &FinalizedMatrixIdTable,
        _recipe: &PreparedNativeRecipe,
        _stores: &[PreparedStorePlan],
        _owners: &[PreparedResolvedOwner],
    ) -> Result<Option<GpuPreparedAccumulateLayout>, String> {
        Ok(None)
    }

    fn plan_replay_upload(
        &self,
        finalized_matrices: &FinalizedMatrixIdTable,
        recipe: &PreparedNativeRecipe,
        replay: &PreparedReplayUploadRecipe,
        stores: &[PreparedStorePlan],
        owners: &[PreparedResolvedOwner],
    ) -> Result<PreparedPlanLayout, String>;

    fn plan_schedule(
        &self,
        _finalized_matrices: &FinalizedMatrixIdTable,
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
        variants: Box<[Box<[PreparedReplayStep]>]>,
        variant_indices: Box<[usize]>,
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
    /// One canonical table for every finalized matrix site.  Native stores,
    /// descriptors, and replay bindings consume IDs from this table directly.
    pub finalized_matrices: FinalizedMatrixIdTable,
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
    pub node_bindings: BTreeMap<u32, (Box<[WireRef]>, Box<[WireRef]>)>,
    pub scalar_commands: BTreeMap<u32, PreparedScalar>,
    /// Stable scalar slots assigned by lowering.  Replay uses these bindings
    /// for inputs, constants, and intermediate results instead of inferring
    /// positions from a node's local argument count.
    pub scalar_slots: BTreeMap<WireRef, usize>,
    pub scalar_slot_count: usize,
    pub scalar_initializers: BTreeMap<usize, ScalarValue>,
    /// Fixed operation-aware scalar widths computed from the warmup replay.
    pub scalar_projections: BTreeMap<usize, usize>,
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
    pub selection_commands: BTreeMap<u32, PreparedSelection>,
    /// Candidate-wire provenance for lowered family/selection descriptors.
    pub selection_candidate_wires: BTreeMap<u32, Box<[WireRef]>>,
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
    /// Promote ownerless root-input warmup records to the synthetic upload
    /// command used by host-staged runtime inputs.  Input nodes have no IR
    /// operation of their own, but the materializer still needs the exact
    /// upload layout and admission claims before it can bind the host bytes.
    /// Keep this transformation in the immutable plan so upload binding does
    /// not rediscover a resource by probing the accepted inventory.
    pub fn mark_host_input_uploads(
        &mut self,
        host_wires: &BTreeSet<WireRef>,
    ) -> Result<(), PreparedLoweringError> {
        for wire in host_wires {
            let mut matched = false;
            for command in &mut self.commands {
                if !matches!(command.operation, PreparedOperation::Warmup) ||
                    !command.outputs.iter().any(|output| output.wire == *wire) ||
                    !matches!(
                        command.recipe.source.as_ref().map(PreparedNodeSource::kind),
                        Some(NodeKind::Input { .. })
                    )
                {
                    continue;
                }
                let stores = command
                    .outputs
                    .iter()
                    .filter_map(|output| output.store)
                    .collect::<BTreeSet<_>>();
                let Some(store_index) = stores.iter().next().copied().filter(|_| stores.len() == 1)
                else {
                    return Err(PreparedLoweringError::InvalidContract(
                        "host input has no unique physical output store",
                    ));
                };
                let store =
                    self.stores.get(store_index).ok_or(PreparedLoweringError::InvalidIndex)?;
                let owner =
                    PreparedOwnerKey { matrix_id: store.matrix_id, instance: command.instance };
                command.operation = PreparedOperation::Gpu(PreparedGpuOperation::RnsUpload);
                command.stream = Some(PreparedStreamKey {
                    stage: PreparedStageRole::Transfer,
                    instance: command.instance,
                    stores: vec![store_index].into_boxed_slice(),
                    placements: vec![PreparedPlacementKey {
                        store: store_index,
                        matrix_id: store.matrix_id,
                    }]
                    .into_boxed_slice(),
                });
                command.recipe.stage =
                    PreparedNativeStage::Transfer(PreparedGpuOperation::RnsUpload);
                command.recipe.inputs = Box::new([]);
                command.recipe.outputs = vec![store_index].into_boxed_slice();
                command.recipe.owners = vec![owner].into_boxed_slice();
                command.recipe.matrix_staging = None;
                command.recipe.scalar = None;
                command.recipe.replay_upload = None;
                matched = true;
            }
            if !matched {
                return Err(PreparedLoweringError::InvalidContract(
                    "host input has no ownerless input resource command",
                ));
            }
        }
        Ok(())
    }

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
                    None if command.recipe.owners.is_empty() => PreparedNativeStage::Control,
                    _ => PreparedNativeStage::Selection,
                }
            } else if matches!(command.operation, PreparedOperation::Scalar) &&
                command.recipe.owners.is_empty()
            {
                // Host-only scalar nodes are replayed by the fixed control
                // tape and intentionally have no native resource plan.
                PreparedNativeStage::Control
            } else {
                prepared_native_stage(&command.operation)
            };
            if command.recipe.stage != expected_stage {
                return Err(PreparedLoweringError::InvalidContract(
                    "native recipe stage does not match command operation",
                ));
            }
            if command.recipe.owners.iter().any(|key| {
                !self
                    .stores
                    .iter()
                    .any(|store| store.matrix_id == key.matrix_id && store.instance == key.instance)
            }) {
                return Err(PreparedLoweringError::InvalidContract(
                    "native recipe owner has no exact matrix store",
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
                        !stream
                            .placements
                            .iter()
                            .any(|placement| placement.matrix_id == staging.matrix_id)
                    {
                        return Err(PreparedLoweringError::InvalidContract(
                            "threshold staging owner is absent from its stream",
                        ));
                    }
                }
                if stream.stores.iter().zip(&stream.placements).any(|(index, placement)| {
                    self.stores.get(*index).is_none_or(|store| {
                        let Some(identity) = self.finalized_matrices.identity(store.matrix_id)
                        else {
                            return true;
                        };
                        store.instance != stream.instance ||
                            placement.store != *index ||
                            placement.matrix_id != store.matrix_id ||
                            identity.execution_instance != stream.instance
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
        if program.instance_count == 0 {
            return Err(PreparedLoweringError::InvalidContract(
                "prepared program must declare at least one execution instance",
            ));
        }
        let mut seeds = Vec::<(
            FinalizedMatrixId,
            WireRef,
            Option<ConcreteWireType>,
            ValueLocation,
            usize,
            usize,
            usize,
        )>::new();
        for (wire, location) in &program.values {
            let key = (location.owner, location.device, location.level, location.format);
            let common_identity = exact_common_instance_identity(
                &program.finalized_matrices,
                *wire,
                location.device,
                program.instance_count,
            )
            .ok_or(PreparedLoweringError::InvalidContract("finalized CRT context is missing"))?;
            let location_context = common_identity.physical.context_identity;
            if let Some((_, (_, _, _, _existing_location, existing_rows, existing_columns, _))) =
                seeds.iter_mut().enumerate().find(
                    |(_, (_, _, _, candidate, _, _, candidate_context))| {
                        (candidate.owner, candidate.device, candidate.level, candidate.format) ==
                            key &&
                            *candidate_context == location_context
                    },
                )
            {
                *existing_rows = (*existing_rows).max(location.rows.end);
                *existing_columns = (*existing_columns).max(location.columns.end);
            } else {
                let matrix_id = program.finalized_matrices.id(common_identity.site).ok_or(
                    PreparedLoweringError::InvalidContract(
                        "ordinary matrix site has no finalized ID",
                    ),
                )?;
                let (capacity_rows, capacity_columns) = program
                    .storage_capacities
                    .get(&location.owner)
                    .copied()
                    .unwrap_or((location.rows.end, location.columns.end));
                seeds.push((
                    matrix_id,
                    *wire,
                    program.wire_types.get(wire).cloned(),
                    location.clone(),
                    capacity_rows,
                    capacity_columns,
                    location_context,
                ));
            }
        }
        for (site, identity) in program.finalized_matrices.iter_variants() {
            let MatrixSite::Variant { node, variant, port, .. } = site else { continue };
            let location = finalized_matrix_location(identity);
            let wire = WireRef { node: NodeId(u64::from(node)), port: Port(port as u32) };
            let matrix_id = program.finalized_matrices.id(site).ok_or(
                PreparedLoweringError::InvalidContract("variant matrix site has no finalized ID"),
            )?;
            let key = (location.owner, location.device, location.level, location.format);
            let location_context = identity.physical.context_identity;
            if let Some((_, (_, _, _, _existing_location, existing_rows, existing_columns, _))) =
                seeds.iter_mut().enumerate().find(
                    |(_, (_, _, _, candidate, _, _, candidate_context))| {
                        (candidate.owner, candidate.device, candidate.level, candidate.format) ==
                            key &&
                            *candidate_context == location_context
                    },
                )
            {
                *existing_rows = (*existing_rows).max(location.rows.end);
                *existing_columns = (*existing_columns).max(location.columns.end);
            } else {
                let (capacity_rows, capacity_columns) = program
                    .storage_capacities
                    .get(&location.owner)
                    .copied()
                    .unwrap_or((location.rows.end, location.columns.end));
                let wire_type = program
                    .node_sources
                    .get(&node)
                    .and_then(|source| source.variant_output_types.get(variant))
                    .and_then(|types| types.get(port))
                    .cloned();
                seeds.push((
                    matrix_id,
                    wire,
                    wire_type,
                    location.clone(),
                    capacity_rows,
                    capacity_columns,
                    location_context,
                ));
            }
        }
        for (site, identity) in program.finalized_matrices.iter_host_staging() {
            let MatrixSite::HostStaging { node, instance: _, device: _ } = site else { continue };
            if !program.values.values().any(|location| location.device == identity.physical.device)
            {
                continue;
            }
            let location = finalized_matrix_location(identity);
            let wire = WireRef { node: NodeId(u64::from(node)), port: Port(0) };
            let matrix_id = program.finalized_matrices.id(site).ok_or(
                PreparedLoweringError::InvalidContract("host staging site has no finalized ID"),
            )?;
            let key = (location.owner, location.device, location.level, location.format);
            let location_context = identity.physical.context_identity;
            if let Some((_, (_, _, _, _existing_location, existing_rows, existing_columns, _))) =
                seeds.iter_mut().enumerate().find(
                    |(_, (_, _, _, candidate, _, _, candidate_context))| {
                        (candidate.owner, candidate.device, candidate.level, candidate.format) ==
                            key &&
                            *candidate_context == location_context
                    },
                )
            {
                *existing_rows = (*existing_rows).max(location.rows.end);
                *existing_columns = (*existing_columns).max(location.columns.end);
            } else {
                let (capacity_rows, capacity_columns) = program
                    .storage_capacities
                    .get(&location.owner)
                    .copied()
                    .unwrap_or((location.rows.end, location.columns.end));
                let wire_type = program
                    .node_bindings
                    .get(&node)
                    .and_then(|(inputs, _)| inputs.first())
                    .and_then(|wire| first_matrix_wire(program, *wire))
                    .and_then(|wire| program.wire_types.get(&wire))
                    .cloned();
                seeds.push((
                    matrix_id,
                    wire,
                    wire_type,
                    location.clone(),
                    capacity_rows,
                    capacity_columns,
                    location_context,
                ));
            }
        }
        if program.instance_count == 0 {
            return Err(PreparedLoweringError::InvalidContract(
                "prepared program must declare at least one execution instance",
            ));
        }
        let instances = program.instance_count;
        let mut stores = Vec::with_capacity(seeds.len() * instances);
        // Resolve every wire through its canonical physical identity.  A view
        // or another alias of the same colored owner must point at the one
        // store entry; using the first wire as the key would leave later
        // aliases unresolved (and could tempt the binder to allocate again).
        let mut store_index = BTreeMap::<(FinalizedMatrixId, usize), usize>::new();
        for instance in 0..instances {
            for (
                matrix_id,
                wire,
                wire_type,
                location,
                _capacity_rows,
                _capacity_columns,
                _context_identity,
            ) in &seeds
            {
                let index = stores.len();
                let matrix_id = program
                    .finalized_matrices
                    .for_execution_instance(*matrix_id, instance)
                    .ok_or(PreparedLoweringError::InvalidContract(
                        "matrix site has no exact execution-instance identity",
                    ))?;
                stores.push(PreparedStorePlan {
                    matrix_id,
                    wire: *wire,
                    wire_type: wire_type.clone().or_else(|| program.wire_types.get(wire).cloned()),
                    instance,
                    logical_rows: location.rows.end,
                    logical_columns: location.columns.end,
                });
                store_index.insert((matrix_id, instance), index);
            }
        }
        // Every semantic alias in the finalized table resolves to the one
        // physical store selected by its canonical physical record. Aliases
        // never allocate a second store or depend on which seed wire won the
        // deduplication pass.
        let canonical_store_keys = stores
            .iter()
            .map(|store| {
                let identity = program.finalized_matrices.identity(store.matrix_id).ok_or(
                    PreparedLoweringError::InvalidContract(
                        "store has no canonical matrix identity",
                    ),
                )?;
                Ok((store.instance, identity.physical.clone()))
            })
            .collect::<Result<Vec<_>, PreparedLoweringError>>()?;
        for (matrix_id, identity) in &program.finalized_matrices.identities {
            let instance = identity.execution_instance;
            let candidates = canonical_store_keys
                .iter()
                .enumerate()
                .filter(|(_, (store_instance, physical))| {
                    *store_instance == instance && *physical == identity.physical
                })
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            if let [index] = candidates.as_slice() {
                store_index.insert((*matrix_id, instance), *index);
            }
        }
        // Threshold replay uses a dedicated coefficient-domain staging owner
        // as the input to the native scalar kernel. It is not an IR value, so
        // reserve it explicitly before constructing command streams.
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
                let source_id = program
                    .finalized_matrices
                    .id(MatrixSite::Ordinary { wire: source_wire, device: source.device, instance })
                    .ok_or(PreparedLoweringError::InvalidContract(
                        "ordinary matrix site has no exact execution-instance ID",
                    ))?;
                let staging_id = program
                    .finalized_matrices
                    .id(MatrixSite::HostStaging { node: node.id, instance, device: source.device })
                    .ok_or(PreparedLoweringError::InvalidContract(
                        "threshold staging matrix site has no exact execution-instance ID",
                    ))?;
                let source_store = *store_index
                    .get(&(source_id, instance))
                    .ok_or(PreparedLoweringError::InvalidRange)?;
                let index = if let Some(index) = store_index.get(&(staging_id, instance)).copied() {
                    index
                } else {
                    let index = stores.len();
                    stores.push(PreparedStorePlan {
                        matrix_id: staging_id,
                        wire: source_wire,
                        wire_type: stores[source_store].wire_type.clone(),
                        instance,
                        logical_rows: 1,
                        logical_columns: 1,
                    });
                    store_index.insert((staging_id, instance), index);
                    index
                };
                threshold_staging_stores.insert((node.id, instance), index);
            }
        }
        // Scalar backings are fixed warmup resources as well. Keep one
        // descriptor per scalar wire and execution instance; replay may only
        // copy values that fit this immutable projection.
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
                .map_err(|_| {
                    PreparedLoweringError::InvalidContract(
                        "threshold length cannot be evaluated during warmup",
                    )
                })?
                .to_usize()
                .ok_or(PreparedLoweringError::InvalidContract(
                    "threshold length does not fit the fixed warmup projection",
                ))?;
            if count == 0 {
                return Err(PreparedLoweringError::InvalidContract(
                    "threshold length must be nonzero",
                ));
            }
            scalar_counts
                .entry(*output)
                .and_modify(|existing| *existing = (*existing).max(count))
                .or_insert(count);
        }
        let scalar_width = |wire: &WireRef| -> Result<usize, PreparedLoweringError> {
            let slot = program.scalar_slots.get(wire).copied().ok_or(
                PreparedLoweringError::InvalidContract("device scalar wire has no scalar slot"),
            )?;
            program.scalar_projections.get(&slot).copied().ok_or_else(|| {
                PreparedLoweringError::InvalidContract("scalar wire has no fixed warmup projection")
            })
        };
        // Scalar buffers and scalar native commands are placed by the matrix
        // value they depend on.  A graph can have several unrelated matrix
        // owners (or no matrix owner at all), so a global first/unique output
        // anchor is not a valid placement rule.  Build a small, immutable
        // scalar-to-matrix provenance table from the lowered topology.
        let mut scalar_placement_wires = BTreeMap::<WireRef, WireRef>::new();
        let mut scalar_dependencies = BTreeMap::<WireRef, Vec<WireRef>>::new();
        for node in &program.topology.nodes {
            let Some((inputs, outputs)) = program.node_bindings.get(&node.id) else { continue };
            let scalar_outputs = outputs
                .iter()
                .flat_map(|wire| std::iter::once(*wire).chain(family_leaf_wires(program, *wire)))
                .filter(|wire| program.scalar_slots.contains_key(wire))
                .collect::<Vec<_>>();
            if scalar_outputs.is_empty() {
                continue;
            }
            let scalar_inputs = inputs
                .iter()
                .flat_map(|wire| std::iter::once(*wire).chain(family_leaf_wires(program, *wire)))
                .filter(|wire| program.scalar_slots.contains_key(wire))
                .collect::<Vec<_>>();
            for output in scalar_outputs {
                scalar_dependencies.insert(output, scalar_inputs.clone());
            }
            // Threshold reads a matrix and writes a scalar; that matrix is
            // the direct physical provenance of its result.
            if matches!(
                node.command.operation,
                PreparedOperation::Gpu(PreparedGpuOperation::ThresholdDecode)
            ) {
                if let Some(matrix) =
                    inputs.iter().copied().find_map(|wire| first_matrix_wire(program, wire))
                {
                    for output in outputs
                        .iter()
                        .copied()
                        .filter(|wire| program.scalar_slots.contains_key(wire))
                    {
                        scalar_placement_wires.insert(output, matrix);
                    }
                }
            }
            // A scalar consumed by a matrix selection or scalar pack inherits
            // the output matrix's owner. This covers scalar chains whose
            // first matrix-dependent node is several operations downstream.
            if let Some(matrix) = outputs.iter().copied().find(|wire| {
                program.wire_types.get(wire).and_then(ConcreteWireType::matrix_type).is_some()
            }) {
                for input in scalar_inputs {
                    scalar_placement_wires.insert(input, matrix);
                }
            }
        }
        // A scalar operation binds all of its scalar operands and results to
        // one physical execution owner.  Resolve this relation to a fixed
        // point from the threshold matrix seeds and matrix-producing pack
        // nodes.  In particular, a host input such as an offset can become a
        // device scalar when it is combined with a threshold result; leaving
        // it unanchored would force the allocator to invent an owner.
        loop {
            let mut changed = false;
            for node in &program.topology.nodes {
                let Some((inputs, outputs)) = program.node_bindings.get(&node.id) else {
                    continue;
                };
                let wires = inputs
                    .iter()
                    .chain(outputs.iter())
                    .flat_map(|wire| {
                        std::iter::once(*wire).chain(family_leaf_wires(program, *wire))
                    })
                    .filter(|wire| program.scalar_slots.contains_key(wire))
                    .collect::<Vec<_>>();
                let mut placements = wires
                    .iter()
                    .filter_map(|wire| scalar_placement_wires.get(wire).copied())
                    .collect::<Vec<_>>();
                placements.sort_unstable();
                placements.dedup();
                if placements.len() > 1 {
                    return Err(PreparedLoweringError::InvalidContract(
                        "scalar value has conflicting matrix placement provenance",
                    ));
                }
                let Some(placement) = placements.first().copied() else { continue };
                for wire in wires {
                    if scalar_placement_wires.insert(wire, placement) != Some(placement) {
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        fn resolve_scalar_placement(
            wire: WireRef,
            direct: &BTreeMap<WireRef, WireRef>,
            dependencies: &BTreeMap<WireRef, Vec<WireRef>>,
            visiting: &mut BTreeSet<WireRef>,
        ) -> Result<Option<WireRef>, PreparedLoweringError> {
            if let Some(placement) = direct.get(&wire).copied() {
                return Ok(Some(placement));
            }
            if !visiting.insert(wire) {
                return Err(PreparedLoweringError::InvalidContract(
                    "scalar placement provenance contains a cycle",
                ));
            }
            let mut placements = Vec::new();
            if let Some(inputs) = dependencies.get(&wire) {
                for input in inputs {
                    if let Some(placement) =
                        resolve_scalar_placement(*input, direct, dependencies, visiting)?
                    {
                        placements.push(placement);
                    }
                }
            }
            visiting.remove(&wire);
            placements.sort_unstable();
            placements.dedup();
            match placements.as_slice() {
                [] => Ok(None),
                [placement] => Ok(Some(*placement)),
                _ => Err(PreparedLoweringError::InvalidContract(
                    "scalar value has conflicting matrix placement provenance",
                )),
            }
        }
        let store_for_scalar =
            |wire: WireRef,
             instance: usize|
             -> Result<Option<&PreparedStorePlan>, PreparedLoweringError> {
                let Some(placement_wire) = resolve_scalar_placement(
                    wire,
                    &scalar_placement_wires,
                    &scalar_dependencies,
                    &mut BTreeSet::new(),
                )?
                else {
                    return Ok(None)
                };
                let placement = program.values.get(&placement_wire).ok_or(
                    PreparedLoweringError::InvalidContract(
                        "scalar placement matrix wire is missing",
                    ),
                )?;
                let context = program
                    .finalized_matrices
                    .ordinary_for_instance(placement_wire, placement.device, instance)
                    .map(|identity| identity.physical.context_identity)
                    .ok_or(PreparedLoweringError::InvalidContract(
                        "scalar placement matrix has no finalized CRT context",
                    ))?;
                let candidates = stores
                    .iter()
                    .filter(|store| {
                        let Some(identity) = program.finalized_matrices.identity(store.matrix_id)
                        else {
                            return false;
                        };
                        store.instance == instance &&
                            identity.physical.owner == placement.owner &&
                            identity.physical.device == placement.device &&
                            identity.physical.level == placement.level &&
                            identity.physical.format == placement.format &&
                            identity.physical.context_identity == context
                    })
                    .collect::<Vec<_>>();
                match candidates.as_slice() {
                    [store] => Ok(Some(*store)),
                    [] => Err(PreparedLoweringError::InvalidContract(
                        "scalar placement has no exact matrix store",
                    )),
                    _ => Err(PreparedLoweringError::InvalidContract(
                        "scalar placement has ambiguous matrix stores",
                    )),
                }
            };
        let mut scalar_stores = BTreeMap::new();
        let mut scalar_buffers = Vec::new();
        for instance in 0..instances {
            if program.scalar_slots.is_empty() {
                break;
            }
            for wire in program
                .scalar_slots
                .keys()
                .copied()
                .filter(|wire| program.device_scalar_wires.contains(wire))
            {
                let placement = match store_for_scalar(wire, instance)? {
                    Some(store) => store,
                    None => {
                        return Err(PreparedLoweringError::InvalidContract(
                            "device scalar has no matrix placement provenance",
                        ));
                    }
                };
                let store_index_for_scalar = store_index
                    .get(&(placement.matrix_id, instance))
                    .copied()
                    .ok_or(PreparedLoweringError::InvalidContract(
                        "device scalar placement store is not indexed",
                    ))?;
                scalar_stores.insert((wire, instance), store_index_for_scalar);
                let words = scalar_width(&wire)?;
                let count = scalar_counts.get(&wire).copied().unwrap_or(1).max(1);
                let pinned_host_bytes = count
                    .checked_mul(words.checked_add(1).ok_or(PreparedLoweringError::InvalidRange)?)
                    .and_then(|value| value.checked_mul(std::mem::size_of::<u64>()))
                    .ok_or(PreparedLoweringError::InvalidRange)?;
                scalar_buffers.push(PreparedScalarBufferPlan {
                    wire,
                    instance,
                    owner: PreparedOwnerKey { matrix_id: placement.matrix_id, instance },
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
            store_index: &BTreeMap<(FinalizedMatrixId, usize), usize>,
            out: &mut Vec<PreparedDescriptorRef>,
        ) -> Result<(), PreparedLoweringError> {
            if let Some(members) = program.family_wires.get(&wire) {
                for member in members {
                    descriptors(program, *member, instance, store_index, out)?;
                }
                return Ok(());
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
            let store = if let Some(location) = program.values.get(&wire) {
                let matrix_id = program
                    .finalized_matrices
                    .id(MatrixSite::Ordinary { wire, device: location.device, instance })
                    .ok_or(PreparedLoweringError::InvalidContract(
                        "matrix descriptor has no exact execution-instance identity",
                    ))?;
                Some(*store_index.get(&(matrix_id, instance)).ok_or(
                    PreparedLoweringError::InvalidContract(
                        "matrix descriptor has no canonical physical store",
                    ),
                )?)
            } else {
                None
            };
            let matrix_id = if kind == PreparedDescriptorKind::Matrix {
                let location =
                    program.values.get(&wire).ok_or(PreparedLoweringError::InvalidContract(
                        "matrix descriptor has no physical value location",
                    ))?;
                Some(
                    program
                        .finalized_matrices
                        .id(MatrixSite::Ordinary { wire, device: location.device, instance })
                        .ok_or(PreparedLoweringError::InvalidContract(
                            "matrix descriptor has no exact execution-instance identity",
                        ))?,
                )
            } else {
                None
            };
            out.push(PreparedDescriptorRef { wire, matrix_id, instance, store, kind });
            Ok(())
        }
        fn stream_key(
            node: &PreparedTopologyNode,
            inputs: &[PreparedDescriptorRef],
            outputs: &[PreparedDescriptorRef],
            stores: &[PreparedStorePlan],
            finalized_matrices: &FinalizedMatrixIdTable,
            scalar_stores: &BTreeMap<(WireRef, usize), usize>,
            instance: usize,
        ) -> Result<Option<PreparedStreamKey>, PreparedLoweringError> {
            if !matches!(
                node.command.operation,
                PreparedOperation::Gpu(_) |
                    PreparedOperation::Scalar |
                    PreparedOperation::Selection
            ) {
                return Ok(None);
            }
            let mut store_ids = inputs
                .iter()
                .chain(outputs.iter())
                .filter_map(|descriptor| {
                    descriptor.store.or_else(|| {
                        (descriptor.kind == PreparedDescriptorKind::Scalar)
                            .then(|| scalar_stores.get(&(descriptor.wire, instance)).copied())
                            .flatten()
                    })
                })
                .collect::<Vec<_>>();
            store_ids.sort_unstable();
            store_ids.dedup();
            if store_ids.is_empty() {
                return Ok(None);
            }
            let placements = store_ids
                .iter()
                .map(|index| {
                    let store =
                        stores.get(*index).ok_or(PreparedLoweringError::InvalidContract(
                            "GPU stream references a missing store",
                        ))?;
                    finalized_matrices.identity(store.matrix_id).ok_or(
                        PreparedLoweringError::InvalidContract(
                            "GPU stream store has no canonical matrix identity",
                        ),
                    )?;
                    Ok(PreparedPlacementKey { store: *index, matrix_id: store.matrix_id })
                })
                .collect::<Result<Vec<_>, PreparedLoweringError>>()?
                .into_boxed_slice();
            Ok(Some(PreparedStreamKey {
                stage: prepared_stage_role(&node.command.operation),
                instance,
                stores: store_ids.into_boxed_slice(),
                placements,
            }))
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
                    descriptors(program, *wire, instance, &store_index, &mut inputs)?;
                }
                for wire in &output_wires {
                    descriptors(program, *wire, instance, &store_index, &mut outputs)?;
                }
                // FamilyPack materializes only the host-side family table; it
                // has no native selection stage. Do not give it a matrix
                // stream (and therefore a schedule) merely because its
                // members carry physical stores.
                let stream = if matches!(
                    program.node_sources.get(&node.id).map(PreparedNodeSource::kind),
                    Some(NodeKind::FamilyPack { .. })
                ) {
                    None
                } else {
                    stream_key(
                        node,
                        &inputs,
                        &outputs,
                        &stores,
                        &program.finalized_matrices,
                        &scalar_stores,
                        instance,
                    )?
                };
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
                                    .map(|index| {
                                        let store = stores.get(*index).ok_or(
                                            PreparedLoweringError::InvalidContract(
                                                "threshold stream references a missing store",
                                            ),
                                        )?;
                                        program.finalized_matrices.identity(store.matrix_id).ok_or(
                                            PreparedLoweringError::InvalidContract(
                                                "threshold stream store has no canonical matrix identity",
                                            ),
                                        )?;
                                        Ok(PreparedPlacementKey {
                                            store: *index,
                                            matrix_id: store.matrix_id,
                                        })
                                    })
                                    .collect::<Result<Vec<_>, PreparedLoweringError>>()?;
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
                                    matrix_id: stores[placement.store].matrix_id,
                                    instance,
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
                    &stores,
                    Some(node.completion),
                );
                if let Some(&staging_store) = threshold_staging_stores.get(&(node.id, instance)) {
                    let store =
                        stores.get(staging_store).ok_or(PreparedLoweringError::InvalidRange)?;
                    recipe.matrix_staging =
                        Some(PreparedOwnerKey { matrix_id: store.matrix_id, instance });
                }
                if matches!(node.command.operation, PreparedOperation::Scalar) {
                    let output_is_device_scalar =
                        output_wires.iter().any(|wire| program.device_scalar_wires.contains(wire));
                    if output_is_device_scalar {
                        let width = |wire: &WireRef| -> Result<usize, PreparedLoweringError> {
                            program
                                .scalar_slots
                                .get(wire)
                                .and_then(|slot| program.scalar_projections.get(slot).copied())
                                .ok_or(PreparedLoweringError::InvalidContract(
                                    "scalar operation has no fixed warmup projection",
                                ))
                        };
                        recipe.scalar = Some(PreparedScalarResource::Op {
                            left_words: input_wires
                                .first()
                                .map(|wire| width(wire))
                                .transpose()?
                                .unwrap_or(1),
                            right_words: input_wires
                                .get(1)
                                .map(|wire| width(wire))
                                .transpose()?
                                .unwrap_or(0),
                            output_words: output_wires
                                .first()
                                .map(|wire| width(wire))
                                .transpose()?
                                .unwrap_or(1),
                            candidate_count: 0,
                        });
                        let output_wire =
                            *output_wires.first().ok_or(PreparedLoweringError::InvalidContract(
                                "device scalar operation has no output wire",
                            ))?;
                        let store = store_for_scalar(output_wire, instance)?.ok_or(
                            PreparedLoweringError::InvalidContract(
                                "device scalar operation has no placement provenance",
                            ),
                        )?;
                        recipe.owners =
                            vec![PreparedOwnerKey { matrix_id: store.matrix_id, instance }]
                                .into_boxed_slice();
                    } else {
                        recipe.stage = PreparedNativeStage::Control;
                        recipe.scalar = None;
                        recipe.owners = Box::new([]);
                    }
                }
                if matches!(node.command.operation, PreparedOperation::Selection) {
                    let scalar_selection_is_device_backed =
                        program.node_bindings.get(&node.id).is_some_and(|(_, outputs)| {
                            outputs.iter().any(|wire| program.device_scalar_wires.contains(wire))
                        }) || program.selection_commands.get(&node.id).is_some_and(|selection| {
                            matches!(
                                selection,
                                PreparedSelection::ScalarDynamic { selector, .. } |
                                    PreparedSelection::ScalarSelect { selector, .. }
                                    if program.device_scalar_wires.contains(selector)
                            )
                        });
                    let scalar_selection = matches!(
                        program.selection_commands.get(&node.id),
                        Some(
                            PreparedSelection::ScalarStatic { .. } |
                                PreparedSelection::ScalarDynamic { .. } |
                                PreparedSelection::ScalarSelect { .. },
                        )
                    );
                    if scalar_selection && scalar_selection_is_device_backed {
                        let Some(
                            PreparedSelection::ScalarStatic { .. } |
                            PreparedSelection::ScalarDynamic { .. } |
                            PreparedSelection::ScalarSelect { .. },
                        ) = program.selection_commands.get(&node.id)
                        else {
                            unreachable!("scalar selection matched above")
                        };
                        recipe.stage = PreparedNativeStage::ScalarOp;
                        let candidate_count = match program.selection_commands.get(&node.id) {
                            Some(PreparedSelection::ScalarStatic { .. }) => 1,
                            Some(PreparedSelection::ScalarDynamic { candidates, .. }) |
                            Some(PreparedSelection::ScalarSelect { candidates, .. }) => {
                                candidates.len()
                            }
                            _ => 0,
                        };
                        let output_wire =
                            *output_wires.first().ok_or(PreparedLoweringError::InvalidContract(
                                "scalar selection has no output wire",
                            ))?;
                        let width = |slot: usize| {
                            program.scalar_projections.get(&slot).copied().ok_or(
                                PreparedLoweringError::InvalidContract(
                                    "scalar selection has no fixed warmup projection",
                                ),
                            )
                        };
                        let left_words = match program.selection_commands.get(&node.id) {
                            Some(PreparedSelection::ScalarStatic { slot }) => width(*slot)?,
                            Some(
                                PreparedSelection::ScalarDynamic { selector, .. } |
                                PreparedSelection::ScalarSelect { selector, .. },
                            ) => width(program.scalar_slots[selector])?,
                            _ => unreachable!("scalar selection matched above"),
                        };
                        let output_words = width(program.scalar_slots[&output_wire])?;
                        recipe.scalar = Some(PreparedScalarResource::Op {
                            left_words,
                            right_words: 0,
                            output_words,
                            candidate_count,
                        });
                        let store = store_for_scalar(output_wire, instance)?.ok_or(
                            PreparedLoweringError::InvalidContract(
                                "scalar selection has no placement provenance",
                            ),
                        )?;
                        recipe.owners =
                            vec![PreparedOwnerKey { matrix_id: store.matrix_id, instance }]
                                .into_boxed_slice();
                    } else if matches!(
                        program.selection_commands.get(&node.id),
                        Some(
                            PreparedSelection::ScalarStatic { .. } |
                                PreparedSelection::ScalarDynamic { .. } |
                                PreparedSelection::ScalarSelect { .. },
                        )
                    ) {
                        // Host-only scalar selections are already represented
                        // by ScalarAlias/ScalarSelection control commands.
                        // They must not acquire a native scalar resource or
                        // invent a matrix placement anchor.
                        recipe.stage = PreparedNativeStage::Control;
                        recipe.scalar = None;
                        recipe.owners = Box::new([]);
                    }
                    if let Some(
                        PreparedSelection::Dynamic { candidates, selector } |
                        PreparedSelection::Select { candidates, selector },
                    ) = program.selection_commands.get(&node.id)
                    {
                        if program.device_scalar_wires.contains(selector) {
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
                                let store_index = outputs
                                    .first()
                                    .and_then(|descriptor| descriptor.store)
                                    .ok_or(PreparedLoweringError::InvalidContract(
                                        "scalar matrix selection has no output store",
                                    ))?;
                                let store = stores
                                    .get(store_index)
                                    .ok_or(PreparedLoweringError::InvalidRange)?;
                                recipe.owners =
                                    vec![PreparedOwnerKey { matrix_id: store.matrix_id, instance }]
                                        .into_boxed_slice();
                            }
                        }
                    }
                    if recipe.owners.is_empty() && recipe.scalar.is_none() {
                        // Host-only family selections are replayed by the
                        // control tape and have no native resource footprint.
                        recipe.stage = PreparedNativeStage::Control;
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
            finalized_matrices: program.finalized_matrices.clone(),
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
            let finalized_matrices = physical.finalized_matrices.clone();
            for (wire, location) in physical.values.iter_mut() {
                let identity = exact_common_instance_identity(
                    &finalized_matrices,
                    *wire,
                    device,
                    physical.instance_count,
                )
                .ok_or(PreparedLoweringError::InvalidContract(
                    "ordinary matrix identity is missing for physical device",
                ))?;
                *location = finalized_matrix_location(identity);
            }
            {
                // Auxiliary descriptors do not carry their wire key.  The
                // preparation pass records that key explicitly, so physical
                // expansion must use the recorded (wire, device) state and
                // never reverse-match by owner/shape geometry.  Equal aliases
                // are valid because they resolve through the same wire state;
                // conflicting finalized states are rejected by the direct
                // table lookup below.
                let remap_aux = |location: &mut ValueLocation, wire: WireRef| {
                    let identity = exact_common_instance_identity(
                        &finalized_matrices,
                        wire,
                        device,
                        physical.instance_count,
                    )
                    .ok_or(PreparedLoweringError::InvalidContract(
                        "auxiliary location has no finalized physical identity",
                    ))?;
                    let rows = location.rows.clone();
                    let columns = location.columns.clone();
                    *location =
                        ValueLocation { rows, columns, ..finalized_matrix_location(identity) };
                    Ok(())
                };
                for (family, members) in physical.family_members.iter_mut() {
                    let member_wires = program
                        .family_wires
                        .get(family)
                        .map(|wires| {
                            wires
                                .iter()
                                .copied()
                                .filter(|wire| program.values.contains_key(wire))
                                .collect::<Vec<_>>()
                        })
                        .ok_or(PreparedLoweringError::InvalidContract(
                            "family has no explicit member-wire provenance",
                        ))?;
                    if members.len() != member_wires.len() {
                        return Err(PreparedLoweringError::InvalidContract(
                            "family/member-wire provenance cardinality mismatch",
                        ));
                    }
                    for (location, wire) in members.iter_mut().zip(member_wires.into_iter()) {
                        remap_aux(location, wire)?;
                    }
                }
                for (node, selection) in physical.selection_commands.iter_mut() {
                    match selection {
                        PreparedSelection::Static { location } => {
                            let wires = program.selection_candidate_wires.get(node).ok_or(
                                PreparedLoweringError::InvalidContract(
                                    "selection has no explicit candidate-wire provenance",
                                ),
                            )?;
                            let wire = wires.first().copied().ok_or(
                                PreparedLoweringError::InvalidContract(
                                    "static selection has no candidate-wire provenance",
                                ),
                            )?;
                            remap_aux(location, wire)?
                        }
                        PreparedSelection::Dynamic { candidates, .. } |
                        PreparedSelection::Select { candidates, .. } => {
                            let wires = program.selection_candidate_wires.get(node).ok_or(
                                PreparedLoweringError::InvalidContract(
                                    "selection has no explicit candidate-wire provenance",
                                ),
                            )?;
                            if candidates.len() != wires.len() {
                                return Err(PreparedLoweringError::InvalidContract(
                                    "selection/candidate-wire provenance cardinality mismatch",
                                ));
                            }
                            for (location, wire) in candidates.iter_mut().zip(wires.iter()) {
                                remap_aux(location, *wire)?;
                            }
                        }
                        PreparedSelection::ScalarStatic { .. } |
                        PreparedSelection::ScalarDynamic { .. } |
                        PreparedSelection::ScalarSelect { .. } => {}
                    }
                }
            }
            physical.finalized_matrices.by_site.retain(|site, _| match site {
                MatrixSite::Ordinary { device: source_device, .. } |
                MatrixSite::Variant { device: source_device, .. } |
                MatrixSite::HostStaging { device: source_device, .. } => *source_device == device,
            });
            physical.finalized_matrices.identities.retain(|_, identity| match identity.site {
                MatrixSite::Ordinary { device: source_device, .. } |
                MatrixSite::Variant { device: source_device, .. } |
                MatrixSite::HostStaging { device: source_device, .. } => source_device == device,
            });
            plans.push(Self::from_preparation(&physical)?);
        }
        Self::merge_physical_plans(plans)
    }

    fn merge_physical_plans(plans: Vec<Self>) -> Result<Self, PreparedLoweringError> {
        let instance_count = plans
            .first()
            .map(|plan| plan.instance_count)
            .ok_or(PreparedLoweringError::InvalidContract("physical plan merge has no plans"))?;
        if instance_count == 0 || plans.iter().any(|plan| plan.instance_count != instance_count) {
            return Err(PreparedLoweringError::InvalidContract(
                "physical plan merge requires one identical nonzero instance count",
            ));
        }
        let mut finalized_matrices = FinalizedMatrixIdTable::default();
        let mut id_maps = Vec::<BTreeMap<FinalizedMatrixId, FinalizedMatrixId>>::new();
        for plan in &plans {
            let mut id_map = BTreeMap::new();
            for (old_id, identity) in &plan.finalized_matrices.identities {
                let new_id = if let Some(site_id) = finalized_matrices.by_site.get(&identity.site) {
                    let existing = finalized_matrices.identities.get(site_id).ok_or(
                        PreparedLoweringError::InvalidContract(
                            "physical plan merge has a dangling site identity",
                        ),
                    )?;
                    if existing != identity {
                        return Err(PreparedLoweringError::InvalidContract(
                            "physical plan merge has conflicting site identities",
                        ));
                    }
                    *site_id
                } else if let Some(existing) = finalized_matrices.identities.get(old_id) {
                    if existing == identity {
                        finalized_matrices.by_site.insert(identity.site, *old_id);
                        *old_id
                    } else {
                        let fresh = FinalizedMatrixId(finalized_matrices.next);
                        finalized_matrices.next = finalized_matrices
                            .next
                            .checked_add(1)
                            .ok_or(PreparedLoweringError::InvalidContract("matrix ID overflow"))?;
                        finalized_matrices.by_site.insert(identity.site, fresh);
                        finalized_matrices.identities.insert(fresh, identity.clone());
                        fresh
                    }
                } else {
                    finalized_matrices.next = finalized_matrices.next.max(old_id.0);
                    finalized_matrices.by_site.insert(identity.site, *old_id);
                    finalized_matrices.identities.insert(*old_id, identity.clone());
                    finalized_matrices.next = finalized_matrices.next.max(old_id.0 + 1);
                    *old_id
                };
                id_map.insert(*old_id, new_id);
            }
            id_maps.push(id_map);
        }
        for (plan, id_map) in plans.iter().zip(&id_maps) {
            for (site, id) in &plan.finalized_matrices.by_site {
                if id_map.get(id) != finalized_matrices.by_site.get(site) {
                    return Err(PreparedLoweringError::InvalidContract(
                        "physical plan merge produced an inconsistent site remap",
                    ));
                }
            }
        }
        let mut stores = Vec::new();
        let mut commands = Vec::new();
        let mut schedules = Vec::new();
        let mut scalar_buffers = Vec::new();
        for (plan, id_map) in plans.into_iter().zip(id_maps) {
            let store_offset = stores.len();
            stores.extend(plan.stores.iter().cloned().map(|mut store| {
                store.matrix_id = *id_map.get(&store.matrix_id).expect("store ID was mapped");
                store
            }));
            let remap_store = |index: usize| index + store_offset;
            let remap_owner = |mut owner: PreparedOwnerKey| {
                owner.matrix_id = *id_map.get(&owner.matrix_id).expect("owner ID was mapped");
                owner
            };
            let remap_descriptor = |descriptor: PreparedDescriptorRef| PreparedDescriptorRef {
                store: descriptor.store.map(remap_store),
                matrix_id: descriptor
                    .matrix_id
                    .map(|id| *id_map.get(&id).expect("descriptor ID was mapped")),
                ..descriptor
            };
            let remap_stream = |stream: &PreparedStreamKey| PreparedStreamKey {
                stores: stream.stores.iter().copied().map(remap_store).collect(),
                placements: stream
                    .placements
                    .iter()
                    .map(|placement| PreparedPlacementKey {
                        store: remap_store(placement.store),
                        matrix_id: *id_map
                            .get(&placement.matrix_id)
                            .expect("placement ID was mapped"),
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
                command.recipe.owners =
                    command.recipe.owners.iter().copied().map(remap_owner).collect();
                command.recipe.matrix_staging = command.recipe.matrix_staging.map(remap_owner);
                commands.push(command);
            }
            for mut schedule in plan.schedules.iter().cloned() {
                schedule.stream = remap_stream(&schedule.stream);
                schedule.recipe.inputs =
                    schedule.recipe.inputs.iter().copied().map(remap_store).collect();
                schedule.recipe.outputs =
                    schedule.recipe.outputs.iter().copied().map(remap_store).collect();
                schedule.recipe.owners =
                    schedule.recipe.owners.iter().copied().map(remap_owner).collect();
                schedule.recipe.matrix_staging = schedule.recipe.matrix_staging.map(remap_owner);
                schedules.push(schedule);
            }
            scalar_buffers.extend(plan.scalar_buffers.iter().cloned().map(|mut buffer| {
                buffer.owner = remap_owner(buffer.owner);
                buffer
            }));
        }
        // Device expansion duplicates ownerless control records even though
        // they have no physical resource identity. Keep one logical control
        // command per `(instance,node)` and reject divergent records instead
        // of selecting a device-dependent first match.
        let mut canonical_controls = BTreeMap::<(usize, u32), PreparedCommandPlan>::new();
        let mut deduplicated_commands = Vec::with_capacity(commands.len());
        for command in commands {
            if command.recipe.stage == PreparedNativeStage::Control &&
                command.recipe.owners.is_empty() &&
                command.stream.is_none()
            {
                let key = (command.instance, command.node);
                if let Some(existing) = canonical_controls.get(&key) {
                    if existing != &command {
                        return Err(PreparedLoweringError::InvalidContract(
                            "physical plan merge has conflicting control identities",
                        ));
                    }
                    continue;
                }
                canonical_controls.insert(key, command.clone());
            }
            deduplicated_commands.push(command);
        }
        let merged = Self {
            instance_count,
            finalized_matrices,
            stores: stores.into_boxed_slice(),
            commands: deduplicated_commands.into_boxed_slice(),
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
        let instance_key = |instance: usize| {
            u64::try_from(instance)
                .map_err(|_| "prepared execution instance overflows resource key".to_owned())
        };
        let stamp_allocation = |mut layout: PreparedAllocationLayout, instance: usize| {
            layout.key.instance = instance_key(instance)?;
            Ok::<_, String>(layout)
        };
        let stamp_stream = |mut layout: PreparedStreamFootprint, instance: usize| {
            layout.key.instance = instance_key(instance)?;
            Ok::<_, String>(layout)
        };
        let mut owner_keys = self.stores.iter().map(store_owner_key).collect::<Vec<_>>();
        for command in &self.commands {
            owner_keys.extend(command.recipe.owners.iter().copied());
        }
        owner_keys.extend(self.scalar_buffers.iter().map(|buffer| buffer.owner));
        owner_keys.sort_unstable();
        owner_keys.dedup();
        let mut owners = Vec::with_capacity(owner_keys.len());
        for key in owner_keys {
            let store = self
                .stores
                .iter()
                .find(|store| store.matrix_id == key.matrix_id && store.instance == key.instance)
                .ok_or_else(|| {
                    format!("prepared matrix {:?} has no matrix store", key.matrix_id)
                })?;
            let layout = backend.plan_owner(&self.finalized_matrices, &key, store)?;
            owners.push(PreparedResolvedOwner {
                key,
                layout,
                slot: None,
                input_copy_slots: Vec::new(),
            });
        }
        let owner_layout_for_key =
            |key: &PreparedResourceKey| -> Result<PreparedOwnerLayout, String> {
                let matches = owners
                    .iter()
                    .filter(|owner| {
                        owner.layout.execution_owner_identity() == key.execution_owner_identity &&
                            owner.layout.contains_resource_key(key)
                    })
                    .map(|owner| owner.layout)
                    .collect::<Vec<_>>();
                match matches.as_slice() {
                    [layout] => Ok(*layout),
                    [] => Err("prepared matrix claim has no exact owner layout".into()),
                    _ => Err("prepared matrix claim has ambiguous owner layouts".into()),
                }
            };
        let owner_layout_for_store = |store_index: usize| -> Result<PreparedOwnerLayout, String> {
            let store = self
                .stores
                .get(store_index)
                .ok_or("prepared matrix claim store provenance is invalid")?;
            let matches = owners
                .iter()
                .filter(|owner| {
                    owner.key.matrix_id == store.matrix_id && owner.key.instance == store.instance
                })
                .map(|owner| owner.layout)
                .collect::<Vec<_>>();
            match matches.as_slice() {
                [layout] => Ok(*layout),
                [] => Err("prepared matrix claim store has no exact owner layout".into()),
                _ => Err("prepared matrix claim store has ambiguous owner layouts".into()),
            }
        };
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
                .plan_stage(&self.finalized_matrices, &recipe, &self.stores, &owners)?
                .ok_or("scalar buffer native plan missing")?;
            validate_prepared_layout_keys(&layout, "scalar buffer native layout")?;
            let layout = layout.with_instance(instance_key(buffer.instance)?);
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
            let native = backend.plan_stage(
                &self.finalized_matrices,
                &command.recipe,
                &self.stores,
                &owners,
            )?;
            if let Some(layout) = native.as_ref() {
                validate_prepared_layout_keys(layout, "native command layout")?;
            }
            let (preimage, trapdoor) = backend.plan_sampler_bundles(
                &self.finalized_matrices,
                &command.recipe,
                &self.stores,
                &owners,
            )?;
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
            let composite_store = if composite_claims.is_empty() {
                None
            } else {
                exact_recipe_store(&command.recipe, &self.stores)?
            };
            if !composite_claims.is_empty() && composite_store.is_none() {
                return Err(format!(
                    "prepared node {} composite recipe has no exact store",
                    command.node
                ));
            }
            let composite_owner_layouts = if let Some(layout) = preimage.as_ref() {
                layout.claim_owner_layouts()?
            } else if let Some(layout) = trapdoor.as_ref() {
                layout.claim_owner_layouts()?
            } else {
                Vec::new()
            };
            if composite_owner_layouts.len() != composite_claims.len() {
                return Err(format!(
                    "prepared node {} composite claim/owner table mismatch",
                    command.node
                ));
            }
            let composite_allocations = composite_claims
                .into_iter()
                .zip(composite_layouts)
                .enumerate()
                .map(|(ordinal, (claim, layout))| {
                    if let Some(layout) = layout.as_ref() {
                        validate_prepared_resource_key(&layout.key, "composite allocation layout")?;
                    }
                    let layout = layout
                        .map(|layout| stamp_allocation(layout, command.instance))
                        .transpose()?;
                    // Every composite phase claim carries the exact
                    // finalized logical owner saved by the primitive layout.
                    // Workspace claims use this same provenance; no phase
                    // ordinal or native-key inference is allowed.
                    let owner_layout = Some(composite_owner_layouts[ordinal]);
                    Ok(PreparedCompositeClaim {
                        command: command.node,
                        ordinal,
                        claim,
                        store: composite_store,
                        layout,
                        owner_layout,
                        slot: None,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?
                .into_boxed_slice();
            let composite_streams = preimage
                .as_ref()
                .map(|layout| layout.streams())
                .or_else(|| trapdoor.as_ref().map(|layout| layout.streams()))
                .unwrap_or_default()
                .into_iter()
                .enumerate()
                .map(|(ordinal, layout)| {
                    validate_prepared_resource_key(&layout.key, "composite stream layout")?;
                    Ok(PreparedCompositeStream {
                        command: command.node,
                        ordinal,
                        layout: stamp_stream(layout, command.instance)?,
                        slot: None,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?
                .into_boxed_slice();
            if !composite_allocations.is_empty() {
                validate_composite_stream_claims(&composite_allocations, &composite_streams)
                    .map_err(|error| format!("prepared node {}: {error}", command.node))?;
            }
            let accumulate = backend.plan_accumulate(
                &self.finalized_matrices,
                &command.recipe,
                &self.stores,
                &owners,
            )?;
            // Control/view/selection commands carry source and destination
            // stores for dependency metadata, but do not own an allocation.
            // Resolve a recipe store only when the replay-upload binder will
            // actually attach one of its native claims to this command.
            let store = if command.recipe.replay_upload.is_some() {
                exact_recipe_store(&command.recipe, &self.stores)?
            } else {
                None
            };
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
                        // A row/diagonal concat is represented by one
                        // prepared input-copy descriptor per source.  The
                        // native descriptor is reused for each copy. Retain
                        // one exact claim copy per source so each independent
                        // copy bind consumes its own completion resources.
                        let allocation_repetitions = if matches!(
                            command.operation,
                            PreparedOperation::Gpu(
                                PreparedGpuOperation::FixedCopies |
                                    PreparedGpuOperation::ConcatRows
                            )
                        ) && matches!(
                            command.recipe.source.as_ref().map(PreparedNodeSource::kind),
                            Some(NodeKind::Concat { .. })
                        ) {
                            command.recipe.inputs.len().max(1)
                        } else {
                            1
                        };
                        let allocations = native
                            .allocations()
                            .iter()
                            .copied()
                            .cycle()
                            .take(native.allocations().len() * allocation_repetitions)
                            .enumerate()
                            .map(|(ordinal, layout)| {
                                let store = recipe_store_for_layout(
                                    &command.recipe,
                                    &self.stores,
                                    &layout,
                                )?;
                                let owner_layout = if layout.kind == 0 &&
                                    matches!(
                                        command.operation,
                                        PreparedOperation::Gpu(
                                            PreparedGpuOperation::HashCompactDecompose
                                        )
                                    ) {
                                    Some(native
                                        .owner_layout()
                                        .ok_or(
                                            "prepared compact hash matrix scratch owner is missing or conflicting",
                                        )?)
                                } else if layout.kind != 100 {
                                    Some(match store {
                                        Some(store) => owner_layout_for_store(store)?,
                                        None => owner_layout_for_key(&layout.key)?,
                                    })
                                } else {
                                    None
                                };
                                Ok(PreparedAllocationClaim {
                                    command: command.node,
                                    ordinal,
                                    store,
                                    layout: stamp_allocation(layout, command.instance)?,
                                    owner_layout,
                                    slot: None,
                                })
                            })
                            .collect::<Result<Vec<_>, String>>()?
                            .into_boxed_slice();
                        let streams = native
                            .streams()
                            .iter()
                            .copied()
                            .enumerate()
                            .map(|(ordinal, layout)| {
                                Ok(PreparedStreamClaim {
                                    command: command.node,
                                    ordinal,
                                    layout: stamp_stream(layout, command.instance)?,
                                    slot: None,
                                })
                            })
                            .collect::<Result<Vec<_>, String>>()?
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
            let mut allocations = allocations.into_vec();
            if matches!(
                command.operation,
                PreparedOperation::Gpu(PreparedGpuOperation::GadgetDecompose)
            ) {
                let output_store = match command.recipe.outputs.as_ref() {
                    [store] => *store,
                    _ => {
                        return Err(format!(
                            "prepared gadget command {} must have one exact output store",
                            command.node
                        ));
                    }
                };
                let output = self.stores.get(output_store).ok_or_else(|| {
                    format!("prepared gadget command {} output store is invalid", command.node)
                })?;
                let (matrix, bound) = match output.wire_type.as_ref() {
                    Some(ConcreteWireType::Preimage { matrix, max_coefficient_bound }) |
                    Some(ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound }) => {
                        (matrix, max_coefficient_bound)
                    }
                    _ => {
                        return Err(format!(
                            "prepared gadget command {} output is not compact typed",
                            command.node
                        ));
                    }
                };
                let magnitude_bytes = bound.to_bytes_le().1.len().max(1);
                let payload_bytes = matrix
                    .rows
                    .checked_mul(matrix.columns)
                    .and_then(|count| count.checked_mul(matrix.ring_dimension))
                    .and_then(|count| count.checked_mul(1usize.checked_add(magnitude_bytes)?))
                    .ok_or_else(|| {
                        format!("prepared gadget command {} payload size overflow", command.node)
                    })?;
                let payload_template = allocations
                    .iter()
                    .find(|allocation| {
                        allocation.layout.kind == 0 && allocation.store == Some(output_store)
                    })
                    .ok_or_else(|| {
                        format!(
                            "prepared gadget command {} has no exact output matrix allocation",
                            command.node
                        )
                    })?;
                let mut payload_layout = payload_template.layout;
                payload_layout.kind = 4;
                payload_layout.rows = 0;
                payload_layout.columns = 0;
                payload_layout.bytes = payload_bytes;
                payload_layout.alignment = 256;
                payload_layout.level = -1;
                payload_layout.format = -1;
                allocations.push(PreparedAllocationClaim {
                    command: command.node,
                    ordinal: allocations.len(),
                    store: Some(output_store),
                    layout: stamp_allocation(payload_layout, command.instance)?,
                    owner_layout: Some(owner_layout_for_store(output_store)?),
                    slot: None,
                });
            }
            let replay_upload = command
                .recipe
                .replay_upload
                .as_ref()
                .map(|replay| -> Result<PreparedResolvedReplayUpload, String> {
                    let layout = backend.plan_replay_upload(
                        &self.finalized_matrices,
                        &command.recipe,
                        replay,
                        &self.stores,
                        &owners,
                    )?;
                    validate_prepared_layout_keys(&layout, "replay upload layout")?;
                    let replay_layout = layout.with_instance(instance_key(command.instance)?);
                    let replay_allocations = replay_layout
                        .allocations()
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(ordinal, layout)| {
                            Ok(PreparedAllocationClaim {
                                command: command.node,
                                ordinal,
                                store,
                                layout,
                                owner_layout: if layout.kind != 100 {
                                    Some(match store {
                                        Some(store) => owner_layout_for_store(store)?,
                                        None => owner_layout_for_key(&layout.key)?,
                                    })
                                } else {
                                    None
                                },
                                slot: None,
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?
                        .into_boxed_slice();
                    let replay_streams = replay_layout
                        .streams()
                        .iter()
                        .copied()
                        .enumerate()
                        .map(|(ordinal, layout)| {
                            Ok(PreparedStreamClaim {
                                command: command.node,
                                ordinal,
                                layout,
                                slot: None,
                            })
                        })
                        .collect::<Result<Vec<_>, String>>()?
                        .into_boxed_slice();
                    Ok(PreparedResolvedReplayUpload {
                        recipe: replay.clone(),
                        layout: replay_layout,
                        allocations: replay_allocations,
                        streams: replay_streams,
                    })
                })
                .transpose()?;
            commands.push(PreparedResolvedCommand {
                command: command.clone(),
                schedule_id: None,
                native,
                accumulate,
                preimage,
                trapdoor,
                allocations: allocations.into_boxed_slice(),
                composite_allocations,
                composite_streams,
                streams,
                replay_upload,
            });
        }
        let mut schedules = Vec::with_capacity(self.schedules.len());
        for schedule in &self.schedules {
            // Resolve the schedule's member commands once, retaining the
            // member identity alongside its native descriptor.  The native
            // schedule planner emits member-preserving allocation/stream
            // entries; resource binding below consumes this table directly.
            let member_commands = schedule
                .command_group
                .iter()
                .map(|node| {
                    commands
                        .iter()
                        .find(|command| schedule_member_matches(schedule, &command.command, *node))
                        .ok_or_else(|| {
                            format!(
                                "prepared schedule {} has no exact member command {node}",
                                schedule.instance
                            )
                        })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let members = member_commands
                .iter()
                .filter_map(|command| command.native.clone())
                .collect::<Vec<_>>();
            let native =
                backend.plan_schedule(&self.finalized_matrices, &schedule.recipe, &members)?;
            validate_prepared_layout_keys(&native, "native schedule layout")?;
            for (index, stream) in native.streams().iter().enumerate() {
                if native.streams()[..index]
                    .iter()
                    .any(|previous| same_prepared_stream_physical(previous, stream))
                {
                    return Err(format!(
                        "prepared schedule {} has ambiguous duplicate physical stream provenance",
                        schedule.instance
                    ));
                }
            }
            let member_stream_matches =
                |schedule_stream: &PreparedStreamFootprint| -> Result<(usize, usize), String> {
                    let mut candidates = Vec::new();
                    for (member, command) in member_commands.iter().enumerate() {
                        let Some(member_layout) = command.native.as_ref() else { continue };
                        let Some(store) =
                            exact_recipe_store(&command.command.recipe, &self.stores)?
                        else {
                            continue;
                        };
                        let store_plan = self.stores.get(store).ok_or_else(|| {
                            format!(
                                "prepared schedule {} member stream has invalid store",
                                schedule.instance
                            )
                        })?;
                        let identity = self
                            .finalized_matrices
                            .identity(store_plan.matrix_id)
                            .ok_or_else(|| {
                                format!(
                                    "prepared schedule {} member store has no canonical identity",
                                    schedule.instance
                                )
                            })?;
                        for member_stream in member_layout.streams() {
                            // The schedule rewrites the role to SCHEDULE, but
                            // every other physical stream identity remains
                            // immutable. Include origin and pool slot: a
                            // same-device stream with a different pool slot
                            // is a distinct completion target.
                            if same_prepared_stream_placement(member_stream, schedule_stream) {
                                candidates.push(PreparedScheduleMemberCandidate {
                                    member,
                                    stream: *member_stream,
                                    store,
                                    owner: identity.physical.owner,
                                    context_identity: identity.physical.context_identity,
                                    device: identity.physical.device,
                                    instance: store_plan.instance,
                                    level: identity.physical.level,
                                    format: identity.physical.format,
                                });
                            }
                        }
                    }
                    let member = unique_prepared_schedule_member(&candidates, schedule_stream)
                        .map_err(|error| {
                            format!("prepared schedule {} {error}", schedule.instance)
                        })?;
                    let store =
                        exact_recipe_store(&member_commands[member].command.recipe, &self.stores)?
                            .ok_or_else(|| {
                                format!(
                                    "prepared schedule {} member stream has no exact store",
                                    schedule.instance
                                )
                            })?;
                    Ok((member, store))
                };
            let streams = native
                .streams()
                .iter()
                .copied()
                .enumerate()
                .map(|(ordinal, layout)| {
                    Ok(PreparedStreamClaim {
                        command: schedule.command_group.first().copied().unwrap_or_default(),
                        ordinal,
                        layout: stamp_stream(layout, schedule.instance)?,
                        slot: None,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?
                .into_boxed_slice();
            let completion_count =
                native.allocations().iter().filter(|layout| layout.kind == 8).count();
            if completion_count != native.streams().len() {
                return Err(format!(
                    "prepared schedule {} has {} completion allocations for {} streams",
                    schedule.instance,
                    completion_count,
                    native.streams().len()
                ));
            }
            let allocations = native
                .allocations()
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, layout)| layout.kind == 8)
                .map(|(ordinal, layout)| {
                    let schedule_stream = native
                        .streams()
                        .iter()
                        .find(|stream| stream.key == layout.key)
                        .ok_or_else(|| {
                            format!(
                                "prepared schedule {} completion has no exact stream provenance",
                                schedule.instance
                            )
                        })?;
                    let (_member, store) = member_stream_matches(schedule_stream)?;
                    let allocation = PreparedAllocationClaim {
                        command: schedule.command_group.first().copied().unwrap_or_default(),
                        ordinal,
                        store: Some(store),
                        layout: stamp_allocation(layout, schedule.instance)?,
                        owner_layout: Some(owner_layout_for_store(store)?),
                        slot: None,
                    };
                    Ok(allocation)
                })
                .collect::<Result<Vec<_>, String>>()?
                .into_boxed_slice();
            let resources = allocations
                .into_iter()
                .map(PreparedScheduleResource::Allocation)
                .chain(streams.into_iter().map(PreparedScheduleResource::Stream))
                .collect::<Vec<_>>()
                .into_boxed_slice();
            schedules.push(PreparedResolvedSchedule { schedule: schedule.clone(), resources });
        }
        for command in &mut commands {
            let schedule_matches = schedules
                .iter()
                .enumerate()
                .filter(|(_, schedule)| {
                    if schedule.schedule.instance != command.command.instance {
                        return false;
                    }
                    if !schedule.schedule.command_group.contains(&command.command.node) {
                        return false;
                    }
                    if command.command.stream.as_ref() == Some(&schedule.schedule.stream) {
                        return true;
                    }
                    let Some(native) = command.native.as_ref() else { return false };
                    let native_streams = native.streams();
                    let schedule_streams = schedule.stream_claims().collect::<Vec<_>>();
                    let native_stream_match = native_streams.len() == schedule_streams.len() &&
                        native_streams.iter().all(|native_stream| {
                            schedule_streams.iter().any(|schedule_stream| {
                                same_prepared_stream_placement(
                                    native_stream,
                                    &schedule_stream.layout,
                                )
                            })
                        });
                    let staging_owner_match =
                        command.command.recipe.matrix_staging.is_some_and(|staging| {
                            schedule.schedule.recipe.owners.iter().copied().any(|candidate| {
                                candidate.instance == staging.instance &&
                                    self.finalized_matrices
                                        .identity(candidate.matrix_id)
                                        .zip(self.finalized_matrices.identity(staging.matrix_id))
                                        .is_some_and(|(candidate, staging)| {
                                            candidate.physical == staging.physical
                                        })
                            })
                        });
                    native_stream_match || staging_owner_match
                })
                .map(|(index, _)| index)
                .collect::<Vec<_>>();
            command.schedule_id = match schedule_matches.as_slice() {
                [] => None,
                [schedule_id] => Some(*schedule_id),
                _ => {
                    return Err("prepared command has ambiguous canonical schedule identity".into())
                }
            };
        }
        Ok(PreparedResolvedResources {
            finalized_matrices: self.finalized_matrices.clone(),
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
    stores: &[PreparedStorePlan],
    completion: Option<u32>,
) -> PreparedNativeRecipe {
    let mut input_stores =
        inputs.iter().filter_map(|descriptor| descriptor.store).collect::<Vec<_>>();
    // TrapdoorSample has two IR output ports: public matrix and secret
    // trapdoor metadata. The secret wire aliases the public owner for replay,
    // but is not a second native sampler output/store. Preserve the port
    // identity here instead of selecting an arbitrary store later.
    let recipe_outputs =
        if matches!(operation, PreparedOperation::Gpu(PreparedGpuOperation::TrapdoorSample)) {
            outputs.iter().take(1).collect::<Vec<_>>()
        } else {
            outputs.iter().collect::<Vec<_>>()
        };
    let mut output_stores =
        recipe_outputs.iter().filter_map(|descriptor| descriptor.store).collect::<Vec<_>>();
    if !matches!(
        operation,
        PreparedOperation::Gpu(
            PreparedGpuOperation::MatrixMulAccumulate | PreparedGpuOperation::Tensor
        )
    ) {
        input_stores.sort_unstable();
        input_stores.dedup();
    }
    output_stores.sort_unstable();
    output_stores.dedup();
    let mut owners = stream
        .into_iter()
        .flat_map(|stream| stream.placements.iter())
        .map(|placement| PreparedOwnerKey {
            matrix_id: stores[placement.store].matrix_id,
            instance: stream.map_or(0, |stream| stream.instance),
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
            for leaf in family_leaf_wires(&program, wire) {
                if matches!(
                    program.wire_types.get(&leaf),
                    Some(
                        ConcreteWireType::Int |
                            ConcreteWireType::Bool |
                            ConcreteWireType::Real |
                            ConcreteWireType::ConstantInt |
                            ConcreteWireType::ConstantBool |
                            ConcreteWireType::ConstantReal
                    )
                ) && program.scalar_slots.contains_key(&leaf)
                {
                    device_scalars.insert(leaf);
                }
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
                .flat_map(|wire| family_leaf_wires(&program, *wire))
                .any(|wire| program.device_scalar_wires.contains(&wire))
            {
                let propagated = arguments
                    .iter()
                    .chain(outputs.iter())
                    .flat_map(|wire| family_leaf_wires(&program, *wire))
                    .filter(|wire| scalar(wire) && program.scalar_slots.contains_key(wire))
                    .collect::<Vec<_>>();
                for wire in propagated {
                    program.device_scalar_wires.insert(wire);
                }
            }
        }
        if program.device_scalar_wires.len() == previous {
            break;
        }
    }
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
        let mut program = GpuPreparation { instance_count: 1, ..GpuPreparation::default() };
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
        finalize_fixture(&mut program, &[2, 3, 7]);
        for device in [2, 3, 7] {
            let context = program
                .finalized_matrices
                .ordinary_for_instance(source, device, 0)
                .expect("threshold source identity")
                .physical
                .context_identity;
            program
                .finalized_matrices
                .insert(
                    MatrixSite::HostStaging { node: node_id, instance: 0, device },
                    ValueLocation {
                        owner: 71,
                        rows: 0..1,
                        columns: 0..1,
                        level: 0,
                        format: PreparedFormat::Coefficient,
                        device,
                    },
                    context,
                    1,
                    1,
                )
                .unwrap();
        }
        program
    }

    fn scalar_pack_program() -> GpuPreparation {
        let unrelated = WireRef { node: NodeId(1), port: Port(0) };
        let output = WireRef { node: NodeId(2), port: Port(0) };
        let scalar = WireRef { node: NodeId(3), port: Port(0) };
        let node_id = 4;
        let matrix = ConcreteMatrixType::scalar(BigInt::from(17), 8);
        let mut program = GpuPreparation { instance_count: 2, ..GpuPreparation::default() };
        // Put a separate matrix owner first so the test catches code that
        // replaces ScalarPack's output owner with the first canonical store.
        for (wire, owner) in [(unrelated, 70), (output, 80)] {
            program.values.insert(
                wire,
                ValueLocation {
                    owner,
                    rows: 0..1,
                    columns: 0..1,
                    level: 0,
                    format: PreparedFormat::Evaluation,
                    device: 0,
                },
            );
            program.wire_types.insert(wire, ConcreteWireType::Matrix(matrix.clone()));
        }
        program.wire_types.insert(scalar, ConcreteWireType::Int);
        program
            .node_bindings
            .insert(node_id, (vec![scalar].into_boxed_slice(), vec![output].into_boxed_slice()));
        program.topology.nodes = vec![PreparedTopologyNode {
            id: node_id,
            command: PreparedCommandRequirement {
                operation: PreparedOperation::Gpu(PreparedGpuOperation::PackPolynomialCoefficients),
                inputs: 1,
                outputs: 1,
            },
            stream: 0,
            waits: Box::new([]),
            completion: 5,
        }]
        .into_boxed_slice();
        finalize_fixture(&mut program, &[0, 2, 7]);
        program
    }

    /// Explicit finalized fixture used by structural resource-plan tests.
    /// Production never derives a context or owner from a missing table.
    fn finalize_fixture(program: &mut GpuPreparation, devices: &[i32]) {
        if program.instance_count == 0 {
            program.instance_count = 1;
        }
        let values = program.values.clone();
        let mut capacities = BTreeMap::<u64, (usize, usize)>::new();
        for (owner, capacity) in &program.storage_capacities {
            capacities.insert(*owner, *capacity);
        }
        for location in values.values() {
            capacities
                .entry(location.owner)
                .and_modify(|capacity| {
                    capacity.0 = capacity.0.max(location.rows.end);
                    capacity.1 = capacity.1.max(location.columns.end);
                })
                .or_insert((location.rows.end, location.columns.end));
        }
        for (wire, location) in values {
            for device in devices {
                let mut physical = location.clone();
                physical.device = *device;
                let (capacity_rows, capacity_columns) = capacities[&location.owner];
                for instance in 0..program.instance_count {
                    program
                        .finalized_matrices
                        .insert(
                            MatrixSite::Ordinary { wire, device: *device, instance },
                            physical.clone(),
                            100 + location.level * 2 +
                                usize::from(location.format == PreparedFormat::Evaluation),
                            capacity_rows,
                            capacity_columns,
                        )
                        .unwrap();
                }
            }
        }
    }

    #[test]
    fn finalized_matrix_ids_tag_each_semantic_site() {
        let wire = WireRef { node: NodeId(70), port: Port(0) };
        let location = ValueLocation {
            owner: 9,
            rows: 0..1,
            columns: 0..1,
            level: 0,
            format: PreparedFormat::Coefficient,
            device: 2,
        };
        let mut table = FinalizedMatrixIdTable::default();
        let ordinary = table
            .insert(
                MatrixSite::Ordinary { wire, device: 2, instance: 0 },
                location.clone(),
                11,
                1,
                1,
            )
            .unwrap();
        let variant = table
            .insert(
                MatrixSite::Variant { node: 70, variant: 0, port: 0, device: 2, instance: 0 },
                location.clone(),
                11,
                1,
                1,
            )
            .unwrap();
        let staging = table
            .insert(
                MatrixSite::HostStaging { node: 70, instance: 0, device: 2 },
                location,
                11,
                1,
                1,
            )
            .unwrap();
        assert_ne!(ordinary, variant);
        assert_ne!(variant, staging);
        assert_eq!(
            table.ordinary_for_instance(wire, 2, 0).unwrap().site,
            MatrixSite::Ordinary { wire, device: 2, instance: 0 }
        );
        assert_eq!(
            table.variant(70, 0, 0, 2).unwrap().site,
            MatrixSite::Variant { node: 70, variant: 0, port: 0, device: 2, instance: 0 }
        );
        assert_eq!(table.host_staging(70, 0, 2).unwrap().physical.context_identity, 11);
    }

    #[test]
    fn finalized_matrix_id_rejects_conflicting_site_identity() {
        let wire = WireRef { node: NodeId(71), port: Port(0) };
        let site = MatrixSite::Ordinary { wire, device: 7, instance: 0 };
        let mut table = FinalizedMatrixIdTable::default();
        let first = table
            .insert(
                site,
                ValueLocation {
                    owner: 1,
                    rows: 0..1,
                    columns: 0..1,
                    level: 0,
                    format: PreparedFormat::Evaluation,
                    device: 7,
                },
                31,
                1,
                1,
            )
            .unwrap();
        let duplicate = table
            .insert(
                site,
                ValueLocation {
                    owner: 1,
                    rows: 0..1,
                    columns: 0..1,
                    level: 0,
                    format: PreparedFormat::Evaluation,
                    device: 7,
                },
                31,
                1,
                1,
            )
            .unwrap();
        assert_eq!(duplicate, first);
        let second = table.insert(
            site,
            ValueLocation {
                owner: 2,
                rows: 0..1,
                columns: 0..1,
                level: 1,
                format: PreparedFormat::Coefficient,
                device: 7,
            },
            32,
            2,
            2,
        );
        assert!(second.is_err());
        let identity = table.identity(first).unwrap();
        assert_eq!(identity.physical.owner, 1);
        assert_eq!(identity.physical.context_identity, 31);
    }

    #[test]
    fn physical_plan_merge_remaps_id_collisions_and_rejects_site_conflicts() {
        let location = ValueLocation {
            owner: 1,
            rows: 0..1,
            columns: 0..1,
            level: 0,
            format: PreparedFormat::Coefficient,
            device: 0,
        };
        let mut first_table = FinalizedMatrixIdTable::default();
        let first_wire = WireRef { node: NodeId(81), port: Port(0) };
        first_table
            .insert(
                MatrixSite::Ordinary { wire: first_wire, device: 0, instance: 0 },
                location.clone(),
                7,
                1,
                1,
            )
            .unwrap();
        let mut second_table = FinalizedMatrixIdTable::default();
        let second_wire = WireRef { node: NodeId(82), port: Port(0) };
        second_table
            .insert(
                MatrixSite::Ordinary { wire: second_wire, device: 0, instance: 0 },
                ValueLocation { owner: 2, ..location.clone() },
                8,
                1,
                1,
            )
            .unwrap();
        let mut conflict_table = FinalizedMatrixIdTable::default();
        conflict_table
            .insert(
                MatrixSite::Ordinary { wire: first_wire, device: 0, instance: 0 },
                ValueLocation { owner: 9, ..location.clone() },
                7,
                1,
                1,
            )
            .unwrap();
        let empty = |finalized_matrices| PreparedResourcePlan {
            instance_count: 1,
            finalized_matrices,
            stores: Box::new([]),
            scalar_buffers: Box::new([]),
            commands: Box::new([]),
            schedules: Box::new([]),
        };
        let merged = PreparedResourcePlan::merge_physical_plans(vec![
            empty(first_table),
            empty(second_table),
        ])
        .unwrap();
        assert_eq!(merged.finalized_matrices.identities.len(), 2);
        assert_ne!(
            merged.finalized_matrices.id(MatrixSite::Ordinary {
                wire: first_wire,
                device: 0,
                instance: 0,
            }),
            merged.finalized_matrices.id(MatrixSite::Ordinary {
                wire: second_wire,
                device: 0,
                instance: 0,
            }),
        );
        assert!(
            PreparedResourcePlan::merge_physical_plans(vec![
                empty(merged.finalized_matrices.clone()),
                empty(conflict_table),
            ])
            .is_err()
        );
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
            context_identity: 0,
            instance: 0,
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
                store: None,
                layout: Some(layout),
                owner_layout: None,
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
        let trapdoor_owner_count = trapdoor.len() - TrapdoorStage::ALL.len();
        assert_eq!(
            trapdoor[..trapdoor_owner_count].iter().map(|entry| entry.owner).collect::<Vec<_>>(),
            (0..trapdoor_owner_count).map(Some).collect::<Vec<_>>()
        );
        assert_eq!(
            trapdoor[trapdoor_owner_count..].iter().map(|entry| entry.stage).collect::<Vec<_>>(),
            TrapdoorStage::ALL.into_iter().map(Some).collect::<Vec<_>>()
        );
        assert_eq!(
            trapdoor[trapdoor_owner_count..].iter().map(|entry| entry.kind).collect::<Vec<_>>(),
            TrapdoorStage::ALL.into_iter().map(|stage| stage.kind()).collect::<Vec<_>>()
        );
        assert_eq!(
            mxx_primitives::sampler::trapdoor::gpu::TRAPDOOR_STAGE_KINDS.to_vec(),
            TrapdoorStage::ALL.into_iter().map(|stage| stage.kind()).collect::<Vec<_>>()
        );
        let stage_entries = &trapdoor[trapdoor_owner_count..];
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
                store: None,
                layout: None,
                owner_layout: None,
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
                store: None,
                layout: None,
                owner_layout: None,
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
        let binding = PreparedBindingId { matrix_id: FinalizedMatrixId(0), instance: 1 };
        assert_eq!(binding.instance, 1);
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
            PreparedViewShape::Alias(_)
        ));
        let sliced = NodeKind::Slice {
            rows: Some(mxx_ir_core::node::IndexRange {
                start: IntExpr::constant(1),
                end: IntExpr::constant(3),
            }),
            columns: None,
        };
        let PreparedViewShape::Alias(sliced) =
            lower_slice(&sliced, &ParamEnv::default(), source.clone(), location(1, 0..2, 0..4))
                .unwrap()
        else {
            panic!("slice must be an alias");
        };
        assert_eq!(sliced.rows, 1..3);
        assert!(matches!(
            lower_slice(&slice, &ParamEnv::default(), source.clone(), location(9, 0..4, 0..4))
                .unwrap(),
            PreparedViewShape::FixedCopies(copies) if copies.len() == 1
        ));
        assert!(matches!(
            lower_transpose(&NodeKind::Transpose, source.clone(), location(1, 0..4, 0..4)).unwrap(),
            PreparedViewShape::TransposeAlias(_)
        ));
        assert!(matches!(
            lower_transpose(&NodeKind::Transpose, source.clone(), location(9, 0..4, 0..4))
                .unwrap(),
            PreparedViewShape::FixedCopies(copies) if copies.len() == 1
        ));
        let other = location(2, 0..4, 0..4);
        let concat = NodeKind::Concat { axis: ConcatAxis::Diagonal };
        assert!(matches!(
            lower_concat(&concat, &[source, other], location(3, 0..8, 0..8)).unwrap(),
            PreparedViewShape::FixedCopies(_)
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
            PreparedViewShape::Alias(_)
        ));
        assert!(matches!(
            lower_concat(
                &NodeKind::Concat { axis: ConcatAxis::Columns },
                &[source, same_owner],
                location(9, 0..4, 0..8)
            )
            .unwrap(),
            PreparedViewShape::FixedCopies(copies) if copies.len() == 2
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
    fn host_only_scalar_selections_use_control_recipes_without_matrix_placement() {
        let selector = WireRef { node: NodeId(1), port: Port(0) };
        let output = WireRef { node: NodeId(2), port: Port(0) };
        let static_output = WireRef { node: NodeId(3), port: Port(0) };
        let select_output = WireRef { node: NodeId(4), port: Port(0) };
        let mut program = GpuPreparation { instance_count: 1, ..GpuPreparation::default() };
        for (wire, slot) in [(selector, 0), (output, 1), (static_output, 2), (select_output, 3)] {
            program.scalar_slots.insert(wire, slot);
            program.wire_types.insert(wire, ConcreteWireType::Int);
        }
        program
            .node_bindings
            .insert(2, (vec![selector].into_boxed_slice(), vec![output].into_boxed_slice()));
        program.node_bindings.insert(3, (Box::new([]), vec![static_output].into_boxed_slice()));
        program
            .node_bindings
            .insert(4, (vec![selector].into_boxed_slice(), vec![select_output].into_boxed_slice()));
        program.selection_commands.insert(
            2,
            PreparedSelection::ScalarDynamic {
                candidates: vec![1, 2].into_boxed_slice(),
                selector,
            },
        );
        program.selection_commands.insert(3, PreparedSelection::ScalarStatic { slot: 1 });
        program.selection_commands.insert(
            4,
            PreparedSelection::ScalarSelect { candidates: vec![1, 2].into_boxed_slice(), selector },
        );
        program.topology.nodes = vec![2, 3, 4]
            .into_iter()
            .map(|id| PreparedTopologyNode {
                id,
                command: PreparedCommandRequirement {
                    operation: PreparedOperation::Selection,
                    inputs: 1,
                    outputs: 1,
                },
                stream: 0,
                waits: Box::new([]),
                completion: id + 10,
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let plan = PreparedResourcePlan::from_preparation(&program).unwrap();
        assert_eq!(plan.stores.len(), 0);
        assert_eq!(plan.commands.len(), 3);
        assert!(plan.commands.iter().all(|command| {
            command.recipe.stage == PreparedNativeStage::Control &&
                command.recipe.scalar.is_none() &&
                command.recipe.owners.is_empty()
        }));
        plan.validate_for_warmup().unwrap();

        let expanded = PreparedResourcePlan::from_preparation_for_devices(&program, &[2, 7])
            .expect("ownerless controls merge across devices");
        assert_eq!(expanded.commands.len(), 3);
        assert!(expanded.commands.iter().all(|command| {
            command.recipe.stage == PreparedNativeStage::Control && command.recipe.owners.is_empty()
        }));
    }

    #[test]
    fn schedule_stream_identity_rejects_role_origin_and_pool_collisions() {
        let key = PreparedResourceKey {
            execution_owner_identity: 7,
            context_identity: 11,
            instance: 0,
            partition: 0,
            device: 3,
            limb_x: 0,
            limb_y: 1,
            role: PreparedStageRole::Matrix as i32,
        };
        let base = PreparedStreamFootprint { key, origin: 0, pool_slot: 4 };
        let mut different_pool = base;
        different_pool.pool_slot += 1;
        let mut different_origin = base;
        different_origin.origin = 1;
        let mut different_role = base;
        different_role.key.role = PreparedStageRole::Schedule as i32;
        assert!(same_prepared_stream_physical(&base, &base));
        assert!(!same_prepared_stream_physical(&base, &different_pool));
        assert!(!same_prepared_stream_physical(&base, &different_origin));
        assert!(!same_prepared_stream_physical(&base, &different_role));
    }

    #[test]
    fn schedule_member_aliases_canonicalize_but_different_stores_reject() {
        let key = PreparedResourceKey {
            execution_owner_identity: 7,
            context_identity: 11,
            instance: 0,
            partition: 0,
            device: 3,
            limb_x: 0,
            limb_y: 1,
            role: PreparedStageRole::Matrix as i32,
        };
        let stream = PreparedStreamFootprint { key, origin: 0, pool_slot: 4 };
        let candidate = |member, store| PreparedScheduleMemberCandidate {
            member,
            stream,
            store,
            owner: 19,
            context_identity: 11,
            device: 3,
            instance: 0,
            level: 2,
            format: PreparedFormat::Evaluation,
        };
        assert_eq!(
            unique_prepared_schedule_member(&[candidate(4, 9), candidate(2, 9)], &stream).unwrap(),
            2
        );
        assert!(
            unique_prepared_schedule_member(&[candidate(2, 9), candidate(4, 10)], &stream).is_err()
        );
    }

    #[test]
    fn resource_keys_require_paired_host_sentinels() {
        let mut key = PreparedResourceKey {
            execution_owner_identity: 0,
            context_identity: 0,
            instance: 0,
            partition: -1,
            device: -1,
            limb_x: 0,
            limb_y: 0,
            role: PreparedStageRole::Transfer as i32,
        };
        assert!(validate_prepared_resource_key(&key, "test").is_ok());
        key.partition = 0;
        assert!(validate_prepared_resource_key(&key, "test").is_err());
        key.device = 0;
        assert!(validate_prepared_resource_key(&key, "test").is_ok());
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

        finalize_fixture(&mut program, &[2]);
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
    fn finalized_resource_plan_rejects_missing_crt_context_identity() {
        let wire = WireRef { node: NodeId(41), port: Port(0) };
        let mut program = GpuPreparation { instance_count: 1, ..GpuPreparation::default() };
        program.values.insert(
            wire,
            ValueLocation {
                owner: 7,
                rows: 0..1,
                columns: 0..1,
                level: 0,
                format: PreparedFormat::Coefficient,
                device: 3,
            },
        );
        assert!(matches!(
            PreparedResourcePlan::from_preparation(&program),
            Err(PreparedLoweringError::InvalidContract("finalized CRT context is missing"))
        ));
    }

    #[test]
    fn physical_expansion_accepts_aliases_with_shared_physical_identity() {
        let first = WireRef { node: NodeId(43), port: Port(0) };
        let second = WireRef { node: NodeId(44), port: Port(0) };
        let location = ValueLocation {
            owner: 7,
            rows: 0..2,
            columns: 0..2,
            level: 0,
            format: PreparedFormat::Coefficient,
            device: 0,
        };
        let mut program = GpuPreparation { instance_count: 1, ..GpuPreparation::default() };
        program.values.insert(first, location.clone());
        program.values.insert(second, location.clone());
        for wire in [first, second] {
            program
                .finalized_matrices
                .insert(
                    MatrixSite::Ordinary { wire, device: 2, instance: 0 },
                    ValueLocation { owner: 17, device: 2, ..location.clone() },
                    19,
                    2,
                    2,
                )
                .unwrap();
        }
        program.family_members.insert(first, vec![location.clone()].into_boxed_slice());
        program.family_wires.insert(first, vec![first].into_boxed_slice());
        program.selection_commands.insert(10, PreparedSelection::Static { location });
        program.selection_candidate_wires.insert(10, vec![second].into_boxed_slice());

        let plan = PreparedResourcePlan::from_preparation_for_devices(&program, &[2]).unwrap();
        assert_eq!(plan.stores.len(), 1);
        let physical =
            &plan.finalized_matrices.identity(plan.stores[0].matrix_id).unwrap().physical;
        assert_eq!(physical.owner, 17);
        assert_eq!(physical.device, 2);
    }

    #[test]
    fn resource_plan_keeps_same_owner_states_distinct_across_devices_and_instances() {
        let evaluation = WireRef { node: NodeId(1), port: Port(0) };
        let coefficient = WireRef { node: NodeId(2), port: Port(0) };
        let output = WireRef { node: NodeId(3), port: Port(0) };
        let matrix = ConcreteWireType::Matrix(ConcreteMatrixType::scalar(BigInt::from(17), 8));
        let mut program = GpuPreparation { instance_count: 2, ..GpuPreparation::default() };
        for (wire, level, format) in [
            (evaluation, 1, PreparedFormat::Evaluation),
            (coefficient, 0, PreparedFormat::Coefficient),
        ] {
            program.values.insert(
                wire,
                ValueLocation { owner: 7, rows: 0..1, columns: 0..2, level, format, device: 0 },
            );
            program.wire_types.insert(wire, matrix.clone());
            for device in [2, 7] {
                for instance in 0..2 {
                    program
                        .finalized_matrices
                        .insert(
                            MatrixSite::Ordinary { wire, device, instance },
                            ValueLocation {
                                owner: 7,
                                device,
                                level,
                                format,
                                ..program.values[&wire].clone()
                            },
                            19 + level,
                            3,
                            4,
                        )
                        .unwrap();
                }
            }
        }
        program.values.insert(
            output,
            ValueLocation {
                owner: 7,
                rows: 0..1,
                columns: 0..2,
                level: 1,
                format: PreparedFormat::Evaluation,
                device: 0,
            },
        );
        program.wire_types.insert(output, matrix);
        for device in [2, 7] {
            for instance in 0..2 {
                program
                    .finalized_matrices
                    .insert(
                        MatrixSite::Ordinary { wire: output, device, instance },
                        ValueLocation {
                            owner: 7,
                            device,
                            level: 1,
                            format: PreparedFormat::Evaluation,
                            ..program.values[&output].clone()
                        },
                        20,
                        3,
                        4,
                    )
                    .unwrap();
            }
        }
        program.storage_capacities.insert(7, (3, 4));
        program.node_bindings.insert(
            4,
            (vec![evaluation, coefficient].into_boxed_slice(), vec![output].into_boxed_slice()),
        );
        program.topology.nodes = vec![PreparedTopologyNode {
            id: 4,
            command: PreparedCommandRequirement {
                operation: PreparedOperation::Gpu(PreparedGpuOperation::MatrixBinary(
                    MatrixBinaryOp::Add,
                )),
                inputs: 2,
                outputs: 1,
            },
            stream: 0,
            waits: Box::new([]),
            completion: 5,
        }]
        .into_boxed_slice();

        let plan = PreparedResourcePlan::from_preparation_for_devices(&program, &[2, 7]).unwrap();
        assert_eq!(plan.stores.len(), 2 * 2 * 2);
        let states = plan
            .stores
            .iter()
            .map(|store| {
                let physical = &plan.finalized_matrices.identity(store.matrix_id).unwrap().physical;
                (store.instance, physical.device, physical.level, physical.format)
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(states.len(), plan.stores.len());
        assert_eq!(
            states,
            BTreeSet::from([
                (0, 2, 0, PreparedFormat::Coefficient),
                (0, 2, 1, PreparedFormat::Evaluation),
                (1, 2, 0, PreparedFormat::Coefficient),
                (1, 2, 1, PreparedFormat::Evaluation),
                (0, 7, 0, PreparedFormat::Coefficient),
                (0, 7, 1, PreparedFormat::Evaluation),
                (1, 7, 0, PreparedFormat::Coefficient),
                (1, 7, 1, PreparedFormat::Evaluation),
            ])
        );
        assert_eq!(plan.commands.len(), 4);
        assert!(plan.commands.iter().all(|command| {
            command.stream.as_ref().is_some_and(|stream| stream.placements.len() == 2)
        }));
        plan.validate_for_warmup().unwrap();
    }

    #[test]
    fn resource_plan_includes_finite_variant_output_context_stores() {
        let output = WireRef { node: NodeId(10), port: Port(0) };
        let matrix_a = ConcreteWireType::Matrix(ConcreteMatrixType::scalar(BigInt::from(17), 8));
        let matrix_b = ConcreteWireType::Matrix(ConcreteMatrixType::scalar(BigInt::from(19), 8));
        let mut program = GpuPreparation { instance_count: 1, ..GpuPreparation::default() };
        program.values.insert(
            output,
            ValueLocation {
                owner: 3,
                rows: 0..1,
                columns: 0..1,
                level: 1,
                format: PreparedFormat::Evaluation,
                device: 0,
            },
        );
        program.wire_types.insert(output, matrix_a.clone());
        for device in [2, 7] {
            program
                .finalized_matrices
                .insert(
                    MatrixSite::Ordinary { wire: output, device, instance: 0 },
                    ValueLocation {
                        owner: 3,
                        rows: 0..1,
                        columns: 0..1,
                        level: 1,
                        format: PreparedFormat::Evaluation,
                        device,
                    },
                    20,
                    1,
                    1,
                )
                .unwrap();
        }
        program.node_bindings.insert(10, (Box::new([]), vec![output].into_boxed_slice()));
        program.node_sources.insert(
            10,
            PreparedNodeSource {
                kind: NodeKind::MatrixNegate,
                environment: ParamEnv::default(),
                variants: Box::new([]),
                variant_indices: Box::new([]),
                variant_input_types: Box::new([]),
                variant_output_types: vec![
                    vec![matrix_a].into_boxed_slice(),
                    vec![matrix_b].into_boxed_slice(),
                ]
                .into_boxed_slice(),
            },
        );
        // Both variants use the same CRT level but distinct concrete moduli;
        // finalization therefore gives them separate owners even though their
        // logical output wire is shared.
        for (variant, (owner, level)) in [(0, (3, 1)), (1, (4, 1))] {
            for device in [2, 7] {
                program
                    .finalized_matrices
                    .insert(
                        MatrixSite::Variant { node: 10, variant, port: 0, device, instance: 0 },
                        ValueLocation {
                            owner,
                            rows: 0..1,
                            columns: 0..1,
                            level,
                            format: PreparedFormat::Evaluation,
                            device,
                        },
                        19 + variant,
                        1,
                        1,
                    )
                    .unwrap();
            }
        }
        let plan = PreparedResourcePlan::from_preparation_for_devices(&program, &[2, 7]).unwrap();
        assert_eq!(plan.stores.len(), 6);
        for device in [2, 7] {
            assert!(plan.stores.iter().any(|store| {
                let physical = &plan.finalized_matrices.identity(store.matrix_id).unwrap().physical;
                physical.device == device && physical.owner == 3 && physical.level == 1
            }));
            assert!(plan.stores.iter().any(|store| {
                let physical = &plan.finalized_matrices.identity(store.matrix_id).unwrap().physical;
                physical.device == device && physical.owner == 4 && physical.level == 1
            }));
        }
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
        finalize_fixture(&mut program, &[3]);
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
        finalize_fixture(&mut program, &[4]);
        let plan = PreparedResourcePlan::from_preparation(&program).unwrap();
        assert_eq!(plan.instance_count, 2);
        assert_eq!(plan.stores.len(), 2);
        let physical =
            &plan.finalized_matrices.identity(plan.stores[0].matrix_id).unwrap().physical;
        assert_eq!(physical.capacity_rows, 9);
        assert_eq!(physical.capacity_columns, 11);
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
            .find(|store| {
                plan.finalized_matrices.identity(store.matrix_id).unwrap().physical.owner != 70
            })
            .expect("threshold staging store");
        let staging_identity = plan.finalized_matrices.identity(staging.matrix_id).unwrap();
        assert_eq!(staging.logical_rows, 1);
        assert_eq!(staging.logical_columns, 1);
        assert_eq!(staging_identity.physical.format, PreparedFormat::Coefficient);
        assert_eq!(staging_identity.physical.device, 3);
        let command = &plan.commands[0];
        let staging_owner = command.recipe.matrix_staging.expect("threshold staging owner");
        assert_eq!(staging_owner.matrix_id, staging.matrix_id);
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
            .filter(|(_, store)| {
                plan.finalized_matrices.identity(store.matrix_id).unwrap().physical.format ==
                    PreparedFormat::Coefficient
            })
            .collect::<Vec<_>>();
        assert_eq!(staging_stores.len(), 2);
        assert_eq!(
            staging_stores
                .iter()
                .map(|(_, store)| plan
                    .finalized_matrices
                    .identity(store.matrix_id)
                    .unwrap()
                    .physical
                    .device)
                .collect::<Vec<_>>(),
            [2, 7]
        );

        for device in [2, 7] {
            let [(staging_index, staging)] = staging_stores
                .iter()
                .filter(|(_, store)| {
                    plan.finalized_matrices.identity(store.matrix_id).unwrap().physical.device ==
                        device
                })
                .copied()
                .collect::<Vec<_>>()[..]
            else {
                panic!("expected exactly one threshold staging store on device {device}");
            };
            let [command] = plan
                .commands
                .iter()
                .filter(|command| {
                    command.recipe.matrix_staging.is_some_and(|owner| {
                        plan.finalized_matrices
                            .identity(owner.matrix_id)
                            .is_some_and(|identity| identity.physical.device == device)
                    })
                })
                .collect::<Vec<_>>()[..]
            else {
                panic!("expected exactly one threshold command on device {device}");
            };
            let staging_owner = command.recipe.matrix_staging.unwrap();
            assert_eq!(staging_owner.matrix_id, staging.matrix_id);
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
                [&PreparedPlacementKey { store: staging_index, matrix_id: staging.matrix_id }]
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
            .find(|index| {
                valid
                    .finalized_matrices
                    .identity(valid.stores[*index].matrix_id)
                    .unwrap()
                    .physical
                    .format ==
                    PreparedFormat::Coefficient
            })
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
    fn scalar_pack_keeps_its_output_stream_owner() {
        let plan = PreparedResourcePlan::from_preparation(&scalar_pack_program()).unwrap();
        let command = &plan.commands[0];
        let output_store = &plan.stores[command.outputs[0].store.unwrap()];
        assert_eq!(command.recipe.stage, PreparedNativeStage::ScalarPack);
        assert_eq!(
            command.recipe.owners.as_ref(),
            [PreparedOwnerKey {
                matrix_id: plan.stores[command.outputs[0].store.unwrap()].matrix_id,
                instance: output_store.instance,
            }]
        );
        assert_eq!(command.recipe.owners[0].matrix_id, output_store.matrix_id);
        assert_eq!(
            command.stream.as_ref().unwrap().placements[0].matrix_id,
            output_store.matrix_id
        );
    }

    #[test]
    fn scalar_pack_schedule_groups_match_one_command_per_instance_and_device() {
        let plan =
            PreparedResourcePlan::from_preparation_for_devices(&scalar_pack_program(), &[2, 7])
                .unwrap();
        assert_eq!(plan.instance_count, 2);
        assert_eq!(plan.schedules.len(), 4);
        for schedule in &plan.schedules {
            let members = schedule
                .command_group
                .iter()
                .filter_map(|node| {
                    plan.commands
                        .iter()
                        .find(|command| schedule_member_matches(schedule, command, *node))
                })
                .collect::<Vec<_>>();
            assert!(!members.is_empty(), "schedule group has no matching command");
            assert_eq!(members.len(), schedule.command_group.len());
            assert_eq!(members[0].recipe.stage, PreparedNativeStage::ScalarPack);
            assert_eq!(
                members[0].recipe.owners[0].matrix_id,
                schedule.stream.placements[0].matrix_id
            );
        }
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

        finalize_fixture(&mut program, &[2, 7]);
        let plan = PreparedResourcePlan::from_preparation_for_devices(&program, &[2, 7]).unwrap();
        assert_eq!(plan.stores.len(), 2);
        assert_eq!(plan.commands.len(), 2);
        assert_eq!(plan.schedules.len(), 2);
        assert_eq!(
            plan.stores
                .iter()
                .map(|store| plan
                    .finalized_matrices
                    .identity(store.matrix_id)
                    .unwrap()
                    .physical
                    .device)
                .collect::<Vec<_>>(),
            [2, 7]
        );
        assert_eq!(
            plan.commands
                .iter()
                .map(|command| {
                    plan.finalized_matrices
                        .identity(command.stream.as_ref().unwrap().placements[0].matrix_id)
                        .unwrap()
                        .physical
                        .device
                })
                .collect::<Vec<_>>(),
            [2, 7]
        );
        assert_ne!(
            plan.commands[0].stream.as_ref().unwrap().stores[0],
            plan.commands[1].stream.as_ref().unwrap().stores[0]
        );
        let schedule_devices = plan
            .schedules
            .iter()
            .map(|schedule| {
                plan.finalized_matrices
                    .identity(schedule.stream.placements[0].matrix_id)
                    .unwrap()
                    .physical
                    .device
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(schedule_devices, BTreeSet::from([2, 7]));
        assert_ne!(
            plan.schedules[0].stream.placements[0].matrix_id,
            plan.schedules[1].stream.placements[0].matrix_id
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
                    matrix_id: FinalizedMatrixId(if device == 2 { 0 } else { 1 }),
                    instance,
                }]
                .into_boxed_slice(),
                completion: Some(51),
                matrix_staging: None,
                scalar: None,
                replay_upload: None,
            },
        };
        let commands = [command(2, 0), command(7, 0), command(2, 1), command(7, 1)];
        let mut finalized_matrices = FinalizedMatrixIdTable::default();
        for (id, device) in [(0, 2), (1, 7)] {
            finalized_matrices
                .insert(
                    MatrixSite::Ordinary {
                        wire: WireRef { node: NodeId(50), port: Port(0) },
                        device,
                        instance: 0,
                    },
                    ValueLocation {
                        owner: 80,
                        rows: 0..1,
                        columns: 0..1,
                        level: 0,
                        format: PreparedFormat::Evaluation,
                        device,
                    },
                    1,
                    1,
                    1,
                )
                .unwrap();
            assert_eq!(finalized_matrices.identities.get(&FinalizedMatrixId(id)).is_some(), true);
        }
        assert!(physical_command_matches(&finalized_matrices, &commands[0], 50, 0, 2));
        assert!(!physical_command_matches(&finalized_matrices, &commands[0], 50, 0, 7));
        assert!(physical_command_matches(&finalized_matrices, &commands[1], 50, 0, 7));
        assert!(!physical_command_matches(&finalized_matrices, &commands[0], 50, 1, 2));
        assert!(physical_command_matches(&finalized_matrices, &commands[2], 50, 1, 2));
        assert!(physical_command_matches(&finalized_matrices, &commands[3], 50, 1, 7));
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
        finalize_fixture(&mut program, &[8]);
        let plan = PreparedResourcePlan::from_preparation(&program).unwrap();
        let stream = plan.commands[0].stream.as_ref().unwrap();
        assert_eq!(stream.stores.len(), 1);
        assert_eq!(
            plan.finalized_matrices
                .identity(stream.placements[0].matrix_id)
                .unwrap()
                .physical
                .device,
            8
        );
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
        finalize_fixture(&mut program, &[5]);
        let plan = PreparedResourcePlan::from_preparation(&program).unwrap();
        assert_eq!(plan.stores.len(), 1);
        assert_eq!(plan.commands[0].inputs[0].store, Some(0));
        assert_eq!(plan.commands[0].inputs[1].store, Some(0));
        assert_eq!(plan.commands[0].outputs[0].store, Some(0));
        assert_eq!(plan.stores[0].logical_rows, 2);
        assert_eq!(plan.stores[0].logical_columns, 3);
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
