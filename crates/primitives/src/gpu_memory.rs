//! Operation-specific GPU memory evidence.
//!
//! The runtime planner must not infer an operation's footprint from the
//! allocation of its output matrix.  In particular, a zero scratch value is
//! meaningful only for a native operation whose implementation has been
//! audited to allocate no additional device owner.  This module is the
//! primitive-side contract for those queries.  Callers can either obtain an
//! exact/certified footprint or receive an explicit `Unsupported` result;
//! there is deliberately no implicit "matrix bytes + scratch = 0" fallback.

use crate::{
    matrix::gpu_dcrt_poly::{GpuDCRTPolyMatrix, GpuSmallMatrix, GpuSmallMatrixAllocationReport},
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, GpuMatrixAllocationBytes},
    },
    sampler::trapdoor::gpu::{
        PreimageAllocationEvidence, PreimageAllocationEvidenceKind, TrapdoorAllocationEvidence,
    },
};
use num_bigint::BigUint;
use num_traits::One;
use std::{fmt, ops::Range};

/// Operation families understood by the native memory-query contract.
///
/// The enum is intentionally exhaustive over the GPU operation families used
/// by runtime warmup.  New native entry points must add a variant here and
/// choose an evidence policy in [`query_matrix_operation_memory`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GpuMemoryOperation {
    Transpose,
    Slice,
    Tensor,
    Concat,
    Add,
    Sub,
    Scale,
    Negate,
    MatMul,
    Accumulate,
    SmallRhs,
    Hash,
    UniformSampler,
    GaussianSampler,
    ModulusConversion,
    /// Full DCRT centered rebase.  This is deliberately distinct from
    /// modulus conversion: the native kernel reconstructs the centered
    /// mixed-radix representative and may extend the destination basis.
    CenteredRebase,
    BlockModSwitch,
    RnsConversion,
    RingAutomorphism,
    CrtRecompose,
    Decompose,
    Constant,
    Lift,
    Trapdoor,
    Preimage,
    FusedRowSum,
    FusedTensorRowSum,
    FusedRowBlockAdd,
    FusedDecompose,
    FusedSmallProduct,
    FusedPreimageBatch,
}

/// Native topology limits shared with `MatrixCrt.cu`'s centered-rebase
/// metadata.  Keep these public so admission tests and production callers
/// cannot silently drift from the CUDA contract.
pub const CENTERED_REBASE_MAX_LIMBS: usize = 64;
pub const CENTERED_REBASE_METADATA_BYTES: usize = 2 * std::mem::size_of::<*const u8>() +
    2 * std::mem::size_of::<usize>() +
    3 * CENTERED_REBASE_MAX_LIMBS * std::mem::size_of::<u64>() +
    CENTERED_REBASE_MAX_LIMBS * std::mem::size_of::<i32>();

/// Native implementation selected by `tensor_sum_rows` for one row grouping.
///
/// The limits are part of the primitive contract, rather than a warmup
/// heuristic.  Keeping this decision here makes the allocator query and the
/// production dispatch use the same topology.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorRowSumImplementation {
    FusedKernel,
    MaterializedTensor,
}

impl TensorRowSumImplementation {
    pub const MAX_GROUPS: usize = 16;
    pub const MAX_TERMS: usize = 32;

    pub fn for_groups(groups: &[Vec<usize>]) -> Self {
        let terms = groups.iter().map(Vec::len).sum::<usize>();
        if groups.len() > Self::MAX_GROUPS || terms > Self::MAX_TERMS {
            Self::MaterializedTensor
        } else {
            Self::FusedKernel
        }
    }

    pub fn is_materialized(self) -> bool {
        matches!(self, Self::MaterializedTensor)
    }
}

impl GpuMemoryOperation {
    /// Canonical registry used by warmup coverage tests and provider
    /// dispatch.  Keeping this list next to the operation enum makes adding a
    /// new primitive a compile-time-visible review point.
    pub const CANONICAL: &'static [Self] = &[
        Self::Transpose,
        Self::Slice,
        Self::Tensor,
        Self::Concat,
        Self::Add,
        Self::Sub,
        Self::Scale,
        Self::Negate,
        Self::MatMul,
        Self::Accumulate,
        Self::SmallRhs,
        Self::Hash,
        Self::UniformSampler,
        Self::GaussianSampler,
        Self::ModulusConversion,
        Self::CenteredRebase,
        Self::BlockModSwitch,
        Self::RnsConversion,
        Self::RingAutomorphism,
        Self::CrtRecompose,
        Self::Decompose,
        Self::Constant,
        Self::Lift,
        Self::Trapdoor,
        Self::Preimage,
        Self::FusedRowSum,
        Self::FusedTensorRowSum,
        Self::FusedRowBlockAdd,
        Self::FusedDecompose,
        Self::FusedSmallProduct,
        Self::FusedPreimageBatch,
    ];
}

impl fmt::Display for GpuMemoryOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Transpose => "transpose",
            Self::Slice => "slice",
            Self::Tensor => "tensor",
            Self::Concat => "concat",
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Scale => "scale",
            Self::Negate => "negate",
            Self::MatMul => "matmul",
            Self::Accumulate => "accumulate",
            Self::SmallRhs => "small_rhs",
            Self::Hash => "hash",
            Self::UniformSampler => "uniform_sampler",
            Self::GaussianSampler => "gaussian_sampler",
            Self::ModulusConversion => "modulus_conversion",
            Self::CenteredRebase => "centered_rebase",
            Self::BlockModSwitch => "block_mod_switch",
            Self::RnsConversion => "rns_conversion",
            Self::RingAutomorphism => "ring_automorphism",
            Self::CrtRecompose => "crt_recompose",
            Self::Decompose => "decompose",
            Self::Constant => "constant",
            Self::Lift => "lift",
            Self::Trapdoor => "trapdoor",
            Self::Preimage => "preimage",
            Self::FusedRowSum => "fused_row_sum",
            Self::FusedTensorRowSum => "fused_tensor_row_sum",
            Self::FusedRowBlockAdd => "fused_row_block_add",
            Self::FusedDecompose => "fused_decompose",
            Self::FusedSmallProduct => "fused_small_product",
            Self::FusedPreimageBatch => "fused_preimage_batch",
        })
    }
}

/// A checked local output range.  Ranges are part of the query key: a full
/// width allocation may not be reused for a narrower dispatch and vice versa.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GpuMemoryRange {
    pub start: usize,
    pub end: usize,
}

/// One physical source owner consumed by the native fused row-block
/// decomposition. `range` is the local owner range represented by this
/// invocation; it is intentionally independent from the output range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuDecomposeBlockShape {
    pub level: usize,
    pub rows: usize,
    pub columns: usize,
    pub is_ntt: bool,
    pub range: GpuMemoryRange,
}

/// Complete native allocation evidence for one
/// `gadget_decompose_row_blocks` call. The native implementation consumes all
/// input owners, performs any NTT/CRT correction in place, and creates one
/// compact output owner. The vectors preserve per-input lifetime evidence so
/// a fleet caller cannot accidentally account only for the first physical
/// block.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FusedDecomposeAllocationEvidence {
    pub operation: GpuMemoryOperation,
    pub blocks: Vec<GpuDecomposeBlockShape>,
    pub input_allocations: Vec<GpuMatrixAllocationBytes>,
    pub per_block_scratch_bytes: Vec<usize>,
    pub per_block_event_bytes: Vec<usize>,
    pub per_block_assembly_bytes: Vec<usize>,
    pub per_block_reuse_bytes: Vec<usize>,
    pub output_allocation: GpuMatrixAllocationBytes,
    pub output_rows: usize,
    pub output_columns: usize,
    pub output_range: GpuMemoryRange,
    pub output_ports: usize,
    pub input_resident_bytes: usize,
    pub output_event_bytes: usize,
    pub scratch_bytes: usize,
    pub control_bytes: usize,
    pub cache_bytes: usize,
    pub assembly_bytes: usize,
    pub transfer_bytes: usize,
    pub host_bytes: usize,
    pub pinned_host_bytes: usize,
    /// Sources are moved into the native call and no input owner is retained
    /// by the compact result. This is distinct from the in-place transforms
    /// performed before the compact kernel launch.
    pub inputs_consumed: bool,
    pub evidence_kind: PreimageAllocationEvidenceKind,
}

impl FusedDecomposeAllocationEvidence {
    pub fn memory_evidence(&self) -> GpuOperationMemoryEvidence {
        let shape = GpuMemoryShape {
            level: self.blocks.first().map_or(0, |block| block.level),
            rows: self.blocks.iter().map(|block| block.rows).sum(),
            columns: self.blocks.first().map_or(0, |block| block.range.width()),
            is_ntt: self.blocks.first().is_some_and(|block| block.is_ntt),
            range: self.blocks.first().map_or(self.output_range, |block| block.range),
        };
        let output_shape = GpuMemoryShape {
            level: shape.level,
            rows: self.output_rows,
            columns: self.output_columns,
            is_ntt: false,
            range: self.output_range,
        };
        let footprint = GpuOperationMemoryFootprint {
            operation: self.operation,
            shape,
            output_shape,
            output_range: self.output_range,
            output: None,
            output_bytes: self.output_allocation.total_bytes,
            input_resident_bytes: self.input_resident_bytes,
            auxiliary_bytes: self.output_event_bytes,
            scratch_bytes: self.scratch_bytes,
            control_bytes: self.control_bytes,
            cache_bytes: self.cache_bytes,
            assembly_bytes: self.assembly_bytes,
            transfer_bytes: self.transfer_bytes,
            host_bytes: self.host_bytes,
            pinned_host_bytes: self.pinned_host_bytes,
        };
        match self.evidence_kind {
            PreimageAllocationEvidenceKind::Exact => GpuOperationMemoryEvidence::Exact(footprint),
            PreimageAllocationEvidenceKind::Certified => {
                GpuOperationMemoryEvidence::Certified(footprint)
            }
        }
    }
}

impl GpuMemoryRange {
    pub fn new(start: usize, end: usize) -> Option<Self> {
        (start < end).then_some(Self { start, end })
    }

    pub fn width(self) -> usize {
        self.end - self.start
    }

    pub fn as_range(self) -> Range<usize> {
        self.start..self.end
    }
}

/// Shape and representation used by an operation-specific query.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GpuMemoryShape {
    pub level: usize,
    pub rows: usize,
    pub columns: usize,
    pub is_ntt: bool,
    pub range: GpuMemoryRange,
}

impl GpuMemoryShape {
    pub fn new(
        level: usize,
        rows: usize,
        columns: usize,
        is_ntt: bool,
        range: GpuMemoryRange,
    ) -> Result<Self, GpuMemoryQueryError> {
        if range.end > columns {
            return Err(GpuMemoryQueryError::InvalidRange);
        }
        Ok(Self { level, rows, columns, is_ntt, range })
    }
}

/// Typed request consumed by the native query dispatcher.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuMatrixMemoryQuery {
    pub operation: GpuMemoryOperation,
    /// Shape of the actual source owner(s), including their representation.
    pub shape: GpuMemoryShape,
    /// Shape of the newly allocated output owner.  This is separate from the
    /// input because transpose, tensor, matmul, and decomposition change
    /// rows, columns, level, or format.
    pub output_shape: GpuMemoryShape,
    /// Exact local output range allocated by this invocation.
    pub output_range: GpuMemoryRange,
    pub output_rows: usize,
    pub output_columns: usize,
    pub output_is_ntt: bool,
    /// Bytes of owners already retained by the caller.  They are carried in
    /// the evidence for admission diagnostics but are never confused with
    /// operation scratch or the new output owner.
    pub input_resident_bytes: usize,
    /// Decomposition-only parameters.  They are ignored by other families.
    pub base_bits: u32,
    pub dropped_moduli: usize,
    /// Number of independent matrices in a batch operation.
    pub batch_count: usize,
    /// Product terms represented by a fused accumulate operation.
    pub product_count: usize,
    /// Number of mixed CRT levels for CRT recomposition.
    pub level_count: usize,
    /// RNS digit grouping width.
    pub digit_size: usize,
    /// Detailed native report required by compact-RHS and preimage families.
    /// Generic matrix allocation bytes are deliberately insufficient for
    /// these operations.
    pub native_evidence: Option<GpuNativeMemoryEvidence>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GpuNativeMemoryEvidence {
    SmallRhs(GpuSmallMatrixAllocationReport),
    Preimage(PreimageAllocationEvidence),
    Trapdoor(TrapdoorAllocationEvidence),
    FusedDecompose(FusedDecomposeAllocationEvidence),
    TensorRowSum(TensorRowSumAllocationEvidence),
}

impl GpuMatrixMemoryQuery {
    pub fn new(
        operation: GpuMemoryOperation,
        shape: GpuMemoryShape,
        output_rows: usize,
        output_columns: usize,
        output_is_ntt: bool,
    ) -> Self {
        Self {
            operation,
            shape,
            output_shape: GpuMemoryShape {
                level: shape.level,
                rows: output_rows,
                columns: output_columns,
                is_ntt: output_is_ntt,
                range: shape.range,
            },
            output_range: shape.range,
            output_rows,
            output_columns,
            output_is_ntt,
            input_resident_bytes: 0,
            base_bits: 0,
            dropped_moduli: 0,
            batch_count: 1,
            product_count: 1,
            level_count: 1,
            digit_size: 1,
            native_evidence: None,
        }
    }

    pub fn with_input_residency(mut self, bytes: usize) -> Self {
        self.input_resident_bytes = bytes;
        self
    }

    pub fn with_output_shape(
        mut self,
        output_shape: GpuMemoryShape,
        output_range: GpuMemoryRange,
    ) -> Self {
        self.output_rows = output_shape.rows;
        self.output_columns = output_shape.columns;
        self.output_is_ntt = output_shape.is_ntt;
        self.output_shape = output_shape;
        self.output_range = output_range;
        self
    }

    pub fn for_decomposition(mut self, base_bits: u32, dropped_moduli: usize) -> Self {
        self.base_bits = base_bits;
        self.dropped_moduli = dropped_moduli;
        self
    }

    pub fn with_batch_topology(mut self, batch_count: usize, product_count: usize) -> Self {
        self.batch_count = batch_count;
        self.product_count = product_count;
        self
    }

    pub fn with_crt_topology(mut self, level_count: usize, digit_size: usize) -> Self {
        self.level_count = level_count;
        self.digit_size = digit_size;
        self
    }

    pub fn with_small_rhs_report(mut self, report: GpuSmallMatrixAllocationReport) -> Self {
        self.native_evidence = Some(GpuNativeMemoryEvidence::SmallRhs(report));
        self
    }

    pub fn with_preimage_evidence(mut self, evidence: PreimageAllocationEvidence) -> Self {
        self.native_evidence = Some(GpuNativeMemoryEvidence::Preimage(evidence));
        self
    }

    pub fn with_trapdoor_evidence(mut self, evidence: TrapdoorAllocationEvidence) -> Self {
        self.native_evidence = Some(GpuNativeMemoryEvidence::Trapdoor(evidence));
        self
    }

    pub fn with_fused_decompose_evidence(
        mut self,
        evidence: FusedDecomposeAllocationEvidence,
    ) -> Self {
        self.native_evidence = Some(GpuNativeMemoryEvidence::FusedDecompose(evidence));
        self
    }

    pub fn with_tensor_row_sum_evidence(
        mut self,
        evidence: TensorRowSumAllocationEvidence,
    ) -> Self {
        self.native_evidence = Some(GpuNativeMemoryEvidence::TensorRowSum(evidence));
        self
    }
}

/// Why a query cannot be certified by the currently exposed native contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum UnsupportedMemoryEvidence {
    MissingNativeSizeQuery { operation: GpuMemoryOperation },
    VariableScratchTopology { operation: GpuMemoryOperation },
    MissingHostStagingContract { operation: GpuMemoryOperation },
    InvalidShape,
}

impl fmt::Display for UnsupportedMemoryEvidence {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingNativeSizeQuery { operation } => {
                write!(formatter, "no native size query for {operation}")
            }
            Self::VariableScratchTopology { operation } => {
                write!(formatter, "scratch topology is not fixed for {operation}")
            }
            Self::MissingHostStagingContract { operation } => {
                write!(formatter, "host staging contract is not exposed for {operation}")
            }
            Self::InvalidShape => formatter.write_str("invalid GPU memory-query shape"),
        }
    }
}

/// A complete operation footprint.  All fields are separate so callers do
/// not accidentally charge scratch, assembly, staging, or cache bytes as an
/// output matrix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GpuOperationMemoryFootprint {
    pub operation: GpuMemoryOperation,
    pub shape: GpuMemoryShape,
    pub output_shape: GpuMemoryShape,
    pub output_range: GpuMemoryRange,
    pub output: Option<GpuMatrixAllocationBytes>,
    /// Complete output-owner bytes when a native report does not expose the
    /// regular matrix descriptor breakdown (for example compact RHS output).
    pub output_bytes: usize,
    pub input_resident_bytes: usize,
    pub auxiliary_bytes: usize,
    pub scratch_bytes: usize,
    pub control_bytes: usize,
    pub cache_bytes: usize,
    pub assembly_bytes: usize,
    pub transfer_bytes: usize,
    pub host_bytes: usize,
    pub pinned_host_bytes: usize,
}

/// Evidence accepted by setup-time admission.  Unsupported is explicit and
/// cannot be mistaken for a zero-sized exact allocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuOperationMemoryEvidence {
    Exact(GpuOperationMemoryFootprint),
    Certified(GpuOperationMemoryFootprint),
    Unsupported {
        operation: GpuMemoryOperation,
        shape: GpuMemoryShape,
        reason: UnsupportedMemoryEvidence,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GpuMemoryQueryError {
    InvalidRange,
    Native(String),
    ArithmeticOverflow,
    MissingNativeEvidence { operation: GpuMemoryOperation },
}

/// Complete allocation topology for the native `tensor_sum_rows` call.
///
/// The materialized implementation first creates the full tensor and then
/// lowers the row groups through bounded row-sum batches.  Those owners are
/// deliberately represented as scratch/workspace instead of being hidden in
/// an output-only exact footprint.  Conversion owners are likewise retained
/// when either source is supplied in coefficient format.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorRowSumAllocationEvidence {
    pub implementation: TensorRowSumImplementation,
    pub left_shape: GpuMemoryShape,
    pub right_shape: GpuMemoryShape,
    pub output_shape: GpuMemoryShape,
    pub tensor_allocation: Option<GpuMatrixAllocationBytes>,
    pub tensor_shape: Option<GpuMemoryShape>,
    pub left_conversion_allocation: Option<GpuMatrixAllocationBytes>,
    pub right_conversion_allocation: Option<GpuMatrixAllocationBytes>,
    pub row_sum_output: GpuMatrixAllocationBytes,
    pub row_sum_workspace_bytes: usize,
    pub row_sum_workspace_owners: usize,
    pub input_resident_bytes: usize,
    pub output_range: GpuMemoryRange,
}

impl TensorRowSumAllocationEvidence {
    pub fn total_scratch_bytes(&self) -> Result<usize, GpuMemoryQueryError> {
        let conversions = self
            .left_conversion_allocation
            .map_or(0, |allocation| allocation.total_bytes)
            .checked_add(
                self.right_conversion_allocation.map_or(0, |allocation| allocation.total_bytes),
            )
            .ok_or(GpuMemoryQueryError::ArithmeticOverflow)?;
        let tensor = self.tensor_allocation.map_or(0, |allocation| allocation.total_bytes);
        conversions
            .checked_add(tensor)
            .and_then(|bytes| bytes.checked_add(self.row_sum_workspace_bytes))
            .ok_or(GpuMemoryQueryError::ArithmeticOverflow)
    }

    pub fn memory_evidence(&self) -> Result<GpuOperationMemoryEvidence, GpuMemoryQueryError> {
        let output = self.row_sum_output;
        let scratch_bytes = self.total_scratch_bytes()?;
        Ok(GpuOperationMemoryEvidence::Certified(GpuOperationMemoryFootprint {
            operation: GpuMemoryOperation::FusedTensorRowSum,
            shape: self.left_shape,
            output_shape: self.output_shape,
            output_range: self.output_range,
            output: Some(output),
            output_bytes: output.data_bytes,
            input_resident_bytes: self.input_resident_bytes,
            auxiliary_bytes: output.aux_bytes.saturating_add(output.event_bytes),
            scratch_bytes,
            control_bytes: 0,
            cache_bytes: 0,
            assembly_bytes: 0,
            transfer_bytes: 0,
            host_bytes: 0,
            pinned_host_bytes: 0,
        }))
    }
}

/// Query the complete native allocation envelope for `tensor_sum_rows`.
///
/// The returned evidence is certified rather than exact because CUDA event
/// allocator overhead is represented by each queried matrix owner and the
/// fallback's bounded reduction may release owners between launches.  The
/// envelope intentionally sums every owner topology, so it is safe for
/// admission even when the allocator reuses released slabs differently.
pub fn tensor_sum_rows_allocation_evidence(
    params: &GpuDCRTPolyParams,
    left_shape: GpuMemoryShape,
    right_shape: GpuMemoryShape,
    groups: &[Vec<usize>],
    output_range: GpuMemoryRange,
) -> Result<TensorRowSumAllocationEvidence, GpuMemoryQueryError> {
    if groups.is_empty() ||
        groups.iter().any(Vec::is_empty) ||
        left_shape.level != right_shape.level ||
        left_shape.range.start >= left_shape.range.end ||
        right_shape.range.start >= right_shape.range.end
    {
        return Err(GpuMemoryQueryError::InvalidRange);
    }
    let left_columns = left_shape.range.width();
    let right_columns = right_shape.range.width();
    let tensor_columns =
        left_columns.checked_mul(right_columns).ok_or(GpuMemoryQueryError::ArithmeticOverflow)?;
    if output_range.start >= output_range.end || output_range.end > tensor_columns {
        return Err(GpuMemoryQueryError::InvalidRange);
    }
    let tensor_rows = left_shape
        .rows
        .checked_mul(right_shape.rows)
        .ok_or(GpuMemoryQueryError::ArithmeticOverflow)?;
    if groups.iter().flatten().any(|&row| row >= tensor_rows) {
        return Err(GpuMemoryQueryError::InvalidRange);
    }
    let output_shape = GpuMemoryShape {
        level: left_shape.level,
        rows: groups.len(),
        columns: tensor_columns,
        is_ntt: true,
        range: output_range,
    };
    let row_sum_output = params
        .matrix_allocation_bytes(left_shape.level, groups.len(), tensor_columns, true)
        .map_err(GpuMemoryQueryError::Native)?;
    let implementation = TensorRowSumImplementation::for_groups(groups);
    let left_conversion_allocation = (!left_shape.is_ntt)
        .then(|| {
            params.matrix_allocation_bytes(left_shape.level, left_shape.rows, left_columns, true)
        })
        .transpose()
        .map_err(GpuMemoryQueryError::Native)?;
    let right_conversion_allocation = (!right_shape.is_ntt)
        .then(|| {
            params.matrix_allocation_bytes(right_shape.level, right_shape.rows, right_columns, true)
        })
        .transpose()
        .map_err(GpuMemoryQueryError::Native)?;
    let (tensor_allocation, row_sum_workspace_bytes, row_sum_workspace_owners) = if implementation
        .is_materialized()
    {
        let tensor = params
            .matrix_allocation_bytes(left_shape.level, tensor_rows, tensor_columns, true)
            .map_err(GpuMemoryQueryError::Native)?;
        // Mirror the production bounded `sum_rows` lowering.  Every batch
        // output and pairwise partial addition is retained in the certified
        // envelope; this may be above the instantaneous high-water mark but
        // cannot under-account a driver that delays an asynchronous release.
        let one_row = params
            .matrix_allocation_bytes(left_shape.level, 1, tensor_columns, true)
            .map_err(GpuMemoryQueryError::Native)?
            .total_bytes;
        let mut workspace = 0usize;
        let mut owners = 0usize;
        let mut batch_groups = 0usize;
        let mut batch_terms = 0usize;
        let flush_batch = |workspace: &mut usize,
                           owners: &mut usize,
                           batch_groups: &mut usize,
                           batch_terms: &mut usize|
         -> Result<(), GpuMemoryQueryError> {
            if *batch_groups == 0 {
                return Ok(());
            }
            let allocation = params
                .matrix_allocation_bytes(left_shape.level, *batch_groups, tensor_columns, true)
                .map_err(GpuMemoryQueryError::Native)?
                .total_bytes;
            *workspace =
                workspace.checked_add(allocation).ok_or(GpuMemoryQueryError::ArithmeticOverflow)?;
            *owners = owners.checked_add(1).ok_or(GpuMemoryQueryError::ArithmeticOverflow)?;
            *batch_groups = 0;
            *batch_terms = 0;
            Ok(())
        };
        for group in groups {
            if group.len() > TensorRowSumImplementation::MAX_TERMS {
                flush_batch(&mut workspace, &mut owners, &mut batch_groups, &mut batch_terms)?;
                let partials = group.len().div_ceil(TensorRowSumImplementation::MAX_TERMS);
                workspace = workspace
                    .checked_add(
                        one_row
                            .checked_mul(partials)
                            .ok_or(GpuMemoryQueryError::ArithmeticOverflow)?,
                    )
                    .ok_or(GpuMemoryQueryError::ArithmeticOverflow)?;
                workspace = workspace
                    .checked_add(
                        one_row
                            .checked_mul(partials.saturating_sub(1))
                            .ok_or(GpuMemoryQueryError::ArithmeticOverflow)?,
                    )
                    .ok_or(GpuMemoryQueryError::ArithmeticOverflow)?;
                owners = owners
                    .checked_add(partials.saturating_mul(2).saturating_sub(1))
                    .ok_or(GpuMemoryQueryError::ArithmeticOverflow)?;
            } else {
                if batch_groups > 0 &&
                    (batch_groups == TensorRowSumImplementation::MAX_GROUPS ||
                        batch_terms + group.len() > TensorRowSumImplementation::MAX_TERMS)
                {
                    flush_batch(&mut workspace, &mut owners, &mut batch_groups, &mut batch_terms)?;
                }
                batch_groups += 1;
                batch_terms += group.len();
            }
        }
        flush_batch(&mut workspace, &mut owners, &mut batch_groups, &mut batch_terms)?;
        (Some(tensor), workspace, owners)
    } else {
        (None, 0, 0)
    };
    let tensor_shape = tensor_allocation.map(|_| GpuMemoryShape {
        level: left_shape.level,
        rows: tensor_rows,
        columns: tensor_columns,
        is_ntt: true,
        range: GpuMemoryRange::new(0, tensor_columns)
            .expect("validated tensor columns are nonzero"),
    });
    Ok(TensorRowSumAllocationEvidence {
        implementation,
        left_shape,
        right_shape,
        output_shape,
        tensor_allocation,
        tensor_shape,
        left_conversion_allocation,
        right_conversion_allocation,
        row_sum_output,
        row_sum_workspace_bytes,
        row_sum_workspace_owners,
        input_resident_bytes: 0,
        output_range,
    })
}

impl fmt::Display for GpuMemoryQueryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidRange => formatter.write_str("invalid GPU memory-query range"),
            Self::Native(error) => formatter.write_str(error),
            Self::ArithmeticOverflow => formatter.write_str("GPU memory-query arithmetic overflow"),
            Self::MissingNativeEvidence { operation } => {
                write!(formatter, "native allocation report required for {operation}")
            }
        }
    }
}

impl std::error::Error for GpuMemoryQueryError {}

fn output_only_footprint(
    params: &GpuDCRTPolyParams,
    query: &GpuMatrixMemoryQuery,
) -> Result<GpuOperationMemoryFootprint, GpuMemoryQueryError> {
    let shape = query.shape;
    let output_shape = query.output_shape;
    if query.output_range.start >= query.output_range.end ||
        query.output_range.end > output_shape.columns ||
        query.output_range.end > query.output_columns
    {
        return Err(GpuMemoryQueryError::InvalidRange);
    }
    let output = params
        .matrix_allocation_bytes(
            output_shape.level,
            output_shape.rows,
            query.output_range.width(),
            output_shape.is_ntt,
        )
        .map_err(GpuMemoryQueryError::Native)?;
    let output_bytes = output.total_bytes;
    Ok(GpuOperationMemoryFootprint {
        operation: query.operation,
        shape,
        output_shape,
        output_range: query.output_range,
        output: Some(output),
        output_bytes,
        input_resident_bytes: query.input_resident_bytes,
        auxiliary_bytes: 0,
        scratch_bytes: 0,
        control_bytes: 0,
        cache_bytes: 0,
        assembly_bytes: 0,
        transfer_bytes: 0,
        host_bytes: 0,
        pinned_host_bytes: 0,
    })
}

fn checked_mul(a: usize, b: usize) -> Result<usize, GpuMemoryQueryError> {
    a.checked_mul(b).ok_or(GpuMemoryQueryError::ArithmeticOverflow)
}

fn checked_add(a: usize, b: usize) -> Result<usize, GpuMemoryQueryError> {
    a.checked_add(b).ok_or(GpuMemoryQueryError::ArithmeticOverflow)
}

fn operation_metadata_bytes(
    params: &GpuDCRTPolyParams,
    query: &GpuMatrixMemoryQuery,
) -> Result<(usize, usize, usize), GpuMemoryQueryError> {
    let limbs = query.shape.level.saturating_add(1).min(params.crt_depth());
    let pointers = std::mem::size_of::<*const u8>();
    let words = std::mem::size_of::<usize>();
    let u64_bytes = std::mem::size_of::<u64>();
    let batch = query.batch_count.max(1);
    let products = query.product_count.max(1);
    match query.operation {
        // MatrixArithBatch allocates pointer tables plus one stride/width/
        // modulus table per active CRT limb.  These are transient device
        // owners, not output descriptors.
        GpuMemoryOperation::Add | GpuMemoryOperation::Sub => {
            let pointer_count = checked_mul(batch, limbs)?;
            let pointer_bytes = checked_mul(checked_mul(pointer_count, 3)?, pointers)?;
            let limb_bytes = checked_add(
                checked_mul(limbs, words)?,
                checked_add(limbs, checked_mul(limbs, u64_bytes)?)?,
            )?;
            Ok((checked_add(pointer_bytes, limb_bytes)?, 0, 0))
        }
        GpuMemoryOperation::Scale | GpuMemoryOperation::Negate => {
            let pointer_count = checked_mul(batch, limbs)?;
            let pointer_tables =
                if matches!(query.operation, GpuMemoryOperation::Negate) { 2 } else { 3 };
            let pointer_bytes = checked_mul(checked_mul(pointer_count, pointer_tables)?, pointers)?;
            let limb_bytes = checked_add(
                checked_mul(limbs, words)?,
                checked_add(limbs, checked_mul(limbs, u64_bytes)?)?,
            )?;
            Ok((checked_add(pointer_bytes, limb_bytes)?, 0, 0))
        }
        GpuMemoryOperation::RingAutomorphism => {
            let pointer_count = checked_mul(batch, limbs)?;
            let pointer_bytes = checked_mul(checked_mul(pointer_count, 2)?, pointers)?;
            let limb_bytes = checked_add(
                checked_mul(limbs, words)?,
                checked_add(limbs, checked_mul(limbs, u64_bytes)?)?,
            )?;
            Ok((checked_add(pointer_bytes, limb_bytes)?, checked_mul(batch, words)?, 0))
        }
        GpuMemoryOperation::Accumulate | GpuMemoryOperation::MatMul => {
            let pointer_count = checked_mul(batch, limbs)?;
            let product_pointer_count = checked_mul(products, pointer_count)?;
            let pointer_tables =
                if matches!(query.operation, GpuMemoryOperation::Accumulate) { 5 } else { 3 };
            let pointer_bytes = checked_mul(
                checked_mul(
                    if matches!(query.operation, GpuMemoryOperation::Accumulate) {
                        product_pointer_count
                    } else {
                        pointer_count
                    },
                    pointer_tables,
                )?,
                pointers,
            )?;
            let limb_bytes = checked_add(
                checked_mul(limbs, words)?,
                checked_add(limbs, checked_mul(limbs, u64_bytes)?)?,
            )?;
            let inner_bytes = if matches!(query.operation, GpuMemoryOperation::Accumulate) {
                checked_mul(products, words)?
            } else {
                0
            };
            Ok((checked_add(pointer_bytes, checked_add(limb_bytes, inner_bytes)?)?, 0, 0))
        }
        GpuMemoryOperation::ModulusConversion => {
            // ModulusConversionMetadata in MatrixCrt.cu: three size_t
            // scalars, two size_t[64] index arrays, four uint64_t[64]
            // arrays, and uint64_t[64*64] Garner entries.  The native call
            // owns one device copy and one pinned host copy.
            const MAX_LIMBS: usize = 64;
            let metadata =
                checked_mul(3 + 2 * MAX_LIMBS + 4 * MAX_LIMBS + MAX_LIMBS * MAX_LIMBS, u64_bytes)?;
            Ok((metadata, 0, metadata))
        }
        GpuMemoryOperation::CenteredRebase => {
            // CenteredRebaseMetadata in MatrixCrt.cu contains source and
            // destination descriptor pointers, destination moduli, source
            // moduli, mixed-radix prefix inverses, and the destination-to-
            // source limb map.  crt_alloc_and_copy_async owns one device
            // copy and one pinned host staging copy until the output event
            // completes; neither is the matrix output owner itself.
            Ok((CENTERED_REBASE_METADATA_BYTES, 0, CENTERED_REBASE_METADATA_BYTES))
        }
        GpuMemoryOperation::BlockModSwitch => {
            // BlockModSwitchMetadata in MatrixCrt.cu: source/output
            // descriptors plus source/target/dropped basis maps and the
            // mixed-radix inverse tables.  The metadata is copied to device
            // and retained by the launch until its completion event.
            const MAX_LIMBS: usize = 64;
            let metadata = checked_add(
                pointers +
                    3 * std::mem::size_of::<usize>() +
                    2 * std::mem::size_of::<usize>() +
                    (4 * MAX_LIMBS + MAX_LIMBS * 2) * u64_bytes +
                    MAX_LIMBS * std::mem::size_of::<usize>() +
                    MAX_LIMBS * std::mem::size_of::<i32>(),
                0,
            )?;
            Ok((metadata, 0, metadata))
        }
        GpuMemoryOperation::RnsConversion => {
            const MAX_LIMBS: usize = 64;
            let launch = 4 * words + MAX_LIMBS * words + 4 * MAX_LIMBS * u64_bytes + u64_bytes;
            let full = launch + MAX_LIMBS * MAX_LIMBS * u64_bytes;
            let source_count = query.shape.level.saturating_add(1);
            let target_count = query.output_shape.level.saturating_add(1).min(params.crt_depth());
            let metadata =
                if checked_mul(source_count, target_count)? <= MAX_LIMBS { 0 } else { full };
            Ok((metadata, 0, 0))
        }
        GpuMemoryOperation::CrtRecompose => {
            const MAX_LIMBS: usize = 64;
            const MAX_WORDS: usize = 64;
            let level = checked_add(
                pointers + words + std::mem::size_of::<i32>() + u64_bytes,
                (MAX_LIMBS + MAX_LIMBS * MAX_LIMBS + MAX_WORDS + MAX_LIMBS) * u64_bytes,
            )?;
            let output = checked_add(pointers + words, MAX_LIMBS * u64_bytes)?;
            let levels = checked_mul(query.level_count.max(1), level)?;
            Ok((checked_add(levels, output)?, 0, checked_add(levels, output)?))
        }
        // These native entry points only use the caller-owned output (or
        // write directly into a sampler output), so there is no additional
        // operation metadata beyond the output allocation itself.
        GpuMemoryOperation::Transpose |
        GpuMemoryOperation::Slice |
        GpuMemoryOperation::Tensor |
        GpuMemoryOperation::Concat |
        GpuMemoryOperation::Hash |
        GpuMemoryOperation::UniformSampler |
        GpuMemoryOperation::GaussianSampler |
        GpuMemoryOperation::Constant |
        GpuMemoryOperation::Lift |
        GpuMemoryOperation::FusedRowSum |
        GpuMemoryOperation::FusedTensorRowSum |
        GpuMemoryOperation::FusedRowBlockAdd => Ok((0, 0, 0)),
        // The query dispatcher supplies operation-specific native evidence
        // for these families before reaching this helper.  If that invariant
        // changes, fail explicitly rather than certifying a zero workspace.
        GpuMemoryOperation::SmallRhs |
        GpuMemoryOperation::Decompose |
        GpuMemoryOperation::Trapdoor |
        GpuMemoryOperation::Preimage |
        GpuMemoryOperation::FusedDecompose |
        GpuMemoryOperation::FusedSmallProduct |
        GpuMemoryOperation::FusedPreimageBatch => {
            Err(GpuMemoryQueryError::MissingNativeEvidence { operation: query.operation })
        }
    }
}

fn certified_operation_footprint(
    params: &GpuDCRTPolyParams,
    query: &GpuMatrixMemoryQuery,
) -> Result<GpuOperationMemoryFootprint, GpuMemoryQueryError> {
    let mut footprint = output_only_footprint(params, query)?;
    footprint.operation = query.operation;
    footprint.output = footprint.output.take();
    let (scratch, control, pinned) = operation_metadata_bytes(params, query)?;
    footprint.scratch_bytes = scratch;
    footprint.control_bytes = control;
    footprint.pinned_host_bytes = pinned;

    // Hash and GPU samplers call the native distribution kernel directly into
    // the newly allocated output owner.  The hash-to-seed state is host stack
    // state and therefore contributes no device scratch.
    if matches!(
        query.operation,
        GpuMemoryOperation::Hash |
            GpuMemoryOperation::UniformSampler |
            GpuMemoryOperation::GaussianSampler
    ) {
        footprint.host_bytes = 0;
    }
    Ok(footprint)
}

/// Account the complete full-matrix centered-rebase topology.
///
/// Production first materializes an output in coefficient format, launches
/// the mixed-radix kernel, and converts that same owner back to evaluation
/// format.  An evaluation-format input is first cloned and transformed to a
/// temporary coefficient owner; that owner (including its lifetime events)
/// remains live through the kernel launch and is therefore scratch rather
/// than resident input.  The destination output is queried using the native
/// matrix allocator, while metadata and pinned staging are charged from the
/// dedicated centered-rebase contract above.
fn centered_rebase_operation_footprint(
    params: &GpuDCRTPolyParams,
    query: &GpuMatrixMemoryQuery,
) -> Result<GpuOperationMemoryFootprint, GpuMemoryQueryError> {
    if query.shape.level >= params.crt_depth() ||
        query.output_shape.level >= params.crt_depth() ||
        query.output_shape.level < query.shape.level ||
        query.shape.rows != query.output_shape.rows ||
        query.output_range.width() != query.shape.range.width()
    {
        return Err(GpuMemoryQueryError::InvalidRange);
    }
    // The native destination is allocated in coefficient format even though
    // the public operation returns an evaluation-domain matrix after the
    // in-place NTT.  Query the same owner layout, then retain the caller's
    // final shape in the evidence for diagnostics.
    let mut allocation_query = query.clone();
    allocation_query.output_shape.is_ntt = false;
    allocation_query.output_is_ntt = false;
    let mut footprint = output_only_footprint(params, &allocation_query)?;
    footprint.output_shape = query.output_shape;
    let (metadata, control, pinned) = operation_metadata_bytes(params, query)?;
    footprint.scratch_bytes = metadata;
    footprint.control_bytes = control;
    footprint.pinned_host_bytes = pinned;

    if query.shape.is_ntt {
        // `into_coeff_domain` clones the source owner and retains its matrix
        // descriptor/event allocation until the centered-rebase launch has
        // recorded all source consumers.
        let source = params
            .matrix_allocation_bytes(
                query.shape.level,
                query.shape.rows,
                query.shape.range.width(),
                false,
            )
            .map_err(GpuMemoryQueryError::Native)?;
        footprint.scratch_bytes = footprint
            .scratch_bytes
            .checked_add(source.total_bytes)
            .ok_or(GpuMemoryQueryError::ArithmeticOverflow)?;
    }
    Ok(footprint)
}

/// Query the native matrix owner and operation contract for one exact range.
///
/// The arithmetic kernels below are audited to launch into a caller-owned
/// output and allocate no additional device owner where marked `Exact`.
/// Decomposition and batch/CRT/sampler families use `Certified` footprints
/// containing their native temporary metadata or workspace branches. No
/// operation receives a fabricated exact `scratch_bytes == 0` result.
pub fn query_matrix_operation_memory(
    params: &GpuDCRTPolyParams,
    query: GpuMatrixMemoryQuery,
) -> Result<GpuOperationMemoryEvidence, GpuMemoryQueryError> {
    if query.output_range.start >= query.output_range.end ||
        query.output_range.end > query.output_shape.columns ||
        query.output_range.end > query.output_columns
    {
        return Err(GpuMemoryQueryError::InvalidRange);
    }
    if matches!(
        query.operation,
        GpuMemoryOperation::SmallRhs |
            GpuMemoryOperation::FusedSmallProduct |
            GpuMemoryOperation::Trapdoor |
            GpuMemoryOperation::Preimage |
            GpuMemoryOperation::FusedPreimageBatch |
            GpuMemoryOperation::FusedDecompose |
            GpuMemoryOperation::FusedTensorRowSum
    ) {
        return match (query.operation, query.native_evidence.clone()) {
            (
                GpuMemoryOperation::SmallRhs | GpuMemoryOperation::FusedSmallProduct,
                Some(GpuNativeMemoryEvidence::SmallRhs(report)),
            ) => Ok(small_rhs_memory_evidence(query.operation, query.shape, report)),
            (
                GpuMemoryOperation::Preimage | GpuMemoryOperation::FusedPreimageBatch,
                Some(GpuNativeMemoryEvidence::Preimage(evidence)),
            ) => Ok(preimage_memory_evidence(query.operation, &evidence)),
            (GpuMemoryOperation::Trapdoor, Some(GpuNativeMemoryEvidence::Trapdoor(evidence))) => {
                Ok(trapdoor_memory_evidence(query.operation, query.shape, &evidence))
            }
            (
                GpuMemoryOperation::FusedDecompose,
                Some(GpuNativeMemoryEvidence::FusedDecompose(evidence)),
            ) => Ok(evidence.memory_evidence()),
            (
                GpuMemoryOperation::FusedTensorRowSum,
                Some(GpuNativeMemoryEvidence::TensorRowSum(evidence)),
            ) => evidence.memory_evidence(),
            _ => Err(GpuMemoryQueryError::MissingNativeEvidence { operation: query.operation }),
        };
    }
    let output_only = matches!(
        query.operation,
        GpuMemoryOperation::Transpose |
            GpuMemoryOperation::Slice |
            GpuMemoryOperation::Tensor |
            GpuMemoryOperation::Concat |
            GpuMemoryOperation::FusedRowSum |
            GpuMemoryOperation::FusedRowBlockAdd
    );
    if output_only {
        return output_only_footprint(params, &query).map(GpuOperationMemoryEvidence::Exact);
    }
    if query.operation == GpuMemoryOperation::CenteredRebase {
        return centered_rebase_operation_footprint(params, &query)
            .map(GpuOperationMemoryEvidence::Certified);
    }
    if matches!(query.operation, GpuMemoryOperation::Decompose | GpuMemoryOperation::FusedDecompose)
    {
        if query.dropped_moduli >= params.crt_depth() {
            return Err(GpuMemoryQueryError::InvalidRange);
        }
        let mut footprint = output_only_footprint(params, &query)?;
        // MatrixDecompose allocates and copies a temporary source whenever
        // the source is in Eval format or dropped CRT towers need correction.
        if query.shape.is_ntt || query.dropped_moduli > 0 {
            let temporary = params
                .matrix_allocation_bytes(
                    query.shape.level,
                    query.shape.rows,
                    query.shape.range.width(),
                    query.shape.is_ntt,
                )
                .map_err(GpuMemoryQueryError::Native)?;
            footprint.scratch_bytes = temporary.total_bytes;
        }
        return Ok(GpuOperationMemoryEvidence::Certified(footprint));
    }
    certified_operation_footprint(params, &query).map(GpuOperationMemoryEvidence::Certified)
}

/// Query the complete native envelope for a fused row-block decomposition.
///
/// This is deliberately separate from the regular matrix decomposition query:
/// the production call creates a compact output whose dimensions depend on
/// every physical input block and whose native payload sizing is not the
/// regular matrix allocator. No scratch or assembly bytes are inferred; the
/// native compact-owner size query is used for the output and every source is
/// queried independently.
pub fn fused_decompose_row_blocks_allocation_evidence(
    params: &GpuDCRTPolyParams,
    blocks: &[GpuDecomposeBlockShape],
    small: bool,
    digits: usize,
    output_range: GpuMemoryRange,
) -> Result<FusedDecomposeAllocationEvidence, GpuMemoryQueryError> {
    let first = blocks.first().ok_or(GpuMemoryQueryError::InvalidRange)?;
    if blocks.len() > 32 || digits == 0 || output_range.start >= output_range.end {
        return Err(GpuMemoryQueryError::InvalidRange);
    }
    if first.level >= params.crt_depth() ||
        first.range.start >= first.range.end ||
        first.range.end > first.columns
    {
        return Err(GpuMemoryQueryError::InvalidRange);
    }
    if blocks.iter().any(|block| {
        block.level != first.level ||
            block.columns != first.columns ||
            block.range.start >= block.range.end ||
            block.range.end > block.columns
    }) {
        return Err(GpuMemoryQueryError::InvalidRange);
    }

    let default_digits = if small {
        params.crt_bits().div_ceil(params.base_bits() as usize)
    } else {
        params.modulus_digits()
    };
    if small && digits != default_digits {
        return Err(GpuMemoryQueryError::InvalidRange);
    }
    let dropped = if small {
        params.dropped_moduli()
    } else {
        params.gadget_dropped_moduli(Some(digits)).ok_or(GpuMemoryQueryError::InvalidRange)?
    };
    if dropped >= params.crt_depth() {
        return Err(GpuMemoryQueryError::InvalidRange);
    }
    let output_limb_rows = if small {
        1
    } else {
        params.crt_depth().checked_sub(dropped).ok_or(GpuMemoryQueryError::ArithmeticOverflow)?
    };
    let source_rows = blocks.iter().try_fold(0usize, |total, block| {
        total.checked_add(block.rows).ok_or(GpuMemoryQueryError::ArithmeticOverflow)
    })?;
    let output_rows = source_rows
        .checked_mul(digits)
        .and_then(|rows| rows.checked_mul(output_limb_rows))
        .ok_or(GpuMemoryQueryError::ArithmeticOverflow)?;
    let output_columns = output_range.width();
    if output_range.end > first.columns || output_columns == 0 {
        return Err(GpuMemoryQueryError::InvalidRange);
    }

    let mut input_allocations = Vec::with_capacity(blocks.len());
    for block in blocks {
        let allocation = params
            .matrix_allocation_bytes(block.level, block.rows, block.range.width(), block.is_ntt)
            .map_err(GpuMemoryQueryError::Native)?;
        input_allocations.push(allocation);
    }
    let input_resident_bytes = input_allocations.iter().try_fold(0usize, |total, allocation| {
        total.checked_add(allocation.total_bytes).ok_or(GpuMemoryQueryError::ArithmeticOverflow)
    })?;

    let base = BigUint::one() << params.base_bits();
    let bound = if small { &base - BigUint::one() } else { (&base + BigUint::one()) >> 1 };
    let magnitude_bytes = usize::try_from(bound.bits().div_ceil(8))
        .map_err(|_| GpuMemoryQueryError::ArithmeticOverflow)?
        .max(1);
    let output_allocation = GpuSmallMatrix::allocation_bytes_for_shape(
        params,
        output_rows,
        output_columns,
        magnitude_bytes,
    )
    .map_err(GpuMemoryQueryError::Native)?;
    let per_block_event_bytes =
        input_allocations.iter().map(|allocation| allocation.event_bytes).collect();
    Ok(FusedDecomposeAllocationEvidence {
        operation: GpuMemoryOperation::FusedDecompose,
        blocks: blocks.to_vec(),
        input_allocations,
        per_block_scratch_bytes: vec![0; blocks.len()],
        per_block_event_bytes,
        per_block_assembly_bytes: vec![0; blocks.len()],
        per_block_reuse_bytes: vec![0; blocks.len()],
        output_event_bytes: output_allocation.event_bytes,
        output_allocation,
        output_rows,
        output_columns,
        output_range,
        output_ports: 1,
        input_resident_bytes,
        scratch_bytes: 0,
        control_bytes: 0,
        cache_bytes: 0,
        assembly_bytes: 0,
        transfer_bytes: 0,
        host_bytes: 0,
        pinned_host_bytes: 0,
        inputs_consumed: true,
        evidence_kind: PreimageAllocationEvidenceKind::Certified,
    })
}

/// Convert the compact-RHS native report into the operation vocabulary.  The
/// report is already width-specific and includes the expanded RHS workspace,
/// output owner, event overhead, and retained compact owner.
pub fn small_rhs_memory_evidence(
    operation: GpuMemoryOperation,
    shape: GpuMemoryShape,
    report: GpuSmallMatrixAllocationReport,
) -> GpuOperationMemoryEvidence {
    let footprint = GpuOperationMemoryFootprint {
        operation,
        shape,
        output_shape: shape,
        output_range: shape.range,
        output: None,
        output_bytes: report.full_output_bytes,
        input_resident_bytes: report.lhs_eval_bytes.saturating_add(report.compact_rhs_bytes),
        auxiliary_bytes: report.event_overhead_bytes,
        scratch_bytes: report.expanded_rhs_workspace_bytes,
        control_bytes: 0,
        cache_bytes: 0,
        assembly_bytes: 0,
        transfer_bytes: 0,
        host_bytes: 0,
        pinned_host_bytes: 0,
    };
    GpuOperationMemoryEvidence::Exact(footprint)
}

/// Convert the fixed preimage sampler's typed cold/warm envelope without
/// dropping cache, control, staging, or retry workspace components.
pub fn preimage_memory_evidence(
    operation: GpuMemoryOperation,
    evidence: &PreimageAllocationEvidence,
) -> GpuOperationMemoryEvidence {
    let envelope = &evidence.envelope;
    let shape = GpuMemoryShape {
        level: evidence.context.format.crt_level,
        rows: 0,
        columns: evidence.context.tile_columns,
        is_ntt: evidence.context.format.compact_output,
        range: GpuMemoryRange { start: 0, end: evidence.context.tile_columns },
    };
    let footprint = GpuOperationMemoryFootprint {
        operation,
        shape,
        output_shape: shape,
        output_range: shape.range,
        output: None,
        output_bytes: envelope.compact_output_bytes,
        input_resident_bytes: envelope
            .public_matrix_bytes
            .saturating_add(envelope.trapdoor_bytes)
            .saturating_add(envelope.resident_target_bytes),
        auxiliary_bytes: envelope.sampler_event_bytes,
        scratch_bytes: envelope
            .candidate_workspace_bytes
            .saturating_add(envelope.perturbation_workspace_bytes)
            .saturating_add(envelope.scratch_bytes)
            .saturating_add(envelope.cold_transient_workspace_bytes),
        control_bytes: envelope
            .device_control_bytes
            .saturating_add(envelope.pinned_host_control_bytes),
        cache_bytes: envelope.retained_covariance_cache_bytes,
        assembly_bytes: envelope.compact_output_bytes,
        transfer_bytes: envelope
            .source_device_staging_bytes
            .saturating_add(envelope.destination_device_staging_bytes),
        host_bytes: envelope.host_staging_bytes,
        pinned_host_bytes: envelope.pinned_host_control_bytes,
    };
    match envelope.evidence_kind {
        PreimageAllocationEvidenceKind::Exact => GpuOperationMemoryEvidence::Exact(footprint),
        PreimageAllocationEvidenceKind::Certified => {
            GpuOperationMemoryEvidence::Certified(footprint)
        }
    }
}

/// Convert the native trapdoor-generation report into the operation
/// vocabulary. The report is independent from preimage sampling and must not
/// be substituted by the preimage envelope.
pub fn trapdoor_memory_evidence(
    operation: GpuMemoryOperation,
    shape: GpuMemoryShape,
    evidence: &TrapdoorAllocationEvidence,
) -> GpuOperationMemoryEvidence {
    let footprint = GpuOperationMemoryFootprint {
        operation,
        shape,
        output_shape: shape,
        output_range: shape.range,
        output: None,
        output_bytes: evidence.public_output_bytes,
        input_resident_bytes: evidence.public_a_bar_bytes.saturating_add(evidence.trapdoor_bytes()),
        auxiliary_bytes: 0,
        scratch_bytes: evidence.scratch_bytes,
        control_bytes: evidence.control_bytes,
        cache_bytes: evidence.cache_bytes,
        assembly_bytes: evidence.assembly_bytes(),
        transfer_bytes: 0,
        host_bytes: evidence.host_bytes,
        pinned_host_bytes: evidence.pinned_host_bytes,
    };
    match evidence.evidence_kind {
        PreimageAllocationEvidenceKind::Exact => GpuOperationMemoryEvidence::Exact(footprint),
        PreimageAllocationEvidenceKind::Certified => {
            GpuOperationMemoryEvidence::Certified(footprint)
        }
    }
}

/// Return the compact owner report for a fixed production RHS.  Keeping this
/// helper here gives fleet callers one stable operation-facing entry point
/// while preserving the native report's detailed fields for diagnostics.
pub fn small_rhs_report(
    rhs: &GpuSmallMatrix,
    lhs: &GpuDCRTPolyMatrix,
) -> Result<GpuSmallMatrixAllocationReport, crate::matrix::SmallMatrixError> {
    rhs.allocation_report(lhs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_families_never_look_like_zero_scratch_exact() {
        let params = GpuDCRTPolyParams::new(32, vec![131_009], 2, None);
        let range = GpuMemoryRange::new(0, 2).unwrap();
        let shape = GpuMemoryShape::new(0, 2, 2, true, range).unwrap();
        let query = GpuMatrixMemoryQuery::new(GpuMemoryOperation::Decompose, shape, 4, 2, true)
            .for_decomposition(2, 0);
        let evidence = query_matrix_operation_memory(&params, query).unwrap();
        let GpuOperationMemoryEvidence::Certified(footprint) = evidence else {
            panic!("decomposition must carry certified temporary scratch")
        };
        assert!(footprint.scratch_bytes > 0);

        let hash = GpuMatrixMemoryQuery::new(GpuMemoryOperation::Hash, shape, 2, 2, true);
        assert!(matches!(
            query_matrix_operation_memory(&params, hash).unwrap(),
            GpuOperationMemoryEvidence::Certified(_)
        ));
    }

    #[test]
    fn output_query_is_width_specific_and_uses_native_alignment() {
        let params = GpuDCRTPolyParams::new(32, vec![131_009], 2, None);
        let narrow =
            GpuMemoryShape::new(0, 3, 8, true, GpuMemoryRange::new(0, 2).unwrap()).unwrap();
        let wide = GpuMemoryShape::new(0, 3, 8, true, GpuMemoryRange::new(0, 8).unwrap()).unwrap();
        let narrow_query = GpuMatrixMemoryQuery::new(GpuMemoryOperation::Add, narrow, 3, 8, true);
        let wide_query = GpuMatrixMemoryQuery::new(GpuMemoryOperation::Add, wide, 3, 8, true);
        let GpuOperationMemoryEvidence::Certified(narrow) =
            query_matrix_operation_memory(&params, narrow_query).unwrap()
        else {
            panic!("add output query should be certified")
        };
        let GpuOperationMemoryEvidence::Certified(wide) =
            query_matrix_operation_memory(&params, wide_query).unwrap()
        else {
            panic!("add output query should be certified")
        };
        assert!(narrow.output.unwrap().total_bytes < wide.output.unwrap().total_bytes);
    }

    #[test]
    fn centered_rebase_has_a_distinct_certified_destination_contract() {
        let params = GpuDCRTPolyParams::new(32, vec![131_009], 2, None);
        let range = GpuMemoryRange::new(0, 2).unwrap();
        let coefficient_shape = GpuMemoryShape::new(0, 3, 2, false, range).unwrap();
        let query = GpuMatrixMemoryQuery::new(
            GpuMemoryOperation::CenteredRebase,
            coefficient_shape,
            3,
            2,
            true,
        );
        let GpuOperationMemoryEvidence::Certified(footprint) =
            query_matrix_operation_memory(&params, query).unwrap()
        else {
            panic!("centered rebase must use its certified native contract")
        };
        assert_eq!(footprint.operation, GpuMemoryOperation::CenteredRebase);
        assert_eq!(footprint.output_shape.is_ntt, true);
        assert_eq!(footprint.scratch_bytes, CENTERED_REBASE_METADATA_BYTES);
        assert_eq!(footprint.pinned_host_bytes, CENTERED_REBASE_METADATA_BYTES);
        assert_eq!(footprint.control_bytes, 0);

        let eval_shape = GpuMemoryShape::new(0, 3, 2, true, range).unwrap();
        let eval_query =
            GpuMatrixMemoryQuery::new(GpuMemoryOperation::CenteredRebase, eval_shape, 3, 2, true);
        let GpuOperationMemoryEvidence::Certified(eval) =
            query_matrix_operation_memory(&params, eval_query).unwrap()
        else {
            panic!("evaluation input must retain a coefficient conversion owner")
        };
        let source = params.matrix_allocation_bytes(0, 3, 2, false).unwrap();
        assert_eq!(eval.scratch_bytes, CENTERED_REBASE_METADATA_BYTES + source.total_bytes);
        assert_ne!(
            eval.scratch_bytes,
            operation_metadata_bytes(
                &params,
                &GpuMatrixMemoryQuery::new(
                    GpuMemoryOperation::ModulusConversion,
                    coefficient_shape,
                    3,
                    2,
                    true,
                )
            )
            .unwrap()
            .0
        );
    }

    #[test]
    fn centered_rebase_rejects_shape_or_range_mismatches() {
        let params = GpuDCRTPolyParams::new(32, vec![131_009], 2, None);
        let source =
            GpuMemoryShape::new(0, 2, 4, false, GpuMemoryRange::new(0, 2).unwrap()).unwrap();
        let rows_mismatch =
            GpuMatrixMemoryQuery::new(GpuMemoryOperation::CenteredRebase, source, 3, 4, true);
        assert_eq!(
            query_matrix_operation_memory(&params, rows_mismatch),
            Err(GpuMemoryQueryError::InvalidRange)
        );
        let width_mismatch =
            GpuMatrixMemoryQuery::new(GpuMemoryOperation::CenteredRebase, source, 2, 4, true)
                .with_output_shape(
                    GpuMemoryShape::new(0, 2, 4, true, GpuMemoryRange::new(0, 4).unwrap()).unwrap(),
                    GpuMemoryRange::new(0, 4).unwrap(),
                );
        assert_eq!(
            query_matrix_operation_memory(&params, width_mismatch),
            Err(GpuMemoryQueryError::InvalidRange)
        );
    }

    #[test]
    fn every_canonical_operation_has_native_or_certified_memory_evidence() {
        let params = GpuDCRTPolyParams::new(32, vec![131_009], 2, None);
        let shape = GpuMemoryShape::new(0, 2, 4, true, GpuMemoryRange::new(0, 2).unwrap()).unwrap();
        let fused_decompose = fused_decompose_row_blocks_allocation_evidence(
            &params,
            &[GpuDecomposeBlockShape {
                level: 0,
                rows: 2,
                columns: 4,
                is_ntt: true,
                range: GpuMemoryRange::new(0, 2).unwrap(),
            }],
            false,
            params.modulus_digits(),
            GpuMemoryRange::new(0, 2).unwrap(),
        )
        .expect("fused decomposition must expose native allocation evidence");
        for &operation in GpuMemoryOperation::CANONICAL.iter().filter(|operation| {
            !matches!(
                operation,
                GpuMemoryOperation::SmallRhs |
                    GpuMemoryOperation::FusedSmallProduct |
                    GpuMemoryOperation::Trapdoor |
                    GpuMemoryOperation::Preimage |
                    GpuMemoryOperation::FusedPreimageBatch
            )
        }) {
            let query = GpuMatrixMemoryQuery::new(operation, shape, 2, 4, true)
                .for_decomposition(2, 0)
                .with_batch_topology(2, 2)
                .with_crt_topology(2, 1);
            let query = match operation {
                GpuMemoryOperation::FusedDecompose => {
                    query.with_fused_decompose_evidence(fused_decompose.clone())
                }
                GpuMemoryOperation::FusedTensorRowSum => {
                    // This operation's native allocation contract includes
                    // the tensor owner and row-reduction workspace.  Supply
                    // the same valid topology that production queries use;
                    // output-only accounting is intentionally not accepted.
                    let groups = vec![vec![0], vec![1]];
                    let output_range = GpuMemoryRange::new(0, 4).unwrap();
                    let tensor_evidence = tensor_sum_rows_allocation_evidence(
                        &params,
                        shape,
                        shape,
                        &groups,
                        output_range,
                    )
                    .expect("tensor row sum must expose native allocation evidence");
                    let output_shape =
                        GpuMemoryShape::new(0, groups.len(), 4, true, output_range).unwrap();
                    query
                        .with_output_shape(output_shape, output_range)
                        .with_tensor_row_sum_evidence(tensor_evidence)
                }
                _ => query,
            };
            let evidence = query_matrix_operation_memory(&params, query)
                .unwrap_or_else(|error| panic!("{operation} query failed: {error}"));
            assert!(
                !matches!(evidence, GpuOperationMemoryEvidence::Unsupported { .. }),
                "canonical operation {operation} must not be unsupported"
            );
        }
    }

    #[test]
    fn compact_rhs_uses_existing_native_report_without_recomputation() {
        let report = GpuSmallMatrixAllocationReport {
            lhs_eval_bytes: 101,
            compact_rhs_bytes: 37,
            full_output_bytes: 211,
            expanded_rhs_workspace_bytes: 307,
            event_overhead_bytes: 19,
            high_water_bytes: 675,
            full_expanded_rhs_bytes: 307,
            workspace_word_bytes: 4,
            ntt_preparation_launches: 2,
            u32_workspace_limb_count: 1,
            u64_workspace_limb_count: 0,
        };
        let evidence = small_rhs_memory_evidence(
            GpuMemoryOperation::FusedSmallProduct,
            GpuMemoryShape::new(0, 2, 3, true, GpuMemoryRange::new(0, 2).unwrap()).unwrap(),
            report,
        );
        let GpuOperationMemoryEvidence::Exact(footprint) = evidence else {
            panic!("small RHS native report must be exact")
        };
        assert_eq!(footprint.output_bytes, 211);
        assert_eq!(footprint.scratch_bytes, 307);
        assert_eq!(footprint.auxiliary_bytes, 19);
    }

    #[test]
    fn input_and_output_ranges_are_independent_for_decomposition() {
        let params = GpuDCRTPolyParams::new(32, vec![131_009], 2, None);
        let input = GpuMemoryShape::new(0, 2, 8, true, GpuMemoryRange::new(2, 4).unwrap()).unwrap();
        let output =
            GpuMemoryShape::new(0, 8, 3, true, GpuMemoryRange::new(0, 3).unwrap()).unwrap();
        let query = GpuMatrixMemoryQuery::new(GpuMemoryOperation::Decompose, input, 8, 3, true)
            .with_output_shape(output, GpuMemoryRange::new(0, 3).unwrap())
            .for_decomposition(2, 0);
        let GpuOperationMemoryEvidence::Certified(evidence) =
            query_matrix_operation_memory(&params, query).unwrap()
        else {
            panic!("decomposition must be certified")
        };
        assert_eq!(evidence.shape.columns, 8);
        assert_eq!(evidence.output_shape.columns, 3);
        assert_eq!(evidence.output_range.width(), 3);
        assert!(evidence.scratch_bytes > 0);
    }

    #[test]
    fn tensor_row_sum_native_boundaries_select_and_account_the_real_topology() {
        let params = GpuDCRTPolyParams::new(32, vec![131_009], 2, None);
        let left = GpuMemoryShape::new(0, 2, 2, false, GpuMemoryRange::new(0, 2).unwrap()).unwrap();
        let right =
            GpuMemoryShape::new(0, 2, 2, false, GpuMemoryRange::new(0, 2).unwrap()).unwrap();
        let output_range = GpuMemoryRange::new(0, 4).unwrap();
        // The 2x2 tensor has only four valid rows.  Repeat valid rows while
        // crossing the group/term boundaries so this test exercises dispatch
        // selection rather than an invalid-row rejection.
        let groups16 = (0..16).map(|row| vec![row % 4]).collect::<Vec<_>>();
        let groups17 = (0..17).map(|row| vec![row % 4]).collect::<Vec<_>>();
        let groups32 = (0..16).map(|row| vec![row % 4, row % 4]).collect::<Vec<_>>();
        let mut groups33 = groups32.clone();
        groups33[0].push(0);

        assert_eq!(groups17.len(), 17);
        assert_eq!(groups17.iter().map(Vec::len).sum::<usize>(), 17);
        assert_eq!(groups33.len(), 16);
        assert_eq!(groups33.iter().map(Vec::len).sum::<usize>(), 33);

        for (groups, expected) in [
            (&groups16, TensorRowSumImplementation::FusedKernel),
            (&groups32, TensorRowSumImplementation::FusedKernel),
            (&groups17, TensorRowSumImplementation::MaterializedTensor),
            (&groups33, TensorRowSumImplementation::MaterializedTensor),
        ] {
            let evidence =
                tensor_sum_rows_allocation_evidence(&params, left, right, groups, output_range)
                    .unwrap();
            assert_eq!(evidence.implementation, expected);
            assert_eq!(evidence.output_shape.rows, groups.len());
            if expected.is_materialized() {
                assert!(evidence.tensor_allocation.is_some());
                assert_eq!(evidence.tensor_shape.unwrap().rows, 4);
                assert_eq!(evidence.tensor_shape.unwrap().columns, 4);
                assert!(evidence.left_conversion_allocation.is_some());
                assert!(evidence.right_conversion_allocation.is_some());
                assert!(evidence.row_sum_workspace_bytes > 0);
                assert!(matches!(
                    evidence.memory_evidence().unwrap(),
                    GpuOperationMemoryEvidence::Certified(_)
                ));
            } else {
                assert!(evidence.tensor_allocation.is_none());
                assert_eq!(evidence.row_sum_workspace_bytes, 0);
                assert!(evidence.left_conversion_allocation.is_some());
                assert!(evidence.right_conversion_allocation.is_some());
            }
        }
    }
}
