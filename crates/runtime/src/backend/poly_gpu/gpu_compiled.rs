//! Bind ledger-issued column plans to ordinary and compact backend calls. Admitted
//! output owners are initialized once, then filled through allocation-free
//! range views on the existing enqueue workers. Setup/sealing and compilation
//! of other invocation classes remain separate requirements.

#[path = "gpu_batch.rs"]
mod gpu_batch;
#[path = "gpu_preimage_prepared.rs"]
mod gpu_preimage_prepared;

pub(super) use super::gpu_claims::PreparedClaimBroker;

use super::{
    gpu_prepare::{
        MatrixInputLayouts, PreparedFleetOutput, PreparedMatrixInputs, PreparedMatrixSource,
        PreparedMatrixValue,
    },
    *,
};
use crate::{
    gpu_invocation::{GpuInvocation, GpuNodeOperation},
    gpu_memory::{GpuAdmissionError, GpuColumnMemoryPlan},
};
use mxx_ir_core::{
    ParamEnv,
    node::{ConcatAxis, MatrixBinaryOp, NodeKind},
    types::{ConcreteMatrixType, ConcreteWireType},
};
use mxx_primitives::{
    matrix::{
        PolyMatrixColumnData,
        gpu_dcrt_poly::{
            GpuCompactDecompositionLayout, GpuCompactTransferKind, GpuMatrixCrtOperation,
            GpuMatrixModulusConversion, GpuMatrixRangeConstant, GpuMatrixRnsConversion,
            GpuMatrixSampleDist, GpuPreparedRequest, GpuPreparedSlotIdentity, GpuPreparedSlotKind,
            GpuPreparedStorage, GpuPreparedWorkspaceLayout, GpuTracedClaim,
        },
    },
    poly::dcrt::gpu::GpuRngSeed,
    sampler::{
        PolyTrapdoorSampler,
        trapdoor::gpu::{GpuDCRTPolyTrapdoorSampler, GpuDCRTTrapdoor},
    },
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum PreparedOperation {
    Constant {
        ty: ConcreteMatrixType,
        value: GpuMatrixRangeConstant,
    },
    Sample {
        ty: ConcreteMatrixType,
        distribution: GpuMatrixSampleDist,
        sigma_bits: u64,
        max_coefficient_bound: BigInt,
    },
    Polynomial {
        ty: ConcreteMatrixType,
        coefficients: Vec<BigInt>,
        evaluation: bool,
    },
    PolynomialReadback {
        evaluation: bool,
        claims: Vec<GpuTracedClaim>,
    },
    Hash {
        ty: ConcreteMatrixType,
    },
    ModulusConversion {
        destination: ConcreteMatrixType,
        conversion: GpuMatrixModulusConversion,
    },
    CrtRecompose {
        destination: ConcreteMatrixType,
        plaintext_moduli: Vec<u64>,
        reconstruction_coefficients: Vec<BigInt>,
    },
    RnsConversion {
        destination: ConcreteMatrixType,
        source_moduli: Vec<u64>,
        conversion: GpuMatrixRnsConversion,
    },
    CenteredRebase {
        destination: ConcreteMatrixType,
    },
    /// Copy compact columns into a retained compact owner of a containing CRT
    /// basis. Rows, columns and the inclusive bound come from the compact input.
    CenteredExtendCompact {
        destination: ConcreteMatrixType,
        bound: num_bigint::BigUint,
    },
    /// Ordinary LHS (fixed, evaluation format) times compact RHS columns. The
    /// compact input owns the output columns; expansion workspace scales with
    /// the admitted width and is claimed per range.
    MultiplyCompact {
        columns: usize,
        inner: usize,
    },
    /// Canonical compact-bytes artifact import into a retained ordinary owner.
    /// The payload is supplied at execution and is not part of the identity.
    ImportMatrix {
        ty: ConcreteMatrixType,
        evaluation: bool,
        max_coefficient_bits: u16,
    },
    /// Canonical compact-coefficient artifact import into a retained compact owner.
    ImportCompact {
        ty: ConcreteMatrixType,
        bound: num_bigint::BigUint,
    },
    /// Native RNS staging bytes loaded per admitted range through the
    /// production column staging loader into a same-format scratch owner.
    ImportStaging {
        ty: ConcreteMatrixType,
        evaluation: bool,
        bytes_per_poly: usize,
        payload_len: usize,
    },
    /// Bounded trapdoor preimage columns into a retained compact owner. Every
    /// attempt, the destination's hard-cutoff plan and the target tile hold
    /// exact native claims recorded by a trace of the same production kernels.
    Preimage {
        ty: ConcreteMatrixType,
        bound: num_bigint::BigUint,
        sigma_bits: u64,
        gadget_base: BigInt,
        digit_count: usize,
        public_rows: usize,
        plan: PreimageClaimPlan,
    },
    Decompose {
        small: bool,
        digit_count: Option<usize>,
        input_rows: Vec<usize>,
        layout: GpuCompactDecompositionLayout,
        /// Compact hash sampling: the output type of a seeded
        /// COEFF gadget source generated per admitted range instead of a
        /// prepared input. The seed never enters admission identity.
        hash: Option<ConcreteMatrixType>,
    },
    Negate,
    Transpose,
    Tensor {
        right_rows: usize,
        right_columns: usize,
        groups: Option<Vec<Vec<usize>>>,
    },
    Slice {
        rows: std::ops::Range<usize>,
        columns: std::ops::Range<usize>,
    },
    SumRows(Vec<Vec<usize>>),
    Add,
    ConcatRows,
    ConcatColumns {
        diagonal: bool,
        rows: usize,
        columns: usize,
        offsets: Vec<(usize, usize, usize)>,
    },
    AddRowBlocks,
    Subtract,
    Scale(BigInt),
    Automorphism(usize),
    Multiply {
        scales_left: bool,
    },
    Accumulate {
        products: Vec<(BigInt, bool)>,
        bias: bool,
        rows: usize,
    },
}

pub(super) type PolynomialReadbackKey = (String, usize, usize, bool, bool);

pub(super) fn polynomial_readback_key(
    parameters: &GpuDCRTPolyParams,
    level: usize,
    evaluation: bool,
    input_ntt: bool,
) -> PolynomialReadbackKey {
    (
        parameters.modulus().to_string(),
        parameters.ring_dimension() as usize,
        level,
        evaluation,
        input_ntt,
    )
}

pub(super) fn polynomial_readback_claims(
    plans: &std::collections::HashMap<PolynomialReadbackKey, Vec<GpuTracedClaim>>,
    parameters: &GpuDCRTPolyParams,
    level: usize,
    evaluation: bool,
    input_ntt: bool,
) -> Option<Vec<GpuTracedClaim>> {
    plans.get(&polynomial_readback_key(parameters, level, evaluation, input_ntt)).cloned()
}

pub(super) fn polynomial_readback_evaluation(kind: &NodeKind) -> Option<bool> {
    match kind {
        NodeKind::PolynomialValues { evaluation } => Some(*evaluation),
        NodeKind::ThresholdDecode { .. } | NodeKind::ExtractCoefficient { .. } => Some(false),
        _ => None,
    }
}

/// One admitted, not yet consumed, invocation of the fleet backend for
/// diagnostics and estimation. The seed and operand payloads are not included.
#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize)]
pub struct GpuAdmittedInvocationSummary {
    pub operation: &'static str,
    pub rows: usize,
    pub columns: usize,
    pub compact_output: bool,
    /// Ordinary operands in the lowered operation's order (left, then right).
    /// IDs are first-use ordinals within this admitted batch, not process-local
    /// matrix IDs. Equal ordinals establish shared inputs across invocations;
    /// equal shapes alone do not. These labels are not native slot identities.
    pub input_owners: Vec<usize>,
    /// Uses the same batch-local identity namespace as ordinary operands.
    pub compact_input_owner: Option<usize>,
    pub plan: crate::gpu_memory::GpuAdmittedPlanSummary,
}

/// Per-invocation data that is not part of the admitted identity: the
/// production hash seed, or the artifact payload bytes of an import.
#[derive(Clone, Debug)]
pub(super) enum ExecutionPayload {
    None,
    Seed(GpuRngSeed),
    Bytes(Arc<Vec<u8>>),
    /// Production preimage operands: the device trapdoors, the resident public
    /// matrix, the staged or resident target, and the production seed.
    Preimage(Arc<PreimagePayload>),
}

pub(super) struct PreimagePayload {
    pub trapdoors: Arc<Vec<GpuDCRTTrapdoor>>,
    pub public: GpuFleetMatrix,
    pub target: PolyMatrixColumnData<GpuFleetMatrix>,
    pub target_global_column_start: usize,
    pub seed: [u8; 32],
}

impl std::fmt::Debug for PreimagePayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreimagePayload")
            .field("public", &self.public.size())
            .field("target_columns", &self.target.columns())
            .finish()
    }
}

/// Exact native claims of one preimage invocation class, recorded before the
/// inventory sealed by tracing one column; native layout queries specialize widths.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PreimageClaimPlan {
    /// Claims of the destination's hard-cutoff plan (after its payload owner).
    pub destination: Vec<GpuTracedClaim>,
    /// One-column claim order for materializing a staged target tile.
    pub tile: Vec<GpuTracedClaim>,
    /// One-column claim order for a candidate attempt with a prepared covariance cache.
    pub attempt: [Vec<GpuTracedClaim>; 16],
    pub attempts: usize,
    /// Native RNS staging width of one full-level polynomial, so pilots stage
    /// a zero target through the same loader as production.
    pub bytes_per_poly: usize,
}

impl PreimageClaimPlan {
    /// The trace fixes claim order; native planners supply width-dependent
    /// spans. Cached covariance and cutoff metadata are separate fixed owners.
    pub(super) fn attempt_claims(
        &self,
        parameters: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        output_rows: usize,
        bound: &num_bigint::BigUint,
    ) -> Result<[Vec<GpuTracedClaim>; 16], String> {
        let mut workspaces = parameters
            .p1_sampling_workspaces(rows, columns, true)?
            .into_iter()
            // Batched gadget sampling writes digits directly, without scratch.
            .filter(|layout| layout.bytes != 0);
        let mut phases: [Vec<GpuTracedClaim>; 16] = Default::default();
        for (phase, recorded) in phases.iter_mut().zip(&self.attempt) {
            *phase = recorded
                .iter()
                .map(|claim| {
                    if claim.kind() == GpuPreparedSlotKind::SamplerWorkspace {
                        Ok(GpuTracedClaim::workspace(
                            workspaces
                                .next()
                                .ok_or("preimage trace and sampler workspace planners differ")?,
                        ))
                    } else if claim.kind() == GpuPreparedSlotKind::CompactWorkspace {
                        Ok(GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                            kind: GpuPreparedSlotKind::CompactWorkspace,
                            bytes: GpuSmallMatrix::allocation_bytes(
                                parameters,
                                output_rows,
                                columns,
                                bound,
                            )
                            .map_err(|error| error.to_string())?,
                            alignment: claim.alignment(),
                        }))
                    } else {
                        Ok(claim.with_columns(columns))
                    }
                })
                .collect::<Result<Vec<_>, String>>()?;
        }
        Ok(phases)
    }

    /// Shared metadata of the residual's two-product native batch. Per-job
    /// owners/events are supplied by the recorded residual phase; these claims
    /// occur once for the whole batch and are queried without GPU execution.
    pub(super) fn residual_batch_metadata(
        parameters: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
        jobs: usize,
    ) -> Result<Vec<GpuTracedClaim>, String> {
        if jobs <= 1 {
            return Ok(Vec::new());
        }
        let workspace = parameters.matrix_batch_workspace_bytes(
            parameters.crt_depth() - 1,
            (rows, columns),
            jobs,
            2,
            mxx_primitives::poly::dcrt::gpu::GpuMatrixBatchOperation::Accumulate,
            true,
        )?;
        let mut claims = Vec::new();
        if workspace.additional_bytes != 0 {
            claims.push(GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::BatchWorkspace,
                bytes: workspace.additional_bytes,
                alignment: workspace.alignment,
            }));
        }
        if workspace.pinned_bytes != 0 {
            claims.push(GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::PinnedHost,
                bytes: workspace.pinned_bytes,
                alignment: workspace.alignment,
            }));
            claims.push(GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            }));
        }
        Ok(claims)
    }

    pub(super) fn tile_claims(
        &self,
        parameters: &GpuDCRTPolyParams,
        rows: usize,
        columns: usize,
    ) -> Result<Vec<GpuTracedClaim>, String> {
        let transfer =
            parameters.rns_transfer_workspace(parameters.crt_depth() - 1, rows, columns)?;
        Ok(self
            .tile
            .iter()
            .map(|claim| match claim.kind() {
                GpuPreparedSlotKind::PinnedHost => {
                    GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::PinnedHost,
                        bytes: transfer.bytes,
                        alignment: 1,
                    })
                }
                GpuPreparedSlotKind::TransferWorkspace => GpuTracedClaim::workspace(transfer),
                _ => claim.with_columns(columns),
            })
            .collect())
    }
}

/// Identity of a preimage invocation class whose claim plan was traced.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct PreimagePlanKey {
    pub modulus: String,
    pub ring_dimension: usize,
    pub rows: usize,
    pub public_rows: usize,
    pub bound: String,
    pub sigma_bits: u64,
    pub gadget_base: String,
    pub digit_count: usize,
}

/// Existing fixed-owner claim plan, split at the real transfer boundary.
#[derive(Clone, Debug)]
pub(super) struct TrapdoorClaimPlan {
    pub sample: Vec<GpuTracedClaim>,
    pub export: Vec<GpuTracedClaim>,
    pub import: Vec<GpuTracedClaim>,
}

/// Identity of a trapdoor sampling class whose claim plan was traced.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct TrapdoorPlanKey {
    pub modulus: String,
    pub ring_dimension: usize,
    pub rows: usize,
    pub columns: usize,
    pub sigma_bits: u64,
    pub gadget_base: String,
    pub digit_count: usize,
}

/// Holds exact ordered native claims from the accepted inventory around a
/// host-driven step of an admitted operation. Prepared backing is already
/// charged by setup; this establishes slot exclusivity and the codec/sampler
/// resources, and releases them when the step's owners are gone.
/// Materialize the evaluation-format target tile for one admitted range from a
/// staged or resident preimage target on the destination's context.
pub(super) fn materialize_preimage_tile(
    params: &GpuDCRTPolyParams,
    target: &PolyMatrixColumnData<GpuFleetMatrix>,
    rows: usize,
    start: usize,
    end: usize,
) -> Result<GpuDCRTPolyMatrix, String> {
    let mut tile = match target.subrange(start, end) {
        PolyMatrixColumnData::CpuStaging { bytes, start, end, .. } => {
            <GpuDCRTPolyMatrix as mxx_primitives::matrix::PolyMatrix>::from_cpu_staging_columns(
                params, &bytes, start, end,
            )
        }
        PolyMatrixColumnData::Resident { value, start, end } => {
            let shard = value
                .shards()
                .iter()
                .find(|shard| {
                    shard.value.params() == params &&
                        shard.global_column_start <= start &&
                        end - shard.global_column_start <= shard.value.col_size()
                })
                .ok_or("preimage target is not resident on the destination context")?;
            shard
                .value
                .column_view(start - shard.global_column_start..end - shard.global_column_start)?
                .copy(None)?
        }
    };
    if tile.size() != (rows, end - start) {
        return Err("preimage target tile shape differs from the public matrix".into());
    }
    if !tile.is_ntt() {
        tile.ntt_all_in_place();
    }
    Ok(tile)
}

impl ExecutionPayload {
    fn seed(&self) -> Option<GpuRngSeed> {
        match self {
            Self::Seed(seed) => Some(*seed),
            _ => None,
        }
    }
    fn bytes(&self) -> Option<&Arc<Vec<u8>>> {
        match self {
            Self::Bytes(bytes) => Some(bytes),
            _ => None,
        }
    }
}

/// Leaf matrix type of one wire type, unwrapping every family layer.
fn leaf_matrix_type(wire_type: &ConcreteWireType) -> Option<&ConcreteMatrixType> {
    let mut leaf = wire_type;
    while let ConcreteWireType::IndexedFamily { element, .. } = leaf {
        leaf = element;
    }
    leaf.matrix_type()
}

/// Orient one multiplication's operands: the scalable input, its fixed operand,
/// and whether the caller's left operand is the scalable one.
fn scalable_product<'a>(
    left: &'a GpuFleetMatrix,
    right: &'a GpuFleetMatrix,
) -> (&'a GpuFleetMatrix, &'a GpuFleetMatrix, bool) {
    let scales_left =
        gpu_matrix_multiply_scales_left(left.rows, left.columns, right.rows, right.columns);
    if scales_left { (left, right, true) } else { (right, left, false) }
}

/// Number of output blocks one declared RNS conversion produces, or `None` for
/// parameters the conversion does not support.
fn rns_conversion_groups(
    conversion: GpuMatrixRnsConversion,
    source_moduli: usize,
) -> Option<usize> {
    match conversion {
        GpuMatrixRnsConversion::Up { digit_size, .. } if digit_size != 0 => {
            Some(source_moduli.div_ceil(digit_size))
        }
        GpuMatrixRnsConversion::Down { plaintext_modulus } if plaintext_modulus >= 2 => Some(1),
        _ => None,
    }
}

impl PreparedOperation {
    /// Bind the plan discovered by explicit warmup. Metadata planning and real
    /// sampling use the same cache key; neither may probe on a cache miss.
    fn preimage(
        backend: &GpuDcrtBackend,
        ty: &ConcreteMatrixType,
        bound: num_bigint::BigUint,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        public_rows: usize,
    ) -> Result<Self, PolyBackendError> {
        let key = PreimagePlanKey {
            modulus: ty.modulus.to_string(),
            ring_dimension: ty.ring_dimension,
            rows: ty.rows,
            public_rows,
            bound: bound.to_string(),
            sigma_bits: sigma.to_bits(),
            gadget_base: gadget_base.to_string(),
            digit_count,
        };
        let plan = backend.preimage_plans.get(&key).ok_or_else(|| {
            PolyBackendError::GpuSubmission(format!(
                "explicit graph warmup required: missing preimage resource plan for {key:?}"
            ))
        })?;
        Ok(Self::Preimage {
            ty: ty.clone(),
            bound,
            sigma_bits: sigma.to_bits(),
            gadget_base: gadget_base.clone(),
            digit_count,
            public_rows,
            plan: plan.clone(),
        })
    }

    pub(super) fn kind_name(&self) -> &'static str {
        match self {
            Self::Constant { .. } => "Constant",
            Self::Sample { .. } => "Sample",
            Self::Polynomial { .. } => "Polynomial",
            Self::PolynomialReadback { .. } => "PolynomialReadback",
            Self::Hash { .. } => "Hash",
            Self::ModulusConversion { .. } => "ModulusConversion",
            Self::CrtRecompose { .. } => "CrtRecompose",
            Self::RnsConversion { .. } => "RnsConversion",
            Self::CenteredRebase { .. } => "CenteredRebase",
            Self::CenteredExtendCompact { .. } => "CenteredExtendCompact",
            Self::MultiplyCompact { .. } => "MultiplyCompact",
            Self::ImportMatrix { .. } => "ImportMatrix",
            Self::ImportCompact { .. } => "ImportCompact",
            Self::ImportStaging { .. } => "ImportStaging",
            Self::Preimage { .. } => "Preimage",
            Self::Decompose { hash: Some(_), .. } => "HashDecompose",
            Self::Decompose { hash: None, .. } => "Decompose",
            Self::Negate => "Negate",
            Self::Transpose => "Transpose",
            Self::Tensor { .. } => "Tensor",
            Self::Slice { .. } => "Slice",
            Self::SumRows(_) => "SumRows",
            Self::Add => "Add",
            Self::ConcatRows => "ConcatRows",
            Self::ConcatColumns { .. } => "ConcatColumns",
            Self::AddRowBlocks => "AddRowBlocks",
            Self::Subtract => "Subtract",
            Self::Scale(_) => "Scale",
            Self::Automorphism(_) => "Automorphism",
            Self::Multiply { .. } => "Multiply",
            Self::Accumulate { .. } => "Accumulate",
        }
    }

    /// Single IR → prepared-operation mapping for the operations that are
    /// determined by the validated wire types and the scope parameters.
    ///
    /// Payload fields that only execution can know (sampled coefficients,
    /// hash seeds and tag lengths, artifact bytes, traced claim plans) are
    /// deliberately not derived here: they are bound when the invocation is
    /// executed, exactly like `ExecutionPayload`. A `None` result means the
    /// node has no type-determined prepared matrix operation (scalar nodes,
    /// artifact codecs, the traced preimage sampler and
    /// optimizer-only fusions such as row sums, row-block adds and tensor sums,
    /// which are fused in the executor).
    ///
    /// Both the GPU inventory and the admitted production path build their
    /// operations from this mapping, so a kind is interpreted in one place.
    pub(super) fn from_ir(
        kind: &NodeKind,
        arguments: &[ConcreteWireType],
        outputs: &[ConcreteWireType],
        bindings: &ParamEnv,
        parameters: &GpuDCRTPolyParams,
    ) -> Result<Option<Self>, PolyBackendError> {
        let matrix = |ty: &ConcreteWireType| leaf_matrix_type(ty).cloned();
        let index = |expression: &mxx_ir_core::expr::IntExpr| -> Result<usize, PolyBackendError> {
            expression
                .evaluate(bindings)
                .ok()
                .and_then(|value| value.to_usize())
                .ok_or(PolyBackendError::InvalidInteger)
        };
        let scalar = |expression: &mxx_ir_core::expr::IntExpr| -> Result<BigInt, PolyBackendError> {
            expression.evaluate(bindings).map_err(|_| PolyBackendError::InvalidInteger)
        };
        let output = || -> Result<ConcreteMatrixType, PolyBackendError> {
            outputs.first().and_then(&matrix).ok_or(PolyBackendError::InvalidConstantShape)
        };
        Ok(match kind {
            NodeKind::ConstantMatrix { value, .. } => {
                Some(Self::constant(&output()?, value, bindings, parameters)?)
            }
            NodeKind::UniformResidueSample { .. } => Some(Self::uniform(
                &output()?,
                &SampleRange { minimum: BigInt::from(0), maximum: &output()?.modulus - 1 },
                parameters,
            )?),
            NodeKind::UniformIntervalSample { range, .. } => Some(Self::uniform(
                &output()?,
                &SampleRange { minimum: scalar(&range.minimum)?, maximum: scalar(&range.maximum)? },
                parameters,
            )?),
            NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => Some(Self::gaussian(
                &output()?,
                sigma
                    .evaluate_f64(bindings)
                    .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?,
                &scalar(max_coefficient_bound)?,
            )?),
            NodeKind::MatrixBinary(operation) => {
                let (Some(left), Some(right)) =
                    (arguments.first().and_then(&matrix), arguments.get(1).and_then(&matrix))
                else {
                    return Err(PolyBackendError::InvalidConstantShape);
                };
                match operation {
                    MatrixBinaryOp::Add => Some(Self::Add),
                    MatrixBinaryOp::Subtract => Some(Self::Subtract),
                    MatrixBinaryOp::Multiply => Some(Self::Multiply {
                        scales_left: gpu_matrix_multiply_scales_left(
                            left.rows,
                            left.columns,
                            right.rows,
                            right.columns,
                        ),
                    }),
                }
            }
            NodeKind::MatrixNegate => Some(Self::Negate),
            NodeKind::MatrixScale { scalar: expression } => Some(Self::Scale(scalar(expression)?)),
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                let rows = output()?.rows;
                let mut products = Vec::with_capacity(coefficients.len());
                for (product, coefficient) in coefficients.iter().enumerate() {
                    let (Some(left), Some(right)) = (
                        arguments.get(2 * product).and_then(&matrix),
                        arguments.get(2 * product + 1).and_then(&matrix),
                    ) else {
                        return Err(PolyBackendError::InvalidConstantShape);
                    };
                    products.push((
                        scalar(coefficient)?,
                        gpu_matrix_multiply_scales_left(
                            left.rows,
                            left.columns,
                            right.rows,
                            right.columns,
                        ),
                    ));
                }
                Some(Self::Accumulate { products, bias: *has_bias, rows })
            }
            NodeKind::MatrixMulSmallRhs => {
                if arguments.first().and_then(&matrix).is_none() {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                let Some(right) = arguments.get(1).and_then(|ty| match ty {
                    ConcreteWireType::SmallMatrix { matrix, .. } |
                    ConcreteWireType::Preimage { matrix, .. } => Some(matrix.clone()),
                    _ => None,
                }) else {
                    return Err(PolyBackendError::InvalidConstantShape);
                };
                // The inner dimension is the compact RHS's own row count, so
                // `output_rows` still rejects a compact product whose operand
                // column count disagrees with the RHS the node declares.
                Some(Self::MultiplyCompact { columns: right.columns, inner: right.rows })
            }
            NodeKind::GadgetDecompose { digit_count, small, .. } => {
                let digits = index(digit_count)?;
                if digits == 0 {
                    return Err(PolyBackendError::InvalidInteger);
                }
                let ty = output()?;
                if !ty.rows.is_multiple_of(digits) {
                    return Err(PolyBackendError::InvalidInteger);
                }
                Some(Self::decompose(
                    *small,
                    Some(digits),
                    vec![ty.rows / digits],
                    parameters,
                    None,
                )?)
            }
            NodeKind::HashSample { variant, digit_count, .. } => {
                use mxx_ir_core::node::HashVariant;
                let ty = output()?;
                Some(match variant {
                    HashVariant::Plain => Self::Hash { ty },
                    HashVariant::Decomposed | HashVariant::SmallDecomposed => {
                        let digits =
                            index(digit_count.as_ref().expect("validated hash decomposition"))?;
                        Self::decompose(
                            *variant == HashVariant::SmallDecomposed,
                            Some(digits),
                            vec![ty.rows / digits],
                            parameters,
                            Some(ty),
                        )?
                    }
                })
            }
            NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients, .. } => {
                let destination = output()?;
                let plaintext_moduli = plaintext_moduli
                    .par_iter()
                    .map(|p| scalar(p).ok().and_then(|value| value.to_u64()).filter(|p| *p != 0))
                    .collect::<Option<Vec<_>>>()
                    .ok_or(PolyBackendError::InvalidInteger)?;
                let reconstruction_coefficients = reconstruction_coefficients
                    .par_iter()
                    .map(|coefficient| scalar(coefficient))
                    .collect::<Result<Vec<_>, _>>()?;
                Some(Self::crt_recompose(
                    destination,
                    plaintext_moduli,
                    reconstruction_coefficients,
                ))
            }
            NodeKind::RnsModUp { source_moduli, digit_size, normalize, .. } => {
                let source = arguments
                    .first()
                    .and_then(&matrix)
                    .ok_or(PolyBackendError::InvalidConstantShape)?;
                let conversion =
                    GpuMatrixRnsConversion::Up { digit_size: *digit_size, normalize: *normalize };
                Some(Self::rns_conversion(
                    output()?,
                    source_moduli.clone(),
                    conversion,
                    source.rows,
                    source.columns,
                    parameters,
                )?)
            }
            NodeKind::RnsModDown { source_moduli, plaintext_modulus, .. } => {
                let source = arguments
                    .first()
                    .and_then(&matrix)
                    .ok_or(PolyBackendError::InvalidConstantShape)?;
                let plaintext_modulus =
                    scalar(plaintext_modulus)?.to_u64().ok_or(PolyBackendError::InvalidInteger)?;
                let conversion = GpuMatrixRnsConversion::Down { plaintext_modulus };
                Some(Self::rns_conversion(
                    output()?,
                    source_moduli.clone(),
                    conversion,
                    source.rows,
                    source.columns,
                    parameters,
                )?)
            }
            NodeKind::ModulusSwitch { .. } => {
                Some(Self::modulus_conversion(output()?, GpuMatrixModulusConversion::Round))
            }
            NodeKind::ModulusReduce { .. } => {
                Some(Self::modulus_conversion(output()?, GpuMatrixModulusConversion::Reduce))
            }
            NodeKind::CenteredExtend { .. } => {
                let destination = output()?;
                match arguments.first() {
                    Some(
                        ConcreteWireType::SmallMatrix { max_coefficient_bound, .. } |
                        ConcreteWireType::Preimage { max_coefficient_bound, .. },
                    ) => Some(Self::CenteredExtendCompact {
                        destination,
                        bound: max_coefficient_bound
                            .to_biguint()
                            .ok_or(PolyBackendError::InvalidInteger)?,
                    }),
                    _ => Some(Self::modulus_conversion(
                        destination,
                        GpuMatrixModulusConversion::CenteredExtend,
                    )),
                }
            }
            NodeKind::BlockModSwitch { plaintext_modulus, .. } => {
                let plaintext_modulus =
                    scalar(plaintext_modulus)?.to_u64().ok_or(PolyBackendError::InvalidInteger)?;
                Some(Self::modulus_conversion(
                    output()?,
                    GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus },
                ))
            }
            NodeKind::CenteredRebase { .. } => Some(Self::centered_rebase(output()?)),
            NodeKind::Transpose => Some(Self::Transpose),
            NodeKind::Slice { rows, columns } => {
                let source = arguments
                    .first()
                    .and_then(&matrix)
                    .ok_or(PolyBackendError::InvalidConstantShape)?;
                let rows = match rows {
                    Some(range) => index(&range.start)?..index(&range.end)?,
                    None => 0..source.rows,
                };
                let columns = match columns {
                    Some(range) => index(&range.start)?..index(&range.end)?,
                    None => 0..source.columns,
                };
                Some(Self::Slice { rows, columns })
            }
            NodeKind::Tensor => {
                let right = arguments
                    .get(1)
                    .and_then(&matrix)
                    .ok_or(PolyBackendError::InvalidConstantShape)?;
                Some(Self::Tensor {
                    right_rows: right.rows,
                    right_columns: right.columns,
                    groups: None,
                })
            }
            NodeKind::RingAutomorphism { index: expression } => {
                Some(Self::Automorphism(index(expression)?))
            }
            NodeKind::Concat { axis } => {
                let inputs = arguments
                    .iter()
                    .map(|ty| matrix(ty).ok_or(PolyBackendError::InvalidConstantShape))
                    .collect::<Result<Vec<_>, _>>()?;
                if inputs.is_empty() {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                if *axis == ConcatAxis::Rows {
                    Some(Self::ConcatRows)
                } else {
                    let diagonal = *axis == ConcatAxis::Diagonal;
                    let mut rows = 0usize;
                    let mut columns = 0usize;
                    let mut offsets = Vec::with_capacity(inputs.len());
                    for input in &inputs {
                        let end = columns
                            .checked_add(input.columns)
                            .ok_or(PolyBackendError::InvalidConstantShape)?;
                        offsets.push((if diagonal { rows } else { 0 }, columns, end));
                        if diagonal {
                            rows = rows
                                .checked_add(input.rows)
                                .ok_or(PolyBackendError::InvalidConstantShape)?;
                        }
                        columns = end;
                    }
                    Some(Self::ConcatColumns {
                        diagonal,
                        rows: if diagonal { rows } else { inputs[0].rows },
                        columns,
                        offsets,
                    })
                }
            }
            _ => None,
        })
    }

    fn uniform(
        ty: &ConcreteMatrixType,
        range: &SampleRange,
        parameters: &GpuDCRTPolyParams,
    ) -> Result<Self, PolyBackendError> {
        let maximum = BigInt::from(parameters.modulus().as_ref().clone()) - 1;
        let distribution = if range.minimum == BigInt::from(-1) && range.maximum == BigInt::from(1)
        {
            GpuMatrixSampleDist::Ternary
        } else if range.minimum == BigInt::from(0) && range.maximum == BigInt::from(1) {
            GpuMatrixSampleDist::Bit
        } else if range.minimum == BigInt::from(0) && range.maximum == maximum {
            GpuMatrixSampleDist::Uniform
        } else {
            return Err(PolyBackendError::UnsupportedUniformRange {
                minimum: range.minimum.clone(),
                maximum: range.maximum.clone(),
            });
        };
        Ok(Self::Sample {
            ty: ty.clone(),
            distribution,
            sigma_bits: 0.0_f64.to_bits(),
            max_coefficient_bound: BigInt::from(u64::MAX),
        })
    }

    fn gaussian(
        ty: &ConcreteMatrixType,
        sigma: f64,
        max_coefficient_bound: &BigInt,
    ) -> Result<Self, PolyBackendError> {
        if max_coefficient_bound.to_biguint().is_none() {
            return Err(PolyBackendError::InvalidInteger);
        }
        if !sigma.is_finite() || sigma < 0.0 {
            return Err(PolyBackendError::GpuSubmission(
                "Gaussian sigma must be finite and nonnegative".into(),
            ));
        }
        Ok(Self::Sample {
            ty: ty.clone(),
            distribution: GpuMatrixSampleDist::Gauss,
            sigma_bits: sigma.to_bits(),
            max_coefficient_bound: max_coefficient_bound.clone(),
        })
    }

    /// Resolve a constant once from its declared expressions. Inventory and
    /// production use the same GPU generator or polynomial-upload description.
    fn constant(
        ty: &ConcreteMatrixType,
        value: &ConstantMatrix,
        env: &ParamEnv,
        parameters: &GpuDCRTPolyParams,
    ) -> Result<Self, PolyBackendError> {
        let index = |expression: &mxx_ir_core::expr::IntExpr| {
            expression
                .evaluate(env)
                .ok()
                .and_then(|value| value.to_usize())
                .ok_or(PolyBackendError::InvalidInteger)
        };
        let polynomial = match value {
            ConstantMatrix::PowerOfBase { base, exponent } if ty.rows == 1 && ty.columns == 1 => {
                let base = base.evaluate(env).map_err(|_| PolyBackendError::InvalidInteger)?;
                let exponent = exponent
                    .evaluate(env)
                    .ok()
                    .and_then(|value| value.to_u32())
                    .ok_or(PolyBackendError::InvalidInteger)?;
                Some(vec![base.modpow(&BigInt::from(exponent), &ty.modulus)])
            }
            ConstantMatrix::Rotation { exponent } if ty.rows == 1 && ty.columns == 1 => {
                let exponent = index(exponent)?;
                if exponent >= ty.ring_dimension {
                    return Err(PolyBackendError::InvalidInteger);
                }
                let mut coefficients = vec![BigInt::from(0); exponent + 1];
                coefficients[exponent] = BigInt::from(1);
                Some(coefficients)
            }
            ConstantMatrix::Polynomial { coefficients } if ty.rows == 1 && ty.columns == 1 => {
                if coefficients.len() > ty.ring_dimension {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                Some(
                    coefficients
                        .par_iter()
                        .map(|coefficient| {
                            coefficient.evaluate(env).map_err(|_| PolyBackendError::InvalidInteger)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                )
            }
            _ => None,
        };
        if let Some(coefficients) = polynomial {
            return Ok(Self::Polynomial { ty: ty.clone(), coefficients, evaluation: false });
        }
        let value = match value {
            ConstantMatrix::Zero => GpuMatrixRangeConstant::Zero { total_columns: ty.columns },
            ConstantMatrix::Identity if ty.rows == ty.columns => GpuMatrixRangeConstant::Identity,
            ConstantMatrix::UnitRow { index: expression } if ty.rows == 1 => {
                let index = index(expression)?;
                if index >= ty.columns {
                    return Err(PolyBackendError::InvalidInteger);
                }
                GpuMatrixRangeConstant::UnitRow { total_columns: ty.columns, index }
            }
            ConstantMatrix::UnitColumn { index: expression } if ty.columns == 1 => {
                let index = index(expression)?;
                if index >= ty.rows {
                    return Err(PolyBackendError::InvalidInteger);
                }
                GpuMatrixRangeConstant::UnitColumn { index }
            }
            ConstantMatrix::Gadget { base, small } => {
                if ty.rows == 0 || !ty.columns.is_multiple_of(ty.rows) {
                    return Err(PolyBackendError::InvalidInteger);
                }
                let digits = ty.columns / ty.rows;
                let base = base.evaluate(env).map_err(|_| PolyBackendError::InvalidInteger)?;
                DeviceBackend::validate_gadget_layout_for_params(
                    parameters, &base, digits, *small,
                )?;
                GpuMatrixRangeConstant::Gadget { small: *small, digit_count: Some(digits) }
            }
            _ => {
                return Err(PolyBackendError::GpuSubmission(
                    "constant has no prepared range generator".into(),
                ))
            }
        };
        Ok(Self::Constant { ty: ty.clone(), value })
    }

    /// Compact decomposition operation. The layout comes from the same native
    /// planner the range runner uses; `hash` marks a seeded COEFF gadget source.
    fn decompose(
        small: bool,
        digit_count: Option<usize>,
        input_rows: Vec<usize>,
        parameters: &GpuDCRTPolyParams,
        hash: Option<ConcreteMatrixType>,
    ) -> Result<Self, PolyBackendError> {
        let layout = parameters
            .compact_decomposition_layout(small, digit_count)
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        Ok(Self::decompose_with_layout(small, digit_count, input_rows, layout, hash))
    }

    /// Same operation for a caller that already queried the native layout.
    fn decompose_with_layout(
        small: bool,
        digit_count: Option<usize>,
        input_rows: Vec<usize>,
        layout: GpuCompactDecompositionLayout,
        hash: Option<ConcreteMatrixType>,
    ) -> Self {
        Self::Decompose { small, digit_count, input_rows, layout, hash }
    }

    fn crt_recompose(
        destination: ConcreteMatrixType,
        plaintext_moduli: Vec<u64>,
        reconstruction_coefficients: Vec<BigInt>,
    ) -> Self {
        Self::CrtRecompose { destination, plaintext_moduli, reconstruction_coefficients }
    }

    /// Declared basis relation of one destination context: exactly the relation
    /// the admitted production path requires before admitting the operation.
    fn validate_rns_basis(
        target: &GpuDCRTPolyParams,
        source_moduli: &[u64],
        conversion: GpuMatrixRnsConversion,
    ) -> Result<(), PolyBackendError> {
        let valid = match conversion {
            GpuMatrixRnsConversion::Up { .. } => {
                source_moduli.iter().all(|q| target.moduli().contains(q))
            }
            GpuMatrixRnsConversion::Down { plaintext_modulus } => {
                target.crt_depth() < source_moduli.len() &&
                    target.moduli().iter().all(|q| source_moduli.contains(q)) &&
                    source_moduli.iter().filter(|q| !target.moduli().contains(q)).all(|q| {
                        *q > 1 &&
                            mxx_primitives::utils::mod_inverse(plaintext_modulus % q, *q)
                                .is_some()
                    })
            }
        };
        if !valid {
            return Err(PolyBackendError::BasisConversion(
                "invalid declared RNS basis relation".into(),
            ));
        }
        Ok(())
    }

    /// RNS basis conversion for one resolved destination context, including the
    /// conversion's group/row relation and its declared basis.
    fn rns_conversion(
        destination: ConcreteMatrixType,
        source_moduli: Vec<u64>,
        conversion: GpuMatrixRnsConversion,
        source_rows: usize,
        source_columns: usize,
        parameters: &GpuDCRTPolyParams,
    ) -> Result<Self, PolyBackendError> {
        if source_moduli.is_empty() || source_moduli.len() > 64 {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let groups = rns_conversion_groups(conversion, source_moduli.len())
            .ok_or(PolyBackendError::InvalidConstantShape)?;
        if source_rows.checked_mul(groups) != Some(destination.rows) ||
            source_columns != destination.columns
        {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        Self::validate_rns_basis(parameters, &source_moduli, conversion)?;
        Ok(Self::RnsConversion { destination, source_moduli, conversion })
    }

    fn modulus_conversion(
        destination: ConcreteMatrixType,
        conversion: GpuMatrixModulusConversion,
    ) -> Self {
        Self::ModulusConversion { destination, conversion }
    }

    fn centered_rebase(destination: ConcreteMatrixType) -> Self {
        Self::CenteredRebase { destination }
    }

    pub(super) fn compact_bound(&self) -> Option<&num_bigint::BigUint> {
        match self {
            Self::Decompose { layout, .. } => Some(&layout.max_coefficient_bound),
            Self::CenteredExtendCompact { bound, .. } |
            Self::ImportCompact { bound, .. } |
            Self::Preimage { bound, .. } => Some(bound),
            _ => None,
        }
    }
    fn output_request(
        &self,
        slot: GpuPreparedSlotIdentity,
        parameters: &GpuDCRTPolyParams,
        level: usize,
        rows: usize,
        columns: usize,
        evaluation: bool,
    ) -> Result<Option<GpuPreparedRequest>, PolyBackendError> {
        if let Some(bound) = self.compact_bound() {
            if slot.kind() != GpuPreparedSlotKind::CompactPayload {
                return Ok(None);
            }
            Ok(Some(
                slot.workspace_request(
                    GpuSmallMatrix::allocation_bytes(parameters, rows, columns, bound)
                        .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?,
                    256,
                ),
            ))
        } else if slot.kind() == GpuPreparedSlotKind::Matrix && slot.level() == Some(level) {
            Ok(Some(slot.matrix_request(rows, columns, evaluation)))
        } else {
            Ok(None)
        }
    }

    pub(super) fn fresh_type(&self) -> Option<&ConcreteMatrixType> {
        match self {
            Self::Constant { ty, .. } |
            Self::Sample { ty, .. } |
            Self::Polynomial { ty, .. } |
            Self::Hash { ty, .. } |
            Self::ImportMatrix { ty, .. } |
            Self::ImportCompact { ty, .. } |
            Self::ImportStaging { ty, .. } |
            Self::Preimage { ty, .. } |
            Self::Decompose { hash: Some(ty), .. } => Some(ty),
            _ => None,
        }
    }

    pub(super) fn is_hash(&self) -> bool {
        matches!(self, Self::Hash { .. } | Self::Decompose { hash: Some(_), .. })
    }

    pub(super) fn is_import(&self) -> bool {
        matches!(
            self,
            Self::ImportMatrix { .. } | Self::ImportCompact { .. } | Self::ImportStaging { .. }
        )
    }

    /// Exact byte length of the complete canonical payload an import consumes.
    pub(super) fn import_payload_len(&self) -> Result<usize, PolyBackendError> {
        let (ty, width) = match self {
            Self::ImportMatrix { ty, max_coefficient_bits, .. } => {
                let count = ty
                    .rows
                    .checked_mul(ty.columns)
                    .and_then(|count| count.checked_mul(ty.ring_dimension))
                    .ok_or(PolyBackendError::InvalidInteger)?;
                return count
                    .checked_mul(usize::from(*max_coefficient_bits))
                    .map(|bits| bits.div_ceil(8))
                    .ok_or(PolyBackendError::InvalidInteger);
            }
            Self::ImportCompact { ty, bound } => (ty, Self::compact_coefficient_width(bound)?),
            Self::ImportStaging { payload_len, .. } => return Ok(*payload_len),
            _ => return Err(PolyBackendError::InvalidConstantShape),
        };
        ty.rows
            .checked_mul(ty.columns)
            .and_then(|count| count.checked_mul(ty.ring_dimension))
            .and_then(|count| count.checked_mul(width))
            .ok_or(PolyBackendError::InvalidInteger)
    }

    /// Canonical compact coefficient width: one sign byte plus the magnitude bytes.
    fn compact_coefficient_width(bound: &num_bigint::BigUint) -> Result<usize, PolyBackendError> {
        usize::try_from(bound.bits().div_ceil(8))
            .map(|bytes| bytes.max(1) + 1)
            .map_err(|_| PolyBackendError::InvalidInteger)
    }

    /// Validate the caller-supplied production payload against the operation.
    pub(super) fn production_payload(
        &self,
        supplied: ExecutionPayload,
    ) -> Result<ExecutionPayload, PolyBackendError> {
        match (&supplied, self.is_hash(), self.is_import()) {
            (ExecutionPayload::Preimage(_), _, _) if matches!(self, Self::Preimage { .. }) => {
                Ok(supplied)
            }
            (ExecutionPayload::Seed(_), true, _) => Ok(supplied),
            (ExecutionPayload::Bytes(bytes), _, true)
                if bytes.len() == self.import_payload_len()? =>
            {
                Ok(supplied)
            }
            (ExecutionPayload::None, false, false) => Ok(if matches!(self, Self::Sample { .. }) {
                ExecutionPayload::Seed(GpuRngSeed::from_bytes(rand::random()))
            } else {
                ExecutionPayload::None
            }),
            _ => Err(PolyBackendError::GpuSubmission(
                "incorrect production hash seed or import payload".into(),
            )),
        }
    }

    pub(super) fn requires_evaluation(&self) -> bool {
        matches!(
            self,
            Self::Constant { .. } |
                Self::Sample { .. } |
                Self::Hash { .. } |
                Self::Polynomial { .. } |
                Self::Scale(_) |
                Self::Automorphism(_) |
                Self::Add |
                Self::ConcatRows |
                Self::ConcatColumns { .. } |
                Self::AddRowBlocks |
                Self::Subtract |
                Self::Multiply { .. } |
                Self::MultiplyCompact { .. } |
                Self::Accumulate { .. } |
                Self::Tensor { .. }
        )
    }

    pub(super) fn input_evaluation(&self, original: bool) -> bool {
        if matches!(self, Self::Preimage { .. }) {
            return true;
        }
        if let Self::ImportMatrix { evaluation, .. } | Self::ImportStaging { evaluation, .. } = self
        {
            return *evaluation;
        }
        if matches!(
            self,
            Self::CenteredRebase { .. } |
                Self::RnsConversion { .. } |
                Self::CrtRecompose { .. } |
                Self::Decompose { .. } |
                Self::CenteredExtendCompact { .. } |
                Self::ImportCompact { .. }
        ) || matches!(self,Self::ModulusConversion {conversion,..} if *conversion != GpuMatrixModulusConversion::Reduce)
        {
            false
        } else {
            self.requires_evaluation() || original
        }
    }

    pub(super) fn output_layout(
        &self,
        backend: &DeviceBackend,
        input: &GpuDCRTPolyParams,
        level: usize,
        evaluation: bool,
    ) -> Result<(GpuDCRTPolyParams, usize, bool), PolyBackendError> {
        if let Self::CenteredRebase { destination } |
        Self::ModulusConversion { destination, .. } |
        Self::RnsConversion { destination, .. } |
        Self::CrtRecompose { destination, .. } |
        Self::CenteredExtendCompact { destination, .. } = self
        {
            let parameters = backend.parameters(destination)?.clone();
            let level = parameters.crt_depth() - 1;
            let output_evaluation = if matches!(
                self,
                Self::ModulusConversion { conversion: GpuMatrixModulusConversion::Reduce, .. }
            ) {
                evaluation
            } else {
                true
            };
            Ok((parameters, level, output_evaluation))
        } else if matches!(self, Self::MultiplyCompact { .. }) {
            Ok((input.clone(), level, true))
        } else {
            Ok((input.clone(), level, evaluation))
        }
    }

    /// The scalable owner of a global output range. Column concatenation
    /// selects the corresponding original operand instead of redistributing it.
    pub(super) fn primary<'a, M>(
        &self,
        left: Option<&'a M>,
        right: &'a [M],
        start: usize,
    ) -> Option<&'a M> {
        if let Self::ConcatColumns { offsets, .. } = self {
            let index = offsets.partition_point(|&(_, _, end)| end <= start);
            if index == 0 { left } else { Some(&right[index - 1]) }
        } else {
            left
        }
    }

    pub(super) fn dependent_inputs<'a, M>(&self, right: &'a [M]) -> &'a [M] {
        if matches!(self, Self::ConcatColumns { .. }) { &[] } else { right }
    }

    pub(super) fn initialize_output(
        &self,
        parameters: &GpuDCRTPolyParams,
        level: usize,
        evaluation: bool,
        rows: usize,
        columns: usize,
    ) -> Result<PreparedMatrixValue, String> {
        if let Some(bound) = self.compact_bound() {
            return GpuSmallMatrix::new_zero(parameters, rows, columns, bound.clone())
                .map(PreparedMatrixValue::Compact)
                .map_err(|e| e.to_string());
        }
        Ok(PreparedMatrixValue::Matrix(
            if matches!(self, Self::ConcatColumns { diagonal: true, .. }) {
                GpuDCRTPolyMatrix::new_zero_with_state(parameters, rows, columns, level, evaluation)
            } else {
                GpuDCRTPolyMatrix::new_empty_with_state(
                    parameters, rows, columns, level, evaluation, None,
                )
            },
        ))
    }

    pub(super) fn output_rows<M: MatrixShape>(
        &self,
        scalable: Option<&M>,
        other: &[M],
    ) -> Result<usize, PolyBackendError> {
        if let Some(ty) = self.fresh_type() {
            return Ok(ty.rows);
        }
        if let Self::CenteredExtendCompact { destination, .. } = self {
            if scalable.is_some() || !other.is_empty() {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            return Ok(destination.rows);
        }
        if let Self::MultiplyCompact { inner, .. } = self {
            let [lhs] = other else {
                return Err(PolyBackendError::InvalidConstantShape);
            };
            if scalable.is_some() || lhs.shape().1 != *inner || *inner == 0 {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            return Ok(lhs.shape().0);
        }
        let scalable = scalable.ok_or(PolyBackendError::InvalidConstantShape)?;
        if let Self::Decompose { input_rows, layout, .. } = self {
            if input_rows
                .iter()
                .copied()
                .ne(std::iter::once(scalable.shape().0).chain(other.iter().map(|v| v.shape().0))) ||
                other.iter().any(|v| v.shape().1 != scalable.shape().1)
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            return input_rows
                .iter()
                .try_fold(0usize, |sum, rows| sum.checked_add(*rows))
                .and_then(|rows| rows.checked_mul(layout.rows_per_input_row))
                .ok_or(PolyBackendError::InvalidConstantShape);
        }
        if let Self::RnsConversion { destination, .. } = self {
            return Ok(destination.rows);
        }
        if let Self::ConcatColumns { rows, .. } = self {
            return Ok(*rows);
        }
        if matches!(self, Self::ConcatRows | Self::AddRowBlocks) {
            if other.iter().any(|block| block.shape().1 != scalable.shape().1) ||
                (*self == Self::AddRowBlocks && other.is_empty())
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let rows = other
                .iter()
                .try_fold(0usize, |sum, block| sum.checked_add(block.shape().0))
                .ok_or(PolyBackendError::InvalidConstantShape)?;
            return if *self == Self::ConcatRows {
                rows.checked_add(scalable.shape().0).ok_or(PolyBackendError::InvalidConstantShape)
            } else if rows == scalable.shape().0 {
                Ok(rows)
            } else {
                Err(PolyBackendError::InvalidConstantShape)
            };
        }
        if let Self::Tensor { right_rows, right_columns, groups } = self {
            let fixed = other.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            if fixed.shape() != (*right_rows, *right_columns) {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let rows = scalable
                .shape()
                .0
                .checked_mul(*right_rows)
                .ok_or(PolyBackendError::InvalidConstantShape)?;
            scalable
                .shape()
                .1
                .checked_mul(*right_columns)
                .ok_or(PolyBackendError::InvalidConstantShape)?;
            if let Some(groups) = groups {
                if groups
                    .iter()
                    .any(|group| group.is_empty() || group.iter().any(|&row| row >= rows))
                {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                return Ok(groups.len());
            }
            return Ok(rows);
        }
        if let Self::Slice { rows, columns } = self {
            if rows.start > rows.end ||
                rows.end > scalable.shape().0 ||
                columns.start > columns.end ||
                columns.end > scalable.shape().1
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            return Ok(rows.end - rows.start);
        }
        if let Self::SumRows(rows) = self {
            if rows
                .iter()
                .any(|group| group.is_empty() || group.iter().any(|&row| row >= scalable.shape().0))
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            return Ok(rows.len());
        }
        if *self == Self::Transpose {
            return Ok(scalable.shape().1);
        }
        if let Self::Accumulate { rows, .. } = self {
            return Ok(*rows);
        }
        if matches!(self, Self::Multiply { .. }) {
            let fixed = other.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            if fixed.shape() == (1, 1) {
                return Ok(scalable.shape().0);
            }
            if fixed.shape().1 != scalable.shape().0 || fixed.shape().1 == 0 {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            Ok(fixed.shape().0)
        } else if other.iter().any(|other| other.shape() != scalable.shape()) {
            Err(PolyBackendError::InvalidConstantShape)
        } else {
            Ok(scalable.shape().0)
        }
    }

    /// The native batch family used by submission and resource accounting.
    /// A supported family prepares shared inputs once for the sibling batch.
    pub(super) fn matrix_batch_kind(
        &self,
        matrix_scalar: bool,
    ) -> Option<(mxx_primitives::poly::dcrt::gpu::GpuMatrixBatchOperation, usize)> {
        use mxx_primitives::poly::dcrt::gpu::GpuMatrixBatchOperation;
        Some(match self {
            Self::Negate => (GpuMatrixBatchOperation::Negate, 1),
            Self::Add | Self::Subtract => (GpuMatrixBatchOperation::Binary, 1),
            Self::Scale(_) => (GpuMatrixBatchOperation::Scalar, 1),
            Self::Automorphism(_) => (GpuMatrixBatchOperation::Automorphism, 1),
            Self::Multiply { .. } => (
                if matrix_scalar {
                    GpuMatrixBatchOperation::Scalar
                } else {
                    GpuMatrixBatchOperation::Multiply
                },
                1,
            ),
            Self::Accumulate { products, .. } => {
                (GpuMatrixBatchOperation::Accumulate, products.len())
            }
            _ => return None,
        })
    }

    /// Native shared metadata of one homogeneous batch. Both inventory and
    /// submission consume this operation lowering; no representative is run.
    pub(super) fn batch_workspaces(
        &self,
        parameters: &GpuDCRTPolyParams,
        level: usize,
        output_shape: (usize, usize),
        jobs: usize,
        matrix_scalar: bool,
    ) -> Result<Vec<GpuPreparedWorkspaceLayout>, PolyBackendError> {
        let Some((kind, products)) = self.matrix_batch_kind(matrix_scalar) else {
            return Ok(Vec::new());
        };
        let workspace = parameters
            .matrix_batch_workspace_bytes(level, output_shape, jobs, products, kind, true)
            .map_err(PolyBackendError::GpuSubmission)?;
        let mut layouts = Vec::new();
        if workspace.additional_bytes != 0 {
            layouts.push(GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::BatchWorkspace,
                bytes: workspace.additional_bytes,
                alignment: workspace.alignment,
            });
        }
        if workspace.pinned_bytes != 0 {
            layouts.push(GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::PinnedHost,
                bytes: workspace.pinned_bytes,
                alignment: workspace.alignment,
            });
            layouts.push(GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            });
        }
        Ok(layouts)
    }

    pub(super) fn fixed_workspaces(
        &self,
        parameters: &GpuDCRTPolyParams,
        level: usize,
    ) -> Result<Vec<GpuPreparedWorkspaceLayout>, PolyBackendError> {
        if let Self::Decompose { small: false, input_rows, layout, .. } = self {
            return input_rows
                .iter()
                .map(|&rows| {
                    parameters
                        .matrix_gadget_correction_workspace_bytes(
                            level,
                            rows,
                            1,
                            layout.dropped_moduli,
                        )
                        .map_err(PolyBackendError::GpuSubmission)
                })
                .filter_map(|workspace| match workspace {
                    Ok(workspace) if workspace.additional_bytes == 0 => None,
                    Ok(workspace) => Some(Ok(GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::TransformWorkspace,
                        bytes: workspace.additional_bytes,
                        alignment: workspace.alignment,
                    })),
                    Err(error) => Some(Err(error)),
                })
                .collect();
        }
        if let Self::CrtRecompose { plaintext_moduli, .. } = self {
            let layout = parameters
                .matrix_crt_workspace_bytes(
                    level,
                    1,
                    1,
                    GpuMatrixCrtOperation::Recompose,
                    1,
                    plaintext_moduli.len(),
                    true,
                )
                .map_err(PolyBackendError::GpuSubmission)?;
            return Ok(if layout.additional_bytes == 0 {
                Vec::new()
            } else {
                vec![GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::TransformWorkspace,
                    bytes: layout.additional_bytes,
                    alignment: layout.alignment,
                }]
            });
        }
        if let Self::RnsConversion { source_moduli, .. } = self {
            let layout = parameters
                .matrix_crt_workspace_bytes(
                    level,
                    1,
                    1,
                    GpuMatrixCrtOperation::RnsConversion,
                    source_moduli.len(),
                    1,
                    true,
                )
                .map_err(PolyBackendError::GpuSubmission)?;
            return Ok(if layout.additional_bytes == 0 {
                Vec::new()
            } else {
                vec![GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::TransformWorkspace,
                    bytes: layout.additional_bytes,
                    alignment: layout.alignment,
                }]
            });
        }
        if !matches!(self, Self::Polynomial { .. }) {
            return Ok(Vec::new());
        }
        let transfer = parameters
            .rns_transfer_workspace(level, 1, 1)
            .map_err(PolyBackendError::GpuSubmission)?;
        // Exact allocation order in fill_polynomial/load_rns_bytes.
        Ok(vec![
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::PinnedHost,
                bytes: transfer.bytes,
                alignment: 1,
            },
            transfer,
            // The single-device RNS loader dispatches once on the first limb stream.
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
        ])
    }

    /// Native claims a one-column pilot of this operation holds through the
    /// claim broker, so the pilot's declared demand and peak bound cover them.

    /// Format of the ordinary scratch claimed for one range. Imports decode
    /// into coefficient scratch before converting into an evaluation output.
    pub(super) fn scratch_evaluation(&self, output_evaluation: bool) -> bool {
        if matches!(self, Self::ImportMatrix { .. }) { false } else { output_evaluation }
    }

    pub(super) fn scratch_rows(&self) -> Result<Vec<usize>, PolyBackendError> {
        if let Self::ImportMatrix { ty, .. } | Self::ImportStaging { ty, .. } = self {
            return Ok(vec![ty.rows]);
        }
        if let Self::Decompose { small, layout, input_rows, hash, .. } = self {
            // A hashed source is sampled into COEFF scratch for each range; the
            // full-basis correction copy follows it with the same row count.
            let mut rows = hash.iter().map(|_| input_rows[0]).collect::<Vec<_>>();
            if !small && layout.dropped_moduli != 0 {
                rows.extend_from_slice(input_rows);
            }
            return Ok(rows);
        }
        Ok(vec![1; self.reduction_intermediates()?])
    }

    /// Native workspace whose exact size depends on the admitted range width.
    /// Queried through the same native planner the range runner uses.
    pub(super) fn width_workspaces(
        &self,
        parameters: &GpuDCRTPolyParams,
        level: usize,
        columns: usize,
    ) -> Result<Vec<GpuPreparedWorkspaceLayout>, PolyBackendError> {
        if let Self::MultiplyCompact { inner, .. } = self {
            return parameters
                .small_rhs_workspaces(level, *inner, columns.max(1))
                .map_err(PolyBackendError::GpuSubmission);
        }
        if let Self::ImportMatrix { ty, max_coefficient_bits, .. } = self {
            // Native load order: private submission stream, then the decode span.
            let transfer = parameters
                .compact_transfer_workspace(
                    level,
                    ty.rows,
                    columns.max(1),
                    GpuCompactTransferKind::Load { max_coefficient_bits: *max_coefficient_bits },
                )
                .map_err(PolyBackendError::GpuSubmission)?;
            return Ok(vec![
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::SubmissionStream,
                    bytes: 0,
                    alignment: 1,
                },
                transfer,
            ]);
        }
        if let Self::ImportStaging { ty, .. } = self {
            // Production column staging loader: pinned tile, RNS transfer
            // span, then the loader's completion event.
            let transfer = parameters
                .rns_transfer_workspace(level, ty.rows, columns.max(1))
                .map_err(PolyBackendError::GpuSubmission)?;
            return Ok(vec![
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::PinnedHost,
                    bytes: transfer.bytes,
                    alignment: 1,
                },
                transfer,
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                },
            ]);
        }
        if let Self::ImportCompact { ty, bound } = self {
            // Native order: compact scratch payload, then the pinned upload staging.
            let columns = columns.max(1);
            let payload = GpuSmallMatrix::allocation_bytes(parameters, ty.rows, columns, bound)
                .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
            let width = Self::compact_coefficient_width(bound)?;
            let pinned = ty
                .rows
                .checked_mul(columns)
                .and_then(|count| count.checked_mul(ty.ring_dimension))
                .and_then(|count| count.checked_mul(width))
                .ok_or(PolyBackendError::InvalidInteger)?;
            return Ok(vec![
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompactPayload,
                    bytes: payload,
                    alignment: 256,
                },
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::PinnedHost,
                    bytes: pinned,
                    alignment: 1,
                },
                // The deferred pinned release records its retirement event.
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                },
            ]);
        }
        Ok(Vec::new())
    }

    pub(super) fn reduction_intermediates(&self) -> Result<usize, PolyBackendError> {
        match self {
            Self::SumRows(groups) | Self::Tensor { groups: Some(groups), .. } => {
                GpuDCRTPolyMatrix::row_reduction_intermediates(groups)
                    .map_err(PolyBackendError::GpuSubmission)
            }
            _ => Ok(0),
        }
    }

    pub(super) fn output_columns<M: MatrixShape>(&self, scalable: Option<&M>) -> usize {
        if let Some(ty) = self.fresh_type() {
            return ty.columns;
        }
        if let Self::CenteredExtendCompact { destination, .. } = self {
            return destination.columns;
        }
        if let Self::MultiplyCompact { columns, .. } = self {
            return *columns;
        }
        let scalable = scalable.expect("validated operation has an input");
        match self {
            Self::ConcatColumns { columns, .. } => *columns,
            Self::Transpose => scalable.shape().0,
            Self::Tensor { right_columns, .. } => scalable.shape().1 * right_columns,
            Self::Slice { columns, .. } => columns.end - columns.start,
            _ => scalable.shape().1,
        }
    }

    /// CPU-only output partition in operation operand order. The inputs are
    /// source owner boundaries, not proposed scratch tiles or native pointers.
    pub(super) fn column_boundaries(&self, inputs: &[Vec<usize>], width: usize) -> Vec<usize> {
        let mut boundaries = vec![0, width];
        match self {
            Self::Transpose | Self::Tensor { .. } => {}
            Self::ConcatColumns { offsets, .. } => {
                boundaries.extend(
                    inputs
                        .par_iter()
                        .zip(offsets.par_iter())
                        .flat_map_iter(|(cuts, &(_, offset, _))| {
                            cuts.iter().map(move |column| offset + column)
                        })
                        .collect::<Vec<_>>(),
                );
            }
            Self::Slice { columns, .. } => {
                boundaries.extend(
                    inputs
                        .first()
                        .into_iter()
                        .flatten()
                        .filter_map(|column| column.checked_sub(columns.start))
                        .filter(|column| *column < width),
                );
            }
            _ => {
                boundaries.extend(
                    inputs
                        .par_iter()
                        .enumerate()
                        .filter(|(index, _)| *index == 0 || !self.fixed_input(index - 1))
                        .flat_map_iter(|(_, cuts)| cuts.iter().copied())
                        .collect::<Vec<_>>(),
                );
            }
        }
        boundaries.par_sort_unstable();
        boundaries.dedup();
        boundaries
    }

    /// Select inherited output owners using only ordered input fragment metadata.
    /// Both native admission and symbolic layout propagation consume these exact
    /// ranges. Containing capacity envelopes must not be passed as fragments.
    pub(super) fn inherited_output_ranges(
        &self,
        inputs: &[(usize, &[super::gpu_prepare::MatrixInputFragment])],
        columns: usize,
    ) -> Result<Vec<InheritedMatrixRange>, PolyBackendError> {
        let boundaries = inputs
            .par_iter()
            .map(|(columns, fragments)| {
                fragments
                    .iter()
                    .map(|fragment| fragment.start)
                    .chain(std::iter::once(*columns))
                    .collect()
            })
            .collect::<Vec<_>>();
        self.column_boundaries(&boundaries, columns)
            .windows(2)
            .map(|range| {
                let (start, end) = (range[0], range[1]);
                let primary = if let Self::ConcatColumns { offsets, .. } = self {
                    offsets.partition_point(|&(_, _, end)| end <= start)
                } else {
                    0
                };
                let (width, fragments) =
                    inputs.get(primary).ok_or(PolyBackendError::InvalidConstantShape)?;
                let source = self.source_columns(*width, start, end);
                let fragment = fragments
                    .iter()
                    .position(|fragment| {
                        fragment.start <= source.start && source.end <= fragment.end
                    })
                    .ok_or(PolyBackendError::InvalidConstantShape)?;
                Ok(InheritedMatrixRange { columns: start..end, primary, fragment, source })
            })
            .collect()
    }

    pub(super) fn source_columns(
        &self,
        source_width: usize,
        start: usize,
        end: usize,
    ) -> std::ops::Range<usize> {
        match self {
            Self::ConcatColumns { offsets, .. } => {
                let index = offsets.partition_point(|&(_, _, end)| end <= start);
                start - offsets[index].1..end - offsets[index].1
            }
            Self::Transpose | Self::Tensor { .. } => 0..source_width,
            Self::Slice { columns, .. } => start + columns.start..end + columns.start,
            _ => start..end,
        }
    }

    pub(super) fn validate_output_workspace<M: MatrixShape>(
        &self,
        parameters: &GpuDCRTPolyParams,
        level: usize,
        other: &[M],
        shape: (usize, usize),
    ) -> Result<(), PolyBackendError> {
        if matches!(self, Self::Multiply { .. }) {
            use mxx_primitives::poly::dcrt::gpu::GpuMatrixBatchOperation;
            let operation = if other.first().is_some_and(|fixed| fixed.shape() == (1, 1)) {
                GpuMatrixBatchOperation::Scalar
            } else {
                GpuMatrixBatchOperation::Multiply
            };
            let workspace = parameters
                .matrix_batch_workspace_bytes(level, shape, 1, 1, operation, true)
                .map_err(PolyBackendError::GpuCalibration)?;
            if workspace.additional_bytes != 0 {
                return Err(PolyBackendError::GpuSubmission(
                    "prepared product needs a separately reserved batch workspace".into(),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn fixed_input(&self, index: usize) -> bool {
        matches!(self, Self::Multiply { .. } | Self::MultiplyCompact { .. } | Self::Tensor { .. }) ||
            matches!(self, Self::Accumulate { products, .. } if index < 2 * products.len() - 1 && index % 2 == 0)
    }

    pub(super) fn other_columns<M: MatrixShape>(
        &self,
        index: usize,
        other: &M,
        start: usize,
        end: usize,
    ) -> std::ops::Range<usize> {
        if self.fixed_input(index) { 0..other.shape().1 } else { start..end }
    }

    pub(super) fn source<'a>(
        &self,
        inputs: &'a PreparedMatrixInputs,
        matrix: &'a GpuFleetMatrix,
        index: PreparedMatrixSource,
    ) -> &'a GpuColumnShard<GpuDCRTPolyMatrix> {
        match index {
            PreparedMatrixSource::Shard(index) => &matrix.shards[index],
            PreparedMatrixSource::Replica { .. } | PreparedMatrixSource::Fragment { .. } => {
                inputs.get(&(matrix.id, index)).expect("compiled input has an admitted preparation")
            }
        }
    }

    /// Shared pilot/production range kernel. Sources remain resident and the
    /// caller owns the complete destination until all of its ranges are filled.
    pub(super) fn run(
        &self,
        left: Option<&GpuColumnShard<GpuDCRTPolyMatrix>>,
        right: &[&GpuColumnShard<GpuDCRTPolyMatrix>],
        compact: Option<&GpuColumnShard<GpuSmallMatrix>>,
        start: usize,
        end: usize,
        destination: (PreparedMatrixValue, std::ops::Range<usize>, std::ops::Range<usize>),
        payload: &ExecutionPayload,
        broker: &PreparedClaimBroker,
    ) -> Result<PreparedMatrixValue, String> {
        let seed = payload.seed();
        if matches!(self, Self::Preimage { .. }) {
            let (PreparedMatrixValue::Compact(output), rows, columns) = destination else {
                return Err("compact destination required".into());
            };
            if rows != (0..output.rows_count()) {
                return Err("preimage fills complete rows".into());
            }
            let ExecutionPayload::Preimage(payload) = payload else {
                return Err("preimage invocation has no operands".into());
            };
            return self
                .run_preimage_batch(
                    vec![gpu_preimage_prepared::PreparedPreimageJob {
                        output,
                        destination_column: columns.start,
                        start,
                        end,
                        payload,
                        public: &left.expect("admitted Preimage public matrix").value,
                    }],
                    broker,
                )
                .map(|mut outputs| PreparedMatrixValue::Compact(outputs.remove(0)));
        }
        if let Self::ImportCompact { ty, bound } = self {
            let (PreparedMatrixValue::Compact(mut output), rows, columns) = destination else {
                return Err("compact destination required".into());
            };
            let bytes = payload.bytes().ok_or("compact import has no payload")?;
            let width = end - start;
            let coefficient_width =
                Self::compact_coefficient_width(bound).map_err(|e| e.to_string())?;
            let row_bytes = width * ty.ring_dimension * coefficient_width;
            let mut local = vec![0u8; ty.rows * row_bytes];
            local.par_chunks_mut(row_bytes).enumerate().for_each(|(row, target)| {
                let source = (row * ty.columns + start) * ty.ring_dimension * coefficient_width;
                target.copy_from_slice(&bytes[source..source + row_bytes]);
            });
            let scratch = GpuSmallMatrix::from_canonical_coefficients(
                output.params(),
                ty.rows,
                width,
                bound.clone(),
                &local,
            )
            .map_err(|e| e.to_string())?;
            if rows != (0..ty.rows) {
                return Err("compact import fills complete rows".into());
            }
            output.copy_columns_from(columns.start, &scratch, 0, width)?;
            // The background reclaimer owns this upload's pinned staging.
            drop(scratch);
            return Ok(PreparedMatrixValue::Compact(output));
        }
        if let Self::ImportStaging { ty, .. } = self {
            let (PreparedMatrixValue::Matrix(output), rows, columns) = destination else {
                return Err("ordinary destination required".into());
            };
            let bytes = payload.bytes().ok_or("staging import has no payload")?;
            let width = end - start;
            let staged =
                <GpuDCRTPolyMatrix as mxx_primitives::matrix::PolyMatrix>::from_cpu_staging_columns(
                    output.params(),
                    bytes,
                    start,
                    end,
                );
            if staged.size() != (ty.rows, width) || staged.is_ntt() != output.is_ntt() {
                return Err("staging payload layout differs from the admitted import".into());
            }
            let copied = staged.column_view(0..width)?.copy(Some((output, rows, columns)))?;
            drop(staged);
            return Ok(PreparedMatrixValue::Matrix(copied));
        }
        if let Self::ImportMatrix { ty, evaluation, max_coefficient_bits } = self {
            let (PreparedMatrixValue::Matrix(output), rows, columns) = destination else {
                return Err("ordinary destination required".into());
            };
            let bytes = payload.bytes().ok_or("matrix import has no payload")?;
            let width = end - start;
            let bits = usize::from(*max_coefficient_bits);
            let row_bits = width * ty.ring_dimension * bits;
            let mut local = vec![0u8; (ty.rows * row_bits).div_ceil(8)];
            for row in 0..ty.rows {
                super::copy_packed_bits(
                    bytes,
                    (row * ty.columns + start) * ty.ring_dimension * bits,
                    &mut local,
                    row * row_bits,
                    row_bits,
                );
            }
            // Decode into coefficient scratch; the retained destination keeps
            // the artifact's format after an in-place NTT of the scratch.
            let mut scratch = GpuDCRTPolyMatrix::new_empty_with_state(
                output.params(),
                ty.rows,
                width,
                output.level(),
                false,
                None,
            );
            scratch.load_compact_payload(&local, *max_coefficient_bits)?;
            if *evaluation {
                scratch.ntt_all_in_place();
            }
            return scratch
                .column_view(0..width)?
                .copy(Some((output, rows, columns)))
                .map(PreparedMatrixValue::Matrix);
        }
        if let Self::CenteredExtendCompact { .. } = self {
            let (PreparedMatrixValue::Compact(mut output), rows, columns) = destination else {
                return Err("compact destination required".into());
            };
            let source = compact.ok_or("compact extension has no compact input")?;
            if left.is_some() || !right.is_empty() || rows != (0..output.rows_count()) {
                return Err("compact extension takes exactly one complete compact input".into());
            }
            output.copy_columns_from(
                columns.start,
                &source.value,
                start - source.global_column_start,
                end - start,
            )?;
            return Ok(PreparedMatrixValue::Compact(output));
        }
        if let Self::MultiplyCompact { .. } = self {
            let (PreparedMatrixValue::Matrix(output), rows, columns) = destination else {
                return Err("ordinary destination required".into());
            };
            let source = compact.ok_or("compact product has no compact input")?;
            let [lhs] = right else {
                return Err("compact product takes exactly one fixed LHS".into());
            };
            if left.is_some() {
                return Err("compact product has no scalable ordinary input".into());
            }
            let lhs = lhs.value.column_view(0..lhs.value.col_size())?;
            let rhs = source
                .value
                .column_view(start - source.global_column_start, end - source.global_column_start);
            return mxx_primitives::matrix::gpu_dcrt_poly::GpuDCRTPolyMatrixColumnView::multiply_small_rhs_row_blocks(
                    &[lhs],
                    rhs.as_ref(),
                    Some(vec![(output, rows, columns)]),
                )
                .map_err(|error| error.to_string())?
                .pop()
                .map(PreparedMatrixValue::Matrix)
                .ok_or_else(|| "compact product produced no output".into());
        }
        if compact.is_some() {
            return Err("operation takes no compact input".into());
        }
        if let Self::Decompose { small, digit_count, input_rows, hash, .. } = self {
            let (PreparedMatrixValue::Compact(output), rows, columns) = destination else {
                return Err("compact destination required".into());
            };
            if let Some(ty) = hash {
                if left.is_some() || !right.is_empty() {
                    return Err("compact hash sampling takes no prepared input".into());
                }
                // The same seeded COEFF source stream as the production hash
                // sampler, generated only for this range on the output owner.
                let source = mxx_primitives::sampler::gpu::sample_seeded_gadget_source_columns(
                    output.params(),
                    input_rows[0],
                    ty.columns,
                    start,
                    end - start,
                    seed.ok_or("compact hash invocation has no execution seed")?,
                );
                let view = source.column_view(0..end - start)?;
                return mxx_primitives::matrix::gpu_dcrt_poly::GpuDCRTPolyMatrixColumnView::gadget_decompose_row_blocks(
                    &[view],*small,*digit_count,Some((output,rows,columns))).map(PreparedMatrixValue::Compact);
            }
            let blocks = left
                .into_iter()
                .chain(right.iter().copied())
                .map(|source| {
                    source.value.column_view(
                        start - source.global_column_start..end - source.global_column_start,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            return mxx_primitives::matrix::gpu_dcrt_poly::GpuDCRTPolyMatrixColumnView::gadget_decompose_row_blocks(
                &blocks,*small,*digit_count,Some((output,rows,columns))).map(PreparedMatrixValue::Compact);
        }
        let (PreparedMatrixValue::Matrix(output), rows, columns) = destination else {
            return Err("ordinary destination required".into());
        };
        let destination = (output, rows, columns);
        (|| {
        if let Self::Constant { value, .. } = self {
            let (mut output, rows, columns) = destination;
            output.fill_constant_columns(rows, columns, start, *value)?;
            return Ok(output);
        }
        if let Self::Polynomial { coefficients, evaluation, .. } = self {
            let (mut output, rows, columns) = destination;
            if start != 0 ||
                end != 1 ||
                rows != (0..1) ||
                columns != (0..1) ||
                output.size() != (1, 1)
            {
                return Err("polynomial constant must execute as a complete singleton".into());
            }
            output.fill_polynomial(coefficients, *evaluation)?;
            return Ok(output);
        }
        if let Self::Hash { ty, .. } = self {
            let (mut output, rows, columns) = destination;
            output.fill_distribution_columns(
                rows,
                columns,
                ty.columns,
                start,
                GpuMatrixSampleDist::Uniform,
                0.0,
                u64::MAX,
                seed.ok_or("hash invocation has no execution seed")?,
            )?;
            return Ok(output);
        }
        if let Self::Sample { ty, distribution, sigma_bits, max_coefficient_bound } = self {
            let (mut output, rows, columns) = destination;
            let sigma = f64::from_bits(*sigma_bits);
            if *distribution == GpuMatrixSampleDist::Gauss && sigma == 0.0 {
                output.fill_constant_columns(
                    rows,
                    columns,
                    start,
                    GpuMatrixRangeConstant::Zero { total_columns: ty.columns },
                )?;
            } else {
                output.fill_distribution_columns(
                    rows,
                    columns,
                    ty.columns,
                    start,
                    *distribution,
                    sigma,
                    max_coefficient_bound.to_u64().unwrap_or(u64::MAX),
                    seed.ok_or("sampling invocation has no execution seed")?,
                )?;
            }
            return Ok(output);
        }
        let left = left.ok_or("prepared operation has no input")?;
        if let Self::ConcatColumns { offsets, .. } = self {
            let index = offsets.partition_point(|&(_, _, end)| end <= start);
            let (row_offset, column_offset, _) = offsets[index];
            let (output, rows, columns) = destination;
            return left
                .value
                .column_view(
                    start - column_offset - left.global_column_start..
                        end - column_offset - left.global_column_start,
                )?
                .copy(Some((
                    output,
                    rows.start + row_offset..rows.start + row_offset + left.value.row_size(),
                    columns,
                )));
        }
        let input = if *self == Self::Transpose {
            left.value.column_view(0..left.value.col_size())?.row_view(start..end)?
        } else if matches!(self, Self::Tensor { .. }) {
            left.value.column_view(0..left.value.col_size())?
        } else if let Self::Slice { rows, columns } = self {
            left.value
                .column_view(
                    start + columns.start - left.global_column_start..
                        end + columns.start - left.global_column_start,
                )?
                .row_view(rows.clone())?
        } else {
            left.value
                .column_view(start - left.global_column_start..end - left.global_column_start)?
        };
        match self {
            Self::Constant { .. } |
            Self::Sample { .. } |
            Self::Hash { .. } |
            Self::Polynomial { .. } |
            Self::PolynomialReadback { .. } |
            Self::ConcatColumns { .. } => {
                unreachable!("constant and column concatenation handled above")
            }
            Self::ConcatRows => {
                let (mut output, rows, columns) = destination;
                let mut offset = rows.start;
                // One output owner orders writes to successive, disjoint rows.
                for source in std::iter::once(left).chain(right.iter().copied()) {
                    let height = source.value.row_size();
                    output = source
                        .value
                        .column_view(
                            start - source.global_column_start..end - source.global_column_start,
                        )?
                        .copy(Some((output, offset..offset + height, columns.clone())))?;
                    offset += height;
                }
                Ok(output)
            }
            Self::AddRowBlocks => {
                let blocks = right
                    .par_iter()
                    .map(|block| {
                        block.value.column_view(
                            start - block.global_column_start..end - block.global_column_start,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                input.add_row_blocks(&blocks, Some(destination))
            }
            Self::CrtRecompose { plaintext_moduli, reconstruction_coefficients, .. } => {
                let parameters = destination.0.params().clone();
                let residues = reconstruction_coefficients
                    .par_iter()
                    .flat_map_iter(|coefficient| {
                        parameters.moduli().iter().map(move |prime| {
                            let modulus = BigInt::from(*prime);
                            (((coefficient % &modulus) + &modulus) % &modulus).to_u64().unwrap()
                        })
                    })
                    .collect::<Vec<_>>();
                let mut levels = Vec::with_capacity(1 + right.len());
                levels.push(input);
                for source in right {
                    levels.push(source.value.column_view(
                        start - source.global_column_start..end - source.global_column_start,
                    )?);
                }
                mxx_primitives::matrix::gpu_dcrt_poly::GpuDCRTPolyMatrixColumnView::crt_recompose(
                    &levels,
                    plaintext_moduli,
                    &residues,
                    &parameters,
                    Some(destination),
                )
            }
            Self::RnsConversion { conversion, .. } => {
                let parameters = destination.0.params().clone();
                input.rns_conversion(&parameters, *conversion, Some(destination))
            }
            Self::ModulusConversion { conversion, .. } => {
                let parameters = destination.0.params().clone();
                input.convert_modulus(&parameters, *conversion, Some(destination))
            }
            Self::CenteredRebase { .. } => {
                let parameters = destination.0.params().clone();
                input.centered_rebase(&parameters, Some(destination))
            }
            Self::Negate => input.negate(Some(destination)),
            Self::Tensor { groups, .. } => {
                let right = &right.first().ok_or("tensor is missing its fixed operand")?.value;
                input.tensor(
                    &right.column_view(0..right.col_size())?,
                    groups.as_deref(),
                    start..end,
                    Some(destination),
                )
            }

            Self::Slice { .. } => input.copy(Some(destination)),
            Self::Transpose => input.transpose(Some(destination)),
            Self::SumRows(rows) => input.sum_rows(rows, Some(destination)),
            Self::Scale(scalar) => input.scale_integer(scalar, Some(destination)),
            Self::Automorphism(index) => input.ring_automorphism(*index, Some(destination)),
            Self::Accumulate { products, bias, .. } => {
                let products = products
                    .iter()
                    .enumerate()
                    .map(|(index, (coefficient, scales_left))| {
                        let scalable = if index == 0 {
                            input
                        } else {
                            let source = right[2 * index - 1];
                            source.value.column_view(
                                start - source.global_column_start..
                                    end - source.global_column_start,
                            )?
                        };
                        let fixed = &right[2 * index].value;
                        let fixed = fixed.column_view(0..fixed.col_size())?;
                        let (left, right) =
                            if *scales_left { (scalable, fixed) } else { (fixed, scalable) };
                        Ok((coefficient.clone(), left, right))
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                let bias = if *bias {
                    let source = right.last().ok_or("accumulate bias is missing")?;
                    Some(source.value.column_view(
                        start - source.global_column_start..end - source.global_column_start,
                    )?)
                } else {
                    None
                };
                mxx_primitives::matrix::gpu_dcrt_poly::GpuDCRTPolyMatrixColumnView::multiply_accumulate(
                    &products, bias, Some(destination))
            }
            Self::Multiply { scales_left } => {
                let fixed =
                    &right.first().ok_or("prepared product is missing its fixed operand")?.value;
                let fixed = fixed.column_view(0..fixed.col_size())?;
                if *scales_left {
                    input.multiply(&fixed, Some(destination))
                } else {
                    fixed.multiply(&input, Some(destination))
                }
            }
            Self::Add | Self::Subtract => {
                let right = right.first().ok_or("binary prepared range is missing its RHS")?;
                let rhs = right.value.column_view(
                    start - right.global_column_start..end - right.global_column_start,
                )?;
                if *self == Self::Add {
                    input.add(&rhs, Some(destination))
                } else {
                    input.sub(&rhs, Some(destination))
                }
            }
            Self::Decompose { .. } |
            Self::CenteredExtendCompact { .. } |
            Self::MultiplyCompact { .. } |
            Self::ImportMatrix { .. } |
            Self::ImportStaging { .. } |
            Self::Preimage { .. } |
            Self::ImportCompact { .. } => {
                unreachable!("handled compact operands")
            }
        }
        })().map(PreparedMatrixValue::Matrix)
    }
}

pub(super) type MatrixInvocation<'a> =
    GpuInvocation<'a, GpuFleetMatrix, GpuFleetSmallMatrix, GpuFleetTrapdoor>;

/// One preflight request: its destination placement, the validated IR node it
/// belongs to when it is that node's operation, and the invocation itself.
pub(super) type MatrixInvocationRequest<'a> =
    (usize, Option<GpuNodeOperation>, MatrixInvocation<'a>);

struct CompiledInterval {
    interval: GpuColumnInterval,
    destination: usize,
    left_source: Option<PreparedMatrixSource>,
    parameters: GpuDCRTPolyParams,
    level: usize,
    evaluation: bool,
    right_source: Vec<PreparedMatrixSource>,
}

pub(super) struct CompiledMatrixInvocation {
    operation: PreparedOperation,
    left: Option<GpuFleetMatrix>,
    right: Vec<GpuFleetMatrix>,
    compact: Option<GpuFleetSmallMatrix>,
    /// Ordinary operand ids in caller argument order, and the compact input id.
    /// Execution binds real values and rejects any that differ from these.
    caller_ids: Vec<u64>,
    caller_compact: Option<u64>,
    plan: GpuColumnMemoryPlan,
    intervals: Vec<CompiledInterval>,
    prepared: Arc<PreparedMatrixInputs>,
}

/// Real operand owners of one invocation, in the operation's own order, with
/// every check that inspects actual native owners: device membership, registered
/// parameter contexts and owners, formats, RNS bases, gadget layouts, decomposed
/// block consistency and native depth limits. It never derives the resource
/// operation.
pub(super) struct InvocationOperands<M = GpuFleetMatrix, S = GpuFleetSmallMatrix> {
    pub(super) left: Option<M>,
    pub(super) right: Vec<M>,
    pub(super) compact: Option<S>,
}

impl InvocationOperands {
    fn none() -> Self {
        Self { left: None, right: Vec::new(), compact: None }
    }
}

/// Output context, format and input identities accepted for one retained
/// output interval. The input identities refer to
/// the invocation's owned inputs or its shared prepared input map; they carry
/// no independent native lease and remain protected by the admitted plan.
#[derive(Clone)]
pub(super) struct AdmittedMatrixBinding {
    pub parameters: GpuDCRTPolyParams,
    pub level: usize,
    pub evaluation: bool,
    pub left: Option<PreparedMatrixSource>,
    pub right: Vec<PreparedMatrixSource>,
    pub compact: Option<PreparedMatrixSource>,
}

// CPU lowering of existing IR positions. No second schedule or instruction
// stream: execution still traverses the validated scope in its original order.
pub(super) type InventoryOperation = (GpuNodeOperation, Option<PreparedOperation>);

/// The exact bounded scope lowerings used by a committed wave's capacity fit.
/// The reservation owns this record; the backend keeps only weak access to
/// active records so nested waves cannot overwrite their parents' bindings.
pub(super) struct AdmittedScopeOperations {
    pub scope: mxx_ir_core::FrozenGraphScopeId,
    pub instances: Vec<Vec<InventoryOperation>>,
    pub input_layouts: MatrixInputLayouts,
    pub value_layouts: Vec<super::gpu_prepare::SymbolicMatrixLayouts>,
}

/// Shape-only input to operation layout lowering. Metadata consumers need no
/// GPU value, native allocation or sampled payload to determine output geometry.
pub(super) trait MatrixShape {
    fn shape(&self) -> (usize, usize);
}

impl MatrixShape for GpuFleetMatrix {
    fn shape(&self) -> (usize, usize) {
        self.size()
    }
}

impl MatrixShape for (usize, usize) {
    fn shape(&self) -> (usize, usize) {
        *self
    }
}

/// One inherited destination range and the exact source fragment serving it.
/// No native owner, allocation or execution permission is created by this value.
pub(super) struct InheritedMatrixRange {
    pub columns: std::ops::Range<usize>,
    pub primary: usize,
    pub fragment: usize,
    pub source: std::ops::Range<usize>,
}

/// One lowered production invocation: the resource operation (from the validated
/// IR node when the request identifies one, otherwise from the invocation's own
/// boundary constructor), the real operand owners in the operation's own order,
/// and the caller's operand identities used to reject a substituted operand.
pub(super) struct LoweredMatrixInvocation {
    pub(super) operation: PreparedOperation,
    pub(super) operands: InvocationOperands,
    pub(super) input_layouts: MatrixInputLayouts,
    pub(super) caller_ids: Vec<u64>,
    pub(super) caller_compact: Option<u64>,
}

impl CompiledMatrixInvocation {
    pub(super) fn lower_polynomial_readback(
        value: &GpuFleetMatrix,
        evaluation: bool,
        backend: &GpuDcrtBackend,
    ) -> Result<PreparedOperation, PolyBackendError> {
        let first = value.shards.first().ok_or(PolyBackendError::InvalidInteger)?;
        let device = backend
            .devices
            .iter()
            .position(|(id, _)| *id == first.device_id)
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        let parameters = backend.devices[device].1.parameters_for_matrix(&first.value)?;
        let claims = polynomial_readback_claims(
            &backend.polynomial_value_plans,
            parameters,
            first.value.level(),
            evaluation,
            first.value.is_ntt(),
        )
        .ok_or_else(|| {
            PolyBackendError::GpuSubmission(
                "polynomial readback has no admitted operation projection".into(),
            )
        })?;
        Ok(PreparedOperation::PolynomialReadback { evaluation, claims })
    }

    /// Single lowering of one preflight request into the resource operation
    /// admission and execution both consume.
    ///
    /// A request that identifies its validated IR node is lowered from that node
    /// alone: the IR is the only authority for the operation, and a node that
    /// does not determine one is an error rather than a reason to re-derive it
    /// from values. Every other request — a fusion result, a sampler algorithm
    /// step, an artifact or transcript import, or a direct primitive call — is
    /// lowered by the one boundary constructor [`Self::from_invocation`].
    pub(super) fn lower(
        request: &MatrixInvocation<'_>,
        node: Option<&GpuNodeOperation>,
        backend: &GpuDcrtBackend,
    ) -> Result<LoweredMatrixInvocation, PolyBackendError> {
        let operands = Self::operands(request, backend)?;
        let operation = match node {
            Some(node) => Self::lower_ir(node, backend)?.ok_or_else(|| {
                PolyBackendError::GpuSubmission(
                    "validated IR node does not determine a prepared matrix operation".into(),
                )
            })?,
            None => Self::from_invocation(request, &operands, backend)?,
        };
        let mut input_layouts: MatrixInputLayouts = operands
            .left
            .iter()
            .chain(&operands.right)
            .filter_map(|matrix| {
                backend
                    .admitted_scope_operations
                    .iter()
                    .rev()
                    .filter_map(Weak::upgrade)
                    .find_map(|admitted| admitted.input_layouts.get(&matrix.id).cloned())
                    .map(|layout| (matrix.id, layout))
            })
            .collect();
        let (caller_ids, caller_compact) = Self::caller_operands(request);
        if let Some(node) = node {
            let wires = node
                .argument_wires()
                .iter()
                .zip(node.arguments())
                .filter_map(|(wire, ty)| matches!(ty, ConcreteWireType::Matrix(_)).then_some(wire))
                .collect::<Vec<_>>();
            // Only direct IR operands have this correspondence. Fused boundary
            // calls retain their own accepted operand mapping.
            if wires.len() == caller_ids.len() {
                for admitted in
                    backend.admitted_scope_operations.iter().rev().filter_map(Weak::upgrade)
                {
                    if &admitted.scope != node.scope() {
                        continue;
                    }
                    let Some(index) = admitted.instances.iter().position(|instance| {
                        instance
                            .first()
                            .is_some_and(|(operation, _)| operation.bindings() == node.bindings())
                    }) else {
                        continue
                    };
                    for (wire, owner) in wires.iter().zip(&caller_ids) {
                        if let Some(layout) = admitted.value_layouts[index].get(wire) {
                            // Unit executions compare the pre-execution symbolic
                            // layout with the native producer's published layout.
                            #[cfg(test)]
                            if let Some(actual) = operands
                                .left
                                .iter()
                                .chain(&operands.right)
                                .find(|matrix| matrix.id == *owner)
                            {
                                assert_eq!(
                                    layout.as_ref(),
                                    actual.input_layout.as_ref(),
                                    "symbolic input layout at {:?} {:?} {:?}",
                                    node.scope(),
                                    node.node(),
                                    wire
                                );
                            }
                            input_layouts.entry(*owner).or_insert_with(|| layout.clone());
                        }
                    }
                    break;
                }
            }
        }
        Ok(LoweredMatrixInvocation {
            operation,
            operands,
            input_layouts,
            caller_ids,
            caller_compact,
        })
    }

    /// The operation the validated IR node determines, asked through the native
    /// layout queries of the context that owns the node's output ring.
    pub(super) fn lower_ir(
        node: &GpuNodeOperation,
        backend: &GpuDcrtBackend,
    ) -> Result<Option<PreparedOperation>, PolyBackendError> {
        let mut admitted_scope = false;
        for admitted in backend.admitted_scope_operations.iter().rev().filter_map(Weak::upgrade) {
            if &admitted.scope != node.scope() {
                continue;
            }
            admitted_scope = true;
            if let Some(instance) = admitted.instances.iter().find(|instance| {
                instance.first().is_some_and(|(accepted, _)| accepted.bindings() == node.bindings())
            }) {
                // Node IDs are positions in this validated scope. The executor
                // supplies the same bindings admitted before input loading;
                // no new resource lowering or native query occurs here.
                return Ok(instance[node.node().0 as usize].1.clone());
            }
        }
        if admitted_scope {
            return Err(PolyBackendError::GpuSubmission(format!(
                "scope {:?} node {:?} has no admitted operation for its production bindings",
                node.scope(),
                node.node(),
            )));
        }
        if matches!(node.kind(), NodeKind::PreimageSample { .. }) {
            let (
                ConcreteWireType::Matrix(public),
                ConcreteWireType::Trapdoor { sigma, gadget_base, digit_count, .. },
                ConcreteWireType::Preimage { matrix, max_coefficient_bound },
            ) = (&node.arguments()[0], &node.arguments()[1], &node.outputs()[0])
            else {
                return Err(PolyBackendError::InvalidConstantShape);
            };
            let sigma = sigma
                .evaluate_f64(node.bindings())
                .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
            return PreparedOperation::preimage(
                backend,
                matrix,
                max_coefficient_bound.to_biguint().ok_or(PolyBackendError::InvalidInteger)?,
                sigma,
                gadget_base,
                *digit_count,
                public.rows,
            )
            .map(Some);
        }
        let output = node
            .outputs()
            .first()
            .and_then(leaf_matrix_type)
            .ok_or(PolyBackendError::InvalidConstantShape)?;
        let parameters = backend.devices[0].1.parameters(output)?.clone();
        PreparedOperation::from_ir(
            node.kind(),
            node.arguments(),
            node.outputs(),
            node.bindings(),
            &parameters,
        )
    }

    /// Ordinary operand identities of one invocation in caller argument order,
    /// plus its compact input. Execution compares the values it binds with these.
    fn caller_operands(request: &MatrixInvocation<'_>) -> (Vec<u64>, Option<u64>) {
        let ordinary = |value: &GpuFleetMatrix| value.id;
        match request {
            GpuInvocation::Binary { left, right, .. } => {
                (vec![ordinary(left), ordinary(right)], None)
            }
            GpuInvocation::Accumulate { request } => (
                request
                    .products
                    .iter()
                    .flat_map(|(_, left, right)| [ordinary(left), ordinary(right)])
                    .chain(request.bias.iter().map(|bias| ordinary(bias)))
                    .collect(),
                None,
            ),
            GpuInvocation::MultiplySmallRhs { left, right } => {
                (vec![ordinary(left)], Some(right.id))
            }
            GpuInvocation::MultiplySmallRhsRowBlocks { blocks, right } => {
                (blocks.iter().map(|block| ordinary(block)).collect(), Some(right.id))
            }
            GpuInvocation::CenteredExtendSmall { value, .. } => (Vec::new(), Some(value.id)),
            GpuInvocation::Concat { inputs, .. } => {
                (inputs.iter().map(|input| ordinary(input)).collect(), None)
            }
            GpuInvocation::AddRowBlocks { blocks, right } => (
                blocks
                    .iter()
                    .map(|block| ordinary(block))
                    .chain(std::iter::once(ordinary(right)))
                    .collect(),
                None,
            ),
            GpuInvocation::SumRows { value, .. } |
            GpuInvocation::Transpose { value } |
            GpuInvocation::Slice { value, .. } |
            GpuInvocation::ScaleInteger { value, .. } |
            GpuInvocation::RingAutomorphism { value, .. } |
            GpuInvocation::Negate { value } |
            GpuInvocation::GadgetDecompose { value, .. } |
            GpuInvocation::ModulusSwitch { value, .. } |
            GpuInvocation::ReduceModulus { value, .. } |
            GpuInvocation::CenteredExtend { value, .. } |
            GpuInvocation::BlockModSwitch { value, .. } |
            GpuInvocation::CenteredRebase { value, .. } |
            GpuInvocation::RnsModUp { value, .. } |
            GpuInvocation::RnsModDown { value, .. } => (vec![ordinary(value)], None),
            GpuInvocation::GadgetDecomposeRowBlocks { blocks, .. } => {
                (blocks.iter().map(|block| ordinary(block)).collect(), None)
            }
            GpuInvocation::CrtRecompose { levels, .. } => {
                (levels.iter().map(|level| ordinary(level)).collect(), None)
            }
            GpuInvocation::Tensor { left, right } |
            GpuInvocation::TensorSumRows { left, right, .. } => {
                (vec![ordinary(left), ordinary(right)], None)
            }
            GpuInvocation::SamplePreimage { public, .. } => (vec![ordinary(public)], None),
            GpuInvocation::PolynomialValues { value, .. } |
            GpuInvocation::ThresholdDecode { value, .. } |
            GpuInvocation::ExtractCoefficient { value, .. } => (vec![ordinary(value)], None),
            GpuInvocation::Constant { .. } |
            GpuInvocation::SampleUniform { .. } |
            GpuInvocation::SampleGaussian { .. } |
            GpuInvocation::SampleHash { .. } |
            GpuInvocation::SampleTrapdoor { .. } |
            GpuInvocation::ImportMatrix { .. } |
            GpuInvocation::ImportSmallMatrix { .. } |
            GpuInvocation::ImportTrapdoor { .. } |
            GpuInvocation::ImportCpuStaging { .. } |
            GpuInvocation::PackPolynomialCoefficients { .. } |
            GpuInvocation::PolynomialFromValues { .. } => (Vec::new(), None),
        }
    }

    /// Real operand owners of one invocation, in the operation's own order.
    fn operands(
        request: &MatrixInvocation<'_>,
        backend: &GpuDcrtBackend,
    ) -> Result<InvocationOperands, PolyBackendError> {
        if let GpuInvocation::SamplePreimage { public, .. } = request {
            return Ok(InvocationOperands {
                left: Some((*public).clone()),
                right: Vec::new(),
                compact: None,
            });
        }
        if let GpuInvocation::CenteredExtendSmall { value, destination } = request {
            if (destination.rows, destination.columns) != value.size() {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let bound =
                value.shards.first().map(|shard| shard.value.bound().clone()).unwrap_or_default();
            for shard in value.shards.iter() {
                let (_, device) = backend
                    .devices
                    .iter()
                    .find(|(id, _)| *id == shard.device_id)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                let registered = device.parameters_for_small_matrix(&shard.value)?;
                let target = device.parameters(destination)?;
                let source_primes = shard.value.params().to_crt().0;
                if registered != shard.value.params() ||
                    registered.execution_owner_id() != shard.value.params().execution_owner_id() ||
                    target.execution_owner_id() != registered.execution_owner_id() ||
                    target.ring_dimension() != registered.ring_dimension() ||
                    target.ring_dimension() as usize != destination.ring_dimension ||
                    source_primes.iter().any(|prime| !target.to_crt().0.contains(prime)) ||
                    shard.value.bound() != &bound
                {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
            }
            return Ok(InvocationOperands {
                left: None,
                right: Vec::new(),
                compact: Some((*value).clone()),
            });
        }
        if let GpuInvocation::MultiplySmallRhs { left, right } = request {
            let inner = right.rows;
            if left.columns != inner || inner == 0 {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            for shard in right.shards.iter() {
                let (_, device) = backend
                    .devices
                    .iter()
                    .find(|(id, _)| *id == shard.device_id)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                let registered = device.parameters_for_small_matrix(&shard.value)?;
                if registered != shard.value.params() ||
                    registered.execution_owner_id() != shard.value.params().execution_owner_id() ||
                    left.shards.iter().any(|lhs| {
                        lhs.value.params().ring_dimension() != registered.ring_dimension() ||
                            lhs.value.params().moduli() != registered.moduli() ||
                            lhs.value.level() + 1 != registered.crt_depth()
                    })
                {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
            }
            return Ok(InvocationOperands {
                left: None,
                right: vec![(*left).clone()],
                compact: Some((*right).clone()),
            });
        }
        if let GpuInvocation::MultiplySmallRhsRowBlocks { .. } = request {
            return Err(PolyBackendError::GpuSubmission(
                "compact row-block products are admitted as one invocation per block".into(),
            ));
        }
        let decomposition = match request {
            GpuInvocation::GadgetDecompose { value, small, digit_count } => {
                Some((vec![*value], *small, *digit_count))
            }
            GpuInvocation::GadgetDecomposeRowBlocks { blocks, small, digit_count } => {
                Some((blocks.to_vec(), *small, *digit_count))
            }
            _ => None,
        };
        if let Some((blocks, small, digit_count)) = decomposition {
            let first = blocks.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            let shard = first.shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            let parameters = shard.value.params();
            let layout = parameters
                .compact_decomposition_layout(small, digit_count)
                .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
            if blocks.len() > 32 ||
                blocks.iter().any(|v| {
                    v.rows == 0 ||
                        v.columns != first.columns ||
                        v.shards.iter().any(|s| {
                            s.value.params().ring_dimension() != parameters.ring_dimension() ||
                                s.value.params().moduli() != parameters.moduli() ||
                                s.value
                                    .params()
                                    .compact_decomposition_layout(small, digit_count)
                                    .as_ref() !=
                                    Ok(&layout) ||
                                s.value.level() + 1 != parameters.crt_depth() ||
                                backend
                                    .devices
                                    .iter()
                                    .find(|(id, _)| *id == s.device_id)
                                    .is_none_or(|(_, b)| {
                                        b.parameters_for_matrix(&s.value).map_or(true, |p| {
                                            p != s.value.params() ||
                                                p.execution_owner_id() !=
                                                    s.value.params().execution_owner_id()
                                        })
                                    })
                        })
                })
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let (first, rest) = blocks.split_first().expect("nonempty decomposition blocks");
            return Ok(InvocationOperands {
                left: Some((**first).clone()),
                right: rest.iter().map(|v| (**v).clone()).collect(),
                compact: None,
            });
        }
        if let GpuInvocation::CrtRecompose {
            levels,
            plaintext_moduli,
            reconstruction_coefficients,
            destination,
        } = request
        {
            let first = levels.first().ok_or(PolyBackendError::InvalidInteger)?;
            if destination.rows != 1 ||
                first.size() != (1, destination.columns) ||
                levels.len() != plaintext_moduli.len() ||
                levels.len() != reconstruction_coefficients.len() ||
                levels.iter().any(|level| level.size() != first.size())
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            for (_, device) in &backend.devices {
                if device.parameters(destination)?.crt_depth() > 64 {
                    return Err(PolyBackendError::InvalidInteger);
                }
            }
            for matrix in *levels {
                for shard in matrix.shards.iter() {
                    let device = backend
                        .devices
                        .iter()
                        .find(|(id, _)| *id == shard.device_id)
                        .ok_or(PolyBackendError::UnsupportedPlacement)?;
                    let registered = device.1.parameters_for_matrix(&shard.value)?;
                    let target = device.1.parameters(destination)?;
                    if registered != shard.value.params() ||
                        registered.execution_owner_id() !=
                            shard.value.params().execution_owner_id() ||
                        registered.execution_owner_id() != target.execution_owner_id() ||
                        registered.ring_dimension() != target.ring_dimension() ||
                        shard.value.level() >= 64 ||
                        shard.value.level() >= registered.crt_depth()
                    {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    }
                }
            }
            return Ok(InvocationOperands {
                left: Some(first.clone()),
                right: levels[1..].to_vec(),
                compact: None,
            });
        }
        let rns = match request {
            GpuInvocation::RnsModUp {
                value,
                destination,
                source_moduli,
                digit_size,
                normalize,
            } => Some((
                *value,
                *destination,
                *source_moduli,
                GpuMatrixRnsConversion::Up { digit_size: *digit_size, normalize: *normalize },
            )),
            GpuInvocation::RnsModDown { value, destination, source_moduli, plaintext_modulus } => {
                Some((
                    *value,
                    *destination,
                    *source_moduli,
                    GpuMatrixRnsConversion::Down { plaintext_modulus: *plaintext_modulus },
                ))
            }
            _ => None,
        };
        if let Some((value, destination, source_moduli, conversion)) = rns {
            if source_moduli.is_empty() || source_moduli.len() > 64 {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            if rns_conversion_groups(conversion, source_moduli.len()).is_none_or(|groups| {
                value.rows.checked_mul(groups) != Some(destination.rows) ||
                    value.columns != destination.columns
            }) {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            for (_, device) in &backend.devices {
                let target = device.parameters(destination)?;
                // Validate declared bases even when the matrix has no shards.
                PreparedOperation::validate_rns_basis(target, source_moduli, conversion)?;
            }
            for shard in value.shards.iter() {
                let device = backend
                    .devices
                    .iter()
                    .find(|(id, _)| *id == shard.device_id)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                let registered = device.1.parameters_for_matrix(&shard.value)?;
                if registered != shard.value.params() ||
                    registered.execution_owner_id() != shard.value.params().execution_owner_id() ||
                    shard.value.params().moduli() != source_moduli
                {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
                conversion
                    .validate(
                        shard.value.params(),
                        shard.value.level(),
                        device.1.parameters(destination)?,
                    )
                    .map_err(PolyBackendError::BasisConversion)?;
            }
            return Ok(InvocationOperands {
                left: Some((*value).clone()),
                right: Vec::new(),
                compact: None,
            });
        }
        let conversion = match request {
            GpuInvocation::ReduceModulus { value, destination } => {
                Some((*value, *destination, GpuMatrixModulusConversion::Reduce))
            }
            GpuInvocation::ModulusSwitch { value, destination } => {
                Some((*value, *destination, GpuMatrixModulusConversion::Round))
            }
            GpuInvocation::CenteredExtend { value, destination } => {
                Some((*value, *destination, GpuMatrixModulusConversion::CenteredExtend))
            }
            GpuInvocation::BlockModSwitch { value, destination, plaintext_modulus } => Some((
                *value,
                *destination,
                GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus: *plaintext_modulus },
            )),
            _ => None,
        };
        if let Some((value, destination, conversion)) = conversion {
            if value.size() != (destination.rows, destination.columns) {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            for (_, device) in &backend.devices {
                device.parameters(destination)?;
            }
            for shard in value.shards.iter() {
                let device = backend
                    .devices
                    .iter()
                    .find(|(id, _)| *id == shard.device_id)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                let registered = device.1.parameters_for_matrix(&shard.value)?;
                if registered != shard.value.params() ||
                    registered.execution_owner_id() != shard.value.params().execution_owner_id()
                {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
                conversion
                    .validate(
                        shard.value.params(),
                        shard.value.level(),
                        device.1.parameters(destination)?,
                    )
                    .map_err(PolyBackendError::BasisConversion)?;
            }
            return Ok(InvocationOperands {
                left: Some((*value).clone()),
                right: Vec::new(),
                compact: None,
            });
        }
        if let GpuInvocation::CenteredRebase { value, destination } = request {
            if value.size() != (destination.rows, destination.columns) {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            for (_, device) in &backend.devices {
                device.parameters(destination)?;
            }
            for shard in value.shards.iter() {
                let device = backend
                    .devices
                    .iter()
                    .find(|(id, _)| *id == shard.device_id)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                let input = device.1.parameters_for_matrix(&shard.value)?;
                let output = device.1.parameters(destination)?;
                if input != shard.value.params() ||
                    input.execution_owner_id() != shard.value.params().execution_owner_id() ||
                    input.ring_dimension() as usize != destination.ring_dimension ||
                    input.crt_depth() != 1 ||
                    shard.value.level() != 0 ||
                    input.execution_owner_id() != output.execution_owner_id()
                {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
            }
            return Ok(InvocationOperands {
                left: Some((*value).clone()),
                right: Vec::new(),
                compact: None,
            });
        }
        // Invocations whose operation is determined by their own descriptor or
        // payload own no ordinary operands: constants, sampled distributions,
        // hashed sources, the preimage sampler and the artifact codecs.
        match request {
            GpuInvocation::Constant { .. } |
            GpuInvocation::SampleUniform { .. } |
            GpuInvocation::SampleGaussian { .. } |
            GpuInvocation::SampleHash { .. } |
            GpuInvocation::SampleTrapdoor { .. } |
            GpuInvocation::SamplePreimage { .. } |
            GpuInvocation::ImportMatrix { .. } |
            GpuInvocation::ImportSmallMatrix { .. } |
            GpuInvocation::ImportTrapdoor { .. } |
            GpuInvocation::ImportCpuStaging { .. } |
            GpuInvocation::PackPolynomialCoefficients { .. } |
            GpuInvocation::PolynomialFromValues { .. } => return Ok(InvocationOperands::none()),
            GpuInvocation::PolynomialValues { value, .. } |
            GpuInvocation::ThresholdDecode { value, .. } |
            GpuInvocation::ExtractCoefficient { value, .. } => {
                return Ok(InvocationOperands {
                    left: Some((*value).clone()),
                    right: Vec::new(),
                    compact: None,
                })
            }
            _ => {}
        }
        let unary = match request {
            GpuInvocation::Slice { value, .. } |
            GpuInvocation::Transpose { value } |
            GpuInvocation::SumRows { value, .. } |
            GpuInvocation::ScaleInteger { value, .. } |
            GpuInvocation::Negate { value } => Some(*value),
            GpuInvocation::RingAutomorphism { value, index } => {
                if value.shards.iter().any(|shard| {
                    let n = shard.value.params().ring_dimension() as usize;
                    !n.is_power_of_two() ||
                        n > usize::MAX / 2 ||
                        *index == 0 ||
                        *index >= 2 * n ||
                        index % 2 == 0
                }) {
                    return Err(PolyBackendError::InvalidInteger);
                }
                Some(*value)
            }
            _ => None,
        };
        if let Some(value) = unary {
            return Ok(InvocationOperands {
                left: Some(value.clone()),
                right: Vec::new(),
                compact: None,
            });
        }
        if let GpuInvocation::Accumulate { request: accumulate } = request {
            let first =
                accumulate.products.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            let (primary, _, _) = scalable_product(&first.1, &first.2);
            let mut others = Vec::with_capacity(2 * accumulate.products.len());
            let mut rows = None;
            for (index, (_, left, right)) in accumulate.products.iter().enumerate() {
                let (input, fixed, _) = scalable_product(left, right);
                let output_rows = if fixed.size() == (1, 1) {
                    input.rows
                } else if fixed.columns == input.rows {
                    fixed.rows
                } else {
                    return Err(PolyBackendError::InvalidConstantShape);
                };
                if input.columns != primary.columns || rows.is_some_and(|rows| rows != output_rows)
                {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                rows = Some(output_rows);
                if index != 0 {
                    others.push(input.clone());
                }
                others.push(fixed.clone());
            }
            if let Some(bias) = &accumulate.bias {
                if bias.size() != (rows.expect("nonempty products"), primary.columns) {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                others.push((**bias).clone());
            }
            return Ok(InvocationOperands {
                left: Some(primary.clone()),
                right: others,
                compact: None,
            });
        }
        if let GpuInvocation::Concat { inputs, axis } = request {
            let (first, rest) =
                inputs.split_first().ok_or(PolyBackendError::InvalidConstantShape)?;
            if *axis == ConcatAxis::Columns && inputs.iter().any(|value| value.rows != first.rows) {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            if let Some(reference) = inputs.iter().find_map(|input| input.shards.first()) {
                if inputs.iter().flat_map(|input| input.shards.iter()).any(|shard| {
                    shard.value.level() != reference.value.level() ||
                        shard.value.params().ring_dimension() !=
                            reference.value.params().ring_dimension() ||
                        shard.value.params().moduli() != reference.value.params().moduli()
                }) {
                    return Err(PolyBackendError::GpuSubmission(
                        "concatenated inputs have incompatible rings or levels".into(),
                    ));
                }
            }
            return Ok(InvocationOperands {
                left: Some((**first).clone()),
                right: rest.par_iter().map(|input| (**input).clone()).collect(),
                compact: None,
            });
        }
        if let GpuInvocation::AddRowBlocks { blocks, right } = request {
            let blocks = blocks.par_iter().map(|input| (**input).clone()).collect::<Vec<_>>();
            return Ok(InvocationOperands {
                left: Some((*right).clone()),
                right: blocks,
                compact: None,
            });
        }
        if let GpuInvocation::Tensor { left, right } |
        GpuInvocation::TensorSumRows { left, right, .. } = request
        {
            return Ok(InvocationOperands {
                left: Some((*left).clone()),
                right: vec![(*right).clone()],
                compact: None,
            });
        }
        match request {
            GpuInvocation::Binary { operation: MatrixBinaryOp::Add, left, right } |
            GpuInvocation::Binary { operation: MatrixBinaryOp::Subtract, left, right } => {
                Ok(InvocationOperands {
                    left: Some((*left).clone()),
                    right: vec![(*right).clone()],
                    compact: None,
                })
            }
            GpuInvocation::Binary { operation: MatrixBinaryOp::Multiply, left, right } => {
                let scales_left = gpu_matrix_multiply_scales_left(
                    left.rows,
                    left.columns,
                    right.rows,
                    right.columns,
                );
                let (scalable, fixed) = if scales_left { (*left, *right) } else { (*right, *left) };
                Ok(InvocationOperands {
                    left: Some(scalable.clone()),
                    right: vec![fixed.clone()],
                    compact: None,
                })
            }
            _ => Err(PolyBackendError::GpuSubmission(
                "invocation has no compiled prepared matrix runner".into(),
            )),
        }
    }

    /// The invocation's own operation, for a caller that identifies no IR node: a
    /// fusion result, a sampler algorithm step, an artifact or transcript import,
    /// or a direct primitive call. Operations whose kind the validated IR
    /// determines are lowered from that IR in production and reach this
    /// constructor only from such a boundary.
    fn from_invocation(
        request: &MatrixInvocation<'_>,
        operands: &InvocationOperands,
        backend: &GpuDcrtBackend,
    ) -> Result<PreparedOperation, PolyBackendError> {
        if matches!(
            request,
            GpuInvocation::PolynomialValues { .. } |
                GpuInvocation::ThresholdDecode { .. } |
                GpuInvocation::ExtractCoefficient { .. }
        ) {
            let (value, evaluation) = match request {
                GpuInvocation::PolynomialValues { value, evaluation } => (*value, *evaluation),
                // Threshold host rounding consumes coefficient-domain values;
                // the shared readback projection preserves that request.
                GpuInvocation::ThresholdDecode { value, .. } |
                GpuInvocation::ExtractCoefficient { value, .. } => (*value, false),
                _ => unreachable!(),
            };
            return Self::lower_polynomial_readback(value, evaluation, backend);
        }
        if let GpuInvocation::CenteredExtendSmall { value, destination } = request {
            let bound =
                value.shards.first().map(|shard| shard.value.bound().clone()).unwrap_or_default();
            return Ok(PreparedOperation::CenteredExtendCompact {
                destination: (**destination).clone(),
                bound,
            });
        }
        if let GpuInvocation::MultiplySmallRhs { right, .. } = request {
            return Ok(PreparedOperation::MultiplyCompact {
                columns: right.columns,
                inner: right.rows,
            });
        }
        if let GpuInvocation::MultiplySmallRhsRowBlocks { .. } = request {
            return Err(PolyBackendError::GpuSubmission(
                "compact row-block products are admitted as one invocation per block".into(),
            ));
        }
        let decomposition = match request {
            GpuInvocation::GadgetDecompose { value, small, digit_count } => {
                Some((vec![*value], *small, *digit_count))
            }
            GpuInvocation::GadgetDecomposeRowBlocks { blocks, small, digit_count } => {
                Some((blocks.to_vec(), *small, *digit_count))
            }
            _ => None,
        };
        if let Some((blocks, small, digit_count)) = decomposition {
            let first = *blocks.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            let shard = first.shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            let layout = shard
                .value
                .params()
                .compact_decomposition_layout(small, digit_count)
                .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
            let operation = PreparedOperation::decompose_with_layout(
                small,
                digit_count,
                blocks.iter().map(|v| v.rows).collect(),
                layout,
                None,
            );
            operation.output_rows(Some(first), &operands.right)?;
            return Ok(operation);
        }
        if let GpuInvocation::CrtRecompose {
            plaintext_moduli,
            reconstruction_coefficients,
            destination,
            ..
        } = request
        {
            let plaintext_moduli = plaintext_moduli
                .par_iter()
                .map(|p| p.to_u64().filter(|p| *p != 0))
                .collect::<Option<Vec<_>>>()
                .ok_or(PolyBackendError::InvalidInteger)?;
            return Ok(PreparedOperation::crt_recompose(
                (*destination).clone(),
                plaintext_moduli,
                reconstruction_coefficients.to_vec(),
            ));
        }
        let rns = match request {
            GpuInvocation::RnsModUp {
                value,
                destination,
                source_moduli,
                digit_size,
                normalize,
            } => Some((
                *value,
                *destination,
                *source_moduli,
                GpuMatrixRnsConversion::Up { digit_size: *digit_size, normalize: *normalize },
            )),
            GpuInvocation::RnsModDown { value, destination, source_moduli, plaintext_modulus } => {
                Some((
                    *value,
                    *destination,
                    *source_moduli,
                    GpuMatrixRnsConversion::Down { plaintext_modulus: *plaintext_modulus },
                ))
            }
            _ => None,
        };
        if let Some((value, destination, source_moduli, conversion)) = rns {
            return Ok(PreparedOperation::rns_conversion(
                destination.clone(),
                source_moduli.to_vec(),
                conversion,
                value.rows,
                value.columns,
                backend.devices[0].1.parameters(destination)?,
            )?);
        }
        let conversion = match request {
            GpuInvocation::ReduceModulus { destination, .. } => {
                Some((destination, GpuMatrixModulusConversion::Reduce))
            }
            GpuInvocation::ModulusSwitch { destination, .. } => {
                Some((destination, GpuMatrixModulusConversion::Round))
            }
            GpuInvocation::CenteredExtend { destination, .. } => {
                Some((destination, GpuMatrixModulusConversion::CenteredExtend))
            }
            GpuInvocation::BlockModSwitch { destination, plaintext_modulus, .. } => Some((
                destination,
                GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus: *plaintext_modulus },
            )),
            _ => None,
        };
        if let Some((destination, conversion)) = conversion {
            return Ok(PreparedOperation::modulus_conversion((**destination).clone(), conversion));
        }
        if let GpuInvocation::CenteredRebase { destination, .. } = request {
            return Ok(PreparedOperation::centered_rebase((**destination).clone()));
        }
        if let GpuInvocation::ImportMatrix { ty, bytes } = request {
            let (version, format, level, rows, columns, max_coefficient_bits, bytes_per, _) =
                super::decode_compact_matrix(bytes)?;
            let parameters = backend.devices[0].1.parameters(ty)?;
            let evaluation = match i32::from(format) {
                mxx_primitives::poly::dcrt::gpu::GPU_POLY_FORMAT_COEFF => false,
                mxx_primitives::poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL => true,
                _ => return Err(PolyBackendError::InvalidInteger),
            };
            if version != 1 ||
                rows != ty.rows ||
                columns != ty.columns ||
                usize::try_from(level).ok() != Some(parameters.crt_depth() - 1) ||
                usize::from(bytes_per) != usize::from(max_coefficient_bits).div_ceil(8) ||
                parameters.ring_dimension() as usize != ty.ring_dimension
            {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            return Ok(PreparedOperation::ImportMatrix {
                ty: (**ty).clone(),
                evaluation,
                max_coefficient_bits,
            });
        }
        if let GpuInvocation::SamplePreimage {
            schema,
            sigma,
            gadget_base,
            digit_count,
            trapdoor,
            public,
            ..
        } = request
        {
            backend.devices[0].1.validate_preimage_bound(
                &schema.matrix,
                *sigma,
                gadget_base,
                *digit_count,
                &schema.max_coefficient_bound,
            )?;
            if trapdoor.values.len() != backend.devices.len() ||
                public.columns != schema.matrix.rows ||
                !sigma.is_finite()
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let bound = schema
                .max_coefficient_bound
                .to_biguint()
                .ok_or(PolyBackendError::InvalidInteger)?;
            return PreparedOperation::preimage(
                backend,
                &schema.matrix,
                bound,
                *sigma,
                gadget_base,
                *digit_count,
                public.rows,
            );
        }
        if let GpuInvocation::ImportCpuStaging { ty, bytes } = request {
            let parameters = backend.devices[0].1.parameters(ty)?;
            let layout = GpuDCRTPolyMatrix::cpu_staging_layout(parameters, bytes)
                .map_err(PolyBackendError::GpuCalibration)?;
            if (layout.rows, layout.columns) != (ty.rows, ty.columns) {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            if layout.level != parameters.crt_depth() - 1 {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            return Ok(PreparedOperation::ImportStaging {
                ty: (**ty).clone(),
                evaluation: layout.is_ntt,
                bytes_per_poly: layout.bytes_per_poly,
                payload_len: bytes.len(),
            });
        }
        if let GpuInvocation::ImportSmallMatrix { schema, bytes, semantic_kind } = request {
            let (bound, _) =
                crate::backend::poly::decode_small_matrix_artifact(schema, bytes, *semantic_kind)?;
            backend.devices[0].1.parameters(&schema.matrix)?;
            return Ok(PreparedOperation::ImportCompact { ty: schema.matrix.clone(), bound });
        }
        if let GpuInvocation::SampleHash { ty, variant, gadget_base, digit_count, .. } = request {
            use mxx_ir_core::node::HashVariant;
            if let (
                HashVariant::Decomposed | HashVariant::SmallDecomposed,
                Some(base),
                Some(count),
            ) = (variant, gadget_base, digit_count)
            {
                let small = *variant == HashVariant::SmallDecomposed;
                backend.devices[0].1.validate_gadget_layout(ty, base, *count, small)?;
                if *count == 0 || !ty.rows.is_multiple_of(*count) {
                    return Err(PolyBackendError::InvalidInteger);
                }
                let parameters = backend.devices[0].1.parameters(ty)?;
                let layout = parameters
                    .compact_decomposition_layout(small, Some(*count))
                    .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
                let source_rows = ty.rows / count;
                if source_rows
                    .checked_mul(layout.rows_per_input_row)
                    .is_none_or(|rows| rows != ty.rows)
                {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                return Ok(PreparedOperation::decompose_with_layout(
                    small,
                    Some(*count),
                    vec![source_rows],
                    layout,
                    Some((**ty).clone()),
                ));
            }
            if *variant != HashVariant::Plain || gadget_base.is_some() || digit_count.is_some() {
                return Err(PolyBackendError::InvalidInteger);
            }
            backend.devices[0].1.parameters(ty)?;
            return Ok(PreparedOperation::Hash { ty: (**ty).clone() });
        }
        match request {
            GpuInvocation::SampleUniform { ty, range } => {
                return PreparedOperation::uniform(ty, range, backend.devices[0].1.parameters(ty)?);
            }
            GpuInvocation::SampleGaussian { ty, sigma, max_coefficient_bound } => {
                backend.devices[0].1.parameters(ty)?;
                return PreparedOperation::gaussian(ty, *sigma, max_coefficient_bound);
            }
            _ => {}
        }
        if let GpuInvocation::PolynomialFromValues { ty, values, evaluation } = request {
            backend.devices[0].1.parameters(ty)?;
            return Ok(PreparedOperation::Polynomial {
                ty: (**ty).clone(),
                coefficients: values.to_vec(),
                evaluation: *evaluation,
            });
        }
        if let GpuInvocation::Constant { ty, value, env } = request {
            return PreparedOperation::constant(
                ty,
                value,
                env,
                backend.devices[0].1.parameters(ty)?,
            );
        }
        let unary = match request {
            GpuInvocation::Slice { value, rows, columns } => {
                let operation = PreparedOperation::Slice {
                    rows: rows.map(|r| r.start..r.end).unwrap_or(0..value.rows),
                    columns: columns.map(|r| r.start..r.end).unwrap_or(0..value.columns),
                };
                operation.output_rows(Some(*value), &[])?;
                Some(operation)
            }
            GpuInvocation::Transpose { .. } => Some(PreparedOperation::Transpose),
            GpuInvocation::SumRows { rows, .. } => {
                let operation = PreparedOperation::SumRows(rows.to_vec());
                operation.output_rows(operands.left.as_ref(), &[])?;
                Some(operation)
            }
            GpuInvocation::ScaleInteger { scalar, .. } => {
                Some(PreparedOperation::Scale((*scalar).clone()))
            }
            GpuInvocation::RingAutomorphism { index, .. } => {
                Some(PreparedOperation::Automorphism(*index))
            }
            GpuInvocation::Negate { .. } => Some(PreparedOperation::Negate),
            _ => None,
        };
        if let Some(operation) = unary {
            return Ok(operation);
        }
        if let GpuInvocation::Accumulate { request: accumulate } = request {
            let first =
                accumulate.products.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            let (primary, _, _) = scalable_product(&first.1, &first.2);
            let mut products = Vec::with_capacity(accumulate.products.len());
            let mut rows = None;
            for (coefficient, left, right) in accumulate.products.iter() {
                let (input, fixed, scales_left) = scalable_product(left, right);
                let output_rows = if fixed.size() == (1, 1) {
                    input.rows
                } else if fixed.columns == input.rows {
                    fixed.rows
                } else {
                    return Err(PolyBackendError::InvalidConstantShape);
                };
                if input.columns != primary.columns || rows.is_some_and(|rows| rows != output_rows)
                {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
                rows = Some(output_rows);
                products.push((coefficient.clone(), scales_left));
            }
            let operation = PreparedOperation::Accumulate {
                products,
                bias: accumulate.bias.is_some(),
                rows: rows.expect("nonempty products"),
            };
            operation.output_rows(operands.left.as_ref(), &operands.right)?;
            return Ok(operation);
        }
        let operation = match request {
            GpuInvocation::Concat { axis: ConcatAxis::Rows, .. } => PreparedOperation::ConcatRows,
            GpuInvocation::Concat { inputs, axis } => {
                let first = inputs.first().ok_or(PolyBackendError::InvalidConstantShape)?;
                let diagonal = *axis == ConcatAxis::Diagonal;
                let mut rows = 0usize;
                let mut columns = 0usize;
                let mut offsets = Vec::with_capacity(inputs.len());
                for input in *inputs {
                    let end = columns
                        .checked_add(input.columns)
                        .ok_or(PolyBackendError::InvalidConstantShape)?;
                    offsets.push((if diagonal { rows } else { 0 }, columns, end));
                    if diagonal {
                        rows = rows
                            .checked_add(input.rows)
                            .ok_or(PolyBackendError::InvalidConstantShape)?;
                    }
                    columns = end;
                }
                PreparedOperation::ConcatColumns {
                    diagonal,
                    rows: if diagonal { rows } else { first.rows },
                    columns,
                    offsets,
                }
            }
            GpuInvocation::AddRowBlocks { .. } => PreparedOperation::AddRowBlocks,
            GpuInvocation::Tensor { right, .. } | GpuInvocation::TensorSumRows { right, .. } => {
                PreparedOperation::Tensor {
                    right_rows: right.rows,
                    right_columns: right.columns,
                    groups: match request {
                        GpuInvocation::TensorSumRows { rows, .. } => Some(rows.to_vec()),
                        _ => None,
                    },
                }
            }
            GpuInvocation::Binary { operation: MatrixBinaryOp::Add, .. } => PreparedOperation::Add,
            GpuInvocation::Binary { operation: MatrixBinaryOp::Subtract, .. } => {
                PreparedOperation::Subtract
            }
            GpuInvocation::Binary { operation: MatrixBinaryOp::Multiply, left, right } => {
                PreparedOperation::Multiply {
                    scales_left: gpu_matrix_multiply_scales_left(
                        left.rows,
                        left.columns,
                        right.rows,
                        right.columns,
                    ),
                }
            }
            _ => {
                return Err(PolyBackendError::GpuSubmission(
                    "invocation has no compiled prepared matrix runner".into(),
                ))
            }
        };
        operation.output_rows(operands.left.as_ref(), &operands.right)?;
        Ok(operation)
    }
    /// Confirm that a later request is exactly the admitted invocation: the same
    /// resource operation and the same operands in caller argument order.
    fn matches_lowered(&self, lowered: &LoweredMatrixInvocation) -> bool {
        self.operation == lowered.operation &&
            self.caller_ids == lowered.caller_ids &&
            self.caller_compact == lowered.caller_compact
    }

    /// Confirm that the values a primitive call binds are the admitted operands,
    /// in caller argument order.
    fn matches_caller_operands(
        &self,
        operands: &[&GpuFleetMatrix],
        compact: Option<&GpuFleetSmallMatrix>,
    ) -> bool {
        self.caller_ids.len() == operands.len() &&
            self.caller_ids.iter().zip(operands).all(|(id, value)| *id == value.id) &&
            self.caller_compact == compact.map(|value| value.id)
    }

    /// The resident compact shard of an admitted interval, when the operation
    /// consumes a compact input. Compact inputs have no replicas or fragments.
    pub(super) fn compact_shard<'a>(
        compact: Option<&'a GpuFleetSmallMatrix>,
        source: Option<PreparedMatrixSource>,
    ) -> Option<&'a GpuColumnShard<GpuSmallMatrix>> {
        match (compact, source) {
            (Some(compact), Some(PreparedMatrixSource::Shard(index))) => compact.shards.get(index),
            _ => None,
        }
    }
}

impl GpuDcrtBackend {
    /// Bind already lowered invocations to their admitted plans. Admission lowers
    /// each request once and passes the same records here, so this step makes no
    /// resource decision of its own.
    pub(super) fn compile_matrix_invocations(
        &self,
        admitted: Vec<(LoweredMatrixInvocation, GpuColumnMemoryPlan, Vec<AdmittedMatrixBinding>)>,
        prepared: Arc<PreparedMatrixInputs>,
    ) -> Result<VecDeque<CompiledMatrixInvocation>, PolyBackendError> {
        if !self.prepared_invocations.is_empty() || !self.enqueue.is_healthy() {
            return Err(PolyBackendError::GpuSubmission(
                "previous admitted batch is still pending or workers are unavailable".into(),
            ));
        }
        if admitted.is_empty() {
            return Err(PolyBackendError::GpuSubmission("admitted batch is empty".into()));
        }
        // Publication is sequential and atomic. Partial compilation owns its
        // plans and drops every lease if any later invocation is invalid.
        let mut compiled = VecDeque::with_capacity(admitted.len());
        for (lowered, plan, sources) in admitted {
            let LoweredMatrixInvocation { operation, operands, caller_ids, caller_compact, .. } =
                lowered;
            let InvocationOperands { left, right, compact } = operands;
            let left = left.as_ref();
            let right = &right;
            let compact = compact.as_ref();
            let rows = operation.output_rows(left, right)?;
            if plan.schedule.local_job_counts().len() != self.devices.len() ||
                plan.schedule.intervals().last().map_or(0, |range| range.end) !=
                    operation.output_columns(left)
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let mut counts = vec![0usize; self.devices.len()];
            let mut intervals = Vec::with_capacity(plan.schedule.intervals().len());
            if sources.len() != plan.schedule.intervals().len() {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            for (&interval, sources) in plan.schedule.intervals().iter().zip(sources) {
                let backend = &self.devices[interval.device].1;
                // Admission owns source selection. Submission consumes the
                // accepted identities, including shared normalization/replicas.
                let primary = operation.primary(left, right, interval.start);
                let AdmittedMatrixBinding {
                    parameters,
                    level,
                    evaluation,
                    left: left_source,
                    right: right_source,
                    compact: compact_source,
                } = sources;
                let source = primary
                    .zip(left_source)
                    .map(|(primary, index)| &operation.source(&prepared, primary, index).value);
                // Context and output format were selected before reserving
                // native output slots. Check input ownership, not a new layout.
                if let Some(shard) =
                    CompiledMatrixInvocation::compact_shard(compact, compact_source)
                {
                    if backend.parameters_for_small_matrix(&shard.value)? != shard.value.params() {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    }
                } else if let Some(source) = source {
                    let registered = backend.parameters_for_matrix(source)?;
                    if registered != source.params() ||
                        registered.execution_owner_id() != source.params().execution_owner_id()
                    {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    }
                }
                if operation.requires_evaluation() && !evaluation {
                    return Err(PolyBackendError::GpuSubmission(
                        "coefficient input is missing its admitted evaluation preparation".into(),
                    ));
                }
                for (right, &index) in right.iter().zip(&right_source) {
                    let input = &operation.source(&prepared, right, index).value;
                    let compatible = if matches!(operation, PreparedOperation::CrtRecompose { .. })
                    {
                        backend.parameters_for_matrix(input)? == input.params() &&
                            input.params().execution_owner_id() ==
                                parameters.execution_owner_id() &&
                            input.params().ring_dimension() == parameters.ring_dimension() &&
                            !input.is_ntt()
                    } else {
                        input.params() == &parameters &&
                            input.params().execution_owner_id() ==
                                parameters.execution_owner_id() &&
                            input.level() == level &&
                            input.is_ntt() == evaluation
                    };
                    if !compatible {
                        return Err(PolyBackendError::GpuSubmission(
                            "prepared inputs differ in parameters, ownership or format".into(),
                        ));
                    }
                }
                if let Some(source) = source {
                    operation.validate_output_workspace(
                        source.params(),
                        source.level(),
                        right,
                        (rows, interval.end - interval.start),
                    )?;
                }
                intervals.push(CompiledInterval {
                    interval,
                    destination: counts[interval.device],
                    left_source,
                    parameters,
                    level,
                    evaluation,
                    right_source,
                });
                counts[interval.device] += 1;
            }
            let mut seen = vec![false; self.devices.len()];
            for leases in &plan.devices {
                if leases.device >= self.devices.len() ||
                    seen[leases.device] ||
                    counts[leases.device] == 0 ||
                    !leases.fixed.is_empty() ||
                    !leases.outputs.is_empty() ||
                    !leases.scratch.is_empty() ||
                    !leases.prepared_fixed.is_empty()
                {
                    return Err(PolyBackendError::GpuSubmission(
                        "prepared view runner requires native output and reduction scratch claims"
                            .into(),
                    ));
                }
                seen[leases.device] = true;
                let expected = intervals
                    .iter()
                    .filter(|range| range.interval.device == leases.device)
                    .collect::<Vec<_>>();
                let mut scratch_count = 0;
                let scratch_rows = operation.scratch_rows()?;
                let workspaces =
                    operation.fixed_workspaces(&expected[0].parameters, expected[0].level)?;
                let mut workspace_count = 0;
                let capacity = plan
                    .widths()
                    .device_capacities(self.devices.len())
                    .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?[leases.device];
                let width_workspaces = operation.width_workspaces(
                    &expected[0].parameters,
                    expected[0].level,
                    capacity,
                )?;
                let mut width_count = 0;
                for reservation in &leases.prepared_scratch {
                    for (slot, request) in
                        reservation.slot_identities().iter().zip(reservation.requests())
                    {
                        for interval in &expected {
                            if !reservation.matches_parameters(&interval.parameters) {
                                return Err(PolyBackendError::GpuSubmission(
                                    "scratch has the wrong parameter context".into(),
                                ));
                            }
                        }
                        if slot.kind() == GpuPreparedSlotKind::Matrix {
                            let rows = *scratch_rows
                                .get(scratch_count)
                                .ok_or(PolyBackendError::InvalidConstantShape)?;
                            for interval in &expected {
                                if slot.level() != Some(interval.level) ||
                                    *request !=
                                        slot.matrix_request(
                                            rows,
                                            plan.widths()
                                                .device_capacities(self.devices.len())
                                                .map_err(|e| {
                                                    PolyBackendError::GpuSubmission(e.to_string())
                                                })?[leases.device],
                                            operation.scratch_evaluation(interval.evaluation),
                                        )
                                {
                                    return Err(PolyBackendError::GpuSubmission("reduction scratch differs from its compiled context or width".into()));
                                }
                            }
                            scratch_count += 1;
                        } else if !width_workspaces.is_empty() {
                            let Some(layout) = width_workspaces.get(width_count) else {
                                return Err(PolyBackendError::GpuSubmission(
                                    "unexpected width-scaled native workspace claim".into(),
                                ));
                            };
                            if slot.kind() != layout.kind ||
                                *request !=
                                    slot.workspace_request(layout.bytes, layout.alignment)
                            {
                                return Err(PolyBackendError::GpuSubmission("width-scaled workspace differs from the operation at its admitted width".into()));
                            }
                            width_count += 1;
                        } else {
                            let Some(layout) = workspaces.get(workspace_count) else {
                                return Err(PolyBackendError::GpuSubmission(
                                    "unexpected fixed native workspace claim".into(),
                                ));
                            };
                            if slot.kind() != layout.kind ||
                                *request !=
                                    slot.workspace_request(layout.bytes, layout.alignment)
                            {
                                return Err(PolyBackendError::GpuSubmission("fixed native workspace order or shape differs from the operation".into()));
                            }
                            workspace_count += 1;
                        }
                    }
                }
                if scratch_count != scratch_rows.len() ||
                    workspace_count != workspaces.len() ||
                    width_count != width_workspaces.len()
                {
                    return Err(PolyBackendError::GpuSubmission(
                        "scratch inventory differs from the native operation".into(),
                    ));
                }
                let mut offset = 0;
                for reservation in &leases.prepared_outputs {
                    for (slot, request) in
                        reservation.slot_identities().iter().zip(reservation.requests())
                    {
                        let range =
                            expected.get(offset).ok_or(PolyBackendError::InvalidConstantShape)?;
                        if !reservation.matches_parameters(&range.parameters) ||
                            operation.output_request(
                                *slot,
                                &range.parameters,
                                range.level,
                                rows,
                                range.interval.end - range.interval.start,
                                range.evaluation,
                            )? != Some(*request)
                        {
                            return Err(PolyBackendError::GpuSubmission("output reservation does not match its compiled context, range or format".into()));
                        }
                        offset += 1;
                    }
                }
                if offset != expected.len() {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
            }
            if counts.iter().zip(seen).any(|(count, seen)| (*count != 0) != seen) {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            // Retain only this invocation's actual fixed inputs. A later,
            // unrelated call cannot keep earlier normalization owners alive.
            let used = intervals
                .iter()
                .flat_map(|interval| {
                    operation
                        .primary(left, right, interval.interval.start)
                        .zip(interval.left_source)
                        .map(|(primary, source)| (primary.id, source))
                        .into_iter()
                        .chain(
                            right
                                .iter()
                                .zip(&interval.right_source)
                                .map(|(right, &source)| (right.id, source)),
                        )
                })
                .filter(|(_, source)| !matches!(source, PreparedMatrixSource::Shard(_)))
                .collect::<HashSet<_>>();
            let retained_inputs = prepared
                .par_iter()
                .filter(|(key, _)| used.contains(key))
                .map(|(key, source)| (*key, source.clone()))
                .collect();
            compiled.push_back(CompiledMatrixInvocation {
                operation,
                left: left.cloned(),
                right: right.clone(),
                compact: compact.cloned(),
                caller_ids,
                caller_compact,
                plan,
                intervals,
                prepared: Arc::new(retained_inputs),
            });
        }
        Ok(compiled)
    }

    /// Bind a host-driven step to its parameter context before choosing claims.
    pub(super) fn claim_broker(&self, parameters: &GpuDCRTPolyParams) -> PreparedClaimBroker {
        PreparedClaimBroker::new(
            parameters,
            self.prepared_ledger
                .as_ref()
                .map(|ledger| {
                    ledger.prepared_inventory().map(|(_, storage)| storage.clone()).collect()
                })
                .unwrap_or_default(),
        )
    }

    /// Summaries of the currently admitted, unconsumed invocation batch in
    /// execution order. Reading them performs no GPU work and holds no waits.
    pub fn admitted_invocation_summaries(&self) -> Vec<GpuAdmittedInvocationSummary> {
        // Canonicalize over the whole bounded batch. Persisting native matrix
        // IDs would create a new measurement class every time a protocol runs.
        // First-use numbering is deliberately ordered, unlike resource reads.
        let mut owners = HashMap::new();
        let mut owner = |id| {
            let next = owners.len();
            *owners.entry(id).or_insert(next)
        };
        self.prepared_invocations
            .iter()
            .map(|compiled| GpuAdmittedInvocationSummary {
                operation: compiled.operation.kind_name(),
                rows: compiled
                    .operation
                    .output_rows(compiled.left.as_ref(), &compiled.right)
                    .unwrap_or(0),
                columns: compiled.operation.output_columns(compiled.left.as_ref()),
                compact_output: compiled.operation.compact_bound().is_some(),
                input_owners: compiled
                    .left
                    .iter()
                    .chain(&compiled.right)
                    .map(|matrix| owner(matrix.id))
                    .collect(),
                compact_input_owner: compiled.compact.as_ref().map(|matrix| owner(matrix.id)),
                plan: compiled.plan.summary(),
            })
            .collect()
    }

    /// Confirm that an already admitted batch still matches the requests about to
    /// consume it: the same lowered operation and the same caller operands.
    pub(super) fn validate_admitted_matrix_invocations(
        &self,
        requests: &[MatrixInvocationRequest<'_>],
    ) -> Result<(), PolyBackendError> {
        if requests.len() != self.prepared_invocations.len() {
            return Err(PolyBackendError::GpuSubmission(
                "preflight batch differs from admitted invocation count".into(),
            ));
        }
        for ((placement, node, request), compiled) in
            requests.iter().zip(&self.prepared_invocations)
        {
            if *placement != 0 {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            let lowered = CompiledMatrixInvocation::lower(request, node.as_ref(), self)?;
            if !compiled.matches_lowered(&lowered) {
                return Err(PolyBackendError::GpuSubmission(
                    "preflight arguments differ from admitted invocation".into(),
                ));
            }
        }
        Ok(())
    }

    /// Execute the admitted invocation with the caller's actual operands. The
    /// values bind real owners and a payload only; the resource operation is the
    /// one admission lowered and this batch retains.
    pub(super) fn execute_admitted_matrix<M: PreparedFleetOutput>(
        &mut self,
        operands: &[&GpuFleetMatrix],
        compact: Option<&GpuFleetSmallMatrix>,
        payload: ExecutionPayload,
    ) -> Result<M, PolyBackendError> {
        let Some(front) = self.prepared_invocations.front() else {
            return Err(PolyBackendError::GpuSubmission(
                "matrix call has no matching admitted invocation".into(),
            ));
        };
        if front.operation.compact_bound().is_some() != M::COMPACT ||
            !front.matches_caller_operands(operands, compact)
        {
            return Err(PolyBackendError::GpuSubmission(
                "matrix call has no matching admitted invocation".into(),
            ));
        }
        let payload = front.operation.production_payload(payload)?;
        let CompiledMatrixInvocation {
            operation,
            left,
            right,
            compact,
            plan,
            intervals,
            prepared,
            ..
        } = self.prepared_invocations.pop_front().unwrap();
        let rows = operation.output_rows(left.as_ref(), &right)?;
        let columns = operation.output_columns(left.as_ref());
        if let Some(operation) = self.active_operation {
            self.operation_widths.insert(operation, plan.widths());
        }
        let intervals = Arc::new(intervals);
        let initialize_intervals = intervals.clone();
        let initialize_operation = operation.clone();
        let scratch_intervals = intervals.clone();
        let scratch_operation = operation.clone();
        let scratch_rows = operation.scratch_rows()?;
        let brokers = Arc::new(
            intervals.iter().map(|range| self.claim_broker(&range.parameters)).collect::<Vec<_>>(),
        );
        let mut measurement = self.admitted_measurement_sink.as_mut().map(|sink| {
            // Normalize logical owner aliases within the invocation, so fresh
            // allocations with the same layout do not create a new scenario.
            let mut owners = HashMap::new();
            let inputs = left
                .iter()
                .chain(&right)
                .map(|matrix| {
                    let next = owners.len();
                    let owner = *owners.entry(matrix.id).or_insert(next);
                    let shards = matrix
                        .shards
                        .iter()
                        .map(|shard| {
                            (
                                shard.device_id,
                                shard.global_column_start,
                                shard.value.size(),
                                shard.value.params().context_identity(),
                                shard.value.level(),
                                shard.value.is_ntt(),
                            )
                        })
                        .collect::<Vec<_>>();
                    (owner, matrix.size(), shards)
                })
                .collect::<Vec<_>>();
            let compact_layout = compact.as_ref().map(|matrix| {
                (
                    matrix.size(),
                    matrix
                        .shards
                        .iter()
                        .map(|shard| {
                            (
                                shard.device_id,
                                shard.global_column_start,
                                shard.value.size(),
                                shard.value.params().context_identity(),
                                shard.value.bound().clone(),
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            });
            let mappings = intervals
                .iter()
                .map(|range| {
                    (
                        range.interval,
                        range.destination,
                        range.parameters.context_identity(),
                        range.level,
                        range.evaluation,
                        range.left_source,
                        &range.right_source,
                    )
                })
                .collect::<Vec<_>>();
            let sigma = match &operation {
                PreparedOperation::Sample { sigma_bits, .. } |
                PreparedOperation::Preimage { sigma_bits, .. } => Some(*sigma_bits),
                _ => None,
            };
            let summary = plan.summary();
            let scenario = mxx_ir_core::encoding::hash_canonical(&(
                "compiled prepared invocation; owner events; host imports owned by dataflow",
                self.active_operation,
                operation.kind_name(),
                rows,
                columns,
                sigma,
                inputs,
                compact_layout,
                mappings,
                &summary,
            ))
            .expect("concrete measurement layout is serializable");
            crate::gpu_measurement::GpuColumnMeasurement::new(
                intervals
                    .iter()
                    .map(|range| (range.interval.device, range.parameters.clone()))
                    .collect(),
                self.prepared_ledger.as_ref().unwrap().prepared_inventory().fold(
                    std::collections::BTreeMap::<usize, Vec<Arc<GpuPreparedStorage>>>::new(),
                    |mut stores, (device, storage)| {
                        stores.entry(device).or_default().push(storage.clone());
                        stores
                    },
                ),
                sink,
                Some((self.active_operation, summary, operation.is_import(), scenario)),
                matches!(
                    operation,
                    PreparedOperation::Constant { .. } |
                        PreparedOperation::Tensor { .. } |
                        PreparedOperation::ConcatColumns { .. } |
                        PreparedOperation::Preimage { .. } |
                        PreparedOperation::ImportMatrix { .. } |
                        PreparedOperation::ImportCompact { .. } |
                        PreparedOperation::ImportStaging { .. }
                ),
            )
        });
        let outputs = GpuColumnMemoryPlan::execute(
            vec![plan],
            &mut self.enqueue,
            &mut self.devices,
            measurement.as_mut(),
            move |_, device, _, fixed, physical_outputs| {
                if !fixed.is_empty() || !physical_outputs.is_empty() {
                    return Err(GpuAdmissionError::InvalidPlan(
                        "unexpected physical output demand".into(),
                    ));
                }
                Ok(initialize_intervals
                    .iter()
                    .filter(|range| range.interval.device == device)
                    .map(|range| {
                        Ok(Some(GpuColumnShard {
                            device_id: range.parameters.device_ids()[0],
                            global_column_start: range.interval.start,
                            // Diagonal gaps are initialized once; all other
                            // destinations are completely filled by their jobs.
                            value: initialize_operation
                                .initialize_output(
                                    &range.parameters,
                                    range.level,
                                    range.evaluation,
                                    rows,
                                    range.interval.end - range.interval.start,
                                )
                                .map_err(GpuAdmissionError::NativeReservation)?,
                        }))
                    })
                    .collect::<Result<Vec<_>, GpuAdmissionError>>()?)
            },
            move |_, job, reservations| {
                let range = &scratch_intervals[job.source_interval];
                let mut matrix_index = 0;
                let mut width_index = 0;
                let width_workspaces = scratch_operation
                    .width_workspaces(&range.parameters, range.level, job.end - job.start)
                    .map_err(|error| GpuAdmissionError::InvalidPlan(error.to_string()))?;
                Ok(reservations
                    .iter()
                    .map(|reservation| {
                        reservation
                            .slot_identities()
                            .iter()
                            .zip(reservation.requests())
                            .map(|(slot, request)| {
                                if slot.kind() == GpuPreparedSlotKind::Matrix {
                                    let rows = scratch_rows[matrix_index];
                                    matrix_index += 1;
                                    slot.matrix_request(
                                        rows,
                                        job.end - job.start,
                                        scratch_operation.scratch_evaluation(range.evaluation),
                                    )
                                } else if !width_workspaces.is_empty() {
                                    let layout = width_workspaces[width_index];
                                    width_index += 1;
                                    slot.workspace_request(layout.bytes, layout.alignment)
                                } else {
                                    *request
                                }
                            })
                            .collect()
                    })
                    .collect())
            },
            move |jobs, _, instances, leases| {
                let (_, job) = jobs[0];
                let outputs = instances[0].as_mut().unwrap();
                if !leases[0].as_ref().unwrap().scratch.is_empty() {
                    return Err(GpuAdmissionError::InvalidPlan(
                        "view job has unexpected physical scratch".into(),
                    ));
                }
                let range = &intervals[job.source_interval];
                let source = operation
                    .primary(left.as_ref(), &right, range.interval.start)
                    .zip(range.left_source)
                    .map(|(primary, index)| operation.source(&prepared, primary, index));
                let output = outputs[range.destination].take().expect("retained destination");
                let rhs = right
                    .iter()
                    .zip(&range.right_source)
                    .map(|(right, &index)| operation.source(&prepared, right, index))
                    .collect::<Vec<_>>();
                let value = operation
                    .run(
                        source,
                        &rhs,
                        CompiledMatrixInvocation::compact_shard(
                            compact.as_ref(),
                            range.left_source,
                        ),
                        job.start,
                        job.end,
                        (
                            output.value,
                            0..rows,
                            job.start - range.interval.start..job.end - range.interval.start,
                        ),
                        &payload,
                        &brokers[job.source_interval],
                    )
                    .map_err(GpuAdmissionError::NativeReservation)?;
                outputs[range.destination] = Some(GpuColumnShard { value, ..output });
                Ok(())
            },
        )
        .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        let mut shards = outputs
            .into_iter()
            .flat_map(|(_, outputs)| {
                outputs.into_iter().flatten().flatten().map(|output| output.unwrap())
            })
            .collect::<Vec<_>>();
        shards.par_sort_unstable_by_key(|shard| shard.global_column_start);
        M::from_prepared(rows, columns, shards)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_primitives::{
        matrix::{
            PolyMatrix,
            gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedStorage, GpuPreparedWorkspaceLayout},
        },
        poly::dcrt::params::DCRTPolyParams,
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_ir_centered_extend_of_bounded_compact_value_uses_compact_operation() {
        if mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids().is_empty() {
            return;
        }
        let high = GpuDCRTPolyParams::new(4, vec![131_041, 131_009], 1, None);
        let low = high.select_modulus(&131_041u32.into()).unwrap();
        let matrix = ConcreteMatrixType {
            rows: 2,
            columns: 1,
            ring_dimension: 4,
            modulus: BigInt::from(low.modulus().as_ref().clone()),
        };
        let destination = ConcreteMatrixType {
            rows: 2,
            columns: 1,
            ring_dimension: 4,
            modulus: BigInt::from(high.modulus().as_ref().clone()),
        };
        for source in [
            ConcreteWireType::Preimage {
                matrix: matrix.clone(),
                max_coefficient_bound: BigInt::from(32_768u32),
            },
            ConcreteWireType::SmallMatrix {
                matrix,
                max_coefficient_bound: BigInt::from(32_768u32),
            },
        ] {
            let operation = PreparedOperation::from_ir(
                &NodeKind::CenteredExtend { modulus: destination.modulus.clone().into() },
                std::slice::from_ref(&source),
                &[ConcreteWireType::Matrix(destination.clone())],
                &ParamEnv::default(),
                &high,
            )
            .unwrap()
            .unwrap();
            assert_eq!(
                operation,
                PreparedOperation::CenteredExtendCompact {
                    destination: destination.clone(),
                    bound: 32_768u32.into()
                }
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preimage_residual_batch_claims_match_native_trace() {
        use mxx_primitives::matrix::gpu_dcrt_poly::trace_native_claims;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(3)
            .max(1);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let public = GpuDCRTPolyMatrix::zero(&params, 1, 3);
        let target = GpuDCRTPolyMatrix::zero(&params, 1, columns);
        let p1 = GpuDCRTPolyMatrix::zero(&params, 1, columns);
        let p2 = GpuDCRTPolyMatrix::zero(&params, 2, columns);
        let (single, template) = trace_native_claims(|| {
            GpuDCRTPolyMatrix::preimage_residual_batch([(&target, &public, &p1, &p2, None)])
        })
        .unwrap();
        drop(single.unwrap());
        for jobs in [2, 3] {
            let (outputs, observed) = trace_native_claims(|| {
                GpuDCRTPolyMatrix::preimage_residual_batch(
                    (0..jobs).map(|_| (&target, &public, &p1, &p2, None)),
                )
            })
            .unwrap();
            let outputs = outputs.unwrap();
            assert!(outputs.iter().all(|output| output == &target));
            let mut expected = (0..jobs)
                .flat_map(|_| template.iter().copied())
                .chain(
                    PreimageClaimPlan::residual_batch_metadata(&params, 1, columns, jobs).unwrap(),
                )
                .map(|claim| format!("{claim:?}"))
                .collect::<Vec<_>>();
            let mut observed =
                observed.into_iter().map(|claim| format!("{claim:?}")).collect::<Vec<_>>();
            expected.sort_unstable();
            observed.sort_unstable();
            assert_eq!(observed, expected, "native claims for {jobs} shared-input jobs");
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_admitted_compact_correction_workspace_reuses_tail() {
        use mxx_primitives::matrix::{
            PolyMatrixSmallRhs, SmallPolyMatrix, dcrt_poly::DCRTPolyMatrix,
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(16);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(3)
            .max(3);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 64, 30, 4, None, Some(32));
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            Some(32),
        );
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, columns, DistType::FinRingDist);
        let expected =
            DCRTPolyMatrix::gadget_decompose_row_blocks(vec![original.clone()], false, None)
                .unwrap();
        let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
        value.intt_all_in_place();
        let value = GpuFleetMatrix::from_matrix(value);
        let layout = params.compact_decomposition_layout(false, None).unwrap();
        let correction = params.matrix_gadget_correction_workspace_bytes(63, 1, 1, 32).unwrap();
        assert!(correction.additional_bytes > 0);
        assert_eq!(
            params.matrix_gadget_correction_workspace_bytes(63, 1, columns, 32).unwrap(),
            correction
        );
        let layouts = [
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompactPayload,
                bytes: GpuSmallMatrix::allocation_bytes(
                    &params,
                    expected.rows(),
                    columns,
                    &layout.max_coefficient_bound,
                )
                .unwrap(),
                alignment: 256,
            },
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::TransformWorkspace,
                bytes: correction.additional_bytes,
                alignment: correction.alignment,
            },
        ];
        let storage = Arc::new(
            GpuPreparedStorage::new(
                None,
                (0..6).into_par_iter().map(|_| GpuDCRTPolyMatrix::zero(&params, 1, 2)).collect(),
                None,
                Some(&layouts),
            )
            .unwrap(),
        );
        let readback = Arc::new(
            GpuPreparedStorage::new(
                None,
                vec![GpuDCRTPolyMatrix::zero(&params, 1, 2)],
                None,
                Some(&[GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                }]),
            )
            .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        for _ in 0..2 {
            backend
                .preflight_gpu_operations(&[(
                    0,
                    None,
                    GpuInvocation::GadgetDecompose {
                        value: &value,
                        small: false,
                        digit_count: None,
                    },
                )])
                .unwrap();
            let widths = backend
                .prepared_invocations
                .front()
                .unwrap()
                .plan
                .widths()
                .device_capacities(1)
                .unwrap();
            assert_eq!(widths, vec![2]);
            let output = backend.gadget_decompose(&value, false, None).unwrap();
            assert_eq!(output.shards().len(), 1);
            let claim = readback.slot_identity(1).unwrap().workspace_request(0, 1);
            let dispatch = readback.reserve(&[claim]).unwrap().enter(Vec::new()).unwrap();
            assert_eq!(
                output.shards()[0].value.to_canonical_coefficients().unwrap(),
                expected.to_canonical_coefficients().unwrap()
            );
            drop(dispatch.finish().unwrap());
        }
        assert!(!value.shards()[0].value.is_ntt());
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_admitted_compact_hash_matches_column_runner_and_reuses_outputs() {
        use mxx_primitives::{
            matrix::{PolyMatrixSmallRhs, SmallPolyMatrix},
            sampler::{PolyHashSampler, gpu::GpuDCRTPolyHashSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(3);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        let source_rows = 2;
        for dropped in [0, 1] {
            for small in [false, true] {
                let params = GpuDCRTPolyParams::new_with_gpu(
                    n,
                    vec![narrow[0], wide, narrow[1]],
                    4,
                    vec![device],
                    Some(1),
                    None,
                    Some(dropped),
                );
                let mut backend =
                    crate::backend::poly_gpu::gpu_backend_on(vec![params.clone()], [device]);
                let layout = params.compact_decomposition_layout(small, None).unwrap();
                let digit_count = layout.rows_per_input_row;
                let base = BigInt::from(1u8) << params.base_bits() as usize;
                let rows = source_rows * digit_count;
                let ty = ConcreteMatrixType {
                    modulus: BigInt::from(params.modulus().as_ref().clone()),
                    ring_dimension: n as usize,
                    rows,
                    columns,
                };
                // Existing production route: seeded COEFF gadget source, then
                // decomposition. Computed before the native domains close.
                let cases = (0..2)
                    .map(|i| {
                        let key: [u8; 32] = rand::random();
                        let tag = vec![i as u8 + 7; 3];
                        let expected = GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                            .sample_hash_gadget_source(
                                &params,
                                key,
                                &tag,
                                source_rows,
                                columns,
                                DistType::FinRingDist,
                            )
                            .gadget_decompose(small, Some(digit_count))
                            .unwrap();
                        assert_eq!(expected.bound(), &layout.max_coefficient_bound);
                        let bytes = expected.to_canonical_coefficients().unwrap();
                        (key, tag, bytes)
                    })
                    .collect::<Vec<_>>();
                let output_bytes = GpuSmallMatrix::allocation_bytes(
                    &params,
                    rows,
                    columns,
                    &layout.max_coefficient_bound,
                )
                .unwrap();
                let mut layouts = vec![
                    GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::CompactPayload,
                        bytes: output_bytes,
                        alignment: 256,
                    };
                    4
                ];
                let correction = params
                    .matrix_gadget_correction_workspace_bytes(
                        params.crt_depth() - 1,
                        source_rows,
                        1,
                        dropped,
                    )
                    .unwrap();
                if !small && correction.additional_bytes != 0 {
                    layouts.push(GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::TransformWorkspace,
                        bytes: correction.additional_bytes,
                        alignment: correction.alignment,
                    });
                }
                // Width-two COEFF scratch forces partial waves over five columns:
                // the hashed source and its correction copy for each range.
                let storage = Arc::new(
                    GpuPreparedStorage::new(
                        None,
                        (0..8)
                            .into_par_iter()
                            .map(|_| GpuDCRTPolyMatrix::zero(&params, source_rows, 2))
                            .collect(),
                        None,
                        Some(&layouts),
                    )
                    .unwrap(),
                );
                let readback = Arc::new(
                    GpuPreparedStorage::new(
                        None,
                        vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
                        None,
                        Some(&[GpuPreparedWorkspaceLayout {
                            kind: GpuPreparedSlotKind::CompletionEvent,
                            bytes: 0,
                            alignment: 1,
                        }]),
                    )
                    .unwrap(),
                );
                backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
                let variant = if small {
                    mxx_ir_core::node::HashVariant::SmallDecomposed
                } else {
                    mxx_ir_core::node::HashVariant::Decomposed
                };
                let requests = cases
                    .iter()
                    .map(|(_, tag, _)| {
                        (
                            0,
                            None,
                            GpuInvocation::SampleHash {
                                ty: &ty,
                                variant,
                                tag_bytes: tag.len(),
                                gadget_base: Some(&base),
                                digit_count: Some(digit_count),
                            },
                        )
                    })
                    .collect::<Vec<_>>();
                for bad in [
                    GpuInvocation::SampleHash {
                        ty: &ty,
                        variant,
                        tag_bytes: 3,
                        gadget_base: Some(&base),
                        digit_count: Some(0),
                    },
                    GpuInvocation::SampleHash {
                        ty: &ty,
                        variant,
                        tag_bytes: 3,
                        gadget_base: None,
                        digit_count: Some(digit_count),
                    },
                    GpuInvocation::SampleHash {
                        ty: &ty,
                        variant,
                        tag_bytes: 3,
                        gadget_base: Some(&base),
                        digit_count: Some(digit_count + 1),
                    },
                    GpuInvocation::SampleHash {
                        ty: &ty,
                        variant: mxx_ir_core::node::HashVariant::Plain,
                        tag_bytes: 3,
                        gadget_base: Some(&base),
                        digit_count: None,
                    },
                ] {
                    assert!(backend.preflight_gpu_operations(&[(0, None, bad)]).is_err());
                    assert!(backend.prepared_invocations.is_empty());
                }
                for _ in 0..2 {
                    backend.preflight_gpu_operations(&requests).unwrap();
                    let widths = backend
                        .prepared_invocations
                        .front()
                        .unwrap()
                        .plan
                        .widths()
                        .device_capacities(1)
                        .unwrap();
                    assert_eq!(widths, vec![2]);
                    // A plain hash cannot consume a compact output capability.
                    assert!(backend.sample_hash(&ty, cases[0].0, &cases[0].1).is_err());
                    // Prepared submissions use the operation and parameters
                    // compiled during preflight; callers submit that same operation.
                    let outputs = cases
                        .iter()
                        .map(|(key, tag, _)| {
                            if small {
                                backend.sample_hash_small_decomposed(
                                    &ty,
                                    *key,
                                    tag,
                                    &base,
                                    digit_count,
                                )
                            } else {
                                backend.sample_hash_decomposed(&ty, *key, tag, &base, digit_count)
                            }
                            .unwrap()
                        })
                        .collect::<Vec<_>>();
                    assert!(backend.prepared_invocations.is_empty());
                    for (output, (_, _, expected)) in outputs.iter().zip(&cases) {
                        assert_eq!(output.size(), (rows, columns));
                        assert_eq!(output.shards().len(), 1);
                        let shard = &output.shards()[0];
                        assert_eq!(shard.global_column_start, 0);
                        assert_eq!(shard.value.bound(), &layout.max_coefficient_bound);
                        let claim = readback.slot_identity(1).unwrap().workspace_request(0, 1);
                        let dispatch =
                            readback.reserve(&[claim]).unwrap().enter(Vec::new()).unwrap();
                        assert_eq!(&shard.value.to_canonical_coefficients().unwrap(), expected);
                        drop(dispatch.finish().unwrap());
                    }
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_admitted_compact_centered_extension_preserves_payload_and_shards() {
        use mxx_primitives::matrix::{PolyMatrixSmallRhs, SmallPolyMatrix};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(3);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        let cpu_low = DCRTPolyParams::new(n, 2, 17, 4, Some(narrow.clone()), None);
        let root = GpuDCRTPolyParams::new_with_gpu(
            n,
            vec![narrow[0], wide, narrow[1]],
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let related = |primes: Vec<u64>| {
            GpuDCRTPolyParams::new_with_gpu(n, primes, 4, vec![device], Some(1), Some(&root), None)
        };
        let low = related(narrow.clone());
        let other = related(vec![wide]);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on(
            vec![root.clone(), low.clone(), other.clone()],
            [device],
        );
        // Two differently sized resident compact shards in the low basis.
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu_low,
            2,
            columns,
            DistType::FinRingDist,
        );
        let split = columns / 2 + 1;
        let shards = [(0, split), (split, columns)]
            .iter()
            .map(|&(start, end)| {
                let value =
                    GpuDCRTPolyMatrix::from_cpu_matrix(&low, &original.slice_columns(start, end))
                        .gadget_decompose(false, None)
                        .unwrap();
                GpuColumnShard { device_id: device, global_column_start: start, value }
            })
            .collect::<Vec<_>>();
        let rows = shards[0].value.rows_count();
        let bound = shards[0].value.bound().clone();
        // Existing whole-copy primitive and the basis-independent payload agree.
        let expected = shards
            .iter()
            .map(|shard| {
                let bytes = shard.value.to_canonical_coefficients().unwrap();
                let extended = shard.value.centered_extend(&root).unwrap();
                assert_eq!(extended.to_canonical_coefficients().unwrap(), bytes);
                assert_eq!(extended.bound(), &bound);
                bytes
            })
            .collect::<Vec<_>>();
        let input = GpuFleetSmallMatrix::new(rows, columns, shards);
        let ty = |params: &GpuDCRTPolyParams, rows: usize| ConcreteMatrixType {
            modulus: BigInt::from(params.modulus().as_ref().clone()),
            ring_dimension: n as usize,
            rows,
            columns,
        };
        let destination = ty(&root, rows);
        let layouts = vec![
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompactPayload,
                bytes: GpuSmallMatrix::allocation_bytes(&root, rows, split, &bound).unwrap(),
                alignment: 256,
            };
            4
        ];
        let storage = Arc::new(
            GpuPreparedStorage::new(
                None,
                (0..2).map(|_| GpuDCRTPolyMatrix::zero(&root, 1, 1)).collect(),
                None,
                Some(&layouts),
            )
            .unwrap(),
        );
        // Readback permits belong to the context being read: root outputs and
        // low-basis inputs.
        let readback = |params: &GpuDCRTPolyParams| {
            Arc::new(
                GpuPreparedStorage::new(
                    None,
                    vec![GpuDCRTPolyMatrix::zero(params, 1, 1)],
                    None,
                    Some(&[GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::CompletionEvent,
                        bytes: 0,
                        alignment: 1,
                    }]),
                )
                .unwrap(),
            )
        };
        let (readback, readback_low) = (readback(&root), readback(&low));
        backend
            .prepare_memory(
                vec![(0, storage), (0, readback.clone()), (0, readback_low.clone())],
                true,
            )
            .unwrap();
        // Wrong shape, and a target basis that does not contain the source.
        for bad in [ty(&root, rows + 1), ty(&other, rows)] {
            assert!(
                backend
                    .preflight_gpu_operations(&[(
                        0,
                        None,
                        GpuInvocation::CenteredExtendSmall { value: &input, destination: &bad },
                    )])
                    .is_err()
            );
            assert!(backend.prepared_invocations.is_empty());
        }
        let request = [(
            0,
            None,
            GpuInvocation::CenteredExtendSmall { value: &input, destination: &destination },
        )];
        for _ in 0..2 {
            backend.preflight_gpu_operations(&request).unwrap();
            // The same operand is required; a distinct logical value over the
            // same resident shards is not the admitted invocation.
            let alias = GpuFleetSmallMatrix {
                id: NEXT_FLEET_VALUE_ID.fetch_add(1, Ordering::Relaxed),
                ..input.clone()
            };
            assert!(backend.centered_extend_small(&alias, &destination).is_err());
            let output = backend.centered_extend_small(&input, &destination).unwrap();
            assert!(backend.prepared_invocations.is_empty());
            assert_eq!(output.size(), (rows, columns));
            assert_eq!(output.shards().len(), 2);
            for ((shard, source), expected) in
                output.shards().iter().zip(input.shards.iter()).zip(&expected)
            {
                assert_eq!(shard.global_column_start, source.global_column_start);
                assert_eq!(shard.value.size(), source.value.size());
                assert_eq!(shard.value.bound(), &bound);
                assert_eq!(shard.value.params().to_crt().0, root.to_crt().0);
                for (store, value) in [(&readback, &shard.value), (&readback_low, &source.value)] {
                    let claim = store.slot_identity(1).unwrap().workspace_request(0, 1);
                    let dispatch = store.reserve(&[claim]).unwrap().enter(Vec::new()).unwrap();
                    assert_eq!(&value.to_canonical_coefficients().unwrap(), expected);
                    drop(dispatch.finish().unwrap());
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_admitted_compact_multiplication_matches_cpu_over_partial_waves() {
        use mxx_primitives::matrix::{CpuSmallMatrix, PolyMatrixSmallRhs, SmallPolyMatrix};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(3);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        let primes = vec![narrow[0], wide, narrow[1]];
        let cpu = DCRTPolyParams::new(n, 3, 54, 4, Some(primes.clone()), None);
        let params =
            GpuDCRTPolyParams::new_with_gpu(n, primes, 4, vec![device], Some(1), None, None);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on(vec![params.clone()], [device]);
        let level = params.crt_depth() - 1;
        // CPU oracle: compact RHS from decomposition, ordinary LHS in three row
        // blocks, and their exact products.
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, columns, DistType::FinRingDist);
        let compact = original.gadget_decompose(false, None).unwrap();
        let inner = compact.rows();
        let bound = compact.max_coefficient_bound().clone();
        let lhs_cpu =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 3, inner, DistType::FinRingDist);
        let expected = lhs_cpu.multiply_small_rhs(&compact).unwrap();
        let split = columns / 2 + 1;
        let rhs = GpuFleetSmallMatrix::new(
            inner,
            columns,
            [(0, split), (split, columns)]
                .iter()
                .map(|&(start, end)| {
                    let piece = CpuSmallMatrix::new(
                        compact.value().slice_columns(start, end),
                        bound.clone(),
                    )
                    .unwrap();
                    let value = GpuSmallMatrix::from_canonical_coefficients(
                        &params,
                        inner,
                        end - start,
                        bound.clone(),
                        &piece.to_canonical_coefficients().unwrap(),
                    )
                    .unwrap();
                    GpuColumnShard { device_id: device, global_column_start: start, value }
                })
                .collect(),
        );
        let lhs =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &lhs_cpu));
        assert!(lhs.shards()[0].value.is_ntt());
        let blocks = [(0, 1), (1, 3)]
            .iter()
            .map(|&(start, end)| {
                GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
                    &params,
                    &lhs_cpu.slice_rows(start, end),
                ))
            })
            .collect::<Vec<_>>();
        let wrong = GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
            &params,
            &lhs_cpu.slice_columns(0, inner - 1),
        ));
        // Width-two expansion workspace forces partial waves over both shards.
        // Each of the three admitted invocations selects its own pair.
        let workspaces = params.small_rhs_workspaces(level, inner, 2).unwrap();
        assert_eq!(workspaces.len(), 2);
        let workspace_slots = workspaces.repeat(3);
        let storage = Arc::new(
            GpuPreparedStorage::new(
                None,
                (0..8)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, 3, split))
                    .collect(),
                None,
                Some(&workspace_slots),
            )
            .unwrap(),
        );
        let mut layouts = vec![params.rns_transfer_workspace(level, 3, columns).unwrap()];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            params.crt_depth(),
        ));
        let readback = Arc::new(
            GpuPreparedStorage::new(
                None,
                vec![GpuDCRTPolyMatrix::zero(&params, 3, columns)],
                None,
                Some(&layouts),
            )
            .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        assert!(
            backend
                .preflight_gpu_operations(&[(
                    0,
                    None,
                    GpuInvocation::MultiplySmallRhs { left: &wrong, right: &rhs },
                )])
                .is_err()
        );
        assert!(backend.prepared_invocations.is_empty());
        let block_refs = blocks.iter().collect::<Vec<_>>();
        let requests = [
            (0, None, GpuInvocation::MultiplySmallRhs { left: &lhs, right: &rhs }),
            (
                0,
                None,
                GpuInvocation::MultiplySmallRhsRowBlocks { blocks: &block_refs, right: &rhs },
            ),
        ];
        let read = |shard: &GpuColumnShard<GpuDCRTPolyMatrix>| {
            assert!(shard.value.is_ntt());
            let events = shard.value.rns_store_completion_events().unwrap();
            let transfer = params
                .rns_transfer_workspace(level, shard.value.row_size(), shard.value.col_size())
                .unwrap();
            let claims = (1..2 + events)
                .map(|i| {
                    let slot = readback.slot_identity(i).unwrap();
                    slot.workspace_request(
                        if i == 1 { transfer.bytes } else { 0 },
                        slot.alignment(),
                    )
                })
                .collect::<Vec<_>>();
            let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
            let value = shard.value.to_cpu_matrix();
            drop(dispatch.finish().unwrap());
            value
        };
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert_eq!(backend.prepared_invocations.len(), 3);
            let widths = backend
                .prepared_invocations
                .front()
                .unwrap()
                .plan
                .widths()
                .device_capacities(1)
                .unwrap();
            assert_eq!(widths, vec![2]);
            // The admitted batch is observable before consumption: actual owner
            // intervals, admitted widths, exact wave classes and claimed resources.
            let summaries = backend.admitted_invocation_summaries();
            assert_eq!(
                summaries.iter().map(|summary| summary.input_owners.clone()).collect::<Vec<_>>(),
                vec![vec![0], vec![2], vec![3]],
                "equal-sized or derived inputs do not establish shared matrix owners"
            );
            assert!(
                summaries.iter().all(|summary| summary.compact_input_owner == Some(1)),
                "the compact RHS is shared across the whole admitted batch"
            );
            assert_eq!(
                summaries.iter().map(|s| (s.operation, s.rows, s.columns)).collect::<Vec<_>>(),
                vec![
                    ("MultiplyCompact", 3, columns),
                    ("MultiplyCompact", 1, columns),
                    ("MultiplyCompact", 2, columns)
                ]
            );
            let plan = &summaries[0].plan;
            assert_eq!(plan.widths, vec![2]);
            assert_eq!(plan.columns, columns);
            assert_eq!(
                plan.schedule.intervals().iter().map(|i| (i.start, i.end)).collect::<Vec<_>>(),
                vec![(0, split), (split, columns)]
            );
            assert_eq!(plan.wave_count, split.div_ceil(2) + (columns - split).div_ceil(2));
            assert_eq!(
                plan.wave_classes.iter().map(|c| c.multiplicity).sum::<usize>(),
                plan.wave_count
            );
            assert_eq!(plan.devices.len(), 1);
            assert_eq!(
                plan.devices[0]
                    .outputs
                    .iter()
                    .filter(|c| c.kind ==
                        crate::gpu_memory::GpuAdmittedResourceKind::Prepared(
                            GpuPreparedSlotKind::Matrix
                        ))
                    .count(),
                2
            );
            assert_eq!(
                plan.devices[0]
                    .scratch
                    .iter()
                    .filter(|c| c.kind ==
                        crate::gpu_memory::GpuAdmittedResourceKind::Prepared(
                            GpuPreparedSlotKind::CompactWorkspace
                        ))
                    .count(),
                2
            );
            assert!(plan.devices[0].scratch.iter().all(|c| c.bytes > 0));
            assert!(serde_json::to_string(plan).is_ok());
            // Row blocks are not the admitted first invocation.
            assert!(backend.multiply_small_rhs(&blocks[0], &rhs).is_err());
            let full = backend.multiply_small_rhs(&lhs, &rhs).unwrap();
            let parts = backend.multiply_small_rhs_row_blocks(&block_refs, &rhs).unwrap();
            assert!(backend.prepared_invocations.is_empty());
            assert_eq!(full.size(), (3, columns));
            assert_eq!(full.shards().len(), 2);
            for shard in full.shards() {
                let start = shard.global_column_start;
                assert_eq!(
                    read(shard),
                    expected.slice_columns(start, start + shard.value.col_size())
                );
            }
            for (part, (start, end)) in parts.iter().zip([(0, 1), (1, 3)]) {
                assert_eq!(part.size(), (end - start, columns));
                for shard in part.shards() {
                    let first = shard.global_column_start;
                    assert_eq!(
                        read(shard),
                        expected
                            .slice_rows(start, end)
                            .slice_columns(first, first + shard.value.col_size())
                    );
                }
            }
        }
        for shard in rhs.shards() {
            assert_eq!(shard.value.bound(), &bound);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_readbacks_keep_related_parameter_contexts_separate() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let primes = DCRTPolyParams::new(n, 2, 30, 4, None, None).to_crt().0;
        let parent = GpuDCRTPolyParams::new_with_gpu(
            n,
            primes.clone(),
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let parameters = primes
            .iter()
            .map(|&prime| {
                GpuDCRTPolyParams::new_with_gpu(
                    n,
                    vec![prime],
                    4,
                    vec![device],
                    Some(1),
                    Some(&parent),
                    None,
                )
            })
            .collect::<Vec<_>>();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on(parameters.clone(), [device]);
        let originals = parameters
            .iter()
            .map(|params| GpuDCRTPolyMatrix::zero(params, 2, 3))
            .collect::<Vec<_>>();
        let expected =
            originals.iter().map(GpuDCRTPolyMatrix::to_compact_bytes).collect::<Vec<_>>();
        let storages = parameters
            .iter()
            .map(|params| {
                let store = params
                    .compact_transfer_workspace(0, 2, 3, GpuCompactTransferKind::Store)
                    .unwrap();
                let stream = GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::SubmissionStream,
                    bytes: 0,
                    alignment: 1,
                };
                (
                    0,
                    Arc::new(
                        GpuPreparedStorage::new(
                            None,
                            vec![GpuDCRTPolyMatrix::zero(params, 2, 3)],
                            None,
                            Some(&[stream, store]),
                        )
                        .unwrap(),
                    ),
                )
            })
            .collect();
        backend.prepare_memory(storages, true).unwrap();
        let values = originals.into_iter().map(GpuFleetMatrix::from_matrix).collect::<Vec<_>>();
        for _ in 0..2 {
            for (value, expected) in values.iter().zip(&expected) {
                assert_eq!(&backend.matrix_to_bytes(value).unwrap(), expected);
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_admitted_imports_and_exports_preserve_canonical_bytes() {
        use crate::backend::poly::encode_small_matrix_artifact;
        use mxx_ir_core::artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind};
        use mxx_primitives::matrix::{
            PolyMatrixSmallRhs, SmallPolyMatrix, gpu_dcrt_poly::GpuCompactTransferKind,
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(3);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        let primes = vec![narrow[0], wide, narrow[1]];
        let cpu = DCRTPolyParams::new(n, 3, 54, 4, Some(primes.clone()), None);
        let params =
            GpuDCRTPolyParams::new_with_gpu(n, primes, 4, vec![device], Some(1), None, None);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on(vec![params.clone()], [device]);
        let level = params.crt_depth() - 1;
        let rows = 2;
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        // Canonical artifacts in both formats, and a compact artifact.
        let eval_bytes = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original).to_compact_bytes();
        let coeff_bytes = {
            let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original);
            value.intt_all_in_place();
            value.to_compact_bytes()
        };
        assert_ne!(eval_bytes, coeff_bytes);
        let compact = original.clone().gadget_decompose(false, None).unwrap();
        let bound = compact.max_coefficient_bound().clone();
        let schema = ConcreteBoundedMatrixSchema {
            matrix: ConcreteMatrixType {
                modulus: BigInt::from(params.modulus().as_ref().clone()),
                ring_dimension: n as usize,
                rows: compact.rows(),
                columns,
            },
            max_coefficient_bound: BigInt::from(bound.clone()),
        };
        let artifact = encode_small_matrix_artifact(
            &schema,
            &compact.to_canonical_coefficients().unwrap(),
            SmallMatrixSemanticKind::Generic,
        )
        .unwrap();
        let ty = ConcreteMatrixType { rows, ..schema.matrix.clone() };
        let bits = |bytes: &[u8]| super::super::decode_compact_matrix(bytes).unwrap().5;
        // Inventory: retained outputs, width-two coefficient staging for
        // partial waves, the codec's stream/workspace claims for two imports
        // and one export, and compact staging with its pinned upload.
        let load = |bits| {
            params
                .compact_transfer_workspace(
                    level,
                    rows,
                    2,
                    GpuCompactTransferKind::Load { max_coefficient_bits: bits },
                )
                .unwrap()
        };
        let store = params
            .compact_transfer_workspace(level, rows, columns, GpuCompactTransferKind::Store)
            .unwrap();
        let stream = GpuPreparedWorkspaceLayout {
            kind: GpuPreparedSlotKind::SubmissionStream,
            bytes: 0,
            alignment: 1,
        };
        let event = GpuPreparedWorkspaceLayout {
            kind: GpuPreparedSlotKind::CompletionEvent,
            bytes: 0,
            alignment: 1,
        };
        let magnitude = usize::try_from(bound.bits().div_ceil(8)).unwrap().max(1);
        let layouts = vec![
            load(bits(&eval_bytes)),
            load(bits(&coeff_bytes)),
            store,
            stream,
            stream,
            stream,
            event,
            event,
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompactPayload,
                bytes: GpuSmallMatrix::allocation_bytes(&params, compact.rows(), columns, &bound)
                    .unwrap(),
                alignment: 256,
            },
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompactPayload,
                bytes: GpuSmallMatrix::allocation_bytes(&params, compact.rows(), 2, &bound)
                    .unwrap(),
                alignment: 256,
            },
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::PinnedHost,
                bytes: compact.rows() * 2 * n as usize * (1 + magnitude),
                alignment: 1,
            },
        ];
        let mut matrices =
            (0..3).map(|_| GpuDCRTPolyMatrix::zero(&params, rows, columns)).collect::<Vec<_>>();
        matrices.extend((0..2).map(|_| GpuDCRTPolyMatrix::zero(&params, rows, 2)));
        let storage =
            Arc::new(GpuPreparedStorage::new(None, matrices, None, Some(&layouts)).unwrap());
        backend.prepare_memory(vec![(0, storage)], true).unwrap();
        let wrong = ConcreteMatrixType { rows: rows + 1, ..ty.clone() };
        assert!(
            backend
                .preflight_gpu_operations(&[(
                    0,
                    None,
                    GpuInvocation::ImportMatrix { ty: &wrong, bytes: &eval_bytes },
                )])
                .is_err()
        );
        assert!(backend.prepared_invocations.is_empty());
        let requests = [
            (0, None, GpuInvocation::ImportMatrix { ty: &ty, bytes: &eval_bytes }),
            (0, None, GpuInvocation::ImportMatrix { ty: &ty, bytes: &coeff_bytes }),
            (
                0,
                None,
                GpuInvocation::ImportSmallMatrix {
                    schema: &schema,
                    bytes: &artifact,
                    semantic_kind: SmallMatrixSemanticKind::Generic,
                },
            ),
        ];
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            let summaries = backend.admitted_invocation_summaries();
            assert_eq!(
                summaries.iter().map(|s| s.operation).collect::<Vec<_>>(),
                vec!["ImportMatrix", "ImportMatrix", "ImportCompact"]
            );
            assert_eq!(summaries[0].plan.widths, vec![2]);
            assert_eq!(summaries[0].plan.wave_count, columns.div_ceil(2));
            let evaluation = backend.matrix_from_bytes(&ty, &eval_bytes).unwrap();
            let coefficient = backend.matrix_from_bytes(&ty, &coeff_bytes).unwrap();
            let small = backend
                .small_matrix_from_bytes(&schema, &artifact, SmallMatrixSemanticKind::Generic)
                .unwrap();
            assert!(backend.prepared_invocations.is_empty());
            assert_eq!(evaluation.shards().len(), 1);
            assert!(evaluation.shards()[0].value.is_ntt());
            assert!(!coefficient.shards()[0].value.is_ntt());
            // Exports run through the explicit prepared readback boundary and
            // reproduce the canonical bytes exactly.
            let exported_eval = backend.matrix_to_bytes(&evaluation).unwrap();
            let exported_coeff = backend.matrix_to_bytes(&coefficient).unwrap();
            assert_eq!(exported_eval, eval_bytes);
            assert_eq!(exported_coeff, coeff_bytes);
            assert_eq!(small.size(), (compact.rows(), columns));
            assert_eq!(small.shards()[0].value.bound(), &bound);
            assert_eq!(
                backend
                    .small_matrix_to_bytes(&small, &schema, SmallMatrixSemanticKind::Generic)
                    .unwrap(),
                artifact
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_admitted_compact_decomposition_preserves_inputs_and_reuses_outputs() {
        use mxx_primitives::matrix::{
            PolyMatrixSmallRhs, SmallPolyMatrix, dcrt_poly::DCRTPolyMatrix,
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        for dropped in [0, 1] {
            for small in [false, true] {
                let primes = vec![narrow[0], wide, narrow[1]];
                let cpu = DCRTPolyParams::new(n, 3, 54, 4, Some(primes.clone()), Some(dropped));
                let params = GpuDCRTPolyParams::new_with_gpu(
                    n,
                    primes,
                    4,
                    vec![device],
                    Some(1),
                    None,
                    Some(dropped),
                );
                let mut backend =
                    crate::backend::poly_gpu::gpu_backend_on(vec![params.clone()], [device]);
                let originals = (1..=2)
                    .into_par_iter()
                    .map(|rows| {
                        DCRTPolyUniformSampler::new().sample_uniform(
                            &cpu,
                            rows,
                            columns,
                            if small { DistType::BitDist } else { DistType::FinRingDist },
                        )
                    })
                    .collect::<Vec<_>>();
                let inputs = originals
                    .iter()
                    .enumerate()
                    .map(|(i, v)| {
                        GpuFleetMatrix::new(
                            i + 1,
                            columns,
                            (0..columns)
                                .step_by(2 + i)
                                .enumerate()
                                .map(|(j, start)| {
                                    let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                                        &params,
                                        &v.slice_columns(start, (start + 2 + i).min(columns)),
                                    );
                                    if (i + j) % 2 == 0 {
                                        value.intt_all_in_place();
                                    }
                                    GpuColumnShard {
                                        device_id: device,
                                        global_column_start: start,
                                        value,
                                    }
                                })
                                .collect(),
                        )
                    })
                    .collect::<Vec<_>>();
                let expected = [vec![&originals[0]], originals.iter().collect()]
                    .iter()
                    .map(|blocks| {
                        DCRTPolyMatrix::gadget_decompose_row_blocks(
                            blocks.iter().map(|v| (*v).clone()).collect(),
                            small,
                            None,
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                let layout = params.compact_decomposition_layout(small, None).unwrap();
                let zero = GpuSmallMatrix::new_zero(
                    &params,
                    1,
                    columns,
                    layout.max_coefficient_bound.clone(),
                )
                .unwrap();
                assert!(zero.to_canonical_coefficients().unwrap().iter().all(|byte| *byte == 0));
                drop(zero);
                let output_bytes = GpuSmallMatrix::allocation_bytes(
                    &params,
                    expected[1].rows(),
                    3,
                    &layout.max_coefficient_bound,
                )
                .unwrap();
                let layouts = vec![
                    GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::CompactPayload,
                        bytes: output_bytes,
                        alignment: 256
                    };
                    4 * columns
                ];
                let storage = Arc::new(
                    GpuPreparedStorage::new(
                        None,
                        (0..12 * columns)
                            .into_par_iter()
                            .map(|_| GpuDCRTPolyMatrix::zero(&params, 3, 3))
                            .collect(),
                        None,
                        Some(&layouts),
                    )
                    .unwrap(),
                );
                let readback = Arc::new(
                    GpuPreparedStorage::new(
                        None,
                        vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
                        None,
                        Some(&[GpuPreparedWorkspaceLayout {
                            kind: GpuPreparedSlotKind::CompletionEvent,
                            bytes: 0,
                            alignment: 1,
                        }]),
                    )
                    .unwrap(),
                );
                backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
                let blocks = inputs.iter().collect::<Vec<_>>();
                let requests = [
                    (
                        0,
                        None,
                        GpuInvocation::GadgetDecompose {
                            value: &inputs[0],
                            small,
                            digit_count: None,
                        },
                    ),
                    (
                        0,
                        None,
                        GpuInvocation::GadgetDecomposeRowBlocks {
                            blocks: &blocks,
                            small,
                            digit_count: None,
                        },
                    ),
                ];
                let bad = [(
                    0,
                    None,
                    GpuInvocation::GadgetDecompose {
                        value: &inputs[0],
                        small,
                        digit_count: Some(0),
                    },
                )];
                assert!(backend.preflight_gpu_operations(&bad).is_err());
                assert!(backend.prepared_invocations.is_empty());
                let formats = inputs
                    .iter()
                    .flat_map(|v| v.shards.iter().map(|s| s.value.is_ntt()))
                    .collect::<Vec<_>>();
                for _ in 0..2 {
                    backend.preflight_gpu_operations(&requests).unwrap();
                    assert!(backend.gadget_decompose(&inputs[1], small, None).is_err());
                    let outputs = [
                        backend.gadget_decompose(&inputs[0], small, None).unwrap(),
                        backend.gadget_decompose_row_blocks(&blocks, small, None).unwrap(),
                    ];
                    assert!(backend.prepared_invocations.is_empty());
                    for (output, expected) in outputs.iter().zip(&expected) {
                        assert_eq!(output.size(), (expected.rows(), columns));
                        for shard in output.shards() {
                            assert_eq!(shard.value.bound(), &layout.max_coefficient_bound);
                            let expected = mxx_primitives::matrix::CpuSmallMatrix::new(
                                expected.value().slice_columns(
                                    shard.global_column_start,
                                    shard.global_column_start + shard.value.columns(),
                                ),
                                layout.max_coefficient_bound.clone(),
                            )
                            .unwrap();
                            let claim = readback.slot_identity(1).unwrap().workspace_request(0, 1);
                            let dispatch =
                                readback.reserve(&[claim]).unwrap().enter(Vec::new()).unwrap();
                            assert_eq!(
                                shard.value.to_canonical_coefficients().unwrap(),
                                expected.to_canonical_coefficients().unwrap()
                            );
                            drop(dispatch.finish().unwrap());
                        }
                    }
                }
                assert_eq!(
                    formats,
                    inputs
                        .iter()
                        .flat_map(|v| v.shards.iter().map(|s| s.value.is_ntt()))
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_crt_recompose_reuses_mixed_basis_inputs() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        let cpus = [
            DCRTPolyParams::new(n, 3, 54, 4, Some(vec![narrow[0], wide, narrow[1]]), None),
            DCRTPolyParams::new(n, 2, 54, 4, Some(vec![wide, narrow[0]]), None),
            DCRTPolyParams::new(n, 1, 17, 4, Some(vec![narrow[1]]), None),
        ];
        let root = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpus[0].to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let parameters = cpus
            .iter()
            .map(|p| {
                GpuDCRTPolyParams::new_with_gpu(
                    n,
                    p.to_crt().0,
                    4,
                    vec![device],
                    Some(1),
                    Some(&root),
                    None,
                )
            })
            .collect::<Vec<_>>();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on(parameters.clone(), [device]);
        let originals = cpus
            .par_iter()
            .map(|p| {
                DCRTPolyUniformSampler::new().sample_uniform(p, 1, columns, DistType::FinRingDist)
            })
            .collect::<Vec<_>>();
        let inputs = originals
            .iter()
            .zip(&parameters)
            .enumerate()
            .map(|(i, (cpu, p))| {
                GpuFleetMatrix::new(
                    1,
                    columns,
                    (0..columns)
                        .step_by(2 + i % 2)
                        .enumerate()
                        .map(|(j, start)| {
                            let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                                p,
                                &cpu.slice_columns(start, (start + 2 + i % 2).min(columns)),
                            );
                            if j % 2 == 0 {
                                value.intt_all_in_place();
                            }
                            GpuColumnShard { device_id: device, global_column_start: start, value }
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(parameters[0].modulus().as_ref().clone()),
            ring_dimension: n as usize,
            rows: 1,
            columns,
        };
        let indices = [vec![0, 1], vec![0, 2], vec![0, 1, 2, 0, 1], vec![0, 1, 2, 0, 1]];
        let level_sets = indices
            .iter()
            .map(|indices| indices.iter().map(|&i| inputs[i].clone()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let plaintext = indices
            .iter()
            .map(|indices| {
                (0..indices.len())
                    .map(|i| BigInt::from([17, 19, 257, 263, 65537][i]))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let coefficients = indices
            .iter()
            .enumerate()
            .map(|(j, indices)| {
                (0..indices.len())
                    .map(|i| BigInt::from(-23 - i as i64 - i64::from(j == 3)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let expected = indices
            .iter()
            .enumerate()
            .map(|(j, indices)| {
                crate::backend::poly::crt_recompose_cpu(
                    &indices.iter().map(|&i| originals[i].clone()).collect::<Vec<_>>(),
                    &plaintext[j],
                    &coefficients[j],
                    &cpus[0],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let mut inventory = Vec::new();
        for (index, p) in parameters.iter().enumerate() {
            let layouts = if index == 0 {
                (0..indices.len())
                    .flat_map(|j| {
                        PreparedOperation::CrtRecompose {
                            destination: ty.clone(),
                            plaintext_moduli: plaintext[j]
                                .iter()
                                .map(|p| p.to_u64().unwrap())
                                .collect(),
                            reconstruction_coefficients: coefficients[j].clone(),
                        }
                        .fixed_workspaces(p, p.crt_depth() - 1)
                        .unwrap()
                    })
                    .collect::<Vec<_>>()
            } else {
                Vec::new()
            };
            inventory.push((
                0,
                Arc::new(
                    GpuPreparedStorage::new(
                        None,
                        (0..12 * columns)
                            .into_par_iter()
                            .map(|_| GpuDCRTPolyMatrix::zero(p, 1, 3))
                            .collect(),
                        None,
                        Some(&layouts),
                    )
                    .unwrap(),
                ),
            ));
        }
        let target = &parameters[0];
        let mut layouts =
            vec![target.rns_transfer_workspace(target.crt_depth() - 1, 1, 3).unwrap()];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            target.crt_depth(),
        ));
        let readback = Arc::new(
            GpuPreparedStorage::new(
                None,
                vec![GpuDCRTPolyMatrix::zero(target, 1, 3)],
                None,
                Some(&layouts),
            )
            .unwrap(),
        );
        inventory.push((0, readback.clone()));
        backend.prepare_memory(inventory, true).unwrap();
        let requests = (0..indices.len())
            .map(|j| {
                (
                    0,
                    None,
                    GpuInvocation::CrtRecompose {
                        levels: &level_sets[j],
                        plaintext_moduli: &plaintext[j],
                        reconstruction_coefficients: &coefficients[j],
                        destination: &ty,
                    },
                )
            })
            .collect::<Vec<_>>();
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(backend.calibration_registry.is_empty(), "production must not calibrate");
            assert!(
                backend
                    .crt_recompose(&level_sets[1], &plaintext[0], &coefficients[0], &ty)
                    .is_err()
            );
            let outputs = (0..indices.len())
                .map(|j| {
                    backend
                        .crt_recompose(&level_sets[j], &plaintext[j], &coefficients[j], &ty)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            for (output, expected) in outputs.iter().zip(&expected) {
                for shard in output.shards() {
                    assert!(shard.value.is_ntt());
                    let events = shard.value.rns_store_completion_events().unwrap();
                    let transfer = target
                        .rns_transfer_workspace(target.crt_depth() - 1, 1, shard.value.col_size())
                        .unwrap();
                    let claims = (1..2 + events)
                        .map(|i| {
                            let slot = readback.slot_identity(i).unwrap();
                            slot.workspace_request(
                                if i == 1 { transfer.bytes } else { 0 },
                                slot.alignment(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                    assert_eq!(
                        shard.value.to_cpu_matrix(),
                        expected.slice_columns(
                            shard.global_column_start,
                            shard.global_column_start + shard.value.col_size()
                        )
                    );
                    drop(dispatch.finish().unwrap());
                }
            }
            for input in &inputs {
                for (j, shard) in input.shards().iter().enumerate() {
                    assert_eq!(shard.value.is_ntt(), j % 2 != 0);
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_rns_reuses_workspace_and_normalization() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let full_cpu = DCRTPolyParams::new(n, 9, 17, 4, None, None);
        let primes = full_cpu.to_crt().0;
        let small_cpu = DCRTPolyParams::new(
            n,
            8,
            17,
            4,
            Some(primes[..8].iter().copied().rev().collect()),
            None,
        );
        let full = GpuDCRTPolyParams::new_with_gpu(n, primes, 4, vec![device], Some(1), None, None);
        let small = GpuDCRTPolyParams::new_with_gpu(
            n,
            small_cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            Some(&full),
            None,
        );
        let parameters = [full, small];
        let cpus = [full_cpu, small_cpu];
        let mut backend = crate::backend::poly_gpu::gpu_backend_on(parameters.clone(), [device]);
        let originals = cpus
            .par_iter()
            .map(|cpu| {
                DCRTPolyUniformSampler::new().sample_uniform(cpu, 3, columns, DistType::FinRingDist)
            })
            .collect::<Vec<_>>();
        let inputs = originals
            .par_iter()
            .zip(&parameters)
            .map(|(original, p)| {
                GpuFleetMatrix::new(
                    3,
                    columns,
                    (0..columns)
                        .step_by(2)
                        .enumerate()
                        .map(|(index, start)| {
                            let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                                p,
                                &original.slice_columns(start, (start + 2).min(columns)),
                            );
                            if index % 2 == 1 {
                                value.intt_all_in_place();
                            }
                            GpuColumnShard { device_id: device, global_column_start: start, value }
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let conversions = [
            GpuMatrixRnsConversion::Up { digit_size: 3, normalize: false },
            GpuMatrixRnsConversion::Up { digit_size: 3, normalize: true },
            GpuMatrixRnsConversion::Up { digit_size: 4, normalize: true },
            GpuMatrixRnsConversion::Down { plaintext_modulus: 3 },
            GpuMatrixRnsConversion::Down { plaintext_modulus: 5 },
        ];
        let target_indices = conversions
            .iter()
            .map(|c| usize::from(matches!(c, GpuMatrixRnsConversion::Down { .. })))
            .collect::<Vec<_>>();
        let types = conversions
            .iter()
            .zip(&target_indices)
            .map(|(c, &target)| ConcreteMatrixType {
                modulus: BigInt::from(parameters[target].modulus().as_ref().clone()),
                ring_dimension: n as usize,
                columns,
                rows: match c {
                    GpuMatrixRnsConversion::Up { digit_size, .. } => {
                        3 * 8usize.div_ceil(*digit_size)
                    }
                    _ => 3,
                },
            })
            .collect::<Vec<_>>();
        let expected = conversions
            .iter()
            .map(|c| match *c {
                GpuMatrixRnsConversion::Up { digit_size, normalize } => {
                    originals[1].rns_mod_up(&cpus[0], digit_size, normalize).unwrap()
                }
                GpuMatrixRnsConversion::Down { plaintext_modulus } => {
                    originals[0].rns_mod_down(&cpus[1], plaintext_modulus).unwrap()
                }
            })
            .collect::<Vec<_>>();
        let mut inventory = Vec::new();
        let mut readbacks = Vec::new();
        for (index, p) in parameters.iter().enumerate() {
            let layouts = conversions
                .iter()
                .zip(&types)
                .zip(&target_indices)
                .filter(|(_, target)| **target == index)
                .flat_map(|((conversion, ty), _)| {
                    PreparedOperation::RnsConversion {
                        destination: ty.clone(),
                        source_moduli: parameters[1 - index].moduli().to_vec(),
                        conversion: *conversion,
                    }
                    .fixed_workspaces(p, p.crt_depth() - 1)
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let storage = Arc::new(
                GpuPreparedStorage::new(
                    None,
                    (0..10 * columns)
                        .into_par_iter()
                        .map(|_| GpuDCRTPolyMatrix::zero(p, 9, 2))
                        .collect(),
                    None,
                    Some(&layouts),
                )
                .unwrap(),
            );
            let mut layouts = vec![p.rns_transfer_workspace(p.crt_depth() - 1, 9, 2).unwrap()];
            layouts.extend(std::iter::repeat_n(
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                },
                p.crt_depth(),
            ));
            let readback = Arc::new(
                GpuPreparedStorage::new(
                    None,
                    vec![GpuDCRTPolyMatrix::zero(p, 9, 2)],
                    None,
                    Some(&layouts),
                )
                .unwrap(),
            );
            inventory.push((0, storage));
            inventory.push((0, readback.clone()));
            readbacks.push(readback);
        }
        backend.prepare_memory(inventory, true).unwrap();
        let requests = conversions
            .iter()
            .zip(&types)
            .map(|(c, ty)| {
                (
                    0,
                    None,
                    match *c {
                        GpuMatrixRnsConversion::Up { digit_size, normalize } => {
                            GpuInvocation::RnsModUp {
                                value: &inputs[1],
                                destination: ty,
                                source_moduli: parameters[1].moduli(),
                                digit_size,
                                normalize,
                            }
                        }
                        GpuMatrixRnsConversion::Down { plaintext_modulus } => {
                            GpuInvocation::RnsModDown {
                                value: &inputs[0],
                                destination: ty,
                                source_moduli: parameters[0].moduli(),
                                plaintext_modulus,
                            }
                        }
                    },
                )
            })
            .collect::<Vec<_>>();
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(backend.calibration_registry.is_empty(), "production must not calibrate");
            let outputs = conversions
                .iter()
                .zip(&types)
                .map(|(c, ty)| {
                    match *c {
                        GpuMatrixRnsConversion::Up { digit_size, normalize } => backend.rns_mod_up(
                            &inputs[1],
                            ty,
                            parameters[1].moduli(),
                            digit_size,
                            normalize,
                        ),
                        GpuMatrixRnsConversion::Down { plaintext_modulus } => backend.rns_mod_down(
                            &inputs[0],
                            ty,
                            parameters[0].moduli(),
                            plaintext_modulus,
                        ),
                    }
                    .unwrap()
                })
                .collect::<Vec<_>>();
            for ((output, expected), &target) in outputs.iter().zip(&expected).zip(&target_indices)
            {
                let readback = &readbacks[target];
                for shard in output.shards() {
                    assert!(shard.value.is_ntt());
                    let events = shard.value.rns_store_completion_events().unwrap();
                    let transfer = shard
                        .value
                        .params()
                        .rns_transfer_workspace(
                            shard.value.level(),
                            shard.value.row_size(),
                            shard.value.col_size(),
                        )
                        .unwrap();
                    let claims = (1..2 + events)
                        .map(|i| {
                            let slot = readback.slot_identity(i).unwrap();
                            slot.workspace_request(
                                if i == 1 { transfer.bytes } else { 0 },
                                slot.alignment(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                    assert_eq!(
                        shard.value.to_cpu_matrix(),
                        expected.slice_columns(
                            shard.global_column_start,
                            shard.global_column_start + shard.value.col_size()
                        )
                    );
                    drop(dispatch.finish().unwrap());
                }
            }
            for input in &inputs {
                for (index, shard) in input.shards().iter().enumerate() {
                    assert_eq!(shard.value.is_ntt(), index % 2 == 0);
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_modulus_conversions_reuse_metadata_and_normalization() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let narrow = DCRTPolyParams::new(n, 2, 17, 4, None, None).to_crt().0;
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        let full_cpu =
            DCRTPolyParams::new(n, 3, 54, 4, Some(vec![narrow[0], wide, narrow[1]]), None);
        let small_cpu = DCRTPolyParams::new(n, 2, 17, 4, Some(vec![narrow[1], narrow[0]]), None);
        let full = GpuDCRTPolyParams::new_with_gpu(
            n,
            full_cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let small = GpuDCRTPolyParams::new_with_gpu(
            n,
            small_cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            Some(&full),
            None,
        );
        let parameters = [full, small];
        let cpus = [full_cpu, small_cpu];
        let mut backend = crate::backend::poly_gpu::gpu_backend_on(parameters.clone(), [device]);
        let originals = cpus
            .par_iter()
            .map(|cpu| {
                DCRTPolyUniformSampler::new().sample_uniform(cpu, 3, columns, DistType::FinRingDist)
            })
            .collect::<Vec<_>>();
        let inputs = originals
            .par_iter()
            .zip(&parameters)
            .map(|(original, parameters)| {
                GpuFleetMatrix::new(
                    3,
                    columns,
                    (0..columns)
                        .step_by(2)
                        .enumerate()
                        .map(|(index, start)| {
                            let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                                parameters,
                                &original.slice_columns(start, (start + 2).min(columns)),
                            );
                            if index % 2 == 1 {
                                value.intt_all_in_place();
                            }
                            GpuColumnShard { device_id: device, global_column_start: start, value }
                        })
                        .collect(),
                )
            })
            .collect::<Vec<_>>();
        let types = parameters
            .iter()
            .map(|p| ConcreteMatrixType {
                modulus: BigInt::from(p.modulus().as_ref().clone()),
                ring_dimension: n as usize,
                rows: 3,
                columns,
            })
            .collect::<Vec<_>>();
        let conversions = [
            GpuMatrixModulusConversion::Reduce,
            GpuMatrixModulusConversion::Round,
            GpuMatrixModulusConversion::Round,
            GpuMatrixModulusConversion::CenteredExtend,
            GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus: 3 },
            GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus: 5 },
        ];
        let expected = conversions
            .iter()
            .map(|conversion| match *conversion {
                GpuMatrixModulusConversion::Reduce => originals[0].reduce_modulus(&cpus[1]),
                GpuMatrixModulusConversion::Round => originals[0].modulus_switch(&cpus[1]),
                GpuMatrixModulusConversion::CenteredExtend => {
                    originals[1].centered_extend(&cpus[0]).unwrap()
                }
                GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus } => {
                    originals[0].block_mod_switch(&cpus[1], plaintext_modulus).unwrap()
                }
            })
            .collect::<Vec<_>>();
        let mut inventory = Vec::new();
        let mut readbacks = Vec::new();
        for (index, parameters) in parameters.iter().enumerate() {
            let layouts = (0..conversions.len())
                .flat_map(|_| {
                    PreparedOperation::ModulusConversion {
                        destination: types[index].clone(),
                        conversion: GpuMatrixModulusConversion::Round,
                    }
                    .fixed_workspaces(parameters, parameters.crt_depth() - 1)
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let storage = Arc::new(
                GpuPreparedStorage::new(
                    None,
                    (0..8 * columns)
                        .into_par_iter()
                        .map(|_| GpuDCRTPolyMatrix::zero(parameters, 3, 2))
                        .collect(),
                    None,
                    Some(&layouts),
                )
                .unwrap(),
            );
            let mut layouts =
                vec![parameters.rns_transfer_workspace(parameters.crt_depth() - 1, 3, 2).unwrap()];
            layouts.extend(std::iter::repeat_n(
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                },
                parameters.crt_depth(),
            ));
            let readback = Arc::new(
                GpuPreparedStorage::new(
                    None,
                    vec![GpuDCRTPolyMatrix::zero(parameters, 3, 2)],
                    None,
                    Some(&layouts),
                )
                .unwrap(),
            );
            inventory.push((0, storage));
            inventory.push((0, readback.clone()));
            readbacks.push(readback);
        }
        backend.prepare_memory(inventory, true).unwrap();
        let requests = conversions
            .iter()
            .map(|conversion| {
                (
                    0,
                    None,
                    match *conversion {
                        GpuMatrixModulusConversion::Reduce => GpuInvocation::ReduceModulus {
                            value: &inputs[0],
                            destination: &types[1],
                        },
                        GpuMatrixModulusConversion::Round => GpuInvocation::ModulusSwitch {
                            value: &inputs[0],
                            destination: &types[1],
                        },
                        GpuMatrixModulusConversion::CenteredExtend => {
                            GpuInvocation::CenteredExtend {
                                value: &inputs[1],
                                destination: &types[0],
                            }
                        }
                        GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus } => {
                            GpuInvocation::BlockModSwitch {
                                value: &inputs[0],
                                destination: &types[1],
                                plaintext_modulus,
                            }
                        }
                    },
                )
            })
            .collect::<Vec<_>>();
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(backend.calibration_registry.is_empty(), "production must not calibrate");
            let outputs = conversions
                .iter()
                .map(|conversion| {
                    match *conversion {
                        GpuMatrixModulusConversion::Reduce => {
                            backend.reduce_modulus(&inputs[0], &types[1])
                        }
                        GpuMatrixModulusConversion::Round => {
                            backend.modulus_switch(&inputs[0], &types[1])
                        }
                        GpuMatrixModulusConversion::CenteredExtend => {
                            backend.centered_extend(&inputs[1], &types[0])
                        }
                        GpuMatrixModulusConversion::BlockSwitch { plaintext_modulus } => {
                            backend.block_mod_switch(&inputs[0], &types[1], plaintext_modulus)
                        }
                    }
                    .unwrap()
                })
                .collect::<Vec<_>>();
            for ((output, expected), conversion) in outputs.iter().zip(&expected).zip(&conversions)
            {
                let readback = &readbacks
                    [usize::from(*conversion != GpuMatrixModulusConversion::CenteredExtend)];
                for shard in output.shards() {
                    if *conversion != GpuMatrixModulusConversion::Reduce {
                        assert!(shard.value.is_ntt());
                    }
                    let events = shard.value.rns_store_completion_events().unwrap();
                    let transfer = shard
                        .value
                        .params()
                        .rns_transfer_workspace(shard.value.level(), 3, shard.value.col_size())
                        .unwrap();
                    let mut claims = (1..2 + events)
                        .map(|i| {
                            let slot = readback.slot_identity(i).unwrap();
                            slot.workspace_request(
                                if i == 1 { transfer.bytes } else { 0 },
                                slot.alignment(),
                            )
                        })
                        .collect::<Vec<_>>();
                    if !shard.value.is_ntt() {
                        claims.insert(
                            0,
                            readback.slot_identity(0).unwrap().matrix_request(
                                3,
                                shard.value.col_size(),
                                false,
                            ),
                        );
                    }
                    let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                    assert_eq!(
                        shard.value.to_cpu_matrix(),
                        expected.slice_columns(
                            shard.global_column_start,
                            shard.global_column_start + shard.value.col_size()
                        )
                    );
                    drop(dispatch.finish().unwrap());
                }
            }
            for input in &inputs {
                for (index, shard) in input.shards().iter().enumerate() {
                    assert_eq!(shard.value.is_ntt(), index % 2 == 0);
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_centered_rebase_shares_normalization_and_typed_outputs() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let narrow = DCRTPolyParams::new(n, 1, 17, 4, None, None).to_crt().0[0];
        let wide = DCRTPolyParams::new(n, 1, 54, 4, None, None).to_crt().0[0];
        let cpu = DCRTPolyParams::new(n, 1, 54, 4, Some(vec![wide]), None);
        let target_cpu = DCRTPolyParams::new(n, 2, 54, 4, Some(vec![narrow, wide]), None);
        let target = GpuDCRTPolyParams::new_with_gpu(
            n,
            vec![narrow, wide],
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            vec![wide],
            4,
            vec![device],
            Some(1),
            Some(&target),
            None,
        );
        let narrow_cpu = DCRTPolyParams::new(n, 1, 17, 4, Some(vec![narrow]), None);
        let narrow_params = GpuDCRTPolyParams::new_with_gpu(
            n,
            vec![narrow],
            4,
            vec![device],
            Some(1),
            Some(&target),
            None,
        );
        let narrow_registration = GpuDCRTPolyParams::new_with_gpu(
            n,
            vec![narrow],
            4,
            vec![device],
            Some(1),
            Some(&target),
            None,
        );
        let mut backend = crate::backend::poly_gpu::gpu_backend_on(
            [params.clone(), target.clone(), narrow_registration],
            [device],
        );
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 3, columns, DistType::FinRingDist);
        let input = GpuFleetMatrix::new(
            3,
            columns,
            (0..columns)
                .step_by(2)
                .enumerate()
                .map(|(index, start)| {
                    let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                        &params,
                        &original.slice_columns(start, (start + 2).min(columns)),
                    );
                    if index % 2 == 1 {
                        value.intt_all_in_place();
                    }
                    GpuColumnShard { device_id: device, global_column_start: start, value }
                })
                .collect(),
        );
        let narrow_original = DCRTPolyUniformSampler::new().sample_uniform(
            &narrow_cpu,
            3,
            columns,
            DistType::FinRingDist,
        );
        let narrow_input = GpuFleetMatrix::new(
            3,
            columns,
            (0..columns)
                .step_by(2)
                .enumerate()
                .map(|(index, start)| {
                    let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                        &narrow_params,
                        &narrow_original.slice_columns(start, (start + 2).min(columns)),
                    );
                    if index % 2 == 1 {
                        value.intt_all_in_place();
                    }
                    GpuColumnShard { device_id: device, global_column_start: start, value }
                })
                .collect(),
        );
        let narrow_expected = narrow_original.centered_rebase(&target_cpu).unwrap();
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(target.modulus().as_ref().clone()),
            ring_dimension: n as usize,
            rows: 3,
            columns,
        };
        let empty_input = GpuFleetMatrix::new(3, 0, Vec::new());
        let empty_ty = ConcreteMatrixType { columns: 0, ..ty.clone() };
        let scalar = BigInt::from(-3);
        let expected = original.centered_rebase(&target_cpu).unwrap();
        let scaled = crate::backend::poly::cpu_backend([cpu.clone()])
            .scale_integer(&original, &scalar)
            .unwrap();
        let mut inventory = Vec::new();
        let mut readbacks = Vec::new();
        for parameters in [&params, &target, &narrow_params] {
            let storage = Arc::new(
                GpuPreparedStorage::new(
                    None,
                    (0..(8 * columns))
                        .into_par_iter()
                        .map(|_| GpuDCRTPolyMatrix::zero(parameters, 3, 2))
                        .collect(),
                    None,
                    None,
                )
                .unwrap(),
            );
            let mut layouts =
                vec![parameters.rns_transfer_workspace(parameters.crt_depth() - 1, 3, 2).unwrap()];
            layouts.extend(std::iter::repeat_n(
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                },
                parameters.crt_depth(),
            ));
            let readback = Arc::new(
                GpuPreparedStorage::new(
                    None,
                    vec![GpuDCRTPolyMatrix::zero(parameters, 3, 2)],
                    None,
                    Some(&layouts),
                )
                .unwrap(),
            );
            inventory.push((0, storage));
            inventory.push((0, readback.clone()));
            readbacks.push(readback);
        }
        backend.prepare_memory(inventory, true).unwrap();
        let requests = [
            (0, None, GpuInvocation::CenteredRebase { value: &input, destination: &ty }),
            (0, None, GpuInvocation::CenteredRebase { value: &input, destination: &ty }),
            (0, None, GpuInvocation::ScaleInteger { value: &input, scalar: &scalar }),
            (0, None, GpuInvocation::Negate { value: &input }),
            (0, None, GpuInvocation::CenteredRebase { value: &narrow_input, destination: &ty }),
            (
                0,
                None,
                GpuInvocation::CenteredRebase { value: &empty_input, destination: &empty_ty },
            ),
        ];
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert_eq!(
                backend.calibration_registry.len(),
                0,
                "source bases must not trigger production calibration"
            );
            let outputs = [
                backend.centered_rebase(&input, &ty).unwrap(),
                backend.centered_rebase(&input, &ty).unwrap(),
                backend.scale_integer(&input, &scalar).unwrap(),
                backend.negate(&input).unwrap(),
                backend.centered_rebase(&narrow_input, &ty).unwrap(),
            ];
            assert_eq!(backend.centered_rebase(&empty_input, &empty_ty).unwrap().size(), (3, 0));
            for (index, output) in outputs.iter().enumerate() {
                let expected = match index {
                    0 | 1 => expected.clone(),
                    2 => scaled.clone(),
                    3 => -original.clone(),
                    _ => narrow_expected.clone(),
                };
                let readback = &readbacks[usize::from(index < 2 || index == 4)];
                for shard in output.shards() {
                    if index != 3 {
                        assert!(shard.value.is_ntt());
                    }
                    let events = shard.value.rns_store_completion_events().unwrap();
                    let transfer = shard
                        .value
                        .params()
                        .rns_transfer_workspace(
                            shard.value.level(),
                            shard.value.row_size(),
                            shard.value.col_size(),
                        )
                        .unwrap();
                    let mut claims = (1..2 + events)
                        .map(|i| {
                            let slot = readback.slot_identity(i).unwrap();
                            slot.workspace_request(
                                if i == 1 {
                                    transfer.bytes
                                } else {
                                    slot.requested_backing_bytes()
                                },
                                slot.alignment(),
                            )
                        })
                        .collect::<Vec<_>>();
                    if !shard.value.is_ntt() {
                        claims.insert(
                            0,
                            readback.slot_identity(0).unwrap().matrix_request(
                                shard.value.row_size(),
                                shard.value.col_size(),
                                false,
                            ),
                        );
                    }
                    let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                    assert_eq!(
                        shard.value.to_cpu_matrix(),
                        expected.slice_columns(
                            shard.global_column_start,
                            shard.global_column_start + shard.value.col_size()
                        )
                    );
                    drop(dispatch.finish().unwrap());
                }
            }
            for (index, shard) in input.shards().iter().enumerate() {
                assert_eq!(shard.value.is_ntt(), index % 2 == 0);
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_polynomial_constants_reserve_transfer_resources() {
        use mxx_ir_core::expr::IntExpr;
        use mxx_primitives::{
            matrix::dcrt_poly::DCRTPolyMatrix,
            poly::{Poly, dcrt::poly::DCRTPoly},
        };
        use num_integer::Integer;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 17, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(params.modulus().as_ref().clone()),
            ring_dimension: n as usize,
            rows: 1,
            columns: 1,
        };
        let coefficients = (0..n)
            .into_par_iter()
            .map(|_| BigInt::from(rand::random::<i64>()) << 20usize)
            .collect::<Vec<_>>();
        let normalized = coefficients
            .par_iter()
            .map(|value| value.mod_floor(&ty.modulus).to_biguint().unwrap())
            .collect::<Vec<_>>();
        let power = BigInt::from(-3).pow(17).mod_floor(&ty.modulus).to_biguint().unwrap();
        let cases = [
            (
                ConstantMatrix::PowerOfBase {
                    base: IntExpr::constant(BigInt::from(-3)),
                    exponent: IntExpr::constant(BigInt::from(17)),
                },
                DCRTPoly::from_biguint_to_constant(&cpu, power),
            ),
            (
                ConstantMatrix::Rotation { exponent: IntExpr::constant(BigInt::from(n - 1)) },
                DCRTPoly::const_rotate_poly(&cpu, n as usize - 1),
            ),
            (
                ConstantMatrix::Polynomial {
                    coefficients: coefficients.into_iter().map(IntExpr::constant).collect(),
                },
                DCRTPoly::from_biguints(&cpu, &normalized),
            ),
            (ConstantMatrix::Polynomial { coefficients: Vec::new() }, DCRTPoly::const_zero(&cpu)),
        ];
        let expected = cases
            .iter()
            .map(|(_, poly)| DCRTPolyMatrix::from_poly_vec(&cpu, vec![vec![poly.clone()]]))
            .collect::<Vec<_>>();
        let operation = PreparedOperation::Polynomial {
            ty: ty.clone(),
            coefficients: Vec::new(),
            evaluation: false,
        };
        let layouts = (0..cases.len())
            .flat_map(|_| operation.fixed_workspaces(&params, params.crt_depth() - 1).unwrap())
            .collect::<Vec<_>>();
        let storage = Arc::new(
            GpuPreparedStorage::new(
                None,
                (0..cases.len() + 1)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, 1, 1))
                    .collect(),
                None,
                Some(&layouts),
            )
            .unwrap(),
        );
        let mut readback_layout =
            vec![params.rns_transfer_workspace(params.crt_depth() - 1, 1, 1).unwrap()];
        readback_layout.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            params.crt_depth(),
        ));
        let readback = Arc::new(
            GpuPreparedStorage::new(
                None,
                vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
                None,
                Some(&readback_layout),
            )
            .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        let env = ParamEnv::default();
        let requests = cases
            .iter()
            .map(|(value, _)| (0, None, GpuInvocation::Constant { ty: &ty, value, env: &env }))
            .collect::<Vec<_>>();
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            let actual = cases
                .iter()
                .map(|(value, _)| backend.constant_matrix(&ty, value, &env).unwrap())
                .collect::<Vec<_>>();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.size(), (1, 1));
                let shard = &actual.shards()[0];
                assert!(shard.value.is_ntt());
                let events = shard.value.rns_store_completion_events().unwrap();
                let claims = (1..2 + events)
                    .map(|i| {
                        let slot = readback.slot_identity(i).unwrap();
                        slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                    })
                    .collect::<Vec<_>>();
                let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                assert_eq!(&shard.value.to_cpu_matrix(), expected);
                drop(dispatch.finish().unwrap());
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_hash_uses_production_payload_and_cached_layout() {
        use mxx_primitives::sampler::{PolyHashSampler, gpu::GpuDCRTPolyHashSampler};
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 17, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(params.modulus().as_ref().clone()),
            ring_dimension: n as usize,
            rows: 3,
            columns,
        };
        let lengths = [0, 1, 13, 13, 65];
        // References use the original public full-matrix sampler before sealing.
        // Equal tag lengths with distinct payloads deliberately share calibration.
        let cycles = (0..2)
            .map(|_| {
                lengths
                    .iter()
                    .map(|&length| {
                        let key = rand::random();
                        let tag = (0..length).map(|_| rand::random::<u8>()).collect::<Vec<_>>();
                        let expected = GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                            .sample_hash(
                                &params,
                                key,
                                &tag,
                                ty.rows,
                                ty.columns,
                                DistType::FinRingDist,
                            )
                            .to_cpu_matrix();
                        (key, tag, expected)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let storage = Arc::new(
            GpuPreparedStorage::new(
                None,
                (0..lengths.len() + 1)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, 3, columns))
                    .collect(),
                None,
                None,
            )
            .unwrap(),
        );
        let mut layouts =
            vec![params.rns_transfer_workspace(params.crt_depth() - 1, 3, columns).unwrap()];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            params.crt_depth(),
        ));
        let readback = Arc::new(
            GpuPreparedStorage::new(
                None,
                vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
                None,
                Some(&layouts),
            )
            .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        let empty = ConcreteMatrixType { columns: 0, ..ty.clone() };
        let mut requests = lengths
            .iter()
            .map(|&tag_bytes| {
                (
                    0,
                    None,
                    GpuInvocation::SampleHash {
                        ty: &ty,
                        variant: mxx_ir_core::node::HashVariant::Plain,
                        // Native sampling receives a fixed-size seed. A
                        // different tag length must reuse the resource plan
                        // while still hashing the actual production payload.
                        tag_bytes: if tag_bytes == 1 { 0 } else { tag_bytes },
                        gadget_base: None,
                        digit_count: None,
                    },
                )
            })
            .collect::<Vec<_>>();
        requests.push((
            0,
            None,
            GpuInvocation::SampleHash {
                ty: &empty,
                variant: mxx_ir_core::node::HashVariant::Plain,
                tag_bytes: 0,
                gadget_base: None,
                digit_count: None,
            },
        ));
        for cases in cycles {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(
                backend
                    .execute_admitted_matrix::<GpuFleetMatrix>(&[], None, ExecutionPayload::None)
                    .is_err()
            );
            let actual = cases
                .iter()
                .map(|(key, tag, _)| backend.sample_hash(&ty, *key, tag).unwrap())
                .collect::<Vec<_>>();
            assert_eq!(backend.sample_hash(&empty, rand::random(), &[]).unwrap().size(), (3, 0));
            for (actual, (_, _, expected)) in actual.iter().zip(&cases) {
                assert_eq!(actual.size(), expected.size());
                for shard in actual.shards() {
                    let start = shard.global_column_start;
                    let end = start + shard.value.col_size();
                    assert!(shard.value.is_ntt());
                    let events = shard.value.rns_store_completion_events().unwrap();
                    let claims = (1..2 + events)
                        .map(|i| {
                            let slot = readback.slot_identity(i).unwrap();
                            slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                        })
                        .collect::<Vec<_>>();
                    let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                    assert_eq!(shard.value.to_cpu_matrix(), expected.slice_columns(start, end));
                    drop(dispatch.finish().unwrap());
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_sampling_preserves_bounds_order_and_reuses_outputs() {
        use mxx_primitives::{element::PolyElem, poly::Poly};
        use num_bigint::BigUint;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 17, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let ty = ConcreteMatrixType {
            modulus: BigInt::from(params.modulus().as_ref().clone()),
            ring_dimension: n as usize,
            rows: 3,
            columns,
        };
        let ranges = [
            SampleRange { minimum: BigInt::from(0), maximum: ty.modulus.clone() - 1 },
            SampleRange { minimum: BigInt::from(0), maximum: BigInt::from(1) },
            SampleRange { minimum: BigInt::from(-1), maximum: BigInt::from(1) },
        ];
        let storage = Arc::new(
            GpuPreparedStorage::new(
                None,
                (0..7)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, 3, columns))
                    .collect(),
                None,
                None,
            )
            .unwrap(),
        );
        let mut layouts =
            vec![params.rns_transfer_workspace(params.crt_depth() - 1, 3, columns).unwrap()];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            params.crt_depth(),
        ));
        let readback = Arc::new(
            GpuPreparedStorage::new(
                None,
                vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
                None,
                Some(&layouts),
            )
            .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        let bound = BigInt::from(2);
        let empty = ConcreteMatrixType { columns: 0, ..ty.clone() };
        let mut requests = ranges
            .iter()
            .map(|range| (0, None, GpuInvocation::SampleUniform { ty: &ty, range }))
            .collect::<Vec<_>>();
        requests.extend([
            (
                0,
                None,
                GpuInvocation::SampleGaussian {
                    ty: &ty,
                    sigma: 4.578,
                    max_coefficient_bound: &bound,
                },
            ),
            (
                0,
                None,
                GpuInvocation::SampleGaussian {
                    ty: &ty,
                    sigma: 0.0,
                    max_coefficient_bound: &bound,
                },
            ),
            (0, None, GpuInvocation::SampleUniform { ty: &empty, range: &ranges[1] }),
        ]);
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            let mut actual = ranges
                .iter()
                .map(|range| backend.sample_uniform(&ty, range).unwrap())
                .collect::<Vec<_>>();
            actual.push(backend.sample_gaussian(&ty, 4.578, &bound).unwrap());
            actual.push(backend.sample_gaussian(&ty, 0.0, &bound).unwrap());
            assert_eq!(backend.sample_uniform(&empty, &ranges[1]).unwrap().size(), (3, 0));
            for (kind, matrix) in actual.iter().enumerate() {
                assert_eq!(matrix.size(), (3, columns));
                for shard in matrix.shards() {
                    assert!(shard.value.is_ntt());
                    let events = shard.value.rns_store_completion_events().unwrap();
                    let claims = (1..2 + events)
                        .map(|i| {
                            let slot = readback.slot_identity(i).unwrap();
                            slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                        })
                        .collect::<Vec<_>>();
                    let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                    let sampled = shard.value.to_cpu_matrix();
                    drop(dispatch.finish().unwrap());
                    for row in 0..sampled.row_size() {
                        for col in 0..sampled.col_size() {
                            for coefficient in sampled.entry(row, col).coeffs() {
                                let value = coefficient.value();
                                let q = params.modulus();
                                assert!(value < q.as_ref());
                                match kind {
                                    0 => {}
                                    1 => assert!(value <= &BigUint::from(1u8)),
                                    2 => assert!(
                                        value <= &BigUint::from(1u8) ||
                                            q.as_ref() - value == BigUint::from(1u8)
                                    ),
                                    3 => assert!(
                                        value <= &BigUint::from(2u8) ||
                                            q.as_ref() - value <= BigUint::from(2u8)
                                    ),
                                    4 => assert_eq!(value, &BigUint::from(0u8)),
                                    _ => unreachable!(),
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_constant_gadget_rejects_zero_rows_at_every_entry() {
        use mxx_ir_core::expr::IntExpr;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let cpu = DCRTPolyParams::new(n, 2, 17, 4, None, None);
        let params = GpuDCRTPolyParams::new(n, cpu.to_crt().0, 4, None);
        let mut backend = crate::backend::poly_gpu::gpu_backend_on(
            [params.clone()],
            params.device_ids().iter().copied(),
        );
        let env = ParamEnv::default();
        for columns in [0, 1] {
            let ty = ConcreteMatrixType {
                modulus: BigInt::from(params.modulus().as_ref().clone()),
                ring_dimension: n as usize,
                rows: 0,
                columns,
            };
            for small in [false, true] {
                let value = ConstantMatrix::Gadget {
                    base: IntExpr::constant(BigInt::from(1) << params.base_bits()),
                    small,
                };
                assert!(matches!(
                    backend.devices[0].1.constant_matrix(&ty, &value, &env),
                    Err(PolyBackendError::InvalidInteger)
                ));
                assert!(matches!(
                    backend.constant_column_runner(&ty, &value, &env),
                    Err(PolyBackendError::InvalidInteger)
                ));
                assert!(matches!(
                    backend.constant_matrix(&ty, &value, &env),
                    Err(PolyBackendError::InvalidInteger)
                ));
                let request = GpuInvocation::Constant { ty: &ty, value: &value, env: &env };
                assert!(matches!(
                    CompiledMatrixInvocation::lower(&request, None, &backend),
                    Err(PolyBackendError::InvalidInteger)
                ));
            }
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_fresh_constants_share_admission_with_owned_operations() {
        use mxx_ir_core::expr::IntExpr;
        use mxx_primitives::matrix::dcrt_poly::DCRTPolyMatrix;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 17, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let digits = params.crt_bits().div_ceil(params.base_bits() as usize);
        let base = IntExpr::constant(BigInt::from(1) << params.base_bits());
        let cases = vec![
            (ConstantMatrix::Zero, DCRTPolyMatrix::zero(&cpu, 3, columns)),
            (ConstantMatrix::Identity, DCRTPolyMatrix::identity(&cpu, columns, None)),
            (
                ConstantMatrix::UnitRow { index: IntExpr::constant(BigInt::from(columns - 1)) },
                DCRTPolyMatrix::unit_row_vector(&cpu, columns, columns - 1),
            ),
            (
                ConstantMatrix::UnitColumn { index: IntExpr::constant(BigInt::from(1)) },
                DCRTPolyMatrix::unit_column_vector(&cpu, 3, 1),
            ),
            (
                ConstantMatrix::Gadget { base: base.clone(), small: false },
                DCRTPolyMatrix::gadget_matrix(&cpu, 3, None),
            ),
            (
                ConstantMatrix::Gadget { base: base.clone(), small: false },
                DCRTPolyMatrix::gadget_matrix(&cpu, 3, Some(digits)),
            ),
            (
                ConstantMatrix::Gadget { base, small: true },
                DCRTPolyMatrix::small_gadget_matrix(&cpu, 3),
            ),
            (ConstantMatrix::Zero, DCRTPolyMatrix::zero(&cpu, 3, 0)),
        ];
        let types = cases
            .iter()
            .map(|(_, matrix)| ConcreteMatrixType {
                modulus: BigInt::from(params.modulus().as_ref().clone()),
                ring_dimension: n as usize,
                rows: matrix.row_size(),
                columns: matrix.col_size(),
            })
            .collect::<Vec<_>>();
        let original =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, columns, DistType::FinRingDist);
        let input =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &original));
        let maximum_rows = types.iter().map(|ty| ty.rows).max().unwrap();
        let maximum_columns = types.iter().map(|ty| ty.columns).max().unwrap();
        let storage = Arc::new(
            GpuPreparedStorage::new(
                None,
                (0..cases.len() + 2)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, maximum_rows, maximum_columns))
                    .collect(),
                None,
                None,
            )
            .unwrap(),
        );
        let mut layouts = vec![
            params
                .rns_transfer_workspace(params.crt_depth() - 1, maximum_rows, maximum_columns)
                .unwrap(),
        ];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            params.crt_depth(),
        ));
        let readback = Arc::new(
            GpuPreparedStorage::new(
                None,
                vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
                None,
                Some(&layouts),
            )
            .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        let env = ParamEnv::default();
        let mut requests = cases
            .iter()
            .zip(&types)
            .map(|((value, _), ty)| (0, None, GpuInvocation::Constant { ty, value, env: &env }))
            .collect::<Vec<_>>();
        requests.push((0, None, GpuInvocation::Negate { value: &input }));
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            let mut actual = cases
                .iter()
                .zip(&types)
                .map(|((value, _), ty)| backend.constant_matrix(ty, value, &env).unwrap())
                .collect::<Vec<_>>();
            actual.push(backend.negate(&input).unwrap());
            let expected = cases
                .iter()
                .map(|(_, value)| value.clone())
                .chain([original.negate_out_of_place()])
                .collect::<Vec<_>>();
            for (actual, expected) in actual.iter().zip(&expected) {
                assert_eq!(actual.size(), expected.size());
                for shard in actual.shards() {
                    let start = shard.global_column_start;
                    let end = start + shard.value.col_size();
                    assert_eq!(shard.value.params().context_identity(), params.context_identity());
                    let transfer = params
                        .rns_transfer_workspace(
                            params.crt_depth() - 1,
                            expected.row_size(),
                            end - start,
                        )
                        .unwrap();
                    let events = shard.value.rns_store_completion_events().unwrap();
                    let claims = (1..2 + events)
                        .map(|i| {
                            let slot = readback.slot_identity(i).unwrap();
                            slot.workspace_request(
                                if i == 1 {
                                    transfer.bytes
                                } else {
                                    slot.requested_backing_bytes()
                                },
                                slot.alignment(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                    assert_eq!(shard.value.to_cpu_matrix(), expected.slice_columns(start, end));
                    drop(dispatch.finish().unwrap());
                }
            }
            drop(actual);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_accumulate_shares_inputs_and_reuses_outputs() {
        use mxx_primitives::{
            matrix::dcrt_poly::DCRTPolyMatrix,
            poly::{Poly, dcrt::poly::DCRTPoly},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params.clone()], [device]);
        let originals = [(3, 2), (2, columns), (3, columns), (1, 1), (3, columns)]
            .into_par_iter()
            .map(|(rows, columns)| {
                DCRTPolyUniformSampler::new().sample_uniform(
                    &cpu,
                    rows,
                    columns,
                    DistType::FinRingDist,
                )
            })
            .collect::<Vec<_>>();
        let inputs = originals
            .par_iter()
            .enumerate()
            .map(|(index, value)| {
                let width = 2 + index % 2;
                let shards = (0..value.col_size())
                    .step_by(width)
                    .map(|start| {
                        let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                            &params,
                            &value.slice_columns(start, (start + width).min(value.col_size())),
                        );
                        if index % 2 != 0 {
                            value.intt_all_in_place();
                        }
                        GpuColumnShard { device_id: device, global_column_start: start, value }
                    })
                    .collect();
                Arc::new(GpuFleetMatrix::new(value.row_size(), value.col_size(), shards))
            })
            .collect::<Vec<_>>();
        let specs = [(7i64, 0, 1), (-3, 0, 1), (2, 3, 2), (-11, 2, 3), (0, 0, 1), (1, 0, 1)];
        let products = specs
            .iter()
            .map(|&(coefficient, left, right)| {
                (BigInt::from(coefficient), inputs[left].clone(), inputs[right].clone())
            })
            .collect::<Vec<_>>();
        let request = MatrixMulAccumulateRequest { products, bias: Some(inputs[4].clone()) };
        let without_bias =
            MatrixMulAccumulateRequest { products: request.products.clone(), bias: None };
        let expected = specs
            .par_iter()
            .map(|&(coefficient, left, right)| {
                let left = &originals[left];
                let right = &originals[right];
                let product = if left.size() == (1, 1) {
                    right.multiply_poly_out_of_place(&left.entry(0, 0))
                } else if right.size() == (1, 1) {
                    left.multiply_poly_out_of_place(&right.entry(0, 0))
                } else {
                    left.multiply_out_of_place(right)
                };
                let modulus = BigInt::from(cpu.modulus().as_ref().clone());
                let residue = ((BigInt::from(coefficient) % &modulus + &modulus) % &modulus)
                    .to_biguint()
                    .unwrap();
                product
                    .multiply_poly_out_of_place(&DCRTPoly::from_biguint_to_constant(&cpu, residue))
            })
            .collect::<Vec<_>>()
            .iter()
            .fold(DCRTPolyMatrix::zero(&cpu, 3, columns), |sum, term| sum.add_out_of_place(term));
        let shapes = inputs
            .iter()
            .flat_map(|input| input.shards.iter())
            .filter(|shard| !shard.value.is_ntt())
            .map(|shard| shard.value.size())
            .chain((0..8 * columns).map(|_| (3, columns)))
            .collect::<Vec<_>>();
        let storage = Arc::new(
            GpuPreparedStorage::new(
                None,
                shapes
                    .into_par_iter()
                    .map(|(r, c)| GpuDCRTPolyMatrix::zero(&params, r, c))
                    .collect(),
                None,
                None,
            )
            .unwrap(),
        );
        let mut layouts =
            vec![params.rns_transfer_workspace(params.crt_depth() - 1, 3, columns).unwrap()];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            params.crt_depth(),
        ));
        let readback = Arc::new(
            GpuPreparedStorage::new(
                None,
                vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
                None,
                Some(&layouts),
            )
            .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        let requests = [
            (0, None, GpuInvocation::Accumulate { request: &request }),
            (0, None, GpuInvocation::Accumulate { request: &without_bias }),
        ];
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            let outputs = [
                backend.matrix_mul_accumulate(request.clone()).unwrap(),
                backend.matrix_mul_accumulate(without_bias.clone()).unwrap(),
            ];
            for (index, output) in outputs.iter().enumerate() {
                let expected = if index == 0 {
                    expected.add_out_of_place(&originals[4])
                } else {
                    expected.clone()
                };
                for shard in output.shards() {
                    let start = shard.global_column_start;
                    let end = start + shard.value.col_size();
                    assert_eq!(shard.value.params().context_identity(), params.context_identity());
                    let transfer = params
                        .rns_transfer_workspace(params.crt_depth() - 1, 3, end - start)
                        .unwrap();
                    let events = shard.value.rns_store_completion_events().unwrap();
                    let claims = (1..2 + events)
                        .map(|i| {
                            let slot = readback.slot_identity(i).unwrap();
                            slot.workspace_request(
                                if i == 1 {
                                    transfer.bytes
                                } else {
                                    slot.requested_backing_bytes()
                                },
                                slot.alignment(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                    assert_eq!(shard.value.to_cpu_matrix(), expected.slice_columns(start, end));
                    drop(dispatch.finish().unwrap());
                }
            }
            drop(outputs);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_column_and_diagonal_concat_keep_original_owners() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let base = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let related = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            Some(&base),
            None,
        );
        let params = [base, related];
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params[0].clone()], [device]);
        let rows = 3;
        let originals = [0, columns, 0, columns + 1, columns - 1]
            .into_par_iter()
            .map(|width| {
                DCRTPolyUniformSampler::new().sample_uniform(
                    &cpu,
                    rows,
                    width,
                    DistType::FinRingDist,
                )
            })
            .collect::<Vec<_>>();
        let inputs = originals
            .par_iter()
            .enumerate()
            .map(|(i, value)| {
                if value.col_size() == 0 {
                    return GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(
                        &params[i % 2],
                        value,
                    ));
                }
                let width = 2 + i % 2;
                let shards = (0..value.col_size())
                    .step_by(width)
                    .map(|start| {
                        let mut matrix = GpuDCRTPolyMatrix::from_cpu_matrix(
                            &params[i % 2],
                            &value.slice_columns(start, (start + width).min(value.col_size())),
                        );
                        if i % 2 != 0 {
                            matrix.intt_all_in_place();
                        }
                        GpuColumnShard {
                            device_id: device,
                            global_column_start: start,
                            value: matrix,
                        }
                    })
                    .collect();
                GpuFleetMatrix::new(rows, value.col_size(), shards)
            })
            .collect::<Vec<_>>();
        let refs = inputs.iter().collect::<Vec<_>>();
        let total_rows = rows * inputs.len();
        let mut inventory = Vec::new();
        let mut readbacks = Vec::new();
        for parameters in &params {
            let shapes = inputs
                .iter()
                .flat_map(|input| input.shards.iter())
                .filter(|shard| {
                    !shard.value.is_ntt() &&
                        shard.value.params().context_identity() == parameters.context_identity()
                })
                .map(|shard| shard.value.size())
                .chain((0..2 * columns * inputs.len()).map(|_| (total_rows, 3)))
                .collect::<Vec<_>>();
            inventory.push((
                0,
                Arc::new(
                    GpuPreparedStorage::new(
                        None,
                        shapes
                            .into_par_iter()
                            .map(|(r, c)| GpuDCRTPolyMatrix::zero(parameters, r, c))
                            .collect(),
                        None,
                        None,
                    )
                    .unwrap(),
                ),
            ));
            // All CRT limbs share one readback batch and completion event.
            let events = 1;
            let mut layouts = vec![
                parameters
                    .rns_transfer_workspace(parameters.crt_depth() - 1, total_rows, 3)
                    .unwrap(),
            ];
            layouts.extend(std::iter::repeat_n(
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::CompletionEvent,
                    bytes: 0,
                    alignment: 1,
                },
                events,
            ));
            let readback = Arc::new(
                GpuPreparedStorage::new(
                    None,
                    vec![GpuDCRTPolyMatrix::zero(parameters, 1, 1)],
                    None,
                    Some(&layouts),
                )
                .unwrap(),
            );
            inventory.push((0, readback.clone()));
            readbacks.push(readback);
        }
        backend.prepare_memory(inventory, true).unwrap();
        let requests = [
            (0, None, GpuInvocation::Concat { inputs: &refs, axis: ConcatAxis::Columns }),
            (0, None, GpuInvocation::Concat { inputs: &refs, axis: ConcatAxis::Diagonal }),
        ];
        let cpu_refs = originals[1..].iter().collect::<Vec<_>>();
        let expected =
            [originals[0].concat_columns(&cpu_refs), originals[0].concat_diag(&cpu_refs)];
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            let actual = [
                backend.concat(&refs, ConcatAxis::Columns).unwrap(),
                backend.concat(&refs, ConcatAxis::Diagonal).unwrap(),
            ];
            for (output, expected) in actual.iter().zip(&expected) {
                for shard in output.shards() {
                    let start = shard.global_column_start;
                    let end = start + shard.value.col_size();
                    let mut offset = 0;
                    let owner = inputs
                        .iter()
                        .find_map(|input| {
                            let found = (start >= offset && end <= offset + input.columns)
                                .then(|| {
                                    input.shards().iter().find(|source| {
                                        source.global_column_start <= start - offset &&
                                            end - offset <=
                                                source.global_column_start +
                                                    source.value.col_size()
                                    })
                                })
                                .flatten();
                            offset += input.columns;
                            found
                        })
                        .expect("an original shard owns the entire concatenated interval");
                    assert_eq!(
                        shard.value.params().context_identity(),
                        owner.value.params().context_identity()
                    );
                    let index = params
                        .iter()
                        .position(|p| {
                            p.context_identity() == shard.value.params().context_identity()
                        })
                        .unwrap();
                    let readback = &readbacks[index];
                    let transfer = params[index]
                        .rns_transfer_workspace(
                            params[index].crt_depth() - 1,
                            output.rows,
                            end - start,
                        )
                        .unwrap();
                    let events = shard.value.rns_store_completion_events().unwrap();
                    let claims = (1..2 + events)
                        .map(|i| {
                            let slot = readback.slot_identity(i).unwrap();
                            slot.workspace_request(
                                if i == 1 {
                                    transfer.bytes
                                } else {
                                    slot.requested_backing_bytes()
                                },
                                slot.alignment(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                    assert_eq!(shard.value.to_cpu_matrix(), expected.slice_columns(start, end));
                    drop(dispatch.finish().unwrap());
                }
            }
            drop(actual);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_row_blocks_share_inputs_and_preserve_column_intersections() {
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = super::super::super::gpu_backend_on([params.clone()], [device]);
        let cpu_blocks = (0..18)
            .into_par_iter()
            .map(|i| {
                DCRTPolyUniformSampler::new().sample_uniform(
                    &cpu,
                    1 + i % 2,
                    columns,
                    DistType::FinRingDist,
                )
            })
            .collect::<Vec<_>>();
        let stacked = cpu_blocks[0].concat_rows(&cpu_blocks[1..].iter().collect::<Vec<_>>());
        let rows = stacked.row_size();
        let rhs_cpu = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let blocks = cpu_blocks
            .par_iter()
            .enumerate()
            .map(|(i, block)| {
                let width = 2 + i % 2;
                let shards = (0..columns)
                    .step_by(width)
                    .map(|start| {
                        let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                            &params,
                            &block.slice_columns(start, (start + width).min(columns)),
                        );
                        if i % 2 == 0 {
                            value.intt_all_in_place();
                        }
                        GpuColumnShard { device_id: device, global_column_start: start, value }
                    })
                    .collect();
                GpuFleetMatrix::new(block.row_size(), columns, shards)
            })
            .collect::<Vec<_>>();
        let right =
            GpuFleetMatrix::from_matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &rhs_cpu));
        let refs = blocks.iter().collect::<Vec<_>>();
        let normalizations = blocks
            .iter()
            .flat_map(|block| block.shards.iter())
            .filter(|shard| !shard.value.is_ntt())
            .map(|shard| shard.value.size())
            .collect::<Vec<_>>();
        let input_storage = Arc::new(
            GpuPreparedStorage::new(
                None,
                normalizations
                    .into_par_iter()
                    .map(|(r, c)| GpuDCRTPolyMatrix::zero(&params, r, c))
                    .collect(),
                None,
                None,
            )
            .unwrap(),
        );
        let outputs = Arc::new(
            GpuPreparedStorage::new(
                None,
                (0..2 * columns)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, rows, 2))
                    .collect(),
                None,
                None,
            )
            .unwrap(),
        );
        // All CRT limbs share one readback batch and completion event.
        let events = 1;
        let mut layouts =
            vec![params.rns_transfer_workspace(params.crt_depth() - 1, rows, 2).unwrap()];
        layouts.extend(std::iter::repeat_n(
            GpuPreparedWorkspaceLayout {
                kind: GpuPreparedSlotKind::CompletionEvent,
                bytes: 0,
                alignment: 1,
            },
            events,
        ));
        let readback = Arc::new(
            GpuPreparedStorage::new(
                None,
                vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
                None,
                Some(&layouts),
            )
            .unwrap(),
        );
        backend
            .prepare_memory(
                vec![(0, input_storage.clone()), (0, outputs.clone()), (0, readback.clone())],
                true,
            )
            .unwrap();
        let requests = [
            (0, None, GpuInvocation::Concat { inputs: &refs, axis: ConcatAxis::Rows }),
            (0, None, GpuInvocation::AddRowBlocks { blocks: &refs, right: &right }),
        ];
        let expected = [stacked.clone(), stacked.add_out_of_place(&rhs_cpu)];
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            let actual = [
                backend.concat(&refs, ConcatAxis::Rows).unwrap(),
                backend.add_row_blocks(&refs, &right).unwrap(),
            ];
            for (output, expected) in actual.iter().zip(&expected) {
                for shard in output.shards() {
                    let start = shard.global_column_start;
                    let end = start + shard.value.col_size();
                    assert!(blocks.iter().all(|input| {
                        input.shards().iter().any(|source| {
                            source.global_column_start <= start &&
                                end <= source.global_column_start + source.value.col_size()
                        })
                    }));
                    let transfer = params
                        .rns_transfer_workspace(params.crt_depth() - 1, rows, end - start)
                        .unwrap();
                    let claims = (1..readback.slot_count())
                        .map(|i| {
                            let slot = readback.slot_identity(i).unwrap();
                            slot.workspace_request(
                                if i == 1 {
                                    transfer.bytes
                                } else {
                                    slot.requested_backing_bytes()
                                },
                                slot.alignment(),
                            )
                        })
                        .collect::<Vec<_>>();
                    let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                    assert_eq!(shard.value.to_cpu_matrix(), expected.slice_columns(start, end));
                    drop(dispatch.finish().unwrap());
                }
            }
            drop(actual);
        }
    }
}
