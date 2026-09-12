//! Bind ledger-issued column plans to ordinary and compact backend calls. Admitted
//! output owners are initialized once, then filled through allocation-free
//! range views on the existing enqueue workers. Setup/sealing and compilation
//! of other invocation classes remain separate requirements.

use super::{
    gpu_prepare::{
        PreparedFleetOutput, PreparedMatrixInputs, PreparedMatrixSource, PreparedMatrixValue,
    },
    *,
};
use crate::{
    gpu_invocation::GpuInvocation,
    gpu_memory::{GpuAdmissionError, GpuColumnMemoryPlan},
};
use mxx_ir_core::node::MatrixBinaryOp;
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
pub(super) enum PreparedMatrixOperation {
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
    },
    Hash {
        ty: ConcreteMatrixType,
        tag_bytes: usize,
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
        /// Compact hash sampling: the output type and tag length of a seeded
        /// COEFF gadget source generated per admitted range instead of a
        /// prepared input. The seed never enters admission identity.
        hash: Option<(ConcreteMatrixType, usize)>,
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

/// One admitted, not yet consumed, invocation of the fleet backend for
/// diagnostics and estimation. The seed and operand payloads are not included.
#[derive(Clone, Debug, serde::Serialize)]
pub struct GpuAdmittedInvocationSummary {
    pub operation: &'static str,
    pub rows: usize,
    pub columns: usize,
    pub compact_output: bool,
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
    /// Isolated pilot: synthetic trapdoor and public matrix, zero target.
    PreimagePilot,
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
/// inventory sealed by tracing the production kernels at each admitted width.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct PreimageClaimPlan {
    /// Claims of the destination's hard-cutoff plan (after its payload owner).
    pub destination: Vec<GpuTracedClaim>,
    /// Per-width claims of materializing the target tile.
    pub tile: std::collections::BTreeMap<usize, Vec<GpuTracedClaim>>,
    /// Per-width claims of one candidate attempt.
    pub attempt: std::collections::BTreeMap<usize, Vec<GpuTracedClaim>>,
    /// Claims of sampling a synthetic trapdoor and public matrix for pilots.
    pub trapdoor: Vec<GpuTracedClaim>,
    pub attempts: usize,
    /// Native RNS staging width of one full-level polynomial, so pilots stage
    /// a zero target through the same loader as production.
    pub bytes_per_poly: usize,
}

/// Identity of a preimage invocation class whose claim plan was traced.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct PreimagePlanKey {
    pub modulus: String,
    pub ring_dimension: usize,
    pub rows: usize,
    pub columns: usize,
    pub public_rows: usize,
    pub bound: String,
    pub sigma_bits: u64,
    pub gadget_base: String,
    pub digit_count: usize,
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
#[derive(Clone, Default)]
pub(super) struct PreparedClaimBroker {
    storages: Vec<Arc<GpuPreparedStorage>>,
}

impl PreparedClaimBroker {
    pub(super) fn new(storages: Vec<Arc<GpuPreparedStorage>>) -> Self {
        Self { storages }
    }

    /// Hold `claims` around `run`. Use for steps whose claim lists were derived
    /// from the live owners (exports, readbacks).
    pub(super) fn hold<T>(
        &self,
        claims: &[GpuTracedClaim],
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.hold_inner(claims, false, run)
    }

    /// Hold recorded (traced) claims around `run`: the step executes with
    /// deterministic consumer-event claims, exactly as it was traced.
    pub(super) fn hold_traced<T>(
        &self,
        claims: &[GpuTracedClaim],
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        self.hold_inner(claims, true, run)
    }

    fn hold_inner<T>(
        &self,
        claims: &[GpuTracedClaim],
        traced: bool,
        run: impl FnOnce() -> Result<T, String>,
    ) -> Result<T, String> {
        let mut used = std::collections::HashSet::new();
        let mut reservations = Vec::with_capacity(claims.len());
        for claim in claims {
            // Smallest fitting backing first, as output selection does, so a
            // small claim never consumes the only slot a larger one needs.
            let mut candidates = Vec::new();
            for storage in &self.storages {
                for index in 0..storage.slot_count() {
                    let slot = storage.slot_identity(index).expect("declared slot");
                    if used.contains(&slot.slot_id()) || slot.kind() != claim.kind() {
                        continue;
                    }
                    let request = if claim.kind() == GpuPreparedSlotKind::Matrix {
                        if slot.level() != claim.level() {
                            continue;
                        }
                        slot.matrix_request(
                            claim.rows(),
                            claim.columns(),
                            claim.is_evaluation().unwrap_or(true),
                        )
                    } else {
                        slot.workspace_request(claim.bytes(), claim.alignment().max(1))
                    };
                    if storage.fits(&[request])? {
                        candidates.push((slot.requested_backing_bytes(), storage, slot, request));
                    }
                }
            }
            candidates.sort_by_key(|(bytes, _, _, _)| *bytes);
            let mut found = None;
            let mut failures = Vec::new();
            for (_, storage, slot, request) in candidates {
                match storage.reserve(&[request]) {
                    Ok(reservation) => {
                        used.insert(slot.slot_id());
                        found = Some(reservation);
                        break;
                    }
                    Err(error) => failures.push(format!(
                        "slot {} ({}x{}): {error}",
                        slot.slot_id(),
                        slot.rows(),
                        slot.columns()
                    )),
                }
            }
            reservations.push(found.ok_or_else(|| {
                let same_kind = self
                    .storages
                    .iter()
                    .flat_map(|storage| {
                        (0..storage.slot_count()).filter_map(|index| storage.slot_identity(index))
                    })
                    .filter(|slot| slot.kind() == claim.kind())
                    .map(|slot| (slot.rows(), slot.columns(), slot.level()))
                    .collect::<Vec<_>>();
                format!(
                    "prepared inventory cannot claim {claim:?} for an admitted step; candidates: {failures:?}; same-kind slots: {same_kind:?}"
                )
            })?);
        }
        for reservation in &mut reservations {
            reservation.require_all_resources()?;
        }
        let mut reservations = reservations.into_iter();
        let dispatch = match reservations.next() {
            Some(first) => Some(first.enter_or_extend(reservations.collect())?),
            None => None,
        };
        let guard = traced.then(mxx_primitives::matrix::gpu_dcrt_poly::GpuTracedStepGuard::enter);
        let result = run();
        drop(guard);
        match (result, dispatch) {
            // A failed step retracts its permit (dropping cancels); its own
            // error is reported rather than the unconsumed-claims diagnostic.
            (Err(error), dispatch) => {
                drop(dispatch);
                Err(error)
            }
            (Ok(value), Some(dispatch)) => {
                drop(dispatch.finish()?);
                Ok(value)
            }
            (Ok(value), None) => Ok(value),
        }
    }
}

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

impl PreparedMatrixOperation {
    pub(super) fn kind_name(&self) -> &'static str {
        match self {
            Self::Constant { .. } => "Constant",
            Self::Sample { .. } => "Sample",
            Self::Polynomial { .. } => "Polynomial",
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
            Self::Decompose { hash: Some((ty, _)), .. } => Some(ty),
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

    /// Staging imports are admitted for full-level owners; the header level is
    /// implied by the polynomial byte width of eight bytes per limb coefficient.
    fn staging_level(&self) -> Result<usize, PolyBackendError> {
        let Self::ImportStaging { ty, bytes_per_poly, .. } = self else {
            return Err(PolyBackendError::InvalidConstantShape);
        };
        let limb_bytes =
            ty.ring_dimension.checked_mul(8).ok_or(PolyBackendError::InvalidInteger)?;
        if limb_bytes == 0 || *bytes_per_poly == 0 || !bytes_per_poly.is_multiple_of(limb_bytes) {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        Ok(bytes_per_poly / limb_bytes - 1)
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

    /// Payload for an isolated pilot: independent randomness for sampling, and
    /// an all-zero canonical payload of the exact production size for imports.
    pub(super) fn pilot_payload(&self) -> Result<ExecutionPayload, PolyBackendError> {
        if matches!(self, Self::Preimage { .. }) {
            return Ok(ExecutionPayload::PreimagePilot);
        }
        if let Self::ImportStaging { ty, evaluation, bytes_per_poly, payload_len } = self {
            let bytes = ty
                .rows
                .checked_mul(ty.columns)
                .and_then(|count| count.checked_mul(*bytes_per_poly))
                .ok_or(PolyBackendError::InvalidInteger)?;
            let level = self.staging_level()?;
            let encoded = bincode::encode_to_vec(
                (1u8, ty.rows, ty.columns, level, *evaluation, *bytes_per_poly, vec![0u8; bytes]),
                bincode::config::standard(),
            )
            .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?;
            if encoded.len() != *payload_len {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            return Ok(ExecutionPayload::Bytes(Arc::new(encoded)));
        }
        if self.is_import() {
            return Ok(ExecutionPayload::Bytes(Arc::new(vec![0u8; self.import_payload_len()?])));
        }
        Ok(self.execution_seed().map_or(ExecutionPayload::None, ExecutionPayload::Seed))
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
            (ExecutionPayload::None, false, false) => {
                Ok(self.execution_seed().map_or(ExecutionPayload::None, ExecutionPayload::Seed))
            }
            _ => Err(PolyBackendError::GpuSubmission(
                "incorrect production hash seed or import payload".into(),
            )),
        }
    }

    // One seed per production invocation, shared by every range and device.
    // Pilots use independent randomness; seeds never enter calibration keys.
    pub(super) fn execution_seed(&self) -> Option<GpuRngSeed> {
        match self {
            Self::Sample { .. } => Some(GpuRngSeed::from_bytes(rand::random())),
            Self::Hash { tag_bytes, .. } | Self::Decompose { hash: Some((_, tag_bytes)), .. } => {
                Some(mxx_primitives::sampler::gpu::hash_seed_for_matrix::<keccak_asm::Keccak256>(
                    rand::random(),
                    &vec![0; *tag_bytes],
                ))
            }
            _ => None,
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
                Self::ImportCompact { .. } |
                Self::Preimage { .. }
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
    pub(super) fn primary<'a>(
        &self,
        left: Option<&'a GpuFleetMatrix>,
        right: &'a [GpuFleetMatrix],
        start: usize,
    ) -> Option<&'a GpuFleetMatrix> {
        if let Self::ConcatColumns { offsets, .. } = self {
            let index = offsets.partition_point(|&(_, _, end)| end <= start);
            if index == 0 { left } else { Some(&right[index - 1]) }
        } else {
            left
        }
    }

    pub(super) fn dependent_inputs<'a>(&self, right: &'a [GpuFleetMatrix]) -> &'a [GpuFleetMatrix] {
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

    pub(super) fn output_rows(
        &self,
        scalable: Option<&GpuFleetMatrix>,
        other: &[GpuFleetMatrix],
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
            if scalable.is_some() || lhs.columns != *inner || *inner == 0 {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            return Ok(lhs.rows);
        }
        let scalable = scalable.ok_or(PolyBackendError::InvalidConstantShape)?;
        if let Self::Decompose { input_rows, layout, .. } = self {
            if input_rows
                .iter()
                .copied()
                .ne(std::iter::once(scalable.rows).chain(other.iter().map(|v| v.rows))) ||
                other.iter().any(|v| v.columns != scalable.columns)
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
            if other.iter().any(|block| block.columns != scalable.columns) ||
                (*self == Self::AddRowBlocks && other.is_empty())
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let rows = other
                .iter()
                .try_fold(0usize, |sum, block| sum.checked_add(block.rows))
                .ok_or(PolyBackendError::InvalidConstantShape)?;
            return if *self == Self::ConcatRows {
                rows.checked_add(scalable.rows).ok_or(PolyBackendError::InvalidConstantShape)
            } else if rows == scalable.rows {
                Ok(rows)
            } else {
                Err(PolyBackendError::InvalidConstantShape)
            };
        }
        if let Self::Tensor { right_rows, right_columns, groups } = self {
            let fixed = other.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            if fixed.size() != (*right_rows, *right_columns) {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let rows = scalable
                .rows
                .checked_mul(*right_rows)
                .ok_or(PolyBackendError::InvalidConstantShape)?;
            scalable
                .columns
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
                rows.end > scalable.rows ||
                columns.start > columns.end ||
                columns.end > scalable.columns
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            return Ok(rows.end - rows.start);
        }
        if let Self::SumRows(rows) = self {
            if rows
                .iter()
                .any(|group| group.is_empty() || group.iter().any(|&row| row >= scalable.rows))
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            return Ok(rows.len());
        }
        if *self == Self::Transpose {
            return Ok(scalable.columns);
        }
        if let Self::Accumulate { rows, .. } = self {
            return Ok(*rows);
        }
        if matches!(self, Self::Multiply { .. }) {
            let fixed = other.first().ok_or(PolyBackendError::InvalidConstantShape)?;
            if fixed.size() == (1, 1) {
                return Ok(scalable.rows);
            }
            if fixed.columns != scalable.rows || fixed.columns == 0 {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            Ok(fixed.rows)
        } else if other.iter().any(|other| other.size() != scalable.size()) {
            Err(PolyBackendError::InvalidConstantShape)
        } else {
            Ok(scalable.rows)
        }
    }

    /// Fixed native scratch for singleton transfers and large RNS launch plans.
    /// RNS plans are written on-device and can reuse event-ordered GPU storage.
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
    pub(super) fn pilot_step_claims(&self) -> Vec<GpuTracedClaim> {
        let Self::Preimage { plan, .. } = self else { return Vec::new() };
        let mut claims = plan.trapdoor.clone();
        claims.extend(plan.tile.get(&1).into_iter().flatten().cloned());
        claims.extend(plan.destination.iter().cloned());
        claims.extend(plan.attempt.get(&1).into_iter().flatten().cloned());
        claims
    }

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

    pub(super) fn output_columns(&self, scalable: Option<&GpuFleetMatrix>) -> usize {
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
            Self::Transpose => scalable.rows,
            Self::Tensor { right_columns, .. } => scalable.columns * right_columns,
            Self::Slice { columns, .. } => columns.end - columns.start,
            _ => scalable.columns,
        }
    }

    pub(super) fn source_columns(
        &self,
        scalable: &GpuFleetMatrix,
        start: usize,
        end: usize,
    ) -> std::ops::Range<usize> {
        match self {
            Self::ConcatColumns { offsets, .. } => {
                let index = offsets.partition_point(|&(_, _, end)| end <= start);
                start - offsets[index].1..end - offsets[index].1
            }
            Self::Transpose | Self::Tensor { .. } => 0..scalable.columns,
            Self::Slice { columns, .. } => start + columns.start..end + columns.start,
            _ => start..end,
        }
    }

    pub(super) fn validate_output_workspace(
        &self,
        source: &GpuDCRTPolyMatrix,
        other: &[GpuFleetMatrix],
        shape: (usize, usize),
    ) -> Result<(), PolyBackendError> {
        if matches!(self, Self::Multiply { .. }) {
            use mxx_primitives::poly::dcrt::gpu::GpuMatrixBatchOperation;
            let operation = if other.first().is_some_and(|fixed| fixed.size() == (1, 1)) {
                GpuMatrixBatchOperation::Scalar
            } else {
                GpuMatrixBatchOperation::Multiply
            };
            let workspace = source
                .params()
                .matrix_batch_workspace_bytes(source.level(), shape, 1, 1, operation, true)
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

    pub(super) fn other_columns(
        &self,
        index: usize,
        other: &GpuFleetMatrix,
        start: usize,
        end: usize,
    ) -> std::ops::Range<usize> {
        if self.fixed_input(index) { 0..other.columns } else { start..end }
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
        if let Self::Preimage { sigma_bits, public_rows, plan, .. } = self {
            let (PreparedMatrixValue::Compact(mut output), rows, columns) = destination else {
                return Err("compact destination required".into());
            };
            let width = end - start;
            let params = output.params().clone();
            let sampler = GpuDCRTPolyTrapdoorSampler::new(&params, f64::from_bits(*sigma_bits));
            let attempt_claims = plan
                .attempt
                .get(&width)
                .ok_or_else(|| format!("preimage plan has no claims for width {width}"))?;
            let tile_claims = plan
                .tile
                .get(&width)
                .ok_or_else(|| format!("preimage plan has no target claims for width {width}"))?;
            if rows != (0..output.rows_count()) {
                return Err("preimage fills complete rows".into());
            }
            // Synthetic operands for the isolated pilot, production operands
            // otherwise. The pilot never touches production randomness.
            let synthetic;
            let (trapdoor, public, tile, seed_bytes, global_column) = match payload {
                ExecutionPayload::Preimage(payload) => {
                    let public = payload
                        .public
                        .shards()
                        .iter()
                        .find(|shard| shard.value.params() == &params)
                        .filter(|shard| {
                            shard.global_column_start == 0 &&
                                shard.value.col_size() == payload.public.size().1
                        })
                        .ok_or("preimage public matrix is not resident in this context")?;
                    let tile = broker.hold_traced(tile_claims, || {
                        materialize_preimage_tile(
                            &params,
                            &payload.target,
                            *public_rows,
                            start,
                            end,
                        )
                    })?;
                    (
                        &payload.trapdoors[0],
                        &public.value,
                        tile,
                        payload.seed,
                        payload.target_global_column_start + start,
                    )
                }
                ExecutionPayload::PreimagePilot => {
                    let (trapdoor, public) = broker.hold_traced(&plan.trapdoor, || {
                        let (trapdoor, public) = sampler.trapdoor(&params, *public_rows);
                        sampler.prepare_preimage_cache(&params, &trapdoor, *public_rows);
                        Ok((trapdoor, public))
                    })?;
                    let staging = mxx_primitives::matrix::gpu_dcrt_poly::GpuCpuStagingLayout {
                        rows: *public_rows,
                        columns: width,
                        level: params.crt_depth() - 1,
                        is_ntt: true,
                        bytes_per_poly: plan.bytes_per_poly,
                    }
                    .zero_bytes(&params)?;
                    let target = PolyMatrixColumnData::<GpuFleetMatrix>::staged(
                        &params,
                        Arc::new(staging),
                        0,
                        width,
                    );
                    let tile = broker.hold_traced(tile_claims, || {
                        materialize_preimage_tile(&params, &target, *public_rows, 0, width)
                    })?;
                    synthetic = (trapdoor, public);
                    (&synthetic.0, &synthetic.1, tile, rand::random(), start)
                }
                _ => return Err("preimage invocation has no operands".into()),
            };
            if columns.start == 0 {
                // The retained destination's immutable hard-cutoff plan, once.
                broker.hold_traced(&plan.destination, || {
                    output.prepare_preimage_hard_cutoff();
                    Ok(())
                })?;
            }
            let mut accepted = false;
            for attempt in 0..plan.attempts {
                accepted = broker.hold_traced(attempt_claims, || {
                    sampler
                        .preimage_attempt(
                            &params,
                            trapdoor,
                            public,
                            &tile,
                            &mut output,
                            columns.start,
                            global_column,
                            attempt,
                            seed_bytes,
                        )
                        .map_err(|error| error.to_string())
                })?;
                if accepted {
                    break;
                }
            }
            if !accepted {
                return Err(format!(
                    "preimage columns {start}..{end} exhausted {} bounded attempts",
                    plan.attempts
                ));
            }
            return Ok(PreparedMatrixValue::Compact(output));
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
            // Bounded host staging: the pinned upload buffer and its retirement
            // event are recycled through the pinned reclaimer. Retire them at
            // this import boundary so the next range can rearm the same slots.
            scratch.wait_until_ready();
            drop(scratch);
            output.params().fence_released_memory();
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
            // Bounded host staging: retire the pinned tile and its completion
            // event before the next range rearms the same slots.
            staged.wait_until_ready();
            drop(staged);
            copied.params().fence_released_memory();
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
            if let Some((ty, _)) = hash {
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
        if let Self::Polynomial { coefficients, .. } = self {
            let (mut output, rows, columns) = destination;
            if start != 0 ||
                end != 1 ||
                rows != (0..1) ||
                columns != (0..1) ||
                output.size() != (1, 1)
            {
                return Err("polynomial constant must execute as a complete singleton".into());
            }
            output.fill_polynomial(coefficients)?;
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
    operation: PreparedMatrixOperation,
    left: Option<GpuFleetMatrix>,
    right: Vec<GpuFleetMatrix>,
    compact: Option<GpuFleetSmallMatrix>,
    plan: GpuColumnMemoryPlan,
    intervals: Vec<CompiledInterval>,
    prepared: Arc<PreparedMatrixInputs>,
}

impl CompiledMatrixInvocation {
    /// Typed operands of one invocation: the operation, its scalable ordinary
    /// input, fixed ordinary inputs, and a compact input for compact-input
    /// operations. Shapes, contexts and ownership are validated here.
    pub(super) fn arguments<'a>(
        request: &'a MatrixInvocation<'_>,
        backend: &GpuDcrtBackend,
    ) -> Result<
        (
            PreparedMatrixOperation,
            Option<&'a GpuFleetMatrix>,
            Vec<GpuFleetMatrix>,
            Option<&'a GpuFleetSmallMatrix>,
        ),
        PolyBackendError,
    > {
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
            return Ok((
                PreparedMatrixOperation::CenteredExtendCompact {
                    destination: (**destination).clone(),
                    bound,
                },
                None,
                Vec::new(),
                Some(*value),
            ));
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
            return Ok((
                PreparedMatrixOperation::MultiplyCompact { columns: right.columns, inner },
                None,
                vec![(*left).clone()],
                Some(*right),
            ));
        }
        if let GpuInvocation::MultiplySmallRhsRowBlocks { .. } = request {
            return Err(PolyBackendError::GpuSubmission(
                "compact row-block products are admitted as one invocation per block".into(),
            ));
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
            return Ok((
                PreparedMatrixOperation::ImportMatrix {
                    ty: (**ty).clone(),
                    evaluation,
                    max_coefficient_bits,
                },
                None,
                Vec::new(),
                None,
            ));
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
            let key = PreimagePlanKey {
                modulus: schema.matrix.modulus.to_string(),
                ring_dimension: schema.matrix.ring_dimension,
                rows: schema.matrix.rows,
                columns: schema.matrix.columns,
                public_rows: public.rows,
                bound: bound.to_string(),
                sigma_bits: sigma.to_bits(),
                gadget_base: gadget_base.to_string(),
                digit_count: *digit_count,
            };
            let plan = backend.preimage_plans.get(&key).ok_or_else(|| {
                PolyBackendError::GpuSubmission(
                    "preimage sampling has no derived claim plan for this shape".into(),
                )
            })?;
            return Ok((
                PreparedMatrixOperation::Preimage {
                    ty: schema.matrix.clone(),
                    bound,
                    sigma_bits: sigma.to_bits(),
                    gadget_base: (*gadget_base).clone(),
                    digit_count: *digit_count,
                    public_rows: public.rows,
                    plan: plan.clone(),
                },
                None,
                Vec::new(),
                None,
            ));
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
            return Ok((
                PreparedMatrixOperation::ImportStaging {
                    ty: (**ty).clone(),
                    evaluation: layout.is_ntt,
                    bytes_per_poly: layout.bytes_per_poly,
                    payload_len: bytes.len(),
                },
                None,
                Vec::new(),
                None,
            ));
        }
        if let GpuInvocation::ImportSmallMatrix { schema, bytes, semantic_kind } = request {
            let (bound, _) =
                crate::backend::poly::decode_small_matrix_artifact(schema, bytes, *semantic_kind)?;
            backend.devices[0].1.parameters(&schema.matrix)?;
            return Ok((
                PreparedMatrixOperation::ImportCompact { ty: schema.matrix.clone(), bound },
                None,
                Vec::new(),
                None,
            ));
        }
        let (operation, left, right) = Self::ordinary_arguments(request, backend)?;
        Ok((operation, left, right, None))
    }

    fn ordinary_arguments<'a>(
        request: &'a MatrixInvocation<'_>,
        backend: &GpuDcrtBackend,
    ) -> Result<
        (PreparedMatrixOperation, Option<&'a GpuFleetMatrix>, Vec<GpuFleetMatrix>),
        PolyBackendError,
    > {
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
            let operation = PreparedMatrixOperation::Decompose {
                small,
                digit_count,
                input_rows: blocks.iter().map(|v| v.rows).collect(),
                layout,
                hash: None,
            };
            let right = blocks[1..].iter().map(|v| (*v).clone()).collect::<Vec<_>>();
            operation.output_rows(Some(first), &right)?;
            return Ok((operation, Some(first), right));
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
            let plaintext_moduli = plaintext_moduli
                .par_iter()
                .map(|p| p.to_u64().filter(|p| *p != 0))
                .collect::<Option<Vec<_>>>()
                .ok_or(PolyBackendError::InvalidInteger)?;
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
            return Ok((
                PreparedMatrixOperation::CrtRecompose {
                    destination: (*destination).clone(),
                    plaintext_moduli,
                    reconstruction_coefficients: reconstruction_coefficients.to_vec(),
                },
                Some(first),
                levels[1..].to_vec(),
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
            if source_moduli.is_empty() || source_moduli.len() > 64 {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let groups = match conversion {
                GpuMatrixRnsConversion::Up { digit_size, .. } if digit_size != 0 => {
                    source_moduli.len().div_ceil(digit_size)
                }
                GpuMatrixRnsConversion::Down { plaintext_modulus } if plaintext_modulus >= 2 => 1,
                _ => return Err(PolyBackendError::InvalidConstantShape),
            };
            if value.rows.checked_mul(groups) != Some(destination.rows) ||
                value.columns != destination.columns
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            for (_, device) in &backend.devices {
                let target = device.parameters(destination)?;
                // Validate declared bases even when the matrix has no shards.
                let valid = match conversion {
                    GpuMatrixRnsConversion::Up { .. } => {
                        source_moduli.iter().all(|q| target.moduli().contains(q))
                    }
                    GpuMatrixRnsConversion::Down { plaintext_modulus } => {
                        target.crt_depth() < source_moduli.len() &&
                            target.moduli().iter().all(|q| source_moduli.contains(q)) &&
                            source_moduli.iter().filter(|q| !target.moduli().contains(q)).all(
                                |q| {
                                    *q > 1 &&
                                        mxx_primitives::utils::mod_inverse(
                                            plaintext_modulus % q,
                                            *q,
                                        )
                                        .is_some()
                                },
                            )
                    }
                };
                if !valid {
                    return Err(PolyBackendError::BasisConversion(
                        "invalid declared RNS basis relation".into(),
                    ));
                }
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
            return Ok((
                PreparedMatrixOperation::RnsConversion {
                    destination: destination.clone(),
                    source_moduli: source_moduli.to_vec(),
                    conversion,
                },
                Some(value),
                Vec::new(),
            ));
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
            return Ok((
                PreparedMatrixOperation::ModulusConversion {
                    destination: destination.clone(),
                    conversion,
                },
                Some(value),
                Vec::new(),
            ));
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
            return Ok((
                PreparedMatrixOperation::CenteredRebase { destination: (**destination).clone() },
                Some(*value),
                Vec::new(),
            ));
        }
        if let GpuInvocation::SampleHash { ty, variant, tag_bytes, gadget_base, digit_count } =
            request
        {
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
                return Ok((
                    PreparedMatrixOperation::Decompose {
                        small,
                        digit_count: Some(*count),
                        input_rows: vec![source_rows],
                        layout,
                        hash: Some(((**ty).clone(), *tag_bytes)),
                    },
                    None,
                    Vec::new(),
                ));
            }
            if *variant != HashVariant::Plain || gadget_base.is_some() || digit_count.is_some() {
                return Err(PolyBackendError::InvalidInteger);
            }
            backend.devices[0].1.parameters(ty)?;
            return Ok((
                PreparedMatrixOperation::Hash { ty: (**ty).clone(), tag_bytes: *tag_bytes },
                None,
                Vec::new(),
            ));
        }
        let sample = match request {
            GpuInvocation::SampleUniform { ty, range } => {
                let parameters = backend.devices[0].1.parameters(ty)?;
                let maximum = BigInt::from(parameters.modulus().as_ref().clone()) - 1;
                let distribution =
                    if range.minimum == BigInt::from(-1) && range.maximum == BigInt::from(1) {
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
                Some((*ty, distribution, 0.0_f64, BigInt::from(u64::MAX)))
            }
            GpuInvocation::SampleGaussian { ty, sigma, max_coefficient_bound } => {
                backend.devices[0].1.parameters(ty)?;
                if max_coefficient_bound.to_biguint().is_none() {
                    return Err(PolyBackendError::InvalidInteger);
                }
                if !sigma.is_finite() || *sigma < 0.0 {
                    return Err(PolyBackendError::GpuSubmission(
                        "Gaussian sigma must be finite and nonnegative".into(),
                    ));
                }
                Some((*ty, GpuMatrixSampleDist::Gauss, *sigma, (*max_coefficient_bound).clone()))
            }
            _ => None,
        };
        if let Some((ty, distribution, sigma, max_coefficient_bound)) = sample {
            return Ok((
                PreparedMatrixOperation::Sample {
                    ty: ty.clone(),
                    distribution,
                    sigma_bits: sigma.to_bits(),
                    max_coefficient_bound,
                },
                None,
                Vec::new(),
            ));
        }
        if let GpuInvocation::Constant { ty, value, env } = request {
            let index = |expression: &mxx_ir_core::expr::IntExpr| {
                expression
                    .evaluate(env)
                    .ok()
                    .and_then(|value| value.to_usize())
                    .ok_or(PolyBackendError::InvalidInteger)
            };
            let polynomial = match value {
                ConstantMatrix::PowerOfBase { base, exponent }
                    if ty.rows == 1 && ty.columns == 1 =>
                {
                    let base = base.evaluate(env).map_err(|_| PolyBackendError::InvalidInteger)?;
                    let exponent = exponent
                        .evaluate(env)
                        .ok()
                        .and_then(|value| value.to_u32())
                        .ok_or(PolyBackendError::InvalidInteger)?;
                    backend.devices[0].1.parameters(ty)?;
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
                                coefficient
                                    .evaluate(env)
                                    .map_err(|_| PolyBackendError::InvalidInteger)
                            })
                            .collect::<Result<Vec<_>, _>>()?,
                    )
                }
                _ => None,
            };
            if let Some(coefficients) = polynomial {
                backend.devices[0].1.parameters(ty)?;
                return Ok((
                    PreparedMatrixOperation::Polynomial { ty: (**ty).clone(), coefficients },
                    None,
                    Vec::new(),
                ));
            }
            let value = match value {
                ConstantMatrix::Zero => GpuMatrixRangeConstant::Zero { total_columns: ty.columns },
                ConstantMatrix::Identity if ty.rows == ty.columns => {
                    GpuMatrixRangeConstant::Identity
                }
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
                    let base = base.evaluate(env).map_err(|_| PolyBackendError::InvalidInteger)?;
                    let digits = ty.columns / ty.rows;
                    backend.validate_gadget_layout(ty, &base, digits, *small)?;
                    GpuMatrixRangeConstant::Gadget { small: *small, digit_count: Some(digits) }
                }
                _ => {
                    return Err(PolyBackendError::GpuSubmission(
                        "constant has no prepared range generator".into(),
                    ))
                }
            };
            return Ok((
                PreparedMatrixOperation::Constant { ty: (**ty).clone(), value },
                None,
                Vec::new(),
            ));
        }
        let (operation,left,right)=(|| -> Result<(PreparedMatrixOperation, &'a GpuFleetMatrix, Vec<GpuFleetMatrix>),PolyBackendError> {
        let unary = match request {
            GpuInvocation::Slice { value, rows, columns } => {
                let operation = PreparedMatrixOperation::Slice {
                    rows: rows.map(|r| r.start..r.end).unwrap_or(0..value.rows),
                    columns: columns.map(|r| r.start..r.end).unwrap_or(0..value.columns),
                };
                operation.output_rows(Some(value), &[])?;
                Some((operation, *value))
            }
            GpuInvocation::Transpose { value } => {
                Some((PreparedMatrixOperation::Transpose, *value))
            }
            GpuInvocation::SumRows { value, rows } => {
                let operation = PreparedMatrixOperation::SumRows(rows.to_vec());
                operation.output_rows(Some(value), &[])?;
                Some((operation, *value))
            }
            GpuInvocation::ScaleInteger { value, scalar } => {
                Some((PreparedMatrixOperation::Scale((*scalar).clone()), *value))
            }
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
                Some((PreparedMatrixOperation::Automorphism(*index), *value))
            }
            _ => None,
        };
        if let Some((operation, value)) = unary {
            return Ok((operation, value, Vec::new()));
        }
        match request {
            GpuInvocation::Accumulate { request } => {
                let first =
                    request.products.first().ok_or(PolyBackendError::InvalidConstantShape)?;
                let scalable = |left: &'a GpuFleetMatrix, right: &'a GpuFleetMatrix| {
                    let scales_left = gpu_matrix_multiply_scales_left(
                        left.rows,
                        left.columns,
                        right.rows,
                        right.columns,
                    );
                    if scales_left { (left, right, true) } else { (right, left, false) }
                };
                let (primary, _, _) = scalable(&first.1, &first.2);
                let mut others = Vec::with_capacity(2 * request.products.len());
                let mut products = Vec::with_capacity(request.products.len());
                let mut rows = None;
                for (index, (coefficient, left, right)) in request.products.iter().enumerate() {
                    let (input, fixed, scales_left) = scalable(left, right);
                    let output_rows = if fixed.size() == (1, 1) {
                        input.rows
                    } else if fixed.columns == input.rows {
                        fixed.rows
                    } else {
                        return Err(PolyBackendError::InvalidConstantShape);
                    };
                    if input.columns != primary.columns ||
                        rows.is_some_and(|rows| rows != output_rows)
                    {
                        return Err(PolyBackendError::InvalidConstantShape);
                    }
                    rows = Some(output_rows);
                    products.push((coefficient.clone(), scales_left));
                    if index != 0 {
                        others.push(input.clone());
                    }
                    others.push(fixed.clone());
                }
                let rows = rows.unwrap();
                if let Some(bias) = &request.bias {
                    if bias.size() != (rows, primary.columns) {
                        return Err(PolyBackendError::InvalidConstantShape);
                    }
                    others.push((**bias).clone());
                }
                Ok((
                    PreparedMatrixOperation::Accumulate {
                        products,
                        bias: request.bias.is_some(),
                        rows,
                    },
                    primary,
                    others,
                ))
            }
            GpuInvocation::Concat { inputs, axis: ConcatAxis::Rows } => {
                let (first, rest) =
                    inputs.split_first().ok_or(PolyBackendError::InvalidConstantShape)?;
                let others = rest.par_iter().map(|input| (**input).clone()).collect::<Vec<_>>();
                let operation = PreparedMatrixOperation::ConcatRows;
                operation.output_rows(Some(first), &others)?;
                Ok((operation, first, others))
            }
            GpuInvocation::Concat { inputs, axis } => {
                let (first, rest) =
                    inputs.split_first().ok_or(PolyBackendError::InvalidConstantShape)?;
                let diagonal = *axis == ConcatAxis::Diagonal;
                if !diagonal && inputs.iter().any(|value| value.rows != first.rows) {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
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
                let operation = PreparedMatrixOperation::ConcatColumns {
                    diagonal,
                    rows: if diagonal { rows } else { first.rows },
                    columns,
                    offsets,
                };
                Ok((operation, first, rest.par_iter().map(|input| (**input).clone()).collect()))
            }
            GpuInvocation::AddRowBlocks { blocks, right } => {
                let blocks = blocks.par_iter().map(|input| (**input).clone()).collect::<Vec<_>>();
                let operation = PreparedMatrixOperation::AddRowBlocks;
                operation.output_rows(Some(right), &blocks)?;
                Ok((operation, right, blocks))
            }
            GpuInvocation::Tensor { left, right } |
            GpuInvocation::TensorSumRows { left, right, .. } => {
                let groups = match request {
                    GpuInvocation::TensorSumRows { rows, .. } => Some(rows.to_vec()),
                    _ => None,
                };
                let operation = PreparedMatrixOperation::Tensor {
                    right_rows: right.rows,
                    right_columns: right.columns,
                    groups,
                };
                operation.output_rows(Some(left), std::slice::from_ref(*right))?;
                Ok((operation, left, vec![(*right).clone()]))
            }
            GpuInvocation::Negate { value } => {
                Ok((PreparedMatrixOperation::Negate, value, Vec::new()))
            }
            GpuInvocation::Binary { operation: MatrixBinaryOp::Add, left, right } => {
                Ok((PreparedMatrixOperation::Add, left, vec![(*right).clone()]))
            }
            GpuInvocation::Binary { operation: MatrixBinaryOp::Subtract, left, right } => {
                Ok((PreparedMatrixOperation::Subtract, left, vec![(*right).clone()]))
            }
            GpuInvocation::Binary { operation: MatrixBinaryOp::Multiply, left, right } => {
                let scales_left = gpu_matrix_multiply_scales_left(
                    left.rows,
                    left.columns,
                    right.rows,
                    right.columns,
                );
                let (scalable, fixed) = if scales_left { (*left, *right) } else { (*right, *left) };
                let operation = PreparedMatrixOperation::Multiply { scales_left };
                operation.output_rows(Some(scalable), std::slice::from_ref(fixed))?;
                Ok((operation, scalable, vec![fixed.clone()]))
            }
            _ => Err(PolyBackendError::GpuSubmission(
                "invocation has no compiled prepared matrix runner".into(),
            )),
        }
        })()?;
        Ok((operation, Some(left), right))
    }

    fn matches(
        &self,
        operation: PreparedMatrixOperation,
        left: Option<&GpuFleetMatrix>,
        right: &[GpuFleetMatrix],
        compact: Option<&GpuFleetSmallMatrix>,
    ) -> bool {
        self.operation == operation &&
            self.left.as_ref().map(|value| value.id) == left.map(|value| value.id) &&
            self.right.iter().map(|value| value.id).eq(right.iter().map(|value| value.id)) &&
            self.compact.as_ref().map(|value| value.id) == compact.map(|value| value.id)
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
    /// Bind a complete ordered batch of admitted plans to its actual operands.
    /// The normal preflight/add/subtract/negate calls subsequently consume these
    /// plans. All host validation finishes before publication or GPU submission.
    /// Missing, reordered, unsupported or substituted invocations are errors.
    ///
    /// Plans originate from GpuMemoryLedger::reserve_columns; their construction
    /// is private to the runtime. This API does not provision storage or certify
    /// a physical seal. It cannot turn diagnostic widths into admitted plans.
    pub fn set_admitted_matrix_invocations(
        &mut self,
        requests: Vec<(MatrixInvocation<'_>, GpuColumnMemoryPlan)>,
    ) -> Result<(), PolyBackendError> {
        self.prepared_invocations = {
            let (requests, plans): (Vec<_>, Vec<_>) = requests.into_iter().unzip();
            self.compile_matrix_invocations(
                requests.iter().zip(plans).collect(),
                Arc::new(HashMap::new()),
            )?
        };
        self.prepared_required = true;
        self.pending_pilot = None;
        self.pending_profile = None;
        Ok(())
    }

    pub(super) fn compile_matrix_invocations(
        &self,
        requests: Vec<(&MatrixInvocation<'_>, GpuColumnMemoryPlan)>,
        prepared: Arc<PreparedMatrixInputs>,
    ) -> Result<VecDeque<CompiledMatrixInvocation>, PolyBackendError> {
        if !self.prepared_invocations.is_empty() || !self.enqueue.is_healthy() {
            return Err(PolyBackendError::GpuSubmission(
                "previous admitted batch is still pending or workers are unavailable".into(),
            ));
        }
        if requests.is_empty() {
            return Err(PolyBackendError::GpuSubmission("admitted batch is empty".into()));
        }
        // Publication is sequential and atomic. Partial compilation owns its
        // plans and drops every lease if any later invocation is invalid.
        let mut compiled = VecDeque::with_capacity(requests.len());
        for (request, plan) in requests {
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(request, self)?;
            let rows = operation.output_rows(left, &right)?;
            if plan.schedule.local_job_counts().len() != self.devices.len() ||
                plan.schedule.intervals().last().map_or(0, |range| range.end) !=
                    operation.output_columns(left)
            {
                return Err(PolyBackendError::InvalidConstantShape);
            }
            let mut counts = vec![0usize; self.devices.len()];
            let mut intervals = Vec::with_capacity(plan.schedule.intervals().len());
            for &interval in plan.schedule.intervals() {
                let (device_id, backend) = &self.devices[interval.device];
                let output_claim = plan
                    .devices
                    .iter()
                    .find(|lease| lease.device == interval.device)
                    .and_then(|lease| {
                        lease
                            .prepared_outputs
                            .iter()
                            .flat_map(|reservation| {
                                reservation
                                    .slot_identities()
                                    .iter()
                                    .copied()
                                    .zip(reservation.requests().iter().copied())
                                    .map(move |(slot, request)| (reservation, slot, request))
                            })
                            .nth(counts[interval.device])
                    })
                    .ok_or(PolyBackendError::InvalidConstantShape)?;
                let locate = |value: &GpuFleetMatrix, columns: std::ops::Range<usize>| {
                    let fits = |source: PreparedMatrixSource| {
                        let input = &operation.source(&prepared, value, source).value;
                        if operation.input_evaluation(input.is_ntt()) != input.is_ntt() {
                            return false;
                        }
                        let Ok((parameters, level, evaluation)) = operation.output_layout(
                            backend,
                            input.params(),
                            input.level(),
                            input.is_ntt(),
                        ) else {
                            return false;
                        };
                        output_claim.0.matches_parameters(&parameters) &&
                            operation
                                .output_request(
                                    output_claim.1,
                                    &parameters,
                                    level,
                                    rows,
                                    interval.end - interval.start,
                                    evaluation,
                                )
                                .ok()
                                .flatten() ==
                                Some(output_claim.2)
                    };
                    if let Some(index) =
                        value.shards.iter().enumerate().find_map(|(index, shard)| {
                            (shard.device_id == *device_id &&
                                shard.global_column_start <= columns.start &&
                                columns.end - shard.global_column_start <=
                                    shard.value.col_size() &&
                                fits(PreparedMatrixSource::Shard(index)))
                            .then_some(index)
                        })
                    {
                        return Ok(PreparedMatrixSource::Shard(index));
                    }
                    if let Some(((_, source), _)) = prepared.iter().find(|((id, source), shard)|
                        *id == value.id && matches!(source, PreparedMatrixSource::Replica { device, .. } | PreparedMatrixSource::Fragment { device, .. } if *device == interval.device) &&
                        shard.device_id == *device_id && shard.global_column_start <= columns.start &&
                        columns.end - shard.global_column_start <= shard.value.col_size() && fits(*source)) {
                        return Ok(*source);
                    }
                    Err(PolyBackendError::GpuSubmission("admitted interval crosses an input owner or requires unplanned redistribution".into()))
                };
                let primary = operation.primary(left, &right, interval.start);
                let compact_source = compact
                    .map(|compact| {
                        compact
                            .shards
                            .iter()
                            .position(|shard| {
                                shard.device_id == *device_id &&
                                    shard.global_column_start <= interval.start &&
                                    interval.end - shard.global_column_start <=
                                        shard.value.columns_count()
                            })
                            .map(PreparedMatrixSource::Shard)
                            .ok_or(PolyBackendError::GpuSubmission(
                                "admitted interval crosses a compact input owner".into(),
                            ))
                    })
                    .transpose()?;
                let left_source = primary
                    .map(|primary| {
                        locate(
                            primary,
                            operation.source_columns(primary, interval.start, interval.end),
                        )
                    })
                    .transpose()?
                    .or(compact_source);
                let source = primary
                    .zip(left_source)
                    .map(|(primary, index)| &operation.source(&prepared, primary, index).value);
                let (parameters, level, evaluation) = if let Some(shard) =
                    CompiledMatrixInvocation::compact_shard(compact, compact_source)
                {
                    let registered = backend.parameters_for_small_matrix(&shard.value)?;
                    if registered != shard.value.params() {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    }
                    operation.output_layout(
                        backend,
                        shard.value.params(),
                        shard.value.params().crt_depth() - 1,
                        false,
                    )?
                } else if let Some(source) = source {
                    let registered = backend.parameters_for_matrix(source)?;
                    if registered != source.params() ||
                        registered.execution_owner_id() != source.params().execution_owner_id()
                    {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    }
                    operation.output_layout(
                        backend,
                        source.params(),
                        source.level(),
                        source.is_ntt(),
                    )?
                } else if let Some(ty) = operation.fresh_type() {
                    let parameters = backend.parameters(ty)?.clone();
                    let level = parameters.crt_depth() - 1;
                    (parameters, level, operation.input_evaluation(true))
                } else {
                    return Err(PolyBackendError::InvalidConstantShape);
                };
                if operation.requires_evaluation() && !evaluation {
                    return Err(PolyBackendError::GpuSubmission(
                        "coefficient input is missing its admitted evaluation preparation".into(),
                    ));
                }
                let right_source = operation
                    .dependent_inputs(&right)
                    .iter()
                    .enumerate()
                    .map(|(index, right)| {
                        locate(
                            right,
                            operation.other_columns(index, right, interval.start, interval.end),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                for (right, &index) in right.iter().zip(&right_source) {
                    let input = &operation.source(&prepared, right, index).value;
                    let compatible =
                        if matches!(operation, PreparedMatrixOperation::CrtRecompose { .. }) {
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
                        source,
                        &right,
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
                        .primary(left, &right, interval.interval.start)
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
                right,
                compact: compact.cloned(),
                plan,
                intervals,
                prepared: Arc::new(retained_inputs),
            });
        }
        Ok(compiled)
    }

    /// One claim broker per configured device over the accepted inventory.
    pub(super) fn claim_brokers(&self) -> Vec<PreparedClaimBroker> {
        (0..self.devices.len())
            .map(|device| {
                PreparedClaimBroker::new(
                    self.prepared_ledger
                        .as_ref()
                        .map(|ledger| {
                            ledger
                                .prepared_inventory()
                                .filter(|(owner, _)| *owner == device)
                                .map(|(_, storage)| storage.clone())
                                .collect()
                        })
                        .unwrap_or_default(),
                )
            })
            .collect()
    }

    /// Summaries of the currently admitted, unconsumed invocation batch in
    /// execution order. Reading them performs no GPU work and holds no waits.
    pub fn admitted_invocation_summaries(&self) -> Vec<GpuAdmittedInvocationSummary> {
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
                plan: compiled.plan.summary(),
            })
            .collect()
    }

    pub(super) fn validate_admitted_matrix_invocations(
        &self,
        requests: &[(usize, MatrixInvocation<'_>)],
    ) -> Result<(), PolyBackendError> {
        if requests.len() != self.prepared_invocations.len() {
            return Err(PolyBackendError::GpuSubmission(
                "preflight batch differs from admitted invocation count".into(),
            ));
        }
        for ((placement, request), compiled) in requests.iter().zip(&self.prepared_invocations) {
            let (operation, left, right, compact) =
                CompiledMatrixInvocation::arguments(request, self)?;
            if *placement != 0 || !compiled.matches(operation, left, &right, compact) {
                return Err(PolyBackendError::GpuSubmission(
                    "preflight arguments differ from admitted invocation".into(),
                ));
            }
        }
        Ok(())
    }

    pub(super) fn execute_admitted_matrix<M: PreparedFleetOutput>(
        &mut self,
        operation: PreparedMatrixOperation,
        left: Option<&GpuFleetMatrix>,
        right: &[GpuFleetMatrix],
        compact: Option<&GpuFleetSmallMatrix>,
        payload: ExecutionPayload,
    ) -> Result<M, PolyBackendError> {
        if operation.compact_bound().is_some() != M::COMPACT {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        if !self
            .prepared_invocations
            .front()
            .is_some_and(|compiled| compiled.matches(operation, left, &right, compact))
        {
            return Err(PolyBackendError::GpuSubmission(
                "matrix call has no matching admitted invocation".into(),
            ));
        }
        let payload =
            self.prepared_invocations.front().unwrap().operation.production_payload(payload)?;
        let CompiledMatrixInvocation { operation, left, right, compact, plan, intervals, prepared } =
            self.prepared_invocations.pop_front().unwrap();
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
        let brokers = Arc::new(self.claim_brokers());
        let outputs = plan
            .execute(
                &mut self.enqueue,
                &mut self.devices,
                move |device, _, fixed, physical_outputs| {
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
                move |job, reservations| {
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
                move |job, _, outputs, physical| {
                    if !physical.is_empty() {
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
                            &brokers[job.device],
                        )
                        .map_err(GpuAdmissionError::NativeReservation)?;
                    outputs[range.destination] = Some(GpuColumnShard { value, ..output });
                    Ok(())
                },
            )
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        let mut shards = outputs
            .into_iter()
            .flat_map(|(_, outputs)| outputs.into_iter().map(|output| output.unwrap()))
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
        poly::dcrt::{gpu::GpuMatrixExecutionClass, params::DCRTPolyParams},
        sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
    };

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
                (0..6).into_par_iter().map(|_| GpuDCRTPolyMatrix::zero(&params, 1, 2)).collect(),
                Some(&layouts),
            )
            .unwrap(),
        );
        let readback = Arc::new(
            GpuPreparedStorage::new(
                vec![GpuDCRTPolyMatrix::zero(&params, 1, 2)],
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
                        (0..8)
                            .into_par_iter()
                            .map(|_| GpuDCRTPolyMatrix::zero(&params, source_rows, 2))
                            .collect(),
                        Some(&layouts),
                    )
                    .unwrap(),
                );
                let readback = Arc::new(
                    GpuPreparedStorage::new(
                        vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
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
                    assert!(backend.preflight_gpu_operations(&[(0, bad)]).is_err());
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
                    // The plain hash and the other compact variant are not the
                    // admitted invocation.
                    assert!(backend.sample_hash(&ty, cases[0].0, &cases[0].1).is_err());
                    assert!(
                        if small {
                            backend.sample_hash_decomposed(
                                &ty,
                                cases[0].0,
                                &cases[0].1,
                                &base,
                                digit_count,
                            )
                        } else {
                            backend.sample_hash_small_decomposed(
                                &ty,
                                cases[0].0,
                                &cases[0].1,
                                &base,
                                digit_count,
                            )
                        }
                        .is_err()
                    );
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
                (0..2).map(|_| GpuDCRTPolyMatrix::zero(&root, 1, 1)).collect(),
                Some(&layouts),
            )
            .unwrap(),
        );
        // Readback permits belong to the context being read: root outputs and
        // low-basis inputs.
        let readback = |params: &GpuDCRTPolyParams| {
            Arc::new(
                GpuPreparedStorage::new(
                    vec![GpuDCRTPolyMatrix::zero(params, 1, 1)],
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
                        GpuInvocation::CenteredExtendSmall { value: &input, destination: &bad },
                    )])
                    .is_err()
            );
            assert!(backend.prepared_invocations.is_empty());
        }
        let request =
            [(0, GpuInvocation::CenteredExtendSmall { value: &input, destination: &destination })];
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
                (0..8)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, 3, split))
                    .collect(),
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
                vec![GpuDCRTPolyMatrix::zero(&params, 3, columns)],
                Some(&layouts),
            )
            .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        assert!(
            backend
                .preflight_gpu_operations(&[(
                    0,
                    GpuInvocation::MultiplySmallRhs { left: &wrong, right: &rhs },
                )])
                .is_err()
        );
        assert!(backend.prepared_invocations.is_empty());
        let block_refs = blocks.iter().collect::<Vec<_>>();
        let requests = [
            (0, GpuInvocation::MultiplySmallRhs { left: &lhs, right: &rhs }),
            (0, GpuInvocation::MultiplySmallRhsRowBlocks { blocks: &block_refs, right: &rhs }),
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
            assert_eq!(plan.devices[0].outputs.iter().filter(|c| c.kind == "Matrix").count(), 2);
            assert_eq!(
                plan.devices[0].scratch.iter().filter(|c| c.kind == "CompactWorkspace").count(),
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
        let storage = Arc::new(GpuPreparedStorage::new(matrices, Some(&layouts)).unwrap());
        backend.prepare_memory(vec![(0, storage)], true).unwrap();
        let wrong = ConcreteMatrixType { rows: rows + 1, ..ty.clone() };
        assert!(
            backend
                .preflight_gpu_operations(&[(
                    0,
                    GpuInvocation::ImportMatrix { ty: &wrong, bytes: &eval_bytes },
                )])
                .is_err()
        );
        assert!(backend.prepared_invocations.is_empty());
        let requests = [
            (0, GpuInvocation::ImportMatrix { ty: &ty, bytes: &eval_bytes }),
            (0, GpuInvocation::ImportMatrix { ty: &ty, bytes: &coeff_bytes }),
            (
                0,
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
            // The coefficient artifact is a different admitted operation.
            assert!(backend.matrix_from_bytes(&ty, &coeff_bytes).is_err());
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
                        (0..12 * columns)
                            .into_par_iter()
                            .map(|_| GpuDCRTPolyMatrix::zero(&params, 3, 3))
                            .collect(),
                        Some(&layouts),
                    )
                    .unwrap(),
                );
                let readback = Arc::new(
                    GpuPreparedStorage::new(
                        vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
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
                        GpuInvocation::GadgetDecompose {
                            value: &inputs[0],
                            small,
                            digit_count: None,
                        },
                    ),
                    (
                        0,
                        GpuInvocation::GadgetDecomposeRowBlocks {
                            blocks: &blocks,
                            small,
                            digit_count: None,
                        },
                    ),
                ];
                let bad = [(
                    0,
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
                        PreparedMatrixOperation::CrtRecompose {
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
                        (0..12 * columns)
                            .into_par_iter()
                            .map(|_| GpuDCRTPolyMatrix::zero(p, 1, 3))
                            .collect(),
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
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(target, 1, 3)], Some(&layouts))
                .unwrap(),
        );
        inventory.push((0, readback.clone()));
        backend.prepare_memory(inventory, true).unwrap();
        let requests = (0..indices.len())
            .map(|j| {
                (
                    0,
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
            assert_eq!(backend.calibration_registry.len(), 4);
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
                    PreparedMatrixOperation::RnsConversion {
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
                    (0..10 * columns)
                        .into_par_iter()
                        .map(|_| GpuDCRTPolyMatrix::zero(p, 9, 2))
                        .collect(),
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
                GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(p, 9, 2)], Some(&layouts))
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
            assert_eq!(backend.calibration_registry.len(), 5);
            assert!(
                backend.rns_mod_up(&inputs[1], &types[0], parameters[1].moduli(), 3, true).is_err()
            );
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
                    PreparedMatrixOperation::ModulusConversion {
                        destination: types[index].clone(),
                        conversion: GpuMatrixModulusConversion::Round,
                    }
                    .fixed_workspaces(parameters, parameters.crt_depth() - 1)
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let storage = Arc::new(
                GpuPreparedStorage::new(
                    (0..8 * columns)
                        .into_par_iter()
                        .map(|_| GpuDCRTPolyMatrix::zero(parameters, 3, 2))
                        .collect(),
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
                    vec![GpuDCRTPolyMatrix::zero(parameters, 3, 2)],
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
            assert_eq!(backend.calibration_registry.len(), 5);
            assert!(backend.block_mod_switch(&inputs[0], &types[1], 7).is_err());
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
                    (0..(8 * columns))
                        .into_par_iter()
                        .map(|_| GpuDCRTPolyMatrix::zero(parameters, 3, 2))
                        .collect(),
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
                    vec![GpuDCRTPolyMatrix::zero(parameters, 3, 2)],
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
            (0, GpuInvocation::CenteredRebase { value: &input, destination: &ty }),
            (0, GpuInvocation::CenteredRebase { value: &input, destination: &ty }),
            (0, GpuInvocation::ScaleInteger { value: &input, scalar: &scalar }),
            (0, GpuInvocation::Negate { value: &input }),
            (0, GpuInvocation::CenteredRebase { value: &narrow_input, destination: &ty }),
            (0, GpuInvocation::CenteredRebase { value: &empty_input, destination: &empty_ty }),
        ];
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert_eq!(
                backend.calibration_registry.len(),
                4,
                "different source bases require different profiles"
            );
            assert!(backend.centered_rebase(&input, &empty_ty).is_err());
            assert!(backend.negate(&input).is_err());
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
        let operation =
            PreparedMatrixOperation::Polynomial { ty: ty.clone(), coefficients: Vec::new() };
        let layouts = (0..cases.len())
            .flat_map(|_| operation.fixed_workspaces(&params, params.crt_depth() - 1).unwrap())
            .collect::<Vec<_>>();
        let storage = Arc::new(
            GpuPreparedStorage::new(
                (0..cases.len() + 1)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, 1, 1))
                    .collect(),
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
                vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)],
                Some(&readback_layout),
            )
            .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        let env = ParamEnv::default();
        let requests = cases
            .iter()
            .map(|(value, _)| (0, GpuInvocation::Constant { ty: &ty, value, env: &env }))
            .collect::<Vec<_>>();
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(backend.constant_matrix(&ty, &cases[1].0, &env).is_err());
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
        let lengths = [0, 13, 13, 65];
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
                (0..lengths.len() + 1)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, 3, columns))
                    .collect(),
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
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&layouts))
                .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        let empty = ConcreteMatrixType { columns: 0, ..ty.clone() };
        let mut requests = lengths
            .iter()
            .map(|&tag_bytes| {
                (
                    0,
                    GpuInvocation::SampleHash {
                        ty: &ty,
                        variant: mxx_ir_core::node::HashVariant::Plain,
                        tag_bytes,
                        gadget_base: None,
                        digit_count: None,
                    },
                )
            })
            .collect::<Vec<_>>();
        requests.push((
            0,
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
            assert!(backend.sample_hash(&ty, rand::random(), &[1]).is_err());
            assert!(
                backend
                    .execute_admitted_matrix::<GpuFleetMatrix>(
                        PreparedMatrixOperation::Hash { ty: ty.clone(), tag_bytes: 0 },
                        None,
                        &[],
                        None,
                        ExecutionPayload::None
                    )
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
                (0..7)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, 3, columns))
                    .collect(),
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
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&layouts))
                .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        let bound = BigInt::from(2);
        let empty = ConcreteMatrixType { columns: 0, ..ty.clone() };
        let mut requests = ranges
            .iter()
            .map(|range| (0, GpuInvocation::SampleUniform { ty: &ty, range }))
            .collect::<Vec<_>>();
        requests.extend([
            (
                0,
                GpuInvocation::SampleGaussian {
                    ty: &ty,
                    sigma: 4.578,
                    max_coefficient_bound: &bound,
                },
            ),
            (
                0,
                GpuInvocation::SampleGaussian {
                    ty: &ty,
                    sigma: 0.0,
                    max_coefficient_bound: &bound,
                },
            ),
            (0, GpuInvocation::SampleUniform { ty: &empty, range: &ranges[1] }),
        ]);
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(backend.sample_uniform(&ty, &ranges[1]).is_err());
            let mut actual = ranges
                .iter()
                .map(|range| backend.sample_uniform(&ty, range).unwrap())
                .collect::<Vec<_>>();
            assert!(backend.sample_gaussian(&ty, 4.579, &bound).is_err());
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
                    CompiledMatrixInvocation::arguments(&request, &backend),
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
                (0..cases.len() + 2)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, maximum_rows, maximum_columns))
                    .collect(),
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
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&layouts))
                .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        let env = ParamEnv::default();
        let mut requests = cases
            .iter()
            .zip(&types)
            .map(|((value, _), ty)| (0, GpuInvocation::Constant { ty, value, env: &env }))
            .collect::<Vec<_>>();
        requests.push((0, GpuInvocation::Negate { value: &input }));
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(backend.constant_matrix(&types[1], &cases[1].0, &env).is_err());
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
                shapes
                    .into_par_iter()
                    .map(|(r, c)| GpuDCRTPolyMatrix::zero(&params, r, c))
                    .collect(),
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
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&layouts))
                .unwrap(),
        );
        backend.prepare_memory(vec![(0, storage), (0, readback.clone())], true).unwrap();
        let requests = [
            (0, GpuInvocation::Accumulate { request: &request }),
            (0, GpuInvocation::Accumulate { request: &without_bias }),
        ];
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            let mut wrong = request.clone();
            wrong.products.swap(0, 1);
            assert!(backend.matrix_mul_accumulate(wrong).is_err());
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
                        shapes
                            .into_par_iter()
                            .map(|(r, c)| GpuDCRTPolyMatrix::zero(parameters, r, c))
                            .collect(),
                        None,
                    )
                    .unwrap(),
                ),
            ));
            let allocation = parameters
                .matrix_allocation_bytes(parameters.crt_depth() - 1, total_rows, 3, true)
                .unwrap();
            let events = if allocation.execution_class == GpuMatrixExecutionClass::PerLimbStreams {
                parameters.crt_depth()
            } else {
                1
            };
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
                    vec![GpuDCRTPolyMatrix::zero(parameters, 1, 1)],
                    Some(&layouts),
                )
                .unwrap(),
            );
            inventory.push((0, readback.clone()));
            readbacks.push(readback);
        }
        backend.prepare_memory(inventory, true).unwrap();
        let requests = [
            (0, GpuInvocation::Concat { inputs: &refs, axis: ConcatAxis::Columns }),
            (0, GpuInvocation::Concat { inputs: &refs, axis: ConcatAxis::Diagonal }),
        ];
        let cpu_refs = originals[1..].iter().collect::<Vec<_>>();
        let expected =
            [originals[0].concat_columns(&cpu_refs), originals[0].concat_diag(&cpu_refs)];
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(backend.concat(&refs, ConcatAxis::Diagonal).is_err());
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
                normalizations
                    .into_par_iter()
                    .map(|(r, c)| GpuDCRTPolyMatrix::zero(&params, r, c))
                    .collect(),
                None,
            )
            .unwrap(),
        );
        let outputs = Arc::new(
            GpuPreparedStorage::new(
                (0..2 * columns)
                    .into_par_iter()
                    .map(|_| GpuDCRTPolyMatrix::zero(&params, rows, 2))
                    .collect(),
                None,
            )
            .unwrap(),
        );
        let allocation =
            params.matrix_allocation_bytes(params.crt_depth() - 1, rows, 2, true).unwrap();
        let events = if allocation.execution_class == GpuMatrixExecutionClass::PerLimbStreams {
            params.crt_depth()
        } else {
            1
        };
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
            GpuPreparedStorage::new(vec![GpuDCRTPolyMatrix::zero(&params, 1, 1)], Some(&layouts))
                .unwrap(),
        );
        backend
            .prepare_memory(
                vec![(0, input_storage.clone()), (0, outputs.clone()), (0, readback.clone())],
                true,
            )
            .unwrap();
        let requests = [
            (0, GpuInvocation::Concat { inputs: &refs, axis: ConcatAxis::Rows }),
            (0, GpuInvocation::AddRowBlocks { blocks: &refs, right: &right }),
        ];
        let expected = [stacked.clone(), stacked.add_out_of_place(&rhs_cpu)];
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            let mut substituted = refs.clone();
            substituted.swap(0, 2);
            assert!(backend.concat(&substituted, ConcatAxis::Rows).is_err());
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
