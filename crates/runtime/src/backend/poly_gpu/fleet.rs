use super::{
    super::poly::{PolyBackend, PolyBackendError, encode_small_matrix_artifact},
    gpu_prepared_lowering,
};

pub(crate) fn prepared_runtime_value(
    value: &crate::backend::RuntimeValue<GpuDcrtBackend>,
) -> Result<super::PreparedRuntimeValue, PolyBackendError> {
    match value {
        crate::backend::RuntimeValue::HostMatrix { matrix_type, bytes } => {
            Ok(super::PreparedRuntimeValue::HostMatrix {
                matrix_type: matrix_type.clone(),
                bytes: bytes.as_ref().clone().into_boxed_slice(),
            })
        }
        crate::backend::RuntimeValue::Matrix(value) => {
            Ok(super::PreparedRuntimeValue::FleetMatrix(Arc::clone(value)))
        }
        crate::backend::RuntimeValue::SmallMatrix(value) => {
            Ok(super::PreparedRuntimeValue::FleetSmallMatrix(Arc::clone(value)))
        }
        crate::backend::RuntimeValue::Int(value) => {
            Ok(super::PreparedRuntimeValue::Int(value.clone()))
        }
        crate::backend::RuntimeValue::Real(value) => Ok(super::PreparedRuntimeValue::Real(*value)),
        crate::backend::RuntimeValue::Bool(value) => Ok(super::PreparedRuntimeValue::Bool(*value)),
        crate::backend::RuntimeValue::Bytes(value) |
        crate::backend::RuntimeValue::TypedBlob(value) => {
            Ok(super::PreparedRuntimeValue::Bytes(value.as_slice().to_owned().into_boxed_slice()))
        }
        crate::backend::RuntimeValue::IndexedFamily(values) => {
            let members =
                values.iter().map(prepared_runtime_value).collect::<Result<Vec<_>, _>>()?;
            Ok(super::PreparedRuntimeValue::Family(members.into()))
        }
        crate::backend::RuntimeValue::Trapdoor { secret: Some(secret), public, .. } => {
            Ok(super::PreparedRuntimeValue::Trapdoor {
                secret: Arc::clone(secret),
                public: Arc::clone(public),
            })
        }
        _ => Err(PolyBackendError::GpuSubmission(
            "prepared input kind has no fixed runtime binding".into(),
        )),
    }
}

/// Convert warmup inputs to metadata-only prepared values. Host staging is
/// validated before reservation; the generic prepared builder later binds
/// each device's reserved upload target without materializing a source owner.
pub(crate) fn prepared_runtime_value_for_warmup(
    backend: &mut GpuDcrtBackend,
    value: &crate::backend::RuntimeValue<GpuDcrtBackend>,
) -> Result<super::PreparedRuntimeValue, PolyBackendError> {
    match value {
        crate::backend::RuntimeValue::HostMatrix { matrix_type, bytes } => {
            let parameters = backend
                .devices
                .iter()
                .map(|(_, device)| {
                    device
                        .parameters(matrix_type)
                        .map_err(|_| PolyBackendError::InvalidConstantShape)
                })
                .collect::<Result<Vec<_>, _>>()?;
            for parameters in &parameters {
                GpuDCRTPolyMatrix::cpu_staging_layout(parameters, bytes)
                    .map_err(PolyBackendError::GpuSubmission)?;
            }
            Ok(super::PreparedRuntimeValue::HostMatrix {
                matrix_type: matrix_type.clone(),
                bytes: bytes.as_ref().clone().into_boxed_slice(),
            })
        }
        crate::backend::RuntimeValue::IndexedFamily(values) => {
            let members = values
                .iter()
                .map(|value| prepared_runtime_value_for_warmup(backend, value))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(super::PreparedRuntimeValue::Family(members.into()))
        }
        _ => prepared_runtime_value(value),
    }
}

use crate::{
    backend::{
        Backend, ExecutionStrategy, IndexRange, MatrixMulAccumulateRequest, PreimageTarget,
        SampleRange,
    },
    transcript::{SamplingMode, TranscriptRecorder, TranscriptReplayer},
};
use mxx_ir_core::{
    ParamEnv,
    artifact::{ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind},
    node::{ConcatAxis, ConstantMatrix},
    types::ConcreteMatrixType,
};
use mxx_primitives::{
    matrix::{
        PolyMatrix, PolyMatrixColumnSource, SmallPolyMatrix,
        gpu_dcrt_poly::{
            GpuDCRTMatrixRnsSnapshot, GpuDCRTPolyMatrix, GpuRnsSnapshotTransfer, GpuSmallMatrix,
        },
    },
    poly::{
        PolyParams,
        dcrt::gpu::{GpuDCRTPolyParams, gpu_device_identity},
    },
    sampler::{
        gpu::{GpuDCRTPolyHashSampler, GpuDCRTPolyUniformSampler},
        trapdoor::{GpuDCRTPolyTrapdoorSampler, GpuDCRTTrapdoor},
    },
};
use num_bigint::BigInt;
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

#[path = "gpu_claims.rs"]
mod gpu_claims;
#[path = "gpu_inventory.rs"]
mod gpu_inventory;
#[path = "gpu_prepare.rs"]
mod gpu_prepare;
use super::gpu_prepared::PreparedGpuProgram;
pub(crate) use gpu_inventory::PreparedScheduleStreamKey;
pub use gpu_prepare::{
    MatrixDescriptor as GpuMatrixDescriptor,
    MatrixFragmentDescriptor as GpuMatrixFragmentDescriptor,
    MatrixInputFragment as GpuMatrixInputFragment, MatrixInputLayout as GpuMatrixInputLayout,
    MatrixInputRequest as GpuMatrixInputRequest, MatrixSlotContext as GpuMatrixSlotContext,
    MatrixSlotInventory as GpuMatrixSlotInventory, PreparedMatrixSource as GpuMatrixInputSource,
};

type DeviceBackend = PolyBackend<
    GpuDCRTPolyMatrix,
    GpuDCRTPolyUniformSampler,
    GpuDCRTPolyHashSampler<keccak_asm::Keccak256>,
    GpuDCRTPolyTrapdoorSampler,
>;

static NEXT_FLEET_VALUE_ID: AtomicU64 = AtomicU64::new(1);

type CompactMatrixEncoding = (u8, u8, u32, usize, usize, u16, u16, Vec<u8>);

impl GpuDcrtBackend {
    fn start_rns_snapshot<'a>(
        &mut self,
        matrix: &'a GpuDCRTPolyMatrix,
    ) -> Result<GpuRnsSnapshotTransfer<'a>, PolyBackendError> {
        Ok(matrix.start_rns_snapshot(self.rns_staging_buffers.pop()))
    }
}

fn decode_compact_matrix(bytes: &[u8]) -> Result<CompactMatrixEncoding, PolyBackendError> {
    bincode::decode_from_slice(bytes, bincode::config::standard())
        .map(|decoded| decoded.0)
        .map_err(|_| PolyBackendError::InvalidInteger)
}

fn copy_packed_bits(
    source: &[u8],
    source_bit: usize,
    destination: &mut [u8],
    destination_bit: usize,
    bit_count: usize,
) {
    let mut copied = 0;
    while copied < bit_count && (destination_bit + copied) % 8 != 0 {
        let bit = (source[(source_bit + copied) / 8] >> ((source_bit + copied) % 8)) & 1;
        destination[(destination_bit + copied) / 8] |= bit << ((destination_bit + copied) % 8);
        copied += 1;
    }
    let bytes = (bit_count - copied) / 8;
    let source_start = (source_bit + copied) / 8;
    let target_start = (destination_bit + copied) / 8;
    let shift = (source_bit + copied) % 8;
    let target = &mut destination[target_start..target_start + bytes];
    if shift == 0 {
        for (target, source) in target.iter_mut().zip(&source[source_start..source_start + bytes]) {
            *target |= source;
        }
    } else {
        for (index, target) in target.iter_mut().enumerate() {
            *target |= (source[source_start + index] >> shift) |
                (source[source_start + index + 1] << (8 - shift));
        }
    }
    copied += bytes * 8;
    while copied < bit_count {
        let bit = (source[(source_bit + copied) / 8] >> ((source_bit + copied) % 8)) & 1;
        destination[(destination_bit + copied) / 8] |= bit << ((destination_bit + copied) % 8);
        copied += 1;
    }
}

fn fleet_context_vram_percent<T>(
    placements: &[Vec<T>],
    fixed_percent: impl Fn(&T) -> u32,
    fixed_budget: impl Fn(&T) -> usize,
) -> Result<u32, String> {
    let first = placements
        .first()
        .and_then(|placement| placement.first())
        .ok_or_else(|| "a GPU fleet needs nonempty device parameters".to_owned())?;
    let fleet_percent = fixed_percent(first);
    for (placement, parameters) in placements.iter().enumerate() {
        let first = parameters
            .first()
            .ok_or_else(|| format!("GPU placement {placement} has no parameters"))?;
        let placement_budget = fixed_budget(first);
        for parameters in parameters {
            let percent = fixed_percent(parameters);
            if percent != fleet_percent {
                return Err(format!(
                    "GPU fleet contexts disagree on fixed VRAM percentage: expected {fleet_percent}, got {percent} at placement {placement}"
                ));
            }
            let budget = fixed_budget(parameters);
            if budget != placement_budget {
                return Err(format!(
                    "GPU placement {placement} contexts disagree on fixed VRAM budget: expected {placement_budget}, got {budget}"
                ));
            }
        }
    }
    Ok(fleet_percent)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuColumnShard<T> {
    pub device_id: i32,
    pub global_column_start: usize,
    pub value: T,
}

#[derive(Clone, Debug)]
pub struct GpuFleetMatrix {
    id: u64,
    rows: usize,
    columns: usize,
    // Logical aliases share native allocation and release ownership. Cloning a
    // primitive matrix here would allocate a second, unreserved GPU payload.
    shards: Arc<Vec<GpuColumnShard<Arc<GpuDCRTPolyMatrix>>>>,
    // Publish immutable source geometry with the actual owner. Execution and
    // downstream preparation share this metadata without re-inspecting shards.
    input_layout: Arc<[gpu_prepare::MatrixInputFragment]>,
    // A prepared execution slot remains pinned while the returned resident
    // value is alive. Ordinary values leave this empty.
    prepared_lease: Option<Arc<super::gpu_prepared::PreparedGpuFleetOutput>>,
}

impl PartialEq for GpuFleetMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.columns == other.columns && self.shards == other.shards
    }
}

impl Eq for GpuFleetMatrix {}

impl GpuFleetMatrix {
    pub fn new(
        rows: usize,
        columns: usize,
        shards: Vec<GpuColumnShard<GpuDCRTPolyMatrix>>,
    ) -> Self {
        let shards = shards
            .into_iter()
            .map(|shard| GpuColumnShard {
                device_id: shard.device_id,
                global_column_start: shard.global_column_start,
                value: Arc::new(shard.value),
            })
            .collect::<Vec<_>>();
        validate_shards(rows, columns, &shards, |matrix| matrix.size());
        let input_layout = shards
            .iter()
            .map(|shard| gpu_prepare::MatrixInputFragment {
                device: shard.device_id,
                context: shard.value.params().context_identity(),
                start: shard.global_column_start,
                end: shard.global_column_start + shard.value.col_size(),
                level: shard.value.level(),
                evaluation: shard.value.is_ntt(),
            })
            .collect();
        Self {
            id: NEXT_FLEET_VALUE_ID.fetch_add(1, Ordering::Relaxed),
            rows,
            columns,
            shards: Arc::new(shards),
            input_layout,
            prepared_lease: None,
        }
    }

    pub(super) fn with_prepared_lease(
        mut value: Self,
        lease: Arc<super::gpu_prepared::PreparedGpuFleetOutput>,
    ) -> Self {
        value.prepared_lease = Some(lease);
        value
    }

    pub(super) fn from_shared_shards<I>(rows: usize, columns: usize, shards: I) -> Self
    where
        I: IntoIterator<Item = GpuColumnShard<Arc<GpuDCRTPolyMatrix>>>,
    {
        let shards = shards.into_iter().collect::<Vec<_>>();
        validate_shards(rows, columns, &shards, |matrix| matrix.size());
        let input_layout = shards
            .iter()
            .map(|shard| gpu_prepare::MatrixInputFragment {
                device: shard.device_id,
                context: shard.value.params().context_identity(),
                start: shard.global_column_start,
                end: shard.global_column_start + shard.value.col_size(),
                level: shard.value.level(),
                evaluation: shard.value.is_ntt(),
            })
            .collect();
        Self {
            id: NEXT_FLEET_VALUE_ID.fetch_add(1, Ordering::Relaxed),
            rows,
            columns,
            shards: Arc::new(shards),
            input_layout,
            prepared_lease: None,
        }
    }

    pub fn from_matrix(value: GpuDCRTPolyMatrix) -> Self {
        let (rows, columns) = value.size();
        let device_id = value.params().device_ids().first().copied().unwrap_or(0);
        Self::new(rows, columns, vec![GpuColumnShard { device_id, global_column_start: 0, value }])
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub fn shards(&self) -> &[GpuColumnShard<Arc<GpuDCRTPolyMatrix>>] {
        &self.shards
    }
    pub fn wait_until_ready(&self) -> Result<(), String> {
        self.shards.iter().try_for_each(|shard| shard.value.wait_until_ready_result())?;
        if let Some(lease) = &self.prepared_lease {
            lease.check_device_scalar_status()?;
        }
        Ok(())
    }
}

impl From<GpuDCRTPolyMatrix> for GpuFleetMatrix {
    fn from(value: GpuDCRTPolyMatrix) -> Self {
        Self::from_matrix(value)
    }
}

#[derive(Clone, Debug)]
pub struct GpuFleetSmallMatrix {
    id: u64,
    rows: usize,
    columns: usize,
    // Keep compact allocations alive until the last logical alias is dropped.
    shards: Arc<Vec<GpuColumnShard<Arc<GpuSmallMatrix>>>>,
    prepared_lease: Option<Arc<super::gpu_prepared::PreparedGpuFleetOutput>>,
}

impl PartialEq for GpuFleetSmallMatrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.columns == other.columns && self.shards == other.shards
    }
}

impl Eq for GpuFleetSmallMatrix {}

impl GpuFleetSmallMatrix {
    pub fn new(rows: usize, columns: usize, shards: Vec<GpuColumnShard<GpuSmallMatrix>>) -> Self {
        validate_shards(rows, columns, &shards, |matrix| matrix.size());
        let shards = shards
            .into_iter()
            .map(|shard| GpuColumnShard {
                device_id: shard.device_id,
                global_column_start: shard.global_column_start,
                value: Arc::new(shard.value),
            })
            .collect();
        Self {
            id: NEXT_FLEET_VALUE_ID.fetch_add(1, Ordering::Relaxed),
            rows,
            columns,
            shards: Arc::new(shards),
            prepared_lease: None,
        }
    }

    pub(super) fn with_prepared_lease(
        mut value: Self,
        lease: Arc<super::gpu_prepared::PreparedGpuFleetOutput>,
    ) -> Self {
        value.prepared_lease = Some(lease);
        value
    }

    pub(super) fn from_shared_shards<I>(rows: usize, columns: usize, shards: I) -> Self
    where
        I: IntoIterator<Item = GpuColumnShard<Arc<GpuSmallMatrix>>>,
    {
        let shards = shards.into_iter().collect::<Vec<_>>();
        validate_shards(rows, columns, &shards, |matrix| matrix.size());
        Self {
            id: NEXT_FLEET_VALUE_ID.fetch_add(1, Ordering::Relaxed),
            rows,
            columns,
            shards: Arc::new(shards),
            prepared_lease: None,
        }
    }

    pub fn from_matrix(value: GpuSmallMatrix) -> Self {
        let (rows, columns) = value.size();
        let device_id = value.params().device_ids().first().copied().unwrap_or(0);
        Self::new(rows, columns, vec![GpuColumnShard { device_id, global_column_start: 0, value }])
    }

    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
    pub fn shards(&self) -> &[GpuColumnShard<Arc<GpuSmallMatrix>>] {
        &self.shards
    }
    pub fn wait_until_ready(&self) -> Result<(), String> {
        self.shards.iter().try_for_each(|shard| shard.value.wait_until_ready_result())?;
        if let Some(lease) = &self.prepared_lease {
            lease.check_device_scalar_status()?;
        }
        Ok(())
    }
}

impl From<GpuSmallMatrix> for GpuFleetSmallMatrix {
    fn from(value: GpuSmallMatrix) -> Self {
        Self::from_matrix(value)
    }
}

#[derive(Clone, Debug)]
pub struct GpuFleetTrapdoor {
    pub(super) values: Arc<Vec<Arc<GpuDCRTTrapdoor>>>,
    pub(super) prepared_lease: Option<Arc<super::gpu_prepared::PreparedGpuFleetOutput>>,
}

impl GpuFleetTrapdoor {
    pub fn wait_until_ready(&self) -> Result<(), String> {
        if let Some(lease) = &self.prepared_lease {
            lease.wait_until_ready()?;
        }
        self.values.iter().try_for_each(|value| value.wait_until_ready_result())?;
        Ok(())
    }
}

fn validate_shards<T>(
    rows: usize,
    columns: usize,
    shards: &[GpuColumnShard<T>],
    size: impl Fn(&T) -> (usize, usize),
) {
    assert!(columns == 0 || !shards.is_empty(), "a nonempty fleet value needs a shard");
    let mut next = 0usize;
    for shard in shards {
        let (local_rows, local_columns) = size(&shard.value);
        assert_eq!(local_rows, rows, "fleet shard row mismatch");
        assert_eq!(shard.global_column_start, next, "fleet shards must be ordered and contiguous");
        next = next.checked_add(local_columns).expect("fleet column count overflow");
    }
    assert_eq!(next, columns, "fleet shards must cover every logical column exactly once");
}

impl GpuFleetTrapdoor {
    /// Prepare the fixed covariance owner consumed by subsequent Preimage
    /// sampling. Explicit benchmark setup calls this before its storage seal,
    /// matching production trapdoor preparation without a sampler trial.
    pub fn prepare_preimage_cache(&self, sigma: f64, public_rows: usize) {
        self.values.par_iter().for_each(|trapdoor| {
            let parameters = trapdoor.r.params();
            <GpuDCRTPolyTrapdoorSampler as mxx_primitives::sampler::PolyTrapdoorSampler>::new(
                parameters, sigma,
            )
            .prepare_preimage_cache(parameters, trapdoor, public_rows);
        });
    }
}

impl fmt::Display for GpuFleetTrapdoor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "GPU trapdoor replicated on {} device(s)", self.values.len())
    }
}

pub struct GpuDcrtBackend {
    pub(super) devices: Vec<(i32, DeviceBackend)>,
    vram_percent: u32,
    rns_staging_buffers: Vec<GpuDCRTMatrixRnsSnapshot>,
    prepared_wave_bound: Option<std::num::NonZeroUsize>,
    prepared_live_executions: Option<std::num::NonZeroUsize>,
    prepared_graph: Option<PreparedGpuProgram>,
    prepared_spec_hash: Option<[u8; 32]>,
    prepared_ledger: Option<crate::gpu_memory::GpuMemoryLedger>,
}

impl GpuDcrtBackend {
    #[cfg(all(test, feature = "gpu-instrumentation"))]
    pub(super) fn prepared_execution_for_test(&self) -> &PreparedGpuProgram {
        self.prepared_graph.as_ref().expect("warmed execution")
    }

    /// Prepare a static graph at the explicit warmup boundary. Unsupported
    /// topologies fail here; they are never silently executed by the prepared
    /// replay path. Reusing the same validated graph is already warmed.
    pub fn warm_up_prepared_graph(
        &mut self,
        validated: &mxx_ir_core::ValidatedGraph,
        inputs: &BTreeMap<String, crate::backend::RuntimeValue<Self>>,
        config: &crate::ExecutionConfig,
    ) -> Result<(), PolyBackendError> {
        let wave_bound = config.max_parallel_instances;
        let live_executions = config.max_live_gpu_executions;
        let spec_hash = validated.spec_hash();
        if self.prepared_wave_bound == Some(wave_bound) &&
            self.prepared_live_executions == Some(live_executions) &&
            self.prepared_spec_hash == Some(spec_hash.0) &&
            self.prepared_graph.is_some()
        {
            return Ok(());
        }
        // Capture export metadata before publishing a new program.  This is
        // warmup-only graph access; production execute consumes the immutable
        // table retained by PreparedGpuProgram.
        let artifact_descriptors = crate::executor::prepared_artifact_descriptors(validated)
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        let previous_graph = self.prepared_graph.take();
        let previous_spec_hash = self.prepared_spec_hash;
        let previous_wave_bound = self.prepared_wave_bound;
        let previous_live_executions = self.prepared_live_executions;
        let warmup_checkpoint =
            self.prepared_ledger.as_ref().map(|ledger| ledger.warmup_checkpoint());
        // Warmup is transactional across all discovery caches as well as the
        // published replay.  A failed lowering/provisioning attempt must not
        // leave a partial preimage or polynomial class visible to a later
        // retry, since those entries may refer to owners that were rolled back.
        let warmup = self.warm_up_prepared_graph_impl(
            validated,
            inputs,
            wave_bound.get(),
            config.max_live_gpu_executions.get(),
        );
        let warmup = warmup.and_then(|_| {
            if self.prepared_graph.is_some() {
                Ok(())
            } else {
                Err(PolyBackendError::GpuSubmission(
                    "prepared graph warmup did not publish an executable".into(),
                ))
            }
        });
        if warmup.is_err() {
            if let (Some(ledger), Some(checkpoint)) =
                (self.prepared_ledger.as_mut(), warmup_checkpoint)
            {
                ledger.rollback_warmup(checkpoint);
            }
            self.prepared_graph = previous_graph;
            self.prepared_wave_bound = previous_wave_bound;
            self.prepared_live_executions = previous_live_executions;
            self.prepared_spec_hash = previous_spec_hash;
        } else {
            self.prepared_wave_bound = Some(wave_bound);
            self.prepared_live_executions = Some(live_executions);
            self.prepared_spec_hash = Some(spec_hash.0);
            if let Some(execution) = self.prepared_graph.as_mut() {
                execution.set_spec_hash(spec_hash.0);
                execution.set_artifact_descriptors(artifact_descriptors);
            }
        }
        warmup
    }

    pub fn new(placements: Vec<Vec<GpuDCRTPolyParams>>) -> Self {
        assert!(!placements.is_empty(), "a GPU fleet needs at least one device");
        let vram_percent = fleet_context_vram_percent(
            &placements,
            GpuDCRTPolyParams::vram_percent,
            GpuDCRTPolyParams::vram_budget_bytes,
        )
        .unwrap_or_else(|error| panic!("invalid GPU fleet context configuration: {error}"));
        let devices: Vec<_> = placements
            .into_iter()
            .map(|parameters| {
                let device_id = parameters
                    .first()
                    .and_then(|parameters| parameters.device_ids().first().copied())
                    .expect("each GPU placement needs device parameters");
                (device_id, DeviceBackend::new(parameters))
            })
            .collect();
        let mut unique = HashSet::new();
        assert!(
            devices.iter().all(|(id, _)| unique.insert(*id)),
            "GPU fleet device IDs must be unique"
        );
        let identities = devices
            .par_iter()
            .map(|(device, _)| gpu_device_identity(*device))
            .collect::<Result<Vec<_>, _>>()
            .expect("query GPU fleet identities");
        assert!(
            identities.iter().all(|identity| identity == &identities[0]),
            "GPU fleet requires matching models, compute capabilities and physical VRAM"
        );
        Self {
            devices,
            vram_percent,
            rns_staging_buffers: Vec::new(),
            prepared_wave_bound: None,
            prepared_live_executions: None,
            prepared_graph: None,
            prepared_spec_hash: None,
            prepared_ledger: None,
        }
    }

    /// Percentage of physical VRAM fixed when this fleet context was created.
    pub fn vram_percent(&self) -> u32 {
        self.vram_percent
    }

    /// One context per configured device, in fleet order. Related parameter
    /// views on a device share the execution owner used by device-span timing.
    /// Clone registered parameter views, retaining each device's exact contexts.
    /// This does not clone allocations, reservations, or calibration state.
    pub fn parameter_placements(&self) -> Vec<Vec<GpuDCRTPolyParams>> {
        self.devices
            .iter()
            .map(|(_, backend)| backend.parameters[0].values().cloned().collect())
            .collect()
    }

    pub fn device_parameters(&self) -> Vec<GpuDCRTPolyParams> {
        self.devices
            .par_iter()
            .map(|(_, backend)| {
                backend.parameters[0]
                    .values()
                    .next()
                    .expect("each GPU placement has parameters")
                    .clone()
            })
            .collect()
    }

    /// Registered parameter views for this ring, in configured device order.
    /// This only reads context metadata; it creates no GPU resource.
    pub fn resource_parameters(
        &self,
        matrix: &ConcreteMatrixType,
    ) -> Result<Vec<GpuDCRTPolyParams>, PolyBackendError> {
        self.devices.par_iter().map(|(_, backend)| backend.parameters(matrix).cloned()).collect()
    }

    /// Concrete resident CRT tower count for an exactly registered ring.
    pub fn ring_crt_depth(&self, matrix: &ConcreteMatrixType) -> Result<usize, PolyBackendError> {
        Ok(self.devices[0].1.parameters(matrix)?.crt_depth())
    }

    /// Builds a local representative of a regular gadget's global column range.
    ///
    /// This is used by the estimator's single-device worker. Validation is
    /// deliberately against the complete declared matrix and gadget layout;
    /// only allocation and construction are restricted to the measured range.
    fn gather_matrix(
        &mut self,
        value: &GpuFleetMatrix,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError> {
        Self::matrix_piece_on_device(&mut self.devices[0].1, value, 0, value.columns)
    }

    /// Materializes one logical fleet value on GPU 0 for host-facing decode or
    /// compatibility at explicit artifact/test boundaries. Production
    /// column-separable operations must keep using the shard methods.
    pub fn gather_matrix_for_host(
        &mut self,
        value: &GpuFleetMatrix,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError> {
        self.gather_matrix(value)
    }

    fn matrix_piece_on_device(
        backend: &mut DeviceBackend,
        value: &GpuFleetMatrix,
        start: usize,
        end: usize,
    ) -> Result<GpuDCRTPolyMatrix, PolyBackendError> {
        let mut pieces = Vec::new();
        for shard in value.shards.iter() {
            let shard_start = shard.global_column_start;
            let shard_end = shard_start + shard.value.col_size();
            let overlap_start = start.max(shard_start);
            let overlap_end = end.min(shard_end);
            if overlap_start >= overlap_end {
                continue;
            }
            let local =
                shard.value.slice_columns(overlap_start - shard_start, overlap_end - shard_start);
            pieces.push(backend.matrix_to_active_placement_peer_only(&local)?);
        }
        let mut pieces = pieces.into_iter();
        let first = pieces.next().expect("requested matrix range must be covered");
        Ok(first.concat_columns_owned(pieces.collect()))
    }
}

impl Backend for GpuDcrtBackend {
    type Matrix = GpuFleetMatrix;
    type SmallMatrix = GpuFleetSmallMatrix;
    type Trapdoor = GpuFleetTrapdoor;
    type Error = PolyBackendError;
    const EXECUTION_STRATEGY: ExecutionStrategy = ExecutionStrategy::Prepared;

    fn polynomial_values(
        &mut self,
        value: &Self::Matrix,
        evaluation: bool,
    ) -> Result<Vec<BigInt>, Self::Error> {
        if value.size() != (1, 1) || value.shards.len() != 1 {
            return Err(PolyBackendError::InvalidInteger);
        }
        let first = &value.shards[0];
        let device = self
            .devices
            .iter()
            .position(|(id, _)| *id == first.device_id)
            .ok_or(PolyBackendError::InvalidInteger)?;
        self.devices[device].1.polynomial_values(&first.value, evaluation)
    }

    fn polynomial_from_values(
        &mut self,
        _ty: &ConcreteMatrixType,
        _values: &[BigInt],
        _evaluation: bool,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn warm_up_prepared_graph(
        &mut self,
        validated: &mxx_ir_core::ValidatedGraph,
        inputs: &std::collections::BTreeMap<String, crate::backend::RuntimeValue<Self>>,
        config: &crate::executor::ExecutionConfig,
    ) -> Result<(), Self::Error> {
        GpuDcrtBackend::warm_up_prepared_graph(self, validated, inputs, config)
    }

    fn prepared_spec_hash(&self) -> Option<[u8; 32]> {
        self.prepared_spec_hash
    }

    fn execute_prepared_graph(
        &mut self,
        spec_hash: [u8; 32],
        _validated: &mxx_ir_core::ValidatedGraph,
        inputs: &BTreeMap<String, crate::backend::RuntimeValue<Self>>,
        mut context: crate::executor::PreparedExecutionContext<'_, Self>,
    ) -> Result<crate::executor::ExecutionResult<Self>, Self::Error> {
        let session = context.session.clone();
        let mut sampling_mode = context.sampling_mode;
        let mut transcript_store = context.transcript_store.take();
        if self.prepared_graph.is_none() {
            return Err(PolyBackendError::NotPrepared);
        }
        let execution = self
            .prepared_graph
            .as_ref()
            .expect("prepared graph checked immediately before input binding");
        if execution.spec_hash() != spec_hash {
            return Err(PolyBackendError::PreparedContractMismatch);
        }

        // A prepared session owns its transcript policy.  Existing entries
        // select replay; an empty session selects record.  Partial tapes are
        // Sparse entries are intentional: zero-count loops and unselected
        // branches have no draw. Any selected site is checked at its fixed
        // replay step before submission.
        let mut recorder = None;
        let replayer = if let Some(production) = session.as_ref() {
            let store = transcript_store.as_deref_mut().ok_or_else(|| {
                PolyBackendError::GpuSubmission(
                    "prepared session has no transcript store".to_owned(),
                )
            })?;
            let entries =
                store.transcript_entries(production).map_err(PolyBackendError::GpuSubmission)?;
            if entries.is_empty() {
                recorder = Some(TranscriptRecorder::default());
                sampling_mode = SamplingMode::Record(recorder.as_mut().expect("recorder bound"));
                None
            } else {
                Some(TranscriptReplayer::from_entries(entries))
            }
        } else {
            None
        };
        if let Some(replayer) = replayer.as_ref() {
            sampling_mode = SamplingMode::Replay(replayer);
        }

        let mut scalar_allocator = self
            .prepared_ledger
            .as_mut()
            .map(super::gpu_prepared::LedgerScalarCapacityAllocator::new);
        let execution = execution
            .run_with_runtime_bindings(
                inputs,
                &mut sampling_mode,
                scalar_allocator.as_mut().map(|allocator| {
                    allocator as &mut dyn mxx_primitives::matrix::gpu_dcrt_poly::GpuScalarCapacityAllocator
                }),
            )
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        if let (Some(production), Some(recorder), Some(store)) =
            (session.as_ref(), recorder.as_ref(), transcript_store.as_deref_mut())
        {
            let entries = recorder
                .iter()
                .map(|(site, value)| (site.clone(), value.clone()))
                .collect::<Vec<_>>();
            store
                .record_transcript_batch(production, &entries)
                .map_err(PolyBackendError::GpuSubmission)?;
        }
        let execution = Arc::new(execution);
        if context.capture_trace {
            let trace = context.trace.as_deref_mut().ok_or_else(|| {
                PolyBackendError::GpuSubmission(
                    "prepared trace capture requested without a trace sink".into(),
                )
            })?;
            self.prepared_graph
                .as_ref()
                .expect("prepared graph checked immediately before trace publication")
                .capture_trace(&execution, trace)
                .map_err(PolyBackendError::GpuSubmission)?;
        }
        Ok(crate::executor::ExecutionResult {
            outputs: BTreeMap::new(),
            production_id: session,
            artifact_handles: BTreeMap::new(),
            staged_family_leases: Vec::new(),
            prepared_outputs: Some(Arc::new(Arc::clone(&execution))),
        })
    }

    // A fleet is one production placement. Device parallelism is internal to
    // each primitive call, so the executor must not multiply work by GPU count.
    fn placement_count(&self) -> usize {
        1
    }

    fn fence_released_memory(&mut self) -> Result<(), Self::Error> {
        self.devices.par_iter_mut().try_for_each(|(_, backend)| backend.fence_released_memory())
    }

    fn constant_matrix(
        &mut self,
        _ty: &ConcreteMatrixType,
        _value: &ConstantMatrix,
        _env: &ParamEnv,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn add(
        &mut self,
        _left: &Self::Matrix,
        _right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn add_batch(
        &mut self,
        _inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn add_row_blocks(
        &mut self,
        _blocks: &[&Self::Matrix],
        _right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn sub(
        &mut self,
        _left: &Self::Matrix,
        _right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn sub_batch(
        &mut self,
        _inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn multiply(
        &mut self,
        _left: &Self::Matrix,
        _right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn multiply_batch(
        &mut self,
        _inputs: Vec<(Arc<Self::Matrix>, Arc<Self::Matrix>)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn matrix_mul_accumulate(
        &mut self,
        _request: MatrixMulAccumulateRequest<Self::Matrix>,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn matrix_mul_accumulate_batch(
        &mut self,
        _requests: Vec<MatrixMulAccumulateRequest<Self::Matrix>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn negate(&mut self, _value: &Self::Matrix) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn negate_batch(
        &mut self,
        _inputs: Vec<Arc<Self::Matrix>>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn scale_integer(
        &mut self,
        _value: &Self::Matrix,
        _scalar: &BigInt,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn scale_integer_batch(
        &mut self,
        _inputs: Vec<(Arc<Self::Matrix>, BigInt)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn ring_automorphism(
        &mut self,
        _value: &Self::Matrix,
        _index: usize,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn ring_automorphism_batch(
        &mut self,
        _inputs: Vec<(Arc<Self::Matrix>, usize)>,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn modulus_switch(
        &mut self,
        _value: &Self::Matrix,
        _destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn centered_rebase(
        &mut self,
        _value: &Self::Matrix,
        _destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn rns_mod_up(
        &mut self,
        _value: &Self::Matrix,
        _destination: &ConcreteMatrixType,
        _source_moduli: &[u64],
        _digit_size: usize,
        _normalize: bool,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn rns_mod_down(
        &mut self,
        _value: &Self::Matrix,
        _destination: &ConcreteMatrixType,
        _source_moduli: &[u64],
        _plaintext_modulus: u64,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn reduce_modulus(
        &mut self,
        _value: &Self::Matrix,
        _destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn centered_extend(
        &mut self,
        _value: &Self::Matrix,
        _destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn centered_extend_small(
        &mut self,
        _value: &Self::SmallMatrix,
        _destination: &ConcreteMatrixType,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn block_mod_switch(
        &mut self,
        _value: &Self::Matrix,
        _destination: &ConcreteMatrixType,
        _plaintext_modulus: u64,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn preimage_target(
        &mut self,
        value: Arc<Self::Matrix>,
    ) -> Result<(Arc<dyn PolyMatrixColumnSource<Self::Matrix>>, Arc<Vec<u8>>), Self::Error> {
        let rows = value.rows;
        let columns = value.columns;
        let first = value.shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
        let params = first.value.params().clone();
        let mut metadata = None;
        let mut bytes = Vec::new();
        let mut payload_start = 0;
        // Copy existing shards directly to host. Gathering on a
        // device would temporarily allocate the entire logical target there.
        // Two stages overlap transfers while bounding pinned/unpack scratch
        // to two shards. The next transfer starts before the previous wait.
        let mut chunks = value.shards.iter();
        let mut pending = chunks
            .next()
            .map(|shard| {
                self.start_rns_snapshot(&shard.value)
                    .map(|transfer| (shard.global_column_start, transfer))
            })
            .transpose()?;
        while let Some((global_column_start, transfer)) = pending {
            let next = chunks
                .next()
                .map(|shard| {
                    self.start_rns_snapshot(&shard.value)
                        .map(|transfer| (shard.global_column_start, transfer))
                })
                .transpose()?;
            let snapshot = transfer.finish();
            let current = (snapshot.level(), snapshot.is_ntt(), snapshot.bytes_per_poly());
            if let Some(expected) = metadata {
                if current != expected {
                    return Err(PolyBackendError::InvalidConstantShape);
                }
            } else {
                metadata = Some(current);
                let length = rows
                    .checked_mul(columns)
                    .and_then(|size| size.checked_mul(snapshot.bytes_per_poly()))
                    .ok_or(PolyBackendError::InvalidInteger)?;
                // The byte-slice encoding is its length followed by raw bytes.
                // Allocate the final staging representation once and write
                // shards directly into it, avoiding a second full-size copy.
                let header = bincode::encode_to_vec(
                    (1u8, rows, columns, current.0, current.1, current.2, length),
                    bincode::config::standard(),
                )
                .map_err(|_| PolyBackendError::InvalidInteger)?;
                payload_start = header.len();
                bytes = vec![0; payload_start + length];
                bytes[..payload_start].copy_from_slice(&header);
            }
            let row_bytes = columns * snapshot.bytes_per_poly();
            let shard_row_bytes = snapshot.ncol() * snapshot.bytes_per_poly();
            if row_bytes != 0 {
                bytes[payload_start..].par_chunks_mut(row_bytes).enumerate().for_each(
                    |(row, target)| {
                        let start = global_column_start * snapshot.bytes_per_poly();
                        target[start..start + shard_row_bytes]
                            .par_chunks_mut(snapshot.bytes_per_poly())
                            .enumerate()
                            .for_each(|(column, target)| {
                                let offset =
                                    row * shard_row_bytes + column * snapshot.bytes_per_poly();
                                target.copy_from_slice(
                                    &snapshot.bytes()[offset..offset + target.len()],
                                );
                            });
                    },
                );
            }
            self.rns_staging_buffers.push(snapshot);
            pending = next;
        }
        let bytes = Arc::new(bytes);
        let source = PreimageTarget::staged(&params, rows, columns, bytes.clone());
        Ok((Arc::new(source), bytes))
    }

    fn matrix_from_cpu_staging_bytes(
        &mut self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Matrix, Self::Error> {
        let (rows, columns) = {
            let parameters = self
                .devices
                .first()
                .ok_or(PolyBackendError::InvalidConstantShape)?
                .1
                .parameters(ty)?;
            let layout = GpuDCRTPolyMatrix::cpu_staging_layout(parameters, bytes)
                .map_err(PolyBackendError::GpuSubmission)?;
            (layout.rows, layout.columns)
        };
        let device_count = self.devices.len();
        let mut shards = Vec::with_capacity(device_count);
        for (index, (device_id, backend)) in self.devices.iter_mut().enumerate() {
            let parameters = backend.parameters(ty)?.clone();
            let start = columns.saturating_mul(index) / device_count;
            let end = columns.saturating_mul(index + 1) / device_count;
            let value = GpuDCRTPolyMatrix::from_cpu_staging_columns(&parameters, bytes, start, end);
            shards.push(GpuColumnShard {
                device_id: *device_id,
                global_column_start: start,
                value: Arc::new(value),
            });
        }
        Ok(GpuFleetMatrix::from_shared_shards(rows, columns, shards))
    }

    fn preimage_target_from_staging(
        &self,
        ty: &ConcreteMatrixType,
        rows: usize,
        columns: usize,
        bytes: Arc<Vec<u8>>,
    ) -> Result<Arc<dyn PolyMatrixColumnSource<Self::Matrix>>, Self::Error> {
        if rows != ty.rows || columns != ty.columns {
            return Err(PolyBackendError::InvalidConstantShape);
        }
        let params = self.devices[0].1.parameters(ty)?.clone();
        Ok(Arc::new(PreimageTarget::staged(&params, rows, columns, bytes)))
    }

    fn validate_preimage_bound(
        &self,
        ty: &ConcreteMatrixType,
        sigma: f64,
        gadget_base: &BigInt,
        digit_count: usize,
        max_coefficient_bound: &BigInt,
    ) -> Result<(), Self::Error> {
        self.devices[0].1.validate_preimage_bound(
            ty,
            sigma,
            gadget_base,
            digit_count,
            max_coefficient_bound,
        )
    }

    fn transpose(&mut self, _value: &Self::Matrix) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn slice(
        &mut self,
        _value: &Self::Matrix,
        _rows: Option<&IndexRange>,
        _columns: Option<&IndexRange>,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn sum_rows(
        &mut self,
        _value: &Self::Matrix,
        _rows: &[Vec<usize>],
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn tensor(
        &mut self,
        _left: &Self::Matrix,
        _right: &Self::Matrix,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn tensor_sum_rows(
        &mut self,
        _left: &Self::Matrix,
        _right: &Self::Matrix,
        _groups: &[Vec<usize>],
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn concat(
        &mut self,
        _inputs: &[&Self::Matrix],
        _axis: ConcatAxis,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn sample_uniform(
        &mut self,
        _ty: &ConcreteMatrixType,
        _range: &SampleRange,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn sample_gaussian(
        &mut self,
        _ty: &ConcreteMatrixType,
        _sigma: f64,
        _max_coefficient_bound: &BigInt,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn sample_hash(
        &mut self,
        _ty: &ConcreteMatrixType,
        _key: [u8; 32],
        _tag: &[u8],
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn sample_hash_decomposed(
        &mut self,
        _ty: &ConcreteMatrixType,
        _key: [u8; 32],
        _tag: &[u8],
        _gadget_base: &BigInt,
        _digit_count: usize,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn sample_hash_small_decomposed(
        &mut self,
        _ty: &ConcreteMatrixType,
        _key: [u8; 32],
        _tag: &[u8],
        _gadget_base: &BigInt,
        _digit_count: usize,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn sample_trapdoor(
        &mut self,
        _ty: &ConcreteMatrixType,
        _sigma: f64,
        _gadget_base: &BigInt,
        _digit_count: usize,
    ) -> Result<(Self::Matrix, Self::Trapdoor), Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn sample_preimage_batch(
        &mut self,
        _requests: Vec<crate::backend::PreimageRequest<Self::Matrix, Self::Trapdoor>>,
    ) -> Result<Vec<Self::SmallMatrix>, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn sample_preimage(
        &mut self,
        _ty: &ConcreteMatrixType,
        _sigma: f64,
        _gadget_base: &BigInt,
        _digit_count: usize,
        _max_coefficient_bound: &BigInt,
        _trapdoor: &Self::Trapdoor,
        _public: &Self::Matrix,
        _target: &dyn PolyMatrixColumnSource<Self::Matrix>,
        _randomness_seed: [u8; 32],
    ) -> Result<Self::SmallMatrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn validate_gadget_layout(
        &self,
        ty: &ConcreteMatrixType,
        gadget_base: &BigInt,
        digit_count: usize,
        small: bool,
    ) -> Result<(), Self::Error> {
        self.devices[0].1.validate_gadget_layout(ty, gadget_base, digit_count, small)
    }

    fn gadget_decompose(
        &mut self,
        _value: &Self::Matrix,
        _small: bool,
        _digit_count: Option<usize>,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn gadget_decompose_row_blocks(
        &mut self,
        _blocks: &[&Self::Matrix],
        _small: bool,
        _digit_count: Option<usize>,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn gadget_error_bound(
        &self,
        ty: &ConcreteMatrixType,
        digit_count: Option<usize>,
    ) -> Result<BigInt, Self::Error> {
        self.devices[0].1.gadget_error_bound(ty, digit_count)
    }

    fn multiply_small_rhs(
        &mut self,
        _lhs: &Self::Matrix,
        _rhs: &Self::SmallMatrix,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn multiply_small_rhs_row_blocks(
        &mut self,
        _blocks: &[&Self::Matrix],
        _rhs: &Self::SmallMatrix,
    ) -> Result<Vec<Self::Matrix>, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn extract_coefficient(
        &mut self,
        _value: &Self::Matrix,
        _position: usize,
    ) -> Result<BigInt, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn threshold_decode(
        &mut self,
        _value: &Self::Matrix,
        _plaintext_modulus: &BigInt,
        _length: usize,
    ) -> Result<Vec<BigInt>, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn pack_polynomial_coefficients(
        &mut self,
        _ty: &ConcreteMatrixType,
        _bits: &[bool],
        _coefficient_bits: usize,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn crt_recompose(
        &mut self,
        _levels: &[Self::Matrix],
        _plaintext_moduli: &[BigInt],
        _reconstruction_coefficients: &[BigInt],
        _destination: &ConcreteMatrixType,
    ) -> Result<Self::Matrix, Self::Error> {
        Err(PolyBackendError::PreparedExecutionRequired)
    }

    fn matrix_to_bytes(&self, value: &Self::Matrix) -> Result<Vec<u8>, Self::Error> {
        if let Some(lease) = &value.prepared_lease {
            lease.check_device_scalar_status().map_err(PolyBackendError::GpuSubmission)?;
        }
        let shard_bytes = value
            .shards
            .iter()
            .map(|shard| {
                let (_, device) = self
                    .devices
                    .iter()
                    .find(|(id, _)| *id == shard.device_id)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                device.matrix_to_bytes(&shard.value)
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Construction guarantees that a lone shard covers the whole matrix.
        // Its canonical encoding already has the global shape and bit width.
        if let [bytes] = shard_bytes.as_slice() {
            return Ok(bytes.clone());
        }
        let decoded = shard_bytes
            .iter()
            .map(|bytes| decode_compact_matrix(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let first = decoded.first().expect("nonempty matrix has a shard");
        let bytes_per_coefficient = decoded.iter().map(|encoding| encoding.6).max().unwrap_or(0);
        let max_coefficient_bits = decoded.iter().map(|encoding| encoding.5).max().unwrap_or(0);
        assert!(
            decoded.iter().all(|encoding| {
                encoding.0 == first.0 &&
                    encoding.1 == first.1 &&
                    encoding.2 == first.2 &&
                    encoding.3 == value.rows
            }),
            "fleet shards disagree on compact matrix state"
        );
        let ring_dimension = value.shards[0].value.params().ring_dimension() as usize;
        let global_bits = usize::from(max_coefficient_bits);
        let global_count = value.rows * value.columns * ring_dimension;
        let mut payload = vec![0u8; (global_count * global_bits).div_ceil(8)];
        for (shard, encoding) in value.shards.iter().zip(&decoded) {
            let local_bits = usize::from(encoding.5);
            for row in 0..value.rows {
                // Equal-width shards are contiguous rows of packed coefficients.
                // Copy the entire row, without visiting every coefficient or bit.
                if local_bits == global_bits {
                    let row_bits = shard.value.col_size() * ring_dimension * local_bits;
                    copy_packed_bits(
                        &encoding.7,
                        row * row_bits,
                        &mut payload,
                        (row * value.columns + shard.global_column_start) *
                            ring_dimension *
                            global_bits,
                        row_bits,
                    );
                    continue;
                }
                for column in 0..shard.value.col_size() {
                    for coefficient in 0..ring_dimension {
                        let source_index = ((row * shard.value.col_size() + column) *
                            ring_dimension +
                            coefficient) *
                            local_bits;
                        let target_column = shard.global_column_start + column;
                        let target_index = ((row * value.columns + target_column) * ring_dimension +
                            coefficient) *
                            global_bits;
                        copy_packed_bits(
                            &encoding.7,
                            source_index,
                            &mut payload,
                            target_index,
                            local_bits,
                        );
                    }
                }
            }
        }
        Ok(bincode::encode_to_vec(
            (
                first.0,
                first.1,
                first.2,
                value.rows,
                value.columns,
                max_coefficient_bits,
                bytes_per_coefficient,
                payload,
            ),
            bincode::config::standard(),
        )
        .expect("fleet matrix serialization"))
    }

    fn matrix_from_bytes(
        &mut self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Matrix, Self::Error> {
        self.devices[0].1.matrix_from_bytes(ty, bytes).map(GpuFleetMatrix::from_matrix)
    }

    fn small_matrix_to_bytes(
        &self,
        value: &Self::SmallMatrix,
        expected_schema: &ConcreteBoundedMatrixSchema,
        semantic_kind: SmallMatrixSemanticKind,
    ) -> Result<Vec<u8>, Self::Error> {
        if value.size() != (expected_schema.matrix.rows, expected_schema.matrix.columns) {
            return Err(PolyBackendError::InvalidSmallMatrixArtifact("fleet shape mismatch"));
        }
        let ring_dimension = expected_schema.matrix.ring_dimension;
        let bound = expected_schema
            .max_coefficient_bound
            .to_biguint()
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("bound is negative"))?;
        let expected_magnitude_width = usize::try_from(bound.bits().div_ceil(8))
            .map_err(|_| PolyBackendError::InvalidSmallMatrixArtifact("bound width overflows"))?
            .max(1);
        let coefficient_width = expected_magnitude_width
            .checked_add(1)
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("coefficient width overflows"))?;
        let mut shard_payloads = Vec::with_capacity(value.shards.len());
        for shard in value.shards.iter() {
            let device =
                self.devices.iter().position(|(device, _)| *device == shard.device_id).ok_or(
                    PolyBackendError::InvalidSmallMatrixArtifact("shard device is not registered"),
                )?;
            let local_type = ConcreteMatrixType {
                columns: shard.value.columns(),
                ..expected_schema.matrix.clone()
            };
            let params = self.devices[device].1.parameters(&local_type)?;
            shard.value.validate_metadata(params, value.rows, shard.value.columns(), &bound)?;
            if shard.value.magnitude_width() != expected_magnitude_width {
                return Err(PolyBackendError::InvalidSmallMatrixArtifact(
                    "shards disagree on encoded coefficient width",
                ));
            }
            let local = shard.value.to_canonical_coefficients()?;
            let local_count = value
                .rows
                .checked_mul(shard.value.columns())
                .and_then(|count| count.checked_mul(ring_dimension))
                .ok_or(PolyBackendError::InvalidSmallMatrixArtifact(
                    "coefficient count overflows",
                ))?;
            let expected_length = local_count
                .checked_mul(coefficient_width)
                .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("payload length overflows"))?;
            if local.len() != expected_length {
                return Err(PolyBackendError::InvalidSmallMatrixArtifact(
                    "owner returned a payload with the wrong length",
                ));
            }
            shard_payloads.push(local);
        }
        let coefficient_count = value
            .rows
            .checked_mul(value.columns)
            .and_then(|count| count.checked_mul(ring_dimension))
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("coefficient count overflows"))?;
        let payload_length = coefficient_count
            .checked_mul(coefficient_width)
            .ok_or(PolyBackendError::InvalidSmallMatrixArtifact("payload length overflows"))?;
        let mut payload = vec![0u8; payload_length];
        for (shard, local) in value.shards.iter().zip(shard_payloads) {
            for row in 0..value.rows {
                let local_row_bytes = shard.value.columns() * ring_dimension * coefficient_width;
                let source_start = row * local_row_bytes;
                let target_start = (row * value.columns + shard.global_column_start) *
                    ring_dimension *
                    coefficient_width;
                payload[target_start..target_start + local_row_bytes]
                    .copy_from_slice(&local[source_start..source_start + local_row_bytes]);
            }
        }
        encode_small_matrix_artifact(expected_schema, &payload, semantic_kind)
    }

    fn small_matrix_from_bytes(
        &mut self,
        expected_schema: &ConcreteBoundedMatrixSchema,
        bytes: &[u8],
        expected_semantic_kind: SmallMatrixSemanticKind,
    ) -> Result<Self::SmallMatrix, Self::Error> {
        self.devices[0]
            .1
            .small_matrix_from_bytes(expected_schema, bytes, expected_semantic_kind)
            .map(GpuFleetSmallMatrix::from_matrix)
    }

    fn trapdoor_to_bytes(&self, value: &Self::Trapdoor) -> Vec<u8> {
        self.devices[0]
            .1
            .trapdoor_to_bytes(value.values.first().expect("fleet trapdoor is nonempty"))
    }

    fn trapdoor_from_bytes(
        &mut self,
        ty: &ConcreteMatrixType,
        bytes: &[u8],
    ) -> Result<Self::Trapdoor, Self::Error> {
        let values = self
            .devices
            .iter_mut()
            .map(|(_, backend)| backend.trapdoor_from_bytes(ty, bytes).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(GpuFleetTrapdoor { values: Arc::new(values), prepared_lease: None })
    }
}
