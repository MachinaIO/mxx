//! Typed prepared values and fixed input normalization in reserved native storage.

use super::*;
use crate::gpu_memory::{GpuMemoryLedger, GpuPreparedAllocationRequirement};
use mxx_primitives::matrix::gpu_dcrt_poly::{
    GpuPreparedRequest, GpuPreparedSlotKind, GpuPreparedStorage,
};

pub(super) enum PreparedMatrixValue {
    Matrix(GpuDCRTPolyMatrix),
    Compact(GpuSmallMatrix),
}

pub(super) trait PreparedFleetOutput: Sized {
    const COMPACT: bool;
    fn from_prepared(
        rows: usize,
        columns: usize,
        shards: Vec<GpuColumnShard<PreparedMatrixValue>>,
    ) -> Result<Self, PolyBackendError>;
}
impl PreparedFleetOutput for GpuFleetMatrix {
    const COMPACT: bool = false;
    fn from_prepared(
        rows: usize,
        columns: usize,
        shards: Vec<GpuColumnShard<PreparedMatrixValue>>,
    ) -> Result<Self, PolyBackendError> {
        let shards = shards
            .into_iter()
            .map(|s| match s.value {
                PreparedMatrixValue::Matrix(value) => Ok(GpuColumnShard {
                    value,
                    device_id: s.device_id,
                    global_column_start: s.global_column_start,
                }),
                _ => Err(PolyBackendError::InvalidConstantShape),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(rows, columns, shards))
    }
}
impl PreparedFleetOutput for GpuFleetSmallMatrix {
    const COMPACT: bool = true;
    fn from_prepared(
        rows: usize,
        columns: usize,
        shards: Vec<GpuColumnShard<PreparedMatrixValue>>,
    ) -> Result<Self, PolyBackendError> {
        let shards = shards
            .into_iter()
            .map(|s| match s.value {
                PreparedMatrixValue::Compact(value) => Ok(GpuColumnShard {
                    value,
                    device_id: s.device_id,
                    global_column_start: s.global_column_start,
                }),
                _ => Err(PolyBackendError::InvalidConstantShape),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(rows, columns, shards))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, serde::Serialize)]
pub enum PreparedMatrixSource {
    Shard(usize),
    Replica { device: usize, context: usize, evaluation: bool },
    Fragment { device: usize, context: usize, index: usize, evaluation: bool },
}

/// CPU-only column-owner layout, observed on an input or derived for an output.
/// It carries no native pointer or lease. Derived layouts describe coverage and
/// format only; they do not establish an actual matrix identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixInputFragment {
    pub device: i32,
    pub context: usize,
    pub start: usize,
    pub end: usize,
    pub level: usize,
    pub evaluation: bool,
}

impl PreparedMatrixSource {
    /// Shared source choice for metadata admission and concrete preparation.
    /// None means a replica is required; Fragment means a format conversion.
    pub(super) fn existing(
        fragments: impl IntoIterator<Item = MatrixInputFragment>,
        columns: std::ops::Range<usize>,
        device: usize,
        parameters: &GpuDCRTPolyParams,
        evaluation: bool,
    ) -> Option<Self> {
        fragments.into_iter().enumerate().find_map(|(index, fragment)| {
            (fragment.device == parameters.device_ids()[0] &&
                fragment.context == parameters.context_identity() &&
                fragment.start <= columns.start &&
                columns.end <= fragment.end)
                .then_some(if fragment.evaluation == evaluation {
                    Self::Shard(index)
                } else {
                    Self::Fragment {
                        device,
                        context: parameters.context_identity(),
                        index,
                        evaluation,
                    }
                })
        })
    }

    /// Lower source selection and every required copy/normalization from
    /// layout metadata alone. Order is significant: mixed-format fragments
    /// are normalized before the replica that consumes them.
    pub fn plan(
        fragments: impl Iterator<Item = MatrixInputFragment> + Clone,
        shape: (usize, usize),
        columns: std::ops::Range<usize>,
        device: usize,
        parameters: &GpuDCRTPolyParams,
        evaluation: bool,
    ) -> (Self, Vec<MatrixInputLayout>) {
        let source =
            Self::existing(fragments.clone(), columns, device, parameters, evaluation).unwrap_or(
                Self::Replica { device, context: parameters.context_identity(), evaluation },
            );
        if matches!(source, Self::Shard(_)) {
            return (source, Vec::new());
        }
        let first = fragments.clone().next().expect("nonempty matrix layout");
        let mixed = fragments.clone().any(|fragment| fragment.evaluation != first.evaluation);
        let mut preparation = Vec::new();
        if let Self::Replica { device, context, evaluation } = source &&
            mixed
        {
            for (index, fragment) in fragments.clone().enumerate() {
                if fragment.evaluation != evaluation {
                    preparation.push(MatrixInputLayout {
                        source: Self::Fragment { device, context, index, evaluation },
                        shape: (shape.0, fragment.end - fragment.start),
                        level: fragment.level,
                        evaluation: fragment.evaluation,
                    });
                }
            }
        }
        preparation.push(match source {
            Self::Fragment { index, .. } => {
                let fragment = fragments.clone().nth(index).expect("selected fragment");
                MatrixInputLayout {
                    source,
                    shape: (shape.0, fragment.end - fragment.start),
                    level: fragment.level,
                    evaluation: fragment.evaluation,
                }
            }
            Self::Replica { .. } => MatrixInputLayout {
                source,
                shape,
                level: first.level,
                evaluation: if mixed { evaluation } else { first.evaluation },
            },
            Self::Shard(_) => unreachable!("borrowed source needs no preparation"),
        });
        (source, preparation)
    }
}

/// A fully determined input allocation and its initial format, before its
/// required normalization. Both inventory and real preparation use this layout.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MatrixInputLayout {
    pub source: PreparedMatrixSource,
    pub shape: (usize, usize),
    pub level: usize,
    pub evaluation: bool,
}

impl MatrixInputLayout {
    /// Assign the complete ordered preparation transaction without reserving or
    /// executing anything. Callers supply only preparations absent for this owner,
    /// with their actual source levels. Failure leaves the supplied inventory intact.
    pub fn assign(
        layouts: impl IntoIterator<Item = Self>,
        parameters: &GpuDCRTPolyParams,
        slots: &[(mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotSnapshot, bool)],
    ) -> Result<Option<Vec<(Self, GpuPreparedRequest)>>, String> {
        use mxx_primitives::matrix::gpu_dcrt_poly::{GpuPreparedSlotSnapshot, GpuTracedClaim};
        let mut layouts = layouts.into_iter().peekable();
        if layouts.peek().is_none() {
            return Ok(Some(Vec::new()));
        }
        let mut slots = slots.to_vec();
        let mut selected = Vec::new();
        // Normalize fragments before selecting the replica that consumes them.
        // Sequential choice preserves native size-class preference and prevents
        // two preparations from claiming the same backing slot.
        for layout in layouts {
            let claim = GpuTracedClaim::matrix(
                layout.shape.0,
                layout.shape.1,
                layout.level,
                layout.evaluation,
            );
            let Some(request) = GpuPreparedSlotSnapshot::assign(parameters, &slots, &[claim])?
                .into_iter()
                .next()
                .flatten()
            else {
                return Ok(None);
            };
            for (slot, eligible) in &mut slots {
                let id = slot.identity();
                *eligible &= (id.storage_id(), id.slot_id(), id.slot_index()) != request.slot_key();
            }
            selected.push((layout, request));
        }
        Ok(Some(selected))
    }
}

/// Observed immutable input layouts retained by a root/wave admission.
/// Keys are actual fleet owner IDs, never shape-based aliases.
pub(super) type MatrixInputLayouts = HashMap<u64, Arc<[MatrixInputFragment]>>;
pub(super) type SymbolicMatrixLayouts =
    std::collections::BTreeMap<mxx_ir_core::types::WireRef, Arc<[MatrixInputFragment]>>;

pub(super) type PreparedMatrixInputs =
    HashMap<(u64, PreparedMatrixSource), Arc<GpuColumnShard<GpuDCRTPolyMatrix>>>;

pub(super) struct MatrixInputPreparation {
    pub matrix: GpuFleetMatrix,
    pub layout: MatrixInputLayout,
    pub parameters: GpuDCRTPolyParams,
    pub device: usize,
    pub storage: Arc<GpuPreparedStorage>,
    pub request: GpuPreparedRequest,
}

pub(super) use mxx_primitives::matrix::gpu_dcrt_poly::{capacity_class, matrix_capacity_class};

/// Immutable capacity view for one registered native parameter context.
/// It holds no backing storage or reservation. Eligibility is the observed or
/// hypothetical scenario; native commit independently rechecks every request.
pub struct MatrixSlotContext {
    pub device: usize,
    pub context: usize,
    pub slots: Vec<(mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotSnapshot, bool)>,
}

pub type MatrixSlotInventory = Vec<MatrixSlotContext>;
// Hypothetical slot IDs are local to a parameter context; native IDs must not
// accidentally make the same planner depend on process-global uniqueness.
pub(super) type MatrixSlotKey = (usize, (u64, u64, usize));

/// Input geometry and parameter handles, with no GPU payload or backing owner.
/// Native admission obtains this from its real inputs; hypothetical admission
/// supplies the same metadata from an explicit placement scenario.
#[derive(Clone)]
pub struct MatrixDescriptor {
    pub id: u64,
    pub rows: usize,
    pub columns: usize,
    pub shards: Vec<GpuColumnShard<MatrixFragmentDescriptor>>,
    pub input_layout: Arc<[MatrixInputFragment]>,
}

#[derive(Clone)]
pub struct MatrixFragmentDescriptor {
    pub parameters: GpuDCRTPolyParams,
    pub level: usize,
    pub columns: usize,
    pub evaluation: bool,
}

impl MatrixFragmentDescriptor {
    pub fn params(&self) -> &GpuDCRTPolyParams {
        &self.parameters
    }
    pub fn level(&self) -> usize {
        self.level
    }
    pub fn is_ntt(&self) -> bool {
        self.evaluation
    }
    pub fn columns_count(&self) -> usize {
        self.columns
    }

    pub fn registered_parameters<'a>(
        &self,
        backend: &'a DeviceBackend,
    ) -> Result<&'a GpuDCRTPolyParams, PolyBackendError> {
        let key = crate::backend::poly::RingKey {
            modulus: BigInt::from(self.parameters.modulus().as_ref().clone()),
            ring_dimension: self.parameters.ring_dimension() as usize,
        };
        backend.parameters[backend.active_placement]
            .get(&key)
            .ok_or(PolyBackendError::MissingParameters(key))
    }
}

impl MatrixDescriptor {
    pub fn size(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }
}

impl super::gpu_compiled::MatrixShape for MatrixDescriptor {
    fn shape(&self) -> (usize, usize) {
        self.size()
    }
}

impl From<&GpuFleetMatrix> for MatrixDescriptor {
    fn from(matrix: &GpuFleetMatrix) -> Self {
        Self {
            id: matrix.id,
            rows: matrix.rows,
            columns: matrix.columns,
            input_layout: matrix.input_layout.clone(),
            shards: matrix
                .shards
                .iter()
                .map(|shard| GpuColumnShard {
                    device_id: shard.device_id,
                    global_column_start: shard.global_column_start,
                    value: MatrixFragmentDescriptor {
                        parameters: shard.value.params().clone(),
                        level: shard.value.level(),
                        columns: shard.value.col_size(),
                        evaluation: shard.value.is_ntt(),
                    },
                })
                .collect(),
        }
    }
}

impl From<&GpuFleetSmallMatrix> for MatrixDescriptor {
    fn from(matrix: &GpuFleetSmallMatrix) -> Self {
        let shards = matrix
            .shards
            .iter()
            .map(|shard| GpuColumnShard {
                device_id: shard.device_id,
                global_column_start: shard.global_column_start,
                value: MatrixFragmentDescriptor {
                    parameters: shard.value.params().clone(),
                    level: shard.value.params().crt_depth() - 1,
                    columns: shard.value.columns_count(),
                    evaluation: false,
                },
            })
            .collect::<Vec<_>>();
        let input_layout = shards
            .iter()
            .map(|shard| MatrixInputFragment {
                device: shard.device_id,
                context: shard.value.parameters.context_identity(),
                start: shard.global_column_start,
                end: shard.global_column_start + shard.value.columns,
                level: shard.value.level,
                evaluation: false,
            })
            .collect();
        Self { id: matrix.id, rows: matrix.rows, columns: matrix.columns, shards, input_layout }
    }
}

/// One selected copy/normalization, before native values or storage are bound.
pub struct MatrixInputRequest {
    pub owner: u64,
    pub layout: MatrixInputLayout,
    pub parameters: GpuDCRTPolyParams,
    pub device: usize,
    pub request: GpuPreparedRequest,
}

pub(super) fn select_prepared_matrix(
    inventory: &MatrixSlotInventory,
    chosen: &mut HashSet<MatrixSlotKey>,
    device: usize,
    parameters: &GpuDCRTPolyParams,
    level: usize,
    shape: (usize, usize),
    evaluation: bool,
    compact_bound: Option<&num_bigint::BigUint>,
) -> Result<Option<GpuPreparedRequest>, PolyBackendError> {
    use mxx_primitives::matrix::gpu_dcrt_poly::{
        GpuPreparedSlotSnapshot, GpuPreparedWorkspaceLayout, GpuTracedClaim,
    };
    let claim = if let Some(bound) = compact_bound {
        GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
            kind: GpuPreparedSlotKind::CompactPayload,
            bytes: GpuSmallMatrix::allocation_bytes(parameters, shape.0, shape.1, bound)
                .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?,
            alignment: 256,
        })
    } else {
        GpuTracedClaim::matrix(shape.0, shape.1, level, evaluation)
    };
    // Native and hypothetical selection consume the same typed layout matcher
    // and size-class preferences. Snapshot eligibility includes region ownership
    // and pending readers; commit still rechecks every selected native request.
    let slots = inventory
        .par_iter()
        .filter(|entry| entry.device == device && entry.context == parameters.context_identity())
        .flat_map_iter(|entry| {
            entry.slots.iter().map(|&(slot, available)| {
                let eligible = available &&
                    !chosen.contains(&(parameters.context_identity(), {
                        let id = slot.identity();
                        (id.storage_id(), id.slot_id(), id.slot_index())
                    }));
                (slot, eligible)
            })
        })
        .collect::<Vec<_>>();
    let selected = GpuPreparedSlotSnapshot::assign(parameters, &slots, &[claim])
        .map_err(PolyBackendError::GpuSubmission)?
        .into_iter()
        .next()
        .flatten();
    if let Some(request) = selected {
        chosen.insert((parameters.context_identity(), request.slot_key()));
    }
    Ok(selected)
}

/// Select one immutable input range, with a shared full replica when existing
/// owners cannot serve it on the required native parameter context.
pub(super) fn select_matrix_input(
    matrix: &MatrixDescriptor,
    columns: std::ops::Range<usize>,
    device: usize,
    parameters: &GpuDCRTPolyParams,
    evaluation: bool,
    inventory: &MatrixSlotInventory,
    chosen: &mut HashSet<MatrixSlotKey>,
    planned: &mut HashSet<(u64, PreparedMatrixSource)>,
    preparation: &mut Vec<MatrixInputRequest>,
    admitted_inputs: &MatrixInputLayouts,
) -> Result<Option<PreparedMatrixSource>, PolyBackendError> {
    let first = matrix.shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
    let fragments = admitted_inputs.get(&matrix.id).unwrap_or(&matrix.input_layout);
    let (source, layouts) = PreparedMatrixSource::plan(
        fragments.iter().copied(),
        matrix.size(),
        columns,
        device,
        parameters,
        evaluation,
    );
    if matches!(source, PreparedMatrixSource::Replica { .. }) &&
        matrix.shards.iter().any(|shard| {
            shard.value.level() != first.value.level() ||
                shard.value.params().ring_dimension() != parameters.ring_dimension() ||
                shard.value.params().moduli() != parameters.moduli()
        })
    {
        return Err(PolyBackendError::GpuSubmission(
            "prepared replica has incompatible source parameters".into(),
        ));
    }
    if layouts.iter().all(|layout| planned.contains(&(matrix.id, layout.source))) {
        return Ok(Some(source));
    }
    let slots = inventory
        .par_iter()
        .filter(|entry| entry.device == device && entry.context == parameters.context_identity())
        .flat_map_iter(|entry| {
            entry.slots.iter().map(|&(slot, available)| {
                (
                    slot,
                    available &&
                        !chosen.contains(&(parameters.context_identity(), {
                            let id = slot.identity();
                            (id.storage_id(), id.slot_id(), id.slot_index())
                        })),
                )
            })
        })
        .collect::<Vec<_>>();
    let layouts =
        layouts.into_iter().filter(|layout| !planned.contains(&(matrix.id, layout.source)));
    let Some(selected) = MatrixInputLayout::assign(layouts, parameters, &slots)
        .map_err(PolyBackendError::GpuSubmission)?
    else {
        return Ok(None);
    };
    // Publish metadata only after the entire input preparation fits. Native
    // reservation later rechecks all requests before any copy or conversion.
    for (layout, request) in selected {
        chosen.insert((parameters.context_identity(), request.slot_key()));
        planned.insert((matrix.id, layout.source));
        preparation.push(MatrixInputRequest {
            owner: matrix.id,
            layout,
            parameters: parameters.clone(),
            device,
            request,
        });
    }
    Ok(Some(source))
}

impl GpuDcrtBackend {
    pub(super) fn prepare_matrix_inputs(
        &mut self,
        ledger: &mut GpuMemoryLedger,
        preparation: Vec<MatrixInputPreparation>,
    ) -> Result<Arc<PreparedMatrixInputs>, PolyBackendError> {
        if preparation.is_empty() {
            return Ok(Arc::new(HashMap::new()));
        }
        let requirements = preparation
            .iter()
            .map(|input| GpuPreparedAllocationRequirement {
                device: input.device,
                storage: &input.storage,
                requests: std::slice::from_ref(&input.request),
            })
            .collect::<Vec<_>>();
        let reservation = ledger
            .reserve(&[], &requirements)
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        debug_assert!(reservation.allocations.is_empty());
        let mut groups = (0..self.devices.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        for (index, (device, mut reservation)) in reservation.prepared.into_iter().enumerate() {
            reservation.require_all_resources().map_err(PolyBackendError::GpuSubmission)?;
            groups[device].push((index, reservation));
        }
        // Owned, non-Sync reservation tokens move with each device state. The
        // existing enqueue workers retain their concurrency and error recovery.
        let mut workers =
            std::mem::take(&mut self.devices).into_iter().zip(groups).collect::<Vec<_>>();
        let preparation = Arc::new(preparation);
        let mut prepared = Arc::new(PreparedMatrixInputs::new());
        // Input-normalization dependencies are submitted before their replicas.
        // These are host submission phases, not GPU completion barriers: copy
        // readers retain the existing native producer/release event edges.
        for replicas in [false, true] {
            let mut pending = Vec::with_capacity(workers.len());
            let mut phase = std::mem::take(&mut workers)
                .into_iter()
                .map(|(state, claims)| {
                    let (selected, later) =
                        claims.into_iter().partition::<Vec<_>, _>(|(index, _)| {
                            matches!(
                                preparation[*index].layout.source,
                                PreparedMatrixSource::Replica { .. }
                            ) == replicas
                        });
                    pending.push(later);
                    (state, selected)
                })
                .collect::<Vec<_>>();
            if phase.iter().all(|(_, claims)| claims.is_empty()) {
                workers = phase
                    .into_iter()
                    .zip(pending)
                    .map(|((state, _), claims)| (state, claims))
                    .collect();
                continue;
            }
            let inputs = prepared.clone();
            let preparation = preparation.clone();
            let result = self
                .enqueue
                .map(&mut phase, move |_, (_, claims)| {
                    let mut outputs = Vec::with_capacity(claims.len());
                    for (index, reservation) in std::mem::take(claims) {
                        let input = &preparation[index];
                        let MatrixInputLayout { shape, evaluation, .. } = input.layout;
                        let source_shards = match input.layout.source {
                            PreparedMatrixSource::Shard(index) |
                            PreparedMatrixSource::Fragment { index, .. } => {
                                &input.matrix.shards[index..index + 1]
                            }
                            PreparedMatrixSource::Replica { .. } => input.matrix.shards.as_slice(),
                        };
                        let start = match input.layout.source {
                            PreparedMatrixSource::Replica { .. } => 0,
                            _ => source_shards[0].global_column_start,
                        };
                        let dispatch = reservation
                            .enter(Vec::new())
                            .map_err(PolyBackendError::GpuSubmission)?;
                        let mut output = GpuDCRTPolyMatrix::new_empty_with_state(
                            &input.parameters,
                            shape.0,
                            shape.1,
                            source_shards[0].value.level(),
                            evaluation,
                            None,
                        );
                        // One destination owns these disjoint copies. It may depend
                        // on multiple prior streams without blocking this worker.
                        for (index, original) in source_shards.iter().enumerate() {
                            let source =
                                if let PreparedMatrixSource::Replica { device, context, .. } =
                                    input.layout.source
                                {
                                    if original.value.is_ntt() != evaluation {
                                        inputs
                                            .get(&(
                                                input.matrix.id,
                                                PreparedMatrixSource::Fragment {
                                                    device,
                                                    context,
                                                    index,
                                                    evaluation,
                                                },
                                            ))
                                            .ok_or_else(|| {
                                                PolyBackendError::GpuSubmission(
                                    "mixed replica is missing its reserved normalization".into())
                                            })?
                                            .as_ref()
                                    } else {
                                        original
                                    }
                                } else {
                                    original
                                };
                            let offset = source.global_column_start - start;
                            output = source
                                .value
                                .column_view(0..source.value.col_size())
                                .map_err(PolyBackendError::GpuSubmission)?
                                .copy(Some((
                                    output,
                                    0..shape.0,
                                    offset..offset + source.value.col_size(),
                                )))
                                .map_err(PolyBackendError::GpuSubmission)?;
                        }
                        let desired_evaluation = match input.layout.source {
                            PreparedMatrixSource::Replica { evaluation, .. } |
                            PreparedMatrixSource::Fragment { evaluation, .. } => evaluation,
                            PreparedMatrixSource::Shard(_) => true,
                        };
                        if desired_evaluation {
                            output.ntt_all_in_place();
                        } else {
                            output.intt_all_in_place();
                        }
                        drop(dispatch.finish().map_err(PolyBackendError::GpuSubmission)?);
                        outputs.push((
                            (input.matrix.id, input.layout.source),
                            Arc::new(GpuColumnShard {
                                device_id: input.parameters.device_ids()[0],
                                global_column_start: start,
                                value: output,
                            }),
                        ));
                    }
                    Ok::<_, PolyBackendError>(outputs)
                })
                .map_err(PolyBackendError::from);
            workers = phase
                .into_iter()
                .zip(pending)
                .map(|((state, _), claims)| (state, claims))
                .collect();
            match result {
                Ok(outputs) => Arc::make_mut(&mut prepared).extend(outputs.into_iter().flatten()),
                Err(error) => {
                    self.devices = workers.into_iter().map(|(state, _)| state).collect();
                    return Err(error);
                }
            }
        }
        self.devices = workers.into_iter().map(|(state, _)| state).collect();
        Ok(prepared)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_primitives::matrix::PolyMatrix;
    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_preflight_normalizes_mixed_fragments_before_shared_replication() {
        use crate::{
            backend::{
                Backend,
                poly_gpu::{GpuFleetMatrix, gpu_backend_on},
            },
            gpu_invocation::GpuInvocation,
        };
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{GpuPreparedSlotKind, GpuPreparedWorkspaceLayout},
            poly::dcrt::{gpu::detected_gpu_device_ids, params::DCRTPolyParams},
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|v| v.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|v| v.parse::<usize>().unwrap())
            .unwrap_or(5)
            .max(2);
        let rows = 3;
        let cpu = DCRTPolyParams::new(n, 2, 54, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let mut backend = gpu_backend_on([params.clone()], [device]);
        let original = DCRTPolyUniformSampler::new().sample_uniform(
            &cpu,
            rows,
            columns,
            DistType::FinRingDist,
        );
        let source_params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            Some(&params),
            None,
        );
        let input = GpuFleetMatrix::new(
            rows,
            columns,
            (0..columns)
                .into_par_iter()
                .map(|column| {
                    let mut value = GpuDCRTPolyMatrix::from_cpu_matrix(
                        &source_params,
                        &original.slice_columns(column, column + 1),
                    );
                    if column % 2 == 0 {
                        value.intt_all_in_place();
                    }
                    GpuColumnShard { device_id: device, global_column_start: column, value }
                })
                .collect(),
        );
        assert_eq!(input.shards().len(), columns);
        let tensor_groups = vec![vec![5, 0, 5], vec![2]];
        let other = DCRTPolyUniformSampler::new().sample_uniform(&cpu, 2, 2, DistType::FinRingDist);
        let mut right = GpuDCRTPolyMatrix::from_cpu_matrix(&params, &other);
        right.intt_all_in_place();
        let right = GpuFleetMatrix::from_matrix(right);
        let storage = Arc::new(
            GpuPreparedStorage::new(
                None,
                [
                    (columns, rows),
                    (rows, columns),
                    (rows, columns),
                    (2, 2),
                    (6, 2 * columns),
                    (2, 2 * columns),
                ]
                .into_par_iter()
                .chain((0..columns).into_par_iter().map(|_| (rows, 1)))
                .map(|(r, c)| GpuDCRTPolyMatrix::zero(&params, r, c))
                .collect(),
                None,
                None,
            )
            .unwrap(),
        );
        let shapes = [(columns, rows), (6, 2 * columns), (2, 2 * columns)];
        let readbacks = shapes
            .into_iter()
            .map(|(rows, columns)| {
                // All CRT limbs share one readback batch and completion event.
                let events = 1;
                let mut layouts = vec![
                    params.rns_transfer_workspace(params.crt_depth() - 1, rows, columns).unwrap(),
                ];
                layouts.extend(std::iter::repeat_n(
                    GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::CompletionEvent,
                        bytes: 0,
                        alignment: 1,
                    },
                    events,
                ));
                Arc::new(
                    GpuPreparedStorage::new(
                        None,
                        vec![GpuDCRTPolyMatrix::zero(&params, rows, columns)],
                        None,
                        Some(&layouts),
                    )
                    .unwrap(),
                )
            })
            .collect::<Vec<_>>();
        backend
            .prepare_memory(
                std::iter::once((0, storage.clone()))
                    .chain(readbacks.iter().map(|s| (0, s.clone())))
                    .collect(),
                true,
            )
            .unwrap();
        backend.select_gpu_operation([124; 32]).unwrap();
        let requests = [
            (0, None, GpuInvocation::Transpose { value: &input }),
            (0, None, GpuInvocation::Tensor { left: &input, right: &right }),
            (
                0,
                None,
                GpuInvocation::TensorSumRows { left: &input, right: &right, rows: &tensor_groups },
            ),
        ];
        let tensor = original.tensor(&other);
        let expected = [original.transpose(), tensor.clone(), tensor.sum_rows(&tensor_groups)];
        // A cached invocation must reuse native backing without a host release fence.
        for _ in 0..2 {
            backend.preflight_gpu_operations(&requests).unwrap();
            assert!(backend.tensor(&input, &right).is_err());
            let outputs = [
                backend.transpose(&input).unwrap(),
                backend.tensor(&input, &right).unwrap(),
                backend.tensor_sum_rows(&input, &right, &tensor_groups).unwrap(),
            ];
            for ((output, expected), readback) in outputs.iter().zip(&expected).zip(&readbacks) {
                let claims = (1..readback.slot_count())
                    .map(|i| {
                        let slot = readback.slot_identity(i).unwrap();
                        slot.workspace_request(slot.requested_backing_bytes(), slot.alignment())
                    })
                    .collect::<Vec<_>>();
                let mut claims = claims;
                if !output.shards()[0].value.is_ntt() {
                    let (rows, columns) = output.size();
                    claims.insert(
                        0,
                        readback.slot_identity(0).unwrap().matrix_request(rows, columns, false),
                    );
                }
                let dispatch = readback.reserve(&claims).unwrap().enter(Vec::new()).unwrap();
                assert_eq!(&output.shards()[0].value.to_cpu_matrix(), expected);
                drop(dispatch.finish().unwrap());
            }
            for (column, shard) in input.shards().iter().enumerate() {
                assert_eq!(shard.value.is_ntt(), column % 2 != 0);
            }
            drop(outputs);
        }
    }
}
