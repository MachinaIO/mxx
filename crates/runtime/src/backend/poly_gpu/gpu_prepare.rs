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
pub(super) enum PreparedMatrixSource {
    Shard(usize),
    Replica { device: usize, context: usize, evaluation: bool },
    Fragment { device: usize, context: usize, index: usize, evaluation: bool },
}

impl PreparedMatrixSource {
    pub(super) fn layout(self, matrix: &GpuFleetMatrix) -> ((usize, usize), bool) {
        match self {
            Self::Shard(index) | Self::Fragment { index, .. } => {
                (matrix.shards[index].value.size(), matrix.shards[index].value.is_ntt())
            }
            Self::Replica { evaluation, .. } => {
                let original = matrix.shards[0].value.is_ntt();
                let mixed = matrix.shards.iter().any(|shard| shard.value.is_ntt() != original);
                (matrix.size(), if mixed { evaluation } else { original })
            }
        }
    }
}

pub(super) type PreparedMatrixInputs =
    HashMap<(u64, PreparedMatrixSource), Arc<GpuColumnShard<GpuDCRTPolyMatrix>>>;

pub(super) struct MatrixInputPreparation {
    pub matrix: GpuFleetMatrix,
    pub source: PreparedMatrixSource,
    pub parameters: GpuDCRTPolyParams,
    pub device: usize,
    pub storage: Arc<GpuPreparedStorage>,
    pub request: GpuPreparedRequest,
}

/// Matrix size classes keep incompatible aspect ratios from consuming one
/// another's peak capacity. No dimensions are rounded in the native backing.
pub(super) fn matrix_capacity_class(rows: usize, columns: usize) -> (u32, u32) {
    (capacity_class(rows), capacity_class(columns))
}

pub(super) fn capacity_class(size: usize) -> u32 {
    usize::BITS - size.max(1).saturating_sub(1).leading_zeros()
}

pub(super) fn select_prepared_matrix(
    inventory: &[(usize, Arc<GpuPreparedStorage>)],
    chosen: &mut HashSet<u64>,
    device: usize,
    parameters: &GpuDCRTPolyParams,
    level: usize,
    shape: (usize, usize),
    evaluation: bool,
    compact_bound: Option<&num_bigint::BigUint>,
) -> Result<Option<(Arc<GpuPreparedStorage>, GpuPreparedRequest)>, PolyBackendError> {
    // Prefer the prepared size class. Its slots cover peak simultaneous
    // demand; taking another class first can starve a later larger output.
    let mut selected = None;
    for (_, storage) in inventory
        .iter()
        .filter(|(owner, storage)| *owner == device && storage.matches_parameters(parameters))
    {
        for index in 0..storage.slot_count() {
            let slot = storage.slot_identity(index).unwrap();
            if slot.kind() !=
                if compact_bound.is_some() {
                    GpuPreparedSlotKind::CompactPayload
                } else {
                    GpuPreparedSlotKind::Matrix
                } ||
                (compact_bound.is_none() && slot.level() != Some(level)) ||
                chosen.contains(&slot.slot_id())
            {
                continue;
            }
            let request = if let Some(bound) = compact_bound {
                slot.workspace_request(
                    GpuSmallMatrix::allocation_bytes(parameters, shape.0, shape.1, bound)
                        .map_err(|e| PolyBackendError::GpuSubmission(e.to_string()))?,
                    256,
                )
            } else {
                slot.matrix_request(shape.0, shape.1, evaluation)
            };
            if storage.fits(&[request]).map_err(PolyBackendError::GpuCalibration)? {
                let size = (
                    if compact_bound.is_some() {
                        capacity_class(slot.requested_backing_bytes()) !=
                            capacity_class(request.bytes())
                    } else {
                        matrix_capacity_class(slot.rows(), slot.columns()) !=
                            matrix_capacity_class(shape.0, shape.1)
                    },
                    slot.requested_backing_bytes(),
                );
                if selected.as_ref().is_none_or(|(_, _, _, previous)| size < *previous) {
                    selected = Some((storage.clone(), request, slot.slot_id(), size));
                }
            }
        }
    }
    Ok(selected.map(|(storage, request, slot, _)| {
        chosen.insert(slot);
        (storage, request)
    }))
}

/// Select one immutable input range, with a shared full replica when existing
/// owners cannot serve it on the required native parameter context.
pub(super) fn select_matrix_input(
    matrix: &GpuFleetMatrix,
    columns: std::ops::Range<usize>,
    device: usize,
    parameters: &GpuDCRTPolyParams,
    evaluation: bool,
    inventory: &[(usize, Arc<GpuPreparedStorage>)],
    chosen: &mut HashSet<u64>,
    planned: &mut HashSet<(u64, PreparedMatrixSource)>,
    preparation: &mut Vec<MatrixInputPreparation>,
) -> Result<Option<PreparedMatrixSource>, PolyBackendError> {
    let first = matrix.shards.first().ok_or(PolyBackendError::InvalidConstantShape)?;
    let original = matrix.shards.iter().position(|shard| {
        shard.device_id == parameters.device_ids()[0] &&
            shard.value.params().context_identity() == parameters.context_identity() &&
            shard.global_column_start <= columns.start &&
            columns.end - shard.global_column_start <= shard.value.col_size()
    });
    let source = if let Some(index) = original {
        if matrix.shards[index].value.is_ntt() == evaluation {
            return Ok(Some(PreparedMatrixSource::Shard(index)));
        }
        PreparedMatrixSource::Fragment {
            device,
            context: parameters.context_identity(),
            index,
            evaluation,
        }
    } else {
        if matrix.shards.iter().any(|shard| {
            shard.value.level() != first.value.level() ||
                shard.value.params().ring_dimension() != parameters.ring_dimension() ||
                shard.value.params().moduli() != parameters.moduli()
        }) {
            return Err(PolyBackendError::GpuSubmission(
                "prepared replica has incompatible source parameters".into(),
            ));
        }
        PreparedMatrixSource::Replica { device, context: parameters.context_identity(), evaluation }
    };
    if planned.contains(&(matrix.id, source)) {
        return Ok(Some(source));
    }
    if let PreparedMatrixSource::Replica { device, context, evaluation } = source {
        let mixed = matrix.shards.iter().any(|shard| shard.value.is_ntt() != first.value.is_ntt());
        if mixed {
            // Each mismatching fragment crosses devices at most once, into its
            // destination-context normalization owner. Replicas in this batch
            // share it; original inputs are immutable. Reserve every fragment
            // before submission, including those needed only by a pilot owner.
            for (index, shard) in matrix.shards.iter().enumerate() {
                if shard.value.is_ntt() == evaluation {
                    continue;
                }
                let fragment =
                    PreparedMatrixSource::Fragment { device, context, index, evaluation };
                if planned.contains(&(matrix.id, fragment)) {
                    continue;
                }
                let Some((storage, request)) = select_prepared_matrix(
                    inventory,
                    chosen,
                    device,
                    parameters,
                    shard.value.level(),
                    shard.value.size(),
                    shard.value.is_ntt(),
                    None,
                )?
                else {
                    return Ok(None);
                };
                planned.insert((matrix.id, fragment));
                preparation.push(MatrixInputPreparation {
                    matrix: matrix.clone(),
                    source: fragment,
                    parameters: parameters.clone(),
                    device,
                    storage,
                    request,
                });
            }
        }
    }
    let (shape, original_evaluation) = source.layout(matrix);
    let level = match source {
        PreparedMatrixSource::Shard(index) | PreparedMatrixSource::Fragment { index, .. } => {
            matrix.shards[index].value.level()
        }
        _ => first.value.level(),
    };
    let Some((storage, request)) = select_prepared_matrix(
        inventory,
        chosen,
        device,
        parameters,
        level,
        shape,
        original_evaluation,
        None,
    )?
    else {
        return Ok(None);
    };
    planned.insert((matrix.id, source));
    preparation.push(MatrixInputPreparation {
        matrix: matrix.clone(),
        source,
        parameters: parameters.clone(),
        device,
        storage,
        request,
    });
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
                                preparation[*index].source,
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
                        let (shape, evaluation) = input.source.layout(&input.matrix);
                        let source_shards = match input.source {
                            PreparedMatrixSource::Shard(index) |
                            PreparedMatrixSource::Fragment { index, .. } => {
                                &input.matrix.shards[index..index + 1]
                            }
                            PreparedMatrixSource::Replica { .. } => input.matrix.shards.as_slice(),
                        };
                        let start = match input.source {
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
                                    input.source
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
                        let desired_evaluation = match input.source {
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
                            (input.matrix.id, input.source),
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
            poly::dcrt::{
                gpu::{GpuMatrixExecutionClass, detected_gpu_device_ids},
                params::DCRTPolyParams,
            },
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
            )
            .unwrap(),
        );
        let shapes = [(columns, rows), (6, 2 * columns), (2, 2 * columns)];
        let readbacks = shapes
            .into_iter()
            .map(|(rows, columns)| {
                let demand = params
                    .matrix_allocation_bytes(params.crt_depth() - 1, rows, columns, true)
                    .unwrap();
                let events = if demand.execution_class == GpuMatrixExecutionClass::PerLimbStreams {
                    params.crt_depth()
                } else {
                    1
                };
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
                        vec![GpuDCRTPolyMatrix::zero(&params, rows, columns)],
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
            (0, GpuInvocation::Transpose { value: &input }),
            (0, GpuInvocation::Tensor { left: &input, right: &right }),
            (0, GpuInvocation::TensorSumRows { left: &input, right: &right, rows: &tensor_groups }),
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
