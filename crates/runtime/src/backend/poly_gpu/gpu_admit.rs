//! Prepared matrix preflight: select native output slots, measure an isolated
//! range, reserve every retained destination, then publish the compiled batch.

use super::{
    gpu_compiled::{MatrixInvocation, PreparedClaimBroker},
    gpu_prepare::{
        MatrixInputPreparation, PreparedMatrixSource, select_matrix_input, select_prepared_matrix,
    },
    *,
};
use crate::gpu_memory::{
    GpuAdmissionError, GpuColumnAllocations, GpuColumnMemoryRequirements, GpuMemoryLedger,
    GpuOutputOwnership,
};
use mxx_primitives::matrix::gpu_dcrt_poly::{
    GpuPreparedRequest, GpuPreparedSlotIdentity, GpuPreparedSlotKind, GpuPreparedStorage,
    GpuPreparedWorkspaceLayout,
};

struct MatrixDestination {
    interval: GpuColumnInterval,
    source: Option<PreparedMatrixSource>,
    right: Vec<PreparedMatrixSource>,
    parameters: GpuDCRTPolyParams,
    level: usize,
    evaluation: bool,
    storage: Arc<GpuPreparedStorage>,
    request: GpuPreparedRequest,
}

struct MatrixScratch {
    device: usize,
    rows: usize,
    storage: Arc<GpuPreparedStorage>,
    slot: GpuPreparedSlotIdentity,
    evaluation: bool,
    workspace: Option<GpuPreparedWorkspaceLayout>,
    /// Width-scaled native workspace: the operation, context, level and the
    /// index into its `width_workspaces` for the requested column count.
    width: Option<(PreparedMatrixOperation, GpuDCRTPolyParams, usize, usize)>,
}

fn select_operation_scratch(
    operation: &PreparedMatrixOperation,
    inventory: &[(usize, Arc<GpuPreparedStorage>)],
    chosen: &mut HashSet<u64>,
    device: usize,
    parameters: &GpuDCRTPolyParams,
    level: usize,
    evaluation: bool,
) -> Result<Option<Vec<MatrixScratch>>, PolyBackendError> {
    let mut scratch = Vec::new();
    for rows in operation.scratch_rows()? {
        let Some((storage, request)) = select_prepared_matrix(
            inventory,
            chosen,
            device,
            parameters,
            level,
            (rows, 1),
            evaluation,
            None,
        )?
        else {
            return Ok(None);
        };
        let slot = (0..storage.slot_count())
            .filter_map(|index| storage.slot_identity(index))
            .find(|slot| slot.matrix_request(rows, 1, evaluation) == request)
            .unwrap();
        scratch.push(MatrixScratch {
            device,
            rows,
            storage,
            slot,
            evaluation,
            workspace: None,
            width: None,
        });
    }
    let fixed = operation.fixed_workspaces(parameters, level)?;
    let fixed_count = fixed.len();
    let width = operation.width_workspaces(parameters, level, 1)?;
    for (index, layout) in fixed.into_iter().chain(width).enumerate() {
        let mut selected = None;
        for (_, storage) in inventory
            .iter()
            .filter(|(owner, storage)| *owner == device && storage.matches_parameters(parameters))
        {
            for index in 0..storage.slot_count() {
                let slot = storage.slot_identity(index).unwrap();
                if slot.kind() != layout.kind || chosen.contains(&slot.slot_id()) {
                    continue;
                }
                let request = slot.workspace_request(layout.bytes, layout.alignment);
                if storage.fits(&[request]).map_err(PolyBackendError::GpuCalibration)? &&
                    selected.as_ref().is_none_or(
                        |(_, previous): &(Arc<GpuPreparedStorage>, GpuPreparedSlotIdentity)| {
                            slot.requested_backing_bytes() < previous.requested_backing_bytes()
                        },
                    )
                {
                    selected = Some((storage.clone(), slot));
                }
            }
        }
        let Some((storage, slot)) = selected else {
            return Ok(None);
        };
        chosen.insert(slot.slot_id());
        let scaled = index >= fixed_count;
        scratch.push(MatrixScratch {
            device,
            rows: 0,
            storage,
            slot,
            evaluation,
            workspace: (!scaled).then_some(layout),
            width: scaled
                .then(|| (operation.clone(), parameters.clone(), level, index - fixed_count)),
        });
    }
    Ok(Some(scratch))
}

struct MatrixDeviceInputPlan {
    device: usize,
    capacity: usize,
    source: Option<PreparedMatrixSource>,
    right: Vec<PreparedMatrixSource>,
    parameters: GpuDCRTPolyParams,
    level: usize,
    evaluation: bool,
    chosen: HashSet<u64>,
    planned: HashSet<(u64, PreparedMatrixSource)>,
    preparation: Vec<MatrixInputPreparation>,
    scratch: Vec<MatrixScratch>,
}

struct MatrixRequirements {
    outputs: Vec<MatrixDestination>,
    representative: Option<MatrixDestination>,
    scratch: Vec<MatrixScratch>,
}

impl MatrixRequirements {
    fn scratch_allocations(&self, device: usize, columns: usize) -> GpuColumnAllocations<'_> {
        GpuColumnAllocations {
            managed_bounds: Vec::new(),
            prepared: self
                .scratch
                .iter()
                .filter(|scratch| scratch.device == device)
                .map(|scratch| {
                    (
                        &*scratch.storage,
                        vec![match (&scratch.workspace, &scratch.width) {
                            (Some(layout), _) => {
                                scratch.slot.workspace_request(layout.bytes, layout.alignment)
                            }
                            (None, Some((operation, parameters, level, index))) => {
                                // Exact native size for this width; a failed query
                                // cannot fit and is rejected by the ledger's fit check.
                                let layout = operation
                                    .width_workspaces(parameters, *level, columns)
                                    .ok()
                                    .and_then(|layouts| layouts.get(*index).copied())
                                    .unwrap_or(GpuPreparedWorkspaceLayout {
                                        kind: scratch.slot.kind(),
                                        bytes: usize::MAX,
                                        alignment: 1,
                                    });
                                scratch.slot.workspace_request(layout.bytes, layout.alignment)
                            }
                            (None, None) => scratch.slot.matrix_request(
                                scratch.rows,
                                columns,
                                scratch.evaluation,
                            ),
                        }],
                    )
                })
                .collect(),
        }
    }
}

impl GpuColumnMemoryRequirements for MatrixRequirements {
    fn fixed_allocations(&self, _: usize) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
        Ok(GpuColumnAllocations::default())
    }
    fn minimum_temporary_allocations(
        &self,
        device: usize,
    ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
        Ok(self.scratch_allocations(device, 1))
    }
    fn output_bound(
        &self,
        device: usize,
        columns: usize,
    ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
        let mut result = GpuColumnAllocations::default();
        if columns != 0 {
            // A monotone envelope for this frozen inherited layout. Repeated
            // storage groups are merged only in the bound, not dispatch order.
            for output in self.outputs.iter().filter(|output| output.interval.device == device) {
                if let Some((_, requests)) = result
                    .prepared
                    .iter_mut()
                    .find(|(store, _)| store.identity() == output.storage.identity())
                {
                    requests.push(output.request);
                } else {
                    result.prepared.push((&output.storage, vec![output.request]));
                }
            }
        }
        Ok(result)
    }
    fn output_allocations(
        &self,
        device: usize,
        intervals: &[GpuColumnInterval],
    ) -> Result<GpuColumnAllocations<'_>, GpuAdmissionError> {
        if intervals.iter().ne(self.outputs.iter().map(|output| &output.interval)) {
            return Err(GpuAdmissionError::InvalidPlan(
                "matrix ownership changed after native output selection".into(),
            ));
        }
        Ok(GpuColumnAllocations {
            managed_bounds: Vec::new(),
            prepared: self
                .outputs
                .iter()
                .filter(|output| output.interval.device == device)
                .map(|output| (&*output.storage, vec![output.request]))
                .collect(),
        })
    }
    fn temporary_allocations(
        &self,
        device: usize,
        _: &GpuAllocationClass,
        columns: usize,
    ) -> Result<GpuColumnAllocations<'_>, GpuCalibrationError> {
        Ok(self.scratch_allocations(device, columns))
    }
    fn validate_ranges(
        &self,
        intervals: &[GpuColumnInterval],
        _: &[usize],
    ) -> Result<(), GpuAdmissionError> {
        if intervals.iter().ne(self.outputs.iter().map(|output| &output.interval)) {
            return Err(GpuAdmissionError::InvalidPlan(
                "matrix range boundaries changed after compilation".into(),
            ));
        }
        Ok(())
    }
}

impl GpuDcrtBackend {
    /// Accept the fleet's complete prepared inventory at the explicit setup
    /// boundary. Close audited managed allocation domains, obtain actual native
    /// residency receipts in parallel, and install the dispatcher-owned ledger.
    /// The caller excludes unrelated CUDA allocations during setup observation.
    /// Setup may wait for its own streams; subsequent admission does not.
    pub fn prepare_memory(
        &mut self,
        prepared: Vec<(usize, Arc<GpuPreparedStorage>)>,
        external_pool_exclusive: bool,
    ) -> Result<(), PolyBackendError> {
        use mxx_primitives::poly::dcrt::gpu::{
            GpuAllocationEpochBoundary, GpuAllocationEpochObservation,
        };
        if !external_pool_exclusive ||
            self.prepared_required ||
            !self.prepared_invocations.is_empty()
        {
            return Err(PolyBackendError::GpuSubmission(
                "prepared setup requires exclusive observation and an unconfigured fleet".into(),
            ));
        }
        let parameters = self.device_parameters();
        let mut identities = HashSet::new();
        // Validate the entire declared fleet before closing any allocation domain.
        for (device, storage) in &prepared {
            let params = parameters.get(*device).ok_or(PolyBackendError::UnsupportedPlacement)?;
            if params.execution_owner_id() != Some(storage.execution_owner_id()) ||
                params.device_ids() != vec![storage.device()] ||
                !identities.insert(storage.identity())
            {
                return Err(PolyBackendError::GpuSubmission(
                    "prepared setup has foreign or duplicate storage".into(),
                ));
            }
        }
        if (0..parameters.len()).any(|device| !prepared.iter().any(|(owner, _)| *owner == device)) {
            return Err(PolyBackendError::GpuSubmission(
                "prepared setup is missing a configured device".into(),
            ));
        }
        let epochs = parameters.par_iter().enumerate().map(|(device, params)| {
            let inventory = prepared.iter().filter(|(owner, _)| *owner == device)
                .map(|(_, storage)| &**storage).collect::<Vec<_>>();
            GpuPreparedStorage::finish_setup(&inventory).map_err(PolyBackendError::GpuSubmission)?;
            match params.observe_allocation_epoch(params.device_ids()[0],
                GpuAllocationEpochBoundary::InitialSetup, true).map_err(PolyBackendError::GpuSubmission)? {
                GpuAllocationEpochObservation::Verified(epoch) => Ok(epoch),
                GpuAllocationEpochObservation::Unverified(reason) => Err(PolyBackendError::GpuSubmission(
                    format!("prepared setup observation was not verified on device {device}: {reason:?}"))),
            }
        }).collect::<Result<Vec<_>, _>>()?;
        let ledger = GpuMemoryLedger::new(epochs, self.vram_percent, prepared)
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        self.set_memory_ledger(ledger)
    }

    /// Install an accepted setup ledger once. Native epoch receipts remain the
    /// only public way to construct that ledger. Ordinary preflight then selects
    /// native slots, calibrates supported matrix classes and acquires their plans.
    /// This does not provision backing or enable an incomplete physical seal.
    pub fn set_memory_ledger(&mut self, ledger: GpuMemoryLedger) -> Result<(), PolyBackendError> {
        let params = self.device_parameters();
        let identities = params
            .iter()
            .zip(&self.devices)
            .map(|(params, (device, _))| params.execution_owner_id().map(|owner| (*device, owner)))
            .collect::<Option<Vec<_>>>()
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        if self.prepared_ledger.is_some() ||
            !self.prepared_invocations.is_empty() ||
            ledger.execution_identities() != Some(identities.as_slice()) ||
            ledger.devices().len() != params.len() ||
            ledger
                .devices()
                .iter()
                .zip(&params)
                .any(|(device, params)| device.budget_bytes != params.vram_budget_bytes() as u64)
        {
            return Err(PolyBackendError::GpuSubmission(
                "prepared ledger differs from this fleet's execution owners or fixed budgets"
                    .into(),
            ));
        }
        self.prepared_ledger = Some(ledger);
        self.prepared_required = true;
        self.pending_pilot = None;
        self.pending_profile = None;
        self.operation_profiles.clear();
        Ok(())
    }

    pub(super) fn admit_matrix_invocations(
        &mut self,
        requests: &[(usize, MatrixInvocation<'_>)],
    ) -> Result<(), PolyBackendError> {
        if requests.is_empty() {
            return Ok(());
        }
        let environment = crate::gpu_calibration::gpu_calibration_environment(
            &gpu_device_identity(self.devices[0].0).map_err(PolyBackendError::GpuCalibration)?,
            self.devices.len(),
            self.vram_percent,
        );
        // Keep the ledger owned by the dispatcher across all early returns.
        let mut ledger = self.prepared_ledger.take().expect("prepared preflight has a ledger");
        let result = (|| {
            let inventory = Arc::new(
                ledger
                    .prepared_inventory()
                    .map(|(device, storage)| (device, storage.clone()))
                    .collect::<Vec<_>>(),
            );
            let mut chosen = HashSet::new();
            let mut layouts = Vec::with_capacity(requests.len());
            let mut preparation = Vec::new();
            let mut planned = HashSet::new();
            // Validate the entire batch and its native fits before any pilot.
            // Selection is ordered because later outputs must exclude earlier
            // choices; acquiring real reservations below rechecks every slot.
            for (placement, request) in requests {
                if *placement != 0 {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
                let (operation, left, right, compact) =
                    CompiledMatrixInvocation::arguments(request, self)?;
                let rows = operation.output_rows(left, &right)?;
                let columns = operation.output_columns(left);
                let mut outputs = Vec::new();
                let mut scratch = Vec::new();
                let capacity_error = || {
                    PolyBackendError::GpuSubmission(
                        "complete matrix inputs and outputs do not fit the prepared inventory"
                            .into(),
                    )
                };
                if columns != 0 &&
                    matches!(
                        operation,
                        PreparedMatrixOperation::Constant { .. } |
                            PreparedMatrixOperation::Sample { .. } |
                            PreparedMatrixOperation::Hash { .. } |
                            PreparedMatrixOperation::Decompose { hash: Some(_), .. } |
                            PreparedMatrixOperation::ImportMatrix { .. } |
                            PreparedMatrixOperation::ImportCompact { .. } |
                            PreparedMatrixOperation::ImportStaging { .. } |
                            PreparedMatrixOperation::Preimage { .. } |
                            PreparedMatrixOperation::Polynomial { .. } |
                            PreparedMatrixOperation::Transpose |
                            PreparedMatrixOperation::Tensor { .. }
                    )
                {
                    let first = left.and_then(|left| left.shards.first());
                    let evaluation =
                        operation.input_evaluation(first.is_some_and(|first| first.value.is_ntt()));
                    // Fresh output columns use each owner's actual native capacity
                    // after that owner's complete fixed inputs have been selected.
                    let trials = self
                        .devices
                        .par_iter()
                        .enumerate()
                        .map(|(device, (_, backend))| {
                            let (parameters, level) = if let Some(first) = first {
                                (
                                    backend.parameters_for_matrix(&first.value)?.clone(),
                                    first.value.level(),
                                )
                            } else if let Some(ty) = operation.fresh_type() {
                                let parameters = backend.parameters(ty)?.clone();
                                let level = parameters.crt_depth() - 1;
                                (parameters, level)
                            } else {
                                return Err(PolyBackendError::InvalidConstantShape);
                            };
                            let mut chosen = chosen.clone();
                            let mut planned = planned.clone();
                            let mut preparation = Vec::new();
                            let source = if let Some(left) = left {
                                let Some(source) = select_matrix_input(
                                    left,
                                    0..left.columns,
                                    device,
                                    &parameters,
                                    evaluation,
                                    &inventory,
                                    &mut chosen,
                                    &mut planned,
                                    &mut preparation,
                                )?
                                else {
                                    return Ok(None);
                                };
                                Some(source)
                            } else {
                                None
                            };
                            let mut rhs = Vec::with_capacity(right.len());
                            for right in operation.dependent_inputs(&right) {
                                let Some(source) = select_matrix_input(
                                    right,
                                    0..right.columns,
                                    device,
                                    &parameters,
                                    true,
                                    &inventory,
                                    &mut chosen,
                                    &mut planned,
                                    &mut preparation,
                                )?
                                else {
                                    return Ok(None);
                                };
                                rhs.push(source);
                            }
                            let Some(scratch) = select_operation_scratch(
                                &operation,
                                &inventory,
                                &mut chosen,
                                device,
                                &parameters,
                                level,
                                operation.scratch_evaluation(evaluation),
                            )?
                            else {
                                return Ok(None);
                            };
                            let (mut low, mut high) = (0usize, columns);
                            while low < high {
                                let middle = low + (high - low).div_ceil(2);
                                let mut candidate = chosen.clone();
                                if select_prepared_matrix(
                                    &inventory,
                                    &mut candidate,
                                    device,
                                    &parameters,
                                    level,
                                    (rows, middle),
                                    evaluation,
                                    operation.compact_bound(),
                                )?
                                .is_some()
                                {
                                    low = middle;
                                } else {
                                    high = middle - 1;
                                }
                            }
                            Ok::<_, PolyBackendError>((low != 0).then_some(MatrixDeviceInputPlan {
                                device,
                                capacity: low,
                                source,
                                right: rhs,
                                parameters,
                                level,
                                evaluation,
                                chosen,
                                planned,
                                preparation,
                                scratch,
                            }))
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let capacities = trials
                        .iter()
                        .map(|trial| trial.as_ref().map_or(0, |trial| trial.capacity))
                        .collect::<Vec<_>>();
                    let assigned =
                        crate::gpu_calibration::gpu_capped_waterfill_columns(&capacities, columns)
                            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
                    if assigned.iter().sum::<usize>() != columns {
                        return Err(capacity_error());
                    }
                    let mut start = 0;
                    for (trial, count) in trials.into_iter().zip(assigned) {
                        if count == 0 {
                            continue;
                        }
                        let trial = trial.ok_or_else(capacity_error)?;
                        chosen.extend(trial.chosen);
                        planned.extend(trial.planned);
                        preparation.extend(trial.preparation);
                        scratch.extend(trial.scratch);
                        let (storage, request) = select_prepared_matrix(
                            &inventory,
                            &mut chosen,
                            trial.device,
                            &trial.parameters,
                            trial.level,
                            (rows, count),
                            trial.evaluation,
                            operation.compact_bound(),
                        )?
                        .ok_or_else(capacity_error)?;
                        outputs.push(MatrixDestination {
                            interval: GpuColumnInterval {
                                device: trial.device,
                                start,
                                end: start + count,
                            },
                            source: trial.source,
                            right: trial.right,
                            parameters: trial.parameters,
                            level: trial.level,
                            evaluation: trial.evaluation,
                            storage,
                            request,
                        });
                        start += count;
                    }
                } else if let Some(compact) = compact {
                    // Compact inputs inherit their resident column ownership. No
                    // replica or fragment preparation exists for compact payloads.
                    for (index, shard) in compact.shards.iter().enumerate() {
                        let device = self
                            .devices
                            .iter()
                            .position(|(id, _)| *id == shard.device_id)
                            .ok_or(PolyBackendError::UnsupportedPlacement)?;
                        let (parameters, level, evaluation) = operation.output_layout(
                            &self.devices[device].1,
                            shard.value.params(),
                            shard.value.params().crt_depth() - 1,
                            false,
                        )?;
                        let (start, end) = (
                            shard.global_column_start,
                            shard.global_column_start + shard.value.columns_count(),
                        );
                        // Fixed ordinary inputs are prepared once per device in the
                        // compact input's context and evaluation format.
                        let mut rhs = Vec::with_capacity(right.len());
                        for (position, fixed) in
                            operation.dependent_inputs(&right).iter().enumerate()
                        {
                            rhs.push(
                                select_matrix_input(
                                    fixed,
                                    operation.other_columns(position, fixed, start, end),
                                    device,
                                    shard.value.params(),
                                    true,
                                    &inventory,
                                    &mut chosen,
                                    &mut planned,
                                    &mut preparation,
                                )?
                                .ok_or_else(capacity_error)?,
                            );
                        }
                        if !scratch.iter().any(|s: &MatrixScratch| s.device == device) {
                            scratch.extend(
                                select_operation_scratch(
                                    &operation,
                                    &inventory,
                                    &mut chosen,
                                    device,
                                    &parameters,
                                    level,
                                    operation.scratch_evaluation(evaluation),
                                )?
                                .ok_or_else(capacity_error)?,
                            );
                        }
                        let (storage, request) = select_prepared_matrix(
                            &inventory,
                            &mut chosen,
                            device,
                            &parameters,
                            level,
                            (rows, end - start),
                            evaluation,
                            operation.compact_bound(),
                        )?
                        .ok_or_else(capacity_error)?;
                        outputs.push(MatrixDestination {
                            interval: GpuColumnInterval { device, start, end },
                            source: Some(PreparedMatrixSource::Shard(index)),
                            right: rhs,
                            parameters,
                            level,
                            evaluation,
                            storage,
                            request,
                        });
                    }
                } else {
                    let Some(left) = left else {
                        if columns == 0 {
                            layouts.push((
                                operation,
                                None,
                                right,
                                None,
                                Arc::new(MatrixRequirements {
                                    outputs,
                                    representative: None,
                                    scratch,
                                }),
                            ));
                            continue;
                        }
                        return Err(PolyBackendError::InvalidConstantShape);
                    };
                    let mut boundaries = if let PreparedMatrixOperation::ConcatColumns {
                        offsets,
                        ..
                    } = &operation
                    {
                        [left]
                            .into_par_iter()
                            .chain(right.par_iter())
                            .zip(offsets.par_iter())
                            .flat_map_iter(|(input, &(_, offset, _))| {
                                input
                                    .shards
                                    .iter()
                                    .map(move |shard| offset + shard.global_column_start)
                            })
                            .chain([columns].into_par_iter())
                            .collect::<Vec<_>>()
                    } else {
                        left.shards
                            .par_iter()
                            .map(|shard| shard.global_column_start)
                            .chain(
                                right
                                    .par_iter()
                                    .enumerate()
                                    .filter(|(index, _)| !operation.fixed_input(*index))
                                    .flat_map_iter(|(_, right)| {
                                        right.shards.iter().map(|shard| shard.global_column_start)
                                    }),
                            )
                            .chain([left.columns].into_par_iter())
                            .collect::<Vec<_>>()
                    };
                    if let PreparedMatrixOperation::Slice { columns, .. } = &operation {
                        boundaries.retain(|&start| start > columns.start && start < columns.end);
                        boundaries.extend([columns.start, columns.end]);
                        for start in &mut boundaries {
                            *start -= columns.start;
                        }
                    }
                    boundaries.par_sort_unstable();
                    boundaries.dedup();
                    for range in boundaries.windows(2) {
                        let (start, end) = (range[0], range[1]);
                        let primary = operation
                            .primary(Some(left), &right, start)
                            .ok_or(PolyBackendError::InvalidConstantShape)?;
                        let source_columns = operation.source_columns(primary, start, end);
                        let input = primary
                            .shards
                            .iter()
                            .find(|shard| {
                                shard.global_column_start <= source_columns.start &&
                                    source_columns.end - shard.global_column_start <=
                                        shard.value.col_size()
                            })
                            .ok_or(PolyBackendError::InvalidConstantShape)?;
                        let device = self
                            .devices
                            .iter()
                            .position(|(id, _)| *id == input.device_id)
                            .ok_or(PolyBackendError::UnsupportedPlacement)?;
                        let registered =
                            self.devices[device].1.parameters_for_matrix(&input.value)?;
                        if registered != input.value.params() ||
                            registered.execution_owner_id() !=
                                input.value.params().execution_owner_id()
                        {
                            return Err(PolyBackendError::UnsupportedPlacement);
                        }
                        let evaluation = operation.input_evaluation(input.value.is_ntt());
                        let source = select_matrix_input(
                            primary,
                            source_columns,
                            device,
                            input.value.params(),
                            evaluation,
                            &inventory,
                            &mut chosen,
                            &mut planned,
                            &mut preparation,
                        )?
                        .ok_or_else(capacity_error)?;
                        let mut rhs = Vec::with_capacity(right.len());
                        for (index, right) in operation.dependent_inputs(&right).iter().enumerate()
                        {
                            let parameters = if matches!(
                                operation,
                                PreparedMatrixOperation::CrtRecompose { .. }
                            ) {
                                self.devices[device].1.parameters_for_matrix(
                                    &right
                                        .shards
                                        .first()
                                        .ok_or(PolyBackendError::InvalidConstantShape)?
                                        .value,
                                )?
                            } else {
                                input.value.params()
                            };
                            rhs.push(
                                select_matrix_input(
                                    right,
                                    operation.other_columns(index, right, start, end),
                                    device,
                                    parameters,
                                    evaluation,
                                    &inventory,
                                    &mut chosen,
                                    &mut planned,
                                    &mut preparation,
                                )?
                                .ok_or_else(capacity_error)?,
                            );
                        }
                        let (parameters, level, evaluation) = operation.output_layout(
                            &self.devices[device].1,
                            input.value.params(),
                            input.value.level(),
                            evaluation,
                        )?;
                        if let Some(existing) = scratch.iter().find(|s| s.device == device) {
                            if !existing.storage.matches_parameters(&parameters) ||
                                (existing.slot.kind() == GpuPreparedSlotKind::Matrix &&
                                    (existing.slot.level() != Some(level) ||
                                        existing.evaluation !=
                                            operation.scratch_evaluation(evaluation)))
                            {
                                return Err(PolyBackendError::GpuSubmission("one reduction job owner requires a consistent scratch context and format".into()));
                            }
                        } else {
                            scratch.extend(
                                select_operation_scratch(
                                    &operation,
                                    &inventory,
                                    &mut chosen,
                                    device,
                                    &parameters,
                                    level,
                                    operation.scratch_evaluation(evaluation),
                                )?
                                .ok_or_else(capacity_error)?,
                            );
                        }
                        operation.validate_output_workspace(
                            &input.value,
                            &right,
                            (rows, end - start),
                        )?;
                        let (storage, request) = select_prepared_matrix(
                            &inventory,
                            &mut chosen,
                            device,
                            &parameters,
                            level,
                            (rows, end - start),
                            evaluation,
                            operation.compact_bound(),
                        )?
                        .ok_or_else(capacity_error)?;
                        outputs.push(MatrixDestination {
                            interval: GpuColumnInterval { device, start, end },
                            source: Some(source),
                            right: rhs,
                            parameters,
                            level,
                            evaluation,
                            storage,
                            request,
                        });
                    }
                }
                // Device 1 remains the nonzero-role representative even when
                // inherited ownership leaves it idle. Prepare its actual one-column
                // class separately; no production interval is moved onto it.
                let representative = if outputs.iter().any(|output| output.interval.device > 0) &&
                    !outputs.iter().any(|output| output.interval.device == 1)
                {
                    let reference =
                        outputs.iter().find(|output| output.interval.device > 0).unwrap();
                    let primary = operation.primary(left, &right, reference.interval.start);
                    let (parameters, level, evaluation) = if let Some(primary) = primary {
                        let source =
                            reference.source.ok_or(PolyBackendError::InvalidConstantShape)?;
                        let exemplar = match source {
                            PreparedMatrixSource::Shard(index) |
                            PreparedMatrixSource::Fragment { index, .. } => {
                                &primary.shards[index].value
                            }
                            PreparedMatrixSource::Replica { .. } => &primary.shards[0].value,
                        };
                        (
                            self.devices[1].1.parameters_for_matrix(exemplar)?.clone(),
                            exemplar.level(),
                            operation.input_evaluation(source.layout(primary).1),
                        )
                    } else if let Some(shard) =
                        CompiledMatrixInvocation::compact_shard(compact, reference.source)
                    {
                        operation.output_layout(
                            &self.devices[1].1,
                            shard.value.params(),
                            shard.value.params().crt_depth() - 1,
                            false,
                        )?
                    } else if let Some(ty) = operation.fresh_type() {
                        let parameters = self.devices[1].1.parameters(ty)?.clone();
                        let level = parameters.crt_depth() - 1;
                        // Fresh ordinary outputs are evaluation-format; a fresh
                        // compact decomposition samples COEFF scratch instead.
                        (parameters, level, operation.input_evaluation(true))
                    } else {
                        return Err(PolyBackendError::InvalidConstantShape);
                    };
                    let start = reference.interval.start;
                    let source = primary
                        .map(|primary| {
                            select_matrix_input(
                                primary,
                                operation.source_columns(primary, start, start + 1),
                                1,
                                &parameters,
                                evaluation,
                                &inventory,
                                &mut chosen,
                                &mut planned,
                                &mut preparation,
                            )
                            .and_then(|source| source.ok_or_else(capacity_error))
                        })
                        .transpose()?;
                    let mut rhs = Vec::with_capacity(right.len());
                    for (index, right) in operation.dependent_inputs(&right).iter().enumerate() {
                        let input_parameters =
                            if matches!(operation, PreparedMatrixOperation::CrtRecompose { .. }) {
                                self.devices[1].1.parameters_for_matrix(
                                    &right
                                        .shards
                                        .first()
                                        .ok_or(PolyBackendError::InvalidConstantShape)?
                                        .value,
                                )?
                            } else {
                                &parameters
                            };
                        rhs.push(
                            select_matrix_input(
                                right,
                                operation.other_columns(index, right, start, start + 1),
                                1,
                                input_parameters,
                                evaluation,
                                &inventory,
                                &mut chosen,
                                &mut planned,
                                &mut preparation,
                            )?
                            .ok_or_else(capacity_error)?,
                        );
                    }
                    let (parameters, level, evaluation) = operation.output_layout(
                        &self.devices[1].1,
                        &parameters,
                        level,
                        evaluation,
                    )?;
                    scratch.extend(
                        select_operation_scratch(
                            &operation,
                            &inventory,
                            &mut chosen,
                            1,
                            &parameters,
                            level,
                            operation.scratch_evaluation(evaluation),
                        )?
                        .ok_or_else(capacity_error)?,
                    );
                    let (storage, request) = select_prepared_matrix(
                        &inventory,
                        &mut chosen,
                        1,
                        &parameters,
                        level,
                        (rows, 1),
                        evaluation,
                        operation.compact_bound(),
                    )?
                    .ok_or_else(capacity_error)?;
                    Some(MatrixDestination {
                        interval: GpuColumnInterval { device: 1, start, end: start + 1 },
                        source,
                        right: rhs,
                        parameters,
                        level,
                        evaluation,
                        storage,
                        request,
                    })
                } else {
                    None
                };
                layouts.push((
                    operation,
                    left.cloned(),
                    right,
                    compact.cloned(),
                    Arc::new(MatrixRequirements { outputs, representative, scratch }),
                ));
            }
            // All layouts and fixed/output fits are known before preparation.
            // Native owners retain the fixed slots after their tokens retire,
            // so calibration observes them in its excluded fixed baseline.
            let prepared = self.prepare_matrix_inputs(&mut ledger, preparation)?;
            let mut profiles = Vec::with_capacity(layouts.len());
            // All isolated pilots finish before any production reservation is
            // held, so joint occupancy resets cannot overlap admitted outputs.
            for (operation, left, right, compact, requirements) in &layouts {
                let rows = operation.output_rows(left.as_ref(), right)?;
                if requirements.outputs.is_empty() {
                    profiles.push(GpuCalibrationProfile { gpu0: None, nonzero: None });
                    continue;
                }
                let maximum = requirements
                    .outputs
                    .iter()
                    .map(|output| output.interval.end - output.interval.start)
                    .max()
                    .unwrap();
                let layout_key = requirements
                    .outputs
                    .par_iter()
                    .chain(requirements.representative.par_iter())
                    .map(|output| {
                        (
                            output.interval,
                            (rows, output.interval.end - output.interval.start),
                            output.parameters.ring_dimension(),
                            output.parameters.moduli().to_vec(),
                            output.level,
                            output.evaluation,
                            operation
                                .primary(left.as_ref(), right, output.interval.start)
                                .zip(output.source)
                                .map(|(primary, source)| {
                                    let input = &operation.source(&prepared, primary, source).value;
                                    (
                                        source.layout(primary),
                                        input.params().ring_dimension(),
                                        input.params().moduli().to_vec(),
                                        input.level(),
                                    )
                                }),
                            output
                                .right
                                .iter()
                                .zip(right)
                                .map(|(&index, value)| {
                                    let input = &operation.source(&prepared, value, index).value;
                                    (
                                        index.layout(value),
                                        input.params().ring_dimension(),
                                        input.params().moduli().to_vec(),
                                        input.level(),
                                    )
                                })
                                .collect::<Vec<_>>(),
                            CompiledMatrixInvocation::compact_shard(
                                compact.as_ref(),
                                output.source,
                            )
                            .map(|shard| {
                                (
                                    shard.value.size(),
                                    shard.value.params().moduli().to_vec(),
                                    shard.value.bound().clone(),
                                )
                            }),
                        )
                    })
                    .collect::<Vec<_>>();
                let inventory_key = inventory
                    .par_iter()
                    .map(|(device, storage)| {
                        (
                            *device,
                            (0..storage.slot_count())
                                .map(|index| {
                                    let slot = storage.slot_identity(index).unwrap();
                                    (
                                        slot.kind() as i32,
                                        slot.rows(),
                                        slot.columns(),
                                        slot.level(),
                                        slot.requested_backing_bytes(),
                                        slot.alignment(),
                                    )
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                let configuration = mxx_ir_core::encoding::hash_canonical(&(
                    "prepared-matrix-inventory/v1",
                    inventory_key,
                ))
                .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
                let identity = mxx_ir_core::encoding::hash_canonical(&(
                    "prepared-matrix-range/v1",
                    match operation {
                        PreparedMatrixOperation::Negate => 0u8,
                        PreparedMatrixOperation::Constant { .. } => 17,
                        PreparedMatrixOperation::Sample { .. } => 18,
                        PreparedMatrixOperation::Hash { .. } => 19,
                        PreparedMatrixOperation::Polynomial { .. } => 20,
                        PreparedMatrixOperation::CenteredRebase { .. } => 21,
                        PreparedMatrixOperation::ModulusConversion {..}=>22,
                        PreparedMatrixOperation::RnsConversion {..}=>23,
                        PreparedMatrixOperation::CrtRecompose {..}=>24,
                        PreparedMatrixOperation::Decompose {..}=>25,
                        PreparedMatrixOperation::CenteredExtendCompact {..}=>26,
                        PreparedMatrixOperation::MultiplyCompact {..}=>27,
                        PreparedMatrixOperation::ImportMatrix {..}=>28,
                        PreparedMatrixOperation::ImportCompact {..}=>29,
                        PreparedMatrixOperation::ImportStaging {..}=>30,
                        PreparedMatrixOperation::Preimage {..}=>31,
                        PreparedMatrixOperation::Accumulate { .. } => 16,
                        PreparedMatrixOperation::ConcatRows => 12,
                        PreparedMatrixOperation::ConcatColumns { diagonal: false, .. } => 14,
                        PreparedMatrixOperation::ConcatColumns { diagonal: true, .. } => 15,
                        PreparedMatrixOperation::AddRowBlocks => 13,
                        PreparedMatrixOperation::Add => 1,
                        PreparedMatrixOperation::Subtract => 2,
                        PreparedMatrixOperation::Scale(_) => 3,
                        PreparedMatrixOperation::Automorphism(_) => 4,
                        PreparedMatrixOperation::Multiply { scales_left: false } => 5,
                        PreparedMatrixOperation::Multiply { scales_left: true } => 6,
                        PreparedMatrixOperation::Transpose => 7,
                        PreparedMatrixOperation::SumRows(_) => 8,
                        PreparedMatrixOperation::Slice { .. } => 9,
                        PreparedMatrixOperation::Tensor { groups: None, .. } => 10,
                        PreparedMatrixOperation::Tensor { groups: Some(_), .. } => 11,
                    },
                    match operation {
                        PreparedMatrixOperation::SumRows(rows) |
                        PreparedMatrixOperation::Tensor { groups: Some(rows), .. } => {
                            rows.as_slice()
                        }
                        _ => &[],
                    },
                    match operation {
                        PreparedMatrixOperation::Slice { rows, columns } => {
                            Some((rows.start, rows.end, columns.start, columns.end))
                        }
                        _ => None,
                    },
                    match operation {
                        PreparedMatrixOperation::ConcatColumns {
                            diagonal,
                            rows,
                            columns,
                            offsets,
                        } => Some((*diagonal, *rows, *columns, offsets)),
                        _ => None,
                    },
                    match operation {
                        PreparedMatrixOperation::Accumulate { products, bias, rows } => {
                            Some((products, *bias, *rows))
                        }
                        _ => None,
                    },
                    match operation {
                        PreparedMatrixOperation::Constant { ty, value } => {
                            use mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixRangeConstant;
                            let fields = match value {
                                GpuMatrixRangeConstant::Zero { total_columns } => {
                                    (0u8, *total_columns, 0, false, None)
                                }
                                GpuMatrixRangeConstant::Identity => (1, ty.columns, 0, false, None),
                                GpuMatrixRangeConstant::UnitRow { total_columns, index } => {
                                    (2, *total_columns, *index, false, None)
                                }
                                GpuMatrixRangeConstant::UnitColumn { index } => {
                                    (3, 1, *index, false, None)
                                }
                                GpuMatrixRangeConstant::Gadget { small, digit_count } => {
                                    (4, ty.columns, 0, *small, *digit_count)
                                }
                            };
                            Some((ty, fields))
                        }
                        _ => None,
                    },
                    match operation {
                        PreparedMatrixOperation::Sample { ty, distribution, sigma_bits, max_coefficient_bound } => {
                            let kind = match distribution {
                                mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixSampleDist::Uniform => 0u8,
                                mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixSampleDist::Gauss => 1,
                                mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixSampleDist::Bit => 2,
                                mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixSampleDist::Ternary => 3,
                            };
                            Some((ty, kind, sigma_bits, max_coefficient_bound))
                        }
                        _ => None,
                    },
                    match operation {
                        PreparedMatrixOperation::Hash { ty, tag_bytes } => Some((ty, tag_bytes)),
                        _ => None,
                    },
                    (
                    match operation {
                        PreparedMatrixOperation::Polynomial { ty, coefficients } => Some((ty,Some(coefficients),None)),
                        PreparedMatrixOperation::CenteredRebase { destination } => Some((destination,None,None)),
                        PreparedMatrixOperation::ModulusConversion {destination,conversion}=>Some((destination,None,Some(match conversion {
                            mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixModulusConversion::Reduce=>(0u8,0u64),
                            mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixModulusConversion::Round=>(1,0),
                            mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixModulusConversion::CenteredExtend=>(2,0),
                            mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixModulusConversion::BlockSwitch {plaintext_modulus}=>(3,*plaintext_modulus),
                        }))),
                        _ => None,
                    },
                    match operation {
                        PreparedMatrixOperation::RnsConversion { destination, source_moduli, conversion } => Some((destination, source_moduli, match conversion {
                            mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixRnsConversion::Up { digit_size, normalize } => (0u8, *digit_size, *normalize, 0u64),
                            mxx_primitives::matrix::gpu_dcrt_poly::GpuMatrixRnsConversion::Down { plaintext_modulus } => (1, 0, false, *plaintext_modulus),
                        })),
                        _ => None,
                    }),
                    ((layout_key, match operation { PreparedMatrixOperation::Decompose {small,digit_count,input_rows,layout,hash} => Some((small,digit_count,input_rows,layout.rows_per_input_row,layout.dropped_moduli,&layout.max_coefficient_bound,hash)), _=>None }), match operation {
                        PreparedMatrixOperation::CrtRecompose {destination,plaintext_moduli,reconstruction_coefficients} => Some((destination,plaintext_moduli,reconstruction_coefficients)),
                        _ => None,
                    }, match operation {
                        PreparedMatrixOperation::CenteredExtendCompact { destination, bound } => Some((destination, bound)),
                        _ => None,
                    }, match operation {
                        PreparedMatrixOperation::MultiplyCompact { columns, inner } => Some((columns, inner)),
                        _ => None,
                    }, match operation {
                        PreparedMatrixOperation::ImportMatrix { ty, evaluation, max_coefficient_bits } => Some((ty, *evaluation, *max_coefficient_bits, None)),
                        PreparedMatrixOperation::ImportCompact { ty, bound } => Some((ty, false, 0u16, Some(bound))),
                        PreparedMatrixOperation::ImportStaging { ty, evaluation, .. } => Some((ty, *evaluation, 0u16, None)),
                        _ => None,
                    }),
                    match operation {
                        PreparedMatrixOperation::ImportStaging { bytes_per_poly, payload_len, .. } => Some((bytes_per_poly, payload_len)),
                        _ => None,
                    },
                    match operation {
                        PreparedMatrixOperation::Preimage { ty, bound, sigma_bits, gadget_base, digit_count, public_rows, plan } => Some((ty, bound, sigma_bits, gadget_base, digit_count, public_rows, plan.attempts, plan.attempt.keys().copied().collect::<Vec<_>>())),
                        _ => None,
                    },
                    configuration,
                ))
                .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
                let class = GpuAllocationClass {
                    identity,
                    bound_identity: Some(identity),
                    minimum_columns: 1,
                    maximum_columns: maximum,
                };
                let key = GpuCalibrationKey::new(
                    identity.as_slice(),
                    environment.clone(),
                    class,
                    GpuCalibrationMetric::PreparedOccupiedSpanBytes {
                        storage_configuration: configuration,
                    },
                );
                if let Some(profile) = self.calibration_registry.get(&key) {
                    let needs_zero =
                        requirements.outputs.iter().any(|output| output.interval.device == 0);
                    let needs_nonzero =
                        requirements.outputs.iter().any(|output| output.interval.device > 0);
                    if (!needs_zero || profile.gpu0.is_some()) &&
                        (!needs_nonzero || profile.nonzero.is_some())
                    {
                        // A profile avoids only measurement. The ledger below
                        // still acquires complete outputs and rechecks every
                        // active device's live charges and native scratch fit.
                        profiles.push((*profile).clone());
                        continue;
                    }
                }
                let operation = operation.clone();
                let pilot_payload = operation.pilot_payload()?;
                // The ledger is held locally during admission; brokers see the
                // same accepted inventory the production runners will use.
                let brokers = Arc::new(
                    (0..self.devices.len())
                        .map(|device| {
                            PreparedClaimBroker::new(
                                inventory
                                    .iter()
                                    .filter(|(owner, _)| *owner == device)
                                    .map(|(_, storage)| storage.clone())
                                    .collect(),
                            )
                        })
                        .collect::<Vec<_>>(),
                );
                let left = left.clone();
                let right = right.clone();
                let compact = compact.clone();
                let requirements = requirements.clone();
                let inventory = inventory.clone();
                let prepared = prepared.clone();
                let measured = self
                    .enqueue
                    .map(&mut self.devices, move |device, (_, _)| {
                        let local = requirements
                            .outputs
                            .iter()
                            .chain(requirements.representative.iter())
                            .filter(|output| output.interval.device == device)
                            .collect::<Vec<_>>();
                        if device > 1 || local.is_empty() {
                            return Ok(None);
                        }
                        let params = &local[0].parameters;
                        // The only input waits are at the explicit pilot boundary.
                        let mut ready = HashSet::new();
                        for output in &local {
                            if let Some((primary, index)) = operation
                                .primary(left.as_ref(), &right, output.interval.start)
                                .zip(output.source)
                            {
                                if ready.insert((primary.id, index)) {
                                    operation
                                        .source(&prepared, primary, index)
                                        .value
                                        .wait_until_ready();
                                }
                            }
                            for (right, &index) in right.iter().zip(&output.right) {
                                if ready.insert((right.id, index)) {
                                    operation
                                        .source(&prepared, right, index)
                                        .value
                                        .wait_until_ready();
                                }
                            }
                        }
                        params.fence_released_memory();
                        let mut groups = local
                            .iter()
                            .map(|output| (&*output.storage, std::slice::from_ref(&output.request)))
                            .collect::<Vec<_>>();
                        let scratch = requirements.scratch_allocations(device, 1);
                        groups.extend(
                            scratch
                                .prepared
                                .iter()
                                .map(|(storage, requests)| (*storage, requests.as_slice())),
                        );
                        // Broker-held pilot steps (trapdoor, target tile,
                        // destination plan, one attempt) count toward demand.
                        let step_claims = operation.pilot_step_claims();
                        let mut step_requests: Vec<(
                            Arc<GpuPreparedStorage>,
                            Vec<GpuPreparedRequest>,
                        )> = Vec::new();
                        {
                            let mut taken = HashSet::new();
                            for claim in &step_claims {
                                let mut best: Option<(
                                    usize,
                                    Arc<GpuPreparedStorage>,
                                    GpuPreparedRequest,
                                    u64,
                                )> = None;
                                for (owner, storage) in inventory.iter() {
                                    if *owner != device {
                                        continue;
                                    }
                                    for index in 0..storage.slot_count() {
                                        let slot = storage.slot_identity(index).unwrap();
                                        if taken.contains(&slot.slot_id()) ||
                                            slot.kind() != claim.kind()
                                        {
                                            continue;
                                        }
                                        let request = if claim.kind() == GpuPreparedSlotKind::Matrix
                                        {
                                            if slot.level() != claim.level() {
                                                continue;
                                            }
                                            slot.matrix_request(
                                                claim.rows(),
                                                claim.columns(),
                                                claim.is_evaluation().unwrap_or(true),
                                            )
                                        } else {
                                            slot.workspace_request(
                                                claim.bytes(),
                                                claim.alignment().max(1),
                                            )
                                        };
                                        if storage
                                            .fits(&[request])
                                            .map_err(PolyBackendError::GpuCalibration)? &&
                                            best.as_ref().is_none_or(|(bytes, _, _, _)| {
                                                slot.requested_backing_bytes() < *bytes
                                            })
                                        {
                                            best = Some((
                                                slot.requested_backing_bytes(),
                                                storage.clone(),
                                                request,
                                                slot.slot_id(),
                                            ));
                                        }
                                    }
                                }
                                let (_, storage, request, slot_id) = best.ok_or_else(|| {
                                    PolyBackendError::GpuSubmission(
                                        "pilot step claims do not fit the prepared inventory"
                                            .into(),
                                    )
                                })?;
                                taken.insert(slot_id);
                                step_requests.push((storage, vec![request]));
                            }
                        }
                        let steps = step_requests
                            .iter()
                            .map(|(storage, requests)| (&**storage, requests.as_slice()))
                            .collect::<Vec<_>>();
                        for (_, store) in inventory.iter().filter(|(owner, _)| *owner == device) {
                            if !groups
                                .iter()
                                .any(|(included, _)| included.identity() == store.identity())
                            {
                                groups.push((store, &[]));
                            }
                        }
                        let measured = GpuDeviceCalibration::measure_prepared_with_steps(
                            class,
                            1,
                            configuration,
                            &groups,
                            &steps,
                            || {
                                let mut outputs = local
                                    .iter()
                                    .map(|output| {
                                        operation
                                            .initialize_output(
                                                &output.parameters,
                                                output.level,
                                                output.evaluation,
                                                rows,
                                                output.interval.end - output.interval.start,
                                            )
                                            .map(Some)
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                let first = local[0];
                                let destination = outputs[0].take().unwrap();
                                outputs[0] = Some(
                                    operation.run(
                                        operation
                                            .primary(left.as_ref(), &right, first.interval.start)
                                            .zip(first.source)
                                            .map(|(primary, index)| {
                                                operation.source(&prepared, primary, index)
                                            }),
                                        &right
                                            .iter()
                                            .zip(&first.right)
                                            .map(|(right, &index)| {
                                                operation.source(&prepared, right, index)
                                            })
                                            .collect::<Vec<_>>(),
                                        CompiledMatrixInvocation::compact_shard(
                                            compact.as_ref(),
                                            first.source,
                                        ),
                                        first.interval.start,
                                        first.interval.start + 1,
                                        (destination, 0..rows, 0..1),
                                        &pilot_payload,
                                        &brokers[device],
                                    )?,
                                );
                                for output in &outputs {
                                    output.as_ref().unwrap().wait_until_ready();
                                }
                                Ok(outputs)
                            },
                        );
                        let result = measured
                            .map(|(profile, outputs)| {
                                drop(outputs);
                                Some(profile)
                            })
                            .map_err(PolyBackendError::GpuCalibration);
                        params.fence_released_memory();
                        result
                    })
                    .map_err(PolyBackendError::from)?;
                let profile = GpuCalibrationProfile {
                    gpu0: measured[0],
                    nonzero: measured.get(1).copied().flatten(),
                };
                let registry = crate::gpu_calibration::GpuCalibrationRegistry::from(
                    self.calibration_registry.clone(),
                );
                registry
                    .insert(key, profile.clone())
                    .map_err(|error| PolyBackendError::GpuCalibration(error.to_string()))?;
                self.calibration_registry = registry.freeze();
                profiles.push(profile);
            }
            let mut admitted = Vec::with_capacity(layouts.len());
            for ((operation, left, _, _, requirements), profile) in layouts.iter().zip(&profiles) {
                let intervals =
                    requirements.outputs.iter().map(|output| output.interval).collect::<Vec<_>>();
                let plan = ledger
                    .reserve_columns(
                        operation.output_columns(left.as_ref()),
                        GpuOutputOwnership::Inherited(&intervals),
                        profile,
                        &**requirements,
                    )
                    .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
                // Original requests remain borrowed and are validated against
                // their frozen operation/layout before the batch is published.
                let request = &requests[admitted.len()].1;
                admitted.push((request, plan));
            }
            self.prepared_invocations = self.compile_matrix_invocations(admitted, prepared)?;
            // Only an identity the executor selected for this admission keys
            // the log; admissions without a fresh selection are not logged and
            // the estimator measures those nodes nominally.
            if let Some(operation_identity) = self.unlogged_operation.take() {
                for summary in self.admitted_invocation_summaries() {
                    self.admitted_plan_log.push((operation_identity, summary));
                }
            }
            self.prepared_required = true;
            self.pending_pilot = None;
            self.pending_profile = None;
            Ok(())
        })();
        self.prepared_ledger = Some(ledger);
        result
    }
}
