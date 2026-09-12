//! Prepared matrix preflight: select native slots, fit scratch using allocation
//! bounds, reserve every retained destination, then publish the compiled batch.

use super::{
    gpu_compiled::{MatrixInvocation, PreparedClaimBroker},
    gpu_prepare::{
        MatrixInputPreparation, PreparedMatrixSource, select_matrix_input, select_prepared_matrix,
    },
    *,
};
use crate::gpu_memory::{
    GpuAdmissionError, GpuColumnAllocations, GpuColumnMemoryRequirements, GpuColumnWidthPolicy,
    GpuMemoryLedger, GpuOutputOwnership,
};
use mxx_primitives::matrix::gpu_dcrt_poly::{
    GpuPreparedRequest, GpuPreparedSlotIdentity, GpuPreparedSlotKind, GpuPreparedStorage,
    GpuPreparedWorkspaceLayout,
};

struct MatrixDestination {
    interval: GpuColumnInterval,
    parameters: GpuDCRTPolyParams,
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
                            let rank = |slot: &GpuPreparedSlotIdentity| {
                                (
                                    super::gpu_prepare::capacity_class(
                                        slot.requested_backing_bytes(),
                                    ) != super::gpu_prepare::capacity_class(layout.bytes),
                                    slot.requested_backing_bytes(),
                                )
                            };
                            rank(&slot) < rank(previous)
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
    /// native slots, fits native allocation bounds and acquires their plans.
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
            // Validate the entire batch and its native fits before preparing inputs.
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
                    PolyBackendError::GpuSubmission(format!(
                        "{}: complete matrix inputs and {}x{} outputs do not fit the prepared inventory",
                        operation.kind_name(),
                        rows,
                        columns
                    ))
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
                            if let Some(left) = left {
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
                            parameters: trial.parameters,
                            storage,
                            request,
                        });
                        start += count;
                    }
                } else if let Some(compact) = compact {
                    // Compact inputs inherit their resident column ownership. No
                    // replica or fragment preparation exists for compact payloads.
                    for shard in compact.shards.iter() {
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
                            parameters,
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
                                Arc::new(MatrixRequirements { outputs, scratch }),
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
                        select_matrix_input(
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
                            parameters,
                            storage,
                            request,
                        });
                    }
                }
                layouts.push((
                    operation,
                    left.cloned(),
                    right,
                    compact.cloned(),
                    Arc::new(MatrixRequirements { outputs, scratch }),
                ));
            }
            // All layouts and fixed/output fits are known before preparation.
            // Native owners retain the fixed slots after their tokens retire,
            // and remain live while scratch widths are selected.
            let prepared = if self.admitted_measurement_sink.is_some() && !preparation.is_empty() {
                let mut sink = self.admitted_measurement_sink.take().unwrap();
                let measured = (|| {
                    let device_indices = self
                        .devices
                        .iter()
                        .enumerate()
                        .map(|(index, (device, _))| (*device, index))
                        .collect::<HashMap<_, _>>();
                    let active_devices = preparation
                        .par_iter()
                        .flat_map_iter(|input| {
                            let sources = match input.source {
                                PreparedMatrixSource::Shard(index) |
                                PreparedMatrixSource::Fragment { index, .. } => {
                                    &input.matrix.shards[index..index + 1]
                                }
                                PreparedMatrixSource::Replica { .. } => {
                                    input.matrix.shards.as_slice()
                                }
                            };
                            std::iter::once(input.device).chain(
                                sources.iter().map(|source| device_indices[&source.device_id]),
                            )
                        })
                        .collect::<HashSet<_>>();
                    let parameters = self
                        .device_parameters()
                        .into_iter()
                        .enumerate()
                        .filter(|(device, _)| active_devices.contains(device))
                        .collect();
                    let stores = inventory.iter().fold(
                        std::collections::BTreeMap::<usize, Vec<Arc<GpuPreparedStorage>>>::new(),
                        |mut stores, (device, storage)| {
                            stores.entry(*device).or_default().push(storage.clone());
                            stores
                        },
                    );
                    let mut measurement = crate::gpu_measurement::GpuColumnMeasurement::new(
                        parameters, stores, &mut sink, None, true,
                    );
                    let timer = measurement
                        .begin(None)
                        .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
                    let prepared = self.prepare_matrix_inputs(&mut ledger, preparation)?;
                    measurement
                        .finish(
                            timer,
                            crate::gpu_measurement::GpuMeasuredStage::InputPreparation(
                                self.active_operation,
                            ),
                        )
                        .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
                    Ok::<_, PolyBackendError>(prepared)
                })();
                self.admitted_measurement_sink = Some(sink);
                measured?
            } else {
                self.prepare_matrix_inputs(&mut ledger, preparation)?
            };
            let mut classes = Vec::with_capacity(layouts.len());
            // Native layout bounds determine scratch widths without calibration kernels.
            for (operation, _, _, _, requirements) in &layouts {
                if requirements.outputs.is_empty() {
                    classes.push(GpuAllocationClass {
                        identity: [0; 32],
                        bound_identity: None,
                        minimum_columns: 1,
                        maximum_columns: 1,
                    });
                    continue;
                }
                let mut maximum = requirements
                    .outputs
                    .iter()
                    .map(|output| output.interval.end - output.interval.start)
                    .max()
                    .unwrap();
                if let PreparedMatrixOperation::Preimage { ty, bound, public_rows, plan, .. } =
                    operation
                {
                    // Step claims use the same accepted inventory as fixed and
                    // retained output owners. Bound the native allocation class
                    // before width selection, including the arbitrary-width tail.
                    for output in &requirements.outputs {
                        let params = &output.parameters;
                        let broker = PreparedClaimBroker::new(
                            params,
                            inventory.iter().map(|(_, storage)| storage.clone()).collect(),
                        );
                        let mut low = 0usize;
                        let mut high = maximum;
                        while low < high {
                            let width = low + (high - low).div_ceil(2);
                            let mut claims = plan.destination.clone();
                            claims.extend(
                                plan.tile_claims(params, *public_rows, width)
                                    .map_err(PolyBackendError::GpuCalibration)?,
                            );
                            claims.extend(
                                plan.attempt_claims(params, *public_rows, width, ty.rows, bound)
                                    .map_err(PolyBackendError::GpuCalibration)?,
                            );
                            if broker
                                .fits(&claims, &chosen)
                                .map_err(PolyBackendError::GpuCalibration)?
                            {
                                low = width;
                            } else {
                                high = width - 1;
                            }
                        }
                        if low == 0 {
                            return Err(PolyBackendError::GpuSubmission(
                                "preimage one-column step demand does not fit the prepared inventory".into()));
                        }
                        maximum = low;
                    }
                }
                classes.push(GpuAllocationClass {
                    identity: self.active_operation.unwrap_or([0; 32]),
                    bound_identity: None,
                    minimum_columns: 1,
                    maximum_columns: maximum,
                });
            }
            let mut admitted = Vec::with_capacity(layouts.len());
            for ((operation, left, _, _, requirements), class) in layouts.iter().zip(&classes) {
                let intervals =
                    requirements.outputs.iter().map(|output| output.interval).collect::<Vec<_>>();
                let plan = ledger
                    .reserve_columns(
                        operation.output_columns(left.as_ref()),
                        GpuOutputOwnership::Inherited(&intervals),
                        GpuColumnWidthPolicy::Native(*class),
                        &**requirements,
                    )
                    .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
                // Original requests remain borrowed and are validated against
                // their frozen operation/layout before the batch is published.
                let request = &requests[admitted.len()].1;
                admitted.push((request, plan));
            }
            self.prepared_invocations = self.compile_matrix_invocations(admitted, prepared)?;
            // Retain distinct diagnostic plans in first-admission order. The
            // measurement observer owns execution counts; repeated loop members
            // must not append identical plan snapshots indefinitely.
            if let Some(operation_identity) = self.unlogged_operation.take() {
                for summary in self.admitted_invocation_summaries() {
                    let entry = (operation_identity, summary);
                    if !self.admitted_plan_log.contains(&entry) {
                        self.admitted_plan_log.push(entry);
                    }
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
