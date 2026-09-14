//! Prepared matrix preflight: select native slots, fit scratch using allocation
//! bounds, reserve every retained destination, then publish the compiled batch.

use super::{
    gpu_compiled::{AdmittedMatrixBinding, InvocationOperands, LoweredMatrixInvocation},
    gpu_prepare::{
        MatrixDescriptor, MatrixInputPreparation, MatrixInputRequest, MatrixSlotContext,
        MatrixSlotKey, PreparedMatrixSource, select_matrix_input, select_prepared_matrix,
    },
    *,
};
use crate::gpu_memory::{
    GpuAdmissionError, GpuColumnAllocations, GpuColumnMemoryRequirements, GpuColumnWidthPolicy,
    GpuMemoryLedger, GpuOutputOwnership,
};
use mxx_primitives::matrix::gpu_dcrt_poly::{
    GpuPreparedRequest, GpuPreparedSlotIdentity, GpuPreparedSlotKind, GpuPreparedSlotSnapshot,
    GpuPreparedStorage, GpuPreparedWorkspaceLayout, GpuTracedClaim,
};

/// Complete metadata consumed by shared destination/input/scratch selection.
pub struct MatrixPlacementInvocation {
    operation: PreparedMatrixOperation,
    operands: InvocationOperands<MatrixDescriptor, MatrixDescriptor>,
    input_layouts: super::gpu_prepare::MatrixInputLayouts,
}

impl MatrixPlacementInvocation {
    /// Bind explicit input placement descriptors in validated IR argument order.
    /// None identifies a node without a single prepared matrix operation; it is
    /// not permission to replace an unknown placement with fresh GPU inputs.
    pub fn new(
        backend: &GpuDcrtBackend,
        node: &crate::gpu_invocation::GpuNodeOperation,
        arguments: Vec<Option<MatrixDescriptor>>,
    ) -> Result<Option<Self>, PolyBackendError> {
        use mxx_ir_core::types::ConcreteWireType;
        let Some(operation) = CompiledMatrixInvocation::lower_ir(node, backend)? else {
            return Ok(None);
        };
        let mut ordinary = Vec::new();
        let mut compact = None;
        let input_count = if matches!(operation, PreparedMatrixOperation::Preimage { .. }) {
            1
        } else {
            usize::MAX
        };
        for (argument, ty) in arguments.into_iter().zip(node.arguments()).take(input_count) {
            if operation.fresh_type().is_some() &&
                !matches!(operation, PreparedMatrixOperation::Preimage { .. })
            {
                continue;
            }
            match ty {
                ConcreteWireType::Matrix(_) => ordinary.push(argument.ok_or_else(|| {
                    PolyBackendError::GpuSubmission("matrix input placement is unresolved".into())
                })?),
                ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => {
                    compact = argument;
                }
                _ => {}
            }
        }
        let operands = if matches!(operation, PreparedMatrixOperation::Preimage { .. }) {
            InvocationOperands {
                left: ordinary.into_iter().next(),
                right: Vec::new(),
                compact: None,
            }
        } else if operation.fresh_type().is_some() {
            InvocationOperands { left: None, right: Vec::new(), compact: None }
        } else if matches!(
            operation,
            PreparedMatrixOperation::MultiplyCompact { .. } |
                PreparedMatrixOperation::CenteredExtendCompact { .. }
        ) {
            let compact = compact.ok_or_else(|| {
                PolyBackendError::GpuSubmission("compact input placement is unresolved".into())
            })?;
            InvocationOperands { left: None, right: ordinary, compact: Some(compact) }
        } else {
            if matches!(operation, PreparedMatrixOperation::Multiply { scales_left: false }) {
                ordinary.swap(0, 1);
            }
            if let PreparedMatrixOperation::Accumulate { products, .. } = &operation {
                for (index, (_, scales_left)) in products.iter().enumerate() {
                    if !scales_left {
                        ordinary.swap(2 * index, 2 * index + 1);
                    }
                }
            }
            let mut ordinary = ordinary.into_iter();
            InvocationOperands { left: ordinary.next(), right: ordinary.collect(), compact: None }
        };
        Ok(Some(Self { operation, operands, input_layouts: HashMap::new() }))
    }
}

/// Fully selected input preparations, retained outputs and reusable workspaces.
/// This contains no GPU values or leases. Fitting it never acquires resources.
pub struct GpuMatrixLayoutPlan {
    layouts: Vec<(usize, Arc<MatrixRequirements>)>,
    classes: Vec<GpuAllocationClass>,
    pub inputs: Vec<MatrixInputRequest>,
    invocations: Vec<MatrixInvocationGeometry>,
}

struct MatrixInvocationGeometry {
    operation: &'static str,
    rows: usize,
    columns: usize,
    compact_output: bool,
    input_owners: Vec<usize>,
    compact_input_owner: Option<usize>,
}

impl GpuMatrixLayoutPlan {
    /// Fit each selected invocation using its ring's eligible inventory. Views
    /// are in invocation order; context-local hypothetical IDs never cross rings.
    pub fn fit<I: crate::gpu_memory::GpuColumnInventory>(
        &self,
        admissions: &[crate::gpu_memory::GpuDeviceAdmission],
        column_cap: usize,
        inventories: &[I],
    ) -> Result<Vec<super::gpu_compiled::GpuAdmittedInvocationSummary>, GpuAdmissionError> {
        self.invocations
            .par_iter()
            .enumerate()
            .map(|(index, invocation)| {
                let requirements = &self.layouts[index].1;
                let intervals =
                    requirements.outputs.iter().map(|output| output.interval).collect::<Vec<_>>();
                let fit = crate::gpu_memory::GpuColumnFit::new(
                    admissions,
                    column_cap,
                    &inventories[index],
                    invocation.columns,
                    GpuOutputOwnership::Inherited(&intervals),
                    GpuColumnWidthPolicy::Native(self.classes[index]),
                    &requirements.resources,
                )?;
                Ok(super::gpu_compiled::GpuAdmittedInvocationSummary {
                    operation: invocation.operation,
                    rows: invocation.rows,
                    columns: invocation.columns,
                    compact_output: invocation.compact_output,
                    input_owners: invocation.input_owners.clone(),
                    compact_input_owner: invocation.compact_input_owner,
                    plan: fit.summary(),
                })
            })
            .collect()
    }

    /// Exact selected output owners for binding the model's subsequent values.
    /// The context is explicit because unbacked storage IDs are context-local.
    pub fn outputs(
        &self,
        invocation: usize,
    ) -> Vec<(GpuColumnInterval, super::gpu_prepare::MatrixFragmentDescriptor, GpuPreparedRequest)>
    {
        self.layouts[invocation]
            .1
            .outputs
            .iter()
            .map(|output| {
                (
                    output.interval,
                    super::gpu_prepare::MatrixFragmentDescriptor {
                        parameters: output.sources.parameters.clone(),
                        level: output.sources.level,
                        columns: output.interval.end - output.interval.start,
                        evaluation: output.sources.evaluation,
                    },
                    output.request,
                )
            })
            .collect()
    }
}

impl From<&LoweredMatrixInvocation> for MatrixPlacementInvocation {
    fn from(invocation: &LoweredMatrixInvocation) -> Self {
        Self {
            operation: invocation.operation.clone(),
            operands: InvocationOperands {
                left: invocation.operands.left.as_ref().map(MatrixDescriptor::from),
                right: invocation.operands.right.iter().map(MatrixDescriptor::from).collect(),
                compact: invocation.operands.compact.as_ref().map(MatrixDescriptor::from),
            },
            input_layouts: invocation.input_layouts.clone(),
        }
    }
}

struct MatrixDestination {
    sources: AdmittedMatrixBinding,
    interval: GpuColumnInterval,
    request: GpuPreparedRequest,
}

struct MatrixScratch {
    device: usize,
    rows: usize,
    parameters: GpuDCRTPolyParams,
    slot: GpuPreparedSlotIdentity,
    evaluation: bool,
    workspace: Option<GpuPreparedWorkspaceLayout>,
    /// Width-scaled native workspace: the operation, context, level and the
    /// index into its `width_workspaces` for the requested column count.
    width: Option<(PreparedMatrixOperation, GpuDCRTPolyParams, usize, usize)>,
}

fn select_operation_scratch(
    operation: &PreparedMatrixOperation,
    inventory: &super::gpu_prepare::MatrixSlotInventory,
    deferred: &HashSet<u64>,
    chosen: &mut HashSet<MatrixSlotKey>,
    device: usize,
    parameters: &GpuDCRTPolyParams,
    level: usize,
    evaluation: bool,
) -> Result<Option<Vec<MatrixScratch>>, PolyBackendError> {
    let slots = inventory
        .iter()
        .filter(|entry| entry.device == device && entry.context == parameters.context_identity())
        .flat_map(|entry| {
            entry.slots.iter().map(|&(slot, available)| {
                let id = slot.identity();
                let available = available || deferred.contains(&id.slot_id());
                (
                    slot,
                    available &&
                        !chosen.contains(&(
                            parameters.context_identity(),
                            (id.storage_id(), id.slot_id(), id.slot_index()),
                        )),
                )
            })
        })
        .collect();
    let context =
        GpuMatrixColumnContext { device, parameters: parameters.clone(), level, evaluation, slots };
    let Some(scratch) = operation.select_column_scratch(context)? else { return Ok(None) };
    chosen.extend(scratch.iter().map(|scratch| {
        let slot = scratch.slot;
        (parameters.context_identity(), (slot.storage_id(), slot.slot_id(), slot.slot_index()))
    }));
    Ok(Some(scratch))
}

struct MatrixDeviceInputPlan {
    sources: AdmittedMatrixBinding,
    device: usize,
    capacity: usize,
    chosen: HashSet<MatrixSlotKey>,
    planned: HashSet<(u64, PreparedMatrixSource)>,
    preparation: Vec<MatrixInputRequest>,
    scratch: Vec<MatrixScratch>,
}

pub(super) struct MatrixRequirements {
    outputs: Vec<MatrixDestination>,
    resources: GpuMatrixColumnRequirements,
}

impl MatrixRequirements {
    fn new(outputs: Vec<MatrixDestination>, scratch: Vec<MatrixScratch>) -> Self {
        let resources = GpuMatrixColumnRequirements {
            outputs: outputs.iter().map(|output| (output.interval, output.request)).collect(),
            scratch,
        };
        Self { outputs, resources }
    }
}

/// The selected destinations and width-dependent scratch of a matrix invocation.
/// This owns CPU layout metadata only, with no matrix values, storage or leases.
/// Production and hypothetical fitting consume this same resource provider.
pub struct GpuMatrixColumnRequirements {
    outputs: Vec<(GpuColumnInterval, GpuPreparedRequest)>,
    scratch: Vec<MatrixScratch>,
}

/// One owner's CPU scratch-selection scenario. Eligibility excludes retained
/// owners and pending readers. Destination slots are excluded by the provider;
/// no native backing or execution permission is created by this metadata.
pub struct GpuMatrixColumnContext {
    pub device: usize,
    pub parameters: GpuDCRTPolyParams,
    pub level: usize,
    pub evaluation: bool,
    pub slots: Vec<(GpuPreparedSlotSnapshot, bool)>,
}

impl PreparedMatrixOperation {
    /// One CPU-only scratch assignment for both native and hypothetical plans.
    /// The caller supplies availability, including any explicitly accepted
    /// deferred upload resources. Selection never waits or acquires a lease.
    fn select_column_scratch(
        &self,
        context: GpuMatrixColumnContext,
    ) -> Result<Option<Vec<MatrixScratch>>, PolyBackendError> {
        let (claims, width_start) =
            self.column_scratch_claims(&context.parameters, context.level, context.evaluation)?;
        let requests =
            GpuPreparedSlotSnapshot::assign(&context.parameters, &context.slots, &claims)
                .map_err(PolyBackendError::GpuSubmission)?;
        let Some(requests) = requests.into_iter().collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        let slots = requests
            .into_iter()
            .map(|request| {
                context
                    .slots
                    .iter()
                    .find(|(slot, _)| {
                        let id = slot.identity();
                        (id.storage_id(), id.slot_id(), id.slot_index()) == request.slot_key()
                    })
                    .expect("selected request belongs to the supplied context")
                    .0
                    .identity()
            })
            .collect();
        self.column_scratch(
            context.device,
            context.parameters,
            context.level,
            context.evaluation,
            slots,
            &claims,
            width_start,
        )
        .map(Some)
    }

    fn column_scratch_claims(
        &self,
        parameters: &GpuDCRTPolyParams,
        level: usize,
        evaluation: bool,
    ) -> Result<(Vec<GpuTracedClaim>, usize), PolyBackendError> {
        let rows = self.scratch_rows()?;
        let fixed = self.fixed_workspaces(parameters, level)?;
        let width = self.width_workspaces(parameters, level, 1)?;
        let width_start = rows.len() + fixed.len();
        let claims = rows
            .into_iter()
            .map(|rows| GpuTracedClaim::matrix(rows, 1, level, evaluation))
            .chain(fixed.into_iter().chain(width).map(GpuTracedClaim::workspace))
            .collect();
        Ok((claims, width_start))
    }

    fn column_scratch(
        &self,
        device: usize,
        parameters: GpuDCRTPolyParams,
        level: usize,
        evaluation: bool,
        scratch_slots: Vec<GpuPreparedSlotIdentity>,
        claims: &[GpuTracedClaim],
        width_start: usize,
    ) -> Result<Vec<MatrixScratch>, PolyBackendError> {
        // Selected slots and operation layout are one internal lowering contract.
        // Do not silently omit a resource if a warmed template changes.
        if scratch_slots.len() != claims.len() {
            return Err(PolyBackendError::GpuSubmission(
                "selected scratch slots do not match the operation resource layout".into(),
            ));
        }
        Ok(scratch_slots
            .into_iter()
            .enumerate()
            .map(|(index, slot)| MatrixScratch {
                device,
                parameters: parameters.clone(),
                rows: claims[index].rows(),
                slot,
                evaluation,
                workspace: (index < width_start).then(|| claims[index].layout()).flatten(),
                width: (index >= width_start)
                    .then(|| (self.clone(), parameters.clone(), level, index - width_start)),
            })
            .collect())
    }
}

impl GpuMatrixColumnRequirements {
    fn scratch_allocations(
        &self,
        device: usize,
        columns: usize,
    ) -> Result<GpuColumnAllocations, GpuCalibrationError> {
        Ok(GpuColumnAllocations {
            managed_bounds: Vec::new(),
            prepared: self
                .scratch
                .iter()
                .filter(|scratch| scratch.device == device)
                .map(|scratch| {
                    Ok((
                        scratch.slot.storage_id(),
                        vec![match (&scratch.workspace, &scratch.width) {
                            (Some(layout), _) => {
                                scratch.slot.workspace_request(layout.bytes, layout.alignment)
                            }
                            (None, Some((operation, parameters, level, index))) => {
                                // A missing warmup resource class or failed CPU
                                // query is an error, not a fabricated oversized request.
                                let layouts = operation
                                    .width_workspaces(parameters, *level, columns)
                                    .map_err(|error| {
                                        GpuCalibrationError::NativeFit(error.to_string())
                                    })?;
                                let layout = layouts.get(*index).ok_or_else(|| {
                                    GpuCalibrationError::NativeFit(
                                        "width workspace layout changed after scratch selection"
                                            .into(),
                                    )
                                })?;
                                scratch.slot.workspace_request(layout.bytes, layout.alignment)
                            }
                            (None, None) => scratch.slot.matrix_request(
                                scratch.rows,
                                columns,
                                scratch.evaluation,
                            ),
                        }],
                    ))
                })
                .collect::<Result<Vec<_>, GpuCalibrationError>>()?,
        })
    }
}

impl GpuColumnMemoryRequirements for GpuMatrixColumnRequirements {
    fn fixed_allocations(&self, _: usize) -> Result<GpuColumnAllocations, GpuAdmissionError> {
        Ok(GpuColumnAllocations::default())
    }
    fn minimum_temporary_allocations(
        &self,
        device: usize,
    ) -> Result<GpuColumnAllocations, GpuAdmissionError> {
        Ok(self.scratch_allocations(device, 1)?)
    }
    fn output_bound(
        &self,
        device: usize,
        columns: usize,
    ) -> Result<GpuColumnAllocations, GpuAdmissionError> {
        let mut result = GpuColumnAllocations::default();
        if columns != 0 {
            // A monotone envelope for this frozen inherited layout. Repeated
            // storage groups are merged only in the bound, not dispatch order.
            for (_, request) in
                self.outputs.iter().filter(|(interval, _)| interval.device == device)
            {
                if let Some((_, requests)) =
                    result.prepared.iter_mut().find(|(store, _)| *store == request.slot_key().0)
                {
                    requests.push(*request);
                } else {
                    result.prepared.push((request.slot_key().0, vec![*request]));
                }
            }
        }
        Ok(result)
    }
    fn output_allocations(
        &self,
        device: usize,
        intervals: &[GpuColumnInterval],
    ) -> Result<GpuColumnAllocations, GpuAdmissionError> {
        if intervals.iter().ne(self.outputs.iter().map(|(interval, _)| interval)) {
            return Err(GpuAdmissionError::InvalidPlan(
                "matrix ownership changed after native output selection".into(),
            ));
        }
        Ok(GpuColumnAllocations {
            managed_bounds: Vec::new(),
            prepared: self
                .outputs
                .iter()
                .filter(|(interval, _)| interval.device == device)
                .map(|(_, request)| (request.slot_key().0, vec![*request]))
                .collect(),
        })
    }
    fn temporary_allocations(
        &self,
        device: usize,
        _: &GpuAllocationClass,
        columns: usize,
    ) -> Result<GpuColumnAllocations, GpuCalibrationError> {
        self.scratch_allocations(device, columns)
    }
    fn validate_ranges(
        &self,
        intervals: &[GpuColumnInterval],
        _: &[usize],
    ) -> Result<(), GpuAdmissionError> {
        if intervals.iter().ne(self.outputs.iter().map(|(interval, _)| interval)) {
            return Err(GpuAdmissionError::InvalidPlan(
                "matrix range boundaries changed after compilation".into(),
            ));
        }
        Ok(())
    }
}

impl GpuDcrtBackend {
    /// Bind selected destinations and fit scratch through production's typed
    /// claims and simultaneous slot matcher. This is CPU-only; the caller must
    /// supply proven destination intervals and an eligible context inventory.
    /// Containing input-layout bounds alone are not a source placement.
    pub fn matrix_column_requirements(
        &self,
        node: &crate::gpu_invocation::GpuNodeOperation,
        outputs: Vec<(GpuColumnInterval, GpuPreparedRequest)>,
        contexts: Vec<GpuMatrixColumnContext>,
    ) -> Result<GpuMatrixColumnRequirements, PolyBackendError> {
        let operation = CompiledMatrixInvocation::lower_ir(node, self)?.ok_or_else(|| {
            PolyBackendError::GpuSubmission("node has no prepared matrix operation".into())
        })?;
        let scratch = contexts
            .into_par_iter()
            .map(|mut context| {
                let output_slots = outputs.iter().filter(|(interval, _)| interval.device == context.device)
                    .map(|(_, request)| request.slot_key()).collect::<HashSet<_>>();
                context.slots.iter_mut().for_each(|(slot, eligible)| {
                    let id = slot.identity();
                    *eligible &= !output_slots.contains(&(id.storage_id(), id.slot_id(), id.slot_index()));
                });
                operation.select_column_scratch(context)?.ok_or_else(||
                    PolyBackendError::GpuSubmission(
                        "operation scratch does not fit around selected destinations and retained owners".into()))

            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect();
        Ok(GpuMatrixColumnRequirements { outputs, scratch })
    }

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

    /// CPU layout planning used by native admission, independent of GPU values.
    /// All inputs are descriptors and slot snapshots; no storage is acquired,
    /// no payload is created and no GPU command or event query is issued here.
    pub fn plan_matrix_layouts(
        &self,
        lowered: &[MatrixPlacementInvocation],
        matrix_inventory: &super::gpu_prepare::MatrixSlotInventory,
        deferred: &HashSet<u64>,
    ) -> Result<GpuMatrixLayoutPlan, PolyBackendError> {
        let mut chosen = HashSet::new();
        let mut preparation = Vec::new();
        let mut planned = HashSet::new();
        let mut layouts = Vec::with_capacity(lowered.len());
        // Validate the entire batch and its native fits before preparing inputs.
        // Selection is ordered because later outputs must exclude earlier
        // choices; acquiring real reservations below rechecks every slot.
        for invocation in lowered {
            let MatrixPlacementInvocation { operation, operands, .. } = invocation;
            let left = operands.left.as_ref();
            let right = &operands.right;
            let compact = operands.compact.as_ref();
            let rows = operation.output_rows(left, right)?;
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
                                first.value.registered_parameters(backend)?.clone(),
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
                        let left_source = if let Some(left) = left {
                            let Some(source) = select_matrix_input(
                                left,
                                0..left.columns,
                                device,
                                &parameters,
                                evaluation,
                                &matrix_inventory,
                                &mut chosen,
                                &mut planned,
                                &mut preparation,
                                &invocation.input_layouts,
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
                                &matrix_inventory,
                                &mut chosen,
                                &mut planned,
                                &mut preparation,
                                &invocation.input_layouts,
                            )?
                            else {
                                return Ok(None);
                            };
                            rhs.push(source);
                        }
                        let Some(scratch) = select_operation_scratch(
                            &operation,
                            &matrix_inventory,
                            &deferred,
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
                                &matrix_inventory,
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
                            sources: AdmittedMatrixBinding {
                                parameters,
                                level,
                                evaluation,
                                left: left_source,
                                right: rhs,
                                compact: None,
                            },
                            device,
                            capacity: low,
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
                    let request = select_prepared_matrix(
                        &matrix_inventory,
                        &mut chosen,
                        trial.device,
                        &trial.sources.parameters,
                        trial.sources.level,
                        (rows, count),
                        trial.sources.evaluation,
                        operation.compact_bound(),
                    )?
                    .ok_or_else(capacity_error)?;
                    outputs.push(MatrixDestination {
                        sources: trial.sources,
                        interval: GpuColumnInterval {
                            device: trial.device,
                            start,
                            end: start + count,
                        },
                        request,
                    });
                    start += count;
                }
            } else if let Some(compact) = compact {
                // Compact inputs inherit their resident column ownership. No
                // replica or fragment preparation exists for compact payloads.
                for (compact_index, shard) in compact.shards.iter().enumerate() {
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
                    for (position, fixed) in operation.dependent_inputs(&right).iter().enumerate() {
                        rhs.push(
                            select_matrix_input(
                                fixed,
                                operation.other_columns(position, fixed, start, end),
                                device,
                                shard.value.params(),
                                true,
                                &matrix_inventory,
                                &mut chosen,
                                &mut planned,
                                &mut preparation,
                                &invocation.input_layouts,
                            )?
                            .ok_or_else(capacity_error)?,
                        );
                    }
                    if !scratch.iter().any(|s: &MatrixScratch| s.device == device) {
                        scratch.extend(
                            select_operation_scratch(
                                &operation,
                                &matrix_inventory,
                                &deferred,
                                &mut chosen,
                                device,
                                &parameters,
                                level,
                                operation.scratch_evaluation(evaluation),
                            )?
                            .ok_or_else(capacity_error)?,
                        );
                    }
                    let request = select_prepared_matrix(
                        &matrix_inventory,
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
                        sources: AdmittedMatrixBinding {
                            parameters,
                            level,
                            evaluation,
                            left: Some(PreparedMatrixSource::Shard(compact_index)),
                            right: rhs,
                            compact: Some(PreparedMatrixSource::Shard(compact_index)),
                        },
                        interval: GpuColumnInterval { device, start, end },
                        request,
                    });
                }
            } else {
                let Some(left) = left else {
                    if columns == 0 {
                        layouts.push((
                            layouts.len(),
                            Arc::new(MatrixRequirements::new(outputs, scratch)),
                        ));
                        continue;
                    }
                    return Err(PolyBackendError::InvalidConstantShape);
                };
                let inputs = std::iter::once(left).chain(right.iter()).collect::<Vec<_>>();
                let input_layouts = inputs
                    .iter()
                    .map(|input| (input.columns, input.input_layout.as_ref()))
                    .collect::<Vec<_>>();
                for range in operation.inherited_output_ranges(&input_layouts, columns)? {
                    let (start, end) = (range.columns.start, range.columns.end);
                    let primary = inputs[range.primary];
                    let source_columns = range.source;
                    let input = &primary.shards[range.fragment];
                    let device = self
                        .devices
                        .iter()
                        .position(|(id, _)| *id == input.device_id)
                        .ok_or(PolyBackendError::UnsupportedPlacement)?;
                    let registered = input.value.registered_parameters(&self.devices[device].1)?;
                    if registered != input.value.params() ||
                        registered.execution_owner_id() !=
                            input.value.params().execution_owner_id()
                    {
                        return Err(PolyBackendError::UnsupportedPlacement);
                    }
                    let evaluation = operation.input_evaluation(input.value.is_ntt());
                    let left_source = select_matrix_input(
                        primary,
                        source_columns,
                        device,
                        input.value.params(),
                        evaluation,
                        &matrix_inventory,
                        &mut chosen,
                        &mut planned,
                        &mut preparation,
                        &invocation.input_layouts,
                    )?
                    .ok_or_else(capacity_error)?;
                    let mut rhs = Vec::with_capacity(right.len());
                    for (index, right) in operation.dependent_inputs(&right).iter().enumerate() {
                        let parameters =
                            if matches!(operation, PreparedMatrixOperation::CrtRecompose { .. }) {
                                right
                                    .shards
                                    .first()
                                    .ok_or(PolyBackendError::InvalidConstantShape)?
                                    .value
                                    .registered_parameters(&self.devices[device].1)?
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
                                &matrix_inventory,
                                &mut chosen,
                                &mut planned,
                                &mut preparation,
                                &invocation.input_layouts,
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
                        if existing.parameters.context_identity() != parameters.context_identity() ||
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
                                &matrix_inventory,
                                &deferred,
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
                        input.value.params(),
                        input.value.level(),
                        &right,
                        (rows, end - start),
                    )?;
                    let request = select_prepared_matrix(
                        &matrix_inventory,
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
                        sources: AdmittedMatrixBinding {
                            parameters,
                            level,
                            evaluation,
                            left: Some(left_source),
                            right: rhs,
                            compact: None,
                        },
                        interval: GpuColumnInterval { device, start, end },
                        request,
                    });
                }
            }
            layouts.push((layouts.len(), Arc::new(MatrixRequirements::new(outputs, scratch))));
        }
        let mut classes = Vec::with_capacity(layouts.len());
        // Native layout bounds determine scratch widths without calibration kernels.
        for ((index, requirements), invocation) in layouts.iter().zip(lowered) {
            let operation = &invocation.operation;
            debug_assert_eq!(*index, classes.len());
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
                    let params = &output.sources.parameters;
                    let slots = matrix_inventory
                        .iter()
                        .filter(|entry| entry.context == params.context_identity())
                        .flat_map(|entry| {
                            entry.slots.iter().map(|&(slot, available)| {
                                let id = slot.identity();
                                let key = (
                                    params.context_identity(),
                                    (id.storage_id(), id.slot_id(), id.slot_index()),
                                );
                                (slot, available && !chosen.contains(&key))
                            })
                        })
                        .collect::<Vec<_>>();
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
                                .map_err(PolyBackendError::GpuCalibration)?
                                .into_iter()
                                .flatten(),
                        );
                        if GpuPreparedSlotSnapshot::assign(params, &slots, &claims)
                            .map_err(PolyBackendError::GpuCalibration)?
                            .iter()
                            .all(Option::is_some)
                        {
                            low = width;
                        } else {
                            high = width - 1;
                        }
                    }
                    if low == 0 {
                        return Err(PolyBackendError::GpuSubmission(
                            "preimage one-column step demand does not fit the prepared inventory"
                                .into(),
                        ));
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
        let mut owners = HashMap::new();
        let mut owner = |id| {
            let next = owners.len();
            *owners.entry(id).or_insert(next)
        };
        let invocations = lowered
            .iter()
            .map(|invocation| {
                let left = invocation.operands.left.as_ref();
                let right = &invocation.operands.right;
                Ok(MatrixInvocationGeometry {
                    operation: invocation.operation.kind_name(),
                    rows: invocation.operation.output_rows(left, right)?,
                    columns: invocation.operation.output_columns(left),
                    compact_output: invocation.operation.compact_bound().is_some(),
                    input_owners: left
                        .into_iter()
                        .chain(right)
                        .map(|matrix| owner(matrix.id))
                        .collect(),
                    compact_input_owner: invocation
                        .operands
                        .compact
                        .as_ref()
                        .map(|matrix| owner(matrix.id)),
                })
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        Ok(GpuMatrixLayoutPlan { layouts, classes, inputs: preparation, invocations })
    }

    /// Begin one explicit benchmark class with no live invocation/output leases.
    /// Only this setup boundary reopens allocation; production never calls it.
    pub fn begin_prepared_measurement(
        &mut self,
    ) -> Result<mxx_primitives::matrix::gpu_dcrt_poly::GpuGraphAdmissionGuard, PolyBackendError>
    {
        if !self.prepared_invocations.is_empty() {
            return Err(PolyBackendError::GpuSubmission(
                "measurement class has unconsumed invocations".into(),
            ));
        }
        self.fence_released_memory()?;
        let guard = mxx_primitives::matrix::gpu_dcrt_poly::GpuGraphAdmissionGuard::new(
            self.device_parameters(),
        )
        .map_err(PolyBackendError::GpuSubmission)?;
        self.prepared_ledger = None;
        self.prepared_required = false;
        self.graph_prepared = false;
        self.admitted_plan_log.clear();
        Ok(guard)
    }

    /// Allocate the class's typed capacity before closing its native setup epoch.
    pub fn prepare_measurement_storage(
        &mut self,
        claims: &[(ConcreteMatrixType, Vec<GpuTracedClaim>)],
    ) -> Result<(), PolyBackendError> {
        let mut storage = Vec::new();
        for (device, (_, backend)) in self.devices.iter().enumerate() {
            for (ty, claims) in claims {
                let parameters = backend.parameters(ty)?;
                let mut matrices = claims
                    .iter()
                    .copied()
                    .filter(|claim| claim.kind() == GpuPreparedSlotKind::Matrix)
                    .collect::<Vec<_>>();
                let layouts = claims.iter().filter_map(GpuTracedClaim::layout).collect::<Vec<_>>();
                if matrices.is_empty() && layouts.is_empty() {
                    continue;
                }
                if matrices.is_empty() {
                    matrices.push(GpuTracedClaim::matrix(1, 1, parameters.crt_depth() - 1, true));
                }
                storage.push((
                    device,
                    Arc::new(
                        GpuPreparedStorage::new(
                            Some(parameters),
                            Vec::new(),
                            Some(&matrices),
                            Some(&layouts),
                        )
                        .map_err(PolyBackendError::GpuSubmission)?,
                    ),
                ));
            }
        }
        self.prepare_memory(storage, true)
    }

    /// Bind real operands to the same IR placement lowering used by the model,
    /// then reserve and compile them through ordinary native admission.
    pub fn admit_measurement_node(
        &mut self,
        node: &crate::gpu_invocation::GpuNodeOperation,
        matrices: &[Vec<Option<Arc<GpuFleetMatrix>>>],
        compact: &[Vec<Option<Arc<GpuFleetSmallMatrix>>>],
        column_cap: usize,
    ) -> Result<(), PolyBackendError> {
        let lowered = matrices
            .iter()
            .zip(compact)
            .map(|(matrices, compact)| {
                let descriptors = matrices
                    .iter()
                    .zip(compact)
                    .map(|(matrix, compact)| {
                        matrix
                            .as_ref()
                            .map(|matrix| MatrixDescriptor::from(matrix.as_ref()))
                            .or_else(|| {
                                compact
                                    .as_ref()
                                    .map(|matrix| MatrixDescriptor::from(matrix.as_ref()))
                            })
                    })
                    .collect();
                let placement = MatrixPlacementInvocation::new(self, node, descriptors)?
                    .ok_or(PolyBackendError::InvalidConstantShape)?;
                let matrix_owners = matrices
                    .iter()
                    .flatten()
                    .map(|matrix| (matrix.id, matrix))
                    .collect::<HashMap<_, _>>();
                let compact_owners = compact
                    .iter()
                    .flatten()
                    .map(|matrix| (matrix.id, matrix))
                    .collect::<HashMap<_, _>>();
                let fresh = placement.operation.fresh_type().is_some() &&
                    !matches!(placement.operation, PreparedMatrixOperation::Preimage { .. });
                let caller_count =
                    if matches!(placement.operation, PreparedMatrixOperation::Preimage { .. }) {
                        1
                    } else {
                        usize::MAX
                    };
                Ok(LoweredMatrixInvocation {
                    operation: placement.operation,
                    operands: InvocationOperands {
                        left: placement
                            .operands
                            .left
                            .map(|matrix| (**matrix_owners[&matrix.id]).clone()),
                        right: placement
                            .operands
                            .right
                            .into_iter()
                            .map(|matrix| (**matrix_owners[&matrix.id]).clone())
                            .collect(),
                        compact: placement
                            .operands
                            .compact
                            .map(|matrix| (**compact_owners[&matrix.id]).clone()),
                    },
                    input_layouts: placement.input_layouts,
                    caller_ids: if fresh {
                        Vec::new()
                    } else {
                        matrices
                            .iter()
                            .flatten()
                            .take(caller_count)
                            .map(|matrix| matrix.id)
                            .collect()
                    },
                    caller_compact: if fresh {
                        None
                    } else {
                        compact.iter().flatten().next().map(|matrix| matrix.id)
                    },
                })
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        self.admit_matrix_invocations(lowered, column_cap)
    }

    pub(super) fn admit_matrix_invocations(
        &mut self,
        lowered: Vec<LoweredMatrixInvocation>,
        column_cap: usize,
    ) -> Result<(), PolyBackendError> {
        if lowered.is_empty() {
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
            // Freeze native eligibility once for the entire CPU selection phase.
            // All choices below precede input preparation and native reservation.
            let snapshots = inventory
                .par_iter()
                .map(|(device, storage)| {
                    // Poll only deferred upload resources, without waiting. Their
                    // selected subset is still retired at the transaction boundary.
                    let has_deferred = (0..storage.slot_count()).any(|index| {
                        matches!(
                            storage.slot_identity(index).unwrap().kind(),
                            GpuPreparedSlotKind::PinnedHost | GpuPreparedSlotKind::CompletionEvent
                        )
                    });
                    let pending =
                        if has_deferred { storage.poll_releases(&[])? } else { Vec::new() };
                    let deferred = pending
                        .into_iter()
                        .filter_map(|index| {
                            let slot = storage.slot_identity(index).unwrap();
                            matches!(
                                slot.kind(),
                                GpuPreparedSlotKind::PinnedHost |
                                    GpuPreparedSlotKind::CompletionEvent
                            )
                            .then_some(slot.slot_id())
                        })
                        .collect::<Vec<_>>();
                    let context = storage.context_identity();
                    let slots = storage
                        .snapshot()?
                        .into_iter()
                        .map(|slot| (slot, slot.is_available()))
                        .collect();
                    Ok::<_, String>((
                        MatrixSlotContext { device: *device, context, slots },
                        deferred,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(PolyBackendError::GpuSubmission)?;
            let mut matrix_inventory = Vec::with_capacity(snapshots.len());
            let mut deferred = HashSet::new();
            for (snapshot, pending) in snapshots {
                matrix_inventory.push(snapshot);
                deferred.extend(pending);
            }
            let metadata =
                lowered.par_iter().map(MatrixPlacementInvocation::from).collect::<Vec<_>>();
            let GpuMatrixLayoutPlan { layouts, mut classes, inputs: preparation, .. } =
                self.plan_matrix_layouts(&metadata, &matrix_inventory, &deferred)?;
            for class in &mut classes {
                class.maximum_columns = class.maximum_columns.min(column_cap);
            }
            // All minimum-width inputs, outputs and scratch now fit. Retire
            // only selected upload resources before making native reservations;
            // do not drain unrelated uploads or wait during rejected trials.
            let mut retirements =
                std::collections::BTreeMap::<u64, (Arc<GpuPreparedStorage>, Vec<usize>)>::new();
            for (_, requirements) in &layouts {
                for scratch in &requirements.resources.scratch {
                    if matches!(
                        scratch.slot.kind(),
                        GpuPreparedSlotKind::PinnedHost | GpuPreparedSlotKind::CompletionEvent
                    ) {
                        let storage = inventory
                            .iter()
                            .find(|(_, storage)| storage.identity() == scratch.slot.storage_id())
                            .expect("selected scratch belongs to the accepted inventory")
                            .1
                            .clone();
                        retirements
                            .entry(storage.identity())
                            .or_insert_with(|| (storage, Vec::new()))
                            .1
                            .push(scratch.slot.slot_index());
                    }
                }
            }
            for (_, (storage, slots)) in retirements {
                let pending =
                    storage.poll_releases(&[]).map_err(PolyBackendError::GpuSubmission)?;
                let selected =
                    slots.into_iter().filter(|slot| pending.contains(slot)).collect::<Vec<_>>();
                if !selected.is_empty() {
                    storage.poll_releases(&selected).map_err(PolyBackendError::GpuSubmission)?;
                }
            }
            let mut admitted = Vec::with_capacity(lowered.len());
            for (index, invocation) in lowered.into_iter().enumerate() {
                let requirements = &layouts[index].1;
                let intervals =
                    requirements.outputs.iter().map(|output| output.interval).collect::<Vec<_>>();
                let plan = ledger
                    .reserve_columns(
                        invocation.operation.output_columns(invocation.operands.left.as_ref()),
                        GpuOutputOwnership::Inherited(&intervals),
                        GpuColumnWidthPolicy::Native(classes[index]),
                        &requirements.resources,
                    )
                    .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
                let sources =
                    requirements.outputs.iter().map(|output| output.sources.clone()).collect();
                admitted.push((invocation, plan, sources));
            }
            let input_owners = admitted
                .iter()
                .flat_map(|(invocation, _, _)| {
                    invocation.operands.left.iter().chain(&invocation.operands.right)
                })
                .map(|matrix| (matrix.id, matrix))
                .collect::<HashMap<_, _>>();
            let preparation = preparation
                .into_iter()
                .map(|input| {
                    let storage = inventory
                        .iter()
                        .find(|(_, storage)| storage.identity() == input.request.slot_key().0)
                        .expect("selected request belongs to the accepted inventory")
                        .1
                        .clone();
                    MatrixInputPreparation {
                        matrix: (*input_owners[&input.owner]).clone(),
                        layout: input.layout,
                        parameters: input.parameters,
                        device: input.device,
                        storage,
                        request: input.request,
                    }
                })
                .collect::<Vec<_>>();
            // Commit every sibling's output and reusable scratch before the
            // first normalization or replica copy. Input preparation reserves
            // its entire selected input list before submitting either phase.
            // Thus a later sibling's capacity failure drops all parked plans
            // without first performing GPU work for earlier siblings.
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
                            let sources = match input.layout.source {
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
