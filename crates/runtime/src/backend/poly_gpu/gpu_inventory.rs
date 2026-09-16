//! Materialize the fixed prepared inventory from a resolved resource plan.
//!
//! Prepared lowering resolves retained owners and native resource claims once
//! at warmup. Published slots are then replayed without dynamic operation
//! selection, candidate fitting, or discovery during execution.

use super::{gpu_claims::PreparedClaimBroker, *};
use crate::gpu_memory::GpuMemoryLedger;
use mxx_ir_core::{ValidatedGraph, node::NodeKind, types::ConcreteWireType};
use mxx_primitives::{
    matrix::gpu_dcrt_poly::{
        GpuCompactTransferKind, GpuPreparedProvisioningPermit, GpuPreparedRequest,
        GpuPreparedSlotKind, GpuPreparedSlotSnapshot, GpuPreparedStorage,
        GpuPreparedWorkspaceLayout, GpuTracedClaim,
    },
    poly::dcrt::gpu::GPU_POLY_FORMAT_EVAL,
};
use std::collections::{BTreeMap, BTreeSet};

/// Stable identity of one prepared command stream.  The physical device is
/// explicit because fleet order is not a CUDA device identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct PreparedScheduleStreamKey {
    pub instance: usize,
    pub command: usize,
    pub stream: usize,
    pub device: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct PreparedScheduleResourceBinding {
    pub key: PreparedScheduleStreamKey,
    pub storage: super::gpu_prepared_lowering::PreparedStorageBinding,
    pub request: GpuPreparedRequest,
}

/// The detached region and exact event-slot bindings remain owned together
/// until every schedule using them has been dropped.  The native schedule
/// owns the actual `GpuCudaResource`; this table only records its authorized
/// physical slot, so replay never searches inventory or creates an event.
pub(crate) struct PreparedScheduleBindings {
    pub entries: Box<[PreparedScheduleResourceBinding]>,
}

impl GpuDcrtBackend {
    fn prepared_output_codec_claims(
        &self,
        program: &super::gpu_prepared_lowering::GpuPreparation,
        physical_devices: &[i32],
        instance_count: usize,
    ) -> Result<Vec<(GpuDCRTPolyParams, Vec<GpuTracedClaim>)>, PolyBackendError> {
        let mut claims = Vec::new();
        for (wire, _instance, physical_device) in
            prepared_output_codec_keys(program, physical_devices, instance_count)
        {
            let Some(matrix) = program.wire_types[&wire].matrix_type() else {
                continue;
            };
            let location = program.values.get(&wire).ok_or_else(|| {
                PolyBackendError::GpuSubmission("prepared output has no location".into())
            })?;
            let backend = self
                .devices
                .iter()
                .find(|(device, _)| *device == physical_device)
                .map(|(_, backend)| backend)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let params = backend.parameters(matrix)?.clone();
            if !params.device_ids().contains(&physical_device) {
                return Err(PolyBackendError::UnsupportedPlacement);
            }
            let transfer = params
                .compact_transfer_workspace(
                    location.level,
                    matrix.rows,
                    matrix.columns,
                    GpuCompactTransferKind::Store,
                )
                .map_err(PolyBackendError::GpuSubmission)?;
            // Keep one independent codec claim for every physical output
            // shard and execution instance. These claims are intentionally
            // unbound: output materialization consumes the command-owned
            // lease, while admission reserves a distinct slot for every
            // shard/instance.
            claims.push((
                params,
                vec![
                    GpuTracedClaim::matrix(
                        matrix.rows,
                        matrix.columns,
                        location.level,
                        matches!(
                            location.format,
                            super::gpu_prepared_lowering::PreparedFormat::Evaluation
                        ),
                    ),
                    GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                        kind: GpuPreparedSlotKind::SubmissionStream,
                        bytes: 0,
                        alignment: 1,
                    }),
                    GpuTracedClaim::workspace(transfer),
                ],
            ));
        }
        Ok(claims)
    }

    pub(crate) fn prepared_device_index(&self, physical_device: i32) -> Option<usize> {
        prepared_device_index_from_identities(
            self.prepared_ledger.as_ref()?.execution_identities()?,
            physical_device,
        )
    }

    pub(crate) fn accepted_prepared_storage_inventory(
        &self,
    ) -> Option<BTreeMap<u64, (usize, Arc<GpuPreparedStorage>)>> {
        self.prepared_ledger.as_ref().map(|ledger| {
            ledger
                .accepted_prepared_inventory()
                .map(|(device, storage)| (storage.identity(), (device, storage)))
                .collect()
        })
    }

    /// Resolve every slot described by the metadata resolver in one setup pass.
    /// This is deliberately separate from schedule provisioning: the latter
    /// may only consume the region and slot references returned here.
    pub(crate) fn reserve_resolved_resources(
        &mut self,
        plan: &super::gpu_prepared_lowering::PreparedResourcePlan,
        resources: &mut super::gpu_prepared_lowering::PreparedResolvedResources,
        extra_claims: &[(GpuDCRTPolyParams, Vec<GpuTracedClaim>)],
    ) -> Result<
        (Arc<crate::gpu_memory::GpuMemoryRegion>, BTreeMap<u64, Arc<GpuPreparedStorage>>),
        PolyBackendError,
    > {
        let mut claims = BTreeMap::<usize, (GpuDCRTPolyParams, Vec<GpuTracedClaim>)>::new();
        enum ClaimTarget {
            Owner(usize),
            Command(usize, usize),
            Composite(usize, usize),
            Replay(usize, usize),
            Schedule(usize, usize),
            ScalarBuffer(usize, usize),
            Unbound,
        }
        let mut claim_entries = Vec::<(GpuDCRTPolyParams, GpuTracedClaim, ClaimTarget)>::new();

        // Logical stores are the complete retained owner set. Native stage
        // layouts below add only their auxiliary resources and events.
        for store in &plan.stores {
            let matrix =
                store.wire_type.as_ref().and_then(ConcreteWireType::matrix_type).ok_or_else(
                    || {
                        PolyBackendError::GpuSubmission(
                            "resolved store has no matrix parameters".into(),
                        )
                    },
                )?;
            let params = self
                .resource_parameters(matrix)
                .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?
                .into_iter()
                .find(|params| params.device_ids().contains(&store.location.device))
                .ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "resolved store has no device parameters".into(),
                    )
                })?;
            let claim = GpuTracedClaim::matrix(
                store.capacity_rows,
                store.capacity_columns,
                store.location.level,
                store.location.format == super::gpu_prepared_lowering::PreparedFormat::Evaluation,
            );
            claims
                .entry(params.context_identity())
                .or_insert_with(|| (params.clone(), Vec::new()))
                .1
                .push(claim);
            let owner_index = resources
                .owners
                .iter()
                .position(|owner| {
                    owner.key.owner == store.location.owner &&
                        owner.key.device == store.location.device &&
                        owner.key.instance == store.instance &&
                        owner.key.level == store.location.level &&
                        owner.key.format == store.location.format
                })
                .ok_or_else(|| {
                    PolyBackendError::GpuSubmission("resolved store has no resolved owner".into())
                })?;
            claim_entries.push((params, claim, ClaimTarget::Owner(owner_index)));
        }

        let params_for_layout =
            |layout: &mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationLayout,
             store_hint: Option<usize>| {
                // Pinned host buffers are deliberately keyed with device -1;
                // their resource domain is the device owner of the stage's
                // matrix store, not a fabricated host device.
                let store = if layout.key.device < 0 {
                    store_hint.and_then(|index| plan.stores.get(index))
                } else {
                    resources
                        .owners
                        .iter()
                        .find(|owner| {
                            (owner.layout.execution_owner_identity() ==
                                layout.key.execution_owner_identity ||
                                owner.key.owner == layout.key.execution_owner_identity) &&
                                owner.key.device == layout.key.device &&
                                owner.key.level ==
                                    usize::try_from(layout.level).unwrap_or(owner.key.level)
                        })
                        .and_then(|owner| {
                            plan.stores.iter().find(|store| {
                                store.location.device == owner.key.device &&
                                    store.location.level == owner.key.level &&
                                    store.location.format == owner.key.format
                            })
                        })
                };
                store
                    .and_then(|store| {
                        store.wire_type.as_ref().and_then(ConcreteWireType::matrix_type)
                    })
                    .and_then(|matrix| {
                        self.resource_parameters(matrix).ok()?.into_iter().find(|params| {
                            layout.key.device < 0 ||
                                params.device_ids().contains(&layout.key.device)
                        })
                    })
            };
        let traced = |layout: &mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationLayout,
                      store_hint: Option<usize>| {
            // Native host-only geometry is metadata, not an inventory slot.
            if layout.kind == 100 {
                return None;
            }
            let params = params_for_layout(layout, store_hint)?;
            let claim = if layout.kind == 0 {
                GpuTracedClaim::matrix(
                    layout.rows,
                    layout.columns,
                    usize::try_from(layout.level).ok()?,
                    layout.format == GPU_POLY_FORMAT_EVAL,
                )
            } else {
                let kind = match layout.kind {
                    1 => GpuPreparedSlotKind::BatchWorkspace,
                    2 => GpuPreparedSlotKind::TransformWorkspace,
                    3 => GpuPreparedSlotKind::PinnedHost,
                    4 => GpuPreparedSlotKind::CompactPayload,
                    5 => GpuPreparedSlotKind::CompactWorkspace,
                    6 => GpuPreparedSlotKind::SamplerWorkspace,
                    7 => GpuPreparedSlotKind::TransferWorkspace,
                    8 => GpuPreparedSlotKind::CompletionEvent,
                    9 => GpuPreparedSlotKind::SubmissionStream,
                    _ => return None,
                };
                GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                    kind,
                    bytes: layout.bytes,
                    alignment: layout.alignment.max(1),
                })
            };
            Some((params, claim))
        };
        let traced_composite =
            |claim: &mxx_primitives::matrix::gpu_dcrt_poly::GpuTracedClaim,
             layout: Option<&mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationLayout>,
             store_hint: Option<usize>| {
                let store = store_hint.and_then(|index| plan.stores.get(index));
                let params =
                    layout.and_then(|layout| params_for_layout(layout, store_hint)).or_else(|| {
                        store
                            .and_then(|store| {
                                store.wire_type.as_ref().and_then(ConcreteWireType::matrix_type)
                            })
                            .and_then(|matrix| self.resource_parameters(matrix).ok())
                            .and_then(|params| params.into_iter().next())
                    });
                let traced = if claim.kind() == GpuPreparedSlotKind::Matrix {
                    *claim
                } else {
                    GpuTracedClaim::workspace(claim.layout()?)
                };
                params.map(|params| (params, traced))
            };

        for (buffer_index, buffer) in resources.scalar_buffers.iter().enumerate() {
            let store_hint = plan.stores.iter().position(|store| {
                store.location.owner == buffer.plan.owner.owner &&
                    store.location.device == buffer.plan.owner.device &&
                    store.instance == buffer.plan.owner.instance &&
                    store.location.level == buffer.plan.owner.level &&
                    store.location.format == buffer.plan.owner.format
            });
            for (allocation_index, allocation) in buffer.layout.allocations().iter().enumerate() {
                let Some((params, claim)) = traced(allocation, store_hint) else {
                    continue;
                };
                claims
                    .entry(params.context_identity())
                    .or_insert_with(|| (params.clone(), Vec::new()))
                    .1
                    .push(claim);
                claim_entries.push((
                    params,
                    claim,
                    ClaimTarget::ScalarBuffer(buffer_index, allocation_index),
                ));
            }
        }

        for (command_index, command) in resources.commands.iter_mut().enumerate() {
            let store_hint =
                command.command.outputs.first().and_then(|descriptor| descriptor.store).or_else(
                    || command.command.inputs.first().and_then(|descriptor| descriptor.store),
                );
            for (allocation_index, allocation) in
                command.composite_allocations.iter_mut().enumerate()
            {
                let Some((params, claim)) =
                    traced_composite(&allocation.claim, allocation.layout.as_ref(), store_hint)
                else {
                    continue;
                };
                claims
                    .entry(params.context_identity())
                    .or_insert_with(|| (params.clone(), Vec::new()))
                    .1
                    .push(claim);
                claim_entries.push((
                    params,
                    claim,
                    ClaimTarget::Composite(command_index, allocation_index),
                ));
            }
            for (allocation_index, allocation) in command.allocations.iter_mut().enumerate() {
                let Some((params, claim)) = traced(&allocation.layout, allocation.store) else {
                    continue;
                };
                claims
                    .entry(params.context_identity())
                    .or_insert_with(|| (params.clone(), Vec::new()))
                    .1
                    .push(claim);
                claim_entries.push((
                    params,
                    claim,
                    ClaimTarget::Command(command_index, allocation_index),
                ));
            }
            if let Some(replay) = command.replay_upload.as_mut() {
                for (allocation_index, allocation) in replay.allocations.iter_mut().enumerate() {
                    let Some((params, claim)) = traced(&allocation.layout, allocation.store) else {
                        continue;
                    };
                    claims
                        .entry(params.context_identity())
                        .or_insert_with(|| (params.clone(), Vec::new()))
                        .1
                        .push(claim);
                    claim_entries.push((
                        params,
                        claim,
                        ClaimTarget::Replay(command_index, allocation_index),
                    ));
                }
            }
        }
        for (schedule_index, schedule) in resources.schedules.iter_mut().enumerate() {
            for (allocation_index, allocation) in schedule.allocations.iter_mut().enumerate() {
                let Some((params, claim)) = traced(&allocation.layout, allocation.store) else {
                    continue;
                };
                claims
                    .entry(params.context_identity())
                    .or_insert_with(|| (params.clone(), Vec::new()))
                    .1
                    .push(claim);
                claim_entries.push((
                    params,
                    claim,
                    ClaimTarget::Schedule(schedule_index, allocation_index),
                ));
            }
        }
        // Output codecs are lowered from the graph boundary rather than from
        // a native command recipe.  They are nevertheless fixed warmup
        // resources and must join this same provisioning transaction.  Their
        // leases are consumed by the codec owner, so there is intentionally no
        // command slot to write here.
        for (params, extra) in extra_claims {
            let entry = &mut claims
                .entry(params.context_identity())
                .or_insert_with(|| (params.clone(), Vec::new()))
                .1;
            for &claim in extra {
                entry.push(claim);
                claim_entries.push((params.clone(), claim, ClaimTarget::Unbound));
            }
        }
        self.provision_resolved_storage(claims.into_values().collect())?;
        let inventory = self.accepted_prepared_storage_inventory().ok_or_else(|| {
            PolyBackendError::GpuSubmission("resolved resource inventory is missing".into())
        })?;
        let mut used = BTreeSet::new();
        let mut requests = BTreeMap::<u64, Vec<GpuPreparedRequest>>::new();
        for (params, claim, target) in claim_entries {
            let mut found = None;
            for (identity, (device, storage)) in &inventory {
                if !params.device_ids().contains(&storage.device()) {
                    continue;
                }
                for snapshot in storage.snapshot().map_err(PolyBackendError::GpuSubmission)? {
                    if !snapshot.is_available() {
                        continue;
                    }
                    if let Some(request) = snapshot
                        .request(&params, &claim)
                        .map_err(PolyBackendError::GpuSubmission)?
                    {
                        if used.insert(request.slot_key()) {
                            found = Some((*identity, *device, request));
                            break;
                        }
                    }
                }
                if found.is_some() {
                    break;
                }
            }
            let (identity, _device, request) = found.ok_or_else(|| {
                PolyBackendError::GpuSubmission("resolved resource has no accepted slot".into())
            })?;
            requests.entry(identity).or_default().push(request);
            let slot = Some(super::gpu_prepared_lowering::PreparedSlotRef {
                // `device` is the ledger/fleet ordinal.  Slot bindings carry
                // the physical CUDA identity used by the native region.
                device: inventory.get(&identity).map(|(_, storage)| storage.device()).ok_or_else(
                    || {
                        PolyBackendError::GpuSubmission(
                            "resolved resource storage disappeared".into(),
                        )
                    },
                )?,
                request,
            });
            match target {
                ClaimTarget::Owner(owner) => {
                    resources.owners[owner].slot = slot;
                }
                ClaimTarget::Command(command, allocation) => {
                    resources.commands[command].allocations[allocation].slot = slot;
                }
                ClaimTarget::Composite(command, allocation) => {
                    resources.commands[command].composite_allocations[allocation].slot = slot;
                }
                ClaimTarget::Replay(command, allocation) => {
                    resources.commands[command]
                        .replay_upload
                        .as_mut()
                        .expect("replay claim target must have a replay layout")
                        .allocations[allocation]
                        .slot = slot;
                }
                ClaimTarget::Schedule(schedule, allocation) => {
                    resources.schedules[schedule].allocations[allocation].slot = slot;
                }
                ClaimTarget::ScalarBuffer(buffer, allocation) => {
                    resources.scalar_buffers[buffer].slots[allocation] = slot;
                }
                ClaimTarget::Unbound => {}
            }
        }
        let ledger = self.prepared_ledger.as_mut().ok_or_else(|| {
            PolyBackendError::GpuSubmission("resolved resource ledger is missing".into())
        })?;
        let requirements = requests
            .iter()
            .map(|(identity, requests)| {
                let (device, storage) = inventory.get(identity).ok_or_else(|| {
                    PolyBackendError::GpuSubmission("resolved resource storage disappeared".into())
                })?;
                Ok(crate::gpu_memory::GpuPreparedAllocationRequirement {
                    device: *device,
                    storage,
                    requests,
                })
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        let region = ledger
            .reserve_region(&requirements)
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        let storages = region
            .prepared_inventory()
            .map(|(device, storage)| (storage.identity(), (device, storage)))
            .filter(|(identity, _)| requests.contains_key(identity))
            .map(|(identity, (_, storage))| (identity, storage))
            .collect();
        // Stream claims identify their native submission-stream allocation by
        // the full physical key.  Ordinal matching is incorrect when a plan
        // has matrix/workspace allocations interleaved with streams.
        for command in &mut resources.commands {
            super::gpu_prepared_lowering::validate_composite_stream_claims(
                &command.composite_allocations,
                &command.composite_streams,
            )
            .map_err(PolyBackendError::GpuSubmission)?;
            for stream in &mut command.composite_streams {
                if stream.layout.origin == 0 {
                    // Context-reused streams are not allocations in the
                    // bundle; their owner layout supplies the fixed stream
                    // identity directly at native bind time.
                    stream.slot = None;
                    continue;
                }
                let matches = command
                    .composite_allocations
                    .iter()
                    .filter(|allocation| {
                        allocation.claim.kind() == GpuPreparedSlotKind::SubmissionStream &&
                            allocation
                                .layout
                                .is_some_and(|layout| layout.key == stream.layout.key)
                    })
                    .collect::<Vec<_>>();
                if matches.len() != 1 {
                    return Err(PolyBackendError::GpuSubmission(format!(
                        "composite stream key {:?} has {} matching allocation claims",
                        stream.layout.key,
                        matches.len()
                    )));
                }
                stream.slot = matches[0].slot;
            }
            for stream in &mut command.streams {
                stream.slot = command
                    .allocations
                    .iter()
                    .find(|allocation| {
                        allocation.layout.kind == 9 && allocation.layout.key == stream.layout.key
                    })
                    .and_then(|allocation| allocation.slot);
            }
            if let Some(replay) = command.replay_upload.as_mut() {
                for stream in &mut replay.streams {
                    stream.slot = replay
                        .allocations
                        .iter()
                        .find(|allocation| {
                            allocation.layout.kind == 9 &&
                                allocation.layout.key == stream.layout.key
                        })
                        .and_then(|allocation| allocation.slot);
                }
            }
        }
        for schedule in &mut resources.schedules {
            for stream in &mut schedule.streams {
                stream.slot = schedule
                    .allocations
                    .iter()
                    .find(|allocation| {
                        allocation.layout.kind == 9 && allocation.layout.key == stream.layout.key
                    })
                    .and_then(|allocation| allocation.slot);
            }
        }
        Ok((region, storages))
    }

    /// Provision completion events from an already detached prepared region.
    /// Matrix, compact-payload, and event claims are acquired by the caller's
    /// single warmup transaction before this method is entered.
    pub(crate) fn provision_prepared_schedules_in_region(
        &mut self,
        schedules: &mut [Option<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSchedule>],
        streams: &[Box<[PreparedScheduleStreamKey]>],
        region: &mut Arc<crate::gpu_memory::GpuMemoryRegion>,
    ) -> Result<PreparedScheduleBindings, PolyBackendError> {
        use mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotKind;

        if schedules.len() != streams.len() {
            return Err(PolyBackendError::GpuSubmission(
                "prepared schedule stream table length mismatch".into(),
            ));
        }
        let mut stream_count = 0usize;
        let mut parameters = Vec::with_capacity(schedules.len());
        for (schedule, schedule_streams) in schedules.iter().zip(streams) {
            let schedule = schedule.as_ref().ok_or_else(|| {
                PolyBackendError::GpuSubmission("missing prepared schedule".into())
            })?;
            let expected = schedule.stream_count();
            if expected != schedule_streams.len() {
                return Err(PolyBackendError::GpuSubmission(format!(
                    "prepared schedule stream table has {} entries, native plan reports {expected}",
                    schedule_streams.len()
                )));
            }
            stream_count =
                stream_count.checked_add(expected).ok_or(PolyBackendError::InvalidInteger)?;
            let owners = schedule.stream_parameters().map_err(PolyBackendError::GpuSubmission)?;
            for (key, owner) in schedule_streams.iter().zip(&owners) {
                if !owner.device_ids().contains(&key.device) {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
            }
            parameters.push(owners);
        }
        let inventory = region.prepared_inventory().collect::<Vec<_>>();
        let mut used = BTreeSet::new();
        let mut selected = Vec::with_capacity(stream_count);
        for (schedule_streams, owners) in streams.iter().zip(&parameters) {
            for (&key, owner) in schedule_streams.iter().zip(owners) {
                let candidate = inventory
                    .iter()
                    .filter(|(_, storage)| {
                        storage.device() == key.device &&
                            storage.context_identity() == owner.context_identity()
                    })
                    .flat_map(|(device, storage)| {
                        (0..storage.slot_count()).map(move |slot| (*device, storage, slot))
                    })
                    .filter_map(|(device, storage, slot)| {
                        let identity = storage.slot_identity(slot)?;
                        (identity.kind() == GpuPreparedSlotKind::CompletionEvent &&
                            storage.snapshot().ok()?.get(slot)?.is_available())
                        .then(|| {
                            let request = identity.workspace_request(0, 1);
                            (device, Arc::clone(storage), request)
                        })
                    })
                    .find(|(_, _, request)| used.insert(request.slot_key()));
                let (device, storage, request) = candidate.ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "prepared inventory cannot fit schedule completion events".into(),
                    )
                })?;
                selected.push((key, device, storage, request));
            }
        }

        let region_inventory = region
            .prepared_inventory()
            .map(|(_, storage)| (storage.identity(), storage))
            .collect::<BTreeMap<_, _>>();

        let provision = (|| {
            let mut entries = Vec::with_capacity(selected.len());
            let mut selected_offset = 0;
            for (schedule, schedule_streams) in schedules.iter_mut().zip(streams) {
                let schedule =
                    schedule.as_mut().ok_or_else(|| "missing prepared schedule".to_owned())?;
                let count = schedule_streams.len();
                let schedule_selected = &selected[selected_offset..selected_offset + count];
                selected_offset += count;
                let mut grouped =
                    BTreeMap::<u64, (Arc<GpuPreparedStorage>, Vec<GpuPreparedRequest>)>::new();
                for (_, _, storage, request) in schedule_selected {
                    let authorized = region_inventory
                        .get(&storage.identity())
                        .ok_or_else(|| "detached schedule storage disappeared".to_owned())?;
                    grouped
                        .entry(authorized.identity())
                        .or_insert_with(|| (Arc::clone(authorized), Vec::new()))
                        .1
                        .push(*request);
                }
                let mut reservations = grouped
                    .values()
                    .map(|(storage, requests)| storage.reserve(requests))
                    .collect::<Result<Vec<_>, _>>()?;
                let first = reservations
                    .drain(..1)
                    .next()
                    .ok_or_else(|| "prepared schedule has no event reservation".to_owned())?;
                let dispatch = first.enter(reservations)?;
                schedule.provision()?;
                drop(dispatch.finish()?);
                for (key, _, storage, request) in schedule_selected {
                    let identity = storage
                        .slot_identity(request.slot_key().2)
                        .ok_or_else(|| "prepared schedule event slot disappeared".to_owned())?;
                    entries.push(PreparedScheduleResourceBinding {
                        key: *key,
                        storage: super::gpu_prepared_lowering::PreparedStorageBinding {
                            storage_id: identity.storage_id(),
                            slot_id: identity.slot_id(),
                            slot_index: identity.slot_index(),
                            context: storage.context_identity(),
                            basis: request.level().unwrap_or_default(),
                        },
                        request: *request,
                    });
                }
            }
            entries.sort_by_key(|entry| entry.key);
            Ok::<_, String>(entries.into_boxed_slice())
        })();
        match provision {
            Ok(entries) => Ok(PreparedScheduleBindings { entries }),
            Err(error) => Err(PolyBackendError::GpuSubmission(error)),
        }
    }

    /// Build and publish the complete prepared inventory at the warmup
    /// boundary. The caller has already computed the graph identity and owns
    /// the public warmup contract; this helper contains no compatibility or
    /// production replay path.
    pub(crate) fn warm_up_prepared_graph_impl(
        &mut self,
        validated: &ValidatedGraph,
        inputs: &BTreeMap<String, crate::backend::RuntimeValue<Self>>,
        wave_bound: usize,
        max_live_gpu_executions: usize,
    ) -> Result<(), PolyBackendError> {
        crate::backend::poly_gpu::record_prepared_forbidden(0);
        if self.prepared_graph.is_some() {
            return Ok(());
        }
        let mut program = super::gpu_prepared_lowering::lower_graph(
            validated,
            std::num::NonZeroUsize::new(wave_bound.max(1)).expect("nonzero warmup wave"),
        )
        .map_err(|error| {
            PolyBackendError::GpuSubmission(format!("prepared topology: {error:?}"))
        })?;
        program.instance_count = max_live_gpu_executions;
        // `lower_graph` starts with one representative instance. Rebuild its
        // pure resource plan after applying the execution concurrency bound so
        // owner, command, schedule, and slot multiplicity describe the fixed
        // replay pool provisioned below.
        let physical_devices = self.devices.iter().map(|(device, _)| *device).collect::<Vec<_>>();
        program.resource_plan =
            super::gpu_prepared_lowering::PreparedResourcePlan::from_preparation_for_devices(
                &program,
                &physical_devices,
            )
            .map_err(|error| {
                PolyBackendError::GpuSubmission(format!("prepared resources: {error:?}"))
            })?;
        program.resource_plan.validate_for_warmup().map_err(|error| {
            PolyBackendError::GpuSubmission(format!("prepared resources: {error:?}"))
        })?;
        // Resolve every owner and native stage through the metadata-only
        // backend before materializing the replay tape. This is the warmup
        // contract boundary: a partially adaptive tape must never be
        // published when native selection cannot consume the recipe.
        let mut resolved_resources = program
            .resource_plan
            .resolve_prepared_resources(self)
            .map_err(PolyBackendError::GpuSubmission)?;
        // Codec envelopes are known from the lowered output locations and are
        // published together with stores, native scratch, streams and events.
        // Materializing host inputs below is deliberately after this commit.
        let codec_claims =
            self.prepared_output_codec_claims(&program, &physical_devices, program.instance_count)?;
        let has_resolved_claims = !program.resource_plan.stores.is_empty() ||
            resolved_resources.commands.iter().any(|command| {
                !command.allocations.is_empty() || !command.composite_allocations.is_empty()
            }) ||
            resolved_resources.commands.iter().any(|command| {
                command.replay_upload.as_ref().is_some_and(|replay| !replay.allocations.is_empty())
            }) ||
            resolved_resources.schedules.iter().any(|schedule| !schedule.allocations.is_empty()) ||
            !codec_claims.is_empty();
        let reservation = if !has_resolved_claims {
            None
        } else {
            Some(self.reserve_resolved_resources(
                &program.resource_plan,
                &mut resolved_resources,
                &codec_claims,
            )?)
        };
        program.resolved_resources = Some(resolved_resources.clone());
        // Host staging must not create an owner before the fixed resource
        // transaction has committed. This ordering is part of the warmup
        // contract, not an optimization.
        let prepared_runtime_inputs = validated
            .root_scope()
            .execution_order
            .iter()
            .filter_map(|node| {
                let NodeKind::Input { name, .. } = node.kind() else {
                    return None;
                };
                Some(
                    super::prepared_runtime_value_for_warmup(self, inputs.get(name)?)
                        .map_err(|error| error.to_string()),
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(PolyBackendError::GpuSubmission)?;
        let execution = crate::backend::poly_gpu::from_lowered_program(
            self,
            &prepared_runtime_inputs,
            &mut program,
            &resolved_resources,
            reservation,
        )
        .map_err(PolyBackendError::GpuSubmission)?;
        let execution =
            execution.from_preparation(program).map_err(PolyBackendError::GpuSubmission)?;
        self.prepared_graph = Some(execution);
        self.fence_released_memory()?;
        Ok(())
    }

    fn provision_resolved_storage(
        &mut self,
        demands: Vec<(GpuDCRTPolyParams, Vec<GpuTracedClaim>)>,
    ) -> Result<(), PolyBackendError> {
        crate::backend::poly_gpu::record_prepared_forbidden(7);
        let mut prepared = self
            .accepted_prepared_storage_inventory()
            .map(|inventory| inventory.into_values().collect::<Vec<_>>())
            .unwrap_or_default();
        let original_count = prepared.len();
        let mut used = HashSet::new();
        struct PendingStorage {
            device: usize,
            params: GpuDCRTPolyParams,
            matrices: Vec<GpuTracedClaim>,
            layouts: Vec<GpuPreparedWorkspaceLayout>,
        }
        let mut pending = Vec::new();
        for (params, claims) in demands {
            let device = self
                .devices
                .iter()
                .position(|(physical, _)| params.device_ids().contains(physical))
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let broker = PreparedClaimBroker::new(
                &params,
                prepared
                    .iter()
                    .filter(|(owner, _)| *owner == device)
                    .map(|(_, storage)| Arc::clone(storage))
                    .collect(),
            );
            let missing =
                broker.missing(&claims, &mut used).map_err(PolyBackendError::GpuSubmission)?;
            let mut matrices = Vec::new();
            let mut layouts = Vec::new();
            for claim in missing {
                if claim.kind() == GpuPreparedSlotKind::Matrix {
                    matrices.push(claim);
                } else {
                    layouts.push(claim.layout().ok_or(PolyBackendError::InvalidInteger)?);
                }
            }
            if matrices.is_empty() && layouts.is_empty() {
                continue;
            }
            if matrices.is_empty() {
                matrices.push(GpuTracedClaim::matrix(1, 1, params.crt_depth() - 1, true));
            }
            pending.push(PendingStorage { device, params, matrices, layouts });
        }
        if pending.is_empty() {
            return Ok(());
        }

        if original_count == 0 {
            for request in pending {
                let storage = GpuPreparedStorage::new(
                    Some(&request.params),
                    Vec::new(),
                    Some(&request.matrices),
                    Some(&request.layouts),
                )
                .map_err(PolyBackendError::GpuSubmission)?;
                prepared.push((request.device, Arc::new(storage)));
            }
            return self.prepare_memory(prepared, true);
        }

        // Reserve the complete fleet charge before the first native backing is
        // constructed. The transaction owns this charge through publication;
        // native permits below consume precisely this already-accounted request.
        let mut additional = vec![0u64; self.devices.len()];
        let mut additional_pinned = vec![0u64; self.devices.len()];
        for request in &pending {
            let claims = request
                .matrices
                .iter()
                .copied()
                .chain(request.layouts.iter().copied().map(GpuTracedClaim::workspace))
                .collect::<Vec<_>>();
            let slots = GpuPreparedSlotSnapshot::plan(&request.params, &claims)
                .map_err(PolyBackendError::GpuCalibration)?;
            let bytes = slots.into_iter().try_fold(0u64, |total, slot| {
                let identity = slot.identity();
                if matches!(
                    identity.kind(),
                    GpuPreparedSlotKind::CompletionEvent | GpuPreparedSlotKind::SubmissionStream
                ) {
                    return Ok(total);
                }
                let bytes = u64::try_from(identity.requested_backing_bytes())
                    .map_err(|_| PolyBackendError::InvalidInteger)?;
                if identity.kind() == GpuPreparedSlotKind::PinnedHost {
                    additional_pinned[request.device] = additional_pinned[request.device]
                        .checked_add(bytes)
                        .ok_or(PolyBackendError::InvalidInteger)?;
                    Ok(total)
                } else {
                    total.checked_add(bytes).ok_or(PolyBackendError::InvalidInteger)
                }
            })?;
            additional[request.device] = additional[request.device]
                .checked_add(bytes)
                .ok_or(PolyBackendError::InvalidInteger)?;
        }
        let mut provisioning = self
            .prepared_ledger
            .as_mut()
            .expect("existing prepared inventory has a ledger")
            .begin_prepared_provisioning(&additional, &additional_pinned)
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;

        let mut constructed = Vec::with_capacity(pending.len());
        let construction = (|| {
            for request in pending {
                // Production backing is allocated, not zeroed: these slots are
                // full-overwrite destinations claimed by dispatch, so a
                // placeholder zero kernel would be disposable work. The shapes
                // are appended uninitialized inside the primitives storage
                // constructor and never exist as a readable matrix here.
                let provisioning_claims = request
                    .matrices
                    .iter()
                    .copied()
                    .chain(request.layouts.iter().copied().map(GpuTracedClaim::workspace))
                    .collect::<Vec<_>>();
                let provisioning =
                    GpuPreparedProvisioningPermit::begin(&request.params, &provisioning_claims)
                        .map_err(PolyBackendError::GpuSubmission)?
                        .enter()
                        .map_err(PolyBackendError::GpuSubmission)?;
                let storage = GpuPreparedStorage::new(
                    Some(&request.params),
                    Vec::new(),
                    Some(&request.matrices),
                    Some(&request.layouts),
                )
                .map_err(PolyBackendError::GpuSubmission)?;
                provisioning.finish().map_err(PolyBackendError::GpuSubmission)?;
                let storage = Arc::new(storage);
                constructed.push((request.device, storage));
            }
            Ok::<(), PolyBackendError>(())
        })();
        construction?;
        provisioning
            .append(constructed)
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        provisioning.commit();
        Ok(())
    }
    /// Accept the fleet's complete prepared inventory at the explicit setup
    /// boundary. Native epoch receipts are the only way to construct the
    /// dispatcher ledger; production replay consumes that ledger directly.
    pub fn prepare_memory(
        &mut self,
        prepared: Vec<(usize, Arc<GpuPreparedStorage>)>,
        external_pool_exclusive: bool,
    ) -> Result<(), PolyBackendError> {
        crate::backend::poly_gpu::record_provisioning_begin();
        use mxx_primitives::poly::dcrt::gpu::{
            GpuAllocationEpochBoundary, GpuAllocationEpochObservation,
        };
        if !external_pool_exclusive {
            return Err(PolyBackendError::GpuSubmission(
                "prepared setup requires exclusive observation and an unconfigured fleet".into(),
            ));
        }
        let parameters = self.device_parameters();
        let mut identities = HashSet::new();
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
        let epochs = parameters
            .par_iter()
            .enumerate()
            .map(|(device, params)| {
                let inventory = prepared
                    .iter()
                    .filter(|(owner, _)| *owner == device)
                    .map(|(_, storage)| &**storage)
                    .collect::<Vec<_>>();
                GpuPreparedStorage::finish_setup(&inventory)
                    .map_err(PolyBackendError::GpuSubmission)?;
                match params
                    .observe_allocation_epoch(
                        params.device_ids()[0],
                        GpuAllocationEpochBoundary::InitialSetup,
                        true,
                    )
                    .map_err(PolyBackendError::GpuSubmission)?
                {
                    GpuAllocationEpochObservation::Verified(epoch) => Ok(epoch),
                    GpuAllocationEpochObservation::Unverified(reason) => {
                        Err(PolyBackendError::GpuSubmission(format!(
                            "prepared setup observation was not verified on device {device}: {reason:?}"
                        )))
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let ledger = GpuMemoryLedger::new(epochs, self.vram_percent, prepared)
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        self.set_memory_ledger(ledger)
    }

    pub fn set_memory_ledger(&mut self, ledger: GpuMemoryLedger) -> Result<(), PolyBackendError> {
        let params = self.device_parameters();
        let identities = params
            .iter()
            .zip(&self.devices)
            .map(|(params, (device, _))| params.execution_owner_id().map(|owner| (*device, owner)))
            .collect::<Option<Vec<_>>>()
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        if self.prepared_ledger.is_some() ||
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
        Ok(())
    }
}

fn prepared_output_codec_keys(
    program: &super::gpu_prepared_lowering::GpuPreparation,
    physical_devices: &[i32],
    instance_count: usize,
) -> Vec<(mxx_ir_core::types::WireRef, usize, i32)> {
    program
        .outputs
        .iter()
        .filter(|wire| program.wire_types[wire].matrix_type().is_some())
        .flat_map(|wire| {
            (0..instance_count.max(1)).flat_map(move |instance| {
                physical_devices.iter().copied().map(move |device| (*wire, instance, device))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::types::{ConcreteMatrixType, ConcreteWireType, NodeId, Port, WireRef};

    #[test]
    fn prepared_output_codec_keys_cover_each_noncontiguous_shard_and_instance() {
        let wire = WireRef { node: NodeId(3), port: Port(0) };
        let mut program = super::super::gpu_prepared_lowering::GpuPreparation::default();
        program.outputs = vec![wire].into_boxed_slice();
        program.wire_types.insert(
            wire,
            ConcreteWireType::Matrix(ConcreteMatrixType {
                modulus: 97.into(),
                ring_dimension: 8,
                rows: 1,
                columns: 2,
            }),
        );
        let keys = prepared_output_codec_keys(&program, &[2, 7], 2);
        assert_eq!(keys.len(), 4);
        assert_eq!(keys.iter().collect::<BTreeSet<_>>().len(), keys.len());
        assert_eq!(keys, vec![(wire, 0, 2), (wire, 0, 7), (wire, 1, 2), (wire, 1, 7)]);
    }
}

fn prepared_device_index_from_identities(
    identities: &[(i32, u64)],
    physical_device: i32,
) -> Option<usize> {
    identities.iter().position(|(device, _)| *device == physical_device)
}
