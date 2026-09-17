//! Materialize the fixed prepared inventory from a resolved resource plan.
//!
//! Prepared lowering resolves retained owners and native resource claims once
//! at warmup. Published slots are then replayed without dynamic operation
//! selection, candidate fitting, or discovery during execution.

use super::{
    super::gpu_prepared::{self, PreparedRuntimeValue},
    *,
};
use crate::gpu_memory::GpuMemoryLedger;
use mxx_ir_core::{
    ValidatedGraph,
    types::{ConcreteWireType, WireRef},
};
use mxx_primitives::{
    matrix::gpu_dcrt_poly::{
        GpuPreparedProvisioningPermit, GpuPreparedRequest, GpuPreparedSlotKind,
        GpuPreparedSlotSnapshot, GpuPreparedStorage, GpuPreparedWorkspaceClaim,
        GpuPreparedWorkspaceLayout, GpuPreparedWorkspaceRequest, GpuTracedClaim,
        PreparedAllocationLayout, PreparedOwnerLayout, PreparedPlanLayout, PreparedStorageRecipe,
    },
    poly::dcrt::gpu::{GPU_POLY_FORMAT_COEFF, GPU_POLY_FORMAT_EVAL},
};
use std::collections::{BTreeMap, BTreeSet};

/// Close all matrix state identities before the pure resource plan is built.
/// Lowering only knows representative wire shapes; warmup is the first point
/// where concrete CRT parameters and runtime input levels are available.
fn finalize_prepared_matrix_contexts(
    backend: &GpuDcrtBackend,
    program: &mut super::gpu_prepared_lowering::GpuPreparation,
    runtime_inputs: &[PreparedRuntimeValue],
) -> Result<(), PolyBackendError> {
    use super::gpu_prepared_lowering::{MatrixSite, PreparedFormat, family_leaf_wires};
    if program.instance_count == 0 {
        return Err(PolyBackendError::GpuSubmission(
            "prepared program must declare at least one execution instance".into(),
        ));
    }
    let expanded = gpu_prepared::expand_prepared_runtime_inputs(program, runtime_inputs)
        .map_err(PolyBackendError::GpuSubmission)?;
    let mut runtime_states = BTreeMap::<(WireRef, i32), (usize, usize, PreparedFormat)>::new();
    for (wire, input) in program.runtime_input_wires.iter().copied().zip(expanded) {
        match input {
            PreparedRuntimeValue::FleetMatrix(value) => {
                let first = value.shards().first().ok_or(PolyBackendError::InvalidConstantShape)?;
                // Level/format are one fleet-input contract. Context identity
                // is physical-device state and may differ between shards.
                if value.shards().iter().any(|candidate| {
                    candidate.value.level() != first.value.level() ||
                        candidate.value.is_ntt() != first.value.is_ntt()
                }) {
                    return Err(PolyBackendError::GpuSubmission(
                        "prepared fleet matrix shards must agree on level and format".into(),
                    ));
                }
                for shard in value.shards() {
                    runtime_states.insert(
                        (wire, shard.device_id),
                        (
                            shard.value.params().context_identity(),
                            shard.value.level(),
                            if shard.value.is_ntt() {
                                PreparedFormat::Evaluation
                            } else {
                                PreparedFormat::Coefficient
                            },
                        ),
                    );
                }
            }
            PreparedRuntimeValue::Trapdoor { public: value, .. } => {
                let first = value.shards().first().ok_or(PolyBackendError::InvalidConstantShape)?;
                if value.shards().iter().any(|candidate| {
                    candidate.value.level() != first.value.level() ||
                        candidate.value.is_ntt() != first.value.is_ntt()
                }) {
                    return Err(PolyBackendError::GpuSubmission(
                        "prepared fleet trapdoor shards must agree on level and format".into(),
                    ));
                }
                for shard in value.shards() {
                    runtime_states.insert(
                        (wire, shard.device_id),
                        (
                            shard.value.params().context_identity(),
                            shard.value.level(),
                            if shard.value.is_ntt() {
                                PreparedFormat::Evaluation
                            } else {
                                PreparedFormat::Coefficient
                            },
                        ),
                    );
                }
            }
            PreparedRuntimeValue::HostMatrix { matrix_type, bytes, .. } => {
                for (device, device_backend) in &backend.devices {
                    let parameters = device_backend
                        .parameters(&matrix_type)
                        .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
                    let layout = GpuDCRTPolyMatrix::cpu_staging_layout(parameters, &bytes)
                        .map_err(PolyBackendError::GpuSubmission)?;
                    runtime_states.insert(
                        (wire, *device),
                        (
                            parameters.context_identity(),
                            layout.level,
                            if layout.is_ntt {
                                PreparedFormat::Evaluation
                            } else {
                                PreparedFormat::Coefficient
                            },
                        ),
                    );
                }
            }
            _ => {}
        }
    }

    // Outputs created during warmup do not have caller-owned runtime payloads,
    // so they cannot contribute a state entry through `runtime_inputs`. Resolve
    // their physical context from the producing operation before validating the
    // canonical identity table. Matrix operations inherit level/format from a
    // matrix input when available. A producer with no matrix input is an
    // explicit source (sampler, constant, or scalar-to-matrix conversion), so
    // its device parameters establish the full level while the lowered output
    // format remains authoritative. A node that has matrix inputs but cannot
    // inherit one remains unresolved and is rejected below; it must never
    // receive fabricated context or level.
    // Propagate finalized state through the complete topology. A single
    // traversal is insufficient for nested/aliased scopes whose producer
    // appears after a consumer in the lowered command order; iterate until no
    // new wire/device state is discovered, then fail explicitly below if a
    // graph still has no physical source.
    loop {
        let state_count = runtime_states.len();
        for node in &program.topology.nodes {
            let Some((inputs, outputs)) = program.node_bindings.get(&node.id) else { continue };
            let input_matrix_wires = inputs
                .iter()
                .flat_map(|wire| family_leaf_wires(program, *wire))
                .filter(|wire| program.values.contains_key(wire))
                .collect::<Vec<_>>();
            let output_matrix_wires = outputs
                .iter()
                .flat_map(|wire| family_leaf_wires(program, *wire))
                .filter(|wire| program.values.contains_key(wire))
                .collect::<Vec<_>>();
            for wire in output_matrix_wires {
                let Some(matrix) =
                    program.wire_types.get(&wire).and_then(ConcreteWireType::matrix_type)
                else {
                    continue;
                };
                for (device, device_backend) in &backend.devices {
                    if runtime_states.contains_key(&(wire, *device)) {
                        continue;
                    }
                    let parameters = device_backend
                        .parameters(matrix)
                        .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
                    let inherited = input_matrix_wires
                        .iter()
                        .find_map(|input| runtime_states.get(&(*input, *device)).copied());
                    let format =
                        program.values.get(&wire).map(|location| location.format).ok_or_else(
                            || {
                                PolyBackendError::GpuSubmission(format!(
                                    "prepared matrix wire {wire:?} has no lowered format"
                                ))
                            },
                        )?;
                    let level = if let Some((_, level, _)) = inherited {
                        level.min(parameters.moduli().len().saturating_sub(1))
                    } else if input_matrix_wires.is_empty() {
                        parameters
                            .moduli()
                            .len()
                            .checked_sub(1)
                            .ok_or_else(|| {
                                PolyBackendError::GpuSubmission(format!(
                                    "prepared matrix source wire {wire:?} has no CRT level on device {device}"
                                ))
                            })?
                    } else {
                        continue;
                    };
                    runtime_states
                        .insert((wire, *device), (parameters.context_identity(), level, format));
                }
            }
        }
        if runtime_states.len() == state_count {
            break;
        }
    }

    type WireState = (usize, usize, i32, PreparedFormat);
    let mut state_for_wire = BTreeMap::<(WireRef, i32), WireState>::new();
    for (wire, _location) in &program.values {
        let Some(matrix) = program.wire_types.get(wire).and_then(ConcreteWireType::matrix_type)
        else {
            continue;
        };
        for (device, device_backend) in &backend.devices {
            let parameters = device_backend
                .parameters(matrix)
                .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
            let (context, level, format) = runtime_states
                .get(&(*wire, *device))
                .copied()
                .ok_or_else(|| {
                    PolyBackendError::GpuSubmission(format!(
                        "prepared matrix wire {wire:?} has no finalized runtime state on device {device}"
                    ))
                })?;
            if parameters.context_identity() != context || level >= parameters.moduli().len() {
                return Err(PolyBackendError::GpuSubmission(format!(
                    "prepared matrix wire {wire:?} runtime state does not match device {device}"
                )));
            }
            state_for_wire.insert((*wire, *device), (context, level, *device, format));
        }
    }
    let mut next_owner = program
        .values
        .values()
        .map(|location| location.owner)
        .chain(program.storage_capacities.keys().copied())
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or(PolyBackendError::InvalidInteger)?;
    type OwnerState = (u64, usize, usize, PreparedFormat);
    let mut owners = BTreeMap::<OwnerState, u64>::new();
    let mut owner_for =
        |owner: u64, state: (usize, usize, PreparedFormat)| -> Result<u64, PolyBackendError> {
            let key = (owner, state.0, state.1, state.2);
            if let Some(mapped) = owners.get(&key) {
                return Ok(*mapped);
            }
            let mapped = if !owners.values().any(|candidate| *candidate == owner) {
                owner
            } else {
                let mapped = next_owner;
                next_owner = next_owner.checked_add(1).ok_or(PolyBackendError::InvalidInteger)?;
                mapped
            };
            owners.insert(key, mapped);
            Ok(mapped)
        };
    let original_value_owners = program
        .values
        .iter()
        .map(|(wire, location)| (*wire, location.owner))
        .collect::<BTreeMap<_, _>>();
    let original_value_locations = program.values.clone();
    let original_capacities = program.storage_capacities.clone();
    let mut remapped_capacities = BTreeMap::new();
    // Remap ordinary locations in the map directly and use explicit
    // source-wire provenance for every auxiliary location.  Geometry is not
    // an identity: identical aliases are valid and must not be reverse
    // matched by owner/shape scans.
    program.finalized_matrices.clear();
    for (wire, location) in &mut program.values {
        let original_owner = original_value_owners[wire];
        let physical_states = backend
            .devices
            .iter()
            .map(|(device, _)| {
                let state = state_for_wire
                    .get(&(*wire, *device))
                    .copied()
                    .ok_or(PolyBackendError::UnsupportedPlacement)?;
                let mapped = owner_for(original_owner, (state.0, state.1, state.3))?;
                if let Some(capacity) = original_capacities.get(&original_owner) {
                    remapped_capacities.insert(mapped, *capacity);
                }
                let (capacity_rows, capacity_columns) = original_capacities
                    .get(&original_owner)
                    .copied()
                    .unwrap_or((location.rows.end, location.columns.end));
                for instance in 0..program.instance_count {
                    program
                        .finalized_matrices
                        .insert(
                            MatrixSite::Ordinary { wire: *wire, device: *device, instance },
                            super::gpu_prepared_lowering::ValueLocation {
                                owner: mapped,
                                rows: location.rows.clone(),
                                columns: location.columns.clone(),
                                level: state.1,
                                format: state.3,
                                device: *device,
                            },
                            state.0,
                            capacity_rows,
                            capacity_columns,
                        )
                        .map_err(PolyBackendError::GpuSubmission)?;
                }
                Ok((*device, state, mapped))
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        let (_, (_, level, _, format), mapped) =
            physical_states.first().copied().ok_or(PolyBackendError::UnsupportedPlacement)?;
        location.level = level;
        location.format = format;
        location.device = physical_states[0].0;
        location.owner = mapped;
    }
    let mut remap_aux_location = |location: &mut super::gpu_prepared_lowering::ValueLocation,
                                  wire: WireRef|
     -> Result<(), PolyBackendError> {
        let original = original_value_locations.get(&wire).ok_or_else(|| {
            PolyBackendError::GpuSubmission(
                "prepared auxiliary location has missing state-wire provenance".into(),
            )
        })?;
        let target_device = location.device;
        let Some((context, level, physical_device, format)) =
            state_for_wire.get(&(wire, target_device)).copied()
        else {
            return Err(PolyBackendError::GpuSubmission(
                "prepared auxiliary location has no finalized wire state".into(),
            ));
        };
        // A finalized wire may legitimately have different contexts on
        // different physical devices.  The explicit (wire, device)
        // lookup is authoritative; a conflicting state is malformed.
        if let Some((_, original_level, _, original_format)) =
            state_for_wire.get(&(wire, original.device)).copied()
        {
            if original_level != level && original.device == target_device ||
                original_format != format && original.device == target_device
            {
                return Err(PolyBackendError::GpuSubmission(
                    "prepared auxiliary location conflicts with finalized wire state".into(),
                ));
            }
        }
        location.level = level;
        location.format = format;
        location.owner = owner_for(original.owner, (context, level, format))?;
        location.device = physical_device;
        Ok(())
    };
    let family_provenance = program
        .family_wires
        .iter()
        .map(|(family, wires)| (*family, wires.clone()))
        .collect::<BTreeMap<_, _>>();
    for (family, members) in &mut program.family_members {
        let wires = family_provenance
            .get(family)
            .map(|wires| {
                wires
                    .iter()
                    .copied()
                    .filter(|wire| original_value_locations.contains_key(wire))
                    .collect::<Vec<_>>()
            })
            .ok_or_else(|| {
                PolyBackendError::GpuSubmission(
                    "prepared family has missing member-wire provenance".into(),
                )
            })?;
        if members.len() != wires.len() {
            return Err(PolyBackendError::GpuSubmission(
                "prepared family provenance cardinality mismatch".into(),
            ));
        }
        for (location, wire) in members.iter_mut().zip(wires.into_iter()) {
            remap_aux_location(location, wire)?;
        }
    }
    let selection_provenance = program.selection_candidate_wires.clone();
    for (node_id, selection) in &mut program.selection_commands {
        let wires = selection_provenance.get(node_id).ok_or_else(|| {
            PolyBackendError::GpuSubmission(
                "prepared selection has missing candidate-wire provenance".into(),
            )
        })?;
        match selection {
            super::gpu_prepared_lowering::PreparedSelection::Static { location } => {
                let wire = wires.first().copied().ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "prepared static selection has no candidate".into(),
                    )
                })?;
                remap_aux_location(location, wire)?;
            }
            super::gpu_prepared_lowering::PreparedSelection::Dynamic { candidates, .. } |
            super::gpu_prepared_lowering::PreparedSelection::Select { candidates, .. } => {
                if candidates.len() != wires.len() {
                    return Err(PolyBackendError::GpuSubmission(
                        "prepared selection provenance cardinality mismatch".into(),
                    ));
                }
                for (location, wire) in candidates.iter_mut().zip(wires.iter().copied()) {
                    remap_aux_location(location, wire)?;
                }
            }
            super::gpu_prepared_lowering::PreparedSelection::ScalarStatic { .. } |
            super::gpu_prepared_lowering::PreparedSelection::ScalarDynamic { .. } |
            super::gpu_prepared_lowering::PreparedSelection::ScalarSelect { .. } => {}
        }
    }
    program.storage_capacities = remapped_capacities;
    // Add one exact logical location for every finite output variant. These
    // stores are consumed by the same owner remapping used by generic replay.
    for (node, source) in &program.node_sources {
        for (variant, types) in source.variant_output_types.iter().enumerate() {
            for (port, ty) in types.iter().enumerate() {
                let Some(matrix) = ty.matrix_type() else { continue };
                let wire = program
                    .node_bindings
                    .get(node)
                    .and_then(|(_, outputs)| outputs.get(port))
                    .copied()
                    .ok_or(PolyBackendError::GpuSubmission(
                        "prepared variant output is unbound".into(),
                    ))?;
                let base = program.values.get(&wire).cloned().ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "prepared variant output has no location".into(),
                    )
                })?;
                let original_owner =
                    original_value_owners.get(&wire).copied().unwrap_or(base.owner);
                for (device, device_backend) in &backend.devices {
                    let parameters = device_backend
                        .parameters(matrix)
                        .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
                    let level = parameters.moduli().len().saturating_sub(1);
                    let owner = owner_for(
                        original_owner,
                        (parameters.context_identity(), level, base.format),
                    )?;
                    let (capacity_rows, capacity_columns) = original_capacities
                        .get(&original_owner)
                        .copied()
                        .unwrap_or((matrix.rows, matrix.columns));
                    for instance in 0..program.instance_count {
                        program
                            .finalized_matrices
                            .insert(
                                MatrixSite::Variant {
                                    node: *node,
                                    variant,
                                    port,
                                    device: *device,
                                    instance,
                                },
                                super::gpu_prepared_lowering::ValueLocation {
                                    owner,
                                    rows: 0..matrix.rows,
                                    columns: 0..matrix.columns,
                                    level,
                                    format: base.format,
                                    device: *device,
                                },
                                parameters.context_identity(),
                                capacity_rows,
                                capacity_columns,
                            )
                            .map_err(PolyBackendError::GpuSubmission)?;
                    }
                    if let Some(capacity) = original_capacities.get(&original_owner) {
                        program.storage_capacities.entry(owner).or_insert(*capacity);
                    }
                }
            }
        }
    }
    // Threshold decode consumes a dedicated coefficient-domain staging owner;
    // it is not represented by an IR value and therefore must be finalized
    // explicitly for every execution instance and physical device.
    for node in &program.topology.nodes {
        if !matches!(
            node.command.operation,
            super::gpu_prepared_lowering::PreparedOperation::Gpu(
                super::gpu_prepared_lowering::PreparedGpuOperation::ThresholdDecode
            )
        ) {
            continue;
        }
        let source_wire = program
            .node_bindings
            .get(&node.id)
            .and_then(|(inputs, _)| inputs.first())
            .and_then(|wire| super::gpu_prepared_lowering::first_matrix_wire(&program, *wire))
            .ok_or_else(|| {
                PolyBackendError::GpuSubmission("prepared threshold source is missing".into())
            })?;
        let source_location = original_value_locations.get(&source_wire).ok_or_else(|| {
            PolyBackendError::GpuSubmission("prepared threshold source location is missing".into())
        })?;
        for instance in 0..program.instance_count {
            for (device, _) in &backend.devices {
                let (context, level, _, _) =
                    state_for_wire.get(&(source_wire, *device)).copied().ok_or_else(|| {
                        PolyBackendError::GpuSubmission(
                            "prepared threshold source has no finalized state".into(),
                        )
                    })?;
                let owner = owner_for(
                    source_location.owner,
                    (context, level, PreparedFormat::Coefficient),
                )?;
                program
                    .finalized_matrices
                    .insert(
                        super::gpu_prepared_lowering::MatrixSite::HostStaging {
                            node: node.id,
                            instance,
                            device: *device,
                        },
                        super::gpu_prepared_lowering::ValueLocation {
                            owner,
                            rows: 0..source_location.rows.end,
                            columns: 0..source_location.columns.end,
                            level,
                            format: PreparedFormat::Coefficient,
                            device: *device,
                        },
                        context,
                        source_location.rows.end,
                        source_location.columns.end,
                    )
                    .map_err(PolyBackendError::GpuSubmission)?;
            }
        }
    }
    for node in &program.topology.nodes {
        if !matches!(
            node.command.operation,
            super::gpu_prepared_lowering::PreparedOperation::Gpu(
                super::gpu_prepared_lowering::PreparedGpuOperation::RnsReadback
            )
        ) {
            continue;
        }
        if matches!(
            program.node_sources.get(&node.id).map(|source| source.kind()),
            Some(mxx_ir_core::node::NodeKind::PolynomialValues { evaluation: true })
        ) {
            // Evaluation-domain polynomial values use the source owner
            // directly; no coefficient staging owner or inverse transform is
            // part of their finalized command contract.
            continue;
        }
        let source_wire = program
            .node_bindings
            .get(&node.id)
            .and_then(|(inputs, _)| inputs.first())
            .and_then(|wire| super::gpu_prepared_lowering::first_matrix_wire(program, *wire))
            .ok_or_else(|| {
                PolyBackendError::GpuSubmission("prepared readback source is missing".into())
            })?;
        let matrix = program
            .wire_types
            .get(&source_wire)
            .and_then(ConcreteWireType::matrix_type)
            .ok_or_else(|| {
            PolyBackendError::GpuSubmission("prepared readback source is not a matrix".into())
        })?;
        for instance in 0..program.instance_count {
            for (device, device_backend) in &backend.devices {
                let source_identity = program
                    .finalized_matrices
                    .ordinary_for_instance(source_wire, *device, instance)
                    .ok_or_else(|| {
                        PolyBackendError::GpuSubmission(
                            "prepared readback source has no per-device finalized state".into(),
                        )
                    })?;
                let source_context = source_identity.physical.context_identity;
                let source_level = source_identity.physical.level;
                let parameters = device_backend
                    .parameters(matrix)
                    .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
                if parameters.context_identity() != source_context ||
                    source_level >= parameters.moduli().len()
                {
                    return Err(PolyBackendError::GpuSubmission(
                        "prepared readback source context/level does not match its device".into(),
                    ));
                }
                let owner = owner_for(
                    original_value_owners[&source_wire],
                    (source_context, source_level, PreparedFormat::Coefficient),
                )?;
                let (capacity_rows, capacity_columns) = original_capacities
                    .get(&original_value_owners[&source_wire])
                    .copied()
                    .unwrap_or((matrix.rows, matrix.columns));
                program
                    .finalized_matrices
                    .insert(
                        MatrixSite::HostStaging { node: node.id, instance, device: *device },
                        super::gpu_prepared_lowering::ValueLocation {
                            owner,
                            rows: 0..matrix.rows,
                            columns: 0..matrix.columns,
                            level: source_level,
                            format: PreparedFormat::Coefficient,
                            device: *device,
                        },
                        source_context,
                        capacity_rows,
                        capacity_columns,
                    )
                    .map_err(PolyBackendError::GpuSubmission)?;
                if let Some(capacity) =
                    original_capacities.get(&original_value_owners[&source_wire])
                {
                    program.storage_capacities.entry(owner).or_insert(*capacity);
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct PreparedScheduleStreamKey {
    pub instance: usize,
    pub command: usize,
    pub stream: usize,
    pub device: i32,
}

fn exact_prepared_stream_slot(
    allocations: &[super::gpu_prepared_lowering::PreparedAllocationClaim],
    key: &mxx_primitives::matrix::gpu_dcrt_poly::PreparedResourceKey,
    kind: i32,
    label: &str,
) -> Result<super::gpu_prepared_lowering::PreparedSlotRef, PolyBackendError> {
    let matches = allocations
        .iter()
        .filter(|allocation| allocation.layout.kind == kind && allocation.layout.key == *key)
        .collect::<Vec<_>>();
    let allocation = match matches.as_slice() {
        [allocation] => *allocation,
        [] => {
            return Err(PolyBackendError::GpuSubmission(format!(
                "prepared {label} stream has no matching allocation"
            )))
        }
        _ => {
            return Err(PolyBackendError::GpuSubmission(format!(
                "prepared {label} stream has ambiguous matching allocations"
            )))
        }
    };
    allocation.slot.clone().ok_or_else(|| {
        PolyBackendError::GpuSubmission(format!(
            "prepared {label} stream allocation slot is unresolved"
        ))
    })
}

impl GpuDcrtBackend {
    fn prepared_output_codec_claims(
        &self,
        program: &super::gpu_prepared_lowering::GpuPreparation,
        plan: &super::gpu_prepared_lowering::PreparedResourcePlan,
        resources: &super::gpu_prepared_lowering::PreparedResolvedResources,
        physical_devices: &[i32],
        instance_count: usize,
    ) -> Result<
        Vec<(GpuDCRTPolyParams, Vec<GpuTracedClaim>, Vec<GpuPreparedWorkspaceClaim>)>,
        PolyBackendError,
    > {
        if instance_count == 0 {
            return Err(PolyBackendError::GpuSubmission(
                "prepared output codec requires at least one execution instance".into(),
            ));
        }
        let mut claims = Vec::new();
        for (wire, instance, physical_device) in
            prepared_output_codec_keys(program, physical_devices, instance_count)
        {
            let Some(matrix) = program.wire_types[&wire].matrix_type() else {
                continue;
            };
            let identity = program
                .finalized_matrices
                .ordinary_for_instance(wire, physical_device, instance)
                .ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "prepared output has no finalized per-device state".into(),
                    )
                })?;
            let context_identity = identity.physical.context_identity;
            let level = identity.physical.level;
            let format = identity.physical.format;
            let backend = self
                .devices
                .iter()
                .find(|(device, _)| *device == physical_device)
                .map(|(_, backend)| backend)
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let params = backend.parameters(matrix)?.clone();
            if !params.device_ids().contains(&physical_device) ||
                params.context_identity() != context_identity ||
                level >= params.moduli().len()
            {
                return Err(PolyBackendError::GpuSubmission(
                    "prepared output state does not match its exact device context".into(),
                ));
            }
            let store = plan
                .stores
                .iter()
                .find(|store| {
                    let Some(store_identity) = plan.finalized_matrices.identity(store.matrix_id)
                    else {
                        return false;
                    };
                    store.wire == wire &&
                        store.instance == instance &&
                        store_identity.physical.device == physical_device &&
                        store_identity.physical.level == level &&
                        store_identity.physical.context_identity == context_identity
                })
                .ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "prepared output codec has no exact resource-plan store".into(),
                    )
                })?;
            let owner = resources
                .owners
                .iter()
                .find(|owner| {
                    owner.key.matrix_id == store.matrix_id && owner.key.instance == store.instance
                })
                .ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "prepared output codec has no exact owner layout".into(),
                    )
                })?;
            let codec_plan = PreparedPlanLayout::borrowed_compact_store_with_owner(
                &params,
                matrix.rows,
                matrix.columns,
                level,
                match format {
                    super::gpu_prepared_lowering::PreparedFormat::Coefficient => {
                        GPU_POLY_FORMAT_COEFF
                    }
                    super::gpu_prepared_lowering::PreparedFormat::Evaluation => {
                        GPU_POLY_FORMAT_EVAL
                    }
                },
                &owner.layout,
            )
            .map_err(PolyBackendError::GpuSubmission)?;
            let Some((stream_allocation, transfer_allocation, stream)) =
                validate_borrowed_codec_descriptor(
                    matrix.rows,
                    matrix.columns,
                    codec_plan.allocations(),
                    codec_plan.streams(),
                    owner.layout.execution_owner_identity(),
                    context_identity,
                    physical_device,
                )
                .map_err(PolyBackendError::GpuSubmission)?
            else {
                claims.push((params, Vec::new(), Vec::new()));
                continue;
            };
            if !owner.layout.contains_resource_key(&stream.key) {
                return Err(PolyBackendError::GpuSubmission(
                    "borrowed compact store stream provenance differs from its owner".into(),
                ));
            }
            let placement = GpuPreparedWorkspaceClaim::new(
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::SubmissionStream,
                    bytes: stream_allocation.bytes,
                    alignment: stream_allocation.alignment,
                },
                stream.key.device,
                usize::try_from(stream.key.partition).map_err(|_| {
                    PolyBackendError::GpuSubmission(
                        "borrowed compact store stream has no physical partition".into(),
                    )
                })?,
                stream.pool_slot,
            );
            let transfer_placement = GpuPreparedWorkspaceClaim::new(
                GpuPreparedWorkspaceLayout {
                    kind: GpuPreparedSlotKind::TransferWorkspace,
                    bytes: transfer_allocation.bytes,
                    alignment: transfer_allocation.alignment,
                },
                stream.key.device,
                placement.partition,
                placement.stream_slot,
            );
            let mut codec_claims = Vec::new();
            let mut codec_placements = Vec::new();
            for (allocation, placement) in
                [(stream_allocation, placement), (transfer_allocation, transfer_placement)]
            {
                let kind = match allocation.allocation_kind() {
                    Some(mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationKind::SubmissionStream) =>
                        GpuPreparedSlotKind::SubmissionStream,
                    Some(mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationKind::TransferWorkspace) =>
                        GpuPreparedSlotKind::TransferWorkspace,
                    _ => {
                        return Err(PolyBackendError::GpuSubmission(
                            "borrowed compact store plan contains an unsupported allocation".into(),
                        ));
                    }
                };
                codec_claims.push(GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                    kind,
                    bytes: allocation.bytes,
                    alignment: allocation.alignment,
                }));
                if placement.layout.kind != kind ||
                    placement.layout.bytes != allocation.bytes ||
                    placement.layout.alignment != allocation.alignment
                {
                    return Err(PolyBackendError::GpuSubmission(
                        "borrowed compact store placement differs from its allocation".into(),
                    ));
                }
                codec_placements.push(placement);
            }
            claims.push((params, codec_claims, codec_placements));
        }
        Ok(claims)
    }

    /// Resolve every slot described by the metadata resolver in one setup pass.
    /// This is deliberately separate from schedule provisioning: the latter
    /// may only consume the region and slot references returned here.
    pub(crate) fn reserve_resolved_resources(
        &mut self,
        plan: &super::gpu_prepared_lowering::PreparedResourcePlan,
        resources: &mut super::gpu_prepared_lowering::PreparedResolvedResources,
        extra_claims: &[(GpuDCRTPolyParams, Vec<GpuTracedClaim>, Vec<GpuPreparedWorkspaceClaim>)],
    ) -> Result<
        (
            Arc<crate::gpu_memory::GpuMemoryRegion>,
            BTreeMap<u64, Arc<GpuPreparedStorage>>,
            Vec<Box<[super::gpu_prepared_lowering::PreparedSlotRef]>>,
        ),
        PolyBackendError,
    > {
        let store_identity = |store: &super::gpu_prepared_lowering::PreparedStorePlan| {
            plan.finalized_matrices.identity(store.matrix_id).ok_or_else(|| {
                PolyBackendError::GpuSubmission(
                    "prepared store has no canonical matrix identity".into(),
                )
            })
        };
        let mut claims = BTreeMap::<usize, (GpuDCRTPolyParams, Vec<GpuTracedClaim>)>::new();
        #[derive(Debug)]
        enum ClaimTarget {
            Owner(usize),
            OwnerInputCopy(usize, usize),
            Command(usize, usize),
            Composite(usize, usize),
            Replay(usize, usize),
            Schedule(usize, usize),
            ScalarBuffer(usize, usize),
            Unbound { codec: usize, claim: usize, placement: GpuPreparedWorkspaceClaim },
        }
        let mut claim_entries = Vec::<(
            GpuDCRTPolyParams,
            GpuTracedClaim,
            ClaimTarget,
            Option<PreparedAllocationLayout>,
        )>::new();

        // Logical stores are the complete retained owner set. Native stage
        // layouts below add only their auxiliary resources and events.
        for store in &plan.stores {
            let identity = store_identity(store)?;
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
                .find(|params| {
                    params.device_ids().contains(&identity.physical.device) &&
                        params.context_identity() == identity.physical.context_identity
                })
                .ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "resolved store has no device parameters".into(),
                    )
                })?;
            let claim = GpuTracedClaim::matrix(
                identity.physical.capacity_rows,
                identity.physical.capacity_columns,
                identity.physical.level,
                identity.physical.format ==
                    super::gpu_prepared_lowering::PreparedFormat::Evaluation,
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
                    owner.key.matrix_id == store.matrix_id && owner.key.instance == store.instance
                })
                .ok_or_else(|| {
                    PolyBackendError::GpuSubmission("resolved store has no resolved owner".into())
                })?;
            claim_entries.push((params, claim, ClaimTarget::Owner(owner_index), None));
        }

        // Root matrix inputs are materialized by synthesized rectangular-copy
        // commands, so their completion events are not represented by an IR
        // command recipe. Admit one exact event per active limb beside each
        // physical owner; the bind pass consumes these slots positionally.
        for (owner_index, owner) in resources.owners.iter().enumerate() {
            let store = plan
                .stores
                .iter()
                .find(|store| {
                    store.matrix_id == owner.key.matrix_id && store.instance == owner.key.instance
                })
                .ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "input-copy owner has no exact matrix store".into(),
                    )
                })?;
            let matrix =
                store.wire_type.as_ref().and_then(ConcreteWireType::matrix_type).ok_or_else(
                    || {
                        PolyBackendError::GpuSubmission(
                            "input-copy owner store has no matrix type".into(),
                        )
                    },
                )?;
            let identity = store_identity(store)?;
            let params = self
                .resource_parameters(matrix)
                .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?
                .into_iter()
                .find(|params| {
                    params.device_ids().contains(&identity.physical.device) &&
                        params.context_identity() == identity.physical.context_identity
                })
                .ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "input-copy owner has no exact device parameters".into(),
                    )
                })?;
            let input_copy_count = identity.physical.level.checked_add(1).ok_or_else(|| {
                PolyBackendError::GpuSubmission("input-copy event count overflow".into())
            })?;
            for ordinal in 0..input_copy_count {
                let claim = GpuTracedClaim::workspace(
                    mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedWorkspaceLayout {
                        bytes: 0,
                        alignment: 1,
                        kind: GpuPreparedSlotKind::CompletionEvent,
                    },
                );
                claims
                    .entry(params.context_identity())
                    .or_insert_with(|| (params.clone(), Vec::new()))
                    .1
                    .push(claim);
                claim_entries.push((
                    params.clone(),
                    claim,
                    ClaimTarget::OwnerInputCopy(owner_index, ordinal),
                    None,
                ));
            }
        }

        let params_for_layout =
            |layout: &mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationLayout,
             store_hint: Option<usize>|
             -> Result<GpuDCRTPolyParams, String> {
                // Pinned host buffers are deliberately keyed with device -1;
                // their resource domain is the device owner of the stage's
                // matrix store, not a fabricated host device.
                let expected_level = usize::try_from(layout.level).ok();
                let expected_format = match layout.format {
                    GPU_POLY_FORMAT_EVAL => {
                        Some(super::gpu_prepared_lowering::PreparedFormat::Evaluation)
                    }
                    GPU_POLY_FORMAT_COEFF => {
                        Some(super::gpu_prepared_lowering::PreparedFormat::Coefficient)
                    }
                    _ => None,
                };
                // Only matrix allocations carry meaningful level/format
                // geometry.  Auxiliary workspaces/events/streams use the
                // native zero geometry fields and inherit their device
                // parameters from the provenance store.
                let matrix_layout = layout.kind == 0;
                if layout.key.stage().is_none() {
                    return Err("native claim has an unknown resource role".into());
                }
                // Host-side claims intentionally use the sentinel placement
                // (partition/device = -1, context/owner = 0).  Their store
                // provenance is carried by `store_hint`; requiring the
                // sentinel context or execution owner to match the matrix
                // store would reject every pinned allocation.  Device claims,
                // in contrast, must match all physical identity fields.
                let host_key = layout.key.device < 0 || layout.key.partition < 0;
                let store: &super::gpu_prepared_lowering::PreparedStorePlan = if let Some(index) =
                    store_hint
                {
                    // A supplied hint is authoritative. Resolve it first and
                    // validate every native identity component against the
                    // hinted store; never use execution-owner identity to
                    // discover a different store and only compare instance.
                    let store = plan.stores.get(index).ok_or("native claim store is invalid")?;
                    let identity = store_identity(store).map_err(|error| error.to_string())?;
                    if layout.key.device >= 0 && identity.physical.device != layout.key.device {
                        return Err("native claim store device differs from layout".into());
                    }
                    if !host_key &&
                        layout.key.context_identity != identity.physical.context_identity as u64
                    {
                        return Err("native claim store context differs from layout".into());
                    }
                    if layout.key.instance != store.instance as u64 {
                        return Err("native claim store instance differs from layout".into());
                    }
                    if matrix_layout &&
                        expected_level.is_some_and(|level| identity.physical.level != level)
                    {
                        return Err("native claim store level differs from layout".into());
                    }
                    let scratch_matrix =
                        matrix_layout && layout.key.stage().map(|role| role as i32) == Some(6);
                    if matrix_layout &&
                        !scratch_matrix &&
                        expected_format.is_some_and(|format| identity.physical.format != format)
                    {
                        return Err("native claim store format differs from layout".into());
                    }
                    let owners = resources
                        .owners
                        .iter()
                        .filter(|owner| {
                            let Some(identity) =
                                resources.finalized_matrices.identity(owner.key.matrix_id)
                            else {
                                return false;
                            };
                            owner.key.matrix_id == store.matrix_id &&
                                owner.key.instance == store.instance &&
                                (host_key ||
                                    (owner.layout.execution_owner_identity() ==
                                        layout.key.execution_owner_identity &&
                                        identity.physical.context_identity as u64 ==
                                            layout.key.context_identity &&
                                        owner.key.instance as u64 == layout.key.instance &&
                                        owner.layout.contains_resource_key(&layout.key)))
                        })
                        .collect::<Vec<_>>();
                    if host_key {
                        // There is no physical owner resource for a host
                        // sentinel key.  `store_hint` is the canonical
                        // provenance for this claim.
                    } else if owners.len() != 1 {
                        return Err(if owners.is_empty() {
                            "native claim store has no exact owner"
                        } else {
                            "native claim store has ambiguous owners"
                        }
                        .into());
                    }
                    store
                } else {
                    let owners = resources
                        .owners
                        .iter()
                        .filter(|owner| {
                            let Some(identity) =
                                resources.finalized_matrices.identity(owner.key.matrix_id)
                            else {
                                return false;
                            };
                            owner.layout.execution_owner_identity() ==
                                layout.key.execution_owner_identity &&
                                identity.physical.context_identity as u64 ==
                                    layout.key.context_identity &&
                                owner.key.instance as u64 == layout.key.instance &&
                                owner.layout.contains_resource_key(&layout.key) &&
                                (layout.key.device < 0 ||
                                    identity.physical.device == layout.key.device) &&
                                (!matrix_layout ||
                                    expected_level
                                        .is_none_or(|level| identity.physical.level == level)) &&
                                (!matrix_layout ||
                                    expected_format.is_none_or(|format| {
                                        identity.physical.format == format
                                    }))
                        })
                        .collect::<Vec<_>>();
                    let owner = match owners.as_slice() {
                        [owner] => *owner,
                        [] => return Err("native claim has no exact resolved owner".into()),
                        _ => {
                            eprintln!(
                                "ambiguous resolved owners claim={layout:?} owners={:?}",
                                owners.iter().map(|owner| owner.key).collect::<Vec<_>>()
                            );
                            return Err("native claim has ambiguous resolved owners".into());
                        }
                    };
                    let stores = plan
                        .stores
                        .iter()
                        .filter(|store| {
                            store.matrix_id == owner.key.matrix_id &&
                                store.instance == owner.key.instance
                        })
                        .collect::<Vec<_>>();
                    match stores.as_slice() {
                        [store] => *store,
                        [] => return Err("native claim has no exact owner store".into()),
                        _ => return Err("native claim has ambiguous owner stores".into()),
                    }
                };
                let matrix = store
                    .wire_type
                    .as_ref()
                    .and_then(ConcreteWireType::matrix_type)
                    .ok_or("native claim store has no matrix type")?;
                let identity = store_identity(store).map_err(|error| error.to_string())?;
                let candidates =
                    self.resource_parameters(matrix)
                        .map_err(|error| error.to_string())?
                        .into_iter()
                        .filter(|params| {
                            (layout.key.device < 0 ||
                                params.device_ids().contains(&layout.key.device)) &&
                                (!matrix_layout ||
                                    expected_level
                                        .is_none_or(|level| level < params.moduli().len())) &&
                                params.context_identity() == identity.physical.context_identity
                        })
                        .collect::<Vec<_>>();
                match candidates.as_slice() {
                    [params] => Ok(params.clone()),
                    [] => Err("native claim has no exact device/context parameters".into()),
                    _ => Err("native claim has ambiguous device/context parameters".into()),
                }
            };
        let traced = |layout: &mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationLayout,
                      store_hint: Option<usize>|
         -> Result<Option<(GpuDCRTPolyParams, GpuTracedClaim)>, String> {
            // Native host-only geometry is metadata, not an inventory slot.
            if layout.kind == 100 {
                return Ok(None);
            }
            let params = params_for_layout(layout, store_hint)?;
            let claim = if layout.kind == 0 {
                GpuTracedClaim::matrix(
                    layout.rows,
                    layout.columns,
                    usize::try_from(layout.level)
                        .map_err(|_| "matrix claim has an invalid level".to_owned())?,
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
                    _ => return Err("native allocation has an unknown claim kind".into()),
                };
                GpuTracedClaim::workspace(GpuPreparedWorkspaceLayout {
                    kind,
                    bytes: layout.bytes,
                    alignment: layout.alignment.max(1),
                })
            };
            Ok(Some((params, claim)))
        };
        let traced_composite = |claim: &mxx_primitives::matrix::gpu_dcrt_poly::GpuTracedClaim,
                                layout: Option<
            &mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationLayout,
        >,
                                store_hint: Option<usize>|
         -> Result<(GpuDCRTPolyParams, GpuTracedClaim), String> {
            let params = if let Some(layout) = layout {
                params_for_layout(layout, store_hint)?
            } else {
                let index = store_hint.ok_or("composite claim has no exact matrix store")?;
                let store = plan.stores.get(index).ok_or("composite claim store is invalid")?;
                let matrix = store
                    .wire_type
                    .as_ref()
                    .and_then(ConcreteWireType::matrix_type)
                    .ok_or("composite claim store has no matrix type")?;
                let identity = store_identity(store).map_err(|error| error.to_string())?;
                let candidates = self
                    .resource_parameters(matrix)
                    .map_err(|error| error.to_string())?
                    .into_iter()
                    .filter(|params| {
                        params.device_ids().contains(&identity.physical.device) &&
                            params.context_identity() == identity.physical.context_identity
                    })
                    .collect::<Vec<_>>();
                match candidates.as_slice() {
                    [params] => params.clone(),
                    _ => return Err("composite claim has missing/ambiguous exact context".into()),
                }
            };
            let traced = if claim.kind() == GpuPreparedSlotKind::Matrix {
                *claim
            } else {
                GpuTracedClaim::workspace(
                    claim.layout().ok_or("composite workspace claim has no layout")?,
                )
            };
            if claim.kind() == GpuPreparedSlotKind::Matrix {
                let (level, evaluation) = if let Some(layout) = layout {
                    (
                        usize::try_from(layout.level)
                            .map_err(|_| "composite matrix layout has invalid level")?,
                        layout.format == GPU_POLY_FORMAT_EVAL,
                    )
                } else {
                    let index = store_hint.ok_or("composite matrix claim has no store")?;
                    let store =
                        plan.stores.get(index).ok_or("composite matrix claim store is invalid")?;
                    let identity = store_identity(store).map_err(|error| error.to_string())?;
                    (
                        identity.physical.level,
                        identity.physical.format ==
                            super::gpu_prepared_lowering::PreparedFormat::Evaluation,
                    )
                };
                if claim.level() != Some(level) || claim.is_evaluation() != Some(evaluation) {
                    return Err("composite matrix claim state differs from exact store".into());
                }
            }
            Ok((params, traced))
        };

        for (buffer_index, buffer) in resources.scalar_buffers.iter().enumerate() {
            let store_hint = plan
                .stores
                .iter()
                .position(|store| {
                    store.matrix_id == buffer.plan.owner.matrix_id &&
                        store.instance == buffer.plan.owner.instance
                })
                .ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "scalar buffer owner has no exact matrix store".into(),
                    )
                })?;
            for (allocation_index, allocation) in buffer.layout.allocations().iter().enumerate() {
                let Some((params, claim)) = traced(allocation, Some(store_hint))
                    .map_err(PolyBackendError::GpuSubmission)?
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
                    ClaimTarget::ScalarBuffer(buffer_index, allocation_index),
                    Some(*allocation),
                ));
            }
        }

        for (command_index, command) in resources.commands.iter_mut().enumerate() {
            for (allocation_index, allocation) in
                command.composite_allocations.iter_mut().enumerate()
            {
                let (params, claim) = traced_composite(
                    &allocation.claim,
                    allocation.layout.as_ref(),
                    allocation.store,
                )
                .map_err(PolyBackendError::GpuSubmission)?;
                claims
                    .entry(params.context_identity())
                    .or_insert_with(|| (params.clone(), Vec::new()))
                    .1
                    .push(claim);
                claim_entries.push((
                    params,
                    claim,
                    ClaimTarget::Composite(command_index, allocation_index),
                    allocation.layout,
                ));
            }
            for (allocation_index, allocation) in command.allocations.iter_mut().enumerate() {
                let Some((params, claim)) = traced(&allocation.layout, allocation.store)
                    .map_err(PolyBackendError::GpuSubmission)?
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
                    ClaimTarget::Command(command_index, allocation_index),
                    Some(allocation.layout),
                ));
            }
            if let Some(replay) = command.replay_upload.as_mut() {
                for (allocation_index, allocation) in replay.allocations.iter_mut().enumerate() {
                    let Some((params, claim)) = traced(&allocation.layout, allocation.store)
                        .map_err(PolyBackendError::GpuSubmission)?
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
                        ClaimTarget::Replay(command_index, allocation_index),
                        Some(allocation.layout),
                    ));
                }
            }
        }
        for (schedule_index, schedule) in resources.schedules.iter_mut().enumerate() {
            for allocation_index in 0..schedule.schedule_allocations_len() {
                let allocation = schedule.allocation_mut(allocation_index).ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "resolved schedule allocation index is invalid".into(),
                    )
                })?;
                let Some((params, claim)) = traced(&allocation.layout, allocation.store)
                    .map_err(PolyBackendError::GpuSubmission)?
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
                    ClaimTarget::Schedule(schedule_index, allocation_index),
                    Some(allocation.layout),
                ));
            }
        }
        // Output codecs are lowered from the graph boundary rather than from
        // a native command recipe.  They are nevertheless fixed warmup
        // resources and must join this same provisioning transaction.  Their
        // leases are consumed by the codec owner, so there is intentionally no
        // command slot to write here.
        let mut codec_slots = (0..extra_claims.len())
            .map(|_| Vec::<super::gpu_prepared_lowering::PreparedSlotRef>::new())
            .collect::<Vec<_>>();
        for (codec, (params, extra, extra_layouts)) in extra_claims.iter().enumerate() {
            if extra.len() != extra_layouts.len() {
                return Err(PolyBackendError::GpuSubmission(
                    "prepared extra claim/layout provenance length mismatch".into(),
                ));
            }
            let entry = &mut claims
                .entry(params.context_identity())
                .or_insert_with(|| (params.clone(), Vec::new()))
                .1;
            for (claim_index, (&claim, &placement)) in extra.iter().zip(extra_layouts).enumerate() {
                entry.push(claim);
                claim_entries.push((
                    params.clone(),
                    claim,
                    ClaimTarget::Unbound { codec, claim: claim_index, placement },
                    None,
                ));
            }
        }
        // Preserve the exact owner descriptor beside every ordered claim.  A
        // claim's shape/level is not enough to identify its owner: instances,
        // devices, contexts, and interleaved stream slots can all coincide.
        // Provisioning therefore receives this positional table and rejects a
        // matrix claim that cannot be tied to one resolved owner.
        // Zero-sized non-resource claims are structural placeholders, not
        // native workspace slots.  Drop them before provisioning; native
        // workspace storage accepts zero bytes only for completion events and
        // submission streams.  Their command allocations intentionally remain
        // unbound and therefore cannot be acquired at execution.
        claim_entries.retain(|(_, claim, _, _)| {
            claim.kind() == GpuPreparedSlotKind::Matrix ||
                claim.layout().is_some_and(|layout| {
                    layout.bytes > 0 ||
                        matches!(
                            layout.kind,
                            GpuPreparedSlotKind::CompletionEvent |
                                GpuPreparedSlotKind::SubmissionStream
                        )
                })
        });
        let exact_layout_for_target = |claim: &GpuTracedClaim,
                                       target: &ClaimTarget|
         -> Result<Option<PreparedOwnerLayout>, String> {
            if claim.kind() != GpuPreparedSlotKind::Matrix {
                return Ok(None);
            }
            let layout = match target {
                ClaimTarget::Owner(index) => {
                    resources
                        .owners
                        .get(*index)
                        .ok_or("resolved owner claim index is invalid")?
                        .layout
                }
                ClaimTarget::OwnerInputCopy(_, _) => {
                    unreachable!("input-copy completion claims have no matrix owner layout")
                }
                ClaimTarget::Command(command, allocation) => {
                    let allocation = resources.commands[*command]
                        .allocations
                        .get(*allocation)
                        .ok_or("resolved command claim index is invalid")?;
                    allocation
                        .owner_layout
                        .ok_or("resolved command matrix claim has no exact owner layout")?
                }
                ClaimTarget::Composite(command, allocation) => resources.commands[*command]
                    .composite_allocations
                    .get(*allocation)
                    .ok_or("resolved composite claim index is invalid")?
                    .owner_layout
                    .ok_or("resolved composite matrix claim has no exact owner layout")?,
                ClaimTarget::Replay(command, allocation) => resources.commands[*command]
                    .replay_upload
                    .as_ref()
                    .ok_or("resolved replay claim has no upload")?
                    .allocations
                    .get(*allocation)
                    .ok_or("resolved replay claim index is invalid")?
                    .owner_layout
                    .ok_or("resolved replay matrix claim has no exact owner layout")?,
                ClaimTarget::Schedule(schedule, allocation) => resources.schedules[*schedule]
                    .allocation(*allocation)
                    .ok_or("resolved schedule claim index is invalid")?
                    .owner_layout
                    .ok_or("resolved schedule matrix claim has no exact owner layout")?,
                ClaimTarget::ScalarBuffer(buffer, allocation) => {
                    let _ = resources.scalar_buffers[*buffer]
                        .layout
                        .allocations()
                        .get(*allocation)
                        .ok_or("resolved scalar-buffer claim index is invalid")?;
                    resources
                        .owners
                        .iter()
                        .find(|owner| owner.key == resources.scalar_buffers[*buffer].plan.owner)
                        .map(|owner| owner.layout)
                        .ok_or("resolved scalar-buffer owner is unresolved")?
                }
                ClaimTarget::Unbound { .. } => return Ok(None),
            };
            Ok(Some(layout))
        };
        let workspace_claim_for = |params: &GpuDCRTPolyParams,
                                   claim: &GpuTracedClaim,
                                   target: &ClaimTarget,
                                   allocation: Option<PreparedAllocationLayout>|
         -> Result<Option<GpuPreparedWorkspaceClaim>, String> {
            if claim.kind() == GpuPreparedSlotKind::Matrix {
                return Ok(None);
            }
            if let ClaimTarget::OwnerInputCopy(owner_index, ordinal) = target {
                let owner = resources
                    .owners
                    .get(*owner_index)
                    .ok_or("workspace input-copy owner index is invalid")?;
                let identity = resources
                    .finalized_matrices
                    .identity(owner.key.matrix_id)
                    .ok_or("workspace input-copy owner identity is missing")?;
                let mut remaining = *ordinal;
                let mut placement = None;
                'partitions: for partition in 0..owner.layout.partition_count() {
                    for limb in 0..=identity.physical.level {
                        if let Some(stream_slot) = owner.layout.stream_slot(partition, limb) {
                            if remaining == 0 {
                                placement = Some((partition, stream_slot));
                                break 'partitions;
                            }
                            remaining -= 1;
                        }
                    }
                }
                let (partition, stream_slot) =
                    placement.ok_or("workspace input-copy owner has no exact event stream slot")?;
                return Ok(Some(GpuPreparedWorkspaceClaim::new(
                    claim.layout().ok_or("workspace input-copy claim has no layout")?,
                    identity.physical.device,
                    partition,
                    stream_slot,
                )));
            }
            if let ClaimTarget::Unbound { placement, .. } = target {
                if placement.layout !=
                    claim.layout().ok_or("unbound workspace claim has no layout")?
                {
                    return Err("unbound workspace placement differs from its claim".into());
                }
                return Ok(Some(*placement));
            }
            if let ClaimTarget::Composite(command, index) = target {
                let composite = resources.commands[*command]
                    .composite_allocations
                    .get(*index)
                    .ok_or("workspace composite index is invalid")?;
                let owner_layout =
                    composite.owner_layout.ok_or("workspace composite owner layout is missing")?;
                let store_index = composite
                    .store
                    .ok_or("workspace composite finalized owner store is missing")?;
                let store = plan
                    .stores
                    .get(store_index)
                    .ok_or("workspace composite finalized owner store is invalid")?;
                let identity = resources
                    .finalized_matrices
                    .identity(store.matrix_id)
                    .ok_or("workspace composite finalized owner identity is missing")?;
                if identity.physical.context_identity != params.context_identity() {
                    return Err(
                        "workspace composite finalized owner context differs from claim".into()
                    );
                }
                let partition = params
                    .device_ids()
                    .iter()
                    .position(|device| *device == identity.physical.device)
                    .ok_or("workspace composite owner device is not in its exact context")?;
                // Composite phase workspaces have no native allocation key.
                // Their CUDA binder uses the phase owner's limb-zero stream;
                // derive that exact placement from the finalized owner and
                // physical device partition, never from a phase ordinal.
                let stream_slot = owner_layout
                    .stream_slot(partition, 0)
                    .ok_or("workspace composite owner has no exact phase stream slot")?;
                return Ok(Some(GpuPreparedWorkspaceClaim::new(
                    claim.layout().ok_or("workspace composite claim has no layout")?,
                    identity.physical.device,
                    partition,
                    stream_slot,
                )));
            }
            let allocation = allocation.ok_or_else(|| {
                format!(
                    "workspace claim has no native allocation (target {target:?}, kind {:?})",
                    claim.kind()
                )
            })?;
            let owner_layout = match target {
                ClaimTarget::Owner(index) => {
                    resources.owners.get(*index).ok_or("workspace owner index is invalid")?.layout
                }
                ClaimTarget::Command(command, index) => resources.commands[*command]
                    .allocations
                    .get(*index)
                    .ok_or("workspace command index is invalid")?
                    .owner_layout
                    .ok_or_else(|| {
                        format!(
                            "workspace command owner layout is missing for command {command} allocation {index} (kind {}, key {:?})",
                            allocation.kind,
                            allocation.key
                        )
                    })?,
                ClaimTarget::Composite(command, index) => resources.commands[*command]
                    .composite_allocations
                    .get(*index)
                    .ok_or("workspace composite index is invalid")?
                    .owner_layout
                    .ok_or("workspace composite owner layout is missing")?,
                ClaimTarget::Replay(command, index) => resources.commands[*command]
                    .replay_upload
                    .as_ref()
                    .ok_or("workspace replay upload is missing")?
                    .allocations
                    .get(*index)
                    .ok_or("workspace replay index is invalid")?
                    .owner_layout
                    .ok_or("workspace replay owner layout is missing")?,
                ClaimTarget::Schedule(schedule, index) => resources.schedules[*schedule]
                    .allocation(*index)
                    .ok_or("workspace schedule index is invalid")?
                    .owner_layout
                    .ok_or("workspace schedule owner layout is missing")?,
                ClaimTarget::ScalarBuffer(buffer, index) => {
                    let _ = resources.scalar_buffers[*buffer]
                        .layout
                        .allocations()
                        .get(*index)
                        .ok_or("workspace scalar allocation is invalid")?;
                    resources
                        .owners
                        .iter()
                        .find(|owner| owner.key == resources.scalar_buffers[*buffer].plan.owner)
                        .map(|owner| owner.layout)
                        .ok_or("workspace scalar allocation owner is unresolved")?
                }
                ClaimTarget::Unbound { .. } => return Ok(None),
                ClaimTarget::OwnerInputCopy(_, _) => unreachable!("handled above"),
            };
            let (device, partition, stream_slot) = if allocation.key.device >= 0 &&
                allocation.key.partition >= 0
            {
                let partition = usize::try_from(allocation.key.partition)
                    .map_err(|_| "workspace partition is invalid")?;
                let stream_slot = owner_layout
                    .stream_slot(partition, allocation.key.limb_y as usize)
                    .ok_or("workspace claim has no exact compute stream slot")?;
                (allocation.key.device, partition, stream_slot)
            } else if claim.kind() == GpuPreparedSlotKind::PinnedHost {
                let (device, partition, stream_slot) = resources
                    .owners
                    .iter()
                    .find_map(|owner| {
                        if owner.layout != owner_layout {
                            return None;
                        }
                        let identity =
                            resources.finalized_matrices.identity(owner.key.matrix_id)?;
                        let partition = 0;
                        let stream_slot = owner_layout.stream_slot(partition, 0)?;
                        Some((identity.physical.device, partition, stream_slot))
                    })
                    .ok_or("pinned host claim has no exact owner placement")?;
                (device, partition, stream_slot)
            } else {
                return Err(format!(
                    "workspace claim has no physical placement (target {target:?}, kind {}, key {:?})",
                    allocation.kind, allocation.key
                ));
            };
            Ok(Some(GpuPreparedWorkspaceClaim::new(
                claim.layout().ok_or("workspace claim has no layout")?,
                device,
                partition,
                stream_slot,
            )))
        };
        let mut exact_claims = BTreeMap::<
            usize,
            (
                GpuDCRTPolyParams,
                Vec<GpuTracedClaim>,
                Vec<Option<PreparedOwnerLayout>>,
                Vec<Option<GpuPreparedWorkspaceClaim>>,
            ),
        >::new();
        for (params, claim, target, allocation_layout) in &claim_entries {
            let exact =
                exact_layout_for_target(claim, target).map_err(PolyBackendError::GpuSubmission)?;
            let workspace = workspace_claim_for(params, claim, target, *allocation_layout)
                .map_err(PolyBackendError::GpuSubmission)?;
            let entry = exact_claims
                .entry(params.context_identity())
                .or_insert_with(|| (params.clone(), Vec::new(), Vec::new(), Vec::new()));
            entry.1.push(*claim);
            entry.2.push(exact);
            entry.3.push(workspace);
        }
        let resolved_workspaces = claim_entries
            .iter()
            .map(|(params, claim, target, allocation_layout)| {
                workspace_claim_for(params, claim, target, *allocation_layout)
                    .map_err(PolyBackendError::GpuSubmission)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let provisioned = self.provision_resolved_storage(exact_claims.into_values().collect())?;
        let direct_storage = provisioned
            .into_iter()
            .map(|(_, params, claims, workspace_claims, storage)| {
                (
                    (params.context_identity(), !workspace_claims.is_empty()),
                    (claims, workspace_claims, storage),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let inventory = direct_storage
            .values()
            .map(|(_, _, storage)| {
                let physical_device = if storage.is_workspace_only() {
                    storage.workspace_slot_identity(0).ok()?.device
                } else {
                    storage.device()
                };
                let device =
                    self.devices.iter().position(|(physical, _)| *physical == physical_device)?;
                Some((storage.identity(), (device, Arc::clone(storage))))
            })
            .collect::<Option<BTreeMap<_, _>>>()
            .ok_or(PolyBackendError::UnsupportedPlacement)?;
        let mut workspace_next = BTreeMap::<usize, usize>::new();
        let mut matrix_next = BTreeMap::<usize, usize>::new();
        let mut requests = BTreeMap::<u64, Vec<GpuPreparedRequest>>::new();
        let mut workspace_requests = BTreeMap::<u64, Vec<GpuPreparedWorkspaceRequest>>::new();
        for ((params, claim, target, _allocation_layout), workspace) in
            claim_entries.into_iter().zip(resolved_workspaces)
        {
            if claim.kind() != GpuPreparedSlotKind::Matrix {
                let expected = workspace.ok_or_else(|| {
                    PolyBackendError::GpuSubmission("workspace claim has no placement".into())
                })?;
                let (_claims, workspace_claims, storage) =
                    direct_storage.get(&(params.context_identity(), true)).ok_or_else(|| {
                        PolyBackendError::GpuSubmission(
                            "resolved workspace has no direct storage".into(),
                        )
                    })?;
                let slot = workspace_next.entry(params.context_identity()).or_insert(0);
                let expected_claim = workspace_claims.get(*slot).ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "resolved workspace claim order is invalid".into(),
                    )
                })?;
                if expected_claim != &expected {
                    return Err(PolyBackendError::GpuSubmission(
                        "resolved workspace claim differs from direct storage".into(),
                    ));
                }
                let slot_identity = storage
                    .workspace_slot_identity(*slot)
                    .map_err(PolyBackendError::GpuSubmission)?;
                let identity = storage.identity();
                *slot += 1;
                let request = GpuPreparedWorkspaceRequest {
                    storage_id: storage.identity(),
                    slot: slot_identity.slot_index,
                    device: slot_identity.device,
                    partition: slot_identity.partition,
                    stream_slot: slot_identity.stream_slot,
                    bytes: slot_identity.bytes,
                    alignment: slot_identity.alignment,
                    kind: slot_identity.kind,
                };
                workspace_requests.entry(identity).or_default().push(request);
                let slot = super::gpu_prepared_lowering::PreparedSlotRef {
                    storage: Arc::clone(storage),
                    device: slot_identity.device,
                    request: super::gpu_prepared_lowering::PreparedSlotRequest::Workspace(request),
                };
                match target {
                    ClaimTarget::Owner(owner) => resources.owners[owner].slot = Some(slot),
                    ClaimTarget::Command(command, allocation) => {
                        resources.commands[command].allocations[allocation].slot = Some(slot)
                    }
                    ClaimTarget::Composite(command, allocation) => {
                        resources.commands[command].composite_allocations[allocation].slot =
                            Some(slot)
                    }
                    ClaimTarget::Replay(command, allocation) => {
                        resources.commands[command]
                            .replay_upload
                            .as_mut()
                            .ok_or_else(|| {
                                PolyBackendError::GpuSubmission(
                                    "workspace replay upload is missing".into(),
                                )
                            })?
                            .allocations[allocation]
                            .slot = Some(slot);
                    }
                    ClaimTarget::Schedule(schedule, allocation) => {
                        resources.schedules[schedule]
                            .allocation_mut(allocation)
                            .ok_or_else(|| {
                                PolyBackendError::GpuSubmission(
                                    "resolved schedule allocation index is invalid".into(),
                                )
                            })?
                            .slot = Some(slot)
                    }
                    ClaimTarget::ScalarBuffer(buffer, allocation) => {
                        resources.scalar_buffers[buffer].slots[allocation] = Some(slot)
                    }
                    ClaimTarget::OwnerInputCopy(owner, ordinal) => {
                        let slots = &mut resources.owners[owner].input_copy_slots;
                        append_input_copy_slot(slots, ordinal, slot)
                            .map_err(|error| PolyBackendError::GpuSubmission(error.into()))?;
                    }
                    ClaimTarget::Unbound { codec, claim, .. } => {
                        codec_slots[codec].push(slot);
                        debug_assert_eq!(claim, codec_slots[codec].len() - 1);
                    }
                }
                continue;
            }
            let (_claims, _workspace_claims, storage) =
                direct_storage.get(&(params.context_identity(), false)).ok_or_else(|| {
                    PolyBackendError::GpuSubmission(format!(
                        "resolved matrix has no direct storage (target {target:?}, context {}, claim {:?})",
                        params.context_identity(),
                        claim
                    ))
                })?;
            let claim_index = matrix_next.entry(params.context_identity()).or_insert(0);
            let slot_identity = storage.slot_identity(*claim_index).ok_or_else(|| {
                PolyBackendError::GpuSubmission("resolved matrix slot disappeared".into())
            })?;
            let request = slot_identity.matrix_request(
                claim.rows(),
                claim.columns(),
                claim.is_evaluation().unwrap_or(false),
            );
            let identity = storage.identity();
            *claim_index += 1;
            requests.entry(identity).or_default().push(request);
            let slot = Some(super::gpu_prepared_lowering::PreparedSlotRef {
                storage: Arc::clone(storage),
                // `device` is the ledger/fleet ordinal.  Slot bindings carry
                // the physical CUDA identity used by the native region.
                device: inventory.get(&identity).map(|(_, storage)| storage.device()).ok_or_else(
                    || {
                        PolyBackendError::GpuSubmission(
                            "resolved resource storage disappeared".into(),
                        )
                    },
                )?,
                request: super::gpu_prepared_lowering::PreparedSlotRequest::Matrix(request),
            });
            match target {
                ClaimTarget::Owner(owner) => {
                    resources.owners[owner].slot = slot;
                }
                ClaimTarget::OwnerInputCopy(_, _) => {
                    return Err(PolyBackendError::GpuSubmission(
                        "input-copy completion claim must use a workspace resource".into(),
                    ));
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
                    resources.schedules[schedule]
                        .allocation_mut(allocation)
                        .ok_or_else(|| {
                            PolyBackendError::GpuSubmission(
                                "resolved schedule allocation index is invalid".into(),
                            )
                        })?
                        .slot = slot;
                }
                ClaimTarget::ScalarBuffer(buffer, allocation) => {
                    resources.scalar_buffers[buffer].slots[allocation] = slot;
                }
                ClaimTarget::Unbound { codec, claim, .. } => {
                    codec_slots[codec].push(slot.ok_or_else(|| {
                        PolyBackendError::GpuSubmission(
                            "unbound matrix codec slot is unresolved".into(),
                        )
                    })?);
                    debug_assert_eq!(claim, codec_slots[codec].len() - 1);
                }
            }
        }
        let ledger = self.prepared_ledger.as_mut().ok_or_else(|| {
            PolyBackendError::GpuSubmission("resolved resource ledger is missing".into())
        })?;
        let requirements = inventory
            .iter()
            .filter(|(identity, _)| {
                requests.contains_key(identity) || workspace_requests.contains_key(identity)
            })
            .map(|(identity, (device, storage))| {
                Ok(crate::gpu_memory::GpuPreparedAllocationRequirement {
                    device: *device,
                    storage,
                    requests: requests.get(identity).map_or(&[][..], Vec::as_slice),
                    workspace_requests: workspace_requests
                        .get(identity)
                        .map_or(&[][..], Vec::as_slice),
                })
            })
            .collect::<Result<Vec<_>, PolyBackendError>>()?;
        let region = ledger
            .reserve_region(&requirements)
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
        let storages = inventory
            .iter()
            .filter(|(identity, _)| {
                requests.contains_key(identity) || workspace_requests.contains_key(identity)
            })
            .map(|(identity, (_, storage))| (*identity, Arc::clone(storage)))
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
                stream.slot = Some(matches[0].slot.clone().ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "composite stream allocation slot is unresolved".into(),
                    )
                })?);
            }
            for stream in &mut command.streams {
                if stream.layout.origin == 0 {
                    stream.slot = None;
                } else {
                    stream.slot = Some(exact_prepared_stream_slot(
                        &command.allocations,
                        &stream.layout.key,
                        9,
                        "command",
                    )?);
                }
            }
            if let Some(replay) = command.replay_upload.as_mut() {
                for stream in &mut replay.streams {
                    if stream.layout.origin == 0 {
                        stream.slot = None;
                    } else {
                        stream.slot = Some(exact_prepared_stream_slot(
                            &replay.allocations,
                            &stream.layout.key,
                            9,
                            "replay",
                        )?);
                    }
                }
            }
        }
        for schedule in &mut resources.schedules {
            let stream_keys =
                schedule.stream_claims().map(|stream| stream.layout.key).collect::<Vec<_>>();
            for (stream_index, stream_key) in stream_keys.into_iter().enumerate() {
                // The merged schedule owns one completion-event allocation
                // for every emitted stream.  Resolve it by the full native
                // provenance key; ordinal matching is invalid when member
                // descriptors are interleaved.
                let matches = schedule
                    .allocation_claims()
                    .filter(|allocation| {
                        allocation.layout.kind == 8 && allocation.layout.key == stream_key
                    })
                    .collect::<Vec<_>>();
                let allocation = match matches.as_slice() {
                    [allocation] => *allocation,
                    [] => {
                        return Err(PolyBackendError::GpuSubmission(
                            "schedule completion allocation is missing".into(),
                        ))
                    }
                    _ => {
                        return Err(PolyBackendError::GpuSubmission(
                            "schedule completion allocation is ambiguous".into(),
                        ))
                    }
                };
                let slot = allocation.slot.clone().ok_or_else(|| {
                    PolyBackendError::GpuSubmission(
                        "schedule completion allocation slot is unresolved".into(),
                    )
                })?;
                schedule
                    .stream_claim_mut(stream_index)
                    .ok_or_else(|| {
                        PolyBackendError::GpuSubmission("schedule stream index is invalid".into())
                    })?
                    .slot = Some(slot);
            }
        }
        let codec_slots = codec_slots.into_iter().map(Vec::into_boxed_slice).collect();
        Ok((region, storages, codec_slots))
    }

    /// Provision completion events from an already detached prepared region.
    /// Matrix, compact-payload, and event claims are acquired by the caller's
    /// single warmup transaction before this method is entered.
    pub(crate) fn provision_prepared_schedules_in_region(
        &mut self,
        schedules: &mut [Option<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSchedule>],
        streams: &[Box<[PreparedScheduleStreamKey]>],
        completion_slots: &[Box<[super::gpu_prepared_lowering::PreparedSlotRef]>],
        region: &Arc<crate::gpu_memory::GpuMemoryRegion>,
    ) -> Result<(), PolyBackendError> {
        if schedules.len() != streams.len() || schedules.len() != completion_slots.len() {
            return Err(PolyBackendError::GpuSubmission(
                "prepared schedule resource table length mismatch".into(),
            ));
        }
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
            let owners = schedule.stream_parameters().map_err(PolyBackendError::GpuSubmission)?;
            for (key, owner) in schedule_streams.iter().zip(&owners) {
                if !owner.device_ids().contains(&key.device) {
                    return Err(PolyBackendError::UnsupportedPlacement);
                }
            }
        }
        for (schedule, completion_slots) in schedules.iter_mut().zip(completion_slots) {
            let schedule = schedule.as_mut().ok_or_else(|| {
                PolyBackendError::GpuSubmission("missing prepared schedule".into())
            })?;
            super::super::gpu_prepared::bind_prepared_slots(region, completion_slots, || {
                schedule.provision()
            })
            .map_err(PolyBackendError::GpuSubmission)?;
        }
        Ok(())
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
        artifact_descriptors: &[crate::executor::PreparedArtifactDescriptor],
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
        // Runtime matrix levels/formats and concrete output CRT contexts are
        // unavailable to structural lowering. Resolve them before creating
        // stores or native owner descriptors so generic replay and the pure
        // plan share one stable owner identity table.
        // Preserve the exact positional order established by lowering. The
        // validated graph execution order is a topology traversal and is not
        // the canonical runtime-input order for family leaves; zipping that
        // traversal with `runtime_input_wires` can assign a valid matrix's
        // context to the wrong wire and leave later owners without finalized
        // device state.
        let prepared_runtime_inputs = program
            .input_names
            .iter()
            .map(|(name, _)| {
                let input = inputs.get(name).ok_or_else(|| {
                    PolyBackendError::GpuSubmission(format!("prepared input {name:?} is missing"))
                })?;
                super::prepared_runtime_value_for_warmup(self, input)
                    .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let expanded_runtime_inputs =
            gpu_prepared::expand_prepared_runtime_inputs(&program, &prepared_runtime_inputs)
                .map_err(PolyBackendError::GpuSubmission)?;
        program.scalar_projections = gpu_prepared::gpu_prepared_scalar::scalar_capacities(
            &program,
            &expanded_runtime_inputs,
        )
        .map_err(|error| {
            PolyBackendError::GpuSubmission(format!(
                "prepared scalar warmup projection failed: {error}"
            ))
        })?;
        finalize_prepared_matrix_contexts(self, &mut program, &prepared_runtime_inputs)?;
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
        let host_input_wires = program
            .runtime_input_wires
            .iter()
            .copied()
            .zip(expanded_runtime_inputs.iter())
            .filter_map(|(wire, value)| {
                matches!(value, PreparedRuntimeValue::HostMatrix { .. }).then_some(wire)
            })
            .collect::<BTreeSet<_>>();
        program.resource_plan.mark_host_input_uploads(&host_input_wires).map_err(|error| {
            PolyBackendError::GpuSubmission(format!("prepared host inputs: {error:?}"))
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
        let codec_claims = self.prepared_output_codec_claims(
            &program,
            &program.resource_plan,
            &resolved_resources,
            &physical_devices,
            program.instance_count,
        )?;
        let has_resolved_claims = !program.resource_plan.stores.is_empty() ||
            resolved_resources.commands.iter().any(|command| {
                !command.allocations.is_empty() || !command.composite_allocations.is_empty()
            }) ||
            resolved_resources.commands.iter().any(|command| {
                command.replay_upload.as_ref().is_some_and(|replay| !replay.allocations.is_empty())
            }) ||
            resolved_resources
                .schedules
                .iter()
                .any(|schedule| schedule.allocation_claims().next().is_some()) ||
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
        let execution = crate::backend::poly_gpu::from_lowered_program(
            self,
            &prepared_runtime_inputs,
            &mut program,
            &resolved_resources,
            reservation,
            artifact_descriptors,
        )
        .map_err(PolyBackendError::GpuSubmission)?;
        self.prepared_graph = Some(execution);
        self.fence_released_memory()?;
        Ok(())
    }

    fn provision_resolved_storage(
        &mut self,
        demands: Vec<(
            GpuDCRTPolyParams,
            Vec<GpuTracedClaim>,
            Vec<Option<PreparedOwnerLayout>>,
            Vec<Option<GpuPreparedWorkspaceClaim>>,
        )>,
    ) -> Result<
        Vec<(
            usize,
            GpuDCRTPolyParams,
            Vec<GpuTracedClaim>,
            Vec<GpuPreparedWorkspaceClaim>,
            Arc<GpuPreparedStorage>,
        )>,
        PolyBackendError,
    > {
        crate::backend::poly_gpu::record_prepared_forbidden(7);
        struct PendingStorage {
            device: usize,
            params: GpuDCRTPolyParams,
            matrices: Vec<GpuTracedClaim>,
            owner_layouts: Vec<PreparedOwnerLayout>,
            layouts: Vec<GpuPreparedWorkspaceLayout>,
            workspace_claims: Vec<GpuPreparedWorkspaceClaim>,
            claims: Vec<GpuTracedClaim>,
        }
        let mut pending = Vec::new();
        for (params, claims, claim_layouts, allocation_layouts) in demands {
            if claims.len() != claim_layouts.len() || claims.len() != allocation_layouts.len() {
                return Err(PolyBackendError::GpuSubmission(
                    "prepared claim/layout provenance length mismatch".into(),
                ));
            }
            let device = self
                .devices
                .iter()
                .position(|(physical, _)| params.device_ids().contains(physical))
                .ok_or(PolyBackendError::UnsupportedPlacement)?;
            let missing = claims
                .iter()
                .copied()
                .zip(claim_layouts.iter().copied())
                .zip(allocation_layouts.iter().copied())
                .map(|((claim, owner_layout), workspace_claim)| {
                    (claim, owner_layout, workspace_claim)
                })
                .collect::<Vec<_>>();
            let mut matrices = Vec::new();
            let mut matrix_claims = Vec::new();
            let mut owner_layouts = Vec::new();
            let mut layouts = Vec::new();
            let mut workspace_claims = Vec::new();
            let mut workspace_traced_claims = Vec::new();
            for (claim, owner_layout, workspace_claim) in missing {
                if claim.kind() == GpuPreparedSlotKind::Matrix {
                    let owner_layout = owner_layout.ok_or_else(|| {
                        PolyBackendError::GpuSubmission(
                            "new prepared matrix claim has no exact owner layout".into(),
                        )
                    })?;
                    matrices.push(claim);
                    matrix_claims.push(claim);
                    owner_layouts.push(owner_layout);
                } else {
                    let layout = claim.layout().ok_or(PolyBackendError::InvalidInteger)?;
                    let mut workspace_claim = workspace_claim.ok_or_else(|| {
                        PolyBackendError::GpuSubmission(
                            "workspace claim has no exact native allocation placement".into(),
                        )
                    })?;
                    let physical_device =
                        *params.device_ids().get(workspace_claim.partition).ok_or_else(|| {
                            PolyBackendError::GpuSubmission(
                                "workspace claim partition is outside the exact device context"
                                    .into(),
                            )
                        })?;
                    // Resource allocation keys carry a partition-local device
                    // ordinal for some event/stream claims. The finalized
                    // parameter context is authoritative for the physical
                    // device at that partition; preserve the exact partition
                    // and stream slot while normalizing that physical field.
                    workspace_claim.device = physical_device;
                    layouts.push(layout);
                    workspace_claims.push(workspace_claim);
                    workspace_traced_claims.push(claim);
                }
            }
            let workspace_device = if let Some(first) = workspace_claims.first() {
                if workspace_claims.iter().any(|claim| claim.device != first.device) {
                    return Err(PolyBackendError::GpuSubmission(
                        "workspace-only claims must have one exact physical device".into(),
                    ));
                }
                self.devices
                    .iter()
                    .position(|(physical, _)| *physical == first.device)
                    .ok_or(PolyBackendError::UnsupportedPlacement)?
            } else {
                device
            };
            if matrices.len() != owner_layouts.len() {
                return Err(PolyBackendError::GpuSubmission(
                    "prepared matrix/layout order became inconsistent".into(),
                ));
            }
            if !matrices.is_empty() {
                pending.push(PendingStorage {
                    device,
                    params: params.clone(),
                    matrices,
                    owner_layouts,
                    layouts: Vec::new(),
                    workspace_claims: Vec::new(),
                    claims: matrix_claims,
                });
            }
            if !layouts.is_empty() {
                pending.push(PendingStorage {
                    device: workspace_device,
                    params,
                    matrices: Vec::new(),
                    owner_layouts: Vec::new(),
                    layouts,
                    workspace_claims,
                    claims: workspace_traced_claims,
                });
            }
        }
        if pending.is_empty() {
            return Ok(Vec::new());
        }

        // First-generation backing uses the same charge journal as later
        // replacements. Establish a provisional ledger before native backing
        // is constructed so every owner is covered by PendingProvisioning.
        let initial_generation = self.prepared_ledger.is_none();
        if initial_generation && self.prepared_ledger.is_none() {
            self.device_parameters().par_iter().try_for_each(|params| {
                params.begin_prepared_setup().map_err(PolyBackendError::GpuSubmission)
            })?;
            use mxx_primitives::poly::dcrt::gpu::GpuAllocationEpochBoundary;
            let parameters = self.device_parameters();
            let epochs = parameters
                .par_iter()
                .map(|params| {
                    params
                        .observe_allocation_epoch(
                            params.device_ids()[0],
                            GpuAllocationEpochBoundary::InitialSetup,
                            true,
                        )
                        .map_err(PolyBackendError::GpuSubmission)
                        .and_then(|observation| match observation {
                            mxx_primitives::poly::dcrt::gpu::GpuAllocationEpochObservation::Verified(
                                epoch,
                            ) => Ok(epoch),
                            mxx_primitives::poly::dcrt::gpu::GpuAllocationEpochObservation::Unverified(
                                reason,
                            ) => Err(PolyBackendError::GpuSubmission(format!(
                                "prepared setup observation was not verified: {reason:?}"
                            ))),
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let ledger = GpuMemoryLedger::new(epochs, self.vram_percent, Vec::new())
                .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
            self.set_memory_ledger(ledger)?;
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
                .chain(
                    request
                        .layouts
                        .iter()
                        .copied()
                        .filter(|layout| {
                            layout.bytes > 0 ||
                                matches!(
                                    layout.kind,
                                    GpuPreparedSlotKind::CompletionEvent |
                                        GpuPreparedSlotKind::SubmissionStream
                                )
                        })
                        .map(GpuTracedClaim::workspace),
                )
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
        let mut charge = self
            .prepared_ledger
            .as_mut()
            .expect("existing prepared inventory has a ledger")
            .begin_prepared_provisioning(&additional, &additional_pinned)
            .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;

        let mut provisioned = Vec::with_capacity(pending.len());
        let construction = (|| {
            for request in pending {
                // Production backing is allocated, not zeroed: these slots are
                // full-overwrite destinations claimed by dispatch, so a
                // placeholder zero kernel would be disposable work. The shapes
                // are appended uninitialized inside the primitives storage
                // constructor and never exist as a readable matrix here.
                let layouts = request
                    .layouts
                    .iter()
                    .copied()
                    .filter(|layout| {
                        layout.bytes > 0 ||
                            matches!(
                                layout.kind,
                                GpuPreparedSlotKind::CompletionEvent |
                                    GpuPreparedSlotKind::SubmissionStream
                            )
                    })
                    .collect::<Vec<_>>();
                let provisioning_claims = request
                    .matrices
                    .iter()
                    .copied()
                    .chain(layouts.iter().copied().map(GpuTracedClaim::workspace))
                    .collect::<Vec<_>>();
                let provisioning =
                    GpuPreparedProvisioningPermit::begin(&request.params, &provisioning_claims)
                        .map_err(PolyBackendError::GpuSubmission)?
                        .enter()
                        .map_err(PolyBackendError::GpuSubmission)?;
                let storage = match if request.matrices.is_empty() {
                    GpuPreparedStorage::from_recipe(PreparedStorageRecipe::WorkspaceOnly {
                        params: &request.params,
                        claims: &request.workspace_claims,
                    })
                } else {
                    GpuPreparedStorage::from_recipe(PreparedStorageRecipe::Matrix {
                        params: Some(&request.params),
                        backing: Vec::new(),
                        uninitialized_matrices: Some(&request.matrices),
                        workspaces: Some(&layouts),
                        owner_layouts: Some(&request.owner_layouts),
                    })
                } {
                    Ok(storage) => storage,
                    Err(error) => {
                        // Native construction may have acquired backing
                        // before failing to return its owner. Keep the charge
                        // conservative rather than refunding an unobservable
                        // asynchronous free.
                        charge.mark_uncertain();
                        return Err(PolyBackendError::GpuSubmission(format!(
                            "{error}; storage device {}, workspace kinds {:?}",
                            request.device,
                            request
                                .workspace_claims
                                .iter()
                                .map(|claim| {
                                    (
                                        claim.layout.kind,
                                        claim.layout.bytes,
                                        claim.layout.alignment,
                                        claim.device,
                                        claim.partition,
                                        claim.stream_slot,
                                    )
                                })
                                .collect::<Vec<_>>()
                        )));
                    }
                };
                let storage = Arc::new(storage);
                provisioned.push((
                    request.device,
                    request.params.clone(),
                    request.claims.clone(),
                    request.workspace_claims.clone(),
                    Arc::clone(&storage),
                ));
                // Transfer each native owner to the charge journal before the
                // next fallible construction. If a later device fails, the
                // transaction can queue asynchronous release for every owner
                // already allocated instead of refunding its bytes early.
                charge
                    .append(vec![(request.device, Arc::clone(&storage))])
                    .map_err(|error| PolyBackendError::GpuSubmission(error.to_string()))?;
                if initial_generation && !storage.is_workspace_only() {
                    GpuPreparedStorage::finish_setup(&[&storage])
                        .map_err(PolyBackendError::GpuSubmission)?;
                }
                provisioning.finish().map_err(PolyBackendError::GpuSubmission)?;
            }
            Ok::<(), PolyBackendError>(())
        })();
        construction?;
        charge.commit();
        Ok(provisioned)
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

fn validate_borrowed_codec_descriptor(
    rows: usize,
    columns: usize,
    allocations: &[PreparedAllocationLayout],
    streams: &[mxx_primitives::matrix::gpu_dcrt_poly::PreparedStreamFootprint],
    owner_identity: u64,
    context_identity: usize,
    physical_device: i32,
) -> Result<
    Option<(
        PreparedAllocationLayout,
        PreparedAllocationLayout,
        mxx_primitives::matrix::gpu_dcrt_poly::PreparedStreamFootprint,
    )>,
    String,
> {
    if rows == 0 || columns == 0 {
        return if allocations.is_empty() && streams.is_empty() {
            Ok(None)
        } else {
            Err("empty borrowed compact store unexpectedly has resources".into())
        };
    }
    let [stream_allocation, transfer_allocation] = allocations else {
        return Err("borrowed compact store must expose stream then transfer allocations".into());
    };
    let [stream] = streams else {
        return Err("borrowed compact store must expose one stream footprint".into());
    };
    if stream.origin !=
        mxx_primitives::matrix::gpu_dcrt_poly::PreparedStreamOrigin::AddedSubmission as i32 ||
        stream.pool_slot != 0 ||
        stream_allocation.key != stream.key ||
        transfer_allocation.key != stream.key ||
        stream.key.execution_owner_identity != owner_identity ||
        stream.key.context_identity !=
            u64::try_from(context_identity).map_err(|_| "context identity overflow")? ||
        stream.key.device != physical_device
    {
        return Err("borrowed compact store stream provenance differs from its owner".into());
    }
    if stream_allocation.allocation_kind() !=
        Some(mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationKind::SubmissionStream) ||
        transfer_allocation.allocation_kind() !=
            Some(
                mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationKind::TransferWorkspace,
            )
    {
        return Err("borrowed compact store allocation order differs from its descriptor".into());
    }
    Ok(Some((*stream_allocation, *transfer_allocation, *stream)))
}

pub(crate) fn prepared_output_codec_keys(
    program: &super::gpu_prepared_lowering::GpuPreparation,
    physical_devices: &[i32],
    instance_count: usize,
) -> Vec<(mxx_ir_core::types::WireRef, usize, i32)> {
    fn matrix_leaves(
        program: &super::gpu_prepared_lowering::GpuPreparation,
        wire: mxx_ir_core::types::WireRef,
        leaves: &mut Vec<mxx_ir_core::types::WireRef>,
    ) {
        match program.wire_types.get(&wire) {
            Some(mxx_ir_core::types::ConcreteWireType::Matrix(_)) |
            Some(mxx_ir_core::types::ConcreteWireType::Trapdoor { .. }) => leaves.push(wire),
            Some(mxx_ir_core::types::ConcreteWireType::IndexedFamily { .. }) => {
                if let Some(members) = program.family_wires.get(&wire) {
                    for member in members.iter().copied() {
                        matrix_leaves(program, member, leaves);
                    }
                }
            }
            _ => {}
        }
    }
    let mut output_wires = Vec::new();
    for wire in program.outputs.iter().copied() {
        matrix_leaves(program, wire, &mut output_wires);
    }
    let mut seen = BTreeSet::new();
    output_wires
        .into_iter()
        .flat_map(|wire| {
            (0..instance_count).flat_map(move |instance| {
                physical_devices.iter().copied().map(move |device| (wire, instance, device))
            })
        })
        .filter(|key| seen.insert(*key))
        .collect()
}

fn append_input_copy_slot<T>(
    slots: &mut Vec<T>,
    ordinal: usize,
    slot: T,
) -> Result<(), &'static str> {
    if ordinal != slots.len() {
        return Err("resolved input-copy completion claim is missing, duplicated, or out of order");
    }
    slots.push(slot);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::types::{ConcreteMatrixType, ConcreteWireType, NodeId, Port, WireRef};
    use std::collections::BTreeSet;

    fn test_stream_key() -> mxx_primitives::matrix::gpu_dcrt_poly::PreparedResourceKey {
        mxx_primitives::matrix::gpu_dcrt_poly::PreparedResourceKey {
            execution_owner_identity: 1,
            context_identity: 2,
            instance: 0,
            partition: 0,
            device: 3,
            limb_x: 0,
            limb_y: 0,
            role: 4,
        }
    }

    fn test_stream_claim(
        key: mxx_primitives::matrix::gpu_dcrt_poly::PreparedResourceKey,
        kind: i32,
    ) -> super::super::gpu_prepared_lowering::PreparedAllocationClaim {
        super::super::gpu_prepared_lowering::PreparedAllocationClaim {
            command: 1,
            ordinal: 0,
            store: None,
            layout: mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationLayout {
                key,
                kind,
                rows: 0,
                columns: 0,
                bytes: 0,
                alignment: 1,
                level: -1,
                format: -1,
            },
            owner_layout: None,
            slot: None,
        }
    }

    fn test_schedule_candidate(
        member: usize,
        stream: mxx_primitives::matrix::gpu_dcrt_poly::PreparedStreamFootprint,
        store: usize,
    ) -> super::super::gpu_prepared_lowering::PreparedScheduleMemberCandidate {
        super::super::gpu_prepared_lowering::PreparedScheduleMemberCandidate {
            member,
            stream,
            store,
            owner: 1,
            context_identity: 2,
            device: 3,
            instance: 0,
            level: 0,
            format: super::super::gpu_prepared_lowering::PreparedFormat::Evaluation,
        }
    }

    #[test]
    fn schedule_member_and_completion_provenance_rejects_missing_or_ambiguous_matches() {
        let stream = mxx_primitives::matrix::gpu_dcrt_poly::PreparedStreamFootprint {
            key: test_stream_key(),
            origin: 0,
            pool_slot: 5,
        };
        assert!(
            super::super::gpu_prepared_lowering::unique_prepared_schedule_member(&[], &stream)
                .is_err()
        );
        assert_eq!(
            super::super::gpu_prepared_lowering::unique_prepared_schedule_member(
                &[test_schedule_candidate(0, stream, 7), test_schedule_candidate(1, stream, 7)],
                &stream,
            )
            .unwrap(),
            0
        );
        assert!(
            super::super::gpu_prepared_lowering::unique_prepared_schedule_member(
                &[test_schedule_candidate(0, stream, 7), test_schedule_candidate(1, stream, 8)],
                &stream,
            )
            .is_err()
        );

        let key = test_stream_key();
        assert!(exact_prepared_stream_slot(&[], &key, 8, "schedule completion").is_err());
        let unresolved = test_stream_claim(key, 8);
        assert!(
            exact_prepared_stream_slot(
                std::slice::from_ref(&unresolved),
                &key,
                8,
                "schedule completion",
            )
            .is_err()
        );
        assert!(
            exact_prepared_stream_slot(
                &[test_stream_claim(key, 8), test_stream_claim(key, 8)],
                &key,
                8,
                "schedule completion",
            )
            .is_err()
        );
    }

    #[test]
    fn command_and_replay_stream_slots_reject_ambiguous_or_missing_allocations() {
        let key = test_stream_key();
        assert!(exact_prepared_stream_slot(&[], &key, 9, "command").is_err());
        assert!(exact_prepared_stream_slot(&[], &key, 9, "replay").is_err());
        let unresolved = test_stream_claim(key, 9);
        assert!(
            exact_prepared_stream_slot(std::slice::from_ref(&unresolved), &key, 9, "command",)
                .is_err()
        );
        assert!(
            exact_prepared_stream_slot(std::slice::from_ref(&unresolved), &key, 9, "replay",)
                .is_err()
        );
        let claims = vec![test_stream_claim(key, 9), test_stream_claim(key, 9)];
        assert!(exact_prepared_stream_slot(&claims, &key, 9, "command").is_err());
        assert!(exact_prepared_stream_slot(&claims, &key, 9, "replay").is_err());
    }

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

    #[test]
    fn prepared_output_codec_keys_cover_nested_matrix_family_leaves() {
        let root = WireRef { node: NodeId(10), port: Port(0) };
        let nested_a = WireRef { node: NodeId(11), port: Port(0) };
        let nested_b = WireRef { node: NodeId(12), port: Port(0) };
        let leaves = (0..4)
            .map(|node| WireRef { node: NodeId(20 + node), port: Port(0) })
            .collect::<Vec<_>>();
        let matrix =
            ConcreteMatrixType { modulus: 97.into(), ring_dimension: 8, rows: 1, columns: 1 };
        let mut program = super::super::gpu_prepared_lowering::GpuPreparation::default();
        program.outputs = vec![root].into_boxed_slice();
        program.wire_types.insert(
            root,
            ConcreteWireType::IndexedFamily {
                element: Box::new(ConcreteWireType::IndexedFamily {
                    element: Box::new(ConcreteWireType::Matrix(matrix.clone())),
                    count: 2,
                }),
                count: 2,
            },
        );
        program.wire_types.insert(
            nested_a,
            ConcreteWireType::IndexedFamily {
                element: Box::new(ConcreteWireType::Matrix(matrix.clone())),
                count: 2,
            },
        );
        program.wire_types.insert(
            nested_b,
            ConcreteWireType::IndexedFamily {
                element: Box::new(ConcreteWireType::Matrix(matrix.clone())),
                count: 2,
            },
        );
        for wire in leaves.iter().copied() {
            program.wire_types.insert(wire, ConcreteWireType::Matrix(matrix.clone()));
        }
        program.family_wires.insert(root, vec![nested_a, nested_b].into_boxed_slice());
        program.family_wires.insert(nested_a, leaves[..2].to_vec().into_boxed_slice());
        program.family_wires.insert(nested_b, leaves[2..].to_vec().into_boxed_slice());
        let keys = prepared_output_codec_keys(&program, &[7], 1);
        assert_eq!(keys, leaves.into_iter().map(|wire| (wire, 0, 7)).collect::<Vec<_>>());
    }

    #[test]
    fn prepared_output_codec_keys_canonicalize_repeated_family_leaf() {
        let root = WireRef { node: NodeId(30), port: Port(0) };
        let leaf = WireRef { node: NodeId(31), port: Port(0) };
        let matrix =
            ConcreteMatrixType { modulus: 97.into(), ring_dimension: 8, rows: 1, columns: 1 };
        let mut program = super::super::gpu_prepared_lowering::GpuPreparation::default();
        program.outputs = vec![root].into_boxed_slice();
        program.wire_types.insert(
            root,
            ConcreteWireType::IndexedFamily {
                element: Box::new(ConcreteWireType::Matrix(matrix.clone())),
                count: 2,
            },
        );
        program.wire_types.insert(leaf, ConcreteWireType::Matrix(matrix));
        program.family_wires.insert(root, vec![leaf, leaf].into_boxed_slice());
        assert_eq!(prepared_output_codec_keys(&program, &[7], 1), vec![(leaf, 0, 7)]);
    }

    #[test]
    fn prepared_output_codec_keys_keep_distinct_alias_wires_separate() {
        let root = WireRef { node: NodeId(40), port: Port(0) };
        let left = WireRef { node: NodeId(41), port: Port(0) };
        let right = WireRef { node: NodeId(42), port: Port(0) };
        let matrix =
            ConcreteMatrixType { modulus: 97.into(), ring_dimension: 8, rows: 1, columns: 1 };
        let mut program = super::super::gpu_prepared_lowering::GpuPreparation::default();
        program.outputs = vec![root].into_boxed_slice();
        program.wire_types.insert(
            root,
            ConcreteWireType::IndexedFamily {
                element: Box::new(ConcreteWireType::Matrix(matrix.clone())),
                count: 2,
            },
        );
        program.wire_types.insert(left, ConcreteWireType::Matrix(matrix.clone()));
        program.wire_types.insert(right, ConcreteWireType::Matrix(matrix));
        program.family_wires.insert(root, vec![left, right].into_boxed_slice());
        assert_eq!(
            prepared_output_codec_keys(&program, &[7], 1),
            vec![(left, 0, 7), (right, 0, 7)]
        );
    }

    #[test]
    fn borrowed_codec_descriptor_requires_exact_stream_and_allocation_order() {
        let key = test_stream_key();
        let stream = mxx_primitives::matrix::gpu_dcrt_poly::PreparedStreamFootprint {
            key,
            origin: mxx_primitives::matrix::gpu_dcrt_poly::PreparedStreamOrigin::AddedSubmission
                as i32,
            pool_slot: 0,
        };
        let allocations = [
            mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationLayout {
                key,
                kind: 9,
                rows: 0,
                columns: 0,
                bytes: 0,
                alignment: 1,
                level: -1,
                format: -1,
            },
            mxx_primitives::matrix::gpu_dcrt_poly::PreparedAllocationLayout {
                key,
                kind: 7,
                rows: 0,
                columns: 0,
                bytes: 128,
                alignment: 256,
                level: -1,
                format: -1,
            },
        ];
        assert!(
            validate_borrowed_codec_descriptor(
                1,
                1,
                &allocations,
                std::slice::from_ref(&stream),
                1,
                2,
                3
            )
            .is_ok()
        );
        let wrong_stream = mxx_primitives::matrix::gpu_dcrt_poly::PreparedStreamFootprint {
            pool_slot: 1,
            ..stream
        };
        assert!(
            validate_borrowed_codec_descriptor(
                1,
                1,
                &allocations,
                std::slice::from_ref(&wrong_stream),
                1,
                2,
                3,
            )
            .is_err()
        );
        assert_eq!(validate_borrowed_codec_descriptor(0, 1, &[], &[], 1, 2, 3).unwrap(), None);
    }

    #[test]
    fn root_matrix_input_completion_slots_are_complete_and_positional() {
        let mut slots = Vec::new();
        append_input_copy_slot(&mut slots, 0, 10u8).unwrap();
        append_input_copy_slot(&mut slots, 1, 11u8).unwrap();
        assert_eq!(slots, vec![10, 11]);
        assert!(append_input_copy_slot(&mut slots, 1, 12u8).is_err());
        assert!(append_input_copy_slot(&mut slots, 3, 13u8).is_err());
    }
}
