//! CPU-only ownership input to GPU admission. No GPU values, allocations,
//! reservations, events, measurements or discovery are created here.

use crate::{
    dataflow::{LoopAdmissionRequest, ValueLayout},
    gpu::GpuMeasurementError,
};
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv,
    types::{NodeId, WireRef},
};
use mxx_primitives::poly::PolyParams;
use mxx_runtime::{
    backend::poly_gpu::{GpuScopeProgress, GpuScopeResources},
    executor::gpu_plan::{InventoryPlan, inventory_plan, owner_liveness},
};
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

/// A cached value whose owner/descriptor remains live at this scope boundary.
/// An alias names the shared lowering's proven source wire. It does not prove
/// that two separately materialized descriptors have the same native owner.
pub struct GpuRetainedValue<'a> {
    pub wire: WireRef,
    pub alias: Option<WireRef>,
    pub layout: &'a ValueLayout,
}

/// Live metadata of one sibling in an active frame, in outermost-frame-first
/// order. Scope and instance identities stay separate: equal wire numbers do
/// not establish aliases between siblings. Progress describes host dispatch,
/// not native completion or reusable storage.
pub struct GpuRetainedScope<'a> {
    pub instance: usize,
    pub progress: mxx_runtime::backend::poly_gpu::GpuScopeProgress,
    pub scope: &'a FrozenGraphScopeId,
    pub node: NodeId,
    pub bindings: &'a ParamEnv,
    pub values: Vec<GpuRetainedValue<'a>>,
}

/// One context's fixed hypothetical native capacity. Eligibility is supplied by
/// the ownership scenario: retained owners and pending external readers must be
/// excluded before the search. Neither cloning this metadata nor fitting it
/// reserves native storage. Planned slot IDs are local to this context entry.
#[derive(Clone)]
pub struct GpuContextInventory {
    pub parameters: mxx_primitives::poly::dcrt::gpu::GpuDCRTPolyParams,
    pub slots: Vec<(mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotSnapshot, bool)>,
}

/// One ring's fleet capacity for production's shared column fitter. Context-local
/// slot IDs stay scoped to this view, never combined with another ring's IDs.
pub struct GpuColumnContextInventory<'a> {
    pub contexts: &'a [GpuContextInventory],
}

// Planned storage IDs are local to a context. One vector represents exactly
// one ring context across devices, so lookup never merges unrelated ID-zero plans.
impl mxx_runtime::gpu_memory::GpuColumnInventory for GpuColumnContextInventory<'_> {
    fn fits_requests(
        &self,
        device: usize,
        storage: u64,
        requests: &[mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRequest],
    ) -> Result<bool, mxx_runtime::gpu_memory::GpuAdmissionError> {
        use mxx_runtime::gpu_memory::GpuAdmissionError;
        let context = self.contexts.get(device).ok_or(GpuAdmissionError::InvalidDevice(device))?;
        let mut used = BTreeSet::new();
        for request in requests {
            if !used.insert(request.slot_key()) {
                return Ok(false);
            }
            let Some((slot, eligible)) = context.slots.iter().find(|(slot, _)| {
                let id = slot.identity();
                (id.storage_id(), id.slot_id(), id.slot_index()) == request.slot_key() &&
                    id.storage_id() == storage
            }) else {
                return Err(GpuAdmissionError::UnknownPreparedStorage);
            };
            if !eligible ||
                !slot
                    .fits_request(&context.parameters, request)
                    .map_err(GpuAdmissionError::NativeReservation)?
            {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn available_prepared_bytes(
        &self,
        device: usize,
        storage: u64,
    ) -> Result<u64, mxx_runtime::gpu_memory::GpuAdmissionError> {
        use mxx_runtime::gpu_memory::GpuAdmissionError;
        let context = self.contexts.get(device).ok_or(GpuAdmissionError::InvalidDevice(device))?;
        context
            .slots
            .iter()
            .filter(|(slot, eligible)| *eligible && slot.identity().storage_id() == storage)
            .try_fold(0u64, |total, (slot, _)| {
                let bytes = u64::try_from(slot.identity().requested_backing_bytes())
                    .map_err(|_| GpuAdmissionError::Overflow)?;
                total.checked_add(bytes).ok_or(GpuAdmissionError::Overflow)
            })
    }

    fn temporary_requirement(
        &self,
        _device: usize,
        _calibration: mxx_runtime::gpu_calibration::GpuDeviceCalibration,
        _allocations: mxx_runtime::gpu_memory::GpuColumnAllocations,
    ) -> Result<
        mxx_runtime::gpu_calibration::GpuTemporaryRequirement<'_>,
        mxx_runtime::gpu_calibration::GpuCalibrationError,
    > {
        Err(mxx_runtime::gpu_calibration::GpuCalibrationError::PreparedNativeFitRequired)
    }
}

impl GpuContextInventory {
    /// Select preparation for one proven input owner using production's source
    /// projection and native slot preference. `prepared` belongs to this exact
    /// input owner; equal-shaped independent inputs must pass their own list.
    /// The inventory already excludes all retained and pending owners. This
    /// returns metadata only and leaves it unchanged even when a candidate fails.
    pub fn matrix_input_plan(
        &self,
        input: &mxx_runtime::backend::poly_gpu::GpuInventoryValue,
        shape: (usize, usize),
        columns: std::ops::Range<usize>,
        device: usize,
        evaluation: bool,
        prepared: &[mxx_runtime::backend::poly_gpu::GpuMatrixInputLayout],
    ) -> Result<
        Option<(
            mxx_runtime::backend::poly_gpu::GpuMatrixInputSource,
            Vec<(
                mxx_runtime::backend::poly_gpu::GpuMatrixInputLayout,
                mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRequest,
            )>,
        )>,
        GpuMeasurementError,
    > {
        use mxx_runtime::backend::poly_gpu::{GpuMatrixInputLayout, GpuMatrixInputSource};
        let fragments = input.fragments.as_ref().ok_or_else(|| GpuMeasurementError(
            "input preparation needs proven source fragments; a containing layout is insufficient".into()))?;
        let (source, layouts) = GpuMatrixInputSource::plan(
            fragments.iter().copied(),
            shape,
            columns,
            device,
            &self.parameters,
            evaluation,
        );
        let layouts = layouts.into_iter().filter_map(|layout| {
            (!prepared.iter().any(|previous| previous.source == layout.source)).then_some(layout)
        });
        GpuMatrixInputLayout::assign(layouts, &self.parameters, &self.slots)
            .map(|selected| selected.map(|selected| (source, selected)))
            .map_err(GpuMeasurementError)
    }

    /// Exclude modeled retained owners through their existing typed assignment.
    /// Assignment entries correspond to this exact demand's `claims()` order
    /// and this context's slots. The caller supplies that established contract;
    /// equal shapes and matching wire numbers never imply a slot binding.
    /// Existing exclusions (including pending native readers) are preserved.
    pub fn exclude_retained(
        &mut self,
        demand: &mxx_runtime::backend::poly_gpu::GpuContextDemand,
        assignment: &[mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRequest],
        progress: impl Fn(usize) -> mxx_runtime::backend::poly_gpu::GpuScopeProgress,
    ) {
        let retained = demand
            .retained_claim_indices(None, progress)
            .into_iter()
            .map(|index| assignment[index].slot_key())
            .collect::<std::collections::BTreeSet<_>>();
        self.slots.par_iter_mut().for_each(|(slot, eligible)| {
            let identity = slot.identity();
            *eligible &= !retained.contains(&(
                identity.storage_id(),
                identity.slot_id(),
                identity.slot_index(),
            ));
        });
    }
}

/// The complete prepared-capacity scenario at one loop boundary. Each ring key
/// contains all target device/context inventories for that ring. Resident inputs
/// have explicit symbolic-owner-to-source-fragment bindings; equal shapes never
/// establish identity or placement. The column cap is a bound on existing storage.
#[derive(Clone)]
pub struct GpuAdmissionInventory {
    pub contexts: std::collections::BTreeMap<(String, usize), Vec<GpuContextInventory>>,
    pub sources: std::collections::BTreeMap<
        crate::dataflow::LayoutOwner,
        mxx_runtime::backend::poly_gpu::GpuInventoryValue,
    >,
    pub column_cap: usize,
}

impl GpuAdmissionInventory {
    /// Bind a single scope instance's already-derived source layouts to its
    /// logical materializations. Call separately for each instance's value map.
    /// Bounds retain unresolved formats/placements without inventing fragments.
    /// Missing layouts stay unbound. This does not create a native owner or
    /// assign a slot.
    pub fn bind_sources(
        &mut self,
        values: &std::collections::BTreeMap<WireRef, ValueLayout>,
        layouts: &std::collections::BTreeMap<
            WireRef,
            mxx_runtime::backend::poly_gpu::GpuInventoryValue,
        >,
    ) {
        fn owners(
            value: &ValueLayout,
            result: &mut std::collections::BTreeSet<crate::dataflow::LayoutOwner>,
        ) {
            use crate::dataflow::ValueStorage;
            match value {
                ValueLayout::Leaf(_, ValueStorage::Device, Some(owner)) => {
                    result.insert(owner.clone());
                }
                ValueLayout::Family { count, members, overrides, replication, .. } => {
                    let uniform = members.len() == 1 &&
                        match &members[0] {
                            ValueLayout::Leaf(_, _, owner) => owner.as_ref().is_none_or(|owner| {
                                replication.as_ref().is_none_or(|replication| {
                                    !replication.owners.contains(&owner.id)
                                })
                            }),
                            _ => replication.is_none(),
                        };
                    if uniform {
                        if *count > overrides.len() {
                            owners(&members[0], result);
                        }
                        for value in overrides.values() {
                            owners(value, result);
                        }
                    } else {
                        // Member selection applies the shared replication/override rules.
                        // Distinct resident members require distinct logical bindings.
                        for index in 0..*count {
                            owners(&value.member(index), result);
                        }
                    }
                }
                _ => {}
            }
        }
        let additions = values
            .par_iter()
            .filter_map(|(wire, value)| {
                let fragments = layouts.get(wire)?;
                let mut identities = std::collections::BTreeSet::new();
                owners(value, &mut identities);
                Some((identities, fragments))
            })
            .collect::<Vec<_>>();
        // Owner identity is the caller's established materialization contract:
        // aliases have the same layout, independently created values have
        // distinct keys. Preserve bindings already known by the scenario.
        for (identities, fragments) in additions {
            for owner in identities {
                self.sources.entry(owner).or_insert_with(|| fragments.clone());
            }
        }
    }
}

/// One setup-selected root scenario for a measurement worker. Typed assignments
/// refer to the immutable hypothetical inventory, never native backing IDs.
/// The regular estimator remains a synthetic scenario with fresh root inputs.
pub(crate) struct GpuGraphInventory {
    pub(crate) inventory: GpuAdmissionInventory,
    root: Arc<GpuScopeResources>,
    frames: Vec<GpuScopeFrame>,
    pending: Option<(GpuHypotheticalWave, Vec<bool>)>,
    owners: BTreeMap<crate::dataflow::LayoutOwner, GpuSlotSets>,
}

type GpuIssuedClaims = std::collections::BTreeMap<
    (String, usize),
    Vec<Vec<(usize, mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRequest)>>,
>;

type GpuSlotSets = BTreeMap<(String, usize), Vec<BTreeSet<(u64, u64, usize)>>>;
struct GpuScopeFrame {
    scope: FrozenGraphScopeId,
    resources: Arc<GpuScopeResources>,
    progress: Vec<GpuScopeProgress>,
    until: Vec<Vec<Vec<usize>>>,
    issued: BTreeMap<usize, GpuIssuedClaims>,
    initial: GpuIssuedClaims,
    placement: Option<(NodeId, usize, GpuSlotSets)>,
    returned: BTreeSet<crate::dataflow::LayoutOwner>,
    staged: Vec<bool>,
    parallel: bool,
    column_cap: usize,
    selected_node: Option<GpuSelectedNode>,
}

struct GpuSelectedNode {
    imported: GpuIssuedClaims,
    materialized: BTreeMap<(usize, WireRef), mxx_runtime::backend::poly_gpu::GpuInventoryValue>,
    inputs: Vec<Vec<Option<mxx_runtime::backend::poly_gpu::GpuMatrixDescriptor>>>,
    node: NodeId,
    plans: Vec<mxx_runtime::backend::poly_gpu::GpuAdmittedInvocationSummary>,
    setup: Vec<(
        mxx_ir_core::types::ConcreteMatrixType,
        Vec<mxx_primitives::matrix::gpu_dcrt_poly::GpuTracedClaim>,
    )>,
    outputs: BTreeMap<usize, (GpuSlotSets, mxx_runtime::backend::poly_gpu::GpuInventoryValue)>,
}

fn device_members(
    value: &ValueLayout,
    path: &mut Vec<usize>,
    visit: &mut impl FnMut(&crate::dataflow::LayoutOwner, &[usize]),
) {
    use crate::dataflow::ValueStorage;
    if !value.has_device_owner() {
        return
    }
    match value {
        ValueLayout::Leaf(_, ValueStorage::Device, Some(owner)) => visit(owner, path),
        ValueLayout::Family { count, members, overrides, replication, .. } => {
            if members.len() == 1 && overrides.is_empty() && replication.is_none() {
                path.push(0);
                device_members(&members[0], path, visit);
                path.pop();
            } else {
                for index in 0..*count {
                    path.push(index);
                    device_members(&value.member(index), path, visit);
                    path.pop();
                }
            }
        }
        _ => {}
    }
}

impl GpuGraphInventory {
    pub(crate) fn new(
        backend: &mut mxx_runtime::backend::poly_gpu::GpuDcrtBackend,
        graph: &mxx_ir_core::ValidatedGraph,
        wave_limit: usize,
    ) -> Result<Self, GpuMeasurementError> {
        use mxx_ir_core::{node::NodeKind, types::Port};
        use mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotSnapshot;
        use mxx_runtime::backend::poly_gpu::GpuInventoryValue;
        let mut inputs = graph
            .root_scope()
            .execution_order
            .par_iter()
            .enumerate()
            .filter_map(|(index, node)| {
                let NodeKind::Input { artifact, .. } = node.kind() else { return None };
                let wire = WireRef { node: NodeId(index as u64), port: Port(0) };
                let mut ty = &graph.root_scope().wire_types[&wire];
                while let mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. } = ty {
                    ty = element;
                }
                let mut value =
                    ty.matrix_type().map_or_else(GpuInventoryValue::default, |matrix| {
                        GpuInventoryValue::new(vec![0, matrix.columns])
                    });
                value.lazy = artifact.is_some();
                Some((wire, value))
            })
            .collect::<BTreeMap<_, _>>();
        // The backend owns non-Sync reservation tokens. Resolve its contexts
        // outside the parallel descriptor walk; no GPU work is performed here.
        for (wire, value) in &mut inputs {
            let mut ty = &graph.root_scope().wire_types[wire];
            while let mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. } = ty {
                ty = element;
            }
            // Establish the regular estimate's explicit fresh-input scenario
            // once. Later outputs use production's inherited source projection.
            // Artifact descriptors remain unresolved until materialization.
            if !value.lazy &&
                let mxx_ir_core::types::ConcreteWireType::Matrix(matrix) = ty
            {
                let parameters = backend
                    .resource_parameters(matrix)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let counts = mxx_runtime::gpu_calibration::gpu_capped_waterfill_columns(
                    &vec![matrix.columns; parameters.len()],
                    matrix.columns,
                )
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let mut start = 0;
                value.fragments = Some(
                    parameters
                        .iter()
                        .zip(counts)
                        .filter_map(|(parameters, count)| {
                            let end = start + count;
                            let fragment = (start < end).then_some(
                                mxx_runtime::backend::poly_gpu::GpuMatrixInputFragment {
                                    device: parameters.device_ids()[0],
                                    context: parameters.context_identity(),
                                    level: parameters.crt_depth() - 1,
                                    start,
                                    end,
                                    evaluation: true,
                                },
                            );
                            start = end;
                            fragment
                        })
                        .collect::<Vec<_>>()
                        .into(),
                );
            }
        }
        let available = backend
            .device_parameters()
            .par_iter()
            .map(|parameters| {
                mxx_primitives::poly::dcrt::gpu::gpu_device_memory_usage(parameters.device_ids()[0])
                    .map(|memory| parameters.vram_budget_bytes().saturating_sub(memory.resident))
                    .map_err(GpuMeasurementError)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (_, column_cap, root) = backend
            .graph_resource_demand(graph, false, &inputs, wave_limit, &available)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let contexts = root
            .contexts
            .iter()
            .map(|(key, (ty, demand))| {
                let parameters = backend
                    .resource_parameters(ty)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let contexts = parameters
                    .into_par_iter()
                    .map(|parameters| {
                        let slots =
                            GpuPreparedSlotSnapshot::plan(&parameters, &demand.claims(&parameters))
                                .map_err(GpuMeasurementError)?
                                .into_iter()
                                .map(|slot| (slot, true))
                                .collect();
                        Ok(GpuContextInventory { parameters, slots })
                    })
                    .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                Ok((key.clone(), contexts))
            })
            .collect::<Result<std::collections::BTreeMap<_, _>, GpuMeasurementError>>()?;
        Ok(Self {
            root: Arc::new(root),
            frames: Vec::new(),
            pending: None,
            owners: BTreeMap::new(),
            inventory: GpuAdmissionInventory { contexts, sources: Default::default(), column_cap },
        })
    }

    pub(crate) fn column_cap(&self) -> usize {
        self.frames.last().map_or(self.inventory.column_cap, |frame| frame.column_cap)
    }

    pub(crate) fn reset(&mut self) {
        self.frames.clear();
        self.pending = None;
        self.owners.clear();
        self.inventory.sources.clear();
    }

    fn available(&self) -> GpuAdmissionInventory {
        let mut inventory = self.inventory.clone();
        for (key, contexts) in &mut inventory.contexts {
            contexts.par_iter_mut().enumerate().for_each(|(index, context)| {
                let mut keys = BTreeSet::new();
                for frame in &self.frames {
                    if let Some((_, demand)) = frame.resources.contexts.get(key) {
                        for (begin, issued) in &frame.issued {
                            if let Some(assignments) = issued.get(key) {
                                let retained = demand
                                    .retained_claim_indices(Some(*begin), |i| frame.progress[i]);
                                keys.extend(assignments[index].iter().filter_map(
                                    |(claim, assigned)| {
                                        retained.contains(claim).then_some(assigned.slot_key())
                                    },
                                ));
                            }
                        }
                    }
                    if let Some((node, instance, placement)) = &frame.placement &&
                        matches!(frame.progress[*instance], GpuScopeProgress::Before(position) if position == node.0 as usize) &&
                        let Some(placement) = placement.get(key)
                    {
                        keys.extend(&placement[index]);
                    }
                }
                for owner in self.owners.values() {
                    if let Some(slots) = owner.get(key) {
                        keys.extend(&slots[index]);
                    }
                }
                context.slots.par_iter_mut().for_each(|(slot, eligible)| {
                    let identity = slot.identity();
                    *eligible &= !keys.contains(&(
                        identity.storage_id(),
                        identity.slot_id(),
                        identity.slot_index(),
                    ));
                });
            });
        }
        inventory
    }

    pub(crate) fn at_scope(&mut self, request: &LoopAdmissionRequest<'_>) -> GpuAdmissionInventory {
        if !self.frames.is_empty() {
            let instances = std::iter::once(crate::dataflow::ScopeSiblingState {
                index: request.active_instance,
                bindings: request.bindings.clone(),
                inputs: request.inputs.to_vec(),
                values: request.values.clone(),
            })
            .chain(request.siblings.iter().cloned())
            .collect::<Vec<_>>();
            // Returned values staged by the preceding wave no longer retain
            // device slots when the next wave is admitted.
            self.prune(&instances, request.ancestors);
        }
        let mut inventory = self.available();
        let resources = self.frames.last().map_or(&*self.root, |frame| &*frame.resources);
        inventory.bind_sources(request.values, &resources.value_layouts[request.active_instance]);
        inventory
            .bind_sources(request.broadcasts, &resources.value_layouts[request.active_instance]);
        inventory
    }

    pub(crate) fn accept(&mut self, request: &LoopAdmissionRequest<'_>, wave: GpuHypotheticalWave) {
        let stage_outputs = mxx_runtime::executor::staged_loop_outputs(
            request.scope,
            request.total_count,
            wave.instances,
            request.retained_outputs,
        );
        if request.first_index == 0 &&
            let Some(parent) = self.frames.last_mut()
        {
            let placement = wave
                .context_demands
                .iter()
                .map(|(key, (_, demand))| {
                    let indices =
                        demand.retained_claim_indices(None, |_| GpuScopeProgress::Before(0));
                    let slots = wave.assignments[key]
                        .iter()
                        .map(|assignment| {
                            indices.iter().map(|&index| assignment[index].slot_key()).collect()
                        })
                        .collect();
                    (key.clone(), slots)
                })
                .collect();
            parent.placement = Some((request.node, request.active_instance, placement));
        }
        self.pending = Some((wave, stage_outputs));
    }

    /// Keep logical returned values bound to their established hypothetical slots.
    /// Neither a scope transition nor a pooled index creates native ownership.
    pub(crate) fn step(
        &mut self,
        backend: &mut mxx_runtime::backend::poly_gpu::GpuDcrtBackend,
        graph: &mxx_ir_core::ValidatedGraph,
        scope: &FrozenGraphScopeId,
        instances: &[crate::dataflow::ScopeSiblingState],
        ancestors: &[crate::dataflow::ScopeAdmissionState],
        step: crate::dataflow::DataflowStep,
    ) -> Result<(), GpuMeasurementError> {
        use crate::dataflow::DataflowStep;
        use mxx_ir_core::node::NodeKind;
        match step {
            DataflowStep::Enter => {
                let (resources, initial, staged, parallel, column_cap) = if *scope ==
                    FrozenGraphScopeId::Root
                {
                    (
                        self.root.clone(),
                        GpuIssuedClaims::new(),
                        Vec::new(),
                        false,
                        self.inventory.column_cap,
                    )
                } else if let Some((wave, staged)) = self.pending.take() {
                    let initial = wave
                        .context_demands
                        .iter()
                        .map(|(key, (_, demand))| {
                            let indices = demand
                                .retained_claim_indices(None, |_| GpuScopeProgress::Before(0));
                            let assigned = wave.assignments[key]
                                .iter()
                                .map(|assignment| {
                                    indices.iter().map(|&i| (i, assignment[i])).collect()
                                })
                                .collect();
                            (key.clone(), assigned)
                        })
                        .collect();
                    (
                        Arc::new(GpuScopeResources {
                            contexts: wave.context_demands,
                            value_layouts: wave.value_layouts,
                        }),
                        initial,
                        staged,
                        true,
                        wave.column_cap,
                    )
                } else {
                    let input_wires = graph.source.scope(scope).unwrap().inputs();
                    let inputs = instances.par_iter().map(|instance| {
                        let values = instance.inputs.iter().zip(input_wires).map(|(value, wire)| {
                            let mut ty = graph.concrete_wire_type(scope, *wire, &instance.bindings)
                                .map_err(|error| GpuMeasurementError(error.to_string()))?;
                            while let mxx_ir_core::types::ConcreteWireType::IndexedFamily { element, .. } = ty {
                                ty = *element;
                            }
                            if ty.matrix_type().is_none() { return Ok(None) }
                            value.gpu_inventory(&self.inventory.sources).map(|value| Some((*wire, value)))
                        }).collect::<Result<Vec<_>, GpuMeasurementError>>()?
                            .into_iter().flatten().collect();
                        Ok((instance.bindings.clone(), values))
                    }).collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                    let cap = self
                        .frames
                        .last()
                        .map_or(self.inventory.column_cap, |frame| frame.column_cap);
                    let resources = backend
                        .scope_resource_demand(graph, scope, &inputs, cap, false)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    (Arc::new(resources), GpuIssuedClaims::new(), Vec::new(), false, cap)
                };
                let optimizer = if *scope == FrozenGraphScopeId::Root {
                    inventory_plan(graph, false)
                } else {
                    InventoryPlan::default()
                };
                let until = instances
                    .par_iter()
                    .map(|instance| {
                        let liveness =
                            owner_liveness(graph, scope, &instance.bindings, &optimizer, false)
                                .map_err(|_| {
                                    GpuMeasurementError(
                                        "invalid active-scope owner liveness".into(),
                                    )
                                })?;
                        Ok(graph
                            .scope(scope)
                            .unwrap()
                            .execution_order
                            .par_iter()
                            .enumerate()
                            .map(|(node, handle)| {
                                (0..handle.output_types().len())
                                    .map(|port| liveness.output_end(node, port))
                                    .collect()
                            })
                            .collect())
                    })
                    .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                self.frames.push(GpuScopeFrame {
                    scope: scope.clone(),
                    resources,
                    progress: vec![GpuScopeProgress::Before(0); instances.len()],
                    until,
                    issued: BTreeMap::new(),
                    initial,
                    placement: None,
                    returned: BTreeSet::new(),
                    staged,
                    parallel,
                    column_cap,
                    selected_node: None,
                });
                // Entered inputs already exist, even before their Input nodes publish.
                let inputs = graph.source.scope(scope).unwrap().inputs();
                let frame = self.frames.last().unwrap();
                for instance in instances {
                    let values =
                        inputs.iter().copied().zip(instance.inputs.iter().cloned()).collect();
                    self.inventory
                        .bind_sources(&values, &frame.resources.value_layouts[instance.index]);
                }
            }
            DataflowStep::BeforeNode(node) => {
                let frame = self.frames.last_mut().unwrap();
                frame.progress.fill(GpuScopeProgress::Before(node.0 as usize));
                frame.returned.clear();
                frame.issued.retain(|begin, assignments| {
                    assignments.keys().any(|key| {
                        !frame.resources.contexts[key]
                            .1
                            .retained_claim_indices(Some(*begin), |i| frame.progress[i])
                            .is_empty()
                    })
                });
                self.prune(instances, ancestors);
                if !matches!(
                    graph.source.scope(scope).unwrap().node(node).unwrap().kind(),
                    NodeKind::ParallelLoop(_) |
                        NodeKind::SequentialLoop(_) |
                        NodeKind::SubgraphCall(_)
                ) {
                    let selected = self.select_node(backend, graph, scope, node, instances)?;
                    if selected.is_none() {
                        self.issue_node(node)?;
                    }
                    let frame = self.frames.last_mut().unwrap();
                    if let Some(selected) = &selected &&
                        !selected.imported.is_empty()
                    {
                        frame.issued.insert(node.0 as usize, selected.imported.clone());
                    }
                    frame.selected_node = selected;
                }
            }
            DataflowStep::AfterNode { node, instance } => {
                self.frames.last_mut().unwrap().progress[instance] =
                    GpuScopeProgress::Issued(node.0 as usize);
                self.bind_values(graph, &instances[instance..instance + 1], Some(node));
            }
            DataflowStep::Leave => {
                self.bind_values(graph, instances, None);
                let outputs = graph.source.scope(scope).unwrap().outputs();
                let frame = self.frames.pop().unwrap();
                debug_assert_eq!(&frame.scope, scope);
                if let Some(parent) = self.frames.last_mut() {
                    if !frame.parallel {
                        parent.returned.clear();
                    }
                    for instance in instances {
                        for (port, wire) in outputs.iter().enumerate() {
                            let value = &instance.values[wire];
                            let artifact = frame.parallel &&
                                matches!(value,
                                ValueLayout::Leaf(ty, _, _) if
                                    mxx_ir_core::artifact::ArtifactType::from_wire_type(ty).is_some() &&
                                    !matches!(ty, mxx_ir_core::types::ConcreteWireType::Matrix(_)));
                            if frame.staged.get(port).copied().unwrap_or(false) || artifact {
                                continue
                            }
                            device_members(value, &mut Vec::new(), &mut |owner, _| {
                                parent.returned.insert(owner.clone());
                            });
                        }
                    }
                } else {
                    self.owners.clear();
                    self.inventory.sources.clear();
                }
            }
        }
        Ok(())
    }

    fn select_node(
        &self,
        backend: &mxx_runtime::backend::poly_gpu::GpuDcrtBackend,
        graph: &mxx_ir_core::ValidatedGraph,
        scope: &FrozenGraphScopeId,
        node: NodeId,
        instances: &[crate::dataflow::ScopeSiblingState],
    ) -> Result<Option<GpuSelectedNode>, GpuMeasurementError> {
        use mxx_ir_core::types::ConcreteWireType;
        use mxx_runtime::{
            backend::poly_gpu::{
                GpuColumnShard, GpuMatrixDescriptor, GpuMatrixFragmentDescriptor,
                GpuMatrixInputFragment, GpuMatrixPlacementInvocation, GpuMatrixSlotContext,
            },
            gpu_invocation::GpuNodeOperation,
            gpu_memory::GpuDeviceAdmission,
        };
        let handle = graph.source.scope(scope).unwrap().node(node).unwrap();
        let arguments = graph.source.scope(scope).unwrap().arguments(handle).unwrap();
        let mut available = self.available();
        let mut imported = GpuIssuedClaims::new();
        let mut materialized = BTreeMap::new();
        let mut invocations = Vec::with_capacity(instances.len());
        let mut inputs = Vec::with_capacity(instances.len());
        let mut output_keys = Vec::with_capacity(instances.len());
        let mut owners = BTreeMap::new();
        for instance in instances {
            let operation = GpuNodeOperation::new(graph, scope, node, &instance.bindings)
                .map_err(GpuMeasurementError)?;
            let [output] = operation.outputs() else { return Ok(None) };
            let output = match output {
                ConcreteWireType::Matrix(matrix) |
                ConcreteWireType::SmallMatrix { matrix, .. } |
                ConcreteWireType::Preimage { matrix, .. } => matrix,
                _ => return Ok(None),
            };
            let mut descriptors = Vec::with_capacity(arguments.len());
            for (wire, ty) in arguments.iter().zip(operation.arguments()) {
                // The traced sampler's fixed public/trapdoor/target resources
                // are owned outside its column invocation, as in native lowering.
                if matches!(operation.kind(), mxx_ir_core::node::NodeKind::PreimageSample { .. }) {
                    descriptors.push(None);
                    continue;
                }

                let evaluation = matches!(ty, ConcreteWireType::Matrix(_));
                let ty = match ty {
                    ConcreteWireType::Matrix(matrix) |
                    ConcreteWireType::SmallMatrix { matrix, .. } |
                    ConcreteWireType::Preimage { matrix, .. } => matrix,
                    ConcreteWireType::IndexedFamily { .. } | ConcreteWireType::Trapdoor { .. } => {
                        return Ok(None)
                    }
                    _ => {
                        descriptors.push(None);
                        continue;
                    }
                };
                let value = &instance.values[wire];
                let ValueLayout::Leaf(_, storage, owner) = value else { return Ok(None) };
                let key = (ty.modulus.to_string(), ty.ring_dimension);
                let imported_key = (instance.index, *wire);
                let fragments = if *storage == crate::dataflow::ValueStorage::Device {
                    let layout = value.gpu_inventory(&self.inventory.sources)?;
                    if layout.alternatives || layout.lazy {
                        return Ok(None)
                    }
                    let Some(fragments) = layout.fragments else { return Ok(None) };
                    fragments
                } else if let Some(layout) = materialized.get(&imported_key) {
                    let layout: &mxx_runtime::backend::poly_gpu::GpuInventoryValue = layout;
                    layout.fragments.clone().unwrap()
                } else {
                    // A descriptor has no owner until this consumer imports it.
                    // Retain its typed import claims before choosing outputs; the
                    // separately measured transfer accounts for actual upload work.
                    let demand = &self.frames.last().unwrap().resources.contexts[&key].1;
                    let indices = demand.value_claim_indices(
                        instance.index,
                        *wire,
                        &[],
                        GpuScopeProgress::Issued(node.0 as usize),
                    );
                    let contexts = available.contexts.get_mut(&key).unwrap();
                    let assignments = imported
                        .entry(key.clone())
                        .or_insert_with(|| vec![Vec::new(); contexts.len()]);
                    for (device, context) in contexts.iter_mut().enumerate() {
                        let claims = demand.claims(&context.parameters);
                        let claims = indices.iter().map(|&index| claims[index]).collect::<Vec<_>>();
                        let chosen =
                            mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotSnapshot::assign(
                                &context.parameters,
                                &context.slots,
                                &claims,
                            )
                            .map_err(GpuMeasurementError)?
                            .into_iter()
                            .collect::<Option<Vec<_>>>()
                            .ok_or_else(|| {
                                GpuMeasurementError(
                                    "materialized input claims do not fit the selected inventory"
                                        .into(),
                                )
                            })?;
                        for slot in &mut context.slots {
                            if chosen.iter().any(|claim| {
                                claim.slot_key() ==
                                    (
                                        slot.0.identity().storage_id(),
                                        slot.0.identity().slot_id(),
                                        slot.0.identity().slot_index(),
                                    )
                            }) {
                                slot.1 = false;
                            }
                        }
                        assignments[device].extend(indices.iter().copied().zip(chosen));
                    }
                    // The report's SyntheticFreshPlacement scenario fixes imported
                    // source geometry, without pretending that descriptor lifetime is
                    // device residency. Artifact bytes need not be loaded to estimate.
                    let counts = mxx_runtime::gpu_calibration::gpu_capped_waterfill_columns(
                        &vec![ty.columns; contexts.len()],
                        ty.columns,
                    )
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                    let mut start = 0;
                    let fragments = contexts
                        .iter()
                        .zip(counts)
                        .filter_map(|(context, count)| {
                            let end = start + count;
                            let fragment = (start < end).then_some(GpuMatrixInputFragment {
                                device: context.parameters.device_ids()[0],
                                context: context.parameters.context_identity(),
                                start,
                                end,
                                level: context.parameters.crt_depth() - 1,
                                evaluation,
                            });
                            start = end;
                            fragment
                        })
                        .collect::<Vec<_>>();
                    let mut layout = mxx_runtime::backend::poly_gpu::GpuInventoryValue::new(
                        fragments
                            .iter()
                            .flat_map(|fragment| [fragment.start, fragment.end])
                            .collect(),
                    );
                    let fragments: Arc<[_]> = fragments.into();
                    layout.fragments = Some(fragments.clone());
                    materialized.insert(imported_key, layout);
                    fragments
                };
                let contexts = &available.contexts[&key];
                let next = owners.len() as u64;
                let identity = if let Some(owner) = owner {
                    (Some(owner.clone()), None)
                } else {
                    (None, Some(imported_key))
                };
                let id = *owners.entry(identity).or_insert(next);
                let shards = fragments
                    .iter()
                    .map(|fragment| {
                        let parameters = contexts
                            .iter()
                            .find(|context| {
                                context.parameters.context_identity() == fragment.context
                            })
                            .ok_or_else(|| {
                                GpuMeasurementError(
                                    "input context is absent from the selected inventory".into(),
                                )
                            })?
                            .parameters
                            .clone();
                        Ok(GpuColumnShard {
                            device_id: fragment.device,
                            global_column_start: fragment.start,
                            value: GpuMatrixFragmentDescriptor {
                                parameters,
                                level: fragment.level,
                                columns: fragment.end - fragment.start,
                                evaluation: fragment.evaluation,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
                descriptors.push(Some(GpuMatrixDescriptor {
                    id,
                    rows: ty.rows,
                    columns: ty.columns,
                    shards,
                    input_layout: fragments,
                }));
            }
            inputs.push(descriptors.clone());
            let Some(invocation) =
                GpuMatrixPlacementInvocation::new(backend, &operation, descriptors)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?
            else {
                return Ok(None)
            };
            invocations.push(invocation);
            output_keys.push((output.modulus.to_string(), output.ring_dimension));
        }
        let slots = available
            .contexts
            .values()
            .flat_map(|contexts| {
                contexts.iter().enumerate().map(|(device, context)| GpuMatrixSlotContext {
                    device,
                    context: context.parameters.context_identity(),
                    slots: context.slots.clone(),
                })
            })
            .collect();
        let selected = backend
            .plan_matrix_layouts(&invocations, &slots, &Default::default())
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let inventories = output_keys
            .iter()
            .map(|key| GpuColumnContextInventory { contexts: &available.contexts[key] })
            .collect::<Vec<_>>();
        let admissions = backend
            .device_parameters()
            .iter()
            .map(|_| GpuDeviceAdmission {
                budget_bytes: 0,
                baseline_bytes: 0,
                allocation_bytes: 0,
                physical_baseline_bytes: 0,
                observed_resident_bytes: 0,
                observed_physical_bytes: 0,
            })
            .collect::<Vec<_>>();
        let plans = selected
            .fit(&admissions, self.column_cap(), &inventories)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let outputs = instances
            .iter()
            .enumerate()
            .map(|(index, instance)| {
                let selected_outputs = selected.outputs(index);
                let fragments = selected_outputs
                    .iter()
                    .map(|(interval, source, _)| GpuMatrixInputFragment {
                        device: source.parameters.device_ids()[0],
                        context: source.parameters.context_identity(),
                        start: interval.start,
                        end: interval.end,
                        level: source.level,
                        evaluation: source.evaluation,
                    })
                    .collect::<Vec<_>>();
                let mut layout = mxx_runtime::backend::poly_gpu::GpuInventoryValue::new(
                    fragments.iter().flat_map(|fragment| [fragment.start, fragment.end]).collect(),
                );
                layout.fragments = Some(fragments.into());
                let mut slots = vec![BTreeSet::new(); admissions.len()];
                for (interval, _, request) in selected_outputs {
                    slots[interval.device].insert(request.slot_key());
                }
                (instance.index, (BTreeMap::from([(output_keys[index].clone(), slots)]), layout))
            })
            .collect();
        let setup = self
            .frames
            .last()
            .unwrap()
            .resources
            .contexts
            .iter()
            .map(|(key, (ty, demand))| {
                let claims = demand.claims(&available.contexts[key][0].parameters);
                let selected = demand.retained_claim_indices(Some(node.0 as usize), |_| {
                    GpuScopeProgress::Issued(node.0 as usize)
                });
                (
                    ty.clone(),
                    selected
                        .into_iter()
                        .filter(|index| {
                            !imported.get(key).is_some_and(|devices| {
                                devices[0].iter().any(|(imported, _)| imported == index)
                            })
                        })
                        .map(|index| claims[index])
                        .collect(),
                )
            })
            .collect();
        Ok(Some(GpuSelectedNode { node, plans, outputs, setup, inputs, imported, materialized }))
    }

    pub(crate) fn selected_node_plans(
        &self,
        node: NodeId,
    ) -> Option<&[mxx_runtime::backend::poly_gpu::GpuAdmittedInvocationSummary]> {
        let selected = self.frames.last()?.selected_node.as_ref()?;
        (selected.node == node).then_some(selected.plans.as_slice())
    }

    pub(crate) fn selected_node_inputs(
        &self,
        node: NodeId,
    ) -> Option<&[Vec<Option<mxx_runtime::backend::poly_gpu::GpuMatrixDescriptor>>]> {
        let selected = self.frames.last()?.selected_node.as_ref()?;
        (selected.node == node).then_some(selected.inputs.as_slice())
    }

    pub(crate) fn selected_node_setup(
        &self,
        node: NodeId,
    ) -> Option<
        &[(
            mxx_ir_core::types::ConcreteMatrixType,
            Vec<mxx_primitives::matrix::gpu_dcrt_poly::GpuTracedClaim>,
        )],
    > {
        let selected = self.frames.last()?.selected_node.as_ref()?;
        (selected.node == node).then_some(selected.setup.as_slice())
    }

    fn issue_node(&mut self, node: NodeId) -> Result<(), GpuMeasurementError> {
        use mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotSnapshot;
        let inventory = self.available();
        let frame = self.frames.last_mut().unwrap();
        let issued = frame.resources.contexts.par_iter().map(|(key, (_, demand))| {
            let indices = demand.retained_claim_indices(Some(node.0 as usize), |_| GpuScopeProgress::Issued(node.0 as usize))
                .into_iter().filter(|index| !frame.initial.get(key).is_some_and(|contexts|
                    contexts.first().is_some_and(|claims| claims.iter().any(|(i, _)| i == index))))
                .collect::<Vec<_>>();
            let assignments = inventory.contexts[key].par_iter().map(|context| {
                let all = demand.claims(&context.parameters);
                let claims = indices.iter().map(|&index| all[index]).collect::<Vec<_>>();
                let assigned = GpuPreparedSlotSnapshot::assign(&context.parameters, &context.slots, &claims)
                    .map_err(GpuMeasurementError)?;
                indices.iter().copied().zip(assigned).map(|(index, request)| {
                    request.map(|request| (index, request)).ok_or_else(|| GpuMeasurementError(format!(
                        "scope {:?} node {node:?} cannot fit context {key:?} around retained assignments", frame.scope)))
                }).collect::<Result<Vec<_>, _>>()
            }).collect::<Result<Vec<_>, _>>()?;
            Ok((key.clone(), assignments))
        }).collect::<Result<GpuIssuedClaims, GpuMeasurementError>>()?;
        frame.issued.insert(node.0 as usize, issued);
        Ok(())
    }

    fn bind_values(
        &mut self,
        graph: &mxx_ir_core::ValidatedGraph,
        instances: &[crate::dataflow::ScopeSiblingState],
        node: Option<NodeId>,
    ) {
        let frame = self.frames.last().unwrap();
        let scope = graph.source.scope(&frame.scope).unwrap();
        let wires = if let Some(node) = node {
            let handle = scope.node(node).unwrap();
            scope
                .arguments(handle)
                .unwrap()
                .into_iter()
                .chain(
                    (0..handle.output_types().len())
                        .map(|port| WireRef { node, port: mxx_ir_core::types::Port(port as u32) }),
                )
                .collect::<BTreeSet<_>>()
        } else {
            scope.outputs().iter().copied().collect()
        };
        for instance in instances {
            let values = wires
                .iter()
                .filter_map(|wire| instance.values.get(wire).map(|value| (*wire, value.clone())))
                .collect();
            self.inventory.bind_sources(&values, &frame.resources.value_layouts[instance.index]);
            for (wire, value) in &values {
                device_members(value, &mut Vec::new(), &mut |owner, members| {
                    if let Some(layout) = frame
                        .selected_node
                        .as_ref()
                        .and_then(|selected| selected.materialized.get(&(instance.index, *wire)))
                    {
                        self.inventory.sources.insert(owner.clone(), layout.clone());
                    }

                    if self.owners.contains_key(owner) {
                        return
                    }
                    if node == Some(wire.node) &&
                        wire.port.0 == 0 &&
                        let Some(selected) = frame
                            .selected_node
                            .as_ref()
                            .filter(|selected| selected.node == wire.node) &&
                        let Some((slots, layout)) = selected.outputs.get(&instance.index)
                    {
                        self.owners.insert(owner.clone(), slots.clone());
                        self.inventory.sources.insert(owner.clone(), layout.clone());
                        return;
                    }
                    // A later argument/alias publication must preserve the exact
                    // owner assignment made when its output was selected.
                    if self.owners.contains_key(owner) {
                        return;
                    }
                    let mut slots = GpuSlotSets::new();
                    for (key, (_, demand)) in &frame.resources.contexts {
                        let selected = demand.value_claim_indices(
                            instance.index,
                            *wire,
                            members,
                            frame.progress[instance.index],
                        );
                        let mut contexts =
                            vec![BTreeSet::new(); self.inventory.contexts[key].len()];
                        for (begin, issued) in &frame.issued {
                            let Some(assignments) = issued.get(key) else { continue };
                            let live =
                                demand.retained_claim_indices(Some(*begin), |i| frame.progress[i]);
                            for (context, assignments) in contexts.iter_mut().zip(assignments) {
                                context.extend(assignments.iter().filter_map(
                                    |(index, request)| {
                                        (selected.contains(index) && live.contains(index))
                                            .then_some(request.slot_key())
                                    },
                                ));
                            }
                        }
                        if let Some(initial) = frame.initial.get(key) {
                            for (context, assignments) in contexts.iter_mut().zip(initial) {
                                context.extend(assignments.iter().filter_map(
                                    |(index, request)| {
                                        selected.contains(index).then_some(request.slot_key())
                                    },
                                ));
                            }
                        }
                        if contexts.iter().any(|context| !context.is_empty()) {
                            slots.insert(key.clone(), contexts);
                        }
                    }
                    if slots.is_empty() {
                        let scope = graph.source.scope(&frame.scope).unwrap();
                        let handle = scope.node(wire.node).unwrap();
                        if matches!(
                            handle.kind(),
                            mxx_ir_core::node::NodeKind::Select { .. } |
                                mxx_ir_core::node::NodeKind::FamilyGetDynamic |
                                mxx_ir_core::node::NodeKind::TrapdoorPublic
                        ) {
                            // An unresolved selection has a new logical ID, but
                            // no new backing. Retain the containing candidate
                            // owner set rather than inventing a native identity.
                            for argument in scope.arguments(handle).unwrap() {
                                if let Some(value) = instance.values.get(&argument) {
                                    device_members(value, &mut Vec::new(), &mut |source, _| {
                                        if let Some(bound) = self.owners.get(source) {
                                            for (key, contexts) in bound {
                                                let target =
                                                    slots.entry(key.clone()).or_insert_with(|| {
                                                        vec![BTreeSet::new(); contexts.len()]
                                                    });
                                                for (target, source) in
                                                    target.iter_mut().zip(contexts)
                                                {
                                                    target.extend(source.iter().copied());
                                                }
                                            }
                                        }
                                    });
                                }
                            }
                        }
                    }
                    if !slots.is_empty() {
                        self.owners.insert(owner.clone(), slots);
                    }
                });
            }
        }
    }

    fn prune(
        &mut self,
        instances: &[crate::dataflow::ScopeSiblingState],
        ancestors: &[crate::dataflow::ScopeAdmissionState],
    ) {
        let mut live = self
            .frames
            .iter()
            .flat_map(|frame| frame.returned.iter().cloned())
            .collect::<BTreeSet<_>>();
        let current = self.frames.len() - 1;
        let mut states = instances
            .iter()
            .map(|instance| (current, instance.index, &instance.inputs, &instance.values))
            .collect::<Vec<_>>();
        for (depth, ancestor) in ancestors.iter().enumerate() {
            states.push((depth, ancestor.active_instance, &ancestor.inputs, &ancestor.values));
            states.extend(
                ancestor
                    .siblings
                    .iter()
                    .map(|instance| (depth, instance.index, &instance.inputs, &instance.values)),
            );
        }
        let retained =
            states
                .into_par_iter()
                .map(|(depth, instance, inputs, values)| {
                    let mut live = BTreeSet::new();
                    let frame = &self.frames[depth];
                    for input in inputs {
                        device_members(input, &mut Vec::new(), &mut |owner, _| {
                            live.insert(owner.clone());
                        });
                    }
                    let (position, issued) = match frame.progress[instance] {
                        GpuScopeProgress::Before(position) => (position, false),
                        GpuScopeProgress::Issued(position) => (position, true),
                    };
                    for (wire, value) in values {
                        if (wire.node.0 < position as u64 ||
                            (issued && wire.node.0 == position as u64)) &&
                            frame.until[instance][wire.node.0 as usize][wire.port.0 as usize] >=
                                position
                        {
                            device_members(value, &mut Vec::new(), &mut |owner, _| {
                                live.insert(owner.clone());
                            });
                        }
                    }
                    live
                })
                .collect::<Vec<_>>();
        for owners in retained {
            live.extend(owners);
        }
        self.owners.retain(|owner, _| live.contains(owner));
        self.inventory.sources.retain(|owner, _| live.contains(owner));
    }
}

/// A CPU-only fit, not a native reservation or permission to execute. Assignments
/// retain ring and context-entry identity because unbacked slot IDs are local.
pub struct GpuHypotheticalWave {
    pub instances: usize,
    pub column_cap: usize,
    /// Accepted per-body wire layouts, retained from production lowering.
    /// These describe placement/format, not native owner or slot identity.
    pub value_layouts:
        Vec<std::collections::BTreeMap<WireRef, mxx_runtime::backend::poly_gpu::GpuInventoryValue>>,
    pub context_demands: std::collections::BTreeMap<
        (String, usize),
        (mxx_ir_core::types::ConcreteMatrixType, mxx_runtime::backend::poly_gpu::GpuContextDemand),
    >,
    pub assignments: std::collections::BTreeMap<
        (String, usize),
        Vec<Vec<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRequest>>,
    >,
}

impl GpuHypotheticalWave {
    /// Resolve a scope-owned value bound through this wave's established typed
    /// assignment. Context ordering is unchanged. The returned unbacked requests
    /// are not actual native output addresses and grant no reservation.
    pub fn value_claims(
        &self,
        instance: usize,
        wire: WireRef,
        members: &[usize],
        progress: mxx_runtime::backend::poly_gpu::GpuScopeProgress,
    ) -> std::collections::BTreeMap<
        (String, usize),
        Vec<Vec<mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedRequest>>,
    > {
        self.context_demands
            .par_iter()
            .map(|(key, (_, demand))| {
                let indices = demand.value_claim_indices(instance, wire, members, progress);
                let contexts = self.assignments[key]
                    .par_iter()
                    .map(|assignment| indices.iter().map(|&index| assignment[index]).collect())
                    .collect();
                (key.clone(), contexts)
            })
            .collect()
    }
}

impl LoopAdmissionRequest<'_> {
    /// Choose the largest feasible prefix against one immutable inventory, then
    /// the largest supported halving column cap. No rejected candidate changes
    /// input materialization, ownership, eligibility, backing or GPU execution.
    /// The supplied scenario must cover every active parent and future owner;
    /// this query does not construct that scenario from shapes alone.
    pub fn gpu_admit_wave(
        &self,
        backend: &mut mxx_runtime::backend::poly_gpu::GpuDcrtBackend,
        first_index: usize,
        caller_limit: usize,
        inventory: &GpuAdmissionInventory,
    ) -> Result<GpuHypotheticalWave, GpuMeasurementError> {
        use mxx_ir_core::node::NodeKind;
        use mxx_primitives::matrix::gpu_dcrt_poly::GpuPreparedSlotSnapshot;
        let limit = caller_limit.min(self.total_count - first_index).min(u16::MAX as usize);
        if limit == 0 {
            return Ok(GpuHypotheticalWave {
                instances: 0,
                column_cap: 0,
                value_layouts: Vec::new(),
                context_demands: Default::default(),
                assignments: Default::default(),
            });
        }
        let parent = self.graph.source.scope(self.scope).expect("validated scope");
        let handle = parent.node(self.node).expect("validated loop");
        let NodeKind::ParallelLoop(spec) = handle.kind() else { unreachable!("loop admission") };
        let child =
            self.graph.source.child_scope_id(self.scope, self.node).expect("validated child");
        let count = self.graph.scope(&child).expect("validated child").execution_order.len();
        let maxima = (first_index..first_index + limit)
            .into_par_iter()
            .map(|index| {
                let env = self
                    .bindings
                    .child(&spec.bindings, Some((spec.index_slot, index)))
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                (0..count)
                    .into_par_iter()
                    .map(|position| {
                        let operation = mxx_runtime::gpu_invocation::GpuNodeOperation::new(
                            self.graph,
                            &child,
                            NodeId(position as u64),
                            &env,
                        )
                        .map_err(GpuMeasurementError)?;
                        Ok(operation
                            .arguments()
                            .iter()
                            .chain(operation.outputs())
                            .filter_map(|ty| ty.matrix_type().map(|ty| ty.columns))
                            .max()
                            .unwrap_or(1))
                    })
                    .try_reduce(|| 1, |a, b| Ok(a.max(b)))
            })
            .collect::<Result<Vec<usize>, GpuMeasurementError>>()?;
        for wave in (1..=limit).rev() {
            let mut cap =
                maxima[..wave].iter().copied().max().unwrap_or(1).min(inventory.column_cap).max(1);
            loop {
                let demands =
                    self.gpu_wave_demand(backend, first_index, wave, cap, &inventory.sources)?;
                let mut assignments = std::collections::BTreeMap::new();
                let mut fits = true;
                for (key, (_, demand)) in &demands.contexts {
                    let contexts = inventory.contexts.get(key).ok_or_else(|| {
                        GpuMeasurementError(format!(
                            "missing prepared context {key:?} at {:?} node {:?}",
                            self.scope, self.node
                        ))
                    })?;
                    if contexts.is_empty() {
                        fits = false;
                        break;
                    }
                    let selected = contexts
                        .par_iter()
                        .map(|context| {
                            let claims = demand.claims(&context.parameters);
                            GpuPreparedSlotSnapshot::assign(
                                &context.parameters,
                                &context.slots,
                                &claims,
                            )
                            .map(|requests| requests.into_iter().collect::<Option<Vec<_>>>())
                            .map_err(GpuMeasurementError)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let Some(selected) = selected.into_iter().collect::<Option<Vec<_>>>() else {
                        fits = false;
                        break;
                    };
                    assignments.insert(key.clone(), selected);
                }
                if fits {
                    return Ok(GpuHypotheticalWave {
                        instances: wave,
                        column_cap: cap,
                        value_layouts: demands.value_layouts,
                        context_demands: demands.contexts,
                        assignments,
                    });
                }
                if cap == 1 {
                    break;
                }
                cap = cap.div_ceil(2);
            }
        }
        Err(GpuMeasurementError(format!(
            "one body with one-column scratch does not fit the supplied eligible inventory at {:?} node {:?}",
            self.scope, self.node
        )))
    }

    /// Lower an ordered candidate through production's complete sibling-demand
    /// calculation. Caller-bound fragments establish the selected resident
    /// sources; no input is imported here. The caller must separately exclude
    /// all retained owners when fitting these claims to its eligible inventory.
    pub fn gpu_wave_demand(
        &self,
        backend: &mut mxx_runtime::backend::poly_gpu::GpuDcrtBackend,
        first_index: usize,
        wave: usize,
        scratch_columns: usize,
        bindings: &std::collections::BTreeMap<
            crate::dataflow::LayoutOwner,
            mxx_runtime::backend::poly_gpu::GpuInventoryValue,
        >,
    ) -> Result<mxx_runtime::backend::poly_gpu::GpuScopeResources, GpuMeasurementError> {
        use crate::dataflow::ValueStorage;
        use mxx_ir_core::{
            node::{LoopInputMode, NodeKind},
            types::ConcreteWireType,
        };
        let parent = self.graph.source.scope(self.scope).expect("validated scope");
        let handle = parent.node(self.node).expect("validated loop");
        let NodeKind::ParallelLoop(spec) = handle.kind() else { unreachable!("loop admission") };
        let child_id =
            self.graph.source.child_scope_id(self.scope, self.node).expect("validated child");
        let child = self.graph.source.scope(&child_id).expect("validated child");
        let instances = (first_index..first_index + wave)
            .into_par_iter()
            .map(|index| {
                let env = self
                    .bindings
                    .child(&spec.bindings, Some((spec.index_slot, index)))
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let inputs = self
                    .selected_inputs(index)
                    .into_iter()
                    .zip(child.inputs())
                    .zip(self.input_modes)
                    .map(|(((parent, value), wire), mode)| {
                        let ty = self
                            .graph
                            .concrete_wire_type(&child_id, *wire, &env)
                            .map_err(|error| GpuMeasurementError(error.to_string()))?;
                        if ty.matrix_type().is_none() {
                            return Ok(None);
                        }
                        let mut layout = value.gpu_inventory(bindings)?;
                        if matches!(value, ValueLayout::Leaf(_, ValueStorage::Artifact, _)) &&
                            matches!(mode, LoopInputMode::Broadcast) &&
                            matches!(ty, ConcreteWireType::Matrix(_))
                        {
                            layout.broadcast = Some(parent);
                            layout.borrowed = true;
                            layout.lazy = false;
                        }
                        Ok(Some((*wire, layout)))
                    })
                    .collect::<Result<Vec<_>, GpuMeasurementError>>()?
                    .into_iter()
                    .flatten()
                    .collect();
                Ok((env, inputs))
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
        backend
            .scope_resource_demand(self.graph, &child_id, &instances, scratch_columns, false)
            .map_err(|error| GpuMeasurementError(error.to_string()))
    }

    /// Filter the current and enclosing value caches with the exact liveness
    /// used by production GPU inventory. In particular, borrowed child inputs
    /// survive their last wire use, aliases extend the original owner, and
    /// unfinished sibling outputs do not exist yet, while previously issued
    /// siblings retain this node's outputs. The result is not an eligible
    /// slot inventory: future/materialized owners still need explicit bindings.
    pub fn gpu_retained_scopes(&self) -> Result<Vec<GpuRetainedScope<'_>>, GpuMeasurementError> {
        let frames = self
            .ancestors
            .iter()
            .flat_map(|parent| {
                std::iter::once((
                    &parent.scope,
                    parent.node,
                    &parent.bindings,
                    &parent.values,
                    parent.active_instance,
                    false,
                ))
                .chain(parent.siblings.iter().map(move |sibling| {
                    (
                        &parent.scope,
                        parent.node,
                        &sibling.bindings,
                        &sibling.values,
                        sibling.index,
                        sibling.index < parent.active_instance,
                    )
                }))
            })
            .chain(std::iter::once((
                self.scope,
                self.node,
                self.bindings,
                self.values,
                self.active_instance,
                false,
            )))
            .chain(self.siblings.iter().map(|sibling| {
                (
                    self.scope,
                    self.node,
                    &sibling.bindings,
                    &sibling.values,
                    sibling.index,
                    sibling.index < self.active_instance,
                )
            }))
            .collect::<Vec<_>>();
        frames
            .par_iter()
            .map(|&(scope, node, bindings, values, instance, issued)| {
                let optimizer = if *scope == FrozenGraphScopeId::Root {
                    inventory_plan(self.graph, false)
                } else {
                    InventoryPlan::default()
                };
                let liveness = owner_liveness(self.graph, scope, bindings, &optimizer, false)
                    .map_err(|_| {
                        GpuMeasurementError(format!(
                            "invalid GPU owner liveness at {scope:?} node {node:?}"
                        ))
                    })?;
                let values = values
                    .iter()
                    .filter(|(wire, _)| {
                        (wire.node.0 < node.0 || (issued && wire.node == node)) &&
                            liveness.output_end(wire.node.0 as usize, wire.port.0 as usize) >=
                                node.0 as usize
                    })
                    .map(|(&wire, layout)| GpuRetainedValue {
                        wire,
                        layout,
                        alias: liveness.alias_source(wire).copied(),
                    })
                    .collect();
                let progress = if issued {
                    mxx_runtime::backend::poly_gpu::GpuScopeProgress::Issued(node.0 as usize)
                } else {
                    mxx_runtime::backend::poly_gpu::GpuScopeProgress::Before(node.0 as usize)
                };
                Ok(GpuRetainedScope { scope, node, bindings, values, instance, progress })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataflow::{ScopeAdmissionState, ValueStorage};
    use mxx_dsl::{DslContext, Family, Ring, parallel};
    use mxx_ir_core::{
        node::NodeKind,
        types::{ConcreteWireType, Port},
    };
    use std::collections::BTreeMap;

    #[test]
    fn test_gpu_retained_scopes_drop_dead_packed_members_and_preserve_aliases() {
        fn layout(ty: &ConcreteWireType) -> ValueLayout {
            match ty {
                ConcreteWireType::IndexedFamily { count, element } => ValueLayout::Family {
                    count: *count,
                    members: vec![layout(element)],
                    overrides: BTreeMap::new(),
                    replication: None,
                    representation: crate::dataflow::FamilyRepresentation::Members,
                },
                _ => ValueLayout::Leaf(ty.clone(), ValueStorage::Device, None),
            }
        }
        let ring = Ring::new(257, 16);
        let packed =
            Family::pack(vec![ring.uniform_residue((1, 1)), ring.uniform_residue((1, 1))]).unwrap();
        let selected = packed.at(0);
        let output = parallel(2, |_| Ok(-selected.clone())).unwrap();
        let graph = DslContext::new("retained-admission-values")
            .output("output", output)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let root = FrozenGraphScopeId::Root;
        let scope = graph.source.scope(&root).unwrap();
        let checked = graph.root_scope();
        let (position, handle) = checked
            .execution_order
            .iter()
            .enumerate()
            .find(|(_, handle)| matches!(handle.kind(), NodeKind::ParallelLoop(_)))
            .unwrap();
        let node = NodeId(position as u64);
        let NodeKind::ParallelLoop(spec) = handle.kind() else { unreachable!() };
        let arguments = scope.arguments(handle).unwrap();
        let values = checked.wire_types.iter().map(|(&wire, ty)| (wire, layout(ty))).collect();
        let child_id = graph.source.child_scope_id(&root, node).unwrap();
        let child = graph.source.scope(&child_id).unwrap();
        let retained =
            mxx_runtime::executor::retained_loop_output_ports(&graph, &root, node, child.outputs());
        let request = LoopAdmissionRequest {
            graph: &graph,
            scope: &root,
            node,
            bindings: &graph.bindings,
            total_count: 2,
            first_index: 0,
            inputs: &[],
            siblings: &[],
            active_instance: 0,
            broadcasts: &Default::default(),
            arguments: &arguments,
            input_modes: &spec.input_modes,
            values: &values,
            retained_outputs: &retained,
            ancestors: &[],
        };
        let frames = request.gpu_retained_scopes().unwrap();
        assert_eq!(frames.len(), 1);
        let (get_position, getter) = checked
            .execution_order
            .iter()
            .enumerate()
            .find(|(_, handle)| matches!(handle.kind(), NodeKind::FamilyGetStatic { .. }))
            .unwrap();
        let family = scope.arguments(getter).unwrap()[0];
        let members = scope.arguments(scope.node(family.node).unwrap()).unwrap();
        let getter = WireRef { node: NodeId(get_position as u64), port: Port(0) };
        let live = &frames[0].values;
        assert!(live.iter().any(|entry| entry.wire == members[0]));
        assert!(!live.iter().any(|entry| entry.wire == members[1]));
        assert!(live.iter().any(|entry| entry.wire == getter && entry.alias == Some(members[0])));
        assert!(live.iter().all(|entry| entry.wire.node.0 < node.0), "future outputs do not exist");

        // An input remains borrowed by the whole child map even at the end of
        // that scope, after the input's direct last consumer has completed.
        let env = graph.bindings.child(&spec.bindings, Some((spec.index_slot, 0))).unwrap();
        let child_values = graph
            .scope(&child_id)
            .unwrap()
            .wire_types
            .iter()
            .map(|(&wire, ty)| (wire, layout(ty)))
            .collect();
        let parents = [ScopeAdmissionState {
            scope: root.clone(),
            node,
            bindings: graph.bindings.clone(),
            values: values.clone(),
            inputs: Vec::new(),
            siblings: Vec::new(),
            active_instance: 0,
        }];
        let request = LoopAdmissionRequest {
            graph: &graph,
            scope: &child_id,
            node: NodeId(graph.scope(&child_id).unwrap().execution_order.len() as u64),
            bindings: &env,
            total_count: 1,
            first_index: 0,
            inputs: &[],
            siblings: &[],
            active_instance: 0,
            broadcasts: &Default::default(),
            arguments: &[],
            input_modes: &[],
            values: &child_values,
            retained_outputs: &[],
            ancestors: &parents,
        };
        let frames = request.gpu_retained_scopes().unwrap();
        assert_eq!(frames.len(), 2);
        assert_eq!(*frames[0].scope, root);
        assert_eq!(*frames[1].scope, child_id);
        assert!(frames[1].values.iter().any(|entry| entry.wire == child.inputs()[0]));

        let output = child.outputs()[0];
        let siblings = [0, 2].map(|index| crate::dataflow::ScopeSiblingState {
            index,
            bindings: env.clone(),
            inputs: Vec::new(),
            values: child_values.clone(),
        });
        let request = LoopAdmissionRequest {
            node: output.node,
            active_instance: 1,
            siblings: &siblings,
            ..request
        };
        let frames = request.gpu_retained_scopes().unwrap();
        assert_eq!(frames.len(), 4);
        for frame in frames.iter().filter(|frame| *frame.scope == child_id) {
            assert_eq!(frame.values.iter().any(|entry| entry.wire == output), frame.instance == 0);
            assert_eq!(
                matches!(
                    frame.progress,
                    mxx_runtime::backend::poly_gpu::GpuScopeProgress::Issued(_)
                ),
                frame.instance == 0
            );
            assert!(frame.values.iter().any(|entry| entry.wire == child.inputs()[0]));
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_symbolic_inputs_use_production_resource_lowering_without_backing() {
        use crate::dataflow::LayoutOwner;
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::{
                GpuGraphAdmissionGuard, GpuPreparedSlotKind, GpuPreparedSlotSnapshot,
            },
            poly::{
                PolyParams,
                dcrt::{
                    gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
                    params::DCRTPolyParams,
                },
            },
        };
        use mxx_runtime::backend::poly_gpu::{GpuMatrixInputFragment, gpu_backend_on};
        use std::sync::Arc;
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let columns = std::env::var("MXX_PRIMITIVE_TEST_MATRIX_SIZE")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(3);
        let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
        let device = detected_gpu_device_ids()[0];
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
        let graph = DslContext::new("symbolic-production-resource-demand")
            .output(
                "result",
                (ring.input("a", (2, columns)) + ring.input("b", (2, columns))).transpose(),
            )
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let scope = FrozenGraphScopeId::Root;
        let wires = graph
            .root_scope()
            .execution_order
            .iter()
            .enumerate()
            .filter(|(_, node)| matches!(node.kind(), NodeKind::Input { .. }))
            .map(|(index, _)| WireRef { node: NodeId(index as u64), port: Port(0) })
            .collect::<Vec<_>>();
        let ty = &graph.root_scope().wire_types[&wires[0]];
        let fragments: Arc<[GpuMatrixInputFragment]> = vec![GpuMatrixInputFragment {
            device,
            context: params.context_identity(),
            level: params.crt_depth() - 1,
            start: 0,
            end: columns,
            evaluation: false,
        }]
        .into();
        let value = |id| {
            ValueLayout::Leaf(
                ty.clone(),
                ValueStorage::Device,
                Some(LayoutOwner { id, members: BTreeMap::new(), component: None }),
            )
        };
        let bindings = [0, 1]
            .into_iter()
            .map(|id| {
                let mut source =
                    mxx_runtime::backend::poly_gpu::GpuInventoryValue::new(vec![0, columns]);
                source.fragments = Some(fragments.clone());
                (LayoutOwner { id, members: BTreeMap::new(), component: None }, source)
            })
            .collect();
        let first = value(0).gpu_inventory(&bindings).unwrap();
        assert!(first.owner.is_none(), "symbolic identities must never impersonate native IDs");
        assert!(first.symbolic_owner.is_some());
        assert!(value(0).gpu_inventory(&BTreeMap::new()).is_err());
        let packed = |members| ValueLayout::Family {
            count: 2,
            members,
            overrides: BTreeMap::new(),
            replication: None,
            representation: crate::dataflow::FamilyRepresentation::Members,
        };
        let shared = packed(vec![value(0), value(0)]).gpu_inventory(&bindings).unwrap();
        let distinct = packed(vec![value(0), value(1)]).gpu_inventory(&bindings).unwrap();
        assert!(shared.borrowed && shared.packed_family && shared.symbolic_owner.is_some());
        assert!(distinct.borrowed && distinct.packed_family && distinct.symbolic_owner.is_none());
        let cold = ValueLayout::Leaf(ty.clone(), ValueStorage::Artifact, None);
        let mixed = packed(vec![value(0), cold.clone()]).gpu_inventory(&bindings).unwrap();
        assert!(mixed.lazy && mixed.packed_family && !mixed.borrowed);
        let descriptor = ValueLayout::Family {
            count: 1_000_000,
            members: vec![cold],
            overrides: BTreeMap::new(),
            replication: None,
            representation: crate::dataflow::FamilyRepresentation::ArtifactDescriptor,
        }
        .gpu_inventory(&BTreeMap::new())
        .unwrap();
        assert!(descriptor.lazy && !descriptor.packed_family && !descriptor.borrowed);

        let mut backend = gpu_backend_on([params.clone()], [device]);
        // No GPU matrix or prepared storage exists in this fixture. Native
        // allocation/dispatch is sealed while lowering and typed planning run.
        let guard = GpuGraphAdmissionGuard::new(vec![params.clone()]).unwrap();
        let mut matrix_counts = Vec::new();
        for id in [0, 1] {
            let inputs = BTreeMap::from([
                (wires[0], first.clone()),
                (wires[1], value(id).gpu_inventory(&bindings).unwrap()),
            ]);
            let demand = backend
                .scope_resource_demand(
                    &graph,
                    &scope,
                    &[(graph.bindings.clone(), inputs)],
                    1,
                    false,
                )
                .unwrap();
            let mut matrix_count = 0;
            for (_, context) in demand.contexts.values() {
                let claims = context.claims(&params);
                matrix_count += claims
                    .iter()
                    .filter(|claim| claim.kind() == GpuPreparedSlotKind::Matrix)
                    .count();
                let slots = GpuPreparedSlotSnapshot::plan(&params, &claims)
                    .unwrap()
                    .into_iter()
                    .map(|slot| (slot, true))
                    .collect::<Vec<_>>();
                let input_inventory =
                    GpuContextInventory { parameters: params.clone(), slots: slots.clone() };
                let (source, preparation) = input_inventory
                    .matrix_input_plan(&first, (2, columns), 0..columns, 0, true, &[])
                    .unwrap()
                    .unwrap();
                assert!(matches!(
                    source,
                    mxx_runtime::backend::poly_gpu::GpuMatrixInputSource::Fragment { .. }
                ));
                assert_eq!(preparation.len(), 1);
                assert!(!preparation[0].0.evaluation, "normalization starts in the source format");
                // A failed multi-preparation transaction must not consume the
                // slot needed by a subsequent smaller candidate.
                let one_slot = GpuContextInventory {
                    parameters: params.clone(),
                    slots: slots
                        .iter()
                        .map(|&(slot, _)| {
                            let id = slot.identity();
                            (
                                slot,
                                (id.storage_id(), id.slot_id(), id.slot_index()) ==
                                    preparation[0].1.slot_key(),
                            )
                        })
                        .collect(),
                };
                let layout = preparation[0].0;
                assert!(
                    mxx_runtime::backend::poly_gpu::GpuMatrixInputLayout::assign(
                        [layout; 2],
                        &params,
                        &one_slot.slots,
                    )
                    .unwrap()
                    .is_none()
                );
                assert_eq!(
                    one_slot
                        .matrix_input_plan(&first, (2, columns), 0..columns, 0, true, &[],)
                        .unwrap()
                        .unwrap()
                        .1[0]
                        .1
                        .slot_key(),
                    preparation[0].1.slot_key(),
                );
                let prepared = preparation.iter().map(|(layout, _)| *layout).collect::<Vec<_>>();
                assert!(
                    input_inventory
                        .matrix_input_plan(&first, (2, columns), 0..columns, 0, true, &prepared)
                        .unwrap()
                        .unwrap()
                        .1
                        .is_empty(),
                    "the same broadcast owner's preparation is reused"
                );
                let mut unavailable = input_inventory.clone();
                unavailable.slots.iter_mut().for_each(|(_, eligible)| *eligible = false);
                assert!(
                    unavailable
                        .matrix_input_plan(&first, (2, columns), 0..columns, 0, true, &[])
                        .unwrap()
                        .is_none()
                );
                let mut unresolved = first.clone();
                unresolved.fragments = None;
                assert!(
                    input_inventory
                        .matrix_input_plan(&unresolved, (2, columns), 0..columns, 0, true, &[])
                        .is_err()
                );
                let assigned = GpuPreparedSlotSnapshot::assign(&params, &slots, &claims)
                    .unwrap()
                    .into_iter()
                    .collect::<Option<Vec<_>>>()
                    .unwrap();
                assert!(
                    assigned.iter().all(|request| request.slot_key().0 == 0),
                    "all requirements fit unbacked planned slots; no native reservation was created"
                );
                // The complete native placement planner consumes descriptors
                // while GPU allocation and dispatch remain sealed.
                use mxx_runtime::{
                    backend::poly_gpu::{
                        GpuColumnShard, GpuMatrixDescriptor, GpuMatrixFragmentDescriptor,
                        GpuMatrixPlacementInvocation, GpuMatrixSlotContext,
                    },
                    gpu_invocation::GpuNodeOperation,
                    gpu_memory::{GpuColumnInventory, GpuDeviceAdmission},
                };
                let add = graph
                    .root_scope()
                    .execution_order
                    .iter()
                    .enumerate()
                    .find(|(_, node)| matches!(node.kind(), NodeKind::MatrixBinary { .. }))
                    .map(|(index, _)| NodeId(index as u64))
                    .unwrap();
                let node = GpuNodeOperation::new(&graph, &scope, add, &graph.bindings).unwrap();
                let descriptor = |owner| GpuMatrixDescriptor {
                    id: owner,
                    rows: 2,
                    columns,
                    input_layout: fragments.clone(),
                    shards: fragments
                        .iter()
                        .map(|fragment| GpuColumnShard {
                            device_id: fragment.device,
                            global_column_start: fragment.start,
                            value: GpuMatrixFragmentDescriptor {
                                parameters: params.clone(),
                                level: fragment.level,
                                columns: fragment.end - fragment.start,
                                evaluation: fragment.evaluation,
                            },
                        })
                        .collect(),
                };
                let invocation = GpuMatrixPlacementInvocation::new(
                    &backend,
                    &node,
                    vec![Some(descriptor(0)), Some(descriptor(id as u64))],
                )
                .unwrap()
                .unwrap();
                let selected = backend
                    .plan_matrix_layouts(
                        &[invocation],
                        &vec![GpuMatrixSlotContext {
                            device: 0,
                            context: params.context_identity(),
                            slots: slots.clone(),
                        }],
                        &Default::default(),
                    )
                    .unwrap();
                let admissions = [GpuDeviceAdmission {
                    budget_bytes: 0,
                    baseline_bytes: 0,
                    allocation_bytes: 0,
                    physical_baseline_bytes: 0,
                    observed_resident_bytes: 0,
                    observed_physical_bytes: 0,
                }];
                let capacity =
                    [GpuContextInventory { parameters: params.clone(), slots: slots.clone() }];
                let capacity = super::GpuColumnContextInventory { contexts: &capacity };
                let cap = columns.div_ceil(2);
                let invocations =
                    selected.fit(&admissions, cap, std::slice::from_ref(&capacity)).unwrap();
                assert_eq!(
                    invocations[0].plan.schedule.local_job_counts(),
                    &[columns.div_ceil(cap)]
                );
                assert_eq!(
                    invocations[0].input_owners,
                    if id == 0 { vec![0, 0] } else { vec![0, 1] }
                );
                assert_eq!(selected.outputs(0).len(), 1);
                assert_eq!(selected.outputs(0)[0].2.columns(), columns);
                let estimated =
                    crate::gpu::estimate_admitted_batch(&invocations, &[false], |class| {
                        // Deterministic synthetic timing units test accounting,
                        // not GPU speed: the fitted ranges must cover each column once.
                        Ok(crate::NodeMeasurement {
                            work_seconds: class
                                .jobs
                                .iter()
                                .map(|(_, job)| (job.end - job.start) as f64)
                                .sum(),
                            ..Default::default()
                        })
                    })
                    .unwrap();
                assert_eq!(estimated.work_seconds, columns as f64);
                assert_eq!(estimated.independent_wave_count, columns.div_ceil(cap));

                assert!(capacity.fits_requests(0, 0, &assigned).unwrap());
                let mut conflicting = assigned.clone();
                conflicting.push(assigned[0]);
                assert!(
                    !capacity.fits_requests(0, 0, &conflicting).unwrap(),
                    "simultaneous roles cannot share a slot"
                );
                let position = graph.root_scope().execution_order.len();
                assert!(
                    !context
                        .retained_claim_indices(None, |_| {
                            mxx_runtime::backend::poly_gpu::GpuScopeProgress::Before(position)
                        })
                        .is_empty(),
                    "the returned matrix remains owned"
                );
                let mut retained_inventory =
                    GpuContextInventory { parameters: params.clone(), slots: slots.clone() };
                retained_inventory.exclude_retained(context, &assigned, |_| {
                    mxx_runtime::backend::poly_gpu::GpuScopeProgress::Before(position)
                });
                assert!(
                    !super::GpuColumnContextInventory {
                        contexts: std::slice::from_ref(&retained_inventory),
                    }
                    .fits_requests(0, 0, &assigned)
                    .unwrap(),
                    "the shared fit inventory must preserve retained-owner exclusions"
                );
                assert!(
                    GpuPreparedSlotSnapshot::assign(&params, &retained_inventory.slots, &claims)
                        .unwrap()
                        .iter()
                        .any(Option::is_none),
                    "a new full invocation cannot reuse the still-retained output"
                );
                assert!(
                    slots.iter().all(|(_, eligible)| *eligible),
                    "the original scenario is unchanged"
                );
            }
            matrix_counts.push(matrix_count);
        }
        assert_eq!(
            matrix_counts[1],
            matrix_counts[0] + 1,
            "equal-shaped independent inputs need one more normalization owner than a shared input"
        );
        // Negation preserves the artifact's payload format. Without reading
        // that payload, its returned format is unresolved and must stay unbound.
        let unknown_graph = DslContext::new("unresolved-artifact-format")
            .output("result", -ring.input("cold", (2, columns)))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let unknown_scope = unknown_graph.source.scope(&scope).unwrap();
        let unknown_inputs = unknown_graph
            .root_scope()
            .execution_order
            .iter()
            .enumerate()
            .filter(|(_, handle)| matches!(handle.kind(), NodeKind::Input { .. }))
            .map(|(position, _)| {
                let wire = WireRef { node: NodeId(position as u64), port: Port(0) };
                let value = ValueLayout::Leaf(
                    unknown_graph.root_scope().wire_types[&wire].clone(),
                    ValueStorage::Artifact,
                    None,
                );
                (wire, value.gpu_inventory(&BTreeMap::new()).unwrap())
            })
            .collect();
        let unknown = backend
            .scope_resource_demand(
                &unknown_graph,
                &scope,
                &[(unknown_graph.bindings.clone(), unknown_inputs)],
                1,
                false,
            )
            .unwrap();
        let unknown_output = &unknown.value_layouts[0][&unknown_scope.outputs()[0]];
        assert!(unknown_output.fragments.is_none(), "payload format must not be fabricated");
        assert_eq!(unknown_output.cuts, vec![0, columns]);
        let unknown_wire = unknown_scope.outputs()[0];
        let future = ValueLayout::Leaf(
            unknown_graph.root_scope().wire_types[&unknown_wire].clone(),
            ValueStorage::Device,
            Some(LayoutOwner { id: 88, members: BTreeMap::new(), component: None }),
        );
        let mut future_inventory = GpuAdmissionInventory {
            contexts: BTreeMap::new(),
            sources: BTreeMap::new(),
            column_cap: 1,
        };
        future_inventory.bind_sources(
            &BTreeMap::from([(unknown_wire, future.clone())]),
            &unknown.value_layouts[0],
        );
        let source_bound = future.gpu_inventory(&future_inventory.sources).unwrap();
        assert!(source_bound.borrowed && !source_bound.lazy && source_bound.fragments.is_none());
        let mut consumer = |source: mxx_runtime::backend::poly_gpu::GpuInventoryValue| {
            backend
                .scope_resource_demand(
                    &graph,
                    &scope,
                    &[(
                        graph.bindings.clone(),
                        BTreeMap::from([(wires[0], source.clone()), (wires[1], source)]),
                    )],
                    1,
                    false,
                )
                .unwrap()
        };
        let containing = consumer(source_bound.clone());
        for evaluation in [false, true] {
            let mut concrete = source_bound.clone();
            concrete.fragments = Some(
                vec![GpuMatrixInputFragment {
                    device,
                    context: params.context_identity(),
                    level: params.crt_depth() - 1,
                    start: 0,
                    end: columns,
                    evaluation,
                }]
                .into(),
            );
            let actual = consumer(concrete);
            for (key, (_, demand)) in actual.contexts {
                let slots = GpuPreparedSlotSnapshot::plan(
                    &params,
                    &containing.contexts[&key].1.claims(&params),
                )
                .unwrap()
                .into_iter()
                .map(|slot| (slot, true))
                .collect::<Vec<_>>();
                assert!(
                    GpuPreparedSlotSnapshot::assign(&params, &slots, &demand.claims(&params))
                        .unwrap()
                        .iter()
                        .all(Option::is_some),
                    "the unknown-format bound must cover each concrete format"
                );
            }
        }

        let nested_input = ring.input("nested", (2, columns));
        let nested = parallel(2, |_| {
            let family = parallel(2, |_| Ok(-nested_input.clone()))?;
            let selected = family.at(0);
            Ok(Family::pack(vec![selected.clone(), selected])?.at(1))
        })
        .unwrap();
        let nested = DslContext::new("nested-value-claims")
            .output("result", nested)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let (position, handle) = nested
            .root_scope()
            .execution_order
            .iter()
            .enumerate()
            .find(|(_, node)| matches!(node.kind(), NodeKind::ParallelLoop(_)))
            .unwrap();
        let NodeKind::ParallelLoop(outer) = handle.kind() else { unreachable!() };
        let outer_id = nested
            .source
            .child_scope_id(&FrozenGraphScopeId::Root, NodeId(position as u64))
            .unwrap();
        let outer_scope = nested.source.scope(&outer_id).unwrap();
        let instances = (0..2)
            .map(|index| {
                let inputs = outer_scope
                    .inputs()
                    .iter()
                    .map(|&wire| {
                        let mut value =
                            mxx_runtime::backend::poly_gpu::GpuInventoryValue::new(vec![
                                0, columns,
                            ]);
                        value.borrowed = true;
                        value.symbolic_owner = Some([83; 32]);
                        (wire, value)
                    })
                    .collect();
                (
                    nested
                        .bindings
                        .child(&outer.bindings, Some((outer.index_slot, index)))
                        .unwrap(),
                    inputs,
                )
            })
            .collect::<Vec<_>>();
        let resources =
            backend.scope_resource_demand(&nested, &outer_id, &instances, columns, false).unwrap();
        let inner_family = nested
            .scope(&outer_id)
            .unwrap()
            .execution_order
            .iter()
            .enumerate()
            .find(|(_, node)| matches!(node.kind(), NodeKind::ParallelLoop(_)))
            .map(|(index, _)| WireRef { node: NodeId(index as u64), port: Port(0) })
            .unwrap();
        let packed = nested
            .scope(&outer_id)
            .unwrap()
            .execution_order
            .iter()
            .enumerate()
            .find(|(_, node)| matches!(node.kind(), NodeKind::FamilyPack { .. }))
            .map(|(index, _)| WireRef { node: NodeId(index as u64), port: Port(0) })
            .unwrap();
        for (_, context) in resources.contexts.values() {
            let progress = mxx_runtime::backend::poly_gpu::GpuScopeProgress::Before(usize::MAX);
            let first = context.value_claim_indices(0, inner_family, &[0], progress);
            let second = context.value_claim_indices(0, inner_family, &[1], progress);
            assert_eq!(first.len(), 1);
            assert_eq!(second.len(), 1);
            assert!(first.is_disjoint(&second));
            assert_eq!(
                context.value_claim_indices(0, outer_scope.outputs()[0], &[], progress),
                first
            );
            assert_eq!(context.value_claim_indices(0, packed, &[0], progress), first);
            assert_eq!(
                context.value_claim_indices(0, packed, &[1], progress),
                first,
                "both occurrences of a repeated pack member preserve its owner"
            );
            let sibling = context.value_claim_indices(1, outer_scope.outputs()[0], &[], progress);
            assert_eq!(sibling.len(), 1);
            assert!(first.is_disjoint(&sibling));
        }

        // The loop adapter uses the production sibling fold: a cold broadcast
        // imports once for the whole wave, while each body owns its real result.
        let input = ring.input("shared", (2, columns));
        let loop_graph = DslContext::new("symbolic-sibling-resource-demand")
            .output("result", parallel(3, |_| Ok(input.clone() + input.clone())).unwrap())
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let root = FrozenGraphScopeId::Root;
        let root_scope = loop_graph.source.scope(&root).unwrap();
        let (position, loop_handle) = loop_graph
            .root_scope()
            .execution_order
            .iter()
            .enumerate()
            .find(|(_, handle)| matches!(handle.kind(), NodeKind::ParallelLoop(_)))
            .unwrap();
        let node = NodeId(position as u64);
        let NodeKind::ParallelLoop(spec) = loop_handle.kind() else { unreachable!() };
        let arguments = root_scope.arguments(loop_handle).unwrap();
        let values = arguments
            .iter()
            .map(|wire| {
                (
                    *wire,
                    ValueLayout::Leaf(
                        loop_graph.root_scope().wire_types[wire].clone(),
                        ValueStorage::Artifact,
                        None,
                    ),
                )
            })
            .collect();
        let child_id = loop_graph.source.child_scope_id(&root, node).unwrap();
        let retained = mxx_runtime::executor::retained_loop_output_ports(
            &loop_graph,
            &root,
            node,
            loop_graph.source.scope(&child_id).unwrap().outputs(),
        );
        let request = LoopAdmissionRequest {
            graph: &loop_graph,
            scope: &root,
            node,
            bindings: &loop_graph.bindings,
            total_count: 3,
            first_index: 0,
            inputs: &[],
            siblings: &[],
            active_instance: 0,
            broadcasts: &Default::default(),
            arguments: &arguments,
            input_modes: &spec.input_modes,
            values: &values,
            retained_outputs: &retained,
            ancestors: &[],
        };
        let counts = (1..=3)
            .map(|wave| {
                request
                    .gpu_wave_demand(&mut backend, 0, wave, 1, &BTreeMap::new())
                    .unwrap()
                    .contexts
                    .values()
                    .flat_map(|(_, context)| context.claims(&params))
                    .filter(|claim| claim.kind() == GpuPreparedSlotKind::Matrix)
                    .count()
            })
            .collect::<Vec<_>>();
        assert!(counts[1] > counts[0], "another body retains its own result");
        assert!(counts[1] < 2 * counts[0], "broadcast import is shared, not repeated per body");
        assert_eq!(counts[2] - counts[1], counts[1] - counts[0]);
        let demands = request.gpu_wave_demand(&mut backend, 0, 2, 1, &BTreeMap::new()).unwrap();
        let mut inventory = GpuAdmissionInventory {
            contexts: demands
                .contexts
                .into_iter()
                .map(|(key, (_, demand))| {
                    let slots = GpuPreparedSlotSnapshot::plan(&params, &demand.claims(&params))
                        .unwrap()
                        .into_iter()
                        .map(|slot| (slot, true))
                        .collect();
                    (key, vec![GpuContextInventory { parameters: params.clone(), slots }])
                })
                .collect(),
            sources: BTreeMap::new(),
            column_cap: columns,
        };
        let accepted = request.gpu_admit_wave(&mut backend, 0, 3, &inventory).unwrap();
        assert_eq!(accepted.instances, 2, "three bodies cannot use two bodies' typed slots");
        assert_eq!(accepted.value_layouts.len(), accepted.instances);
        let result_wire = loop_graph.source.scope(&child_id).unwrap().outputs()[0];
        let output_claims = [0, 1].map(|instance| {
            accepted.value_claims(
                instance,
                result_wire,
                &[],
                mxx_runtime::backend::poly_gpu::GpuScopeProgress::Before(usize::MAX),
            )
        });
        for (key, contexts) in &output_claims[0] {
            for (first, second) in contexts.iter().zip(&output_claims[1][key]) {
                assert_eq!(first.len(), 1);
                assert_eq!(second.len(), 1);
                assert_ne!(
                    first[0].slot_key(),
                    second[0].slot_key(),
                    "sibling outputs occupy distinct planned capacity"
                );
            }
        }
        let progress_demand = request
            .gpu_wave_demand(&mut backend, 0, 2, accepted.column_cap, &inventory.sources)
            .unwrap();
        for (_, context) in progress_demand.contexts.values() {
            use mxx_runtime::backend::poly_gpu::GpuScopeProgress::{Before, Issued};
            let position = result_wire.node.0 as usize;
            assert!(
                !context.retained_claim_indices(None, |_| Before(0)).is_empty(),
                "cold broadcasts are already placed before the first body node"
            );
            let before = context.retained_claim_indices(None, |index| {
                assert!(index < 2);
                Before(position)
            });
            let mixed = context.retained_claim_indices(None, |index| {
                assert!(index < 2);
                if index == 0 { Issued(position) } else { Before(position) }
            });
            let issued = context.retained_claim_indices(None, |index| {
                assert!(index < 2);
                Issued(position)
            });
            assert!(before.is_subset(&mixed) && mixed.is_subset(&issued));
            assert!(
                before.len() < mixed.len() && mixed.len() < issued.len(),
                "one earlier sibling's output is owned before the other sibling produces its output"
            );
        }

        for layouts in &accepted.value_layouts {
            let output = &layouts[&result_wire];
            assert_eq!(output.cuts, vec![0, columns]);
            assert!(
                output.fragments.is_none(),
                "a cold artifact with unknown CRT level cannot prove an output fragment level"
            );
        }

        // Bind each accepted body's returned value for a later consumer.
        // Shared layout geometry must not equate the two logical owners.
        for (instance, layouts) in accepted.value_layouts.iter().enumerate() {
            let output_value = ValueLayout::Leaf(
                loop_graph.scope(&child_id).unwrap().wire_types[&result_wire].clone(),
                ValueStorage::Device,
                Some(LayoutOwner { id: 90 + instance, members: BTreeMap::new(), component: None }),
            );
            inventory.bind_sources(&BTreeMap::from([(result_wire, output_value.clone())]), layouts);
            let bound = output_value.gpu_inventory(&inventory.sources).unwrap();
            assert!(bound.borrowed && bound.symbolic_owner.is_some());
            assert_eq!(bound.fragments, layouts[&result_wire].fragments);
        }
        assert_eq!(inventory.sources.len(), 2);
        assert!(
            accepted
                .assignments
                .values()
                .flatten()
                .flatten()
                .all(|request| request.slot_key().0 == 0),
            "fit creates no native reservation"
        );
        assert_eq!(request.gpu_admit_wave(&mut backend, 0, 1, &inventory).unwrap().instances, 1);
        assert_eq!(request.gpu_admit_wave(&mut backend, 2, 3, &inventory).unwrap().instances, 1);
        for context in inventory.contexts.values_mut().flatten() {
            for (slot, eligible) in &mut context.slots {
                assert!(*eligible, "candidate search does not mutate eligibility");
                if slot.identity().kind() == GpuPreparedSlotKind::Matrix {
                    *eligible = false; // All matrix slots now belong to retained owners.
                }
            }
        }
        assert!(request.gpu_admit_wave(&mut backend, 0, 3, &inventory).is_err());
        inventory.contexts.clear();
        assert_eq!(request.gpu_admit_wave(&mut backend, 3, 3, &inventory).unwrap().instances, 0);
        assert!(
            request
                .gpu_wave_demand(&mut backend, 0, 0, 1, &BTreeMap::new())
                .unwrap()
                .contexts
                .is_empty()
        );
        drop(guard);
    }
}

impl ValueLayout {
    /// Bind scalar or family inputs to the production inventory's metadata.
    /// Resident source bounds are explicit CPU bindings, never inferred from
    /// type/shape. Definite fragments are optional when the shared resource
    /// bound covers all unresolved source alternatives. Packed members use the same
    /// containing-envelope fold as native inputs. Artifact-family descriptors stay cold and
    /// unexpanded. The result describes demand; it proves neither native ownership nor fit.
    pub fn gpu_inventory(
        &self,
        bindings: &std::collections::BTreeMap<
            crate::dataflow::LayoutOwner,
            mxx_runtime::backend::poly_gpu::GpuInventoryValue,
        >,
    ) -> Result<mxx_runtime::backend::poly_gpu::GpuInventoryValue, GpuMeasurementError> {
        use crate::dataflow::{FamilyRepresentation, ValueStorage};
        use mxx_runtime::backend::poly_gpu::GpuInventoryValue;
        use sha2::{Digest, Sha256};
        if let Self::Family { count, members, overrides, replication, representation } = self {
            if *count == 0 {
                return Ok(GpuInventoryValue::default());
            }
            if *representation == FamilyRepresentation::ArtifactDescriptor {
                let mut value = self.member(0).gpu_inventory(bindings)?;
                value.packed_family = false;
                return Ok(value);
            }
            // A uniform cold family or a captured shared owner has one
            // containing layout regardless of count. Only resident body-local
            // owners can have different explicit fragment bindings per member.
            let uniform = members.len() == 1 &&
                match &members[0] {
                    Self::Leaf(_, storage, owner) => {
                        *storage != ValueStorage::Device ||
                            owner.as_ref().is_some_and(|owner| {
                                replication.as_ref().is_none_or(|replication| {
                                    !replication.owners.contains(&owner.id)
                                })
                            })
                    }
                    _ => false,
                };
            if uniform {
                let mut representatives = overrides.values().collect::<Vec<_>>();
                if *count > overrides.len() {
                    representatives.push(&members[0]);
                }
                let mut value = representatives
                    .into_par_iter()
                    .map(|value| value.gpu_inventory(bindings))
                    .try_reduce(GpuInventoryValue::default, |mut all, member| {
                        all.include_alternative(member);
                        Ok(all)
                    })?;
                value.packed_family = true;
                return Ok(value);
            }
            let mut value = (0..*count)
                .into_par_iter()
                .map(|index| self.member(index).gpu_inventory(bindings))
                .try_reduce(GpuInventoryValue::default, |mut all, member| {
                    all.include_alternative(member);
                    Ok(all)
                })?;
            value.packed_family = true;
            return Ok(value);
        }
        let Self::Leaf(ty, storage, owner) = self else { unreachable!() };
        let ty = ty
            .matrix_type()
            .ok_or_else(|| GpuMeasurementError("GPU inventory needs a matrix-like value".into()))?;
        let mut value = if *storage == ValueStorage::Device {
            let owner = owner.as_ref().ok_or_else(|| {
                GpuMeasurementError("GPU input materialization identity is unresolved".into())
            })?;
            let mut value = bindings.get(owner).cloned().ok_or_else(|| {
                GpuMeasurementError("GPU input source resource bound is not bound".into())
            })?;
            value.symbolic_owner = Some(
                Sha256::digest(
                    mxx_ir_core::encoding::canonical_json(owner)
                        .map_err(|error| GpuMeasurementError(error.to_string()))?,
                )
                .into(),
            );
            value.broadcast = None;
            value.packed_family = false;
            value.borrowed = true;
            value
        } else {
            GpuInventoryValue::new(vec![0, ty.columns])
        };
        value.lazy = *storage != ValueStorage::Device;
        Ok(value)
    }
}
