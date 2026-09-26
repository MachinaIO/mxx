//! Graph-owned scratch memory.
//!
//! Lowering gives a scratch matrix only its layout. When the plan's Graph
//! regions are compiled, each scratch allocation is either owned by the Graph
//! or allocated once for the plan. A Graph-owned allocation is created by a
//! memory node right before its first top-level operation is emitted and freed
//! by one right after its last. The memory nodes of a region form a chain in
//! operation order, so a free precedes every later allocation and CUDA may
//! place that allocation in the freed memory. Inside a parallel loop, whose
//! lanes are independent, every allocation follows only the chain at the
//! loop's start and no memory is reused: a free joining several readers must
//! not gate later work, or CUDA runs the lanes one after another. The chain
//! joins every memory node of the loop when it ends, so memory freed by the
//! W lanes is reused after them, and the loop's scratch peak grows with W. An
//! operation with a conditional body counts as one top-level operation,
//! because CUDA forbids memory nodes inside such bodies: scratch used by a
//! loop body lives across the whole loop.

use crate::{
    backend::{BoundStorage, GpuResidentValue, poly_gpu::GpuDcrtBackend},
    gpu_execution_plan::{
        CompiledGpuOp, KernelArg, PhysicalEncoding, PhysicalValue, PhysicalValueId, StorageRef,
    },
    gpu_physical_lowering::{PhysicalFrame, matrix_physical_value},
    matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
    poly::dcrt::gpu::{GpuNativeGraphBuilder, GpuNativeGraphError},
};
use mxx_ir_core::types::ConcreteMatrixType;
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

/// Owner of a scratch matrix whose memory is chosen when the plan's Graph
/// regions are compiled.
pub(crate) struct GpuDeferredScratch {
    ty: ConcreteMatrixType,
    encoding: PhysicalEncoding,
    device: i32,
}

/// Owner marker of a Graph-owned allocation. The memory itself is allocated
/// and freed by memory nodes of the compiled Graph.
pub(crate) struct GpuGraphScratch;

/// Whether `bound` has no plan-owned memory: its allocation is deferred to
/// compilation or owned by a compiled Graph.
pub(crate) fn is_graph_managed(bound: &BoundStorage) -> bool {
    bound.owner.is::<GpuDeferredScratch>() || bound.owner.is::<GpuGraphScratch>()
}

/// A distinct, never dereferenced address for each deferred allocation, so
/// storage identities stay unique until real memory is bound.
fn placeholder_address() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1 << 62);
    NEXT.fetch_add(1 << 40, Ordering::Relaxed)
}

/// Describe a full scratch matrix without allocating it.
pub(crate) fn deferred_scratch_matrix(
    backend: &GpuDcrtBackend,
    device: i32,
    ty: &ConcreteMatrixType,
    encoding: PhysicalEncoding,
    storage: StorageRef,
) -> Result<(PhysicalValue, Arc<GpuResidentValue>), String> {
    let params = backend.physical_matrix_parameters(ty, device)?;
    let (limbs, data_bytes) = GpuDCRTPolyMatrix::binding_layout(&params, ty.rows, ty.columns)
        .map_err(|error| error.to_string())?;
    let first = limbs.first().ok_or("GPU scratch layout has no CRT limb")?;
    let (limb_device, bytes_per_poly) = (first.physical_device, first.poly_stride_bytes);
    let physical = matrix_physical_value(
        ty,
        encoding.clone(),
        storage,
        &limbs,
        limb_device,
        data_bytes,
        bytes_per_poly,
        0,
    )?;
    let bound = BoundStorage {
        device: limb_device,
        address: placeholder_address(),
        bytes: u64::try_from(data_bytes).map_err(|_| "GPU scratch size exceeds u64")?,
        owner: Arc::new(GpuDeferredScratch { ty: ty.clone(), encoding, device }),
    };
    let resident = GpuResidentValue::new(
        Arc::new(physical.clone()),
        BTreeMap::from([(storage, bound)]),
        Box::new([]),
    )
    .map_err(str::to_owned)?;
    Ok((physical, Arc::new(resident)))
}

struct GraphAllocation {
    members: Vec<PhysicalValueId>,
    device: i32,
    bytes: usize,
    references: BTreeSet<usize>,
    /// Whether a later region's launch, not this Graph, frees it.
    outlives_region: bool,
    /// Address, and the start and allocation-node token of the region whose
    /// Graph allocates it.
    allocated: Option<(u64, usize, u32)>,
}

/// Graph-owned scratch allocations of one plan, in operation order.
pub(crate) struct GraphScratchPlan {
    allocations: Vec<GraphAllocation>,
    allocate_before: BTreeMap<usize, Vec<usize>>,
    free_after: BTreeMap<usize, Vec<usize>>,
    referenced_by: BTreeMap<usize, Vec<usize>>,
    /// Operation ranges of the parallel loops, ordered by start and, for
    /// loops starting together, outermost first.
    loops: Vec<Range<usize>>,
    /// Memory nodes the next memory node outside a parallel loop follows.
    chain: Vec<u32>,
    /// Start of the region whose builder issued the `chain` tokens.
    chain_region: Option<usize>,
    /// The outermost open parallel loop: its end and the memory nodes
    /// created inside it so far. Its nodes follow `chain`, frozen meanwhile.
    open_loop: Option<(usize, Vec<u32>)>,
    /// First loop of `loops` not reached yet.
    next_loop: usize,
}

fn visit(op: &CompiledGpuOp, each: &mut dyn FnMut(PhysicalValueId, bool, i32)) {
    for argument in op.arguments.iter() {
        if let KernelArg::Value(id) | KernelArg::OptionalValue(Some(id)) = argument {
            each(*id, false, op.device);
        }
    }
    for output in op.outputs.iter() {
        each(*output, true, op.device);
    }
    for inner in op.body.iter().flatten() {
        visit(inner, each);
    }
}

/// Swap every storage of `owners` that views a replaced allocation.
fn rebind(
    owners: &mut BTreeMap<PhysicalValueId, Arc<GpuResidentValue>>,
    replacements: &BTreeMap<*const (), BoundStorage>,
) -> Result<(), String> {
    for owner in owners.values_mut() {
        let replaced = owner
            .storages()
            .filter_map(|(_, bound)| {
                let pointer = Arc::as_ptr(&bound.owner).cast::<()>();
                replacements.get(&pointer).map(|new| (pointer, new.clone()))
            })
            .collect::<std::collections::HashMap<_, _>>();
        if replaced.is_empty() {
            continue;
        }
        if let Some(rebound) = owner.rebound(&replaced, owner.ready_events())? {
            *owner = Arc::new(rebound);
        }
    }
    Ok(())
}

/// Decide every deferred scratch allocation of `frame` for the Graph regions
/// starting at `starts`, a sorted list whose first entry is 0. Scratch
/// written before it is read, used on one device, and live within one region,
/// or on the home device across regions that each run once per execution,
/// becomes Graph-owned. Every other deferred allocation is allocated for the
/// plan here.
pub(crate) fn plan_graph_scratch(
    backend: &GpuDcrtBackend,
    frame: &mut PhysicalFrame,
    starts: &[usize],
) -> Result<GraphScratchPlan, String> {
    struct Group {
        placeholder: BoundStorage,
        members: Vec<PhysicalValueId>,
        eligible: bool,
        references: BTreeSet<usize>,
        first_write: Option<usize>,
    }
    let mut groups = BTreeMap::<*const (), Group>::new();
    let mut group_of = BTreeMap::<PhysicalValueId, *const ()>::new();
    let mut collect = |id: PhysicalValueId, owner: &GpuResidentValue, eligible: bool| {
        for (_, bound) in owner.storages() {
            if !bound.owner.is::<GpuDeferredScratch>() {
                continue;
            }
            let pointer = Arc::as_ptr(&bound.owner).cast::<()>();
            let group = groups.entry(pointer).or_insert_with(|| Group {
                placeholder: bound.clone(),
                members: Vec::new(),
                eligible: true,
                references: BTreeSet::new(),
                first_write: None,
            });
            group.members.push(id);
            group.eligible &= eligible;
            group_of.insert(id, pointer);
        }
    };
    for (&id, owner) in &frame.owners {
        let physical = &frame.program.values[id.0 as usize];
        let eligible = physical.ty.matrix_type().is_some() &&
            matches!(
                physical.encodings.as_ref(),
                [PhysicalEncoding::FullCoeff] | [PhysicalEncoding::FullEval]
            ) &&
            owner.storages().count() == 1 &&
            !frame.scratch_protected.contains(&id);
        collect(id, owner, eligible);
    }
    // Wave bindings are swapped in by the host between launches.
    for wave in &frame.waves {
        for (&id, owner) in &wave.owner_bindings {
            collect(id, owner, false);
        }
    }
    for (index, op) in frame.program.operations.iter().enumerate() {
        visit(op, &mut |id, written, device| {
            let Some(group) = group_of.get(&id).and_then(|pointer| groups.get_mut(pointer)) else {
                return;
            };
            group.references.insert(index);
            if written {
                group.first_write = Some(group.first_write.map_or(index, |first| first.min(index)));
            }
            group.eligible &= device == group.placeholder.device;
        });
    }
    let region_of = |index: usize| starts.partition_point(|start| *start <= index) - 1;
    let replayed_bodies = frame
        .waves
        .iter()
        .map(|wave| (wave.body_start as usize, wave.body_end as usize))
        .chain(
            frame
                .external_io_loops
                .iter()
                .map(|body| (body.body_start as usize, body.body_end as usize)),
        )
        .collect::<Vec<_>>();
    let replayed = |region: usize| {
        let start = starts[region];
        replayed_bodies
            .iter()
            .any(|&(body_start, body_end)| body_start <= start && start < body_end)
    };
    let mut loops = frame
        .parallel_lanes
        .iter()
        .filter_map(|lanes| Some(lanes.first()?.start as usize..lanes.last()?.end as usize))
        .filter(|range| !range.is_empty())
        .collect::<Vec<_>>();
    loops.sort_by_key(|range| (range.start, std::cmp::Reverse(range.end)));
    let mut plan = GraphScratchPlan {
        allocations: Vec::new(),
        allocate_before: BTreeMap::new(),
        free_after: BTreeMap::new(),
        referenced_by: BTreeMap::new(),
        loops,
        chain: Vec::new(),
        chain_region: None,
        open_loop: None,
        next_loop: 0,
    };
    let mut persistent = BTreeMap::<*const (), BoundStorage>::new();
    for (pointer, group) in groups {
        let graph_owned = match (group.references.first(), group.references.last()) {
            (Some(&first), Some(&last)) if group.eligible => {
                let (first_region, last_region) = (region_of(first), region_of(last));
                // An allocation that outlives its region is freed by the host
                // after the last region using it launches, on the home stream;
                // it must not live across a replayed region's relaunches.
                group.first_write.is_some_and(|write| write <= first) &&
                    (first_region == last_region ||
                        (!replayed(first_region) &&
                            !replayed(last_region) &&
                            group.placeholder.device == frame.device))
            }
            _ => false,
        };
        if graph_owned {
            let index = plan.allocations.len();
            let first = *group.references.first().expect("graph scratch is referenced");
            let last = *group.references.last().expect("graph scratch is referenced");
            plan.allocate_before.entry(first).or_default().push(index);
            plan.free_after.entry(last).or_default().push(index);
            for &reference in &group.references {
                plan.referenced_by.entry(reference).or_default().push(index);
            }
            plan.allocations.push(GraphAllocation {
                members: group.members,
                device: group.placeholder.device,
                bytes: usize::try_from(group.placeholder.bytes)
                    .map_err(|_| "GPU scratch size exceeds usize")?,
                outlives_region: region_of(first) != region_of(last),
                references: group.references,
                allocated: None,
            });
            continue;
        }
        let deferred = group
            .placeholder
            .owner
            .downcast_ref::<GpuDeferredScratch>()
            .ok_or("GPU deferred scratch owner changed type")?;
        let native = backend.allocate_physical_matrix(
            &deferred.ty,
            deferred.device,
            deferred.encoding.clone(),
        )?;
        let bound = BoundStorage::from_matrix_data(native, 0)?;
        if bound.device != group.placeholder.device || bound.bytes != group.placeholder.bytes {
            return Err("GPU persistent scratch layout differs from its plan".into());
        }
        persistent.insert(pointer, bound);
    }
    rebind(&mut frame.owners, &persistent)?;
    for wave in &mut frame.waves {
        rebind(&mut wave.owner_bindings, &persistent)?;
    }
    Ok(plan)
}

/// CUDA backs Graph allocations in 2 MiB granules.
const GRAPH_ALLOCATION_GRANULE: u64 = 2 << 20;

impl GraphScratchPlan {
    /// Peak Graph-owned bytes live at once on each device: every allocation
    /// counts, rounded up to CUDA's granule, from its first use until its
    /// memory can be reused, which is after its last use, or after the
    /// outermost parallel loop that contains its last use. The memory-node
    /// chain lets CUDA reuse memory freed earlier, so no point of the schedule
    /// needs more; a launch that could not fit would leave its reservation
    /// behind, so plans are admitted on this peak.
    pub(crate) fn peak_bytes(&self) -> BTreeMap<i32, u64> {
        let reusable_after = |last: usize| {
            self.loops
                .iter()
                .filter(|range| range.contains(&last))
                .map(|range| range.end)
                .max()
                .unwrap_or(last + 1)
        };
        let mut events = BTreeMap::<(i32, usize), i128>::new();
        for allocation in &self.allocations {
            let (Some(&first), Some(&last)) =
                (allocation.references.first(), allocation.references.last())
            else {
                continue;
            };
            let bytes = (allocation.bytes as u64).div_ceil(GRAPH_ALLOCATION_GRANULE) *
                GRAPH_ALLOCATION_GRANULE;
            *events.entry((allocation.device, first)).or_default() += i128::from(bytes);
            *events.entry((allocation.device, reusable_after(last))).or_default() -=
                i128::from(bytes);
        }
        let mut live = BTreeMap::<i32, i128>::new();
        let mut peaks = BTreeMap::<i32, u64>::new();
        for ((device, _), delta) in events {
            let current = live.entry(device).or_default();
            *current += delta;
            let peak = peaks.entry(device).or_default();
            *peak = (*peak).max(u64::try_from(*current).unwrap_or(0));
        }
        peaks
    }

    /// Move the memory-node chain to top-level operation `index` of the
    /// region starting at `start`: a new region starts an empty chain, and a
    /// parallel loop's end joins every memory node created inside it.
    fn enter_operation(&mut self, start: usize, index: usize) {
        if self.chain_region != Some(start) {
            self.chain_region = Some(start);
            self.chain.clear();
            if let Some((_, created)) = &mut self.open_loop {
                created.clear();
            }
        }
        if let Some((end, _)) = &self.open_loop &&
            index >= *end
        {
            let (_, created) = self.open_loop.take().expect("the loop is open");
            self.chain.extend(created);
        }
        while let Some(range) = self.loops.get(self.next_loop) &&
            range.start <= index
        {
            if self.open_loop.is_none() && index < range.end {
                self.open_loop = Some((range.end, Vec::new()));
            }
            self.next_loop += 1;
        }
    }

    /// Record a memory node created after `self.chain`.
    fn push_memory_node(&mut self, token: u32) {
        match &mut self.open_loop {
            Some((_, created)) => created.push(token),
            None => self.chain = vec![token],
        }
    }

    /// Allocate the scratch whose first use is top-level operation `index`,
    /// bind it into `frame`, and return the allocation tokens that operation
    /// must follow. The address of an allocation a later region frees is
    /// appended to `outliving`.
    pub(crate) fn before_operation(
        &mut self,
        builder: &mut GpuNativeGraphBuilder,
        frame: &mut PhysicalFrame,
        start: usize,
        index: usize,
        outliving: &mut Vec<u64>,
    ) -> Result<Vec<u32>, GpuNativeGraphError> {
        let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
        self.enter_operation(start, index);
        for allocation in self.allocate_before.get(&index).cloned().unwrap_or_default() {
            let (device, bytes) =
                (self.allocations[allocation].device, self.allocations[allocation].bytes);
            let (token, address) = builder.add_memory_alloc(device, bytes, &self.chain)?;
            self.push_memory_node(token);
            let allocation = &mut self.allocations[allocation];
            allocation.allocated = Some((address, start, token));
            if allocation.outlives_region {
                outliving.push(address);
            }
            let bound = BoundStorage {
                device: allocation.device,
                address,
                bytes: allocation.bytes as u64,
                owner: Arc::new(GpuGraphScratch),
            };
            for id in &allocation.members {
                let owner = frame
                    .owners
                    .get_mut(id)
                    .ok_or_else(|| invalid("Graph scratch owner is missing"))?;
                let (_, current) = owner
                    .storages()
                    .next()
                    .ok_or_else(|| invalid("Graph scratch has no storage"))?;
                let replaced = std::collections::HashMap::from([(
                    Arc::as_ptr(&current.owner).cast::<()>(),
                    bound.clone(),
                )]);
                if let Some(rebound) =
                    owner.rebound(&replaced, owner.ready_events()).map_err(invalid)?
                {
                    *owner = Arc::new(rebound);
                }
            }
        }
        Ok(self
            .referenced_by
            .get(&index)
            .into_iter()
            .flatten()
            .filter_map(|&allocation| match self.allocations[allocation].allocated {
                // Allocations of an earlier region's Graph precede this Graph.
                Some((_, region, token)) if region == start => Some(token),
                _ => None,
            })
            .collect())
    }

    /// Free the scratch whose last use is top-level operation `index` of the
    /// region starting at `start`, after every emitted operation of that
    /// region that uses it. An allocation of an earlier region's Graph is not
    /// freed by a node: CUDA cannot update a Graph with conditional nodes that
    /// frees another Graph's allocation, so its address is appended to
    /// `host_frees` for the host to free after launching this region.
    pub(crate) fn after_operation(
        &mut self,
        builder: &mut GpuNativeGraphBuilder,
        start: usize,
        index: usize,
        host_frees: &mut Vec<u64>,
    ) -> Result<(), GpuNativeGraphError> {
        for allocation in self.free_after.get(&index).cloned().unwrap_or_default() {
            let allocation = &self.allocations[allocation];
            let (address, region, _) = allocation.allocated.ok_or_else(|| {
                GpuNativeGraphError::Native("Graph scratch is freed before its allocation".into())
            })?;
            if region != start {
                host_frees.push(address);
                continue;
            }
            let readers = allocation
                .references
                .range(start..=index)
                .map(|&reference| (reference - start) as u32)
                .collect::<Vec<_>>();
            let token = builder.add_memory_free(address, &readers, &self.chain)?;
            self.push_memory_node(token);
        }
        Ok(())
    }
}
