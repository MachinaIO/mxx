//! Scratch memory of the compiled Graph regions.
//!
//! Lowering gives a scratch matrix only its layout. When the plan's Graph
//! regions are compiled, each scratch allocation either takes a byte range of
//! a region scratch pool or is allocated once for the plan. Pool scratch is
//! placed as CUDA would place a Graph allocation made by a memory node right
//! before its first top-level operation and freed by one right after its
//! last, but the nodes are empty barrier nodes and the memory is one buffer
//! per device the plan owns: CUDA reserves address space for each Graph's
//! allocations for the Graph's lifetime and caps the total at about twice the
//! device memory, which a plan of many regions would exceed. The barriers of
//! each device's pool in a region form a chain in operation order, so a free
//! precedes every later allocation from that pool and that allocation may take
//! the freed bytes; the pools of different devices order nothing between them,
//! so each device's work waits only for its own memory. Inside a parallel
//! loop, whose lanes are independent, every allocation follows only the chain
//! at the loop's start, and memory freed inside the loop is reused only by
//! the lane that freed it: a free joining several readers must not gate
//! another lane, or CUDA runs the lanes one after another. Scratch whose uses
//! all lie in one lane, and in no inner loop, follows its lane's own chain
//! and reuses what that lane freed; other scratch of the loop reuses nothing.
//! The chain joins every barrier of the loop when it ends, so memory freed by
//! the W lanes is reused after them. An
//! operation with a conditional body counts as one top-level operation,
//! because CUDA forbids memory nodes inside such bodies: scratch used by a
//! loop body lives across the whole loop. Regions launch in order, so memory
//! freed in one region is free in every later one, and scratch used across
//! regions is freed by a barrier in the region of its last use, after that
//! region's uses.
//! Device-only carry scratch and imports local to a nested host loop retain
//! their actual region lifetimes inside a wave. Only the wave's own import
//! templates extend across that entire wave to protect overlapping read-ahead;
//! unrelated nested carries must not accumulate with the number of stages.
//! Region scratch and host-bound arena ranges normally share the same device buffer.
//! When that pool exceeds the configured device memory fraction, long-lived arena
//! ranges may opt into GPU-addressable pinned host backing with
//! `MXX_GPU_HOST_BACKING=1` (disabled by default), admitted against available
//! host RAM. Arithmetic stays on the GPU, with unchanged barriers and replay spans.
//! Mapped bytes are reported separately and never counted as device residency.
//! They can reuse bytes across disjoint region lifetimes, while overlapping
//! regions keep them separate; wave read-ahead lifetimes remain extended above.
//!
//! Other scratch lives in memory the plan owns. A scratch value that is written before it is read
//! shares one arena per device with the others, in the arena of the device that holds it even when
//! operations on other devices use it (a copy between devices, for example), since every region
//! joins all devices before the next one launches: it takes a byte range for
//! its live span, from the region of its first use to the region of its last, widened across any
//! replayed host-loop boundary it crosses and every wave it overlaps. A value initialized and
//! consumed wholly within a host-loop body starts fresh on replay; wave staging can overlap
//! replays and keeps the whole wave live. Spans are
//! regions because operations of one region may run concurrently while regions launch in order.
//! Values whose spans share no region share bytes, so a chain of loops, each reading the running
//! result of the one before, needs about two of them at once. Any other scratch value, together
//! with inputs, outputs, and nonplaced imports, is an allocation of its own.
//! Wave-bound members stay in persistent storage, but distinct source allocations
//! use independent ranges and may share bytes outside their widened live regions.

#[path = "gpu_graph_memory/gpu_host_backing.rs"]
mod gpu_host_backing;

use gpu_host_backing::ResidencySpan;

#[path = "gpu_graph_memory/gpu_body_scratch.rs"]
mod gpu_body_scratch;
#[path = "gpu_graph_memory/gpu_replay_spans.rs"]
mod gpu_replay_spans;

use crate::{
    backend::{BoundStorage, GpuResidentValue, poly_gpu::GpuDcrtBackend},
    gpu_execution_plan::{
        CompiledGpuOp, GpuNativePrimitive, KernelArg, PhysicalEncoding, PhysicalValue,
        PhysicalValueId, StorageRef,
    },
    gpu_physical_lowering::{PhysicalFrame, matrix_physical_value},
    matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
    poly::dcrt::gpu::{GpuDeviceBuffer, GpuNativeGraphBuilder, GpuNativeGraphError},
};
use mxx_ir_core::types::{ConcreteMatrixType, ConcreteWireType};
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
    /// A full matrix, a compact bounded matrix, or an address-bound byte allocation.
    ty: ConcreteWireType,
    device: i32,
    /// A later wave's member of a parallel loop output, which the host binds
    /// for its wave: it holds memory from its loop to the family's last read.
    replayed: bool,
}

/// Whether `bound` has no memory yet: its allocation is deferred to
/// compilation.
pub(crate) fn is_graph_managed(bound: &BoundStorage) -> bool {
    bound.owner.is::<GpuDeferredScratch>()
}

/// Reserve disjoint synthetic address ranges in creation order. They are never
/// dereferenced. Advancing by the actual aligned size avoids exhausting the
/// counter after millions of small values, as a fixed 1 TiB stride did.
fn reserve_placeholder_address(next: &AtomicU64, bytes: u64) -> Result<u64, String> {
    let stride = bytes
        .max(1)
        .checked_add(255)
        .map(|size| size & !255)
        .ok_or("GPU deferred address size overflows")?;
    next.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |address| address.checked_add(stride))
        .map_err(|_| "GPU deferred address space exhausted".to_owned())
}

fn placeholder_address(bytes: u64) -> Result<u64, String> {
    static NEXT: AtomicU64 = AtomicU64::new(1 << 62);
    reserve_placeholder_address(&NEXT, bytes)
}

/// Storage of `bytes` for a scratch value of type `ty` on `device`, whose
/// memory is chosen when the plan's Graph regions are compiled.
pub(crate) fn deferred_scratch_storage(
    device: i32,
    ty: ConcreteWireType,
    bytes: u64,
) -> Result<BoundStorage, String> {
    Ok(BoundStorage {
        device,
        address: placeholder_address(bytes)?,
        bytes,
        owner: Arc::new(GpuDeferredScratch { ty, device, replayed: false }),
    })
}

/// Storage of `bytes` for a later wave's member of a parallel loop output of
/// type `ty` on `device`. Its memory is a range of the plan's arena, live from
/// the loop to the last read of the member.
pub(crate) fn deferred_replay_storage(
    device: i32,
    ty: ConcreteWireType,
    bytes: u64,
) -> Result<BoundStorage, String> {
    Ok(BoundStorage {
        device,
        address: placeholder_address(bytes)?,
        bytes,
        owner: Arc::new(GpuDeferredScratch { ty, device, replayed: true }),
    })
}

/// Describe a compact scratch value of the bounded type `ty`, whose payload
/// `layout` holds, without allocating it.
pub(crate) fn deferred_scratch_compact(
    device: i32,
    ty: ConcreteWireType,
    payload_bytes: usize,
    storage: StorageRef,
    physical: PhysicalValue,
) -> Result<Arc<GpuResidentValue>, String> {
    let bound = BoundStorage {
        device,
        address: placeholder_address(
            u64::try_from(payload_bytes).map_err(|_| "GPU compact scratch size exceeds u64")?,
        )?,
        bytes: u64::try_from(payload_bytes).map_err(|_| "GPU compact scratch size exceeds u64")?,
        owner: Arc::new(GpuDeferredScratch { ty, device, replayed: false }),
    };
    GpuResidentValue::new(Arc::new(physical), BTreeMap::from([(storage, bound)]), Box::new([]))
        .map(Arc::new)
        .map_err(str::to_owned)
}

/// Describe a full scratch matrix without allocating it. Its polynomials
/// are packed back to back: a native matrix reserves a second half per
/// polynomial as workspace for its own transforms, which the direct plan's
/// operations never use, so scratch needs only the coefficient half.
pub(crate) fn deferred_scratch_matrix(
    backend: &GpuDcrtBackend,
    device: i32,
    ty: &ConcreteMatrixType,
    encoding: PhysicalEncoding,
    storage: StorageRef,
    metadata: Option<&mut crate::gpu_execution_plan::PhysicalMetadataPool>,
) -> Result<(PhysicalValue, Arc<GpuResidentValue>), String> {
    let params = backend.physical_matrix_parameters(ty, device)?;
    let (native_limbs, _) = GpuDCRTPolyMatrix::binding_layout(&params, ty.rows, ty.columns)
        .map_err(|error| error.to_string())?;
    let first = native_limbs.first().ok_or("GPU scratch layout has no CRT limb")?;
    let (limb_device, bytes_per_poly) = (first.physical_device, first.scratch_offset_bytes);
    let data_bytes = ty
        .rows
        .checked_mul(ty.columns)
        .and_then(|polys| polys.checked_mul(bytes_per_poly))
        .ok_or("GPU scratch size overflows")?;
    let limbs = native_limbs
        .iter()
        .map(|limb| crate::matrix::gpu_dcrt_poly::GpuMatrixBindingLimb {
            poly_stride_bytes: bytes_per_poly,
            row_stride_bytes: ty.columns * bytes_per_poly,
            data_bytes,
            data_address: limb.byte_offset as u64,
            ..limb.clone()
        })
        .collect::<Vec<_>>();
    let physical = matrix_physical_value(
        ty,
        encoding.clone(),
        storage,
        &limbs,
        limb_device,
        data_bytes,
        bytes_per_poly,
        0,
        metadata,
    )?;
    let bound = BoundStorage {
        device: limb_device,
        address: placeholder_address(
            u64::try_from(data_bytes).map_err(|_| "GPU scratch size exceeds u64")?,
        )?,
        bytes: u64::try_from(data_bytes).map_err(|_| "GPU scratch size exceeds u64")?,
        owner: Arc::new(GpuDeferredScratch {
            ty: ConcreteWireType::Matrix(ty.clone()),
            device,
            replayed: false,
        }),
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
    /// Identity of the deferred allocation, even in a view of several allocations.
    placeholder: *const (),
    members: Vec<PhysicalValueId>,
    device: i32,
    bytes: usize,
    references: BTreeSet<usize>,
    /// Byte offset in its device's region scratch pool.
    offset: u64,
    /// The lane of its outermost parallel loop that holds every use, when
    /// no inner loop holds any: it reuses memory that lane freed.
    lane: Option<usize>,
    /// Address, and the start and allocation-barrier token of the region
    /// whose Graph first uses it.
    allocated: Option<(u64, usize, u32)>,
}

/// Exact memory-barrier recipe for rebuilding one native region without
/// mutating the already-bound physical frame or serializing independent lanes.
#[derive(Default)]
pub(crate) struct RegionScratchRecipe {
    before: BTreeMap<usize, Vec<(Vec<u32>, Vec<u32>, u32)>>,
    after: BTreeMap<usize, Vec<(Vec<u32>, Vec<u32>, u32)>>,
    pending: BTreeMap<usize, Vec<u32>>,
}

impl RegionScratchRecipe {
    fn emit(
        barriers: Option<&Vec<(Vec<u32>, Vec<u32>, u32)>>,
        builder: &mut GpuNativeGraphBuilder,
    ) -> Result<(), GpuNativeGraphError> {
        for (operations, dependencies, expected) in barriers.into_iter().flatten() {
            let token = builder.add_memory_barrier(operations, dependencies)?;
            if token != *expected {
                return Err(GpuNativeGraphError::Native(
                    "rebuilt region changed scratch barrier identity".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn before(
        &self,
        builder: &mut GpuNativeGraphBuilder,
        index: usize,
    ) -> Result<(), GpuNativeGraphError> {
        Self::emit(self.before.get(&index), builder)?;
        if let Some(tokens) = self.pending.get(&index) {
            builder.set_pending_memory_dependencies(tokens)?;
        }
        Ok(())
    }

    pub(crate) fn after(
        &self,
        builder: &mut GpuNativeGraphBuilder,
        index: usize,
    ) -> Result<(), GpuNativeGraphError> {
        Self::emit(self.after.get(&index), builder)
    }
}

/// Region scratch allocations of one plan, in operation order.
pub(crate) struct GraphScratchPlan {
    allocations: Vec<GraphAllocation>,
    allocate_before: BTreeMap<usize, Vec<usize>>,
    free_after: BTreeMap<usize, Vec<usize>>,
    referenced_by: BTreeMap<usize, Vec<usize>>,
    /// Operation ranges of the parallel loops, ordered by start and, for
    /// loops starting together, outermost first.
    loops: Vec<Range<usize>>,
    /// Operation ranges of each loop's lanes.
    lanes: BTreeMap<(usize, usize), Vec<Range<usize>>>,
    /// Per device, the memory nodes its pool's next memory node outside a
    /// parallel loop follows.
    chain: BTreeMap<i32, Vec<u32>>,
    /// Start of the region whose builder issued the `chain` tokens.
    chain_region: Option<usize>,
    /// The outermost open parallel loop: its end, and per device the memory
    /// nodes created inside it so far and each lane's chain. Its nodes follow
    /// `chain`, frozen meanwhile, and a lane's own nodes also follow its chain.
    open_loop: Option<(usize, BTreeMap<i32, Vec<u32>>, Vec<BTreeMap<i32, Vec<u32>>>)>,
    /// First loop of `loops` not reached yet.
    next_loop: usize,
    /// Each device's region scratch pool.
    pools: BTreeMap<i32, Arc<GpuDeviceBuffer>>,
}

/// Visit direct and resource-table operands of one operation. Import readiness
/// and scratch liveness must agree about these otherwise invisible reads.
pub(super) fn visit_operation(
    implementations: &crate::gpu_execution_plan::GpuImplementationRegistry,
    indexed_tables: &[crate::gpu_physical_lowering::IndexedMatrixTableReplay],
    hash_resources: &BTreeMap<u32, crate::gpu_execution_plan::GpuHashResourceSpec>,
    op: &CompiledGpuOp,
    each: &mut dyn FnMut(PhysicalValueId, bool),
) -> Result<(), String> {
    for argument in op.arguments.iter() {
        if let KernelArg::Value(id) | KernelArg::OptionalValue(Some(id)) = argument {
            each(*id, false);
        }
    }
    for output in op.outputs.iter() {
        each(*output, true);
    }
    // Pointer-table operands are reads too. In particular, later waves'
    // family members may use the arena even when scratch_protected excludes
    // their views from region pools. Omitting these reads lets another wave
    // reuse a member's bytes before its last dynamic selection.
    let mut indexed_read = |resource: u32| -> Result<(), String> {
        let table = indexed_tables
            .get(resource as usize)
            .filter(|table| table.resource_id == resource)
            .ok_or("GPU scratch read has no indexed matrix table")?;
        for &(id, _) in &table.candidates {
            each(id, false);
        }
        Ok(())
    };
    match implementations.resolve(op.implementation)?.primitive {
        GpuNativePrimitive::MatrixIndexedCopy => {
            let Some(KernelArg::U32(resource)) = op.arguments.first() else {
                return Err("GPU indexed copy has no table resource".into());
            };
            indexed_read(*resource)?;
        }
        GpuNativePrimitive::SubgraphKernel => {
            for operand in op.arguments.get(6..).unwrap_or_default().chunks_exact(4) {
                if let [KernelArg::U32(1), _, KernelArg::U32(resource), _] = operand {
                    indexed_read(*resource)?;
                }
            }
        }
        GpuNativePrimitive::HashSample => {
            let Some(KernelArg::U32(resource)) = op.arguments.first() else {
                return Err("GPU hash sample has no tag resource".into());
            };
            let spec =
                hash_resources.get(resource).ok_or("GPU scratch read has no hash tag resource")?;
            for &(id, _, _) in spec.operands.iter() {
                each(id, false);
            }
        }
        _ => {}
    }
    Ok(())
}

fn visit_node(
    frame: &PhysicalFrame,
    op: &CompiledGpuOp,
    each: &mut dyn FnMut(PhysicalValueId, bool),
) -> Result<(), String> {
    visit_operation(
        &frame.program.implementations,
        &frame.indexed_tables,
        &frame.hash_resources,
        op,
        each,
    )
}

fn visit(
    frame: &PhysicalFrame,
    op: &CompiledGpuOp,
    each: &mut dyn FnMut(PhysicalValueId, bool),
) -> Result<(), String> {
    visit_operation(
        &frame.program.implementations,
        &frame.indexed_tables,
        &frame.hash_resources,
        op,
        each,
    )?;
    for inner in op.body.as_deref().into_iter().flatten() {
        visit(frame, inner, each)?;
    }
    Ok(())
}

/// `owner` with every storage that views a replaced allocation swapped, or `None` if it views
/// none.
fn rebound(
    owner: &GpuResidentValue,
    replacements: &BTreeMap<*const (), BoundStorage>,
) -> Result<Option<GpuResidentValue>, String> {
    let replaced = owner
        .storages()
        .filter_map(|(_, bound)| {
            let pointer = Arc::as_ptr(&bound.owner).cast::<()>();
            replacements.get(&pointer).map(|new| (pointer, new.clone()))
        })
        .collect::<std::collections::HashMap<_, _>>();
    if replaced.is_empty() {
        return Ok(None);
    }
    owner.rebound(&replaced, owner.ready_events()).map_err(str::to_owned)
}

/// Swap every storage of `owners` that views a replaced allocation.
fn rebind(
    owners: &mut BTreeMap<PhysicalValueId, Arc<GpuResidentValue>>,
    replacements: &BTreeMap<*const (), BoundStorage>,
) -> Result<(), String> {
    for owner in owners.values_mut() {
        rebind_owner(owner, replacements)?;
    }
    Ok(())
}

/// Materialize a replay destination using the same allocation spans as its views.
fn rebind_owner(
    owner: &mut Arc<GpuResidentValue>,
    replacements: &BTreeMap<*const (), BoundStorage>,
) -> Result<(), String> {
    if let Some(rebound) = rebound(owner, replacements)? {
        *owner = Arc::new(rebound);
    }
    Ok(())
}

/// Decide every deferred scratch allocation of `frame` for the Graph regions
/// starting at `starts`, a sorted list whose first entry is 0. Scratch
/// written before it is read and live within one region, or held on the home
/// device across regions that each run once per execution, takes a range of
/// its device's region scratch pool, whichever devices use it. Every other
/// deferred allocation is allocated for the plan here, and the pools are
/// allocated too. Host-bound, body-local imports introduce region boundaries
/// at their request and after their last use, so reserving a whole region does
/// not retain their bytes during unrelated computation before or after them.
#[cfg(test)]
thread_local! {
    /// The minimum shared scratch bytes of the last planned frame on this thread.
    pub(crate) static LAST_MINIMUM_SCRATCH: std::cell::RefCell<u64> =
        const { std::cell::RefCell::new(0) };
}

pub(crate) fn plan_graph_scratch(
    backend: &GpuDcrtBackend,
    frame: &mut PhysicalFrame,
    starts: &mut Vec<usize>,
) -> Result<GraphScratchPlan, String> {
    gpu_body_scratch::coalesce(frame)?;
    struct Group {
        placeholder: BoundStorage,
        members: Vec<PhysicalValueId>,
        eligible: bool,
        /// Whether an arena range may hold this allocation; composite views
        /// are rebound allocation by allocation, including host wave bindings.
        packable: bool,
        references: BTreeSet<usize>,
        first_write: Option<usize>,
        /// Wave-level host uploads can overlap waves through read-ahead or staging.
        wave_import: bool,
        placed_import: bool,
        /// A staging upload for the next iteration overlaps the entire current body.
        ahead_span: Option<(usize, usize)>,
    }
    let mut groups = BTreeMap::<*const (), Group>::new();
    let mut group_of = BTreeMap::<PhysicalValueId, BTreeSet<*const ()>>::new();
    let mut collect = |id: PhysicalValueId,
                       owner: &GpuResidentValue,
                       eligible: bool,
                       packable: bool,
                       host_read: bool| {
        for (_, bound) in owner.storages() {
            let Some(deferred) = bound.owner.downcast_ref::<GpuDeferredScratch>() else {
                continue;
            };
            let pointer = Arc::as_ptr(&bound.owner).cast::<()>();
            let group = groups.entry(pointer).or_insert_with(|| Group {
                placeholder: bound.clone(),
                members: Vec::new(),
                eligible: true,
                packable: true,
                references: BTreeSet::new(),
                first_write: None,
                wave_import: false,
                placed_import: false,
                ahead_span: None,
            });
            group.members.push(id);
            group.eligible &= eligible;
            group.packable &= if deferred.replayed { !host_read } else { packable };
            group_of.entry(id).or_default().insert(pointer);
        }
    };
    for (&id, owner) in frame.owners.iter() {
        let physical = &frame.program.values[id.0 as usize];
        let packable = !frame.scratch_protected.contains(&id);
        // Full matrices, and compact scratch (whose producer writes every byte
        // it leaves meaningful), are bound by address alone.
        let eligible = packable &&
            physical.encodings.iter().all(|encoding| {
                matches!(
                    encoding,
                    PhysicalEncoding::FullCoeff |
                        PhysicalEncoding::FullEval |
                        PhysicalEncoding::CompactCoeff { .. } |
                        PhysicalEncoding::CompactCoeffPerCrtLimb { .. }
                )
            });
        collect(
            id,
            owner,
            eligible,
            packable,
            frame.output_ids.values().any(|&output| output == id),
        );
    }
    // Wave bindings are swapped in by the host between launches. They must
    // remain host-addressable, but their persistent arena ranges can be reused
    // outside their widened replay lifetimes.
    let mut replay_writes = Vec::new();
    for wave in &frame.waves {
        for (&id, owner) in &wave.owner_bindings {
            // Stored families return durable descriptors, not these GPU owners.
            // Their export operations and later GPU reads already define their
            // lifetime. Only a family returned resident must survive execution.
            let host_read = wave.family_members.iter().any(|(name, members)| {
                frame.output_ids.contains_key(name) &&
                    members.iter().any(|&(_, member)| member == id)
            });
            collect(id, owner, false, !frame.scratch_protected.contains(&id), host_read);
            for (_, bound) in owner.storages() {
                if bound.owner.downcast_ref::<GpuDeferredScratch>().is_some_and(|d| d.replayed) {
                    replay_writes
                        .push((Arc::as_ptr(&bound.owner).cast::<()>(), wave.body_start as usize));
                }
            }
        }
        for (_, id, owner) in &wave.nested_owner_bindings {
            // These are alternate destinations installed for inner writers,
            // not payloads returned to the host. Keep their addresses in the
            // arena, with actual reads and replay spans defining their life.
            // Externally retained aliases remain protected when frame.owners
            // above collects their ordinary output or family owner.
            collect(*id, owner, false, !frame.scratch_protected.contains(id), false);
        }
    }
    // Read-ahead has a separate staging allocation with the same logical destination.
    let mut ahead_spans = Vec::new();
    for body in &frame.external_io_loops {
        for import in &body.ahead {
            collect(import.template.destination, &import.staging, false, false, true);
            for (_, bound) in import.staging.storages() {
                ahead_spans.push((
                    Arc::as_ptr(&bound.owner).cast::<()>(),
                    (body.body_start as usize, body.body_end as usize),
                ));
            }
        }
    }
    // Runtime starts iteration k+1's upload before executing iteration k's
    // body. Only staging needs this reservation: its destination is refreshed
    // at body entry and remains reusable after its last current-iteration read.
    // Scope the reservation to the owning loop so later loops can reuse it.
    for (pointer, (start, end)) in ahead_spans {
        if let Some(group) = groups.get_mut(&pointer) {
            group.ahead_span = Some(
                group
                    .ahead_span
                    .map_or((start, end), |(first, last)| (first.min(start), last.max(end))),
            );
        }
    }
    for (index, op) in frame.program.operations.iter().enumerate() {
        // Operations of any device may share one allocation: its barriers
        // order every use, and only copies reach across devices.
        visit(frame, op, &mut |id, written| {
            // Composite views may read several allocations through one value.
            // Track all of them; arena rebinding replaces each allocation independently.
            for pointer in group_of.get(&id).into_iter().flatten() {
                let group = groups.get_mut(pointer).expect("collected scratch group");
                group.references.insert(index);
                if written {
                    group.first_write =
                        Some(group.first_write.map_or(index, |first| first.min(index)));
                }
            }
        })?;
    }
    for (pointer, start) in replay_writes {
        if let Some(group) = groups.get_mut(&pointer) {
            group.references.insert(start);
            group.first_write = Some(group.first_write.map_or(start, |first| first.min(start)));
        }
    }
    // A placed import is written by the host from its load site on: every
    // owner of its destination (a wave's read-ahead twin too) spans from
    // there to the destination's last use. It is never region scratch, which
    // could reuse its bytes inside the region it is uploaded in, and it takes
    // arena bytes although the host binds it: the host writes it only inside
    // that span.
    let placed = {
        use crate::gpu_physical_lowering::ImportDestination::{Placed, PlacedBounded};
        frame
            .external_io_imports
            .iter()
            .chain(frame.external_io_loops.iter().flat_map(|body| &body.imports))
            .filter(|import| matches!(import.upload_owner, Placed { .. } | PlacedBounded { .. }))
            .map(|import| (import.destination, import.request_operation as usize))
            .chain(frame.import_templates.iter().enumerate().filter_map(|(index, import)| {
                // A wave's template is read from its wave group's start (a windowed one from its
                // window start), a root one from the start of execution.
                let in_wave =
                    frame.waves.iter().any(|wave| wave.import_template_indices.contains(&index));
                matches!(import.upload_owner, Placed { .. } | PlacedBounded { .. }).then(|| {
                    let load = match (in_wave, import.window_start) {
                        (true, Some(start)) => start as usize,
                        (true, None) => import.before_operation as usize,
                        (false, _) => 0,
                    };
                    (import.destination, load)
                })
            }))
            .chain(frame.external_io_loops.iter().flat_map(|body| {
                body.ahead.iter().filter_map(move |import| {
                    matches!(import.template.upload_owner, Placed { .. } | PlacedBounded { .. })
                        .then_some((import.template.destination, body.body_start as usize))
                })
            }))
            .collect::<Vec<_>>()
    };
    // A windowed import is written only from its window start, inside its wave body, so it is
    // not reserved for the whole wave.
    let wave_imports = frame
        .waves
        .iter()
        .flat_map(|wave| wave.import_template_indices.iter())
        .filter(|&&index| frame.import_templates[index].window_start.is_none())
        .map(|&index| frame.import_templates[index].destination)
        .collect::<BTreeSet<_>>();
    for (id, load) in placed {
        // Collection indexes every member-to-allocation association, including
        // composite views and wave/read-ahead twins. Reuse that index instead of
        // scanning every group for every placed import.
        let Some(owners) = group_of.get(&id) else { continue };
        let mut references = BTreeSet::from([load]);
        for pointer in owners {
            references.extend(groups[pointer].references.iter().copied());
        }
        for pointer in owners {
            let group = groups.get_mut(pointer).expect("the group exists");
            group.references = references.clone();
            group.first_write = Some(load);
            group.wave_import |= wave_imports.contains(&id);
            group.placed_import = true;
            group.eligible = false;
            group.packable = true;
        }
    }
    let wave_bodies = frame
        .waves
        .iter()
        .map(|wave| (wave.body_start as usize, wave.body_end as usize))
        .collect::<BTreeSet<_>>();
    let replayed_bodies = gpu_replay_spans::repeated_bodies(
        frame
            .waves
            .iter()
            .map(|wave| (wave.body_start as usize, wave.body_end as usize, wave.repetitions))
            .chain(frame.external_io_loops.iter().map(|body| {
                (body.body_start as usize, body.body_end as usize, u128::from(body.count))
            })),
    );
    // A host upload can overlap every operation in its request's region, so
    // its arena reservation must cover that whole region. Splitting at the
    // request and after the last alias read makes that reservation precise
    // without changing the upload's readiness dependency or replay lifetime.
    // Wave read-ahead remains reserved across its complete wave; values that
    // cross replay boundaries likewise retain their conservative lifetimes.
    let import_spans = groups
        .values()
        .filter_map(|group| {
            if !group.placed_import || group.wave_import || !group.packable {
                return None;
            }
            let (&first, &last) = group.references.first().zip(group.references.last())?;
            group.first_write.is_some_and(|write| write <= first).then_some((first, last))
        })
        .collect::<Vec<_>>();
    starts.extend(gpu_replay_spans::import_region_boundaries(
        &replayed_bodies,
        &import_spans,
        frame.program.operations.len(),
    ));
    starts.sort_unstable();
    starts.dedup();
    let region_of = |index: usize| starts.partition_point(|start| *start <= index) - 1;
    let lanes = frame
        .parallel_lanes
        .iter()
        .filter_map(|lanes| {
            // One lane has no inter-lane concurrency to preserve. Its ordinary
            // allocation barriers can reuse bytes throughout nested operations.
            if lanes.len() == 1 {
                return None;
            }
            let range = lanes.first()?.start as usize..lanes.last()?.end as usize;
            let lanes = lanes.iter().map(|lane| lane.start as usize..lane.end as usize);
            (!range.is_empty()).then(|| ((range.start, range.end), lanes.collect::<Vec<_>>()))
        })
        .collect::<BTreeMap<_, _>>();
    let mut loops = lanes.keys().map(|&(start, end)| start..end).collect::<Vec<_>>();
    loops.sort_by_key(|range| (range.start, std::cmp::Reverse(range.end)));
    let mut plan = GraphScratchPlan {
        allocations: Vec::new(),
        allocate_before: BTreeMap::new(),
        free_after: BTreeMap::new(),
        referenced_by: BTreeMap::new(),
        loops,
        lanes,
        chain: BTreeMap::new(),
        chain_region: None,
        open_loop: None,
        next_loop: 0,
        pools: BTreeMap::new(),
    };
    let mut persistent = BTreeMap::<*const (), BoundStorage>::new();
    // Plan-owned scratch written before it is read: device, live span, bytes.
    let mut arena = Vec::<(*const (), i32, usize, usize, u64)>::new();
    let mut arena_members = Vec::new();
    let mut arena_use_spans = Vec::new();
    // Allocation identities are host pointers, not a placement priority. Keep
    // equal-sized candidates in logical-value order so allocator reuse on the
    // host cannot change GPU fragmentation or the chosen concurrency geometry.
    let mut groups = groups.into_iter().collect::<Vec<_>>();
    groups
        .sort_by_key(|(_, group)| (group.members.iter().copied().min(), group.placeholder.address));
    // Batch interval queries: production graphs can contain many allocations
    // and replayed bodies, so a full body scan per allocation is quadratic.
    let group_count = groups.len();
    let spans = groups
        .iter()
        .map(|(_, group)| {
            group
                .references
                .first()
                .zip(group.references.last())
                .map_or((0, 0), |(&first, &last)| (first, last))
        })
        .chain(starts.iter().map(|&start| (start, start)))
        .collect::<Vec<_>>();
    let replay_spans = gpu_replay_spans::replay_span_properties(&replayed_bodies, &spans);
    let replayed = |region: usize| replay_spans[group_count + region].0;
    for (group_index, (pointer, group)) in groups.into_iter().enumerate() {
        let graph_owned = match (group.references.first(), group.references.last()) {
            (Some(&first), Some(&last)) if group.eligible => {
                let (first_region, last_region) = (region_of(first), region_of(last));
                // An allocation that outlives its region keeps its bytes until
                // the region after its last use. It must not live across a
                // replayed region's relaunches, unless every use lies in one
                // replayed body: each replay then runs its whole life again,
                // after the previous replay's has ended. Nor may it end or
                // begin inside a body nested in that one, such as a loop in a
                // wave, which would free or allocate it on that body's first
                // replay only.
                let (one_body, nested) = replay_spans[group_index];
                group.first_write.is_some_and(|write| write <= first) &&
                    (first_region == last_region ||
                        ((one_body || (!replayed(first_region) && !replayed(last_region))) &&
                            nested &&
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
                placeholder: pointer,
                members: group.members,
                device: group.placeholder.device,
                bytes: usize::try_from(group.placeholder.bytes)
                    .map_err(|_| "GPU scratch size exceeds usize")?,
                lane: plan.lane_of(first, last),
                references: group.references,
                offset: 0,
                allocated: None,
            });
            continue;
        }
        let written_first = group
            .first_write
            .is_some_and(|write| group.references.first().is_some_and(|first| write <= *first));
        if group.packable && written_first {
            let (Some(&first), Some(&last)) = (group.references.first(), group.references.last())
            else {
                return Err("GPU plan scratch is not referenced".into());
            };
            let (mut start, mut end) = (first, last);
            if let Some((body_start, body_end)) = group.ahead_span.filter(|(a, b)| a < b) {
                start = start.min(body_start);
                end = end.max(body_end - 1);
            }
            loop {
                // Even one wave can upload/read ahead asynchronously. Preserve
                // that reservation independently of actual body replay counts.
                let bodies =
                    replayed_bodies.iter().chain(wave_bodies.iter().filter(|_| group.wave_import));
                let widened = bodies.fold((start, end), |(start, end), &body| {
                    let (body_start, body_end) = body;
                    // Body-local device scratch starts fresh on replay. Wave
                    // imports can overlap the next wave through read-ahead or
                    // staging and must cover the whole wave. Extending device
                    // temporaries or inner host-loop imports instead retains all
                    // nested carries, despite their ordered device uses.
                    if body_start < body_end &&
                        start < body_end &&
                        body_start <= end &&
                        (!(body_start <= start && end < body_end) ||
                            (group.wave_import && wave_bodies.contains(&body)))
                    {
                        (start.min(body_start), end.max(body_end - 1))
                    } else {
                        (start, end)
                    }
                });
                if widened == (start, end) {
                    break;
                }
                (start, end) = widened;
            }
            // Operations of one region may run concurrently (parallel lanes,
            // or any two without a dependency path), while regions launch one
            // after another: spans are compared by region.
            arena.push((
                pointer,
                group.placeholder.device,
                region_of(start),
                region_of(end),
                group.placeholder.bytes,
            ));
            arena_members.push(group.members);
            arena_use_spans.push((first, last, group.first_write, group.wave_import));
            continue;
        }
        let deferred = group
            .placeholder
            .owner
            .downcast_ref::<GpuDeferredScratch>()
            .ok_or("GPU deferred scratch owner changed type")?;
        let bound = match &deferred.ty {
            // Packed scratch is bound by address: a raw buffer of its size.
            ConcreteWireType::Matrix(_) | ConcreteWireType::Bytes { .. } => {
                let stream = backend
                    .parameters_on_device(deferred.device)?
                    .native_launch_stream(deferred.device)
                    .map_err(|error| error.to_string())?;
                let size = usize::try_from(group.placeholder.bytes)
                    .map_err(|_| "GPU scratch size exceeds usize")?;
                let buffer = Arc::new(
                    crate::poly::dcrt::gpu::GpuDeviceBuffer::allocate(
                        &stream,
                        size,
                        crate::poly::dcrt::gpu::GpuBufferBacking::Device,
                    )
                    .map_err(|error| {
                        let members = group.members.iter().copied().collect::<BTreeSet<_>>();
                        let inputs = frame.input_ids.iter()
                            .filter(|(_, id)| members.contains(id))
                            .map(|(name, _)| name.as_str()).collect::<Vec<_>>();
                        let protected = members.intersection(&frame.scratch_protected)
                            .copied().collect::<Vec<_>>();
                        let families = frame.waves.iter().flat_map(|wave| wave.family_members.iter())
                            .filter(|(_, entries)| entries.iter().any(|(_, id)| members.contains(id)))
                            .map(|(name, _)| name.as_str()).collect::<BTreeSet<_>>();
                        let retained = frame.output_ids.iter()
                            .filter(|(_, id)| group.members.contains(id))
                            .map(|(name, _)| name.as_str()).collect::<Vec<_>>();
                        let public_of = frame.output_ids.iter()
                            .filter(|(_, id)| frame.trapdoor_public_ids.get(id)
                                .is_some_and(|public| group.members.contains(public)))
                            .map(|(name, _)| name.as_str()).collect::<Vec<_>>();
                        format!("GPU persistent scratch allocation failed: bytes={} members={members:?} inputs={inputs:?} protected={protected:?} wave_families={families:?} retained_outputs={retained:?} public_of_outputs={public_of:?} replayed={} packable={} first_write={:?} first_use={:?} last_use={:?}: {error}",
                            group.placeholder.bytes, deferred.replayed, group.packable, group.first_write,
                            group.references.first(), group.references.last())
                    })?,
                );
                BoundStorage {
                    device: deferred.device,
                    address: buffer.as_ptr() as u64,
                    bytes: group.placeholder.bytes,
                    owner: buffer,
                }
            }
            bounded => {
                let storage = StorageRef::Scratch(0);
                let (_, resident, _, _) = crate::gpu_physical_lowering::compact_value_owner(
                    backend,
                    deferred.device,
                    bounded.clone(),
                    storage,
                    None,
                )?;
                resident.storage(storage).cloned().ok_or("GPU compact scratch has no storage")?
            }
        };
        if bound.device != group.placeholder.device || bound.bytes != group.placeholder.bytes {
            return Err("GPU persistent scratch layout differs from its plan".into());
        }
        persistent.insert(pointer, bound);
    }
    let describe_admission_failure =
        |error: String, plan: &GraphScratchPlan, sizes: &BTreeMap<i32, u64>| {
            // Describe logical use-span overlap on admission failure. This is not
            // a measured VRAM peak: parallel barriers may widen these spans and
            // placement fragmentation may require more bytes. Keep this analysis
            // off the successful path, and identify large values for diagnosis.
            let mut spans = Vec::new();
            for (index, allocation) in plan.allocations.iter().enumerate() {
                spans.push((
                    allocation.device,
                    *allocation.references.first().expect("scratch is referenced"),
                    allocation.references.last().expect("scratch is referenced") + 1,
                    allocation.bytes as u64,
                    true,
                    index,
                ));
            }
            for (index, &(_, device, first, last, bytes)) in arena.iter().enumerate() {
                spans.push((
                    device,
                    starts[first],
                    starts.get(last + 1).copied().unwrap_or(frame.program.operations.len()),
                    bytes,
                    false,
                    index,
                ));
            }
            let mut events = BTreeMap::<(i32, usize), i128>::new();
            for &(device, first, end, bytes, _, _) in &spans {
                *events.entry((device, first)).or_default() += i128::from(bytes);
                *events.entry((device, end)).or_default() -= i128::from(bytes);
            }
            let mut live = BTreeMap::<i32, i128>::new();
            let mut peaks = BTreeMap::<i32, (i128, usize)>::new();
            for ((device, operation), delta) in events {
                let bytes = live.entry(device).or_default();
                *bytes += delta;
                let peak = peaks.entry(device).or_default();
                if *bytes > peak.0 {
                    *peak = (*bytes, operation);
                }
            }
            let mut details = Vec::new();
            for (&device, &(bytes, operation)) in &peaks {
                let mut contributors = spans
                    .iter()
                    .filter(|&&(owner, first, end, _, _, _)| {
                        owner == device && first <= operation && operation < end
                    })
                    .collect::<Vec<_>>();
                contributors.sort_unstable_by_key(|span| std::cmp::Reverse(span.3));
                let count = contributors.len();
                // The live allocations grouped by where they start and end, most bytes first.
                let mut groups = BTreeMap::<String, (usize, i128)>::new();
                for &&(_, first, end, bytes, graph, _) in &contributors {
                    let primitive = |position: usize| {
                        frame.program.operations.get(position).map(|op| {
                            format!(
                                "{:?}",
                                frame
                                    .program
                                    .implementations
                                    .resolve(op.implementation)
                                    .map(|imp| imp.primitive)
                            )
                        })
                    };
                    let key = format!(
                        "graph={graph} first={:?} last={:?} span={}",
                        primitive(first),
                        primitive(end - 1),
                        end - first
                    );
                    let entry = groups.entry(key).or_default();
                    entry.0 += 1;
                    entry.1 += i128::from(bytes);
                }
                let mut groups = groups.into_iter().collect::<Vec<_>>();
                groups.sort_by_key(|(_, (_, bytes))| std::cmp::Reverse(*bytes));
                details
                    .push(format!("live groups: {:?}", groups.iter().take(60).collect::<Vec<_>>()));
                let largest = contributors.into_iter().take(80).map(|&(_, first, end, bytes, graph, index)| {
                    let members = if graph { &plan.allocations[index].members } else { &arena_members[index] };
                    let shape = members.first().and_then(|id| {
                        frame.program.values[id.0 as usize]
                            .ty
                            .matrix_type()
                            .map(|ty| (ty.rows, ty.columns))
                    });
                    let describe = |position: usize| {
                        frame.program.operations.get(position).map(|op| {
                            let primitive = frame.program.implementations.resolve(op.implementation).map(|imp| imp.primitive);
                            let export = match primitive {
                                Ok(crate::gpu_execution_plan::GpuNativePrimitive::ExportCopy) => {
                                    match op.arguments.get(2) {
                                        Some(KernelArg::U32(slot)) => frame.export_templates.iter()
                                            .find(|template| template.slot == *slot as usize)
                                            .map(|template| (&template.name, template.index)),
                                        _ => None,
                                    }
                                }
                                _ => None,
                            };
                            format!("{primitive:?} outputs={:?} export={export:?}", op.outputs)
                        })
                    };
                    let replay_context = if graph {
                        None
                    } else {
                        let (use_first, use_last, write, wave_import) = arena_use_spans[index];
                        let boundaries = replayed_bodies.iter().filter(|&&(begin, end)| {
                            (begin <= use_first && use_first < end) !=
                                (begin <= use_last && use_last < end)
                        }).copied().collect::<Vec<_>>();
                        // The primitives of the bodies the value crosses, by count.
                        let kinds = boundaries.iter().map(|&(begin, end)| {
                            let mut counts = BTreeMap::<String, usize>::new();
                            for position in begin..end {
                                if let Some(op) = frame.program.operations.get(position) {
                                    let name = frame.program.implementations.resolve(op.implementation).map(|imp| format!("{:?}", imp.primitive)).unwrap_or_default();
                                    *counts.entry(name).or_default() += 1;
                                }
                            }
                            counts
                        }).collect::<Vec<_>>();
                        let near = |position: usize| (position.saturating_sub(3)..position + 8).filter_map(|p| describe(p).map(|d| format!("{p}:{d}"))).collect::<Vec<_>>();
                        Some(format!("uses={use_first}..={use_last} write={write:?} wave_import={wave_import} crossed_bodies={boundaries:?} body_kinds={kinds:?} near_first={:?} near_last={:?} first_use={:?} last_use={:?}", near(use_first), near(use_last), describe(use_first), describe(use_last)))
                    };
                    let imports = frame.external_io_imports.iter()
                        .chain(frame.external_io_loops.iter().flat_map(|body| &body.imports))
                        .filter(|import| members.contains(&import.destination))
                        .map(|import| (&import.key.name, import.key.index, import.request_operation, import.before_operation))
                        .collect::<Vec<_>>();
                    let templates = frame.import_templates.iter()
                        .filter(|import| members.contains(&import.destination))
                        .map(|import| (&import.key.name, import.key.index, import.before_operation))
                        .collect::<Vec<_>>();
                    // The innermost profiled graph node whose operations contain `position`.
                    let node_at = |position: usize| {
                        frame
                            .node_operations
                            .iter()
                            .filter(|(_, _, range)| range.contains(&(position as u32)))
                            .min_by_key(|(_, _, range)| range.end - range.start)
                            .map(|(scope, node, _)| format!("{scope:?}#{}", node.0))
                    };
                    let uses = if graph { None } else { Some(arena_use_spans[index]) };
                    let nodes = uses
                        .map(|(use_first, use_last, _, _)| (node_at(use_first), node_at(use_last)))
                        .or_else(|| Some((node_at(first), node_at(end - 1))));
                    format!("graph={graph} index={index} value={:?} members={:?} shape={shape:?} bytes={bytes} span={first}..{end} nodes={nodes:?} first_op={:?} last_op={:?} replay_context={replay_context:?} imports={imports:?} import_templates={templates:?}", members.first(), members.iter().take(8).collect::<Vec<_>>(), describe(first), describe(end - 1))
            }).collect::<Vec<_>>();
                details.push(format!("device={device} logical_span_peak_bytes={bytes} operation={operation} live_allocations={count} largest={largest:?}"));
            }
            format!(
                "GPU shared scratch pools: {error}; graph allocations={}, arena ranges={}, pool bytes or lower bounds by device={sizes:?}; {}",
                plan.allocations.len(),
                arena.len(),
                details.join("; "),
            )
        };
    // Graph temporaries must remain device-local; reject their lower bound
    // before packing. Arena ranges may use mapped host backing, so their
    // device-only bound cannot reject a candidate before that choice is made.
    let host_backing = crate::env::gpu_host_backing()?;
    let minimum = minimum_scratch_bytes(
        plan.allocations.iter().map(|allocation| (allocation.device, allocation.bytes as u64)),
        if host_backing { &[] } else { &arena },
    )?;
    #[cfg(test)]
    LAST_MINIMUM_SCRATCH.with(|last| *last.borrow_mut() = minimum.values().sum());
    admit_scratch_bytes(&minimum)
        .map_err(|error| describe_admission_failure(error, &plan, &minimum))?;
    // `MXX_GPU_SCRATCH_DIAGNOSIS=1`: log the admission diagnosis of every plan, admitted or not.
    if std::env::var("MXX_GPU_SCRATCH_DIAGNOSIS").is_ok_and(|value| value == "1") {
        tracing::info!(
            diagnosis = %describe_admission_failure("admitted".into(), &plan, &minimum),
            "GPU scratch diagnosis"
        );
    }
    let (mut sizes, mut arena_offsets, graph_spans) =
        plan.place(starts, frame.program.operations.len(), &arena);
    // Keep fast device backing when it fits. Only persistent arena ranges are
    // candidates for mapped host backing; short-lived graph temporaries stay
    // device-local. Neither scheduling nor replay liveness changes.
    let fraction = crate::env::gpu_memory_fraction()?;
    let mut limits = BTreeMap::new();
    for &device in sizes.keys() {
        let memory = crate::poly::dcrt::gpu::gpu_device_memory_usage(device)?;
        let budget = (memory.total as f64 * fraction) as u64;
        let used = memory.resident as u64;
        limits.insert(device, budget.saturating_sub(used));
    }
    let mut host_indices = BTreeSet::new();
    let mut host_sizes = BTreeMap::new();
    let mut host_offsets = Vec::new();
    // With host backing disabled, let device admission below reject an oversized
    // pool before allocation and attach the same lifetime ledger as other pool
    // failures. Returning here would discard the evidence needed to distinguish
    // genuine live data from fragmentation in the packed layout.
    if host_backing && sizes.iter().any(|(device, bytes)| *bytes > limits[device]) {
        let arena_spans = arena
            .iter()
            .map(|&(_, device, first, last, bytes)| ResidencySpan {
                device,
                first: starts[first],
                end: starts.get(last + 1).copied().unwrap_or(frame.program.operations.len()),
                bytes,
            })
            .collect::<Vec<_>>();
        host_indices =
            gpu_host_backing::select_host_ranges(&graph_spans, &arena_spans, &sizes, &limits)?;
        let host_arena = host_indices.iter().map(|&index| arena[index]).collect::<Vec<_>>();
        let items = host_arena
            .iter()
            .map(|&(_, device, _, _, bytes)| (device, bytes.div_ceil(ALIGNMENT) * ALIGNMENT))
            .collect::<Vec<_>>();
        (host_offsets, host_sizes) = pack_intervals(&items, None, |left, right| {
            let (_, _, first_a, last_a, _) = host_arena[left];
            let (_, _, first_b, last_b, _) = host_arena[right];
            first_a <= last_b && first_b <= last_a
        });
        let host_bytes = host_sizes
            .values()
            .try_fold(0u64, |sum, bytes| sum.checked_add(*bytes))
            .ok_or("mapped host scratch size overflows")?;
        let host_budget = crate::gpu_physical_lowering::available_mapped_host_budget()? as u64;
        if host_bytes > host_budget {
            return Err(format!(
                "GPU scratch cannot fit device budgets {limits:?}: mapped host scratch requires {host_bytes} bytes but only {host_budget} bytes are admitted; original device pools={sizes:?}"
            ));
        }
        let device_arena = arena
            .iter()
            .enumerate()
            .filter(|(index, _)| !host_indices.contains(index))
            .map(|(_, item)| *item)
            .collect::<Vec<_>>();
        let (placed_sizes, placed_offsets, _) =
            plan.place(starts, frame.program.operations.len(), &device_arena);
        let mut offsets = placed_offsets.into_iter();
        for (index, offset) in arena_offsets.iter_mut().enumerate() {
            if !host_indices.contains(&index) {
                *offset = offsets.next().expect("each device arena range is placed");
            }
        }
        sizes = placed_sizes;
        if sizes.iter().any(|(device, bytes)| *bytes > limits[device]) {
            return Err(format!(
                "GPU scratch after mapped host placement still exceeds device budgets: device pools={sizes:?}, limits={limits:?}, mapped host pools={host_sizes:?}"
            ));
        }
        tracing::info!(device_pools = ?sizes, mapped_host_pools = ?host_sizes,
            device_limits = ?limits, host_budget, "admitted GPU scratch with mapped host backing");
    }
    plan.pools =
        allocate_buffers(backend, &sizes, crate::poly::dcrt::gpu::GpuBufferBacking::Device)
            .map_err(|error| describe_admission_failure(error, &plan, &sizes))?;
    let host_pools = allocate_buffers(
        backend,
        &host_sizes,
        crate::poly::dcrt::gpu::GpuBufferBacking::HostMapped,
    )?;
    let mut mapped_offsets = host_offsets.into_iter();
    for (index, ((pointer, device, _, _, bytes), device_offset)) in
        arena.into_iter().zip(arena_offsets).enumerate()
    {
        let (buffer, offset) = if host_indices.contains(&index) {
            (&host_pools[&device], mapped_offsets.next().expect("each host arena range is placed"))
        } else {
            (&plan.pools[&device], device_offset)
        };
        persistent.insert(
            pointer,
            BoundStorage {
                device,
                address: buffer.as_ptr() as u64 + offset,
                bytes,
                owner: Arc::clone(buffer) as Arc<dyn std::any::Any + Send + Sync>,
            },
        );
    }
    frame.owners.rebind(|_, owner| rebound(owner, &persistent))?;
    for wave in &mut frame.waves {
        rebind(&mut wave.owner_bindings, &persistent)?;
        for (_, _, owner) in &mut wave.nested_owner_bindings {
            rebind_owner(owner, &persistent)?;
        }
    }
    for body in &mut frame.external_io_loops {
        for import in &mut body.ahead {
            if let Some(owner) = rebound(&import.staging, &persistent)? {
                import.staging = Arc::new(owner);
            }
        }
    }
    Ok(plan)
}

/// Byte alignment of every value placed in a shared buffer.
const ALIGNMENT: u64 = 256;

/// Place each item of `items` (its device and bytes) at an offset of its
/// device's buffer, the largest first, each at the lowest offset clear of
/// every placed item of that device that `overlap`s it in time. Returns the
/// offsets and each device's buffer size.
fn pack_intervals(
    items: &[(i32, u64)],
    bounds: Option<&[(usize, usize)]>,
    overlap: impl Fn(usize, usize) -> bool,
) -> (Vec<u64>, BTreeMap<i32, u64>) {
    let mut order = (0..items.len()).collect::<Vec<_>>();
    order.sort_by_key(|&item| (std::cmp::Reverse(items[item].1), item));
    let mut offsets = vec![0u64; items.len()];
    // Keep spatial order incrementally: once a sufficiently large gap is
    // reached, no later allocation can block it. Avoid collecting and sorting
    // every temporal overlap for each item in large production plans.
    let mut placed = BTreeMap::<i32, BTreeSet<(u64, usize)>>::new();
    let mut sizes = BTreeMap::<i32, u64>::new();
    if let Some(bounds) = bounds {
        assert_eq!(bounds.len(), items.len());
        let mut index = ScratchIntervalIndex::new(bounds);
        let mut busy = Vec::new();
        for item in order {
            let (device, size) = items[item];
            busy.clear();
            index.overlapping(item, &mut |other| {
                if items[other].0 == device && overlap(item, other) {
                    busy.push((offsets[other], offsets[other] + items[other].1));
                }
            });
            busy.sort_unstable();
            let mut offset = 0u64;
            for &(start, end) in &busy {
                if start >= offset + size {
                    break;
                }
                offset = offset.max(end);
            }
            offsets[item] = offset;
            index.insert(item);
            let top = sizes.entry(device).or_default();
            *top = (*top).max(offset + size);
        }
        return (offsets, sizes);
    }
    for item in order {
        let (device, size) = items[item];
        let on_device = placed.entry(device).or_default();
        let mut offset = 0u64;
        for &(start, other) in on_device.iter() {
            if start >= offset + size {
                break;
            }
            let end = start + items[other].1;
            if end > offset && overlap(item, other) {
                offset = end;
            }
        }
        offsets[item] = offset;
        on_device.insert((offset, item));
        let top = sizes.entry(device).or_default();
        *top = (*top).max(offset + size);
    }
    (offsets, sizes)
}

/// Index placed allocations by inclusive conservative lifetime bounds. A subtree
/// can be skipped when its latest placed end precedes the query or its earliest
/// start follows it. The exact overlap predicate still decides every candidate;
/// callers must ensure that disjoint bounds imply non-overlap. Storage is linear.
struct ScratchIntervalIndex<'a> {
    bounds: &'a [(usize, usize)],
    order: Vec<usize>,
    positions: Vec<usize>,
    ends: Vec<Option<usize>>,
    leaves: usize,
}

impl<'a> ScratchIntervalIndex<'a> {
    fn new(bounds: &'a [(usize, usize)]) -> Self {
        let mut order = (0..bounds.len()).collect::<Vec<_>>();
        order.sort_unstable_by_key(|&item| (bounds[item].0, item));
        let mut positions = vec![0; bounds.len()];
        for (position, &item) in order.iter().enumerate() {
            positions[item] = position;
            assert!(bounds[item].0 <= bounds[item].1);
        }
        let leaves = bounds.len().max(1).next_power_of_two();
        Self { bounds, order, positions, ends: vec![None; 2 * leaves], leaves }
    }

    fn insert(&mut self, item: usize) {
        let mut node = self.leaves + self.positions[item];
        self.ends[node] = Some(self.bounds[item].1);
        while node > 1 {
            node /= 2;
            self.ends[node] = self.ends[node * 2].max(self.ends[node * 2 + 1]);
        }
    }

    fn overlapping(&self, item: usize, visit: &mut impl FnMut(usize)) {
        self.visit(1, 0, self.leaves, self.bounds[item], visit);
    }

    fn visit(
        &self,
        node: usize,
        begin: usize,
        end: usize,
        query: (usize, usize),
        visit: &mut impl FnMut(usize),
    ) {
        if self.ends[node].is_none_or(|last| last < query.0) ||
            begin >= self.order.len() ||
            self.bounds[self.order[begin]].0 > query.1
        {
            return;
        }
        if end - begin == 1 {
            visit(self.order[begin]);
            return;
        }
        let middle = (begin + end) / 2;
        self.visit(node * 2, begin, middle, query, visit);
        self.visit(node * 2 + 1, middle, end, query, visit);
    }
}

/// Conservative residency lower bound without solving the packing problem.
/// Arena spans are inclusive region intervals; graph allocations contribute
/// only their largest individual size because their lifetimes can overlap arenas.
fn minimum_scratch_bytes(
    graph: impl Iterator<Item = (i32, u64)>,
    arena: &[(*const (), i32, usize, usize, u64)],
) -> Result<BTreeMap<i32, u64>, String> {
    let mut minimum = BTreeMap::<i32, u64>::new();
    for (device, bytes) in graph {
        let entry = minimum.entry(device).or_default();
        *entry = (*entry).max(bytes);
    }
    let mut events = BTreeMap::<(i32, usize), (u128, u128)>::new();
    for &(_, device, first, last, bytes) in arena {
        events.entry((device, first)).or_default().0 += bytes as u128;
        let after = last.checked_add(1).ok_or("scratch interval overflows")?;
        events.entry((device, after)).or_default().1 += bytes as u128;
    }
    let mut current = BTreeMap::<i32, u128>::new();
    for ((device, _), (added, removed)) in events {
        let used = current.entry(device).or_default();
        *used = used.checked_sub(removed).ok_or("invalid scratch interval events")? + added;
        let peak = minimum.entry(device).or_default();
        *peak = (*peak).max(u64::try_from(*used).map_err(|_| "scratch size exceeds u64")?);
    }
    Ok(minimum)
}

/// Check both physically free memory and the configured total-device fraction
/// before allocating any pool. This also gates the lower-bound check above.
fn admit_scratch_bytes(sizes: &BTreeMap<i32, u64>) -> Result<(), String> {
    let fraction = crate::env::gpu_memory_fraction()?;
    for (&device, &bytes) in sizes {
        let info = crate::poly::dcrt::gpu::gpu_memory_info(device)?;
        let budget = (info.total as f64 * fraction) as u64;
        let used = (info.total - info.free) as u64;
        let available = (info.free as u64).min(budget.saturating_sub(used));
        if bytes > available {
            return Err(format!(
                "GPU {device} scratch needs at least {bytes} bytes, above {available} available bytes within the memory budget"
            ));
        }
    }
    Ok(())
}

/// Allocate one device buffer of each size of `sizes`.
fn allocate_buffers(
    backend: &GpuDcrtBackend,
    sizes: &BTreeMap<i32, u64>,
    backing: crate::poly::dcrt::gpu::GpuBufferBacking,
) -> Result<BTreeMap<i32, Arc<GpuDeviceBuffer>>, String> {
    if matches!(backing, crate::poly::dcrt::gpu::GpuBufferBacking::Device) {
        admit_scratch_bytes(sizes)?;
    }
    sizes
        .iter()
        .filter(|(_, bytes)| **bytes > 0)
        .map(|(&device, &bytes)| {
            let stream = backend
                .parameters_on_device(device)?
                .native_launch_stream(device)
                .map_err(|error| error.to_string())?;
            let size = usize::try_from(bytes).map_err(|_| "GPU scratch buffer exceeds usize")?;
            let buffer = GpuDeviceBuffer::allocate(&stream, size, backing).map_err(|error| {
                format!("device {device} scratch buffer of {bytes} bytes: {error}")
            })?;
            Ok((device, Arc::new(buffer)))
        })
        .collect()
}

impl GraphScratchPlan {
    /// The lane of the outermost loop containing `first` that also contains
    /// `last`, when no other loop contains either.
    fn lane_of(&self, first: usize, last: usize) -> Option<usize> {
        let outer = self.loops.iter().find(|range| range.contains(&first))?;
        let inner = self
            .loops
            .iter()
            .any(|range| range != outer && (range.contains(&first) || range.contains(&last)));
        if inner {
            return None;
        }
        self.lanes[&(outer.start, outer.end)]
            .iter()
            .position(|lane| lane.contains(&first) && lane.contains(&last))
    }

    /// Place every allocation in its device's region scratch pool and return
    /// the pool sizes. Placement follows the order the barriers impose: an
    /// allocation reuses memory whose barrier after its last use precedes the
    /// allocation's barrier. Outside a parallel loop the barrier chain orders
    /// every free before later allocations; inside one, allocations follow
    /// the chain at the loop's start, and a lane's own scratch also its
    /// lane's chain, so a lane reuses what it freed and the loop's other
    /// frees are reused after the loop ends. Regions launch in order, so a new region reuses
    /// everything freed before it, including allocations whose last use is in the region
    /// before.
    fn place(
        &mut self,
        starts: &[usize],
        operations: usize,
        arena: &[(*const (), i32, usize, usize, u64)],
    ) -> (BTreeMap<i32, u64>, Vec<u64>, Vec<ResidencySpan>) {
        // When each allocation's bytes may be reused: by any allocation from
        // `release` on, and by one of the same lane of the same loop from
        // `lane_release` on.
        let count = self.allocations.len();
        let (mut release, mut lane_release) = (vec![0usize; count], vec![None; count]);
        let mut lane_of = vec![None; count];
        let (mut open, mut next_loop, mut region) = (None::<(usize, usize)>, 0, 0);
        for index in 0..operations {
            while starts.get(region + 1).is_some_and(|start| *start <= index) {
                region += 1;
            }
            if open.is_some_and(|(_, end)| index >= end) {
                open = None;
            }
            while let Some(range) = self.loops.get(next_loop) &&
                range.start <= index
            {
                if open.is_none() && index < range.end {
                    open = Some((next_loop, range.end));
                }
                next_loop += 1;
            }
            let next_region = starts.get(region + 1).copied().unwrap_or(operations);
            for &allocation in self.free_after.get(&index).into_iter().flatten() {
                let placed = &self.allocations[allocation];
                release[allocation] = match open {
                    Some((loop_index, end)) => {
                        if let Some(lane) = placed.lane {
                            lane_release[allocation] = Some(index + 1);
                            lane_of[allocation] = Some((loop_index, lane));
                        }
                        end.min(next_region)
                    }
                    None => index + 1,
                };
            }
        }
        let firsts = self
            .allocations
            .iter()
            .map(|allocation| *allocation.references.first().expect("scratch is referenced"))
            .collect::<Vec<_>>();
        let first = |allocation: usize| firsts[allocation];
        // Whether `later` may take bytes `earlier` held.
        let after = |earlier: usize, later: usize| {
            let start = first(later);
            start >= release[earlier] ||
                (lane_of[earlier].is_some() &&
                    lane_of[earlier] == lane_of[later] &&
                    lane_release[earlier].is_some_and(|free| start >= free))
        };
        let graph_items = self
            .allocations
            .iter()
            .map(|allocation| {
                (allocation.device, (allocation.bytes as u64).div_ceil(ALIGNMENT) * ALIGNMENT)
            })
            .collect::<Vec<_>>();
        let arena_items = arena
            .iter()
            .map(|&(_, device, _, _, bytes)| (device, bytes.div_ceil(ALIGNMENT) * ALIGNMENT))
            .collect::<Vec<_>>();
        let graph_overlap = |left, right| !after(left, right) && !after(right, left);
        let arena_overlap = |left: usize, right: usize| {
            let (_, _, first_a, last_a, _) = arena[left];
            let (_, _, first_b, last_b, _) = arena[right];
            first_a <= last_b && first_b <= last_a
        };
        let region_of = |operation| starts.partition_point(|start| *start <= operation) - 1;
        let graph_regions = firsts
            .iter()
            .zip(&release)
            .map(|(&first, &release)| (region_of(first), region_of(release - 1)))
            .collect::<Vec<_>>();
        let items = graph_items.iter().chain(&arena_items).copied().collect::<Vec<_>>();
        let overlap = |left, right| {
            match (left < count, right < count) {
                (true, true) => graph_overlap(left, right),
                (false, false) => arena_overlap(left - count, right - count),
                _ => {
                    let (graph, arena_index) =
                        if left < count { (left, right - count) } else { (right, left - count) };
                    let (first, last) = graph_regions[graph];
                    let (_, _, arena_first, arena_last, _) = arena[arena_index];
                    // Host-bound ranges are reserved for their entire regions.
                    // Cross-class reuse occurs only across ordered region launches,
                    // never between a host upload and device work in one region.
                    first <= arena_last && arena_first <= last
                }
            }
        };
        // Region bounds conservatively cover both graph allocations and host
        // captures, including the wider cross-class reservation above.
        let bounds = graph_regions
            .iter()
            .copied()
            .chain(arena.iter().map(|&(_, _, first, last, _)| (first, last)))
            .collect::<Vec<_>>();
        let (mut offsets, mut sizes) = pack_intervals(&items, Some(&bounds), overlap);
        // Equal-sized long-lived captures can fragment a pool when placed after
        // short temporaries. Try them first without changing any overlap rule,
        // keeping the original layout whenever the alternative is not smaller.
        let mut priority = (0..items.len()).collect::<Vec<_>>();
        priority.sort_by_key(|&index| {
            let duration = if index < count {
                release[index] - firsts[index]
            } else {
                let (_, _, first, last, _) = arena[index - count];
                starts.get(last + 1).copied().unwrap_or(operations) - starts[first]
            };
            (std::cmp::Reverse(duration), index)
        });
        let prioritized = priority.iter().map(|&index| items[index]).collect::<Vec<_>>();
        let priority_bounds = priority.iter().map(|&index| bounds[index]).collect::<Vec<_>>();
        let (alternative, alternative_sizes) =
            pack_intervals(&prioritized, Some(&priority_bounds), |left, right| {
                overlap(priority[left], priority[right])
            });
        for (&device, size) in &mut sizes {
            if alternative_sizes[&device] < *size {
                *size = alternative_sizes[&device];
                for (position, &index) in priority.iter().enumerate() {
                    if items[index].0 == device {
                        offsets[index] = alternative[position];
                    }
                }
            }
        }
        // Greedy packing is not globally optimal. Keep the old separate placement
        // as an upper bound per device so combining the pools cannot inflate VRAM.
        let (graph_offsets, graph_sizes) =
            pack_intervals(&graph_items, Some(&bounds[..count]), graph_overlap);
        let (arena_offsets, arena_sizes) =
            pack_intervals(&arena_items, Some(&bounds[count..]), arena_overlap);
        for (&device, size) in &mut sizes {
            let graph_size = graph_sizes.get(&device).copied().unwrap_or(0);
            let separate = graph_size + arena_sizes.get(&device).copied().unwrap_or(0);
            if *size > separate {
                *size = separate;
                for (index, &(owner, _)) in items.iter().enumerate() {
                    if owner == device {
                        offsets[index] = if index < count {
                            graph_offsets[index]
                        } else {
                            graph_size + arena_offsets[index - count]
                        };
                    }
                }
            }
        }
        for (allocation, &offset) in self.allocations.iter_mut().zip(&offsets) {
            allocation.offset = offset;
        }
        let spans = self
            .allocations
            .iter()
            .enumerate()
            .map(|(index, allocation)| ResidencySpan {
                device: allocation.device,
                first: firsts[index],
                end: lane_release[index].unwrap_or(release[index]),
                bytes: allocation.bytes as u64,
            })
            .collect();
        (sizes, offsets[count..].to_vec(), spans)
    }

    /// Move the memory-node chain to top-level operation `index` of the
    /// region starting at `start`: a new region starts an empty chain, and a
    /// parallel loop's end joins every memory node created inside it.
    fn enter_operation(&mut self, start: usize, index: usize) {
        if self.chain_region != Some(start) {
            self.chain_region = Some(start);
            self.chain.clear();
            if let Some((_, created, lanes)) = &mut self.open_loop {
                created.clear();
                lanes.iter_mut().for_each(BTreeMap::clear);
            }
        }
        if let Some((end, _, _)) = &self.open_loop &&
            index >= *end
        {
            let (_, created, _) = self.open_loop.take().expect("the loop is open");
            for (device, created) in created {
                self.chain.entry(device).or_default().extend(created);
            }
        }
        while let Some(range) = self.loops.get(self.next_loop) &&
            range.start <= index
        {
            if self.open_loop.is_none() && index < range.end {
                let lanes = self.lanes[&(range.start, range.end)].len();
                self.open_loop = Some((range.end, BTreeMap::new(), vec![BTreeMap::new(); lanes]));
            }
            self.next_loop += 1;
        }
    }

    /// The memory nodes the next memory node of `allocation` follows, of its
    /// device's pool: its lane's chain inside the open loop, otherwise
    /// `self.chain`.
    fn chain_of(&self, allocation: usize) -> &[u32] {
        let (device, lane) =
            (self.allocations[allocation].device, self.allocations[allocation].lane);
        let chain = self.chain.get(&device).map_or(&[][..], Vec::as_slice);
        match (&self.open_loop, lane) {
            (Some((_, _, lanes)), Some(lane)) => {
                lanes[lane].get(&device).map_or(chain, Vec::as_slice)
            }
            _ => chain,
        }
    }

    /// Record a memory node of `allocation` created after `chain_of`.
    fn push_memory_node(&mut self, allocation: usize, token: u32) {
        let (device, lane) =
            (self.allocations[allocation].device, self.allocations[allocation].lane);
        match &mut self.open_loop {
            Some((_, created, lanes)) => {
                created.entry(device).or_default().push(token);
                if let Some(lane) = lane {
                    lanes[lane].insert(device, vec![token]);
                }
            }
            None => {
                self.chain.insert(device, vec![token]);
            }
        }
    }

    /// Bind the scratch whose first use is top-level operation `index` into
    /// `frame`, and return the allocation-barrier tokens that operation must
    /// follow.
    pub(crate) fn before_operation(
        &mut self,
        builder: &mut GpuNativeGraphBuilder,
        frame: &mut PhysicalFrame,
        recipe: &mut RegionScratchRecipe,
        start: usize,
        index: usize,
    ) -> Result<Vec<u32>, GpuNativeGraphError> {
        let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
        self.enter_operation(start, index);
        for allocation in self.allocate_before.get(&index).cloned().unwrap_or_default() {
            let dependencies = self.chain_of(allocation).to_vec();
            let token = builder.add_memory_barrier(&[], &dependencies)?;
            recipe.before.entry(index).or_default().push((Vec::new(), dependencies, token));
            self.push_memory_node(allocation, token);
            let allocation = &mut self.allocations[allocation];
            let pool = self
                .pools
                .get(&allocation.device)
                .ok_or_else(|| invalid("Graph scratch has no pool on its device"))?;
            let address = pool.as_ptr() as u64 + allocation.offset;
            allocation.allocated = Some((address, start, token));
            let bound = BoundStorage {
                device: allocation.device,
                address,
                bytes: allocation.bytes as u64,
                owner: Arc::clone(pool) as Arc<dyn std::any::Any + Send + Sync>,
            };
            for id in &allocation.members {
                let owner = frame
                    .owners
                    .get(id)
                    .ok_or_else(|| invalid("Graph scratch owner is missing"))?;
                let replaced =
                    std::collections::HashMap::from([(allocation.placeholder, bound.clone())]);
                if let Some(rebound) =
                    owner.rebound(&replaced, owner.ready_events()).map_err(invalid)?
                {
                    frame.owners.insert(*id, Arc::new(rebound));
                }
            }
        }
        let tokens = self
            .referenced_by
            .get(&index)
            .into_iter()
            .flatten()
            .filter_map(|&allocation| match self.allocations[allocation].allocated {
                // Allocations of an earlier region's Graph precede this Graph.
                Some((_, region, token)) if region == start => Some(token),
                _ => None,
            })
            .collect::<Vec<_>>();
        if !tokens.is_empty() {
            recipe.pending.insert(index, tokens.clone());
        }
        Ok(tokens)
    }

    /// Add the free barrier of the scratch whose last use is top-level
    /// operation `index` of the region starting at `start`, after every
    /// emitted operation of that region that uses it. Uses in earlier regions
    /// have ended: this region launches after them.
    pub(crate) fn after_operation(
        &mut self,
        builder: &mut GpuNativeGraphBuilder,
        recipe: &mut RegionScratchRecipe,
        start: usize,
        index: usize,
    ) -> Result<(), GpuNativeGraphError> {
        for index_of in self.free_after.get(&index).cloned().unwrap_or_default() {
            let allocation = &self.allocations[index_of];
            allocation.allocated.ok_or_else(|| {
                GpuNativeGraphError::Native("Graph scratch is freed before its allocation".into())
            })?;
            let readers = allocation
                .references
                .range(start..=index)
                .map(|&reference| (reference - start) as u32)
                .collect::<Vec<_>>();
            let dependencies = self.chain_of(index_of).to_vec();
            let token = builder.add_memory_barrier(&readers, &dependencies)?;
            recipe.after.entry(index).or_default().push((readers, dependencies, token));
            self.push_memory_node(index_of, token);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_deferred_addresses_reserve_disjoint_ranges_and_fail_without_wrap() {
        let next = AtomicU64::new(1 << 62);
        let small = reserve_placeholder_address(&next, 1).unwrap();
        let large = reserve_placeholder_address(&next, (1 << 40) + 1).unwrap();
        let following = reserve_placeholder_address(&next, 8).unwrap();
        assert_eq!(large - small, 256);
        assert!(following >= large + (1 << 40) + 1);
        assert_eq!(following % 256, 0);

        let near_end = AtomicU64::new(u64::MAX - 255);
        assert!(reserve_placeholder_address(&near_end, 1).is_err());
        assert_eq!(near_end.load(Ordering::Relaxed), u64::MAX - 255);
        assert!(reserve_placeholder_address(&next, u64::MAX).is_err());
        assert_eq!(next.load(Ordering::Relaxed), following + 256);
    }

    #[test]
    fn test_scratch_admission_lower_bound_respects_regions_and_devices() {
        let arena = [
            (std::ptr::null(), 0, 0, 1, 100),
            (std::ptr::null(), 0, 1, 2, 200),
            (std::ptr::null(), 0, 3, 3, 250),
            (std::ptr::null(), 1, 0, 3, 400),
        ];
        let minimum = minimum_scratch_bytes([(0, 280), (1, 500)].into_iter(), &arena).unwrap();
        assert_eq!(minimum, BTreeMap::from([(0, 300), (1, 500)]));
        assert!(
            minimum_scratch_bytes(std::iter::empty(), &[(std::ptr::null(), 0, 0, usize::MAX, 1)])
                .is_err()
        );
    }

    #[test]
    fn test_gpu_scratch_packing_reuses_gaps_and_separates_devices() {
        let items = [(0, 8), (0, 6), (0, 4), (0, 2), (1, 8)];
        let lifetimes = [(0, 1), (1, 2), (2, 3), (0, 3), (0, 3)];
        let (offsets, sizes) = pack_intervals(&items, None, |a, b| {
            lifetimes[a].0 <= lifetimes[b].1 && lifetimes[b].0 <= lifetimes[a].1
        });
        assert_eq!(offsets, [0, 8, 0, 14, 0]);
        assert_eq!(sizes, BTreeMap::from([(0, 16), (1, 8)]));
        let (offsets, sizes) = pack_intervals(&[], None, |_, _| unreachable!());
        assert!(offsets.is_empty());
        assert!(sizes.is_empty());
    }

    #[test]
    fn test_gpu_scratch_interval_index_preserves_placement() {
        for count in 0..100 {
            let items = (0..count)
                .map(|_| ((rand::random::<u8>() % 3) as i32, (rand::random::<u8>() % 65) as u64))
                .collect::<Vec<_>>();
            let lifetimes = (0..count)
                .map(|_| {
                    let start = rand::random::<u8>() as usize;
                    (start, start + (rand::random::<u8>() % 32) as usize)
                })
                .collect::<Vec<_>>();
            let bounds = lifetimes
                .iter()
                .map(|&(first, last)| (first.saturating_sub(5), last + 7))
                .collect::<Vec<_>>();
            let overlap = |a: usize, b: usize| {
                lifetimes[a].0 <= lifetimes[b].1 && lifetimes[b].0 <= lifetimes[a].1
            };
            assert_eq!(
                pack_intervals(&items, Some(&bounds), overlap),
                pack_intervals(&items, None, overlap)
            );
        }
    }

    #[test]
    fn test_gpu_scratch_interval_index_skips_disjoint_lifetimes() {
        let items = vec![(0, 1024); 4096];
        let bounds = (0..items.len()).map(|index| (index, index)).collect::<Vec<_>>();
        let (offsets, sizes) = pack_intervals(&items, Some(&bounds), |_, _| {
            panic!("disjoint bounds must exclude the pair before the overlap predicate")
        });
        assert!(offsets.iter().all(|&offset| offset == 0));
        assert_eq!(sizes, BTreeMap::from([(0, 1024)]));
    }

    #[test]
    fn test_joint_scratch_pool_reuses_only_disjoint_regions() {
        let make_plan = || GraphScratchPlan {
            allocations: vec![GraphAllocation {
                placeholder: std::ptr::null(),
                members: Vec::new(),
                device: 0,
                bytes: 1024,
                references: BTreeSet::from([0]),
                offset: 0,
                lane: None,
                allocated: None,
            }],
            allocate_before: BTreeMap::from([(0, vec![0])]),
            free_after: BTreeMap::from([(0, vec![0])]),
            referenced_by: BTreeMap::from([(0, vec![0])]),
            loops: Vec::new(),
            lanes: BTreeMap::new(),
            chain: BTreeMap::new(),
            chain_region: None,
            open_loop: None,
            next_loop: 0,
            pools: BTreeMap::new(),
        };
        let mut disjoint = make_plan();
        let (sizes, offsets, _) = disjoint.place(&[0, 1], 2, &[(std::ptr::null(), 0, 1, 1, 2048)]);
        assert_eq!(sizes[&0], 2048);
        assert_eq!(disjoint.allocations[0].offset, offsets[0]);
        for last in [0, 1] {
            let mut overlapping = make_plan();
            let (sizes, offsets, _) =
                overlapping.place(&[0, 1], 2, &[(std::ptr::null(), 0, 0, last, 2048)]);
            assert_eq!(sizes[&0], 3072);
            let graph = overlapping.allocations[0].offset;
            assert!(graph + 1024 <= offsets[0] || offsets[0] + 2048 <= graph);
        }
    }

    use crate::{
        GpuRuntime, MemoryArtifactStore, RuntimeValue,
        backend::{poly::cpu_backend, poly_gpu::gpu_backend},
        executor::{ExecutionConfig, execute_in_session},
        poly::{
            Poly,
            dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
        },
    };
    use mxx_dsl::{DslContext, HashTag, Int, IntType, Ring, parallel};
    use mxx_ir_core::ParamEnv;

    /// A later wave's family member must survive intervening waves until
    /// its last dynamic read, even though the read uses a pointer table.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_replayed_family_survives_late_dynamic_read() {
        let parameter = |name, default| {
            std::env::var(name).map_or(default, |value| value.parse::<usize>().unwrap())
        };
        let dimension = parameter("MXX_TEST_SCRATCH_RING_DIMENSION", 16) as u32;
        let columns = parameter("MXX_TEST_SCRATCH_COLUMNS", 4);
        let count = parameter("MXX_TEST_SCRATCH_MEMBERS", 4);
        assert!(count >= 2);
        let params = DCRTPolyParams::new(dimension, 2, 20, 4, None, None);
        let gpu_params = GpuDCRTPolyParams::new(dimension, params.moduli().to_vec(), 4, None);
        let ring = Ring::from_crt_moduli(
            params.moduli().iter().copied().map(Into::into).collect(),
            dimension,
        );
        let context = DslContext::new("late-dynamic-family-read");
        let selector: Int = context.input("selector", IntType).unwrap();
        let key = ring.bytes_input("key", 32);
        let members = parallel(count, |index| {
            let mut tag = HashTag::from(b"scratch-family".as_slice());
            tag.push(index);
            Ok(ring.hash_matrix(key.clone(), tag, (1, columns)))
        })
        .unwrap();
        let mut sum = members.at(selector.clone());
        for _ in 0..2 {
            let previous = sum.clone();
            let work = parallel(count, |index| {
                let mut tag = HashTag::from(b"scratch-intervening".as_slice());
                tag.push(index);
                Ok(previous.clone() + ring.hash_matrix(key.clone(), tag, (1, columns)))
            })
            .unwrap();
            sum = work.at(selector.clone()) + members.at(selector.clone());
        }
        let graph = context
            .output("sum", sum)
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let inputs = BTreeMap::from([
            ("key".into(), RuntimeValue::Bytes(Arc::from(rand::random::<[u8; 32]>()))),
            ("selector".into(), RuntimeValue::Int((count - 1).into())),
        ]);
        let expected = execute_in_session(
            &graph,
            &mut cpu_backend([params]),
            inputs.clone(),
            &mut MemoryArtifactStore::default(),
            rand::random(),
            ExecutionConfig::default(),
        )
        .unwrap();
        let RuntimeValue::Matrix(expected) = &expected.outputs["sum"] else {
            panic!("matrix output")
        };
        let expected = expected.cpu_full_arc().expect("CPU reference matrix");
        let mut runtime = GpuRuntime::new(gpu_backend([gpu_params])).unwrap();
        runtime
            .options_mut()
            .integer_input_ranges
            .insert("selector".into(), 0.into()..=(count - 1).into());
        let mut plan =
            runtime.plan_with_fixed_geometry_for_test(graph, &inputs, columns, 1).unwrap();
        for _ in 0..3 {
            let actual = runtime.execute(&mut plan, inputs.clone()).unwrap();
            let actual = runtime.download_matrix_output(&actual.output("sum").unwrap()).unwrap();
            for column in 0..columns {
                assert_eq!(
                    actual.entry(0, column).coeffs_biguints(),
                    expected.entry(0, column).coeffs_biguints()
                );
            }
        }
    }
}
