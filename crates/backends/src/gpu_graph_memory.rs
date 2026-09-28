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
//! a region form a chain in operation order, so a free precedes every later
//! allocation and that allocation may take the freed bytes. Inside a parallel
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
//!
//! Other scratch lives in memory the plan owns. A scratch value that is written before it is read
//! and used on one device shares one arena per device with the others: it takes a byte range for
//! its live span, from the region of its first use to the region of its last, widened to every
//! replayed wave or host-loop body it overlaps, since a replay runs the whole body again. Spans are
//! regions because operations of one region may run concurrently while regions launch in order.
//! Values whose spans share no region share bytes, so a chain of loops, each reading the running
//! result of the one before, needs about two of them at once. Any other scratch value, together
//! with inputs, outputs, wave-bound members, and imports, is an allocation of its own.

use crate::{
    backend::{BoundStorage, GpuResidentValue, poly_gpu::GpuDcrtBackend},
    gpu_execution_plan::{
        CompiledGpuOp, KernelArg, PhysicalEncoding, PhysicalValue, PhysicalValueId, StorageRef,
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
    /// A full matrix, or a bounded matrix held compactly.
    ty: ConcreteWireType,
    device: i32,
}

/// Whether `bound` has no memory yet: its allocation is deferred to
/// compilation.
pub(crate) fn is_graph_managed(bound: &BoundStorage) -> bool {
    bound.owner.is::<GpuDeferredScratch>()
}

/// A distinct, never dereferenced address for each deferred allocation, so
/// storage identities stay unique until real memory is bound.
fn placeholder_address() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1 << 62);
    NEXT.fetch_add(1 << 40, Ordering::Relaxed)
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
        address: placeholder_address(),
        bytes: u64::try_from(payload_bytes).map_err(|_| "GPU compact scratch size exceeds u64")?,
        owner: Arc::new(GpuDeferredScratch { ty, device }),
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
    )?;
    let bound = BoundStorage {
        device: limb_device,
        address: placeholder_address(),
        bytes: u64::try_from(data_bytes).map_err(|_| "GPU scratch size exceeds u64")?,
        owner: Arc::new(GpuDeferredScratch { ty: ConcreteWireType::Matrix(ty.clone()), device }),
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
    /// Byte offset in its device's region scratch pool.
    offset: u64,
    /// The lane of its outermost parallel loop that holds every use, when
    /// no inner loop holds any: it reuses memory that lane freed.
    lane: Option<usize>,
    /// Address, and the start and allocation-barrier token of the region
    /// whose Graph first uses it.
    allocated: Option<(u64, usize, u32)>,
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
    /// Memory nodes the next memory node outside a parallel loop follows.
    chain: Vec<u32>,
    /// Start of the region whose builder issued the `chain` tokens.
    chain_region: Option<usize>,
    /// The outermost open parallel loop: its end, the memory nodes created
    /// inside it so far, and each lane's chain. Its nodes follow `chain`,
    /// frozen meanwhile, and a lane's own nodes also follow its chain.
    open_loop: Option<(usize, Vec<u32>, Vec<Option<Vec<u32>>>)>,
    /// First loop of `loops` not reached yet.
    next_loop: usize,
    /// Each device's region scratch pool.
    pools: BTreeMap<i32, Arc<GpuDeviceBuffer>>,
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
/// takes a range of a region scratch pool. Every other deferred allocation is
/// allocated for the plan here, and the pools are allocated too.
pub(crate) fn plan_graph_scratch(
    backend: &GpuDcrtBackend,
    frame: &mut PhysicalFrame,
    starts: &[usize],
) -> Result<GraphScratchPlan, String> {
    struct Group {
        placeholder: BoundStorage,
        members: Vec<PhysicalValueId>,
        eligible: bool,
        /// Whether an arena range may hold it: one storage the host never
        /// binds, used on its own device.
        packable: bool,
        references: BTreeSet<usize>,
        first_write: Option<usize>,
    }
    let mut groups = BTreeMap::<*const (), Group>::new();
    let mut group_of = BTreeMap::<PhysicalValueId, *const ()>::new();
    let mut collect =
        |id: PhysicalValueId, owner: &GpuResidentValue, eligible: bool, packable: bool| {
            for (_, bound) in owner.storages() {
                if !bound.owner.is::<GpuDeferredScratch>() {
                    continue;
                }
                let pointer = Arc::as_ptr(&bound.owner).cast::<()>();
                let group = groups.entry(pointer).or_insert_with(|| Group {
                    placeholder: bound.clone(),
                    members: Vec::new(),
                    eligible: true,
                    packable: true,
                    references: BTreeSet::new(),
                    first_write: None,
                });
                group.members.push(id);
                group.eligible &= eligible;
                group.packable &= packable;
                group_of.insert(id, pointer);
            }
        };
    for (&id, owner) in &frame.owners {
        let physical = &frame.program.values[id.0 as usize];
        let packable = owner.storages().count() == 1 && !frame.scratch_protected.contains(&id);
        // Full matrices, and compact scratch (whose producer writes every byte
        // it leaves meaningful), are bound by address alone.
        let eligible = matches!(
            (&physical.ty, physical.encodings.as_ref()),
            (
                ConcreteWireType::Matrix(_),
                [PhysicalEncoding::FullCoeff] | [PhysicalEncoding::FullEval]
            ) | (
                ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. },
                [PhysicalEncoding::CompactCoeff { .. }] |
                    [PhysicalEncoding::CompactCoeffPerCrtLimb { .. }]
            )
        ) && packable;
        collect(id, owner, eligible, packable);
    }
    // Wave bindings are swapped in by the host between launches.
    for wave in &frame.waves {
        for (&id, owner) in &wave.owner_bindings {
            collect(id, owner, false, false);
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
            group.packable &= device == group.placeholder.device;
        });
    }
    // A placed import is written by the host from its load site on: every
    // owner of its destination (a wave's read-ahead twin too) spans from
    // there to the destination's last use. It is never region scratch, which
    // could reuse its bytes inside the region it is uploaded in, and it takes
    // arena bytes although the host binds it: the host writes it only inside
    // that span.
    let placed = {
        use crate::gpu_physical_lowering::ImportDestination::Placed;
        frame
            .external_io_imports
            .iter()
            .chain(frame.external_io_loops.iter().flat_map(|body| &body.imports))
            .filter(|import| matches!(import.upload_owner, Placed { .. }))
            .map(|import| (import.destination, import.request_operation as usize))
            .chain(frame.import_templates.iter().enumerate().filter_map(|(index, import)| {
                // A wave's template is read from its wave group's start, a
                // root one from the start of execution.
                let in_wave =
                    frame.waves.iter().any(|wave| wave.import_template_indices.contains(&index));
                matches!(import.upload_owner, Placed { .. }).then(|| {
                    (import.destination, if in_wave { import.before_operation as usize } else { 0 })
                })
            }))
            .collect::<Vec<_>>()
    };
    for (id, load) in placed {
        let owners = groups
            .iter()
            .filter(|(_, group)| group.members.contains(&id))
            .map(|(pointer, _)| *pointer)
            .collect::<Vec<_>>();
        let mut references = BTreeSet::from([load]);
        for pointer in &owners {
            references.extend(groups[pointer].references.iter().copied());
        }
        for pointer in &owners {
            let group = groups.get_mut(pointer).expect("the group exists");
            group.references = references.clone();
            group.first_write = Some(load);
            group.eligible = false;
            group.packable = true;
        }
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
    let lanes = frame
        .parallel_lanes
        .iter()
        .filter_map(|lanes| {
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
        chain: Vec::new(),
        chain_region: None,
        open_loop: None,
        next_loop: 0,
        pools: BTreeMap::new(),
    };
    let mut persistent = BTreeMap::<*const (), BoundStorage>::new();
    // Plan-owned scratch written before it is read: device, live span, bytes.
    let mut arena = Vec::<(*const (), i32, usize, usize, u64)>::new();
    for (pointer, group) in groups {
        let graph_owned = match (group.references.first(), group.references.last()) {
            (Some(&first), Some(&last)) if group.eligible => {
                let (first_region, last_region) = (region_of(first), region_of(last));
                // An allocation that outlives its region keeps its bytes until
                // the region after its last use. It must not live across a
                // replayed region's relaunches, unless every use lies in one
                // replayed body: each replay then runs its whole life again,
                // after the previous replay's has ended.
                let one_body = replayed_bodies
                    .iter()
                    .any(|&(body_start, body_end)| body_start <= first && last < body_end);
                group.first_write.is_some_and(|write| write <= first) &&
                    (first_region == last_region ||
                        ((one_body || (!replayed(first_region) && !replayed(last_region))) &&
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
            loop {
                let widened = replayed_bodies.iter().fold((start, end), |(start, end), &body| {
                    let (body_start, body_end) = body;
                    if body_start < body_end && start < body_end && body_start <= end {
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
            continue;
        }
        let deferred = group
            .placeholder
            .owner
            .downcast_ref::<GpuDeferredScratch>()
            .ok_or("GPU deferred scratch owner changed type")?;
        let bound = match &deferred.ty {
            // Packed scratch is bound by address: a raw buffer of its size.
            ConcreteWireType::Matrix(_) => {
                let stream = backend
                    .parameters_on_device(deferred.device)?
                    .native_launch_stream(deferred.device)
                    .map_err(|error| error.to_string())?;
                let size = usize::try_from(group.placeholder.bytes)
                    .map_err(|_| "GPU scratch size exceeds usize")?;
                let buffer = Arc::new(
                    crate::poly::dcrt::gpu::GpuDeviceBuffer::allocate(&stream, size)
                        .map_err(|error| error.to_string())?,
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
                )?;
                resident.storage(storage).cloned().ok_or("GPU compact scratch has no storage")?
            }
        };
        if bound.device != group.placeholder.device || bound.bytes != group.placeholder.bytes {
            return Err("GPU persistent scratch layout differs from its plan".into());
        }
        persistent.insert(pointer, bound);
    }
    persistent.extend(pack_arena(backend, arena)?);
    let sizes = plan.place(starts, frame.program.operations.len());
    plan.pools = allocate_buffers(backend, &sizes)?;
    rebind(&mut frame.owners, &persistent)?;
    for wave in &mut frame.waves {
        rebind(&mut wave.owner_bindings, &persistent)?;
    }
    Ok(plan)
}

/// Place each plan-owned scratch value of `arena` (its placeholder, device,
/// first and last region, and bytes) at a byte range of its device's arena, reusing the
/// bytes of values whose spans ended before its own begins, and allocate the
/// arenas.
fn pack_arena(
    backend: &GpuDcrtBackend,
    arena: Vec<(*const (), i32, usize, usize, u64)>,
) -> Result<BTreeMap<*const (), BoundStorage>, String> {
    let items = arena
        .iter()
        .map(|&(_, device, _, _, bytes)| (device, bytes.div_ceil(ALIGNMENT) * ALIGNMENT))
        .collect::<Vec<_>>();
    let (offsets, sizes) = pack_intervals(&items, |left, right| {
        let ((_, _, left_start, left_end, _), (_, _, right_start, right_end, _)) =
            (arena[left], arena[right]);
        left_start <= right_end && right_start <= left_end
    });
    let placed = arena
        .iter()
        .zip(offsets)
        .map(|(&(pointer, device, _, _, bytes), offset)| (pointer, device, offset, bytes))
        .collect::<Vec<_>>();
    let buffers = allocate_buffers(backend, &sizes)?;
    placed
        .into_iter()
        .map(|(pointer, device, offset, bytes)| {
            let buffer = &buffers[&device];
            Ok((
                pointer,
                BoundStorage {
                    device,
                    address: buffer.as_ptr() as u64 + offset,
                    bytes,
                    owner: Arc::clone(buffer) as Arc<dyn std::any::Any + Send + Sync>,
                },
            ))
        })
        .collect()
}

/// Byte alignment of every value placed in a shared buffer.
const ALIGNMENT: u64 = 256;

/// Place each item of `items` (its device and bytes) at an offset of its
/// device's buffer, the largest first, each at the lowest offset clear of
/// every placed item of that device that `overlap`s it in time. Returns the
/// offsets and each device's buffer size.
fn pack_intervals(
    items: &[(i32, u64)],
    overlap: impl Fn(usize, usize) -> bool,
) -> (Vec<u64>, BTreeMap<i32, u64>) {
    let mut order = (0..items.len()).collect::<Vec<_>>();
    order.sort_by_key(|&item| (std::cmp::Reverse(items[item].1), item));
    let mut offsets = vec![0u64; items.len()];
    let mut placed = BTreeMap::<i32, Vec<usize>>::new();
    let mut sizes = BTreeMap::<i32, u64>::new();
    for item in order {
        let (device, size) = items[item];
        let on_device = placed.entry(device).or_default();
        let mut busy = on_device
            .iter()
            .filter(|&&other| overlap(item, other))
            .map(|&other| (offsets[other], offsets[other] + items[other].1))
            .collect::<Vec<_>>();
        busy.sort_unstable();
        let mut offset = 0u64;
        for (start, end) in busy {
            if start >= offset + size {
                break;
            }
            offset = offset.max(end);
        }
        offsets[item] = offset;
        on_device.push(item);
        let top = sizes.entry(device).or_default();
        *top = (*top).max(offset + size);
    }
    (offsets, sizes)
}

/// Allocate one device buffer of each size of `sizes`.
fn allocate_buffers(
    backend: &GpuDcrtBackend,
    sizes: &BTreeMap<i32, u64>,
) -> Result<BTreeMap<i32, Arc<GpuDeviceBuffer>>, String> {
    sizes
        .iter()
        .filter(|(_, bytes)| **bytes > 0)
        .map(|(&device, &bytes)| {
            let stream = backend
                .parameters_on_device(device)?
                .native_launch_stream(device)
                .map_err(|error| error.to_string())?;
            let size = usize::try_from(bytes).map_err(|_| "GPU scratch buffer exceeds usize")?;
            let buffer =
                GpuDeviceBuffer::allocate(&stream, size).map_err(|error| error.to_string())?;
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
    fn place(&mut self, starts: &[usize], operations: usize) -> BTreeMap<i32, u64> {
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
        let items = self
            .allocations
            .iter()
            .map(|allocation| {
                (allocation.device, (allocation.bytes as u64).div_ceil(ALIGNMENT) * ALIGNMENT)
            })
            .collect::<Vec<_>>();
        let (offsets, sizes) =
            pack_intervals(&items, |left, right| !after(left, right) && !after(right, left));
        for (allocation, offset) in self.allocations.iter_mut().zip(offsets) {
            allocation.offset = offset;
        }
        sizes
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
                lanes.iter_mut().for_each(|lane| *lane = None);
            }
        }
        if let Some((end, _, _)) = &self.open_loop &&
            index >= *end
        {
            let (_, created, _) = self.open_loop.take().expect("the loop is open");
            self.chain.extend(created);
        }
        while let Some(range) = self.loops.get(self.next_loop) &&
            range.start <= index
        {
            if self.open_loop.is_none() && index < range.end {
                let lanes = self.lanes[&(range.start, range.end)].len();
                self.open_loop = Some((range.end, Vec::new(), vec![None; lanes]));
            }
            self.next_loop += 1;
        }
    }

    /// The memory nodes the next memory node of `allocation` follows: its
    /// lane's chain inside the open loop, otherwise `self.chain`.
    fn chain_of(&self, allocation: usize) -> &[u32] {
        match (&self.open_loop, self.allocations[allocation].lane) {
            (Some((_, _, lanes)), Some(lane)) => lanes[lane].as_deref().unwrap_or(&self.chain),
            _ => &self.chain,
        }
    }

    /// Record a memory node of `allocation` created after `chain_of`.
    fn push_memory_node(&mut self, allocation: usize, token: u32) {
        let lane = self.allocations[allocation].lane;
        match &mut self.open_loop {
            Some((_, created, lanes)) => {
                created.push(token);
                if let Some(lane) = lane {
                    lanes[lane] = Some(vec![token]);
                }
            }
            None => self.chain = vec![token],
        }
    }

    /// Bind the scratch whose first use is top-level operation `index` into
    /// `frame`, and return the allocation-barrier tokens that operation must
    /// follow.
    pub(crate) fn before_operation(
        &mut self,
        builder: &mut GpuNativeGraphBuilder,
        frame: &mut PhysicalFrame,
        start: usize,
        index: usize,
    ) -> Result<Vec<u32>, GpuNativeGraphError> {
        let invalid = |message: &str| GpuNativeGraphError::Native(message.into());
        self.enter_operation(start, index);
        for allocation in self.allocate_before.get(&index).cloned().unwrap_or_default() {
            let token = builder.add_memory_barrier(&[], self.chain_of(allocation))?;
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

    /// Add the free barrier of the scratch whose last use is top-level
    /// operation `index` of the region starting at `start`, after every
    /// emitted operation of that region that uses it. Uses in earlier regions
    /// have ended: this region launches after them.
    pub(crate) fn after_operation(
        &mut self,
        builder: &mut GpuNativeGraphBuilder,
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
            let token = builder.add_memory_barrier(&readers, self.chain_of(index_of))?;
            self.push_memory_node(index_of, token);
        }
        Ok(())
    }
}
