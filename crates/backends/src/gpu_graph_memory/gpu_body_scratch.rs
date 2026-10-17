//! Reuse body-local scratch with explicit dependencies between overlapping byte ranges.
//!
//! CUDA conditional bodies cannot contain allocation nodes. Their temporary values
//! can nevertheless share a preallocated range. A later first write waits for every
//! last reader of the earlier value using those bytes. Only forward edges are added;
//! operations using different ranges remain independent. This is the body equivalent
//! of the outer graph scratch allocation barriers.
use super::*;

type Path = Vec<usize>;
type Pointer = *const ();

fn paths(
    frame: &PhysicalFrame,
    operations: &[CompiledGpuOp],
    prefix: &mut Path,
    uses: &mut BTreeMap<PhysicalValueId, BTreeSet<(Path, bool)>>,
    bodies: &mut Vec<Path>,
) -> Result<(), String> {
    for (index, operation) in operations.iter().enumerate() {
        prefix.push(index);
        visit_node(frame, operation, &mut |id, written| {
            uses.entry(id).or_default().insert((prefix.clone(), written));
        })?;
        if let Some(body) = &operation.body {
            bodies.push(prefix.clone());
            paths(frame, body, prefix, uses, bodies)?;
        }
        prefix.pop();
    }
    Ok(())
}

struct Group {
    bound: BoundStorage,
    uses: BTreeSet<(Path, bool)>,
    eligible: bool,
}

/// Coalesce innermost bodies first, allowing their pools to share storage in an
/// enclosing body too. Captures, carried values, and host-bound allocations are
/// excluded whenever any use escapes the body being considered.
pub(super) fn coalesce(frame: &mut PhysicalFrame) -> Result<(), String> {
    let mut uses = BTreeMap::new();
    let mut bodies = Vec::new();
    paths(frame, &frame.program.operations, &mut Vec::new(), &mut uses, &mut bodies)?;
    bodies.sort_by_key(|path| std::cmp::Reverse(path.len()));
    let mut protected = frame.scratch_protected.clone();
    for wave in &frame.waves {
        protected.extend(wave.owner_bindings.keys().copied());
    }
    protected.extend(frame.import_templates.iter().map(|import| import.destination));
    protected.extend(frame.external_io_imports.iter().map(|import| import.destination));
    for body in &frame.external_io_loops {
        protected.extend(body.imports.iter().map(|import| import.destination));
        protected.extend(body.ahead.iter().map(|import| import.template.destination));
    }
    // Index value uses once. Only values wholly contained in a body can be
    // candidates there; aliases are checked through the owner's allocation index.
    let mut local_values = BTreeMap::<Path, BTreeSet<PhysicalValueId>>::new();
    for (&id, value_uses) in &uses {
        let Some((first, _)) = value_uses.first() else { continue };
        let common = value_uses
            .iter()
            .map(|(other, _)| {
                first
                    .iter()
                    .zip(other)
                    .take_while(|(a, b)| a == b)
                    .count()
                    .min(other.len().saturating_sub(1))
            })
            .min()
            .unwrap_or(0)
            .min(first.len().saturating_sub(1));
        for depth in 1..=common {
            local_values.entry(first[..depth].to_vec()).or_default().insert(id);
        }
    }
    let eligible = |id: PhysicalValueId, owner: &GpuResidentValue| {
        !protected.contains(&id) &&
            owner.physical().encodings.iter().all(|encoding| {
                matches!(
                    encoding,
                    PhysicalEncoding::FullCoeff |
                        PhysicalEncoding::FullEval |
                        PhysicalEncoding::CompactCoeff { .. } |
                        PhysicalEncoding::CompactCoeffPerCrtLimb { .. }
                )
            })
    };
    let mut values = frame.program.values.to_vec();
    for path in bodies {
        let Some(local) = local_values.get(&path) else { continue };
        let mut operations = frame.program.operations.as_ref();
        for &index in &path {
            operations = operations[index].body.as_deref().ok_or("scratch body path is invalid")?;
        }
        let mut ancestors = Vec::<BTreeSet<usize>>::with_capacity(operations.len());
        for (index, operation) in operations.iter().enumerate() {
            let mut earlier = BTreeSet::new();
            for &predecessor in &operation.predecessors {
                let predecessor = predecessor as usize;
                if predecessor >= index {
                    return Err("scratch body dependency is not topological".into());
                }
                earlier.insert(predecessor);
                earlier.extend(ancestors[predecessor].iter().copied());
            }
            ancestors.push(earlier);
        }
        let mut groups = BTreeMap::<Pointer, Group>::new();
        for &id in local {
            let Some(owner) = frame.owners.get(&id) else { continue };
            let eligible = eligible(id, owner);
            for (_, bound) in owner.storages() {
                if !is_graph_managed(bound) {
                    continue;
                }
                let pointer = Arc::as_ptr(&bound.owner).cast::<()>();
                let group = groups.entry(pointer).or_insert_with(|| Group {
                    bound: bound.clone(),
                    uses: BTreeSet::new(),
                    eligible: true,
                });
                // Partial allocation views need a separate base-normalization proof.
                group.eligible &= eligible &&
                    bound.address == group.bound.address &&
                    bound.bytes == group.bound.bytes &&
                    bound.device == group.bound.device;
                if let Some(value_uses) = uses.get(&id) {
                    group.uses.extend(value_uses.iter().cloned());
                }
            }
        }
        for (&pointer, group) in &mut groups {
            for id in frame.owners.viewing(pointer) {
                let owner = &frame.owners[&id];
                group.eligible &=
                    eligible(id, owner) && (local.contains(&id) || !uses.contains_key(&id));
                group.eligible &= owner
                    .storages()
                    .filter(|(_, bound)| Arc::as_ptr(&bound.owner).cast::<()>() == pointer)
                    .all(|(_, bound)| {
                        bound.address == group.bound.address &&
                            bound.bytes == group.bound.bytes &&
                            bound.device == group.bound.device
                    });
            }
        }
        let mut candidates = groups
            .into_iter()
            .filter_map(|(pointer, group)| {
                if !group.eligible ||
                    group.uses.is_empty() ||
                    group.uses.iter().any(|(use_path, _)| {
                        use_path.len() <= path.len() || !use_path.starts_with(&path)
                    })
                {
                    return None;
                }
                let references =
                    group.uses.iter().map(|(p, _)| p[path.len()]).collect::<BTreeSet<_>>();
                let first_write = group
                    .uses
                    .iter()
                    .filter(|(_, write)| *write)
                    .map(|(p, _)| p[path.len()])
                    .min()?;
                if references
                    .iter()
                    .any(|&index| index != first_write && !ancestors[index].contains(&first_write))
                {
                    return None;
                }
                Some((pointer, group.bound, references, first_write))
            })
            .collect::<Vec<_>>();
        // The placeholder address follows deferred-allocation creation order;
        // unlike the Arc pointer it is independent of host allocator reuse.
        candidates.sort_by_key(|(_, bound, _, first_write)| (*first_write, bound.address));
        if candidates.len() < 2 {
            continue;
        }
        let items = candidates
            .iter()
            .map(|(_, bound, _, _)| (bound.device, bound.bytes.div_ceil(ALIGNMENT) * ALIGNMENT))
            .collect::<Vec<_>>();
        let lifetimes =
            candidates.iter().map(|(_, _, uses, first)| (uses.clone(), *first)).collect::<Vec<_>>();
        let (offsets, sizes, dependencies) = place_body(&items, &lifetimes, &ancestors);
        let mut operations = frame.program.operations.as_mut();
        for &index in &path {
            operations =
                operations[index].body.as_deref_mut().ok_or("scratch body path is invalid")?;
        }
        for (write, readers) in dependencies {
            let mut predecessors =
                operations[write].predecessors.iter().copied().collect::<BTreeSet<_>>();
            for reader in readers {
                predecessors
                    .insert(u32::try_from(reader).map_err(|_| "scratch body reader exceeds u32")?);
            }
            operations[write].predecessors = predecessors.into_iter().collect();
        }
        let mut pools = BTreeMap::new();
        for (&device, &bytes) in &sizes {
            let ty = candidates
                .iter()
                .find(|(_, bound, _, _)| bound.device == device)
                .and_then(|(_, bound, _, _)| bound.owner.downcast_ref::<GpuDeferredScratch>())
                .ok_or("scratch body has no deferred type")?
                .ty
                .clone();
            pools.insert(device, deferred_scratch_storage(device, ty, bytes));
        }
        let replacements = candidates
            .iter()
            .zip(offsets)
            .map(|((pointer, bound, _, _), offset)| {
                (*pointer, (pools[&bound.device].clone(), offset))
            })
            .collect::<BTreeMap<_, _>>();
        let changed = replacements
            .keys()
            .flat_map(|&pointer| frame.owners.viewing(pointer))
            .collect::<BTreeSet<_>>();
        for id in changed {
            let owner = Arc::clone(&frame.owners[&id]);
            let mut physical = (**owner.physical()).clone();
            let mut storage = owner
                .storages()
                .map(|(slot, bound)| (*slot, bound.clone()))
                .collect::<BTreeMap<_, _>>();
            for part in physical.parts.iter_mut() {
                let bound = owner.storage(part.storage).ok_or("scratch part has no storage")?;
                if let Some((_, offset)) = replacements.get(&Arc::as_ptr(&bound.owner).cast::<()>())
                {
                    part.view.byte_offset = part
                        .view
                        .byte_offset
                        .checked_add(*offset)
                        .ok_or("scratch body offset overflow")?;
                }
            }
            for bound in storage.values_mut() {
                if let Some((replacement, _)) =
                    replacements.get(&Arc::as_ptr(&bound.owner).cast::<()>())
                {
                    *bound = replacement.clone();
                }
            }
            values[id.0 as usize] = physical.clone();
            let owner = GpuResidentValue::new(
                Arc::new(physical),
                storage,
                owner.ready_events().to_vec().into_boxed_slice(),
            )
            .map_err(str::to_owned)?;
            frame.owners.insert(id, Arc::new(owner));
        }
    }
    frame.program.values = values.into();
    Ok(())
}

/// Pack topologically disjoint lifetimes, then order only accesses that reuse
/// bytes. Keep the existing-DAG placement on devices where the alternate greedy
/// placement does not save space; no new dependencies are needed there.
fn place_body(
    items: &[(i32, u64)],
    lifetimes: &[(BTreeSet<usize>, usize)],
    ancestors: &[BTreeSet<usize>],
) -> (Vec<u64>, BTreeMap<i32, u64>, BTreeMap<usize, BTreeSet<usize>>) {
    let precedes = |a: usize, b: usize| {
        lifetimes[a].0.iter().all(|index| ancestors[lifetimes[b].1].contains(index))
    };
    let ordered =
        |a: usize, b: usize| lifetimes[a].0.last().is_some_and(|&last| last < lifetimes[b].1);
    let (old_offsets, old_sizes) = pack_intervals(items, |a, b| !precedes(a, b) && !precedes(b, a));
    let (mut offsets, mut sizes) = pack_intervals(items, |a, b| !ordered(a, b) && !ordered(b, a));
    let mut changed = BTreeSet::new();
    for (&device, size) in &mut sizes {
        if *size < old_sizes[&device] {
            changed.insert(device);
        } else {
            *size = old_sizes[&device];
            for (index, &(owner, _)) in items.iter().enumerate() {
                if owner == device {
                    offsets[index] = old_offsets[index];
                }
            }
        }
    }
    let mut dependencies = BTreeMap::<usize, BTreeSet<usize>>::new();
    for earlier in 0..items.len() {
        if !changed.contains(&items[earlier].0) {
            continue;
        }
        for later in 0..items.len() {
            if items[earlier].0 != items[later].0 ||
                !ordered(earlier, later) ||
                offsets[earlier] >= offsets[later] + items[later].1 ||
                offsets[later] >= offsets[earlier] + items[earlier].1
            {
                continue;
            }
            let write = lifetimes[later].1;
            for &reader in &lifetimes[earlier].0 {
                // Terminal readers cover their ancestors, including the producer.
                if !ancestors[write].contains(&reader) &&
                    !lifetimes[earlier].0.iter().any(|&other| ancestors[other].contains(&reader))
                {
                    dependencies.entry(write).or_default().insert(reader);
                }
            }
        }
    }
    (offsets, sizes, dependencies)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_body_scratch_reuse_orders_all_readers() {
        // Two unordered readers of A must both finish before B overwrites A.
        let ancestors = vec![
            BTreeSet::new(),
            BTreeSet::new(),
            BTreeSet::from([0]),
            BTreeSet::from([0]),
            BTreeSet::new(),
            BTreeSet::from([4]),
        ];
        let lifetimes = [(BTreeSet::from([0, 2, 3]), 0), (BTreeSet::from([4, 5]), 4)];
        let (offsets, sizes, dependencies) =
            place_body(&[(0, 256), (0, 256)], &lifetimes, &ancestors);
        assert_eq!(offsets, [0, 0]);
        assert_eq!(sizes[&0], 256);
        assert_eq!(dependencies, BTreeMap::from([(4, BTreeSet::from([2, 3]))]));

        // A still live after B's first write cannot share B's range.
        let overlapping = [(BTreeSet::from([0, 2, 5]), 0), (BTreeSet::from([4, 5]), 4)];
        let (_, sizes, dependencies) = place_body(&[(0, 256), (0, 256)], &overlapping, &ancestors);
        assert_eq!(sizes[&0], 512);
        assert!(dependencies.is_empty());
        let (_, sizes, dependencies) = place_body(&[(0, 256), (1, 256)], &lifetimes, &ancestors);
        assert_eq!(sizes, BTreeMap::from([(0, 256), (1, 256)]));
        assert!(dependencies.is_empty());
    }
}
