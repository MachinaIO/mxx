//! Warmup-only instantiation of structural scopes into fixed wire bindings.

use super::*;

pub(super) fn instantiate(
    graph: &ValidatedGraph,
    wave_bound: std::num::NonZeroUsize,
) -> Result<GpuPreparation, PreparedLoweringError> {
    let mut compiler = ScopeInstantiator {
        graph,
        program: GpuPreparation { instance_count: 1, ..GpuPreparation::default() },
        nodes: Vec::new(),
        families: BTreeMap::new(),
        next_node: 1,
        replay: Vec::new(),
        variant_environments: Vec::new(),
        wave_bound,
        next_virtual_wire: u64::MAX,
    };
    let root = compiler.scope(&FrozenGraphScopeId::Root, &graph.bindings, &[])?;
    compiler.program.trace_keys = root
        .iter()
        .map(|(source, prepared)| {
            (
                *prepared,
                mxx_ir_core::types::WireId { instantiation_path: Vec::new(), wire: *source },
            )
        })
        .collect();
    compiler.program.inputs = compiler
        .nodes
        .iter()
        .filter(|node| {
            matches!(
                compiler.program.node_sources[&node.id].kind,
                NodeKind::Input { artifact: None, .. }
            )
        })
        .flat_map(|node| compiler.program.node_bindings[&node.id].1.iter().copied())
        .collect();
    compiler.program.runtime_input_wires = compiler
        .program
        .inputs
        .iter()
        .flat_map(|wire| family_leaf_wires(&compiler.families, *wire))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    compiler.program.runtime_input_roots = compiler
        .program
        .runtime_input_wires
        .iter()
        .map(|wire| {
            compiler
                .program
                .inputs
                .iter()
                .position(|root| family_leaf_wires(&compiler.families, *root).contains(wire))
                .unwrap_or_else(|| {
                    compiler
                        .program
                        .inputs
                        .iter()
                        .position(|root| root == wire)
                        .expect("runtime input leaf has no root")
                })
        })
        .collect();
    compiler.program.family_wires = compiler.families.clone();
    compiler.program.input_names = compiler
        .program
        .inputs
        .iter()
        .filter_map(|wire| {
            let NodeKind::Input { name, .. } =
                &compiler.program.node_sources[&(wire.node.0 as u32)].kind
            else {
                return None
            };
            Some((name.clone(), *wire))
        })
        .collect();
    compiler.program.outputs =
        graph.source.root_scope().outputs().iter().map(|wire| root[wire]).collect();
    compiler.program.trace_wires = compiler
        .program
        .node_bindings
        .values()
        .flat_map(|(_, outputs)| outputs.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .into_boxed_slice();
    compiler.program.output_names = graph
        .source
        .outputs()
        .iter()
        .map(|(name, output)| (name.clone(), root[&output.value]))
        .collect();
    compiler.program.output_bindings = compiler
        .program
        .output_names
        .iter()
        .map(|(name, wire)| {
            let kind = match &compiler.program.wire_types[wire] {
                ConcreteWireType::Matrix(_) | ConcreteWireType::Trapdoor { .. } => {
                    PreparedOutputKind::Matrix
                }
                ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => {
                    PreparedOutputKind::SmallMatrix
                }
                ConcreteWireType::IndexedFamily { .. }
                    if matches!(
                        compiler.program.node_sources[&(wire.node.0 as u32)].kind,
                        NodeKind::PolynomialValues { .. }
                    ) =>
                {
                    PreparedOutputKind::HostReconstruction
                }
                ConcreteWireType::IndexedFamily { .. } => PreparedOutputKind::Family,
                _ => PreparedOutputKind::Scalar,
            };
            PreparedOutputBinding { name: name.clone(), wire: *wire, kind }
        })
        .collect();
    compiler.program.bindings = compiler
        .program
        .values
        .iter()
        .map(|(wire, location)| {
            (
                *wire,
                PreparedBindingId {
                    owner: location.owner,
                    device: location.device,
                    instance: 0,
                    storage: None,
                },
            )
        })
        .collect();
    compiler.program.topology.edges = compiler
        .nodes
        .iter()
        .flat_map(|node| {
            node.waits.iter().map(|event| PreparedTopologyEdge {
                from: *event,
                to: node.id,
                event: *event,
            })
        })
        .collect();
    compiler.program.topology.nodes = compiler.nodes.into_boxed_slice();
    compiler.program.replay = compiler.replay.into_boxed_slice();
    color_storage(&mut compiler.program);
    Ok(compiler.program)
}

fn family_leaf_wires(families: &BTreeMap<WireRef, Box<[WireRef]>>, wire: WireRef) -> Vec<WireRef> {
    let Some(members) = families.get(&wire) else { return vec![wire] };
    members.iter().flat_map(|member| family_leaf_wires(families, *member)).collect()
}

/// Assign physical owners after lowering has fixed the complete topology.
///
/// Reuse is valid only when every access to the previous owner happens-before
/// the next owner's first write. Reuse also requires the complete fixed
/// storage descriptor to match.
pub(super) fn color_storage(program: &mut GpuPreparation) {
    #[derive(Clone)]
    struct StorageLifetime {
        owner: u64,
        order: usize,
        rows: usize,
        columns: usize,
        level: usize,
        format: PreparedFormat,
        descriptor_compatible: bool,
        device: i32,
        reads: Vec<usize>,
        writes: Vec<usize>,
        pinned: bool,
    }

    // Alias views do not own storage. Resolve their owner to the source owner
    // before collecting accesses, so a chain of slices/transposes cannot create
    // duplicate physical allocations.
    let mut parent = BTreeMap::<u64, u64>::new();
    let mut ensure_owner = |owner: u64| {
        parent.entry(owner).or_insert(owner);
    };
    for location in program.values.values() {
        ensure_owner(location.owner);
    }
    for view in program.view_commands.values() {
        match view {
            PreparedView::Alias(location) | PreparedView::TransposeAlias(location) => {
                ensure_owner(location.owner)
            }
            PreparedView::FixedCopies(copies) => {
                for copy in copies {
                    ensure_owner(copy.source.owner);
                    ensure_owner(copy.destination.owner);
                }
            }
        }
    }
    for selection in program.selection_commands.values() {
        match selection {
            PreparedSelection::Static { location } => ensure_owner(location.owner),
            PreparedSelection::Dynamic { candidates, .. } |
            PreparedSelection::Select { candidates, .. } => {
                for location in candidates {
                    ensure_owner(location.owner);
                }
            }
            PreparedSelection::ScalarStatic { .. } |
            PreparedSelection::ScalarDynamic { .. } |
            PreparedSelection::ScalarSelect { .. } => {}
        }
    }
    for members in program.family_members.values() {
        for location in members {
            ensure_owner(location.owner);
        }
    }
    fn root(parent: &mut BTreeMap<u64, u64>, owner: u64) -> u64 {
        let parent_owner = parent[&owner];
        if parent_owner == owner {
            owner
        } else {
            let root_owner = root(parent, parent_owner);
            parent.insert(owner, root_owner);
            root_owner
        }
    }
    let mut union = |left: u64, right: u64| {
        let left = root(&mut parent, left);
        let right = root(&mut parent, right);
        if left != right {
            parent.insert(left, right);
        }
    };
    for (node_id, view) in &program.view_commands {
        if !matches!(view, PreparedView::Alias(_) | PreparedView::TransposeAlias(_)) {
            continue;
        }
        let Some((_, outputs)) = program.node_bindings.get(node_id) else { continue };
        let Some(output) = outputs.first().and_then(|wire| program.values.get(wire)) else {
            continue
        };
        let source = match view {
            PreparedView::Alias(source) | PreparedView::TransposeAlias(source) => source,
            PreparedView::FixedCopies(_) => unreachable!(),
        };
        union(output.owner, source.owner);
    }
    for (node_id, selection) in &program.selection_commands {
        let PreparedSelection::Static { location } = selection else { continue };
        let Some((_, outputs)) = program.node_bindings.get(node_id) else { continue };
        let Some(output) = outputs.first().and_then(|wire| program.values.get(wire)) else {
            continue
        };
        union(output.owner, location.owner);
    }
    let canonical = |owner: u64, parent: &mut BTreeMap<u64, u64>| root(parent, owner);
    for location in program.values.values_mut() {
        location.owner = canonical(location.owner, &mut parent);
    }
    for view in program.view_commands.values_mut() {
        match view {
            PreparedView::Alias(location) | PreparedView::TransposeAlias(location) => {
                location.owner = canonical(location.owner, &mut parent)
            }
            PreparedView::FixedCopies(copies) => {
                for copy in copies {
                    copy.source.owner = canonical(copy.source.owner, &mut parent);
                    copy.destination.owner = canonical(copy.destination.owner, &mut parent);
                }
            }
        }
    }
    for selection in program.selection_commands.values_mut() {
        match selection {
            PreparedSelection::Static { location } => {
                location.owner = canonical(location.owner, &mut parent)
            }
            PreparedSelection::Dynamic { candidates, .. } |
            PreparedSelection::Select { candidates, .. } => {
                for location in candidates {
                    location.owner = canonical(location.owner, &mut parent);
                }
            }
            PreparedSelection::ScalarStatic { .. } |
            PreparedSelection::ScalarDynamic { .. } |
            PreparedSelection::ScalarSelect { .. } => {}
        }
    }
    for members in program.family_members.values_mut() {
        for location in members {
            location.owner = canonical(location.owner, &mut parent);
        }
    }

    let node_count = program.topology.nodes.len();
    let node_devices = program
        .topology
        .nodes
        .iter()
        .map(|node| {
            let mut devices = BTreeSet::new();
            if let Some((arguments, outputs)) = program.node_bindings.get(&node.id) {
                for wire in arguments.iter().chain(outputs.iter()) {
                    if let Some(location) = program.values.get(wire) {
                        devices.insert(location.device);
                    }
                    if let Some(members) = program.family_members.get(wire) {
                        devices.extend(members.iter().map(|location| location.device));
                    }
                }
            }
            devices
        })
        .collect::<Vec<_>>();
    let mut completion_to_node = BTreeMap::<u32, usize>::new();
    for (index, node) in program.topology.nodes.iter().enumerate() {
        completion_to_node.insert(node.id, index);
        completion_to_node.insert(node.completion, index);
    }
    let mut predecessors = vec![BTreeSet::<usize>::new(); node_count];
    for (index, node) in program.topology.nodes.iter().enumerate() {
        for wait in node.waits.iter().copied() {
            if let Some(previous) = completion_to_node.get(&wait) {
                predecessors[index].insert(*previous);
            }
        }
    }
    for edge in &program.topology.edges {
        if let (Some(from), Some(to)) =
            (completion_to_node.get(&edge.from), completion_to_node.get(&edge.to))
        {
            predecessors[*to].insert(*from);
        }
    }
    let mut indegree = predecessors.iter().map(BTreeSet::len).collect::<Vec<_>>();
    let mut ready = (0..node_count).filter(|index| indegree[*index] == 0).collect::<BTreeSet<_>>();
    let mut order = Vec::with_capacity(node_count);
    while let Some(index) = ready.pop_first() {
        order.push(index);
        for (successor, dependencies) in predecessors.iter().enumerate() {
            if dependencies.contains(&index) {
                indegree[successor] -= 1;
                if indegree[successor] == 0 {
                    ready.insert(successor);
                }
            }
        }
    }
    if order.len() != node_count {
        // A malformed topology must not make two live regions alias. Leave all
        // owners distinct; validation reports the topology error elsewhere.
        return;
    }
    let mut ancestors = vec![BTreeSet::<usize>::new(); node_count];
    for index in order {
        for predecessor in &predecessors[index] {
            ancestors[index].insert(*predecessor);
            let inherited = ancestors[*predecessor].clone();
            ancestors[index].extend(inherited);
        }
    }
    let happens_before = |from: usize, to: usize| ancestors[to].contains(&from);

    let mut lifetimes = BTreeMap::<u64, StorageLifetime>::new();
    let mut next_order = 0;
    let mut register = |location: &ValueLocation,
                        lifetimes: &mut BTreeMap<u64, StorageLifetime>| {
        let owner = canonical(location.owner, &mut parent);
        let entry = lifetimes.entry(owner).or_insert_with(|| {
            let order = next_order;
            next_order += 1;
            StorageLifetime {
                owner,
                order,
                rows: 0,
                columns: 0,
                level: location.level,
                format: location.format,
                descriptor_compatible: true,
                device: location.device,
                reads: Vec::new(),
                writes: Vec::new(),
                pinned: false,
            }
        });
        entry.rows = entry.rows.max(location.rows.end);
        entry.columns = entry.columns.max(location.columns.end);
        entry.descriptor_compatible &= entry.level == location.level &&
            entry.format == location.format &&
            entry.device == location.device;
        entry.device = location.device;
    };
    for location in program.values.values() {
        register(location, &mut lifetimes);
    }
    for view in program.view_commands.values() {
        match view {
            PreparedView::Alias(location) | PreparedView::TransposeAlias(location) => {
                register(location, &mut lifetimes)
            }
            PreparedView::FixedCopies(copies) => {
                for copy in copies {
                    register(&copy.source, &mut lifetimes);
                    register(&copy.destination, &mut lifetimes);
                }
            }
        }
    }
    for selection in program.selection_commands.values() {
        match selection {
            PreparedSelection::Static { location } => register(location, &mut lifetimes),
            PreparedSelection::Dynamic { candidates, .. } |
            PreparedSelection::Select { candidates, .. } => {
                for location in candidates {
                    register(location, &mut lifetimes);
                }
            }
            PreparedSelection::ScalarStatic { .. } |
            PreparedSelection::ScalarDynamic { .. } |
            PreparedSelection::ScalarSelect { .. } => {}
        }
    }
    for members in program.family_members.values() {
        for location in members {
            register(location, &mut lifetimes);
        }
    }
    let mut input_owners = BTreeSet::new();
    for (node_index, node) in program.topology.nodes.iter().enumerate() {
        let Some((arguments, outputs)) = program.node_bindings.get(&node.id) else { continue };
        let input_node = matches!(
            program.node_sources.get(&node.id).map(|source| &source.kind),
            Some(NodeKind::Input { .. })
        );
        for wire in arguments {
            if let Some(location) = program.values.get(wire) {
                let owner = canonical(location.owner, &mut parent);
                lifetimes.get_mut(&owner).unwrap().reads.push(node_index);
            }
            if let Some(members) = program.family_members.get(wire) {
                for location in members {
                    let owner = canonical(location.owner, &mut parent);
                    lifetimes.get_mut(&owner).unwrap().reads.push(node_index);
                }
            }
        }
        for wire in outputs {
            if let Some(location) = program.values.get(wire) {
                let owner = canonical(location.owner, &mut parent);
                if input_node {
                    input_owners.insert(owner);
                }
                let is_alias = matches!(node.command.operation, PreparedOperation::Alias) ||
                    matches!(
                        program.selection_commands.get(&node.id),
                        Some(PreparedSelection::Static { .. })
                    );
                if !is_alias {
                    lifetimes.get_mut(&owner).unwrap().writes.push(node_index);
                }
            }
            if let Some(members) = program.family_members.get(wire) {
                for location in members {
                    let owner = canonical(location.owner, &mut parent);
                    lifetimes.get_mut(&owner).unwrap().reads.push(node_index);
                }
            }
        }
    }
    for owner in input_owners {
        lifetimes.get_mut(&owner).unwrap().pinned = true;
    }
    for wire in program.trace_wires.iter().chain(program.outputs.iter()) {
        if let Some(location) = program.values.get(wire) {
            lifetimes.get_mut(&canonical(location.owner, &mut parent)).unwrap().pinned = true;
        }
        if let Some(members) = program.family_members.get(wire) {
            for location in members {
                lifetimes.get_mut(&canonical(location.owner, &mut parent)).unwrap().pinned = true;
            }
        }
    }
    let accesses_by_owner = lifetimes
        .values()
        .map(|lifetime| {
            let mut accesses = lifetime.reads.clone();
            accesses.extend_from_slice(&lifetime.writes);
            (lifetime.owner, accesses)
        })
        .collect::<BTreeMap<_, _>>();
    let mut lifetimes = lifetimes.into_values().collect::<Vec<_>>();
    lifetimes.sort_by_key(|lifetime| {
        (lifetime.writes.iter().copied().min().unwrap_or(usize::MAX), lifetime.order)
    });
    struct StorageColor {
        rows: usize,
        columns: usize,
        order: usize,
        level: usize,
        format: PreparedFormat,
        device: i32,
        owner: u64,
        reusable: bool,
    }
    let mut colors = Vec::<StorageColor>::new();
    let mut remap = BTreeMap::new();
    for lifetime in lifetimes {
        let first_write = lifetime.writes.iter().copied().min();
        let color = if lifetime.pinned || first_write.is_none() {
            let color = colors.len();
            colors.push(StorageColor {
                rows: lifetime.rows,
                columns: lifetime.columns,
                order: lifetime.order,
                level: lifetime.level,
                format: lifetime.format,
                device: lifetime.device,
                owner: lifetime.owner,
                reusable: false,
            });
            color
        } else {
            let first_write = first_write.unwrap();
            let candidate = colors
                .iter()
                .enumerate()
                .filter(|(_, color)| {
                    color.reusable &&
                        color.device == lifetime.device &&
                        color.level == lifetime.level &&
                        color.format == lifetime.format &&
                        lifetime.descriptor_compatible &&
                        lifetime.rows <= color.rows &&
                        lifetime.columns <= color.columns &&
                        // The owner stored in a color is the last writer. Its
                        // every access must happen before this first write.
                        accesses_by_owner[&color.owner].iter().all(|access| {
                            node_devices[*access].iter().all(|device| *device == lifetime.device) &&
                                happens_before(*access, first_write)
                        })
                })
                .map(|(index, color)| {
                    (index, color.rows.saturating_mul(color.columns), color.order)
                })
                .min_by_key(|(_, capacity, order)| (*capacity, *order));
            if let Some((color, ..)) = candidate {
                colors[color].rows = colors[color].rows.max(lifetime.rows);
                colors[color].columns = colors[color].columns.max(lifetime.columns);
                colors[color].owner = lifetime.owner;
                color
            } else {
                let color = colors.len();
                colors.push(StorageColor {
                    rows: lifetime.rows,
                    columns: lifetime.columns,
                    order: lifetime.order,
                    level: lifetime.level,
                    format: lifetime.format,
                    device: lifetime.device,
                    owner: lifetime.owner,
                    reusable: lifetime.descriptor_compatible,
                });
                color
            }
        };
        remap.insert(lifetime.owner, color as u64);
    }
    let apply = |location: &mut ValueLocation| {
        if let Some(owner) = remap.get(&location.owner) {
            location.owner = *owner;
        }
    };
    for location in program.values.values_mut() {
        apply(location);
    }
    for view in program.view_commands.values_mut() {
        match view {
            PreparedView::Alias(location) | PreparedView::TransposeAlias(location) => {
                apply(location)
            }
            PreparedView::FixedCopies(copies) => {
                for copy in copies {
                    apply(&mut copy.source);
                    apply(&mut copy.destination);
                }
            }
        }
    }
    for selection in program.selection_commands.values_mut() {
        match selection {
            PreparedSelection::Static { location } => apply(location),
            PreparedSelection::Dynamic { candidates, .. } |
            PreparedSelection::Select { candidates, .. } => {
                for location in candidates {
                    apply(location);
                }
            }
            PreparedSelection::ScalarStatic { .. } |
            PreparedSelection::ScalarDynamic { .. } |
            PreparedSelection::ScalarSelect { .. } => {}
        }
    }
    for members in program.family_members.values_mut() {
        for location in members {
            apply(location);
        }
    }
    program.storage_capacities = colors
        .iter()
        .enumerate()
        .map(|(index, color)| (index as u64, (color.rows, color.columns)))
        .collect();
    for (wire, binding) in &mut program.bindings {
        if let Some(location) = program.values.get(wire) {
            binding.owner = location.owner;
        }
    }
}

struct ScopeInstantiator<'a> {
    graph: &'a ValidatedGraph,
    program: GpuPreparation,
    nodes: Vec<PreparedTopologyNode>,
    families: BTreeMap<WireRef, Box<[WireRef]>>,
    next_node: u32,
    replay: Vec<PreparedReplayStep>,
    variant_environments: Vec<ParamEnv>,
    wave_bound: std::num::NonZeroUsize,
    next_virtual_wire: u64,
}

impl ScopeInstantiator<'_> {
    fn scope(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        environment: &ParamEnv,
        arguments: &[WireRef],
    ) -> Result<BTreeMap<WireRef, WireRef>, PreparedLoweringError> {
        let source = self.graph.source.scope(scope_id).ok_or(
            PreparedLoweringError::InvalidContract("prepared source scope is unavailable"),
        )?;
        let validated = self.graph.scope(scope_id).ok_or(
            PreparedLoweringError::InvalidContract("prepared validated scope is unavailable"),
        )?;
        let mut wires = BTreeMap::new();
        if *scope_id != FrozenGraphScopeId::Root {
            if source.inputs().len() != arguments.len() {
                return Err(PreparedLoweringError::InvalidContract("prepared child arity mismatch"));
            }
            wires.extend(source.inputs().iter().copied().zip(arguments.iter().copied()));
        }
        for node in &validated.execution_order {
            let id = source.node_id(node).expect("validated scope node");
            let local_outputs = (0..node.output_types().len())
                .map(|port| WireRef { node: id, port: Port(port as u32) })
                .collect::<Vec<_>>();
            if local_outputs.iter().all(|wire| wires.contains_key(wire)) {
                continue;
            }
            let arguments = source
                .arguments(node)
                .expect("validated scope arguments")
                .iter()
                .map(|wire| {
                    wires.get(wire).copied().ok_or(PreparedLoweringError::InvalidContract(
                        "prepared child argument is unbound",
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let types = local_outputs
                .iter()
                .map(|wire| {
                    self.graph.concrete_wire_type(scope_id, *wire, environment).map_err(|_| {
                        PreparedLoweringError::InvalidContract(
                            "prepared child type cannot be resolved",
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let outputs = match node.kind() {
                NodeKind::SubgraphCall(call) => {
                    let replay_start = self.replay.len();
                    let child = self.graph.source.child_scope_id(scope_id, id).ok_or(
                        PreparedLoweringError::InvalidContract("prepared call has no child"),
                    )?;
                    let environment = environment.child(&call.bindings, None).map_err(|_| {
                        PreparedLoweringError::InvalidContract(
                            "prepared call bindings cannot be resolved",
                        )
                    })?;
                    let enclosing_variants = std::mem::take(&mut self.variant_environments);
                    self.variant_environments = enclosing_variants
                        .iter()
                        .map(|environment| {
                            environment
                                .child(&call.bindings, None)
                                .map_err(|_| PreparedLoweringError::InvalidLoop)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let instantiated = self.scope(&child, &environment, &arguments)?;
                    self.variant_environments = enclosing_variants;
                    let body = self.replay.split_off(replay_start).into_boxed_slice();
                    self.replay
                        .push(PreparedReplayStep::Subgraph { call: Some(NodeId(id.0)), body });
                    self.graph
                        .source
                        .scope(&child)
                        .expect("validated child")
                        .outputs()
                        .iter()
                        .map(|wire| instantiated[wire])
                        .collect()
                }
                NodeKind::ParallelLoop(specification) => {
                    let enclosing = if self.variant_environments.is_empty() {
                        vec![environment.clone()]
                    } else {
                        self.variant_environments.clone()
                    };
                    let counts = enclosing
                        .iter()
                        .map(|environment| {
                            specification
                                .count
                                .evaluate(environment)
                                .ok()
                                .and_then(|value| value.to_usize())
                                .ok_or(PreparedLoweringError::InvalidLoop)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let count = counts.iter().copied().max().unwrap_or(0);
                    let child = self
                        .graph
                        .source
                        .child_scope_id(scope_id, id)
                        .ok_or(PreparedLoweringError::InvalidLoop)?;
                    let child_outputs = self
                        .graph
                        .source
                        .scope(&child)
                        .ok_or(PreparedLoweringError::InvalidLoop)?
                        .outputs()
                        .to_vec();
                    let mut members = vec![Vec::with_capacity(count); child_outputs.len()];
                    let width = self.wave_bound.get().min(count.max(1));
                    let wave_steps_start = self.replay.len();
                    let mut replay_waves = Vec::new();
                    for iteration in 0..count {
                        let active_environment = enclosing
                            .iter()
                            .zip(&counts)
                            .find(|(_, count)| iteration < **count)
                            .map(|(environment, _)| environment)
                            .expect("maximum prepared iteration");
                        let environment = active_environment
                            .child(
                                &specification.bindings,
                                Some((specification.index_slot, iteration)),
                            )
                            .map_err(|_| PreparedLoweringError::InvalidLoop)?;
                        let arguments = arguments
                            .iter()
                            .zip(&specification.input_modes)
                            .map(|(wire, mode)| match mode {
                                LoopInputMode::Broadcast => Ok(*wire),
                                LoopInputMode::Zip | LoopInputMode::ZipOffset { .. } => {
                                    let offset = match mode {
                                        LoopInputMode::ZipOffset { offset } => *offset,
                                        _ => 0,
                                    };
                                    self.families
                                        .get(wire)
                                        .and_then(|members| members.get(iteration + offset))
                                        .copied()
                                        .ok_or(PreparedLoweringError::InvalidIndex)
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let enclosing_variants = std::mem::take(&mut self.variant_environments);
                        self.variant_environments = enclosing_variants
                            .iter()
                            .zip(&counts)
                            .map(|(environment, count)| {
                                let environment = if iteration < *count {
                                    environment
                                } else {
                                    active_environment
                                };
                                environment
                                    .child(
                                        &specification.bindings,
                                        Some((specification.index_slot, iteration)),
                                    )
                                    .map_err(|_| PreparedLoweringError::InvalidLoop)
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let iteration_start = self.replay.len();
                        let instantiated = self.scope(&child, &environment, &arguments)?;
                        self.variant_environments = enclosing_variants;
                        let body = self.replay.split_off(iteration_start).into_boxed_slice();
                        self.replay.push(PreparedReplayStep::Subgraph { call: None, body });
                        for (members, output) in members.iter_mut().zip(&child_outputs) {
                            members.push(instantiated[output]);
                        }
                        if (iteration + 1) % width == 0 || iteration + 1 == count {
                            replay_waves
                                .push(self.replay.split_off(wave_steps_start).into_boxed_slice());
                        }
                    }
                    self.replay.push(PreparedReplayStep::Parallel {
                        call: NodeId(id.0),
                        counts: if counts.iter().any(|value| *value != count) {
                            counts.into_boxed_slice()
                        } else {
                            Box::new([])
                        },
                        waves: replay_waves.into_boxed_slice(),
                    });
                    members
                        .into_iter()
                        .zip(types.iter())
                        .map(|(members, ty)| {
                            self.emit(
                                NodeKind::FamilyPack { count: IntExpr::Const(count.into()) },
                                environment,
                                members,
                                vec![ty.clone()],
                            )
                            .map(|outputs| outputs[0])
                        })
                        .collect::<Result<Vec<_>, _>>()?
                }
                NodeKind::SequentialLoop(specification) => {
                    let enclosing_environments = if self.variant_environments.is_empty() {
                        std::slice::from_ref(environment)
                    } else {
                        &self.variant_environments
                    };
                    let counts = enclosing_environments
                        .iter()
                        .map(|environment| {
                            specification
                                .count
                                .evaluate(environment)
                                .ok()
                                .and_then(|value| value.to_usize())
                                .ok_or(PreparedLoweringError::InvalidLoop)
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    let count = counts[0];
                    let maximum_count = counts.iter().copied().max().unwrap_or(0);
                    if maximum_count == 0 {
                        arguments[..specification.carried_count].to_vec()
                    } else {
                        let child = self
                            .graph
                            .source
                            .child_scope_id(scope_id, id)
                            .ok_or(PreparedLoweringError::InvalidLoop)?;
                        let child_outputs = self
                            .graph
                            .source
                            .scope(&child)
                            .ok_or(PreparedLoweringError::InvalidLoop)?
                            .outputs()
                            .to_vec();
                        let mut offsets = Vec::with_capacity(counts.len());
                        let mut environments = Vec::new();
                        for (outer, count) in enclosing_environments.iter().zip(&counts) {
                            offsets.push(environments.len());
                            for index in 0..*count {
                                environments.push(
                                    outer
                                        .child(
                                            &specification.bindings,
                                            Some((specification.index_slot, index)),
                                        )
                                        .map_err(|_| PreparedLoweringError::InvalidLoop)?,
                                );
                            }
                        }
                        for environment in &environments {
                            for (port, output) in child_outputs.iter().enumerate() {
                                let output_type = self
                                    .graph
                                    .concrete_wire_type(&child, *output, environment)
                                    .map_err(|_| PreparedLoweringError::InvalidLoop)?;
                                if output_type != self.program.wire_types[&arguments[port]] {
                                    return Err(PreparedLoweringError::InvalidContract(
                                        "prepared carried state changes its storage contract",
                                    ));
                                }
                            }
                        }
                        let mut bank_a = Vec::new();
                        for source in &arguments[..specification.carried_count] {
                            bank_a.push(self.copy(*source, None, environment)?);
                        }
                        let mut bodies = [Box::new([]) as Box<[PreparedReplayStep]>, Box::new([])];
                        let mut bank_b = Vec::new();
                        let enclosing_variants =
                            std::mem::replace(&mut self.variant_environments, environments);
                        for bank in 0..2 {
                            let body_start = self.replay.len();
                            let body_inputs = if bank == 0 { &bank_a } else { &bank_b };
                            let mut bound_inputs = body_inputs.clone();
                            bound_inputs
                                .extend_from_slice(&arguments[specification.carried_count..]);
                            let body_environment = self.variant_environments
                                [bank.min(self.variant_environments.len() - 1)]
                            .clone();
                            let instantiated =
                                self.scope(&child, &body_environment, &bound_inputs)?;
                            let results = child_outputs
                                .iter()
                                .map(|wire| instantiated[wire])
                                .collect::<Vec<_>>();
                            for (port, source) in results.into_iter().enumerate() {
                                if bank == 0 {
                                    bank_b.push(self.copy(source, None, &body_environment)?);
                                } else {
                                    self.copy(source, Some(bank_a[port]), &body_environment)?;
                                }
                            }
                            bodies[bank] = self.replay.split_off(body_start).into_boxed_slice();
                        }
                        self.variant_environments = enclosing_variants;
                        // A fixed canonical owner makes the enclosing binding independent
                        // of the selected count's parity (including zero iterations).
                        let tail_start = self.replay.len();
                        for (source, destination) in bank_b.iter().zip(&bank_a) {
                            self.copy(*source, Some(*destination), environment)?;
                        }
                        let tail = self.replay.split_off(tail_start).into_boxed_slice();
                        // Both tapes repeat: a linear last use is not a lifetime end
                        // until the back edge has retired. Keep every referenced owner
                        // through the loop's canonical-output copy.
                        let variable = counts.iter().any(|value| *value != count);
                        self.replay.push(PreparedReplayStep::Sequential {
                            call: NodeId(id.0),
                            count,
                            counts: if variable { counts.into_boxed_slice() } else { Box::new([]) },
                            offsets: if variable {
                                offsets.into_boxed_slice()
                            } else {
                                Box::new([])
                            },
                            banks: bodies,
                            tail,
                        });
                        bank_a
                    }
                }
                NodeKind::FamilyGetStatic { index } => {
                    if self.variant_environments.iter().any(|variant| {
                        index.evaluate(variant).ok() != index.evaluate(environment).ok()
                    }) {
                        let selector = self.emit(
                            NodeKind::EvaluateInt(index.clone()),
                            environment,
                            vec![],
                            vec![ConcreteWireType::ConstantInt],
                        )?[0];
                        self.emit(
                            NodeKind::FamilyGetDynamic,
                            environment,
                            vec![arguments[0], selector],
                            types,
                        )?
                    } else {
                        let index = index
                            .evaluate(environment)
                            .ok()
                            .and_then(|value| value.to_usize())
                            .ok_or(PreparedLoweringError::InvalidIndex)?;
                        vec![
                            *self
                                .families
                                .get(&arguments[0])
                                .and_then(|members| members.get(index))
                                .ok_or(PreparedLoweringError::InvalidIndex)?,
                        ]
                    }
                }
                kind => {
                    let outputs = self.emit(kind.clone(), environment, arguments, types)?;
                    if *scope_id == FrozenGraphScopeId::Root &&
                        matches!(kind, NodeKind::Input { artifact: None, .. })
                    {
                        if let Some(ConcreteWireType::IndexedFamily { element, count }) =
                            self.program.wire_types.get(&outputs[0]).cloned()
                        {
                            let members =
                                self.register_family_input(&element, count, outputs[0], &[])?;
                            self.families.insert(outputs[0], members.clone());
                            self.program.family_wires.insert(outputs[0], members);
                        }
                    }
                    if !self.variant_environments.is_empty() {
                        let local_arguments = source.arguments(node).expect("validated arguments");
                        let mut kinds = Vec::new();
                        let mut inputs = Vec::<Box<[ConcreteWireType]>>::new();
                        let mut output_types = Vec::<Box<[ConcreteWireType]>>::new();
                        let mut indices = Vec::new();
                        for environment in &self.variant_environments {
                            let kind = close_kind(kind.clone(), environment)?;
                            let concrete = |wires: &[WireRef]| {
                                wires
                                    .iter()
                                    .map(|wire| {
                                        self.graph
                                            .concrete_wire_type(scope_id, *wire, environment)
                                            .map_err(|_| PreparedLoweringError::InvalidLoop)
                                    })
                                    .collect::<Result<Box<[_]>, _>>()
                            };
                            let input = concrete(&local_arguments)?;
                            let output = concrete(&local_outputs)?;
                            let variant = (0..kinds.len()).find(|index| {
                                kinds[*index] == kind &&
                                    inputs[*index] == input &&
                                    output_types[*index] == output
                            });
                            indices.push(variant.unwrap_or_else(|| {
                                kinds.push(kind);
                                inputs.push(input);
                                output_types.push(output);
                                kinds.len() - 1
                            }));
                        }
                        for (port, wire) in outputs.iter().enumerate() {
                            for types in &output_types {
                                maximize_shape(
                                    self.program
                                        .wire_types
                                        .get_mut(wire)
                                        .expect("prepared output type"),
                                    &types[port],
                                )?;
                            }
                            if let Some(matrix) = self.program.wire_types[wire].matrix_type() {
                                if let Some(location) = self.program.values.get_mut(wire) {
                                    location.rows.end = matrix.rows;
                                    location.columns.end = matrix.columns;
                                }
                            }
                        }
                        if kinds.len() > 1 {
                            let prepared = self
                                .program
                                .node_sources
                                .get_mut(&(outputs[0].node.0 as u32))
                                .expect("prepared source");
                            prepared.variants = kinds.into_boxed_slice();
                            prepared.variant_indices = indices.into_boxed_slice();
                            prepared.variant_input_types = inputs.into_boxed_slice();
                            prepared.variant_output_types = output_types.into_boxed_slice();
                        }
                    }
                    outputs
                }
            };
            for (source_wire, prepared_wire) in local_outputs.into_iter().zip(outputs) {
                // Every instantiated wire gets a stable logical key. Loop
                // replay adds its fixed iteration/path frame at execution;
                // keeping the local source wire here prevents nested trace
                // entries from falling back to physical node identities.
                self.program.trace_keys.insert(
                    prepared_wire,
                    mxx_ir_core::types::WireId {
                        instantiation_path: Vec::new(),
                        wire: source_wire,
                    },
                );
                wires.insert(source_wire, prepared_wire);
            }
        }
        Ok(wires)
    }

    fn emit(
        &mut self,
        kind: NodeKind,
        environment: &ParamEnv,
        arguments: Vec<WireRef>,
        types: Vec<ConcreteWireType>,
    ) -> Result<Vec<WireRef>, PreparedLoweringError> {
        let id = self.next_node;
        self.next_node = self
            .next_node
            .checked_add(1)
            .ok_or(PreparedLoweringError::InvalidContract("prepared node identity overflow"))?;
        let mut operation = prepared_operation(&kind);
        if matches!(kind, NodeKind::EvaluateInt(_) | NodeKind::ConstantReal(_)) {
            operation = PreparedOperation::Scalar;
        }
        if matches!(kind, NodeKind::UniformIntervalSample { .. }) {
            // The native GPU sampler has no lower-bound descriptor. Reject
            // unsupported intervals while lowering instead of silently
            // widening them to an incorrect distribution.
            close_kind(kind.clone(), environment)?;
        }
        let outputs = types
            .iter()
            .enumerate()
            .map(|(port, ty)| {
                let wire = WireRef { node: NodeId(id as u64), port: Port(port as u32) };
                self.program.wire_types.insert(wire, ty.clone());
                let format = if matches!(
                    kind,
                    NodeKind::RnsModUp { .. } |
                        NodeKind::RnsModDown { .. } |
                        NodeKind::ModulusSwitch { .. } |
                        NodeKind::ModulusReduce { .. }
                ) {
                    PreparedFormat::Coefficient
                } else {
                    PreparedFormat::Evaluation
                };
                if let Some(location) =
                    matrix_location((u64::from(id) << 32) | port as u64, ty, format)
                {
                    self.program.values.insert(wire, location);
                }
                if matches!(
                    ty,
                    ConcreteWireType::Int |
                        ConcreteWireType::Real |
                        ConcreteWireType::Bool |
                        ConcreteWireType::ConstantInt |
                        ConcreteWireType::ConstantReal |
                        ConcreteWireType::ConstantBool
                ) {
                    let slot = self.program.scalar_slot_count;
                    self.program.scalar_slot_count += 1;
                    self.program.scalar_slots.insert(wire, slot);
                    let value = match &kind {
                        NodeKind::ConstantInt(value) => Some(ScalarValue::Int(value.clone())),
                        NodeKind::EvaluateInt(expression) => {
                            Some(constant_int(expression, environment)?)
                        }
                        NodeKind::ConstantReal(expression) => {
                            Some(constant_real(expression, environment)?)
                        }
                        NodeKind::ConstantBool(value) => Some(ScalarValue::Bool(*value)),
                        _ => None,
                    };
                    if let Some(value) = value {
                        self.program.scalar_initializers.insert(slot, value);
                    }
                }
                Ok(wire)
            })
            .collect::<Result<Vec<_>, PreparedLoweringError>>()?;
        if operation == PreparedOperation::Scalar {
            let slots = arguments
                .iter()
                .map(|wire| {
                    self.program.scalar_slots.get(wire).copied().ok_or(
                        PreparedLoweringError::InvalidContract(
                            "prepared scalar argument has no slot",
                        ),
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            self.program.scalar_commands.insert(id, lower_scalar(&kind, environment, &slots)?);
        }
        let locations = arguments
            .iter()
            .filter_map(|wire| self.program.values.get(wire).cloned())
            .collect::<Vec<_>>();
        let destination = outputs.first().and_then(|wire| self.program.values.get(wire)).cloned();
        let view = match (&kind, destination) {
            (NodeKind::Slice { .. }, Some(destination)) => {
                Some(lower_slice(&kind, environment, locations[0].clone(), destination)?)
            }
            (NodeKind::Transpose, Some(destination)) => {
                Some(lower_transpose(&kind, locations[0].clone(), destination)?)
            }
            (NodeKind::Concat { .. }, Some(destination)) => {
                Some(lower_concat(&kind, &locations, destination)?)
            }
            _ => None,
        };
        if let Some(view) = view {
            if matches!(view, PreparedView::FixedCopies(_)) &&
                matches!(kind, NodeKind::Slice { .. } | NodeKind::Concat { .. })
            {
                operation = PreparedOperation::Gpu(PreparedGpuOperation::FixedCopies);
            }
            self.program.view_commands.insert(id, view);
        }
        match &kind {
            NodeKind::FamilyPack { .. } => {
                self.families.insert(outputs[0], arguments.clone().into_boxed_slice());
                self.program.family_wires.insert(outputs[0], arguments.clone().into_boxed_slice());
                self.program.family_members.insert(outputs[0], locations.into_boxed_slice());
            }
            NodeKind::FamilyGetDynamic | NodeKind::Select { .. } => {
                let (candidate_wires, selector) = if matches!(kind, NodeKind::FamilyGetDynamic) {
                    let family_arguments = self
                        .families
                        .get(&arguments[0])
                        .cloned()
                        .ok_or(PreparedLoweringError::InvalidIndex)?;
                    (family_arguments, arguments[1])
                } else {
                    (arguments[1..].to_vec().into_boxed_slice(), arguments[0])
                };
                let scalar_candidates = candidate_wires
                    .iter()
                    .map(|wire| self.program.scalar_slots.get(wire).copied())
                    .collect::<Option<Vec<_>>>();
                let selection = if let Some(scalar_candidates) = scalar_candidates {
                    lower_scalar_family(
                        &kind,
                        scalar_candidates.into_boxed_slice(),
                        selector,
                        None,
                    )?
                } else {
                    let candidates = candidate_wires
                        .iter()
                        .map(|wire| {
                            self.program
                                .values
                                .get(wire)
                                .cloned()
                                .ok_or(PreparedLoweringError::InvalidIndex)
                        })
                        .collect::<Result<Box<[_]>, _>>()?;
                    lower_family(&kind, &PreparedFamily { members: candidates }, None)?
                        .with_selector(selector)
                };
                self.program.selection_commands.insert(id, selection);
            }
            _ => {}
        }
        let waits = arguments
            .iter()
            .map(|wire| wire.node.0 as u32)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Box<[_]>>();
        self.nodes.push(PreparedTopologyNode {
            id,
            command: PreparedCommandRequirement {
                operation,
                inputs: arguments.len(),
                outputs: outputs.len(),
            },
            stream: id % 4,
            waits,
            completion: id,
        });
        self.program
            .node_bindings
            .insert(id, (arguments.into_boxed_slice(), outputs.clone().into_boxed_slice()));
        let mut variants = Vec::new();
        let mut variant_indices = Vec::new();
        for environment in &self.variant_environments {
            let closed = close_kind(kind.clone(), environment)?;
            let index =
                variants.iter().position(|candidate| *candidate == closed).unwrap_or_else(|| {
                    variants.push(closed);
                    variants.len() - 1
                });
            variant_indices.push(index);
        }
        if variants.len() < 2 {
            variants.clear();
            variant_indices.clear();
        }
        self.program.node_sources.insert(
            id,
            PreparedNodeSource {
                kind,
                environment: environment.clone(),
                variants: variants.into_boxed_slice(),
                variant_indices: variant_indices.into_boxed_slice(),
                variant_input_types: Box::new([]),
                variant_output_types: Box::new([]),
            },
        );
        self.replay.push(PreparedReplayStep::Node(id));
        Ok(outputs)
    }

    fn register_family_input(
        &mut self,
        element: &ConcreteWireType,
        count: usize,
        root: WireRef,
        path: &[usize],
    ) -> Result<Box<[WireRef]>, PreparedLoweringError> {
        let mut members = Vec::with_capacity(count);
        for index in 0..count {
            let wire = WireRef {
                node: mxx_ir_core::types::NodeId(self.next_virtual_wire),
                port: Port(index as u32),
            };
            self.next_virtual_wire = self.next_virtual_wire.checked_sub(1).ok_or(
                PreparedLoweringError::InvalidContract("prepared family input identity overflow"),
            )?;
            self.program.wire_types.insert(wire, element.clone());
            let mut member_path = path.to_vec();
            member_path.push(index);
            if let ConcreteWireType::IndexedFamily { element, count } = element {
                let nested = self.register_family_input(element, *count, root, &member_path)?;
                self.families.insert(wire, nested.clone());
                self.program.family_wires.insert(wire, nested);
            } else {
                self.program
                    .input_leaf_bindings
                    .insert(wire, PreparedInputLeaf { root, path: member_path.into_boxed_slice() });
                if let Some(location) =
                    matrix_location(wire.node.0, element, PreparedFormat::Evaluation)
                {
                    self.program.values.insert(wire, location);
                }
                if matches!(
                    element,
                    ConcreteWireType::Int |
                        ConcreteWireType::Real |
                        ConcreteWireType::Bool |
                        ConcreteWireType::ConstantInt |
                        ConcreteWireType::ConstantReal |
                        ConcreteWireType::ConstantBool
                ) {
                    let slot = self.program.scalar_slot_count;
                    self.program.scalar_slot_count += 1;
                    self.program.scalar_slots.insert(wire, slot);
                }
            }
            members.push(wire);
        }
        if members.is_empty() {
            return Err(PreparedLoweringError::InvalidContract(
                "prepared family input has no members",
            ));
        }
        Ok(members.into_boxed_slice())
    }

    fn copy(
        &mut self,
        source: WireRef,
        destination: Option<WireRef>,
        environment: &ParamEnv,
    ) -> Result<WireRef, PreparedLoweringError> {
        let ty = self.program.wire_types[&source].clone();
        if let Some(source_location) = self.program.values.get(&source).cloned() {
            let output = self.emit(
                NodeKind::Slice { rows: None, columns: None },
                environment,
                vec![source],
                vec![ty],
            )?[0];
            let mut target = self.program.values[&output].clone();
            target.format = source_location.format;
            target.level = source_location.level;
            if let Some(destination) = destination {
                target.owner = self.program.values[&destination].owner;
            }
            self.program.values.insert(output, target.clone());
            self.program.view_commands.insert(
                output.node.0 as u32,
                PreparedView::FixedCopies(
                    vec![FixedCopy { source: source_location, destination: target }]
                        .into_boxed_slice(),
                ),
            );
            Ok(output)
        } else {
            let source_slot = *self.program.scalar_slots.get(&source).ok_or(
                PreparedLoweringError::InvalidContract(
                    "prepared carry is neither a matrix nor a scalar",
                ),
            )?;
            let output =
                self.emit(NodeKind::ConstantBool(false), environment, vec![source], vec![ty])?[0];
            let id = output.node.0 as u32;
            if let Some(destination) = destination {
                self.program.scalar_slots.insert(output, self.program.scalar_slots[&destination]);
            }
            self.program.scalar_commands.insert(
                id,
                PreparedScalar {
                    constants: vec![ScalarValue::Slot(source_slot)].into_boxed_slice(),
                    instructions: Box::new([]),
                    result: 0,
                },
            );
            self.nodes.last_mut().expect("prepared scalar carry").command.operation =
                PreparedOperation::Scalar;
            Ok(output)
        }
    }
}

fn maximize_shape(
    capacity: &mut ConcreteWireType,
    active: &ConcreteWireType,
) -> Result<(), PreparedLoweringError> {
    let Some(active_matrix) = active.matrix_type() else { return Ok(()) };
    let capacity_matrix = match capacity {
        ConcreteWireType::Matrix(matrix) |
        ConcreteWireType::SmallMatrix { matrix, .. } |
        ConcreteWireType::Preimage { matrix, .. } |
        ConcreteWireType::Trapdoor { matrix, .. } => matrix,
        _ => return Err(PreparedLoweringError::InvalidLoop),
    };
    // The representative type keeps its basis. The factory partitions the finite
    // variant table into separate accepted backing classes before allocation.
    capacity_matrix.rows = capacity_matrix.rows.max(active_matrix.rows);
    capacity_matrix.columns = capacity_matrix.columns.max(active_matrix.columns);
    Ok(())
}

fn close_kind(
    mut kind: NodeKind,
    environment: &ParamEnv,
) -> Result<NodeKind, PreparedLoweringError> {
    fn integer(value: &mut IntExpr, environment: &ParamEnv) -> Result<(), PreparedLoweringError> {
        *value = IntExpr::Const(
            value.evaluate(environment).map_err(|_| PreparedLoweringError::InvalidLoop)?,
        );
        Ok(())
    }
    fn matrix(
        value: &mut mxx_ir_core::types::MatrixType,
        environment: &ParamEnv,
    ) -> Result<(), PreparedLoweringError> {
        integer(&mut value.modulus, environment)?;
        integer(&mut value.ring_dimension, environment)?;
        integer(&mut value.rows, environment)?;
        integer(&mut value.columns, environment)
    }
    match &mut kind {
        NodeKind::ConstantMatrix { matrix_type, .. } |
        NodeKind::GadgetTrapdoor { matrix_type, .. } |
        NodeKind::UniformResidueSample { matrix_type } |
        NodeKind::UniformIntervalSample { matrix_type, .. } |
        NodeKind::GaussianSample { matrix_type, .. } |
        NodeKind::HashSample { matrix_type, .. } |
        NodeKind::TrapdoorSample { matrix_type, .. } |
        NodeKind::PreimageSample { matrix_type, .. } |
        NodeKind::LiftIntegerToConstantPolynomial { matrix_type } |
        NodeKind::PackPolynomialCoefficients { matrix_type, .. } |
        NodeKind::PolynomialFromValues { matrix_type, .. } => matrix(matrix_type, environment)?,
        _ => {}
    }
    match &mut kind {
        NodeKind::EvaluateInt(value) |
        NodeKind::BitExtract { bit: value } |
        NodeKind::MatrixScale { scalar: value } |
        NodeKind::RingAutomorphism { index: value } |
        NodeKind::ModulusSwitch { modulus: value } |
        NodeKind::ModulusReduce { modulus: value } |
        NodeKind::CenteredRebase { modulus: value } |
        NodeKind::CenteredExtend { modulus: value } |
        NodeKind::RnsModUp { modulus: value, .. } |
        NodeKind::GadgetTrapdoor { base: value, .. } |
        NodeKind::FamilyPack { count: value } |
        NodeKind::FamilyGetStatic { index: value } |
        NodeKind::Select { count: value } |
        NodeKind::ExtractCoefficient { position: value, .. } |
        NodeKind::PreimageSample { max_coefficient_bound: value, .. } |
        NodeKind::PackPolynomialCoefficients { coefficient_bits: value, .. } => {
            integer(value, environment)?
        }
        NodeKind::MatrixMulAccumulate { coefficients, .. } => {
            for value in coefficients {
                integer(value, environment)?;
            }
        }
        NodeKind::RnsModDown { modulus, plaintext_modulus, .. } |
        NodeKind::BlockModSwitch { modulus, plaintext_modulus } => {
            integer(modulus, environment)?;
            integer(plaintext_modulus, environment)?;
        }
        NodeKind::ConstantReal(value) => {
            *value = value.close(environment).map_err(|_| PreparedLoweringError::InvalidLoop)?
        }
        NodeKind::Slice { rows, columns } => {
            for range in rows.iter_mut().chain(columns.iter_mut()) {
                integer(&mut range.start, environment)?;
                integer(&mut range.end, environment)?;
            }
        }
        NodeKind::UniformIntervalSample { range, .. } => {
            integer(&mut range.minimum, environment)?;
            integer(&mut range.maximum, environment)?;
        }
        NodeKind::GaussianSample { sigma, max_coefficient_bound, .. } => {
            *sigma = sigma.close(environment).map_err(|_| PreparedLoweringError::InvalidLoop)?;
            integer(max_coefficient_bound, environment)?;
        }
        NodeKind::HashSample { tag_components, base, digit_count, .. } => {
            for value in base.iter_mut().chain(digit_count.iter_mut()) {
                integer(value, environment)?;
            }
            for component in tag_components {
                use mxx_ir_core::node::HashTagComponent;
                if let HashTagComponent::Integer(value) |
                HashTagComponent::Decimal(value) |
                HashTagComponent::U64Le(value) = component
                {
                    integer(value, environment)?;
                }
            }
        }
        NodeKind::GadgetDecompose { base, digit_count, .. } => {
            integer(base, environment)?;
            integer(digit_count, environment)?;
        }
        NodeKind::ThresholdDecode { plaintext_modulus, length, .. } => {
            integer(plaintext_modulus, environment)?;
            integer(length, environment)?;
        }
        NodeKind::CrtRecompose { modulus, plaintext_moduli, reconstruction_coefficients } => {
            integer(modulus, environment)?;
            for value in plaintext_moduli.iter_mut().chain(reconstruction_coefficients.iter_mut()) {
                integer(value, environment)?;
            }
        }
        NodeKind::TrapdoorSample {
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
            ..
        } => {
            *sigma = sigma.close(environment).map_err(|_| PreparedLoweringError::InvalidLoop)?;
            for value in [gadget_base, digit_count, preimage_max_coefficient_bound] {
                integer(value, environment)?;
            }
        }
        NodeKind::ConstantMatrix { value, .. } => {
            use mxx_ir_core::node::ConstantMatrix;
            match value {
                ConstantMatrix::UnitRow { index } |
                ConstantMatrix::UnitColumn { index } |
                ConstantMatrix::Gadget { base: index, .. } |
                ConstantMatrix::Rotation { exponent: index } => integer(index, environment)?,
                ConstantMatrix::PowerOfBase { base, exponent } => {
                    integer(base, environment)?;
                    integer(exponent, environment)?;
                }
                ConstantMatrix::Polynomial { coefficients } => {
                    for value in coefficients {
                        integer(value, environment)?;
                    }
                }
                ConstantMatrix::Zero | ConstantMatrix::Identity => {}
            }
        }
        _ => {}
    }
    if let NodeKind::UniformIntervalSample { matrix_type, range } = &kind {
        let minimum =
            range.minimum.evaluate(environment).map_err(|_| PreparedLoweringError::InvalidLoop)?;
        let maximum =
            range.maximum.evaluate(environment).map_err(|_| PreparedLoweringError::InvalidLoop)?;
        let modulus = matrix_type
            .modulus
            .evaluate(environment)
            .map_err(|_| PreparedLoweringError::InvalidLoop)?;
        if minimum != num_bigint::BigInt::from(0) ||
            maximum != modulus - num_bigint::BigInt::from(1)
        {
            return Err(PreparedLoweringError::InvalidContract(
                "GPU prepared uniform interval requires the full residue range",
            ));
        }
    }
    Ok(kind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        graph::{
            Graph, GraphOutput, NodeHandle, SubgraphHandle, ValueHandle,
            with_new_construction_scope,
        },
        node::{ParallelLoop, SampleRange, SequentialLoop},
        types::{MatrixType, WireType},
        validate::validate,
    };

    fn input(name: &str) -> ValueHandle {
        NodeHandle::new(
            NodeKind::Input { name: name.into(), wire_type: WireType::Int, artifact: None },
            vec![],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap()
    }

    fn body() -> SubgraphHandle {
        with_new_construction_scope(|scope| {
            let carried = input("carried");
            let increment = input("increment");
            let result = NodeHandle::new(
                NodeKind::IntBinary(IntBinaryOp::Add),
                vec![carried.clone(), increment.clone()],
                vec![WireType::Int],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("add_body", scope, vec![carried, increment], vec![result]).unwrap()
        })
    }

    struct ScalarResourcePlanner;

    impl crate::backend::poly_gpu::gpu_prepared_lowering::PreparedResourceBackend
        for ScalarResourcePlanner
    {
        fn plan_owner(
            &self,
            _: &PreparedOwnerKey,
            _: &PreparedStorePlan,
            _: usize,
        ) -> Result<(PreparedOwnerLayout, usize), String> {
            unreachable!("scalar test graph has no matrix owners")
        }

        fn plan_stage(
            &self,
            _: &PreparedNativeRecipe,
            _: &[PreparedStorePlan],
            _: &[PreparedResolvedOwner],
        ) -> Result<Option<PreparedPlanLayout>, String> {
            unreachable!("scalar test graph has no native stages")
        }

        fn plan_replay_upload(
            &self,
            _: &PreparedNativeRecipe,
            _: &PreparedReplayUploadRecipe,
            _: &[PreparedStorePlan],
            _: &[PreparedResolvedOwner],
        ) -> Result<PreparedPlanLayout, String> {
            unreachable!("scalar test graph has no replay uploads")
        }

        fn plan_schedule(
            &self,
            _: &PreparedNativeRecipe,
            _: &[PreparedPlanLayout],
        ) -> Result<PreparedPlanLayout, String> {
            unreachable!("scalar test graph has no native schedules")
        }
    }

    fn freeze(output: ValueHandle) -> ValidatedGraph {
        use mxx_ir_core::graph::{CompileParameter, CompileParameterKind};
        let (graph, _) = Graph::freeze(
            "prepared_scope",
            vec![CompileParameter {
                name: "iteration".into(),
                kind: CompileParameterKind::Integer,
            }],
            BTreeMap::from([(
                "result".into(),
                GraphOutput { value: output, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let mut bindings = ParamEnv::default();
        bindings.integers.insert("iteration".into(), 0.into());
        validate(&graph, &bindings).unwrap()
    }

    fn storage_location(owner: u64, rows: usize, columns: usize) -> ValueLocation {
        ValueLocation {
            owner,
            rows: 0..rows,
            columns: 0..columns,
            level: 0,
            format: PreparedFormat::Evaluation,
            device: 0,
        }
    }

    fn storage_node(id: u32, stream: u32, waits: &[u32]) -> PreparedTopologyNode {
        PreparedTopologyNode {
            id,
            command: PreparedCommandRequirement {
                operation: PreparedOperation::Gpu(PreparedGpuOperation::MatrixNegate),
                inputs: 0,
                outputs: 1,
            },
            stream,
            waits: waits.into(),
            completion: id,
        }
    }

    fn storage_program(
        nodes: Vec<PreparedTopologyNode>,
        values: &[(u32, u64, usize, usize)],
    ) -> GpuPreparation {
        let mut program = GpuPreparation::default();
        program.topology.nodes = nodes.into_boxed_slice();
        for (node, owner, rows, columns) in values {
            let wire = WireRef { node: NodeId(*node as u64), port: Port(0) };
            program.values.insert(wire, storage_location(*owner, *rows, *columns));
            program.wire_types.insert(
                wire,
                ConcreteWireType::Matrix(mxx_ir_core::types::ConcreteMatrixType {
                    modulus: 17.into(),
                    ring_dimension: 1,
                    rows: *rows,
                    columns: *columns,
                }),
            );
            program.node_bindings.insert(*node, (Box::new([]), vec![wire].into_boxed_slice()));
        }
        program
    }

    #[test]
    fn test_gpu_prepared_storage_reuse_requires_explicit_happens_before() {
        let program = storage_program(
            vec![storage_node(1, 0, &[]), storage_node(2, 0, &[])],
            &[(1, 11, 1, 1), (2, 22, 1, 1)],
        );
        let mut program = program;
        color_storage(&mut program);
        assert_ne!(
            program.values[&WireRef { node: NodeId(1), port: Port(0) }].owner,
            program.values[&WireRef { node: NodeId(2), port: Port(0) }].owner
        );
    }

    #[test]
    fn test_gpu_prepared_storage_reuse_uses_explicit_order_and_smallest_capacity() {
        let mut program = storage_program(
            vec![storage_node(1, 0, &[]), storage_node(2, 0, &[1]), storage_node(3, 0, &[2])],
            &[(1, 11, 1, 1), (2, 22, 4, 4), (3, 33, 1, 1)],
        );
        color_storage(&mut program);
        let owner = |node| program.values[&WireRef { node: NodeId(node), port: Port(0) }].owner;
        assert_eq!(owner(1), owner(3));
        assert_ne!(owner(1), owner(2));
        assert_ne!(owner(2), owner(3));
    }

    #[test]
    fn test_gpu_prepared_storage_reuse_rejects_cross_device_ancestor() {
        let first = WireRef { node: NodeId(1), port: Port(0) };
        let second = WireRef { node: NodeId(2), port: Port(0) };
        let third = WireRef { node: NodeId(3), port: Port(0) };
        let mut program = storage_program(
            vec![storage_node(1, 0, &[]), storage_node(2, 1, &[1]), storage_node(3, 0, &[2])],
            &[(1, 11, 1, 1), (2, 22, 1, 1), (3, 33, 1, 1)],
        );
        program.values.get_mut(&second).unwrap().device = 1;
        program.node_bindings.get_mut(&2).unwrap().0 = vec![first].into_boxed_slice();
        color_storage(&mut program);
        assert_ne!(program.values[&first].owner, program.values[&third].owner);
    }

    #[test]
    fn test_gpu_prepared_storage_does_not_reuse_pinned_output() {
        let first = WireRef { node: NodeId(1), port: Port(0) };
        let second = WireRef { node: NodeId(2), port: Port(0) };
        let mut program = storage_program(
            vec![storage_node(1, 0, &[]), storage_node(2, 0, &[])],
            &[(1, 11, 1, 1), (2, 22, 1, 1)],
        );
        program.outputs = vec![first].into_boxed_slice();
        color_storage(&mut program);
        assert_ne!(program.values[&first].owner, program.values[&second].owner);
    }

    #[test]
    fn test_gpu_prepared_storage_reuse_requires_matching_level_and_format() {
        let first = WireRef { node: NodeId(1), port: Port(0) };
        let second = WireRef { node: NodeId(2), port: Port(0) };
        let third = WireRef { node: NodeId(3), port: Port(0) };
        let mut program = storage_program(
            vec![storage_node(1, 0, &[]), storage_node(2, 0, &[]), storage_node(3, 0, &[])],
            &[(1, 11, 1, 1), (2, 22, 1, 1), (3, 33, 1, 1)],
        );
        program.values.get_mut(&second).unwrap().format = PreparedFormat::Coefficient;
        program.values.get_mut(&third).unwrap().level = 1;
        color_storage(&mut program);
        assert_ne!(program.values[&first].owner, program.values[&second].owner);
        assert_ne!(program.values[&first].owner, program.values[&third].owner);
        assert_ne!(program.values[&second].owner, program.values[&third].owner);
    }

    #[test]
    fn test_gpu_prepared_alias_chain_shares_base_owner_without_losing_offset() {
        let first = WireRef { node: NodeId(1), port: Port(0) };
        let second = WireRef { node: NodeId(2), port: Port(0) };
        let mut program = storage_program(
            vec![
                storage_node(1, 0, &[]),
                PreparedTopologyNode {
                    id: 2,
                    command: PreparedCommandRequirement {
                        operation: PreparedOperation::Alias,
                        inputs: 1,
                        outputs: 1,
                    },
                    stream: 0,
                    waits: vec![1].into_boxed_slice(),
                    completion: 2,
                },
            ],
            &[(1, 11, 4, 4), (2, 22, 2, 2)],
        );
        program
            .node_bindings
            .insert(2, (vec![first].into_boxed_slice(), vec![second].into_boxed_slice()));
        program.view_commands.insert(
            2,
            PreparedView::Alias(ValueLocation {
                owner: 11,
                rows: 1..3,
                columns: 1..3,
                level: 0,
                format: PreparedFormat::Evaluation,
                device: 0,
            }),
        );
        color_storage(&mut program);
        assert_eq!(program.values[&first].owner, program.values[&second].owner);
        let PreparedView::Alias(view) = &program.view_commands[&2] else {
            panic!("expected alias view")
        };
        assert_eq!(view.rows, 1..3);
        assert_eq!(view.columns, 1..3);
    }

    #[test]
    fn test_gpu_prepared_subgraph_formals_bind_parent_scalar_slots() {
        let lhs = input("lhs");
        let rhs = input("rhs");
        let call = NodeHandle::subgraph_call(body(), vec![lhs, rhs], vec![], vec![None, None]);
        let program =
            instantiate(&freeze(call.output(0).unwrap()), std::num::NonZeroUsize::new(2).unwrap())
                .unwrap();
        assert_eq!(program.inputs.len(), 2);
        assert_eq!(program.scalar_commands.len(), 1);
        assert_eq!(program.topology.nodes.len(), 3);
        let scalar = program.scalar_commands.values().next().unwrap();
        assert!(scalar.constants.iter().all(|value| matches!(value, ScalarValue::Slot(_))));
        assert!(program.topology.nodes.iter().all(|node| !matches!(
            node.command.operation,
            PreparedOperation::ParallelLoop | PreparedOperation::SequentialLoop
        )));
    }

    #[test]
    fn test_gpu_prepared_subgraph_compiles_warms_and_replays_scalar_result() {
        use crate::{
            backend::poly_gpu::{PreparedGpuProgram, gpu_prepared::PreparedRuntimeValue},
            transcript::SamplingMode,
        };
        use std::{collections::BTreeMap, sync::Arc};
        let lhs = input("lhs");
        let rhs = input("rhs");
        let call = NodeHandle::subgraph_call(body(), vec![lhs, rhs], vec![], vec![None, None]);
        let graph = freeze(call.output(0).unwrap());
        let program = instantiate(&graph, std::num::NonZeroUsize::new(2).unwrap()).unwrap();
        assert!(program.resource_plan.resolve_prepared_resources(&ScalarResourcePlanner).is_ok());
        for wire in program.node_bindings.values().flat_map(|(_, outputs)| outputs.iter()) {
            assert!(program.trace_keys[wire].instantiation_path.is_empty());
        }
        let output_slot = program.scalar_slots[&program.outputs[0]];
        let mut execution = PreparedGpuProgram::from_command_instances(
            vec![Box::new([])],
            Arc::new(crate::gpu_memory::GpuMemoryRegion::empty_for_test()),
            0,
            0,
        )
        .from_preparation(program)
        .unwrap();
        execution
            .initialize_runtime_roots(&[
                PreparedRuntimeValue::Int(7.into()),
                PreparedRuntimeValue::Int(11.into()),
            ])
            .unwrap();
        let inputs = BTreeMap::from([
            ("lhs".to_owned(), crate::backend::RuntimeValue::Int(7.into())),
            ("rhs".to_owned(), crate::backend::RuntimeValue::Int(11.into())),
        ]);
        let mut sampling = SamplingMode::Fresh;
        let output = execution.run_with_runtime_bindings(&inputs, &mut sampling, None).unwrap();
        assert_eq!(output.scalar_slot_values()[output_slot], ScalarValue::Int(18.into()));
    }

    #[test]
    fn test_gpu_prepared_zero_iteration_loop_replays_canonical_input() {
        use crate::{
            backend::poly_gpu::{PreparedGpuProgram, gpu_prepared::PreparedRuntimeValue},
            transcript::SamplingMode,
        };
        use std::{collections::BTreeMap, sync::Arc};
        let node = NodeHandle::sequential_loop(
            body(),
            vec![input("carried"), input("increment")],
            vec![WireType::Int],
            SequentialLoop {
                count: IntExpr::constant(0),
                index_slot: 0,
                bindings: vec![],
                carried_count: 1,
            },
        );
        let graph = freeze(node.output(0).unwrap());
        let program = instantiate(&graph, std::num::NonZeroUsize::new(2).unwrap()).unwrap();
        let output_slot = program.scalar_slots[&program.outputs[0]];
        let mut execution = PreparedGpuProgram::from_command_instances(
            vec![Box::new([])],
            Arc::new(crate::gpu_memory::GpuMemoryRegion::empty_for_test()),
            0,
            0,
        )
        .from_preparation(program)
        .unwrap();
        execution
            .initialize_runtime_roots(&[
                PreparedRuntimeValue::Int(13.into()),
                PreparedRuntimeValue::Int(5.into()),
            ])
            .unwrap();
        let inputs = BTreeMap::from([
            ("carried".to_owned(), crate::backend::RuntimeValue::Int(13.into())),
            ("increment".to_owned(), crate::backend::RuntimeValue::Int(5.into())),
        ]);
        let mut sampling = SamplingMode::Fresh;
        let output = execution.run_with_runtime_bindings(&inputs, &mut sampling, None).unwrap();
        assert_eq!(output.scalar_slot_values()[output_slot], ScalarValue::Int(13.into()));
    }

    #[test]
    fn test_gpu_prepared_lowering_rejects_asymmetric_uniform_interval() {
        let matrix = MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(1),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let sample = NodeHandle::new(
            NodeKind::UniformIntervalSample {
                matrix_type: matrix,
                range: SampleRange {
                    minimum: IntExpr::constant(-1),
                    maximum: IntExpr::constant(1),
                },
            },
            vec![],
            vec![WireType::Matrix(MatrixType {
                modulus: IntExpr::constant(17),
                ring_dimension: IntExpr::constant(1),
                rows: IntExpr::constant(1),
                columns: IntExpr::constant(1),
            })],
        );
        let error = instantiate(
            &freeze(sample.output(0).unwrap()),
            std::num::NonZeroUsize::new(1).unwrap(),
        )
        .unwrap_err();
        assert_eq!(
            error,
            PreparedLoweringError::InvalidContract(
                "GPU prepared uniform interval requires the full residue range"
            )
        );
    }

    #[test]
    fn test_gpu_prepared_root_family_input_freezes_leaf_paths() {
        let family = NodeHandle::new(
            NodeKind::Input {
                name: "values".into(),
                wire_type: WireType::IndexedFamily {
                    element: Box::new(WireType::Int),
                    count: IntExpr::constant(3),
                },
                artifact: None,
            },
            vec![],
            vec![WireType::IndexedFamily {
                element: Box::new(WireType::Int),
                count: IntExpr::constant(3),
            }],
        )
        .output(0)
        .unwrap();
        let selected = NodeHandle::new(
            NodeKind::FamilyGetStatic { index: IntExpr::constant(1) },
            vec![family],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        let program =
            instantiate(&freeze(selected), std::num::NonZeroUsize::new(1).unwrap()).unwrap();
        assert_eq!(program.runtime_input_wires.len(), 3);
        assert_eq!(program.input_leaf_bindings.len(), 3);
        assert_eq!(program.input_leaf_bindings[&program.runtime_input_wires[1]].path.as_ref(), [1]);
        assert_eq!(program.scalar_slots.len(), 3);
    }

    #[test]
    fn test_gpu_prepared_parallel_scope_binds_each_member() {
        let node = NodeHandle::parallel_loop(
            body(),
            vec![input("lhs"), input("rhs")],
            vec![WireType::IndexedFamily {
                element: Box::new(WireType::Int),
                count: IntExpr::constant(3),
            }],
            ParallelLoop {
                count: IntExpr::constant(3),
                minimum_count: 0,
                index_slot: 0,
                bindings: vec![],
                input_modes: vec![LoopInputMode::Broadcast; 2],
            },
        );
        let graph = freeze(node.output(0).unwrap());
        let program = instantiate(&graph, std::num::NonZeroUsize::new(2).unwrap()).unwrap();
        let widths = |program: &GpuPreparation| {
            program
                .replay
                .iter()
                .find_map(|step| match step {
                    PreparedReplayStep::Parallel { waves, .. } => {
                        Some(waves.iter().map(|wave| wave.len()).collect::<Vec<_>>())
                    }
                    _ => None,
                })
                .unwrap()
        };
        assert_eq!(widths(&program), vec![2, 1]);
        let serial = instantiate(&graph, std::num::NonZeroUsize::new(1).unwrap()).unwrap();
        assert_eq!(widths(&serial), vec![1, 1, 1]);
        assert_eq!(program.scalar_commands.len(), 3);
        let members = &program.node_bindings[&(program.outputs[0].node.0 as u32)].0;
        assert_eq!(members.len(), 3);
        assert_eq!(members.iter().collect::<BTreeSet<_>>().len(), 3);
    }

    #[test]
    fn test_gpu_prepared_sequential_body_storage_does_not_scale_with_count() {
        let compile = |count| {
            let node = NodeHandle::sequential_loop(
                body(),
                vec![input("carried"), input("increment")],
                vec![WireType::Int],
                SequentialLoop {
                    count: IntExpr::constant(count),
                    index_slot: 0,
                    bindings: vec![],
                    carried_count: 1,
                },
            );
            instantiate(&freeze(node.output(0).unwrap()), std::num::NonZeroUsize::new(2).unwrap())
                .unwrap()
        };
        let short = compile(3);
        let long = compile(300);
        assert_eq!(short.topology.nodes.len(), long.topology.nodes.len());
        assert_eq!(short.scalar_slot_count, long.scalar_slot_count);
        let PreparedReplayStep::Sequential { count, banks, .. } = long.replay.last().unwrap()
        else {
            panic!("sequential scope must retain one loop instruction");
        };
        assert_eq!(*count, 300);
        assert_eq!(banks[0].len(), banks[1].len());
        assert!(long.node_sources.values().all(|source| source.variants.is_empty()));
    }

    #[test]
    fn test_gpu_prepared_nested_counts_keep_prefix_offsets_and_canonical_tail() {
        let nested = with_new_construction_scope(|scope| {
            let carried = input("carried");
            let increment = input("increment");
            let inner = NodeHandle::sequential_loop(
                body(),
                vec![carried.clone(), increment.clone()],
                vec![WireType::Int],
                SequentialLoop {
                    count: IntExpr::Var("iteration".into()),
                    index_slot: 1,
                    bindings: vec![],
                    carried_count: 1,
                },
            );
            SubgraphHandle::new(
                "nested",
                scope,
                vec![carried, increment],
                vec![inner.output(0).unwrap()],
            )
            .unwrap()
        });
        let outer = NodeHandle::sequential_loop(
            nested,
            vec![input("carried"), input("increment")],
            vec![WireType::Int],
            SequentialLoop {
                count: IntExpr::constant(3),
                index_slot: 0,
                bindings: vec![("iteration".into(), IntExpr::LoopIndex(0))],
                carried_count: 1,
            },
        );
        let program =
            instantiate(&freeze(outer.output(0).unwrap()), std::num::NonZeroUsize::new(2).unwrap())
                .unwrap();
        let PreparedReplayStep::Sequential { banks, tail, .. } = program.replay.last().unwrap()
        else {
            panic!("outer loop")
        };
        assert_eq!(tail.len(), 1);
        for bank in banks {
            let nested = bank
                .iter()
                .find_map(|step| match step {
                    PreparedReplayStep::Sequential { counts, offsets, tail, .. } => {
                        Some((counts, offsets, tail))
                    }
                    _ => None,
                })
                .unwrap();
            assert_eq!(nested.0.as_ref(), &[0, 1, 2]);
            assert_eq!(nested.1.as_ref(), &[0, 0, 1]);
            assert_eq!(nested.2.len(), 1);
        }
    }

    #[test]
    fn test_gpu_prepared_indexed_scalar_carried_replay_uses_finite_variants() {
        use crate::{
            backend::poly_gpu::{PreparedGpuProgram, gpu_prepared::PreparedRuntimeValue},
            transcript::SamplingMode,
        };
        use std::sync::Arc;
        let body = with_new_construction_scope(|scope| {
            let carried = input("carried");
            let index = NodeHandle::new(
                NodeKind::EvaluateInt(IntExpr::Var("iteration".into())),
                vec![],
                vec![WireType::ConstantInt],
            )
            .output(0)
            .unwrap();
            let sum = NodeHandle::new(
                NodeKind::IntBinary(IntBinaryOp::Add),
                vec![carried.clone(), index],
                vec![WireType::Int],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("indexed_scalar", scope, vec![carried], vec![sum]).unwrap()
        });
        let count = 5;
        let node = NodeHandle::sequential_loop(
            body,
            vec![input("start")],
            vec![WireType::Int],
            SequentialLoop {
                count: IntExpr::constant(count),
                index_slot: 0,
                bindings: vec![("iteration".into(), IntExpr::LoopIndex(0))],
                carried_count: 1,
            },
        );
        let program =
            instantiate(&freeze(node.output(0).unwrap()), std::num::NonZeroUsize::new(2).unwrap())
                .unwrap();
        let output = program.scalar_slots[&program.outputs[0]];
        let mut execution = PreparedGpuProgram::from_command_instances(
            vec![Box::new([])],
            Arc::new(crate::gpu_memory::GpuMemoryRegion::empty_for_test()),
            0,
            0,
        )
        .from_preparation(program)
        .unwrap();
        let mut ledger = crate::gpu_memory::GpuMemoryLedger::synthetic_for_test(
            &[crate::gpu_calibration::GpuDeviceMemory { total_bytes: 1 << 40, resident_bytes: 0 }],
            &[0],
            100,
            Some(&[(0, 1)]),
        )
        .unwrap();
        let mut allocator =
            crate::backend::poly_gpu::gpu_prepared::LedgerScalarCapacityAllocator::new(&mut ledger);
        for start in 0..3 {
            execution.initialize_runtime_roots(&[PreparedRuntimeValue::Int(start.into())]).unwrap();
            let inputs = BTreeMap::from([(
                "start".to_owned(),
                crate::backend::RuntimeValue::Int(start.into()),
            )]);
            let mut sampling = SamplingMode::Fresh;
            let result = execution
                .run_with_runtime_bindings(&inputs, &mut sampling, Some(&mut allocator))
                .unwrap();
            let expected = num_bigint::BigInt::from(start) +
                (0..count).map(num_bigint::BigInt::from).sum::<num_bigint::BigInt>();
            assert_eq!(result.scalar_slot_values()[output], ScalarValue::Int(expected));
        }
    }

    #[test]
    fn test_gpu_prepared_nested_parallel_freezes_maximum_waves_and_active_counts() {
        let nested = with_new_construction_scope(|scope| {
            let carried = input("carried");
            let increment = input("increment");
            let count = IntExpr::Add(
                Box::new(IntExpr::Var("iteration".into())),
                Box::new(IntExpr::constant(1)),
            );
            let parallel = NodeHandle::parallel_loop(
                body(),
                vec![carried.clone(), increment.clone()],
                vec![WireType::IndexedFamily {
                    element: Box::new(WireType::Int),
                    count: count.clone(),
                }],
                ParallelLoop {
                    count,
                    minimum_count: 1,
                    index_slot: 1,
                    bindings: vec![],
                    input_modes: vec![LoopInputMode::Broadcast; 2],
                },
            );
            let first = NodeHandle::new(
                NodeKind::FamilyGetStatic { index: IntExpr::constant(0) },
                vec![parallel.output(0).unwrap()],
                vec![WireType::Int],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("nested_parallel", scope, vec![carried, increment], vec![first])
                .unwrap()
        });
        let outer = NodeHandle::sequential_loop(
            nested,
            vec![input("carried"), input("increment")],
            vec![WireType::Int],
            SequentialLoop {
                count: IntExpr::constant(3),
                index_slot: 0,
                bindings: vec![("iteration".into(), IntExpr::LoopIndex(0))],
                carried_count: 1,
            },
        );
        let program =
            instantiate(&freeze(outer.output(0).unwrap()), std::num::NonZeroUsize::new(2).unwrap())
                .unwrap();
        let PreparedReplayStep::Sequential { banks, .. } = program.replay.last().unwrap() else {
            panic!("outer loop")
        };
        for bank in banks {
            let (counts, waves) = bank
                .iter()
                .find_map(|step| match step {
                    PreparedReplayStep::Parallel { counts, waves, .. } => Some((counts, waves)),
                    _ => None,
                })
                .unwrap();
            assert_eq!(counts.as_ref(), &[1, 2, 3]);
            assert_eq!(waves.iter().map(|wave| wave.len()).collect::<Vec<_>>(), vec![2, 1]);
            assert!(
                waves
                    .iter()
                    .flatten()
                    .all(|step| matches!(step, PreparedReplayStep::Subgraph { .. }))
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_threshold_exports_wide_plaintext() {
        use crate::{
            Backend, MemoryArtifactStore, RuntimeValue,
            executor::{ExecutionConfig, execute_with_config},
            transcript::SamplingMode,
        };
        use mxx_ir_core::types::MatrixType;
        use mxx_primitives::{
            matrix::gpu_dcrt_poly::GpuDCRTPolyMatrix,
            poly::{
                PolyParams,
                dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
            },
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
        let cpu = DCRTPolyParams::new(n, 3, 30, 4, None, None);
        let params = GpuDCRTPolyParams::new_with_gpu(
            n,
            cpu.to_crt().0,
            4,
            vec![device],
            Some(1),
            None,
            None,
        );
        let modulus: num_bigint::BigInt = params.modulus().as_ref().clone().into();
        let plaintext = &modulus * 2u32 + 3u32;
        let ty = WireType::Matrix(MatrixType {
            modulus: IntExpr::Const(modulus),
            ring_dimension: IntExpr::constant(n),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        });
        let input = NodeHandle::new(
            NodeKind::Input { name: "input".into(), wire_type: ty.clone(), artifact: None },
            vec![],
            vec![ty],
        )
        .output(0)
        .unwrap();
        let decoded = NodeHandle::new(
            NodeKind::ThresholdDecode {
                plaintext_modulus: IntExpr::Const(plaintext.clone()),
                length: IntExpr::constant(1),
                output_bool: false,
            },
            vec![input],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        let graph = freeze(decoded);
        let source =
            DCRTPolyUniformSampler::new().sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
        let expected = crate::backend::poly::cpu_backend([cpu])
            .threshold_decode(&source, &plaintext, 1)
            .unwrap();
        let source: crate::backend::poly_gpu::GpuFleetMatrix =
            GpuDCRTPolyMatrix::from_cpu_matrix(&params, &source).into();
        let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
        let inputs = BTreeMap::from([("input".into(), RuntimeValue::matrix(source))]);
        backend.warm_up_prepared_graph(&graph, &inputs, &ExecutionConfig::default()).unwrap();
        let mut store = MemoryArtifactStore::default();
        for _ in 0..3 {
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                inputs.clone(),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .unwrap();
            assert!(
                execute_with_config(
                    &graph,
                    &mut backend,
                    inputs.clone(),
                    &mut store,
                    SamplingMode::Fresh,
                    ExecutionConfig::default()
                )
                .is_err(),
                "deferred scalar retains its execution instance"
            );
            let RuntimeValue::Int(actual) =
                result.materialize_output("result", &mut backend, &mut store).unwrap()
            else {
                panic!("threshold scalar output")
            };
            assert_eq!(actual, &expected[0]);
        }
    }

    #[test]
    fn test_gpu_prepared_finite_shape_and_crt_variants_remain_per_node() {
        use mxx_ir_core::{node::MatrixBinaryOp, types::MatrixType};
        let fixed = MatrixType {
            modulus: IntExpr::constant(97),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
        };
        let body = with_new_construction_scope(|scope| {
            let carried = NodeHandle::new(
                NodeKind::Input {
                    name: "carry".into(),
                    wire_type: WireType::Matrix(fixed.clone()),
                    artifact: None,
                },
                vec![],
                vec![WireType::Matrix(fixed.clone())],
            )
            .output(0)
            .unwrap();
            let varying = MatrixType {
                modulus: IntExpr::Select {
                    selector: Box::new(IntExpr::Var("iteration".into())),
                    branches: vec![
                        IntExpr::constant(97),
                        IntExpr::constant(193),
                        IntExpr::constant(257),
                    ],
                },
                rows: IntExpr::Add(
                    Box::new(IntExpr::Var("iteration".into())),
                    Box::new(IntExpr::constant(1)),
                ),
                ..fixed.clone()
            };
            let sampled = NodeHandle::new(
                NodeKind::UniformResidueSample { matrix_type: varying.clone() },
                vec![],
                vec![WireType::Matrix(varying.clone())],
            )
            .output(0)
            .unwrap();
            let rebased_type = MatrixType { modulus: fixed.modulus.clone(), ..varying };
            let rebased = NodeHandle::new(
                NodeKind::CenteredRebase { modulus: fixed.modulus.clone() },
                vec![sampled],
                vec![WireType::Matrix(rebased_type)],
            )
            .output(0)
            .unwrap();
            let selected = NodeHandle::new(
                NodeKind::Slice {
                    rows: Some(IndexRange {
                        start: IntExpr::constant(0),
                        end: IntExpr::constant(1),
                    }),
                    columns: None,
                },
                vec![rebased],
                vec![WireType::Matrix(fixed.clone())],
            )
            .output(0)
            .unwrap();
            let result = NodeHandle::new(
                NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                vec![carried.clone(), selected],
                vec![WireType::Matrix(fixed.clone())],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("finite_shape", scope, vec![carried], vec![result]).unwrap()
        });
        let initial = NodeHandle::new(
            NodeKind::Input {
                name: "initial".into(),
                wire_type: WireType::Matrix(fixed.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(fixed.clone())],
        )
        .output(0)
        .unwrap();
        let node = NodeHandle::sequential_loop(
            body,
            vec![initial],
            vec![WireType::Matrix(fixed)],
            SequentialLoop {
                count: IntExpr::constant(3),
                index_slot: 0,
                bindings: vec![("iteration".into(), IntExpr::LoopIndex(0))],
                carried_count: 1,
            },
        );
        let program =
            instantiate(&freeze(node.output(0).unwrap()), std::num::NonZeroUsize::new(2).unwrap())
                .unwrap();
        let sampled = program
            .node_sources
            .values()
            .filter(|source| matches!(source.kind, NodeKind::UniformResidueSample { .. }))
            .collect::<Vec<_>>();
        assert_eq!(sampled.len(), 2);
        for source in sampled {
            assert_eq!(source.variants.len(), 3);
            assert_eq!(source.variant_indices.as_ref(), &[0, 1, 2]);
            let matrices = source
                .variant_output_types
                .iter()
                .map(|types| types[0].matrix_type().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(
                matrices.iter().map(|matrix| matrix.rows).collect::<Vec<_>>(),
                vec![1, 2, 3]
            );
            assert_eq!(
                matrices.iter().map(|matrix| matrix.modulus.clone()).collect::<Vec<_>>(),
                vec![97.into(), 193.into(), 257.into()]
            );
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_compact_hash_declared_rows_match_ordinary() {
        use crate::{
            MemoryArtifactStore, RuntimeValue,
            executor::{ExecutionConfig, execute_with_config},
            transcript::SamplingMode,
        };
        use mxx_dsl::{DslContext, Ring};
        use mxx_primitives::{
            matrix::{
                PolyMatrix, PolyMatrixSmallRhs, SmallPolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix,
            },
            poly::{
                PolyParams,
                dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
            },
            sampler::{DistType, PolyHashSampler, gpu::GpuDCRTPolyHashSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        for with_matrix in [false, true] {
            crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
            let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
            let params = GpuDCRTPolyParams::new_with_gpu(
                n,
                cpu.to_crt().0,
                4,
                vec![device],
                Some(1),
                None,
                None,
            );
            let digits =
                params.compact_decomposition_layout(true, None).unwrap().rows_per_input_row;
            let key: [u8; 32] = rand::random();
            let tag = b"prepared-compact-shape";
            let expected = GpuDCRTPolyHashSampler::<keccak_asm::Keccak256>::new()
                .sample_hash_gadget_source(&params, key, tag, 2, 1, DistType::FinRingDist)
                .gadget_decompose(true, Some(digits))
                .unwrap();
            let expected_shape = (expected.rows(), expected.columns());
            let expected = expected.to_canonical_coefficients().unwrap();
            let ring = Ring::new(params.modulus().as_ref().clone(), n as usize);
            let output = ring.hash_small_decomposed(
                ring.bytes_input("key", 32),
                tag.as_slice(),
                (2 * digits, 1),
                16,
                digits,
            );
            let mut context =
                DslContext::new("prepared-compact-shape").output("output", output).unwrap();
            let mut inputs = BTreeMap::from([("key".into(), RuntimeValue::Bytes(key.to_vec()))]);
            if with_matrix {
                context = context.output("anchor", ring.input("anchor", (1, 1))).unwrap();
                inputs.insert(
                    "anchor".into(),
                    RuntimeValue::matrix(GpuDCRTPolyMatrix::identity(&params, 1, None).into()),
                );
            }
            let graph = context.build().unwrap().validate(&ParamEnv::default()).unwrap();
            let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
            backend.warm_up_prepared_graph(&graph, &inputs, &ExecutionConfig::default()).unwrap();
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                inputs.clone(),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .unwrap();
            let RuntimeValue::SmallMatrix(actual) =
                result.materialize_output("output", &mut backend, &mut store).unwrap()
            else {
                panic!("compact hash output")
            };
            assert_eq!(actual.shards().len(), 1);
            assert_eq!(
                (actual.shards()[0].value.rows(), actual.shards()[0].value.columns()),
                expected_shape
            );
            assert_eq!(actual.shards()[0].value.to_canonical_coefficients().unwrap(), expected);
        }
    }

    #[test]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_prepared_scopes_execute_matrix_subgraph_parallel_and_carried_state() {
        use crate::{
            Backend, MemoryArtifactStore, RuntimeValue,
            executor::{ExecutionConfig, execute_with_config},
            transcript::SamplingMode,
        };
        use mxx_ir_core::types::MatrixType;
        use mxx_primitives::{
            matrix::{PolyMatrix, gpu_dcrt_poly::GpuDCRTPolyMatrix},
            poly::{
                PolyParams,
                dcrt::{gpu::GpuDCRTPolyParams, params::DCRTPolyParams},
            },
            sampler::{DistType, PolyUniformSampler, uniform::DCRTPolyUniformSampler},
        };
        let n = std::env::var("MXX_PRIMITIVE_TEST_RING_DIMENSION")
            .map(|value| value.parse::<u32>().unwrap())
            .unwrap_or(32);
        let count = std::env::var("MXX_PRIMITIVE_TEST_LOOP_COUNT")
            .map(|value| value.parse::<usize>().unwrap())
            .unwrap_or(3);
        let device = mxx_primitives::poly::dcrt::gpu::detected_gpu_device_ids()[0];
        for mode in 0..3 {
            crate::backend::poly_gpu::wait_for_gpu_test_context_quiescence(device);
            let cpu = DCRTPolyParams::new(n, 2, 30, 4, None, None);
            let params = GpuDCRTPolyParams::new_with_gpu(
                n,
                cpu.to_crt().0,
                4,
                vec![device],
                Some(1),
                None,
                None,
            );
            let matrix = MatrixType {
                modulus: IntExpr::Const(params.modulus().as_ref().clone().into()),
                ring_dimension: IntExpr::constant(n),
                rows: IntExpr::constant(1),
                columns: IntExpr::constant(1),
            };
            let ty = WireType::Matrix(matrix);
            let matrix_input = |name: &str| {
                NodeHandle::new(
                    NodeKind::Input { name: name.into(), wire_type: ty.clone(), artifact: None },
                    vec![],
                    vec![ty.clone()],
                )
                .output(0)
                .unwrap()
            };
            let add = with_new_construction_scope(|scope| {
                let left = matrix_input("left");
                let right = matrix_input("right");
                let sum = NodeHandle::new(
                    NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                    vec![left.clone(), right.clone()],
                    vec![ty.clone()],
                )
                .output(0)
                .unwrap();
                SubgraphHandle::new("matrix_add", scope, vec![left, right], vec![sum]).unwrap()
            });
            let left = matrix_input("left");
            let right = matrix_input("right");
            let output = match mode {
                0 => NodeHandle::subgraph_call(add, vec![left, right], vec![], vec![None, None])
                    .output(0)
                    .unwrap(),
                1 => {
                    let family = NodeHandle::parallel_loop(
                        add,
                        vec![left, right],
                        vec![WireType::IndexedFamily {
                            element: Box::new(ty.clone()),
                            count: IntExpr::constant(count),
                        }],
                        ParallelLoop {
                            count: IntExpr::constant(count),
                            minimum_count: 2,
                            index_slot: 0,
                            bindings: vec![],
                            input_modes: vec![LoopInputMode::Broadcast; 2],
                        },
                    )
                    .output(0)
                    .unwrap();
                    NodeHandle::new(
                        NodeKind::FamilyGetStatic { index: IntExpr::constant(count - 1) },
                        vec![family],
                        vec![ty.clone()],
                    )
                    .output(0)
                    .unwrap()
                }
                _ => NodeHandle::sequential_loop(
                    add,
                    vec![left, right],
                    vec![ty.clone()],
                    SequentialLoop {
                        count: IntExpr::constant(count),
                        index_slot: 0,
                        bindings: vec![],
                        carried_count: 1,
                    },
                )
                .output(0)
                .unwrap(),
            };
            let graph = freeze(output);
            let sampler = DCRTPolyUniformSampler::new();
            let left = sampler.sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
            let right = sampler.sample_uniform(&cpu, 1, 1, DistType::FinRingDist);
            let mut expected = left.clone();
            for _ in 0..if mode == 2 { count } else { 1 } {
                expected = &expected + &right;
            }
            let expected =
                GpuDCRTPolyMatrix::from_cpu_matrix(&params, &expected).to_compact_bytes();
            let inputs = BTreeMap::from([
                (
                    "left".into(),
                    RuntimeValue::matrix(GpuDCRTPolyMatrix::from_cpu_matrix(&params, &left).into()),
                ),
                (
                    "right".into(),
                    RuntimeValue::matrix(
                        GpuDCRTPolyMatrix::from_cpu_matrix(&params, &right).into(),
                    ),
                ),
            ]);
            let mut backend = crate::backend::poly_gpu::gpu_backend_on([params], [device]);
            backend.warm_up_prepared_graph(&graph, &inputs, &ExecutionConfig::default()).unwrap();
            let mut store = MemoryArtifactStore::default();
            let mut result = execute_with_config(
                &graph,
                &mut backend,
                inputs.clone(),
                &mut store,
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .unwrap();
            let RuntimeValue::Matrix(actual) =
                result.materialize_output("result", &mut backend, &mut store).unwrap()
            else {
                panic!("matrix scope result")
            };
            assert_eq!(backend.matrix_to_bytes(&actual).unwrap(), expected, "scope mode {mode}");
            if mode == 0 {
                let config = ExecutionConfig {
                    max_parallel_instances: std::num::NonZeroUsize::MIN,
                    ..ExecutionConfig::default()
                };
                backend.warm_up_prepared_graph(&graph, &inputs, &config).unwrap();
                let first = execute_with_config(
                    &graph,
                    &mut backend,
                    inputs.clone(),
                    &mut store,
                    SamplingMode::Fresh,
                    config,
                )
                .unwrap();
                let second = execute_with_config(
                    &graph,
                    &mut backend,
                    inputs.clone(),
                    &mut store,
                    SamplingMode::Fresh,
                    config,
                )
                .unwrap();
                assert!(
                    execute_with_config(
                        &graph,
                        &mut backend,
                        inputs.clone(),
                        &mut store,
                        SamplingMode::Fresh,
                        config
                    )
                    .err()
                    .expect("third live request must be busy")
                    .to_string()
                    .contains("busy")
                );
                drop((first, second));
                let again = execute_with_config(
                    &graph,
                    &mut backend,
                    inputs,
                    &mut store,
                    SamplingMode::Fresh,
                    config,
                )
                .unwrap();
                drop(again);
                assert_eq!(
                    backend.matrix_to_bytes(&actual).unwrap(),
                    expected,
                    "old output survives replacement"
                );
            }
        }
    }
}
