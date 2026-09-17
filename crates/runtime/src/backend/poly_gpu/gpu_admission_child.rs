//! Exact child-scope evaluation at a parent node. Candidate demand is projected
//! into the parent's ordered timeline through one shared append operation.

use super::*;

pub(super) struct ChildNodePlan {
    pub(super) scope_id: FrozenGraphScopeId,
    pub(super) bindings: ParamEnv,
    pub(super) operation: crate::gpu_invocation::GpuNodeOperation,
    pub(super) node_position: usize,
    pub(super) input_layouts: BTreeMap<WireRef, GpuInventoryValue>,
    pub(super) output_ends: Vec<usize>,
    pub(super) capture_trace: bool,
    pub(super) expected_layout_updates: BTreeMap<WireRef, GpuInventoryValue>,
}

struct ChildContext {
    ty: ConcreteMatrixType,
    demand: Arc<GpuContextDemand>,
    projection: ClaimProjection,
    typed_claims: Vec<GpuTracedClaim>,
}

enum CarriedOwner {
    Matrix { ty: ConcreteMatrixType },
    Compact { ty: ConcreteMatrixType, layout: GpuPreparedWorkspaceLayout },
}

impl CarriedOwner {
    fn matrix_type(&self) -> &ConcreteMatrixType {
        match self {
            Self::Matrix { ty } | Self::Compact { ty, .. } => ty,
        }
    }
}

pub(super) struct ChildNodeResult {
    contexts: BTreeMap<(String, usize), ChildContext>,
    pub(super) layout_updates: BTreeMap<WireRef, GpuInventoryValue>,
    staged_ports: Vec<bool>,
    child_outputs: Vec<WireRef>,
    node: mxx_ir_core::types::NodeId,
    position: usize,
    output_end: usize,
    parallel: bool,
    carried: Vec<CarriedOwner>,
}

impl ChildNodeResult {
    /// Append requested shapes and lifetimes in the original projection order.
    /// Capacity slots belong to the parent timeline and are assigned here;
    /// child slot numbers are used only to resolve escaping-owner provenance.
    pub(super) fn append(
        &self,
        demand: &mut BTreeMap<(String, usize), (ConcreteMatrixType, GpuContextDemand)>,
    ) {
        for (key, child) in &self.contexts {
            if child.demand.matrices.is_empty() &&
                child.demand.layouts.is_empty() &&
                child.typed_claims.is_empty()
            {
                continue;
            }
            let context = &mut demand
                .entry(key.clone())
                .or_insert_with(|| (child.ty.clone(), GpuContextDemand::default()))
                .1;
            let escaping = |claims: &[usize]| {
                self.child_outputs
                    .iter()
                    .enumerate()
                    .filter(|entry| !self.staged_ports.get(entry.0).copied().unwrap_or(false))
                    .flat_map(|(port, output)| {
                        claims.iter().copied().map(move |claim| ValueOrigin::Child {
                            instance: 0,
                            wire: WireRef { node: self.node, port: Port(port as u32) },
                            output: *output,
                            parallel: self.parallel,
                            demand: Arc::clone(&child.demand),
                            claim,
                        })
                    })
                    .collect::<Vec<_>>()
            };
            for (index, ((rows, columns), end)) in
                child.demand.matrices.iter().zip(&child.demand.matrix_until).enumerate()
            {
                let parent_end = if *end == usize::MAX { self.output_end } else { self.position };
                context.matrix(self.position, parent_end, *rows, *columns);
                if *end == usize::MAX {
                    context.matrix_demands.last_mut().unwrap().owners =
                        escaping(&child.projection.matrix_indices[index]);
                }
            }
            // Exact typed envelopes supplement ordinary capacity shapes just
            // as in the full evaluator; they are not an alternative branch.
            //
            // Keep the envelope's exact occurrences associated with the
            // projected claim indices.  The old child path appended every
            // exact claim at the child node only, which made an escaping
            // output look as though its level/format owner had already been
            // released.  That is especially visible when a child output is
            // consumed by a later GPU node: the ordinary shape survived to
            // `output_end`, but its exact typed claim did not.
            let mut exact_indices = child
                .projection
                .claims
                .iter()
                .enumerate()
                .filter_map(|(index, _)| {
                    matches!(
                        child.projection.provenance[index],
                        ClaimProvenance::SlotExact |
                            ClaimProvenance::RequiredExact |
                            ClaimProvenance::AlternativeEnvelopeIncrement
                    )
                    .then_some(index)
                })
                .collect::<Vec<_>>();
            let mut escaping_exact = std::collections::BTreeSet::new();
            for (index, end) in child.demand.matrix_until.iter().enumerate() {
                if *end == usize::MAX {
                    escaping_exact.extend(child.projection.matrix_indices[index].iter().copied());
                }
            }
            for &claim in &child.typed_claims {
                let exact_index = exact_indices
                    .iter()
                    .position(|&index| {
                        claim_key(child.projection.claims[index]) == claim_key(claim)
                    })
                    .map(|offset| exact_indices.remove(offset));
                let escapes = exact_index.map_or(false, |index| escaping_exact.contains(&index));
                let parent_end = if escapes { self.output_end } else { self.position };
                context.matrix_claim(self.position, parent_end, claim);
                if escapes {
                    let owner_index =
                        exact_index.expect("escaping exact claim must have a projected index");
                    context.matrix_demands.last_mut().unwrap().owners =
                        escaping(std::slice::from_ref(&owner_index));
                }
            }
            for (index, ((layout, end), shared)) in child
                .demand
                .layouts
                .iter()
                .zip(&child.demand.layout_until)
                .zip(&child.demand.layout_shared)
                .enumerate()
            {
                let parent_end = if *end == usize::MAX { self.output_end } else { self.position };
                context.workspace(self.position, parent_end, *layout, *shared);
                if *end == usize::MAX {
                    context.workspace_demands.last_mut().unwrap().owners =
                        escaping(std::slice::from_ref(&child.projection.workspace_indices[index]));
                }
            }
        }
        // A sequential iteration retains its incoming state while producing
        // the replacement. These additional owners are ordinary parent claims.
        for owner in &self.carried {
            let ty = owner.matrix_type();
            let context = &mut demand
                .entry((ty.modulus.to_string(), ty.ring_dimension))
                .or_insert_with(|| (ty.clone(), GpuContextDemand::default()))
                .1;
            match owner {
                CarriedOwner::Matrix { ty } => {
                    context.matrix(self.position, self.position, ty.rows, ty.columns)
                }
                CarriedOwner::Compact { layout, .. } => {
                    context.workspace(self.position, self.position, *layout, false)
                }
            }
        }
    }
}

impl GpuDcrtBackend {
    pub(super) fn evaluate_child_node(
        &mut self,
        validated: &ValidatedGraph,
        plan: &ChildNodePlan,
        scratch_columns: usize,
        wave: usize,
        warm_up: bool,
    ) -> Result<ChildNodeResult, PolyBackendError> {
        let scope_id = &plan.scope_id;
        let bindings = &plan.bindings;
        let node_position = plan.node_position;
        let id = mxx_ir_core::types::NodeId(node_position as u64);
        let kind = plan.operation.kind();
        let arguments = plan.operation.argument_wires();
        let output_types = plan.operation.outputs();
        let column_layouts = &plan.input_layouts;
        let capture_trace = plan.capture_trace;
        let child_id = validated.source.child_scope_id(scope_id, id).expect("validated child node");
        let mut result = ChildNodeResult {
            contexts: BTreeMap::new(),
            layout_updates: BTreeMap::new(),
            staged_ports: Vec::new(),
            child_outputs: validated.source.scope(&child_id).unwrap().outputs().to_vec(),
            node: id,
            position: node_position,
            output_end: node_position,
            parallel: matches!(kind, NodeKind::ParallelLoop(_)),
            carried: Vec::new(),
        };
        let (child_bindings, loop_slot, count) = match kind {
            NodeKind::SubgraphCall(call) => (&call.bindings, None, 1),
            NodeKind::ParallelLoop(body) => (
                &body.bindings,
                Some(body.index_slot),
                body.count
                    .evaluate(bindings)
                    .ok()
                    .and_then(|value| value.to_usize())
                    .ok_or(PolyBackendError::InvalidInteger)?,
            ),
            NodeKind::SequentialLoop(body) => (
                &body.bindings,
                Some(body.index_slot),
                body.count
                    .evaluate(bindings)
                    .ok()
                    .and_then(|value| value.to_usize())
                    .ok_or(PolyBackendError::InvalidInteger)?,
            ),
            _ => unreachable!("validated child scope"),
        };
        if count == 0 {
            return Ok(result);
        }
        let mut child_inputs = arguments
            .iter()
            .map(|wire| column_layouts.get(wire).cloned().unwrap_or_default())
            .zip(validated.source.scope(&child_id).unwrap().inputs().iter().copied())
            .map(|(mut cuts, wire)| {
                if matches!(kind, NodeKind::ParallelLoop(_)) &&
                    !matches!(
                        validated.scope(&child_id).unwrap().wire_types[&wire],
                        ConcreteWireType::IndexedFamily { .. }
                    )
                {
                    cuts.lazy = false;
                }
                (wire, cuts)
            })
            .collect::<BTreeMap<_, _>>();
        let parallel_wave =
            if matches!(kind, NodeKind::ParallelLoop(_)) { wave.min(count) } else { 1 };
        let mut child = BTreeMap::<(String, usize), (ConcreteMatrixType, GpuContextDemand)>::new();
        let mut child_outputs = Vec::<GpuInventoryValue>::new();
        // Walk finite alternatives on CPU and immediately fold their
        // resource bounds. Never store descriptors for future bodies.
        // Explicit discovery caches primitive classes, so repeated
        // bindings do not add GPU probes. Production only looks up
        // those templates and errors if warmup missed a class.
        for index in 0..count {
            let child_env = bindings
                .child(child_bindings, loop_slot.map(|slot| (slot, index)))
                .map_err(|_| PolyBackendError::InvalidInteger)?;
            let child_demand = self.graph_admission_demand(
                validated,
                &child_id,
                &child_env,
                None,
                capture_trace,
                Some(&child_inputs),
                scratch_columns,
                if matches!(kind, NodeKind::ParallelLoop(_)) { parallel_wave } else { wave },
                warm_up,
            )?;
            let alternative = child_demand.0;
            let outputs = child_demand.1;
            if let NodeKind::SequentialLoop(body) = kind {
                // The next iteration receives these actual carried
                // layouts, not the original loop inputs. In particular,
                // a staged descriptor can become an eager fragmented
                // owner (or the reverse) after the first iteration.
                for (wire, layout) in validated
                    .source
                    .scope(&child_id)
                    .unwrap()
                    .inputs()
                    .iter()
                    .take(body.carried_count)
                    .zip(&outputs)
                {
                    child_inputs.insert(*wire, layout.clone());
                }
            }
            for (key, (ty, alternative)) in alternative {
                child
                    .entry(key)
                    .or_insert_with(|| (ty, GpuContextDemand::default()))
                    .1
                    .include_alternative(alternative);
            }
            if child_outputs.is_empty() || matches!(kind, NodeKind::SequentialLoop(_)) {
                // A sequential loop returns only its final iteration's
                // layout. Earlier iterations contribute to peak demand
                // above, but are not alternative returned partitions.
                child_outputs = outputs;
            } else {
                for (layout, alternative) in child_outputs.iter_mut().zip(outputs) {
                    layout.include_alternative(alternative);
                }
            }
        }
        // Inventory reads the executor's shared per-port staging
        // decision instead of a second, node-level rule, using the same
        // configured wave the backing is provisioned for: a parallel
        // loop whose whole count fits one wave keeps its members as
        // device owners, exactly as production does when it admits that
        // wave. A body the fleet cannot run simultaneously is never
        // planned as unstaged.
        let staged_ports = if matches!(kind, NodeKind::ParallelLoop(_)) {
            let child_scope = validated.source.scope(&child_id).expect("validated child scope");
            let retained = crate::executor::retained_loop_output_ports(
                validated,
                scope_id,
                id,
                child_scope.outputs(),
            );
            crate::executor::staged_loop_outputs(scope_id, count, parallel_wave, &retained)
        } else {
            Vec::new()
        };
        // Per output port: a staged port returns a fresh complete matrix
        // through host storage, so its previous device fragments do not
        // survive; an unstaged port keeps the child's derived layout and
        // lives until its own owner end. A retained child output that
        // maps to any unstaged port stays charged to that port's end.
        let mut output_end = node_position;
        for (port, boundaries) in child_outputs.into_iter().enumerate() {
            if staged_ports.get(port).copied().unwrap_or(false) {
                let Some(output) = output_types.get(port) else { continue };
                let mut leaf = output;
                while let ConcreteWireType::IndexedFamily { element, .. } = leaf {
                    leaf = element;
                }
                if let Some(ty) = leaf.matrix_type() {
                    let mut layout = GpuInventoryValue::new(vec![0, ty.columns]);
                    layout.lazy = true;
                    result
                        .layout_updates
                        .insert(WireRef { node: id, port: Port(port as u32) }, layout);
                } else if let Some(bound) = bound_of(leaf) {
                    let ty = bound.0;
                    let mut layout = GpuInventoryValue::new(vec![0, ty.columns]);
                    layout.lazy = true;
                    result
                        .layout_updates
                        .insert(WireRef { node: id, port: Port(port as u32) }, layout);
                }
            } else {
                result
                    .layout_updates
                    .insert(WireRef { node: id, port: Port(port as u32) }, boundaries);
                output_end = output_end.max(plan.output_ends[port]);
            }
        }

        result.output_end = output_end;
        result.staged_ports = staged_ports;
        for (key, (ty, child)) in child {
            let child = Arc::new(GpuContextDemand::simultaneous(
                std::iter::repeat_with(|| child.clone()).take(parallel_wave),
            ));
            let projection =
                child.claim_projection(self.devices[0].1.parameters(&ty)?.crt_depth() - 1);
            let typed_claims = child.matrix_claim_envelope();
            result
                .contexts
                .insert(key, ChildContext { ty, demand: child, projection, typed_claims });
        }
        if matches!(kind, NodeKind::SequentialLoop(_)) {
            for ty in output_types.iter().flat_map(family_leaves).map(|entry| entry.1) {
                if let ConcreteWireType::Matrix(ty) = ty {
                    result.carried.push(CarriedOwner::Matrix { ty: ty.clone() });
                } else if let Some((ty, bound)) = bound_of(ty) {
                    let params = self.devices[0].1.parameters(ty)?;
                    result.carried.push(CarriedOwner::Compact {
                        ty: ty.clone(),
                        layout: compact_payload(params, ty.rows, ty.columns, &bound)?,
                    });
                }
            }
        }
        Ok(result)
    }
}
