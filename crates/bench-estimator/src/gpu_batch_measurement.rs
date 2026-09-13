//! Frozen observations for bounded synthetic batches in the normal estimator.
//! Collection walks CPU metadata; only `measure_collected` executes representatives.

use super::*;
use crate::dataflow::{LayoutOwner, ScopeSiblingState, ValueLayout};
use mxx_ir_core::{FrozenGraphScopeId, NodeId, Port, ValidatedGraph, WireRef};
use std::collections::BTreeMap;

pub(super) struct BatchRequest {
    pub inputs: Vec<Vec<Option<mxx_runtime::backend::poly_gpu::GpuMatrixDescriptor>>>,
    pub request: PendingMeasurement,
    pub owners: Vec<Vec<usize>>,
    pub column_cap: usize,
    pub plans: Option<Vec<mxx_runtime::backend::poly_gpu::GpuAdmittedInvocationSummary>>,
    pub native: Option<(
        mxx_runtime::gpu_invocation::GpuNodeOperation,
        Vec<(ConcreteMatrixType, Vec<mxx_primitives::matrix::gpu_dcrt_poly::GpuTracedClaim>)>,
    )>,
}

impl GpuNodeMeasurementBackend {
    pub(super) fn dataflow_batch_measurement(
        &mut self,
        graph: &ValidatedGraph,
        scope_id: &FrozenGraphScopeId,
        id: NodeId,
        instances: &[ScopeSiblingState],
    ) -> Result<Option<NodeMeasurement>, GpuMeasurementError> {
        if self.collecting {
            return Ok(None);
        }
        let scope = graph.source.scope(scope_id).expect("validated scope");
        let handle = scope.node(id).expect("validated node");
        if Self::zero_cost(handle.kind()) {
            return Ok(Some(NodeMeasurement { independent_wave_count: 0, ..Default::default() }));
        }
        let arguments = scope.arguments(handle).expect("validated arguments");
        let requests = instances
            .par_iter()
            .map(|instance| {
                let concrete_argument_types = arguments
                    .iter()
                    .map(|&wire| {
                        graph
                            .concrete_wire_type(scope_id, wire, &instance.bindings)
                            .map_err(|error| GpuMeasurementError(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let concrete_output_types = (0..handle.output_types().len())
                    .map(|port| {
                        graph
                            .concrete_wire_type(
                                scope_id,
                                WireRef { node: id, port: Port(port as u32) },
                                &instance.bindings,
                            )
                            .map_err(|error| GpuMeasurementError(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let node = MeasurementNode {
                    scope: scope_id,
                    id,
                    kind: handle.kind(),
                    arguments: &arguments,
                    argument_kinds: &[],
                    argument_types: &[],
                    output_types: handle.output_types(),
                    concrete_argument_types,
                    concrete_output_types,
                };
                Ok(PendingMeasurement {
                    key: Self::measurement_key(&node, &instance.bindings)?,
                    scope: scope_id.clone(),
                    id,
                    kind: handle.kind().clone(),
                    concrete_argument_types: node.concrete_argument_types,
                    concrete_output_types: node.concrete_output_types,
                    bindings: instance.bindings.clone(),
                    preimage_sample: matches!(handle.kind(), NodeKind::PreimageSample { .. }),
                })
            })
            .collect::<Result<Vec<_>, GpuMeasurementError>>()?;
        let mut request = requests[0].clone();
        if requests.iter().any(|member| member.key != request.key) {
            return Err(GpuMeasurementError(format!(
                "joint GPU measurement for heterogeneous sibling parameters at {scope_id:?} node {id:?} is not implemented"
            )));
        }
        // Canonical first-use IDs preserve captured/shared operands without tying
        // the observation to an invocation ordinal or to graph-local wire IDs.
        let mut canonical = BTreeMap::<LayoutOwner, usize>::new();
        let mut pending_owners = BTreeMap::new();
        let mut next = 0;
        let owners = instances
            .iter()
            .map(|instance| {
                arguments
                    .iter()
                    .map(|wire| {
                        let owner = match &instance.values[wire] {
                            ValueLayout::Leaf(_, _, owner) => owner.as_ref(),
                            ValueLayout::Family { .. } => None,
                        };
                        if let Some(owner) = owner {
                            *canonical.entry(owner.clone()).or_insert_with(|| {
                                let id = next;
                                next += 1;
                                id
                            })
                        } else {
                            *pending_owners.entry((instance.index, *wire)).or_insert_with(|| {
                                let id = next;
                                next += 1;
                                id
                            })
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let graph_key = self.active_dataflow.expect("prepared dataflow graph");
        let inventory = self.admission_graphs[&graph_key].inventory.as_ref().unwrap();
        let column_cap = inventory.column_cap();
        let plans = inventory.selected_node_plans(id).map(<[_]>::to_vec);
        let inputs = inventory.selected_node_inputs(id).map(<[_]>::to_vec).unwrap_or_default();
        let input_key = inputs
            .iter()
            .map(|member| {
                member
                    .iter()
                    .map(|input| {
                        input.as_ref().map(|input| {
                            input
                                .input_layout
                                .iter()
                                .map(|fragment| {
                                    (
                                        fragment.device,
                                        fragment.context,
                                        fragment.start,
                                        fragment.end,
                                        fragment.level,
                                        fragment.evaluation,
                                    )
                                })
                                .collect::<Vec<_>>()
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let native = plans
            .as_ref()
            .map(|_| {
                let node = mxx_runtime::gpu_invocation::GpuNodeOperation::new(
                    graph,
                    scope_id,
                    id,
                    &request.bindings,
                )
                .map_err(GpuMeasurementError)?;
                let setup = inventory.selected_node_setup(id).unwrap().to_vec();
                Ok((node, setup))
            })
            .transpose()?;
        let key =
            encoding::hash_canonical(&(request.key, &owners, &column_cap, &plans, &input_key))
                .map_err(|error| GpuMeasurementError(error.to_string()))?;
        if let Some(measurement) = self.measured_batches.get(&key) {
            return Ok(Some(measurement.clone()));
        }
        if self.batch_collection {
            request.key = key;
            self.pending_batches.entry(key).or_insert(BatchRequest {
                inputs,
                request,
                owners,
                column_cap,
                plans,
                native,
            });
            return Ok(Some(NodeMeasurement::default()));
        }
        Err(GpuMeasurementError(format!(
            "GPU batch class at {scope_id:?} node {id:?} was not measured during explicit warmup"
        )))
    }

    /// Measure the full primitive class through native prepared submission.
    /// The native observer measures wave classes; no protocol graph is executed.
    pub(super) fn measure_planned_batch(
        &mut self,
        request: &PendingMeasurement,
        batch: &BatchRequest,
    ) -> Result<NodeMeasurement, GpuMeasurementError> {
        use mxx_runtime::gpu_measurement::GpuAdmittedMeasurement;
        if batch.plans.as_ref().unwrap().iter().all(|plan| plan.rows == 0 || plan.columns == 0) {
            return Ok(NodeMeasurement { independent_wave_count: 0, ..Default::default() });
        }
        let (node, claims) = batch.native.as_ref().expect("selected native class");
        let representative = RepresentativeMeasurement {
            kind: request.kind.clone(),
            concrete_argument_types: request.concrete_argument_types.clone(),
            concrete_output_types: request.concrete_output_types.clone(),
            fixed_arguments: Self::fixed_arguments(&request.kind, &request.concrete_argument_types),
            output_range: None,
        };
        let backend = self.prepared_backend.as_mut().expect("explicitly prepared fleet");
        let prepared = Self::prepare_batch(backend, request, &representative, batch)?;
        backend
            .prepare_measurement_storage(claims)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let matrices = prepared.iter().map(|member| member.arguments.clone()).collect::<Vec<_>>();
        let compact =
            prepared.iter().map(|member| member.small_arguments.clone()).collect::<Vec<_>>();
        let measurement_node = MeasurementNode {
            scope: &request.scope,
            id: request.id,
            kind: &request.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: request.concrete_argument_types.clone(),
            concrete_output_types: request.concrete_output_types.clone(),
        };
        let iterations = self.harness.measured_iterations;
        let mut total = NodeMeasurement { independent_wave_count: 0, ..Default::default() };
        for iteration in 0..self.harness.warm_up_iterations + iterations {
            // Aggregate as observations arrive; retained CPU state is bounded
            // by the sibling count, not the number of column waves.
            let state = Arc::new(std::sync::Mutex::new((
                NodeMeasurement { independent_wave_count: 0, ..Default::default() },
                0.0_f64,
                Vec::new(),
            )));
            let observed_state = state.clone();
            backend.set_admitted_measurement_sink(Some(Box::new(move |event| {
                let mut state = observed_state.lock().unwrap();
                let (observed, wave_latency, schedules) = &mut *state;
                let (timing, wave) = match event {
                    GpuAdmittedMeasurement::Invocation { plans, .. } => {
                        schedules.extend(plans.into_iter().map(|plan| plan.schedule));
                        return;
                    }
                    GpuAdmittedMeasurement::Wave { timing, .. } => (timing, true),
                    GpuAdmittedMeasurement::InputPreparation { timing, .. } |
                    GpuAdmittedMeasurement::OutputInitialization(timing) => (timing, false),
                    _ => return,
                };
                observed.work_seconds +=
                    timing.device_elapsed_seconds.iter().map(|(_, seconds)| seconds).sum::<f64>();
                observed.cumulative_wave_seconds += timing.fleet_wall_seconds;
                if wave {
                    *wave_latency = wave_latency.max(timing.fleet_wall_seconds);
                } else {
                    observed.latency_seconds += timing.fleet_wall_seconds;
                }
                observed.independent_wave_count += usize::from(wave);
                let bytes = timing
                    .prepared_memory
                    .iter()
                    .map(|memory| memory.peak_bytes.saturating_sub(memory.baseline_bytes) as u64)
                    .sum::<u64>();
                observed.measured_wave_workspace_bytes =
                    observed.measured_wave_workspace_bytes.max(bytes);
                observed.workspace_bytes = observed.workspace_bytes.saturating_add(bytes);
            })));
            let result = (|| {
                backend
                    .admit_measurement_node(node, &matrices, &compact, batch.column_cap)
                    .map_err(|error| GpuMeasurementError(error.to_string()))?;
                let outputs =
                    Self::run_node(backend, &measurement_node, &request.bindings, &prepared, None)?;
                outputs.into_iter().for_each(GpuMeasurementOutput::retire);
                Ok::<_, GpuMeasurementError>(())
            })();
            backend.set_admitted_measurement_sink(None);
            result?;
            let (mut observed, wave_latency, schedules) =
                Arc::try_unwrap(state).unwrap().into_inner().unwrap();
            let selected = batch
                .plans
                .as_ref()
                .unwrap()
                .iter()
                .map(|plan| plan.plan.schedule.clone())
                .collect::<Vec<_>>();
            if schedules != selected {
                return Err(GpuMeasurementError(format!(
                    "native measured schedules differ from selected class at {:?} {:?}: native={schedules:?}, selected={selected:?}",
                    request.scope, request.id
                )));
            }
            if iteration < self.harness.warm_up_iterations {
                continue;
            }
            observed.latency_seconds += wave_latency;
            total.work_seconds += observed.work_seconds / iterations as f64;
            total.cumulative_wave_seconds += observed.cumulative_wave_seconds / iterations as f64;
            total.latency_seconds += observed.latency_seconds / iterations as f64;
            total.independent_wave_count = observed.independent_wave_count;
            total.measured_wave_workspace_bytes =
                total.measured_wave_workspace_bytes.max(observed.measured_wave_workspace_bytes);
            total.workspace_bytes = total.workspace_bytes.max(observed.workspace_bytes);
        }
        info!(scope = ?request.scope, node = request.id.0, siblings = prepared.len(),
            column_cap = batch.column_cap, work_seconds = total.work_seconds,
            latency_seconds = total.latency_seconds,
            cumulative_wave_seconds = total.cumulative_wave_seconds,
            wave_count = total.independent_wave_count,
            measured_wave_workspace_bytes = total.measured_wave_workspace_bytes,
            ideal_concurrent_workspace_bytes = total.workspace_bytes,
            scenario = "native prepared storage; synthetic operands",
            memory_metric = "incremental prepared occupied bytes above retained baseline",
            "measured prepared primitive class");
        Ok(total)
    }

    pub(super) fn prepare_batch(
        backend: &mut GpuDcrtBackend,
        request: &PendingMeasurement,
        representative: &RepresentativeMeasurement,
        batch: &BatchRequest,
    ) -> Result<Vec<PreparedMeasurement>, GpuMeasurementError> {
        let node = MeasurementNode {
            scope: &request.scope,
            id: request.id,
            kind: &representative.kind,
            arguments: &[],
            argument_kinds: &[],
            argument_types: &[],
            output_types: &[],
            concrete_argument_types: representative.concrete_argument_types.clone(),
            concrete_output_types: representative.concrete_output_types.clone(),
        };
        // Fixed operands (especially Preimage's public matrix) are complete
        // resident owners, even when their width exceeds the measured output.
        let columns = representative
            .concrete_argument_types
            .iter()
            .chain(&representative.concrete_output_types)
            .filter_map(Self::matrix_columns)
            .max()
            .unwrap_or(1)
            .max(1);
        let operation = Self::calibration_operation_key(request)?;
        backend.set_column_widths_for_operation(
            operation,
            GpuColumnWidths { gpu0: Some(columns), nonzero: Some(columns) },
        );
        backend
            .select_operation(operation, true)
            .map_err(|error| GpuMeasurementError(error.to_string()))?;
        let projections = if matches!(request.kind, NodeKind::Concat { axis: ConcatAxis::Columns }) &&
            let Some(range) = representative.output_range.as_ref()
        {
            let mut start = 0;
            request
                .concrete_argument_types
                .iter()
                .enumerate()
                .filter_map(|(index, ty)| {
                    let end = start + Self::matrix_columns(ty).expect("concat matrix");
                    let overlap = (range.start.max(start), range.end.min(end));
                    let projection = (overlap.0 < overlap.1).then_some((
                        index,
                        Some((overlap.0 - start, overlap.1.saturating_sub(start))),
                    ));
                    start = end;
                    projection
                })
                .collect::<Vec<_>>()
        } else {
            (0..node.concrete_argument_types.len()).map(|index| (index, None)).collect()
        };
        // Width-local operands only alias when the same logical owner also has
        // the same projected descriptor. Fixed and sliced operands can differ.
        let mut matrices = HashMap::new();
        let mut compact = HashMap::new();
        let mut trapdoors = HashMap::new();
        batch
            .owners
            .iter()
            .enumerate()
            .map(|(member, owners)| {
                let keys = projections
                    .iter()
                    .enumerate()
                    .map(|(index, (source, range))| {
                        encoding::hash_canonical(&(
                            owners[*source],
                            range,
                            &node.concrete_argument_types[index],
                        ))
                        .map_err(|error| GpuMeasurementError(error.to_string()))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let present = keys
                    .iter()
                    .map(|key| matrices.contains_key(key) || compact.contains_key(key))
                    .collect::<Vec<_>>();
                let preimage = matches!(node.kind, NodeKind::PreimageSample { .. });
                let new_trapdoor = preimage && !trapdoors.contains_key(&owners[1]);
                let mut prepared = Self::prepare(
                    backend,
                    &node,
                    &request.bindings,
                    if new_trapdoor { None } else { Some((&present, false)) },
                    batch.inputs.get(member).map(Vec::as_slice),
                )?;
                for (index, key) in keys.into_iter().enumerate() {
                    if let Some(value) = &prepared.arguments[index] {
                        matrices.entry(key).or_insert_with(|| value.clone());
                    }
                    if let Some(value) = &prepared.small_arguments[index] {
                        compact.entry(key).or_insert_with(|| value.clone());
                    }
                    prepared.arguments[index] = matrices.get(&key).cloned();
                    prepared.small_arguments[index] = compact.get(&key).cloned();
                }
                if preimage {
                    if let Some(trapdoor) = prepared.preimage_trapdoor.take() {
                        trapdoors.entry(owners[1]).or_insert(trapdoor);
                    }
                    prepared.preimage_trapdoor = trapdoors.get(&owners[1]).cloned();
                    prepared.preimage_target = Some(
                        backend
                            .preimage_target(
                                prepared.arguments[2].as_ref().expect("prepared target").clone(),
                            )
                            .map_err(|error| GpuMeasurementError(error.to_string()))?
                            .0,
                    );
                }
                prepared.finish();
                Ok(prepared)
            })
            .collect()
    }

    pub(super) fn zero_cost(kind: &NodeKind) -> bool {
        matches!(
            kind,
            NodeKind::Input { .. } |
                NodeKind::ConstantInt(_) |
                NodeKind::EvaluateInt(_) |
                NodeKind::ConstantReal(_) |
                NodeKind::ConstantBool(_) |
                NodeKind::TrapdoorPublic |
                NodeKind::IntBinary(_) |
                NodeKind::IntCompare(_) |
                NodeKind::BitExtract { .. } |
                NodeKind::IntToReal |
                NodeKind::BoolToInt |
                NodeKind::RealBinary(_) |
                NodeKind::RealSqrt |
                NodeKind::FamilyPack { .. } |
                NodeKind::FamilyGetStatic { .. } |
                NodeKind::FamilyGetDynamic |
                NodeKind::Select { .. }
        )
    }
}
