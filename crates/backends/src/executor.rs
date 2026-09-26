use crate::{
    artifact::{ArtifactKey, ArtifactPayload, ArtifactStore},
    backend::{
        DynamicFusedBatchRequest, FusedBatchOutput, IndexRange as RuntimeIndexRange,
        MatrixMulAccumulateRequest, PolyMatrix, RuntimeValue, SampleRange as RuntimeSampleRange,
        TrapdoorValue, poly::CpuDcrtBackend,
    },
    gpu_execution_plan::{GpuExecutionSiteKey, GpuLoopSiteKey},
    host_control::{
        HostPrimitiveError, HostPrimitiveValue, clone_typed_runtime_input, dispatch_host_primitive,
        project_trapdoor_public,
    },
    session::{ArtifactHandle, SessionDescriptor, SessionStatus, SessionStore},
    transcript::{DrawSite, RecordedValue, SamplingMode, TranscriptError},
};
use mxx_ir_core::{
    ParamEnv, ValidatedGraph,
    artifact::{
        ArtifactAvailability, ArtifactType, ConcreteBoundedMatrixSchema, ManifestArtifact,
        ProductionId, SmallMatrixSemanticKind, SpecHash,
    },
    graph::{FrozenGraphScopeId, GraphScope},
    node::{HashVariant, LoopInputMode, MatrixBinaryOp, NodeKind},
    types::{
        ConcreteMatrixType, ConcreteWireType, InstantiationFrame, NodeId, Port, WireId, WireRef,
    },
};
use num_bigint::{BigInt, Sign};
use num_traits::{Signed, ToPrimitive, Zero};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroUsize,
    sync::Arc,
    time::{Duration, Instant},
};
use thiserror::Error;
use tracing::info;

mod cpu;
mod plan_cache;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExecutionConfig {
    /// Maximum number of sibling loop-body instances executed in one wave.
    ///
    /// This bounds each wave's intermediate working set and backend batch
    /// size. Artifact-compatible family outputs are streamed through the
    /// artifact store; scalar-only families are accumulated in memory.
    pub max_parallel_instances: NonZeroUsize,
    /// Optional progress reporting for actual preimage sampler invocations.
    pub preimage_progress: Option<PreimageProgressConfig>,
    /// Optionally fence backend release streams after this many executed nodes.
    /// This bounds queued releases without waiting unrelated live matrices.
    /// When set, also drain pending releases before returning. With `None`,
    /// releases remain asynchronous and are protected by backend lifetime events.
    pub release_fence_interval: Option<NonZeroUsize>,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_parallel_instances: NonZeroUsize::new(64).expect("64 is nonzero"),
            preimage_progress: None,
            release_fence_interval: None,
        }
    }
}

/// Reporting contract for a known number of preimage sampler invocations.
///
/// The executor increments this count only after its backend has returned the
/// sampled preimages. Replayed transcript values are deliberately excluded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PreimageProgressConfig {
    pub total: usize,
    pub report_interval: NonZeroUsize,
}

pub struct ExecutionResult {
    pub outputs: BTreeMap<String, RuntimeValue>,
    pub production_id: Option<ProductionId>,
    pub artifact_handles: BTreeMap<String, Vec<ArtifactHandle>>,
    staged_family_leases: Vec<StagedFamilyLease>,
}

/// A dispatch-only optimization: the original graph, validation and producer
/// execution remain intact. Aliases are installed before concat input release.
#[derive(Default)]
struct RootBlockAliases {
    arguments: Vec<Vec<WireRef>>,
    absent_run_ends: Vec<usize>,
    concats: BTreeMap<NodeId, Vec<(WireRef, WireRef)>>,
    slices: BTreeSet<NodeId>,
    row_block_concats: BTreeSet<NodeId>,
    adds: BTreeMap<NodeId, (NodeId, WireRef)>,
    decompositions: BTreeMap<NodeId, NodeId>,
    compact_products: BTreeMap<NodeId, (NodeId, WireRef, Vec<Vec<WireRef>>)>,
    row_sums: BTreeMap<NodeId, RootRowSumPlan>,
    /// Row-sum outputs sharing one tensor source. The first output is the
    /// single fused dispatch leader; the remaining outputs consume its
    /// multi-output result.
    tensor_row_sum_groups: BTreeMap<NodeId, Vec<NodeId>>,
    tensor_row_sum_leaders: BTreeMap<NodeId, NodeId>,
    row_sum_interiors: BTreeSet<NodeId>,
    row_sum_captures: BTreeMap<NodeId, Vec<NodeId>>,
    // A whole root composed only of resident inputs and one fused row sum.
    input_row_sum: Option<(NodeId, Vec<String>)>,
}

impl RootBlockAliases {
    fn absent_run_end(
        &self,
        position: usize,
        release_interval: Option<NonZeroUsize>,
        report_progress: bool,
    ) -> Option<usize> {
        if release_interval.is_some() || report_progress {
            return None;
        }
        self.absent_run_ends.get(position).copied().filter(|end| *end > position)
    }
}

/// Recognized runtime row sums, exposed for explicit calibration setup.
/// The original graph must already have passed validation.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct RootRowSumPlan {
    pub source: WireRef,
    pub tensor_operands: Option<[WireRef; 2]>,
    pub rows: Vec<Vec<usize>>,
    interiors: BTreeSet<NodeId>,
    /// Leader output for a shared tensor-row-sum dispatch, when present.
    pub tensor_group: Option<NodeId>,
}

#[doc(hidden)]
pub fn root_row_sum_plans(validated: &ValidatedGraph) -> BTreeMap<NodeId, RootRowSumPlan> {
    let scope = validated.source.scope(&FrozenGraphScopeId::Root).expect("validated root");
    let checked = validated.root_scope();
    let mut uses = BTreeMap::<WireRef, Vec<NodeId>>::new();
    for (index, node) in checked.execution_order.iter().enumerate() {
        for argument in scope.arguments(node).expect("validated arguments") {
            uses.entry(argument).or_default().push(NodeId(index as u64));
        }
    }
    let mut plans = BTreeMap::new();
    for (index, node) in checked.execution_order.iter().enumerate() {
        if !matches!(node.kind(), NodeKind::Concat { axis: mxx_ir_core::node::ConcatAxis::Rows }) {
            continue;
        }
        let output = WireRef { node: NodeId(index as u64), port: Port(0) };
        let Some(ConcreteWireType::Matrix(output_type)) = checked.wire_types.get(&output) else {
            continue;
        };
        let arguments = scope.arguments(node).expect("validated concat arguments");
        if arguments.is_empty() || output_type.rows != arguments.len() {
            continue;
        }
        let mut source = None;
        let mut groups = Vec::new();
        let mut interiors = BTreeSet::new();
        let mut valid = true;
        for argument in arguments {
            let mut pending = vec![argument];
            let mut rows = Vec::new();
            while let Some(wire) = pending.pop() {
                let Some(ConcreteWireType::Matrix(ty)) = checked.wire_types.get(&wire) else {
                    valid = false;
                    break;
                };
                if wire.port != Port(0) ||
                    uses.get(&wire).map(Vec::len) != Some(1) ||
                    checked.liveness.retained.contains(&wire) ||
                    scope.outputs().contains(&wire) ||
                    ty.rows != 1 ||
                    ty.columns != output_type.columns ||
                    ty.ring != output_type.ring
                {
                    valid = false;
                    break;
                }
                let producer = scope.node(wire.node).expect("validated row producer");
                let inputs = scope.arguments(producer).expect("validated row arguments");
                match producer.kind() {
                    NodeKind::MatrixBinary(MatrixBinaryOp::Add) if inputs.len() == 2 => {
                        pending.push(inputs[1]);
                        pending.push(inputs[0]);
                    }
                    NodeKind::Slice { rows: range, columns } if inputs.len() == 1 => {
                        let input = inputs[0];
                        let Some(ConcreteWireType::Matrix(input_type)) =
                            checked.wire_types.get(&input)
                        else {
                            valid = false;
                            break;
                        };
                        let evaluate = |range: &mxx_ir_core::node::IndexRange| {
                            Some((
                                range
                                    .start
                                    .evaluate_with_rings(
                                        &validated.bindings,
                                        crate::openfhe_guard::gen_modulus_and_warmup,
                                    )
                                    .ok()?
                                    .to_usize()?,
                                range
                                    .end
                                    .evaluate_with_rings(
                                        &validated.bindings,
                                        crate::openfhe_guard::gen_modulus_and_warmup,
                                    )
                                    .ok()?
                                    .to_usize()?,
                            ))
                        };
                        let row_range = range.as_ref().map_or(Some((0, input_type.rows)), evaluate);
                        let column_range =
                            columns.as_ref().map_or(Some((0, input_type.columns)), evaluate);
                        let Some((start, end)) = row_range else {
                            valid = false;
                            break;
                        };
                        if end.checked_sub(start) != Some(1) ||
                            end > input_type.rows ||
                            column_range != Some((0, input_type.columns)) ||
                            input_type.columns != output_type.columns ||
                            input_type.ring != output_type.ring ||
                            source.is_some_and(|source| source != input)
                        {
                            valid = false;
                            break;
                        }
                        source = Some(input);
                        rows.push(start);
                    }
                    _ => {
                        valid = false;
                        break;
                    }
                }
                interiors.insert(wire.node);
            }
            if !valid || rows.is_empty() {
                valid = false;
                break;
            }
            groups.push(rows);
        }
        if valid {
            plans.insert(
                output.node,
                RootRowSumPlan {
                    source: source.expect("nonempty row groups have a source"),
                    tensor_operands: None,
                    rows: groups,
                    interiors,
                    tensor_group: None,
                },
            );
        }
    }
    // Compose only when every original consumer remains represented. A
    // retained parent or a mixed consumer keeps the materialized boundary.
    loop {
        let mut replacement = None;
        for (parent_id, parent) in &plans {
            let wire = WireRef { node: *parent_id, port: Port(0) };
            if checked.liveness.retained.contains(&wire) || scope.outputs().contains(&wire) {
                continue;
            }
            let Some(consumers) = uses.get(&wire) else { continue };
            let parent_type = checked.wire_types[&wire].matrix_type().expect("row sum output");
            let mut children = plans
                .iter()
                .filter(|(_, plan)| plan.source == wire)
                .map(|(node, plan)| (*node, plan.clone()))
                .collect::<BTreeMap<_, _>>();
            let mut covered = true;
            for consumer in consumers {
                if children
                    .iter()
                    .any(|(node, plan)| node == consumer || plan.interiors.contains(consumer))
                {
                    continue;
                }
                let node = scope.node(*consumer).expect("validated row sum consumer");
                let NodeKind::Slice { rows, columns } = node.kind() else {
                    covered = false;
                    break;
                };
                let output = WireRef { node: *consumer, port: Port(0) };
                let Some(ConcreteWireType::Matrix(output_type)) = checked.wire_types.get(&output)
                else {
                    covered = false;
                    break;
                };
                let evaluate = |range: &mxx_ir_core::node::IndexRange| {
                    Some((
                        range
                            .start
                            .evaluate_with_rings(
                                &validated.bindings,
                                crate::openfhe_guard::gen_modulus_and_warmup,
                            )
                            .ok()?
                            .to_usize()?,
                        range
                            .end
                            .evaluate_with_rings(
                                &validated.bindings,
                                crate::openfhe_guard::gen_modulus_and_warmup,
                            )
                            .ok()?
                            .to_usize()?,
                    ))
                };
                let selected = rows.as_ref().map_or(Some((0, parent_type.rows)), evaluate);
                let columns = columns.as_ref().map_or(Some((0, parent_type.columns)), evaluate);
                let Some((start, end)) = selected else {
                    covered = false;
                    break;
                };
                if end.checked_sub(start) != Some(1) ||
                    end > parent.rows.len() ||
                    columns != Some((0, parent_type.columns)) ||
                    output_type.rows != 1 ||
                    output_type.columns != parent_type.columns ||
                    output_type.ring != parent_type.ring
                {
                    covered = false;
                    break;
                }
                children.insert(
                    *consumer,
                    RootRowSumPlan {
                        source: wire,
                        tensor_operands: None,
                        rows: vec![vec![start]],
                        interiors: BTreeSet::new(),
                        tensor_group: None,
                    },
                );
            }
            if !covered || children.is_empty() {
                continue;
            }
            // A compact DAG can repeatedly sum two copies of a prior row.
            // Keep that materialized boundary instead of expanding its terms
            // exponentially. This matches the native row-sum term capacity.
            let expanded_terms = children.values().try_fold(0usize, |total, child| {
                child
                    .rows
                    .iter()
                    .flatten()
                    .try_fold(total, |total, row| total.checked_add(parent.rows[*row].len()))
            });
            if expanded_terms.is_none_or(|terms| terms > 32) {
                continue;
            }
            for child in children.values_mut() {
                child.rows = child
                    .rows
                    .iter()
                    .map(|group| {
                        group.iter().flat_map(|row| parent.rows[*row].iter().copied()).collect()
                    })
                    .collect();
                child.source = parent.source;
                child.interiors.extend(parent.interiors.iter().copied());
                child.interiors.insert(*parent_id);
            }
            replacement = Some((*parent_id, children));
            break;
        }
        let Some((parent, children)) = replacement else { break };
        plans.remove(&parent);
        plans.extend(children);
    }
    // A tensor may feed several row-sum outputs in a composite graph. Fuse
    // those outputs as one multi-output TensorRowSum only when the union of
    // their interiors covers every source consumer. This preserves the
    // source's single production launch while retaining the materialized
    // fallback whenever an observable/external consumer exists.
    let mut groups = BTreeMap::<WireRef, Vec<NodeId>>::new();
    for (node, plan) in &plans {
        groups.entry(plan.source).or_default().push(*node);
    }
    for (source, mut outputs) in groups {
        if checked.liveness.retained.contains(&source) ||
            scope.outputs().contains(&source) ||
            !matches!(
                scope.node(source.node).expect("validated source").kind(),
                NodeKind::Tensor
            )
        {
            continue;
        }
        outputs.sort_unstable();
        let mut union_interiors = BTreeSet::new();
        for output in &outputs {
            union_interiors.extend(plans[output].interiors.iter().copied());
        }
        if !uses
            .get(&source)
            .is_some_and(|users| users.iter().all(|user| union_interiors.contains(user)))
        {
            continue;
        }
        let arguments = scope.arguments(scope.node(source.node).unwrap()).unwrap();
        if arguments.len() != 2 ||
            arguments.iter().any(|wire| {
                !matches!(checked.wire_types.get(wire), Some(ConcreteWireType::Matrix(_)))
            })
        {
            continue;
        }
        let leader = outputs[0];
        for output in &outputs {
            let plan = plans.get_mut(output).expect("row sum group output");
            plan.tensor_operands = Some([arguments[0], arguments[1]]);
            plan.tensor_group = Some(leader);
            plan.interiors.insert(source.node);
        }
    }
    plans
}

fn root_block_aliases(
    validated: &ValidatedGraph,
    scope_id: &FrozenGraphScopeId,
    capture_trace: bool,
) -> Arc<RootBlockAliases> {
    if capture_trace || scope_id != &FrozenGraphScopeId::Root {
        return Arc::new(RootBlockAliases::default());
    }
    plan_cache::get(validated, || build_root_block_aliases(validated))
}

fn build_root_block_aliases(validated: &ValidatedGraph) -> RootBlockAliases {
    let mut plan = RootBlockAliases::default();
    let scope = validated.source.scope(&FrozenGraphScopeId::Root).expect("validated root");
    let checked = validated.root_scope();
    plan.arguments = checked
        .execution_order
        .iter()
        .map(|node| scope.arguments(node).expect("validated arguments"))
        .collect();
    plan.row_sums = root_row_sum_plans(validated);
    for (output, row_sum) in &plan.row_sums {
        if let Some(leader) = row_sum.tensor_group {
            plan.tensor_row_sum_groups.entry(leader).or_default().push(*output);
            plan.tensor_row_sum_leaders.insert(*output, leader);
        }
    }
    for outputs in plan.tensor_row_sum_groups.values_mut() {
        outputs.sort_unstable();
    }
    let mut reserved = BTreeSet::new();
    for (output, row_sum) in &plan.row_sums {
        reserved.insert(*output);
        reserved.insert(row_sum.source.node);
        reserved.extend(row_sum.interiors.iter().copied());
        plan.row_sum_interiors.extend(row_sum.interiors.iter().copied());
        let first = *row_sum.interiors.first().expect("nonempty row sum interiors");
        plan.row_sum_captures.entry(first).or_default().push(*output);
    }
    let mut users = BTreeMap::<WireRef, Vec<usize>>::new();
    for (index, node) in checked.execution_order.iter().enumerate() {
        for wire in scope.arguments(node).expect("validated arguments") {
            users.entry(wire).or_default().push(index);
        }
    }
    for (index, node) in checked.execution_order.iter().enumerate() {
        if reserved.contains(&NodeId(index as u64)) {
            continue;
        }
        if !matches!(node.kind(), NodeKind::Concat { axis: mxx_ir_core::node::ConcatAxis::Rows }) {
            continue;
        }
        let wire = WireRef { node: NodeId(index as u64), port: Port(0) };
        if checked.liveness.retained.contains(&wire) || scope.outputs().contains(&wire) {
            continue;
        }
        let Some(consumers) = users.get(&wire) else { continue };
        let Some(ConcreteWireType::Matrix(concat_type)) = checked.wire_types.get(&wire) else {
            continue;
        };
        let args = scope.arguments(node).expect("validated concat arguments");
        let mut blocks = Vec::with_capacity(args.len());
        let mut start = 0usize;
        for arg in args {
            let Some(ConcreteWireType::Matrix(ty)) = checked.wire_types.get(&arg) else {
                break;
            };
            let Some(end) = start.checked_add(ty.rows) else { break };
            blocks.push((start, end, arg));
            start = end;
        }
        if blocks.len() != node.arguments().len() {
            continue;
        }
        if consumers.len() == 1 {
            let consumer = consumers[0];
            let add = &checked.execution_order[consumer];
            let args = scope.arguments(add).expect("validated add arguments");
            let output = WireRef { node: NodeId(consumer as u64), port: Port(0) };
            if blocks.len() <= 32 &&
                matches!(add.kind(), NodeKind::GadgetDecompose { .. }) &&
                args == [wire]
            {
                plan.row_block_concats.insert(wire.node);
                plan.decompositions.insert(output.node, wire.node);
                continue;
            }
            if blocks.len() <= 32 &&
                matches!(add.kind(), NodeKind::MatrixMulSmallRhs) &&
                args.len() == 2 &&
                args[0] == wire &&
                !checked.liveness.retained.contains(&output) &&
                !scope.outputs().contains(&output)
            {
                let Some(ConcreteWireType::Matrix(product_type)) = checked.wire_types.get(&output)
                else {
                    continue;
                };
                let mut aliases = vec![Vec::new(); blocks.len()];
                let mut covered = true;
                for user in users.get(&output).into_iter().flatten() {
                    let slice = &checked.execution_order[*user];
                    let NodeKind::Slice { rows, columns } = slice.kind() else {
                        covered = false;
                        break;
                    };
                    let range = |range: &mxx_ir_core::node::IndexRange| {
                        Some((
                            range
                                .start
                                .evaluate_with_rings(
                                    &validated.bindings,
                                    crate::openfhe_guard::gen_modulus_and_warmup,
                                )
                                .ok()?
                                .to_usize()?,
                            range
                                .end
                                .evaluate_with_rings(
                                    &validated.bindings,
                                    crate::openfhe_guard::gen_modulus_and_warmup,
                                )
                                .ok()?
                                .to_usize()?,
                        ))
                    };
                    let rows = rows.as_ref().map_or(Some((0, product_type.rows)), range);
                    let columns = columns.as_ref().map_or(Some((0, product_type.columns)), range);
                    let Some(block) =
                        blocks.iter().position(|(start, end, _)| rows == Some((*start, *end)))
                    else {
                        covered = false;
                        break;
                    };
                    if columns != Some((0, product_type.columns)) ||
                        reserved.contains(&NodeId(*user as u64))
                    {
                        covered = false;
                        break;
                    }
                    aliases[block].push(WireRef { node: NodeId(*user as u64), port: Port(0) });
                }
                if covered && aliases.iter().all(|aliases| !aliases.is_empty()) {
                    plan.row_block_concats.insert(wire.node);
                    plan.slices.extend(aliases.iter().flatten().map(|wire| wire.node));
                    plan.compact_products.insert(output.node, (wire.node, args[1], aliases));
                    continue;
                }
            }
            if matches!(add.kind(), NodeKind::MatrixBinary(MatrixBinaryOp::Add)) &&
                args.len() == 2 &&
                args[0] == wire &&
                args[1] != wire &&
                checked.wire_types.get(&args[1]) == checked.wire_types.get(&wire) &&
                checked.wire_types.get(&output) == checked.wire_types.get(&wire)
            {
                plan.row_block_concats.insert(wire.node);
                plan.adds.insert(output.node, (wire.node, args[1]));
                continue;
            }
        }
        let mut aliases = Vec::with_capacity(consumers.len());
        for &consumer in consumers {
            let slice = &checked.execution_order[consumer];
            let NodeKind::Slice { rows, columns } = slice.kind() else { break };
            let range = |range: &mxx_ir_core::node::IndexRange| {
                Some((
                    range
                        .start
                        .evaluate_with_rings(
                            &validated.bindings,
                            crate::openfhe_guard::gen_modulus_and_warmup,
                        )
                        .ok()?
                        .to_usize()?,
                    range
                        .end
                        .evaluate_with_rings(
                            &validated.bindings,
                            crate::openfhe_guard::gen_modulus_and_warmup,
                        )
                        .ok()?
                        .to_usize()?,
                ))
            };
            let row_range = match rows {
                Some(rows) => range(rows),
                None => Some((0, concat_type.rows)),
            };
            let col_range = match columns {
                Some(columns) => range(columns),
                None => Some((0, concat_type.columns)),
            };
            if col_range != Some((0, concat_type.columns)) {
                break;
            }
            let Some((_, _, source)) =
                blocks.iter().find(|(start, end, _)| row_range == Some((*start, *end)))
            else {
                break;
            };
            let output = WireRef { node: NodeId(consumer as u64), port: Port(0) };
            if checked.wire_types.get(&output) != checked.wire_types.get(source) {
                break;
            }
            aliases.push((output, *source));
        }
        if aliases.len() == consumers.len() {
            plan.slices.extend(aliases.iter().map(|(output, _)| output.node));
            plan.concats.insert(wire.node, aliases);
        }
    }
    // Omit only bookkeeping for wires which no dispatch action can insert.
    // In particular, do not move captures or last-use releases of live sources.
    let absent = plan
        .row_sum_interiors
        .iter()
        .filter_map(|id| {
            let wire = WireRef { node: *id, port: Port(0) };
            (!plan.row_sums.contains_key(id) &&
                !plan.adds.contains_key(id) &&
                !plan.slices.contains(id) &&
                !checked.liveness.retained.contains(&wire) &&
                !scope.outputs().contains(&wire) &&
                checked.execution_order[id.0 as usize].output_types().len() == 1)
                .then_some(wire)
        })
        .collect::<BTreeSet<_>>();
    plan.absent_run_ends = (0..checked.execution_order.len()).collect();
    let mut run_end = checked.execution_order.len();
    for position in (0..checked.execution_order.len()).rev() {
        let id = NodeId(position as u64);
        let wire = WireRef { node: id, port: Port(0) };
        let empty = absent.contains(&wire) &&
            !plan.row_sum_captures.contains_key(&id) &&
            !plan.concats.contains_key(&id) &&
            !plan.row_block_concats.contains(&id) &&
            plan.arguments[position].iter().all(|argument| absent.contains(argument));
        if empty {
            plan.absent_run_ends[position] = run_end;
        } else {
            run_end = position;
        }
    }
    if let [output] = scope.outputs() {
        if let Some(row_sum) = plan.row_sums.get(&output.node) {
            let operands = row_sum
                .tensor_operands
                .as_ref()
                .map_or(std::slice::from_ref(&row_sum.source), |operands| operands.as_slice());
            let names = operands
                .iter()
                .map(|wire| match scope.node(wire.node)?.kind() {
                    NodeKind::Input { name, artifact: None, .. } => Some(name.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>();
            // No other producer, artifact load, or observable intermediate may
            // disappear when dispatching the already-fused operation directly.
            if names.is_some() &&
                checked.execution_order.iter().enumerate().all(|(index, _)| {
                    let id = NodeId(index as u64);
                    id == output.node ||
                        row_sum.interiors.contains(&id) ||
                        operands.iter().any(|wire| wire.node == id)
                })
            {
                plan.input_row_sum = names.map(|names| (output.node, names));
            }
        }
    }
    plan
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StagedFamilyLease {
    pub production: ProductionId,
    pub name: String,
    pub descriptor: ManifestArtifact,
}

impl ExecutionResult {
    /// Materializes a named output that the executor returned as a lazy or streamed artifact.
    ///
    /// Parallel-loop families may be streamed through the artifact store so their live backend
    /// memory stays bounded by the configured wave size. Callers that need the complete output
    /// value can explicitly load it after execution. The staged payloads remain owned by this
    /// result and are deleted by [`Self::cleanup_staged`].
    pub fn materialize_output<S: ArtifactStore>(
        &mut self,
        name: &str,
        backend: &CpuDcrtBackend,
        store: &mut S,
    ) -> Result<&RuntimeValue, ExecutionError> {
        let value = self
            .outputs
            .get_mut(name)
            .ok_or_else(|| ExecutionError::MissingOutput(name.to_owned()))?;
        // Resident matrices already are the materialized output. In particular,
        // keep their owners in place instead of cloning and reinserting them.
        if !matches!(value, RuntimeValue::Matrix(_)) {
            *value = materialize_runtime_value(value.clone(), backend, store)?;
        }
        Ok(value)
    }

    /// Deletes ephemeral streamed families returned by this execution.
    ///
    /// A returned staged family remains readable until this method is called.
    /// Persisted families are replaced with final lazy artifact handles and do
    /// not require this cleanup.
    pub fn cleanup_staged<S: ArtifactStore>(
        &mut self,
        store: &mut S,
    ) -> Result<(), ExecutionError> {
        for lease in &self.staged_family_leases {
            let count = lease.descriptor.family_count.ok_or_else(|| {
                ExecutionError::Manifest("staged family lease has no cardinality".to_owned())
            })?;
            for index in 0..count {
                store
                    .remove_staged(&ArtifactKey {
                        production: lease.production.clone(),
                        name: lease.name.clone(),
                        index: Some(index),
                    })
                    .map_err(|error| ExecutionError::Artifact(error.to_string()))?;
            }
        }
        self.staged_family_leases.clear();
        Ok(())
    }
}

pub type ExecutionTrace = BTreeMap<WireId, RuntimeValue>;

type TrapdoorParts = (
    Option<Arc<crate::sampler::trapdoor::DCRTTrapdoor>>,
    Arc<crate::matrix::dcrt_poly::DCRTPolyMatrix>,
    ConcreteMatrixType,
    f64,
    BigInt,
    usize,
    Option<bool>,
);

struct InstanceResult {
    outputs: Vec<RuntimeValue>,
}

struct ExecutableNode<'a> {
    id: NodeId,
    kind: &'a NodeKind,
    args: &'a [WireRef],
}

#[derive(Debug, Error)]
pub enum ExecutionError {
    #[error("backend operation failed: {0}")]
    Backend(String),
    #[error("preimage progress expected {expected} generated preimages but observed {actual}")]
    PreimageProgressMismatch { expected: usize, actual: usize },
    #[error("artifact operation failed: {0}")]
    Artifact(String),
    #[error(transparent)]
    Transcript(#[from] TranscriptError),
    #[error("input {0} was not provided")]
    MissingInput(String),
    #[error("output {0} does not exist")]
    MissingOutput(String),
    #[error("wire {0:?} is unavailable")]
    MissingWire(WireRef),
    #[error("wire {0:?} has the wrong runtime value kind")]
    ValueKind(WireRef),
    #[error("integer division by zero at node {0:?}")]
    DivisionByZero(NodeId),
    #[error("invalid real operation at node {0:?}")]
    InvalidRealOperation(NodeId),
    #[error("select index {index} is outside [0, {count}) at node {node:?}")]
    SelectIndexOutOfRange { node: NodeId, index: BigInt, count: usize },
    #[error("subgraph {name} does not exist at node {node:?}")]
    MissingSubgraph { node: NodeId, name: String },
    #[error("validated wire metadata is missing for {0:?}")]
    MissingMetadata(WireId),
    #[error("runtime expression failed at node {node:?}: {message}")]
    Expression { node: NodeId, message: String },
    #[error("backend placement {placement} is outside [0, {count})")]
    BackendPlacement { placement: usize, count: usize },
    #[error("fixed GPU plan is missing execution site {site:?}")]
    MissingSitePlan { site: GpuExecutionSiteKey },
    #[error("fixed GPU plan is missing loop site {site:?}")]
    MissingLoopPlan { site: GpuLoopSiteKey },
    #[error("fixed GPU plan cannot execute while value trace capture is enabled")]
    GpuPlanTraceUnsupported,
    #[error("fixed GPU plan operation identity mismatch at site {site:?}")]
    GpuPlanOperationMismatch { site: GpuExecutionSiteKey },
    #[error("fixed GPU plan does not support operation at site {site:?}")]
    UnsupportedGpuOperation { site: GpuExecutionSiteKey },
    #[error("invalid frozen GPU plan: {0}")]
    InvalidGpuPlan(String),
    #[error("execution plan/backend mismatch: frozen_gpu={plan_gpu}, backend_gpu={backend_gpu}")]
    GpuPlanBackendMismatch { plan_gpu: bool, backend_gpu: bool },
    #[error("backend returned an invalid parallel batch length at node {0:?}")]
    InvalidBatch(NodeId),
    #[error("preimage public matrix does not match the trapdoor public matrix at node {0:?}")]
    PreimagePublicMismatch(NodeId),
    #[error("manifest operation failed: {0}")]
    Manifest(String),
    #[error("scratch cleanup failed: {message}")]
    StagedCleanup { message: String, leases: Vec<StagedFamilyLease> },
}

/// Executes trusted inputs prepared for the exact validated graph and parameter bindings.
/// All declared non-artifact inputs must be supplied, and every value must satisfy the
/// full metadata and payload contract documented on [`RuntimeValue`].
pub fn execute<S>(
    validated: &ValidatedGraph,
    backend: &mut CpuDcrtBackend,
    inputs: BTreeMap<String, RuntimeValue>,
    artifact_store: &mut S,
    sampling_mode: SamplingMode<'_>,
    config: ExecutionConfig,
) -> Result<ExecutionResult, ExecutionError>
where
    S: SessionStore,
{
    let preflight = execution_preflight(validated, &inputs)?;
    execute_internal(
        validated,
        backend,
        inputs,
        artifact_store,
        sampling_mode,
        false,
        None,
        false,
        config,
        preflight,
    )
    .map(|(result, _)| result)
}

/// Starts or resumes a session over a complete, immutable input map satisfying [`execute`]'s
/// contract. An invalid invocation may leave a durable descriptor; correcting inputs requires
/// a new nonce (and a new stable alias if used), not a resume with the original identity.
pub fn execute_in_session<S>(
    validated: &ValidatedGraph,
    backend: &mut CpuDcrtBackend,
    inputs: BTreeMap<String, RuntimeValue>,
    artifact_store: &mut S,
    execution_nonce: [u8; 32],
    config: ExecutionConfig,
) -> Result<ExecutionResult, ExecutionError>
where
    S: SessionStore,
{
    let preflight = execution_preflight(validated, &inputs)?;
    let input_digest = runtime_inputs_digest(validated, backend, &inputs)?;
    let spec_hash = preflight.spec_hash.clone();
    let production = mxx_ir_core::artifact::production_id(spec_hash, execution_nonce);
    let descriptor = SessionDescriptor::new(
        production.clone(),
        validated.source.name().to_owned(),
        input_digest,
    );
    let session_status = artifact_store
        .open_session(&descriptor)
        .map_err(|error| ExecutionError::Artifact(error.to_string()))?;
    match execute_internal(
        validated,
        backend,
        inputs,
        artifact_store,
        SamplingMode::Fresh,
        false,
        Some(production.clone()),
        session_status == SessionStatus::Finalized,
        config,
        preflight,
    ) {
        Ok((result, _)) => Ok(result),
        Err(error) => {
            artifact_store.release_session(&production).map_err(|release_error| {
                ExecutionError::Artifact(format!(
                    "execution failed: {error}; session release failed: {release_error}"
                ))
            })?;
            Err(error)
        }
    }
}

/// Executes a prepared graph through the transaction semantics declared by
/// its validated artifact outputs. Producer graphs open a durable session;
/// transient graphs (including artifact consumers) remain non-session
/// executions so GPU inputs are never serialized merely to obtain a nonce.
pub fn execute_prepared<S>(
    validated: &ValidatedGraph,
    backend: &mut CpuDcrtBackend,
    inputs: BTreeMap<String, RuntimeValue>,
    artifact_store: &mut S,
    execution_nonce: [u8; 32],
    config: ExecutionConfig,
) -> Result<ExecutionResult, ExecutionError>
where
    S: SessionStore,
{
    let produces_artifacts =
        validated.source.outputs().values().any(|output| output.availability.is_some());
    if produces_artifacts {
        execute_in_session(validated, backend, inputs, artifact_store, execution_nonce, config)
    } else {
        execute(validated, backend, inputs, artifact_store, SamplingMode::Fresh, config)
    }
}

/// Executes a graph while retaining every intermediate wire value. This is
/// intended for diagnostics and optional analysis; ordinary execution keeps
/// using the liveness drop schedule without retaining a trace.
pub fn execute_with_trace<S>(
    validated: &ValidatedGraph,
    backend: &mut CpuDcrtBackend,
    inputs: BTreeMap<String, RuntimeValue>,
    artifact_store: &mut S,
    sampling_mode: SamplingMode<'_>,
    config: ExecutionConfig,
) -> Result<(ExecutionResult, ExecutionTrace), ExecutionError>
where
    S: SessionStore,
{
    let preflight = execution_preflight(validated, &inputs)?;
    execute_internal(
        validated,
        backend,
        inputs,
        artifact_store,
        sampling_mode,
        true,
        None,
        false,
        config,
        preflight,
    )
}

struct ExecutionPreflight {
    spec_hash: SpecHash,
}

/// Reject an invocation before opening durable session state.
fn execution_preflight(
    validated: &ValidatedGraph,
    inputs: &BTreeMap<String, RuntimeValue>,
) -> Result<ExecutionPreflight, ExecutionError> {
    let root = validated.source.root_scope();
    let mut declared = BTreeMap::new();
    for (index, node) in root.nodes().iter().enumerate() {
        let NodeKind::Input { name, artifact: None, .. } = node.kind() else {
            continue;
        };
        let wire = WireRef { node: NodeId(index as u64), port: Port(0) };
        let ty = validated.root_scope().wire_types.get(&wire).ok_or_else(|| {
            ExecutionError::MissingMetadata(WireId { instantiation_path: Vec::new(), wire })
        })?;
        declared.insert(name.as_str(), ty);
    }
    for (name, ty) in &declared {
        let value =
            inputs.get(*name).ok_or_else(|| ExecutionError::MissingInput((*name).to_owned()))?;
        if !value.matches_wire_type(ty) {
            return Err(ExecutionError::Manifest(format!(
                "runtime input {name} does not match its validated wire type"
            )));
        }
    }
    for name in inputs.keys() {
        if !declared.contains_key(name.as_str()) {
            return Err(ExecutionError::Manifest(format!(
                "runtime input {name} is not declared by the graph"
            )));
        }
    }
    let spec_hash = mxx_ir_core::encoding::spec_hash(&validated.source, &validated.bindings)
        .map_err(|error| ExecutionError::Manifest(error.to_string()))?;
    Ok(ExecutionPreflight { spec_hash })
}

fn execute_internal<S>(
    validated: &ValidatedGraph,
    backend: &mut CpuDcrtBackend,
    inputs: BTreeMap<String, RuntimeValue>,
    artifact_store: &mut S,
    sampling_mode: SamplingMode<'_>,
    capture_trace: bool,
    session: Option<ProductionId>,
    finalized_session_replay: bool,
    config: ExecutionConfig,
    preflight: ExecutionPreflight,
) -> Result<(ExecutionResult, ExecutionTrace), ExecutionError>
where
    S: SessionStore,
{
    let ExecutionPreflight { spec_hash } = preflight;
    let release_fence_requested = config.release_fence_interval.is_some();
    let preimage_progress = config.preimage_progress.map(PreimageProgress::new);
    let production = session
        .clone()
        .unwrap_or_else(|| mxx_ir_core::artifact::production_id(spec_hash, rand::random()));
    let mut executor = Executor {
        validated,
        backend,
        artifact_store,
        sampling_mode,
        trace: capture_trace.then(BTreeMap::new),
        session,
        finalized_session_replay,
        config,
        production,
        scratch_production: None,
        staged_families: BTreeMap::new(),
        preimage_progress,
        executed_node_count: 0,
        last_release_fence_node_count: 0,
        has_pending_releases: false,
        execution_started: Instant::now(),
    };
    let mut instance = match executor.execute_instance(
        &FrozenGraphScopeId::Root,
        &validated.bindings,
        Vec::new(),
        inputs,
        0,
    ) {
        Ok(instance) => instance,
        Err(error) => {
            return match executor.cleanup_all_staged_families() {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(cleanup_error),
            };
        }
    };
    if let Err(error) = executor.finish_preimage_progress() {
        return match executor.cleanup_all_staged_families() {
            Ok(()) => Err(error),
            Err(cleanup_error) => Err(cleanup_error),
        };
    }
    let mut named_outputs = validated
        .source
        .outputs()
        .keys()
        .cloned()
        .zip(instance.outputs.drain(..))
        .collect::<BTreeMap<_, _>>();
    let (production_id, artifact_handles) = match executor.persist_outputs(&mut named_outputs) {
        Ok(persisted) => persisted,
        Err(error) => {
            return match executor.cleanup_all_staged_families() {
                Ok(()) => Err(error),
                Err(cleanup_error) => Err(cleanup_error),
            };
        }
    };
    let staged_family_leases = executor.cleanup_unreturned_staged_families(&named_outputs)?;
    if let Some(production) = &executor.session {
        if let Err(error) = executor.artifact_store.release_session(production) {
            return Err(ExecutionError::StagedCleanup {
                message: format!("session release failed: {error}"),
                leases: staged_family_leases,
            });
        }
    }
    if release_fence_requested {
        executor.fence_pending_releases()?;
    }
    info!(
        graph = validated.source.name(),
        host_returned_node_instances = executor.executed_node_count,
        elapsed_seconds = executor.execution_started.elapsed().as_secs_f64(),
        execution_backend = "cpu",
        "runtime graph execution returned after output finalization"
    );
    let result = ExecutionResult {
        outputs: named_outputs,
        production_id,
        artifact_handles,
        staged_family_leases,
    };
    Ok((result, executor.trace.take().unwrap_or_default()))
}

struct Executor<'a, S: SessionStore> {
    validated: &'a ValidatedGraph,
    backend: &'a mut CpuDcrtBackend,
    artifact_store: &'a mut S,
    sampling_mode: SamplingMode<'a>,
    trace: Option<ExecutionTrace>,
    session: Option<ProductionId>,
    finalized_session_replay: bool,
    config: ExecutionConfig,
    production: ProductionId,
    scratch_production: Option<ProductionId>,
    staged_families: BTreeMap<(ProductionId, String), ManifestArtifact>,
    /// A graph may expose the same persisted artifact under multiple input
    /// wires.  Keep one canonical payload read per immutable key for the
    /// duration of an execution, while still checking a later descriptor
    /// against the descriptor that was validated on the first read.
    preimage_progress: Option<PreimageProgress>,
    executed_node_count: usize,
    last_release_fence_node_count: usize,
    has_pending_releases: bool,
    execution_started: Instant,
}

struct PreimageProgress {
    config: PreimageProgressConfig,
    completed: usize,
    last_reported: usize,
    started: Instant,
}

impl PreimageProgress {
    fn new(config: PreimageProgressConfig) -> Self {
        info!(
            total = config.total,
            report_interval = config.report_interval.get(),
            "preimage generation progress started"
        );
        Self { config, completed: 0, last_reported: 0, started: Instant::now() }
    }

    fn record(&mut self, count: usize) {
        self.completed = self.completed.saturating_add(count);
        let final_report = self.completed >= self.config.total;
        if !final_report &&
            self.completed.saturating_sub(self.last_reported) < self.config.report_interval.get()
        {
            return;
        }
        let elapsed = self.started.elapsed();
        let elapsed_seconds = elapsed.as_secs_f64();
        let rate_per_second = (self.completed as f64) / elapsed_seconds.max(f64::MIN_POSITIVE);
        let remaining = self.config.total.saturating_sub(self.completed);
        let eta = Duration::from_secs_f64((remaining as f64) / rate_per_second);
        info!(
            completed = self.completed,
            total = self.config.total,
            percent = (self.completed as f64) * 100.0 / (self.config.total.max(1) as f64),
            rate_per_second,
            elapsed = ?elapsed,
            eta = ?eta,
            "preimage generation progress"
        );
        self.last_reported = self.completed;
    }

    fn finish(&self) -> Result<(), ExecutionError> {
        if self.completed != self.config.total {
            return Err(ExecutionError::PreimageProgressMismatch {
                expected: self.config.total,
                actual: self.completed,
            });
        }
        info!(
            completed = self.completed,
            total = self.config.total,
            elapsed = ?self.started.elapsed(),
            "preimage generation completed"
        );
        Ok(())
    }
}

impl<S: SessionStore> Executor<'_, S> {
    fn persist_outputs(
        &mut self,
        outputs: &mut BTreeMap<String, RuntimeValue>,
    ) -> Result<(Option<ProductionId>, BTreeMap<String, Vec<ArtifactHandle>>), ExecutionError> {
        if self.finalized_session_replay {
            return self.replay_finalized_outputs(outputs);
        }
        let production = self.production.clone();
        let mut artifacts = BTreeMap::new();
        let mut handles = BTreeMap::<String, Vec<ArtifactHandle>>::new();
        let mut staged_replacements = Vec::new();
        for (name, output_root) in self.validated.source.outputs() {
            let Some(availability) = output_root.availability else {
                continue;
            };
            let Some(output) = outputs.get(name) else {
                continue;
            };
            let wire = WireId { instantiation_path: Vec::new(), wire: output_root.value };
            let concrete_type = self
                .validated
                .root_scope()
                .wire_types
                .get(&output_root.value)
                .ok_or_else(|| ExecutionError::MissingMetadata(wire.clone()))?;
            let (element_type, family_count) = match concrete_type {
                ConcreteWireType::IndexedFamily { element, count } => {
                    (element.as_ref(), Some(*count))
                }
                scalar => (scalar, None),
            };
            let artifact_type = ArtifactType::from_wire_type(element_type).ok_or_else(|| {
                ExecutionError::Manifest(format!("output {name} is not artifact-compatible"))
            })?;
            if let RuntimeValue::StagedArtifactFamily {
                production: staged_production,
                name: staged_name,
                descriptor,
            } = output
            {
                let Some(count) = family_count else {
                    return Err(ExecutionError::Manifest(format!(
                        "output {name} is staged as a family but validated as a scalar"
                    )));
                };
                if descriptor.artifact_type != artifact_type ||
                    descriptor.family_count != Some(count)
                {
                    return Err(ExecutionError::Manifest(format!(
                        "output {name} staged descriptor does not match validated metadata"
                    )));
                }
                for index in 0..count {
                    let staged_key = ArtifactKey {
                        production: staged_production.clone(),
                        name: staged_name.clone(),
                        index: Some(index),
                    };
                    let payload = self
                        .artifact_store
                        .load_staged(&staged_key, descriptor)
                        .map_err(Self::artifact_error)?;
                    let handle = ArtifactHandle {
                        key: ArtifactKey {
                            production: production.clone(),
                            name: name.clone(),
                            index: Some(index),
                        },
                        artifact_type: artifact_type.clone(),
                        availability,
                        layout: None,
                    };
                    self.artifact_store
                        .store(
                            handle.key.clone(),
                            &artifact_type,
                            availability,
                            handle.layout.as_deref(),
                            payload,
                        )
                        .map_err(Self::artifact_error)?;
                    if self.session.is_some() {
                        self.artifact_store
                            .commit_artifact(&handle)
                            .map_err(Self::artifact_error)?;
                    }
                    handles.entry(name.clone()).or_default().push(handle);
                }
                artifacts.insert(
                    name.clone(),
                    mxx_ir_core::artifact::ExportArtifact {
                        wire,
                        artifact_type,
                        family_count,
                        availability,
                        layout: None,
                    },
                );
                staged_replacements.push((
                    name.clone(),
                    staged_production.clone(),
                    staged_name.clone(),
                    count,
                ));
                continue;
            }
            if let RuntimeValue::IndexedFamily { values: members, .. } = output {
                if family_count != Some(members.len()) {
                    return Err(ExecutionError::Manifest(format!(
                        "output {name} family count does not match validated metadata"
                    )));
                }
                for (index, member) in members.iter().enumerate() {
                    let payload = self.encode_artifact(member, &artifact_type)?;
                    let handle = ArtifactHandle {
                        key: ArtifactKey {
                            production: production.clone(),
                            name: name.clone(),
                            index: Some(index),
                        },
                        artifact_type: artifact_type.clone(),
                        availability,
                        layout: None,
                    };
                    self.artifact_store
                        .store(
                            handle.key.clone(),
                            &artifact_type,
                            availability,
                            handle.layout.as_deref(),
                            payload,
                        )
                        .map_err(Self::artifact_error)?;
                    if self.session.is_some() {
                        self.artifact_store
                            .commit_artifact(&handle)
                            .map_err(Self::artifact_error)?;
                    }
                    handles.entry(name.clone()).or_default().push(handle);
                }
                artifacts.insert(
                    name.clone(),
                    mxx_ir_core::artifact::ExportArtifact {
                        wire,
                        artifact_type,
                        family_count,
                        availability,
                        layout: None,
                    },
                );
                continue;
            }
            let payload = self.encode_artifact(output, &artifact_type)?;
            let handle = ArtifactHandle {
                key: ArtifactKey {
                    production: production.clone(),
                    name: name.clone(),
                    index: None,
                },
                artifact_type: artifact_type.clone(),
                availability,
                layout: None,
            };
            self.artifact_store
                .store(
                    handle.key.clone(),
                    &artifact_type,
                    availability,
                    handle.layout.as_deref(),
                    payload,
                )
                .map_err(Self::artifact_error)?;
            if self.session.is_some() {
                self.artifact_store.commit_artifact(&handle).map_err(Self::artifact_error)?;
            }
            handles.entry(name.clone()).or_default().push(handle);
            artifacts.insert(
                name.clone(),
                mxx_ir_core::artifact::ExportArtifact {
                    wire,
                    artifact_type,
                    family_count: None,
                    availability,
                    layout: None,
                },
            );
        }
        if artifacts.is_empty() {
            if self.session.is_none() {
                return Ok((None, handles));
            }
        }
        let manifest = mxx_ir_core::artifact::export_manifest(production.clone(), &artifacts);
        let replacement_descriptors = staged_replacements
            .iter()
            .map(|(name, _, _, _)| {
                let descriptor = manifest
                    .artifacts
                    .get(name)
                    .cloned()
                    .expect("staged output was inserted into the manifest");
                (name.clone(), descriptor)
            })
            .collect::<BTreeMap<_, _>>();
        if self.session.is_some() {
            self.artifact_store.finalize_session(manifest).map_err(Self::artifact_error)?;
        } else {
            self.artifact_store.store_manifest(manifest).map_err(Self::artifact_error)?;
        }
        for (name, staged_production, staged_name, count) in staged_replacements {
            for index in 0..count {
                self.artifact_store
                    .remove_staged(&ArtifactKey {
                        production: staged_production.clone(),
                        name: staged_name.clone(),
                        index: Some(index),
                    })
                    .map_err(Self::artifact_error)?;
            }
            outputs.insert(
                name.clone(),
                RuntimeValue::LazyArtifactFamily {
                    production: production.clone(),
                    name: name.clone(),
                    descriptor: replacement_descriptors[&name].clone(),
                },
            );
        }
        Ok((Some(production), handles))
    }

    /// Reopening a finalized session is a read-only, idempotent operation.
    /// The graph is still evaluated so callers receive the same runtime
    /// values (including transcript-backed samples), but output persistence
    /// is reconstructed from the immutable manifest rather than attempting a
    /// mutation against the finalized session.
    fn replay_finalized_outputs(
        &mut self,
        outputs: &BTreeMap<String, RuntimeValue>,
    ) -> Result<(Option<ProductionId>, BTreeMap<String, Vec<ArtifactHandle>>), ExecutionError> {
        let manifest = self
            .artifact_store
            .load_finalized_manifest(&self.production)
            .map_err(Self::artifact_error)?;
        let mut handles = BTreeMap::new();
        for (name, output_root) in self.validated.source.outputs() {
            let Some(availability) = output_root.availability else {
                continue;
            };
            if !outputs.contains_key(name) {
                continue;
            }
            let wire = WireId { instantiation_path: Vec::new(), wire: output_root.value };
            let concrete_type = self
                .validated
                .root_scope()
                .wire_types
                .get(&output_root.value)
                .ok_or_else(|| ExecutionError::MissingMetadata(wire.clone()))?;
            let (element_type, family_count) = match concrete_type {
                ConcreteWireType::IndexedFamily { element, count } => {
                    (element.as_ref(), Some(*count))
                }
                scalar => (scalar, None),
            };
            let artifact_type = ArtifactType::from_wire_type(element_type).ok_or_else(|| {
                ExecutionError::Manifest(format!("output {name} is not artifact-compatible"))
            })?;
            let descriptor = manifest.artifacts.get(name).ok_or_else(|| {
                ExecutionError::Manifest(format!(
                    "finalized session manifest is missing output {name}"
                ))
            })?;
            if descriptor.artifact_type != artifact_type ||
                descriptor.availability != availability ||
                descriptor.family_count != family_count
            {
                return Err(ExecutionError::Manifest(format!(
                    "finalized session manifest does not match output {name}"
                )));
            }
            let indices: Box<dyn Iterator<Item = Option<usize>>> = match family_count {
                Some(count) => Box::new((0..count).map(Some)),
                None => Box::new(std::iter::once(None)),
            };
            let output_handles = indices
                .map(|index| ArtifactHandle {
                    key: ArtifactKey {
                        production: self.production.clone(),
                        name: name.clone(),
                        index,
                    },
                    artifact_type: descriptor.artifact_type.clone(),
                    availability: descriptor.availability,
                    layout: descriptor.layout.clone(),
                })
                .collect();
            handles.insert(name.clone(), output_handles);
        }
        Ok((Some(self.production.clone()), handles))
    }

    fn staged_family_descriptor(
        &mut self,
        scope_id: &FrozenGraphScopeId,
        path: &[InstantiationFrame],
        node: NodeId,
        port: u32,
        count: usize,
    ) -> Result<Option<(String, ManifestArtifact)>, ExecutionError> {
        let wire_id =
            WireId { instantiation_path: path.to_vec(), wire: WireRef { node, port: Port(port) } };
        let Some(ConcreteWireType::IndexedFamily { element, count: validated_count }) =
            self.validated_wire_type(scope_id, wire_id.wire)
        else {
            return Ok(None);
        };
        if *validated_count != count {
            return Err(ExecutionError::MissingMetadata(wire_id));
        }
        let Some(artifact_type) = ArtifactType::from_wire_type(element) else {
            return Ok(None);
        };
        let encoded = mxx_ir_core::encoding::canonical_json(&wire_id)
            .map_err(|error| ExecutionError::Manifest(error.to_string()))?;
        let digest = Sha256::digest(encoded);
        let name = format!("runtime-staged-{}", hex_bytes(&digest));
        let descriptor = ManifestArtifact {
            artifact_type,
            family_count: Some(count),
            // This is a materialized value produced by the active runtime
            // scope.  It has no public deterministic regeneration recipe;
            // staging therefore carries the payload across the scope
            // boundary instead of masquerading as a cache hit.
            availability: ArtifactAvailability::Transferred,
            layout: Some("runtime/staged-family-v1".to_owned()),
        };
        // Scratch identity is private to streamed families. Ordinary resident
        // executions need neither its random nonce nor its domain-separated hash.
        // Initialize at registration so even an empty returned family owns the
        // same identity as every other scratch artifact in this execution.
        let production = &self.production;
        let scratch = self
            .scratch_production
            .get_or_insert_with(|| scratch_production_id(production, rand::random()));
        self.staged_families.insert((scratch.clone(), name.clone()), descriptor.clone());
        Ok(Some((name, descriptor)))
    }

    fn cleanup_unreturned_staged_families(
        &mut self,
        outputs: &BTreeMap<String, RuntimeValue>,
    ) -> Result<Vec<StagedFamilyLease>, ExecutionError> {
        let mut retained = BTreeMap::new();
        for value in outputs.values() {
            collect_staged_families(value, &mut retained);
        }
        let staged = self.staged_families.clone();
        let mut leases = Vec::new();
        for ((production, name), descriptor) in staged {
            if retained.contains_key(&(production.clone(), name.clone())) {
                leases.push(StagedFamilyLease { production, name, descriptor });
                continue;
            }
            let Some(count) = descriptor.family_count else {
                return Err(self.staged_cleanup_error("staged family descriptor has no cardinality"));
            };
            for index in 0..count {
                let key = ArtifactKey {
                    production: production.clone(),
                    name: name.clone(),
                    index: Some(index),
                };
                if let Err(error) = self.artifact_store.remove_staged(&key) {
                    return Err(self.staged_cleanup_error(error));
                }
            }
        }
        self.staged_families.clear();
        Ok(leases)
    }

    fn cleanup_all_staged_families(&mut self) -> Result<(), ExecutionError> {
        let staged = self.staged_families.clone();
        for ((production, name), descriptor) in staged {
            let Some(count) = descriptor.family_count else {
                return Err(self.staged_cleanup_error("staged family descriptor has no cardinality"));
            };
            for index in 0..count {
                let key = ArtifactKey {
                    production: production.clone(),
                    name: name.clone(),
                    index: Some(index),
                };
                if let Err(error) = self.artifact_store.remove_staged(&key) {
                    return Err(self.staged_cleanup_error(error));
                }
            }
        }
        self.staged_families.clear();
        Ok(())
    }

    fn staged_cleanup_error(&self, error: impl std::fmt::Display) -> ExecutionError {
        let leases = self
            .staged_families
            .iter()
            .map(|((production, name), descriptor)| StagedFamilyLease {
                production: production.clone(),
                name: name.clone(),
                descriptor: descriptor.clone(),
            })
            .collect();
        ExecutionError::StagedCleanup { message: error.to_string(), leases }
    }

    fn encode_artifact(
        &self,
        value: &RuntimeValue,
        artifact_type: &ArtifactType,
    ) -> Result<ArtifactPayload, ExecutionError> {
        match (value, artifact_type) {
            (RuntimeValue::Int(value), ArtifactType::Int) => {
                Ok(ArtifactPayload::Bytes(value.to_signed_bytes_le()))
            }
            (RuntimeValue::Matrix(value), ArtifactType::Matrix(matrix_type))
                if value.wire_type() == &ConcreteWireType::Matrix(matrix_type.clone()) =>
            {
                let matrix = value.as_cpu_full().ok_or_else(|| {
                    ExecutionError::Backend("CPU executor cannot encode a non-CPU matrix".into())
                })?;
                Ok(ArtifactPayload::Matrix(self.backend.matrix_to_bytes(matrix)))
            }
            (
                RuntimeValue::Matrix(value),
                ArtifactType::SmallMatrix { matrix, max_coefficient_bound, bound_domain },
            ) if value.wire_type() ==
                &ConcreteWireType::SmallMatrix {
                    matrix: matrix.clone(),
                    max_coefficient_bound: max_coefficient_bound.clone(),
                    bound_domain: *bound_domain,
                } =>
            {
                let native = value.as_cpu_compact().ok_or_else(|| {
                    ExecutionError::Backend(
                        "CPU executor cannot encode a non-CPU compact matrix".into(),
                    )
                })?;
                let schema = ConcreteBoundedMatrixSchema {
                    matrix: matrix.clone(),
                    max_coefficient_bound: max_coefficient_bound.clone(),
                    bound_domain: *bound_domain,
                };
                let bytes = self
                    .backend
                    .small_matrix_to_bytes(native, &schema, SmallMatrixSemanticKind::Generic)
                    .map_err(Self::backend_error)?;
                Ok(ArtifactPayload::SmallMatrix(bytes))
            }
            (
                RuntimeValue::Matrix(value),
                ArtifactType::Preimage { matrix, max_coefficient_bound, bound_domain },
            ) if value.wire_type() ==
                &ConcreteWireType::Preimage {
                    matrix: matrix.clone(),
                    max_coefficient_bound: max_coefficient_bound.clone(),
                    bound_domain: *bound_domain,
                } =>
            {
                let native = value.as_cpu_compact().ok_or_else(|| {
                    ExecutionError::Backend("CPU executor cannot encode a non-CPU preimage".into())
                })?;
                let schema = ConcreteBoundedMatrixSchema {
                    matrix: matrix.clone(),
                    max_coefficient_bound: max_coefficient_bound.clone(),
                    bound_domain: *bound_domain,
                };
                let bytes = self
                    .backend
                    .small_matrix_to_bytes(native, &schema, SmallMatrixSemanticKind::Preimage)
                    .map_err(Self::backend_error)?;
                Ok(ArtifactPayload::SmallMatrix(bytes))
            }
            (RuntimeValue::Bytes(bytes), ArtifactType::Bytes { length })
                if bytes.len() == *length =>
            {
                Ok(ArtifactPayload::Bytes(bytes.to_vec()))
            }
            (
                RuntimeValue::TypedBlob { type_name, schema_hash, bytes },
                ArtifactType::TypedBlob { type_name: expected_name, schema_hash: expected_hash },
            ) if type_name == expected_name && schema_hash == expected_hash => {
                Ok(ArtifactPayload::TypedBlob(bytes.to_vec()))
            }
            (RuntimeValue::Trapdoor(value), ArtifactType::Trapdoor { .. })
                if ArtifactType::from_wire_type(value.wire_type()).as_ref() ==
                    Some(artifact_type) =>
            {
                let public = value.public_matrix().as_cpu_full().ok_or_else(|| {
                    ExecutionError::Backend(
                        "CPU executor cannot encode a non-CPU trapdoor public matrix".into(),
                    )
                })?;
                let secret = value.cpu_secret().ok_or_else(|| {
                    ExecutionError::Manifest("trapdoor artifact has no secret".into())
                })?;
                Ok(ArtifactPayload::Trapdoor {
                    public_bytes: self.backend.matrix_to_bytes(public),
                    secret_bytes: self.backend.trapdoor_to_bytes(secret),
                })
            }
            _ => Err(ExecutionError::Manifest(
                "runtime value does not match declared artifact type".into(),
            )),
        }
    }

    fn materialize(
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue>,
        wire: WireRef,
    ) -> Result<RuntimeValue, ExecutionError> {
        let value = values.get(&wire).cloned().ok_or(ExecutionError::MissingWire(wire))?;
        let is_artifact = matches!(
            value,
            RuntimeValue::LazyArtifact { .. } | RuntimeValue::StagedArtifact { .. }
        );
        let resolved = self.materialize_value(value)?;
        if is_artifact {
            values.insert(wire, resolved.clone());
        }
        Ok(resolved)
    }

    fn materialize_value(&mut self, value: RuntimeValue) -> Result<RuntimeValue, ExecutionError> {
        match value {
            RuntimeValue::LazyArtifact { production, name, index, descriptor } => {
                let key = ArtifactKey { production, name, index };
                let payload = self.load_artifact_payload(&key, &descriptor, false)?;
                self.decode_artifact(descriptor.artifact_type, payload)
            }
            RuntimeValue::StagedArtifact { production, name, index, descriptor } => {
                let key = ArtifactKey { production, name, index: Some(index) };
                let payload = self.load_artifact_payload(&key, &descriptor, true)?;
                self.decode_artifact(descriptor.artifact_type, payload)
            }
            other => Ok(other),
        }
    }

    fn load_artifact_payload(
        &mut self,
        key: &ArtifactKey,
        descriptor: &ManifestArtifact,
        staged: bool,
    ) -> Result<ArtifactPayload, ExecutionError> {
        let payload = if staged {
            self.artifact_store.load_staged(key, descriptor)
        } else {
            self.artifact_store.load(key, descriptor)
        };
        payload.map_err(Self::artifact_error)
    }

    fn decode_artifact(
        &self,
        artifact_type: ArtifactType,
        payload: ArtifactPayload,
    ) -> Result<RuntimeValue, ExecutionError> {
        decode_artifact(self.backend, artifact_type, payload)
    }

    fn matrix(
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue>,
        wire: WireRef,
    ) -> Result<Arc<crate::matrix::dcrt_poly::DCRTPolyMatrix>, ExecutionError> {
        let RuntimeValue::Matrix(value) = self.materialize(values, wire)? else {
            return Err(ExecutionError::ValueKind(wire));
        };
        value.cpu_full_arc().ok_or(ExecutionError::ValueKind(wire))
    }

    fn small_matrix(
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue>,
        wire: WireRef,
    ) -> Result<
        Arc<crate::matrix::CpuSmallMatrix<crate::matrix::dcrt_poly::DCRTPolyMatrix>>,
        ExecutionError,
    > {
        let RuntimeValue::Matrix(value) = self.materialize(values, wire)? else {
            return Err(ExecutionError::ValueKind(wire));
        };
        value.cpu_compact_arc().ok_or(ExecutionError::ValueKind(wire))
    }

    fn fence_pending_releases(&mut self) -> Result<(), ExecutionError> {
        if self.has_pending_releases {
            self.backend.fence_released_memory().map_err(Self::backend_error)?;
            self.has_pending_releases = false;
            self.last_release_fence_node_count = self.executed_node_count;
        }
        Ok(())
    }

    fn value(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue>,
        wire: WireRef,
    ) -> Result<RuntimeValue, ExecutionError> {
        values.get(&wire).cloned().ok_or(ExecutionError::MissingWire(wire))
    }

    fn int(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue>,
        wire: WireRef,
    ) -> Result<BigInt, ExecutionError> {
        match self.value(values, wire)? {
            RuntimeValue::Int(value) => Ok(value),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn host_primitive_value(
        &self,
        values: &BTreeMap<WireRef, RuntimeValue>,
        wire: WireRef,
    ) -> Result<HostPrimitiveValue, ExecutionError> {
        match self.value(values, wire)? {
            RuntimeValue::Int(value) => Ok(HostPrimitiveValue::Int(value)),
            RuntimeValue::Real(value) => Ok(HostPrimitiveValue::Real(value)),
            RuntimeValue::Bool(value) => Ok(HostPrimitiveValue::Bool(value)),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn runtime_host_primitive(value: HostPrimitiveValue) -> RuntimeValue {
        match value {
            HostPrimitiveValue::Int(value) => RuntimeValue::Int(value),
            HostPrimitiveValue::Real(value) => RuntimeValue::Real(value),
            HostPrimitiveValue::Bool(value) => RuntimeValue::Bool(value),
        }
    }

    fn host_primitive_error(&self, node: NodeId, error: HostPrimitiveError) -> ExecutionError {
        match error {
            HostPrimitiveError::DivisionByZero => ExecutionError::DivisionByZero(node),
            HostPrimitiveError::InvalidRealOperation => ExecutionError::InvalidRealOperation(node),
            HostPrimitiveError::Expression(message) => ExecutionError::Expression { node, message },
            HostPrimitiveError::ValueKind { .. } | HostPrimitiveError::NotScalar(_) => {
                ExecutionError::ValueKind(WireRef { node, port: Port(0) })
            }
        }
    }

    fn bytes(
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue>,
        wire: WireRef,
    ) -> Result<Vec<u8>, ExecutionError> {
        match self.materialize(values, wire)? {
            RuntimeValue::Bytes(value) => Ok(value.to_vec()),
            _ => Err(ExecutionError::ValueKind(wire)),
        }
    }

    fn trapdoor(
        &mut self,
        values: &mut BTreeMap<WireRef, RuntimeValue>,
        wire: WireRef,
    ) -> Result<TrapdoorParts, ExecutionError> {
        let RuntimeValue::Trapdoor(value) = self.materialize(values, wire)? else {
            return Err(ExecutionError::ValueKind(wire));
        };
        let ConcreteWireType::Trapdoor { matrix, sigma, gadget_base, digit_count, .. } =
            value.wire_type()
        else {
            return Err(ExecutionError::ValueKind(wire));
        };
        let public = value.public_matrix().cpu_full_arc().ok_or(ExecutionError::ValueKind(wire))?;
        let sigma = sigma
            .evaluate_f64_with_rings(
                &ParamEnv::default(),
                crate::openfhe_guard::gen_modulus_and_warmup,
            )
            .map_err(|error| self.expression_error(wire.node, error))?;
        let secret = value.cpu_secret_arc();
        let gadget_small = secret.is_none().then_some(false);
        Ok((secret, public, matrix.clone(), sigma, gadget_base.clone(), *digit_count, gadget_small))
    }

    fn put(
        &self,
        values: &mut BTreeMap<WireRef, RuntimeValue>,
        node: NodeId,
        port: u32,
        value: RuntimeValue,
    ) {
        values.insert(WireRef { node, port: Port(port) }, value);
    }

    fn set_placement(&mut self, placement: usize) -> Result<(), ExecutionError> {
        if self.backend.set_active_placement(placement) {
            Ok(())
        } else {
            Err(ExecutionError::BackendPlacement {
                placement,
                count: self.backend.placement_count(),
            })
        }
    }

    fn value_for_placement(
        &mut self,
        value: RuntimeValue,
        placement: usize,
    ) -> Result<RuntimeValue, ExecutionError> {
        self.set_placement(placement)?;
        Ok(value)
    }

    fn values_for_placements(
        &mut self,
        value: RuntimeValue,
    ) -> Result<Vec<RuntimeValue>, ExecutionError> {
        Ok(vec![value; self.backend.placement_count()])
    }

    fn resolved_wire_type(
        &self,
        scope_id: &FrozenGraphScopeId,
        wire: WireRef,
        env: &ParamEnv,
    ) -> Result<ConcreteWireType, ExecutionError> {
        if scope_id == &FrozenGraphScopeId::Root {
            return self.validated_wire_type(scope_id, wire).cloned().ok_or_else(|| {
                ExecutionError::MissingMetadata(WireId { instantiation_path: Vec::new(), wire })
            });
        }
        let scope = self.validated.source.scope(scope_id).ok_or_else(|| {
            ExecutionError::MissingSubgraph { node: wire.node, name: format!("{scope_id:?}") }
        })?;
        let declaration = scope
            .node(wire.node)
            .and_then(|node| node.output_types().get(wire.port.0 as usize))
            .ok_or_else(|| {
                ExecutionError::MissingMetadata(WireId { instantiation_path: Vec::new(), wire })
            })?;
        mxx_ir_core::concretize_wire_type(
            declaration,
            env,
            scope_id,
            wire.node,
            crate::openfhe_guard::gen_modulus_and_warmup,
        )
        .map_err(|error| self.expression_error(wire.node, error))
    }

    fn matrix_type(
        &self,
        scope_id: &FrozenGraphScopeId,
        path: &[InstantiationFrame],
        env: &ParamEnv,
        wire: WireRef,
    ) -> Result<ConcreteMatrixType, ExecutionError> {
        let id = WireId { instantiation_path: path.to_vec(), wire };
        self.resolved_wire_type(scope_id, wire, env)?
            .matrix_type()
            .cloned()
            .ok_or(ExecutionError::MissingMetadata(id))
    }

    fn bounded_matrix_schema(
        &self,
        scope_id: &FrozenGraphScopeId,
        path: &[InstantiationFrame],
        env: &ParamEnv,
        wire: WireRef,
    ) -> Result<(ConcreteBoundedMatrixSchema, SmallMatrixSemanticKind), ExecutionError> {
        let id = WireId { instantiation_path: path.to_vec(), wire };
        ArtifactType::from_wire_type(&self.resolved_wire_type(scope_id, wire, env)?)
            .and_then(|artifact| artifact.bounded_matrix_schema())
            .ok_or(ExecutionError::MissingMetadata(id))
    }

    fn validated_wire_type(
        &self,
        scope_id: &FrozenGraphScopeId,
        wire: WireRef,
    ) -> Option<&ConcreteWireType> {
        self.validated.scope(scope_id)?.wire_types.get(&wire)
    }

    fn trapdoor_type(
        &self,
        scope_id: &FrozenGraphScopeId,
        path: &[InstantiationFrame],
        env: &ParamEnv,
        wire: WireRef,
    ) -> Result<ConcreteMatrixType, ExecutionError> {
        self.matrix_type(scope_id, path, env, wire)
    }

    fn child_inputs(
        &mut self,
        child: &GraphScope,
        node: &ExecutableNode<'_>,
        values: &BTreeMap<WireRef, RuntimeValue>,
    ) -> Result<BTreeMap<String, RuntimeValue>, ExecutionError> {
        let names = child.inputs().iter().map(|wire| {
            let input = child.node(wire.node).expect("validated child input node");
            let NodeKind::Input { name, .. } = input.kind() else {
                unreachable!("validated child input must reference an input node")
            };
            name
        });
        names
            .zip(node.args)
            .map(|(name, wire)| {
                let value = self.value(values, *wire)?;
                Ok((name.clone(), self.materialize_value(value)?))
            })
            .collect()
    }

    fn loop_child_inputs(
        &mut self,
        child: &GraphScope,
        node: &ExecutableNode<'_>,
        modes: &[LoopInputMode],
        index: usize,
        placement: usize,
        broadcast_inputs: &[Option<RuntimeValue>],
        values: &BTreeMap<WireRef, RuntimeValue>,
    ) -> Result<BTreeMap<String, RuntimeValue>, ExecutionError> {
        let names = child.inputs().iter().map(|wire| {
            let input = child.node(wire.node).expect("validated child input node");
            let NodeKind::Input { name, .. } = input.kind() else {
                unreachable!("validated child input must reference an input node")
            };
            name
        });
        if modes.len() != node.args.len() || broadcast_inputs.len() != node.args.len() {
            return Err(ExecutionError::ValueKind(WireRef { node: node.id, port: Port(0) }));
        }
        names
            .zip(node.args)
            .zip(modes)
            .zip(broadcast_inputs)
            .map(|(((name, wire), mode), broadcast)| {
                let value = match mode {
                    LoopInputMode::Broadcast => {
                        let value =
                            broadcast.clone().ok_or(ExecutionError::ValueKind(WireRef {
                                node: node.id,
                                port: Port(0),
                            }))?;
                        let value = self.materialize_value(value)?;
                        self.value_for_placement(value, placement)?
                    }
                    LoopInputMode::Zip | LoopInputMode::ZipOffset { .. } => {
                        let offset = match mode {
                            LoopInputMode::Zip => 0,
                            LoopInputMode::ZipOffset { offset } => *offset,
                            LoopInputMode::Broadcast => unreachable!(),
                        };
                        let index = index.checked_add(offset).ok_or(
                            ExecutionError::SelectIndexOutOfRange {
                                node: node.id,
                                index: BigInt::from(index),
                                count: self.family_count(values, *wire)?,
                            },
                        )?;
                        let member = self.family_member_value(values, *wire, index)?;
                        let member = self.materialize_value(member)?;
                        self.value_for_placement(member, placement)?
                    }
                };
                Ok((name.clone(), value))
            })
            .collect()
    }

    fn eval_usize(
        &self,
        node: NodeId,
        expression: &mxx_ir_core::IntExpr,
        env: &ParamEnv,
    ) -> Result<usize, ExecutionError> {
        expression
            .evaluate_with_rings(env, crate::openfhe_guard::gen_modulus_and_warmup)
            .map_err(|error| self.expression_error(node, error))?
            .to_usize()
            .ok_or_else(|| ExecutionError::Expression {
                node,
                message: "expression does not fit usize".to_owned(),
            })
    }

    fn expression_error(&self, node: NodeId, error: impl std::fmt::Display) -> ExecutionError {
        ExecutionError::Expression { node, message: error.to_string() }
    }

    fn backend_error(error: crate::backend::poly::PolyBackendError) -> ExecutionError {
        ExecutionError::Backend(error.to_string())
    }

    fn artifact_error(error: S::Error) -> ExecutionError {
        ExecutionError::Artifact(error.to_string())
    }
}

fn append_tag_integer(tag: &mut Vec<u8>, value: &BigInt) {
    let (sign, bytes) = value.to_bytes_be();
    tag.push(match sign {
        Sign::Minus => 1,
        Sign::NoSign | Sign::Plus => 0,
    });
    tag.extend_from_slice(&(bytes.len() as u64).to_be_bytes());
    tag.extend_from_slice(&bytes);
}

fn preimage_request_seed(
    execution_nonce: [u8; 32],
    path: &[InstantiationFrame],
    wire: WireRef,
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"mxx-backends/preimage-request/v1");
    hasher.update(execution_nonce);
    hasher.update((path.len() as u64).to_le_bytes());
    for frame in path {
        hasher.update(frame.call.0.to_le_bytes());
        match frame.loop_index {
            Some(index) => {
                hasher.update([1]);
                hasher.update(index.to_le_bytes());
            }
            None => hasher.update([0]),
        }
    }
    hasher.update(wire.node.0.to_le_bytes());
    hasher.update(wire.port.0.to_le_bytes());
    hasher.finalize().into()
}

/// Remove CPU values after their final scheduled consumer. Dropping these
/// values does not enqueue asynchronous backend releases.
pub(crate) fn release_last_uses(
    schedule: &mxx_ir_core::LivenessSchedule,
    position: usize,
    arguments: &[WireRef],
    values: &mut BTreeMap<WireRef, RuntimeValue>,
) {
    for argument in arguments {
        if schedule.last_use.get(argument) == Some(&position) &&
            !schedule.retained.contains(argument)
        {
            values.remove(argument);
        }
    }
}

/// Canonical digest used by the producer-session path. Transient execution
/// must not call this helper; it serializes concrete input contents.
pub(crate) fn runtime_inputs_digest(
    validated: &ValidatedGraph,
    backend: &CpuDcrtBackend,
    inputs: &BTreeMap<String, RuntimeValue>,
) -> Result<[u8; 32], ExecutionError> {
    let root = validated.source.root_scope();
    let input_types = root
        .nodes()
        .iter()
        .enumerate()
        .filter_map(|(index, node)| {
            let NodeKind::Input { name, artifact: None, .. } = node.kind() else {
                return None;
            };
            let wire = WireRef { node: NodeId(index as u64), port: Port(0) };
            let concrete = validated
                .root_scope()
                .wire_types
                .get(&wire)
                .expect("validated root input has concrete metadata");
            Some((name.as_str(), concrete))
        })
        .collect::<BTreeMap<_, _>>();
    let mut hasher = Sha256::new();
    hasher.update(b"mxx-backends-session-inputs-v1");
    for (name, value) in inputs {
        let concrete = input_types.get(name.as_str()).ok_or_else(|| {
            ExecutionError::Manifest(format!("runtime input {name} is not declared by the graph"))
        })?;
        if !runtime_value_matches_wire_type(value, concrete) {
            return Err(ExecutionError::Manifest(format!(
                "runtime input {name} does not match its validated wire type"
            )));
        }
        hash_sized(&mut hasher, name.as_bytes());
        hash_runtime_value(backend, value, concrete, &mut hasher)?;
    }
    Ok(hasher.finalize().into())
}

pub(crate) fn hash_runtime_value(
    backend: &CpuDcrtBackend,
    value: &RuntimeValue,
    concrete: &ConcreteWireType,
    hasher: &mut Sha256,
) -> Result<(), ExecutionError> {
    match value {
        RuntimeValue::Composite(_) => {
            return Err(ExecutionError::Backend(
                "composite inputs are expanded into their leaves before hashing".into(),
            ));
        }
        RuntimeValue::Int(value) => {
            hasher.update([0]);
            hash_sized(hasher, value.to_string().as_bytes());
        }
        RuntimeValue::Real(value) => {
            hasher.update([1]);
            hasher.update(value.to_bits().to_le_bytes());
        }
        RuntimeValue::Bool(value) => hasher.update([2, u8::from(*value)]),
        RuntimeValue::Bytes(value) => {
            hasher.update([3]);
            hash_sized(hasher, value);
        }
        RuntimeValue::TypedBlob { bytes, .. } => {
            hasher.update([4]);
            hash_sized(hasher, bytes);
        }
        RuntimeValue::Matrix(value) => match concrete {
            ConcreteWireType::Matrix(_) => {
                let native = value.as_cpu_full().ok_or_else(|| {
                    ExecutionError::Backend(
                        "CPU input digest cannot encode a non-CPU matrix".into(),
                    )
                })?;
                hasher.update([5]);
                hash_sized(hasher, &backend.matrix_to_bytes(native));
            }
            ConcreteWireType::SmallMatrix { .. } | ConcreteWireType::Preimage { .. } => {
                let native = value.as_cpu_compact().ok_or_else(|| {
                    ExecutionError::Backend(
                        "CPU input digest cannot encode a non-CPU compact matrix".into(),
                    )
                })?;
                let artifact = ArtifactType::from_wire_type(concrete).ok_or_else(|| {
                    ExecutionError::Manifest("bounded matrix input has no artifact schema".into())
                })?;
                let (schema, semantic_kind) =
                    artifact.bounded_matrix_schema().ok_or_else(|| {
                        ExecutionError::Manifest(
                            "bounded matrix input has no artifact schema".into(),
                        )
                    })?;
                let bytes = backend
                    .small_matrix_to_bytes(native, &schema, semantic_kind)
                    .map_err(|error| ExecutionError::Backend(error.to_string()))?;
                hasher.update([if semantic_kind == SmallMatrixSemanticKind::Preimage {
                    13
                } else {
                    12
                }]);
                hash_sized(hasher, &bytes);
            }
            _ => {
                return Err(ExecutionError::Manifest(
                    "matrix input has non-matrix wire metadata".into(),
                ))
            }
        },
        RuntimeValue::Trapdoor(value) => {
            hasher.update([6]);
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(value.wire_type())
                    .map_err(|error| ExecutionError::Manifest(error.to_string()))?,
            );
            let public = value.public_matrix().as_cpu_full().ok_or_else(|| {
                ExecutionError::Backend(
                    "CPU input digest cannot encode a non-CPU trapdoor public matrix".into(),
                )
            })?;
            hash_sized(hasher, &backend.matrix_to_bytes(public));
            match value.cpu_secret() {
                Some(secret) => {
                    hasher.update([1]);
                    hash_sized(hasher, &backend.trapdoor_to_bytes(secret));
                }
                None => hasher.update([0]),
            }
        }
        #[cfg(feature = "gpu")]
        RuntimeValue::Resident(_) => {
            return Err(ExecutionError::Backend(
                "CPU input digest cannot encode a GPU resident value".into(),
            ));
        }
        RuntimeValue::LazyArtifact { production, name, index, descriptor } => {
            hasher.update([7]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hasher.update(index.unwrap_or(usize::MAX).to_le_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| ExecutionError::Manifest(error.to_string()))?,
            );
        }
        RuntimeValue::LazyArtifactFamily { production, name, descriptor } => {
            hasher.update([8]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| ExecutionError::Manifest(error.to_string()))?,
            );
        }
        RuntimeValue::StagedArtifact { production, name, index, descriptor } => {
            hasher.update([9]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hasher.update(index.to_le_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| ExecutionError::Manifest(error.to_string()))?,
            );
        }
        RuntimeValue::StagedArtifactFamily { production, name, descriptor } => {
            hasher.update([10]);
            hasher.update(production.spec_hash.0);
            hasher.update(production.execution_nonce);
            hash_sized(hasher, name.as_bytes());
            hash_sized(
                hasher,
                &mxx_ir_core::encoding::canonical_json(descriptor)
                    .map_err(|error| ExecutionError::Manifest(error.to_string()))?,
            );
        }
        RuntimeValue::IndexedFamily { element_type, values } => {
            let ConcreteWireType::IndexedFamily { element, count } = concrete else {
                return Err(ExecutionError::Manifest(
                    "indexed input has non-family metadata".into(),
                ));
            };
            if values.len() != *count || element_type != element.as_ref() {
                return Err(ExecutionError::Manifest(
                    "indexed input does not match validated family metadata".into(),
                ));
            }
            hasher.update([11]);
            hasher.update(values.len().to_le_bytes());
            for value in values.iter() {
                hash_runtime_value(backend, value, element, hasher)?;
            }
        }
    }
    Ok(())
}

fn runtime_value_matches_wire_type(value: &RuntimeValue, concrete: &ConcreteWireType) -> bool {
    value.matches_wire_type(concrete)
}

fn compact_runtime_value(
    value: crate::matrix::CpuSmallMatrix<crate::matrix::dcrt_poly::DCRTPolyMatrix>,
    semantic_kind: SmallMatrixSemanticKind,
) -> RuntimeValue {
    match semantic_kind {
        SmallMatrixSemanticKind::Generic => RuntimeValue::small_matrix(value),
        SmallMatrixSemanticKind::Preimage => RuntimeValue::preimage(value),
    }
}

pub(crate) fn hash_sized(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update(bytes.len().to_le_bytes());
    hasher.update(bytes);
}

fn scratch_production_id(
    production: &ProductionId,
    scratch_execution_nonce: [u8; 32],
) -> ProductionId {
    let mut hasher = Sha256::new();
    hasher.update(b"mxx-backends/staged-family/v1");
    hasher.update(&production.spec_hash.0);
    hasher.update(production.execution_nonce);
    ProductionId {
        spec_hash: mxx_ir_core::artifact::SpecHash(hasher.finalize().into()),
        execution_nonce: scratch_execution_nonce,
    }
}

fn collect_staged_families(
    value: &RuntimeValue,
    families: &mut BTreeMap<(ProductionId, String), usize>,
) {
    match value {
        RuntimeValue::StagedArtifactFamily { production, name, descriptor } => {
            if let Some(count) = descriptor.family_count {
                families.insert((production.clone(), name.clone()), count);
            }
        }
        RuntimeValue::IndexedFamily { values, .. } => {
            for value in values.iter() {
                collect_staged_families(value, families);
            }
        }
        _ => {}
    }
}

pub(crate) fn materialize_runtime_value<S: ArtifactStore>(
    value: RuntimeValue,
    backend: &CpuDcrtBackend,
    store: &mut S,
) -> Result<RuntimeValue, ExecutionError> {
    match value {
        RuntimeValue::LazyArtifact { production, name, index, descriptor } => {
            let payload = store
                .load(&ArtifactKey { production, name, index }, &descriptor)
                .map_err(|error| ExecutionError::Artifact(error.to_string()))?;
            decode_artifact(backend, descriptor.artifact_type, payload)
        }
        RuntimeValue::StagedArtifact { production, name, index, descriptor } => {
            let payload = store
                .load_staged(&ArtifactKey { production, name, index: Some(index) }, &descriptor)
                .map_err(|error| ExecutionError::Artifact(error.to_string()))?;
            decode_artifact(backend, descriptor.artifact_type, payload)
        }
        RuntimeValue::LazyArtifactFamily { production, name, descriptor } => {
            materialize_artifact_family(production, name, descriptor, false, backend, store)
        }
        RuntimeValue::StagedArtifactFamily { production, name, descriptor } => {
            materialize_artifact_family(production, name, descriptor, true, backend, store)
        }
        RuntimeValue::IndexedFamily { element_type, values } => {
            let materialized = values
                .iter()
                .cloned()
                .map(|value| materialize_runtime_value(value, backend, store))
                .collect::<Result<Vec<_>, _>>()?;
            RuntimeValue::indexed_family(element_type, materialized)
                .map_err(|error| ExecutionError::Manifest(error.into()))
        }
        value => Ok(value),
    }
}

fn materialize_artifact_family<S: ArtifactStore>(
    production: ProductionId,
    name: String,
    descriptor: ManifestArtifact,
    staged: bool,
    backend: &CpuDcrtBackend,
    store: &mut S,
) -> Result<RuntimeValue, ExecutionError> {
    let count = descriptor
        .family_count
        .ok_or_else(|| ExecutionError::Manifest("artifact family has no cardinality".into()))?;
    let element_type = artifact_wire_type(&descriptor.artifact_type);
    let mut values = Vec::with_capacity(count);
    for index in 0..count {
        let key =
            ArtifactKey { production: production.clone(), name: name.clone(), index: Some(index) };
        let payload = if staged {
            store.load_staged(&key, &descriptor)
        } else {
            store.load(&key, &descriptor)
        }
        .map_err(|error| ExecutionError::Artifact(error.to_string()))?;
        values.push(decode_artifact(backend, descriptor.artifact_type.clone(), payload)?);
    }
    RuntimeValue::indexed_family(element_type, values)
        .map_err(|error| ExecutionError::Manifest(error.into()))
}

fn artifact_wire_type(artifact: &ArtifactType) -> ConcreteWireType {
    match artifact {
        ArtifactType::Int => ConcreteWireType::Int,
        ArtifactType::Matrix(matrix) => ConcreteWireType::Matrix(matrix.clone()),
        ArtifactType::SmallMatrix { matrix, max_coefficient_bound, bound_domain } => {
            ConcreteWireType::SmallMatrix {
                matrix: matrix.clone(),
                max_coefficient_bound: max_coefficient_bound.clone(),
                bound_domain: *bound_domain,
            }
        }
        ArtifactType::Preimage { matrix, max_coefficient_bound, bound_domain } => {
            ConcreteWireType::Preimage {
                matrix: matrix.clone(),
                max_coefficient_bound: max_coefficient_bound.clone(),
                bound_domain: *bound_domain,
            }
        }
        ArtifactType::Bytes { length } => ConcreteWireType::Bytes { length: *length },
        ArtifactType::Trapdoor {
            matrix,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => ConcreteWireType::Trapdoor {
            matrix: matrix.clone(),
            sigma: sigma.clone(),
            gadget_base: gadget_base.clone(),
            digit_count: *digit_count,
            preimage_max_coefficient_bound: preimage_max_coefficient_bound.clone(),
        },
        ArtifactType::TypedBlob { type_name, schema_hash } => {
            ConcreteWireType::TypedBlob { type_name: type_name.clone(), schema_hash: *schema_hash }
        }
    }
}

fn decode_artifact(
    backend: &CpuDcrtBackend,
    artifact_type: ArtifactType,
    payload: ArtifactPayload,
) -> Result<RuntimeValue, ExecutionError> {
    match (artifact_type, payload) {
        (ArtifactType::Int, ArtifactPayload::Bytes(bytes))
            if BigInt::from_signed_bytes_le(&bytes).to_signed_bytes_le() == bytes =>
        {
            Ok(RuntimeValue::Int(BigInt::from_signed_bytes_le(&bytes)))
        }
        (ArtifactType::Matrix(matrix_type), ArtifactPayload::Matrix(bytes)) => {
            let matrix = backend
                .matrix_from_bytes(&matrix_type, &bytes)
                .map_err(|error| ExecutionError::Backend(error.to_string()))?;
            let value = PolyMatrix::cpu_full(ConcreteWireType::Matrix(matrix_type), matrix)
                .map_err(|error| ExecutionError::Artifact(error.into()))?;
            Ok(RuntimeValue::Matrix(value))
        }
        (
            ArtifactType::SmallMatrix { matrix, max_coefficient_bound, bound_domain },
            ArtifactPayload::SmallMatrix(bytes),
        ) => {
            let schema = ConcreteBoundedMatrixSchema {
                matrix: matrix.clone(),
                max_coefficient_bound: max_coefficient_bound.clone(),
                bound_domain,
            };
            let native = backend
                .small_matrix_from_bytes(&schema, &bytes, SmallMatrixSemanticKind::Generic)
                .map_err(|error| ExecutionError::Backend(error.to_string()))?;
            let ty = ConcreteWireType::SmallMatrix { matrix, max_coefficient_bound, bound_domain };
            let value = PolyMatrix::cpu_compact(ty, native)
                .map_err(|error| ExecutionError::Artifact(error.into()))?;
            Ok(RuntimeValue::Matrix(value))
        }
        (
            ArtifactType::Preimage { matrix, max_coefficient_bound, bound_domain },
            ArtifactPayload::SmallMatrix(bytes),
        ) => {
            let schema = ConcreteBoundedMatrixSchema {
                matrix: matrix.clone(),
                max_coefficient_bound: max_coefficient_bound.clone(),
                bound_domain,
            };
            let native = backend
                .small_matrix_from_bytes(&schema, &bytes, SmallMatrixSemanticKind::Preimage)
                .map_err(|error| ExecutionError::Backend(error.to_string()))?;
            let ty = ConcreteWireType::Preimage { matrix, max_coefficient_bound, bound_domain };
            let value = PolyMatrix::cpu_compact(ty, native)
                .map_err(|error| ExecutionError::Artifact(error.into()))?;
            Ok(RuntimeValue::Matrix(value))
        }
        (ArtifactType::Bytes { length }, ArtifactPayload::Bytes(bytes))
            if bytes.len() == length =>
        {
            Ok(RuntimeValue::Bytes(bytes.into()))
        }
        (ArtifactType::TypedBlob { type_name, schema_hash }, ArtifactPayload::TypedBlob(bytes)) => {
            Ok(RuntimeValue::TypedBlob { type_name, schema_hash, bytes: bytes.into() })
        }
        (
            ArtifactType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            },
            ArtifactPayload::Trapdoor { public_bytes, secret_bytes },
        ) => {
            let public = backend
                .matrix_from_bytes(&matrix, &public_bytes)
                .map_err(|error| ExecutionError::Backend(error.to_string()))?;
            let secret = backend
                .trapdoor_from_bytes(&matrix, &secret_bytes)
                .map_err(|error| ExecutionError::Backend(error.to_string()))?;
            let public = PolyMatrix::cpu_full(ConcreteWireType::Matrix(matrix.clone()), public)
                .map_err(|error| ExecutionError::Artifact(error.into()))?;
            let ty = ConcreteWireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            };
            let value = TrapdoorValue::new(ty, public, Some(Arc::new(secret)))
                .map_err(|error| ExecutionError::Artifact(error.into()))?;
            Ok(RuntimeValue::Trapdoor(value))
        }
        _ => Err(ExecutionError::Artifact(
            "stored payload kind does not match artifact descriptor".into(),
        )),
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        backend::poly::cpu_backend,
        matrix::{CpuSmallMatrix, dcrt_poly::DCRTPolyMatrix},
        poly::{PolyParams, dcrt::params::DCRTPolyParams},
    };
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::{IntExpr, types::CoefficientBoundDomain};

    fn matrix_type(parameters: &DCRTPolyParams) -> ConcreteMatrixType {
        let ring = Ring::from_crt_moduli(
            parameters.to_crt().0.into_iter().map(IntExpr::from).collect(),
            parameters.ring_dimension(),
        );
        ConcreteMatrixType {
            ring: ring
                .as_ref()
                .resolve(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
                .expect("test CRT basis resolves"),
            rows: 1,
            columns: 1,
        }
    }

    #[test]
    fn decode_artifact_preserves_compact_semantic_kind() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let matrix = matrix_type(&parameters);
        let bound = BigInt::from(1u8);
        let schema = ConcreteBoundedMatrixSchema {
            matrix: matrix.clone(),
            max_coefficient_bound: bound.clone(),
            bound_domain: CoefficientBoundDomain::Global,
        };
        let native =
            CpuSmallMatrix::new(DCRTPolyMatrix::identity(&parameters, 1, None), 1u8.into())
                .expect("bounded compact matrix");
        let backend = cpu_backend([parameters]);
        for (kind, artifact_type, expected_type) in [
            (
                SmallMatrixSemanticKind::Generic,
                ArtifactType::SmallMatrix {
                    matrix: matrix.clone(),
                    max_coefficient_bound: bound.clone(),
                    bound_domain: CoefficientBoundDomain::Global,
                },
                ConcreteWireType::SmallMatrix {
                    matrix: matrix.clone(),
                    max_coefficient_bound: bound.clone(),
                    bound_domain: CoefficientBoundDomain::Global,
                },
            ),
            (
                SmallMatrixSemanticKind::Preimage,
                ArtifactType::Preimage {
                    matrix: matrix.clone(),
                    max_coefficient_bound: bound.clone(),
                    bound_domain: CoefficientBoundDomain::Global,
                },
                ConcreteWireType::Preimage {
                    matrix: matrix.clone(),
                    max_coefficient_bound: bound.clone(),
                    bound_domain: CoefficientBoundDomain::Global,
                },
            ),
        ] {
            let bytes = backend
                .small_matrix_to_bytes(&native, &schema, kind)
                .expect("canonical compact bytes");
            let value =
                decode_artifact(&backend, artifact_type, ArtifactPayload::SmallMatrix(bytes))
                    .expect("compact artifact decodes");
            assert!(value.matches_wire_type(&expected_type));
            assert!(matches!(value, RuntimeValue::Matrix(_)));
        }
    }

    #[test]
    fn session_digest_rejects_cross_kind_compact_inputs() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4, None, None);
        let ring = Ring::from_crt_moduli(
            parameters.to_crt().0.into_iter().map(IntExpr::from).collect(),
            parameters.ring_dimension(),
        );
        let generic = DslContext::new("generic-session-input")
            .output("value", ring.small_matrix_input("value", (1, 1), 1))
            .expect("generic output")
            .build()
            .expect("generic graph")
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .expect("generic validation");
        let preimage = DslContext::new("preimage-session-input")
            .output("value", ring.preimage_input("value", (1, 1), 1))
            .expect("preimage output")
            .build()
            .expect("preimage graph")
            .validate(&ParamEnv::default(), crate::openfhe_guard::gen_modulus_and_warmup)
            .expect("preimage validation");
        let native =
            CpuSmallMatrix::new(DCRTPolyMatrix::identity(&parameters, 1, None), 1u8.into())
                .expect("bounded compact matrix");
        let backend = cpu_backend([parameters]);
        let small = RuntimeValue::small_matrix(native.clone());
        let relation = RuntimeValue::preimage(native);
        let small_digest = runtime_inputs_digest(
            &generic,
            &backend,
            &BTreeMap::from([("value".into(), small.clone())]),
        )
        .expect("generic digest");
        let preimage_digest = runtime_inputs_digest(
            &preimage,
            &backend,
            &BTreeMap::from([("value".into(), relation.clone())]),
        )
        .expect("preimage digest");
        assert_ne!(small_digest, preimage_digest);
        assert!(
            runtime_inputs_digest(
                &generic,
                &backend,
                &BTreeMap::from([("value".into(), relation)]),
            )
            .is_err()
        );
        assert!(
            runtime_inputs_digest(&preimage, &backend, &BTreeMap::from([("value".into(), small)]),)
                .is_err()
        );
    }
}
