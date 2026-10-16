//! Private, bounded artifact generations for GPU sequential matrix-family carries.
//!
//! Physical family identifiers carry metadata only. Members have owners only while a wave
//! reads, computes, and exports them; the runtime resolves generation keys per invocation.

use crate::{
    artifact::ArtifactKey,
    gpu_execution_plan::{PhysicalValue, PhysicalValueId},
    gpu_physical_lowering::{PhysicalLoweringContext, value_id},
};
use mxx_ir_core::{
    IntExpr, ParamEnv,
    artifact::{ArtifactAvailability, ArtifactType, ManifestArtifact, ProductionId},
    graph::{FrozenGraphScopeId, Graph, GraphScope, NodeHandle},
    node::{IntBinaryOp, LoopInputMode, NodeKind, ParallelLoop},
    types::{ConcreteWireType, NodeId, Port, WireRef},
};
use num_bigint::BigInt;
use num_traits::ToPrimitive;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

/// Keep private artifact names path-safe and disjoint across scopes and physical templates.
fn carry_name(
    scope: &FrozenGraphScopeId,
    node: NodeId,
    port: usize,
    template: impl std::fmt::Debug,
) -> String {
    let digest = Sha256::digest(format!("{scope:?}/{}/{port}/{template:?}", node.0));
    let suffix = digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>();
    format!("runtime-carry-{suffix}")
}

/// Initial members either belong to an immutable production or parity zero of scratch.
#[derive(Clone, Debug)]
pub(crate) enum CarrySource {
    Artifact(ArtifactKey),
    Scratch,
}

/// One family with exactly two reusable generations under a fresh execution production.
#[derive(Clone, Debug)]
pub(crate) struct StreamedCarry {
    pub name: String,
    pub count: usize,
    pub element: ArtifactType,
    pub initial: CarrySource,
    /// Preserve a shared in-graph initial generation under its original scratch name.
    pub initial_name: Option<String>,
    pub availability: ArtifactAvailability,
    pub layout: Option<String>,
    pub updated: bool,
}

/// A view of a carry at a host iteration, or at a fixed final iteration outside its loop.
#[derive(Clone, Debug)]
pub(crate) struct StreamedFamily {
    pub carry: StreamedCarry,
    pub iteration: IntExpr,
}

/// A parallel output that writes one scratch generation, with no resident family allocation.
#[derive(Clone)]
pub(crate) struct StreamedExport {
    pub family: StreamedFamily,
    pub member_base: Option<(u32, IntExpr)>,
}

/// Scope-qualified export sites and metadata-only values shared across physical contexts.
#[derive(Default)]
pub(super) struct StreamingLowering {
    pub families: BTreeMap<PhysicalValueId, StreamedFamily>,
    pub exports: BTreeMap<(FrozenGraphScopeId, WireRef), StreamedExport>,
    pub member_overrides: BTreeMap<(FrozenGraphScopeId, WireRef), PhysicalValueId>,
    /// Body outputs of a root wave loop that its flattened artifact export names, by body scope
    /// and output wire: a wave loop producing one exports its members itself.
    pub nested_exports: BTreeMap<(FrozenGraphScopeId, WireRef), NestedExport>,
}

/// The flattened export of a root wave loop's family-valued output: the artifact names, the
/// members of each lane's family (the stride of a parent occurrence), and whether the wave loop
/// producing the current lane's family exports it.
pub(super) struct NestedExport {
    pub names: Vec<String>,
    pub stride: usize,
    pub exported: bool,
}

/// Execution-local key state copied by selected-import factories at their load sites.
pub(crate) struct CarryRuntimeState {
    pub scratch: ProductionId,
    pub indices: BTreeMap<u32, u64>,
}

/// A harmless plan-time key; the runtime substitutes its fresh scratch identity before I/O.
pub(super) fn scratch_key(name: &str) -> ArtifactKey {
    ArtifactKey {
        production: ProductionId {
            spec_hash: mxx_ir_core::artifact::SpecHash([0; 32]),
            execution_nonce: [0; 32],
        },
        name: name.to_owned(),
        index: None,
    }
}

/// Recognize `family.at(base + lane)` where `base` is a broadcast runtime scalar.
pub(super) fn runtime_base_reads(
    child: &GraphScope,
    spec: &ParallelLoop,
) -> BTreeMap<usize, (usize, Vec<WireRef>)> {
    let mut reads = BTreeMap::<usize, (usize, Vec<WireRef>)>::new();
    let lane = |wire: WireRef| {
        matches!(child.node(wire.node).map(NodeHandle::kind),
        Some(NodeKind::EvaluateInt(IntExpr::LoopIndex(slot))) if *slot == spec.index_slot)
    };
    for (position, node) in child.nodes().iter().enumerate() {
        if !matches!(node.kind(), NodeKind::FamilyGetDynamic) {
            continue;
        }
        let Some(arguments) = child.arguments(node) else { continue };
        let [family, index] = arguments.as_slice() else { continue };
        let Some(family_argument) = child.inputs().iter().position(|input| input == family) else {
            continue
        };
        if spec.input_modes[family_argument] != LoopInputMode::Broadcast {
            continue;
        }
        let Some(add) = child.node(index.node) else { continue };
        if !matches!(add.kind(), NodeKind::IntBinary(IntBinaryOp::Add)) {
            continue;
        }
        let Some(operands) = child.arguments(add) else { continue };
        let [left, right] = operands.as_slice() else { continue };
        let base = if lane(*left) {
            right
        } else if lane(*right) {
            left
        } else {
            continue
        };
        let Some(base_argument) = child.inputs().iter().position(|input| input == base) else {
            continue
        };
        if spec.input_modes[base_argument] != LoopInputMode::Broadcast {
            continue;
        }
        let entry = reads.entry(family_argument).or_insert((base_argument, Vec::new()));
        if entry.0 == base_argument {
            entry.1.push(WireRef { node: NodeId(position as u64), port: Port(0) });
        }
    }
    reads
}

/// Recognize the only reader of the root family `wire` (`n` members, each a family of `k`) as
/// its flattening `parallel(n k, |m| wire.at(m / k).at(m % k))` whose output is an artifact output
/// of the graph and nothing else. Returns the flattening loop and `k`: a wave loop producing
/// `wire` exports member `j` of instance `i` as member `i k + j` of the flattened artifact, so
/// neither family is ever resident.
pub(super) fn flattened_family_reader(
    graph: &Graph,
    wire: WireRef,
    ty: &ConcreteWireType,
    env: &ParamEnv,
) -> Option<(NodeId, usize)> {
    let ConcreteWireType::IndexedFamily { element, count } = ty else { return None };
    let ConcreteWireType::IndexedFamily { count: inner, .. } = element.as_ref() else {
        return None;
    };
    let (count, inner) = (usize::try_from(*count).ok()?, usize::try_from(*inner).ok()?);
    let root = graph.root_scope();
    let reads = |node: &NodeHandle, wire: WireRef| {
        root.arguments(node).is_some_and(|arguments| arguments.contains(&wire))
    };
    let mut readers = root.nodes().iter().enumerate().filter(|(_, node)| reads(node, wire));
    let (position, reader) = readers.next()?;
    if readers.next().is_some() {
        return None;
    }
    let reader_id = NodeId(position as u64);
    let NodeKind::ParallelLoop(spec) = reader.kind() else { return None };
    let reader_arguments = root.arguments(reader)?;
    if reader_arguments.first() != Some(&wire) ||
        spec.input_modes.iter().any(|mode| *mode != LoopInputMode::Broadcast) ||
        reader.output_types().len() != 1
    {
        return None;
    }
    // An integer constant of the root.
    let root_constant = |wire: &WireRef| match root.node(wire.node)?.kind() {
        NodeKind::EvaluateInt(expression) => expression.evaluate(env).ok()?.to_usize(),
        NodeKind::ConstantInt(value) => value.to_usize(),
        _ => None,
    };
    let flat = crate::gpu_physical_control::finite_loop_count(
        &FrozenGraphScopeId::Root,
        reader_id,
        reader.kind(),
        env,
    )
    .ok()?;
    if flat != (count as u64).checked_mul(inner as u64)? {
        return None;
    }
    let output = WireRef { node: reader_id, port: Port(0) };
    if !graph.outputs().values().any(|root| root.value == output && root.availability.is_some()) ||
        root.nodes().iter().any(|node| reads(node, output))
    {
        return None;
    }
    let child = graph.scope(&graph.child_scope_id(&FrozenGraphScopeId::Root, reader_id)?)?;
    let node = |wire: WireRef| child.node(wire.node).map(NodeHandle::kind);
    let arguments = |wire: WireRef| child.node(wire.node).and_then(|node| child.arguments(node));
    let lane = |wire: WireRef| {
        matches!(node(wire), Some(NodeKind::EvaluateInt(IntExpr::LoopIndex(slot)))
            if *slot == spec.index_slot)
    };
    // An integer constant of the body, or a broadcast root constant.
    let constant = |wire: WireRef| match node(wire)? {
        NodeKind::EvaluateInt(expression) => expression.evaluate(env).ok()?.to_usize(),
        NodeKind::ConstantInt(value) => value.to_usize(),
        NodeKind::Input { .. } => {
            let position = child.inputs().iter().position(|input| *input == wire)?;
            root_constant(reader_arguments.get(position).filter(|_| position > 0)?)
        }
        _ => None,
    };
    // `lane op k`.
    let by_inner = |wire: WireRef, op: IntBinaryOp| {
        node(wire) == Some(&NodeKind::IntBinary(op)) &&
            arguments(wire).is_some_and(|operands| {
                matches!(operands.as_slice(), [left, right]
                    if lane(*left) && constant(*right) == Some(inner))
            })
    };
    let [result] = child.outputs() else { return None };
    let [member, remainder] = arguments(*result)?[..] else { return None };
    let [family, quotient] = arguments(member)?[..] else { return None };
    (node(*result) == Some(&NodeKind::FamilyGetDynamic) &&
        node(member) == Some(&NodeKind::FamilyGetDynamic) &&
        child.inputs().first() == Some(&family) &&
        by_inner(quotient, IntBinaryOp::Divide) &&
        by_inner(remainder, IntBinaryOp::Remainder))
    .then_some((reader_id, inner))
}

impl StreamedFamily {
    /// Resolve a member without reading a payload or allocating any device memory.
    pub(crate) fn resolve(
        &self,
        member: usize,
        indices: &BTreeMap<u32, u64>,
        scratch: &ProductionId,
    ) -> Result<(ArtifactKey, ManifestArtifact, bool), String> {
        if member >= self.carry.count {
            return Err("GPU streamed carry member is outside its family".into());
        }
        let env = ParamEnv {
            loop_indices: indices
                .iter()
                .map(|(&slot, &value)| (slot, BigInt::from(value)))
                .collect(),
            ..ParamEnv::default()
        };
        let iteration = self
            .iteration
            .evaluate(&env)
            .map_err(|error| error.to_string())?
            .to_usize()
            .ok_or("GPU streamed carry iteration exceeds usize")?;
        if iteration == 0 || !self.carry.updated {
            if let CarrySource::Artifact(initial) = &self.carry.initial {
                let mut key = initial.clone();
                key.index = Some(member);
                return Ok((
                    key,
                    ManifestArtifact {
                        artifact_type: self.carry.element.clone(),
                        availability: self.carry.availability,
                        family_count: Some(self.carry.count as u128),
                        layout: self.carry.layout.clone(),
                    },
                    false,
                ));
            }
            return Ok((
                ArtifactKey {
                    production: scratch.clone(),
                    name: self.carry.initial_name.as_ref().unwrap_or(&self.carry.name).clone(),
                    index: Some(member),
                },
                self.carry.scratch_descriptor(),
                true,
            ));
        }
        let parity = if self.carry.updated { iteration % 2 } else { 0 };
        Ok((
            ArtifactKey {
                production: scratch.clone(),
                name: self.carry.name.clone(),
                index: Some(parity * self.carry.count + member),
            },
            self.carry.scratch_descriptor(),
            true,
        ))
    }
}

impl StreamedCarry {
    /// Scratch has room for both parities, and is never part of the public manifest.
    pub(crate) fn scratch_descriptor(&self) -> ManifestArtifact {
        ManifestArtifact {
            artifact_type: self.element.clone(),
            availability: ArtifactAvailability::Cached,
            family_count: Some(2 * self.count as u128),
            layout: None,
        }
    }
}

/// Whether this wire type must use bounded member streaming on the GPU.
pub(super) fn is_streamed_carry(ty: &ConcreteWireType) -> bool {
    matches!(ty, ConcreteWireType::IndexedFamily { element, .. }
        if matches!(element.as_ref(), ConcreteWireType::Matrix(_) | ConcreteWireType::SmallMatrix { .. }))
}

/// Bind a metadata-only family. Whole-family consumers have no physical owner to read.
pub(super) fn bind_family(
    ctx: &mut PhysicalLoweringContext<'_>,
    ty: ConcreteWireType,
    family: StreamedFamily,
) -> Result<PhysicalValueId, String> {
    let id = value_id(ctx.values.len())?;
    ctx.values.push(PhysicalValue {
        ty,
        encodings: Box::new([]),
        parts: Box::new([]),
        integer_ranges: BTreeMap::new(),
    });
    ctx.streaming.families.insert(id, family);
    Ok(id)
}

/// Give an in-graph initial parallel family a scratch export before it can become resident.
pub(super) fn initial_parallel_export(
    ctx: &PhysicalLoweringContext<'_>,
    graph: &Graph,
    scope: &FrozenGraphScopeId,
    wire: WireRef,
    ty: &ConcreteWireType,
) -> Result<Option<StreamedExport>, String> {
    if let Some(export) = ctx.streaming.exports.get(&(scope.clone(), wire)) {
        return Ok(Some(export.clone()));
    }
    if !is_streamed_carry(ty) {
        return Ok(None);
    }
    let parent = graph.scope(scope).ok_or("GPU streamed initial scope is missing")?;
    let consumed = parent.nodes().iter().any(|node| {
        let NodeKind::SequentialLoop(spec) = node.kind() else { return false };
        parent.arguments(node).is_some_and(|args| {
            args.iter().take(spec.carried_count).any(|argument| *argument == wire)
        })
    });
    if !consumed {
        return Ok(None);
    }
    if matches!(scope, FrozenGraphScopeId::Root) &&
        graph.outputs().values().any(|out| out.value == wire)
    {
        return Err("GPU streamed carry cannot be a root output".into());
    }
    let ConcreteWireType::IndexedFamily { element, count } = ty else { unreachable!() };
    let count = usize::try_from(*count)
        .ok()
        .filter(|count| count.checked_mul(2).is_some())
        .ok_or("GPU streamed carry cardinality exceeds usize")?;
    let family = StreamedFamily {
        carry: StreamedCarry {
            name: carry_name(scope, wire.node, wire.port.0 as usize, ctx.active_parallel_template),
            count,
            element: ArtifactType::from_wire_type(element)
                .ok_or("GPU carry element has no artifact type")?,
            initial: CarrySource::Scratch,
            initial_name: None,
            availability: ArtifactAvailability::Cached,
            layout: None,
            updated: true,
        },
        iteration: IntExpr::constant(0),
    };
    Ok(Some(StreamedExport { family, member_base: None }))
}

/// Classify a sequential carry and connect its input, top-level output, and scratch export.
pub(super) fn prepare_carry(
    ctx: &mut PhysicalLoweringContext<'_>,
    graph: &Graph,
    scope: &FrozenGraphScopeId,
    node: NodeId,
    argument: WireRef,
    child_id: &FrozenGraphScopeId,
    position: usize,
    slot: u32,
    ty: &ConcreteWireType,
) -> Result<Option<(PhysicalValueId, StreamedFamily)>, String> {
    if !is_streamed_carry(ty) {
        return Ok(None);
    }
    let child = graph.scope(child_id).ok_or("GPU streamed carry body is missing")?;
    let input = child.inputs()[position];
    let output = child.outputs()[position];
    let alias = output == input;
    if !alias &&
        !matches!(child.node(output.node).map(NodeHandle::kind), Some(NodeKind::ParallelLoop(_)))
    {
        return Err(
            "GPU streamed carry must be returned by a top-level parallel loop or unchanged".into()
        );
    }
    if matches!(scope, FrozenGraphScopeId::Root) &&
        graph
            .outputs()
            .values()
            .any(|out| out.value == WireRef { node, port: Port(position as u32) })
    {
        return Err("GPU streamed carry cannot be a root output".into());
    }
    let initial =
        ctx.wire_ids.get(&argument).and_then(|id| ctx.streaming.families.get(id)).cloned();
    let mut family = if let Some(family) = initial {
        if family.iteration != IntExpr::constant(0) {
            return Err("GPU streamed carry cannot reuse another loop's scratch generations".into());
        }
        let parent = graph.scope(scope).ok_or("GPU carry parent is missing")?;
        let uses = parent
            .nodes()
            .iter()
            .filter_map(|node| parent.arguments(node))
            .flatten()
            .filter(|wire| *wire == argument)
            .count();
        let mut family = family;
        if !alias && matches!(family.carry.initial, CarrySource::Scratch) && uses > 1 {
            family.carry.initial_name = Some(family.carry.name.clone());
            family.carry.name = carry_name(scope, node, position, ctx.active_parallel_template);
        }
        family
    } else {
        let parent = graph.scope(scope).ok_or("GPU carry parent is missing")?;
        let Some(NodeKind::Input { artifact: Some(artifact), .. }) =
            parent.node(argument.node).map(NodeHandle::kind)
        else {
            return Err(
                "GPU streamed carry initial value must be an artifact or top-level parallel family"
                    .into(),
            );
        };
        let descriptor = ctx
            .validated
            .scope(scope)
            .and_then(|scope| scope.artifact_inputs.get(&argument))
            .ok_or("GPU streamed carry initial artifact lacks a descriptor")?;
        let ConcreteWireType::IndexedFamily { element, count } = ty else { unreachable!() };
        let count = usize::try_from(*count)
            .ok()
            .filter(|count| count.checked_mul(2).is_some())
            .ok_or("GPU streamed carry cardinality exceeds usize")?;
        StreamedFamily {
            carry: StreamedCarry {
                name: carry_name(scope, node, position, ctx.active_parallel_template),
                count,
                element: ArtifactType::from_wire_type(element)
                    .ok_or("GPU carry element has no artifact type")?,
                initial: CarrySource::Artifact(ArtifactKey {
                    production: artifact.production_id.clone(),
                    name: artifact.artifact_name.clone(),
                    index: None,
                }),
                initial_name: None,
                availability: descriptor.availability,
                layout: descriptor.layout.clone(),
                updated: !alias,
            },
            iteration: IntExpr::constant(0),
        }
    };
    family.carry.updated = !alias;
    family.iteration = IntExpr::LoopIndex(slot);
    let id = bind_family(ctx, ty.clone(), family.clone())?;
    if !alias {
        let base = IntExpr::Mul(
            Box::new(IntExpr::Rem(
                Box::new(IntExpr::Add(Box::new(IntExpr::LoopIndex(slot)), Box::new(1.into()))),
                Box::new(2.into()),
            )),
            Box::new(IntExpr::constant(family.carry.count)),
        )
        .canonicalize();
        ctx.streaming.exports.insert(
            (child_id.clone(), output),
            StreamedExport { family: family.clone(), member_base: Some((slot, base)) },
        );
    }
    Ok(Some((id, family)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        artifact::MemoryArtifactStore,
        backend::{RuntimeValue, poly::cpu_backend, poly_gpu::gpu_backend_on},
        executor::{ExecutionConfig, execute, execute_in_session},
        gpu_execution_plan::PhysicalEncoding,
        gpu_runtime::GpuRuntime,
        poly::{
            Poly, PolyParams,
            dcrt::{
                gpu::{GpuDCRTPolyParams, detected_gpu_device_ids},
                params::DCRTPolyParams,
            },
        },
        session::SessionStore,
        transcript::SamplingMode,
    };
    use mxx_dsl::{DslContext, Family, Int, Mat, Ring, iterate, parallel};
    use mxx_ir_core::artifact::SpecHash;
    use std::collections::BTreeSet;

    /// Key resolution uses the immutable initial production, then alternating scratch parities.
    #[test]
    fn streamed_carry_resolves_parities_and_bounds() {
        let initial = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
        let scratch = ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] };
        let mut family = StreamedFamily {
            carry: StreamedCarry {
                name: "carry".into(),
                count: 8,
                element: ArtifactType::Int,
                initial: CarrySource::Artifact(ArtifactKey {
                    production: initial.clone(),
                    name: "initial".into(),
                    index: None,
                }),
                initial_name: None,
                availability: ArtifactAvailability::Cached,
                layout: None,
                updated: true,
            },
            iteration: IntExpr::LoopIndex(7),
        };
        for iteration in 0..5 {
            let (key, descriptor, staged) =
                family.resolve(3, &BTreeMap::from([(7, iteration)]), &scratch).unwrap();
            assert_eq!(staged, iteration != 0);
            assert_eq!(key.production, if staged { scratch.clone() } else { initial.clone() });
            assert_eq!(key.index, Some(if staged { (iteration as usize % 2) * 8 + 3 } else { 3 }));
            assert_eq!(descriptor.family_count, Some(if staged { 16 } else { 8 }));
        }
        assert!(family.resolve(8, &BTreeMap::from([(7, 1)]), &scratch).is_err());
        assert!(family.resolve(0, &BTreeMap::new(), &scratch).is_err());
        family.carry.updated = false;
        assert_eq!(
            family.resolve(2, &BTreeMap::from([(7, 99)]), &scratch).unwrap().0.production,
            initial
        );
        family.carry.initial = CarrySource::Scratch;
        assert_eq!(
            family.resolve(2, &BTreeMap::from([(7, 99)]), &scratch).unwrap().0.index,
            Some(2)
        );
    }

    /// The DSL's real capture shape keeps a broadcast runtime base identifiable before lowering.
    #[test]
    fn runtime_base_zip_matches_dsl_capture() {
        let ring = Ring::from_crt_moduli(vec![97.into()], 8);
        let context = DslContext::new("runtime-base-recognition");
        let family = ring.family_artifact_input(
            scratch_key("data").production,
            "data",
            16,
            (1, 1),
            ArtifactAvailability::Cached,
        );
        let base = context.input::<Int>("base", mxx_dsl::IntType).unwrap();
        let out =
            parallel(4, |lane| Ok(family.at(base.clone() + Int::evaluate(lane.expression()?))))
                .unwrap();
        let graph = context.output("out", out).unwrap().build().unwrap().graph;
        let matches = graph
            .scopes()
            .iter()
            .flat_map(|(scope_id, scope)| {
                scope.nodes().iter().enumerate().filter_map(|(position, node)| {
                    let NodeKind::ParallelLoop(spec) = node.kind() else { return None };
                    let child = graph
                        .scope(&graph.child_scope_id(scope_id, NodeId(position as u64)).unwrap())
                        .unwrap();
                    Some(runtime_base_reads(child, spec))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].len(), 1);
        assert_eq!(matches[0].values().next().unwrap().1.len(), 1);
    }

    /// Shared parameter setup keeps CPU and GPU tests on exactly the same CRT ring.
    fn parameters() -> (DCRTPolyParams, GpuDCRTPolyParams, Ring) {
        let dimension = crate::env::test_streamed_ring_dimension().try_into().unwrap();
        let cpu = DCRTPolyParams::new(dimension, 1, 20, 4, None, None);
        let gpu = GpuDCRTPolyParams::new(dimension, cpu.to_crt().0, 4, None);
        let ring = Ring::from_crt_moduli(
            cpu.to_crt().0.into_iter().map(IntExpr::from).collect(),
            dimension,
        );
        (cpu, gpu, ring)
    }

    /// Build the supported carry contract without returning a whole family at the root.
    fn carry_graph(
        ring: &Ring,
        members: usize,
        steps: usize,
        seed: usize,
    ) -> mxx_ir_core::ValidatedGraph {
        let initial =
            parallel(members, |member| Ok(ring.polynomial([member.expression()? + seed]))).unwrap();
        let carried = iterate(steps, initial, |iteration, state| {
            parallel(members, |member| {
                Ok(state.at(&member) +
                    ring.polynomial([iteration.expression()? + member.expression()? + 1]))
            })
        })
        .unwrap();
        member_outputs("streamed-carry", carried, members)
    }

    /// Only final member reads escape a private, scratch-backed family.
    fn member_outputs(
        name: &str,
        family: Family<Mat>,
        members: usize,
    ) -> mxx_ir_core::ValidatedGraph {
        DslContext::new(name)
            .output("first", family.at(0))
            .unwrap()
            .output("last", family.at(members - 1))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap()
    }

    /// Count plan-owned matrices, including import twins and constants, without double-counting
    /// views. This fixture has no external resident inputs; all these owners are plan scratch.
    fn matrix_scratch_bytes(plan: &crate::gpu_runtime::GpuExecutionPlan) -> u64 {
        let frame = plan.physical_frame_for_test();
        frame
            .owners
            .values()
            .chain(frame.waves.iter().flat_map(|wave| wave.owner_bindings.values()))
            .filter(|owner| {
                owner.physical().encodings.iter().any(|encoding| {
                    matches!(encoding, PhysicalEncoding::FullCoeff | PhysicalEncoding::FullEval)
                })
            })
            .flat_map(|owner| {
                owner
                    .storages()
                    .map(|(_, storage)| (storage.device, storage.address, storage.bytes))
            })
            .collect::<BTreeSet<_>>()
            .iter()
            .map(|(_, _, bytes)| *bytes)
            .sum()
    }

    /// Every replay matches CPU semantics and the fixed-width matrix allocation is
    /// cardinality-independent.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_sequential_loop_streams_matrix_family_carry() {
        let (cpu, gpu, ring) = parameters();
        let mut runtime =
            GpuRuntime::new(gpu_backend_on([gpu], detected_gpu_device_ids())).unwrap();
        let steps = crate::env::test_streamed_steps();
        let seed = usize::from(rand::random::<u16>());
        let mut bytes = Vec::new();
        let mut slots = Vec::new();
        for members in [8, 64] {
            let graph = carry_graph(&ring, members, steps, seed);
            let expected = execute(
                &graph,
                &mut cpu_backend([cpu.clone()]),
                BTreeMap::new(),
                &mut MemoryArtifactStore::default(),
                SamplingMode::Fresh,
                ExecutionConfig::default(),
            )
            .unwrap();
            let mut plan =
                runtime.plan_with_fixed_geometry_for_test(graph, &BTreeMap::new(), 1, 4).unwrap();
            bytes.push(matrix_scratch_bytes(&plan));
            slots.push(plan.physical_frame_for_test().slots.len());
            for _replay in 0..2 {
                let mut store = MemoryArtifactStore::default();
                let result = runtime
                    .execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, rand::random())
                    .unwrap();
                for name in ["first", "last"] {
                    let actual =
                        runtime.download_matrix_output(&result.output(name).unwrap()).unwrap();
                    let RuntimeValue::Matrix(expected) = &expected.outputs[name] else {
                        panic!("CPU matrix output")
                    };
                    assert_eq!(&actual, expected.as_cpu_full().unwrap());
                }
                assert!(store.test_peak_entries <= 2 * members);
                assert_eq!(
                    store.test_entry_count(),
                    0,
                    "private scratch must be removed after replay"
                );
            }
        }
        assert_eq!(bytes[0], bytes[1], "matrix scratch must depend on W, not family cardinality");
        assert_eq!(slots[0], slots[1], "publication slots must depend on W, not cardinality");
        println!("streamed matrix scratch: m=8 {} bytes; m=64 {} bytes", bytes[0], bytes[1]);
    }

    /// Immutable artifact initials and alias/zero-count carries survive multiple invocations.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_streamed_artifact_initial_and_aliases() {
        let (cpu, gpu, ring) = parameters();
        let members = 8;
        let producer = DslContext::new("artifact-carry-initial")
            .cached_output(
                "initial",
                parallel(members, |member| Ok(ring.polynomial([member.expression()? + 1])))
                    .unwrap(),
            )
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut store = MemoryArtifactStore::default();
        let produced = execute_in_session(
            &producer,
            &mut cpu_backend([cpu.clone()]),
            BTreeMap::new(),
            &mut store,
            rand::random(),
            ExecutionConfig::default(),
        )
        .unwrap();
        let production = produced.production_id.unwrap();
        let manifest = store.load_finalized_manifest(&production).unwrap();
        let mut runtime =
            GpuRuntime::new(gpu_backend_on([gpu], detected_gpu_device_ids())).unwrap();
        for steps in [0, 3] {
            for alias in [false, true] {
                let initial = ring.family_artifact_input(
                    production.clone(),
                    "initial",
                    members,
                    (1, 1),
                    ArtifactAvailability::Cached,
                );
                let out = iterate(steps, initial, |iteration, state| {
                    if alias {
                        Ok(state)
                    } else {
                        parallel(members, |member| {
                            Ok(state.at(&member) + ring.polynomial([iteration.expression()? + 1]))
                        })
                    }
                })
                .unwrap();
                let graph = DslContext::new("artifact-carry-consumer")
                    .output("out", out.at(members - 1))
                    .unwrap()
                    .build()
                    .unwrap()
                    .validate_with_manifests(
                        &ParamEnv::default(),
                        &BTreeMap::from([(production.clone(), manifest.clone())]),
                    )
                    .unwrap();
                let expected = execute(
                    &graph,
                    &mut cpu_backend([cpu.clone()]),
                    BTreeMap::new(),
                    &mut store,
                    SamplingMode::Fresh,
                    ExecutionConfig::default(),
                )
                .unwrap();
                let mut plan = runtime
                    .plan_with_fixed_geometry_for_test(graph, &BTreeMap::new(), 1, 2)
                    .unwrap();
                for _replay in 0..2 {
                    let result = runtime
                        .execute_with_artifacts(
                            &mut plan,
                            BTreeMap::new(),
                            &mut store,
                            rand::random(),
                        )
                        .unwrap();
                    let actual =
                        runtime.download_matrix_output(&result.output("out").unwrap()).unwrap();
                    let RuntimeValue::Matrix(expected) = &expected.outputs["out"] else {
                        panic!("CPU matrix output")
                    };
                    assert_eq!(&actual, expected.as_cpu_full().unwrap());
                    assert_eq!(
                        store.test_entry_count(),
                        members,
                        "immutable initial must remain; scratch must be gone"
                    );
                }
            }
        }
    }

    /// The body retains reusable W-lane waves inside its host-driven iteration boundary.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_parallel_waves_inside_host_driven_loop() {
        let (_, gpu, ring) = parameters();
        let seed = usize::from(rand::random::<u16>());
        let mut runtime =
            GpuRuntime::new(gpu_backend_on([gpu], detected_gpu_device_ids())).unwrap();
        let mut plan = runtime
            .plan_with_fixed_geometry_for_test(
                carry_graph(&ring, 8, 3, seed),
                &BTreeMap::new(),
                1,
                2,
            )
            .unwrap();
        let frame = plan.physical_frame_for_test();
        let body =
            frame.external_io_loops.iter().find(|body| !body.streamed_carries.is_empty()).unwrap();
        assert!(
            frame
                .waves
                .iter()
                .filter(|wave| wave.body_start >= body.body_start && wave.body_end <= body.body_end)
                .count() >=
                4
        );
        let result = runtime
            .execute_with_artifacts(
                &mut plan,
                BTreeMap::new(),
                &mut MemoryArtifactStore::default(),
                rand::random(),
            )
            .unwrap();
        let first = runtime.download_matrix_output(&result.output("first").unwrap()).unwrap();
        assert_eq!(
            first.entry(0, 0).to_bytes(),
            crate::poly::dcrt::poly::DCRTPoly::from_usize_to_constant(&parameters().0, seed + 6)
                .to_bytes()
        );
    }

    /// Shared initials remain immutable while dynamic polynomial coefficients follow both indices.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_streamed_shared_initial_and_polynomial_coefficients() {
        let (cpu, gpu, ring) = parameters();
        let members = crate::env::test_streamed_members();
        let steps = crate::env::test_streamed_steps();
        let seed = usize::from(rand::random::<u16>());
        let initial =
            parallel(members, |member| Ok(ring.polynomial([member.expression()? + seed]))).unwrap();
        let out = iterate(steps, initial.clone(), |iteration, state| {
            parallel(members, |member| {
                Ok(state.at(&member) +
                    initial.at(&member) +
                    ring.polynomial([
                        iteration.expression()? + member.expression()? + 1,
                        iteration.expression()? + 1,
                    ]))
            })
        })
        .unwrap();
        let graph = DslContext::new("shared-streamed-initial")
            .output("out", out.at(members - 1))
            .unwrap()
            .output("initial", initial.at(0))
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let expected = execute(
            &graph,
            &mut cpu_backend([cpu]),
            BTreeMap::new(),
            &mut MemoryArtifactStore::default(),
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let mut runtime =
            GpuRuntime::new(gpu_backend_on([gpu], detected_gpu_device_ids())).unwrap();
        let mut plan =
            runtime.plan_with_fixed_geometry_for_test(graph, &BTreeMap::new(), 1, 2).unwrap();
        for _replay in 0..2 {
            let mut store = MemoryArtifactStore::default();
            let result = runtime
                .execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, rand::random())
                .unwrap();
            for name in ["out", "initial"] {
                let actual = runtime.download_matrix_output(&result.output(name).unwrap()).unwrap();
                let RuntimeValue::Matrix(expected) = &expected.outputs[name] else {
                    panic!("CPU matrix output")
                };
                assert_eq!(&actual, expected.as_cpu_full().unwrap());
            }
            assert!(store.test_peak_entries <= 3 * members);
            assert_eq!(store.test_entry_count(), 0);
        }
    }

    /// Compact family members are streamed without materializing the family as full CRT matrices.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_streamed_small_matrix_carry() {
        let (cpu, gpu, ring) = parameters();
        let members = crate::env::test_streamed_members();
        let producer = DslContext::new("compact-carry-initial")
            .cached_output(
                "initial",
                parallel(members, |_| Ok(ring.small_gaussian((1, 1), 1, 8))).unwrap(),
            )
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut store = MemoryArtifactStore::default();
        let produced = execute_in_session(
            &producer,
            &mut cpu_backend([cpu.clone()]),
            BTreeMap::new(),
            &mut store,
            rand::random(),
            ExecutionConfig::default(),
        )
        .unwrap();
        let production = produced.production_id.unwrap();
        let manifest = store.load_finalized_manifest(&production).unwrap();
        let initial = ring.small_matrix_family_artifact_input(
            production.clone(),
            "initial",
            members,
            (1, 1),
            8,
            ArtifactAvailability::Cached,
        );
        let out = iterate(crate::env::test_streamed_steps(), initial, |_, state| {
            parallel(members, |member| Ok(state.at(&member)))
        })
        .unwrap();
        let graph = DslContext::new("compact-carry-consumer")
            .output("out", ring.identity(1).mul_small_rhs(out.at(members - 1)))
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production, manifest)]),
            )
            .unwrap();
        let expected = execute(
            &graph,
            &mut cpu_backend([cpu]),
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let mut runtime =
            GpuRuntime::new(gpu_backend_on([gpu], detected_gpu_device_ids())).unwrap();
        let mut plan =
            runtime.plan_with_fixed_geometry_for_test(graph, &BTreeMap::new(), 1, 2).unwrap();
        for _replay in 0..2 {
            let result = runtime
                .execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, rand::random())
                .unwrap();
            let actual = runtime.download_matrix_output(&result.output("out").unwrap()).unwrap();
            let RuntimeValue::Matrix(expected) = &expected.outputs["out"] else {
                panic!("CPU matrix output")
            };
            assert_eq!(&actual, expected.as_cpu_full().unwrap());
            assert_eq!(store.test_entry_count(), members);
        }
    }

    /// Artifact windows use a freshly published scalar base on every host iteration and replay.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_direct_zip_with_runtime_base() {
        let (cpu, gpu, ring) = parameters();
        let members = 8;
        let steps = 3;
        let producer = DslContext::new("runtime-base-data")
            .cached_output(
                "data",
                parallel(members * steps, |member| Ok(ring.polynomial([member.expression()? + 1])))
                    .unwrap(),
            )
            .unwrap()
            .build()
            .unwrap()
            .validate(&ParamEnv::default())
            .unwrap();
        let mut store = MemoryArtifactStore::default();
        let produced = execute_in_session(
            &producer,
            &mut cpu_backend([cpu.clone()]),
            BTreeMap::new(),
            &mut store,
            rand::random(),
            ExecutionConfig::default(),
        )
        .unwrap();
        let production = produced.production_id.unwrap();
        let manifest = store.load_finalized_manifest(&production).unwrap();
        let data = ring.family_artifact_input(
            production.clone(),
            "data",
            members * steps,
            (1, 1),
            ArtifactAvailability::Cached,
        );
        let initial = parallel(members, |_| Ok(ring.zero((1, 1)))).unwrap();
        let (out, _) = iterate(steps, (initial, Int::constant(0)), |_, (state, base)| {
            let next = parallel(members, |lane| {
                Ok(state.at(&lane) + data.at(base.clone() + Int::evaluate(lane.expression()?)))
            })?;
            Ok((next, base + Int::constant(members)))
        })
        .unwrap();
        let graph = DslContext::new("runtime-base-loop")
            .output("out", out.at(members - 1))
            .unwrap()
            .build()
            .unwrap()
            .validate_with_manifests(
                &ParamEnv::default(),
                &BTreeMap::from([(production, manifest)]),
            )
            .unwrap();
        let expected = execute(
            &graph,
            &mut cpu_backend([cpu]),
            BTreeMap::new(),
            &mut store,
            SamplingMode::Fresh,
            ExecutionConfig::default(),
        )
        .unwrap();
        let mut runtime =
            GpuRuntime::new(gpu_backend_on([gpu], detected_gpu_device_ids())).unwrap();
        let mut plan =
            runtime.plan_with_fixed_geometry_for_test(graph, &BTreeMap::new(), 1, 2).unwrap();
        assert!(
            plan.physical_frame_for_test()
                .import_templates
                .iter()
                .any(|import| import.base.is_some())
        );
        for _replay in 0..2 {
            let result = runtime
                .execute_with_artifacts(&mut plan, BTreeMap::new(), &mut store, rand::random())
                .unwrap();
            let actual = runtime.download_matrix_output(&result.output("out").unwrap()).unwrap();
            let RuntimeValue::Matrix(expected) = &expected.outputs["out"] else {
                panic!("CPU matrix output")
            };
            assert_eq!(&actual, expected.as_cpu_full().unwrap());
        }
    }

    /// A select consumes a whole family and must fail explicitly, rather than materializing it.
    #[test]
    #[ignore = "requires a CUDA GPU"]
    #[serial_test::serial(gpu_context)]
    fn test_gpu_streamed_carry_whole_family_consumer_is_planning_error() {
        let (_, gpu, ring) = parameters();
        let initial = parallel(8, |_| Ok(ring.zero((1, 1)))).unwrap();
        let out = iterate(3, initial, |_, state| {
            let selected = mxx_dsl::select(Int::constant(0), vec![state.clone(), state]).unwrap();
            parallel(8, |member| Ok(selected.at(&member)))
        })
        .unwrap();
        let mut runtime =
            GpuRuntime::new(gpu_backend_on([gpu], detected_gpu_device_ids())).unwrap();
        let error =
            match runtime.plan(member_outputs("whole-family-error", out, 8), &BTreeMap::new()) {
                Ok(_) => panic!("whole-family consumer was accepted"),
                Err(error) => error,
            };
        assert!(error.to_string().contains("whole-family"), "{error}");
    }
}
