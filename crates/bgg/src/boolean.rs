//! BGG+ handlers for parameterized dynamic Boolean circuit families.

use crate::{BggEncodingCompiler, BggEncodingWire, BggPublicKeyCompiler, BggPublicKeyWire};
use mxx_dsl::{
    DslContext, DslError, Family, Int, LoopIndex, Mat, Parallel, ParallelSelectTrace,
    ProofTraceTransport, Sequential, parallel_zip_bundle_trace,
};
use mxx_gadgets::circuit::{
    BooleanCircuitFamilyInputs, BooleanCircuitFamilyParams, BooleanLayerGate, GateSlot,
    evaluate_boolean_matrix_family,
};
use mxx_ir_core::{
    FreezeMap, FreezeResolveError, FrozenStructuralIntExpr, FrozenValueRef, Graph,
    StructuralValueRoute, ValueHandle, derive_structural_value_route,
    follows_structural_value_route,
};
use thiserror::Error;

#[derive(Clone, Debug)]
struct BggTraceEntrySpec {
    layer: Option<mxx_ir_core::IntExpr>,
    gate_slot: Option<mxx_ir_core::IntExpr>,
    candidate: Option<mxx_ir_core::IntExpr>,
    lane: BggTraceLane,
    subrole: BggTraceSubrole,
    role: BggTraceRole,
    handle: ValueHandle,
    operands: Vec<ValueHandle>,
    step: BggTraceStep,
    phase: BggTracePhase,
    operand_sources: Vec<BggOperandSource>,
}

type BggTraceSpec = BggTraceEntrySpec;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum BggTraceLane {
    Vector,
    PublicKey,
    Plaintext,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BggTraceSubrole {
    Decompose,
    MaterializeExact,
    Multiply,
    ApplyPreimage,
    Select,
    GateOutput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BggTracePhase {
    Layer,
    Epilogue,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum BggTraceStep {
    ZeroPlaintext,
    ZeroVector,
    ZeroPublicKey,
    NotPlaintext,
    NotVector,
    NotPublicKey,
    ProductPublicKeyDecompose,
    ProductPublicKeyMaterialize,
    ProductPublicKeyMultiply,
    ProductVectorDecompose,
    ProductVectorApplyPreimage,
    ProductVectorMultiply,
    ProductVectorOutput,
    ProductPlaintextOutput,
    SumPlaintext,
    SumVector,
    SumPublicKey,
    TwoProductPublicKey,
    TwoProductVector,
    TwoProductPlaintext,
    XorPlaintext,
    XorVector,
    XorPublicKey,
    CandidateVectorSelect,
    CandidatePublicKeySelect,
    CandidatePlaintextSelect,
    ActiveVectorSelect,
    ActivePublicKeySelect,
    ActivePlaintextSelect,
    LayerOutput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub enum BggTraceAnchor {
    One,
    Left,
    Right,
    Scalar,
    Selector,
    Active,
}

fn empty_route() -> StructuralValueRoute {
    StructuralValueRoute { exits: Vec::new(), enters: Vec::new() }
}

/// The source kind required by one operand position in the fixed BGG protocol.
/// Handles and child paths are intentionally absent: they are checked only
/// after freezing, against the executable graph.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExpectedOperandSource {
    External(BggTraceAnchor),
    Prior(BggTraceStep),
}

fn source_descriptor_matches(expected: ExpectedOperandSource, actual: &BggOperandSource) -> bool {
    match (expected, actual) {
        (
            ExpectedOperandSource::External(expected_role),
            BggOperandSource::External { role, .. },
        ) => expected_role == *role,
        (ExpectedOperandSource::Prior(expected_step), BggOperandSource::Prior { step, .. }) => {
            expected_step == *step
        }
        _ => false,
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BggOperandSource {
    External { role: BggTraceAnchor, handle: ValueHandle, path: StructuralValueRoute },
    Prior { step: BggTraceStep, handle: ValueHandle, path: StructuralValueRoute },
}

#[derive(Clone, Copy)]
enum BggOperandSourceRole {
    External(BggTraceAnchor),
    Prior(BggTraceStep),
}

fn operand_source(role: BggOperandSourceRole, handle: ValueHandle) -> BggOperandSource {
    match role {
        BggOperandSourceRole::External(role) => {
            BggOperandSource::External { role, handle, path: empty_route() }
        }
        BggOperandSourceRole::Prior(step) => {
            BggOperandSource::Prior { step, handle, path: empty_route() }
        }
    }
}

fn selection_sources(
    selection: &ParallelSelectTrace,
    roles: impl IntoIterator<Item = BggOperandSourceRole>,
) -> Vec<BggOperandSource> {
    selection
        .operands
        .iter()
        .cloned()
        .zip(roles)
        .map(|(handle, role)| operand_source(role, handle))
        .collect()
}

fn push_trace_spec_with_provenance(
    specs: &std::rc::Rc<std::cell::RefCell<Vec<BggTraceSpec>>>,
    lane: BggTraceLane,
    subrole: BggTraceSubrole,
    role: BggTraceRole,
    handle: ValueHandle,
    operands: Vec<ValueHandle>,
    step: BggTraceStep,
    operand_roles: [BggOperandSourceRole; 2],
) {
    let operand_sources = operands
        .iter()
        .cloned()
        .zip(operand_roles)
        .map(|(handle, role)| operand_source(role, handle))
        .collect();
    specs.borrow_mut().push(BggTraceSpec {
        layer: None,
        gate_slot: None,
        candidate: None,
        lane,
        subrole,
        role,
        handle,
        operands,
        step,
        phase: BggTracePhase::Layer,
        operand_sources,
    });
}

fn record_trace_fragment(
    specs: &std::rc::Rc<std::cell::RefCell<Vec<BggTraceSpec>>>,
    fragment: BggTraceFragment,
    layer: mxx_ir_core::IntExpr,
) -> ProofTraceTransport {
    let (transport, mut fragment_specs) = fragment.into_parts();
    for spec in &mut fragment_specs {
        spec.layer = Some(layer.clone());
    }
    specs.borrow_mut().extend(fragment_specs);
    transport
}

/// Records exactly one BGG trace site for the `Select` producer returned by
/// the DSL. The DSL trace owns the sealed producer identity and the exact
/// selector/branch argument order; no handle enumeration or index inference is
/// needed here.
fn record_selection_specs(
    specs: &std::rc::Rc<std::cell::RefCell<Vec<BggTraceSpec>>>,
    selection: &ParallelSelectTrace,
    lane: BggTraceLane,
    role: BggTraceRole,
    layer: mxx_ir_core::IntExpr,
    step: BggTraceStep,
    operand_sources: Vec<BggOperandSource>,
) {
    specs.borrow_mut().push(BggTraceSpec {
        layer: Some(layer),
        gate_slot: Some(selection.gate_slot.clone()),
        candidate: None,
        lane,
        subrole: BggTraceSubrole::Select,
        role,
        handle: selection.producer.clone(),
        operands: selection.operands.clone(),
        step,
        phase: BggTracePhase::Layer,
        operand_sources,
    });
}

/// The operation represented by one retained BGG producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BggTraceRole {
    Decomposition,
    MaterializePreimageExact,
    ApplyPreimage,
    MatrixMultiply,
    CandidateSelect,
    ActiveSelect,
    GateOutput,
}

pub const BGG_TRACE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub struct BooleanEncodingTraceEntry {
    pub layer: Option<mxx_ir_core::IntExpr>,
    pub gate_slot: Option<mxx_ir_core::IntExpr>,
    pub candidate: Option<mxx_ir_core::IntExpr>,
    pub lane: BggTraceLane,
    pub subrole: BggTraceSubrole,
    pub role: BggTraceRole,
    pub handle: ValueHandle,
    pub operands: Vec<ValueHandle>,
    pub step: BggTraceStep,
    pub phase: BggTracePhase,
    pub operand_sources: Vec<BggOperandSource>,
}

#[derive(Clone, Debug, Default)]
pub struct BggTraceFragment {
    entries: Vec<BooleanEncodingTraceEntry>,
    transport: ProofTraceTransport,
    specs: Vec<BggTraceSpec>,
}

impl BggTraceFragment {
    pub const SCHEMA_VERSION: u32 = BGG_TRACE_SCHEMA_VERSION;

    pub fn validate_schema(&self) -> Result<(), String> {
        if self.entries.len() != EXPECTED_LAYER_TRACE.len() + 1 {
            return Err(format!(
                "BGG trace manifest has {} entries; expected {}",
                self.entries.len(),
                EXPECTED_LAYER_TRACE.len() + 1
            ));
        }
        let layer = self.entries[0].layer.clone();
        let mut step_handles = std::collections::BTreeMap::new();
        for (index, entry) in self.entries.iter().enumerate() {
            if entry.step != EXPECTED_STEPS[index] {
                return Err(format!("BGG trace entry {index} has the wrong protocol step"));
            }
            let expected_phase = if index + 1 == EXPECTED_STEPS.len() {
                BggTracePhase::Epilogue
            } else {
                BggTracePhase::Layer
            };
            if entry.phase != expected_phase {
                return Err(format!("BGG trace entry {index} has the wrong protocol phase"));
            }
            if step_handles.insert(entry.step, entry.handle.clone()).is_some() {
                return Err(format!("BGG trace step {:?} is duplicated", entry.step));
            }
        }
        for (index, expected) in EXPECTED_LAYER_TRACE.iter().enumerate() {
            let entry = &self.entries[index];
            if entry.lane != expected.lane ||
                entry.subrole != expected.subrole ||
                entry.role != expected.role ||
                entry.operands.len() != expected.operands ||
                entry.layer.is_some() != expected.has_layer ||
                entry.gate_slot.is_some() != expected.has_gate_slot ||
                entry.candidate.is_some()
            {
                return Err(format!("BGG trace entry {index} does not match the protocol template"));
            }
            if entry.operand_sources.len() != entry.operands.len() {
                return Err(format!("BGG trace entry {index} has incomplete operand provenance"));
            }
            let expected_sources = expected_operand_sources(entry.step);
            if expected_sources.len() != entry.operand_sources.len() ||
                expected_sources
                    .iter()
                    .zip(&entry.operand_sources)
                    .any(|(expected, actual)| !source_descriptor_matches(*expected, actual))
            {
                return Err(format!(
                    "BGG trace entry {index} has the wrong operand source descriptor"
                ));
            }
            // Keep an explicit per-site anchor table.  An anchor may occur in
            // several operand positions (for example, `one - one`), but all
            // occurrences must have the same declared source descriptor.  The
            // concrete wire/path identity is checked after freezing.
            let mut anchors = std::collections::BTreeMap::new();
            for expected_source in expected_sources {
                if let ExpectedOperandSource::External(role) = *expected_source {
                    if let Some(previous) = anchors.insert(role, *expected_source) {
                        if previous != *expected_source {
                            return Err(format!(
                                "BGG trace entry {index} has a conflicting external anchor {:?}",
                                role
                            ));
                        }
                    }
                }
            }
            for (operand, source) in entry.operands.iter().zip(&entry.operand_sources) {
                match source {
                    BggOperandSource::External { role: _, handle, .. } => {
                        if operand != handle {
                            return Err(format!(
                                "BGG trace entry {index} has an external operand mismatch"
                            ));
                        }
                    }
                    BggOperandSource::Prior { step, handle, .. } => {
                        let _ = (operand, handle);
                        let Some(_previous) = step_handles.get(step) else {
                            return Err(format!(
                                "BGG trace entry {index} references an unknown prior step"
                            ));
                        };
                        // Transport merging may rebind an intermediate producer
                        // independently of the consumer operand.  The frozen
                        // validator below checks the exact wire/path relation.
                        if EXPECTED_STEPS
                            .iter()
                            .position(|candidate| candidate == step)
                            .is_some_and(|prior| prior >= index)
                        {
                            return Err(format!(
                                "BGG trace entry {index} references a non-prior step {:?}",
                                step
                            ));
                        }
                    }
                }
            }
            if expected.has_layer && entry.layer != layer {
                return Err(format!("BGG trace entry {index} has the wrong layer coordinate"));
            }
        }
        let final_entry = self.entries.last().expect("checked non-empty trace");
        if final_entry.lane != BggTraceLane::Vector ||
            final_entry.subrole != BggTraceSubrole::GateOutput ||
            final_entry.role != BggTraceRole::GateOutput ||
            !final_entry.operands.is_empty() ||
            !final_entry.operand_sources.is_empty() ||
            final_entry.layer.is_some() ||
            final_entry.gate_slot.is_some() ||
            final_entry.candidate.is_some()
        {
            return Err("BGG trace epilogue does not match the protocol template".to_owned());
        }
        Ok(())
    }

    fn from_transport(transport: ProofTraceTransport, specs: Vec<BggTraceEntrySpec>) -> Self {
        let mut entries = specs
            .clone()
            .into_iter()
            .map(|spec| BooleanEncodingTraceEntry {
                layer: spec.layer,
                gate_slot: spec.gate_slot,
                candidate: spec.candidate,
                lane: spec.lane,
                subrole: spec.subrole,
                role: spec.role,
                handle: transport.remap_handle(&spec.handle),
                operands: spec
                    .operands
                    .iter()
                    .map(|operand| transport.remap_handle(operand))
                    .collect(),
                step: spec.step,
                phase: spec.phase,
                operand_sources: spec
                    .operand_sources
                    .iter()
                    .map(|source| match source {
                        BggOperandSource::External { role, handle, path } => {
                            BggOperandSource::External {
                                role: *role,
                                handle: transport.remap_handle(handle),
                                path: path.clone(),
                            }
                        }
                        BggOperandSource::Prior { step, handle, path } => BggOperandSource::Prior {
                            step: *step,
                            handle: transport.remap_handle(handle),
                            path: path.clone(),
                        },
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        let step_handles: std::collections::BTreeMap<_, _> =
            entries.iter().map(|entry| (entry.step, entry.handle.clone())).collect();
        for entry in &mut entries {
            for source in &mut entry.operand_sources {
                if let BggOperandSource::Prior { step, handle, .. } = source {
                    if let Some(actual) = step_handles.get(step) {
                        *handle = actual.clone();
                    }
                }
            }
        }
        Self { entries, transport, specs }
    }

    fn merge(fragments: impl IntoIterator<Item = Self>) -> Self {
        let mut transports = Vec::new();
        let mut specs = Vec::new();
        for fragment in fragments {
            transports.push(fragment.transport);
            specs.extend(fragment.specs);
        }
        let mut merged = Self::from_transport(ProofTraceTransport::merge(transports), specs);
        let step_handles: std::collections::BTreeMap<_, _> =
            merged.entries.iter().map(|entry| (entry.step, entry.handle.clone())).collect();
        for entry in &mut merged.entries {
            for source in &mut entry.operand_sources {
                if let BggOperandSource::Prior { step, handle, .. } = source {
                    if let Some(actual) = step_handles.get(step) {
                        *handle = actual.clone();
                    }
                }
            }
        }
        merged
    }

    fn into_parts(self) -> (ProofTraceTransport, Vec<BggTraceSpec>) {
        (self.transport, self.specs)
    }

    pub fn entries(&self) -> &[BooleanEncodingTraceEntry] {
        &self.entries
    }

    pub fn into_retained_values(self) -> Vec<ValueHandle> {
        let mut values = self.transport.into_retained_values();
        for entry in &self.entries {
            values.push(entry.handle.clone());
            values.extend(entry.operands.iter().cloned());
            values.extend(entry.operand_sources.iter().filter_map(|source| match source {
                BggOperandSource::External { handle, .. } |
                BggOperandSource::Prior { handle, .. } => Some(handle.clone()),
            }));
        }
        values
    }

    pub fn resolve(
        &self,
        map: &FreezeMap,
    ) -> Result<Vec<FrozenBooleanEncodingTraceEntry>, FreezeResolveError> {
        self.resolve_with_graph(map, None)
    }

    pub fn resolve_with_graph(
        &self,
        map: &FreezeMap,
        graph: Option<&Graph>,
    ) -> Result<Vec<FrozenBooleanEncodingTraceEntry>, FreezeResolveError> {
        let mut prior_handles = std::collections::BTreeMap::new();
        self.entries
            .iter()
            .map(|entry| {
                let handle = map.resolve_typed(&entry.handle)?;
                let operands = entry
                    .operands
                    .iter()
                    .map(|operand| map.resolve_typed(operand))
                    .collect::<Result<_, _>>()?;
                let operand_sources = entry
                    .operand_sources
                    .iter()
                    .zip(&operands)
                    .map(|(source, operand)| -> Result<_, FreezeResolveError> {
                        Ok(match source {
                            BggOperandSource::External { role, handle, path } => {
                                let source_ref = map.resolve_typed(handle)?;
                                let path = graph
                                    .map(|graph| {
                                        derive_structural_value_route(graph, &source_ref, operand)
                                    })
                                    .transpose()
                                    .map_err(|_| FreezeResolveError::Missing)?
                                    .unwrap_or_else(|| path.clone());
                                FrozenBggOperandSource::External {
                                    role: *role,
                                    handle: source_ref,
                                    path,
                                }
                            }
                            BggOperandSource::Prior { step, handle, path } => {
                                let declared = map.resolve_typed(handle)?;
                                let source_ref =
                                    prior_handles.get(step).ok_or(FreezeResolveError::Missing)?;
                                if declared != *source_ref {
                                    return Err(FreezeResolveError::Missing);
                                }
                                let path = graph
                                    .map(|graph| {
                                        derive_structural_value_route(graph, source_ref, operand)
                                    })
                                    .transpose()
                                    .map_err(|_| FreezeResolveError::Missing)?
                                    .unwrap_or_else(|| path.clone());
                                FrozenBggOperandSource::Prior { step: *step, path }
                            }
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let frozen = FrozenBooleanEncodingTraceEntry {
                    layer: entry.layer.clone().map(|expr| map.freeze_structural_expr(expr)),
                    gate_slot: entry.gate_slot.clone().map(|expr| map.freeze_structural_expr(expr)),
                    candidate: entry.candidate.clone().map(|expr| map.freeze_structural_expr(expr)),
                    lane: entry.lane,
                    subrole: entry.subrole,
                    role: entry.role,
                    handle,
                    operands,
                    step: entry.step,
                    phase: entry.phase,
                    operand_sources,
                };
                prior_handles.insert(entry.step, frozen.handle.clone());
                Ok(frozen)
            })
            .collect()
    }

    /// Validate frozen provenance against the executable graph.  Empty paths
    /// are accepted only for exact wire equality by the generic IR checker;
    /// non-empty paths are checked hop-by-hop.
    pub fn validate_frozen_paths(
        entries: &[FrozenBooleanEncodingTraceEntry],
        graph: &Graph,
    ) -> Result<(), String> {
        if entries.len() != EXPECTED_STEPS.len() {
            return Err(format!(
                "frozen BGG trace has {} entries; expected {}",
                entries.len(),
                EXPECTED_STEPS.len()
            ));
        }
        let mut steps = std::collections::BTreeMap::new();
        for (index, entry) in entries.iter().enumerate() {
            if entry.step !=
                EXPECTED_STEPS
                    .get(index)
                    .copied()
                    .ok_or_else(|| "frozen BGG trace has an unexpected entry".to_owned())?
            {
                return Err(format!("frozen BGG entry {index} has the wrong protocol step"));
            }
            let expected_sources = expected_operand_sources(entry.step);
            if expected_sources.len() != entry.operand_sources.len() ||
                expected_sources.iter().zip(&entry.operand_sources).any(
                    |(expected, actual)| match (expected, actual) {
                        (
                            ExpectedOperandSource::External(role),
                            FrozenBggOperandSource::External { role: actual, .. },
                        ) => role != actual,
                        (
                            ExpectedOperandSource::Prior(step),
                            FrozenBggOperandSource::Prior { step: actual, .. },
                        ) => step != actual,
                        _ => true,
                    },
                )
            {
                return Err(format!(
                    "frozen BGG entry {index} has the wrong operand source descriptor"
                ));
            }
            for (operand, source) in entry.operands.iter().zip(&entry.operand_sources) {
                let start = match source {
                    FrozenBggOperandSource::External { handle, .. } => handle,
                    FrozenBggOperandSource::Prior { step, .. } => {
                        steps.get(step).ok_or_else(|| {
                            format!(
                                "entry {index} references a future or missing prior step {:?}",
                                step
                            )
                        })?
                    }
                };
                let path = match source {
                    FrozenBggOperandSource::External { path, .. } |
                    FrozenBggOperandSource::Prior { path, .. } => path,
                };
                let ok = follows_structural_value_route(
                    graph,
                    start.reference(),
                    path,
                    operand.reference(),
                )
                .map_err(|error| format!("entry {index} has invalid operand path: {error}"))?;
                if !ok {
                    return Err(format!("entry {index} operand path does not reach operand"));
                }
            }
            if steps.insert(entry.step, entry.handle.clone()).is_some() {
                return Err(format!("duplicate trace step at entry {index}"));
            }
        }
        Ok(())
    }
}

/// Compatibility name for the public boolean-layer trace API. The fragment
/// itself remains BGG-owned; DSL only transports opaque handles.
pub type BooleanEncodingTrace = BggTraceFragment;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FrozenBooleanEncodingTraceEntry {
    pub layer: Option<FrozenStructuralIntExpr>,
    pub gate_slot: Option<FrozenStructuralIntExpr>,
    pub candidate: Option<FrozenStructuralIntExpr>,
    pub lane: BggTraceLane,
    pub subrole: BggTraceSubrole,
    pub role: BggTraceRole,
    pub handle: FrozenValueRef,
    pub operands: Vec<FrozenValueRef>,
    pub step: BggTraceStep,
    pub phase: BggTracePhase,
    pub operand_sources: Vec<FrozenBggOperandSource>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum FrozenBggOperandSource {
    External { role: BggTraceAnchor, handle: FrozenValueRef, path: StructuralValueRoute },
    Prior { step: BggTraceStep, path: StructuralValueRoute },
}

#[derive(Clone, Copy)]
struct ExpectedTraceEntry {
    lane: BggTraceLane,
    subrole: BggTraceSubrole,
    role: BggTraceRole,
    operands: usize,
    has_layer: bool,
    has_gate_slot: bool,
}

const EXPECTED_LAYER_TRACE: [ExpectedTraceEntry; 29] = [
    ExpectedTraceEntry {
        lane: BggTraceLane::Plaintext,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::PublicKey,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Plaintext,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::PublicKey,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::PublicKey,
        subrole: BggTraceSubrole::Decompose,
        role: BggTraceRole::Decomposition,
        operands: 1,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::PublicKey,
        subrole: BggTraceSubrole::MaterializeExact,
        role: BggTraceRole::MaterializePreimageExact,
        operands: 1,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::PublicKey,
        subrole: BggTraceSubrole::Multiply,
        role: BggTraceRole::MatrixMultiply,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::Decompose,
        role: BggTraceRole::Decomposition,
        operands: 1,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::ApplyPreimage,
        role: BggTraceRole::ApplyPreimage,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::Multiply,
        role: BggTraceRole::MatrixMultiply,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Plaintext,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Plaintext,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::PublicKey,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::PublicKey,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Plaintext,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Plaintext,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::PublicKey,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        operands: 2,
        has_layer: true,
        has_gate_slot: false,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::Select,
        role: BggTraceRole::CandidateSelect,
        operands: 7,
        has_layer: true,
        has_gate_slot: true,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::PublicKey,
        subrole: BggTraceSubrole::Select,
        role: BggTraceRole::CandidateSelect,
        operands: 7,
        has_layer: true,
        has_gate_slot: true,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Plaintext,
        subrole: BggTraceSubrole::Select,
        role: BggTraceRole::CandidateSelect,
        operands: 7,
        has_layer: true,
        has_gate_slot: true,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::Select,
        role: BggTraceRole::ActiveSelect,
        operands: 3,
        has_layer: true,
        has_gate_slot: true,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::PublicKey,
        subrole: BggTraceSubrole::Select,
        role: BggTraceRole::ActiveSelect,
        operands: 3,
        has_layer: true,
        has_gate_slot: true,
    },
    ExpectedTraceEntry {
        lane: BggTraceLane::Plaintext,
        subrole: BggTraceSubrole::Select,
        role: BggTraceRole::ActiveSelect,
        operands: 3,
        has_layer: true,
        has_gate_slot: true,
    },
];

const EXPECTED_STEPS: [BggTraceStep; 30] = [
    BggTraceStep::ZeroPlaintext,
    BggTraceStep::ZeroVector,
    BggTraceStep::ZeroPublicKey,
    BggTraceStep::NotPlaintext,
    BggTraceStep::NotVector,
    BggTraceStep::NotPublicKey,
    BggTraceStep::ProductPublicKeyDecompose,
    BggTraceStep::ProductPublicKeyMaterialize,
    BggTraceStep::ProductPublicKeyMultiply,
    BggTraceStep::ProductVectorDecompose,
    BggTraceStep::ProductVectorApplyPreimage,
    BggTraceStep::ProductVectorMultiply,
    BggTraceStep::ProductVectorOutput,
    BggTraceStep::ProductPlaintextOutput,
    BggTraceStep::SumPlaintext,
    BggTraceStep::SumVector,
    BggTraceStep::SumPublicKey,
    BggTraceStep::TwoProductPublicKey,
    BggTraceStep::TwoProductVector,
    BggTraceStep::TwoProductPlaintext,
    BggTraceStep::XorPlaintext,
    BggTraceStep::XorVector,
    BggTraceStep::XorPublicKey,
    BggTraceStep::CandidateVectorSelect,
    BggTraceStep::CandidatePublicKeySelect,
    BggTraceStep::CandidatePlaintextSelect,
    BggTraceStep::ActiveVectorSelect,
    BggTraceStep::ActivePublicKeySelect,
    BggTraceStep::ActivePlaintextSelect,
    BggTraceStep::LayerOutput,
];

const SOURCE_ONE_ONE: [ExpectedOperandSource; 2] = [
    ExpectedOperandSource::External(BggTraceAnchor::One),
    ExpectedOperandSource::External(BggTraceAnchor::One),
];
const SOURCE_ONE_LEFT: [ExpectedOperandSource; 2] = [
    ExpectedOperandSource::External(BggTraceAnchor::One),
    ExpectedOperandSource::External(BggTraceAnchor::Left),
];
const SOURCE_LEFT_RIGHT: [ExpectedOperandSource; 2] = [
    ExpectedOperandSource::External(BggTraceAnchor::Left),
    ExpectedOperandSource::External(BggTraceAnchor::Right),
];
const SOURCE_LEFT_PRIOR_MATERIALIZE: [ExpectedOperandSource; 2] = [
    ExpectedOperandSource::External(BggTraceAnchor::Left),
    ExpectedOperandSource::Prior(BggTraceStep::ProductPublicKeyMaterialize),
];
const SOURCE_LEFT_PRIOR_DECOMPOSE: [ExpectedOperandSource; 2] = [
    ExpectedOperandSource::External(BggTraceAnchor::Left),
    ExpectedOperandSource::Prior(BggTraceStep::ProductVectorDecompose),
];
const SOURCE_PRIOR_SUM_TWO: [ExpectedOperandSource; 2] = [
    ExpectedOperandSource::Prior(BggTraceStep::SumVector),
    ExpectedOperandSource::Prior(BggTraceStep::TwoProductVector),
];
const SOURCE_PRIOR_SUM_TWO_KEY: [ExpectedOperandSource; 2] = [
    ExpectedOperandSource::Prior(BggTraceStep::SumPublicKey),
    ExpectedOperandSource::Prior(BggTraceStep::TwoProductPublicKey),
];
const SOURCE_PRIOR_SUM_TWO_PLAIN: [ExpectedOperandSource; 2] = [
    ExpectedOperandSource::Prior(BggTraceStep::SumPlaintext),
    ExpectedOperandSource::Prior(BggTraceStep::TwoProductPlaintext),
];
const SOURCE_LEFT_SCALAR: [ExpectedOperandSource; 2] = [
    ExpectedOperandSource::External(BggTraceAnchor::Left),
    ExpectedOperandSource::External(BggTraceAnchor::Scalar),
];
const SOURCE_EXTERNAL_RIGHT: [ExpectedOperandSource; 1] =
    [ExpectedOperandSource::External(BggTraceAnchor::Right)];
const SOURCE_PRIOR_DECOMPOSE: [ExpectedOperandSource; 1] =
    [ExpectedOperandSource::Prior(BggTraceStep::ProductPublicKeyDecompose)];
const SOURCE_ACTIVE_ZERO_CANDIDATE: [ExpectedOperandSource; 3] = [
    ExpectedOperandSource::External(BggTraceAnchor::Active),
    ExpectedOperandSource::Prior(BggTraceStep::ZeroVector),
    ExpectedOperandSource::Prior(BggTraceStep::CandidateVectorSelect),
];
const SOURCE_ACTIVE_ZERO_CANDIDATE_KEY: [ExpectedOperandSource; 3] = [
    ExpectedOperandSource::External(BggTraceAnchor::Active),
    ExpectedOperandSource::Prior(BggTraceStep::ZeroPublicKey),
    ExpectedOperandSource::Prior(BggTraceStep::CandidatePublicKeySelect),
];
const SOURCE_ACTIVE_ZERO_CANDIDATE_PLAIN: [ExpectedOperandSource; 3] = [
    ExpectedOperandSource::External(BggTraceAnchor::Active),
    ExpectedOperandSource::Prior(BggTraceStep::ZeroPlaintext),
    ExpectedOperandSource::Prior(BggTraceStep::CandidatePlaintextSelect),
];
const SOURCE_SELECTOR_CANDIDATES_VECTOR: [ExpectedOperandSource; 7] = [
    ExpectedOperandSource::External(BggTraceAnchor::Selector),
    ExpectedOperandSource::Prior(BggTraceStep::ZeroVector),
    ExpectedOperandSource::External(BggTraceAnchor::One),
    ExpectedOperandSource::External(BggTraceAnchor::Left),
    ExpectedOperandSource::Prior(BggTraceStep::NotVector),
    ExpectedOperandSource::Prior(BggTraceStep::ProductVectorOutput),
    ExpectedOperandSource::Prior(BggTraceStep::XorVector),
];
const SOURCE_SELECTOR_CANDIDATES_KEY: [ExpectedOperandSource; 7] = [
    ExpectedOperandSource::External(BggTraceAnchor::Selector),
    ExpectedOperandSource::Prior(BggTraceStep::ZeroPublicKey),
    ExpectedOperandSource::External(BggTraceAnchor::One),
    ExpectedOperandSource::External(BggTraceAnchor::Left),
    ExpectedOperandSource::Prior(BggTraceStep::NotPublicKey),
    ExpectedOperandSource::Prior(BggTraceStep::ProductPublicKeyMultiply),
    ExpectedOperandSource::Prior(BggTraceStep::XorPublicKey),
];
const SOURCE_SELECTOR_CANDIDATES_PLAIN: [ExpectedOperandSource; 7] = [
    ExpectedOperandSource::External(BggTraceAnchor::Selector),
    ExpectedOperandSource::Prior(BggTraceStep::ZeroPlaintext),
    ExpectedOperandSource::External(BggTraceAnchor::One),
    ExpectedOperandSource::External(BggTraceAnchor::Left),
    ExpectedOperandSource::Prior(BggTraceStep::NotPlaintext),
    ExpectedOperandSource::Prior(BggTraceStep::ProductPlaintextOutput),
    ExpectedOperandSource::Prior(BggTraceStep::XorPlaintext),
];

fn expected_operand_sources(step: BggTraceStep) -> &'static [ExpectedOperandSource] {
    match step {
        BggTraceStep::ZeroPlaintext | BggTraceStep::ZeroVector | BggTraceStep::ZeroPublicKey => {
            &SOURCE_ONE_ONE
        }
        BggTraceStep::NotPlaintext | BggTraceStep::NotVector | BggTraceStep::NotPublicKey => {
            &SOURCE_ONE_LEFT
        }
        BggTraceStep::ProductPublicKeyDecompose | BggTraceStep::ProductVectorDecompose => {
            &SOURCE_EXTERNAL_RIGHT
        }
        BggTraceStep::ProductPublicKeyMaterialize => &SOURCE_PRIOR_DECOMPOSE,
        BggTraceStep::ProductPublicKeyMultiply => &SOURCE_LEFT_PRIOR_MATERIALIZE,
        BggTraceStep::ProductVectorApplyPreimage => &SOURCE_LEFT_PRIOR_DECOMPOSE,
        BggTraceStep::ProductVectorMultiply |
        BggTraceStep::ProductVectorOutput |
        BggTraceStep::ProductPlaintextOutput |
        BggTraceStep::SumPlaintext |
        BggTraceStep::SumVector |
        BggTraceStep::SumPublicKey => &SOURCE_LEFT_RIGHT,
        BggTraceStep::TwoProductPublicKey |
        BggTraceStep::TwoProductVector |
        BggTraceStep::TwoProductPlaintext => &SOURCE_LEFT_SCALAR,
        BggTraceStep::XorPlaintext => &SOURCE_PRIOR_SUM_TWO_PLAIN,
        BggTraceStep::XorVector => &SOURCE_PRIOR_SUM_TWO,
        BggTraceStep::XorPublicKey => &SOURCE_PRIOR_SUM_TWO_KEY,
        BggTraceStep::CandidateVectorSelect => &SOURCE_SELECTOR_CANDIDATES_VECTOR,
        BggTraceStep::CandidatePublicKeySelect => &SOURCE_SELECTOR_CANDIDATES_KEY,
        BggTraceStep::CandidatePlaintextSelect => &SOURCE_SELECTOR_CANDIDATES_PLAIN,
        BggTraceStep::ActiveVectorSelect => &SOURCE_ACTIVE_ZERO_CANDIDATE,
        BggTraceStep::ActivePublicKeySelect => &SOURCE_ACTIVE_ZERO_CANDIDATE_KEY,
        BggTraceStep::ActivePlaintextSelect => &SOURCE_ACTIVE_ZERO_CANDIDATE_PLAIN,
        BggTraceStep::LayerOutput => &[],
    }
}

#[derive(Clone)]
pub struct BggPublicKeyFamily {
    pub matrices: Family<Mat>,
    pub reveal_plaintext: bool,
}

#[derive(Clone)]
pub struct BggEncodingFamily {
    pub vectors: Family<Mat>,
    pub public_keys: BggPublicKeyFamily,
    pub plaintexts: Family<Mat>,
}

#[derive(Debug, Error)]
pub enum DynamicBooleanBggError {
    #[error(transparent)]
    Dsl(#[from] DslError),
    #[error("dynamic Boolean BGG evaluation requires revealed plaintexts for every input")]
    PlaintextRequired,
    #[error("dynamic Boolean BGG input component families have different counts")]
    FamilyLayout,
    #[error("invalid BGG trace schema: {0}")]
    TraceSchema(String),
}

impl BggPublicKeyFamily {
    pub fn pack(values: Vec<BggPublicKeyWire>) -> Result<Self, DynamicBooleanBggError> {
        let reveal_plaintext = values.iter().all(|value| value.reveal_plaintext);
        Ok(Self {
            matrices: Family::pack(values.into_iter().map(|value| value.matrix).collect())?,
            reveal_plaintext,
        })
    }
}

impl BggEncodingFamily {
    pub fn pack(values: Vec<BggEncodingWire>) -> Result<Self, DynamicBooleanBggError> {
        if values.iter().any(|value| !value.pubkey.reveal_plaintext || value.plaintext.is_none()) {
            return Err(DynamicBooleanBggError::PlaintextRequired);
        }
        let vectors = Family::pack(values.iter().map(|value| value.vector.clone()).collect())?;
        let public_keys =
            BggPublicKeyFamily::pack(values.iter().map(|value| value.pubkey.clone()).collect())?;
        let plaintexts = Family::pack(
            values.into_iter().map(|value| value.plaintext.expect("checked above")).collect(),
        )?;
        Ok(Self { vectors, public_keys, plaintexts })
    }

    fn validate(&self) -> Result<(), DynamicBooleanBggError> {
        if self.vectors.count() != self.public_keys.matrices.count() ||
            self.vectors.count() != self.plaintexts.count() ||
            !self.public_keys.reveal_plaintext
        {
            return Err(DynamicBooleanBggError::FamilyLayout);
        }
        Ok(())
    }

    fn gather(self, indices: Family<mxx_dsl::Int>) -> Result<Self, DynamicBooleanBggError> {
        self.validate()?;
        let vectors = self.vectors.parallel_gather(indices.clone())?;
        let public_keys = self.public_keys.matrices.parallel_gather(indices.clone())?;
        let plaintexts = self.plaintexts.parallel_gather(indices)?;
        Ok(Self {
            vectors,
            public_keys: BggPublicKeyFamily {
                matrices: public_keys,
                reveal_plaintext: self.public_keys.reveal_plaintext,
            },
            plaintexts,
        })
    }
}

pub fn evaluate_boolean_public_key_layers(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: BggPublicKeyFamily,
    one: BggPublicKeyWire,
    compiler: BggPublicKeyCompiler,
) -> Result<BggPublicKeyFamily, DynamicBooleanBggError> {
    if !preceding.reveal_plaintext || !one.reveal_plaintext {
        return Err(DynamicBooleanBggError::PlaintextRequired);
    }
    let matrices = evaluate_boolean_matrix_family(
        context,
        params,
        circuit,
        preceding.matrices,
        PublicKeyBooleanGate { compiler, one },
    )?;
    Ok(BggPublicKeyFamily { matrices, reveal_plaintext: true })
}

pub fn evaluate_boolean_encoding_layers(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: BggEncodingFamily,
    one: BggEncodingWire,
    compiler: BggEncodingCompiler,
) -> Result<BggEncodingFamily, DynamicBooleanBggError> {
    let (output, _) = evaluate_boolean_encoding_layers_with_trace(
        context, params, circuit, preceding, one, compiler,
    )?;
    Ok(output)
}

/// Proof-only variant of boolean layer evaluation. The executable encoding
/// graph is identical to [`evaluate_boolean_encoding_layers`]; the transport
/// exposes the final carried family handle for structural certificates.
pub fn evaluate_boolean_encoding_layers_with_trace(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    circuit: BooleanCircuitFamilyInputs,
    preceding: BggEncodingFamily,
    one: BggEncodingWire,
    compiler: BggEncodingCompiler,
) -> Result<(BggEncodingFamily, BooleanEncodingTrace), DynamicBooleanBggError> {
    let trace_specs = std::rc::Rc::new(std::cell::RefCell::new(Vec::<BggTraceSpec>::new()));
    preceding.validate()?;
    if !one.pubkey.reveal_plaintext || one.plaintext.is_none() {
        return Err(DynamicBooleanBggError::PlaintextRequired);
    }
    let BooleanCircuitFamilyInputs {
        active_gate_counts,
        gate_kinds,
        left_sources,
        right_sources,
        output_sources: _,
    } = circuit;
    let invariants = (active_gate_counts, (gate_kinds, (left_sources, right_sources)));
    let initial = (preceding.vectors, preceding.public_keys.matrices, preceding.plaintexts);
    let ((vectors, public_keys, plaintexts), layer_trace) = Sequential::range(params.depth.clone())
        .scan_with_trace_handles(
            initial,
            invariants,
            |layer,
             (vectors, public_keys, plaintexts),
             (active_gate_counts, (gate_kinds, (left_sources, right_sources)))| {
                let trace_specs = trace_specs.clone();
                let preceding = BggEncodingFamily {
                    vectors,
                    public_keys: BggPublicKeyFamily {
                        matrices: public_keys,
                        reveal_plaintext: true,
                    },
                    plaintexts,
                };
                let active_count = active_gate_counts.get(layer.as_int());
                let (_, kinds, left_indices, right_indices) = layer_metadata(
                    context,
                    params,
                    &layer,
                    gate_kinds,
                    left_sources,
                    right_sources,
                )?;
                let left = scan_result(preceding.clone().gather(left_indices))?;
                let right = scan_result(preceding.gather(right_indices))?;
                let one_family = scan_result(repeated_encoding(params, &one))?;
                let (zero, zero_fragment) = scan_result(encoding_binary_with_trace(
                    &compiler,
                    &one_family,
                    &one_family,
                    EncodingOp::Sub,
                    [
                        BggTraceStep::ZeroVector,
                        BggTraceStep::ZeroPublicKey,
                        BggTraceStep::ZeroPlaintext,
                    ],
                    [[
                        BggOperandSourceRole::External(BggTraceAnchor::One),
                        BggOperandSourceRole::External(BggTraceAnchor::One),
                    ]; 3],
                ))?;
                let zero_trace =
                    record_trace_fragment(&trace_specs, zero_fragment, layer.expression());
                let (not, not_fragment) = scan_result(encoding_binary_with_trace(
                    &compiler,
                    &one_family,
                    &left,
                    EncodingOp::Sub,
                    [
                        BggTraceStep::NotVector,
                        BggTraceStep::NotPublicKey,
                        BggTraceStep::NotPlaintext,
                    ],
                    [[
                        BggOperandSourceRole::External(BggTraceAnchor::One),
                        BggOperandSourceRole::External(BggTraceAnchor::Left),
                    ]; 3],
                ))?;
                let not_trace =
                    record_trace_fragment(&trace_specs, not_fragment, layer.expression());
                let (product, product_trace, product_specs) =
                    scan_result(encoding_multiply_with_trace(&compiler, &left, &right))?;
                trace_specs.borrow_mut().extend(product_specs.into_iter().map(|mut spec| {
                    spec.layer = Some(layer.expression());
                    spec
                }));
                let (sum, sum_fragment) = scan_result(encoding_binary_with_trace(
                    &compiler,
                    &left,
                    &right,
                    EncodingOp::Add,
                    [
                        BggTraceStep::SumVector,
                        BggTraceStep::SumPublicKey,
                        BggTraceStep::SumPlaintext,
                    ],
                    [[
                        BggOperandSourceRole::External(BggTraceAnchor::Left),
                        BggOperandSourceRole::External(BggTraceAnchor::Right),
                    ]; 3],
                ))?;
                let sum_trace =
                    record_trace_fragment(&trace_specs, sum_fragment, layer.expression());
                let (two_product, two_product_fragment) = scan_result(encoding_scalar_with_trace(
                    &compiler,
                    &product,
                    compiler.public_key.ring.polynomial([2.into()]),
                ))?;
                let two_product_trace =
                    record_trace_fragment(&trace_specs, two_product_fragment, layer.expression());
                let (xor, xor_fragment) = scan_result(encoding_binary_with_trace(
                    &compiler,
                    &sum,
                    &two_product,
                    EncodingOp::Sub,
                    [
                        BggTraceStep::XorVector,
                        BggTraceStep::XorPublicKey,
                        BggTraceStep::XorPlaintext,
                    ],
                    [
                        [
                            BggOperandSourceRole::Prior(BggTraceStep::SumVector),
                            BggOperandSourceRole::Prior(BggTraceStep::TwoProductVector),
                        ],
                        [
                            BggOperandSourceRole::Prior(BggTraceStep::SumPublicKey),
                            BggOperandSourceRole::Prior(BggTraceStep::TwoProductPublicKey),
                        ],
                        [
                            BggOperandSourceRole::Prior(BggTraceStep::SumPlaintext),
                            BggOperandSourceRole::Prior(BggTraceStep::TwoProductPlaintext),
                        ],
                    ],
                ))?;
                let xor_trace =
                    record_trace_fragment(&trace_specs, xor_fragment, layer.expression());
                let active =
                    Parallel::range(params.max_layer_width.clone()).map_values(|slot| {
                        slot.as_int()
                            .less_equal(active_count.clone().sub(Int::constant(1)))
                            .to_int()
                    })?;

                let (selected_vectors, selected_vectors_trace, vectors_selection) =
                    kinds.clone().parallel_select_mats_with_trace(vec![
                        zero.vectors.clone(),
                        one_family.vectors.clone(),
                        left.vectors.clone(),
                        not.vectors.clone(),
                        product.vectors.clone(),
                        xor.vectors.clone(),
                    ])?;
                record_selection_specs(
                    &trace_specs,
                    &vectors_selection,
                    BggTraceLane::Vector,
                    BggTraceRole::CandidateSelect,
                    layer.expression(),
                    BggTraceStep::CandidateVectorSelect,
                    selection_sources(
                        &vectors_selection,
                        std::iter::once(BggOperandSourceRole::External(BggTraceAnchor::Selector))
                            .chain([
                                BggOperandSourceRole::Prior(BggTraceStep::ZeroVector),
                                BggOperandSourceRole::External(BggTraceAnchor::One),
                                BggOperandSourceRole::External(BggTraceAnchor::Left),
                                BggOperandSourceRole::Prior(BggTraceStep::NotVector),
                                BggOperandSourceRole::Prior(BggTraceStep::ProductVectorOutput),
                                BggOperandSourceRole::Prior(BggTraceStep::XorVector),
                            ]),
                    ),
                );
                let (selected_public_keys, selected_public_keys_trace, keys_selection) =
                    kinds.clone().parallel_select_mats_with_trace(vec![
                        zero.public_keys.matrices.clone(),
                        one_family.public_keys.matrices.clone(),
                        left.public_keys.matrices.clone(),
                        not.public_keys.matrices.clone(),
                        product.public_keys.matrices.clone(),
                        xor.public_keys.matrices.clone(),
                    ])?;
                record_selection_specs(
                    &trace_specs,
                    &keys_selection,
                    BggTraceLane::PublicKey,
                    BggTraceRole::CandidateSelect,
                    layer.expression(),
                    BggTraceStep::CandidatePublicKeySelect,
                    selection_sources(
                        &keys_selection,
                        std::iter::once(BggOperandSourceRole::External(BggTraceAnchor::Selector))
                            .chain([
                                BggOperandSourceRole::Prior(BggTraceStep::ZeroPublicKey),
                                BggOperandSourceRole::External(BggTraceAnchor::One),
                                BggOperandSourceRole::External(BggTraceAnchor::Left),
                                BggOperandSourceRole::Prior(BggTraceStep::NotPublicKey),
                                BggOperandSourceRole::Prior(BggTraceStep::ProductPublicKeyMultiply),
                                BggOperandSourceRole::Prior(BggTraceStep::XorPublicKey),
                            ]),
                    ),
                );
                let (selected_plaintexts, selected_plaintexts_trace, plaintext_selection) =
                    kinds.clone().parallel_select_mats_with_trace(vec![
                        zero.plaintexts.clone(),
                        one_family.plaintexts.clone(),
                        left.plaintexts.clone(),
                        not.plaintexts.clone(),
                        product.plaintexts.clone(),
                        xor.plaintexts.clone(),
                    ])?;
                record_selection_specs(
                    &trace_specs,
                    &plaintext_selection,
                    BggTraceLane::Plaintext,
                    BggTraceRole::CandidateSelect,
                    layer.expression(),
                    BggTraceStep::CandidatePlaintextSelect,
                    selection_sources(
                        &plaintext_selection,
                        std::iter::once(BggOperandSourceRole::External(BggTraceAnchor::Selector))
                            .chain([
                                BggOperandSourceRole::Prior(BggTraceStep::ZeroPlaintext),
                                BggOperandSourceRole::External(BggTraceAnchor::One),
                                BggOperandSourceRole::External(BggTraceAnchor::Left),
                                BggOperandSourceRole::Prior(BggTraceStep::NotPlaintext),
                                BggOperandSourceRole::Prior(BggTraceStep::ProductPlaintextOutput),
                                BggOperandSourceRole::Prior(BggTraceStep::XorPlaintext),
                            ]),
                    ),
                );
                let (output_vectors, output_vectors_trace, active_vectors_selection) =
                    active.clone().parallel_select_mats_with_trace(vec![
                        zero.vectors.clone(),
                        selected_vectors.clone(),
                    ])?;
                record_selection_specs(
                    &trace_specs,
                    &active_vectors_selection,
                    BggTraceLane::Vector,
                    BggTraceRole::ActiveSelect,
                    layer.expression(),
                    BggTraceStep::ActiveVectorSelect,
                    selection_sources(
                        &active_vectors_selection,
                        [
                            BggOperandSourceRole::External(BggTraceAnchor::Active),
                            BggOperandSourceRole::Prior(BggTraceStep::ZeroVector),
                            BggOperandSourceRole::Prior(BggTraceStep::CandidateVectorSelect),
                        ],
                    ),
                );
                let (output_public_keys, output_public_keys_trace, active_keys_selection) =
                    active.clone().parallel_select_mats_with_trace(vec![
                        zero.public_keys.matrices.clone(),
                        selected_public_keys.clone(),
                    ])?;
                record_selection_specs(
                    &trace_specs,
                    &active_keys_selection,
                    BggTraceLane::PublicKey,
                    BggTraceRole::ActiveSelect,
                    layer.expression(),
                    BggTraceStep::ActivePublicKeySelect,
                    selection_sources(
                        &active_keys_selection,
                        [
                            BggOperandSourceRole::External(BggTraceAnchor::Active),
                            BggOperandSourceRole::Prior(BggTraceStep::ZeroPublicKey),
                            BggOperandSourceRole::Prior(BggTraceStep::CandidatePublicKeySelect),
                        ],
                    ),
                );
                let (output_plaintexts, output_plaintexts_trace, active_plaintexts_selection) =
                    active.clone().parallel_select_mats_with_trace(vec![
                        zero.plaintexts.clone(),
                        selected_plaintexts.clone(),
                    ])?;
                record_selection_specs(
                    &trace_specs,
                    &active_plaintexts_selection,
                    BggTraceLane::Plaintext,
                    BggTraceRole::ActiveSelect,
                    layer.expression(),
                    BggTraceStep::ActivePlaintextSelect,
                    selection_sources(
                        &active_plaintexts_selection,
                        [
                            BggOperandSourceRole::External(BggTraceAnchor::Active),
                            BggOperandSourceRole::Prior(BggTraceStep::ZeroPlaintext),
                            BggOperandSourceRole::Prior(BggTraceStep::CandidatePlaintextSelect),
                        ],
                    ),
                );
                let output = BggEncodingFamily {
                    vectors: output_vectors,
                    public_keys: BggPublicKeyFamily {
                        matrices: output_public_keys,
                        reveal_plaintext: true,
                    },
                    plaintexts: output_plaintexts,
                };
                Ok((
                    (output.vectors, output.public_keys.matrices, output.plaintexts),
                    ProofTraceTransport::merge([
                        zero_trace,
                        not_trace,
                        product_trace,
                        sum_trace,
                        two_product_trace,
                        xor_trace,
                        selected_vectors_trace,
                        selected_public_keys_trace,
                        selected_plaintexts_trace,
                        output_vectors_trace,
                        output_public_keys_trace,
                        output_plaintexts_trace,
                    ]),
                ))
            },
        )?;
    let output = BggEncodingFamily {
        vectors,
        public_keys: BggPublicKeyFamily { matrices: public_keys, reveal_plaintext: true },
        plaintexts,
    };
    let handle = output.vectors.value_handle().clone();
    let output_trace =
        ProofTraceTransport::select([handle.clone()]).map_err(DynamicBooleanBggError::Dsl)?;
    let mut specs = std::rc::Rc::try_unwrap(trace_specs)
        .map_err(|_| DynamicBooleanBggError::FamilyLayout)?
        .into_inner();
    specs.push(BggTraceSpec {
        layer: None,
        gate_slot: None,
        candidate: None,
        lane: BggTraceLane::Vector,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        handle: handle.clone(),
        operands: Vec::new(),
        step: BggTraceStep::LayerOutput,
        phase: BggTracePhase::Epilogue,
        operand_sources: Vec::new(),
    });
    let trace = BooleanEncodingTrace::from_transport(
        ProofTraceTransport::merge([layer_trace, output_trace]),
        specs,
    );
    trace.validate_schema().map_err(DynamicBooleanBggError::TraceSchema)?;
    Ok((output, trace))
}

#[derive(Clone)]
struct PublicKeyBooleanGate {
    compiler: BggPublicKeyCompiler,
    one: BggPublicKeyWire,
}

impl BooleanLayerGate<Mat> for PublicKeyBooleanGate {
    fn candidates(&self, _slot: GateSlot, left: Mat, right: Mat) -> Result<[Mat; 6], DslError> {
        let left = BggPublicKeyWire { matrix: left, reveal_plaintext: true };
        let right = BggPublicKeyWire { matrix: right, reveal_plaintext: true };
        let zero = self.compiler.sub(&self.one, &self.one);
        let not = self.compiler.sub(&self.one, &left);
        let right_decomposition = right
            .matrix
            .clone()
            .decompose(self.compiler.base.clone(), self.compiler.digit_count.clone());
        let product = self.compiler.mul_with_decomposition(&left, &right, right_decomposition);
        let sum = self.compiler.add(&left, &right);
        let two_scalar = self.compiler.ring.polynomial([2.into()]);
        let two_product = self.compiler.small_scalar_mul(&product, &two_scalar);
        let xor = self.compiler.sub(&sum, &two_product);
        Ok([
            zero.matrix,
            self.one.matrix.clone(),
            left.matrix,
            not.matrix,
            product.matrix,
            xor.matrix,
        ])
    }
}

fn layer_metadata(
    context: &DslContext,
    params: &BooleanCircuitFamilyParams,
    layer: &LoopIndex,
    gate_kinds: Family<Int>,
    left_sources: Family<Int>,
    right_sources: Family<Int>,
) -> Result<(Family<Int>, Family<Int>, Family<Int>, Family<Int>), DslError> {
    let flattened = Parallel::range(params.max_layer_width.clone()).map_values(|slot| {
        context.evaluate_int(mxx_ir_core::IntExpr::Add(
            Box::new(mxx_ir_core::IntExpr::Mul(
                Box::new(layer.expression()),
                Box::new(params.max_layer_width.clone()),
            )),
            Box::new(slot.expression()),
        ))
    })?;
    let kinds = gate_kinds.parallel_gather(flattened.clone())?;
    let left = left_sources.parallel_gather(flattened.clone())?;
    let right = right_sources.parallel_gather(flattened.clone())?;
    Ok((flattened, kinds, left, right))
}

fn repeated_encoding(
    params: &BooleanCircuitFamilyParams,
    one: &BggEncodingWire,
) -> Result<BggEncodingFamily, DynamicBooleanBggError> {
    let plaintext = one.plaintext.clone().ok_or(DynamicBooleanBggError::PlaintextRequired)?;
    let vectors =
        Parallel::range(params.max_layer_width.clone()).map_values(|_| one.vector.clone())?;
    let public_keys = Parallel::range(params.max_layer_width.clone())
        .map_values(|_| one.pubkey.matrix.clone())?;
    let plaintexts =
        Parallel::range(params.max_layer_width.clone()).map_values(|_| plaintext.clone())?;
    Ok(BggEncodingFamily {
        vectors,
        public_keys: BggPublicKeyFamily { matrices: public_keys, reveal_plaintext: true },
        plaintexts,
    })
}

fn scan_result<T>(result: Result<T, DynamicBooleanBggError>) -> Result<T, DslError> {
    result.map_err(|error| match error {
        DynamicBooleanBggError::Dsl(error) => error,
        DynamicBooleanBggError::PlaintextRequired |
        DynamicBooleanBggError::FamilyLayout |
        DynamicBooleanBggError::TraceSchema(_) => DslError::Schema,
    })
}

#[derive(Clone, Copy)]
enum KeyOp {
    Add,
    Sub,
}

fn key_binary_with_trace(
    compiler: &BggPublicKeyCompiler,
    left: &BggPublicKeyFamily,
    right: &BggPublicKeyFamily,
    operation: KeyOp,
    step: BggTraceStep,
    operand_roles: [BggOperandSourceRole; 2],
) -> Result<(BggPublicKeyFamily, BggTraceFragment), DslError> {
    let compiler = compiler.clone();
    let specs = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let output_specs = specs.clone();
    let (matrices, trace) = parallel_zip_bundle_trace(
        (left.matrices.clone(), right.matrices.clone()),
        move |_, (left_matrix, right_matrix)| {
            let left_handle = left_matrix.value_handle().clone();
            let right_handle = right_matrix.value_handle().clone();
            let left = BggPublicKeyWire { matrix: left_matrix, reveal_plaintext: true };
            let right = BggPublicKeyWire { matrix: right_matrix, reveal_plaintext: true };
            let output = match operation {
                KeyOp::Add => compiler.add(&left, &right),
                KeyOp::Sub => compiler.sub(&left, &right),
            }
            .matrix;
            let output_handle = output.value_handle().clone();
            push_trace_spec_with_provenance(
                &output_specs,
                BggTraceLane::PublicKey,
                BggTraceSubrole::GateOutput,
                BggTraceRole::GateOutput,
                output_handle.clone(),
                vec![left_handle, right_handle],
                step,
                operand_roles,
            );
            Ok((output, ProofTraceTransport::select([output_handle])?))
        },
    )?;
    let specs = std::rc::Rc::try_unwrap(specs).map_err(|_| DslError::Schema)?.into_inner();
    Ok((
        BggPublicKeyFamily { matrices, reveal_plaintext: true },
        BggTraceFragment::from_transport(trace, specs),
    ))
}

#[derive(Clone, Copy)]
enum EncodingOp {
    Add,
    Sub,
}

fn encoding_binary_with_trace(
    compiler: &BggEncodingCompiler,
    left: &BggEncodingFamily,
    right: &BggEncodingFamily,
    operation: EncodingOp,
    steps: [BggTraceStep; 3],
    operand_roles: [[BggOperandSourceRole; 2]; 3],
) -> Result<(BggEncodingFamily, BggTraceFragment), DynamicBooleanBggError> {
    left.validate()?;
    right.validate()?;
    let specs = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let plaintext_specs = specs.clone();
    let (plaintexts, plaintext_trace) = parallel_zip_bundle_trace(
        (left.plaintexts.clone(), right.plaintexts.clone()),
        move |_, (left_value, right_value)| {
            let left_handle = left_value.value_handle().clone();
            let right_handle = right_value.value_handle().clone();
            let output = match operation {
                EncodingOp::Add => left_value + right_value,
                EncodingOp::Sub => left_value - right_value,
            };
            push_trace_spec_with_provenance(
                &plaintext_specs,
                BggTraceLane::Plaintext,
                BggTraceSubrole::GateOutput,
                BggTraceRole::GateOutput,
                output.value_handle().clone(),
                vec![left_handle, right_handle],
                steps[2],
                operand_roles[2],
            );
            Ok((output.clone(), ProofTraceTransport::select([output.value_handle().clone()])?))
        },
    )
    .map_err(DynamicBooleanBggError::Dsl)?;
    let vector_specs = specs.clone();
    let (vectors, vector_trace, public_keys, key_fragment) = match operation {
        EncodingOp::Add => {
            let (vectors, vector_trace) = parallel_zip_bundle_trace(
                (left.vectors.clone(), right.vectors.clone()),
                |_, (left_value, right_value)| {
                    let left_handle = left_value.value_handle().clone();
                    let right_handle = right_value.value_handle().clone();
                    let output = left_value + right_value;
                    push_trace_spec_with_provenance(
                        &vector_specs,
                        BggTraceLane::Vector,
                        BggTraceSubrole::GateOutput,
                        BggTraceRole::GateOutput,
                        output.value_handle().clone(),
                        vec![left_handle, right_handle],
                        steps[0],
                        operand_roles[0],
                    );
                    Ok((
                        output.clone(),
                        ProofTraceTransport::select([output.value_handle().clone()])?,
                    ))
                },
            )?;
            let (public_keys, key_fragment) = key_binary_with_trace(
                &compiler.public_key,
                &left.public_keys,
                &right.public_keys,
                KeyOp::Add,
                steps[1],
                operand_roles[1],
            )?;
            (vectors, vector_trace, public_keys, key_fragment)
        }
        EncodingOp::Sub => {
            let (vectors, vector_trace) = parallel_zip_bundle_trace(
                (left.vectors.clone(), right.vectors.clone()),
                |_, (left_value, right_value)| {
                    let left_handle = left_value.value_handle().clone();
                    let right_handle = right_value.value_handle().clone();
                    let output = left_value - right_value;
                    push_trace_spec_with_provenance(
                        &vector_specs,
                        BggTraceLane::Vector,
                        BggTraceSubrole::GateOutput,
                        BggTraceRole::GateOutput,
                        output.value_handle().clone(),
                        vec![left_handle, right_handle],
                        steps[0],
                        operand_roles[0],
                    );
                    Ok((
                        output.clone(),
                        ProofTraceTransport::select([output.value_handle().clone()])?,
                    ))
                },
            )?;
            let (public_keys, key_fragment) = key_binary_with_trace(
                &compiler.public_key,
                &left.public_keys,
                &right.public_keys,
                KeyOp::Sub,
                steps[1],
                operand_roles[1],
            )?;
            (vectors, vector_trace, public_keys, key_fragment)
        }
    };
    drop(vector_specs);
    let specs = std::rc::Rc::try_unwrap(specs)
        .map_err(|_| DynamicBooleanBggError::FamilyLayout)?
        .into_inner();
    let arithmetic_fragment = BggTraceFragment::from_transport(
        ProofTraceTransport::merge([plaintext_trace, vector_trace]),
        specs,
    );
    Ok((
        BggEncodingFamily { vectors, public_keys, plaintexts },
        BggTraceFragment::merge([arithmetic_fragment, key_fragment]),
    ))
}

fn encoding_multiply_with_trace(
    compiler: &BggEncodingCompiler,
    left: &BggEncodingFamily,
    right: &BggEncodingFamily,
) -> Result<(BggEncodingFamily, ProofTraceTransport, Vec<BggTraceSpec>), DynamicBooleanBggError> {
    left.validate()?;
    right.validate()?;
    let specs = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let public_specs = specs.clone();
    let (public_keys, public_key_trace) = parallel_zip_bundle_trace(
        (left.public_keys.matrices.clone(), right.public_keys.matrices.clone()),
        {
            let base = compiler.public_key.base.clone();
            let digits = compiler.public_key.digit_count.clone();
            move |_, (key, rhs)| {
                let key_handle = key.value_handle().clone();
                let rhs_handle = rhs.value_handle().clone();
                let decomposition = rhs.decompose(base.clone(), digits.clone());
                let decomposition_handle =
                    decomposition.clone().into_preimage_relation().value_handle().clone();
                let materialized = decomposition.into_preimage_relation().materialize_exact();
                let materialized_handle = materialized.value_handle().clone();
                let output = key * materialized;
                push_trace_spec_with_provenance(
                    &public_specs,
                    BggTraceLane::PublicKey,
                    BggTraceSubrole::Decompose,
                    BggTraceRole::Decomposition,
                    decomposition_handle.clone(),
                    vec![rhs_handle],
                    BggTraceStep::ProductPublicKeyDecompose,
                    [
                        BggOperandSourceRole::External(BggTraceAnchor::Right),
                        BggOperandSourceRole::External(BggTraceAnchor::Right),
                    ],
                );
                push_trace_spec_with_provenance(
                    &public_specs,
                    BggTraceLane::PublicKey,
                    BggTraceSubrole::MaterializeExact,
                    BggTraceRole::MaterializePreimageExact,
                    materialized_handle.clone(),
                    vec![decomposition_handle.clone()],
                    BggTraceStep::ProductPublicKeyMaterialize,
                    [
                        BggOperandSourceRole::Prior(BggTraceStep::ProductPublicKeyDecompose),
                        BggOperandSourceRole::Prior(BggTraceStep::ProductPublicKeyDecompose),
                    ],
                );
                push_trace_spec_with_provenance(
                    &public_specs,
                    BggTraceLane::PublicKey,
                    BggTraceSubrole::Multiply,
                    BggTraceRole::MatrixMultiply,
                    output.value_handle().clone(),
                    vec![key_handle, materialized_handle.clone()],
                    BggTraceStep::ProductPublicKeyMultiply,
                    [
                        BggOperandSourceRole::External(BggTraceAnchor::Left),
                        BggOperandSourceRole::Prior(BggTraceStep::ProductPublicKeyMaterialize),
                    ],
                );
                Ok((
                    output.clone(),
                    ProofTraceTransport::select([
                        decomposition_handle,
                        materialized_handle,
                        output.value_handle().clone(),
                    ])?,
                ))
            }
        },
    )?;
    let first_specs = specs.clone();
    let (first, first_trace) =
        parallel_zip_bundle_trace((left.vectors.clone(), right.public_keys.matrices.clone()), {
            let base = compiler.public_key.base.clone();
            let digits = compiler.public_key.digit_count.clone();
            move |_, (vector, rhs)| {
                let vector_handle = vector.value_handle().clone();
                let rhs_handle = rhs.value_handle().clone();
                let decomposition = rhs.decompose(base.clone(), digits.clone());
                let decomposition_handle =
                    decomposition.clone().into_preimage_relation().value_handle().clone();
                let output = vector.mul_decomposed(decomposition);
                push_trace_spec_with_provenance(
                    &first_specs,
                    BggTraceLane::Vector,
                    BggTraceSubrole::Decompose,
                    BggTraceRole::Decomposition,
                    decomposition_handle.clone(),
                    vec![rhs_handle],
                    BggTraceStep::ProductVectorDecompose,
                    [
                        BggOperandSourceRole::External(BggTraceAnchor::Right),
                        BggOperandSourceRole::External(BggTraceAnchor::Right),
                    ],
                );
                push_trace_spec_with_provenance(
                    &first_specs,
                    BggTraceLane::Vector,
                    BggTraceSubrole::ApplyPreimage,
                    BggTraceRole::ApplyPreimage,
                    output.value_handle().clone(),
                    vec![vector_handle, decomposition_handle.clone()],
                    BggTraceStep::ProductVectorApplyPreimage,
                    [
                        BggOperandSourceRole::External(BggTraceAnchor::Left),
                        BggOperandSourceRole::Prior(BggTraceStep::ProductVectorDecompose),
                    ],
                );
                Ok((
                    output.clone(),
                    ProofTraceTransport::select([
                        decomposition_handle,
                        output.value_handle().clone(),
                    ])?,
                ))
            }
        })?;
    let second_specs = specs.clone();
    let (second, second_trace) = parallel_zip_bundle_trace(
        (right.vectors.clone(), left.plaintexts.clone()),
        move |_, (vector, plaintext)| {
            let vector_handle = vector.value_handle().clone();
            let plaintext_handle = plaintext.value_handle().clone();
            let output = plaintext * vector;
            push_trace_spec_with_provenance(
                &second_specs,
                BggTraceLane::Vector,
                BggTraceSubrole::Multiply,
                BggTraceRole::MatrixMultiply,
                output.value_handle().clone(),
                vec![plaintext_handle, vector_handle],
                BggTraceStep::ProductVectorMultiply,
                [
                    BggOperandSourceRole::External(BggTraceAnchor::Left),
                    BggOperandSourceRole::External(BggTraceAnchor::Right),
                ],
            );
            Ok((output.clone(), ProofTraceTransport::select([output.value_handle().clone()])?))
        },
    )?;
    let vectors_specs = specs.clone();
    let (vectors, vectors_trace) = parallel_zip_bundle_trace(
        (first.clone(), second.clone()),
        move |_, (left_value, right_value)| {
            let left_handle = left_value.value_handle().clone();
            let right_handle = right_value.value_handle().clone();
            let output = left_value + right_value;
            push_trace_spec_with_provenance(
                &vectors_specs,
                BggTraceLane::Vector,
                BggTraceSubrole::GateOutput,
                BggTraceRole::GateOutput,
                output.value_handle().clone(),
                vec![left_handle, right_handle],
                BggTraceStep::ProductVectorOutput,
                [
                    BggOperandSourceRole::External(BggTraceAnchor::Left),
                    BggOperandSourceRole::External(BggTraceAnchor::Right),
                ],
            );
            Ok((output.clone(), ProofTraceTransport::select([output.value_handle().clone()])?))
        },
    )?;
    let plaintext_specs = specs.clone();
    let (plaintexts, plaintext_trace) = parallel_zip_bundle_trace(
        (left.plaintexts.clone(), right.plaintexts.clone()),
        move |_, (left_value, right_value)| {
            let left_handle = left_value.value_handle().clone();
            let right_handle = right_value.value_handle().clone();
            let output = left_value * right_value;
            push_trace_spec_with_provenance(
                &plaintext_specs,
                BggTraceLane::Plaintext,
                BggTraceSubrole::GateOutput,
                BggTraceRole::GateOutput,
                output.value_handle().clone(),
                vec![left_handle, right_handle],
                BggTraceStep::ProductPlaintextOutput,
                [
                    BggOperandSourceRole::External(BggTraceAnchor::Left),
                    BggOperandSourceRole::External(BggTraceAnchor::Right),
                ],
            );
            Ok((output.clone(), ProofTraceTransport::select([output.value_handle().clone()])?))
        },
    )?;
    let specs = std::rc::Rc::try_unwrap(specs)
        .map_err(|_| DynamicBooleanBggError::FamilyLayout)?
        .into_inner();
    Ok((
        BggEncodingFamily {
            vectors,
            public_keys: BggPublicKeyFamily { matrices: public_keys, reveal_plaintext: true },
            plaintexts,
        },
        ProofTraceTransport::merge([
            public_key_trace,
            first_trace,
            second_trace,
            vectors_trace,
            plaintext_trace,
        ]),
        specs,
    ))
}

fn encoding_scalar_with_trace(
    compiler: &BggEncodingCompiler,
    input: &BggEncodingFamily,
    scalar: Mat,
) -> Result<(BggEncodingFamily, BggTraceFragment), DynamicBooleanBggError> {
    input.validate()?;
    let (public_keys, public_key_fragment) =
        key_scalar_with_trace(&compiler.public_key, &input.public_keys, scalar.clone())?;
    let scalar_handle = scalar.value_handle().clone();
    let vectors = input.vectors.clone().parallel_map_values({
        let scalar = scalar.clone();
        move |_, value| scalar * value
    })?;
    let vector_trace = ProofTraceTransport::select([vectors.value_handle().clone()])?;
    let plaintexts =
        input.plaintexts.clone().parallel_map_values(move |_, value| value * scalar)?;
    let plaintext_trace = ProofTraceTransport::select([plaintexts.value_handle().clone()])?;
    let specs = vec![
        BggTraceSpec {
            layer: None,
            gate_slot: None,
            candidate: None,
            lane: BggTraceLane::Vector,
            subrole: BggTraceSubrole::GateOutput,
            role: BggTraceRole::GateOutput,
            handle: vectors.value_handle().clone(),
            operands: vec![input.vectors.value_handle().clone(), scalar_handle.clone()],
            step: BggTraceStep::TwoProductVector,
            phase: BggTracePhase::Layer,
            operand_sources: vec![
                BggOperandSource::External {
                    role: BggTraceAnchor::Left,
                    handle: input.vectors.value_handle().clone(),
                    path: empty_route(),
                },
                BggOperandSource::External {
                    role: BggTraceAnchor::Scalar,
                    handle: scalar_handle.clone(),
                    path: empty_route(),
                },
            ],
        },
        BggTraceSpec {
            layer: None,
            gate_slot: None,
            candidate: None,
            lane: BggTraceLane::Plaintext,
            subrole: BggTraceSubrole::GateOutput,
            role: BggTraceRole::GateOutput,
            handle: plaintexts.value_handle().clone(),
            operands: vec![input.plaintexts.value_handle().clone(), scalar_handle.clone()],
            step: BggTraceStep::TwoProductPlaintext,
            phase: BggTracePhase::Layer,
            operand_sources: vec![
                BggOperandSource::External {
                    role: BggTraceAnchor::Left,
                    handle: input.plaintexts.value_handle().clone(),
                    path: empty_route(),
                },
                BggOperandSource::External {
                    role: BggTraceAnchor::Scalar,
                    handle: scalar_handle.clone(),
                    path: empty_route(),
                },
            ],
        },
    ];
    let arithmetic_fragment = BggTraceFragment::from_transport(
        ProofTraceTransport::merge([vector_trace, plaintext_trace]),
        specs,
    );
    Ok((
        BggEncodingFamily { vectors, public_keys, plaintexts },
        BggTraceFragment::merge([public_key_fragment, arithmetic_fragment]),
    ))
}

fn key_scalar_with_trace(
    compiler: &BggPublicKeyCompiler,
    input: &BggPublicKeyFamily,
    scalar: Mat,
) -> Result<(BggPublicKeyFamily, BggTraceFragment), DslError> {
    let compiler = compiler.clone();
    let scalar_handle = scalar.value_handle().clone();
    let matrices = input.matrices.clone().parallel_map_values(move |_, matrix| {
        compiler
            .small_scalar_mul(&BggPublicKeyWire { matrix, reveal_plaintext: true }, &scalar)
            .matrix
    })?;
    let trace = ProofTraceTransport::select([matrices.value_handle().clone()])?;
    let specs = vec![BggTraceSpec {
        layer: None,
        gate_slot: None,
        candidate: None,
        lane: BggTraceLane::PublicKey,
        subrole: BggTraceSubrole::GateOutput,
        role: BggTraceRole::GateOutput,
        handle: matrices.value_handle().clone(),
        operands: vec![input.matrices.value_handle().clone(), scalar_handle.clone()],
        step: BggTraceStep::TwoProductPublicKey,
        phase: BggTracePhase::Layer,
        operand_sources: vec![
            BggOperandSource::External {
                role: BggTraceAnchor::Left,
                handle: input.matrices.value_handle().clone(),
                path: empty_route(),
            },
            BggOperandSource::External {
                role: BggTraceAnchor::Scalar,
                handle: scalar_handle.clone(),
                path: empty_route(),
            },
        ],
    }];
    Ok((
        BggPublicKeyFamily { matrices, reveal_plaintext: true },
        BggTraceFragment::from_transport(trace, specs),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::{ParamEnv, node::NodeKind, types::NodeId};

    #[test]
    fn public_key_and_encoding_candidates_have_uniform_selected_schemas() {
        let ring = Ring::new(257, 8);
        let public_key =
            BggPublicKeyCompiler { ring: ring.clone(), base: 2.into(), digit_count: 4.into() };

        let (public_context, public_params) =
            BooleanCircuitFamilyParams::declare(DslContext::new("dynamic-bgg-public-key"));
        let public_circuit =
            BooleanCircuitFamilyInputs::protocol_inputs(&public_context, &public_params);
        let one_key =
            BggPublicKeyWire { matrix: ring.input("one-key", (1, 4)), reveal_plaintext: true };
        let public_inputs = BggPublicKeyFamily {
            matrices: ring.input_family(
                "public-key-inputs",
                public_params.max_layer_width.clone(),
                (1, 4),
            ),
            reveal_plaintext: true,
        };
        let public_output = evaluate_boolean_public_key_layers(
            &public_context,
            &public_params,
            public_circuit,
            public_inputs,
            one_key.clone(),
            public_key.clone(),
        )
        .unwrap();
        let public_graph = public_context
            .family_output("output", public_output.matrices)
            .unwrap()
            .build()
            .unwrap();
        public_graph.validate(&bindings()).unwrap();

        let (encoding_context, encoding_params) =
            BooleanCircuitFamilyParams::declare(DslContext::new("dynamic-bgg-encoding"));
        let encoding_circuit =
            BooleanCircuitFamilyInputs::protocol_inputs(&encoding_context, &encoding_params);
        let one_encoding = BggEncodingWire {
            vector: ring.input("one-vector", (1, 4)),
            pubkey: one_key.clone(),
            plaintext: Some(ring.input("one-plaintext", (1, 1))),
        };
        let encoding_inputs = BggEncodingFamily {
            vectors: ring.input_family(
                "encoding-input-vectors",
                encoding_params.max_layer_width.clone(),
                (1, 4),
            ),
            public_keys: BggPublicKeyFamily {
                matrices: ring.input_family(
                    "encoding-input-public-keys",
                    encoding_params.max_layer_width.clone(),
                    (1, 4),
                ),
                reveal_plaintext: true,
            },
            plaintexts: ring.input_family(
                "encoding-input-plaintexts",
                encoding_params.max_layer_width.clone(),
                (1, 1),
            ),
        };
        let encoding_output = evaluate_boolean_encoding_layers(
            &encoding_context,
            &encoding_params,
            encoding_circuit,
            encoding_inputs,
            one_encoding,
            BggEncodingCompiler { public_key },
        )
        .unwrap();
        let encoding_graph = encoding_context
            .family_output("vector", encoding_output.vectors)
            .unwrap()
            .family_output("public-key", encoding_output.public_keys.matrices)
            .unwrap()
            .family_output("plaintext", encoding_output.plaintexts)
            .unwrap()
            .build()
            .unwrap();
        encoding_graph.validate(&bindings()).unwrap();
        let decomposition_count = encoding_graph
            .graph
            .scopes()
            .values()
            .flat_map(|scope| scope.nodes())
            .filter(|node| matches!(node.kind(), NodeKind::GadgetDecompose { .. }))
            .count();
        assert!(
            decomposition_count >= 1,
            "the encoding family contains an explicit deterministic decomposition"
        );
        assert!(
            encoding_graph
                .graph
                .scopes()
                .values()
                .flat_map(|scope| scope.nodes())
                .any(|node| matches!(node.kind(), NodeKind::ApplyPreimage)),
            "dynamic BGG multiplication must consume the decomposition relation"
        );
    }

    #[test]
    fn encoding_trace_is_freeze_only_and_keeps_graph_shape() {
        fn build(
            retain_trace: bool,
        ) -> (mxx_ir_core::Graph, usize, Vec<FrozenBooleanEncodingTraceEntry>) {
            let ring = Ring::new(257, 8);
            let compiler = BggEncodingCompiler {
                public_key: BggPublicKeyCompiler {
                    ring: ring.clone(),
                    base: 2.into(),
                    digit_count: 4.into(),
                },
            };
            let (context, params) =
                BooleanCircuitFamilyParams::declare(DslContext::new("trace-shape-equivalence"));
            let circuit = BooleanCircuitFamilyInputs::protocol_inputs(&context, &params);
            let one = BggEncodingWire {
                vector: ring.input("one-vector", (1, 4)),
                pubkey: BggPublicKeyWire {
                    matrix: ring.input("one-key", (1, 4)),
                    reveal_plaintext: true,
                },
                plaintext: Some(ring.input("one-plaintext", (1, 1))),
            };
            let preceding = BggEncodingFamily {
                vectors: ring.input_family(
                    "encoding-input-vectors",
                    params.max_layer_width.clone(),
                    (1, 4),
                ),
                public_keys: BggPublicKeyFamily {
                    matrices: ring.input_family(
                        "encoding-input-public-keys",
                        params.max_layer_width.clone(),
                        (1, 4),
                    ),
                    reveal_plaintext: true,
                },
                plaintexts: ring.input_family(
                    "encoding-input-plaintexts",
                    params.max_layer_width.clone(),
                    (1, 1),
                ),
            };
            let (output, trace) = if retain_trace {
                let (output, trace) = evaluate_boolean_encoding_layers_with_trace(
                    &context, &params, circuit, preceding, one, compiler,
                )
                .unwrap();
                (output, Some(trace))
            } else {
                (
                    evaluate_boolean_encoding_layers(
                        &context, &params, circuit, preceding, one, compiler,
                    )
                    .unwrap(),
                    None,
                )
            };
            let retained = trace
                .as_ref()
                .map(|trace| trace.clone().into_retained_values())
                .unwrap_or_default();
            let retained_count = retained.len();
            let context = context
                .family_output("vector", output.vectors)
                .unwrap()
                .family_output("public-key", output.public_keys.matrices)
                .unwrap()
                .family_output("plaintext", output.plaintexts)
                .unwrap();
            let (built, freeze_map) = context.build_retaining(retained.clone()).unwrap();
            let frozen_trace = trace
                .as_ref()
                .map(|trace| trace.resolve_with_graph(&freeze_map, Some(&built.graph)).unwrap())
                .unwrap_or_default();
            if retain_trace {
                BggTraceFragment::validate_frozen_paths(&frozen_trace, &built.graph).unwrap();
                let mut wrong_route = frozen_trace.clone();
                let mut changed_route = false;
                for entry in &mut wrong_route {
                    for source in &mut entry.operand_sources {
                        let route = match source {
                            FrozenBggOperandSource::External { path, .. } |
                            FrozenBggOperandSource::Prior { path, .. } => path,
                        };
                        if let Some(hop) = route.exits.first_mut() {
                            hop.owner = NodeId(u64::MAX);
                            changed_route = true;
                            break;
                        }
                    }
                    if changed_route {
                        break;
                    }
                }
                assert!(changed_route, "BGG trace must contain a routed operand");
                assert!(
                    BggTraceFragment::validate_frozen_paths(&wrong_route, &built.graph).is_err()
                );
                let mut wrong_exposed = frozen_trace.clone();
                let mut changed_exposed = false;
                for entry in &mut wrong_exposed {
                    for source in &mut entry.operand_sources {
                        let route = match source {
                            FrozenBggOperandSource::External { path, .. } |
                            FrozenBggOperandSource::Prior { path, .. } => path,
                        };
                        if let Some(hop) = route.exits.first_mut() {
                            hop.output_index = hop.output_index.saturating_add(1);
                            changed_exposed = true;
                            break;
                        }
                        if let Some(hop) = route.enters.first_mut() {
                            hop.input_index = hop.input_index.saturating_add(1);
                            changed_exposed = true;
                            break;
                        }
                    }
                    if changed_exposed {
                        break;
                    }
                }
                assert!(changed_exposed);
                assert!(
                    BggTraceFragment::validate_frozen_paths(&wrong_exposed, &built.graph).is_err()
                );
                let mut saw_decomposition = false;
                let mut saw_materialize = false;
                let mut saw_preimage = false;
                let mut saw_multiply = false;
                let mut saw_candidate_select = false;
                let mut saw_active_select = false;
                for value in retained {
                    let reference = freeze_map.resolve_typed(&value).expect("trace reference");
                    let scope =
                        built.graph.scope(&reference.reference().scope).expect("trace scope");
                    let node = scope.node(reference.reference().wire.node).expect("trace node");
                    assert_eq!(
                        node.output_types()[reference.reference().wire.port.0 as usize],
                        reference.wire_type().clone()
                    );
                    saw_decomposition |= matches!(node.kind(), NodeKind::GadgetDecompose { .. });
                    saw_materialize |= matches!(node.kind(), NodeKind::MaterializePreimageExact);
                    saw_preimage |= matches!(node.kind(), NodeKind::ApplyPreimage);
                    saw_multiply |= matches!(
                        node.kind(),
                        NodeKind::MatrixBinary(mxx_ir_core::node::MatrixBinaryOp::Multiply)
                    );
                    if let NodeKind::Select { count } = node.kind() {
                        if *count == mxx_ir_core::IntExpr::constant(6) {
                            saw_candidate_select = true;
                        }
                        if *count == mxx_ir_core::IntExpr::constant(2) {
                            saw_active_select = true;
                        }
                    }
                    let arguments = scope.arguments(node).expect("frozen producer arguments");
                    assert!(
                        arguments
                            .iter()
                            .all(|argument| { (argument.node.0 as usize) < scope.nodes().len() })
                    );
                }
                assert!(saw_decomposition, "decomposition producer is retained");
                assert!(saw_materialize, "materialization producer is retained");
                assert!(saw_preimage, "preimage producer is retained");
                assert!(saw_multiply, "matrix multiplication producer is retained");
                assert!(saw_candidate_select, "candidate selection producer is retained");
                assert!(saw_active_select, "active selection producer is retained");
                for entry in frozen_trace.iter().filter(|entry| {
                    matches!(entry.role, BggTraceRole::CandidateSelect | BggTraceRole::ActiveSelect)
                }) {
                    let scope = built
                        .graph
                        .scope(&entry.handle.reference().scope)
                        .expect("selection trace scope");
                    let node = scope
                        .node(entry.handle.reference().wire.node)
                        .expect("selection trace node");
                    assert!(matches!(node.kind(), NodeKind::Select { .. }));
                    let stored_operands = entry
                        .operands
                        .iter()
                        .map(|operand| operand.reference().wire)
                        .collect::<Vec<_>>();
                    assert_eq!(stored_operands, scope.arguments(node).unwrap());
                }
            }
            (built.graph, retained_count, frozen_trace)
        }

        let (without_trace, without_count, _) = build(false);
        let (with_trace, with_count, _) = build(true);
        assert_eq!(without_trace.outputs(), with_trace.outputs());
        assert_eq!(without_trace.effect_roots(), with_trace.effect_roots());
        assert_eq!(
            without_trace.scopes().keys().collect::<Vec<_>>(),
            with_trace.scopes().keys().collect::<Vec<_>>()
        );
        for scope in without_trace.scopes().keys() {
            let left = without_trace.scope(scope).expect("ordinary scope");
            let right = with_trace.scope(scope).expect("traced scope");
            assert_eq!(left.nodes().len(), right.nodes().len());
            for (left_node, right_node) in left.nodes().iter().zip(right.nodes()) {
                assert_eq!(left_node.kind(), right_node.kind());
                assert_eq!(left.arguments(left_node), right.arguments(right_node));
                assert_eq!(left_node.output_types(), right_node.output_types());
            }
            assert_eq!(left.inputs(), right.inputs());
            assert_eq!(left.outputs(), right.outputs());
        }
        assert_eq!(without_count, 0);
        assert!(with_count > 0, "trace must retain at least one nested producer");
    }

    fn schema_fixture() -> BggTraceFragment {
        let ring = Ring::new(257, 8);
        let handle = ring.input("schema-trace", (1, 1)).value_handle().clone();
        let layer = mxx_ir_core::IntExpr::constant(0);
        let mut entries = EXPECTED_LAYER_TRACE
            .iter()
            .enumerate()
            .map(|(index, expected)| BooleanEncodingTraceEntry {
                layer: expected.has_layer.then(|| layer.clone()),
                gate_slot: expected.has_gate_slot.then(|| mxx_ir_core::IntExpr::constant(0)),
                candidate: None,
                lane: expected.lane,
                subrole: expected.subrole,
                role: expected.role,
                handle: handle.clone(),
                operands: vec![handle.clone(); expected.operands],
                step: EXPECTED_STEPS[index],
                phase: BggTracePhase::Layer,
                operand_sources: expected_operand_sources(EXPECTED_STEPS[index])
                    .iter()
                    .map(|source| match source {
                        ExpectedOperandSource::External(role) => BggOperandSource::External {
                            role: *role,
                            handle: handle.clone(),
                            path: empty_route(),
                        },
                        ExpectedOperandSource::Prior(step) => BggOperandSource::Prior {
                            step: *step,
                            handle: handle.clone(),
                            path: empty_route(),
                        },
                    })
                    .collect(),
            })
            .collect::<Vec<_>>();
        entries.push(BooleanEncodingTraceEntry {
            layer: None,
            gate_slot: None,
            candidate: None,
            lane: BggTraceLane::Vector,
            subrole: BggTraceSubrole::GateOutput,
            role: BggTraceRole::GateOutput,
            handle,
            operands: Vec::new(),
            step: BggTraceStep::LayerOutput,
            phase: BggTracePhase::Epilogue,
            operand_sources: Vec::new(),
        });
        BggTraceFragment { entries, transport: ProofTraceTransport::default(), specs: Vec::new() }
    }

    #[test]
    fn trace_schema_rejects_structural_mutations() {
        assert!(schema_fixture().validate_schema().is_ok());

        let mut all_external_left = schema_fixture();
        for entry in &mut all_external_left.entries {
            for source in &mut entry.operand_sources {
                *source = BggOperandSource::External {
                    role: BggTraceAnchor::Left,
                    handle: entry.operands.first().cloned().unwrap_or_else(|| entry.handle.clone()),
                    path: empty_route(),
                };
            }
        }
        assert!(all_external_left.validate_schema().is_err());

        let mut removed = schema_fixture();
        removed.entries.pop();
        assert!(removed.validate_schema().is_err());

        let mut extra = schema_fixture();
        extra.entries.push(extra.entries.last().unwrap().clone());
        assert!(extra.validate_schema().is_err());

        let mut swapped = schema_fixture();
        swapped.entries.swap(0, 1);
        assert!(swapped.validate_schema().is_err());

        let mut wrong_lane = schema_fixture();
        wrong_lane.entries[0].lane = BggTraceLane::Vector;
        assert!(wrong_lane.validate_schema().is_err());

        let mut wrong_subrole = schema_fixture();
        wrong_subrole.entries[0].subrole = BggTraceSubrole::Select;
        assert!(wrong_subrole.validate_schema().is_err());

        let mut wrong_coordinate = schema_fixture();
        wrong_coordinate.entries[0].layer = None;
        assert!(wrong_coordinate.validate_schema().is_err());

        let mut wrong_arity = schema_fixture();
        wrong_arity.entries[0].operands.pop();
        assert!(wrong_arity.validate_schema().is_err());

        let mut wrong_descriptor = schema_fixture();
        wrong_descriptor.entries[0].operand_sources[0] = BggOperandSource::External {
            role: BggTraceAnchor::Left,
            handle: wrong_descriptor.entries[0].handle.clone(),
            path: empty_route(),
        };
        assert!(wrong_descriptor.validate_schema().is_err());

        let mut wrong_prior = schema_fixture();
        wrong_prior.entries[20].operand_sources[0] = BggOperandSource::Prior {
            step: BggTraceStep::TwoProductPlaintext,
            handle: wrong_prior.entries[20].handle.clone(),
            path: empty_route(),
        };
        assert!(wrong_prior.validate_schema().is_err());

        let mut permuted = schema_fixture();
        permuted.entries[20].operand_sources.swap(0, 1);
        assert!(permuted.validate_schema().is_err());

        let ring = Ring::new(257, 8);
        let other = ring.input("schema-trace-other", (1, 1)).value_handle().clone();
        let mut wrong_source = schema_fixture();
        wrong_source.entries[0].operand_sources[0] = BggOperandSource::External {
            role: BggTraceAnchor::One,
            handle: other,
            path: empty_route(),
        };
        assert!(wrong_source.validate_schema().is_err());
    }

    fn bindings() -> ParamEnv {
        ParamEnv {
            integers: std::collections::BTreeMap::from([
                (BooleanCircuitFamilyParams::INSTANCE_WIDTH_PARAMETER.to_owned(), 1.into()),
                (BooleanCircuitFamilyParams::WITNESS_WIDTH_PARAMETER.to_owned(), 1.into()),
                (BooleanCircuitFamilyParams::DEPTH_PARAMETER.to_owned(), 1.into()),
                (BooleanCircuitFamilyParams::MAX_LAYER_WIDTH_PARAMETER.to_owned(), 2.into()),
            ]),
            ..ParamEnv::default()
        }
    }
}
