//! Concrete abstract interpretation of frozen executable IR.

use crate::{
    AbstractValue, DiagnosticSite, ExternalInputValue, FamilyState, MatrixState, SimulationError,
    SimulationRequest,
    bound::ProductGeometry,
    relation::{self, RightPreimage},
    state::{self, TrapdoorState},
};
use mxx_ir_core::{
    FrozenGraphScopeId, Graph, GraphScope, ParamEnv, WireRef, WireType,
    artifact::ArtifactConfidentiality,
    node::{IntBinaryOp, MatrixBinaryOp, NodeKind},
    types::ConcreteMatrixType,
};
use num_bigint::{BigInt, BigUint};
use num_traits::{One, Signed, ToPrimitive, Zero};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Clone)]
struct Info {
    value: AbstractValue,
    ty: Option<ConcreteMatrixType>,
    relation: Option<RightPreimage>,
    view: crate::FamilyViewId,
    /// If this value is a trapdoor (or a family whose element is a
    /// trapdoor), this is the exact canonical view of its paired public
    /// matrix/family.  Pairing is an evaluator fact carried with the value;
    /// it is not reconstructed from node or wire names later.
    paired_public: Option<crate::FamilyViewId>,
}

fn artifact_validation_order(
    request: &SimulationRequest,
    needed: &HashSet<crate::StageId>,
) -> Result<Vec<usize>, SimulationError> {
    fn visit(
        index: usize,
        request: &SimulationRequest,
        needed: &HashSet<crate::StageId>,
        by_production: &HashMap<mxx_ir_core::artifact::ProductionId, usize>,
        visiting: &mut HashSet<usize>,
        visited: &mut HashSet<usize>,
        order: &mut Vec<usize>,
    ) -> Result<(), SimulationError> {
        if visited.contains(&index) {
            return Ok(());
        }
        if !visiting.insert(index) {
            return Err(SimulationError::ArtifactResolution {
                message: "cyclic artifact validation dependency".into(),
                site: None,
            });
        }
        let stage = &request.program.stages[index];
        for scope in stage.graph.scopes().values() {
            for node in scope.nodes() {
                let mxx_ir_core::node::NodeKind::Input { artifact: Some(artifact), .. } =
                    node.kind()
                else {
                    continue;
                };
                let producer = by_production.get(&artifact.production_id).ok_or_else(|| {
                    SimulationError::ArtifactResolution {
                        message: "artifact producer missing".into(),
                        site: None,
                    }
                })?;
                if needed.contains(&request.program.stages[*producer].id) {
                    visit(*producer, request, needed, by_production, visiting, visited, order)?;
                }
            }
        }
        visiting.remove(&index);
        visited.insert(index);
        order.push(index);
        Ok(())
    }

    let by_production = request
        .program
        .stages
        .iter()
        .enumerate()
        .map(|(index, stage)| (stage.production_id.clone(), index))
        .collect::<HashMap<_, _>>();
    let mut visiting = HashSet::new();
    let mut visited = HashSet::new();
    let mut order = Vec::new();
    for (index, stage) in request.program.stages.iter().enumerate() {
        if needed.contains(&stage.id) {
            visit(index, request, needed, &by_production, &mut visiting, &mut visited, &mut order)?;
        }
    }
    Ok(order)
}

pub(crate) fn run(request: &SimulationRequest) -> Result<crate::SimulationReport, SimulationError> {
    request.validate()?;
    let plan = crate::plan::Plan::build(request)?;
    // Validate stages in program order while accumulating the exact manifests
    // exported by their producers.  Cross-stage artifact inputs cannot be
    // validated in isolation: the consumer must see the producer's concrete
    // output type, family shape, and confidentiality.
    // Validation elaborates every stage's complete graph, including dead
    // artifact inputs, so close the manifest dependency graph over all stages.
    let needed =
        request.program.stages.iter().map(|stage| stage.id.clone()).collect::<HashSet<_>>();
    let validation_order = artifact_validation_order(request, &needed)?;
    let mut manifests = BTreeMap::new();
    for index in validation_order {
        let stage = &request.program.stages[index];
        let validated =
            mxx_ir_core::validate_with_manifests(&stage.graph, &request.environment, &manifests)
                .map_err(|error| SimulationError::InvalidGraph {
                    message: format!("stage {:?}: {error}", stage.id),
                    site: None,
                })?;
        let manifest = mxx_ir_core::artifact::export_validated_manifest(
            stage.production_id.clone(),
            &validated,
        )
        .map_err(|error| SimulationError::InvalidGraph {
            message: error.to_string(),
            site: None,
        })?;
        manifests.insert(stage.production_id.clone(), manifest);
    }
    if request.limits.maximum_planned_wires.is_some_and(|limit| plan.wires.len() > limit) {
        return Err(SimulationError::ResourceLimitExceeded {
            message: "maximum planned wires exceeded".into(),
            site: None,
        });
    }
    let mut e = Evaluator {
        request,
        stages: HashMap::new(),
        visiting: HashSet::new(),
        sources: HashMap::new(),
        gadget_sources: HashMap::new(),
        source_lineages: HashMap::new(),
        lineage_sources: HashMap::new(),
        mapped_sources: HashMap::new(),
        gathered_sources: HashMap::new(),
        binder_sources: HashMap::new(),
        next_source: 0,
        preimages: HashMap::new(),
        states: HashMap::new(),
        selector_views: HashMap::new(),
        next_selector: plan.interners.selectors.len() as u32,
        planned: plan.wires.len(),
        transfers: 0,
        dropped: Vec::new(),
        interners: plan.interners,
        reached: plan.wires.iter().cloned().collect(),
        artifact_outputs: plan.artifact_outputs,
    };
    let mut roots = Vec::new();
    for root in &request.roots {
        let out = e.stage(&root.stage)?.get(&root.output).cloned().ok_or_else(|| {
            SimulationError::UnknownOutput {
                stage: root.stage.clone(),
                output: root.output.clone(),
            }
        })?;
        let state = matrix_state(&out.value).ok_or_else(|| SimulationError::InvalidGraph {
            message: "root is not a matrix or matrix family".into(),
            site: None,
        })?;
        roots.push(crate::RootNoiseReport {
            root: root.clone(),
            maximum_absolute_coefficient_error: state.error_bound,
        });
    }
    Ok(crate::SimulationReport::new(
        roots,
        crate::SimulationDiagnostics {
            planned_wires: e.planned,
            transfer_steps: e.transfers,
            dropped_carriers: e.dropped,
        },
    ))
}

struct Evaluator<'a> {
    request: &'a SimulationRequest,
    stages: HashMap<crate::StageId, BTreeMap<String, Info>>,
    visiting: HashSet<crate::StageId>,
    sources: HashMap<SourceKey, crate::SourceId>,
    gadget_sources: HashMap<crate::GadgetDescriptor, crate::SourceId>,
    /// Canonical coordinate function carried by each source identity.  A
    /// family source is represented by its ordered scalar source leaves, so
    /// packing individual values and reindexing an existing family can
    /// converge to the same source identity.
    source_lineages: HashMap<crate::SourceId, SourceLineage>,
    lineage_sources: HashMap<SourceLineage, crate::SourceId>,
    mapped_sources: HashMap<(crate::SourceId, mxx_ir_core::IndexMap, Vec<usize>), crate::SourceId>,
    gathered_sources:
        HashMap<(crate::SourceId, Vec<crate::SelectorId>, Vec<usize>), crate::SourceId>,
    /// Scalar source identities created while freezing a body evaluated under
    /// a structural binder.  The evaluator visits a ParallelGrid body once
    /// (at a representative environment), so its primitive samples need
    /// fresh, coordinate-indexed leaves when the body is frozen as a family.
    /// The key is the representative source, output shape, and flat lane;
    /// consequently public/trapdoor paired views receive the same leaves.
    binder_sources: HashMap<(crate::SourceId, Vec<usize>, usize), crate::SourceId>,
    next_source: u32,
    preimages: HashMap<crate::FamilyViewId, RightPreimage>,
    states: HashMap<crate::FamilyViewId, MatrixState>,
    /// Runtime selector identity follows the normalized semantic family view.
    /// Reusing one family through the same map is correlated across structural
    /// scopes, while unrelated wires and different maps remain distinct.
    selector_views: HashMap<crate::FamilyViewId, crate::SelectorId>,
    next_selector: u32,
    planned: usize,
    transfers: u64,
    dropped: Vec<crate::DroppedCarrierDiagnostic>,
    interners: crate::identity::Interners,
    reached: HashSet<crate::plan::PlannedWire>,
    artifact_outputs:
        std::collections::BTreeMap<crate::StageId, std::collections::BTreeSet<String>>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct SourceKey {
    stage: crate::StageId,
    scope: FrozenGraphScopeId,
    occurrence: Vec<String>,
    node: usize,
    role: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct SourceLineage {
    shape: Vec<usize>,
    leaves: Vec<crate::SourceId>,
}

impl<'a> Evaluator<'a> {
    /// Record each input carrier whose source identity is absent from the
    /// result.  Carrier identities are the simulator's symbolic witnesses;
    /// losing one means the corresponding source can no longer be related to
    /// the output, even when the numeric bound remains conservative.
    fn record_carrier_drops(
        &mut self,
        inputs: &[Info],
        output: Option<&MatrixState>,
        site: Option<DiagnosticSite>,
        reason: &str,
    ) {
        self.record_carrier_drops_iter(inputs.iter(), output, site, reason);
    }

    fn record_carrier_drops_iter<'inputs, I>(
        &mut self,
        inputs: I,
        output: Option<&MatrixState>,
        site: Option<DiagnosticSite>,
        reason: &str,
    ) where
        I: IntoIterator<Item = &'inputs Info>,
    {
        let retained = output.and_then(|state| state.right_carrier.as_ref().map(|c| c.source));
        let mut seen = HashSet::new();
        for source in inputs.into_iter().filter_map(|input| {
            matrix_state(&input.value).and_then(|state| state.right_carrier.map(|c| c.source))
        }) {
            // Family packing and structural reindexing may replace a scalar
            // source with a canonical coordinate function.  Its leaf set is
            // still the same witness, so that replacement is not a loss.
            let represented_by_output = retained.is_some_and(|output_source| {
                output_source == source ||
                    self.source_lineages
                        .get(&output_source)
                        .is_some_and(|lineage| lineage.leaves.contains(&source))
            });
            if represented_by_output || !seen.insert((source, retained)) {
                continue;
            }
            self.dropped.push(crate::DroppedCarrierDiagnostic {
                site: site.clone().unwrap_or_default(),
                reason: reason.into(),
                expected_source: Some(source),
                actual_source: retained,
            });
        }
    }

    fn join_uniform_with_diagnostics(
        &mut self,
        left: Info,
        right: Info,
        ty: Option<&ConcreteMatrixType>,
        site: Option<DiagnosticSite>,
    ) -> Result<Info, SimulationError> {
        let joined = join_uniform(left.clone(), right.clone(), ty)?;
        let output = matrix_state(&joined.value);
        self.record_carrier_drops(
            &[left, right],
            output.as_ref(),
            site,
            "carrier lost when matrix branches have incompatible sources",
        );
        Ok(joined)
    }

    fn stage(&mut self, id: &crate::StageId) -> Result<BTreeMap<String, Info>, SimulationError> {
        if let Some(x) = self.stages.get(id) {
            return Ok(x.clone());
        }
        if !self.visiting.insert(id.clone()) {
            return Err(SimulationError::ArtifactResolution {
                message: "cyclic artifact resolution".into(),
                site: None,
            });
        }
        let stage = self
            .request
            .program
            .stage(id)
            .ok_or_else(|| SimulationError::UnknownStage { stage: id.clone() })?;
        let vals = self.scope(
            id,
            &stage.graph,
            &FrozenGraphScopeId::Root,
            &[],
            self.request.environment.clone(),
            HashMap::new(),
        )?;
        self.visiting.remove(id);
        let mut out = BTreeMap::new();
        for name in self.requested_outputs(id) {
            let root = stage.graph.outputs().get(&name).ok_or_else(|| {
                SimulationError::ArtifactResolution {
                    message: format!("requested artifact output {name} is missing"),
                    site: None,
                }
            })?;
            out.insert(
                name.clone(),
                vals.get(&root.value).cloned().ok_or_else(|| SimulationError::InvalidGraph {
                    message: format!("output {name} was not evaluated"),
                    site: None,
                })?,
            );
        }
        self.stages.insert(id.clone(), out.clone());
        Ok(out)
    }

    fn requested_outputs(&self, id: &crate::StageId) -> Vec<String> {
        if self.request.program.stage(id).is_none() {
            return Vec::new();
        }
        let mut names = self
            .request
            .roots
            .iter()
            .filter(|root| root.stage == *id)
            .map(|root| root.output.clone())
            .collect::<std::collections::BTreeSet<_>>();
        if let Some(artifacts) = self.artifact_outputs.get(id) {
            names.extend(artifacts.iter().cloned());
        }
        names.into_iter().collect()
    }

    fn source_for(
        &mut self,
        stage: &crate::StageId,
        scope: &FrozenGraphScopeId,
        occurrence: &[String],
        node: usize,
        role: &'static str,
    ) -> crate::SourceId {
        let key = SourceKey {
            stage: stage.clone(),
            scope: scope.clone(),
            occurrence: occurrence.to_vec(),
            node,
            role,
        };
        if let Some(source) = self.sources.get(&key) {
            return *source;
        }
        let source = crate::SourceId(self.next_source);
        self.next_source = self.next_source.saturating_add(1);
        self.sources.insert(key, source);
        self.register_lineage(source, SourceLineage { shape: Vec::new(), leaves: vec![source] });
        source
    }

    fn register_lineage(&mut self, source: crate::SourceId, lineage: SourceLineage) {
        self.source_lineages.insert(source, lineage.clone());
        self.lineage_sources.entry(lineage).or_insert(source);
    }

    fn source_origin(&self, source: crate::SourceId) -> String {
        self.source_origin_with_depth(source, 0)
    }

    fn source_origin_with_depth(&self, source: crate::SourceId, depth: usize) -> String {
        if let Some(key) =
            self.sources.iter().find_map(|(key, value)| (*value == source).then_some(key))
        {
            return format!("primitive {key:?}");
        }
        if let Some((key, _)) = self.mapped_sources.iter().find(|(_, value)| **value == source) {
            return if depth >= 8 {
                format!("mapped {key:?}")
            } else {
                format!("mapped {key:?} <- {}", self.source_origin_with_depth(key.0, depth + 1))
            };
        }
        if let Some((key, _)) = self.gathered_sources.iter().find(|(_, value)| **value == source) {
            let parent = self.source_lineages.get(&key.0);
            return if depth >= 8 {
                format!(
                    "gathered {key:?}, parent_shape={:?}, parent_leaves={:?}",
                    parent.map(|lineage| &lineage.shape),
                    parent.map(|lineage| lineage.leaves.iter().take(8).collect::<Vec<_>>())
                )
            } else {
                format!(
                    "gathered {key:?}, parent_shape={:?}, parent_leaves={:?} <- {}",
                    parent.map(|lineage| &lineage.shape),
                    parent.map(|lineage| lineage.leaves.iter().take(8).collect::<Vec<_>>()),
                    self.source_origin_with_depth(key.0, depth + 1)
                )
            };
        }
        if let Some((key, _)) = self.binder_sources.iter().find(|(_, value)| **value == source) {
            return format!(
                "binder {:?} <- {}",
                key,
                self.source_origin_with_depth(key.0, depth + 1)
            );
        }
        match self.source_lineages.get(&source) {
            Some(lineage) if depth < 8 => format!(
                "canonical-lineage shape={:?} leaves={:?} <- [{}]",
                lineage.shape,
                lineage.leaves.iter().take(8).collect::<Vec<_>>(),
                lineage
                    .leaves
                    .iter()
                    .take(8)
                    .map(|leaf| self.source_origin_with_depth(*leaf, depth + 1))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Some(lineage) => format!("canonical-lineage {lineage:?}"),
            None => "unknown-source".into(),
        }
    }

    fn view_shape(&self, view: crate::FamilyViewId) -> Option<Vec<usize>> {
        self.interners.views.iter().find_map(|(key, id)| (*id == view).then(|| key.shape.clone()))
    }

    fn canonical_source_for_lineage(&mut self, lineage: SourceLineage) -> crate::SourceId {
        if let Some(first) = lineage.leaves.first().copied() &&
            lineage.leaves.iter().all(|leaf| *leaf == first) &&
            self.gadget_sources.values().any(|source| *source == first)
        {
            // G is one public, index-independent source. Keep its scalar
            // identity instead of manufacturing a shape-specific wrapper
            // around repeated gadget leaves. Other repeated sources may have
            // acquired their equality through a structural map, whose axes
            // must remain available for later relation matching.
            return first;
        }
        if let Some(source) = self.lineage_sources.get(&lineage) {
            return *source;
        }
        let source = crate::SourceId(self.next_source);
        self.next_source = self.next_source.saturating_add(1);
        self.register_lineage(source, lineage);
        source
    }

    fn selector_for(&mut self, view: crate::FamilyViewId) -> crate::SelectorId {
        if let Some(id) = self.selector_views.get(&view) {
            return *id;
        }
        let id = crate::SelectorId(self.next_selector);
        self.next_selector = self.next_selector.saturating_add(1);
        self.selector_views.insert(view, id);
        id
    }

    /// Intern a group-indexed source function from its exact coordinate
    /// mapping.  The source leaves are flattened before interning.  This is
    /// what makes equivalent structural forms (pointwise pack versus family
    /// reindex) share an identity without comparing bounds or matrix values.
    fn group_source_for(
        &mut self,
        sources: Vec<crate::SourceId>,
        shape: Vec<usize>,
    ) -> crate::SourceId {
        let mut leaves = sources
            .iter()
            .flat_map(|source| {
                self.source_lineages
                    .get(source)
                    .map(|lineage| lineage.leaves.clone())
                    .unwrap_or_else(|| vec![*source])
            })
            .collect::<Vec<_>>();
        let count = shape.iter().copied().product::<usize>().max(1);
        // A single scalar source entering a family-valued structural node is
        // an index-independent broadcast.  Record the full coordinate
        // function so later identity reindexes can resolve every lane.  This
        // does not merge distinct sources: a multi-leaf lineage is preserved
        // verbatim and must already match the requested family cardinality.
        if leaves.len() == 1 && count > 1 {
            leaves.resize(count, leaves[0]);
        }
        let canonical = self.canonical_source_for_lineage(SourceLineage { shape, leaves });
        canonical
    }

    fn mapped_source_for(
        &mut self,
        source: crate::SourceId,
        map: &mxx_ir_core::IndexMap,
        shape: Vec<usize>,
        env: Option<&ParamEnv>,
    ) -> crate::SourceId {
        if self.gadget_sources.values().any(|gadget| *gadget == source) {
            return source;
        }
        if let Some(parent) = self.source_lineages.get(&source).cloned() {
            if let Some(leaves) = map_source_leaves(&parent, map, &shape, env) {
                return self
                    .canonical_source_for_lineage(SourceLineage { shape: shape.clone(), leaves });
            }
        }
        let key = (source, map.normalize(), shape);
        if let Some(mapped) = self.mapped_sources.get(&key) {
            return *mapped;
        }
        let mapped = crate::SourceId(self.next_source);
        self.next_source = self.next_source.saturating_add(1);
        self.mapped_sources.insert(key, mapped);
        self.register_lineage(mapped, SourceLineage { shape: Vec::new(), leaves: vec![mapped] });
        mapped
    }

    fn gathered_source_for(
        &mut self,
        source: crate::SourceId,
        selectors: Vec<crate::SelectorId>,
        shape: Vec<usize>,
    ) -> crate::SourceId {
        self.gathered_source_for_concrete(source, selectors, shape, None)
    }

    fn source_after_axis_selection(
        &mut self,
        source: crate::SourceId,
        relation_bearing: bool,
        axis: usize,
        family_rank: usize,
        selectors: Vec<crate::SelectorId>,
        shape: Vec<usize>,
    ) -> crate::SourceId {
        if relation_bearing && axis + 1 == family_rank {
            source
        } else {
            self.gathered_source_for(source, selectors, shape)
        }
    }

    fn gathered_source_for_concrete(
        &mut self,
        source: crate::SourceId,
        selectors: Vec<crate::SelectorId>,
        shape: Vec<usize>,
        indices: Option<&[usize]>,
    ) -> crate::SourceId {
        if self.gadget_sources.values().any(|gadget| *gadget == source) {
            return source;
        }
        // A shared-source relation is defined only on its group axes.  A
        // preimage/target family may add a final digit branch axis, but that
        // branch must not enter the source function.  The source lineage's
        // rank is the canonical boundary, so opaque selectors beyond it are
        // ignored without inspecting their runtime values.
        let source_rank = self.source_lineages.get(&source).map(|lineage| lineage.shape.len());
        let (selectors, indices) = if let Some(rank) = source_rank {
            (
                selectors.into_iter().take(rank).collect(),
                indices.map(|indices| &indices[..rank.min(indices.len())]),
            )
        } else {
            (selectors, indices)
        };
        // A concrete dynamic get is just as canonical as a static map.  This
        // matters for loop-unrolled bodies: `family.get(loop_index)` carries
        // a singleton index in the current environment and must select the
        // corresponding source leaf instead of creating an opaque gathered
        // identity.
        if let (Some(parent), Some(indices)) = (self.source_lineages.get(&source).cloned(), indices)
        {
            if indices.len() == parent.shape.len() &&
                indices.iter().enumerate().all(|(axis, index)| *index < parent.shape[axis])
            {
                let flat =
                    indices.iter().zip(&parent.shape).fold(0usize, |flat, (index, extent)| {
                        flat.saturating_mul(*extent).saturating_add(*index)
                    });
                if let Some(leaf) = parent.leaves.get(flat) {
                    let count = shape.iter().copied().product::<usize>().max(1);
                    return self.canonical_source_for_lineage(SourceLineage {
                        shape: shape.clone(),
                        leaves: vec![*leaf; count],
                    });
                }
            }
        }
        // A selector cannot distinguish coordinates when the source function
        // is uniform.  Preserve that fact even though the selector value is
        // intentionally opaque; this is the common broadcast case for
        // uniform state and relation families.
        if let Some(parent) = self.source_lineages.get(&source).cloned() {
            if let Some(lineage) = uniform_gathered_lineage(&parent, &shape) {
                return self.canonical_source_for_lineage(lineage);
            }
        }
        self.gathered_source_fallback(source, selectors, shape)
    }

    fn gathered_source_fallback(
        &mut self,
        source: crate::SourceId,
        selectors: Vec<crate::SelectorId>,
        shape: Vec<usize>,
    ) -> crate::SourceId {
        let key = (source, selectors, shape);
        if let Some(mapped) = self.gathered_sources.get(&key) {
            return *mapped;
        }
        let mapped = crate::SourceId(self.next_source);
        self.next_source = self.next_source.saturating_add(1);
        self.gathered_sources.insert(key, mapped);
        self.register_lineage(mapped, SourceLineage { shape: Vec::new(), leaves: vec![mapped] });
        mapped
    }

    fn remap_target_with_map(
        &mut self,
        old_target: crate::FamilyViewId,
        map: &mxx_ir_core::IndexMap,
        shape: Vec<usize>,
        env: Option<&ParamEnv>,
    ) -> Option<MatrixState> {
        let mut state = match self.states.get(&old_target).cloned() {
            Some(state) => state,
            None => return None,
        };
        state.right_carrier = state.right_carrier.map(|carrier| crate::RightCarrier {
            source: self.mapped_source_for(carrier.source, map, shape.clone(), env),
            left_gain: carrier.left_gain,
        });
        Some(state)
    }

    fn remap_target_with_selectors(
        &mut self,
        old_target: crate::FamilyViewId,
        selectors: Vec<crate::SelectorId>,
        shape: Vec<usize>,
    ) -> Option<MatrixState> {
        let mut state = match self.states.get(&old_target).cloned() {
            Some(state) => state,
            None => return None,
        };
        state.right_carrier = state.right_carrier.map(|carrier| crate::RightCarrier {
            source: self.gathered_source_for(carrier.source, selectors, shape),
            left_gain: carrier.left_gain,
        });
        Some(state)
    }

    fn remap_target_after_axis_selection(
        &mut self,
        old_target: crate::FamilyViewId,
        axis: usize,
        selectors: Vec<crate::SelectorId>,
        shape: Vec<usize>,
    ) -> Option<MatrixState> {
        let mut state = self.states.get(&old_target).cloned()?;
        state.right_carrier = state.right_carrier.map(|carrier| {
            let source = self
                .source_lineages
                .get(&carrier.source)
                .and_then(|lineage| uniform_axis_selection_lineage(lineage, axis, &shape))
                .map(|lineage| self.canonical_source_for_lineage(lineage))
                .unwrap_or_else(|| self.gathered_source_for(carrier.source, selectors, shape));
            crate::RightCarrier { source, left_gain: carrier.left_gain }
        });
        Some(state)
    }

    /// Lift a representative grid-body source back to the symbolic family
    /// shape.  Gather/reindex provenance is retained; only a genuinely
    /// primitive scalar is broadcast.
    fn lift_source_for_shape(
        &mut self,
        source: crate::SourceId,
        shape: Vec<usize>,
    ) -> crate::SourceId {
        // A gadget denotes one public, index-independent preimage source.
        // Unlike a trapdoor/public sample created inside the grid body, it is
        // not resampled per lane: every family coordinate refers to the same G.
        let gadget_leaf = self.source_lineages.get(&source).and_then(|lineage| {
            let first = *lineage.leaves.first()?;
            (lineage.leaves.iter().all(|leaf| *leaf == first) &&
                self.gadget_sources.values().any(|gadget| *gadget == first))
            .then_some(first)
        });
        if let Some(gadget) = gadget_leaf.or_else(|| {
            self.gadget_sources.values().any(|gadget| *gadget == source).then_some(source)
        }) {
            let count = shape.iter().copied().product::<usize>().max(1);
            return self
                .canonical_source_for_lineage(SourceLineage { shape, leaves: vec![gadget; count] });
        }
        if let Some((parent, selectors, _)) = self
            .gathered_sources
            .iter()
            .find(|(_, mapped)| **mapped == source)
            .map(|((parent, selectors, shape), _)| (*parent, selectors.clone(), shape.clone()))
        {
            return self.gathered_source_for(parent, selectors, shape);
        }
        if let Some((parent, map, _)) = self
            .mapped_sources
            .iter()
            .find(|(_, mapped)| **mapped == source)
            .map(|((parent, map, shape), _)| (*parent, map.clone(), shape.clone()))
        {
            return self.mapped_source_for(parent, &map, shape, None);
        }
        // A canonical family source may already have been interned from a
        // pointwise pack, so it is not present in `mapped_sources` even
        // though all of its leaves are primitive samples.  It is still a
        // representative produced under the binder and needs the same
        // coordinate lifting as a scalar primitive.  Non-primitive leaves
        // retain their existing structural source function.
        let primitive_lineage = self.source_lineages.get(&source).is_some_and(|lineage| {
            lineage.leaves.iter().all(|leaf| self.sources.values().any(|value| value == leaf))
        });
        if !primitive_lineage && self.source_lineages.contains_key(&source) {
            return self.group_source_for(vec![source], shape);
        }
        // A primitive produced inside a ParallelGrid body is
        // evaluated once with representative binder indices.  Do not turn
        // that representative into a broadcast source: freezing the body
        // must expose one source leaf per structural lane.  This is keyed by
        // the representative source, so paired public/trapdoor outputs share
        // the exact same lane identities.
        let count = shape.iter().copied().product::<usize>().max(1);
        if count == 1 {
            return source;
        }
        let leaves = (0..count)
            .map(|lane| {
                let key = (source, shape.clone(), lane);
                if let Some(lifted) = self.binder_sources.get(&key) {
                    return *lifted;
                }
                let lifted = crate::SourceId(self.next_source);
                self.next_source = self.next_source.saturating_add(1);
                self.binder_sources.insert(key, lifted);
                self.register_lineage(
                    lifted,
                    SourceLineage { shape: Vec::new(), leaves: vec![lifted] },
                );
                lifted
            })
            .collect();
        self.canonical_source_for_lineage(SourceLineage { shape, leaves })
    }

    fn scope(
        &mut self,
        stage: &crate::StageId,
        graph: &Graph,
        sid: &FrozenGraphScopeId,
        occurrence: &[String],
        env: ParamEnv,
        mut vals: HashMap<WireRef, Info>,
    ) -> Result<HashMap<WireRef, Info>, SimulationError> {
        let scope = graph
            .scope(sid)
            .ok_or_else(|| SimulationError::InvalidGraph {
                message: "missing graph scope".into(),
                site: None,
            })?
            .clone();
        // Structural child inputs are preloaded and their Input nodes are skipped below.
        // Register the same numeric and relation facts that an evaluated producer would
        // register, including grid-specialized views used as decomposition targets.
        for value in vals.values() {
            if let Some(relation) = &value.relation {
                if let Some(existing) = self.preimages.get(&value.view) &&
                    existing != relation
                {
                    return Err(SimulationError::Relation {
                        message: "conflicting preimage relation on a structural child input".into(),
                        site: None,
                    });
                }
                self.preimages.insert(value.view, relation.clone());
            }
            if let Some(state) = matrix_state(&value.value) {
                self.states.insert(value.view, state);
            }
        }
        for (n, node) in scope.nodes().iter().enumerate() {
            let reachable = node.output_types().iter().enumerate().any(|(port, _)| {
                self.reached.contains(&crate::plan::PlannedWire {
                    stage: stage.clone(),
                    scope: sid.clone(),
                    occurrence: occurrence.to_vec(),
                    wire: WireRef {
                        node: mxx_ir_core::NodeId(n as u64),
                        port: mxx_ir_core::Port(port as u32),
                    },
                })
            });
            if !reachable {
                continue;
            }
            // Structural calls preload child input wires.  Those wires are
            // already concrete values, so the child Input node must not try
            // to resolve them as root external facts again.
            if matches!(node.kind(), NodeKind::Input { .. }) &&
                vals.contains_key(&WireRef {
                    node: mxx_ir_core::NodeId(n as u64),
                    port: mxx_ir_core::Port(0),
                })
            {
                continue;
            }
            let args = scope.arguments(node).ok_or_else(|| SimulationError::InvalidGraph {
                message: "foreign argument".into(),
                site: None,
            })?;
            let inputs = args
                .iter()
                .map(|x| {
                    vals.get(x).cloned().ok_or_else(|| SimulationError::InvalidGraph {
                        message: format!("unavailable wire {x:?}"),
                        site: None,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut out = self
                .node(stage, graph, sid, occurrence, &scope, n, node.kind(), &inputs, &env)
                .map_err(|error| SimulationError::InvalidGraph {
                    message: format!("stage {:?}, node {n} ({:?}): {error}", stage, node.kind()),
                    site: Some(DiagnosticSite {
                        stage: Some(stage.clone()),
                        occurrence: occurrence.to_vec(),
                        node: Some(mxx_ir_core::NodeId(n as u64)),
                        port: Some(mxx_ir_core::Port(0)),
                        operation: Some(format!("{:?}", node.kind())),
                    }),
                })?;
            // Views must be assigned to all ports before pairing is attached:
            // the public output of a two-port sampler is often port 0 while
            // the trapdoor is port 1.
            for (port, value) in out.iter_mut().enumerate() {
                if value.view == crate::FamilyViewId(u32::MAX) {
                    value.view = self.view_for_wire(
                        stage,
                        sid,
                        occurrence,
                        WireRef {
                            node: mxx_ir_core::NodeId(n as u64),
                            port: mxx_ir_core::Port(port as u32),
                        },
                        node.output_types().get(port),
                        node.kind(),
                        &inputs,
                        &env,
                    )?;
                }
            }
            // Attach pair facts only after all output views are frozen.  This
            // keeps sampler pairing local and avoids any raw-wire side table.
            if matches!(
                node.kind(),
                NodeKind::GadgetTrapdoor { .. } | NodeKind::TrapdoorSample { .. }
            ) && out.len() >= 2
            {
                let public = out[0].view;
                out[1].paired_public = Some(public);
            }
            if matches!(node.kind(), NodeKind::FamilyPack { .. }) && out.len() == 1 {
                if let Some(pair) =
                    inputs.iter().map(|input| input.paired_public).collect::<Option<Vec<_>>>()
                {
                    let shape = value_family_shape(&out[0].value).unwrap_or_default();
                    out[0].paired_public =
                        Some(self.interners.intern_composed_view(pair, shape, &[]));
                }
            }
            for (port, mut value) in out.into_iter().enumerate() {
                if !matches!(node.kind(), NodeKind::ParallelGrid(_)) &&
                    let AbstractValue::Family(family) = &mut value.value &&
                    let AbstractValue::Matrix(matrix) = family.element.as_mut() &&
                    matrix.right_carrier.is_some()
                {
                    let relation_source = value.relation.as_ref().map(|r| r.source);
                    matrix.right_carrier =
                        matrix.right_carrier.take().map(|carrier| crate::RightCarrier {
                            source: self
                                .group_source_for(vec![carrier.source], family.shape.clone()),
                            left_gain: carrier.left_gain,
                        });
                    if let (Some(relation), Some(source)) =
                        (value.relation.as_mut(), relation_source)
                    {
                        relation.source = self.group_source_for(vec![source], family.shape.clone());
                    }
                }
                if let Some(mut relation) = value.relation.take() {
                    let view = value.view;
                    relation.view = Some(view);
                    // A relation-bearing alias keeps its producer view;
                    // newly produced preimages are keyed by this exact
                    // canonical view rather than by a display or node label.
                    if let Some(existing) = self.preimages.get(&view) &&
                        existing != &relation
                    {
                        return Err(SimulationError::Relation {
                            message: "duplicate preimage relation for one canonical value view"
                                .into(),
                            site: None,
                        });
                    }
                    self.preimages.insert(view, relation.clone());
                    value.relation = Some(relation);
                }
                if let Some(state) = matrix_state(&value.value) {
                    self.states.insert(value.view, state);
                }
                vals.insert(
                    WireRef {
                        node: mxx_ir_core::NodeId(n as u64),
                        port: mxx_ir_core::Port(port as u32),
                    },
                    value,
                );
            }
            self.transfers += 1;
            if self.request.limits.maximum_transfer_steps.is_some_and(|x| self.transfers > x) {
                return Err(SimulationError::ResourceLimitExceeded {
                    message: "maximum transfer steps exceeded".into(),
                    site: None,
                });
            }
        }
        Ok(vals)
    }

    fn view_for_wire(
        &mut self,
        stage: &crate::StageId,
        scope: &FrozenGraphScopeId,
        occurrence: &[String],
        wire: WireRef,
        ty: Option<&WireType>,
        kind: &NodeKind,
        inputs: &[Info],
        env: &ParamEnv,
    ) -> Result<crate::FamilyViewId, SimulationError> {
        let key = crate::identity::ValueKey {
            stage: stage.clone(),
            scope: scope.clone(),
            occurrence: occurrence.to_vec(),
            wire,
        };
        let next = crate::ValueId(self.interners.values.len() as u32);
        let value = *self.interners.values.entry(key).or_insert(next);
        let shape = ty
            .and_then(|ty| match kind {
                _ => family_shape(ty, env),
            })
            .unwrap_or_default();
        let composed = match kind {
            NodeKind::FamilyPack { .. } => Some(self.interners.intern_composed_view(
                inputs.iter().map(|input| input.view).collect(),
                shape.clone(),
                &[],
            )),
            NodeKind::FamilyReindex { map, .. } => inputs.first().map(|input| {
                self.interners.intern_composed_view(
                    vec![input.view],
                    shape.clone(),
                    std::slice::from_ref(map),
                )
            }),
            NodeKind::FamilyGather { .. } |
            NodeKind::FamilyGetDynamic { .. } |
            NodeKind::FamilySelectAxis { .. } => Some(self.interners.intern_composed_view(
                inputs.iter().map(|input| input.view).collect(),
                shape.clone(),
                &[],
            )),
            NodeKind::FamilyGetStatic { indices } => {
                let map = mxx_ir_core::IndexMap::new(indices.clone());
                Some(self.interners.intern_composed_view(
                    inputs.iter().map(|input| input.view).collect(),
                    shape.clone(),
                    std::slice::from_ref(&map),
                ))
            }
            _ => None,
        };
        let view = composed.unwrap_or_else(|| self.interners.intern_view(vec![value], shape, &[]));
        Ok(view)
    }

    #[allow(clippy::too_many_arguments)]
    fn node(
        &mut self,
        stage: &crate::StageId,
        graph: &Graph,
        sid: &FrozenGraphScopeId,
        occurrence: &[String],
        scope: &GraphScope,
        n: usize,
        kind: &NodeKind,
        xs: &[Info],
        env: &ParamEnv,
    ) -> Result<Vec<Info>, SimulationError> {
        let site = || {
            Some(DiagnosticSite {
                stage: Some(stage.clone()),
                occurrence: occurrence
                    .iter()
                    .cloned()
                    .chain(std::iter::once(format!("scope={sid:?}")))
                    .collect(),
                node: Some(mxx_ir_core::NodeId(n as u64)),
                port: Some(mxx_ir_core::Port(0)),
                operation: Some(format!("{kind:?}")),
            })
        };
        let bad = |m: &str| SimulationError::InvalidGraph { message: m.into(), site: site() };
        let mt = |x: &Info| x.ty.clone().ok_or_else(|| bad("matrix type is unavailable"));
        let matrix = |x: &Info| matrix_state(&x.value).ok_or_else(|| bad("matrix value required"));
        match kind {
            NodeKind::Input { name, artifact: None, .. } => {
                Ok(vec![self.input(stage, graph, sid, occurrence, n, name, env)?])
            }
            NodeKind::Input { artifact: Some(a), wire_type, .. } => {
                let p = self
                    .request
                    .program
                    .stages
                    .iter()
                    .find(|s| s.production_id == a.production_id)
                    .ok_or_else(|| SimulationError::ArtifactResolution {
                        message: "artifact producer missing".into(),
                        site: site(),
                    })?;
                let x = self.stage(&p.id)?.get(&a.artifact_name).cloned().ok_or_else(|| {
                    SimulationError::ArtifactResolution {
                        message: "artifact output missing".into(),
                        site: site(),
                    }
                })?;
                if a.confidentiality == ArtifactConfidentiality::Private &&
                    p.graph.outputs().get(&a.artifact_name).and_then(|o| o.confidentiality) !=
                        Some(ArtifactConfidentiality::Private)
                {
                    return Err(SimulationError::ArtifactResolution {
                        message: "artifact confidentiality mismatch".into(),
                        site: site(),
                    });
                }
                let produced_output = p.graph.outputs().get(&a.artifact_name).ok_or_else(|| {
                    SimulationError::ArtifactResolution {
                        message: "artifact producer output missing".into(),
                        site: site(),
                    }
                })?;
                let produced_type = p
                    .graph
                    .root_scope()
                    .node(produced_output.value.node)
                    .and_then(|node| node.output_types().get(produced_output.value.port.0 as usize))
                    .ok_or_else(|| SimulationError::ArtifactResolution {
                        message: "artifact producer type missing".into(),
                        site: site(),
                    })?;
                if !wire_types_compatible(wire_type, produced_type, env) {
                    return Err(SimulationError::ArtifactResolution {
                        message: "artifact type or concrete shape mismatch".into(),
                        site: site(),
                    });
                }
                Ok(vec![x])
            }
            NodeKind::ConstantInt(v) => Ok(vec![integer(v.clone())]),
            NodeKind::EvaluateInt(v) => {
                Ok(vec![integer(v.evaluate(env).map_err(|e| bad(&e.to_string()))?)])
            }
            NodeKind::ConstantMatrix { matrix_type, value } => {
                let t = concrete_matrix(&WireType::Matrix(matrix_type.clone()), env)
                    .ok_or_else(|| bad("invalid matrix type"))?;
                let mut z = match value {
                    mxx_ir_core::node::ConstantMatrix::Zero => {
                        state::exact_matrix(&t, 0u8.into(), true)?
                    }
                    mxx_ir_core::node::ConstantMatrix::Identity |
                    mxx_ir_core::node::ConstantMatrix::UnitRow { .. } |
                    mxx_ir_core::node::ConstantMatrix::UnitColumn { .. } => {
                        state::exact_matrix(&t, 1u8.into(), true)?
                    }
                    mxx_ir_core::node::ConstantMatrix::Gadget { base, small } => {
                        let base = base.evaluate(env).map_err(|e| bad(&e.to_string()))?;
                        if t.rows == 0 || t.columns == 0 || !t.columns.is_multiple_of(t.rows) {
                            return Err(bad("gadget matrix dimensions are incompatible"));
                        }
                        let digits = t.columns / t.rows;
                        let mut gadget = state::gadget_matrix(&t, &base, digits)?;
                        if *small {
                            gadget.coefficient_magnitude_bound =
                                crate::centered_residue_bound(&t.modulus)?.min(1u8.into());
                        }
                        gadget
                    }
                    mxx_ir_core::node::ConstantMatrix::PowerOfBase { base, exponent } => {
                        let base = base.evaluate(env).map_err(|e| bad(&e.to_string()))?;
                        let exponent = exponent
                            .evaluate(env)
                            .map_err(|e| bad(&e.to_string()))?
                            .to_u32()
                            .ok_or_else(|| bad("invalid power-of-base exponent"))?;
                        let base = base
                            .abs()
                            .to_biguint()
                            .ok_or_else(|| bad("invalid power-of-base base"))?;
                        state::exact_matrix(&t, base.pow(exponent), true)?
                    }
                    mxx_ir_core::node::ConstantMatrix::Rotation { .. } => {
                        state::exact_matrix(&t, crate::centered_residue_bound(&t.modulus)?, false)?
                    }
                    mxx_ir_core::node::ConstantMatrix::Polynomial { coefficients } => {
                        let evaluated = coefficients
                            .iter()
                            .map(|coefficient| {
                                coefficient.evaluate(env).map_err(|e| bad(&e.to_string()))
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let magnitude = evaluated
                            .iter()
                            .map(|coefficient| coefficient.abs().to_biguint().unwrap_or_default())
                            .max()
                            .unwrap_or_default();
                        let constant = evaluated.iter().skip(1).all(Zero::is_zero);
                        state::exact_matrix(&t, magnitude, constant)?
                    }
                };
                let source =
                    if let mxx_ir_core::node::ConstantMatrix::Gadget { base, small } = value {
                        let base = base.evaluate(env).map_err(|e| bad(&e.to_string()))?;
                        let descriptor = crate::GadgetDescriptor {
                            modulus: t.modulus.clone(),
                            ring_dimension: t.ring_dimension,
                            rows: t.rows,
                            columns: t.columns,
                            base,
                            digit_count: t.columns / t.rows.max(1),
                            small: *small,
                        };
                        if let Some(source) = self.gadget_sources.get(&descriptor) {
                            *source
                        } else {
                            let source = self.source_for(stage, sid, occurrence, n, "constant");
                            self.gadget_sources.insert(descriptor, source);
                            source
                        }
                    } else {
                        self.source_for(stage, sid, occurrence, n, "constant")
                    };
                z.right_carrier = Some(crate::RightCarrier { source, left_gain: 1u8.into() });
                Ok(vec![Info {
                    value: AbstractValue::Matrix(z),
                    ty: Some(t),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::GadgetTrapdoor { matrix_type, base } => {
                let t = concrete_matrix(&WireType::Matrix(matrix_type.clone()), env)
                    .ok_or_else(|| bad("invalid gadget type"))?;
                let b = base.evaluate(env).map_err(|e| bad(&e.to_string()))?;
                let digits = t.columns.checked_div(t.rows).unwrap_or(1);
                let mut p = state::gadget_matrix(&t, &b, digits)?;
                let source = self.source_for(stage, sid, occurrence, n, "gadget");
                p.right_carrier = Some(crate::RightCarrier { source, left_gain: 1u8.into() });
                Ok(vec![
                    Info {
                        value: AbstractValue::Matrix(p),
                        ty: Some(t.clone()),
                        relation: None,
                        view: crate::FamilyViewId(u32::MAX),
                        paired_public: None,
                    },
                    Info {
                        value: AbstractValue::Trapdoor(TrapdoorState {
                            matrix: t,
                            sigma: mxx_ir_core::RealExpr::FromInt(b.clone().into()),
                            gadget_base: b,
                            digit_count: digits,
                            preimage_max_coefficient_bound: 0.into(),
                        }),
                        ty: None,
                        relation: None,
                        view: crate::FamilyViewId(u32::MAX),
                        paired_public: None,
                    },
                ])
            }
            NodeKind::ConstantReal(_) |
            NodeKind::IntToReal |
            NodeKind::RealBinary(_) |
            NodeKind::RealSqrt => Err(SimulationError::Unsupported {
                operation: "real-valued node transfer".into(),
                site: site(),
            }),
            NodeKind::ConstantBool(value) => Ok(vec![Info {
                value: AbstractValue::Boolean(if *value {
                    state::BooleanState::TrueOnly
                } else {
                    state::BooleanState::FalseOnly
                }),
                ty: None,
                relation: None,
                view: crate::FamilyViewId(u32::MAX),
                paired_public: None,
            }]),
            NodeKind::IntCompare(op) => {
                let a = int(&xs[0])?;
                let b = int(&xs[1])?;
                let value = match op {
                    mxx_ir_core::node::IntCompareOp::Equal
                        if a.maximum_inclusive < b.minimum || b.maximum_inclusive < a.minimum =>
                    {
                        state::BooleanState::FalseOnly
                    }
                    mxx_ir_core::node::IntCompareOp::Equal
                        if a.minimum == a.maximum_inclusive &&
                            b.minimum == b.maximum_inclusive &&
                            a.minimum == b.minimum =>
                    {
                        state::BooleanState::TrueOnly
                    }
                    mxx_ir_core::node::IntCompareOp::LessEqual
                        if a.maximum_inclusive <= b.minimum =>
                    {
                        state::BooleanState::TrueOnly
                    }
                    mxx_ir_core::node::IntCompareOp::LessEqual
                        if a.minimum > b.maximum_inclusive =>
                    {
                        state::BooleanState::FalseOnly
                    }
                    _ => state::BooleanState::Either,
                };
                Ok(vec![Info {
                    value: AbstractValue::Boolean(value),
                    ty: None,
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::BitExtract { .. } => Ok(vec![Info {
                value: AbstractValue::Boolean(state::BooleanState::Either),
                ty: None,
                relation: None,
                view: crate::FamilyViewId(u32::MAX),
                paired_public: None,
            }]),
            NodeKind::BoolToInt => Ok(vec![Info {
                value: AbstractValue::Integer(match xs[0].value {
                    AbstractValue::Boolean(state::BooleanState::FalseOnly) => {
                        state::IntegerState::singleton(0)
                    }
                    AbstractValue::Boolean(state::BooleanState::TrueOnly) => {
                        state::IntegerState::singleton(1)
                    }
                    _ => state::IntegerState::new(0.into(), 1.into())?,
                }),
                ty: None,
                relation: None,
                view: crate::FamilyViewId(u32::MAX),
                paired_public: None,
            }]),
            NodeKind::IntBinary(op) => {
                let a = int(&xs[0])?;
                let b = int(&xs[1])?;
                let z = match op {
                    IntBinaryOp::Add => a.add(&b),
                    IntBinaryOp::Subtract => a.subtract(&b),
                    IntBinaryOp::Multiply => a.multiply(&b),
                    IntBinaryOp::Divide => a.divide(&b).map_err(|e| bad(&e.to_string()))?,
                    IntBinaryOp::Remainder => a.remainder(&b).map_err(|e| bad(&e.to_string()))?,
                };
                Ok(vec![Info {
                    value: AbstractValue::Integer(z),
                    ty: None,
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::MatrixBinary(op) => {
                let a = matrix(&xs[0])?;
                let b = matrix(&xs[1])?;
                let left_type = mt(&xs[0])?;
                let result_type = output_type(scope, n, env)?;
                let z = match op {
                    MatrixBinaryOp::Add => a.add(&b, &result_type.modulus)?,
                    MatrixBinaryOp::Subtract => a.subtract(&b, &result_type.modulus)?,
                    MatrixBinaryOp::Multiply => a.ordinary_product(
                        &b,
                        ProductGeometry {
                            inner_dimension: left_type.columns,
                            ring_dimension: left_type.ring_dimension,
                        },
                        &result_type.modulus,
                    )?,
                };
                // Addition/subtraction can lose either input when distinct
                // sources are combined.  Ordinary multiplication is
                // different: A * B retains B's rightmost carrier by
                // construction, while A's carrier is intentionally consumed
                // by the left action, so it is not a diagnostic drop.
                if !matches!(op, MatrixBinaryOp::Multiply) {
                    self.record_carrier_drops(
                        xs,
                        Some(&z),
                        site(),
                        "carrier lost by matrix binary arithmetic",
                    );
                }
                Ok(vec![Info {
                    value: AbstractValue::Matrix(z),
                    ty: Some(result_type),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::ApplyPreimage => {
                let a = matrix(&xs[0])?;
                let b = matrix(&xs[1])?;
                let left_type = mt(&xs[0])?;
                let result_type = output_type(scope, n, env)?;
                let view = xs[1].relation.as_ref().and_then(|r| r.view).ok_or_else(|| {
                    SimulationError::Relation {
                        message: format!(
                            "explicit preimage use has no relation (value={:?}, relation={:?}, view={:?})",
                            xs[1].value, xs[1].relation, xs[1].view
                        ),
                        site: site(),
                    }
                })?;
                let r = self.preimages.get(&view).ok_or_else(|| SimulationError::Relation {
                    message: "explicit preimage relation is not registered".into(),
                    site: site(),
                })?;
                if a.right_carrier.as_ref().is_some_and(|carrier| carrier.source != r.source) {
                    return Err(SimulationError::Relation {
                        message: format!(
                            "preimage relation source mismatch: expected {:?} for relation view {:?} with lineage {:?} (origin {:?}), actual {:?} for left view {:?} with lineage {:?} (origin {:?})",
                            r.source,
                            r.view,
                            self.source_lineages.get(&r.source),
                            self.source_origin(r.source),
                            a.right_carrier.as_ref().map(|carrier| carrier.source),
                            xs[0].view,
                            a.right_carrier
                                .as_ref()
                                .and_then(|carrier| self.source_lineages.get(&carrier.source)),
                            a.right_carrier
                                .as_ref()
                                .map(|carrier| self.source_origin(carrier.source)),
                        ),
                        site: site(),
                    });
                }
                let z = relation::consume(
                    &a,
                    &b,
                    self.states.get(&r.target).ok_or_else(|| SimulationError::Relation {
                        message: format!("preimage target state is unavailable target={:?} relation={:?} view={:?} occurrence={:?}", r.target, r, xs[1].view, occurrence),
                        site: site(),
                    })?,
                    r,
                    xs[1].view,
                    r.target,
                    ProductGeometry {
                        inner_dimension: left_type.columns,
                        ring_dimension: left_type.ring_dimension,
                    },
                    &result_type.modulus,
                )
                .map_err(|e| SimulationError::Relation {
                    message: format!(
                        "preimage relation error: {e}; relation={r:?}, preimage_view={:?}, left_view={:?}, target_state_view={:?}, target_state_available={}, left_carrier={:?}, left_state={a:?}, preimage_state={b:?}",
                        xs[1].view,
                        xs[0].view,
                        r.target,
                        self.states.contains_key(&r.target),
                        a.right_carrier,
                    ),
                    site: site(),
                })?;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(z),
                    ty: Some(result_type),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::MatrixNegate => {
                let t = mt(&xs[0])?;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(matrix(&xs[0])?.negate(&t.modulus)?),
                    ty: Some(t),
                    relation: xs[0].relation.clone(),
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::MatrixScale { scalar } => {
                let t = mt(&xs[0])?;
                let s = scalar.evaluate(env).map_err(|e| bad(&e.to_string()))?;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(matrix(&xs[0])?.scale(&s, &t.modulus)?),
                    ty: Some(t),
                    relation: xs[0].relation.clone(),
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::RingAutomorphism { .. } => {
                // A valid ring automorphism only permutes (and possibly negates)
                // polynomial coefficients.  It is therefore an exact isometry
                // for the coefficient error and magnitude bounds.  The source
                // carrier and any paired public view remain attached to the
                // same abstract value because the operation does not introduce
                // a new sampled value or discard the existing witness relation.
                let t = mt(&xs[0])?;
                let value = matrix(&xs[0])?;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(value),
                    ty: Some(t),
                    relation: xs[0].relation.clone(),
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: xs[0].paired_public,
                }])
            }
            NodeKind::Transpose | NodeKind::Slice { .. } => {
                let t = output_type(scope, n, env)?;
                let mut value = matrix(&xs[0])?;
                // Reindexing a matrix changes coefficient placement, so this
                // abstract state deliberately drops its source witness.
                self.record_carrier_drops(
                    xs,
                    None,
                    site(),
                    "carrier discarded by transpose or slice",
                );
                value.right_carrier = None;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(value),
                    ty: Some(t),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::Concat { .. } => {
                if xs.is_empty() {
                    return Err(bad("concat requires at least one matrix"));
                }
                let t = output_type(scope, n, env)?;
                let mut value = state::zero_matrix(&t)?;
                for input in xs {
                    let part = matrix(input)?;
                    value.error_bound = value.error_bound.max(part.error_bound);
                    value.coefficient_magnitude_bound =
                        value.coefficient_magnitude_bound.max(part.coefficient_magnitude_bound);
                    value.is_constant_polynomial &= part.is_constant_polynomial;
                }
                value.coefficient_magnitude_bound = value
                    .coefficient_magnitude_bound
                    .min(crate::centered_residue_bound(&t.modulus)?);
                // Concatenation summarizes several placements into one
                // matrix and has no sound single-source carrier witness.
                self.record_carrier_drops(
                    xs,
                    None,
                    site(),
                    "carrier discarded by matrix concatenation",
                );
                value.right_carrier = None;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(value),
                    ty: Some(t),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::Tensor => {
                let left = matrix(&xs[0])?;
                let right = matrix(&xs[1])?;
                let t = output_type(scope, n, env)?;
                let mut value = left.ordinary_product(
                    &right,
                    ProductGeometry { inner_dimension: 1, ring_dimension: t.ring_dimension },
                    &t.modulus,
                )?;
                // Tensor products combine independent coefficient layouts and
                // intentionally clear the result carrier.  Only the right
                // operand could have been the product's retained source; the
                // left operand is consumed by the left action and is not a
                // meaningful drop.
                self.record_carrier_drops_iter(
                    std::iter::once(&xs[1]),
                    None,
                    site(),
                    "carrier discarded by tensor product",
                );
                value.right_carrier = None;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(value),
                    ty: Some(t),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::MatrixMulAccumulate { coefficients, has_bias } => {
                let t = output_type(scope, n, env)?;
                let expected = coefficients.len() * 2 + usize::from(*has_bias);
                if xs.len() != expected {
                    return Err(bad("matrix accumulate arity mismatch"));
                }
                let mut value = if *has_bias {
                    matrix(&xs[coefficients.len() * 2])?
                } else {
                    state::zero_matrix(&t)?
                };
                for (product, coefficient) in coefficients.iter().enumerate() {
                    let left = matrix(&xs[2 * product])?;
                    let right = matrix(&xs[2 * product + 1])?;
                    let left_type = xs[2 * product]
                        .ty
                        .as_ref()
                        .ok_or_else(|| bad("matrix accumulate left type unavailable"))?;
                    let mut term = left.ordinary_product(
                        &right,
                        ProductGeometry {
                            inner_dimension: left_type.columns,
                            ring_dimension: left_type.ring_dimension,
                        },
                        &t.modulus,
                    )?;
                    let coefficient = coefficient.evaluate(env).map_err(|e| bad(&e.to_string()))?;
                    term = term.scale(&coefficient, &t.modulus)?;
                    value = value.add(&term, &t.modulus)?;
                }
                // The accumulated result is a bound over many products and
                // an optional bias, so no one input source can witness it.
                // Each product consumes its left factor while retaining its
                // right factor, and the bias is itself a directly retainable
                // input; report only those right/bias sources that are then
                // discarded by the aggregate.
                let retainable_inputs = coefficients
                    .iter()
                    .enumerate()
                    .map(|(product, _)| &xs[2 * product + 1])
                    .chain(has_bias.then(|| &xs[coefficients.len() * 2]));
                self.record_carrier_drops_iter(
                    retainable_inputs,
                    None,
                    site(),
                    "carrier discarded by matrix multiply-accumulate",
                );
                value.right_carrier = None;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(value),
                    ty: Some(t),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::UniformResidueSample { .. } | NodeKind::HashSample { .. } => {
                let t = output_type(scope, n, env)?;
                let mut z = state::plain_hash_sample(&t)?;
                z.right_carrier = Some(crate::RightCarrier {
                    source: self.source_for(stage, sid, occurrence, n, "uniform-residue"),
                    left_gain: 1u8.into(),
                });
                Ok(vec![Info {
                    value: AbstractValue::Matrix(z),
                    ty: Some(t),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::UniformIntervalSample { range, .. } => {
                let t = output_type(scope, n, env)?;
                let mut z = state::uniform_interval_sample(
                    &t,
                    &range.minimum.evaluate(env).map_err(|e| bad(&e.to_string()))?,
                    &range.maximum.evaluate(env).map_err(|e| bad(&e.to_string()))?,
                )?;
                z.right_carrier = Some(crate::RightCarrier {
                    source: self.source_for(stage, sid, occurrence, n, "uniform-interval"),
                    left_gain: 1u8.into(),
                });
                Ok(vec![Info {
                    value: AbstractValue::Matrix(z),
                    ty: Some(t),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::GaussianSample { max_coefficient_bound, .. } => {
                let t = output_type(scope, n, env)?;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(state::gaussian_sample(
                        &t,
                        &max_coefficient_bound.evaluate(env).map_err(|e| bad(&e.to_string()))?,
                    )?),
                    ty: Some(t),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::TrapdoorSample {
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
                ..
            } => {
                let t = output_type(scope, n, env)?;
                let mut p = state::trapdoor_public_matrix(&t)?;
                p.right_carrier = Some(crate::RightCarrier {
                    source: self.source_for(stage, sid, occurrence, n, "trapdoor"),
                    left_gain: 1u8.into(),
                });
                Ok(vec![
                    Info {
                        value: AbstractValue::Matrix(p),
                        ty: Some(t.clone()),
                        relation: None,
                        view: crate::FamilyViewId(u32::MAX),
                        paired_public: None,
                    },
                    Info {
                        value: AbstractValue::Trapdoor(TrapdoorState {
                            matrix: t,
                            sigma: sigma.clone(),
                            gadget_base: gadget_base
                                .evaluate(env)
                                .map_err(|e| bad(&e.to_string()))?,
                            digit_count: digit_count
                                .evaluate(env)
                                .map_err(|e| bad(&e.to_string()))?
                                .to_usize()
                                .ok_or_else(|| bad("invalid digit count"))?,
                            preimage_max_coefficient_bound: preimage_max_coefficient_bound
                                .evaluate(env)
                                .map_err(|e| bad(&e.to_string()))?,
                        }),
                        ty: None,
                        relation: None,
                        view: crate::FamilyViewId(u32::MAX),
                        paired_public: None,
                    },
                ])
            }
            NodeKind::PreimageSample { max_coefficient_bound, .. } => {
                if xs.len() != 3 {
                    return Err(bad("preimage sampler arity mismatch"));
                }
                let p = matrix(&xs[0])?;
                let public_type = mt(&xs[0])?;
                if xs[1].paired_public != Some(xs[0].view) {
                    return Err(SimulationError::Relation {
                        message: format!(
                            "preimage sampler public input is not the trapdoor's paired wire (paired={:?}, public={:?})",
                            xs[1].paired_public, xs[0].view,
                        ),
                        site: site(),
                    });
                }
                let trapdoor = match &xs[1].value {
                    AbstractValue::Trapdoor(trapdoor) => trapdoor,
                    _ => return Err(bad("preimage sampler requires its matching trapdoor")),
                };
                if trapdoor.matrix != public_type {
                    return Err(SimulationError::Relation {
                        message: "preimage trapdoor and public source types do not match".into(),
                        site: site(),
                    });
                }
                if !p.error_bound.is_zero() {
                    return Err(SimulationError::Relation {
                        message: "direct public relation source must have zero error".into(),
                        site: site(),
                    });
                }
                let t = output_type(scope, n, env)?;
                let source =
                    p.right_carrier.as_ref().map(|carrier| carrier.source).ok_or_else(|| {
                        SimulationError::Relation {
                            message: "public source has no identity".into(),
                            site: site(),
                        }
                    })?;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(state::preimage_sample(
                        &t,
                        &max_coefficient_bound.evaluate(env).map_err(|e| bad(&e.to_string()))?,
                    )?),
                    ty: Some(t),
                    relation: Some(RightPreimage {
                        source,
                        target: xs[2].view,
                        view: None,
                        selector: None,
                    }),
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::FamilyPreimageSample { max_coefficient_bound, .. } => {
                if xs.len() != 3 {
                    return Err(bad("family preimage sampler arity mismatch"));
                }
                let p = matrix(&xs[0])?;
                let public_type = mt(&xs[0])?;
                let trapdoor_element = match &xs[1].value {
                    AbstractValue::Family(family) => family.element.as_ref(),
                    AbstractValue::Trapdoor(_) => &xs[1].value,
                    _ => return Err(bad("family preimage sampler requires a trapdoor")),
                };
                let trapdoor = match trapdoor_element {
                    AbstractValue::Trapdoor(trapdoor) => trapdoor,
                    _ => return Err(bad("family preimage sampler requires a trapdoor")),
                };
                if trapdoor.matrix != public_type {
                    return Err(SimulationError::Relation {
                        message: "family preimage trapdoor and public source types do not match"
                            .into(),
                        site: site(),
                    });
                }
                if xs[1].paired_public != Some(xs[0].view) {
                    return Err(SimulationError::Relation {
                        message:
                            "family preimage sampler public input is not paired with its trapdoor"
                                .into(),
                        site: site(),
                    });
                }
                let t = output_type(scope, n, env)?;
                let shape = output_family_shape(scope, n, env).unwrap_or_default();
                let public_shape = value_family_shape(&xs[0].value);
                let trapdoor_shape = value_family_shape(&xs[1].value);
                let target_shape = value_family_shape(&xs[2].value);
                let expected_group = target_shape
                    .as_ref()
                    .and_then(|shape| {
                        (!shape.is_empty()).then(|| shape[..shape.len() - 1].to_vec())
                    })
                    .unwrap_or_default();
                if public_shape != trapdoor_shape || public_shape != Some(expected_group) {
                    return Err(SimulationError::Relation {
                        message: "family preimage source and target group shapes do not match"
                            .into(),
                        site: site(),
                    });
                }
                if target_shape != Some(shape.clone()) {
                    return Err(SimulationError::Relation {
                        message: "family preimage output and target shapes do not match".into(),
                        site: site(),
                    });
                }
                let mut source =
                    p.right_carrier.as_ref().map(|carrier| carrier.source).ok_or_else(|| {
                        SimulationError::Relation {
                            message: "family public source has no identity".into(),
                            site: site(),
                        }
                    })?;
                // The relation contract applies to the actual public source
                // received by the sampler. Keep that identity registered even
                // when the grouped representation below interns a normalized
                // source for the preimage family.
                // FamilyPreimageSample consumes a group-indexed public
                // source.  Its output adds only the final preimage branch
                // axis, so retain the group's canonical source function and
                // never replace it with a representative scalar branch.
                if let Some(group_shape) = public_shape.clone() {
                    if self
                        .source_lineages
                        .get(&source)
                        .is_none_or(|lineage| lineage.shape != group_shape)
                    {
                        source = self.group_source_for(vec![source], group_shape);
                    }
                }
                Ok(vec![Info {
                    value: AbstractValue::Family(FamilyState::new(
                        shape,
                        AbstractValue::Matrix(state::preimage_sample(
                            &t,
                            &max_coefficient_bound
                                .evaluate(env)
                                .map_err(|e| bad(&e.to_string()))?,
                        )?),
                    )?),
                    ty: Some(t),
                    relation: Some(RightPreimage {
                        source,
                        target: xs[2].view,
                        view: None,
                        selector: None,
                    }),
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::GadgetDecompose { base, small, digit_count } => {
                let t = output_type(scope, n, env)?;
                let b = base.evaluate(env).map_err(|e| bad(&e.to_string()))?;
                let d = digit_count
                    .evaluate(env)
                    .map_err(|e| bad(&e.to_string()))?
                    .to_usize()
                    .ok_or_else(|| bad("invalid digit count"))?;
                let gadget_rows = t
                    .rows
                    .checked_div(d)
                    .filter(|rows| rows.saturating_mul(d) == t.rows)
                    .ok_or_else(|| bad("decomposition rows are not divisible by digit count"))?;
                let descriptor = crate::GadgetDescriptor {
                    modulus: t.modulus.clone(),
                    ring_dimension: t.ring_dimension,
                    rows: gadget_rows,
                    columns: t.rows,
                    base: b.clone(),
                    digit_count: d,
                    small: *small,
                };
                let source = if let Some(source) = self.gadget_sources.get(&descriptor) {
                    *source
                } else {
                    let source = self.source_for(stage, sid, occurrence, n, "gadget-decompose");
                    self.gadget_sources.insert(descriptor, source);
                    source
                };
                Ok(vec![Info {
                    value: AbstractValue::Matrix(state::gadget_decomposition(&t, &b, *small, d)?),
                    ty: Some(t),
                    relation: Some(RightPreimage {
                        source,
                        target: xs[0].view,
                        view: None,
                        selector: None,
                    }),
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::PreimageBinary(operation) => {
                let left_relation_view = xs[0]
                    .relation
                    .as_ref()
                    .and_then(|relation| relation.view)
                    .ok_or_else(|| bad("preimage algebra left relation is unavailable"))?;
                let left_relation = self
                    .preimages
                    .get(&left_relation_view)
                    .cloned()
                    .ok_or_else(|| bad("preimage algebra left relation is not registered"))?;
                let left = matrix(&xs[0])?;
                let right = matrix(&xs[1])?;
                let output_type = output_type(scope, n, env)?;
                let (value, target, target_parents) = match operation {
                    mxx_ir_core::node::PreimageBinaryOp::Add => {
                        let right_relation_view = xs[1]
                            .relation
                            .as_ref()
                            .and_then(|relation| relation.view)
                            .ok_or_else(|| bad("preimage sum right relation is unavailable"))?;
                        let right_relation =
                            self.preimages.get(&right_relation_view).cloned().ok_or_else(|| {
                                bad("preimage sum right relation is not registered")
                            })?;
                        if left_relation.source != right_relation.source {
                            return Err(bad("preimage sum requires a common source"));
                        }
                        let left_target = self
                            .states
                            .get(&left_relation.target)
                            .cloned()
                            .ok_or_else(|| bad("preimage sum left target is unavailable"))?;
                        let right_target = self
                            .states
                            .get(&right_relation.target)
                            .cloned()
                            .ok_or_else(|| bad("preimage sum right target is unavailable"))?;
                        (
                            left.add(&right, &output_type.modulus)?,
                            left_target.add(&right_target, &output_type.modulus)?,
                            vec![left_relation.target, right_relation.target],
                        )
                    }
                    mxx_ir_core::node::PreimageBinaryOp::RightMultiplyExact => {
                        if !right.error_bound.is_zero() {
                            return Err(bad("preimage right multiplier must be exact"));
                        }
                        let left_type = mt(&xs[0])?;
                        let geometry = ProductGeometry {
                            inner_dimension: left_type.columns,
                            ring_dimension: output_type.ring_dimension,
                        };
                        let left_target = self
                            .states
                            .get(&left_relation.target)
                            .cloned()
                            .ok_or_else(|| bad("preimage product target is unavailable"))?;
                        (
                            left.ordinary_product(&right, geometry, &output_type.modulus)?,
                            left_target.ordinary_product(&right, geometry, &output_type.modulus)?,
                            vec![left_relation.target, xs[1].view],
                        )
                    }
                    mxx_ir_core::node::PreimageBinaryOp::ComposeExactDecomposition => {
                        let right_relation_view =
                            xs[1].relation.as_ref().and_then(|relation| relation.view).ok_or_else(
                                || bad("composed decomposition relation is unavailable"),
                            )?;
                        let right_relation = self
                            .preimages
                            .get(&right_relation_view)
                            .cloned()
                            .ok_or_else(|| bad("composed decomposition is not registered"))?;
                        let right_target = self
                            .states
                            .get(&right_relation.target)
                            .ok_or_else(|| bad("composed decomposition target is unavailable"))?;
                        if !right_target.error_bound.is_zero() {
                            return Err(bad("composed decomposition target must be exact"));
                        }
                        let left_type = mt(&xs[0])?;
                        let geometry = ProductGeometry {
                            inner_dimension: left_type.columns,
                            ring_dimension: output_type.ring_dimension,
                        };
                        let left_target = self
                            .states
                            .get(&left_relation.target)
                            .cloned()
                            .ok_or_else(|| bad("preimage composition target is unavailable"))?;
                        (
                            left.ordinary_product(&right, geometry, &output_type.modulus)?,
                            left_target.ordinary_product(&right, geometry, &output_type.modulus)?,
                            vec![left_relation.target, xs[1].view],
                        )
                    }
                };
                let target_view =
                    self.interners.intern_composed_view(target_parents, Vec::new(), &[]);
                self.states.insert(target_view, target);
                Ok(vec![Info {
                    value: AbstractValue::Matrix(value),
                    ty: Some(output_type),
                    relation: Some(RightPreimage {
                        source: left_relation.source,
                        target: target_view,
                        view: None,
                        selector: None,
                    }),
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::PreimageConcatColumns => {
                if xs.is_empty() {
                    return Err(bad("preimage concat requires at least one input"));
                }
                let output_type = output_type(scope, n, env)?;
                let mut value = state::zero_matrix(&output_type)?;
                let mut target = state::zero_matrix(&output_type)?;
                let mut source = None;
                let mut target_views = Vec::with_capacity(xs.len());
                for input in xs {
                    let relation_view = input
                        .relation
                        .as_ref()
                        .and_then(|relation| relation.view)
                        .ok_or_else(|| bad("preimage concat relation is unavailable"))?;
                    let relation = self
                        .preimages
                        .get(&relation_view)
                        .ok_or_else(|| bad("preimage concat relation is not registered"))?;
                    if source.is_some_and(|current| current != relation.source) {
                        return Err(bad("preimage concat requires a common source"));
                    }
                    source = Some(relation.source);
                    let part = matrix(input)?;
                    value.error_bound = value.error_bound.max(part.error_bound);
                    value.coefficient_magnitude_bound =
                        value.coefficient_magnitude_bound.max(part.coefficient_magnitude_bound);
                    value.is_constant_polynomial &= part.is_constant_polynomial;
                    let target_part = self
                        .states
                        .get(&relation.target)
                        .ok_or_else(|| bad("preimage concat target is unavailable"))?;
                    target.error_bound = target.error_bound.max(target_part.error_bound.clone());
                    target.coefficient_magnitude_bound = target
                        .coefficient_magnitude_bound
                        .max(target_part.coefficient_magnitude_bound.clone());
                    target.is_constant_polynomial &= target_part.is_constant_polynomial;
                    target_views.push(relation.target);
                }
                let target_view =
                    self.interners.intern_composed_view(target_views, Vec::new(), &[]);
                self.states.insert(target_view, target);
                Ok(vec![Info {
                    value: AbstractValue::Matrix(value),
                    ty: Some(output_type),
                    relation: Some(RightPreimage {
                        source: source.expect("nonempty preimage concat"),
                        target: target_view,
                        view: None,
                        selector: None,
                    }),
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::MaterializePreimageExact | NodeKind::DecompositionEntry { .. } => {
                let relation_view =
                    xs[0].relation.as_ref().and_then(|relation| relation.view).ok_or_else(
                        || SimulationError::Relation {
                            message: "exact preimage projection has no registered relation".into(),
                            site: site(),
                        },
                    )?;
                let relation = self.preimages.get(&relation_view).ok_or_else(|| {
                    SimulationError::Relation {
                        message: "exact preimage projection relation is unavailable".into(),
                        site: site(),
                    }
                })?;
                let target =
                    self.states.get(&relation.target).ok_or_else(|| SimulationError::Relation {
                        message: "exact preimage projection target state is unavailable".into(),
                        site: site(),
                    })?;
                if !target.error_bound.is_zero() {
                    return Err(SimulationError::Relation {
                        message: "preimage can be projected only when its relation target is exact"
                            .into(),
                        site: site(),
                    });
                }
                let mut value = matrix(&xs[0])?;
                value.right_carrier = None;
                let ty = match kind {
                    NodeKind::MaterializePreimageExact => mt(&xs[0])?,
                    NodeKind::DecompositionEntry { .. } => output_type(scope, n, env)?,
                    _ => unreachable!(),
                };
                Ok(vec![Info {
                    value: AbstractValue::Matrix(value),
                    ty: Some(ty),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::TrapdoorPublic => Ok(vec![xs[0].clone()]),
            NodeKind::LiftIntegerToConstantPolynomial { .. } => {
                let t = output_type(scope, n, env)?;
                let interval = int(&xs[0])?;
                // A constant polynomial must cover every integer in the
                // interval, so its coefficient magnitude is max(|min|,|max|)
                // rather than only the positive endpoint.
                Ok(vec![Info {
                    value: AbstractValue::Matrix(state::exact_matrix(
                        &t,
                        crate::bound::max_abs_interval(
                            &interval.minimum,
                            &interval.maximum_inclusive,
                        ),
                        true,
                    )?),
                    ty: Some(t),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::ExtractCoefficient { canonical_input_exclusive_upper, .. } => {
                let upper = canonical_input_exclusive_upper.clone().ok_or_else(|| {
                    SimulationError::InvalidGraph {
                        message: "coefficient extraction requires an authoritative canonical range"
                            .into(),
                        site: site(),
                    }
                })?;
                if upper.is_zero() {
                    return Err(bad("coefficient extraction canonical range is empty"));
                }
                Ok(vec![Info {
                    value: AbstractValue::Integer(state::IntegerState::new(
                        BigInt::zero(),
                        BigInt::from(upper - BigUint::one()),
                    )?),
                    ty: None,
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
                let modulus = plaintext_modulus.evaluate(env).map_err(|e| bad(&e.to_string()))?;
                let length = length.evaluate(env).map_err(|e| bad(&e.to_string()))?;
                if modulus <= BigInt::zero() || length < BigInt::zero() {
                    return Err(bad("invalid threshold-decode parameters"));
                }
                Ok(vec![if *output_bool {
                    Info {
                        value: AbstractValue::Boolean(state::BooleanState::Either),
                        ty: None,
                        relation: None,
                        view: crate::FamilyViewId(u32::MAX),
                        paired_public: None,
                    }
                } else {
                    Info {
                        value: AbstractValue::Integer(state::IntegerState::new(
                            BigInt::zero(),
                            &modulus - BigInt::one(),
                        )?),
                        ty: None,
                        relation: None,
                        view: crate::FamilyViewId(u32::MAX),
                        paired_public: None,
                    }
                }])
            }
            NodeKind::CrtRecompose { reconstruction_coefficients, .. } => {
                if xs.is_empty() || xs.len() != reconstruction_coefficients.len() {
                    return Err(bad("CRT recomposition arity does not match coefficients"));
                }
                let t = mt(&xs[0])?;
                let mut result = matrix(&xs[0])?.scale(
                    &reconstruction_coefficients[0]
                        .evaluate(env)
                        .map_err(|e| bad(&e.to_string()))?,
                    &t.modulus,
                )?;
                for (x, coefficient) in
                    xs.iter().skip(1).zip(reconstruction_coefficients.iter().skip(1))
                {
                    let term = matrix(x)?.scale(
                        &coefficient.evaluate(env).map_err(|e| bad(&e.to_string()))?,
                        &t.modulus,
                    )?;
                    result = result.add(&term, &t.modulus)?;
                }
                Ok(vec![Info {
                    value: AbstractValue::Matrix(result),
                    ty: Some(t),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::PackPolynomialCoefficients { matrix_type: _, coefficient_bits } => {
                let t = output_type(scope, n, env)?;
                let bits = coefficient_bits
                    .evaluate(env)
                    .map_err(|e| bad(&e.to_string()))?
                    .to_usize()
                    .ok_or_else(|| bad("coefficient bit count is not nonnegative"))?;
                if bits == 0 {
                    return Err(bad("packed polynomial requires at least one coefficient bit"));
                }
                let expected = t
                    .rows
                    .checked_mul(t.columns)
                    .and_then(|x| x.checked_mul(t.ring_dimension))
                    .and_then(|x| x.checked_mul(bits))
                    .ok_or_else(|| bad("packed coefficient shape overflows"))?;
                let actual = match xs.first().map(|x| &x.value) {
                    Some(AbstractValue::Family(f)) => {
                        f.shape.iter().try_fold(1usize, |a, b| a.checked_mul(*b)).unwrap_or(0)
                    }
                    _ => xs.len(),
                };
                if actual != expected {
                    return Err(bad("packed coefficient input length does not match matrix shape"));
                }
                Ok(vec![Info {
                    value: AbstractValue::Matrix(state::exact_matrix(
                        &t,
                        (&BigUint::one() << bits) - BigUint::one(),
                        t.ring_dimension == 1,
                    )?),
                    ty: Some(t),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::FamilyPack { shape } => {
                let shape = shape
                    .iter()
                    .map(|e| {
                        e.evaluate(env)
                            .ok()
                            .and_then(|x| x.to_usize())
                            .ok_or_else(|| bad("invalid family extent"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let first = xs.first().ok_or_else(|| bad("empty family"))?;
                let mut packed = crate::family::pack(
                    shape.clone(),
                    &xs.iter().map(|x| x.value.clone()).collect::<Vec<_>>(),
                )?;
                // A family is one structural source function.  Preserve the
                // mapping when every coordinate has an exact source, even
                // when those coordinates use distinct scalar producers.
                if let (Some(source), AbstractValue::Family(family)) = (
                    xs.iter()
                        .map(|x| {
                            matrix_state(&x.value)
                                .and_then(|m| m.right_carrier.as_ref().map(|c| c.source))
                        })
                        .collect::<Option<Vec<_>>>()
                        .map(|sources| self.group_source_for(sources, shape.clone())),
                    &mut packed,
                ) {
                    if let AbstractValue::Matrix(matrix) = family.element.as_mut() {
                        let gain = xs
                            .iter()
                            .filter_map(|x| {
                                matrix_state(&x.value).and_then(|m| {
                                    m.right_carrier.as_ref().map(|c| c.left_gain.clone())
                                })
                            })
                            .max()
                            .unwrap_or_default();
                        matrix.right_carrier =
                            Some(crate::RightCarrier { source, left_gain: gain });
                    }
                }
                // A family summary can preserve one uniform source function,
                // but a mixed set of source identities cannot be represented
                // by its single element state.
                let packed_state = matrix_state(&packed);
                self.record_carrier_drops(
                    xs,
                    packed_state.as_ref(),
                    site(),
                    "carrier lost while packing a family",
                );
                let paired_public =
                    xs.iter().map(|input| input.paired_public).collect::<Option<Vec<_>>>().map(
                        |views| self.interners.intern_composed_view(views, shape.clone(), &[]),
                    );
                Ok(vec![Info {
                    value: packed,
                    ty: first.ty.clone(),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public,
                }])
            }
            NodeKind::FamilyGetStatic { indices } => {
                let f = match &xs[0].value {
                    AbstractValue::Family(f) => f,
                    _ => return Err(bad("family required")),
                };
                if indices.len() != f.shape.len() ||
                    indices.iter().enumerate().any(|(axis, index)| {
                        index
                            .evaluate(env)
                            .ok()
                            .and_then(|v| v.to_usize())
                            .is_none_or(|v| v >= f.shape[axis])
                    })
                {
                    return Err(SimulationError::SelectorOutOfRange {
                        message: "static family index is outside its axis".into(),
                        site: site(),
                    });
                }
                let mut value = family_element(&xs[0]).ok_or_else(|| bad("family required"))?;
                let mut relation = xs[0].relation.clone();
                let map = mxx_ir_core::IndexMap::new(indices.clone());
                let paired_public = xs[0].paired_public.map(|view| {
                    self.interners.intern_composed_view(
                        vec![view],
                        Vec::new(),
                        std::slice::from_ref(&map),
                    )
                });
                let relation_source = relation.as_ref().map(|relation| relation.source);
                remap_carriers(&mut value, |source| {
                    self.mapped_source_for(source, &map, Vec::new(), Some(env))
                });
                if let Some(source) = relation_source {
                    let mapped = self.mapped_source_for(source, &map, Vec::new(), Some(env));
                    if let Some(relation) = relation.as_mut() {
                        relation.source = mapped;
                        let old_target = relation.target;
                        let target = self.interners.intern_composed_view(
                            vec![relation.target],
                            Vec::new(),
                            std::slice::from_ref(&map),
                        );
                        if let Some(state) =
                            self.remap_target_with_map(old_target, &map, Vec::new(), Some(env))
                        {
                            self.states.insert(target, state);
                        }
                        relation.target = target;
                    }
                }
                Ok(vec![Info {
                    value,
                    ty: xs[0].ty.clone(),
                    relation,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public,
                }])
            }
            NodeKind::FamilyGetDynamic { rank } => {
                let f = match &xs[0].value {
                    AbstractValue::Family(f) => f,
                    _ => return Err(bad("family required")),
                };
                if *rank != f.shape.len() || xs.len() != rank.saturating_add(1) {
                    return Err(bad("family rank mismatch"));
                }
                for (axis, selector) in xs[1..].iter().enumerate() {
                    validate_index(selector, f.shape[axis], site())?;
                }
                let selectors = xs[1..]
                    .iter()
                    .map(|selector| self.selector_for(selector.view))
                    .collect::<Vec<_>>();
                let concrete_indices = xs[1..]
                    .iter()
                    .map(|selector| match &selector.value {
                        AbstractValue::Integer(range)
                            if range.minimum == range.maximum_inclusive =>
                        {
                            range.minimum.to_usize()
                        }
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>();
                let mut value = family_element(&xs[0]).ok_or_else(|| bad("family required"))?;
                let mut relation =
                    xs[0].relation.clone().map(|r| specialize_relation(r, &selectors));
                let paired_public = xs[0].paired_public.map(|view| {
                    self.interners.intern_composed_view(
                        std::iter::once(view)
                            .chain(xs[1..].iter().map(|selector| selector.view))
                            .collect(),
                        Vec::new(),
                        &[],
                    )
                });
                let relation_source = relation.as_ref().map(|relation| relation.source);
                remap_carriers(&mut value, |source| {
                    self.gathered_source_for_concrete(
                        source,
                        selectors.clone(),
                        Vec::new(),
                        concrete_indices.as_deref(),
                    )
                });
                if let Some(source) = relation_source {
                    let mapped = self.gathered_source_for_concrete(
                        source,
                        selectors.clone(),
                        Vec::new(),
                        concrete_indices.as_deref(),
                    );
                    if let Some(relation) = relation.as_mut() {
                        relation.source = mapped;
                    }
                }
                if let Some(relation) = relation.as_mut() {
                    let old_target = relation.target;
                    let target = self.interners.intern_composed_view(
                        std::iter::once(relation.target)
                            .chain(xs[1..].iter().map(|selector| selector.view))
                            .collect(),
                        value_family_shape(&value).unwrap_or_default(),
                        &[],
                    );
                    if let Some(state) = self.remap_target_with_selectors(
                        old_target,
                        selectors.clone(),
                        value_family_shape(&value).unwrap_or_default(),
                    ) {
                        self.states.insert(target, state);
                    }
                    relation.target = target;
                }
                Ok(vec![Info {
                    value,
                    ty: xs[0].ty.clone(),
                    relation,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public,
                }])
            }
            NodeKind::FamilySelectAxis { axis } => {
                let f = match &xs[0].value {
                    AbstractValue::Family(f) => f,
                    _ => return Err(bad("family required")),
                };
                if *axis >= f.shape.len() {
                    return Err(bad("family axis out of range"));
                }
                if xs[0].relation.is_some() && *axis + 1 != f.shape.len() {
                    return Err(SimulationError::Relation {
                        message: "a relation-bearing family may select only its final branch axis"
                            .into(),
                        site: site(),
                    });
                }
                if xs.len() != 2 {
                    return Err(bad("family selector is missing"));
                }
                validate_axis_selector(&xs[1], &f.shape, *axis, site())?;
                let e = f.element.as_ref().clone();
                let mut v = if f.shape.len() == 1 {
                    e
                } else {
                    AbstractValue::Family(FamilyState::new(
                        f.shape
                            .iter()
                            .enumerate()
                            .filter_map(|(i, x)| (i != *axis).then_some(*x))
                            .collect(),
                        e,
                    )?)
                };
                let selectors = xs[1..]
                    .iter()
                    .map(|selector| self.selector_for(selector.view))
                    .collect::<Vec<_>>();
                let mut relation =
                    xs[0].relation.clone().map(|r| specialize_relation(r, &selectors));
                if let Some(source) = relation.as_ref().map(|relation| relation.source) {
                    // A relation-bearing family has a grouped source and the
                    // selected final axis is only the preimage branch.  It
                    // must specialize the target, but cannot specialize B[g]
                    // by the branch selector d.  Non-relation carriers still
                    // use the ordinary selector mapping.
                    let mapped = self.source_after_axis_selection(
                        source,
                        relation.is_some(),
                        *axis,
                        f.shape.len(),
                        selectors.clone(),
                        value_family_shape(&v).unwrap_or_default(),
                    );
                    remap_carriers(&mut v, |candidate| {
                        if candidate == source {
                            mapped
                        } else {
                            self.gathered_source_for_concrete(
                                candidate,
                                selectors.clone(),
                                Vec::new(),
                                None,
                            )
                        }
                    });
                    if let Some(relation) = relation.as_mut() {
                        relation.source = mapped;
                    }
                } else {
                    remap_carriers(&mut v, |source| {
                        self.gathered_source_for_concrete(
                            source,
                            selectors.clone(),
                            Vec::new(),
                            None,
                        )
                    });
                }
                let paired_public = xs[0].paired_public.map(|view| {
                    self.interners.intern_composed_view(
                        vec![view, xs[1].view],
                        value_family_shape(&v).unwrap_or_default(),
                        &[],
                    )
                });
                if let Some(relation) = relation.as_mut() {
                    let old_target = relation.target;
                    let target = self.interners.intern_composed_view(
                        vec![relation.target, xs[1].view],
                        value_family_shape(&v).unwrap_or_default(),
                        &[],
                    );
                    if let Some(state) = self.remap_target_after_axis_selection(
                        old_target,
                        *axis,
                        selectors.clone(),
                        value_family_shape(&v).unwrap_or_default(),
                    ) {
                        self.states.insert(target, state);
                    }
                    relation.target = target;
                }
                Ok(vec![Info {
                    value: v,
                    ty: xs[0].ty.clone(),
                    relation,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public,
                }])
            }
            NodeKind::FamilyReindex { output_shape, map } => {
                let shape = output_shape
                    .iter()
                    .map(|e| {
                        e.evaluate(env)
                            .ok()
                            .and_then(|x| x.to_usize())
                            .ok_or_else(|| bad("invalid family extent"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if xs[0].relation.is_some() && shape.last().is_some() {
                    let branch_axis = shape.len() - 1;
                    // The source of a shared-source relation is defined on
                    // input group coordinates only.  A group coordinate that
                    // depends on the output branch would require B[g,d].
                    if map.input_indices.len() > 1 &&
                        map.input_indices[..map.input_indices.len() - 1]
                            .iter()
                            .any(|expr| index_expr_depends_axis(expr, branch_axis))
                    {
                        return Err(SimulationError::BranchDependentSource { site: site() });
                    }
                }
                let mut relation = xs[0].relation.clone();
                let mut value = AbstractValue::Family(FamilyState::new(
                    shape.clone(),
                    family_element(&xs[0]).ok_or_else(|| bad("family required"))?,
                )?);
                let relation_source = relation.as_ref().map(|relation| relation.source);
                remap_carriers(&mut value, |source| {
                    self.mapped_source_for(source, map, shape.clone(), Some(env))
                });
                let paired_public = xs[0].paired_public.map(|view| {
                    self.interners.intern_composed_view(
                        vec![view],
                        shape.clone(),
                        std::slice::from_ref(map),
                    )
                });
                if let (Some(relation), Some(source)) = (relation.as_mut(), relation_source) {
                    let source_rank = self
                        .source_lineages
                        .get(&source)
                        .map(|lineage| lineage.shape.len())
                        .unwrap_or_else(|| map.input_indices.len().saturating_sub(1));
                    let group_shape = shape[..shape.len().saturating_sub(1)].to_vec();
                    let source_map = mxx_ir_core::IndexMap::new(
                        map.input_indices.iter().take(source_rank).cloned().collect::<Vec<_>>(),
                    );
                    relation.source =
                        self.mapped_source_for(source, &source_map, group_shape, Some(env));
                    let old_target = relation.target;
                    let target = self.interners.intern_composed_view(
                        vec![relation.target],
                        shape.clone(),
                        std::slice::from_ref(map),
                    );
                    if let Some(state) =
                        self.remap_target_with_map(old_target, map, shape.clone(), Some(env))
                    {
                        self.states.insert(target, state);
                    }
                    relation.target = target;
                }
                Ok(vec![Info {
                    value,
                    ty: xs[0].ty.clone(),
                    relation,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public,
                }])
            }
            NodeKind::FamilyGather { output_shape, input_rank } => {
                let shape = output_shape
                    .iter()
                    .map(|e| {
                        e.evaluate(env)
                            .ok()
                            .and_then(|x| x.to_usize())
                            .ok_or_else(|| bad("invalid family extent"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if *input_rank + 1 != xs.len() {
                    return Err(bad("family gather selector arity mismatch"));
                }
                let source_shape = match &xs[0].value {
                    AbstractValue::Family(family) => family.shape.clone(),
                    _ => return Err(bad("family gather source must be a family")),
                };
                for (axis, selector) in xs[1..].iter().enumerate() {
                    match &selector.value {
                        AbstractValue::Integer(_) => {
                            validate_index(
                                selector,
                                source_shape.get(axis).copied().unwrap_or(0),
                                site(),
                            )?;
                        }
                        AbstractValue::Family(family) => {
                            if family.shape != shape {
                                return Err(SimulationError::SelectorOutOfRange {
                                    message:
                                        "gather selector family shape does not match output shape"
                                            .into(),
                                    site: site(),
                                });
                            }
                            validate_integer_range(
                                family.element.as_ref(),
                                source_shape.get(axis).copied().unwrap_or(0),
                                site(),
                            )?;
                        }
                        _ => {
                            return Err(bad("gather selectors must be integers or integer families"))
                        }
                    }
                }
                let mut value = AbstractValue::Family(FamilyState::new(
                    shape.clone(),
                    family_element(&xs[0]).ok_or_else(|| bad("family required"))?,
                )?);
                let selectors = xs[1..]
                    .iter()
                    .map(|selector| self.selector_for(selector.view))
                    .collect::<Vec<_>>();
                let selector_views = xs[1..].iter().map(|selector| selector.view);
                let mut relation = xs[0].relation.clone();
                let paired_public = xs[0].paired_public.map(|view| {
                    let selector_views = xs[1..].iter().map(|selector| selector.view);
                    self.interners.intern_composed_view(
                        std::iter::once(view).chain(selector_views).collect(),
                        shape.clone(),
                        &[],
                    )
                });
                let relation_source = relation.as_ref().map(|relation| relation.source);
                remap_carriers(&mut value, |source| {
                    self.gathered_source_for(source, selectors.clone(), shape.clone())
                });
                if let Some(source) = relation_source {
                    let mapped = self.gathered_source_for(source, selectors.clone(), shape.clone());
                    if let Some(relation) = relation.as_mut() {
                        relation.source = mapped;
                        let old_target = relation.target;
                        let target = self.interners.intern_composed_view(
                            std::iter::once(relation.target).chain(selector_views).collect(),
                            shape.clone(),
                            &[],
                        );
                        if let Some(state) = self.remap_target_with_selectors(
                            old_target,
                            selectors.clone(),
                            shape.clone(),
                        ) {
                            self.states.insert(target, state);
                        }
                        relation.target = target;
                    }
                }
                Ok(vec![Info {
                    value,
                    ty: xs[0].ty.clone(),
                    relation,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public,
                }])
            }
            NodeKind::Select { count } => {
                let count = count
                    .evaluate(env)
                    .map_err(|e| bad(&e.to_string()))?
                    .to_usize()
                    .ok_or_else(|| bad("invalid select count"))?;
                if xs.len() != count.saturating_add(1) {
                    return Err(bad("select branch count mismatch"));
                }
                validate_index(&xs[0], count, site())?;
                // Structural loops are evaluated with a concrete binder in
                // `env`.  Preserve that precision when the selector is a
                // singleton; joining all branches here would manufacture
                // impossible values (for example `slot - base` from an
                // inactive branch) and make a later dynamic gather reject a
                // valid loop.  Non-singleton selectors retain the sound
                // conservative join below.
                let selector = int(&xs[0])?;
                let mut selected = if selector.minimum == selector.maximum_inclusive {
                    xs[selector.minimum.to_usize().ok_or_else(|| bad("invalid select index"))? + 1]
                        .clone()
                } else {
                    xs[1].clone()
                };
                if let Some(relation) = selected.relation.take() {
                    let selector_id = self.selector_for(xs[0].view);
                    selected.relation = Some(specialize_relation(relation, &[selector_id]));
                }
                if selector.minimum != selector.maximum_inclusive {
                    for branch in &xs[2..] {
                        let type_info = selected.ty.clone();
                        selected = self.join_uniform_with_diagnostics(
                            selected,
                            branch.clone(),
                            type_info.as_ref(),
                            site(),
                        )?;
                    }
                }
                Ok(vec![selected])
            }
            NodeKind::SubgraphCall(_) | NodeKind::SequentialLoop(_) | NodeKind::ParallelGrid(_) => {
                self.structural(stage, graph, sid, occurrence, n, xs, env)
            }
        }
    }
    fn input(
        &mut self,
        stage: &crate::StageId,
        graph: &Graph,
        sid: &FrozenGraphScopeId,
        occurrence: &[String],
        n: usize,
        name: &str,
        env: &ParamEnv,
    ) -> Result<Info, SimulationError> {
        let node = graph.scope(sid).unwrap().node(mxx_ir_core::NodeId(n as u64)).unwrap();
        let ty = node.output_types().first().ok_or_else(|| SimulationError::InvalidGraph {
            message: "input type missing".into(),
            site: None,
        })?;
        let f = self
            .request
            .external_inputs
            .iter()
            .find(|x| x.stage == *stage && x.input == name)
            .map(|x| &x.value)
            .ok_or_else(|| SimulationError::MissingExternalInputFact {
                stage: stage.clone(),
                input: name.into(),
            })?;
        let mut result = value_fact(ty, f, env)
            .map_err(|m| SimulationError::InvalidGraph { message: m, site: None })?;
        let paired_public_name = self.request.external_inputs.iter().find_map(|x| {
            (x.stage == *stage && x.input == name)
                .then(|| trapdoor_public_input(&x.value).map(str::to_owned))
                .flatten()
        });
        if self.request.external_inputs.iter().any(|x| {
            x.stage == *stage &&
                trapdoor_public_input(&x.value).is_some_and(|public| public == name)
        }) {
            let source = self.source_for(stage, sid, occurrence, n, "external-public");
            attach_carrier(&mut result.value, source);
        }
        if let Some(public_name) = paired_public_name {
            if let Some((public_node, candidate)) = graph.scope(sid).and_then(|scope| {
                    scope.nodes().iter().enumerate().find(|(_, candidate)| {
                        matches!(candidate.kind(), NodeKind::Input { name: candidate_name, .. } if candidate_name == &public_name)
                    })
                }) {
                    result.paired_public = Some(self.view_for_wire(
                        stage,
                        sid,
                        occurrence,
                        WireRef {
                            node: mxx_ir_core::NodeId(public_node as u64),
                            port: mxx_ir_core::Port(0),
                        },
                        candidate.output_types().first(),
                        candidate.kind(),
                        &[],
                        env,
                    )?);
            }
        }
        Ok(result)
    }
    fn structural(
        &mut self,
        stage: &crate::StageId,
        graph: &Graph,
        parent: &FrozenGraphScopeId,
        occurrence: &[String],
        n: usize,
        xs: &[Info],
        env: &ParamEnv,
    ) -> Result<Vec<Info>, SimulationError> {
        let child =
            graph.child_scope_id(parent, mxx_ir_core::NodeId(n as u64)).ok_or_else(|| {
                SimulationError::InvalidGraph { message: "missing child scope".into(), site: None }
            })?;
        let cs = graph.scope(&child).ok_or_else(|| SimulationError::InvalidGraph {
            message: "missing child".into(),
            site: None,
        })?;
        let outputs = if let mxx_ir_core::node::NodeKind::SequentialLoop(spec) =
            graph.scope(parent).unwrap().node(mxx_ir_core::NodeId(n as u64)).unwrap().kind()
        {
            let count = spec
                .count
                .evaluate(env)
                .map_err(|e| SimulationError::InvalidParameterEnvironment {
                    message: e.to_string(),
                })?
                .to_usize()
                .ok_or_else(|| SimulationError::InvalidGraph {
                    message: "loop count is not usize".into(),
                    site: None,
                })?;
            let carried = spec.carried_count;
            if carried > xs.len() || cs.inputs().len() != xs.len() {
                return Err(SimulationError::InvalidGraph {
                    message: "sequential loop carried arity mismatch".into(),
                    site: None,
                });
            }
            let mut current = xs[..carried].to_vec();
            let invariant = &xs[carried..];
            for iteration in 0..count {
                let mut loop_env = env.clone();
                loop_env.loop_indices.insert(spec.index_slot, iteration.into());
                loop_env = apply_bindings(loop_env, &spec.bindings)?;
                let mut args = current.clone();
                args.extend_from_slice(invariant);
                let preload = cs.inputs().iter().copied().zip(args).collect();
                let mut child_occurrence = occurrence.to_vec();
                child_occurrence.push(format!("node:{n}/iteration:{iteration}"));
                let vals =
                    self.scope(stage, graph, &child, &child_occurrence, loop_env, preload)?;
                current = cs
                    .outputs()
                    .iter()
                    .take(carried)
                    .map(|wire| {
                        vals.get(wire).cloned().ok_or_else(|| SimulationError::InvalidGraph {
                            message: "missing carried output".into(),
                            site: None,
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            current
                .into_iter()
                .map(|mut info| {
                    let relation_shape = info
                        .relation
                        .as_ref()
                        .and_then(|relation| self.view_shape(relation.target));
                    if let Some(shape) = relation_shape.or_else(|| value_family_shape(&info.value))
                    {
                        let shape_for_carrier = shape.clone();
                        if let Some(relation) = info.relation.as_mut() {
                            relation.source =
                                self.lift_source_for_shape(relation.source, shape.clone());
                        }
                        if let Some(matrix) = matrix_state_mut(&mut info.value) {
                            matrix.right_carrier =
                                matrix.right_carrier.take().map(|carrier| crate::RightCarrier {
                                    source: self.lift_source_for_shape(
                                        carrier.source,
                                        shape_for_carrier.clone(),
                                    ),
                                    left_gain: carrier.left_gain,
                                });
                        }
                    }
                    info
                })
                .collect()
        } else if let mxx_ir_core::node::NodeKind::ParallelGrid(spec) =
            graph.scope(parent).unwrap().node(mxx_ir_core::NodeId(n as u64)).unwrap().kind()
        {
            if spec.shape.len() != spec.index_slots.len() || cs.inputs().len() != xs.len() {
                return Err(SimulationError::InvalidGraph {
                    message: "parallel grid shape, binder, or input arity mismatch".into(),
                    site: None,
                });
            }
            let grid_shape = spec
                .shape
                .iter()
                .map(|extent| {
                    extent
                        .evaluate(env)
                        .map_err(|e| SimulationError::InvalidParameterEnvironment {
                            message: e.to_string(),
                        })?
                        .to_usize()
                        .ok_or_else(|| SimulationError::InvalidGraph {
                            message: "parallel grid extent is not usize".into(),
                            site: None,
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            if graph
                .scope(parent)
                .and_then(|scope| scope.node(mxx_ir_core::NodeId(n as u64)))
                .is_some_and(|node| {
                    node.output_types().iter().any(|ty| {
                        family_shape(ty, env).is_some_and(|output_shape| output_shape != grid_shape)
                    })
                })
            {
                return Err(SimulationError::InvalidGraph {
                    message: "parallel grid output shape does not match its body grid".into(),
                    site: None,
                });
            }
            let mut child_occurrence = occurrence.to_vec();
            child_occurrence.push(format!("node:{n}/grid"));
            let lane_count = grid_shape
                .iter()
                .try_fold(1usize, |count, extent| count.checked_mul(*extent))
                .filter(|count| *count > 0)
                .ok_or_else(|| SimulationError::InvalidGraph {
                    message: "parallel grid cardinality is zero or overflows usize".into(),
                    site: None,
                })?;
            let mut joined_outputs: Vec<Option<Info>> = vec![None; cs.outputs().len()];
            let mut joined_target_states: Vec<Option<MatrixState>> = vec![None; cs.outputs().len()];
            let base_preimages = self.preimages.clone();
            let base_states = self.states.clone();
            let mut representative_preimages = None;
            let mut representative_states = None;
            for lane in 0..lane_count {
                if lane > 0 {
                    // Every lane executes the same frozen body occurrence but
                    // with different loop-index values. Restore its incoming
                    // relation tables so lane-local producers cannot collide
                    // with or consume facts left by the preceding lane.
                    self.preimages = base_preimages.clone();
                    self.states = base_states.clone();
                }
                let mut remainder = lane;
                let mut coordinates = vec![0usize; grid_shape.len()];
                for axis in (0..grid_shape.len()).rev() {
                    coordinates[axis] = remainder % grid_shape[axis];
                    remainder /= grid_shape[axis];
                }
                let mut grid_env = env.clone();
                for (slot, coordinate) in spec.index_slots.iter().zip(coordinates) {
                    grid_env.loop_indices.insert(*slot, coordinate.into());
                }
                grid_env = apply_bindings(grid_env, &spec.bindings)?;
                let preload = cs
                    .inputs()
                    .iter()
                    .copied()
                    .zip(xs.iter().cloned())
                    .enumerate()
                    .map(|(arg, (wire, value))| {
                        let mapped = match spec.input_modes.get(arg) {
                            Some(mxx_ir_core::node::GridInputMode::Reindex { map }) => {
                                let coordinates = map
                                    .input_indices
                                    .iter()
                                    .map(|expr| eval_grid_index(expr, &grid_env, &spec.index_slots))
                                    .collect::<Result<Vec<_>, _>>()?;
                                let family_shape = match &value.value {
                                    AbstractValue::Family(family) => family.shape.clone(),
                                    _ => unreachable!(),
                                };
                                if coordinates.len() != family_shape.len() ||
                                    coordinates.iter().enumerate().any(|(axis, coordinate)| {
                                        *coordinate >= family_shape[axis]
                                    })
                                {
                                    return Err(SimulationError::SelectorOutOfRange {
                                        message:
                                            "parallel-grid reindex is outside its input family"
                                                .into(),
                                        site: None,
                                    });
                                }
                                let mut mapped = value;
                                let mapped_view = self.interners.intern_composed_view(
                                    vec![mapped.view],
                                    Vec::new(),
                                    std::slice::from_ref(map),
                                );
                                mapped.view = mapped_view;
                                if let Some(relation) = mapped.relation.as_mut() {
                                    relation.view = Some(mapped_view);
                                }
                                mapped.paired_public = mapped.paired_public.map(|paired| {
                                    self.interners.intern_composed_view(
                                        vec![paired],
                                        Vec::new(),
                                        std::slice::from_ref(map),
                                    )
                                });
                                let family_element = match &mapped.value {
                                    AbstractValue::Family(family) => {
                                        family.element.as_ref().clone()
                                    }
                                    _ => unreachable!(),
                                };
                                mapped.value = family_element;
                                let relation_source = mapped.relation.as_ref().map(|r| r.source);
                                remap_carriers(&mut mapped.value, |source| {
                                    self.mapped_source_for(
                                        source,
                                        map,
                                        grid_shape.clone(),
                                        Some(&grid_env),
                                    )
                                });
                                if let Some(source) = relation_source {
                                    let mapped_source = self.mapped_source_for(
                                        source,
                                        map,
                                        grid_shape.clone(),
                                        Some(&grid_env),
                                    );
                                    if let Some(relation) = mapped.relation.as_mut() {
                                        relation.source = mapped_source;
                                    }
                                }
                                if let Some(relation) = mapped.relation.as_mut() {
                                    let old_target = relation.target;
                                    let target = self.interners.intern_composed_view(
                                        vec![relation.target],
                                        grid_shape.clone(),
                                        std::slice::from_ref(map),
                                    );
                                    if let Some(state) = self.remap_target_with_map(
                                        old_target,
                                        map,
                                        grid_shape.clone(),
                                        Some(&grid_env),
                                    ) {
                                        self.states.insert(target, state);
                                    }
                                    relation.target = target;
                                }
                                mapped
                            }
                            _ => value,
                        };
                        Ok((wire, mapped))
                    })
                    .collect::<Result<HashMap<_, _>, SimulationError>>()?;
                let vals =
                    self.scope(stage, graph, &child, &child_occurrence, grid_env, preload)?;
                if lane == 0 {
                    representative_preimages = Some(self.preimages.clone());
                    representative_states = Some(self.states.clone());
                }
                for (port, wire) in cs.outputs().iter().enumerate() {
                    let info =
                        vals.get(wire).cloned().ok_or_else(|| SimulationError::InvalidGraph {
                            message: "missing parallel-grid output".into(),
                            site: None,
                        })?;
                    if let Some(state) = info
                        .relation
                        .as_ref()
                        .and_then(|relation| self.states.get(&relation.target))
                        .cloned()
                    {
                        joined_target_states[port] =
                            Some(match joined_target_states[port].take() {
                                Some(previous) => {
                                    let representative_carrier = previous.right_carrier.clone();
                                    let AbstractValue::Matrix(joined) = crate::family::join(
                                        &AbstractValue::Matrix(previous),
                                        &AbstractValue::Matrix(state),
                                    )?
                                    else {
                                        unreachable!("joining matrix states returns a matrix")
                                    };
                                    MatrixState { right_carrier: representative_carrier, ..joined }
                                }
                                None => state,
                            });
                    }
                    joined_outputs[port] = Some(match joined_outputs[port].take() {
                        Some(previous) => {
                            let ty = previous.ty.clone().or_else(|| info.ty.clone());
                            let joined = self.join_uniform_with_diagnostics(
                                previous.clone(),
                                info,
                                ty.as_ref(),
                                None,
                            )?;
                            let mut joined_value = joined.value;
                            preserve_grid_carriers(&previous.value, &mut joined_value);
                            // Bounds are uniform across the frozen family, but
                            // relation/view identity is the one symbolic grid
                            // occurrence. Keep the representative provenance
                            // while replacing only its joined abstract value.
                            Info { value: joined_value, ty: joined.ty, ..previous }
                        }
                        None => info,
                    });
                }
            }
            self.preimages = representative_preimages.expect("positive grid cardinality");
            self.states = representative_states.expect("positive grid cardinality");
            let vals = cs
                .outputs()
                .iter()
                .enumerate()
                .map(|(port, wire)| {
                    let info = joined_outputs[port].clone().expect("positive grid cardinality");
                    if let (Some(relation), Some(state)) =
                        (&info.relation, joined_target_states[port].clone())
                    {
                        self.states.insert(relation.target, state);
                    }
                    (*wire, info)
                })
                .collect::<HashMap<_, _>>();
            cs.outputs()
                .iter()
                .map(|wire| {
                    let mut info =
                        vals.get(wire).cloned().ok_or_else(|| SimulationError::InvalidGraph {
                            message: "missing parallel-grid output".into(),
                            site: None,
                        })?;
                    // Freeze the symbolic body as one binder-indexed family
                    // view.  The child occurrence remains an implementation
                    // detail and cannot alias another grid's family.
                    let view = self.interners.intern_composed_view(
                        vec![info.view],
                        grid_shape.clone(),
                        &[],
                    );
                    let paired_public = info.paired_public.map(|paired| {
                        self.interners.intern_composed_view(vec![paired], grid_shape.clone(), &[])
                    });
                    if let Some(relation) = info.relation.as_mut() {
                        let old_target = relation.target;
                        let target = self.interners.intern_composed_view(
                            vec![old_target],
                            grid_shape.clone(),
                            &[],
                        );
                        if let Some(state) = self.states.get(&old_target).cloned() {
                            self.states.insert(target, state);
                        }
                        relation.target = target;
                        relation.view = Some(view);
                    }
                    info.view = view;
                    info.paired_public = paired_public;
                    let mut family_value =
                        FamilyState::new(grid_shape.clone(), info.value.clone())?;
                    let relation_source = info.relation.as_ref().map(|r| r.source);
                    remap_carriers(&mut family_value.element, |source| {
                        self.lift_source_for_shape(source, grid_shape.clone())
                    });
                    if let (Some(relation), Some(source)) =
                        (info.relation.as_mut(), relation_source)
                    {
                        relation.source = self.lift_source_for_shape(source, grid_shape.clone());
                    }
                    info.value = AbstractValue::Family(family_value);
                    Ok(info)
                })
                .collect::<Result<Vec<_>, SimulationError>>()?
        } else {
            let preload = cs.inputs().iter().copied().zip(xs.iter().cloned()).collect();
            let child_env = match graph
                .scope(parent)
                .and_then(|scope| scope.node(mxx_ir_core::NodeId(n as u64)))
                .map(|node| node.kind())
            {
                Some(mxx_ir_core::node::NodeKind::SubgraphCall(spec)) => {
                    apply_bindings(env.clone(), &spec.bindings)?
                }
                _ => env.clone(),
            };
            let mut child_occurrence = occurrence.to_vec();
            child_occurrence.push(format!("node:{n}"));
            let vals = self.scope(stage, graph, &child, &child_occurrence, child_env, preload)?;
            cs.outputs()
                .iter()
                .map(|x| {
                    vals.get(x).cloned().ok_or_else(|| SimulationError::InvalidGraph {
                        message: "missing structural output".into(),
                        site: None,
                    })
                })
                .collect::<Result<Vec<_>, _>>()?
        };
        let declared = graph
            .scope(parent)
            .and_then(|scope| scope.node(mxx_ir_core::NodeId(n as u64)))
            .map(|node| node.output_types().to_vec())
            .unwrap_or_default();
        outputs
            .into_iter()
            .enumerate()
            .map(|(port, mut info)| {
                if let Some(extents) = declared.get(port).and_then(|ty| family_shape(ty, env)) &&
                    !matches!(info.value, AbstractValue::Family(_)) &&
                    let Ok(family) = FamilyState::new(extents, info.value.clone())
                {
                    info.value = AbstractValue::Family(family);
                }
                Ok(info)
            })
            .collect()
    }
}

fn integer(x: BigInt) -> Info {
    Info {
        value: AbstractValue::Integer(state::IntegerState::singleton(x)),
        ty: None,
        relation: None,
        view: crate::FamilyViewId(u32::MAX),
        paired_public: None,
    }
}

fn wire_types_compatible(expected: &WireType, actual: &WireType, env: &ParamEnv) -> bool {
    match (expected, actual) {
        (
            WireType::Matrix(a) | WireType::Preimage(a),
            WireType::Matrix(b) | WireType::Preimage(b),
        ) => {
            concrete_matrix(&WireType::Matrix(a.clone()), env) ==
                concrete_matrix(&WireType::Matrix(b.clone()), env)
        }
        (WireType::Trapdoor { matrix: a, .. }, WireType::Trapdoor { matrix: b, .. }) => {
            concrete_matrix(&WireType::Matrix(a.clone()), env) ==
                concrete_matrix(&WireType::Matrix(b.clone()), env)
        }
        (
            WireType::Family { element: ae, shape: as_ },
            WireType::Family { element: be, shape: bs },
        ) => {
            as_.len() == bs.len() &&
                as_.iter().zip(bs).all(|(a, b)| a.evaluate(env).ok() == b.evaluate(env).ok()) &&
                wire_types_compatible(ae, be, env)
        }
        (WireType::Bytes { length: a }, WireType::Bytes { length: b }) => {
            a.evaluate(env).ok() == b.evaluate(env).ok()
        }
        (
            WireType::TypedBlob { type_name: an, schema_hash: ah },
            WireType::TypedBlob { type_name: bn, schema_hash: bh },
        ) => an == bn && ah == bh,
        (WireType::ConstantInt, WireType::ConstantInt) |
        (WireType::ConstantReal, WireType::ConstantReal) |
        (WireType::ConstantBool, WireType::ConstantBool) |
        (WireType::Int, WireType::Int) |
        (WireType::Real, WireType::Real) |
        (WireType::Bool, WireType::Bool) => true,
        _ => false,
    }
}

fn apply_bindings(
    mut env: ParamEnv,
    bindings: &[(String, mxx_ir_core::IntExpr)],
) -> Result<ParamEnv, SimulationError> {
    for (name, expression) in bindings {
        let value = expression.evaluate(&env).map_err(|error| {
            SimulationError::InvalidParameterEnvironment { message: error.to_string() }
        })?;
        env.integers.insert(name.clone(), value);
    }
    Ok(env)
}

fn int(x: &Info) -> Result<state::IntegerState, SimulationError> {
    match &x.value {
        AbstractValue::Integer(x) => Ok(x.clone()),
        _ => Err(SimulationError::InvalidGraph { message: "integer required".into(), site: None }),
    }
}
fn join_uniform(
    a: Info,
    b: Info,
    ty: Option<&ConcreteMatrixType>,
) -> Result<Info, SimulationError> {
    let relation = match (&a.relation, &b.relation) {
        (Some(left), Some(right)) if left.source == right.source && left.target == right.target => {
            // A Select has one symbolic selector for all of its alternatives;
            // retain the first branch's exact relation identity when the
            // alternatives describe the same source and target family.
            a.relation.clone()
        }
        (Some(relation), None) | (None, Some(relation)) => Some(relation.clone()),
        _ => None,
    };
    let value = match (&a.value, &b.value) {
        (AbstractValue::Matrix(left), AbstractValue::Matrix(right)) => {
            let right_carrier = match (&left.right_carrier, &right.right_carrier) {
                (Some(a), Some(b)) if a.source == b.source => Some(crate::RightCarrier {
                    source: a.source,
                    left_gain: a.left_gain.clone().max(b.left_gain.clone()),
                }),
                (Some(a), None)
                    if right.error_bound.is_zero() &&
                        right.coefficient_magnitude_bound.is_zero() =>
                {
                    Some(a.clone())
                }
                (None, Some(b))
                    if left.error_bound.is_zero() && left.coefficient_magnitude_bound.is_zero() =>
                {
                    Some(b.clone())
                }
                (None, None) => None,
                _ => None,
            };
            AbstractValue::Matrix(MatrixState {
                error_bound: left.error_bound.clone().max(right.error_bound.clone()),
                coefficient_magnitude_bound: left
                    .coefficient_magnitude_bound
                    .clone()
                    .max(right.coefficient_magnitude_bound.clone())
                    .min(crate::centered_residue_bound(
                        &ty.ok_or_else(|| SimulationError::InvalidGraph {
                            message: "matrix select branch type is unavailable".into(),
                            site: None,
                        })?
                        .modulus,
                    )?),
                is_constant_polynomial: left.is_constant_polynomial && right.is_constant_polynomial,
                right_carrier,
            })
        }
        (AbstractValue::Integer(left), AbstractValue::Integer(right)) => {
            AbstractValue::Integer(left.join(right))
        }
        (AbstractValue::Boolean(left), AbstractValue::Boolean(right)) => {
            AbstractValue::Boolean(left.join(*right))
        }
        (AbstractValue::Bytes, AbstractValue::Bytes) => AbstractValue::Bytes,
        (AbstractValue::Trapdoor(left), AbstractValue::Trapdoor(right)) if left == right => {
            AbstractValue::Trapdoor(left.clone())
        }
        (AbstractValue::Family(_), AbstractValue::Family(_)) => {
            crate::family::join(&a.value, &b.value)?
        }
        (
            AbstractValue::TypedBlob { type_name, schema_hash },
            AbstractValue::TypedBlob { type_name: other, schema_hash: other_hash },
        ) if type_name == other && schema_hash == other_hash => a.value.clone(),
        _ => {
            return Err(SimulationError::InvalidGraph {
                message: "select branches have incompatible types".into(),
                site: None,
            })
        }
    };
    let paired_public = (a.paired_public == b.paired_public).then_some(a.paired_public).flatten();
    Ok(Info {
        value,
        ty: a.ty.or(b.ty),
        relation,
        view: crate::FamilyViewId(u32::MAX),
        paired_public,
    })
}
fn matrix_state(x: &AbstractValue) -> Option<MatrixState> {
    match x {
        AbstractValue::Matrix(x) => Some(x.clone()),
        AbstractValue::Family(f) => matrix_state(&f.element),
        _ => None,
    }
}

fn matrix_state_mut(x: &mut AbstractValue) -> Option<&mut MatrixState> {
    match x {
        AbstractValue::Matrix(x) => Some(x),
        AbstractValue::Family(f) => matrix_state_mut(f.element.as_mut()),
        _ => None,
    }
}

fn preserve_grid_carriers(representative: &AbstractValue, joined: &mut AbstractValue) {
    match (representative, joined) {
        (AbstractValue::Matrix(representative), AbstractValue::Matrix(joined)) => {
            joined.right_carrier = representative.right_carrier.clone();
        }
        (AbstractValue::Family(representative), AbstractValue::Family(joined))
            if representative.shape == joined.shape =>
        {
            preserve_grid_carriers(representative.element.as_ref(), joined.element.as_mut());
        }
        _ => {}
    }
}

fn attach_carrier(value: &mut AbstractValue, source: crate::SourceId) {
    match value {
        AbstractValue::Matrix(matrix) => {
            matrix.right_carrier = Some(crate::RightCarrier { source, left_gain: 1u8.into() });
        }
        AbstractValue::Family(family) => attach_carrier(family.element.as_mut(), source),
        _ => {}
    }
}

fn remap_carriers(
    value: &mut AbstractValue,
    mut map_source: impl FnMut(crate::SourceId) -> crate::SourceId,
) {
    if let Some(matrix) = matrix_state_mut(value) {
        matrix.right_carrier = matrix.right_carrier.take().map(|carrier| crate::RightCarrier {
            source: map_source(carrier.source),
            left_gain: carrier.left_gain,
        });
    }
}

/// Resolve a structural map whose indices are determined by output
/// coordinates and the concrete loop environment.  If a parameter or loop
/// slot is not available (for example while freezing a symbolic body), the
/// caller uses the conservative mapped-source key instead.
fn map_source_leaves(
    parent: &SourceLineage,
    map: &mxx_ir_core::IndexMap,
    output_shape: &[usize],
    env: Option<&ParamEnv>,
) -> Option<Vec<crate::SourceId>> {
    if map.input_indices.len() != parent.shape.len() {
        return None;
    }
    let count = output_shape.iter().copied().product::<usize>().max(1);
    let mut leaves = Vec::with_capacity(count);
    for flat in 0..count {
        let coordinates = unravel_index(flat, output_shape);
        let input = map
            .input_indices
            .iter()
            .map(|expr| eval_concrete_index(expr, &coordinates, env))
            .collect::<Option<Vec<_>>>()?;
        if input.iter().enumerate().any(|(axis, index)| *index >= parent.shape[axis]) {
            return None;
        }
        let flat_input = input.iter().zip(&parent.shape).fold(0usize, |flat, (index, extent)| {
            flat.saturating_mul(*extent).saturating_add(*index)
        });
        leaves.push(*parent.leaves.get(flat_input)?);
    }
    Some(leaves)
}

fn uniform_gathered_lineage(
    parent: &SourceLineage,
    output_shape: &[usize],
) -> Option<SourceLineage> {
    let first = parent.leaves.first()?;
    if !parent.leaves.iter().all(|leaf| leaf == first) {
        return None;
    }
    let count = output_shape.iter().copied().product::<usize>().max(1);
    Some(SourceLineage { shape: output_shape.to_vec(), leaves: vec![*first; count] })
}

fn uniform_axis_selection_lineage(
    parent: &SourceLineage,
    axis: usize,
    output_shape: &[usize],
) -> Option<SourceLineage> {
    if axis >= parent.shape.len() || output_shape.len() + 1 != parent.shape.len() {
        return None;
    }
    let expected_shape = parent
        .shape
        .iter()
        .enumerate()
        .filter_map(|(candidate, extent)| (candidate != axis).then_some(*extent))
        .collect::<Vec<_>>();
    if expected_shape != output_shape {
        return None;
    }
    let count = output_shape.iter().copied().product::<usize>().max(1);
    let mut leaves = Vec::with_capacity(count);
    for flat in 0..count {
        let output = unravel_index(flat, output_shape);
        let mut selected = None;
        for branch in 0..parent.shape[axis] {
            let mut input = output.clone();
            input.insert(axis, branch);
            let input_flat =
                input.iter().zip(&parent.shape).fold(0usize, |offset, (index, extent)| {
                    offset.saturating_mul(*extent).saturating_add(*index)
                });
            let leaf = *parent.leaves.get(input_flat)?;
            if selected.is_some_and(|current| current != leaf) {
                return None;
            }
            selected = Some(leaf);
        }
        leaves.push(selected?);
    }
    Some(SourceLineage { shape: output_shape.to_vec(), leaves })
}

fn unravel_index(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut coordinates = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        let extent = shape[axis].max(1);
        coordinates[axis] = flat % extent;
        flat /= extent;
    }
    coordinates
}

fn eval_concrete_index(
    expr: &mxx_ir_core::IndexExpr,
    coordinates: &[usize],
    env: Option<&ParamEnv>,
) -> Option<usize> {
    use mxx_ir_core::IndexExpr;
    fn eval(expr: &IndexExpr, coordinates: &[usize], env: Option<&ParamEnv>) -> Option<BigInt> {
        match expr {
            IndexExpr::Axis(axis) => coordinates.get(*axis).copied().map(BigInt::from),
            IndexExpr::Parameter(name) => env?.integers.get(name).cloned(),
            // The evaluator unrolls concrete loop bodies and supplies the
            // active slot in `env`.  Use it for those concrete maps; a map
            // from a symbolic structural body has no slot binding and
            // intentionally falls back to the opaque mapped-source key.
            IndexExpr::LoopIndex(slot) => env?.loop_indices.get(slot).cloned(),
            IndexExpr::Constant(value) => Some(value.clone()),
            IndexExpr::Add(a, b) => Some(eval(a, coordinates, env)? + eval(b, coordinates, env)?),
            IndexExpr::Subtract(a, b) => {
                Some(eval(a, coordinates, env)? - eval(b, coordinates, env)?)
            }
            IndexExpr::Multiply(a, b) => {
                Some(eval(a, coordinates, env)? * eval(b, coordinates, env)?)
            }
            IndexExpr::Divide(a, b) => {
                let denominator = eval(b, coordinates, env)?;
                if denominator.is_zero() {
                    None
                } else {
                    Some(eval(a, coordinates, env)? / denominator)
                }
            }
            IndexExpr::Remainder(a, b) => {
                let denominator = eval(b, coordinates, env)?;
                if denominator.is_zero() {
                    None
                } else {
                    Some(eval(a, coordinates, env)? % denominator)
                }
            }
            IndexExpr::Equal(a, b) => {
                Some(BigInt::from(eval(a, coordinates, env)? == eval(b, coordinates, env)?))
            }
            IndexExpr::Less(a, b) => {
                Some(BigInt::from(eval(a, coordinates, env)? < eval(b, coordinates, env)?))
            }
            IndexExpr::LessEqual(a, b) => {
                Some(BigInt::from(eval(a, coordinates, env)? <= eval(b, coordinates, env)?))
            }
            IndexExpr::Log2Ceil(value) => {
                let value = eval(value, coordinates, env)?.to_biguint()?;
                (!value.is_zero()).then(|| {
                    let floor = value.bits() - 1;
                    BigInt::from(if value == (BigUint::one() << floor as usize) {
                        floor
                    } else {
                        floor + 1
                    })
                })
            }
            IndexExpr::Select { selector, branches } => {
                let index = eval(selector, coordinates, env)?.to_usize()?;
                eval(branches.get(index)?, coordinates, env)
            }
        }
    }
    eval(expr, coordinates, env)?.to_usize()
}

fn value_family_shape(x: &AbstractValue) -> Option<Vec<usize>> {
    match x {
        AbstractValue::Family(family) => Some(family.shape.clone()),
        AbstractValue::Matrix(_) | AbstractValue::Trapdoor(_) => Some(Vec::new()),
        _ => None,
    }
}

fn trapdoor_public_input(value: &ExternalInputValue) -> Option<&str> {
    match value {
        ExternalInputValue::Trapdoor { public_matrix_input } => Some(public_matrix_input),
        ExternalInputValue::Family { element, .. } => trapdoor_public_input(element),
        _ => None,
    }
}

fn family_element(x: &Info) -> Option<AbstractValue> {
    match &x.value {
        AbstractValue::Family(f) => Some(f.element.as_ref().clone()),
        _ => None,
    }
}
fn validate_index(
    x: &Info,
    extent: usize,
    site: Option<DiagnosticSite>,
) -> Result<(), SimulationError> {
    let AbstractValue::Integer(range) = &x.value else {
        return Err(SimulationError::SelectorOutOfRange {
            message: "selector is not an integer range".into(),
            site,
        });
    };
    if range.minimum < 0.into() || range.maximum_inclusive >= BigInt::from(extent) {
        return Err(SimulationError::SelectorOutOfRange {
            message: format!(
                "selector range [{}, {}] is outside [0, {})",
                range.minimum, range.maximum_inclusive, extent
            ),
            site,
        });
    }
    Ok(())
}

fn validate_axis_selector(
    selector: &Info,
    shape: &[usize],
    axis: usize,
    site: Option<DiagnosticSite>,
) -> Result<(), SimulationError> {
    if let AbstractValue::Family(family) = &selector.value {
        let mut expected = shape.to_vec();
        expected.remove(axis);
        if family.shape != expected {
            return Err(SimulationError::SelectorOutOfRange {
                message: "family selector shape does not match the selected axis".into(),
                site,
            });
        }
        return validate_integer_range(family.element.as_ref(), shape[axis], site);
    }
    validate_index(selector, shape[axis], site)
}

fn validate_integer_range(
    value: &AbstractValue,
    extent: usize,
    site: Option<DiagnosticSite>,
) -> Result<(), SimulationError> {
    let AbstractValue::Integer(range) = value else {
        return Err(SimulationError::SelectorOutOfRange {
            message: "selector is not an integer range".into(),
            site,
        });
    };
    if range.minimum < 0.into() || range.maximum_inclusive >= BigInt::from(extent) {
        return Err(SimulationError::SelectorOutOfRange {
            message: format!(
                "selector range [{}, {}] is outside [0, {})",
                range.minimum, range.maximum_inclusive, extent
            ),
            site,
        });
    }
    Ok(())
}

fn index_expr_depends_axis(expr: &mxx_ir_core::IndexExpr, axis: usize) -> bool {
    use mxx_ir_core::IndexExpr;
    match expr {
        IndexExpr::Axis(index) => *index == axis,
        IndexExpr::Parameter(_) | IndexExpr::LoopIndex(_) | IndexExpr::Constant(_) => false,
        IndexExpr::Add(a, b) |
        IndexExpr::Subtract(a, b) |
        IndexExpr::Multiply(a, b) |
        IndexExpr::Divide(a, b) |
        IndexExpr::Remainder(a, b) |
        IndexExpr::Equal(a, b) |
        IndexExpr::Less(a, b) |
        IndexExpr::LessEqual(a, b) => {
            index_expr_depends_axis(a, axis) || index_expr_depends_axis(b, axis)
        }
        IndexExpr::Log2Ceil(value) => index_expr_depends_axis(value, axis),
        IndexExpr::Select { selector, branches } => {
            index_expr_depends_axis(selector, axis) ||
                branches.iter().any(|branch| index_expr_depends_axis(branch, axis))
        }
    }
}

fn eval_grid_index(
    expr: &mxx_ir_core::IndexExpr,
    env: &ParamEnv,
    axis_slots: &[u32],
) -> Result<usize, SimulationError> {
    let value = match expr {
        mxx_ir_core::IndexExpr::Axis(axis) => BigInt::from(
            env.loop_indices
                .get(axis_slots.get(*axis).ok_or_else(|| SimulationError::InvalidIndexMap {
                    message: "grid map axis is out of range".into(),
                    site: None,
                })?)
                .cloned()
                .unwrap_or_else(|| BigInt::from(*axis)),
        ),
        _ => expr.evaluate(env).map_err(|error| SimulationError::InvalidIndexMap {
            message: error.to_string(),
            site: None,
        })?,
    };
    value.to_usize().ok_or_else(|| SimulationError::InvalidIndexMap {
        message: "grid index is not a nonnegative usize".into(),
        site: None,
    })
}

fn specialize_relation(
    mut relation: RightPreimage,
    selectors: &[crate::SelectorId],
) -> RightPreimage {
    // Selector identity is the normalized semantic family view. Numeric
    // interval equality is intentionally not used as evidence of correlation.
    relation.selector = selectors.first().copied();
    relation
}
fn output_type(
    scope: &GraphScope,
    n: usize,
    env: &ParamEnv,
) -> Result<ConcreteMatrixType, SimulationError> {
    let ty = scope
        .node(mxx_ir_core::NodeId(n as u64))
        .and_then(|x| x.output_types().first())
        .ok_or_else(|| SimulationError::InvalidGraph {
            message: "matrix output type missing".into(),
            site: None,
        })?;
    concrete_matrix(ty, env).ok_or_else(|| SimulationError::InvalidGraph {
        message: "matrix output is not concrete".into(),
        site: None,
    })
}
fn output_family_shape(scope: &GraphScope, n: usize, env: &ParamEnv) -> Option<Vec<usize>> {
    let ty = scope.node(mxx_ir_core::NodeId(n as u64))?.output_types().first()?;
    match ty {
        WireType::Family { shape, .. } => {
            shape.iter().map(|x| x.evaluate(env).ok()?.to_usize()).collect()
        }
        _ => None,
    }
}
fn family_shape(ty: &WireType, env: &ParamEnv) -> Option<Vec<usize>> {
    match ty {
        WireType::Family { shape, .. } => {
            shape.iter().map(|x| x.evaluate(env).ok()?.to_usize()).collect()
        }
        _ => None,
    }
}

fn concrete_matrix(ty: &WireType, env: &ParamEnv) -> Option<ConcreteMatrixType> {
    let m = match ty {
        WireType::Matrix(m) | WireType::Preimage(m) => m,
        WireType::Family { element, .. } => return concrete_matrix(element, env),
        _ => return None,
    };
    Some(ConcreteMatrixType {
        modulus: m.modulus.evaluate(env).ok()?,
        ring_dimension: m.ring_dimension.evaluate(env).ok()?.to_usize()?,
        rows: m.rows.evaluate(env).ok()?.to_usize()?,
        columns: m.columns.evaluate(env).ok()?.to_usize()?,
    })
}
fn value_fact(ty: &WireType, f: &ExternalInputValue, env: &ParamEnv) -> Result<Info, String> {
    match (ty, f) {
        (
            WireType::Trapdoor {
                matrix,
                sigma,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
            },
            ExternalInputValue::Trapdoor { .. },
        ) => {
            let matrix = concrete_matrix(&WireType::Matrix(matrix.clone()), env)
                .ok_or("invalid trapdoor matrix")?;
            Ok(Info {
                value: AbstractValue::Trapdoor(TrapdoorState {
                    matrix,
                    sigma: sigma.clone(),
                    gadget_base: gadget_base.evaluate(env).map_err(|e| e.to_string())?,
                    digit_count: digit_count
                        .evaluate(env)
                        .map_err(|e| e.to_string())?
                        .to_usize()
                        .ok_or("invalid trapdoor digit count")?,
                    preimage_max_coefficient_bound: preimage_max_coefficient_bound
                        .evaluate(env)
                        .map_err(|e| e.to_string())?,
                }),
                ty: None,
                relation: None,
                view: crate::FamilyViewId(u32::MAX),
                paired_public: None,
            })
        }
        (
            WireType::Matrix(_) | WireType::Preimage(_),
            ExternalInputValue::Matrix {
                maximum_absolute_coefficient_error,
                maximum_absolute_coefficient_value,
                is_constant_polynomial,
            },
        ) => {
            let m = concrete_matrix(ty, env).ok_or("invalid matrix")?;
            let cap = crate::centered_residue_bound(&m.modulus).map_err(|e| e.to_string())?;
            Ok(Info {
                value: AbstractValue::Matrix(
                    MatrixState::new(
                        maximum_absolute_coefficient_error.clone(),
                        maximum_absolute_coefficient_value.clone().unwrap_or(cap).min(
                            crate::centered_residue_bound(&m.modulus).map_err(|e| e.to_string())?,
                        ),
                        *is_constant_polynomial,
                    )
                    .map_err(|e| e.to_string())?,
                ),
                ty: Some(m),
                relation: None,
                view: crate::FamilyViewId(u32::MAX),
                paired_public: None,
            })
        }
        (
            WireType::Family { element, shape },
            ExternalInputValue::Family { shape: actual, element: ef },
        ) => {
            let declared = shape
                .iter()
                .map(|e| e.evaluate(env).ok().and_then(|x| x.to_usize()))
                .collect::<Option<Vec<_>>>()
                .ok_or("invalid family shape")?;
            if declared != *actual {
                return Err("family shape mismatch".into());
            }
            let x = value_fact(element, ef, env)?;
            Ok(Info {
                value: AbstractValue::Family(
                    FamilyState::new(actual.clone(), x.value).map_err(|e| e.to_string())?,
                ),
                ty: x.ty,
                relation: None,
                view: crate::FamilyViewId(u32::MAX),
                paired_public: None,
            })
        }
        (
            WireType::Int | WireType::ConstantInt,
            ExternalInputValue::IntegerRange { minimum, maximum_inclusive },
        ) => Ok(Info {
            value: AbstractValue::Integer(
                state::IntegerState::new(minimum.clone(), maximum_inclusive.clone())
                    .map_err(|e| e.to_string())?,
            ),
            ty: None,
            relation: None,
            view: crate::FamilyViewId(u32::MAX),
            paired_public: None,
        }),
        (WireType::Bytes { .. }, ExternalInputValue::Bytes) => Ok(Info {
            value: AbstractValue::Bytes,
            ty: None,
            relation: None,
            view: crate::FamilyViewId(u32::MAX),
            paired_public: None,
        }),
        (WireType::Bool | WireType::ConstantBool, ExternalInputValue::Boolean) => Ok(Info {
            value: AbstractValue::Boolean(state::BooleanState::Either),
            ty: None,
            relation: None,
            view: crate::FamilyViewId(u32::MAX),
            paired_public: None,
        }),
        _ => Err("external fact does not match wire type".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        GraphOutput, NodeHandle, SubgraphHandle,
        artifact::{ArtifactConfidentiality, ProductionId},
        encoding::spec_hash,
        node::{ArtifactInput, ConstantMatrix},
        types::MatrixType,
        with_new_construction_scope,
    };
    use std::collections::BTreeMap;

    #[test]
    fn opaque_dynamic_selection_of_uniform_family_keeps_source_identity() {
        let parent = SourceLineage { shape: vec![2, 3], leaves: vec![crate::SourceId(7); 6] };
        let selected = uniform_gathered_lineage(&parent, &[]).unwrap();
        assert_eq!(selected.shape, Vec::<usize>::new());
        assert_eq!(selected.leaves, vec![crate::SourceId(7)]);
    }

    #[test]
    fn direct_constant_graph_evaluates_to_a_root_bound() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let node = NodeHandle::new(
            NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value: ConstantMatrix::Zero },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        );
        let graph = Graph::freeze(
            "direct",
            vec![],
            BTreeMap::from([(
                String::from("out"),
                GraphOutput { value: node.output(0).unwrap(), confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let env = ParamEnv::default();
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: crate::StageId("s".into()),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &env).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment: env,
            roots: vec![crate::SimulationRoot {
                stage: crate::StageId("s".into()),
                output: "out".into(),
            }],
            external_inputs: vec![],
            limits: crate::SimulationLimits::default(),
        };
        let report = run(&request).unwrap();
        assert_eq!(report.roots[0].maximum_absolute_coefficient_error, BigUint::ZERO);
        assert!(report.diagnostics.dropped_carriers.is_empty());
    }

    #[test]
    fn ring_automorphism_preserves_error_bound_and_carrier_path() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let sampled = NodeHandle::new(
            NodeKind::GaussianSample {
                matrix_type: matrix.clone(),
                sigma: mxx_ir_core::RealExpr::from_integer(1),
                max_coefficient_bound: 7.into(),
            },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let automorphed = NodeHandle::new(
            NodeKind::RingAutomorphism { index: 3.into() },
            vec![sampled.clone()],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "ring-automorphism-bound",
            vec![],
            BTreeMap::from([
                ("sampled".into(), GraphOutput { value: sampled, confidentiality: None }),
                ("automorphed".into(), GraphOutput { value: automorphed, confidentiality: None }),
            ]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let stage = crate::StageId("ring-automorphism-bound".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![
                crate::SimulationRoot { stage: stage.clone(), output: "sampled".into() },
                crate::SimulationRoot { stage, output: "automorphed".into() },
            ],
            external_inputs: vec![],
            limits: crate::SimulationLimits::default(),
        };
        let report = run(&request).unwrap();
        assert_eq!(report.roots[0].maximum_absolute_coefficient_error, BigUint::from(7u8));
        assert_eq!(
            report.roots[1].maximum_absolute_coefficient_error,
            report.roots[0].maximum_absolute_coefficient_error
        );
        assert!(report.diagnostics.dropped_carriers.is_empty());
    }

    #[test]
    fn multiply_drops_no_left_carrier_but_tensor_and_fma_report_right_carriers() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let constant = || {
            NodeHandle::new(
                NodeKind::ConstantMatrix {
                    matrix_type: matrix.clone(),
                    value: ConstantMatrix::Zero,
                },
                vec![],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap()
        };
        let multiply = NodeHandle::new(
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            vec![constant(), constant()],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let tensor = NodeHandle::new(
            NodeKind::Tensor,
            vec![constant(), constant()],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let fma = NodeHandle::new(
            NodeKind::MatrixMulAccumulate {
                coefficients: vec![mxx_ir_core::IntExpr::constant(1)],
                has_bias: true,
            },
            vec![constant(), constant(), constant()],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "carrier-drop-filtering",
            vec![],
            BTreeMap::from([
                ("multiply".into(), GraphOutput { value: multiply, confidentiality: None }),
                ("tensor".into(), GraphOutput { value: tensor, confidentiality: None }),
                ("fma".into(), GraphOutput { value: fma, confidentiality: None }),
            ]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let stage = crate::StageId("carrier-drop-filtering".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![
                crate::SimulationRoot { stage: stage.clone(), output: "multiply".into() },
                crate::SimulationRoot { stage: stage.clone(), output: "tensor".into() },
                crate::SimulationRoot { stage: stage.clone(), output: "fma".into() },
            ],
            external_inputs: vec![],
            limits: crate::SimulationLimits::default(),
        };
        let diagnostics = run(&request).unwrap().diagnostics.dropped_carriers;
        assert!(diagnostics.iter().all(|diagnostic| {
            !diagnostic
                .site
                .operation
                .as_deref()
                .is_some_and(|operation| operation.contains("MatrixBinary(Multiply)"))
        }));
        let tensor_drops = diagnostics
            .iter()
            .filter(|diagnostic| {
                diagnostic.site.operation.as_deref().is_some_and(|operation| operation == "Tensor")
            })
            .collect::<Vec<_>>();
        assert_eq!(tensor_drops.len(), 1);
        assert!(tensor_drops[0].expected_source.is_some());
        let fma_drops = diagnostics
            .iter()
            .filter(|diagnostic| {
                diagnostic
                    .site
                    .operation
                    .as_deref()
                    .is_some_and(|operation| operation.contains("MatrixMulAccumulate"))
            })
            .collect::<Vec<_>>();
        assert_eq!(fma_drops.len(), 2);
        assert!(fma_drops.iter().all(|diagnostic| diagnostic.expected_source.is_some()));
    }

    #[test]
    fn constant_polynomial_and_identity_avoid_ring_convolution_factor() {
        let scalar = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(1009),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let left = NodeHandle::new(
            NodeKind::Input {
                name: "left".into(),
                wire_type: WireType::Matrix(scalar.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(scalar.clone())],
        )
        .output(0)
        .unwrap();
        let identity = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: scalar.clone(),
                value: ConstantMatrix::Identity,
            },
            vec![],
            vec![WireType::Matrix(scalar.clone())],
        )
        .output(0)
        .unwrap();
        let polynomial = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: scalar.clone(),
                value: ConstantMatrix::Polynomial {
                    coefficients: vec![mxx_ir_core::IntExpr::constant(2)],
                },
            },
            vec![],
            vec![WireType::Matrix(scalar.clone())],
        )
        .output(0)
        .unwrap();
        let identity_product = NodeHandle::new(
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            vec![left.clone(), identity],
            vec![WireType::Matrix(scalar.clone())],
        )
        .output(0)
        .unwrap();
        let polynomial_product = NodeHandle::new(
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            vec![left, polynomial],
            vec![WireType::Matrix(scalar.clone())],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "constant-polynomial-product",
            vec![],
            BTreeMap::from([
                ("identity".into(), GraphOutput { value: identity_product, confidentiality: None }),
                (
                    "polynomial".into(),
                    GraphOutput { value: polynomial_product, confidentiality: None },
                ),
            ]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let stage = crate::StageId("constant-polynomial-product".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![
                crate::SimulationRoot { stage: stage.clone(), output: "identity".into() },
                crate::SimulationRoot { stage: stage.clone(), output: "polynomial".into() },
            ],
            external_inputs: vec![crate::ExternalInputFact {
                stage,
                input: "left".into(),
                value: crate::ExternalInputValue::Matrix {
                    maximum_absolute_coefficient_error: 3u8.into(),
                    maximum_absolute_coefficient_value: Some(7u8.into()),
                    is_constant_polynomial: false,
                },
            }],
            limits: crate::SimulationLimits::default(),
        };
        let report = run(&request).unwrap();
        assert_eq!(report.roots[0].maximum_absolute_coefficient_error, BigUint::from(3u8));
        assert_eq!(report.roots[1].maximum_absolute_coefficient_error, BigUint::from(6u8));
        assert!(report.diagnostics.dropped_carriers.is_empty());
    }

    #[test]
    fn constant_gadget_uses_concrete_matrix_digit_count_for_magnitude() {
        let left_type = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(1009),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(2),
        };
        let gadget_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(2),
            columns: mxx_ir_core::IntExpr::constant(6),
            ..left_type.clone()
        };
        let output_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(6),
            ..left_type.clone()
        };
        let left = NodeHandle::new(
            NodeKind::Input {
                name: "left".into(),
                wire_type: WireType::Matrix(left_type.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(left_type.clone())],
        )
        .output(0)
        .unwrap();
        let gadget = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: gadget_type.clone(),
                value: ConstantMatrix::Gadget {
                    base: mxx_ir_core::IntExpr::constant(4),
                    small: false,
                },
            },
            vec![],
            vec![WireType::Matrix(gadget_type)],
        )
        .output(0)
        .unwrap();
        let product = NodeHandle::new(
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            vec![left, gadget],
            vec![WireType::Matrix(output_type)],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "constant-gadget-product",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: product, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let stage = crate::StageId("constant-gadget-product".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage: stage.clone(), output: "out".into() }],
            external_inputs: vec![crate::ExternalInputFact {
                stage,
                input: "left".into(),
                value: crate::ExternalInputValue::Matrix {
                    maximum_absolute_coefficient_error: 3u8.into(),
                    maximum_absolute_coefficient_value: Some(7u8.into()),
                    is_constant_polynomial: false,
                },
            }],
            limits: crate::SimulationLimits::default(),
        };
        assert_eq!(
            run(&request).unwrap().roots[0].maximum_absolute_coefficient_error,
            BigUint::from(96u8)
        );
    }

    #[test]
    fn rectangular_matrix_multiply_chain_uses_actual_intermediate_columns() {
        let matrix_type = |rows, columns| MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(1009),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(rows),
            columns: mxx_ir_core::IntExpr::constant(columns),
        };
        let left_type = matrix_type(1, 2);
        let middle_type = matrix_type(2, 3);
        let intermediate_type = matrix_type(1, 3);
        let right_type = matrix_type(3, 1);
        let output_type = matrix_type(1, 1);
        let input = |name: &str, ty: MatrixType| {
            NodeHandle::new(
                NodeKind::Input {
                    name: name.into(),
                    wire_type: WireType::Matrix(ty.clone()),
                    artifact: None,
                },
                vec![],
                vec![WireType::Matrix(ty)],
            )
            .output(0)
            .unwrap()
        };
        let left = input("left", left_type);
        let middle = input("middle", middle_type);
        let right = input("right", right_type);
        let first = NodeHandle::new(
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            vec![left, middle],
            vec![WireType::Matrix(intermediate_type)],
        )
        .output(0)
        .unwrap();
        let chained = NodeHandle::new(
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            vec![first, right],
            vec![WireType::Matrix(output_type)],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "rectangular-multiply-chain",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: chained, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let stage = crate::StageId("rectangular-multiply-chain".into());
        let fact = |input: &str, error: u8, magnitude: u8| crate::ExternalInputFact {
            stage: stage.clone(),
            input: input.into(),
            value: crate::ExternalInputValue::Matrix {
                maximum_absolute_coefficient_error: error.into(),
                maximum_absolute_coefficient_value: Some(magnitude.into()),
                is_constant_polynomial: true,
            },
        };
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage: stage.clone(), output: "out".into() }],
            external_inputs: vec![fact("left", 1, 2), fact("middle", 3, 4), fact("right", 5, 6)],
            limits: crate::SimulationLimits::default(),
        };
        assert_eq!(
            run(&request).unwrap().roots[0].maximum_absolute_coefficient_error,
            BigUint::from(600u16)
        );
    }

    #[test]
    fn rectangular_apply_preimage_chain_uses_actual_intermediate_columns() {
        let matrix_type = |rows, columns| MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(1009),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(rows),
            columns: mxx_ir_core::IntExpr::constant(columns),
        };
        let public_type = matrix_type(1, 2);
        let preimage_type = matrix_type(2, 3);
        let target_type = matrix_type(1, 3);
        let right_type = matrix_type(3, 1);
        let output_type = matrix_type(1, 1);
        let public = NodeHandle::new(
            NodeKind::Input {
                name: "public".into(),
                wire_type: WireType::Matrix(public_type.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let trapdoor_type = WireType::Trapdoor {
            matrix: public_type.clone(),
            sigma: mxx_ir_core::RealExpr::from_integer(1),
            gadget_base: mxx_ir_core::IntExpr::constant(2),
            digit_count: mxx_ir_core::IntExpr::constant(1),
            preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(5),
        };
        let trapdoor = NodeHandle::new(
            NodeKind::Input {
                name: "trapdoor".into(),
                wire_type: trapdoor_type.clone(),
                artifact: None,
            },
            vec![],
            vec![trapdoor_type],
        )
        .output(0)
        .unwrap();
        let target = NodeHandle::new(
            NodeKind::Input {
                name: "target".into(),
                wire_type: WireType::Matrix(target_type.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let preimage = NodeHandle::new(
            NodeKind::PreimageSample {
                matrix_type: preimage_type.clone(),
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(5),
            },
            vec![public.clone(), trapdoor, target],
            vec![WireType::Preimage(preimage_type)],
        )
        .output(0)
        .unwrap();
        let applied = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![public, preimage],
            vec![WireType::Matrix(target_type)],
        )
        .output(0)
        .unwrap();
        let right = NodeHandle::new(
            NodeKind::Input {
                name: "right".into(),
                wire_type: WireType::Matrix(right_type.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(right_type)],
        )
        .output(0)
        .unwrap();
        let chained = NodeHandle::new(
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            vec![applied, right],
            vec![WireType::Matrix(output_type)],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "rectangular-apply-preimage-chain",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: chained, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let stage = crate::StageId("rectangular-apply-preimage-chain".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage: stage.clone(), output: "out".into() }],
            external_inputs: vec![
                crate::ExternalInputFact {
                    stage: stage.clone(),
                    input: "public".into(),
                    value: crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: BigUint::ZERO,
                        maximum_absolute_coefficient_value: Some(2u8.into()),
                        is_constant_polynomial: true,
                    },
                },
                crate::ExternalInputFact {
                    stage: stage.clone(),
                    input: "trapdoor".into(),
                    value: crate::ExternalInputValue::Trapdoor {
                        public_matrix_input: "public".into(),
                    },
                },
                crate::ExternalInputFact {
                    stage: stage.clone(),
                    input: "target".into(),
                    value: crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: 3u8.into(),
                        maximum_absolute_coefficient_value: Some(4u8.into()),
                        is_constant_polynomial: true,
                    },
                },
                crate::ExternalInputFact {
                    stage,
                    input: "right".into(),
                    value: crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: 6u8.into(),
                        maximum_absolute_coefficient_value: Some(7u8.into()),
                        is_constant_polynomial: true,
                    },
                },
            ],
            limits: crate::SimulationLimits::default(),
        };
        assert_eq!(
            run(&request).unwrap().roots[0].maximum_absolute_coefficient_error,
            BigUint::from(423u16)
        );
    }

    #[test]
    fn boolean_and_integer_select_transfers_are_typed() {
        let matrix = ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 1,
            rows: 1,
            columns: 1,
        };
        let bool_info = Info {
            value: AbstractValue::Boolean(state::BooleanState::Either),
            ty: None,
            relation: None,
            view: crate::FamilyViewId(u32::MAX),
            paired_public: None,
        };
        let joined = join_uniform(bool_info.clone(), bool_info, Some(&matrix)).unwrap();
        assert!(matches!(joined.value, AbstractValue::Boolean(_)));
        let left = integer((-2).into());
        let right = integer(4.into());
        let joined = join_uniform(left, right, Some(&matrix)).unwrap();
        assert!(matches!(
            joined.value,
            AbstractValue::Integer(state::IntegerState { minimum, maximum_inclusive })
                if minimum == BigInt::from(-2) && maximum_inclusive == BigInt::from(4)
        ));
    }

    #[test]
    fn dynamic_select_reports_distinct_source_carrier_loss() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let selector = NodeHandle::new(
            NodeKind::Input { name: "selector".into(), wire_type: WireType::Int, artifact: None },
            vec![],
            vec![WireType::Int],
        )
        .output(0)
        .unwrap();
        let branch = || {
            NodeHandle::new(
                NodeKind::ConstantMatrix {
                    matrix_type: matrix.clone(),
                    value: ConstantMatrix::Zero,
                },
                vec![],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap()
        };
        let selected = NodeHandle::new(
            NodeKind::Select { count: mxx_ir_core::IntExpr::constant(2) },
            vec![selector, branch(), branch()],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "dynamic-select-diagnostics",
            vec![],
            BTreeMap::from([(
                "out".into(),
                GraphOutput { value: selected, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let stage = crate::StageId("dynamic-select-diagnostics".into());
        let environment = ParamEnv::default();
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage: stage.clone(), output: "out".into() }],
            external_inputs: vec![crate::ExternalInputFact {
                stage,
                input: "selector".into(),
                value: crate::ExternalInputValue::IntegerRange {
                    minimum: 0.into(),
                    maximum_inclusive: 1.into(),
                },
            }],
            limits: crate::SimulationLimits::default(),
        };
        let diagnostics = run(&request).unwrap().diagnostics.dropped_carriers;
        assert!(!diagnostics.is_empty());
        assert!(diagnostics.iter().all(|diagnostic| {
            diagnostic
                .site
                .operation
                .as_deref()
                .is_some_and(|operation| operation.contains("Select"))
        }));
        let sources = diagnostics
            .iter()
            .filter_map(|diagnostic| diagnostic.expected_source)
            .collect::<HashSet<_>>();
        assert!(sources.len() >= 2, "distinct branch sources must be exposed: {diagnostics:?}");
    }

    #[test]
    fn matrix_select_joins_same_source_with_maximum_gain() {
        let matrix = ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 1,
            rows: 1,
            columns: 1,
        };
        let branch = |source: u32, gain: u32| Info {
            value: AbstractValue::Matrix(MatrixState {
                error_bound: 2u8.into(),
                coefficient_magnitude_bound: 5u8.into(),
                is_constant_polynomial: true,
                right_carrier: Some(crate::RightCarrier {
                    source: crate::SourceId(source),
                    left_gain: BigUint::from(gain),
                }),
            }),
            ty: Some(matrix.clone()),
            relation: None,
            view: crate::FamilyViewId(u32::MAX),
            paired_public: None,
        };

        let joined = join_uniform(branch(7, 3), branch(7, 11), Some(&matrix)).unwrap();
        let AbstractValue::Matrix(joined) = joined.value else {
            panic!("matrix branches must join to a matrix")
        };
        assert_eq!(
            joined.right_carrier,
            Some(crate::RightCarrier {
                source: crate::SourceId(7),
                left_gain: BigUint::from(11u8),
            })
        );
    }

    #[test]
    fn matrix_select_drops_distinct_sources_and_nonzero_carrierless_branches() {
        let matrix = ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 1,
            rows: 1,
            columns: 1,
        };
        let branch = |source: Option<u32>| Info {
            value: AbstractValue::Matrix(MatrixState {
                error_bound: 2u8.into(),
                coefficient_magnitude_bound: 5u8.into(),
                is_constant_polynomial: true,
                right_carrier: source.map(|source| crate::RightCarrier {
                    source: crate::SourceId(source),
                    left_gain: BigUint::from(3u8),
                }),
            }),
            ty: Some(matrix.clone()),
            relation: None,
            view: crate::FamilyViewId(u32::MAX),
            paired_public: None,
        };

        for joined in [
            join_uniform(branch(Some(7)), branch(Some(8)), Some(&matrix)).unwrap(),
            join_uniform(branch(Some(7)), branch(None), Some(&matrix)).unwrap(),
        ] {
            let AbstractValue::Matrix(joined) = joined.value else {
                panic!("matrix branches must join to a matrix")
            };
            assert_eq!(joined.right_carrier, None);
        }
    }

    #[test]
    fn integer_select_does_not_require_matrix_type_metadata() {
        let selector =
            NodeHandle::new(NodeKind::ConstantInt(0.into()), vec![], vec![WireType::ConstantInt])
                .output(0)
                .unwrap();
        let left = NodeHandle::new(
            NodeKind::ConstantInt((-3).into()),
            vec![],
            vec![WireType::ConstantInt],
        )
        .output(0)
        .unwrap();
        let right =
            NodeHandle::new(NodeKind::ConstantInt(7.into()), vec![], vec![WireType::ConstantInt])
                .output(0)
                .unwrap();
        let selected = NodeHandle::new(
            NodeKind::Select { count: mxx_ir_core::IntExpr::constant(2) },
            vec![selector, left, right],
            vec![WireType::ConstantInt],
        )
        .output(0)
        .unwrap();
        let matrix_type = mxx_ir_core::types::MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let output = NodeHandle::new(
            NodeKind::LiftIntegerToConstantPolynomial { matrix_type: matrix_type.clone() },
            vec![selected],
            vec![WireType::Matrix(matrix_type)],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "integer-select",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: output, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let stage = crate::StageId("integer-select".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage, output: "out".into() }],
            external_inputs: vec![],
            limits: crate::SimulationLimits::default(),
        };
        let _ = request;
        let joined = join_uniform(integer((-3).into()), integer(7.into()), None).unwrap();
        assert!(matches!(joined.value, AbstractValue::Integer(_)));
    }

    #[test]
    fn structural_bindings_and_grid_indices_are_substituted() {
        let mut env = ParamEnv::default();
        env.loop_indices.insert(3, BigInt::from(5));
        let env =
            apply_bindings(env, &[("bound".into(), mxx_ir_core::IntExpr::LoopIndex(3))]).unwrap();
        assert_eq!(env.integers["bound"], BigInt::from(5));
        let index = mxx_ir_core::IndexExpr::Axis(0);
        assert_eq!(eval_grid_index(&index, &env, &[3]).unwrap(), 5);
    }

    #[test]
    fn parallel_grid_joins_loop_index_dependent_lanes() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(97),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let noisy = NodeHandle::new(
            NodeKind::Input {
                name: "noisy".into(),
                wire_type: WireType::Matrix(matrix.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let body = with_new_construction_scope(|scope| {
            let body_noisy = NodeHandle::new(
                NodeKind::Input {
                    name: "body-noisy".into(),
                    wire_type: WireType::Matrix(matrix.clone()),
                    artifact: None,
                },
                vec![],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap();
            let zero = NodeHandle::new(
                NodeKind::ConstantMatrix {
                    matrix_type: matrix.clone(),
                    value: ConstantMatrix::Zero,
                },
                vec![],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap();
            let selector = NodeHandle::new(
                NodeKind::EvaluateInt(mxx_ir_core::IntExpr::LoopIndex(0)),
                vec![],
                vec![WireType::ConstantInt],
            )
            .output(0)
            .unwrap();
            let selected = NodeHandle::new(
                NodeKind::Select { count: mxx_ir_core::IntExpr::constant(2) },
                vec![selector, zero, body_noisy.clone()],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("loop-index-select-body", scope, vec![body_noisy], vec![selected])
                .unwrap()
        });
        let family_type = WireType::Family {
            element: Box::new(WireType::Matrix(matrix)),
            shape: vec![mxx_ir_core::IntExpr::constant(2)],
        };
        let output = NodeHandle::parallel_grid(
            body,
            vec![noisy],
            vec![family_type],
            mxx_ir_core::node::ParallelGrid {
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
                index_slots: vec![0],
                bindings: vec![],
                input_modes: vec![mxx_ir_core::node::GridInputMode::Broadcast],
            },
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "loop-index-select-grid",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: output, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let stage = crate::StageId("loop-index-select-grid".into());
        let report = run(&SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage: stage.clone(), output: "out".into() }],
            external_inputs: vec![crate::ExternalInputFact {
                stage,
                input: "noisy".into(),
                value: crate::ExternalInputValue::Matrix {
                    maximum_absolute_coefficient_error: 7u8.into(),
                    maximum_absolute_coefficient_value: Some(7u8.into()),
                    is_constant_polynomial: true,
                },
            }],
            limits: crate::SimulationLimits::default(),
        })
        .unwrap();
        assert_eq!(report.roots[0].maximum_absolute_coefficient_error, 7u8.into());
    }

    #[test]
    fn scalar_grid_source_reindex_matches_shared_preimage_group_source() {
        let request = SimulationRequest {
            program: crate::SimulationProgram { stages: Vec::new() },
            environment: ParamEnv::default(),
            roots: Vec::new(),
            external_inputs: Vec::new(),
            limits: crate::SimulationLimits::default(),
        };
        let mut evaluator = Evaluator {
            request: &request,
            stages: HashMap::new(),
            visiting: HashSet::new(),
            sources: HashMap::new(),
            gadget_sources: HashMap::new(),
            source_lineages: HashMap::new(),
            lineage_sources: HashMap::new(),
            mapped_sources: HashMap::new(),
            gathered_sources: HashMap::new(),
            binder_sources: HashMap::new(),
            next_source: 0,
            preimages: HashMap::new(),
            states: HashMap::new(),
            selector_views: HashMap::new(),
            next_selector: 0,
            planned: 0,
            transfers: 0,
            dropped: Vec::new(),
            interners: crate::identity::Interners::default(),
            reached: HashSet::new(),
            artifact_outputs: BTreeMap::new(),
        };
        let primitive = evaluator.source_for(
            &crate::StageId("source-lineage".into()),
            &FrozenGraphScopeId::Root,
            &[],
            0,
            "public",
        );

        // A trapdoor sampled inside a six-lane grid denotes six distinct
        // public sources, even though the symbolic body is evaluated once.
        let flat = evaluator.lift_source_for_shape(primitive, vec![6]);
        let flat_lineage = evaluator.source_lineages.get(&flat).unwrap();
        assert_eq!(flat_lineage.shape, vec![6]);
        assert_eq!(flat_lineage.leaves.len(), 6);
        assert_eq!(flat_lineage.leaves.iter().copied().collect::<HashSet<_>>().len(), 6);

        // The public gadget relation is index-independent: every grid lane
        // consumes a preimage of the same G rather than sampling a new source.
        let gadget = evaluator.source_for(
            &crate::StageId("source-lineage".into()),
            &FrozenGraphScopeId::Root,
            &[],
            1,
            "gadget",
        );
        evaluator.gadget_sources.insert(
            crate::GadgetDescriptor {
                modulus: BigInt::from(17),
                ring_dimension: 8,
                rows: 1,
                columns: 2,
                base: BigInt::from(2),
                digit_count: 2,
                small: false,
            },
            gadget,
        );
        let gadget_family = evaluator.lift_source_for_shape(gadget, vec![6]);
        assert_eq!(
            gadget_family, gadget,
            "an index-independent gadget family keeps the scalar source identity"
        );
        assert_eq!(
            evaluator.mapped_source_for(
                gadget,
                &mxx_ir_core::IndexMap::new([mxx_ir_core::IndexExpr::Axis(0)]),
                vec![6],
                None,
            ),
            gadget,
            "reindexing an index-independent gadget does not create a new source"
        );
        assert_eq!(
            evaluator.gathered_source_for(gadget, vec![crate::SelectorId(4)], Vec::new()),
            gadget,
            "gathering an index-independent gadget does not create a new source"
        );

        let flatten_map = mxx_ir_core::IndexMap::new([mxx_ir_core::IndexExpr::Add(
            Box::new(mxx_ir_core::IndexExpr::Multiply(
                Box::new(mxx_ir_core::IndexExpr::Axis(0)),
                Box::new(mxx_ir_core::IndexExpr::constant(2)),
            )),
            Box::new(mxx_ir_core::IndexExpr::Axis(1)),
        )]);
        let bases = evaluator.mapped_source_for(flat, &flatten_map, vec![3, 2], None);
        let group = evaluator.mapped_source_for(
            bases,
            &mxx_ir_core::IndexMap::new([
                mxx_ir_core::IndexExpr::Axis(0),
                mxx_ir_core::IndexExpr::Axis(1),
            ]),
            vec![2, 2],
            None,
        );
        let left = evaluator.gathered_source_for_concrete(
            bases,
            vec![crate::SelectorId(0), crate::SelectorId(1)],
            Vec::new(),
            Some(&[0, 1]),
        );
        let relation = evaluator.gathered_source_for_concrete(
            group,
            vec![crate::SelectorId(0), crate::SelectorId(1), crate::SelectorId(2)],
            Vec::new(),
            Some(&[0, 1, 1]),
        );
        assert_eq!(left, relation);

        // Selecting the final branch axis of a [group, branch] relation
        // changes the target/preimage view, not the grouped public source.
        let selected = evaluator.source_after_axis_selection(
            group,
            true,
            1,
            2,
            vec![crate::SelectorId(3)],
            vec![2],
        );
        assert_eq!(selected, group);

        // Opaque selectors over an ordinary non-uniform family remain
        // distinct even when they descend from the same family root. Only a
        // source explicitly consumed by FamilyPreimageSample receives the
        // index-independent relation contract used by grid-lane joining.
        let opaque_left = evaluator.gathered_source_for(
            bases,
            vec![crate::SelectorId(10), crate::SelectorId(11)],
            Vec::new(),
        );
        let opaque_right = evaluator.gathered_source_for(
            bases,
            vec![crate::SelectorId(12), crate::SelectorId(13)],
            Vec::new(),
        );
        assert_ne!(opaque_left, opaque_right);
        assert_eq!(
            evaluator.selector_for(crate::FamilyViewId(10)),
            evaluator.selector_for(crate::FamilyViewId(10)),
            "one normalized semantic view must retain one selector identity"
        );
        assert_ne!(
            evaluator.selector_for(crate::FamilyViewId(10)),
            evaluator.selector_for(crate::FamilyViewId(11)),
            "different semantic views must remain uncorrelated"
        );

        // A transition target may retain a final preimage-branch axis while
        // its public carrier is B[level + 1, state].  After the branch is
        // selected and the state is carried out of a sequential loop, it must
        // be the same source function as a separate final-level reindex.
        let final_map = mxx_ir_core::IndexMap::new([
            mxx_ir_core::IndexExpr::constant(2),
            mxx_ir_core::IndexExpr::Axis(0),
        ]);
        let final_direct = evaluator.mapped_source_for(bases, &final_map, vec![2], None);
        let transition_targets = evaluator.mapped_source_for(bases, &final_map, vec![2, 3], None);
        let carried = uniform_axis_selection_lineage(
            evaluator.source_lineages.get(&transition_targets).unwrap(),
            1,
            &[2],
        )
        .unwrap();
        let carried = evaluator.canonical_source_for_lineage(carried);
        assert_eq!(carried, final_direct);

        // An identically shaped but independently sampled base family remains
        // distinct; canonical coordinate composition is not source aliasing.
        let other_primitive = evaluator.source_for(
            &crate::StageId("source-lineage".into()),
            &FrozenGraphScopeId::Root,
            &[],
            2,
            "public",
        );
        let other_flat = evaluator.lift_source_for_shape(other_primitive, vec![6]);
        let other_bases = evaluator.mapped_source_for(other_flat, &flatten_map, vec![3, 2], None);
        let other_final = evaluator.mapped_source_for(other_bases, &final_map, vec![2], None);
        assert_ne!(final_direct, other_final);
    }

    #[test]
    fn scalar_preimage_graph_uses_paired_public_and_strict_consumer() {
        let public_type = mxx_ir_core::types::MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(8),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(2),
        };
        let output_type = mxx_ir_core::types::MatrixType {
            rows: mxx_ir_core::IntExpr::constant(2),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..public_type.clone()
        };
        let target_type = mxx_ir_core::types::MatrixType {
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..public_type.clone()
        };
        let public = NodeHandle::new(
            NodeKind::Input {
                name: "public".into(),
                wire_type: WireType::Matrix(public_type.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let trapdoor_wire = WireType::Trapdoor {
            matrix: public_type.clone(),
            sigma: mxx_ir_core::RealExpr::from_integer(1),
            gadget_base: mxx_ir_core::IntExpr::constant(2),
            digit_count: mxx_ir_core::IntExpr::constant(1),
            preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
        };
        let trapdoor = NodeHandle::new(
            NodeKind::Input {
                name: "trapdoor".into(),
                wire_type: trapdoor_wire.clone(),
                artifact: None,
            },
            vec![],
            vec![trapdoor_wire],
        )
        .output(0)
        .unwrap();
        let target = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: target_type.clone(),
                value: ConstantMatrix::Zero,
            },
            vec![],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let family_target = NodeHandle::new(
            NodeKind::FamilyPack { shape: vec![mxx_ir_core::IntExpr::constant(2)] },
            vec![target.clone(), target.clone()],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(target_type.clone())),
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let family_preimage = NodeHandle::new(
            NodeKind::FamilyPreimageSample {
                matrix_type: output_type.clone(),
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
            },
            vec![public.clone(), trapdoor.clone(), family_target],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let family_selected = NodeHandle::new(
            NodeKind::FamilySelectAxis { axis: 0 },
            vec![
                family_preimage,
                NodeHandle::new(
                    NodeKind::ConstantInt(0.into()),
                    vec![],
                    vec![WireType::ConstantInt],
                )
                .output(0)
                .unwrap(),
            ],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let preimage = NodeHandle::new(
            NodeKind::PreimageSample {
                matrix_type: output_type.clone(),
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
            },
            vec![public.clone(), trapdoor, target.clone()],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let unrelated = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: public_type.clone(),
                value: ConstantMatrix::Zero,
            },
            vec![],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let fallback = NodeHandle::new(
            NodeKind::MatrixBinary(MatrixBinaryOp::Multiply),
            vec![unrelated, preimage.clone()],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let applied = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![public, preimage],
            vec![WireType::Matrix(target_type)],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "scalar-preimage",
            vec![],
            BTreeMap::from([
                ("out".into(), GraphOutput { value: applied, confidentiality: None }),
                ("fallback".into(), GraphOutput { value: fallback, confidentiality: None }),
                ("family".into(), GraphOutput { value: family_selected, confidentiality: None }),
            ]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let environment = ParamEnv::default();
        let stage_id = crate::StageId("preimage".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage_id.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![
                crate::SimulationRoot { stage: stage_id.clone(), output: "out".into() },
                crate::SimulationRoot { stage: stage_id.clone(), output: "fallback".into() },
                crate::SimulationRoot { stage: stage_id.clone(), output: "family".into() },
            ],
            external_inputs: vec![
                crate::ExternalInputFact {
                    stage: stage_id.clone(),
                    input: "public".into(),
                    value: crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: BigUint::ZERO,
                        maximum_absolute_coefficient_value: None,
                        is_constant_polynomial: false,
                    },
                },
                crate::ExternalInputFact {
                    stage: stage_id.clone(),
                    input: "trapdoor".into(),
                    value: crate::ExternalInputValue::Trapdoor {
                        public_matrix_input: "public".into(),
                    },
                },
            ],
            limits: crate::SimulationLimits::default(),
        };
        assert!(run(&request).is_ok());
    }

    #[test]
    fn exact_decomposition_target_can_be_materialized() {
        let target_type = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let decomposition_type =
            MatrixType { rows: mxx_ir_core::IntExpr::constant(2), ..target_type.clone() };
        let target = NodeHandle::new(
            NodeKind::Input {
                name: "target".into(),
                wire_type: WireType::Matrix(target_type.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(target_type)],
        )
        .output(0)
        .unwrap();
        let decomposition = NodeHandle::new(
            NodeKind::GadgetDecompose {
                base: mxx_ir_core::IntExpr::constant(4),
                small: false,
                digit_count: mxx_ir_core::IntExpr::constant(2),
            },
            vec![target],
            vec![WireType::Preimage(decomposition_type.clone())],
        )
        .output(0)
        .unwrap();
        let materialized = NodeHandle::new(
            NodeKind::MaterializePreimageExact,
            vec![decomposition],
            vec![WireType::Matrix(decomposition_type)],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "exact-decomposition-materialization",
            vec![],
            BTreeMap::from([(
                "out".into(),
                GraphOutput { value: materialized, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let environment = ParamEnv::default();
        let stage = crate::StageId("exact-decomposition-materialization".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage: stage.clone(), output: "out".into() }],
            external_inputs: vec![crate::ExternalInputFact {
                stage,
                input: "target".into(),
                value: crate::ExternalInputValue::Matrix {
                    maximum_absolute_coefficient_error: BigUint::ZERO,
                    maximum_absolute_coefficient_value: Some(7u8.into()),
                    is_constant_polynomial: false,
                },
            }],
            limits: crate::SimulationLimits::default(),
        };
        let result = run(&request).unwrap();
        assert_eq!(result.roots[0].maximum_absolute_coefficient_error, BigUint::ZERO);
    }

    #[test]
    fn explicit_gadget_and_decomposition_share_one_canonical_source() {
        let target_type = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let decomposition_type =
            MatrixType { rows: mxx_ir_core::IntExpr::constant(2), ..target_type.clone() };
        let gadget_type =
            MatrixType { columns: mxx_ir_core::IntExpr::constant(2), ..target_type.clone() };
        let target = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: target_type.clone(),
                value: ConstantMatrix::Zero,
            },
            vec![],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let decomposition = NodeHandle::new(
            NodeKind::GadgetDecompose {
                base: mxx_ir_core::IntExpr::constant(4),
                small: false,
                digit_count: mxx_ir_core::IntExpr::constant(2),
            },
            vec![target],
            vec![WireType::Preimage(decomposition_type)],
        )
        .output(0)
        .unwrap();
        let gadget = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: gadget_type.clone(),
                value: ConstantMatrix::Gadget {
                    base: mxx_ir_core::IntExpr::constant(4),
                    small: false,
                },
            },
            vec![],
            vec![WireType::Matrix(gadget_type)],
        )
        .output(0)
        .unwrap();
        let product = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![gadget, decomposition],
            vec![WireType::Matrix(target_type)],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "canonical-gadget-source",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: product, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let environment = ParamEnv::default();
        let stage = crate::StageId("canonical-gadget-source".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage, output: "out".into() }],
            external_inputs: vec![],
            limits: crate::SimulationLimits::default(),
        };
        assert_eq!(
            run(&request).unwrap().roots[0].maximum_absolute_coefficient_error,
            BigUint::ZERO
        );
    }

    #[test]
    fn noisy_sampled_preimage_target_cannot_be_materialized() {
        let public_type = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(2),
        };
        let preimage_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(2),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..public_type.clone()
        };
        let target_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..public_type.clone()
        };
        let public = NodeHandle::new(
            NodeKind::Input {
                name: "public".into(),
                wire_type: WireType::Matrix(public_type.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let trapdoor_type = WireType::Trapdoor {
            matrix: public_type.clone(),
            sigma: mxx_ir_core::RealExpr::from_integer(1),
            gadget_base: mxx_ir_core::IntExpr::constant(2),
            digit_count: mxx_ir_core::IntExpr::constant(1),
            preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
        };
        let trapdoor = NodeHandle::new(
            NodeKind::Input {
                name: "trapdoor".into(),
                wire_type: trapdoor_type.clone(),
                artifact: None,
            },
            vec![],
            vec![trapdoor_type],
        )
        .output(0)
        .unwrap();
        let target = NodeHandle::new(
            NodeKind::Input {
                name: "target".into(),
                wire_type: WireType::Matrix(target_type.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(target_type)],
        )
        .output(0)
        .unwrap();
        let preimage = NodeHandle::new(
            NodeKind::PreimageSample {
                matrix_type: preimage_type.clone(),
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
            },
            vec![public, trapdoor, target],
            vec![WireType::Preimage(preimage_type.clone())],
        )
        .output(0)
        .unwrap();
        let materialized = NodeHandle::new(
            NodeKind::MaterializePreimageExact,
            vec![preimage],
            vec![WireType::Matrix(preimage_type)],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "noisy-preimage-materialization",
            vec![],
            BTreeMap::from([(
                "out".into(),
                GraphOutput { value: materialized, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let environment = ParamEnv::default();
        let stage = crate::StageId("noisy-preimage-materialization".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage: stage.clone(), output: "out".into() }],
            external_inputs: vec![
                crate::ExternalInputFact {
                    stage: stage.clone(),
                    input: "public".into(),
                    value: crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: BigUint::ZERO,
                        maximum_absolute_coefficient_value: None,
                        is_constant_polynomial: false,
                    },
                },
                crate::ExternalInputFact {
                    stage: stage.clone(),
                    input: "trapdoor".into(),
                    value: crate::ExternalInputValue::Trapdoor {
                        public_matrix_input: "public".into(),
                    },
                },
                crate::ExternalInputFact {
                    stage,
                    input: "target".into(),
                    value: crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: 3u8.into(),
                        maximum_absolute_coefficient_value: Some(7u8.into()),
                        is_constant_polynomial: false,
                    },
                },
            ],
            limits: crate::SimulationLimits::default(),
        };
        let error = run(&request).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("preimage can be projected only when its relation target is exact"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn trapdoor_family_pair_survives_rank_n_reindex_and_static_get() {
        let public_type = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(97),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(3),
        };
        let preimage_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(3),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..public_type.clone()
        };
        let target_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..public_type.clone()
        };
        let trapdoor_type = WireType::Trapdoor {
            matrix: public_type.clone(),
            sigma: mxx_ir_core::RealExpr::from_integer(1),
            gadget_base: mxx_ir_core::IntExpr::constant(2),
            digit_count: mxx_ir_core::IntExpr::constant(1),
            preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(8),
        };
        let sampled = NodeHandle::new(
            NodeKind::TrapdoorSample {
                matrix_type: public_type.clone(),
                sigma: mxx_ir_core::RealExpr::from_integer(1),
                gadget_base: mxx_ir_core::IntExpr::constant(2),
                digit_count: mxx_ir_core::IntExpr::constant(1),
                preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(8),
            },
            vec![],
            vec![WireType::Matrix(public_type.clone()), trapdoor_type.clone()],
        );
        let public = sampled.output(0).unwrap();
        let trapdoor = sampled.output(1).unwrap();
        let shape = vec![mxx_ir_core::IntExpr::constant(2), mxx_ir_core::IntExpr::constant(2)];
        let public_family = NodeHandle::new(
            NodeKind::FamilyPack { shape: shape.clone() },
            vec![public.clone(), public.clone(), public.clone(), public.clone()],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(public_type.clone())),
                shape: shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let trapdoor_family = NodeHandle::new(
            NodeKind::FamilyPack { shape: shape.clone() },
            vec![trapdoor.clone(), trapdoor.clone(), trapdoor.clone(), trapdoor],
            vec![WireType::Family { element: Box::new(trapdoor_type), shape: shape.clone() }],
        )
        .output(0)
        .unwrap();
        let map = mxx_ir_core::IndexMap::new(vec![
            mxx_ir_core::IndexExpr::Axis(0),
            mxx_ir_core::IndexExpr::Axis(1),
        ]);
        let public_reindexed = NodeHandle::new(
            NodeKind::FamilyReindex { output_shape: shape.clone(), map: map.clone() },
            vec![public_family],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(public_type.clone())),
                shape: shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let trapdoor_reindexed = NodeHandle::new(
            NodeKind::FamilyReindex { output_shape: shape.clone(), map },
            vec![trapdoor_family],
            vec![WireType::Family {
                element: Box::new(WireType::Trapdoor {
                    matrix: public_type.clone(),
                    sigma: mxx_ir_core::RealExpr::from_integer(1),
                    gadget_base: mxx_ir_core::IntExpr::constant(2),
                    digit_count: mxx_ir_core::IntExpr::constant(1),
                    preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(8),
                }),
                shape: shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let public_element = NodeHandle::new(
            NodeKind::FamilyGetStatic {
                indices: vec![
                    mxx_ir_core::IndexExpr::constant(1),
                    mxx_ir_core::IndexExpr::constant(0),
                ],
            },
            vec![public_reindexed],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let trapdoor_element = NodeHandle::new(
            NodeKind::FamilyGetStatic {
                indices: vec![
                    mxx_ir_core::IndexExpr::constant(1),
                    mxx_ir_core::IndexExpr::constant(0),
                ],
            },
            vec![trapdoor_reindexed],
            vec![WireType::Trapdoor {
                matrix: public_type.clone(),
                sigma: mxx_ir_core::RealExpr::from_integer(1),
                gadget_base: mxx_ir_core::IntExpr::constant(2),
                digit_count: mxx_ir_core::IntExpr::constant(1),
                preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(8),
            }],
        )
        .output(0)
        .unwrap();
        // Build the same scalar source through a pointwise family pack.  The
        // strict preimage consumer below uses the reindexed public value as
        // its left operand, while the sampler sees this independently packed
        // public/trapdoor pair.  Source identity must follow the canonical
        // coordinate function, not the construction route.
        let pointwise_public = NodeHandle::new(
            NodeKind::FamilyPack { shape: shape.clone() },
            vec![
                public_element.clone(),
                public_element.clone(),
                public_element.clone(),
                public_element.clone(),
            ],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(public_type.clone())),
                shape: shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let pointwise_trapdoor = NodeHandle::new(
            NodeKind::FamilyPack { shape: shape.clone() },
            vec![
                trapdoor_element.clone(),
                trapdoor_element.clone(),
                trapdoor_element.clone(),
                trapdoor_element.clone(),
            ],
            vec![WireType::Family {
                element: Box::new(WireType::Trapdoor {
                    matrix: public_type.clone(),
                    sigma: mxx_ir_core::RealExpr::from_integer(1),
                    gadget_base: mxx_ir_core::IntExpr::constant(2),
                    digit_count: mxx_ir_core::IntExpr::constant(1),
                    preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(8),
                }),
                shape: shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let pointwise_public_element = NodeHandle::new(
            NodeKind::FamilyGetStatic {
                indices: vec![
                    mxx_ir_core::IndexExpr::constant(1),
                    mxx_ir_core::IndexExpr::constant(0),
                ],
            },
            vec![pointwise_public],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let pointwise_trapdoor_element = NodeHandle::new(
            NodeKind::FamilyGetStatic {
                indices: vec![
                    mxx_ir_core::IndexExpr::constant(1),
                    mxx_ir_core::IndexExpr::constant(0),
                ],
            },
            vec![pointwise_trapdoor],
            vec![WireType::Trapdoor {
                matrix: public_type.clone(),
                sigma: mxx_ir_core::RealExpr::from_integer(1),
                gadget_base: mxx_ir_core::IntExpr::constant(2),
                digit_count: mxx_ir_core::IntExpr::constant(1),
                preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(8),
            }],
        )
        .output(0)
        .unwrap();
        let target = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: target_type.clone(),
                value: ConstantMatrix::Gadget {
                    base: mxx_ir_core::IntExpr::constant(2),
                    small: true,
                },
            },
            vec![],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let preimage = NodeHandle::new(
            NodeKind::PreimageSample {
                matrix_type: preimage_type,
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(8),
            },
            vec![pointwise_public_element, pointwise_trapdoor_element, target],
            vec![WireType::Preimage(MatrixType {
                rows: mxx_ir_core::IntExpr::constant(3),
                columns: mxx_ir_core::IntExpr::constant(1),
                ..public_type.clone()
            })],
        )
        .output(0)
        .unwrap();
        let applied = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![public_element, preimage],
            vec![WireType::Matrix(target_type)],
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "trapdoor-family-pair",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: applied, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let stage = crate::StageId("trapdoor-family-pair".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage, output: "out".into() }],
            external_inputs: vec![],
            limits: crate::SimulationLimits::default(),
        };
        assert!(run(&request).is_ok());
    }

    #[test]
    fn parallel_grid_zip_public_and_trapdoor_families_preserve_pairing() {
        let public_type = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(97),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(3),
        };
        let target_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..public_type.clone()
        };
        let preimage_type = MatrixType {
            rows: mxx_ir_core::IntExpr::constant(3),
            columns: mxx_ir_core::IntExpr::constant(1),
            ..public_type.clone()
        };
        let trapdoor_type = WireType::Trapdoor {
            matrix: public_type.clone(),
            sigma: mxx_ir_core::RealExpr::from_integer(1),
            gadget_base: mxx_ir_core::IntExpr::constant(2),
            digit_count: mxx_ir_core::IntExpr::constant(1),
            preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(8),
        };
        let sampled = NodeHandle::new(
            NodeKind::TrapdoorSample {
                matrix_type: public_type.clone(),
                sigma: mxx_ir_core::RealExpr::from_integer(1),
                gadget_base: mxx_ir_core::IntExpr::constant(2),
                digit_count: mxx_ir_core::IntExpr::constant(1),
                preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(8),
            },
            vec![],
            vec![WireType::Matrix(public_type.clone()), trapdoor_type.clone()],
        );
        let public = sampled.output(0).unwrap();
        let trapdoor = sampled.output(1).unwrap();
        let public_family = NodeHandle::new(
            NodeKind::FamilyPack { shape: vec![mxx_ir_core::IntExpr::constant(2)] },
            vec![public.clone(), public],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(public_type.clone())),
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let trapdoor_family = NodeHandle::new(
            NodeKind::FamilyPack { shape: vec![mxx_ir_core::IntExpr::constant(2)] },
            vec![trapdoor.clone(), trapdoor],
            vec![WireType::Family {
                element: Box::new(trapdoor_type.clone()),
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let target = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: target_type.clone(),
                value: ConstantMatrix::Zero,
            },
            vec![],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let body = with_new_construction_scope(|scope| {
            let body_public = NodeHandle::new(
                NodeKind::Input {
                    name: "body-public".into(),
                    wire_type: WireType::Matrix(public_type.clone()),
                    artifact: None,
                },
                vec![],
                vec![WireType::Matrix(public_type.clone())],
            )
            .output(0)
            .unwrap();
            let body_trapdoor = NodeHandle::new(
                NodeKind::Input {
                    name: "body-trapdoor".into(),
                    wire_type: trapdoor_type.clone(),
                    artifact: None,
                },
                vec![],
                vec![trapdoor_type.clone()],
            )
            .output(0)
            .unwrap();
            let body_target = NodeHandle::new(
                NodeKind::Input {
                    name: "body-target".into(),
                    wire_type: WireType::Matrix(target_type.clone()),
                    artifact: None,
                },
                vec![],
                vec![WireType::Matrix(target_type.clone())],
            )
            .output(0)
            .unwrap();
            let preimage = NodeHandle::new(
                NodeKind::PreimageSample {
                    matrix_type: preimage_type.clone(),
                    max_coefficient_bound: mxx_ir_core::IntExpr::constant(8),
                },
                vec![body_public.clone(), body_trapdoor.clone(), body_target.clone()],
                vec![WireType::Preimage(preimage_type.clone())],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new(
                "parallel-zip-preimage-body",
                scope,
                vec![body_public, body_trapdoor, body_target],
                vec![preimage],
            )
            .unwrap()
        });
        let output = NodeHandle::parallel_grid(
            body,
            vec![public_family, trapdoor_family, target],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(preimage_type)),
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
            }],
            mxx_ir_core::node::ParallelGrid {
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
                index_slots: vec![0],
                bindings: vec![],
                input_modes: vec![
                    mxx_ir_core::node::GridInputMode::Reindex {
                        map: mxx_ir_core::IndexMap::new(vec![mxx_ir_core::IndexExpr::Axis(0)]),
                    },
                    mxx_ir_core::node::GridInputMode::Reindex {
                        map: mxx_ir_core::IndexMap::new(vec![mxx_ir_core::IndexExpr::Axis(0)]),
                    },
                    mxx_ir_core::node::GridInputMode::Broadcast,
                ],
            },
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "parallel-zip-preimage",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: output, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let stage = crate::StageId("parallel-zip-preimage".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage, output: "out".into() }],
            external_inputs: vec![],
            limits: crate::SimulationLimits::default(),
        };
        assert!(run(&request).is_ok());
    }

    #[test]
    fn stage_evaluates_requested_outputs_only() {
        let matrix = mxx_ir_core::types::MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let good = NodeHandle::new(
            NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value: ConstantMatrix::Zero },
            vec![],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .unwrap();
        let unused_real = NodeHandle::new(
            NodeKind::ConstantReal(mxx_ir_core::RealExpr::from_integer(1)),
            vec![],
            vec![WireType::ConstantReal],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "liveness",
            vec![],
            BTreeMap::from([
                ("good".into(), GraphOutput { value: good, confidentiality: None }),
                ("unused".into(), GraphOutput { value: unused_real, confidentiality: None }),
            ]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let environment = ParamEnv::default();
        let stage = crate::SimulationStage {
            id: crate::StageId("liveness".into()),
            production_id: ProductionId {
                spec_hash: spec_hash(&graph, &environment).unwrap(),
                execution_nonce: [0; 32],
            },
            graph,
        };
        let request = SimulationRequest {
            program: crate::SimulationProgram { stages: vec![stage] },
            environment,
            roots: vec![crate::SimulationRoot {
                stage: crate::StageId("liveness".into()),
                output: "good".into(),
            }],
            external_inputs: vec![],
            limits: crate::SimulationLimits::default(),
        };
        assert!(run(&request).is_ok());
    }

    #[test]
    fn artifact_producer_may_appear_after_consumer() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let produced = NodeHandle::new(
            NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value: ConstantMatrix::Zero },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let (producer_graph, _) = Graph::freeze(
            "late-producer",
            vec![],
            BTreeMap::from([(
                "artifact".into(),
                GraphOutput {
                    value: produced,
                    confidentiality: Some(ArtifactConfidentiality::Public),
                },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let producer_id = ProductionId {
            spec_hash: spec_hash(&producer_graph, &environment).unwrap(),
            execution_nonce: [3; 32],
        };
        let consumed = NodeHandle::new(
            NodeKind::Input {
                name: "consumed".into(),
                wire_type: WireType::Matrix(matrix.clone()),
                artifact: Some(ArtifactInput {
                    production_id: producer_id.clone(),
                    artifact_name: "artifact".into(),
                    confidentiality: ArtifactConfidentiality::Public,
                }),
            },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let dead_artifact = NodeHandle::new(
            NodeKind::Input {
                name: "dead-artifact".into(),
                wire_type: WireType::Matrix(matrix.clone()),
                artifact: Some(ArtifactInput {
                    production_id: producer_id.clone(),
                    artifact_name: "artifact".into(),
                    confidentiality: ArtifactConfidentiality::Public,
                }),
            },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let (consumer_graph, _) = Graph::freeze(
            "early-consumer",
            vec![],
            BTreeMap::from([(
                "out".into(),
                GraphOutput { value: consumed, confidentiality: None },
            )]),
            vec![dead_artifact],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let consumer = crate::StageId("consumer".into());
        let producer = crate::StageId("producer".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![
                    crate::SimulationStage {
                        id: consumer.clone(),
                        production_id: ProductionId {
                            spec_hash: spec_hash(&consumer_graph, &environment).unwrap(),
                            execution_nonce: [4; 32],
                        },
                        graph: consumer_graph,
                    },
                    crate::SimulationStage {
                        id: producer,
                        production_id: producer_id,
                        graph: producer_graph,
                    },
                ],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage: consumer, output: "out".into() }],
            external_inputs: vec![],
            limits: crate::SimulationLimits::default(),
        };
        assert!(run(&request).is_ok());
    }

    #[test]
    fn parallel_grid_freezes_one_binder_indexed_family_occurrence() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let (body, _) = with_new_construction_scope(|scope| {
            let source = NodeHandle::new(
                NodeKind::UniformResidueSample { matrix_type: matrix.clone() },
                vec![],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap();
            (
                SubgraphHandle::new("grid-identity-body", scope, vec![], vec![source.clone()])
                    .unwrap(),
                source,
            )
        });
        let shape = vec![mxx_ir_core::IntExpr::constant(2), mxx_ir_core::IntExpr::constant(3)];
        let output_type = WireType::Family {
            element: Box::new(WireType::Matrix(matrix.clone())),
            shape: shape.clone(),
        };
        let grid = NodeHandle::parallel_grid(
            body,
            vec![],
            vec![output_type],
            mxx_ir_core::node::ParallelGrid {
                shape: shape.clone(),
                index_slots: vec![0, 1],
                bindings: vec![],
                input_modes: vec![],
            },
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "grid-identity",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: grid, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let environment = ParamEnv::default();
        let stage = crate::StageId("grid".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage: stage.clone(), output: "out".into() }],
            external_inputs: vec![],
            limits: crate::SimulationLimits::default(),
        };
        let plan = crate::plan::Plan::build(&request).unwrap();
        assert!(plan.wires.iter().any(|wire| {
            wire.occurrence == vec!["node:0/grid"] &&
                matches!(wire.scope, mxx_ir_core::FrozenGraphScopeId::ParallelBody { .. })
        }));
        assert!(
            !plan.wires.iter().any(|wire| wire.occurrence.iter().any(|part| part.contains("lane")))
        );
        let report = run(&request).unwrap();
        assert_eq!(report.roots[0].maximum_absolute_coefficient_error, BigUint::ZERO);
    }

    #[test]
    fn same_typed_family_trapdoor_mismatch_is_rejected() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let matrix_wire = |name: &str| {
            NodeHandle::new(
                NodeKind::Input {
                    name: name.into(),
                    wire_type: WireType::Matrix(matrix.clone()),
                    artifact: None,
                },
                vec![],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap()
        };
        let public_a = matrix_wire("public-a");
        let public_b = matrix_wire("public-b");
        let trapdoor_wire = || WireType::Trapdoor {
            matrix: matrix.clone(),
            sigma: mxx_ir_core::RealExpr::from_integer(1),
            gadget_base: mxx_ir_core::IntExpr::constant(2),
            digit_count: mxx_ir_core::IntExpr::constant(1),
            preimage_max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
        };
        let trapdoor_a = NodeHandle::new(
            NodeKind::Input {
                name: "trapdoor-a".into(),
                wire_type: trapdoor_wire(),
                artifact: None,
            },
            vec![],
            vec![trapdoor_wire()],
        )
        .output(0)
        .unwrap();
        let trapdoor_b = NodeHandle::new(
            NodeKind::Input {
                name: "trapdoor-b".into(),
                wire_type: trapdoor_wire(),
                artifact: None,
            },
            vec![],
            vec![trapdoor_wire()],
        )
        .output(0)
        .unwrap();
        let target = NodeHandle::new(
            NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value: ConstantMatrix::Zero },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let target_family = NodeHandle::new(
            NodeKind::FamilyPack { shape: vec![mxx_ir_core::IntExpr::constant(2)] },
            vec![target.clone(), target],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(matrix.clone())),
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let sampled = NodeHandle::new(
            NodeKind::FamilyPreimageSample {
                matrix_type: matrix.clone(),
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
            },
            vec![public_a, trapdoor_b, target_family],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(matrix.clone())),
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "same-typed-trapdoors",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: sampled, confidentiality: None })]),
            vec![public_b.clone(), trapdoor_a.clone()],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let environment = ParamEnv::default();
        let stage = crate::StageId("pairing".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![crate::SimulationStage {
                    id: stage.clone(),
                    production_id: ProductionId {
                        spec_hash: spec_hash(&graph, &environment).unwrap(),
                        execution_nonce: [0; 32],
                    },
                    graph,
                }],
            },
            environment,
            roots: vec![crate::SimulationRoot { stage: stage.clone(), output: "out".into() }],
            external_inputs: vec![
                crate::ExternalInputFact {
                    stage: stage.clone(),
                    input: "public-a".into(),
                    value: crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: BigUint::ZERO,
                        maximum_absolute_coefficient_value: None,
                        is_constant_polynomial: false,
                    },
                },
                crate::ExternalInputFact {
                    stage: stage.clone(),
                    input: "public-b".into(),
                    value: crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: BigUint::ZERO,
                        maximum_absolute_coefficient_value: None,
                        is_constant_polynomial: false,
                    },
                },
                crate::ExternalInputFact {
                    stage: stage.clone(),
                    input: "trapdoor-a".into(),
                    value: crate::ExternalInputValue::Trapdoor {
                        public_matrix_input: "public-a".into(),
                    },
                },
                crate::ExternalInputFact {
                    stage: stage.clone(),
                    input: "trapdoor-b".into(),
                    value: crate::ExternalInputValue::Trapdoor {
                        public_matrix_input: "public-b".into(),
                    },
                },
            ],
            limits: crate::SimulationLimits::default(),
        };
        let result = run(&request);
        assert!(matches!(
            result,
            Err(SimulationError::InvalidGraph { message, .. })
                if message.contains("not paired")
        ));
        let _ = (public_b, trapdoor_a);
    }
}
