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
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

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

/// Evaluator-private numeric provenance for one structural loop binder.
///
/// The public abstract domain remains an interval.  This sidecar retains only
/// the small amount of correlation needed to refine an affine scalar on a
/// comparison branch; expressions involving multiple binders or non-affine
/// arithmetic simply have no provenance and continue with interval semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
enum ScalarFacts {
    Affine(AffineScalar),
    Truth(TruthFacts),
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum BinderDependencies {
    Known(BTreeSet<u64>),
    Unknown,
}

impl BinderDependencies {
    fn union<'b>(dependencies: impl IntoIterator<Item = &'b Self>) -> Self {
        let mut combined = BTreeSet::new();
        for dependency in dependencies {
            match dependency {
                Self::Known(binders) => combined.extend(binders.iter().copied()),
                Self::Unknown => return Self::Unknown,
            }
        }
        Self::Known(combined)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AffineScalar {
    binder: Option<u64>,
    coefficient: BigInt,
    offset: BigInt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct BinderRefinement {
    binder: u64,
    range: state::IntegerState,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum OutcomeRefinement {
    Impossible,
    Unconstrained,
    Restricted(BinderRefinement),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TruthFacts {
    when_zero: OutcomeRefinement,
    when_one: OutcomeRefinement,
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
    // Validate stages in program order while accumulating the exact manifests
    // exported by their producers.  Cross-stage artifact inputs cannot be
    // validated in isolation: the consumer must see the producer's concrete
    // output type, family shape, and confidentiality.
    // Validation elaborates every stage's complete graph, including dead
    // artifact inputs, so close the manifest dependency graph over all stages.
    {
        let needed =
            request.program.stages.iter().map(|stage| stage.id.clone()).collect::<HashSet<_>>();
        let validation_order = artifact_validation_order(request, &needed)?;
        let mut manifests = BTreeMap::new();
        for index in validation_order {
            let stage = &request.program.stages[index];
            let validated = mxx_ir_core::validate_with_manifests(
                &stage.graph,
                &request.environment,
                &manifests,
            )
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
    }
    // The occurrence-aware plan is built only after every requested graph and
    // its artifact manifest have passed deep validation.  Keep the temporary
    // validation graph and manifest data out of the peak evaluation lifetime.
    let plan = crate::plan::Plan::build(request)?;
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
        abstract_integers: request
            .environment
            .integers
            .iter()
            .map(|(name, value)| (name.clone(), state::IntegerState::singleton(value.clone())))
            .collect(),
        abstract_integer_facts: request
            .environment
            .integers
            .iter()
            .map(|(name, value)| (name.clone(), ScalarFacts::Affine(AffineScalar::constant(value))))
            .collect(),
        abstract_integer_dependencies: request
            .environment
            .integers
            .keys()
            .map(|name| (name.clone(), BinderDependencies::Known(BTreeSet::new())))
            .collect(),
        abstract_loop_indices: HashMap::new(),
        abstract_loop_atoms: HashMap::new(),
        binder_ranges: HashMap::new(),
        next_binder_atom: 0,
        scalar_facts: HashMap::new(),
        scalar_dependencies: HashMap::new(),
        family_axis_dependencies: HashMap::new(),
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
    /// Integer parameters and loop binders visible to the current symbolic
    /// scope.  Parallel-grid binders are intervals, so the body is evaluated
    /// once while still covering every concrete lane.
    abstract_integers: BTreeMap<String, state::IntegerState>,
    abstract_integer_facts: BTreeMap<String, ScalarFacts>,
    abstract_integer_dependencies: BTreeMap<String, BinderDependencies>,
    abstract_loop_indices: HashMap<u32, state::IntegerState>,
    /// Loop slots are lexical names and may be reused by nested grids.  Each
    /// active slot therefore maps to a fresh atom before it enters affine
    /// provenance, preventing unrelated binders from being correlated.
    abstract_loop_atoms: HashMap<u32, u64>,
    binder_ranges: HashMap<u64, state::IntegerState>,
    next_binder_atom: u64,
    scalar_facts: HashMap<crate::FamilyViewId, ScalarFacts>,
    scalar_dependencies: HashMap<crate::FamilyViewId, BinderDependencies>,
    /// Output-coordinate axes on which a family-valued selector can vary.
    /// Absence means unknown and is interpreted as dependence on every axis.
    family_axis_dependencies: HashMap<crate::FamilyViewId, BTreeSet<usize>>,
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

/// Projection from a relation-valued family to the public source coordinates
/// consumed by that relation.  A shared source omits a trailing branch suffix;
/// a pointwise source has exactly the same rank as the preimage family.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RelationSourceProjection {
    relation_rank: usize,
    source_rank: usize,
}

impl RelationSourceProjection {
    fn input_prefix<T: Clone>(&self, coordinates: &[T]) -> Option<Vec<T>> {
        (coordinates.len() == self.relation_rank).then(|| coordinates[..self.source_rank].to_vec())
    }

    fn output_prefix(&self, shape: &[usize]) -> Option<Vec<usize>> {
        let shared_suffix_rank = self.relation_rank.checked_sub(self.source_rank)?;
        let source_output_rank = shape.len().checked_sub(shared_suffix_rank)?;
        Some(shape[..source_output_rank].to_vec())
    }

    fn is_shared(&self) -> bool {
        self.source_rank < self.relation_rank
    }
}

impl<'a> Evaluator<'a> {
    fn scalar_fact(&self, value: &Info) -> Option<&ScalarFacts> {
        self.scalar_facts.get(&value.view)
    }

    fn derived_scalar_dependencies(
        &self,
        kind: &NodeKind,
        inputs: &[Info],
    ) -> Option<BinderDependencies> {
        match kind {
            NodeKind::ConstantInt(_) | NodeKind::ConstantBool(_) => {
                Some(BinderDependencies::Known(BTreeSet::new()))
            }
            NodeKind::EvaluateInt(expression) => Some(int_expr_dependencies(
                expression,
                &self.abstract_integer_dependencies,
                &self.abstract_loop_atoms,
            )),
            NodeKind::IntBinary(_) |
            NodeKind::IntCompare(_) |
            NodeKind::BoolToInt |
            NodeKind::Select { .. } => {
                let dependencies = inputs
                    .iter()
                    .map(|input| self.scalar_dependencies.get(&input.view))
                    .collect::<Option<Vec<_>>>();
                Some(
                    dependencies
                        .map(BinderDependencies::union)
                        .unwrap_or(BinderDependencies::Unknown),
                )
            }
            _ => None,
        }
    }

    fn family_dependencies(&self, value: &Info) -> BTreeSet<usize> {
        match &value.value {
            AbstractValue::Family(family) => self
                .family_axis_dependencies
                .get(&value.view)
                .cloned()
                .unwrap_or_else(|| (0..family.shape.len()).collect()),
            _ => BTreeSet::new(),
        }
    }

    fn record_family_axis_dependencies(
        &mut self,
        kind: &NodeKind,
        inputs: &[Info],
        outputs: &[Info],
    ) {
        for output in outputs {
            let AbstractValue::Family(family) = &output.value else { continue };
            let all = || (0..family.shape.len()).collect::<BTreeSet<_>>();
            let dependencies = match kind {
                // Packing distinct scalar wires provides no syntactic proof
                // that coordinates are equal, even when their intervals or
                // binder dependencies happen to match.
                NodeKind::FamilyPack { .. } => all(),
                NodeKind::FamilyReindex { map, .. } => {
                    let input_dependencies = inputs
                        .first()
                        .map(|input| self.family_dependencies(input))
                        .unwrap_or_default();
                    let mut dependencies = BTreeSet::new();
                    for input_axis in input_dependencies {
                        let Some(expression) = map.input_indices.get(input_axis) else {
                            dependencies = all();
                            break;
                        };
                        dependencies.extend(
                            (0..family.shape.len())
                                .filter(|axis| index_expr_depends_axis(expression, *axis)),
                        );
                    }
                    dependencies
                }
                NodeKind::FamilySelectAxis { axis } => {
                    let input_dependencies = inputs
                        .first()
                        .map(|input| self.family_dependencies(input))
                        .unwrap_or_default();
                    let selector_dependencies = inputs
                        .get(1)
                        .map(|selector| self.family_dependencies(selector))
                        .unwrap_or_default();
                    let mut dependencies = BTreeSet::new();
                    for input_axis in input_dependencies {
                        if input_axis < *axis {
                            dependencies.insert(input_axis);
                        } else if input_axis > *axis {
                            dependencies.insert(input_axis - 1);
                        } else {
                            dependencies.extend(selector_dependencies.iter().copied());
                        }
                    }
                    dependencies
                }
                NodeKind::FamilyGather { .. } => {
                    let source_dependencies = inputs
                        .first()
                        .map(|input| self.family_dependencies(input))
                        .unwrap_or_default();
                    let mut dependencies = BTreeSet::new();
                    for source_axis in source_dependencies {
                        let Some(selector) = inputs.get(source_axis + 1) else {
                            dependencies = all();
                            break;
                        };
                        dependencies.extend(self.family_dependencies(selector));
                    }
                    dependencies
                }
                NodeKind::Select { .. } => inputs
                    .iter()
                    .skip(1)
                    .flat_map(|input| self.family_dependencies(input))
                    .collect(),
                NodeKind::ParallelGrid(_) => continue,
                _ => all(),
            };
            self.family_axis_dependencies.insert(output.view, dependencies);
        }
    }

    fn int_binary_transfer(
        &self,
        operation: IntBinaryOp,
        left: &Info,
        right: &Info,
    ) -> Result<(state::IntegerState, Option<ScalarFacts>), SimulationError> {
        let a = int(left)?;
        let b = int(right)?;
        let left_fact = self.scalar_fact(left);
        let right_fact = self.scalar_fact(right);
        let ordinary = || match operation {
            IntBinaryOp::Add => Ok(a.add(&b)),
            IntBinaryOp::Subtract => Ok(a.subtract(&b)),
            IntBinaryOp::Multiply => Ok(a.multiply(&b)),
            IntBinaryOp::Divide => a.divide(&b).map_err(|error| SimulationError::InvalidGraph {
                message: error.to_string(),
                site: None,
            }),
            IntBinaryOp::Remainder => a.remainder(&b).map_err(|error| {
                SimulationError::InvalidGraph { message: error.to_string(), site: None }
            }),
        };

        let affine = (|| match operation {
            IntBinaryOp::Add => affine_fact(left_fact)?.add(&affine_fact(right_fact)?),
            IntBinaryOp::Subtract => affine_fact(left_fact)?.subtract(&affine_fact(right_fact)?),
            IntBinaryOp::Multiply => {
                let left = affine_fact(left_fact)?;
                let right = affine_fact(right_fact)?;
                if left.binder.is_none() {
                    Some(right.multiply_constant(&left.offset))
                } else if right.binder.is_none() {
                    Some(left.multiply_constant(&right.offset))
                } else {
                    None
                }
            }
            IntBinaryOp::Divide => {
                let left = affine_fact(left_fact)?;
                let right = affine_fact(right_fact)?;
                if right.binder.is_none() &&
                    !right.offset.is_zero() &&
                    &left.coefficient % &right.offset == BigInt::zero() &&
                    &left.offset % &right.offset == BigInt::zero()
                {
                    Some(AffineScalar {
                        binder: left.binder,
                        coefficient: left.coefficient / &right.offset,
                        offset: left.offset / right.offset,
                    })
                } else {
                    None
                }
            }
            IntBinaryOp::Remainder => None,
        })();
        if let Some(affine) = affine {
            return Ok((ordinary()?, Some(ScalarFacts::Affine(affine))));
        }

        if operation == IntBinaryOp::Multiply {
            if let (Some(left), Some(right)) = (truth_fact(left_fact), truth_fact(right_fact)) {
                let when_one = intersect_outcomes(&left.when_one, &right.when_one);
                let when_zero = if matches!(left.when_zero, OutcomeRefinement::Impossible) &&
                    matches!(right.when_zero, OutcomeRefinement::Impossible)
                {
                    OutcomeRefinement::Impossible
                } else {
                    OutcomeRefinement::Unconstrained
                };
                let minimum =
                    if matches!(when_zero, OutcomeRefinement::Impossible) { 1 } else { 0 };
                let maximum = if matches!(when_one, OutcomeRefinement::Impossible) { 0 } else { 1 };
                return Ok((
                    state::IntegerState::new(minimum.into(), maximum.into())?,
                    Some(ScalarFacts::Truth(TruthFacts { when_zero, when_one })),
                ));
            }
        }
        Ok((ordinary()?, None))
    }

    fn derived_scalar_fact(
        &self,
        kind: &NodeKind,
        inputs: &[Info],
        env: &ParamEnv,
    ) -> Option<ScalarFacts> {
        match kind {
            NodeKind::ConstantInt(value) => {
                Some(ScalarFacts::Affine(AffineScalar::constant(value)))
            }
            NodeKind::EvaluateInt(expression) => {
                eval_int_facts(expression, &self.abstract_integer_facts, &self.abstract_loop_atoms)
            }
            NodeKind::ConstantBool(value) => Some(ScalarFacts::Truth(TruthFacts {
                when_zero: if *value {
                    OutcomeRefinement::Impossible
                } else {
                    OutcomeRefinement::Unconstrained
                },
                when_one: if *value {
                    OutcomeRefinement::Unconstrained
                } else {
                    OutcomeRefinement::Impossible
                },
            })),
            NodeKind::IntCompare(operation) => comparison_facts(
                *operation,
                inputs.first().and_then(|value| self.scalar_fact(value)),
                inputs.get(1).and_then(|value| self.scalar_fact(value)),
                &self.binder_ranges,
            ),
            NodeKind::BoolToInt => {
                inputs.first().and_then(|value| self.scalar_fact(value)).cloned()
            }
            NodeKind::IntBinary(operation) => {
                let (left, right) = (inputs.first()?, inputs.get(1)?);
                self.int_binary_transfer(*operation, left, right).ok()?.1
            }
            NodeKind::Select { count }
                if eval_int_interval(
                    count,
                    env,
                    &self.abstract_integers,
                    &self.abstract_loop_indices,
                )
                .ok()
                .is_some_and(|range| {
                    range.minimum == 2.into() && range.maximum_inclusive == 2.into()
                }) =>
            {
                let left = inputs.get(1).and_then(|value| self.scalar_fact(value));
                let right = inputs.get(2).and_then(|value| self.scalar_fact(value));
                (left == right).then(|| left.cloned()).flatten()
            }
            _ => None,
        }
    }

    fn integer_expression(
        &self,
        expression: &mxx_ir_core::IntExpr,
        env: &ParamEnv,
    ) -> Result<state::IntegerState, SimulationError> {
        eval_int_interval(expression, env, &self.abstract_integers, &self.abstract_loop_indices)
    }

    fn singleton_integer_expression(
        &self,
        expression: &mxx_ir_core::IntExpr,
        env: &ParamEnv,
        purpose: &str,
    ) -> Result<BigInt, SimulationError> {
        let range = self.integer_expression(expression, env)?;
        if range.minimum != range.maximum_inclusive {
            return Err(SimulationError::InvalidGraph {
                message: format!("{purpose} must be uniform across a parallel grid"),
                site: None,
            });
        }
        Ok(range.minimum)
    }

    fn integer_expression_magnitude(
        &self,
        expression: &mxx_ir_core::IntExpr,
        env: &ParamEnv,
    ) -> Result<BigInt, SimulationError> {
        let range = self.integer_expression(expression, env)?;
        Ok(BigInt::from(crate::bound::max_abs_interval(&range.minimum, &range.maximum_inclusive)))
    }

    fn require_uniform_wire_type(
        &self,
        ty: &WireType,
        env: &ParamEnv,
    ) -> Result<(), SimulationError> {
        let require = |expression: &mxx_ir_core::IntExpr, purpose: &str| {
            self.singleton_integer_expression(expression, env, purpose).map(|_| ())
        };
        match ty {
            WireType::Matrix(matrix) | WireType::Preimage(matrix) => {
                require(&matrix.modulus, "matrix modulus")?;
                require(&matrix.ring_dimension, "matrix ring dimension")?;
                require(&matrix.rows, "matrix row count")?;
                require(&matrix.columns, "matrix column count")?;
            }
            WireType::Trapdoor {
                matrix,
                gadget_base,
                digit_count,
                preimage_max_coefficient_bound,
                ..
            } => {
                self.require_uniform_wire_type(&WireType::Matrix(matrix.clone()), env)?;
                require(gadget_base, "trapdoor gadget base")?;
                require(digit_count, "trapdoor digit count")?;
                require(preimage_max_coefficient_bound, "trapdoor preimage bound")?;
            }
            WireType::Family { element, shape } => {
                self.require_uniform_wire_type(element, env)?;
                for extent in shape {
                    require(extent, "family extent")?;
                }
            }
            WireType::Bytes { length } => require(length, "byte length")?,
            WireType::ConstantInt |
            WireType::ConstantReal |
            WireType::ConstantBool |
            WireType::Int |
            WireType::Real |
            WireType::Bool |
            WireType::TypedBlob { .. } => {}
        }
        Ok(())
    }

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

    /// Return whether a source is the canonical gadget source or a structural
    /// family lineage made entirely from that source.  This is deliberately
    /// identity-only: numeric values, labels, and protocol metadata are not
    /// evidence that an automorphism may transform a tracked matrix.
    fn is_gadget_source(&self, source: crate::SourceId) -> bool {
        let is_gadget = |candidate: &crate::SourceId| {
            self.gadget_sources.values().any(|gadget| gadget == candidate)
        };
        is_gadget(&source) ||
            self.source_lineages.get(&source).is_some_and(|lineage| {
                !lineage.leaves.is_empty() && lineage.leaves.iter().all(is_gadget)
            })
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

    fn relation_source_projection(
        &self,
        relation: &RightPreimage,
        relation_shape: &[usize],
        site: Option<DiagnosticSite>,
    ) -> Result<RelationSourceProjection, SimulationError> {
        let source_shape = self
            .source_lineages
            .get(&relation.source)
            .map(|lineage| lineage.shape.clone())
            .ok_or_else(|| SimulationError::Relation {
                message: "preimage relation source has no canonical lineage".into(),
                site: site.clone(),
            })?;
        let source_rank = source_shape.len();
        if source_rank > relation_shape.len() {
            return Err(SimulationError::Relation {
                message: "preimage relation source rank exceeds its preimage family rank".into(),
                site: site.clone(),
            });
        }
        if source_rank != relation_shape.len() &&
            source_rank.checked_add(1) != Some(relation_shape.len())
        {
            return Err(SimulationError::Relation {
                message:
                    "preimage relation must be pointwise or omit exactly one final branch axis"
                        .into(),
                site: site.clone(),
            });
        }
        if source_shape != relation_shape[..source_rank] {
            return Err(SimulationError::Relation {
                message: "preimage relation source shape does not match its group prefix".into(),
                site: site.clone(),
            });
        }
        if relation
            .view
            .and_then(|view| self.view_shape(view))
            .is_none_or(|shape| shape != relation_shape)
        {
            return Err(SimulationError::Relation {
                message: "preimage relation view rank does not match its value family".into(),
                site: site.clone(),
            });
        }
        if self.view_shape(relation.target).is_none_or(|shape| shape != relation_shape) {
            return Err(SimulationError::Relation {
                message: "preimage relation target rank does not match its value family".into(),
                site,
            });
        }
        Ok(RelationSourceProjection { relation_rank: relation_shape.len(), source_rank })
    }

    /// Pack scalar equations into the shared-source family equation
    /// `B[g] * K[g,d] = T[g,d]`. Row-major inputs with one fixed `g` must all
    /// use the same source; only targets and preimages vary along final branch
    /// coordinate `d`. Numeric target bounds remain uniform by taking maxima.
    fn pack_shared_preimage_relation(
        &mut self,
        inputs: &[Info],
        shape: &[usize],
        site: Option<DiagnosticSite>,
    ) -> Result<RightPreimage, SimulationError> {
        let expected_count =
            shape.iter().try_fold(1usize, |count, extent| count.checked_mul(*extent));
        if expected_count != Some(inputs.len()) {
            return Err(SimulationError::Relation {
                message: "packed preimage relation cardinality does not match its family shape"
                    .into(),
                site,
            });
        }
        let branch_count = shape.last().copied().ok_or_else(|| SimulationError::Relation {
            message: "packed preimage relation requires one final branch axis".into(),
            site: site.clone(),
        })?;
        if branch_count == 0 {
            return Err(SimulationError::Relation {
                message: "packed preimage relation branch axis is empty".into(),
                site,
            });
        }

        let mut sources = Vec::with_capacity(inputs.len());
        let mut target_views = Vec::with_capacity(inputs.len());
        let mut target_states = Vec::with_capacity(inputs.len());
        for input in inputs {
            let relation = input.relation.as_ref().ok_or_else(|| SimulationError::Relation {
                message: "every packed preimage must carry a relation".into(),
                site: site.clone(),
            })?;
            let relation_view = relation.view.ok_or_else(|| SimulationError::Relation {
                message: "packed preimage relation has no canonical view".into(),
                site: site.clone(),
            })?;
            if self.preimages.get(&relation_view) != Some(relation) {
                return Err(SimulationError::Relation {
                    message: "packed preimage relation is not registered on its canonical view"
                        .into(),
                    site: site.clone(),
                });
            }
            self.relation_source_projection(relation, &[], site.clone())?;
            let target = self.states.get(&relation.target).cloned().ok_or_else(|| {
                SimulationError::Relation {
                    message: "packed preimage target state is unavailable".into(),
                    site: site.clone(),
                }
            })?;
            sources.push(relation.source);
            target_views.push(relation.target);
            target_states.push(target);
        }

        let mut group_sources = Vec::with_capacity(sources.len() / branch_count);
        for branches in sources.chunks_exact(branch_count) {
            let source = branches[0];
            if branches.iter().any(|candidate| *candidate != source) {
                return Err(SimulationError::Relation {
                    message: "packed preimage branches within one group must share one source"
                        .into(),
                    site: site.clone(),
                });
            }
            group_sources.push(source);
        }

        let mut target =
            target_states.first().cloned().ok_or_else(|| SimulationError::Relation {
                message: "packed preimage family is empty".into(),
                site: site.clone(),
            })?;
        for state in &target_states[1..] {
            target.error_bound = target.error_bound.max(state.error_bound.clone());
            target.coefficient_magnitude_bound =
                target.coefficient_magnitude_bound.max(state.coefficient_magnitude_bound.clone());
            target.is_constant_polynomial &= state.is_constant_polynomial;
        }
        let target_carriers = target_states
            .iter()
            .map(|state| state.right_carrier.as_ref())
            .collect::<Option<Vec<_>>>();
        if let Some(carriers) = target_carriers {
            let target_source = self.group_source_for(
                carriers.iter().map(|carrier| carrier.source).collect(),
                shape.to_vec(),
            );
            let target_gain =
                carriers.iter().map(|carrier| carrier.left_gain.clone()).max().unwrap_or_default();
            target.right_carrier =
                Some(crate::RightCarrier { source: target_source, left_gain: target_gain });
        } else if target_states.iter().any(|state| state.right_carrier.is_some()) {
            return Err(SimulationError::Relation {
                message: "packed preimage targets have incompatible source carriers".into(),
                site,
            });
        } else {
            target.right_carrier = None;
        }

        let target_view = self.interners.intern_composed_view(target_views, shape.to_vec(), &[]);
        self.states.insert(target_view, target);
        let group_shape = shape[..shape.len() - 1].to_vec();
        let source = if group_shape.is_empty() {
            group_sources[0]
        } else {
            self.group_source_for(group_sources, group_shape)
        };
        Ok(RightPreimage { source, target: target_view, view: None, selector: None })
    }

    fn gathered_relation_source_projection(
        &self,
        relation: &RightPreimage,
        relation_shape: &[usize],
        output_shape: &[usize],
        selector_values: &[Info],
        selectors: &[crate::SelectorId],
        site: Option<DiagnosticSite>,
    ) -> Result<(Vec<crate::SelectorId>, Vec<usize>), SimulationError> {
        let projection = self.relation_source_projection(relation, relation_shape, site.clone())?;
        if selector_values.len() != projection.relation_rank ||
            selectors.len() != projection.relation_rank
        {
            return Err(SimulationError::Relation {
                message: "preimage gather selector rank does not match its input family".into(),
                site,
            });
        }
        // B[g] may use a varying group selector, but that selector must not
        // depend on the final output branch coordinate d.  Missing structural
        // provenance is interpreted as dependence on every output axis.
        if projection.is_shared() {
            let branch_axis =
                output_shape.len().checked_sub(1).ok_or_else(|| SimulationError::Relation {
                    message: "shared preimage gather output has no final branch axis".into(),
                    site: site.clone(),
                })?;
            if selector_values[..projection.source_rank]
                .iter()
                .any(|selector| self.family_dependencies(selector).contains(&branch_axis))
            {
                return Err(SimulationError::BranchDependentSource { site });
            }
        }
        let source_selectors = projection.input_prefix(selectors).expect("ranks checked above");
        let source_shape =
            projection.output_prefix(output_shape).ok_or_else(|| SimulationError::Relation {
                message: "preimage gather output cannot preserve its shared-source suffix".into(),
                site,
            })?;
        Ok((source_selectors, source_shape))
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
        let mut has_opaque_source = false;
        let mut leaves = Vec::new();
        for source in &sources {
            match self.source_lineages.get(source) {
                Some(lineage) if lineage_is_complete(lineage) => {
                    leaves.extend(lineage.leaves.iter().copied());
                }
                Some(lineage) if sources.len() == 1 && lineage.shape == shape => return *source,
                _ => {
                    // Preserve an opaque source as one sentinel instead of
                    // expanding it into a false uniform coordinate function.
                    has_opaque_source = true;
                    leaves.push(*source);
                }
            }
        }
        let count = shape.iter().copied().product::<usize>().max(1);
        // A single scalar source entering a family-valued structural node is
        // an index-independent broadcast.  Record the full coordinate
        // function so later identity reindexes can resolve every lane.  This
        // does not merge distinct sources: a multi-leaf lineage is preserved
        // verbatim and must already match the requested family cardinality.
        if !has_opaque_source && leaves.len() == 1 && count > 1 {
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
        let key = (source, map.normalize(), shape.clone());
        if let Some(mapped) = self.mapped_sources.get(&key) {
            return *mapped;
        }
        let mapped = crate::SourceId(self.next_source);
        self.next_source = self.next_source.saturating_add(1);
        self.mapped_sources.insert(key, mapped);
        // A structural map with unresolved parameters still has a known
        // output domain.  Keep one self sentinel rather than enumerating that
        // domain or pretending that every output coordinate maps to one leaf.
        // Exact lineage consumers reject this incomplete representation.
        self.register_lineage(mapped, SourceLineage { shape, leaves: vec![mapped] });
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
            if lineage_is_complete(&parent) &&
                indices.len() == parent.shape.len() &&
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
        let key = (source, selectors, shape.clone());
        if let Some(mapped) = self.gathered_sources.get(&key) {
            return *mapped;
        }
        let mapped = crate::SourceId(self.next_source);
        self.next_source = self.next_source.saturating_add(1);
        self.gathered_sources.insert(key, mapped);
        // An opaque selector still has a statically known output domain.  A
        // single self leaf is an incomplete sentinel for a multi-coordinate
        // domain: it preserves rank without claiming that all coordinates
        // select the same public source or enumerating a potentially huge
        // family.  Exact and uniform lineage consumers reject incomplete
        // lineages before inspecting their leaves.
        self.register_lineage(mapped, SourceLineage { shape, leaves: vec![mapped] });
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
        // A family source that already has this coordinate domain is already
        // the normalized lane-indexed function.  Reusing it through a grid
        // must not replace its exact public/trapdoor leaves with fresh body
        // sources.  Only a shape-less primitive created by the body needs the
        // coordinate lifting below.
        if self.source_lineages.get(&source).is_some_and(|lineage| lineage.shape == shape) {
            return source;
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
        // Every concrete type used by a transfer must be uniform while a
        // symbolic grid binder is active.  Checking on scope entry covers
        // nested subgraph intermediates as well as the grid body's declared
        // outputs; otherwise a grandchild could silently use lane zero's
        // matrix dimensions in its noise geometry.
        for node in scope.nodes() {
            for output_type in node.output_types() {
                self.require_uniform_wire_type(output_type, &env)?;
            }
        }
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
                .map_err(|error| {
                    let site = Some(DiagnosticSite {
                        stage: Some(stage.clone()),
                        occurrence: occurrence.to_vec(),
                        node: Some(mxx_ir_core::NodeId(n as u64)),
                        port: Some(mxx_ir_core::Port(0)),
                        operation: Some(format!("{:?}", node.kind())),
                    });
                    match error {
                        // Relation failures are semantic violations at the
                        // operation site, not malformed graph syntax. Keep
                        // their typed category so callers can fail closed.
                        SimulationError::Relation { message, .. }
                            if matches!(node.kind(), NodeKind::RingAutomorphism { .. }) =>
                        {
                            SimulationError::Relation { message, site }
                        }
                        error => SimulationError::InvalidGraph {
                            message: format!(
                                "stage {:?}, node {n} ({:?}): {error}",
                                stage,
                                node.kind()
                            ),
                            site,
                        },
                    }
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
            if let Some(value) = out.first() {
                if let Some(facts) = self.derived_scalar_fact(node.kind(), &inputs, &env) {
                    self.scalar_facts.insert(value.view, facts);
                }
                if let Some(dependencies) = self.derived_scalar_dependencies(node.kind(), &inputs) {
                    self.scalar_dependencies.insert(value.view, dependencies);
                }
            }
            self.record_family_axis_dependencies(node.kind(), &inputs, &out);
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
            NodeKind::EvaluateInt(v) => Ok(vec![integer_range(eval_int_interval(
                v,
                env,
                &self.abstract_integers,
                &self.abstract_loop_indices,
            )?)]),
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
                        let base = self.singleton_integer_expression(base, env, "gadget base")?;
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
                        let base =
                            self.singleton_integer_expression(base, env, "power-of-base base")?;
                        let exponent = self
                            .singleton_integer_expression(exponent, env, "power-of-base exponent")?
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
                            .map(|coefficient| self.integer_expression(coefficient, env))
                            .collect::<Result<Vec<_>, _>>()?;
                        let magnitude = evaluated
                            .iter()
                            .map(|coefficient| {
                                crate::bound::max_abs_interval(
                                    &coefficient.minimum,
                                    &coefficient.maximum_inclusive,
                                )
                            })
                            .max()
                            .unwrap_or_default();
                        let constant = evaluated.iter().skip(1).all(|coefficient| {
                            coefficient.minimum.is_zero() && coefficient.maximum_inclusive.is_zero()
                        });
                        state::exact_matrix(&t, magnitude, constant)?
                    }
                };
                let source =
                    if let mxx_ir_core::node::ConstantMatrix::Gadget { base, small } = value {
                        let base = self.singleton_integer_expression(base, env, "gadget base")?;
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
                let b = self.singleton_integer_expression(base, env, "gadget base")?;
                let digits = t.columns.checked_div(t.rows).unwrap_or(1);
                // The public matrix emitted with a trapdoor is the same
                // structural gadget as ConstantMatrix::Gadget.  Register
                // that descriptor once, without inspecting values or node
                // names, so automorphisms and decompositions share G's
                // canonical source identity.
                let descriptor = crate::GadgetDescriptor {
                    modulus: t.modulus.clone(),
                    ring_dimension: t.ring_dimension,
                    rows: t.rows,
                    columns: t.columns,
                    base: b.clone(),
                    digit_count: digits,
                    small: false,
                };
                if !self.gadget_sources.contains_key(&descriptor) {
                    let source = self.source_for(stage, sid, occurrence, n, "gadget");
                    self.gadget_sources.insert(descriptor, source);
                }
                // The IR exposes the trapdoor as one value; TrapdoorPublic
                // materializes this registered G when needed.  Keeping the
                // public matrix out of the trapdoor value matches the typed
                // runtime representation and avoids an unpaired second port.
                Ok(vec![Info {
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
                }])
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
                let (z, _) = self
                    .int_binary_transfer(*op, &xs[0], &xs[1])
                    .map_err(|error| bad(&error.to_string()))?;
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
                let s = self.integer_expression_magnitude(scalar, env)?;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(matrix(&xs[0])?.scale(&s, &t.modulus)?),
                    ty: Some(t),
                    relation: xs[0].relation.clone(),
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
            NodeKind::RingAutomorphism { .. } => {
                let t = mt(&xs[0])?;
                let value = matrix(&xs[0])?;
                if xs[0].relation.is_some() {
                    return Err(SimulationError::Relation {
                        message: "ring automorphism cannot transform a relation-bearing preimage"
                            .into(),
                        site: site(),
                    });
                }
                if let Some(carrier) = &value.right_carrier {
                    let source = carrier.source;
                    if !self.is_gadget_source(source) {
                        return Err(SimulationError::Relation {
                            message: format!(
                                "ring automorphism requires an untracked matrix or canonical gadget source, got {source:?}"
                            ),
                            site: site(),
                        });
                    }
                }
                // A valid ring automorphism only permutes (and possibly
                // negates) polynomial coefficients, so all numeric bounds and
                // the canonical gadget carrier are unchanged.
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
                    let coefficient = self.integer_expression_magnitude(coefficient, env)?;
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
                let minimum = self.integer_expression(&range.minimum, env)?;
                let maximum = self.integer_expression(&range.maximum, env)?;
                let mut z = state::uniform_interval_sample(
                    &t,
                    &minimum.minimum,
                    &maximum.maximum_inclusive,
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
                let bound = self.integer_expression_magnitude(max_coefficient_bound, env)?;
                Ok(vec![Info {
                    value: AbstractValue::Matrix(state::gaussian_sample(&t, &bound)?),
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
                            gadget_base: self.singleton_integer_expression(
                                gadget_base,
                                env,
                                "trapdoor gadget base",
                            )?,
                            digit_count: self
                                .singleton_integer_expression(
                                    digit_count,
                                    env,
                                    "trapdoor digit count",
                                )?
                                .to_usize()
                                .ok_or_else(|| bad("invalid digit count"))?,
                            preimage_max_coefficient_bound: self.integer_expression_magnitude(
                                preimage_max_coefficient_bound,
                                env,
                            )?,
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
                        &self.integer_expression_magnitude(max_coefficient_bound, env)?,
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
                // A scalar public/trapdoor pair is the rank-zero group case:
                // B * K[d] = T[d].  Normalize both scalar operands to the
                // empty group prefix exactly as validation and runtime do.
                let public_shape = value_family_shape(&xs[0].value).unwrap_or_default();
                let trapdoor_shape = value_family_shape(&xs[1].value).unwrap_or_default();
                let target_shape = value_family_shape(&xs[2].value);
                let expected_group = target_shape
                    .as_ref()
                    .and_then(|shape| {
                        (!shape.is_empty()).then(|| shape[..shape.len() - 1].to_vec())
                    })
                    .unwrap_or_default();
                if public_shape != trapdoor_shape || public_shape != expected_group {
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
                if self
                    .source_lineages
                    .get(&source)
                    .is_none_or(|lineage| lineage.shape != public_shape)
                {
                    source = self.group_source_for(vec![source], public_shape);
                }
                Ok(vec![Info {
                    value: AbstractValue::Family(FamilyState::new(
                        shape,
                        AbstractValue::Matrix(state::preimage_sample(
                            &t,
                            &self.integer_expression_magnitude(max_coefficient_bound, env)?,
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
                let b = self.singleton_integer_expression(base, env, "decomposition base")?;
                let d = self
                    .singleton_integer_expression(digit_count, env, "decomposition digit count")?
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
            NodeKind::TrapdoorPublic => {
                let AbstractValue::Trapdoor(trapdoor) = &xs[0].value else {
                    return Err(bad("trapdoor public projection requires a trapdoor"));
                };
                let descriptor = crate::GadgetDescriptor {
                    modulus: trapdoor.matrix.modulus.clone(),
                    ring_dimension: trapdoor.matrix.ring_dimension,
                    rows: trapdoor.matrix.rows,
                    columns: trapdoor.matrix.columns,
                    base: trapdoor.gadget_base.clone(),
                    digit_count: trapdoor.digit_count,
                    small: false,
                };
                let mut public = if self.gadget_sources.contains_key(&descriptor) {
                    state::gadget_matrix(
                        &trapdoor.matrix,
                        &trapdoor.gadget_base,
                        trapdoor.digit_count,
                    )?
                } else {
                    state::trapdoor_public_matrix(&trapdoor.matrix)?
                };
                if let Some(source) = self.gadget_sources.get(&descriptor).copied() {
                    public.right_carrier =
                        Some(crate::RightCarrier { source, left_gain: 1u8.into() });
                }
                Ok(vec![Info {
                    value: AbstractValue::Matrix(public),
                    ty: Some(trapdoor.matrix.clone()),
                    relation: None,
                    view: crate::FamilyViewId(u32::MAX),
                    paired_public: None,
                }])
            }
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
                let modulus =
                    self.singleton_integer_expression(plaintext_modulus, env, "plaintext modulus")?;
                let length =
                    self.singleton_integer_expression(length, env, "decoded output length")?;
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
                    &self.integer_expression_magnitude(&reconstruction_coefficients[0], env)?,
                    &t.modulus,
                )?;
                for (x, coefficient) in
                    xs.iter().skip(1).zip(reconstruction_coefficients.iter().skip(1))
                {
                    let term = matrix(x)?
                        .scale(&self.integer_expression_magnitude(coefficient, env)?, &t.modulus)?;
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
                let bits = self
                    .singleton_integer_expression(coefficient_bits, env, "coefficient bit count")?
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
                        self.singleton_integer_expression(e, env, "family extent")
                            .ok()
                            .and_then(|value| value.to_usize())
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
                let packs_preimages = scope
                    .node(mxx_ir_core::NodeId(n as u64))
                    .and_then(|node| node.output_types().first())
                    .is_some_and(|ty| {
                        matches!(
                            ty,
                            WireType::Family { element, .. }
                                if matches!(element.as_ref(), WireType::Preimage(_))
                        )
                    });
                let relation = packs_preimages
                    .then(|| self.pack_shared_preimage_relation(xs, &shape, site()))
                    .transpose()?;
                Ok(vec![Info {
                    value: packed,
                    ty: first.ty.clone(),
                    relation,
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
                        eval_index_interval(
                            index,
                            env,
                            &self.abstract_integers,
                            &self.abstract_loop_indices,
                            &[],
                        )
                        .ok()
                        .is_none_or(|range| {
                            range.minimum < BigInt::zero() ||
                                range.maximum_inclusive >= BigInt::from(f.shape[axis])
                        })
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
                let source_map = relation
                    .as_ref()
                    .map(|relation| {
                        let projection =
                            self.relation_source_projection(relation, &f.shape, site())?;
                        let coordinates =
                            projection.input_prefix(&map.input_indices).ok_or_else(|| {
                                SimulationError::Relation {
                                    message:
                                        "static preimage projection rank does not match its family"
                                            .into(),
                                    site: site(),
                                }
                            })?;
                        Ok::<_, SimulationError>(mxx_ir_core::IndexMap::new(coordinates))
                    })
                    .transpose()?;
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
                    let source_map = source_map.as_ref().expect("relation projection exists");
                    let mapped = self.mapped_source_for(source, source_map, Vec::new(), Some(env));
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
                        message: "a relation-bearing family may select only its final axis".into(),
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
                let relation_projection = xs[0]
                    .relation
                    .as_ref()
                    .map(|relation| self.relation_source_projection(relation, &f.shape, site()))
                    .transpose()?;
                let mut relation =
                    xs[0].relation.clone().map(|r| specialize_relation(r, &selectors));
                if let Some(source) = relation.as_ref().map(|relation| relation.source) {
                    let projection = relation_projection.expect("relation projection exists");
                    let relation_output_shape = value_family_shape(&v).unwrap_or_default();
                    // Selecting the final branch of shared B[g]K[g,d]=T[g,d]
                    // leaves B[g] unchanged. Selecting the final axis of a
                    // pointwise relation applies the same selector to B and K.
                    let source_output_shape = relation_output_shape.clone();
                    let mapped = self.source_after_axis_selection(
                        source,
                        projection.is_shared(),
                        *axis,
                        f.shape.len(),
                        selectors.clone(),
                        source_output_shape,
                    );
                    remap_carriers(&mut v, |candidate| {
                        if candidate == source {
                            mapped
                        } else {
                            self.gathered_source_for_concrete(
                                candidate,
                                selectors.clone(),
                                relation_output_shape.clone(),
                                None,
                            )
                        }
                    });
                    if let Some(relation) = relation.as_mut() {
                        relation.source = mapped;
                    }
                } else {
                    let output_shape = value_family_shape(&v).unwrap_or_default();
                    remap_carriers(&mut v, |source| {
                        self.gathered_source_for_concrete(
                            source,
                            selectors.clone(),
                            output_shape.clone(),
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
                        self.singleton_integer_expression(e, env, "family extent")
                            .ok()
                            .and_then(|value| value.to_usize())
                            .ok_or_else(|| bad("invalid family extent"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let mut relation = xs[0].relation.clone();
                let relation_projection = relation
                    .as_ref()
                    .map(|relation| {
                        let input_shape = value_family_shape(&xs[0].value)
                            .ok_or_else(|| bad("relation-bearing reindex input must be a family"))?;
                        let projection =
                            self.relation_source_projection(relation, &input_shape, site())?;
                        let source_coordinates = projection
                            .input_prefix(&map.input_indices)
                            .ok_or_else(|| SimulationError::Relation {
                                message: "preimage reindex map rank does not match its input family"
                                    .into(),
                                site: site(),
                            })?;
                        let source_shape = projection.output_prefix(&shape).ok_or_else(|| {
                            SimulationError::Relation {
                                message:
                                    "preimage reindex output cannot preserve its shared-source suffix"
                                        .into(),
                                site: site(),
                            }
                        })?;
                        if projection.is_shared() &&
                            source_coordinates.iter().any(|expression| {
                                (source_shape.len()..shape.len())
                                    .any(|axis| index_expr_depends_axis(expression, axis))
                            })
                        {
                            return Err(SimulationError::BranchDependentSource { site: site() });
                        }
                        Ok((mxx_ir_core::IndexMap::new(source_coordinates), source_shape))
                    })
                    .transpose()?;
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
                    let (source_map, source_shape) =
                        relation_projection.as_ref().expect("relation projection exists");
                    relation.source =
                        self.mapped_source_for(source, source_map, source_shape.clone(), Some(env));
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
                        self.singleton_integer_expression(e, env, "family extent")
                            .ok()
                            .and_then(|value| value.to_usize())
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
                let relation_projection = relation
                    .as_ref()
                    .map(|relation| {
                        self.gathered_relation_source_projection(
                            relation,
                            &source_shape,
                            &shape,
                            &xs[1..],
                            &selectors,
                            site(),
                        )
                    })
                    .transpose()?;
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
                    let (source_selectors, source_shape) =
                        relation_projection.as_ref().expect("relation projection exists");
                    let mapped = self.gathered_source_for(
                        source,
                        source_selectors.clone(),
                        source_shape.clone(),
                    );
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
                let count = self
                    .singleton_integer_expression(count, env, "select branch count")?
                    .to_usize()
                    .ok_or_else(|| bad("invalid select count"))?;
                if xs.len() != count.saturating_add(1) {
                    return Err(bad("select branch count mismatch"));
                }
                validate_index(&xs[0], count, site())?;
                // A singleton selector chooses one branch directly.  For a
                // symbolic Boolean selector, recompute an affine integer
                // branch over the binder interval where that selector value
                // holds before joining the alternatives.  Thus an inactive
                // `slot - base` value cannot pollute a later dynamic gather,
                // while selectors without such provenance use the ordinary
                // conservative interval join.
                let selector = int(&xs[0])?;
                let selector_truth = truth_fact(self.scalar_fact(&xs[0]));
                let refine_branch = |branch: &Info, outcome: &OutcomeRefinement| {
                    let mut branch = branch.clone();
                    let affine = affine_fact(self.scalar_fact(&branch));
                    if let AbstractValue::Integer(range) = &mut branch.value &&
                        let Some(affine) = affine &&
                        let Some(refined) = affine.range_under(outcome, &self.binder_ranges)
                    {
                        *range = refined;
                    }
                    branch
                };
                let mut branches = xs[1..].to_vec();
                if count == 2 &&
                    let Some(truth) = &selector_truth
                {
                    branches[0] = refine_branch(&branches[0], &truth.when_zero);
                    branches[1] = refine_branch(&branches[1], &truth.when_one);
                }
                let mut reachable = reachable_select_branches(&branches, &selector)?.iter();
                let mut selected =
                    reachable.next().expect("validated nonempty selector range").clone();
                if let Some(relation) = selected.relation.take() {
                    let selector_id = self.selector_for(xs[0].view);
                    selected.relation = Some(specialize_relation(relation, &[selector_id]));
                }
                for branch in reachable {
                    let type_info = selected.ty.clone();
                    selected = self.join_uniform_with_diagnostics(
                        selected,
                        branch.clone(),
                        type_info.as_ref(),
                        site(),
                    )?;
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
            let count = self
                .singleton_integer_expression(&spec.count, env, "sequential loop count")?
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
                let saved_integers = self.abstract_integers.clone();
                let saved_integer_facts = self.abstract_integer_facts.clone();
                let saved_integer_dependencies = self.abstract_integer_dependencies.clone();
                let saved_loop_indices = self.abstract_loop_indices.clone();
                let saved_loop_atoms = self.abstract_loop_atoms.clone();
                let mut loop_env = env.clone();
                loop_env.loop_indices.insert(spec.index_slot, iteration.into());
                self.abstract_loop_indices
                    .insert(spec.index_slot, state::IntegerState::singleton(iteration));
                let atom = self.next_binder_atom;
                self.next_binder_atom += 1;
                self.abstract_loop_atoms.insert(spec.index_slot, atom);
                self.binder_ranges.insert(atom, state::IntegerState::singleton(iteration));
                let binding_env = loop_env.clone();
                loop_env = apply_bindings(loop_env, &spec.bindings)?;
                apply_abstract_bindings(
                    &mut self.abstract_integers,
                    &mut self.abstract_integer_facts,
                    &mut self.abstract_integer_dependencies,
                    &spec.bindings,
                    &binding_env,
                    &self.abstract_loop_indices,
                    &self.abstract_loop_atoms,
                )?;
                let mut args = current.clone();
                args.extend_from_slice(invariant);
                let preload = cs.inputs().iter().copied().zip(args).collect();
                let mut child_occurrence = occurrence.to_vec();
                child_occurrence.push(format!("node:{n}/iteration:{iteration}"));
                let result = self.scope(stage, graph, &child, &child_occurrence, loop_env, preload);
                self.abstract_integers = saved_integers;
                self.abstract_integer_facts = saved_integer_facts;
                self.abstract_integer_dependencies = saved_integer_dependencies;
                self.abstract_loop_indices = saved_loop_indices;
                self.abstract_loop_atoms = saved_loop_atoms;
                let vals = result?;
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
                    self.singleton_integer_expression(extent, env, "parallel grid extent")?
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
                .ok_or_else(|| SimulationError::InvalidGraph {
                    message: "parallel grid cardinality overflows usize".into(),
                    site: None,
                })?;
            let mut grid_binder_axes = HashMap::new();
            let vals = if lane_count == 0 {
                let grid_node = graph
                    .scope(parent)
                    .and_then(|scope| scope.node(mxx_ir_core::NodeId(n as u64)))
                    .expect("validated parallel grid node");
                cs.outputs()
                    .iter()
                    .copied()
                    .zip(grid_node.output_types().iter())
                    .enumerate()
                    .map(|(port, (wire, output_type))| {
                        let WireType::Family { element, .. } = output_type else {
                            return Err(SimulationError::InvalidGraph {
                                message: "parallel grid output must be a family".into(),
                                site: None,
                            });
                        };
                        let mut info = empty_info_for_type(element, env)?;
                        info.view = self.view_for_wire(
                            stage,
                            parent,
                            occurrence,
                            WireRef {
                                node: mxx_ir_core::NodeId(n as u64),
                                port: mxx_ir_core::Port(port as u32),
                            },
                            Some(output_type),
                            grid_node.kind(),
                            xs,
                            env,
                        )?;
                        Ok((wire, info))
                    })
                    .collect::<Result<HashMap<_, _>, SimulationError>>()?
            } else {
                // The body is one symbolic occurrence.  Loop slots carry their
                // full coordinate intervals, so one transfer covers every
                // concrete lane without making simulation cost depend on the
                // family cardinality.
                let saved_integers = self.abstract_integers.clone();
                let saved_integer_facts = self.abstract_integer_facts.clone();
                let saved_integer_dependencies = self.abstract_integer_dependencies.clone();
                let saved_loop_indices = self.abstract_loop_indices.clone();
                let saved_loop_atoms = self.abstract_loop_atoms.clone();
                let mut grid_env = env.clone();
                for (axis, ((slot, extent), representative)) in spec
                    .index_slots
                    .iter()
                    .zip(&grid_shape)
                    .zip(std::iter::repeat(0usize))
                    .enumerate()
                {
                    grid_env.loop_indices.insert(*slot, representative.into());
                    self.abstract_loop_indices.insert(
                        *slot,
                        state::IntegerState::new(0.into(), BigInt::from(extent - 1))?,
                    );
                    let atom = self.next_binder_atom;
                    self.next_binder_atom += 1;
                    self.abstract_loop_atoms.insert(*slot, atom);
                    grid_binder_axes.insert(atom, axis);
                    self.binder_ranges.insert(
                        atom,
                        state::IntegerState::new(0.into(), BigInt::from(extent - 1))?,
                    );
                }
                let binding_env = grid_env.clone();
                grid_env = apply_bindings(grid_env, &spec.bindings)?;
                apply_abstract_bindings(
                    &mut self.abstract_integers,
                    &mut self.abstract_integer_facts,
                    &mut self.abstract_integer_dependencies,
                    &spec.bindings,
                    &binding_env,
                    &self.abstract_loop_indices,
                    &self.abstract_loop_atoms,
                )?;
                let preload = cs
                    .inputs()
                    .iter()
                    .copied()
                    .zip(xs.iter().cloned())
                    .enumerate()
                    .map(|(arg, (wire, value))| {
                        let mapped = match spec.input_modes.get(arg) {
                            Some(mxx_ir_core::node::GridInputMode::Reindex { map }) => {
                                let coordinate_ranges = map
                                    .input_indices
                                    .iter()
                                    .map(|expr| {
                                        eval_index_interval(
                                            expr,
                                            &grid_env,
                                            &self.abstract_integers,
                                            &self.abstract_loop_indices,
                                            &spec.index_slots,
                                        )
                                    })
                                    .collect::<Result<Vec<_>, _>>()?;
                                let family_shape = match &value.value {
                                    AbstractValue::Family(family) => family.shape.clone(),
                                    _ => unreachable!(),
                                };
                                if coordinate_ranges.len() != family_shape.len() ||
                                    coordinate_ranges.iter().enumerate().any(|(axis, range)| {
                                        range.minimum < BigInt::zero() ||
                                            range.maximum_inclusive >=
                                                BigInt::from(family_shape[axis])
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
                                    self.mapped_source_for(source, map, grid_shape.clone(), None)
                                });
                                if let Some(source) = relation_source {
                                    let mapped_source = self.mapped_source_for(
                                        source,
                                        map,
                                        grid_shape.clone(),
                                        None,
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
                                        None,
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
                let result = self.scope(stage, graph, &child, &child_occurrence, grid_env, preload);
                self.abstract_integers = saved_integers;
                self.abstract_integer_facts = saved_integer_facts;
                self.abstract_integer_dependencies = saved_integer_dependencies;
                self.abstract_loop_indices = saved_loop_indices;
                self.abstract_loop_atoms = saved_loop_atoms;
                result?
            };
            cs.outputs()
                .iter()
                .map(|wire| {
                    let mut info =
                        vals.get(wire).cloned().ok_or_else(|| SimulationError::InvalidGraph {
                            message: "missing parallel-grid output".into(),
                            site: None,
                        })?;
                    let body_view = info.view;
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
                    let dependencies = if lane_count == 0 {
                        BTreeSet::new()
                    } else {
                        self.scalar_dependencies
                            .get(&body_view)
                            .and_then(|dependencies| match dependencies {
                                BinderDependencies::Known(binders) => Some(binders),
                                BinderDependencies::Unknown => None,
                            })
                            .map(|binders| {
                                binders
                                    .iter()
                                    .map(|binder| grid_binder_axes.get(binder).copied())
                                    .collect::<Option<BTreeSet<_>>>()
                                    .unwrap_or_else(|| (0..grid_shape.len()).collect())
                            })
                            .unwrap_or_else(|| (0..grid_shape.len()).collect())
                    };
                    self.family_axis_dependencies.insert(view, dependencies);
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
            let saved_integers = self.abstract_integers.clone();
            let saved_integer_facts = self.abstract_integer_facts.clone();
            let saved_integer_dependencies = self.abstract_integer_dependencies.clone();
            if let Some(mxx_ir_core::node::NodeKind::SubgraphCall(spec)) = graph
                .scope(parent)
                .and_then(|scope| scope.node(mxx_ir_core::NodeId(n as u64)))
                .map(|node| node.kind())
            {
                apply_abstract_bindings(
                    &mut self.abstract_integers,
                    &mut self.abstract_integer_facts,
                    &mut self.abstract_integer_dependencies,
                    &spec.bindings,
                    env,
                    &self.abstract_loop_indices,
                    &self.abstract_loop_atoms,
                )?;
            }
            let mut child_occurrence = occurrence.to_vec();
            child_occurrence.push(format!("node:{n}"));
            let result = self.scope(stage, graph, &child, &child_occurrence, child_env, preload);
            self.abstract_integers = saved_integers;
            self.abstract_integer_facts = saved_integer_facts;
            self.abstract_integer_dependencies = saved_integer_dependencies;
            let vals = result?;
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
    integer_range(state::IntegerState::singleton(x))
}

fn integer_range(range: state::IntegerState) -> Info {
    Info {
        value: AbstractValue::Integer(range),
        ty: None,
        relation: None,
        view: crate::FamilyViewId(u32::MAX),
        paired_public: None,
    }
}

fn empty_info_for_type(ty: &WireType, env: &ParamEnv) -> Result<Info, SimulationError> {
    let invalid =
        |message: &str| SimulationError::InvalidGraph { message: message.into(), site: None };
    let (value, matrix_type) = match ty {
        WireType::Matrix(_) | WireType::Preimage(_) => {
            let matrix = concrete_matrix(ty, env).ok_or_else(|| invalid("invalid matrix type"))?;
            (AbstractValue::Matrix(state::zero_matrix(&matrix)?), Some(matrix))
        }
        WireType::ConstantInt | WireType::Int => {
            (AbstractValue::Integer(state::IntegerState::singleton(0)), None)
        }
        WireType::ConstantBool | WireType::Bool => {
            (AbstractValue::Boolean(state::BooleanState::FalseOnly), None)
        }
        WireType::Bytes { .. } => (AbstractValue::Bytes, None),
        WireType::TypedBlob { type_name, schema_hash } => (
            AbstractValue::TypedBlob { type_name: type_name.clone(), schema_hash: *schema_hash },
            None,
        ),
        WireType::Trapdoor {
            matrix,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => {
            let matrix = concrete_matrix(&WireType::Matrix(matrix.clone()), env)
                .ok_or_else(|| invalid("invalid trapdoor matrix type"))?;
            let digit_count = digit_count
                .evaluate(env)
                .ok()
                .and_then(|value| value.to_usize())
                .ok_or_else(|| invalid("invalid trapdoor digit count"))?;
            (
                AbstractValue::Trapdoor(TrapdoorState {
                    matrix,
                    sigma: sigma.clone(),
                    gadget_base: gadget_base
                        .evaluate(env)
                        .map_err(|error| invalid(&error.to_string()))?,
                    digit_count,
                    preimage_max_coefficient_bound: preimage_max_coefficient_bound
                        .evaluate(env)
                        .map_err(|error| invalid(&error.to_string()))?,
                }),
                None,
            )
        }
        WireType::ConstantReal | WireType::Real => (AbstractValue::Real, None),
        WireType::Family { .. } => {
            return Err(SimulationError::Unsupported {
                operation: "empty parallel-grid output element type".into(),
                site: None,
            });
        }
    };
    Ok(Info {
        value,
        ty: matrix_type,
        relation: None,
        view: crate::FamilyViewId(u32::MAX),
        paired_public: None,
    })
}

impl AffineScalar {
    fn constant(value: &BigInt) -> Self {
        Self { binder: None, coefficient: BigInt::zero(), offset: value.clone() }
    }

    fn binder(binder: u64) -> Self {
        Self { binder: Some(binder), coefficient: BigInt::one(), offset: BigInt::zero() }
    }

    fn add(&self, other: &Self) -> Option<Self> {
        match (self.binder, other.binder) {
            (Some(left), Some(right)) if left != right => None,
            (binder, _) => Some(Self {
                binder: binder.or(other.binder),
                coefficient: &self.coefficient + &other.coefficient,
                offset: &self.offset + &other.offset,
            }),
        }
    }

    fn subtract(&self, other: &Self) -> Option<Self> {
        match (self.binder, other.binder) {
            (Some(left), Some(right)) if left != right => None,
            (binder, _) => Some(Self {
                binder: binder.or(other.binder),
                coefficient: &self.coefficient - &other.coefficient,
                offset: &self.offset - &other.offset,
            }),
        }
    }

    fn multiply_constant(&self, scalar: &BigInt) -> Self {
        Self {
            binder: self.binder,
            coefficient: &self.coefficient * scalar,
            offset: &self.offset * scalar,
        }
    }

    fn range_under(
        &self,
        refinement: &OutcomeRefinement,
        binders: &HashMap<u64, state::IntegerState>,
    ) -> Option<state::IntegerState> {
        let Some(binder_atom) = self.binder else {
            return Some(state::IntegerState::singleton(self.offset.clone()));
        };
        let mut binder = binders.get(&binder_atom)?.clone();
        match refinement {
            OutcomeRefinement::Impossible => return None,
            OutcomeRefinement::Unconstrained => {}
            OutcomeRefinement::Restricted(required) if required.binder == binder_atom => {
                binder.minimum = binder.minimum.max(required.range.minimum.clone());
                binder.maximum_inclusive =
                    binder.maximum_inclusive.min(required.range.maximum_inclusive.clone());
                if binder.minimum > binder.maximum_inclusive {
                    return None;
                }
            }
            OutcomeRefinement::Restricted(_) => return None,
        }
        let low = &self.coefficient * &binder.minimum + &self.offset;
        let high = &self.coefficient * &binder.maximum_inclusive + &self.offset;
        state::IntegerState::new(low.clone().min(high.clone()), low.max(high)).ok()
    }
}

fn intersect_outcomes(left: &OutcomeRefinement, right: &OutcomeRefinement) -> OutcomeRefinement {
    match (left, right) {
        (OutcomeRefinement::Impossible, _) | (_, OutcomeRefinement::Impossible) => {
            OutcomeRefinement::Impossible
        }
        (OutcomeRefinement::Unconstrained, other) | (other, OutcomeRefinement::Unconstrained) => {
            other.clone()
        }
        (OutcomeRefinement::Restricted(left), OutcomeRefinement::Restricted(right))
            if left.binder == right.binder =>
        {
            let minimum = left.range.minimum.clone().max(right.range.minimum.clone());
            let maximum =
                left.range.maximum_inclusive.clone().min(right.range.maximum_inclusive.clone());
            if minimum > maximum {
                OutcomeRefinement::Impossible
            } else {
                OutcomeRefinement::Restricted(BinderRefinement {
                    binder: left.binder,
                    range: state::IntegerState::new(minimum, maximum)
                        .expect("ordered intersection"),
                })
            }
        }
        // A conjunction over different binders is outside this deliberately
        // one-binder reduced product.  Dropping it is conservative.
        _ => OutcomeRefinement::Unconstrained,
    }
}

fn affine_fact(fact: Option<&ScalarFacts>) -> Option<AffineScalar> {
    match fact {
        Some(ScalarFacts::Affine(affine)) => Some(affine.clone()),
        _ => None,
    }
}

fn truth_fact(fact: Option<&ScalarFacts>) -> Option<TruthFacts> {
    match fact {
        Some(ScalarFacts::Truth(truth)) => Some(truth.clone()),
        _ => None,
    }
}

fn comparison_facts(
    operation: mxx_ir_core::node::IntCompareOp,
    left: Option<&ScalarFacts>,
    right: Option<&ScalarFacts>,
    binders: &HashMap<u64, state::IntegerState>,
) -> Option<ScalarFacts> {
    let difference = affine_fact(left)?.subtract(&affine_fact(right)?)?;
    let binder_atom = difference.binder?;
    // Unit slope is sufficient for loop-index boundary predicates and keeps
    // integer rounding out of the trusted refinement logic.
    if difference.coefficient != BigInt::one() && difference.coefficient != -BigInt::one() {
        return None;
    }
    let binder = binders.get(&binder_atom)?;
    let restricted = |minimum: BigInt, maximum: BigInt| {
        let minimum = minimum.max(binder.minimum.clone());
        let maximum = maximum.min(binder.maximum_inclusive.clone());
        if minimum > maximum {
            OutcomeRefinement::Impossible
        } else {
            OutcomeRefinement::Restricted(BinderRefinement {
                binder: binder_atom,
                range: state::IntegerState::new(minimum, maximum).expect("ordered refinement"),
            })
        }
    };
    let (when_zero, when_one) = match (operation, difference.coefficient.sign()) {
        (mxx_ir_core::node::IntCompareOp::LessEqual, num_bigint::Sign::Plus) => (
            restricted(-&difference.offset + 1, binder.maximum_inclusive.clone()),
            restricted(binder.minimum.clone(), -&difference.offset),
        ),
        (mxx_ir_core::node::IntCompareOp::LessEqual, num_bigint::Sign::Minus) => (
            restricted(binder.minimum.clone(), difference.offset.clone() - 1),
            restricted(difference.offset.clone(), binder.maximum_inclusive.clone()),
        ),
        (mxx_ir_core::node::IntCompareOp::Less, num_bigint::Sign::Plus) => (
            restricted(-&difference.offset, binder.maximum_inclusive.clone()),
            restricted(binder.minimum.clone(), -&difference.offset - 1),
        ),
        (mxx_ir_core::node::IntCompareOp::Less, num_bigint::Sign::Minus) => (
            restricted(binder.minimum.clone(), difference.offset.clone()),
            restricted(difference.offset.clone() + 1, binder.maximum_inclusive.clone()),
        ),
        (mxx_ir_core::node::IntCompareOp::Equal, _) => (OutcomeRefinement::Unconstrained, {
            let value = if difference.coefficient.is_positive() {
                -difference.offset
            } else {
                difference.offset
            };
            restricted(value.clone(), value)
        }),
        _ => return None,
    };
    Some(ScalarFacts::Truth(TruthFacts { when_zero, when_one }))
}

fn eval_int_facts(
    expression: &mxx_ir_core::IntExpr,
    integers: &BTreeMap<String, ScalarFacts>,
    loop_atoms: &HashMap<u32, u64>,
) -> Option<ScalarFacts> {
    use mxx_ir_core::IntExpr;
    fn evaluate(
        expression: &IntExpr,
        integers: &BTreeMap<String, ScalarFacts>,
        loop_atoms: &HashMap<u32, u64>,
    ) -> Option<AffineScalar> {
        match expression {
            IntExpr::Const(value) => Some(AffineScalar::constant(value)),
            IntExpr::Var(name) => affine_fact(integers.get(name)),
            IntExpr::LoopIndex(slot) => Some(AffineScalar::binder(*loop_atoms.get(slot)?)),
            IntExpr::Add(left, right) => {
                evaluate(left, integers, loop_atoms)?.add(&evaluate(right, integers, loop_atoms)?)
            }
            IntExpr::Sub(left, right) => evaluate(left, integers, loop_atoms)?
                .subtract(&evaluate(right, integers, loop_atoms)?),
            IntExpr::Mul(left, right) => {
                let left = evaluate(left, integers, loop_atoms)?;
                let right = evaluate(right, integers, loop_atoms)?;
                if left.binder.is_none() {
                    Some(right.multiply_constant(&left.offset))
                } else if right.binder.is_none() {
                    Some(left.multiply_constant(&right.offset))
                } else {
                    None
                }
            }
            IntExpr::Div(left, right) => {
                let left = evaluate(left, integers, loop_atoms)?;
                let right = evaluate(right, integers, loop_atoms)?;
                if right.binder.is_some() ||
                    right.offset.is_zero() ||
                    &left.coefficient % &right.offset != BigInt::zero() ||
                    &left.offset % &right.offset != BigInt::zero()
                {
                    None
                } else {
                    Some(AffineScalar {
                        binder: left.binder,
                        coefficient: left.coefficient / &right.offset,
                        offset: left.offset / right.offset,
                    })
                }
            }
            IntExpr::RoundDiv(_, _) | IntExpr::Log2Ceil(_) => None,
        }
    }
    evaluate(&expression.canonicalize(), integers, loop_atoms).map(ScalarFacts::Affine)
}

fn int_expr_dependencies(
    expression: &mxx_ir_core::IntExpr,
    variables: &BTreeMap<String, BinderDependencies>,
    loop_atoms: &HashMap<u32, u64>,
) -> BinderDependencies {
    use mxx_ir_core::IntExpr;
    match expression {
        IntExpr::Const(_) => BinderDependencies::Known(BTreeSet::new()),
        IntExpr::Var(name) => variables.get(name).cloned().unwrap_or(BinderDependencies::Unknown),
        IntExpr::LoopIndex(slot) => loop_atoms
            .get(slot)
            .map(|atom| BinderDependencies::Known(BTreeSet::from([*atom])))
            .unwrap_or(BinderDependencies::Unknown),
        IntExpr::Add(left, right) |
        IntExpr::Sub(left, right) |
        IntExpr::Mul(left, right) |
        IntExpr::Div(left, right) |
        IntExpr::RoundDiv(left, right) => BinderDependencies::union([
            &int_expr_dependencies(left, variables, loop_atoms),
            &int_expr_dependencies(right, variables, loop_atoms),
        ]),
        IntExpr::Log2Ceil(value) => int_expr_dependencies(value, variables, loop_atoms),
    }
}

fn eval_int_interval(
    expression: &mxx_ir_core::IntExpr,
    concrete: &ParamEnv,
    integers: &BTreeMap<String, state::IntegerState>,
    loop_indices: &HashMap<u32, state::IntegerState>,
) -> Result<state::IntegerState, SimulationError> {
    use mxx_ir_core::IntExpr;

    fn log2_ceil(value: &BigInt) -> Result<BigInt, SimulationError> {
        let positive = value.to_biguint().filter(|value| !value.is_zero()).ok_or_else(|| {
            SimulationError::InvalidParameterEnvironment {
                message: "log2ceil argument must be positive".into(),
            }
        })?;
        let floor = positive.bits() - 1;
        Ok(BigInt::from(if positive == (BigUint::one() << floor as usize) {
            floor
        } else {
            floor + 1
        }))
    }

    fn evaluate(
        expression: &IntExpr,
        concrete: &ParamEnv,
        integers: &BTreeMap<String, state::IntegerState>,
        loop_indices: &HashMap<u32, state::IntegerState>,
    ) -> Result<state::IntegerState, SimulationError> {
        let invalid = |message: String| SimulationError::InvalidParameterEnvironment { message };
        Ok(match expression {
            IntExpr::Const(value) => state::IntegerState::singleton(value.clone()),
            IntExpr::Var(name) => integers
                .get(name)
                .cloned()
                .or_else(|| {
                    concrete.integers.get(name).cloned().map(state::IntegerState::singleton)
                })
                .ok_or_else(|| invalid(format!("unbound integer variable {name}")))?,
            IntExpr::LoopIndex(slot) => loop_indices
                .get(slot)
                .cloned()
                .or_else(|| {
                    concrete.loop_indices.get(slot).cloned().map(state::IntegerState::singleton)
                })
                .ok_or_else(|| invalid(format!("unbound loop-index[{slot}]")))?,
            IntExpr::Add(left, right) => evaluate(left, concrete, integers, loop_indices)?
                .add(&evaluate(right, concrete, integers, loop_indices)?),
            IntExpr::Sub(left, right) => evaluate(left, concrete, integers, loop_indices)?
                .subtract(&evaluate(right, concrete, integers, loop_indices)?),
            IntExpr::Mul(left, right) => evaluate(left, concrete, integers, loop_indices)?
                .multiply(&evaluate(right, concrete, integers, loop_indices)?),
            IntExpr::Div(left, right) => {
                let numerator = evaluate(left, concrete, integers, loop_indices)?;
                let denominator = evaluate(right, concrete, integers, loop_indices)?;
                if numerator.minimum != numerator.maximum_inclusive ||
                    denominator.minimum != denominator.maximum_inclusive
                {
                    return Err(invalid(
                        "exact division of a non-singleton symbolic interval is unsupported".into(),
                    ));
                }
                if denominator.minimum.is_zero() ||
                    &numerator.minimum % &denominator.minimum != BigInt::zero()
                {
                    return Err(invalid("symbolic integer division is not exact".into()));
                }
                state::IntegerState::singleton(&numerator.minimum / &denominator.minimum)
            }
            IntExpr::RoundDiv(left, right) => {
                let numerator = evaluate(left, concrete, integers, loop_indices)?;
                let denominator = evaluate(right, concrete, integers, loop_indices)?;
                if denominator.minimum <= BigInt::zero() {
                    return Err(invalid("RoundDiv denominator must be positive".into()));
                }
                let two = BigInt::from(2);
                let rounded = |n: &BigInt, d: &BigInt| {
                    let numerator = n * &two + d;
                    let denominator = d * &two;
                    let quotient = &numerator / &denominator;
                    if &numerator % &denominator < BigInt::zero() {
                        quotient - BigInt::one()
                    } else {
                        quotient
                    }
                };
                let candidates = [
                    rounded(&numerator.minimum, &denominator.minimum),
                    rounded(&numerator.minimum, &denominator.maximum_inclusive),
                    rounded(&numerator.maximum_inclusive, &denominator.minimum),
                    rounded(&numerator.maximum_inclusive, &denominator.maximum_inclusive),
                ];
                state::IntegerState::new(
                    candidates.iter().min().expect("four rounded quotients").clone(),
                    candidates.iter().max().expect("four rounded quotients").clone(),
                )?
            }
            IntExpr::Log2Ceil(value) => {
                let range = evaluate(value, concrete, integers, loop_indices)?;
                state::IntegerState::new(
                    log2_ceil(&range.minimum)?,
                    log2_ceil(&range.maximum_inclusive)?,
                )?
            }
        })
    }

    // Canonicalization preserves correlations that interval arithmetic alone
    // cannot see, such as `i - i = 0`, before ranges are propagated.
    evaluate(&expression.canonicalize(), concrete, integers, loop_indices)
}

fn eval_index_interval(
    expression: &mxx_ir_core::IndexExpr,
    concrete: &ParamEnv,
    integers: &BTreeMap<String, state::IntegerState>,
    loop_indices: &HashMap<u32, state::IntegerState>,
    axis_slots: &[u32],
) -> Result<state::IntegerState, SimulationError> {
    use mxx_ir_core::IndexExpr;

    let invalid = |message: String| SimulationError::InvalidIndexMap { message, site: None };
    let evaluate = |expression: &IndexExpr| {
        eval_index_interval(expression, concrete, integers, loop_indices, axis_slots)
    };
    Ok(match expression.normalize() {
        IndexExpr::Axis(axis) => {
            let slot = axis_slots
                .get(axis)
                .ok_or_else(|| invalid("grid map axis is out of range".into()))?;
            loop_indices
                .get(slot)
                .cloned()
                .ok_or_else(|| invalid(format!("unbound grid axis {axis}")))?
        }
        IndexExpr::Parameter(name) => integers
            .get(&name)
            .cloned()
            .or_else(|| concrete.integers.get(&name).cloned().map(state::IntegerState::singleton))
            .ok_or_else(|| invalid(format!("unbound index parameter {name}")))?,
        IndexExpr::LoopIndex(slot) => loop_indices
            .get(&slot)
            .cloned()
            .or_else(|| {
                concrete.loop_indices.get(&slot).cloned().map(state::IntegerState::singleton)
            })
            .ok_or_else(|| invalid(format!("unbound loop-index[{slot}]")))?,
        IndexExpr::Constant(value) => state::IntegerState::singleton(value),
        IndexExpr::Add(left, right) => evaluate(&left)?.add(&evaluate(&right)?),
        IndexExpr::Subtract(left, right) if left == right => state::IntegerState::singleton(0),
        IndexExpr::Subtract(left, right) => evaluate(&left)?.subtract(&evaluate(&right)?),
        IndexExpr::Multiply(left, right) => evaluate(&left)?.multiply(&evaluate(&right)?),
        IndexExpr::Divide(left, right) => {
            let numerator = evaluate(&left)?;
            let denominator = evaluate(&right)?;
            if denominator.minimum <= BigInt::zero() &&
                denominator.maximum_inclusive >= BigInt::zero()
            {
                return Err(invalid("index divisor range contains zero".into()));
            }
            let candidates = [
                &numerator.minimum / &denominator.minimum,
                &numerator.minimum / &denominator.maximum_inclusive,
                &numerator.maximum_inclusive / &denominator.minimum,
                &numerator.maximum_inclusive / &denominator.maximum_inclusive,
            ];
            state::IntegerState::new(
                candidates.iter().min().expect("four quotients").clone(),
                candidates.iter().max().expect("four quotients").clone(),
            )?
        }
        IndexExpr::Remainder(left, right) => {
            let numerator = evaluate(&left)?;
            let denominator = evaluate(&right)?;
            if denominator.minimum <= BigInt::zero() &&
                denominator.maximum_inclusive >= BigInt::zero()
            {
                return Err(invalid("index divisor range contains zero".into()));
            }
            let maximum =
                denominator.minimum.abs().max(denominator.maximum_inclusive.abs()) - BigInt::one();
            if numerator.minimum >= BigInt::zero() {
                state::IntegerState::new(BigInt::zero(), maximum)?
            } else if numerator.maximum_inclusive <= BigInt::zero() {
                state::IntegerState::new(-maximum, BigInt::zero())?
            } else {
                state::IntegerState::new(-maximum.clone(), maximum)?
            }
        }
        IndexExpr::Equal(left, right) if left == right => state::IntegerState::singleton(1),
        IndexExpr::Equal(left, right) => {
            let left = evaluate(&left)?;
            let right = evaluate(&right)?;
            if left.maximum_inclusive < right.minimum || right.maximum_inclusive < left.minimum {
                state::IntegerState::singleton(0)
            } else if left.minimum == left.maximum_inclusive &&
                right.minimum == right.maximum_inclusive &&
                left.minimum == right.minimum
            {
                state::IntegerState::singleton(1)
            } else {
                state::IntegerState::new(0.into(), 1.into())?
            }
        }
        IndexExpr::Less(left, right) | IndexExpr::LessEqual(left, right) => {
            let strict = matches!(expression.normalize(), IndexExpr::Less(_, _));
            let left = evaluate(&left)?;
            let right = evaluate(&right)?;
            let always_true = if strict {
                left.maximum_inclusive < right.minimum
            } else {
                left.maximum_inclusive <= right.minimum
            };
            let always_false = if strict {
                left.minimum >= right.maximum_inclusive
            } else {
                left.minimum > right.maximum_inclusive
            };
            if always_true {
                state::IntegerState::singleton(1)
            } else if always_false {
                state::IntegerState::singleton(0)
            } else {
                state::IntegerState::new(0.into(), 1.into())?
            }
        }
        IndexExpr::Log2Ceil(value) => {
            let range = evaluate(&value)?;
            let evaluate_endpoint = |value: &BigInt| {
                let positive = value
                    .to_biguint()
                    .filter(|value| !value.is_zero())
                    .ok_or_else(|| invalid("log2ceil argument must be positive".into()))?;
                let floor = positive.bits() - 1;
                Ok::<_, SimulationError>(BigInt::from(
                    if positive == (BigUint::one() << floor as usize) { floor } else { floor + 1 },
                ))
            };
            state::IntegerState::new(
                evaluate_endpoint(&range.minimum)?,
                evaluate_endpoint(&range.maximum_inclusive)?,
            )?
        }
        IndexExpr::Select { selector, branches } => {
            let selector = evaluate(&selector)?;
            let minimum = selector
                .minimum
                .to_usize()
                .ok_or_else(|| invalid("negative index-map selector".into()))?;
            let maximum = selector
                .maximum_inclusive
                .to_usize()
                .ok_or_else(|| invalid("invalid index-map selector".into()))?;
            let selected = branches
                .get(minimum..=maximum)
                .ok_or_else(|| invalid("index-map selector is out of range".into()))?;
            let mut ranges = selected.iter().map(evaluate);
            let first =
                ranges.next().ok_or_else(|| invalid("index-map select has no branch".into()))??;
            ranges
                .try_fold(first, |joined, range| Ok::<_, SimulationError>(joined.join(&range?)))?
        }
    })
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
    // Bindings are simultaneous: every right-hand side reads the unchanged
    // parent environment, including an outer variable shadowed by another
    // binding in this same list.
    let parent = env.clone();
    let evaluated = bindings
        .iter()
        .map(|(name, expression)| {
            expression.evaluate(&parent).map(|value| (name.clone(), value)).map_err(|error| {
                SimulationError::InvalidParameterEnvironment { message: error.to_string() }
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (name, value) in evaluated {
        env.integers.insert(name.clone(), value);
    }
    Ok(env)
}

fn apply_abstract_bindings(
    integers: &mut BTreeMap<String, state::IntegerState>,
    integer_facts: &mut BTreeMap<String, ScalarFacts>,
    integer_dependencies: &mut BTreeMap<String, BinderDependencies>,
    bindings: &[(String, mxx_ir_core::IntExpr)],
    concrete: &ParamEnv,
    loop_indices: &HashMap<u32, state::IntegerState>,
    loop_atoms: &HashMap<u32, u64>,
) -> Result<(), SimulationError> {
    let parent = integers.clone();
    let parent_facts = integer_facts.clone();
    let parent_dependencies = integer_dependencies.clone();
    let evaluated = bindings
        .iter()
        .map(|(name, expression)| {
            eval_int_interval(expression, concrete, &parent, loop_indices)
                .map(|range| (name.clone(), range))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (name, range) in evaluated {
        integers.insert(name, range);
    }
    for (name, expression) in bindings {
        if let Some(facts) = eval_int_facts(expression, &parent_facts, loop_atoms) {
            integer_facts.insert(name.clone(), facts);
        } else {
            integer_facts.remove(name);
        }
        integer_dependencies.insert(
            name.clone(),
            int_expr_dependencies(expression, &parent_dependencies, loop_atoms),
        );
    }
    Ok(())
}

fn int(x: &Info) -> Result<state::IntegerState, SimulationError> {
    match &x.value {
        AbstractValue::Integer(x) => Ok(x.clone()),
        _ => Err(SimulationError::InvalidGraph { message: "integer required".into(), site: None }),
    }
}

fn reachable_select_branches<'a>(
    branches: &'a [Info],
    selector: &state::IntegerState,
) -> Result<&'a [Info], SimulationError> {
    let minimum = selector.minimum.to_usize().ok_or_else(|| SimulationError::InvalidGraph {
        message: "invalid select index".into(),
        site: None,
    })?;
    let maximum = selector.maximum_inclusive.to_usize().ok_or_else(|| {
        SimulationError::InvalidGraph { message: "invalid select index".into(), site: None }
    })?;
    branches.get(minimum..=maximum).ok_or_else(|| SimulationError::InvalidGraph {
        message: "select index range is outside its branches".into(),
        site: None,
    })
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
        // A relation is a universal equation for the selected value.  If one
        // reachable branch lacks that equation, the join cannot attach the
        // other branch's relation to the combined output.
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
    if !lineage_is_complete(parent) || map.input_indices.len() != parent.shape.len() {
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
    if !lineage_is_complete(parent) {
        return None;
    }
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
    if !lineage_is_complete(parent) ||
        axis >= parent.shape.len() ||
        output_shape.len() + 1 != parent.shape.len()
    {
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

/// A complete lineage has one source leaf for every coordinate.  Opaque
/// structural selections intentionally retain only a sentinel leaf, so they
/// preserve the output rank but cannot be mistaken for an exact or uniform
/// coordinate function.
fn lineage_is_complete(lineage: &SourceLineage) -> bool {
    lineage.leaves.len() == lineage.shape.iter().copied().product::<usize>().max(1)
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
            // A zero extent makes every element inaccessible at runtime.  Use
            // the same type-derived bottom summary as an empty ParallelGrid so
            // the caller's illustrative element bound cannot become an
            // observable bound for a family with no members.
            let x = if actual.contains(&0) {
                empty_info_for_type(element, env).map_err(|error| error.to_string())?
            } else {
                value_fact(element, ef, env)?
            };
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
        node::{ArtifactInput, ConstantMatrix, IndexRange},
        types::MatrixType,
        with_new_construction_scope,
    };
    use std::collections::BTreeMap;

    fn run_single_output(
        graph: Graph,
        name: &str,
    ) -> Result<crate::SimulationReport, SimulationError> {
        run_single_output_with_inputs(graph, name, vec![])
    }

    fn run_single_output_with_inputs(
        graph: Graph,
        name: &str,
        external_inputs: Vec<crate::ExternalInputFact>,
    ) -> Result<crate::SimulationReport, SimulationError> {
        let environment = ParamEnv::default();
        let stage = crate::StageId(name.into());
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
            external_inputs,
            limits: crate::SimulationLimits::default(),
        };
        run(&request)
    }

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
    fn zero_extent_external_matrix_family_uses_typed_bottom() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let family_type = WireType::Family {
            element: Box::new(WireType::Matrix(matrix)),
            shape: vec![mxx_ir_core::IntExpr::constant(0)],
        };
        let input = NodeHandle::new(
            NodeKind::Input {
                name: "empty".into(),
                wire_type: family_type.clone(),
                artifact: None,
            },
            vec![],
            vec![family_type],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "empty-external-family",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: input, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let environment = ParamEnv::default();
        let stage = crate::StageId("empty-external-family".into());
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
                input: "empty".into(),
                value: crate::ExternalInputValue::Family {
                    shape: vec![0],
                    // No member exists, so this illustrative element bound is
                    // intentionally replaced by the matrix typed bottom.
                    element: Box::new(crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: 7u8.into(),
                        maximum_absolute_coefficient_value: Some(8u8.into()),
                        is_constant_polynomial: false,
                    }),
                },
            }],
            limits: crate::SimulationLimits::default(),
        };

        let report = run(&request).expect("a validated empty external family must simulate");
        assert_eq!(report.roots[0].maximum_absolute_coefficient_error, BigUint::ZERO);
        assert!(report.diagnostics.dropped_carriers.is_empty());
    }

    #[test]
    fn ring_automorphism_accepts_gadget_trapdoor_public_matrix() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(2),
        };
        let trapdoor = NodeHandle::new(
            NodeKind::GadgetTrapdoor { matrix_type: matrix.clone(), base: 4.into() },
            vec![],
            vec![WireType::Trapdoor {
                matrix: matrix.clone(),
                sigma: mxx_ir_core::RealExpr::Rational(mxx_ir_core::Rational::from_integer(
                    4.into(),
                )),
                gadget_base: 4.into(),
                digit_count: 2.into(),
                preimage_max_coefficient_bound: 0.into(),
            }],
        );
        let public = NodeHandle::new(
            NodeKind::TrapdoorPublic,
            vec![trapdoor.output(0).unwrap()],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let automorphism = NodeHandle::new(
            NodeKind::RingAutomorphism { index: 3.into() },
            vec![public],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "automorphism-gadget-trapdoor",
            vec![],
            BTreeMap::from([(
                String::from("out"),
                GraphOutput { value: automorphism, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let report = run_single_output(graph, "automorphism-gadget-trapdoor").unwrap();
        assert_eq!(report.roots[0].maximum_absolute_coefficient_error, BigUint::ZERO);
    }

    #[test]
    fn ring_automorphism_rejects_tracked_non_gadget_matrices() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let cases = [NodeHandle::new(
            NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value: ConstantMatrix::Zero },
            vec![],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap()];
        for (index, source) in cases.into_iter().enumerate() {
            let automorphism = NodeHandle::new(
                NodeKind::RingAutomorphism { index: 3.into() },
                vec![source],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap();
            let graph = Graph::freeze(
                format!("automorphism-tracked-{index}"),
                vec![],
                BTreeMap::from([(
                    String::from("out"),
                    GraphOutput { value: automorphism, confidentiality: None },
                )]),
                vec![],
                vec![],
                BTreeMap::new(),
            )
            .unwrap()
            .0;
            let result = run_single_output(graph, &format!("automorphism-tracked-{index}"));
            assert!(matches!(result, Err(SimulationError::Relation { .. })));
        }

        let name = "automorphism-tracked-hash";
        let key = NodeHandle::new(
            NodeKind::Input {
                name: "key".into(),
                wire_type: WireType::Bytes { length: mxx_ir_core::IntExpr::constant(32) },
                artifact: None,
            },
            vec![],
            vec![WireType::Bytes { length: mxx_ir_core::IntExpr::constant(32) }],
        )
        .output(0)
        .unwrap();
        let hash = NodeHandle::new(
            NodeKind::HashSample {
                matrix_type: matrix.clone(),
                tag_prefix: vec![1],
                tag_expressions: vec![],
                tag_decimal_expressions: vec![],
                tag_u64_le_expressions: vec![],
            },
            vec![key],
            vec![WireType::Matrix(matrix.clone())],
        )
        .output(0)
        .unwrap();
        let automorphism = NodeHandle::new(
            NodeKind::RingAutomorphism { index: 3.into() },
            vec![hash],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            name,
            vec![],
            BTreeMap::from([(
                String::from("out"),
                GraphOutput { value: automorphism, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let result = run_single_output_with_inputs(
            graph,
            name,
            vec![crate::ExternalInputFact {
                stage: crate::StageId(name.into()),
                input: "key".into(),
                value: crate::ExternalInputValue::Bytes,
            }],
        );
        assert!(matches!(result, Err(SimulationError::Relation { .. })));
    }

    #[test]
    fn ring_automorphism_rejects_tracked_trapdoor_sample_public_matrix() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(3),
        };
        let sample = NodeHandle::new(
            NodeKind::TrapdoorSample {
                matrix_type: matrix.clone(),
                sigma: mxx_ir_core::RealExpr::from_integer(1),
                gadget_base: 4.into(),
                digit_count: 1.into(),
                preimage_max_coefficient_bound: 8.into(),
            },
            vec![],
            vec![
                WireType::Matrix(matrix.clone()),
                WireType::Trapdoor {
                    matrix: matrix.clone(),
                    sigma: mxx_ir_core::RealExpr::from_integer(1),
                    gadget_base: 4.into(),
                    digit_count: 1.into(),
                    preimage_max_coefficient_bound: 8.into(),
                },
            ],
        );
        let automorphism = NodeHandle::new(
            NodeKind::RingAutomorphism { index: 3.into() },
            vec![sample.output(0).unwrap()],
            vec![WireType::Matrix(matrix)],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "automorphism-tracked-trapdoor",
            vec![],
            BTreeMap::from([(
                String::from("out"),
                GraphOutput { value: automorphism, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        assert!(matches!(
            run_single_output(graph, "automorphism-tracked-trapdoor"),
            Err(SimulationError::Relation { .. })
        ));
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
    fn select_relation_join_requires_compatible_relation_on_every_branch() {
        let matrix = ConcreteMatrixType {
            modulus: BigInt::from(17),
            ring_dimension: 1,
            rows: 1,
            columns: 1,
        };
        let branch = |relation: Option<RightPreimage>| Info {
            value: AbstractValue::Matrix(MatrixState {
                error_bound: 2u8.into(),
                coefficient_magnitude_bound: 5u8.into(),
                is_constant_polynomial: true,
                right_carrier: None,
            }),
            ty: Some(matrix.clone()),
            relation,
            view: crate::FamilyViewId(u32::MAX),
            paired_public: None,
        };
        let relation = RightPreimage {
            source: crate::SourceId(7),
            target: crate::FamilyViewId(11),
            view: Some(crate::FamilyViewId(13)),
            selector: None,
        };

        let compatible = join_uniform(
            branch(Some(relation.clone())),
            branch(Some(relation.clone())),
            Some(&matrix),
        )
        .unwrap();
        assert_eq!(compatible.relation, Some(relation.clone()));

        let missing =
            join_uniform(branch(Some(relation.clone())), branch(None), Some(&matrix)).unwrap();
        assert_eq!(missing.relation, None);

        let mut incompatible = relation.clone();
        incompatible.source = crate::SourceId(8);
        let incompatible =
            join_uniform(branch(Some(relation.clone())), branch(Some(incompatible)), Some(&matrix))
                .unwrap();
        assert_eq!(incompatible.relation, None);

        let nested =
            join_uniform(branch(Some(relation.clone())), incompatible, Some(&matrix)).unwrap();
        assert_eq!(nested.relation, None, "a later join cannot revive a dropped relation");

        let selector = state::IntegerState::new(1.into(), 2.into()).unwrap();
        let branches =
            vec![branch(None), branch(Some(relation.clone())), branch(Some(relation.clone()))];
        let reachable = reachable_select_branches(&branches, &selector).unwrap();
        let joined =
            join_uniform(reachable[0].clone(), reachable[1].clone(), Some(&matrix)).unwrap();
        assert_eq!(joined.relation, Some(relation.clone()));

        let branches = vec![branch(None), branch(Some(relation)), branch(None)];
        let reachable = reachable_select_branches(&branches, &selector).unwrap();
        let joined =
            join_uniform(reachable[0].clone(), reachable[1].clone(), Some(&matrix)).unwrap();
        assert_eq!(joined.relation, None);
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
            roots: vec![crate::SimulationRoot { stage: stage.clone(), output: "out".into() }],
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
        env.integers.insert("outer".into(), BigInt::from(9));
        let bindings = [
            ("outer".into(), mxx_ir_core::IntExpr::LoopIndex(3)),
            ("copy".into(), mxx_ir_core::IntExpr::Var("outer".into())),
            ("bound".into(), mxx_ir_core::IntExpr::LoopIndex(3)),
        ];
        let env = apply_bindings(env, &bindings).unwrap();
        assert_eq!(env.integers["bound"], BigInt::from(5));
        assert_eq!(env.integers["outer"], BigInt::from(5));
        assert_eq!(env.integers["copy"], BigInt::from(9));
        let mut abstract_integers =
            BTreeMap::from([("outer".into(), state::IntegerState::singleton(9))]);
        let abstract_loop_indices =
            HashMap::from([(3, state::IntegerState::new(2.into(), 5.into()).unwrap())]);
        apply_abstract_bindings(
            &mut abstract_integers,
            &mut BTreeMap::new(),
            &mut BTreeMap::new(),
            &bindings,
            &env,
            &abstract_loop_indices,
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(
            abstract_integers["outer"],
            state::IntegerState::new(2.into(), 5.into()).unwrap()
        );
        assert_eq!(abstract_integers["copy"], state::IntegerState::singleton(9));
        let index = mxx_ir_core::IndexExpr::Axis(0);
        let loop_indices = HashMap::from([(
            3,
            state::IntegerState::new(BigInt::from(2), BigInt::from(5)).unwrap(),
        )]);
        assert_eq!(
            eval_index_interval(&index, &env, &BTreeMap::new(), &loop_indices, &[3]).unwrap(),
            state::IntegerState::new(BigInt::from(2), BigInt::from(5)).unwrap()
        );
        let truncating_division = mxx_ir_core::IndexExpr::Divide(
            Box::new(mxx_ir_core::IndexExpr::constant(-1)),
            Box::new(mxx_ir_core::IndexExpr::constant(2)),
        );
        assert_eq!(
            eval_index_interval(
                &truncating_division,
                &env,
                &BTreeMap::new(),
                &HashMap::new(),
                &[],
            )
            .unwrap(),
            state::IntegerState::singleton(0),
        );
        assert!(matches!(
            empty_info_for_type(&WireType::Real, &env).unwrap().value,
            AbstractValue::Real
        ));
        let multi_axis = mxx_ir_core::IntExpr::Add(
            Box::new(mxx_ir_core::IntExpr::LoopIndex(0)),
            Box::new(mxx_ir_core::IntExpr::Mul(
                Box::new(mxx_ir_core::IntExpr::constant(2)),
                Box::new(mxx_ir_core::IntExpr::LoopIndex(1)),
            )),
        );
        assert_eq!(
            int_expr_dependencies(&multi_axis, &BTreeMap::new(), &HashMap::from([(0, 7), (1, 8)]),),
            BinderDependencies::Known(BTreeSet::from([7, 8]))
        );
    }

    #[test]
    fn affine_provenance_distinguishes_reused_loop_slot_names() {
        let expression = mxx_ir_core::IntExpr::LoopIndex(0);
        let outer = eval_int_facts(&expression, &BTreeMap::new(), &HashMap::from([(0, 7)]));
        let inner = eval_int_facts(&expression, &BTreeMap::new(), &HashMap::from([(0, 8)]));
        assert_ne!(outer, inner, "lexically distinct binders cannot share provenance");
        assert!(
            affine_fact(outer.as_ref())
                .unwrap()
                .add(&affine_fact(inner.as_ref()).unwrap())
                .is_none(),
            "multi-binder arithmetic must conservatively drop affine provenance",
        );
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
                NodeKind::EvaluateInt(mxx_ir_core::IntExpr::Var("selector".into())),
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
            let scaled = NodeHandle::new(
                NodeKind::MatrixScale {
                    scalar: mxx_ir_core::IntExpr::Add(
                        Box::new(mxx_ir_core::IntExpr::LoopIndex(0)),
                        Box::new(mxx_ir_core::IntExpr::constant(1)),
                    ),
                },
                vec![selected],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("loop-index-select-body", scope, vec![body_noisy], vec![scaled])
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
                bindings: vec![("selector".into(), mxx_ir_core::IntExpr::LoopIndex(0))],
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
        assert_eq!(report.roots[0].maximum_absolute_coefficient_error, 14u8.into());
    }

    #[test]
    fn parallel_grid_symbolic_cost_is_independent_of_cardinality_and_zero_is_empty() {
        fn request(extent: usize) -> SimulationRequest {
            let matrix = MatrixType {
                modulus: mxx_ir_core::IntExpr::constant(97),
                ring_dimension: mxx_ir_core::IntExpr::constant(1),
                rows: mxx_ir_core::IntExpr::constant(1),
                columns: mxx_ir_core::IntExpr::constant(1),
            };
            let body = with_new_construction_scope(|scope| {
                let constants = (0..2)
                    .map(|_| {
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
                    })
                    .collect::<Vec<_>>();
                let sum = NodeHandle::new(
                    NodeKind::MatrixBinary(MatrixBinaryOp::Add),
                    constants,
                    vec![WireType::Matrix(matrix.clone())],
                )
                .output(0)
                .unwrap();
                SubgraphHandle::new("uniform-grid-body", scope, vec![], vec![sum]).unwrap()
            });
            let shape = vec![mxx_ir_core::IntExpr::constant(extent)];
            let output = NodeHandle::parallel_grid(
                body,
                vec![],
                vec![WireType::Family {
                    element: Box::new(WireType::Matrix(matrix)),
                    shape: shape.clone(),
                }],
                mxx_ir_core::node::ParallelGrid {
                    shape,
                    index_slots: vec![0],
                    bindings: vec![],
                    input_modes: vec![],
                },
            )
            .output(0)
            .unwrap();
            let (graph, _) = Graph::freeze(
                format!("uniform-grid-{extent}"),
                vec![],
                BTreeMap::from([(
                    "out".into(),
                    GraphOutput { value: output, confidentiality: None },
                )]),
                vec![],
                vec![],
                BTreeMap::new(),
            )
            .unwrap();
            let environment = ParamEnv::default();
            let stage = crate::StageId(format!("uniform-grid-{extent}"));
            SimulationRequest {
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
            }
        }

        let small = run(&request(1)).unwrap();
        let large = run(&request(100_000)).unwrap();
        assert_eq!(small.diagnostics.transfer_steps, large.diagnostics.transfer_steps);
        assert_eq!(large.roots[0].maximum_absolute_coefficient_error, BigUint::zero());

        let empty = run(&request(0)).unwrap();
        assert_eq!(empty.roots[0].maximum_absolute_coefficient_error, BigUint::zero());
        assert!(empty.diagnostics.transfer_steps < small.diagnostics.transfer_steps);
    }

    #[test]
    fn parallel_grid_refines_diamond_witness_selector_before_dynamic_gather() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(97),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let family_type = WireType::Family {
            element: Box::new(WireType::Matrix(matrix.clone())),
            shape: vec![mxx_ir_core::IntExpr::constant(3)],
        };
        let packed = NodeHandle::new(
            NodeKind::Input {
                name: "witnesses".into(),
                wire_type: family_type.clone(),
                artifact: None,
            },
            vec![],
            vec![family_type.clone()],
        )
        .output(0)
        .unwrap();
        let body = with_new_construction_scope(|scope| {
            let witnesses = NodeHandle::new(
                NodeKind::Input {
                    name: "witness-family".into(),
                    wire_type: family_type.clone(),
                    artifact: None,
                },
                vec![],
                vec![family_type.clone()],
            )
            .output(0)
            .unwrap();
            let integer = |kind, inputs, output_type| {
                NodeHandle::new(kind, inputs, vec![output_type]).output(0).unwrap()
            };
            let slot = integer(
                NodeKind::EvaluateInt(mxx_ir_core::IntExpr::LoopIndex(0)),
                vec![],
                WireType::ConstantInt,
            );
            let witness_end =
                integer(NodeKind::ConstantInt(4.into()), vec![], WireType::ConstantInt);
            let instance_width =
                integer(NodeKind::ConstantInt(2.into()), vec![], WireType::ConstantInt);
            let after_instance = integer(
                NodeKind::IntCompare(mxx_ir_core::node::IntCompareOp::LessEqual),
                vec![instance_width.clone(), slot.clone()],
                WireType::Bool,
            );
            let before_end = integer(
                NodeKind::IntCompare(mxx_ir_core::node::IntCompareOp::LessEqual),
                vec![slot.clone(), witness_end],
                WireType::Bool,
            );
            let after_instance = integer(NodeKind::BoolToInt, vec![after_instance], WireType::Int);
            let before_end = integer(NodeKind::BoolToInt, vec![before_end], WireType::Int);
            let witness_active = integer(
                NodeKind::IntBinary(IntBinaryOp::Multiply),
                vec![after_instance, before_end],
                WireType::Int,
            );
            // Diamond chooses a nonnegative base before subtracting the
            // instance width: active witness lanes select `slot`, and padded
            // lanes select `instance_width`.  Refining the selected affine
            // branch under `witness_active == 1` proves a final range 0..2.
            let zero = integer(NodeKind::ConstantInt(0.into()), vec![], WireType::ConstantInt);
            let inactive_base = integer(
                NodeKind::IntBinary(IntBinaryOp::Add),
                vec![instance_width.clone(), zero.clone()],
                WireType::Int,
            );
            let active_base =
                integer(NodeKind::IntBinary(IntBinaryOp::Add), vec![slot, zero], WireType::Int);
            let selected_base = integer(
                NodeKind::Select { count: mxx_ir_core::IntExpr::constant(2) },
                vec![witness_active, inactive_base, active_base],
                WireType::Int,
            );
            let witness_index = integer(
                NodeKind::IntBinary(IntBinaryOp::Subtract),
                vec![selected_base, instance_width],
                WireType::Int,
            );
            let selected = NodeHandle::new(
                NodeKind::FamilyGetDynamic { rank: 1 },
                vec![witnesses.clone(), witness_index],
                vec![WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("diamond-witness-selector", scope, vec![witnesses], vec![selected])
                .unwrap()
        });
        let output = NodeHandle::parallel_grid(
            body,
            vec![packed],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(matrix)),
                shape: vec![mxx_ir_core::IntExpr::constant(7)],
            }],
            mxx_ir_core::node::ParallelGrid {
                shape: vec![mxx_ir_core::IntExpr::constant(7)],
                index_slots: vec![0],
                bindings: vec![],
                input_modes: vec![mxx_ir_core::node::GridInputMode::Broadcast],
            },
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "diamond-witness-selector",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: output, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = ParamEnv::default();
        let stage = crate::StageId("diamond-witness-selector".into());
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
                stage: stage.clone(),
                input: "witnesses".into(),
                value: crate::ExternalInputValue::Family {
                    shape: vec![3],
                    element: Box::new(crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: 0u8.into(),
                        maximum_absolute_coefficient_value: Some(0u8.into()),
                        is_constant_polynomial: true,
                    }),
                },
            }],
            limits: crate::SimulationLimits::default(),
        })
        .unwrap();
        assert!(report.roots[0].maximum_absolute_coefficient_error.is_zero());
    }

    #[test]
    fn parallel_grid_rejects_nested_nonuniform_matrix_dimensions() {
        let varying = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(97),
            ring_dimension: mxx_ir_core::IntExpr::constant(1),
            rows: mxx_ir_core::IntExpr::Var("nested_rows".into()),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let uniform = MatrixType { rows: mxx_ir_core::IntExpr::constant(1), ..varying.clone() };
        let grandchild = with_new_construction_scope(|scope| {
            let intermediate = NodeHandle::new(
                NodeKind::ConstantMatrix {
                    matrix_type: varying.clone(),
                    value: ConstantMatrix::Zero,
                },
                vec![],
                vec![WireType::Matrix(varying.clone())],
            )
            .output(0)
            .unwrap();
            let output = NodeHandle::new(
                NodeKind::Slice {
                    rows: Some(IndexRange {
                        start: mxx_ir_core::IntExpr::constant(0),
                        end: mxx_ir_core::IntExpr::constant(1),
                    }),
                    columns: None,
                },
                vec![intermediate],
                vec![WireType::Matrix(uniform.clone())],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("nonuniform-grandchild", scope, vec![], vec![output]).unwrap()
        });
        let body = with_new_construction_scope(|scope| {
            let output = NodeHandle::subgraph_call(
                grandchild,
                vec![],
                vec![("nested_rows".into(), mxx_ir_core::IntExpr::Var("lane_rows".into()))],
                vec![],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("uniform-grid-caller", scope, vec![], vec![output]).unwrap()
        });
        let shape = vec![mxx_ir_core::IntExpr::constant(2)];
        let output = NodeHandle::parallel_grid(
            body,
            vec![],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(uniform)),
                shape: shape.clone(),
            }],
            mxx_ir_core::node::ParallelGrid {
                shape,
                index_slots: vec![4],
                bindings: vec![(
                    "lane_rows".into(),
                    mxx_ir_core::IntExpr::Add(
                        Box::new(mxx_ir_core::IntExpr::LoopIndex(4)),
                        Box::new(mxx_ir_core::IntExpr::constant(1)),
                    ),
                )],
                input_modes: vec![],
            },
        )
        .output(0)
        .unwrap();
        let (graph, _) = Graph::freeze(
            "nested-nonuniform-grid",
            vec![],
            BTreeMap::from([("out".into(), GraphOutput { value: output, confidentiality: None })]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let stage = crate::StageId("nested-nonuniform-grid".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram { stages: vec![] },
            environment: ParamEnv::default(),
            roots: vec![],
            external_inputs: vec![],
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
            abstract_integers: BTreeMap::from([(
                "nested_rows".into(),
                state::IntegerState::new(1.into(), 2.into()).unwrap(),
            )]),
            abstract_integer_facts: BTreeMap::new(),
            abstract_integer_dependencies: BTreeMap::new(),
            abstract_loop_indices: HashMap::new(),
            abstract_loop_atoms: HashMap::new(),
            binder_ranges: HashMap::new(),
            next_binder_atom: 0,
            scalar_facts: HashMap::new(),
            scalar_dependencies: HashMap::new(),
            family_axis_dependencies: HashMap::new(),
            next_source: 0,
            preimages: HashMap::new(),
            states: HashMap::new(),
            selector_views: HashMap::new(),
            next_selector: 0,
            planned: 0,
            transfers: 0,
            dropped: vec![],
            interners: crate::identity::Interners::default(),
            reached: HashSet::new(),
            artifact_outputs: BTreeMap::new(),
        };
        let body_scope =
            graph.child_scope_id(&FrozenGraphScopeId::Root, mxx_ir_core::NodeId(0)).unwrap();
        let nested_scope = graph.child_scope_id(&body_scope, mxx_ir_core::NodeId(0)).unwrap();
        let mut environment = ParamEnv::default();
        environment.integers.insert("nested_rows".into(), BigInt::one());
        let error = match evaluator.scope(
            &stage,
            &graph,
            &nested_scope,
            &["nested".into()],
            environment,
            HashMap::new(),
        ) {
            Err(error) => error,
            Ok(_) => panic!("nested nonuniform matrix type must be rejected"),
        };
        assert!(
            error.to_string().contains("matrix row count must be uniform"),
            "unexpected error: {error}",
        );
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
            abstract_integers: BTreeMap::new(),
            abstract_integer_facts: BTreeMap::new(),
            abstract_integer_dependencies: BTreeMap::new(),
            abstract_loop_indices: HashMap::new(),
            abstract_loop_atoms: HashMap::new(),
            binder_ranges: HashMap::new(),
            next_binder_atom: 0,
            scalar_facts: HashMap::new(),
            scalar_dependencies: HashMap::new(),
            family_axis_dependencies: HashMap::new(),
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

        let other_primitive = evaluator.source_for(
            &crate::StageId("source-lineage".into()),
            &FrozenGraphScopeId::Root,
            &[],
            1,
            "public",
        );
        let packed_distinct = evaluator.group_source_for(vec![primitive, other_primitive], vec![2]);
        assert_eq!(
            evaluator.lift_source_for_shape(packed_distinct, vec![2]),
            packed_distinct,
            "a normalized family source must survive grid reuse without fresh lane identities",
        );

        // The public gadget relation is index-independent: every grid lane
        // consumes a preimage of the same G rather than sampling a new source.
        let gadget = evaluator.source_for(
            &crate::StageId("source-lineage".into()),
            &FrozenGraphScopeId::Root,
            &[],
            2,
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
        let opaque_map =
            mxx_ir_core::IndexMap::new([mxx_ir_core::IndexExpr::Parameter("opaque-index".into())]);
        let opaque_mapped = evaluator.mapped_source_for(flat, &opaque_map, vec![4], None);
        assert_eq!(
            evaluator.mapped_source_for(flat, &opaque_map, vec![4], None),
            opaque_mapped,
            "one normalized opaque map keeps one canonical source identity",
        );
        let opaque_mapped_lineage = evaluator.source_lineages.get(&opaque_mapped).unwrap();
        assert_eq!(opaque_mapped_lineage.shape, vec![4]);
        assert!(!lineage_is_complete(opaque_mapped_lineage));
        assert_eq!(
            evaluator.lift_source_for_shape(opaque_mapped, vec![4]),
            opaque_mapped,
            "lifting an opaque map over its existing domain must preserve its identity",
        );
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

        // An opaque final-axis selection of a pointwise [group, branch]
        // source retains the group domain without claiming that its lanes are
        // equal.  The same rule applies to a target's independent carrier.
        let opaque_pointwise = evaluator.source_after_axis_selection(
            bases,
            false,
            1,
            2,
            vec![crate::SelectorId(9)],
            vec![3],
        );
        let opaque_lineage = evaluator.source_lineages.get(&opaque_pointwise).unwrap();
        assert_eq!(opaque_lineage.shape, vec![3]);
        assert!(!lineage_is_complete(opaque_lineage));
        let opaque_target =
            evaluator.interners.intern_view(vec![crate::ValueId(89)], vec![3, 2], &[]);
        evaluator.states.insert(
            opaque_target,
            MatrixState::new(BigUint::ZERO, 1u8.into(), false)
                .unwrap()
                .with_carrier(bases, 1u8.into()),
        );
        let selected_target = evaluator
            .remap_target_after_axis_selection(
                opaque_target,
                1,
                vec![crate::SelectorId(9)],
                vec![3],
            )
            .unwrap();
        let target_carrier = selected_target.right_carrier.unwrap();
        let target_lineage = evaluator.source_lineages.get(&target_carrier.source).unwrap();
        assert_eq!(target_lineage.shape, vec![3]);
        assert!(!lineage_is_complete(target_lineage));

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

        let relation_view =
            evaluator.interners.intern_view(vec![crate::ValueId(90)], vec![2, 2], &[]);
        let target_view =
            evaluator.interners.intern_view(vec![crate::ValueId(91)], vec![2, 2], &[]);
        let gap_relation = RightPreimage {
            source: primitive,
            target: target_view,
            view: Some(relation_view),
            selector: None,
        };
        assert!(matches!(
            evaluator.relation_source_projection(&gap_relation, &[2, 2], None),
            Err(SimulationError::Relation { .. })
        ));

        let equal_rank_extent_mismatch = RightPreimage { source: bases, ..gap_relation.clone() };
        assert!(matches!(
            evaluator.relation_source_projection(&equal_rank_extent_mismatch, &[2, 2], None,),
            Err(SimulationError::Relation { .. })
        ));
        let mismatched_group = evaluator.group_source_for(vec![primitive], vec![3]);
        let shared_extent_mismatch =
            RightPreimage { source: mismatched_group, ..gap_relation.clone() };
        assert!(matches!(
            evaluator.relation_source_projection(&shared_extent_mismatch, &[2, 2], None),
            Err(SimulationError::Relation { .. })
        ));

        let scalar_relation_view =
            evaluator.interners.intern_view(vec![crate::ValueId(92)], vec![2], &[]);
        let scalar_target_view =
            evaluator.interners.intern_view(vec![crate::ValueId(93)], vec![2], &[]);
        let scalar_shared_relation = RightPreimage {
            source: primitive,
            target: scalar_target_view,
            view: Some(scalar_relation_view),
            selector: None,
        };
        assert_eq!(
            evaluator.relation_source_projection(&scalar_shared_relation, &[2], None).unwrap(),
            RelationSourceProjection { relation_rank: 1, source_rank: 0 }
        );

        let shared_source = evaluator.group_source_for(vec![primitive], vec![2]);
        let shared_relation = RightPreimage { source: shared_source, ..gap_relation };
        let family_selector = |range: state::IntegerState, view| Info {
            value: AbstractValue::Family(
                FamilyState::new(vec![2, 3], AbstractValue::Integer(range)).unwrap(),
            ),
            ty: None,
            relation: None,
            view,
            paired_public: None,
        };
        let branch_dependent_view =
            evaluator.interners.intern_view(vec![crate::ValueId(94)], vec![2, 3], &[]);
        evaluator.family_axis_dependencies.insert(branch_dependent_view, BTreeSet::from([1]));
        let varying_selector = family_selector(
            state::IntegerState::new(0.into(), 1.into()).unwrap(),
            branch_dependent_view,
        );
        assert!(matches!(
            evaluator.gathered_relation_source_projection(
                &shared_relation,
                &[2, 2],
                &[2, 3],
                &[
                    varying_selector,
                    family_selector(
                        state::IntegerState::singleton(0),
                        crate::FamilyViewId(u32::MAX),
                    ),
                ],
                &[crate::SelectorId(30), crate::SelectorId(31)],
                None,
            ),
            Err(SimulationError::BranchDependentSource { .. })
        ));
        let group_dependent_view =
            evaluator.interners.intern_view(vec![crate::ValueId(95)], vec![2, 3], &[]);
        evaluator.family_axis_dependencies.insert(group_dependent_view, BTreeSet::from([0]));
        let (source_selectors, source_shape) = evaluator
            .gathered_relation_source_projection(
                &shared_relation,
                &[2, 2],
                &[2, 3],
                &[
                    family_selector(
                        state::IntegerState::new(0.into(), 1.into()).unwrap(),
                        group_dependent_view,
                    ),
                    family_selector(
                        state::IntegerState::singleton(0),
                        crate::FamilyViewId(u32::MAX),
                    ),
                ],
                &[crate::SelectorId(30), crate::SelectorId(31)],
                None,
            )
            .unwrap();
        assert_eq!(source_selectors, vec![crate::SelectorId(30)]);
        assert_eq!(source_shape, vec![2]);
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
            vec![trapdoor_wire.clone()],
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
        let carrier_target = NodeHandle::new(
            NodeKind::Input {
                name: "carrier-target".into(),
                wire_type: WireType::Matrix(target_type.clone()),
                artifact: None,
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
                family_preimage.clone(),
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
        let family_static_selected = NodeHandle::new(
            NodeKind::FamilyGetStatic { indices: vec![mxx_ir_core::IndexExpr::constant(1)] },
            vec![family_preimage.clone()],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let family_static_applied = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![public.clone(), family_static_selected],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let scalar_selector = || {
            NodeHandle::new(NodeKind::ConstantInt(0.into()), vec![], vec![WireType::ConstantInt])
                .output(0)
                .unwrap()
        };
        let branch_selector_family = NodeHandle::new(
            NodeKind::FamilyPack { shape: vec![mxx_ir_core::IntExpr::constant(2)] },
            vec![scalar_selector(), scalar_selector()],
            vec![WireType::Family {
                element: Box::new(WireType::ConstantInt),
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let gathered_preimages = NodeHandle::new(
            NodeKind::FamilyGather {
                output_shape: vec![mxx_ir_core::IntExpr::constant(2)],
                input_rank: 1,
            },
            vec![family_preimage.clone(), branch_selector_family],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let gathered_preimage = NodeHandle::new(
            NodeKind::FamilySelectAxis { axis: 0 },
            vec![gathered_preimages, scalar_selector()],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let gathered_applied = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![public.clone(), gathered_preimage],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let group_shape = vec![mxx_ir_core::IntExpr::constant(2)];
        let grouped_public = NodeHandle::new(
            NodeKind::FamilyPack { shape: group_shape.clone() },
            vec![public.clone(), public.clone()],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(public_type.clone())),
                shape: group_shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let grouped_trapdoor = NodeHandle::new(
            NodeKind::FamilyPack { shape: group_shape.clone() },
            vec![trapdoor.clone(), trapdoor.clone()],
            vec![WireType::Family {
                element: Box::new(trapdoor_wire.clone()),
                shape: group_shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let relation_shape =
            vec![mxx_ir_core::IntExpr::constant(2), mxx_ir_core::IntExpr::constant(2)];
        let grouped_target = NodeHandle::new(
            NodeKind::FamilyPack { shape: relation_shape.clone() },
            vec![target.clone(), target.clone(), target.clone(), target.clone()],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(target_type.clone())),
                shape: relation_shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let grouped_preimage = NodeHandle::new(
            NodeKind::FamilyPreimageSample {
                matrix_type: output_type.clone(),
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
            },
            vec![grouped_public.clone(), grouped_trapdoor, grouped_target],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: relation_shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let selector_body = with_new_construction_scope(|scope| {
            let group = NodeHandle::new(
                NodeKind::EvaluateInt(mxx_ir_core::IntExpr::LoopIndex(0)),
                vec![],
                vec![WireType::ConstantInt],
            )
            .output(0)
            .unwrap();
            let branch = NodeHandle::new(
                NodeKind::EvaluateInt(mxx_ir_core::IntExpr::LoopIndex(1)),
                vec![],
                vec![WireType::ConstantInt],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new("gather-selector-body", scope, vec![], vec![group, branch]).unwrap()
        });
        let selector_families = NodeHandle::parallel_grid(
            selector_body,
            vec![],
            vec![
                WireType::Family {
                    element: Box::new(WireType::ConstantInt),
                    shape: relation_shape.clone(),
                },
                WireType::Family {
                    element: Box::new(WireType::ConstantInt),
                    shape: relation_shape.clone(),
                },
            ],
            mxx_ir_core::node::ParallelGrid {
                shape: relation_shape.clone(),
                index_slots: vec![0, 1],
                bindings: vec![],
                input_modes: vec![],
            },
        );
        let varying_gather = NodeHandle::new(
            NodeKind::FamilyGather { output_shape: relation_shape.clone(), input_rank: 2 },
            vec![
                grouped_preimage,
                selector_families.output(0).unwrap(),
                selector_families.output(1).unwrap(),
            ],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: relation_shape,
            }],
        )
        .output(0)
        .unwrap();
        let varying_gather_group = NodeHandle::new(
            NodeKind::FamilySelectAxis { axis: 1 },
            vec![varying_gather, scalar_selector()],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: group_shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let grouped_public_element = NodeHandle::new(
            NodeKind::FamilySelectAxis { axis: 0 },
            vec![grouped_public, scalar_selector()],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let varying_gather_element = NodeHandle::new(
            NodeKind::FamilySelectAxis { axis: 0 },
            vec![varying_gather_group, scalar_selector()],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let varying_gather_applied = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![grouped_public_element, varying_gather_element],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let preimage = NodeHandle::new(
            NodeKind::PreimageSample {
                matrix_type: output_type.clone(),
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
            },
            vec![public.clone(), trapdoor.clone(), target.clone()],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let packed_second_target = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: target_type.clone(),
                value: ConstantMatrix::Zero,
            },
            vec![],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let packed_second_preimage = NodeHandle::new(
            NodeKind::PreimageSample {
                matrix_type: output_type.clone(),
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
            },
            vec![public.clone(), trapdoor.clone(), packed_second_target],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let packed_shape = vec![mxx_ir_core::IntExpr::constant(2)];
        let packed_public = NodeHandle::new(
            NodeKind::FamilyPack { shape: packed_shape.clone() },
            vec![public.clone(), public.clone()],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(public_type.clone())),
                shape: packed_shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let packed_preimages = NodeHandle::new(
            NodeKind::FamilyPack { shape: packed_shape.clone() },
            vec![preimage.clone(), packed_second_preimage.clone()],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: packed_shape,
            }],
        )
        .output(0)
        .unwrap();
        let rank_two_packed_preimages = NodeHandle::new(
            NodeKind::FamilyPack {
                shape: vec![mxx_ir_core::IntExpr::constant(2), mxx_ir_core::IntExpr::constant(2)],
            },
            vec![
                preimage.clone(),
                packed_second_preimage.clone(),
                preimage.clone(),
                packed_second_preimage,
            ],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: vec![mxx_ir_core::IntExpr::constant(2), mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let branch_dependent_reindex = NodeHandle::new(
            NodeKind::FamilyReindex {
                output_shape: vec![
                    mxx_ir_core::IntExpr::constant(2),
                    mxx_ir_core::IntExpr::constant(2),
                ],
                map: mxx_ir_core::IndexMap::new([
                    mxx_ir_core::IndexExpr::Axis(1),
                    mxx_ir_core::IndexExpr::Axis(1),
                ]),
            },
            vec![rank_two_packed_preimages],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: vec![mxx_ir_core::IntExpr::constant(2), mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let packed_static_public = NodeHandle::new(
            NodeKind::FamilyGetStatic { indices: vec![mxx_ir_core::IndexExpr::constant(1)] },
            vec![packed_public.clone()],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let packed_static_preimage = NodeHandle::new(
            NodeKind::FamilyGetStatic { indices: vec![mxx_ir_core::IndexExpr::constant(1)] },
            vec![packed_preimages.clone()],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let packed_static_applied = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![packed_static_public, packed_static_preimage],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let packed_selector =
            NodeHandle::new(NodeKind::ConstantInt(0.into()), vec![], vec![WireType::ConstantInt])
                .output(0)
                .unwrap();
        let packed_dynamic_public = NodeHandle::new(
            NodeKind::FamilyGetDynamic { rank: 1 },
            vec![packed_public, packed_selector.clone()],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let packed_dynamic_preimage = NodeHandle::new(
            NodeKind::FamilyGetDynamic { rank: 1 },
            vec![packed_preimages, packed_selector],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let packed_dynamic_applied = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![packed_dynamic_public, packed_dynamic_preimage],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let unrelated_preimage = NodeHandle::new(
            NodeKind::Input {
                name: "unrelated-preimage".into(),
                wire_type: WireType::Preimage(output_type.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let incompatible_packed_preimages = NodeHandle::new(
            NodeKind::FamilyPack { shape: vec![mxx_ir_core::IntExpr::constant(2)] },
            vec![preimage.clone(), unrelated_preimage],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let second_public = NodeHandle::new(
            NodeKind::Input {
                name: "second-public".into(),
                wire_type: WireType::Matrix(public_type.clone()),
                artifact: None,
            },
            vec![],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let second_trapdoor = NodeHandle::new(
            NodeKind::Input {
                name: "second-trapdoor".into(),
                wire_type: trapdoor_wire.clone(),
                artifact: None,
            },
            vec![],
            vec![trapdoor_wire.clone()],
        )
        .output(0)
        .unwrap();
        let second_source_preimage = NodeHandle::new(
            NodeKind::PreimageSample {
                matrix_type: output_type.clone(),
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
            },
            vec![second_public, second_trapdoor, target.clone()],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let distinct_group_packed_preimages = NodeHandle::new(
            NodeKind::FamilyPack {
                shape: vec![mxx_ir_core::IntExpr::constant(2), mxx_ir_core::IntExpr::constant(2)],
            },
            vec![
                preimage.clone(),
                preimage.clone(),
                second_source_preimage.clone(),
                second_source_preimage.clone(),
            ],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: vec![mxx_ir_core::IntExpr::constant(2), mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let mixed_source_packed_preimages = NodeHandle::new(
            NodeKind::FamilyPack { shape: vec![mxx_ir_core::IntExpr::constant(2)] },
            vec![preimage.clone(), second_source_preimage],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: vec![mxx_ir_core::IntExpr::constant(2)],
            }],
        )
        .output(0)
        .unwrap();
        let carrier_target_preimage = NodeHandle::new(
            NodeKind::PreimageSample {
                matrix_type: output_type.clone(),
                max_coefficient_bound: mxx_ir_core::IntExpr::constant(4),
            },
            vec![public.clone(), trapdoor, carrier_target],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let pointwise_body = with_new_construction_scope(|scope| {
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
            let body_preimage = NodeHandle::new(
                NodeKind::Input {
                    name: "body-preimage".into(),
                    wire_type: WireType::Preimage(output_type.clone()),
                    artifact: None,
                },
                vec![],
                vec![WireType::Preimage(output_type.clone())],
            )
            .output(0)
            .unwrap();
            SubgraphHandle::new(
                "pointwise-preimage-body",
                scope,
                vec![body_public.clone(), body_preimage.clone()],
                vec![body_public, body_preimage],
            )
            .unwrap()
        });
        let pointwise_shape =
            vec![mxx_ir_core::IntExpr::constant(2), mxx_ir_core::IntExpr::constant(2)];
        let pointwise = NodeHandle::parallel_grid(
            pointwise_body,
            vec![public.clone(), carrier_target_preimage],
            vec![
                WireType::Family {
                    element: Box::new(WireType::Matrix(public_type.clone())),
                    shape: pointwise_shape.clone(),
                },
                WireType::Family {
                    element: Box::new(WireType::Preimage(output_type.clone())),
                    shape: pointwise_shape.clone(),
                },
            ],
            mxx_ir_core::node::ParallelGrid {
                shape: pointwise_shape.clone(),
                index_slots: vec![0, 1],
                bindings: vec![],
                input_modes: vec![
                    mxx_ir_core::node::GridInputMode::Broadcast,
                    mxx_ir_core::node::GridInputMode::Broadcast,
                ],
            },
        );
        let identity_map = mxx_ir_core::IndexMap::new([
            mxx_ir_core::IndexExpr::Axis(0),
            mxx_ir_core::IndexExpr::Axis(1),
        ]);
        let pointwise_public = NodeHandle::new(
            NodeKind::FamilyReindex {
                output_shape: pointwise_shape.clone(),
                map: identity_map.clone(),
            },
            vec![pointwise.output(0).unwrap()],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(public_type.clone())),
                shape: pointwise_shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let pointwise_preimage = NodeHandle::new(
            NodeKind::FamilyReindex { output_shape: pointwise_shape.clone(), map: identity_map },
            vec![pointwise.output(1).unwrap()],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: pointwise_shape,
            }],
        )
        .output(0)
        .unwrap();
        let pointwise_selector = scalar_selector();
        let selected_pointwise_public = NodeHandle::new(
            NodeKind::FamilySelectAxis { axis: 1 },
            vec![pointwise_public.clone(), pointwise_selector.clone()],
            vec![WireType::Family {
                element: Box::new(WireType::Matrix(public_type.clone())),
                shape: group_shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let selected_pointwise_preimage = NodeHandle::new(
            NodeKind::FamilySelectAxis { axis: 1 },
            vec![pointwise_preimage.clone(), pointwise_selector],
            vec![WireType::Family {
                element: Box::new(WireType::Preimage(output_type.clone())),
                shape: group_shape.clone(),
            }],
        )
        .output(0)
        .unwrap();
        let pointwise_group_selector = scalar_selector();
        let selected_pointwise_public = NodeHandle::new(
            NodeKind::FamilySelectAxis { axis: 0 },
            vec![selected_pointwise_public, pointwise_group_selector.clone()],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let selected_pointwise_preimage = NodeHandle::new(
            NodeKind::FamilySelectAxis { axis: 0 },
            vec![selected_pointwise_preimage, pointwise_group_selector],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let pointwise_selected_applied = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![selected_pointwise_public, selected_pointwise_preimage],
            vec![WireType::Matrix(target_type.clone())],
        )
        .output(0)
        .unwrap();
        let static_index =
            vec![mxx_ir_core::IndexExpr::constant(0), mxx_ir_core::IndexExpr::constant(0)];
        let pointwise_public = NodeHandle::new(
            NodeKind::FamilyGetStatic { indices: static_index.clone() },
            vec![pointwise_public],
            vec![WireType::Matrix(public_type.clone())],
        )
        .output(0)
        .unwrap();
        let pointwise_preimage = NodeHandle::new(
            NodeKind::FamilyGetStatic { indices: static_index },
            vec![pointwise_preimage],
            vec![WireType::Preimage(output_type.clone())],
        )
        .output(0)
        .unwrap();
        let pointwise_applied = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![pointwise_public, pointwise_preimage],
            vec![WireType::Matrix(target_type.clone())],
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
                (
                    "family-static".into(),
                    GraphOutput { value: family_static_applied, confidentiality: None },
                ),
                (
                    "pointwise-reindex".into(),
                    GraphOutput { value: pointwise_applied, confidentiality: None },
                ),
                (
                    "pointwise-select".into(),
                    GraphOutput { value: pointwise_selected_applied, confidentiality: None },
                ),
                (
                    "shared-gather".into(),
                    GraphOutput { value: gathered_applied, confidentiality: None },
                ),
                (
                    "varying-shared-gather".into(),
                    GraphOutput { value: varying_gather_applied, confidentiality: None },
                ),
                (
                    "packed-static".into(),
                    GraphOutput { value: packed_static_applied, confidentiality: None },
                ),
                (
                    "packed-dynamic".into(),
                    GraphOutput { value: packed_dynamic_applied, confidentiality: None },
                ),
                (
                    "packed-incompatible".into(),
                    GraphOutput { value: incompatible_packed_preimages, confidentiality: None },
                ),
                (
                    "packed-shared-classification".into(),
                    GraphOutput { value: branch_dependent_reindex, confidentiality: None },
                ),
                (
                    "packed-mixed-source".into(),
                    GraphOutput { value: mixed_source_packed_preimages, confidentiality: None },
                ),
                (
                    "packed-distinct-groups".into(),
                    GraphOutput { value: distinct_group_packed_preimages, confidentiality: None },
                ),
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
                crate::SimulationRoot { stage: stage_id.clone(), output: "family-static".into() },
                crate::SimulationRoot {
                    stage: stage_id.clone(),
                    output: "pointwise-reindex".into(),
                },
                crate::SimulationRoot { stage: stage_id.clone(), output: "shared-gather".into() },
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
                crate::ExternalInputFact {
                    stage: stage_id.clone(),
                    input: "second-public".into(),
                    value: crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: BigUint::ZERO,
                        maximum_absolute_coefficient_value: None,
                        is_constant_polynomial: false,
                    },
                },
                crate::ExternalInputFact {
                    stage: stage_id.clone(),
                    input: "second-trapdoor".into(),
                    value: crate::ExternalInputValue::Trapdoor {
                        public_matrix_input: "second-public".into(),
                    },
                },
                crate::ExternalInputFact {
                    stage: stage_id.clone(),
                    input: "carrier-target".into(),
                    value: crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: BigUint::ZERO,
                        maximum_absolute_coefficient_value: None,
                        is_constant_polynomial: false,
                    },
                },
                crate::ExternalInputFact {
                    stage: stage_id.clone(),
                    input: "unrelated-preimage".into(),
                    value: crate::ExternalInputValue::Matrix {
                        maximum_absolute_coefficient_error: BigUint::ZERO,
                        maximum_absolute_coefficient_value: None,
                        is_constant_polynomial: false,
                    },
                },
            ],
            limits: crate::SimulationLimits::default(),
        };
        for output in [
            "family",
            "family-static",
            "pointwise-reindex",
            "pointwise-select",
            "shared-gather",
            "varying-shared-gather",
            "packed-static",
            "packed-dynamic",
            "packed-distinct-groups",
        ] {
            let mut focused = request.clone();
            focused.roots =
                vec![crate::SimulationRoot { stage: stage_id.clone(), output: output.into() }];
            let result = run(&focused);
            assert!(result.is_ok(), "{output} failed relation projection: {result:?}");
        }
        let mut incompatible = request.clone();
        incompatible.roots = vec![crate::SimulationRoot {
            stage: stage_id.clone(),
            output: "packed-incompatible".into(),
        }];
        let error =
            run(&incompatible).expect_err("a relationless packed preimage must fail closed");
        assert!(error.to_string().contains("every packed preimage must carry a relation"));
        let mut shared = request.clone();
        shared.roots = vec![crate::SimulationRoot {
            stage: stage_id.clone(),
            output: "packed-shared-classification".into(),
        }];
        let error = run(&shared).expect_err("packed relation must remain shared-source");
        assert!(error.to_string().contains("shared-source relation depends on its branch axis"));
        let mut mixed = request.clone();
        mixed.roots = vec![crate::SimulationRoot {
            stage: stage_id.clone(),
            output: "packed-mixed-source".into(),
        }];
        let error = run(&mixed).expect_err("mixed sources inside one branch family must fail");
        assert!(
            error
                .to_string()
                .contains("packed preimage branches within one group must share one source")
        );
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
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
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
            vec![WireType::Matrix(gadget_type.clone())],
        )
        .output(0)
        .unwrap();
        let automorphed_gadget = NodeHandle::new(
            NodeKind::RingAutomorphism { index: mxx_ir_core::IntExpr::constant(3) },
            vec![gadget],
            vec![WireType::Matrix(gadget_type)],
        )
        .output(0)
        .unwrap();
        let product = NodeHandle::new(
            NodeKind::ApplyPreimage,
            vec![automorphed_gadget, decomposition],
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
    fn invalid_cross_stage_artifact_is_rejected_before_planning() {
        let producer_matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let consumer_matrix =
            MatrixType { modulus: mxx_ir_core::IntExpr::constant(19), ..producer_matrix.clone() };
        let produced = NodeHandle::new(
            NodeKind::ConstantMatrix {
                matrix_type: producer_matrix.clone(),
                value: ConstantMatrix::Zero,
            },
            vec![],
            vec![WireType::Matrix(producer_matrix)],
        )
        .output(0)
        .unwrap();
        let (producer_graph, _) = Graph::freeze(
            "invalid-artifact-producer",
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
            execution_nonce: [5; 32],
        };
        let consumed = NodeHandle::new(
            NodeKind::Input {
                name: "consumed".into(),
                wire_type: WireType::Matrix(consumer_matrix.clone()),
                artifact: Some(ArtifactInput {
                    production_id: producer_id.clone(),
                    artifact_name: "artifact".into(),
                    confidentiality: ArtifactConfidentiality::Public,
                }),
            },
            vec![],
            vec![WireType::Matrix(consumer_matrix)],
        )
        .output(0)
        .unwrap();
        let (consumer_graph, _) = Graph::freeze(
            "invalid-artifact-consumer",
            vec![],
            BTreeMap::from([(
                "out".into(),
                GraphOutput { value: consumed, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let consumer = crate::StageId("invalid-consumer".into());
        let request = SimulationRequest {
            program: crate::SimulationProgram {
                stages: vec![
                    crate::SimulationStage {
                        id: consumer.clone(),
                        production_id: ProductionId {
                            spec_hash: spec_hash(&consumer_graph, &environment).unwrap(),
                            execution_nonce: [6; 32],
                        },
                        graph: consumer_graph,
                    },
                    crate::SimulationStage {
                        id: crate::StageId("invalid-producer".into()),
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

        let result = run(&request);
        assert!(matches!(
            result,
            Err(SimulationError::InvalidGraph { message, .. })
                if message.contains("artifact type does not match manifest")
        ));
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
