//! Occurrence-aware reachability and identity keys.

use crate::{SimulationError, SimulationProgram, SimulationRequest, StageId};
use mxx_ir_core::{FrozenGraphScopeId, Graph, WireRef};
use num_traits::ToPrimitive;
use std::collections::{BTreeMap, BTreeSet, HashMap};

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) struct PlannedWire {
    pub stage: StageId,
    pub scope: FrozenGraphScopeId,
    pub occurrence: Vec<String>,
    pub wire: WireRef,
}

#[derive(Default)]
pub(crate) struct Plan {
    pub wires: Vec<PlannedWire>,
    pub by_key: HashMap<PlannedWire, crate::ValueId>,
    pub interners: crate::identity::Interners,
    pub artifact_outputs: BTreeMap<StageId, BTreeSet<String>>,
}

impl Plan {
    pub(crate) fn build(request: &SimulationRequest) -> Result<Self, SimulationError> {
        let mut plan = Self::default();
        let mut seen = BTreeSet::new();
        for root in &request.roots {
            let stage = request
                .program
                .stage(&root.stage)
                .ok_or_else(|| SimulationError::UnknownStage { stage: root.stage.clone() })?;
            let output = stage.graph.outputs().get(&root.output).ok_or_else(|| {
                SimulationError::UnknownOutput {
                    stage: root.stage.clone(),
                    output: root.output.clone(),
                }
            })?;
            reach_scope(
                &mut plan,
                &mut seen,
                &root.stage,
                &stage.graph,
                &FrozenGraphScopeId::Root,
                &[],
                output.value,
                &request.environment,
                &request.program,
            )?;
        }
        let _identity_table_sizes = plan.interners.values.len() +
            plan.interners.views.len() +
            plan.interners.selectors.len();
        Ok(plan)
    }
}

fn reach_scope(
    plan: &mut Plan,
    seen: &mut BTreeSet<(StageId, FrozenGraphScopeId, Vec<String>, WireRef)>,
    stage: &StageId,
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    occurrence: &[String],
    wire: WireRef,
    env: &mxx_ir_core::ParamEnv,
    program: &SimulationProgram,
) -> Result<(), SimulationError> {
    if !seen.insert((stage.clone(), scope_id.clone(), occurrence.to_vec(), wire)) {
        return Ok(());
    }
    let scope = graph.scope(scope_id).ok_or_else(|| SimulationError::InvalidGraph {
        message: format!("missing scope {scope_id:?}"),
        site: None,
    })?;
    let node = scope.node(wire.node).ok_or_else(|| SimulationError::InvalidGraph {
        message: "planned wire node is unavailable".into(),
        site: None,
    })?;
    if let mxx_ir_core::node::NodeKind::Input { artifact: Some(artifact), .. } = node.kind() {
        let producer = program
            .stages
            .iter()
            .find(|stage| stage.production_id == artifact.production_id)
            .ok_or_else(|| SimulationError::ArtifactResolution {
                message: "artifact producer missing".into(),
                site: None,
            })?;
        let output = producer.graph.outputs().get(&artifact.artifact_name).ok_or_else(|| {
            SimulationError::ArtifactResolution {
                message: "artifact producer output missing".into(),
                site: None,
            }
        })?;
        plan.artifact_outputs
            .entry(producer.id.clone())
            .or_default()
            .insert(artifact.artifact_name.clone());
        reach_scope(
            plan,
            seen,
            &producer.id,
            &producer.graph,
            &FrozenGraphScopeId::Root,
            &[],
            output.value,
            env,
            program,
        )?;
    }
    let key = PlannedWire {
        stage: stage.clone(),
        scope: scope_id.clone(),
        occurrence: occurrence.to_vec(),
        wire,
    };
    let id = crate::ValueId(plan.wires.len() as u32);
    plan.wires.push(key.clone());
    plan.by_key.insert(key, id);
    plan.interners.values.insert(
        crate::identity::ValueKey {
            stage: stage.clone(),
            scope: scope_id.clone(),
            occurrence: occurrence.to_vec(),
            wire,
        },
        id,
    );
    for arg in scope.arguments(node).ok_or_else(|| SimulationError::InvalidGraph {
        message: "foreign argument in plan".into(),
        site: None,
    })? {
        reach_scope(plan, seen, stage, graph, scope_id, occurrence, arg, env, program)?;
    }
    let arg_ids = scope
        .arguments(node)
        .unwrap_or_default()
        .iter()
        .filter_map(|arg| {
            plan.by_key
                .get(&PlannedWire {
                    stage: stage.clone(),
                    scope: scope_id.clone(),
                    occurrence: occurrence.to_vec(),
                    wire: *arg,
                })
                .copied()
        })
        .collect::<Vec<_>>();
    match node.kind() {
        mxx_ir_core::node::NodeKind::FamilyGetDynamic { .. } |
        mxx_ir_core::node::NodeKind::FamilySelectAxis { .. } |
        mxx_ir_core::node::NodeKind::FamilyGather { .. } => {
            for value in arg_ids.iter().skip(1) {
                plan.interners.intern_selector(vec![*value]);
            }
        }
        mxx_ir_core::node::NodeKind::Select { .. } => {
            if let Some(value) = arg_ids.first() {
                plan.interners.intern_selector(vec![*value]);
            }
        }
        mxx_ir_core::node::NodeKind::FamilyReindex { map, .. } => {
            plan.interners.intern_view(arg_ids, Vec::new(), std::slice::from_ref(map));
        }
        _ => {}
    }
    if let Some(child) = graph.child_scope_id(scope_id, wire.node) &&
        let Some(child_scope) = graph.scope(&child)
    {
        let occurrences = match node.kind() {
            mxx_ir_core::node::NodeKind::SequentialLoop(spec) => {
                let count =
                    spec.count.evaluate(env).ok().and_then(|x| x.to_usize()).ok_or_else(|| {
                        SimulationError::InvalidGraph {
                            message: "sequential loop count is not usize".into(),
                            site: None,
                        }
                    })?;
                (0..count)
                    .map(|iteration| format!("node:{}/iteration:{iteration}", wire.node.0))
                    .collect::<Vec<_>>()
            }
            mxx_ir_core::node::NodeKind::ParallelGrid(_) => {
                vec![format!("node:{}/grid", wire.node.0)]
            }
            _ => vec![format!("node:{}", wire.node.0)],
        };
        for child_path in occurrences {
            for output in child_scope.outputs() {
                let mut child_occurrence = occurrence.to_vec();
                child_occurrence.push(child_path.clone());
                reach_scope(
                    plan,
                    seen,
                    stage,
                    graph,
                    &child,
                    &child_occurrence,
                    *output,
                    env,
                    program,
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SimulationLimits, SimulationRoot, SimulationStage};
    use mxx_ir_core::{
        GraphOutput, NodeHandle,
        artifact::{ArtifactConfidentiality, ProductionId},
        encoding::spec_hash,
        node::{ArtifactInput, ConstantMatrix, NodeKind},
        types::MatrixType,
    };

    #[test]
    fn artifact_reachability_plans_only_the_reached_producer_output() {
        let matrix = MatrixType {
            modulus: mxx_ir_core::IntExpr::constant(17),
            ring_dimension: mxx_ir_core::IntExpr::constant(4),
            rows: mxx_ir_core::IntExpr::constant(1),
            columns: mxx_ir_core::IntExpr::constant(1),
        };
        let constant = |value| {
            NodeHandle::new(
                NodeKind::ConstantMatrix { matrix_type: matrix.clone(), value },
                vec![],
                vec![mxx_ir_core::WireType::Matrix(matrix.clone())],
            )
            .output(0)
            .unwrap()
        };
        let needed = constant(ConstantMatrix::Zero);
        let unused = constant(ConstantMatrix::Identity);
        let (producer_graph, _) = mxx_ir_core::Graph::freeze(
            "artifact-producer",
            vec![],
            BTreeMap::from([
                (
                    "needed".into(),
                    GraphOutput {
                        value: needed,
                        confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                ),
                (
                    "unused".into(),
                    GraphOutput {
                        value: unused,
                        confidentiality: Some(ArtifactConfidentiality::Public),
                    },
                ),
            ]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let environment = mxx_ir_core::ParamEnv::default();
        let producer_id = ProductionId {
            spec_hash: spec_hash(&producer_graph, &environment).unwrap(),
            execution_nonce: [7; 32],
        };
        let artifact = NodeHandle::new(
            NodeKind::Input {
                name: "artifact".into(),
                wire_type: mxx_ir_core::WireType::Matrix(matrix.clone()),
                artifact: Some(ArtifactInput {
                    production_id: producer_id.clone(),
                    artifact_name: "needed".into(),
                    confidentiality: ArtifactConfidentiality::Public,
                }),
            },
            vec![],
            vec![mxx_ir_core::WireType::Matrix(matrix)],
        )
        .output(0)
        .unwrap();
        let (consumer_graph, _) = mxx_ir_core::Graph::freeze(
            "artifact-consumer",
            vec![],
            BTreeMap::from([(
                "out".into(),
                GraphOutput { value: artifact, confidentiality: None },
            )]),
            vec![],
            vec![],
            BTreeMap::new(),
        )
        .unwrap();
        let consumer_id = StageId("consumer".into());
        let producer_stage = StageId("producer".into());
        let request = SimulationRequest {
            program: SimulationProgram {
                stages: vec![
                    SimulationStage {
                        id: producer_stage.clone(),
                        production_id: producer_id,
                        graph: producer_graph,
                    },
                    SimulationStage {
                        id: consumer_id.clone(),
                        production_id: ProductionId {
                            spec_hash: spec_hash(&consumer_graph, &environment).unwrap(),
                            execution_nonce: [8; 32],
                        },
                        graph: consumer_graph,
                    },
                ],
            },
            environment,
            roots: vec![SimulationRoot { stage: consumer_id, output: "out".into() }],
            external_inputs: vec![],
            limits: SimulationLimits::default(),
        };
        let plan = Plan::build(&request).unwrap();
        assert_eq!(
            plan.artifact_outputs.get(&producer_stage),
            Some(&BTreeSet::from(["needed".into()]))
        );
        let producer = request.program.stage(&producer_stage).unwrap();
        assert!(!plan.wires.iter().any(|wire| {
            wire.stage == producer_stage && wire.wire == producer.graph.outputs()["unused"].value
        }));
    }
}
