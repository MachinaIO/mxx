//! Public simulation input types and request-level uniqueness validation.

use crate::SimulationError;
use mxx_ir_core::{
    Graph, WireType, artifact::ProductionId, encoding::spec_hash, expr::IntExpr, node::NodeKind,
    types::ConcreteMatrixType,
};
use num_bigint::{BigInt, BigUint};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct StageId(pub String);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationStage {
    pub id: StageId,
    pub production_id: ProductionId,
    pub graph: Graph,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct SimulationProgram {
    pub stages: Vec<SimulationStage>,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub struct SimulationRoot {
    pub stage: StageId,
    pub output: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExternalInputFact {
    pub stage: StageId,
    pub input: String,
    pub value: ExternalInputValue,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum ExternalInputValue {
    Matrix {
        maximum_absolute_coefficient_error: BigUint,
        maximum_absolute_coefficient_value: Option<BigUint>,
        is_constant_polynomial: bool,
    },
    IntegerRange {
        minimum: BigInt,
        maximum_inclusive: BigInt,
    },
    Boolean,
    Bytes,
    Trapdoor {
        public_matrix_input: String,
    },
    Family {
        shape: Vec<usize>,
        element: Box<ExternalInputValue>,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct SimulationLimits {
    /// No request-level cap when absent.
    pub maximum_planned_wires: Option<usize>,
    pub maximum_transfer_steps: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationRequest {
    pub program: SimulationProgram,
    pub environment: mxx_ir_core::ParamEnv,
    pub roots: Vec<SimulationRoot>,
    pub external_inputs: Vec<ExternalInputFact>,
    pub limits: SimulationLimits,
}

impl SimulationProgram {
    pub fn stage(&self, id: &StageId) -> Option<&SimulationStage> {
        self.stages.iter().find(|stage| stage.id == *id)
    }
}

impl SimulationRequest {
    /// Checks request-level identity and exact root/input names. Deep graph
    /// validation belongs to the occurrence-aware planner/evaluator gate.
    pub fn validate(&self) -> Result<(), SimulationError> {
        let mut stage_ids = BTreeSet::new();
        let mut production_ids = BTreeSet::new();
        for stage in &self.program.stages {
            if !stage_ids.insert(stage.id.clone()) {
                return Err(SimulationError::DuplicateStage { stage: stage.id.clone() });
            }
            if !production_ids.insert(stage.production_id.clone()) {
                return Err(SimulationError::DuplicateProduction);
            }
            for parameter in stage.graph.parameters() {
                let bound = match parameter.kind {
                    mxx_ir_core::CompileParameterKind::Integer => {
                        self.environment.integers.contains_key(&parameter.name)
                    }
                    mxx_ir_core::CompileParameterKind::Real => {
                        self.environment.reals.contains_key(&parameter.name)
                    }
                };
                if !bound {
                    return Err(SimulationError::InvalidParameterEnvironment {
                        message: format!("missing binding for {}", parameter.name),
                    });
                }
            }
            let expected = spec_hash(&stage.graph, &self.environment).map_err(|error| {
                SimulationError::InvalidParameterEnvironment { message: error.to_string() }
            })?;
            if stage.production_id.spec_hash != expected {
                return Err(SimulationError::InvalidGraph {
                    message: format!(
                        "stage {:?} production spec hash does not match its frozen graph",
                        stage.id
                    ),
                    site: None,
                });
            }
        }

        let mut roots = BTreeSet::new();
        for root in &self.roots {
            let stage = self
                .program
                .stage(&root.stage)
                .ok_or_else(|| SimulationError::UnknownStage { stage: root.stage.clone() })?;
            if !roots.insert(root.clone()) {
                return Err(SimulationError::DuplicateRoot);
            }
            if !stage.graph.outputs().contains_key(&root.output) {
                return Err(SimulationError::UnknownOutput {
                    stage: root.stage.clone(),
                    output: root.output.clone(),
                });
            }
            let output = stage.graph.outputs().get(&root.output).expect("checked graph output");
            let output_type = stage
                .graph
                .root_scope()
                .node(output.value.node)
                .and_then(|node| node.output_types().get(output.value.port.0 as usize));
            if !output_type.is_some_and(is_matrix_or_matrix_family) {
                return Err(SimulationError::InvalidGraph {
                    message: format!("root {:?} must be a matrix or family of matrices", root),
                    site: None,
                });
            }
        }

        let mut external = BTreeSet::new();
        for fact in &self.external_inputs {
            let stage = self
                .program
                .stage(&fact.stage)
                .ok_or_else(|| SimulationError::UnknownStage { stage: fact.stage.clone() })?;
            if !external.insert((fact.stage.clone(), fact.input.clone())) {
                return Err(SimulationError::DuplicateExternalInput);
            }
            let input = stage.graph.root_scope().nodes().iter().find(
                |node| matches!(node.kind(), NodeKind::Input { name, .. } if name == &fact.input),
            );
            let Some(input) = input else {
                return Err(SimulationError::InvalidGraph {
                    message: format!("unknown input {}", fact.input),
                    site: None,
                });
            };
            if let NodeKind::Input { artifact: Some(_), .. } = input.kind() {
                return Err(SimulationError::ArtifactResolution {
                    message: format!(
                        "artifact input {} must inherit its producer state",
                        fact.input
                    ),
                    site: None,
                });
            }
            validate_external_fact_shape(
                &fact.value,
                input.output_types().first(),
                &self.environment,
            )
            .map_err(|message| SimulationError::InvalidGraph { message, site: None })?;
        }
        let facts = self
            .external_inputs
            .iter()
            .map(|fact| ((fact.stage.clone(), fact.input.clone()), fact))
            .collect::<BTreeMap<_, _>>();
        for fact in &self.external_inputs {
            let ExternalInputValue::Trapdoor { public_matrix_input } = &fact.value else {
                continue;
            };
            let stage = self.program.stage(&fact.stage).expect("stage checked above");
            let trapdoor_node = stage.graph.root_scope().nodes().iter().find(|node| {
                matches!(node.kind(), NodeKind::Input { name, .. } if name == &fact.input)
            }).expect("input checked above");
            let NodeKind::Input { wire_type: trapdoor_type, .. } = trapdoor_node.kind() else {
                unreachable!("input lookup only finds input nodes");
            };
            let public_node = stage.graph.root_scope().nodes().iter().find(|node| {
                matches!(node.kind(), NodeKind::Input { name, .. } if name == public_matrix_input)
            }).ok_or_else(|| SimulationError::Relation {
                message: format!("trapdoor {} names missing public input {}", fact.input, public_matrix_input),
                site: None,
            })?;
            let NodeKind::Input { wire_type: public_type, artifact: public_artifact, .. } =
                public_node.kind()
            else {
                unreachable!("input lookup only finds input nodes");
            };
            if public_artifact.is_some() {
                return Err(SimulationError::Relation {
                    message: format!(
                        "trapdoor public input {} cannot be an artifact",
                        public_matrix_input
                    ),
                    site: None,
                });
            }
            let public_fact = facts
                .get(&(fact.stage.clone(), public_matrix_input.clone()))
                .ok_or_else(|| SimulationError::Relation {
                    message: format!(
                        "trapdoor public input {} has no external fact",
                        public_matrix_input
                    ),
                    site: None,
                })?;
            if !external_fact_is_zero_matrix(&public_fact.value) {
                return Err(SimulationError::Relation {
                    message: format!(
                        "trapdoor public input {} must have zero declared error",
                        public_matrix_input
                    ),
                    site: None,
                });
            }
            if !trapdoor_public_types_match(trapdoor_type, public_type, &self.environment)? {
                return Err(SimulationError::Relation {
                    message: format!(
                        "trapdoor public input {} has incompatible matrix or family shape",
                        public_matrix_input
                    ),
                    site: None,
                });
            }
        }
        Ok(())
    }
}

fn validate_external_fact_shape(
    fact: &ExternalInputValue,
    wire_type: Option<&WireType>,
    environment: &mxx_ir_core::ParamEnv,
) -> Result<(), String> {
    match fact {
        ExternalInputValue::Family { shape, element } => {
            if shape.is_empty() {
                return Err("external family shape must have positive rank".to_owned());
            }
            if matches!(**element, ExternalInputValue::Family { .. }) {
                return Err("nested external families are unsupported".to_owned());
            }
            let mut product = 1usize;
            for extent in shape {
                product = product
                    .checked_mul(*extent)
                    .ok_or_else(|| "external family extent product overflows usize".to_owned())?;
            }
            let (declared_element, declared): (&WireType, Vec<usize>) = match wire_type {
                Some(WireType::Family { element, shape }) => {
                    let concrete = shape
                        .iter()
                        .map(|extent| {
                            extent
                                .evaluate(environment)
                                .map_err(|error| format!("family extent is unresolved: {error}"))?
                                .to_usize()
                                .ok_or_else(|| {
                                    "family extent is not a nonnegative usize".to_owned()
                                })
                        })
                        .collect::<Result<Vec<_>, String>>()?;
                    (element.as_ref(), concrete)
                }
                _ => return Err("external family fact does not match input wire".to_owned()),
            };
            if declared.len() != shape.len() {
                return Err("external family rank does not match input wire".to_owned());
            }
            for (position, concrete) in declared.iter().enumerate() {
                if *concrete != shape[position] {
                    return Err(format!(
                        "external family extent {position} does not match input wire"
                    ));
                }
            }
            let _ = product;
            validate_external_fact_shape(element, Some(declared_element), environment)
        }
        ExternalInputValue::Matrix {
            maximum_absolute_coefficient_error,
            maximum_absolute_coefficient_value,
            ..
        } => {
            let Some((matrix_type, declared_bound)) = (match wire_type {
                Some(WireType::Matrix(matrix_type)) => Some((matrix_type, None)),
                Some(WireType::SmallMatrix { matrix: matrix_type, max_coefficient_bound }) |
                Some(WireType::Preimage { matrix: matrix_type, max_coefficient_bound }) => {
                    Some((matrix_type, Some(max_coefficient_bound)))
                }
                _ => None,
            }) else {
                return Err("external matrix fact does not match input wire".to_owned());
            };
            let matrix = concrete_matrix(matrix_type, environment)?;
            let cap = crate::centered_residue_bound(&matrix.modulus)
                .map_err(|error| error.to_string())?;
            if maximum_absolute_coefficient_value.as_ref().is_some_and(|value| value > &cap) {
                return Err("external matrix coefficient magnitude exceeds centered residue bound"
                    .to_owned());
            }
            if let (Some(value), Some(bound)) =
                (maximum_absolute_coefficient_value.as_ref(), declared_bound)
            {
                let bound = bound
                    .evaluate(environment)
                    .map_err(|error| {
                        format!("bounded RHS coefficient bound is unresolved: {error}")
                    })?
                    .to_biguint()
                    .ok_or_else(|| "bounded RHS coefficient bound is negative".to_owned())?;
                if value > &bound {
                    return Err(
                        "external matrix coefficient magnitude exceeds declared bounded RHS bound"
                            .to_owned(),
                    );
                }
            }
            let _ = maximum_absolute_coefficient_error;
            Ok(())
        }
        ExternalInputValue::IntegerRange { minimum, maximum_inclusive } => {
            if minimum > maximum_inclusive {
                Err("external integer range minimum exceeds maximum".to_owned())
            } else if !matches!(wire_type, Some(WireType::Int | WireType::ConstantInt)) {
                Err("external integer range fact requires an integer input wire".to_owned())
            } else {
                Ok(())
            }
        }
        ExternalInputValue::Boolean => {
            if matches!(wire_type, Some(WireType::Bool | WireType::ConstantBool)) {
                Ok(())
            } else {
                Err("external boolean fact does not match input wire".to_owned())
            }
        }
        ExternalInputValue::Bytes => {
            if matches!(wire_type, Some(WireType::Bytes { .. })) {
                Ok(())
            } else {
                Err("external bytes fact does not match input wire".to_owned())
            }
        }
        ExternalInputValue::Trapdoor { public_matrix_input } => {
            if public_matrix_input.is_empty() ||
                !matches!(wire_type, Some(WireType::Trapdoor { .. }))
            {
                return Err("external trapdoor fact does not match input wire".to_owned());
            }
            Ok(())
        }
    }
}

fn is_matrix_or_matrix_family(wire_type: &WireType) -> bool {
    match wire_type {
        WireType::Matrix(_) | WireType::SmallMatrix { .. } | WireType::Preimage { .. } => true,
        WireType::Family { element, .. } => {
            matches!(
                element.as_ref(),
                WireType::Matrix(_) | WireType::SmallMatrix { .. } | WireType::Preimage { .. }
            )
        }
        _ => false,
    }
}

fn external_fact_is_zero_matrix(fact: &ExternalInputValue) -> bool {
    match fact {
        ExternalInputValue::Matrix { maximum_absolute_coefficient_error, .. } => {
            maximum_absolute_coefficient_error == &BigUint::from(0u8)
        }
        ExternalInputValue::Family { element, .. } => external_fact_is_zero_matrix(element),
        _ => false,
    }
}

fn trapdoor_public_types_match(
    trapdoor: &WireType,
    public: &WireType,
    environment: &mxx_ir_core::ParamEnv,
) -> Result<bool, SimulationError> {
    match (trapdoor, public) {
        (
            WireType::Trapdoor { matrix: trapdoor_matrix, .. },
            WireType::Matrix(public_matrix) |
            WireType::SmallMatrix { matrix: public_matrix, .. } |
            WireType::Preimage { matrix: public_matrix, .. },
        ) => Ok(concrete_matrix(trapdoor_matrix, environment)
            .map_err(|message| SimulationError::InvalidGraph { message, site: None })? ==
            concrete_matrix(public_matrix, environment)
                .map_err(|message| SimulationError::InvalidGraph { message, site: None })?),
        (
            WireType::Family { element: trapdoor_element, shape: trapdoor_shape },
            WireType::Family { element: public_element, shape: public_shape },
        ) => {
            if trapdoor_shape.len() != public_shape.len() {
                return Ok(false);
            }
            for (left, right) in trapdoor_shape.iter().zip(public_shape) {
                if left.evaluate(environment).map_err(|error| SimulationError::InvalidGraph {
                    message: error.to_string(),
                    site: None,
                })? != right.evaluate(environment).map_err(|error| {
                    SimulationError::InvalidGraph { message: error.to_string(), site: None }
                })? {
                    return Ok(false);
                }
            }
            trapdoor_public_types_match(trapdoor_element, public_element, environment)
        }
        _ => Ok(false),
    }
}

fn concrete_matrix(
    matrix: &mxx_ir_core::types::MatrixType,
    environment: &mxx_ir_core::ParamEnv,
) -> Result<ConcreteMatrixType, String> {
    let eval = |expression: &IntExpr, name: &str| {
        expression
            .evaluate(environment)
            .map_err(|error| format!("{name} is unresolved: {error}"))
            .and_then(|value| {
                value.to_usize().ok_or_else(|| format!("{name} is not a nonnegative usize"))
            })
    };
    let modulus = matrix
        .modulus
        .evaluate(environment)
        .map_err(|error| format!("modulus is unresolved: {error}"))?;
    if modulus <= 0.into() {
        return Err("matrix modulus must be positive".to_owned());
    }
    Ok(ConcreteMatrixType {
        modulus,
        ring_dimension: eval(&matrix.ring_dimension, "ring dimension")?,
        rows: eval(&matrix.rows, "matrix rows")?,
        columns: eval(&matrix.columns, "matrix columns")?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        GraphOutput, NodeHandle, ParamEnv,
        artifact::ProductionId,
        encoding::spec_hash,
        expr::RealExpr,
        node::NodeKind,
        types::{MatrixType, WireType},
    };

    fn matrix_type(rows: i64, columns: i64) -> MatrixType {
        MatrixType {
            modulus: IntExpr::constant(17),
            ring_dimension: IntExpr::constant(8),
            rows: IntExpr::constant(rows),
            columns: IntExpr::constant(columns),
        }
    }

    fn request(
        public_matrix: MatrixType,
        trapdoor_matrix: MatrixType,
        public_fact: ExternalInputValue,
    ) -> SimulationRequest {
        let public = NodeHandle::new(
            NodeKind::Input {
                name: "public".to_owned(),
                wire_type: WireType::Matrix(public_matrix.clone()),
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Matrix(public_matrix)],
        )
        .output(0)
        .unwrap();
        let trapdoor = NodeHandle::new(
            NodeKind::Input {
                name: "trapdoor".to_owned(),
                wire_type: WireType::Trapdoor {
                    matrix: trapdoor_matrix.clone(),
                    sigma: RealExpr::FromInt(IntExpr::constant(1)),
                    gadget_base: IntExpr::constant(2),
                    digit_count: IntExpr::constant(2),
                    preimage_max_coefficient_bound: IntExpr::constant(4),
                },
                artifact: None,
            },
            Vec::new(),
            vec![WireType::Trapdoor {
                matrix: trapdoor_matrix,
                sigma: RealExpr::FromInt(IntExpr::constant(1)),
                gadget_base: IntExpr::constant(2),
                digit_count: IntExpr::constant(2),
                preimage_max_coefficient_bound: IntExpr::constant(4),
            }],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "pairing",
            Vec::new(),
            BTreeMap::from([(
                String::from("output"),
                GraphOutput { value: public, confidentiality: None },
            )]),
            vec![trapdoor],
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let environment = ParamEnv::default();
        let production_id = ProductionId {
            spec_hash: spec_hash(&graph, &environment).unwrap(),
            execution_nonce: [0; 32],
        };
        SimulationRequest {
            program: SimulationProgram {
                stages: vec![SimulationStage {
                    id: StageId("stage".to_owned()),
                    production_id,
                    graph,
                }],
            },
            environment,
            roots: vec![SimulationRoot {
                stage: StageId("stage".to_owned()),
                output: "output".to_owned(),
            }],
            external_inputs: vec![
                ExternalInputFact {
                    stage: StageId("stage".to_owned()),
                    input: "public".to_owned(),
                    value: public_fact,
                },
                ExternalInputFact {
                    stage: StageId("stage".to_owned()),
                    input: "trapdoor".to_owned(),
                    value: ExternalInputValue::Trapdoor {
                        public_matrix_input: "public".to_owned(),
                    },
                },
            ],
            limits: SimulationLimits::default(),
        }
    }

    fn zero_matrix_fact() -> ExternalInputValue {
        ExternalInputValue::Matrix {
            maximum_absolute_coefficient_error: BigUint::from(0u8),
            maximum_absolute_coefficient_value: None,
            is_constant_polynomial: false,
        }
    }

    #[test]
    fn trapdoor_fact_requires_exact_zero_public_fact_and_pairing() {
        let ty = matrix_type(2, 2);
        assert!(request(ty.clone(), ty.clone(), zero_matrix_fact()).validate().is_ok());
        let nonzero = ExternalInputValue::Matrix {
            maximum_absolute_coefficient_error: BigUint::from(1u8),
            maximum_absolute_coefficient_value: None,
            is_constant_polynomial: false,
        };
        assert!(matches!(
            request(ty.clone(), ty.clone(), nonzero).validate(),
            Err(SimulationError::Relation { .. })
        ));
        let missing = request(ty.clone(), ty.clone(), zero_matrix_fact());
        let mut facts = missing.external_inputs;
        facts.remove(0);
        let missing = SimulationRequest { external_inputs: facts, ..missing };
        assert!(matches!(missing.validate(), Err(SimulationError::Relation { .. })));
    }

    #[test]
    fn trapdoor_pairing_rejects_incompatible_public_matrix() {
        let request = request(matrix_type(1, 2), matrix_type(2, 2), zero_matrix_fact());
        assert!(matches!(request.validate(), Err(SimulationError::Relation { .. })));
    }

    #[test]
    fn bounded_external_matrix_fact_cannot_exceed_declared_rhs_bound() {
        let ty = matrix_type(1, 1);
        let wire = WireType::SmallMatrix { matrix: ty, max_coefficient_bound: 3.into() };
        let fact = ExternalInputValue::Matrix {
            maximum_absolute_coefficient_error: BigUint::ZERO,
            maximum_absolute_coefficient_value: Some(BigUint::from(4u8)),
            is_constant_polynomial: false,
        };
        let error = validate_external_fact_shape(&fact, Some(&wire), &ParamEnv::default())
            .expect_err("bounded RHS facts above the wire bound must be rejected");
        assert!(error.contains("declared bounded RHS bound"));
    }

    #[test]
    fn integer_range_fact_and_non_matrix_root_are_rejected() {
        let ty = matrix_type(2, 2);
        let mut request = request(
            ty.clone(),
            ty,
            ExternalInputValue::IntegerRange { minimum: 0.into(), maximum_inclusive: 1.into() },
        );
        assert!(matches!(request.validate(), Err(SimulationError::InvalidGraph { .. })));
        let public = NodeHandle::new(
            NodeKind::ConstantInt(1.into()),
            Vec::new(),
            vec![WireType::ConstantInt],
        )
        .output(0)
        .unwrap();
        let graph = Graph::freeze(
            "integer",
            Vec::new(),
            BTreeMap::from([(
                String::from("output"),
                GraphOutput { value: public, confidentiality: None },
            )]),
            Vec::new(),
            Vec::new(),
            BTreeMap::new(),
        )
        .unwrap()
        .0;
        let environment = ParamEnv::default();
        let stage = SimulationStage {
            id: StageId("integer".to_owned()),
            production_id: ProductionId {
                spec_hash: spec_hash(&graph, &environment).unwrap(),
                execution_nonce: [1; 32],
            },
            graph,
        };
        request = SimulationRequest {
            program: SimulationProgram { stages: vec![stage] },
            environment,
            roots: vec![SimulationRoot {
                stage: StageId("integer".to_owned()),
                output: "output".to_owned(),
            }],
            external_inputs: Vec::new(),
            limits: SimulationLimits::default(),
        };
        assert!(matches!(request.validate(), Err(SimulationError::InvalidGraph { .. })));
    }
}
