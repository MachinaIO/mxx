//! Shared executable Graph IR construction used by the iO application models.
//!
//! This module contains only scheme-independent control flow: preprocessing
//! exports an initial state, a transition family for every round, and final
//! projection matrices; evaluation imports those artifacts, selects one branch
//! per round, and decodes each projected output. Scheme modules own their
//! public configuration types and map them into this representation.

use mxx_bgg::{GraphBuilder, OutputFamilyError};
use mxx_graph_ir::{
    Graph, IntExpr,
    artifact::ProductionId,
    expr::RealExpr,
    node::{HashVariant, MatrixBinaryOp},
    types::MatrixType,
};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct IoGraphConfig {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub state_columns: IntExpr,
    pub round_count: usize,
    pub branch_count: usize,
    pub output_count: usize,
    pub sample_sigma: RealExpr,
    pub tag: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct IoArtifactNames {
    pub initial_state: String,
    pub transition_prefix: String,
    pub output_projections: String,
}

impl IoArtifactNames {
    fn transition(&self, round: usize) -> String {
        format!("{}_round_{round}", self.transition_prefix)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum IoGraphBuildError {
    #[error("round, branch, output, and state dimensions must be positive")]
    NonPositiveParameter,
    #[error("a graph dimension overflowed usize")]
    DimensionOverflow,
    #[error(transparent)]
    OutputFamily(#[from] OutputFamilyError),
}

impl IoGraphConfig {
    fn validate(&self) -> Result<(), IoGraphBuildError> {
        if self.round_count == 0 || self.branch_count == 0 || self.output_count == 0 {
            return Err(IoGraphBuildError::NonPositiveParameter);
        }
        Ok(())
    }

    fn state_type(&self) -> MatrixType {
        MatrixType {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.clone(),
            rows: IntExpr::constant(1),
            columns: self.state_columns.clone(),
        }
    }

    fn transition_type(&self) -> MatrixType {
        MatrixType {
            rows: self.state_columns.clone(),
            columns: self.state_columns.clone(),
            ..self.state_type()
        }
    }

    fn projection_type(&self) -> MatrixType {
        MatrixType {
            rows: self.state_columns.clone(),
            columns: IntExpr::constant(1),
            ..self.state_type()
        }
    }

    fn scalar_type(&self) -> MatrixType {
        MatrixType {
            rows: IntExpr::constant(1),
            columns: IntExpr::constant(1),
            ..self.state_type()
        }
    }
}

pub(crate) fn build_obfuscation_graph(
    graph_name: &str,
    config: &IoGraphConfig,
    names: &IoArtifactNames,
) -> Result<Graph, IoGraphBuildError> {
    config.validate()?;
    let mut builder = GraphBuilder::new(graph_name, Vec::new());
    let hash_key = builder.bytes_input("hash_key", 32);

    let initial_state = builder.gaussian_sample(config.state_type(), config.sample_sigma.clone());
    builder.output(names.initial_state.clone(), &initial_state);

    for round in 0..config.round_count {
        let transitions = (0..config.branch_count)
            .map(|branch| {
                let mut tag = config.tag.clone();
                tag.extend_from_slice(b":transition");
                builder.hash_sample(
                    hash_key,
                    config.transition_type(),
                    HashVariant::Plain,
                    tag,
                    vec![IntExpr::constant(round), IntExpr::constant(branch)],
                    None,
                    None,
                )
            })
            .collect::<Vec<_>>();
        builder.output_family(names.transition(round), &transitions)?;
    }

    let projections = (0..config.output_count)
        .map(|output| {
            let mut tag = config.tag.clone();
            tag.extend_from_slice(b":projection");
            builder.hash_sample(
                hash_key,
                config.projection_type(),
                HashVariant::Plain,
                tag,
                vec![IntExpr::constant(output)],
                None,
                None,
            )
        })
        .collect::<Vec<_>>();
    builder.output_family(names.output_projections.clone(), &projections)?;
    Ok(builder.finish())
}

pub(crate) fn build_evaluation_graph(
    graph_name: &str,
    config: &IoGraphConfig,
    names: &IoArtifactNames,
    production_id: ProductionId,
) -> Result<Graph, IoGraphBuildError> {
    config.validate()?;
    let mut builder = GraphBuilder::new(graph_name, Vec::new());
    let mut state = builder.artifact_input(
        "initial_state_artifact",
        config.state_type(),
        production_id.clone(),
        names.initial_state.clone(),
    );

    for round in 0..config.round_count {
        let branch = builder.integer_input(format!("branch_{round}"));
        let transitions = if config.branch_count == 1 {
            vec![builder.artifact_input(
                format!("transition_artifacts_{round}"),
                config.transition_type(),
                production_id.clone(),
                names.transition(round),
            )]
        } else {
            builder.artifact_family_input(
                format!("transition_artifacts_{round}"),
                config.transition_type(),
                production_id.clone(),
                names.transition(round),
                IntExpr::constant(config.branch_count),
                config.branch_count,
            )
        };
        let selected = builder.select(branch, &transitions);
        state =
            builder.matrix_binary(MatrixBinaryOp::Multiply, &state, &selected, config.state_type());
    }

    let projections = if config.output_count == 1 {
        vec![builder.artifact_input(
            "output_projection_artifacts",
            config.projection_type(),
            production_id,
            names.output_projections.clone(),
        )]
    } else {
        builder.artifact_family_input(
            "output_projection_artifacts",
            config.projection_type(),
            production_id,
            names.output_projections.clone(),
            IntExpr::constant(config.output_count),
            config.output_count,
        )
    };
    for (output, projection) in projections.iter().enumerate() {
        let noisy_plaintext = builder.matrix_binary(
            MatrixBinaryOp::Multiply,
            &state,
            projection,
            config.scalar_type(),
        );
        let decoded = builder.threshold_decode(
            &noisy_plaintext,
            IntExpr::constant(2),
            IntExpr::constant(1),
            true,
        );
        builder.output_wire(format!("output_{output}"), decoded);
    }
    Ok(builder.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_graph_ir::{ParamEnv, artifact::Manifest, validate, validate_with_manifests};
    use mxx_primitives::poly::{PolyParams, dcrt::params::DCRTPolyParams};
    use mxx_runtime::{
        RuntimeValue, artifact::MemoryArtifactStore, backend::poly::cpu_backend, execute,
        transcript::SamplingMode,
    };
    use num_bigint::{BigInt, Sign};
    use std::{collections::BTreeMap, sync::Arc};

    #[test]
    fn artifact_backed_state_path_executes_end_to_end() {
        let parameters = DCRTPolyParams::new(8, 1, 20, 4);
        let modulus: Arc<num_bigint::BigUint> = parameters.modulus().into();
        let config = IoGraphConfig {
            modulus: IntExpr::constant(BigInt::from_biguint(Sign::Plus, modulus.as_ref().clone())),
            ring_dimension: IntExpr::constant(parameters.ring_dimension()),
            state_columns: IntExpr::constant(2),
            round_count: 1,
            branch_count: 2,
            output_count: 1,
            sample_sigma: RealExpr::FromInt(IntExpr::constant(3)),
            tag: b"io-graph-e2e".to_vec(),
        };
        let names = IoArtifactNames {
            initial_state: "initial".to_owned(),
            transition_prefix: "transition".to_owned(),
            output_projections: "projections".to_owned(),
        };
        let producer =
            build_obfuscation_graph("io-test-producer", &config, &names).expect("producer");
        let producer = validate(&producer, &ParamEnv::default()).expect("producer validation");
        let mut backend = cpu_backend([parameters.clone()]);
        let mut store = MemoryArtifactStore::default();
        let produced = execute(
            &producer,
            &mut backend,
            BTreeMap::from([("hash_key".to_owned(), RuntimeValue::Bytes(vec![7; 32]))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("producer execution");
        let production_id = produced.production_id.expect("producer manifest");
        let manifest: Manifest = store.manifest(&production_id).expect("stored manifest").clone();

        let consumer =
            build_evaluation_graph("io-test-consumer", &config, &names, production_id.clone())
                .expect("consumer");
        let consumer = validate_with_manifests(
            &consumer,
            &ParamEnv::default(),
            &BTreeMap::from([(production_id, manifest)]),
        )
        .expect("consumer validation");
        let result = execute(
            &consumer,
            &mut backend,
            BTreeMap::from([("branch_0".to_owned(), RuntimeValue::Int(BigInt::from(0)))]),
            &mut store,
            SamplingMode::Fresh,
        )
        .expect("consumer execution");
        assert!(matches!(result.outputs["output_0"], RuntimeValue::Bool(_)));
    }
}
