//! Executable Graph IR builders for the Diamond iO application path.

use crate::graph::{
    IoArtifactNames, IoGraphBuildError, IoGraphConfig, build_evaluation_graph,
    build_obfuscation_graph,
};
use mxx_ir_core::{Graph, IntExpr, artifact::ProductionId, expr::RealExpr};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondIOGraphConfig {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub state_columns: IntExpr,
    pub input_digit_count: usize,
    pub digit_branch_count: usize,
    pub output_count: usize,
    pub sample_sigma: RealExpr,
    pub tag: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondIOArtifactNames {
    pub initial_state: String,
    pub transition_prefix: String,
    pub output_projections: String,
}

impl Default for DiamondIOArtifactNames {
    fn default() -> Self {
        Self {
            initial_state: "diamond_io_initial_state".to_owned(),
            transition_prefix: "diamond_io_transition".to_owned(),
            output_projections: "diamond_io_output_projections".to_owned(),
        }
    }
}

impl DiamondIOGraphConfig {
    fn shared(&self) -> IoGraphConfig {
        IoGraphConfig {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.clone(),
            state_columns: self.state_columns.clone(),
            round_count: self.input_digit_count,
            branch_count: self.digit_branch_count,
            output_count: self.output_count,
            sample_sigma: self.sample_sigma.clone(),
            tag: self.tag.clone(),
        }
    }
}

impl From<&DiamondIOArtifactNames> for IoArtifactNames {
    fn from(names: &DiamondIOArtifactNames) -> Self {
        Self {
            initial_state: names.initial_state.clone(),
            transition_prefix: names.transition_prefix.clone(),
            output_projections: names.output_projections.clone(),
        }
    }
}

pub fn build_diamond_io_obfuscation_graph(
    config: &DiamondIOGraphConfig,
    names: &DiamondIOArtifactNames,
) -> Result<Graph, IoGraphBuildError> {
    build_obfuscation_graph("diamond-io-obfuscation", &config.shared(), &names.into())
}

pub fn build_diamond_io_evaluation_graph(
    config: &DiamondIOGraphConfig,
    names: &DiamondIOArtifactNames,
    production_id: ProductionId,
) -> Result<Graph, IoGraphBuildError> {
    build_evaluation_graph("diamond-io-evaluation", &config.shared(), &names.into(), production_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_ir_core::{
        ParamEnv,
        artifact::{SpecHash, export_validated_manifest},
        validate, validate_with_manifests,
    };
    use std::collections::BTreeMap;

    fn config() -> DiamondIOGraphConfig {
        DiamondIOGraphConfig {
            modulus: IntExpr::constant(257),
            ring_dimension: IntExpr::constant(8),
            state_columns: IntExpr::constant(4),
            input_digit_count: 2,
            digit_branch_count: 4,
            output_count: 2,
            sample_sigma: RealExpr::FromInt(IntExpr::constant(3)),
            tag: b"diamond-io-test".to_vec(),
        }
    }

    #[test]
    fn obfuscation_graph_validates_and_exports_all_rounds() {
        let graph =
            build_diamond_io_obfuscation_graph(&config(), &Default::default()).expect("graph");
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        assert!(validated.outputs.contains_key("diamond_io_transition_round_0"));
        assert!(validated.outputs.contains_key("diamond_io_transition_round_1"));
        assert!(validated.outputs.contains_key("diamond_io_output_projections"));
    }

    #[test]
    fn evaluation_graph_validates_against_obfuscation_manifest() {
        let config = config();
        let names = DiamondIOArtifactNames::default();
        let producer = build_diamond_io_obfuscation_graph(&config, &names).expect("producer graph");
        let producer = validate(&producer, &ParamEnv::default()).expect("producer validation");
        let production_id = ProductionId { spec_hash: SpecHash([1; 32]), execution_nonce: [2; 32] };
        let manifest =
            export_validated_manifest(production_id.clone(), &producer).expect("manifest");
        let consumer = build_diamond_io_evaluation_graph(&config, &names, production_id.clone())
            .expect("consumer graph");
        let validated = validate_with_manifests(
            &consumer,
            &ParamEnv::default(),
            &BTreeMap::from([(production_id, manifest)]),
        )
        .expect("consumer validation");
        assert_eq!(validated.outputs.len(), config.output_count);
    }
}
