//! Executable Graph IR builders for the maintained AKY24 iO model.

use crate::graph::{
    IoArtifactNames, IoGraphBuildError, IoGraphConfig, build_evaluation_graph,
    build_obfuscation_graph,
};
use mxx_ir_core::{Graph, IntExpr, artifact::ProductionId, expr::RealExpr};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Aky24IOGraphConfig {
    pub modulus: IntExpr,
    pub ring_dimension: IntExpr,
    pub state_columns: IntExpr,
    pub input_size: usize,
    pub prf_batch_bits: usize,
    pub output_count: usize,
    pub sample_sigma: RealExpr,
    pub tag: Vec<u8>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Aky24IOArtifactNames {
    pub initial_state: String,
    pub transition_prefix: String,
    pub output_projections: String,
}

impl Default for Aky24IOArtifactNames {
    fn default() -> Self {
        Self {
            initial_state: "aky24_io_initial_state".to_owned(),
            transition_prefix: "aky24_io_prf_transition".to_owned(),
            output_projections: "aky24_io_output_projections".to_owned(),
        }
    }
}

impl Aky24IOGraphConfig {
    fn shared(&self) -> Result<IoGraphConfig, IoGraphBuildError> {
        if self.prf_batch_bits == 0 || !self.input_size.is_multiple_of(self.prf_batch_bits) {
            return Err(IoGraphBuildError::NonPositiveParameter);
        }
        let branch_count = 1usize
            .checked_shl(
                self.prf_batch_bits.try_into().map_err(|_| IoGraphBuildError::DimensionOverflow)?,
            )
            .ok_or(IoGraphBuildError::DimensionOverflow)?;
        Ok(IoGraphConfig {
            modulus: self.modulus.clone(),
            ring_dimension: self.ring_dimension.clone(),
            state_columns: self.state_columns.clone(),
            round_count: self.input_size / self.prf_batch_bits,
            branch_count,
            output_count: self.output_count,
            sample_sigma: self.sample_sigma.clone(),
            tag: self.tag.clone(),
        })
    }
}

impl From<&Aky24IOArtifactNames> for IoArtifactNames {
    fn from(names: &Aky24IOArtifactNames) -> Self {
        Self {
            initial_state: names.initial_state.clone(),
            transition_prefix: names.transition_prefix.clone(),
            output_projections: names.output_projections.clone(),
        }
    }
}

pub fn build_aky24_io_obfuscation_graph(
    config: &Aky24IOGraphConfig,
    names: &Aky24IOArtifactNames,
) -> Result<Graph, IoGraphBuildError> {
    build_obfuscation_graph("aky24-io-obfuscation", &config.shared()?, &names.into())
}

pub fn build_aky24_io_evaluation_graph(
    config: &Aky24IOGraphConfig,
    names: &Aky24IOArtifactNames,
    production_id: ProductionId,
) -> Result<Graph, IoGraphBuildError> {
    build_evaluation_graph("aky24-io-evaluation", &config.shared()?, &names.into(), production_id)
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

    fn config() -> Aky24IOGraphConfig {
        Aky24IOGraphConfig {
            modulus: IntExpr::constant(257),
            ring_dimension: IntExpr::constant(8),
            state_columns: IntExpr::constant(4),
            input_size: 4,
            prf_batch_bits: 2,
            output_count: 1,
            sample_sigma: RealExpr::FromInt(IntExpr::constant(3)),
            tag: b"aky24-io-test".to_vec(),
        }
    }

    #[test]
    fn obfuscation_graph_uses_one_family_per_prf_round() {
        let config = config();
        let graph = build_aky24_io_obfuscation_graph(&config, &Default::default()).expect("graph");
        let validated = validate(&graph, &ParamEnv::default()).expect("validation");
        assert!(validated.outputs.contains_key("aky24_io_prf_transition_round_0"));
        assert!(validated.outputs.contains_key("aky24_io_prf_transition_round_1"));
    }

    #[test]
    fn evaluation_graph_validates_against_obfuscation_manifest() {
        let config = config();
        let names = Aky24IOArtifactNames::default();
        let producer = build_aky24_io_obfuscation_graph(&config, &names).expect("producer graph");
        let producer = validate(&producer, &ParamEnv::default()).expect("producer validation");
        let production_id = ProductionId { spec_hash: SpecHash([3; 32]), execution_nonce: [4; 32] };
        let manifest =
            export_validated_manifest(production_id.clone(), &producer).expect("manifest");
        let consumer = build_aky24_io_evaluation_graph(&config, &names, production_id.clone())
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
