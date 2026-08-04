use super::{DiamondCompileError, DiamondWeCompiler};
use mxx_bench_estimator::{
    CostReport, EstimateConfig, EstimateError, MeasurementBackend, estimate,
};
use mxx_ir_core::{
    artifact::{export_validated_manifest, production_id},
    encoding::spec_hash,
};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct DiamondCostEstimate {
    pub encryption: CostReport,
    pub decryption: CostReport,
}

#[derive(Debug, Error)]
pub enum DiamondEstimateError {
    #[error(transparent)]
    Compile(#[from] DiamondCompileError),
    #[error("Diamond estimate graph validation failed: {0}")]
    Validation(String),
    #[error(transparent)]
    Estimate(#[from] EstimateError),
    #[error("Diamond estimate manifest construction failed: {0}")]
    Manifest(String),
}

pub fn estimate_diamond_cost<B>(
    compiler: &DiamondWeCompiler,
    backend: &mut B,
    config: &EstimateConfig,
) -> Result<DiamondCostEstimate, DiamondEstimateError>
where
    B: MeasurementBackend,
{
    let bindings = compiler.circuit_bindings()?;
    let encryption = compiler.build_encryption()?.graph;
    let validated_encryption = encryption
        .validate(&bindings)
        .map_err(|error| DiamondEstimateError::Validation(error.to_string()))?;
    let encryption_id = production_id(
        spec_hash(&validated_encryption.source, &validated_encryption.bindings)
            .map_err(|error| DiamondEstimateError::Manifest(error.to_string()))?,
        [0; 32],
    );
    let artifact_manifest = export_validated_manifest(encryption_id.clone(), &validated_encryption)
        .map_err(|error| DiamondEstimateError::Manifest(error.to_string()))?;
    let encryption_report = estimate(&validated_encryption, backend, config)?;

    let decryption = compiler.build_decryption(encryption_id.clone())?.graph;
    let validated_decryption = decryption
        .validate_with_manifests(&bindings, &BTreeMap::from([(encryption_id, artifact_manifest)]))
        .map_err(|error| DiamondEstimateError::Validation(error.to_string()))?;
    let decryption_report = estimate(&validated_decryption, backend, config)?;
    Ok(DiamondCostEstimate { encryption: encryption_report, decryption: decryption_report })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diamond::{DiamondWeCompiler, DiamondWeConfig};
    use mxx_bench_estimator::{MeasurementNode, NodeMeasurement};
    use mxx_gadgets::circuit::BooleanCircuitShape;
    use mxx_ir_core::{ParamEnv, RealExpr, types::ConcreteWireType};
    use std::convert::Infallible;

    struct UnitBackend;

    impl MeasurementBackend for UnitBackend {
        type Error = Infallible;

        fn measure(
            &mut self,
            _graph: &str,
            _node: &MeasurementNode<'_>,
            _bindings: &ParamEnv,
        ) -> Result<NodeMeasurement, Self::Error> {
            Ok(NodeMeasurement { work_seconds: 1.0, latency_seconds: 1.0, workspace_bytes: 8 })
        }

        fn persistent_bytes(&self, wire_type: &ConcreteWireType) -> u64 {
            match wire_type {
                ConcreteWireType::Matrix(matrix) | ConcreteWireType::Preimage(matrix) => {
                    (matrix.rows * matrix.columns * matrix.ring_dimension * 8) as u64
                }
                _ => 8,
            }
        }
    }

    #[test]
    fn estimator_consumes_the_actual_encryption_and_decryption_graphs() {
        let compiler = DiamondWeCompiler::new(
            DiamondWeConfig {
                modulus: 257.into(),
                ring_dimension: 8,
                input_count: 1,
                digit_base: 2,
                batch_bits: 1,
                gadget_base: 4.into(),
                digit_count: 2,
                trapdoor_sigma: RealExpr::from_integer(4),
                error_sigma: RealExpr::from_integer(1),
                error_max_coefficient_bound: 6.into(),
                preimage_max_coefficient_bound: 26.into(),
                bgg_tag: b"diamond-estimate-test".to_vec(),
            },
            BooleanCircuitShape {
                instance_width: 0,
                witness_width: 1,
                depth: 1,
                max_layer_width: 1,
            },
        )
        .unwrap();
        let estimate = estimate_diamond_cost(
            &compiler,
            &mut UnitBackend,
            &EstimateConfig { device_pool_size: 2, per_instance_occupancy: 1 },
        )
        .unwrap();
        assert!(estimate.encryption.total_work_seconds > 0.0);
        assert!(estimate.decryption.total_work_seconds > 0.0);
        assert!(estimate.encryption.peak_memory_bytes > 0);
        assert!(estimate.decryption.maximum_parallelism >= 2);
    }
}
