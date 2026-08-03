//! Shared manifest-linked noise simulation for iO producer/consumer graphs.

use mxx_dsl::BuiltGraph;
use mxx_ir_core::{
    ParamEnv, ScopedWireRef,
    artifact::{ProductionId, export_validated_manifest, production_id},
    encoding::spec_hash,
};
use mxx_noise_simulator::{NoiseReport, simulate_with_selection_values};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub(crate) enum LinkedNoiseError {
    #[error("producer validation failed: {0}")]
    ProducerValidation(String),
    #[error("producer symbolic elaboration failed: {0}")]
    ProducerElaboration(String),
    #[error("producer manifest export failed: {0}")]
    Manifest(String),
    #[error("consumer graph construction failed: {0}")]
    ConsumerBuild(String),
    #[error("consumer symbolic elaboration failed: {0}")]
    ConsumerElaboration(String),
    #[error("noise simulation failed: {0}")]
    Simulation(String),
}

pub(crate) struct LinkedNoiseSimulation {
    pub production: ProductionId,
    pub report: NoiseReport,
}

/// Elaborates a producer and its artifact-consuming graph in one symbolic
/// provenance chain, then simulates the consumer.  The consumer is constructed
/// only after deriving the producer identity, so callers cannot accidentally
/// simulate artifacts from a graph with a different specification hash.
pub(crate) fn simulate_linked_graphs<F>(
    producer: &BuiltGraph,
    build_consumer: F,
    selection_values: &BTreeMap<ScopedWireRef, u64>,
) -> Result<LinkedNoiseSimulation, LinkedNoiseError>
where
    F: FnOnce(ProductionId) -> Result<BuiltGraph, String>,
{
    let bindings = ParamEnv::default();
    let validated_producer = producer
        .validate(&bindings)
        .map_err(|error| LinkedNoiseError::ProducerValidation(error.to_string()))?;
    let production = production_id(
        spec_hash(&validated_producer.source, &validated_producer.bindings)
            .map_err(|error| LinkedNoiseError::Manifest(error.to_string()))?,
        [0; 32],
    );
    let artifact_manifest = export_validated_manifest(production.clone(), &validated_producer)
        .map_err(|error| LinkedNoiseError::Manifest(error.to_string()))?;
    let elaborated_producer = producer
        .elaborate(&bindings)
        .map_err(|error| LinkedNoiseError::ProducerElaboration(error.to_string()))?;
    let symbolic_manifest = export_symbolic_manifest(production.clone(), &elaborated_producer)?;

    let consumer = build_consumer(production.clone()).map_err(LinkedNoiseError::ConsumerBuild)?;
    let artifact_manifests = BTreeMap::from([(production.clone(), artifact_manifest)]);
    let elaborated_consumer = consumer
        .elaborate_with_manifests(&bindings, &artifact_manifests, &[symbolic_manifest])
        .map_err(|error| LinkedNoiseError::ConsumerElaboration(error.to_string()))?;
    let report = simulate_with_selection_values(&elaborated_consumer, selection_values)
        .map_err(|error| LinkedNoiseError::Simulation(error.to_string()))?;
    Ok(LinkedNoiseSimulation { production, report })
}

fn export_symbolic_manifest(
    production: ProductionId,
    elaborated: &mxx_ir_symbolic::ElaboratedGraph,
) -> Result<mxx_ir_symbolic::manifest::Manifest, LinkedNoiseError> {
    let artifacts = elaborated
        .outputs
        .iter()
        .map(|(name, reference)| {
            let wire = elaborated
                .wire(reference)
                .ok_or_else(|| LinkedNoiseError::Manifest(format!("missing output {name}")))?;
            Ok((
                name.clone(),
                mxx_ir_symbolic::manifest::ExportArtifact {
                    wire_type: wire.wire_type.clone(),
                    expression: wire.expression,
                    family: wire.family.clone(),
                    content_hash: None,
                    layout: None,
                },
            ))
        })
        .collect::<Result<BTreeMap<_, _>, LinkedNoiseError>>()?;
    mxx_ir_symbolic::manifest::export_manifest(
        production,
        &artifacts,
        &elaborated.atoms,
        &elaborated.expressions,
        &elaborated.preimage_relations,
        elaborated.assumption_digest,
    )
    .map_err(|error| LinkedNoiseError::Manifest(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Ring};
    use mxx_ir_core::{RealExpr, artifact::ArtifactConfidentiality};

    #[test]
    fn producer_noise_identity_survives_the_artifact_boundary() {
        let ring = Ring::new(257, 8);
        let producer = DslContext::new("linked-noise-producer")
            .public_output("sample", ring.gaussian((1, 1), RealExpr::from_integer(2)))
            .unwrap()
            .build()
            .unwrap();
        let simulation = simulate_linked_graphs(
            &producer,
            |production| {
                let ring = Ring::new(257, 8);
                DslContext::new("linked-noise-consumer")
                    .output(
                        "result",
                        ring.artifact_input(
                            production,
                            "sample",
                            (1, 1),
                            ArtifactConfidentiality::Public,
                        ),
                    )
                    .and_then(DslContext::build)
                    .map_err(|error| error.to_string())
            },
            &BTreeMap::new(),
        )
        .unwrap();
        let result = simulation.report.outputs.get("result").expect("result noise");
        assert!(
            result
                .noise
                .as_ref()
                .is_some_and(|noise| noise.bound > bigdecimal::BigDecimal::from(0))
        );
        assert_eq!(simulation.production.execution_nonce, [0; 32]);
    }
}
