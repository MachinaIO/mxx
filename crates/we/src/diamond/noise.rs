use super::{DiamondCompileError, DiamondWeCompiler, graph::NOISY_PLAINTEXT_OUTPUT};
use bigdecimal::BigDecimal;
use mxx_gadgets::{Poly, circuit::PolyCircuit};
use mxx_ir_core::{
    FrozenGraphScopeId, ParamEnv, ScopedWireRef,
    artifact::{ProductionId, export_validated_manifest, production_id},
    encoding::spec_hash,
    node::NodeKind,
    types::WireRef,
};
use mxx_noise_simulator::{NoiseReport, WireNoiseReport, simulate_with_selection_values};
use std::collections::BTreeMap;
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondNoiseSimulation {
    pub report: NoiseReport,
    pub final_decode: DiamondDecodeNoiseReport,
}

/// Noise estimate for the value consumed by Diamond WE's interval decoder.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiamondDecodeNoiseReport {
    pub estimate: WireNoiseReport,
    pub threshold: BigDecimal,
    pub within_threshold: bool,
}

#[derive(Debug, Error)]
pub enum DiamondNoiseError {
    #[error(transparent)]
    Compile(#[from] DiamondCompileError),
    #[error("Diamond noise graph validation failed: {0}")]
    Validation(String),
    #[error("Diamond symbolic elaboration failed: {0}")]
    Elaboration(String),
    #[error("Diamond symbolic manifest export failed: {0}")]
    Manifest(String),
    #[error("Diamond noise simulation failed: {0}")]
    Simulation(String),
    #[error("Diamond decryption graph has no noisy-plaintext output")]
    MissingNoisyPlaintext,
}

pub fn simulate_diamond_noise<P: Poly>(
    compiler: &DiamondWeCompiler,
    circuit: &PolyCircuit<P>,
    instance: &[bool],
) -> Result<DiamondNoiseSimulation, DiamondNoiseError> {
    let bindings = ParamEnv::default();
    let encryption = compiler.build_encryption(circuit, instance)?.graph;
    let validated_encryption = encryption
        .validate(&bindings)
        .map_err(|error| DiamondNoiseError::Validation(error.to_string()))?;
    let encryption_id = production_id(
        spec_hash(&validated_encryption.source, &validated_encryption.bindings)
            .map_err(|error| DiamondNoiseError::Manifest(error.to_string()))?,
        [0; 32],
    );
    let artifact_manifest = export_validated_manifest(encryption_id.clone(), &validated_encryption)
        .map_err(|error| DiamondNoiseError::Manifest(error.to_string()))?;
    let elaborated_encryption = encryption
        .elaborate(&bindings)
        .map_err(|error| DiamondNoiseError::Elaboration(error.to_string()))?;
    let symbolic_manifest =
        export_symbolic_manifest(encryption_id.clone(), &elaborated_encryption)?;

    let decryption = compiler.build_decryption(circuit, instance, encryption_id.clone())?.graph;
    let artifact_manifests = BTreeMap::from([(encryption_id, artifact_manifest)]);
    let elaborated_decryption = decryption
        .elaborate_with_manifests(&bindings, &artifact_manifests, &[symbolic_manifest])
        .map_err(|error| DiamondNoiseError::Elaboration(error.to_string()))?;
    // Use the all-one satisfying witness for correctness simulation. Taking
    // a max over every digit branch would include invalid witnesses whose
    // plaintext gadget term is intentionally Large.
    let selections = all_one_witness_selections(&elaborated_decryption, compiler)?;
    let report = simulate_with_selection_values(&elaborated_decryption, &selections)
        .map_err(|error| DiamondNoiseError::Simulation(error.to_string()))?;
    let estimate = report
        .outputs
        .get(NOISY_PLAINTEXT_OUTPUT)
        .cloned()
        .ok_or(DiamondNoiseError::MissingNoisyPlaintext)?;
    // The runtime decoder accepts the closed interval beginning at floor(q/4),
    // so correctness requires the error envelope to remain strictly below it.
    let threshold = BigDecimal::from(&compiler.config.modulus / 4);
    let within_threshold = estimate.noise.as_ref().is_some_and(|noise| noise.bound < threshold);
    let final_decode = DiamondDecodeNoiseReport { estimate, threshold, within_threshold };
    Ok(DiamondNoiseSimulation { report, final_decode })
}

fn all_one_witness_selections(
    graph: &mxx_ir_symbolic::ElaboratedGraph,
    compiler: &DiamondWeCompiler,
) -> Result<BTreeMap<ScopedWireRef, u64>, DiamondNoiseError> {
    let all_ones = 1u64
        .checked_shl(compiler.config.batch_bits as u32)
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| DiamondNoiseError::Simulation("witness digit exceeds u64".to_owned()))?;
    let scope = graph.source.root_scope();
    let mut matrix_values = BTreeMap::<WireRef, u64>::new();
    let mut int_values = BTreeMap::<WireRef, u64>::new();
    let mut bool_values = BTreeMap::<WireRef, bool>::new();
    for node in scope.nodes() {
        let output = || {
            node.output(0)
                .and_then(|value| scope.wire_ref(&value))
                .ok_or_else(|| DiamondNoiseError::Simulation("missing selector wire".to_owned()))
        };
        let arguments = scope.arguments(node).unwrap_or_default();
        match node.kind() {
            NodeKind::Input { name, .. } if name.starts_with("witness-digit-") => {
                matrix_values.insert(output()?, all_ones);
            }
            NodeKind::ExtractCoefficient { position }
                if position
                    .evaluate(&graph.bindings)
                    .is_ok_and(|position| position == 0.into()) =>
            {
                if let Some(value) = arguments.first().and_then(|wire| matrix_values.get(wire)) {
                    int_values.insert(output()?, *value);
                }
            }
            NodeKind::BitExtract { bit } => {
                let bit: usize = bit
                    .evaluate(&graph.bindings)
                    .map_err(|error| DiamondNoiseError::Simulation(error.to_string()))?
                    .try_into()
                    .map_err(|_| DiamondNoiseError::Simulation("invalid witness bit".to_owned()))?;
                if let Some(value) = arguments.first().and_then(|wire| int_values.get(wire)) {
                    bool_values.insert(output()?, ((value >> bit) & 1) != 0);
                }
            }
            NodeKind::BoolToInt => {
                if let Some(value) = arguments.first().and_then(|wire| bool_values.get(wire)) {
                    int_values.insert(output()?, u64::from(*value));
                }
            }
            _ => {}
        }
    }
    Ok(int_values
        .into_iter()
        .map(|(wire, value)| (ScopedWireRef { scope: FrozenGraphScopeId::Root, wire }, value))
        .collect())
}

fn export_symbolic_manifest(
    production: ProductionId,
    elaborated: &mxx_ir_symbolic::ElaboratedGraph,
) -> Result<mxx_ir_symbolic::manifest::Manifest, DiamondNoiseError> {
    let artifacts = elaborated
        .outputs
        .iter()
        .map(|(name, reference)| {
            let wire = elaborated
                .wire(reference)
                .ok_or_else(|| DiamondNoiseError::Manifest(format!("missing output {name}")))?;
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
        .collect::<Result<BTreeMap<_, _>, DiamondNoiseError>>()?;
    mxx_ir_symbolic::manifest::export_manifest(
        production,
        &artifacts,
        &elaborated.atoms,
        &elaborated.expressions,
        &elaborated.preimage_relations,
        elaborated.assumption_digest,
    )
    .map_err(|error| DiamondNoiseError::Manifest(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diamond::{DiamondWeCompiler, DiamondWeConfig};
    use mxx_ir_core::RealExpr;
    use mxx_primitives::poly::dcrt::poly::DCRTPoly;

    #[test]
    fn simulation_uses_the_manifest_linked_decryption_graph() {
        let compiler = DiamondWeCompiler::new(DiamondWeConfig {
            modulus: 1_048_573.into(),
            ring_dimension: 8,
            input_count: 1,
            digit_base: 2,
            batch_bits: 1,
            gadget_base: 16.into(),
            digit_count: 5,
            trapdoor_sigma: RealExpr::from_f64_exact(4.578).unwrap(),
            error_sigma: RealExpr::from_integer(0),
            bgg_tag: b"diamond-noise-test".to_vec(),
        })
        .unwrap();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let input = circuit.input(1);
        circuit.output([input]);
        let simulation = simulate_diamond_noise(&compiler, &circuit, &[]).unwrap();
        // The q/2-encoded plaintext is the intended Large signal. Correctness
        // is governed by the bounded noise envelope around that signal.
        assert!(simulation.final_decode.estimate.has_signal);
        assert_eq!(simulation.final_decode.threshold, BigDecimal::from(262_143u64));
        assert!(simulation.final_decode.estimate.noise.is_some());
    }

    #[test]
    fn simulation_tracks_noise_for_a_batched_witness_gate() {
        let compiler = DiamondWeCompiler::new(DiamondWeConfig {
            modulus: 1_048_573.into(),
            ring_dimension: 8,
            input_count: 1,
            digit_base: 4,
            batch_bits: 2,
            gadget_base: 16.into(),
            digit_count: 5,
            trapdoor_sigma: RealExpr::from_f64_exact(4.578).unwrap(),
            error_sigma: RealExpr::from_integer(1),
            bgg_tag: b"diamond-noise-batched-test".to_vec(),
        })
        .unwrap();
        let mut circuit = PolyCircuit::<DCRTPoly>::new();
        let inputs = circuit.input(2);
        let output = circuit.and_gate(inputs.at(0), inputs.at(1));
        circuit.output([output]);
        let simulation = simulate_diamond_noise(&compiler, &circuit, &[]).unwrap();
        let noise = simulation.final_decode.estimate.noise.as_ref().expect("bounded noise");
        assert!(simulation.final_decode.estimate.has_signal);
        assert!(noise.bound > BigDecimal::from(0));
    }
}
