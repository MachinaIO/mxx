//! WE adapters for application-independent IR Lean export.
pub mod check;
pub mod diamond;
pub mod numeric;

use crate::WitnessEncryptionProtocolDecl;
use mxx_correctness::{ComparatorSpec, InputValueContract, ProtocolInputDestination, StageId};
use mxx_ir_core::{
    ParamEnv,
    lean::{
        LeanArtifact,
        claim::{
            self, ClaimBackend, ClaimRoot, ClaimSemantics, ExternalInput, InputContract, Link,
            LinkedClaim, Port,
        },
    },
};
use std::collections::{BTreeMap, BTreeSet};

pub struct ExportedRoots {
    pub stages: BTreeMap<StageId, LeanArtifact>,
    pub requirements: Vec<LeanArtifact>,
    pub ideal: LeanArtifact,
}

/// Export every graph and the linked claim for this exact declaration and backend.
pub fn export_claim(
    protocol: &WitnessEncryptionProtocolDecl,
    bindings: &ParamEnv,
    backend: &mxx_runtime::lean::LeanBackendArtifact,
    manifests: &BTreeMap<mxx_ir_core::artifact::ProductionId, mxx_ir_core::artifact::Manifest>,
    directory: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use mxx_ir_core::{
        lean::{ExportOptions, export},
        validate_with_manifests,
    };
    use std::fs;
    let declaration = protocol.protocol();
    let mut graphs = declaration
        .stages()
        .iter()
        .map(|stage| (format!("Stage_{}", stage.id.0), &stage.graph))
        .collect::<Vec<_>>();
    graphs.extend(
        declaration
            .bundle
            .requirements
            .iter()
            .enumerate()
            .map(|(index, requirement)| (format!("Requirement_{index}"), &requirement.graph)),
    );
    graphs.push(("Ideal".into(), &declaration.bundle.ideal.graph));
    let mut generated = BTreeMap::new();
    for (name, graph) in graphs {
        let validated = validate_with_manifests(graph, bindings, manifests)?;
        let artifact = export(
            &validated,
            &ExportOptions {
                namespace: name.clone(),
                module_name: name.clone(),
                backend_layouts: backend.exporter_bindings(),
                ..ExportOptions::default()
            },
        )?;
        fs::write(directory.join(format!("{name}.lean")), &artifact.source)?;
        generated.insert(name, artifact);
    }
    let roots = crate::lean::ExportedRoots {
        stages: declaration
            .stages()
            .iter()
            .map(|stage| {
                (
                    stage.id.clone(),
                    generated.remove(&format!("Stage_{}", stage.id.0)).expect("exported stage"),
                )
            })
            .collect(),
        requirements: (0..declaration.bundle.requirements.len())
            .map(|index| {
                generated.remove(&format!("Requirement_{index}")).expect("exported requirement")
            })
            .collect(),
        ideal: generated.remove("Ideal").expect("exported ideal"),
    };
    let claim = crate::lean::assemble_claim(protocol, &roots, bindings, backend)?;
    fs::write(directory.join("Claim.lean"), claim)?;
    Ok(())
}

fn input_contract(value: &InputValueContract) -> Result<InputContract, String> {
    Ok(match value {
        InputValueContract::IntegerRange { lower, upper } => {
            InputContract::IntegerRange { lower: lower.clone(), upper: upper.clone() }
        }
        InputValueContract::Boolean => InputContract::Boolean,
        InputValueContract::Bytes { length } => InputContract::Bytes { length: length.clone() },
        InputValueContract::Family { count, element } => InputContract::Family {
            count: count.clone(),
            element: Box::new(input_contract(element)?),
        },
        _ => return Err("unsupported external input contract variant".into()),
    })
}

fn input_contracts(
    contract: &mxx_correctness::InputContract,
    bindings: &[mxx_correctness::ProtocolInputBinding],
) -> Result<Vec<InputContract>, String> {
    let mut contracts = BTreeMap::new();
    for entry in &contract.inputs {
        if contracts.insert(&entry.id, &entry.value).is_some() {
            return Err("duplicate external input contract ID".into());
        }
    }
    let mut external_ids = BTreeSet::new();
    let predicates = bindings
        .iter()
        .map(|binding| {
            if !external_ids.insert(&binding.input) {
                return Err("duplicate external input ID".into());
            }
            input_contract(
                contracts.remove(&binding.input).ok_or("missing external input contract")?,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    if !contracts.is_empty() {
        return Err("unknown external input contract ID".into());
    }
    Ok(predicates)
}

/// Convert WE declaration identities to the generic linked-graph claim.
pub fn assemble_claim(
    declaration: &WitnessEncryptionProtocolDecl,
    roots: &ExportedRoots,
    bindings: &ParamEnv,
    backend: &mxx_runtime::lean::LeanBackendArtifact,
) -> Result<String, String> {
    let bundle = &declaration.protocol().bundle;
    let mut positions = BTreeMap::new();
    let mut entries = Vec::new();
    for (index, stage) in bundle.workflow.stages.iter().enumerate() {
        if positions.insert(stage.id.clone(), index).is_some() {
            return Err("duplicate workflow stage".into());
        }
        entries.push(ClaimRoot {
            graph: &stage.graph,
            artifact: roots.stages.get(&stage.id).ok_or("missing generated stage")?,
            field: format!("stage_{index}"),
        });
    }
    if roots.stages.len() != entries.len() ||
        roots.requirements.len() != bundle.requirements.len() ||
        bundle.precondition_spec.requirement_outputs.len() != bundle.requirements.len()
    {
        return Err("generated root count mismatch".into());
    }
    let requirement_start = entries.len();
    for (index, (requirement, artifact)) in
        bundle.requirements.iter().zip(&roots.requirements).enumerate()
    {
        entries.push(ClaimRoot {
            graph: &requirement.graph,
            artifact,
            field: format!("requirement_{index}"),
        });
    }
    let ideal_position = entries.len();
    entries.push(ClaimRoot {
        graph: &bundle.ideal.graph,
        artifact: &roots.ideal,
        field: "ideal".into(),
    });
    let position =
        |stage: &StageId| positions.get(stage).copied().ok_or_else(|| "unknown stage".to_string());
    let contracts = input_contracts(&bundle.input_contract, &bundle.input_bindings)?;
    let externals = bundle
        .input_bindings
        .iter()
        .zip(contracts)
        .map(|(binding, contract)| {
            let destinations = binding
                .destinations
                .iter()
                .map(|destination| {
                    let (root, name) = match destination {
                        ProtocolInputDestination::WorkflowStage { stage, input } => {
                            (position(stage)?, input.0.clone())
                        }
                        ProtocolInputDestination::Requirement { requirement, input } => {
                            if *requirement >= roots.requirements.len() {
                                return Err("unknown requirement destination".into());
                            }
                            (requirement_start + requirement, input.clone())
                        }
                        ProtocolInputDestination::Ideal { input } => {
                            (ideal_position, input.clone())
                        }
                    };
                    Ok(Port { root, name })
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(ExternalInput { contract, destinations })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut links = Vec::new();
    for (index, stage) in bundle.workflow.stages.iter().enumerate() {
        for binding in &stage.bindings {
            links.push(Link {
                producer: Port {
                    root: position(&binding.producer_stage)?,
                    name: binding.producer_output.0.clone(),
                },
                consumer: Port { root: index, name: binding.consumer_input.0.clone() },
            });
        }
    }
    let ComparatorSpec::Equality { endpoints } = &bundle.comparator else {
        return Err("unsupported comparator".into());
    };
    if endpoints.len() != 1 ||
        bundle.endpoints.entries.len() != 1 ||
        bundle.operational_decoder_targets.len() != 1
    {
        return Err("WE claim currently requires one exact operational endpoint".into());
    }
    let endpoint = &bundle.endpoints.entries[0];
    let comparison = &endpoints[0];
    if comparison.endpoint != endpoint.spec ||
        comparison.actual_input != endpoint.workflow_output.output ||
        comparison.ideal_input != endpoint.ideal_output
    {
        return Err("comparator endpoint mismatch".into());
    }
    let actual_position = position(&endpoint.workflow_output.stage)?;
    let actual = entries[actual_position]
        .artifact
        .root
        .outputs
        .get(&endpoint.workflow_output.output)
        .ok_or("missing actual endpoint")?;
    let target = &bundle.operational_decoder_targets[0];
    if target.decoder_stage != endpoint.workflow_output.stage ||
        target.decoder_node != actual.wire.node ||
        !matches!(target.kind, mxx_correctness::OperationalDecoderKind::BooleanInterval)
    {
        return Err("operational decoder does not identify the actual endpoint".into());
    }
    let claim = LinkedClaim {
        roots: entries,
        externals,
        links,
        requirements: bundle
            .precondition_spec
            .requirement_outputs
            .iter()
            .enumerate()
            .map(|(index, name)| Port { root: requirement_start + index, name: name.clone() })
            .collect(),
        actual: Port { root: actual_position, name: endpoint.workflow_output.output.clone() },
        ideal: Port { root: ideal_position, name: endpoint.ideal_output.clone() },
        residual: Port {
            root: position(&target.residual_stage)?,
            name: target.residual_output.clone(),
        },
    };
    claim::assemble_claim(
        &claim,
        bindings,
        &ClaimBackend {
            module_name: backend.module_name(),
            context_name: backend.context_name(),
            layouts: &backend.exporter_bindings(),
        },
        &ClaimSemantics {
            imports: &["Decoder"],
            hash_model_type: "MxxRuntime.HashModel",
            centered_lift: "Mxx.Primitives.centeredLift",
            message_center: "MxxWe.messageCenter",
            decoder_radius: "MxxWe.decoderRadius",
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_correctness::{InputContractEntry, ProtocolInputBinding, ProtocolInputId};
    use mxx_ir_core::IntExpr;

    #[test]
    fn input_contract_mapping_uses_exact_ids_and_coverage() {
        let id = ProtocolInputId::from("raw_bits");
        let bindings = vec![ProtocolInputBinding { input: id.clone(), destinations: vec![] }];
        let entry = InputContractEntry {
            id,
            name: "not_the_identity".into(),
            value: InputValueContract::IntegerRange { lower: 0.into(), upper: 1.into() },
        };
        let mut contract = mxx_correctness::InputContract { inputs: vec![entry.clone()] };
        assert!(
            matches!(&input_contracts(&contract, &bindings).unwrap()[0], InputContract::IntegerRange { lower, upper } if *lower == IntExpr::constant(0) && *upper == IntExpr::constant(1))
        );
        contract.inputs.push(entry);
        assert!(input_contracts(&contract, &bindings).is_err());
        contract.inputs.clear();
        assert!(input_contracts(&contract, &bindings).is_err());
        contract.inputs.push(InputContractEntry {
            id: ProtocolInputId::from("unknown"),
            name: "raw_bits".into(),
            value: InputValueContract::Boolean,
        });
        assert!(input_contracts(&contract, &bindings).is_err());
        contract.inputs[0].id = bindings[0].input.clone();
        assert!(input_contracts(&contract, &[bindings[0].clone(), bindings[0].clone()]).is_err());
        assert!(input_contracts(&contract, &[]).is_err());
    }
}
