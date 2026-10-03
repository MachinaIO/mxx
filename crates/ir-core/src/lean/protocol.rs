//! Export a protocol declaration's correctness claim: [`export`] writes every Lean module of it.
use crate::{
    lean::{
        LeanArtifact,
        claim::{self, ClaimRoot, ExternalInput, InputContract, Link, LinkedClaim, Port},
    },
    protocol::{
        ComparatorSpec, InputValueContract, ProtocolDecl, ProtocolInputDestination, StageId,
    },
};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

pub struct ExportedRoots {
    pub stages: BTreeMap<StageId, LeanArtifact>,
    pub requirements: Vec<LeanArtifact>,
    pub ideal: LeanArtifact,
}

/// Errors at the protocol/Lean export boundary.  These are deliberately
/// distinct from GPU operation or placement errors.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum ProtocolExportError {
    #[error("unsupported external input contract variant: {variant}")]
    UnsupportedExternalInputContract { variant: &'static str },
    #[error("protocol export failed: {0}")]
    Invalid(String),
}

/// Writes every Lean module of `protocol`'s correctness claim into `directory`: one module per
/// stage (`Stage_<id>`), per requirement (`Requirement_<i>`), and for the ideal graph (`Ideal`),
/// the gadget layouts they use (`Backend`), and the linked claim (`Claim`). The claim is stated at
/// the declaration's parameter bindings and failure probability; a proof package imports `Claim`
/// and proves `GeneratedClaim.CorrectnessClaim`.
pub fn export(
    protocol: &ProtocolDecl,
    directory: &std::path::Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use crate::{
        artifact::export_validated_manifest,
        lean::{BackendLayout, ExportOptions, export},
        node::NodeKind,
        validate_with_manifests,
    };
    use std::fs;
    let declaration = protocol;
    let bindings = &declaration.bindings;
    declaration.validate()?;
    // Validate each stage against the manifests of the productions its artifact inputs name,
    // exported from the producer stages, which precede their consumers.
    let mut validated = BTreeMap::new();
    for stage in declaration.stages() {
        let mut manifests = BTreeMap::new();
        for binding in &stage.bindings {
            let artifact =
                stage.graph.root_scope().nodes().iter().find_map(|node| match node.kind() {
                    NodeKind::Input { name, artifact: Some(artifact), .. }
                        if *name == binding.consumer_input.0 =>
                    {
                        Some(artifact)
                    }
                    _ => None,
                });
            if let Some(artifact) = artifact {
                let producer = validated
                    .get(&binding.producer_stage)
                    .ok_or("an artifact producer must precede its consumer")?;
                manifests.insert(
                    artifact.production_id.clone(),
                    export_validated_manifest(artifact.production_id.clone(), producer)?,
                );
            }
        }
        validated
            .insert(stage.id.clone(), validate_with_manifests(&stage.graph, bindings, &manifests)?);
    }
    let mut graphs = declaration
        .stages()
        .iter()
        .map(|stage| {
            Ok((
                stage_module_name(&stage.id)?,
                validated.remove(&stage.id).expect("validated stage"),
            ))
        })
        .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
    for (index, requirement) in declaration.bundle.requirements.iter().enumerate() {
        graphs.push((
            format!("Requirement_{index}"),
            crate::validate(requirement.graph(), bindings)?,
        ));
    }
    graphs.push(("Ideal".into(), crate::validate(declaration.bundle.ideal.graph(), bindings)?));
    fs::create_dir_all(directory)?;
    let mut generated = BTreeMap::new();
    let mut layouts = BTreeMap::<_, BackendLayout>::new();
    for (name, graph) in graphs {
        let artifact = export(
            &graph,
            &ExportOptions {
                namespace: name.clone(),
                module_name: name.clone(),
                sampling_tape: declaration.failure_probability_log2.is_some(),
                ..ExportOptions::default()
            },
        )?;
        for layout in &artifact.backend_layouts {
            let key = (layout.modulus.clone(), layout.ring_dimension);
            let merged = match layouts.get(&key) {
                Some(previous) => previous.merge(layout).ok_or_else(|| {
                    format!(
                        "graphs disagree on the gadget layout of ring ({}, {})",
                        layout.modulus, layout.ring_dimension
                    )
                })?,
                None => layout.clone(),
            };
            layouts.insert(key, merged);
        }
        fs::write(directory.join(format!("{name}.lean")), &artifact.source)?;
        generated.insert(name, artifact);
    }
    let layouts = layouts.into_values().collect::<Vec<_>>();
    fs::write(
        directory.join(format!("{}.lean", claim::BACKEND_MODULE)),
        super::backend::render_backend(&layouts),
    )?;
    let roots = ExportedRoots {
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
    fs::write(directory.join("Claim.lean"), assemble_claim(protocol, &roots)?)?;
    Ok(())
}

fn stage_module_name(stage: &StageId) -> Result<String, String> {
    if stage.0.is_empty() ||
        !stage.0.bytes().all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(format!(
            "invalid Lean export stage ID {:?}: expected nonempty ASCII letters, digits, or underscores",
            stage.0
        ));
    }
    Ok(format!("Stage_{}", stage.0))
}

fn input_contract(value: &InputValueContract) -> Result<InputContract, ProtocolExportError> {
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
        InputValueContract::MatrixExact { .. } => {
            return Err(ProtocolExportError::UnsupportedExternalInputContract {
                variant: "MatrixExact",
            })
        }
        InputValueContract::MatrixBounded { .. } => {
            return Err(ProtocolExportError::UnsupportedExternalInputContract {
                variant: "MatrixBounded",
            })
        }
        InputValueContract::MatrixLarge { .. } => {
            return Err(ProtocolExportError::UnsupportedExternalInputContract {
                variant: "MatrixLarge",
            })
        }
        InputValueContract::Trapdoor { .. } => {
            return Err(ProtocolExportError::UnsupportedExternalInputContract {
                variant: "Trapdoor",
            })
        }
    })
}

fn input_contracts(
    contract: &crate::protocol::InputContract,
    bindings: &[crate::protocol::ProtocolInputBinding],
) -> Result<Vec<InputContract>, ProtocolExportError> {
    let mut contracts = BTreeMap::new();
    for entry in &contract.inputs {
        if contracts.insert(&entry.id, &entry.value).is_some() {
            return Err(ProtocolExportError::Invalid("duplicate external input contract ID".into()));
        }
    }
    let mut external_ids = BTreeSet::new();
    let predicates = bindings
        .iter()
        .map(|binding| {
            if !external_ids.insert(&binding.input) {
                return Err(ProtocolExportError::Invalid("duplicate external input ID".into()));
            }
            input_contract(contracts.remove(&binding.input).ok_or_else(|| {
                ProtocolExportError::Invalid("missing external input contract".into())
            })?)
        })
        .collect::<Result<Vec<_>, ProtocolExportError>>()?;
    if !contracts.is_empty() {
        return Err(ProtocolExportError::Invalid("unknown external input contract ID".into()));
    }
    Ok(predicates)
}

/// Convert protocol declaration identities to the generic linked-graph claim.
pub fn assemble_claim(
    declaration: &ProtocolDecl,
    roots: &ExportedRoots,
) -> Result<String, ProtocolExportError> {
    declaration.validate().map_err(|error| ProtocolExportError::Invalid(error.to_string()))?;
    let bundle = &declaration.bundle;
    let mut positions = BTreeMap::new();
    let mut entries = Vec::new();
    for (index, stage) in bundle.workflow.stages.iter().enumerate() {
        if positions.insert(stage.id.clone(), index).is_some() {
            return Err(ProtocolExportError::Invalid("duplicate workflow stage".into()));
        }
        entries.push(ClaimRoot {
            graph: &stage.graph,
            artifact: roots
                .stages
                .get(&stage.id)
                .ok_or_else(|| ProtocolExportError::Invalid("missing generated stage".into()))?,
            field: format!("stage_{index}"),
        });
    }
    if roots.stages.len() != entries.len() ||
        roots.requirements.len() != bundle.requirements.len() ||
        bundle.precondition_spec.requirement_outputs.len() != bundle.requirements.len()
    {
        return Err(ProtocolExportError::Invalid("generated root count mismatch".into()));
    }
    let requirement_start = entries.len();
    for (index, (requirement, artifact)) in
        bundle.requirements.iter().zip(&roots.requirements).enumerate()
    {
        entries.push(ClaimRoot {
            graph: requirement.graph(),
            artifact,
            field: format!("requirement_{index}"),
        });
    }
    let ideal_position = entries.len();
    entries.push(ClaimRoot {
        graph: bundle.ideal.graph(),
        artifact: &roots.ideal,
        field: "ideal".into(),
    });
    let position = |stage: &StageId| {
        positions
            .get(stage)
            .copied()
            .ok_or_else(|| ProtocolExportError::Invalid("unknown stage".into()))
    };
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
                                return Err(ProtocolExportError::Invalid(
                                    "unknown requirement destination".into(),
                                ));
                            }
                            (requirement_start + requirement, input.clone())
                        }
                        ProtocolInputDestination::Ideal { input } => {
                            (ideal_position, input.clone())
                        }
                    };
                    Ok(Port { root, name })
                })
                .collect::<Result<Vec<_>, ProtocolExportError>>()?;
            Ok(ExternalInput { contract, destinations })
        })
        .collect::<Result<Vec<_>, ProtocolExportError>>()?;
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
        return Err(ProtocolExportError::Invalid("unsupported comparator".into()));
    };
    if endpoints.len() != 1 || bundle.endpoints.entries.len() != 1 {
        return Err(ProtocolExportError::Invalid(
            "a claim currently requires exactly one compared endpoint".into(),
        ));
    }
    let endpoint = &bundle.endpoints.entries[0];
    let comparison = &endpoints[0];
    if comparison.endpoint != endpoint.spec ||
        comparison.actual_input != endpoint.workflow_output.output ||
        comparison.ideal_input != endpoint.ideal_output
    {
        return Err(ProtocolExportError::Invalid("comparator endpoint mismatch".into()));
    }
    let actual_position = position(&endpoint.workflow_output.stage)?;
    entries[actual_position]
        .artifact
        .root
        .outputs
        .get(&endpoint.workflow_output.output)
        .ok_or_else(|| ProtocolExportError::Invalid("missing actual endpoint".into()))?;
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
        endpoint: claim::Endpoint::Exact,
        failure_probability_log2: declaration.failure_probability_log2,
    };
    claim::assemble_claim(&claim, &declaration.bindings).map_err(ProtocolExportError::Invalid)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        IntExpr,
        protocol::{InputContractEntry, ProtocolInputBinding, ProtocolInputId},
        types::MatrixType,
    };
    #[test]
    fn test_stage_module_name_accepts_ascii_suffixes() {
        for name in ["encrypt", "decoder_stage", "Stage42", "123", "match", "_"] {
            assert_eq!(stage_module_name(&StageId(name.into())).unwrap(), format!("Stage_{name}"));
        }
    }

    #[test]
    fn input_contract_mapping_uses_exact_ids_and_coverage() {
        let id = ProtocolInputId::from("raw_bits");
        let bindings = vec![ProtocolInputBinding { input: id.clone(), destinations: vec![] }];
        let entry = InputContractEntry {
            id,
            name: "not_the_identity".into(),
            value: InputValueContract::IntegerRange { lower: 0.into(), upper: 1.into() },
        };
        let mut contract = crate::protocol::InputContract { inputs: vec![entry.clone()] };
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

        let unsupported = InputValueContract::MatrixLarge {
            matrix_type: MatrixType {
                ring: crate::ring::test_ring(17, 8),
                rows: 1.into(),
                columns: 1.into(),
            },
        };
        assert!(matches!(
            input_contract(&unsupported),
            Err(ProtocolExportError::UnsupportedExternalInputContract { variant: "MatrixLarge" })
        ));
    }
}
