//! Building blocks of the closed-protocol claim of recorded stage graphs.
//!
//! A workflow that records the validated graph of every stage it executes states its correctness
//! claim as a [`mxx_ir_core::protocol::ProtocolDecl`]: each recorded graph becomes a protocol
//! stage ([`stage`]) whose artifact inputs read the stages that produced them, and the plain graph
//! inputs are fed by external inputs ([`external`]) through their destinations ([`to_stage`]).

use crate::api::ApiError;
use mxx_ir_core::{
    Graph,
    artifact::ProductionId,
    node::NodeKind,
    protocol::{
        ArtifactBinding, ArtifactName, InputContractEntry, InputValueContract,
        ProtocolInputBinding, ProtocolInputDestination, ProtocolInputId, ProtocolStage, StageId,
        StageInputName,
    },
};
use std::collections::BTreeMap;

/// The protocol stage `id` of `graph`. Each artifact input is linked to the stage whose
/// production it reads (`productions`), and each plain input named in `links` to the given
/// producer output, as the workflow passes it. An artifact input of an unknown production is an
/// error.
pub fn stage(
    id: &str,
    graph: &Graph,
    productions: &BTreeMap<ProductionId, StageId>,
    links: &BTreeMap<String, (StageId, String)>,
) -> Result<ProtocolStage, ApiError> {
    let bindings = graph
        .root_scope()
        .nodes()
        .iter()
        .filter_map(|node| match node.kind() {
            NodeKind::Input { name, artifact: Some(artifact), .. } => {
                Some(match productions.get(&artifact.production_id) {
                    Some(producer) => Ok(ArtifactBinding {
                        consumer_input: StageInputName(name.clone()),
                        producer_stage: producer.clone(),
                        producer_output: ArtifactName(artifact.artifact_name.clone()),
                    }),
                    None => Err(ApiError::input(format!(
                        "artifact input {name} of stage {id} reads an unknown production"
                    ))),
                })
            }
            NodeKind::Input { name, artifact: None, .. } => {
                links.get(name).map(|(producer, output)| {
                    Ok(ArtifactBinding {
                        consumer_input: StageInputName(name.clone()),
                        producer_stage: producer.clone(),
                        producer_output: ArtifactName(output.clone()),
                    })
                })
            }
            _ => None,
        })
        .collect::<Result<_, _>>()?;
    Ok(ProtocolStage { id: StageId(id.to_owned()), graph: graph.clone(), bindings })
}

/// The names of the plain (non-artifact) root inputs of `graph` that end with `suffix`.
pub fn inputs_ending_with(graph: &Graph, suffix: &str) -> Vec<String> {
    graph
        .root_scope()
        .nodes()
        .iter()
        .filter_map(|node| match node.kind() {
            NodeKind::Input { name, artifact: None, .. } if name.ends_with(suffix) => {
                Some(name.clone())
            }
            _ => None,
        })
        .collect()
}

/// Destination `input` of workflow stage `stage`.
pub fn to_stage(stage: &str, input: &str) -> ProtocolInputDestination {
    ProtocolInputDestination::WorkflowStage {
        stage: StageId(stage.to_owned()),
        input: StageInputName(input.to_owned()),
    }
}

/// An external input named `name`: its contract `value` and every destination it feeds.
pub fn external(
    name: &str,
    value: InputValueContract,
    destinations: Vec<ProtocolInputDestination>,
) -> (InputContractEntry, ProtocolInputBinding) {
    (
        InputContractEntry { id: ProtocolInputId::from(name), name: name.to_owned(), value },
        ProtocolInputBinding { input: ProtocolInputId::from(name), destinations },
    )
}
