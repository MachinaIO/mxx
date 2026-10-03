//! Helpers of the integration targets that export the Lean correctness claim of the workflow
//! they execute.
//!
//! A target that runs a recorded parameter set reads its row from
//! `lean/<target>/parameters.csv` and writes the claim of the executed graphs to
//! `lean/<target>/<parameter id>/generated`.

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
use std::{collections::BTreeMap, path::PathBuf};

fn lean_directory(target: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lean").join(target)
}

/// Row `id` of `lean/<target>/parameters.csv`, as column name to value.
pub fn parameter_row(target: &str, id: &str) -> BTreeMap<String, String> {
    let path = lean_directory(target).join("parameters.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    let mut lines = text.lines();
    let header = lines.next().expect("parameter CSV header").split(',').collect::<Vec<_>>();
    let rows = lines
        .map(|line| {
            header.iter().map(|name| name.to_string()).zip(line.split(',').map(str::to_owned))
        })
        .map(BTreeMap::from_iter)
        .filter(|row: &BTreeMap<String, String>| row["parameter_id"] == id)
        .collect::<Vec<_>>();
    let [row] = <[_; 1]>::try_from(rows)
        .unwrap_or_else(|_| panic!("{} must have exactly one row {id}", path.display()));
    row
}

/// A numeric column of a parameter row.
pub fn column<T: std::str::FromStr>(row: &BTreeMap<String, String>, name: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    row.get(name)
        .unwrap_or_else(|| panic!("parameter column {name} is missing"))
        .parse()
        .unwrap_or_else(|error| panic!("parameter column {name} is malformed: {error:?}"))
}

/// The directory receiving the generated modules of parameter set `id`.
pub fn generated_directory(target: &str, id: &str) -> PathBuf {
    lean_directory(target).join(id).join("generated")
}

/// The protocol stage `id` of `graph`. Each artifact input is linked to the stage whose
/// production it reads, and each plain input named in `links` to the given producer output,
/// as the workflow passes it.
pub fn stage(
    id: &str,
    graph: &Graph,
    productions: &BTreeMap<ProductionId, StageId>,
    links: &BTreeMap<String, (StageId, String)>,
) -> ProtocolStage {
    let bindings = graph
        .root_scope()
        .nodes()
        .iter()
        .filter_map(|node| match node.kind() {
            NodeKind::Input { name, artifact: Some(artifact), .. } => Some(ArtifactBinding {
                consumer_input: StageInputName(name.clone()),
                producer_stage: productions
                    .get(&artifact.production_id)
                    .unwrap_or_else(|| panic!("artifact input {name} reads an unknown production"))
                    .clone(),
                producer_output: ArtifactName(artifact.artifact_name.clone()),
            }),
            NodeKind::Input { name, artifact: None, .. } => {
                links.get(name).map(|(producer, output)| ArtifactBinding {
                    consumer_input: StageInputName(name.clone()),
                    producer_stage: producer.clone(),
                    producer_output: ArtifactName(output.clone()),
                })
            }
            _ => None,
        })
        .collect();
    ProtocolStage { id: StageId(id.to_owned()), graph: graph.clone(), bindings }
}

/// The names of the root inputs of `graph` that end with `suffix`.
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

/// An external input: its contract and every destination it feeds.
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
