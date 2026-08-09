use crate::{
    BundleLeanEmitError, BundleProgramNames, FreshnessError, FreshnessMetadata, GENERATOR_VERSION,
    ProtocolDecl, ProtocolError, ProtocolStage, StageId, emit_closed_protocol_bundle,
    protocol_source_hash, toolkit_hash,
};
use mxx_ir_core::{
    FrozenGraphScopeId, Graph, GraphScope, IntExpr, WireType,
    node::{
        ConcatAxis, ConstantMatrix, HashVariant, IntBinaryOp, IntCompareOp, LoopInputMode,
        MatrixBinaryOp, NodeKind, RealBinaryOp,
    },
    types::MatrixType,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
use thiserror::Error;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EmittedProtocol {
    pub freshness: FreshnessMetadata,
    pub derivation_hash: String,
    pub module_root: String,
    pub lean_name: String,
    pub stage_ids: Vec<String>,
    pub ir: String,
    pub derivation_ir: String,
}

#[derive(Debug, Error)]
pub enum EmitError {
    #[error(transparent)]
    Protocol(#[from] ProtocolError),
    #[error("protocol serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error(transparent)]
    Freshness(#[from] FreshnessError),
    #[error(transparent)]
    Bundle(#[from] BundleLeanEmitError),
    #[error(
        "Lean denotation is not implemented for Graph IR node {node} ({kind}) in scope {scope} of stage {stage}"
    )]
    UnsupportedNode { stage: String, scope: String, node: u64, kind: &'static str },
    #[error("Lean denotation does not yet support child Graph IR scopes")]
    ChildScope,
    #[error("derivation attachment {namespace}.{rule} contains roles from different scopes")]
    DerivationAttachmentScope { namespace: String, rule: String },
    #[error("operational protocol inventory is missing the emitted variant row `{variant}`")]
    MissingOperationalInventory { variant: &'static str },
}

const OPERATIONAL_PROTOCOL_INVENTORY: &str =
    include_str!("../../../docs/correctness/operational-protocol-inventory.md");

fn operational_inventory_key(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::Input { .. } => "Input",
        NodeKind::ConstantInt(_) => "ConstantInt",
        NodeKind::EvaluateInt(_) => "EvaluateInt",
        NodeKind::ConstantReal(_) => "ConstantReal",
        NodeKind::ConstantBool(_) => "ConstantBool",
        NodeKind::ConstantMatrix { value: ConstantMatrix::Zero, .. } => "ConstantMatrix.Zero",
        NodeKind::ConstantMatrix { value: ConstantMatrix::Identity, .. } => {
            "ConstantMatrix.Identity"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::UnitRow { .. }, .. } => {
            "ConstantMatrix.UnitRow"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::UnitColumn { .. }, .. } => {
            "ConstantMatrix.UnitColumn"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { small: false, .. }, .. } => {
            "ConstantMatrix.Gadget"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { small: true, .. }, .. } => {
            "ConstantMatrix.Gadget(small)"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::PowerOfBase { .. }, .. } => {
            "ConstantMatrix.PowerOfBase"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::Rotation { .. }, .. } => {
            "ConstantMatrix.Rotation"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::Polynomial { .. }, .. } => {
            "ConstantMatrix.Polynomial"
        }
        NodeKind::GadgetTrapdoor { .. } => "GadgetTrapdoor",
        NodeKind::TrapdoorPublic => "TrapdoorPublic",
        NodeKind::IntBinary(IntBinaryOp::Add) => "IntBinary.Add",
        NodeKind::IntBinary(IntBinaryOp::Subtract) => "IntBinary.Subtract",
        NodeKind::IntBinary(IntBinaryOp::Multiply) => "IntBinary.Multiply",
        NodeKind::IntBinary(IntBinaryOp::Divide) => "IntBinary.Divide",
        NodeKind::IntBinary(IntBinaryOp::Remainder) => "IntBinary.Remainder",
        NodeKind::IntCompare(IntCompareOp::Equal) => "IntCompare.Equal",
        NodeKind::IntCompare(IntCompareOp::Less) => "IntCompare.Less",
        NodeKind::IntCompare(IntCompareOp::LessEqual) => "IntCompare.LessEqual",
        NodeKind::BitExtract { .. } => "BitExtract",
        NodeKind::IntToReal => "IntToReal",
        NodeKind::BoolToInt => "BoolToInt",
        NodeKind::RealBinary(RealBinaryOp::Add) => "RealBinary.Add",
        NodeKind::RealBinary(RealBinaryOp::Subtract) => "RealBinary.Subtract",
        NodeKind::RealBinary(RealBinaryOp::Multiply) => "RealBinary.Multiply",
        NodeKind::RealBinary(RealBinaryOp::Divide) => "RealBinary.Divide",
        NodeKind::RealSqrt => "RealSqrt",
        NodeKind::MatrixBinary(MatrixBinaryOp::Add) => "MatrixBinary.Add",
        NodeKind::MatrixBinary(MatrixBinaryOp::Subtract) => "MatrixBinary.Subtract",
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) => "MatrixBinary.Multiply",
        NodeKind::MatrixNegate => "MatrixNegate",
        NodeKind::MatrixScale { .. } => "MatrixScale",
        NodeKind::Transpose => "Transpose",
        NodeKind::Slice { .. } => "Slice",
        NodeKind::Tensor => "Tensor",
        NodeKind::Concat { axis: ConcatAxis::Rows } => "Concat.Rows",
        NodeKind::Concat { axis: ConcatAxis::Columns } => "Concat.Columns",
        NodeKind::Concat { axis: ConcatAxis::Diagonal } => "Concat.Diagonal",
        NodeKind::Reshape { .. } => "Reshape",
        NodeKind::UniformResidueSample { .. } => "UniformResidueSample",
        NodeKind::UniformIntervalSample { .. } => "UniformIntervalSample",
        NodeKind::GaussianSample { .. } => "GaussianSample",
        NodeKind::HashSample { variant: HashVariant::Plain, .. } => "HashSample.Plain",
        NodeKind::HashSample { variant: HashVariant::Decomposed, .. } => "HashSample.Decomposed",
        NodeKind::HashSample { variant: HashVariant::SmallDecomposed, .. } => {
            "HashSample.SmallDecomposed"
        }
        NodeKind::TrapdoorSample { .. } => "TrapdoorSample",
        NodeKind::PreimageSample { .. } => "PreimageSample",
        NodeKind::GadgetDecompose { small: false, .. } => "GadgetDecompose(regular)",
        NodeKind::GadgetDecompose { small: true, .. } => "GadgetDecompose(small)",
        NodeKind::ExtractCoefficient { .. } => "ExtractCoefficient",
        NodeKind::ConstantCoefficient { .. } => "ConstantCoefficient",
        NodeKind::ThresholdDecode { output_bool: true, .. } => "ThresholdDecode(bool)",
        NodeKind::ThresholdDecode { output_bool: false, .. } => "ThresholdDecode(int)",
        NodeKind::CrtRecompose { .. } => "CrtRecompose",
        NodeKind::PackPolynomialCoefficients { .. } => "PackPolynomialCoefficients",
        NodeKind::SubgraphCall(_) => "SubgraphCall",
        NodeKind::ParallelLoop(_) => "ParallelLoop",
        NodeKind::SequentialLoop(_) => "SequentialLoop",
        NodeKind::FamilyPack { .. } => "FamilyPack",
        NodeKind::FamilyGetStatic { .. } => "FamilyGetStatic",
        NodeKind::FamilyGetDynamic => "FamilyGetDynamic",
        NodeKind::Select { .. } => "Select",
    }
}

fn require_operational_inventory_row(key: &'static str) -> Result<(), EmitError> {
    let row_prefix = format!("| `{key}` |");
    if OPERATIONAL_PROTOCOL_INVENTORY.contains(&row_prefix) {
        Ok(())
    } else {
        Err(EmitError::MissingOperationalInventory { variant: key })
    }
}

fn validate_operational_inventory(kind: &NodeKind) -> Result<(), EmitError> {
    require_operational_inventory_row(operational_inventory_key(kind))?;
    if let NodeKind::ParallelLoop(spec) = kind {
        for mode in &spec.input_modes {
            require_operational_inventory_row(match mode {
                LoopInputMode::Broadcast => "ParallelLoop.Broadcast",
                LoopInputMode::Zip => "ParallelLoop.Zip",
                LoopInputMode::ZipOffset { .. } => "ParallelLoop.ZipOffset",
            })?;
        }
    }
    Ok(())
}

pub fn emit_protocol_for(
    name: &str,
    protocol: &ProtocolDecl,
    module_root: &str,
    protocol_source_paths: &[&str],
) -> Result<EmittedProtocol, EmitError> {
    protocol.validate()?;
    let normalized = normalized_protocol(protocol)?;
    let canonical = serde_json::to_vec(&normalized)?;
    let workflow_hash = format!("{:x}", Sha256::digest(&canonical));
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("correctness crate is under the workspace crates directory");
    let mut protocol_source_paths =
        protocol_source_paths.iter().map(|path| (*path).to_owned()).collect::<Vec<_>>();
    protocol_source_paths.sort();
    protocol_source_paths.dedup();
    let source_path_refs = protocol_source_paths.iter().map(String::as_str).collect::<Vec<_>>();
    let freshness = FreshnessMetadata {
        generator_version: GENERATOR_VERSION.to_owned(),
        protocol_source_hash: protocol_source_hash(workspace_root, &source_path_refs)?,
        protocol_source_paths,
        workflow_hash,
        toolkit_hash: toolkit_hash(workspace_root)?,
    };
    let lean_name = lean_identifier(name);
    let transport = lean_protocol_transport(&lean_name, protocol)?;
    let derivations = lean_protocol_derivations(&lean_name, protocol)?;
    let derivation_hash = format!("{:x}", Sha256::digest(derivations.as_bytes()));
    let derivation_transport = lean_derivation_transport(&lean_name, protocol)?;
    let ir = format!(
        "-- Generated by {GENERATOR_VERSION}; do not edit.\nimport Mxx.Certificate.Derivation\nimport Mxx.Certificate.Workflow\n\nnamespace {module_root}.Generated.{lean_name}\n\ndef {lean_name}_generatorVersion : String := \"{}\"\n\ndef {lean_name}_protocolSourcePaths : List String := [{}]\n\ndef {lean_name}_protocolSourceHash : String := \"{}\"\n\ndef {lean_name}_workflowHash : String := \"{}\"\n\ndef {lean_name}_toolkitHash : String := \"{}\"\n\ndef {lean_name}_derivationHash : String := \"{derivation_hash}\"\n\n{transport}\n{derivations}\nend {module_root}.Generated.{lean_name}\n",
        freshness.generator_version,
        freshness
            .protocol_source_paths
            .iter()
            .map(|path| lean_string(path))
            .collect::<Vec<_>>()
            .join(", "),
        freshness.protocol_source_hash,
        freshness.workflow_hash,
        freshness.toolkit_hash,
    );
    let derivation_ir = format!(
        "-- Generated by {GENERATOR_VERSION}; do not edit.\nimport Mxx.Certificate.Derivation\n\nnamespace {module_root}.Generated.{lean_name}\n\ndef {lean_name}_derivationHash : String := \"{derivation_hash}\"\n\n{derivation_transport}\n{derivations}\nend {module_root}.Generated.{lean_name}\n"
    );
    let stage_ids =
        topological_stages(protocol).into_iter().map(|stage| stage.id.0.clone()).collect();
    Ok(EmittedProtocol {
        freshness,
        derivation_hash,
        module_root: module_root.to_owned(),
        lean_name,
        stage_ids,
        ir,
        derivation_ir,
    })
}

fn lean_derivation_transport(
    lean_name: &str,
    protocol: &ProtocolDecl,
) -> Result<String, EmitError> {
    let mut definitions = Vec::new();
    for stage in topological_stages(protocol) {
        definitions.push(format!(
            "def {lean_name}_stage_{} : Mxx.Ir.Prog :=\n{}",
            lower_identifier(&stage.id.0),
            lean_program(&stage.id.0, &stage.graph, 2)?
        ));
    }
    definitions.push(format!(
        "def {lean_name}_ideal : Mxx.Ir.Prog :=\n{}",
        lean_program("ideal", &protocol.bundle.ideal.graph, 2)?
    ));
    for (index, requirement) in protocol.bundle.requirements.iter().enumerate() {
        definitions.push(format!(
            "def {lean_name}_requirement_{index} : Mxx.Ir.Prog :=\n{}",
            lean_program(&format!("requirement-{index}"), &requirement.graph, 2)?
        ));
    }
    if let Some(program) = protocol.bundle.comparator.program() {
        definitions.push(format!(
            "def {lean_name}_comparator : Mxx.Ir.Prog :=\n{}",
            lean_program("comparator", &program.graph, 2)?
        ));
    }
    Ok(definitions.join("\n\n") + "\n")
}

fn lean_protocol_derivations(
    lean_name: &str,
    protocol: &ProtocolDecl,
) -> Result<String, EmitError> {
    let mut definitions = Vec::new();
    let mut entries = Vec::new();
    for stage in topological_stages(protocol) {
        let stage_name = lower_identifier(&stage.id.0);
        let name = format!("{lean_name}_stage_{stage_name}_derivation");
        definitions.push(format!(
            "def {name} : Mxx.Certificate.ProgramDerivation :=\n{}",
            lean_program_derivation_with_attachments(
                &stage.id.0,
                &stage.graph,
                Some(&stage.derivation_attachments),
                2,
            )?
        ));
        entries.push(format!("({}, {name})", lean_string(&format!("stage:{}", stage.id.0))));
    }
    let ideal_name = format!("{lean_name}_ideal_derivation");
    definitions.push(format!(
        "def {ideal_name} : Mxx.Certificate.ProgramDerivation :=\n{}",
        lean_program_derivation("ideal", &protocol.bundle.ideal.graph, 2)?
    ));
    entries.push(format!("(\"ideal\", {ideal_name})"));
    for (index, requirement) in protocol.bundle.requirements.iter().enumerate() {
        let name = format!("{lean_name}_requirement_{index}_derivation");
        definitions.push(format!(
            "def {name} : Mxx.Certificate.ProgramDerivation :=\n{}",
            lean_program_derivation(&format!("requirement-{index}"), &requirement.graph, 2)?
        ));
        entries.push(format!("(\"requirement-{index}\", {name})"));
    }
    if let Some(program) = protocol.bundle.comparator.program() {
        let name = format!("{lean_name}_comparator_derivation");
        definitions.push(format!(
            "def {name} : Mxx.Certificate.ProgramDerivation :=\n{}",
            lean_program_derivation("comparator", &program.graph, 2)?
        ));
        entries.push(format!("(\"comparator\", {name})"));
    }
    definitions.push(format!(
        "def {lean_name}_derivations : List (String × Mxx.Certificate.ProgramDerivation) := [{}]",
        entries.join(", ")
    ));
    Ok(definitions.join("\n\n") + "\n")
}
fn lean_protocol_transport(lean_name: &str, protocol: &ProtocolDecl) -> Result<String, EmitError> {
    let mut definitions = Vec::new();
    for stage in topological_stages(protocol) {
        definitions.push(format!(
            "def {lean_name}_stage_{} : Mxx.Ir.Prog :=\n{}",
            lower_identifier(&stage.id.0),
            lean_program(&stage.id.0, &stage.graph, 2)?
        ));
    }
    definitions.push(format!(
        "def {lean_name}_ideal : Mxx.Ir.Prog :=\n{}",
        lean_program("ideal", &protocol.bundle.ideal.graph, 2)?
    ));
    for (index, requirement) in protocol.bundle.requirements.iter().enumerate() {
        definitions.push(format!(
            "def {lean_name}_requirement_{index} : Mxx.Ir.Prog :=\n{}",
            lean_program(&format!("requirement-{index}"), &requirement.graph, 2)?
        ));
    }
    let comparator_program = protocol
        .bundle
        .comparator
        .program()
        .map(|program| {
            definitions.push(format!(
                "def {lean_name}_comparator : Mxx.Ir.Prog :=\n{}",
                lean_program("comparator", &program.graph, 2)?
            ));
            Ok::<_, EmitError>(format!("{lean_name}_comparator"))
        })
        .transpose()?;
    let stage_programs = protocol
        .bundle
        .workflow
        .stages
        .iter()
        .map(|stage| {
            (stage.id.clone(), format!("{lean_name}_stage_{}", lower_identifier(&stage.id.0)))
        })
        .collect();
    let names = BundleProgramNames {
        stage_programs,
        ideal_program: format!("{lean_name}_ideal"),
        requirement_programs: (0..protocol.bundle.requirements.len())
            .map(|index| format!("{lean_name}_requirement_{index}"))
            .collect(),
        comparator_program,
    };
    let bundle = emit_closed_protocol_bundle(&protocol.bundle, &names)?;
    let parameters = protocol.params.iter().map(|parameter| {
        let kind = match parameter.kind {
            crate::ParameterKind::Dimension => ".dimension",
            crate::ParameterKind::Integer => ".integer",
            crate::ParameterKind::Rational => ".rational",
        };
        format!("{{ name := {}, kind := {kind} }}", lean_string(&parameter.name))
    });
    definitions.push(format!(
        "def {lean_name}_protocol : Mxx.Certificate.ClosedProtocolDecl :=\n  {{ parameters := [{}], bundle := {bundle} }}",
        parameters.collect::<Vec<_>>().join(", ")
    ));
    Ok(definitions.join("\n\n") + "\n")
}

fn topological_stages(protocol: &ProtocolDecl) -> Vec<&ProtocolStage> {
    fn visit<'a>(
        id: &StageId,
        stages: &BTreeMap<&'a StageId, &'a ProtocolStage>,
        visited: &mut BTreeSet<StageId>,
        output: &mut Vec<&'a ProtocolStage>,
    ) {
        if !visited.insert(id.clone()) {
            return;
        }
        let stage = stages[id];
        for dependency in stage.bindings.iter().map(|binding| &binding.producer_stage) {
            visit(dependency, stages, visited, output);
        }
        output.push(stage);
    }
    let stages = protocol
        .bundle
        .workflow
        .stages
        .iter()
        .map(|stage| (&stage.id, stage))
        .collect::<BTreeMap<_, _>>();
    let mut output = Vec::new();
    visit(&protocol.bundle.workflow.entrypoint, &stages, &mut BTreeSet::new(), &mut output);
    output
}

fn lean_program(stage: &str, graph: &Graph, indent: usize) -> Result<String, EmitError> {
    let padding = " ".repeat(indent);
    let root = lean_scope(stage, graph, &FrozenGraphScopeId::Root, graph.root_scope(), indent + 4)?;
    let definitions = graph
        .scopes()
        .iter()
        .filter(|(id, _)| !matches!(id, FrozenGraphScopeId::Root))
        .map(|(id, scope)| {
            Ok(format!(
                "{}({},\n{})",
                " ".repeat(indent + 2),
                lean_string(&lean_scope_name(id)),
                lean_scope(stage, graph, id, scope, indent + 4)?
            ))
        })
        .collect::<Result<Vec<_>, EmitError>>()?
        .join(",\n");
    Ok(format!(
        "{padding}{{ root :=\n{root}\n{padding}  definitions := [\n{definitions}\n{padding}  ]\n{padding}}}"
    ))
}

fn lean_program_derivation(stage: &str, graph: &Graph, indent: usize) -> Result<String, EmitError> {
    lean_program_derivation_with_attachments(stage, graph, None, indent)
}

fn lean_program_derivation_with_attachments(
    stage: &str,
    graph: &Graph,
    attachments: Option<&mxx_dsl::FrozenDerivationAttachments>,
    indent: usize,
) -> Result<String, EmitError> {
    let padding = " ".repeat(indent);
    let root = lean_scope_derivation(
        stage,
        &FrozenGraphScopeId::Root,
        graph.root_scope(),
        attachments,
        indent + 4,
    )?;
    let definitions = graph
        .scopes()
        .iter()
        .filter(|(id, _)| !matches!(id, FrozenGraphScopeId::Root))
        .map(|(id, scope)| {
            Ok(format!(
                "{}({},\n{})",
                " ".repeat(indent + 2),
                lean_string(&lean_scope_name(id)),
                lean_scope_derivation(stage, id, scope, attachments, indent + 4)?
            ))
        })
        .collect::<Result<Vec<_>, EmitError>>()?
        .join(",\n");
    Ok(format!(
        "{padding}{{ root :=\n{root}\n{padding}  definitions := [\n{definitions}\n{padding}  ]\n{padding}}}"
    ))
}

const LEAN_LIST_CHUNK_SIZE: usize = 256;

fn lean_list_literal(items: Vec<String>, element_type: &str, padding: &str) -> String {
    if items.is_empty() {
        return "[]".to_owned();
    }
    if items.len() <= LEAN_LIST_CHUNK_SIZE {
        return format!("[\n{}\n{padding}]", items.join(",\n"));
    }
    let chunk_padding = format!("{padding}  ");
    let chunks = items
        .chunks(LEAN_LIST_CHUNK_SIZE)
        .map(|chunk| {
            format!(
                "{chunk_padding}([\n{}\n{chunk_padding}] : List {element_type})",
                chunk.join(",\n")
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");
    format!("List.flatten [\n{chunks}\n{padding}]")
}

fn lean_scope_derivation(
    stage: &str,
    scope_id: &FrozenGraphScopeId,
    scope: &GraphScope,
    attachments: Option<&mxx_dsl::FrozenDerivationAttachments>,
    indent: usize,
) -> Result<String, EmitError> {
    let padding = " ".repeat(indent);
    let child_padding = " ".repeat(indent + 2);
    let steps = lean_list_literal(
        scope
        .nodes()
        .iter()
        .enumerate()
        .map(|(node_id, node)| -> Result<_, EmitError> {
            let arguments = scope
                .arguments(node)
                .expect("frozen scope arguments")
                .iter()
                .map(lean_wire_ref)
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!(
                "{child_padding}{{ sourceNode := {node_id}, rule := {}, arguments := [{arguments}] }}",
                lean_derivation_rule(
                    stage,
                    scope_id,
                    mxx_ir_core::NodeId(node_id as u64),
                    node,
                    scope,
                )?,
            ))
        })
        .collect::<Result<Vec<_>, _>>()?,
        "Mxx.Certificate.NodeDerivation",
        &padding,
    );
    let attachments = attachments
        .into_iter()
        .flat_map(|attachments| attachments.iter())
        .filter_map(|attachment| {
            attachment.roles.first().map(|(_, wire)| (attachment, &wire.scope))
        })
        .filter(|(_, attachment_scope)| *attachment_scope == scope_id)
        .map(|(attachment, attachment_scope)| {
            if attachment.roles.iter().any(|(_, wire)| &wire.scope != attachment_scope) {
                return Err(EmitError::DerivationAttachmentScope {
                    namespace: attachment.namespace.clone(),
                    rule: attachment.rule.clone(),
                });
            }
            let roles = attachment
                .roles
                .iter()
                .map(|(role, wire)| {
                    format!("({}, {})", lean_string(role), lean_wire_ref(&wire.wire))
                })
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!(
                "{child_padding}{{ ownerNamespace := {}, ruleName := {}, roles := [{roles}] }}",
                lean_string(&attachment.namespace),
                lean_string(&attachment.rule),
            ))
        })
        .collect::<Result<Vec<_>, EmitError>>()?
        .join(",\n");
    Ok(format!(
        "{padding}{{ steps := {steps}\n{padding}  attachments := [\n{attachments}\n{padding}] }}"
    ))
}

fn lean_derivation_rule(
    stage: &str,
    scope_id: &FrozenGraphScopeId,
    node_id: mxx_ir_core::NodeId,
    node: &mxx_ir_core::NodeHandle,
    scope: &GraphScope,
) -> Result<String, EmitError> {
    let _ = (stage, scope_id, node_id);
    let rule = match node.kind() {
        NodeKind::Input { .. } => ".input",
        NodeKind::ConstantInt(_) => ".constantInt",
        NodeKind::EvaluateInt(_) => ".evaluateInt",
        NodeKind::ConstantReal(_) => ".constantReal",
        NodeKind::ConstantBool(_) => ".constantBool",
        NodeKind::ConstantMatrix { value: ConstantMatrix::Zero, .. } => ".zeroMatrix",
        NodeKind::ConstantMatrix { value: ConstantMatrix::Identity, .. } => ".identityMatrix",
        NodeKind::ConstantMatrix { value: ConstantMatrix::UnitRow { .. }, .. } => ".unitRowMatrix",
        NodeKind::ConstantMatrix { value: ConstantMatrix::UnitColumn { .. }, .. } => {
            ".unitColumnMatrix"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::Polynomial { .. }, .. } => {
            ".constantMatrix"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { small: false, .. }, .. } => {
            ".gadgetMatrix"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { small: true, .. }, .. } => {
            ".smallGadgetMatrix"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::PowerOfBase { .. }, .. } => {
            ".powerOfBaseMatrix"
        }
        NodeKind::ConstantMatrix { value: ConstantMatrix::Rotation { .. }, .. } => {
            ".rotationMatrix"
        }
        NodeKind::GadgetTrapdoor { .. } => ".gadgetTrapdoor",
        NodeKind::IntToReal => ".intToReal",
        NodeKind::BoolToInt => ".boolToInt",
        NodeKind::IntBinary(_) => ".intBinary",
        NodeKind::RealBinary(_) => ".realBinary",
        NodeKind::RealSqrt => ".realSqrt",
        NodeKind::IntCompare(_) => ".intCompare",
        NodeKind::BitExtract { .. } => ".bitExtract",
        NodeKind::ExtractCoefficient { .. } => ".extractCoefficient",
        NodeKind::ConstantCoefficient { .. } => ".constantCoefficient",
        NodeKind::Select { .. } => ".select",
        NodeKind::UniformResidueSample { .. } => ".uniformResidueSample",
        NodeKind::UniformIntervalSample { .. } => ".uniformIntervalSample",
        NodeKind::GaussianSample { .. } => ".gaussianSample",
        NodeKind::HashSample { .. } => ".hashSample",
        NodeKind::GadgetDecompose { .. } => ".gadgetDecompose",
        NodeKind::TrapdoorSample { .. } => ".trapdoorSample",
        NodeKind::TrapdoorPublic => ".trapdoorPublic",
        NodeKind::PreimageSample { .. } => ".preimageSample",
        NodeKind::MatrixBinary(MatrixBinaryOp::Add) => ".matrixAdd",
        NodeKind::MatrixBinary(MatrixBinaryOp::Subtract) => ".matrixSubtract",
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) => {
            let arguments = scope.arguments(node).expect("frozen multiplication arguments");
            let left = arguments.first().expect("validated multiplication has two arguments");
            let right = arguments.get(1).expect("validated multiplication has two arguments");
            let relation_available = match scope.node(right.node).map(|producer| producer.kind()) {
                Some(NodeKind::GadgetDecompose { .. }) => {
                    scope.node(left.node).is_some_and(|producer| {
                        matches!(
                            producer.kind(),
                            NodeKind::ConstantMatrix { value: ConstantMatrix::Gadget { .. }, .. }
                        )
                    })
                }
                Some(NodeKind::PreimageSample { .. }) => {
                    scope
                        .node(right.node)
                        .and_then(|producer| scope.arguments(producer))
                        .and_then(|preimage_arguments| preimage_arguments.first().copied()) ==
                        Some(*left)
                }
                _ => false,
            };
            if relation_available {
                return Ok(format!(".matrixMultiplyRelation {}", lean_wire_ref(right)));
            }
            ".matrixMultiplyBound"
        }
        NodeKind::MatrixNegate => ".matrixNegate",
        NodeKind::MatrixScale { .. } => ".matrixScale",
        NodeKind::Transpose => ".transpose",
        NodeKind::Slice { .. } => ".slice",
        NodeKind::Tensor => ".tensor",
        NodeKind::Reshape { .. } => ".reshape",
        NodeKind::Concat { .. } => ".concat",
        NodeKind::ThresholdDecode { output_bool: true, .. } => ".thresholdDecodeBool",
        NodeKind::ThresholdDecode { output_bool: false, .. } => ".thresholdDecodeInt",
        NodeKind::CrtRecompose { .. } => ".crtRecompose",
        NodeKind::PackPolynomialCoefficients { .. } => ".packPolynomialCoefficients",
        NodeKind::FamilyPack { .. } => ".familyPack",
        NodeKind::FamilyGetStatic { .. } => ".familyGetStatic",
        NodeKind::FamilyGetDynamic => ".familyGetDynamic",
        NodeKind::SubgraphCall(_) => ".subgraphCall",
        NodeKind::ParallelLoop(_) => ".parallelLoop",
        NodeKind::SequentialLoop(_) => ".sequentialLoop",
    };
    Ok(rule.to_owned())
}

fn lean_scope(
    stage: &str,
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    scope: &GraphScope,
    indent: usize,
) -> Result<String, EmitError> {
    let padding = " ".repeat(indent);
    let child_padding = " ".repeat(indent + 2);
    let nodes = lean_list_literal(
        scope
        .nodes()
        .iter()
        .enumerate()
        .map(|(node_id, node)| -> Result<_, EmitError> {
            let arguments = scope
                .arguments(node)
                .expect("frozen root arguments")
                .iter()
                .map(lean_wire_ref)
                .collect::<Vec<_>>()
                .join(", ");
            let output_types = node
                .output_types()
                .iter()
                .map(lean_wire_type_expr)
                .collect::<Vec<_>>()
                .join(", ");
            Ok(format!(
                "{child_padding}{{ kind := {}, arguments := [{arguments}], outputCount := {}, outputTypes := [{output_types}] }}",
                lean_node_kind(stage, graph, scope_id, mxx_ir_core::NodeId(node_id as u64), node,)?,
                node.output_types().len()
            ))
        })
        .collect::<Result<Vec<_>, _>>()?,
        "Mxx.Ir.Node",
        &padding,
    );
    let outputs = if matches!(scope_id, FrozenGraphScopeId::Root) {
        graph
            .outputs()
            .iter()
            .map(|(name, output)| {
                format!("({}, {})", lean_string(name), lean_wire_ref(&output.value))
            })
            .collect::<Vec<_>>()
    } else {
        scope
            .outputs()
            .iter()
            .enumerate()
            .map(|(index, output)| {
                format!("({}, {})", lean_string(&format!("output-{index}")), lean_wire_ref(output))
            })
            .collect::<Vec<_>>()
    }
    .join(", ");
    let input_names = if matches!(scope_id, FrozenGraphScopeId::Root) {
        scope
            .nodes()
            .iter()
            .filter_map(|node| match node.kind() {
                NodeKind::Input { name, .. } => Some(lean_string(name)),
                _ => None,
            })
            .collect::<Vec<_>>()
    } else {
        scope
            .inputs()
            .iter()
            .map(|wire| {
                let node = scope.node(wire.node).expect("scope input references an existing node");
                match node.kind() {
                    NodeKind::Input { name, .. } => lean_string(name),
                    _ => unreachable!("scope inputs always reference input nodes"),
                }
            })
            .collect::<Vec<_>>()
    }
    .join(", ");
    Ok(format!(
        "{padding}{{ nodes := {nodes}\n{padding}  outputs := [{outputs}]\n{padding}  inputNames := [{input_names}] }}"
    ))
}

fn lean_scope_name(id: &FrozenGraphScopeId) -> String {
    match id {
        FrozenGraphScopeId::Root => "__root".to_owned(),
        FrozenGraphScopeId::Subgraph { canonical_name } => {
            format!("subgraph:{canonical_name}")
        }
        FrozenGraphScopeId::ParallelBody { parent, owner } => {
            format!("parallel:{}:{}", lean_scope_name(parent), owner.0)
        }
        FrozenGraphScopeId::SequentialBody { parent, owner } => {
            format!("sequential:{}:{}", lean_scope_name(parent), owner.0)
        }
    }
}

fn lean_wire_ref(wire: &mxx_ir_core::WireRef) -> String {
    format!("{{ node := {}, port := {} }}", wire.node.0, wire.port.0)
}

fn lean_node_kind(
    stage: &str,
    graph: &Graph,
    scope_id: &FrozenGraphScopeId,
    node_id: mxx_ir_core::NodeId,
    node: &mxx_ir_core::NodeHandle,
) -> Result<String, EmitError> {
    validate_operational_inventory(node.kind())?;
    let unsupported = |kind| EmitError::UnsupportedNode {
        stage: stage.to_owned(),
        scope: lean_scope_name(scope_id),
        node: node_id.0,
        kind,
    };
    Ok(match node.kind() {
        NodeKind::Input { name, .. } => format!(".input {}", lean_string(name)),
        NodeKind::ConstantInt(value) => format!(".constantInt ({value} : Int)"),
        NodeKind::EvaluateInt(value) => format!(".evaluateInt ({})", lean_ir_int_expr(value)),
        NodeKind::ConstantReal(value) => format!(".constantReal ({})", lean_ir_real_expr(value)),
        NodeKind::ConstantBool(value) => format!(".constantBool {value}"),
        NodeKind::ConstantMatrix { matrix_type, value: ConstantMatrix::Zero } => {
            format!(".zeroMatrix {}", lean_matrix_type(matrix_type))
        }
        NodeKind::ConstantMatrix { matrix_type, value: ConstantMatrix::Identity } => {
            format!(".identityMatrix {}", lean_matrix_type(matrix_type))
        }
        NodeKind::ConstantMatrix {
            matrix_type,
            value: ConstantMatrix::Polynomial { coefficients },
        } => format!(
            ".constantMatrix {} [{}]",
            lean_matrix_type(matrix_type),
            coefficients
                .iter()
                .map(|coefficient| format!("({})", lean_ir_int_expr(coefficient)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        NodeKind::ConstantMatrix {
            matrix_type,
            value: ConstantMatrix::Gadget { base, small: false },
        } => {
            format!(".gadgetMatrix {} ({})", lean_matrix_type(matrix_type), lean_ir_int_expr(base))
        }
        NodeKind::ConstantMatrix {
            matrix_type,
            value: ConstantMatrix::Gadget { base, small: true },
        } => format!(
            ".smallGadgetMatrix {} ({})",
            lean_matrix_type(matrix_type),
            lean_ir_int_expr(base)
        ),
        NodeKind::ConstantMatrix { matrix_type, value: ConstantMatrix::UnitRow { index } } => {
            format!(
                ".unitRowMatrix {} ({})",
                lean_matrix_type(matrix_type),
                lean_ir_int_expr(index)
            )
        }
        NodeKind::ConstantMatrix { matrix_type, value: ConstantMatrix::UnitColumn { index } } => {
            format!(
                ".unitColumnMatrix {} ({})",
                lean_matrix_type(matrix_type),
                lean_ir_int_expr(index)
            )
        }
        NodeKind::ConstantMatrix {
            matrix_type,
            value: ConstantMatrix::PowerOfBase { base, exponent },
        } => format!(
            ".powerOfBaseMatrix {} ({}) ({})",
            lean_matrix_type(matrix_type),
            lean_ir_int_expr(base),
            lean_ir_int_expr(exponent)
        ),
        NodeKind::ConstantMatrix { matrix_type, value: ConstantMatrix::Rotation { exponent } } => {
            format!(
                ".rotationMatrix {} ({})",
                lean_matrix_type(matrix_type),
                lean_ir_int_expr(exponent)
            )
        }
        NodeKind::GadgetTrapdoor { matrix_type, base } => format!(
            ".gadgetTrapdoor {} ({})",
            lean_matrix_type(matrix_type),
            lean_ir_int_expr(base)
        ),
        NodeKind::IntToReal => ".intToReal".to_owned(),
        NodeKind::BoolToInt => ".boolToInt".to_owned(),
        NodeKind::IntBinary(operation) => format!(
            ".intBinary .{}",
            match operation {
                IntBinaryOp::Add => "add",
                IntBinaryOp::Subtract => "subtract",
                IntBinaryOp::Multiply => "multiply",
                IntBinaryOp::Divide => "divide",
                IntBinaryOp::Remainder => "remainder",
            }
        ),
        NodeKind::IntCompare(operation) => format!(
            ".intCompare .{}",
            match operation {
                IntCompareOp::Equal => "equal",
                IntCompareOp::Less => "less",
                IntCompareOp::LessEqual => "lessEqual",
            }
        ),
        NodeKind::RealBinary(operation) => format!(
            ".realBinary .{}",
            match operation {
                RealBinaryOp::Add => "add",
                RealBinaryOp::Subtract => "subtract",
                RealBinaryOp::Multiply => "multiply",
                RealBinaryOp::Divide => "divide",
            }
        ),
        NodeKind::RealSqrt => ".realSqrt".to_owned(),
        NodeKind::BitExtract { bit } => format!(".bitExtract ({})", lean_ir_int_expr(bit)),
        NodeKind::ExtractCoefficient { position } => {
            format!(".extractCoefficient ({})", lean_ir_int_expr(position))
        }
        NodeKind::ConstantCoefficient { position } => {
            format!(".constantCoefficient ({})", lean_ir_int_expr(position))
        }
        NodeKind::Select { .. } => ".select".to_owned(),
        NodeKind::UniformResidueSample { matrix_type } => {
            format!(".uniformResidueSample {}", lean_matrix_type(matrix_type))
        }
        NodeKind::UniformIntervalSample { matrix_type, range } => format!(
            ".uniformIntervalSample {} ({}) ({})",
            lean_matrix_type(matrix_type),
            lean_ir_int_expr(&range.minimum),
            lean_ir_int_expr(&range.maximum)
        ),
        NodeKind::GaussianSample { matrix_type, max_coefficient_bound, .. } => {
            format!(
                ".gaussianSample {} ({})",
                lean_matrix_type(matrix_type),
                lean_ir_int_expr(max_coefficient_bound)
            )
        }
        NodeKind::HashSample {
            matrix_type,
            variant,
            tag_prefix,
            tag_expressions,
            tag_decimal_expressions,
            tag_u64_le_expressions,
            base,
            digit_count,
        } => {
            let variant = match variant {
                HashVariant::Plain => "plain",
                HashVariant::Decomposed => "decomposed",
                HashVariant::SmallDecomposed => "smallDecomposed",
            };
            let bytes = tag_prefix.iter().map(u8::to_string).collect::<Vec<_>>().join(", ");
            let expressions = |values: &[IntExpr]| {
                values
                    .iter()
                    .map(|value| format!("({})", lean_ir_int_expr(value)))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            let optional = |value: &Option<IntExpr>| match value {
                Some(value) => format!("some ({})", lean_ir_int_expr(value)),
                None => "none".to_owned(),
            };
            format!(
                ".hashSample {} .{} [{}] [{}] [{}] [{}] {} {}",
                lean_matrix_type(matrix_type),
                variant,
                bytes,
                expressions(tag_expressions),
                expressions(tag_decimal_expressions),
                expressions(tag_u64_le_expressions),
                optional(base),
                optional(digit_count),
            )
        }
        NodeKind::GadgetDecompose { base, small, digit_count } => {
            let matrix_type = match node.output_types().first() {
                Some(WireType::Matrix(matrix_type) | WireType::Preimage(matrix_type)) => {
                    matrix_type
                }
                _ => return Err(unsupported("GadgetDecompose")),
            };
            format!(
                ".gadgetDecompose {} ({}) {} ({})",
                lean_matrix_type(matrix_type),
                lean_ir_int_expr(base),
                if *small { "true" } else { "false" },
                lean_ir_int_expr(digit_count)
            )
        }
        NodeKind::TrapdoorSample { matrix_type, preimage_max_coefficient_bound, .. } => format!(
            ".trapdoorSample {} ({})",
            lean_matrix_type(matrix_type),
            lean_ir_int_expr(preimage_max_coefficient_bound)
        ),
        NodeKind::TrapdoorPublic => ".trapdoorPublic".to_owned(),
        NodeKind::PreimageSample { matrix_type, max_coefficient_bound } => format!(
            ".preimageSample {} ({})",
            lean_matrix_type(matrix_type),
            lean_ir_int_expr(max_coefficient_bound)
        ),
        NodeKind::MatrixBinary(MatrixBinaryOp::Add) => ".matrixAdd".to_owned(),
        NodeKind::MatrixBinary(MatrixBinaryOp::Subtract) => ".matrixSubtract".to_owned(),
        NodeKind::MatrixBinary(MatrixBinaryOp::Multiply) => ".matrixMultiply".to_owned(),
        NodeKind::MatrixNegate => ".matrixNegate".to_owned(),
        NodeKind::MatrixScale { scalar } => {
            format!(".matrixScale ({})", lean_ir_int_expr(scalar))
        }
        NodeKind::Transpose => ".transpose".to_owned(),
        NodeKind::Slice { rows, columns } => {
            let range = |range: &Option<mxx_ir_core::node::IndexRange>| match range {
                Some(range) => format!(
                    "some (({}), ({}))",
                    lean_ir_int_expr(&range.start),
                    lean_ir_int_expr(&range.end)
                ),
                None => "none".to_owned(),
            };
            format!(".slice ({}) ({})", range(rows), range(columns))
        }
        NodeKind::Tensor => ".tensor".to_owned(),
        NodeKind::Reshape { rows, columns } => {
            format!(".reshape ({}) ({})", lean_ir_int_expr(rows), lean_ir_int_expr(columns))
        }
        NodeKind::Concat { axis } => format!(
            ".concat .{}",
            match axis {
                ConcatAxis::Rows => "rows",
                ConcatAxis::Columns => "columns",
                ConcatAxis::Diagonal => "diagonal",
            }
        ),
        NodeKind::ThresholdDecode { plaintext_modulus, length, output_bool } => {
            let scope = graph.scope(scope_id).ok_or(EmitError::ChildScope)?;
            let modulus = scope
                .arguments(node)
                .and_then(|arguments| arguments.first().copied())
                .and_then(|wire| scope.node(wire.node))
                .and_then(|source| source.output_types().first())
                .and_then(|wire_type| match wire_type {
                    WireType::Matrix(matrix) | WireType::Preimage(matrix) => Some(matrix),
                    WireType::Trapdoor { matrix, .. } => Some(matrix),
                    _ => None,
                })
                .map(|matrix| matrix.modulus.clone());
            match modulus {
                Some(modulus) => {
                    let rule =
                        if *output_bool { ".thresholdDecodeBool" } else { ".thresholdDecodeInt" };
                    format!(
                        "{rule} ({}) ({}) ({})",
                        lean_ir_int_expr(&modulus),
                        lean_ir_int_expr(plaintext_modulus),
                        lean_ir_int_expr(length)
                    )
                }
                None => return Err(unsupported("ThresholdDecode")),
            }
        }
        NodeKind::CrtRecompose { plaintext_moduli, reconstruction_coefficients } => format!(
            ".crtRecompose [{}] [{}]",
            plaintext_moduli
                .iter()
                .map(|value| format!("({})", lean_ir_int_expr(value)))
                .collect::<Vec<_>>()
                .join(", "),
            reconstruction_coefficients
                .iter()
                .map(|value| format!("({})", lean_ir_int_expr(value)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        NodeKind::PackPolynomialCoefficients { matrix_type, coefficient_bits } => format!(
            ".packPolynomialCoefficients {} ({})",
            lean_matrix_type(matrix_type),
            lean_ir_int_expr(coefficient_bits)
        ),
        NodeKind::FamilyPack { .. } => ".familyPack".to_owned(),
        NodeKind::FamilyGetStatic { index } => {
            format!(".familyGetStatic ({})", lean_ir_int_expr(index))
        }
        NodeKind::FamilyGetDynamic => ".familyGetDynamic".to_owned(),
        NodeKind::SubgraphCall(call) => {
            let child = graph.child_scope_id(scope_id, node_id).ok_or(EmitError::ChildScope)?;
            format!(
                ".subgraphCall {} [{}]",
                lean_string(&lean_scope_name(&child)),
                lean_bindings(&call.bindings)
            )
        }
        NodeKind::ParallelLoop(loop_spec) => {
            let child = graph.child_scope_id(scope_id, node_id).ok_or(EmitError::ChildScope)?;
            let modes = loop_spec
                .input_modes
                .iter()
                .map(|mode| match mode {
                    LoopInputMode::Broadcast => ".broadcast".to_owned(),
                    LoopInputMode::Zip => ".zip".to_owned(),
                    LoopInputMode::ZipOffset { offset } => format!(".zipOffset {offset}"),
                })
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                ".parallelLoop {} ({}) {} [{}] [{}]",
                lean_string(&lean_scope_name(&child)),
                lean_ir_int_expr(&loop_spec.count),
                loop_spec.index_slot,
                lean_bindings(&loop_spec.bindings),
                modes
            )
        }
        NodeKind::SequentialLoop(loop_spec) => {
            let child = graph.child_scope_id(scope_id, node_id).ok_or(EmitError::ChildScope)?;
            format!(
                ".sequentialLoop {} ({}) {} [{}] {}",
                lean_string(&lean_scope_name(&child)),
                lean_ir_int_expr(&loop_spec.count),
                loop_spec.index_slot,
                lean_bindings(&loop_spec.bindings),
                loop_spec.carried_count,
            )
        }
    })
}

fn lean_bindings(bindings: &[(String, IntExpr)]) -> String {
    bindings
        .iter()
        .map(|(name, value)| format!("({}, {})", lean_string(name), lean_ir_int_expr(value)))
        .collect::<Vec<_>>()
        .join(", ")
}

fn lean_matrix_type(matrix_type: &MatrixType) -> String {
    format!(
        "{{ modulus := {}, ringDimension := {}, rows := {}, columns := {} }}",
        lean_ir_int_expr(&matrix_type.modulus),
        lean_ir_int_expr(&matrix_type.ring_dimension),
        lean_ir_int_expr(&matrix_type.rows),
        lean_ir_int_expr(&matrix_type.columns)
    )
}

fn lean_wire_type_expr(wire_type: &WireType) -> String {
    match wire_type {
        WireType::ConstantInt => ".constantInt".to_owned(),
        WireType::ConstantReal => ".constantReal".to_owned(),
        WireType::ConstantBool => ".constantBool".to_owned(),
        WireType::Int => ".integer".to_owned(),
        WireType::Real => ".real".to_owned(),
        WireType::Bool => ".boolean".to_owned(),
        WireType::Bytes { length } => format!(".bytes ({})", lean_ir_int_expr(length)),
        WireType::TypedBlob { type_name, schema_hash } => format!(
            ".typedBlob {} [{}]",
            lean_string(type_name),
            schema_hash.iter().map(u8::to_string).collect::<Vec<_>>().join(", ")
        ),
        WireType::Matrix(matrix_type) => {
            format!(".matrix ({})", lean_matrix_type(matrix_type))
        }
        WireType::Trapdoor {
            matrix,
            sigma,
            gadget_base,
            digit_count,
            preimage_max_coefficient_bound,
        } => format!(
            ".trapdoor ({}) ({}) ({}) ({}) ({})",
            lean_matrix_type(matrix),
            lean_ir_real_expr(sigma),
            lean_ir_int_expr(gadget_base),
            lean_ir_int_expr(digit_count),
            lean_ir_int_expr(preimage_max_coefficient_bound)
        ),
        WireType::Preimage(matrix_type) => {
            format!(".preimage ({})", lean_matrix_type(matrix_type))
        }
        WireType::IndexedFamily { element, count } => format!(
            ".indexedFamily ({}) ({})",
            lean_wire_type_expr(element),
            lean_ir_int_expr(count)
        ),
    }
}

fn lean_ir_real_expr(expression: &mxx_ir_core::RealExpr) -> String {
    match expression {
        mxx_ir_core::RealExpr::Rational(value) => {
            format!(".rational (({} : Rat) / ({} : Rat))", value.numerator(), value.denominator())
        }
        mxx_ir_core::RealExpr::Var(name) => format!(".parameter {}", lean_string(name)),
        mxx_ir_core::RealExpr::FromInt(value) => {
            format!(".fromInt ({})", lean_ir_int_expr(value))
        }
        mxx_ir_core::RealExpr::Add(left, right) => {
            format!(".add ({}) ({})", lean_ir_real_expr(left), lean_ir_real_expr(right))
        }
        mxx_ir_core::RealExpr::Sub(left, right) => {
            format!(".subtract ({}) ({})", lean_ir_real_expr(left), lean_ir_real_expr(right))
        }
        mxx_ir_core::RealExpr::Mul(left, right) => {
            format!(".multiply ({}) ({})", lean_ir_real_expr(left), lean_ir_real_expr(right))
        }
        mxx_ir_core::RealExpr::Div(left, right) => {
            format!(".divide ({}) ({})", lean_ir_real_expr(left), lean_ir_real_expr(right))
        }
        mxx_ir_core::RealExpr::Sqrt(value) => {
            format!(".sqrt ({})", lean_ir_real_expr(value))
        }
    }
}

fn lean_ir_int_expr(expression: &IntExpr) -> String {
    match expression {
        IntExpr::Const(value) => format!(".constant ({value} : Int)"),
        IntExpr::Var(name) => format!(".parameter {}", lean_string(name)),
        IntExpr::LoopIndex(slot) => format!(".loopIndex {slot}"),
        IntExpr::Add(left, right) => {
            format!(".add ({}) ({})", lean_ir_int_expr(left), lean_ir_int_expr(right))
        }
        IntExpr::Sub(left, right) => {
            format!(".subtract ({}) ({})", lean_ir_int_expr(left), lean_ir_int_expr(right))
        }
        IntExpr::Mul(left, right) => {
            format!(".multiply ({}) ({})", lean_ir_int_expr(left), lean_ir_int_expr(right))
        }
        IntExpr::Div(left, right) => {
            format!(".divide ({}) ({})", lean_ir_int_expr(left), lean_ir_int_expr(right))
        }
        IntExpr::RoundDiv(left, right) => {
            format!(".roundDivide ({}) ({})", lean_ir_int_expr(left), lean_ir_int_expr(right))
        }
        IntExpr::Log2Ceil(value) => format!(".log2Ceil ({})", lean_ir_int_expr(value)),
    }
}

fn normalized_protocol(protocol: &ProtocolDecl) -> Result<Value, serde_json::Error> {
    let mut stages = Vec::new();
    for stage in &protocol.bundle.workflow.stages {
        let mut graph = serde_json::to_value(&stage.graph)?;
        strip_runtime_identity(&mut graph);
        stages.push(json!({
            "id": stage.id,
            "graph": graph,
            "semantic_anchors": stage.semantic_anchors,
            "derivation_attachments": stage.derivation_attachments,
            "bindings": stage.bindings,
        }));
    }
    let mut ideal = serde_json::to_value(&protocol.bundle.ideal.graph)?;
    strip_runtime_identity(&mut ideal);
    let mut requirements = Vec::new();
    for requirement in &protocol.bundle.requirements {
        let mut graph = serde_json::to_value(&requirement.graph)?;
        strip_runtime_identity(&mut graph);
        requirements.push(graph);
    }
    let comparator = match &protocol.bundle.comparator {
        crate::ComparatorSpec::Equality { endpoints } => {
            json!({"kind": "equality", "endpoints": endpoints})
        }
        crate::ComparatorSpec::EqualityAfterMap { program, endpoints } => {
            let mut graph = serde_json::to_value(&program.graph)?;
            strip_runtime_identity(&mut graph);
            json!({"kind": "equality_after_map", "program": graph, "endpoints": endpoints})
        }
    };
    Ok(json!({
        "params": protocol.params,
        "bundle": {
            "workflow": {
                "stages": stages,
                "entrypoint": protocol.bundle.workflow.entrypoint,
            },
            "ideal": ideal,
            "requirements": requirements,
            "comparator": comparator,
            "endpoints": protocol.bundle.endpoints,
            "endpoint_specs": protocol.bundle.endpoint_specs,
            "input_contract": protocol.bundle.input_contract,
            "input_bindings": protocol.bundle.input_bindings,
            "precondition_spec": protocol.bundle.precondition_spec,
        }
    }))
}

fn strip_runtime_identity(value: &mut Value) {
    match value {
        Value::Object(map) => {
            map.remove("production_id");
            map.remove("production");
            map.remove("execution_nonce");
            for value in map.values_mut() {
                strip_runtime_identity(value);
            }
        }
        Value::Array(values) => values.iter_mut().for_each(strip_runtime_identity),
        _ => {}
    }
}

fn lean_identifier(value: &str) -> String {
    let mut output = String::new();
    let mut upper = true;
    for character in value.chars() {
        if character.is_ascii_alphanumeric() {
            output.push(if upper { character.to_ascii_uppercase() } else { character });
            upper = false;
        } else {
            upper = true;
        }
    }
    if output.is_empty() { "Protocol".to_owned() } else { output }
}

fn lower_identifier(value: &str) -> String {
    let value = lean_identifier(value);
    let mut characters = value.chars();
    characters
        .next()
        .map(|first| first.to_ascii_lowercase().to_string() + characters.as_str())
        .unwrap_or_else(|| "protocol".to_owned())
}

fn lean_string(value: &str) -> String {
    format!("\"{}\"", value.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mxx_dsl::{DslContext, Family, Int, Ring, Sequential};
    use mxx_ir_core::{RealExpr, WireType};

    #[test]
    fn large_lean_lists_are_emitted_as_typed_chunks() {
        let items = (0..=LEAN_LIST_CHUNK_SIZE).map(|index| index.to_string()).collect();
        let emitted = lean_list_literal(items, "Nat", "  ");

        assert!(emitted.starts_with("List.flatten ["));
        assert_eq!(emitted.matches(": List Nat").count(), 2);
    }

    #[test]
    fn operational_inventory_has_every_current_variant_row() {
        let keys = [
            "Input",
            "ConstantInt",
            "EvaluateInt",
            "ConstantReal",
            "ConstantBool",
            "ConstantMatrix.Zero",
            "ConstantMatrix.Identity",
            "ConstantMatrix.UnitRow",
            "ConstantMatrix.UnitColumn",
            "ConstantMatrix.Gadget",
            "ConstantMatrix.Gadget(small)",
            "ConstantMatrix.PowerOfBase",
            "ConstantMatrix.Rotation",
            "ConstantMatrix.Polynomial",
            "GadgetTrapdoor",
            "TrapdoorPublic",
            "IntBinary.Add",
            "IntBinary.Subtract",
            "IntBinary.Multiply",
            "IntBinary.Divide",
            "IntBinary.Remainder",
            "IntCompare.Equal",
            "IntCompare.Less",
            "IntCompare.LessEqual",
            "BitExtract",
            "IntToReal",
            "BoolToInt",
            "RealBinary.Add",
            "RealBinary.Subtract",
            "RealBinary.Multiply",
            "RealBinary.Divide",
            "RealSqrt",
            "MatrixBinary.Add",
            "MatrixBinary.Subtract",
            "MatrixBinary.Multiply",
            "MatrixNegate",
            "MatrixScale",
            "Transpose",
            "Slice",
            "Tensor",
            "Concat.Rows",
            "Concat.Columns",
            "Concat.Diagonal",
            "Reshape",
            "UniformResidueSample",
            "UniformIntervalSample",
            "GaussianSample",
            "HashSample.Plain",
            "HashSample.Decomposed",
            "HashSample.SmallDecomposed",
            "TrapdoorSample",
            "PreimageSample",
            "GadgetDecompose(regular)",
            "GadgetDecompose(small)",
            "ExtractCoefficient",
            "ConstantCoefficient",
            "ThresholdDecode(bool)",
            "ThresholdDecode(int)",
            "CrtRecompose",
            "PackPolynomialCoefficients",
            "SubgraphCall",
            "ParallelLoop",
            "ParallelLoop.Broadcast",
            "ParallelLoop.Zip",
            "ParallelLoop.ZipOffset",
            "SequentialLoop",
            "FamilyPack",
            "FamilyGetStatic",
            "FamilyGetDynamic",
            "Select",
        ];
        for key in keys {
            require_operational_inventory_row(key).unwrap();
        }
    }

    #[test]
    fn closed_bundle_is_embedded_with_transport_only() {
        let emitted = emit_protocol_for(
            "toy-example",
            &crate::toy_example::protocol(),
            "MxxTest",
            crate::toy_example::PROTOCOL_SOURCE_PATHS,
        )
        .unwrap();
        assert!(
            emitted.ir.contains("def ToyExample_protocol : Mxx.Certificate.ClosedProtocolDecl")
        );
        assert!(emitted.ir.contains(".toyThresholdDecode"));
        assert!(emitted.ir.contains("decoded-endpoint"));
        assert!(emitted.ir.contains(".equality ("));
        assert!(emitted.ir.contains(".boolean"));
        assert!(!emitted.ir.contains("RuleUse"));
        assert!(!emitted.ir.contains("SparseCertificate"));
        assert!(!emitted.ir.contains("AffineForm"));
    }

    #[test]
    fn parameter_declarations_and_sampler_cutoff_are_transported() {
        let emitted = emit_protocol_for(
            "toy-example",
            &crate::toy_example::protocol(),
            "MxxTest",
            crate::toy_example::PROTOCOL_SOURCE_PATHS,
        )
        .unwrap();
        assert!(emitted.ir.contains("ToyExample_protocol : Mxx.Certificate.ClosedProtocolDecl"));
        assert!(emitted.ir.contains("{ name := \"cutoff\", kind := .dimension }"));
        assert!(emitted.ir.contains("(.parameter \"cutoff\")"));
    }

    #[test]
    fn every_rust_wire_type_has_lossless_lean_transport() {
        let matrix = Ring::new(17, 8).matrix_type((2, 3));
        let trapdoor = WireType::Trapdoor {
            matrix: matrix.clone(),
            sigma: RealExpr::Sqrt(Box::new(RealExpr::from_integer(9))),
            gadget_base: IntExpr::constant(4),
            digit_count: IntExpr::constant(5),
            preimage_max_coefficient_bound: IntExpr::constant(30),
        };
        assert!(lean_wire_type_expr(&trapdoor).starts_with(".trapdoor ("));
        assert!(lean_wire_type_expr(&trapdoor).contains(".sqrt ("));

        let blob = WireType::TypedBlob { type_name: "fixture".to_owned(), schema_hash: [7; 32] };
        assert!(lean_wire_type_expr(&blob).contains(".typedBlob \"fixture\" [7, 7"));

        let family = WireType::IndexedFamily {
            element: Box::new(WireType::Preimage(matrix)),
            count: IntExpr::constant(6),
        };
        assert!(lean_wire_type_expr(&family).starts_with(".indexedFamily (.preimage"));
    }

    #[test]
    fn lean_ir_emits_sequential_scan_structure() {
        let context = DslContext::new("sequential-emission");
        let increments = context.int_family_input("increments", 2);
        let total = Sequential::range(2)
            .scan(Int::constant(0), increments, |index, total, increments| {
                Ok(total.add(increments.get(index.as_int())))
            })
            .unwrap();
        let built = context.int_output("total", total).unwrap().build().unwrap();
        let emitted = lean_program("test", &built.graph, 0).unwrap();
        assert!(emitted.contains(".sequentialLoop"));
        assert!(emitted.contains("\"sequential:__root:"));
    }

    #[test]
    fn parallel_body_input_names_follow_binding_order() {
        let ring = Ring::new(17, 1);
        let captured = ring.input("captured", (1, 1));
        let values = Family::pack(vec![ring.zero((1, 1)), ring.identity(1)]).unwrap();
        let mapped = values
            .parallel_map({
                let captured = captured.clone();
                move |_, item| captured * item
            })
            .unwrap();
        let graph = DslContext::new("parallel-input-order")
            .family_output("result", mapped)
            .unwrap()
            .build()
            .unwrap()
            .graph;

        let emitted = lean_program("test", &graph, 2).unwrap();
        let capture_node = emitted.find(".input \"__capture_0\"").unwrap();
        let item_node = emitted.find(".input \"item\"").unwrap();
        assert!(capture_node < item_node);
        assert!(emitted.contains("inputNames := [\"item\", \"__capture_0\"]"));
    }
}
